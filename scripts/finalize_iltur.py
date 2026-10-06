from pathlib import Path
import hashlib
import re
import unicodedata

import pandas as pd


ROOT = Path.home() / "TaxFacto"

SRC = ROOT / "data/standardized/iltur_rr.parquet"

FINAL = ROOT / "data/manifests_final"
NATIVE = ROOT / "data/tasks/native"
COMMON = ROOT / "data/tasks/common7"
REP = ROOT / "reports/dataset_audit"

for p in [FINAL, NATIVE, COMMON, REP]:
    p.mkdir(parents=True, exist_ok=True)


def norm_text(x):
    x = unicodedata.normalize("NFKC", str(x))
    x = x.lower()
    x = re.sub(r"\s+", " ", x).strip()
    x = re.sub(r"[^\w\s]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


COMMON7 = [
    "ARG",
    "FAC",
    "PRE",
    "RATIO",
    "RLC",
    "RPC",
    "STA",
]

MAP7 = {
    "Fact": "FAC",

    "ArgumentPetitioner": "ARG",
    "ArgumentRespondent": "ARG",

    "Statute": "STA",

    "PrecedentReliedUpon": "PRE",
    "PrecedentNotReliedUpon": "PRE",
    "PrecedentOverruled": "PRE",

    "RulingByLowerCourt": "RLC",
    "RatioOfTheDecision": "RATIO",
    "RulingByPresentCourt": "RPC",
}

SPLIT_PRIORITY = {
    "train": 0,
    "dev": 1,
    "test": 2,
}


df = pd.read_parquet(SRC)

df["norm_text"] = df["text"].map(norm_text)

print("=" * 90)
print("RAW IL-TUR")
print("=" * 90)

print(
    df.groupby(
        ["dataset", "split_canonical"]
    ).agg(
        rows=("text", "size"),
        docs=("doc_id", "nunique"),
        labels=("label", "nunique"),
    ).to_string()
)


def split_aware_dedup(d):
    """
    Preserve evaluation examples over training examples:

        test > dev > train

    Within the same split, retain first occurrence.
    """

    d = d.copy()

    d["_priority"] = (
        d["split_canonical"]
        .map(SPLIT_PRIORITY)
    )

    d["_original_order"] = range(len(d))

    # Highest-priority split comes first.
    d = d.sort_values(
        ["_priority", "_original_order"],
        ascending=[False, True],
    )

    before = len(d)

    d = d.drop_duplicates(
        subset=["norm_text"],
        keep="first",
    )

    removed = before - len(d)

    d = d.sort_values(
        "_original_order"
    ).drop(
        columns=[
            "_priority",
            "_original_order",
        ]
    )

    return d, removed


native_outputs = {}
common_outputs = []

for dataset in [
    "iltur_cl",
    "iltur_it",
]:

    print()
    print("=" * 90)
    print(dataset.upper())
    print("=" * 90)

    d = df[
        df["dataset"] == dataset
    ].copy()

    clean, removed = split_aware_dedup(d)

    print("before :", len(d))
    print("after  :", len(clean))
    print("removed:", removed)

    # --------------------------------------------------------
    # Verify split overlap after cleaning
    # --------------------------------------------------------

    sets = {
        s: set(
            clean.loc[
                clean["split_canonical"] == s,
                "norm_text",
            ]
        )
        for s in [
            "train",
            "dev",
            "test",
        ]
    }

    print()
    print("POST-CLEAN SPLIT OVERLAP")

    for a, b in [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]:
        n = len(sets[a] & sets[b])

        print(
            f"{a:5s}-{b:5s}: {n}"
        )

        if n != 0:
            raise RuntimeError(
                f"{dataset}: remaining "
                f"{a}-{b} overlap = {n}"
            )

    # --------------------------------------------------------
    # Document leakage check
    # --------------------------------------------------------

    docsets = {
        s: set(
            clean.loc[
                clean["split_canonical"] == s,
                "doc_id",
            ].astype(str)
        )
        for s in [
            "train",
            "dev",
            "test",
        ]
    }

    print()
    print("DOCUMENT OVERLAP")

    for a, b in [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]:
        n = len(
            docsets[a] & docsets[b]
        )

        print(
            f"{a:5s}-{b:5s}: {n}"
        )

        if n != 0:
            raise RuntimeError(
                f"{dataset}: document leakage"
            )

    # --------------------------------------------------------
    # Save native
    # --------------------------------------------------------

    native = clean.drop(
        columns=["norm_text"]
    ).reset_index(drop=True)

    native_fn = FINAL / f"{dataset}.parquet"

    native.to_parquet(
        native_fn,
        index=False,
    )

    native.to_parquet(
        NATIVE / f"{dataset}.parquet",
        index=False,
    )

    native_outputs[dataset] = native

    # --------------------------------------------------------
    # Common-7 projection
    # --------------------------------------------------------

    c = clean[
        clean["label"].isin(MAP7)
    ].copy()

    c["label_native"] = c["label"]
    c["label"] = c["label"].map(MAP7)

    if not set(c["label"]).issubset(
        COMMON7
    ):
        raise RuntimeError(
            "Unexpected Common-7 label"
        )

    c = c.drop(
        columns=["norm_text"]
    ).reset_index(drop=True)

    c.to_parquet(
        COMMON / f"{dataset}.parquet",
        index=False,
    )

    common_outputs.append(c)

    print()
    print("NATIVE")
    print(
        native.groupby(
            "split_canonical"
        ).agg(
            rows=("text", "size"),
            docs=("doc_id", "nunique"),
            labels=("label", "nunique"),
        ).to_string()
    )

    print()
    print("COMMON-7")
    print(
        c.groupby(
            "split_canonical"
        ).agg(
            rows=("text", "size"),
            docs=("doc_id", "nunique"),
            labels=("label", "nunique"),
        ).to_string()
    )

    print()
    print(
        pd.crosstab(
            c["label"],
            c["split_canonical"],
        ).to_string()
    )


# ============================================================
# Combined IL-TUR Common-7
# ============================================================

common_all = pd.concat(
    common_outputs,
    ignore_index=True,
)

common_all.to_parquet(
    COMMON / "iltur_all.parquet",
    index=False,
)


# ============================================================
# Summary report
# ============================================================

frames = []

for dataset, native in native_outputs.items():

    x = (
        native.groupby(
            "split_canonical"
        )
        .agg(
            rows=("text", "size"),
            documents=("doc_id", "nunique"),
            labels=("label", "nunique"),
        )
        .reset_index()
    )

    x.insert(
        0,
        "dataset",
        dataset,
    )

    x.insert(
        1,
        "task",
        "native",
    )

    frames.append(x)

for dataset in [
    "iltur_cl",
    "iltur_it",
]:

    c = pd.read_parquet(
        COMMON / f"{dataset}.parquet"
    )

    x = (
        c.groupby(
            "split_canonical"
        )
        .agg(
            rows=("text", "size"),
            documents=("doc_id", "nunique"),
            labels=("label", "nunique"),
        )
        .reset_index()
    )

    x.insert(
        0,
        "dataset",
        dataset,
    )

    x.insert(
        1,
        "task",
        "common7",
    )

    frames.append(x)


summary = pd.concat(
    frames,
    ignore_index=True,
)

summary.to_csv(
    REP / "iltur_final_summary.csv",
    index=False,
)

print()
print("=" * 90)
print("FINAL SUMMARY")
print("=" * 90)
print(summary.to_string(index=False))


# ============================================================
# SHA256
# ============================================================

print()
print("=" * 90)
print("SHA256")
print("=" * 90)

files = [
    FINAL / "iltur_cl.parquet",
    FINAL / "iltur_it.parquet",
    COMMON / "iltur_cl.parquet",
    COMMON / "iltur_it.parquet",
    COMMON / "iltur_all.parquet",
]

for fn in files:
    sha = hashlib.sha256(
        fn.read_bytes()
    ).hexdigest()

    print(
        f"{fn.relative_to(ROOT)!s:45s} "
        f"{sha}"
    )

print()
print("IL-TUR FINALIZED")
