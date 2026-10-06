from pathlib import Path
from collections import Counter
import hashlib
import re
import unicodedata

import pandas as pd


ROOT = Path.home() / "TaxFacto"

RAW = ROOT / "data/external/IL-TUR/rr"
OUT = ROOT / "data/standardized"
REP = ROOT / "reports/dataset_audit"

OUT.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)


LABELS = {
    0: "Fact",
    1: "Issue",
    2: "ArgumentPetitioner",
    3: "ArgumentRespondent",
    4: "Statute",
    5: "Dissent",
    6: "PrecedentReliedUpon",
    7: "PrecedentNotReliedUpon",
    8: "PrecedentOverruled",
    9: "RulingByLowerCourt",
    10: "RatioOfTheDecision",
    11: "RulingByPresentCourt",
    12: "None",
}


FILES = {
    ("CL", "train"): "CL_train-00000-of-00001.parquet",
    ("CL", "dev"):   "CL_dev-00000-of-00001.parquet",
    ("CL", "test"):  "CL_test-00000-of-00001.parquet",

    ("IT", "train"): "IT_train-00000-of-00001.parquet",
    ("IT", "dev"):   "IT_dev-00000-of-00001.parquet",
    ("IT", "test"):  "IT_test-00000-of-00001.parquet",
}


def norm_text(x):
    x = unicodedata.normalize("NFKC", str(x))
    x = x.lower()
    x = re.sub(r"\s+", " ", x).strip()
    x = re.sub(r"[^\w\s]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


# ============================================================
# 1. Flatten documents -> sentence rows
# ============================================================

rows = []

print("=" * 90)
print("IL-TUR RR FLATTEN")
print("=" * 90)

for (domain, split), name in FILES.items():
    fn = RAW / name

    if not fn.exists():
        raise FileNotFoundError(fn)

    df = pd.read_parquet(fn)

    print()
    print(domain, split, "documents:", len(df))

    for _, r in df.iterrows():

        texts = list(r["text"])
        labels = list(r["labels"])

        if len(texts) != len(labels):
            raise RuntimeError(
                f"{r['id']}: text={len(texts)} labels={len(labels)}"
            )

        for sid, (text, lid) in enumerate(
            zip(texts, labels)
        ):
            lid = int(lid)

            if lid not in LABELS:
                raise RuntimeError(
                    f"Unknown label {lid}"
                )

            rows.append({
                "dataset": f"iltur_{domain.lower()}",
                "domain": domain,
                "split_canonical": split,
                "doc_id": str(r["id"]),
                "sentence_id": sid,
                "text": str(text),
                "label_id": lid,
                "label": LABELS[lid],
            })


all_df = pd.DataFrame(rows)

all_df["norm_text"] = all_df["text"].map(
    norm_text
)

print()
print("TOTAL SENTENCES:", len(all_df))
print("TOTAL DOCUMENTS:", all_df["doc_id"].nunique())


# ============================================================
# 2. Basic statistics
# ============================================================

summary = (
    all_df
    .groupby(
        ["dataset", "split_canonical"],
        observed=True,
    )
    .agg(
        rows=("text", "size"),
        documents=("doc_id", "nunique"),
        labels=("label", "nunique"),
    )
    .reset_index()
)

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(summary.to_string(index=False))


counts = (
    all_df
    .groupby(
        ["dataset", "split_canonical", "label"],
        observed=True,
    )
    .size()
    .reset_index(name="rows")
)

print()
print("=" * 90)
print("LABEL COUNTS")
print("=" * 90)

for dataset in sorted(
    all_df["dataset"].unique()
):
    print()
    print(dataset)

    x = counts[
        counts["dataset"] == dataset
    ]

    pivot = x.pivot(
        index="label",
        columns="split_canonical",
        values="rows",
    ).fillna(0).astype(int)

    print(pivot.to_string())


# ============================================================
# 3. Exact/normalized duplicates inside IL-TUR
# ============================================================

print()
print("=" * 90)
print("WITHIN-DOMAIN SPLIT OVERLAP")
print("=" * 90)

overlap_rows = []

for dataset in [
    "iltur_cl",
    "iltur_it",
]:
    d = all_df[
        all_df["dataset"] == dataset
    ]

    sets = {
        split: set(
            d.loc[
                d["split_canonical"] == split,
                "norm_text",
            ]
        )
        for split in [
            "train",
            "dev",
            "test",
        ]
    }

    for a, b in [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]:
        ov = sets[a] & sets[b]

        print(
            f"{dataset:12s} "
            f"{a:5s}-{b:5s}: "
            f"{len(ov):6d}"
        )

        overlap_rows.append({
            "dataset": dataset,
            "split_a": a,
            "split_b": b,
            "overlap": len(ov),
        })


# ============================================================
# 4. CL <-> IT overlap
# ============================================================

cl = set(
    all_df.loc[
        all_df["dataset"] == "iltur_cl",
        "norm_text",
    ]
)

it = set(
    all_df.loc[
        all_df["dataset"] == "iltur_it",
        "norm_text",
    ]
)

both = cl & it

print()
print("=" * 90)
print("CL <-> IT")
print("=" * 90)
print("CL unique:", len(cl))
print("IT unique:", len(it))
print("overlap  :", len(both))


# ============================================================
# 5. Compare against existing TaxFacto corpora
# ============================================================

print()
print("=" * 90)
print("CROSS-DATASET OVERLAP")
print("=" * 90)

existing = {
    "legaleval":
        ROOT / "data/manifests_final/legaleval.parquet",

    "legalseg":
        ROOT / "data/manifests_final/legalseg.parquet",

    "marro_india":
        ROOT / "data/manifests_final/marro_india.parquet",

    "marro_uk":
        ROOT / "data/manifests_final/marro_uk.parquet",
}

cross_rows = []

for il_name in [
    "iltur_cl",
    "iltur_it",
]:
    a = set(
        all_df.loc[
            all_df["dataset"] == il_name,
            "norm_text",
        ]
    )

    for other_name, fn in existing.items():

        if not fn.exists():
            print(
                "MISSING:",
                other_name,
                fn,
            )
            continue

        other = pd.read_parquet(
            fn,
            columns=["text"],
        )

        b = set(
            other["text"]
            .astype(str)
            .map(norm_text)
        )

        ov = a & b

        pct_a = (
            100.0 * len(ov) / len(a)
            if a else 0.0
        )

        pct_b = (
            100.0 * len(ov) / len(b)
            if b else 0.0
        )

        print(
            f"{il_name:12s} <-> "
            f"{other_name:14s}: "
            f"{len(ov):6d} "
            f"| IL-TUR {pct_a:7.2f}% "
            f"| other {pct_b:7.2f}%"
        )

        cross_rows.append({
            "iltur_dataset": il_name,
            "other_dataset": other_name,
            "overlap": len(ov),
            "pct_iltur": pct_a,
            "pct_other": pct_b,
        })


# ============================================================
# 6. Save standardized version
# ============================================================

save_cols = [
    "dataset",
    "domain",
    "split_canonical",
    "doc_id",
    "sentence_id",
    "text",
    "label_id",
    "label",
]

final = all_df[save_cols].copy()

final.to_parquet(
    OUT / "iltur_rr.parquet",
    index=False,
)

final[
    final["dataset"] == "iltur_cl"
].to_parquet(
    OUT / "iltur_cl.parquet",
    index=False,
)

final[
    final["dataset"] == "iltur_it"
].to_parquet(
    OUT / "iltur_it.parquet",
    index=False,
)

summary.to_csv(
    REP / "iltur_summary.csv",
    index=False,
)

counts.to_csv(
    REP / "iltur_label_counts.csv",
    index=False,
)

pd.DataFrame(
    overlap_rows
).to_csv(
    REP / "iltur_split_overlap.csv",
    index=False,
)

pd.DataFrame(
    cross_rows
).to_csv(
    REP / "iltur_cross_dataset_overlap.csv",
    index=False,
)


# ============================================================
# 7. SHA256
# ============================================================

print()
print("=" * 90)
print("FILES / SHA256")
print("=" * 90)

for name in [
    "iltur_rr.parquet",
    "iltur_cl.parquet",
    "iltur_it.parquet",
]:
    fn = OUT / name

    sha = hashlib.sha256(
        fn.read_bytes()
    ).hexdigest()

    print(
        f"{name:24s} "
        f"{fn.stat().st_size:12d} "
        f"{sha}"
    )

print()
print("DONE")
