from pathlib import Path
import hashlib
import pandas as pd
import re
import unicodedata

ROOT = Path.home() / "TaxFacto"
SRC = ROOT / "data/standardized/rhetorical_roles_all.parquet"
OUT = ROOT / "data/manifests_final"
REP = ROOT / "reports/dataset_audit"

OUT.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

SEED = 42

DATASETS = [
    "legalseg",
    "legaleval",
    "marro_india",
    "marro_uk",
]


def norm(x):
    x = "" if x is None else str(x)
    x = unicodedata.normalize("NFKC", x).casefold()
    x = re.sub(r"[^\w]+", " ", x, flags=re.UNICODE)
    return re.sub(r"\s+", " ", x).strip()


def stable_fraction(s):
    h = hashlib.blake2b(
        str(s).encode("utf-8"),
        digest_size=8
    ).digest()
    return int.from_bytes(h, "big") / (2**64 - 1)


def canon_split(x):
    x = str(x).strip().lower()

    aliases = {
        "validation": "dev",
        "valid": "dev",
        "val": "dev",
        "development": "dev",
    }

    return aliases.get(x, x)


df = pd.read_parquet(SRC)

df = df[df["dataset"].isin(DATASETS)].copy()

for c in [
    "dataset", "split", "doc_id",
    "sentence_id", "text", "label"
]:
    df[c] = df[c].fillna("").astype(str)

df["clean_text"] = df["text"].map(norm)
df["split_original"] = df["split"].map(canon_split)

summary = []
leakage_report = []
parts = []

for dataset in DATASETS:

    d = df[df["dataset"] == dataset].copy()

    print()
    print("=" * 80)
    print(dataset)
    print("=" * 80)

    print("raw rows:", f"{len(d):,}")
    print("raw docs:", d["doc_id"].nunique())
    print("original split counts:")
    print(d["split_original"].value_counts())

    # ========================================================
    # DEFINE CANONICAL SPLITS
    # ========================================================

    if dataset in {"legalseg", "legaleval"}:

        required = {"train", "dev", "test"}
        present = set(d["split_original"].unique())

        if not required.issubset(present):
            raise RuntimeError(
                f"{dataset}: expected official train/dev/test, "
                f"got {sorted(present)}"
            )

        d["split_canonical"] = d["split_original"]

        print("Using OFFICIAL train/dev/test.")

    else:
        # MARRO: preserve provided test;
        # split official training docs into train/dev.
        d["split_canonical"] = "train"

        is_test = d["split_original"] == "test"
        d.loc[is_test, "split_canonical"] = "test"

        train_docs = sorted(
            d.loc[~is_test, "doc_id"].unique()
        )

        dev_docs = set()

        for doc in train_docs:
            x = stable_fraction(
                f"{SEED}:{dataset}:dev:{doc}"
            )

            # ~10% of original training documents
            if x >= 0.90:
                dev_docs.add(doc)

        d.loc[
            (~is_test) & d["doc_id"].isin(dev_docs),
            "split_canonical"
        ] = "dev"

        print(
            "Preserved official test; "
            "created deterministic dev from train."
        )

    # ========================================================
    # CHECK DOCUMENT SPLIT LEAKAGE
    # ========================================================

    doc_splits = (
        d.groupby("doc_id")["split_canonical"]
         .nunique()
    )

    leaking_docs = doc_splits[doc_splits > 1]

    print("documents crossing splits:", len(leaking_docs))

    if len(leaking_docs):
        raise RuntimeError(
            f"{dataset}: {len(leaking_docs)} documents "
            "occur in multiple canonical splits."
        )

    # ========================================================
    # SPLIT-AWARE DEDUPLICATION
    #
    # Preserve test first, then dev, then train.
    # ========================================================

    priorities = {
        "test": 0,
        "dev": 1,
        "train": 2,
    }

    d["_priority"] = d["split_canonical"].map(priorities)

    before = len(d)

    d = (
        d.sort_values(
            ["_priority", "doc_id", "sentence_id"]
        )
        .drop_duplicates(
            subset=["clean_text"],
            keep="first"
        )
        .copy()
    )

    d.drop(columns="_priority", inplace=True)

    print(
        "after split-aware dedup:",
        f"{len(d):,}",
        "removed:",
        f"{before - len(d):,}"
    )

    # ========================================================
    # VERIFY ZERO NORMALIZED SENTENCE LEAKAGE
    # ========================================================

    sets = {}

    for split in ["train", "dev", "test"]:
        sets[split] = set(
            d.loc[
                d["split_canonical"] == split,
                "clean_text"
            ]
        )

    pairs = [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]

    for a, b in pairs:
        n = len(sets[a] & sets[b])

        leakage_report.append({
            "dataset": dataset,
            "split_a": a,
            "split_b": b,
            "normalized_overlap": n,
        })

        if n:
            raise RuntimeError(
                f"{dataset}: leakage remains "
                f"{a}<->{b}: {n}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    keep = [
        "dataset",
        "split_original",
        "split_canonical",
        "doc_id",
        "sentence_id",
        "text",
        "label",
        "clean_text",
    ]

    d = d[keep].copy()

    d.to_parquet(
        OUT / f"{dataset}.parquet",
        index=False,
        compression="zstd",
    )

    parts.append(d)

    print()
    print(
        d.groupby("split_canonical").agg(
            rows=("text", "size"),
            documents=("doc_id", "nunique"),
            labels=("label", "nunique"),
        )
    )

    for split, g in d.groupby("split_canonical"):
        summary.append({
            "dataset": dataset,
            "split": split,
            "rows": len(g),
            "documents": g["doc_id"].nunique(),
            "labels": g["label"].nunique(),
        })


# ============================================================
# COMBINED
# ============================================================

all_df = pd.concat(parts, ignore_index=True)

all_df.to_parquet(
    OUT / "rr_benchmark_all.parquet",
    index=False,
    compression="zstd",
)

summary = pd.DataFrame(summary)

summary.to_csv(
    REP / "final_split_summary.csv",
    index=False
)

pd.DataFrame(leakage_report).to_csv(
    REP / "final_split_leakage.csv",
    index=False
)

labels = (
    all_df.groupby(
        ["dataset", "split_canonical", "label"]
    )
    .size()
    .reset_index(name="count")
)

labels.to_csv(
    REP / "final_label_counts.csv",
    index=False
)

print()
print("=" * 80)
print("FINAL MANIFESTS READY")
print("=" * 80)
print(summary.to_string(index=False))

print()
print("TOTAL:", f"{len(all_df):,}")

print()
print("Leakage:")
print(pd.DataFrame(leakage_report).to_string(index=False))
