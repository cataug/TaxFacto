from pathlib import Path
import pandas as pd
import hashlib
import re
import unicodedata

ROOT = Path.home() / "TaxFacto"
SRC = ROOT / "data/standardized/rhetorical_roles_all.parquet"
OUT = ROOT / "data/benchmark"
REP = ROOT / "reports/dataset_audit"

OUT.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(SRC)

print("Loaded:", len(df))
print(df.groupby("dataset").size())


def normalize(text):
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


df["clean_text"] = df["text"].map(normalize)

# ------------------------------------------------------------
# 1. Remove exact duplicates WITHIN each dataset
# ------------------------------------------------------------

before = len(df)

dedup = (
    df.sort_values(
        ["dataset", "doc_id", "sentence_id"]
    )
    .drop_duplicates(
        subset=["dataset", "clean_text"],
        keep="first"
    )
    .copy()
)

print()
print("After within-dataset dedup:")
print("before:", before)
print("after: ", len(dedup))
print()

print(dedup.groupby("dataset").size())


# ------------------------------------------------------------
# 2. Build cross-dataset contamination table
# ------------------------------------------------------------

datasets = sorted(dedup["dataset"].unique())

sets = {
    d: set(
        dedup.loc[
            dedup.dataset == d,
            "clean_text"
        ]
    )
    for d in datasets
}

rows = []

for i, a in enumerate(datasets):
    for b in datasets[i + 1:]:
        common = sets[a] & sets[b]

        rows.append({
            "dataset_a": a,
            "dataset_b": b,
            "n_a": len(sets[a]),
            "n_b": len(sets[b]),
            "overlap": len(common),
            "pct_a": 100 * len(common) / max(1, len(sets[a])),
            "pct_b": 100 * len(common) / max(1, len(sets[b])),
        })

overlap = pd.DataFrame(rows)
overlap.to_csv(
    REP / "benchmark_cross_dataset_overlap.csv",
    index=False
)

print()
print("Cross-dataset overlap:")
print(overlap.to_string(index=False))


# ------------------------------------------------------------
# 3. Create decontaminated dataset versions
#
# For each corpus, remove sentences appearing in ANY other corpus.
# This is deliberately conservative.
# ------------------------------------------------------------

for d in datasets:

    other_text = set()

    for other in datasets:
        if other != d:
            other_text |= sets[other]

    part = dedup[dedup.dataset == d].copy()

    part["cross_dataset_duplicate"] = (
        part["clean_text"].isin(other_text)
    )

    clean = part[
        ~part["cross_dataset_duplicate"]
    ].copy()

    print()
    print(
        f"{d:15s}: "
        f"dedup={len(part):,} "
        f"clean={len(clean):,} "
        f"removed={len(part)-len(clean):,}"
    )

    part.to_parquet(
        OUT / f"{d}_dedup.parquet",
        index=False
    )

    clean.to_parquet(
        OUT / f"{d}_clean.parquet",
        index=False
    )


# ------------------------------------------------------------
# 4. Label statistics
# ------------------------------------------------------------

stats = (
    dedup.groupby(["dataset", "label"])
    .size()
    .reset_index(name="count")
)

stats.to_csv(
    REP / "benchmark_label_counts_dedup.csv",
    index=False
)


# ------------------------------------------------------------
# 5. Corpus summary
# ------------------------------------------------------------

summary = []

for d in datasets:

    original = df[df.dataset == d]
    dd = dedup[dedup.dataset == d]

    clean_path = OUT / f"{d}_clean.parquet"
    clean = pd.read_parquet(clean_path)

    summary.append({
        "dataset": d,
        "raw_rows": len(original),
        "dedup_rows": len(dd),
        "cross_clean_rows": len(clean),
        "documents": (
            original["doc_id"]
            .astype(str)
            .replace("", pd.NA)
            .nunique()
        ),
        "labels": original["label"].nunique(),
    })

summary = pd.DataFrame(summary)

summary.to_csv(
    REP / "benchmark_summary.csv",
    index=False
)

print()
print("=" * 72)
print("FINAL CLEAN BENCHMARK")
print("=" * 72)
print(summary.to_string(index=False))

print()
print("Saved to:", OUT)
