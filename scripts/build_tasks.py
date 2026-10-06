from pathlib import Path
import shutil
import pandas as pd

ROOT = Path.home() / "TaxFacto"

SRC = ROOT / "data/manifests_final"
OUT = ROOT / "data/tasks"
REP = ROOT / "reports/dataset_audit"

NATIVE = OUT / "native"
COMMON = OUT / "common7"

NATIVE.mkdir(parents=True, exist_ok=True)
COMMON.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Native tasks: preserve original taxonomies exactly
# ============================================================

for name in [
    "legalseg",
    "legaleval",
    "marro_india",
    "marro_uk",
]:
    src = SRC / f"{name}.parquet"
    dst = NATIVE / f"{name}.parquet"
    shutil.copy2(src, dst)

print("Native tasks copied.")


# ============================================================
# 2. Common 7-role taxonomy
# ============================================================

MAP_LEGALEVAL = {
    "FAC": "FAC",

    "ARG_PETITIONER": "ARG",
    "ARG_RESPONDENT": "ARG",

    "PRE_RELIED": "PRE",
    "PRE_NOT_RELIED": "PRE",

    "RATIO": "RATIO",
    "RLC": "RLC",
    "RPC": "RPC",
    "STA": "STA",
}

COMMON_LABELS = [
    "ARG",
    "FAC",
    "PRE",
    "RATIO",
    "RLC",
    "RPC",
    "STA",
]

summary = []


# ------------------------------------------------------------
# LegalEval projection
# ------------------------------------------------------------

df = pd.read_parquet(SRC / "legaleval.parquet")

before = len(df)

df = df[df["label"].isin(MAP_LEGALEVAL)].copy()

df["label_native"] = df["label"]
df["label"] = df["label"].map(MAP_LEGALEVAL)

assert not df["label"].isna().any()
assert set(df["label"].unique()) == set(COMMON_LABELS)

df.to_parquet(
    COMMON / "legaleval.parquet",
    index=False,
    compression="zstd",
)

print()
print("LegalEval -> Common7")
print("before:", f"{before:,}")
print("after :", f"{len(df):,}")
print(
    df.groupby(["split_canonical", "label"])
      .size()
      .unstack(fill_value=0)
      .to_string()
)

for split, g in df.groupby("split_canonical"):
    summary.append({
        "dataset": "legaleval",
        "split": split,
        "rows": len(g),
        "documents": g["doc_id"].nunique(),
        "labels": g["label"].nunique(),
    })


# ------------------------------------------------------------
# MARRO already uses exactly Common-7
# ------------------------------------------------------------

for name in ["marro_india", "marro_uk"]:

    df = pd.read_parquet(SRC / f"{name}.parquet").copy()

    observed = set(df["label"].unique())

    if observed != set(COMMON_LABELS):
        raise RuntimeError(
            f"{name}: unexpected labels: {sorted(observed)}"
        )

    df["label_native"] = df["label"]

    df.to_parquet(
        COMMON / f"{name}.parquet",
        index=False,
        compression="zstd",
    )

    print()
    print(name, "-> Common7")
    print(
        df.groupby(["split_canonical", "label"])
          .size()
          .unstack(fill_value=0)
          .to_string()
    )

    for split, g in df.groupby("split_canonical"):
        summary.append({
            "dataset": name,
            "split": split,
            "rows": len(g),
            "documents": g["doc_id"].nunique(),
            "labels": g["label"].nunique(),
        })


# ============================================================
# Combined Common7
# ============================================================

parts = []

for name in [
    "legaleval",
    "marro_india",
    "marro_uk",
]:
    x = pd.read_parquet(COMMON / f"{name}.parquet")
    parts.append(x)

combined = pd.concat(parts, ignore_index=True)

combined.to_parquet(
    COMMON / "common7_all.parquet",
    index=False,
    compression="zstd",
)

summary = pd.DataFrame(summary)

summary.to_csv(
    REP / "common7_summary.csv",
    index=False,
)

counts = (
    combined.groupby(
        ["dataset", "split_canonical", "label"]
    )
    .size()
    .reset_index(name="count")
)

counts.to_csv(
    REP / "common7_label_counts.csv",
    index=False,
)

print()
print("=" * 78)
print("COMMON-7 READY")
print("=" * 78)
print(summary.to_string(index=False))

print()
print("Combined rows:", f"{len(combined):,}")
