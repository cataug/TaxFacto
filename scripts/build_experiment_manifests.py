from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

ROOT = Path.home() / "TaxFacto"
BENCH = ROOT / "data/benchmark"
OUT = ROOT / "data/manifests"
REP = ROOT / "reports/dataset_audit"

OUT.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "legalseg",
    "legaleval",
    "marro_india",
    "marro_uk",
]

SEED = 42


def stable_hash(text):
    return int(
        hashlib.blake2b(
            str(text).encode("utf-8"),
            digest_size=8
        ).hexdigest(),
        16
    )


def split_docs_hash(df, train=0.80, dev=0.10):
    """
    Deterministic document-level split.
    Same document can never occur in multiple splits.
    """
    docs = sorted(df["doc_id"].astype(str).unique())

    rows = []

    for doc in docs:
        x = stable_hash(f"{SEED}:{doc}") / (2**64 - 1)

        if x < train:
            split = "train"
        elif x < train + dev:
            split = "dev"
        else:
            split = "test"

        rows.append((doc, split))

    return dict(rows)


def normalize_split(x):
    x = str(x).strip().lower()

    aliases = {
        "validation": "dev",
        "valid": "dev",
        "val": "dev",
        "development": "dev",
    }

    return aliases.get(x, x)


summary = []
all_parts = []

for name in DATASETS:
    fn = BENCH / f"{name}_dedup.parquet"

    if not fn.exists():
        print("MISSING:", fn)
        continue

    df = pd.read_parquet(fn).copy()

    df["dataset"] = name
    df["doc_id"] = df["doc_id"].astype(str)
    df["split_original"] = df["split"].map(normalize_split)

    print()
    print("=" * 76)
    print(name)
    print("=" * 76)

    print("rows:", f"{len(df):,}")
    print("docs:", df["doc_id"].nunique())
    print("original splits:")
    print(df["split_original"].value_counts(dropna=False))

    # --------------------------------------------------------
    # MARRO: preserve provided train/test.
    # Split original train documents deterministically
    # into train/dev (90/10 within train).
    # --------------------------------------------------------

    if name.startswith("marro_"):
        test_mask = df["split_original"] == "test"
        train_pool = df.loc[~test_mask].copy()

        train_docs = sorted(
            train_pool["doc_id"].unique()
        )

        dev_docs = set()

        for doc in train_docs:
            x = stable_hash(f"{SEED}:marro-dev:{doc}") / (2**64 - 1)
            if x >= 0.90:
                dev_docs.add(doc)

        df["split_canonical"] = "train"

        df.loc[
            df["doc_id"].isin(dev_docs),
            "split_canonical"
        ] = "dev"

        df.loc[
            test_mask,
            "split_canonical"
        ] = "test"

    # --------------------------------------------------------
    # LegalEval: preserve official train/dev/test if all
    # three are actually present.
    # --------------------------------------------------------

    elif name == "legaleval":
        present = set(df["split_original"].unique())

        if {"train", "dev", "test"}.issubset(present):
            df["split_canonical"] = df["split_original"]
            print("Preserving official LegalEval splits.")

        else:
            print(
                "Official train/dev/test not all visible; "
                "creating document-level 80/10/10."
            )

            mapping = split_docs_hash(df)
            df["split_canonical"] = df["doc_id"].map(mapping)

    # --------------------------------------------------------
    # LegalSeg: deterministic document-level 80/10/10
    # --------------------------------------------------------

    else:
        mapping = split_docs_hash(df)
        df["split_canonical"] = df["doc_id"].map(mapping)

    # --------------------------------------------------------
    # Assert absolutely no document leakage.
    # --------------------------------------------------------

    doc_split_counts = (
        df.groupby("doc_id")["split_canonical"]
          .nunique()
    )

    assert doc_split_counts.max() == 1, (
        f"{name}: document leakage detected"
    )

    # --------------------------------------------------------
    # Sentence leakage across splits.
    #
    # Preserve test, then dev; remove duplicates from TRAIN.
    # This is much better than globally deleting from all sets.
    # --------------------------------------------------------

    test_text = set(
        df.loc[
            df["split_canonical"] == "test",
            "clean_text"
        ]
    )

    dev_text = set(
        df.loc[
            df["split_canonical"] == "dev",
            "clean_text"
        ]
    )

    train_mask = df["split_canonical"] == "train"

    leak_to_test = (
        train_mask
        & df["clean_text"].isin(test_text)
    )

    leak_to_dev = (
        train_mask
        & df["clean_text"].isin(dev_text)
    )

    remove_train = leak_to_test | leak_to_dev

    removed = int(remove_train.sum())

    if removed:
        print(
            "Removing train sentences duplicated in "
            f"dev/test: {removed:,}"
        )

    df = df.loc[~remove_train].copy()

    # Also remove dev duplicates that appear in test;
    # test always has priority as untouched evaluation data.
    test_text = set(
        df.loc[
            df["split_canonical"] == "test",
            "clean_text"
        ]
    )

    dev_leak_test = (
        (df["split_canonical"] == "dev")
        & df["clean_text"].isin(test_text)
    )

    removed_dev = int(dev_leak_test.sum())

    if removed_dev:
        print(
            "Removing dev sentences duplicated in test:",
            f"{removed_dev:,}"
        )

    df = df.loc[~dev_leak_test].copy()

    # --------------------------------------------------------
    # Final leakage assertions.
    # --------------------------------------------------------

    sets = {
        s: set(
            df.loc[
                df["split_canonical"] == s,
                "clean_text"
            ]
        )
        for s in ["train", "dev", "test"]
    }

    assert not (sets["train"] & sets["dev"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["dev"] & sets["test"])

    # --------------------------------------------------------
    # Save canonical manifest
    # --------------------------------------------------------

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

    out = df[keep].copy()

    out.to_parquet(
        OUT / f"{name}.parquet",
        index=False,
        compression="zstd",
    )

    all_parts.append(out)

    print()
    print("canonical:")
    print(
        out.groupby("split_canonical").agg(
            rows=("text", "size"),
            docs=("doc_id", "nunique"),
        )
    )

    for split, g in out.groupby("split_canonical"):
        summary.append({
            "dataset": name,
            "split": split,
            "rows": len(g),
            "documents": g["doc_id"].nunique(),
            "labels": g["label"].nunique(),
        })


# ============================================================
# Combined manifest
# ============================================================

combined = pd.concat(
    all_parts,
    ignore_index=True,
)

combined.to_parquet(
    OUT / "rr_benchmark_all.parquet",
    index=False,
    compression="zstd",
)

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    REP / "canonical_split_summary.csv",
    index=False,
)

# ============================================================
# Label distributions
# ============================================================

labels = (
    combined.groupby(
        ["dataset", "split_canonical", "label"]
    )
    .size()
    .reset_index(name="count")
)

labels.to_csv(
    REP / "canonical_split_label_counts.csv",
    index=False,
)

print()
print("=" * 76)
print("FINAL CANONICAL BENCHMARK")
print("=" * 76)
print(summary_df.to_string(index=False))

print()
print("TOTAL ROWS:", f"{len(combined):,}")
print("Saved:", OUT)
