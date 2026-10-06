from pathlib import Path
import hashlib
import re
import unicodedata

import pandas as pd


ROOT = Path.home() / "TaxFacto"

COMMON = ROOT / "data/tasks/common7"
NATIVE_SRC = ROOT / "data/tasks/native"

TRANSFER_OUT = ROOT / "data/tasks/transfer"
NATIVE_OUT = ROOT / "data/tasks/native_eval"

REPORT = ROOT / "reports/additional"

for p in [
    TRANSFER_OUT,
    NATIVE_OUT,
    REPORT,
]:
    p.mkdir(parents=True, exist_ok=True)


DATASETS = [
    "legaleval",
    "marro_india",
    "marro_uk",
    "iltur_cl",
    "iltur_it",
]

NATIVE_TASKS = [
    "legaleval",
    "iltur_cl",
    "iltur_it",
]

COMMON7 = [
    "ARG",
    "FAC",
    "PRE",
    "RATIO",
    "RLC",
    "RPC",
    "STA",
]


def norm_text(x):
    x = unicodedata.normalize(
        "NFKC",
        str(x),
    )
    x = x.lower()
    x = re.sub(
        r"\s+",
        " ",
        x,
    ).strip()
    x = re.sub(
        r"[^\w\s]",
        "",
        x,
    )
    x = re.sub(
        r"\s+",
        " ",
        x,
    ).strip()
    return x


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def standardize(
    x,
    origin,
):
    x = x.copy()

    required = [
        "text",
        "label",
        "doc_id",
        "split_canonical",
    ]

    missing = [
        c
        for c in required
        if c not in x.columns
    ]

    if missing:
        raise RuntimeError(
            f"{origin}: missing columns "
            f"{missing}"
        )

    if "sentence_id" not in x.columns:
        x["sentence_id"] = (
            x.groupby("doc_id")
            .cumcount()
            .astype(str)
        )

    y = pd.DataFrame({
        "text":
            x["text"].astype(str),

        "label":
            x["label"].astype(str),

        "doc_id":
            (
                origin
                + "::"
                + x["doc_id"].astype(str)
            ),

        "sentence_id":
            x["sentence_id"].astype(str),

        "split_canonical":
            x["split_canonical"].astype(str),

        "origin_dataset":
            origin,
    })

    return y


# ============================================================
# LOAD COMMON-7
# ============================================================

common = {}

for name in DATASETS:

    fn = COMMON / f"{name}.parquet"

    if not fn.exists():
        raise FileNotFoundError(fn)

    x = pd.read_parquet(fn)

    x = standardize(
        x,
        name,
    )

    labels = sorted(
        x["label"].unique()
    )

    if set(labels) != set(COMMON7):
        raise RuntimeError(
            f"{name}: expected Common-7, "
            f"got {labels}"
        )

    x["_norm"] = (
        x["text"].map(norm_text)
    )

    common[name] = x


# ============================================================
# BUILD ALL 20 DIRECTED TRANSFER TASKS
# ============================================================

manifest_rows = []

for source in DATASETS:

    for target in DATASETS:

        if source == target:
            continue

        src = common[source]
        tgt = common[target]

        src_train = src[
            src["split_canonical"]
            == "train"
        ].copy()

        src_dev = src[
            src["split_canonical"]
            == "dev"
        ].copy()

        tgt_test = tgt[
            tgt["split_canonical"]
            == "test"
        ].copy()

        target_test_norm = set(
            tgt_test["_norm"]
        )

        train_before = len(
            src_train
        )

        dev_before = len(
            src_dev
        )

        # Pair-specific decontamination:
        # target test wins.
        src_train = src_train[
            ~src_train["_norm"].isin(
                target_test_norm
            )
        ].copy()

        src_dev = src_dev[
            ~src_dev["_norm"].isin(
                target_test_norm
            )
        ].copy()

        train_removed = (
            train_before
            - len(src_train)
        )

        dev_removed = (
            dev_before
            - len(src_dev)
        )

        # Explicit leakage audit.
        train_overlap = len(
            set(src_train["_norm"])
            & target_test_norm
        )

        dev_overlap = len(
            set(src_dev["_norm"])
            & target_test_norm
        )

        if train_overlap != 0:
            raise RuntimeError(
                f"{source}->{target}: "
                "train/test overlap remains"
            )

        if dev_overlap != 0:
            raise RuntimeError(
                f"{source}->{target}: "
                "dev/test overlap remains"
            )

        task = pd.concat(
            [
                src_train,
                src_dev,
                tgt_test,
            ],
            ignore_index=True,
        )

        # train.py is called with --dataset legaleval.
        # This protects us if it additionally filters
        # by the dataset column.
        task["dataset"] = "legaleval"

        task["transfer_source"] = (
            source
        )

        task["transfer_target"] = (
            target
        )

        task = task.drop(
            columns=["_norm"],
            errors="ignore",
        )

        # Standardize object types for Arrow.
        for col in [
            "text",
            "label",
            "doc_id",
            "sentence_id",
            "split_canonical",
            "origin_dataset",
            "dataset",
            "transfer_source",
            "transfer_target",
        ]:
            task[col] = (
                task[col].astype(str)
            )

        fn = (
            TRANSFER_OUT
            / f"xfer__{source}__to__{target}.parquet"
        )

        task.to_parquet(
            fn,
            index=False,
        )

        manifest_rows.append({
            "source":
                source,

            "target":
                target,

            "train_rows":
                int(
                    (
                        task["split_canonical"]
                        == "train"
                    ).sum()
                ),

            "dev_rows":
                int(
                    (
                        task["split_canonical"]
                        == "dev"
                    ).sum()
                ),

            "test_rows":
                int(
                    (
                        task["split_canonical"]
                        == "test"
                    ).sum()
                ),

            "train_removed_against_target_test":
                train_removed,

            "dev_removed_against_target_test":
                dev_removed,

            "remaining_train_test_overlap":
                train_overlap,

            "remaining_dev_test_overlap":
                dev_overlap,

            "sha256":
                sha256(fn),

            "file":
                str(
                    fn.relative_to(ROOT)
                ),
        })


transfer_manifest = pd.DataFrame(
    manifest_rows
)

transfer_manifest.to_csv(
    REPORT
    / "transfer_task_manifest.csv",
    index=False,
)


# ============================================================
# NATIVE TASK ALIASES
# ============================================================

native_rows = []

for name in NATIVE_TASKS:

    fn = (
        NATIVE_SRC
        / f"{name}.parquet"
    )

    if not fn.exists():
        raise FileNotFoundError(fn)

    x = pd.read_parquet(fn)

    x = standardize(
        x,
        name,
    )

    # Same reason as transfer aliases:
    # train.py receives dataset=legaleval.
    x["dataset"] = "legaleval"

    for col in [
        "text",
        "label",
        "doc_id",
        "sentence_id",
        "split_canonical",
        "origin_dataset",
        "dataset",
    ]:
        x[col] = (
            x[col].astype(str)
        )

    out = (
        NATIVE_OUT
        / f"native__{name}.parquet"
    )

    x.to_parquet(
        out,
        index=False,
    )

    for split in [
        "train",
        "dev",
        "test",
    ]:

        q = x[
            x["split_canonical"]
            == split
        ]

        native_rows.append({
            "dataset":
                name,

            "split":
                split,

            "rows":
                len(q),

            "documents":
                q["doc_id"].nunique(),

            "labels":
                q["label"].nunique(),

            "label_names":
                "|".join(
                    sorted(
                        q["label"].unique()
                    )
                ),

            "file":
                str(
                    out.relative_to(ROOT)
                ),

            "sha256":
                sha256(out),
        })


native_manifest = pd.DataFrame(
    native_rows
)

native_manifest.to_csv(
    REPORT
    / "native_task_manifest.csv",
    index=False,
)


print("=" * 100)
print("TRANSFER TASKS")
print("=" * 100)

print(
    transfer_manifest[
        [
            "source",
            "target",
            "train_rows",
            "dev_rows",
            "test_rows",
            "train_removed_against_target_test",
            "dev_removed_against_target_test",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "transfer tasks:",
    len(transfer_manifest),
)

print()
print("=" * 100)
print("NATIVE TASKS")
print("=" * 100)

print(
    native_manifest.to_string(
        index=False
    )
)

print()
print("ADDITIONAL TASK PREPARATION COMPLETE")
