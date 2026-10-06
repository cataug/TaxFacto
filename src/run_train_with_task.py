from pathlib import Path
import os
import runpy
import sys

import pandas as pd


ROOT = Path.home() / "TaxFacto"

TASK_FILE = os.environ.get(
    "TAXFACTO_TASK_FILE"
)

if not TASK_FILE:
    raise RuntimeError(
        "TAXFACTO_TASK_FILE is not set"
    )

TASK_FILE = Path(
    TASK_FILE
).expanduser().resolve()

if not TASK_FILE.exists():
    raise FileNotFoundError(
        TASK_FILE
    )


# ============================================================
# Inspect task before executing train.py
# ============================================================

_original_read_parquet = (
    pd.read_parquet
)

task_df = _original_read_parquet(
    TASK_FILE
)

if "label" not in task_df.columns:
    raise RuntimeError(
        f"{TASK_FILE}: no label column"
    )


COMMON7 = [
    "ARG",
    "FAC",
    "PRE",
    "RATIO",
    "RLC",
    "RPC",
    "STA",
]

observed = list(
    dict.fromkeys(
        task_df["label"]
        .astype(str)
        .tolist()
    )
)

if set(observed) == set(COMMON7):
    LABELS = COMMON7
else:
    LABELS = sorted(
        set(observed)
    )


# ============================================================
# Redirect the parquet read used by train.py
# ============================================================

def redirected_read_parquet(
    path,
    *args,
    **kwargs,
):

    p = Path(path)

    s = str(p)

    if (
        p.name == "legaleval.parquet"
        or "data/tasks/common7" in s
    ):
        return _original_read_parquet(
            TASK_FILE,
            *args,
            **kwargs,
        )

    return _original_read_parquet(
        path,
        *args,
        **kwargs,
    )


pd.read_parquet = (
    redirected_read_parquet
)


# ============================================================
# Load train.py without executing main().
# ============================================================

train_path = (
    ROOT / "src/train.py"
)

ns = runpy.run_path(
    str(train_path),
    run_name="taxfacto_train_module",
)


# ============================================================
# Patch module-level Common-7 globals when a native
# label space is used.
#
# Mutation is used where possible because function defaults
# may hold references to the original list/dict object.
# ============================================================

common_set = set(COMMON7)

patched = []


for name, value in list(
    ns.items()
):

    lname = name.lower()

    # --------------------------------------------
    # List
    # --------------------------------------------

    if isinstance(value, list):

        values = set(
            str(v)
            for v in value
        )

        if values == common_set:

            value[:] = LABELS

            patched.append(
                name
            )

    # --------------------------------------------
    # Set
    # --------------------------------------------

    elif isinstance(value, set):

        values = set(
            str(v)
            for v in value
        )

        if values == common_set:

            value.clear()
            value.update(
                LABELS
            )

            patched.append(
                name
            )

    # --------------------------------------------
    # Tuple
    # --------------------------------------------

    elif isinstance(value, tuple):

        values = set(
            str(v)
            for v in value
        )

        if values == common_set:

            ns[name] = tuple(
                LABELS
            )

            patched.append(
                name
            )

    # --------------------------------------------
    # dict LABEL -> ID
    # --------------------------------------------

    elif isinstance(value, dict):

        key_strings = set(
            str(k)
            for k in value.keys()
        )

        val_strings = set(
            str(v)
            for v in value.values()
        )

        if key_strings == common_set:

            value.clear()

            value.update({
                label: i
                for i, label
                in enumerate(LABELS)
            })

            patched.append(
                name
            )

        elif val_strings == common_set:

            value.clear()

            value.update({
                i: label
                for i, label
                in enumerate(LABELS)
            })

            patched.append(
                name
            )

    # --------------------------------------------
    # NUM_LABELS style constant
    # --------------------------------------------

    elif (
        isinstance(value, int)
        and value == 7
        and "label" in lname
    ):

        ns[name] = len(
            LABELS
        )

        patched.append(
            name
        )


print("=" * 80)
print("TASK OVERRIDE")
print("=" * 80)
print("task file :", TASK_FILE)
print("labels    :", LABELS)
print("n_labels  :", len(LABELS))
print(
    "patched globals:",
    patched if patched else "none needed",
)
print("=" * 80)


if "main" not in ns:
    raise RuntimeError(
        "src/train.py has no main()"
    )


# train.py's argparse sees the same command line
# arguments that were passed to this wrapper.
ns["main"]()
