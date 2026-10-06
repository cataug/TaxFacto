from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


ROOT = Path.home() / "TaxFacto"

TRANSFER = (
    ROOT
    / "results/additional/transfer"
)

NATIVE = (
    ROOT
    / "results/additional/native"
)

OUT = (
    ROOT
    / "results/additional/analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


DATASETS = [
    "legaleval",
    "marro_india",
    "marro_uk",
    "iltur_cl",
    "iltur_it",
]

METHODS_CORE = [
    "full_ft",
    "shared_lora",
    "category_lora",
    "role_adapter",
]


# ============================================================
# EXISTING WITHIN-DOMAIN RESULTS
# ============================================================

main = pd.read_csv(
    ROOT
    / "results/final_analysis"
    / "main_cell_mean_std.csv"
)

main = main[
    main["model_key"]
    == "inlegalbert"
].copy()

within = {
    (
        r["dataset_key"],
        r["method_key"],
    ):
    r["macro_f1_mean"]

    for _, r
    in main.iterrows()
}


# ============================================================
# TRANSFER
# ============================================================

transfer_rows = []

pattern = re.compile(
    r"^xfer-(.+)-to-(.+)"
    r"__legaleval"
    r"__(.+)"
    r"__seed(\d+)$"
)

for fn in sorted(
    TRANSFER.glob(
        "*/run_summary.json"
    )
):

    run = fn.parent.name

    m = pattern.match(
        run
    )

    if not m:
        raise RuntimeError(
            f"Cannot parse {run}"
        )

    source = m.group(1)
    target = m.group(2)
    method = m.group(3)
    seed = int(
        m.group(4)
    )

    with open(fn) as f:
        x = json.load(f)

    transfer_rows.append({
        "source":
            source,

        "target":
            target,

        "method":
            method,

        "seed":
            seed,

        "macro_f1":
            x["macro_f1"],

        "weighted_f1":
            x["weighted_f1"],

        "accuracy":
            x["accuracy"],

        "train_seconds":
            x["train_seconds"],

        "peak_gpu_memory_gb":
            x["peak_gpu_memory_gb"],
    })


tdf = pd.DataFrame(
    transfer_rows
)

if len(tdf) != 168:
    raise RuntimeError(
        f"Expected 168 transfer runs, "
        f"got {len(tdf)}"
    )

tdf.to_csv(
    OUT
    / "transfer_all_runs.csv",
    index=False,
)


transfer_summary = (
    tdf
    .groupby(
        [
            "source",
            "target",
            "method",
        ],
        as_index=False,
    )
    .agg(
        runs=(
            "macro_f1",
            "count",
        ),

        macro_f1_mean=(
            "macro_f1",
            "mean",
        ),

        macro_f1_std=(
            "macro_f1",
            "std",
        ),

        weighted_f1_mean=(
            "weighted_f1",
            "mean",
        ),

        accuracy_mean=(
            "accuracy",
            "mean",
        ),

        train_seconds_mean=(
            "train_seconds",
            "mean",
        ),
    )
)


def target_baseline(row):

    return within.get(
        (
            row["target"],
            row["method"],
        ),
        np.nan,
    )


transfer_summary[
    "target_within_f1"
] = transfer_summary.apply(
    target_baseline,
    axis=1,
)

transfer_summary[
    "transfer_gap"
] = (
    transfer_summary[
        "target_within_f1"
    ]
    -
    transfer_summary[
        "macro_f1_mean"
    ]
)

transfer_summary[
    "retention"
] = (
    transfer_summary[
        "macro_f1_mean"
    ]
    /
    transfer_summary[
        "target_within_f1"
    ]
)

transfer_summary.to_csv(
    OUT
    / "transfer_summary.csv",
    index=False,
)


# ============================================================
# SHARED / ROLE FULL 5×5 TRANSFER MATRICES
# ============================================================

matrices = {}

for method in [
    "shared_lora",
    "role_adapter",
]:

    mat = pd.DataFrame(
        np.nan,
        index=DATASETS,
        columns=DATASETS,
    )

    # Diagonal = within-domain baseline.
    for d in DATASETS:
        mat.loc[
            d,
            d,
        ] = within[
            (
                d,
                method,
            )
        ]

    q = transfer_summary[
        transfer_summary["method"]
        == method
    ]

    for _, r in q.iterrows():

        mat.loc[
            r["source"],
            r["target"],
        ] = r[
            "macro_f1_mean"
        ]

    matrices[method] = mat

    mat.to_csv(
        OUT
        / f"transfer_matrix_{method}.csv"
    )


delta = (
    matrices["role_adapter"]
    -
    matrices["shared_lora"]
)

delta.to_csv(
    OUT
    / "transfer_matrix_role_minus_shared.csv"
)


# ============================================================
# PAIRED ROLE VS SHARED ACROSS 20 DIRECTIONS
# ============================================================

role = (
    transfer_summary[
        transfer_summary["method"]
        == "role_adapter"
    ]
    .set_index(
        ["source", "target"]
    )["macro_f1_mean"]
)

shared = (
    transfer_summary[
        transfer_summary["method"]
        == "shared_lora"
    ]
    .set_index(
        ["source", "target"]
    )["macro_f1_mean"]
)

idx = role.index.intersection(
    shared.index
)

d = (
    role.loc[idx]
    -
    shared.loc[idx]
)

if len(d) != 20:
    raise RuntimeError(
        f"Expected 20 paired directions, "
        f"got {len(d)}"
    )

if HAVE_SCIPY:

    w = wilcoxon(
        d.to_numpy(),
        zero_method="wilcox",
        alternative="two-sided",
    )

    wil_stat = float(
        w.statistic
    )

    wil_p = float(
        w.pvalue
    )

else:
    wil_stat = np.nan
    wil_p = np.nan


paired = pd.DataFrame({
    "source":
        [
            a
            for a, b
            in idx
        ],

    "target":
        [
            b
            for a, b
            in idx
        ],

    "role_adapter_f1":
        role.loc[idx].values,

    "shared_lora_f1":
        shared.loc[idx].values,

    "delta_role_minus_shared":
        d.values,
})

paired.to_csv(
    OUT
    / "transfer_role_vs_shared.csv",
    index=False,
)


# ============================================================
# NATIVE
# ============================================================

native_rows = []

native_pattern = re.compile(
    r"^native-(.+)"
    r"__legaleval"
    r"__(.+)"
    r"__seed(\d+)$"
)

for fn in sorted(
    NATIVE.glob(
        "*/run_summary.json"
    )
):

    run = fn.parent.name

    m = native_pattern.match(
        run
    )

    if not m:
        raise RuntimeError(
            f"Cannot parse {run}"
        )

    dataset = m.group(1)
    method = m.group(2)
    seed = int(
        m.group(3)
    )

    with open(fn) as f:
        x = json.load(f)

    native_rows.append({
        "dataset":
            dataset,

        "method":
            method,

        "seed":
            seed,

        "macro_f1":
            x["macro_f1"],

        "weighted_f1":
            x["weighted_f1"],

        "accuracy":
            x["accuracy"],

        "train_seconds":
            x["train_seconds"],

        "peak_gpu_memory_gb":
            x["peak_gpu_memory_gb"],
    })


ndf = pd.DataFrame(
    native_rows
)

if len(ndf) != 27:
    raise RuntimeError(
        f"Expected 27 native runs, "
        f"got {len(ndf)}"
    )

ndf.to_csv(
    OUT
    / "native_all_runs.csv",
    index=False,
)


native_summary = (
    ndf
    .groupby(
        [
            "dataset",
            "method",
        ],
        as_index=False,
    )
    .agg(
        macro_f1_mean=(
            "macro_f1",
            "mean",
        ),

        macro_f1_std=(
            "macro_f1",
            "std",
        ),

        weighted_f1_mean=(
            "weighted_f1",
            "mean",
        ),

        accuracy_mean=(
            "accuracy",
            "mean",
        ),
    )
)

native_summary.to_csv(
    OUT
    / "native_summary.csv",
    index=False,
)


# ============================================================
# PRINT PAPER-LEVEL DIGEST
# ============================================================

print("=" * 110)
print("ADDITIONAL EXPERIMENTS")
print("=" * 110)

print(
    "transfer runs:",
    len(tdf),
)

print(
    "native runs  :",
    len(ndf),
)


print()
print("=" * 110)
print("IL-TUR CONTROLLED TRANSFER")
print("=" * 110)

controlled = transfer_summary[
    (
        (
            transfer_summary["source"]
            == "iltur_cl"
        )
        &
        (
            transfer_summary["target"]
            == "iltur_it"
        )
    )
    |
    (
        (
            transfer_summary["source"]
            == "iltur_it"
        )
        &
        (
            transfer_summary["target"]
            == "iltur_cl"
        )
    )
]

print(
    controlled[
        [
            "source",
            "target",
            "method",
            "macro_f1_mean",
            "macro_f1_std",
            "target_within_f1",
            "transfer_gap",
            "retention",
        ]
    ].to_string(
        index=False,
        formatters={
            "macro_f1_mean":
                lambda x: f"{x:.4f}",

            "macro_f1_std":
                lambda x: f"{x:.4f}",

            "target_within_f1":
                lambda x: f"{x:.4f}",

            "transfer_gap":
                lambda x: f"{x:+.4f}",

            "retention":
                lambda x: f"{x:.3f}",
        }
    )
)


print()
print("=" * 110)
print("ROLE ADAPTER TRANSFER MATRIX")
print("rows=source, cols=target; diagonal=within-domain")
print("=" * 110)

print(
    matrices[
        "role_adapter"
    ].to_string(
        float_format=lambda x: (
            f"{x:.4f}"
        )
    )
)


print()
print("=" * 110)
print("SHARED LORA TRANSFER MATRIX")
print("=" * 110)

print(
    matrices[
        "shared_lora"
    ].to_string(
        float_format=lambda x: (
            f"{x:.4f}"
        )
    )
)


print()
print("=" * 110)
print("ROLE ADAPTER - SHARED LORA")
print("=" * 110)

print(
    delta.to_string(
        float_format=lambda x: (
            f"{x:+.4f}"
        )
    )
)


print()
print("20-DIRECTION SUMMARY")
print(
    f"mean delta = "
    f"{d.mean():+.4f}"
)

print(
    f"median delta = "
    f"{d.median():+.4f}"
)

print(
    "wins/ties/losses =",
    int(
        (d > 0).sum()
    ),
    "/",
    int(
        (d == 0).sum()
    ),
    "/",
    int(
        (d < 0).sum()
    ),
)

if HAVE_SCIPY:
    print(
        f"Wilcoxon W={wil_stat:.3f}, "
        f"p={wil_p:.6g}"
    )


print()
print("=" * 110)
print("NATIVE-TAXONOMY VALIDATION")
print("=" * 110)

native_pivot = (
    native_summary
    .pivot(
        index="method",
        columns="dataset",
        values="macro_f1_mean",
    )
)

print(
    native_pivot.to_string(
        float_format=lambda x: (
            f"{x:.4f}"
        )
    )
)


print()
print("NATIVE ROLE-ADAPTER DELTAS")

for dataset in sorted(
    native_summary[
        "dataset"
    ].unique()
):

    q = (
        native_summary[
            native_summary["dataset"]
            == dataset
        ]
        .set_index("method")
    )

    role_f1 = q.loc[
        "role_adapter",
        "macro_f1_mean",
    ]

    print()
    print(dataset)

    for baseline in [
        "shared_lora",
        "full_ft",
    ]:

        b = q.loc[
            baseline,
            "macro_f1_mean",
        ]

        print(
            f"  role_adapter - "
            f"{baseline:12s}"
            f" = {role_f1-b:+.4f}"
        )


print()
print("=" * 110)
print("SAVED")
print("=" * 110)

for fn in sorted(
    OUT.glob("*")
):
    print(
        fn.relative_to(ROOT)
    )

print()
print("ADDITIONAL ANALYSIS COMPLETE")
