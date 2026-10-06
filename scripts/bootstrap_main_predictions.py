from pathlib import Path
from itertools import combinations
import json
import os

import numpy as np
import pandas as pd


ROOT = Path.home() / "TaxFacto"

RUNS = (
    ROOT
    / "results/final_common7"
)

TASKS = (
    ROOT
    / "data/tasks/common7"
)

OUT = (
    ROOT
    / "results/additional/bootstrap"
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

MODELS = [
    "inlegalbert",
    "legalbert",
    "deberta",
]

METHODS = [
    "frozen",
    "full_ft",
    "shared_lora",
    "shared_lora_widehead",
    "category_lora",
    "role_adapter",
]

SEEDS = [
    42,
    43,
    44,
]

# All 15 method-pair comparisons.
COMPARISONS = list(
    combinations(
        METHODS,
        2,
    )
)

B = int(
    os.environ.get(
        "BOOTSTRAPS",
        "10000",
    )
)

RNG_SEED = 20261005

rng = np.random.default_rng(
    RNG_SEED
)


TRUE_CANDIDATES = [
    "true_label",
    "gold_label",
    "y_true",
    "target",
    "gold",
    "label",
    "true",
    "true_id",
]

PRED_CANDIDATES = [
    "pred_label",
    "predicted_label",
    "prediction",
    "y_pred",
    "pred",
    "predicted",
    "pred_id",
]


def canon(v):

    if pd.isna(v):
        return "<NA>"

    if isinstance(
        v,
        (float, np.floating),
    ):
        if float(v).is_integer():
            return str(
                int(v)
            )

    return str(v)


def detect_columns(df):

    true_col = next(
        (
            c
            for c in TRUE_CANDIDATES
            if c in df.columns
        ),
        None,
    )

    pred_col = next(
        (
            c
            for c in PRED_CANDIDATES
            if c in df.columns
        ),
        None,
    )

    if (
        true_col is None
        or pred_col is None
    ):
        raise RuntimeError(
            "Could not detect true/pred columns.\n"
            f"Columns: {list(df.columns)}"
        )

    return (
        true_col,
        pred_col,
    )


def run_dir(
    model,
    dataset,
    method,
    seed,
):

    return (
        RUNS
        / (
            f"main__{model}"
            f"__r8a16"
            f"__{dataset}"
            f"__{method}"
            f"__seed{seed}"
        )
    )


def macro_f1_from_cm(cm):
    """
    cm shape:
        [B, C, C]

    Rows=true, cols=pred.
    """

    tp = np.diagonal(
        cm,
        axis1=1,
        axis2=2,
    )

    rows = cm.sum(
        axis=2
    )

    cols = cm.sum(
        axis=1
    )

    fn = rows - tp
    fp = cols - tp

    denom = (
        2 * tp
        + fp
        + fn
    )

    f1 = np.divide(
        2 * tp,
        denom,
        out=np.full(
            denom.shape,
            np.nan,
            dtype=float,
        ),
        where=(
            denom != 0
        ),
    )

    return np.nanmean(
        f1,
        axis=1,
    )


def build_doc_cm(
    y_true,
    y_pred,
    doc_idx,
    n_docs,
    class_to_id,
):

    c = len(
        class_to_id
    )

    out = np.zeros(
        (
            n_docs,
            c,
            c,
        ),
        dtype=np.int32,
    )

    for i in range(
        len(y_true)
    ):

        t = class_to_id[
            y_true[i]
        ]

        p = class_to_id[
            y_pred[i]
        ]

        d = doc_idx[i]

        out[
            d,
            t,
            p,
        ] += 1

    return out


cell_rows = []

# Store bootstrap deltas for later
# hierarchical aggregate bootstrap.
cell_boot = {}


print("=" * 100)
print("DOCUMENT-LEVEL PAIRED BOOTSTRAP")
print("=" * 100)
print("bootstrap replicates:", B)
print()


for model in MODELS:

    for dataset in DATASETS:

        task = pd.read_parquet(
            TASKS
            / f"{dataset}.parquet"
        )

        test = task[
            task["split_canonical"]
            == "test"
        ].reset_index(
            drop=True
        )

        docs = (
            test["doc_id"]
            .astype(str)
            .tolist()
        )

        doc_order = list(
            dict.fromkeys(docs)
        )

        doc_to_idx = {
            d: i
            for i, d
            in enumerate(doc_order)
        }

        doc_idx = np.array(
            [
                doc_to_idx[d]
                for d in docs
            ],
            dtype=np.int32,
        )

        n_docs = len(
            doc_order
        )

        loaded = {}

        reference_true = None

        all_values = set()

        point_f1 = {}

        for method in METHODS:

            loaded[method] = {}

            point_vals = []

            for seed in SEEDS:

                rd = run_dir(
                    model,
                    dataset,
                    method,
                    seed,
                )

                pred_fn = (
                    rd
                    / "test_predictions.csv"
                )

                sum_fn = (
                    rd
                    / "run_summary.json"
                )

                if not pred_fn.exists():
                    raise FileNotFoundError(
                        pred_fn
                    )

                if not sum_fn.exists():
                    raise FileNotFoundError(
                        sum_fn
                    )

                pred = pd.read_csv(
                    pred_fn
                )

                if len(pred) != len(test):
                    raise RuntimeError(
                        f"{rd.name}: "
                        f"{len(pred)} predictions "
                        f"!= {len(test)} test rows"
                    )

                tc, pc = detect_columns(
                    pred
                )

                yt = np.array(
                    [
                        canon(v)
                        for v in pred[tc]
                    ],
                    dtype=object,
                )

                yp = np.array(
                    [
                        canon(v)
                        for v in pred[pc]
                    ],
                    dtype=object,
                )

                if reference_true is None:
                    reference_true = yt
                else:
                    if not np.array_equal(
                        reference_true,
                        yt,
                    ):
                        raise RuntimeError(
                            f"y_true mismatch in "
                            f"{model}/{dataset}"
                        )

                all_values.update(
                    yt.tolist()
                )

                all_values.update(
                    yp.tolist()
                )

                loaded[
                    method
                ][seed] = (
                    yt,
                    yp,
                )

                with open(
                    sum_fn
                ) as f:
                    s = json.load(f)

                point_vals.append(
                    float(
                        s["macro_f1"]
                    )
                )

            point_f1[method] = (
                float(
                    np.mean(
                        point_vals
                    )
                )
            )

        classes = sorted(
            all_values
        )

        class_to_id = {
            v: i
            for i, v
            in enumerate(classes)
        }

        # Paired document bootstrap indices.
        counts = rng.multinomial(
            n_docs,
            np.repeat(
                1.0 / n_docs,
                n_docs,
            ),
            size=B,
        ).astype(
            np.int16
        )

        boot_scores = {}

        for method in METHODS:

            seed_scores = []

            for seed in SEEDS:

                yt, yp = (
                    loaded[
                        method
                    ][seed]
                )

                doc_cm = build_doc_cm(
                    yt,
                    yp,
                    doc_idx,
                    n_docs,
                    class_to_id,
                )

                flat = doc_cm.reshape(
                    n_docs,
                    -1,
                )

                boot_flat = (
                    counts @ flat
                )

                boot_cm = (
                    boot_flat.reshape(
                        B,
                        len(classes),
                        len(classes),
                    )
                )

                seed_scores.append(
                    macro_f1_from_cm(
                        boot_cm
                    )
                )

            boot_scores[method] = (
                np.mean(
                    np.stack(
                        seed_scores,
                        axis=0,
                    ),
                    axis=0,
                )
            )

        for a, b in COMPARISONS:

            delta = (
                boot_scores[a]
                - boot_scores[b]
            )

            point_delta = (
                point_f1[a]
                - point_f1[b]
            )

            lo, hi = np.quantile(
                delta,
                [
                    0.025,
                    0.975,
                ],
            )

            key = (
                model,
                dataset,
                a,
                b,
            )

            cell_boot[key] = (
                delta.astype(
                    np.float32
                )
            )

            cell_rows.append({
                "model":
                    model,

                "dataset":
                    dataset,

                "method_a":
                    a,

                "method_b":
                    b,

                "documents":
                    n_docs,

                "point_delta":
                    point_delta,

                "ci95_low":
                    float(lo),

                "ci95_high":
                    float(hi),

                "p_delta_gt_0":
                    float(
                        np.mean(
                            delta > 0
                        )
                    ),

                "p_delta_lt_0":
                    float(
                        np.mean(
                            delta < 0
                        )
                    ),
            })

        print(
            f"{model:14s} "
            f"{dataset:14s} "
            f"docs={n_docs:4d} "
            "OK"
        )


cell_df = pd.DataFrame(
    cell_rows
)

cell_df.to_csv(
    OUT
    / "document_bootstrap_cells.csv",
    index=False,
)


# ============================================================
# HIERARCHICAL GLOBAL BOOTSTRAP
#
# Resample the 15 dataset×backbone blocks,
# then draw one document-bootstrap replicate
# from each selected block.
# ============================================================

global_rows = []

blocks = [
    (m, d)
    for m in MODELS
    for d in DATASETS
]

n_blocks = len(
    blocks
)

for a, b in COMPARISONS:

    stack = np.stack(
        [
            cell_boot[
                (
                    model,
                    dataset,
                    a,
                    b,
                )
            ]
            for model, dataset
            in blocks
        ],
        axis=0,
    )

    block_choices = rng.integers(
        0,
        n_blocks,
        size=(
            B,
            n_blocks,
        ),
    )

    draw_choices = rng.integers(
        0,
        B,
        size=(
            B,
            n_blocks,
        ),
    )

    vals = np.empty(
        (
            B,
            n_blocks,
        ),
        dtype=np.float32,
    )

    for j in range(
        n_blocks
    ):

        vals[:, j] = stack[
            block_choices[:, j],
            draw_choices[:, j],
        ]

    aggregate = vals.mean(
        axis=1
    )

    point = (
        cell_df[
            (
                cell_df["method_a"]
                == a
            )
            &
            (
                cell_df["method_b"]
                == b
            )
        ]["point_delta"]
        .mean()
    )

    lo, hi = np.quantile(
        aggregate,
        [
            0.025,
            0.975,
        ],
    )

    global_rows.append({
        "method_a":
            a,

        "method_b":
            b,

        "point_mean_delta":
            point,

        "ci95_low":
            float(lo),

        "ci95_high":
            float(hi),

        "p_delta_gt_0":
            float(
                np.mean(
                    aggregate > 0
                )
            ),

        "p_delta_lt_0":
            float(
                np.mean(
                    aggregate < 0
                )
            ),
    })


global_df = pd.DataFrame(
    global_rows
)

global_df.to_csv(
    OUT
    / "hierarchical_bootstrap_global.csv",
    index=False,
)


print()
print("=" * 100)
print("GLOBAL HIERARCHICAL BOOTSTRAP")
print("=" * 100)

focus = global_df[
    (
        global_df["method_a"]
        == "role_adapter"
    )
    |
    (
        global_df["method_b"]
        == "role_adapter"
    )
    |
    (
        (
            global_df["method_a"]
            == "full_ft"
        )
        &
        (
            global_df["method_b"]
            == "shared_lora"
        )
    )
]

print(
    focus.to_string(
        index=False,
        formatters={
            "point_mean_delta":
                lambda x: f"{x:+.4f}",

            "ci95_low":
                lambda x: f"{x:+.4f}",

            "ci95_high":
                lambda x: f"{x:+.4f}",

            "p_delta_gt_0":
                lambda x: f"{x:.4f}",

            "p_delta_lt_0":
                lambda x: f"{x:.4f}",
        }
    )
)

print()
print("BOOTSTRAP COMPLETE")
