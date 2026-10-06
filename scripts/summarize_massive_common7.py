from pathlib import Path
import json
import re
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon, friedmanchisquare
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


ROOT = Path.home() / "TaxFacto"
SRC = ROOT / "results/final_common7"
OUT = ROOT / "results/final_analysis"

OUT.mkdir(parents=True, exist_ok=True)

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

SEEDS = [42, 43, 44]

PEFT_RANK_METHODS = [
    "shared_lora",
    "category_lora",
    "role_adapter",
]


def fmt4(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.4f}"


def fmt3(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.3f}"


def parse_run_name(name):
    """
    Expected:
      main__inlegalbert__r8a16__legaleval__role_adapter__seed42
      rank__inlegalbert__r16a32__...
      mechanistic__inlegalbert__r8a16__...
    """
    parts = name.split("__")

    if len(parts) != 6:
        raise ValueError(
            f"Unexpected run directory name: {name}"
        )

    phase, model_key, ra, dataset, method, seedpart = parts

    m = re.fullmatch(r"r(\d+)a(\d+)", ra)
    if not m:
        raise ValueError(f"Bad rank/alpha component: {ra}")

    sm = re.fullmatch(r"seed(\d+)", seedpart)
    if not sm:
        raise ValueError(f"Bad seed component: {seedpart}")

    return {
        "phase": phase,
        "model_key": model_key,
        "rank": int(m.group(1)),
        "alpha": int(m.group(2)),
        "dataset_key": dataset,
        "method_key": method,
        "seed_key": int(sm.group(1)),
    }


# =============================================================================
# LOAD ALL 335 RUNS
# =============================================================================

rows = []
per_class_rows = []
bad = []

summary_files = sorted(
    SRC.glob("*/run_summary.json")
)

for fn in summary_files:
    run_dir = fn.parent
    run_name = run_dir.name

    try:
        parsed = parse_run_name(run_name)

        with open(fn, "r") as f:
            x = json.load(f)

        row = {
            **parsed,

            "run_name": run_name,

            "dataset": x.get(
                "dataset",
                parsed["dataset_key"],
            ),
            "model": x.get("model"),
            "method": x.get(
                "method",
                parsed["method_key"],
            ),
            "seed": int(
                x.get(
                    "seed",
                    parsed["seed_key"],
                )
            ),

            "epochs": x.get("epochs"),
            "max_length": x.get("max_length"),
            "batch_size": x.get("batch_size"),
            "learning_rate": x.get("learning_rate"),

            "train_rows": x.get("train_rows"),
            "dev_rows": x.get("dev_rows"),
            "test_rows": x.get("test_rows"),

            "parameters_total": x.get("parameters_total"),
            "parameters_trainable": x.get(
                "parameters_trainable"
            ),
            "trainable_pct": x.get("trainable_pct"),

            "train_seconds": x.get("train_seconds"),
            "peak_gpu_memory_gb": x.get(
                "peak_gpu_memory_gb"
            ),

            "best_dev_macro_f1": x.get(
                "best_dev_macro_f1"
            ),
            "best_dev_epoch": x.get(
                "best_dev_epoch"
            ),

            "accuracy": x.get("accuracy"),
            "macro_precision": x.get(
                "macro_precision"
            ),
            "macro_recall": x.get(
                "macro_recall"
            ),
            "macro_f1": x.get("macro_f1"),
            "weighted_f1": x.get(
                "weighted_f1"
            ),
        }

        rows.append(row)

        # ---------------------------------------------------------
        # PER-CLASS REPORT
        # ---------------------------------------------------------

        report_fn = run_dir / "classification_report.json"

        if report_fn.exists():
            with open(report_fn, "r") as f:
                rep = json.load(f)

            for label, vals in rep.items():
                if not isinstance(vals, dict):
                    continue

                if "f1-score" not in vals:
                    continue

                # Skip sklearn aggregate pseudo-labels.
                if label in {
                    "macro avg",
                    "weighted avg",
                    "micro avg",
                    "samples avg",
                }:
                    continue

                per_class_rows.append({
                    **parsed,
                    "run_name": run_name,
                    "label": str(label),
                    "precision": vals.get(
                        "precision"
                    ),
                    "recall": vals.get("recall"),
                    "f1": vals.get("f1-score"),
                    "support": vals.get("support"),
                })

    except Exception as e:
        bad.append((str(fn), str(e)))


df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError("No final runs found.")


# =============================================================================
# INVENTORY CHECK
# =============================================================================

print("=" * 120)
print("TAXFACTO — MASSIVE COMMON-7 FINAL ANALYSIS")
print("=" * 120)

print("summary files found :", len(summary_files))
print("successfully parsed :", len(df))
print("parse failures       :", len(bad))

if bad:
    print("\nBAD FILES:")
    for fn, err in bad:
        print(" ", fn)
        print("    ", err)

print()
print("PHASE COUNTS")
print(
    df["phase"]
    .value_counts()
    .sort_index()
    .to_string()
)

expected_phase_counts = {
    "main": 270,
    "rank": 60,
    "mechanistic": 5,
}

for phase, n in expected_phase_counts.items():
    got = int((df["phase"] == phase).sum())
    if got != n:
        raise RuntimeError(
            f"{phase}: expected {n}, got {got}"
        )

if len(df) != 335:
    raise RuntimeError(
        f"Expected 335 total runs, got {len(df)}"
    )

if df["run_name"].duplicated().any():
    raise RuntimeError("Duplicate run names found.")

print("\nINVENTORY CHECK: 335/335 OK")


# =============================================================================
# SAVE ALL RUNS
# =============================================================================

df = df.sort_values(
    [
        "phase",
        "model_key",
        "dataset_key",
        "method_key",
        "seed",
        "rank",
    ]
).reset_index(drop=True)

df.to_csv(
    OUT / "all_335_runs.csv",
    index=False,
)


# =============================================================================
# MAIN GRID
# =============================================================================

main = df[df["phase"] == "main"].copy()

if len(main) != 270:
    raise RuntimeError(
        f"Main grid should contain 270 rows; got {len(main)}"
    )

# Verify every cell has exactly 3 seeds.
seed_check = (
    main.groupby(
        ["model_key", "dataset_key", "method_key"]
    )["seed"]
    .nunique()
)

if not (seed_check == 3).all():
    print(
        seed_check[
            seed_check != 3
        ].to_string()
    )
    raise RuntimeError(
        "Some main cells do not have 3 seeds."
    )


# =============================================================================
# MAIN: CELL MEAN ± STD ACROSS SEEDS
# =============================================================================

main_cells = (
    main
    .groupby(
        ["model_key", "dataset_key", "method_key"],
        as_index=False,
    )
    .agg(
        n=("macro_f1", "count"),

        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),

        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),

        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),

        macro_precision_mean=(
            "macro_precision",
            "mean",
        ),
        macro_recall_mean=(
            "macro_recall",
            "mean",
        ),

        train_seconds_mean=(
            "train_seconds",
            "mean",
        ),
        train_seconds_std=(
            "train_seconds",
            "std",
        ),

        peak_vram_gb_mean=(
            "peak_gpu_memory_gb",
            "mean",
        ),

        parameters_total=(
            "parameters_total",
            "first",
        ),
        parameters_trainable=(
            "parameters_trainable",
            "first",
        ),
        trainable_pct=(
            "trainable_pct",
            "first",
        ),
    )
)

main_cells.to_csv(
    OUT / "main_cell_mean_std.csv",
    index=False,
)


# =============================================================================
# MAIN: OVERALL METHOD RANKING
#
# Average equally over the 15 model × dataset cells.
# Not sentence-weighted.
# =============================================================================

method_overall = (
    main_cells
    .groupby("method_key", as_index=False)
    .agg(
        cells=("macro_f1_mean", "count"),

        macro_f1_mean=(
            "macro_f1_mean",
            "mean",
        ),
        macro_f1_std_across_cells=(
            "macro_f1_mean",
            "std",
        ),

        min_cell_f1=(
            "macro_f1_mean",
            "min",
        ),
        max_cell_f1=(
            "macro_f1_mean",
            "max",
        ),

        weighted_f1_mean=(
            "weighted_f1_mean",
            "mean",
        ),
        accuracy_mean=(
            "accuracy_mean",
            "mean",
        ),

        train_seconds_mean=(
            "train_seconds_mean",
            "mean",
        ),
        peak_vram_gb_mean=(
            "peak_vram_gb_mean",
            "mean",
        ),
    )
    .sort_values(
        "macro_f1_mean",
        ascending=False,
    )
)

method_overall.to_csv(
    OUT / "main_method_overall.csv",
    index=False,
)


print()
print("=" * 120)
print("MAIN GRID — OVERALL METHOD RANKING")
print("equal weight for each of 15 dataset × backbone cells")
print("=" * 120)

print(
    method_overall.to_string(
        index=False,
        formatters={
            "macro_f1_mean": fmt4,
            "macro_f1_std_across_cells": fmt4,
            "min_cell_f1": fmt4,
            "max_cell_f1": fmt4,
            "weighted_f1_mean": fmt4,
            "accuracy_mean": fmt4,
            "train_seconds_mean": lambda x: f"{x:.1f}",
            "peak_vram_gb_mean": lambda x: f"{x:.2f}",
        }
    )
)


# =============================================================================
# MAIN: METHOD × BACKBONE
# =============================================================================

by_backbone = (
    main_cells
    .groupby(
        ["model_key", "method_key"],
        as_index=False,
    )
    .agg(
        macro_f1_mean=(
            "macro_f1_mean",
            "mean",
        ),
        macro_f1_std=(
            "macro_f1_mean",
            "std",
        ),
        weighted_f1_mean=(
            "weighted_f1_mean",
            "mean",
        ),
        accuracy_mean=(
            "accuracy_mean",
            "mean",
        ),
    )
)

by_backbone.to_csv(
    OUT / "main_by_backbone_method.csv",
    index=False,
)

print()
print("=" * 120)
print("MAIN GRID — MACRO-F1 BY BACKBONE")
print("=" * 120)

bb_pivot = by_backbone.pivot(
    index="method_key",
    columns="model_key",
    values="macro_f1_mean",
).reindex(METHODS)

print(
    bb_pivot.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# =============================================================================
# MAIN: METHOD × DATASET
# =============================================================================

by_dataset = (
    main_cells
    .groupby(
        ["dataset_key", "method_key"],
        as_index=False,
    )
    .agg(
        macro_f1_mean=(
            "macro_f1_mean",
            "mean",
        ),
        macro_f1_std=(
            "macro_f1_mean",
            "std",
        ),
        weighted_f1_mean=(
            "weighted_f1_mean",
            "mean",
        ),
        accuracy_mean=(
            "accuracy_mean",
            "mean",
        ),
    )
)

by_dataset.to_csv(
    OUT / "main_by_dataset_method.csv",
    index=False,
)

print()
print("=" * 120)
print("MAIN GRID — MACRO-F1 BY DATASET")
print("averaged over 3 backbones after seed averaging")
print("=" * 120)

ds_pivot = by_dataset.pivot(
    index="method_key",
    columns="dataset_key",
    values="macro_f1_mean",
).reindex(METHODS)

print(
    ds_pivot.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# =============================================================================
# BEST METHOD PER DATASET × BACKBONE
# =============================================================================

idx = (
    main_cells
    .groupby(
        ["model_key", "dataset_key"]
    )["macro_f1_mean"]
    .idxmax()
)

best_cells = (
    main_cells
    .loc[idx]
    .sort_values(
        ["model_key", "dataset_key"]
    )
    .reset_index(drop=True)
)

best_cells.to_csv(
    OUT / "main_best_method_per_cell.csv",
    index=False,
)

print()
print("=" * 120)
print("BEST METHOD PER DATASET × BACKBONE")
print("=" * 120)

for _, r in best_cells.iterrows():
    print(
        f"{r['model_key']:14s} "
        f"{r['dataset_key']:14s} "
        f"{r['method_key']:24s} "
        f"{r['macro_f1_mean']:.4f} "
        f"± {r['macro_f1_std']:.4f}"
    )


# =============================================================================
# GLOBAL BLOCKED STATISTICS
#
# Experimental blocks = 15 dataset × backbone combinations.
# Seed means are used first, avoiding treating the three seeds as
# 45 completely independent experimental units.
# =============================================================================

cell_pivot = main_cells.pivot(
    index=["model_key", "dataset_key"],
    columns="method_key",
    values="macro_f1_mean",
)

cell_pivot = cell_pivot[METHODS]

if cell_pivot.isna().any().any():
    raise RuntimeError(
        "Missing cells in paired method comparison."
    )


print()
print("=" * 120)
print("BLOCKED STATISTICAL COMPARISON")
print("15 blocks = 5 datasets × 3 backbones; value = mean over 3 seeds")
print("=" * 120)

friedman_stat = np.nan
friedman_p = np.nan

if HAVE_SCIPY:
    arrays = [
        cell_pivot[m].to_numpy()
        for m in METHODS
    ]

    fr = friedmanchisquare(*arrays)

    friedman_stat = float(fr.statistic)
    friedman_p = float(fr.pvalue)

    print(
        f"Friedman chi-square = {friedman_stat:.4f}, "
        f"p = {friedman_p:.6g}"
    )
else:
    print("SciPy unavailable: statistical tests skipped.")


pair_rows = []

for a, b in combinations(METHODS, 2):

    va = cell_pivot[a].to_numpy()
    vb = cell_pivot[b].to_numpy()

    d = va - vb

    wins = int((d > 1e-12).sum())
    losses = int((d < -1e-12).sum())
    ties = int(len(d) - wins - losses)

    p = np.nan
    stat = np.nan

    if HAVE_SCIPY:
        try:
            res = wilcoxon(
                d,
                zero_method="wilcox",
                alternative="two-sided",
            )
            stat = float(res.statistic)
            p = float(res.pvalue)
        except ValueError:
            p = 1.0
            stat = 0.0

    pair_rows.append({
        "method_a": a,
        "method_b": b,
        "mean_delta_a_minus_b": float(
            np.mean(d)
        ),
        "median_delta_a_minus_b": float(
            np.median(d)
        ),
        "wins_a": wins,
        "ties": ties,
        "losses_a": losses,
        "wilcoxon_stat": stat,
        "p_raw": p,
    })


pair = pd.DataFrame(pair_rows)


# Holm correction
pair["p_holm"] = np.nan

valid = pair["p_raw"].notna()

if valid.any():
    temp = (
        pair.loc[valid, ["p_raw"]]
        .sort_values("p_raw")
        .copy()
    )

    m = len(temp)

    adjusted = []
    running = 0.0

    for i, (_, row) in enumerate(
        temp.iterrows()
    ):
        val = min(
            1.0,
            (m - i) * row["p_raw"],
        )

        running = max(
            running,
            val,
        )

        adjusted.append(
            min(1.0, running)
        )

    temp["p_holm"] = adjusted

    pair.loc[
        temp.index,
        "p_holm"
    ] = temp["p_holm"]


pair = pair.sort_values(
    "p_raw",
    na_position="last",
)

pair.to_csv(
    OUT / "main_pairwise_wilcoxon.csv",
    index=False,
)


print()
print("PAIRWISE WILCOXON + HOLM")
print(
    pair.to_string(
        index=False,
        formatters={
            "mean_delta_a_minus_b": fmt4,
            "median_delta_a_minus_b": fmt4,
            "wilcoxon_stat": fmt3,
            "p_raw": lambda x: (
                "NA"
                if pd.isna(x)
                else f"{x:.6g}"
            ),
            "p_holm": lambda x: (
                "NA"
                if pd.isna(x)
                else f"{x:.6g}"
            ),
        }
    )
)


# =============================================================================
# FOCUSED ROLE-ADAPTER COMPARISONS
# =============================================================================

focus_rows = []

for baseline in [
    "frozen",
    "full_ft",
    "shared_lora",
    "shared_lora_widehead",
    "category_lora",
]:

    d = (
        cell_pivot["role_adapter"]
        - cell_pivot[baseline]
    )

    focus_rows.append({
        "comparison":
            f"role_adapter - {baseline}",
        "mean_delta": d.mean(),
        "median_delta": d.median(),
        "wins": int((d > 1e-12).sum()),
        "ties": int(
            (np.abs(d) <= 1e-12).sum()
        ),
        "losses": int((d < -1e-12).sum()),
    })

focus = pd.DataFrame(focus_rows)

focus.to_csv(
    OUT / "role_adapter_comparisons.csv",
    index=False,
)

print()
print("=" * 120)
print("ROLE ADAPTER — 15-CELL COMPARISON")
print("=" * 120)

print(
    focus.to_string(
        index=False,
        formatters={
            "mean_delta": fmt4,
            "median_delta": fmt4,
        }
    )
)


# =============================================================================
# COMPUTATIONAL EFFICIENCY
# =============================================================================

efficiency = (
    main
    .groupby(
        ["model_key", "method_key"],
        as_index=False,
    )
    .agg(
        runs=("run_name", "count"),
        parameters_total=(
            "parameters_total",
            "median",
        ),
        parameters_trainable=(
            "parameters_trainable",
            "median",
        ),
        trainable_pct=(
            "trainable_pct",
            "median",
        ),
        train_seconds_mean=(
            "train_seconds",
            "mean",
        ),
        train_seconds_std=(
            "train_seconds",
            "std",
        ),
        peak_vram_gb_mean=(
            "peak_gpu_memory_gb",
            "mean",
        ),
        peak_vram_gb_max=(
            "peak_gpu_memory_gb",
            "max",
        ),
    )
)

efficiency.to_csv(
    OUT / "main_efficiency.csv",
    index=False,
)

print()
print("=" * 120)
print("COMPUTATIONAL EFFICIENCY")
print("=" * 120)

print(
    efficiency.to_string(
        index=False,
        formatters={
            "trainable_pct": fmt3,
            "train_seconds_mean":
                lambda x: f"{x:.1f}",
            "train_seconds_std":
                lambda x: f"{x:.1f}",
            "peak_vram_gb_mean":
                lambda x: f"{x:.2f}",
            "peak_vram_gb_max":
                lambda x: f"{x:.2f}",
        }
    )
)


# =============================================================================
# RANK ABLATION
#
# rank phase contains r=2,4,16,32.
# Add r=8 from main / InLegalBERT / seed42.
# =============================================================================

rank_extra = df[
    df["phase"] == "rank"
].copy()

rank_r8 = main[
    (main["model_key"] == "inlegalbert")
    & (main["seed"] == 42)
    & (main["method_key"].isin(
        PEFT_RANK_METHODS
    ))
].copy()

rank_all = pd.concat(
    [rank_extra, rank_r8],
    ignore_index=True,
)

rank_all = rank_all[
    [
        "dataset_key",
        "method_key",
        "rank",
        "alpha",
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "parameters_trainable",
        "trainable_pct",
        "train_seconds",
        "peak_gpu_memory_gb",
        "run_name",
    ]
].sort_values(
    ["method_key", "dataset_key", "rank"]
)

expected_rank_rows = (
    len(DATASETS)
    * len(PEFT_RANK_METHODS)
    * 5
)

if len(rank_all) != expected_rank_rows:
    raise RuntimeError(
        "Expected "
        f"{expected_rank_rows} rank points, "
        f"got {len(rank_all)}"
    )

rank_all.to_csv(
    OUT / "rank_ablation_all_points.csv",
    index=False,
)

rank_summary = (
    rank_all
    .groupby(
        ["method_key", "rank"],
        as_index=False,
    )
    .agg(
        datasets=("dataset_key", "nunique"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        parameters_trainable=(
            "parameters_trainable",
            "median",
        ),
        train_seconds_mean=(
            "train_seconds",
            "mean",
        ),
        peak_vram_gb_mean=(
            "peak_gpu_memory_gb",
            "mean",
        ),
    )
)

rank_summary.to_csv(
    OUT / "rank_ablation_summary.csv",
    index=False,
)


print()
print("=" * 120)
print("RANK ABLATION — MEAN TEST MACRO-F1 ACROSS 5 DATASETS")
print("InLegalBERT, seed 42")
print("=" * 120)

rp = rank_summary.pivot(
    index="rank",
    columns="method_key",
    values="macro_f1_mean",
)

print(
    rp.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# Best rank for each method × dataset
best_rank_idx = (
    rank_all
    .groupby(
        ["method_key", "dataset_key"]
    )["macro_f1"]
    .idxmax()
)

best_rank = (
    rank_all
    .loc[best_rank_idx]
    .sort_values(
        ["method_key", "dataset_key"]
    )
)

best_rank.to_csv(
    OUT / "rank_ablation_best_per_dataset.csv",
    index=False,
)


# =============================================================================
# MECHANISTIC ABLATION
# =============================================================================

mech = df[
    df["phase"] == "mechanistic"
].copy()

if len(mech) != 5:
    raise RuntimeError(
        f"Expected 5 mechanistic runs, got {len(mech)}"
    )

baseline_seed42 = main[
    (main["model_key"] == "inlegalbert")
    & (main["seed"] == 42)
    & (
        main["method_key"].isin(
            [
                "shared_lora",
                "shared_lora_widehead",
                "category_lora",
                "role_adapter",
                "full_ft",
            ]
        )
    )
][
    [
        "dataset_key",
        "method_key",
        "macro_f1",
        "parameters_trainable",
        "train_seconds",
        "peak_gpu_memory_gb",
    ]
]

mech_rows = []

for _, m in mech.iterrows():

    ds = m["dataset_key"]

    row = {
        "dataset": ds,

        "true_category_f1":
            m["macro_f1"],

        "true_category_params":
            m["parameters_trainable"],

        "true_category_seconds":
            m["train_seconds"],

        "true_category_vram_gb":
            m["peak_gpu_memory_gb"],
    }

    sub = baseline_seed42[
        baseline_seed42["dataset_key"] == ds
    ]

    for _, b in sub.iterrows():
        method = b["method_key"]

        row[f"{method}_f1"] = (
            b["macro_f1"]
        )

        row[
            f"delta_true_minus_{method}"
        ] = (
            m["macro_f1"]
            - b["macro_f1"]
        )

    mech_rows.append(row)

mech_cmp = pd.DataFrame(mech_rows)

mech_cmp.to_csv(
    OUT / "mechanistic_comparison.csv",
    index=False,
)

print()
print("=" * 120)
print("TRUE CATEGORY LORA — MECHANISTIC COMPARISON")
print("=" * 120)

cols = [
    "dataset",
    "true_category_f1",
    "role_adapter_f1",
    "delta_true_minus_role_adapter",
    "category_lora_f1",
    "delta_true_minus_category_lora",
    "shared_lora_f1",
    "delta_true_minus_shared_lora",
]

present = [
    c for c in cols
    if c in mech_cmp.columns
]

print(
    mech_cmp[present].to_string(
        index=False,
        formatters={
            c: fmt4
            for c in present
            if c != "dataset"
        }
    )
)


# =============================================================================
# PER-CLASS RESULTS
# =============================================================================

pc = pd.DataFrame(per_class_rows)

if len(pc):
    pc.to_csv(
        OUT / "per_class_all_runs.csv",
        index=False,
    )

    pc_main = pc[
        pc["phase"] == "main"
    ].copy()

    pc_summary = (
        pc_main
        .groupby(
            [
                "model_key",
                "dataset_key",
                "method_key",
                "label",
            ],
            as_index=False,
        )
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=(
                "precision",
                "mean",
            ),
            recall_mean=(
                "recall",
                "mean",
            ),
            support_mean=(
                "support",
                "mean",
            ),
        )
    )

    pc_summary.to_csv(
        OUT / "per_class_main_mean_std.csv",
        index=False,
    )

    role_pc = (
        pc_main[
            pc_main["method_key"]
            == "role_adapter"
        ]
        .groupby(
            "label",
            as_index=False,
        )
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .sort_values(
            "f1_mean",
            ascending=False,
        )
    )

    print()
    print("=" * 120)
    print("ROLE ADAPTER — PER-CLASS F1")
    print("averaged over all main runs")
    print("=" * 120)

    print(
        role_pc.to_string(
            index=False,
            formatters={
                "f1_mean": fmt4,
                "f1_std": fmt4,
            }
        )
    )


# =============================================================================
# COMPUTE TOTALS
# =============================================================================

compute_by_phase = (
    df
    .groupby("phase", as_index=False)
    .agg(
        runs=("run_name", "count"),
        train_seconds=(
            "train_seconds",
            "sum",
        ),
        mean_peak_vram_gb=(
            "peak_gpu_memory_gb",
            "mean",
        ),
        max_peak_vram_gb=(
            "peak_gpu_memory_gb",
            "max",
        ),
    )
)

compute_by_phase[
    "train_hours"
] = (
    compute_by_phase["train_seconds"]
    / 3600
)

compute_by_phase.to_csv(
    OUT / "compute_by_phase.csv",
    index=False,
)

print()
print("=" * 120)
print("RECORDED COMPUTE")
print("=" * 120)

print(
    compute_by_phase.to_string(
        index=False,
        formatters={
            "train_seconds":
                lambda x: f"{x:.1f}",
            "train_hours":
                lambda x: f"{x:.2f}",
            "mean_peak_vram_gb":
                lambda x: f"{x:.2f}",
            "max_peak_vram_gb":
                lambda x: f"{x:.2f}",
        }
    )
)

total_sec = df["train_seconds"].sum()

print()
print(
    f"Total summed training time: "
    f"{total_sec/3600:.2f} GPU-process hours"
)


# =============================================================================
# SAVE COMPACT REPORT INFO
# =============================================================================

stats_meta = {
    "total_runs": int(len(df)),
    "main_runs": int(
        (df["phase"] == "main").sum()
    ),
    "rank_runs": int(
        (df["phase"] == "rank").sum()
    ),
    "mechanistic_runs": int(
        (df["phase"] == "mechanistic").sum()
    ),
    "main_cells": int(len(main_cells)),
    "main_seeds_per_cell": 3,
    "friedman_stat": (
        None
        if pd.isna(friedman_stat)
        else friedman_stat
    ),
    "friedman_p": (
        None
        if pd.isna(friedman_p)
        else friedman_p
    ),
    "total_train_seconds": float(
        total_sec
    ),
    "total_train_hours": float(
        total_sec / 3600
    ),
}

with open(
    OUT / "analysis_metadata.json",
    "w",
) as f:
    json.dump(
        stats_meta,
        f,
        indent=2,
    )


print()
print("=" * 120)
print("SAVED")
print("=" * 120)

for fn in sorted(OUT.glob("*")):
    if fn.is_file():
        print(fn.relative_to(ROOT))

print()
print("FINAL ANALYSIS COMPLETE.")
