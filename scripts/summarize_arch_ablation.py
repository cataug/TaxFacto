from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path.home() / "TaxFacto"
R = ROOT / "results/stage1"
OUT = ROOT / "results"

OUT.mkdir(parents=True, exist_ok=True)

MAIN_DATASETS = [
    "legaleval",
    "marro_india",
    "marro_uk",
]

MAIN_METHODS = [
    "frozen",
    "full_ft",
    "shared_lora",
    "shared_lora_widehead",
    "category_lora",
    "role_adapter",
]

ALL_EXPECTED = [
    f"{dataset}__{method}__seed42"
    for dataset in MAIN_DATASETS
    for method in MAIN_METHODS
]

ALL_EXPECTED.append(
    "legaleval__true_category_lora__seed42"
)


# ============================================================
# Load expected runs
# ============================================================

rows = []
missing = []

for run in ALL_EXPECTED:
    fn = R / run / "smoke_dev_summary.json"

    if not fn.exists():
        missing.append(run)
        continue

    with open(fn) as f:
        x = json.load(f)

    rows.append({
        "run_name": x.get("run_name", run),
        "dataset": x["dataset"],
        "method": x["method"],
        "seed": x["seed"],
        "epochs": x["epochs"],

        "best_dev_macro_f1":
            x.get(
                "best_dev_macro_f1",
                x.get("eval_macro_f1"),
            ),

        "best_dev_epoch":
            x.get(
                "best_dev_epoch",
                x.get("epoch"),
            ),

        "final_dev_macro_f1":
            x.get("eval_macro_f1"),

        "dev_weighted_f1":
            x.get("eval_weighted_f1"),

        "dev_accuracy":
            x.get("eval_accuracy"),

        "dev_loss":
            x.get("eval_loss"),

        "parameters_total":
            x.get("parameters_total"),

        "parameters_trainable":
            x.get("parameters_trainable"),

        "trainable_pct":
            x.get("trainable_pct"),

        "train_seconds":
            x.get("train_seconds"),

        "peak_gpu_memory_gb":
            x.get("peak_gpu_memory_gb"),
    })


df = pd.DataFrame(rows)

print("=" * 110)
print("ARCHITECTURE ABLATION INVENTORY")
print("=" * 110)

print("expected:", len(ALL_EXPECTED))
print("found   :", len(df))
print("missing :", len(missing))

if missing:
    print()
    print("MISSING RUNS:")
    for x in missing:
        print("  ", x)

print()


# ============================================================
# Full table
# ============================================================

method_order = {
    "frozen": 0,
    "full_ft": 1,
    "shared_lora": 2,
    "shared_lora_widehead": 3,
    "category_lora": 4,
    "role_adapter": 5,
    "true_category_lora": 6,
}

dataset_order = {
    "legaleval": 0,
    "marro_india": 1,
    "marro_uk": 2,
}

df["_dataset_order"] = (
    df["dataset"].map(dataset_order).fillna(999)
)

df["_method_order"] = (
    df["method"].map(method_order).fillna(999)
)

df = df.sort_values(
    ["_dataset_order", "_method_order"]
).drop(
    columns=[
        "_dataset_order",
        "_method_order",
    ]
)

show_cols = [
    "dataset",
    "method",
    "best_dev_macro_f1",
    "best_dev_epoch",
    "final_dev_macro_f1",
    "dev_weighted_f1",
    "dev_accuracy",
    "parameters_trainable",
    "trainable_pct",
    "train_seconds",
    "peak_gpu_memory_gb",
]

print("=" * 110)
print("FULL RESULTS")
print("=" * 110)

print(
    df[show_cols].to_string(
        index=False,
        formatters={
            "best_dev_macro_f1":
                lambda x: f"{x:.4f}",
            "best_dev_epoch":
                lambda x: f"{x:.1f}",
            "final_dev_macro_f1":
                lambda x: f"{x:.4f}",
            "dev_weighted_f1":
                lambda x: f"{x:.4f}",
            "dev_accuracy":
                lambda x: f"{x:.4f}",
            "trainable_pct":
                lambda x: f"{x:.3f}",
            "train_seconds":
                lambda x: f"{x:.1f}",
            "peak_gpu_memory_gb":
                lambda x: f"{x:.2f}",
        },
    )
)


# ============================================================
# Macro-F1 pivot
# ============================================================

main = df[
    df["dataset"].isin(MAIN_DATASETS)
    & df["method"].isin(MAIN_METHODS)
].copy()

pivot = main.pivot(
    index="method",
    columns="dataset",
    values="best_dev_macro_f1",
)

pivot = pivot.reindex(MAIN_METHODS)
pivot = pivot.reindex(columns=MAIN_DATASETS)

pivot["mean"] = pivot.mean(
    axis=1,
    skipna=True,
)

pivot["std_across_datasets"] = pivot[
    MAIN_DATASETS
].std(
    axis=1,
    ddof=0,
)

print()
print("=" * 110)
print("BEST DEV MACRO-F1")
print("=" * 110)

print(
    pivot.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# Ranking by average across datasets
# ============================================================

avg = (
    main
    .groupby("method", as_index=False)
    .agg(
        mean_macro_f1=(
            "best_dev_macro_f1",
            "mean",
        ),
        min_macro_f1=(
            "best_dev_macro_f1",
            "min",
        ),
        max_macro_f1=(
            "best_dev_macro_f1",
            "max",
        ),
        mean_train_seconds=(
            "train_seconds",
            "mean",
        ),
        mean_vram_gb=(
            "peak_gpu_memory_gb",
            "mean",
        ),
        trainable_params=(
            "parameters_trainable",
            "first",
        ),
        trainable_pct=(
            "trainable_pct",
            "first",
        ),
    )
    .sort_values(
        "mean_macro_f1",
        ascending=False,
    )
)

print()
print("=" * 110)
print("MEAN ACROSS 3 DATASETS")
print("=" * 110)

print(
    avg.to_string(
        index=False,
        formatters={
            "mean_macro_f1":
                lambda x: f"{x:.4f}",
            "min_macro_f1":
                lambda x: f"{x:.4f}",
            "max_macro_f1":
                lambda x: f"{x:.4f}",
            "mean_train_seconds":
                lambda x: f"{x:.1f}",
            "mean_vram_gb":
                lambda x: f"{x:.2f}",
            "trainable_pct":
                lambda x: f"{x:.3f}",
        },
    )
)


# ============================================================
# Role-adapter deltas
# ============================================================

print()
print("=" * 110)
print("ROLE ADAPTER DELTAS")
print("=" * 110)

for dataset in MAIN_DATASETS:
    d = main[
        main["dataset"] == dataset
    ].set_index("method")

    if "role_adapter" not in d.index:
        continue

    role = d.loc[
        "role_adapter",
        "best_dev_macro_f1",
    ]

    print()
    print(dataset)

    for baseline in [
        "frozen",
        "full_ft",
        "shared_lora",
        "shared_lora_widehead",
        "category_lora",
    ]:
        if baseline not in d.index:
            continue

        b = d.loc[
            baseline,
            "best_dev_macro_f1",
        ]

        print(
            f"  role_adapter - "
            f"{baseline:22s} "
            f"= {role-b:+.4f}"
        )


# ============================================================
# Best method per dataset
# ============================================================

print()
print("=" * 110)
print("TOP METHOD PER DATASET")
print("=" * 110)

for dataset in MAIN_DATASETS:
    d = (
        main[
            main["dataset"] == dataset
        ]
        .sort_values(
            "best_dev_macro_f1",
            ascending=False,
        )
    )

    if len(d) == 0:
        continue

    r = d.iloc[0]

    print(
        f"{dataset:14s} "
        f"{r['method']:24s} "
        f"{r['best_dev_macro_f1']:.4f} "
        f"(epoch {r['best_dev_epoch']:.1f})"
    )


# ============================================================
# Expensive true-category ablation
# ============================================================

tc = df[
    df["method"] == "true_category_lora"
]

if len(tc):
    print()
    print("=" * 110)
    print("TRUE CATEGORY LORA — MECHANISTIC ABLATION")
    print("=" * 110)

    print(
        tc[[
            "dataset",
            "best_dev_macro_f1",
            "best_dev_epoch",
            "parameters_trainable",
            "trainable_pct",
            "train_seconds",
            "peak_gpu_memory_gb",
        ]].to_string(
            index=False,
            formatters={
                "best_dev_macro_f1":
                    lambda x: f"{x:.4f}",
                "best_dev_epoch":
                    lambda x: f"{x:.1f}",
                "trainable_pct":
                    lambda x: f"{x:.3f}",
                "train_seconds":
                    lambda x: f"{x:.1f}",
                "peak_gpu_memory_gb":
                    lambda x: f"{x:.2f}",
            },
        )
    )


# ============================================================
# Total compute
# ============================================================

total_seconds = df["train_seconds"].sum()

print()
print("=" * 110)
print("COMPUTE")
print("=" * 110)

print(
    f"Total training time recorded: "
    f"{total_seconds:.1f} s "
    f"= {total_seconds/60:.1f} min "
    f"= {total_seconds/3600:.2f} h"
)


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUT / "architecture_ablation_summary.csv",
    index=False,
)

pivot.to_csv(
    OUT / "architecture_ablation_macro_f1.csv"
)

avg.to_csv(
    OUT / "architecture_ablation_method_means.csv",
    index=False,
)

print()
print("=" * 110)
print("SAVED")
print("=" * 110)
print(
    OUT / "architecture_ablation_summary.csv"
)
print(
    OUT / "architecture_ablation_macro_f1.csv"
)
print(
    OUT / "architecture_ablation_method_means.csv"
)
