from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path.home() / "TaxFacto"

MAIN = ROOT / "results/final_analysis"
ADD = ROOT / "results/additional/analysis"
BOOT = ROOT / "results/additional/bootstrap"
CORE8 = ROOT / "results/additional/fair_core8"

OUT = ROOT / "results/paper_figures_v3"
OUT.mkdir(parents=True, exist_ok=True)


# =====================================================================
# GLOBAL STYLE
# =====================================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",

    # deliberately large: figures remain readable after reduction
    "font.size": 18,
    "axes.titlesize": 23,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,

    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,

    "figure.facecolor": "white",
    "savefig.facecolor": "white",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 260,
})


# =====================================================================
# LABELS
# =====================================================================

METHODS6 = [
    "full_ft",
    "role_adapter",
    "category_lora",
    "shared_lora_widehead",
    "shared_lora",
    "frozen",
]

METHODS4 = [
    "full_ft",
    "role_adapter",
    "category_lora",
    "shared_lora",
]

METHODS3 = [
    "full_ft",
    "role_adapter",
    "shared_lora",
]

METHOD_LABEL = {
    "full_ft": "Full FT",
    "role_adapter": "Role Adapter",
    "category_lora": "Category LoRA",
    "shared_lora_widehead": "Wide-head",
    "shared_lora": "Shared LoRA",
    "frozen": "Frozen",
    "true_category_lora": "True category LoRA",
}

DATASETS = [
    "legaleval",
    "marro_india",
    "marro_uk",
    "iltur_cl",
    "iltur_it",
]

DATASET_LABEL = {
    "legaleval": "LegalEval",
    "marro_india": "MARRO-IN",
    "marro_uk": "MARRO-UK",
    "iltur_cl": "IL-TUR CL",
    "iltur_it": "IL-TUR IT",
}

BACKBONE_LABEL = {
    "inlegalbert": "InLegalBERT",
    "legalbert": "LegalBERT",
    "deberta": "DeBERTa-v3",
}


# =====================================================================
# VIRIDIS/CIVIDIS-INSPIRED COLORS
# =====================================================================

vir = plt.cm.viridis
civ = plt.cm.cividis

METHOD_COLOR = {
    "full_ft": vir(0.94),
    "role_adapter": vir(0.64),
    "category_lora": vir(0.48),
    "shared_lora_widehead": civ(0.42),
    "shared_lora": vir(0.18),
    "frozen": (0.70, 0.72, 0.76, 1.0),
    "true_category_lora": civ(0.78),
}

REGIME_COLOR = {
    "Within-domain": vir(0.18),
    "Transfer": vir(0.58),
    "Native": vir(0.90),
}

DATASET_COLOR = {
    ds: vir(v)
    for ds, v in zip(
        DATASETS,
        np.linspace(0.12, 0.90, len(DATASETS))
    )
}


# =====================================================================
# CUSTOM COLORMAPS
# =====================================================================

CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "viridis_diverging",
    [
        "#482878",
        "#3B528B",
        "#EDEDE8",
        "#35B779",
        "#FDE725",
    ],
)

CMAP_POSITIVE = LinearSegmentedColormap.from_list(
    "viridis_soft",
    [
        "#EEEAF5",
        "#BDD7E7",
        "#5EC9A8",
        "#A6DB36",
        "#FDE725",
    ],
)


# =====================================================================
# HELPERS
# =====================================================================

def method_name(x):
    return METHOD_LABEL.get(x, x)


def dataset_name(x):
    return DATASET_LABEL.get(x, x)


def gradient_background(
    ax,
    top="#EDF7FF",
    bottom="#FFF8ED",
    alpha=0.90,
):
    arr = np.linspace(0, 1, 300).reshape(-1, 1)

    cmap = LinearSegmentedColormap.from_list(
        "background",
        [top, bottom],
    )

    ax.imshow(
        arr,
        extent=(0, 1, 0, 1),
        transform=ax.transAxes,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        alpha=alpha,
        zorder=-100,
    )


def style_ax(
    ax,
    grid_axis="y",
):
    ax.set_facecolor("#FBFAF7")

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)

    if grid_axis is not None:
        ax.grid(
            True,
            axis=grid_axis,
            color="black",
            alpha=0.12,
            linewidth=0.7,
            zorder=-10,
        )

    ax.set_axisbelow(True)


def bbox(
    ax,
    x,
    y,
    text,
    fontsize=11,
    ha="center",
    va="center",
    face="#FFFDF7",
):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor=face,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.92,
        ),
        zorder=50,
    )


def save(fig, stem):
    fig.savefig(
        OUT / f"{stem}.png",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def bootstrap_mean_ci(
    values,
    B=10000,
    seed=20261006,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    rng = np.random.default_rng(
        seed
    )

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    samples = rng.choice(
        values,
        size=(B, len(values)),
        replace=True,
    )

    means = samples.mean(
        axis=1
    )

    return (
        float(values.mean()),
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def pairwise_matrix(
    pivot,
    methods,
):
    methods = [
        m
        for m in methods
        if m in pivot.columns
    ]

    n = len(methods)

    delta = np.full(
        (n, n),
        np.nan,
    )

    wins = np.empty(
        (n, n),
        dtype=object,
    )

    for i, a in enumerate(methods):
        for j, b in enumerate(methods):

            if i == j:
                delta[i, j] = 0.0
                wins[i, j] = "—"
                continue

            d = (
                pivot[a]
                - pivot[b]
            ).dropna()

            delta[i, j] = d.mean()

            wins[i, j] = (
                f"{int((d > 0).sum())}"
                f"/{len(d)}"
            )

    return methods, delta, wins


# =====================================================================
# LOAD DATA
# =====================================================================

main_cells = pd.read_csv(
    MAIN / "main_cell_mean_std.csv"
)

main_overall = pd.read_csv(
    MAIN / "main_method_overall.csv"
)

main_eff = pd.read_csv(
    MAIN / "main_efficiency.csv"
)

rank_points = pd.read_csv(
    MAIN / "rank_ablation_all_points.csv"
)

rank_summary = pd.read_csv(
    MAIN / "rank_ablation_summary.csv"
)

mech = pd.read_csv(
    MAIN / "mechanistic_comparison.csv"
)

transfer = pd.read_csv(
    ADD / "transfer_summary.csv"
)

native = pd.read_csv(
    ADD / "native_summary.csv"
)

role_shared_transfer = pd.read_csv(
    ADD / "transfer_role_vs_shared.csv"
)

boot = pd.read_csv(
    BOOT / "hierarchical_bootstrap_global.csv"
)

core8 = pd.read_csv(
    CORE8 / "transfer_core8_pivot.csv"
)


# =====================================================================
# PREPARE WITHIN PIVOTS
# =====================================================================

within4 = (
    main_cells[
        main_cells["method_key"].isin(
            METHODS4
        )
    ]
    .pivot_table(
        index=[
            "model_key",
            "dataset_key",
        ],
        columns="method_key",
        values="macro_f1_mean",
    )
    .dropna()
)

within3 = within4[
    [
        "full_ft",
        "role_adapter",
        "shared_lora",
    ]
].copy()


# =====================================================================
# FAIR CORE8 PIVOT
# =====================================================================

core8 = core8.set_index(
    [
        "source",
        "target",
    ]
)


# =====================================================================
# NATIVE PIVOT
# =====================================================================

native_pivot = (
    native
    .pivot_table(
        index="dataset",
        columns="method",
        values="macro_f1_mean",
    )
)


# =====================================================================
# FIGURE 1
# ACCURACY–EFFICIENCY PARETO FRONTIER
# NO INSET
# =====================================================================

overall = (
    main_overall[
        [
            "method_key",
            "macro_f1_mean",
            "macro_f1_std_across_cells",
        ]
    ]
    .copy()
)

eff_avg = (
    main_eff
    .groupby(
        "method_key",
        as_index=False,
    )
    .agg(
        trainable_pct=(
            "trainable_pct",
            "mean",
        ),
        train_seconds=(
            "train_seconds_mean",
            "mean",
        ),
        peak_vram=(
            "peak_vram_gb_mean",
            "mean",
        ),
    )
)

pareto = overall.merge(
    eff_avg,
    on="method_key",
    how="left",
)

fig, ax = plt.subplots(
    figsize=(14.5, 8.7),
    layout="constrained",
)

gradient_background(ax)
style_ax(ax, "both")

ax.set_xscale("log")

ax.set_xlabel(
    "Trainable parameters (%) — logarithmic scale"
)

ax.set_ylabel(
    "Mean within-domain Macro-F1"
)

ax.set_title(
    "Accuracy–efficiency landscape"
)

# Pareto frontier
points = []

for _, r in pareto.iterrows():
    points.append(
        (
            r["trainable_pct"],
            r["macro_f1_mean"],
            r["method_key"],
        )
    )

front = []

for x, y, m in points:

    dominated = False

    for x2, y2, m2 in points:

        if m2 == m:
            continue

        if (
            x2 <= x
            and y2 >= y
            and (
                x2 < x
                or y2 > y
            )
        ):
            dominated = True
            break

    if not dominated:
        front.append(
            (x, y, m)
        )

front = sorted(front)

ax.plot(
    [x for x, y, m in front],
    [y for x, y, m in front],
    linestyle="--",
    linewidth=2.2,
    color="black",
    alpha=0.45,
    zorder=2,
)

# shaded PEFT region
ax.axvspan(
    0.001,
    1.0,
    color=vir(0.45),
    alpha=0.045,
    zorder=-20,
)

offsets = {
    "frozen": (14, -12),
    "shared_lora": (14, -22),
    "shared_lora_widehead": (14, -2),
    "category_lora": (14, 17),
    "role_adapter": (14, 38),
    "full_ft": (-18, 2),
}

for _, r in pareto.iterrows():

    m = r["method_key"]

    size = (
        180
        + 1.2
        * r["train_seconds"]
    )

    ax.errorbar(
        r["trainable_pct"],
        r["macro_f1_mean"],
        yerr=r[
            "macro_f1_std_across_cells"
        ],
        color="black",
        linewidth=1.0,
        capsize=4,
        zorder=3,
    )

    ax.scatter(
        r["trainable_pct"],
        r["macro_f1_mean"],
        s=size,
        color=METHOD_COLOR[m],
        edgecolor="black",
        linewidth=0.9,
        alpha=0.82,
        zorder=5,
    )

    dx, dy = offsets[m]

    ax.annotate(
        (
            f"{method_name(m)}\n"
            f"F1 {r['macro_f1_mean']:.3f} · "
            f"{r['trainable_pct']:.3f}%"
        ),
        xy=(
            r["trainable_pct"],
            r["macro_f1_mean"],
        ),
        xytext=(
            dx,
            dy,
        ),
        textcoords="offset points",
        ha=(
            "right"
            if dx < 0
            else "left"
        ),
        va="center",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="#FFFDF7",
            edgecolor="black",
            linewidth=0.75,
            alpha=0.94,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color="black",
            linewidth=0.7,
        ),
        zorder=20,
    )

ax.text(
    0.025,
    0.035,
    "Dashed line: empirical Pareto frontier   ·   marker area: mean training time",
    transform=ax.transAxes,
    fontsize=13,
    va="bottom",
    ha="left",
)

save(
    fig,
    "01_accuracy_efficiency_pareto",
)


# =====================================================================
# FIGURE 2
# DOMINANCE TRIPTYCH
# ONE COMMON AXIS LABEL ONLY
# =====================================================================

m1, d1, w1 = pairwise_matrix(
    within4,
    METHODS4,
)

m2, d2, w2 = pairwise_matrix(
    core8,
    METHODS4,
)

native3 = native_pivot[
    [
        "full_ft",
        "role_adapter",
        "shared_lora",
    ]
].dropna()

m3, d3, w3 = pairwise_matrix(
    native3,
    METHODS3,
)

fig = plt.figure(
    figsize=(22.5, 7.4),
    layout="constrained",
)

gs = fig.add_gridspec(
    1,
    4,
    width_ratios=[
        1,
        1,
        0.82,
        0.045,
    ],
)

panels = [
    (
        fig.add_subplot(
            gs[0, 0]
        ),
        m1,
        d1,
        w1,
        "Within-domain",
        "15 dataset × backbone cells",
    ),
    (
        fig.add_subplot(
            gs[0, 1]
        ),
        m2,
        d2,
        w2,
        "Transfer",
        "8 fair transfer directions",
    ),
    (
        fig.add_subplot(
            gs[0, 2]
        ),
        m3,
        d3,
        w3,
        "Native taxonomy",
        "3 native datasets",
    ),
]

max_abs = max(
    np.nanmax(
        np.abs(d1)
    ),
    np.nanmax(
        np.abs(d2)
    ),
    np.nanmax(
        np.abs(d3)
    ),
)

norm = Normalize(
    -max_abs,
    max_abs,
)

for ax, methods, mat, wins, ttl, subtitle in panels:

    gradient_background(ax)
    style_ax(
        ax,
        None,
    )

    ax.imshow(
        mat,
        cmap=CMAP_DIVERGING,
        norm=norm,
        alpha=0.92,
    )

    n = len(methods)

    ax.set_xticks(
        range(n)
    )

    ax.set_yticks(
        range(n)
    )

    ax.set_xticklabels(
        [
            method_name(x)
            for x in methods
        ],
        rotation=26,
        ha="right",
    )

    ax.set_yticklabels(
        [
            method_name(x)
            for x in methods
        ]
    )

    ax.set_title(
        f"{ttl}\n{subtitle}",
        fontsize=20,
        pad=12,
    )

    ax.set_xticks(
        np.arange(
            -0.5,
            n,
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            n,
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="black",
        linewidth=0.75,
        alpha=0.48,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for i in range(n):
        for j in range(n):

            if i == j:
                bbox(
                    ax,
                    j,
                    i,
                    "—",
                    fontsize=13,
                    face="#F3F3EF",
                )
                continue

            bbox(
                ax,
                j,
                i,
                (
                    f"{mat[i,j]:+.3f}\n"
                    f"{wins[i,j]} wins"
                ),
                fontsize=10,
            )

cax = fig.add_subplot(
    gs[0, 3]
)

sm = mpl.cm.ScalarMappable(
    cmap=CMAP_DIVERGING,
    norm=norm,
)

cb = fig.colorbar(
    sm,
    cax=cax,
)

cb.set_label(
    "Row method − column method",
    fontsize=16,
)

fig.suptitle(
    "Pairwise dominance across evaluation regimes",
    fontsize=26,
)

fig.supxlabel(
    "Column method",
    fontsize=18,
)

fig.supylabel(
    "Row method",
    fontsize=18,
)

fig.text(
    0.5,
    -0.015,
    "Each cell reports mean Δ Macro-F1 and the number of atomic cells won by the row method.",
    ha="center",
    va="top",
    fontsize=13,
)

save(
    fig,
    "02_pairwise_dominance",
)


# =====================================================================
# FIGURE 3
# TRANSFER ATLAS
# COLOR = ROLE - SHARED
# MINI BAR = RETENTION
# DIAGONAL = WITHIN-DOMAIN
# =====================================================================

role_matrix = pd.read_csv(
    ADD
    / "transfer_matrix_role_adapter.csv",
    index_col=0,
)

shared_matrix = pd.read_csv(
    ADD
    / "transfer_matrix_shared_lora.csv",
    index_col=0,
)

delta_matrix = (
    role_matrix
    - shared_matrix
)

retention = (
    transfer[
        transfer["method"]
        == "role_adapter"
    ]
    .pivot(
        index="source",
        columns="target",
        values="retention",
    )
)

retention = retention.reindex(
    index=DATASETS,
    columns=DATASETS,
)

for d in DATASETS:
    retention.loc[d, d] = 1.0

role_matrix = role_matrix.loc[
    DATASETS,
    DATASETS,
]

delta_matrix = delta_matrix.loc[
    DATASETS,
    DATASETS,
]

fig, ax = plt.subplots(
    figsize=(12.0, 10.2),
    layout="constrained",
)

gradient_background(ax)
style_ax(
    ax,
    None,
)

n = len(DATASETS)

ax.set_xlim(
    -0.5,
    n - 0.5,
)

ax.set_ylim(
    n - 0.5,
    -0.5,
)

ax.set_xticks(
    range(n)
)

ax.set_yticks(
    range(n)
)

ax.set_xticklabels(
    [
        dataset_name(d)
        for d in DATASETS
    ],
    rotation=25,
    ha="right",
)

ax.set_yticklabels(
    [
        dataset_name(d)
        for d in DATASETS
    ],
)

max_delta = np.nanmax(
    np.abs(
        delta_matrix.to_numpy()
    )
)

norm = Normalize(
    -max_delta,
    max_delta,
)

for i, src in enumerate(
    DATASETS
):

    for j, tgt in enumerate(
        DATASETS
    ):

        d = float(
            delta_matrix.loc[
                src,
                tgt,
            ]
        )

        f1 = float(
            role_matrix.loc[
                src,
                tgt,
            ]
        )

        r = float(
            retention.loc[
                src,
                tgt,
            ]
        )

        if src == tgt:
            face = "#303030"
        else:
            face = (
                CMAP_DIVERGING(
                    norm(d)
                )
            )

        tile = FancyBboxPatch(
            (
                j - 0.45,
                i - 0.45,
            ),
            0.90,
            0.90,
            boxstyle=(
                "round,pad=0.015,"
                "rounding_size=0.07"
            ),
            facecolor=face,
            edgecolor="black",
            linewidth=0.85,
            alpha=0.91,
            zorder=2,
        )

        ax.add_patch(tile)

        if src == tgt:

            ax.text(
                j,
                i - 0.10,
                "within",
                color="white",
                fontsize=11,
                ha="center",
                va="center",
                fontweight="bold",
            )

            ax.text(
                j,
                i + 0.13,
                f"F1 {f1:.3f}",
                color="white",
                fontsize=12,
                ha="center",
                va="center",
            )

            continue

        bbox(
            ax,
            j,
            i - 0.13,
            f"Δ {d:+.3f}",
            fontsize=11,
        )

        # retention mini-bar
        bar_left = (
            j - 0.31
        )

        bar_y = (
            i + 0.22
        )

        max_width = 0.62

        ax.plot(
            [
                bar_left,
                bar_left
                + max_width,
            ],
            [
                bar_y,
                bar_y,
            ],
            color="white",
            linewidth=5.0,
            solid_capstyle="round",
            alpha=0.65,
            zorder=5,
        )

        ax.plot(
            [
                bar_left,
                bar_left
                + max_width
                * min(
                    max(
                        r,
                        0,
                    ),
                    1.15,
                )
                / 1.15,
            ],
            [
                bar_y,
                bar_y,
            ],
            color="black",
            linewidth=3.2,
            solid_capstyle="round",
            alpha=0.75,
            zorder=6,
        )

        ax.text(
            j,
            i + 0.34,
            f"ret. {r:.2f}",
            ha="center",
            va="center",
            fontsize=9.5,
            color="black",
            zorder=10,
        )

ax.set_xlabel(
    "Target domain"
)

ax.set_ylabel(
    "Source domain"
)

ax.set_title(
    "Cross-domain transfer atlas",
    pad=15,
)

sm = mpl.cm.ScalarMappable(
    cmap=CMAP_DIVERGING,
    norm=norm,
)

cb = fig.colorbar(
    sm,
    ax=ax,
    shrink=0.80,
    pad=0.02,
)

cb.set_label(
    "Role Adapter − Shared LoRA"
)

fig.text(
    0.5,
    0.005,
    "Cell color: Δ Macro-F1 · black mini-bar: Role Adapter transfer retention · diagonal: within-domain Role Adapter performance.",
    ha="center",
    va="bottom",
    fontsize=12,
)

save(
    fig,
    "03_transfer_atlas",
)


# =====================================================================
# FIGURE 4
# CLEAN EFFECT FOREST
# =====================================================================

rows = []

# within bootstrap
for other in [
    "shared_lora",
    "shared_lora_widehead",
    "category_lora",
    "full_ft",
]:

    q = boot[
        (
            boot["method_a"]
            == other
        )
        &
        (
            boot["method_b"]
            == "role_adapter"
        )
    ]

    if len(q) != 1:
        continue

    r = q.iloc[0]

    rows.append({
        "regime":
            "Within-domain",

        "comparison":
            method_name(other),

        "delta":
            -r[
                "point_mean_delta"
            ],

        "low":
            -r["ci95_high"],

        "high":
            -r["ci95_low"],

        "extra":
            "document bootstrap",
    })


# fair transfer bootstrap
for other in [
    "shared_lora",
    "category_lora",
    "full_ft",
]:

    d = (
        core8["role_adapter"]
        - core8[other]
    ).to_numpy()

    mean, lo, hi = (
        bootstrap_mean_ci(
            d
        )
    )

    rows.append({
        "regime":
            "Transfer",

        "comparison":
            method_name(other),

        "delta":
            mean,

        "low":
            lo,

        "high":
            hi,

        "extra":
            f"{int((d > 0).sum())}/8 wins",
    })


# native
for other in [
    "shared_lora",
    "full_ft",
]:

    d = (
        native_pivot[
            "role_adapter"
        ]
        - native_pivot[
            other
        ]
    ).dropna().to_numpy()

    mean, lo, hi = (
        bootstrap_mean_ci(
            d
        )
    )

    rows.append({
        "regime":
            "Native",

        "comparison":
            method_name(other),

        "delta":
            mean,

        "low":
            lo,

        "high":
            hi,

        "extra":
            f"{int((d > 0).sum())}/3 wins",
    })


forest = pd.DataFrame(
    rows
)

regime_order = [
    "Within-domain",
    "Transfer",
    "Native",
]

# generate separated y positions
positions = []
labels = []

y = 0

for regime in regime_order:

    sub = forest[
        forest["regime"]
        == regime
    ]

    for idx in sub.index:
        positions.append(
            (
                idx,
                y,
            )
        )
        y += 1

    y += 0.8

pos = dict(
    positions
)

forest["y"] = forest.index.map(
    pos
)

fig, ax = plt.subplots(
    figsize=(15.0, 9.5),
)

fig.subplots_adjust(
    left=0.25,
    right=0.88,
    top=0.91,
    bottom=0.10,
)

gradient_background(ax)
style_ax(
    ax,
    "x",
)

ax.axvline(
    0,
    color="black",
    linewidth=1.2,
    alpha=0.8,
)

for regime in regime_order:

    sub = forest[
        forest["regime"]
        == regime
    ]

    if len(sub) == 0:
        continue

    y0 = (
        sub["y"].min()
        - 0.45
    )

    y1 = (
        sub["y"].max()
        + 0.45
    )

    rect = Rectangle(
        (
            ax.get_xlim()[0],
            y0,
        ),
        100,
        y1 - y0,
        transform=ax.get_yaxis_transform(),
        facecolor=REGIME_COLOR[
            regime
        ],
        alpha=0.055,
        edgecolor="none",
        zorder=-20,
    )

    ax.add_patch(
        rect
    )

    ax.text(
        -0.235,
        (
            sub["y"].min()
            + sub["y"].max()
        )
        / 2,
        regime,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="black",
    )


for _, r in forest.iterrows():

    y = r["y"]

    ax.plot(
        [
            r["low"],
            r["high"],
        ],
        [
            y,
            y,
        ],
        color="black",
        linewidth=2.1,
        zorder=3,
    )

    ax.scatter(
        r["delta"],
        y,
        s=155,
        color=(
            vir(0.62)
            if r["delta"] >= 0
            else vir(0.95)
        ),
        edgecolor="black",
        linewidth=0.85,
        zorder=5,
    )

    bbox(
        ax,
        r["high"]
        + 0.008,
        y,
        (
            f"{r['delta']:+.3f}  "
            f"[{r['low']:+.3f}, "
            f"{r['high']:+.3f}]"
        ),
        fontsize=10,
        ha="left",
    )


ax.set_yticks(
    forest["y"]
)

ax.set_yticklabels(
    [
        f"vs {x}"
        for x in forest[
            "comparison"
        ]
    ]
)

ax.invert_yaxis()

ax.set_xlabel(
    "Role Adapter − comparator (Macro-F1)"
)

ax.set_title(
    "Effect sizes and uncertainty across evaluation regimes"
)

fig.text(
    0.5,
    0.025,
    "Within-domain: hierarchical document bootstrap · transfer/native: paired bootstrap over atomic evaluation cells.",
    ha="center",
    fontsize=12,
)

save(
    fig,
    "04_effect_sizes_forest",
)


# =====================================================================
# FIGURE 5
# METHOD DISTRIBUTIONS ACROSS 15 WITHIN-DOMAIN CELLS
# VIOLIN + RAW CELLS
# =====================================================================

order = [
    "full_ft",
    "role_adapter",
    "category_lora",
    "shared_lora_widehead",
    "shared_lora",
    "frozen",
]

values = []

for m in order:

    vals = (
        main_cells[
            main_cells[
                "method_key"
            ]
            == m
        ][
            "macro_f1_mean"
        ]
        .to_numpy()
    )

    values.append(
        vals
    )


fig, ax = plt.subplots(
    figsize=(14.2, 8.5),
    layout="constrained",
)

gradient_background(ax)
style_ax(
    ax,
    "y",
)

vp = ax.violinplot(
    values,
    positions=np.arange(
        len(order)
    ),
    widths=0.72,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

for body, m in zip(
    vp["bodies"],
    order,
):
    body.set_facecolor(
        METHOD_COLOR[m]
    )

    body.set_edgecolor(
        "black"
    )

    body.set_linewidth(
        0.9
    )

    body.set_alpha(
        0.48
    )


rng = np.random.default_rng(
    42
)

for i, (
    vals,
    m,
) in enumerate(
    zip(
        values,
        order,
    )
):

    jitter = rng.normal(
        0,
        0.045,
        size=len(vals),
    )

    ax.scatter(
        np.full(
            len(vals),
            i,
        )
        + jitter,
        vals,
        s=42,
        color=METHOD_COLOR[m],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.70,
        zorder=4,
    )

    mean = np.mean(
        vals
    )

    ax.scatter(
        i,
        mean,
        marker="D",
        s=125,
        color="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=7,
    )

    bbox(
        ax,
        i,
        mean + 0.045,
        f"μ {mean:.3f}",
        fontsize=10,
    )


ax.set_xticks(
    np.arange(
        len(order)
    )
)

ax.set_xticklabels(
    [
        method_name(m)
        for m in order
    ],
    rotation=18,
    ha="right",
)

ax.set_ylabel(
    "Test Macro-F1"
)

ax.set_title(
    "Performance distributions across 15 dataset × backbone cells"
)

fig.text(
    0.5,
    0.015,
    "Each dot is one dataset × backbone cell after averaging the three random seeds; diamonds show means.",
    ha="center",
    fontsize=12,
)

save(
    fig,
    "05_within_domain_distributions",
)


# =====================================================================
# FIGURE 6
# IL-TUR DIRECTIONAL ASYMMETRY AS 2D MAPS
# NO SLOPE LINES
# =====================================================================

ctrl = transfer[
    (
        (
            transfer["source"]
            == "iltur_cl"
        )
        &
        (
            transfer["target"]
            == "iltur_it"
        )
    )
    |
    (
        (
            transfer["source"]
            == "iltur_it"
        )
        &
        (
            transfer["target"]
            == "iltur_cl"
        )
    )
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15.5, 7.4),
    layout="constrained",
)

metrics = [
    (
        "macro_f1_mean",
        "Transfer Macro-F1",
    ),
    (
        "retention",
        "Transfer retention",
    ),
]

point_offsets = {
    "full_ft": (-38, -26),
    "role_adapter": (12, 12),
    "category_lora": (12, -24),
    "shared_lora": (10, 24),
}

for ax, (
    metric,
    ttl,
) in zip(
    axes,
    metrics,
):

    gradient_background(ax)
    style_ax(
        ax,
        "both",
    )

    coords = []

    for m in METHODS4:

        q1 = ctrl[
            (
                ctrl["source"]
                == "iltur_cl"
            )
            &
            (
                ctrl["target"]
                == "iltur_it"
            )
            &
            (
                ctrl["method"]
                == m
            )
        ]

        q2 = ctrl[
            (
                ctrl["source"]
                == "iltur_it"
            )
            &
            (
                ctrl["target"]
                == "iltur_cl"
            )
            &
            (
                ctrl["method"]
                == m
            )
        ]

        x = float(
            q1[
                metric
            ].iloc[0]
        )

        y = float(
            q2[
                metric
            ].iloc[0]
        )

        coords.append(
            (
                x,
                y,
            )
        )

        ax.scatter(
            x,
            y,
            s=185,
            color=METHOD_COLOR[m],
            edgecolor="black",
            linewidth=0.9,
            alpha=0.90,
            zorder=4,
        )

        dx, dy = (
            point_offsets[m]
        )

        ax.annotate(
            method_name(m),
            (
                x,
                y,
            ),
            xytext=(
                dx,
                dy,
            ),
            textcoords="offset points",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="#FFFDF7",
                edgecolor="black",
                linewidth=0.7,
                alpha=0.93,
            ),
            arrowprops=dict(
                arrowstyle="-",
                linewidth=0.6,
                color="black",
            ),
        )

    allvals = np.array(
        coords
    ).ravel()

    margin = (
        allvals.max()
        - allvals.min()
    ) * 0.20

    if margin == 0:
        margin = 0.03

    lo = (
        allvals.min()
        - margin
    )

    hi = (
        allvals.max()
        + margin
    )

    ax.plot(
        [
            lo,
            hi,
        ],
        [
            lo,
            hi,
        ],
        linestyle="--",
        color="black",
        alpha=0.45,
        linewidth=1.2,
    )

    ax.set_xlim(
        lo,
        hi,
    )

    ax.set_ylim(
        lo,
        hi,
    )

    ax.set_xlabel(
        "CL → IT"
    )

    ax.set_ylabel(
        "IT → CL"
    )

    ax.set_title(
        ttl
    )

    ax.text(
        0.97,
        0.04,
        "above diagonal:\nIT → CL transfers better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
    )


fig.suptitle(
    "Directional asymmetry in controlled IL-TUR transfer",
    fontsize=25,
)

save(
    fig,
    "06_iltur_directional_asymmetry",
)


# =====================================================================
# FIGURE 7
# LoRA RANK RESPONSE
# DATASET TRAJECTORIES + AGGREGATE BAND
# =====================================================================

rank_methods = [
    "shared_lora",
    "category_lora",
    "role_adapter",
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(19.0, 6.7),
    layout="constrained",
    sharey=True,
)

for ax, method in zip(
    axes,
    rank_methods,
):

    gradient_background(ax)
    style_ax(
        ax,
        "both",
    )

    sub = rank_points[
        rank_points[
            "method_key"
        ]
        == method
    ].copy()

    for ds in DATASETS:

        q = (
            sub[
                sub[
                    "dataset_key"
                ]
                == ds
            ]
            .sort_values(
                "rank"
            )
        )

        ax.plot(
            q["rank"],
            q["macro_f1"],
            linewidth=1.2,
            marker="o",
            markersize=4,
            color=DATASET_COLOR[ds],
            alpha=0.30,
            zorder=2,
        )

    grouped = (
        sub
        .groupby(
            "rank",
            as_index=False,
        )
        .agg(
            mean=(
                "macro_f1",
                "mean",
            ),
            std=(
                "macro_f1",
                "std",
            ),
        )
        .sort_values(
            "rank"
        )
    )

    ax.fill_between(
        grouped["rank"],
        grouped["mean"]
        - grouped["std"],
        grouped["mean"]
        + grouped["std"],
        color=METHOD_COLOR[method],
        alpha=0.16,
        zorder=1,
    )

    ax.plot(
        grouped["rank"],
        grouped["mean"],
        linewidth=3.0,
        marker="o",
        markersize=9,
        markeredgecolor="black",
        markeredgewidth=0.8,
        color=METHOD_COLOR[method],
        zorder=5,
    )

    best = grouped.loc[
        grouped[
            "mean"
        ].idxmax()
    ]

    bbox(
        ax,
        best["rank"],
        best["mean"]
        + 0.028,
        (
            f"best r={int(best['rank'])}\n"
            f"{best['mean']:.3f}"
        ),
        fontsize=10,
    )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_xticks(
        [
            2,
            4,
            8,
            16,
            32,
        ]
    )

    ax.get_xaxis().set_major_formatter(
        mpl.ticker.ScalarFormatter()
    )

    ax.set_title(
        method_name(method)
    )

    ax.set_xlabel(
        "LoRA rank"
    )


axes[0].set_ylabel(
    "Test Macro-F1"
)

fig.suptitle(
    "Rank sensitivity and cross-dataset stability",
    fontsize=25,
)

legend = [
    Line2D(
        [0],
        [0],
        color=DATASET_COLOR[d],
        lw=3,
        alpha=0.7,
        label=dataset_name(d),
    )
    for d in DATASETS
]

fig.legend(
    handles=legend,
    loc="outside lower center",
    ncol=5,
    frameon=True,
    edgecolor="black",
)

save(
    fig,
    "07_rank_sensitivity",
)


# =====================================================================
# FIGURE 8
# BACKBONE-SPECIFIC GAP TO FULL FT
# FOREST SMALL MULTIPLES
# =====================================================================

peft_methods = [
    "role_adapter",
    "category_lora",
    "shared_lora_widehead",
    "shared_lora",
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(19.5, 7.0),
    layout="constrained",
    sharex=True,
    sharey=True,
)

for ax, model in zip(
    axes,
    [
        "inlegalbert",
        "legalbert",
        "deberta",
    ],
):

    gradient_background(ax)
    style_ax(
        ax,
        "x",
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1.1,
        alpha=0.65,
    )

    p = (
        main_cells[
            main_cells[
                "model_key"
            ]
            == model
        ]
        .pivot_table(
            index="dataset_key",
            columns="method_key",
            values="macro_f1_mean",
        )
    )

    ypos = np.arange(
        len(peft_methods)
    )

    for yy, method in zip(
        ypos,
        peft_methods,
    ):

        d = (
            p[method]
            - p["full_ft"]
        ).dropna().to_numpy()

        mean, lo, hi = (
            bootstrap_mean_ci(
                d,
                B=10000,
                seed=42 + yy,
            )
        )

        ax.plot(
            [
                lo,
                hi,
            ],
            [
                yy,
                yy,
            ],
            color="black",
            linewidth=2.0,
        )

        ax.scatter(
            mean,
            yy,
            s=150,
            color=METHOD_COLOR[
                method
            ],
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

        bbox(
            ax,
            hi + 0.005,
            yy,
            f"{mean:+.3f}",
            fontsize=9.5,
            ha="left",
        )

    ax.set_yticks(
        ypos
    )

    ax.set_yticklabels(
        [
            method_name(m)
            for m in peft_methods
        ]
    )

    ax.invert_yaxis()

    ax.set_title(
        BACKBONE_LABEL[model]
    )

    ax.set_xlabel(
        "PEFT − Full FT"
    )


fig.suptitle(
    "Backbone-specific adaptation gap to full fine-tuning",
    fontsize=25,
)

fig.text(
    0.5,
    0.01,
    "Points are mean Δ Macro-F1 across five datasets; intervals are paired bootstrap confidence intervals over datasets.",
    ha="center",
    fontsize=12,
)

save(
    fig,
    "08_backbone_gap_to_fullft",
)


# =====================================================================
# FIGURE 9
# GLOBAL ROLE-vs-SHARED WATERFALL
# 15 WITHIN + 20 TRANSFER + 3 NATIVE = 38 ATOMIC CELLS
# =====================================================================

atomic = []

# within 15
for idx, r in within4.iterrows():

    atomic.append({
        "regime":
            "Within-domain",

        "name":
            (
                f"{idx[0]} / "
                f"{idx[1]}"
            ),

        "delta":
            (
                r[
                    "role_adapter"
                ]
                - r[
                    "shared_lora"
                ]
            ),
    })


# transfer 20
for _, r in role_shared_transfer.iterrows():

    atomic.append({
        "regime":
            "Transfer",

        "name":
            (
                f"{r['source']}→"
                f"{r['target']}"
            ),

        "delta":
            r[
                "delta_role_minus_shared"
            ],
    })


# native 3
for ds in native_pivot.index:

    if (
        "role_adapter"
        not in native_pivot.columns
        or
        "shared_lora"
        not in native_pivot.columns
    ):
        continue

    atomic.append({
        "regime":
            "Native",

        "name":
            ds,

        "delta":
            (
                native_pivot.loc[
                    ds,
                    "role_adapter",
                ]
                -
                native_pivot.loc[
                    ds,
                    "shared_lora",
                ]
            ),
    })


atomic = pd.DataFrame(
    atomic
)

atomic = atomic.sort_values(
    "delta"
).reset_index(
    drop=True
)

fig, ax = plt.subplots(
    figsize=(15.5, 8.3),
    layout="constrained",
)

gradient_background(ax)
style_ax(
    ax,
    "y",
)

ax.axhline(
    0,
    color="black",
    linewidth=1.2,
)

for regime in [
    "Within-domain",
    "Transfer",
    "Native",
]:

    q = atomic[
        atomic[
            "regime"
        ]
        == regime
    ]

    ax.scatter(
        q.index,
        q["delta"],
        s=70,
        color=REGIME_COLOR[
            regime
        ],
        edgecolor="black",
        linewidth=0.55,
        alpha=0.84,
        label=regime,
        zorder=4,
    )

    for idx, row in q.iterrows():

        ax.plot(
            [
                idx,
                idx,
            ],
            [
                0,
                row["delta"],
            ],
            color=REGIME_COLOR[
                regime
            ],
            linewidth=1.0,
            alpha=0.38,
            zorder=2,
        )


overall_delta = (
    atomic["delta"].mean()
)

ax.axhline(
    overall_delta,
    linestyle="--",
    linewidth=1.8,
    color=vir(0.60),
    alpha=0.9,
)

bbox(
    ax,
    len(atomic)
    - 0.5,
    overall_delta,
    (
        f"overall mean "
        f"{overall_delta:+.3f}"
    ),
    fontsize=11,
    ha="right",
)

ax.set_xlabel(
    "Atomic evaluation cells sorted by Role Adapter advantage"
)

ax.set_ylabel(
    "Role Adapter − Shared LoRA (Macro-F1)"
)

ax.set_title(
    "Consistency of Role Adapter gains across all evaluation regimes"
)

ax.legend(
    loc="upper left",
    frameon=True,
    edgecolor="black",
)

wins = (
    atomic.groupby(
        "regime"
    )["delta"]
    .apply(
        lambda x:
        f"{int((x > 0).sum())}/{len(x)}"
    )
)

fig.text(
    0.5,
    0.015,
    (
        "Positive-cell counts — "
        + " · ".join(
            [
                f"{k}: {v}"
                for k, v
                in wins.items()
            ]
        )
    ),
    ha="center",
    fontsize=12,
)

save(
    fig,
    "09_role_vs_shared_global_consistency",
)


# =====================================================================
# FIGURE 10
# TRANSFER SOURCE ROBUSTNESS PROFILE
#
# For each source:
# mean transfer F1 across other targets
# and role advantage over shared
# =====================================================================

source_rows = []

for src in DATASETS:

    qr = transfer[
        (
            transfer["source"]
            == src
        )
        &
        (
            transfer["target"]
            != src
        )
        &
        (
            transfer["method"]
            == "role_adapter"
        )
    ]

    qs = transfer[
        (
            transfer["source"]
            == src
        )
        &
        (
            transfer["target"]
            != src
        )
        &
        (
            transfer["method"]
            == "shared_lora"
        )
    ]

    rr = qr.set_index(
        "target"
    )[
        "macro_f1_mean"
    ]

    ss = qs.set_index(
        "target"
    )[
        "macro_f1_mean"
    ]

    common = rr.index.intersection(
        ss.index
    )

    source_rows.append({
        "source":
            src,

        "role_mean":
            rr.loc[
                common
            ].mean(),

        "shared_mean":
            ss.loc[
                common
            ].mean(),

        "role_advantage":
            (
                rr.loc[
                    common
                ]
                - ss.loc[
                    common
                ]
            ).mean(),

        "role_retention":
            qr[
                "retention"
            ].mean(),
    })


source_profile = pd.DataFrame(
    source_rows
).sort_values(
    "role_mean",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(13.5, 8.0),
    layout="constrained",
)

gradient_background(ax)
style_ax(
    ax,
    "x",
)

yy = np.arange(
    len(source_profile)
)

for y, (
    _,
    r,
) in zip(
    yy,
    source_profile.iterrows(),
):

    ax.plot(
        [
            r["shared_mean"],
            r["role_mean"],
        ],
        [
            y,
            y,
        ],
        color="black",
        linewidth=1.4,
        alpha=0.55,
    )

    ax.scatter(
        r["shared_mean"],
        y,
        s=130,
        color=METHOD_COLOR[
            "shared_lora"
        ],
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
    )

    ax.scatter(
        r["role_mean"],
        y,
        s=150,
        color=METHOD_COLOR[
            "role_adapter"
        ],
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )

    bbox(
        ax,
        max(
            r[
                "shared_mean"
            ],
            r[
                "role_mean"
            ],
        )
        + 0.012,
        y,
        (
            f"Δ {r['role_advantage']:+.3f}"
            f" · ret {r['role_retention']:.2f}"
        ),
        fontsize=10,
        ha="left",
    )


ax.set_yticks(
    yy
)

ax.set_yticklabels(
    [
        dataset_name(x)
        for x in source_profile[
            "source"
        ]
    ]
)

ax.set_xlabel(
    "Mean cross-domain target Macro-F1"
)

ax.set_title(
    "Source-domain transfer robustness"
)

legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        markersize=10,
        linestyle="none",
        markerfacecolor=METHOD_COLOR[
            "shared_lora"
        ],
        markeredgecolor="black",
        label="Shared LoRA",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        markersize=10,
        linestyle="none",
        markerfacecolor=METHOD_COLOR[
            "role_adapter"
        ],
        markeredgecolor="black",
        label="Role Adapter",
    ),
]

ax.legend(
    handles=legend,
    loc="lower right",
    frameon=True,
    edgecolor="black",
)

save(
    fig,
    "10_source_transfer_robustness",
)


# =====================================================================
# FIGURE 11
# MECHANISTIC ABLATION:
# TRUE CATEGORY LORA VS CHEAPER ALTERNATIVES
# =====================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16.5, 7.6),
    layout="constrained",
)


# ---------------------------------------------------------------------
# left: per-dataset true category minus Role Adapter
# ---------------------------------------------------------------------

ax = axes[0]

gradient_background(ax)
style_ax(
    ax,
    "x",
)

mech2 = mech.copy()

mech2[
    "delta_true_role"
] = (
    mech2[
        "true_category_f1"
    ]
    -
    mech2[
        "role_adapter_f1"
    ]
)

mech2 = mech2.sort_values(
    "delta_true_role"
)

yy = np.arange(
    len(mech2)
)

ax.axvline(
    0,
    color="black",
    linewidth=1.1,
)

for y, (
    _,
    r,
) in zip(
    yy,
    mech2.iterrows(),
):

    d = r[
        "delta_true_role"
    ]

    ax.plot(
        [
            0,
            d,
        ],
        [
            y,
            y,
        ],
        color=(
            vir(0.68)
            if d > 0
            else vir(0.94)
        ),
        linewidth=2.2,
        alpha=0.7,
    )

    ax.scatter(
        d,
        y,
        s=145,
        color=(
            vir(0.68)
            if d > 0
            else vir(0.94)
        ),
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
    )

    bbox(
        ax,
        d
        + (
            0.004
            if d >= 0
            else -0.004
        ),
        y,
        f"{d:+.3f}",
        fontsize=10,
        ha=(
            "left"
            if d >= 0
            else "right"
        ),
    )


ax.set_yticks(
    yy
)

ax.set_yticklabels(
    [
        dataset_name(x)
        for x in mech2[
            "dataset"
        ]
    ]
)

ax.set_xlabel(
    "True category LoRA − Role Adapter"
)

ax.set_title(
    "Does role-specific encoder LoRA help?"
)


# ---------------------------------------------------------------------
# right: performance vs VRAM
# ---------------------------------------------------------------------

ax = axes[1]

gradient_background(ax)
style_ax(
    ax,
    "both",
)

mean_true = (
    mech[
        "true_category_f1"
    ].mean()
)

# InLegalBERT efficiency rows
ineff = main_eff[
    main_eff[
        "model_key"
    ]
    == "inlegalbert"
].set_index(
    "method_key"
)

perf = (
    main_cells[
        main_cells[
            "model_key"
        ]
        == "inlegalbert"
    ]
    .groupby(
        "method_key"
    )[
        "macro_f1_mean"
    ]
    .mean()
)

compute_points = [
    (
        "shared_lora",
        float(
            ineff.loc[
                "shared_lora",
                "peak_vram_gb_mean",
            ]
        ),
        float(
            perf.loc[
                "shared_lora"
            ]
        ),
    ),
    (
        "category_lora",
        float(
            ineff.loc[
                "category_lora",
                "peak_vram_gb_mean",
            ]
        ),
        float(
            perf.loc[
                "category_lora"
            ]
        ),
    ),
    (
        "role_adapter",
        float(
            ineff.loc[
                "role_adapter",
                "peak_vram_gb_mean",
            ]
        ),
        float(
            perf.loc[
                "role_adapter"
            ]
        ),
    ),
    (
        "true_category_lora",
        float(
            mech[
                "true_category_vram_gb"
            ].mean()
        ),
        float(
            mean_true
        ),
    ),
]

offsets = {
    "shared_lora": (8, -22),
    "category_lora": (8, 16),
    "role_adapter": (8, 34),
    "true_category_lora": (-8, 12),
}

for m, vram, f1 in compute_points:

    ax.scatter(
        vram,
        f1,
        s=190,
        color=METHOD_COLOR[m],
        edgecolor="black",
        linewidth=0.9,
        alpha=0.88,
    )

    dx, dy = offsets[m]

    ax.annotate(
        (
            f"{method_name(m)}\n"
            f"{vram:.2f} GB · F1 {f1:.3f}"
        ),
        (
            vram,
            f1,
        ),
        xytext=(
            dx,
            dy,
        ),
        textcoords="offset points",
        ha=(
            "right"
            if dx < 0
            else "left"
        ),
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="#FFFDF7",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.93,
        ),
        arrowprops=dict(
            arrowstyle="-",
            linewidth=0.6,
            color="black",
        ),
    )


ax.set_xlabel(
    "Peak GPU memory (GB)"
)

ax.set_ylabel(
    "Mean Macro-F1"
)

ax.set_title(
    "Mechanistic cost–performance trade-off"
)

fig.suptitle(
    "Mechanistic ablation of role-specific adaptation",
    fontsize=25,
)

save(
    fig,
    "11_mechanistic_cost_performance",
)


# =====================================================================
# SAVE DERIVED TABLES
# =====================================================================

atomic.to_csv(
    OUT
    / "global_role_vs_shared_cells.csv",
    index=False,
)

source_profile.to_csv(
    OUT
    / "source_transfer_profile.csv",
    index=False,
)

forest.to_csv(
    OUT
    / "effect_forest_data.csv",
    index=False,
)


# =====================================================================
# CREATE GALLERY
# =====================================================================

pngs = sorted(
    OUT.glob(
        "[0-9][0-9]_*.png"
    )
)

cols = 3
rows = math.ceil(
    len(pngs)
    / cols
)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(
        18,
        5.4 * rows,
    ),
)

axes = np.atleast_1d(
    axes
).ravel()

for ax, fn in zip(
    axes,
    pngs,
):

    img = plt.imread(
        fn
    )

    ax.imshow(
        img
    )

    ax.set_title(
        fn.stem,
        fontsize=12,
    )

    ax.axis(
        "off"
    )


for ax in axes[
    len(pngs):
]:
    ax.axis(
        "off"
    )


fig.tight_layout()

fig.savefig(
    OUT
    / "00_gallery.png",
    dpi=170,
    bbox_inches="tight",
)

plt.close(
    fig
)


# =====================================================================
# PRINT INVENTORY
# =====================================================================

print()
print("=" * 110)
print("NEW PAPER FIGURE PACKAGE COMPLETE")
print("=" * 110)

for fn in sorted(
    OUT.glob(
        "*.pdf"
    )
):
    print(fn.name)

print()
print(
    "Gallery:",
    OUT / "00_gallery.png"
)
