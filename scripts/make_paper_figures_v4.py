from pathlib import Path
import math
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
    Circle,
    Rectangle,
)
from matplotlib.lines import Line2D

from scipy.stats import spearmanr, gaussian_kde


# ======================================================================
# PATHS
# ======================================================================

ROOT = Path.home() / "TaxFacto"

MAIN = ROOT / "results/final_analysis"
ADD = ROOT / "results/additional/analysis"
BOOT = ROOT / "results/additional/bootstrap"
CORE8 = ROOT / "results/additional/fair_core8"

OUT = ROOT / "results/paper_figures_v4"
OUT.mkdir(parents=True, exist_ok=True)


# ======================================================================
# STYLE
# ======================================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",

    # Intentionally large for reduction in paper
    "font.size": 18,
    "axes.titlesize": 24,
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

    "savefig.dpi": 280,
})


# ======================================================================
# LABELS
# ======================================================================

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


# ======================================================================
# COLORS
# ======================================================================

vir = plt.cm.viridis
civ = plt.cm.cividis

METHOD_COLOR = {
    "full_ft": vir(0.94),
    "role_adapter": vir(0.64),
    "category_lora": vir(0.46),
    "shared_lora_widehead": civ(0.39),
    "shared_lora": vir(0.18),
    "frozen": (0.72, 0.74, 0.78, 1.0),
    "true_category_lora": civ(0.78),
}

DATASET_COLOR = {
    ds: vir(v)
    for ds, v in zip(
        DATASETS,
        np.linspace(
            0.10,
            0.90,
            len(DATASETS),
        ),
    )
}

REGIME_COLOR = {
    "Within-domain": vir(0.16),
    "Transfer": vir(0.58),
    "Native": vir(0.90),
}


CMAP_DELTA = LinearSegmentedColormap.from_list(
    "delta",
    [
        "#482878",
        "#365C8D",
        "#F6F4EF",
        "#27AD81",
        "#FDE725",
    ],
)


# ======================================================================
# HELPERS
# ======================================================================

def mname(x):
    return METHOD_LABEL.get(
        x,
        x,
    )


def dname(x):
    return DATASET_LABEL.get(
        x,
        x,
    )


def bg(
    ax,
    top="#EDF8FF",
    bottom="#FFF8EE",
    alpha=0.92,
):
    arr = np.linspace(
        0,
        1,
        300,
    ).reshape(
        -1,
        1,
    )

    cmap = LinearSegmentedColormap.from_list(
        "bg",
        [
            top,
            bottom,
        ],
    )

    ax.imshow(
        arr,
        extent=(
            0,
            1,
            0,
            1,
        ),
        transform=ax.transAxes,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        alpha=alpha,
        zorder=-100,
    )


def style(
    ax,
    grid="y",
):
    ax.set_facecolor(
        "#FBFAF7"
    )

    for s in ax.spines.values():
        s.set_color(
            "black"
        )
        s.set_linewidth(
            0.9
        )

    if grid:
        ax.grid(
            True,
            axis=grid,
            color="black",
            alpha=0.12,
            linewidth=0.7,
            zorder=-10,
        )

    ax.set_axisbelow(
        True
    )


def labelbox(
    ax,
    x,
    y,
    txt,
    fs=10.5,
    ha="center",
    va="center",
    face="#FFFDF8",
):
    ax.text(
        x,
        y,
        txt,
        ha=ha,
        va=va,
        fontsize=fs,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor=face,
            edgecolor="black",
            linewidth=0.65,
            alpha=0.94,
        ),
        zorder=50,
    )


def save(
    fig,
    stem,
):
    fig.savefig(
        OUT / f"{stem}.png",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def boot_ci(
    x,
    B=10000,
    seed=42,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    rng = np.random.default_rng(
        seed
    )

    samples = rng.choice(
        x,
        (
            B,
            len(x),
        ),
        replace=True,
    )

    means = samples.mean(
        axis=1
    )

    return (
        x.mean(),
        np.quantile(
            means,
            0.025,
        ),
        np.quantile(
            means,
            0.975,
        ),
    )


# ======================================================================
# LOAD
# ======================================================================

main_cells = pd.read_csv(
    MAIN / "main_cell_mean_std.csv"
)

main_overall = pd.read_csv(
    MAIN / "main_method_overall.csv"
)

main_eff = pd.read_csv(
    MAIN / "main_efficiency.csv"
)

transfer = pd.read_csv(
    ADD / "transfer_summary.csv"
)

transfer_role_shared = pd.read_csv(
    ADD / "transfer_role_vs_shared.csv"
)

native = pd.read_csv(
    ADD / "native_summary.csv"
)

boot = pd.read_csv(
    BOOT / "hierarchical_bootstrap_global.csv"
)

core8 = pd.read_csv(
    CORE8 / "transfer_core8_pivot.csv"
)

core8 = core8.set_index(
    [
        "source",
        "target",
    ]
)


# ======================================================================
# COMMON DERIVED DATA
# ======================================================================

within = (
    main_cells[
        main_cells[
            "method_key"
        ].isin(
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

native_p = (
    native
    .pivot_table(
        index="dataset",
        columns="method",
        values="macro_f1_mean",
    )
)


# ======================================================================
# FIGURE 01
# BROKEN X AXIS — NO LINE, NO INSET
# ======================================================================

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

eff = (
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

p = overall.merge(
    eff,
    on="method_key",
)

fig = plt.figure(
    figsize=(15.4, 8.5),
)

gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[
        4.8,
        1.25,
    ],
    wspace=0.045,
)

axL = fig.add_subplot(
    gs[0, 0]
)

axR = fig.add_subplot(
    gs[0, 1],
    sharey=axL,
)

for ax in [
    axL,
    axR,
]:
    bg(ax)
    style(
        ax,
        "both",
    )


peft = p[
    p["method_key"]
    != "full_ft"
].copy()

full = p[
    p["method_key"]
    == "full_ft"
].iloc[0]


# left axis targeted to actual PEFT space
left_max = (
    peft[
        "trainable_pct"
    ].max()
    * 1.22
)

axL.set_xlim(
    0,
    left_max,
)

# right axis around full FT only
axR.set_xlim(
    98.7,
    100.8,
)

ymin = (
    p[
        "macro_f1_mean"
    ].min()
    - 0.045
)

ymax = (
    p[
        "macro_f1_mean"
    ].max()
    + 0.075
)

axL.set_ylim(
    ymin,
    ymax,
)

axL.set_ylabel(
    "Mean within-domain Macro-F1"
)

fig.supxlabel(
    "Trainable parameters (%)",
    fontsize=20,
    y=0.035,
)

fig.suptitle(
    "Accuracy–efficiency landscape",
    fontsize=25,
    y=0.975,
)


# annotation slots — manually separated, not over point
slot_y = {
    "role_adapter": 0.620,
    "category_lora": 0.585,
    "shared_lora_widehead": 0.545,
    "shared_lora": 0.492,
    "frozen": 0.210,
}


for _, r in peft.iterrows():

    m = r[
        "method_key"
    ]

    x = r[
        "trainable_pct"
    ]

    y = r[
        "macro_f1_mean"
    ]

    size = (
        160
        + 1.2
        * r[
            "train_seconds"
        ]
    )

    axL.errorbar(
        x,
        y,
        yerr=r[
            "macro_f1_std_across_cells"
        ],
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=4,
        zorder=3,
    )

    axL.scatter(
        x,
        y,
        s=size,
        color=METHOD_COLOR[m],
        edgecolor="black",
        linewidth=0.85,
        alpha=0.84,
        zorder=5,
    )

    label_x = (
        left_max
        * 0.70
    )

    label_y = slot_y[m]

    axL.annotate(
        (
            f"{mname(m)}\n"
            f"F1 {y:.3f} · "
            f"{x:.3f}%"
        ),
        xy=(
            x,
            y,
        ),
        xytext=(
            label_x,
            label_y,
        ),
        textcoords="data",
        fontsize=11,
        va="center",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="#FFFDF8",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.95,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color="black",
            linewidth=0.65,
            alpha=0.70,
        ),
        zorder=20,
    )


# Full FT
axR.errorbar(
    full[
        "trainable_pct"
    ],
    full[
        "macro_f1_mean"
    ],
    yerr=full[
        "macro_f1_std_across_cells"
    ],
    fmt="none",
    ecolor="black",
    elinewidth=1.0,
    capsize=4,
)

axR.scatter(
    full[
        "trainable_pct"
    ],
    full[
        "macro_f1_mean"
    ],
    s=(
        160
        + 1.2
        * full[
            "train_seconds"
        ]
    ),
    color=METHOD_COLOR[
        "full_ft"
    ],
    edgecolor="black",
    linewidth=0.85,
    alpha=0.88,
    zorder=5,
)

labelbox(
    axR,
    99.12,
    full[
        "macro_f1_mean"
    ]
    + 0.028,
    (
        "Full FT\n"
        f"F1 {full['macro_f1_mean']:.3f}\n"
        "100%"
    ),
    fs=11,
)


# broken-axis cosmetics
axL.spines[
    "right"
].set_visible(
    False
)

axR.spines[
    "left"
].set_visible(
    False
)

axR.tick_params(
    labelleft=False,
    left=False,
)

d = 0.014

kwargs = dict(
    transform=axL.transAxes,
    color="black",
    clip_on=False,
    linewidth=1.2,
)

axL.plot(
    (
        1 - d,
        1 + d,
    ),
    (
        -d,
        +d,
    ),
    **kwargs,
)

axL.plot(
    (
        1 - d,
        1 + d,
    ),
    (
        1 - d,
        1 + d,
    ),
    **kwargs,
)

kwargs.update(
    transform=axR.transAxes
)

axR.plot(
    (
        -d,
        +d,
    ),
    (
        -d,
        +d,
    ),
    **kwargs,
)

axR.plot(
    (
        -d,
        +d,
    ),
    (
        1 - d,
        1 + d,
    ),
    **kwargs,
)


axL.text(
    0.015,
    0.025,
    "Marker area ∝ mean training time",
    transform=axL.transAxes,
    fontsize=12,
)

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.90,
    bottom=0.12,
)

save(
    fig,
    "01_accuracy_efficiency_broken_axis",
)


# ======================================================================
# FIGURE 04
# TARGETED EFFECT FOREST — NO FROZEN
# ======================================================================

rows = []


# within-domain
for comparator in [
    "shared_lora",
    "shared_lora_widehead",
    "category_lora",
    "full_ft",
]:

    q = boot[
        (
            boot[
                "method_a"
            ]
            == comparator
        )
        &
        (
            boot[
                "method_b"
            ]
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
            mname(
                comparator
            ),

        "delta":
            -r[
                "point_mean_delta"
            ],

        "low":
            -r[
                "ci95_high"
            ],

        "high":
            -r[
                "ci95_low"
            ],
    })


# transfer
for comparator in [
    "shared_lora",
    "category_lora",
    "full_ft",
]:

    d = (
        core8[
            "role_adapter"
        ]
        - core8[
            comparator
        ]
    ).to_numpy()

    mean, low, high = (
        boot_ci(
            d,
            seed=100
            + len(rows),
        )
    )

    rows.append({
        "regime":
            "Transfer",

        "comparison":
            mname(
                comparator
            ),

        "delta":
            mean,

        "low":
            low,

        "high":
            high,
    })


# native
for comparator in [
    "shared_lora",
    "full_ft",
]:

    d = (
        native_p[
            "role_adapter"
        ]
        - native_p[
            comparator
        ]
    ).dropna().to_numpy()

    mean, low, high = (
        boot_ci(
            d,
            seed=200
            + len(rows),
        )
    )

    rows.append({
        "regime":
            "Native",

        "comparison":
            mname(
                comparator
            ),

        "delta":
            mean,

        "low":
            low,

        "high":
            high,
    })


forest = pd.DataFrame(
    rows
)


# separated y coords
ycoords = []
cursor = 0

for regime in [
    "Within-domain",
    "Transfer",
    "Native",
]:

    inds = forest[
        forest[
            "regime"
        ]
        == regime
    ].index

    for idx in inds:

        ycoords.append(
            (
                idx,
                cursor,
            )
        )

        cursor += 1

    cursor += 0.75


ymap = dict(
    ycoords
)

forest["y"] = forest.index.map(
    ymap
)


fig, ax = plt.subplots(
    figsize=(14.8, 9.2),
)

fig.subplots_adjust(
    left=0.28,
    right=0.91,
    top=0.91,
    bottom=0.11,
)

bg(ax)
style(
    ax,
    "x",
)

ax.axvline(
    0,
    color="black",
    linewidth=1.15,
)


for regime in [
    "Within-domain",
    "Transfer",
    "Native",
]:

    q = forest[
        forest[
            "regime"
        ]
        == regime
    ]

    y0 = (
        q[
            "y"
        ].min()
        - 0.42
    )

    y1 = (
        q[
            "y"
        ].max()
        + 0.42
    )

    color = {
        "Within-domain":
            vir(0.15),

        "Transfer":
            vir(0.58),

        "Native":
            vir(0.90),
    }[
        regime
    ]

    ax.axhspan(
        y0,
        y1,
        color=color,
        alpha=0.055,
        zorder=-20,
    )

    ax.text(
        -0.185,
        (
            y0
            + y1
        )
        / 2,
        regime,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=14,
        fontweight="bold",
    )


for _, r in forest.iterrows():

    y = r[
        "y"
    ]

    ax.plot(
        [
            r[
                "low"
            ],
            r[
                "high"
            ],
        ],
        [
            y,
            y,
        ],
        color="black",
        linewidth=2.0,
        zorder=3,
    )

    ax.scatter(
        r[
            "delta"
        ],
        y,
        s=165,
        color=(
            vir(0.62)
            if r[
                "delta"
            ]
            >= 0
            else vir(0.94)
        ),
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )

    labelbox(
        ax,
        r[
            "high"
        ]
        + 0.004,
        y,
        (
            f"{r['delta']:+.3f}\n"
            f"[{r['low']:+.3f},"
            f" {r['high']:+.3f}]"
        ),
        fs=9.6,
        ha="left",
    )


ax.set_yticks(
    forest[
        "y"
    ]
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

# Targeted range — relevant comparisons only
low = forest[
    "low"
].min()

high = forest[
    "high"
].max()

pad = (
    high
    - low
) * 0.18

ax.set_xlim(
    low - pad,
    high + pad * 2.0,
)

ax.set_xlabel(
    "Role Adapter − comparator (Macro-F1)"
)

ax.set_title(
    "Effect sizes across evaluation regimes"
)

ax.text(
    0.99,
    0.02,
    (
        "Within-domain: document bootstrap\n"
        "Transfer/native: paired cell bootstrap"
    ),
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
)

save(
    fig,
    "04_effect_sizes_targeted",
)


# ======================================================================
# FIGURE 06
# DIRECTIONAL ASYMMETRY — SIGNED BARS
# ======================================================================

ctrl = transfer[
    (
        (
            (
                transfer[
                    "source"
                ]
                == "iltur_cl"
            )
            &
            (
                transfer[
                    "target"
                ]
                == "iltur_it"
            )
        )
        |
        (
            (
                transfer[
                    "source"
                ]
                == "iltur_it"
            )
            &
            (
                transfer[
                    "target"
                ]
                == "iltur_cl"
            )
        )
    )
    &
    (
        transfer[
            "method"
        ].isin(
            METHODS4
        )
    )
]


rows = []

for method in METHODS4:

    a = ctrl[
        (
            ctrl[
                "source"
            ]
            == "iltur_cl"
        )
        &
        (
            ctrl[
                "target"
            ]
            == "iltur_it"
        )
        &
        (
            ctrl[
                "method"
            ]
            == method
        )
    ].iloc[0]

    b = ctrl[
        (
            ctrl[
                "source"
            ]
            == "iltur_it"
        )
        &
        (
            ctrl[
                "target"
            ]
            == "iltur_cl"
        )
        &
        (
            ctrl[
                "method"
            ]
            == method
        )
    ].iloc[0]

    rows.append({
        "method":
            method,

        "cl_it_f1":
            a[
                "macro_f1_mean"
            ],

        "it_cl_f1":
            b[
                "macro_f1_mean"
            ],

        "f1_asym":
            (
                b[
                    "macro_f1_mean"
                ]
                - a[
                    "macro_f1_mean"
                ]
            ),

        "cl_it_ret":
            a[
                "retention"
            ],

        "it_cl_ret":
            b[
                "retention"
            ],

        "ret_asym":
            (
                b[
                    "retention"
                ]
                - a[
                    "retention"
                ]
            ),
    })


asym = pd.DataFrame(
    rows
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16.5, 7.4),
    layout="constrained",
)

spec = [
    (
        axes[0],
        "f1_asym",
        "Directional Macro-F1 asymmetry",
        "IT → CL minus CL → IT",
        "cl_it_f1",
        "it_cl_f1",
    ),
    (
        axes[1],
        "ret_asym",
        "Directional retention asymmetry",
        "IT → CL minus CL → IT",
        "cl_it_ret",
        "it_cl_ret",
    ),
]


for ax, col, ttl, xlabel, leftcol, rightcol in spec:

    bg(ax)
    style(
        ax,
        "x",
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1.1,
    )

    yy = np.arange(
        len(
            asym
        )
    )

    vals = asym[
        col
    ].to_numpy()

    for y, (
        _,
        r,
    ) in zip(
        yy,
        asym.iterrows(),
    ):

        v = r[
            col
        ]

        ax.barh(
            y,
            v,
            height=0.46,
            color=METHOD_COLOR[
                r[
                    "method"
                ]
            ],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.78,
            zorder=3,
        )

        pointx = (
            v
        )

        labelbox(
            ax,
            pointx
            + (
                0.005
                if v >= 0
                else -0.005
            ),
            y,
            (
                f"{v:+.3f}\n"
                f"{r[leftcol]:.3f}"
                f" → "
                f"{r[rightcol]:.3f}"
            ),
            fs=9.5,
            ha=(
                "left"
                if v >= 0
                else "right"
            ),
        )

    ax.set_yticks(
        yy
    )

    ax.set_yticklabels(
        [
            mname(m)
            for m in asym[
                "method"
            ]
        ]
    )

    ax.invert_yaxis()

    maxabs = max(
        np.abs(
            vals
        ).max(),
        0.025,
    )

    ax.set_xlim(
        -maxabs
        * 0.35,
        maxabs
        * 1.90,
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_title(
        ttl
    )

    ax.text(
        0.98,
        0.04,
        (
            "positive → IT→CL\n"
            "transfers better"
        ),
        transform=ax.transAxes,
        ha="right",
        fontsize=11,
    )


fig.suptitle(
    "Controlled IL-TUR directional asymmetry",
    fontsize=25,
)

save(
    fig,
    "06_iltur_asymmetry_bars",
)


# ======================================================================
# FIGURE 09
# THREE-BLOCK GAIN SPECTRUM
# NO BOTTOM OVERLAP
# ======================================================================

atomic = []


# within
for idx, r in within.iterrows():

    atomic.append({
        "regime":
            "Within-domain",

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


# transfer
for _, r in transfer_role_shared.iterrows():

    atomic.append({
        "regime":
            "Transfer",

        "delta":
            r[
                "delta_role_minus_shared"
            ],
    })


# native
for ds in native_p.index:

    atomic.append({
        "regime":
            "Native",

        "delta":
            (
                native_p.loc[
                    ds,
                    "role_adapter",
                ]
                - native_p.loc[
                    ds,
                    "shared_lora",
                ]
            ),
    })


atomic = pd.DataFrame(
    atomic
)

fig, ax = plt.subplots(
    figsize=(15.5, 8.0),
)

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.88,
    bottom=0.14,
)

bg(ax)
style(
    ax,
    "y",
)

ax.axhline(
    0,
    color="black",
    linewidth=1.1,
)


cursor = 0

block_centers = []

for regime in [
    "Within-domain",
    "Transfer",
    "Native",
]:

    q = (
        atomic[
            atomic[
                "regime"
            ]
            == regime
        ]
        .sort_values(
            "delta"
        )
        .copy()
    )

    xs = np.arange(
        cursor,
        cursor
        + len(q),
    )

    q["x"] = xs

    color = REGIME_COLOR[
        regime
    ]

    for x, val in zip(
        xs,
        q[
            "delta"
        ],
    ):

        ax.plot(
            [
                x,
                x,
            ],
            [
                0,
                val,
            ],
            color=color,
            linewidth=1.1,
            alpha=0.40,
        )

    ax.scatter(
        xs,
        q[
            "delta"
        ],
        s=65,
        color=color,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.88,
        zorder=4,
    )

    wins = int(
        (
            q[
                "delta"
            ]
            > 0
        ).sum()
    )

    center = (
        xs.min()
        + xs.max()
    ) / 2

    block_centers.append(
        (
            center,
            regime,
            wins,
            len(q),
        )
    )

    if regime != "Native":

        ax.axvline(
            xs.max()
            + 0.8,
            color="black",
            linewidth=0.8,
            alpha=0.18,
        )

    cursor = (
        xs.max()
        + 2
    )


for center, regime, wins, n in block_centers:

    ax.text(
        center,
        0.97,
        (
            f"{regime}\n"
            f"{wins}/{n} positive"
        ),
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#FFFDF8",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.93,
        ),
    )


mean_delta = atomic[
    "delta"
].mean()

ax.axhline(
    mean_delta,
    linestyle="--",
    color=vir(0.61),
    linewidth=1.7,
)

ax.text(
    0.99,
    mean_delta,
    f" overall mean {mean_delta:+.3f} ",
    transform=ax.get_yaxis_transform(),
    ha="right",
    va="bottom",
    fontsize=11,
    bbox=dict(
        facecolor="#FFFDF8",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.92,
    ),
)

ax.set_xticks(
    []
)

ax.set_xlabel(
    "Atomic evaluation cells, ordered within each regime"
)

ax.set_ylabel(
    "Role Adapter − Shared LoRA (Macro-F1)"
)

ax.set_title(
    "Cross-regime consistency spectrum"
)

save(
    fig,
    "09_cross_regime_gain_spectrum",
)


# ======================================================================
# FIGURE 12
# WHEN DOES ROLE ADAPTER HELP?
# BASELINE DIFFICULTY VS GAIN
# ======================================================================

role_ret = (
    transfer[
        transfer[
            "method"
        ]
        == "role_adapter"
    ][
        [
            "source",
            "target",
            "retention",
        ]
    ]
)

difficulty = (
    transfer_role_shared
    .merge(
        role_ret,
        on=[
            "source",
            "target",
        ],
        how="left",
    )
)

rho, rho_p = spearmanr(
    difficulty[
        "shared_lora_f1"
    ],
    difficulty[
        "delta_role_minus_shared"
    ],
)


fig, ax = plt.subplots(
    figsize=(13.5, 8.5),
    layout="constrained",
)

bg(ax)
style(
    ax,
    "both",
)

ax.axhline(
    0,
    color="black",
    linewidth=1.1,
)

for src in DATASETS:

    q = difficulty[
        difficulty[
            "source"
        ]
        == src
    ]

    sizes = (
        120
        + 350
        * np.clip(
            q[
                "retention"
            ],
            0,
            1.1,
        )
    )

    ax.scatter(
        q[
            "shared_lora_f1"
        ],
        q[
            "delta_role_minus_shared"
        ],
        s=sizes,
        color=DATASET_COLOR[
            src
        ],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.72,
        label=dname(
            src
        ),
        zorder=4,
    )


# trend only as analytical guide
x = difficulty[
    "shared_lora_f1"
].to_numpy()

y = difficulty[
    "delta_role_minus_shared"
].to_numpy()

coef = np.polyfit(
    x,
    y,
    1,
)

xx = np.linspace(
    x.min(),
    x.max(),
    200,
)

ax.plot(
    xx,
    np.polyval(
        coef,
        xx,
    ),
    linestyle="--",
    color="black",
    linewidth=1.2,
    alpha=0.35,
)


# label only most informative directions
labels = pd.concat([
    difficulty.nlargest(
        4,
        "delta_role_minus_shared",
    ),
    difficulty.nsmallest(
        3,
        "delta_role_minus_shared",
    ),
]).drop_duplicates(
    subset=[
        "source",
        "target",
    ]
)

offsets = [
    (
        10,
        12,
    ),
    (
        10,
        -18,
    ),
    (
        -12,
        12,
    ),
    (
        -12,
        -20,
    ),
    (
        10,
        22,
    ),
    (
        -10,
        22,
    ),
    (
        12,
        -28,
    ),
]

for (
    _,
    r,
), (
    dx,
    dy,
) in zip(
    labels.iterrows(),
    offsets,
):

    ax.annotate(
        (
            f"{dname(r['source'])}"
            " → "
            f"{dname(r['target'])}"
        ),
        (
            r[
                "shared_lora_f1"
            ],
            r[
                "delta_role_minus_shared"
            ],
        ),
        xytext=(
            dx,
            dy,
        ),
        textcoords="offset points",
        fontsize=9.5,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="#FFFDF8",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color="black",
            linewidth=0.55,
        ),
    )


ax.set_xlabel(
    "Shared LoRA transfer Macro-F1"
)

ax.set_ylabel(
    "Role Adapter advantage (Δ Macro-F1)"
)

ax.set_title(
    "Where does role-aware adaptation help most?"
)

ax.text(
    0.98,
    0.97,
    (
        f"Spearman ρ = {rho:+.2f}\n"
        f"p = {rho_p:.3g}\n"
        "marker area ∝ retention"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="#FFFDF8",
        edgecolor="black",
        linewidth=0.7,
        alpha=0.94,
    ),
)

ax.legend(
    title="Source",
    ncol=2,
    loc="upper left",
    frameon=True,
    edgecolor="black",
)

save(
    fig,
    "12_transfer_difficulty_response",
)


# ======================================================================
# FIGURE 13
# HEADROOM RECOVERY
#
# 0 = no gain over Shared
# 1 = closes entire Shared→FullFT gap
# >1 = beats Full FT
# ======================================================================

rec = (
    within[
        [
            "full_ft",
            "shared_lora",
            "role_adapter",
        ]
    ]
    .copy()
)

den = (
    rec[
        "full_ft"
    ]
    - rec[
        "shared_lora"
    ]
)

rec[
    "recovery"
] = (
    (
        rec[
            "role_adapter"
        ]
        - rec[
            "shared_lora"
        ]
    )
    / den
)

rec = rec.replace(
    [
        np.inf,
        -np.inf,
    ],
    np.nan,
).dropna(
    subset=[
        "recovery"
    ]
)

rec = rec.reset_index()


fig, ax = plt.subplots(
    figsize=(13.5, 8.3),
    layout="constrained",
)

bg(ax)
style(
    ax,
    "y",
)

models = [
    "inlegalbert",
    "legalbert",
    "deberta",
]

positions = np.arange(
    len(models)
)

for i, model in enumerate(
    models
):

    q = rec[
        rec[
            "model_key"
        ]
        == model
    ][
        "recovery"
    ].to_numpy()

    vp = ax.violinplot(
        [
            q
        ],
        positions=[
            i
        ],
        widths=0.68,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    body = vp[
        "bodies"
    ][0]

    body.set_facecolor(
        vir(
            0.25
            + 0.28
            * i
        )
    )

    body.set_edgecolor(
        "black"
    )

    body.set_linewidth(
        0.8
    )

    body.set_alpha(
        0.33
    )

    rng = np.random.default_rng(
        123
        + i
    )

    jitter = rng.normal(
        0,
        0.045,
        len(q),
    )

    ax.scatter(
        np.full(
            len(q),
            i,
        )
        + jitter,
        q,
        s=70,
        color=vir(
            0.25
            + 0.28
            * i
        ),
        edgecolor="black",
        linewidth=0.5,
        alpha=0.80,
        zorder=4,
    )

    mean = np.mean(
        q
    )

    ax.scatter(
        i,
        mean,
        marker="D",
        s=135,
        color="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=6,
    )

    labelbox(
        ax,
        i,
        mean
        + 0.12,
        f"mean {mean:.2f}",
        fs=10,
    )


ax.axhline(
    0,
    color="black",
    linewidth=0.9,
    alpha=0.45,
)

ax.axhline(
    1,
    color=vir(0.62),
    linestyle="--",
    linewidth=1.7,
)

ax.text(
    0.99,
    1.0,
    " complete Full-FT gap recovery ",
    transform=ax.get_yaxis_transform(),
    ha="right",
    va="bottom",
    fontsize=11,
)

ax.set_xticks(
    positions
)

ax.set_xticklabels(
    [
        BACKBONE_LABEL[x]
        for x in models
    ]
)

ax.set_ylabel(
    "Fraction of Shared-LoRA → Full-FT gap recovered"
)

ax.set_title(
    "How much Full-FT headroom does Role Adapter recover?"
)

ax.text(
    0.02,
    0.03,
    (
        "0 = no improvement over Shared LoRA\n"
        "1 = matches Full FT\n"
        ">1 = exceeds Full FT"
    ),
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="#FFFDF8",
        edgecolor="black",
        linewidth=0.7,
        alpha=0.93,
    ),
)

save(
    fig,
    "13_fullft_headroom_recovery",
)


# ======================================================================
# FIGURE 14
# CIRCULAR DIRECTED TRANSFER NETWORK
# ======================================================================

net = difficulty.copy()


fig, ax = plt.subplots(
    figsize=(11.5, 10.5),
    layout="constrained",
)

bg(
    ax,
    top="#F1F9FF",
    bottom="#FFF8EF",
)

ax.set_aspect(
    "equal"
)

ax.axis(
    "off"
)


angles = np.linspace(
    np.pi / 2,
    np.pi / 2
    - 2 * np.pi,
    len(DATASETS),
    endpoint=False,
)

pos = {}

for ds, ang in zip(
    DATASETS,
    angles,
):

    pos[
        ds
    ] = (
        np.cos(
            ang
        ),
        np.sin(
            ang
        ),
    )


# source robustness for node fill
source_mean = (
    net
    .groupby(
        "source"
    )[
        "role_adapter_f1"
    ]
    .mean()
)

node_norm = Normalize(
    source_mean.min(),
    source_mean.max(),
)


# edges
for _, r in net.iterrows():

    src = r[
        "source"
    ]

    tgt = r[
        "target"
    ]

    x1, y1 = pos[
        src
    ]

    x2, y2 = pos[
        tgt
    ]

    delta = r[
        "delta_role_minus_shared"
    ]

    retention = r[
        "retention"
    ]

    positive = (
        delta >= 0
    )

    color = (
        vir(0.65)
        if positive
        else "#5B2A86"
    )

    width = (
        0.7
        + 16
        * abs(
            delta
        )
    )

    alpha = (
        0.20
        + 0.60
        * min(
            max(
                retention,
                0,
            ),
            1,
        )
    )

    # opposite directions bend oppositely
    sign = (
        1
        if DATASETS.index(
            src
        )
        <
        DATASETS.index(
            tgt
        )
        else -1
    )

    arrow = FancyArrowPatch(
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=width,
        color=color,
        alpha=alpha,
        connectionstyle=(
            f"arc3,rad="
            f"{0.16 * sign}"
        ),
        shrinkA=24,
        shrinkB=24,
        zorder=2,
    )

    ax.add_patch(
        arrow
    )


# nodes
for ds in DATASETS:

    x, y = pos[
        ds
    ]

    value = source_mean[
        ds
    ]

    circle = Circle(
        (
            x,
            y,
        ),
        radius=0.14,
        facecolor=vir(
            node_norm(
                value
            )
        ),
        edgecolor="black",
        linewidth=1.1,
        zorder=5,
        alpha=0.94,
    )

    ax.add_patch(
        circle
    )

    ax.text(
        x,
        y,
        dname(
            ds
        ),
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        zorder=6,
    )


# annotate strongest 5 advantages
top = net.nlargest(
    5,
    "delta_role_minus_shared",
)

for _, r in top.iterrows():

    x1, y1 = pos[
        r[
            "source"
        ]
    ]

    x2, y2 = pos[
        r[
            "target"
        ]
    ]

    mx = (
        x1
        + x2
    ) / 2

    my = (
        y1
        + y2
    ) / 2

    labelbox(
        ax,
        mx,
        my,
        f"{r['delta_role_minus_shared']:+.3f}",
        fs=8.5,
    )


ax.set_xlim(
    -1.45,
    1.45,
)

ax.set_ylim(
    -1.35,
    1.45,
)

ax.set_title(
    "Directed transfer network of Role Adapter gains",
    pad=15,
)

ax.text(
    0.02,
    0.02,
    (
        "edge width ∝ |Role−Shared gain|\n"
        "edge opacity ∝ retention\n"
        "node color ∝ mean outgoing Role Adapter F1"
    ),
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="#FFFDF8",
        edgecolor="black",
        linewidth=0.7,
        alpha=0.94,
    ),
)

legend = [
    Line2D(
        [0],
        [0],
        color=vir(0.65),
        linewidth=3,
        label="Role Adapter gain",
    ),
    Line2D(
        [0],
        [0],
        color="#5B2A86",
        linewidth=3,
        label="Role Adapter loss",
    ),
]

ax.legend(
    handles=legend,
    loc="upper right",
    frameon=True,
    edgecolor="black",
)

save(
    fig,
    "14_directed_transfer_network",
)


# ======================================================================
# FIGURE 15
# RIDGELINE DISTRIBUTIONS OF ROLE-SHARED GAINS
# ======================================================================

gain_sets = {
    "Within-domain":
        np.array(
            [
                r[
                    "role_adapter"
                ]
                - r[
                    "shared_lora"
                ]
                for _,
                r
                in within.iterrows()
            ]
        ),

    "Transfer":
        transfer_role_shared[
            "delta_role_minus_shared"
        ].to_numpy(),

    "Native":
        (
            native_p[
                "role_adapter"
            ]
            - native_p[
                "shared_lora"
            ]
        ).dropna().to_numpy(),
}


allvals = np.concatenate(
    list(
        gain_sets.values()
    )
)

xmin = (
    allvals.min()
    - 0.02
)

xmax = (
    allvals.max()
    + 0.025
)

xx = np.linspace(
    xmin,
    xmax,
    500,
)


fig, ax = plt.subplots(
    figsize=(14.5, 8.2),
    layout="constrained",
)

bg(ax)
style(
    ax,
    "x",
)

bases = {
    "Within-domain": 2.0,
    "Transfer": 1.0,
    "Native": 0.0,
}


for regime in [
    "Within-domain",
    "Transfer",
    "Native",
]:

    vals = gain_sets[
        regime
    ]

    base = bases[
        regime
    ]

    color = REGIME_COLOR[
        regime
    ]

    # KDE only if variance exists
    if (
        len(vals) >= 3
        and np.std(
            vals
        )
        > 1e-8
    ):

        kde = gaussian_kde(
            vals,
            bw_method=0.35,
        )

        dens = kde(
            xx
        )

        dens = (
            dens
            / dens.max()
            * 0.60
        )

        ax.fill_between(
            xx,
            base,
            base
            + dens,
            color=color,
            alpha=0.28,
            edgecolor="black",
            linewidth=0.8,
        )

    # rug
    rng = np.random.default_rng(
        int(
            1000
            + base
            * 10
        )
    )

    jitter = rng.uniform(
        -0.055,
        0.055,
        len(vals),
    )

    ax.scatter(
        vals,
        np.full(
            len(vals),
            base
        )
        + jitter,
        s=55,
        color=color,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.82,
        zorder=5,
    )

    mean = vals.mean()

    wins = int(
        (
            vals > 0
        ).sum()
    )

    labelbox(
        ax,
        mean,
        base
        + 0.69,
        (
            f"mean {mean:+.3f}\n"
            f"{wins}/{len(vals)} positive"
        ),
        fs=10,
    )


ax.axvline(
    0,
    color="black",
    linewidth=1.1,
)

ax.set_yticks(
    [
        0,
        1,
        2,
    ]
)

ax.set_yticklabels(
    [
        "Native",
        "Transfer",
        "Within-domain",
    ]
)

ax.set_ylim(
    -0.25,
    2.92,
)

ax.set_xlim(
    xmin,
    xmax,
)

ax.set_xlabel(
    "Role Adapter − Shared LoRA (Macro-F1)"
)

ax.set_title(
    "Distribution of Role Adapter gains across evaluation regimes"
)

save(
    fig,
    "15_gain_ridgelines",
)


# ======================================================================
# FIGURE 16
# TAXONOMY SENSITIVITY BRIDGE
# Common-7 InLegalBERT vs Native taxonomy
# ======================================================================

common_in = (
    main_cells[
        (
            main_cells[
                "model_key"
            ]
            == "inlegalbert"
        )
        &
        (
            main_cells[
                "dataset_key"
            ].isin(
                [
                    "legaleval",
                    "iltur_cl",
                    "iltur_it",
                ]
            )
        )
    ]
    .pivot_table(
        index="dataset_key",
        columns="method_key",
        values="macro_f1_mean",
    )
)


bridge_rows = []

for ds in [
    "legaleval",
    "iltur_cl",
    "iltur_it",
]:

    common_gain = (
        common_in.loc[
            ds,
            "role_adapter",
        ]
        - common_in.loc[
            ds,
            "shared_lora",
        ]
    )

    native_gain = (
        native_p.loc[
            ds,
            "role_adapter",
        ]
        - native_p.loc[
            ds,
            "shared_lora",
        ]
    )

    bridge_rows.append({
        "dataset":
            ds,

        "common7":
            common_gain,

        "native":
            native_gain,

        "shift":
            (
                native_gain
                - common_gain
            ),
    })


bridge = pd.DataFrame(
    bridge_rows
)


fig, ax = plt.subplots(
    figsize=(13.5, 7.5),
    layout="constrained",
)

bg(ax)
style(
    ax,
    "x",
)

yy = np.arange(
    len(
        bridge
    )
)


for y, (
    _,
    r,
) in zip(
    yy,
    bridge.iterrows(),
):

    lo = min(
        r[
            "common7"
        ],
        r[
            "native"
        ],
    )

    hi = max(
        r[
            "common7"
        ],
        r[
            "native"
        ],
    )

    # broad translucent capsule
    ax.plot(
        [
            lo,
            hi,
        ],
        [
            y,
            y,
        ],
        linewidth=13,
        color=DATASET_COLOR[
            r[
                "dataset"
            ]
        ],
        alpha=0.16,
        solid_capstyle="round",
        zorder=2,
    )

    ax.plot(
        [
            lo,
            hi,
        ],
        [
            y,
            y,
        ],
        linewidth=1.0,
        color="black",
        alpha=0.55,
        zorder=3,
    )

    ax.scatter(
        r[
            "common7"
        ],
        y,
        s=180,
        marker="o",
        color=vir(0.30),
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )

    ax.scatter(
        r[
            "native"
        ],
        y,
        s=180,
        marker="D",
        color=vir(0.78),
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
    )

    labelbox(
        ax,
        hi
        + 0.007,
        y,
        (
            f"shift "
            f"{r['shift']:+.3f}"
        ),
        fs=10,
        ha="left",
    )


ax.axvline(
    0,
    color="black",
    linewidth=1.0,
)

ax.set_yticks(
    yy
)

ax.set_yticklabels(
    [
        dname(x)
        for x in bridge[
            "dataset"
        ]
    ]
)

ax.invert_yaxis()

ax.set_xlabel(
    "Role Adapter − Shared LoRA (Macro-F1)"
)

ax.set_title(
    "How much does taxonomy harmonization change the observed gain?"
)

legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=10,
        markerfacecolor=vir(0.30),
        markeredgecolor="black",
        label="Common-7",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        linestyle="none",
        markersize=10,
        markerfacecolor=vir(0.78),
        markeredgecolor="black",
        label="Native taxonomy",
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
    "16_taxonomy_sensitivity_bridge",
)


# ======================================================================
# SUMMARY TABLES
# ======================================================================

difficulty.to_csv(
    OUT
    / "transfer_difficulty_response.csv",
    index=False,
)

rec.to_csv(
    OUT
    / "headroom_recovery.csv",
    index=False,
)

asym.to_csv(
    OUT
    / "iltur_directional_asymmetry.csv",
    index=False,
)

forest.to_csv(
    OUT
    / "effect_forest_v4.csv",
    index=False,
)

atomic.to_csv(
    OUT
    / "cross_regime_gain_cells.csv",
    index=False,
)

bridge.to_csv(
    OUT
    / "taxonomy_sensitivity.csv",
    index=False,
)


# ======================================================================
# GALLERY
# ======================================================================

pngs = sorted(
    OUT.glob(
        "*.png"
    )
)

cols = 3

rows = math.ceil(
    len(
        pngs
    )
    / cols
)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(
        19,
        5.4
        * rows,
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
    len(
        pngs
    ):
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


# ======================================================================
# PRINT
# ======================================================================

print()
print("=" * 110)
print("V4 PATCH + NEW FIGURES COMPLETE")
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
    OUT
    / "00_gallery.png"
)
