from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ============================================================
# CONFIG
# ============================================================

ROOT = Path.home() / "TaxFacto"

OUT = ROOT / "results/editorial_figures"
OUT.mkdir(parents=True, exist_ok=True)

ADD_ANALYSIS = ROOT / "results/additional/analysis"
ADD_BOOT = ROOT / "results/additional/bootstrap"
ADD_CORE8 = ROOT / "results/additional/fair_core8"

FINAL_RUNS = ROOT / "results/final_common7"
FINAL_ANALYSIS = ROOT / "results/final_analysis"

RNG = np.random.default_rng(42)

COMMON4 = ["full_ft", "role_adapter", "category_lora", "shared_lora"]
COMMON3 = ["full_ft", "role_adapter", "shared_lora"]
WITHIN_EXTRA = ["shared_lora_widehead", "frozen"]

METHOD_LABEL = {
    "full_ft": "Full FT",
    "role_adapter": "Role Adapter",
    "category_lora": "Category LoRA",
    "shared_lora": "Shared LoRA",
    "shared_lora_widehead": "Wide-head",
    "frozen": "Frozen",
}

DATASET_LABEL = {
    "legaleval": "LegalEval",
    "marro_india": "MARRO-IN",
    "marro_uk": "MARRO-UK",
    "iltur_cl": "IL-TUR CL",
    "iltur_it": "IL-TUR IT",
}

VIR = plt.cm.viridis
CIV = plt.cm.cividis

METHOD_COLOR = {
    "full_ft": VIR(0.92),
    "role_adapter": VIR(0.62),
    "category_lora": VIR(0.46),
    "shared_lora": VIR(0.20),
    "shared_lora_widehead": CIV(0.35),
    "frozen": (0.75, 0.77, 0.80, 1.0),
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 25,
    "axes.labelsize": 21,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.titlesize": 28,
    "axes.linewidth": 1.2,
    "savefig.dpi": 240,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# UTILS
# ============================================================

def titleprint(s):
    print("\n" + "=" * 110)
    print(s)
    print("=" * 110)

def pretty_method(x):
    return METHOD_LABEL.get(x, x)

def pretty_dataset(x):
    return DATASET_LABEL.get(x, x)

def ensure_cols(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"{name}: missing columns {miss}")

def axis_frame(ax):
    ax.set_facecolor("#FBFAF7")
    for sp in ax.spines.values():
        sp.set_color("black")
        sp.set_linewidth(1.0)

def gradient_bg(ax, top="#F7FBFF", bottom="#FFF8F0", alpha=0.95):
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list("bg", [top, bottom])
    ax.imshow(
        grad,
        extent=(0, 1, 0, 1),
        transform=ax.transAxes,
        aspect="auto",
        cmap=cmap,
        alpha=alpha,
        zorder=0,
    )

def add_panel_label(ax, txt):
    ax.text(
        0.012, 0.988, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=18, fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#FFFDF8",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.94,
        ),
        zorder=50,
    )

def bbox_text(ax, x, y, s, fontsize=12, ha="center", va="center", fc="#FFFDF8", alpha=0.94):
    ax.text(
        x, y, s,
        ha=ha, va=va,
        fontsize=fontsize,
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor=fc,
            edgecolor="black",
            linewidth=0.85,
            alpha=alpha,
        ),
        zorder=20,
        color="black",
    )

def note_box(ax, xy_axes, text, width=0.28, height=0.12, fc="#F8FBFF"):
    x, y = xy_axes
    patch = FancyBboxPatch(
        (x, y), width, height,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=fc,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.95,
        zorder=5,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.015, y + height - 0.015, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, color="black",
        zorder=6,
    )

def alias_backbone(model):
    s = str(model).lower()
    if "inlegalbert" in s:
        return "InLegalBERT"
    if "legalbert" in s and "inlegalbert" not in s:
        return "LegalBERT"
    if "deberta" in s:
        return "DeBERTa-v3"
    return Path(str(model)).name

def bootstrap_mean_ci(values, n_boot=5000, alpha=0.05, rng=None):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng(42)
    means = np.empty(n_boot, dtype=float)
    n = len(vals)
    for i in range(n_boot):
        sample = vals[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(vals.mean()), float(lo), float(hi)

def pairwise_from_pivot(piv, methods):
    methods = [m for m in methods if m in piv.columns]
    n = len(methods)
    mean_delta = np.zeros((n, n), dtype=float)
    wins = np.empty((n, n), dtype=object)
    totals = np.zeros((n, n), dtype=int)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j:
                mean_delta[i, j] = 0.0
                wins[i, j] = "—"
                totals[i, j] = 0
            else:
                d = (piv[a] - piv[b]).dropna()
                mean_delta[i, j] = d.mean()
                wa = int((d > 0).sum())
                totals[i, j] = int(len(d))
                wins[i, j] = f"{wa}/{len(d)}"
    return methods, mean_delta, wins, totals

def load_csv_or_fail(fn):
    if not fn.exists():
        raise FileNotFoundError(fn)
    return pd.read_csv(fn)

# ============================================================
# LOAD / BUILD DATA
# ============================================================

def build_within_from_raw():
    titleprint("BUILD WITHIN-DOMAIN SUMMARY FROM RAW FINAL RUNS")
    files = sorted(FINAL_RUNS.rglob("run_summary.json"))
    if not files:
        raise RuntimeError("No run_summary.json found under results/final_common7")

    rows = []
    accepted = set(COMMON4 + WITHIN_EXTRA)

    for fn in files:
        with open(fn) as f:
            x = json.load(f)

        method = x.get("method")
        if method not in accepted:
            continue

        # comparable main grid
        if method in {"shared_lora", "category_lora", "role_adapter", "shared_lora_widehead"}:
            if int(x.get("rank", -1)) != 8:
                continue
            if int(x.get("alpha", -1)) != 16:
                continue

        dataset = x.get("dataset")
        model = x.get("model")
        seed = x.get("seed")

        if dataset is None or model is None or seed is None:
            continue

        rows.append({
            "dataset": dataset,
            "backbone": alias_backbone(model),
            "method": method,
            "seed": int(seed),
            "macro_f1": float(x.get("macro_f1", np.nan)),
            "weighted_f1": float(x.get("weighted_f1", np.nan)),
            "accuracy": float(x.get("accuracy", np.nan)),
            "train_seconds": float(x.get("train_seconds", np.nan)),
            "peak_gpu_memory_gb": float(x.get("peak_gpu_memory_gb", np.nan)),
            "parameters_trainable": float(x.get("parameters_trainable", np.nan)),
            "trainable_pct": float(x.get("trainable_pct", np.nan)),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No matching comparable within-domain runs extracted.")

    raw_out = OUT / "within_main_runs_extracted.csv"
    df.to_csv(raw_out, index=False)

    cell = (
        df.groupby(["dataset", "backbone", "method"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            accuracy_mean=("accuracy", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            peak_gpu_memory_gb=("peak_gpu_memory_gb", "mean"),
            parameters_trainable=("parameters_trainable", "mean"),
            trainable_pct=("trainable_pct", "mean"),
        )
    )
    cell.to_csv(OUT / "within_main_cells.csv", index=False)

    method = (
        cell.groupby("method", as_index=False)
        .agg(
            mean_macro_f1=("macro_f1_mean", "mean"),
            std_macro_f1=("macro_f1_mean", "std"),
            mean_weighted_f1=("weighted_f1_mean", "mean"),
            mean_accuracy=("accuracy_mean", "mean"),
            mean_train_seconds=("train_seconds_mean", "mean"),
            mean_trainable_pct=("trainable_pct", "mean"),
            mean_trainable_params=("parameters_trainable", "mean"),
            mean_peak_vram=("peak_gpu_memory_gb", "mean"),
        )
        .sort_values("mean_macro_f1", ascending=False)
    )
    method.to_csv(OUT / "within_main_method_summary.csv", index=False)

    print(cell.head())
    print("\nSaved:", raw_out)
    return df, cell, method

def load_transfer_data():
    x = load_csv_or_fail(ADD_ANALYSIS / "transfer_summary.csv")
    ensure_cols(x, ["source", "target", "method", "macro_f1_mean", "retention", "transfer_gap"], "transfer_summary")
    return x

def load_native_data():
    x = load_csv_or_fail(ADD_ANALYSIS / "native_summary.csv")
    ensure_cols(x, ["dataset", "method", "macro_f1_mean"], "native_summary")
    return x

def load_bootstrap_global():
    x = load_csv_or_fail(ADD_BOOT / "hierarchical_bootstrap_global.csv")
    ensure_cols(x, ["method_a", "method_b", "point_mean_delta", "ci95_low", "ci95_high"], "hierarchical_bootstrap_global")
    return x

def load_core8():
    summary = load_csv_or_fail(ADD_CORE8 / "transfer_core8_method_summary.csv")
    pivot = load_csv_or_fail(ADD_CORE8 / "transfer_core8_pivot.csv")
    pairs = load_csv_or_fail(ADD_CORE8 / "transfer_core8_pairwise.csv")
    role_focus = load_csv_or_fail(ADD_CORE8 / "transfer_core8_role_focus.csv")
    return summary, pivot, pairs, role_focus

within_runs, within_cell, within_method = build_within_from_raw()
transfer = load_transfer_data()
native = load_native_data()
boot_global = load_bootstrap_global()
core8_summary, core8_pivot, core8_pairs, core8_role_focus = load_core8()

# ============================================================
# DERIVED TABLES
# ============================================================

# Within-domain common4 pivot
within_pivot4 = (
    within_cell[within_cell["method"].isin(COMMON4)]
    .pivot_table(index=["dataset", "backbone"], columns="method", values="macro_f1_mean")
    .dropna()
    .reset_index()
)

# Within common3 for regime ranking
within_pivot3 = (
    within_cell[within_cell["method"].isin(COMMON3)]
    .pivot_table(index=["dataset", "backbone"], columns="method", values="macro_f1_mean")
    .dropna()
    .reset_index()
)

# Transfer core8 pivot
transfer_core8 = core8_pivot.copy()

# Native pivot
native_pivot3 = (
    native[native["method"].isin(COMMON3)]
    .pivot_table(index="dataset", columns="method", values="macro_f1_mean")
    .dropna()
    .reset_index()
)

# Pairwise matrices
methods_within4, within_delta4, within_wins4, within_tot4 = pairwise_from_pivot(within_pivot4[COMMON4], COMMON4)
methods_transfer4, transfer_delta4, transfer_wins4, transfer_tot4 = pairwise_from_pivot(transfer_core8[COMMON4], COMMON4)
methods_native3, native_delta3, native_wins3, native_tot3 = pairwise_from_pivot(native_pivot3[COMMON3], COMMON3)

# Transfer atlas matrices
role_t = (
    transfer[transfer["method"] == "role_adapter"]
    .pivot(index="source", columns="target", values="macro_f1_mean")
    .loc[list(DATASET_LABEL.keys()), list(DATASET_LABEL.keys())]
)

shared_t = (
    transfer[transfer["method"] == "shared_lora"]
    .pivot(index="source", columns="target", values="macro_f1_mean")
    .loc[list(DATASET_LABEL.keys()), list(DATASET_LABEL.keys())]
)

role_ret = (
    transfer[transfer["method"] == "role_adapter"]
    .pivot(index="source", columns="target", values="retention")
    .loc[list(DATASET_LABEL.keys()), list(DATASET_LABEL.keys())]
)

delta_t = role_t - shared_t

# IL-TUR controlled transfer
iltur_ctrl = transfer[
    (
        ((transfer["source"] == "iltur_cl") & (transfer["target"] == "iltur_it")) |
        ((transfer["source"] == "iltur_it") & (transfer["target"] == "iltur_cl"))
    )
    &
    (transfer["method"].isin(COMMON4))
].copy()

# Regime ranks
def average_ranks_from_pivot(piv, methods):
    vals = piv[methods].copy()
    ranks = vals.rank(axis=1, ascending=False, method="average")
    out = pd.DataFrame({
        "method": methods,
        "rank_mean": ranks.mean(axis=0).values,
        "rank_std": ranks.std(axis=0).fillna(0.0).values,
    })
    return out

rank_within = average_ranks_from_pivot(within_pivot3, COMMON3)
rank_transfer = average_ranks_from_pivot(transfer_core8, COMMON3)
rank_native = average_ranks_from_pivot(native_pivot3, COMMON3)

rank_within["regime"] = "Within-domain"
rank_transfer["regime"] = "Transfer (core-8)"
rank_native["regime"] = "Native taxonomy"

rank_all = pd.concat([rank_within, rank_transfer, rank_native], ignore_index=True)

# Forest data
forest_rows = []

def add_forest_row(block, comparator, delta, lo, hi, note):
    forest_rows.append({
        "block": block,
        "label": comparator,
        "delta": float(delta),
        "lo": float(lo),
        "hi": float(hi),
        "note": note,
    })

# Block A: within-domain from hierarchical bootstrap, convert to role - other
for other in ["shared_lora", "shared_lora_widehead", "category_lora", "full_ft", "frozen"]:
    q = boot_global[(boot_global["method_a"] == other) & (boot_global["method_b"] == "role_adapter")]
    if len(q) == 1:
        r = q.iloc[0]
        delta = -float(r["point_mean_delta"])  # role - other
        lo = -float(r["ci95_high"])
        hi = -float(r["ci95_low"])
        add_forest_row("Within-domain", pretty_method(other), delta, lo, hi, "doc bootstrap")

# Block B: transfer core-8, paired bootstrap over 8 directions
for other in ["shared_lora", "category_lora", "full_ft"]:
    d = (transfer_core8["role_adapter"] - transfer_core8[other]).to_numpy()
    m, lo, hi = bootstrap_mean_ci(d, n_boot=8000, rng=RNG)
    wins = int((d > 0).sum())
    add_forest_row("Transfer (core-8)", pretty_method(other), m, lo, hi, f"{wins}/8 wins")

# Block C: native taxonomy, paired bootstrap over datasets
for other in ["shared_lora", "full_ft"]:
    d = (native_pivot3["role_adapter"] - native_pivot3[other]).to_numpy()
    m, lo, hi = bootstrap_mean_ci(d, n_boot=8000, rng=RNG)
    wins = int((d > 0).sum())
    add_forest_row("Native taxonomy", pretty_method(other), m, lo, hi, f"{wins}/3 wins")

forest_df = pd.DataFrame(forest_rows)

forest_df.to_csv(OUT / "forest_effects_table.csv", index=False)
rank_all.to_csv(OUT / "rank_regimes_table.csv", index=False)

# ============================================================
# COLORMAPS
# ============================================================

CMAP_DELTA = LinearSegmentedColormap.from_list(
    "delta_div",
    [
        (0.00, "#482878"),  # viridis purple
        (0.22, "#3E6BA5"),
        (0.48, "#F8F7F2"),
        (0.72, "#2FB47C"),
        (1.00, "#FDE725"),  # viridis yellow
    ]
)

CMAP_ATLAS = LinearSegmentedColormap.from_list(
    "atlas",
    [
        (0.00, "#482878"),
        (0.30, "#355F8D"),
        (0.50, "#F8F7F2"),
        (0.72, "#1FA187"),
        (1.00, "#A0DA39"),
    ]
)

# ============================================================
# FIGURE 1 — PARETO WITH INSET
# ============================================================

titleprint("FIGURE 1 — PARETO WITH PEFT ZOOM")

fig = plt.figure(figsize=(16.2, 10.2))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0, 0])
gradient_bg(ax, top="#EFF8FF", bottom="#FFF7EF")
axis_frame(ax)
ax.grid(True, which="major", axis="both", alpha=0.18, color="black", linewidth=0.75, zorder=1)

methods_f1 = ["frozen", "shared_lora", "category_lora", "role_adapter", "shared_lora_widehead", "full_ft"]
pm = within_method[within_method["method"].isin(methods_f1)].copy()

# log x solves the collapse on the left
ax.set_xscale("log")
ax.set_xlabel("Trainable parameters (%)  [log scale]")
ax.set_ylabel("Mean within-domain Macro-F1")
ax.set_title("Accuracy–efficiency Pareto map with PEFT zoom", pad=16)

# ranges
xmin = max(0.003, pm["mean_trainable_pct"].min() * 0.7)
xmax = pm["mean_trainable_pct"].max() * 1.5
ax.set_xlim(xmin, xmax)
ax.set_ylim(pm["mean_macro_f1"].min() - 0.03, pm["mean_macro_f1"].max() + 0.08)

# nice ticks
ax.set_xticks([0.005, 0.01, 0.1, 1, 10, 100])
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.ticklabel_format(axis='x', style='plain')

# PEFT region highlight
ax.axvspan(0.004, 1.0, color="#DDEFFA", alpha=0.18, zorder=0)
ax.axvspan(50, 130, color="#FFF2CF", alpha=0.12, zorder=0)

# plot points
for _, r in pm.iterrows():
    m = r["method"]
    x = float(r["mean_trainable_pct"])
    y = float(r["mean_macro_f1"])
    yerr = float(r["std_macro_f1"]) if pd.notna(r["std_macro_f1"]) else 0.0
    seconds = float(r["mean_train_seconds"])
    size = 240 + 11 * math.sqrt(max(seconds, 1))

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=5,
        zorder=2,
        alpha=0.85,
    )

    ax.scatter(
        x, y,
        s=size,
        color=METHOD_COLOR[m],
        edgecolor="black",
        linewidth=1.0,
        alpha=0.78,
        zorder=4,
    )

# annotations with manual offsets to avoid overlap
offsets = {
    "frozen": (16, -6),
    "shared_lora": (12, -26),
    "category_lora": (12, 18),
    "role_adapter": (12, 38),
    "shared_lora_widehead": (12, 0),
    "full_ft": (-44, 2),
}

for _, r in pm.iterrows():
    m = r["method"]
    x = float(r["mean_trainable_pct"])
    y = float(r["mean_macro_f1"])
    dx, dy = offsets.get(m, (10, 10))
    txt = (
        f"{pretty_method(m)}\n"
        f"F1={y:.4f}\n"
        f"{x:.3f}%"
    )
    ax.annotate(
        txt,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.26",
            facecolor="#FFFDF8",
            edgecolor="black",
            linewidth=0.9,
            alpha=0.95,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color="black",
            linewidth=0.9,
            alpha=0.7,
            shrinkA=0,
            shrinkB=4,
        ),
        zorder=10,
    )

# inset: PEFT zoom
axins = inset_axes(ax, width="44%", height="42%", loc="lower left",
                   bbox_to_anchor=(0.10, 0.08, 0.88, 0.88),
                   bbox_transform=ax.transAxes, borderpad=0)
gradient_bg(axins, top="#F2FAFF", bottom="#FFF9F2", alpha=0.98)
axis_frame(axins)
axins.grid(True, axis="both", alpha=0.14, color="black", linewidth=0.6)

p_zoom = pm[pm["method"] != "full_ft"].copy()
axins.set_xscale("log")
axins.set_xlim(0.004, 0.6)
axins.set_ylim(p_zoom["mean_macro_f1"].min() - 0.02, p_zoom["mean_macro_f1"].max() + 0.04)

for _, r in p_zoom.iterrows():
    m = r["method"]
    x = float(r["mean_trainable_pct"])
    y = float(r["mean_macro_f1"])
    yerr = float(r["std_macro_f1"]) if pd.notna(r["std_macro_f1"]) else 0.0
    size = 170 + 8 * math.sqrt(max(float(r["mean_train_seconds"]), 1))
    axins.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=4, zorder=2)
    axins.scatter(
        x, y, s=size,
        color=METHOD_COLOR[m],
        edgecolor="black",
        linewidth=0.9,
        alpha=0.84,
        zorder=4,
    )

small_offsets = {
    "frozen": (10, -10),
    "shared_lora": (10, -18),
    "category_lora": (10, 8),
    "role_adapter": (10, 22),
    "shared_lora_widehead": (10, 2),
}
for _, r in p_zoom.iterrows():
    m = r["method"]
    x = float(r["mean_trainable_pct"])
    y = float(r["mean_macro_f1"])
    dx, dy = small_offsets[m]
    axins.annotate(
        pretty_method(m),
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="#FFFDF8",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.95,
        ),
        arrowprops=dict(arrowstyle="-", color="black", linewidth=0.8),
        zorder=8,
    )
axins.set_xticks([0.005, 0.01, 0.1])
axins.get_xaxis().set_major_formatter(ScalarFormatter())
axins.set_title("PEFT zoom", fontsize=15, pad=6)

note_box(
    ax, (0.66, 0.08),
    "Bubble area ∝ mean train time\nError bars: std across 15 within-domain\n(dataset × backbone) cells",
    width=0.30, height=0.12, fc="#F8FBFF"
)
add_panel_label(ax, "Fig. 1")

fig.tight_layout()
fig.savefig(OUT / "fig01_pareto_inset.png", bbox_inches="tight")
fig.savefig(OUT / "fig01_pareto_inset.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 2 — DOMINANCE HEATMAP TRIPTYCH
# ============================================================

titleprint("FIGURE 2 — DOMINANCE HEATMAP TRIPTYCH")

fig = plt.figure(figsize=(19.5, 6.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.95], wspace=0.18)

def plot_dominance(ax, methods, mat, wins, regime_title, total_note):
    gradient_bg(ax, top="#F3FAFF", bottom="#FFF8EF")
    axis_frame(ax)

    n = len(methods)
    vmax = max(0.001, np.max(np.abs(mat[np.isfinite(mat)])))
    im = ax.imshow(mat, cmap=CMAP_DELTA, vmin=-vmax, vmax=vmax, zorder=1, alpha=0.93)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([pretty_method(m) for m in methods], rotation=28, ha="right")
    ax.set_yticklabels([pretty_method(m) for m in methods])

    ax.set_title(regime_title, pad=12)

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.85, alpha=0.52)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            if i == j:
                bbox_text(ax, j, i, "—", fontsize=14, fc="#F7F6F2")
            else:
                txt = f"{mat[i,j]:+.3f}\n{wins[i,j]}"
                bbox_text(ax, j, i, txt, fontsize=11, fc="#FFFDF8")

    note_box(ax, (0.03, 0.02), total_note, width=0.42, height=0.12, fc="#F8FBFF")
    return im

ax1 = fig.add_subplot(gs[0, 0])
im1 = plot_dominance(
    ax1, methods_within4, within_delta4, within_wins4,
    "Within-domain dominance",
    "Cell = mean Δ Macro-F1\nand wins / total over 15\n(dataset × backbone) cells"
)
add_panel_label(ax1, "Fig. 2a")

ax2 = fig.add_subplot(gs[0, 1])
im2 = plot_dominance(
    ax2, methods_transfer4, transfer_delta4, transfer_wins4,
    "Transfer dominance (fair core-8)",
    "Cell = mean Δ Macro-F1\nand wins / total over 8\nshared transfer directions"
)
add_panel_label(ax2, "Fig. 2b")

ax3 = fig.add_subplot(gs[0, 2])
im3 = plot_dominance(
    ax3, methods_native3, native_delta3, native_wins3,
    "Native-taxonomy dominance",
    "Cell = mean Δ Macro-F1\nand wins / total over 3\nnative datasets"
)
add_panel_label(ax3, "Fig. 2c")

cbar = fig.colorbar(im2, ax=[ax1, ax2, ax3], shrink=0.88, pad=0.012)
cbar.set_label("Row method minus column method")
cbar.outline.set_edgecolor("black")
cbar.outline.set_linewidth(1.0)

fig.suptitle("Pairwise dominance structure across evaluation regimes", y=1.02, fontsize=27)
fig.tight_layout()
fig.savefig(OUT / "fig02_dominance_triptych.png", bbox_inches="tight")
fig.savefig(OUT / "fig02_dominance_triptych.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 3 — TRANSFER ATLAS
# ============================================================

titleprint("FIGURE 3 — TRANSFER ATLAS")

fig = plt.figure(figsize=(12.8, 10.8))
ax = fig.add_subplot(111)
gradient_bg(ax, top="#F0F9FF", bottom="#FFF8EF")
axis_frame(ax)

sources = list(role_t.index)
targets = list(role_t.columns)

nrow, ncol = len(sources), len(targets)
ax.set_xlim(-0.5, ncol - 0.5)
ax.set_ylim(nrow - 0.5, -0.5)

ax.set_xticks(range(ncol))
ax.set_yticks(range(nrow))
ax.set_xticklabels([pretty_dataset(x) for x in targets], rotation=30, ha="right")
ax.set_yticklabels([pretty_dataset(x) for x in sources])

ax.set_xlabel("Target domain")
ax.set_ylabel("Source domain")
ax.set_title("Transfer atlas: Role Adapter advantage and retention", pad=18)

vmax = max(abs(delta_t.min().min()), abs(delta_t.max().max()))
norm = Normalize(vmin=-vmax, vmax=vmax)

# draw custom rounded cells
for i, src in enumerate(sources):
    for j, tgt in enumerate(targets):
        d = float(delta_t.loc[src, tgt])
        r = float(role_ret.loc[src, tgt])
        absf = float(role_t.loc[src, tgt])

        color = CMAP_ATLAS(norm(d))
        patch = FancyBboxPatch(
            (j - 0.46, i - 0.46), 0.92, 0.92,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor="black",
            linewidth=1.0 if src != tgt else 1.5,
            alpha=0.88,
            zorder=2,
        )
        ax.add_patch(patch)

        # overlay circle with area proportional to absolute transfer F1
        radius = 0.10 + 0.18 * (absf - role_t.values.min()) / (role_t.values.max() - role_t.values.min() + 1e-9)
        circ = Circle(
            (j, i), radius=radius,
            facecolor=(1, 1, 1, 0.18),
            edgecolor="black",
            linewidth=0.75,
            zorder=3,
        )
        ax.add_patch(circ)

        bbox_text(ax, j, i - 0.12, f"Δ {d:+.3f}", fontsize=12, fc="#FFFDF8")
        bbox_text(ax, j, i + 0.17, f"ret {r:.2f}", fontsize=11, fc="#F8FBFF")

ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=0.8, alpha=0.45)
ax.tick_params(which="minor", bottom=False, left=False)

sm = mpl.cm.ScalarMappable(cmap=CMAP_ATLAS, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.86, pad=0.02)
cbar.set_label("Role Adapter − Shared LoRA (Macro-F1)")
cbar.outline.set_edgecolor("black")
cbar.outline.set_linewidth(1.0)

# bubble legend
note_box(
    ax, (0.02, 0.01),
    "Cell color = Role Adapter − Shared LoRA\nTop number = Δ Macro-F1\nBottom number = retention\nCircle size ∝ absolute Role Adapter F1",
    width=0.40, height=0.15, fc="#F8FBFF"
)
add_panel_label(ax, "Fig. 3")

fig.tight_layout()
fig.savefig(OUT / "fig03_transfer_atlas.png", bbox_inches="tight")
fig.savefig(OUT / "fig03_transfer_atlas.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 4 — EFFECT FOREST
# ============================================================

titleprint("FIGURE 4 — EFFECT FOREST")

fig = plt.figure(figsize=(14.8, 10.4))
ax = fig.add_subplot(111)
gradient_bg(ax, top="#F2FAFF", bottom="#FFF9F0")
axis_frame(ax)
ax.grid(True, axis="x", alpha=0.18, color="black", linewidth=0.8, zorder=1)
ax.axvline(0, color="black", linewidth=1.25, alpha=0.9, zorder=2)

# prepare rows grouped by block
block_order = ["Within-domain", "Transfer (core-8)", "Native taxonomy"]
forest_df["block"] = pd.Categorical(forest_df["block"], categories=block_order, ordered=True)
forest_df = forest_df.sort_values(["block"]).copy()

y_positions = []
block_ranges = {}
cursor = 0
for block in block_order:
    sub = forest_df[forest_df["block"] == block]
    start = cursor
    for _ in range(len(sub)):
        y_positions.append(cursor)
        cursor += 1
    end = cursor - 1
    block_ranges[block] = (start, end)
    cursor += 1  # spacer

forest_df["y"] = y_positions

# background bands by block
band_colors = {
    "Within-domain": (0.85, 0.94, 0.98, 0.18),
    "Transfer (core-8)": (0.90, 0.97, 0.92, 0.18),
    "Native taxonomy": (0.99, 0.94, 0.84, 0.18),
}
for block in block_order:
    sub = forest_df[forest_df["block"] == block]
    if len(sub) == 0:
        continue
    y0 = sub["y"].min() - 0.5
    h = sub["y"].max() - sub["y"].min() + 1.0
    rect = Rectangle(
        (0, y0), 1, h,
        transform=ax.get_yaxis_transform(),
        facecolor=band_colors[block],
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(rect)

# plot rows
for _, r in forest_df.iterrows():
    y = r["y"]
    delta = float(r["delta"])
    lo = float(r["lo"])
    hi = float(r["hi"])
    color = METHOD_COLOR["role_adapter"] if delta >= 0 else METHOD_COLOR["full_ft"]

    ax.plot([lo, hi], [y, y], color="black", linewidth=2.0, zorder=3)
    ax.scatter(
        [delta], [y], s=190,
        color=color,
        edgecolor="black",
        linewidth=1.0,
        zorder=4,
        alpha=0.88,
    )
    bbox_text(
        ax,
        hi + 0.010,
        y,
        f"{delta:+.3f}\n[{lo:+.3f}, {hi:+.3f}]",
        fontsize=11,
        ha="left",
        fc="#FFFDF8",
    )
    bbox_text(
        ax,
        -0.185,
        y,
        r["note"],
        fontsize=10,
        ha="left",
        fc="#F8FBFF",
    )

# y labels
ax.set_yticks(forest_df["y"])
ax.set_yticklabels([f"Role Adapter − {lab}" for lab in forest_df["label"]])

# block headers on left
for block in block_order:
    sub = forest_df[forest_df["block"] == block]
    if len(sub) == 0:
        continue
    y_mid = 0.5 * (sub["y"].min() + sub["y"].max())
    ax.text(
        -0.24, y_mid, block,
        transform=ax.get_yaxis_transform(),
        rotation=90,
        va="center", ha="center",
        fontsize=16, fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#FFFDF8",
            edgecolor="black",
            linewidth=0.9,
            alpha=0.95,
        ),
        zorder=5,
    )

ax.set_xlabel("Effect size: Role Adapter minus comparator (Macro-F1)")
ax.set_title("Effect sizes with uncertainty across evaluation regimes", pad=16)

xmin = min(forest_df["lo"].min(), -0.18) - 0.03
xmax = max(forest_df["hi"].max(), 0.08) + 0.12
ax.set_xlim(xmin, xmax)
ax.set_ylim(forest_df["y"].max() + 1.0, -1.0)

note_box(
    ax, (0.66, 0.03),
    "Within-domain uses hierarchical\ndocument-level bootstrap.\nTransfer/native use paired\nbootstrap over atomic cells.",
    width=0.28, height=0.14, fc="#F8FBFF"
)
add_panel_label(ax, "Fig. 4")

fig.tight_layout()
fig.savefig(OUT / "fig04_effect_forest.png", bbox_inches="tight")
fig.savefig(OUT / "fig04_effect_forest.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 5 — RANK STABILITY BUMP CHART
# ============================================================

titleprint("FIGURE 5 — RANK STABILITY BUMP CHART")

fig = plt.figure(figsize=(13.8, 7.8))
ax = fig.add_subplot(111)
gradient_bg(ax, top="#F1FAFF", bottom="#FFF8EF")
axis_frame(ax)
ax.grid(True, axis="y", alpha=0.18, color="black", linewidth=0.8, zorder=1)

regimes = ["Within-domain", "Transfer (core-8)", "Native taxonomy"]
xpos = np.arange(len(regimes))

for method in COMMON3:
    sub = rank_all[rank_all["method"] == method].set_index("regime").loc[regimes].reset_index()
    y = sub["rank_mean"].to_numpy()
    ys = sub["rank_std"].to_numpy()

    # translucent ribbon
    ax.fill_between(
        xpos, y - 0.35 * ys, y + 0.35 * ys,
        color=METHOD_COLOR[method],
        alpha=0.18,
        zorder=2,
    )
    ax.plot(
        xpos, y,
        color=METHOD_COLOR[method],
        linewidth=3.2,
        marker="o",
        markersize=13,
        markeredgecolor="black",
        markeredgewidth=1.0,
        alpha=0.95,
        zorder=4,
        label=pretty_method(method),
    )
    for x, yy in zip(xpos, y):
        bbox_text(ax, x, yy - 0.14, f"{yy:.2f}", fontsize=11, fc="#FFFDF8")

    ax.text(
        xpos[-1] + 0.08, y[-1],
        pretty_method(method),
        va="center", ha="left",
        fontsize=14,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="#FFFDF8",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.94,
        ),
        zorder=5,
    )

ax.set_xticks(xpos)
ax.set_xticklabels(regimes)
ax.set_yticks([1, 2, 3])
ax.set_ylim(3.35, 0.65)   # rank 1 at top
ax.set_ylabel("Average rank (lower is better)")
ax.set_title("Rank stability across evaluation regimes", pad=15)

note_box(
    ax, (0.02, 0.02),
    "Ranks computed per atomic comparison cell:\nWithin = 15 dataset×backbone cells\nTransfer = 8 shared directions\nNative = 3 native datasets",
    width=0.36, height=0.16, fc="#F8FBFF"
)
add_panel_label(ax, "Fig. 5")

fig.tight_layout()
fig.savefig(OUT / "fig05_rank_stability.png", bbox_inches="tight")
fig.savefig(OUT / "fig05_rank_stability.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 6 — IL-TUR ASYMMETRY
# ============================================================

titleprint("FIGURE 6 — IL-TUR ASYMMETRY")

fig = plt.figure(figsize=(15.8, 7.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.16)

# Panel A: transfer macro-F1
axA = fig.add_subplot(gs[0, 0])
gradient_bg(axA, top="#F0F8FF", bottom="#FFF8EF")
axis_frame(axA)
axA.grid(True, axis="y", alpha=0.18, color="black", linewidth=0.8)

dirs = [("iltur_cl", "iltur_it"), ("iltur_it", "iltur_cl")]
x0, x1 = 0, 1

for method in COMMON4:
    sub = iltur_ctrl[iltur_ctrl["method"] == method].copy()
    if len(sub) != 2:
        continue
    y_left = float(sub[(sub["source"] == "iltur_cl") & (sub["target"] == "iltur_it")]["macro_f1_mean"].iloc[0])
    y_right = float(sub[(sub["source"] == "iltur_it") & (sub["target"] == "iltur_cl")]["macro_f1_mean"].iloc[0])

    axA.plot(
        [x0, x1], [y_left, y_right],
        color=METHOD_COLOR[method],
        linewidth=3.0,
        marker="o",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=1.0,
        alpha=0.92,
        zorder=3,
    )
    bbox_text(axA, x0 - 0.02, y_left, f"{y_left:.3f}", fontsize=11, ha="right")
    bbox_text(axA, x1 + 0.02, y_right, f"{y_right:.3f}", fontsize=11, ha="left")

axA.set_xlim(-0.25, 1.25)
axA.set_xticks([0, 1])
axA.set_xticklabels(["CL → IT", "IT → CL"])
axA.set_ylabel("Transfer Macro-F1")
axA.set_title("Controlled IL-TUR transfer", pad=12)

legend_elems = [
    Line2D([0], [0], color=METHOD_COLOR[m], lw=3, marker="o",
           markeredgecolor="black", markerfacecolor=METHOD_COLOR[m], label=pretty_method(m))
    for m in COMMON4
]
axA.legend(handles=legend_elems, loc="lower center", ncol=2,
           frameon=True, facecolor="#FFFDF8", edgecolor="black")
add_panel_label(axA, "Fig. 6a")

# Panel B: retention
axB = fig.add_subplot(gs[0, 1])
gradient_bg(axB, top="#F4FBFF", bottom="#FFF8F1")
axis_frame(axB)
axB.grid(True, axis="y", alpha=0.18, color="black", linewidth=0.8)

for method in COMMON4:
    sub = iltur_ctrl[iltur_ctrl["method"] == method].copy()
    if len(sub) != 2:
        continue
    y_left = float(sub[(sub["source"] == "iltur_cl") & (sub["target"] == "iltur_it")]["retention"].iloc[0])
    y_right = float(sub[(sub["source"] == "iltur_it") & (sub["target"] == "iltur_cl")]["retention"].iloc[0])

    axB.plot(
        [x0, x1], [y_left, y_right],
        color=METHOD_COLOR[method],
        linewidth=3.0,
        marker="o",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=1.0,
        alpha=0.92,
        zorder=3,
    )
    bbox_text(axB, x0 - 0.02, y_left, f"{y_left:.3f}", fontsize=11, ha="right")
    bbox_text(axB, x1 + 0.02, y_right, f"{y_right:.3f}", fontsize=11, ha="left")

axB.set_xlim(-0.25, 1.25)
axB.set_xticks([0, 1])
axB.set_xticklabels(["CL → IT", "IT → CL"])
axB.set_ylabel("Retention")
axB.set_title("Directional asymmetry in retention", pad=12)
note_box(
    axB, (0.52, 0.03),
    "IT → CL retains much more\nwithin-domain performance\nthan CL → IT across methods.",
    width=0.40, height=0.14, fc="#F8FBFF"
)
add_panel_label(axB, "Fig. 6b")

fig.tight_layout()
fig.savefig(OUT / "fig06_iltur_asymmetry.png", bbox_inches="tight")
fig.savefig(OUT / "fig06_iltur_asymmetry.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# SAVE SMALL TABLES FOR PAPER WRITING
# ============================================================

titleprint("SAVE SUMMARY TABLES")

# quick summary tables useful later
within_method.to_csv(OUT / "table_within_method_summary.csv", index=False)
core8_summary.to_csv(OUT / "table_core8_method_summary.csv", index=False)
core8_pairs.to_csv(OUT / "table_core8_pairwise.csv", index=False)
native.to_csv(OUT / "table_native_summary_raw.csv", index=False)
transfer.to_csv(OUT / "table_transfer_summary_raw.csv", index=False)

print("\nSaved figures:")
for fn in sorted(OUT.glob("fig*")):
    print(" ", fn.name)

print("\nSaved tables:")
for fn in sorted(OUT.glob("table*")):
    print(" ", fn.name)

print("\nAll outputs:", OUT)
