from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path.home() / "TaxFacto"

ADD_A = ROOT / "results/additional/analysis"
ADD_B = ROOT / "results/additional/bootstrap"
CORE8 = ROOT / "results/additional/fair_core8"
MAIN = ROOT / "results/final_analysis"

OUT = ROOT / "results/paper_figures"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 21,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "figure.titlesize": 26,
    "axes.linewidth": 1.2,
    "patch.linewidth": 1.0,
    "savefig.dpi": 240,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
})

COLORS = {
    "full_ft": "#E8B86D",              # soft warm gold
    "shared_lora": "#AFC7F8",          # pastel blue
    "category_lora": "#BFE6C8",        # pastel mint
    "role_adapter": "#D7B8F3",         # pastel violet
    "shared_lora_widehead": "#F2C6B4", # pastel coral
    "frozen": "#D7D7D7",               # soft gray
}

METHOD_ORDER = [
    "full_ft",
    "role_adapter",
    "category_lora",
    "shared_lora",
    "shared_lora_widehead",
    "frozen",
]

LABELS = {
    "full_ft": "Full FT",
    "shared_lora": "Shared LoRA",
    "category_lora": "Category LoRA",
    "role_adapter": "Role Adapter",
    "shared_lora_widehead": "Wide-head",
    "frozen": "Frozen",
    "legaleval": "LegalEval",
    "marro_india": "MARRO-IN",
    "marro_uk": "MARRO-UK",
    "iltur_cl": "IL-TUR CL",
    "iltur_it": "IL-TUR IT",
}


def apply_ax_style(ax):
    ax.set_facecolor("#FCFBF8")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8, color="black")
    ax.set_axisbelow(True)


def add_background_gradient(fig, ax, top="#FFF8F2", bottom="#EEF6FF", alpha=0.90):
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


def text_bbox(ax, x, y, s, fontsize=14, ha="center", va="center", fc="#FFFDF7"):
    ax.text(
        x, y, s,
        ha=ha, va=va,
        fontsize=fontsize,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor=fc,
            edgecolor="black",
            linewidth=0.9,
            alpha=0.92,
        ),
        zorder=10,
    )


def prettify_method(m):
    return LABELS.get(m, m)


def prettify_ds(d):
    return LABELS.get(d, d)


# ============================================================
# LOAD DATA
# ============================================================

core_summary = pd.read_csv(CORE8 / "transfer_core8_method_summary.csv")
core_pairs = pd.read_csv(CORE8 / "transfer_core8_pairwise.csv")
role_focus = pd.read_csv(CORE8 / "transfer_core8_role_focus.csv")

role_mat = pd.read_csv(ADD_A / "transfer_matrix_role_adapter.csv", index_col=0)
shared_mat = pd.read_csv(ADD_A / "transfer_matrix_shared_lora.csv", index_col=0)
delta_mat = pd.read_csv(ADD_A / "transfer_matrix_role_minus_shared.csv", index_col=0)

native = pd.read_csv(ADD_A / "native_summary.csv")
boot = pd.read_csv(ADD_B / "hierarchical_bootstrap_global.csv")

main_cell = pd.read_csv(MAIN / "main_cell_mean_std.csv")


# ============================================================
# FIG A — FAIR CORE-8 BARS
# ============================================================

fig, ax = plt.subplots(figsize=(12.8, 8.2))
add_background_gradient(fig, ax, top="#FFF7F2", bottom="#EEF7FF")
apply_ax_style(ax)

df = core_summary.copy()
df["method_label"] = df["method"].map(prettify_method)
df = df.sort_values("mean_macro_f1", ascending=True)

y = np.arange(len(df))
vals = df["mean_macro_f1"].to_numpy()

bars = ax.barh(
    y,
    vals,
    color=[COLORS[m] for m in df["method"]],
    edgecolor="black",
    linewidth=1.0,
    alpha=0.88,
    height=0.70,
    zorder=3,
)

for b in bars:
    b.set_path_effects([
        pe.withStroke(linewidth=1.5, foreground="black", alpha=0.15)
    ])

ax.set_yticks(y)
ax.set_yticklabels(df["method_label"])
ax.set_xlabel("Macro-F1 on fair 8-direction transfer")
ax.set_title("Fair 8-direction transfer comparison", pad=14)

xmin = max(0, vals.min() - 0.08)
xmax = min(1.0, vals.max() + 0.08)
ax.set_xlim(xmin, xmax)

for yy, v, s in zip(y, vals, df["std_across_dirs"]):
    text_bbox(
        ax,
        v + 0.012,
        yy,
        f"{v:.4f}\n±{s:.4f}",
        fontsize=13,
        ha="left",
        fc="#FFF9EF",
    )

# decorative panel
panel = FancyBboxPatch(
    (0.015, 0.03), 0.28, 0.17,
    transform=ax.transAxes,
    boxstyle="round,pad=0.02,rounding_size=0.03",
    facecolor="#F9F7FF",
    edgecolor="black",
    linewidth=1.0,
    alpha=0.92,
    zorder=2,
)
ax.add_patch(panel)
ax.text(
    0.035, 0.165,
    "Core-8 = 6 directions among\nLegalEval/MARRO + 2 IL-TUR CL↔IT",
    transform=ax.transAxes,
    fontsize=14,
    va="top",
    color="black",
    zorder=3,
)

fig.tight_layout()
fig.savefig(OUT / "fig_core8_transfer_bars.png", bbox_inches="tight")
fig.savefig(OUT / "fig_core8_transfer_bars.pdf", bbox_inches="tight")
plt.close(fig)


# ============================================================
# FIG B — ROLE TRANSFER MATRIX
# ============================================================

def matrix_plot(df, title, outstem, cmap_colors, fmt="{:.4f}", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(10.8, 8.9))
    add_background_gradient(fig, ax, top="#FFF8F1", bottom="#F1F8FF")
    ax.set_facecolor("#FCFBF8")

    cmap = LinearSegmentedColormap.from_list("custom", cmap_colors)
    arr = df.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticklabels([prettify_ds(c) for c in df.columns], rotation=30, ha="right")
    ax.set_yticklabels([prettify_ds(i) for i in df.index])

    ax.set_title(title, pad=16)
    ax.set_xlabel("Target domain")
    ax.set_ylabel("Source domain")

    # cell borders
    ax.set_xticks(np.arange(-.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.9, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    # annotate with bbox
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = arr[i, j]
            text_bbox(
                ax, j, i, fmt.format(val),
                fontsize=13,
                fc="#FFFDF5",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.90, pad=0.02)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(labelsize=15)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")

    fig.tight_layout()
    fig.savefig(OUT / f"{outstem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{outstem}.pdf", bbox_inches="tight")
    plt.close(fig)

matrix_plot(
    role_mat,
    "Role Adapter transfer matrix",
    "fig_transfer_matrix_role_adapter",
    ["#FFF8EE", "#F5D6FA", "#D7B8F3", "#AE8DDD"],
    vmin=float(np.nanmin(role_mat.values)),
    vmax=float(np.nanmax(role_mat.values)),
)

matrix_plot(
    delta_mat,
    "Role Adapter minus Shared LoRA",
    "fig_transfer_matrix_role_minus_shared",
    ["#F5E7E7", "#FFF9F2", "#DDF3E3", "#BFE6C8", "#8FD3A7"],
    fmt="{:+.4f}",
    vmin=-max(abs(np.nanmin(delta_mat.values)), abs(np.nanmax(delta_mat.values))),
    vmax=max(abs(np.nanmin(delta_mat.values)), abs(np.nanmax(delta_mat.values))),
)


# ============================================================
# FIG C — ROLE-FOCUS DUMBBELL ON CORE-8
# ============================================================

fig, ax = plt.subplots(figsize=(12.8, 9.0))
add_background_gradient(fig, ax, top="#FFF8F2", bottom="#EEF7FF")
apply_ax_style(ax)

rf = role_focus.copy()
rf["dir"] = rf["source"].map(prettify_ds) + " → " + rf["target"].map(prettify_ds)
rf["delta_role_minus_shared_lora"] = rf["delta_role_minus_shared_lora"].astype(float)
rf = rf.sort_values("delta_role_minus_shared_lora", ascending=True).reset_index(drop=True)

yy = np.arange(len(rf))

for i, r in rf.iterrows():
    ax.plot(
        [r["shared_lora"], r["role_adapter"]],
        [i, i],
        color="black",
        linewidth=1.3,
        alpha=0.55,
        zorder=2,
    )

ax.scatter(
    rf["shared_lora"], yy,
    s=170,
    color=COLORS["shared_lora"],
    edgecolor="black",
    linewidth=1.0,
    zorder=3,
    label="Shared LoRA",
)

ax.scatter(
    rf["role_adapter"], yy,
    s=170,
    color=COLORS["role_adapter"],
    edgecolor="black",
    linewidth=1.0,
    zorder=4,
    label="Role Adapter",
)

for i, r in rf.iterrows():
    mid = max(r["shared_lora"], r["role_adapter"]) + 0.010
    text_bbox(
        ax, mid, i,
        f"{r['delta_role_minus_shared_lora']:+.4f}",
        fontsize=12,
        ha="left",
        fc="#FFF8EF",
    )

ax.set_yticks(yy)
ax.set_yticklabels(rf["dir"])
ax.set_xlabel("Macro-F1")
ax.set_title("Role Adapter vs Shared LoRA on the fair 8 transfer directions", pad=15)
ax.legend(loc="lower right", frameon=True, edgecolor="black", facecolor="#FFFDF8")

ax.set_xlim(
    min(rf["shared_lora"].min(), rf["role_adapter"].min()) - 0.04,
    max(rf["shared_lora"].max(), rf["role_adapter"].max()) + 0.08,
)

fig.tight_layout()
fig.savefig(OUT / "fig_core8_role_vs_shared_dumbbell.png", bbox_inches="tight")
fig.savefig(OUT / "fig_core8_role_vs_shared_dumbbell.pdf", bbox_inches="tight")
plt.close(fig)


# ============================================================
# FIG D — NATIVE TAXONOMY GROUPED BARS
# ============================================================

fig, ax = plt.subplots(figsize=(12.6, 8.5))
add_background_gradient(fig, ax, top="#FFF7F0", bottom="#F0F8FF")
apply_ax_style(ax)

n = native.copy()
datasets = ["legaleval", "iltur_cl", "iltur_it"]
methods = ["full_ft", "role_adapter", "shared_lora"]

pivot = n.pivot(index="dataset", columns="method", values="macro_f1_mean").loc[datasets, methods]

x = np.arange(len(datasets))
w = 0.23

for k, m in enumerate(methods):
    vals = pivot[m].to_numpy()
    bars = ax.bar(
        x + (k - 1) * w,
        vals,
        width=w,
        color=COLORS[m],
        edgecolor="black",
        linewidth=1.0,
        alpha=0.90,
        label=prettify_method(m),
        zorder=3,
    )
    for b, v in zip(bars, vals):
        text_bbox(
            ax,
            b.get_x() + b.get_width()/2,
            v + 0.012,
            f"{v:.4f}",
            fontsize=12,
            fc="#FFFDF7",
        )

ax.set_xticks(x)
ax.set_xticklabels([prettify_ds(d) for d in datasets])
ax.set_ylabel("Test Macro-F1")
ax.set_title("Native-taxonomy validation", pad=15)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02),
          frameon=True, facecolor="#FFFDF8", edgecolor="black")

ax.set_ylim(0, max(pivot.max()) + 0.16)

fig.tight_layout()
fig.savefig(OUT / "fig_native_taxonomy_bars.png", bbox_inches="tight")
fig.savefig(OUT / "fig_native_taxonomy_bars.pdf", bbox_inches="tight")
plt.close(fig)


# ============================================================
# FIG E — DOCUMENT BOOTSTRAP FOREST PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(12.8, 7.8))
add_background_gradient(fig, ax, top="#FFF8F2", bottom="#EFF7FF")
apply_ax_style(ax)

focus_order = [
    ("shared_lora", "role_adapter"),
    ("shared_lora_widehead", "role_adapter"),
    ("category_lora", "role_adapter"),
    ("full_ft", "role_adapter"),
]

bb = boot.copy()

rows = []
for a, b in focus_order:
    q = bb[(bb["method_a"] == a) & (bb["method_b"] == b)]
    if len(q) == 1:
        rows.append(q.iloc[0])

f = pd.DataFrame(rows)
f["label"] = [f"{prettify_method(r['method_a'])} vs {prettify_method(r['method_b'])}" for _, r in f.iterrows()]

yy = np.arange(len(f))[::-1]

ax.axvline(0, color="black", linewidth=1.2, alpha=0.8, zorder=1)

for i, (_, r) in enumerate(f.iterrows()):
    y = yy[i]
    c = COLORS["role_adapter"] if r["point_mean_delta"] < 0 else "#E8B86D"
    ax.plot(
        [r["ci95_low"], r["ci95_high"]],
        [y, y],
        color="black",
        linewidth=2.1,
        zorder=2,
    )
    ax.scatter(
        [r["point_mean_delta"]],
        [y],
        s=180,
        color="#D7B8F3" if "role_adapter" in r["method_b"] else "#E8B86D",
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )
    text_bbox(
        ax,
        r["ci95_high"] + 0.008,
        y,
        f"{r['point_mean_delta']:+.4f}\n[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]",
        fontsize=12,
        ha="left",
        fc="#FFF9F1",
    )

ax.set_yticks(yy)
ax.set_yticklabels(f["label"])
ax.set_xlabel("Document-level hierarchical bootstrap Δ Macro-F1")
ax.set_title("Bootstrap confidence intervals", pad=14)

xmin = f["ci95_low"].min() - 0.04
xmax = f["ci95_high"].max() + 0.16
ax.set_xlim(xmin, xmax)

fig.tight_layout()
fig.savefig(OUT / "fig_bootstrap_forest.png", bbox_inches="tight")
fig.savefig(OUT / "fig_bootstrap_forest.pdf", bbox_inches="tight")
plt.close(fig)


# ============================================================
# FIG F — EFFICIENCY VS ACCURACY
# ============================================================

fig, ax = plt.subplots(figsize=(11.8, 8.2))
add_background_gradient(fig, ax, top="#FFF8F3", bottom="#EEF7FF")
apply_ax_style(ax)

m = main_cell.copy()

# aggregate over all dataset/model cells by method
eff = (
    m.groupby("method_key", as_index=False)
    .agg(
        macro_f1_mean=("macro_f1_mean", "mean"),
        trainable_pct=("trainable_pct", "mean"),
        train_seconds=("train_seconds_mean", "mean"),
    )
)

eff = eff[eff["method_key"].isin(METHOD_ORDER)].copy()

for _, r in eff.iterrows():
    method = r["method_key"]
    size = 120 + 0.8 * r["train_seconds"]
    ax.scatter(
        r["trainable_pct"],
        r["macro_f1_mean"],
        s=size,
        color=COLORS[method],
        edgecolor="black",
        linewidth=1.1,
        alpha=0.90,
        zorder=3,
    )
    text_bbox(
        ax,
        r["trainable_pct"] + 0.5,
        r["macro_f1_mean"] + 0.004,
        f"{prettify_method(method)}\nF1={r['macro_f1_mean']:.4f}\n{r['trainable_pct']:.3f}%",
        fontsize=12,
        ha="left",
        fc="#FFFDF8",
    )

ax.set_xlabel("Trainable parameters (%)")
ax.set_ylabel("Mean within-domain Macro-F1")
ax.set_title("Accuracy–efficiency trade-off", pad=14)

ax.set_xlim(-2, max(eff["trainable_pct"]) * 1.08)
ax.set_ylim(min(eff["macro_f1_mean"]) - 0.03, max(eff["macro_f1_mean"]) + 0.05)

# background note
note = FancyBboxPatch(
    (0.03, 0.04), 0.29, 0.12,
    transform=ax.transAxes,
    boxstyle="round,pad=0.02,rounding_size=0.03",
    facecolor="#FAF7FF",
    edgecolor="black",
    linewidth=1.0,
    alpha=0.93,
    zorder=2,
)
ax.add_patch(note)
ax.text(
    0.05, 0.135,
    "Bubble size ∝ mean train time\nAcross the 335 within-domain runs",
    transform=ax.transAxes,
    fontsize=13,
    va="top",
    color="black",
    zorder=3,
)

fig.tight_layout()
fig.savefig(OUT / "fig_efficiency_tradeoff.png", bbox_inches="tight")
fig.savefig(OUT / "fig_efficiency_tradeoff.pdf", bbox_inches="tight")
plt.close(fig)

print("=" * 100)
print("SAVED FIGURES")
print("=" * 100)
for fn in sorted(OUT.iterdir()):
    print(fn)
