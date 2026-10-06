from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


ROOT = Path.home() / "TaxFacto"
A = ROOT / "results/additional/analysis"
OUT = ROOT / "results/additional/fair_core8"

OUT.mkdir(parents=True, exist_ok=True)

x = pd.read_csv(A / "transfer_summary.csv")

ALL_METHODS = [
    "full_ft",
    "shared_lora",
    "category_lora",
    "role_adapter",
]

OLD = {"legaleval", "marro_india", "marro_uk"}

def is_core8(src, tgt):
    if src == "iltur_cl" and tgt == "iltur_it":
        return True
    if src == "iltur_it" and tgt == "iltur_cl":
        return True
    if src in OLD and tgt in OLD and src != tgt:
        return True
    return False

core = x[x.apply(lambda r: is_core8(r["source"], r["target"]), axis=1)].copy()

# sanity
dirs = sorted(set(zip(core["source"], core["target"])))
if len(dirs) != 8:
    raise RuntimeError(f"Expected 8 directions, got {len(dirs)}: {dirs}")

# keep only 4 methods
core = core[core["method"].isin(ALL_METHODS)].copy()

# all 8 dirs must have all 4 methods
counts = (
    core.groupby(["source", "target"])["method"]
    .nunique()
    .reset_index(name="n_methods")
)
if (counts["n_methods"] != 4).any():
    raise RuntimeError("Some core directions do not contain all 4 methods.")

core.to_csv(OUT / "transfer_core8_all_rows.csv", index=False)

# pivot table: rows = directions, cols = methods
p = core.pivot_table(
    index=["source", "target"],
    columns="method",
    values="macro_f1_mean",
    aggfunc="mean",
).reset_index()

p.to_csv(OUT / "transfer_core8_pivot.csv", index=False)

# summary stats
summary = (
    core.groupby("method", as_index=False)
    .agg(
        mean_macro_f1=("macro_f1_mean", "mean"),
        std_across_dirs=("macro_f1_mean", "std"),
        mean_retention=("retention", "mean"),
        mean_gap=("transfer_gap", "mean"),
        mean_weighted_f1=("weighted_f1_mean", "mean"),
        mean_accuracy=("accuracy_mean", "mean"),
    )
    .sort_values("mean_macro_f1", ascending=False)
)
summary.to_csv(OUT / "transfer_core8_method_summary.csv", index=False)

# paired deltas
pair_rows = []
for a, b in combinations(ALL_METHODS, 2):
    da = p[a].to_numpy()
    db = p[b].to_numpy()
    d = da - db

    row = {
        "method_a": a,
        "method_b": b,
        "mean_delta": float(np.mean(d)),
        "median_delta": float(np.median(d)),
        "wins_a": int(np.sum(d > 0)),
        "ties": int(np.sum(d == 0)),
        "wins_b": int(np.sum(d < 0)),
    }

    if HAVE_SCIPY:
        w = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        row["wilcoxon_W"] = float(w.statistic)
        row["wilcoxon_p"] = float(w.pvalue)
    else:
        row["wilcoxon_W"] = np.nan
        row["wilcoxon_p"] = np.nan

    pair_rows.append(row)

pairs = pd.DataFrame(pair_rows).sort_values(
    ["mean_delta"], ascending=False
)
pairs.to_csv(OUT / "transfer_core8_pairwise.csv", index=False)

# role adapter focus
role = p[["source", "target", "role_adapter"]].copy()
for other in ["shared_lora", "category_lora", "full_ft"]:
    role[other] = p[other].values
    role[f"delta_role_minus_{other}"] = role["role_adapter"] - role[other]

role.to_csv(OUT / "transfer_core8_role_focus.csv", index=False)

# LaTeX tables
def latex_escape(s):
    return (
        str(s)
        .replace("_", r"\_")
        .replace("%", r"\%")
    )

# main summary latex
with open(OUT / "table_transfer_core8_summary.tex", "w") as f:
    f.write("\\begin{tabular}{lccccc}\n")
    f.write("\\toprule\n")
    f.write("Method & Macro-F1 & Std. & Retention & Gap & Accuracy \\\\\n")
    f.write("\\midrule\n")
    for _, r in summary.iterrows():
        f.write(
            f"{latex_escape(r['method'])} & "
            f"{r['mean_macro_f1']:.4f} & "
            f"{r['std_across_dirs']:.4f} & "
            f"{r['mean_retention']:.3f} & "
            f"{r['mean_gap']:+.4f} & "
            f"{r['mean_accuracy']:.4f} \\\\\n"
        )
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

# pairwise latex
with open(OUT / "table_transfer_core8_pairwise.tex", "w") as f:
    f.write("\\begin{tabular}{llcccc}\n")
    f.write("\\toprule\n")
    f.write("Method A & Method B & Mean $\\Delta$ & Wins & Ties & Losses \\\\\n")
    f.write("\\midrule\n")
    for _, r in pairs.iterrows():
        f.write(
            f"{latex_escape(r['method_a'])} & "
            f"{latex_escape(r['method_b'])} & "
            f"{r['mean_delta']:+.4f} & "
            f"{int(r['wins_a'])} & "
            f"{int(r['ties'])} & "
            f"{int(r['wins_b'])} \\\\\n"
        )
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print("=" * 100)
print("FAIR CORE-8 TRANSFER SUMMARY")
print("=" * 100)
print(summary.to_string(index=False, formatters={
    "mean_macro_f1": lambda v: f"{v:.4f}",
    "std_across_dirs": lambda v: f"{v:.4f}",
    "mean_retention": lambda v: f"{v:.3f}",
    "mean_gap": lambda v: f"{v:+.4f}",
    "mean_weighted_f1": lambda v: f"{v:.4f}",
    "mean_accuracy": lambda v: f"{v:.4f}",
}))
print()
print("=" * 100)
print("PAIRWISE COMPARISONS")
print("=" * 100)
print(pairs.to_string(index=False, formatters={
    "mean_delta": lambda v: f"{v:+.4f}",
    "median_delta": lambda v: f"{v:+.4f}",
    "wilcoxon_W": lambda v: f"{v:.3f}" if pd.notna(v) else "nan",
    "wilcoxon_p": lambda v: f"{v:.6g}" if pd.notna(v) else "nan",
}))
print()
print("Saved to:", OUT)
