"""Supplementary robustness figure (Figure S7).

Renders the three sensitivity analyses of scripts/25_review_sensitivity.py from
the tables that script writes, so the figure and the tables cannot drift apart.

Usage: python scripts/27_fig_robustness.py [--tables results/tables/supplement] [--out preprint/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COL = {"Logistic": "#009E73", "MLP": "#0072B2", "MLP + neighbors": "#56B4E9",
       "GraphSAGE": "#000000", "GAT": "#E69F00"}
GREY = "#5A5A5A"


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "figure.dpi": 300,
    })


def letter(ax, c):
    ax.text(-0.14, 1.06, c, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables/supplement")
    ap.add_argument("--out", default="preprint/figures")
    a = ap.parse_args()
    T, out = Path(a.tables), Path(a.out)
    style()

    floor = pd.read_csv(T / "TableS12_lopo_support_floor.csv")
    rule = pd.read_csv(T / "TableS13_decision_rule.csv")
    sup = pd.read_csv(T / "TableS10_split_tuning_support.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))

    # (A) per-patient support floor
    ax = axes[0]
    order = ["none (published)", ">= 50 spots", ">= 200 spots"]
    for name, mk in [("Logistic - GraphSAGE", "o"), ("MLP - GraphSAGE", "s")]:
        sub = floor[floor.contrast == name].set_index("floor").loc[order]
        ax.plot(range(3), sub.wilcoxon_p, marker=mk, lw=1.2, ms=5,
                color=COL[name.split(" - ")[0]], label=name.replace(" - ", " vs "))
    ax.axhline(0.05, ls="--", lw=0.8, color=GREY)
    ax.annotate("p = 0.05", (0.02, 0.055), xycoords=("axes fraction", "data"),
                fontsize=6, color=GREY, va="bottom")
    ax.set_yscale("log"); ax.set_xticks(range(3))
    ax.set_xticklabels(["none\n(published)", "≥ 50", "≥ 200"])
    ax.set_xlabel("Minimum labelled spots per class, per patient")
    ax.set_ylabel("Paired Wilcoxon $p$ ($n$ = 13)")
    ax.set_title("Per-patient tests under a support floor", loc="left")
    ax.set_xlim(-0.25, 2.3); ax.legend(frameon=False, loc="lower right")
    letter(ax, "a")

    # (B) decision rule
    ax = axes[1]
    models = ["Logistic", "MLP", "MLP + neighbors", "GraphSAGE", "GAT"]
    r = rule.set_index("model")
    for i, m in enumerate(models):
        x0, x1 = r.loc[m, "pooled_macro_f1_argmax"], r.loc[m, "pooled_macro_f1_tuned"]
        ax.annotate("", xy=(x1, i), xytext=(x0, i),
                    arrowprops=dict(arrowstyle="-|>", color=COL[m], lw=1.3, shrinkA=0, shrinkB=0))
        ax.plot(x0, i, "o", ms=4, mfc="white", mec=COL[m], mew=1.1, zorder=3)
        ax.plot(x1, i, "o", ms=5, color=COL[m], zorder=3)
    ax.set_yticks(range(5)); ax.set_yticklabels(models); ax.invert_yaxis()
    ax.set_xlabel("Pooled leave-one-patient-out macro-F1")
    ax.set_title("Argmax (open) → tuned thresholds (filled)", loc="left")
    ax.margins(x=0.10)
    letter(ax, "b")

    # (C) tuning-block support
    ax = axes[2]
    v = sup.val_immune_hot.tolist()
    for i, y in enumerate(v):
        ax.plot([i, i], [0.6, y], "-", color=GREY, lw=0.8, zorder=1)
        ax.annotate(f"{y:,}" if y >= 100 else str(y), (i, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6)
    ax.plot(range(len(v)), v, "o", ms=6, color="#D55E00", zorder=3)
    ax.axhline(183, ls="--", lw=0.8, color="#009E73")
    ax.annotate("183 = smallest LOPO fold", (0.03, 210), xycoords=("axes fraction", "data"),
                fontsize=6, color="#009E73", va="bottom")
    ax.set_yscale("log"); ax.set_ylim(0.6, 12000)
    ax.set_xticks(range(len(v))); ax.set_xticklabels([s.replace("split", "S") for s in sup.split])
    ax.set_xlabel("Patient split")
    ax.set_ylabel("Immune-hot spots in the\nthreshold-tuning block")
    ax.set_title("Tuning support of the six splits", loc="left")
    letter(ax, "c")

    fig.tight_layout(pad=0.8)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "FigS7_robustness.png", dpi=300, bbox_inches="tight")
    print("wrote", out / "FigS7_robustness.png")


if __name__ == "__main__":
    main()
