"""Data-driven graphical abstract for the revised manuscript.

Left: benchmark design (Visium spots -> spatial kNN graph -> graph vs graph-free models under one
protocol). Middle: pooled leave-one-patient-out macro-F1 against expression-derived (SNUH) and
neuropathologist (UKF) labels. Right: take-home messages. All numbers are read from the result
summaries written by scripts 17 and 21.

Usage: .venv/bin/python scripts/22_graphical_abstract.py [--out preprint/figures/graphical_abstract.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

MODEL_COLORS = {"logistic": "#009E73", "mlp": "#0072B2", "mlpnbr": "#56B4E9", "sage": "#000000", "gat": "#E69F00"}
MODEL_NAMES = {"logistic": "Logistic", "mlp": "MLP", "mlpnbr": "MLP+nbr", "sage": "GraphSAGE", "gat": "GAT"}
NICHE_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]  # cold, hot, normal, necrotic


def load(root: Path):
    snuh = json.loads((root / "results/revision/revision_summary.json").read_text())["lopo"]
    ukf = json.loads((root / "results/revision_ukf/histology_summary.json").read_text())["E1_ukf_lopo"]["models"]
    s = {m: (snuh[m]["pooled_macro_f1_mean"], snuh[m]["pooled_macro_f1_sd"]) for m in MODEL_COLORS}
    # UKF has one- and two-hop MLP + neighbors; show the better one
    nbr = max((k for k in ukf if k.startswith("mlpnbr")), key=lambda k: ukf[k]["pooled_macro_f1"])
    key = {"logistic": "logistic", "mlp": "mlp", "mlpnbr": nbr, "sage": "sage", "gat": "gat"}
    u = {m: (ukf[k]["pooled_macro_f1"], ukf[k]["pooled_sd"]) for m, k in key.items()}
    return s, u


def spots(ax, rng):
    # a hexagonal Visium-like patch with four contiguous niches
    pts, cols = [], []
    for r in range(9):
        for c in range(9):
            x, y = c + 0.5 * (r % 2), r * 0.87
            if (x - 4.2) ** 2 + (y - 3.5) ** 2 > 21:
                continue
            niche = (0 if x < 4.2 else 1) if y > 3.5 else (2 if x < 4.2 else 3)
            if rng.random() < 0.12:
                niche = rng.integers(4)
            pts.append((x, y)); cols.append(NICHE_COLORS[niche])
    pts = np.array(pts)
    return pts, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="preprint/figures/graphical_abstract.png")
    args = ap.parse_args()
    root = Path(args.root)
    snuh, ukf = load(root)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(12, 5.2))
    fig.text(0.5, 0.955, "Does spatial graph structure improve supervised niche prediction in glioblastoma Visium data?",
             ha="center", fontsize=13, fontweight="bold")

    # ---- design panel
    ax = fig.add_axes([0.01, 0.06, 0.36, 0.82]); ax.set_xlim(0, 20); ax.set_ylim(0, 18); ax.axis("off")
    rng = np.random.default_rng(3)
    pts, cols = spots(ax, rng)
    p1 = pts * 0.62 + [0.6, 10.2]
    ax.scatter(*p1.T, s=22, c=cols, edgecolors="none")
    ax.text(3.2, 9.6, "Visium spots (13 + 18 patients)", ha="center", va="top", fontsize=8.5)
    p2 = pts * 0.62 + [11.6, 10.2]
    for i in range(len(p2)):
        d = np.linalg.norm(p2 - p2[i], axis=1)
        for j in np.argsort(d)[1:4]:
            ax.plot(*zip(p2[i], p2[j]), color="0.75", lw=0.5, zorder=0)
    ax.scatter(*p2.T, s=22, c=cols, edgecolors="none", zorder=1)
    ax.text(14.2, 9.6, "Spatial kNN graph", ha="center", va="top", fontsize=8.5)
    ax.add_patch(FancyArrowPatch((6.6, 13.0), (10.8, 13.0), arrowstyle="-|>", mutation_scale=14, color="0.3"))
    boxes = [(0.3, "Graph-free\nLogistic regression, MLP,\nMLP + neighbor features", MODEL_COLORS["logistic"]),
             (10.3, "Graph encoders\nGraphSAGE, GAT", "#333333")]
    for x0, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x0, 3.9), 9.4, 3.4, boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.4))
        ax.text(x0 + 4.7, 5.6, txt, ha="center", va="center", fontsize=8.5)
    ax.add_patch(FancyArrowPatch((5.0, 8.6), (5.0, 7.7), arrowstyle="-|>", mutation_scale=12, color="0.3"))
    ax.add_patch(FancyArrowPatch((15.0, 8.6), (15.0, 7.7), arrowstyle="-|>", mutation_scale=12, color="0.3"))
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 19.4, 2.4, boxstyle="round,pad=0.25", fc="#F2F2F2", ec="none"))
    ax.text(10, 1.5, "Same features, training and tuned thresholds; leave-one-patient-out\n"
            "SNUH: nested tuning + ablations (1,053 runs); UKF: defaults (304 runs)",
            ha="center", va="center", fontsize=8)

    # ---- result panels
    order = ["logistic", "mlp", "mlpnbr", "sage", "gat"]
    for k, (res, title, x0) in enumerate([(snuh, "Expression-derived labels\n(SNUH, 13 patients)", 0.42),
                                          (ukf, "Neuropathologist H&E labels\n(UKF, 18 patients)", 0.60)]):
        a = fig.add_axes([x0, 0.2, 0.15, 0.6])
        xs = np.arange(len(order))
        a.bar(xs, [res[m][0] for m in order], yerr=[res[m][1] for m in order], capsize=2,
              color=[MODEL_COLORS[m] for m in order], edgecolor="black", linewidth=0.5)
        for x, m in zip(xs, order):
            a.text(x, res[m][0] + res[m][1] + 0.01, f"{res[m][0]:.2f}", ha="center", fontsize=7)
        a.set_xticks(xs, [MODEL_NAMES[m] for m in order], rotation=45, ha="right", fontsize=8)
        a.set_ylim(0, 0.8); a.set_title(title, fontsize=9)
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
        if k == 0:
            a.set_ylabel("Pooled LOPO macro-F1")
        a.axvspan(2.5, 4.5, color="0.92", zorder=0)
        a.text(3.5, 0.76, "graph", ha="center", fontsize=7.5, color="0.35")

    # ---- take-home panel
    ax = fig.add_axes([0.78, 0.1, 0.21, 0.76]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02", fc="#FFF8E7",
                                ec="#E69F00", lw=1.2, transform=ax.transAxes))
    msgs = ["Logistic regression ranked first\nwithin both cohorts (LOPO)",
            "Graph encoders were never\nsignificantly better than an MLP,\nalthough homophily was 0.80",
            "No depth, neighborhood size or\nmarker-gene removal produced\na graph advantage",
            "Held-out patient changed\nmacro-F1 far more than the\nchoice of model"]
    ax.text(0.5, 0.92, "Take-home", ha="center", fontweight="bold", fontsize=10.5, transform=ax.transAxes)
    for i, m in enumerate(msgs):
        y = 0.8 - i * 0.2
        ax.text(0.08, y, f"{i + 1}", fontweight="bold", fontsize=11, color="#D55E00", va="top", transform=ax.transAxes)
        ax.text(0.17, y, m, fontsize=8.5, va="top", transform=ax.transAxes)

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
