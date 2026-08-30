"""Protocol-reversal figure: how the scoring protocol changes the GNN-vs-baseline
ranking on the fixed split.

Panel A: the comparison as originally scored (baselines: single-run argmax, Table S1;
GNN: tuned thresholds, 3 seeds) — the GNN appears best.
Panel B: the matched protocol (all models tuned thresholds, 3-seed mean +/- SD,
Table 1) — the MLP is significantly best.
Panel C: scoring-rule isolation — the deterministic logistic model and the
matched-protocol MLP each scored by argmax vs tuned thresholds.

Usage: python scripts/14_fig_protocol_reversal.py \
    --table-s1 results/tables/supplement/TableS1_baseline_metrics.json \
    --table1 results/tables/main/Table1_model_comparison.json \
    --control results/baselines_multiseed_matched/mlp_argmax_control.json \
    --out results/figures/main/Figure_protocol_reversal
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepgraph_gbm.utils import setup_logging

log = logging.getLogger("deepgraph_gbm")

matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

COLORS = {"GNN": "#D55E00", "MLP": "#0072B2", "Logistic": "#009E73"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-s1", required=True)
    ap.add_argument("--table1", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    setup_logging()

    s1 = json.load(open(args.table_s1))
    t1 = json.load(open(args.table1))
    ctrl = json.load(open(args.control))

    models = ["GNN", "MLP", "Logistic"]
    # Panel A: original scoring (baselines argmax single-run; GNN tuned 3-seed)
    orig = {
        "GNN": (t1["gnn_multiseed"]["macro_f1"]["mean"], t1["gnn_multiseed"]["macro_f1"]["std"]),
        "MLP": (s1["mlp"]["macro_f1"], 0.0),
        "Logistic": (s1["logistic"]["macro_f1"], 0.0),
    }
    # Panel B: matched protocol (all tuned, 3 seeds)
    matched = {
        "GNN": (t1["gnn_multiseed"]["macro_f1"]["mean"], t1["gnn_multiseed"]["macro_f1"]["std"]),
        "MLP": (t1["mlp_multiseed"]["macro_f1"]["mean"], t1["mlp_multiseed"]["macro_f1"]["std"]),
        "Logistic": (t1["logistic_multiseed"]["macro_f1"]["mean"],
                     t1["logistic_multiseed"]["macro_f1"]["std"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4),
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})

    for ax, data, title in (
        (axes[0], orig, "A  Original scoring\n(baselines: argmax, 1 run)"),
        (axes[1], matched, "B  Matched protocol\n(all: tuned thresholds, 3 seeds)"),
    ):
        x = np.arange(len(models))
        vals = [data[m][0] for m in models]
        errs = [data[m][1] for m in models]
        ax.bar(x, vals, 0.62, yerr=errs, capsize=3,
               color=[COLORS[m] for m in models], edgecolor="black", linewidth=0.4,
               error_kw={"linewidth": 0.8})
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylim(0, 0.72)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test macro-F1", fontsize=9)

    # Panel C: scoring-rule isolation (argmax vs tuned for the same models)
    axc = axes[2]
    pairs = [("Logistic", s1["logistic"]["macro_f1"],
              t1["logistic_multiseed"]["macro_f1"]["mean"]),
             ("MLP", ctrl["argmax_macro_f1_mean"], ctrl["tuned_macro_f1_mean"])]
    x = np.arange(len(pairs))
    w = 0.34
    axc.bar(x - w / 2, [p[1] for p in pairs], w * 0.9, label="argmax",
            color="#BBBBBB", edgecolor="black", linewidth=0.4)
    axc.bar(x + w / 2, [p[2] for p in pairs], w * 0.9, label="tuned thresholds",
            color="#56B4E9", edgecolor="black", linewidth=0.4)
    for xi, p in zip(x, pairs):
        axc.text(xi - w / 2, p[1] + 0.012, f"{p[1]:.3f}", ha="center", va="bottom", fontsize=7.5)
        axc.text(xi + w / 2, p[2] + 0.012, f"{p[2]:.3f}", ha="center", va="bottom", fontsize=7.5)
    axc.set_xticks(x)
    axc.set_xticklabels([p[0] for p in pairs], fontsize=8)
    axc.set_ylim(0, 0.72)
    axc.set_title("C  Same model, two scoring rules", fontsize=8.5, loc="left")
    axc.legend(frameon=False, fontsize=7.5, loc="upper left")
    axc.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=300)
    fig.savefig(str(out) + ".svg")
    log.info("saved -> %s.png/.svg", out)


if __name__ == "__main__":
    main()
