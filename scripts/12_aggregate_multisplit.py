"""Aggregate the multi-split robustness runs (6 splits x 3 seeds x models).

Produces:
  1. multisplit_summary.json  — per-split per-seed macro-F1, split means, paired
     tests across splits (t-test df=5 + Wilcoxon), and a split-vs-seed variance
     decomposition per model.
  2. TableS6_per_split_metrics.csv — long table: model x split x seed x metrics.
  3. fig_split_robustness.{png,svg} — per-split paired macro-F1 differences
     (baseline - GNN) with seed-level spread.

Usage: python scripts/12_aggregate_multisplit.py \
    --runs results/multisplit --out-dir results/multisplit \
    --table results/tables/supplement/TableS6_per_split_metrics.csv \
    --fig results/figures/main/Figure_split_robustness
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepgraph_gbm.utils import save_json, setup_logging

log = logging.getLogger("deepgraph_gbm")

SPLITS = [f"split{i}" for i in range(6)]
SEEDS = [42, 7, 123]
MODELS = ["gnn", "mlp", "logistic"]  # gat added automatically when present
METRICS = ["macro_f1", "weighted_f1", "accuracy",
           "auroc_immune_cold", "auroc_immune_hot", "auroc_normal", "auroc_necrotic",
           "mes_auroc", "mes_pearson"]


def _load_runs(runs_dir: Path) -> dict:
    """runs[model][split][seed] = test metric dict."""
    runs: dict = {}
    for path in sorted(runs_dir.glob("*_split*_seed*.json")):
        d = json.load(open(path))
        model = d.get("model")
        if model == "gnn" and d.get("encoder") not in (None, "graphsage"):
            model = d["encoder"]  # e.g. "gat"
        runs.setdefault(model, {}).setdefault(d["split"], {})[str(d["seed"])] = d["test"]
    return runs


def _paired(x: np.ndarray, y: np.ndarray) -> dict:
    diff = x - y
    n = len(diff)
    md, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    t_stat, t_p = stats.ttest_rel(x, y)
    ci = stats.t.interval(0.95, df=n - 1, loc=md, scale=sd / np.sqrt(n)) if sd > 0 else (md, md)
    try:
        w_stat, w_p = stats.wilcoxon(x, y)
        w_stat, w_p = float(w_stat), float(w_p)
    except ValueError:
        w_stat, w_p = None, None
    return {"n": n, "mean_diff": md, "sd_diff": sd, "ci95": [float(ci[0]), float(ci[1])],
            "t_stat": float(t_stat), "t_p": float(t_p),
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p}


def _variance_decomposition(per_seed: dict) -> dict:
    """Split-level vs seed-level variance for a balanced split x seed design."""
    split_means = np.array([np.mean([per_seed[sp][str(s)] for s in SEEDS]) for sp in SPLITS])
    within = np.array([np.var([per_seed[sp][str(s)] for s in SEEDS], ddof=1) for sp in SPLITS])
    return {
        "split_means": {sp: float(m) for sp, m in zip(SPLITS, split_means)},
        "sd_across_splits": float(split_means.std(ddof=1)),
        "mean_within_split_seed_sd": float(np.sqrt(within.mean())),
        "variance_ratio_split_over_seed": float(split_means.var(ddof=1) / within.mean())
        if within.mean() > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--fig", required=True)
    args = ap.parse_args()
    setup_logging()

    runs = _load_runs(Path(args.runs))
    models = [m for m in MODELS if m in runs] + [m for m in runs if m not in MODELS]
    log.info("models found: %s", models)

    # completeness report
    missing = [(m, sp, s) for m in models for sp in SPLITS for s in SEEDS
               if sp not in runs.get(m, {}) or str(s) not in runs[m].get(sp, {})]
    if missing:
        log.warning("MISSING %d runs: %s", len(missing), missing)

    # per-split mean macro-F1 per model
    split_mean = {}
    per_seed_macro = {}
    for m in models:
        per_seed_macro[m] = {sp: {s: runs[m][sp][s]["macro_f1"] for s in map(str, SEEDS)
                                  if s in runs[m].get(sp, {})} for sp in SPLITS if sp in runs[m]}
        split_mean[m] = {sp: float(np.mean(list(v.values())))
                         for sp, v in per_seed_macro[m].items() if len(v) == len(SEEDS)}

    complete_splits = [sp for sp in SPLITS if all(sp in split_mean.get(m, {}) for m in models)]

    # paired tests across splits (reference = gnn)
    paired = {}
    if "gnn" in models and complete_splits:
        g = np.array([split_mean["gnn"][sp] for sp in complete_splits])
        for m in models:
            if m == "gnn":
                continue
            b = np.array([split_mean[m][sp] for sp in complete_splits])
            paired[f"gnn_vs_{m}"] = _paired(g, b)
            paired[f"gnn_vs_{m}"]["per_split"] = {
                sp: {"gnn": split_mean["gnn"][sp], m: split_mean[m][sp],
                     "diff_gnn_minus_model": split_mean["gnn"][sp] - split_mean[m][sp]}
                for sp in complete_splits}

    summary = {
        "design": "6 patient-level splits (8 train/2 val/3 test) x 3 seeds (42/7/123); "
                  "split0 = original fixed split; matched training + tuned-threshold scoring",
        "models": models,
        "complete_splits": complete_splits,
        "missing_runs": [list(x) for x in missing],
        "per_seed_macro_f1": per_seed_macro,
        "split_mean_macro_f1": split_mean,
        "paired_across_splits": paired,
        "variance_decomposition": {m: _variance_decomposition(per_seed_macro[m])
                                   for m in models if len(per_seed_macro[m]) == len(SPLITS)},
        "notes": "Paired tests pair split-mean macro-F1 (avg over 3 seeds) across n=6 splits; "
                 "t-test df=5; Wilcoxon min two-sided p at n=6 is 0.03125. Variance "
                 "decomposition contrasts SD across split means (patient-sampling variability) "
                 "with mean within-split seed SD (training-seed variability).",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, out_dir / "multisplit_summary.json")

    # long table
    table_path = Path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "seed"] + METRICS)
        for m in models:
            for sp in SPLITS:
                for s in SEEDS:
                    if sp in runs.get(m, {}) and str(s) in runs[m].get(sp, {}):
                        t = runs[m][sp][str(s)]
                        w.writerow([m, sp, s] + [round(float(t[k]), 4) for k in METRICS
                                                 if k in t])
    log.info("saved -> %s", table_path)

    # figure: (A) absolute per-split macro-F1 with per-seed points;
    #         (B) per-split paired macro-F1 differences vs GraphSAGE GNN
    if "gnn" in models and complete_splits:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
        matplotlib.rcParams["svg.fonttype"] = "none"

        labels = {"gnn": "GraphSAGE", "mlp": "MLP", "logistic": "Logistic", "gat": "GAT"}
        colors = {"gnn": "#000000", "mlp": "#0072B2", "logistic": "#009E73", "gat": "#E69F00"}
        xpos = np.arange(len(complete_splits))
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 3.8))

        # panel A: absolute performance, per-seed points + split-mean lines
        for m in models:
            means = [np.mean([per_seed_macro[m][sp][s] for s in map(str, SEEDS)])
                     for sp in complete_splits]
            axA.plot(xpos, means, "-o", color=colors.get(m, "#666666"), ms=5, lw=1.4,
                     label=labels.get(m, m), zorder=3)
            for si, sp in enumerate(complete_splits):
                seeds = [per_seed_macro[m][sp][s] for s in map(str, SEEDS)]
                axA.scatter(np.full(len(seeds), si) + np.linspace(-0.12, 0.12, len(seeds)),
                            seeds, color=colors.get(m, "#666666"), s=9, alpha=0.45,
                            zorder=2, edgecolors="none")
        axA.set_xticks(xpos)
        axA.set_xticklabels([sp.replace("split", "S") for sp in complete_splits])
        axA.set_xlabel("Patient-level split")
        axA.set_ylabel("Test macro-F1")
        axA.legend(frameon=False, fontsize=8, loc="lower left")
        axA.spines[["top", "right"]].set_visible(False)
        axA.set_title("A", loc="left", fontweight="bold")

        # panel B: paired difference vs GraphSAGE
        others = [m for m in models if m != "gnn"]
        width = 0.8 / max(len(others), 1)
        for j, m in enumerate(others):
            diffs, errs = [], []
            for sp in complete_splits:
                g_seeds = np.array([per_seed_macro["gnn"][sp][s] for s in map(str, SEEDS)])
                b_seeds = np.array([per_seed_macro[m][sp][s] for s in map(str, SEEDS)])
                d = b_seeds - g_seeds
                diffs.append(d.mean())
                errs.append(d.std(ddof=1))
            axB.bar(xpos + j * width, diffs, width * 0.9, yerr=errs, capsize=3,
                    label=labels.get(m, m),
                    color=colors.get(m, "#666666"), edgecolor="black", linewidth=0.4,
                    error_kw={"linewidth": 0.8})
        axB.axhline(0, color="black", linewidth=0.8)
        axB.set_xticks(xpos + width * (len(others) - 1) / 2)
        axB.set_xticklabels([sp.replace("split", "S") for sp in complete_splits])
        axB.set_xlabel("Patient-level split")
        axB.set_ylabel("Macro-F1 difference\n(model − GraphSAGE)")
        axB.legend(frameon=False, fontsize=8)
        axB.spines[["top", "right"]].set_visible(False)
        axB.set_title("B", loc="left", fontweight="bold")

        fig.tight_layout()
        fig_path = Path(args.fig)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(fig_path) + ".png", dpi=300)
        fig.savefig(str(fig_path) + ".svg")
        log.info("saved -> %s.png/.svg", fig_path)

    for m in models:
        if m in split_mean and complete_splits:
            vals = [split_mean[m][sp] for sp in complete_splits]
            log.info("%s split-mean macro-F1: %.4f +/- %.4f (range %.4f-%.4f)", m,
                     float(np.mean(vals)), float(np.std(vals, ddof=1)),
                     float(np.min(vals)), float(np.max(vals)))
    for comp, d in paired.items():
        log.info("%s: mean diff %.4f, t=%.2f, p=%.4f, wilcoxon p=%s", comp,
                 d["mean_diff"], d["t_stat"], d["t_p"], d["wilcoxon_p"])


if __name__ == "__main__":
    main()
