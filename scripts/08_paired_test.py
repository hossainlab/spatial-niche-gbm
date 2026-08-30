"""Paired significance test: GNN vs baselines across seeds.

Loads per-seed test macro-F1 for GNN (seeds 42/7/123), MLP, and logistic regression,
then computes a paired t-test and Wilcoxon signed-rank test on the per-seed
differences (GNN - baseline). Reports mean difference, 95% CI (t-based), t statistic,
df, and p-values, plus the Wilcoxon statistic/p. n=3 seeds => df=2, low power;
Wilcoxon min achievable two-sided p at n=3 is 0.25.

Usage: python scripts/08_paired_test.py \
    --gnn results/models --baselines results/baselines_multiseed \
    --out results/tables/main/Table1_paired_test.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepgraph_gbm.utils import save_json, setup_logging

log = logging.getLogger("deepgraph_gbm")

SEEDS = [42, 7, 123]


def _load_gnn_macro_f1(gnn_dir: Path) -> dict:
    """Per-seed GNN test macro-F1. Looks for seed{seed}/metrics.json first, then
    seed{seed}_metrics.json (the original seed-7 file uses the flat layout)."""
    out = {}
    for s in SEEDS:
        for cand in (gnn_dir / f"seed{s}" / "metrics.json", gnn_dir / f"seed{s}_metrics.json"):
            if cand.exists():
                d = json.load(open(cand))
                out[s] = float(d["test"]["macro_f1"])
                break
        else:
            raise FileNotFoundError(f"no metrics found for seed {s} under {gnn_dir}")
    return out


def _load_baseline_macro_f1(bdir: Path, model: str) -> dict:
    out = {}
    for s in SEEDS:
        d = json.load(open(bdir / f"{model}_seed{s}.json"))
        out[s] = float(d["macro_f1"])
    return out


def _paired(gnn: np.ndarray, base: np.ndarray) -> dict:
    diff = gnn - base
    n = len(diff)
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / np.sqrt(n)
    # paired t-test
    t_stat, t_p = stats.ttest_rel(gnn, base)
    ci = stats.t.interval(0.95, df=n - 1, loc=md, scale=se) if sd > 0 else (md, md)
    # Wilcoxon signed-rank (fails if all diffs are 0)
    try:
        w_stat, w_p = stats.wilcoxon(gnn, base)
        w_stat, w_p = float(w_stat), float(w_p)
    except ValueError as e:
        w_stat, w_p = None, None
        log.warning("Wilcoxon failed: %s", e)
    return {
        "n_seeds": n,
        "df": n - 1,
        "gnn_per_seed": gnn.tolist(),
        "baseline_per_seed": base.tolist(),
        "diff_per_seed": diff.tolist(),
        "mean_diff": md,
        "sd_diff": sd,
        "ci95": [float(ci[0]), float(ci[1])],
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    setup_logging()

    gnn_dir = Path(args.gnn)
    bdir = Path(args.baselines)

    gnn = _load_gnn_macro_f1(gnn_dir)
    mlp = _load_baseline_macro_f1(bdir, "mlp")
    logreg = _load_baseline_macro_f1(bdir, "logistic")

    gnn_arr = np.array([gnn[s] for s in SEEDS])
    mlp_arr = np.array([mlp[s] for s in SEEDS])
    logreg_arr = np.array([logreg[s] for s in SEEDS])

    result = {
        "metric": "macro_f1",
        "seeds": SEEDS,
        "gnn": {"per_seed": {str(s): gnn[s] for s in SEEDS},
                "mean": float(gnn_arr.mean()), "std": float(gnn_arr.std(ddof=1))},
        "mlp": {"per_seed": {str(s): mlp[s] for s in SEEDS},
                "mean": float(mlp_arr.mean()), "std": float(mlp_arr.std(ddof=1))},
        "logistic": {"per_seed": {str(s): logreg[s] for s in SEEDS},
                     "mean": float(logreg_arr.mean()), "std": float(logreg_arr.std(ddof=1))},
        "gnn_vs_mlp": _paired(gnn_arr, mlp_arr),
        "gnn_vs_logistic": _paired(gnn_arr, logreg_arr),
        "protocol": ("All models scored with per-class thresholds tuned on validation patients, "
                     "applied to test; seeds 42/7/123; single fixed patient-level split; MLP "
                     "trained with the same optimization protocol as the GNN (AdamW, cosine "
                     "schedule, boosted class weights, early stopping on validation macro-F1 "
                     "with best-checkpoint selection)."),
        "notes": ("Paired t-test and Wilcoxon signed-rank on per-seed macro-F1 (GNN - baseline). "
                  "n=3 seeds, df=2: very low power; a non-significant result does not imply no effect. "
                  "Wilcoxon min achievable two-sided p at n=3 is 0.25. Two pre-specified comparisons; "
                  "Bonferroni alpha=0.025 as sensitivity. IMPORTANT: all seeds share a single fixed "
                  "patient-level split, so p-values quantify training-seed variability only, NOT "
                  "patient-sampling (split) variability, which is likely the dominant source of "
                  "uncertainty with three held-out patients."),
        "table_note": ("Under the matched training + identical tuned-threshold scoring protocol, the "
                       "no-graph MLP OUTPERFORMS the GNN on macro-F1 on every seed. The earlier "
                       "GNN>baseline result was an artifact of scoring baselines with default argmax. "
                       "Paired-test p-values reflect seed-to-seed variability under a single fixed "
                       "patient split (n=3, low power)."),
    }
    save_json(result, args.out)
    log.info("GNN macro-F1: %.4f +/- %.4f", result["gnn"]["mean"], result["gnn"]["std"])
    log.info("MLP macro-F1: %.4f +/- %.4f", result["mlp"]["mean"], result["mlp"]["std"])
    log.info("logistic macro-F1: %.4f +/- %.4f", result["logistic"]["mean"], result["logistic"]["std"])
    log.info("GNN vs MLP: mean diff %.4f, t=%.3f, p=%.4f, wilcoxon p=%s",
             result["gnn_vs_mlp"]["mean_diff"], result["gnn_vs_mlp"]["t_stat"],
             result["gnn_vs_mlp"]["t_p"], result["gnn_vs_mlp"]["wilcoxon_p"])
    log.info("GNN vs logistic: mean diff %.4f, t=%.3f, p=%.4f, wilcoxon p=%s",
             result["gnn_vs_logistic"]["mean_diff"], result["gnn_vs_logistic"]["t_stat"],
             result["gnn_vs_logistic"]["t_p"], result["gnn_vs_logistic"]["wilcoxon_p"])
    log.info("saved -> %s", args.out)


if __name__ == "__main__":
    main()
