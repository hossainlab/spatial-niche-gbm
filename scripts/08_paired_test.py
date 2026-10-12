"""Significance tests for Table 1 (fixed primary split, seeds 42/7/123).

Seeds are NOT shared units across models (seed 42 of the MLP has nothing in common
with seed 42 of GraphSAGE), so per-seed results are compared as independent samples:

  * GraphSAGE vs MLP and GraphSAGE vs GAT: Welch t-test on per-seed test macro-F1,
    with a Welch-Satterthwaite 95% CI of the difference in means;
  * GraphSAGE vs logistic regression: one-sample t-test of the GraphSAGE seeds against
    the deterministic logistic value (logistic regression has no seed variance).

These are the tests reported in the manuscript (Table 1 caption and section 2.3). The
seed-PAIRED t-test used before v5 (p = 0.0017) treated seeds as shared units and was
withdrawn (preprint/ANALYSIS_AUDIT.md, finding 5). With three seeds on one split,
every p-value here reflects training-seed variability only, not patient sampling.

Usage: python scripts/08_paired_test.py \
    --gnn results/models --baselines results/baselines_multiseed_matched \
    --gat results/multisplit --out results/tables/main/Table1_paired_test.json
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


def _load_gat_macro_f1(ms_dir: Path) -> dict:
    """Table 1's GAT row is split 0 of the six-split benchmark (= the primary split)."""
    return {s: float(json.load(open(ms_dir / f"gat_split0_seed{s}.json"))["test"]["macro_f1"])
            for s in SEEDS}


def _welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch t-test of mean(a) - mean(b) with a Welch-Satterthwaite 95% CI."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    df = float((va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)))
    md = float(a.mean() - b.mean())
    t = stats.ttest_ind(a, b, equal_var=False)
    half = float(stats.t.ppf(0.975, df) * se)
    return {"test": "welch_t", "mean_diff": md, "ci95": [md - half, md + half],
            "t": float(t.statistic), "df": df, "p": float(t.pvalue)}


def _one_sample(a: np.ndarray, value: float) -> dict:
    """One-sample t-test of value - mean(a), for a deterministic comparator."""
    t = stats.ttest_1samp(a, value)
    md = float(value - a.mean())
    half = float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
    return {"test": "one_sample_t", "mean_diff": md, "ci95": [md - half, md + half],
            "t": float(-t.statistic), "df": float(len(a) - 1), "p": float(t.pvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--gat", default="results/multisplit",
                    help="six-split run directory holding gat_split0_seed{s}.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    setup_logging()

    gnn_dir = Path(args.gnn)
    bdir = Path(args.baselines)

    gnn = _load_gnn_macro_f1(gnn_dir)
    mlp = _load_baseline_macro_f1(bdir, "mlp")
    logreg = _load_baseline_macro_f1(bdir, "logistic")
    gat = _load_gat_macro_f1(Path(args.gat))

    gnn_arr = np.array([gnn[s] for s in SEEDS])
    mlp_arr = np.array([mlp[s] for s in SEEDS])
    logreg_arr = np.array([logreg[s] for s in SEEDS])
    gat_arr = np.array([gat[s] for s in SEEDS])

    result = {
        "metric": "macro_f1",
        "seeds": SEEDS,
        "gnn": {"per_seed": {str(s): gnn[s] for s in SEEDS},
                "mean": float(gnn_arr.mean()), "std": float(gnn_arr.std(ddof=1))},
        "mlp": {"per_seed": {str(s): mlp[s] for s in SEEDS},
                "mean": float(mlp_arr.mean()), "std": float(mlp_arr.std(ddof=1))},
        "logistic": {"per_seed": {str(s): logreg[s] for s in SEEDS},
                     "mean": float(logreg_arr.mean()), "std": float(logreg_arr.std(ddof=1))},
        "gat": {"per_seed": {str(s): gat[s] for s in SEEDS},
                "mean": float(gat_arr.mean()), "std": float(gat_arr.std(ddof=1))},
        # each contrast is (comparator - GraphSAGE); positive = comparator ahead
        "mlp_vs_gnn": _welch(mlp_arr, gnn_arr),
        "logistic_vs_gnn": _one_sample(gnn_arr, float(logreg_arr[0])),
        "gat_vs_gnn": _welch(gat_arr, gnn_arr),
        "protocol": ("All models scored with per-class thresholds tuned on validation patients, "
                     "applied to test; seeds 42/7/123; single fixed patient-level split; MLP "
                     "trained with the same optimization protocol as the GNN (AdamW, cosine "
                     "schedule, boosted class weights, early stopping on validation macro-F1 "
                     "with best-checkpoint selection)."),
        "notes": ("Seeds are not shared across models, so MLP and GAT are compared with GraphSAGE "
                  "by Welch t-tests on per-seed macro-F1, and logistic regression (deterministic) by "
                  "a one-sample t-test of the GraphSAGE seeds against its value. n = 3 seeds on one "
                  "fixed patient split: p-values quantify training-seed variability only, not "
                  "patient sampling. The seed-paired t-test used before v5 (p = 0.0017) was withdrawn."),
        "table_note": ("Matched protocol on the fixed primary split: the MLP had the highest macro-F1 and "
                       "every MLP seed exceeded every GraphSAGE seed. This single-split comparison is "
                       "secondary; the leave-one-patient-out benchmark (results/revision) is the primary "
                       "analysis."),
    }
    save_json(result, args.out)
    log.info("GNN macro-F1: %.4f +/- %.4f", result["gnn"]["mean"], result["gnn"]["std"])
    log.info("MLP macro-F1: %.4f +/- %.4f", result["mlp"]["mean"], result["mlp"]["std"])
    log.info("logistic macro-F1: %.4f +/- %.4f", result["logistic"]["mean"], result["logistic"]["std"])
    for comp in ("mlp_vs_gnn", "logistic_vs_gnn", "gat_vs_gnn"):
        r = result[comp]
        log.info("%s: diff %.4f (95%% CI %.3f to %.3f), t=%.3f, p=%.4f (%s)", comp, r["mean_diff"],
                 r["ci95"][0], r["ci95"][1], r["t"], r["p"], r["test"])
    log.info("saved -> %s", args.out)


if __name__ == "__main__":
    main()
