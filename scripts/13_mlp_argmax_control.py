"""Fixed-split decision-rule control for the MLP.

Train the matched-protocol MLP (early stopping, 3 seeds) on the primary split and
report macro-F1 under BOTH argmax and validation-tuned thresholds. The manuscript
cites the result in section 2.3 (MLP 0.557 +/- 0.050 argmax vs 0.571 +/- 0.021
tuned); the logistic counterpart (0.541 vs 0.557) comes from
scripts/run_baselines.py (results/baselines_original_protocol_rerun) and
results/baselines_multiseed_matched/logistic_decision_rule_control.json.

The decision-rule analysis that the conclusions rest on is the leave-one-patient-out
version for every model in scripts/25_review_sensitivity.py (Supplementary Table S13).
This script originally supported the retracted "protocol reversal" claim; only the
argmax-vs-tuned comparison is still used.

Usage: python scripts/13_mlp_argmax_control.py --config configs/default.yaml \
    --data <processed_dir> --out results/baselines_multiseed_matched/mlp_argmax_control.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepgraph_gbm.utils import load_config, load_json, save_json, setup_logging
from run_baselines_multiseed import _mlp_prob, _stack, _train_mlp

log = logging.getLogger("deepgraph_gbm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    data_list = torch.load(Path(args.data) / "graphs.pt", weights_only=False)
    splits = load_json(cfg["data"]["splits_file"])
    train_data = [d for d in data_list if d.patient in splits["train"]]
    val_data = [d for d in data_list if d.patient in splits["val"]]
    test_data = [d for d in data_list if d.patient in splits["test"]]
    Xtr, ytr, mtr = _stack(train_data)
    Xva, yva, mva = _stack(val_data)
    Xte, yte, mte = _stack(test_data)

    from deepgraph_gbm.training.evaluate import tune_thresholds, apply_thresholds

    per_seed = {}
    for seed in cfg["evaluation"]["seeds"]:
        mlp = _train_mlp(seed, cfg, Xtr, ytr, mtr, train_data, Xva, yva)
        prob_va, _ = _mlp_prob(mlp, Xva)
        prob_te, _ = _mlp_prob(mlp, Xte)
        tmask = yte >= 0
        y, prob = yte[tmask], prob_te[tmask]
        # argmax scoring
        f1_argmax = float(f1_score(y, prob.argmax(1), average="macro"))
        # tuned-threshold scoring (thresholds from val)
        vmask = yva >= 0
        thr = tune_thresholds(yva[vmask], prob_va[vmask])
        f1_tuned = float(f1_score(y, apply_thresholds(prob, thr), average="macro"))
        per_seed[str(seed)] = {"argmax_macro_f1": f1_argmax, "tuned_macro_f1": f1_tuned,
                               "best_epoch": mlp._best_epoch}
        log.info("seed %d: argmax %.4f | tuned %.4f (best epoch %d)",
                 seed, f1_argmax, f1_tuned, mlp._best_epoch)

    out = {
        "note": "Matched-protocol MLP (early stopping) on the fixed split, scored two ways. "
                "Isolates the scoring-rule effect for the current model; compare with the "
                "original 120-epoch no-early-stop MLP scored by argmax (Table S1: 0.4488).",
        "per_seed": per_seed,
        "argmax_macro_f1_mean": float(np.mean([v["argmax_macro_f1"] for v in per_seed.values()])),
        "argmax_macro_f1_std": float(np.std([v["argmax_macro_f1"] for v in per_seed.values()], ddof=1)),
        "tuned_macro_f1_mean": float(np.mean([v["tuned_macro_f1"] for v in per_seed.values()])),
        "tuned_macro_f1_std": float(np.std([v["tuned_macro_f1"] for v in per_seed.values()], ddof=1)),
    }
    save_json(out, args.out)
    log.info("argmax %.4f +/- %.4f | tuned %.4f +/- %.4f -> saved %s",
             out["argmax_macro_f1_mean"], out["argmax_macro_f1_std"],
             out["tuned_macro_f1_mean"], out["tuned_macro_f1_std"], args.out)


if __name__ == "__main__":
    main()
