"""Train + evaluate baselines (logistic regression, MLP) across multiple seeds,
scored identically to the GNN: per-class decision thresholds tuned on VALIDATION
patients, then applied to TEST patients (no test-label leakage).

This mirrors deepgraph_gbm.training.evaluate.classification_metrics with
thr=tune_thresholds(val) then applied to test, so baseline metrics are directly
comparable to the GNN multi-seed summary in Table 1.

Usage: python scripts/run_baselines_multiseed.py --config configs/default.yaml \
           --data <processed_dir> --out results/baselines_multiseed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepgraph_gbm.models.baselines import LogisticBaseline, MLPBaseline
from deepgraph_gbm.models.multitask import multitask_loss
from deepgraph_gbm.training.evaluate import NICHE_CLASSES, tune_thresholds, apply_thresholds
from deepgraph_gbm.training.train import make_class_weights
from deepgraph_gbm.utils import load_config, load_json, save_json, set_seed, setup_logging

log = logging.getLogger("deepgraph_gbm")


def _stack(data_list):
    X = torch.cat([d.x for d in data_list]).numpy()
    niche = torch.cat([d.niche_y for d in data_list]).numpy()
    mes = torch.cat([d.mes_y for d in data_list]).numpy()
    return X, niche, mes


def _metrics_tuned(y_val, prob_val, mes_y_val, mes_p_val,
                   y_te, prob_te, mes_y_te, mes_p_te):
    """Tune per-class thresholds on VAL, apply to TEST; compute full metric dict."""
    # tune thresholds on validation (labeled spots only)
    vmask = y_val >= 0
    thr = tune_thresholds(y_val[vmask], prob_val[vmask])

    # apply to test
    tmask = y_te >= 0
    y, prob = y_te[tmask], prob_te[tmask]
    yhat = apply_thresholds(prob, thr)

    m = {
        "macro_f1": float(f1_score(y, yhat, average="macro")),
        "weighted_f1": float(f1_score(y, yhat, average="weighted")),
        "accuracy": float((yhat == y).mean()),
        "thresholds": thr.tolist(),
    }
    for i, c in enumerate(NICHE_CLASSES):
        yi = (y == i).astype(int)
        if 0 < yi.sum() < len(yi):
            m[f"auroc_{c}"] = float(roc_auc_score(yi, prob[:, i]))
            m[f"auprc_{c}"] = float(average_precision_score(yi, prob[:, i]))
        m[f"f1_{c}"] = float(f1_score(y == i, yhat == i))

    mm = ~np.isnan(mes_y_te)
    if len(np.unique(mes_y_te[mm])) > 1:
        m["mes_auroc"] = float(roc_auc_score((mes_y_te[mm] > 0.5).astype(int), mes_p_te[mm]))
        m["mes_pearson"] = float(np.corrcoef(mes_y_te[mm], mes_p_te[mm])[0, 1])
    return m


def _val_macro_f1_tuned(mlp, Xva, yva):
    """Validation macro-F1 with thresholds tuned on val -- identical to the GNN
    early-stopping monitor (evaluate_graphs -> classification_metrics with thr=None)."""
    mlp.eval()
    prob, _ = _mlp_prob(mlp, Xva)
    vmask = yva >= 0
    thr = tune_thresholds(yva[vmask], prob[vmask])
    yhat = apply_thresholds(prob[vmask], thr)
    return float(f1_score(yva[vmask], yhat, average="macro"))


def _train_mlp(seed, cfg, Xtr, ytr, mtr, train_data, Xva, yva):
    """Train MLP under the SAME optimization protocol as the GNN (03_train.py /
    train_model): AdamW + cosine schedule, identical boosted class weights, and
    early stopping on validation macro-F1 with best-checkpoint selection, so the
    GNN-vs-MLP contrast isolates neighborhood aggregation rather than training setup."""
    set_seed(seed)
    tcfg = cfg["training"]
    mlp = MLPBaseline(in_dim=Xtr.shape[1], hidden_dims=cfg["model"]["hidden_dims"],
                      dropout=cfg["model"]["dropout"])
    opt = torch.optim.AdamW(mlp.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["max_epochs"])
    class_w = make_class_weights(train_data, boost=tcfg.get("class_boost"))
    lw = tcfg["loss_weights"]
    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr)
    mtr_t = torch.from_numpy(mtr).float()

    best_val, best_epoch, best_state = -np.inf, -1, None
    patience_left = tcfg["patience"]
    for epoch in range(1, tcfg["max_epochs"] + 1):
        mlp.train()
        out_d = mlp(Xtr_t)
        loss, _ = multitask_loss(out_d, ytr_t, mtr_t, None, class_w, lw)
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        val_f1 = _val_macro_f1_tuned(mlp, Xva, yva)
        if val_f1 > best_val:
            best_val, best_epoch = val_f1, epoch
            best_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
            patience_left = tcfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("MLP seed %d: early stop at epoch %d (best %d, val macroF1 %.3f)",
                         seed, epoch, best_epoch, best_val)
                break

    mlp.load_state_dict(best_state)
    mlp.eval()
    mlp._best_epoch = best_epoch
    mlp._best_val_macro_f1 = best_val
    return mlp


def _mlp_prob(mlp, X):
    with torch.no_grad():
        o = mlp(torch.from_numpy(X).float())
    prob = torch.softmax(o["niche_logits"], 1).numpy()
    return prob, o["mes_prob"].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/baselines_multiseed")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seeds = cfg["evaluation"]["seeds"]  # [42, 7, 123]
    data_list = torch.load(Path(args.data) / "graphs.pt", weights_only=False)
    splits = load_json(cfg["data"]["splits_file"])
    train_data = [d for d in data_list if d.patient in splits["train"]]
    val_data = [d for d in data_list if d.patient in splits["val"]]
    test_data = [d for d in data_list if d.patient in splits["test"]]

    Xtr, ytr, mtr = _stack(train_data)
    Xva, yva, mva = _stack(val_data)
    Xte, yte, mte = _stack(test_data)
    log.info("spots: train %d | val %d | test %d", len(ytr), len(yva), len(yte))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_seed = {"logistic": {}, "mlp": {}}

    for seed in seeds:
        log.info("=== seed %d ===", seed)
        # logistic (deterministic given data, but run per-seed for clean design)
        set_seed(seed)
        lr = LogisticBaseline().fit(Xtr, ytr, mtr)
        prob_va, mes_va = lr.predict_proba(Xva), lr.mes_proba(Xva)
        prob_te, mes_te = lr.predict_proba(Xte), lr.mes_proba(Xte)
        per_seed["logistic"][str(seed)] = _metrics_tuned(
            yva, prob_va, mva, mes_va, yte, prob_te, mte, mes_te)
        log.info("logistic seed %d: macro_f1 %.4f acc %.4f", seed,
                 per_seed["logistic"][str(seed)]["macro_f1"],
                 per_seed["logistic"][str(seed)]["accuracy"])

        # MLP (seed-dependent; same early-stopping protocol as the GNN)
        mlp = _train_mlp(seed, cfg, Xtr, ytr, mtr, train_data, Xva, yva)
        prob_va, mes_va = _mlp_prob(mlp, Xva)
        prob_te, mes_te = _mlp_prob(mlp, Xte)
        per_seed["mlp"][str(seed)] = _metrics_tuned(
            yva, prob_va, mva, mes_va, yte, prob_te, mte, mes_te)
        per_seed["mlp"][str(seed)]["best_epoch"] = mlp._best_epoch
        per_seed["mlp"][str(seed)]["best_val_macro_f1"] = mlp._best_val_macro_f1
        log.info("MLP seed %d: macro_f1 %.4f acc %.4f", seed,
                 per_seed["mlp"][str(seed)]["macro_f1"],
                 per_seed["mlp"][str(seed)]["accuracy"])

        save_json(per_seed["mlp"][str(seed)], out / f"mlp_seed{seed}.json")
        save_json(per_seed["logistic"][str(seed)], out / f"logistic_seed{seed}.json")

    # aggregate mean +/- std across seeds per model
    summary = {}
    for model in ("logistic", "mlp"):
        keys = [k for k in per_seed[model][str(seeds[0])].keys()
                if isinstance(per_seed[model][str(seeds[0])][k], float)]
        summary[model] = {}
        for k in keys:
            vals = [per_seed[model][str(s)][k] for s in seeds if k in per_seed[model][str(s)]]
            summary[model][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                                 "values": {str(s): per_seed[model][str(s)][k] for s in seeds}}
    summary["seeds"] = seeds
    summary["protocol"] = ("per-class thresholds tuned on validation patients, applied to test; "
                           "identical to GNN multi-seed protocol. MLP trained with the same "
                           "optimization protocol as the GNN (AdamW + cosine schedule, boosted "
                           "class weights, early stopping on validation macro-F1 with "
                           "best-checkpoint selection) so the contrast isolates neighborhood "
                           "aggregation. Logistic regression is deterministic (lbfgs).")
    save_json(summary, out / "baseline_multiseed_summary.json")
    save_json(per_seed, out / "baseline_per_seed.json")
    log.info("saved -> %s", out / "baseline_multiseed_summary.json")

    for model in ("logistic", "mlp"):
        log.info("%s macro_f1: %.4f +/- %.4f", model,
                 summary[model]["macro_f1"]["mean"], summary[model]["macro_f1"]["std"])


if __name__ == "__main__":
    main()
