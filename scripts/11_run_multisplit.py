"""Run ONE (split, seed, model) training + evaluation for the split-robustness
analysis, under the exact matched protocol of Table 1:

- GNN: GraphSAGE trained with train_model (early stopping on val macro-F1,
  best-checkpoint selection), thresholds tuned on val, applied to test.
- MLP: same optimization protocol as the GNN (AdamW, cosine schedule, boosted
  class weights, early stopping on val macro-F1) via run_baselines_multiseed.
- Logistic: deterministic lbfgs; same val-tuned threshold scoring.

Usage:
  python scripts/11_run_multisplit.py --config configs/default.yaml \
      --data <processed_dir> --splits configs/splits_multisplit.json \
      --split split0 --seed 42 --model gnn --out results/multisplit
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spatial_niche_gbm.models.baselines import LogisticBaseline
from spatial_niche_gbm.models.multitask import MultiTaskGNN
from spatial_niche_gbm.training.evaluate import evaluate_graphs
from spatial_niche_gbm.training.train import train_model
from spatial_niche_gbm.utils import load_config, load_json, save_json, set_seed, setup_logging
from run_baselines_multiseed import _metrics_tuned, _mlp_prob, _stack, _train_mlp

log = logging.getLogger("spatial_niche_gbm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="configs/splits_multisplit.json")
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", choices=["gnn", "mlp", "logistic"], required=True)
    ap.add_argument("--encoder", default=None,
                    help="gnn encoder type: graphsage (default) or gat")
    ap.add_argument("--tag", default=None,
                    help="output filename tag (default: model, or encoder for gnn)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override training max_epochs (smoke tests only)")
    args = ap.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
        cfg["training"]["patience"] = args.max_epochs + 1  # disable early stop for smoke

    data_list = torch.load(Path(args.data) / "graphs.pt", weights_only=False)
    splits = load_json(args.splits)["splits"]
    sp = splits[args.split]
    train_data = [d for d in data_list if d.patient in sp["train"]]
    val_data = [d for d in data_list if d.patient in sp["val"]]
    test_data = [d for d in data_list if d.patient in sp["test"]]
    log.info("%s | %s seed %d: train %d graphs (%d spots) | val %d (%d) | test %d (%d)",
             args.split, args.model, args.seed, len(train_data),
             sum(d.n_spots for d in train_data), len(val_data),
             sum(d.n_spots for d in val_data), len(test_data),
             sum(d.n_spots for d in test_data))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"split": args.split, "seed": args.seed, "model": args.model,
           "train_patients": sp["train"], "val_patients": sp["val"],
           "test_patients": sp["test"]}

    if args.model == "gnn":
        encoder_type = args.encoder or cfg["model"].get("encoder", "graphsage")
        set_seed(args.seed)
        model = MultiTaskGNN(in_dim=data_list[0].x.shape[1],
                             hidden_dims=cfg["model"]["hidden_dims"],
                             dropout=cfg["model"]["dropout"], aggr=cfg["model"]["aggr"],
                             encoder_type=encoder_type)
        rec["encoder"] = encoder_type
        with tempfile.TemporaryDirectory() as ckpt_dir:
            result = train_model(model, train_data, val_data, cfg, ckpt_dir)
            ckpt = torch.load(Path(ckpt_dir) / "best_model.pt", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            val_metrics = evaluate_graphs(model, val_data)  # tunes thresholds on val
            thr = val_metrics.get("thresholds")
            test_metrics = evaluate_graphs(model, test_data, thr=thr)
        rec.update({"best_epoch": result["best_epoch"],
                    "best_val_macro_f1": result["best_val_macro_f1"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "thresholds": thr if isinstance(thr, list) else thr,
                    "test": test_metrics})
        log.info("TEST macro_f1 %.4f acc %.4f (best epoch %d)",
                 test_metrics["macro_f1"], test_metrics["accuracy"], result["best_epoch"])
    else:
        Xtr, ytr, mtr = _stack(train_data)
        Xva, yva, mva = _stack(val_data)
        Xte, yte, mte = _stack(test_data)
        if args.model == "logistic":
            set_seed(args.seed)
            clf = LogisticBaseline().fit(Xtr, ytr, mtr)
            prob_va, mes_va = clf.predict_proba(Xva), clf.mes_proba(Xva)
            prob_te, mes_te = clf.predict_proba(Xte), clf.mes_proba(Xte)
            extra = {}
        else:
            mlp = _train_mlp(args.seed, cfg, Xtr, ytr, mtr, train_data, Xva, yva)
            prob_va, mes_va = _mlp_prob(mlp, Xva)
            prob_te, mes_te = _mlp_prob(mlp, Xte)
            extra = {"best_epoch": mlp._best_epoch,
                     "best_val_macro_f1": mlp._best_val_macro_f1}
        test_metrics = _metrics_tuned(yva, prob_va, mva, mes_va,
                                      yte, prob_te, mte, mes_te)
        rec.update(extra)
        rec["test"] = test_metrics
        log.info("TEST macro_f1 %.4f acc %.4f", test_metrics["macro_f1"],
                 test_metrics["accuracy"])

    tag = args.tag or args.model  # gnn default keeps "gnn_" prefix; pass --tag gat for GAT
    path = out / f"{tag}_{args.split}_seed{args.seed}.json"
    save_json(rec, path)
    log.info("saved -> %s", path)


if __name__ == "__main__":
    main()
