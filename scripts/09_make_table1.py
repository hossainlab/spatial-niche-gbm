"""Build Table 1 (model comparison, mean +/- SD across seeds) and Table S5
(per-seed metrics) from per-seed metric JSONs, and embed the significance tests of
scripts/08_paired_test.py (Welch / one-sample t; see its docstring).

Inputs:
  --gnn       results/models            (seed{s}/metrics.json or seed{s}_metrics.json)
  --baselines results/baselines_multiseed_matched  (mlp_seed{s}.json, logistic_seed{s}.json)
  --paired    results/tables/main/Table1_paired_test.json
  --out-main  results/tables/main/Table1_model_comparison.json
  --out-supp  results/tables/supplement/TableS5_per_seed_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_niche_gbm.utils import save_json, setup_logging

log = logging.getLogger("spatial_niche_gbm")

SEEDS = [42, 7, 123]
METRICS = ["macro_f1", "weighted_f1", "accuracy",
           "auroc_immune_cold", "auroc_immune_hot", "auroc_normal", "auroc_necrotic",
           "mes_auroc", "mes_pearson"]


def _load_gnn(gnn_dir: Path) -> dict:
    out = {}
    for s in SEEDS:
        for cand in (gnn_dir / f"seed{s}" / "metrics.json", gnn_dir / f"seed{s}_metrics.json"):
            if cand.exists():
                out[s] = json.load(open(cand))["test"]
                break
        else:
            raise FileNotFoundError(f"no GNN metrics for seed {s} under {gnn_dir}")
    return out


def _load_baseline(bdir: Path, model: str) -> dict:
    return {s: json.load(open(bdir / f"{model}_seed{s}.json")) for s in SEEDS}


def _summarize(per_seed: dict) -> dict:
    out = {}
    for k in METRICS:
        vals = [float(per_seed[s][k]) for s in SEEDS if k in per_seed[s]]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--paired", required=True)
    ap.add_argument("--out-main", required=True)
    ap.add_argument("--out-supp", required=True)
    args = ap.parse_args()
    setup_logging()

    per_seed = {
        "GNN": _load_gnn(Path(args.gnn)),
        "MLP": _load_baseline(Path(args.baselines), "mlp"),
        "Logistic": _load_baseline(Path(args.baselines), "logistic"),
    }
    paired = json.load(open(args.paired))

    table1 = {
        "protocol": paired.get("protocol",
            "all models scored with per-class thresholds tuned on validation patients, "
            "applied to test; seeds 42/7/123; MLP trained with the same optimization "
            "protocol as the GNN (early stopping on validation macro-F1)"),
        "seeds": SEEDS,
        "gnn_multiseed": _summarize(per_seed["GNN"]),
        "mlp_multiseed": _summarize(per_seed["MLP"]),
        "logistic_multiseed": _summarize(per_seed["Logistic"]),
        "gat_split0_macro_f1": paired["gat"],
        "tests_macro_f1": {comp: paired[comp] for comp in
                           ("mlp_vs_gnn", "logistic_vs_gnn", "gat_vs_gnn")},
        "tests_note": paired.get("notes", ""),
        "note": paired.get("table_note", ""),
    }

    out_main = Path(args.out_main)
    out_main.parent.mkdir(parents=True, exist_ok=True)
    save_json(table1, out_main)

    out_supp = Path(args.out_supp)
    out_supp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_supp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed"] + METRICS)
        for model in ("GNN", "MLP", "Logistic"):
            for s in SEEDS:
                w.writerow([model, s] + [round(float(per_seed[model][s][k]), 4)
                                         for k in METRICS if k in per_seed[model][s]])

    for model in ("GNN", "MLP", "Logistic"):
        key = {"GNN": "gnn_multiseed", "MLP": "mlp_multiseed",
               "Logistic": "logistic_multiseed"}[model]
        m = table1[key]["macro_f1"]
        log.info("%s macro-F1: %.4f +/- %.4f", model, m["mean"], m["std"])
    log.info("saved -> %s and %s", out_main, out_supp)


if __name__ == "__main__":
    main()
