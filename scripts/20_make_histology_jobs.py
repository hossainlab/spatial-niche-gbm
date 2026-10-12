"""Job lists for the pathologist-label experiments (run with
scripts/15_revision_benchmark.py run --task histology).

E1  results/revision_ukf       leave-one-patient-out within the UKF cohort
                               (neuropathologist H&E labels), default
                               hyperparameters, 3 seeds (logistic: 1).
E2  results/revision_transfer_v2  train on SNUH (expression-derived AF labels),
                               validate on 2 SNUH patients, test on ALL UKF
                               patients (pathologist labels); 3 seeds.

The E2 runs in the manuscript read data/preprocessed_histology/transfer_v2, in which
UKF counts are scaled to 100,000 per spot to match the SNUH matrix. The current
scripts/19_build_histology_graphs.py writes exactly that scaling to <out>/transfer
(the v2 directory was made from the earlier CP10k build by the exact rescale
log1p(10 * expm1(x)); the two agree to float precision). results/revision_transfer
holds the earlier CP10k runs, reported in the manuscript only as a scaling
sensitivity analysis.

Usage: python scripts/20_make_histology_jobs.py --data data/preprocessed_histology
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "15_revision_benchmark.py")
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)

MODELS = {
    "logistic": {"C": 0.1},
    "mlp": dict(bench.DEFAULT_CFG),
    "mlpnbr": {**bench.DEFAULT_CFG, "hops": 1, "k": 6},
    "mlpnbr2": {**bench.DEFAULT_CFG, "hops": 2, "k": 6},
    "sage": dict(bench.DEFAULT_CFG),
    "gat": dict(bench.DEFAULT_CFG),
}


def jobs_for(folds, seeds):
    jobs = []
    for fold in folds:
        for name, cfg in MODELS.items():
            kind = "mlpnbr" if name.startswith("mlpnbr") else name
            for seed in ([42] if kind == "logistic" else seeds):
                jobs.append({"name": f"final_{name}_{fold}_s{seed}.json", "phase": "final", "model": kind,
                             "variant": name, "cfg": cfg, "fold": fold, "seed": seed, "save_preds": True})
    return jobs


def write(out: Path, folds, jobs):
    (out / "runs").mkdir(parents=True, exist_ok=True); (out / "preds").mkdir(exist_ok=True)
    (out / "folds.json").write_text(json.dumps(folds, indent=1))
    with open(out / "jobs.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(out, len(folds), "folds,", len(jobs), "jobs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/preprocessed_histology")
    args = ap.parse_args()
    seeds = [42, 7, 123]

    graphs, _ = bench.load_data(Path(args.data) / "ukf")
    bench.configure_task("histology", graphs)
    folds = bench.lopo_folds(graphs, min_val_class=50)
    for k, v in folds.items():
        print(k, "val", v["val"])
    write(ROOT / "results" / "revision_ukf", folds, jobs_for(folds, seeds))

    graphs, _ = bench.load_data(Path(args.data) / "transfer")
    snuh = [g for g in graphs if g.cohort == "SNUH"]
    ukf_patients = sorted({g.patient for g in graphs if g.cohort == "UKF"})
    counts = bench.patient_class_counts(snuh)
    import itertools
    val = max(itertools.combinations(sorted(counts), 2),
              key=lambda pr: (min(counts[pr[0]] + counts[pr[1]]), pr))
    train = [p for p in sorted(counts) if p not in val]
    folds = {"SNUH_to_UKF": {"train": train, "val": list(val), "test": ukf_patients}}
    print("transfer val", val)
    write(ROOT / "results" / "revision_transfer_v2", folds, jobs_for(folds, seeds))


if __name__ == "__main__":
    main()
