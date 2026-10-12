"""Generate 6 random patient-level splits (8 train / 2 val / 3 test) for the
split-robustness analysis, plus a per-split composition report.

Split 0 reproduces the original fixed split (configs/splits.json) so the
multi-split analysis connects directly to Table 1. Splits 1-5 are new random
assignments with fixed generation seeds. n=6 splits is chosen so a Wilcoxon
signed-rank test across splits can reach p < 0.05 (min two-sided p at n=6 is
0.03125; at n=5 it is 0.0625).

Usage: python scripts/10_make_splits.py --config configs/default.yaml \
           --data <processed_dir> --out configs/splits_multisplit.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_niche_gbm.utils import load_config, load_json, save_json, setup_logging

log = logging.getLogger("spatial_niche_gbm")

N_TEST, N_VAL = 3, 2
GEN_SEEDS = [1001, 1002, 1003, 1004, 1005]  # splits 1-5; split 0 = original


def _composition(meta: pd.DataFrame, patients: list[str]) -> dict:
    sub = meta[meta["patient"].isin(patients)]
    return {
        "n_sections": int(len(sub)),
        "n_spots": int(sub["n_spots"].sum()),
        "n_labeled": int(sub["n_labeled"].sum()),
        "n_frozen_sections": int((sub["source"] == "Frozen").sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="configs/splits_multisplit.json")
    args = ap.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    orig = load_json(cfg["data"]["splits_file"])
    patients = sorted(orig["all_patients"])
    meta = pd.read_csv(Path(args.data) / "sections_meta.csv")

    splits: dict[str, dict] = {}
    seen: set[tuple] = set()

    # split 0 = the original fixed split (connects multi-split analysis to Table 1)
    splits["split0"] = {
        "train": sorted(orig["train"]), "val": sorted(orig["val"]),
        "test": sorted(orig["test"]), "gen_seed": None,
        "note": "original fixed split from configs/splits.json",
    }
    seen.add((tuple(splits["split0"]["train"]), tuple(splits["split0"]["val"]),
              tuple(splits["split0"]["test"])))

    used_test_sets = {tuple(splits["split0"]["test"])}
    for i, gs in enumerate(GEN_SEEDS, start=1):
        rng = np.random.RandomState(gs)
        for _attempt in range(1000):
            perm = [str(p) for p in rng.permutation(patients)]
            test = sorted(perm[:N_TEST])
            val = sorted(perm[N_TEST:N_TEST + N_VAL])
            train = sorted(perm[N_TEST + N_VAL:])
            key = (tuple(train), tuple(val), tuple(test))
            # require a DISTINCT test set per split: the analysis targets
            # patient-sampling variability, so no two splits may share a test set
            if key not in seen and tuple(test) not in used_test_sets:
                seen.add(key)
                used_test_sets.add(tuple(test))
                splits[f"split{i}"] = {"train": train, "val": val, "test": test,
                                       "gen_seed": gs}
                break
        else:
            raise RuntimeError(f"could not sample a unique split for seed {gs}")

    # composition report
    for sid, sp in splits.items():
        for role in ("train", "val", "test"):
            sp[f"{role}_composition"] = _composition(meta, sp[role])
        log.info("%s: test=%s (%d sections, %d spots) | val=%s | train=%d patients",
                 sid, sp["test"], sp["test_composition"]["n_sections"],
                 sp["test_composition"]["n_spots"], sp["val"], len(sp["train"]))

    out = Path(args.out)
    save_json({"n_splits": len(splits), "patients": patients,
               "design": "8 train / 2 val / 3 test patients; split0 = original fixed split",
               "splits": splits}, out)
    log.info("saved %d splits -> %s", len(splits), out)


if __name__ == "__main__":
    main()
