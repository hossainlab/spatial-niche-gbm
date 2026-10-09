"""Graph-geometry sensitivity: LOPO final GraphSAGE/GAT runs on exact hexagonal
6-NN graphs (results/revision_hex, data/preprocessed_hex) vs the original graphs
built on Visium array indices (results/revision). Same folds, configs and seeds.

Usage: python scripts/23_hex_sensitivity.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

ORIG, HEX = Path("results/revision"), Path("results/revision_hex")
SEEDS = [42, 7, 123]


def f1m(y, yh):
    return float(f1_score(y, yh, average="macro", labels=sorted(set(y.tolist())), zero_division=0))


def pooled_and_pp(rev, model, folds):
    pooled, pp = [], {p: [] for p in folds}
    for s in SEEDS:
        Y, YH = [], []
        for p in folds:
            z = np.load(rev / "preds" / f"final_{model}_{p}_s{s}.npz"); m = z["y"] >= 0
            y, yh = z["y"][m], (z["prob"][m] / z["thr"]).argmax(1)
            Y.append(y); YH.append(yh); pp[p].append(f1m(y, yh))
        pooled.append(f1m(np.concatenate(Y), np.concatenate(YH)))
    return np.array(pooled), np.array([np.mean(pp[p]) for p in folds])


def main():
    folds = list(json.loads((ORIG / "folds.json").read_text()))
    out = {}
    _, lg = pooled_and_pp(ORIG, "logistic", folds)
    lg_pooled = pooled_and_pp(ORIG, "logistic", folds)[0].mean()
    for m in ["sage", "gat"]:
        po, ppo = pooled_and_pp(ORIG, m, folds)
        ph, pph = pooled_and_pp(HEX, m, folds)
        out[m] = {"pooled_orig": [po.mean(), po.std(ddof=1)], "pooled_hex": [ph.mean(), ph.std(ddof=1)],
                  "per_patient_hex_minus_orig": float((pph - ppo).mean()),
                  "wilcoxon_p_hex_vs_orig": float(stats.wilcoxon(pph, ppo).pvalue),
                  "logistic_minus_hex_pooled": float(lg_pooled - ph.mean()),
                  "logistic_wins_vs_hex": int((lg > pph).sum()),
                  "wilcoxon_p_logistic_vs_hex": float(stats.wilcoxon(lg, pph).pvalue)}
        print(m, json.dumps(out[m]))
    (HEX / "hex_sensitivity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
