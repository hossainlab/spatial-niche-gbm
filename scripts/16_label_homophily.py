"""Label homophily of the spatial graphs: for each labeled spot, the fraction of
its labeled k-NN neighbors that carry the same niche label. High homophily means
neighbors are informative about a spot's label (favorable for message passing);
low homophily means neighbor averaging mixes in other classes.

Also reports the chance level (expected same-label fraction if labels were
shuffled within each section) so homophily can be read relative to composition.

Usage: python scripts/16_label_homophily.py --data data/preprocessed --out results/revision
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CLASSES = ["immune_cold", "immune_hot", "normal", "necrotic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/preprocessed")
    ap.add_argument("--out", default="results/revision")
    args = ap.parse_args()
    graphs = torch.load(Path(args.data) / "graphs.pt", weights_only=False)

    same = {c: [] for c in CLASSES}
    chance = {c: [] for c in CLASSES}
    per_patient = {}
    for g in graphs:
        y = g.niche_y.numpy()
        src, dst = g.edge_index.numpy()
        keep = (y[src] >= 0) & (y[dst] >= 0)
        src, dst = src[keep], dst[keep]
        lab = y[y >= 0]
        frac = np.array([(lab == i).mean() if len(lab) else 0 for i in range(4)])
        agree = (y[src] == y[dst]).astype(float)
        for i, c in enumerate(CLASSES):
            m = y[src] == i
            if m.any():
                # per-spot mean over its neighbors, then pooled over spots
                s = np.bincount(src[m], weights=agree[m], minlength=len(y))
                n = np.bincount(src[m], minlength=len(y))
                v = s[n > 0] / n[n > 0]
                same[c].extend(v.tolist())
                chance[c].extend([frac[i]] * len(v))
        pp = per_patient.setdefault(g.patient, [0.0, 0])
        pp[0] += agree.sum(); pp[1] += len(agree)

    res = {
        "per_class": {c: {"n_spots": len(same[c]),
                          "homophily": float(np.mean(same[c])),
                          "chance": float(np.mean(chance[c])),
                          "frac_spots_majority_neighbors_differ": float(np.mean(np.array(same[c]) < 0.5))}
                      for c in CLASSES},
        "overall_edge_homophily": float(sum(v[0] for v in per_patient.values()) /
                                        sum(v[1] for v in per_patient.values())),
        "per_patient_edge_homophily": {p: v[0] / v[1] for p, v in sorted(per_patient.items())},
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "label_homophily.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["per_class"], indent=1)); print("overall", res["overall_edge_homophily"])


if __name__ == "__main__":
    main()
