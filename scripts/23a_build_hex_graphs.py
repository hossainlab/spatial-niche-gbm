"""Build the exact-hexagonal graphs used by the graph-geometry sensitivity analysis.

The SNUH atlas coordinates are Visium array indices, in which the column index is
doubled, so the 6-NN graphs of data/preprocessed join about 24% second-ring spots.
This script rebuilds every section's graph on true hexagonal coordinates
(array column x 1/2, array row x sqrt(3)/2) and leaves everything else unchanged:
features, labels, HVG panel and section metadata are copied from the source.

The output is the input of the hex LOPO runs (results/revision_hex) summarised by
scripts/23_hex_sensitivity.py. Re-running this script reproduces the graphs used
in the manuscript exactly (edge sets and edge weights).

Usage: python scripts/23a_build_hex_graphs.py [--src data/preprocessed] [--out data/preprocessed_hex]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_niche_gbm.data.graph import build_knn_graph  # noqa: E402

HEX_SCALE = np.array([0.5, np.sqrt(3) / 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/preprocessed")
    ap.add_argument("--out", default="data/preprocessed_hex")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    graphs = torch.load(src / "graphs.pt", weights_only=False)
    for g in graphs:
        coords = np.load(src / f"{g.section}_coords.npy") * HEX_SCALE
        g.edge_index, g.edge_attr = build_knn_graph(coords, args.k)
        np.save(out / f"{g.section}_coords.npy", coords)
    torch.save(graphs, out / "graphs.pt")
    for name in ("hvg_genes.json", "sections_meta.csv"):
        shutil.copy(src / name, out / name)
    print(f"wrote {len(graphs)} hexagonal graphs to {out}")


if __name__ == "__main__":
    main()
