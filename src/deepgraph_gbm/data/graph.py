"""kNN graph construction from Visium spot coordinates."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    coords: np.ndarray, k: int = 6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a symmetric kNN graph from 2D spot coordinates.

    Parameters
    ----------
    coords : (n_spots, 2) array of spot coordinates (array or pixel units).
    k : number of nearest neighbors (6 for Visium hex grid).

    Returns
    -------
    edge_index : LongTensor (2, n_edges), symmetrized, no self-loops.
    edge_attr : FloatTensor (n_edges, 1), Gaussian-kernel edge weights
        exp(-d^2 / (2 * median_d^2)) where d is the Euclidean distance.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    if n < 2:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))

    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords)
    dists, idx = nn.kneighbors(coords)

    # drop self (first column)
    src = np.repeat(np.arange(n), k_eff)
    dst = idx[:, 1:].ravel()
    d = dists[:, 1:].ravel()

    # symmetrize: keep union of directed edges
    edges = np.concatenate(
        [np.stack([src, dst], 0), np.stack([dst, src], 0)], axis=1
    )
    edges = np.unique(edges, axis=1)

    # recompute distances for the symmetrized edge set
    diff = coords[edges[0]] - coords[edges[1]]
    dist = np.sqrt((diff**2).sum(axis=1))
    median_d = np.median(dist[dist > 0]) if (dist > 0).any() else 1.0
    weights = np.exp(-(dist**2) / (2 * median_d**2))

    edge_index = torch.from_numpy(edges).long()
    edge_attr = torch.from_numpy(weights).float().unsqueeze(1)
    return edge_index, edge_attr
