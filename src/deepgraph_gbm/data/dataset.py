"""PyG dataset of per-section Visium graphs with niche/MES labels."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data, InMemoryDataset

from ..data.graph import build_knn_graph
from ..data.labels import encode_niche, mes_target

log = logging.getLogger("deepgraph_gbm")


class VisiumGraphDataset(InMemoryDataset):
    """One Data object per tissue section.

    Each Data carries:
      x (n, n_hvgs) log-normalized expression, edge_index/edge_attr (kNN graph),
      niche_y (n,) int64 (-1 = excluded), mes_y (n,) float (NaN = missing),
      risk_y scalar pseudo-risk, plus section/patient metadata.
    """

    def __init__(self, root: str | Path, transform=None):
        super().__init__(str(root), transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.meta = pd.read_csv(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ["sections.npz", "sections_meta.csv"]

    @property
    def processed_file_names(self):
        return ["graphs.pt", "graphs_meta.csv"]

    def download(self):
        raise FileNotFoundError(
            "Run scripts/02_build_graphs.py to create raw sections first."
        )

    def process(self):
        raise NotImplementedError("Use build_dataset() below.")


def build_graph_data(
    X: sp.csr_matrix,
    coords: np.ndarray,
    niche_labels: pd.Series,
    mes_y: pd.Series,
    risk_y: float,
    section: str,
    patient: str,
    k: int = 6,
) -> Data:
    """Assemble one PyG Data object for a section."""
    edge_index, edge_attr = build_knn_graph(coords, k=k)
    x = torch.from_numpy(X.toarray()).float() if sp.issparse(X) else torch.from_numpy(np.asarray(X)).float()
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        niche_y=torch.from_numpy(encode_niche(niche_labels)),
        mes_y=torch.from_numpy(mes_y.values.astype(np.float32)),
        risk_y=torch.tensor(float(risk_y), dtype=torch.float),
    )
    data.section = section
    data.patient = patient
    data.n_spots = x.shape[0]
    return data
