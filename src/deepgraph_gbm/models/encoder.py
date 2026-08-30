"""Graph encoders: GraphSAGE (mean aggregation) and GAT (attention aggregation)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, SAGEConv


class GraphSAGEEncoder(nn.Module):
    """Stack of SAGEConv layers with BatchNorm, ReLU and dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] = (256, 256, 128),
        dropout: float = 0.3,
        aggr: str = "mean",
    ):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1], aggr=aggr) for i in range(len(dims) - 1)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(d) for d in hidden_dims])
        self.dropout = dropout
        self.out_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    """Stack of GATConv layers with BatchNorm, ReLU and dropout.

    Multi-head attention aggregation (default 4 heads): hidden layers concatenate
    heads (out_dim // heads per head), the final layer averages them. Same hidden
    dims and dropout as GraphSAGEEncoder so the two encoders differ only in how
    neighbor information is aggregated (learned attention vs uniform mean).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] = (256, 256, 128),
        dropout: float = 0.3,
        heads: int = 4,
    ):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        convs = []
        for i in range(len(dims) - 1):
            last = i == len(dims) - 2
            if last:
                convs.append(GATConv(dims[i], dims[i + 1], heads=heads,
                                     concat=False, dropout=dropout))
            else:
                if dims[i + 1] % heads != 0:
                    raise ValueError(f"hidden dim {dims[i + 1]} not divisible by heads={heads}")
                convs.append(GATConv(dims[i], dims[i + 1] // heads, heads=heads,
                                     concat=True, dropout=dropout))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.BatchNorm1d(d) for d in hidden_dims])
        self.dropout = dropout
        self.out_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
