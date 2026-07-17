"""Module: stitched-topology encoder — edge-weighted MP over the scaffold.

The promoted s4 lineage (spec §5): per layer, one weight matrix per edge type,
messages aggregated by the weighted adjacency, residual + LayerNorm + MLP.
Token-level output (no pooled readout) — the learned topology representation.
"""

from __future__ import annotations

import torch
from torch import nn

from src.model.egostitch.scaffold import EDGE_TYPES, FEAT_DIM, ScaffoldTokens


class _MPLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.msg = nn.ModuleList(nn.Linear(dim, dim) for _ in range(EDGE_TYPES))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Aggregate per-edge-type messages, then residual + LayerNorm + MLP."""
        agg = torch.zeros_like(h)
        for t, lin in enumerate(self.msg):
            agg = agg + torch.bmm(adj[:, t], lin(h))
        h = self.norm1(h + agg)
        out: torch.Tensor = self.norm2(h + self.mlp(h))
        return out


class STEncoder(nn.Module):
    """Structure-only scaffold encoder producing token-level topology states."""

    def __init__(self, d_model: int, ste_dim: int = 128, n_layers: int = 3) -> None:
        super().__init__()
        self.embed = nn.Linear(FEAT_DIM, ste_dim)
        self.layers = nn.ModuleList(_MPLayer(ste_dim) for _ in range(n_layers))
        self.out = nn.Linear(ste_dim, d_model)

    def forward(self, scaffold: ScaffoldTokens) -> torch.Tensor:
        """Encode scaffold tokens to ``(B, V, d_model)`` conditioning states."""
        # normalize adjacency rows so hub scaffolds do not blow up activations
        deg = scaffold.adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj = scaffold.adj / deg
        h = self.embed(scaffold.feats)
        for layer in self.layers:
            h = layer(h, adj)
        out: torch.Tensor = self.out(h)
        return out
