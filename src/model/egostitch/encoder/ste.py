"""`TypedMessagePassingEncoder` -- behaviour-preserving port of `STEncoder`.

Today's `STEncoder` (`src/model/egostitch/ste.py`) hardcodes `FEAT_DIM = 11`
and `EDGE_TYPES = 4` as module-level constants imported from `scaffold.py`
and consumes a `ScaffoldTokens`. That is precisely why no off-the-shelf GNN
can occupy this slot (design §1). This module ports the same architecture --
per-relation linear message, degree-normalised aggregation, residual +
LayerNorm + MLP -- onto the `ImaginedGraph` contract, with every dimension
read from constructor arguments (in turn read from the graph the generator
actually emits) instead of module constants.

Numerical equivalence with `STEncoder.forward` for the same weights and
input is the proof the port is safe; it is asserted in
`tests/model/test_encoder_component.py` by loading one module's `state_dict`
into the other.
"""

from __future__ import annotations

import torch
from torch import nn

from src.model.egostitch.encoder.base import GraphEncoder
from src.model.egostitch.graph import GraphEmbedding, ImaginedGraph


class _MPLayer(nn.Module):
    """One typed message-passing layer -- identical architecture to `ste._MPLayer`.

    Parameterised by `num_relations` instead of the module-constant
    `EDGE_TYPES`, which is the only difference from the original.
    """

    def __init__(self, dim: int, num_relations: int) -> None:
        super().__init__()
        self.msg = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_relations))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Aggregate per-relation messages, then residual + LayerNorm + MLP."""
        agg = torch.zeros_like(h)
        for t, lin in enumerate(self.msg):
            agg = agg + torch.bmm(adj[:, t], lin(h))
        h = self.norm1(h + agg)
        out: torch.Tensor = self.norm2(h + self.mlp(h))
        return out


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted mean over the node axis, finite for an all-zero mask row.

    Args:
        tokens: Shape `(B, N, d)`.
        mask: Shape `(B, N)`, soft node existence in `[0, 1]`.

    Returns:
        Shape `(B, d)`. Rows whose mask sums to zero return the zero vector
        rather than `nan` (the numerator is also zero there, and the
        denominator is clamped away from zero).
    """
    weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    numerator = (tokens * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp(min=1e-8)
    return numerator / denominator


class TypedMessagePassingEncoder(GraphEncoder):
    """Structure-only typed message-passing encoder over an `ImaginedGraph`.

    The behaviour-preserving port of `STEncoder`
    (`src/model/egostitch/ste.py`): `in_dim`/`num_relations` replace the
    `FEAT_DIM`/`EDGE_TYPES` module constants as constructor arguments, read
    at construction time from the graph the generator actually emits
    (`ImaginedGraph.feature_dim`, `.num_relations`). Everything else --
    per-relation linear message, row-normalised adjacency aggregation,
    residual + LayerNorm + MLP stack -- is unchanged.
    """

    def __init__(
        self,
        in_dim: int,
        num_relations: int,
        d_model: int,
        dim: int = 128,
        layers: int = 3,
        *,
        w_rel: float = 0.25,
    ) -> None:
        """Build the encoder.

        Args:
            in_dim: `F` -- node feature width the graph's `x` must carry.
                Read from `ImaginedGraph.feature_dim` by the caller, not
                hardcoded (contrast today's module-level `FEAT_DIM`).
            num_relations: `R` -- typed-adjacency relation count. Read from
                `ImaginedGraph.num_relations` by the caller (contrast
                today's module-level `EDGE_TYPES`).
            d_model: Output token/pooled width -- what a downstream
                classifier conditions on.
            dim: Hidden width of the message-passing stack. Default `128`
                matches `STEncoder`'s `ste_dim` default.
            layers: Message-passing depth. Default `3` matches `STEncoder`.
            w_rel: Relational auxiliary-loss weight, forwarded to
                `GraphEncoder.__init__`. `0.0` omits the relational head.

        Raises:
            ValueError: If any dimension argument is not positive.
        """
        super().__init__(d_model=d_model, w_rel=w_rel)
        if in_dim <= 0 or num_relations <= 0 or d_model <= 0 or dim <= 0 or layers <= 0:
            raise ValueError(
                "in_dim, num_relations, d_model, dim, and layers must all be positive; got "
                f"in_dim={in_dim}, num_relations={num_relations}, d_model={d_model}, "
                f"dim={dim}, layers={layers}"
            )
        self.in_dim = in_dim
        self.num_relations = num_relations
        self.out_dim = d_model
        self.embed = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList(_MPLayer(dim, num_relations) for _ in range(layers))
        self.out = nn.Linear(dim, d_model)

    def forward(self, graph: ImaginedGraph) -> GraphEmbedding:
        """Encode `graph` to token and mask-weighted-pooled embeddings.

        Args:
            graph: Must carry `x.shape[-1] == self.in_dim` and
                `adj.shape[1] == self.num_relations`. Never reads `graph.aux`.

        Returns:
            `tokens` numerically identical to `STEncoder.forward` for the
            same weights and `(x, adj)`; `pooled` is new, a mask-weighted
            mean over the node axis.

        Raises:
            ValueError: On a feature-width or relation-count mismatch (fails
                closed, as `STEncoder.forward` does today).
        """
        if graph.x.ndim != 3 or graph.x.shape[-1] != self.in_dim:
            raise ValueError(
                f"graph feature channel count must be {self.in_dim}, got {tuple(graph.x.shape)}"
            )
        batch, nodes, _ = graph.x.shape
        expected_adj = (batch, self.num_relations, nodes, nodes)
        if graph.adj.shape != expected_adj:
            raise ValueError(
                f"graph adjacency shape must be {expected_adj}, got {tuple(graph.adj.shape)}"
            )
        # normalize adjacency rows so hub scaffolds do not blow up activations
        # (preserved verbatim from `ste.py:58-59`)
        deg = graph.adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj = graph.adj / deg
        h = self.embed(graph.x)
        for layer in self.layers:
            h = layer(h, adj)
        tokens: torch.Tensor = self.out(h)
        pooled = _masked_mean(tokens, graph.mask)
        return GraphEmbedding(tokens=tokens, pooled=pooled)
