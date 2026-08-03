"""The interfaces between the three swappable components (design 2026-08-02 §3).

One graph type flows generator -> encoder; one embedding type flows encoder ->
classifier. Both are deliberately plain: node/edge dimensions are read from the
tensors at runtime, never from module-level constants, which is what lets a
stock GNN occupy the encoder slot without touching the generator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class ImaginedGraph:
    """One imagined local graph around a queried pair.

    The generator emits exactly this; the encoder consumes exactly this. No
    other type crosses that seam.

    Attributes:
        x: Shape ``(B, N, F)`` node features. ``F`` is whatever the generator
            chose to expose and is read at runtime -- the `egostitch_imagine`
            generator emits the 11 structural channels, but a generator that
            also exposes slot content simply emits a wider ``F`` and needs no
            interface or encoder change.
        adj: Shape ``(B, R, N, N)`` typed soft adjacency, one dense matrix per
            relation. Weights are continuous, so the whole path stays
            differentiable.
        mask: Shape ``(B, N)`` soft node existence in ``[0, 1]``. Node presence
            is also encoded inside ``x`` for encoders that read it positionally;
            this field is the layout-independent form.
        aux: Generator-private tensors -- slot-level state, alignment plans,
            anything that generator's own losses and telemetry need.
            **No encoder may read this.** Swapping the generator invalidates
            every key, which is correct: what reads `aux` belongs to the
            generator that produced it.
        directed: Whether ``x``'s role channels are labelled for the AB stream.
            :meth:`swapped` flips it.
    """

    x: torch.Tensor
    adj: torch.Tensor
    mask: torch.Tensor
    aux: Mapping[str, torch.Tensor]
    directed: bool = True

    def __post_init__(self) -> None:
        """Validate the shape agreement the encoder contract depends on.

        Raises:
            ValueError: If the three tensors disagree on batch size or node
                count, or if the ranks are wrong.
        """
        if self.x.ndim != 3:
            raise ValueError(f"x must be (B, N, F), got {tuple(self.x.shape)}")
        if self.adj.ndim != 4:
            raise ValueError(f"adj must be (B, R, N, N), got {tuple(self.adj.shape)}")
        if self.mask.ndim != 2:
            raise ValueError(f"mask must be (B, N), got {tuple(self.mask.shape)}")
        batch, nodes, _ = self.x.shape
        if self.adj.shape[0] != batch or self.adj.shape[2:] != (nodes, nodes):
            raise ValueError(
                f"adj must be {(batch, self.adj.shape[1], nodes, nodes)}, "
                f"got {tuple(self.adj.shape)}"
            )
        if self.mask.shape != (batch, nodes):
            raise ValueError(f"mask must be {(batch, nodes)}, got {tuple(self.mask.shape)}")

    @property
    def num_nodes(self) -> int:
        """``N``."""
        return int(self.x.shape[1])

    @property
    def num_relations(self) -> int:
        """``R``."""
        return int(self.adj.shape[1])

    @property
    def feature_dim(self) -> int:
        """``F`` -- what an encoder sizes its input projection from."""
        return int(self.x.shape[2])

    def swapped(self) -> ImaginedGraph:
        """Return the graph relabelled for the BA stream.

        The undirected edge task is scored in both endpoint orders and the two
        pair representations fused, so the conditioning must be presented in
        both orientations. Only direction-carrying node features change;
        ``adj``, ``mask`` and ``aux`` are shared by reference, because the graph
        itself is built exactly once per pair.

        Subclasses that carry directional information in other channels must
        override this. The base implementation delegates to `swap_node_roles`,
        which the generator supplies.
        """
        raise NotImplementedError(
            "ImaginedGraph.swapped requires a generator-specific relabelling; "
            "use the subclass the generator emits"
        )

    def with_x(self, x: torch.Tensor, *, directed: bool) -> ImaginedGraph:
        """Return a copy carrying relabelled node features, sharing everything else."""
        return replace(self, x=x, directed=directed)


@dataclass(frozen=True)
class GraphEmbedding:
    """What an encoder returns.

    Attributes:
        tokens: Shape ``(B, N, d)`` per-node states. The `b0_v31` classifier
            attends over these.
        pooled: Shape ``(B, d)`` graph-level summary. Present so a classifier
            that conditions by FiLM or addition needs no encoder change.
    """

    tokens: torch.Tensor
    pooled: torch.Tensor

    def __post_init__(self) -> None:
        """Validate rank and batch agreement.

        Raises:
            ValueError: On rank mismatch or disagreeing batch/width.
        """
        if self.tokens.ndim != 3:
            raise ValueError(f"tokens must be (B, N, d), got {tuple(self.tokens.shape)}")
        if self.pooled.ndim != 2:
            raise ValueError(f"pooled must be (B, d), got {tuple(self.pooled.shape)}")
        if self.pooled.shape[0] != self.tokens.shape[0]:
            raise ValueError("tokens and pooled disagree on batch size")
        if self.pooled.shape[1] != self.tokens.shape[2]:
            raise ValueError("tokens and pooled disagree on width")


@dataclass(frozen=True)
class PairConditioning:
    """Both orientations of the conditioning for one pair batch.

    AB and BA must reach the classifier together: every gated cross-attention
    block centers on the mean over the joint AB-union-BA batch and advances its
    EMA exactly once per step. Presenting one orientation at a time would center
    twice against two different statistics and break train/eval agreement.
    """

    ab: GraphEmbedding
    ba: GraphEmbedding


@dataclass(frozen=True)
class PairInputs:
    """What the pair classifier's pair-level phase consumes.

    Deliberately free of endpoint features and grounding pools: those go to the
    generator and never here. That is what makes a null generator yield a true
    pairwise baseline rather than a masked variant of the full model.

    ``tokens_a``/``tokens_b`` are **already encoded** -- the output of
    `PairClassifier.encode_tokens`, not the raw per-node token stream that
    feeds it (correction, 2026-08-03: the first cut of this contract carried
    raw tokens and had `PairClassifier.forward` re-run its own token encoder
    every call, which silently defeated the per-node encoding cache callers
    that score many pairs over a shared node universe depend on --
    `score_universe.py`'s ``node_cache``. Splitting `PairClassifier` into a
    cacheable `encode_tokens` phase and a pair-level `forward` phase, mirroring
    `NeighborhoodGenerator.encode_node`/`.stitch`, is the fix: `PairInputs`
    now carries the cacheable phase's *output*, and `forward` never re-derives
    it.

    Attributes:
        tokens_a: Shape ``(B, T_a, d_model)`` already-encoded per-node token
            states for endpoint A, produced by `PairClassifier.encode_tokens`.
        tokens_b: Shape ``(B, T_b, d_model)`` already-encoded per-node token
            states for endpoint B.
        len_a: Shape ``(B,)`` unpadded token counts for A.
        len_b: Shape ``(B,)`` unpadded token counts for B.
        edge_mask: Shape ``(B,)`` real-row mask. DDP filler rows must be
            excluded from conditioning statistics.
    """

    tokens_a: torch.Tensor
    tokens_b: torch.Tensor
    len_a: torch.Tensor
    len_b: torch.Tensor
    edge_mask: torch.Tensor | None = None
