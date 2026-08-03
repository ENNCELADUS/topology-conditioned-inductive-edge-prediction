"""``NeighborhoodGenerator`` base class (three-component refactor design §3.3, §5).

Split into a cacheable per-node phase and a pair-level phase (design
correction, 2026-08-03): a single fused ``forward(x_a, x_b, ...)`` would
re-encode both endpoints for every pair, but `src/score_universe.py:1859-1926`
and `src/train_egostitch.py:2950-3078` encode each node in a scored/validated
universe exactly once, cache that state, and reassemble per-pair batches from
it by index-select / index-copy / concatenation (`E2ENodeState` today).
`encode_node` is that cacheable half; `stitch` is the pair-level half that
consumes two already-encoded states; `forward` stays as the convenience
composition for callers that do not need cross-pair reuse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Generic, TypeVar

import torch
from torch import nn

from src.model.egostitch.graph import ImaginedGraph

NodeStateT = TypeVar("NodeStateT")
"""A generator's own cacheable per-node state type, opaque to every other component."""

PerturbationT = TypeVar("PerturbationT")
"""A generator's own deterministic scaffold-structure control input type."""

GraphT = TypeVar("GraphT", bound=ImaginedGraph)
"""A generator's own imagined-graph type (always an `ImaginedGraph` or subclass)."""


class NeighborhoodGenerator(nn.Module, ABC, Generic[NodeStateT, PerturbationT, GraphT]):
    """Consumes two endpoints' features and grounding pools, emits one graph (or none).

    Concrete generators own every auxiliary loss that supervises their own
    imagination (design §6): a generator swap swaps its losses with it,
    because those losses read only this generator's own `graph.aux`, which
    is otherwise off-limits to every other component (`graph.py`).

    Generic over ``NodeStateT`` (the per-node cache type `encode_node`
    produces and `stitch` consumes), ``PerturbationT`` (the scaffold-control
    input type `stitch` accepts) and ``GraphT`` (the concrete `ImaginedGraph`
    subclass `stitch` emits): a concrete generator binds all three to its own
    types (e.g. ``NeighborhoodGenerator[GeneratorNodeState,
    ScaffoldInputPerturbation, StitchedGraph]``), which lets `encode_node`/
    `stitch`/`auxiliary_losses` be overridden with that generator's own exact
    argument and return types instead of the opaque `object` every generator
    would otherwise be forced to share -- a second generator implementation
    (`NullGenerator` is the existing one) binds its own, different types the
    same way.
    """

    @abstractmethod
    def encode_node(
        self,
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
    ) -> NodeStateT:
        """Run the cacheable per-node half of imagination for one endpoint batch.

        Callers that score many pairs over a shared node universe (candidate
        scoring, validation) call this exactly once per unique node, cache
        the result, and reassemble per-pair batches from it by
        index-select / index-copy / concatenation -- the same pattern
        `E2ENodeState` supports today (`e2e_model.py`,
        `score_universe.py:1859-1882`, `train_egostitch.py:2950-2974`).

        Args:
            x: Shape ``(B, d)`` frozen node features.
            ground: Shape ``(B, n_g, d)`` grounding-candidate features.
            ground_ids: Optional shape ``(B, n_g)`` grounding-candidate
                global ids, carried through unchanged for the caller's own
                bookkeeping.

        Returns:
            A generator-specific, cacheable per-node state. The concrete
            type is this generator's own contract -- every other component
            treats it as opaque, exactly like `aux`.
        """
        raise NotImplementedError

    @abstractmethod
    def stitch(
        self,
        state_a: NodeStateT,
        state_b: NodeStateT,
        is_self: torch.Tensor,
        *,
        perturbation: PerturbationT | None = None,
    ) -> GraphT | None:
        """Imagine the joint graph from two already-encoded endpoint states.

        Args:
            state_a: Endpoint-A state from `encode_node`.
            state_b: Endpoint-B state from `encode_node`.
            is_self: Shape ``(B,)`` boolean; ``True`` marks self-pairs (spec
                Sec 13.9's single-ego path), which a generator that aligns
                the two sides must treat as an exact identity rather than
                running its alignment mechanism on a pairing it should
                already know is trivial.
            perturbation: Optional deterministic scaffold-structure control
                (e.g. the mandatory structure-control arms), applied before
                any graph channel is derived. ``None`` leaves the graph
                unperturbed. A generator with no such mechanism may ignore
                a non-``None`` value or reject it; that is its own contract.

        Returns:
            One `ImaginedGraph`, or `None` for the null case.
        """
        raise NotImplementedError

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        ground_a: torch.Tensor,
        ground_b: torch.Tensor,
        *,
        is_self: torch.Tensor,
        perturbation: PerturbationT | None = None,
    ) -> GraphT | None:
        """Convenience composition: `encode_node` both endpoints, then `stitch`.

        Callers that do not need cross-pair node-state reuse (a single
        one-off pair batch) can call this directly. Callers that score many
        pairs over a shared node universe should call `encode_node` once per
        unique node and `stitch` per pair batch instead -- see
        `encode_node`.

        Args:
            x_a: Shape ``(B, d)`` endpoint-A node features.
            x_b: Shape ``(B, d)`` endpoint-B node features.
            ground_a: Shape ``(B, n_g, d)`` endpoint-A grounding candidates.
            ground_b: Shape ``(B, n_g, d)`` endpoint-B grounding candidates.
            is_self: Shape ``(B,)`` boolean self-pair mask.
            perturbation: Optional deterministic scaffold-structure control,
                forwarded to `stitch` unchanged.

        Returns:
            One `ImaginedGraph`, or `None` for the null case.
        """
        state_a = self.encode_node(x_a, ground_a)
        state_b = self.encode_node(x_b, ground_b)
        return self.stitch(state_a, state_b, is_self, perturbation=perturbation)

    @abstractmethod
    def auxiliary_losses(
        self, graph: GraphT | None, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute every loss that supervises this generator's own imagination.

        Args:
            graph: This generator's own most recent output for `batch` (or
                `None`, for a null generator).
            batch: The training batch; the concrete keys a generator reads
                are its own contract to document.

        Returns:
            Named, unweighted loss components. The composite applies the
            registered weights (design §6).
        """
        raise NotImplementedError
