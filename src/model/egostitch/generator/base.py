"""``NeighborhoodGenerator`` base class (three-component refactor design §3.3, §5).

Split into a cacheable per-node phase and a pair-level phase (design
correction, 2026-08-03): a single fused ``forward(x_a, x_b, ...)`` would
re-encode both endpoints for every pair, but `src/score_universe.py:1859-1926`
and `src/train_egostitch.py:2944-3072` encode each node in a scored/validated
universe exactly once, cache that state, and reassemble per-pair batches from
it by index-select / index-copy / concatenation (`E2ENodeState` today).
`encode_node` is that cacheable half; `stitch` is the pair-level half that
consumes two already-encoded states; `forward` stays as the convenience
composition for callers that do not need cross-pair reuse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn

from src.model.egostitch.graph import ImaginedGraph


class NeighborhoodGenerator(nn.Module, ABC):
    """Consumes two endpoints' features and grounding pools, emits one graph (or none).

    Concrete generators own every auxiliary loss that supervises their own
    imagination (design §6): a generator swap swaps its losses with it,
    because those losses read only this generator's own `graph.aux`, which
    is otherwise off-limits to every other component (`graph.py`).
    """

    @abstractmethod
    def encode_node(
        self,
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
    ) -> object:
        """Run the cacheable per-node half of imagination for one endpoint batch.

        Callers that score many pairs over a shared node universe (candidate
        scoring, validation) call this exactly once per unique node, cache
        the result, and reassemble per-pair batches from it by
        index-select / index-copy / concatenation -- the same pattern
        `E2ENodeState` supports today (`e2e_model.py`,
        `score_universe.py:1859-1882`, `train_egostitch.py:2944-2968`).

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
        state_a: object,
        state_b: object,
        is_self: torch.Tensor,
        *,
        perturbation: object | None = None,
    ) -> ImaginedGraph | None:
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
        perturbation: object | None = None,
    ) -> ImaginedGraph | None:
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
        self, graph: ImaginedGraph | None, batch: Mapping[str, torch.Tensor]
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
