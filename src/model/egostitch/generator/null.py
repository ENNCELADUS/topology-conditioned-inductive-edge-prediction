"""``NullGenerator``: makes the pure pairwise baseline reachable by config alone.

Design §3.3, §8: setting ``generator.name: null`` yields exactly ``cond=None``
at the composite, so the classifier runs unconditioned on `PairInputs` alone
-- numerically the existing eval-time hard bypass
(``masks_for_null(NULL_ALL_HEAD, ...)``, `conditioning.py:42-49`), now
reachable without a registered null condition. No parameters: there is
nothing to imagine, nothing to cache, and nothing to stitch. `forward` is
inherited from `NeighborhoodGenerator` unchanged (`encode_node` twice, then
`stitch`); since `stitch` always returns `None` regardless of its inputs,
that composition is `None` for every call.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from src.model.egostitch.generator.base import NeighborhoodGenerator
from src.model.egostitch.graph import ImaginedGraph


class NullGenerator(NeighborhoodGenerator):
    """Imagines nothing. `stitch` always returns `None`; no auxiliary losses."""

    def encode_node(
        self,
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the cheapest possible placeholder: `x` itself, unmodified.

        There is nothing to encode and nothing worth caching -- `stitch`
        never reads its `state_a`/`state_b` arguments -- so this simply
        echoes `x` back rather than allocating a dedicated state object.

        Args:
            x: Shape ``(B, d)`` node features, returned unchanged.
            ground: Unused.
            ground_ids: Unused.

        Returns:
            `x`, unchanged.
        """
        del ground, ground_ids
        return x

    def stitch(
        self,
        state_a: object,
        state_b: object,
        is_self: torch.Tensor,
        *,
        perturbation: object | None = None,
    ) -> ImaginedGraph | None:
        """Always opt out of conditioning.

        Args:
            state_a: Unused; accepted only to satisfy the shared interface.
            state_b: Unused.
            is_self: Unused.
            perturbation: Unused.

        Returns:
            `None`, unconditionally.
        """
        del state_a, state_b, is_self, perturbation
        return None

    def auxiliary_losses(
        self, graph: ImaginedGraph | None, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return no losses: nothing was imagined, so nothing needs supervision."""
        del graph, batch
        return {}
