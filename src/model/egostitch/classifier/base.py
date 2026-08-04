"""``PairClassifier`` base class (three-component refactor design §3.3, §5).

The pairwise classifier pathway: consumes an optional graph conditioning
(`PairConditioning`) plus already-encoded endpoint token states (`PairInputs`)
and emits one edge logit per pair. `cond=None` is the unconditioned path --
with a null generator this is exactly the pure B0 pairwise baseline,
numerically the same eval-time hard bypass realized today by
``masks_for_null(NULL_ALL_HEAD, ...)`` (`b0_v31.py`).

Split into a cacheable per-node phase and a pair-level phase (correction,
2026-08-03, mirroring `NeighborhoodGenerator.encode_node`/`.stitch`): a
`forward` that re-derives its own token encoding from raw tokens on every
call defeats the per-node encoding cache callers that score many pairs over
a shared node universe depend on (`score_universe.py`'s ``node_cache``,
documented at `score_universe.py:1713` as encoding "each unique node exactly
once"). `encode_tokens` is that cacheable half; `forward` consumes only its
*output*, via `PairInputs.tokens_a`/`tokens_b`, and never calls its own token
encoder itself.

Unlike the generator and the encoder, the classifier owns no auxiliary loss
of its own: design §6 assigns ``L_edge`` to the trainer, not the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
from torch import nn

from src.model.egostitch.graph import PairConditioning, PairInputs


class HeadNullMasks(NamedTuple):
    """Per-pair pathway activity masks (True = active).

    Defined here (rather than in `b0_v31.py`, where the mask-producing
    helpers `sample_branch_masks`/`masks_for_null` live) so this base module
    can carry it in `PairClassifier.forward`'s signature without importing
    `b0_v31.py` -- which itself imports `PairClassifier` from here, and a
    reverse import would be circular.
    """

    topo: torch.Tensor


class PairClassifier(nn.Module, ABC):
    """Consumes `PairInputs` and optional `PairConditioning`, emits `(B,)` logits.

    AB/BA symmetrization is internal to `forward` -- this is the design §4
    hard invariant. A compliant implementation must run both endpoint orders
    as *one* trunk batch so every conditioning layer centers on the joint
    AB-union-BA statistic and advances its EMA exactly once per call. Two
    separate trunk calls would center twice against two different means,
    break `ema_updates == 1`, and break train/eval agreement (evaluation
    reads the single frozen `ema_mu`, `b0_v31.py`).

    Must never see endpoint features or grounding pools -- `PairInputs`
    deliberately excludes them. That exclusion is what makes the
    null-generator configuration a true pairwise baseline rather than a
    masked variant of the full model.
    """

    @abstractmethod
    def encode_tokens(self, emb: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Run the cacheable per-node token-encoding half for one endpoint batch.

        Callers that score many pairs over a shared node universe (candidate
        scoring, validation) call this exactly once per unique node, cache
        the result, and reassemble per-pair batches from it by
        index-select / index-copy / concatenation -- the same pattern
        `E2ENodeState.encoded` supports (`composite.py`,
        `score_universe.py:1852` populating ``node_cache``).

        Args:
            emb: Shape ``(B, T, d_in)`` raw per-node token stream.
            length: Shape ``(B,)`` unpadded token counts.

        Returns:
            Shape ``(B, T', d_model)`` encoded token states -- what
            `PairInputs.tokens_a`/`tokens_b` carry.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        pair: PairInputs,
        cond: PairConditioning | None,
        *,
        masks: HeadNullMasks | None = None,
    ) -> torch.Tensor:
        """Score one pair batch.

        Args:
            pair: Already-encoded per-endpoint token states (`encode_tokens`'s
                output), lengths, and the DDP filler-row `edge_mask`. Must not
                call `encode_tokens` itself -- that is the caller's job,
                exactly once per unique node.
            cond: Both orientations of the graph conditioning, or `None` for
                the unconditioned baseline -- numerically the same bypass as
                `masks_for_null(NULL_ALL_HEAD, ...)` with `cond` present,
                because a fully gated-off topo pathway and an absent one both
                contribute exactly zero residual.
            masks: Optional per-pair topo-pathway activity mask (design §3.5
                head-null conditions). `None` defaults to the topo pathway
                fully active whenever `cond` is not `None`; it has no effect
                when `cond` is `None` since there is no topology to gate.

        Returns:
            Shape `(B,)` edge logits.
        """
        raise NotImplementedError

    def freeze_unreachable_conditioning(self) -> None:
        """Freeze any conditioning-only submodule that can never receive `cond`.

        Called by `EgoStitchModel.__init__` (`composite.py`) exactly when the
        composed model's generator never imagines a graph
        (`generator.name: "null"` -> `EgoStitchModel.encoder is None`), which
        is precisely the condition under which `score_pair_context` clamps
        `need_topo` off and this classifier's `forward` is *always* called
        with `cond=None`. A submodule that only activates when `cond is not
        None` therefore never runs a single forward pass under that
        composition, so it never receives a gradient -- DDP's
        `find_unused_parameters=False` (CLAUDE.md P1 trap) raises on the
        first backward otherwise.

        Freezing (`requires_grad_(False)`) rather than not constructing the
        submodule at all keeps `state_dict` key layout, parameter count, and
        RNG draw sequence identical to the live-encoder arms (a scoring-only
        null-generator checkpoint must keep loading), and keeps any
        unconditional readout of the submodule (e.g. telemetry) working.

        Default no-op: a classifier with no such submodule -- or one that is
        never reachable under a null generator in the first place -- has
        nothing to freeze. Override where applicable (`B0V31PairClassifier`).
        """
        return None
