"""``PairClassifier`` base class (three-component refactor design §3.3, §5).

The pairwise classifier pathway: consumes an optional graph conditioning
(`PairConditioning`) plus already-encoded endpoint token states (`PairInputs`)
and emits one edge logit per pair. `cond=None` is the unconditioned path --
with a null generator this is exactly the pure B0 pairwise baseline,
numerically the same eval-time hard bypass realized today by
``masks_for_null(NULL_ALL_HEAD, ...)`` (`conditioning.py:42-49`).

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

import torch
from torch import nn

from src.model.egostitch.conditioning import HeadNullMasks
from src.model.egostitch.graph import PairConditioning, PairInputs


class PairClassifier(nn.Module, ABC):
    """Consumes `PairInputs` and optional `PairConditioning`, emits `(B,)` logits.

    AB/BA symmetrization is internal to `forward` -- this is the design §4
    hard invariant. A compliant implementation must run both endpoint orders
    as *one* trunk batch so every conditioning layer centers on the joint
    AB-union-BA statistic and advances its EMA exactly once per call. Two
    separate trunk calls would center twice against two different means,
    break `ema_updates == 1`, and break train/eval agreement (evaluation
    reads the single frozen `ema_mu`, `conditioning.py:203`).

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
