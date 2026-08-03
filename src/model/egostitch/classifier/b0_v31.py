"""B0-V3.1 pairwise classifier: the mature baseline plus a conditioning socket.

Composes `SiameseEncoder` + `MLPHead` (`src.model.B0`) with
`ConditionedPairCrossAttention` (`src.model.egostitch.trunk`). The AB/BA
doubled-batch construction, the feature-wise max fuse, and the fp32 head
island are lifted verbatim from `EgoStitchE2E.score_pair_context`
(`src/model/egostitch/e2e_model.py:430-480`) -- this is exactly where design
§4's AB/BA invariant is realized: one trunk batch, one synchronized
centering statistic, one EMA update per call.

`SiameseEncoder` is split across `encode_tokens` (the cacheable per-node
phase) and `forward` (the pair-level phase, correction 2026-08-03): `forward`
consumes `pair.tokens_a`/`tokens_b` directly and never calls `self.encoder`
itself, so a caller scoring many pairs over a shared node universe encodes
each node once via `encode_tokens` and reuses it, instead of paying the
`SiameseEncoder` cost again on every pair.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.distributed as dist
from torch.nn import functional as F

from src.model.B0 import MLPHead, SiameseEncoder
from src.model.egostitch.classifier.base import PairClassifier
from src.model.egostitch.conditioning import NULL_NONE, HeadNullMasks, masks_for_null
from src.model.egostitch.graph import PairConditioning, PairInputs
from src.model.egostitch.trunk import ConditionedPairCrossAttention


class B0V31PairClassifier(PairClassifier):
    """The mature B0-V3.1 pairwise baseline with a topo-conditioning socket.

    `cond=None` takes the exact-bypass path: the topo cross-attention module
    is never invoked at all (not merely gated to zero). That is numerically
    identical to the `f_logit` / `NULL_ALL_HEAD` decomposition arm, because
    that arm's `active=False` gate already discards the whole pathway
    contribution unconditionally -- see the final `torch.where` in
    `GatedCrossAttention.forward` (`src/model/egostitch/conditioning.py:200`).
    Reproducing the hard-bypass form (rather than computing-then-discarding)
    also keeps this classifier importable with no generator or encoder in
    the loop at all, which is what makes the null-generator configuration a
    true, self-contained pairwise baseline.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int = 512,
        encoder_layers: int = 3,
        n_heads: int = 8,
        cross_attn_layers: int = 3,
        n_inj: int = 1,
        xattn_heads: int = 8,
        conditioning_ema_decay: float = 0.99,
        dropout: float = 0.1,
        token_dropout: float = 0.0,
    ) -> None:
        """Build the encoder, conditioned trunk, and head (`e2e_model.py:116-143`).

        Args:
            input_dim: Width `d_in` of the raw per-node token streams
                `encode_tokens` consumes (not `forward`'s `PairInputs`, which
                carries `encode_tokens`'s already-encoded `d_model`-wide
                output).
            d_model: Pair-trunk hidden width.
            encoder_layers: `SiameseEncoder` depth (per-item token encoder).
            n_heads: Attention heads for the item encoder and pair trunk.
            cross_attn_layers: `ConditionedPairCrossAttention` depth.
            n_inj: Trailing trunk layers receiving gated topo injection.
            xattn_heads: Gated cross-attention heads (topo pathway).
            conditioning_ema_decay: Decay for the synchronized
                conditioning-center EMA.
            dropout: Shared dropout rate for the encoder, trunk, and head.
            token_dropout: `SiameseEncoder` token-level dropout rate.
        """
        super().__init__()
        self.encoder = SiameseEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
            token_dropout=token_dropout,
        )
        self.trunk = ConditionedPairCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=cross_attn_layers,
            dropout=dropout,
            pair_readout_mode="pair_context_gated",
            mixing_mode="bidirectional_cross",
            n_inj=n_inj,
            xattn_heads=xattn_heads,
            conditioning_ema_decay=conditioning_ema_decay,
        )
        self.head = MLPHead(
            input_dim=d_model,
            hidden_dims=[d_model // 2],
            output_dim=1,
            dropout=dropout,
            activation="gelu",
            norm="layernorm",
        )

    def encode_tokens(self, emb: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Run the cacheable `SiameseEncoder` pass for one endpoint batch.

        See `PairClassifier.encode_tokens`. This is the *only* place
        `self.encoder` is called -- `forward` consumes only its output, via
        `PairInputs.tokens_a`/`tokens_b`.
        """
        # `nn.Module.__call__` is typed to return `Any`; `SiameseEncoder.forward`
        # itself is annotated `-> torch.Tensor`, so this narrows back to what
        # the module actually returns rather than widening the contract.
        return cast(torch.Tensor, self.encoder(emb, length))

    def forward(
        self,
        pair: PairInputs,
        cond: PairConditioning | None,
        *,
        masks: HeadNullMasks | None = None,
    ) -> torch.Tensor:
        """Score one pair batch (see `PairClassifier.forward`).

        AB and BA are run as one trunk batch (design §4): every conditioning
        layer therefore centers both directions with one synchronized
        statistic and updates its single EMA exactly once, regardless of
        `cond`. Consumes `pair.tokens_a`/`tokens_b` directly -- never calls
        `self.encoder` (that is `encode_tokens`'s job, exactly once per
        unique node, not once per pair).
        """
        batch_size = pair.tokens_a.size(0)
        device = pair.tokens_a.device
        if masks is None:
            masks = masks_for_null(NULL_NONE, batch_size, device)

        encoded_a = pair.tokens_a
        encoded_b = pair.tokens_b

        # `cond` is a whole-batch, model-wide decision (the generator either
        # emits a graph for every row or returns `None` for the whole step),
        # so forcing the topo module to fire under distributed training is
        # safe here exactly as it was in `score_pair_context`: every rank
        # sees the same `cond is not None` state.
        distributed_training = self.training and dist.is_available() and dist.is_initialized()
        need_topo = cond is not None and (bool(masks.topo.any()) or distributed_training)
        topo_tokens: torch.Tensor | None = None
        if need_topo:
            assert cond is not None
            topo_tokens = torch.cat((cond.ab.tokens, cond.ba.tokens))

        max_tokens = max(encoded_a.size(1), encoded_b.size(1))
        encoded_a = F.pad(encoded_a, (0, 0, 0, max_tokens - encoded_a.size(1)))
        encoded_b = F.pad(encoded_b, (0, 0, 0, max_tokens - encoded_b.size(1)))

        pair_features = self.trunk(
            torch.cat((encoded_a, encoded_b)),
            torch.cat((encoded_b, encoded_a)),
            torch.cat((pair.len_a, pair.len_b)),
            torch.cat((pair.len_b, pair.len_a)),
            topo_tokens=topo_tokens,
            topo_active=torch.cat((masks.topo, masks.topo)) if need_topo else None,
            edge_mask=(
                torch.cat((pair.edge_mask, pair.edge_mask))
                if pair.edge_mask is not None
                else None
            ),
        )
        feat_ab, feat_ba = pair_features.chunk(2)
        feat = torch.max(torch.stack([feat_ab, feat_ba], dim=-1), dim=-1).values
        # The registered mixed-precision contract keeps logits in fp32.
        # Casting after the head is too late: autocast has already quantized
        # the linear outputs, which can destroy the small full-minus-f_logit
        # residual (CLAUDE.md "fp32 island"; design §4).
        with torch.autocast(device_type=feat.device.type, enabled=False):
            logits: torch.Tensor = self.head(feat.float()).squeeze(-1)
        return logits
