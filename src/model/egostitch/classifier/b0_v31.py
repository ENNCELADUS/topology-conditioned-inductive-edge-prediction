"""B0-V3.1 pairwise classifier: the mature baseline plus a conditioning socket.

Merges four former standalone modules (three-component refactor design §5
consolidation pass):

- ``conditioning.py``: E2E head-null branch masks and `GatedCrossAttention`,
  the centered tanh-gated cross-attention residual sublayer.
- ``trunk.py``: `ConditionedPairCrossAttention`, a `PairCrossAttention`
  subclass adding gated topo conditioning after the final ``n_inj`` blocks
  (design rev 3 §3.4).
- The V3.1-specific half of `src/model/B0.py` (its config-parsing `__init__`,
  `V3_1`, `BEST_V3_1_CONFIG`, `BEST_V3_1_SELECTION`, `build_best_v3_1`) --
  the reusable building blocks it composes (`SiameseEncoder`, `MLPHead`,
  `PairCrossAttention`, ...) stay in `classifier/layers.py`, genuinely shared
  by any classifier version.
- `B0V31PairClassifier` itself (already here): composes `SiameseEncoder` +
  `MLPHead` (`classifier/layers.py`) with `ConditionedPairCrossAttention`.
  The AB/BA doubled-batch construction, the feature-wise max fuse, and the
  fp32 head island are lifted verbatim from the pre-refactor
  `EgoStitchE2E.score_pair_context` -- this is exactly where design §4's
  AB/BA invariant is realized: one trunk batch, one synchronized centering
  statistic, one EMA update per call.

`SiameseEncoder` is split across `encode_tokens` (the cacheable per-node
phase) and `forward` (the pair-level phase, correction 2026-08-03): `forward`
consumes `pair.tokens_a`/`tokens_b` directly and never calls `self.encoder`
itself, so a caller scoring many pairs over a shared node universe encodes
each node once via `encode_tokens` and reuses it, instead of paying the
`SiameseEncoder` cost again on every pair.

Design contract for the conditioning primitives: docs/superpowers/specs/
2026-07-16-egostitch-e2e-conditioned-encoder-design.md rev 3 — three
mutually exclusive head nulls; training uses per-sample multiplicative masks,
evaluation uses batch-level hard bypasses. Mask semantics: True = pathway
ACTIVE for that pair (shared across AB/BA). The rev-3.1 centered residual is
governed by spec §14.4.2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn import functional as dist_functional
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.utils import checkpoint as checkpoint_module

from src.model.egostitch.classifier.base import HeadNullMasks, PairClassifier
from src.model.egostitch.classifier.layers import (
    MIXING_MODES,
    ORDER_AGGREGATION_MODES,
    PAIR_READOUT_MODES,
    MLPHead,
    PairCrossAttention,
    SiameseEncoder,
    _build_padding_mask,
    _parse_rich_pooling_components,
)
from src.model.egostitch.graph import PairConditioning, PairInputs

# --------------------------------------------------------------------------- head-null masks

NULL_NONE = "none"
NULL_ALL_HEAD = "all_head"
_KNOWN_NULLS = (NULL_NONE, NULL_ALL_HEAD)


def sample_branch_masks(
    batch_size: int,
    p_topo: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> HeadNullMasks:
    """Sample independent per-pair branch-dropout masks (design §4)."""
    topo = torch.rand(batch_size, generator=generator) >= p_topo
    return HeadNullMasks(topo=topo.to(device))


def masks_for_null(null: str, batch_size: int, device: torch.device) -> HeadNullMasks:
    """Deterministic masks realizing one of the §3.5 null conditions."""
    if null not in _KNOWN_NULLS:
        raise ValueError(f"unknown head-null condition: {null!r}")
    on = torch.ones(batch_size, dtype=torch.bool, device=device)
    off = torch.zeros(batch_size, dtype=torch.bool, device=device)
    topo = off if null == NULL_ALL_HEAD else on
    return HeadNullMasks(topo=topo)


class GatedCrossAttention(nn.Module):
    """Centered tanh-gated cross-attention residual sublayer (spec §14.4.2).

    ``cls <- cls + active * tanh(gate) * (XAttn(...) - mu)``. Training ``mu`` is
    the synchronized mean over active, real rows from the joint AB/BA batch;
    evaluation uses its frozen checkpointed EMA. With one shared ``mu``, an
    individual direction may inject a nonzero residual when the AB and BA means
    differ. Training and evaluation nevertheless agree, and a pathway constant
    across both directions contributes exactly nothing. Inactive rows retain the
    checkpoint-exact identity bypass.
    """

    ema_mu: torch.Tensor
    ema_updates: torch.Tensor

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
        self.ema_decay = float(ema_decay)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))
        self.register_buffer("ema_mu", torch.zeros(1, 1, d_model))
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.int64))
        self.register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
            self._fill_legacy_ema_state
        )

    def _fill_legacy_ema_state(
        self,
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Fill absent EMA buffers for checkpoints using the current scaffold shape."""
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        assert module is self
        state_dict.setdefault(f"{prefix}ema_mu", self.ema_mu.detach().clone())
        state_dict.setdefault(f"{prefix}ema_updates", self.ema_updates.detach().clone())

    @staticmethod
    def _global_center(
        attn_out: torch.Tensor, include: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Center included rows around a synchronized, differentiable mean."""
        weight = include.to(dtype=attn_out.dtype).view(-1, 1, 1)
        count = weight.sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        has_rows = bool(count.item() > 0)
        if not has_rows:
            return torch.zeros_like(attn_out), torch.zeros_like(attn_out[:1]), False

        # Subtracting a synchronized reference before summation makes a
        # constant attention output produce an exact zero residual even when
        # its fp32 value is not exactly representable.
        masked = torch.where(
            include.view(-1, 1, 1),
            attn_out.detach(),
            torch.full_like(attn_out, torch.inf),
        )
        reference = masked.amin(dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reference, op=dist.ReduceOp.MIN)
        deviations = attn_out - reference
        total_deviation = (deviations * weight).sum(dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            total_deviation = dist_functional.all_reduce(  # type: ignore[no-untyped-call]
                total_deviation, op=dist.ReduceOp.SUM
            )
        mean_deviation = total_deviation / count
        return deviations - mean_deviation, reference + mean_deviation, True

    def _update_ema(self, mu: torch.Tensor) -> None:
        """Update the post-all-reduce EMA identically on every rank."""
        with torch.no_grad():
            if int(self.ema_updates.item()) == 0:
                self.ema_mu.copy_(mu.detach())
            else:
                self.ema_mu.mul_(self.ema_decay).add_(
                    mu.detach(), alpha=1.0 - self.ema_decay
                )
            self.ema_updates.add_(1)

    def forward(
        self,
        cls: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        active: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the gated cross-attention residual update to ``cls``.

        Args:
            cls: Shape ``(B, 1, d_model)`` query token.
            tokens: Shape ``(B, T, d_model)`` key/value tokens.
            token_mask: Optional shape ``(B, T)`` bool mask, True = valid
                token; invalid tokens are excluded from attention.
            active: Shape ``(B,)`` bool mask, True = pathway active for that
                sample; inactive samples get an exact identity bypass.
            edge_mask: Optional shape ``(B,)`` real-row mask. False/zero padded
                filler rows are excluded from the centering mean.

        Returns:
            Shape ``(B, 1, d_model)`` updated ``cls``.
        """
        key_padding_mask = None if token_mask is None else ~token_mask
        attn_out: torch.Tensor
        attn_out, _ = self.attn(
            self.norm_q(cls),
            self.norm_kv(tokens),
            self.norm_kv(tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        with torch.autocast(device_type=cls.device.type, enabled=False):
            cls32 = cls.float()
            attn32 = attn_out.float()
            if self.training:
                real = (
                    torch.ones_like(active, dtype=torch.bool)
                    if edge_mask is None
                    else edge_mask.to(dtype=torch.bool)
                )
                centered, mu, has_rows = self._global_center(attn32, active & real)
                if has_rows:
                    self._update_ema(mu)
                else:
                    centered = attn32 - self.ema_mu
            else:
                centered = attn32 - self.ema_mu
            residual = torch.tanh(self.gate.float()) * centered
            updated = cls32 + residual
            return torch.where(active.view(-1, 1, 1), updated, cls32)


# --------------------------------------------------------------------------- conditioned trunk


class ConditionedPairCrossAttention(PairCrossAttention):
    """PairCrossAttention + zero-init gated cls conditioning (§3.4 pins)."""

    def __init__(
        self,
        *,
        n_inj: int = 1,
        xattn_heads: int = 8,
        xattn_dropout: float = 0.0,
        conditioning_ema_decay: float = 0.99,
        d_model: int,
        **kwargs: object,
    ) -> None:
        super().__init__(d_model=d_model, **kwargs)  # type: ignore[arg-type]
        if not 1 <= n_inj <= len(self.layers):
            raise ValueError("n_inj must be in [1, n_layers]")
        self.n_inj = n_inj
        self.topo_xattn = nn.ModuleList(
            GatedCrossAttention(
                d_model,
                xattn_heads,
                xattn_dropout,
                ema_decay=conditioning_ema_decay,
            )
            for _ in range(n_inj)
        )

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        *,
        topo_tokens: torch.Tensor | None = None,
        topo_active: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the pair trunk with optional gated topo cls conditioning.

        Args:
            h_a: Item A hidden states ``(batch, seq_len_a, d_model)``.
            h_b: Item B hidden states ``(batch, seq_len_b, d_model)``.
            lengths_a: Sequence lengths for A ``(batch,)``.
            lengths_b: Sequence lengths for B ``(batch,)``.
            topo_tokens: Optional scaffold tokens ``(batch, T, d_model)`` for
                the topo cross-attention pathway; ``None`` is a hard bypass.
            topo_active: Optional ``(batch,)`` bool mask gating the topo
                pathway per-sample; required alongside ``topo_tokens``.
            edge_mask: Optional ``(batch,)`` real-row mask used to exclude DDP
                filler rows from conditioning means.

        Returns:
            Fused pair representation ``(batch, d_model)``.
        """
        if h_a.dim() != 3 or h_b.dim() != 3:
            raise ValueError(
                "Cross-attention inputs must have shape (batch_size, seq_len, d_model)"
            )
        if h_a.size(0) != h_b.size(0):
            raise ValueError("Item pair batches must have matching batch dimension")

        batch_size = h_a.size(0)
        mask_a = _build_padding_mask(lengths_a, h_a.size(1))
        mask_b = _build_padding_mask(lengths_b, h_b.size(1))
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        n_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            h_a, h_b, cls_token = layer(h_a, h_b, cls_token, mask_a, mask_b)
            inj = idx - (n_layers - self.n_inj)
            if inj >= 0 and topo_tokens is not None and topo_active is not None:
                cls_token = self.topo_xattn[inj](
                    cls_token, topo_tokens, None, topo_active, edge_mask
                )
        cls_vec = cls_token.squeeze(1)
        if self.pair_readout_mode == "pair_context_gated":
            # Full and hard-null heads differ only through the conditioned cls
            # stream. Keep their final pair readout in fp32 so BF16 does not
            # quantize away that small, scientifically load-bearing residual.
            def fp32_readout(
                readout_a: torch.Tensor,
                readout_b: torch.Tensor,
                readout_cls: torch.Tensor,
            ) -> torch.Tensor:
                with torch.autocast(device_type=readout_cls.device.type, enabled=False):
                    return cast(
                        torch.Tensor,
                        self.pair_context_readout(
                            readout_a.float(),
                            readout_b.float(),
                            readout_cls.float(),
                            mask_a,
                            mask_b,
                        ),
                    )

            if self.training and torch.is_grad_enabled():
                return cast(
                    torch.Tensor,
                    checkpoint_module.checkpoint(
                        fp32_readout, h_a, h_b, cls_vec, use_reentrant=False
                    ),
                )
            return fp32_readout(h_a, h_b, cls_vec)
        base_repr = self._rich_pooling_readout(h_a, h_b, cls_vec, mask_a, mask_b)
        if self.pair_readout_mode == "grid_sketch_fusion":
            return cast(
                torch.Tensor,
                self.grid_sketch_readout(base_repr, h_a, h_b, mask_a, mask_b),
            )
        return base_repr


# --------------------------------------------------------------------------- B0-V3.1 classifier


class B0V31PairClassifier(PairClassifier):
    """The mature B0-V3.1 pairwise baseline with a topo-conditioning socket.

    `cond=None` takes the exact-bypass path: the topo cross-attention module
    is never invoked at all (not merely gated to zero). That is numerically
    identical to the `f_logit` / `NULL_ALL_HEAD` decomposition arm, because
    that arm's `active=False` gate already discards the whole pathway
    contribution unconditionally -- see the final `torch.where` in
    `GatedCrossAttention.forward` (above, `:200`).
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
        """Build the encoder, conditioned trunk, and head.

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


# --------------------------------------------------------------------------- V3.1 config helpers


def _to_int(value: object, field_name: str) -> int:
    """Convert config value to integer."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be int-compatible") from error
    raise ValueError(f"{field_name} must be int-compatible")


def _to_float(value: object, field_name: str) -> float:
    """Convert config value to float."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be float-compatible") from error
    raise ValueError(f"{field_name} must be float-compatible")


def _to_mapping(value: object, field_name: str) -> Mapping[str, object]:
    """Validate config value as key-value mapping."""
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{field_name} must be a mapping")


def _to_bool(value: object, field_name: str) -> bool:
    """Convert a config value to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be bool-compatible")


# --------------------------------------------------------------------------- V3_1 top-level model


class V3_1(nn.Module):
    """V3.1 model — V3 with rich per-item pooling (CLS+mean+attn+max+gate).

    Config keys are identical to V3; no new required fields.
    """

    name: str = "v3.1"

    def __init__(self, **model_config: object) -> None:
        super().__init__()
        required_fields = [
            "input_dim",
            "d_model",
            "encoder_layers",
            "cross_attn_layers",
            "n_heads",
        ]
        missing = [f for f in required_fields if f not in model_config]
        if missing:
            raise ValueError(f"Missing required model configuration fields: {missing}")

        self.input_dim = _to_int(model_config["input_dim"], "model_config.input_dim")
        self.d_model = _to_int(model_config["d_model"], "model_config.d_model")
        self.encoder_layers = _to_int(model_config["encoder_layers"], "model_config.encoder_layers")
        self.cross_attn_layers = _to_int(
            model_config["cross_attn_layers"], "model_config.cross_attn_layers"
        )
        self.n_heads = _to_int(model_config["n_heads"], "model_config.n_heads")

        # Loss-side only: symmetric BCE label smoothing, applied in `forward` when
        # `label` is present. Architecture is unaffected, so a scoring-time rebuild
        # from a checkpoint's embedded config carries it harmlessly.
        self.label_smoothing = _to_float(
            model_config.get("label_smoothing", 0.0), "model_config.label_smoothing"
        )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(
                "model_config.label_smoothing must be in [0.0, 1.0), got "
                f"{self.label_smoothing}"
            )

        mlp_cfg_raw = model_config.get("mlp_head")
        if not isinstance(mlp_cfg_raw, dict) or not mlp_cfg_raw:
            raise ValueError("mlp_head configuration is required for V3_1")
        mlp_cfg = _to_mapping(mlp_cfg_raw, "model_config.mlp_head")
        if "hidden_dims" not in mlp_cfg or "dropout" not in mlp_cfg:
            raise ValueError("mlp_head.hidden_dims and mlp_head.dropout must be provided")
        hidden_dims_raw = mlp_cfg["hidden_dims"]
        if not isinstance(hidden_dims_raw, list) or not hidden_dims_raw:
            raise ValueError("mlp_head.hidden_dims must be a non-empty list")
        self.mlp_hidden_dims = [
            _to_int(v, "model_config.mlp_head.hidden_dims") for v in hidden_dims_raw
        ]
        self.mlp_dropout = _to_float(mlp_cfg["dropout"], "model_config.mlp_head.dropout")
        self.mlp_activation = str(mlp_cfg.get("activation", "gelu"))
        self.mlp_norm = str(mlp_cfg.get("norm", "layernorm"))
        self.mlp_spectral_norm = _to_bool(
            mlp_cfg.get("spectral_norm", False),
            "model_config.mlp_head.spectral_norm",
        )

        reg_cfg_raw = model_config.get("regularization")
        if not isinstance(reg_cfg_raw, dict) or "dropout" not in reg_cfg_raw:
            raise ValueError("regularization.dropout must be provided for V3_1")
        reg_cfg = _to_mapping(reg_cfg_raw, "model_config.regularization")
        self.encoder_dropout = _to_float(reg_cfg["dropout"], "model_config.regularization.dropout")
        self.cross_attention_dropout = _to_float(
            reg_cfg.get("cross_attention_dropout", self.encoder_dropout),
            "model_config.regularization.cross_attention_dropout",
        )
        self.token_dropout = _to_float(
            reg_cfg.get("token_dropout", 0.0), "model_config.regularization.token_dropout"
        )
        self.stochastic_depth = _to_float(
            reg_cfg.get("stochastic_depth", 0.0), "model_config.regularization.stochastic_depth"
        )
        mixing_cfg_raw = model_config.get("mixing", {})
        if mixing_cfg_raw is None:
            mixing_cfg_raw = {}
        if not isinstance(mixing_cfg_raw, dict):
            raise ValueError("model_config.mixing must be a mapping")
        mixing_cfg = _to_mapping(mixing_cfg_raw, "model_config.mixing")
        self.mixing_mode = str(mixing_cfg.get("mode", "bidirectional_cross")).lower()
        if self.mixing_mode not in MIXING_MODES:
            raise ValueError(f"model_config.mixing.mode must be one of: {', '.join(MIXING_MODES)}")
        rich_pooling_cfg_raw = model_config.get("rich_pooling", {})
        if rich_pooling_cfg_raw is None:
            rich_pooling_cfg_raw = {}
        if not isinstance(rich_pooling_cfg_raw, dict):
            raise ValueError("model_config.rich_pooling must be a mapping")
        rich_pooling_cfg = _to_mapping(rich_pooling_cfg_raw, "model_config.rich_pooling")
        self.rich_pooling_components = _parse_rich_pooling_components(
            rich_pooling_cfg.get("components")
        )
        pair_readout_cfg_raw = model_config.get("pair_readout", {})
        if pair_readout_cfg_raw is None:
            pair_readout_cfg_raw = {}
        if not isinstance(pair_readout_cfg_raw, dict):
            raise ValueError("model_config.pair_readout must be a mapping")
        pair_readout_cfg = _to_mapping(pair_readout_cfg_raw, "model_config.pair_readout")
        self.pair_readout_mode = str(pair_readout_cfg.get("mode", "rich_pooling")).lower()
        if self.pair_readout_mode not in PAIR_READOUT_MODES:
            raise ValueError(
                f"model_config.pair_readout.mode must be one of: {', '.join(PAIR_READOUT_MODES)}"
            )
        self.order_aggregation = str(pair_readout_cfg.get("order_aggregation", "single")).lower()
        if self.order_aggregation not in ORDER_AGGREGATION_MODES:
            raise ValueError(
                "model_config.pair_readout.order_aggregation must be one of: "
                f"{', '.join(ORDER_AGGREGATION_MODES)}"
            )
        self.pair_readout_spectral_norm = _to_bool(
            pair_readout_cfg.get("spectral_norm", False),
            "model_config.pair_readout.spectral_norm",
        )
        sketch_tokens = _to_int(
            pair_readout_cfg.get("sketch_tokens", 64),
            "model_config.pair_readout.sketch_tokens",
        )
        pair_dim = _to_int(
            pair_readout_cfg.get("pair_dim", 32),
            "model_config.pair_readout.pair_dim",
        )
        cnn_dim = _to_int(
            pair_readout_cfg.get("cnn_dim", 64),
            "model_config.pair_readout.cnn_dim",
        )
        cnn_blocks = _to_int(
            pair_readout_cfg.get("cnn_blocks", 2),
            "model_config.pair_readout.cnn_blocks",
        )
        cnn_dropout = _to_float(
            pair_readout_cfg.get("cnn_dropout", 0.1),
            "model_config.pair_readout.cnn_dropout",
        )

        self.encoder = SiameseEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_layers=self.encoder_layers,
            n_heads=self.n_heads,
            dropout=self.encoder_dropout,
            token_dropout=self.token_dropout,
            stochastic_depth=self.stochastic_depth,
        )
        self.cross_attention = PairCrossAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.cross_attn_layers,
            dropout=self.cross_attention_dropout,
            pooling_components=self.rich_pooling_components,
            pair_readout_mode=self.pair_readout_mode,
            sketch_tokens=sketch_tokens,
            pair_dim=pair_dim,
            cnn_dim=cnn_dim,
            cnn_blocks=cnn_blocks,
            cnn_dropout=cnn_dropout,
            mixing_mode=self.mixing_mode,
            pair_readout_spectral_norm=self.pair_readout_spectral_norm,
        )
        self.output_head = MLPHead(
            input_dim=self.d_model,
            hidden_dims=self.mlp_hidden_dims,
            output_dim=1,
            dropout=self.mlp_dropout,
            activation=self.mlp_activation,
            norm=self.mlp_norm,
        )
        if self.mlp_spectral_norm:
            self._apply_output_head_spectral_norm()

    def _apply_output_head_spectral_norm(self) -> None:
        """Apply spectral norm to output-head linear layers."""
        for module in self.output_head.modules():
            if isinstance(module, nn.Linear):
                spectral_norm(module)

    def _pair_representation(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        """Return a pair representation with optional AB/BA max aggregation."""
        feature_ab = self.cross_attention(encoded_a, encoded_b, lengths_a, lengths_b)
        if self.order_aggregation == "single":
            return cast(torch.Tensor, feature_ab)

        feature_ba = self.cross_attention(encoded_b, encoded_a, lengths_b, lengths_a)
        return torch.max(torch.stack([feature_ab, feature_ba], dim=-1), dim=-1).values

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run a model forward pass.

        Args:
            batch: Optional batch dictionary.
            **kwargs: Additional batch tensors merged into ``batch``.

        Returns:
            Output dictionary containing ``logits`` and optional ``loss``.
        """
        merged: dict[str, torch.Tensor] = {}
        if batch is not None:
            merged.update(batch)
        merged.update(kwargs)

        if "emb_a" not in merged or "emb_b" not in merged:
            raise KeyError("Batch must contain 'emb_a' and 'emb_b' tensors")

        emb_a = merged["emb_a"]
        emb_b = merged["emb_b"]
        if emb_a.dim() != 3 or emb_b.dim() != 3:
            raise ValueError("Input embeddings must be shaped (batch, seq_len, embedding_dim)")
        if emb_a.size(2) != self.input_dim or emb_b.size(2) != self.input_dim:
            raise ValueError("Input embedding dimension must match model input_dim")
        if emb_a.size(0) != emb_b.size(0):
            raise ValueError("Item pair batches must have matching batch dimension")

        device = emb_a.device
        lengths_a = merged.get("len_a")
        lengths_b = merged.get("len_b")
        if lengths_a is None:
            lengths_a = torch.full((emb_a.size(0),), emb_a.size(1), device=device, dtype=torch.long)
        else:
            lengths_a = lengths_a.to(device=device, dtype=torch.long)
        if lengths_b is None:
            lengths_b = torch.full((emb_b.size(0),), emb_b.size(1), device=device, dtype=torch.long)
        else:
            lengths_b = lengths_b.to(device=device, dtype=torch.long)

        encoded_a = self.encoder(emb_a, lengths_a)
        encoded_b = self.encoder(emb_b, lengths_b)
        pair_repr = self._pair_representation(encoded_a, encoded_b, lengths_a, lengths_b)
        logits = self.output_head(pair_repr)

        output: dict[str, torch.Tensor] = {"logits": logits}
        if "label" in merged:
            labels = merged["label"].float()
            logits_for_loss = (
                logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            )
            labels_for_loss = (
                labels.squeeze(-1) if labels.dim() > 1 and labels.size(-1) == 1 else labels
            )
            if self.label_smoothing > 0.0:
                # Symmetric binary smoothing: 1 -> 1 - eps/2, 0 -> eps/2.
                labels_for_loss = (
                    labels_for_loss * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
                )
            output["loss"] = nn.functional.binary_cross_entropy_with_logits(
                logits_for_loss, labels_for_loss
            )

        return output


BEST_V3_1_CONFIG: dict[str, object] = {
    "input_dim": 1536,
    "d_model": 256,
    "encoder_layers": 3,
    "cross_attn_layers": 3,
    "n_heads": 8,
    "mlp_head": {
        "hidden_dims": [256, 128, 64],
        "dropout": 0.2,
        "activation": "gelu",
        "norm": "layernorm",
        "spectral_norm": True,
    },
    "regularization": {
        "dropout": 0.1,
        "token_dropout": 0.1,
        "cross_attention_dropout": 0.1,
        "stochastic_depth": 0.1,
    },
    "rich_pooling": {
        "components": ["mean", "attn", "max", "gated"],
    },
    "pair_readout": {
        "mode": "pair_context_gated",
        "order_aggregation": "abba_max",
        "spectral_norm": True,
    },
    "mixing": {
        "mode": "block_self",
    },
}

BEST_V3_1_SELECTION: dict[str, object] = {
    "run_id": "pair_context_gated_abba_block_sn_d256_s13",
    "seed": 13,
    "selection_metric": "test_auprc",
    "selection_value": 0.7227626490230867,
}


def build_best_v3_1() -> V3_1:
    """Build the selected V3.1 model variant."""
    return V3_1(**BEST_V3_1_CONFIG)


__all__ = [
    "B0V31PairClassifier",
    "BEST_V3_1_CONFIG",
    "BEST_V3_1_SELECTION",
    "ConditionedPairCrossAttention",
    "GatedCrossAttention",
    "NULL_ALL_HEAD",
    "NULL_NONE",
    "V3_1",
    "build_best_v3_1",
    "masks_for_null",
    "sample_branch_masks",
]
