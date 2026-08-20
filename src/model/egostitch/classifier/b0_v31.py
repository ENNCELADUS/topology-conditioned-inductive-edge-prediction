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
from src.model.egostitch.classifier.node_factor import NodeFactorBottleneck, NodeFactorOutputs
from src.model.egostitch.config import CONDITIONING_MODES
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


# ------------------------------------------------------- shared centering machinery
#
# `GatedCrossAttention` owned this logic privately until the conditioning
# ladder (`ClassifierConfig.conditioning_mode`) gave a second module -- the
# `film_logit` conditioner -- the same job: subtract a synchronized,
# differentiable mean over active, real rows during training, and the frozen
# checkpointed EMA of that mean at evaluation. The two must agree exactly on
# the all-reduce pattern or a DDP run and a single-GPU run will disagree, so
# the logic lives here, once, as free functions.
#
# Deliberately *not* refactored into a shared `nn.Module`: the EMA buffers stay
# registered on `GatedCrossAttention` itself under their existing names
# (`ema_mu` / `ema_updates`), because every rev-3.1 checkpoint carries
# `trunk.topo_xattn.<i>.ema_mu` and nesting them one level deeper would rename
# the keys and break `xattn_cls` checkpoint loading for a purely cosmetic gain.


def _include_weight(include: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Reshape an inclusion mask to broadcast against `value`'s trailing dims.

    Args:
        include: Shape ``(B,)`` (per-row: the ``xattn_cls`` case) or ``(B, T)``
            (per-token: the ``xattn_tokens`` case).
        value: Shape ``(B, T_v, d)``.

    Returns:
        `include` viewed with `value`'s rank, trailing dims of size 1.
    """
    return include.view(*include.shape, *((1,) * (value.dim() - include.dim())))


def _global_center(
    value: torch.Tensor, include: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Center included entries around a synchronized, differentiable mean.

    Args:
        value: Shape ``(B, T, d)`` fp32 quantity to center.
        include: Shape ``(B,)`` or ``(B, T)`` bool. A ``(B,)`` mask counts
            *rows*; a ``(B, T)`` mask counts *tokens*, which is what
            ``xattn_tokens`` needs -- the returned mean must be a single
            ``(1, 1, d)`` statistic over every valid token on every rank, not
            a per-position one, or the frozen EMA could not stand in for it at
            evaluation time.

    Returns:
        ``(centered, mu, has_rows)``. ``mu`` is always ``(1, 1, d)``.
    """
    # A `(0,)` tuple dispatches to the same ATen reduction as a bare `dim=0`,
    # so the `(B,)` path is arithmetically identical to the pre-ladder code.
    reduce_dims = (0,) if include.dim() == 1 else (0, 1)
    weight = _include_weight(include, value).to(dtype=value.dtype)
    count = weight.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    has_rows = bool(count.item() > 0)
    if not has_rows:
        return torch.zeros_like(value), torch.zeros_like(value[:1, :1]), False

    # Subtracting a synchronized reference before summation makes a
    # constant attention output produce an exact zero residual even when
    # its fp32 value is not exactly representable.
    masked = torch.where(
        _include_weight(include, value),
        value.detach(),
        torch.full_like(value, torch.inf),
    )
    reference = masked.amin(dim=reduce_dims, keepdim=True)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(reference, op=dist.ReduceOp.MIN)
    deviations = value - reference
    total_deviation = (deviations * weight).sum(dim=reduce_dims, keepdim=True)
    if dist.is_available() and dist.is_initialized():
        total_deviation = dist_functional.all_reduce(  # type: ignore[no-untyped-call]
            total_deviation, op=dist.ReduceOp.SUM
        )
    mean_deviation = total_deviation / count
    return deviations - mean_deviation, reference + mean_deviation, True


def _advance_ema(
    ema_mu: torch.Tensor, ema_updates: torch.Tensor, mu: torch.Tensor, decay: float
) -> None:
    """Advance a post-all-reduce EMA in place, identically on every rank."""
    with torch.no_grad():
        if int(ema_updates.item()) == 0:
            ema_mu.copy_(mu.detach())
        else:
            ema_mu.mul_(decay).add_(mu.detach(), alpha=1.0 - decay)
        ema_updates.add_(1)


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
        """Center included rows around a synchronized, differentiable mean.

        Kept as a staticmethod delegating to the module-level `_global_center`
        so tests that reach for `GatedCrossAttention._global_center` still
        find it; the implementation is shared with the `film_logit` rung of
        the conditioning ladder.
        """
        return _global_center(attn_out, include)

    def _update_ema(self, mu: torch.Tensor) -> None:
        """Update the post-all-reduce EMA identically on every rank."""
        _advance_ema(self.ema_mu, self.ema_updates, mu, self.ema_decay)

    def forward(
        self,
        cls: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        active: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the gated cross-attention residual update to the queries.

        Generalized from a single ``cls`` query to an arbitrary query set for
        the ``xattn_tokens`` rung of the conditioning ladder. With
        ``query_mask=None`` -- which is what ``xattn_cls`` passes, and what
        every existing caller passes -- the arithmetic is unchanged: the
        inclusion mask stays ``(B,)``, so the centering statistic is still a
        mean over *rows*, and the reduction dispatches to the same ATen op.

        Args:
            cls: Shape ``(B, T_q, d_model)`` query tokens. ``T_q == 1`` (the
                trunk's cls token) for ``xattn_cls``; every valid non-cls pair
                token for ``xattn_tokens``. Named ``cls`` still, positionally,
                because every existing call site passes it positionally.
            tokens: Shape ``(B, T, d_model)`` key/value tokens.
            token_mask: Optional shape ``(B, T)`` bool mask, True = valid
                token; invalid tokens are excluded from attention.
            active: Shape ``(B,)`` bool mask, True = pathway active for that
                sample; inactive samples get an exact identity bypass.
            edge_mask: Optional shape ``(B,)`` real-row mask. False/zero padded
                filler rows are excluded from the centering mean.
            query_mask: Optional shape ``(B, T_q)`` bool mask, True = the query
                is a real (non-padding) token. When given, the centering mean
                is taken over valid *tokens* rather than valid rows -- the
                count all-reduce sums token counts -- while the EMA it feeds
                stays a single ``(1, 1, d_model)`` statistic, so training and
                evaluation still agree. Padding queries are still written
                through unchanged by the final `torch.where`, since they are
                excluded from `include` but not from `active`; their updated
                values are discarded downstream by the trunk's own padding
                masks.

        Returns:
            Shape ``(B, T_q, d_model)`` updated queries.
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
                include = active & real
                if query_mask is not None:
                    include = include.unsqueeze(1) & query_mask
                centered, mu, has_rows = self._global_center(attn32, include)
                if has_rows:
                    self._update_ema(mu)
                else:
                    centered = attn32 - self.ema_mu
            else:
                centered = attn32 - self.ema_mu
            residual = torch.tanh(self.gate.float()) * centered
            updated = cls32 + residual
            return torch.where(active.view(-1, 1, 1), updated, cls32)


# ------------------------------------------------------------ conditioning ladder rungs


class PooledAdapter(nn.Module):
    """Zero-init low-rank adapter: a pooled graph summary added to every token.

    The ``pooled_adapter`` rung. One bottleneck MLP per injection site,
    ``d_model -> 32 -> d_model`` with the up-projection zero-initialized, so at
    initialization the delta is exactly zero and the trunk is bit-identical to
    the unconditioned path.

    Weaker than cross-attention by construction: the conditioning enters as a
    single vector broadcast over every token, so it can shift the trunk's
    representation but cannot route different graph tokens to different pair
    tokens. That is the point -- it is the rung between `film_logit`'s scalar
    and `xattn_*`'s full attention.
    """

    def __init__(self, d_model: int, bottleneck: int = 32) -> None:
        """Build the bottleneck, with the up-projection zeroed.

        Args:
            d_model: Trunk width.
            bottleneck: Adapter rank.
        """
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Map a pooled graph summary to a per-row token delta.

        Args:
            pooled: Shape ``(B, d_model)``.

        Returns:
            Shape ``(B, 1, d_model)``, ready to broadcast over the token axis.
        """
        delta: torch.Tensor = self.up(F.gelu(self.down(pooled)))
        return delta.unsqueeze(1)


class FilmLogitConditioner(nn.Module):
    """Zero-init FiLM on the fused pair logit -- the weakest ladder rung.

    Maps the pooled graph summary to ``(s, t)``; the classifier applies
    ``z' = z * (1 + tanh(s)) + t`` to its single fused logit. There is no path
    by which topology can change the *representation*: it can only rescale and
    shift the decision value. If a run improves under `xattn_cls` but not under
    this, the gain came from representation shaping rather than from a
    calibration signal -- precisely the confound the ladder exists to separate.

    ``(s, t)`` are centered with exactly the same synchronized-mean-plus-EMA
    machinery `GatedCrossAttention` uses (`_global_center` / `_advance_ema`,
    shared, not forked): without it, a conditioning signal that happened to be
    constant across the batch would act as a free global bias on every logit
    and would be scored as "topology helped".

    Its EMA buffers are its own (`ema_mu` shape ``(1, 1, 2)``), so they can
    never collide with `GatedCrossAttention`'s ``(1, 1, d_model)`` ones in a
    checkpoint -- the two rungs are mutually exclusive anyway.
    """

    ema_mu: torch.Tensor
    ema_updates: torch.Tensor

    def __init__(self, d_model: int, hidden: int = 64, *, ema_decay: float = 0.99) -> None:
        """Build the zero-initialized ``(s, t)`` head.

        Args:
            d_model: Width of the pooled graph summary.
            hidden: Bottleneck width.
            ema_decay: Decay of the centering EMA.

        Raises:
            ValueError: If `ema_decay` is not in ``(0, 1)``.
        """
        super().__init__()
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
        self.ema_decay = float(ema_decay)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.register_buffer("ema_mu", torch.zeros(1, 1, 2))
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.int64))

    def forward(
        self,
        pooled: torch.Tensor,
        active: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the centered ``(s, t)`` pair for every row.

        Args:
            pooled: Shape ``(B, d_model)`` pooled graph summary, on the same
                doubled AB/BA batch the trunk runs.
            active: Shape ``(B,)`` bool pathway-activity mask.
            edge_mask: Optional shape ``(B,)`` real-row mask.

        Returns:
            Shape ``(B, 2)`` centered ``(s, t)``. Exactly zero at
            initialization, for every row, because `fc2` is zero-initialized
            and centering a constant-zero signal yields zero.
        """
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            raw = self.fc2(F.gelu(self.fc1(self.norm(pooled.float())))).unsqueeze(1)
            if self.training:
                real = (
                    torch.ones_like(active, dtype=torch.bool)
                    if edge_mask is None
                    else edge_mask.to(dtype=torch.bool)
                )
                centered, mu, has_rows = _global_center(raw, active & real)
                if has_rows:
                    _advance_ema(self.ema_mu, self.ema_updates, mu, self.ema_decay)
                else:
                    centered = raw - self.ema_mu
            else:
                centered = raw - self.ema_mu
            return centered.squeeze(1)


# --------------------------------------------------------------------------- conditioned trunk


class ConditionedPairCrossAttention(PairCrossAttention):
    """PairCrossAttention + one zero-init conditioning rung (§3.4 pins).

    Exactly one rung's modules are constructed, selected by
    `conditioning_mode`. Constructing all of them and gating at runtime would
    leave the unselected ones as parameters that never receive gradient, which
    DDP rejects (or, with `find_unused_parameters=True`, pays for on every
    step) -- and would perturb the RNG stream so that two ladder arms differed
    by more than the mechanism under test.

    ``film_logit`` builds nothing here: it acts on the fused logit, which is
    `B0V31PairClassifier`'s to apply, not the trunk's.
    """

    def __init__(
        self,
        *,
        n_inj: int = 1,
        xattn_heads: int = 8,
        xattn_dropout: float = 0.0,
        conditioning_ema_decay: float = 0.99,
        conditioning_mode: str = "xattn_cls",
        d_model: int,
        **kwargs: object,
    ) -> None:
        super().__init__(d_model=d_model, **kwargs)  # type: ignore[arg-type]
        if not 1 <= n_inj <= len(self.layers):
            raise ValueError("n_inj must be in [1, n_layers]")
        if conditioning_mode not in CONDITIONING_MODES:
            raise ValueError(
                f"conditioning_mode must be one of {sorted(CONDITIONING_MODES)}, "
                f"got {conditioning_mode!r}"
            )
        self.n_inj = n_inj
        self.conditioning_mode = conditioning_mode
        # `topo_xattn` keeps its name and its `nn.ModuleList` position, so
        # rev-3.1 `xattn_cls` checkpoints keep loading unchanged. For the two
        # non-cross-attention rungs it is an *empty* list rather than absent:
        # it holds no modules and therefore contributes no parameters and no
        # `state_dict` keys (the DDP requirement is satisfied either way), but
        # `train_egostitch._e2e_gate_tanh` reads `trunk.topo_xattn`
        # unconditionally for its `gate_topo_tanh` telemetry, and an absent
        # attribute would crash those arms mid-run over a diagnostic. An empty
        # readout is the honest answer there: those rungs have no gate.
        self.topo_xattn = nn.ModuleList(
            GatedCrossAttention(
                d_model,
                xattn_heads,
                xattn_dropout,
                ema_decay=conditioning_ema_decay,
            )
            for _ in range(n_inj if conditioning_mode in ("xattn_cls", "xattn_tokens") else 0)
        )
        if conditioning_mode == "pooled_adapter":
            self.pooled_adapter = nn.ModuleList(PooledAdapter(d_model) for _ in range(n_inj))

    def _inject_token_xattn(
        self,
        inj: int,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        pad_a: torch.Tensor | None,
        pad_b: torch.Tensor | None,
        topo_tokens: torch.Tensor,
        topo_active: torch.Tensor,
        edge_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-attend from *every* valid pair token onto the graph tokens.

        The ``xattn_tokens`` rung. Both item streams are queried in a single
        call, concatenated along the token axis, so the whole injection site
        still produces exactly **one** synchronized centering statistic and
        advances its EMA exactly **once** -- the design §4 invariant that two
        separate calls would break.

        Padding queries are excluded from that statistic (via `query_mask`) but
        still pass through the module; their outputs are discarded by the
        trunk's own padding masks in the following layer, so including them
        would only have polluted the mean.

        Args:
            inj: Injection-site index into `topo_xattn`.
            h_a: Item A hidden states.
            h_b: Item B hidden states.
            pad_a: Item A padding mask (True = PAD, `_build_padding_mask`'s
                convention), or `None` when no lengths were supplied.
            pad_b: Item B padding mask.
            topo_tokens: Graph tokens to attend over.
            topo_active: Per-row pathway-activity mask.
            edge_mask: Optional real-row mask.

        Returns:
            The updated ``(h_a, h_b)``.
        """
        split = (h_a.size(1), h_b.size(1))
        queries = torch.cat((h_a, h_b), dim=1)
        if pad_a is None or pad_b is None:
            query_mask = torch.ones(queries.shape[:2], dtype=torch.bool, device=queries.device)
        else:
            query_mask = ~torch.cat((pad_a, pad_b), dim=1)
        updated = self.topo_xattn[inj](
            queries, topo_tokens, None, topo_active, edge_mask, query_mask
        )
        new_a, new_b = updated.split(split, dim=1)
        return new_a, new_b

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        *,
        topo_tokens: torch.Tensor | None = None,
        topo_pooled: torch.Tensor | None = None,
        topo_active: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the pair trunk with the selected zero-init conditioning rung.

        Args:
            h_a: Item A hidden states ``(batch, seq_len_a, d_model)``.
            h_b: Item B hidden states ``(batch, seq_len_b, d_model)``.
            lengths_a: Sequence lengths for A ``(batch,)``.
            lengths_b: Sequence lengths for B ``(batch,)``.
            topo_tokens: Optional scaffold tokens ``(batch, T, d_model)`` for
                the ``xattn_*`` pathways; ``None`` is a hard bypass.
            topo_pooled: Optional pooled graph summary ``(batch, d_model)`` for
                the ``pooled_adapter`` pathway; ``None`` is a hard bypass.
            topo_active: Optional ``(batch,)`` bool mask gating the pathway
                per-sample; required alongside either conditioning input.
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
            if inj < 0 or topo_active is None:
                continue
            mode = self.conditioning_mode
            if mode == "xattn_cls" and topo_tokens is not None:
                cls_token = self.topo_xattn[inj](
                    cls_token, topo_tokens, None, topo_active, edge_mask
                )
            elif mode == "xattn_tokens" and topo_tokens is not None:
                h_a, h_b = self._inject_token_xattn(
                    inj, h_a, h_b, mask_a, mask_b, topo_tokens, topo_active, edge_mask
                )
            elif mode == "pooled_adapter" and topo_pooled is not None:
                delta = self.pooled_adapter[inj](topo_pooled)
                h_a = h_a + torch.where(topo_active.view(-1, 1, 1), delta, torch.zeros_like(delta))
                h_b = h_b + torch.where(topo_active.view(-1, 1, 1), delta, torch.zeros_like(delta))
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
        conditioning_mode: str = "xattn_cls",
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
            conditioning_mode: Which rung of the conditioning ladder to build
                -- exactly one (`ClassifierConfig.conditioning_mode`). All
                four are zero-initialized, so at initialization every one of
                them produces bit-identical logits to the `cond=None` path.
            dropout: Shared dropout rate for the encoder, trunk, and head.
            token_dropout: `SiameseEncoder` token-level dropout rate.

        Raises:
            ValueError: On an unregistered `conditioning_mode`.
        """
        super().__init__()
        if conditioning_mode not in CONDITIONING_MODES:
            raise ValueError(
                f"conditioning_mode must be one of {sorted(CONDITIONING_MODES)}, "
                f"got {conditioning_mode!r}"
            )
        self.conditioning_mode = conditioning_mode
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
            conditioning_mode=conditioning_mode,
        )
        self.head = MLPHead(
            input_dim=d_model,
            hidden_dims=[d_model // 2],
            output_dim=1,
            dropout=dropout,
            activation="gelu",
            norm="layernorm",
        )
        # Built here rather than in the trunk: `film_logit` modulates the fused
        # AB/BA logit, which only exists after `forward`'s max-fuse. Built
        # *only* for that mode, so no other arm carries a never-updated
        # parameter (DDP unused-parameter safety).
        if conditioning_mode == "film_logit":
            self.film = FilmLogitConditioner(d_model, ema_decay=conditioning_ema_decay)

    def freeze_unreachable_conditioning(self) -> None:
        """Freeze whichever conditioning-ladder rung this instance built.

        See `PairClassifier.freeze_unreachable_conditioning` -- this is the
        composite's response to `generator.name: "null"`, under which
        `self.trunk.forward`'s `need_topo` is always `False` and none of
        these submodules ever run. Frozen, not omitted: `self.trunk.topo_xattn`
        keeps its name, `nn.ModuleList` position, and populated-or-empty
        shape exactly as built (the checkpoint-compatibility and
        `train_egostitch._e2e_gate_tanh` telemetry reasons documented at
        `ConditionedPairCrossAttention.__init__`, above), and
        `self.trunk.pooled_adapter` / `self.film` keep their usual
        `state_dict` keys too -- only `requires_grad` changes.
        """
        for module in self.trunk.topo_xattn:
            module.requires_grad_(False)
        pooled_adapter = getattr(self.trunk, "pooled_adapter", None)
        if pooled_adapter is not None:
            for module in pooled_adapter:
                module.requires_grad_(False)
        film = getattr(self, "film", None)
        if film is not None:
            film.requires_grad_(False)

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

        Args:
            pair: See `PairClassifier.forward`.
            cond: See `PairClassifier.forward`.
            masks: See `PairClassifier.forward`.
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
        topo_pooled: torch.Tensor | None = None
        if need_topo:
            assert cond is not None
            if self.conditioning_mode in ("xattn_cls", "xattn_tokens"):
                topo_tokens = torch.cat((cond.ab.tokens, cond.ba.tokens))
            elif self.conditioning_mode == "pooled_adapter":
                topo_pooled = torch.cat((cond.ab.pooled, cond.ba.pooled))

        max_tokens = max(encoded_a.size(1), encoded_b.size(1))
        encoded_a = F.pad(encoded_a, (0, 0, 0, max_tokens - encoded_a.size(1)))
        encoded_b = F.pad(encoded_b, (0, 0, 0, max_tokens - encoded_b.size(1)))

        doubled_active = torch.cat((masks.topo, masks.topo)) if need_topo else None
        doubled_edge_mask = (
            torch.cat((pair.edge_mask, pair.edge_mask)) if pair.edge_mask is not None else None
        )
        pair_features = self.trunk(
            torch.cat((encoded_a, encoded_b)),
            torch.cat((encoded_b, encoded_a)),
            torch.cat((pair.len_a, pair.len_b)),
            torch.cat((pair.len_b, pair.len_a)),
            topo_tokens=topo_tokens,
            topo_pooled=topo_pooled,
            topo_active=doubled_active,
            edge_mask=doubled_edge_mask,
        )
        feat_ab, feat_ba = pair_features.chunk(2)
        feat = torch.max(torch.stack([feat_ab, feat_ba], dim=-1), dim=-1).values
        # The registered mixed-precision contract keeps logits in fp32.
        # Casting after the head is too late: autocast has already quantized
        # the linear outputs, which can destroy the small full-minus-f_logit
        # residual (CLAUDE.md "fp32 island"; design §4).
        with torch.autocast(device_type=feat.device.type, enabled=False):
            logits: torch.Tensor = self.head(feat.float()).squeeze(-1)
            if self.conditioning_mode == "film_logit" and need_topo:
                assert cond is not None and doubled_active is not None
                pooled = torch.cat((cond.ab.pooled, cond.ba.pooled))
                film = self.film(pooled, doubled_active, doubled_edge_mask)
                # The FiLM parameters are produced per *direction* on the same
                # doubled trunk batch every other rung uses, then averaged:
                # the fused logit is symmetric in AB/BA (the max-fuse above),
                # so an asymmetric modulation of it would silently reintroduce
                # an endpoint order the rest of the model is careful not to
                # have. The mean is the only symmetric fusion that is still
                # exactly zero when both directions are zero, which is what
                # keeps the zero-init identity bit-exact.
                film_ab, film_ba = film.chunk(2)
                scale, shift = (0.5 * (film_ab + film_ba)).unbind(dim=-1)
                logits = torch.where(masks.topo, logits * (1.0 + torch.tanh(scale)) + shift, logits)
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


# --------------------------------------------------------------------------- pair-latent generator

_GEN_EPS_SEED = 0xD9
_GEN_LOGVAR_MIN = -6.0
_GEN_LOGVAR_MAX = 2.0


def _masked_token_mean(tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean of the first ``lengths`` token rows per batch element."""
    positions = torch.arange(tokens.size(1), device=tokens.device)
    valid = (positions.unsqueeze(0) < lengths.unsqueeze(1)).to(dtype=tokens.dtype)
    total = (tokens * valid.unsqueeze(-1)).sum(dim=1)
    return total / valid.sum(dim=1).clamp_min(1.0).unsqueeze(-1)


class _SeedRecognition(nn.Module):
    """q(z | c, S*): train-only posterior over the pair latent (kd_d9)."""

    def __init__(self, cond_dim: int, seed_flat: int, z_dim: int) -> None:
        super().__init__()
        self.teacher_proj = nn.Linear(seed_flat, cond_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, 2 * z_dim),
        )

    def forward(self, cond: torch.Tensor, teacher_flat: torch.Tensor) -> torch.Tensor:
        projected = self.teacher_proj(teacher_flat)
        return cast(torch.Tensor, self.net(torch.cat((cond, projected), dim=-1)))


class PairLatentGenerator(nn.Module):
    """CVAE generating the pair's oracle PMA seed tokens from ``(x_u, x_v)`` (kd_d9).

    Deployed path: ``mc_samples`` prior samples decoded and marginalized at
    the logit level through ``delta_head``; the B0 head stays untouched and
    the zero-init ``alpha`` gate makes the model exactly the bare trunk at
    init (and under the alpha=0 scoring control). Training samples fresh
    reparameterized noise; eval uses the fixed ``eval_eps`` buffer (drawn
    once from a dedicated generator, checkpointed), so scoring is RNG-free.
    ``joint_stage`` is trainer-owned: False keeps the fusion-boundary
    stop-gradient (stage 1, generator trained by the KD objective alone);
    True lets task gradients reach decoder/prior/conditioning (stage 2).
    The whole module is an fp32 island (`generator/assemble.py` precedent) --
    cosine/Gram geometry formed on the bf16 ulp grid silently quantizes.
    """

    def __init__(
        self,
        d_model: int,
        *,
        z_dim: int,
        cond_dim: int,
        hidden: int,
        seed_count: int,
        seed_dim: int,
        mc_samples: int,
        kl_free_bits: float,
    ) -> None:
        super().__init__()
        for name, value in (
            ("z_dim", z_dim),
            ("cond_dim", cond_dim),
            ("hidden", hidden),
            ("seed_count", seed_count),
            ("seed_dim", seed_dim),
            ("mc_samples", mc_samples),
        ):
            if value <= 0:
                raise ValueError(f"pair_latent_gen.{name} must be positive, got {value}")
        if kl_free_bits < 0.0:
            raise ValueError(f"pair_latent_gen.kl_free_bits must be >= 0, got {kl_free_bits}")
        self.z_dim = z_dim
        self.seed_count = seed_count
        self.seed_dim = seed_dim
        self.mc_samples = mc_samples
        self.kl_free_bits = kl_free_bits
        self.joint_stage = False
        seed_flat = seed_count * seed_dim
        self.condition = nn.Sequential(
            nn.Linear(3 * d_model, cond_dim),
            nn.GELU(),
            nn.LayerNorm(cond_dim),
        )
        self.prior = nn.Linear(cond_dim, 2 * z_dim)
        self.recognition = _SeedRecognition(cond_dim, seed_flat, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cond_dim + z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, seed_flat),
        )
        self.adapter = nn.Sequential(
            nn.LayerNorm(seed_flat),
            nn.Linear(seed_flat, cond_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(d_model + cond_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.alpha = nn.Parameter(torch.zeros(()))
        eps_generator = torch.Generator().manual_seed(_GEN_EPS_SEED)
        self.register_buffer("eval_eps", torch.randn(mc_samples, z_dim, generator=eps_generator))

    def _condition(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric fp32 conditioning from *detached* trunk endpoint encodings.

        The detach is permanent (the D4 lesson): KD gradients must never
        reshape the trunk, and in stage 2 the task gradient entering through
        the generator stops at this boundary too.
        """
        e_u = _masked_token_mean(encoded_a.detach().float(), lengths_a)
        e_v = _masked_token_mean(encoded_b.detach().float(), lengths_b)
        features = torch.cat((e_u * e_v, e_u + e_v, (e_u - e_v).abs()), dim=-1)
        return cast(torch.Tensor, self.condition(features))

    def _prior_params(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.prior(cond).chunk(2, dim=-1)
        return mu, logvar.clamp(_GEN_LOGVAR_MIN, _GEN_LOGVAR_MAX)

    def _decode(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(torch.cat((cond, z), dim=-1))
        return cast(torch.Tensor, flat.reshape(*flat.shape[:-1], self.seed_count, self.seed_dim))

    def forward_task(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        pair_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The deployed path: fp32 ``(delta (B, 1), delta_std (B,))``."""
        with torch.autocast(device_type=pair_repr.device.type, enabled=False):
            cond = self._condition(encoded_a, encoded_b, lengths_a, lengths_b)
            mu_p, logvar_p = self._prior_params(cond)
            sigma_p = (0.5 * logvar_p).exp()
            if self.training:
                eps = torch.randn(
                    cond.size(0), self.mc_samples, self.z_dim, device=cond.device, dtype=cond.dtype
                )
            else:
                eval_eps = self.eval_eps
                assert isinstance(eval_eps, torch.Tensor)
                eps = eval_eps.to(device=cond.device, dtype=cond.dtype).unsqueeze(0)
            z = mu_p.unsqueeze(1) + sigma_p.unsqueeze(1) * eps
            seeds = self._decode(
                cond.unsqueeze(1).expand(-1, self.mc_samples, -1).reshape(-1, cond.size(-1)),
                z.reshape(-1, self.z_dim),
            ).reshape(cond.size(0), self.mc_samples, self.seed_count, self.seed_dim)
            flat = seeds.reshape(cond.size(0), self.mc_samples, -1)
            if not self.joint_stage:
                flat = flat.detach()
            adapted = self.adapter(flat)
            repeated = pair_repr.float().unsqueeze(1).expand(-1, self.mc_samples, -1)
            per_sample = self.delta_head(torch.cat((repeated, adapted), dim=-1)).squeeze(-1)
            delta = per_sample.mean(dim=1, keepdim=True)
            delta_std = per_sample.std(dim=1, unbiased=False)
        return delta, delta_std

    def forward_kd(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        teacher_seeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The KD-stream path: posterior seeds, KL, prior mean, and dispersion.

        ``seeds_q`` is the posterior-path reconstruction (rsampled once);
        ``kl_per_dim`` the analytic ``KL(q || p)``; ``prior_mean_seeds`` the
        deployed-region decode ``g(c, mu_p)`` for the `kd_prior_cos`
        telemetry (prior-hole witness). ``prior_dispersion`` is the per-row
        seed-space distance between decodes at the first two fixed prior
        samples; with one MC sample both decodes reuse it and the value is zero.
        """
        with torch.autocast(device_type=teacher_seeds.device.type, enabled=False):
            cond = self._condition(encoded_a, encoded_b, lengths_a, lengths_b)
            mu_p, logvar_p = self._prior_params(cond)
            teacher_flat = teacher_seeds.float().reshape(teacher_seeds.size(0), -1)
            mu_q, logvar_q = self.recognition(cond, teacher_flat).chunk(2, dim=-1)
            logvar_q = logvar_q.clamp(_GEN_LOGVAR_MIN, _GEN_LOGVAR_MAX)
            z_q = mu_q + (0.5 * logvar_q).exp() * torch.randn_like(mu_q)
            seeds_q = self._decode(cond, z_q)
            kl = (
                0.5 * (logvar_p - logvar_q)
                + (logvar_q.exp() + (mu_q - mu_p).square()) / (2.0 * logvar_p.exp())
                - 0.5
            )
            prior_mean_seeds = self._decode(cond, mu_p)
            eval_eps = self.eval_eps
            assert isinstance(eval_eps, torch.Tensor)
            eps = eval_eps[:2].to(device=cond.device, dtype=cond.dtype)
            if eps.size(0) == 1:
                eps = eps.expand(2, -1)
            prior_z = mu_p.unsqueeze(1) + (0.5 * logvar_p).exp().unsqueeze(1) * eps.unsqueeze(0)
            prior_samples = self._decode(
                cond.unsqueeze(1).expand(-1, 2, -1).reshape(-1, cond.size(-1)),
                prior_z.reshape(-1, self.z_dim),
            ).reshape(cond.size(0), 2, self.seed_count, self.seed_dim)
            prior_dispersion = torch.linalg.vector_norm(
                prior_samples[:, 0] - prior_samples[:, 1], dim=(1, 2)
            )
        return seeds_q, kl, prior_mean_seeds, prior_dispersion


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
                f"model_config.label_smoothing must be in [0.0, 1.0), got {self.label_smoothing}"
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
        # B1 student path: zero-init low-rank node-factor residual on the pair
        # logit. `node_factor_dim: 0` (the default and BEST_V3_1_CONFIG) keeps
        # the baseline bit-identical; the KD arms set 64.
        node_factor_dim = _to_int(
            model_config.get("node_factor_dim", 0), "model_config.node_factor_dim"
        )
        # Optional embedding-KD alignment head riding the same bottleneck;
        # meaningless without a node-factor path to project out of.
        node_factor_align_dim = _to_int(
            model_config.get("node_factor_align_dim", 0), "model_config.node_factor_align_dim"
        )
        if node_factor_align_dim > 0 and node_factor_dim == 0:
            raise ValueError("model_config.node_factor_align_dim > 0 requires node_factor_dim > 0")
        self.node_factor: NodeFactorBottleneck | None = (
            NodeFactorBottleneck(self.d_model, node_factor_dim, node_factor_align_dim)
            if node_factor_dim > 0
            else None
        )
        # D9 pair-latent generation: absent section => module not built, the
        # baseline stays bit-identical (`node_factor_dim: 0` pattern). Built
        # last so the trunk's RNG init stream is unchanged by its presence.
        plg_raw = model_config.get("pair_latent_gen")
        self.pair_latent_gen: PairLatentGenerator | None = None
        if plg_raw is not None:
            if not isinstance(plg_raw, dict) or not plg_raw:
                raise ValueError("model_config.pair_latent_gen must be a non-empty mapping")
            plg_cfg = _to_mapping(plg_raw, "model_config.pair_latent_gen")
            known_keys = {
                "z_dim",
                "cond_dim",
                "hidden",
                "seed_count",
                "seed_dim",
                "mc_samples",
                "kl_free_bits",
            }
            unknown_keys = sorted(set(plg_cfg) - known_keys)
            if unknown_keys:
                raise ValueError(f"unknown pair_latent_gen keys: {unknown_keys}")
            self.pair_latent_gen = PairLatentGenerator(
                self.d_model,
                z_dim=_to_int(plg_cfg.get("z_dim", 64), "pair_latent_gen.z_dim"),
                cond_dim=_to_int(plg_cfg.get("cond_dim", 256), "pair_latent_gen.cond_dim"),
                hidden=_to_int(plg_cfg.get("hidden", 512), "pair_latent_gen.hidden"),
                seed_count=_to_int(plg_cfg.get("seed_count", 4), "pair_latent_gen.seed_count"),
                seed_dim=_to_int(plg_cfg.get("seed_dim", 512), "pair_latent_gen.seed_dim"),
                mc_samples=_to_int(plg_cfg.get("mc_samples", 4), "pair_latent_gen.mc_samples"),
                kl_free_bits=_to_float(
                    plg_cfg.get("kl_free_bits", 0.05), "pair_latent_gen.kl_free_bits"
                ),
            )

    def pair_latent_gen_parameters(self) -> list[nn.Parameter]:
        """Generator param group for the trainer's stage-2 LR switch (kd_d9)."""
        if self.pair_latent_gen is None:
            return []
        return [
            parameter
            for component in (
                self.pair_latent_gen.condition,
                self.pair_latent_gen.prior,
                self.pair_latent_gen.recognition,
                self.pair_latent_gen.decoder,
            )
            for parameter in component.parameters()
        ]

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
        if self.node_factor is not None:
            factors = cast(
                NodeFactorOutputs,
                self.node_factor(encoded_a, encoded_b, lengths_a, lengths_b),
            )
            residual = factors.residual.reshape(logits.shape).to(dtype=logits.dtype)
            logits = logits + residual
            output["logits"] = logits
            pair_factor = factors.z_a * factors.z_b
            output["node_factor_pair"] = pair_factor
            output["node_factor_residual"] = factors.residual
            if self.node_factor.align_head is not None:
                output["node_factor_align"] = self.node_factor.align_head(pair_factor)
        if self.pair_latent_gen is not None:
            delta, delta_std = self.pair_latent_gen.forward_task(
                encoded_a, encoded_b, lengths_a, lengths_b, pair_repr
            )
            gated = (self.pair_latent_gen.alpha * delta).reshape(logits.shape)
            logits = logits + gated.to(dtype=logits.dtype)
            output["logits"] = logits
            output["gen_delta_std"] = delta_std
            teacher_seeds = merged.get("kd_teacher_seeds")
            if teacher_seeds is not None:
                seeds_q, kl, prior_mean_seeds, prior_dispersion = self.pair_latent_gen.forward_kd(
                    encoded_a, encoded_b, lengths_a, lengths_b, teacher_seeds
                )
                output["gen_seeds_q"] = seeds_q
                output["gen_kl"] = kl
                output["gen_seeds_prior_mean"] = prior_mean_seeds
                output["gen_prior_dispersion"] = prior_dispersion
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
    "FilmLogitConditioner",
    "GatedCrossAttention",
    "NULL_ALL_HEAD",
    "NULL_NONE",
    "PooledAdapter",
    "V3_1",
    "build_best_v3_1",
    "masks_for_null",
    "sample_branch_masks",
]
