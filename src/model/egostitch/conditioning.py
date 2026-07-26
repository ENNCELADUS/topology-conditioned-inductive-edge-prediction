"""E2E head conditioning primitives: branch masks and gated cross-attention.

Design contract: docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-
encoder-design.md rev 3 — three mutually exclusive head nulls; training uses
per-sample multiplicative masks, evaluation uses batch-level hard bypasses.
Mask semantics: True = pathway ACTIVE for that pair (shared across AB/BA).
The rev-3.1 centered residual is governed by spec §14.4.2.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn import functional as dist_functional

NULL_NONE = "none"
NULL_ALL_HEAD = "all_head"
NULL_TOPO_HEAD = "topo_head"
NULL_CONTENT_HEAD = "content_head"
_KNOWN_NULLS = (NULL_NONE, NULL_ALL_HEAD, NULL_TOPO_HEAD, NULL_CONTENT_HEAD)


class HeadNullMasks(NamedTuple):
    """Per-pair pathway activity masks (True = active)."""

    topo: torch.Tensor
    cont: torch.Tensor


def sample_branch_masks(
    batch_size: int,
    p_topo: float,
    p_cont: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> HeadNullMasks:
    """Sample independent per-pair branch-dropout masks (design §4)."""
    topo = torch.rand(batch_size, generator=generator) >= p_topo
    cont = torch.rand(batch_size, generator=generator) >= p_cont
    return HeadNullMasks(topo=topo.to(device), cont=cont.to(device))


def masks_for_null(null: str, batch_size: int, device: torch.device) -> HeadNullMasks:
    """Deterministic masks realizing one of the §3.5 null conditions."""
    if null not in _KNOWN_NULLS:
        raise ValueError(f"unknown head-null condition: {null!r}")
    on = torch.ones(batch_size, dtype=torch.bool, device=device)
    off = torch.zeros(batch_size, dtype=torch.bool, device=device)
    topo = off if null in (NULL_ALL_HEAD, NULL_TOPO_HEAD) else on
    cont = off if null in (NULL_ALL_HEAD, NULL_CONTENT_HEAD) else on
    return HeadNullMasks(topo=topo, cont=cont)


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
