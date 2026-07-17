"""E2E head conditioning primitives: branch masks and gated cross-attention.

Design contract: docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-
encoder-design.md rev 3 — three mutually exclusive head nulls; training uses
per-sample multiplicative masks, evaluation uses batch-level hard bypasses.
Mask semantics: True = pathway ACTIVE for that pair (shared across AB/BA).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

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
    """Zero-init tanh-gated cross-attention residual sublayer (design §3.4).

    ``cls <- cls + active * tanh(gate) * XAttn(LN(cls), tokens)``. The gate is a
    scalar parameter initialized to zero, so at init (and whenever ``active`` is
    False) the sublayer is an exact identity — the checkpoint-exact bypass
    property the §3.5 null taxonomy relies on.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        cls: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the gated cross-attention residual update to ``cls``.

        Args:
            cls: Shape ``(B, 1, d_model)`` query token.
            tokens: Shape ``(B, T, d_model)`` key/value tokens.
            token_mask: Optional shape ``(B, T)`` bool mask, True = valid
                token; invalid tokens are excluded from attention.
            active: Shape ``(B,)`` bool mask, True = pathway active for that
                sample; inactive samples get an exact identity bypass.

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
        scale = active.to(cls.dtype).view(-1, 1, 1)
        return cls + scale * torch.tanh(self.gate) * attn_out
