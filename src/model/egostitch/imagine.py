"""Module 2 (Stage-1 form): Imagine — DETR-style ego-net set decoder (spec Sec 2, Sec 13.2).

Codebook-free conditioning: ``T_cond = [W_x x_u; W_e e_u]`` (2 tokens); query
init from ``e_u``; no CVAE token. Owns the shared projection
``proj: d -> d_p`` consumed by the matching cost, ``L_feat``, and ``s1``
(spec Sec 13.7 — those consumers stop-gradient its target-side output).

The generated ego-net is **intermediate context** for the binary edge decision,
never a final output (docs/lit-review-plan.md Sec 5).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from src.model.egostitch.config import EgoStitchConfig

NULL_MODE_FULL = 0
NULL_MODE_CONTENT = 1
NULL_MODE_ALL = 2


class SlotSet(NamedTuple):
    """Per-node generated slot set (spec Sec 2 heads).

    Attributes:
        h: Shape ``(B, K, d_p)`` slot embeddings in projected feature space.
        pi: Shape ``(B, K)`` slot existence probabilities in ``[0, 1]``.
        mult: Shape ``(B, K)`` slot multiplicities in ``[1, m_max]``.
        gate: Shape ``(B, K)`` grounding-gate probabilities in ``[0, 1]``.
        pointer: Shape ``(B, K, n_g)`` pointer softmax over grounding candidates.
        adj: Shape ``(B, K, K)`` symmetric slot-slot adjacency in ``[0, 1]``.
        adj_logits: Shape ``(B, K, K)`` pre-sigmoid adjacency logits.
    """

    h: torch.Tensor
    pi: torch.Tensor
    mult: torch.Tensor
    gate: torch.Tensor
    pointer: torch.Tensor
    adj: torch.Tensor
    adj_logits: torch.Tensor


class DenoiseSlots(NamedTuple):
    """Denoising-query outputs (fixed-assignment supervision only, spec Sec 2).

    Attributes:
        h: Shape ``(B, K_d, d_p)`` denoise-slot embeddings.
        pi: Shape ``(B, K_d)`` existence probabilities.
        mult: Shape ``(B, K_d)`` multiplicities.
    """

    h: torch.Tensor
    pi: torch.Tensor
    mult: torch.Tensor


class _DecoderLayer(nn.Module):
    """One Imagine decoder layer: slot self-attn -> memory cross-attn -> FFN."""

    def __init__(self, d_h: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.norm_self = nn.LayerNorm(d_h)
        self.norm_cross = nn.LayerNorm(d_h)
        self.norm_ffn = nn.LayerNorm(d_h)
        self.ffn = nn.Sequential(nn.Linear(d_h, 4 * d_h), nn.GELU(), nn.Linear(4 * d_h, d_h))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.norm_self(queries + attn_out)
        attn_out, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        queries = self.norm_cross(queries + attn_out)
        out: torch.Tensor = self.norm_ffn(queries + self.ffn(queries))
        return out


class ImagineDecoder(nn.Module):
    """DETR-style slot decoder with codebook-free conditioning (spec Sec 13.2).

    Query init: ``Q_k = W_q [proj(x_{g_k}); e_u]`` for ``k <= min(K, n_g)``, else
    ``Q_k = q_k^base + W_q' e_u``. Conditioning dropout replaces ``T_cond``
    (``∅_content``) or both ``T_cond`` and ``T_g`` (``∅_all``) with learned null
    tokens.
    """

    def __init__(self, config: EgoStitchConfig) -> None:
        """Build the decoder, its conditioning projections, and the slot heads.

        Args:
            config: The pinned Stage-1 configuration.
        """
        super().__init__()
        self.config = config
        d, d_p, d_z, d_h, k = (
            config.input_dim,
            config.d_p,
            config.d_z,
            config.d_h,
            config.slots,
        )
        # Shared projection (spec Sec 13.7): matching / L_feat / s1 all consume it.
        self.proj = nn.Linear(d, d_p)

        self.w_x = nn.Linear(d, d_h)
        self.w_e = nn.Linear(d_z, d_h)
        self.w_g = nn.Linear(d_p, d_h)
        self.w_q = nn.Linear(d_p + d_z, d_h)
        self.w_q_base = nn.Linear(d_z, d_h)
        self.q_base = nn.Parameter(torch.randn(k, d_h) * 0.02)
        self.null_cond = nn.Parameter(torch.randn(1, d_h) * 0.02)
        self.null_ground = nn.Parameter(torch.randn(1, d_h) * 0.02)

        self.layers = nn.ModuleList(
            _DecoderLayer(d_h, config.n_heads) for _ in range(config.decoder_layers)
        )

        self.head_h = nn.Linear(d_h, d_p)
        self.head_pi = nn.Linear(d_h, 1)
        self.head_mult = nn.Linear(d_h, 1)
        self.head_gate = nn.Linear(d_h, 1)
        self.head_pointer = nn.Linear(d_h, d_h)
        self.head_adj = nn.Linear(d_p, d_p)

    def _build_queries(
        self, e: torch.Tensor, ground_proj: torch.Tensor, extra_proj: torch.Tensor | None
    ) -> torch.Tensor:
        """Assemble the K (+ optional denoise) slot queries."""
        batch, k = e.shape[0], self.config.slots
        n_dynamic = min(k, ground_proj.shape[1])
        dynamic = self.w_q(
            torch.cat([ground_proj[:, :n_dynamic], e[:, None, :].expand(-1, n_dynamic, -1)], dim=-1)
        )
        if n_dynamic < k:
            base = (
                self.q_base[None, n_dynamic:k].expand(batch, -1, -1) + self.w_q_base(e)[:, None, :]
            )
            queries = torch.cat([dynamic, base], dim=1)
        else:
            queries = dynamic
        if extra_proj is not None:
            extra = self.w_q(
                torch.cat([extra_proj, e[:, None, :].expand(-1, extra_proj.shape[1], -1)], dim=-1)
            )
            queries = torch.cat([queries, extra], dim=1)
        return queries

    def _build_memory(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        ground_tokens: torch.Tensor,
        null_mode: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble ``[T_cond; T_g]`` with conditioning-dropout null substitution."""
        cond = torch.stack([self.w_x(x), self.w_e(e)], dim=1)  # (B, 2, d_h)
        drop_cond = (null_mode >= NULL_MODE_CONTENT)[:, None, None]
        cond = torch.where(drop_cond, self.null_cond[None].expand_as(cond), cond)
        drop_ground = (null_mode == NULL_MODE_ALL)[:, None, None]
        ground_tokens = torch.where(
            drop_ground, self.null_ground[None].expand_as(ground_tokens), ground_tokens
        )
        return torch.cat([cond, ground_tokens], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        ground_x: torch.Tensor,
        *,
        null_mode: torch.Tensor | None = None,
        extra_proj: torch.Tensor | None = None,
    ) -> tuple[SlotSet, DenoiseSlots | None]:
        """Generate the slot set for a node batch.

        Args:
            x: Shape ``(B, d)`` frozen node features.
            e: Shape ``(B, d_z)`` Tokenize-lite embeddings.
            ground_x: Shape ``(B, n_g, d)`` grounding-candidate features.
            null_mode: Optional ``(B,)`` int tensor in ``{0, 1, 2}``
                (full / ``∅_content`` / ``∅_all``); defaults to full.
            extra_proj: Optional ``(B, K_d, d_p)`` noised true-neighbor
                projections for denoising queries (fixed assignments; caller
                supervises them, spec Sec 2).

        Returns:
            ``(slots, denoise)`` — `denoise` is ``None`` when `extra_proj` is.
        """
        if null_mode is None:
            null_mode = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        ground_proj = self.proj(ground_x)
        ground_tokens = self.w_g(ground_proj)
        memory = self._build_memory(x, e, ground_tokens, null_mode)
        queries = self._build_queries(e, ground_proj, extra_proj)

        decoded = queries
        for layer in self.layers:
            decoded = layer(decoded, memory)

        k = self.config.slots
        main, extra = decoded[:, :k], decoded[:, k:]

        h = self.head_h(main)
        pi = torch.sigmoid(self.head_pi(main)).squeeze(-1)
        mult = 1.0 + torch.nn.functional.softplus(self.head_mult(main)).squeeze(-1)
        mult = torch.clamp(mult, max=float(self.config.m_max))
        gate = torch.sigmoid(self.head_gate(main)).squeeze(-1)
        pointer_scores = (
            torch.einsum("bkd,bgd->bkg", self.head_pointer(main), ground_tokens)
            / float(self.config.d_h) ** 0.5
        )
        pointer = torch.softmax(pointer_scores, dim=-1)
        adj_proj = self.head_adj(h)
        adj_logits = (
            torch.einsum("bkd,bld->bkl", adj_proj, adj_proj) / float(self.config.d_p) ** 0.5
        )
        adj = torch.sigmoid(adj_logits)

        slots = SlotSet(
            h=h,
            pi=pi,
            mult=mult,
            gate=gate,
            pointer=pointer,
            adj=adj,
            adj_logits=adj_logits,
        )

        denoise: DenoiseSlots | None = None
        if extra_proj is not None:
            d_h_extra = self.head_h(extra)
            d_pi = torch.sigmoid(self.head_pi(extra)).squeeze(-1)
            d_mult = 1.0 + torch.nn.functional.softplus(self.head_mult(extra)).squeeze(-1)
            d_mult = torch.clamp(d_mult, max=float(self.config.m_max))
            denoise = DenoiseSlots(h=d_h_extra, pi=d_pi, mult=d_mult)
        return slots, denoise
