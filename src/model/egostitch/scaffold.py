"""Structure-only stitched scaffold (design rev 3 §3.1–§3.2).

Node order: [endpoint_src, endpoint_dst, slots_src(K), slots_dst(K)].
Node features (FEAT_DIM=9): [onehot4(anchor); pi; mult; deg_star; deg_intra;
deg_align]. Edge types (EDGE_TYPES=3): star / intra-side slot-slot / alignment.
Deliberately EXCLUDED: slot content h, grounding gate g, pointer, and the
grounded-identity-match label — those belong to the content pathway.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from src.model.egostitch.imagine import SlotSet

N_ANCHOR_TYPES = 4
FEAT_DIM = 9
EDGE_TYPES = 3
_STAR, _INTRA, _ALIGN = 0, 1, 2
_SRC, _DST, _SLOT_SRC, _SLOT_DST = 0, 1, 2, 3


class ScaffoldTokens(NamedTuple):
    """Batched structure-only scaffold tensors."""

    feats: torch.Tensor
    adj: torch.Tensor


def build_scaffold(slots_src: SlotSet, slots_dst: SlotSet, plan: torch.Tensor) -> ScaffoldTokens:
    """Assemble the stitched scaffold from two slot sets and the OT plan.

    Args:
        slots_src: Source-side generated slot set (spec Sec 2 heads).
        slots_dst: Destination-side generated slot set.
        plan: Shape ``(B, K, K)`` alignment plan from
            :func:`src.model.egostitch.stitch.sinkhorn_plan`.

    Returns:
        The structure-only ``ScaffoldTokens`` (no slot content, no grounding).
    """
    b, k = slots_src.pi.shape
    v = 2 + 2 * k
    device, dtype = slots_src.pi.device, slots_src.pi.dtype

    adj = torch.zeros(b, EDGE_TYPES, v, v, device=device, dtype=dtype)
    s_src = slice(2, 2 + k)
    s_dst = slice(2 + k, v)

    star_src = slots_src.pi * slots_src.mult
    star_dst = slots_dst.pi * slots_dst.mult
    adj[:, _STAR, 0, s_src] = star_src
    adj[:, _STAR, s_src, 0] = star_src
    adj[:, _STAR, 1, s_dst] = star_dst
    adj[:, _STAR, s_dst, 1] = star_dst

    intra_src = slots_src.adj * slots_src.pi[:, :, None] * slots_src.pi[:, None, :]
    intra_dst = slots_dst.adj * slots_dst.pi[:, :, None] * slots_dst.pi[:, None, :]
    adj[:, _INTRA, s_src, s_src] = intra_src
    adj[:, _INTRA, s_dst, s_dst] = intra_dst

    adj[:, _ALIGN, s_src, s_dst] = plan
    adj[:, _ALIGN, s_dst, s_src] = plan.transpose(1, 2)

    feats = torch.zeros(b, v, FEAT_DIM, device=device, dtype=dtype)
    feats[:, 0, _SRC] = 1.0
    feats[:, 1, _DST] = 1.0
    feats[:, s_src, _SLOT_SRC] = 1.0
    feats[:, s_dst, _SLOT_DST] = 1.0
    ones = torch.ones(b, 1, device=device, dtype=dtype)
    feats[:, :, 4] = torch.cat([ones, ones, slots_src.pi, slots_dst.pi], dim=1)
    feats[:, :, 5] = torch.cat([ones, ones, slots_src.mult, slots_dst.mult], dim=1)
    feats[:, :, 6:9] = adj.sum(dim=-1).permute(0, 2, 1)
    return ScaffoldTokens(feats=feats, adj=adj)


def swap_direction(tokens: ScaffoldTokens) -> ScaffoldTokens:
    """Relabel src<->dst anchor channels for the BA stream (structure fixed).

    Args:
        tokens: A ``ScaffoldTokens`` built by :func:`build_scaffold`.

    Returns:
        A new ``ScaffoldTokens`` with the anchor one-hot channels (endpoint
        src/dst, slot-of-src/slot-of-dst) swapped; all other features, and
        the adjacency, are unchanged (relabeling only, no recomputation).
    """
    perm = [_DST, _SRC, _SLOT_DST, _SLOT_SRC]
    feats = tokens.feats.clone()
    feats[..., :N_ANCHOR_TYPES] = tokens.feats[..., perm]
    return ScaffoldTokens(feats=feats, adj=tokens.adj)


def build_content_tokens(
    slots_src: SlotSet,
    slots_dst: SlotSet,
    matched_src: torch.Tensor,
    matched_dst: torch.Tensor,
) -> torch.Tensor:
    """Content-pathway tokens: ``[h; pi; gate; grounded-identity-match]``.

    Args:
        slots_src: Source-side generated slot set (spec Sec 2 heads).
        slots_dst: Destination-side generated slot set.
        matched_src: Shape ``(B, K)`` grounded-identity-match signal in
            ``[0, 1]`` for the source-side slots (moved here from the anchor
            labels per rev 3).
        matched_dst: Shape ``(B, K)`` grounded-identity-match signal in
            ``[0, 1]`` for the destination-side slots.

    Returns:
        Shape ``(B, 2K, d_p + 3)`` content tokens, src slots first.
    """

    def side(s: SlotSet, matched: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.cat(
            [s.h, s.pi[..., None], s.gate[..., None], matched[..., None]], dim=-1
        )
        return out

    return torch.cat([side(slots_src, matched_src), side(slots_dst, matched_dst)], dim=1)


class ContentProjector(nn.Module):
    """Linear projection of content tokens into the trunk width."""

    def __init__(self, d_p: int, d_model: int) -> None:
        """Build the content-token linear projection.

        Args:
            d_p: Slot content embedding dimension (pre-projection width less
                the 3 scalar channels).
            d_model: Trunk model width to project into.
        """
        super().__init__()
        self.proj = nn.Linear(d_p + 3, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project content tokens ``(B, 2K, d_p + 3)`` to ``(B, 2K, d_model)``.

        Args:
            tokens: Content tokens produced by :func:`build_content_tokens`.

        Returns:
            Shape ``(B, 2K, d_model)`` projected tokens.
        """
        out: torch.Tensor = self.proj(tokens)
        return out
