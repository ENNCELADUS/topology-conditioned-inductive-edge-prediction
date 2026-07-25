"""Structure-only stitched scaffold (design rev 3 §3.1–§3.2).

Node order: [endpoint_src, endpoint_dst, slots_src(K), slots_dst(K)].
Node features (FEAT_DIM=11): [onehot4(anchor); pi; mult; deg_star; deg_intra;
deg_align; deg_close; closed-wedge mass]. Edge types (EDGE_TYPES=4):
star / intra-side slot-slot / alignment / closure.
Deliberately EXCLUDED: slot content h, grounding gate g, pointer, and the
grounded-identity-match label — those belong to the content pathway.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.layers import stable_log

N_ANCHOR_TYPES = 4
FEAT_DIM = 11
EDGE_TYPES = 4
_STAR, _INTRA, _ALIGN, _CLOSE = 0, 1, 2, 3
_SRC, _DST, _SLOT_SRC, _SLOT_DST = 0, 1, 2, 3


class ScaffoldTokens(NamedTuple):
    """Batched structure-only scaffold tensors."""

    feats: torch.Tensor
    adj: torch.Tensor


def counterpart_membership(
    slots: SlotSet, other_proj: torch.Tensor, tau_kappa: torch.Tensor
) -> torch.Tensor:
    """Return each slot's compatibility with the counterpart endpoint."""
    slot_direction = F.normalize(slots.h, p=2.0, dim=-1)
    other_direction = F.normalize(other_proj, p=2.0, dim=-1)
    distance = (slot_direction - other_direction[:, None, :]).square().sum(dim=-1)
    return -distance / tau_kappa + stable_log(slots.pi * slots.mult)


def grounded_identity_match(
    pointer_a: torch.Tensor,
    gate_a: torch.Tensor,
    ids_a: torch.Tensor,
    pointer_b: torch.Tensor,
    gate_b: torch.Tensor,
    ids_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return symmetric flags for gated slots selecting a shared node id."""
    gated_a = gate_a > 0.5
    gated_b = gate_b > 0.5
    selected_a = torch.gather(ids_a, 1, pointer_a.argmax(dim=-1))
    selected_b = torch.gather(ids_b, 1, pointer_b.argmax(dim=-1))
    shared = selected_a[:, :, None] == selected_b[:, None, :]
    matched_a = gated_a & (shared & gated_b[:, None, :]).any(dim=-1)
    matched_b = gated_b & (shared & gated_a[:, :, None]).any(dim=1)
    return matched_a.float(), matched_b.float()


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
    if slots_dst.pi.shape != (b, k):
        raise ValueError(
            "scaffold sides require equal slot counts and batch size: "
            f"{tuple(slots_src.pi.shape)} != {tuple(slots_dst.pi.shape)}"
        )
    if plan.shape != (b, k, k):
        raise ValueError(f"plan shape must be {(b, k, k)}, got {tuple(plan.shape)}")
    v = 2 + 2 * k
    device = slots_src.pi.device

    # fp32 island: promote before forming adjacency products, closed wedges,
    # or closure blocks; casting bf16 products afterward is too late.
    with torch.autocast(device_type=device.type, enabled=False):
        pi_src, pi_dst = slots_src.pi.float(), slots_dst.pi.float()
        mult_src, mult_dst = slots_src.mult.float(), slots_dst.mult.float()
        slot_adj_src, slot_adj_dst = slots_src.adj.float(), slots_dst.adj.float()
        plan32 = plan.float()
        adj = torch.zeros(b, EDGE_TYPES, v, v, device=device, dtype=torch.float32)
        s_src = slice(2, 2 + k)
        s_dst = slice(2 + k, v)

        star_src = pi_src * mult_src
        star_dst = pi_dst * mult_dst
        adj[:, _STAR, 0, s_src] = star_src
        adj[:, _STAR, s_src, 0] = star_src
        adj[:, _STAR, 1, s_dst] = star_dst
        adj[:, _STAR, s_dst, 1] = star_dst

        intra_src = slot_adj_src * pi_src[:, :, None] * pi_src[:, None, :]
        intra_dst = slot_adj_dst * pi_dst[:, :, None] * pi_dst[:, None, :]
        adj[:, _INTRA, s_src, s_src] = intra_src
        adj[:, _INTRA, s_dst, s_dst] = intra_dst

        adj[:, _ALIGN, s_src, s_dst] = plan32
        adj[:, _ALIGN, s_dst, s_src] = plan32.transpose(1, 2)

        adj_src_zero = slot_adj_src - torch.diag_embed(
            torch.diagonal(slot_adj_src, dim1=-2, dim2=-1)
        )
        adj_dst_zero = slot_adj_dst - torch.diag_embed(
            torch.diagonal(slot_adj_dst, dim1=-2, dim2=-1)
        )
        closure = 0.5 * (
            torch.bmm(adj_src_zero, plan32) + torch.bmm(plan32, adj_dst_zero)
        )
        adj[:, _CLOSE, s_src, s_dst] = closure
        adj[:, _CLOSE, s_dst, s_src] = closure.transpose(1, 2)

        t_src = torch.diagonal(
            torch.bmm(torch.bmm(plan32, adj_dst_zero), plan32.transpose(1, 2)),
            dim1=-2,
            dim2=-1,
        )
        t_dst = torch.diagonal(
            torch.bmm(torch.bmm(plan32.transpose(1, 2), adj_src_zero), plan32),
            dim1=-2,
            dim2=-1,
        )

        feats = torch.zeros(b, v, FEAT_DIM, device=device, dtype=torch.float32)
        feats[:, 0, _SRC] = 1.0
        feats[:, 1, _DST] = 1.0
        feats[:, s_src, _SLOT_SRC] = 1.0
        feats[:, s_dst, _SLOT_DST] = 1.0
        ones = torch.ones(b, 1, device=device, dtype=torch.float32)
        feats[:, :, 4] = torch.cat([ones, ones, pi_src, pi_dst], dim=1)
        feats[:, :, 5] = torch.cat([ones, ones, mult_src, mult_dst], dim=1)
        feats[:, :, 6:10] = adj.sum(dim=-1).permute(0, 2, 1)
        feats[:, s_src, 10] = t_src
        feats[:, s_dst, 10] = t_dst
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
    membership_src: torch.Tensor,
    membership_dst: torch.Tensor,
) -> torch.Tensor:
    """Content tokens: ``[h; pi; gate; identity-match; membership]``.

    Args:
        slots_src: Source-side generated slot set (spec Sec 2 heads).
        slots_dst: Destination-side generated slot set.
        matched_src: Shape ``(B, K)`` grounded-identity-match signal in
            ``[0, 1]`` for the source-side slots (moved here from the anchor
            labels per rev 3).
        matched_dst: Shape ``(B, K)`` grounded-identity-match signal in
            ``[0, 1]`` for the destination-side slots.
        membership_src: Shape ``(B, K)`` counterpart-membership compatibility
            for source-side slots (former-s1 per-slot signal).
        membership_dst: Shape ``(B, K)`` counterpart-membership compatibility
            for destination-side slots.

    Returns:
        Shape ``(B, 2K, d_p + 4)`` content tokens, src slots first.
    """
    b, k = slots_src.pi.shape
    if slots_dst.pi.shape != (b, k):
        raise ValueError(
            "content sides require equal slot counts and batch size: "
            f"{tuple(slots_src.pi.shape)} != {tuple(slots_dst.pi.shape)}"
        )
    named = {
        "matched_src": matched_src,
        "matched_dst": matched_dst,
        "membership_src": membership_src,
        "membership_dst": membership_dst,
    }
    for name, value in named.items():
        if value.shape != (b, k):
            raise ValueError(f"{name} shape must be {(b, k)}, got {tuple(value.shape)}")

    def side(s: SlotSet, matched: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.cat(
            [
                s.h,
                s.pi[..., None],
                s.gate[..., None],
                matched[..., None],
                membership[..., None],
            ],
            dim=-1,
        )
        return out

    return torch.cat(
        [
            side(slots_src, matched_src, membership_src),
            side(slots_dst, matched_dst, membership_dst),
        ],
        dim=1,
    )


class ContentProjector(nn.Module):
    """Linear projection of content tokens into the trunk width."""

    def __init__(self, d_p: int, d_model: int) -> None:
        """Build the content-token linear projection.

        Args:
            d_p: Slot content embedding dimension (pre-projection width less
                the 4 scalar channels).
            d_model: Trunk model width to project into.
        """
        super().__init__()
        self.proj = nn.Linear(d_p + 4, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project content tokens ``(B, 2K, d_p + 4)`` to ``(B, 2K, d_model)``.

        Args:
            tokens: Content tokens produced by :func:`build_content_tokens`.

        Returns:
            Shape ``(B, 2K, d_model)`` projected tokens.
        """
        out: torch.Tensor = self.proj(tokens)
        return out
