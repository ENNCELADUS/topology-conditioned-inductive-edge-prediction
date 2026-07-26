"""Structure-only stitched scaffold (design rev 3 §3.1–§3.2, spec §14.4.5).

Node order: [endpoint_src, endpoint_dst, slots_src(K), slots_dst(K)].
Node features (FEAT_DIM=11): [onehot4(anchor); pi; mult; deg_star; deg_intra;
deg_align; deg_close; closed-wedge mass]. Edge types (EDGE_TYPES=4):
star / intra-side slot-slot / alignment / closure.
Deliberately EXCLUDED: slot content h, grounding gate g, pointer, and the
grounded-identity-match label — those belong to the content pathway.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
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
_CONTROL_SEED = 0

ScaffoldInputPerturbation = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


class ScaffoldTokens(NamedTuple):
    """Batched structure-only scaffold tensors."""

    feats: torch.Tensor
    adj: torch.Tensor


def _stable_control_generator(
    node_u: str, node_v: str, side: str, *, seed: int = _CONTROL_SEED
) -> torch.Generator:
    """Return the v2 canonical-pair-keyed CPU generator."""
    src, dst = sorted((node_u, node_v))
    digest = hashlib.blake2b(f"{src}|{dst}|{side}|{seed}".encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], byteorder="little", signed=False))
    return generator


def _stable_slot_permutation(
    node_u: str, node_v: str, side: str, slots: int
) -> torch.Tensor:
    """Return the v2 canonical-pair-keyed slot permutation."""
    return torch.randperm(
        slots,
        generator=_stable_control_generator(node_u, node_v, side),
    )


def _checkerboard_rewire(
    matrices: torch.Tensor,
    generators: Sequence[torch.Generator],
    capacities: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply keyed checkerboard transfers to a matrix batch without Python swap loops."""
    matrix_count, slots, width = matrices.shape
    if width != slots:
        raise ValueError(f"checkerboard inputs must be square, got {tuple(matrices.shape)}")
    if len(generators) != matrix_count:
        raise ValueError("checkerboard generator count must equal the matrix batch size")
    if capacities is not None and capacities.shape != matrices.shape:
        raise ValueError("checkerboard capacities must match the matrix batch shape")
    if slots < 4:
        raise ValueError("off-diagonal checkerboard swaps require at least four slots")

    swap_count = 8 * slots * slots
    pair_count = slots // 2
    draws_per_round = pair_count * (pair_count - 1)
    round_count = (swap_count + draws_per_round - 1) // draws_per_round
    row_pair, column_pair = torch.where(
        ~torch.eye(pair_count, dtype=torch.bool)
    )
    # Each round partitions slots into disjoint pairs and assigns every row
    # pair to every other column pair. The resulting 2x2 blocks contain only
    # off-diagonal cells and do not overlap, so all draws in a round are safe
    # to apply concurrently. Only the much smaller round axis is sequential.
    paired_slots = torch.stack(
        [
            torch.rand((round_count, slots), generator=generator)
            .argsort(dim=-1)[:, : 2 * pair_count]
            .reshape(round_count, pair_count, 2)
            for generator in generators
        ],
        dim=1,
    )
    draw_indices = torch.stack(
        (
            paired_slots[:, :, row_pair, 0],
            paired_slots[:, :, row_pair, 1],
            paired_slots[:, :, column_pair, 0],
            paired_slots[:, :, column_pair, 1],
        ),
        dim=-1,
    ).to(device=matrices.device)
    units = torch.stack(
        [
            torch.rand((swap_count,), generator=generator).clamp(
                min=torch.finfo(torch.float32).eps,
                max=1.0 - torch.finfo(torch.float32).eps,
            )
            for generator in generators
        ],
        dim=0,
    )
    padded_units = torch.zeros(
        matrix_count,
        round_count * draws_per_round,
        dtype=units.dtype,
    )
    padded_units[:, :swap_count] = units
    round_units = (
        padded_units.reshape(matrix_count, round_count, draws_per_round)
        .transpose(0, 1)
        .to(device=matrices.device, dtype=matrices.dtype)
    )
    matrix_offsets = (
        torch.arange(matrix_count, device=matrices.device)[:, None] * slots * slots
    )

    def transfer(
        flat: torch.Tensor,
        draw: tuple[torch.Tensor, torch.Tensor],
        flat_capacities: torch.Tensor | None,
    ) -> torch.Tensor:
        indices, unit = draw
        row_i, row_k, column_j, column_l = indices.unbind(dim=-1)
        recipient_ij = matrix_offsets + row_i * slots + column_j
        recipient_kl = matrix_offsets + row_k * slots + column_l
        donor_il = matrix_offsets + row_i * slots + column_l
        donor_kj = matrix_offsets + row_k * slots + column_j
        bounds = torch.stack((flat[donor_il], flat[donor_kj]), dim=-1)
        if flat_capacities is not None:
            bounds = torch.cat(
                (
                    bounds,
                    torch.stack(
                        (
                            flat_capacities[recipient_ij] - flat[recipient_ij],
                            flat_capacities[recipient_kl] - flat[recipient_kl],
                        ),
                        dim=-1,
                    ).clamp_min(0.0),
                ),
                dim=-1,
            )
        delta = unit * bounds.amin(dim=-1)
        update_indices = torch.stack(
            (
                recipient_ij,
                recipient_kl,
                donor_il,
                donor_kj,
            ),
            dim=-1,
        ).flatten()
        update_values = torch.stack((delta, delta, -delta, -delta), dim=-1).flatten()
        return flat.scatter_add(0, update_indices, update_values)

    flat = matrices.flatten()
    flat_capacities = capacities.flatten() if capacities is not None else None
    for indices, unit in zip(draw_indices, round_units, strict=True):
        flat = transfer(flat, (indices, unit), flat_capacities)
    return flat.reshape_as(matrices)


def make_scaffold_input_perturbation(
    mode: str,
    pairs: Sequence[tuple[str, str]],
) -> ScaffoldInputPerturbation:
    """Build a deterministic §14.4.5 perturbation applied before scaffold assembly."""
    controlled_pairs = tuple(pairs)

    def perturb(
        adj_src: torch.Tensor,
        adj_dst: torch.Tensor,
        plan: torch.Tensor,
        pi_src: torch.Tensor,
        pi_dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, slots, _ = adj_src.shape
        expected = (batch_size, slots, slots)
        if adj_src.shape != expected or adj_dst.shape != expected or plan.shape != expected:
            raise ValueError(
                "scaffold-control inputs must have matching square shapes: "
                f"{tuple(adj_src.shape)}, {tuple(adj_dst.shape)}, {tuple(plan.shape)}"
            )
        if len(controlled_pairs) != batch_size:
            raise ValueError("scaffold-control pair count must equal the scaffold batch size")
        if pi_src.shape != (batch_size, slots) or pi_dst.shape != (batch_size, slots):
            raise ValueError("scaffold-control pi shapes must match the adjacency batch")

        canonical_src = torch.tensor(
            [node_u <= node_v for node_u, node_v in controlled_pairs],
            device=adj_src.device,
            dtype=torch.bool,
        )
        if mode == "shuffle_within_pair_v3":
            canonical_permutations = [
                (
                    _stable_slot_permutation(node_u, node_v, "src", slots),
                    _stable_slot_permutation(node_u, node_v, "dst", slots),
                )
                for node_u, node_v in controlled_pairs
            ]
            perm_src = torch.stack(
                [
                    src_perm if is_src else dst_perm
                    for (src_perm, dst_perm), is_src in zip(
                        canonical_permutations, canonical_src.cpu().tolist(), strict=True
                    )
                ]
            ).to(adj_src.device)
            perm_dst = torch.stack(
                [
                    dst_perm if is_src else src_perm
                    for (src_perm, dst_perm), is_src in zip(
                        canonical_permutations, canonical_src.cpu().tolist(), strict=True
                    )
                ]
            ).to(adj_src.device)
            rows = torch.arange(batch_size, device=adj_src.device)[:, None, None]
            return (
                adj_src[rows, perm_src[:, :, None], perm_src[:, None, :]],
                adj_dst[rows, perm_dst[:, :, None], perm_dst[:, None, :]],
                plan[rows, perm_src[:, :, None], perm_dst[:, None, :]],
            )
        if mode != "rewire_checkerboard_v1":
            raise ValueError(f"unknown scaffold input perturbation: {mode!r}")

        zero_src = adj_src - torch.diag_embed(torch.diagonal(adj_src, dim1=-2, dim2=-1))
        zero_dst = adj_dst - torch.diag_embed(torch.diagonal(adj_dst, dim1=-2, dim2=-1))
        weight_src = pi_src[:, :, None] * pi_src[:, None, :]
        weight_dst = pi_dst[:, :, None] * pi_dst[:, None, :]
        weighted_src = zero_src * weight_src
        weighted_dst = zero_dst * weight_dst
        canonical_plan = torch.where(canonical_src[:, None, None], plan, plan.transpose(1, 2))
        src_generators = [
            _stable_control_generator(node_u, node_v, "src" if node_u <= node_v else "dst")
            for node_u, node_v in controlled_pairs
        ]
        dst_generators = [
            _stable_control_generator(node_u, node_v, "dst" if node_u <= node_v else "src")
            for node_u, node_v in controlled_pairs
        ]
        plan_generators = [
            _stable_control_generator(node_u, node_v, "plan")
            for node_u, node_v in controlled_pairs
        ]
        rewired = _checkerboard_rewire(
            torch.cat((weighted_src, weighted_dst, canonical_plan), dim=0),
            (*src_generators, *dst_generators, *plan_generators),
            torch.cat(
                (
                    weight_src,
                    weight_dst,
                    torch.full_like(canonical_plan, torch.inf),
                ),
                dim=0,
            ),
        )
        rewired_src = rewired[:batch_size]
        rewired_dst = rewired[batch_size : 2 * batch_size]
        rewired_plan = rewired[2 * batch_size :]
        rewired_src = 0.5 * (rewired_src + rewired_src.transpose(1, 2))
        rewired_dst = 0.5 * (rewired_dst + rewired_dst.transpose(1, 2))
        perturbed_src = torch.where(weight_src > 0, rewired_src / weight_src, 0.0)
        perturbed_dst = torch.where(weight_dst > 0, rewired_dst / weight_dst, 0.0)
        perturbed_plan = torch.where(
            canonical_src[:, None, None],
            rewired_plan,
            rewired_plan.transpose(1, 2),
        )
        return perturbed_src, perturbed_dst, perturbed_plan

    return perturb


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


def build_scaffold(
    slots_src: SlotSet,
    slots_dst: SlotSet,
    plan: torch.Tensor,
    *,
    perturbation: ScaffoldInputPerturbation | None = None,
) -> ScaffoldTokens:
    """Assemble the stitched scaffold from two slot sets and the OT plan.

    Args:
        slots_src: Source-side generated slot set (spec Sec 2 heads).
        slots_dst: Destination-side generated slot set.
        plan: Shape ``(B, K, K)`` alignment plan from
            :func:`src.model.egostitch.stitch.sinkhorn_plan`.
        perturbation: Optional deterministic transform of
            ``(adj_src, adj_dst, plan)`` applied before any scaffold channels
            are derived, per spec §14.4.5.

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
        slot_adj_src = slot_adj_src - torch.diag_embed(
            torch.diagonal(slot_adj_src, dim1=-2, dim2=-1)
        )
        slot_adj_dst = slot_adj_dst - torch.diag_embed(
            torch.diagonal(slot_adj_dst, dim1=-2, dim2=-1)
        )
        plan32 = plan.float()
        if perturbation is not None:
            slot_adj_src, slot_adj_dst, plan32 = perturbation(
                slot_adj_src,
                slot_adj_dst,
                plan32,
                pi_src,
                pi_dst,
            )
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

        adj_src_zero = slot_adj_src
        adj_dst_zero = slot_adj_dst
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
