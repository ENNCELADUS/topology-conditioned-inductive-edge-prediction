"""Module 3: Assemble -- Hungarian matching, Sinkhorn stitch, and the scaffold.

Merges three former standalone modules (three-component refactor design §5
consolidation pass): ``matching.py`` (Hungarian slot-target matching),
``stitch.py`` (unbalanced Sinkhorn slot alignment), and ``scaffold.py`` (the
structure-only stitched scaffold assembled from the alignment plan). They form
one pipeline stage -- matching feeds the reconstruction losses, Sinkhorn
alignment feeds both `L_align` and the scaffold's alignment/closure channels,
and the scaffold is literally built from the Sinkhorn plan -- so they now live
in one file rather than three that always change together.

Scaffold (design rev 3 §3.1–§3.2, spec §14.4.5): node order
``[endpoint_src, endpoint_dst, slots_src(K), slots_dst(K)]``. Node features
(FEAT_DIM=11): [onehot4(anchor); pi; mult; deg_star; deg_intra; deg_align;
deg_close; closed-wedge mass]. Edge types (EDGE_TYPES=4): star / intra-side
slot-slot / alignment / closure. Deliberately EXCLUDED: slot content h,
grounding gate g, pointer, and the grounded-identity-match label — those
belong to the content pathway.

Stitch: retained in Stage 1 because the closure channel ``s2`` consumes the
alignment plan (spec Sec 13.1). Stage-1 cost drops the code term (weights
x5/4): ``C = (5/4)·||h_i - h_j||^2 + (5/16)·|pi_i - pi_j|``. Numerics:
log-domain, fixed iteration count (determinism), computed in an fp32 island
under autocast (bf16-safe), marginals ``a ∝ pi·m`` clamped at ``1e-8``
(degenerate all-``pi``-zero inputs stay finite). Unrolled and differentiable.

Matching: Hungarian slot-target matching, Stage-1 cost (spec Sec 2,
Sec 13.3). Two-pass compound cost without the code-agreement term (weights
uniformly rescaled x7/6; ``linear_sum_assignment`` is invariant to the
rescale — recorded for exactness only):
``C = (7/6)·||h - proj(x_v)||^2 + (7/24)·|deg_bucket(k) - deg_bucket(v)|
+ (7/24)·overlap_penalty`` (overlap only on the second pass). ``deg_bucket``
is pinned (spec Sec 13.3) to log2 multiplicity buckets
``{[1,2), [2,4), [4,8), [8,16), [16,32]}``. Costs are built on GPU; the
assignment runs on CPU under ``no_grad`` and is **constant in the backward
pass** (standard DETR practice).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from src.model.egostitch.layers import stable_log

if TYPE_CHECKING:
    # Deferred to break the assemble.py <-> imagine.py import cycle:
    # `EgoStitchStage1.node_losses` (imagine.py) calls `match_slots` (this
    # module), while this module's matching functions only ever use `SlotSet`
    # in type annotations (never as a runtime value) -- `from __future__ import
    # annotations` means those annotations are never evaluated at import time,
    # so the import is safe to defer to type-checking only.
    from src.model.egostitch.generator.imagine import SlotSet

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


def _stable_slot_permutation(node_u: str, node_v: str, side: str, slots: int) -> torch.Tensor:
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
    row_pair, column_pair = torch.where(~torch.eye(pair_count, dtype=torch.bool))
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
    matrix_offsets = torch.arange(matrix_count, device=matrices.device)[:, None] * slots * slots

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
        diagonal_src = torch.diag_embed(torch.diagonal(adj_src, dim1=-2, dim2=-1))
        diagonal_dst = torch.diag_embed(torch.diagonal(adj_dst, dim1=-2, dim2=-1))
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
            _stable_control_generator(node_u, node_v, "plan") for node_u, node_v in controlled_pairs
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
        perturbed_src = torch.where(weight_src > 0, rewired_src / weight_src, 0.0) + diagonal_src
        perturbed_dst = torch.where(weight_dst > 0, rewired_dst / weight_dst, 0.0) + diagonal_dst
        perturbed_plan = torch.where(
            canonical_src[:, None, None],
            rewired_plan,
            rewired_plan.transpose(1, 2),
        )
        return perturbed_src, perturbed_dst, perturbed_plan

    return perturb


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
            :func:`src.model.egostitch.generator.assemble.sinkhorn_plan`.
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
        plan32 = plan.float()
        if perturbation is not None:
            slot_adj_src, slot_adj_dst, plan32 = perturbation(
                slot_adj_src,
                slot_adj_dst,
                plan32,
                pi_src,
                pi_dst,
            )
        adj_src_zero = slot_adj_src - torch.diag_embed(
            torch.diagonal(slot_adj_src, dim1=-2, dim2=-1)
        )
        adj_dst_zero = slot_adj_dst - torch.diag_embed(
            torch.diagonal(slot_adj_dst, dim1=-2, dim2=-1)
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

        closure = 0.5 * (torch.bmm(adj_src_zero, plan32) + torch.bmm(plan32, adj_dst_zero))
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


# --------------------------------------------------------------------------- stitch (Sinkhorn)

_STITCH_W_FEAT = 5.0 / 4.0
_STITCH_W_PI = 5.0 / 16.0


def stitch_cost(
    h_i: torch.Tensor, h_j: torch.Tensor, pi_i: torch.Tensor, pi_j: torch.Tensor
) -> torch.Tensor:
    """Stage-1 OT cost matrix (spec Sec 13.3).

    Args:
        h_i: Shape ``(B, K, d_p)`` side-i slot embeddings.
        h_j: Shape ``(B, K, d_p)`` side-j slot embeddings.
        pi_i: Shape ``(B, K)`` side-i existence probabilities.
        pi_j: Shape ``(B, K)`` side-j existence probabilities.

    Returns:
        Shape ``(B, K, K)`` cost tensor.
    """
    diff = h_i[:, :, None, :] - h_j[:, None, :, :]
    feat = (diff**2).sum(dim=-1)
    pi_gap = (pi_i[:, :, None] - pi_j[:, None, :]).abs()
    return _STITCH_W_FEAT * feat + _STITCH_W_PI * pi_gap


def sinkhorn_log_plan(
    h_i: torch.Tensor,
    h_j: torch.Tensor,
    pi_i: torch.Tensor,
    pi_j: torch.Tensor,
    m_i: torch.Tensor,
    m_j: torch.Tensor,
    *,
    eps: float = 0.1,
    iters: int = 20,
    tau: float = 1.0,
) -> torch.Tensor:
    """Compute ``log(Pi)`` for the unbalanced-OT alignment plan (spec Sec 3).

    Marginals ``a ∝ pi_i·m_i`` and ``b ∝ pi_j·m_j`` are normalized to unit
    mass and clamped at ``1e-8``. KL relaxation with strength `tau` gives the
    standard damped log-domain updates (damping ``phi = tau / (tau + eps)``);
    ``log(Pi) = (f + g - C) / eps + log a + log b`` — slots may be unmatched.
    Fixed `iters` iterations; fully differentiable (unrolled). Exposing this
    quantity lets conditional losses remain live when ``exp(log(Pi))``
    underflows in fp32.

    Args:
        h_i: Shape ``(B, K, d_p)`` side-i slot embeddings.
        h_j: Shape ``(B, K, d_p)`` side-j slot embeddings.
        pi_i: Shape ``(B, K)`` side-i existence probabilities.
        pi_j: Shape ``(B, K)`` side-j existence probabilities.
        m_i: Shape ``(B, K)`` side-i multiplicities.
        m_j: Shape ``(B, K)`` side-j multiplicities.
        eps: Entropic regularizer.
        iters: Fixed iteration count.
        tau: Unbalanced KL relaxation strength.

    Returns:
        Shape ``(B, K, K)`` log-domain alignment plan.
    """
    # fp32 island: inputs must be promoted before either the cost or marginal
    # products are formed; casting their bf16 results afterward is too late.
    with torch.autocast(device_type=h_i.device.type, enabled=False):
        h_i32, h_j32 = h_i.float(), h_j.float()
        pi_i32, pi_j32 = pi_i.float(), pi_j.float()
        m_i32, m_j32 = m_i.float(), m_j.float()
        cost = stitch_cost(h_i32, h_j32, pi_i32, pi_j32)
        a = pi_i32 * m_i32
        b = pi_j32 * m_j32
        a = a / torch.clamp(a.sum(dim=-1, keepdim=True), min=1e-8)
        b = b / torch.clamp(b.sum(dim=-1, keepdim=True), min=1e-8)
        log_a = stable_log(a)
        log_b = stable_log(b)

        phi = tau / (tau + eps)
        f = torch.zeros_like(a)
        g = torch.zeros_like(b)
        for _ in range(iters):
            f = (
                -phi
                * eps
                * torch.logsumexp((g[:, None, :] - cost) / eps + log_b[:, None, :], dim=2)
            )
            g = (
                -phi
                * eps
                * torch.logsumexp((f[:, :, None] - cost) / eps + log_a[:, :, None], dim=1)
            )
        return (f[:, :, None] + g[:, None, :] - cost) / eps + log_a[:, :, None] + log_b[:, None, :]


def sinkhorn_plan(
    h_i: torch.Tensor,
    h_j: torch.Tensor,
    pi_i: torch.Tensor,
    pi_j: torch.Tensor,
    m_i: torch.Tensor,
    m_j: torch.Tensor,
    *,
    eps: float = 0.1,
    iters: int = 20,
    tau: float = 1.0,
) -> torch.Tensor:
    """Compute the fp32 unbalanced-OT alignment plan ``Pi`` (spec Sec 3)."""
    return torch.exp(
        sinkhorn_log_plan(
            h_i,
            h_j,
            pi_i,
            pi_j,
            m_i,
            m_j,
            eps=eps,
            iters=iters,
            tau=tau,
        )
    )


# --------------------------------------------------------------------------- matching (Hungarian)

_MATCH_W_FEAT = 7.0 / 6.0
_MATCH_W_BUCKET = 7.0 / 24.0
_MATCH_W_OVERLAP = 7.0 / 24.0
_N_BUCKETS = 5


class Assignment:
    """One batch of Hungarian slot-target assignments (constant in backward).

    Attributes:
        slot_idx: Per-node int64 arrays of assigned slot indices.
        target_idx: Per-node int64 arrays of assigned target indices, aligned
            with `slot_idx`.
    """

    def __init__(
        self, slot_idx: list[NDArray[np.int64]], target_idx: list[NDArray[np.int64]]
    ) -> None:
        if len(slot_idx) != len(target_idx):
            raise ValueError("slot_idx and target_idx must have equal batch length")
        self.slot_idx = slot_idx
        self.target_idx = target_idx

    def __len__(self) -> int:
        """Return the batch size."""
        return len(self.slot_idx)

    def __eq__(self, other: object) -> bool:
        """Elementwise equality of both index lists (for determinism tests)."""
        if not isinstance(other, Assignment):
            return NotImplemented
        return len(self) == len(other) and all(
            np.array_equal(a, b) and np.array_equal(c, d)
            for a, b, c, d in zip(
                self.slot_idx, other.slot_idx, self.target_idx, other.target_idx, strict=True
            )
        )

    def __hash__(self) -> int:  # pragma: no cover - identity hashing suffices
        """Identity hash (assignments are never dict keys in hot paths)."""
        return id(self)


def deg_bucket_index(m: torch.Tensor) -> torch.Tensor:
    """Log2 multiplicity bucket index (spec Sec 13.3), in ``0..4``."""
    clamped = torch.clamp(m, min=1.0)
    return torch.clamp(torch.floor(torch.log2(clamped)), max=float(_N_BUCKETS - 1))


def base_match_cost(
    slots: SlotSet,
    target_proj: torch.Tensor,
    target_mult: torch.Tensor,
) -> torch.Tensor:
    """First-pass matching cost (feature + degree-bucket terms).

    Args:
        slots: The generated slot set.
        target_proj: Shape ``(B, T, d_p)`` **stop-gradient** projected target
            features (spec Sec 13.7).
        target_mult: Shape ``(B, T)`` target multiplicity labels.

    Returns:
        Shape ``(B, K, T)`` cost tensor (differentiable through the slot side;
        used only to *derive* the constant assignment).
    """
    diff = slots.h[:, :, None, :] - target_proj[:, None, :, :]
    feat = (diff**2).sum(dim=-1)
    bucket_slot = deg_bucket_index(slots.mult)
    bucket_target = deg_bucket_index(target_mult)
    bucket = (bucket_slot[:, :, None] - bucket_target[:, None, :]).abs()
    return _MATCH_W_FEAT * feat + _MATCH_W_BUCKET * bucket


def overlap_penalty(
    slots: SlotSet, target_adj: torch.Tensor, assignment: Assignment
) -> torch.Tensor:
    """Second-pass overlap term (spec Sec 2): slot-adjacency vs target-adjacency.

    For candidate pairing ``(k, v)``: mean over the first-pass matched pairs
    ``(k', v')`` (excluding ``k`` itself) of ``|adj[k, k'] - A[v, v']|``.

    Args:
        slots: The generated slot set.
        target_adj: Shape ``(B, T, T)`` binary adjacency among targets.
        assignment: The first-pass assignment.

    Returns:
        Shape ``(B, K, T)`` penalty tensor (zeros for nodes with < 2 matches).
    """
    batch, k = slots.adj.shape[0], slots.adj.shape[1]
    t = target_adj.shape[1]
    penalty = slots.adj.new_zeros((batch, k, t))
    for b in range(batch):
        s_idx = torch.from_numpy(assignment.slot_idx[b]).to(slots.adj.device)
        t_idx = torch.from_numpy(assignment.target_idx[b]).to(slots.adj.device)
        if s_idx.numel() < 2:
            continue
        # (K, M): each candidate slot k vs matched slots k'; (T, M): each
        # candidate target v vs matched targets v'.
        adj_rows = slots.adj[b][:, s_idx]
        target_rows = target_adj[b][:, t_idx]
        disagreement = (adj_rows[:, None, :] - target_rows[None, :, :]).abs()
        # Exclude the (k == k') self column per candidate slot.
        self_mask = s_idx[None, :] == torch.arange(k, device=s_idx.device)[:, None]  # (K, M)
        weights = (~self_mask).float()[:, None, :]
        denom = torch.clamp(weights.sum(dim=-1), min=1.0)
        penalty[b] = (disagreement * weights).sum(dim=-1) / denom
    return penalty


def hungarian_assign(cost: torch.Tensor, target_mask: torch.Tensor) -> Assignment:
    """Solve the per-node assignment on CPU (one transfer, ``no_grad``).

    Args:
        cost: Shape ``(B, K, T)`` matching costs.
        target_mask: Shape ``(B, T)`` booleans; ``False`` targets are padding.

    Returns:
        The `Assignment` (constant in the backward pass).
    """
    with torch.no_grad():
        cost_np = cost.detach().float().cpu().numpy()
        mask_np = target_mask.detach().cpu().numpy()
    slot_idx: list[NDArray[np.int64]] = []
    target_idx: list[NDArray[np.int64]] = []
    for b in range(cost_np.shape[0]):
        valid = np.flatnonzero(mask_np[b])
        if valid.size == 0:
            slot_idx.append(np.empty(0, dtype=np.int64))
            target_idx.append(np.empty(0, dtype=np.int64))
            continue
        rows, cols = linear_sum_assignment(cost_np[b][:, valid])
        slot_idx.append(rows.astype(np.int64))
        target_idx.append(valid[cols].astype(np.int64))
    return Assignment(slot_idx, target_idx)


def match_slots(
    slots: SlotSet,
    *,
    target_proj: torch.Tensor,
    target_mult: torch.Tensor,
    target_adj: torch.Tensor,
    target_mask: torch.Tensor,
) -> Assignment:
    """Run the pinned two-pass Hungarian matching (spec Sec 2, Sec 13.3).

    Pass 1 uses the feature + degree-bucket terms; pass 2 adds the overlap
    penalty computed against pass 1's matched pairs and re-solves.

    Args:
        slots: The generated slot set.
        target_proj: Shape ``(B, T, d_p)`` stop-gradient projected targets.
        target_mult: Shape ``(B, T)`` target multiplicity labels.
        target_adj: Shape ``(B, T, T)`` adjacency among targets.
        target_mask: Shape ``(B, T)`` target validity mask.

    Returns:
        The final `Assignment`.
    """
    base = base_match_cost(slots, target_proj, target_mult)
    first = hungarian_assign(base, target_mask)
    penalty = overlap_penalty(slots, target_adj, first)
    return hungarian_assign(base + _MATCH_W_OVERLAP * penalty, target_mask)
