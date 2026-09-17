"""``L_slot`` and ``L_topo``, the motif-prompt graph-supervision losses.

Design: ``docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md``
sections 7.3 and 7.5. ``L_slot`` supervises four multisets per row -- the 8 wedge
products, the 64 bridge path products, the 16 raw attachment weights and the 64
raw interior weights -- each through a descending sort and a Huber term.
``L_topo`` is the light representation term comparing the four token fields the
immutable teacher reads from the predicted and the true graph. Sibling of
`src.distill.struct_losses`, where the other training losses live.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional as F

from src.data.motif_template import slot_profiles

_TERMS: tuple[tuple[str, str], ...] = (
    ("p", "beta_p"),
    ("q", "beta_q"),
    ("a", "beta_a"),
    ("b", "beta_i"),
)


def slot_loss_rows(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    beta_p: float,
    beta_q: float,
    beta_a: float,
    beta_i: float,
    huber_delta: float,
) -> torch.Tensor:
    """Per-row ``L_slot``: descending-sorted Huber over four per-family multisets.

    Path products, not raw closure weights, are supervised: the 16 closure edge
    weights as one multiset are invariant to moving weight between the two sides
    of a wedge, so the products are what pins the wedge mass. Because
    ``wedge_mass`` and ``bridge_mass`` are sums over the supervised multisets and
    a sum is permutation-invariant, matching sorted products pins both count-head
    quantities exactly. The two product terms route gradient to both (or all
    three) edges of each motif; the two raw terms keep gradient flowing to an
    edge whose path product vanishes because a *different* edge of that path is
    zero. The raw interior term is not decoration: for ``q = a b c`` the interior
    gradient is attenuated quadratically by small attachments, and the attachment
    term contributes nothing to it (spec section 7.3).

    Sorting is a permutation, so the loss is invariant to within-role slot
    permutation and to the endpoint swap, consistent with the slot randomisation
    of spec section 3. Rows are independent, so a caller masks self rows out.

    Args:
        predicted: ``(B, 96)`` predicted edge weights.
        target: ``(B, 96)`` compiled edge weights; detached here.
        beta_p: Weight of the wedge-product term.
        beta_q: Weight of the bridge-path-product term.
        beta_a: Weight of the raw attachment term.
        beta_i: Weight of the raw interior term.
        huber_delta: Huber transition point.

    Returns:
        ``(B,)`` per-row losses, in float32.

    Raises:
        ValueError: If the two tables disagree in shape.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted {tuple(predicted.shape)} and target {tuple(target.shape)} must match"
        )
    parts = slot_profiles(predicted)
    truth = slot_profiles(target.detach())
    betas = {"beta_p": beta_p, "beta_q": beta_q, "beta_a": beta_a, "beta_i": beta_i}
    total = predicted.new_zeros(predicted.size(0), dtype=torch.float32)
    for key, name in _TERMS:
        scale = betas[name]
        if scale == 0.0:
            continue
        ours = parts[key].sort(dim=1, descending=True).values
        theirs = truth[key].sort(dim=1, descending=True).values
        term = F.huber_loss(ours, theirs, delta=huber_delta, reduction="none").mean(dim=1)
        total = total + scale * term
    return total


def slot_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    beta_p: float,
    beta_q: float,
    beta_a: float,
    beta_i: float,
    huber_delta: float,
) -> torch.Tensor:
    """Batch-mean ``L_slot``.

    Args:
        predicted: ``(B, 96)`` predicted edge weights.
        target: ``(B, 96)`` compiled edge weights.
        beta_p: Weight of the wedge-product term.
        beta_q: Weight of the bridge-path-product term.
        beta_a: Weight of the raw attachment term.
        beta_i: Weight of the raw interior term.
        huber_delta: Huber transition point.

    Returns:
        The scalar mean of `slot_loss_rows`.
    """
    return slot_loss_rows(
        predicted,
        target,
        beta_p=beta_p,
        beta_q=beta_q,
        beta_a=beta_a,
        beta_i=beta_i,
        huber_delta=huber_delta,
    ).mean()


TOPO_FIELDS: tuple[str, ...] = ("topo_u", "topo_v", "topo_rel", "topo_cnt")


def topo_loss_rows(
    student: Mapping[str, torch.Tensor], teacher: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Per-row ``L_topo``: squared error of fixed non-affine LayerNorms (spec section 7.5).

    ``N`` is `torch.nn.functional.layer_norm` with no learnable affine, applied to
    each of the four token fields, so the term compares directions rather than
    scales and a global shift or positive rescaling of a field costs nothing. The
    teacher side is detached here; the caller runs ``R_T(Ahat)`` with autograd so
    the loss reaches the generator, and only ``R_T(A*)`` under `torch.no_grad`.

    Rows are independent, so a caller masks self rows out (spec section 3).

    Args:
        student: The four fields the teacher read from the predicted graph.
        teacher: The four fields the immutable teacher read from the true graph.

    Returns:
        ``(B,)`` per-row mean squared error over fields and dimensions, in float32.

    Raises:
        KeyError: If either mapping is missing one of `TOPO_FIELDS`.
    """
    terms: list[torch.Tensor] = []
    for name in TOPO_FIELDS:
        ours = student[name].float()
        theirs = teacher[name].float().detach()
        width = ours.size(-1)
        normalised = F.layer_norm(ours, (width,))
        reference = F.layer_norm(theirs, (width,))
        terms.append((normalised - reference).square().mean(dim=-1))
    return torch.stack(terms, dim=-1).mean(dim=-1)


def stream_row_counts(rows: Sequence[torch.Tensor], masks: Sequence[torch.Tensor]) -> list[float]:
    """Each stream's rank-local valid-row count, in stream order.

    A stream this rank holds no rows for counts zero, whatever its mask says: the
    structural stream's chunks are distributed by token cost, so a rank can be
    left without any of them.

    Args:
        rows: One ``(n_s,)`` per-row tensor per stream.
        masks: One ``(n_s,)`` valid-row mask per stream, aligned with ``rows``.

    Returns:
        One count per stream.

    Raises:
        ValueError: If ``rows`` and ``masks`` differ in length.
    """
    if len(rows) != len(masks):
        raise ValueError(f"rows ({len(rows)}) and masks ({len(masks)}) must be aligned")
    return [
        0.0 if row.numel() == 0 else float(mask.to(row).sum().item())
        for row, mask in zip(rows, masks, strict=True)
    ]


def stream_mean(
    rows: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    *,
    like: torch.Tensor,
    global_counts: Sequence[float] | None = None,
    world_size: int = 1,
) -> torch.Tensor:
    """Average per-stream row means, then average across the streams that exist.

    The task stream and the structural stream carry very different row counts, so
    the spec averages inside a stream first and weights the streams equally. An
    empty stream -- no rows at all, or no valid nonself rows -- contributes a
    differentiable zero rather than dropping out of the graph, so DDP sees the
    same parameters on every rank (spec section 7.5).

    Under DDP the per-stream mean has to be a *global* mean. Structural chunks are
    distributed by token cost, so ranks hold unequal counts and a rank can hold no
    structural row at all; dividing by rank-local counts and letting DDP average
    the ranks equally then weights the streams by where their rows happened to
    land. The caller reduces `stream_row_counts` across ranks and passes them as
    ``global_counts``: each rank contributes its local sum over the global count,
    scaled by ``world_size`` so DDP's averaging leaves the global stream mean.
    With one rank the scale is 1 and the result is the plain local mean, exactly
    as `src.train_b0.scale_ddp_mean_loss` treats the task loss.

    Args:
        rows: One ``(n_s,)`` per-row tensor per stream.
        masks: One ``(n_s,)`` valid-row mask per stream, aligned with ``rows``.
        like: A tensor supplying dtype, device and the autograd connection.
        global_counts: Valid-row count per stream summed over ranks; rank-local
            counts when omitted.
        world_size: Ranks DDP averages.

    Returns:
        The scalar mean.

    Raises:
        ValueError: If ``rows``, ``masks`` and ``global_counts`` are not aligned,
            or ``world_size`` is below one.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be at least 1, got {world_size}")
    counts = stream_row_counts(rows, masks) if global_counts is None else list(global_counts)
    if len(counts) != len(rows):
        raise ValueError(f"counts ({len(counts)}) and rows ({len(rows)}) must be aligned")
    zero = like.sum() * 0.0
    shares: list[torch.Tensor] = []
    for row, mask, count in zip(rows, masks, counts, strict=True):
        if count <= 0.0:
            continue
        # A rank with no row of a stream other ranks do hold still contributes a
        # differentiable zero, so every rank stacks the same streams.
        local = row.sum() if row.numel() == 0 else (row * mask.to(row)).sum()
        shares.append(local / count)
    if not shares:
        return zero
    return zero + float(world_size) * torch.stack(shares).sum() / len(shares)
