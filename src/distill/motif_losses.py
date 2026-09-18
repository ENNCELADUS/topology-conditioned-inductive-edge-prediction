"""``L_slot`` and ``L_topo``, the motif-prompt graph-supervision losses.

Design: ``docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md``
sections 7.3 and 7.5. ``L_slot`` supervises five multisets per row -- the 8 wedge
products, the 64 bridge path products, the 16 raw attachment weights, the 64
raw interior weights and the 16 raw closure weights -- each through a descending
sort and a Huber term.
``L_topo`` is the light representation term comparing the four token fields the
immutable teacher reads from the predicted and the true graph. Sibling of
`src.distill.struct_losses`, where the other training losses live.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional as F

from src.data.motif_template import CLOSURE_U, CLOSURE_V, slot_profiles

_TERMS: tuple[tuple[str, str], ...] = (
    ("p", "beta_p"),
    ("q", "beta_q"),
    ("a", "beta_a"),
    ("b", "beta_i"),
    ("c", "beta_c"),
)


def slot_loss_rows(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    beta_p: float,
    beta_q: float,
    beta_a: float,
    beta_i: float,
    beta_c: float,
    huber_delta: float,
) -> torch.Tensor:
    """Per-row ``L_slot``: descending-sorted Huber over five per-family multisets.

    Both the wedge products and the raw closure weights are supervised. The
    products are what pins the wedge mass: the 16 closure edge weights as one
    multiset are invariant to moving weight between the two sides of a wedge, so
    the raw term alone cannot tell a wedge from two weights sitting on different
    witnesses. Because ``wedge_mass`` and ``bridge_mass`` are sums over the
    supervised multisets and a sum is permutation-invariant, matching sorted
    products pins both count-head quantities exactly.

    The product terms alone, however, do not *reach* the closure gates. For
    ``p = w(u,c) w(c,v)`` the gradient into one side carries the partner weight
    and the sigmoid derivative, and at a gate initialised from the closure
    *density* both are small: the measured slot gradient into the closure head
    was 1/1800 of the interior head's through the whole of wave 1, and the head
    never left its initialisation (`docs/results/motif_prompt_verdict/README.md`).
    ``beta_c`` restores an O(1) signal on the 16 raw closure weights, exactly as
    ``beta_i`` does for the interior family, whose path gradient is attenuated
    quadratically by small attachments. ``beta_c = 0`` reproduces the wave-1 loss.

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
        beta_c: Weight of the raw closure term.
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
    betas = {
        "beta_p": beta_p,
        "beta_q": beta_q,
        "beta_a": beta_a,
        "beta_i": beta_i,
        "beta_c": beta_c,
    }
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
    beta_c: float,
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
        beta_c: Weight of the raw closure term.
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
        beta_c=beta_c,
        huber_delta=huber_delta,
    ).mean()


def closure_nonempty(target: torch.Tensor) -> torch.Tensor:
    """Which rows of a compiled template carry at least one closure edge.

    A row's closure family is the 16 weights of `CLOSURE_U` and `CLOSURE_V`: the
    eight shared-neighbour witnesses seen from each endpoint. At the task
    stream's 1:5 positive:negative ratio most rows have none of them, so the
    graph loss of spec section 7.5 is dominated by rows whose closure target is
    exactly zero. This is the predicate `closure_balanced_row_weights` splits on.

    Args:
        target: ``(B, 96)`` compiled edge weights.

    Returns:
        ``(B,)`` boolean, true where any closure weight is positive.

    Raises:
        ValueError: If ``target`` is not ``(B, 96)``.
    """
    if target.dim() != 2 or target.size(1) != 96:
        raise ValueError(f"target must be (B, 96), got {tuple(target.shape)}")
    left = (target[:, CLOSURE_U] > 0).any(dim=1)
    right = (target[:, CLOSURE_V] > 0).any(dim=1)
    return left | right


def balanced_row_weights(
    nonempty: torch.Tensor,
    valid: torch.Tensor,
    *,
    positive_share: float,
    global_nonempty: float | None = None,
    global_empty: float | None = None,
) -> torch.Tensor:
    """Row weights giving the non-empty rows a fixed share of the graph loss.

    The weights sum to the valid-row count, so multiplying per-row losses by them
    and reducing through `stream_mean` keeps a mean: only the row *distribution*
    the generator's loss sees changes, never its scale, the task stream, the
    sampler or any other loss.

    Under DDP the shares have to be formed from the counts summed over ranks, or
    each rank re-weights its own batch and DDP's averaging leaves a mixture of
    per-rank distributions. The caller reduces the counts exactly where
    `stream_mean`'s ``global_counts`` are reduced and passes them here.

    Args:
        nonempty: ``(B,)`` boolean, true on the rows to give ``positive_share``.
        valid: ``(B,)`` boolean or 0/1 row mask; invalid rows weigh zero.
        positive_share: Loss mass the non-empty rows carry, in ``(0, 1)``.
        global_nonempty: Valid non-empty rows summed over ranks; rank-local when
            omitted.
        global_empty: Valid empty rows summed over ranks; rank-local when omitted.

    Returns:
        ``(B,)`` float32 weights, zero on invalid rows and uniform whenever one
        of the two classes is empty.

    Raises:
        ValueError: If the two row tensors disagree in shape, ``positive_share``
            lies outside ``(0, 1)``, or exactly one global count is given.
    """
    if nonempty.shape != valid.shape:
        raise ValueError(
            f"nonempty {tuple(nonempty.shape)} and valid {tuple(valid.shape)} must match"
        )
    if not 0.0 < positive_share < 1.0:
        raise ValueError(f"positive_share must lie in (0, 1), got {positive_share}")
    if (global_nonempty is None) != (global_empty is None):
        raise ValueError("global_nonempty and global_empty are given together or not at all")
    valid_row = (valid > 0).to(dtype=torch.float32)
    nonempty_row = (nonempty > 0).to(dtype=torch.float32) * valid_row
    if global_nonempty is None or global_empty is None:
        count_nonempty = float(nonempty_row.sum().item())
        count_empty = float(valid_row.sum().item()) - count_nonempty
    else:
        count_nonempty, count_empty = float(global_nonempty), float(global_empty)
    if count_nonempty <= 0.0 or count_empty <= 0.0:
        # One class alone carries the whole loss already; re-weighting it would
        # only rescale the mean, which is not this intervention.
        return valid_row
    total = count_nonempty + count_empty
    weight_nonempty = positive_share * total / count_nonempty
    weight_empty = (1.0 - positive_share) * total / count_empty
    return valid_row * (nonempty_row * (weight_nonempty - weight_empty) + weight_empty)


def closure_balanced_row_weights(
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    positive_share: float,
    global_nonempty: float | None = None,
    global_empty: float | None = None,
) -> torch.Tensor:
    """`balanced_row_weights` over `closure_nonempty` of a compiled template.

    Args:
        target: ``(B, 96)`` compiled edge weights.
        valid: ``(B,)`` boolean or 0/1 row mask.
        positive_share: Loss mass the closure-non-empty rows carry, in ``(0, 1)``.
        global_nonempty: Valid non-empty rows summed over ranks, or ``None``.
        global_empty: Valid empty rows summed over ranks, or ``None``.

    Returns:
        ``(B,)`` float32 weights summing to the valid-row count.
    """
    return balanced_row_weights(
        closure_nonempty(target),
        valid,
        positive_share=positive_share,
        global_nonempty=global_nonempty,
        global_empty=global_empty,
    )


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
    structural stream's pairs are striped over the ranks per token boundary, so a
    bucket with fewer pairs than ranks leaves a rank without any of them.

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

    Under DDP the per-stream mean has to be a *global* mean. Structural pairs are
    striped over the ranks, so ranks hold counts that differ by up to one pair per
    boundary and a rank can hold no structural row at all; dividing by rank-local
    counts and letting DDP average
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
