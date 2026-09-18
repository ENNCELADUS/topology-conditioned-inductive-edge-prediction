"""Fixed-set, generator-only fit of the Stage II motif generator (wave-3 fix G1).

Pilot B reads a *trained* Stage II prefix end to end, which costs a GPU-day per
arm. The question wave 2 left open is narrower and can be answered in minutes:
given the frozen trunk's residue states, can a freshly initialised
`MotifGenerator` fit the compiled motif graph at all, and does it emit a
row-dependent graph while doing it?

The harness therefore caches the generator's own inputs once -- the frozen
trunk's encoded residue states of a fixed, stratified row set -- and then trains
nothing but a fresh generator on ``L_G`` over those cached inputs, in fp32, for a
few hundred steps. Four groups share the same rows, the same seeds and the same
schedule and differ only in the two wave-3 generator keys:

===========  ========================  ==========================
group        ``slot_read``             ``head_output_init_std``
===========  ========================  ==========================
baseline     ``bare``                  ``1e-3`` (wave-1/2 behaviour)
residual     ``residual_block``        ``1e-3``
head_gain    ``bare``                  ``1e-2``
combined     ``residual_block``        ``1e-2``
===========  ========================  ==========================

Rows are stratified by the true witness count, because the corpus is dominated
by rows with no closure edge at all and an unstratified reading cannot tell a
generator that fits the closure family from one that has learnt to emit zero.

Nothing here reimplements the trunk, the loss or the row selection: the split,
the node-disjoint sides, the loss weights, the fitted-constant comparator and
the transplant null all come from `src.experiments.motif_pilot_b`, and the input
capture rides on the formal packed scoring pass.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from torch import nn

import src.score_universe as su
from src.data.motif_template import (
    C_SLOTS,
    CLOSURE_U,
    EDGE_ENDPOINTS,
    EDGE_TYPES,
    L_SLOTS,
    N_EDGE_TYPES,
    N_EDGES,
    R_SLOTS,
    MotifTemplateTable,
    count_statistics,
    template_statistics,
)
from src.experiments.motif_pilot_b import (
    GraphLossWeights,
    fit_constant_template,
    gate_logit_summary,
    mass_report,
    predicted_bank,
    rows_inside,
    split_training_nodes,
    transplant_permutations,
    transplant_report,
)
from src.model.egostitch.classifier.layers import (
    _build_padding_mask,
    inner_token_mask,
    masked_mean,
)
from src.model.egostitch.classifier.motif_prompt import (
    EDGE_TYPE_NAMES,
    MotifGenerator,
    MotifPromptConfig,
    V3_1MotifPrompt,
)

logger = logging.getLogger("motif_generator_fit")

Pair = tuple[str, str]

#: ``(slot_read, head_output_init_std)`` per comparison group.
GROUPS: dict[str, tuple[str, float]] = {
    "baseline": ("bare", 1e-3),
    "residual": ("residual_block", 1e-3),
    "head_gain": ("bare", 1e-2),
    "combined": ("residual_block", 1e-2),
}
#: Witness-count strata, by the number of non-zero ``CLOSURE_U`` slots.
STRATUM_NAMES: tuple[str, ...] = ("0", "1-2", "3-7", "8")
#: Target share of each stratum: 40% empty, the rest spread over the three
#: non-empty strata as evenly as the corpus allows.
STRATUM_SHARES: tuple[float, ...] = (0.4, 0.2, 0.2, 0.2)
#: Candidate rows drawn (and compiled) per requested row, before stratification.
CANDIDATE_MULTIPLIER = 12
#: Floor on the candidate pool, so a tiny request still sees every stratum.
MIN_CANDIDATE_POOL = 5_000
#: Rows the D(S) chain is measured over per universe.
DISPERSION_ROWS = 512
#: Roles the slot-stage dispersion is reported for.
ROLE_SLOTS: dict[str, tuple[int, ...]] = {"C": C_SLOTS, "L": L_SLOTS, "R": R_SLOTS}
#: The three linear-warm-up steps' share of the schedule.
WARMUP_STEPS = 100
#: A row counts as having identical closure slots below this range.
IDENTICAL_SLOT_TOLERANCE = 1e-3
_EDGE_ROWS = torch.as_tensor([i for i, _ in EDGE_ENDPOINTS], dtype=torch.long)
_EDGE_COLS = torch.as_tensor([j for _, j in EDGE_ENDPOINTS], dtype=torch.long)


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------


def witness_counts(target: torch.Tensor) -> NDArray[np.int64]:
    """Return each row's true witness count, the non-zero ``CLOSURE_U`` slots.

    A common neighbour of ``u`` and ``v`` keeps degree at least two after the
    queried edge is removed, so its compiled weight is strictly positive and the
    count is exactly ``min(|N(u) & N(v)|, 8)``.

    Args:
        target: ``(n, 96)`` compiled weights.

    Returns:
        ``(n,)`` witness counts in ``[0, 8]``.
    """
    return (target[:, CLOSURE_U] > 0).sum(dim=1).cpu().numpy().astype(np.int64)


def strata_of(counts: NDArray[np.int64]) -> NDArray[np.int64]:
    """Map witness counts onto `STRATUM_NAMES` indices.

    Args:
        counts: ``(n,)`` witness counts.

    Returns:
        ``(n,)`` stratum indices ``0`` (none), ``1`` (1-2), ``2`` (3-7), ``3`` (8).
    """
    value = np.asarray(counts, dtype=np.int64)
    out = np.zeros_like(value)
    out[(value >= 1) & (value <= 2)] = 1
    out[(value >= 3) & (value <= 7)] = 2
    out[value >= 8] = 3
    return out


def allocate_strata(available: Sequence[int], total: int) -> list[int]:
    """Split ``total`` rows over the strata at `STRATUM_SHARES`, capped by supply.

    A stratum that cannot supply its share gives the shortfall back, and the
    remainder is handed to whichever strata still have spare capacity, in
    stratum order, until nothing moves. The achieved composition is recorded by
    the caller rather than enforced, because the non-empty tail is genuinely
    thin in this corpus.

    Args:
        available: Rows available per stratum.
        total: Rows wanted overall.

    Returns:
        Rows to draw per stratum; the sum is ``min(total, sum(available))``.

    Raises:
        ValueError: If ``available`` is not one count per stratum or ``total``
            is negative.
    """
    if len(available) != len(STRATUM_SHARES):
        raise ValueError(f"expected {len(STRATUM_SHARES)} strata, got {len(available)}")
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    supply = [max(int(value), 0) for value in available]
    want = [min(supply[i], int(round(total * share))) for i, share in enumerate(STRATUM_SHARES)]
    target = min(total, sum(supply))
    while sum(want) < target:
        spare = [i for i in range(len(want)) if want[i] < supply[i]]
        if not spare:
            break
        deficit = target - sum(want)
        moved = 0
        for index in spare:
            if moved >= deficit:
                break
            step = min(supply[index] - want[index], max((deficit - moved) // len(spare), 1))
            want[index] += step
            moved += step
    while sum(want) > target:
        for index in range(len(want) - 1, -1, -1):
            if sum(want) <= target:
                break
            if want[index] > 0:
                want[index] -= 1
    return want


def stratified_draw(
    target: torch.Tensor, *, total: int, seed: int
) -> tuple[NDArray[np.int64], dict[str, int]]:
    """Draw a witness-count-stratified row subset of a compiled candidate pool.

    Args:
        target: ``(n, 96)`` compiled weights of the candidate pool.
        total: Rows wanted.
        seed: Draw seed.

    Returns:
        ``(indices, achieved)`` with the sorted row indices into the pool and the
        achieved count per stratum name.
    """
    strata = strata_of(witness_counts(target))
    generator = np.random.default_rng(seed)
    members = [np.flatnonzero(strata == index) for index in range(len(STRATUM_NAMES))]
    want = allocate_strata([int(group.size) for group in members], total)
    picked: list[NDArray[np.int64]] = []
    achieved: dict[str, int] = {}
    for index, name in enumerate(STRATUM_NAMES):
        group = members[index]
        take = min(want[index], int(group.size))
        chosen = generator.choice(group, size=take, replace=False) if take else group[:0]
        picked.append(np.asarray(chosen, dtype=np.int64))
        achieved[name] = int(take)
    indices = np.sort(np.concatenate(picked)) if picked else np.zeros(0, dtype=np.int64)
    return indices.astype(np.int64), achieved


# ---------------------------------------------------------------------------
# Cached generator inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowInputs:
    """One row's frozen trunk states, trimmed to the true residue lengths.

    Stored as the raw bf16 bit pattern in an ``int16`` numpy array: the packed
    scorer encodes under bf16 autocast and widens, so the fp32 states it hands
    the generator are exactly bf16-representable and the round trip is lossless,
    at half the host memory of fp32.

    Attributes:
        u: ``(L_u, d_model)`` bf16 bits of the first endpoint.
        v: ``(L_v, d_model)`` bf16 bits of the second endpoint.
    """

    u: NDArray[np.int16]
    v: NDArray[np.int16]


def _to_bits(states: torch.Tensor) -> NDArray[np.int16]:
    """Copy one endpoint's states off the device as bf16 bits."""
    return cast(
        NDArray[np.int16],
        states.detach().to("cpu", dtype=torch.bfloat16).view(torch.int16).numpy().copy(),
    )


def _from_bits(bits: NDArray[np.int16]) -> torch.Tensor:
    """Rebuild one endpoint's bf16 states from their bit pattern."""
    return torch.from_numpy(bits).view(torch.bfloat16)


@contextmanager
def capture_generator_inputs(model: V3_1MotifPrompt) -> Iterator[dict[int, RowInputs]]:
    """Record the generator's own inputs, row by row, during a scoring pass.

    `src.score_universe` hands the generator one padded batch at a time and has
    no hook that names the rows in it, so the batch's row indices are taken from
    the transplant-bank builder it already routes every Stage II prediction
    through and the states from a forward pre-hook on the generator itself. The
    two fire in lockstep, one call each per batch.

    Args:
        model: The Stage II checkpoint whose generator is about to run.

    Yields:
        A mapping from row index to that row's trimmed states; it is filled
        during the pass and complete when the context exits.

    Raises:
        RuntimeError: If the checkpoint has no generator to hook.
    """
    if model.generator is None:
        raise RuntimeError("capturing generator inputs needs a Stage II checkpoint")
    store: dict[int, RowInputs] = {}
    pending: dict[str, list[int]] = {"indices": []}

    def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        """Record the rows of the batch the generator has just been handed."""
        encoded_u, encoded_v, lengths_u, lengths_v = args[:4]
        for position, row in enumerate(pending["indices"]):
            store[int(row)] = RowInputs(
                u=_to_bits(encoded_u[position, : int(lengths_u[position])]),
                v=_to_bits(encoded_v[position, : int(lengths_v[position])]),
            )

    handle = model.generator.register_forward_pre_hook(hook)
    original = su._motif_source_weights

    def spy(
        source_batches: Sequence[Sequence[int]],
        predict_batch: Callable[[Sequence[int]], torch.Tensor],
        *,
        num_rows: int,
    ) -> torch.Tensor:
        """Name each batch's rows before the generator sees it."""

        def wrapped(batch_indices: Sequence[int]) -> torch.Tensor:
            pending["indices"] = [int(index) for index in batch_indices]
            return predict_batch(batch_indices)

        return original(source_batches, wrapped, num_rows=num_rows)

    su._motif_source_weights = spy
    try:
        yield store
    finally:
        su._motif_source_weights = original
        handle.remove()


def _pad_side(sides: Sequence[NDArray[np.int16]], device: torch.device) -> torch.Tensor:
    """Pad one endpoint of a batch onto the device in fp32."""
    width = int(sides[0].shape[1])
    longest = max(int(side.shape[0]) for side in sides)
    padded = torch.zeros(len(sides), longest, width, dtype=torch.float32)
    for index, side in enumerate(sides):
        states = _from_bits(side)
        padded[index, : states.size(0)] = states.float()
    return padded.to(device)


def collate(
    rows: Sequence[RowInputs], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad one batch of cached rows back onto the device in fp32.

    Args:
        rows: The batch's cached states.
        device: Compute device.

    Returns:
        ``(encoded_u, encoded_v, lengths_u, lengths_v)``.

    Raises:
        ValueError: On an empty batch.
    """
    if not rows:
        raise ValueError("a batch needs at least one row")
    lengths_u = torch.as_tensor([row.u.shape[0] for row in rows], dtype=torch.long)
    lengths_v = torch.as_tensor([row.v.shape[0] for row in rows], dtype=torch.long)
    return (
        _pad_side([row.u for row in rows], device),
        _pad_side([row.v for row in rows], device),
        lengths_u.to(device),
        lengths_v.to(device),
    )


# ---------------------------------------------------------------------------
# Stage-by-stage replay and dispersion
# ---------------------------------------------------------------------------


def replay(
    generator: MotifGenerator,
    encoded_u: torch.Tensor,
    encoded_v: torch.Tensor,
    lengths_u: torch.Tensor,
    lengths_v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run `MotifGenerator.forward` stage by stage, keeping every intermediate.

    The replay calls the generator's own submodules in the generator's own
    order; `assert_replay_matches` checks that it reproduces ``forward`` exactly,
    so a future change to the generator that this function does not follow is
    caught rather than silently measured.

    Args:
        generator: The generator to replay; ``gate_mode`` must be ``learned``.
        encoded_u: ``(B, L_u, d_model)`` states of the first endpoint.
        encoded_v: ``(B, L_v, d_model)`` states of the second endpoint.
        lengths_u: True residue lengths of ``u``.
        lengths_v: True residue lengths of ``v``.

    Returns:
        The named intermediates, ending in ``weights``.

    Raises:
        ValueError: On a ``gate_mode`` the replay does not cover.
    """
    if generator.cfg.gate_mode != "learned":
        raise ValueError(f"the replay covers gate_mode 'learned', not {generator.cfg.gate_mode!r}")
    pad_u = _build_padding_mask(lengths_u, encoded_u.size(1))
    pad_v = _build_padding_mask(lengths_v, encoded_v.size(1))
    state_u = generator.residue_proj(encoded_u)
    state_v = generator.residue_proj(encoded_v)
    stages: dict[str, torch.Tensor] = {
        "read_bridge_u": generator._read(
            generator.bridge_queries, state_u, pad_u, generator.bridge_read
        ),
        "read_bridge_v": generator._read(
            generator.bridge_queries, state_v, pad_v, generator.bridge_read
        ),
        "read_witness_u": generator._read(
            generator.witness_queries, state_u, pad_u, generator.witness_read
        ),
        "read_witness_v": generator._read(
            generator.witness_queries, state_v, pad_v, generator.witness_read
        ),
    }
    witness_u, witness_v = stages["read_witness_u"], stages["read_witness_v"]
    stages["witness_mix"] = generator.witness_mix(
        torch.cat([witness_u + witness_v, (witness_u - witness_v).abs()], dim=-1)
    )
    pooled_u = generator.endpoint_proj(
        masked_mean(state_u, inner_token_mask(x=state_u, padding_mask=pad_u))
    )
    pooled_v = generator.endpoint_proj(
        masked_mean(state_v, inner_token_mask(x=state_v, padding_mask=pad_v))
    )
    h = torch.cat(
        [
            pooled_u.unsqueeze(1),
            pooled_v.unsqueeze(1),
            stages["witness_mix"],
            stages["read_bridge_u"],
            stages["read_bridge_v"],
        ],
        dim=1,
    )
    for index, layer in enumerate(generator.message_layers):
        h = layer(h, generator.incidence)
        stages[f"message_{index + 1}"] = h
    rows = _EDGE_ROWS.to(h.device)
    cols = _EDGE_COLS.to(h.device)
    with torch.autocast(device_type=h.device.type, enabled=False):
        h_i = h[:, rows].float()
        h_j = h[:, cols].float()
        features = torch.cat([h_i + h_j, (h_i - h_j).abs()], dim=-1)
        logits = features.new_zeros(encoded_u.size(0), N_EDGES)
        for edge_type, head in enumerate(generator.heads):
            mask = _type_mask(edge_type).to(h.device)
            logits[:, mask] = head(features[:, mask]).squeeze(-1)
    stages["head_features"] = features
    stages["logits"] = logits
    stages["weights"] = torch.sigmoid(logits)
    return stages


def assert_replay_matches(
    generator: MotifGenerator,
    encoded_u: torch.Tensor,
    encoded_v: torch.Tensor,
    lengths_u: torch.Tensor,
    lengths_v: torch.Tensor,
    *,
    tolerance: float = 1e-5,
) -> None:
    """Check the stage-by-stage replay against ``forward`` on one batch.

    Args:
        generator: The generator to check.
        encoded_u: ``(B, L_u, d_model)`` states of the first endpoint.
        encoded_v: ``(B, L_v, d_model)`` states of the second endpoint.
        lengths_u: True residue lengths of ``u``.
        lengths_v: True residue lengths of ``v``.
        tolerance: Maximum allowed absolute difference.

    Raises:
        ValueError: If the replay and ``forward`` disagree.
    """
    with torch.no_grad():
        direct = generator(encoded_u, encoded_v, lengths_u, lengths_v)
        stages = replay(generator, encoded_u, encoded_v, lengths_u, lengths_v)
    gap = float((direct - stages["weights"]).abs().max())
    if gap > tolerance:
        raise ValueError(f"the stage replay disagrees with forward by {gap:.3e}")


def slot_dispersion(block: torch.Tensor) -> float:
    """Return ``D(S) = sqrt(mean_k ||s_k - s_bar||^2 / d)`` of one slot block.

    Args:
        block: ``(B, K, d)`` slot states, or ``(B, K)`` scalars (``d = 1``).

    Returns:
        The batch-mean per-row spread across the ``K`` slots.
    """
    value = block.detach().float()
    if value.dim() == 2:
        value = value.unsqueeze(-1)
    centred = value - value.mean(dim=1, keepdim=True)
    width = float(value.size(-1))
    return float((centred.pow(2).sum(dim=-1) / width).mean().sqrt())


def dispersion_chain(stages: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Return ``D(S)`` along the generator's whole chain.

    Slot stages are read per role -- the eight closure slots ``C`` and the two
    bridge sides ``L`` and ``R`` -- and the edge stages per edge type, which is
    the level the heads and the loss act at.

    Args:
        stages: The `replay` intermediates.

    Returns:
        One mapping of role/type name to ``D(S)`` per stage.
    """
    chain: dict[str, dict[str, float]] = {
        "read_bridge_u": {"L": slot_dispersion(stages["read_bridge_u"])},
        "read_bridge_v": {"R": slot_dispersion(stages["read_bridge_v"])},
        "read_witness_u": {"C": slot_dispersion(stages["read_witness_u"])},
        "read_witness_v": {"C": slot_dispersion(stages["read_witness_v"])},
        "witness_mix": {"C": slot_dispersion(stages["witness_mix"])},
    }
    for name in ("message_1", "message_2"):
        chain[name] = {
            role: slot_dispersion(stages[name][:, list(slots)])
            for role, slots in ROLE_SLOTS.items()
        }
    for name in ("head_features", "logits", "weights"):
        chain[name] = {
            EDGE_TYPE_NAMES[edge_type]: slot_dispersion(stages[name][:, _type_mask(edge_type)])
            for edge_type in range(N_EDGE_TYPES)
        }
    return chain


def _type_mask(edge_type: int) -> torch.Tensor:
    """Return the ``(96,)`` boolean mask of one edge type."""
    return torch.as_tensor([kind == edge_type for kind in EDGE_TYPES])


def identical_closure_fraction(weights: torch.Tensor) -> float:
    """Fraction of rows whose eight closure slots are identical on both sides.

    Args:
        weights: ``(n, 96)`` predicted weights.

    Returns:
        The fraction whose within-side closure range is below
        `IDENTICAL_SLOT_TOLERANCE` on both sides.
    """
    value = weights.detach().float()
    left = value[:, 0:8]
    right = value[:, 8:16]
    flat = (left.max(dim=1).values - left.min(dim=1).values < IDENTICAL_SLOT_TOLERANCE) & (
        right.max(dim=1).values - right.min(dim=1).values < IDENTICAL_SLOT_TOLERANCE
    )
    return float(flat.float().mean())


def wedge_mass_auroc(predicted: torch.Tensor, target: torch.Tensor) -> float | None:
    """AUROC of the predicted wedge mass at separating non-empty from empty rows.

    Args:
        predicted: ``(n, 96)`` predicted weights.
        target: ``(n, 96)`` compiled weights.

    Returns:
        The AUROC, or ``None`` when one class is missing.
    """
    labels = (target[:, CLOSURE_U] > 0).any(dim=1).cpu().numpy().astype(np.int8)
    if labels.min() == labels.max():
        return None
    scores = count_statistics(predicted.float())["wedge_mass"].cpu().numpy()
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Row sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowSet:
    """One stratified universe of rows and everything measured on it.

    Attributes:
        name: Universe name.
        pairs: The rows, in report order.
        target: ``(n, 96)`` compiled weights.
        strata: ``(n,)`` stratum indices.
        composition: Achieved rows per stratum name.
    """

    name: str
    pairs: tuple[Pair, ...]
    target: torch.Tensor
    strata: NDArray[np.int64]
    composition: dict[str, int]


def build_row_set(
    name: str, pairs: Sequence[Pair], table: MotifTemplateTable, *, total: int, seed: int
) -> RowSet:
    """Compile a candidate pool's templates and draw a stratified subset.

    Args:
        name: Universe name.
        pairs: The candidate pool, already restricted to nonself rows.
        table: The universe's compiled-template source.
        total: Rows wanted.
        seed: Draw seed.

    Returns:
        The stratified row set.
    """
    compiled = torch.from_numpy(table.weights(list(pairs)))
    indices, composition = stratified_draw(compiled, total=total, seed=seed)
    logger.info(
        "%s: %d of %d candidates, composition %s", name, indices.size, len(pairs), composition
    )
    target = compiled.index_select(0, torch.from_numpy(indices)).float()
    return RowSet(
        name=name,
        pairs=tuple(pairs[int(index)] for index in indices),
        target=target,
        strata=strata_of(witness_counts(target)),
        composition=composition,
    )


def _pool_size(total: int) -> int:
    """Candidate rows to compile for a request of ``total`` rows."""
    return max(total * CANDIDATE_MULTIPLIER, MIN_CANDIDATE_POOL)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_generator(
    cfg: MotifPromptConfig, *, d_model: int, group: str, fit_target: torch.Tensor, seed: int
) -> MotifGenerator:
    """Build and initialise one group's fresh generator.

    Args:
        cfg: The checkpoint's own motif-prompt block.
        d_model: The frozen trunk's width.
        group: A `GROUPS` key.
        fit_target: ``(n, 96)`` compiled weights of the fit rows, which the
            output biases are initialised from exactly as the trainer does.
        seed: Initialisation seed.

    Returns:
        The generator, biases installed.

    Raises:
        ValueError: On an unknown group.
    """
    if group not in GROUPS:
        raise ValueError(f"group must be one of {sorted(GROUPS)}")
    slot_read, head_std = GROUPS[group]
    group_cfg = replace(cfg, slot_read=slot_read, head_output_init_std=head_std)
    torch.manual_seed(seed)
    generator = MotifGenerator(d_model, group_cfg)
    generator.init_biases(
        template_statistics(np.asarray(fit_target.numpy(), dtype=np.float32)),
        closure_bias_init=group_cfg.closure_bias_init,
    )
    return generator


def _evaluate(
    generator: MotifGenerator,
    rows: RowSet,
    cache: Mapping[int, RowInputs],
    index: Mapping[Pair, int],
    *,
    device: torch.device,
    chunk: int,
) -> torch.Tensor:
    """Predict every row of one universe, in eval mode and without gradients.

    Args:
        generator: The generator to read.
        rows: The universe.
        cache: The captured inputs, by row index of the capture pass.
        index: Row index of every pair in the capture pass.
        device: Compute device.
        chunk: Rows per forward.

    Returns:
        ``(n, 96)`` predictions on the CPU, in float32.
    """
    was_training = generator.training
    generator.eval()
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(rows.pairs), chunk):
            batch = [cache[index[pair]] for pair in rows.pairs[start : start + chunk]]
            encoded_u, encoded_v, lengths_u, lengths_v = collate(batch, device)
            out.append(generator(encoded_u, encoded_v, lengths_u, lengths_v).float().cpu())
    generator.train(was_training)
    return torch.cat(out, dim=0)


def _per_stratum(
    predicted: torch.Tensor, rows: RowSet, weights: GraphLossWeights
) -> dict[str, dict[str, float]]:
    """Mean ``L_G`` and row count per stratum."""
    out: dict[str, dict[str, float]] = {}
    for stratum, name in enumerate(STRATUM_NAMES):
        keep = torch.from_numpy(rows.strata == stratum)
        count = int(keep.sum())
        out[name] = {
            "rows": float(count),
            "L_G": (weights.mean(predicted[keep], rows.target[keep]) if count else float("nan")),
        }
    return out


def _dispersion_for(
    generator: MotifGenerator,
    rows: RowSet,
    cache: Mapping[int, RowInputs],
    index: Mapping[Pair, int],
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Measure the D(S) chain on the universe's first `DISPERSION_ROWS` rows."""
    batch = [cache[index[pair]] for pair in rows.pairs[:DISPERSION_ROWS]]
    encoded_u, encoded_v, lengths_u, lengths_v = collate(batch, device)
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        stages = replay(generator, encoded_u, encoded_v, lengths_u, lengths_v)
    generator.train(was_training)
    return dispersion_chain(stages)


def _lr_lambda(step: int) -> float:
    """Linear warm-up over `WARMUP_STEPS`, then a constant learning rate."""
    return min(1.0, (step + 1) / WARMUP_STEPS)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_fit(
    *,
    checkpoint: Path,
    pack_dir: Path,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    group: str,
    fit_rows: int,
    heldout_rows: int,
    val_rows: int,
    steps: int,
    batch_rows: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    eval_every: int,
    seed: int,
    device: torch.device,
    amp: str,
    token_budget: int,
    eval_chunk: int,
    constant_steps: int,
    permutations: int,
) -> dict[str, object]:
    """Fit one group's generator on the cached fixed row set and read it.

    Args:
        checkpoint: A Stage II checkpoint, read only for its config, its frozen
            trunk and the capture of the generator's inputs.
        pack_dir: The packed feature directory.
        data_root: Benchmark data root.
        strategy: Split strategy.
        output_dir: Destination for ``fit.json``, ``fit.md`` and the state dict.
        group: A `GROUPS` key.
        fit_rows: Training rows.
        heldout_rows: Node-disjoint held-out training rows.
        val_rows: ``val_cls`` rows.
        steps: Optimiser steps.
        batch_rows: Rows per step.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Gradient-norm clip.
        eval_every: Steps between evaluations.
        seed: Master seed.
        device: Compute device.
        amp: Encoder autocast mode of the capture pass.
        token_budget: Capture-pass token budget.
        eval_chunk: Rows per evaluation forward.
        constant_steps: Adam steps for the fitted constant comparator.
        permutations: Transplant permutations.

    Returns:
        The full ``fit.json`` payload.

    Raises:
        ValueError: If the checkpoint is not a full-family, learned-gate Stage II
            motif-prompt checkpoint.
    """
    torch.manual_seed(seed)
    loaded, family, checkpoint_id = su._load_checkpoint(checkpoint)
    if family != su.MOTIF_PROMPT_FAMILY:
        raise ValueError(f"{checkpoint}: model_family {family!r}, expected a motif prompt")
    model = cast(V3_1MotifPrompt, loaded)
    if model.cfg.stage != "two":
        raise ValueError(f"{checkpoint}: the fit harness reads a Stage II checkpoint")
    if model.cfg.gate_mode != "learned":
        raise ValueError(f"{checkpoint}: the fit harness reads gate_mode 'learned'")
    if len(model.cfg.families) != 2:
        raise ValueError(f"{checkpoint}: the fit harness reads a full-family generator")
    model = model.to(device).eval()
    loss = GraphLossWeights.from_config(model.cfg)

    split = su._load_val_region_split(data_root, strategy)
    fit_nodes, heldout_nodes = split_training_nodes(sorted(split.train_nodes), seed=seed)
    training_table = MotifTemplateTable(split.build_training_graph())
    positives, negatives = sorted(split.training_positives), list(split.training_negatives)
    fit_pool, _ = rows_inside(
        positives, negatives, fit_nodes, limit=_pool_size(fit_rows), seed=seed + 1
    )
    heldout_pool, _ = rows_inside(
        positives, negatives, heldout_nodes, limit=_pool_size(heldout_rows), seed=seed + 2
    )
    fit_set = build_row_set("fit", fit_pool, training_table, total=fit_rows, seed=seed + 11)
    heldout_set = build_row_set(
        "heldout", heldout_pool, training_table, total=heldout_rows, seed=seed + 12
    )

    val_pairs, _ = su._resolve_pairs("val_cls", data_root, strategy)
    val_nonself = [(u, v) for u, v in val_pairs if u != v]
    pool = min(len(val_nonself), _pool_size(val_rows))
    chosen = np.sort(
        np.random.default_rng(seed + 3).choice(len(val_nonself), size=pool, replace=False)
    )
    val_table = MotifTemplateTable(
        su._oracle_truth_graph_for_scoring("val_cls", data_root, strategy)
    )
    val_set = build_row_set(
        "val_cls",
        [val_nonself[int(i)] for i in chosen],
        val_table,
        total=val_rows,
        seed=seed + 13,
    )

    universes = (fit_set, heldout_set, val_set)
    capture_pairs: list[Pair] = []
    seen: set[Pair] = set()
    for rows in universes:
        for pair in rows.pairs:
            if pair not in seen:
                seen.add(pair)
                capture_pairs.append(pair)
    index = {pair: position for position, pair in enumerate(capture_pairs)}
    logger.info("capturing generator inputs for %d distinct rows", len(capture_pairs))
    with capture_generator_inputs(model) as cache:
        _bank, _logits = predicted_bank(
            model, capture_pairs, pack_dir, device=device, amp=amp, token_budget=token_budget
        )
    del _bank, _logits
    missing = [pair for pair in capture_pairs if index[pair] not in cache]
    if missing:
        raise ValueError(f"{len(missing)} rows were never handed to the generator")
    d_model = int(model.d_model)
    closure_bias_init = model.cfg.closure_bias_init
    cfg = model.cfg
    del model, loaded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("trunk released; cached %d rows", len(cache))

    generator = build_generator(
        cfg, d_model=d_model, group=group, fit_target=fit_set.target, seed=seed + 21
    ).to(device)
    sample = [cache[index[pair]] for pair in fit_set.pairs[:4]]
    sample_u, sample_v, sample_len_u, sample_len_v = collate(sample, device)
    assert_replay_matches(generator, sample_u, sample_v, sample_len_u, sample_len_v)

    fitted_constant = fit_constant_template(
        fit_set.target, loss, steps=constant_steps, device=device
    )
    mean_template = torch.from_numpy(
        template_statistics(np.asarray(fit_set.target.numpy(), dtype=np.float32)).mean
    ).float()

    optimiser = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, _lr_lambda)
    sampler = np.random.default_rng(seed + 31)
    curve: list[dict[str, object]] = []
    generator.train()
    for step in range(1, steps + 1):
        picks = sampler.integers(0, len(fit_set.pairs), size=min(batch_rows, len(fit_set.pairs)))
        batch = [cache[index[fit_set.pairs[int(i)]]] for i in picks]
        encoded_u, encoded_v, lengths_u, lengths_v = collate(batch, device)
        target = fit_set.target.index_select(0, torch.from_numpy(picks.astype(np.int64))).to(device)
        predicted = generator(encoded_u, encoded_v, lengths_u, lengths_v)
        value = loss.rows(predicted, target).mean()
        optimiser.zero_grad(set_to_none=True)
        value.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
        optimiser.step()
        schedule.step()
        if step % eval_every == 0 or step == steps:
            entry: dict[str, object] = {"step": step, "train_batch_L_G": float(value)}
            for rows in universes:
                prediction = _evaluate(
                    generator, rows, cache, index, device=device, chunk=eval_chunk
                )
                entry[rows.name] = loss.mean(prediction, rows.target)
            curve.append(entry)
            logger.info("step %d: %s", step, entry)

    report = _final_report(
        generator=generator,
        universes=universes,
        cache=cache,
        index=index,
        loss=loss,
        fitted_constant=fitted_constant,
        mean_template=mean_template,
        device=device,
        eval_chunk=eval_chunk,
        permutations=permutations,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    group_cfg = replace(cfg, slot_read=GROUPS[group][0], head_output_init_std=GROUPS[group][1])
    torch.save(
        {"group": group, "cfg": group_cfg.to_dict(), "state": generator.state_dict()},
        output_dir / "generator_state.pt",
    )
    return {
        "harness": "motif_generator_fit",
        "written_at": datetime.now(UTC).isoformat(),
        "group": group,
        "slot_read": GROUPS[group][0],
        "head_output_init_std": GROUPS[group][1],
        "checkpoint": str(checkpoint),
        "checkpoint_id": checkpoint_id,
        "strategy": strategy,
        "seed": seed,
        "closure_bias_init": closure_bias_init,
        "loss_weights": {
            "beta_p": loss.beta_p,
            "beta_q": loss.beta_q,
            "beta_a": loss.beta_a,
            "beta_i": loss.beta_i,
            "beta_c": loss.beta_c,
            "huber_delta": loss.huber_delta,
        },
        "optimisation": {
            "steps": steps,
            "batch_rows": batch_rows,
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "warmup_steps": WARMUP_STEPS,
        },
        "rows": {
            rows.name: {"rows": len(rows.pairs), "composition": rows.composition}
            for rows in universes
        },
        "node_disjoint": {"fit_nodes": len(fit_nodes), "heldout_nodes": len(heldout_nodes)},
        "learning_curve": curve,
        "final": report,
    }


def _final_report(
    *,
    generator: MotifGenerator,
    universes: Sequence[RowSet],
    cache: Mapping[int, RowInputs],
    index: Mapping[Pair, int],
    loss: GraphLossWeights,
    fitted_constant: torch.Tensor,
    mean_template: torch.Tensor,
    device: torch.device,
    eval_chunk: int,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    """Read the trained generator on every universe.

    Args:
        generator: The trained generator.
        universes: The fit, held-out and ``val_cls`` row sets.
        cache: The captured inputs.
        index: Row index of every pair in the capture pass.
        loss: The checkpoint's own ``L_G``.
        fitted_constant: The asymmetric constant fitted on the fit rows.
        mean_template: The fit rows' mean adjacency.
        device: Compute device.
        eval_chunk: Rows per evaluation forward.
        permutations: Transplant permutations.
        seed: Master seed.

    Returns:
        One payload per universe.
    """
    out: dict[str, object] = {}
    for rows in universes:
        predicted = _evaluate(generator, rows, cache, index, device=device, chunk=eval_chunk)
        constant_rows = fitted_constant.unsqueeze(0).expand_as(predicted)
        mean_rows = mean_template.unsqueeze(0).expand_as(predicted)
        transplants: dict[str, object] = {
            "global": transplant_report(
                predicted,
                rows.target,
                loss,
                transplant_permutations(len(rows.pairs), count=permutations, seed=seed + 100),
            )
        }
        for stratum, name in enumerate(STRATUM_NAMES):
            keep = torch.from_numpy(rows.strata == stratum)
            count = int(keep.sum())
            transplants[name] = (
                transplant_report(
                    predicted[keep],
                    rows.target[keep],
                    loss,
                    transplant_permutations(count, count=permutations, seed=seed + 200 + stratum),
                )
                if count >= 2
                else {"rows": count, "note": "too few rows to permute"}
            )
        out[rows.name] = {
            "rows": len(rows.pairs),
            "L_G": {
                "generator": loss.mean(predicted, rows.target),
                "fitted_constant_template": loss.mean(constant_rows, rows.target),
                "mean_template": loss.mean(mean_rows, rows.target),
            },
            "per_stratum_L_G": _per_stratum(predicted, rows, loss),
            "dispersion": _dispersion_for(generator, rows, cache, index, device=device),
            "identical_closure_row_fraction": identical_closure_fraction(predicted),
            "gate_logits": gate_logit_summary(predicted),
            "wedge_mass_auroc": wedge_mass_auroc(predicted, rows.target),
            "masses": _masses_per_stratum(predicted, rows),
            "transplant": transplants,
        }
    return out


def _masses_per_stratum(predicted: torch.Tensor, rows: RowSet) -> dict[str, object]:
    """Predicted-vs-true wedge and bridge mass, overall and per stratum."""

    def summary(keep: torch.Tensor) -> dict[str, object]:
        """Return the two mass families of one row subset."""
        report = mass_report(predicted[keep], rows.target[keep])
        return {key: report[key] for key in ("wedge_mass", "bridge_mass")}

    out: dict[str, object] = {"all": summary(torch.ones(len(rows.pairs), dtype=torch.bool))}
    for stratum, name in enumerate(STRATUM_NAMES):
        keep = torch.from_numpy(rows.strata == stratum)
        if int(keep.sum()):
            out[name] = summary(keep)
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def markdown_summary(report: Mapping[str, object]) -> str:
    """Render the four tables of one group's run.

    Args:
        report: The ``fit.json`` payload.

    Returns:
        The markdown body.
    """
    final = cast(Mapping[str, Mapping[str, object]], report["final"])
    lines = [
        f"# motif generator fit -- group `{report['group']}`",
        "",
        f"`slot_read={report['slot_read']}`, "
        f"`head_output_init_std={report['head_output_init_std']}`, "
        f"checkpoint `{report['checkpoint_id']}`.",
        "",
        "## Learning curve (mean L_G)",
        "",
        "| step | train batch | fit | heldout | val_cls |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in cast(Sequence[Mapping[str, object]], report["learning_curve"]):
        lines.append(
            f"| {entry['step']} | {float(cast(float, entry['train_batch_L_G'])):.5f} "
            f"| {float(cast(float, entry['fit'])):.5f} "
            f"| {float(cast(float, entry['heldout'])):.5f} "
            f"| {float(cast(float, entry['val_cls'])):.5f} |"
        )
    lines += [
        "",
        "## Final fit against the comparators",
        "",
        "| universe | rows | generator | fitted constant | mean template | identical closure rows"
        " | wedge-mass AUROC |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in final.items():
        losses = cast(Mapping[str, float], payload["L_G"])
        auroc = payload["wedge_mass_auroc"]
        lines.append(
            f"| {name} | {payload['rows']} | {losses['generator']:.5f} "
            f"| {losses['fitted_constant_template']:.5f} | {losses['mean_template']:.5f} "
            f"| {float(cast(float, payload['identical_closure_row_fraction'])):.3f} "
            f"| {'n/a' if auroc is None else f'{float(cast(float, auroc)):.3f}'} |"
        )
    lines += [
        "",
        "## Per-stratum L_G (witness count)",
        "",
        "| universe | " + " | ".join(f"{name} (rows)" for name in STRATUM_NAMES) + " |",
        "| --- |" + " --- |" * len(STRATUM_NAMES),
    ]
    for name, payload in final.items():
        strata = cast(Mapping[str, Mapping[str, float]], payload["per_stratum_L_G"])
        cells = [f"{strata[key]['L_G']:.5f} ({int(strata[key]['rows'])})" for key in STRATUM_NAMES]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## D(S) chain (held-out rows)",
        "",
        "| stage | C / closure | L / attach |",
        "| --- | --- | --- |",
    ]
    chain = cast(Mapping[str, Mapping[str, float]], final["heldout"]["dispersion"])
    for stage, values in chain.items():
        first = values.get("C", values.get("closure"))
        second = values.get("L", values.get("attach"))
        lines.append(
            f"| {stage} | {'n/a' if first is None else f'{first:.5f}'} "
            f"| {'n/a' if second is None else f'{second:.5f}'} |"
        )
    lines += [
        "",
        "## Transplant rise of L_G (held-out rows)",
        "",
        "| scope | base L_G | mean relative rise | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    transplants = cast(Mapping[str, Mapping[str, object]], final["heldout"]["transplant"])
    for scope, payload in transplants.items():
        if "base_loss" not in payload:
            lines.append(f"| {scope} | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {scope} | {float(cast(float, payload['base_loss'])):.5f} "
            f"| {float(cast(float, payload['relative_rise_mean'])):.4f} "
            f"| {float(cast(float, payload['relative_rise_min'])):.4f} "
            f"| {float(cast(float, payload['relative_rise_max'])):.4f} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fit a fresh Stage II motif generator on cached frozen-trunk inputs over a "
            "fixed, witness-count-stratified row set, and read its fit and row dependence."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", default="baseline", choices=sorted(GROUPS))
    parser.add_argument("--fit-rows", type=int, default=4000)
    parser.add_argument("--heldout-rows", type=int, default=1500)
    parser.add_argument("--val-rows", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-chunk", type=int, default=128)
    parser.add_argument("--constant-steps", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", default="bf16", choices=["off", "bf16"])
    parser.add_argument("--token-budget", type=int, default=65_536)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one group and write ``fit.json`` and ``fit.md``.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    report = run_fit(
        checkpoint=args.checkpoint,
        pack_dir=args.pack_dir,
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=args.output_dir,
        group=args.group,
        fit_rows=args.fit_rows,
        heldout_rows=args.heldout_rows,
        val_rows=args.val_rows,
        steps=args.steps,
        batch_rows=args.batch_rows,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        seed=args.seed,
        device=su._resolve_device(args.device),
        amp=args.amp,
        token_budget=args.token_budget,
        eval_chunk=args.eval_chunk,
        constant_steps=args.constant_steps,
        permutations=args.permutations,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "fit.md").write_text(markdown_summary(report), encoding="utf-8")
    logger.info("wrote %s", args.output_dir / "fit.json")


if __name__ == "__main__":
    main()


__all__ = [
    "CANDIDATE_MULTIPLIER",
    "DISPERSION_ROWS",
    "GROUPS",
    "STRATUM_NAMES",
    "STRATUM_SHARES",
    "RowInputs",
    "RowSet",
    "allocate_strata",
    "assert_replay_matches",
    "build_generator",
    "build_parser",
    "build_row_set",
    "capture_generator_inputs",
    "collate",
    "dispersion_chain",
    "identical_closure_fraction",
    "main",
    "markdown_summary",
    "replay",
    "run_fit",
    "slot_dispersion",
    "strata_of",
    "stratified_draw",
    "wedge_mass_auroc",
    "witness_counts",
]
