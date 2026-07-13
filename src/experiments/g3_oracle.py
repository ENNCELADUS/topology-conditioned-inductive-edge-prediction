r"""Gate G3 (Oracle) analysis pipeline: cached scores + true neighborhoods -> headroom.

Compares the frozen B0 scorer's density-matched assembly against two
protocol-violating oracle reference arms built from TRUE held-out test-graph
neighborhoods (docs/03-experiment-protocol.md SS5.0 G3, pinned instantiation
2026-07-13): ``oracle_topo`` (common-neighbor count, ties by Adamic-Adar, then
canonical pair order) and ``oracle_blend`` (parameter-free rank fusion of the
B0 probabilities with ``oracle_topo``). Reports the stop-rule quantity: per
statistic, ``mmd_ratio(B0) / mmd_ratio(oracle arm)`` (headroom). No model
scoring happens here -- everything is a row-selection or graph computation
over the cached candidate-universe artifact and the benchmark package.

CLI::

    python -m src.experiments.g3_oracle \
        --universe scores/b0_v31_candidate.npz \
        --data-root data --strategy breadth_first \
        --output-dir outputs/g3 [--seed 0] [--skip-perturbation-check]

Determinism: identical inputs produce a byte-identical ``g3_results.json``
(stable sorts everywhere; ``json.dumps(..., sort_keys=True)``; no wall-clock
fields in the payload).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.eval.graph_metrics import STATISTICS
from src.experiments.g1_hardened_e2 import AssembledRow

# --------------------------------------------------------------------------- rank scalarization


def _average_tie_rank01_from_order(
    order: NDArray[np.int64], new_run: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Average-tie normalized ranks in ``[0, 1]`` from a precomputed sort order.

    Args:
        order: Permutation sorting the values ascending (stable).
        new_run: Shape ``(n - 1,)`` booleans; ``True`` where the sorted value at
            position ``i + 1`` starts a new tie run.

    Returns:
        Shape ``(n,)`` float64 scores: rank / (n - 1), where tied values share
        the mean of the integer ranks they occupy (1 = largest key; all-tied
        input degenerates to a uniform 0.5, matching `normalize_min_max`'s
        uninformative-but-valid convention; a single row maps to 0.5).
    """
    n = order.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.full(1, 0.5, dtype=np.float64)
    run_id = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(new_run, dtype=np.int64)])
    counts = np.bincount(run_id)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)[:-1]])
    mean_ranks = starts.astype(np.float64) + (counts.astype(np.float64) - 1.0) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = mean_ranks[run_id]
    return ranks / (n - 1.0)


def rank01(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average-tie normalized rank of `values` in ``[0, 1]`` (1 = largest).

    Deterministic (stable sort); idempotent on its own output. Rows with equal
    values share the mean of the integer ranks they occupy.

    Args:
        values: Shape ``(n,)`` float64 scores.

    Returns:
        Shape ``(n,)`` float64 normalized ranks.
    """
    order = np.argsort(values, kind="stable").astype(np.int64)
    sorted_values = values[order]
    new_run = cast(NDArray[np.bool_], sorted_values[1:] != sorted_values[:-1])
    return _average_tie_rank01_from_order(order, new_run)


def rank01_lex(primary: NDArray[np.float64], secondary: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average-tie normalized rank of the ``(primary, secondary)`` lexicographic key.

    Rows are ranked ascending by `primary`, ties by `secondary`; rows equal on
    BOTH keys share the mean rank. This is the `oracle_topo` scalarization of
    the ``(CN, AA)`` key (docs/03 SS5.0 G3): a strictly monotone function of
    the pinned lexicographic ordering, so ranking by the returned scalar (with
    any downstream tie-break) reproduces that ordering exactly.

    Args:
        primary: Shape ``(n,)`` float64 primary key (common-neighbor counts).
        secondary: Shape ``(n,)`` float64 secondary key (Adamic-Adar), aligned.

    Returns:
        Shape ``(n,)`` float64 normalized ranks in ``[0, 1]`` (1 = most edge-like).
    """
    order = np.lexsort((secondary, primary)).astype(np.int64)
    p_sorted = primary[order]
    s_sorted = secondary[order]
    new_run = cast(
        NDArray[np.bool_],
        (p_sorted[1:] != p_sorted[:-1]) | (s_sorted[1:] != s_sorted[:-1]),
    )
    return _average_tie_rank01_from_order(order, new_run)


# --------------------------------------------------------------------------- headroom


@dataclass(frozen=True)
class HeadroomRow:
    """Stop-rule headroom of one oracle arm against the B0 assembled row.

    Attributes:
        mmd_ratio_headroom: Statistic -> ``b0.mmd_ratio / arm.mmd_ratio``
            (``None`` where the arm ratio is exactly 0 -- disclosed, never inf).
            Values >> 1 mean the oracle assembles a far more realistic graph
            than B0 (headroom exists); ~1 triggers the G3 stop discussion.
        composite_ratio: ``arm.composite / b0.composite`` when both composites
            were computed and B0's is nonzero, else ``None``.
    """

    mmd_ratio_headroom: dict[str, float | None]
    composite_ratio: float | None


def compute_headroom(b0_row: AssembledRow, arm_row: AssembledRow) -> HeadroomRow:
    """Compute the G3 stop-rule headroom of one oracle arm against B0.

    Args:
        b0_row: The B0 density-matched assembled row.
        arm_row: One oracle arm's assembled row.

    Returns:
        The `HeadroomRow` (see its attribute docs for the exact definitions).
    """
    ratios: dict[str, float | None] = {}
    for stat in STATISTICS:
        arm_ratio = arm_row.mmd_ratio[stat]
        ratios[stat] = None if arm_ratio == 0.0 else b0_row.mmd_ratio[stat] / arm_ratio
    composite_ratio: float | None = None
    if b0_row.composite is not None and arm_row.composite is not None and b0_row.composite != 0.0:
        composite_ratio = arm_row.composite / b0_row.composite
    return HeadroomRow(mmd_ratio_headroom=ratios, composite_ratio=composite_ratio)
