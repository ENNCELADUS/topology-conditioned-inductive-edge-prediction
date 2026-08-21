"""Tests for sampled-only V_val topology selection."""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pytest
from src.data.val_region import ValRegionParams, ValRegionSplit, val_ball_union_universe
from src.eval.val_topology import build_val_topology_reference, val_region_topology_metrics

pytestmark = pytest.mark.unit


def _toy_split() -> ValRegionSplit:
    nodes = frozenset({"a", "b", "c", "d"})
    return ValRegionSplit(
        train_nodes=nodes,
        v_val=nodes,
        region_seeds=("a",),
        training_positives=frozenset(),
        training_negatives=(),
        val_positives=(("a", "b"), ("d", "d")),
        val_negatives=(),
        buckets={2: [{"a", "b"}, {"c", "d"}]},
        params=ValRegionParams(),
    )


def _rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = _toy_split()
    reference = build_val_topology_reference(split)
    universe = val_ball_union_universe(split)
    pairs = [
        (reference.nodes[int(u)], reference.nodes[int(v)])
        for u, v in zip(universe.u_idx, universe.v_idx, strict=True)
    ]
    logits = np.array(
        [2.0 if reference.g_val.has_edge(*pair) else -2.0 for pair in pairs],
        dtype=np.float64,
    )
    return universe.u_idx, universe.v_idx, logits


def test_perfect_sample_union_selects_perfect_fixed_threshold() -> None:
    split = _toy_split()
    reference = build_val_topology_reference(split)
    u_idx, v_idx, logits = _rows()

    result = val_region_topology_metrics(
        u_idx=u_idx,
        v_idx=v_idx,
        logits=logits,
        reference=reference,
    )

    assert result.threshold == pytest.approx(2.0)
    assert result.metrics.gs == pytest.approx(1.0)
    assert result.metrics.rd == pytest.approx(1.0)
    assert result.metrics.degree_mmd == pytest.approx(0.0)
    assert result.metrics.clustering_mmd == pytest.approx(0.0)
    assert result.metrics.spectral_mmd == pytest.approx(0.0)


def test_union_contains_exactly_each_within_ball_pair() -> None:
    split = _toy_split()
    reference = build_val_topology_reference(split)
    universe = val_ball_union_universe(split)
    actual = {
        (reference.nodes[int(u)], reference.nodes[int(v)])
        for u, v in zip(universe.u_idx, universe.v_idx, strict=True)
    }
    expected = {
        pair
        for nodes in split.buckets[2]
        for pair in combinations_with_replacement(sorted(nodes), 2)
    }
    assert actual == expected


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_union_logit_fails_closed(bad: float) -> None:
    split = _toy_split()
    reference = build_val_topology_reference(split)
    u_idx, v_idx, logits = _rows()
    logits[0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        val_region_topology_metrics(
            u_idx=u_idx,
            v_idx=v_idx,
            logits=logits,
            reference=reference,
        )


def test_misaligned_or_out_of_range_rows_fail_closed() -> None:
    split = _toy_split()
    reference = build_val_topology_reference(split)
    u_idx, v_idx, logits = _rows()
    with pytest.raises(ValueError, match="length mismatch"):
        val_region_topology_metrics(
            u_idx=u_idx[:-1],
            v_idx=v_idx,
            logits=logits,
            reference=reference,
        )
    bad_u = u_idx.copy()
    bad_u[0] = len(reference.nodes)
    with pytest.raises(ValueError, match="outside"):
        val_region_topology_metrics(
            u_idx=bad_u,
            v_idx=v_idx,
            logits=logits,
            reference=reference,
        )
