"""Tests for src.experiments.vval_threshold_replay."""

from __future__ import annotations

import math

import numpy as np
import pytest
from src.data.artifacts import canonical_pair
from src.data.val_region import (
    ValRegionParams,
    ValRegionSplit,
    val_ball_union_universe,
    val_universe_arrays,
)
from src.experiments.vval_threshold_replay import _METRIC_NAMES, replay_report

pytestmark = pytest.mark.unit

_N_NODES = 15
_NODES = [f"n{i:02d}" for i in range(_N_NODES)]


def _build_synthetic_split() -> ValRegionSplit:
    """A 15-node cycle-plus-chords graph with two overlapping bucket sizes.

    Mirrors `tests/data/test_val_region.py`'s `_make_split` pattern: a
    hand-built `ValRegionSplit` rather than a full `derive_val_region_split`
    run, since only `v_val`, `val_positives`, and `buckets` matter here.
    """
    v_val = frozenset(_NODES)
    cycle_edges = [(_NODES[i], _NODES[(i + 1) % _N_NODES]) for i in range(_N_NODES)]
    chords = [(_NODES[0], _NODES[7]), (_NODES[3], _NODES[10])]
    val_positives = tuple(sorted({canonical_pair(u, v) for u, v in cycle_edges + chords}))

    buckets = {
        5: [set(_NODES[0:5]), set(_NODES[2:7])],
        4: [set(_NODES[0:4]), set(_NODES[5:9])],
    }

    return ValRegionSplit(
        train_nodes=v_val,
        v_val=v_val,
        region_seeds=(),
        training_positives=frozenset(),
        training_negatives=(),
        val_positives=val_positives,
        val_negatives=(),
        buckets=buckets,
        params=ValRegionParams(),
    )


def _random_full_probs(split: ValRegionSplit, seed: int) -> np.ndarray:
    u_idx, _ = val_universe_arrays(split.v_val)
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=u_idx.size)


class TestFullComplementIsExact:
    def test_zero_deltas_and_matching_thresholds_when_complement_fully_observed(self) -> None:
        split = _build_synthetic_split()
        probs = _random_full_probs(split, seed=0)
        complement_total = val_ball_union_universe(split).complement_total

        report = replay_report(probs, split, redraws=2, sample_size=complement_total)

        exact = report["exact"]
        assert isinstance(exact, dict)
        redraw_deltas = report["redraw_deltas"]
        assert isinstance(redraw_deltas, dict)
        for name in (*_METRIC_NAMES, "admitted_non_self"):
            summary = redraw_deltas[name]
            assert isinstance(summary, dict)
            assert summary == {"p50": 0.0, "p95": 0.0, "max_abs": 0.0}
        thresholds = redraw_deltas["thresholds"]
        assert isinstance(thresholds, list)
        assert thresholds == [exact["threshold"]] * 2


class TestExactThresholdUAssemblyMatchesExact:
    def test_bucket_metrics_are_identical_since_buckets_are_within_u(self) -> None:
        split = _build_synthetic_split()
        probs = _random_full_probs(split, seed=1)

        report = replay_report(probs, split, redraws=1, sample_size=1)

        exact = report["exact"]
        exact_threshold_u = report["exact_threshold_u_assembly"]
        assert isinstance(exact, dict)
        assert isinstance(exact_threshold_u, dict)
        for name in _METRIC_NAMES:
            assert exact_threshold_u[name] == exact[name]


class TestSmallSampleReportStructureAndDeterminism:
    def _report(
        self, split: ValRegionSplit, probs: np.ndarray, *, base_seed: int
    ) -> dict[str, object]:
        return replay_report(probs, split, redraws=5, sample_size=3, base_seed=base_seed)

    def test_report_structure_is_complete_and_finite(self) -> None:
        split = _build_synthetic_split()
        probs = _random_full_probs(split, seed=2)

        report = self._report(split, probs, base_seed=0)

        assert set(report) == {
            "n_val",
            "target_edges",
            "sample_size",
            "redraws",
            "base_seed",
            "complement_total",
            "exact",
            "exact_threshold_u_assembly",
            "redraw_deltas",
        }
        assert report["n_val"] == _N_NODES
        assert report["redraws"] == 5
        assert report["sample_size"] == 3

        exact = report["exact"]
        assert isinstance(exact, dict)
        assert set(exact) == {"threshold", "admitted_non_self", *_METRIC_NAMES}
        assert math.isfinite(exact["threshold"])

        redraw_deltas = report["redraw_deltas"]
        assert isinstance(redraw_deltas, dict)
        assert set(redraw_deltas) == {*_METRIC_NAMES, "admitted_non_self", "thresholds"}
        thresholds = redraw_deltas["thresholds"]
        assert isinstance(thresholds, list)
        assert len(thresholds) == 5
        assert all(math.isfinite(t) for t in thresholds)
        for name in (*_METRIC_NAMES, "admitted_non_self"):
            summary = redraw_deltas[name]
            assert isinstance(summary, dict)
            assert set(summary) == {"p50", "p95", "max_abs"}
            assert all(math.isfinite(v) for v in summary.values())

    def test_same_base_seed_gives_identical_reports(self) -> None:
        split = _build_synthetic_split()
        probs = _random_full_probs(split, seed=2)

        report1 = self._report(split, probs, base_seed=7)
        report2 = self._report(split, probs, base_seed=7)

        assert report1 == report2


class TestValueErrors:
    def test_raises_on_wrong_probs_length(self) -> None:
        split = _build_synthetic_split()
        probs = _random_full_probs(split, seed=3)[:-1]

        with pytest.raises(ValueError, match="probs must have"):
            replay_report(probs, split, redraws=1, sample_size=10)

    def test_raises_on_non_finite_probs(self) -> None:
        split = _build_synthetic_split()
        probs = _random_full_probs(split, seed=3)
        probs[0] = np.nan

        with pytest.raises(ValueError, match="finite"):
            replay_report(probs, split, redraws=1, sample_size=10)
