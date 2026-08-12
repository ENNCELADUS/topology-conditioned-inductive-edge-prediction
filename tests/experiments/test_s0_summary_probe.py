"""Tests for the S0 summary-value diagnostic's pure pieces."""

import json

import networkx as nx
import numpy as np
import pytest
from src.experiments.s0_summary_probe import (
    NestedProbeResult,
    cv_pair_scores,
    nested_probe_r2,
    paired_bootstrap_delta,
    partner_excluded_pair_stats,
    symmetric_pair_features,
    to_json_payload,
)

_GRID = (1e-3, 1e0, 1e3)


def _linear_data(
    n: int = 200, d: int = 5, noise: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n, d))
    weights = rng.normal(size=d)
    targets = states @ weights + noise * rng.normal(size=n)
    return states, targets


def test_nested_probe_r2_recovers_linear_signal() -> None:
    states, targets = _linear_data()
    result = nested_probe_r2(states, targets, lambdas=_GRID)
    assert isinstance(result, NestedProbeResult)
    assert result.r2 > 0.9
    assert len(result.fold_lambdas) == 5
    assert set(result.fold_lambdas) <= set(_GRID)


def test_nested_probe_r2_is_near_zero_for_shuffled_targets() -> None:
    states, targets = _linear_data()
    shuffled = np.random.default_rng(1).permutation(targets)
    result = nested_probe_r2(states, shuffled, lambdas=_GRID)
    assert result.r2 < 0.2


def test_nested_probe_r2_degree_partialled_zeroes_out_degree_target() -> None:
    states, _ = _linear_data()
    degrees = np.arange(states.shape[0], dtype=np.float64)
    result = nested_probe_r2(states, degrees, lambdas=_GRID, degrees=degrees)
    assert result.r2 == 0.0


def test_symmetric_pair_features_is_root_swap_invariant() -> None:
    node_matrix = np.random.default_rng(0).normal(size=(6, 3))
    u_idx = np.array([0, 2], dtype=np.int64)
    v_idx = np.array([1, 5], dtype=np.int64)
    forward = symmetric_pair_features(node_matrix, u_idx, v_idx)
    backward = symmetric_pair_features(node_matrix, v_idx, u_idx)
    assert forward.shape == (2, 6)
    np.testing.assert_allclose(forward, backward)


def test_partner_excluded_pair_stats_decrements_degree_and_swaps() -> None:
    graph = nx.Graph([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])
    forward = partner_excluded_pair_stats(graph, [("a", "b")])
    backward = partner_excluded_pair_stats(graph, [("b", "a")])
    np.testing.assert_allclose(forward, backward)
    # With b excluded, a's ego is {a, c}: degree 1, clustering 0, one edge.
    # With a excluded, b's ego is {b, c}: identical stats, so the symmetric
    # |difference| block must be exactly zero.
    assert forward.shape == (1, 8)
    np.testing.assert_allclose(forward[0, :4], [2.0, 0.0, 2.0, 2.0])
    np.testing.assert_allclose(forward[0, 4:], 0.0)


def test_partner_excluded_pair_stats_ignores_non_neighbor_partner() -> None:
    graph = nx.Graph([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])
    stats = partner_excluded_pair_stats(graph, [("a", "d")])
    # a and d are not adjacent: a keeps degree 2 (triangle: clustering 1),
    # d keeps degree 1.
    su_plus_sv = stats[0, :4]
    assert su_plus_sv[0] == pytest.approx(3.0)  # degree 2 + degree 1


def test_partner_excluded_pair_stats_matches_ego_target_builder() -> None:
    from src.data.ego_targets import EgoTargetBuilder

    graph = nx.Graph([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])
    builder = EgoTargetBuilder(
        graph,
        np.zeros((4, 3), dtype=np.float32),
        {"a": 0, "b": 1, "c": 2, "d": 3},
        {},
        slots=2,
    )
    for u, v in [("a", "b"), ("c", "d"), ("a", "d")]:
        su = np.asarray(builder._node_ego_stats(u, exclude_neighbor=v))
        sv = np.asarray(builder._node_ego_stats(v, exclude_neighbor=u))
        expected = np.concatenate([su + sv, np.abs(su - sv)])
        actual = partner_excluded_pair_stats(graph, [(u, v)])
        np.testing.assert_allclose(actual[0], expected)


def _pair_score_fixture(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.1).astype(np.int64)
    features = labels[:, None] * 2.0 + rng.normal(size=(n, 4))
    return features, labels


def test_cv_pair_scores_separates_separable_labels() -> None:
    features, labels = _pair_score_fixture()
    scores, fold_lambdas = cv_pair_scores(
        lambda idx: features[idx], len(labels), labels, lambdas=_GRID
    )
    assert scores.shape == labels.shape
    assert len(fold_lambdas) == 5
    from sklearn.metrics import average_precision_score

    assert average_precision_score(labels, scores) > 0.6


def test_cv_pair_scores_is_chance_level_on_constant_features() -> None:
    _, labels = _pair_score_fixture()
    constant = np.ones((len(labels), 3))
    scores, _ = cv_pair_scores(lambda idx: constant[idx], len(labels), labels, lambdas=_GRID)
    from sklearn.metrics import average_precision_score

    prevalence = labels.mean()
    assert average_precision_score(labels, scores) < prevalence * 3


def test_paired_bootstrap_delta_contains_zero_for_identical_scores() -> None:
    _, labels = _pair_score_fixture()
    scores = np.random.default_rng(2).random(len(labels))
    delta, lo, hi = paired_bootstrap_delta(labels, scores, scores)
    assert delta == pytest.approx(0.0)
    assert lo <= 0.0 <= hi


def test_paired_bootstrap_delta_detects_clear_improvement() -> None:
    features, labels = _pair_score_fixture()
    good = labels + 0.05 * np.random.default_rng(3).normal(size=len(labels))
    flat = np.random.default_rng(4).random(len(labels))
    delta, lo, _ = paired_bootstrap_delta(labels, good, flat)
    assert delta > 0.0
    assert lo > 0.0


def test_to_json_payload_is_serializable_and_labeled_diagnostic() -> None:
    payload = to_json_payload(
        strategy="breadth_first",
        arm_a={"degree": {"r2_raw": 0.4, "r2_degree_partialled": 0.0}},
        arm_b={"row1_features_only": {"auprc": 0.1, "auroc": 0.7}},
        deltas={"row2_minus_row1": {"delta": 0.01, "lo": 0.0, "hi": 0.02}},
        manifest_digests={"pair_labels_sha256": "deadbeef"},
    )
    assert payload["evidence_class"] == "diagnostic"
    assert json.dumps(payload)
