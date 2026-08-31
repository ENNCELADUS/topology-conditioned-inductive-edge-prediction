from __future__ import annotations

import copy
import inspect
from collections.abc import Sequence
from itertools import combinations_with_replacement
from typing import cast

import networkx as nx
import numpy as np
import pytest
import src.eval.fixed_threshold as fixed_threshold
from src.eval.fixed_threshold import (
    _candidate_thresholds,
    _descriptor_kernel,
    _descriptor_kernel_row,
    _incremental_mmd_ratio_curve,
    _initialize_mmd_state,
    _local_samples,
    _macro_abs_log_rd_curve,
    _mmd_ratio_for_statistic,
    _normalized_descriptor,
    _precompute_reference_descriptors,
    _stratified_paired_se_curve,
    _update_mmd_state,
    evaluate_fixed_threshold,
    select_fixed_threshold,
)
from src.eval.graph_metrics import STATISTICS, MMDConfig

pytestmark = pytest.mark.unit


def _fixture() -> tuple[list[tuple[str, str]], np.ndarray, nx.Graph, dict[int, list[set[str]]]]:
    nodes = ["a", "b", "c"]
    pairs = list(combinations_with_replacement(nodes, 2))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from([("a", "a"), ("a", "b"), ("b", "c")])
    logits = np.array([2.0 if graph.has_edge(*pair) else -2.0 for pair in pairs])
    buckets = {3: [set(nodes), set(nodes), set(nodes), set(nodes)]}
    return pairs, logits, graph, buckets


def _graded_fixture() -> tuple[
    list[tuple[str, str]], np.ndarray, nx.Graph, dict[int, list[set[str]]]
]:
    """Density optimum (RD=1 at 2.0) and GS optimum (at 1.0) deliberately differ."""
    pairs, _, graph, buckets = _fixture()
    by_pair = {
        ("a", "a"): 3.0,
        ("a", "b"): 3.0,
        ("a", "c"): 2.0,
        ("b", "b"): -2.0,
        ("b", "c"): 1.0,
        ("c", "c"): -2.0,
    }
    logits = np.array([by_pair[pair] for pair in pairs])
    return pairs, logits, graph, buckets


def test_atomic_logit_candidates_keep_sparse_optima_and_saturated_order() -> None:
    pairs, _, graph, buckets = _fixture()
    logits = np.array([39.0, 40.0, 41.0, 42.0, 43.0, 44.0])
    samples = _local_samples(pairs=pairs, logits=logits, g_ref=graph, buckets=buckets)
    thresholds = _candidate_thresholds(samples)

    assert thresholds.size == logits.size + 1
    assert thresholds[1:].tolist() == sorted(logits.tolist(), reverse=True)
    assert len({float(1.0 / (1.0 + np.exp(-value))) for value in logits}) < logits.size


def test_macro_curve_averages_buckets_not_samples() -> None:
    """D_RD weighs each size bucket equally, whatever its sample count."""
    nodes = ["a", "b", "c"]
    pairs = list(combinations_with_replacement(nodes, 2))
    graph = nx.Graph([("a", "b"), ("b", "c")])
    by_pair = {("a", "a"): 1.0, ("a", "b"): 1.0}
    logits = np.array([by_pair.get(pair, -3.0) for pair in pairs])
    # Size 2: RD 2 per sample (|log RD| = log 2); size 3: RD 1 (0), four samples.
    buckets = {2: [{"a", "b"}, {"a", "b"}], 3: [set(nodes)] * 4}
    samples = _local_samples(pairs=pairs, logits=logits, g_ref=graph, buckets=buckets)

    curve = _macro_abs_log_rd_curve(samples, np.array([1.0]))

    assert curve[0] == pytest.approx(np.log(2.0) / 2.0)  # macro, not 2*log(2)/6


def test_paired_se_weights_unequal_buckets_equally() -> None:
    """The paired SE estimates the equal-bucket macro D_RD objective."""

    def sample(size: int, logits: list[float]) -> fixed_threshold._LocalSample:
        return fixed_threshold._LocalSample(
            size=size,
            nodes=set(),
            pairs=(("a", "a"), ("a", "b")),
            logits=np.array(logits),
            truth=np.array([True, False]),
        )

    variable_bucket = [sample(2, [2.0, 0.5]), sample(2, [2.0, 1.5])]
    constant_bucket = [sample(3, [2.0, 1.5]) for _ in range(4)]

    se = _stratified_paired_se_curve(
        tuple(variable_bucket + constant_bucket),
        np.array([1.0]),
        reference_threshold=0.0,
    )

    assert se[0] == pytest.approx(np.log(2.0) / np.sqrt(32.0))


def test_density_stage_beats_gs_optimum() -> None:
    """RD=1 at 2.0 wins although 1.0 has strictly higher mean GS."""
    pairs, logits, graph, buckets = _graded_fixture()
    selection = select_fixed_threshold(
        pairs=pairs,
        logits=logits,
        g_ref=graph,
        buckets=buckets,
        config=MMDConfig(),
    )

    assert selection.logit_threshold == pytest.approx(2.0)
    assert selection.report["rule"] == "sampled_subgraph_density_shape_1se_v3"
    density_stage = cast(dict[str, object], selection.report["density_stage"])
    assert density_stage["best_logit_threshold"] == pytest.approx(2.0)
    assert density_stage["min_d_rd"] == pytest.approx(0.0)
    # Identical samples give zero SE: only the density optimum stays feasible.
    assert density_stage["feasible_count"] == 1


def test_empty_graph_candidate_is_masked_not_fail_closed() -> None:
    pairs, logits, graph, buckets = _fixture()
    selection = select_fixed_threshold(
        pairs=pairs,
        logits=logits,
        g_ref=graph,
        buckets=buckets,
        config=MMDConfig(),
    )

    report = selection.report
    density_stage = cast(dict[str, int], report["density_stage"])
    # The empty-graph boundary candidate has D_RD = +inf and never selects.
    assert density_stage["finite_candidate_count"] == cast(int, report["candidate_count"]) - 1
    assert selection.logit_threshold <= float(np.max(logits))


def test_one_se_feasibility_admits_near_optimal_then_shape_argmin_decides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs, logits, graph, buckets = _graded_fixture()
    # Finite candidates descend [3.0, 2.0, 1.0, -2.0] with D_RD
    # [log(3/2), 0, log(4/3), log 2]; this SE admits exactly {2.0, 1.0}.
    monkeypatch.setattr(
        fixed_threshold,
        "_stratified_paired_se_curve",
        lambda *args: np.array([0.0, 0.0, 0.3, 0.0]),
    )
    monkeypatch.setattr(
        fixed_threshold,
        "_incremental_mmd_ratio_curve",
        lambda *args: np.array([[1.0] * 3, [0.1] * 3]),
    )

    selection = select_fixed_threshold(
        pairs=pairs,
        logits=logits,
        g_ref=graph,
        buckets=buckets,
        config=MMDConfig(),
    )

    assert selection.logit_threshold == pytest.approx(1.0)
    density_stage = cast(dict[str, object], selection.report["density_stage"])
    shape_stage = cast(dict[str, object], selection.report["shape_stage"])
    assert density_stage["feasible_count"] == 2
    assert shape_stage["candidate_count"] == 2
    assert shape_stage["selected_d_shape"] == pytest.approx(np.log(0.1))


def test_complete_shape_tie_prefers_higher_logit_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs, logits, graph, buckets = _graded_fixture()
    monkeypatch.setattr(
        fixed_threshold,
        "_stratified_paired_se_curve",
        lambda *args: np.full(4, np.inf),
    )
    monkeypatch.setattr(
        fixed_threshold,
        "_incremental_mmd_ratio_curve",
        lambda *args: np.ones((len(args[2]), 3)),
    )

    selection = select_fixed_threshold(
        pairs=pairs,
        logits=logits,
        g_ref=graph,
        buckets=buckets,
        config=MMDConfig(),
    )

    # Every finite candidate ties on shape: the largest logit wins, never the
    # (infinite-D_RD) empty-graph boundary above it.
    assert selection.logit_threshold == pytest.approx(float(np.max(logits)))
    assert selection.report["tie_break"] == "higher_logit_threshold"


def test_selects_perfect_validation_boundary_and_replays_it_unchanged() -> None:
    pairs, logits, graph, buckets = _fixture()
    selection = select_fixed_threshold(
        pairs=pairs,
        logits=logits,
        g_ref=graph,
        buckets=buckets,
        config=MMDConfig(),
    )

    assert selection.logit_threshold == pytest.approx(2.0)
    assert selection.metrics.graph_similarity == pytest.approx(1.0)

    shifted_test_logits = logits - 0.5
    _, report = evaluate_fixed_threshold(
        pairs=pairs,
        logits=shifted_test_logits,
        g_ref=graph,
        buckets=buckets,
        threshold=selection.logit_threshold,
        config=MMDConfig(),
    )
    assert report["logit_threshold"] == pytest.approx(2.0)
    assert report["matching"] == "fixed_threshold_selected_on_validation"
    graph_similarity = cast(dict[str, float], report["graph_similarity"])
    relative_density = cast(dict[str, float], report["relative_density"])
    assert graph_similarity["bfs_macro"] == pytest.approx(0.0)
    assert relative_density["bfs_macro"] == pytest.approx(0.0)


def test_incremental_shape_curve_matches_brute_force_with_unequal_buckets_and_ties() -> None:
    rng = np.random.default_rng(7391)
    nodes = [f"n{index}" for index in range(7)]
    pairs = list(combinations_with_replacement(nodes, 2))
    logits = rng.integers(-3, 4, size=len(pairs)).astype(np.float64)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((node, node) for node in nodes)
    graph.add_edges_from(pair for pair in pairs if rng.random() < 0.3)
    buckets = {
        3: [{"n0", "n1", "n2"}, {"n1", "n3", "n5"}, {"n2", "n4", "n6"}],
        4: [
            {"n0", "n1", "n3", "n6"},
            {"n0", "n2", "n4", "n5"},
            {"n1", "n2", "n5", "n6"},
            {"n0", "n3", "n4", "n6"},
            {"n1", "n3", "n4", "n5"},
        ],
    }
    config = MMDConfig(sigma=0.7, reference_epsilon=1e-10)
    samples = _local_samples(pairs=pairs, logits=logits, g_ref=graph, buckets=buckets)
    thresholds = _candidate_thresholds(samples)
    feasible_indices = np.unique(np.array([2, 4, thresholds.size - 1], dtype=np.intp))
    assert np.any(np.diff(feasible_indices) > 1)
    assert np.unique(logits).size < logits.size
    assert sum(("n1", "n1") in sample.pairs for sample in samples) > 1
    references, denominators = _precompute_reference_descriptors(samples, config)

    incremental = _incremental_mmd_ratio_curve(
        samples, thresholds, feasible_indices, references, denominators, config
    )
    brute_force = np.array(
        [
            [
                _mmd_ratio_for_statistic(
                    samples,
                    float(thresholds[index]),
                    statistic,
                    references,
                    denominators,
                    config,
                )
                for statistic in STATISTICS
            ]
            for index in feasible_indices
        ]
    )

    np.testing.assert_allclose(incremental, brute_force, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        np.log(np.maximum(incremental, config.reference_epsilon)).mean(axis=1),
        np.log(np.maximum(brute_force, config.reference_epsilon)).mean(axis=1),
        rtol=1e-11,
        atol=1e-11,
    )
    incremental_best = int(np.argmin(np.log(np.maximum(incremental, 1e-10)).mean(axis=1)))
    brute_force_best = int(np.argmin(np.log(np.maximum(brute_force, 1e-10)).mean(axis=1)))
    assert (
        thresholds[feasible_indices[incremental_best]]
        == thresholds[feasible_indices[brute_force_best]]
    )


def test_vectorized_kernel_row_matches_scalar_variable_support() -> None:
    rng = np.random.default_rng(481)
    left = rng.random(17)
    right = [rng.random(size) for size in (3, 17, 29, 7, 21)]

    vectorized = _descriptor_kernel_row(left, right, sigma=0.73)
    scalar = np.array([_descriptor_kernel(left, values, 0.73) for values in right])

    np.testing.assert_allclose(vectorized, scalar, rtol=1e-15, atol=0.0)


def test_vectorized_mmd_update_matches_scalar_and_batches_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(982)
    config = MMDConfig(sigma=0.61)
    predicted = [_normalized_descriptor(rng.random(size)) for size in (11, 17, 23, 29, 31)]
    reference = [_normalized_descriptor(rng.random(size)) for size in (7, 13, 19, 27, 31)]
    state = _initialize_mmd_state(predicted, reference, config)
    expected = copy.deepcopy(state)
    updates = {
        1: _normalized_descriptor(rng.random(25)),
        3: _normalized_descriptor(rng.random(15)),
    }

    for index in sorted(updates):
        descriptor = updates[index]
        expected.predicted[index] = descriptor
        for other_index, other in enumerate(expected.predicted):
            if other_index == index:
                continue
            value = _descriptor_kernel(descriptor, other, config.sigma)
            previous = float(expected.predicted_kernel[index, other_index])
            expected.predicted_sum += 2.0 * (value - previous)
            expected.predicted_kernel[index, other_index] = value
            expected.predicted_kernel[other_index, index] = value
        for reference_index, values in enumerate(expected.reference):
            value = _descriptor_kernel(descriptor, values, config.sigma)
            previous = float(expected.cross_kernel[index, reference_index])
            expected.cross_sum += value - previous
            expected.cross_kernel[index, reference_index] = value

    row_calls = 0
    kernel_row = fixed_threshold._descriptor_kernel_row

    def counted_row(left: np.ndarray, right: Sequence[np.ndarray], sigma: float) -> np.ndarray:
        nonlocal row_calls
        row_calls += 1
        return kernel_row(left, right, sigma)

    monkeypatch.setattr(fixed_threshold, "_descriptor_kernel_row", counted_row)
    _update_mmd_state(state, updates, config)

    assert row_calls == len(updates)
    np.testing.assert_allclose(state.predicted_kernel, expected.predicted_kernel, rtol=1e-15)
    np.testing.assert_allclose(state.cross_kernel, expected.cross_kernel, rtol=1e-15)
    assert state.predicted_sum == pytest.approx(expected.predicted_sum, rel=1e-15)
    assert state.cross_sum == pytest.approx(expected.cross_sum, rel=1e-15)
    assert state.predicted_sum == pytest.approx(
        sum(float(value) for row in state.predicted_kernel for value in row), rel=1e-15
    )
    assert state.cross_sum == pytest.approx(
        sum(float(value) for row in state.cross_kernel for value in row), rel=1e-15
    )


def test_incremental_sweep_refreshes_a_tied_sample_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = (
        fixed_threshold._LocalSample(
            size=2,
            nodes={"a", "b"},
            pairs=(("a", "a"), ("a", "b"), ("b", "b")),
            logits=np.array([2.0, 1.0, 1.0]),
            truth=np.ones(3, dtype=np.bool_),
        ),
        fixed_threshold._LocalSample(
            size=2,
            nodes={"c"},
            pairs=(("c", "c"),),
            logits=np.array([2.0]),
            truth=np.ones(1, dtype=np.bool_),
        ),
    )
    thresholds = np.array([np.nextafter(2.0, np.inf), 2.0, 1.0])
    references = {statistic: {2: [np.ones(1), np.ones(1)]} for statistic in STATISTICS}
    descriptor_calls = 0

    def counted_descriptor(graph: nx.Graph, statistic: str) -> np.ndarray:
        nonlocal descriptor_calls
        descriptor_calls += 1
        return np.array([graph.number_of_edges() + 1.0])

    monkeypatch.setattr(fixed_threshold, "_descriptor", counted_descriptor)
    monkeypatch.setattr(fixed_threshold, "_initialize_mmd_state", lambda *args: object())
    monkeypatch.setattr(fixed_threshold, "_update_mmd_state", lambda *args: None)
    monkeypatch.setattr(fixed_threshold, "_mmd_ratio_from_states", lambda *args: 1.0)

    _incremental_mmd_ratio_curve(
        samples,
        thresholds,
        np.array([1, 2], dtype=np.intp),
        references,
        dict.fromkeys(STATISTICS, 1.0),
        MMDConfig(),
    )

    # Six initial descriptors, then one refresh per statistic although two tied
    # edges enter the first sample together at the second feasible boundary.
    assert descriptor_calls == 9


def test_incremental_sweep_descriptor_work_scales_with_events_not_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production geometry ignores a large non-feasible atomic-threshold prefix."""
    sample_count = 500
    feasible_count = 10_025
    nonfeasible_prefix = 20_000
    event_count = nonfeasible_prefix + feasible_count
    sizes = tuple(range(10))
    samples: list[fixed_threshold._LocalSample] = []
    next_event = 0
    for sample_index in range(sample_count):
        count = event_count // sample_count + (sample_index < event_count % sample_count)
        sample_pairs = tuple(
            (f"u{event_index}", f"v{event_index}")
            for event_index in range(next_event, next_event + count)
        )
        samples.append(
            fixed_threshold._LocalSample(
                size=sizes[sample_index // 50],
                nodes={node for pair in sample_pairs for node in pair},
                pairs=sample_pairs,
                logits=np.arange(next_event, next_event + count, dtype=np.float64),
                truth=np.ones(count, dtype=np.bool_),
            )
        )
        next_event += count
    thresholds = np.concatenate(
        ([float(event_count)], np.arange(event_count - 1, -1, -1, dtype=np.float64))
    )
    feasible_indices = np.arange(nonfeasible_prefix + 1, thresholds.size, dtype=np.intp)
    references = {
        statistic: {size: [np.ones(1)] * 50 for size in sizes} for statistic in STATISTICS
    }
    descriptor_calls = 0

    def counted_descriptor(graph: nx.Graph, statistic: str) -> np.ndarray:
        nonlocal descriptor_calls
        descriptor_calls += 1
        return np.array([graph.number_of_edges() + 1.0])

    monkeypatch.setattr(fixed_threshold, "_descriptor", counted_descriptor)
    monkeypatch.setattr(fixed_threshold, "_initialize_mmd_state", lambda *args: object())
    monkeypatch.setattr(fixed_threshold, "_update_mmd_state", lambda *args: None)
    monkeypatch.setattr(fixed_threshold, "_mmd_ratio_from_states", lambda *args: 1.0)

    ratios = _incremental_mmd_ratio_curve(
        tuple(samples),
        thresholds,
        feasible_indices,
        references,
        dict.fromkeys(STATISTICS, 1.0),
        MMDConfig(),
    )

    assert ratios.shape == (feasible_count, len(STATISTICS))
    touched_across_later_feasible_checkpoints = feasible_count - 1
    assert descriptor_calls == len(STATISTICS) * (
        sample_count + touched_across_later_feasible_checkpoints
    )
    brute_force_descriptor_calls = len(STATISTICS) * sample_count * feasible_count
    assert descriptor_calls * 400 < brute_force_descriptor_calls


def test_evaluate_fixed_threshold_has_no_matching_parameter() -> None:
    """The label is a constant; the fixed-0.5 block that parameterized it is gone."""
    assert "matching" not in inspect.signature(evaluate_fixed_threshold).parameters


def test_missing_pair_and_nonfinite_logit_fail_closed() -> None:
    pairs, logits, graph, buckets = _fixture()
    with pytest.raises(ValueError, match="missing topology score"):
        select_fixed_threshold(
            pairs=pairs[:-1],
            logits=logits[:-1],
            g_ref=graph,
            buckets=buckets,
            config=MMDConfig(),
        )
    bad = logits.copy()
    bad[-1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_fixed_threshold(
            pairs=pairs,
            logits=bad,
            g_ref=graph,
            buckets=buckets,
            config=MMDConfig(),
        )
