from __future__ import annotations

import inspect
from itertools import combinations_with_replacement
from typing import cast

import networkx as nx
import numpy as np
import pytest
import src.eval.fixed_threshold as fixed_threshold
from src.eval.fixed_threshold import (
    _candidate_thresholds,
    _local_samples,
    _macro_abs_log_rd_curve,
    evaluate_fixed_threshold,
    select_fixed_threshold,
)
from src.eval.graph_metrics import MMDConfig

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
        "_mmd_ratio_for_statistic",
        lambda *args: 0.1 if args[1] < 1.5 else 1.0,
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
    monkeypatch.setattr(fixed_threshold, "_mmd_ratio_for_statistic", lambda *args: 1.0)

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
