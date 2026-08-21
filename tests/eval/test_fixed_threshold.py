from __future__ import annotations

from itertools import combinations_with_replacement
from typing import cast

import networkx as nx
import numpy as np
import pytest
import src.eval.fixed_threshold as fixed_threshold
from src.eval.fixed_threshold import (
    _candidate_thresholds,
    _local_samples,
    _stratified_paired_bootstrap_se,
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


def test_paired_bootstrap_se_resamples_within_size_strata() -> None:
    differences = np.array([-1.0, 1.0, 10.0, 14.0])
    strata = np.array([20, 20, 40, 40], dtype=np.int64)
    expected_variance = 0.5**2 * np.var(differences[:2]) / 2
    expected_variance += 0.5**2 * np.var(differences[2:]) / 2
    assert _stratified_paired_bootstrap_se(differences, strata) == pytest.approx(
        np.sqrt(expected_variance)
    )


def test_atomic_logit_candidates_keep_sparse_optima_and_saturated_order() -> None:
    pairs, _, graph, buckets = _fixture()
    logits = np.array([39.0, 40.0, 41.0, 42.0, 43.0, 44.0])
    samples = _local_samples(pairs=pairs, logits=logits, g_ref=graph, buckets=buckets)
    thresholds = _candidate_thresholds(samples)

    assert thresholds.size == logits.size + 1
    assert thresholds[1:].tolist() == sorted(logits.tolist(), reverse=True)
    assert len({float(1.0 / (1.0 + np.exp(-value))) for value in logits}) < logits.size


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
    assert selection.report["rule"] == "sampled_subgraph_gs_rd_mmd_1se_v2"
    gs_stage = cast(dict[str, object], selection.report["gs_stage"])
    rd_stage = cast(dict[str, object], selection.report["rd_stage"])
    assert cast(int, gs_stage["near_optimal_count"]) >= 1
    assert cast(int, rd_stage["near_optimal_count"]) >= 1

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


def test_complete_mmd_tie_prefers_higher_logit_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs, logits, graph, buckets = _fixture()
    monkeypatch.setattr(
        fixed_threshold,
        "_stratified_paired_se_curve",
        lambda *args, **kwargs: np.full(3, np.inf),
    )
    monkeypatch.setattr(fixed_threshold, "_mmd_ratio_for_statistic", lambda *args: 1.0)

    selection = select_fixed_threshold(
        pairs=pairs,
        logits=logits,
        g_ref=graph,
        buckets=buckets,
        config=MMDConfig(),
    )

    assert selection.logit_threshold > float(np.max(logits))


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
