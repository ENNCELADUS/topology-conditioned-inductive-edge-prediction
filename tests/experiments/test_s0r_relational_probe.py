"""Tests for the S0-R relational probe's pure pieces."""

import networkx as nx
import numpy as np
from src.experiments.g1_hardened_e2 import common_neighbor_and_adamic_adar
from src.experiments.probes import _ridge_fit_predict
from src.experiments.s0_summary_probe import (
    chunked_ridge_fit,
    chunked_ridge_predict,
    cv_pair_scores,
)
from src.experiments.s0r_relational_probe import pair_cn_aa, sample_fit_pairs

_GRID = (1e-3, 1e0, 1e3)


def _toy_graph() -> nx.Graph:
    return nx.Graph([("a", "c"), ("b", "c"), ("a", "d"), ("b", "d"), ("d", "e")])


def test_pair_cn_aa_matches_hand_computation() -> None:
    graph = _toy_graph()
    result = pair_cn_aa(graph, [("a", "b"), ("a", "e"), ("a", "c")])
    # CN(a,b) = |{c, d}| = 2; AA = 1/log(deg c=2) + 1/log(deg d=3).
    np.testing.assert_allclose(result[0], [2.0, 1 / np.log(2) + 1 / np.log(3)])
    # CN(a,e) = |{d}| = 1; AA = 1/log(3).
    np.testing.assert_allclose(result[1], [1.0, 1 / np.log(3)])
    # (a,c) is an edge, but the edge itself never counts: no common neighbors.
    np.testing.assert_allclose(result[2], [0.0, 0.0])


def test_pair_cn_aa_is_symmetric_and_matches_dense_reference() -> None:
    graph = _toy_graph()
    nodes = sorted(graph.nodes())
    position = {node: i for i, node in enumerate(nodes)}
    cn_dense, aa_dense = common_neighbor_and_adamic_adar(graph, nodes)
    pairs = [(u, v) for u in nodes for v in nodes if u < v]
    result = pair_cn_aa(graph, pairs)
    swapped = pair_cn_aa(graph, [(v, u) for u, v in pairs])
    np.testing.assert_allclose(result, swapped)
    for row, (u, v) in enumerate(pairs):
        np.testing.assert_allclose(
            result[row], [cn_dense[position[u], position[v]], aa_dense[position[u], position[v]]]
        )


def test_sample_fit_pairs_is_deterministic_and_well_formed() -> None:
    graph = nx.gnm_random_graph(30, 60, seed=0)
    graph = nx.relabel_nodes(graph, {i: f"n{i:02d}" for i in graph.nodes()})
    nodes = sorted(graph.nodes())
    first = sample_fit_pairs(graph, nodes, n_random=50, n_positive=5, seed=7)
    second = sample_fit_pairs(graph, nodes, n_random=50, n_positive=5, seed=7)
    assert first == second
    assert len(first) == len(set(first))
    edge_set = {tuple(sorted(edge)) for edge in graph.edges()}
    positives = [pair for pair in first if pair in edge_set]
    assert len(positives) >= 5
    for u, v in first:
        assert u != v
        assert u < v


def test_chunked_ridge_fit_matches_direct_ridge() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4))
    y = x @ rng.normal(size=4) + 0.01 * rng.normal(size=50)
    weights, feature_mean, target_mean = chunked_ridge_fit(
        lambda idx: x[idx], 50, y, lam=1e-2, chunk=7
    )
    predicted = chunked_ridge_predict(
        lambda idx: x[idx],
        np.arange(50, dtype=np.int64),
        weights,
        feature_mean,
        target_mean,
        chunk=7,
    )
    reference = _ridge_fit_predict(x, y, x, 1e-2)
    np.testing.assert_allclose(predicted, reference, rtol=1e-8)


def test_cv_pair_scores_regression_recovers_linear_target() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(400, 5))
    targets = features @ rng.normal(size=5) + 0.05 * rng.normal(size=400)
    scores, fold_lambdas = cv_pair_scores(
        lambda idx: features[idx], 400, targets, lambdas=_GRID, regression=True
    )
    assert len(fold_lambdas) == 5
    assert np.corrcoef(scores, targets)[0, 1] > 0.9
