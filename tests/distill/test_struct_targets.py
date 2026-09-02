"""`src.distill.struct_targets`: descriptor targets for the kd_struct arm."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from src.distill.struct_targets import (
    STRUCT_NAMES,
    STRUCT_REP_SOURCE,
    structural_row_targets,
    structural_targets,
)


def _graph(edges: list[tuple[str, str]], nodes: list[str]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def test_structural_targets_mask_the_queried_partner() -> None:
    nodes = ["a", "b", "c", "d"]
    # a-b queried; c is a common neighbour; d hangs off a only.
    graph = _graph([("a", "b"), ("a", "c"), ("b", "c"), ("a", "d")], nodes)
    out = structural_targets(
        graph, nodes, np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)
    )
    n_a, n_b = {"c", "d"}, {"c"}
    deg_c = 2
    expected = (
        np.log1p(1),
        np.log1p(len(n_a)) + np.log1p(len(n_b)),
        abs(np.log1p(len(n_a)) - np.log1p(len(n_b))),
        1 / len(n_a | n_b),
        np.log1p(1.0 / np.log1p(deg_c)),
    )
    assert out.shape == (1, len(STRUCT_NAMES))
    np.testing.assert_allclose(out[0], expected)


def test_structural_targets_self_pair_and_isolated_pair() -> None:
    nodes = ["a", "b", "c"]
    graph = _graph([("a", "b")], nodes)
    out = structural_targets(
        graph, nodes, np.array([0, 2], dtype=np.int32), np.array([0, 2], dtype=np.int32)
    )
    # Self-pair a-a: N(a) minus a is {b}, so CN = 1 and Jaccard = 1.
    assert out[0, 0] == pytest.approx(np.log1p(1))
    assert out[0, 3] == pytest.approx(1.0)
    # Isolated c-c: every descriptor is zero, Jaccard falls back to 0.
    np.testing.assert_allclose(out[1], 0.0)


def test_structural_row_targets_zscore_train_rows_and_read_val_from_val_graph() -> None:
    nodes = ["a", "b", "c", "v1", "v2"]
    train_graph = _graph([("a", "b"), ("b", "c"), ("a", "v1")], nodes)
    val_graph = _graph([("a", "b"), ("b", "c"), ("a", "v1"), ("v1", "v2")], nodes)
    train_pairs = [("a", "b"), ("a", "c"), ("b", "c"), ("c", "a")]
    val_pairs = [("v1", "v2"), ("v2", "a")]
    targets = structural_row_targets(
        train_graph=train_graph,
        val_graph=val_graph,
        train_pairs=train_pairs,
        train_labels=[1, 0, 1, 0],
        val_pairs=val_pairs,
        val_labels=[1, 0],
    )
    assert targets.node_ids == sorted(nodes)
    assert targets.teacher_rep.shape == (4, 5) and targets.val_teacher_rep.shape == (2, 5)
    assert targets.teacher_logit.shape == (4,) and not targets.teacher_logit.any()
    assert targets.manifest["rep_source"] == STRUCT_REP_SOURCE
    mean = np.asarray(targets.manifest["train_mean"])
    std = np.asarray(targets.manifest["train_std"])
    raw_train = structural_targets(
        train_graph, targets.node_ids, targets.pair_a_idx, targets.pair_b_idx
    )
    np.testing.assert_allclose(raw_train.mean(axis=0), mean)
    np.testing.assert_allclose(
        targets.teacher_rep.astype(np.float64), (raw_train - mean) / std, atol=2e-3
    )
    # The val row v1-v2 reads its neighbourhood from the val graph: v1 keeps
    # neighbour a after masking v2, so the raw degree sum is log1p(1) + 0.
    raw_val = structural_targets(
        val_graph, targets.node_ids, targets.val_pair_a_idx, targets.val_pair_b_idx
    )
    assert raw_val[0, 1] == pytest.approx(np.log1p(1))
    np.testing.assert_allclose(
        targets.val_teacher_rep.astype(np.float64), (raw_val - mean) / std, atol=2e-3
    )
    assert np.asarray(targets.val_pair_label).tolist() == [1, 0]


def test_structural_row_targets_rejects_node_set_drift() -> None:
    train_graph = _graph([("a", "b")], ["a", "b"])
    val_graph = _graph([("a", "b")], ["a", "b", "c"])
    with pytest.raises(ValueError, match="node set"):
        structural_row_targets(
            train_graph=train_graph,
            val_graph=val_graph,
            train_pairs=[("a", "b")],
            train_labels=[1],
            val_pairs=[("a", "b")],
            val_labels=[1],
        )
