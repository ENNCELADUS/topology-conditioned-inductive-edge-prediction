"""Tests for src.data.ego_targets: G_struct ego-net target construction."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch
from src.data.ego_targets import EgoTargetBuilder, _largest_remainder_allocation

pytestmark = pytest.mark.unit

_NODES = [f"n{i}" for i in range(8)]


def _toy_graph() -> nx.Graph:
    """n0 is a degree-5 hub; (n1, n2) closes a triangle with n0."""
    g = nx.Graph()
    g.add_nodes_from(_NODES)
    g.add_edges_from(
        [
            ("n0", "n1"),
            ("n0", "n2"),
            ("n0", "n3"),
            ("n0", "n4"),
            ("n0", "n5"),
            ("n1", "n2"),
            ("n6", "n7"),
        ]
    )
    return g


def _builder(g: nx.Graph | None = None, *, slots: int = 4) -> EgoTargetBuilder:
    graph = _toy_graph() if g is None else g
    rng = np.random.default_rng(0)
    f0 = rng.normal(size=(len(_NODES), 6)).astype(np.float32)
    index = {node: i for i, node in enumerate(_NODES)}
    pool = {node: [v for v in _NODES if v != node][:3] for node in _NODES}
    return EgoTargetBuilder(graph, f0, index, pool, slots=slots)


class TestLargestRemainderAllocation:
    def test_exact_proportions(self) -> None:
        assert _largest_remainder_allocation([2, 2], 4) == [2, 2]

    def test_total_preserved_and_capped(self) -> None:
        alloc = _largest_remainder_allocation([5, 1, 1], 4)
        assert sum(alloc) == 4
        assert all(a <= s for a, s in zip(alloc, [5, 1, 1], strict=True))

    def test_remainder_goes_to_largest_fraction(self) -> None:
        # raw = [1.33, 0.67, 2.0] over total 4 -> floor [1, 0, 2], remainder to idx 1.
        assert _largest_remainder_allocation([2, 1, 3], 4) == [1, 1, 2]


class TestEgoTargetBuilder:
    def test_small_node_gets_all_neighbors_mult_one(self) -> None:
        builder = _builder()
        out = builder.build(["n1"], np.random.default_rng(0))
        assert int(out.mask.sum()) == 2  # n0 and n2
        assert float(out.degree[0]) == 2.0
        torch.testing.assert_close(out.mult[out.mask], torch.ones(2), atol=0.0, rtol=0.0)

    def test_leave_one_out_removes_queried_partner_and_decrements_degree(self) -> None:
        builder = _builder()
        out = builder.build(
            ["n1"], np.random.default_rng(0), exclude_neighbors=["n0"]
        )

        assert int(out.mask.sum()) == 1
        assert float(out.degree[0]) == 1.0
        assert int(out.node_index[0, 0]) == _NODES.index("n2")
        np.testing.assert_allclose(
            out.ego_stats[0].numpy(), [1.0, 0.0, 1.0, 1.0], rtol=1e-6
        )

    def test_hub_multiplicity_labels_sum_to_true_degree(self) -> None:
        builder = _builder(slots=3)
        out = builder.build(["n0"], np.random.default_rng(1))
        assert int(out.mask.sum()) == 3  # capped at K
        assert float(out.degree[0]) == 5.0
        assert float(out.mult[out.mask].sum()) == pytest.approx(5.0)

    def test_adjacency_reflects_g_struct(self) -> None:
        builder = _builder()
        out = builder.build(["n0"], np.random.default_rng(2))
        selected = [t for t in range(4) if bool(out.mask[0, t])]
        # Find positions of n1 and n2 among the selected targets by feature row.
        adj = out.adj[0]
        # There is exactly one adjacent selected pair iff both n1 and n2 selected.
        n_adjacent_pairs = int(adj.sum()) // 2
        assert n_adjacent_pairs in (0, 1)
        assert len(selected) == 4

    def test_ego_stats_hand_computed(self) -> None:
        builder = _builder()
        out = builder.build(["n1", "n6", "n3"], np.random.default_rng(3))
        # n1: N = {n0, n2}; ego = triangle {n1, n0, n2} -> 3 edges, density 1.0,
        # clustering(n1) = 1.0.
        np.testing.assert_allclose(out.ego_stats[0].numpy(), [2.0, 1.0, 3.0, 1.0], rtol=1e-6)
        # n6: N = {n7}; ego = single edge -> [1, 0, 1, 1.0].
        np.testing.assert_allclose(out.ego_stats[1].numpy(), [1.0, 0.0, 1.0, 1.0], rtol=1e-6)
        # n3: N = {n0}; ego = single edge.
        np.testing.assert_allclose(out.ego_stats[2].numpy(), [1.0, 0.0, 1.0, 1.0], rtol=1e-6)

    def test_isolated_node_all_padding(self) -> None:
        g = _toy_graph()
        g.add_node("n_iso")
        rng = np.random.default_rng(0)
        f0 = rng.normal(size=(9, 6)).astype(np.float32)
        index = {node: i for i, node in enumerate([*_NODES, "n_iso"])}
        builder = EgoTargetBuilder(g, f0, index, {}, slots=4)
        out = builder.build(["n_iso"], np.random.default_rng(0))
        assert int(out.mask.sum()) == 0
        assert float(out.degree[0]) == 0.0
        np.testing.assert_allclose(out.ego_stats[0].numpy(), [0.0, 0.0, 0.0, 0.0])

    def test_deterministic_given_rng_seed(self) -> None:
        builder = _builder(slots=3)
        out_a = builder.build(["n0"], np.random.default_rng(7))
        out_b = builder.build(["n0"], np.random.default_rng(7))
        torch.testing.assert_close(out_a.features, out_b.features)
        torch.testing.assert_close(out_a.mult, out_b.mult)

    def test_in_pool_flags(self) -> None:
        builder = _builder()
        out = builder.build(["n6"], np.random.default_rng(0))
        # n6's single target is n7; pool of n6 = first 3 non-self nodes (n0..n2).
        assert bool(out.mask[0, 0])
        assert not bool(out.in_pool[0, 0])
        assert int(out.pool_index[0, 0]) == -1

    def test_in_pool_target_carries_ordered_pool_index(self) -> None:
        graph = _toy_graph()
        f0 = np.random.default_rng(0).normal(size=(len(_NODES), 6)).astype(np.float32)
        index = {node: i for i, node in enumerate(_NODES)}
        builder = EgoTargetBuilder(graph, f0, index, {"n6": ["n0", "n7"]}, slots=4)
        out = builder.build(["n6"], np.random.default_rng(0))
        assert bool(out.in_pool[0, 0])
        assert int(out.pool_index[0, 0]) == 1

    def test_rejects_self_loops(self) -> None:
        g = _toy_graph()
        g.add_edge("n0", "n0")
        with pytest.raises(ValueError, match="self-loop"):
            _builder(g)

    def test_rejects_missing_f0_rows(self) -> None:
        g = _toy_graph()
        g.add_node("n_missing")
        with pytest.raises(ValueError, match="missing F0"):
            EgoTargetBuilder(
                g,
                np.zeros((8, 6), dtype=np.float32),
                {node: i for i, node in enumerate(_NODES)},
                {},
                slots=4,
            )

    def test_features_are_f0_rows(self) -> None:
        builder = _builder()
        out = builder.build(["n6"], np.random.default_rng(0))
        expected = builder._f0[builder._index["n7"]]
        np.testing.assert_allclose(out.features[0, 0].numpy(), expected)
