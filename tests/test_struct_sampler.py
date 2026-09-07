from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest
from src.data.struct_sampler import StructEpochPlan, StructSampler, StructSubgraph


def _toy_graph(seed: int = 3, n: int = 300, m: int = 3) -> nx.Graph:
    graph = nx.barabasi_albert_graph(n, m, seed=seed)
    return nx.relabel_nodes(graph, {i: f"node_{i:06d}" for i in graph.nodes})


def _sampler(**overrides: object) -> StructSampler:
    graph = _toy_graph()
    v_val = frozenset(f"node_{i:06d}" for i in range(0, 300, 5))
    exclude = frozenset({"node_000007", "node_000011"})
    kwargs: dict[str, object] = {
        "nodes": 24,
        "background_nodes": 4,
        "mix": {"bfs": 0.5, "motif": 0.25, "bridge": 0.25},
        "v_val": v_val,
        "exclude_nodes": exclude,
    }
    kwargs.update(overrides)
    return StructSampler(graph, **kwargs)  # type: ignore[arg-type]


def test_plan_sizes_kinds_and_background_counts() -> None:
    sampler = _sampler()
    plan = sampler.plan(seed=0, epoch=1, count=50)
    assert isinstance(plan, StructEpochPlan)
    assert len(plan.subgraphs) == 50
    for subgraph in plan.subgraphs:
        assert isinstance(subgraph, StructSubgraph)
        assert len(subgraph.nodes) == 24
        assert len(set(subgraph.nodes)) == 24
        assert subgraph.background == 4
        assert subgraph.kind in {"bfs", "motif", "bridge"}
        assert not set(subgraph.nodes) & {"node_000007", "node_000011"}


def test_legal_mask_never_admits_forbidden_pairs() -> None:
    sampler = _sampler()
    v_val = sampler.v_val
    plan = sampler.plan(seed=1, epoch=2, count=40)
    for subgraph in plan.subgraphs:
        mask = sampler.legal_mask(subgraph)
        assert mask.shape == (24, 24)
        assert np.array_equal(mask, mask.T)
        assert float(np.diagonal(mask).sum()) == 0.0
        for i, j in itertools.combinations(range(24), 2):
            both_val = subgraph.nodes[i] in v_val and subgraph.nodes[j] in v_val
            assert bool(mask[i, j]) == (not both_val)
        adjacency = sampler.adjacency(subgraph)
        assert np.array_equal(adjacency, adjacency * mask)
        for i, j in zip(*np.nonzero(adjacency), strict=True):
            assert sampler.graph.has_edge(subgraph.nodes[i], subgraph.nodes[j])


def test_kind_mix_is_respected() -> None:
    plan = _sampler().plan(seed=5, epoch=1, count=400)
    kinds = [subgraph.kind for subgraph in plan.subgraphs]
    assert abs(kinds.count("bfs") / 400 - 0.5) < 0.05
    assert abs(kinds.count("motif") / 400 - 0.25) < 0.05
    assert abs(kinds.count("bridge") / 400 - 0.25) < 0.05


def test_motif_seeds_are_real_wedges_or_triangles() -> None:
    sampler = _sampler(mix={"bfs": 0.0, "motif": 1.0, "bridge": 0.0})
    graph = sampler.graph
    plan = sampler.plan(seed=2, epoch=1, count=30)
    seen_open = seen_closed = False
    for subgraph in plan.subgraphs:
        a, b, c = subgraph.nodes[:3]
        # Wedge seeds are (centre, leaf, leaf); triangle seeds are (u, v, common neighbour).
        assert graph.has_edge(a, b) and graph.has_edge(a, c)
        if graph.has_edge(b, c):
            seen_closed = True
        else:
            seen_open = True
    assert seen_open and seen_closed


def test_bridge_roots_are_far_apart() -> None:
    sampler = _sampler(mix={"bfs": 0.0, "motif": 0.0, "bridge": 1.0})
    graph = sampler.graph
    half = (24 - 4) // 2
    for subgraph in sampler.plan(seed=9, epoch=1, count=20).subgraphs:
        first_root, second_root = subgraph.nodes[0], subgraph.nodes[half]
        assert nx.shortest_path_length(graph, first_root, second_root) >= 3


def test_plan_is_world_size_independent_and_epoch_dependent() -> None:
    sampler = _sampler()
    first = sampler.plan(seed=0, epoch=3, count=16)
    again = sampler.plan(seed=0, epoch=3, count=16)
    other = sampler.plan(seed=0, epoch=4, count=16)
    assert first == again
    assert first != other
    # A shorter plan is a prefix of a longer one: position-keyed randomness, no rank in the plan.
    assert sampler.plan(seed=0, epoch=3, count=8).subgraphs == first.subgraphs[:8]


def test_small_component_falls_back_to_more_roots() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "h")])
    graph.add_nodes_from([f"iso{i}" for i in range(10)])
    sampler = StructSampler(
        graph,
        nodes=8,
        background_nodes=2,
        mix={"bfs": 1.0, "motif": 0.0, "bridge": 0.0},
        v_val=frozenset(),
        exclude_nodes=frozenset(),
    )
    plan = sampler.plan(seed=0, epoch=1, count=5)
    assert all(len(s.nodes) == 8 and len(set(s.nodes)) == 8 for s in plan.subgraphs)


def test_rejects_bad_construction() -> None:
    with pytest.raises(ValueError, match="nodes"):
        _sampler(nodes=3, background_nodes=0)
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b"])
    with pytest.raises(ValueError, match="fewer than"):
        StructSampler(
            graph,
            nodes=8,
            background_nodes=2,
            mix={"bfs": 1.0, "motif": 0.0, "bridge": 0.0},
            v_val=frozenset(),
            exclude_nodes=frozenset(),
        )


def test_statistics_and_coverage_keys() -> None:
    sampler = _sampler()
    plan = sampler.plan(seed=0, epoch=1, count=8)
    stats = sampler.statistics(plan.subgraphs[0])
    assert set(stats) == {
        "struct_nodes",
        "struct_components",
        "struct_legal_fraction",
        "struct_positive_fraction",
        "struct_triangles",
        "struct_open_wedges",
        "struct_background",
        "struct_kind_bfs",
        "struct_kind_motif",
        "struct_kind_bridge",
    }
    assert stats["struct_nodes"] == 24.0
    assert stats["struct_background"] == 4.0
    assert 0.0 < stats["struct_legal_fraction"] <= 1.0
    assert sum(stats[f"struct_kind_{k}"] for k in ("bfs", "motif", "bridge")) == 1.0
    coverage = sampler.coverage(plan)
    assert set(coverage) == {"struct_positive_coverage", "struct_positive_reuse"}
    assert 0.0 < coverage["struct_positive_coverage"] <= 1.0
    assert coverage["struct_positive_reuse"] >= 1.0


def test_statistics_count_triangles_and_wedges_exactly() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("x", "y")])
    graph.add_nodes_from(["p", "q"])
    sampler = StructSampler(
        graph,
        nodes=4,
        background_nodes=0,
        mix={"bfs": 1.0, "motif": 0.0, "bridge": 0.0},
        v_val=frozenset(),
        exclude_nodes=frozenset(),
    )
    subgraph = StructSubgraph(kind="bfs", nodes=("a", "b", "c", "d"), background=0)
    stats = sampler.statistics(subgraph)
    assert stats["struct_triangles"] == 1.0
    # Open wedges: a-c-d and b-c-d.
    assert stats["struct_open_wedges"] == 2.0
    assert stats["struct_components"] == 1.0
    assert stats["struct_positive_fraction"] == pytest.approx(4.0 / 6.0)
