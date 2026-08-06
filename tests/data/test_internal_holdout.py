"""Tests for the current internal topology holdout."""

from __future__ import annotations

import hashlib
import itertools

import networkx as nx
import pytest
from src.data.internal_holdout import (
    _holdout_bfs,
    build_pair_label_manifest,
    canonical_pair_label_sha256,
    derive_internal_holdout,
)


def _path_edges(size: int) -> list[tuple[str, str]]:
    return [(f"n{i:03d}", f"n{i + 1:03d}") for i in range(size - 1)]


def test_partition_is_deterministic_disjoint_and_exhaustive() -> None:
    nodes = {f"n{i:03d}" for i in range(20)}
    edges = _path_edges(20)
    first = derive_internal_holdout(nodes, edges, holdout_size=4)
    second = derive_internal_holdout(
        reversed(tuple(sorted(nodes))), reversed(edges), holdout_size=4
    )

    assert first == second
    assert len(first.v_hold) == 8
    assert first.v_fit | first.v_hold == nodes
    assert not (first.v_fit & first.v_hold)
    assert first.overlap_proof.all_zero


def test_v_hold_is_exactly_the_union_of_the_two_historical_draws() -> None:
    """V_hold remains the union of the two deterministic BFS draws."""
    nodes = {f"n{i:03d}" for i in range(20)}
    edges = _path_edges(20)
    result = derive_internal_holdout(nodes, edges, holdout_size=4)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    qual_draw = frozenset(_holdout_bfs(graph, 4, "g5-v2-qual|"))
    remaining = graph.subgraph(nodes - qual_draw).copy()
    select_draw = frozenset(_holdout_bfs(remaining, 4, "g5-v2-select|"))

    assert not (qual_draw & select_draw)
    assert result.v_hold == qual_draw | select_draw
    assert result.v_fit == frozenset(nodes) - qual_draw - select_draw


def test_bfs_uses_largest_component_and_hash_ordered_frontiers() -> None:
    large = {"root", "a", "b", "c", "d", "e"}
    small = {"x", "y", "z"}
    edges = list(itertools.combinations(sorted(large), 2)) + [("x", "y"), ("y", "z")]
    result = derive_internal_holdout(large | small, edges, holdout_size=2)
    qual_seed = min(
        large,
        key=lambda node: (hashlib.sha256(f"g5-v2-qual|{node}".encode()).digest(), node),
    )

    assert qual_seed in result.v_hold
    # Both draws stay inside the largest component: the small one is never taken.
    assert result.v_hold <= large


def test_fit_topology_and_classification_use_the_same_interactions() -> None:
    nodes = {f"n{i:03d}" for i in range(12)}
    interactions = list(itertools.combinations(sorted(nodes), 2)) + [("n011", "n011")]
    result = derive_internal_holdout(nodes, interactions, holdout_size=2)

    assert all(u in result.v_fit and v in result.v_fit for u, v in result.training_interactions_fit)
    assert result.topology_fit == frozenset(
        (u, v) for u, v in result.training_interactions_fit if u != v
    )
    assert nx.number_of_selfloops(result.build_g_fit()) == 0
    assert set(result.build_g_fit().nodes) == result.v_fit
    assert ("n011", "n011") in result.training_interactions_fit if "n011" in result.v_fit else True


def test_cross_partition_quarantine_counts_match_direct_count() -> None:
    nodes = {f"n{i:03d}" for i in range(14)}
    interactions = list(itertools.combinations(sorted(nodes), 2))
    result = derive_internal_holdout(nodes, interactions, holdout_size=3)
    owner = {
        node: name
        for name, part in (("fit", result.v_fit), ("hold", result.v_hold))
        for node in part
    }
    expected = {"fit__hold": 0}
    for u, v in interactions:
        if owner[u] != owner[v]:
            expected["__".join(sorted((owner[u], owner[v])))] += 1

    assert result.quarantine_counts.training_interactions == expected


def test_complete_manifest_is_loopless_labeled_and_hashes_canonical_rows() -> None:
    nodes = ["c", "a", "b"]
    manifest = build_pair_label_manifest(nodes, [("b", "a"), ("c", "c")])

    assert manifest.nodes == ("a", "b", "c")
    assert manifest.pairs == (("a", "b"), ("a", "c"), ("b", "c"))
    assert manifest.labels == (1, 0, 0)
    assert manifest.positive_edges == (("a", "b"),)
    assert manifest.positive_count == 1
    assert manifest.prevalence == pytest.approx(1 / 3)
    expected = hashlib.sha256(b"a\tb\t1\na\tc\t0\nb\tc\t0\n").hexdigest()
    assert manifest.pair_labels_sha256 == expected
    reversed_digest = canonical_pair_label_sha256(
        reversed(manifest.pairs), reversed(manifest.labels)
    )
    assert reversed_digest == expected


def test_rejects_too_small_remaining_component_and_foreign_endpoints() -> None:
    with pytest.raises(ValueError, match="largest remaining"):
        derive_internal_holdout({"a", "b", "c"}, [("a", "b"), ("b", "c")], holdout_size=2)
    with pytest.raises(ValueError, match="outside operative"):
        derive_internal_holdout({"a", "b"}, [("a", "foreign")], holdout_size=1)
