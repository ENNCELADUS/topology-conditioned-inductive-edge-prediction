"""Contracts for `src.distill.context_sampler.sample_context_sets`."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from src.distill.context_sampler import ContextSets, sample_context_sets

pytestmark = pytest.mark.unit


def _chain_graph(n: int) -> nx.Graph:
    """A path graph n0-n1-...-n(n-1) plus one isolate, so 2-hop sets are small."""
    graph = nx.path_graph([f"n{i}" for i in range(n)])
    graph.add_node("isolate")
    return graph


def _group(context: ContextSets, position: int) -> tuple[int, int]:
    """Return anchor `position`'s ``[start, end)`` row range."""
    return int(context.anchor_offsets[position]), int(context.anchor_offsets[position + 1])


def test_sampling_is_deterministic_given_the_same_seed() -> None:
    graph = _chain_graph(12)
    node_ids = sorted(graph.nodes)

    first = sample_context_sets(graph, node_ids, seed=7, k_near=2, k_rand=2)
    second = sample_context_sets(graph, node_ids, seed=7, k_near=2, k_rand=2)

    np.testing.assert_array_equal(first.anchor_idx, second.anchor_idx)
    np.testing.assert_array_equal(first.partner_idx, second.partner_idx)
    np.testing.assert_array_equal(first.anchor_offsets, second.anchor_offsets)
    np.testing.assert_array_equal(first.is_near, second.is_near)


def test_different_seeds_generally_diverge() -> None:
    graph = _chain_graph(30)
    node_ids = sorted(graph.nodes)

    first = sample_context_sets(graph, node_ids, seed=1, k_near=3, k_rand=3)
    second = sample_context_sets(graph, node_ids, seed=2, k_near=3, k_rand=3)

    assert not np.array_equal(first.partner_idx, second.partner_idx)


def test_near_partners_are_within_two_hops_and_exclude_the_anchor() -> None:
    graph = _chain_graph(20)
    node_ids = sorted(graph.nodes)
    context = sample_context_sets(graph, node_ids, seed=0, k_near=5, k_rand=0)

    for anchor_position, anchor in enumerate(node_ids):
        start, end = _group(context, anchor_position)
        hop_lengths = nx.single_source_shortest_path_length(graph, anchor, cutoff=2)
        for row in range(start, end):
            partner = node_ids[context.partner_idx[row]]
            assert partner != anchor
            assert partner in hop_lengths


def test_anchor_never_selected_as_its_own_random_partner() -> None:
    graph = _chain_graph(15)
    node_ids = sorted(graph.nodes)
    context = sample_context_sets(graph, node_ids, seed=3, k_near=0, k_rand=6)

    for anchor_position, anchor in enumerate(node_ids):
        start, end = _group(context, anchor_position)
        for row in range(start, end):
            assert node_ids[context.partner_idx[row]] != anchor


def test_counts_are_respected_as_a_ceiling() -> None:
    # n5 in a path graph of 40 nodes has plenty of 2-hop and random candidates.
    graph = _chain_graph(40)
    node_ids = sorted(graph.nodes)
    context = sample_context_sets(graph, node_ids, seed=11, k_near=3, k_rand=4)

    anchor_position = node_ids.index("n20")
    start, end = _group(context, anchor_position)
    is_near_group = context.is_near[start:end]
    assert int(is_near_group.sum()) == 3
    assert int((is_near_group == 0).sum()) == 4


def test_all_indices_are_in_range() -> None:
    graph = _chain_graph(25)
    node_ids = sorted(graph.nodes)
    context = sample_context_sets(graph, node_ids, seed=5, k_near=4, k_rand=4)

    n = len(node_ids)
    assert context.anchor_idx.min() >= 0
    assert context.anchor_idx.max() < n
    assert context.partner_idx.min() >= 0
    assert context.partner_idx.max() < n
    assert context.anchor_offsets[-1] == len(context.anchor_idx)
    assert context.anchor_offsets[0] == 0
    assert np.all(np.diff(context.anchor_offsets) >= 0)


def test_isolate_anchor_gets_an_empty_near_group_but_kept_offsets() -> None:
    graph = _chain_graph(10)
    node_ids = sorted(graph.nodes)
    context = sample_context_sets(graph, node_ids, seed=2, k_near=3, k_rand=0)

    isolate_position = node_ids.index("isolate")
    start, end = _group(context, isolate_position)
    assert end == start


def test_rejects_a_node_universe_that_is_not_sorted() -> None:
    graph = _chain_graph(5)
    node_ids = sorted(graph.nodes, reverse=True)
    with pytest.raises(ValueError, match="sorted"):
        sample_context_sets(graph, node_ids, seed=0, k_near=1, k_rand=1)


def test_rejects_a_node_missing_from_truth_graph() -> None:
    graph = _chain_graph(5)
    node_ids = [*sorted(graph.nodes), "phantom"]
    with pytest.raises(ValueError, match="absent"):
        sample_context_sets(graph, node_ids, seed=0, k_near=1, k_rand=1)


def test_small_universe_caps_rand_context_without_crashing() -> None:
    graph = nx.Graph([("a", "b")])
    graph.add_node("c")
    node_ids = sorted(graph.nodes)
    # k_rand larger than the available (node_ids - anchor - near) pool.
    context = sample_context_sets(graph, node_ids, seed=0, k_near=0, k_rand=10)

    for anchor_position in range(len(node_ids)):
        start, end = _group(context, anchor_position)
        assert end - start <= len(node_ids) - 1


# --------------------------------------------------------------------------- forbidden_internal


def _relay_graph() -> nx.Graph:
    """`v_a` and `v_b` are 2 hops apart only via the non-member `relay`.

    `v_a` also has a direct (1-hop) non-member neighbor `x1`, and six more
    nodes `y0..y5` sit outside both the near reach of `v_a` and the
    `{v_a, v_b}` quarantined set, so they are the only legal random-pool
    candidates for `v_a` once `relay`/`x1` are claimed by its near pool.
    """
    graph = nx.Graph()
    graph.add_edges_from([("v_a", "relay"), ("relay", "v_b"), ("v_a", "x1")])
    graph.add_nodes_from([f"y{i}" for i in range(6)])
    return graph


def test_forbidden_internal_excludes_a_two_hop_relay_partner_from_near_and_random_pools() -> None:
    graph = _relay_graph()
    node_ids = sorted(graph.nodes)
    forbidden_internal = frozenset({"v_a", "v_b"})

    context = sample_context_sets(
        graph, node_ids, seed=0, k_near=10, k_rand=10, forbidden_internal=forbidden_internal
    )

    anchor_position = node_ids.index("v_a")
    start, end = _group(context, anchor_position)
    near_partners = {
        node_ids[context.partner_idx[row]] for row in range(start, end) if context.is_near[row]
    }
    random_partners = {
        node_ids[context.partner_idx[row]] for row in range(start, end) if not context.is_near[row]
    }

    # v_b is a <=2-hop candidate (via relay) that the quarantine must drop
    # from both pools; relay and x1 are non-members and stay eligible.
    assert "v_b" not in near_partners
    assert "v_b" not in random_partners
    assert near_partners == {"relay", "x1"}
    assert random_partners == {f"y{i}" for i in range(6)}


def test_forbidden_internal_leaves_a_non_member_anchor_unaffected() -> None:
    graph = _relay_graph()
    node_ids = sorted(graph.nodes)
    forbidden_internal = frozenset({"v_a", "v_b"})

    context = sample_context_sets(
        graph, node_ids, seed=0, k_near=10, k_rand=10, forbidden_internal=forbidden_internal
    )

    # `relay` is not itself a quarantined member, so pairing it with v_a/v_b
    # is a legal cross-boundary context row -- the guard must not touch it.
    anchor_position = node_ids.index("relay")
    start, end = _group(context, anchor_position)
    near_partners = {node_ids[context.partner_idx[row]] for row in range(start, end)}
    assert {"v_a", "v_b"} <= near_partners


def test_forbidden_internal_defaults_to_no_exclusion() -> None:
    graph = _relay_graph()
    node_ids = sorted(graph.nodes)

    with_default = sample_context_sets(graph, node_ids, seed=0, k_near=10, k_rand=10)
    with_empty = sample_context_sets(
        graph, node_ids, seed=0, k_near=10, k_rand=10, forbidden_internal=frozenset()
    )

    np.testing.assert_array_equal(with_default.partner_idx, with_empty.partner_idx)
