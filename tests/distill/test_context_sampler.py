"""Strict LLP invariants for the KD2 context sampler."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from src.distill.context_sampler import (
    ContextBank,
    sample_context_bank,
    sample_context_banks,
    sample_v_val_context_bank,
)

pytestmark = pytest.mark.unit


def _sample(
    graph: nx.Graph,
    *,
    anchors: list[str] | None = None,
    nodes: list[str] | None = None,
    forbidden: frozenset[str] = frozenset(),
    seed: int = 0,
    epoch: int = 0,
    rw_step: int = 3,
    hops: int = 2,
    ns_rate: int = 1,
) -> ContextBank:
    node_ids = sorted(graph.nodes) if nodes is None else nodes
    return sample_context_bank(
        graph,
        anchor_ids=node_ids if anchors is None else anchors,
        node_ids=node_ids,
        forbidden_internal=forbidden,
        seed=seed,
        epoch=epoch,
        rw_step=rw_step,
        hops=hops,
        ns_rate=ns_rate,
    )


def _rows(bank: ContextBank, anchor_position: int) -> slice:
    return slice(
        int(bank.anchor_offsets[anchor_position]),
        int(bank.anchor_offsets[anchor_position + 1]),
    )


def test_default_bank_has_six_walk_visits_then_six_random_draws_per_anchor() -> None:
    graph = nx.cycle_graph([f"n{i}" for i in range(10)])
    bank = _sample(graph)

    for anchor_position in range(10):
        rows = _rows(bank, anchor_position)
        assert bank.is_near[rows].tolist() == [True] * 6 + [False] * 6


def test_sampling_is_deterministic_per_seed_epoch_and_anchor() -> None:
    graph = nx.complete_graph([f"n{i}" for i in range(20)])
    node_ids = sorted(graph.nodes)
    whole = _sample(graph, nodes=node_ids, seed=17, epoch=4)
    repeated = _sample(graph, nodes=node_ids, seed=17, epoch=4)
    subset = _sample(graph, anchors=["n7"], nodes=node_ids, seed=17, epoch=4)

    np.testing.assert_array_equal(whole.partner_idx, repeated.partner_idx)
    np.testing.assert_array_equal(whole.is_near, repeated.is_near)
    whole_rows = _rows(whole, node_ids.index("n7"))
    np.testing.assert_array_equal(whole.partner_idx[whole_rows], subset.partner_idx)
    np.testing.assert_array_equal(whole.is_near[whole_rows], subset.is_near)


def test_seed_and_epoch_each_change_draws() -> None:
    graph = nx.complete_graph([f"n{i}" for i in range(30)])
    baseline = _sample(graph, seed=1, epoch=2)
    new_seed = _sample(graph, seed=2, epoch=2)
    new_epoch = _sample(graph, seed=1, epoch=3)

    assert not np.array_equal(baseline.partner_idx, new_seed.partner_idx)
    assert not np.array_equal(baseline.partner_idx, new_epoch.partner_idx)


def test_walk_visits_keep_step_order_and_duplicates() -> None:
    graph = nx.Graph([("a", "b")])
    node_ids = ["a", "b"]
    bank = _sample(graph, anchors=["a"], nodes=node_ids, ns_rate=0)

    assert [node_ids[index] for index in bank.partner_idx] == ["b", "a"] * 3
    assert bank.is_near.tolist() == [True] * 6


def test_degree_zero_walk_pads_with_self_contexts() -> None:
    graph = nx.Graph()
    graph.add_node("isolate")
    bank = _sample(graph, ns_rate=0)

    assert bank.partner_idx.tolist() == [0] * 6
    assert bank.is_near.tolist() == [True] * 6


def test_illegal_walk_visits_are_dropped_without_resampling() -> None:
    graph = nx.Graph([("anchor", "featureless")])
    bank = _sample(
        graph,
        anchors=["anchor"],
        nodes=["anchor"],
        ns_rate=0,
    )

    # Each two-step walk visits featureless (illegal), then anchor (legal).
    assert bank.partner_idx.tolist() == [0, 0, 0]
    assert bank.is_near.tolist() == [True, True, True]


def test_v_val_internal_walk_visits_are_quarantined_but_cross_boundary_stays_legal() -> None:
    graph = nx.Graph([("v_a", "relay"), ("relay", "v_b")])
    node_ids = sorted(graph.nodes)
    bank = _sample(
        graph,
        anchors=["v_a"],
        nodes=node_ids,
        forbidden=frozenset({"v_a", "v_b"}),
        rw_step=40,
        ns_rate=0,
    )
    partners = [node_ids[index] for index in bank.partner_idx]

    assert partners == ["relay"] * 40
    assert "v_b" not in partners


def test_random_pool_excludes_all_v_val_nodes_for_a_v_val_anchor() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["outside", "v_a", "v_b"])
    node_ids = sorted(graph.nodes)
    bank = _sample(
        graph,
        anchors=["v_a"],
        nodes=node_ids,
        forbidden=frozenset({"v_a", "v_b"}),
        rw_step=3,
        hops=1,
        ns_rate=2,
    )
    random_partners = [
        node_ids[index]
        for index, is_near in zip(bank.partner_idx, bank.is_near, strict=True)
        if not is_near
    ]

    assert random_partners == ["outside"] * 6


def test_random_sampling_is_with_replacement_and_preserves_exact_q() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "only"])
    bank = _sample(
        graph,
        anchors=["a"],
        nodes=["a", "only"],
        forbidden=frozenset({"a"}),
        rw_step=2,
        hops=2,
        ns_rate=3,
    )

    assert int((~bank.is_near).sum()) == 12
    assert bank.partner_idx[~bank.is_near].tolist() == [1] * 12


def test_non_v_val_anchor_may_draw_v_val_nodes_and_itself() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["outside", "v"])
    bank = _sample(
        graph,
        anchors=["outside"],
        nodes=["outside", "v"],
        forbidden=frozenset({"v"}),
        rw_step=50,
        hops=1,
        ns_rate=20,
    )
    random_indices = set(bank.partner_idx[~bank.is_near].tolist())

    assert random_indices == {0, 1}


def test_anchor_subset_uses_indices_from_shared_node_order() -> None:
    graph = nx.path_graph(["a", "b", "c"])
    bank = _sample(graph, anchors=["c", "a"], nodes=["a", "b", "c"], ns_rate=0)

    assert bank.anchor_idx.tolist() == [2, 0]
    assert len(bank.anchor_offsets) == 3
    assert bank.anchor_offsets[-1] == len(bank.partner_idx)


def test_epoch_bank_helper_uses_epoch_index() -> None:
    graph = nx.complete_graph([f"n{i}" for i in range(20)])
    node_ids = sorted(graph.nodes)
    banks = sample_context_banks(
        graph,
        anchor_ids=node_ids,
        node_ids=node_ids,
        forbidden_internal=frozenset(),
        seed=5,
        n_banks=3,
    )

    assert len(banks) == 3
    assert not np.array_equal(banks[0].partner_idx, banks[1].partner_idx)


def test_v_val_diagnostic_bank_is_fixed_and_uses_only_feature_bearing_anchors() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["outside", "v_a", "v_b", "v_featureless"])
    node_ids = ["outside", "v_a", "v_b"]
    v_val = frozenset({"v_a", "v_b", "v_featureless"})

    first = sample_v_val_context_bank(
        graph, v_val=v_val, node_ids=node_ids, rw_step=1, hops=1, ns_rate=1
    )
    second = sample_v_val_context_bank(
        graph, v_val=v_val, node_ids=node_ids, rw_step=1, hops=1, ns_rate=1
    )

    assert first.anchor_idx.tolist() == [1, 2]
    np.testing.assert_array_equal(first.partner_idx, second.partner_idx)
    assert first.partner_idx[~first.is_near].tolist() == [0, 0]


def test_rejects_self_loops_and_an_empty_random_pool() -> None:
    looped = nx.Graph([("a", "a")])
    with pytest.raises(ValueError, match="loopless"):
        _sample(looped)

    isolate = nx.Graph()
    isolate.add_node("v")
    with pytest.raises(ValueError, match="no legal random-context pool"):
        _sample(isolate, forbidden=frozenset({"v"}))


def test_context_bank_validates_csr_shapes() -> None:
    with pytest.raises(ValueError, match="anchor count plus one"):
        ContextBank(
            anchor_idx=np.array([0], dtype=np.int32),
            anchor_offsets=np.array([0], dtype=np.int64),
            partner_idx=np.array([], dtype=np.int32),
            is_near=np.array([], dtype=np.bool_),
        )
