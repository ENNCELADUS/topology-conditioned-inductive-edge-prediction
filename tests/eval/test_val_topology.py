"""Tests for src.eval.val_topology."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import expit
from src.data.val_region import ValRegionParams, ValRegionSplit, val_universe_arrays
from src.eval.assembly import assemble_graph, density_matched_threshold
from src.eval.graph_metrics import (
    MMDConfig,
    evaluate_assembled_graph,
    evaluate_assembled_graph_with_reference,
    precompute_bucket_reference,
    strip_self_loops,
)
from src.eval.val_topology import (
    ValTopologyReference,
    build_val_topology_reference,
    val_region_topology_metrics,
)

pytestmark = pytest.mark.unit

Pair = tuple[str, str]


def _toy_split() -> ValRegionSplit:
    v_val = frozenset({"a", "b", "c", "d"})
    return ValRegionSplit(
        train_nodes=v_val,
        v_val=v_val,
        region_seeds=("a",),
        training_positives=frozenset(),
        training_negatives=(),
        val_positives=(("a", "b"), ("b", "c"), ("d", "d")),
        val_negatives=(),
        buckets={2: [{"a", "b"}, {"c", "d"}]},
        params=ValRegionParams(),
    )


def _toy_reference() -> ValTopologyReference:
    return build_val_topology_reference(_toy_split())


def _build_logits(
    nodes: tuple[str, ...], edge_logits: dict[frozenset[str], float], self_logits: dict[str, float]
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    u_idx, v_idx = val_universe_arrays(nodes)
    logits = []
    for u, v in zip(u_idx.tolist(), v_idx.tolist(), strict=True):
        if u == v:
            logits.append(self_logits[nodes[u]])
        else:
            logits.append(edge_logits[frozenset({nodes[u], nodes[v]})])
    return u_idx, v_idx, np.array(logits, dtype=np.float64)


_EDGE_LOGITS: dict[frozenset[str], float] = {
    frozenset({"a", "b"}): 5.0,
    frozenset({"a", "c"}): -1.0,
    frozenset({"a", "d"}): -2.0,
    frozenset({"b", "c"}): 4.0,
    frozenset({"b", "d"}): -3.0,
    frozenset({"c", "d"}): -4.0,
}
_SELF_LOGITS: dict[str, float] = {"a": -5.0, "b": -6.0, "c": -7.0, "d": 6.0}


class TestGraphSimilarityAndDensity:
    def test_perfect_logits_give_gs_one_rd_one_and_small_ratios(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _build_logits(reference.nodes, _EDGE_LOGITS, _SELF_LOGITS)

        metrics = val_region_topology_metrics(
            u_idx=u_idx, v_idx=v_idx, logits=logits, reference=reference
        )

        assert metrics.gs == pytest.approx(1.0)
        assert metrics.rd == pytest.approx(1.0)
        assert metrics.degree_mmd == pytest.approx(0.0, abs=1e-9)
        assert metrics.clustering_mmd == pytest.approx(0.0, abs=1e-9)
        assert metrics.spectral_mmd == pytest.approx(0.0, abs=1e-9)

    def test_inverted_logits_give_gs_zero(self) -> None:
        reference = _toy_reference()
        inverted_edges = {pair: -value for pair, value in _EDGE_LOGITS.items()}
        inverted_self = {node: -value for node, value in _SELF_LOGITS.items()}
        u_idx, v_idx, logits = _build_logits(reference.nodes, inverted_edges, inverted_self)

        metrics = val_region_topology_metrics(
            u_idx=u_idx, v_idx=v_idx, logits=logits, reference=reference
        )

        assert metrics.gs == pytest.approx(0.0)


class TestSelfPairConvention:
    """Self rows never affect the non-self assembly quota.

    Mirrors `g1_hardened_e2.run_threshold_sweep`'s convention (its docstring,
    lines 821-827): the assembly threshold is density-matched on non-self rows
    only, so a self row's logit can never change which non-self rows clear the
    quota -- but a self row that itself clears the (self-row-independent)
    threshold still assembles as a self-loop.
    """

    def test_self_row_logit_never_moves_the_non_self_quota(self) -> None:
        reference = _toy_reference()
        low_self = {"a": -50.0, "b": -60.0, "c": -70.0, "d": -80.0}
        high_self = {"a": 50.0, "b": 60.0, "c": 70.0, "d": 80.0}
        u_idx, v_idx, logits_low = _build_logits(reference.nodes, _EDGE_LOGITS, low_self)
        _, _, logits_high = _build_logits(reference.nodes, _EDGE_LOGITS, high_self)

        probs_low = expit(logits_low)
        probs_high = expit(logits_high)
        non_self_mask = u_idx != v_idx
        threshold_low = density_matched_threshold(probs_low[non_self_mask], reference.target_edges)
        threshold_high = density_matched_threshold(
            probs_high[non_self_mask], reference.target_edges
        )

        assert threshold_low == threshold_high

        pairs = [
            (reference.nodes[u], reference.nodes[v]) for u, v in zip(u_idx, v_idx, strict=True)
        ]
        g_pred_low = assemble_graph(
            pairs, probs_low, threshold=threshold_low, nodes=reference.nodes
        )
        g_pred_high = assemble_graph(
            pairs, probs_high, threshold=threshold_high, nodes=reference.nodes
        )

        non_self_low = {frozenset(e) for e in strip_self_loops(g_pred_low).edges()}
        non_self_high = {frozenset(e) for e in strip_self_loops(g_pred_high).edges()}
        assert non_self_low == non_self_high == {frozenset({"a", "b"}), frozenset({"b", "c"})}

        # No self row clears a very-low threshold; every self row clears a very-high one.
        assert nx.number_of_selfloops(g_pred_low) == 0
        assert nx.number_of_selfloops(g_pred_high) == 4


class TestEvaluateAssembledGraphWithReferenceEquivalence:
    def test_identical_to_evaluate_assembled_graph_on_same_inputs(self) -> None:
        g_ref = nx.erdos_renyi_graph(20, 0.25, seed=3)
        g_ref = nx.relabel_nodes(g_ref, {i: f"n{i:02d}" for i in g_ref.nodes()})
        g_pred = g_ref.copy()
        g_pred.remove_edges_from(list(g_pred.edges())[::3])
        rng = np.random.default_rng(3)
        nodes = list(g_ref.nodes())
        buckets: dict[int, list[set[str]]] = {
            5: [set(rng.choice(nodes, size=5, replace=False).tolist()) for _ in range(4)]
        }
        config = MMDConfig()

        expected = evaluate_assembled_graph(g_pred, g_ref, buckets, config)
        reference = precompute_bucket_reference(g_ref, buckets, config)
        actual = evaluate_assembled_graph_with_reference(g_pred, reference, config)

        assert actual == expected


class TestFailClosed:
    def test_non_finite_logit_raises(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _build_logits(reference.nodes, _EDGE_LOGITS, _SELF_LOGITS)
        logits[0] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            val_region_topology_metrics(
                u_idx=u_idx, v_idx=v_idx, logits=logits, reference=reference
            )

    def test_length_mismatch_raises(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _build_logits(reference.nodes, _EDGE_LOGITS, _SELF_LOGITS)

        with pytest.raises(ValueError, match="length mismatch"):
            val_region_topology_metrics(
                u_idx=u_idx, v_idx=v_idx, logits=logits[:-1], reference=reference
            )
