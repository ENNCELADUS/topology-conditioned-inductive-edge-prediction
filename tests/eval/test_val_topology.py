"""Tests for src.eval.val_topology."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import expit
from src.data.val_region import ValRegionParams, ValRegionSplit, val_universe_arrays
from src.eval.assembly import assemble_graph, density_matched_threshold
from src.eval.graph_metrics import MMDConfig, evaluate_assembled_graph_with_reference
from src.eval.val_topology import (
    ValTopologyReference,
    ValTopologyResult,
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


_EDGE_LOGITS: dict[frozenset[str], float] = {
    frozenset({"a", "b"}): 5.0,
    frozenset({"a", "c"}): -1.0,
    frozenset({"a", "d"}): -2.0,
    frozenset({"b", "c"}): 4.0,
    frozenset({"b", "d"}): -3.0,
    frozenset({"c", "d"}): -4.0,
}
_SELF_LOGITS: dict[str, float] = {"a": -5.0, "b": -6.0, "c": -7.0, "d": 6.0}

# U = self rows + the two bucket-relevant non-self pairs (both buckets, {a,b}
# and {c,d}, are entirely covered by U); the complement is the rest.
_U_NON_SELF: tuple[frozenset[str], ...] = (frozenset({"a", "b"}), frozenset({"c", "d"}))
_COMPLEMENT: tuple[frozenset[str], ...] = (
    frozenset({"a", "c"}),
    frozenset({"a", "d"}),
    frozenset({"b", "c"}),
    frozenset({"b", "d"}),
)


def _u_rows(
    nodes: tuple[str, ...],
    edge_logits: dict[frozenset[str], float],
    self_logits: dict[str, float],
    non_self_pairs: tuple[frozenset[str], ...] = _U_NON_SELF,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    u_list: list[int] = []
    v_list: list[int] = []
    logit_list: list[float] = []
    for pair in non_self_pairs:
        a, b = sorted(pair)
        u_list.append(nodes.index(a))
        v_list.append(nodes.index(b))
        logit_list.append(edge_logits[pair])
    for node in nodes:
        u_list.append(nodes.index(node))
        v_list.append(nodes.index(node))
        logit_list.append(self_logits[node])
    return (
        np.array(u_list, dtype=np.int32),
        np.array(v_list, dtype=np.int32),
        np.array(logit_list, dtype=np.float64),
    )


def _complement_logits(
    edge_logits: dict[frozenset[str], float],
    complement_pairs: tuple[frozenset[str], ...] = _COMPLEMENT,
) -> NDArray[np.float64]:
    return np.array([edge_logits[pair] for pair in complement_pairs], dtype=np.float64)


class TestGraphSimilarityAndDensity:
    def test_perfect_logits_give_gs_one_rd_one_and_small_ratios(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _u_rows(reference.nodes, _EDGE_LOGITS, _SELF_LOGITS)
        sample_logits = _complement_logits(_EDGE_LOGITS)

        result = val_region_topology_metrics(
            u_idx=u_idx,
            v_idx=v_idx,
            logits=logits,
            sample_logits=sample_logits,
            complement_total=len(sample_logits),
            reference=reference,
        )

        assert isinstance(result, ValTopologyResult)
        assert result.metrics.gs == pytest.approx(1.0)
        assert result.metrics.rd == pytest.approx(1.0)
        assert result.metrics.degree_mmd == pytest.approx(0.0, abs=1e-9)
        assert result.metrics.clustering_mmd == pytest.approx(0.0, abs=1e-9)
        assert result.metrics.spectral_mmd == pytest.approx(0.0, abs=1e-9)

    def test_inverted_logits_give_gs_zero(self) -> None:
        reference = _toy_reference()
        inverted_edges = {pair: -value for pair, value in _EDGE_LOGITS.items()}
        inverted_self = {node: -value for node, value in _SELF_LOGITS.items()}
        u_idx, v_idx, logits = _u_rows(reference.nodes, inverted_edges, inverted_self)
        sample_logits = _complement_logits(inverted_edges)

        result = val_region_topology_metrics(
            u_idx=u_idx,
            v_idx=v_idx,
            logits=logits,
            sample_logits=sample_logits,
            complement_total=len(sample_logits),
            reference=reference,
        )

        assert result.metrics.gs == pytest.approx(0.0)


class TestEquivalenceWithOldSemantics:
    """When the sample is the whole complement, results match exhaustive scoring."""

    def test_equals_full_universe_density_match_and_assembly(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _u_rows(reference.nodes, _EDGE_LOGITS, _SELF_LOGITS)
        sample_logits = _complement_logits(_EDGE_LOGITS)

        result = val_region_topology_metrics(
            u_idx=u_idx,
            v_idx=v_idx,
            logits=logits,
            sample_logits=sample_logits,
            complement_total=len(sample_logits),
            reference=reference,
        )

        # Old-style: full non-self universe density-matched threshold, then
        # assemble every row (U and complement alike).
        full_u_idx, full_v_idx = val_universe_arrays(reference.nodes)
        full_logits = [
            _SELF_LOGITS[reference.nodes[u]]
            if u == v
            else _EDGE_LOGITS[frozenset({reference.nodes[u], reference.nodes[v]})]
            for u, v in zip(full_u_idx.tolist(), full_v_idx.tolist(), strict=True)
        ]
        full_probs = expit(np.array(full_logits, dtype=np.float64))
        non_self_mask = full_u_idx != full_v_idx
        expected_threshold = density_matched_threshold(
            full_probs[non_self_mask], reference.target_edges
        )
        full_pairs = [
            (reference.nodes[u], reference.nodes[v])
            for u, v in zip(full_u_idx, full_v_idx, strict=True)
        ]
        expected_graph = assemble_graph(
            full_pairs, full_probs, threshold=expected_threshold, nodes=reference.nodes
        )
        expected_report = evaluate_assembled_graph_with_reference(
            expected_graph, reference.bucket_ref, MMDConfig()
        )

        assert result.threshold == pytest.approx(expected_threshold)
        assert result.metrics.gs == pytest.approx(expected_report.graph_similarity)
        assert result.metrics.rd == pytest.approx(expected_report.relative_density)
        assert result.metrics.degree_mmd == pytest.approx(expected_report.mmd_ratio["degree"])
        assert result.metrics.clustering_mmd == pytest.approx(
            expected_report.mmd_ratio["clustering"]
        )
        assert result.metrics.spectral_mmd == pytest.approx(expected_report.mmd_ratio["spectral"])


class TestUOnlyAssembly:
    def test_high_complement_logit_raises_threshold_without_creating_edges(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _u_rows(reference.nodes, _EDGE_LOGITS, _SELF_LOGITS)
        baseline_sample = _complement_logits(_EDGE_LOGITS)

        baseline = val_region_topology_metrics(
            u_idx=u_idx,
            v_idx=v_idx,
            logits=logits,
            sample_logits=baseline_sample,
            complement_total=len(baseline_sample),
            reference=reference,
        )

        spiked_sample = baseline_sample.copy()
        spiked_sample[0] = 100.0  # (a, c) becomes near-certain; still out-of-ball.
        spiked = val_region_topology_metrics(
            u_idx=u_idx,
            v_idx=v_idx,
            logits=logits,
            sample_logits=spiked_sample,
            complement_total=len(spiked_sample),
            reference=reference,
        )

        assert spiked.threshold > baseline.threshold
        # (a, c) is outside both bucket node sets and is never assembled, so
        # the bucket-restricted metrics are unaffected by the spike.
        assert spiked.metrics == baseline.metrics


class TestAdmittedNonSelfFraction:
    def test_hand_computable_fraction(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _u_rows(reference.nodes, _EDGE_LOGITS, _SELF_LOGITS)
        sample_logits = _complement_logits(_EDGE_LOGITS)

        result = val_region_topology_metrics(
            u_idx=u_idx,
            v_idx=v_idx,
            logits=logits,
            sample_logits=sample_logits,
            complement_total=len(sample_logits),
            reference=reference,
        )

        # threshold admits (a, b) from U and (b, c) from the complement sample:
        # 2 of the 6 non-self pairs in the 4-node universe.
        assert result.admitted_non_self_fraction == pytest.approx(2.0 / 6.0)

    def test_complement_total_zero(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _u_rows(
            reference.nodes, _EDGE_LOGITS, _SELF_LOGITS, non_self_pairs=_U_NON_SELF + _COMPLEMENT
        )

        result = val_region_topology_metrics(
            u_idx=u_idx,
            v_idx=v_idx,
            logits=logits,
            sample_logits=np.array([], dtype=np.float64),
            complement_total=0,
            reference=reference,
        )

        assert result.admitted_non_self_fraction == pytest.approx(2.0 / 6.0)


class TestFailClosed:
    def test_non_finite_logit_raises(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _u_rows(reference.nodes, _EDGE_LOGITS, _SELF_LOGITS)
        sample_logits = _complement_logits(_EDGE_LOGITS)
        logits[0] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            val_region_topology_metrics(
                u_idx=u_idx,
                v_idx=v_idx,
                logits=logits,
                sample_logits=sample_logits,
                complement_total=len(sample_logits),
                reference=reference,
            )

    def test_non_finite_sample_logit_raises(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _u_rows(reference.nodes, _EDGE_LOGITS, _SELF_LOGITS)
        sample_logits = _complement_logits(_EDGE_LOGITS)
        sample_logits[0] = np.inf

        with pytest.raises(ValueError, match="non-finite"):
            val_region_topology_metrics(
                u_idx=u_idx,
                v_idx=v_idx,
                logits=logits,
                sample_logits=sample_logits,
                complement_total=len(sample_logits),
                reference=reference,
            )

    def test_length_mismatch_raises(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _u_rows(reference.nodes, _EDGE_LOGITS, _SELF_LOGITS)
        sample_logits = _complement_logits(_EDGE_LOGITS)

        with pytest.raises(ValueError, match="length mismatch"):
            val_region_topology_metrics(
                u_idx=u_idx,
                v_idx=v_idx,
                logits=logits[:-1],
                sample_logits=sample_logits,
                complement_total=len(sample_logits),
                reference=reference,
            )

    def test_negative_complement_total_raises(self) -> None:
        reference = _toy_reference()
        u_idx, v_idx, logits = _u_rows(reference.nodes, _EDGE_LOGITS, _SELF_LOGITS)
        sample_logits = _complement_logits(_EDGE_LOGITS)

        with pytest.raises(ValueError, match="complement_total"):
            val_region_topology_metrics(
                u_idx=u_idx,
                v_idx=v_idx,
                logits=logits,
                sample_logits=sample_logits,
                complement_total=-1,
                reference=reference,
            )
