"""Tests for the S1 assembly-coherence diagnostic's pure pieces."""

import networkx as nx
import numpy as np
import pytest
from src.experiments.s1_assembly_coherence import (
    _CN_BETAS,
    apply_coupling,
    assemble_exact_n,
    cn_adjusted_scores,
    coupling_features,
    degree_quota_error,
    fit_ipf_offsets,
    largest_remainder_quotas,
    offset_adjusted_scores,
    scaled_rank_targets,
    validate_v_hold_artifact,
)
from src.score_universe import ScoresArtifact


def test_largest_remainder_quotas_preserves_sum_and_prioritizes_fractions() -> None:
    targets = np.array([2.6, 1.6, 0.5, 0.3])  # sum 5.0
    quotas = largest_remainder_quotas(targets)
    assert quotas.dtype == np.int64
    assert quotas.sum() == 5
    # Floors give [2, 1, 0, 0] = 3; the two remainders go to the largest
    # fractional parts (0.6 and 0.6, tie broken by ascending index).
    np.testing.assert_array_equal(quotas, [3, 2, 0, 0])
    np.testing.assert_array_equal(quotas, largest_remainder_quotas(targets))


def test_cn_betas_are_bidirectional() -> None:
    assert any(beta < 0 for beta in _CN_BETAS)
    assert any(beta > 0 for beta in _CN_BETAS)


def test_degree_quota_error_reports_per_node_gap() -> None:
    graph = nx.Graph([("a", "b"), ("a", "c")])
    graph.add_nodes_from(["d"])
    error = degree_quota_error(graph, {"a": 2, "b": 1, "c": 2, "d": 1})
    # Realized degrees: a=2, b=1, c=1, d=0 -> gaps 0, 0, 1, 1.
    assert error["l1"] == 2.0
    assert error["linf"] == 1.0
    assert error["exact_fraction"] == pytest.approx(0.5)


def _fake_v_hold_artifact(**meta_overrides: object) -> ScoresArtifact:
    meta: dict[str, object] = {
        "pairs_source": "v_hold",
        "checkpoint_id": "cafe",
        "num_rows": 3,
    }
    meta.update(meta_overrides)
    return ScoresArtifact(
        node_ids=["a", "b", "c"],
        u_idx=np.array([0, 0, 1], dtype=np.int32),
        v_idx=np.array([1, 2, 2], dtype=np.int32),
        logit=np.zeros(3, dtype=np.float32),
        label=np.array([1, 0, 1], dtype=np.int8),
        meta=meta,
    )


def test_validate_v_hold_artifact_accepts_matching() -> None:
    artifact = _fake_v_hold_artifact()
    validate_v_hold_artifact(
        artifact, expected_nodes={"a", "b", "c"}, expected_checkpoint_id="cafe"
    )


def test_validate_v_hold_artifact_rejects_drift() -> None:
    with pytest.raises(ValueError, match="pairs_source"):
        validate_v_hold_artifact(
            _fake_v_hold_artifact(pairs_source="candidate"),
            expected_nodes={"a", "b", "c"},
            expected_checkpoint_id="cafe",
        )
    with pytest.raises(ValueError, match="checkpoint"):
        validate_v_hold_artifact(
            _fake_v_hold_artifact(),
            expected_nodes={"a", "b", "c"},
            expected_checkpoint_id="beef",
        )
    with pytest.raises(ValueError, match="node"):
        validate_v_hold_artifact(
            _fake_v_hold_artifact(),
            expected_nodes={"a", "b", "z"},
            expected_checkpoint_id="cafe",
        )


def test_coupling_features_reads_only_the_assembled_graph() -> None:
    node_ids = ["a", "b", "c", "d"]
    logit_rows = np.array([0.5, -1.0, 0.0])
    u_idx = np.array([0, 0, 1], dtype=np.int32)
    v_idx = np.array([1, 2, 2], dtype=np.int32)
    assembled = nx.Graph([("a", "c"), ("b", "c"), ("a", "d")])
    features = coupling_features(logit_rows, u_idx, v_idx, assembled, node_ids)
    assert features.shape == (3, 5)
    np.testing.assert_allclose(features[:, 0], logit_rows)
    # CN: (a,b) share c; (a,c) and (b,c) share none.
    np.testing.assert_allclose(features[:, 1], [1.0, 0.0, 0.0])
    # Degrees in assembled graph: a=2, b=1, c=2, d=1.
    np.testing.assert_allclose(features[:, 3], [3.0, 4.0, 3.0])  # deg sums
    np.testing.assert_allclose(features[:, 4], [1.0, 0.0, 1.0])  # |deg diffs|


def test_apply_coupling_is_standardized_linear() -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    means = np.array([2.0, 3.0])
    stds = np.array([1.0, 1.0])
    coef = np.array([1.0, -1.0])
    scores = apply_coupling(features, coef=coef, intercept=0.5, means=means, stds=stds)
    np.testing.assert_allclose(scores, [(-1.0) - (-1.0) + 0.5, 1.0 - 1.0 + 0.5])


def _synthetic_logits(n: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=2.0, size=(n, n))
    logits = (logits + logits.T) / 2
    mask = ~np.eye(n, dtype=bool)
    return logits, mask


def test_fit_ipf_offsets_matches_degree_targets() -> None:
    logits, mask = _synthetic_logits()
    targets = np.array([3.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    offsets, info = fit_ipf_offsets(logits, mask, targets, max_iter=500)
    probs = 1.0 / (1.0 + np.exp(-(logits + offsets[:, None] + offsets[None, :])))
    expected = (probs * mask).sum(axis=1)
    np.testing.assert_allclose(expected, targets, rtol=0.02)
    assert info["max_rel_err"] < 0.02


def test_fit_ipf_offsets_drives_zero_targets_to_zero() -> None:
    logits, mask = _synthetic_logits()
    targets = np.array([3.0, 2.0, 2.0, 1.0, 1.0, 0.0])
    offsets, _ = fit_ipf_offsets(logits, mask, targets, max_iter=500)
    probs = 1.0 / (1.0 + np.exp(-(logits + offsets[:, None] + offsets[None, :])))
    expected = (probs * mask).sum(axis=1)
    assert expected[-1] < 0.05


def test_offset_adjusted_scores_is_symmetric_additive() -> None:
    logit_rows = np.array([0.5, -1.0, 2.0])
    u_idx = np.array([0, 1, 2], dtype=np.int32)
    v_idx = np.array([1, 2, 0], dtype=np.int32)
    offsets = np.array([0.1, 0.2, 0.4])
    forward = offset_adjusted_scores(logit_rows, u_idx, v_idx, offsets)
    swapped = offset_adjusted_scores(logit_rows, v_idx, u_idx, offsets)
    np.testing.assert_allclose(forward, swapped)
    np.testing.assert_allclose(forward, [0.5 + 0.3, -1.0 + 0.6, 2.0 + 0.5])


def test_scaled_rank_targets_rank_aligns_and_rescales() -> None:
    ranking = np.array([0.1, 5.0, 3.0, 3.0])
    multiset = [4, 3, 2, 1]
    targets = scaled_rank_targets(ranking, multiset, total=20.0)
    # Highest ranking score gets the largest multiset entry; the tie at 3.0
    # breaks by ascending position.
    np.testing.assert_allclose(targets, np.array([1.0, 4.0, 3.0, 2.0]) * 2.0)
    assert targets.sum() == 20.0


def test_cn_adjusted_scores_reads_only_the_given_graph() -> None:
    node_ids = ["a", "b", "c", "d"]
    logit_rows = np.zeros(3)
    u_idx = np.array([0, 0, 1], dtype=np.int32)
    v_idx = np.array([1, 2, 2], dtype=np.int32)
    triangle = nx.Graph([("a", "c"), ("b", "c"), ("a", "d")])
    empty = nx.Graph()
    empty.add_nodes_from(node_ids)
    with_cn = cn_adjusted_scores(logit_rows, u_idx, v_idx, triangle, node_ids, beta=1.0)
    without_cn = cn_adjusted_scores(logit_rows, u_idx, v_idx, empty, node_ids, beta=1.0)
    # (a, b) share neighbor c in the triangle graph; other rows share none.
    np.testing.assert_allclose(with_cn, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(without_cn, [0.0, 0.0, 0.0])


def test_assemble_exact_n_realizes_count_and_self_loops() -> None:
    pairs = [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")]
    u_idx = np.array([0, 0, 1, 2], dtype=np.int32)
    v_idx = np.array([1, 2, 2, 3], dtype=np.int32)
    scores = np.array([3.0, 2.0, 1.0, 0.5])
    graph = assemble_exact_n(
        pairs,
        u_idx,
        v_idx,
        scores,
        n=2,
        nodes=["a", "b", "c", "d"],
        self_loop_edges=[("d", "d")],
    )
    assert set(map(frozenset, graph.edges())) == {
        frozenset({"a", "b"}),
        frozenset({"a", "c"}),
        frozenset({"d"}),
    }
    assert nx.number_of_selfloops(graph) == 1
