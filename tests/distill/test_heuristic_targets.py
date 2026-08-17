"""Contracts for `src.distill.heuristic_targets`.

Covers the leave-edge-out CN/AA/RA math on a hand-built triangle+path graph
(exact values, worked by hand in each test's comment), the zero-denominator
term handling, the `graph_without_edge` removal/restore mechanism, and the
row-copy fidelity of the artifact-assembly step. `main()` (which needs a real
benchmark data root via `_load_val_region_split`) is not exercised here --
`compute_heuristic_logits` and the `write_kd_targets`/`load_kd_targets`
primitives it feeds are the same array-assembly/IO layer `main()` calls, and
are tested directly instead.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import pytest
from src.distill.artifacts import load_kd_targets, write_kd_targets
from src.distill.heuristic_targets import (
    HEURISTICS,
    _aa_term,
    _ra_term,
    build_parser,
    compute_heuristic_logits,
    graph_without_edge,
    heuristic_score,
)

pytestmark = pytest.mark.unit


def _triangle_and_path() -> nx.Graph:
    """Triangle a-b-c plus a path c-d-e-f hanging off c.

    Degrees: a=2, b=2, c=3 (a, b, d), d=2, e=2, f=1.
    """
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "f")])
    return graph


# --------------------------------------------------------------------------- term helpers


def test_aa_term_zero_denominator_contributes_zero() -> None:
    assert _aa_term(0) == 0.0
    assert _aa_term(1) == 0.0  # ln(1) == 0
    assert _aa_term(2) == pytest.approx(1.0 / math.log(2))


def test_ra_term_zero_denominator_contributes_zero() -> None:
    assert _ra_term(0) == 0.0
    assert _ra_term(1) == 0.0  # spec: deg(w) <= 1 contributes 0, even though 1/1 is defined
    assert _ra_term(2) == pytest.approx(0.5)


# --------------------------------------------------------------------------- graph_without_edge


def test_graph_without_edge_removes_and_restores_a_present_edge() -> None:
    graph = _triangle_and_path()
    assert graph.has_edge("a", "b")
    with graph_without_edge(graph, "a", "b"):
        assert not graph.has_edge("a", "b")
        # unrelated edges are untouched mid-removal
        assert graph.has_edge("b", "c")
    assert graph.has_edge("a", "b")  # restored


def test_graph_without_edge_is_a_no_op_when_the_edge_is_absent() -> None:
    graph = _triangle_and_path()
    assert not graph.has_edge("a", "d")
    with graph_without_edge(graph, "a", "d"):
        assert not graph.has_edge("a", "d")
        assert graph.number_of_edges() == 6
    assert not graph.has_edge("a", "d")
    assert graph.number_of_edges() == 6


def test_graph_without_edge_restores_on_exception() -> None:
    graph = _triangle_and_path()
    with pytest.raises(RuntimeError), graph_without_edge(graph, "a", "b"):
        assert not graph.has_edge("a", "b")
        raise RuntimeError("boom")
    assert graph.has_edge("a", "b")


# --------------------------------------------------------------------------- heuristic_score


def test_cn_aa_ra_on_a_directly_adjacent_triangle_pair() -> None:
    # (a, b): edge present -> removed. N(a)\{b}={c}, N(b)\{a}={c}; common={c}.
    # deg(c) in the edge-removed graph is unaffected (c's edges are a, b, d;
    # only the a-b edge was removed) -> deg(c) == 3.
    graph = _triangle_and_path()
    assert heuristic_score(graph, "a", "b", "cn") == pytest.approx(1.0)
    assert heuristic_score(graph, "a", "b", "aa") == pytest.approx(1.0 / math.log(3))
    assert heuristic_score(graph, "a", "b", "ra") == pytest.approx(1.0 / 3.0)
    # the graph must be back to its original state after each call
    assert graph.has_edge("a", "b")
    assert graph.degree("c") == 3


def test_cn_aa_ra_on_a_non_adjacent_pair_across_the_path() -> None:
    # (a, d): no edge to remove. N(a)={b,c}, N(d)={c,e}; common={c}, deg(c)=3.
    graph = _triangle_and_path()
    assert heuristic_score(graph, "a", "d", "cn") == pytest.approx(1.0)
    assert heuristic_score(graph, "a", "d", "aa") == pytest.approx(1.0 / math.log(3))
    assert heuristic_score(graph, "a", "d", "ra") == pytest.approx(1.0 / 3.0)


def test_cn_aa_ra_leaf_pair_has_no_common_neighbors_after_removal() -> None:
    # (e, f): edge present -> removed. N(e)\{f}={d}, N(f)\{e}={} (f is a leaf);
    # common neighbors = {} for all three heuristics.
    graph = _triangle_and_path()
    for heuristic in HEURISTICS:
        assert heuristic_score(graph, "e", "f", heuristic) == pytest.approx(0.0)
    assert graph.has_edge("e", "f")  # restored


def test_self_pair_scores_zero_for_every_heuristic() -> None:
    graph = _triangle_and_path()
    for heuristic in HEURISTICS:
        assert heuristic_score(graph, "a", "a", heuristic) == 0.0


def test_heuristic_score_rejects_an_unknown_heuristic() -> None:
    graph = _triangle_and_path()
    with pytest.raises(ValueError, match="unknown heuristic"):
        heuristic_score(graph, "a", "b", "katz")


# ------------------------------------------------------------------- compute_heuristic_logits


def test_compute_heuristic_logits_applies_log1p_and_matches_heuristic_score() -> None:
    graph = _triangle_and_path()
    node_ids = ["a", "b", "c", "d", "e", "f"]
    pair_anchor_idx = np.array([0, 0], dtype=np.int32)  # (a,b), (a,d)
    pair_partner_idx = np.array([1, 3], dtype=np.int32)

    logits = compute_heuristic_logits(graph, node_ids, pair_anchor_idx, pair_partner_idx, "ra")

    expected = np.array(
        [
            math.log1p(heuristic_score(graph, "a", "b", "ra")),
            math.log1p(heuristic_score(graph, "a", "d", "ra")),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(logits, expected, atol=1e-6)
    assert logits.dtype == np.float32


def test_compute_heuristic_logits_raises_on_non_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.distill.heuristic_targets as ht

    graph = _triangle_and_path()
    monkeypatch.setattr(ht, "heuristic_score", lambda *args, **kwargs: float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        ht.compute_heuristic_logits(
            graph, ["a", "b"], np.array([0], dtype=np.int32), np.array([1], dtype=np.int32), "cn"
        )


def test_build_parser_accepts_the_documented_cli_signature() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--targets",
            "targets_v2",
            "--data-root",
            "data",
            "--strategy",
            "breadth_first",
            "--heuristic",
            "ra",
            "--output",
            "targets_heuristic_ra",
        ]
    )
    assert args.targets == Path("targets_v2")
    assert args.data_root == Path("data")
    assert args.strategy == "breadth_first"
    assert args.heuristic == "ra"
    assert args.output == Path("targets_heuristic_ra")


def test_build_parser_rejects_an_unknown_heuristic_choice() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--targets",
                "t",
                "--data-root",
                "d",
                "--heuristic",
                "katz",
                "--output",
                "o",
            ]
        )


# --------------------------------------------------------------------------- artifact assembly


def _write_v2_artifact(path: Path) -> None:
    rng = np.random.default_rng(0)
    write_kd_targets(
        path,
        node_ids=["a", "b", "c", "d", "e", "f"],
        pair_anchor_idx=np.array([0, 0, 4], dtype=np.int32),  # (a,b), (a,d), (e,f)
        pair_partner_idx=np.array([1, 3, 5], dtype=np.int32),
        anchor_offsets=np.array([0, 2, 2, 2, 2, 3, 3], dtype=np.int64),
        teacher_logit=np.array([0.5, -0.3, 1.2], dtype=np.float32),
        teacher_pooled_ab=rng.standard_normal((3, 6)).astype(np.float16),
        teacher_pooled_ba=rng.standard_normal((3, 6)).astype(np.float16),
        is_near=np.array([1, 0, 1], dtype=np.uint8),
        pair_label=np.array([1, 0, 1], dtype=np.int8),
        truth_graph_sha256="oracle-truth-sha",
        checkpoint_path=Path("oracle.pt"),
        checkpoint_sha256="oracle-sha",
        checkpoint_id="oracle123",
        k_near=2,
        k_rand=1,
        seed=0,
    )


def test_heuristic_artifact_assembly_copies_v2_rows_byte_identical_and_zeros_pooled(
    tmp_path: Path,
) -> None:
    """Mirrors `heuristic_targets.main()`'s artifact-assembly step directly."""
    v2_dir = tmp_path / "v2"
    _write_v2_artifact(v2_dir)
    v2 = load_kd_targets(v2_dir)

    graph = _triangle_and_path()
    teacher_logit = compute_heuristic_logits(
        graph, v2.node_ids, v2.pair_anchor_idx, v2.pair_partner_idx, "ra"
    )
    pooled_dim = v2.teacher_pooled_ab.shape[-1]
    n_pairs = len(v2.pair_anchor_idx)
    pooled_ab = np.zeros((n_pairs, pooled_dim), dtype=np.float16)
    pooled_ba = np.zeros((n_pairs, pooled_dim), dtype=np.float16)

    out_dir = tmp_path / "heuristic"
    write_kd_targets(
        out_dir,
        node_ids=v2.node_ids,
        pair_anchor_idx=v2.pair_anchor_idx,
        pair_partner_idx=v2.pair_partner_idx,
        anchor_offsets=v2.anchor_offsets,
        teacher_logit=teacher_logit,
        teacher_pooled_ab=pooled_ab,
        teacher_pooled_ba=pooled_ba,
        is_near=v2.is_near,
        pair_label=v2.pair_label,
        truth_graph_sha256="truth-sha",
        checkpoint_path=Path("heuristic:ra"),
        checkpoint_sha256="",
        checkpoint_id=None,
        k_near=cast(int, v2.manifest["k_near"]),
        k_rand=cast(int, v2.manifest["k_rand"]),
        seed=cast(int, v2.manifest["seed"]),
    )
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["teacher"] = "heuristic_ra"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    out = load_kd_targets(out_dir)
    # Row identity copied byte-identical from v2.
    assert out.node_ids == v2.node_ids
    np.testing.assert_array_equal(out.pair_anchor_idx, v2.pair_anchor_idx)
    np.testing.assert_array_equal(out.pair_partner_idx, v2.pair_partner_idx)
    np.testing.assert_array_equal(out.anchor_offsets, v2.anchor_offsets)
    np.testing.assert_array_equal(out.is_near, v2.is_near)
    np.testing.assert_array_equal(out.pair_label, v2.pair_label)
    # Pooled arrays are zeros with v2's shape/dtype, never v2's own pooled content.
    assert out.teacher_pooled_ab.shape == v2.teacher_pooled_ab.shape
    assert out.teacher_pooled_ab.dtype == np.float16
    assert out.teacher_pooled_ba.dtype == np.float16
    np.testing.assert_array_equal(out.teacher_pooled_ab, np.zeros_like(v2.teacher_pooled_ab))
    np.testing.assert_array_equal(out.teacher_pooled_ba, np.zeros_like(v2.teacher_pooled_ba))
    # teacher_logit is the heuristic's log1p score, not v2's oracle logit.
    assert not np.allclose(out.teacher_logit, v2.teacher_logit)
    assert out.manifest["format"] == "kd_targets_v2"  # no content_logit written
    assert out.manifest["teacher"] == "heuristic_ra"
    assert out.content_logit is None
