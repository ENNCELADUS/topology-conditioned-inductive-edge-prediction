"""Tests for src.experiments.g3_oracle: the G3 Oracle gate pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from src.eval.graph_metrics import strip_self_loops
from src.experiments import g3_oracle as g3
from src.experiments.g1_hardened_e2 import (
    AssembledRow,
    common_neighbor_and_adamic_adar,
    run_g1_pipeline,
)
from src.score_universe import load_scores

from tests.test_g1_hardened_e2 import (
    _NODES,
    _POSITIVE_EDGES,
    _make_reference_graph,
    _universe_rows,
    _write_benchmark,
    _write_universe_npz,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- rank01 / rank01_lex


class TestRank01:
    def test_distinct_values(self) -> None:
        out = g3.rank01(np.array([3.0, 1.0, 2.0]))
        np.testing.assert_allclose(out, [1.0, 0.0, 0.5])

    def test_average_ties(self) -> None:
        # ascending ranks: the two 1.0s occupy ranks 0 and 1 -> mean 0.5
        out = g3.rank01(np.array([1.0, 2.0, 1.0, 3.0]))
        np.testing.assert_allclose(out, [0.5 / 3, 2.0 / 3, 0.5 / 3, 1.0])

    def test_all_equal_is_uninformative_half(self) -> None:
        out = g3.rank01(np.array([7.0, 7.0, 7.0]))
        np.testing.assert_allclose(out, [0.5, 0.5, 0.5])

    def test_single_and_empty(self) -> None:
        np.testing.assert_allclose(g3.rank01(np.array([42.0])), [0.5])
        assert g3.rank01(np.array([], dtype=np.float64)).size == 0

    def test_idempotent_on_its_own_output(self) -> None:
        values = np.array([1.0, 5.0, 5.0, 2.0, 1.0, 9.0])
        once = g3.rank01(values)
        np.testing.assert_allclose(g3.rank01(once), once)


class TestRank01Lex:
    def test_secondary_breaks_primary_ties(self) -> None:
        primary = np.array([1.0, 1.0, 0.0])
        secondary = np.array([5.0, 9.0, 9.0])
        # keys ascending: (0,9) -> rank 0, (1,5) -> rank 1, (1,9) -> rank 2
        out = g3.rank01_lex(primary, secondary)
        np.testing.assert_allclose(out, [0.5, 1.0, 0.0])

    def test_full_ties_share_mean_rank(self) -> None:
        primary = np.array([1.0, 1.0, 0.0])
        secondary = np.array([5.0, 5.0, 9.0])
        # keys: (0,9) -> 0; (1,5) twice -> ranks 1,2 -> mean 1.5
        out = g3.rank01_lex(primary, secondary)
        np.testing.assert_allclose(out, [0.75, 0.75, 0.0])

    def test_matches_rank01_of_strict_encoding(self) -> None:
        rng = np.random.default_rng(0)
        primary = rng.integers(0, 4, size=200).astype(np.float64)
        secondary = rng.integers(0, 4, size=200).astype(np.float64)
        encoded = primary * 10.0 + secondary  # strictly monotone in the lex key
        np.testing.assert_allclose(g3.rank01_lex(primary, secondary), g3.rank01(encoded))


# --------------------------------------------------------------------------- headroom


def _assembled_row(
    mmd_ratio: dict[str, float],
    graph_similarity: float,
    threshold: float | None = None,
) -> AssembledRow:
    zeros = dict.fromkeys(("degree", "clustering", "spectral"), 0.0)
    return AssembledRow(
        threshold=threshold,
        mmd_ratio=mmd_ratio,
        raw_mmd2=dict(zeros),
        reference_mmd2=dict(zeros),
        graph_similarity=graph_similarity,
        relative_density=1.0,
        per_size_graph_similarity={2: [graph_similarity]},
        per_size_relative_density={2: [1.0]},
        self_loops_pred=0,
        self_loops_ref=0,
        bootstrap_mean=dict(zeros),
        bootstrap_std=dict(zeros),
    )


class TestComputeHeadroom:
    def test_ratios_and_graph_similarity(self) -> None:
        b0 = _assembled_row({"degree": 12.0, "clustering": 9.0, "spectral": 18.0}, 1e-6)
        arm = _assembled_row({"degree": 3.0, "clustering": 4.5, "spectral": 2.0}, 1e-2)
        row = g3.compute_headroom(b0, arm)
        assert row.mmd_ratio_headroom == {"degree": 4.0, "clustering": 2.0, "spectral": 9.0}
        assert row.graph_similarity_ratio == pytest.approx(1e4)

    def test_zero_arm_ratio_yields_none(self) -> None:
        b0 = _assembled_row({"degree": 12.0, "clustering": 9.0, "spectral": 18.0}, 0.5)
        arm = _assembled_row({"degree": 0.0, "clustering": 4.5, "spectral": 2.0}, 0.75)
        row = g3.compute_headroom(b0, arm)
        assert row.mmd_ratio_headroom["degree"] is None
        assert row.mmd_ratio_headroom["clustering"] == 2.0
        assert row.graph_similarity_ratio == pytest.approx(1.5)

    def test_graph_similarity_ratio_none_when_b0_zero(self) -> None:
        ratios = {"degree": 1.0, "clustering": 1.0, "spectral": 1.0}
        assert (
            g3.compute_headroom(
                _assembled_row(ratios, 0.0), _assembled_row(ratios, 0.5)
            ).graph_similarity_ratio
            is None
        )


# --------------------------------------------------------------------------- pipeline


def _d(x: object) -> dict[str, Any]:
    """Cast a JSON-payload value known to be a dict, for concise test assertions."""
    return cast(dict[str, Any], x)


_BUCKETS: dict[int, list[set[str]]] = {
    4: [
        {"n1", "n2", "n3", "n4"},
        {"n5", "n6", "n7", "n8"},
        {"n1", "n3", "n5", "n7"},
        {"n2", "n4", "n6", "n8"},
    ]
}


def _toy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Write the 8-node toy benchmark + a deterministic universe artifact."""
    g = _make_reference_graph()
    data_root = _write_benchmark(tmp_path, "toy", g, _BUCKETS)
    pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
    rng = np.random.default_rng(7)
    logits = rng.normal(size=len(pairs)).astype(np.float32)
    universe_path = tmp_path / "universe.npz"
    _write_universe_npz(
        universe_path, node_ids=_NODES, pairs=pairs, logits=logits, labels=labels, strategy="toy"
    )
    return universe_path, data_root


class TestRunG3Pipeline:
    def test_end_to_end_payload_shape(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        out_dir = tmp_path / "g3"
        payload = g3.run_g3_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=out_dir,
            seed=0,
            skip_perturbation_check=True,
        )
        assert (out_dir / "g3_results.json").exists()
        for scorer in ("b0", "oracle_topo", "oracle_blend"):
            assert scorer in _d(payload["regime_table"])
            assert scorer in _d(payload["assembled"])
            assert scorer in _d(payload["self_pair_edge_metrics"])
        # b0 assembles by threshold; oracle arms by top-N (threshold null)
        assembled = cast(dict[str, dict[str, object]], payload["assembled"])
        assert assembled["b0"]["threshold"] is not None
        assert assembled["oracle_topo"]["threshold"] is None
        assert assembled["oracle_blend"]["threshold"] is None
        # headroom present for both arms with all three statistics
        headroom = cast(dict[str, dict[str, object]], payload["headroom"])
        for arm in ("oracle_topo", "oracle_blend"):
            ratios = cast(dict[str, object], headroom[arm]["mmd_ratio_headroom"])
            assert set(ratios) == {"degree", "clustering", "spectral"}
        assert _d(_d(payload["metadata"])["perturbation_check"])["skipped"] is True
        assert 0.0 <= cast(float, assembled["b0"]["graph_similarity"]) <= 1.0

    def test_oracle_arms_assemble_no_self_loops_and_exact_edge_count(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        payload = g3.run_g3_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "g3",
            seed=0,
            skip_perturbation_check=True,
        )
        assembled = cast(dict[str, dict[str, object]], payload["assembled"])
        for arm in ("oracle_topo", "oracle_blend"):
            assert assembled[arm]["self_loops_pred"] == 0
            # Global top-N matching does not force official per-subgraph RD to one;
            # this toy bucket includes nonempty-prediction / empty-reference cases.
            assert assembled[arm]["relative_density"] == float("inf")

    def test_oracle_topo_ranks_common_neighbor_pairs_first(self, tmp_path: Path) -> None:
        # In the toy graph, (n2,n3),(n1,n2),(n1,n3),(n2,n4),(n3,n4),(n1,n5)... have CN >= 1.
        # oracle_topo's top-5 must all be CN >= 1 pairs; scores must be the
        # rank01_lex of the row-aligned (CN, AA) values.
        universe_path, _data_root = _toy_inputs(tmp_path)
        g_simple = strip_self_loops(_make_reference_graph())

        universe = load_scores(universe_path)
        cn_m, aa_m = common_neighbor_and_adamic_adar(g_simple, universe.node_ids)
        cn_rows = cn_m[universe.u_idx, universe.v_idx]
        aa_rows = aa_m[universe.u_idx, universe.v_idx]
        expected = g3.rank01_lex(cn_rows, aa_rows)
        actual = g3.oracle_topo_scores(g_simple, universe)
        np.testing.assert_allclose(actual, expected)

    def test_b0_row_reproduces_g1_pipeline(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        g3_payload = g3.run_g3_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "g3",
            seed=0,
            skip_perturbation_check=True,
        )
        g1_payload = run_g1_pipeline(
            universe_path=universe_path,
            alt_universe_path=None,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "g1",
            seed=0,
            skip_perturbation_check=True,
        )
        assert _d(g3_payload["assembled"])["b0"] == _d(g1_payload["assembled"])["b0"]
        assert _d(g3_payload["regime_table"])["b0"] == _d(g1_payload["regime_table"])["b0"]

    def test_byte_identical_reruns(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        for out in (out_a, out_b):
            g3.run_g3_pipeline(
                universe_path=universe_path,
                data_root=data_root,
                strategy="toy",
                output_dir=out,
                seed=0,
                skip_perturbation_check=True,
            )
        assert (out_a / "g3_results.json").read_bytes() == (out_b / "g3_results.json").read_bytes()

    def test_validation_rejects_non_candidate_artifact(self, tmp_path: Path) -> None:
        g = _make_reference_graph()
        data_root = _write_benchmark(tmp_path, "toy", g, _BUCKETS)
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        universe_path = tmp_path / "bad.npz"
        _write_universe_npz(
            universe_path,
            node_ids=_NODES,
            pairs=pairs,
            logits=np.zeros(len(pairs), dtype=np.float32),
            labels=labels,
            strategy="toy",
            pairs_source="test",
        )
        with pytest.raises(ValueError, match="pairs_source"):
            g3.run_g3_pipeline(
                universe_path=universe_path,
                data_root=data_root,
                strategy="toy",
                output_dir=tmp_path / "g3",
                seed=0,
                skip_perturbation_check=True,
            )


# --------------------------------------------------------------------------- tables + CLI


class TestRenderTables:
    def test_sections_present(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        payload = g3.run_g3_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "g3",
            seed=0,
            skip_perturbation_check=True,
        )
        text = g3.render_tables_markdown(payload)
        for heading in (
            "# G3 Oracle gate tables",
            "## Regime table",
            "## Assembled-graph rows",
            "## Headroom (stop-rule view)",
            "## MMD ratio components",
            "## Noise floor",
        ):
            assert heading in text
        for scorer in ("b0", "oracle_topo", "oracle_blend"):
            assert scorer in text
        # tables file is written by the pipeline itself
        assert (tmp_path / "g3" / "g3_tables.md").read_text(encoding="utf-8") == text


class TestCli:
    def test_end_to_end(self, tmp_path: Path) -> None:
        universe_path, data_root = _toy_inputs(tmp_path)
        out_dir = tmp_path / "g3_cli"
        g3.main(
            [
                "--universe",
                str(universe_path),
                "--data-root",
                str(data_root),
                "--strategy",
                "toy",
                "--output-dir",
                str(out_dir),
                "--seed",
                "0",
                "--skip-perturbation-check",
            ]
        )
        payload = json.loads((out_dir / "g3_results.json").read_text(encoding="utf-8"))
        assert set(payload["headroom"]) == {"oracle_topo", "oracle_blend"}
        assert (out_dir / "g3_tables.md").exists()

    def test_missing_universe_errors(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            g3.main(
                [
                    "--universe",
                    str(tmp_path / "missing.npz"),
                    "--output-dir",
                    str(tmp_path / "out"),
                ]
            )
