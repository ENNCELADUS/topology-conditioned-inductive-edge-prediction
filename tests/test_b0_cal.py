"""Tests for src.experiments.b0_cal: the B0+cal calibrated-assembly kill-test."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from src.experiments import b0_cal
from src.experiments import g3_oracle as g3
from src.experiments.g1_hardened_e2 import AssembledRow
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

# Balanced toy val pairs (3 positives, 3 negatives) with informative logits.
# Deliberately NOT linearly separable (one positive below a negative): a
# separable fit set drives the NLL-optimal temperature to the lower bound,
# where sigmoid saturation collapses distinct logits into float ties and the
# monotone-identity sanity row no longer holds.
_VAL_PAIRS: list[tuple[str, str]] = [
    ("n1", "n2"),
    ("n1", "n3"),
    ("n2", "n3"),
    ("n1", "n8"),
    ("n5", "n7"),
    ("n6", "n8"),
]
_VAL_LABELS = np.array([1, 1, 0, 1, 0, 0], dtype=np.int8)
_VAL_LOGITS = np.array([2.0, 1.5, 1.0, -0.5, -1.0, -2.0], dtype=np.float32)


def _write_val_npz(
    path: Path,
    *,
    pairs_source: str = "val",
    checkpoint_id: str = "deadbeefcafefeed",
    labels: np.ndarray | None = None,
) -> None:
    _write_universe_npz(
        path,
        node_ids=_NODES,
        pairs=_VAL_PAIRS,
        logits=_VAL_LOGITS,
        labels=_VAL_LABELS if labels is None else labels,
        strategy="toy",
        pairs_source=pairs_source,
        checkpoint_id=checkpoint_id,
    )


def _toy_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write the toy benchmark, a deterministic universe artifact, and a val artifact."""
    g = _make_reference_graph()
    data_root = _write_benchmark(tmp_path, "toy", g, _BUCKETS)
    pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
    rng = np.random.default_rng(7)
    logits = rng.normal(size=len(pairs)).astype(np.float32)
    universe_path = tmp_path / "universe.npz"
    _write_universe_npz(
        universe_path, node_ids=_NODES, pairs=pairs, logits=logits, labels=labels, strategy="toy"
    )
    val_path = tmp_path / "val.npz"
    _write_val_npz(val_path)
    return universe_path, val_path, data_root


def _run(tmp_path: Path) -> dict[str, object]:
    universe_path, val_path, data_root = _toy_inputs(tmp_path)
    return b0_cal.run_b0_cal_pipeline(
        universe_path=universe_path,
        val_scores_path=val_path,
        data_root=data_root,
        strategy="toy",
        output_dir=tmp_path / "b0cal",
        seed=0,
        skip_perturbation_check=True,
    )


# --------------------------------------------------------------------------- validation


class TestValidateValArtifact:
    def test_rejects_wrong_pairs_source(self, tmp_path: Path) -> None:
        universe_path, val_path, data_root = _toy_inputs(tmp_path)
        bad_val = tmp_path / "bad_val.npz"
        _write_val_npz(bad_val, pairs_source="test")
        with pytest.raises(ValueError, match="pairs_source"):
            b0_cal.run_b0_cal_pipeline(
                universe_path=universe_path,
                val_scores_path=bad_val,
                data_root=data_root,
                strategy="toy",
                output_dir=tmp_path / "out",
                skip_perturbation_check=True,
            )

    def test_rejects_checkpoint_mismatch(self, tmp_path: Path) -> None:
        universe_path, val_path, data_root = _toy_inputs(tmp_path)
        bad_val = tmp_path / "bad_val.npz"
        _write_val_npz(bad_val, checkpoint_id="0123456789abcdef")
        with pytest.raises(ValueError, match="checkpoint_id"):
            b0_cal.run_b0_cal_pipeline(
                universe_path=universe_path,
                val_scores_path=bad_val,
                data_root=data_root,
                strategy="toy",
                output_dir=tmp_path / "out",
                skip_perturbation_check=True,
            )

    def test_rejects_unlabeled_val_rows(self, tmp_path: Path) -> None:
        universe_path, val_path, data_root = _toy_inputs(tmp_path)
        bad_val = tmp_path / "bad_val.npz"
        _write_val_npz(bad_val, labels=np.array([1, 1, 1, 0, 0, -1], dtype=np.int8))
        with pytest.raises(ValueError, match="labels"):
            b0_cal.run_b0_cal_pipeline(
                universe_path=universe_path,
                val_scores_path=bad_val,
                data_root=data_root,
                strategy="toy",
                output_dir=tmp_path / "out",
                skip_perturbation_check=True,
            )


# --------------------------------------------------------------------------- pipeline


class TestRunB0CalPipeline:
    def test_end_to_end_payload_shape(self, tmp_path: Path) -> None:
        payload = _run(tmp_path)
        assert (tmp_path / "b0cal" / "b0cal_results.json").exists()
        assert (tmp_path / "b0cal" / "b0cal_tables.md").exists()

        assembled = cast(dict[str, dict[str, object]], payload["assembled"])
        assert set(assembled) == {"b0", "b0_cal_density", "b0_cal_selfdensity", "b0_cal_degseq"}
        assert assembled["b0"]["threshold"] is not None
        assert assembled["b0_cal_density"]["threshold"] is not None
        assert assembled["b0_cal_selfdensity"]["threshold"] is None
        assert assembled["b0_cal_degseq"]["threshold"] is None

        calibration = _d(payload["calibration"])
        assert calibration["applied"] == "temperature"
        assert _d(calibration["temperature"])["temperature"] > 0
        assert "scale" in _d(calibration["platt"])

        headroom = cast(dict[str, dict[str, object]], payload["headroom"])
        assert set(headroom) == {"b0_cal_density", "b0_cal_selfdensity", "b0_cal_degseq"}
        for arm_headroom in headroom.values():
            ratios = _d(arm_headroom["mmd_ratio_headroom"])
            assert set(ratios) == {"degree", "clustering", "spectral"}

        quota_stats = _d(payload["quota_stats"])
        assert set(quota_stats) == {
            "realized_edges",
            "target_edges",
            "shortfall",
            "residual_quota",
        }
        assert quota_stats["shortfall"] >= 0

        # Without --g3-results the closure table and decision reading are absent.
        assert payload["gap_closure"] is None
        assert payload["decision"] is None

    def test_cal_density_reproduces_b0_row(self, tmp_path: Path) -> None:
        payload = _run(tmp_path)
        assembled = cast(dict[str, dict[str, object]], payload["assembled"])
        notes = _d(_d(payload["metadata"])["notes"])
        assert notes["b0_cal_density_identical_edge_set"] is True
        # Same edge set => identical assembled metrics (only the threshold differs).
        assert assembled["b0_cal_density"]["mmd_ratio"] == assembled["b0"]["mmd_ratio"]
        assert (
            assembled["b0_cal_density"]["graph_similarity"] == assembled["b0"]["graph_similarity"]
        )

    def test_degseq_quota_sum_matches_reference_degree_sum(self, tmp_path: Path) -> None:
        universe_path, val_path, data_root = _toy_inputs(tmp_path)
        payload = b0_cal.run_b0_cal_pipeline(
            universe_path=universe_path,
            val_scores_path=val_path,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "b0cal",
            skip_perturbation_check=True,
        )
        quota_stats = _d(payload["quota_stats"])
        # Reference simple graph has 5 edges -> degree sum 10 -> target 5.
        assert quota_stats["target_edges"] == len(_POSITIVE_EDGES)
        assembled = cast(dict[str, dict[str, object]], payload["assembled"])
        degseq_loops = cast(int, assembled["b0_cal_degseq"]["self_loops_pred"])
        assert degseq_loops >= 0  # self-pairs handled outside the quota system

    def test_gap_closure_and_decision_with_g3_results(self, tmp_path: Path) -> None:
        universe_path, val_path, data_root = _toy_inputs(tmp_path)
        g3_dir = tmp_path / "g3"
        g3.run_g3_pipeline(
            universe_path=universe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=g3_dir,
            seed=0,
            skip_perturbation_check=True,
        )
        payload = b0_cal.run_b0_cal_pipeline(
            universe_path=universe_path,
            val_scores_path=val_path,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "b0cal",
            seed=0,
            g3_results_path=g3_dir / "g3_results.json",
            skip_perturbation_check=True,
        )
        gap_closure = cast(dict[str, dict[str, object]], payload["gap_closure"])
        assert set(gap_closure) == {"b0_cal_density", "b0_cal_selfdensity", "b0_cal_degseq"}
        for closure in gap_closure.values():
            assert set(closure) == {
                "degree",
                "clustering",
                "spectral",
                "graph_similarity",
                "relative_density",
            }
        decision = _d(payload["decision"])
        assert decision["axis"] == "clustering"
        assert decision["reading"] in (
            "proceed_to_stage1",
            "halt_for_locked_decision_discussion",
        )
        # b0_cal_density closes exactly 0% of every axis (identical edge set).
        density_closure = _d(gap_closure["b0_cal_density"])
        for axis_value in density_closure.values():
            if axis_value is not None:
                assert axis_value == pytest.approx(0.0, abs=1e-12)
        # Same universe + seed as the G3 run => the recomputed b0 row must match.
        cross_check = _d(_d(payload["metadata"])["g3_b0_cross_check"])
        assert cross_check["matches"] is True

    def test_byte_identical_reruns(self, tmp_path: Path) -> None:
        universe_path, val_path, data_root = _toy_inputs(tmp_path)
        out_a = tmp_path / "run_a"
        out_b = tmp_path / "run_b"
        for out_dir in (out_a, out_b):
            b0_cal.run_b0_cal_pipeline(
                universe_path=universe_path,
                val_scores_path=val_path,
                data_root=data_root,
                strategy="toy",
                output_dir=out_dir,
                seed=0,
                skip_perturbation_check=True,
            )
        assert (out_a / "b0cal_results.json").read_bytes() == (
            out_b / "b0cal_results.json"
        ).read_bytes()
        assert (out_a / "b0cal_tables.md").read_bytes() == (out_b / "b0cal_tables.md").read_bytes()

    def test_calibration_improves_or_preserves_val_nll(self, tmp_path: Path) -> None:
        payload = _run(tmp_path)
        temp = _d(_d(payload["calibration"])["temperature"])
        assert temp["val_nll_after"] <= temp["val_nll_before"] + 1e-12


# --------------------------------------------------------------------------- gap closure math


class TestComputeGapClosure:
    def _g3_assembled(self) -> dict[str, dict[str, object]]:
        return {
            "b0": {
                "mmd_ratio": {"degree": 12.0, "clustering": 12.0, "spectral": 18.0},
                "graph_similarity": 0.3,
                "relative_density": 0.4,
            },
            "oracle_blend": {
                "mmd_ratio": {"degree": 8.0, "clustering": 4.0, "spectral": 9.0},
                "graph_similarity": 0.32,
                "relative_density": 0.65,
            },
            "oracle_topo": {
                "mmd_ratio": {"degree": 14.0, "clustering": 8.0, "spectral": 16.0},
                "graph_similarity": 0.5,
                "relative_density": 0.8,
            },
        }

    def _row(self, mmd_ratio: dict[str, float] | None = None) -> AssembledRow:
        return AssembledRow(
            threshold=None,
            mmd_ratio=(
                {"degree": 10.0, "clustering": 8.0, "spectral": 13.5}
                if mmd_ratio is None
                else mmd_ratio
            ),
            raw_mmd2={},
            reference_mmd2={},
            graph_similarity=0.4,
            relative_density=0.6,
            per_size_graph_similarity={},
            per_size_relative_density={},
            self_loops_pred=0,
            self_loops_ref=0,
            bootstrap_mean={},
            bootstrap_std={},
        )

    def test_half_closure_on_each_axis(self) -> None:
        closure = b0_cal.compute_gap_closure(self._row(), self._g3_assembled())
        assert closure["degree"] == pytest.approx((12.0 - 10.0) / (12.0 - 8.0))
        assert closure["clustering"] == pytest.approx(0.5)
        assert closure["spectral"] == pytest.approx(0.5)
        assert closure["graph_similarity"] == pytest.approx((0.4 - 0.3) / (0.5 - 0.3))
        assert closure["relative_density"] == pytest.approx(0.5)

    def test_zero_gap_yields_none(self) -> None:
        g3_assembled = self._g3_assembled()
        blend_ratio = cast(dict[str, float], g3_assembled["oracle_blend"]["mmd_ratio"])
        b0_ratio = cast(dict[str, float], g3_assembled["b0"]["mmd_ratio"])
        blend_ratio["degree"] = b0_ratio["degree"]
        closure = b0_cal.compute_gap_closure(self._row(), g3_assembled)
        assert closure["degree"] is None

    def test_negative_closure_when_arm_regresses(self) -> None:
        row = self._row(mmd_ratio={"degree": 13.0, "clustering": 14.0, "spectral": 20.0})
        closure = b0_cal.compute_gap_closure(row, self._g3_assembled())
        assert cast(float, closure["clustering"]) < 0.0


# --------------------------------------------------------------------------- CLI


class TestCli:
    def test_missing_universe_errors(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            b0_cal.main(
                [
                    "--universe",
                    str(tmp_path / "missing.npz"),
                    "--val-scores",
                    str(tmp_path / "missing_val.npz"),
                    "--output-dir",
                    str(tmp_path / "out"),
                ]
            )

    def test_end_to_end_cli(self, tmp_path: Path) -> None:
        universe_path, val_path, data_root = _toy_inputs(tmp_path)
        out_dir = tmp_path / "cli_out"
        b0_cal.main(
            [
                "--universe",
                str(universe_path),
                "--val-scores",
                str(val_path),
                "--data-root",
                str(data_root),
                "--strategy",
                "toy",
                "--output-dir",
                str(out_dir),
                "--skip-perturbation-check",
            ]
        )
        payload = json.loads((out_dir / "b0cal_results.json").read_text())
        assert set(_d(payload["assembled"])) == {
            "b0",
            "b0_cal_density",
            "b0_cal_selfdensity",
            "b0_cal_degseq",
        }
        # Artifact provenance is disclosed for both inputs.
        artifacts = _d(_d(payload["metadata"])["artifacts"])
        assert artifacts["universe"] is not None
        assert artifacts["val"] is not None
        assert artifacts["g3_results"] is None

    def test_loaded_artifact_round_trips(self, tmp_path: Path) -> None:
        universe_path, val_path, _ = _toy_inputs(tmp_path)
        universe = load_scores(universe_path)
        val = load_scores(val_path)
        assert universe.meta["pairs_source"] == "candidate"
        assert val.meta["pairs_source"] == "val"
        assert not math.isnan(float(np.sum(universe.probs())))
