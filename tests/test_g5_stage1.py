"""Tests for src.experiments.g5_stage1: the pre-registered Stage-1 gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from src.experiments import b0_cal, g5_stage1
from src.experiments.g1_hardened_e2 import AssembledRow

from tests.test_b0_cal import _toy_inputs as _b0cal_toy_inputs
from tests.test_g1_hardened_e2 import (
    _NODES,
    _POSITIVE_EDGES,
    _universe_rows,
    _write_universe_npz,
)

pytestmark = pytest.mark.unit


def _d(x: object) -> dict[str, Any]:
    return cast(dict[str, Any], x)


_PREREG = {
    "registration_id": "toy-prereg",
    "primary_criteria": {"holm_alpha": 0.05},
    "failure_reading": "pre-registered failure reading text (verbatim)",
    "decision_rules_5_2_verbatim": ["rule one", "rule two"],
}


def _write_prereg(tmp_path: Path) -> Path:
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(_PREREG, sort_keys=True, indent=2) + "\n")
    return path


def _write_run_metadata(tmp_path: Path, name: str, prereg_path: Path) -> Path:
    sha = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
    path = tmp_path / name
    path.write_text(json.dumps({"preregistration_sha256": sha, "checkpoint_id": "abc"}))
    return path


def _gate_inputs(tmp_path: Path, *, n_seeds: int = 2) -> dict[str, Any]:
    """Toy benchmark + b0cal payload + egostitch universes + prereg binding."""
    universe_path, val_path, data_root = _b0cal_toy_inputs(tmp_path)
    b0cal_dir = tmp_path / "b0cal"
    b0_cal.run_b0_cal_pipeline(
        universe_path=universe_path,
        val_scores_path=val_path,
        data_root=data_root,
        strategy="toy",
        output_dir=b0cal_dir,
        seed=0,
        skip_perturbation_check=True,
    )
    prereg_path = _write_prereg(tmp_path)

    pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
    ego_paths: list[Path] = []
    metadata_paths: list[Path] = []
    for s in range(n_seeds):
        rng = np.random.default_rng(100 + s)
        logits = rng.normal(size=len(pairs)).astype(np.float32)
        path = tmp_path / f"ego_{s}.npz"
        _write_universe_npz(
            path,
            node_ids=_NODES,
            pairs=pairs,
            logits=logits,
            labels=labels,
            strategy="toy",
            model_family="egostitch",
            checkpoint_id=f"e60{s}",
        )
        ego_paths.append(path)
        metadata_paths.append(_write_run_metadata(tmp_path, f"run_metadata_{s}.json", prereg_path))

    return {
        "egostitch_universe_paths": ego_paths,
        "run_metadata_paths": metadata_paths,
        "b0_universe_path": universe_path,
        "b0cal_results_path": b0cal_dir / "b0cal_results.json",
        "preregistration_path": prereg_path,
        "data_root": data_root,
        "strategy": "toy",
    }


# --------------------------------------------------------------------------- statistics


class TestHolmStepDown:
    def test_hand_computed(self) -> None:
        # m = 3, alpha = 0.05: thresholds 0.0167 / 0.025 / 0.05 in p order.
        p = {"a": 0.001, "b": 0.02, "c": 0.04}
        out = g5_stage1.holm_step_down(p, 0.05)
        assert out == {"a": True, "b": True, "c": True}

    def test_failure_cascades_to_larger_p(self) -> None:
        p = {"a": 0.001, "b": 0.03, "c": 0.0201}
        # sorted: a (0.0167 ok), c=0.0201 vs 0.025 ok, b=0.03 vs 0.05 ok.
        assert g5_stage1.holm_step_down(p, 0.05) == {"a": True, "b": True, "c": True}
        p = {"a": 0.02, "b": 0.021, "c": 0.022}
        # a=0.02 vs 0.0167 fails -> all fail.
        assert g5_stage1.holm_step_down(p, 0.05) == {"a": False, "b": False, "c": False}

    def test_single_member(self) -> None:
        assert g5_stage1.holm_step_down({"only": 0.04}, 0.05) == {"only": True}


def _row(clustering: float, boot_std: float = 0.0, degree: float = 10.0) -> AssembledRow:
    return AssembledRow(
        threshold=0.5,
        mmd_ratio={"degree": degree, "clustering": clustering, "spectral": 15.0},
        raw_mmd2={},
        reference_mmd2={},
        graph_similarity=0.4,
        relative_density=0.5,
        per_size_graph_similarity={},
        per_size_relative_density={},
        self_loops_pred=0,
        self_loops_ref=0,
        bootstrap_mean={},
        bootstrap_std={"degree": 0.0, "clustering": boot_std, "spectral": 0.0},
    )


def _comparator(clustering: float, boot_std: float = 0.0) -> dict[str, object]:
    return {
        "mmd_ratio": {"degree": 12.0, "clustering": clustering, "spectral": 18.0},
        "bootstrap_std": {"degree": 0.0, "clustering": boot_std, "spectral": 0.0},
        "graph_similarity": 0.3,
        "relative_density": 0.4,
    }


class TestClusteringCriterion:
    def test_hand_math(self) -> None:
        ego_rows = [_row(8.0, boot_std=0.5), _row(9.0, boot_std=0.5)]
        comparators = {"b0": _comparator(12.0, boot_std=0.5), "cal": _comparator(11.0)}
        out = g5_stage1.clustering_criterion(ego_rows, comparators)
        # Best (lowest) comparator is cal at 11.0; diffs = [3.0, 2.0].
        assert out.best_comparator == "cal"
        assert out.mean_diff == pytest.approx(2.5)
        # SE^2 = mean(0.25 + 0, 0.25 + 0) + var([3,2], ddof=1)/2 = 0.25 + 0.25
        assert out.se == pytest.approx(np.sqrt(0.5))
        assert out.dominates_every_comparator is True
        assert out.ci_excludes_zero == (2.5 - 1.959963984540054 * out.se > 0)

    def test_dominance_requires_every_comparator(self) -> None:
        ego_rows = [_row(11.5)]
        comparators = {"b0": _comparator(12.0), "cal": _comparator(11.0)}
        out = g5_stage1.clustering_criterion(ego_rows, comparators)
        assert out.dominates_every_comparator is False


class TestMatchedRdCriterion:
    def test_hand_math(self) -> None:
        matched = {"b0": [0.5, 0.6], "cal": [0.45, 0.55]}
        comp = {"b0": 0.4, "cal": 0.42}
        out = g5_stage1.matched_rd_criterion("bfs_macro_gs", matched, comp)
        assert out.best_comparator == "cal"
        assert out.mean_diff == pytest.approx(np.mean([0.45 - 0.42, 0.55 - 0.42]))
        assert out.dominates_every_comparator is True

    def test_regression_fails_dominance(self) -> None:
        matched = {"b0": [0.3], "cal": [0.5]}
        comp = {"b0": 0.4, "cal": 0.42}
        out = g5_stage1.matched_rd_criterion("bfs_macro_rd", matched, comp)
        assert out.dominates_every_comparator is False


# --------------------------------------------------------------------------- enforcement


class TestPreregistrationEnforcement:
    def test_mismatch_refuses_before_metrics(self, tmp_path: Path) -> None:
        inputs = _gate_inputs(tmp_path, n_seeds=1)
        bad = tmp_path / "bad_metadata.json"
        bad.write_text(json.dumps({"preregistration_sha256": "0" * 64}))
        out_dir = tmp_path / "gate"
        with pytest.raises(g5_stage1.PreregistrationMismatch, match="held-out metrics"):
            g5_stage1.run_g5_stage1_pipeline(
                egostitch_universe_paths=inputs["egostitch_universe_paths"],
                run_metadata_paths=[bad],
                b0_universe_path=inputs["b0_universe_path"],
                b0cal_results_path=inputs["b0cal_results_path"],
                preregistration_path=inputs["preregistration_path"],
                data_root=inputs["data_root"],
                strategy="toy",
                output_dir=out_dir,
            )
        assert not (out_dir / "g5_stage1_results.json").exists()

    def test_missing_hash_refused(self, tmp_path: Path) -> None:
        prereg = _write_prereg(tmp_path)
        empty = tmp_path / "empty_metadata.json"
        empty.write_text("{}")
        with pytest.raises(g5_stage1.PreregistrationMismatch):
            g5_stage1.enforce_preregistration(prereg, [empty])

    def test_matching_hash_accepted(self, tmp_path: Path) -> None:
        prereg = _write_prereg(tmp_path)
        metadata = _write_run_metadata(tmp_path, "ok.json", prereg)
        sha = g5_stage1.enforce_preregistration(prereg, [metadata])
        assert sha == hashlib.sha256(prereg.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- pipeline


class TestRunG5Stage1Pipeline:
    def test_end_to_end_payload_shape(self, tmp_path: Path) -> None:
        inputs = _gate_inputs(tmp_path)
        out_dir = tmp_path / "gate"
        payload = g5_stage1.run_g5_stage1_pipeline(output_dir=out_dir, **inputs)

        assert (out_dir / "g5_stage1_results.json").exists()
        assert (out_dir / "g5_stage1_tables.md").exists()
        assert payload["verdict"] in ("pass", "cut")
        assert set(_d(payload["criteria"])) == {
            "clustering_mmd_ratio",
            "bfs_macro_gs",
            "bfs_macro_rd",
        }
        comparators = _d(payload["comparators"])
        assert set(comparators) == {
            "b0",
            "b0_cal_density",
            "b0_cal_selfdensity",
            "b0_cal_degseq",
        }
        realized = _d(payload["comparator_realized_non_self_edges"])
        assert set(realized) == set(comparators)
        guards = _d(payload["guards"])
        assert set(guards) == {"degree_mmd_non_regression", "matched_edge_auprc"}
        ego = _d(payload["egostitch"])
        assert len(ego["assembled"]) == 2
        assert len(ego["s0_logit_correlation"]) == 2
        # Random-logit egostitch arms should not beat the bar: verdict is cut,
        # and the pre-registered failure reading is written verbatim.
        if payload["verdict"] == "cut":
            assert payload["failure_reading"] == _PREREG["failure_reading"]
        # The gate never edits the decision rules.
        assert payload["decision_rules_5_2_verbatim"] == _PREREG["decision_rules_5_2_verbatim"]

    def test_byte_identical_reruns(self, tmp_path: Path) -> None:
        inputs = _gate_inputs(tmp_path)
        out_a, out_b = tmp_path / "gate_a", tmp_path / "gate_b"
        g5_stage1.run_g5_stage1_pipeline(output_dir=out_a, **inputs)
        g5_stage1.run_g5_stage1_pipeline(output_dir=out_b, **inputs)
        assert (out_a / "g5_stage1_results.json").read_bytes() == (
            out_b / "g5_stage1_results.json"
        ).read_bytes()
        assert (out_a / "g5_stage1_tables.md").read_bytes() == (
            out_b / "g5_stage1_tables.md"
        ).read_bytes()

    def test_seed_count_mismatch_rejected(self, tmp_path: Path) -> None:
        inputs = _gate_inputs(tmp_path, n_seeds=2)
        with pytest.raises(ValueError, match="one --run-metadata"):
            g5_stage1.run_g5_stage1_pipeline(
                egostitch_universe_paths=inputs["egostitch_universe_paths"],
                run_metadata_paths=inputs["run_metadata_paths"][:1],
                b0_universe_path=inputs["b0_universe_path"],
                b0cal_results_path=inputs["b0cal_results_path"],
                preregistration_path=inputs["preregistration_path"],
                data_root=inputs["data_root"],
                strategy="toy",
                output_dir=tmp_path / "gate",
            )

    def test_single_seed_caveat_disclosed(self, tmp_path: Path) -> None:
        inputs = _gate_inputs(tmp_path, n_seeds=1)
        payload = g5_stage1.run_g5_stage1_pipeline(output_dir=tmp_path / "gate", **inputs)
        assert _d(payload["metadata"])["single_seed_caveat"] is not None


class TestCli:
    def test_missing_input_errors(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            g5_stage1.main(
                [
                    "--egostitch-universe",
                    str(tmp_path / "missing.npz"),
                    "--run-metadata",
                    str(tmp_path / "missing.json"),
                    "--b0-universe",
                    str(tmp_path / "missing_b0.npz"),
                    "--b0cal-results",
                    str(tmp_path / "missing_cal.json"),
                    "--preregistration",
                    str(tmp_path / "missing_prereg.json"),
                    "--output-dir",
                    str(tmp_path / "out"),
                ]
            )
