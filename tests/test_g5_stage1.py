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
from src.score_universe import ScoresArtifact, save_scores

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
    "seeds": [0],
    "primary_criteria": {"decision_procedure": "single_seed_point_estimate_dominance"},
    "failure_reading": "pre-registered failure reading text (verbatim)",
    "decision_rules_5_2_verbatim": ["rule one", "rule two"],
    "fidelity_validity_gate": {
        "min_residual_std_ratio": 1e-5,
        "max_spearman": 0.9999,
        "max_topk_overlap": 0.9999,
        "topk_fraction": 0.25,
    },
}


def _write_prereg(tmp_path: Path, b0_path: Path | None = None) -> Path:
    path = tmp_path / "prereg.json"
    payload = dict(_PREREG)
    if b0_path is not None:
        g1_path = tmp_path / "g1_results.json"
        g3_path = tmp_path / "g3_results.json"
        g1_path.write_text("{}\n")
        g3_path.write_text("{}\n")
        payload["frozen_inputs"] = {
            "b0_candidate_scores": {
                "path": str(b0_path),
                "sha256": hashlib.sha256(b0_path.read_bytes()).hexdigest(),
                "checkpoint_id": "deadbeefcafefeed",
            },
            "g1_results": {
                "path": str(g1_path),
                "sha256": hashlib.sha256(g1_path.read_bytes()).hexdigest(),
            },
            "g3_results": {
                "path": str(g3_path),
                "sha256": hashlib.sha256(g3_path.read_bytes()).hexdigest(),
            },
        }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path


def _write_run_metadata(
    tmp_path: Path, name: str, prereg_path: Path, checkpoint_id: str = "abc"
) -> Path:
    sha = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "preregistration_sha256": sha,
                "checkpoint_id": checkpoint_id,
                "s0_checkpoint_id": "deadbeefcafefeed",
                "training_diagnostics": {
                    "fidelity_series": [{"residual_std": 0.1}],
                    "gradient_norm_series": [{"step": 50, "grad_norm_edge": 1.0}],
                    "kendall_fallback": {"active": False},
                },
            }
        )
    )
    return path


def _gate_inputs(tmp_path: Path, *, n_seeds: int = 1) -> dict[str, Any]:
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
    prereg_path = _write_prereg(tmp_path, universe_path)

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
        metadata_paths.append(
            _write_run_metadata(tmp_path, f"run_metadata_{s}.json", prereg_path, f"e60{s}")
        )

    fidelity_paths: list[Path] = []
    for s in range(n_seeds):
        path = tmp_path / f"fidelity_{s}.json"
        path.write_text(
            json.dumps(
                {
                    "degree_calibration_curve": [],
                    "slot_recall_at_k_train": {},
                    "slot_recall_at_k_test": {},
                    "slot_adjacency_clustering_correlation": {},
                    "s_channel_correlation": {},
                    "self_nonself": {},
                    "self_loop_rate": 0.0,
                    "proj_variance_trajectory": [],
                }
            )
        )
        fidelity_paths.append(path)
    cost_path = tmp_path / "cost.json"
    cost_path.write_text(
        json.dumps(
            {
                "per_node_cached": {"flops": 1.0, "wall_seconds": 1.0},
                "per_pair_marginal": {"flops": 1.0, "wall_seconds": 1.0},
                "candidate_universe": {"flops": 1.0, "wall_seconds": 1.0},
                "harmonization_rounds": 0,
            }
        )
    )

    return {
        "egostitch_universe_paths": ego_paths,
        "s0_universe_paths": [universe_path] * n_seeds,
        "run_metadata_paths": metadata_paths,
        "b0_universe_path": universe_path,
        "b0cal_results_path": b0cal_dir / "b0cal_results.json",
        "preregistration_path": prereg_path,
        "data_root": data_root,
        "strategy": "toy",
        "fidelity_report_paths": fidelity_paths,
        "cost_report_path": cost_path,
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


class TestMatchedGlobalRdExactQuota:
    def test_splits_boundary_tie_by_canonical_pair_order(self) -> None:
        selected = g5_stage1.select_matched_global_rd_rows(
            np.array([0.9, 0.8, 0.8, 0.1]),
            np.array([0, 1, 0, 2], dtype=np.int32),
            np.array([3, 3, 2, 3], dtype=np.int32),
            target_edges=2,
            reference_edges=100,
        )
        np.testing.assert_array_equal(selected.selected_rows, np.array([0, 2]))
        assert selected.realized_edges == 2
        assert selected.rd_gap == 0.0
        assert selected.boundary_score == pytest.approx(0.8)
        assert selected.boundary_tie_size == 2
        assert selected.selected_from_boundary_tie == 1
        assert selected.split_boundary_tie is True

    def test_self_pairs_use_boundary_score_but_do_not_consume_quota(self) -> None:
        probs = np.array([0.9, 0.8, 0.9, 0.7])
        u_idx = np.array([0, 1, 0, 2], dtype=np.int32)
        v_idx = np.array([3, 3, 0, 2], dtype=np.int32)
        pairs = [("a", "d"), ("b", "d"), ("a", "a"), ("c", "c")]
        selected = g5_stage1.select_matched_global_rd_rows(
            probs, u_idx, v_idx, target_edges=1, reference_edges=100
        )
        graph = g5_stage1.assemble_matched_global_rd_graph(
            pairs, probs, u_idx, v_idx, selected, ["a", "b", "c", "d"]
        )
        assert set(graph.edges()) == {("a", "a"), ("a", "d")}


class TestDeadResidualGuard:
    @staticmethod
    def _artifact(logits: np.ndarray) -> ScoresArtifact:
        return ScoresArtifact(
            node_ids=["a", "b", "c"],
            u_idx=np.array([0, 0, 1], dtype=np.int32),
            v_idx=np.array([0, 1, 2], dtype=np.int32),
            logit=logits.astype(np.float32),
            label=np.array([0, 1, 0], dtype=np.int8),
            meta={},
        )

    def test_rejects_pair_invariant_residual(self) -> None:
        s0 = self._artifact(np.array([-1.0, 0.0, 1.0]))
        ego = self._artifact(np.array([-1.25, -0.25, 0.75]))
        with pytest.raises(ValueError, match="dead residual"):
            g5_stage1.validate_dead_residual(
                ego,
                s0,
                min_residual_std_ratio=1e-5,
                max_spearman=0.9999,
                max_topk_overlap=0.9999,
                topk_fraction=1 / 3,
            )

    def test_accepts_alive_residual_even_when_small(self) -> None:
        s0 = self._artifact(np.array([-1.0, 0.0, 1.0]))
        ego = self._artifact(np.array([-0.99, -0.02, 1.03]))
        report = g5_stage1.validate_dead_residual(
            ego,
            s0,
            min_residual_std_ratio=1e-5,
            max_spearman=0.9999,
            max_topk_overlap=0.9999,
            topk_fraction=1 / 3,
        )
        assert report["residual_std"] > 0


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
    def test_frozen_b0_hash_drift_refused_before_metrics(self, tmp_path: Path) -> None:
        inputs = _gate_inputs(tmp_path, n_seeds=1)
        Path(inputs["b0_universe_path"]).write_bytes(b"drift")
        with pytest.raises(g5_stage1.PreregistrationMismatch, match="frozen input"):
            g5_stage1.run_g5_stage1_pipeline(output_dir=tmp_path / "gate", **inputs)

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
    def test_required_diagnostics_cannot_be_omitted(self, tmp_path: Path) -> None:
        inputs = _gate_inputs(tmp_path)
        inputs["fidelity_report_paths"] = []
        with pytest.raises(ValueError, match="fidelity report"):
            g5_stage1.run_g5_stage1_pipeline(output_dir=tmp_path / "gate", **inputs)

    def test_cost_report_is_required(self, tmp_path: Path) -> None:
        inputs = _gate_inputs(tmp_path)
        inputs["cost_report_path"] = None
        with pytest.raises(ValueError, match="cost report"):
            g5_stage1.run_g5_stage1_pipeline(output_dir=tmp_path / "gate", **inputs)

    def test_b0cal_lineage_mismatch_rejected(self, tmp_path: Path) -> None:
        inputs = _gate_inputs(tmp_path, n_seeds=1)
        path = Path(inputs["b0cal_results_path"])
        payload = json.loads(path.read_text())
        payload["metadata"]["artifacts"]["universe"]["checkpoint_id"] = "wrong"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="b0cal lineage mismatch"):
            g5_stage1.run_g5_stage1_pipeline(output_dir=tmp_path / "gate", **inputs)

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
        assert len(ego["assembled"]) == 1
        assert len(ego["s0_logit_correlation"]) == 1
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

    def test_single_seed_screening_scope_disclosed(self, tmp_path: Path) -> None:
        inputs = _gate_inputs(tmp_path)
        payload = g5_stage1.run_g5_stage1_pipeline(output_dir=tmp_path / "gate", **inputs)
        metadata = _d(payload["metadata"])
        assert payload["verdict"] in ("pass", "cut")
        assert metadata["evaluation_mode"] == "single_seed_screening"
        assert metadata["binding_verdict"] is True
        assert metadata["single_seed_caveat"] is not None
        assert all(value is None for value in _d(payload["holm_survives"]).values())
        assert all(isinstance(value, bool) for value in _d(payload["primary_pass"]).values())
        for criterion in _d(payload["criteria"]).values():
            assert criterion["se"] is None
            assert criterion["p_value"] is None
            assert criterion["ci_excludes_zero"] is None
        report = (tmp_path / "gate" / "g5_stage1_tables.md").read_text()
        assert "single-seed screening" in report
        assert "does not claim statistical significance" in report

    def test_unregistered_extra_seed_is_rejected(self, tmp_path: Path) -> None:
        inputs = _gate_inputs(tmp_path, n_seeds=2)
        with pytest.raises(ValueError, match="requires 1 seed artifact"):
            g5_stage1.run_g5_stage1_pipeline(output_dir=tmp_path / "gate", **inputs)


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


# --------------------------------------------------------------------------- e2e family:
# within-checkpoint liveness + five-arm summary (Task 15)


class TestWithinCheckpointLivenessGuard:
    @staticmethod
    def _artifact(full: np.ndarray, f_logit: np.ndarray) -> ScoresArtifact:
        n = len(full)
        return ScoresArtifact(
            node_ids=[f"n{i}" for i in range(n)],
            u_idx=np.arange(n, dtype=np.int32),
            v_idx=np.arange(n, dtype=np.int32),
            logit=full.astype(np.float32),
            label=np.zeros(n, dtype=np.int8),
            meta={"model_family": "egostitch_e2e"},
            f_logit=f_logit.astype(np.float32),
        )

    def test_fires_when_full_equals_f_logit(self) -> None:
        f_logit = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        artifact = self._artifact(f_logit.copy(), f_logit)
        with pytest.raises(ValueError, match="dead residual"):
            g5_stage1.validate_dead_residual_within_checkpoint(
                artifact,
                min_residual_std_ratio=1e-5,
                max_spearman=0.9999,
                max_topk_overlap=0.9999,
                topk_fraction=0.01,
            )

    def test_fires_on_pair_invariant_residual(self) -> None:
        # A tiny constant offset scales the residual so small that the
        # conjunctive rule still fires, mirroring the frozen-s0 dead-residual
        # test: the residual must genuinely vary with the pair, not merely be
        # small in magnitude.
        f_logit = np.linspace(-2.0, 2.0, 200)
        full = f_logit + 1e-9
        artifact = self._artifact(full, f_logit)
        with pytest.raises(ValueError, match="dead residual"):
            g5_stage1.validate_dead_residual_within_checkpoint(
                artifact,
                min_residual_std_ratio=1e-5,
                max_spearman=0.9999,
                max_topk_overlap=0.9999,
                topk_fraction=0.01,
            )

    def test_does_not_fire_on_decorrelated_full(self) -> None:
        rng = np.random.default_rng(0)
        f_logit = rng.normal(size=200)
        full = rng.normal(size=200)  # independent of f_logit -> genuinely alive residual
        artifact = self._artifact(full, f_logit)
        report = g5_stage1.validate_dead_residual_within_checkpoint(
            artifact,
            min_residual_std_ratio=1e-5,
            max_spearman=0.9999,
            max_topk_overlap=0.9999,
            topk_fraction=0.01,
        )
        assert report["residual_std"] > 0

    def test_accepts_alive_residual_even_when_small(self) -> None:
        f_logit = np.array([-1.0, 0.0, 1.0])
        full = np.array([-0.99, -0.02, 1.03])
        artifact = self._artifact(full, f_logit)
        report = g5_stage1.validate_dead_residual_within_checkpoint(
            artifact,
            min_residual_std_ratio=1e-5,
            max_spearman=0.9999,
            max_topk_overlap=0.9999,
            topk_fraction=1 / 3,
        )
        assert report["residual_std"] > 0

    def test_requires_f_logit_array(self) -> None:
        artifact = ScoresArtifact(
            node_ids=["a"],
            u_idx=np.array([0], dtype=np.int32),
            v_idx=np.array([0], dtype=np.int32),
            logit=np.array([0.0], dtype=np.float32),
            label=np.array([0], dtype=np.int8),
            meta={},
        )
        with pytest.raises(ValueError, match="f_logit"):
            g5_stage1.validate_dead_residual_within_checkpoint(
                artifact,
                min_residual_std_ratio=1e-5,
                max_spearman=0.9999,
                max_topk_overlap=0.9999,
                topk_fraction=0.01,
            )


def _write_e2e_universe_npz(
    path: Path,
    *,
    node_ids: list[str],
    pairs: list[tuple[str, str]],
    full: np.ndarray,
    f_logit: np.ndarray,
    pair_content: np.ndarray,
    pair_topology: np.ndarray,
    labels: np.ndarray,
    strategy: str = "toy",
    checkpoint_id: str = "e2e0",
) -> None:
    """Write a toy family-``egostitch_e2e`` four-array scores artifact."""
    position = {node_id: i for i, node_id in enumerate(node_ids)}
    u_idx = np.array([position[u] for u, _ in pairs], dtype=np.int32)
    v_idx = np.array([position[v] for _, v in pairs], dtype=np.int32)
    meta: dict[str, object] = {
        "checkpoint_id": checkpoint_id,
        "model_family": "egostitch_e2e",
        "pairs_source": "candidate",
        "strategy": strategy,
        "num_rows": len(pairs),
        "created_utc": "2026-07-17T00:00:00+00:00",
        "torch_version": "2.10.0",
        "score_precision": {
            "contract": "egostitch_e2e_pair_fp32_v1",
            "encode_autocast": "off",
            "pair_compute_dtype": "float32",
            "pair_autocast": False,
            "logit_storage_dtype": "float32",
        },
    }
    save_scores(
        path,
        node_ids=node_ids,
        u_idx=u_idx,
        v_idx=v_idx,
        logit=full.astype(np.float32),
        label=labels.astype(np.int8),
        row_start=0,
        meta=meta,
        f_logit=f_logit.astype(np.float32),
        pair_content=pair_content.astype(np.float32),
        pair_topology=pair_topology.astype(np.float32),
    )


_E2E_LIVENESS_CONFIG = {
    "min_residual_std_ratio": 1e-5,
    "max_spearman": 0.9999,
    "max_topk_overlap": 0.9999,
    "topk_fraction": 0.01,
}


def _five_arm_inputs(tmp_path: Path) -> dict[str, Any]:
    """Toy benchmark + five egostitch_e2e arm universes + per-arm run metadata."""
    _b0_universe_path, _val_path, data_root = _b0cal_toy_inputs(tmp_path)
    pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
    rng = np.random.default_rng(42)
    full = rng.normal(size=len(pairs))
    f_logit = rng.normal(size=len(pairs))  # independent of `full` -> alive residual
    pair_content = rng.normal(size=len(pairs))
    pair_topology = rng.normal(size=len(pairs))

    arm_paths: dict[str, Path] = {}
    for name in g5_stage1._E2E_ARMS:
        path = tmp_path / f"{name}.npz"
        _write_e2e_universe_npz(
            path,
            node_ids=_NODES,
            pairs=pairs,
            full=full,
            f_logit=f_logit,
            pair_content=pair_content,
            pair_topology=pair_topology,
            labels=labels,
            checkpoint_id=f"ckpt_{name}",
        )
        arm_paths[name] = path

    run_metadata_paths: dict[str, Path] = {}
    for name in ("full", "b0_e2e_f_only", "pair_topology", "p0"):
        meta_path = tmp_path / f"{name}_run_metadata.json"
        meta_path.write_text(json.dumps({"preregistration_sha256": "a" * 64}))
        run_metadata_paths[name] = meta_path

    return {
        "arm_universe_paths": arm_paths,
        "run_metadata_paths": run_metadata_paths,
        "data_root": data_root,
        "strategy": "toy",
    }


class TestBuildE2EArmSummary:
    def test_all_five_arms_reported(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        payload = g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

        arms = _d(payload["arms"])
        assert set(arms) == set(g5_stage1._E2E_ARMS)
        assert payload["registration_sha256"] == "a" * 64
        for name in g5_stage1._E2E_ARMS:
            row = _d(arms[name])
            assert row["checkpoint_id"] == f"ckpt_{name}"
            assert "graph_similarity" in _d(row["assembled"])
        # structure_control_6a never had its own run_metadata entry.
        assert _d(arms["structure_control_6a"])["registration_sha256"] is None
        assert _d(arms["full"])["registration_sha256"] == "a" * 64
        liveness = _d(payload["liveness"])
        assert liveness["residual_std"] > 0

    def test_byte_identical_reruns(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        first = g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)
        second = g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_rejects_unknown_arm(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        inputs["arm_universe_paths"] = dict(inputs["arm_universe_paths"])
        inputs["arm_universe_paths"]["bogus"] = inputs["arm_universe_paths"]["full"]
        with pytest.raises(ValueError, match="unrecognized"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_full_arm(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        inputs["arm_universe_paths"] = {
            k: v for k, v in inputs["arm_universe_paths"].items() if k != "full"
        }
        with pytest.raises(ValueError, match="'full' arm is required"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_registration_hash_mismatch_rejected(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        bad_meta = tmp_path / "bad_meta.json"
        bad_meta.write_text(json.dumps({"preregistration_sha256": "b" * 64}))
        inputs["run_metadata_paths"] = dict(inputs["run_metadata_paths"])
        inputs["run_metadata_paths"]["p0"] = bad_meta
        with pytest.raises(ValueError, match="disagree"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_wrong_model_family(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        wrong_path = tmp_path / "wrong_family.npz"
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        _write_universe_npz(
            wrong_path,
            node_ids=_NODES,
            pairs=pairs,
            logits=np.zeros(len(pairs), dtype=np.float32),
            labels=labels,
            strategy="toy",
            model_family="v3_1",
        )
        inputs["arm_universe_paths"] = dict(inputs["arm_universe_paths"])
        inputs["arm_universe_paths"]["p0"] = wrong_path
        with pytest.raises(ValueError, match="model_family"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)
