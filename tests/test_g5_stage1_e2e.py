"""E2E liveness and five-arm gate contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from src import train_egostitch as te
from src.experiments import g5_stage1
from src.score_universe import ScoresArtifact, load_scores, save_scores

from tests.test_b0_cal import _toy_inputs as _b0cal_toy_inputs
from tests.test_g1_hardened_e2 import (
    _NODES,
    _POSITIVE_EDGES,
    _universe_rows,
    _write_universe_npz,
)
from tests.test_g5_stage1 import _d, _write_prereg

pytestmark = pytest.mark.unit

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
    scaffold_control: str = "none",
    permanent_null: str = "none",
    primary_logit: str = "full",
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
        "scaffold_control": {
            "mode": scaffold_control,
            "seed": 0,
            "keying": "canonical_pair_v1",
        },
        "permanent_null": permanent_null,
        "primary_logit": primary_logit,
    }
    arrays = {
        "full": full,
        "f_logit": f_logit,
        "pair_content": pair_content,
        "pair_topology": pair_topology,
    }
    save_scores(
        path,
        node_ids=node_ids,
        u_idx=u_idx,
        v_idx=v_idx,
        logit=arrays[primary_logit].astype(np.float32),
        label=labels.astype(np.int8),
        row_start=0,
        meta=meta,
        f_logit=f_logit.astype(np.float32),
        pair_content=pair_content.astype(np.float32),
        pair_topology=pair_topology.astype(np.float32),
        full_logit=full.astype(np.float32) if primary_logit != "full" else None,
    )


def _rewrite_e2e_artifact(
    path: Path,
    *,
    meta: dict[str, object] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    labels: np.ndarray | None = None,
) -> None:
    artifact = load_scores(path)
    resolved_pairs = pairs if pairs is not None else list(artifact.pairs())
    node_ids = list(artifact.node_ids)
    position = {node_id: i for i, node_id in enumerate(node_ids)}
    u_idx = np.asarray([position[node_u] for node_u, _ in resolved_pairs], dtype=np.int32)
    v_idx = np.asarray([position[node_v] for _, node_v in resolved_pairs], dtype=np.int32)
    save_scores(
        path,
        node_ids=node_ids,
        u_idx=u_idx,
        v_idx=v_idx,
        logit=artifact.logit,
        label=artifact.label if labels is None else labels,
        row_start=0,
        meta=artifact.meta,
        f_logit=artifact.f_logit,
        pair_content=artifact.pair_content,
        pair_topology=artifact.pair_topology,
        full_logit=artifact.full_logit,
    )
    if meta is not None:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
        np.savez_compressed(path, **arrays)


_E2E_LIVENESS_CONFIG = {
    "min_residual_std_ratio": 1e-5,
    "max_spearman": 0.9999,
    "max_topk_overlap": 0.9999,
    "topk_fraction": 0.01,
}


def _five_arm_inputs(tmp_path: Path) -> dict[str, Any]:
    """Toy benchmark + five egostitch_e2e arm universes + per-arm run metadata."""
    b0_universe_path, _val_path, data_root = _b0cal_toy_inputs(tmp_path)
    pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
    rng = np.random.default_rng(42)
    full = rng.normal(size=len(pairs))
    f_logit = rng.normal(size=len(pairs))  # independent of `full` -> alive residual
    pair_content = rng.normal(size=len(pairs))
    pair_topology = rng.normal(size=len(pairs))
    candidate_path = data_root / "benchmark_2025_neurips" / "toy" / "candidate_test_edges.txt"
    candidate_path.write_text(
        "".join(
            f"{node_u}\t{node_v}\t{int(label)}\n"
            for (node_u, node_v), label in zip(pairs, labels, strict=True)
        )
    )

    arm_paths: dict[str, Path] = {}
    for name in g5_stage1._E2E_ARMS:
        provenance = {
            "full": ("none", "none", "full"),
            "b0_e2e_f_only": ("none", "all_head", "f_logit"),
            "pair_topology": ("none", "content_head", "pair_topology"),
            "structure_control_6a": ("shuffle_within_pair", "none", "full"),
            "p0": ("none", "none", "full"),
        }[name]
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
            checkpoint_id=("ckpt_full" if name == "structure_control_6a" else f"ckpt_{name}"),
            scaffold_control=provenance[0],
            permanent_null=provenance[1],
            primary_logit=provenance[2],
        )
        arm_paths[name] = path

    preregistration_path = _write_prereg(tmp_path, b0_universe_path)
    config_root = Path(__file__).resolve().parents[1] / "configs"
    arm_config_paths = {
        "full": config_root / "egostitch_e2e_breadth_first.yaml",
        "b0_e2e_f_only": config_root / "egostitch_e2e_f_only_breadth_first.yaml",
        "pair_topology": (config_root / "egostitch_e2e_pair_topology_breadth_first.yaml"),
        "p0": config_root / "egostitch_e2e_p0_breadth_first.yaml",
    }
    preregistration = json.loads(preregistration_path.read_text())
    preregistration["benchmark"] = {"strategy": "toy"}
    preregistration["arms"] = {
        **{
            name: {
                "training": str(path),
                "scoring_provenance": {
                    "scaffold_control": {
                        "full": "none",
                        "b0_e2e_f_only": "none",
                        "pair_topology": "none",
                        "p0": "none",
                    }[name],
                    "permanent_null": {
                        "full": "none",
                        "b0_e2e_f_only": "all_head",
                        "pair_topology": "content_head",
                        "p0": "none",
                    }[name],
                    "primary_logit": {
                        "full": "full",
                        "b0_e2e_f_only": "f_logit",
                        "pair_topology": "pair_topology",
                        "p0": "full",
                    }[name],
                },
            }
            for name, path in arm_config_paths.items()
        },
        "structure_control_6a": {
            "training": "none (full checkpoint)",
            "scoring_provenance": {
                "scaffold_control": "shuffle_within_pair",
                "seed": 0,
                "keying": "canonical_pair_v1",
                "permanent_null": "none",
                "primary_logit": "full",
                "checkpoint_arm": "full",
            },
        },
    }
    preregistration["evaluator"] = {"seed": 0}
    preregistration_path.write_text(json.dumps(preregistration, sort_keys=True, indent=2) + "\n")
    preregistration_sha256 = hashlib.sha256(preregistration_path.read_bytes()).hexdigest()
    run_metadata_paths: dict[str, Path] = {}
    for name in ("full", "b0_e2e_f_only", "pair_topology", "p0"):
        permanent_null, p_topo, p_cont = {
            "full": ("none", 0.15, 0.15),
            "b0_e2e_f_only": ("all_head", 0.15, 0.15),
            "pair_topology": ("content_head", 0.15, 0.15),
            "p0": ("none", 0.0, 0.0),
        }[name]
        meta_path = tmp_path / f"{name}_run_metadata.json"
        meta_path.write_text(
            json.dumps(
                {
                    "preregistration_sha256": preregistration_sha256,
                    "checkpoint_id": f"ckpt_{name}",
                    "run_kind": "formal",
                    "formal_artifacts_published": True,
                    "status": "complete",
                    "model_family": "egostitch_e2e",
                    "permanent_null": permanent_null,
                    "p_topo": p_topo,
                    "p_cont": p_cont,
                    "seed": 0,
                    "strategy": "toy",
                    "partition_seed": 0,
                    "config_path": str(arm_config_paths[name].resolve()),
                    "config_hash": te._config_hash(te.load_config(arm_config_paths[name])),
                    "training_diagnostics": {
                        "fidelity_series": [{"topology_delta_std": 0.1}],
                        "gradient_norm_series": [
                            {
                                "step": 1,
                                "grad_rms_trunk": 0.1,
                                "grad_rms_ste": 0.1,
                                "grad_rms_content": 0.1,
                            }
                        ],
                        "kendall_fallback": {"active": False},
                    },
                }
            )
        )
        run_metadata_paths[name] = meta_path

    return {
        "arm_universe_paths": arm_paths,
        "run_metadata_paths": run_metadata_paths,
        "preregistration_path": preregistration_path,
        "data_root": data_root,
        "strategy": "toy",
    }


class TestBuildE2EArmSummary:
    def test_all_five_arms_reported_and_rerun_is_deterministic(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        payload = g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

        arms = _d(payload["arms"])
        assert set(arms) == set(g5_stage1._E2E_ARMS)
        expected_sha = hashlib.sha256(_d(inputs)["preregistration_path"].read_bytes()).hexdigest()
        assert payload["registration_sha256"] == expected_sha
        for name in g5_stage1._E2E_ARMS:
            row = _d(arms[name])
            expected_checkpoint = "ckpt_full" if name == "structure_control_6a" else f"ckpt_{name}"
            assert row["checkpoint_id"] == expected_checkpoint
            assert "graph_similarity" in _d(row["assembled"])
        # structure_control_6a has no run_metadata record, but inherits the
        # full arm's registered checkpoint/hash provenance.
        assert _d(arms["structure_control_6a"])["registration_sha256"] == expected_sha
        assert _d(arms["full"])["registration_sha256"] == expected_sha
        liveness = _d(payload["liveness"])
        assert liveness["residual_std"] > 0
        structure_control = _d(payload["structure_control"])
        assert structure_control["n_boot"] == 1000
        assert structure_control["seed"] == 0
        assert structure_control["passed"] is (structure_control["lower_bound"] > 0.0)
        decomposition = _d(payload["decomposition"])
        assert set(_d(decomposition["arms"])) == set(g5_stage1._E2E_ARMS)
        full_deltas = _d(_d(_d(decomposition["arms"])["full"])["deltas"])
        assert set(full_deltas) == {
            "full_minus_f_logit",
            "topology_delta_full_minus_pair_content",
            "content_delta_full_minus_pair_topology",
        }
        rerun = g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)
        assert json.dumps(payload, sort_keys=True) == json.dumps(rerun, sort_keys=True)

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

    def test_requires_exact_five_arm_set(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        inputs["arm_universe_paths"] = dict(inputs["arm_universe_paths"])
        inputs["arm_universe_paths"].pop("p0")
        with pytest.raises(ValueError, match="exactly the five registered arms"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_all_four_formal_run_metadata_records(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        inputs["run_metadata_paths"] = dict(inputs["run_metadata_paths"])
        inputs["run_metadata_paths"].pop("p0")
        with pytest.raises(ValueError, match="exactly the four formal"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    @pytest.mark.parametrize("registration", [None, ""])
    def test_requires_nonempty_shared_registration_hash(
        self, tmp_path: Path, registration: str | None
    ) -> None:
        inputs = _five_arm_inputs(tmp_path)
        metadata_path = _d(inputs["run_metadata_paths"])["p0"]
        payload = json.loads(metadata_path.read_text())
        payload.pop("preregistration_sha256")
        if registration is not None:
            payload["preregistration_sha256"] = registration
        metadata_path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="non-empty preregistration_sha256"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_formal_artifact_checkpoint_mismatch(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        metadata_path = _d(inputs["run_metadata_paths"])["pair_topology"]
        payload = json.loads(metadata_path.read_text())
        payload["checkpoint_id"] = "wrong"
        metadata_path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="checkpoint_id mismatch"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_6a_to_use_full_scoring_checkpoint(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        rng = np.random.default_rng(123)
        _write_e2e_universe_npz(
            _d(inputs["arm_universe_paths"])["structure_control_6a"],
            node_ids=_NODES,
            pairs=pairs,
            full=rng.normal(size=len(pairs)),
            f_logit=rng.normal(size=len(pairs)),
            pair_content=rng.normal(size=len(pairs)),
            pair_topology=rng.normal(size=len(pairs)),
            labels=labels,
            checkpoint_id="ckpt_not_full",
            scaffold_control="shuffle_within_pair",
        )
        with pytest.raises(ValueError, match="structure_control_6a checkpoint_id"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_registration_hash_mismatch_rejected(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        bad_meta = tmp_path / "bad_meta.json"
        payload = json.loads(_d(inputs)["run_metadata_paths"]["p0"].read_text())
        payload["preregistration_sha256"] = "b" * 64
        bad_meta.write_text(json.dumps(payload))
        inputs["run_metadata_paths"] = dict(inputs["run_metadata_paths"])
        inputs["run_metadata_paths"]["p0"] = bad_meta
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="does not match"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_binding_registration_status(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        preregistration_path = _d(inputs)["preregistration_path"]
        preregistration_path.write_text(json.dumps({"status": "DRAFT"}))

        with pytest.raises(g5_stage1.PreregistrationNotBinding, match="status == 'BINDING'"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_debug_run_metadata(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        metadata_path = _d(inputs)["run_metadata_paths"]["full"]
        metadata = json.loads(metadata_path.read_text())
        metadata.update({"run_kind": "debug", "formal_artifacts_published": False})
        metadata_path.write_text(json.dumps(metadata))

        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="debug/non-formal"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_completed_formal_run_metadata(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        metadata_path = _d(inputs)["run_metadata_paths"]["full"]
        metadata = json.loads(metadata_path.read_text())
        metadata.pop("formal_artifacts_published")
        metadata_path.write_text(json.dumps(metadata))

        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="exactly true"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_one_full_run_masquerading_as_all_formal_arms(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        full = _d(inputs)["run_metadata_paths"]["full"]
        inputs["run_metadata_paths"] = dict.fromkeys(g5_stage1._E2E_FORMAL_ARMS, full)
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="distinct run metadata"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_wrong_registered_arm_semantics(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        path = _d(inputs)["run_metadata_paths"]["p0"]
        metadata = json.loads(path.read_text())
        metadata["p_topo"] = 0.15
        path.write_text(json.dumps(metadata))
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="branch dropout"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("seed", 1, "seed must be 0"),
            ("strategy", "alternate", "strategy does not match"),
            ("partition_seed", 1, "partition_seed must be 0"),
            ("config_hash", "0" * 64, "config_hash does not match"),
        ],
    )
    def test_rejects_wrong_registered_run_identity(
        self, tmp_path: Path, field: str, value: object, match: str
    ) -> None:
        inputs = _five_arm_inputs(tmp_path)
        path = _d(inputs)["run_metadata_paths"]["full"]
        metadata = json.loads(path.read_text())
        metadata[field] = value
        path.write_text(json.dumps(metadata))
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match=match):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_wrong_registered_config_path(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        path = _d(inputs)["run_metadata_paths"]["full"]
        metadata = json.loads(path.read_text())
        metadata["config_path"] = json.loads(_d(inputs)["run_metadata_paths"]["p0"].read_text())[
            "config_path"
        ]
        path.write_text(json.dumps(metadata))
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="config_path"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_captured_registration_snapshot_is_not_reopened(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        preregistration_path = _d(inputs)["preregistration_path"]
        snapshot = g5_stage1._preregistration_snapshot(preregistration_path)
        preregistration_path.write_text(json.dumps({"status": "DRAFT"}))
        payload = g5_stage1.build_e2e_arm_summary(
            liveness_config=_E2E_LIVENESS_CONFIG,
            preregistration_snapshot=snapshot,
            **inputs,
        )
        assert payload["registration_sha256"] == snapshot[1]

    @pytest.mark.parametrize(
        ("arm", "field", "value", "match"),
        [
            ("full", "permanent_null", "all_head", "permanent_null"),
            ("p0", "primary_logit", "pair_topology", "primary_logit"),
            ("structure_control_6a", "control_seed", 1, "scaffold_control"),
            ("structure_control_6a", "control_keying", "wrong", "scaffold_control"),
            ("full", "control_mode", "shuffle_within_pair", "scaffold_control"),
        ],
    )
    def test_rejects_each_mutated_score_provenance_field(
        self,
        tmp_path: Path,
        arm: str,
        field: str,
        value: object,
        match: str,
    ) -> None:
        inputs = _five_arm_inputs(tmp_path)
        path = _d(inputs["arm_universe_paths"])[arm]
        artifact = load_scores(path)
        meta = dict(artifact.meta)
        if field.startswith("control_"):
            control = dict(cast(dict[str, object], meta["scaffold_control"]))
            control[field.removeprefix("control_")] = value
            meta["scaffold_control"] = control
        else:
            meta[field] = value
        _rewrite_e2e_artifact(path, meta=meta)
        with pytest.raises((ValueError, g5_stage1.RegistrationShaMismatch), match=match):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_candidate_pair_order_swap(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        path = _d(inputs["arm_universe_paths"])["p0"]
        artifact = load_scores(path)
        pairs = list(artifact.pairs())
        pairs[0], pairs[1] = pairs[1], pairs[0]
        labels = artifact.label.copy()
        labels[[0, 1]] = labels[[1, 0]]
        _rewrite_e2e_artifact(path, pairs=pairs, labels=labels)
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="pair identity/order"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_candidate_label_mutation(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        path = _d(inputs["arm_universe_paths"])["p0"]
        artifact = load_scores(path)
        labels = artifact.label.copy()
        labels[0] = 1 - labels[0]
        _rewrite_e2e_artifact(path, labels=labels)
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="candidate labels"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_binding_registration_rejects_b0cal_marker(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        preregistration_path = _d(inputs)["preregistration_path"]
        registration = json.loads(preregistration_path.read_text())
        registration["frozen_inputs"]["b0cal_results"]["sha256"] = "REQUIRED-BEFORE-BINDING"
        preregistration_path.write_text(json.dumps(registration))
        with pytest.raises(g5_stage1.PreregistrationMismatch, match="real b0cal_results"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_missing_submodule_rms_telemetry_fails_closed(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        metadata_path = _d(inputs["run_metadata_paths"])["full"]
        metadata = json.loads(metadata_path.read_text())
        del metadata["training_diagnostics"]["gradient_norm_series"][0]["grad_rms_content"]
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="submodule RMS telemetry"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_nested_worker_submodule_rms_shape_is_accepted(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        for metadata_path in _d(inputs["run_metadata_paths"]).values():
            metadata = json.loads(metadata_path.read_text())
            row = metadata["training_diagnostics"]["gradient_norm_series"][0]
            nested = {
                key: row.pop(key) for key in ("grad_rms_trunk", "grad_rms_ste", "grad_rms_content")
            }
            row["submodule_gradient_rms"] = nested
            metadata_path.write_text(json.dumps(metadata))
        g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_formal_evaluator_seed_must_be_zero(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        preregistration = json.loads(_d(inputs)["preregistration_path"].read_text())
        g5_stage1._enforce_e2e_evaluator_seed(preregistration, 0)
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="seed 0"):
            g5_stage1._enforce_e2e_evaluator_seed(preregistration, 1)

    def test_b0cal_comparator_path_and_digest_are_exact(self, tmp_path: Path) -> None:
        inputs = _five_arm_inputs(tmp_path)
        preregistration_path = _d(inputs)["preregistration_path"]
        preregistration = json.loads(preregistration_path.read_text())
        frozen = preregistration["frozen_inputs"]
        b0_path = Path(frozen["b0_candidate_scores"]["path"])
        b0cal_path = Path(frozen["b0cal_results"]["path"])
        assert (
            g5_stage1.enforce_e2e_frozen_inputs(
                preregistration, preregistration_path, b0_path, b0cal_path
            )
            == b0cal_path.resolve()
        )

        copy = tmp_path / "copied_b0cal_results.json"
        copy.write_bytes(b0cal_path.read_bytes())
        with pytest.raises(g5_stage1.PreregistrationMismatch, match="path mismatch"):
            g5_stage1.enforce_e2e_frozen_inputs(
                preregistration, preregistration_path, b0_path, copy
            )
        b0cal_path.write_text('{"mutated": true}\n')
        with pytest.raises(g5_stage1.PreregistrationMismatch, match="sha256 mismatch"):
            g5_stage1.enforce_e2e_frozen_inputs(
                preregistration, preregistration_path, b0_path, b0cal_path
            )


class TestPairedBootstrap:
    def test_b0cal_directory_resolves_one_committed_payload(self, tmp_path: Path) -> None:
        directory = tmp_path / "b0cal"
        directory.mkdir()
        expected = directory / "b0cal_results.json"
        expected.write_text("{}")
        assert g5_stage1._resolve_b0cal_results_path(directory) == expected
        (directory / "nested").mkdir()
        (directory / "nested" / "b0cal_results.json").write_text("{}")
        with pytest.raises(ValueError, match="exactly one"):
            g5_stage1._resolve_b0cal_results_path(directory)

    def test_lower_bound_is_paired_and_deterministic(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(0.0, 1.0, 500)
        clearly_higher = base + 1.0
        mean_stat = cast(Callable[[object], float], np.mean)

        lower_bound = g5_stage1.paired_bootstrap_lower_bound(mean_stat, clearly_higher, base)

        assert lower_bound > 0.0
        assert lower_bound == g5_stage1.paired_bootstrap_lower_bound(
            mean_stat, clearly_higher, base
        )
        null_lower_bound = g5_stage1.paired_bootstrap_lower_bound(mean_stat, base + 0.001, base)
        assert null_lower_bound < 0.0 or abs(null_lower_bound) < 0.1


class TestE2EGateCli:
    def test_mode_e2e_forwards_exact_five_arm_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inputs = _five_arm_inputs(tmp_path)
        captured: dict[str, object] = {}
        b0_path = tmp_path / "b0.npz"
        b0cal_path = tmp_path / "b0cal.json"
        probe_path = tmp_path / "probe.npz"
        for path in (b0_path, b0cal_path, probe_path):
            path.write_bytes(b"placeholder")

        def _fake_run(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {}

        monkeypatch.setattr(g5_stage1, "run_g5_e2e_stage1_pipeline", _fake_run)
        g5_stage1.main(
            [
                "--mode",
                "e2e",
                "--full-universe",
                str(_d(inputs)["arm_universe_paths"]["full"]),
                "--control-universe",
                str(_d(inputs)["arm_universe_paths"]["structure_control_6a"]),
                "--fonly-universe",
                str(_d(inputs)["arm_universe_paths"]["b0_e2e_f_only"]),
                "--pt-universe",
                str(_d(inputs)["arm_universe_paths"]["pair_topology"]),
                "--p0-universe",
                str(_d(inputs)["arm_universe_paths"]["p0"]),
                "--run-metadata",
                *[str(path) for path in _d(inputs)["run_metadata_paths"].values()],
                "--b0-universe",
                str(b0_path),
                "--b0cal-results",
                str(b0cal_path),
                "--probe-artifact",
                str(probe_path),
                "--preregistration",
                str(_d(inputs)["preregistration_path"]),
                "--output-dir",
                str(tmp_path / "gate"),
            ]
        )
        assert set(_d(captured["arm_universe_paths"])) == set(g5_stage1._E2E_ARMS)
        assert set(_d(captured["run_metadata_paths"])) == set(g5_stage1._E2E_FORMAL_ARMS)
        assert captured["probe_artifact_path"] == probe_path

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
