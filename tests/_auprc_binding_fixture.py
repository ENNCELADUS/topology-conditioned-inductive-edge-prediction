"""Small exact v4 binding fixture shared by E2E entrance tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments import auprc_tolerance as calibration


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "byte_size": path.stat().st_size}


def _binding_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def bind_active_v4_calibration(
    registration: dict[str, Any],
    registration_path: Path,
    *,
    config_paths: dict[str, Path],
    tolerance: float = 0.02,
) -> Path:
    """Mutate a current-arm fixture into an exact active-v4/v2 binding."""
    sample_sd = tolerance - 0.00009
    empty_digest = hashlib.sha256(b"").hexdigest()
    evidence = registration["binding_evidence"]
    evidence.setdefault("implementation", {"commit": "a" * 40})
    bound_commit = str(evidence["implementation"]["commit"])
    source_commit = bound_commit + "a" * (40 - len(bound_commit))
    attempt_dir = registration_path.parent / "qualification" / "full" / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "source": attempt_dir / calibration.AUPRC_TOLERANCE_SOURCE_FILENAME,
        "run_metadata": attempt_dir / "run_metadata.json",
        "qualification": attempt_dir / "qualification.json",
        "artifact_manifest": attempt_dir / "artifact_manifest.json",
        "validation_events": attempt_dir / "v_hold_validation_events.jsonl",
        "history": attempt_dir.parents[1] / "attempt_history.json",
    }
    for path in artifact_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    replicates = np.linspace(-1.0, 1.0, calibration.REPLICATES, dtype="<f8")
    replicates *= sample_sd / float(np.std(replicates, ddof=1))
    sample_sd = float(np.std(replicates, ddof=1))
    replicate_path = attempt_dir / calibration.AP_REPLICATES_FILENAME
    with replicate_path.open("wb") as handle:
        np.save(handle, replicates, allow_pickle=False)
    source_metadata = {
        "schema": calibration.SOURCE_SCHEMA,
        "arm": "full",
        "seed": 0,
        "run_kind": "qualification",
        "validation_role": "V_hold",
        "validation_epoch": 2,
        "global_step": 2,
        "fixed_source_epoch": 2,
        "phase": "C",
        "full_joint_epochs": 1,
        "source_rule": calibration.SOURCE_RULE,
        "source_validation_reused": True,
        "source_validation_existing_event": True,
        "bootstrap_additional_v_hold_evaluations": 0,
        "pair_count": calibration.ROWS,
        "node_count": 512,
        "positive_count": calibration.POSITIVES,
        "negative_count": calibration.NEGATIVES,
        "pair_labels_sha256": "1" * 64,
        "nodes_sha256": "2" * 64,
        "positive_edges_sha256": "3" * 64,
        "score_transform": "none",
        "active_logits_dtype": "<f4",
        "labels_dtype": "int8",
        "active_logits_sha256": "4" * 64,
        "labels_sha256": "5" * 64,
    }
    artifact_paths["source"].write_bytes(b"fixture calibration source\n")
    if artifact_paths["validation_events"].exists():
        event_rows = [
            json.loads(line)
            for line in artifact_paths["validation_events"].read_text(encoding="utf-8").splitlines()
        ]
    else:
        event_rows = []
    if not any(
        isinstance(row, dict)
        and row.get("kind") == "epoch_end"
        and row.get("epoch") == 2
        and row.get("optimizer_step") == 2
        for row in event_rows
    ):
        event_rows.append(
            {
                "ordinal": len(event_rows) + 1,
                "kind": "epoch_end",
                "epoch": 2,
                "optimizer_step": 2,
                "run_kind": "qualification",
                "arm": "full",
                "validation_role": "V_hold",
            }
        )
    artifact_paths["validation_events"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in event_rows),
        encoding="utf-8",
    )
    run_metadata = {
        "status": "complete",
        "run_kind": "qualification",
        "arm": "full",
        "seed": 0,
        "config_path": str(config_paths["full"]),
        "config_sha256": _sha256(config_paths["full"]),
        "preregistration_sha256": "9" * 64,
        "implementation_commit": source_commit,
        "implementation_tracked_clean": True,
        "implementation_tracked_status_sha256": empty_digest,
        "feature_stats_sha256": "8" * 64,
        "v_hold_validation_evidence": {
            "schema": "egostitch_e2e_v_hold_validation_events_v1",
            "count": len(event_rows),
            "path": artifact_paths["validation_events"].name,
            "sha256": _sha256(artifact_paths["validation_events"]),
        },
    }
    artifact_paths["run_metadata"].write_text(json.dumps(run_metadata), encoding="utf-8")
    artifact_paths["qualification"].write_text(
        json.dumps(
            {
                "verdict": "pass",
                "hparams": {"seed": 0},
                "feature_stats_sha256": "8" * 64,
                "model_config_sha256": "7" * 64,
            }
        ),
        encoding="utf-8",
    )
    artifact_paths["artifact_manifest"].write_text(
        json.dumps(
            {
                path.name: {
                    "sha256": _sha256(path),
                    "byte_size": path.stat().st_size,
                }
                for path in (
                    artifact_paths["source"],
                    artifact_paths["run_metadata"],
                    artifact_paths["qualification"],
                    artifact_paths["validation_events"],
                )
            }
        ),
        encoding="utf-8",
    )
    history_row = {
        "attempt_id": attempt_dir.name,
        "attempt_dir": str(attempt_dir),
        "exit_code": 0,
        "outcome": "success",
        "verdict": "pass",
        "qualification": _binding_record(artifact_paths["qualification"]),
        "run_metadata": _binding_record(artifact_paths["run_metadata"]),
        "validation_events": _binding_record(artifact_paths["validation_events"]),
    }
    artifact_paths["history"].write_text(
        json.dumps(
            {
                "schema_version": "egostitch_e2e_qualification_history_v1",
                "arm": "full",
                "attempts": [history_row],
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": calibration.CALIBRATION_SCHEMA,
        "method_id": calibration.METHOD_ID,
        "versions": {"python": "3.12", "numpy": "2", "scikit_learn": "1"},
        "source_attempt": {
            "attempt_id": "attempt-001",
            "attempt_dir": str(attempt_dir),
            **{
                key: _record(artifact_paths[key])
                for key in (
                    "history",
                    "source",
                    "run_metadata",
                    "qualification",
                    "artifact_manifest",
                    "validation_events",
                )
            },
            "implementation": {
                "run_metadata_commit": source_commit,
                "run_metadata_tracked_clean": True,
                "run_metadata_tracked_status_sha256": empty_digest,
                "live_git_head": source_commit,
                "live_tracked_status_sha256": empty_digest,
                "live_tracked_clean": True,
            },
            "config_sha256": _sha256(config_paths["full"]),
            "model_config_sha256": "7" * 64,
            "feature_stats_sha256": "8" * 64,
            "preregistration_sha256": "9" * 64,
        },
        "source_metadata": source_metadata,
        "point_ap": 0.2,
        "bootstrap": {
            "unit": "pair row",
            "stratified_by": "binary label",
            "replacement": True,
            "replicates": calibration.REPLICATES,
            "rng": "numpy.random.Generator(numpy.random.PCG64(0))",
            "iteration_order": "replicate-major",
            "draw_order_per_replicate": "positive row positions, then negative row positions",
            "metric_input_assembly": "positive sample, then negative sample",
            "metric": "sklearn.metrics.average_precision_score",
            "ap_dtype": "<f8",
            "ap_replicates_path": str(replicate_path),
            "ap_replicates_content_sha256": hashlib.sha256(
                replicates.tobytes(order="C")
            ).hexdigest(),
            "ap_replicates_file_sha256": _sha256(replicate_path),
        },
        "estimator": {
            "name": "sample standard deviation",
            "ddof": 1,
            "value": sample_sd,
        },
        "rounding": {
            "rule": "ceil(10000 * sd) / 10000",
            "clamp": "none",
            "auprc_tolerance": tolerance,
        },
        "access_boundary": {
            "read": ["canonical V_hold binary pair labels", "active full-model fp32 logits"],
            "forbidden": [
                "V_hold topology or clustering/MMD quantities",
                "candidate pairs or scores",
                "test pairs or scores",
                "test graph",
            ],
            "source_validation_existing_event": True,
            "bootstrap_additional_v_hold_evaluations": 0,
        },
    }
    calibration_path = registration_path.parent / "auprc_tolerance_calibration.json"
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")
    registration["registration_id"] = calibration.ACTIVE_V4_REGISTRATION_ID
    registration["data_contract"] = {
        "hold_manifest": {
            "complete_nonself_pair_count": calibration.ROWS,
            "v_hold_node_count": 512,
            "positive_count": calibration.POSITIVES,
            "nodes_sha256": "2" * 64,
            "positive_edges_sha256": "3" * 64,
            "pair_labels_sha256": "1" * 64,
        }
    }
    registration["checkpoint_selection"] = {
        "auprc_tolerance": tolerance,
        "auprc_tolerance_calibration_method": {
            "method_id": calibration.METHOD_ID,
            "pair_universe": {
                "rows": calibration.ROWS,
                "positives": calibration.POSITIVES,
                "negatives": calibration.NEGATIVES,
            },
            "scores": {
                "array": "active full-model logits",
                "dtype": "fp32",
                "transform": "none; no sigmoid",
            },
            "bootstrap": {
                "unit": "pair row",
                "stratification": "binary label",
                "replacement": True,
                "replicates": calibration.REPLICATES,
                "positive_draws_per_replicate": calibration.POSITIVES,
                "negative_draws_per_replicate": calibration.NEGATIVES,
                "rng": (
                    "numpy.random.Generator(numpy.random.PCG64(0)); one sequential "
                    "generator for all draws"
                ),
                "iteration_order": "replicate-major from replicate 0 through 9999",
                "draw_order_per_replicate": (
                    "draw positive row indices first, then negative row indices, from the "
                    "same sequential generator"
                ),
                "metric_input_assembly": (
                    "concatenate the positive sample before the negative sample for y_true, "
                    "and concatenate the corresponding raw logits in the identical order "
                    "for y_score"
                ),
                "metric": "sklearn.metrics.average_precision_score",
            },
            "estimator": (
                "sample standard deviation of the 10000 replicate AP values with ddof=1"
            ),
            "rounding": "ceil(10000 * sd) / 10000",
            "clamp": "none; no floor or cap",
        },
    }
    evidence["schema_version"] = calibration.BINDING_SCHEMA_V2
    evidence["configs"] = {
        arm: {
            "path": registration["arms"][arm]["training"],
            "sha256": _sha256(path),
        }
        for arm, path in config_paths.items()
    }
    evidence["auprc_tolerance_calibration"] = {
        "path": str(calibration_path),
        "sha256": _sha256(calibration_path),
    }
    existing_attempts = evidence.get("qualification_attempts")
    if isinstance(existing_attempts, dict):
        existing_attempts["full"] = [history_row]
    else:
        evidence["qualification_attempts"] = {
            "full": [history_row]
        }
    existing_histories = evidence.get("qualification_history_indexes")
    if isinstance(existing_histories, dict):
        existing_histories["full"] = _binding_record(artifact_paths["history"])
    else:
        evidence["qualification_history_indexes"] = {
            "full": _binding_record(artifact_paths["history"])
        }
    return calibration_path
