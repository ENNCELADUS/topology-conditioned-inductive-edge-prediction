"""Deterministic V_hold bootstrap calibration for the E2E AUPRC band."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import sklearn
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score

from src.train_egostitch import AUPRC_TOLERANCE_SOURCE_FILENAME

SOURCE_SCHEMA = "egostitch_e2e_auprc_tolerance_source_v1"
CALIBRATION_SCHEMA = "egostitch_e2e_auprc_tolerance_calibration_v1"
METHOD_ID = "stratified_pair_bootstrap_ap_sd_v1"
AP_REPLICATES_FILENAME = "ap_replicates.npy"
CALIBRATION_FILENAME = "auprc_tolerance_calibration.json"
ROWS = 130_816
POSITIVES = 1_533
NEGATIVES = 129_283
REPLICATES = 10_000
RNG_SEED = 0
SOURCE_RULE = "first validation after the conditioning ramp plus one complete Phase-C epoch"

BINDING_SCHEMA_V1 = "egostitch_e2e_binding_evidence_v1"
BINDING_SCHEMA_V2 = "egostitch_e2e_binding_evidence_v2"
ACTIVE_V4_REGISTRATION_ID = "g5-e2e-stage1-20260729-two-stage-ladder-screen-v4-draft"
HISTORICAL_V1_REGISTRATION_ID = (
    "g5-e2e-stage1-20260719-conditioned-encoder-stability-screen-v2"
)
ACTIVE_V4_ARMS = frozenset(
    {
        "full",
        "b0_e2e_f_only",
        "pair_topology",
        "p0",
        "cosine_pool",
        "no_l_rel",
        "structure_control_6a_v3",
        "structure_control_6e_v1",
    }
)
ACTIVE_V4_TRAINED_ARMS = frozenset(
    {"full", "b0_e2e_f_only", "pair_topology", "p0", "cosine_pool", "no_l_rel"}
)
HISTORICAL_V1_ARMS = frozenset(
    {"full", "b0_e2e_f_only", "pair_topology", "p0", "structure_control_6a"}
)

_CALIBRATION_KEYS = {
    "schema_version",
    "method_id",
    "versions",
    "source_attempt",
    "source_metadata",
    "point_ap",
    "bootstrap",
    "estimator",
    "rounding",
    "access_boundary",
}
_BOOTSTRAP_KEYS = {
    "unit",
    "stratified_by",
    "replacement",
    "replicates",
    "rng",
    "iteration_order",
    "draw_order_per_replicate",
    "metric_input_assembly",
    "metric",
    "ap_dtype",
    "ap_replicates_path",
    "ap_replicates_content_sha256",
    "ap_replicates_file_sha256",
}
_SOURCE_ATTEMPT_KEYS = {
    "attempt_id",
    "attempt_dir",
    "history",
    "source",
    "run_metadata",
    "qualification",
    "artifact_manifest",
    "validation_events",
    "implementation",
    "config_sha256",
    "model_config_sha256",
    "feature_stats_sha256",
    "preregistration_sha256",
}

_SOURCE_ARRAYS = {"labels", "active_logits", "metadata_json"}
_SOURCE_METADATA_KEYS = {
    "schema",
    "arm",
    "seed",
    "run_kind",
    "validation_role",
    "validation_epoch",
    "global_step",
    "fixed_source_epoch",
    "phase",
    "full_joint_epochs",
    "source_rule",
    "source_validation_reused",
    "source_validation_existing_event",
    "bootstrap_additional_v_hold_evaluations",
    "pair_count",
    "node_count",
    "positive_count",
    "negative_count",
    "pair_labels_sha256",
    "nodes_sha256",
    "positive_edges_sha256",
    "score_transform",
    "active_logits_dtype",
    "labels_dtype",
    "active_logits_sha256",
    "labels_sha256",
}
_HEX64_FIELDS = {
    "pair_labels_sha256",
    "nodes_sha256",
    "positive_edges_sha256",
    "active_logits_sha256",
    "labels_sha256",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _require_hex(value: object, *, label: str, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{label} must be a {length}-character lowercase hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a {length}-character lowercase hex digest")
    return value


def _stable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _validate_source_metadata(metadata: dict[str, Any]) -> None:
    if set(metadata) != _SOURCE_METADATA_KEYS:
        missing = sorted(_SOURCE_METADATA_KEYS - set(metadata))
        extra = sorted(set(metadata) - _SOURCE_METADATA_KEYS)
        raise ValueError(f"source metadata schema mismatch: missing={missing}, extra={extra}")
    expected = {
        "schema": SOURCE_SCHEMA,
        "arm": "full",
        "seed": 0,
        "run_kind": "qualification",
        "validation_role": "V_hold",
        "phase": "C",
        "full_joint_epochs": 1,
        "source_rule": SOURCE_RULE,
        "source_validation_reused": True,
        "source_validation_existing_event": True,
        "bootstrap_additional_v_hold_evaluations": 0,
        "pair_count": ROWS,
        "node_count": 512,
        "positive_count": POSITIVES,
        "negative_count": NEGATIVES,
        "score_transform": "none",
        "active_logits_dtype": "<f4",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"source metadata {key} is {metadata.get(key)!r}, expected {value!r}")
    if metadata["labels_dtype"] not in {"int8", "uint8"}:
        raise ValueError("source metadata labels_dtype must be int8 or uint8")
    for key in ("validation_epoch", "global_step", "fixed_source_epoch"):
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"source metadata {key} must be a positive integer")
    if metadata["validation_epoch"] != metadata["fixed_source_epoch"]:
        raise ValueError("source validation_epoch is not the fixed source epoch")
    for key in _HEX64_FIELDS:
        _require_hex(metadata[key], label=f"source metadata {key}")


def load_source(path: Path) -> tuple[NDArray[np.int8], NDArray[np.float32], dict[str, Any]]:
    """Load and fail closed on any source schema, identity, or precision drift."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"calibration source is missing or empty: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != _SOURCE_ARRAYS:
                raise ValueError(
                    f"source array schema mismatch: expected {sorted(_SOURCE_ARRAYS)}, "
                    f"found {sorted(payload.files)}"
                )
            labels_raw = payload["labels"]
            logits_raw = payload["active_logits"]
            metadata_raw = payload["metadata_json"]
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("source array schema"):
            raise
        raise ValueError(f"calibration source is not a readable no-pickle NPZ: {path}") from error
    if labels_raw.dtype not in (np.dtype("int8"), np.dtype("uint8")):
        raise ValueError(f"source labels dtype must be int8 or uint8, found {labels_raw.dtype}")
    if logits_raw.dtype.str != "<f4":
        raise ValueError(
            "source active_logits must be explicit little-endian fp32, "
            f"found {logits_raw.dtype.str}"
        )
    if labels_raw.ndim != 1 or logits_raw.ndim != 1 or labels_raw.shape != logits_raw.shape:
        raise ValueError(
            "source labels and active_logits must be equal-length one-dimensional arrays"
        )
    if labels_raw.shape != (ROWS,):
        raise ValueError(f"source row count is {labels_raw.size}, expected {ROWS}")
    labels = np.ascontiguousarray(labels_raw, dtype=np.int8)
    logits = np.ascontiguousarray(logits_raw, dtype="<f4")
    if not np.isfinite(logits).all():
        raise ValueError("source active_logits contain NaN or infinity")
    if not np.isin(labels, np.asarray([0, 1], dtype=np.int8)).all():
        raise ValueError("source labels are not binary")
    positive_count = int(labels.sum(dtype=np.int64))
    if positive_count != POSITIVES or labels.size - positive_count != NEGATIVES:
        raise ValueError(
            "source class counts do not match canonical V_hold: "
            f"positives={positive_count}, negatives={labels.size - positive_count}"
        )
    if metadata_raw.ndim != 0 or metadata_raw.dtype.kind != "U":
        raise ValueError("source metadata_json must be a scalar Unicode array")
    try:
        metadata = _require_mapping(
            json.loads(str(metadata_raw.item())), label="source metadata_json"
        )
    except json.JSONDecodeError as error:
        raise ValueError("source metadata_json is invalid JSON") from error
    _validate_source_metadata(metadata)
    if metadata["labels_dtype"] != labels_raw.dtype.name:
        raise ValueError("source labels dtype does not match metadata")
    if metadata["labels_sha256"] != _sha256_bytes(labels_raw.tobytes(order="C")):
        raise ValueError("source labels content hash does not match metadata")
    if metadata["active_logits_sha256"] != _sha256_bytes(logits_raw.tobytes(order="C")):
        raise ValueError("source active-logit content hash does not match metadata")
    return labels, logits, metadata


def _ap_from_group_counts(
    positive_counts: NDArray[np.float64], negative_counts: NDArray[np.float64]
) -> float:
    """Match sklearn's non-interpolated binary AP from descending score groups."""
    true_positives = np.cumsum(positive_counts, dtype=np.float64)
    false_positives = np.cumsum(negative_counts, dtype=np.float64)
    precision = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros_like(true_positives),
        where=(true_positives + false_positives) != 0,
    )
    recall = true_positives / true_positives[-1]
    precision_curve = np.hstack((precision[::-1], 1.0))
    recall_curve = np.hstack((recall[::-1], 0.0))
    return float(max(0.0, -np.sum(np.diff(recall_curve) * precision_curve[:-1])))


def bootstrap_ap_replicates(
    labels: NDArray[np.int8],
    active_logits: NDArray[np.float32],
    *,
    replicates: int = REPLICATES,
    seed: int = RNG_SEED,
) -> NDArray[np.float64]:
    """Execute the pinned replicate-major, positive-draw-first bootstrap."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    positive_scores = np.asarray(active_logits[labels == 1], dtype=np.float32)
    negative_scores = np.asarray(active_logits[labels == 0], dtype=np.float32)
    if positive_scores.size == 0 or negative_scores.size == 0:
        raise ValueError("bootstrap requires both positive and negative rows")

    all_scores = np.concatenate((positive_scores, negative_scores))
    _, inverse = np.unique(all_scores, return_inverse=True)
    group_count = int(inverse.max()) + 1
    descending_group = group_count - 1 - inverse
    positive_group = descending_group[: positive_scores.size]
    negative_group = descending_group[positive_scores.size :]

    rng = np.random.Generator(np.random.PCG64(seed))
    result = np.empty(replicates, dtype="<f8")
    for replicate in range(replicates):
        positive_draws = rng.integers(
            0, positive_scores.size, size=positive_scores.size, dtype=np.int64
        )
        negative_draws = rng.integers(
            0, negative_scores.size, size=negative_scores.size, dtype=np.int64
        )
        positive_row_counts = np.bincount(positive_draws, minlength=positive_scores.size)
        negative_row_counts = np.bincount(negative_draws, minlength=negative_scores.size)
        positive_group_counts = np.asarray(
            np.bincount(positive_group, weights=positive_row_counts, minlength=group_count),
            dtype=np.float64,
        )
        negative_group_counts = np.asarray(
            np.bincount(negative_group, weights=negative_row_counts, minlength=group_count),
            dtype=np.float64,
        )
        result[replicate] = _ap_from_group_counts(
            positive_group_counts, negative_group_counts
        )
    return result


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": _stable_path(path),
        "sha256": _sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    try:
        return _require_mapping(json.loads(path.read_text(encoding="utf-8")), label=label)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON: {path}") from error


def _registration_root(registration_path: Path) -> Path:
    resolved = registration_path.resolve()
    if resolved.parent.name == "registrations" and resolved.parent.parent.name == "docs":
        return resolved.parents[2]
    return resolved.parent


def _registered_path(registration_path: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = _registration_root(registration_path) / path
    return path.resolve()


def _require_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def binding_schema_for_registration(registration: Mapping[str, object]) -> str:
    """Couple each supported registration identity and arm schema to one evidence schema."""
    registration_id = registration.get("registration_id")
    arms = registration.get("arms")
    evidence = registration.get("binding_evidence")
    if not isinstance(arms, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("registration requires structured arms and binding_evidence")
    schema = evidence.get("schema_version")
    expected: tuple[frozenset[str], str]
    if registration_id == ACTIVE_V4_REGISTRATION_ID:
        expected = (ACTIVE_V4_ARMS, BINDING_SCHEMA_V2)
    elif registration_id == HISTORICAL_V1_REGISTRATION_ID:
        expected = (HISTORICAL_V1_ARMS, BINDING_SCHEMA_V1)
    else:
        raise ValueError(f"unsupported E2E registration identity: {registration_id!r}")
    expected_arms, expected_schema = expected
    if set(arms) != expected_arms:
        raise ValueError(
            f"registration {registration_id!r} has an incompatible arm identity schema"
        )
    if schema != expected_schema:
        raise ValueError(
            f"registration {registration_id!r} requires binding evidence {expected_schema}"
        )
    return expected_schema


def _validate_artifact_record(record: object, *, label: str) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be an artifact record")
    _require_hex(record.get("sha256"), label=f"{label}.sha256")
    if not isinstance(record.get("path"), str) or not record["path"]:
        raise ValueError(f"{label}.path must be a non-empty string")
    byte_size = record.get("byte_size")
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size <= 0:
        raise ValueError(f"{label}.byte_size must be a positive integer")


def _artifact_record_occurs(
    value: object,
    *,
    registration_path: Path,
    expected_path: Path,
    expected_sha256: str,
) -> bool:
    if isinstance(value, Mapping):
        raw_path = value.get("path")
        if isinstance(raw_path, str) and value.get("sha256") == expected_sha256:
            if _registered_path(registration_path, raw_path, label="artifact path") == expected_path:
                return True
        return any(
            _artifact_record_occurs(
                nested,
                registration_path=registration_path,
                expected_path=expected_path,
                expected_sha256=expected_sha256,
            )
            for nested in value.values()
        )
    if isinstance(value, list):
        return any(
            _artifact_record_occurs(
                nested,
                registration_path=registration_path,
                expected_path=expected_path,
                expected_sha256=expected_sha256,
            )
            for nested in value
        )
    return False


def _validate_source_attempt(
    value: object,
    *,
    registration: Mapping[str, object],
    registration_path: Path,
    source_metadata: Mapping[str, object],
) -> None:
    attempt = _require_mapping(value, label="calibration source_attempt")
    if set(attempt) != _SOURCE_ATTEMPT_KEYS:
        raise ValueError("calibration source_attempt schema mismatch")
    attempt_id = attempt.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.startswith("attempt-"):
        raise ValueError("calibration source_attempt.attempt_id is invalid")
    if not isinstance(attempt.get("attempt_dir"), str) or not attempt["attempt_dir"]:
        raise ValueError("calibration source_attempt.attempt_dir is invalid")
    attempt_dir = _registered_path(
        registration_path, attempt["attempt_dir"], label="calibration source_attempt.attempt_dir"
    )
    artifact_paths: dict[str, Path] = {}
    for label in (
        "history",
        "source",
        "run_metadata",
        "qualification",
        "artifact_manifest",
        "validation_events",
    ):
        _validate_artifact_record(attempt[label], label=f"calibration source_attempt.{label}")
        record = attempt[label]
        assert isinstance(record, Mapping)
        path = _registered_path(
            registration_path,
            record["path"],
            label=f"calibration source_attempt.{label}.path",
        )
        expected_sha256 = str(record["sha256"])
        if (
            not path.is_file()
            or path.stat().st_size != record["byte_size"]
            or _sha256_file(path) != expected_sha256
        ):
            raise ValueError(f"calibration source_attempt.{label} artifact does not match")
        artifact_paths[label] = path
    expected_locations = {
        "source": attempt_dir / AUPRC_TOLERANCE_SOURCE_FILENAME,
        "run_metadata": attempt_dir / "run_metadata.json",
        "qualification": attempt_dir / "qualification.json",
        "artifact_manifest": attempt_dir / "artifact_manifest.json",
        "validation_events": attempt_dir / "v_hold_validation_events.jsonl",
        "history": attempt_dir.parents[1] / "attempt_history.json",
    }
    for label, expected_path in expected_locations.items():
        if artifact_paths[label] != expected_path.resolve():
            raise ValueError(f"calibration source_attempt.{label} path identity does not match")
    run_metadata = _load_json(artifact_paths["run_metadata"], label="source run_metadata.json")
    expected_run = {"status": "complete", "run_kind": "qualification", "arm": "full", "seed": 0}
    for key, expected_value in expected_run.items():
        if run_metadata.get(key) != expected_value:
            raise ValueError(f"calibration source run_metadata.{key} does not match")
    if (
        run_metadata.get("config_sha256") != attempt.get("config_sha256")
        or run_metadata.get("preregistration_sha256") != attempt.get("preregistration_sha256")
    ):
        raise ValueError("calibration source run metadata identity does not match")
    implementation = _require_mapping(
        attempt.get("implementation"), label="calibration source_attempt.implementation"
    )
    if set(implementation) != {
        "run_metadata_commit",
        "run_metadata_tracked_clean",
        "run_metadata_tracked_status_sha256",
        "live_git_head",
        "live_tracked_status_sha256",
        "live_tracked_clean",
    }:
        raise ValueError("calibration source_attempt.implementation schema mismatch")
    commit = _require_hex(
        implementation.get("run_metadata_commit"),
        label="calibration source_attempt.implementation.run_metadata_commit",
        length=40,
    )
    if implementation.get("live_git_head") != commit:
        raise ValueError("calibration source attempt implementation commits disagree")
    empty_digest = _sha256_bytes(b"")
    if (
        implementation.get("run_metadata_tracked_clean") is not True
        or implementation.get("live_tracked_clean") is not True
        or implementation.get("run_metadata_tracked_status_sha256") != empty_digest
        or implementation.get("live_tracked_status_sha256") != empty_digest
    ):
        raise ValueError("calibration source attempt was not tracked-clean")
    if (
        run_metadata.get("implementation_commit") != commit
        or run_metadata.get("implementation_tracked_clean") is not True
        or run_metadata.get("implementation_tracked_status_sha256") != empty_digest
        or run_metadata.get("feature_stats_sha256") != attempt.get("feature_stats_sha256")
    ):
        raise ValueError("calibration source run implementation identity does not match")
    qualification = _load_json(
        artifact_paths["qualification"], label="source qualification.json"
    )
    hparams = qualification.get("hparams")
    if (
        qualification.get("verdict") != "pass"
        or not isinstance(hparams, Mapping)
        or hparams.get("seed") != 0
        or qualification.get("model_config_sha256") != attempt.get("model_config_sha256")
        or qualification.get("feature_stats_sha256") != attempt.get("feature_stats_sha256")
    ):
        raise ValueError("calibration source qualification identity does not match")
    manifest = _load_json(
        artifact_paths["artifact_manifest"], label="source artifact_manifest.json"
    )
    for label in ("source", "run_metadata", "qualification", "validation_events"):
        path = artifact_paths[label]
        record = manifest.get(path.name)
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != _sha256_file(path)
            or record.get("byte_size") != path.stat().st_size
        ):
            raise ValueError(f"calibration source manifest does not bind {path.name}")
    validation_evidence = run_metadata.get("v_hold_validation_evidence")
    if (
        not isinstance(validation_evidence, Mapping)
        or validation_evidence.get("schema") != "egostitch_e2e_v_hold_validation_events_v1"
        or validation_evidence.get("path") != artifact_paths["validation_events"].name
        or validation_evidence.get("sha256") != _sha256_file(artifact_paths["validation_events"])
    ):
        raise ValueError("calibration source V_hold validation evidence does not match")
    try:
        events = [
            json.loads(line)
            for line in artifact_paths["validation_events"].read_text(encoding="utf-8").splitlines()
        ]
    except json.JSONDecodeError as error:
        raise ValueError("calibration source V_hold validation ledger is invalid JSONL") from error
    if validation_evidence.get("count") != len(events):
        raise ValueError("calibration source V_hold validation ledger count does not match")
    source_events = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("kind") == "epoch_end"
        and event.get("epoch") == source_metadata.get("validation_epoch")
        and event.get("optimizer_step") == source_metadata.get("global_step")
        and event.get("run_kind") == "qualification"
        and event.get("arm") == "full"
        and event.get("validation_role") == "V_hold"
    ]
    if len(source_events) != 1:
        raise ValueError("calibration source is not exactly one recorded V_hold validation event")
    history = _load_json(artifact_paths["history"], label="source attempt_history.json")
    if (
        history.get("schema_version") != "egostitch_e2e_qualification_history_v1"
        or history.get("arm") != "full"
        or not isinstance(history.get("attempts"), list)
    ):
        raise ValueError("calibration source qualification history identity does not match")
    matching_history_rows = []
    for row in history["attempts"]:
        if not isinstance(row, Mapping):
            raise ValueError("calibration source qualification history row is invalid")
        row_dir = _registered_path(
            registration_path, row.get("attempt_dir"), label="qualification history attempt_dir"
        )
        if row.get("attempt_id") == attempt.get("attempt_id") and row_dir == attempt_dir:
            matching_history_rows.append(row)
    if len(matching_history_rows) != 1:
        raise ValueError("calibration source attempt is not unique in qualification history")
    history_row = matching_history_rows[0]
    if (
        history_row.get("exit_code") != 0
        or history_row.get("outcome") != "success"
        or history_row.get("verdict") != "pass"
    ):
        raise ValueError("calibration source history row is not a successful passing attempt")
    for label in ("qualification", "run_metadata", "validation_events"):
        row_record = history_row.get(label)
        attempt_record = attempt[label]
        assert isinstance(attempt_record, Mapping)
        if (
            not isinstance(row_record, Mapping)
            or row_record.get("sha256") != attempt_record.get("sha256")
            or _registered_path(
                registration_path,
                row_record.get("path"),
                label=f"qualification history {label}.path",
            )
            != artifact_paths[label]
        ):
            raise ValueError(f"calibration source history does not bind {label}")
    for label in (
        "config_sha256",
        "model_config_sha256",
        "feature_stats_sha256",
        "preregistration_sha256",
    ):
        _require_hex(attempt.get(label), label=f"calibration source_attempt.{label}")
    evidence = _require_mapping(registration.get("binding_evidence"), label="binding_evidence")
    bound_implementation = _require_mapping(
        evidence.get("implementation"), label="binding_evidence.implementation"
    )
    bound_commit = bound_implementation.get("commit")
    if not isinstance(bound_commit, str) or not commit.startswith(bound_commit):
        raise ValueError("calibration source implementation is not the bound implementation")
    configs = _require_mapping(evidence.get("configs"), label="binding_evidence.configs")
    full_config = _require_mapping(configs.get("full"), label="binding_evidence.configs.full")
    if attempt.get("config_sha256") != full_config.get("sha256"):
        raise ValueError("calibration source config is not the bound full-arm config")
    for label in ("qualification", "run_metadata", "validation_events"):
        record = attempt[label]
        assert isinstance(record, Mapping)
        if not _artifact_record_occurs(
            evidence.get("qualification_attempts"),
            registration_path=registration_path,
            expected_path=artifact_paths[label],
            expected_sha256=str(record["sha256"]),
        ):
            raise ValueError(
                f"calibration source_attempt.{label} is absent from qualification_attempts"
            )
    history_record = attempt["history"]
    assert isinstance(history_record, Mapping)
    if not _artifact_record_occurs(
        evidence.get("qualification_history_indexes"),
        registration_path=registration_path,
        expected_path=artifact_paths["history"],
        expected_sha256=str(history_record["sha256"]),
    ):
        raise ValueError("calibration source history is absent from qualification_history_indexes")


def _validate_registered_method(registration: Mapping[str, object]) -> None:
    checkpoint = _require_mapping(
        registration.get("checkpoint_selection"), label="checkpoint_selection"
    )
    method = _require_mapping(
        checkpoint.get("auprc_tolerance_calibration_method"),
        label="checkpoint_selection.auprc_tolerance_calibration_method",
    )
    if method.get("method_id") != METHOD_ID:
        raise ValueError("registered AUPRC calibration method_id does not match")
    universe = _require_mapping(method.get("pair_universe"), label="registered pair_universe")
    for key, expected_count in (
        ("rows", ROWS),
        ("positives", POSITIVES),
        ("negatives", NEGATIVES),
    ):
        if universe.get(key) != expected_count:
            raise ValueError(f"registered pair_universe.{key} does not match")
    scores = _require_mapping(method.get("scores"), label="registered scores")
    if scores.get("array") != "active full-model logits" or scores.get("dtype") != "fp32":
        raise ValueError("registered calibration score identity does not match")
    if scores.get("transform") != "none; no sigmoid":
        raise ValueError("registered calibration score transform does not match")
    bootstrap = _require_mapping(method.get("bootstrap"), label="registered bootstrap")
    expected_bootstrap: dict[str, object] = {
        "unit": "pair row",
        "stratification": "binary label",
        "replacement": True,
        "replicates": REPLICATES,
        "positive_draws_per_replicate": POSITIVES,
        "negative_draws_per_replicate": NEGATIVES,
        "rng": (
            "numpy.random.Generator(numpy.random.PCG64(0)); one sequential generator "
            "for all draws"
        ),
        "iteration_order": "replicate-major from replicate 0 through 9999",
        "draw_order_per_replicate": (
            "draw positive row indices first, then negative row indices, from the same "
            "sequential generator"
        ),
        "metric_input_assembly": (
            "concatenate the positive sample before the negative sample for y_true, and "
            "concatenate the corresponding raw logits in the identical order for y_score"
        ),
        "metric": "sklearn.metrics.average_precision_score",
    }
    for key, expected_value in expected_bootstrap.items():
        if bootstrap.get(key) != expected_value:
            raise ValueError(f"registered bootstrap.{key} does not match")
    if method.get("estimator") != (
        "sample standard deviation of the 10000 replicate AP values with ddof=1"
    ):
        raise ValueError("registered calibration estimator does not match")
    if method.get("rounding") != "ceil(10000 * sd) / 10000":
        raise ValueError("registered calibration rounding does not match")
    if method.get("clamp") != "none; no floor or cap":
        raise ValueError("registered calibration clamp does not match")


def _validate_hold_identity(
    registration: Mapping[str, object], source_metadata: dict[str, Any]
) -> None:
    data_contract = _require_mapping(registration.get("data_contract"), label="data_contract")
    hold = _require_mapping(data_contract.get("hold_manifest"), label="data_contract.hold_manifest")
    expected = {
        "pair_count": hold.get("complete_nonself_pair_count"),
        "node_count": hold.get("v_hold_node_count"),
        "positive_count": hold.get("positive_count"),
        "negative_count": ROWS - POSITIVES,
        "nodes_sha256": hold.get("nodes_sha256"),
        "positive_edges_sha256": hold.get("positive_edges_sha256"),
        "pair_labels_sha256": hold.get("pair_labels_sha256"),
    }
    for key, value in expected.items():
        if source_metadata.get(key) != value:
            raise ValueError(f"calibration source identity {key} does not match registration")


def _validate_bound_configs(
    registration: Mapping[str, object], registration_path: Path, tolerance: float
) -> None:
    arms = _require_mapping(registration.get("arms"), label="registration arms")
    evidence = _require_mapping(registration.get("binding_evidence"), label="binding_evidence")
    configs = _require_mapping(evidence.get("configs"), label="binding_evidence.configs")
    if set(configs) != ACTIVE_V4_TRAINED_ARMS:
        raise ValueError("binding_evidence.configs must contain exactly six trained arms")
    for arm in ACTIVE_V4_TRAINED_ARMS:
        arm_entry = _require_mapping(arms.get(arm), label=f"registration arms.{arm}")
        if arm_entry.get("kind") != "trained_checkpoint":
            raise ValueError(f"registration arms.{arm} is not a trained checkpoint")
        record = _require_mapping(configs.get(arm), label=f"binding_evidence.configs.{arm}")
        if set(record) != {"path", "sha256"} or record.get("path") != arm_entry.get("training"):
            raise ValueError(f"binding_evidence config identity does not match arm {arm}")
        config_path = _registered_path(
            registration_path, record.get("path"), label=f"binding_evidence.configs.{arm}.path"
        )
        digest = _require_hex(record.get("sha256"), label=f"binding_evidence.configs.{arm}.sha256")
        if not config_path.is_file() or _sha256_file(config_path) != digest:
            raise ValueError(f"binding_evidence config digest does not match arm {arm}")
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"registered config is not readable YAML for arm {arm}") from error
        if not isinstance(config, dict):
            raise ValueError(f"registered config must be a mapping for arm {arm}")
        for section in ("training", "diagnostics"):
            values = config.get(section)
            value = values.get("selection_auprc_tolerance") if isinstance(values, dict) else None
            if (
                _require_number(
                    value, label=f"{arm}.{section}.selection_auprc_tolerance"
                )
                != tolerance
            ):
                raise ValueError(
                    f"{arm}.{section}.selection_auprc_tolerance does not match calibration"
                )


def validate_bound_calibration(
    registration: Mapping[str, object], registration_path: Path
) -> float | None:
    """Validate the bound calibration without opening any held-out pair source.

    Historical v1 evidence predates this calibration and returns ``None``.  The
    active two-stage ladder is accepted only with a complete v2 calibration
    whose source, method, result, registration scalar, and six config scalars
    agree exactly.
    """
    schema = binding_schema_for_registration(registration)
    if schema == BINDING_SCHEMA_V1:
        return None
    evidence = _require_mapping(registration.get("binding_evidence"), label="binding_evidence")
    record = _require_mapping(
        evidence.get("auprc_tolerance_calibration"),
        label="binding_evidence.auprc_tolerance_calibration",
    )
    if set(record) != {"path", "sha256"}:
        raise ValueError(
            "binding_evidence.auprc_tolerance_calibration must contain exactly path and sha256"
        )
    calibration_path = _registered_path(
        registration_path,
        record.get("path"),
        label="binding_evidence.auprc_tolerance_calibration.path",
    )
    digest = _require_hex(
        record.get("sha256"), label="binding_evidence.auprc_tolerance_calibration.sha256"
    )
    if not calibration_path.is_file() or _sha256_file(calibration_path) != digest:
        raise ValueError("bound AUPRC tolerance calibration artifact is missing or hash-mismatched")
    payload = _load_json(calibration_path, label="AUPRC tolerance calibration")
    if set(payload) != _CALIBRATION_KEYS:
        raise ValueError("AUPRC tolerance calibration schema keys do not match")
    if payload.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError("AUPRC tolerance calibration schema_version does not match")
    if payload.get("method_id") != METHOD_ID:
        raise ValueError("AUPRC tolerance calibration method_id does not match")
    versions = _require_mapping(payload.get("versions"), label="calibration versions")
    if set(versions) != {"python", "numpy", "scikit_learn"} or any(
        not isinstance(value, str) or not value for value in versions.values()
    ):
        raise ValueError("AUPRC tolerance calibration versions are invalid")
    source_metadata = _require_mapping(
        payload.get("source_metadata"), label="calibration source_metadata"
    )
    _validate_source_metadata(source_metadata)
    _validate_hold_identity(registration, source_metadata)
    _validate_source_attempt(
        payload.get("source_attempt"),
        registration=registration,
        registration_path=registration_path,
        source_metadata=source_metadata,
    )
    point_ap = _require_number(payload.get("point_ap"), label="calibration point_ap")
    if not 0.0 <= point_ap <= 1.0:
        raise ValueError("calibration point_ap must lie in [0, 1]")
    bootstrap = _require_mapping(payload.get("bootstrap"), label="calibration bootstrap")
    if set(bootstrap) != _BOOTSTRAP_KEYS:
        raise ValueError("AUPRC tolerance calibration bootstrap schema mismatch")
    expected_bootstrap: dict[str, object] = {
        "unit": "pair row",
        "stratified_by": "binary label",
        "replacement": True,
        "replicates": REPLICATES,
        "rng": "numpy.random.Generator(numpy.random.PCG64(0))",
        "iteration_order": "replicate-major",
        "draw_order_per_replicate": "positive row positions, then negative row positions",
        "metric_input_assembly": "positive sample, then negative sample",
        "metric": "sklearn.metrics.average_precision_score",
        "ap_dtype": "<f8",
    }
    for key, expected in expected_bootstrap.items():
        if bootstrap.get(key) != expected:
            raise ValueError(f"AUPRC tolerance calibration bootstrap.{key} does not match")
    replicate_path = _registered_path(
        registration_path,
        bootstrap.get("ap_replicates_path"),
        label="calibration bootstrap.ap_replicates_path",
    )
    content_sha256 = _require_hex(
        bootstrap.get("ap_replicates_content_sha256"),
        label="calibration bootstrap.ap_replicates_content_sha256",
    )
    file_sha256 = _require_hex(
        bootstrap.get("ap_replicates_file_sha256"),
        label="calibration bootstrap.ap_replicates_file_sha256",
    )
    if not replicate_path.is_file() or _sha256_file(replicate_path) != file_sha256:
        raise ValueError("calibration AP replicate artifact is missing or hash-mismatched")
    try:
        replicates = np.load(replicate_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("calibration AP replicate artifact is not a readable no-pickle NPY") from error
    if replicates.dtype.str != "<f8" or replicates.shape != (REPLICATES,):
        raise ValueError("calibration AP replicate artifact has invalid dtype or shape")
    if not np.isfinite(replicates).all():
        raise ValueError("calibration AP replicate artifact contains NaN or infinity")
    if _sha256_bytes(replicates.tobytes(order="C")) != content_sha256:
        raise ValueError("calibration AP replicate content hash does not match")
    estimator = _require_mapping(payload.get("estimator"), label="calibration estimator")
    if set(estimator) != {"name", "ddof", "value"} or estimator.get(
        "name"
    ) != "sample standard deviation" or estimator.get("ddof") != 1:
        raise ValueError("AUPRC tolerance calibration estimator does not match")
    sample_sd = _require_number(estimator.get("value"), label="calibration estimator.value")
    if sample_sd < 0.0:
        raise ValueError("calibration estimator.value must be non-negative")
    measured_sd = float(np.std(replicates, ddof=1))
    if sample_sd != measured_sd:
        raise ValueError("calibration estimator.value does not match AP replicates")
    tolerance = math.ceil(10_000.0 * sample_sd) / 10_000.0
    rounding = _require_mapping(payload.get("rounding"), label="calibration rounding")
    if set(rounding) != {"rule", "clamp", "auprc_tolerance"}:
        raise ValueError("AUPRC tolerance calibration rounding schema mismatch")
    if rounding.get("rule") != "ceil(10000 * sd) / 10000" or rounding.get("clamp") != "none":
        raise ValueError("AUPRC tolerance calibration rounding method does not match")
    if _require_number(
        rounding.get("auprc_tolerance"), label="calibration rounding.auprc_tolerance"
    ) != tolerance:
        raise ValueError("AUPRC tolerance calibration rounded result does not match estimator")
    access = _require_mapping(payload.get("access_boundary"), label="calibration access_boundary")
    if access != {
        "read": ["canonical V_hold binary pair labels", "active full-model fp32 logits"],
        "forbidden": [
            "V_hold topology or clustering/MMD quantities",
            "candidate pairs or scores",
            "test pairs or scores",
            "test graph",
        ],
        "source_validation_existing_event": True,
        "bootstrap_additional_v_hold_evaluations": 0,
    }:
        raise ValueError("AUPRC tolerance calibration access boundary does not match")
    _validate_registered_method(registration)
    checkpoint = _require_mapping(
        registration.get("checkpoint_selection"), label="checkpoint_selection"
    )
    if _require_number(
        checkpoint.get("auprc_tolerance"), label="checkpoint_selection.auprc_tolerance"
    ) != tolerance:
        raise ValueError("checkpoint_selection.auprc_tolerance does not match calibration")
    _validate_bound_configs(registration, registration_path, tolerance)
    return tolerance


def _manifest_entry(manifest: Mapping[str, object], filename: str, path: Path) -> None:
    record = manifest.get(filename)
    if not isinstance(record, dict):
        raise ValueError(f"artifact_manifest.json has no record for {filename}")
    if record.get("sha256") != _sha256_file(path) or record.get("byte_size") != path.stat().st_size:
        raise ValueError(f"artifact_manifest.json does not match {filename}")


def _live_tracked_implementation() -> tuple[str, bytes]:
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.encode("utf-8")
    return git_head, tracked_status


def _validate_attempt(
    attempt_dir: Path, preregistration: Path, source_metadata: Mapping[str, object]
) -> dict[str, object]:
    attempt_dir = attempt_dir.resolve()
    if attempt_dir.parent.name != "attempts" or attempt_dir.parent.parent.name != "full":
        raise ValueError(
            "calibration source must be inside the full-arm immutable attempts directory"
        )
    run_metadata_path = attempt_dir / "run_metadata.json"
    qualification_path = attempt_dir / "qualification.json"
    manifest_path = attempt_dir / "artifact_manifest.json"
    ledger_path = attempt_dir / "v_hold_validation_events.jsonl"
    source_path = attempt_dir / AUPRC_TOLERANCE_SOURCE_FILENAME
    run_metadata = _load_json(run_metadata_path, label="run_metadata.json")
    qualification = _load_json(qualification_path, label="qualification.json")
    manifest = _load_json(manifest_path, label="artifact_manifest.json")

    expected_run = {"status": "complete", "run_kind": "qualification", "arm": "full", "seed": 0}
    for key, value in expected_run.items():
        if run_metadata.get(key) != value:
            raise ValueError(f"run_metadata.json {key} is not {value!r}")
    if qualification.get("verdict") != "pass":
        raise ValueError("qualification.json verdict is not 'pass'")
    hparams = qualification.get("hparams")
    if not isinstance(hparams, dict) or hparams.get("seed") != 0:
        raise ValueError("qualification.json does not bind Seed 0")
    for key in ("feature_stats_sha256", "model_config_sha256"):
        _require_hex(qualification.get(key), label=f"qualification.json {key}")
    if run_metadata.get("feature_stats_sha256") != qualification["feature_stats_sha256"]:
        raise ValueError("run_metadata and qualification feature-statistics digests disagree")
    config_path_raw = run_metadata.get("config_path")
    if not isinstance(config_path_raw, str) or not config_path_raw:
        raise ValueError("run_metadata.json has no config_path")
    config_path = Path(config_path_raw)
    if not config_path.is_file():
        raise ValueError(f"run config no longer exists: {config_path}")
    config_sha256 = _require_hex(run_metadata.get("config_sha256"), label="run config digest")
    if _sha256_file(config_path) != config_sha256:
        raise ValueError("live run config digest does not match run_metadata.json")
    preregistration_sha256 = _require_hex(
        run_metadata.get("preregistration_sha256"), label="preregistration digest"
    )
    if not preregistration.is_file() or _sha256_file(preregistration) != preregistration_sha256:
        raise ValueError("live preregistration digest does not match run_metadata.json")
    for filename, path in (
        (AUPRC_TOLERANCE_SOURCE_FILENAME, source_path),
        ("run_metadata.json", run_metadata_path),
        ("qualification.json", qualification_path),
        ("v_hold_validation_events.jsonl", ledger_path),
    ):
        _manifest_entry(manifest, filename, path)

    evidence = run_metadata.get("v_hold_validation_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("run_metadata.json has no V_hold validation evidence")
    if evidence.get("schema") != "egostitch_e2e_v_hold_validation_events_v1":
        raise ValueError("V_hold validation evidence schema is invalid")
    if evidence.get("path") != "v_hold_validation_events.jsonl":
        raise ValueError("V_hold validation evidence path is invalid")
    if evidence.get("sha256") != _sha256_file(ledger_path):
        raise ValueError("V_hold validation ledger digest does not match run_metadata.json")
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    if evidence.get("count") != len(rows) or not rows:
        raise ValueError("V_hold validation ledger count does not match run_metadata.json")
    source_events = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("kind") == "epoch_end"
        and row.get("epoch") == source_metadata["validation_epoch"]
        and row.get("optimizer_step") == source_metadata["global_step"]
        and row.get("run_kind") == "qualification"
        and row.get("arm") == "full"
        and row.get("validation_role") == "V_hold"
    ]
    if len(source_events) != 1:
        raise ValueError("source validation is not exactly one existing V_hold ledger event")

    history_path = attempt_dir.parent.parent / "attempt_history.json"
    history = _load_json(history_path, label="attempt_history.json")
    if history.get("schema_version") != "egostitch_e2e_qualification_history_v1":
        raise ValueError("qualification attempt-history schema is invalid")
    if history.get("arm") != "full" or not isinstance(history.get("attempts"), list):
        raise ValueError("qualification attempt history is not the full-arm history")
    matches = []
    for row in history["attempts"]:
        if not isinstance(row, dict):
            raise ValueError("qualification attempt-history row must be an object")
        candidate = Path(str(row.get("attempt_dir", "")))
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if row.get("attempt_id") == attempt_dir.name and candidate.resolve() == attempt_dir:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("source attempt is not indexed exactly once in attempt_history.json")
    history_row = matches[0]
    if (
        history_row.get("exit_code") != 0
        or history_row.get("outcome") != "success"
        or history_row.get("verdict") != "pass"
    ):
        raise ValueError("source attempt was not recorded as a successful passing qualification")
    for key, path in (
        ("qualification", qualification_path),
        ("run_metadata", run_metadata_path),
        ("validation_events", ledger_path),
    ):
        record = history_row.get(key)
        if not isinstance(record, dict) or record.get("sha256") != _sha256_file(path):
            raise ValueError(f"attempt_history.json does not bind {path.name}")

    commit = _require_hex(
        run_metadata.get("implementation_commit"), label="run implementation commit", length=40
    )
    if run_metadata.get("implementation_tracked_clean") is not True:
        raise ValueError("source qualification did not start from a tracked-clean implementation")
    empty_status_sha256 = _sha256_bytes(b"")
    if run_metadata.get("implementation_tracked_status_sha256") != empty_status_sha256:
        raise ValueError("source qualification tracked-status digest is not the empty digest")
    git_head, git_status = _live_tracked_implementation()
    _require_hex(git_head, label="live implementation commit", length=40)
    if commit != git_head:
        raise ValueError("live implementation commit does not match run_metadata.json")
    if git_status:
        raise ValueError("live implementation has tracked changes after the source qualification")
    return {
        "attempt_id": attempt_dir.name,
        "attempt_dir": _stable_path(attempt_dir),
        "history": _artifact_record(history_path),
        "source": _artifact_record(source_path),
        "run_metadata": _artifact_record(run_metadata_path),
        "qualification": _artifact_record(qualification_path),
        "artifact_manifest": _artifact_record(manifest_path),
        "validation_events": _artifact_record(ledger_path),
        "implementation": {
            "run_metadata_commit": commit,
            "run_metadata_tracked_clean": True,
            "run_metadata_tracked_status_sha256": empty_status_sha256,
            "live_git_head": git_head,
            "live_tracked_status_sha256": _sha256_bytes(git_status),
            "live_tracked_clean": True,
        },
        "config_sha256": config_sha256,
        "model_config_sha256": qualification["model_config_sha256"],
        "feature_stats_sha256": qualification["feature_stats_sha256"],
        "preregistration_sha256": preregistration_sha256,
    }


def _exclusive_publish(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    """Publish one fsynced file atomically without ever replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_raw = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing calibration artifact: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def calibrate_attempt(
    attempt_dir: Path,
    preregistration: Path,
    *,
    _replicates: int = REPLICATES,
) -> dict[str, object]:
    """Validate one recorded source attempt and publish the immutable calibration."""
    attempt_dir = attempt_dir.resolve()
    replicate_path = attempt_dir / AP_REPLICATES_FILENAME
    calibration_path = attempt_dir / CALIBRATION_FILENAME
    if calibration_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing calibration artifact: {calibration_path}"
        )
    source_path = attempt_dir / AUPRC_TOLERANCE_SOURCE_FILENAME
    labels, active_logits, source_metadata = load_source(source_path)
    attempt_evidence = _validate_attempt(attempt_dir, preregistration, source_metadata)
    point_ap = float(average_precision_score(labels, active_logits))
    replicate_ap = bootstrap_ap_replicates(
        labels, active_logits, replicates=_replicates, seed=RNG_SEED
    )
    if not np.isfinite(replicate_ap).all():
        raise RuntimeError("bootstrap produced a non-finite AP replicate")
    sample_sd = float(np.std(replicate_ap, ddof=1))
    if not math.isfinite(sample_sd):
        raise RuntimeError("bootstrap AP sample standard deviation is not finite")
    tolerance = math.ceil(10_000.0 * sample_sd) / 10_000.0
    replicate_sha256 = _sha256_bytes(replicate_ap.tobytes(order="C"))
    replicate_file_buffer = io.BytesIO()
    np.save(replicate_file_buffer, replicate_ap, allow_pickle=False)
    replicate_file_bytes = replicate_file_buffer.getvalue()
    if replicate_path.exists():
        try:
            resumed_replicates = np.load(replicate_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError("partial AP replicate artifact is not a readable NPY") from error
        if resumed_replicates.dtype.str != "<f8" or resumed_replicates.shape != (_replicates,):
            raise ValueError("partial AP replicate artifact has invalid dtype or shape")
        if not np.isfinite(resumed_replicates).all() or not np.array_equal(
            resumed_replicates, replicate_ap
        ):
            raise ValueError(
                "partial AP replicate artifact does not equal the deterministic recomputation"
            )
        replicate_file_sha256 = _sha256_file(replicate_path)
    else:
        replicate_file_sha256 = _sha256_bytes(replicate_file_bytes)
    payload: dict[str, object] = {
        "schema_version": CALIBRATION_SCHEMA,
        "method_id": METHOD_ID,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "source_attempt": attempt_evidence,
        "source_metadata": source_metadata,
        "point_ap": point_ap,
        "bootstrap": {
            "unit": "pair row",
            "stratified_by": "binary label",
            "replacement": True,
            "replicates": _replicates,
            "rng": "numpy.random.Generator(numpy.random.PCG64(0))",
            "iteration_order": "replicate-major",
            "draw_order_per_replicate": "positive row positions, then negative row positions",
            "metric_input_assembly": "positive sample, then negative sample",
            "metric": "sklearn.metrics.average_precision_score",
            "ap_dtype": "<f8",
            "ap_replicates_path": _stable_path(replicate_path),
            "ap_replicates_content_sha256": replicate_sha256,
            "ap_replicates_file_sha256": replicate_file_sha256,
        },
        "estimator": {"name": "sample standard deviation", "ddof": 1, "value": sample_sd},
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

    def write_replicates(handle: BinaryIO) -> None:
        handle.write(replicate_file_bytes)

    def write_calibration(handle: BinaryIO) -> None:
        handle.write(
            (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            )
        )

    if not replicate_path.exists():
        _exclusive_publish(replicate_path, write_replicates)
    _exclusive_publish(calibration_path, write_calibration)
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the one-shot production calibration CLI."""
    args = _parse_args(argv)
    payload = calibrate_attempt(args.attempt_dir, args.preregistration)
    rounding = payload["rounding"]
    if not isinstance(rounding, dict):
        raise RuntimeError("internal calibration payload has no rounding record")
    sys.stdout.write(json.dumps({"auprc_tolerance": rounding["auprc_tolerance"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
