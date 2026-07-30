from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.metrics import average_precision_score
from src.experiments import auprc_tolerance as calibration
from tests._auprc_binding_fixture import bind_active_v4_calibration

_IMPLEMENTATION_COMMIT = "a" * 40


@pytest.fixture(autouse=True)
def _tracked_clean_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        calibration,
        "_live_tracked_implementation",
        lambda: (_IMPLEMENTATION_COMMIT, b""),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {"sha256": _sha256(path), "byte_size": path.stat().st_size}


def _artifact_record(name: str) -> dict[str, object]:
    return {"path": name, "sha256": "a" * 64, "byte_size": 1}


def _registered_method() -> dict[str, object]:
    return {
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
        },
        "estimator": "sample standard deviation of the 10000 replicate AP values with ddof=1",
        "rounding": "ceil(10000 * sd) / 10000",
        "clamp": "none; no floor or cap",
    }


def _bound_calibration_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    tolerance = 0.0145
    configs: dict[str, dict[str, str]] = {}
    arms: dict[str, dict[str, object]] = {}
    for arm in calibration.ACTIVE_V4_TRAINED_ARMS:
        path = tmp_path / f"{arm}.yaml"
        path.write_text(
            "training:\n"
            f"  selection_auprc_tolerance: {tolerance}\n"
            "diagnostics:\n"
            f"  selection_auprc_tolerance: {tolerance}\n",
            encoding="utf-8",
        )
        configs[arm] = {"path": str(path), "sha256": _sha256(path)}
        arms[arm] = {
            "kind": "trained_checkpoint",
            "training": str(path),
        }
    for arm in calibration.ACTIVE_V4_ARMS - calibration.ACTIVE_V4_TRAINED_ARMS:
        arms[arm] = {"kind": "scoring_time_control", "checkpoint_arm": "full"}

    source_metadata = {
        "schema": calibration.SOURCE_SCHEMA,
        "arm": "full",
        "seed": 0,
        "run_kind": "qualification",
        "validation_role": "V_hold",
        "validation_epoch": 3,
        "global_step": 30,
        "fixed_source_epoch": 3,
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
    empty_digest = hashlib.sha256(b"").hexdigest()
    payload = {
        "schema_version": calibration.CALIBRATION_SCHEMA,
        "method_id": calibration.METHOD_ID,
        "versions": {"python": "3.12", "numpy": "2", "scikit_learn": "1"},
        "source_attempt": {
            "attempt_id": "attempt-001",
            "attempt_dir": "qualification/full/attempts/attempt-001",
            **{
                key: _artifact_record(key)
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
                "run_metadata_commit": _IMPLEMENTATION_COMMIT,
                "run_metadata_tracked_clean": True,
                "run_metadata_tracked_status_sha256": empty_digest,
                "live_git_head": _IMPLEMENTATION_COMMIT,
                "live_tracked_status_sha256": empty_digest,
                "live_tracked_clean": True,
            },
            "config_sha256": "6" * 64,
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
            "ap_replicates_path": "ap_replicates.npy",
            "ap_replicates_content_sha256": "a" * 64,
            "ap_replicates_file_sha256": "b" * 64,
        },
        "estimator": {"name": "sample standard deviation", "ddof": 1, "value": 0.01441},
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
    calibration_path = tmp_path / calibration.CALIBRATION_FILENAME
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")
    registration = {
        "registration_id": calibration.ACTIVE_V4_REGISTRATION_ID,
        "status": "BINDING",
        "arms": arms,
        "data_contract": {
            "hold_manifest": {
                "complete_nonself_pair_count": calibration.ROWS,
                "v_hold_node_count": 512,
                "positive_count": calibration.POSITIVES,
                "nodes_sha256": "2" * 64,
                "positive_edges_sha256": "3" * 64,
                "pair_labels_sha256": "1" * 64,
            }
        },
        "checkpoint_selection": {
            "auprc_tolerance": tolerance,
            "auprc_tolerance_calibration_method": _registered_method(),
        },
        "binding_evidence": {
            "schema_version": calibration.BINDING_SCHEMA_V2,
            "configs": configs,
            "auprc_tolerance_calibration": {
                "path": str(calibration_path),
                "sha256": _sha256(calibration_path),
            },
        },
    }
    registration_path = tmp_path / "registration.json"
    calibration_path = bind_active_v4_calibration(
        registration,
        registration_path,
        config_paths={arm: Path(entry["path"]) for arm, entry in configs.items()},
        tolerance=tolerance,
    )
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    return registration, registration_path, calibration_path


def _source_metadata(labels: np.ndarray, logits: np.ndarray) -> dict[str, object]:
    return {
        "schema": calibration.SOURCE_SCHEMA,
        "arm": "full",
        "seed": 0,
        "run_kind": "qualification",
        "validation_role": "V_hold",
        "validation_epoch": 2,
        "global_step": 20,
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
        "labels_dtype": labels.dtype.name,
        "active_logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
    }


def _write_recorded_attempt(tmp_path: Path) -> tuple[Path, Path]:
    attempt_dir = tmp_path / "qualification" / "full" / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    labels = np.concatenate(
        (
            np.ones(calibration.POSITIVES, dtype=np.int8),
            np.zeros(calibration.NEGATIVES, dtype=np.int8),
        )
    )
    logits = np.linspace(1.0, -1.0, calibration.ROWS, dtype="<f4")
    source = attempt_dir / calibration.AUPRC_TOLERANCE_SOURCE_FILENAME
    with source.open("xb") as handle:
        np.savez(
            handle,
            labels=labels,
            active_logits=logits,
            metadata_json=np.asarray(
                json.dumps(_source_metadata(labels, logits), sort_keys=True), dtype=np.str_
            ),
        )

    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text('{"status":"DRAFT"}\n', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("model: egostitch_e2e\n", encoding="utf-8")
    ledger = attempt_dir / "v_hold_validation_events.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "ordinal": 2,
                "kind": "epoch_end",
                "epoch": 2,
                "optimizer_step": 20,
                "run_kind": "qualification",
                "arm": "full",
                "validation_role": "V_hold",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    feature_digest = "4" * 64
    run_metadata = attempt_dir / "run_metadata.json"
    run_metadata.write_text(
        json.dumps(
            {
                "status": "complete",
                "run_kind": "qualification",
                "arm": "full",
                "seed": 0,
                "config_path": str(config),
                "config_sha256": _sha256(config),
                "preregistration_sha256": _sha256(preregistration),
                "implementation_commit": _IMPLEMENTATION_COMMIT,
                "implementation_tracked_clean": True,
                "implementation_tracked_status_sha256": hashlib.sha256(b"").hexdigest(),
                "feature_stats_sha256": feature_digest,
                "v_hold_validation_evidence": {
                    "schema": "egostitch_e2e_v_hold_validation_events_v1",
                    "count": 1,
                    "path": ledger.name,
                    "sha256": _sha256(ledger),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    qualification = attempt_dir / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "verdict": "pass",
                "hparams": {"seed": 0},
                "feature_stats_sha256": feature_digest,
                "model_config_sha256": "5" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = attempt_dir / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                source.name: _record(source),
                run_metadata.name: _record(run_metadata),
                qualification.name: _record(qualification),
                ledger.name: _record(ledger),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    history = attempt_dir.parent.parent / "attempt_history.json"
    history.write_text(
        json.dumps(
            {
                "schema_version": "egostitch_e2e_qualification_history_v1",
                "arm": "full",
                "attempts": [
                    {
                        "attempt_id": attempt_dir.name,
                        "attempt_dir": str(attempt_dir),
                        "exit_code": 0,
                        "outcome": "success",
                        "verdict": "pass",
                        "qualification": _record(qualification),
                        "run_metadata": _record(run_metadata),
                        "validation_events": _record(ledger),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return attempt_dir, preregistration


def test_optimized_bootstrap_is_replicatewise_identical_to_direct_sklearn() -> None:
    labels = np.asarray([1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
    logits = np.asarray([0.7, 0.7, -0.2, 0.1, 0.1, 0.4, -0.5, 0.4], dtype="<f4")
    replicates = 25

    optimized = calibration.bootstrap_ap_replicates(labels, logits, replicates=replicates)
    positive_scores = logits[labels == 1]
    negative_scores = logits[labels == 0]
    rng = np.random.Generator(np.random.PCG64(0))
    direct = []
    for _ in range(replicates):
        positive = rng.integers(0, positive_scores.size, size=positive_scores.size)
        negative = rng.integers(0, negative_scores.size, size=negative_scores.size)
        y_true = np.concatenate(
            (np.ones(positive.size, dtype=np.int8), np.zeros(negative.size, dtype=np.int8))
        )
        y_score = np.concatenate((positive_scores[positive], negative_scores[negative]))
        direct.append(average_precision_score(y_true, y_score))

    np.testing.assert_array_equal(optimized, np.asarray(direct, dtype="<f8"))


def test_source_loader_rejects_non_little_endian_fp32(tmp_path: Path) -> None:
    labels = np.concatenate(
        (
            np.ones(calibration.POSITIVES, dtype=np.int8),
            np.zeros(calibration.NEGATIVES, dtype=np.int8),
        )
    )
    logits = np.zeros(calibration.ROWS, dtype=">f4")
    path = tmp_path / calibration.AUPRC_TOLERANCE_SOURCE_FILENAME
    with path.open("xb") as handle:
        np.savez(
            handle,
            labels=labels,
            active_logits=logits,
            metadata_json=np.asarray(
                json.dumps(_source_metadata(labels, logits), sort_keys=True), dtype=np.str_
            ),
        )

    with pytest.raises(ValueError, match="little-endian fp32"):
        calibration.load_source(path)


def test_binding_schema_couples_active_v4_to_v2_and_historical_registration_to_v1(
    tmp_path: Path,
) -> None:
    registration, _, _ = _bound_calibration_fixture(tmp_path)
    assert (
        calibration.binding_schema_for_registration(registration)
        == calibration.BINDING_SCHEMA_V2
    )

    registration["binding_evidence"]["schema_version"] = calibration.BINDING_SCHEMA_V1
    with pytest.raises(ValueError, match="requires binding evidence.*v2"):
        calibration.binding_schema_for_registration(registration)

    historical = {
        "registration_id": calibration.HISTORICAL_V1_REGISTRATION_ID,
        "arms": {arm: {} for arm in calibration.HISTORICAL_V1_ARMS},
        "binding_evidence": {"schema_version": calibration.BINDING_SCHEMA_V1},
    }
    assert (
        calibration.binding_schema_for_registration(historical) == calibration.BINDING_SCHEMA_V1
    )
    assert calibration.validate_bound_calibration(historical, tmp_path / "historical.json") is None
    historical["binding_evidence"]["schema_version"] = calibration.BINDING_SCHEMA_V2
    with pytest.raises(ValueError, match="requires binding evidence.*v1"):
        calibration.binding_schema_for_registration(historical)


def test_bound_calibration_accepts_exact_source_method_result_and_six_configs(
    tmp_path: Path,
) -> None:
    registration, registration_path, _ = _bound_calibration_fixture(tmp_path)

    assert calibration.validate_bound_calibration(registration, registration_path) == 0.0145


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        ("schema", "schema_version"),
        ("method", "method_id"),
        ("source", "source identity pair_labels_sha256"),
        ("source_history_link", "absent from qualification_history_indexes"),
        ("source_config_link", "run metadata identity"),
        ("replicates", "bootstrap.replicates"),
        ("replicate_content", "replicate content hash"),
        ("estimator", "does not match AP replicates"),
        ("rounding", "rounding method"),
        ("result", "rounded result"),
        ("registration_tolerance", "checkpoint_selection.auprc_tolerance"),
        ("training_tolerance", "training.selection_auprc_tolerance"),
        ("diagnostics_tolerance", "diagnostics.selection_auprc_tolerance"),
    ],
)
def test_bound_calibration_rejects_layered_drift(
    tmp_path: Path, corrupt: str, message: str
) -> None:
    registration, registration_path, calibration_path = _bound_calibration_fixture(tmp_path)
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if corrupt == "schema":
        payload["schema_version"] = "wrong"
    elif corrupt == "method":
        payload["method_id"] = "wrong"
    elif corrupt == "source":
        payload["source_metadata"]["pair_labels_sha256"] = "f" * 64
    elif corrupt == "source_history_link":
        registration["binding_evidence"]["qualification_history_indexes"] = {}
    elif corrupt == "source_config_link":
        payload["source_attempt"]["config_sha256"] = "f" * 64
    elif corrupt == "replicates":
        payload["bootstrap"]["replicates"] = 9_999
    elif corrupt == "replicate_content":
        replicate_path = Path(payload["bootstrap"]["ap_replicates_path"])
        replicate_values = np.load(replicate_path, allow_pickle=False)
        replicate_values[0] += 1.0
        with replicate_path.open("wb") as handle:
            np.save(handle, replicate_values, allow_pickle=False)
        payload["bootstrap"]["ap_replicates_file_sha256"] = _sha256(replicate_path)
    elif corrupt == "estimator":
        payload["estimator"]["value"] = 0.01431
    elif corrupt == "rounding":
        payload["rounding"]["rule"] = "round(sd, 4)"
    elif corrupt == "result":
        payload["rounding"]["auprc_tolerance"] = 0.0144
    elif corrupt == "registration_tolerance":
        registration["checkpoint_selection"]["auprc_tolerance"] = 0.0144
    else:
        arm = next(iter(calibration.ACTIVE_V4_TRAINED_ARMS))
        config_path = Path(registration["arms"][arm]["training"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        section = "training" if corrupt == "training_tolerance" else "diagnostics"
        config[section]["selection_auprc_tolerance"] = 0.0144
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        registration["binding_evidence"]["configs"][arm]["sha256"] = _sha256(config_path)
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")
    registration["binding_evidence"]["auprc_tolerance_calibration"]["sha256"] = _sha256(
        calibration_path
    )
    registration_path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        calibration.validate_bound_calibration(registration, registration_path)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        ("nonbinary", "not binary"),
        ("nonfinite", "NaN or infinity"),
    ],
)
def test_source_loader_rejects_invalid_values(
    tmp_path: Path, corrupt: str, message: str
) -> None:
    labels = np.concatenate(
        (
            np.ones(calibration.POSITIVES, dtype=np.int8),
            np.zeros(calibration.NEGATIVES, dtype=np.int8),
        )
    )
    logits = np.zeros(calibration.ROWS, dtype="<f4")
    if corrupt == "nonbinary":
        labels[0] = 2
    else:
        logits[0] = np.nan
    metadata = _source_metadata(labels, logits)
    path = tmp_path / calibration.AUPRC_TOLERANCE_SOURCE_FILENAME
    with path.open("xb") as handle:
        np.savez(
            handle,
            labels=labels,
            active_logits=logits,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )

    with pytest.raises(ValueError, match=message):
        calibration.load_source(path)


def test_calibration_binds_recorded_attempt_and_exclusive_outputs(tmp_path: Path) -> None:
    attempt_dir, preregistration = _write_recorded_attempt(tmp_path)

    payload = calibration.calibrate_attempt(attempt_dir, preregistration, _replicates=3)

    replicate_path = attempt_dir / calibration.AP_REPLICATES_FILENAME
    calibration_path = attempt_dir / calibration.CALIBRATION_FILENAME
    replicates = np.load(replicate_path, allow_pickle=False)
    assert replicates.dtype.str == "<f8"
    assert replicates.shape == (3,)
    assert payload["bootstrap"]["ap_replicates_content_sha256"] == hashlib.sha256(
        replicates.tobytes()
    ).hexdigest()
    assert payload["bootstrap"]["ap_replicates_file_sha256"] == _sha256(replicate_path)
    assert payload["estimator"] == {
        "name": "sample standard deviation",
        "ddof": 1,
        "value": float(np.std(replicates, ddof=1)),
    }
    assert payload["rounding"]["clamp"] == "none"
    assert payload["access_boundary"]["bootstrap_additional_v_hold_evaluations"] == 0
    assert json.loads(calibration_path.read_text()) == payload

    before_replicates = replicate_path.read_bytes()
    before_calibration = calibration_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        calibration.calibrate_attempt(attempt_dir, preregistration, _replicates=3)
    assert replicate_path.read_bytes() == before_replicates
    assert calibration_path.read_bytes() == before_calibration


def test_calibration_resumes_matching_partial_replicates_without_overwrite(
    tmp_path: Path,
) -> None:
    attempt_dir, preregistration = _write_recorded_attempt(tmp_path)
    calibration.calibrate_attempt(attempt_dir, preregistration, _replicates=3)
    replicate_path = attempt_dir / calibration.AP_REPLICATES_FILENAME
    calibration_path = attempt_dir / calibration.CALIBRATION_FILENAME
    replicate_bytes = replicate_path.read_bytes()
    calibration_path.unlink()

    payload = calibration.calibrate_attempt(attempt_dir, preregistration, _replicates=3)

    assert replicate_path.read_bytes() == replicate_bytes
    assert calibration_path.is_file()
    assert payload["bootstrap"]["ap_replicates_file_sha256"] == _sha256(replicate_path)


def test_calibration_refuses_mismatched_partial_replicates(tmp_path: Path) -> None:
    attempt_dir, preregistration = _write_recorded_attempt(tmp_path)
    replicate_path = attempt_dir / calibration.AP_REPLICATES_FILENAME
    with replicate_path.open("xb") as handle:
        np.save(handle, np.zeros(3, dtype="<f8"), allow_pickle=False)

    with pytest.raises(ValueError, match="does not equal the deterministic recomputation"):
        calibration.calibrate_attempt(attempt_dir, preregistration, _replicates=3)
    assert not (attempt_dir / calibration.CALIBRATION_FILENAME).exists()


def test_calibration_refuses_an_unrecorded_attempt(tmp_path: Path) -> None:
    attempt_dir, preregistration = _write_recorded_attempt(tmp_path)
    (attempt_dir.parent.parent / "attempt_history.json").unlink()

    with pytest.raises(ValueError, match="attempt_history.json is missing"):
        calibration.calibrate_attempt(attempt_dir, preregistration, _replicates=2)


def test_calibration_refuses_live_config_digest_drift(tmp_path: Path) -> None:
    attempt_dir, preregistration = _write_recorded_attempt(tmp_path)
    run_metadata = json.loads((attempt_dir / "run_metadata.json").read_text())
    Path(run_metadata["config_path"]).write_text("changed: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="live run config digest"):
        calibration.calibrate_attempt(attempt_dir, preregistration, _replicates=2)
