"""E2E fail-closed scoring provenance and four-array publication tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from src import score_universe
from src.experiments import g5_stage1
from src.experiments.e2e_binding import ACTIVE_V4_REGISTRATION_ID
from src.model.egostitch.e2e_model import EgoStitchE2E


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run_provenance(metadata_path: Path, metadata: dict[str, object]) -> None:
    validation_path = metadata_path.parent / "v_hold_validation_events.jsonl"
    validation_path.write_text('{"epoch":1}\n', encoding="utf-8")
    validation = {
        "schema": "egostitch_e2e_v_hold_validation_events_v1",
        "count": 1,
        "path": validation_path.name,
        "sha256": _sha256(validation_path),
    }
    parameter_groups = {"names": ["generator"], "sha256": {"generator": "b" * 64}}
    memories = [1.0, 1.0, 1.0, 1.0]
    access_audit = {"observed_training_access": []}
    metadata.update(
        {
            "world_size": 4,
            "feature_stats_sha256": "c" * 64,
            "implementation_tracked_clean": True,
            "implementation_tracked_status_sha256": hashlib.sha256(b"").hexdigest(),
            "v_hold_validation_evidence": validation,
            "access_audit": access_audit,
            "run_provenance": {
                "schema_version": "egostitch_e2e_run_provenance_v1",
                "implementation": {
                    "commit": metadata["implementation_commit"],
                    "tracked_clean": True,
                    "tracked_status_sha256": hashlib.sha256(b"").hexdigest(),
                },
                "parameter_group_manifests": parameter_groups,
                "packs_and_validation_manifests": {
                    "pipeline_profile_path": "profile.json",
                    "feature_stats_sha256": "c" * 64,
                    "v_hold_validation_evidence": validation,
                },
                "boundary_access_audit": access_audit,
                "runtime_and_peak_memory": {
                    "world_size": 4,
                    "epochs_completed": 30,
                    "optimizer_steps": 2340,
                    "peak_memory_gib_per_rank": memories,
                },
                "checkpoint_policy_version": "egostitch_e2e_all_completed_epochs_v1",
            },
        }
    )
    (metadata_path.parent / "profile.json").write_text(
        json.dumps(
            {
                "parameter_groups": parameter_groups,
                "peak_memory_gib_per_rank": memories,
                "world_size": 4,
                "pack_manifest": {},
                "pack_identity_sha256": "d" * 64,
                "pack_evidence": {},
            }
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def _write_e2e_provenance(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    configs: dict[str, Path] = {}
    for arm in score_universe._EGOSTITCH_E2E_FORMAL_ARMS:
        config = tmp_path / f"{arm}.yaml"
        config.write_text(
            "model:\n"
            "  family: egostitch_e2e\n"
            "training:\n"
            "  selection_auprc_tolerance: 0.02\n"
            "diagnostics:\n"
            "  selection_auprc_tolerance: 0.02\n",
            encoding="utf-8",
        )
        configs[arm] = config
    config = configs["full"]
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"selected checkpoint")
    registration = {
        "registration_id": ACTIVE_V4_REGISTRATION_ID,
        "status": "DRAFT",
        "arms": {
            **{
                arm: {
                    "kind": "trained_checkpoint",
                    "training": str(path),
                    "scoring_provenance": {
                        "scaffold_control": "none",
                        "permanent_null": {
                            "b0_e2e_f_only": "all_head",
                            "pair_topology": "content_head",
                        }.get(arm, "none"),
                        "primary_logit": {
                            "b0_e2e_f_only": "f_logit",
                            "pair_topology": "pair_topology",
                        }.get(arm, "full"),
                    },
                }
                for arm, path in configs.items()
            },
            "structure_control_6a_v3": {
                "kind": "scoring_time_control",
                "training": None,
                "checkpoint_arm": "full",
                "scoring_provenance": {
                    "scaffold_control": "shuffle_within_pair_v3",
                    "seed": 0,
                    "keying": "canonical_pair_v1",
                    "permanent_null": "none",
                    "primary_logit": "full",
                    "checkpoint_arm": "full",
                },
            },
            "structure_control_6e_v1": {
                "kind": "scoring_time_control",
                "training": None,
                "checkpoint_arm": "full",
                "scoring_provenance": {
                    "scaffold_control": "rewire_checkerboard_v1",
                    "seed": 0,
                    "keying": "canonical_pair_v1",
                    "permanent_null": "none",
                    "primary_logit": "full",
                    "checkpoint_arm": "full",
                },
            },
        },
        "config_artifacts": {
            arm: {"path": str(path), "sha256": _sha256(path)}
            for arm, path in configs.items()
        },
        "run_evidence_placeholders": {
            "implementation": None,
            "runtime_and_peak_memory": None,
        },
    }
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    checkpoint_id = "0123456789abcdef"
    metadata = {
        "arm": "full",
        "arm_kind": "trained_checkpoint",
        "checkpoint_arm": "full",
        "scoring_semantics": {
            "scaffold_control": "none",
            "permanent_null": "none",
            "primary_logit": "full",
        },
        "run_kind": "formal",
        "seed": 0,
        "status": "complete",
        "formal_artifacts_published": True,
        "selected_checkpoint_eligible": True,
        "model_family": "egostitch_e2e",
        "config_path": str(config.resolve()),
        "config_sha256": _sha256(config),
        "preregistration_sha256": _sha256(registration_path),
        "implementation_commit": "a" * 40,
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": _sha256(checkpoint),
    }
    metadata_path = tmp_path / "run_metadata.json"
    _write_run_provenance(metadata_path, metadata)
    return registration_path, metadata_path, checkpoint, checkpoint_id


def _write_all_e2e_provenance(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], dict[str, Path], dict[str, str]]:
    registration_path, _metadata, _checkpoint, _checkpoint_id = _write_e2e_provenance(
        tmp_path
    )
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    registration_sha256 = _sha256(registration_path)
    metadata_paths: dict[str, Path] = {}
    checkpoint_paths: dict[str, Path] = {}
    checkpoint_ids: dict[str, str] = {}
    for index, arm in enumerate(score_universe._EGOSTITCH_E2E_FORMAL_ARMS):
        arm_dir = tmp_path / arm
        arm_dir.mkdir(exist_ok=True)
        checkpoint_path = arm_dir / "best.pt"
        checkpoint_path.write_bytes(f"checkpoint:{arm}".encode())
        checkpoint_id = f"{index + 1:016x}"
        arm_registration = registration["arms"][arm]
        config_path = Path(arm_registration["training"])
        scoring_semantics = arm_registration["scoring_provenance"]
        metadata_path = arm_dir / "run_metadata.json"
        metadata: dict[str, object] = {
                    "arm": arm,
                    "arm_kind": "trained_checkpoint",
                    "checkpoint_arm": arm,
                    "scoring_semantics": scoring_semantics,
                    "run_kind": "formal",
                    "seed": 0,
                    "status": "complete",
                    "formal_artifacts_published": True,
                    "selected_checkpoint_eligible": True,
                    "model_family": "egostitch_e2e",
                    "config_path": str(config_path.resolve()),
                    "config_sha256": _sha256(config_path),
                    "preregistration_sha256": registration_sha256,
                    "implementation_commit": "a" * 40,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "selected_checkpoint_path": str(checkpoint_path),
        }
        _write_run_provenance(metadata_path, metadata)
        metadata_paths[arm] = metadata_path
        checkpoint_paths[arm] = checkpoint_path
        checkpoint_ids[arm] = checkpoint_id
    return registration_path, metadata_paths, checkpoint_paths, checkpoint_ids


class _ScoringStub(EgoStitchE2E):
    """Minimal type-correct model for exercising the real score producer."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.cfg = type("Cfg", (), {"permanent_null": "none"})()


def _test_access_context(
    tmp_path: Path,
    *,
    shard: int = 0,
    num_shards: int = 1,
    reason: str | None = None,
) -> score_universe._TestAccessContext:
    return score_universe._TestAccessContext(
        ledger_path=tmp_path / "scores" / "test_access_ledger.jsonl",
        scoring_arm="full",
        seed=0,
        output=tmp_path / "scores" / "full.npz",
        shard=shard,
        num_shards=num_shards,
        rescore_reason=reason,
    )


def _write_test_pairs(tmp_path: Path) -> Path:
    path = (
        tmp_path / "data" / "benchmark_2025_neurips" / "breadth_first" / "test_edges.txt"
    )
    path.parent.mkdir(parents=True)
    path.write_text("a\tb\t1\n", encoding="utf-8")
    return path


def _ledger_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_bound_score_artifact(
    tmp_path: Path,
    *,
    shard: int = 0,
    num_shards: int = 1,
) -> tuple[Path, Path]:
    output = tmp_path / "bound.npz"
    path = output if num_shards == 1 else score_universe._shard_output_path(output, shard)
    ledger_path = tmp_path / "test_access_ledger.jsonl"
    access = score_universe._TestAccessContext(
        ledger_path=ledger_path,
        scoring_arm="full",
        seed=0,
        output=output,
        shard=shard,
        num_shards=num_shards,
        rescore_reason=None,
    )
    score_universe._record_test_access(access, pairs_source="test")
    assert access.ledger_binding is not None
    values = np.array([0.1 + shard], dtype=np.float32)
    score_universe.save_scores(
        path,
        node_ids=["a", "b"],
        u_idx=np.array([0], dtype=np.int32),
        v_idx=np.array([1], dtype=np.int32),
        logit=values,
        label=np.array([1], dtype=np.int8),
        row_start=shard,
        meta={
            "checkpoint_id": "checkpoint",
            "model_family": "egostitch_e2e",
            "pairs_source": "test",
            "strategy": "toy",
            "num_rows": num_shards,
            "created_utc": "2026-07-30T00:00:00Z",
            "torch_version": "test",
            "permanent_null": "none",
            "primary_logit": "full",
            "score_precision": {
                "contract": "egostitch_e2e_pair_fp32_v1",
                "pair_compute_dtype": "float32",
                "pair_autocast": False,
                "logit_storage_dtype": "float32",
            },
            "test_access_ledger": access.ledger_binding,
        },
        f_logit=values,
        pair_content=values,
        pair_topology=values,
    )
    return path, ledger_path


def _rewrite_artifact_meta(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}
    meta = json.loads(str(arrays["meta"][()]))
    mutate(meta)
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez_compressed(path, **arrays)


def test_test_access_ledger_groups_multi_gpu_shards_into_one_epoch(tmp_path: Path) -> None:
    _write_test_pairs(tmp_path)
    def resolve_shard(shard: int) -> None:
        score_universe._resolve_pairs(
            "test",
            tmp_path / "data",
            "breadth_first",
            test_access=_test_access_context(tmp_path, shard=shard, num_shards=4),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(resolve_shard, (2, 0, 3, 1)))

    records = _ledger_records(tmp_path / "scores" / "test_access_ledger.jsonl")
    assert len(records) == 4
    assert {record["scoring_epoch"] for record in records} == {1}
    assert {record["shard"] for record in records} == {0, 1, 2, 3}


def test_ledger_bound_shards_load_and_merge_as_one_scoring_epoch(tmp_path: Path) -> None:
    shard_paths = [
        _write_bound_score_artifact(tmp_path, shard=shard, num_shards=2)[0]
        for shard in range(2)
    ]

    for path in shard_paths:
        score_universe.load_scores(path)
    merged = score_universe.merge_scores(shard_paths)
    score_universe.validate_test_access_ledger_binding(merged.meta, label="merged")
    binding = merged.meta["test_access_ledger"]
    assert isinstance(binding, dict)
    assert binding["scoring_epoch"] == 1
    assert {record["shard"] for record in binding["records"]} == {0, 1}
    merged_path = tmp_path / "bound.npz"
    score_universe.save_scores(
        merged_path,
        node_ids=merged.node_ids,
        u_idx=merged.u_idx,
        v_idx=merged.v_idx,
        logit=merged.logit,
        label=merged.label,
        row_start=0,
        meta=merged.meta,
        f_logit=merged.f_logit,
        pair_content=merged.pair_content,
        pair_topology=merged.pair_topology,
        full_logit=merged.full_logit,
    )
    loaded = score_universe.load_scores(merged_path)
    score_universe.validate_artifact_precision(loaded, label="merged")


def test_ledger_bound_artifact_rejects_deleted_ledger(tmp_path: Path) -> None:
    path, ledger = _write_bound_score_artifact(tmp_path)
    ledger.unlink()

    with pytest.raises(ValueError, match="test-access ledger is missing"):
        score_universe.load_scores(path)


def test_ledger_bound_artifact_rejects_ledger_digest_tamper(tmp_path: Path) -> None:
    path, ledger = _write_bound_score_artifact(tmp_path)
    record = _ledger_records(ledger)[0]
    record["pairs_source"] = "candidate"
    ledger.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        score_universe.load_scores(path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("scoring_arm", "p0", "scoring_arm mismatch"),
        ("seed", 7, "seed mismatch"),
        ("scoring_epoch", 2, "scoring_epoch mismatch"),
    ],
)
def test_ledger_bound_artifact_rejects_identity_mismatch(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    path, _ledger = _write_bound_score_artifact(tmp_path)

    def mutate(meta: dict[str, object]) -> None:
        binding = meta["test_access_ledger"]
        assert isinstance(binding, dict)
        binding[field] = value

    _rewrite_artifact_meta(path, mutate)
    with pytest.raises(ValueError, match=error):
        score_universe.load_scores(path)


def test_test_access_ledger_rejects_repeat_without_reason_and_records_reasoned_repeat(
    tmp_path: Path,
) -> None:
    _write_test_pairs(tmp_path)
    first = _test_access_context(tmp_path)
    score_universe._resolve_pairs(
        "test", tmp_path / "data", "breadth_first", test_access=first
    )

    with pytest.raises(ValueError, match="repeat scoring requires --rescore-reason"):
        score_universe._resolve_pairs(
            "test", tmp_path / "data", "breadth_first", test_access=first
        )

    score_universe._resolve_pairs(
        "test",
        tmp_path / "data",
        "breadth_first",
        test_access=_test_access_context(tmp_path, reason="replace corrupt score artifact"),
    )
    records = _ledger_records(tmp_path / "scores" / "test_access_ledger.jsonl")
    assert [record["scoring_epoch"] for record in records] == [1, 2]
    assert records[1]["rescore_reason"] == "replace corrupt score artifact"


def test_test_access_ledger_refuses_malformed_history_before_pair_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_test_pairs(tmp_path)
    context = _test_access_context(tmp_path)
    context.ledger_path.parent.mkdir(parents=True)
    context.ledger_path.write_text("not-json\n", encoding="utf-8")

    def forbidden_pair_read(_path: Path) -> tuple[object, object]:
        raise AssertionError("pairs were read before the ledger was validated")

    monkeypatch.setattr(score_universe, "_read_pairs_tsv", forbidden_pair_read)
    with pytest.raises(ValueError, match="test-access ledger is malformed at line 1"):
        score_universe._resolve_pairs(
            "test", tmp_path / "data", "breadth_first", test_access=context
        )


def test_arbitrary_file_is_ledgered_as_heldout_before_pair_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration, metadata_paths, checkpoint_paths, checkpoint_ids = (
        _write_all_e2e_provenance(tmp_path)
    )
    copied = tmp_path / "copied-test.tsv"
    copied.write_text("a\tb\t1\n", encoding="utf-8")
    monkeypatch.setattr(
        score_universe,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (
            torch.nn.Linear(1, 1),
            "egostitch_e2e",
            checkpoint_ids["full"],
        ),
    )

    def inspect_after_ledger(_path: Path) -> tuple[object, object]:
        ledger = metadata_paths["full"].parent / "test_access_ledger.jsonl"
        records = _ledger_records(ledger)
        assert records[-1]["event"] == "resolve_pairs"
        assert records[-1]["pairs_source"] == f"file:{copied}"
        assert records[-1]["scoring_arm"] == "full"
        assert records[-1]["seed"] == 0
        raise AssertionError("pair read reached after formal ledger record")

    monkeypatch.setattr(score_universe, "_read_pairs_tsv", inspect_after_ledger)
    cli = [
        "score",
        "--checkpoint",
        str(checkpoint_paths["full"]),
        "--pairs",
        f"file:{copied}",
        "--output",
        str(tmp_path / "scores" / "full.npz"),
        "--preregistration",
        str(registration),
    ]
    for arm, metadata in metadata_paths.items():
        cli.extend(["--arm-run-metadata", f"{arm}={metadata}"])

    with pytest.raises(AssertionError, match="pair read reached after formal ledger record"):
        score_universe.main(cli)


def test_e2e_formal_scoring_provenance_accepts_exact_active_v2_binding(tmp_path: Path) -> None:
    registration, metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)

    provenance = score_universe._validate_e2e_scoring_provenance(
        registration_path=registration,
        run_metadata_path=metadata,
        checkpoint_path=checkpoint,
        checkpoint_id=checkpoint_id,
    )

    assert provenance["arm"] == "full"
    assert provenance["registration_sha256"] == _sha256(registration)
    assert provenance["checkpoint_sha256"] == _sha256(checkpoint)
    assert provenance["selected_checkpoint_eligible"] is True


def test_scoring_accepts_quality_telemetry_miss_without_margin_verdict(tmp_path: Path) -> None:
    registration, metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)
    run = json.loads(metadata.read_text(encoding="utf-8"))
    run["selected_checkpoint_eligible"] = False
    run["validation_liveness_pass"] = False
    run["quality_fields_policy"] = "telemetry_only"
    metadata.write_text(json.dumps(run), encoding="utf-8")
    (metadata.parent / "margin_verdict.json").unlink(missing_ok=True)

    provenance = score_universe._validate_e2e_scoring_provenance(
        registration_path=registration,
        run_metadata_path=metadata,
        checkpoint_path=checkpoint,
        checkpoint_id=checkpoint_id,
    )

    assert provenance["selected_checkpoint_eligible"] is False


def test_e2e_formal_scoring_provenance_revalidates_registration_digest(tmp_path: Path) -> None:
    registration, metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)
    payload = json.loads(registration.read_text(encoding="utf-8"))
    registration.write_text(json.dumps(payload), encoding="utf-8")
    run = json.loads(metadata.read_text(encoding="utf-8"))
    run["preregistration_sha256"] = _sha256(registration)
    metadata.write_text(json.dumps(run), encoding="utf-8")
    provenance = score_universe._validate_e2e_scoring_provenance(
        registration_path=registration,
        run_metadata_path=metadata,
        checkpoint_path=checkpoint,
        checkpoint_id=checkpoint_id,
    )

    assert provenance["registration_sha256"] == _sha256(registration)


def test_scoring_rejects_tampered_run_provenance(tmp_path: Path) -> None:
    registration, _metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)
    metadata = tmp_path / "run_metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["run_provenance"]["runtime_and_peak_memory"]["world_size"] = 3
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime provenance"):
        score_universe._validate_e2e_scoring_provenance(
            registration_path=registration,
            run_metadata_path=metadata,
            checkpoint_path=checkpoint,
            checkpoint_id=checkpoint_id,
        )


@pytest.mark.parametrize(
    ("scoring_arm", "checkpoint_arm", "scaffold_control"),
    [
        ("cosine_pool", "cosine_pool", "none"),
        ("structure_control_6a_v3", "full", "shuffle_within_pair_v3"),
    ],
)
def test_real_scorer_provenance_is_accepted_by_g5_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scoring_arm: str,
    checkpoint_arm: str,
    scaffold_control: str,
) -> None:
    registration_path, metadata_paths, checkpoint_paths, checkpoint_ids = (
        _write_all_e2e_provenance(tmp_path)
    )
    monkeypatch.setattr(
        score_universe,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (
            _ScoringStub(),
            "egostitch_e2e",
            checkpoint_ids[checkpoint_arm],
        ),
    )
    data_root = tmp_path / "data"
    test_pairs = (
        data_root / "benchmark_2025_neurips" / "breadth_first" / "test_edges.txt"
    )
    test_pairs.parent.mkdir(parents=True)
    test_pairs.write_text("a\tb\t1\n", encoding="utf-8")
    monkeypatch.setattr(score_universe, "FeatureStore", lambda *_args, **_kwargs: object())
    values = np.array([0.25], dtype=np.float32)
    monkeypatch.setattr(
        score_universe,
        "_score_egostitch_e2e",
        lambda *_args, **_kwargs: {
            "full": values,
            "f_logit": values,
            "pair_content": values,
            "pair_topology": values,
        },
    )
    output_path = tmp_path / f"{scoring_arm}.npz"
    cli = [
        "score",
        "--checkpoint",
        str(checkpoint_paths[checkpoint_arm]),
        "--pairs",
        "test",
        "--data-root",
        str(data_root),
        "--output",
        str(output_path),
        "--preregistration",
        str(registration_path),
        "--scaffold-control",
        scaffold_control,
        "--device",
        "cpu",
    ]
    for arm, metadata_path in metadata_paths.items():
        cli.extend(["--arm-run-metadata", f"{arm}={metadata_path}"])

    score_universe.main(cli)

    artifact = score_universe.load_scores(output_path)
    ledger = metadata_paths[checkpoint_arm].parent / "test_access_ledger.jsonl"
    records = _ledger_records(ledger)
    assert records[-1]["event"] == "resolve_pairs"
    assert records[-1]["scoring_arm"] == scoring_arm
    assert records[-1]["seed"] == 0
    preregistration = json.loads(registration_path.read_text(encoding="utf-8"))
    run_metadata = {
        arm: json.loads(path.read_text(encoding="utf-8"))
        for arm, path in metadata_paths.items()
    }
    g5_stage1._validate_e2e_scoring_provenance(
        scoring_arm,
        artifact,
        preregistration["arms"][scoring_arm],
        run_metadata=run_metadata,
        run_metadata_paths=metadata_paths,
        preregistration=preregistration,
        preregistration_path=registration_path,
        registration_sha256=_sha256(registration_path),
    )


@pytest.mark.parametrize("extra_arm", [None, "unknown"])
def test_e2e_formal_scoring_provenance_rejects_non_v3_arm_packages(
    tmp_path: Path, extra_arm: str | None
) -> None:
    registration, metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)
    payload = json.loads(registration.read_text(encoding="utf-8"))
    payload["arms"].pop("cosine_pool")
    payload["arms"].pop("no_l_rel")
    payload["arms"]["structure_control_6a"] = payload["arms"].pop(
        "structure_control_6a_v3"
    )
    payload["arms"].pop("structure_control_6e_v1")
    payload["config_artifacts"].pop("cosine_pool")
    payload["config_artifacts"].pop("no_l_rel")
    if extra_arm is not None:
        payload["arms"][extra_arm] = payload["arms"]["full"]
        payload["config_artifacts"][extra_arm] = payload["config_artifacts"]["full"]
    registration.write_text(json.dumps(payload), encoding="utf-8")
    run = json.loads(metadata.read_text(encoding="utf-8"))
    run["preregistration_sha256"] = _sha256(registration)
    metadata.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible arm identity schema"):
        score_universe._validate_e2e_scoring_provenance(
            registration_path=registration,
            run_metadata_path=metadata,
            checkpoint_path=checkpoint,
            checkpoint_id=checkpoint_id,
        )


@pytest.mark.parametrize(
    ("target", "field", "value", "match"),
    [
        ("metadata", "run_kind", "debug", "debug/non-formal"),
        ("metadata", "status", "started", "complete formal"),
        ("metadata", "implementation_commit", "b" * 40, "implementation provenance"),
        ("metadata", "checkpoint_sha256", "b" * 64, "checkpoint_sha256"),
    ],
)
def test_e2e_provenance_rejects_invalid_run_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    value: object,
    match: str,
) -> None:
    registration, metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)
    path = registration if target == "registration" else metadata
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    if target == "registration":
        run = json.loads(metadata.read_text(encoding="utf-8"))
        run["preregistration_sha256"] = _sha256(registration)
        metadata.write_text(json.dumps(run), encoding="utf-8")
    monkeypatch.setattr(
        score_universe,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (torch.nn.Linear(1, 1), "egostitch_e2e", checkpoint_id),
    )
    output = tmp_path / "forbidden.npz"
    with pytest.raises(ValueError, match=match):
        score_universe.main(
            [
                "score",
                "--checkpoint",
                str(checkpoint),
                "--pairs",
                "candidate",
                "--output",
                str(output),
                "--preregistration",
                str(registration),
                "--run-metadata",
                str(metadata),
            ]
        )
    assert not output.exists()


def test_e2e_rejects_bad_registered_config_digest(tmp_path: Path) -> None:
    registration, metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)
    payload = json.loads(registration.read_text(encoding="utf-8"))
    payload["config_artifacts"]["full"]["sha256"] = "f" * 64
    registration.write_text(json.dumps(payload), encoding="utf-8")
    run = json.loads(metadata.read_text(encoding="utf-8"))
    run["preregistration_sha256"] = _sha256(registration)
    run["config_sha256"] = "f" * 64
    metadata.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match=r"registered config digest mismatch"):
        score_universe._validate_e2e_scoring_provenance(
            registration_path=registration,
            run_metadata_path=metadata,
            checkpoint_path=checkpoint,
            checkpoint_id=checkpoint_id,
        )


def test_file_alias_of_candidate_manifest_still_requires_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registration, _metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)
    candidate = (
        tmp_path / "data" / "benchmark_2025_neurips" / "breadth_first" / "candidate_test_edges.txt"
    )
    candidate.parent.mkdir(parents=True)
    candidate.write_text("a\tb\n", encoding="utf-8")
    alias = tmp_path / "copied-candidate.tsv"
    alias.write_bytes(candidate.read_bytes() + b"\n")
    monkeypatch.setattr(
        score_universe,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (torch.nn.Linear(1, 1), "egostitch_e2e", checkpoint_id),
    )
    output = tmp_path / "forbidden.npz"

    with pytest.raises(ValueError, match="requires --preregistration"):
        score_universe.main(
            [
                "score",
                "--checkpoint",
                str(checkpoint),
                "--pairs",
                f"file:{alias}",
                "--data-root",
                str(tmp_path / "data"),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_candidate_scoring_requires_all_six_arm_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration, metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)
    monkeypatch.setattr(
        score_universe,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (torch.nn.Linear(1, 1), "egostitch_e2e", checkpoint_id),
    )
    output = tmp_path / "forbidden.npz"
    with pytest.raises(ValueError, match="exactly six arm metadata"):
        score_universe.main(
            [
                "score",
                "--checkpoint",
                str(checkpoint),
                "--pairs",
                "candidate",
                "--output",
                str(output),
                "--preregistration",
                str(registration),
                "--run-metadata",
                str(metadata),
            ]
        )
    assert not output.exists()


def test_arm_checkpoint_path_falls_back_to_metadata_sibling_best_pt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration, metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)
    monkeypatch.setattr(
        score_universe,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (torch.nn.Linear(1, 1), "egostitch_e2e", checkpoint_id),
    )
    other_dir = tmp_path / "f_only"
    other_dir.mkdir()
    other_metadata = other_dir / "run_metadata.json"
    other_metadata.write_text(
        json.dumps({"arm": "b0_e2e_f_only", "checkpoint_id": "feedfeedfeedfeed"}),
        encoding="utf-8",
    )
    output = tmp_path / "forbidden.npz"
    common = [
        "score",
        "--checkpoint",
        str(checkpoint),
        "--pairs",
        "candidate",
        "--output",
        str(output),
        "--preregistration",
        str(registration),
        "--arm-run-metadata",
        f"b0_e2e_f_only={other_metadata}",
        "--arm-run-metadata",
        f"full={metadata}",
        "--arm-run-metadata",
        f"pair_topology={metadata}",
        "--arm-run-metadata",
        f"p0={metadata}",
        "--arm-run-metadata",
        f"cosine_pool={metadata}",
        "--arm-run-metadata",
        f"no_l_rel={metadata}",
    ]
    with pytest.raises(ValueError, match="selected checkpoint not found"):
        score_universe.main(common)
    (other_dir / "best.pt").write_bytes(b"other arm checkpoint")
    with pytest.raises(ValueError, match="run metadata requires egostitch_e2e_run_provenance"):
        score_universe.main(common)
    assert not output.exists()


def test_e2e_artifact_physically_stores_all_four_decomposition_arrays(tmp_path: Path) -> None:
    output = tmp_path / "scores.npz"
    values = np.array([0.1, 0.2], dtype=np.float32)
    access = score_universe._TestAccessContext(
        ledger_path=tmp_path / "test_access_ledger.jsonl",
        scoring_arm="full",
        seed=0,
        output=output,
        shard=0,
        num_shards=1,
        rescore_reason=None,
    )
    score_universe._record_test_access(access, pairs_source="candidate")
    assert access.ledger_binding is not None
    meta: dict[str, object] = {
        "checkpoint_id": "checkpoint",
        "model_family": "egostitch_e2e",
        "pairs_source": "candidate",
        "strategy": "toy",
        "num_rows": 2,
        "created_utc": "2026-07-19T00:00:00Z",
        "torch_version": "test",
        "permanent_null": "none",
        "primary_logit": "full",
        "score_precision": {
            "contract": "egostitch_e2e_pair_fp32_v1",
            "pair_compute_dtype": "float32",
            "pair_autocast": False,
            "logit_storage_dtype": "float32",
        },
        "test_access_ledger": access.ledger_binding,
    }
    score_universe.save_scores(
        output,
        node_ids=["a", "b"],
        u_idx=np.array([0, 0], dtype=np.int32),
        v_idx=np.array([1, 1], dtype=np.int32),
        logit=values,
        label=np.array([-1, -1], dtype=np.int8),
        row_start=0,
        meta=meta,
        f_logit=values + 1,
        pair_content=values + 2,
        pair_topology=values + 3,
    )

    with np.load(output, allow_pickle=False) as artifact:
        assert {"full", "f_logit", "pair_content", "pair_topology"} <= set(artifact.files)
        np.testing.assert_array_equal(artifact["full"], artifact["logit"])


def test_loader_rejects_v1_e2e_artifact_without_meta_version(tmp_path: Path) -> None:
    output = tmp_path / "legacy-v1.npz"
    values = np.array([0.1], dtype=np.float32)
    resolution = score_universe.score_resolution_diagnostics(values)
    meta = {
        "checkpoint_id": "checkpoint",
        "model_family": "egostitch_e2e",
        "pairs_source": "candidate",
        "strategy": "toy",
        "num_rows": 1,
        "created_utc": "2026-07-19T00:00:00Z",
        "torch_version": "test",
        "permanent_null": "none",
        "primary_logit": "full",
        "score_precision": {
            "contract": "egostitch_e2e_pair_fp32_v1",
            "pair_compute_dtype": "float32",
            "pair_autocast": False,
            "logit_storage_dtype": "float32",
        },
        "score_resolution": dict.fromkeys(
            ("full", "f_logit", "pair_content", "pair_topology"), resolution
        ),
    }
    np.savez_compressed(
        output,
        node_ids=np.array(["a", "b"]),
        u_idx=np.array([0], dtype=np.int32),
        v_idx=np.array([1], dtype=np.int32),
        logit=values,
        label=np.array([-1], dtype=np.int8),
        row_start=np.int64(0),
        meta=np.array(json.dumps(meta)),
        f_logit=values,
        pair_content=values,
        pair_topology=values,
    )

    with pytest.raises(ValueError, match="scores_meta_version"):
        score_universe.load_scores(output)


def test_formal_v2_loader_rejects_missing_or_contradictory_full_array(tmp_path: Path) -> None:
    output = tmp_path / "malformed-e2e.npz"
    values = np.array([0.1], dtype=np.float32)
    resolution = score_universe.score_resolution_diagnostics(values)
    meta = {
        "model_family": "egostitch_e2e",
        "scores_meta_version": score_universe._SCORES_META_VERSION,
        "primary_logit": "full",
        "formal_scoring_provenance": {"registration_sha256": "a" * 64},
        "score_resolution": dict.fromkeys(
            ("full", "f_logit", "pair_content", "pair_topology"), resolution
        ),
    }
    common = {
        "node_ids": np.array(["a", "b"]),
        "u_idx": np.array([0], dtype=np.int32),
        "v_idx": np.array([1], dtype=np.int32),
        "logit": values,
        "label": np.array([-1], dtype=np.int8),
        "row_start": np.int64(0),
        "meta": np.array(json.dumps(meta)),
        "f_logit": values,
        "pair_content": values,
        "pair_topology": values,
    }
    np.savez_compressed(output, **common)
    with pytest.raises(ValueError, match="missing full"):
        score_universe.load_scores(output)

    np.savez_compressed(output, **common, full=values + 1)
    with pytest.raises(ValueError, match="contradicts primary logit"):
        score_universe.load_scores(output)
