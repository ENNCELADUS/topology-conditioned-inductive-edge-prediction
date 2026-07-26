"""E2E fail-closed scoring provenance and four-array publication tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from src import score_universe


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_e2e_provenance(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    configs: dict[str, Path] = {}
    for arm in score_universe._EGOSTITCH_E2E_FORMAL_ARMS:
        config = tmp_path / f"{arm}.yaml"
        config.write_text("model:\n  family: egostitch_e2e\n", encoding="utf-8")
        configs[arm] = config
    config = configs["full"]
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"selected checkpoint")
    digest_record = {"path": "evidence.json", "sha256": "1" * 64}
    registration = {
        "status": "BINDING",
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
        "binding_evidence": {
            "schema_version": "egostitch_e2e_binding_evidence_v1",
            "implementation": {"commit": "a" * 40},
            "configs": {
                arm: {"path": str(path), "sha256": _sha256(path)}
                for arm, path in configs.items()
            },
            "parameter_group_manifests": digest_record,
            "packs_and_validation_manifests": digest_record,
            "qualification_attempts": [digest_record],
            "boundary_access_audit": digest_record,
            "runtime_and_peak_memory": {"runtime_seconds": 1.0, "peak_memory_bytes": 1},
            "checkpoint_policy_version": "v2",
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
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return registration_path, metadata_path, checkpoint, checkpoint_id


def test_e2e_formal_scoring_provenance_accepts_exact_binding(tmp_path: Path) -> None:
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
    payload["binding_evidence"]["configs"].pop("cosine_pool")
    payload["binding_evidence"]["configs"].pop("no_l_rel")
    if extra_arm is not None:
        payload["arms"][extra_arm] = payload["arms"]["full"]
        payload["binding_evidence"]["configs"][extra_arm] = payload["binding_evidence"][
            "configs"
        ]["full"]
    registration.write_text(json.dumps(payload), encoding="utf-8")
    run = json.loads(metadata.read_text(encoding="utf-8"))
    run["preregistration_sha256"] = _sha256(registration)
    metadata.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(ValueError, match="six trained"):
        score_universe._validate_e2e_scoring_provenance(
            registration_path=registration,
            run_metadata_path=metadata,
            checkpoint_path=checkpoint,
            checkpoint_id=checkpoint_id,
        )


@pytest.mark.parametrize(
    ("target", "field", "value", "match"),
    [
        ("registration", "status", "DRAFT", "status 'BINDING'"),
        ("metadata", "run_kind", "debug", "debug/non-formal"),
        ("metadata", "status", "started", "complete formal"),
        ("metadata", "selected_checkpoint_eligible", False, "eligible selected"),
        ("metadata", "implementation_commit", "b" * 40, "implementation_commit"),
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


def test_e2e_rejects_required_marker_and_bad_config_digest(tmp_path: Path) -> None:
    registration, metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)
    payload = json.loads(registration.read_text(encoding="utf-8"))
    payload["required_before_binding"] = ["REQUIRED-BEFORE-BINDING: rehearsal"]
    registration.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="REQUIRED-BEFORE-BINDING"):
        score_universe._validate_e2e_scoring_provenance(
            registration_path=registration,
            run_metadata_path=metadata,
            checkpoint_path=checkpoint,
            checkpoint_id=checkpoint_id,
        )

    registration, metadata, checkpoint, checkpoint_id = _write_e2e_provenance(tmp_path)
    payload = json.loads(registration.read_text(encoding="utf-8"))
    payload["binding_evidence"]["configs"]["full"]["sha256"] = "f" * 64
    registration.write_text(json.dumps(payload), encoding="utf-8")
    run = json.loads(metadata.read_text(encoding="utf-8"))
    run["preregistration_sha256"] = _sha256(registration)
    run["config_sha256"] = "f" * 64
    metadata.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="registered config digest mismatch"):
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
    with pytest.raises(ValueError, match="scoring rejects debug/non-formal runs"):
        score_universe.main(common)
    assert not output.exists()


def test_e2e_artifact_physically_stores_all_four_decomposition_arrays(tmp_path: Path) -> None:
    output = tmp_path / "scores.npz"
    values = np.array([0.1, 0.2], dtype=np.float32)
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


class TestHeldoutPairSourceGuard:
    """`_is_heldout_pair_source` must catch semantically equivalent aliases."""

    @staticmethod
    def _benchmark(tmp_path: Path) -> Path:
        benchmark_dir = tmp_path / "data" / "benchmark_2025_neurips" / "breadth_first"
        benchmark_dir.mkdir(parents=True)
        (benchmark_dir / "candidate_test_edges.txt").write_text(
            "a\tb\t1\nc\td\t0\ne\tf\t1\n", encoding="utf-8"
        )
        (benchmark_dir / "test_edges.txt").write_text("g\th\t1\ni\tj\t0\n", encoding="utf-8")
        return tmp_path / "data"

    def _check(self, tmp_path: Path, supplied: Path) -> bool:
        return score_universe._is_heldout_pair_source(
            f"file:{supplied}", self._benchmark(tmp_path), "breadth_first"
        )

    def test_named_sources_are_heldout(self, tmp_path: Path) -> None:
        data_root = self._benchmark(tmp_path)
        for name in ("candidate", "test"):
            assert score_universe._is_heldout_pair_source(name, data_root, "breadth_first")

    def test_exact_copy_is_heldout(self, tmp_path: Path) -> None:
        supplied = tmp_path / "copy.txt"
        supplied.write_text("a\tb\t1\nc\td\t0\ne\tf\t1\n", encoding="utf-8")
        assert self._check(tmp_path, supplied)

    def test_reordered_copy_is_heldout(self, tmp_path: Path) -> None:
        supplied = tmp_path / "reordered.txt"
        supplied.write_text("e\tf\t1\na\tb\t1\nc\td\t0\n", encoding="utf-8")
        assert self._check(tmp_path, supplied)

    def test_label_stripped_copy_is_heldout(self, tmp_path: Path) -> None:
        supplied = tmp_path / "unlabeled.txt"
        supplied.write_text("a\tb\nc\td\ne\tf\n", encoding="utf-8")
        assert self._check(tmp_path, supplied)

    def test_endpoint_swapped_copy_is_heldout(self, tmp_path: Path) -> None:
        supplied = tmp_path / "swapped.txt"
        supplied.write_text("b\ta\t1\nd\tc\t0\nf\te\t1\n", encoding="utf-8")
        assert self._check(tmp_path, supplied)

    def test_reordered_unlabeled_test_manifest_is_heldout(self, tmp_path: Path) -> None:
        supplied = tmp_path / "test_alias.txt"
        supplied.write_text("j\ti\ng\th\n", encoding="utf-8")
        assert self._check(tmp_path, supplied)

    def test_unrelated_pairs_are_not_heldout(self, tmp_path: Path) -> None:
        supplied = tmp_path / "other.txt"
        supplied.write_text("x\ty\t1\nz\tw\t0\n", encoding="utf-8")
        assert not self._check(tmp_path, supplied)

    def test_proper_subset_is_not_equal_multiset(self, tmp_path: Path) -> None:
        supplied = tmp_path / "subset.txt"
        supplied.write_text("a\tb\t1\nc\td\t0\n", encoding="utf-8")
        assert not self._check(tmp_path, supplied)
