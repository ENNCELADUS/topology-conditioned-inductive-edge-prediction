from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from src.distill.artifacts import (
    KDContextBank,
    load_kd_context_targets,
    write_kd_context_targets,
)


def _bank(
    *,
    offsets: list[int],
    partners: list[int],
    scores: list[int],
    near: list[bool],
    anchors: list[int] | None = None,
) -> KDContextBank:
    return KDContextBank(
        anchor_idx=np.asarray([0, 1, 2] if anchors is None else anchors, dtype=np.int32),
        anchor_offsets=np.asarray(offsets, dtype=np.int64),
        partner_idx=np.asarray(partners, dtype=np.int32),
        score_idx=np.asarray(scores, dtype=np.int32),
        is_near=np.asarray(near, dtype=np.bool_),
    )


def _write_artifact(path: Path, *, teacher_logit: np.ndarray | None = None) -> None:
    write_kd_context_targets(
        path,
        node_ids=["a", "b", "c"],
        pair_a_idx=np.asarray([0, 0, 1, 2, 1], dtype=np.int32),
        pair_b_idx=np.asarray([1, 2, 2, 0, 0], dtype=np.int32),
        teacher_logit=(
            np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
            if teacher_logit is None
            else teacher_logit
        ),
        banks=(
            _bank(
                offsets=[0, 2, 3, 4],
                partners=[1, 2, 2, 0],
                scores=[0, 1, 2, 3],
                near=[True, False, True, False],
            ),
            _bank(
                offsets=[0, 1, 3, 4],
                partners=[2, 2, 0, 0],
                scores=[1, 2, 4, 3],
                near=[True, True, False, True],
            ),
        ),
        val_bank=_bank(anchors=[1], offsets=[0, 1], partners=[0], scores=[4], near=[False]),
        sampler_params={"rw_step": 3, "hops": 2, "ns_rate": 1},
        seed=0,
        truth_graph_sha256="truth-provenance",
        checkpoint_path=Path("teacher.pt"),
        checkpoint_sha256="checkpoint-provenance",
        checkpoint_id="teacher-7",
    )


def _replace_npz_array(path: Path, name: str, value: np.ndarray) -> None:
    npz_path = path / "targets.npz"
    with np.load(npz_path) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays[name] = value
    np.savez(npz_path, **cast(dict[str, Any], arrays))


def test_context_targets_round_trip_preserves_deduplicated_bank_joins(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "context-targets"
    _write_artifact(artifact_dir)

    loaded = load_kd_context_targets(artifact_dir, expected_node_ids=["a", "b", "c"])

    assert loaded.node_ids == ["a", "b", "c"]
    assert loaded.teacher_logit.dtype == np.float32
    np.testing.assert_array_equal(loaded.pair_a_idx, [0, 0, 1, 2, 1])
    np.testing.assert_array_equal(loaded.pair_b_idx, [1, 2, 2, 0, 0])
    assert len(loaded.banks) == 2
    np.testing.assert_array_equal(loaded.banks[0].score_idx, [0, 1, 2, 3])
    np.testing.assert_array_equal(loaded.banks[1].score_idx, [1, 2, 4, 3])
    np.testing.assert_array_equal(loaded.banks[1].partner_idx, [2, 2, 0, 0])
    assert loaded.manifest["format"] == "kd_ctx_targets_v1"
    assert loaded.manifest["sampler_params"] == {"rw_step": 3, "hops": 2, "ns_rate": 1}
    assert loaded.manifest["seed"] == 0
    assert loaded.manifest["n_banks"] == 2
    assert loaded.manifest["n_val_anchors"] == 1
    assert loaded.manifest["checkpoint_id"] == "teacher-7"
    assert "npz_sha256" not in loaded.manifest
    assert "node_ids_sha256" not in loaded.manifest


def test_context_targets_reject_duplicate_unique_score_rows(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "context-targets"
    _write_artifact(artifact_dir)
    _replace_npz_array(
        artifact_dir,
        "pair_a_idx",
        np.asarray([0, 0, 1, 2, 0], dtype=np.int32),
    )
    _replace_npz_array(
        artifact_dir,
        "pair_b_idx",
        np.asarray([1, 2, 2, 0, 1], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="must be unique"):
        load_kd_context_targets(artifact_dir)


def test_context_targets_reject_corrupt_bank_row_identity(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "context-targets"
    _write_artifact(artifact_dir)
    _replace_npz_array(
        artifact_dir,
        "bank_001_partner_idx",
        np.asarray([1, 2, 0, 0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match=r"does not identify its CSR \(anchor, partner\) rows"):
        load_kd_context_targets(artifact_dir)


def test_context_targets_reject_v_val_internal_training_pair(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "context-targets"
    _write_artifact(artifact_dir)
    _replace_npz_array(
        artifact_dir,
        "pair_b_idx",
        np.asarray([1, 2, 1, 0, 0], dtype=np.int32),
    )
    _replace_npz_array(
        artifact_dir,
        "bank_000_partner_idx",
        np.asarray([1, 2, 1, 0], dtype=np.int32),
    )
    _replace_npz_array(
        artifact_dir,
        "bank_001_partner_idx",
        np.asarray([2, 1, 0, 0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="V_val-internal"):
        load_kd_context_targets(artifact_dir)


def test_context_targets_reject_v_val_internal_validation_pair(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "context-targets"
    _write_artifact(artifact_dir)
    _replace_npz_array(
        artifact_dir,
        "pair_b_idx",
        np.asarray([1, 2, 2, 0, 1], dtype=np.int32),
    )
    _replace_npz_array(
        artifact_dir,
        "val_partner_idx",
        np.asarray([1], dtype=np.int32),
    )
    _replace_npz_array(
        artifact_dir,
        "bank_001_partner_idx",
        np.asarray([2, 2, 2, 0], dtype=np.int32),
    )
    _replace_npz_array(
        artifact_dir,
        "bank_001_score_idx",
        np.asarray([1, 2, 2, 3], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="validation context bank contains a V_val-internal pair"):
        load_kd_context_targets(artifact_dir)


def test_context_targets_reject_non_fp32_logits(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "context-targets"
    _write_artifact(artifact_dir)
    _replace_npz_array(
        artifact_dir,
        "teacher_logit",
        np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="teacher_logit must have dtype float32"):
        load_kd_context_targets(artifact_dir)


def test_context_targets_reject_universe_or_manifest_drift(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "context-targets"
    _write_artifact(artifact_dir)

    with pytest.raises(ValueError, match="universe/order"):
        load_kd_context_targets(artifact_dir, expected_node_ids=["a", "c", "b"])
    with pytest.raises(ValueError, match="expected_val_anchor_idx"):
        load_kd_context_targets(
            artifact_dir, expected_val_anchor_idx=np.asarray([2], dtype=np.int32)
        )

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["n_scores"] = 6
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="row/universe counts"):
        load_kd_context_targets(artifact_dir)


def test_context_writer_rejects_float32_overflow(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="remain finite in float32"):
        _write_artifact(
            tmp_path / "context-targets",
            teacher_logit=np.asarray([0.1, 0.2, 0.3, 0.4, 1e100], dtype=np.float64),
        )
