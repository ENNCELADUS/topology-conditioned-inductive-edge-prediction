"""Whitened-axis KD row targets: PCA whitening of a teacher bank into a `kd_row_targets_v1` bank."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from src.distill.artifacts import KDRowTargets, load_kd_targets, write_kd_targets
from src.distill.whiten_targets import whiten_axes, whitened_row_targets


def _bank(rng: np.random.Generator, n: int = 400, n_val: int = 100, d: int = 16) -> KDRowTargets:
    scales = np.array([10.0, 3.0, 1.0, 0.5])
    basis = np.linalg.qr(rng.normal(size=(d, 4)))[0]

    def rep(rows: int) -> np.ndarray:
        return np.asarray((rng.normal(size=(rows, 4)) * scales) @ basis.T + 2.0, dtype=np.float32)

    node_ids = [f"n{i}" for i in range(8)]
    a = rng.integers(0, 8, size=n).astype(np.int32)
    b = rng.integers(0, 8, size=n).astype(np.int32)
    va = rng.integers(0, 8, size=n_val).astype(np.int32)
    vb = rng.integers(0, 8, size=n_val).astype(np.int32)
    return KDRowTargets(
        node_ids=node_ids,
        pair_a_idx=a,
        pair_b_idx=b,
        pair_label=(rng.random(n) < 0.5).astype(np.int8),
        teacher_logit=rng.normal(size=n).astype(np.float32),
        teacher_rep=rep(n).astype(np.float16),
        val_pair_a_idx=va,
        val_pair_b_idx=vb,
        val_pair_label=(rng.random(n_val) < 0.5).astype(np.int8),
        val_teacher_logit=rng.normal(size=n_val).astype(np.float32),
        val_teacher_rep=rep(n_val).astype(np.float16),
        manifest={"rep_source": "topo", "checkpoint_id": "abc"},
    )


def test_whiten_axes_returns_unit_variance_training_coordinates() -> None:
    rng = np.random.default_rng(0)
    bank = _bank(rng)
    white = whiten_axes(
        bank.teacher_rep.astype(np.float64), bank.val_teacher_rep.astype(np.float64), k=3
    )
    assert white.train.shape == (400, 3) and white.val.shape == (100, 3)
    np.testing.assert_allclose(white.train.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(white.train.std(axis=0), 1.0, atol=1e-9)
    assert white.axis_std[0] > white.axis_std[1] > white.axis_std[2]
    with pytest.raises(ValueError, match="k"):
        whiten_axes(
            bank.teacher_rep.astype(np.float64), bank.val_teacher_rep.astype(np.float64), k=17
        )


def test_whitened_row_targets_keeps_rows_and_selects_axes() -> None:
    rng = np.random.default_rng(1)
    bank = _bank(rng)
    out = whitened_row_targets(bank, axes=(2, 4))
    assert out.teacher_rep.shape == (400, 3) and out.teacher_rep.dtype == np.float16
    assert out.val_teacher_rep.shape == (100, 3)
    coords = out.teacher_rep.astype(np.float64)
    np.testing.assert_allclose(coords.mean(axis=0), 0.0, atol=2e-2)
    np.testing.assert_allclose(coords.std(axis=0), 1.0, atol=2e-2)
    # Axis 2 of the bank has std 3 in the synthetic factor model: dropped PC1 (std 10).
    assert out.manifest["descriptors"] == ["pc2", "pc3", "pc4"]
    assert out.manifest["rep_source"] == "whitened_axes"
    assert out.manifest["source_rep_source"] == "topo"
    assert out.manifest["axes"] == [2, 3, 4]
    assert len(cast(list[float], out.manifest["axis_std"])) == 3
    np.testing.assert_array_equal(out.pair_a_idx, bank.pair_a_idx)
    np.testing.assert_array_equal(out.val_pair_label, bank.val_pair_label)
    np.testing.assert_array_equal(out.teacher_logit, bank.teacher_logit)
    assert out.node_ids == bank.node_ids
    with pytest.raises(ValueError, match="axes"):
        whitened_row_targets(bank, axes=(0, 3))
    with pytest.raises(ValueError, match="axes"):
        whitened_row_targets(bank, axes=(3, 2))


def test_whitened_val_block_uses_training_statistics() -> None:
    rng = np.random.default_rng(2)
    bank = _bank(rng)
    out = whitened_row_targets(bank, axes=(1, 2))
    train = out.teacher_rep.astype(np.float64)
    val = out.val_teacher_rep.astype(np.float64)
    # Same generating process, so V_val lands on the training scale, not re-standardized.
    assert abs(val.mean()) < 0.2 and 0.8 < val.std() < 1.25
    assert not np.allclose(val.std(axis=0), 1.0, atol=1e-6)
    assert train.shape[1] == 2


def test_whitened_targets_round_trip_through_artifact(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    out = whitened_row_targets(_bank(rng), axes=(2, 5))
    write_kd_targets(
        tmp_path / "white",
        node_ids=out.node_ids,
        pair_a_idx=out.pair_a_idx,
        pair_b_idx=out.pair_b_idx,
        pair_label=out.pair_label,
        teacher_logit=out.teacher_logit,
        teacher_rep=out.teacher_rep,
        val_pair_a_idx=out.val_pair_a_idx,
        val_pair_b_idx=out.val_pair_b_idx,
        val_pair_label=out.val_pair_label,
        val_teacher_logit=out.val_teacher_logit,
        val_teacher_rep=out.val_teacher_rep,
        truth_graph_sha256="t",
        checkpoint_path=Path("teacher.pt"),
        checkpoint_sha256="c",
        checkpoint_id="abc",
        rep_source="whitened_axes",
        manifest_extra=out.manifest,
    )
    loaded = load_kd_targets(tmp_path / "white")
    np.testing.assert_array_equal(loaded.teacher_rep, out.teacher_rep)
    assert loaded.manifest["descriptors"] == ["pc2", "pc3", "pc4", "pc5"]
    assert loaded.manifest["rep_source"] == "whitened_axes"
    assert loaded.manifest["rep_dim"] == 4
