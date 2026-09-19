from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.data.motif_shift import (
    apply_motif_shift,
    calibrate_motif_shift,
    normalized_entropy,
)
from src.data.motif_template import EDGE_TYPES
from src.score_universe import _load_motif_shift_bank


def _rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.zeros((3, 96), dtype=np.float32)
    predicted = np.zeros_like(truth)
    train = np.zeros((4, 96), dtype=np.float32)
    for kind in range(3):
        idx = np.flatnonzero(np.asarray(EDGE_TYPES) == kind)
        truth[0, idx[:2]] = [0.8, 0.2]
        truth[1, idx[:1]] = 0.5
        predicted[0, idx[:4]] = 0.25
        predicted[1, idx[:2]] = 0.05
        predicted[2, idx[:2]] = 0.3
        train[0, idx[:1]] = 0.25
        train[1, idx[:2]] = 0.25
        train[2, idx[:3]] = 0.25
        train[3, idx[:4]] = 0.25
    return truth, predicted, train


def test_calibration_is_whole_matrix_deterministic_and_training_owned() -> None:
    truth, predicted, train = _rows()
    calibration = calibrate_motif_shift(truth, predicted, train)
    assert len(calibration.lambdas) == 3
    assert all(0.0 <= value <= 1.0 for value in calibration.lambdas)
    assert calibration.nonzero_floor == (0.25, 0.25, 0.25)
    np.testing.assert_array_equal(calibration.train_mean, train.mean(axis=0))


def test_presence_level_and_content_have_distinct_contracts() -> None:
    truth, predicted, train = _rows()
    calibration = calibrate_motif_shift(truth, predicted, train)
    presence = apply_motif_shift("presence_true", truth, predicted, calibration)
    level = apply_motif_shift("level_true", truth, predicted, calibration)
    content = apply_motif_shift("content_true", truth, predicted, calibration)
    blur = apply_motif_shift("blur_true", truth, predicted, calibration)
    types = np.asarray(EDGE_TYPES)
    for kind in range(3):
        idx = np.flatnonzero(types == kind)
        # Presence removes a predicted false-positive family and falls back to
        # truth when predicted mass is below the training non-zero floor.
        assert presence[2, idx].sum() == 0
        np.testing.assert_allclose(presence[1, idx], truth[1, idx])
        # Level preserves predicted support/profile but matches true mass.
        np.testing.assert_allclose(level[:2, idx].sum(axis=1), truth[:2, idx].sum(axis=1))
        assert np.count_nonzero(level[0, idx]) == np.count_nonzero(predicted[0, idx])
        # Content preserves exact true support and mass; blur may add support.
        np.testing.assert_array_equal(content[:, idx] > 0, truth[:, idx] > 0)
        np.testing.assert_allclose(content[:, idx].sum(axis=1), truth[:, idx].sum(axis=1))
        assert np.count_nonzero(blur[0, idx]) >= np.count_nonzero(content[0, idx])


def test_blur_entropy_tracks_the_predicted_target() -> None:
    truth, predicted, train = _rows()
    calibration = calibrate_motif_shift(truth, predicted, train)
    blurred = apply_motif_shift("blur_true", truth, predicted, calibration)
    for kind in range(3):
        idx = np.flatnonzero(np.asarray(EDGE_TYPES) == kind)
        target = normalized_entropy(predicted, idx).mean()
        actual = normalized_entropy(blurred, idx).mean()
        assert abs(float(actual - target)) < 2e-3


def test_score_bank_requires_exact_ordered_pair_identity(tmp_path: Path) -> None:
    path = tmp_path / "bank.npz"
    np.savez(
        path,
        u=np.asarray(["a", "b"]),
        v=np.asarray(["b", "c"]),
        blur_true=np.zeros((2, 96), dtype=np.float32),
    )
    loaded = _load_motif_shift_bank(path, "blur_true", [("a", "b"), ("b", "c")])
    assert tuple(loaded.shape) == (2, 96)
    with pytest.raises(ValueError, match="exactly match"):
        _load_motif_shift_bank(path, "blur_true", [("b", "c"), ("a", "b")])


def test_content_true_never_deletes_a_tiny_true_support_edge() -> None:
    truth, predicted, train = _rows()
    indices = np.flatnonzero(np.asarray(EDGE_TYPES) == 0)
    truth[0, indices] = 0
    truth[0, indices[:2]] = [0.001, 0.1]
    predicted[0, indices] = 0
    predicted[0, indices[:2]] = [0.1, 0.001]
    train[:, indices] = 0
    train[:, indices[:2]] = [1.0, 0.001]
    calibration = calibrate_motif_shift(truth, predicted, train)

    content = apply_motif_shift("content_true", truth, predicted, calibration)

    assert np.all(content[0, indices[:2]] > 0)
    np.testing.assert_allclose(content[0, indices].sum(), 0.101, rtol=0, atol=1e-7)
    assert np.all(content <= 1.0)
