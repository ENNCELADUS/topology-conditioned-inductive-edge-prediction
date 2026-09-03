"""Whitened-axis bank audit: PCA whitening, fp16 noise floor, and probe wiring."""

from __future__ import annotations

import numpy as np
import pytest
from src.distill.whiten_targets import whiten_axes
from src.experiments.kd_whiten_audit import fp16_noise_floor, probe_axes


def _low_rank_bank(rng: np.random.Generator, n: int = 2000, d: int = 32, k: int = 3) -> np.ndarray:
    factors = rng.normal(size=(n, k)) * np.array([10.0, 3.0, 1.0])
    basis = np.linalg.qr(rng.normal(size=(d, k)))[0]
    return np.asarray(factors @ basis.T + 1e-3 * rng.normal(size=(n, d)) + 5.0)


def test_whiten_axes_returns_unit_variance_training_coordinates() -> None:
    rng = np.random.default_rng(0)
    rep_tr = _low_rank_bank(rng)
    rep_va = _low_rank_bank(rng, n=500)
    white = whiten_axes(rep_tr, rep_va, k=3)
    assert white.train.shape == (2000, 3)
    assert white.val.shape == (500, 3)
    np.testing.assert_allclose(white.train.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(white.train.std(axis=0), 1.0, atol=1e-9)
    assert white.axis_std[0] > white.axis_std[1] > white.axis_std[2]
    assert white.var_share[0] > 0.85
    assert white.val_shift.shape == (3,)


def test_whiten_axes_rejects_k_beyond_rank() -> None:
    rng = np.random.default_rng(0)
    rep = rng.normal(size=(50, 4))
    with pytest.raises(ValueError, match="k"):
        whiten_axes(rep, rep, k=5)


def test_fp16_noise_floor_scales_with_magnitude() -> None:
    small = np.full((100, 16), 0.01)
    large = np.full((100, 16), 10.0)
    assert fp16_noise_floor(large) > fp16_noise_floor(small) * 100


def test_probe_axes_recovers_linear_and_nonlinear_targets() -> None:
    rng = np.random.default_rng(1)
    x_tr = rng.normal(size=(3000, 4))
    x_va = rng.normal(size=(1000, 4))
    axes_tr = np.stack([x_tr[:, 0] + 0.5 * x_tr[:, 1], np.sin(3 * x_tr[:, 2])], axis=1)
    axes_va = np.stack([x_va[:, 0] + 0.5 * x_va[:, 1], np.sin(3 * x_va[:, 2])], axis=1)
    fit = np.ones(3000, dtype=bool)
    fit[:600] = False
    report = probe_axes(x_tr, axes_tr, x_va, axes_va, fit=fit, nonlinear=True, seed=0)
    assert report["train_holdout"][0] > 0.95 and report["val"][0] > 0.95
    assert report["train_holdout"][1] > 0.8 and report["val"][1] > 0.8
    linear = probe_axes(x_tr, axes_tr, x_va, axes_va, fit=fit, nonlinear=False, seed=0)
    assert linear["val"][0] > 0.95
    assert linear["val"][1] < 0.3
