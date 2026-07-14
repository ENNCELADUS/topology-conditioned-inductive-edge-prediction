"""Tests for src.eval.calibration: temperature/Platt post-hoc calibration."""

import numpy as np
import pytest
from src.eval.calibration import (
    PlattCalibration,
    TemperatureCalibration,
    bce_nll,
    fit_platt,
    fit_temperature,
    stable_sigmoid,
)

pytestmark = pytest.mark.unit


def _sampled_fit_set(
    *, temperature: float, n: int = 20_000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Logits whose labels are drawn from ``sigmoid(logit / temperature)``.

    The NLL-optimal temperature for this set converges to `temperature` as `n`
    grows; deterministic given `seed`.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=3.0, size=n)
    p_true = stable_sigmoid(logits / temperature)
    labels = (rng.uniform(size=n) < p_true).astype(np.int8)
    return logits, labels


class TestStableSigmoid:
    def test_matches_naive_formula_in_safe_range(self) -> None:
        z = np.linspace(-20, 20, 401)
        np.testing.assert_allclose(stable_sigmoid(z), 1.0 / (1.0 + np.exp(-z)), atol=1e-12)

    def test_no_overflow_at_extreme_logits(self) -> None:
        z = np.array([-1e4, 1e4])
        out = stable_sigmoid(z)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-12)


class TestBceNll:
    def test_matches_direct_computation(self) -> None:
        logits = np.array([0.5, -1.0, 2.0])
        labels = np.array([1.0, 0.0, 1.0])
        p = stable_sigmoid(logits)
        expected = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        assert bce_nll(logits, labels) == pytest.approx(float(expected), abs=1e-12)

    def test_stable_at_extreme_logits(self) -> None:
        logits = np.array([1e4, -1e4])
        labels = np.array([0.0, 1.0])
        out = bce_nll(logits, labels)
        assert np.isfinite(out)
        assert out == pytest.approx(1e4, rel=1e-6)

    def test_rejects_empty_and_mismatched(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            bce_nll(np.array([]), np.array([]))
        with pytest.raises(ValueError, match="shape mismatch"):
            bce_nll(np.array([1.0]), np.array([1.0, 0.0]))


class TestFitTemperature:
    def test_recovers_generating_temperature(self) -> None:
        logits, labels = _sampled_fit_set(temperature=2.0)
        cal = fit_temperature(logits, labels)
        assert cal.temperature == pytest.approx(2.0, rel=0.1)

    def test_recovers_sharpening_temperature_below_one(self) -> None:
        logits, labels = _sampled_fit_set(temperature=0.5, seed=1)
        cal = fit_temperature(logits, labels)
        assert cal.temperature == pytest.approx(0.5, rel=0.1)

    def test_nll_never_worse_after_fit(self) -> None:
        logits, labels = _sampled_fit_set(temperature=3.0, seed=2)
        cal = fit_temperature(logits, labels)
        assert cal.val_nll_after <= cal.val_nll_before + 1e-12

    def test_apply_preserves_ranking(self) -> None:
        logits, labels = _sampled_fit_set(temperature=2.0, n=500)
        cal = fit_temperature(logits, labels)
        np.testing.assert_array_equal(
            np.argsort(cal.apply(logits), kind="stable"),
            np.argsort(stable_sigmoid(logits), kind="stable"),
        )

    def test_deterministic(self) -> None:
        logits, labels = _sampled_fit_set(temperature=2.0)
        assert fit_temperature(logits, labels) == fit_temperature(logits.copy(), labels.copy())

    def test_rejects_single_class(self) -> None:
        with pytest.raises(ValueError, match="single-class"):
            fit_temperature(np.array([1.0, 2.0]), np.array([1, 1], dtype=np.int8))

    def test_rejects_bad_labels(self) -> None:
        with pytest.raises(ValueError, match="labels must be in"):
            fit_temperature(np.array([1.0, 2.0]), np.array([1, -1], dtype=np.int8))

    def test_to_jsonable_round_trip(self) -> None:
        cal = TemperatureCalibration(temperature=2.0, val_nll_before=0.7, val_nll_after=0.6)
        payload = cal.to_jsonable()
        assert payload["method"] == "temperature"
        assert payload["temperature"] == 2.0


class TestFitPlatt:
    def test_recovers_inverse_temperature_scale(self) -> None:
        logits, labels = _sampled_fit_set(temperature=2.0)
        cal = fit_platt(logits, labels)
        assert cal.scale == pytest.approx(0.5, rel=0.1)
        assert cal.bias == pytest.approx(0.0, abs=0.05)

    def test_flipped_labels_yield_negative_scale(self) -> None:
        logits, labels = _sampled_fit_set(temperature=1.0, seed=3)
        cal = fit_platt(logits, (1 - labels).astype(np.int8))
        assert cal.scale < 0

    def test_nll_never_worse_after_fit(self) -> None:
        logits, labels = _sampled_fit_set(temperature=0.5, seed=4)
        cal = fit_platt(logits, labels)
        assert cal.val_nll_after <= cal.val_nll_before + 1e-12

    def test_deterministic(self) -> None:
        logits, labels = _sampled_fit_set(temperature=2.0)
        assert fit_platt(logits, labels) == fit_platt(logits.copy(), labels.copy())

    def test_to_jsonable_round_trip(self) -> None:
        cal = PlattCalibration(scale=0.5, bias=0.1, val_nll_before=0.7, val_nll_after=0.6)
        payload = cal.to_jsonable()
        assert payload["method"] == "platt"
        assert payload["scale"] == 0.5
        assert payload["bias"] == 0.1
