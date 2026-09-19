"""V17 motif-graph shift counterfactuals (design section 7.6.1).

All calibration is performed on the complete V_val matrix before a result is
sliced into score shards.  The corpus mean and non-zero floors are training
statistics supplied by the caller; V_val truth is never used for either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.data.motif_template import EDGE_TYPES, N_EDGE_TYPES, N_EDGES

_TYPE_INDEX = tuple(np.flatnonzero(np.asarray(EDGE_TYPES) == kind) for kind in range(N_EDGE_TYPES))


@dataclass(frozen=True)
class MotifShiftCalibration:
    """Whole-V_val entropy calibration and training-only corpus statistics."""

    lambdas: tuple[float, ...]
    train_mean: NDArray[np.float32]
    nonzero_floor: tuple[float, ...]


def _matrix(value: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != N_EDGES:
        raise ValueError(f"{name} must have shape (rows, {N_EDGES}), got {out.shape}")
    if not np.isfinite(out).all() or (out < 0).any():
        raise ValueError(f"{name} must contain finite non-negative weights")
    return out


def normalized_entropy(
    weights: NDArray[np.floating], indices: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Per-row entropy of the within-type weight profile, normalized to [0, 1]."""
    values = np.asarray(weights, dtype=np.float64)[:, indices]
    return _profile_entropy(values)


def _profile_entropy(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalized entropy for an already-sliced edge-type matrix."""
    mass = values.sum(axis=1, keepdims=True)
    profile = np.divide(values, mass, out=np.zeros_like(values), where=mass > 0)
    terms = np.zeros_like(profile)
    np.log(profile, out=terms, where=profile > 0)
    entropy = -(profile * terms).sum(axis=1) / np.log(values.shape[1])
    return np.where(mass[:, 0] > 0, entropy, 0.0)


def training_statistics(
    training_true: NDArray[np.floating],
) -> tuple[NDArray[np.float32], tuple[float, ...]]:
    """Return the training-corpus mean and smallest observed non-zero type mass."""
    train = _matrix(training_true, "training_true")
    floors: list[float] = []
    for indices in _TYPE_INDEX:
        masses = train[:, indices].sum(axis=1)
        nonzero = masses[masses > 0]
        if not len(nonzero):
            raise ValueError("every motif edge type needs a non-zero training example")
        floors.append(float(nonzero.min()))
    return train.mean(axis=0).astype(np.float32), tuple(floors)


def _entropy_lambda(
    true: NDArray[np.float64],
    predicted: NDArray[np.float64],
    mean: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> float:
    true_type = true[:, indices]
    predicted_type = predicted[:, indices]
    mean_type = mean[indices]
    target = float(_profile_entropy(predicted_type).mean())

    def objective(lam: float) -> float:
        mixed = (1.0 - lam) * true_type + lam * mean_type[None, :]
        return abs(float(_profile_entropy(mixed).mean()) - target)

    # Deterministic dense search plus local refinement. Entropy along this line
    # need not be strictly monotone when the corpus mean changes support.
    grid = np.linspace(0.0, 1.0, 1001)
    costs = np.asarray([objective(float(value)) for value in grid])
    best = int(costs.argmin())
    lo, hi = grid[max(0, best - 1)], grid[min(1000, best + 1)]
    fine = np.linspace(lo, hi, 1001)
    return float(fine[int(np.argmin([objective(float(value)) for value in fine]))])


def _bounded_mass_projection(
    values: NDArray[np.float64], mass: float, *, positive_reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Project onto fixed mass and preserve every positive reference support edge."""
    if mass <= 0:
        return np.zeros_like(values)
    if mass > len(values) + 1e-7:
        raise ValueError(f"type mass {mass} exceeds {len(values)} unit-bounded support edges")
    if (positive_reference <= 0).any():
        raise ValueError("positive_reference must be strictly positive on projected support")
    # Reserve a scale-relative, exactly affordable amount for every true edge.
    # This has no absolute epsilon, so arbitrarily tiny true masses remain valid.
    lower = mass * 1e-6 * positive_reference / float(positive_reference.sum())
    capacity = 1.0 - lower
    remaining = mass - float(lower.sum())
    shifted = np.maximum(values - lower, 0.0)
    lo, hi = float(shifted.min() - 1.0), float(shifted.max())
    for _ in range(64):
        middle = (lo + hi) / 2.0
        projected = np.clip(shifted - middle, 0.0, capacity)
        if float(projected.sum()) > remaining:
            lo = middle
        else:
            hi = middle
    return lower + np.clip(shifted - (lo + hi) / 2.0, 0.0, capacity)


def calibrate_motif_shift(
    true_vval: NDArray[np.floating],
    predicted_vval: NDArray[np.floating],
    training_true: NDArray[np.floating],
) -> MotifShiftCalibration:
    """Fit one entropy-matching lambda per edge type on the complete V_val."""
    truth = _matrix(true_vval, "true_vval")
    predicted = _matrix(predicted_vval, "predicted_vval")
    if truth.shape != predicted.shape:
        raise ValueError("true_vval and predicted_vval must have identical shape")
    mean, floors = training_statistics(training_true)
    lambdas = tuple(
        _entropy_lambda(truth, predicted, mean.astype(np.float64), indices)
        for indices in _TYPE_INDEX
    )
    return MotifShiftCalibration(lambdas, mean, floors)


def apply_motif_shift(
    mode: str,
    true_weights: NDArray[np.floating],
    predicted_weights: NDArray[np.floating],
    calibration: MotifShiftCalibration,
) -> NDArray[np.float32]:
    """Construct one of the four section 7.6.1 counterfactual graph banks.

    Entropy calibration includes every V_val row, including self rows.  The
    literal ``level_true`` rescaling preserves the predicted profile and may
    exceed one when its predicted mass is very small; those are diagnostic
    count weights accepted by the reader. ``content_true`` instead uses a
    bounded fixed-mass projection because it promises true support and mass.
    """
    truth = _matrix(true_weights, "true_weights")
    predicted = _matrix(predicted_weights, "predicted_weights")
    if truth.shape != predicted.shape:
        raise ValueError("true_weights and predicted_weights must have identical shape")
    if mode not in {"blur_true", "presence_true", "level_true", "content_true"}:
        raise ValueError(f"unknown motif shift mode {mode!r}")
    output = np.zeros_like(truth)
    mean = np.asarray(calibration.train_mean, dtype=np.float64)
    for kind, indices in enumerate(_TYPE_INDEX):
        t, p = truth[:, indices], predicted[:, indices]
        tm, pm = t.sum(axis=1), p.sum(axis=1)
        if mode == "blur_true":
            lam = calibration.lambdas[kind]
            output[:, indices] = (1.0 - lam) * t + lam * mean[indices][None, :]
        elif mode == "presence_true":
            present = tm > 0
            usable = present & (pm >= calibration.nonzero_floor[kind])
            output[np.ix_(usable, indices)] = p[usable]
            fallback = present & ~usable
            output[np.ix_(fallback, indices)] = t[fallback]
        elif mode == "level_true":
            usable = pm > 0
            scale = np.divide(tm, pm, out=np.zeros_like(tm), where=usable)
            output[:, indices] = p * scale[:, None]
        else:
            # True presence, support and mass, with predicted-like sharpness.
            # Mixing is restricted to true support, then projected onto the
            # unit-bounded fixed-mass simplex. This can change profile shape;
            # the realized entropy residual is therefore reported by the driver.
            lam = calibration.lambdas[kind]
            support = t > 0
            restricted_mean = mean[indices][None, :] * support
            mixed = (1.0 - lam) * t + lam * restricted_mean
            for row in range(len(mixed)):
                active = support[row]
                output[row, indices[active]] = _bounded_mass_projection(
                    mixed[row, active],
                    float(tm[row]),
                    positive_reference=t[row, active],
                )
    return output.astype(np.float32)
