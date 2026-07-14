"""Post-hoc probability calibration of cached scorer logits (baseline ``B0+cal``).

Implements the protocol Sec 2 ``B0+cal`` calibrators: temperature scaling (primary)
and Platt scaling (disclosed alongside). Both are strictly monotone increasing maps
of the logit (temperature by construction; Platt whenever the fitted scale is
positive), so rank metrics (AUROC/AUPRC) are unchanged — only threshold- and
sum-derived quantities move. Fitting minimizes the binary cross-entropy NLL on a
held-out labeled artifact (the balanced validation pairs); the balanced 1:1 prior is
a disclosed property of the fit, absorbed downstream by density-matched thresholding.

No dependence on ``torch`` — plain numpy plus ``scipy.optimize``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

_TEMPERATURE_BOUNDS = (1e-3, 1e3)


def stable_sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable elementwise sigmoid (mirrors `ScoresArtifact.probs`)."""
    out = np.empty_like(logits)
    positive = logits >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_l = np.exp(logits[~positive])
    out[~positive] = exp_l / (1.0 + exp_l)
    return out


def bce_nll(logits: NDArray[np.float64], labels: NDArray[np.float64]) -> float:
    """Mean binary cross-entropy from raw logits (numerically stable).

    Uses ``max(z, 0) - z*y + log1p(exp(-|z|))``, exact for all logit magnitudes.

    Args:
        logits: Shape ``(n,)`` raw logits.
        labels: Shape ``(n,)`` labels in ``{0, 1}`` (float).

    Returns:
        The mean NLL.

    Raises:
        ValueError: If `logits` is empty or shapes disagree.
    """
    if logits.size == 0:
        raise ValueError("logits must be non-empty")
    if logits.shape != labels.shape:
        raise ValueError(f"shape mismatch: logits {logits.shape} vs labels {labels.shape}")
    per_row = np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(per_row))


@dataclass(frozen=True)
class TemperatureCalibration:
    """Temperature scaling: ``p = sigmoid(logit / temperature)``.

    Attributes:
        temperature: The fitted temperature ``T > 0``. ``T > 1`` softens
            (overconfident scorer), ``T < 1`` sharpens.
        val_nll_before: Mean BCE NLL of the raw logits on the fit set.
        val_nll_after: Mean BCE NLL of the calibrated logits on the fit set.
    """

    temperature: float
    val_nll_before: float
    val_nll_after: float

    def apply(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return calibrated probabilities for `logits` (strictly monotone)."""
        return stable_sigmoid(logits / self.temperature)

    def to_jsonable(self) -> dict[str, object]:
        """Return a plain-dict, JSON-ready representation."""
        return {
            "method": "temperature",
            "temperature": self.temperature,
            "val_nll_before": self.val_nll_before,
            "val_nll_after": self.val_nll_after,
        }


@dataclass(frozen=True)
class PlattCalibration:
    """Platt scaling: ``p = sigmoid(scale * logit + bias)``.

    Attributes:
        scale: Fitted multiplicative logit coefficient (positive ⇒ monotone; a
            negative fitted scale indicates an anti-correlated scorer and is
            surfaced, not hidden).
        bias: Fitted additive logit offset.
        val_nll_before: Mean BCE NLL of the raw logits on the fit set.
        val_nll_after: Mean BCE NLL of the calibrated logits on the fit set.
    """

    scale: float
    bias: float
    val_nll_before: float
    val_nll_after: float

    def apply(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return calibrated probabilities for `logits`."""
        return stable_sigmoid(self.scale * logits + self.bias)

    def to_jsonable(self) -> dict[str, object]:
        """Return a plain-dict, JSON-ready representation."""
        return {
            "method": "platt",
            "scale": self.scale,
            "bias": self.bias,
            "val_nll_before": self.val_nll_before,
            "val_nll_after": self.val_nll_after,
        }


def fit_temperature(
    logits: NDArray[np.float64], labels: NDArray[np.int8] | NDArray[np.int64]
) -> TemperatureCalibration:
    """Fit temperature scaling by minimizing BCE NLL on ``(logits, labels)``.

    Deterministic: bounded scalar minimization over ``log T`` (no random state).

    Args:
        logits: Shape ``(n,)`` raw logits of the fit set.
        labels: Shape ``(n,)`` labels in ``{0, 1}``.

    Returns:
        The fitted `TemperatureCalibration`.

    Raises:
        ValueError: If inputs are empty, misshapen, or single-class.
    """
    y = _validated_labels(logits, labels)
    nll_before = bce_nll(logits, y)

    def objective(log_t: float) -> float:
        return bce_nll(logits / float(np.exp(log_t)), y)

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(_TEMPERATURE_BOUNDS[0])), float(np.log(_TEMPERATURE_BOUNDS[1]))),
        method="bounded",
        options={"xatol": 1e-10},
    )
    temperature = float(np.exp(result.x))
    return TemperatureCalibration(
        temperature=temperature,
        val_nll_before=nll_before,
        val_nll_after=bce_nll(logits / temperature, y),
    )


def fit_platt(
    logits: NDArray[np.float64], labels: NDArray[np.int8] | NDArray[np.int64]
) -> PlattCalibration:
    """Fit Platt scaling ``sigmoid(a * logit + b)`` by minimizing BCE NLL.

    Deterministic: L-BFGS-B from the fixed initial point ``(a, b) = (1, 0)``.

    Args:
        logits: Shape ``(n,)`` raw logits of the fit set.
        labels: Shape ``(n,)`` labels in ``{0, 1}``.

    Returns:
        The fitted `PlattCalibration`.

    Raises:
        ValueError: If inputs are empty, misshapen, or single-class.
    """
    y = _validated_labels(logits, labels)
    nll_before = bce_nll(logits, y)

    def objective(params: NDArray[np.float64]) -> float:
        return bce_nll(params[0] * logits + params[1], y)

    result = minimize(objective, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    scale, bias = float(result.x[0]), float(result.x[1])
    return PlattCalibration(
        scale=scale,
        bias=bias,
        val_nll_before=nll_before,
        val_nll_after=bce_nll(scale * logits + bias, y),
    )


def _validated_labels(
    logits: NDArray[np.float64], labels: NDArray[np.int8] | NDArray[np.int64]
) -> NDArray[np.float64]:
    """Validate the fit set and return labels as float64."""
    if logits.size == 0:
        raise ValueError("cannot fit a calibrator on an empty set")
    if logits.shape != labels.shape:
        raise ValueError(f"shape mismatch: logits {logits.shape} vs labels {labels.shape}")
    unique = np.unique(labels)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"labels must be in {{0, 1}}; got values {unique.tolist()}")
    if unique.size < 2:
        raise ValueError("cannot fit a calibrator on single-class labels")
    return labels.astype(np.float64)
