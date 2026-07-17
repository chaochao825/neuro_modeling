"""Leakage-safe scalar metrics for the associative-actuator trend panel."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _vector(value: ArrayLike, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric vector")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector of length >= 2")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def train_reference_variance(train_targets: ArrayLike) -> float:
    """Fit the held-out error denominator using training targets only."""

    targets = _vector(train_targets, name="train_targets")
    variance = float(np.mean((targets - np.mean(targets)) ** 2))
    if variance <= 0.0:
        raise ValueError("training targets must have positive variance")
    return variance


def train_normalized_score(
    targets: ArrayLike,
    predictions: ArrayLike,
    *,
    train_variance: float,
) -> float:
    """Return ``1 - heldout MSE / train-fitted target variance``."""

    truth = _vector(targets, name="targets")
    predicted = _vector(predictions, name="predictions")
    if truth.shape != predicted.shape:
        raise ValueError("targets and predictions must have identical shape")
    denominator = float(train_variance)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("train_variance must be finite and positive")
    return float(1.0 - np.mean((truth - predicted) ** 2) / denominator)


def sign_accuracy(targets: ArrayLike, predictions: ArrayLike) -> float:
    truth = _vector(targets, name="targets")
    predicted = _vector(predictions, name="predictions")
    if truth.shape != predicted.shape:
        raise ValueError("targets and predictions must have identical shape")
    truth_sign = np.where(truth >= 0.0, 1, -1)
    prediction_sign = np.where(predicted >= 0.0, 1, -1)
    return float(np.mean(truth_sign == prediction_sign))


def matched_rms_scale(reference: ArrayLike, raw_control: ArrayLike) -> float:
    """Fit a positive RMS-only functional-budget scale on training blocks."""

    target = _vector(reference, name="reference")
    control = _vector(raw_control, name="raw_control")
    if target.shape != control.shape:
        raise ValueError("reference and raw_control must have identical shape")
    target_rms = float(np.sqrt(np.mean(target**2)))
    control_rms = float(np.sqrt(np.mean(control**2)))
    if target_rms <= 0.0 or control_rms <= 0.0:
        raise ValueError("reference and raw_control must have positive RMS")
    return target_rms / control_rms


__all__ = [
    "matched_rms_scale",
    "sign_accuracy",
    "train_normalized_score",
    "train_reference_variance",
]
