"""Seed-level metrics and multiplicity helpers for Exp31."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def binary_accuracy(targets: ArrayLike, predictions: ArrayLike) -> float:
    truth = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if truth.ndim != 1 or predicted.shape != truth.shape or truth.size < 1:
        raise ValueError("targets and predictions must be paired non-empty vectors")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(predicted)):
        raise ValueError("targets and predictions must be finite")
    truth_sign = np.where(truth >= 0.0, 1.0, -1.0)
    predicted_sign = np.where(predicted >= 0.0, 1.0, -1.0)
    return float(np.mean(truth_sign == predicted_sign))


def paired_bootstrap_interval(
    differences: ArrayLike,
    *,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a finite vector of length >= 2")
    if isinstance(n_resamples, (bool, np.bool_)) or int(n_resamples) < 100:
        raise ValueError("n_resamples must be an integer >= 100")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(values.size, size=(int(n_resamples), values.size))
    means = np.mean(values[indices], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def sign_flip_pvalue(
    centered_differences: ArrayLike,
    *,
    n_resamples: int,
    seed: int,
) -> float:
    """One-sided paired random-sign p-value for a positive mean."""

    values = np.asarray(centered_differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("centered_differences must be a finite vector")
    if isinstance(n_resamples, (bool, np.bool_)) or int(n_resamples) < 100:
        raise ValueError("n_resamples must be an integer >= 100")
    observed = float(np.mean(values))
    rng = np.random.default_rng(int(seed))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_resamples), values.size))
    null_means = np.mean(signs * values, axis=1)
    return float((1 + np.sum(null_means >= observed)) / (int(n_resamples) + 1))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    if not pvalues:
        raise ValueError("pvalues cannot be empty")
    for value in pvalues.values():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(pvalues[name]))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


__all__ = [
    "binary_accuracy",
    "holm_adjust",
    "paired_bootstrap_interval",
    "sign_flip_pvalue",
]
