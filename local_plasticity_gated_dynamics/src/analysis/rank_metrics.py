"""Rank and dimensionality metrics with explicit numerical conventions.

The effective rank uses the Shannon entropy of the *singular values*.  The
top-k energy uses squared singular values, while participation ratio is
defined on a non-negative variance spectrum.  Keeping these conventions
separate avoids the common ambiguity between singular-value mass and
variance explained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _finite_matrix(matrix: ArrayLike, *, name: str = "matrix") -> FloatArray:
    raw = np.asarray(matrix)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_tolerance(tolerance: float | None) -> float | None:
    if tolerance is None:
        return None
    if isinstance(tolerance, bool) or not np.isscalar(tolerance):
        raise TypeError("tolerance must be a non-negative finite scalar")
    value = float(tolerance)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tolerance must be a non-negative finite scalar")
    return value


def singular_values(matrix: ArrayLike) -> FloatArray:
    """Return descending singular values of a finite real matrix."""

    return np.linalg.svd(_finite_matrix(matrix), compute_uv=False)


def _nonzero_singular_values(
    matrix: FloatArray, *, tolerance: float | None
) -> FloatArray:
    values = np.linalg.svd(matrix, compute_uv=False)
    if values[0] == 0.0:
        return values[:0]
    threshold = (
        max(matrix.shape) * np.finfo(values.dtype).eps * values[0]
        if tolerance is None
        else tolerance
    )
    return values[values > threshold]


def effective_rank(matrix: ArrayLike, *, tolerance: float | None = None) -> float:
    """Return entropy-based effective rank of a matrix.

    Singular values below ``tolerance`` are excluded.  With the default
    tolerance, the standard numerical-rank threshold is used.  A zero matrix
    has effective rank zero.
    """

    array = _finite_matrix(matrix)
    tolerance = _validate_tolerance(tolerance)
    values = _nonzero_singular_values(array, tolerance=tolerance)
    if values.size == 0:
        return 0.0
    probabilities = values / np.sum(values)
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(np.exp(entropy))


def top_k_singular_energy(matrix: ArrayLike, k: int) -> float:
    """Return the fraction of Frobenius energy in the largest ``k`` modes.

    The fraction is ``sum(s[:k]**2) / sum(s**2)``.  It is defined as zero for
    an all-zero matrix.
    """

    array = _finite_matrix(matrix)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if not 1 <= int(k) <= min(array.shape):
        raise ValueError("k must be between 1 and min(matrix.shape)")
    values = np.linalg.svd(array, compute_uv=False)
    squared = values * values
    total = float(np.sum(squared))
    if total == 0.0:
        return 0.0
    return float(np.sum(squared[: int(k)]) / total)


def participation_ratio_from_spectrum(spectrum: ArrayLike) -> float:
    """Return ``(sum(lambda))**2 / sum(lambda**2)`` for a variance spectrum."""

    raw = np.asarray(spectrum)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("spectrum must be a real numeric array")
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("spectrum must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("spectrum must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("spectrum must be non-negative")
    denominator = float(np.sum(values * values))
    if denominator == 0.0:
        return 0.0
    numerator = float(np.sum(values)) ** 2
    return numerator / denominator


def participation_ratio(activity: ArrayLike, *, center: bool = True) -> float:
    """Estimate activity dimension from the sample-by-feature covariance.

    ``activity`` must be shaped ``(samples, features)``.  The covariance
    eigenvalues are computed from squared singular values; their common
    normalization cancels from the participation ratio.
    """

    array = _finite_matrix(activity, name="activity")
    if not isinstance(center, (bool, np.bool_)):
        raise TypeError("center must be boolean")
    if center:
        if array.shape[0] < 2:
            raise ValueError("centered participation ratio requires at least two samples")
        array = array - np.mean(array, axis=0, keepdims=True)
    values = np.linalg.svd(array, compute_uv=False)
    return participation_ratio_from_spectrum(values * values)


@dataclass(frozen=True)
class RankSummary:
    """Compact rank summary for one matrix."""

    effective_rank: float
    top_k_singular_energy: float
    numerical_rank: int
    k: int


def rank_summary(
    matrix: ArrayLike, *, k: int, tolerance: float | None = None
) -> RankSummary:
    """Compute effective rank, top-k energy, and numerical rank consistently."""

    array = _finite_matrix(matrix)
    tolerance = _validate_tolerance(tolerance)
    nonzero = _nonzero_singular_values(array, tolerance=tolerance)
    return RankSummary(
        effective_rank=effective_rank(array, tolerance=tolerance),
        top_k_singular_energy=top_k_singular_energy(array, k),
        numerical_rank=int(nonzero.size),
        k=int(k),
    )
