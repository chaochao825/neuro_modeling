"""Paired Jacobian-outlier and trial-safe empirical Hankel metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.analysis.rank_stage_metrics import matrix_rank_summary


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _finite_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _readonly(value: ArrayLike, *, complex_values: bool = False) -> NDArray:
    dtype = np.complex128 if complex_values else np.float64
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _trial_list(
    trials: ArrayLike | Sequence[ArrayLike], *, name: str
) -> tuple[FloatArray, ...]:
    try:
        raw_candidate = np.asarray(trials)
    except ValueError:
        raw_candidate = np.asarray(trials, dtype=object)
    if raw_candidate.dtype.kind != "O" and raw_candidate.ndim in (2, 3):
        raw = raw_candidate
        if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
            raise TypeError(f"{name} must contain real numeric arrays")
        if raw.ndim == 2:
            candidates = [raw]
        else:
            candidates = [raw[index] for index in range(raw.shape[0])]
    else:
        if isinstance(trials, (str, bytes)):
            raise TypeError(f"{name} must be a trial array or sequence of trial arrays")
        try:
            candidates = list(trials)
        except TypeError as error:
            raise TypeError(
                f"{name} must be a trial array or sequence of trial arrays"
            ) from error
    if not candidates:
        raise ValueError(f"{name} must contain at least one trial")
    arrays = tuple(
        _finite_matrix(candidate, name=f"{name}[{index}]")
        for index, candidate in enumerate(candidates)
    )
    feature_counts = {array.shape[1] for array in arrays}
    if len(feature_counts) != 1:
        raise ValueError(f"all {name} trials must have the same feature count")
    return arrays


@dataclass(frozen=True)
class JacobianOutlierSummary:
    """Right-edge outliers relative to a paired bulk Jacobian."""

    outlier_count: int
    bulk_tail_count: int
    excess_outlier_count: int
    bulk_right_edge: float
    edge_quantile: float
    tolerance: float
    target_eigenvalues: ComplexArray
    bulk_eigenvalues: ComplexArray


def jacobian_outlier_summary(
    jacobian: ArrayLike,
    paired_bulk_jacobian: ArrayLike,
    *,
    edge_quantile: float = 0.99,
    tolerance: float = 1e-9,
) -> JacobianOutlierSummary:
    """Count dynamically relevant eigenvalues beyond a paired bulk right edge.

    The two Jacobians should be evaluated at the same state, gain, activation
    derivative, and time constants.  The paired bulk matrix normally removes
    only the task-plastic component.  An outlier is a target eigenvalue whose
    real part exceeds the configured quantile of the paired bulk real parts by
    more than ``tolerance``.  This definition is intentionally distinct from
    instability relative to zero.
    """

    target = _finite_matrix(jacobian, name="jacobian")
    bulk = _finite_matrix(paired_bulk_jacobian, name="paired_bulk_jacobian")
    if target.shape[0] != target.shape[1]:
        raise ValueError("jacobian must be square")
    if bulk.shape != target.shape:
        raise ValueError("paired_bulk_jacobian must match the square jacobian")
    if (
        isinstance(edge_quantile, (bool, np.bool_))
        or not np.isscalar(edge_quantile)
        or not np.isfinite(float(edge_quantile))
        or not 0.0 < float(edge_quantile) <= 1.0
    ):
        raise ValueError("edge_quantile must be a finite scalar in (0, 1]")
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not np.isscalar(tolerance)
        or not np.isfinite(float(tolerance))
        or float(tolerance) < 0.0
    ):
        raise ValueError("tolerance must be a non-negative finite scalar")

    target_eigenvalues = np.asarray(np.linalg.eigvals(target), dtype=np.complex128)
    bulk_eigenvalues = np.asarray(np.linalg.eigvals(bulk), dtype=np.complex128)
    quantile = float(edge_quantile)
    margin = float(tolerance)
    right_edge = float(np.quantile(np.real(bulk_eigenvalues), quantile))
    target_outliers = np.real(target_eigenvalues) > right_edge + margin
    bulk_tail = np.real(bulk_eigenvalues) > right_edge + margin
    outlier_count = int(np.count_nonzero(target_outliers))
    bulk_tail_count = int(np.count_nonzero(bulk_tail))
    return JacobianOutlierSummary(
        outlier_count=outlier_count,
        bulk_tail_count=bulk_tail_count,
        excess_outlier_count=outlier_count - bulk_tail_count,
        bulk_right_edge=right_edge,
        edge_quantile=quantile,
        tolerance=margin,
        target_eigenvalues=_readonly(target_eigenvalues, complex_values=True),
        bulk_eigenvalues=_readonly(bulk_eigenvalues, complex_values=True),
    )


@dataclass(frozen=True)
class HankelPreprocessor:
    """Immutable centering/scaling fit from training trials only."""

    mean_: FloatArray
    scale_: FloatArray
    normalized_: bool
    n_train_trials_: int
    n_train_observations_: int

    def transform(self, trial: ArrayLike) -> FloatArray:
        """Apply frozen training statistics to one time-by-feature trial."""

        array = _finite_matrix(trial, name="trial")
        if array.shape[1] != self.mean_.size:
            raise ValueError("trial feature count does not match the preprocessor")
        return (array - self.mean_) / self.scale_


def fit_hankel_preprocessor(
    train_trials: ArrayLike | Sequence[ArrayLike], *, normalize: bool = True
) -> HankelPreprocessor:
    """Fit population mean and optional scale using training trials only."""

    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be boolean")
    arrays = _trial_list(train_trials, name="train_trials")
    n_observations = int(sum(array.shape[0] for array in arrays))
    total = np.zeros(arrays[0].shape[1], dtype=np.float64)
    squared_total = np.zeros_like(total)
    for array in arrays:
        total += np.sum(array, axis=0)
        squared_total += np.sum(array * array, axis=0)
    mean = total / n_observations
    if normalize:
        variance = np.maximum(squared_total / n_observations - mean * mean, 0.0)
        empirical = np.sqrt(variance)
        scale = np.where(empirical > 0.0, empirical, 1.0)
    else:
        scale = np.ones(arrays[0].shape[1], dtype=np.float64)
    return HankelPreprocessor(
        mean_=_readonly(mean),
        scale_=_readonly(scale),
        normalized_=bool(normalize),
        n_train_trials_=len(arrays),
        n_train_observations_=n_observations,
    )


def _positive_integer(value: int, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_seed(value: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    return int(value)


def _probability(value: float, *, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
    ):
        raise ValueError(f"{name} must be a finite scalar in (0, 1)")
    return float(value)


def _trial_window_matrices(
    values: FloatArray, *, past: int, future: int
) -> tuple[FloatArray, FloatArray] | None:
    n_windows = values.shape[0] - past - future + 1
    if n_windows <= 0:
        return None
    past_blocks = np.stack(
        [values[index : index + past].reshape(-1) for index in range(n_windows)]
    )
    future_blocks = np.stack(
        [
            values[index + past : index + past + future].reshape(-1)
            for index in range(n_windows)
        ]
    )
    return past_blocks, future_blocks


@dataclass(frozen=True)
class HankelNoiseFloor:
    """Training-fold parallel-analysis threshold for cross-Hankel singular values."""

    singular_value_threshold: float
    singular_value_thresholds: FloatArray
    quantile: float
    n_permutations: int
    seed: int
    null_max_singular_values: FloatArray
    null_singular_values: FloatArray
    past_lags: int
    future_lags: int
    n_features: int
    n_train_trials: int
    n_train_windows: int
    method: str
    preprocessor_mean: FloatArray
    preprocessor_scale: FloatArray


def fit_hankel_noise_floor(
    train_trials: ArrayLike | Sequence[ArrayLike],
    *,
    past_lags: int,
    future_lags: int,
    preprocessor: HankelPreprocessor,
    n_permutations: int = 100,
    quantile: float = 0.95,
    seed: int,
) -> HankelNoiseFloor:
    """Fit an optional noise floor by permuting future windows within trials.

    For every training-fold permutation, future-window rows are shuffled only
    against past rows from the same trial.  Each observed singular value is
    compared with the configured quantile of the corresponding null spectral
    mode.  This is a finite-sample parallel-analysis heuristic, not a claim
    that the resulting count is the true nonlinear system order.
    """

    if not isinstance(preprocessor, HankelPreprocessor):
        raise TypeError("preprocessor must be fit on training trials")
    past = _positive_integer(past_lags, name="past_lags")
    future = _positive_integer(future_lags, name="future_lags")
    permutations = _positive_integer(n_permutations, name="n_permutations")
    if permutations < 10:
        raise ValueError("n_permutations must be at least 10")
    probability = _probability(quantile, name="quantile")
    seed_value = _nonnegative_seed(seed)
    arrays = _trial_list(train_trials, name="train_trials")
    n_features = arrays[0].shape[1]
    if preprocessor.mean_.size != n_features:
        raise ValueError("preprocessor feature count does not match train_trials")

    blocks: list[tuple[FloatArray, FloatArray]] = []
    n_windows = 0
    has_permutable_trial = False
    for trial in arrays:
        pair = _trial_window_matrices(
            preprocessor.transform(trial), past=past, future=future
        )
        if pair is None:
            continue
        blocks.append(pair)
        n_windows += pair[0].shape[0]
        has_permutable_trial |= pair[0].shape[0] > 1
    if not blocks:
        raise ValueError(
            "no training trial is long enough for the requested past and future lags"
        )
    if not has_permutable_trial:
        raise ValueError(
            "noise-floor fitting requires a trial with at least two windows"
        )

    rng = np.random.default_rng(seed_value)
    spectrum_size = min(future * n_features, past * n_features)
    null_spectra = np.empty((permutations, spectrum_size), dtype=np.float64)
    identity_by_size: dict[int, NDArray[np.int64]] = {}
    for permutation_index in range(permutations):
        cross = np.zeros((future * n_features, past * n_features), dtype=np.float64)
        for past_blocks, future_blocks in blocks:
            count = past_blocks.shape[0]
            if count > 1:
                order = rng.permutation(count)
                identity = identity_by_size.setdefault(count, np.arange(count))
                if np.array_equal(order, identity):
                    order = np.roll(order, 1)
                permuted_future = future_blocks[order]
            else:
                permuted_future = future_blocks
            cross += permuted_future.T @ past_blocks
        cross /= n_windows
        null_spectra[permutation_index] = np.linalg.svd(cross, compute_uv=False)
    thresholds = np.quantile(null_spectra, probability, axis=0)
    return HankelNoiseFloor(
        singular_value_threshold=float(thresholds[0]),
        singular_value_thresholds=_readonly(thresholds),
        quantile=probability,
        n_permutations=permutations,
        seed=seed_value,
        null_max_singular_values=_readonly(null_spectra[:, 0]),
        null_singular_values=_readonly(null_spectra),
        past_lags=past,
        future_lags=future,
        n_features=int(n_features),
        n_train_trials=len(arrays),
        n_train_windows=int(n_windows),
        method="within_trial_future_window_permutation",
        preprocessor_mean=_readonly(preprocessor.mean_),
        preprocessor_scale=_readonly(preprocessor.scale_),
    )


@dataclass(frozen=True)
class EmpiricalHankelSummary:
    """Raw and optionally noise-adjusted held-out cross-Hankel dimensions."""

    raw_numerical_rank: int
    raw_effective_rank: float
    raw_numeric_threshold: float
    noise_adjusted_dimension: int | None
    dimension_thresholds: FloatArray | None
    threshold_source: str
    dimension_interpretation: str
    singular_values: FloatArray
    n_trials: int
    n_windows: int
    n_features: int
    past_lags: int
    future_lags: int
    preprocessing: str
    moment_kind: str
    preprocessor_train_observations: int | None


def empirical_hankel_summary(
    trials: ArrayLike | Sequence[ArrayLike],
    *,
    past_lags: int,
    future_lags: int,
    preprocessor: HankelPreprocessor | None = None,
    noise_floor: HankelNoiseFloor | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-12,
    window_chunk_size: int = 256,
) -> EmpiricalHankelSummary:
    """Estimate held-out dynamical dimension without crossing trial boundaries.

    Each valid within-trial split contributes a flattened past block and a
    flattened future block.  Their cross moment is accumulated trial by trial
    and in bounded-size chunks.  No centering or scaling is fit here: callers
    may supply a :class:`HankelPreprocessor` fit on training trials, or request
    the explicit identity/no-preprocessing path with ``preprocessor=None``.
    Without a training-fitted ``noise_floor``, the numerical count is labeled
    raw and must not be interpreted as system order in noisy data.
    """

    arrays = _trial_list(trials, name="trials")
    past = _positive_integer(past_lags, name="past_lags")
    future = _positive_integer(future_lags, name="future_lags")
    chunk_size = _positive_integer(window_chunk_size, name="window_chunk_size")
    n_features = arrays[0].shape[1]
    if preprocessor is not None and not isinstance(preprocessor, HankelPreprocessor):
        raise TypeError("preprocessor must be a HankelPreprocessor or None")
    if preprocessor is not None and preprocessor.mean_.size != n_features:
        raise ValueError("preprocessor feature count does not match trials")
    if noise_floor is not None and not isinstance(noise_floor, HankelNoiseFloor):
        raise TypeError("noise_floor must be a HankelNoiseFloor or None")
    if noise_floor is not None and preprocessor is None:
        raise ValueError("noise_floor requires its training-fitted preprocessor")
    if noise_floor is not None and (
        noise_floor.past_lags != past
        or noise_floor.future_lags != future
        or noise_floor.n_features != n_features
    ):
        raise ValueError("noise_floor lags and feature count must match trials")
    if noise_floor is not None and (
        not np.array_equal(preprocessor.mean_, noise_floor.preprocessor_mean)
        or not np.array_equal(preprocessor.scale_, noise_floor.preprocessor_scale)
    ):
        raise ValueError("noise_floor and preprocessor must come from the same fit")

    cross_hankel = np.zeros((future * n_features, past * n_features), dtype=np.float64)
    n_windows = 0
    for trial in arrays:
        values = preprocessor.transform(trial) if preprocessor is not None else trial
        trial_windows = values.shape[0] - past - future + 1
        if trial_windows <= 0:
            continue
        for start in range(0, trial_windows, chunk_size):
            stop = min(start + chunk_size, trial_windows)
            starts = range(start, stop)
            past_blocks = np.stack(
                [values[index : index + past].reshape(-1) for index in starts]
            )
            future_blocks = np.stack(
                [
                    values[index + past : index + past + future].reshape(-1)
                    for index in range(start, stop)
                ]
            )
            cross_hankel += future_blocks.T @ past_blocks
            n_windows += stop - start
    if n_windows == 0:
        raise ValueError(
            "no trial is long enough for the requested past and future lags"
        )
    cross_hankel /= n_windows
    rank = matrix_rank_summary(cross_hankel, rtol=rtol, atol=atol)
    if noise_floor is None:
        adjusted_dimension = None
        dimension_thresholds = None
        threshold_source = "fixed_numeric_raw_only"
        interpretation = "raw_numerical_rank_not_system_order_without_noise_floor"
    else:
        dimension_thresholds = np.maximum(
            rank.threshold, noise_floor.singular_value_thresholds
        )
        adjusted_dimension = int(
            np.count_nonzero(rank.singular_values > dimension_thresholds)
        )
        threshold_source = "train_fitted_within_trial_permutation"
        interpretation = "noise_adjusted_cross_hankel_dimension"
    return EmpiricalHankelSummary(
        raw_numerical_rank=rank.numerical_rank,
        raw_effective_rank=rank.effective_rank,
        raw_numeric_threshold=rank.threshold,
        noise_adjusted_dimension=adjusted_dimension,
        dimension_thresholds=(
            None if dimension_thresholds is None else _readonly(dimension_thresholds)
        ),
        threshold_source=threshold_source,
        dimension_interpretation=interpretation,
        singular_values=rank.singular_values,
        n_trials=len(arrays),
        n_windows=int(n_windows),
        n_features=int(n_features),
        past_lags=past,
        future_lags=future,
        preprocessing=(
            "train_fitted_center_scale"
            if preprocessor is not None and preprocessor.normalized_
            else "train_fitted_center_only"
            if preprocessor is not None
            else "identity"
        ),
        moment_kind=(
            "train_centered_cross_moment"
            if preprocessor is not None
            else "uncentered_cross_moment"
        ),
        preprocessor_train_observations=(
            preprocessor.n_train_observations_ if preprocessor is not None else None
        ),
    )


__all__ = [
    "EmpiricalHankelSummary",
    "HankelNoiseFloor",
    "HankelPreprocessor",
    "JacobianOutlierSummary",
    "empirical_hankel_summary",
    "fit_hankel_noise_floor",
    "fit_hankel_preprocessor",
    "jacobian_outlier_summary",
]
