"""Leakage-safe PCA, subspace, latent, and rollout metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _finite_real_array(
    values: ArrayLike, *, name: str, ndim: int | None = None, min_ndim: int | None = None
) -> FloatArray:
    raw = np.asarray(values)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if min_ndim is not None and array.ndim < min_ndim:
        raise ValueError(f"{name} must have at least {min_ndim} dimensions")
    if array.size == 0 or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _read_only_copy(values: FloatArray) -> FloatArray:
    copy = np.array(values, dtype=np.float64, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class FittedPCASubspace:
    """An immutable PCA fit whose normalization statistics come from training.

    The basis columns are available through :attr:`basis_`.  This object has
    no ``fit`` method: creating a different fit requires another explicit call
    to :func:`fit_train_pca` with a training array.
    """

    mean_: FloatArray
    scale_: FloatArray
    components_: FloatArray
    explained_variance_: FloatArray
    explained_variance_ratio_: FloatArray
    n_train_samples_: int
    normalized_: bool
    fit_sample_ids_: tuple[object, ...] | None

    @property
    def basis_(self) -> FloatArray:
        """Return a copy of the feature-by-component orthonormal basis."""

        return self.components_.T.copy()

    def transform(self, values: ArrayLike) -> FloatArray:
        """Project data with the frozen training normalization and PCA basis."""

        array = _finite_real_array(values, name="values", ndim=2)
        if array.shape[1] != self.mean_.size:
            raise ValueError("values feature count does not match the fitted PCA")
        standardized = (array - self.mean_) / self.scale_
        return standardized @ self.components_.T

    def inverse_transform(self, scores: ArrayLike) -> FloatArray:
        """Map latent scores back to the original feature coordinates."""

        array = _finite_real_array(scores, name="scores", ndim=2)
        if array.shape[1] != self.components_.shape[0]:
            raise ValueError("scores component count does not match the fitted PCA")
        standardized = array @ self.components_
        return standardized * self.scale_ + self.mean_


def fit_train_pca(
    x_train: ArrayLike,
    n_components: int,
    *,
    normalize: bool = False,
    sample_ids: ArrayLike | None = None,
) -> FittedPCASubspace:
    """Fit PCA and optional z-scoring on training samples only.

    Constant training features are centered but assigned scale one.  This
    avoids division by zero without borrowing variance from validation/test
    data.  At most ``min(n_features, n_train - 1)`` centered components are
    allowed.
    """

    train = _finite_real_array(x_train, name="x_train", ndim=2)
    if train.shape[0] < 2:
        raise ValueError("x_train must contain at least two samples")
    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
        raise TypeError("n_components must be an integer")
    maximum = min(train.shape[1], train.shape[0] - 1)
    if not 1 <= int(n_components) <= maximum:
        raise ValueError(f"n_components must be between 1 and {maximum}")
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be boolean")

    identifiers: tuple[object, ...] | None = None
    if sample_ids is not None:
        raw_ids = np.asarray(sample_ids, dtype=object)
        if raw_ids.ndim != 1 or raw_ids.size != train.shape[0]:
            raise ValueError("sample_ids must be one-dimensional and match x_train rows")
        for item in raw_ids.tolist():
            if item is None:
                raise ValueError("sample_ids cannot contain missing values")
            try:
                missing = item != item
            except (TypeError, ValueError):
                raise ValueError("sample_ids must contain scalar identifiers") from None
            if not isinstance(missing, (bool, np.bool_)):
                raise ValueError("sample_ids must contain scalar identifiers")
            if bool(missing):
                raise ValueError("sample_ids cannot contain missing values")
        identifiers = tuple(raw_ids.tolist())

    mean = np.mean(train, axis=0)
    centered = train - mean
    if normalize:
        empirical_scale = np.std(centered, axis=0, ddof=0)
        scale = np.where(empirical_scale > 0.0, empirical_scale, 1.0)
    else:
        scale = np.ones(train.shape[1], dtype=np.float64)
    standardized = centered / scale
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    variance_all = singular * singular / (train.shape[0] - 1)
    total_variance = float(np.sum(variance_all))
    explained_variance = variance_all[: int(n_components)]
    explained_ratio = (
        explained_variance / total_variance
        if total_variance > 0.0
        else np.zeros(int(n_components), dtype=np.float64)
    )
    return FittedPCASubspace(
        mean_=_read_only_copy(mean),
        scale_=_read_only_copy(scale),
        components_=_read_only_copy(vt[: int(n_components)]),
        explained_variance_=_read_only_copy(explained_variance),
        explained_variance_ratio_=_read_only_copy(explained_ratio),
        n_train_samples_=int(train.shape[0]),
        normalized_=bool(normalize),
        fit_sample_ids_=identifiers,
    )


def _orthonormal_basis(basis: ArrayLike, *, name: str) -> FloatArray:
    array = _finite_real_array(basis, name=name, ndim=2)
    if array.shape[1] > array.shape[0]:
        raise ValueError(f"{name} cannot have more basis vectors than ambient dimensions")
    if np.linalg.matrix_rank(array) != array.shape[1]:
        raise ValueError(f"{name} columns must be linearly independent")
    orthonormal, _ = np.linalg.qr(array, mode="reduced")
    return orthonormal


def principal_angles(
    basis_a: ArrayLike, basis_b: ArrayLike, *, degrees: bool = False
) -> FloatArray:
    """Return canonical principal angles between two column subspaces."""

    if not isinstance(degrees, (bool, np.bool_)):
        raise TypeError("degrees must be boolean")
    first = _orthonormal_basis(basis_a, name="basis_a")
    second = _orthonormal_basis(basis_b, name="basis_b")
    if first.shape[0] != second.shape[0]:
        raise ValueError("bases must have the same ambient feature dimension")
    cosines = np.linalg.svd(first.T @ second, compute_uv=False)
    angles = np.arccos(np.clip(cosines, 0.0, 1.0))
    return np.degrees(angles) if degrees else angles


def subspace_overlap(basis_a: ArrayLike, basis_b: ArrayLike) -> float:
    """Return normalized projection overlap in ``[0, 1]``.

    The normalization by the smaller subspace dimension makes the score one
    when the smaller subspace is fully contained in the larger one.
    """

    first = _orthonormal_basis(basis_a, name="basis_a")
    second = _orthonormal_basis(basis_b, name="basis_b")
    if first.shape[0] != second.shape[0]:
        raise ValueError("bases must have the same ambient feature dimension")
    denominator = min(first.shape[1], second.shape[1])
    overlap = np.linalg.norm(first.T @ second, ord="fro") ** 2 / denominator
    return float(np.clip(overlap, 0.0, 1.0))


def _paired_latents(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[FloatArray, FloatArray]:
    truth = _finite_real_array(y_true, name="y_true", min_ndim=1)
    prediction = _finite_real_array(y_pred, name="y_pred", min_ndim=1)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if truth.ndim == 1:
        truth = truth[:, None]
        prediction = prediction[:, None]
    else:
        truth = truth.reshape(-1, truth.shape[-1])
        prediction = prediction.reshape(-1, prediction.shape[-1])
    if truth.shape[0] < 2:
        raise ValueError("latent metrics require at least two observations")
    return truth, prediction


def latent_r2_per_dimension(y_true: ArrayLike, y_pred: ArrayLike) -> FloatArray:
    """Return one finite R-squared value per latent dimension."""

    truth, prediction = _paired_latents(y_true, y_pred)
    residual = np.sum((truth - prediction) ** 2, axis=0)
    total = np.sum((truth - np.mean(truth, axis=0, keepdims=True)) ** 2, axis=0)
    result = np.empty(truth.shape[1], dtype=np.float64)
    varying = total > np.finfo(np.float64).eps
    result[varying] = 1.0 - residual[varying] / total[varying]
    result[~varying] = np.where(
        residual[~varying] <= np.finfo(np.float64).eps, 1.0, 0.0
    )
    return result


def latent_r2(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    multioutput: Literal["variance_weighted", "uniform_average", "raw_values"] = (
        "variance_weighted"
    ),
) -> float | FloatArray:
    """Return aggregate or per-dimension latent R-squared.

    ``variance_weighted`` weights dimensions using variance from ``y_true``;
    it never fits a transformation to predictions or held-out inputs.
    """

    valid = {"variance_weighted", "uniform_average", "raw_values"}
    if multioutput not in valid:
        raise ValueError(f"multioutput must be one of {sorted(valid)}")
    truth, prediction = _paired_latents(y_true, y_pred)
    per_dimension = latent_r2_per_dimension(truth, prediction)
    if multioutput == "raw_values":
        return per_dimension
    if multioutput == "uniform_average":
        return float(np.mean(per_dimension))
    total = np.sum((truth - np.mean(truth, axis=0, keepdims=True)) ** 2, axis=0)
    if float(np.sum(total)) == 0.0:
        return float(np.mean(per_dimension))
    return float(np.average(per_dimension, weights=total))


@dataclass(frozen=True)
class RolloutMetrics:
    """Prediction errors for an entire held-out rollout."""

    rmse: float
    mae: float
    per_horizon_rmse: FloatArray
    normalized_rmse: float | None


def rollout_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    train_reference: ArrayLike | None = None,
) -> RolloutMetrics:
    """Compute raw rollout errors and optional train-normalized RMSE.

    Arrays are shaped ``(..., horizon, latent_dim)``.  Normalized RMSE is only
    produced when an explicit training reference is supplied; test targets
    are never used to fit a scale.
    """

    truth = _finite_real_array(y_true, name="y_true", min_ndim=2)
    prediction = _finite_real_array(y_pred, name="y_pred", min_ndim=2)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    error = prediction - truth
    rmse = float(np.sqrt(np.mean(error * error)))
    mae = float(np.mean(np.abs(error)))
    horizon_axis = truth.ndim - 2
    reduction_axes = tuple(axis for axis in range(truth.ndim) if axis != horizon_axis)
    per_horizon = np.sqrt(np.mean(error * error, axis=reduction_axes))

    normalized_rmse: float | None = None
    if train_reference is not None:
        reference = _finite_real_array(
            train_reference, name="train_reference", min_ndim=2
        )
        if reference.shape[-1] != truth.shape[-1]:
            raise ValueError("train_reference latent dimension does not match rollout")
        flattened = reference.reshape(-1, reference.shape[-1])
        if flattened.shape[0] < 2:
            raise ValueError("train_reference must contain at least two observations")
        scale = np.std(flattened, axis=0, ddof=0)
        if np.any(scale <= np.finfo(np.float64).eps):
            raise ValueError("train_reference has a constant latent dimension")
        normalized_rmse = float(np.sqrt(np.mean((error / scale) ** 2)))
    return RolloutMetrics(
        rmse=rmse,
        mae=mae,
        per_horizon_rmse=np.asarray(per_horizon, dtype=np.float64),
        normalized_rmse=normalized_rmse,
    )


def rollout_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    normalized: bool = False,
    train_reference: ArrayLike | None = None,
) -> float:
    """Return scalar rollout RMSE, optionally normalized by training variance."""

    if not isinstance(normalized, (bool, np.bool_)):
        raise TypeError("normalized must be boolean")
    if normalized and train_reference is None:
        raise ValueError("normalized rollout error requires train_reference")
    metrics = rollout_metrics(y_true, y_pred, train_reference=train_reference)
    if normalized:
        assert metrics.normalized_rmse is not None
        return metrics.normalized_rmse
    return metrics.rmse
