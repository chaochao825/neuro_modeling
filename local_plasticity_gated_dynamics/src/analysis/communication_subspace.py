"""Train-only reduced-rank communication subspaces between populations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.analysis.manifold_metrics import principal_angles, subspace_overlap


FloatArray = NDArray[np.float64]
PopulationSide = Literal["source", "target"]


def _finite_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric matrix")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _validated_sample_ids(
    sample_ids: ArrayLike | None,
    *,
    n_samples: int,
) -> tuple[object, ...] | None:
    if sample_ids is None:
        return None
    identifiers = np.asarray(sample_ids, dtype=object)
    if identifiers.ndim != 1 or identifiers.size != n_samples:
        raise ValueError("sample_ids must be one-dimensional and match training rows")
    result = tuple(identifiers.tolist())
    for item in result:
        if item is None:
            raise ValueError("sample_ids cannot contain missing values")
        try:
            missing = item != item
        except (TypeError, ValueError):
            raise ValueError("sample_ids must contain scalar identifiers") from None
        if not isinstance(missing, (bool, np.bool_)) or bool(missing):
            raise ValueError("sample_ids must contain non-missing scalar identifiers")
    return result


@dataclass(frozen=True)
class FittedCommunicationSubspace:
    """Frozen reduced-rank regression fit from source to target activity."""

    source_mean_: FloatArray
    source_scale_: FloatArray
    target_mean_: FloatArray
    target_scale_: FloatArray
    standardized_coefficients_: FloatArray
    coefficients_: FloatArray
    intercept_: FloatArray
    source_basis_: FloatArray
    target_basis_: FloatArray
    singular_values_: FloatArray
    rank_: int
    ridge_: float
    normalized_: bool
    n_train_samples_: int
    fit_sample_ids_: tuple[object, ...] | None

    def __post_init__(self) -> None:
        for name in (
            "source_mean_",
            "source_scale_",
            "target_mean_",
            "target_scale_",
            "standardized_coefficients_",
            "coefficients_",
            "intercept_",
            "source_basis_",
            "target_basis_",
            "singular_values_",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def predict(self, source_activity: ArrayLike) -> FloatArray:
        """Predict held-out target rows without any refitting."""

        source = _finite_matrix(source_activity, name="source_activity")
        if source.shape[1] != self.source_mean_.size:
            raise ValueError("source feature count differs from the training fit")
        return source @ self.coefficients_ + self.intercept_

    def transform_source(self, source_activity: ArrayLike) -> FloatArray:
        """Project source rows using frozen training normalization and basis."""

        source = _finite_matrix(source_activity, name="source_activity")
        if source.shape[1] != self.source_mean_.size:
            raise ValueError("source feature count differs from the training fit")
        standardized = (source - self.source_mean_) / self.source_scale_
        return standardized @ self.source_basis_

    def transform_target(self, target_activity: ArrayLike) -> FloatArray:
        """Project target rows using frozen training normalization and basis."""

        target = _finite_matrix(target_activity, name="target_activity")
        if target.shape[1] != self.target_mean_.size:
            raise ValueError("target feature count differs from the training fit")
        standardized = (target - self.target_mean_) / self.target_scale_
        return standardized @ self.target_basis_

    def heldout_r2(
        self,
        source_activity: ArrayLike,
        target_activity: ArrayLike,
    ) -> float:
        """Variance-weighted held-out prediction R-squared."""

        target = _finite_matrix(target_activity, name="target_activity")
        prediction = self.predict(source_activity)
        if target.shape != prediction.shape:
            raise ValueError("source/target held-out rows or target features differ")
        residual = float(np.sum((target - prediction) ** 2))
        total = float(np.sum((target - self.target_mean_) ** 2))
        if total == 0.0:
            return 1.0 if residual == 0.0 else 0.0
        return 1.0 - residual / total


def fit_train_communication_subspace(
    source_train: ArrayLike,
    target_train: ArrayLike,
    *,
    rank: int,
    ridge: float = 0.0,
    normalize: bool = False,
    sample_ids: ArrayLike | None = None,
) -> FittedCommunicationSubspace:
    """Fit reduced-rank regression using training rows only.

    The full ridge map is first fit on centered (optionally z-scored) training
    rows.  Its predicted-target matrix determines the target reduced-rank
    projector, after which an SVD of the reduced coefficient map yields paired
    source and target communication bases.
    """

    source = _finite_matrix(source_train, name="source_train")
    target = _finite_matrix(target_train, name="target_train")
    if source.shape[0] != target.shape[0]:
        raise ValueError("source_train and target_train must share sample rows")
    if source.shape[0] < 2:
        raise ValueError("at least two training rows are required")
    rank = _positive_integer(rank, name="rank")
    maximum_rank = min(
        source.shape[0] - 1,
        source.shape[1],
        target.shape[1],
    )
    if rank > maximum_rank:
        raise ValueError(f"rank must not exceed {maximum_rank}")
    ridge = _nonnegative_scalar(ridge, name="ridge")
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be boolean")
    identifiers = _validated_sample_ids(sample_ids, n_samples=source.shape[0])

    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    if normalize:
        source_std = np.std(centered_source, axis=0, ddof=0)
        target_std = np.std(centered_target, axis=0, ddof=0)
        source_scale = np.where(source_std > 0.0, source_std, 1.0)
        target_scale = np.where(target_std > 0.0, target_std, 1.0)
    else:
        source_scale = np.ones(source.shape[1], dtype=np.float64)
        target_scale = np.ones(target.shape[1], dtype=np.float64)
    standardized_source = centered_source / source_scale
    standardized_target = centered_target / target_scale

    if ridge == 0.0:
        full_coefficients, _, _, _ = np.linalg.lstsq(
            standardized_source,
            standardized_target,
            rcond=None,
        )
    else:
        gram = standardized_source.T @ standardized_source
        full_coefficients = np.linalg.solve(
            gram + ridge * np.eye(gram.shape[0]),
            standardized_source.T @ standardized_target,
        )
    fitted_target = standardized_source @ full_coefficients
    _, predicted_singular, predicted_vt = np.linalg.svd(
        fitted_target,
        full_matrices=False,
    )
    tolerance = (
        (predicted_singular[0] if predicted_singular.size else 0.0)
        * np.finfo(np.float64).eps
        * max(fitted_target.shape)
    )
    predicted_rank = int(np.sum(predicted_singular > tolerance))
    if rank > predicted_rank:
        raise ValueError(
            "requested rank exceeds the numerical predictive rank on training data"
        )
    target_projector = predicted_vt[:rank].T @ predicted_vt[:rank]
    reduced_standardized = full_coefficients @ target_projector
    source_basis, singular_values, target_vt = np.linalg.svd(
        reduced_standardized,
        full_matrices=False,
    )
    coefficient_tolerance = (
        singular_values[0] * np.finfo(np.float64).eps * max(reduced_standardized.shape)
    )
    coefficient_rank = int(np.sum(singular_values > coefficient_tolerance))
    if coefficient_rank < rank:
        raise ValueError("reduced communication map is numerically rank-deficient")
    source_basis = source_basis[:, :rank]
    target_basis = target_vt[:rank].T
    singular_values = singular_values[:rank]

    # Convert the standardized reduced-rank map to original units for direct
    # held-out prediction.  Rank is preserved under invertible diagonal scales.
    coefficients = (
        reduced_standardized * target_scale[np.newaxis, :] / source_scale[:, np.newaxis]
    )
    intercept = target_mean - source_mean @ coefficients
    return FittedCommunicationSubspace(
        source_mean_=_readonly(source_mean),
        source_scale_=_readonly(source_scale),
        target_mean_=_readonly(target_mean),
        target_scale_=_readonly(target_scale),
        standardized_coefficients_=_readonly(reduced_standardized),
        coefficients_=_readonly(coefficients),
        intercept_=_readonly(intercept),
        source_basis_=_readonly(source_basis),
        target_basis_=_readonly(target_basis),
        singular_values_=_readonly(singular_values),
        rank_=rank,
        ridge_=ridge,
        normalized_=bool(normalize),
        n_train_samples_=int(source.shape[0]),
        fit_sample_ids_=identifiers,
    )


# Short alias for callers that already make the training split explicit.
fit_communication_subspace = fit_train_communication_subspace


@dataclass(frozen=True)
class CommunicationSubspaceComparison:
    """Angles and normalized projection overlap for one population side."""

    side: str
    principal_angles_degrees: FloatArray
    overlap: float
    rank_first: int
    rank_second: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "principal_angles_degrees",
            _readonly(self.principal_angles_degrees),
        )


def compare_communication_subspaces(
    first: FittedCommunicationSubspace,
    second: FittedCommunicationSubspace,
    *,
    side: PopulationSide = "source",
) -> CommunicationSubspaceComparison:
    """Compare train-fitted source or target communication directions."""

    if not isinstance(first, FittedCommunicationSubspace) or not isinstance(
        second, FittedCommunicationSubspace
    ):
        raise TypeError("first and second must be FittedCommunicationSubspace")
    if side not in {"source", "target"}:
        raise ValueError("side must be 'source' or 'target'")
    first_basis = first.source_basis_ if side == "source" else first.target_basis_
    second_basis = second.source_basis_ if side == "source" else second.target_basis_
    if first_basis.shape[0] != second_basis.shape[0]:
        raise ValueError(f"{side} population dimensions differ")
    return CommunicationSubspaceComparison(
        side=side,
        principal_angles_degrees=principal_angles(
            first_basis,
            second_basis,
            degrees=True,
        ),
        overlap=subspace_overlap(first_basis, second_basis),
        rank_first=first.rank_,
        rank_second=second.rank_,
    )


def communication_subspace_overlap(
    first: FittedCommunicationSubspace,
    second: FittedCommunicationSubspace,
    *,
    side: PopulationSide = "source",
) -> float:
    """Return normalized overlap of two train-fitted communication subspaces."""

    return compare_communication_subspaces(first, second, side=side).overlap


__all__ = [
    "CommunicationSubspaceComparison",
    "FittedCommunicationSubspace",
    "communication_subspace_overlap",
    "compare_communication_subspaces",
    "fit_communication_subspace",
    "fit_train_communication_subspace",
]
