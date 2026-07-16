"""Leakage-safe covariance geometry for controlled neural trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.analysis.manifold_metrics import principal_angles, subspace_overlap


FloatArray = NDArray[np.float64]


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


def covariance_matrix(
    samples: ArrayLike,
    *,
    ddof: int = 1,
) -> FloatArray:
    """Return the symmetric feature covariance of sample rows."""

    values = _finite_matrix(samples, name="samples")
    if isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, (int, np.integer)):
        raise TypeError("ddof must be an integer")
    ddof = int(ddof)
    if ddof < 0 or values.shape[0] <= ddof:
        raise ValueError("ddof must satisfy 0 <= ddof < n_samples")
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance = centered.T @ centered / (values.shape[0] - ddof)
    return _readonly(0.5 * (covariance + covariance.T))


def _psd_eigendecomposition(
    covariance: ArrayLike,
    *,
    name: str,
) -> tuple[FloatArray, FloatArray]:
    matrix = _finite_matrix(covariance, name=name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    symmetry_scale = max(1.0, float(np.max(np.abs(matrix))))
    if not np.allclose(
        matrix,
        matrix.T,
        rtol=1e-10,
        atol=1e-12 * symmetry_scale,
    ):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = np.finfo(np.float64).eps * matrix.shape[0] * scale * 100.0
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def covariance_eigenspectrum(
    covariance: ArrayLike,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Return descending PSD eigenvalues, optionally normalized to sum one."""

    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be boolean")
    eigenvalues, _ = _psd_eigendecomposition(covariance, name="covariance")
    if normalize:
        total = float(np.sum(eigenvalues))
        eigenvalues = eigenvalues / total if total > 0.0 else np.zeros_like(eigenvalues)
    return _readonly(eigenvalues)


def covariance_participation_ratio(covariance: ArrayLike) -> float:
    """Return ``tr(Sigma)^2 / tr(Sigma^2)``."""

    eigenvalues = covariance_eigenspectrum(covariance)
    denominator = float(np.sum(eigenvalues * eigenvalues))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(eigenvalues) ** 2 / denominator)


def covariance_principal_angles(
    covariance_a: ArrayLike,
    covariance_b: ArrayLike,
    *,
    n_components: int,
    degrees: bool = True,
) -> FloatArray:
    """Compare the leading covariance eigenspaces."""

    n_components = _positive_integer(n_components, name="n_components")
    first_values, first_vectors = _psd_eigendecomposition(
        covariance_a, name="covariance_a"
    )
    second_values, second_vectors = _psd_eigendecomposition(
        covariance_b, name="covariance_b"
    )
    if first_vectors.shape != second_vectors.shape:
        raise ValueError("covariances must have the same feature dimension")
    if n_components > first_vectors.shape[0]:
        raise ValueError("n_components exceeds the feature dimension")
    if first_values[n_components - 1] <= 0.0:
        raise ValueError("covariance_a has insufficient positive eigendirections")
    if second_values[n_components - 1] <= 0.0:
        raise ValueError("covariance_b has insufficient positive eigendirections")
    return _readonly(
        principal_angles(
            first_vectors[:, :n_components],
            second_vectors[:, :n_components],
            degrees=degrees,
        )
    )


def bures_covariance_distance(
    covariance_a: ArrayLike,
    covariance_b: ArrayLike,
    *,
    squared: bool = False,
) -> float:
    """Return the Bures-Wasserstein distance between PSD covariances."""

    if not isinstance(squared, (bool, np.bool_)):
        raise TypeError("squared must be boolean")
    first_values, first_vectors = _psd_eigendecomposition(
        covariance_a, name="covariance_a"
    )
    second_values, second_vectors = _psd_eigendecomposition(
        covariance_b, name="covariance_b"
    )
    if first_vectors.shape != second_vectors.shape:
        raise ValueError("covariances must have the same feature dimension")
    first = (first_vectors * first_values[np.newaxis, :]) @ first_vectors.T
    second = (second_vectors * second_values[np.newaxis, :]) @ second_vectors.T
    # tr[(A^(1/2) B A^(1/2))^(1/2)] is the nuclear norm of
    # A^(1/2) B^(1/2).  Computing that norm from the eigen factors avoids a
    # second eigendecomposition of a very ill-conditioned product.  The latter
    # suffered catastrophic cancellation for the nearly low-rank covariance
    # matrices produced by the formal Exp24 frozen condition.
    cross_factor = (
        np.sqrt(first_values)[:, np.newaxis]
        * (first_vectors.T @ second_vectors)
        * np.sqrt(second_values)[np.newaxis, :]
    )
    fidelity = float(np.sum(np.linalg.svd(cross_factor, compute_uv=False)))
    squared_distance = (
        float(np.trace(first))
        + float(np.trace(second))
        - 2.0 * fidelity
    )
    scale = max(1.0, float(np.trace(first) + np.trace(second)))
    tolerance = np.finfo(np.float64).eps * first.shape[0] * scale * 100.0
    if squared_distance < -tolerance:
        raise RuntimeError("Bures distance became materially negative")
    squared_distance = (
        0.0
        if abs(squared_distance) <= tolerance
        else max(0.0, squared_distance)
    )
    return squared_distance if squared else float(np.sqrt(squared_distance))


def _validated_sample_ids(
    sample_ids: ArrayLike | None,
    *,
    n_samples: int,
) -> tuple[object, ...] | None:
    if sample_ids is None:
        return None
    identifiers = np.asarray(sample_ids, dtype=object)
    if identifiers.ndim != 1 or identifiers.size != n_samples:
        raise ValueError("sample_ids must be one-dimensional and match samples")
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
class FittedCovarianceGeometry:
    """Covariance and eigenspace fit exclusively on supplied training rows."""

    mean_: FloatArray
    covariance_: FloatArray
    eigenvalues_: FloatArray
    eigenvectors_: FloatArray
    n_components_: int
    participation_ratio_: float
    n_train_samples_: int
    ddof_: int
    fit_sample_ids_: tuple[object, ...] | None

    def __post_init__(self) -> None:
        for name in ("mean_", "covariance_", "eigenvalues_", "eigenvectors_"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def basis_(self) -> FloatArray:
        return self.eigenvectors_[:, : self.n_components_].copy()

    def center(self, samples: ArrayLike) -> FloatArray:
        values = _finite_matrix(samples, name="samples")
        if values.shape[1] != self.mean_.size:
            raise ValueError("samples feature count differs from fitted geometry")
        return values - self.mean_

    def transform(self, samples: ArrayLike) -> FloatArray:
        """Project held-out rows without refitting mean or eigendirections."""

        return self.center(samples) @ self.basis_


def fit_covariance_geometry(
    samples_train: ArrayLike,
    *,
    n_components: int,
    ddof: int = 1,
    sample_ids: ArrayLike | None = None,
) -> FittedCovarianceGeometry:
    """Fit covariance geometry on training samples only."""

    samples = _finite_matrix(samples_train, name="samples_train")
    n_components = _positive_integer(n_components, name="n_components")
    if n_components > samples.shape[1]:
        raise ValueError("n_components exceeds the feature dimension")
    covariance = covariance_matrix(samples, ddof=ddof)
    eigenvalues, eigenvectors = _psd_eigendecomposition(
        covariance, name="training covariance"
    )
    if eigenvalues[n_components - 1] <= 0.0:
        raise ValueError("training covariance has insufficient positive rank")
    identifiers = _validated_sample_ids(sample_ids, n_samples=samples.shape[0])
    return FittedCovarianceGeometry(
        mean_=_readonly(np.mean(samples, axis=0)),
        covariance_=_readonly(covariance),
        eigenvalues_=_readonly(eigenvalues),
        eigenvectors_=_readonly(eigenvectors),
        n_components_=n_components,
        participation_ratio_=covariance_participation_ratio(covariance),
        n_train_samples_=int(samples.shape[0]),
        ddof_=int(ddof),
        fit_sample_ids_=identifiers,
    )


@dataclass(frozen=True)
class CovarianceGeometryComparison:
    """Pairwise geometry summary with no neuron-level inferential claims."""

    n_components: int
    principal_angles_degrees: FloatArray
    subspace_overlap: float
    bures_distance: float
    bures_distance_squared: float
    participation_ratio_a: float
    participation_ratio_b: float
    normalized_eigenspectrum_a: FloatArray
    normalized_eigenspectrum_b: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "principal_angles_degrees",
            _readonly(self.principal_angles_degrees),
        )
        object.__setattr__(
            self,
            "normalized_eigenspectrum_a",
            _readonly(self.normalized_eigenspectrum_a),
        )
        object.__setattr__(
            self,
            "normalized_eigenspectrum_b",
            _readonly(self.normalized_eigenspectrum_b),
        )


def compare_covariance_geometries(
    first: FittedCovarianceGeometry,
    second: FittedCovarianceGeometry,
    *,
    n_components: int | None = None,
) -> CovarianceGeometryComparison:
    """Compare two independently train-fitted covariance geometries."""

    if not isinstance(first, FittedCovarianceGeometry) or not isinstance(
        second, FittedCovarianceGeometry
    ):
        raise TypeError("first and second must be FittedCovarianceGeometry")
    if first.mean_.size != second.mean_.size:
        raise ValueError("fitted geometries have different feature dimensions")
    dimension = (
        min(first.n_components_, second.n_components_)
        if n_components is None
        else _positive_integer(n_components, name="n_components")
    )
    if dimension > min(first.n_components_, second.n_components_):
        raise ValueError("n_components exceeds one fitted geometry")
    first_basis = first.eigenvectors_[:, :dimension]
    second_basis = second.eigenvectors_[:, :dimension]
    squared_distance = bures_covariance_distance(
        first.covariance_, second.covariance_, squared=True
    )
    return CovarianceGeometryComparison(
        n_components=dimension,
        principal_angles_degrees=principal_angles(
            first_basis, second_basis, degrees=True
        ),
        subspace_overlap=subspace_overlap(first_basis, second_basis),
        bures_distance=float(np.sqrt(squared_distance)),
        bures_distance_squared=squared_distance,
        participation_ratio_a=first.participation_ratio_,
        participation_ratio_b=second.participation_ratio_,
        normalized_eigenspectrum_a=covariance_eigenspectrum(
            first.covariance_, normalize=True
        ),
        normalized_eigenspectrum_b=covariance_eigenspectrum(
            second.covariance_, normalize=True
        ),
    )


@dataclass(frozen=True)
class FittedConditionalCovarianceGeometry:
    """Training-only linear residualizer plus residual covariance geometry."""

    coefficients_: FloatArray
    n_covariates_: int
    geometry_: FittedCovarianceGeometry

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients_", _readonly(self.coefficients_))

    def residualize(
        self,
        samples: ArrayLike,
        covariates: ArrayLike,
    ) -> FloatArray:
        activity = _finite_matrix(samples, name="samples")
        design_values = _finite_matrix(covariates, name="covariates")
        if activity.shape[0] != design_values.shape[0]:
            raise ValueError("samples and covariates must share rows")
        if design_values.shape[1] != self.n_covariates_:
            raise ValueError("covariate count differs from the training fit")
        if activity.shape[1] != self.coefficients_.shape[1]:
            raise ValueError("sample feature count differs from the training fit")
        design = np.column_stack(
            [np.ones(activity.shape[0], dtype=np.float64), design_values]
        )
        return activity - design @ self.coefficients_


def fit_conditional_covariance_geometry(
    samples_train: ArrayLike,
    covariates_train: ArrayLike,
    *,
    n_components: int,
    ddof: int = 1,
    sample_ids: ArrayLike | None = None,
) -> FittedConditionalCovarianceGeometry:
    """Fit OLS nuisance removal and residual covariance on training rows only."""

    samples = _finite_matrix(samples_train, name="samples_train")
    covariates = _finite_matrix(covariates_train, name="covariates_train")
    if samples.shape[0] != covariates.shape[0]:
        raise ValueError("samples_train and covariates_train must share rows")
    design = np.column_stack([np.ones(samples.shape[0], dtype=np.float64), covariates])
    coefficients, _, rank, _ = np.linalg.lstsq(design, samples, rcond=None)
    if int(rank) != design.shape[1]:
        raise ValueError("training covariate design must have full column rank")
    residuals = samples - design @ coefficients
    geometry = fit_covariance_geometry(
        residuals,
        n_components=n_components,
        ddof=ddof,
        sample_ids=sample_ids,
    )
    return FittedConditionalCovarianceGeometry(
        coefficients_=_readonly(coefficients),
        n_covariates_=int(covariates.shape[1]),
        geometry_=geometry,
    )


__all__ = [
    "CovarianceGeometryComparison",
    "FittedConditionalCovarianceGeometry",
    "FittedCovarianceGeometry",
    "bures_covariance_distance",
    "compare_covariance_geometries",
    "covariance_eigenspectrum",
    "covariance_matrix",
    "covariance_participation_ratio",
    "covariance_principal_angles",
    "fit_conditional_covariance_geometry",
    "fit_covariance_geometry",
]
