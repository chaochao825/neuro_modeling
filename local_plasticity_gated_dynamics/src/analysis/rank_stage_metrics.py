"""Numerically explicit rank and plastic-tangent audits.

The central distinction in this module is between the rank of one realized
weight-change matrix and the dimension of the credit directions that can
produce such a matrix.  A sparse mask can make an outer-product update have
high matrix rank while the fixed-state credit map still has only a few input
channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


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


def _finite_vector(
    value: ArrayLike, *, name: str, length: int | None = None
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if length is not None and array.size != length:
        raise ValueError(f"{name} must have length {length}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _binary_mask(
    value: ArrayLike, *, name: str, shape: tuple[int, int]
) -> NDArray[np.bool_]:
    raw = np.asarray(value)
    if raw.shape != shape or not np.all(np.isin(raw, (False, True, 0, 1))):
        raise ValueError(f"{name} must be a binary matrix with shape {shape}")
    return raw.astype(bool)


def _tolerances(rtol: float, atol: float) -> tuple[float, float]:
    for value, name in ((rtol, "rtol"), (atol, "atol")):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise TypeError(f"{name} must be a non-negative finite scalar")
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be a non-negative finite scalar")
    return float(rtol), float(atol)


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _effective_rank_from_singular_values(values: FloatArray, threshold: float) -> float:
    retained = values[values > threshold]
    if retained.size == 0:
        return 0.0
    probabilities = retained / np.sum(retained)
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(np.exp(entropy))


@dataclass(frozen=True)
class MatrixRankSummary:
    """Numerical and entropy-effective rank under one recorded threshold."""

    numerical_rank: int
    effective_rank: float
    threshold: float
    singular_values: FloatArray


def matrix_rank_summary(
    matrix: ArrayLike, *, rtol: float = 1e-10, atol: float = 1e-12
) -> MatrixRankSummary:
    """Summarize rank without relying on an implicit library tolerance.

    A singular value is retained when it is strictly larger than
    ``max(atol, rtol * largest_singular_value)``.  Effective rank is the
    exponential Shannon entropy of the retained singular values.
    """

    array = _finite_matrix(matrix, name="matrix")
    relative, absolute = _tolerances(rtol, atol)
    singular = np.linalg.svd(array, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    threshold = max(absolute, relative * largest)
    return MatrixRankSummary(
        numerical_rank=int(np.count_nonzero(singular > threshold)),
        effective_rank=_effective_rank_from_singular_values(singular, threshold),
        threshold=float(threshold),
        singular_values=_readonly(singular),
    )


@dataclass(frozen=True)
class MaskedOuterProductIdentitySummary:
    """Audit of ``M * outer(u, v) == diag(u) M diag(v)``."""

    equal: bool
    max_abs_residual: float
    raw_outer: MatrixRankSummary
    mask: MatrixRankSummary
    masked_outer: MatrixRankSummary
    diagonal_form: MatrixRankSummary
    all_factors_nonzero: bool
    exact_rank_preservation_expected: bool
    numerically_preserves_mask_rank: bool | None
    left_diagonal_condition_number: float | None
    right_diagonal_condition_number: float | None


def masked_outer_product_identity(
    mask: ArrayLike,
    left_factor: ArrayLike,
    right_factor: ArrayLike,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> MaskedOuterProductIdentitySummary:
    """Numerically verify the masked outer-product identity and its ranks.

    When every factor entry is nonzero, both diagonal factors are invertible
    and the masked update must have the same rank as ``mask``.  With zero
    entries the equality still holds, but rank can fall to that of an active
    row/column submatrix.
    """

    relative, absolute = _tolerances(rtol, atol)
    raw_mask = np.asarray(mask)
    if raw_mask.ndim != 2 or 0 in raw_mask.shape:
        raise ValueError("mask must be a non-empty two-dimensional matrix")
    binary = _binary_mask(raw_mask, name="mask", shape=raw_mask.shape)
    left = _finite_vector(left_factor, name="left_factor", length=binary.shape[0])
    right = _finite_vector(right_factor, name="right_factor", length=binary.shape[1])
    numeric_mask = binary.astype(np.float64)
    raw_outer = np.outer(left, right)
    masked = numeric_mask * raw_outer
    # Broadcasting is exactly the matrix product diag(left) @ M @ diag(right)
    # without allocating either dense diagonal matrix.
    diagonal_form = (left[:, None] * numeric_mask) * right[None, :]
    residual = masked - diagonal_form
    equal = bool(np.allclose(masked, diagonal_form, rtol=relative, atol=absolute))

    raw_summary = matrix_rank_summary(raw_outer, rtol=relative, atol=absolute)
    mask_summary = matrix_rank_summary(numeric_mask, rtol=relative, atol=absolute)
    masked_summary = matrix_rank_summary(masked, rtol=relative, atol=absolute)
    diagonal_summary = matrix_rank_summary(diagonal_form, rtol=relative, atol=absolute)
    all_nonzero = bool(np.all(left != 0.0) and np.all(right != 0.0))

    def diagonal_condition(values: FloatArray) -> float | None:
        magnitudes = np.abs(values)
        if np.any(magnitudes == 0.0):
            return None
        return float(np.max(magnitudes) / np.min(magnitudes))

    return MaskedOuterProductIdentitySummary(
        equal=equal,
        max_abs_residual=float(np.max(np.abs(residual))),
        raw_outer=raw_summary,
        mask=mask_summary,
        masked_outer=masked_summary,
        diagonal_form=diagonal_summary,
        all_factors_nonzero=all_nonzero,
        exact_rank_preservation_expected=all_nonzero,
        numerically_preserves_mask_rank=(
            masked_summary.numerical_rank == mask_summary.numerical_rank
            if all_nonzero
            else None
        ),
        left_diagonal_condition_number=diagonal_condition(left),
        right_diagonal_condition_number=diagonal_condition(right),
    )


@dataclass(frozen=True)
class UpdateStageRankSummary:
    """Rank summaries for every auditable transformation of one update."""

    hebbian: MatrixRankSummary
    decay: MatrixRankSummary
    raw: MatrixRankSummary
    masked: MatrixRankSummary
    dale_applied: MatrixRankSummary
    normalization_correction: MatrixRankSummary
    total: MatrixRankSummary


def update_stage_rank_summary(
    *,
    hebbian_update: ArrayLike,
    decay_update: ArrayLike,
    raw_update: ArrayLike,
    masked_update: ArrayLike,
    dale_applied_update: ArrayLike,
    normalization_correction: ArrayLike,
    total_update: ArrayLike,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    validate_decomposition: bool = True,
) -> UpdateStageRankSummary:
    """Report numerical/effective rank while keeping decay and normalization separate."""

    if not isinstance(validate_decomposition, (bool, np.bool_)):
        raise TypeError("validate_decomposition must be boolean")
    relative, absolute = _tolerances(rtol, atol)
    named = {
        "hebbian": _finite_matrix(hebbian_update, name="hebbian_update"),
        "decay": _finite_matrix(decay_update, name="decay_update"),
        "raw": _finite_matrix(raw_update, name="raw_update"),
        "masked": _finite_matrix(masked_update, name="masked_update"),
        "dale_applied": _finite_matrix(dale_applied_update, name="dale_applied_update"),
        "normalization_correction": _finite_matrix(
            normalization_correction, name="normalization_correction"
        ),
        "total": _finite_matrix(total_update, name="total_update"),
    }
    shapes = {array.shape for array in named.values()}
    if len(shapes) != 1:
        raise ValueError("all update stages must have an identical matrix shape")
    if validate_decomposition:
        if not np.allclose(
            named["raw"],
            named["hebbian"] + named["decay"],
            rtol=relative,
            atol=absolute,
        ):
            raise ValueError("raw_update must equal hebbian_update + decay_update")
        if not np.allclose(
            named["total"],
            named["dale_applied"] + named["normalization_correction"],
            rtol=relative,
            atol=absolute,
        ):
            raise ValueError(
                "total_update must equal dale_applied_update + normalization_correction"
            )
    summaries = {
        name: matrix_rank_summary(array, rtol=relative, atol=absolute)
        for name, array in named.items()
    }
    return UpdateStageRankSummary(**summaries)


@dataclass(frozen=True)
class CreditTangentSummary:
    """Fixed-state image dimension of feedback channels in weight space."""

    stage: str
    feedback_dim: int
    numerical_dimension: int
    effective_dimension: float
    threshold: float
    singular_values: FloatArray
    gram: FloatArray
    n_active_synapses: int


def credit_tangent_summary(
    feedback_basis: ArrayLike,
    eligibility_trace: ArrayLike,
    *,
    post_derivative: ArrayLike | None = None,
    connectivity_mask: ArrayLike | None = None,
    synaptic_scale: ArrayLike | None = None,
    stage: str = "masked",
    rtol: float = 1e-8,
    atol: float = 1e-12,
    edge_chunk_size: int = 8192,
) -> CreditTangentSummary:
    """Compute the exact credit-tangent Gram matrix in bounded memory.

    For feedback basis column ``B[:, a]`` the corresponding fixed-state
    direction is

    ``scale * mask * outer(post_derivative * B[:, a], eligibility_trace)``.

    Only a ``feedback_dim x feedback_dim`` Gram matrix is retained.  Edges are
    processed in chunks, so memory does not scale as
    ``n_post * n_pre * feedback_dim``.  The default relative tolerance is
    deliberately ``1e-8`` because Gram formation squares the condition number.
    The reported threshold is additionally bounded below by
    ``sqrt(machine_epsilon * feedback_dim) * largest_singular_value`` so
    positive roundoff eigenvalues are not counted as tangent directions.
    """

    basis = _finite_matrix(feedback_basis, name="feedback_basis")
    n_post, feedback_dim = basis.shape
    eligibility = _finite_vector(eligibility_trace, name="eligibility_trace")
    n_pre = eligibility.size
    derivative = (
        np.ones(n_post, dtype=np.float64)
        if post_derivative is None
        else _finite_vector(post_derivative, name="post_derivative", length=n_post)
    )
    mask = (
        np.ones((n_post, n_pre), dtype=bool)
        if connectivity_mask is None
        else _binary_mask(
            connectivity_mask,
            name="connectivity_mask",
            shape=(n_post, n_pre),
        )
    )
    scale = (
        np.ones((n_post, n_pre), dtype=np.float64)
        if synaptic_scale is None
        else _finite_matrix(synaptic_scale, name="synaptic_scale")
    )
    if scale.shape != (n_post, n_pre):
        raise ValueError(f"synaptic_scale must have shape ({n_post}, {n_pre})")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a non-empty string")
    if (
        isinstance(edge_chunk_size, (bool, np.bool_))
        or not isinstance(edge_chunk_size, (int, np.integer))
        or int(edge_chunk_size) <= 0
    ):
        raise ValueError("edge_chunk_size must be a positive integer")
    relative, absolute = _tolerances(rtol, atol)

    rows, columns = np.nonzero(mask)
    gram = np.zeros((feedback_dim, feedback_dim), dtype=np.float64)
    chunk = int(edge_chunk_size)
    for start in range(0, rows.size, chunk):
        stop = min(start + chunk, rows.size)
        row = rows[start:stop]
        column = columns[start:stop]
        directions = (
            derivative[row, None]
            * basis[row]
            * eligibility[column, None]
            * scale[row, column, None]
        )
        gram += directions.T @ directions
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)[::-1]
    singular = np.sqrt(eigenvalues)
    largest = float(singular[0]) if singular.size else 0.0
    gram_noise_floor = np.sqrt(np.finfo(np.float64).eps * max(1, feedback_dim))
    threshold = max(absolute, relative * largest, gram_noise_floor * largest)
    return CreditTangentSummary(
        stage=stage,
        feedback_dim=int(feedback_dim),
        numerical_dimension=int(np.count_nonzero(singular > threshold)),
        effective_dimension=_effective_rank_from_singular_values(singular, threshold),
        threshold=float(threshold),
        singular_values=_readonly(singular),
        gram=_readonly(gram),
        n_active_synapses=int(rows.size),
    )


__all__ = [
    "CreditTangentSummary",
    "MaskedOuterProductIdentitySummary",
    "MatrixRankSummary",
    "UpdateStageRankSummary",
    "credit_tangent_summary",
    "masked_outer_product_identity",
    "matrix_rank_summary",
    "update_stage_rank_summary",
]
