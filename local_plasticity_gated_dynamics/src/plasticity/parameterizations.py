"""Auditable direct, multiplicative, and full per-synapse plasticity controls.

All three implementations are explicit NumPy updates.  They do not retain a
trajectory, construct a loss, invoke a gradient engine, or use BPTT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ParameterizationName = Literal[
    "direct_additive",
    "sign_preserving_multiplicative",
    "full_per_synapse",
]


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


def _learning_rate(value: float) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("learning_rate must be a non-negative finite scalar")
    return float(value)


def _positive_integer(value: int, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validated_weight_state(
    current_weights: ArrayLike,
    connectivity_mask: ArrayLike,
    presynaptic_signs: ArrayLike | None,
) -> tuple[FloatArray, NDArray[np.bool_], FloatArray | None]:
    current = _finite_matrix(current_weights, name="current_weights")
    raw_mask = np.asarray(connectivity_mask)
    if raw_mask.shape != current.shape or not np.all(
        np.isin(raw_mask, (False, True, 0, 1))
    ):
        raise ValueError("connectivity_mask must be a matching binary matrix")
    mask = raw_mask.astype(bool)
    if np.any(np.abs(current[~mask]) > 1e-12):
        raise ValueError("current_weights must respect connectivity_mask")
    if presynaptic_signs is None:
        signs = None
    else:
        signs = _finite_vector(
            presynaptic_signs,
            name="presynaptic_signs",
            length=current.shape[1],
        )
        if not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError("presynaptic_signs must contain only -1 and +1")
        excitatory = signs > 0.0
        if np.any(current[:, excitatory] < -1e-12) or np.any(
            current[:, ~excitatory] > 1e-12
        ):
            raise ValueError("current_weights must already satisfy Dale signs")
    return current, mask, signs


def _dale_applied_update(
    current: FloatArray,
    proposed_update: FloatArray,
    mask: NDArray[np.bool_],
    signs: FloatArray | None,
) -> FloatArray:
    candidate = np.where(mask, current + proposed_update, 0.0)
    if signs is not None:
        excitatory = signs > 0.0
        candidate[:, excitatory] = np.maximum(candidate[:, excitatory], 0.0)
        candidate[:, ~excitatory] = np.minimum(candidate[:, ~excitatory], 0.0)
    return candidate - current


@dataclass(frozen=True)
class ParameterizationCosts:
    """L1/L2 path lengths in control and physical weight coordinates."""

    raw_control_l1: float
    raw_control_l2: float
    masked_control_l1: float
    masked_control_l2: float
    applied_control_l1: float
    applied_control_l2: float
    raw_weight_l1: float
    raw_weight_l2: float
    masked_weight_l1: float
    masked_weight_l2: float
    applied_weight_l1: float
    applied_weight_l2: float


@dataclass(frozen=True)
class ParameterizedPlasticityUpdate:
    """Common audit record for three distinct plastic parameterizations."""

    parameterization: ParameterizationName
    control_space: Literal["weight", "log_magnitude"]
    credit_dimension: int
    raw_control_update: FloatArray
    masked_control_update: FloatArray
    applied_control_update: FloatArray
    raw_weight_update: FloatArray
    masked_weight_update: FloatArray
    dale_applied_update: FloatArray
    control_scale: float
    control_bound_active: bool
    pre_scale_exceedance_fraction: float
    costs: ParameterizationCosts


def _build_result(
    *,
    parameterization: ParameterizationName,
    control_space: Literal["weight", "log_magnitude"],
    credit_dimension: int,
    raw_control: FloatArray,
    masked_control: FloatArray,
    applied_control: FloatArray,
    raw_weight: FloatArray,
    masked_weight: FloatArray,
    dale_applied: FloatArray,
    control_scale: float = 1.0,
    control_bound_active: bool = False,
    pre_scale_exceedance_fraction: float = 0.0,
) -> ParameterizedPlasticityUpdate:
    arrays = (
        raw_control,
        masked_control,
        applied_control,
        raw_weight,
        masked_weight,
        dale_applied,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError(
            "plasticity parameterization produced non-finite values"
        )
    costs = ParameterizationCosts(
        raw_control_l1=float(np.sum(np.abs(raw_control))),
        raw_control_l2=float(np.linalg.norm(raw_control)),
        masked_control_l1=float(np.sum(np.abs(masked_control))),
        masked_control_l2=float(np.linalg.norm(masked_control)),
        applied_control_l1=float(np.sum(np.abs(applied_control))),
        applied_control_l2=float(np.linalg.norm(applied_control)),
        raw_weight_l1=float(np.sum(np.abs(raw_weight))),
        raw_weight_l2=float(np.linalg.norm(raw_weight)),
        masked_weight_l1=float(np.sum(np.abs(masked_weight))),
        masked_weight_l2=float(np.linalg.norm(masked_weight)),
        applied_weight_l1=float(np.sum(np.abs(dale_applied))),
        applied_weight_l2=float(np.linalg.norm(dale_applied)),
    )
    return ParameterizedPlasticityUpdate(
        parameterization=parameterization,
        control_space=control_space,
        credit_dimension=int(credit_dimension),
        raw_control_update=_readonly(raw_control),
        masked_control_update=_readonly(masked_control),
        applied_control_update=_readonly(applied_control),
        raw_weight_update=_readonly(raw_weight),
        masked_weight_update=_readonly(masked_weight),
        dale_applied_update=_readonly(dale_applied),
        control_scale=float(control_scale),
        control_bound_active=bool(control_bound_active),
        pre_scale_exceedance_fraction=float(pre_scale_exceedance_fraction),
        costs=costs,
    )


class DirectAdditivePlasticity:
    """Current local outer-product update in additive weight coordinates."""

    def __init__(self, *, learning_rate: float, credit_dimension: int) -> None:
        self.learning_rate = _learning_rate(learning_rate)
        self.credit_dimension = _positive_integer(
            credit_dimension, name="credit_dimension"
        )

    def propose(
        self,
        post_factor: ArrayLike,
        eligibility_trace: ArrayLike,
        *,
        current_weights: ArrayLike,
        connectivity_mask: ArrayLike,
        presynaptic_signs: ArrayLike | None = None,
    ) -> ParameterizedPlasticityUpdate:
        current, mask, signs = _validated_weight_state(
            current_weights, connectivity_mask, presynaptic_signs
        )
        if self.credit_dimension > current.shape[0]:
            raise ValueError("credit_dimension cannot exceed the postsynaptic size")
        post = _finite_vector(post_factor, name="post_factor", length=current.shape[0])
        eligibility = _finite_vector(
            eligibility_trace,
            name="eligibility_trace",
            length=current.shape[1],
        )
        raw = self.learning_rate * np.outer(post, eligibility)
        masked = np.where(mask, raw, 0.0)
        applied = _dale_applied_update(current, masked, mask, signs)
        return _build_result(
            parameterization="direct_additive",
            control_space="weight",
            credit_dimension=self.credit_dimension,
            raw_control=raw,
            masked_control=masked,
            applied_control=masked,
            raw_weight=raw,
            masked_weight=masked,
            dale_applied=applied,
        )


class SignPreservingMultiplicativePlasticity:
    """Local factorized update in log-magnitude coordinates.

    The physical update is ``W * exp(delta_log) - W``.  Positive exponential
    factors preserve every nonzero Dale sign and the fixed sparse mask while
    permitting both strengthening and weakening of magnitudes.
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        credit_dimension: int,
        max_abs_log_step: float | None = 0.1,
    ) -> None:
        self.learning_rate = _learning_rate(learning_rate)
        self.credit_dimension = _positive_integer(
            credit_dimension, name="credit_dimension"
        )
        if max_abs_log_step is not None and (
            isinstance(max_abs_log_step, (bool, np.bool_))
            or not np.isscalar(max_abs_log_step)
            or not np.isfinite(float(max_abs_log_step))
            or float(max_abs_log_step) <= 0.0
        ):
            raise ValueError("max_abs_log_step must be positive and finite or None")
        self.max_abs_log_step = (
            None if max_abs_log_step is None else float(max_abs_log_step)
        )

    def propose(
        self,
        post_factor: ArrayLike,
        eligibility_trace: ArrayLike,
        *,
        current_weights: ArrayLike,
        connectivity_mask: ArrayLike,
        presynaptic_signs: ArrayLike | None = None,
    ) -> ParameterizedPlasticityUpdate:
        current, mask, signs = _validated_weight_state(
            current_weights, connectivity_mask, presynaptic_signs
        )
        if self.credit_dimension > current.shape[0]:
            raise ValueError("credit_dimension cannot exceed the postsynaptic size")
        post = _finite_vector(post_factor, name="post_factor", length=current.shape[0])
        eligibility = _finite_vector(
            eligibility_trace,
            name="eligibility_trace",
            length=current.shape[1],
        )
        raw_control = self.learning_rate * np.outer(post, eligibility)
        masked_control = np.where(mask, raw_control, 0.0)
        control_scale = 1.0
        boundary_fraction = 0.0
        if self.max_abs_log_step is not None:
            limit = self.max_abs_log_step
            active_values = np.abs(masked_control[mask])
            maximum = float(np.max(active_values)) if active_values.size else 0.0
            if maximum > limit:
                control_scale = limit / maximum
                boundary_fraction = float(np.mean(active_values > limit))
        # A single global scale preserves the masked control direction.  In
        # contrast, elementwise clipping would introduce an extra nonlinear
        # rank-changing stage that is absent from the factorized equation.
        applied_control = masked_control * control_scale
        with np.errstate(over="ignore", invalid="ignore"):
            raw_weight = current * np.expm1(raw_control)
            masked_weight = current * np.expm1(masked_control)
            applied_weight = current * np.expm1(applied_control)
        applied = _dale_applied_update(current, applied_weight, mask, signs)
        return _build_result(
            parameterization="sign_preserving_multiplicative",
            control_space="log_magnitude",
            credit_dimension=self.credit_dimension,
            raw_control=raw_control,
            masked_control=masked_control,
            applied_control=applied_control,
            raw_weight=raw_weight,
            masked_weight=masked_weight,
            dale_applied=applied,
            control_scale=control_scale,
            control_bound_active=control_scale < 1.0,
            pre_scale_exceedance_fraction=boundary_fraction,
        )


class FullPerSynapsePlasticity:
    """High-dimensional control with an independent third factor per edge."""

    def __init__(self, *, learning_rate: float) -> None:
        self.learning_rate = _learning_rate(learning_rate)

    def propose(
        self,
        synaptic_third_factor: ArrayLike,
        eligibility_trace: ArrayLike,
        *,
        current_weights: ArrayLike,
        connectivity_mask: ArrayLike,
        presynaptic_signs: ArrayLike | None = None,
    ) -> ParameterizedPlasticityUpdate:
        current, mask, signs = _validated_weight_state(
            current_weights, connectivity_mask, presynaptic_signs
        )
        third_factor = _finite_matrix(
            synaptic_third_factor, name="synaptic_third_factor"
        )
        if third_factor.shape != current.shape:
            raise ValueError("synaptic_third_factor must match current_weights")
        eligibility = _finite_vector(
            eligibility_trace,
            name="eligibility_trace",
            length=current.shape[1],
        )
        raw = self.learning_rate * third_factor * eligibility[None, :]
        masked = np.where(mask, raw, 0.0)
        applied = _dale_applied_update(current, masked, mask, signs)
        return _build_result(
            parameterization="full_per_synapse",
            control_space="weight",
            credit_dimension=int(np.count_nonzero(mask)),
            raw_control=raw,
            masked_control=masked,
            applied_control=masked,
            raw_weight=raw,
            masked_weight=masked,
            dale_applied=applied,
        )


__all__ = [
    "DirectAdditivePlasticity",
    "FullPerSynapsePlasticity",
    "ParameterizationCosts",
    "ParameterizedPlasticityUpdate",
    "SignPreservingMultiplicativePlasticity",
]
