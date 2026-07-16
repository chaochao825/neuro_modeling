"""Direction and realized-improvement audits for controller updates.

The reference direction is an explicit update vector, normally the negative
exact forward gradient under the same axis-only parameterization.  Zero-norm
comparisons are marked ineligible instead of being assigned an artificial
cosine.  Improvement is always evaluated by rerunning a caller-supplied
objective at the proposed controller parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
Objective = Callable[[FloatArray], float]


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _finite_vector(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric vector")
    vector = np.asarray(raw, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_scalar(value: object, *, name: str) -> float:
    result = _finite_scalar(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class DirectionalCosineAudit:
    """Directional agreement between a proposed and reference update."""

    cosine: float
    dot_product: float
    proposed_norm: float
    reference_norm: float
    eligible: bool
    reason: str


def directional_cosine(
    proposed_update: ArrayLike,
    reference_update: ArrayLike,
    *,
    norm_tolerance: float = 1e-12,
) -> DirectionalCosineAudit:
    """Return cosine agreement, marking either zero direction ineligible."""

    proposed = _finite_vector(proposed_update, name="proposed_update")
    reference = _finite_vector(reference_update, name="reference_update")
    if proposed.shape != reference.shape:
        raise ValueError("proposed_update and reference_update must have equal shape")
    tolerance = _nonnegative_scalar(norm_tolerance, name="norm_tolerance")
    proposed_norm = float(np.linalg.norm(proposed))
    reference_norm = float(np.linalg.norm(reference))
    dot = float(proposed @ reference)
    if proposed_norm <= tolerance:
        return DirectionalCosineAudit(
            cosine=float("nan"),
            dot_product=dot,
            proposed_norm=proposed_norm,
            reference_norm=reference_norm,
            eligible=False,
            reason="proposed_update_norm_below_tolerance",
        )
    if reference_norm <= tolerance:
        return DirectionalCosineAudit(
            cosine=float("nan"),
            dot_product=dot,
            proposed_norm=proposed_norm,
            reference_norm=reference_norm,
            eligible=False,
            reason="reference_update_norm_below_tolerance",
        )
    cosine = float(np.clip(dot / (proposed_norm * reference_norm), -1.0, 1.0))
    return DirectionalCosineAudit(
        cosine=cosine,
        dot_product=dot,
        proposed_norm=proposed_norm,
        reference_norm=reference_norm,
        eligible=True,
        reason="eligible",
    )


def _evaluate_objective(
    objective: Objective,
    parameters: FloatArray,
    *,
    name: str,
) -> float:
    if not callable(objective):
        raise TypeError("objective must be callable")
    protected = np.array(parameters, dtype=np.float64, copy=True)
    protected.setflags(write=False)
    result = _finite_scalar(objective(protected), name=name)
    return result


@dataclass(frozen=True, slots=True)
class UpdateImprovementAudit:
    """Realized objective change after applying one controller update."""

    baseline_loss: float
    candidate_loss: float
    absolute_improvement: float
    relative_improvement: float
    improved: bool
    update_scale: float
    candidate_parameters: FloatArray


def evaluate_update_improvement(
    parameters: ArrayLike,
    proposed_update: ArrayLike,
    objective: Objective,
    *,
    update_scale: float = 1.0,
    improvement_tolerance: float = 0.0,
) -> UpdateImprovementAudit:
    """Rerun an objective before and after a controller-vector update."""

    parameter = _finite_vector(parameters, name="parameters")
    update = _finite_vector(proposed_update, name="proposed_update")
    if update.shape != parameter.shape:
        raise ValueError("proposed_update must match parameters")
    scale = _finite_scalar(update_scale, name="update_scale")
    if scale < 0.0:
        raise ValueError("update_scale must be non-negative")
    tolerance = _nonnegative_scalar(
        improvement_tolerance,
        name="improvement_tolerance",
    )
    before_snapshot = parameter.copy()
    baseline = _evaluate_objective(objective, parameter, name="baseline_loss")
    if not np.array_equal(parameter, before_snapshot):
        raise RuntimeError("objective mutated the baseline parameter vector")
    candidate = parameter + scale * update
    candidate_snapshot = candidate.copy()
    after = _evaluate_objective(objective, candidate, name="candidate_loss")
    if not np.array_equal(candidate, candidate_snapshot):
        raise RuntimeError("objective mutated the candidate parameter vector")
    improvement = baseline - after
    denominator = max(abs(baseline), np.finfo(np.float64).eps)
    return UpdateImprovementAudit(
        baseline_loss=baseline,
        candidate_loss=after,
        absolute_improvement=float(improvement),
        relative_improvement=float(improvement / denominator),
        improved=bool(improvement > tolerance),
        update_scale=scale,
        candidate_parameters=_readonly(candidate),
    )


@dataclass(frozen=True, slots=True)
class ImprovementRateAudit:
    """Paired improvement probability across seed/session-level units."""

    probability_improved: float
    n_improved: int
    n_units: int
    mean_improvement: float
    median_improvement: float
    improvements: FloatArray


def summarize_improvement_trials(
    baseline_losses: ArrayLike,
    candidate_losses: ArrayLike,
    *,
    improvement_tolerance: float = 0.0,
) -> ImprovementRateAudit:
    """Summarize paired ``P[candidate loss < baseline loss]``."""

    baseline = _finite_vector(baseline_losses, name="baseline_losses")
    candidate = _finite_vector(candidate_losses, name="candidate_losses")
    if candidate.shape != baseline.shape:
        raise ValueError("baseline_losses and candidate_losses must align")
    tolerance = _nonnegative_scalar(
        improvement_tolerance,
        name="improvement_tolerance",
    )
    improvement = baseline - candidate
    improved = improvement > tolerance
    return ImprovementRateAudit(
        probability_improved=float(np.mean(improved)),
        n_improved=int(np.count_nonzero(improved)),
        n_units=int(improvement.size),
        mean_improvement=float(np.mean(improvement)),
        median_improvement=float(np.median(improvement)),
        improvements=_readonly(improvement),
    )


@dataclass(frozen=True, slots=True)
class ControllerGradientAudit:
    """Joint direction and closed-loop objective receipt for one update."""

    direction: DirectionalCosineAudit
    improvement: UpdateImprovementAudit
    local_update: FloatArray
    reference_update: FloatArray


def audit_controller_update(
    parameters: ArrayLike,
    local_update: ArrayLike,
    reference_update: ArrayLike,
    objective: Objective,
    *,
    update_scale: float = 1.0,
    norm_tolerance: float = 1e-12,
    improvement_tolerance: float = 0.0,
) -> ControllerGradientAudit:
    """Audit local direction against an oracle update and rerun the objective."""

    parameter = _finite_vector(parameters, name="parameters")
    local = _finite_vector(local_update, name="local_update")
    reference = _finite_vector(reference_update, name="reference_update")
    if local.shape != parameter.shape or reference.shape != parameter.shape:
        raise ValueError("all controller vectors must have equal shape")
    direction = directional_cosine(
        local,
        reference,
        norm_tolerance=norm_tolerance,
    )
    improvement = evaluate_update_improvement(
        parameter,
        local,
        objective,
        update_scale=update_scale,
        improvement_tolerance=improvement_tolerance,
    )
    return ControllerGradientAudit(
        direction=direction,
        improvement=improvement,
        local_update=_readonly(local),
        reference_update=_readonly(reference),
    )


__all__ = [
    "ControllerGradientAudit",
    "DirectionalCosineAudit",
    "ImprovementRateAudit",
    "UpdateImprovementAudit",
    "audit_controller_update",
    "directional_cosine",
    "evaluate_update_improvement",
    "summarize_improvement_trials",
]
