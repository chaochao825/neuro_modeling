"""Match and audit actuator conditions by caused functional changes.

Parameter norms are not comparable across routing, gain, and low-rank
actuators.  This module instead scales a proposed control trajectory until its
closed-loop rollout matches

``mean_t ||x_controlled(t) - x_frozen(t)||_2^2``.

The matcher receives a deterministic rollout callback and never fits or
normalizes on held-out data.  A separate joint audit prevents a state-only
match from being described as a fully matched functional budget: firing-rate
change, gain envelope, and synaptic-event change must also remain inside
pre-registered training-only envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
RolloutAtScale = Callable[[float], ArrayLike | object]


def _finite_states(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    states = np.asarray(raw, dtype=np.float64)
    if states.ndim < 2 or states.size == 0 or 0 in states.shape:
        raise ValueError(f"{name} must have shape [..., time, state]")
    if not np.all(np.isfinite(states)):
        raise ValueError(f"{name} must contain only finite values")
    return states


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


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def functional_state_displacement(
    controlled_states: ArrayLike,
    frozen_states: ArrayLike,
    *,
    sample_mask: ArrayLike | None = None,
    exclude_initial: bool = True,
) -> float:
    """Return mean squared Euclidean state displacement.

    Arrays may have shape ``[time, state]`` or
    ``[episode, time, state]``.  All leading observations are averaged after
    summing over the final state axis.  ``sample_mask`` addresses those leading
    observations after the optional initial-time removal.
    """

    controlled = _finite_states(controlled_states, name="controlled_states")
    frozen = _finite_states(frozen_states, name="frozen_states")
    if controlled.shape != frozen.shape:
        raise ValueError("controlled_states and frozen_states must match exactly")
    if not isinstance(exclude_initial, (bool, np.bool_)):
        raise TypeError("exclude_initial must be boolean")
    if exclude_initial:
        if controlled.shape[-2] < 2:
            raise ValueError("cannot exclude the only time point")
        controlled = controlled[..., 1:, :]
        frozen = frozen[..., 1:, :]
    squared = np.sum((controlled - frozen) ** 2, axis=-1)
    if sample_mask is not None:
        mask = np.asarray(sample_mask)
        if mask.dtype.kind != "b":
            raise TypeError("sample_mask must be boolean")
        if mask.shape != squared.shape:
            raise ValueError(
                "sample_mask must match trajectory leading/time dimensions"
            )
        if not np.any(mask):
            raise ValueError("sample_mask must select at least one observation")
        squared = squared[mask]
    return float(np.mean(squared))


def _states_from_rollout(value: ArrayLike | object) -> FloatArray:
    candidate = getattr(value, "states", value)
    return _finite_states(candidate, name="rollout states")


class FunctionalBudgetError(RuntimeError):
    """Raised when a requested functional displacement cannot be matched."""


@dataclass(frozen=True)
class FunctionalBudgetMatch:
    """Immutable receipt for one functional-budget search."""

    target_displacement: float
    achieved_displacement: float
    scale: float
    absolute_error: float
    relative_error: float
    converged: bool
    n_evaluations: int
    bracket_low: float
    bracket_high: float
    max_scale: float
    exclude_initial: bool
    controlled_states: FloatArray
    evaluated_scales: FloatArray
    evaluated_displacements: FloatArray


def match_functional_state_displacement(
    rollout_at_scale: RolloutAtScale,
    frozen_states: ArrayLike,
    *,
    target_displacement: float,
    initial_scale: float = 1.0,
    max_scale: float = 64.0,
    expansion_factor: float = 2.0,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-10,
    zero_scale_tolerance: float = 1e-12,
    max_iterations: int = 80,
    exclude_initial: bool = True,
    raise_on_unreachable: bool = True,
) -> FunctionalBudgetMatch:
    """Find a scalar control multiplier that matches a state-space budget.

    The algorithm first expands a bracket from zero and then bisects a
    displacement crossing.  It retains every evaluated point and returns the
    closest one.  Scale zero must reproduce the supplied frozen trajectory;
    this audit prevents a condition-specific initialization or noise tape from
    being hidden inside the callback.
    """

    if not callable(rollout_at_scale):
        raise TypeError("rollout_at_scale must be callable")
    frozen = _finite_states(frozen_states, name="frozen_states")
    target = _nonnegative_scalar(target_displacement, name="target_displacement")
    initial = _nonnegative_scalar(initial_scale, name="initial_scale")
    maximum = _nonnegative_scalar(max_scale, name="max_scale")
    expansion = _nonnegative_scalar(expansion_factor, name="expansion_factor")
    relative_tolerance = _nonnegative_scalar(
        relative_tolerance, name="relative_tolerance"
    )
    absolute_tolerance = _nonnegative_scalar(
        absolute_tolerance, name="absolute_tolerance"
    )
    zero_scale_tolerance = _nonnegative_scalar(
        zero_scale_tolerance, name="zero_scale_tolerance"
    )
    max_iterations = _positive_integer(max_iterations, name="max_iterations")
    if initial <= 0.0:
        raise ValueError("initial_scale must be positive")
    if maximum < initial:
        raise ValueError("max_scale must be at least initial_scale")
    if expansion <= 1.0:
        raise ValueError("expansion_factor must exceed one")
    if not isinstance(exclude_initial, (bool, np.bool_)):
        raise TypeError("exclude_initial must be boolean")
    if not isinstance(raise_on_unreachable, (bool, np.bool_)):
        raise TypeError("raise_on_unreachable must be boolean")

    evaluated_scales: list[float] = []
    evaluated_displacements: list[float] = []
    evaluated_states: list[FloatArray] = []

    def evaluate(scale: float) -> tuple[float, FloatArray]:
        states = _states_from_rollout(rollout_at_scale(float(scale)))
        if states.shape != frozen.shape:
            raise ValueError(
                "every rollout_at_scale result must match frozen_states shape"
            )
        displacement = functional_state_displacement(
            states,
            frozen,
            exclude_initial=exclude_initial,
        )
        evaluated_scales.append(float(scale))
        evaluated_displacements.append(displacement)
        evaluated_states.append(states.copy())
        return displacement, states

    zero_displacement, _ = evaluate(0.0)
    if zero_displacement > zero_scale_tolerance:
        raise ValueError("rollout_at_scale(0) does not reproduce the frozen trajectory")
    tolerance = absolute_tolerance + relative_tolerance * target

    def build_result(
        index: int,
        *,
        converged: bool,
        bracket_low: float,
        bracket_high: float,
    ) -> FunctionalBudgetMatch:
        achieved = evaluated_displacements[index]
        error = abs(achieved - target)
        relative_error = (
            error / target if target > 0.0 else (0.0 if error == 0.0 else np.inf)
        )
        return FunctionalBudgetMatch(
            target_displacement=target,
            achieved_displacement=achieved,
            scale=evaluated_scales[index],
            absolute_error=error,
            relative_error=float(relative_error),
            converged=converged,
            n_evaluations=len(evaluated_scales),
            bracket_low=float(bracket_low),
            bracket_high=float(bracket_high),
            max_scale=maximum,
            exclude_initial=bool(exclude_initial),
            controlled_states=_readonly(evaluated_states[index]),
            evaluated_scales=_readonly(evaluated_scales),
            evaluated_displacements=_readonly(evaluated_displacements),
        )

    if target == 0.0:
        return build_result(
            0,
            converged=zero_displacement <= tolerance,
            bracket_low=0.0,
            bracket_high=0.0,
        )

    low_scale = 0.0
    high_scale = initial
    high_displacement, _ = evaluate(high_scale)
    while high_displacement < target and high_scale < maximum:
        low_scale = high_scale
        high_scale = min(maximum, high_scale * expansion)
        high_displacement, _ = evaluate(high_scale)

    if high_displacement < target:
        closest = int(np.argmin(np.abs(np.asarray(evaluated_displacements) - target)))
        result = build_result(
            closest,
            converged=abs(evaluated_displacements[closest] - target) <= tolerance,
            bracket_low=low_scale,
            bracket_high=high_scale,
        )
        if raise_on_unreachable and not result.converged:
            raise FunctionalBudgetError(
                "target functional displacement is unreachable by max_scale"
            )
        return result

    # Preserve a sign-changing bracket.  This remains well-defined even if the
    # displacement curve is not globally monotone.
    for _ in range(max_iterations):
        midpoint = 0.5 * (low_scale + high_scale)
        midpoint_displacement, _ = evaluate(midpoint)
        if abs(midpoint_displacement - target) <= tolerance:
            break
        if midpoint_displacement < target:
            low_scale = midpoint
        else:
            high_scale = midpoint
            high_displacement = midpoint_displacement
        if np.isclose(low_scale, high_scale, rtol=0.0, atol=np.finfo(float).eps):
            break

    closest = int(np.argmin(np.abs(np.asarray(evaluated_displacements) - target)))
    converged = abs(evaluated_displacements[closest] - target) <= tolerance
    result = build_result(
        closest,
        converged=converged,
        bracket_low=low_scale,
        bracket_high=high_scale,
    )
    if raise_on_unreachable and not converged:
        raise FunctionalBudgetError(
            "functional displacement bracket did not converge within tolerance"
        )
    return result


@dataclass(frozen=True)
class FunctionalObservables:
    """Comparable non-parameter diagnostics for an actuator rollout."""

    state_displacement: float
    mean_rate: float
    max_absolute_rate: float
    mean_gain: float
    max_gain: float
    synaptic_event_proxy: float


@dataclass(frozen=True)
class JointFunctionalBudgetAudit:
    """Receipt for a state match plus three functional safety envelopes.

    Only state displacement is an equality target.  Rate, gain, and event
    quantities are explicitly upper-bound constraints because a routing
    actuator has no population-gain excursion by construction.  Treating all
    four quantities as equality targets would therefore be impossible without
    adding a fictitious gain cost to routing and low-rank conditions.
    """

    target_state_displacement: float
    achieved_state_displacement: float
    state_relative_error: float
    state_relative_tolerance: float
    mean_absolute_rate_change: float
    target_mean_absolute_rate_change: float
    rate_change_relative_error: float
    rate_change_relative_to_frozen: float
    rate_change_relative_tolerance: float
    gain_envelope: float
    gain_envelope_limit: float
    gain_envelope_fraction: float
    synaptic_event_proxy_change: float
    event_change_relative_to_frozen: float
    event_change_relative_tolerance: float
    state_valid: bool
    rate_valid: bool
    gain_valid: bool
    event_valid: bool
    joint_valid: bool


def _same_shape_finite(
    value: ArrayLike,
    reference: FloatArray,
    *,
    name: str,
) -> FloatArray:
    result = _finite_states(value, name=name)
    if result.shape != reference.shape:
        raise ValueError(f"{name} must match its frozen reference exactly")
    return result


def audit_joint_functional_budget(
    controlled_states: ArrayLike,
    frozen_states: ArrayLike,
    *,
    controlled_rates: ArrayLike,
    frozen_rates: ArrayLike,
    controlled_gains: ArrayLike,
    frozen_gains: ArrayLike,
    controlled_event_proxy_by_step: ArrayLike,
    frozen_event_proxy_by_step: ArrayLike,
    target_state_displacement: float,
    target_mean_absolute_rate_change: float,
    state_relative_tolerance: float,
    rate_change_relative_tolerance: float,
    gain_envelope_limit: float,
    event_change_relative_tolerance: float,
    denominator_floor: float = 1e-12,
    exclude_initial: bool = True,
) -> JointFunctionalBudgetAudit:
    """Fail-closed audit of four training-only functional budget quantities.

    The rate equality quantity is
    ``mean(abs(r_controlled - r_frozen))`` relative to a supplied common
    target.  Its normalization by frozen mean absolute rate is retained as a
    diagnostic only.  The gain quantity is the maximum absolute gain departure
    from its paired frozen rollout.  The event quantity is the absolute change
    in mean event proxy normalized by the frozen mean.
    """

    controlled_state_array = _finite_states(
        controlled_states, name="controlled_states"
    )
    frozen_state_array = _same_shape_finite(
        frozen_states,
        controlled_state_array,
        name="frozen_states",
    )
    controlled_rate_array = _finite_states(
        controlled_rates, name="controlled_rates"
    )
    frozen_rate_array = _same_shape_finite(
        frozen_rates,
        controlled_rate_array,
        name="frozen_rates",
    )
    controlled_gain_array = _finite_states(
        controlled_gains, name="controlled_gains"
    )
    frozen_gain_array = _same_shape_finite(
        frozen_gains,
        controlled_gain_array,
        name="frozen_gains",
    )
    controlled_event = np.asarray(
        controlled_event_proxy_by_step, dtype=np.float64
    )
    frozen_event = np.asarray(frozen_event_proxy_by_step, dtype=np.float64)
    if controlled_event.shape != frozen_event.shape:
        raise ValueError(
            "controlled and frozen event proxy arrays must match exactly"
        )
    if (
        controlled_event.ndim < 1
        or controlled_event.size == 0
        or not np.all(np.isfinite(controlled_event))
        or not np.all(np.isfinite(frozen_event))
        or np.any(controlled_event < 0.0)
        or np.any(frozen_event < 0.0)
    ):
        raise ValueError("event proxy arrays must be finite and non-negative")

    target = _nonnegative_scalar(
        target_state_displacement, name="target_state_displacement"
    )
    target_rate_change = _nonnegative_scalar(
        target_mean_absolute_rate_change,
        name="target_mean_absolute_rate_change",
    )
    state_tolerance = _nonnegative_scalar(
        state_relative_tolerance, name="state_relative_tolerance"
    )
    rate_tolerance = _nonnegative_scalar(
        rate_change_relative_tolerance,
        name="rate_change_relative_tolerance",
    )
    gain_limit = _nonnegative_scalar(
        gain_envelope_limit, name="gain_envelope_limit"
    )
    event_tolerance = _nonnegative_scalar(
        event_change_relative_tolerance,
        name="event_change_relative_tolerance",
    )
    floor = _nonnegative_scalar(denominator_floor, name="denominator_floor")
    if floor <= 0.0:
        raise ValueError("denominator_floor must be positive")

    achieved = functional_state_displacement(
        controlled_state_array,
        frozen_state_array,
        exclude_initial=exclude_initial,
    )
    if target > 0.0:
        state_relative_error = abs(achieved - target) / target
    else:
        state_relative_error = 0.0 if achieved <= floor else np.inf

    mean_absolute_rate_change = float(
        np.mean(np.abs(controlled_rate_array - frozen_rate_array))
    )
    frozen_rate_scale = max(float(np.mean(np.abs(frozen_rate_array))), floor)
    rate_relative = mean_absolute_rate_change / frozen_rate_scale
    if target_rate_change > 0.0:
        rate_relative_error = (
            abs(mean_absolute_rate_change - target_rate_change)
            / target_rate_change
        )
    else:
        rate_relative_error = (
            0.0 if mean_absolute_rate_change <= floor else np.inf
        )

    gain_envelope = float(
        np.max(np.abs(controlled_gain_array - frozen_gain_array))
    )
    gain_fraction = (
        gain_envelope / gain_limit
        if gain_limit > 0.0
        else (0.0 if gain_envelope <= floor else np.inf)
    )

    # The registered proxy is total/cumulative synaptic traffic, so compare
    # mean event mass rather than pointwise timing rearrangements.
    event_change = abs(float(np.mean(controlled_event) - np.mean(frozen_event)))
    frozen_event_scale = max(float(np.mean(np.abs(frozen_event))), floor)
    event_relative = event_change / frozen_event_scale

    state_valid = bool(state_relative_error <= state_tolerance)
    rate_valid = bool(rate_relative_error <= rate_tolerance)
    gain_valid = bool(gain_envelope <= gain_limit + floor)
    event_valid = bool(event_relative <= event_tolerance)
    joint_valid = state_valid and rate_valid and gain_valid and event_valid
    return JointFunctionalBudgetAudit(
        target_state_displacement=target,
        achieved_state_displacement=achieved,
        state_relative_error=float(state_relative_error),
        state_relative_tolerance=state_tolerance,
        mean_absolute_rate_change=mean_absolute_rate_change,
        target_mean_absolute_rate_change=target_rate_change,
        rate_change_relative_error=float(rate_relative_error),
        rate_change_relative_to_frozen=float(rate_relative),
        rate_change_relative_tolerance=rate_tolerance,
        gain_envelope=gain_envelope,
        gain_envelope_limit=gain_limit,
        gain_envelope_fraction=float(gain_fraction),
        synaptic_event_proxy_change=event_change,
        event_change_relative_to_frozen=float(event_relative),
        event_change_relative_tolerance=event_tolerance,
        state_valid=state_valid,
        rate_valid=rate_valid,
        gain_valid=gain_valid,
        event_valid=event_valid,
        joint_valid=joint_valid,
    )


def functional_observables(
    controlled_states: ArrayLike,
    frozen_states: ArrayLike,
    *,
    rates: ArrayLike,
    gains: ArrayLike,
    synaptic_event_proxy_by_step: ArrayLike,
    exclude_initial: bool = True,
) -> FunctionalObservables:
    """Collect state, firing, gain, and synaptic-event budget diagnostics."""

    rate_array = _finite_states(rates, name="rates")
    gain_array = _finite_states(gains, name="gains")
    event_array = np.asarray(synaptic_event_proxy_by_step)
    if event_array.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("synaptic_event_proxy_by_step must be real numeric")
    event_array = np.asarray(event_array, dtype=np.float64)
    if event_array.ndim < 1 or event_array.size == 0:
        raise ValueError("synaptic_event_proxy_by_step must be non-empty")
    if not np.all(np.isfinite(event_array)) or np.any(event_array < 0.0):
        raise ValueError("synaptic_event_proxy_by_step must be finite and non-negative")
    return FunctionalObservables(
        state_displacement=functional_state_displacement(
            controlled_states,
            frozen_states,
            exclude_initial=exclude_initial,
        ),
        mean_rate=float(np.mean(rate_array)),
        max_absolute_rate=float(np.max(np.abs(rate_array))),
        mean_gain=float(np.mean(gain_array)),
        max_gain=float(np.max(gain_array)),
        synaptic_event_proxy=float(np.mean(event_array)),
    )


__all__ = [
    "FunctionalBudgetError",
    "FunctionalBudgetMatch",
    "FunctionalObservables",
    "JointFunctionalBudgetAudit",
    "audit_joint_functional_budget",
    "functional_observables",
    "functional_state_displacement",
    "match_functional_state_displacement",
]
