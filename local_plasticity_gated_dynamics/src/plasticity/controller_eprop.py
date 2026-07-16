"""Diagonal eligibility propagation for an axis-only local controller.

The learned object is a controller vector, never the recurrent matrix.  Given
state-transition partials, the local approximation replaces the full state
Jacobian by its diagonal:

``E[t + 1] = diag(diag(J[t])) @ E[t] + B[t]``.

Learning signals are contracted with the eligibility available at the same
chronological state.  All updates are explicit NumPy operations and no loss
graph is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _real_array(value: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != ndim or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty {ndim}D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _nonnegative_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _jacobian_diagonal(
    value: ArrayLike,
    *,
    n_steps: int | None = None,
    state_dim: int | None = None,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim == 2:
        diagonal = _real_array(raw, name="state_jacobians", ndim=2)
    elif raw.ndim == 3:
        matrices = _real_array(raw, name="state_jacobians", ndim=3)
        if matrices.shape[1] != matrices.shape[2]:
            raise ValueError("state_jacobians must contain square matrices")
        diagonal = np.diagonal(matrices, axis1=1, axis2=2).copy()
    else:
        raise ValueError(
            "state_jacobians must be [time, state] diagonals or square matrices"
        )
    if n_steps is not None and diagonal.shape[0] != n_steps:
        raise ValueError("state_jacobians have the wrong number of time steps")
    if state_dim is not None and diagonal.shape[1] != state_dim:
        raise ValueError("state_jacobians have the wrong state dimension")
    return diagonal


@dataclass(frozen=True, slots=True)
class DiagonalEPropTrajectory:
    """Immutable trace of the diagonal eligibility approximation."""

    eligibilities: FloatArray
    jacobian_diagonal: FloatArray
    control_jacobians: FloatArray
    max_recurrence_residual: float
    n_steps: int
    state_dim: int
    control_dim: int
    approximation: str = "state_jacobian_diagonal_only"
    recurrent_parameters_trainable: bool = False


def diagonal_eprop_sensitivities(
    state_jacobians: ArrayLike,
    control_jacobians: ArrayLike,
    *,
    initial_eligibility: ArrayLike | None = None,
) -> DiagonalEPropTrajectory:
    """Evaluate the diagonal eligibility recurrence in forward time."""

    control = _real_array(
        control_jacobians,
        name="control_jacobians",
        ndim=3,
    )
    n_steps, state_dim, control_dim = control.shape
    diagonal = _jacobian_diagonal(
        state_jacobians,
        n_steps=n_steps,
        state_dim=state_dim,
    )
    if initial_eligibility is None:
        initial = np.zeros((state_dim, control_dim), dtype=np.float64)
    else:
        initial = _real_array(
            initial_eligibility,
            name="initial_eligibility",
            ndim=2,
        )
        if initial.shape != (state_dim, control_dim):
            raise ValueError(
                "initial_eligibility must have shape [state, parameter]"
            )

    eligibilities = np.empty(
        (n_steps + 1, state_dim, control_dim), dtype=np.float64
    )
    eligibilities[0] = initial
    max_residual = 0.0
    for time in range(n_steps):
        expected = diagonal[time, :, None] * eligibilities[time] + control[time]
        eligibilities[time + 1] = expected
        residual = float(np.max(np.abs(eligibilities[time + 1] - expected)))
        max_residual = max(max_residual, residual)

    return DiagonalEPropTrajectory(
        eligibilities=_readonly(eligibilities),
        jacobian_diagonal=_readonly(diagonal),
        control_jacobians=_readonly(control),
        max_recurrence_residual=max_residual,
        n_steps=int(n_steps),
        state_dim=int(state_dim),
        control_dim=int(control_dim),
    )


@dataclass(frozen=True, slots=True)
class DiagonalEPropGradient:
    """Controller gradient estimated from local learning signals."""

    gradient: FloatArray
    per_time_gradient: FloatArray
    n_loss_times: int
    control_dim: int


@dataclass(frozen=True, slots=True)
class CompactGainAxisEPropTrajectory:
    """Diagonal eligibility using the compact E/I gain-axis partials."""

    eligibilities: FloatArray
    jacobian_diagonal: FloatArray
    rate_axis_direct_derivatives: FloatArray
    max_recurrence_residual: float
    n_steps: int
    n_units: int
    control_dim: int
    approximation: str = "augmented_state_jacobian_diagonal_only"
    direct_control_structure: str = "rate_block_diagonal"
    recurrent_parameters_trainable: bool = False


@dataclass(frozen=True, slots=True)
class BlockLocalGainAxisEPropTrajectory:
    """Per-unit ``[state, rate]`` eligibility for a gain controller.

    The approximation keeps the full local two-state block for each unit but
    omits cross-neuron Jacobian entries.  This remains chronological and
    neuron-local while avoiding the degenerate pure-diagonal trace that occurs
    when a Dale network forbids self-connections.
    """

    eligibilities: FloatArray
    local_jacobian_blocks: FloatArray
    state_axis_direct_derivatives: FloatArray
    rate_axis_direct_derivatives: FloatArray
    max_recurrence_residual: float
    n_steps: int
    n_units: int
    control_dim: int
    approximation: str = "per_unit_state_rate_block_jacobian"
    direct_control_structure: str = "state_and_rate_block_diagonal"
    recurrent_parameters_trainable: bool = False


def block_local_gain_axis_eprop_sensitivities(
    local_jacobian_blocks: ArrayLike,
    state_axis_direct_derivatives: ArrayLike,
    rate_axis_direct_derivatives: ArrayLike,
    *,
    initial_eligibility: ArrayLike | None = None,
) -> BlockLocalGainAxisEPropTrajectory:
    """Propagate a non-degenerate local ``[x_i, r_i]`` eligibility trace.

    No recurrent weight is trainable and no cross-neuron sensitivity is
    transported.  Each local two-state block is evaluated forward in time.
    """

    blocks = _real_array(
        local_jacobian_blocks,
        name="local_jacobian_blocks",
        ndim=4,
    )
    state_direct = _real_array(
        state_axis_direct_derivatives,
        name="state_axis_direct_derivatives",
        ndim=2,
    )
    rate_direct = _real_array(
        rate_axis_direct_derivatives,
        name="rate_axis_direct_derivatives",
        ndim=2,
    )
    n_steps, n_units = state_direct.shape
    if rate_direct.shape != (n_steps, n_units):
        raise ValueError(
            "rate_axis_direct_derivatives must match state direct derivatives"
        )
    if blocks.shape != (n_steps, n_units, 2, 2):
        raise ValueError(
            "local_jacobian_blocks must have shape [time, unit, 2, 2]"
        )
    if initial_eligibility is None:
        initial = np.zeros((2 * n_units, n_units), dtype=np.float64)
    else:
        initial = _real_array(
            initial_eligibility,
            name="initial_eligibility",
            ndim=2,
        )
        if initial.shape != (2 * n_units, n_units):
            raise ValueError(
                "initial_eligibility must have shape [2 * unit, unit]"
            )

    eligibility = np.empty(
        (n_steps + 1, 2 * n_units, n_units),
        dtype=np.float64,
    )
    eligibility[0] = initial
    units = np.arange(n_units)
    max_residual = 0.0
    for time in range(n_steps):
        previous_state = eligibility[time, :n_units]
        previous_rate = eligibility[time, n_units:]
        next_state = (
            blocks[time, :, 0, 0, None] * previous_state
            + blocks[time, :, 0, 1, None] * previous_rate
        )
        next_rate = (
            blocks[time, :, 1, 0, None] * previous_state
            + blocks[time, :, 1, 1, None] * previous_rate
        )
        next_state[units, units] += state_direct[time]
        next_rate[units, units] += rate_direct[time]
        expected = np.concatenate((next_state, next_rate), axis=0)
        eligibility[time + 1] = expected
        residual = float(np.max(np.abs(eligibility[time + 1] - expected)))
        max_residual = max(max_residual, residual)

    return BlockLocalGainAxisEPropTrajectory(
        eligibilities=_readonly(eligibility),
        local_jacobian_blocks=_readonly(blocks),
        state_axis_direct_derivatives=_readonly(state_direct),
        rate_axis_direct_derivatives=_readonly(rate_direct),
        max_recurrence_residual=max_residual,
        n_steps=int(n_steps),
        n_units=int(n_units),
        control_dim=int(n_units),
    )


def diagonal_gain_axis_eprop_sensitivities(
    jacobian_diagonal: ArrayLike,
    rate_axis_direct_derivatives: ArrayLike,
    *,
    initial_eligibility: ArrayLike | None = None,
) -> CompactGainAxisEPropTrajectory:
    """Propagate gain-axis eligibility without materializing ``[T,2N,N]`` B.

    ``jacobian_diagonal`` is the compact ``[time, 2 * unit]`` field returned
    by the frozen gain-axis trajectory.  The direct controller derivative is
    diagonal and exists only in the rate half of the augmented state.
    """

    diagonal = _real_array(
        jacobian_diagonal,
        name="jacobian_diagonal",
        ndim=2,
    )
    direct = _real_array(
        rate_axis_direct_derivatives,
        name="rate_axis_direct_derivatives",
        ndim=2,
    )
    n_steps, n_units = direct.shape
    if diagonal.shape != (n_steps, 2 * n_units):
        raise ValueError(
            "jacobian_diagonal must have shape [time, 2 * unit]"
        )
    if initial_eligibility is None:
        initial = np.zeros((2 * n_units, n_units), dtype=np.float64)
    else:
        initial = _real_array(
            initial_eligibility,
            name="initial_eligibility",
            ndim=2,
        )
        if initial.shape != (2 * n_units, n_units):
            raise ValueError(
                "initial_eligibility must have shape [2 * unit, unit]"
            )

    eligibility = np.empty(
        (n_steps + 1, 2 * n_units, n_units),
        dtype=np.float64,
    )
    eligibility[0] = initial
    units = np.arange(n_units)
    max_residual = 0.0
    for time in range(n_steps):
        expected = diagonal[time, :, None] * eligibility[time]
        expected[n_units + units, units] += direct[time]
        eligibility[time + 1] = expected
        residual = float(np.max(np.abs(eligibility[time + 1] - expected)))
        max_residual = max(max_residual, residual)
    return CompactGainAxisEPropTrajectory(
        eligibilities=_readonly(eligibility),
        jacobian_diagonal=_readonly(diagonal),
        rate_axis_direct_derivatives=_readonly(direct),
        max_recurrence_residual=max_residual,
        n_steps=int(n_steps),
        n_units=int(n_units),
        control_dim=int(n_units),
    )


def diagonal_eprop_gradient(
    eligibilities: (
        ArrayLike
        | DiagonalEPropTrajectory
        | CompactGainAxisEPropTrajectory
        | BlockLocalGainAxisEPropTrajectory
    ),
    learning_signals: ArrayLike,
) -> DiagonalEPropGradient:
    """Contract time-local learning signals with diagonal eligibilities."""

    source = (
        eligibilities.eligibilities
        if isinstance(
            eligibilities,
            (
                DiagonalEPropTrajectory,
                CompactGainAxisEPropTrajectory,
                BlockLocalGainAxisEPropTrajectory,
            ),
        )
        else eligibilities
    )
    trace = _real_array(source, name="eligibilities", ndim=3)
    signals = _real_array(learning_signals, name="learning_signals", ndim=2)
    if signals.shape != trace.shape[:2]:
        raise ValueError("learning_signals must align with eligibility time and state")
    per_time = np.einsum("ts,tsp->tp", signals, trace, optimize=True)
    gradient = np.sum(per_time, axis=0)
    return DiagonalEPropGradient(
        gradient=_readonly(gradient),
        per_time_gradient=_readonly(per_time),
        n_loss_times=int(trace.shape[0]),
        control_dim=int(trace.shape[2]),
    )


@dataclass(frozen=True, slots=True)
class ControllerEPropStep:
    """One causal eligibility update and learning-signal contraction."""

    eligibility: FloatArray
    local_gradient: FloatArray
    accumulated_gradient: FloatArray
    step_index: int


@dataclass(frozen=True, slots=True)
class ControllerEPropUpdate:
    """One explicit controller-vector update proposal."""

    estimated_gradient: FloatArray
    task_gradient: FloatArray
    decay_gradient: FloatArray
    activity_gradient: FloatArray
    raw_update: FloatArray
    raw_l1: float
    raw_l2: float
    steps_accumulated: int
    recurrent_parameters_trainable: bool = False


class DiagonalControllerEPropRule:
    """Stateful online rule for one frozen-dynamics controller vector."""

    def __init__(
        self,
        n_state: int,
        n_parameters: int,
        *,
        learning_rate: float,
        l2_decay: float = 0.0,
    ) -> None:
        self.n_state = _positive_integer(n_state, name="n_state")
        self.n_parameters = _positive_integer(
            n_parameters,
            name="n_parameters",
        )
        self.learning_rate = _nonnegative_scalar(
            learning_rate,
            name="learning_rate",
        )
        self.l2_decay = _nonnegative_scalar(l2_decay, name="l2_decay")
        self._eligibility = np.zeros(
            (self.n_state, self.n_parameters),
            dtype=np.float64,
        )
        self._gradient = np.zeros(self.n_parameters, dtype=np.float64)
        self._steps = 0

    @property
    def eligibility(self) -> FloatArray:
        return self._eligibility.copy()

    @property
    def accumulated_gradient(self) -> FloatArray:
        return self._gradient.copy()

    @property
    def steps_accumulated(self) -> int:
        return self._steps

    def reset(self, *, initial_eligibility: ArrayLike | None = None) -> None:
        """Reset episode-local eligibility and accumulated task gradient."""

        if initial_eligibility is None:
            self._eligibility.fill(0.0)
        else:
            initial = _real_array(
                initial_eligibility,
                name="initial_eligibility",
                ndim=2,
            )
            if initial.shape != self._eligibility.shape:
                raise ValueError(
                    "initial_eligibility must match [state, parameter]"
                )
            self._eligibility[...] = initial
        self._gradient.fill(0.0)
        self._steps = 0

    def advance(
        self,
        state_jacobian: ArrayLike,
        control_jacobian: ArrayLike,
        learning_signal: ArrayLike,
    ) -> ControllerEPropStep:
        """Advance eligibility once and consume the next-state learning signal."""

        raw_jacobian = np.asarray(state_jacobian)
        if raw_jacobian.ndim == 1:
            diagonal = _real_array(
                raw_jacobian,
                name="state_jacobian",
                ndim=1,
            )
        elif raw_jacobian.ndim == 2:
            matrix = _real_array(
                raw_jacobian,
                name="state_jacobian",
                ndim=2,
            )
            if matrix.shape != (self.n_state, self.n_state):
                raise ValueError(
                    "state_jacobian matrix must have shape [state, state]"
                )
            diagonal = np.diag(matrix)
        else:
            raise ValueError("state_jacobian must be a vector or square matrix")
        if diagonal.shape != (self.n_state,):
            raise ValueError("state_jacobian diagonal has the wrong length")
        direct = _real_array(
            control_jacobian,
            name="control_jacobian",
            ndim=2,
        )
        if direct.shape != self._eligibility.shape:
            raise ValueError(
                "control_jacobian must have shape [state, parameter]"
            )
        signal = _real_array(
            learning_signal,
            name="learning_signal",
            ndim=1,
        )
        if signal.shape != (self.n_state,):
            raise ValueError("learning_signal must have one value per state")

        next_eligibility = diagonal[:, None] * self._eligibility + direct
        local_gradient = signal @ next_eligibility
        next_gradient = self._gradient + local_gradient
        self._eligibility = next_eligibility
        self._gradient = next_gradient
        self._steps += 1
        return ControllerEPropStep(
            eligibility=_readonly(next_eligibility),
            local_gradient=_readonly(local_gradient),
            accumulated_gradient=_readonly(next_gradient),
            step_index=self._steps,
        )

    def propose(
        self,
        parameters: ArrayLike,
        *,
        activity_gradient: ArrayLike | None = None,
    ) -> ControllerEPropUpdate:
        """Return a descent proposal without mutating controller parameters."""

        parameter = _real_array(parameters, name="parameters", ndim=1)
        if parameter.shape != (self.n_parameters,):
            raise ValueError("parameters have the wrong length")
        if activity_gradient is None:
            activity = np.zeros(self.n_parameters, dtype=np.float64)
        else:
            activity = _real_array(
                activity_gradient,
                name="activity_gradient",
                ndim=1,
            )
            if activity.shape != (self.n_parameters,):
                raise ValueError("activity_gradient has the wrong length")
        decay = self.l2_decay * parameter
        estimated = self._gradient + decay + activity
        update = -self.learning_rate * estimated
        return ControllerEPropUpdate(
            estimated_gradient=_readonly(estimated),
            task_gradient=_readonly(self._gradient),
            decay_gradient=_readonly(decay),
            activity_gradient=_readonly(activity),
            raw_update=_readonly(update),
            raw_l1=float(np.sum(np.abs(update))),
            raw_l2=float(np.linalg.norm(update)),
            steps_accumulated=self._steps,
        )


@dataclass(frozen=True, slots=True)
class GainBoundApplication:
    """Global downscaling needed to keep all audited gains within bounds."""

    raw_update: FloatArray
    applied_update: FloatArray
    scale_factor: float
    gain_min_observed: float
    gain_max_observed: float
    constrained: bool
    n_belief_values: int


def apply_gain_bounds(
    parameters: ArrayLike,
    proposed_update: ArrayLike,
    signed_beliefs: ArrayLike,
    *,
    gain_min: float,
    gain_max: float,
    tolerance: float = 1e-12,
) -> GainBoundApplication:
    """Globally downscale an axis update to satisfy functional gain bounds.

    A single scale in ``[0, 1]`` preserves the proposed direction.  The current
    parameters must already be feasible for every supplied belief value.
    """

    parameter = _real_array(parameters, name="parameters", ndim=1)
    update = _real_array(proposed_update, name="proposed_update", ndim=1)
    beliefs = _real_array(signed_beliefs, name="signed_beliefs", ndim=1)
    if update.shape != parameter.shape:
        raise ValueError("proposed_update must match parameters")
    if np.any((beliefs < -1.0) | (beliefs > 1.0)):
        raise ValueError("signed_beliefs must lie in [-1, 1]")
    lower = _nonnegative_scalar(gain_min, name="gain_min")
    upper = _nonnegative_scalar(gain_max, name="gain_max")
    tol = _nonnegative_scalar(tolerance, name="tolerance")
    if lower <= 0.0 or not lower <= 1.0 <= upper or lower >= upper:
        raise ValueError(
            "gain bounds must be positive, ordered, and contain neutral gain one"
        )
    current_gain = 1.0 + beliefs[:, None] * parameter[None, :]
    if np.any(current_gain < lower - tol) or np.any(current_gain > upper + tol):
        raise ValueError("current parameters violate the requested gain bounds")

    change = beliefs[:, None] * update[None, :]
    scale = 1.0
    positive = change > 0.0
    negative = change < 0.0
    if np.any(positive):
        scale = min(
            scale,
            float(np.min((upper - current_gain[positive]) / change[positive])),
        )
    if np.any(negative):
        scale = min(
            scale,
            float(np.min((lower - current_gain[negative]) / change[negative])),
        )
    scale = float(np.clip(scale, 0.0, 1.0))
    applied = scale * update
    resulting_gain = 1.0 + beliefs[:, None] * (
        parameter[None, :] + applied[None, :]
    )
    if np.any(resulting_gain < lower - tol) or np.any(
        resulting_gain > upper + tol
    ):
        raise RuntimeError("gain-bound scaling failed its functional constraint")
    return GainBoundApplication(
        raw_update=_readonly(update),
        applied_update=_readonly(applied),
        scale_factor=scale,
        gain_min_observed=float(np.min(resulting_gain)),
        gain_max_observed=float(np.max(resulting_gain)),
        constrained=bool(scale < 1.0 - tol),
        n_belief_values=int(beliefs.size),
    )


__all__ = [
    "BlockLocalGainAxisEPropTrajectory",
    "ControllerEPropStep",
    "ControllerEPropUpdate",
    "CompactGainAxisEPropTrajectory",
    "DiagonalControllerEPropRule",
    "DiagonalEPropGradient",
    "DiagonalEPropTrajectory",
    "GainBoundApplication",
    "apply_gain_bounds",
    "block_local_gain_axis_eprop_sensitivities",
    "diagonal_eprop_gradient",
    "diagonal_eprop_sensitivities",
    "diagonal_gain_axis_eprop_sensitivities",
]
