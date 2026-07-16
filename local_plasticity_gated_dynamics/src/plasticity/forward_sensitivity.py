"""Exact forward-mode sensitivities for a frozen recurrent system.

For a discrete transition ``x[t + 1] = F_t(x[t], a)``, the sensitivity of
state to the controller parameters obeys

``S[t + 1] = J[t] @ S[t] + B[t]``,

where ``J[t] = dF_t/dx`` and ``B[t] = dF_t/da``.  This module evaluates that
recurrence in chronological order.  It does not retain a loss graph or
differentiate recurrent weights.

The gain-axis helpers use the frozen recurrent and input matrices from
:class:`src.models.ei_rate_network.EIRateNetwork`.  The Markov state is the
pair ``(x, rates)``.  Two explicitly audited gain placements are supported:

``x_next = x + dt/tau * (-x + W @ rates + external_drive)``

``rates_next = phi(gain * x_next)``.

and the non-degenerate local-state controller used by Exp23:

``x_next = x + dt/tau * (-x + gain * (W @ rates + external_drive))``

``rates_next = phi(x_next)``.

Only ``axis`` is treated as a parameter; the recurrent matrix is read-only
input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ActivationName = Literal["tanh", "rectified_tanh"]
GainApplication = Literal["rate", "drive"]


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _real_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int,
    nonempty: bool = True,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != ndim or (nonempty and 0 in array.shape):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{name} must be a {qualifier}{ndim}D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


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


@dataclass(frozen=True, slots=True)
class ForwardSensitivityTrajectory:
    """Immutable receipt for an exact chronological sensitivity recurrence."""

    sensitivities: FloatArray
    state_jacobians: FloatArray
    control_jacobians: FloatArray
    max_recurrence_residual: float
    n_steps: int
    state_dim: int
    control_dim: int
    recurrence: str = "S_next_equals_J_times_S_plus_B"
    recurrent_parameters_trainable: bool = False


def exact_forward_sensitivities(
    state_jacobians: ArrayLike,
    control_jacobians: ArrayLike,
    *,
    initial_sensitivity: ArrayLike | None = None,
) -> ForwardSensitivityTrajectory:
    """Evaluate ``S[t + 1] = J[t] @ S[t] + B[t]`` exactly.

    Parameters
    ----------
    state_jacobians:
        Array with shape ``[time, state, state]``.
    control_jacobians:
        Array with shape ``[time, state, parameter]``.
    initial_sensitivity:
        Optional ``dx[0]/da`` matrix.  The default is zero, corresponding to
        an initial state that is independent of the controller.
    """

    jacobian = _real_array(state_jacobians, name="state_jacobians", ndim=3)
    control = _real_array(control_jacobians, name="control_jacobians", ndim=3)
    n_steps, state_dim, second_state_dim = jacobian.shape
    if state_dim != second_state_dim:
        raise ValueError("state_jacobians must contain square matrices")
    if control.shape[:2] != (n_steps, state_dim):
        raise ValueError(
            "control_jacobians must align with state_jacobians in time and state"
        )
    control_dim = int(control.shape[2])
    if initial_sensitivity is None:
        initial = np.zeros((state_dim, control_dim), dtype=np.float64)
    else:
        initial = _real_array(
            initial_sensitivity,
            name="initial_sensitivity",
            ndim=2,
        )
        if initial.shape != (state_dim, control_dim):
            raise ValueError(
                "initial_sensitivity must have shape [state, parameter]"
            )

    sensitivities = np.empty(
        (n_steps + 1, state_dim, control_dim), dtype=np.float64
    )
    sensitivities[0] = initial
    max_residual = 0.0
    for time in range(n_steps):
        expected = jacobian[time] @ sensitivities[time] + control[time]
        sensitivities[time + 1] = expected
        residual = float(np.max(np.abs(sensitivities[time + 1] - expected)))
        max_residual = max(max_residual, residual)

    return ForwardSensitivityTrajectory(
        sensitivities=_readonly(sensitivities),
        state_jacobians=_readonly(jacobian),
        control_jacobians=_readonly(control),
        max_recurrence_residual=max_residual,
        n_steps=int(n_steps),
        state_dim=int(state_dim),
        control_dim=control_dim,
    )


@dataclass(frozen=True, slots=True)
class ForwardGradient:
    """Loss gradient obtained by contracting forward sensitivities."""

    gradient: FloatArray
    state_contribution: FloatArray
    direct_contribution: FloatArray
    n_loss_times: int
    control_dim: int


def exact_forward_gradient(
    sensitivities: ArrayLike | ForwardSensitivityTrajectory,
    state_loss_gradients: ArrayLike,
    *,
    direct_parameter_gradients: ArrayLike | None = None,
) -> ForwardGradient:
    """Contract ``dL/dx[t]`` with ``dx[t]/da`` over chronological states.

    ``state_loss_gradients`` must include the initial state and therefore have
    shape ``[time + 1, state]``.  Optional direct parameter terms may be one
    vector or a time-aligned matrix; in both cases they are added exactly once
    per supplied row.
    """

    source = (
        sensitivities.sensitivities
        if isinstance(sensitivities, ForwardSensitivityTrajectory)
        else sensitivities
    )
    trace = _real_array(source, name="sensitivities", ndim=3)
    loss_gradient = _real_array(
        state_loss_gradients,
        name="state_loss_gradients",
        ndim=2,
    )
    if loss_gradient.shape != trace.shape[:2]:
        raise ValueError(
            "state_loss_gradients must align with sensitivity time and state"
        )
    state_contribution = np.einsum(
        "ts,tsp->p", loss_gradient, trace, optimize=True
    )
    control_dim = int(trace.shape[2])
    if direct_parameter_gradients is None:
        direct = np.zeros(control_dim, dtype=np.float64)
    else:
        raw_direct = np.asarray(direct_parameter_gradients)
        if raw_direct.ndim == 1:
            direct = _real_array(
                raw_direct,
                name="direct_parameter_gradients",
                ndim=1,
            )
            if direct.shape != (control_dim,):
                raise ValueError(
                    "direct_parameter_gradients vector has the wrong length"
                )
        elif raw_direct.ndim == 2:
            direct_time = _real_array(
                raw_direct,
                name="direct_parameter_gradients",
                ndim=2,
            )
            if direct_time.shape[1] != control_dim:
                raise ValueError(
                    "direct_parameter_gradients matrix has the wrong width"
                )
            direct = np.sum(direct_time, axis=0)
        else:
            raise ValueError(
                "direct_parameter_gradients must be a vector or a 2D array"
            )
    gradient = state_contribution + direct
    return ForwardGradient(
        gradient=_readonly(gradient),
        state_contribution=_readonly(state_contribution),
        direct_contribution=_readonly(direct),
        n_loss_times=int(trace.shape[0]),
        control_dim=control_dim,
    )


def _activation_and_derivative(
    preactivation: FloatArray,
    *,
    activation: ActivationName,
) -> tuple[FloatArray, FloatArray]:
    tanh_value = np.tanh(preactivation)
    derivative = 1.0 - tanh_value * tanh_value
    if activation == "tanh":
        return tanh_value, derivative
    if activation == "rectified_tanh":
        active = tanh_value > 0.0
        return np.where(active, tanh_value, 0.0), np.where(active, derivative, 0.0)
    raise ValueError("activation must be 'tanh' or 'rectified_tanh'")


@dataclass(frozen=True, slots=True)
class FrozenGainAxisStepPartials:
    """One frozen-network Euler transition and its analytic partials."""

    next_state: FloatArray
    next_rates: FloatArray
    gains: FloatArray
    gain_derivative: FloatArray
    state_jacobian: FloatArray
    control_jacobian: FloatArray
    saturation_mask: NDArray[np.bool_]
    signed_belief: float
    gain_application: GainApplication
    recurrent_parameters_trainable: bool = False


def frozen_gain_axis_step_partials(
    state: ArrayLike,
    rates: ArrayLike,
    axis: ArrayLike,
    recurrent_weights: ArrayLike,
    external_drive: ArrayLike,
    dt_over_tau: ArrayLike,
    *,
    signed_belief: float,
    gain_min: float,
    gain_max: float,
    activation: ActivationName = "rectified_tanh",
    gain_application: GainApplication = "rate",
) -> FrozenGainAxisStepPartials:
    """Return one E/I Euler step and exact augmented-state partials.

    ``state_jacobian`` differentiates ``[x_next, rates_next]`` with respect to
    ``[x, rates]`` and therefore has shape ``[2N, 2N]``.
    ``control_jacobian`` differentiates the same augmented next state with
    respect to the ``N`` axis parameters.  The recurrent matrix and external
    drive are constants.  At a clipping boundary the gain derivative is
    defined as zero, matching the constant branch used by a clipped controller.
    """

    x = _real_array(state, name="state", ndim=1)
    current_rates = _real_array(rates, name="rates", ndim=1)
    parameter = _real_array(axis, name="axis", ndim=1)
    weights = _real_array(
        recurrent_weights,
        name="recurrent_weights",
        ndim=2,
    )
    drive = _real_array(external_drive, name="external_drive", ndim=1)
    step_fraction = _real_array(dt_over_tau, name="dt_over_tau", ndim=1)
    n_units = int(x.size)
    if current_rates.shape != (n_units,) or parameter.shape != (n_units,):
        raise ValueError("rates and axis must have one value per state coordinate")
    if weights.shape != (n_units, n_units):
        raise ValueError("recurrent_weights must have shape [state, state]")
    if drive.shape != (n_units,) or step_fraction.shape != (n_units,):
        raise ValueError("external_drive and dt_over_tau must match state")
    if np.any((step_fraction <= 0.0) | (step_fraction > 1.0)):
        raise ValueError("dt_over_tau values must lie in (0, 1]")
    belief = _finite_scalar(signed_belief, name="signed_belief")
    if not -1.0 <= belief <= 1.0:
        raise ValueError("signed_belief must lie in [-1, 1]")
    lower = _finite_scalar(gain_min, name="gain_min")
    upper = _finite_scalar(gain_max, name="gain_max")
    if lower <= 0.0 or not lower <= 1.0 <= upper or lower >= upper:
        raise ValueError(
            "gain bounds must be positive, ordered, and contain neutral gain one"
        )

    if gain_application not in {"rate", "drive"}:
        raise ValueError("gain_application must be 'rate' or 'drive'")
    raw_gain = 1.0 + belief * parameter
    gains = np.clip(raw_gain, lower, upper)
    unsaturated = (raw_gain > lower) & (raw_gain < upper)
    gain_derivative = belief * unsaturated.astype(np.float64)
    state_from_state = np.diag(1.0 - step_fraction)
    total_drive = weights @ current_rates + drive
    if gain_application == "rate":
        next_state = x + step_fraction * (-x + total_drive)
        next_rates, activation_derivative = _activation_and_derivative(
            gains * next_state,
            activation=activation,
        )
        state_from_rates = step_fraction[:, None] * weights
        rate_from_next_state = activation_derivative * gains
        state_axis_derivative = np.zeros(n_units, dtype=np.float64)
        rate_axis_derivative = (
            activation_derivative * next_state * gain_derivative
        )
    else:
        next_state = x + step_fraction * (-x + gains * total_drive)
        next_rates, activation_derivative = _activation_and_derivative(
            next_state,
            activation=activation,
        )
        state_from_rates = (step_fraction * gains)[:, None] * weights
        rate_from_next_state = activation_derivative
        state_axis_derivative = (
            step_fraction * total_drive * gain_derivative
        )
        rate_axis_derivative = (
            activation_derivative * state_axis_derivative
        )
    rate_from_state = rate_from_next_state[:, None] * state_from_state
    rate_from_rates = rate_from_next_state[:, None] * state_from_rates
    state_jacobian = np.block(
        [
            [state_from_state, state_from_rates],
            [rate_from_state, rate_from_rates],
        ]
    )
    control_jacobian = np.vstack(
        (
            np.diag(state_axis_derivative),
            np.diag(rate_axis_derivative),
        )
    )
    frozen_saturation = np.array(~unsaturated, dtype=bool, copy=True)
    frozen_saturation.setflags(write=False)
    return FrozenGainAxisStepPartials(
        next_state=_readonly(next_state),
        next_rates=_readonly(next_rates),
        gains=_readonly(gains),
        gain_derivative=_readonly(gain_derivative),
        state_jacobian=_readonly(state_jacobian),
        control_jacobian=_readonly(control_jacobian),
        saturation_mask=frozen_saturation,
        signed_belief=belief,
        gain_application=gain_application,
    )


@dataclass(frozen=True, slots=True)
class FrozenGainAxisTrajectory:
    """Full frozen-network state, rate, and axis-sensitivity trajectory."""

    states: FloatArray
    rates: FloatArray
    gains: FloatArray
    state_sensitivities: FloatArray
    rate_sensitivities: FloatArray
    augmented_sensitivities: FloatArray | None
    jacobian_diagonal: FloatArray
    local_jacobian_blocks: FloatArray
    state_axis_direct_derivatives: FloatArray
    rate_axis_direct_derivatives: FloatArray
    augmented_state_jacobians: FloatArray | None
    augmented_control_jacobians: FloatArray | None
    saturation_mask: NDArray[np.bool_]
    n_steps: int
    n_units: int
    control_dim: int
    stored_augmented_partials: bool
    gain_application: GainApplication
    recurrent_parameters_trainable: bool = False
    trajectory_scope: str = "single_episode_chronological_closed_loop"


def simulate_frozen_gain_axis_trajectory(
    external_drives: ArrayLike,
    signed_beliefs: ArrayLike,
    axis: ArrayLike,
    recurrent_weights: ArrayLike,
    dt_over_tau: ArrayLike,
    *,
    gain_min: float,
    gain_max: float,
    activation: ActivationName = "rectified_tanh",
    initial_state: ArrayLike | None = None,
    initial_rates: ArrayLike | None = None,
    store_augmented_partials: bool = False,
    gain_application: GainApplication = "rate",
) -> FrozenGainAxisTrajectory:
    """Simulate one episode and propagate exact axis sensitivities online.

    ``external_drives`` contains the already projected input and any fixed
    noise tape, with shape ``[time, unit]``.  Initial rates default to the
    activation of the initial state at neutral gain, matching the receiver's
    deterministic initialization.
    """

    drives = _real_array(external_drives, name="external_drives", ndim=2)
    beliefs = _real_array(signed_beliefs, name="signed_beliefs", ndim=1)
    parameter = _real_array(axis, name="axis", ndim=1)
    weights = _real_array(
        recurrent_weights,
        name="recurrent_weights",
        ndim=2,
    )
    step_fraction = _real_array(dt_over_tau, name="dt_over_tau", ndim=1)
    n_steps, n_units = drives.shape
    if beliefs.shape != (n_steps,):
        raise ValueError("signed_beliefs must have one value per transition")
    if np.any((beliefs < -1.0) | (beliefs > 1.0)):
        raise ValueError("signed_beliefs must lie in [-1, 1]")
    if parameter.shape != (n_units,):
        raise ValueError("axis must have one parameter per unit")
    if weights.shape != (n_units, n_units):
        raise ValueError("recurrent_weights must have shape [unit, unit]")
    if step_fraction.shape != (n_units,):
        raise ValueError("dt_over_tau must have one value per unit")
    if np.any((step_fraction <= 0.0) | (step_fraction > 1.0)):
        raise ValueError("dt_over_tau values must lie in (0, 1]")
    if gain_application not in {"rate", "drive"}:
        raise ValueError("gain_application must be 'rate' or 'drive'")
    lower = _finite_scalar(gain_min, name="gain_min")
    upper = _finite_scalar(gain_max, name="gain_max")
    if lower <= 0.0 or not lower <= 1.0 <= upper or lower >= upper:
        raise ValueError(
            "gain bounds must be positive, ordered, and contain neutral gain one"
        )
    if not isinstance(store_augmented_partials, (bool, np.bool_)):
        raise TypeError("store_augmented_partials must be boolean")
    if initial_state is None:
        current_state = np.zeros(n_units, dtype=np.float64)
    else:
        current_state = _real_array(
            initial_state,
            name="initial_state",
            ndim=1,
        ).copy()
        if current_state.shape != (n_units,):
            raise ValueError("initial_state must have one value per unit")
    if initial_rates is None:
        current_rates, _ = _activation_and_derivative(
            current_state,
            activation=activation,
        )
    else:
        current_rates = _real_array(
            initial_rates,
            name="initial_rates",
            ndim=1,
        ).copy()
        if current_rates.shape != (n_units,):
            raise ValueError("initial_rates must have one value per unit")

    states = np.empty((n_steps + 1, n_units), dtype=np.float64)
    rates = np.empty_like(states)
    gains = np.empty((n_steps, n_units), dtype=np.float64)
    state_sensitivities = np.zeros(
        (n_steps + 1, n_units, n_units),
        dtype=np.float64,
    )
    rate_sensitivities = np.zeros_like(state_sensitivities)
    jacobian_diagonal = np.empty((n_steps, 2 * n_units), dtype=np.float64)
    local_jacobian_blocks = np.empty(
        (n_steps, n_units, 2, 2), dtype=np.float64
    )
    state_axis_direct = np.empty((n_steps, n_units), dtype=np.float64)
    rate_axis_direct = np.empty((n_steps, n_units), dtype=np.float64)
    if bool(store_augmented_partials):
        augmented_jacobians: FloatArray | None = np.empty(
            (n_steps, 2 * n_units, 2 * n_units),
            dtype=np.float64,
        )
        control_jacobians: FloatArray | None = np.empty(
            (n_steps, 2 * n_units, n_units),
            dtype=np.float64,
        )
    else:
        augmented_jacobians = None
        control_jacobians = None
    saturation = np.empty((n_steps, n_units), dtype=bool)
    states[0] = current_state
    rates[0] = current_rates
    state_decay = 1.0 - step_fraction
    unit_indices = np.arange(n_units)
    for time in range(n_steps):
        raw_gain = 1.0 + beliefs[time] * parameter
        gain = np.clip(raw_gain, lower, upper)
        unsaturated = (raw_gain > lower) & (raw_gain < upper)
        gain_derivative = beliefs[time] * unsaturated.astype(np.float64)
        total_drive = weights @ current_rates + drives[time]
        if gain_application == "rate":
            next_state = current_state + step_fraction * (
                -current_state + total_drive
            )
            next_rates, activation_derivative = _activation_and_derivative(
                gain * next_state,
                activation=activation,
            )
            state_rate_jacobian = step_fraction[:, None] * weights
            rate_from_next_state = activation_derivative * gain
            state_direct = np.zeros(n_units, dtype=np.float64)
            rate_direct = (
                activation_derivative * next_state * gain_derivative
            )
        else:
            next_state = current_state + step_fraction * (
                -current_state + gain * total_drive
            )
            next_rates, activation_derivative = _activation_and_derivative(
                next_state,
                activation=activation,
            )
            state_rate_jacobian = (
                step_fraction * gain
            )[:, None] * weights
            rate_from_next_state = activation_derivative
            state_direct = (
                step_fraction * total_drive * gain_derivative
            )
            rate_direct = activation_derivative * state_direct
        current_state = next_state
        current_rates = next_rates
        states[time + 1] = current_state
        rates[time + 1] = current_rates
        gains[time] = gain
        saturation[time] = ~unsaturated

        if augmented_jacobians is not None and control_jacobians is not None:
            state_from_state = np.diag(state_decay)
            rate_from_state = rate_from_next_state[:, None] * state_from_state
            rate_from_rates = (
                rate_from_next_state[:, None] * state_rate_jacobian
            )
            augmented_jacobians[time] = np.block(
                [
                    [state_from_state, state_rate_jacobian],
                    [rate_from_state, rate_from_rates],
                ]
            )
            control_jacobians[time].fill(0.0)
            control_jacobians[
                time,
                unit_indices,
                unit_indices,
            ] = state_direct
            control_jacobians[
                time,
                n_units + unit_indices,
                unit_indices,
            ] = rate_direct

        next_state_sensitivity = (
            state_decay[:, None] * state_sensitivities[time]
            + state_rate_jacobian @ rate_sensitivities[time]
        )
        next_state_sensitivity[unit_indices, unit_indices] += state_direct
        next_rate_sensitivity = (
            rate_from_next_state[:, None] * next_state_sensitivity
        )
        if gain_application == "rate":
            next_rate_sensitivity[unit_indices, unit_indices] += rate_direct
        state_sensitivities[time + 1] = next_state_sensitivity
        rate_sensitivities[time + 1] = next_rate_sensitivity
        state_rate_diagonal = np.diag(state_rate_jacobian)
        local_jacobian_blocks[time, :, 0, 0] = state_decay
        local_jacobian_blocks[time, :, 0, 1] = state_rate_diagonal
        local_jacobian_blocks[time, :, 1, 0] = (
            rate_from_next_state * state_decay
        )
        local_jacobian_blocks[time, :, 1, 1] = (
            rate_from_next_state * state_rate_diagonal
        )
        jacobian_diagonal[time, :n_units] = state_decay
        jacobian_diagonal[time, n_units:] = (
            rate_from_next_state * state_rate_diagonal
        )
        state_axis_direct[time] = state_direct
        rate_axis_direct[time] = rate_direct

    augmented = (
        np.concatenate(
            (state_sensitivities, rate_sensitivities),
            axis=1,
        )
        if bool(store_augmented_partials)
        else None
    )
    frozen_saturation = np.array(saturation, dtype=bool, copy=True)
    frozen_saturation.setflags(write=False)
    return FrozenGainAxisTrajectory(
        states=_readonly(states),
        rates=_readonly(rates),
        gains=_readonly(gains),
        state_sensitivities=_readonly(state_sensitivities),
        rate_sensitivities=_readonly(rate_sensitivities),
        augmented_sensitivities=(
            None if augmented is None else _readonly(augmented)
        ),
        jacobian_diagonal=_readonly(jacobian_diagonal),
        local_jacobian_blocks=_readonly(local_jacobian_blocks),
        state_axis_direct_derivatives=_readonly(state_axis_direct),
        rate_axis_direct_derivatives=_readonly(rate_axis_direct),
        augmented_state_jacobians=(
            None
            if augmented_jacobians is None
            else _readonly(augmented_jacobians)
        ),
        augmented_control_jacobians=(
            None
            if control_jacobians is None
            else _readonly(control_jacobians)
        ),
        saturation_mask=frozen_saturation,
        n_steps=int(n_steps),
        n_units=int(n_units),
        control_dim=int(n_units),
        stored_augmented_partials=bool(store_augmented_partials),
        gain_application=gain_application,
    )


@dataclass(frozen=True, slots=True)
class EpochMeanReadoutGradient:
    """Axis gradient of an epoch-mean linear readout score."""

    score_gradient: FloatArray
    readout_time_weights: FloatArray
    epoch_names: tuple[str, ...]
    epoch_counts: NDArray[np.int64]
    n_steps: int
    control_dim: int


def epoch_mean_readout_gradient(
    rate_sensitivities: ArrayLike,
    epoch_labels: ArrayLike,
    readout_coefficients: ArrayLike,
    *,
    epoch_names: tuple[str, ...] = ("sensory", "delay", "response"),
) -> EpochMeanReadoutGradient:
    """Differentiate a concatenated epoch-mean linear readout score.

    ``rate_sensitivities`` may include the initial state
    ``[time + 1, unit, parameter]`` or contain post-transition rates only
    ``[time, unit, parameter]``.  Its time dimension determines which form is
    used.  ``readout_coefficients`` has one ``[unit]`` row per named epoch and
    should already include any frozen readout normalization factor.
    """

    sensitivity = _real_array(
        rate_sensitivities,
        name="rate_sensitivities",
        ndim=3,
    )
    raw_labels = np.asarray(epoch_labels)
    if raw_labels.ndim != 1 or raw_labels.size == 0:
        raise ValueError("epoch_labels must be a non-empty vector")
    labels = np.asarray(raw_labels, dtype="U32")
    coefficients = _real_array(
        readout_coefficients,
        name="readout_coefficients",
        ndim=2,
    )
    if (
        not isinstance(epoch_names, tuple)
        or not epoch_names
        or any(not isinstance(name, str) or not name for name in epoch_names)
        or len(set(epoch_names)) != len(epoch_names)
    ):
        raise ValueError("epoch_names must contain unique non-empty strings")
    n_steps = int(labels.size)
    if sensitivity.shape[0] == n_steps + 1:
        post_transition = sensitivity[1:]
    elif sensitivity.shape[0] == n_steps:
        post_transition = sensitivity
    else:
        raise ValueError(
            "rate_sensitivities time must equal epoch count or epoch count plus one"
        )
    n_units = int(post_transition.shape[1])
    if coefficients.shape != (len(epoch_names), n_units):
        raise ValueError(
            "readout_coefficients must have shape [named_epoch, unit]"
        )

    time_weights = np.zeros((n_steps, n_units), dtype=np.float64)
    counts = np.empty(len(epoch_names), dtype=np.int64)
    for index, name in enumerate(epoch_names):
        selected = labels == name
        count = int(np.count_nonzero(selected))
        if count < 1:
            raise ValueError(f"epoch_labels contain no samples for {name}")
        counts[index] = count
        time_weights[selected] = coefficients[index] / count
    gradient = np.einsum(
        "tu,tup->p",
        time_weights,
        post_transition,
        optimize=True,
    )
    frozen_counts = np.array(counts, dtype=np.int64, copy=True)
    frozen_counts.setflags(write=False)
    return EpochMeanReadoutGradient(
        score_gradient=_readonly(gradient),
        readout_time_weights=_readonly(time_weights),
        epoch_names=epoch_names,
        epoch_counts=frozen_counts,
        n_steps=n_steps,
        control_dim=int(post_transition.shape[2]),
    )


__all__ = [
    "ForwardGradient",
    "ForwardSensitivityTrajectory",
    "EpochMeanReadoutGradient",
    "FrozenGainAxisStepPartials",
    "FrozenGainAxisTrajectory",
    "epoch_mean_readout_gradient",
    "exact_forward_gradient",
    "exact_forward_sensitivities",
    "frozen_gain_axis_step_partials",
    "simulate_frozen_gain_axis_trajectory",
]
