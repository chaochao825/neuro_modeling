"""Dale-constrained excitatory/inhibitory rate network.

The recurrent convention throughout this project is ``weight[post, pre]``.
Consequently Dale's law constrains outgoing *columns*: excitatory columns are
non-negative and inhibitory columns are non-positive.  The background,
task-plastic, and homeostatic components are stored separately so analyses do
not confuse a low-rank task update with the full recurrent matrix.

This module intentionally uses NumPy only.  Learning is performed by explicit
local updates supplied by :mod:`src.plasticity`, never by gradient descent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Array = np.ndarray
ActivationName = Literal["tanh", "rectified_tanh"]


def _finite_vector(value: Array, *, name: str, length: int) -> Array:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def dale_project(weights: Array, connectivity_mask: Array, presynaptic_signs: Array) -> Array:
    """Project a recurrent matrix onto its sparse Dale-constrained set."""

    matrix = np.asarray(weights, dtype=float)
    raw_mask = np.asarray(connectivity_mask)
    signs = np.asarray(presynaptic_signs, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("weights must be a square matrix")
    if raw_mask.shape != matrix.shape:
        raise ValueError("connectivity_mask must match weights")
    if not np.all(np.isin(raw_mask, (False, True, 0, 1))):
        raise ValueError("connectivity_mask must be boolean or binary")
    mask = raw_mask.astype(bool)
    if signs.shape != (matrix.shape[1],) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("presynaptic_signs must contain one +1/-1 value per column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weights must contain only finite values")

    projected = np.where(mask, matrix, 0.0)
    excitatory = signs > 0
    inhibitory = ~excitatory
    projected[:, excitatory] = np.maximum(projected[:, excitatory], 0.0)
    projected[:, inhibitory] = np.minimum(projected[:, inhibitory], 0.0)
    return projected


@dataclass(frozen=True)
class EIRateState:
    """Dynamical state for one population vector."""

    x: Array
    rates: Array


@dataclass(frozen=True)
class EIRateStep:
    """One Euler step and its decomposed recurrent drive."""

    state: EIRateState
    recurrent_drive: Array
    input_drive: Array


@dataclass(frozen=True)
class EIRateTrajectory:
    """Time-major state and rate arrays returned by :meth:`EIRateNetwork.run`."""

    x: Array
    rates: Array


@dataclass(frozen=True)
class AppliedWeightUpdate:
    """Auditable local change, fan-in correction, and their total effect."""

    local_update: Array
    normalization_correction: Array
    total_update: Array
    local_l1_cost: float
    normalization_l1_cost: float
    total_l1_cost: float


class EIRateNetwork:
    """Sparse E/I rate network with explicit local weight components.

    Parameters are expressed in a single time unit (typically milliseconds).
    ``rectified_tanh`` is the default because inhibitory homeostasis consumes
    non-negative firing rates.  ``tanh`` remains available for signed-state
    controls.
    """

    def __init__(
        self,
        n_units: int,
        *,
        n_inputs: int = 0,
        excitatory_fraction: float = 0.8,
        connection_probability: float = 0.1,
        tau_e: float = 20.0,
        tau_i: float = 10.0,
        dt: float = 1.0,
        bulk_gain: float = 1.0,
        inhibitory_gain: float = 1.0,
        input_scale: float = 1.0,
        activation: ActivationName = "rectified_tanh",
        allow_self_connections: bool = False,
        normalize_fan_in_after_update: bool = True,
        seed: int = 0,
    ) -> None:
        if not isinstance(n_units, (int, np.integer)) or isinstance(n_units, bool) or n_units < 2:
            raise ValueError("n_units must be an integer >= 2")
        if not isinstance(n_inputs, (int, np.integer)) or isinstance(n_inputs, bool) or n_inputs < 0:
            raise ValueError("n_inputs must be a non-negative integer")
        if not 0.0 < excitatory_fraction < 1.0:
            raise ValueError("excitatory_fraction must be in (0, 1)")
        if not 0.0 <= connection_probability <= 1.0:
            raise ValueError("connection_probability must be in [0, 1]")
        if not np.isfinite(tau_e) or tau_e <= 0.0:
            raise ValueError("tau_e must be positive and finite")
        if not np.isfinite(tau_i) or tau_i <= 0.0:
            raise ValueError("tau_i must be positive and finite")
        if not np.isfinite(dt) or dt <= 0.0 or dt > min(tau_e, tau_i):
            raise ValueError("dt must be positive and no larger than either time constant")
        if not np.isfinite(bulk_gain) or bulk_gain < 0.0:
            raise ValueError("bulk_gain must be non-negative and finite")
        if not np.isfinite(inhibitory_gain) or inhibitory_gain <= 0.0:
            raise ValueError("inhibitory_gain must be positive and finite")
        if not np.isfinite(input_scale) or input_scale < 0.0:
            raise ValueError("input_scale must be non-negative and finite")
        if activation not in ("tanh", "rectified_tanh"):
            raise ValueError("activation must be 'tanh' or 'rectified_tanh'")
        if not isinstance(allow_self_connections, (bool, np.bool_)):
            raise TypeError("allow_self_connections must be boolean")
        if not isinstance(normalize_fan_in_after_update, (bool, np.bool_)):
            raise TypeError("normalize_fan_in_after_update must be boolean")
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        self.n_units = int(n_units)
        self.n_inputs = int(n_inputs)
        self.excitatory_fraction = float(excitatory_fraction)
        self.connection_probability = float(connection_probability)
        self.tau_e = float(tau_e)
        self.tau_i = float(tau_i)
        self.dt = float(dt)
        self.activation_name = activation
        self.inhibitory_gain = float(inhibitory_gain)
        self.normalize_fan_in_after_update = bool(normalize_fan_in_after_update)
        self.seed = int(seed)

        # Rounding gives the nearest realizable fraction while guaranteeing at
        # least one unit of each type for every valid population size.
        self.n_excitatory = int(np.clip(round(self.n_units * excitatory_fraction), 1, self.n_units - 1))
        self.n_inhibitory = self.n_units - self.n_excitatory
        self.excitatory_mask = np.arange(self.n_units) < self.n_excitatory
        self.inhibitory_mask = ~self.excitatory_mask
        self.presynaptic_signs = np.where(self.excitatory_mask, 1.0, -1.0)
        self.time_constants = np.where(self.excitatory_mask, self.tau_e, self.tau_i)

        rng = np.random.default_rng(self.seed)
        self.connectivity_mask = rng.random((self.n_units, self.n_units)) < connection_probability
        if not allow_self_connections:
            np.fill_diagonal(self.connectivity_mask, False)

        expected_fan_in = max(1.0, connection_probability * (self.n_units - 1))
        magnitudes = rng.exponential(
            scale=bulk_gain / np.sqrt(expected_fan_in),
            size=(self.n_units, self.n_units),
        )
        column_gain = np.where(self.excitatory_mask, 1.0, self.inhibitory_gain)
        signed = magnitudes * column_gain[np.newaxis, :] * self.presynaptic_signs[np.newaxis, :]
        self._W_bulk = dale_project(signed, self.connectivity_mask, self.presynaptic_signs)
        self._W_task = np.zeros_like(self._W_bulk)
        self._W_homeo = np.zeros_like(self._W_bulk)
        # Fan-in normalization is a distinct homeostatic operation.  Keeping
        # it separate prevents dense row rescaling from being misreported as a
        # low-rank task update or an I-to-E inhibitory update.
        self._W_normalization = np.zeros_like(self._W_bulk)
        # Dynamics read this synchronized effective matrix on every time
        # step.  Rebuilding the sum of four dense N x N components inside
        # ``step`` is mathematically redundant and dominates N=512 runs.
        # Component arrays remain separate for the required plasticity audit;
        # sanctioned update methods increment this cache atomically.
        self._effective_recurrent = self._W_bulk.copy()
        self._fan_in_l1_target = np.sum(np.abs(self._W_bulk), axis=1)
        zero_update = np.zeros_like(self._W_bulk)
        self.last_task_application = AppliedWeightUpdate(
            zero_update.copy(), zero_update.copy(), zero_update.copy(), 0.0, 0.0, 0.0
        )
        self.last_homeostatic_application = AppliedWeightUpdate(
            zero_update.copy(), zero_update.copy(), zero_update.copy(), 0.0, 0.0, 0.0
        )

        if self.n_inputs:
            self.input_weights = rng.normal(
                loc=0.0,
                scale=input_scale / np.sqrt(self.n_inputs),
                size=(self.n_units, self.n_inputs),
            )
        else:
            self.input_weights = np.zeros((self.n_units, 0), dtype=float)

    @property
    def recurrent_weights(self) -> Array:
        """Return a copy of the current effective recurrent matrix."""

        return self._effective_recurrent.copy()

    @staticmethod
    def _readonly_component(component: Array) -> Array:
        snapshot = component.copy()
        snapshot.setflags(write=False)
        return snapshot

    @property
    def W_bulk(self) -> Array:
        """Read-only snapshot of the fixed background component."""

        return self._readonly_component(self._W_bulk)

    @property
    def W_task(self) -> Array:
        """Read-only snapshot of the task-plastic component."""

        return self._readonly_component(self._W_task)

    @property
    def W_homeo(self) -> Array:
        """Read-only snapshot of the inhibitory-homeostatic component."""

        return self._readonly_component(self._W_homeo)

    @property
    def W_normalization(self) -> Array:
        """Read-only snapshot of the fan-in-normalization component."""

        return self._readonly_component(self._W_normalization)

    def _task_weights_for_learning(self) -> Array:
        """Internal zero-copy read used by the local plasticity rule."""

        return self._W_task

    def _effective_weights_for_learning(self) -> Array:
        """Internal zero-copy read guarded by the network update invariant."""

        return self._effective_recurrent

    @property
    def fan_in_l1_target(self) -> Array:
        """Read-only snapshot of the fixed row-wise normalization target."""

        return self._readonly_component(self._fan_in_l1_target)

    @property
    def weights(self) -> Array:
        """Alias for :attr:`recurrent_weights`."""

        return self.recurrent_weights

    def _activate(self, x: Array, gain: Array) -> Array:
        activated = np.tanh(gain * x)
        if self.activation_name == "rectified_tanh":
            activated = np.maximum(activated, 0.0)
        return activated

    def initial_state(self, x: Array | None = None, *, gain: Array | float = 1.0) -> EIRateState:
        """Create a validated initial state without hidden random sampling."""

        state_x = np.zeros(self.n_units, dtype=float) if x is None else _finite_vector(
            x, name="x", length=self.n_units
        ).copy()
        gain_vector = self._gain_vector(gain)
        return EIRateState(x=state_x, rates=self._activate(state_x, gain_vector))

    def _gain_vector(self, gain: Array | float) -> Array:
        if np.isscalar(gain):
            scalar = float(gain)
            if not np.isfinite(scalar) or scalar <= 0.0:
                raise ValueError("gain must be positive and finite")
            return np.full(self.n_units, scalar, dtype=float)
        vector = _finite_vector(np.asarray(gain), name="gain", length=self.n_units)
        if np.any(vector <= 0.0):
            raise ValueError("all gain values must be positive")
        return vector

    def step(
        self,
        state: EIRateState,
        input_t: Array | None = None,
        *,
        gain: Array | float = 1.0,
        noise: Array | None = None,
    ) -> EIRateStep:
        """Advance the network by one explicit Euler step."""

        x = _finite_vector(state.x, name="state.x", length=self.n_units)
        rates = _finite_vector(state.rates, name="state.rates", length=self.n_units)
        if self.n_inputs:
            if input_t is None:
                raise ValueError("input_t is required when n_inputs > 0")
            input_vector = _finite_vector(input_t, name="input_t", length=self.n_inputs)
        else:
            if input_t is not None and np.asarray(input_t).size:
                raise ValueError("input_t must be None or empty when n_inputs == 0")
            input_vector = np.zeros(0, dtype=float)
        noise_vector = (
            np.zeros(self.n_units, dtype=float)
            if noise is None
            else _finite_vector(noise, name="noise", length=self.n_units)
        )
        gain_vector = self._gain_vector(gain)

        recurrent_drive = self._effective_recurrent @ rates
        input_drive = self.input_weights @ input_vector
        dx = (-x + recurrent_drive + input_drive + noise_vector) / self.time_constants
        next_x = x + self.dt * dx
        next_rates = self._activate(next_x, gain_vector)
        next_state = EIRateState(x=next_x, rates=next_rates)
        return EIRateStep(
            state=next_state,
            recurrent_drive=recurrent_drive,
            input_drive=input_drive,
        )

    def run(
        self,
        inputs: Array,
        *,
        initial_state: EIRateState | None = None,
        gains: Array | float = 1.0,
    ) -> EIRateTrajectory:
        """Run a deterministic time-major input sequence."""

        input_array = np.asarray(inputs, dtype=float)
        if input_array.ndim != 2 or input_array.shape[1] != self.n_inputs:
            raise ValueError(f"inputs must have shape (time, {self.n_inputs})")
        if not np.all(np.isfinite(input_array)):
            raise ValueError("inputs must contain only finite values")
        n_steps = input_array.shape[0]
        if np.isscalar(gains):
            gain_array = np.full((n_steps, self.n_units), float(gains), dtype=float)
        else:
            raw_gains = np.asarray(gains, dtype=float)
            if raw_gains.shape == (self.n_units,):
                gain_array = np.repeat(raw_gains[np.newaxis, :], n_steps, axis=0)
            elif raw_gains.shape == (n_steps, self.n_units):
                gain_array = raw_gains
            else:
                raise ValueError("gains must be scalar, (n_units,), or (time, n_units)")
        if not np.all(np.isfinite(gain_array)) or np.any(gain_array <= 0.0):
            raise ValueError("all gains must be positive and finite")

        state = self.initial_state() if initial_state is None else EIRateState(
            x=_finite_vector(initial_state.x, name="initial_state.x", length=self.n_units).copy(),
            rates=_finite_vector(
                initial_state.rates, name="initial_state.rates", length=self.n_units
            ).copy(),
        )
        x_history = np.empty((n_steps + 1, self.n_units), dtype=float)
        rate_history = np.empty_like(x_history)
        x_history[0] = state.x
        rate_history[0] = state.rates
        for t in range(n_steps):
            state = self.step(state, input_array[t], gain=gain_array[t]).state
            x_history[t + 1] = state.x
            rate_history[t + 1] = state.rates
        return EIRateTrajectory(x=x_history, rates=rate_history)

    def _apply_component_update(
        self, component: Array, update: Array, *, name: str
    ) -> AppliedWeightUpdate:
        proposed = np.asarray(update, dtype=float)
        if proposed.shape != (self.n_units, self.n_units):
            raise ValueError(f"{name} must have shape ({self.n_units}, {self.n_units})")
        if not np.all(np.isfinite(proposed)):
            raise ValueError(f"{name} must contain only finite values")

        before = self._effective_recurrent.copy()
        candidate = before + np.where(self.connectivity_mask, proposed, 0.0)
        locally_projected = dale_project(
            candidate, self.connectivity_mask, self.presynaptic_signs
        )
        local_update = locally_projected - before
        return self._apply_projected_component_update(component, local_update)

    def _apply_projected_component_update(
        self, component: Array, local_update: Array
    ) -> AppliedWeightUpdate:
        """Apply an internally proven sparse/Dale update without re-projecting."""

        locally_projected = self._effective_recurrent + local_update
        normalization_correction = np.zeros_like(local_update)
        if self.normalize_fan_in_after_update:
            normalized = self._normalize_projected_fan_in(locally_projected)
            normalization_correction = normalized - locally_projected
        total_update = local_update + normalization_correction
        result = AppliedWeightUpdate(
            local_update=local_update.copy(),
            normalization_correction=normalization_correction.copy(),
            total_update=total_update.copy(),
            local_l1_cost=float(np.sum(np.abs(local_update))),
            normalization_l1_cost=float(np.sum(np.abs(normalization_correction))),
            total_l1_cost=float(np.sum(np.abs(total_update))),
        )
        # Commit only after every shape-dependent calculation and result copy
        # succeeds, so a rejected update cannot split audit components from
        # the effective dynamics cache.
        component += local_update
        self._W_normalization += normalization_correction
        self._effective_recurrent += total_update
        return result

    def _normalize_projected_fan_in(self, weights: Array) -> Array:
        """Match each postsynaptic row's initial absolute incoming weight sum.

        Rows whose initial target or current norm is zero remain zero.  Positive
        row scaling preserves the already-proven sparse mask and Dale signs.
        """

        matrix = weights
        current = np.sum(np.abs(matrix), axis=1)
        # A zero reference norm carries no normalization scale.  Leave that
        # row untouched instead of silently erasing learning (e.g. bulk_gain=0).
        normalized = matrix.copy()
        valid = (self._fan_in_l1_target > 0.0) & (current > 0.0)
        scale = self._fan_in_l1_target[valid] / current[valid]
        normalized[valid] *= scale[:, np.newaxis]
        # A sufficiently large depressing proposal can zero a whole row, for
        # which multiplicative normalization is undefined.  Reject that row's
        # collapse by restoring its fixed sparse Dale-compatible bulk template.
        collapsed = (self._fan_in_l1_target > 0.0) & (current == 0.0)
        normalized[collapsed] = self._W_bulk[collapsed]
        return normalized

    @property
    def fan_in_l1_norms(self) -> Array:
        """Absolute incoming recurrent weight sum for every postsynaptic unit."""

        return np.sum(np.abs(self.recurrent_weights), axis=1)

    def apply_task_update(self, update: Array) -> AppliedWeightUpdate:
        """Apply task plasticity and return local/normalization/total stages."""

        result = self._apply_component_update(self._W_task, update, name="task update")
        self.last_task_application = result
        return result

    def _apply_projected_task_update(self, update: Array) -> AppliedWeightUpdate:
        """Trusted local-training path for a preprojected task update."""

        result = self._apply_projected_component_update(self._W_task, update)
        self.last_task_application = result
        return result

    def apply_homeostatic_update(self, update: Array) -> AppliedWeightUpdate:
        """Apply inhibitory homeostasis and return all transformation stages."""

        result = self._apply_component_update(
            self._W_homeo, update, name="homeostatic update"
        )
        self.last_homeostatic_application = result
        return result

    def _apply_projected_homeostatic_update(
        self, update: Array
    ) -> AppliedWeightUpdate:
        """Trusted local-training path for a preprojected homeostatic update."""

        result = self._apply_projected_component_update(self._W_homeo, update)
        self.last_homeostatic_application = result
        return result

    def validate_dale(self, *, atol: float = 1e-12) -> bool:
        """Return whether sparsity and all outgoing-column signs are valid."""

        weights = self.recurrent_weights
        e_valid = np.all(weights[:, self.excitatory_mask] >= -atol)
        i_valid = np.all(weights[:, self.inhibitory_mask] <= atol)
        sparse_valid = np.all(np.abs(weights[~self.connectivity_mask]) <= atol)
        return bool(e_valid and i_valid and sparse_valid)
