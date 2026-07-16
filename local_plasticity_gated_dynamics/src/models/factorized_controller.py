"""Factorized low-dimensional control of a frozen high-rank recurrent base.

The three actuator families use the same control dimension:

``routing``
    Multiplicative input-pathway scales.
``gain``
    Multiplicative population gain on the routed input current.
``low_rank``
    A rank-limited recurrent correction.

The combined ``rgl`` condition activates all three families.  The recurrent
base and every actuator basis are copied into read-only arrays at
construction, so an experiment can vary only the low-dimensional controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ActivationName = Literal["tanh", "identity", "relu"]


def _finite_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != ndim or array.size == 0 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


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


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class ActuatorMode(str, Enum):
    """Canonical Exp24 actuator conditions."""

    FROZEN = "frozen"
    ROUTING = "routing"
    GAIN = "gain"
    LOW_RANK = "low_rank"
    RGL = "rgl"

    @classmethod
    def coerce(cls, value: ActuatorMode | str) -> ActuatorMode:
        """Validate a mode while accepting a small legacy spelling alias."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("mode must be an ActuatorMode or string")
        canonical = "low_rank" if value == "recurrent" else value
        try:
            return cls(canonical)
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"mode must be one of: {choices}") from error

    @property
    def uses_routing(self) -> bool:
        return self in {ActuatorMode.ROUTING, ActuatorMode.RGL}

    @property
    def uses_gain(self) -> bool:
        return self in {ActuatorMode.GAIN, ActuatorMode.RGL}

    @property
    def uses_low_rank(self) -> bool:
        return self in {ActuatorMode.LOW_RANK, ActuatorMode.RGL}


@dataclass(frozen=True)
class FactorizedControllerConfig:
    """Static bounds and activation for a factorized controller."""

    control_dim: int = 2
    activation: ActivationName = "tanh"
    gain_min: float = 0.1
    gain_max: float = 2.0
    routing_min: float = 0.0
    routing_max: float = 2.0
    require_high_rank_base: bool = True

    def __post_init__(self) -> None:
        control_dim = _positive_integer(self.control_dim, name="control_dim")
        if control_dim != 2:
            raise ValueError("Exp24 requires the shared control_dim to equal 2")
        if self.activation not in {"tanh", "identity", "relu"}:
            raise ValueError("activation must be 'tanh', 'identity', or 'relu'")
        gain_min = _finite_scalar(self.gain_min, name="gain_min")
        gain_max = _finite_scalar(self.gain_max, name="gain_max")
        routing_min = _finite_scalar(self.routing_min, name="routing_min")
        routing_max = _finite_scalar(self.routing_max, name="routing_max")
        if not 0.0 < gain_min <= 1.0 <= gain_max:
            raise ValueError("gain bounds must satisfy 0 < min <= 1 <= max")
        if not 0.0 <= routing_min <= 1.0 <= routing_max:
            raise ValueError("routing bounds must satisfy 0 <= min <= 1 <= max")
        if not isinstance(self.require_high_rank_base, (bool, np.bool_)):
            raise TypeError("require_high_rank_base must be boolean")


@dataclass(frozen=True)
class FactorizedStep:
    """One deterministic state transition and its actuator audit values."""

    next_state: FloatArray
    rate: FloatArray
    preactivation: FloatArray
    effective_recurrent: FloatArray
    gain: FloatArray
    routing_scale: FloatArray
    recurrent_update: FloatArray
    input_current: FloatArray
    recurrent_current: FloatArray
    synaptic_event_proxy: float


@dataclass(frozen=True)
class FactorizedRollout:
    """Full controlled trajectory with enough state for Exp24 diagnostics."""

    mode: ActuatorMode
    states: FloatArray
    rates: FloatArray
    preactivations: FloatArray
    effective_recurrent_history: FloatArray
    recurrent_update_history: FloatArray
    gain_history: FloatArray
    routing_scale_history: FloatArray
    routing_controls: FloatArray
    gain_controls: FloatArray
    low_rank_controls: FloatArray
    synaptic_event_proxy_by_step: FloatArray


@dataclass(frozen=True)
class FactorizedRolloutAudit:
    """Functional and structural receipt for one actuator rollout."""

    mode: str
    control_dim: int
    routing_control_dim: int
    gain_control_dim: int
    low_rank_control_dim: int
    base_recurrent_rank: int
    low_rank_update_rank: int
    functional_displacement: float | None
    mean_gain: float
    max_gain: float
    mean_routing_scale: float
    max_routing_scale: float
    synaptic_event_proxy: float
    control_cost: float
    n_steps: int


@dataclass(frozen=True)
class FactorizedController:
    """Frozen high-rank base with routing, gain, and rank-two actuators.

    Zero-valued control is neutral for all actuator families.  Routing and
    population controls are mapped to multiplicative scales around one:

    ``routing_scale = clip(1 + routing_axes @ s)``
    ``gain = clip(1 + gain_axes @ g)``

    The recurrent correction is
    ``left @ diag(h) @ right.T`` and therefore has rank at most
    ``control_dim``.
    """

    base_recurrent: FloatArray
    input_weights: FloatArray
    routing_axes: FloatArray
    gain_axes: FloatArray
    low_rank_left: FloatArray
    low_rank_right: FloatArray
    bias: FloatArray
    config: FactorizedControllerConfig = FactorizedControllerConfig()

    def __post_init__(self) -> None:
        base = _finite_array(self.base_recurrent, name="base_recurrent", ndim=2)
        if base.shape[0] != base.shape[1]:
            raise ValueError("base_recurrent must be square")
        n_units = base.shape[0]
        input_weights = _finite_array(self.input_weights, name="input_weights", ndim=2)
        if input_weights.shape[0] != n_units:
            raise ValueError("input_weights row count must equal the state dimension")
        routing_axes = _finite_array(self.routing_axes, name="routing_axes", ndim=2)
        gain_axes = _finite_array(self.gain_axes, name="gain_axes", ndim=2)
        low_rank_left = _finite_array(self.low_rank_left, name="low_rank_left", ndim=2)
        low_rank_right = _finite_array(
            self.low_rank_right, name="low_rank_right", ndim=2
        )
        bias = _finite_array(self.bias, name="bias", ndim=1)
        control_dim = self.config.control_dim
        if routing_axes.shape != (input_weights.shape[1], control_dim):
            raise ValueError("routing_axes must have shape [input_dim, control_dim]")
        if gain_axes.shape != (n_units, control_dim):
            raise ValueError("gain_axes must have shape [n_units, control_dim]")
        if low_rank_left.shape != (n_units, control_dim):
            raise ValueError("low_rank_left must have shape [n_units, control_dim]")
        if low_rank_right.shape != (n_units, control_dim):
            raise ValueError("low_rank_right must have shape [n_units, control_dim]")
        if bias.shape != (n_units,):
            raise ValueError("bias must have shape [n_units]")
        base_rank = int(np.linalg.matrix_rank(base))
        if self.config.require_high_rank_base and base_rank <= control_dim:
            raise ValueError(
                "base_recurrent rank must exceed the shared control dimension"
            )

        for name, value in (
            ("base_recurrent", base),
            ("input_weights", input_weights),
            ("routing_axes", routing_axes),
            ("gain_axes", gain_axes),
            ("low_rank_left", low_rank_left),
            ("low_rank_right", low_rank_right),
            ("bias", bias),
        ):
            object.__setattr__(self, name, _readonly(value))

    @property
    def n_units(self) -> int:
        return int(self.base_recurrent.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.input_weights.shape[1])

    @property
    def control_dim(self) -> int:
        return int(self.config.control_dim)

    @property
    def base_recurrent_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.base_recurrent))

    @classmethod
    def random(
        cls,
        *,
        n_units: int,
        input_dim: int,
        seed: int,
        spectral_radius: float = 0.8,
        input_scale: float = 0.5,
        actuator_scale: float = 0.25,
        config: FactorizedControllerConfig | None = None,
    ) -> FactorizedController:
        """Create a deterministic dense high-rank benchmark instance."""

        n_units = _positive_integer(n_units, name="n_units")
        input_dim = _positive_integer(input_dim, name="input_dim")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")
        spectral_radius = _finite_scalar(spectral_radius, name="spectral_radius")
        input_scale = _finite_scalar(input_scale, name="input_scale")
        actuator_scale = _finite_scalar(actuator_scale, name="actuator_scale")
        if spectral_radius <= 0.0 or input_scale <= 0.0 or actuator_scale <= 0.0:
            raise ValueError("all random initialization scales must be positive")
        resolved_config = config or FactorizedControllerConfig()
        if n_units <= resolved_config.control_dim:
            raise ValueError("n_units must exceed control_dim for a high-rank base")
        rng = np.random.default_rng(int(seed))
        base = rng.normal(size=(n_units, n_units)) / np.sqrt(n_units)
        radius = float(np.max(np.abs(np.linalg.eigvals(base))))
        if radius == 0.0:
            raise RuntimeError("random base recurrent matrix is degenerate")
        base *= spectral_radius / radius
        input_weights = (
            input_scale * rng.normal(size=(n_units, input_dim)) / np.sqrt(input_dim)
        )
        routing_axes = actuator_scale * rng.normal(
            size=(input_dim, resolved_config.control_dim)
        )
        gain_axes = actuator_scale * rng.normal(
            size=(n_units, resolved_config.control_dim)
        )
        left = rng.normal(size=(n_units, resolved_config.control_dim))
        right = rng.normal(size=(n_units, resolved_config.control_dim))
        left /= np.maximum(np.linalg.norm(left, axis=0, keepdims=True), 1e-12)
        right /= np.maximum(np.linalg.norm(right, axis=0, keepdims=True), 1e-12)
        left *= actuator_scale
        return cls(
            base_recurrent=base,
            input_weights=input_weights,
            routing_axes=routing_axes,
            gain_axes=gain_axes,
            low_rank_left=left,
            low_rank_right=right,
            bias=np.zeros(n_units, dtype=np.float64),
            config=resolved_config,
        )

    def _activate(self, preactivation: FloatArray) -> FloatArray:
        if self.config.activation == "tanh":
            return np.tanh(preactivation)
        if self.config.activation == "relu":
            return np.maximum(preactivation, 0.0)
        return preactivation.copy()

    def _control_vector(self, value: ArrayLike | None, *, name: str) -> FloatArray:
        if value is None:
            return np.zeros(self.control_dim, dtype=np.float64)
        vector = _finite_array(value, name=name, ndim=1)
        if vector.shape != (self.control_dim,):
            raise ValueError(f"{name} must have shape [control_dim]")
        return vector

    def effective_recurrent(
        self,
        low_rank_control: ArrayLike | None = None,
        *,
        mode: ActuatorMode | str = ActuatorMode.LOW_RANK,
    ) -> FloatArray:
        """Return ``W0 + U diag(h) V.T`` for an active low-rank mode."""

        selected = ActuatorMode.coerce(mode)
        control = self._control_vector(low_rank_control, name="low_rank_control")
        if not selected.uses_low_rank:
            return self.base_recurrent.copy()
        update = (self.low_rank_left * control[np.newaxis, :]) @ (self.low_rank_right.T)
        return self.base_recurrent + update

    def step(
        self,
        state: ArrayLike,
        input_value: ArrayLike,
        *,
        mode: ActuatorMode | str,
        control: ArrayLike | None = None,
        routing_control: ArrayLike | None = None,
        gain_control: ArrayLike | None = None,
        low_rank_control: ArrayLike | None = None,
    ) -> FactorizedStep:
        """Advance one state transition using shared or actuator-specific control."""

        selected = ActuatorMode.coerce(mode)
        state_array = _finite_array(state, name="state", ndim=1)
        input_array = _finite_array(input_value, name="input_value", ndim=1)
        if state_array.shape != (self.n_units,):
            raise ValueError("state must have shape [n_units]")
        if input_array.shape != (self.input_dim,):
            raise ValueError("input_value must have shape [input_dim]")
        if control is not None and any(
            item is not None
            for item in (routing_control, gain_control, low_rank_control)
        ):
            raise ValueError(
                "control cannot be combined with actuator-specific controls"
            )
        if control is not None:
            shared = self._control_vector(control, name="control")
            routing = shared
            gain_control_array = shared
            low_rank = shared
        else:
            routing = self._control_vector(routing_control, name="routing_control")
            gain_control_array = self._control_vector(gain_control, name="gain_control")
            low_rank = self._control_vector(low_rank_control, name="low_rank_control")

        if selected.uses_routing:
            routing_scale = np.clip(
                1.0 + self.routing_axes @ routing,
                self.config.routing_min,
                self.config.routing_max,
            )
        else:
            routing_scale = np.ones(self.input_dim, dtype=np.float64)
        if selected.uses_gain:
            gain = np.clip(
                1.0 + self.gain_axes @ gain_control_array,
                self.config.gain_min,
                self.config.gain_max,
            )
        else:
            gain = np.ones(self.n_units, dtype=np.float64)
        if selected.uses_low_rank:
            recurrent_update = (
                self.low_rank_left * low_rank[np.newaxis, :]
            ) @ self.low_rank_right.T
        else:
            recurrent_update = np.zeros_like(self.base_recurrent)
        effective = self.base_recurrent + recurrent_update
        routed_input = routing_scale * input_array
        input_current = gain * (self.input_weights @ routed_input)
        recurrent_current = effective @ state_array
        preactivation = recurrent_current + input_current + self.bias
        rate = self._activate(preactivation)

        recurrent_events = float(np.sum(np.abs(effective) @ np.abs(state_array)))
        input_events = float(
            np.sum(
                np.abs(gain[:, np.newaxis] * self.input_weights)
                * np.abs(routed_input)[np.newaxis, :]
            )
        )
        return FactorizedStep(
            next_state=_readonly(rate),
            rate=_readonly(rate),
            preactivation=_readonly(preactivation),
            effective_recurrent=_readonly(effective),
            gain=_readonly(gain),
            routing_scale=_readonly(routing_scale),
            recurrent_update=_readonly(recurrent_update),
            input_current=_readonly(input_current),
            recurrent_current=_readonly(recurrent_current),
            synaptic_event_proxy=recurrent_events + input_events,
        )

    def _control_trajectory(
        self,
        value: ArrayLike | None,
        *,
        n_steps: int,
        name: str,
    ) -> FloatArray:
        if value is None:
            return np.zeros((n_steps, self.control_dim), dtype=np.float64)
        array = np.asarray(value)
        if array.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
            raise TypeError(f"{name} must be a real numeric array")
        array = np.asarray(array, dtype=np.float64)
        if array.ndim == 1:
            if array.shape != (self.control_dim,):
                raise ValueError(f"{name} vector must have shape [control_dim]")
            array = np.broadcast_to(array, (n_steps, self.control_dim)).copy()
        elif array.ndim == 2:
            if array.shape != (n_steps, self.control_dim):
                raise ValueError(f"{name} matrix must have shape [time, control_dim]")
            array = array.copy()
        else:
            raise ValueError(f"{name} must be one- or two-dimensional")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    def rollout(
        self,
        initial_state: ArrayLike,
        inputs: ArrayLike,
        *,
        mode: ActuatorMode | str,
        controls: ArrayLike | None = None,
        routing_controls: ArrayLike | None = None,
        gain_controls: ArrayLike | None = None,
        low_rank_controls: ArrayLike | None = None,
    ) -> FactorizedRollout:
        """Run a deterministic controlled trajectory.

        ``controls`` broadcasts one shared two-dimensional trajectory to every
        active actuator.  Use the three actuator-specific arguments when an
        oracle supplies different two-dimensional coordinates to R, G, and L.
        """

        selected = ActuatorMode.coerce(mode)
        initial = _finite_array(initial_state, name="initial_state", ndim=1)
        input_array = _finite_array(inputs, name="inputs", ndim=2)
        if initial.shape != (self.n_units,):
            raise ValueError("initial_state must have shape [n_units]")
        if input_array.shape[1] != self.input_dim:
            raise ValueError("inputs must have shape [time, input_dim]")
        n_steps = input_array.shape[0]
        if controls is not None and any(
            value is not None
            for value in (
                routing_controls,
                gain_controls,
                low_rank_controls,
            )
        ):
            raise ValueError(
                "controls cannot be combined with actuator-specific controls"
            )
        if controls is not None:
            shared = self._control_trajectory(
                controls, n_steps=n_steps, name="controls"
            )
            routing = shared.copy()
            gain = shared.copy()
            low_rank = shared.copy()
        else:
            routing = self._control_trajectory(
                routing_controls,
                n_steps=n_steps,
                name="routing_controls",
            )
            gain = self._control_trajectory(
                gain_controls, n_steps=n_steps, name="gain_controls"
            )
            low_rank = self._control_trajectory(
                low_rank_controls,
                n_steps=n_steps,
                name="low_rank_controls",
            )

        states = np.empty((n_steps + 1, self.n_units), dtype=np.float64)
        states[0] = initial
        rates = np.empty((n_steps, self.n_units), dtype=np.float64)
        preactivations = np.empty_like(rates)
        effective_history = np.empty(
            (n_steps, self.n_units, self.n_units), dtype=np.float64
        )
        update_history = np.empty_like(effective_history)
        gain_history = np.empty((n_steps, self.n_units), dtype=np.float64)
        routing_history = np.empty((n_steps, self.input_dim), dtype=np.float64)
        event_proxy = np.empty(n_steps, dtype=np.float64)
        for time_index in range(n_steps):
            result = self.step(
                states[time_index],
                input_array[time_index],
                mode=selected,
                routing_control=routing[time_index],
                gain_control=gain[time_index],
                low_rank_control=low_rank[time_index],
            )
            states[time_index + 1] = result.next_state
            rates[time_index] = result.rate
            preactivations[time_index] = result.preactivation
            effective_history[time_index] = result.effective_recurrent
            update_history[time_index] = result.recurrent_update
            gain_history[time_index] = result.gain
            routing_history[time_index] = result.routing_scale
            event_proxy[time_index] = result.synaptic_event_proxy
        return FactorizedRollout(
            mode=selected,
            states=_readonly(states),
            rates=_readonly(rates),
            preactivations=_readonly(preactivations),
            effective_recurrent_history=_readonly(effective_history),
            recurrent_update_history=_readonly(update_history),
            gain_history=_readonly(gain_history),
            routing_scale_history=_readonly(routing_history),
            routing_controls=_readonly(routing),
            gain_controls=_readonly(gain),
            low_rank_controls=_readonly(low_rank),
            synaptic_event_proxy_by_step=_readonly(event_proxy),
        )

    def audit_rollout(
        self,
        rollout: FactorizedRollout,
        *,
        frozen_states: ArrayLike | None = None,
    ) -> FactorizedRolloutAudit:
        """Summarize structural rank, functional displacement, and cost."""

        if not isinstance(rollout, FactorizedRollout):
            raise TypeError("rollout must be a FactorizedRollout")
        if rollout.states.shape[1] != self.n_units:
            raise ValueError("rollout state dimension differs from this controller")
        displacement: float | None = None
        if frozen_states is not None:
            frozen = _finite_array(frozen_states, name="frozen_states", ndim=2)
            if frozen.shape != rollout.states.shape:
                raise ValueError("frozen_states must match rollout.states")
            displacement = float(
                np.mean(
                    np.sum(
                        (rollout.states[1:] - frozen[1:]) ** 2,
                        axis=-1,
                    )
                )
            )
        update_ranks = [
            int(np.linalg.matrix_rank(update))
            for update in rollout.recurrent_update_history
        ]
        low_rank_update_rank = max(update_ranks, default=0)
        active_controls = []
        if rollout.mode.uses_routing:
            active_controls.append(rollout.routing_controls)
        if rollout.mode.uses_gain:
            active_controls.append(rollout.gain_controls)
        if rollout.mode.uses_low_rank:
            active_controls.append(rollout.low_rank_controls)
        control_cost = (
            float(
                np.mean(
                    np.sum(
                        np.concatenate(active_controls, axis=1) ** 2,
                        axis=1,
                    )
                )
            )
            if active_controls
            else 0.0
        )
        return FactorizedRolloutAudit(
            mode=rollout.mode.value,
            control_dim=self.control_dim,
            routing_control_dim=self.control_dim,
            gain_control_dim=self.control_dim,
            low_rank_control_dim=self.control_dim,
            base_recurrent_rank=self.base_recurrent_rank,
            low_rank_update_rank=low_rank_update_rank,
            functional_displacement=displacement,
            mean_gain=float(np.mean(rollout.gain_history)),
            max_gain=float(np.max(rollout.gain_history)),
            mean_routing_scale=float(np.mean(rollout.routing_scale_history)),
            max_routing_scale=float(np.max(rollout.routing_scale_history)),
            synaptic_event_proxy=float(np.mean(rollout.synaptic_event_proxy_by_step)),
            control_cost=control_cost,
            n_steps=int(rollout.rates.shape[0]),
        )


__all__ = [
    "ActuatorMode",
    "FactorizedController",
    "FactorizedControllerConfig",
    "FactorizedRollout",
    "FactorizedRolloutAudit",
    "FactorizedStep",
]
