"""Task-matched linear actuator families for the Exp26 phase diagram.

This module deliberately does not modify or wrap :mod:`factorized_controller`.
It fits linear, centred-context actuator families to *training* target
trajectories and then rolls every fitted family out on an explicitly supplied
random tape.  The scalar context is a control coordinate; the ranks of the
operators that it selects are separate registered quantities.

For target states ``x``, inputs ``u``, centred scalar context ``s`` and known
process noise ``eps``, fitting uses the teacher-forced residual

``y = x[t + 1] - A0 @ x[t] - B0 @ u[t] - eps[t]``.

The actuator currents are

``routing  : s * dB @ u``
``low_rank: s * dA @ x``
``gain     : s * diag(g) @ (A0 @ x + B0 @ u)``.

Routing and recurrent operators are ridge estimates followed by truncated
SVD.  Gain is an output-wise ridge estimate.  ``rgl`` uses the fixed
sequential order routing, low-rank, gain and is an upper-bound/control
condition rather than a primary scientific contrast.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.models.factorized_controller import ActuatorMode


FloatArray = NDArray[np.float64]


class ActuatorFitError(ValueError):
    """Raised when an active actuator or its functional budget is degenerate."""


def _finite_array(value: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != ndim or array.size == 0 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a numeric scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a numeric scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _fingerprint(label: str, *values: object) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TaskMatchedActuatorConfig:
    """Registered fitting and functional-budget settings."""

    rank_a: int = 1
    rank_b: int = 1
    ridge: float = 1e-8
    max_scale: float = 100.0
    degeneracy_tolerance: float = 1e-12
    budget_relative_tolerance: float = 1e-10
    context_center_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        rank_a = _positive_integer(self.rank_a, name="rank_a")
        rank_b = _positive_integer(self.rank_b, name="rank_b")
        ridge = _finite_scalar(self.ridge, name="ridge")
        max_scale = _finite_scalar(self.max_scale, name="max_scale")
        degeneracy = _finite_scalar(
            self.degeneracy_tolerance, name="degeneracy_tolerance"
        )
        budget_tolerance = _finite_scalar(
            self.budget_relative_tolerance,
            name="budget_relative_tolerance",
        )
        center_tolerance = _finite_scalar(
            self.context_center_tolerance,
            name="context_center_tolerance",
        )
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        if max_scale <= 0.0:
            raise ValueError("max_scale must be positive")
        if degeneracy <= 0.0:
            raise ValueError("degeneracy_tolerance must be positive")
        if budget_tolerance <= 0.0:
            raise ValueError("budget_relative_tolerance must be positive")
        if center_tolerance < 0.0:
            raise ValueError("context_center_tolerance must be non-negative")
        object.__setattr__(self, "rank_a", rank_a)
        object.__setattr__(self, "rank_b", rank_b)


@dataclass(frozen=True)
class ActuatorFitReceipt:
    """Structural and functional receipt for a training-only fit."""

    mode: str
    rank_a_limit: int
    rank_b_limit: int
    raw_recurrent_rank: int
    raw_input_rank: int
    raw_gain_rank: int
    recurrent_rank: int
    input_rank: int
    gain_rank: int
    target_l2_rms: float
    raw_current_l2_rms: float
    matched_current_l2_rms: float
    budget_scale: float
    budget_l2_relative_error: float
    matched_current_l1_mean: float
    teacher_forced_error_rms: float
    teacher_forced_explained_fraction: float
    n_transitions: int
    training_fingerprint: str
    process_noise_fingerprint: str
    correction_fingerprint: str


@dataclass(frozen=True)
class TaskMatchedRollout:
    """A rollout plus component-wise correction-current audit arrays."""

    mode: ActuatorMode
    states: FloatArray
    recurrent_correction_current: FloatArray
    input_correction_current: FloatArray
    gain_correction_current: FloatArray
    total_correction_current: FloatArray
    event_proxy_by_step: FloatArray
    inputs: FloatArray
    contexts: FloatArray
    process_noise: FloatArray
    tape_fingerprint: str


@dataclass(frozen=True)
class TaskMatchedActuator:
    """Immutable task-matched corrections on a frozen linear base."""

    mode: ActuatorMode
    baseline_a: FloatArray
    baseline_b: FloatArray
    delta_a: FloatArray
    delta_b: FloatArray
    gain: FloatArray
    receipt: ActuatorFitReceipt

    def __post_init__(self) -> None:
        mode = ActuatorMode.coerce(self.mode)
        baseline_a = _finite_array(self.baseline_a, name="baseline_a", ndim=2)
        baseline_b = _finite_array(self.baseline_b, name="baseline_b", ndim=2)
        delta_a = _finite_array(self.delta_a, name="delta_a", ndim=2)
        delta_b = _finite_array(self.delta_b, name="delta_b", ndim=2)
        gain = _finite_array(self.gain, name="gain", ndim=1)
        n_state = baseline_a.shape[0]
        if baseline_a.shape != (n_state, n_state):
            raise ValueError("baseline_a must be square")
        if baseline_b.shape[0] != n_state:
            raise ValueError("baseline_b row count must equal the state dimension")
        if delta_a.shape != baseline_a.shape:
            raise ValueError("delta_a must match baseline_a")
        if delta_b.shape != baseline_b.shape:
            raise ValueError("delta_b must match baseline_b")
        if gain.shape != (n_state,):
            raise ValueError("gain must have shape [state_dim]")
        if not isinstance(self.receipt, ActuatorFitReceipt):
            raise TypeError("receipt must be an ActuatorFitReceipt")
        if self.receipt.mode != mode.value:
            raise ValueError("receipt mode does not match actuator mode")
        for name, value in (
            ("baseline_a", baseline_a),
            ("baseline_b", baseline_b),
            ("delta_a", delta_a),
            ("delta_b", delta_b),
            ("gain", gain),
        ):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "mode", mode)

    @property
    def state_dim(self) -> int:
        return int(self.baseline_a.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.baseline_b.shape[1])

    def correction_current(
        self,
        states: ArrayLike,
        inputs: ArrayLike,
        contexts: ArrayLike,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return recurrent, routing, gain and total correction currents.

        The leading dimensions of ``states`` and ``inputs`` must match and the
        final dimensions must be ``state_dim`` and ``input_dim``.  Context is
        scalar and must match their leading dimensions.
        """

        state = _finite_dynamic_array(states, name="states", final=self.state_dim)
        input_array = _finite_dynamic_array(
            inputs, name="inputs", final=self.input_dim
        )
        context = _finite_context_array(contexts, name="contexts")
        if state.shape[:-1] != input_array.shape[:-1]:
            raise ValueError("states and inputs must have matching leading shapes")
        if context.shape != state.shape[:-1]:
            raise ValueError("contexts must match the states' leading shape")
        base = np.einsum("ij,...j->...i", self.baseline_a, state)
        base += np.einsum("ij,...j->...i", self.baseline_b, input_array)
        recurrent = context[..., np.newaxis] * np.einsum(
            "ij,...j->...i", self.delta_a, state
        )
        routing = context[..., np.newaxis] * np.einsum(
            "ij,...j->...i", self.delta_b, input_array
        )
        gain_current = context[..., np.newaxis] * self.gain * base
        total = recurrent + routing + gain_current
        return tuple(
            _readonly(item) for item in (recurrent, routing, gain_current, total)
        )  # type: ignore[return-value]

    def rollout(
        self,
        initial_state: ArrayLike,
        inputs: ArrayLike,
        contexts: ArrayLike,
        *,
        process_noise: ArrayLike | None = None,
    ) -> TaskMatchedRollout:
        """Roll out on one or many trajectories using an explicit shared tape."""

        input_array, context, noise, batched = _normalise_rollout_tape(
            inputs=inputs,
            contexts=contexts,
            process_noise=process_noise,
            state_dim=self.state_dim,
            input_dim=self.input_dim,
        )
        initial = np.asarray(initial_state)
        if initial.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
            raise TypeError("initial_state must be a real numeric array")
        initial = np.asarray(initial, dtype=np.float64)
        expected = (input_array.shape[0], self.state_dim)
        if initial.ndim == 1 and input_array.shape[0] == 1:
            if initial.shape != (self.state_dim,):
                raise ValueError("initial_state must have shape [state_dim]")
            initial = initial[np.newaxis, :]
        elif initial.shape != expected:
            raise ValueError(
                "batched initial_state must have shape [trajectory, state_dim]"
            )
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial_state must contain only finite values")

        n_trajectory, n_steps, _ = input_array.shape
        states = np.empty(
            (n_trajectory, n_steps + 1, self.state_dim), dtype=np.float64
        )
        recurrent = np.empty(
            (n_trajectory, n_steps, self.state_dim), dtype=np.float64
        )
        routing = np.empty_like(recurrent)
        gain_current = np.empty_like(recurrent)
        total = np.empty_like(recurrent)
        events = np.empty((n_trajectory, n_steps), dtype=np.float64)
        states[:, 0] = initial
        for time_index in range(n_steps):
            state = states[:, time_index]
            input_value = input_array[:, time_index]
            scalar_context = context[:, time_index]
            base = np.einsum("ij,bj->bi", self.baseline_a, state)
            base += np.einsum("ij,bj->bi", self.baseline_b, input_value)
            recurrent[:, time_index] = scalar_context[:, None] * np.einsum(
                "ij,bj->bi", self.delta_a, state
            )
            routing[:, time_index] = scalar_context[:, None] * np.einsum(
                "ij,bj->bi", self.delta_b, input_value
            )
            gain_current[:, time_index] = (
                scalar_context[:, None] * self.gain[None, :] * base
            )
            total[:, time_index] = (
                recurrent[:, time_index]
                + routing[:, time_index]
                + gain_current[:, time_index]
            )
            # A uniform current-event proxy: sum of absolute component
            # contributions.  It intentionally does not allow cancellations.
            events[:, time_index] = np.sum(
                np.abs(recurrent[:, time_index])
                + np.abs(routing[:, time_index])
                + np.abs(gain_current[:, time_index]),
                axis=-1,
            )
            states[:, time_index + 1] = (
                base + total[:, time_index] + noise[:, time_index]
            )
            if not np.all(np.isfinite(states[:, time_index + 1])):
                raise FloatingPointError("rollout produced non-finite states")

        tape_fingerprint = _fingerprint(
            "task-matched-rollout-tape-v1",
            initial,
            input_array,
            context,
            noise,
        )
        if not batched:
            return TaskMatchedRollout(
                mode=self.mode,
                states=_readonly(states[0]),
                recurrent_correction_current=_readonly(recurrent[0]),
                input_correction_current=_readonly(routing[0]),
                gain_correction_current=_readonly(gain_current[0]),
                total_correction_current=_readonly(total[0]),
                event_proxy_by_step=_readonly(events[0]),
                inputs=_readonly(input_array[0]),
                contexts=_readonly(context[0]),
                process_noise=_readonly(noise[0]),
                tape_fingerprint=tape_fingerprint,
            )
        return TaskMatchedRollout(
            mode=self.mode,
            states=_readonly(states),
            recurrent_correction_current=_readonly(recurrent),
            input_correction_current=_readonly(routing),
            gain_correction_current=_readonly(gain_current),
            total_correction_current=_readonly(total),
            event_proxy_by_step=_readonly(events),
            inputs=_readonly(input_array),
            contexts=_readonly(context),
            process_noise=_readonly(noise),
            tape_fingerprint=tape_fingerprint,
        )


def _finite_dynamic_array(value: ArrayLike, *, name: str, final: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim < 2 or array.size == 0 or array.shape[-1] != final:
        raise ValueError(f"{name} must be non-empty with final dimension {final}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_context_array(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim < 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty scalar-control array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _normalise_training_data(
    *,
    states: ArrayLike,
    inputs: ArrayLike,
    contexts: ArrayLike,
    process_noise: ArrayLike | None,
    state_dim: int,
    input_dim: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    state = np.asarray(states)
    input_array = np.asarray(inputs)
    context = np.asarray(contexts)
    if state.ndim == 2:
        state = _finite_array(state, name="states", ndim=2)[np.newaxis, ...]
        input_array = _finite_array(input_array, name="inputs", ndim=2)[
            np.newaxis, ...
        ]
        context = _finite_array(context, name="contexts", ndim=1)[np.newaxis, ...]
        if process_noise is None:
            noise = np.zeros(
                (1, input_array.shape[1], state_dim), dtype=np.float64
            )
        else:
            noise = _finite_array(process_noise, name="process_noise", ndim=2)[
                np.newaxis, ...
            ]
    elif state.ndim == 3:
        state = _finite_array(state, name="states", ndim=3)
        input_array = _finite_array(input_array, name="inputs", ndim=3)
        context = _finite_array(context, name="contexts", ndim=2)
        if process_noise is None:
            noise = np.zeros(
                (state.shape[0], input_array.shape[1], state_dim),
                dtype=np.float64,
            )
        else:
            noise = _finite_array(process_noise, name="process_noise", ndim=3)
    else:
        raise ValueError("states must have shape [time+1,state] or [trial,time+1,state]")
    if state.shape[-1] != state_dim:
        raise ValueError("states final dimension must match baseline_a")
    if input_array.shape[-1] != input_dim:
        raise ValueError("inputs final dimension must match baseline_b")
    if state.shape[0] != input_array.shape[0] or state.shape[0] != context.shape[0]:
        raise ValueError("states, inputs, and contexts must have the same trial count")
    if state.shape[1] != input_array.shape[1] + 1:
        raise ValueError("states must contain exactly one more time point than inputs")
    if context.shape != input_array.shape[:2]:
        raise ValueError("contexts must have shape [trial,time]")
    if noise.shape != (state.shape[0], input_array.shape[1], state_dim):
        raise ValueError("process_noise must have shape [trial,time,state_dim]")
    return state, input_array, context, noise


def _normalise_rollout_tape(
    *,
    inputs: ArrayLike,
    contexts: ArrayLike,
    process_noise: ArrayLike | None,
    state_dim: int,
    input_dim: int,
) -> tuple[FloatArray, FloatArray, FloatArray, bool]:
    raw_inputs = np.asarray(inputs)
    if raw_inputs.ndim == 2:
        input_array = _finite_array(raw_inputs, name="inputs", ndim=2)[
            np.newaxis, ...
        ]
        context = _finite_array(contexts, name="contexts", ndim=1)[np.newaxis, ...]
        batched = False
        if process_noise is None:
            noise = np.zeros(
                (1, input_array.shape[1], state_dim), dtype=np.float64
            )
        else:
            noise = _finite_array(process_noise, name="process_noise", ndim=2)[
                np.newaxis, ...
            ]
    elif raw_inputs.ndim == 3:
        input_array = _finite_array(raw_inputs, name="inputs", ndim=3)
        context = _finite_array(contexts, name="contexts", ndim=2)
        batched = True
        if process_noise is None:
            noise = np.zeros(
                (input_array.shape[0], input_array.shape[1], state_dim),
                dtype=np.float64,
            )
        else:
            noise = _finite_array(process_noise, name="process_noise", ndim=3)
    else:
        raise ValueError("inputs must have shape [time,input] or [trial,time,input]")
    if input_array.shape[-1] != input_dim:
        raise ValueError("inputs final dimension must match baseline_b")
    if context.shape != input_array.shape[:2]:
        raise ValueError("contexts must match inputs without the final dimension")
    if noise.shape != (*input_array.shape[:2], state_dim):
        raise ValueError("process_noise must match [trial,time,state_dim]")
    return input_array, context, noise, batched


def _ridge_multivariate(features: FloatArray, targets: FloatArray, ridge: float) -> FloatArray:
    if ridge == 0.0:
        coefficients, _, _, _ = np.linalg.lstsq(features, targets, rcond=None)
    else:
        gram = features.T @ features
        rhs = features.T @ targets
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(features.shape[1], dtype=np.float64), rhs
        )
    return coefficients.T


def _truncate_rank(matrix: FloatArray, rank: int) -> FloatArray:
    limit = min(rank, *matrix.shape)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return (left[:, :limit] * singular[:limit]) @ right[:limit]


def _matrix_rank(matrix: FloatArray, tolerance: float) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=tolerance))


def _component_current(
    *,
    delta_a: FloatArray,
    delta_b: FloatArray,
    gain: FloatArray,
    baseline_current: FloatArray,
    x: FloatArray,
    u: FloatArray,
    s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    recurrent = s[:, None] * (x @ delta_a.T)
    routing = s[:, None] * (u @ delta_b.T)
    gain_current = s[:, None] * gain[None, :] * baseline_current
    return recurrent, routing, gain_current, recurrent + routing + gain_current


def _sequential_fit(
    *,
    mode: ActuatorMode,
    x: FloatArray,
    u: FloatArray,
    s: FloatArray,
    baseline_current: FloatArray,
    target: FloatArray,
    state_dim: int,
    input_dim: int,
    config: TaskMatchedActuatorConfig,
) -> tuple[FloatArray, FloatArray, FloatArray, tuple[int, int, int]]:
    delta_a = np.zeros((state_dim, state_dim), dtype=np.float64)
    delta_b = np.zeros((state_dim, input_dim), dtype=np.float64)
    gain = np.zeros(state_dim, dtype=np.float64)
    raw_a = np.zeros_like(delta_a)
    raw_b = np.zeros_like(delta_b)
    raw_gain = np.zeros_like(gain)
    residual = target.copy()

    # RGL has a registered, deterministic sequential order.  This avoids an
    # unidentifiable joint decomposition when state, input and gain features
    # are collinear.
    if mode.uses_routing:
        raw_b = _ridge_multivariate(s[:, None] * u, residual, config.ridge)
        delta_b = _truncate_rank(raw_b, config.rank_b)
        residual -= s[:, None] * (u @ delta_b.T)
    if mode.uses_low_rank:
        raw_a = _ridge_multivariate(s[:, None] * x, residual, config.ridge)
        delta_a = _truncate_rank(raw_a, config.rank_a)
        residual -= s[:, None] * (x @ delta_a.T)
    if mode.uses_gain:
        feature = s[:, None] * baseline_current
        numerator = np.sum(feature * residual, axis=0)
        denominator = np.sum(feature * feature, axis=0) + config.ridge
        identifiable = denominator > config.degeneracy_tolerance
        raw_gain[identifiable] = numerator[identifiable] / denominator[identifiable]
        gain = raw_gain.copy()

    raw_ranks = (
        _matrix_rank(raw_a, config.degeneracy_tolerance),
        _matrix_rank(raw_b, config.degeneracy_tolerance),
        int(np.count_nonzero(np.abs(raw_gain) > config.degeneracy_tolerance)),
    )
    return delta_a, delta_b, gain, raw_ranks


def fit_task_matched_actuator(
    states: ArrayLike,
    inputs: ArrayLike,
    contexts: ArrayLike,
    baseline_a: ArrayLike,
    baseline_b: ArrayLike,
    *,
    mode: ActuatorMode | str,
    process_noise: ArrayLike | None = None,
    config: TaskMatchedActuatorConfig | None = None,
) -> TaskMatchedActuator:
    """Fit one actuator family using training target trajectories only.

    Every active raw fit is multiplied by one common scalar so that its total
    teacher-forced correction-current RMS equals the registered true task
    residual RMS.  This is an L2 *functional-current* budget.  The L1 current
    is retained only as a descriptive receipt and is not called matched.

    Active fits fail closed if their raw direction is degenerate or if the
    required matching scale exceeds ``config.max_scale``.  ``frozen`` is the
    intentional zero-current control and therefore has no matching scale.
    """

    selected = ActuatorMode.coerce(mode)
    resolved = config or TaskMatchedActuatorConfig()
    if not isinstance(resolved, TaskMatchedActuatorConfig):
        raise TypeError("config must be a TaskMatchedActuatorConfig")
    a0 = _finite_array(baseline_a, name="baseline_a", ndim=2)
    b0 = _finite_array(baseline_b, name="baseline_b", ndim=2)
    if a0.shape[0] != a0.shape[1]:
        raise ValueError("baseline_a must be square")
    if b0.shape[0] != a0.shape[0]:
        raise ValueError("baseline_b row count must equal baseline_a")
    state_dim = a0.shape[0]
    input_dim = b0.shape[1]
    if resolved.rank_a > state_dim:
        raise ValueError("rank_a cannot exceed the state dimension")
    if resolved.rank_b > min(state_dim, input_dim):
        raise ValueError("rank_b cannot exceed min(state_dim, input_dim)")
    state, input_array, context, noise = _normalise_training_data(
        states=states,
        inputs=inputs,
        contexts=contexts,
        process_noise=process_noise,
        state_dim=state_dim,
        input_dim=input_dim,
    )
    flat_context = context.reshape(-1)
    if abs(float(np.mean(flat_context))) > resolved.context_center_tolerance:
        raise ValueError(
            "training contexts must be centered within context_center_tolerance"
        )
    x = state[:, :-1].reshape(-1, state_dim)
    x_next = state[:, 1:].reshape(-1, state_dim)
    u = input_array.reshape(-1, input_dim)
    eps = noise.reshape(-1, state_dim)
    baseline_current = x @ a0.T + u @ b0.T
    target = x_next - baseline_current - eps
    target_l2 = float(np.sqrt(np.mean(target * target)))
    if target_l2 <= resolved.degeneracy_tolerance:
        raise ActuatorFitError("registered task residual is degenerate")

    if selected is ActuatorMode.FROZEN:
        delta_a = np.zeros_like(a0)
        delta_b = np.zeros_like(b0)
        gain = np.zeros(state_dim, dtype=np.float64)
        raw_ranks = (0, 0, 0)
        raw_l2 = 0.0
        matched_l2 = 0.0
        scale = 0.0
        relative_error = 1.0
        matched_l1 = 0.0
        fit_error = target_l2
        explained = 0.0
    else:
        delta_a, delta_b, gain, raw_ranks = _sequential_fit(
            mode=selected,
            x=x,
            u=u,
            s=flat_context,
            baseline_current=baseline_current,
            target=target,
            state_dim=state_dim,
            input_dim=input_dim,
            config=resolved,
        )
        _, _, _, raw_total = _component_current(
            delta_a=delta_a,
            delta_b=delta_b,
            gain=gain,
            baseline_current=baseline_current,
            x=x,
            u=u,
            s=flat_context,
        )
        raw_l2 = float(np.sqrt(np.mean(raw_total * raw_total)))
        if raw_l2 <= resolved.degeneracy_tolerance:
            raise ActuatorFitError("active raw actuator direction is degenerate")
        scale = target_l2 / raw_l2
        if not np.isfinite(scale) or scale > resolved.max_scale:
            raise ActuatorFitError(
                "functional-budget scale is non-finite or exceeds max_scale"
            )
        delta_a *= scale
        delta_b *= scale
        gain *= scale
        recurrent, routing, gain_current, matched_total = _component_current(
            delta_a=delta_a,
            delta_b=delta_b,
            gain=gain,
            baseline_current=baseline_current,
            x=x,
            u=u,
            s=flat_context,
        )
        matched_l2 = float(np.sqrt(np.mean(matched_total * matched_total)))
        relative_error = abs(matched_l2 - target_l2) / max(
            target_l2, resolved.degeneracy_tolerance
        )
        if relative_error > resolved.budget_relative_tolerance:
            raise ActuatorFitError("functional L2 budget matching failed tolerance")
        matched_l1 = float(
            np.mean(np.abs(recurrent) + np.abs(routing) + np.abs(gain_current))
        )
        fit_error = float(np.sqrt(np.mean((target - matched_total) ** 2)))
        explained = 1.0 - (fit_error * fit_error) / (target_l2 * target_l2)

    training_fingerprint = _fingerprint(
        "task-matched-training-v1", state, input_array, context, a0, b0
    )
    process_noise_fingerprint = _fingerprint("process-noise-v1", noise)
    correction_fingerprint = _fingerprint(
        "task-matched-correction-v1",
        selected.value,
        delta_a,
        delta_b,
        gain,
        resolved,
    )
    receipt = ActuatorFitReceipt(
        mode=selected.value,
        rank_a_limit=resolved.rank_a,
        rank_b_limit=resolved.rank_b,
        raw_recurrent_rank=raw_ranks[0],
        raw_input_rank=raw_ranks[1],
        raw_gain_rank=raw_ranks[2],
        recurrent_rank=_matrix_rank(delta_a, resolved.degeneracy_tolerance),
        input_rank=_matrix_rank(delta_b, resolved.degeneracy_tolerance),
        gain_rank=int(
            np.count_nonzero(np.abs(gain) > resolved.degeneracy_tolerance)
        ),
        target_l2_rms=target_l2,
        raw_current_l2_rms=raw_l2,
        matched_current_l2_rms=matched_l2,
        budget_scale=scale,
        budget_l2_relative_error=relative_error,
        matched_current_l1_mean=matched_l1,
        teacher_forced_error_rms=fit_error,
        teacher_forced_explained_fraction=float(explained),
        n_transitions=int(target.shape[0]),
        training_fingerprint=training_fingerprint,
        process_noise_fingerprint=process_noise_fingerprint,
        correction_fingerprint=correction_fingerprint,
    )
    return TaskMatchedActuator(
        mode=selected,
        baseline_a=a0,
        baseline_b=b0,
        delta_a=delta_a,
        delta_b=delta_b,
        gain=gain,
        receipt=receipt,
    )


def fit_task_matched_family(
    states: ArrayLike,
    inputs: ArrayLike,
    contexts: ArrayLike,
    baseline_a: ArrayLike,
    baseline_b: ArrayLike,
    *,
    process_noise: ArrayLike | None = None,
    config: TaskMatchedActuatorConfig | None = None,
    modes: Iterable[ActuatorMode | str] = tuple(ActuatorMode),
) -> dict[str, TaskMatchedActuator]:
    """Fit multiple modes on exactly the same registered training arrays."""

    fitted: dict[str, TaskMatchedActuator] = {}
    for mode in modes:
        selected = ActuatorMode.coerce(mode)
        if selected.value in fitted:
            raise ValueError(f"duplicate mode: {selected.value}")
        fitted[selected.value] = fit_task_matched_actuator(
            states,
            inputs,
            contexts,
            baseline_a,
            baseline_b,
            mode=selected,
            process_noise=process_noise,
            config=config,
        )
    fingerprints = {item.receipt.training_fingerprint for item in fitted.values()}
    noise_fingerprints = {
        item.receipt.process_noise_fingerprint for item in fitted.values()
    }
    if len(fingerprints) != 1 or len(noise_fingerprints) != 1:
        raise RuntimeError("family fit did not preserve a shared training tape")
    return fitted


__all__ = [
    "ActuatorFitError",
    "ActuatorFitReceipt",
    "TaskMatchedActuator",
    "TaskMatchedActuatorConfig",
    "TaskMatchedRollout",
    "fit_task_matched_actuator",
    "fit_task_matched_family",
]
