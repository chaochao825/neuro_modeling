"""Causal jump filters with fixed, factorized, and multiple-model states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.special import logsumexp


ClampName = Literal["h", "q", "r"]
LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class JumpFilterParameters:
    hazard: float
    process_variance: float
    observation_variance: float
    jump_variance: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 < self.hazard < 0.5:
            raise ValueError("hazard must lie strictly between zero and 0.5")
        for name in (
            "process_variance",
            "observation_variance",
            "jump_variance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class AdaptationRates:
    hazard: float
    process_variance: float
    observation_variance: float

    def __post_init__(self) -> None:
        for name in ("hazard", "process_variance", "observation_variance"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} adaptation rate must be in [0, 1]")


@dataclass(frozen=True)
class ParameterBounds:
    hazard: tuple[float, float] = (1e-4, 0.25)
    process_variance: tuple[float, float] = (1e-4, 0.5)
    observation_variance: tuple[float, float] = (1e-4, 1.0)

    def __post_init__(self) -> None:
        for name in ("hazard", "process_variance", "observation_variance"):
            values = getattr(self, name)
            if len(values) != 2 or not 0.0 < values[0] < values[1]:
                raise ValueError(f"{name} bounds must be increasing and positive")
        if self.hazard[1] >= 0.5:
            raise ValueError("upper hazard bound must be below 0.5")


@dataclass(frozen=True)
class FilterTrace:
    predictive_nll: np.ndarray
    predictive_mean: np.ndarray
    filtered_mean: np.ndarray
    filtered_variance: np.ndarray
    hazard: np.ndarray
    process_variance: np.ndarray
    observation_variance: np.ndarray
    jump_probability: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value) for value in self.__dict__.values())
        lengths = {len(value) for value in arrays}
        if len(lengths) != 1:
            raise ValueError("trace arrays must share one length")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("trace arrays must be finite")


@dataclass(frozen=True)
class _Step:
    log_likelihood: float
    predictive_mean: float
    filtered_mean: float
    filtered_variance: float
    jump_probability: float
    process_target: float
    observation_target: float


def _normal_logpdf(value: float, mean: float, variance: float) -> float:
    return -0.5 * (LOG_2PI + np.log(variance) + (value - mean) ** 2 / variance)


def _validate_series(
    observations: np.ndarray, sequence_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(observations, dtype=np.float64)
    groups = np.asarray(sequence_ids)
    if values.ndim != 1 or groups.ndim != 1 or len(values) != len(groups):
        raise ValueError("observations and sequence_ids must be aligned vectors")
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("observations must be non-empty and finite")
    if np.any(groups[1:] < groups[:-1]):
        raise ValueError("sequence_ids must form ordered contiguous groups")
    return values, groups


def _jump_step(
    observation: float,
    *,
    mean: float,
    variance: float,
    parameters: JumpFilterParameters,
) -> _Step:
    h_value = parameters.hazard
    q_value = parameters.process_variance
    r_value = parameters.observation_variance
    continuation_variance = variance + q_value
    continuation_observation_variance = continuation_variance + r_value
    reset_observation_variance = parameters.jump_variance + r_value
    log_continuation = np.log1p(-h_value) + _normal_logpdf(
        observation, mean, continuation_observation_variance
    )
    log_reset = np.log(h_value) + _normal_logpdf(
        observation, 0.0, reset_observation_variance
    )
    log_likelihood = float(logsumexp((log_continuation, log_reset)))
    jump_probability = float(np.exp(log_reset - log_likelihood))

    continuation_gain = continuation_variance / continuation_observation_variance
    continuation_mean = mean + continuation_gain * (observation - mean)
    continuation_posterior_variance = (
        1.0 - continuation_gain
    ) * continuation_variance
    reset_gain = parameters.jump_variance / reset_observation_variance
    reset_mean = reset_gain * observation
    reset_posterior_variance = (1.0 - reset_gain) * parameters.jump_variance
    filtered_mean = (
        (1.0 - jump_probability) * continuation_mean
        + jump_probability * reset_mean
    )
    filtered_variance = (
        (1.0 - jump_probability)
        * (
            continuation_posterior_variance
            + (continuation_mean - filtered_mean) ** 2
        )
        + jump_probability
        * (reset_posterior_variance + (reset_mean - filtered_mean) ** 2)
    )

    continuation_error = observation - mean
    process_mean = q_value / continuation_observation_variance * continuation_error
    process_target = (
        q_value
        - q_value**2 / continuation_observation_variance
        + process_mean**2
    )
    noise_mean_continuation = (
        r_value / continuation_observation_variance * continuation_error
    )
    noise_target_continuation = (
        r_value
        - r_value**2 / continuation_observation_variance
        + noise_mean_continuation**2
    )
    noise_mean_reset = r_value / reset_observation_variance * observation
    noise_target_reset = (
        r_value
        - r_value**2 / reset_observation_variance
        + noise_mean_reset**2
    )
    observation_target = (
        (1.0 - jump_probability) * noise_target_continuation
        + jump_probability * noise_target_reset
    )
    predictive_mean = (1.0 - h_value) * mean
    return _Step(
        log_likelihood=log_likelihood,
        predictive_mean=float(predictive_mean),
        filtered_mean=float(filtered_mean),
        filtered_variance=float(max(filtered_variance, 1e-12)),
        jump_probability=jump_probability,
        process_target=float(max(process_target, 1e-12)),
        observation_target=float(max(observation_target, 1e-12)),
    )


def run_factorized_filter(
    observations: np.ndarray,
    *,
    sequence_ids: np.ndarray,
    initial: JumpFilterParameters,
    adaptation: AdaptationRates,
    bounds: ParameterBounds = ParameterBounds(),
    clamp: ClampName | None = None,
) -> FilterTrace:
    """Run a three-state online-EM controller without test-time gradients."""

    values, groups = _validate_series(observations, sequence_ids)
    if clamp not in {None, "h", "q", "r"}:
        raise ValueError("clamp must be one of h, q, r, or None")
    n_items = len(values)
    outputs = [np.empty(n_items, dtype=np.float64) for _ in range(8)]
    (
        nll,
        predictive_mean,
        filtered_mean,
        filtered_variance,
        hazards,
        process,
        observation_noise,
        jump_probability,
    ) = outputs

    mean = 0.0
    variance = initial.jump_variance
    h_state = initial.hazard
    q_state = initial.process_variance
    r_state = initial.observation_variance
    previous_group: object | None = None
    for index, (value, group) in enumerate(zip(values, groups, strict=True)):
        if index == 0 or group != previous_group:
            mean = 0.0
            variance = initial.jump_variance
            h_state = initial.hazard
            q_state = initial.process_variance
            r_state = initial.observation_variance
        effective_h = initial.hazard if clamp == "h" else h_state
        effective_q = initial.process_variance if clamp == "q" else q_state
        effective_r = initial.observation_variance if clamp == "r" else r_state
        parameters = JumpFilterParameters(
            effective_h, effective_q, effective_r, initial.jump_variance
        )
        step = _jump_step(value, mean=mean, variance=variance, parameters=parameters)
        nll[index] = -step.log_likelihood
        predictive_mean[index] = step.predictive_mean
        filtered_mean[index] = step.filtered_mean
        filtered_variance[index] = step.filtered_variance
        hazards[index] = effective_h
        process[index] = effective_q
        observation_noise[index] = effective_r
        jump_probability[index] = step.jump_probability

        h_state += adaptation.hazard * (step.jump_probability - h_state)
        q_state += (
            adaptation.process_variance
            * (1.0 - step.jump_probability)
            * (step.process_target - q_state)
        )
        r_state += adaptation.observation_variance * (
            step.observation_target - r_state
        )
        h_state = float(np.clip(h_state, *bounds.hazard))
        q_state = float(np.clip(q_state, *bounds.process_variance))
        r_state = float(np.clip(r_state, *bounds.observation_variance))
        mean = step.filtered_mean
        variance = step.filtered_variance
        previous_group = group
    return FilterTrace(*outputs)


def run_fixed_jump_filter(
    observations: np.ndarray,
    *,
    sequence_ids: np.ndarray,
    parameters: JumpFilterParameters,
) -> FilterTrace:
    return run_factorized_filter(
        observations,
        sequence_ids=sequence_ids,
        initial=parameters,
        adaptation=AdaptationRates(0.0, 0.0, 0.0),
    )


def _normal_trace(
    observations: np.ndarray,
    *,
    sequence_ids: np.ndarray,
    update: Literal["ema", "window"],
    predictive_variance: float,
    alpha: float | None = None,
    window: int | None = None,
) -> FilterTrace:
    values, groups = _validate_series(observations, sequence_ids)
    if not np.isfinite(predictive_variance) or predictive_variance <= 0.0:
        raise ValueError("predictive_variance must be positive")
    if update == "ema":
        if alpha is None or not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
    elif window is None or isinstance(window, bool) or int(window) != window or window < 1:
        raise ValueError("window must be a positive integer")
    outputs = [np.empty(len(values), dtype=np.float64) for _ in range(8)]
    history: list[float] = []
    mean = 0.0
    previous_group: object | None = None
    for index, (value, group) in enumerate(zip(values, groups, strict=True)):
        if index == 0 or group != previous_group:
            history = []
            mean = 0.0
        predicted = mean
        outputs[0][index] = -_normal_logpdf(value, predicted, predictive_variance)
        outputs[1][index] = predicted
        if update == "ema":
            assert alpha is not None
            mean = (1.0 - alpha) * mean + alpha * value
        else:
            assert window is not None
            history.append(float(value))
            if len(history) > window:
                history.pop(0)
            mean = float(np.mean(history))
        outputs[2][index] = mean
        outputs[3][index] = predictive_variance
        outputs[4][index] = 0.0
        outputs[5][index] = predictive_variance
        outputs[6][index] = predictive_variance
        outputs[7][index] = 0.0
        previous_group = group
    return FilterTrace(*outputs)


def run_ema_filter(
    observations: np.ndarray,
    *,
    sequence_ids: np.ndarray,
    alpha: float,
    predictive_variance: float,
) -> FilterTrace:
    """Causal exponential mean with a train-calibrated predictive variance."""

    return _normal_trace(
        observations,
        sequence_ids=sequence_ids,
        update="ema",
        predictive_variance=predictive_variance,
        alpha=alpha,
    )


def run_window_filter(
    observations: np.ndarray,
    *,
    sequence_ids: np.ndarray,
    window: int,
    predictive_variance: float,
) -> FilterTrace:
    """Causal rolling mean with a train-calibrated predictive variance."""

    return _normal_trace(
        observations,
        sequence_ids=sequence_ids,
        update="window",
        predictive_variance=predictive_variance,
        window=window,
    )


def _mode_transition(weights: np.ndarray, switch_probability: float) -> np.ndarray:
    return (1.0 - switch_probability) * weights + switch_probability / len(weights)


def run_imm_filter(
    observations: np.ndarray,
    *,
    sequence_ids: np.ndarray,
    modes: Sequence[JumpFilterParameters],
    mode_switch_probability: float,
) -> FilterTrace:
    """Interacting multiple-model jump filter with a symmetric mode prior."""

    values, groups = _validate_series(observations, sequence_ids)
    registered_modes = tuple(modes)
    if len(registered_modes) < 2:
        raise ValueError("IMM requires at least two modes")
    if len({mode.jump_variance for mode in registered_modes}) != 1:
        raise ValueError("IMM modes must share jump variance")
    if not 0.0 <= mode_switch_probability <= 1.0:
        raise ValueError("mode_switch_probability must be in [0, 1]")
    n_modes = len(registered_modes)
    n_items = len(values)
    outputs = [np.empty(n_items, dtype=np.float64) for _ in range(8)]
    (
        nll,
        predictive_mean,
        filtered_mean,
        filtered_variance,
        hazards,
        process,
        observation_noise,
        jump_probability,
    ) = outputs
    weights = np.full(n_modes, 1.0 / n_modes)
    means = np.zeros(n_modes)
    variances = np.full(n_modes, registered_modes[0].jump_variance)
    previous_group: object | None = None

    for index, (value, group) in enumerate(zip(values, groups, strict=True)):
        if index == 0 or group != previous_group:
            weights.fill(1.0 / n_modes)
            means.fill(0.0)
            variances.fill(registered_modes[0].jump_variance)
        prior_weights = _mode_transition(weights, mode_switch_probability)
        if mode_switch_probability == 0.0:
            mixing = np.eye(n_modes)
        else:
            transition = np.full(
                (n_modes, n_modes), mode_switch_probability / n_modes
            )
            transition[np.diag_indices(n_modes)] += 1.0 - mode_switch_probability
            mixing = weights[:, None] * transition / prior_weights[None, :]
        mixed_means = mixing.T @ means
        mixed_variances = np.sum(
            mixing.T
            * (variances[None, :] + (means[None, :] - mixed_means[:, None]) ** 2),
            axis=1,
        )
        steps = tuple(
            _jump_step(
                value,
                mean=float(mixed_means[mode_index]),
                variance=float(mixed_variances[mode_index]),
                parameters=parameters,
            )
            for mode_index, parameters in enumerate(registered_modes)
        )
        log_components = np.asarray(
            [
                np.log(prior_weights[mode_index]) + step.log_likelihood
                for mode_index, step in enumerate(steps)
            ]
        )
        total_log_likelihood = float(logsumexp(log_components))
        weights = np.exp(log_components - total_log_likelihood)
        means = np.asarray([step.filtered_mean for step in steps])
        variances = np.asarray([step.filtered_variance for step in steps])
        predictive_means = np.asarray([step.predictive_mean for step in steps])
        combined_predictive_mean = float(prior_weights @ predictive_means)
        combined_filtered_mean = float(weights @ means)
        combined_filtered_variance = float(
            weights @ (variances + (means - combined_filtered_mean) ** 2)
        )
        nll[index] = -total_log_likelihood
        predictive_mean[index] = combined_predictive_mean
        filtered_mean[index] = combined_filtered_mean
        filtered_variance[index] = combined_filtered_variance
        hazards[index] = float(
            weights @ np.asarray([mode.hazard for mode in registered_modes])
        )
        process[index] = float(
            weights
            @ np.asarray([mode.process_variance for mode in registered_modes])
        )
        observation_noise[index] = float(
            weights
            @ np.asarray([mode.observation_variance for mode in registered_modes])
        )
        jump_probability[index] = float(
            weights @ np.asarray([step.jump_probability for step in steps])
        )
        previous_group = group
    return FilterTrace(*outputs)


def run_oracle_filter(
    observations: np.ndarray,
    *,
    sequence_ids: np.ndarray,
    hazard: Iterable[float],
    process_variance: Iterable[float],
    observation_variance: Iterable[float],
    jump_variance: float,
) -> FilterTrace:
    """Upper bound receiving the generating h/Q/R at every time step."""

    values, groups = _validate_series(observations, sequence_ids)
    h_values = np.asarray(tuple(hazard), dtype=np.float64)
    q_values = np.asarray(tuple(process_variance), dtype=np.float64)
    r_values = np.asarray(tuple(observation_variance), dtype=np.float64)
    if any(len(array) != len(values) for array in (h_values, q_values, r_values)):
        raise ValueError("oracle parameter arrays must align with observations")
    outputs = [np.empty(len(values), dtype=np.float64) for _ in range(8)]
    mean = 0.0
    variance = jump_variance
    previous_group: object | None = None
    for index, (value, group) in enumerate(zip(values, groups, strict=True)):
        if index == 0 or group != previous_group:
            mean = 0.0
            variance = jump_variance
        parameters = JumpFilterParameters(
            h_values[index], q_values[index], r_values[index], jump_variance
        )
        step = _jump_step(value, mean=mean, variance=variance, parameters=parameters)
        outputs[0][index] = -step.log_likelihood
        outputs[1][index] = step.predictive_mean
        outputs[2][index] = step.filtered_mean
        outputs[3][index] = step.filtered_variance
        outputs[4][index] = h_values[index]
        outputs[5][index] = q_values[index]
        outputs[6][index] = r_values[index]
        outputs[7][index] = step.jump_probability
        mean = step.filtered_mean
        variance = step.filtered_variance
        previous_group = group
    return FilterTrace(*outputs)
