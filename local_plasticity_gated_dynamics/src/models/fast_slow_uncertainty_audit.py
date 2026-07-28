"""Causal inference/action exchange audit without changing the Exp39 model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.factorized_uncertainty_filter import (
    AdaptationRates,
    JumpFilterParameters,
    ParameterBounds,
    _jump_step,
    _validate_series,
)


@dataclass(frozen=True)
class FastSlowAuditTrace:
    """Predictive, inference, and executed-action traces for one tape."""

    predictive_nll: np.ndarray
    predictive_mean: np.ndarray
    filtered_mean: np.ndarray
    filtered_variance: np.ndarray
    hazard: np.ndarray
    process_variance: np.ndarray
    observation_variance: np.ndarray
    jump_probability: np.ndarray
    release_probability: np.ndarray
    write_gain: np.ndarray
    reset_gain: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value) for value in self.__dict__.values())
        lengths = {len(value) for value in arrays}
        if len(lengths) != 1 or not lengths:
            raise ValueError("trace arrays must share one length")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("trace arrays must be finite")


def _optional_trace(
    value: np.ndarray | None,
    *,
    name: str,
    length: int,
    probability: bool = False,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or len(array) != length or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector aligned with observations")
    if probability:
        if np.any((array < 0.0) | (array > 1.0)):
            raise ValueError(f"{name} must lie in [0, 1]")
    elif np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    return array


def _actuate_state(
    observation: float,
    *,
    mean: float,
    variance: float,
    parameters: JumpFilterParameters,
    release_probability: float,
) -> tuple[float, float, float, float]:
    continuation_variance = variance + parameters.process_variance
    continuation_observation_variance = (
        continuation_variance + parameters.observation_variance
    )
    write_gain = continuation_variance / continuation_observation_variance
    continuation_mean = mean + write_gain * (observation - mean)
    continuation_posterior_variance = (
        1.0 - write_gain
    ) * continuation_variance

    reset_observation_variance = (
        parameters.jump_variance + parameters.observation_variance
    )
    reset_gain = parameters.jump_variance / reset_observation_variance
    reset_mean = reset_gain * observation
    reset_posterior_variance = (
        1.0 - reset_gain
    ) * parameters.jump_variance
    acted_mean = (
        (1.0 - release_probability) * continuation_mean
        + release_probability * reset_mean
    )
    acted_variance = (
        (1.0 - release_probability)
        * (
            continuation_posterior_variance
            + (continuation_mean - acted_mean) ** 2
        )
        + release_probability
        * (reset_posterior_variance + (reset_mean - acted_mean) ** 2)
    )
    return (
        float(acted_mean),
        float(max(acted_variance, 1e-12)),
        float(write_gain),
        float(reset_gain),
    )


def run_fast_slow_exchange_filter(
    observations: np.ndarray,
    *,
    sequence_ids: np.ndarray,
    initial: JumpFilterParameters,
    adaptation: AdaptationRates,
    bounds: ParameterBounds = ParameterBounds(),
    release_override: np.ndarray | None = None,
    process_override: np.ndarray | None = None,
    observation_override: np.ndarray | None = None,
) -> FastSlowAuditTrace:
    """Run the Exp39 update with optional, explicitly privileged action inputs.

    ``release_override`` is consumed only after scoring the current sample, so
    it cannot improve that sample's predictive NLL. Q/R overrides are used for
    prediction and gain and are therefore privileged dynamic-oracle inputs.
    Learned h/Q/R states continue to update identically in all arms; an
    override changes only the named executed action path.
    """

    values, groups = _validate_series(observations, sequence_ids)
    release_values = _optional_trace(
        release_override,
        name="release_override",
        length=len(values),
        probability=True,
    )
    q_values = _optional_trace(
        process_override,
        name="process_override",
        length=len(values),
    )
    r_values = _optional_trace(
        observation_override,
        name="observation_override",
        length=len(values),
    )
    if (q_values is None) != (r_values is None):
        raise ValueError("process and observation overrides must be supplied together")

    outputs = [np.empty(len(values), dtype=np.float64) for _ in range(11)]
    (
        nll,
        predictive_mean,
        filtered_mean,
        filtered_variance,
        hazards,
        process,
        observation_noise,
        jump_probability,
        release_probability,
        write_gain,
        reset_gain,
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
        effective_q = q_state if q_values is None else float(q_values[index])
        effective_r = r_state if r_values is None else float(r_values[index])
        parameters = JumpFilterParameters(
            h_state,
            effective_q,
            effective_r,
            initial.jump_variance,
        )
        step = _jump_step(value, mean=mean, variance=variance, parameters=parameters)
        release = (
            step.jump_probability
            if release_values is None
            else float(release_values[index])
        )
        acted_mean, acted_variance, gain, restart_gain = _actuate_state(
            value,
            mean=mean,
            variance=variance,
            parameters=parameters,
            release_probability=release,
        )

        nll[index] = -step.log_likelihood
        predictive_mean[index] = step.predictive_mean
        filtered_mean[index] = acted_mean
        filtered_variance[index] = acted_variance
        hazards[index] = h_state
        process[index] = effective_q
        observation_noise[index] = effective_r
        jump_probability[index] = step.jump_probability
        release_probability[index] = release
        write_gain[index] = gain
        reset_gain[index] = restart_gain

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
        mean = acted_mean
        variance = acted_variance
        previous_group = group

    return FastSlowAuditTrace(*outputs)


__all__ = ["FastSlowAuditTrace", "run_fast_slow_exchange_filter"]
