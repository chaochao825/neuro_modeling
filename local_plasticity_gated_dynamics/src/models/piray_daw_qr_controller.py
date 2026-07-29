"""Causal Q/R controllers for the Piray--Daw bucket-prediction task.

The functions in this module operate on complete blocks but update strictly
left-to-right.  They receive neither true condition labels nor hidden bird
positions.  Every block is reset because the experimental cue makes that
boundary observable to participants and all compared methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import logsumexp


LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class VarianceBounds:
    minimum: float = 0.25
    maximum: float = 256.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum) or not np.isfinite(self.maximum):
            raise ValueError("variance bounds must be finite")
        if not 0.0 < self.minimum < self.maximum:
            raise ValueError("variance bounds must be positive and increasing")


@dataclass(frozen=True)
class QRControllerTrace:
    """Causal controller outputs aligned as ``trial x block`` arrays."""

    predictive_mean: np.ndarray
    posterior_mean: np.ndarray
    gain: np.ndarray
    state_variance: np.ndarray
    predictive_nll: np.ndarray | None = None
    process_variance: np.ndarray | None = None
    observation_variance: np.ndarray | None = None

    def __post_init__(self) -> None:
        required = (
            self.predictive_mean,
            self.posterior_mean,
            self.gain,
            self.state_variance,
        )
        arrays = [np.array(value, dtype=np.float64, copy=True) for value in required]
        if any(value.ndim != 2 for value in arrays):
            raise ValueError("controller arrays must be trial-by-block matrices")
        shapes = {value.shape for value in arrays}
        if len(shapes) != 1 or next(iter(shapes))[0] == 0:
            raise ValueError("controller arrays must share one non-empty shape")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("controller arrays must be finite")
        if np.any((arrays[2] < 0.0) | (arrays[2] > 1.0)):
            raise ValueError("gain must lie in [0, 1]")
        if np.any(arrays[3] < 0.0):
            raise ValueError("state variance cannot be negative")

        optional_values: list[np.ndarray | None] = []
        for value in (
            self.predictive_nll,
            self.process_variance,
            self.observation_variance,
        ):
            if value is None:
                optional_values.append(None)
                continue
            array = np.array(value, dtype=np.float64, copy=True)
            if array.shape != arrays[0].shape or not np.all(np.isfinite(array)):
                raise ValueError("optional trace arrays must be finite and aligned")
            optional_values.append(array)
        if optional_values[0] is not None and np.any(optional_values[0] < 0.0):
            raise ValueError("predictive NLL cannot be negative for this variance range")
        for value in optional_values[1:]:
            if value is not None and np.any(value <= 0.0):
                raise ValueError("Q/R estimates must remain positive")

        names = tuple(self.__dataclass_fields__)
        values: list[np.ndarray | None] = arrays + optional_values
        for name, value in zip(names, values, strict=True):
            if value is not None:
                value.setflags(write=False)
            object.__setattr__(self, name, value)


def _observations(value: np.ndarray) -> np.ndarray:
    observations = np.asarray(value, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[0] < 2:
        raise ValueError("observations must be a trial-by-block matrix")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be finite")
    return observations


def _positive(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _rate(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _normal_nll(value: float, mean: float, variance: float) -> float:
    return 0.5 * (LOG_2PI + np.log(variance) + (value - mean) ** 2 / variance)


def run_fixed_gain(
    observations: np.ndarray,
    *,
    gain: float,
    initial_mean: float = 60.0,
) -> QRControllerTrace:
    """Run a constant-gain delta rule with an observable block reset."""

    values = _observations(observations)
    gain_value = float(gain)
    if not np.isfinite(gain_value) or not 0.0 <= gain_value <= 1.0:
        raise ValueError("gain must lie in [0, 1]")
    if not np.isfinite(initial_mean):
        raise ValueError("initial_mean must be finite")
    shape = values.shape
    prior = np.empty(shape)
    posterior = np.empty(shape)
    gains = np.full(shape, gain_value)
    variance = np.zeros(shape)
    for block in range(shape[1]):
        mean = float(initial_mean)
        for trial in range(shape[0]):
            prior[trial, block] = mean
            mean += gain_value * (values[trial, block] - mean)
            posterior[trial, block] = mean
    return QRControllerTrace(prior, posterior, gains, variance)


def _kalman_step(
    observation: float,
    *,
    mean: float,
    state_variance: float,
    process_variance: float,
    observation_variance: float,
) -> tuple[float, float, float, float, float]:
    prior_variance = state_variance + process_variance
    innovation_variance = prior_variance + observation_variance
    gain = prior_variance / innovation_variance
    error = observation - mean
    posterior_mean = mean + gain * error
    posterior_variance = (1.0 - gain) * prior_variance
    return (
        posterior_mean,
        max(posterior_variance, 1e-12),
        gain,
        _normal_nll(observation, mean, innovation_variance),
        error,
    )


def _empty_trace_arrays(
    shape: tuple[int, int],
) -> tuple[np.ndarray, ...]:
    return tuple(np.empty(shape, dtype=np.float64) for _ in range(7))


def run_kalman_schedule(
    observations: np.ndarray,
    *,
    process_variance: np.ndarray,
    observation_variance: np.ndarray,
    initial_mean: float = 60.0,
    initial_state_variance: float = 10.0,
    bounds: VarianceBounds = VarianceBounds(),
) -> QRControllerTrace:
    """Run a Kalman actuator from a predeclared positive Q/R schedule."""

    values = _observations(observations)
    q_values = np.asarray(process_variance, dtype=np.float64)
    r_values = np.asarray(observation_variance, dtype=np.float64)
    if q_values.shape != values.shape or r_values.shape != values.shape:
        raise ValueError("Q/R schedules must align with observations")
    if not np.all(np.isfinite(q_values)) or not np.all(np.isfinite(r_values)):
        raise ValueError("Q/R schedules must be finite")
    if np.any((q_values < bounds.minimum) | (q_values > bounds.maximum)):
        raise ValueError("process variance schedule lies outside bounds")
    if np.any((r_values < bounds.minimum) | (r_values > bounds.maximum)):
        raise ValueError("observation variance schedule lies outside bounds")
    initial_variance = _positive("initial_state_variance", initial_state_variance)
    if not np.isfinite(initial_mean):
        raise ValueError("initial_mean must be finite")
    prior, posterior, gains, state, nll, q_trace, r_trace = _empty_trace_arrays(
        values.shape
    )
    for block in range(values.shape[1]):
        mean = float(initial_mean)
        variance = initial_variance
        for trial in range(values.shape[0]):
            q_value = float(q_values[trial, block])
            r_value = float(r_values[trial, block])
            prior[trial, block] = mean
            q_trace[trial, block] = q_value
            r_trace[trial, block] = r_value
            mean, variance, gain, score, _ = _kalman_step(
                float(values[trial, block]),
                mean=mean,
                state_variance=variance,
                process_variance=q_value,
                observation_variance=r_value,
            )
            posterior[trial, block] = mean
            state[trial, block] = variance
            gains[trial, block] = gain
            nll[trial, block] = score
    return QRControllerTrace(prior, posterior, gains, state, nll, q_trace, r_trace)


def run_factorized_local_em(
    observations: np.ndarray,
    *,
    initial_process_variance: float,
    initial_observation_variance: float,
    process_rate: float,
    observation_rate: float,
    initial_mean: float = 60.0,
    initial_state_variance: float = 10.0,
    bounds: VarianceBounds = VarianceBounds(),
) -> QRControllerTrace:
    """Update separate Q/R coordinates using local conditional moments.

    Q/R used at trial ``t`` depend only on observations through ``t-1``.  The
    current innovation updates the coordinates for trial ``t+1``.
    """

    values = _observations(observations)
    q0 = float(np.clip(_positive("initial_process_variance", initial_process_variance), bounds.minimum, bounds.maximum))
    r0 = float(np.clip(_positive("initial_observation_variance", initial_observation_variance), bounds.minimum, bounds.maximum))
    beta_q = _rate("process_rate", process_rate)
    beta_r = _rate("observation_rate", observation_rate)
    initial_variance = _positive("initial_state_variance", initial_state_variance)
    if not np.isfinite(initial_mean):
        raise ValueError("initial_mean must be finite")
    prior, posterior, gains, state, nll, q_trace, r_trace = _empty_trace_arrays(
        values.shape
    )
    for block in range(values.shape[1]):
        mean = float(initial_mean)
        variance = initial_variance
        q_value = q0
        r_value = r0
        for trial in range(values.shape[0]):
            observation = float(values[trial, block])
            prior[trial, block] = mean
            q_trace[trial, block] = q_value
            r_trace[trial, block] = r_value
            previous_variance = variance
            mean, variance, gain, score, error = _kalman_step(
                observation,
                mean=mean,
                state_variance=previous_variance,
                process_variance=q_value,
                observation_variance=r_value,
            )
            innovation_variance = previous_variance + q_value + r_value
            process_target = (
                q_value
                - q_value**2 / innovation_variance
                + (q_value * error / innovation_variance) ** 2
            )
            observation_target = (
                r_value
                - r_value**2 / innovation_variance
                + (r_value * error / innovation_variance) ** 2
            )
            q_value = float(
                np.clip(
                    q_value + beta_q * (process_target - q_value),
                    bounds.minimum,
                    bounds.maximum,
                )
            )
            r_value = float(
                np.clip(
                    r_value + beta_r * (observation_target - r_value),
                    bounds.minimum,
                    bounds.maximum,
                )
            )
            posterior[trial, block] = mean
            state[trial, block] = variance
            gains[trial, block] = gain
            nll[trial, block] = score
    return QRControllerTrace(prior, posterior, gains, state, nll, q_trace, r_trace)


def run_total_uncertainty(
    observations: np.ndarray,
    *,
    initial_total_variance: float,
    q_fraction: float,
    adaptation_rate: float,
    initial_mean: float = 60.0,
    initial_state_variance: float = 10.0,
    bounds: VarianceBounds = VarianceBounds(),
) -> QRControllerTrace:
    """Run a matched one-coordinate controller with fixed Q/R allocation."""

    values = _observations(observations)
    total0 = _positive("initial_total_variance", initial_total_variance)
    fraction = float(q_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("q_fraction must lie in (0, 1)")
    beta = _rate("adaptation_rate", adaptation_rate)
    initial_variance = _positive("initial_state_variance", initial_state_variance)
    total_min = bounds.minimum / min(fraction, 1.0 - fraction)
    total_max = bounds.maximum / max(fraction, 1.0 - fraction)
    total0 = float(np.clip(total0, total_min, total_max))
    prior, posterior, gains, state, nll, q_trace, r_trace = _empty_trace_arrays(
        values.shape
    )
    for block in range(values.shape[1]):
        mean = float(initial_mean)
        variance = initial_variance
        total = total0
        for trial in range(values.shape[0]):
            q_value = fraction * total
            r_value = (1.0 - fraction) * total
            observation = float(values[trial, block])
            prior[trial, block] = mean
            q_trace[trial, block] = q_value
            r_trace[trial, block] = r_value
            previous_variance = variance
            mean, variance, gain, score, error = _kalman_step(
                observation,
                mean=mean,
                state_variance=previous_variance,
                process_variance=q_value,
                observation_variance=r_value,
            )
            innovation_variance = previous_variance + total
            q_target = (
                q_value
                - q_value**2 / innovation_variance
                + (q_value * error / innovation_variance) ** 2
            )
            r_target = (
                r_value
                - r_value**2 / innovation_variance
                + (r_value * error / innovation_variance) ** 2
            )
            total = float(
                np.clip(
                    total + beta * (q_target + r_target - total),
                    total_min,
                    total_max,
                )
            )
            posterior[trial, block] = mean
            state[trial, block] = variance
            gains[trial, block] = gain
            nll[trial, block] = score
    return QRControllerTrace(prior, posterior, gains, state, nll, q_trace, r_trace)


def run_autocovariance_qr(
    observations: np.ndarray,
    *,
    initial_process_variance: float,
    initial_observation_variance: float,
    statistic_decay: float,
    prior_mass: float,
    initial_mean: float = 60.0,
    initial_state_variance: float = 10.0,
    bounds: VarianceBounds = VarianceBounds(),
) -> QRControllerTrace:
    """Estimate Q/R from causal lag-zero and lag-one observation increments."""

    values = _observations(observations)
    q0 = float(np.clip(_positive("initial_process_variance", initial_process_variance), bounds.minimum, bounds.maximum))
    r0 = float(np.clip(_positive("initial_observation_variance", initial_observation_variance), bounds.minimum, bounds.maximum))
    decay = float(statistic_decay)
    if not np.isfinite(decay) or not 0.0 < decay <= 1.0:
        raise ValueError("statistic_decay must lie in (0, 1]")
    mass0 = _positive("prior_mass", prior_mass)
    initial_variance = _positive("initial_state_variance", initial_state_variance)
    prior, posterior, gains, state, nll, q_trace, r_trace = _empty_trace_arrays(
        values.shape
    )
    for block in range(values.shape[1]):
        mean = float(initial_mean)
        variance = initial_variance
        q_value = q0
        r_value = r0
        s0 = mass0 * (q0 + 2.0 * r0)
        n0 = mass0
        s1 = -mass0 * r0
        n1 = mass0
        previous_observation: float | None = None
        previous_increment: float | None = None
        for trial in range(values.shape[0]):
            observation = float(values[trial, block])
            prior[trial, block] = mean
            q_trace[trial, block] = q_value
            r_trace[trial, block] = r_value
            mean, variance, gain, score, _ = _kalman_step(
                observation,
                mean=mean,
                state_variance=variance,
                process_variance=q_value,
                observation_variance=r_value,
            )
            if previous_observation is not None:
                increment = observation - previous_observation
                s0 = decay * s0 + increment**2
                n0 = decay * n0 + 1.0
                if previous_increment is not None:
                    s1 = decay * s1 + increment * previous_increment
                    n1 = decay * n1 + 1.0
                previous_increment = increment
                gamma0 = s0 / n0
                gamma1 = s1 / n1
                q_value = float(
                    np.clip(gamma0 + 2.0 * gamma1, bounds.minimum, bounds.maximum)
                )
                r_value = float(
                    np.clip(-gamma1, bounds.minimum, bounds.maximum)
                )
            previous_observation = observation
            posterior[trial, block] = mean
            state[trial, block] = variance
            gains[trial, block] = gain
            nll[trial, block] = score
    return QRControllerTrace(prior, posterior, gains, state, nll, q_trace, r_trace)


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def run_hierarchical_particle(
    observations: np.ndarray,
    *,
    change_probability_q: float,
    change_probability_r: float,
    log_step_scale: float,
    particle_count: int,
    seed: int,
    initial_process_variance: float = 1.0,
    initial_observation_variance: float = 1.0,
    initial_mean: float = 60.0,
    initial_state_variance: float = 10.0,
    resample_fraction: float = 0.5,
    bounds: VarianceBounds = VarianceBounds(),
) -> QRControllerTrace:
    """Causal Rao--Blackwellized particle comparator using bag history only.

    The gain recorded for trial ``t`` is previsible: particles have undergone
    their Q/R transition, but have not been reweighted by observation ``t``.
    This matches the timing contract of the local and fixed controllers.
    """

    values = _observations(observations)
    mu_q = _rate("change_probability_q", change_probability_q)
    mu_r = _rate("change_probability_r", change_probability_r)
    step_scale = _positive("log_step_scale", log_step_scale)
    q0 = float(np.clip(_positive("initial_process_variance", initial_process_variance), bounds.minimum, bounds.maximum))
    r0 = float(np.clip(_positive("initial_observation_variance", initial_observation_variance), bounds.minimum, bounds.maximum))
    initial_variance = _positive("initial_state_variance", initial_state_variance)
    if int(particle_count) != particle_count or particle_count < 32:
        raise ValueError("particle_count must be an integer >= 32")
    if not np.isfinite(resample_fraction) or not 0.0 < resample_fraction <= 1.0:
        raise ValueError("resample_fraction must lie in (0, 1]")
    if int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = np.random.default_rng(int(seed))
    prior, posterior, gains, state, nll, q_trace, r_trace = _empty_trace_arrays(
        values.shape
    )
    n_particles = int(particle_count)
    for block in range(values.shape[1]):
        particle_mean = np.full(n_particles, float(initial_mean))
        particle_state_variance = np.full(n_particles, initial_variance)
        q_particles = np.full(n_particles, q0)
        r_particles = np.full(n_particles, r0)
        weights = np.full(n_particles, 1.0 / n_particles)
        for trial in range(values.shape[0]):
            change_q = rng.random(n_particles) < mu_q
            change_r = rng.random(n_particles) < mu_r
            if np.any(change_q):
                q_particles[change_q] *= np.exp(
                    step_scale * rng.standard_normal(int(change_q.sum()))
                )
            if np.any(change_r):
                r_particles[change_r] *= np.exp(
                    step_scale * rng.standard_normal(int(change_r.sum()))
                )
            np.clip(q_particles, bounds.minimum, bounds.maximum, out=q_particles)
            np.clip(r_particles, bounds.minimum, bounds.maximum, out=r_particles)

            prior_variance = particle_state_variance + q_particles
            innovation_variance = prior_variance + r_particles
            particle_gain = prior_variance / innovation_variance
            prior[trial, block] = float(np.dot(weights, particle_mean))
            gains[trial, block] = float(np.dot(weights, particle_gain))
            q_trace[trial, block] = float(np.dot(weights, q_particles))
            r_trace[trial, block] = float(np.dot(weights, r_particles))

            observation = float(values[trial, block])
            log_density = -0.5 * (
                LOG_2PI
                + np.log(innovation_variance)
                + (observation - particle_mean) ** 2 / innovation_variance
            )
            log_weights = np.log(weights) + log_density
            log_evidence = float(logsumexp(log_weights))
            nll[trial, block] = -log_evidence
            weights = np.exp(log_weights - log_evidence)

            if 1.0 / np.sum(weights**2) < resample_fraction * n_particles:
                indices = _systematic_resample(weights, rng)
                particle_mean = particle_mean[indices]
                particle_state_variance = particle_state_variance[indices]
                q_particles = q_particles[indices]
                r_particles = r_particles[indices]
                weights.fill(1.0 / n_particles)
                prior_variance = particle_state_variance + q_particles
                innovation_variance = prior_variance + r_particles
                particle_gain = prior_variance / innovation_variance

            particle_mean += particle_gain * (observation - particle_mean)
            particle_state_variance = (1.0 - particle_gain) * prior_variance
            posterior[trial, block] = float(np.dot(weights, particle_mean))
            state[trial, block] = float(np.dot(weights, particle_state_variance))

    return QRControllerTrace(prior, posterior, gains, state, nll, q_trace, r_trace)


def average_traces(traces: Iterable[QRControllerTrace]) -> QRControllerTrace:
    """Average Monte Carlo traces before participant-level inference."""

    items = tuple(traces)
    if not items:
        raise ValueError("at least one trace is required")
    shape = items[0].gain.shape
    if any(trace.gain.shape != shape for trace in items):
        raise ValueError("all traces must share one shape")
    optional_names = (
        "predictive_nll",
        "process_variance",
        "observation_variance",
    )
    optional: dict[str, np.ndarray | None] = {}
    for name in optional_names:
        values = [getattr(trace, name) for trace in items]
        if all(value is None for value in values):
            optional[name] = None
        elif any(value is None for value in values):
            raise ValueError(f"trace field {name} is inconsistently present")
        else:
            optional[name] = np.mean(np.stack(values), axis=0)
    return QRControllerTrace(
        predictive_mean=np.mean(np.stack([trace.predictive_mean for trace in items]), axis=0),
        posterior_mean=np.mean(np.stack([trace.posterior_mean for trace in items]), axis=0),
        gain=np.mean(np.stack([trace.gain for trace in items]), axis=0),
        state_variance=np.mean(np.stack([trace.state_variance for trace in items]), axis=0),
        **optional,
    )


__all__ = [
    "QRControllerTrace",
    "VarianceBounds",
    "average_traces",
    "run_autocovariance_qr",
    "run_factorized_local_em",
    "run_fixed_gain",
    "run_hierarchical_particle",
    "run_kalman_schedule",
    "run_total_uncertainty",
]
