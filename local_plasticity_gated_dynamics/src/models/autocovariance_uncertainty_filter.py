"""Causal Q/R estimation from local lag-zero and lag-one increments.

For the H=0 random-walk observation model,

``x[t] = x[t-1] + w[t]`` and ``y[t] = x[t] + e[t]``,

the observed increment has ``gamma0 = Q + 2R`` and
``gamma1 = -R``.  This module maintains exponentially discounted local
sufficient statistics for both moments.  It deliberately defines its own
trace contract and leaves the Exp39 ``FilterTrace`` and algorithms untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.factorized_uncertainty_filter import (
    JumpFilterParameters,
    ParameterBounds,
    _jump_step,
)


# Frozen semantic dependency: `_jump_step` is the already-audited Exp39 jump
# update.  Reusing it keeps the likelihood/state update directly comparable;
# Exp41 must not change that function or route generating metadata into it.


@dataclass(frozen=True)
class AutocovarianceUpdateConfig:
    """Rates and prior mass for local autocovariance sufficient statistics."""

    statistic_decay: float = 0.995
    prior_mass: float = 8.0
    hazard_rate: float = 0.0
    minimum_continuation_weight: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.statistic_decay <= 1.0:
            raise ValueError("statistic_decay must lie in (0, 1]")
        if not np.isfinite(self.prior_mass) or self.prior_mass <= 0.0:
            raise ValueError("prior_mass must be positive")
        if not 0.0 <= self.hazard_rate <= 1.0:
            raise ValueError("hazard_rate must lie in [0, 1]")
        if not 0.0 <= self.minimum_continuation_weight <= 1.0:
            raise ValueError("minimum_continuation_weight must lie in [0, 1]")


@dataclass
class _LocalAutocovarianceState:
    """Online state available at one time step; no future or regime metadata."""

    s0: float
    n0: float
    s1: float
    n1: float
    previous_observation: float | None
    previous_increment: float | None
    previous_rho: float


@dataclass(frozen=True)
class AutocovarianceFilterTrace:
    """Audit trace for the autocovariance controller.

    Q/R arrays record the parameters used to score the current observation.
    Sufficient-statistic arrays record state after that observation, which can
    only affect later predictions.  The separation makes prefix causality
    explicit and testable.
    """

    predictive_nll: np.ndarray
    predictive_mean: np.ndarray
    filtered_mean: np.ndarray
    filtered_variance: np.ndarray
    hazard: np.ndarray
    process_variance: np.ndarray
    observation_variance: np.ndarray
    jump_probability: np.ndarray
    s0: np.ndarray
    n0: np.ndarray
    s1: np.ndarray
    n1: np.ndarray
    gamma0: np.ndarray
    gamma1: np.ndarray
    previous_increment: np.ndarray
    previous_rho: np.ndarray
    has_previous_increment: np.ndarray

    def __post_init__(self) -> None:
        raw_arrays = tuple(np.asarray(value) for value in self.__dict__.values())
        if raw_arrays[-1].dtype != np.bool_:
            raise ValueError("has_previous_increment must be boolean")
        arrays = tuple(
            np.array(value, dtype=np.float64, copy=True) for value in raw_arrays[:-1]
        ) + (np.array(raw_arrays[-1], dtype=bool, copy=True),)
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("trace arrays must be one-dimensional")
        lengths = {len(value) for value in arrays}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("trace arrays must share one non-zero length")
        numeric = arrays[:-1]
        if not all(np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("numeric trace arrays must be finite")
        if np.any(arrays[9] <= 0.0) or np.any(arrays[11] <= 0.0):
            raise ValueError("autocovariance masses must remain positive")
        for field, value in zip(self.__dataclass_fields__, arrays, strict=True):
            value.setflags(write=False)
            object.__setattr__(self, field, value)


@dataclass(frozen=True)
class TotalVarianceFilterTrace:
    """Trace for the single variance-coordinate reduced baseline."""

    predictive_nll: np.ndarray
    predictive_mean: np.ndarray
    filtered_mean: np.ndarray
    filtered_variance: np.ndarray
    hazard: np.ndarray
    process_variance: np.ndarray
    observation_variance: np.ndarray
    jump_probability: np.ndarray
    s0: np.ndarray
    n0: np.ndarray
    gamma0: np.ndarray
    q_fraction: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(
            np.array(value, dtype=np.float64, copy=True)
            for value in self.__dict__.values()
        )
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("trace arrays must be one-dimensional")
        lengths = {len(value) for value in arrays}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("trace arrays must share one non-zero length")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("trace arrays must be finite")
        if np.any(arrays[9] <= 0.0):
            raise ValueError("lag-zero mass must remain positive")
        if np.any((arrays[11] <= 0.0) | (arrays[11] >= 1.0)):
            raise ValueError("q_fraction must remain in (0, 1)")
        for field, value in zip(self.__dataclass_fields__, arrays, strict=True):
            value.setflags(write=False)
            object.__setattr__(self, field, value)


def derive_qr_from_increment_covariance(
    gamma0: float,
    gamma1: float,
    *,
    bounds: ParameterBounds = ParameterBounds(),
) -> tuple[float, float]:
    """Map increment autocovariances to bounded ``(Q, R)`` estimates."""

    if not np.isfinite(gamma0) or not np.isfinite(gamma1):
        raise ValueError("increment autocovariances must be finite")
    observation_variance = float(np.clip(-gamma1, *bounds.observation_variance))
    process_variance = float(np.clip(gamma0 + 2.0 * gamma1, *bounds.process_variance))
    return process_variance, observation_variance


def qr_from_total_variance(
    gamma0: float,
    q_fraction: float,
    *,
    bounds: ParameterBounds = ParameterBounds(),
) -> tuple[float, float]:
    """Allocate one total-increment-variance state at a fixed Q/R ratio.

    With fixed hazard, this is mathematically the same one-coordinate model as
    a "tied Q/R" controller.  Exp41 therefore executes it once rather than
    manufacturing two aliases with identical numerical semantics.
    """

    if not np.isfinite(gamma0) or gamma0 <= 0.0:
        raise ValueError("gamma0 must be positive and finite")
    if not np.isfinite(q_fraction) or not 0.0 < q_fraction < 1.0:
        raise ValueError("q_fraction must lie strictly between zero and one")
    process_variance = float(np.clip(q_fraction * gamma0, *bounds.process_variance))
    observation_variance = float(
        np.clip(
            0.5 * (1.0 - q_fraction) * gamma0,
            *bounds.observation_variance,
        )
    )
    return process_variance, observation_variance


def _validate_inputs(
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


def _initial_local_state(
    initial: JumpFilterParameters, update: AutocovarianceUpdateConfig
) -> _LocalAutocovarianceState:
    gamma0 = initial.process_variance + 2.0 * initial.observation_variance
    gamma1 = -initial.observation_variance
    return _LocalAutocovarianceState(
        s0=update.prior_mass * gamma0,
        n0=update.prior_mass,
        s1=update.prior_mass * gamma1,
        n1=update.prior_mass,
        previous_observation=None,
        previous_increment=None,
        previous_rho=1.0,
    )


def _update_local_statistics(
    state: _LocalAutocovarianceState,
    *,
    observation: float,
    rho: float,
    update: AutocovarianceUpdateConfig,
) -> None:
    if state.previous_observation is None:
        state.previous_observation = observation
        state.previous_rho = rho
        return

    increment = observation - state.previous_observation
    continuation = max(1.0 - rho, update.minimum_continuation_weight)
    previous_continuation = max(
        1.0 - state.previous_rho, update.minimum_continuation_weight
    )
    pair_weight = continuation * previous_continuation
    decay = update.statistic_decay
    state.s0 = decay * state.s0 + pair_weight * increment**2
    state.n0 = decay * state.n0 + pair_weight

    if state.previous_increment is not None:
        state.s1 = decay * state.s1 + pair_weight * (
            increment * state.previous_increment
        )
        state.n1 = decay * state.n1 + pair_weight

    state.previous_observation = observation
    state.previous_increment = increment
    state.previous_rho = rho


def run_autocovariance_filter(
    observations: np.ndarray,
    *,
    sequence_ids: np.ndarray,
    initial: JumpFilterParameters,
    update: AutocovarianceUpdateConfig = AutocovarianceUpdateConfig(),
    bounds: ParameterBounds = ParameterBounds(),
) -> AutocovarianceFilterTrace:
    """Run a prefix-causal local Q/R controller.

    The method receives observations and independent-sequence identifiers only.
    It has no arguments for true Q/R, regime identity, or block boundaries.
    """

    values, groups = _validate_inputs(observations, sequence_ids)
    for name, value, limits in (
        ("hazard", initial.hazard, bounds.hazard),
        ("process_variance", initial.process_variance, bounds.process_variance),
        (
            "observation_variance",
            initial.observation_variance,
            bounds.observation_variance,
        ),
    ):
        if not limits[0] <= value <= limits[1]:
            raise ValueError(f"initial {name} must lie within configured bounds")
    output_count = len(AutocovarianceFilterTrace.__dataclass_fields__)
    outputs = [np.empty(len(values), dtype=np.float64) for _ in range(output_count)]
    outputs[-1] = np.empty(len(values), dtype=bool)
    (
        predictive_nll,
        predictive_mean,
        filtered_mean,
        filtered_variance,
        hazards,
        process,
        observation_noise,
        jump_probability,
        s0,
        n0,
        s1,
        n1,
        gamma0,
        gamma1,
        previous_increment,
        previous_rho,
        has_previous_increment,
    ) = outputs

    mean = 0.0
    variance = initial.jump_variance
    h_state = initial.hazard
    q_state = initial.process_variance
    r_state = initial.observation_variance
    local = _initial_local_state(initial, update)
    previous_group: object | None = None

    for index, (value, group) in enumerate(zip(values, groups, strict=True)):
        if index == 0 or group != previous_group:
            mean = 0.0
            variance = initial.jump_variance
            h_state = initial.hazard
            q_state = initial.process_variance
            r_state = initial.observation_variance
            local = _initial_local_state(initial, update)

        had_previous_increment = local.previous_increment is not None
        previous_increment[index] = (
            float(local.previous_increment) if had_previous_increment else 0.0
        )
        previous_rho[index] = local.previous_rho
        has_previous_increment[index] = had_previous_increment

        parameters = JumpFilterParameters(
            h_state,
            q_state,
            r_state,
            initial.jump_variance,
        )
        step = _jump_step(value, mean=mean, variance=variance, parameters=parameters)
        predictive_nll[index] = -step.log_likelihood
        predictive_mean[index] = step.predictive_mean
        filtered_mean[index] = step.filtered_mean
        filtered_variance[index] = step.filtered_variance
        hazards[index] = h_state
        process[index] = q_state
        observation_noise[index] = r_state
        jump_probability[index] = step.jump_probability

        _update_local_statistics(
            local,
            observation=float(value),
            rho=step.jump_probability,
            update=update,
        )
        gamma0_value = local.s0 / local.n0
        gamma1_value = local.s1 / local.n1
        q_state, r_state = derive_qr_from_increment_covariance(
            gamma0_value,
            gamma1_value,
            bounds=bounds,
        )
        h_state += update.hazard_rate * (step.jump_probability - h_state)
        h_state = float(np.clip(h_state, *bounds.hazard))

        s0[index] = local.s0
        n0[index] = local.n0
        s1[index] = local.s1
        n1[index] = local.n1
        gamma0[index] = gamma0_value
        gamma1[index] = gamma1_value
        mean = step.filtered_mean
        variance = step.filtered_variance
        previous_group = group

    return AutocovarianceFilterTrace(*outputs)


def run_total_variance_filter(
    observations: np.ndarray,
    *,
    sequence_ids: np.ndarray,
    initial: JumpFilterParameters,
    q_fraction: float,
    update: AutocovarianceUpdateConfig = AutocovarianceUpdateConfig(),
    bounds: ParameterBounds = ParameterBounds(),
) -> TotalVarianceFilterTrace:
    """Run the one-coordinate total-variance/tied-Q/R reduced baseline."""

    values, groups = _validate_inputs(observations, sequence_ids)
    for name, value, limits in (
        ("hazard", initial.hazard, bounds.hazard),
        ("process_variance", initial.process_variance, bounds.process_variance),
        (
            "observation_variance",
            initial.observation_variance,
            bounds.observation_variance,
        ),
    ):
        if not limits[0] <= value <= limits[1]:
            raise ValueError(f"initial {name} must lie within configured bounds")
    initial_q, initial_r = qr_from_total_variance(
        initial.process_variance + 2.0 * initial.observation_variance,
        q_fraction,
        bounds=bounds,
    )

    output_count = len(TotalVarianceFilterTrace.__dataclass_fields__)
    outputs = [np.empty(len(values), dtype=np.float64) for _ in range(output_count)]
    (
        predictive_nll,
        predictive_mean,
        filtered_mean,
        filtered_variance,
        hazards,
        process,
        observation_noise,
        jump_probability,
        s0,
        n0,
        gamma0,
        q_fraction_trace,
    ) = outputs

    mean = 0.0
    variance = initial.jump_variance
    h_state = initial.hazard
    q_state = initial_q
    r_state = initial_r
    local = _initial_local_state(initial, update)
    previous_group: object | None = None
    for index, (value, group) in enumerate(zip(values, groups, strict=True)):
        if index == 0 or group != previous_group:
            mean = 0.0
            variance = initial.jump_variance
            h_state = initial.hazard
            q_state = initial_q
            r_state = initial_r
            local = _initial_local_state(initial, update)

        parameters = JumpFilterParameters(
            h_state,
            q_state,
            r_state,
            initial.jump_variance,
        )
        step = _jump_step(value, mean=mean, variance=variance, parameters=parameters)
        predictive_nll[index] = -step.log_likelihood
        predictive_mean[index] = step.predictive_mean
        filtered_mean[index] = step.filtered_mean
        filtered_variance[index] = step.filtered_variance
        hazards[index] = h_state
        process[index] = q_state
        observation_noise[index] = r_state
        jump_probability[index] = step.jump_probability
        q_fraction_trace[index] = q_fraction

        _update_local_statistics(
            local,
            observation=float(value),
            rho=step.jump_probability,
            update=update,
        )
        gamma0_value = local.s0 / local.n0
        q_state, r_state = qr_from_total_variance(
            gamma0_value,
            q_fraction,
            bounds=bounds,
        )
        h_state += update.hazard_rate * (step.jump_probability - h_state)
        h_state = float(np.clip(h_state, *bounds.hazard))
        s0[index] = local.s0
        n0[index] = local.n0
        gamma0[index] = gamma0_value
        mean = step.filtered_mean
        variance = step.filtered_variance
        previous_group = group

    return TotalVarianceFilterTrace(*outputs)


__all__ = [
    "AutocovarianceFilterTrace",
    "AutocovarianceUpdateConfig",
    "TotalVarianceFilterTrace",
    "derive_qr_from_increment_covariance",
    "qr_from_total_variance",
    "run_autocovariance_filter",
    "run_total_variance_filter",
]
