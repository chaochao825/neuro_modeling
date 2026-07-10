"""Rate- and spike-count-matched synthetic phase-dependent communication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.utils.reproducibility import make_rng


PhaseCondition = Literal["in_phase", "anti_phase", "random_phase", "no_oscillation"]


@dataclass(frozen=True)
class PhaseGatingConfig:
    n_trials: int = 120
    n_steps: int = 100
    dt_s: float = 0.01
    frequency_hz: float = 8.0
    gamma: float = 0.6
    delay_steps: int = 2
    coupling: float = 0.35
    target_rate_hz: float = 10.0
    spikes_per_trial: int = 100
    block_size: int = 10
    noise_std: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_trials < 4 or self.n_steps < 8 or self.block_size < 1:
            raise ValueError("n_trials/n_steps are too small or block_size invalid")
        if self.dt_s <= 0 or self.frequency_hz <= 0 or not 0 <= self.gamma < 1:
            raise ValueError("dt/frequency must be positive and gamma in [0, 1)")
        if not 0 <= self.delay_steps < self.n_steps:
            raise ValueError("delay_steps must lie within a trial")
        if self.coupling <= 0 or self.target_rate_hz <= 0 or self.spikes_per_trial < 1:
            raise ValueError("coupling/rate/spike count must be positive")
        if self.noise_std < 0 or self.seed < 0:
            raise ValueError("noise_std and seed must be non-negative")


@dataclass(frozen=True)
class PhaseGatingSimulation:
    condition: PhaseCondition
    source: np.ndarray
    downstream: np.ndarray
    rates_hz: np.ndarray
    spikes: np.ndarray
    coupling_trace: np.ndarray
    labels: np.ndarray
    block_ids: np.ndarray


def _phase_trace(
    condition: PhaseCondition,
    config: PhaseGatingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    time = np.arange(config.n_steps) * config.dt_s
    slow_modulation = 2.0 * np.pi * 1.3 * time
    if condition == "in_phase":
        difference = 0.35 * np.sin(slow_modulation)
    elif condition == "anti_phase":
        difference = np.pi + 0.35 * np.sin(slow_modulation)
    elif condition == "random_phase":
        increments = rng.normal(0.0, 0.6, size=(config.n_trials, config.n_steps))
        difference = np.cumsum(increments, axis=1)
    elif condition == "no_oscillation":
        return np.ones((config.n_trials, config.n_steps), dtype=float)
    else:
        raise ValueError(f"unknown phase condition {condition!r}")
    if np.ndim(difference) == 1:
        difference = np.repeat(np.asarray(difference)[None, :], config.n_trials, axis=0)
    raw = 1.0 + config.gamma * np.cos(
        difference - 2.0 * np.pi * config.frequency_hz * config.delay_steps * config.dt_s
    )
    # Every oscillatory condition has the same mean effective coupling.
    return raw / np.mean(raw)


def simulate_phase_gating(
    condition: PhaseCondition, config: PhaseGatingConfig
) -> PhaseGatingSimulation:
    """Simulate matched-rate conditions with exact per-trial spike totals."""

    phase_rng = make_rng(config.seed, "phase-gating", "phase", condition)
    source_rng = make_rng(config.seed, "phase-gating", "shared-source")
    downstream_rng = make_rng(config.seed, "phase-gating", "shared-downstream-noise")
    spike_rng = make_rng(config.seed, "phase-gating", "shared-spike-sampling")
    labels = np.where(np.arange(config.n_trials) // config.block_size % 2 == 0, 1, -1)
    block_ids = np.arange(config.n_trials) // config.block_size
    source = np.empty((config.n_trials, config.n_steps), dtype=float)
    for trial in range(config.n_trials):
        state = 0.0
        for time in range(config.n_steps):
            drive = labels[trial] * (1.0 if time < config.n_steps // 2 else 0.2)
            state = 0.88 * state + 0.12 * drive + source_rng.normal(0.0, config.noise_std)
            source[trial, time] = state
    coupling_trace = _phase_trace(condition, config, phase_rng)
    downstream = np.zeros_like(source)
    for trial in range(config.n_trials):
        for time in range(1, config.n_steps):
            source_time = max(0, time - config.delay_steps)
            downstream[trial, time] = (
                0.82 * downstream[trial, time - 1]
                + config.coupling * coupling_trace[trial, time] * source[trial, source_time]
                + downstream_rng.normal(0.0, config.noise_std)
            )
    rates = np.logaddexp(0.0, downstream) + 0.05
    rates *= config.target_rate_hz / float(np.mean(rates))
    spikes = np.empty_like(rates, dtype=int)
    for trial in range(config.n_trials):
        probability = rates[trial] / np.sum(rates[trial])
        spikes[trial] = spike_rng.multinomial(config.spikes_per_trial, probability)
    return PhaseGatingSimulation(
        condition=condition,
        source=source,
        downstream=downstream,
        rates_hz=rates,
        spikes=spikes,
        coupling_trace=coupling_trace,
        labels=labels,
        block_ids=block_ids,
    )


def information_transfer_latency(simulation: PhaseGatingSimulation, max_lag: int = 20) -> int:
    """Median within-trial cross-correlation peak lag in bins."""

    lags = []
    for source, downstream in zip(simulation.source, simulation.rates_hz, strict=True):
        centered_source = source - source.mean()
        centered_downstream = downstream - downstream.mean()
        values = []
        for lag in range(max_lag + 1):
            values.append(
                np.dot(centered_source[: len(source) - lag], centered_downstream[lag:])
            )
        lags.append(int(np.argmax(values)))
    return int(np.median(lags))
