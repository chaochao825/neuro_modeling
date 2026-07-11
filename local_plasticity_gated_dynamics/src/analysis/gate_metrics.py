"""Leakage-safe metrics for binary hidden-context belief trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _probability_vector(values: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(values)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric vector")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return array


def _binary_vector(values: ArrayLike, *, name: str, length: int) -> IntArray:
    raw = np.asarray(values)
    if raw.shape != (length,) or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be an integer vector matching probabilities")
    if not np.issubdtype(raw.dtype, np.integer) or not np.isin(raw, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return np.asarray(raw, dtype=np.int64)


def _episode_vector(values: ArrayLike, *, length: int) -> IntArray:
    raw = np.asarray(values)
    if (
        raw.shape != (length,)
        or raw.dtype.kind == "b"
        or not np.issubdtype(raw.dtype, np.integer)
        or np.any(raw < 0)
    ):
        raise ValueError("episode_ids must be matching non-negative integers")
    return np.asarray(raw, dtype=np.int64)


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


@dataclass(frozen=True)
class ContextCalibrationSummary:
    """Proper scoring rules and populated reliability-diagram bins."""

    nll: float
    brier: float
    expected_calibration_error: float
    accuracy: float
    bin_indices: IntArray
    bin_counts: IntArray
    bin_mean_probability: FloatArray
    bin_empirical_frequency: FloatArray


def context_calibration_summary(
    posterior_state1: ArrayLike,
    hidden_states: ArrayLike,
    *,
    n_bins: int = 10,
    epsilon: float = 1e-9,
) -> ContextCalibrationSummary:
    """Evaluate a frozen belief trajectory against truth used only here."""

    probabilities = _probability_vector(posterior_state1, name="posterior_state1")
    states = _binary_vector(
        hidden_states, name="hidden_states", length=probabilities.size
    )
    bins = _positive_int(n_bins, name="n_bins")
    if isinstance(epsilon, (bool, np.bool_)) or not np.isscalar(epsilon):
        raise TypeError("epsilon must be a scalar")
    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value) or not 0.0 < epsilon_value < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")

    clipped = np.clip(probabilities, epsilon_value, 1.0 - epsilon_value)
    nll = -np.mean(states * np.log(clipped) + (1 - states) * np.log1p(-clipped))
    brier = np.mean((probabilities - states) ** 2)
    predicted = (probabilities >= 0.5).astype(np.int64)

    assignments = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    populated = np.unique(assignments)
    counts = np.asarray(
        [np.count_nonzero(assignments == index) for index in populated],
        dtype=np.int64,
    )
    mean_probability = np.asarray(
        [np.mean(probabilities[assignments == index]) for index in populated],
        dtype=np.float64,
    )
    frequency = np.asarray(
        [np.mean(states[assignments == index]) for index in populated],
        dtype=np.float64,
    )
    ece = np.sum(counts * np.abs(mean_probability - frequency)) / probabilities.size
    return ContextCalibrationSummary(
        nll=float(nll),
        brier=float(brier),
        expected_calibration_error=float(ece),
        accuracy=float(np.mean(predicted == states)),
        bin_indices=populated.astype(np.int64, copy=False),
        bin_counts=counts,
        bin_mean_probability=mean_probability,
        bin_empirical_frequency=frequency,
    )


@dataclass(frozen=True)
class SwitchInferenceSummary:
    """Latency after true switches and false switches away from boundaries."""

    mean_latency_trials: float
    median_latency_trials: float
    censored_fraction: float
    false_switch_rate: float
    missed_switch_rate: float
    switch_count: int
    false_switch_count: int
    missed_switch_count: int
    eligible_trial_count: int
    per_switch_latency_trials: IntArray


def switch_inference_summary(
    posterior_state1: ArrayLike,
    hidden_states: ArrayLike,
    episode_ids: ArrayLike,
    *,
    max_latency: int = 5,
    sustain_trials: int = 2,
    posterior_threshold: float = 0.8,
    minimum_state_duration: int = 5,
    match_tolerance: int = 1,
) -> SwitchInferenceSummary:
    """Measure causal tracking without allowing windows to cross episodes."""

    probabilities = _probability_vector(posterior_state1, name="posterior_state1")
    states = _binary_vector(
        hidden_states, name="hidden_states", length=probabilities.size
    )
    episodes = _episode_vector(episode_ids, length=probabilities.size)
    latency_limit = _positive_int(max_latency, name="max_latency")
    sustain = _positive_int(sustain_trials, name="sustain_trials")
    minimum_duration = _positive_int(
        minimum_state_duration, name="minimum_state_duration"
    )
    if isinstance(match_tolerance, (bool, np.bool_)) or not isinstance(
        match_tolerance, (int, np.integer)
    ):
        raise TypeError("match_tolerance must be an integer")
    tolerance = int(match_tolerance)
    if tolerance < 0:
        raise ValueError("match_tolerance must be non-negative")
    if isinstance(posterior_threshold, (bool, np.bool_)) or not np.isscalar(
        posterior_threshold
    ):
        raise TypeError("posterior_threshold must be a scalar")
    threshold = float(posterior_threshold)
    if not np.isfinite(threshold) or not 0.5 < threshold < 1.0:
        raise ValueError("posterior_threshold must lie in (0.5, 1)")
    comparison_slack = 8.0 * np.finfo(np.float64).eps
    lower_threshold = 1.0 - threshold

    same_episode = episodes[1:] == episodes[:-1]
    true_switch = same_episode & (states[1:] != states[:-1])
    raw_switch_indices = np.flatnonzero(true_switch) + 1
    raw_segment_stops: list[int] = []
    switch_indices: list[int] = []
    switch_segment_stops: list[int] = []
    for index in raw_switch_indices.tolist():
        episode = episodes[index]
        stop = index
        while (
            stop < states.size
            and episodes[stop] == episode
            and states[stop] == states[index]
        ):
            stop += 1
        raw_segment_stops.append(stop)
        if stop - index < minimum_duration:
            continue
        switch_indices.append(index)
        switch_segment_stops.append(stop)
    if not switch_indices:
        raise ValueError("hidden state trajectory contains no within-episode switches")
    opportunities = int(np.count_nonzero(same_episode))
    if opportunities == 0:
        raise ValueError("trajectory contains no within-episode trial transitions")

    latencies: list[int] = []
    censored_value = latency_limit + 1
    for switch_index, state_stop in zip(
        switch_indices, switch_segment_stops, strict=True
    ):
        new_state = states[switch_index]
        latest_start = min(state_stop - sustain, switch_index + latency_limit)
        latency = censored_value
        for candidate in range(switch_index, latest_start + 1):
            candidate_probability = probabilities[candidate : candidate + sustain]
            detected = (
                np.all(candidate_probability >= threshold - comparison_slack)
                if new_state == 1
                else np.all(candidate_probability <= lower_threshold + comparison_slack)
            )
            if detected:
                latency = candidate - switch_index
                break
        latencies.append(latency)

    predicted_events: list[tuple[int, int]] = []
    for episode in np.unique(episodes).tolist():
        indices = np.flatnonzero(episodes == episode)
        inferred = int(probabilities[indices[0]] >= 0.5)
        for offset in range(1, indices.size):
            index = int(indices[offset])
            if offset + sustain > indices.size:
                break
            window = probabilities[indices[offset : offset + sustain]]
            if inferred == 0 and np.all(window >= threshold - comparison_slack):
                inferred = 1
                predicted_events.append((index, inferred))
            elif inferred == 1 and np.all(window <= lower_threshold + comparison_slack):
                inferred = 0
                predicted_events.append((index, inferred))

    raw_switch_list = raw_switch_indices.tolist()
    unmatched_true_switches = set(range(len(raw_switch_list)))
    false_switches = 0
    for predicted_index, predicted_state in predicted_events:
        candidates = [
            item
            for item in unmatched_true_switches
            if states[raw_switch_list[item]] == predicted_state
            and raw_switch_list[item] - tolerance
            <= predicted_index
            <= min(
                raw_switch_list[item] + latency_limit,
                raw_segment_stops[item] - sustain,
            )
        ]
        if not candidates:
            false_switches += 1
            continue
        matched = min(
            candidates,
            key=lambda item: abs(raw_switch_list[item] - predicted_index),
        )
        unmatched_true_switches.remove(matched)
    latency_array = np.asarray(latencies, dtype=np.int64)
    missed_switches = int(np.count_nonzero(latency_array == censored_value))
    return SwitchInferenceSummary(
        mean_latency_trials=float(np.mean(latency_array)),
        median_latency_trials=float(np.median(latency_array)),
        censored_fraction=float(np.mean(latency_array == censored_value)),
        false_switch_rate=float(false_switches / opportunities),
        missed_switch_rate=float(missed_switches / len(switch_indices)),
        switch_count=len(switch_indices),
        false_switch_count=false_switches,
        missed_switch_count=missed_switches,
        eligible_trial_count=opportunities,
        per_switch_latency_trials=latency_array,
    )


@dataclass(frozen=True)
class GatedBehaviorSummary:
    """Behavior obtained by mixing two evidence streams with frozen belief."""

    accuracy: float
    balanced_accuracy: float
    scores: FloatArray
    predicted_choices: IntArray


def gated_behavior_summary(
    posterior_state1: ArrayLike,
    evidence: ArrayLike,
    target_choices: ArrayLike,
) -> GatedBehaviorSummary:
    """Score behavior after gate inference; truth is not fed back into the gate."""

    probabilities = _probability_vector(posterior_state1, name="posterior_state1")
    raw_evidence = np.asarray(evidence)
    if raw_evidence.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("evidence must be a real numeric matrix")
    values = np.asarray(raw_evidence, dtype=np.float64)
    if values.shape != (probabilities.size, 2) or not np.isfinite(values).all():
        raise ValueError("evidence must have shape [trial, 2] and be finite")
    raw_targets = np.asarray(target_choices)
    if (
        raw_targets.shape != (probabilities.size,)
        or raw_targets.dtype.kind == "b"
        or not np.issubdtype(raw_targets.dtype, np.integer)
        or not np.isin(raw_targets, [-1, 1]).all()
    ):
        raise ValueError("target_choices must contain matching -1/+1 integers")
    scores = (1.0 - probabilities) * values[:, 0] + probabilities * values[:, 1]
    predicted = np.where(scores >= 0.0, 1, -1).astype(np.int64)
    class_recalls = [
        np.mean(predicted[raw_targets == label] == label)
        for label in np.unique(raw_targets)
    ]
    return GatedBehaviorSummary(
        accuracy=float(np.mean(predicted == raw_targets)),
        balanced_accuracy=float(np.mean(class_recalls)),
        scores=np.asarray(scores, dtype=np.float64),
        predicted_choices=predicted,
    )


@dataclass(frozen=True)
class GateEnergySummary:
    """Transparent state-occupation and transition energy proxies."""

    state_energy: float
    transition_energy: float
    total_energy: float
    transition_count: int


def gate_energy_summary(
    posterior_state1: ArrayLike,
    episode_ids: ArrayLike,
) -> GateEnergySummary:
    """Compute belief activity and within-episode update magnitudes."""

    probabilities = _probability_vector(posterior_state1, name="posterior_state1")
    episodes = _episode_vector(episode_ids, length=probabilities.size)
    beliefs = np.column_stack([1.0 - probabilities, probabilities])
    state_energy = float(np.mean(np.sum(beliefs**2, axis=1)))
    same_episode = episodes[1:] == episodes[:-1]
    transitions = np.abs(np.diff(beliefs, axis=0))[same_episode]
    transition_energy = (
        float(np.mean(np.sum(transitions, axis=1))) if transitions.size else 0.0
    )
    return GateEnergySummary(
        state_energy=state_energy,
        transition_energy=transition_energy,
        total_energy=state_energy + transition_energy,
        transition_count=int(np.count_nonzero(same_episode)),
    )
