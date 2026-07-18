"""Trial-stream metrics for persistent sparse-reward actuator selection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _vector(value: ArrayLike, *, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a vector of length at least two")
    if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def stream_accuracy(rewards: ArrayLike) -> float:
    values = _vector(rewards, name="rewards", dtype=np.dtype(np.float64))
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("rewards must lie in [0, 1]")
    return float(np.mean(values))


def binary_belief_scores(
    hidden_states: ArrayLike,
    belief_probability_state_one: ArrayLike,
) -> dict[str, float]:
    states = _vector(hidden_states, name="hidden_states", dtype=np.dtype(np.int64))
    probability = _vector(
        belief_probability_state_one,
        name="belief_probability_state_one",
        dtype=np.dtype(np.float64),
    )
    if probability.shape != states.shape:
        raise ValueError("hidden states and belief probabilities must be paired")
    if np.any((states < 0) | (states > 1)):
        raise ValueError("hidden states must be binary")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("belief probabilities must lie in [0, 1]")
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    nll = -np.mean(states * np.log(clipped) + (1 - states) * np.log(1.0 - clipped))
    brier = np.mean(np.square(probability - states))
    choice = (probability >= 0.5).astype(np.int64)
    return {
        "context_nll": float(nll),
        "context_brier": float(brier),
        "belief_state_accuracy": float(np.mean(choice == states)),
    }


def switch_diagnostics(
    hidden_states: ArrayLike,
    belief_probability_state_one: ArrayLike,
    *,
    stability_window: int = 3,
) -> dict[str, float]:
    states = _vector(hidden_states, name="hidden_states", dtype=np.dtype(np.int64))
    probability = _vector(
        belief_probability_state_one,
        name="belief_probability_state_one",
        dtype=np.dtype(np.float64),
    )
    if probability.shape != states.shape:
        raise ValueError("hidden states and belief probabilities must be paired")
    if isinstance(stability_window, (bool, np.bool_)) or not isinstance(
        stability_window, (int, np.integer)
    ):
        raise TypeError("stability_window must be an integer")
    stability_window = int(stability_window)
    if stability_window < 1:
        raise ValueError("stability_window must be positive")
    predicted = (probability >= 0.5).astype(np.int64)
    switches = np.flatnonzero(states[1:] != states[:-1]) + 1
    latencies: list[int] = []
    for index, start in enumerate(switches):
        stop = int(switches[index + 1]) if index + 1 < switches.size else states.size
        target = int(states[start])
        latency = stop - int(start)
        for trial in range(int(start), stop):
            window_stop = min(stop, trial + stability_window)
            if window_stop - trial < stability_window:
                break
            if np.all(predicted[trial:window_stop] == target):
                latency = trial - int(start)
                break
        latencies.append(latency)
    predicted_switch = predicted[1:] != predicted[:-1]
    true_stable = states[1:] == states[:-1]
    false_switch_count = int(np.count_nonzero(predicted_switch & true_stable))
    stable_count = int(np.count_nonzero(true_stable))
    return {
        "hidden_switch_count": float(switches.size),
        "median_switch_latency": (
            float(np.median(latencies)) if latencies else float("nan")
        ),
        "mean_switch_latency": (
            float(np.mean(latencies)) if latencies else float("nan")
        ),
        "false_switch_rate": (
            float(false_switch_count / stable_count) if stable_count else float("nan")
        ),
        "belief_transition_count": float(np.count_nonzero(predicted_switch)),
    }


def post_switch_cost(
    hidden_states: ArrayLike,
    rewards: ArrayLike,
    *,
    window: int = 8,
) -> float:
    states = _vector(hidden_states, name="hidden_states", dtype=np.dtype(np.int64))
    reward = _vector(rewards, name="rewards", dtype=np.dtype(np.float64))
    if reward.shape != states.shape:
        raise ValueError("hidden states and rewards must be paired")
    if isinstance(window, (bool, np.bool_)) or not isinstance(
        window, (int, np.integer)
    ):
        raise TypeError("window must be an integer")
    window = int(window)
    if window < 1:
        raise ValueError("window must be positive")
    switch = np.zeros(states.size, dtype=bool)
    switch[1:] = states[1:] != states[:-1]
    post = np.zeros(states.size, dtype=bool)
    for start in np.flatnonzero(switch):
        post[start : min(states.size, start + window)] = True
    if not np.any(post) or np.all(post):
        return float("nan")
    return float(np.mean(reward[~post]) - np.mean(reward[post]))


__all__ = [
    "binary_belief_scores",
    "post_switch_cost",
    "stream_accuracy",
    "switch_diagnostics",
]
