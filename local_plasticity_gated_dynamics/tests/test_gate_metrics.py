from __future__ import annotations

import numpy as np
import pytest

from src.analysis.gate_metrics import (
    context_calibration_summary,
    gate_energy_summary,
    gated_behavior_summary,
    switch_inference_summary,
)


def test_perfect_context_belief_has_near_zero_proper_scores() -> None:
    states = np.array([0, 0, 1, 1, 0, 1])
    posterior = np.where(states == 1, 1.0, 0.0)
    summary = context_calibration_summary(posterior, states, n_bins=5)

    assert summary.nll < 1e-8
    assert summary.brier == 0.0
    assert summary.expected_calibration_error == 0.0
    assert summary.accuracy == 1.0
    assert summary.bin_counts.sum() == states.size


def test_switch_metrics_respect_episode_boundaries_and_censor_latency() -> None:
    states = np.array([0, 0, 1, 1, 1, 0, 0, 0])
    episodes = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    posterior = np.array([0.1, 0.1, 0.2, 0.9, 0.9, 0.8, 0.2, 0.1])
    summary = switch_inference_summary(
        posterior,
        states,
        episodes,
        max_latency=2,
        sustain_trials=1,
        minimum_state_duration=1,
    )

    # Switches occur at indices 2 and 5; index 4 is an episode reset, not a switch.
    np.testing.assert_array_equal(summary.per_switch_latency_trials, [1, 1])
    assert summary.switch_count == 2
    assert summary.censored_fraction == 0.0
    assert summary.false_switch_rate == 0.0


def test_switch_latency_does_not_cross_the_next_true_switch() -> None:
    states = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    episodes = np.zeros(states.size, dtype=int)
    posterior = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.9, 0.9, 0.1, 0.1, 0.1])
    summary = switch_inference_summary(posterior, states, episodes)

    # The state-1 posterior crosses threshold only after state 1 has ended.
    # It must not be credited to the switch at index 2.
    assert summary.per_switch_latency_trials[0] == 6


def test_causal_detection_latency_is_not_double_counted_as_false_and_missed() -> None:
    states = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1])
    episodes = np.zeros(states.size, dtype=int)
    posterior = np.array([0.1, 0.1, 0.2, 0.4, 0.9, 0.9, 0.9, 0.9, 0.9])
    summary = switch_inference_summary(posterior, states, episodes)

    assert summary.per_switch_latency_trials.tolist() == [2]
    assert summary.false_switch_count == 0
    assert summary.missed_switch_count == 0


def test_perfect_belief_has_no_false_or_missed_switches_with_short_runs() -> None:
    states = np.array(
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        dtype=int,
    )
    episodes = np.zeros(states.size, dtype=int)
    posterior = states.astype(float)
    summary = switch_inference_summary(posterior, states, episodes)

    assert summary.mean_latency_trials == 0.0
    assert summary.false_switch_count == 0
    assert summary.missed_switch_count == 0


def test_behavior_uses_posterior_to_select_relevant_evidence() -> None:
    posterior = np.array([0.0, 1.0, 0.0, 1.0])
    evidence = np.array([[2.0, -3.0], [-2.0, 3.0], [-1.0, 4.0], [4.0, -1.0]])
    targets = np.array([1, 1, -1, -1])
    summary = gated_behavior_summary(posterior, evidence, targets)

    assert summary.accuracy == 1.0
    assert summary.balanced_accuracy == 1.0
    np.testing.assert_array_equal(summary.predicted_choices, targets)


def test_gate_energy_excludes_episode_reset_transitions() -> None:
    posterior = np.array([0.1, 0.2, 0.9, 0.8])
    episodes = np.array([0, 0, 1, 1])
    summary = gate_energy_summary(posterior, episodes)

    assert summary.transition_count == 2
    assert summary.transition_energy == pytest.approx(0.2)
    assert summary.total_energy == pytest.approx(
        summary.state_energy + summary.transition_energy
    )


@pytest.mark.parametrize(
    ("posterior", "states"),
    [
        (np.array([0.2, np.nan]), np.array([0, 1])),
        (np.array([0.2, 1.2]), np.array([0, 1])),
        (np.array([0.2, 0.8]), np.array([0, 2])),
    ],
)
def test_context_metric_validation(posterior: np.ndarray, states: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        context_calibration_summary(posterior, states)
