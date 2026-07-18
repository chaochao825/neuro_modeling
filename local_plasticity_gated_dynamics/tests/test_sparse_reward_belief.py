from __future__ import annotations

import numpy as np
import pytest

from src.plasticity.sparse_reward_belief import (
    BayesianDelayedRewardFilter,
    PersistentRewardBeliefSelector,
)


def _empty_trial(selector: PersistentRewardBeliefSelector, trial: int) -> None:
    selector.advance(trial)
    action = selector.choose(0.0)
    selector.schedule_feedback(
        source_trial=trial,
        executed_action=action,
        reward=0.0,
        available=False,
        delay=0,
    )


def test_local_selector_delivers_only_after_registered_delay() -> None:
    selector = PersistentRewardBeliefSelector(
        alpha=0.5, retention=1.0, temperature=0.1, q_prior=0.5, seed=1
    )
    selector.advance(0)
    action = selector.choose(0.0)
    assert action == 1
    selector.schedule_feedback(
        source_trial=0,
        executed_action=1,
        reward=1.0,
        available=True,
        delay=2,
    )
    _empty_trial(selector, 1)
    _empty_trial(selector, 2)
    np.testing.assert_allclose(selector.q_values, [0.5, 0.5])
    selector.advance(3)
    assert selector.q_values[1] == 0.75
    action = selector.choose(0.0)
    selector.schedule_feedback(
        source_trial=3,
        executed_action=action,
        reward=1.0,
        available=False,
        delay=0,
    )
    receipt = selector.receipt()
    assert receipt.delivered_source_trials.tolist() == [0]
    assert receipt.used_true_context is False
    assert receipt.used_counterfactual_reward is False
    assert receipt.used_autograd is False
    assert receipt.used_bptt is False


def test_credit_shuffle_preserves_reward_but_breaks_action_eligibility() -> None:
    selector = PersistentRewardBeliefSelector(
        alpha=0.5,
        retention=1.0,
        temperature=0.1,
        q_prior=0.5,
        seed=2,
        credit_assignment="opposite",
    )
    selector.advance(0)
    action = selector.choose(0.0)
    selector.schedule_feedback(
        source_trial=0,
        executed_action=action,
        reward=1.0,
        available=True,
        delay=0,
    )
    selector.advance(1)
    assert selector.q_values[0] == 0.75
    assert selector.q_values[1] == 0.5
    selector.choose(0.0)
    selector.schedule_feedback(
        source_trial=1,
        executed_action=1,
        reward=0.0,
        available=False,
        delay=0,
    )
    assert selector.receipt().method == "opposite_eligibility_local"


def test_receipt_separates_reward_updates_from_retention_drift() -> None:
    selector = PersistentRewardBeliefSelector(
        alpha=0.5, retention=0.5, temperature=0.1, q_prior=0.5, seed=7
    )
    selector.advance(0)
    action = selector.choose(0.0)
    selector.schedule_feedback(
        source_trial=0,
        executed_action=action,
        reward=1.0,
        available=True,
        delay=0,
    )
    selector.advance(1)
    selector.choose(0.0)
    selector.schedule_feedback(
        source_trial=1,
        executed_action=1,
        reward=0.0,
        available=False,
        delay=0,
    )
    selector.advance(2)
    selector.choose(0.0)
    selector.schedule_feedback(
        source_trial=2,
        executed_action=1,
        reward=0.0,
        available=False,
        delay=0,
    )
    receipt = selector.receipt()
    assert receipt.cumulative_update_l1 > 0.0
    assert receipt.cumulative_retention_l1 > 0.0
    assert receipt.cumulative_total_state_update_l1 == pytest.approx(
        receipt.cumulative_update_l1 + receipt.cumulative_retention_l1
    )
    assert receipt.internal_state_dimension == 2
    assert receipt.control_dimension == 1


def test_bayesian_filter_uses_delivered_executed_reward_not_true_state() -> None:
    selector = BayesianDelayedRewardFilter(
        np.array([[0.9, 0.55], [0.55, 0.9]]),
        hazard=0.05,
        temperature=0.1,
        seed=4,
    )
    for trial in range(5):
        selector.advance(trial)
        action = selector.choose(0.0)
        assert action == 1
        selector.schedule_feedback(
            source_trial=trial,
            executed_action=action,
            reward=1.0,
            available=True,
            delay=0,
        )
    receipt = selector.receipt()
    assert receipt.belief_probabilities[-1] > 0.8
    assert receipt.used_true_context is False
    assert receipt.used_counterfactual_reward is False
    assert receipt.delivered_rewards.size == 4


def test_delayed_bayesian_filter_matches_independent_forward_recompute() -> None:
    emission = np.array([[0.88, 0.58], [0.52, 0.91]], dtype=float)
    hazard = 0.07
    transition = np.array([[1.0 - hazard, hazard], [hazard, 1.0 - hazard]])
    selector = BayesianDelayedRewardFilter(
        emission, hazard=hazard, temperature=0.13, seed=9
    )
    uniforms = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    rewards = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    available = np.array([True, False, True, True, False, True, True, False])
    delay = 2
    observations: dict[int, tuple[int, float]] = {}
    scheduled: dict[int, tuple[int, int, float]] = {}

    def brute_prior(trial: int) -> np.ndarray:
        posterior: np.ndarray | None = None
        for source in range(trial):
            prior = np.array([0.5, 0.5]) if source == 0 else posterior @ transition
            observation = observations.get(source)
            if observation is None:
                posterior = prior
            else:
                action, reward = observation
                likelihood = (
                    emission[:, action] if reward >= 0.5 else 1.0 - emission[:, action]
                )
                posterior = prior * likelihood
                posterior /= posterior.sum()
        return np.array([0.5, 0.5]) if trial == 0 else posterior @ transition

    expected = []
    for trial in range(len(uniforms)):
        if trial in scheduled:
            source, action, reward = scheduled[trial]
            observations[source] = (action, reward)
        selector.advance(trial)
        action = selector.choose(float(uniforms[trial]))
        expected.append(float(brute_prior(trial)[1]))
        selector.schedule_feedback(
            source_trial=trial,
            executed_action=action,
            reward=float(rewards[trial]),
            available=bool(available[trial]),
            delay=delay,
        )
        if available[trial]:
            scheduled[trial + delay + 1] = (trial, action, float(rewards[trial]))
    np.testing.assert_allclose(selector.receipt().belief_probabilities, expected)
