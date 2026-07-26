import numpy as np
import pytest

from src.data.ibl_behavior import (
    IBLBehaviorDataError,
    IBLBehaviorObservations,
)
from src.models.ibl_run_length_observer import (
    RunLengthCandidate,
    SemiMarkovBlockObserver,
)


def _observations(*, seed: int = 7, n_trials: int = 330) -> IBLBehaviorObservations:
    rng = np.random.default_rng(seed)
    probability_left = np.full(n_trials, 0.5, dtype=float)
    start = 90
    state = 0
    while start < n_trials:
        stop = min(start + 40, n_trials)
        probability_left[start:stop] = (0.2, 0.8)[state]
        state = 1 - state
        start = stop
    sides = (rng.random(n_trials) < probability_left).astype(int)
    return IBLBehaviorObservations(np.arange(n_trials), sides)


def test_run_length_observer_is_predictive_causal_and_truth_free() -> None:
    observations = _observations()
    train = np.arange(210)
    prediction = (
        SemiMarkovBlockObserver().fit(observations, train).predict(observations)
    )

    assert prediction.beliefs.shape == (330, 2)
    np.testing.assert_allclose(prediction.beliefs[:90], 0.5)
    np.testing.assert_allclose(prediction.beliefs.sum(axis=1), 1.0)
    assert np.all((prediction.release_probability >= 0.0))
    assert np.all((prediction.release_probability <= 1.0))
    assert np.all(prediction.run_length_ess >= 1.0)
    assert not prediction.beliefs.flags.writeable
    assert set(vars(observations)) == {"trial_ids", "stimulus_side"}
    np.testing.assert_array_equal(prediction.fit_trial_ids, train)

    changed_trial = 250
    altered_sides = observations.stimulus_side.copy()
    altered_sides[changed_trial:] = 1 - altered_sides[changed_trial:]
    altered = IBLBehaviorObservations(observations.trial_ids, altered_sides)
    altered_prediction = (
        SemiMarkovBlockObserver().fit(observations, train).predict(altered)
    )
    for name in (
        "beliefs",
        "release_probability",
        "expected_run_length",
        "run_length_sd",
        "run_length_ess",
        "belief_entropy",
    ):
        first = getattr(prediction, name)
        second = getattr(altered_prediction, name)
        np.testing.assert_allclose(
            first[: changed_trial + 1], second[: changed_trial + 1]
        )


def test_candidate_selection_cannot_read_post_train_observations() -> None:
    observations = _observations(seed=22)
    train = np.arange(210)
    candidates = (
        RunLengthCandidate(20, 100, 20.0),
        RunLengthCandidate(20, 100, 80.0),
    )
    first = SemiMarkovBlockObserver(candidates=candidates).fit(observations, train)

    altered_sides = observations.stimulus_side.copy()
    altered_sides[train.size :] = 1 - altered_sides[train.size :]
    altered = IBLBehaviorObservations(observations.trial_ids, altered_sides)
    second = SemiMarkovBlockObserver(candidates=candidates).fit(altered, train)

    assert first.candidate_ == second.candidate_
    assert first.train_stimulus_nll_ == pytest.approx(second.train_stimulus_nll_)
    first_prediction = first.predict(observations)
    second_prediction = second.predict(altered)
    np.testing.assert_allclose(
        first_prediction.beliefs[: train.size + 1],
        second_prediction.beliefs[: train.size + 1],
    )


def test_release_and_belief_respond_to_sustained_contradictory_evidence() -> None:
    sides = np.concatenate(
        [
            np.resize(np.array([0, 1]), 90),
            np.zeros(80, dtype=int),
            np.ones(80, dtype=int),
        ]
    )
    observations = IBLBehaviorObservations(np.arange(sides.size), sides)
    candidate = RunLengthCandidate(20, 100, 30.0)
    prediction = (
        SemiMarkovBlockObserver(candidates=(candidate,))
        .fit(observations, np.arange(190))
        .predict(observations)
    )

    assert prediction.beliefs[160, 0] > 0.95
    assert prediction.beliefs[210, 1] > 0.95
    assert np.max(prediction.release_probability[170:205]) > 0.5


def test_run_length_observer_fails_closed_on_invalid_contracts() -> None:
    observations = _observations()
    with pytest.raises(ValueError, match="min_run"):
        RunLengthCandidate(20, 20, 30.0)
    with pytest.raises(ValueError, match="unique"):
        SemiMarkovBlockObserver(
            candidates=(
                RunLengthCandidate(20, 100, 30.0),
                RunLengthCandidate(20, 100, 30.0),
            )
        )
    with pytest.raises(IBLBehaviorDataError, match="chronological prefix"):
        SemiMarkovBlockObserver().fit(observations, np.arange(200) + 1)
    with pytest.raises(IBLBehaviorDataError, match="too few biased"):
        SemiMarkovBlockObserver().fit(observations, np.arange(95))
    with pytest.raises(RuntimeError, match="must be fit"):
        SemiMarkovBlockObserver().predict(observations)
