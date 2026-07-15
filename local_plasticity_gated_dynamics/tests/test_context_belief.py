from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.models.context_belief import (
    LearnedSymmetricHMM,
    MDRecurrentBeliefGate,
    NoGate,
    OracleBayesianFilter,
    SupervisedCueGate,
    deranged_trajectory_shuffle,
    episode_delay,
    neutral_clamp,
    _posterior,
)
from src.tasks.hidden_context import GateObservationBatch


PROJECT = Path(__file__).resolve().parents[1]


def _observations(cues: np.ndarray, *, episode_length: int) -> GateObservationBatch:
    cues = np.asarray(cues, dtype=int)
    if cues.size % episode_length:
        raise ValueError("test cues must contain complete equal-length episodes")
    n_episodes = cues.size // episode_length
    within = np.tile(np.arange(episode_length), n_episodes)
    return GateObservationBatch(
        cue_observations=cues,
        trial_ids=np.arange(cues.size),
        episode_ids=np.repeat(np.arange(n_episodes), episode_length),
        episode_trial_indices=within,
        episode_start=within == 0,
    )


def _simulate_hmm(
    *,
    seed: int,
    n_episodes: int,
    episode_length: int,
    hazard: float,
    reliability: float,
) -> tuple[GateObservationBatch, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.empty((n_episodes, episode_length), dtype=int)
    states[:, 0] = rng.integers(0, 2, size=n_episodes)
    for time in range(1, episode_length):
        switch = rng.random(n_episodes) < hazard
        states[:, time] = np.where(switch, 1 - states[:, time - 1], states[:, time - 1])
    correct = rng.random(states.shape) < reliability
    cues = np.where(correct, states, 1 - states).reshape(-1)
    return _observations(cues, episode_length=episode_length), states.reshape(-1)


def test_oracle_filter_matches_hand_calculation_and_resets_only_at_episode() -> None:
    observations = _observations(np.array([1, 1, 0, 0, 1, 1]), episode_length=3)
    gate = OracleBayesianFilter(0.1, 0.8, seed=4)
    prediction = gate.predict(observations)

    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    emission = np.array([[0.8, 0.2], [0.2, 0.8]])
    expected = np.empty((6, 2))
    for start in (0, 3):
        belief = np.array([0.5, 0.5])
        for offset in range(3):
            index = start + offset
            prior = belief if offset == 0 else transition.T @ belief
            belief = prior * emission[:, observations.cue_observations[index]]
            belief /= belief.sum()
            expected[index] = belief

    np.testing.assert_allclose(prediction.beliefs, expected)
    assert prediction.fit_trial_ids.size == 0
    assert prediction.audit_metadata()["gate_test_accessed_true_context"] is False
    assert not prediction.beliefs.flags.writeable


def test_inverse_temperature_scales_only_the_cue_likelihood() -> None:
    prior = np.array([0.8, 0.2])
    emission = np.array([[0.6, 0.4], [0.4, 0.6]])
    posterior = _posterior(prior, emission, cue=0, temperature=2.0)
    expected = prior * emission[:, 0] ** 2
    expected /= expected.sum()

    np.testing.assert_allclose(posterior, expected)


def test_oracle_perfect_cue_and_impossible_sequence_are_explicit() -> None:
    observation = _observations(np.array([0, 1]), episode_length=1)
    prediction = OracleBayesianFilter(0.0, 1.0).predict(observation)
    np.testing.assert_array_equal(
        prediction.beliefs, np.array([[1.0, 0.0], [0.0, 1.0]])
    )

    impossible = _observations(np.array([0, 1]), episode_length=2)
    with pytest.raises(ValueError, match="zero probability"):
        OracleBayesianFilter(0.0, 1.0).predict(impossible)


def test_learned_hmm_is_unsupervised_deterministic_and_causal_at_test() -> None:
    train, _ = _simulate_hmm(
        seed=11,
        n_episodes=24,
        episode_length=60,
        hazard=0.08,
        reliability=0.85,
    )
    test, _ = _simulate_hmm(
        seed=12,
        n_episodes=4,
        episode_length=25,
        hazard=0.08,
        reliability=0.85,
    )
    options = dict(
        max_iter=60,
        initial_hazards=(0.03, 0.15),
        initial_reliabilities=(0.65, 0.9),
        seed=7,
    )
    first = LearnedSymmetricHMM(**options).fit(train)
    second = LearnedSymmetricHMM(**options).fit(train)

    assert 0.0 < first.context_hazard_ < 0.5
    assert 0.5 < first.cue_reliability_ < 1.0
    np.testing.assert_allclose(
        np.diff(first.log_likelihood_history_),
        np.maximum(np.diff(first.log_likelihood_history_), 0.0),
        atol=1e-7,
    )
    assert first.context_hazard_ == second.context_hazard_
    assert first.cue_reliability_ == second.cue_reliability_
    np.testing.assert_array_equal(first.fit_trial_ids_, train.trial_ids)
    np.testing.assert_array_equal(first.fit_episode_ids_, np.unique(train.episode_ids))

    first_prediction = first.predict(test)
    second_prediction = second.predict(test)
    np.testing.assert_array_equal(first_prediction.beliefs, second_prediction.beliefs)
    assert first_prediction.fit_accessed_true_context is False
    assert first_prediction.test_accessed_true_context is False


def test_md_recurrent_gate_uses_local_cue_only_fit_and_freezes_for_prediction() -> None:
    train, _ = _simulate_hmm(
        seed=20,
        n_episodes=8,
        episode_length=30,
        hazard=0.1,
        reliability=0.8,
    )
    options = dict(
        learning_rate=0.03,
        inverse_temperature=1.2,
        pseudocount=0.05,
        n_passes=2,
        seed=9,
    )
    first = MDRecurrentBeliefGate(**options).fit(train)
    second = MDRecurrentBeliefGate(**options).fit(train)
    np.testing.assert_array_equal(first.transition_, second.transition_)
    np.testing.assert_array_equal(first.emission_, second.emission_)
    np.testing.assert_allclose(first.transition_.sum(axis=1), 1.0)
    np.testing.assert_allclose(first.emission_.sum(axis=1), 1.0)
    np.testing.assert_allclose(first.transition_[0], first.transition_[1, ::-1])
    np.testing.assert_allclose(first.emission_[0], first.emission_[1, ::-1])
    assert np.all(first.transition_ > 0.0)
    assert np.all(first.emission_ > 0.0)
    assert first.local_update_l1_ > 0.0

    before_transition = first.transition_.copy()
    before_emission = first.emission_.copy()
    prediction = first.predict(train)
    np.testing.assert_array_equal(first.transition_, before_transition)
    np.testing.assert_array_equal(first.emission_, before_emission)
    assert prediction.fit_accessed_true_context is False
    assert prediction.audit_metadata()["local_update_l1"] == pytest.approx(
        first.local_update_l1_
    )
    assert (
        prediction.audit_metadata()["local_update_rule"]
        == "causal_two_slice_with_hebbian_moment_shrinkage"
    )


def test_md_predictive_prior_excludes_current_and_future_cues() -> None:
    train, _ = _simulate_hmm(
        seed=41,
        n_episodes=8,
        episode_length=30,
        hazard=0.1,
        reliability=0.8,
    )
    gate = MDRecurrentBeliefGate(
        learning_rate=0.03,
        inverse_temperature=1.2,
        pseudocount=0.05,
        n_passes=2,
        seed=12,
    ).fit(train)
    cues = np.array([1, 1, 0, 1, 0, 0], dtype=int)
    changed = cues.copy()
    changed[2:] = 1 - changed[2:]
    intact = gate.predict_prior(_observations(cues, episode_length=6))
    counterfactual = gate.predict_prior(
        _observations(changed, episode_length=6)
    )

    # Trial 2 is frozen before cue 2, so altering cue 2 and every later cue
    # cannot change any predictive prior through trial 2.
    np.testing.assert_array_equal(intact.beliefs[:3], counterfactual.beliefs[:3])
    assert not np.array_equal(intact.beliefs[3], counterfactual.beliefs[3])

    current_cue_posterior = gate.predict(_observations(cues, episode_length=6))
    assert not np.array_equal(
        intact.beliefs[0], current_cue_posterior.beliefs[0]
    )
    np.testing.assert_array_equal(intact.beliefs[0], np.array([0.5, 0.5]))


def test_md_predictive_prior_resets_each_episode_and_preserves_provenance() -> None:
    train, _ = _simulate_hmm(
        seed=42,
        n_episodes=8,
        episode_length=30,
        hazard=0.1,
        reliability=0.8,
    )
    gate = MDRecurrentBeliefGate(
        learning_rate=0.03,
        inverse_temperature=1.2,
        pseudocount=0.05,
        n_passes=2,
        seed=13,
    ).fit(train)
    observations = _observations(
        np.array([1, 1, 1, 0, 0, 0], dtype=int), episode_length=3
    )
    prediction = gate.predict_prior(observations)
    metadata = prediction.audit_metadata()

    np.testing.assert_array_equal(
        prediction.beliefs[[0, 3]], np.full((2, 2), 0.5)
    )
    assert prediction.gate_name == "md_recurrent_belief_predictive_prior"
    np.testing.assert_array_equal(prediction.fit_trial_ids, gate.fit_trial_ids_)
    np.testing.assert_array_equal(prediction.fit_episode_ids, gate.fit_episode_ids_)
    assert prediction.fit_accessed_true_context is False
    assert prediction.test_accessed_true_context is False
    assert metadata["belief_timing"] == "predictive_prior_before_current_cue"
    assert metadata["observation_window"] == "strictly_before_current_trial"
    assert metadata["current_cue_accessed_for_same_trial"] is False
    assert metadata["future_cues_accessed"] is False
    assert metadata["fit_observation_fingerprint"] == gate.fit_observation_fingerprint_
    assert metadata["prediction_fingerprint"] == prediction.fingerprint
    assert gate.predictive_prior(observations).fingerprint == prediction.fingerprint


def test_md_local_learning_does_not_collapse_on_long_training_sequences() -> None:
    short, _ = _simulate_hmm(
        seed=40,
        n_episodes=8,
        episode_length=80,
        hazard=0.1,
        reliability=0.85,
    )
    long, _ = _simulate_hmm(
        seed=40,
        n_episodes=60,
        episode_length=200,
        hazard=0.1,
        reliability=0.85,
    )
    heldout, states = _simulate_hmm(
        seed=41,
        n_episodes=20,
        episode_length=100,
        hazard=0.1,
        reliability=0.85,
    )
    options = dict(
        learning_rate=0.03,
        inverse_temperature=1.2,
        pseudocount=0.05,
        n_passes=2,
        seed=9,
    )
    short_gate = MDRecurrentBeliefGate(**options).fit(short)
    long_gate = MDRecurrentBeliefGate(**options).fit(long)

    def nll(gate: MDRecurrentBeliefGate) -> float:
        probability = np.clip(gate.predict(heldout).context_probability, 1e-6, 1 - 1e-6)
        return float(
            -np.mean(
                states * np.log(probability) + (1 - states) * np.log1p(-probability)
            )
        )

    assert nll(long_gate) < np.log(2.0) - 0.05
    assert nll(long_gate) <= nll(short_gate) + 0.02
    assert long_gate.transition_[0, 0] > 0.55
    assert long_gate.emission_[0, 0] > 0.55


def test_md_multilag_shrinkage_avoids_low_hazard_boundary_artifact() -> None:
    train, _ = _simulate_hmm(
        seed=50,
        n_episodes=30,
        episode_length=200,
        hazard=0.05,
        reliability=0.70,
    )
    heldout, states = _simulate_hmm(
        seed=51,
        n_episodes=10,
        episode_length=100,
        hazard=0.05,
        reliability=0.70,
    )
    gate = MDRecurrentBeliefGate(
        learning_rate=0.03,
        inverse_temperature=1.2,
        pseudocount=0.05,
        n_passes=2,
        seed=4,
    ).fit(train)
    probability = np.clip(gate.predict(heldout).context_probability, 1e-6, 1.0 - 1e-6)
    nll = float(
        -np.mean(states * np.log(probability) + (1 - states) * np.log1p(-probability))
    )

    assert 0.005 < gate.transition_[0, 1] < 0.25
    assert gate.emission_[0, 0] > 0.60
    assert nll < 0.65


def test_md_registered_q_h_grid_improves_over_neutral_on_average() -> None:
    options = dict(
        learning_rate=0.03,
        inverse_temperature=1.2,
        pseudocount=0.05,
        n_passes=2,
        seed=5,
    )
    for seed in (60, 61):
        nll_values: list[float] = []
        for reliability in (0.55, 0.70, 0.85, 1.0):
            for hazard in (0.01, 0.05, 0.10, 0.20):
                train, _ = _simulate_hmm(
                    seed=seed,
                    n_episodes=20,
                    episode_length=100,
                    hazard=hazard,
                    reliability=reliability,
                )
                heldout, states = _simulate_hmm(
                    seed=seed + 100,
                    n_episodes=6,
                    episode_length=100,
                    hazard=hazard,
                    reliability=reliability,
                )
                gate = MDRecurrentBeliefGate(**options).fit(train)
                probability = np.clip(
                    gate.predict(heldout).context_probability,
                    1e-6,
                    1.0 - 1e-6,
                )
                nll_values.append(
                    float(
                        -np.mean(
                            states * np.log(probability)
                            + (1 - states) * np.log1p(-probability)
                        )
                    )
                )
        assert float(np.mean(nll_values)) < np.log(2.0) - 0.08


def test_supervised_gate_truth_is_confined_to_fit_supervised() -> None:
    train, hidden = _simulate_hmm(
        seed=30,
        n_episodes=10,
        episode_length=25,
        hazard=0.08,
        reliability=0.8,
    )
    gate = SupervisedCueGate(C=1.0, trace_decays=(0.5, 0.9), seed=3)
    gate.fit_supervised(train, hidden)
    prediction = gate.predict(train)

    assert list(inspect.signature(SupervisedCueGate.predict).parameters) == [
        "self",
        "observations",
    ]
    assert prediction.fit_accessed_true_context is True
    assert prediction.test_accessed_true_context is False
    assert gate.audit_metadata()["evidence_role"] == "supervised_upper_bound"
    assert np.all(
        (prediction.context_probability > 0.0) & (prediction.context_probability < 1.0)
    )


def test_no_gate_is_exactly_neutral() -> None:
    observations = _observations(np.array([0, 1, 1, 0]), episode_length=2)
    prediction = NoGate(seed=5).predict(observations)
    np.testing.assert_array_equal(prediction.beliefs, np.full((4, 2), 0.5))
    assert prediction.fit_trial_ids.size == 0
    assert prediction.test_accessed_true_context is False


def test_clamp_delay_and_deranged_shuffle_preserve_base_fit_provenance() -> None:
    observations = _observations(
        np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1]),
        episode_length=4,
    )
    base = OracleBayesianFilter(0.1, 0.8).predict(observations)

    clamped = neutral_clamp(base)
    np.testing.assert_array_equal(clamped.beliefs, np.full((12, 2), 0.5))
    np.testing.assert_array_equal(clamped.source_trial_ids, np.full(12, -1))

    delayed = episode_delay(base, 2)
    for start in (0, 4, 8):
        np.testing.assert_array_equal(delayed.beliefs[start : start + 2], 0.5)
        np.testing.assert_array_equal(
            delayed.beliefs[start + 2 : start + 4], base.beliefs[start : start + 2]
        )
        np.testing.assert_array_equal(delayed.source_episode_ids[start : start + 2], -1)

    shuffled = deranged_trajectory_shuffle(base, seed=17)
    np.testing.assert_allclose(
        np.sort(shuffled.context_probability), np.sort(base.context_probability)
    )
    assert np.all(shuffled.source_episode_ids != shuffled.episode_ids)
    assert shuffled.base_prediction_fingerprint == base.fingerprint
    again = deranged_trajectory_shuffle(base, seed=17)
    np.testing.assert_array_equal(shuffled.beliefs, again.beliefs)
    np.testing.assert_array_equal(shuffled.source_trial_ids, again.source_trial_ids)

    for intervention in (clamped, delayed, shuffled):
        np.testing.assert_array_equal(intervention.fit_trial_ids, base.fit_trial_ids)
        assert intervention.gate_name == base.gate_name
        assert intervention.base_prediction_fingerprint == base.fingerprint


def test_observation_validation_rejects_reappearing_episode_and_nonbinary_cue() -> None:
    bad_episode = SimpleNamespace(
        cues=np.array([0, 1, 0]),
        episode_ids=np.array([0, 1, 0]),
        trial_in_episode=np.array([0, 0, 0]),
    )
    with pytest.raises(ValueError, match="contiguous"):
        NoGate().predict(bad_episode)

    bad_cue = SimpleNamespace(
        cues=np.array([0, 2]),
        episode_ids=np.array([0, 0]),
        trial_in_episode=np.array([0, 1]),
    )
    with pytest.raises(ValueError, match="binary"):
        NoGate().predict(bad_cue)

    observations = _observations(np.array([0, 1]), episode_length=2)
    with pytest.raises(RuntimeError, match="fit before"):
        LearnedSymmetricHMM().predict(observations)
    with pytest.raises(RuntimeError, match="fit before"):
        MDRecurrentBeliefGate().predict(observations)
    with pytest.raises(RuntimeError, match="fit before"):
        MDRecurrentBeliefGate().predict_prior(observations)
    with pytest.raises(RuntimeError, match="fit before"):
        SupervisedCueGate().predict(observations)


def test_context_belief_module_has_no_torch_or_autograd_dependency() -> None:
    source = (PROJECT / "src" / "models" / "context_belief.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "torch" or name.startswith("torch.") for name in imports)
    assert "torch.autograd" not in source
    assert ".backward(" not in source
