from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from src.tasks.hidden_context import (
    GateObservationBatch,
    HiddenContextConfig,
    generate_hidden_context,
    make_hidden_context_random_tape,
)


def _small_config(**overrides: object) -> HiddenContextConfig:
    options: dict[str, object] = {
        "n_episodes": 8,
        "trials_per_episode": 12,
        "context_hazard": 0.1,
        "cue_reliability": 0.7,
        "dt_ms": 100,
        "cue_ms": 100,
        "sensory_ms": 200,
        "delay_ms": 100,
        "response_ms": 100,
        "sensory_noise_std": 0.25,
    }
    options.update(overrides)
    return HiddenContextConfig(**options)


def test_gate_capability_has_no_truth_choice_or_target_field() -> None:
    names = {field.name for field in fields(GateObservationBatch)}
    assert names == {
        "cue_observations",
        "trial_ids",
        "episode_ids",
        "episode_trial_indices",
        "episode_start",
    }
    forbidden = ("context", "state", "choice", "target", "coherence", "switch")
    assert not any(token in name for name in names for token in forbidden)

    dataset = generate_hidden_context(_small_config(), seed=3)
    assert not hasattr(dataset.gate, "hidden_states")
    assert not hasattr(dataset.gate, "choices")
    assert not hasattr(dataset.gate, "targets")


def test_shared_uniform_tape_thresholds_reliability_without_changing_task() -> None:
    low = _small_config(cue_reliability=0.55)
    high = replace(low, cue_reliability=0.85)
    perfect = replace(low, cue_reliability=1.0)
    tape = make_hidden_context_random_tape(low, seed=17)

    low_data = generate_hidden_context(low, seed=17, random_tape=tape)
    high_data = generate_hidden_context(high, seed=17, random_tape=tape)
    perfect_data = generate_hidden_context(perfect, seed=17, random_tape=tape)

    assert low_data.random_tape_fingerprint == high_data.random_tape_fingerprint
    assert low_data.random_stream_seeds == high_data.random_stream_seeds
    assert low_data.random_stream_fingerprints == high_data.random_stream_fingerprints
    np.testing.assert_array_equal(
        low_data.truth.hidden_states, high_data.truth.hidden_states
    )
    np.testing.assert_array_equal(low_data.task.targets, high_data.task.targets)
    np.testing.assert_array_equal(
        low_data.task.inputs[:, :, :2], high_data.task.inputs[:, :, :2]
    )
    np.testing.assert_array_equal(
        perfect_data.gate.cue_observations, perfect_data.truth.hidden_states
    )

    uniforms = tape.cue_uniform.reshape(-1)
    states = low_data.truth.hidden_states
    np.testing.assert_array_equal(
        low_data.gate.cue_observations,
        np.where(uniforms < 0.55, states, 1 - states),
    )
    np.testing.assert_array_equal(
        high_data.gate.cue_observations,
        np.where(uniforms < 0.85, states, 1 - states),
    )
    low_errors = low_data.gate.cue_observations != states
    high_errors = high_data.gate.cue_observations != states
    assert np.all(~high_errors | low_errors)


def test_hmm_states_are_exact_thresholds_of_the_transition_tape() -> None:
    config = _small_config(context_hazard=0.2, cue_reliability=0.7)
    tape = make_hidden_context_random_tape(config, seed=8)
    dataset = generate_hidden_context(config, seed=8, random_tape=tape)
    states = dataset.truth.hidden_states.reshape(
        config.n_episodes, config.trials_per_episode
    )
    expected_initial = (tape.initial_state_uniform >= 0.5).astype(int)
    np.testing.assert_array_equal(states[:, 0], expected_initial)
    np.testing.assert_array_equal(
        states[:, 1:] != states[:, :-1],
        tape.transition_uniform < config.context_hazard,
    )
    np.testing.assert_array_equal(
        dataset.truth.switch_mask.reshape(config.n_episodes, config.trials_per_episode)[
            :, 0
        ],
        np.zeros(config.n_episodes, dtype=bool),
    )


def test_whole_episode_split_is_reproducible_and_has_no_leakage() -> None:
    dataset = generate_hidden_context(_small_config(), seed=13)
    train, test = dataset.train_test_split(test_fraction=0.25, seed=22)
    again_train, again_test = dataset.train_test_split(test_fraction=0.25, seed=22)

    assert set(train.task.episode_ids).isdisjoint(set(test.task.episode_ids))
    assert set(train.task.trial_ids).isdisjoint(set(test.task.trial_ids))
    assert set(train.task.episode_ids) | set(test.task.episode_ids) == set(
        dataset.task.episode_ids
    )
    assert train.fingerprint == again_train.fingerprint
    assert test.fingerprint == again_test.fingerprint
    for split in (train, test):
        for episode in np.unique(split.task.episode_ids):
            assert (
                np.sum(split.task.episode_ids == episode)
                == dataset.config.trials_per_episode
            )
    with pytest.raises(ValueError, match="complete episodes"):
        dataset.subset(np.arange(1, dataset.config.trials_per_episode))


def test_all_capability_arrays_are_immutable_and_reproducible() -> None:
    first = generate_hidden_context(_small_config(), seed=29)
    second = generate_hidden_context(_small_config(), seed=29)
    assert first.fingerprint == second.fingerprint
    assert first.random_tape_fingerprint == second.random_tape_fingerprint
    arrays = (
        first.task.inputs,
        first.task.targets,
        first.task.loss_mask,
        first.gate.cue_observations,
        first.gate.episode_start,
        first.truth.hidden_states,
        first.truth.choices,
        first.truth.switch_mask,
    )
    assert all(not array.flags.writeable for array in arrays)
    with pytest.raises(ValueError):
        first.gate.cue_observations[0] = 1


def test_independent_streams_and_invalid_tape_seed_are_auditable() -> None:
    config = _small_config()
    first_tape = make_hidden_context_random_tape(config, seed=4)
    second_tape = make_hidden_context_random_tape(config, seed=5)
    names = [name for name, _ in first_tape.stream_seeds]
    seeds = [seed for _, seed in first_tape.stream_seeds]
    assert names == [
        "initial_state",
        "transition",
        "cue",
        "coherence",
        "sensory_noise",
    ]
    assert len(seeds) == len(set(seeds))
    assert first_tape.fingerprint != second_tape.fingerprint
    with pytest.raises(ValueError, match="seed differs"):
        generate_hidden_context(config, seed=5, random_tape=first_tape)


@pytest.mark.parametrize(
    "options",
    [
        {"n_episodes": 1},
        {"trials_per_episode": True},
        {"context_hazard": -0.1},
        {"cue_reliability": 0.49},
        {"cue_reliability": "reliable"},
        {"sensory_noise_std": np.nan},
        {"response_target": "hidden-state"},
    ],
)
def test_hidden_context_config_rejects_invalid_values(
    options: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _small_config(**options)
