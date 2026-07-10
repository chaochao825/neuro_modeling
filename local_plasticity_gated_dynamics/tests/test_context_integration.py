import numpy as np
import pytest

from src.tasks.context_integration import (
    ContextIntegrationConfig,
    generate_context_integration,
)


def test_context_task_has_requested_epoch_timing_and_reproducibility() -> None:
    config = ContextIntegrationConfig(n_trials=40)
    first = generate_context_integration(config, seed=7)
    second = generate_context_integration(config, seed=7)
    assert first.inputs.shape == (40, 85, 4)
    assert np.array_equal(first.inputs, second.inputs)
    assert np.sum(first.epoch == "cue") == 10
    assert np.sum(first.epoch == "sensory") == 40
    assert np.sum(first.epoch == "delay") == 25
    assert np.sum(first.epoch == "response") == 10
    assert set(np.unique(first.contexts)) == {0, 1}


def test_context_task_split_never_leaks_blocks_or_trials() -> None:
    batch = generate_context_integration(
        ContextIntegrationConfig(n_trials=100, context_block_trials=10), seed=11
    )
    train, test = batch.train_test_split(test_fraction=0.3, seed=3)
    assert set(train.block_ids).isdisjoint(set(test.block_ids))
    assert set(train.trial_ids).isdisjoint(set(test.trial_ids))
    assert train.inputs.shape[1:] == test.inputs.shape[1:]
    assert set(train.contexts) == set(test.contexts) == {0, 1}


def test_context_split_rejects_impossible_condition_coverage() -> None:
    batch = generate_context_integration(
        ContextIntegrationConfig(n_trials=4, context_block_trials=2), seed=3
    )
    with pytest.raises(ValueError, match="both contexts"):
        batch.train_test_split(seed=0)


def test_context_cue_and_target_follow_relevant_stream() -> None:
    config = ContextIntegrationConfig(
        n_trials=8,
        sensory_noise_std=0.0,
        coherence_values=(-0.5, 0.5),
        context_block_trials=2,
    )
    batch = generate_context_integration(config, seed=4)
    cue_steps = config.epoch_steps["cue"]
    for trial, context in enumerate(batch.contexts):
        assert np.all(batch.inputs[trial, :cue_steps, 2 + context] == 1.0)
        expected = 1 if batch.coherences[trial, context] > 0 else -1
        assert batch.choices[trial] == expected
        sensory = batch.inputs[trial, cue_steps : cue_steps + config.epoch_steps["sensory"], :2]
        relevant = sensory[:, context] / config.input_scale
        expected_target = np.tanh(
            np.cumsum(relevant) / np.sqrt(np.arange(1, relevant.size + 1))
        )
        np.testing.assert_allclose(
            batch.targets[trial, cue_steps : cue_steps + relevant.size, 0],
            expected_target,
        )


def test_trial_by_trial_regime_uses_one_block_per_trial() -> None:
    config = ContextIntegrationConfig(
        n_trials=30, context_block_trials=10, trial_by_trial_after=20
    )
    batch = generate_context_integration(config, seed=12)
    assert np.unique(batch.block_ids[20:]).size == 10
    assert np.all(np.diff(batch.block_ids[20:]) == 1)


def test_batch_arrays_are_frozen_and_subset_indices_are_not_silently_cast() -> None:
    batch = generate_context_integration(ContextIntegrationConfig(n_trials=8), seed=2)
    assert not batch.inputs.flags.writeable
    assert not batch.block_ids.flags.writeable
    with pytest.raises(ValueError):
        batch.inputs[0, 0, 0] = 1.0
    with pytest.raises(TypeError, match="integers"):
        batch.subset(np.array([0.2, 1.2]))
    with pytest.raises(ValueError, match="duplicates"):
        batch.subset(np.array([0, 0]))


def test_context_config_rejects_non_integral_epochs() -> None:
    with pytest.raises(ValueError, match="multiple"):
        ContextIntegrationConfig(n_trials=10, delay_ms=510)


def test_context_config_rejects_nonfinite_and_noninteger_values() -> None:
    with pytest.raises(TypeError, match="integer"):
        ContextIntegrationConfig(n_trials=4.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-zero"):
        ContextIntegrationConfig(n_trials=8, coherence_values=(0.5, np.nan))
    with pytest.raises(ValueError, match="input_scale"):
        ContextIntegrationConfig(n_trials=8, input_scale=np.inf)
