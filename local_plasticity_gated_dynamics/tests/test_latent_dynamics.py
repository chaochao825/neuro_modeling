from __future__ import annotations

import numpy as np
import pytest

from src.tasks.latent_dynamics import (
    LatentDynamicsConfig,
    generate_latent_dynamics,
    make_blockwise_feedback_permutation,
    make_feedback_subspace,
)


def _small_config(**overrides: object) -> LatentDynamicsConfig:
    values: dict[str, object] = {
        "n_neurons": 12,
        "latent_dim": 4,
        "train_steps_per_context": 24,
        "test_steps_per_context": 12,
        "n_train_blocks_per_context": 3,
        "n_test_blocks_per_context": 2,
        "activity_noise_std": 0.0,
        "seed": 17,
    }
    values.update(overrides)
    return LatentDynamicsConfig(**values)


def test_formal_defaults_match_phase_one_protocol() -> None:
    config = LatentDynamicsConfig()
    assert config.n_neurons == 128
    assert config.latent_dim == 4
    assert config.n_contexts == 2
    assert config.train_steps_per_context == 10_000


def test_generation_is_reproducible_blocked_and_train_test_independent() -> None:
    config = _small_config()
    first = generate_latent_dynamics(config)
    second = generate_latent_dynamics(config)

    np.testing.assert_array_equal(first.embedding, second.embedding)
    np.testing.assert_array_equal(first.context_dynamics, second.context_dynamics)
    np.testing.assert_array_equal(first.train.activities, second.train.activities)
    np.testing.assert_array_equal(first.test.activities, second.test.activities)

    assert first.train.latent_states.shape == (6, 9, 4)
    assert first.test.latent_states.shape == (4, 7, 4)
    assert first.train.activities.shape == (6, 9, 12)
    assert first.train.n_transitions == 2 * config.train_steps_per_context
    assert first.test.n_transitions == 2 * config.test_steps_per_context
    assert np.intersect1d(first.train.block_ids, first.test.block_ids).size == 0
    assert not np.shares_memory(first.train.activities, first.test.activities)
    assert not np.array_equal(
        first.train.latent_states[0, 0], first.test.latent_states[0, 0]
    )

    transitions = first.train.transitions()
    assert len(transitions) == first.train.n_transitions
    for block_index, block_id in enumerate(first.train.block_ids):
        selected = transitions.block_ids == block_id
        np.testing.assert_array_equal(
            transitions.activity_t[selected], first.train.activities[block_index, :-1]
        )
        np.testing.assert_array_equal(
            transitions.activity_tp1[selected], first.train.activities[block_index, 1:]
        )
        assert np.unique(transitions.contexts[selected]).tolist() == [
            int(first.train.block_contexts[block_index])
        ]


def test_activity_is_tanh_embedding_and_context_systems_are_stable() -> None:
    dataset = generate_latent_dynamics(_small_config())
    expected = np.tanh(dataset.train.latent_states @ dataset.embedding.T)
    np.testing.assert_allclose(dataset.train.activities, expected, atol=1e-14)
    assert np.max(np.abs(dataset.train.activities)) < 1.0
    assert not np.allclose(dataset.context_dynamics[0], dataset.context_dynamics[1])
    for system in dataset.context_dynamics:
        assert np.max(np.abs(np.linalg.eigvals(system))) < 1.0


def test_feedback_modes_have_the_claimed_geometry_and_are_reproducible() -> None:
    dataset = generate_latent_dynamics(_small_config(n_neurons=10))
    embedding = dataset.embedding
    aligned = make_feedback_subspace(embedding, 4, "aligned", seed=3)
    random = make_feedback_subspace(embedding, 4, "random", seed=3)
    random_again = make_feedback_subspace(embedding, 4, "random", seed=3)
    orthogonal = make_feedback_subspace(embedding, 4, "orthogonal", seed=3)
    shuffled = make_feedback_subspace(embedding, 4, "shuffled", seed=3)

    for feedback in (aligned, random, orthogonal, shuffled):
        np.testing.assert_allclose(feedback.basis.T @ feedback.basis, np.eye(4), atol=1e-12)
        errors = np.arange(20, dtype=float).reshape(2, 10)
        np.testing.assert_allclose(
            feedback.project(errors), errors @ feedback.projector, atol=1e-12
        )
    assert aligned.alignment_fraction == pytest.approx(1.0, abs=1e-12)
    assert orthogonal.alignment_fraction == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_array_equal(random.basis, random_again.basis)
    # Temporal shuffling intentionally holds feedback geometry fixed.
    np.testing.assert_array_equal(shuffled.basis, aligned.basis)


def test_temporal_feedback_permutation_is_deterministic_and_block_local() -> None:
    block_ids = np.array([10, 10, 10, 10, 20, 20, 20])
    first = make_blockwise_feedback_permutation(block_ids, seed=5)
    second = make_blockwise_feedback_permutation(block_ids, seed=5)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(block_ids.size))
    assert not np.any(first == np.arange(block_ids.size))
    np.testing.assert_array_equal(block_ids[first], block_ids)
    assert not first.flags.writeable

    with pytest.raises(ValueError, match="fewer than two"):
        make_blockwise_feedback_permutation([0, 0, 1], seed=5)
    with pytest.raises(TypeError, match="integers"):
        make_blockwise_feedback_permutation([0.0, 0.0], seed=5)


def test_aligned_feedback_adds_only_orthogonal_directions_above_latent_dim() -> None:
    dataset = generate_latent_dynamics(_small_config(n_neurons=10))
    feedback = make_feedback_subspace(dataset.embedding, 7, "aligned", seed=8)
    np.testing.assert_allclose(
        feedback.basis[:, :4], feedback.task_basis, atol=1e-12
    )
    np.testing.assert_allclose(
        feedback.task_basis.T @ feedback.basis[:, 4:], np.zeros((4, 3)), atol=1e-12
    )
    full = make_feedback_subspace(dataset.embedding, 10, "aligned", seed=8)
    np.testing.assert_allclose(full.projector, np.eye(10), atol=1e-11)


def test_formal_orthogonal_feedback_above_124_is_explicitly_invalid() -> None:
    embedding = np.eye(128, 4)
    with pytest.raises(ValueError, match="maximum is 124"):
        make_feedback_subspace(embedding, 125, "orthogonal", seed=0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"n_contexts": 3}, "exactly two"),
        ({"latent_dim": 12}, "smaller than n_neurons"),
        ({"train_steps_per_context": 25}, "must be divisible"),
        ({"dynamics_decay": 1.0}, "must be <"),
        ({"activity_noise_std": -0.1}, "must be >="),
    ],
)
def test_config_rejects_invalid_protocol_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _small_config(**overrides)


def test_feedback_constructor_rejects_invalid_inputs() -> None:
    embedding = np.eye(8, 2)
    with pytest.raises(ValueError, match="feedback_dim"):
        make_feedback_subspace(embedding, 0, "aligned", seed=0)
    with pytest.raises(ValueError, match="full column rank"):
        make_feedback_subspace(np.ones((8, 2)), 2, "aligned", seed=0)
    with pytest.raises(ValueError, match="unknown feedback mode"):
        make_feedback_subspace(embedding, 2, "bad", seed=0)  # type: ignore[arg-type]
