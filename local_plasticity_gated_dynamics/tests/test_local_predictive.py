from __future__ import annotations

import numpy as np
import pytest

from src.models.local_predictive import LocalPredictiveConfig, LocalPredictiveModel
from src.tasks.latent_dynamics import make_feedback_subspace


def test_one_local_update_matches_the_outer_product_and_accounts_for_cost() -> None:
    basis = np.eye(3)[:, :2]
    initial = 0.1 * np.eye(3)
    config = LocalPredictiveConfig(
        learning_rate=0.2,
        weight_decay=0.1,
        batch_size=None,
        max_epochs=1,
        shuffle_batches=False,
    )
    model = LocalPredictiveModel(basis, config=config, initial_weights=initial)
    inputs = np.array([[1.0, -1.0, 0.5], [0.5, 2.0, -1.0]])
    targets = np.array([[0.2, 0.3, 0.4], [-0.1, 0.5, 0.7]])

    error = targets - inputs @ initial.T
    projected_error = (error @ basis) @ basis.T
    expected_raw = 0.2 * projected_error.T @ inputs / inputs.shape[0]
    # The supplied initialization is a fixed bulk reference, so the first
    # update has no learned component to decay.
    expected_applied = expected_raw

    model.partial_fit(inputs, targets)
    np.testing.assert_allclose(model.raw_plastic_update[0], expected_raw)
    np.testing.assert_allclose(model.applied_plastic_update[0], expected_applied)
    np.testing.assert_allclose(model.weights_[0], initial + expected_applied)
    assert model.raw_plasticity_cost_ == pytest.approx(np.abs(expected_raw).sum())
    assert model.plasticity_cost == pytest.approx(np.abs(expected_applied).sum())
    assert model.plasticity_cost_by_context_[0] == pytest.approx(model.plasticity_cost)
    assert model.update_history_[-1].contexts == (0,)
    assert np.linalg.matrix_rank(model.raw_plastic_update[0], tol=1e-12) <= 2

    before_second = model.weights_.copy()
    second_error = targets - inputs @ before_second[0].T
    second_projected = (second_error @ basis) @ basis.T
    expected_second_raw = 0.2 * second_projected.T @ inputs / inputs.shape[0]
    expected_second_applied = expected_second_raw - 0.2 * 0.1 * expected_raw
    model.partial_fit(inputs, targets)
    np.testing.assert_allclose(model.raw_plastic_update[0], expected_second_raw)
    np.testing.assert_allclose(model.applied_plastic_update[0], expected_second_applied)
    assert np.linalg.matrix_rank(model.plastic_component[0], tol=1e-12) <= 2


def test_sufficient_statistic_fixed_point_learns_two_context_dynamics() -> None:
    rng = np.random.default_rng(4)
    inputs = rng.normal(size=(320, 3))
    contexts = np.tile(np.array([0, 1]), 160)
    true_weights = np.array(
        [
            [[0.7, 0.1, 0.0], [-0.2, 0.5, 0.1], [0.0, 0.1, 0.6]],
            [[0.4, -0.3, 0.1], [0.2, 0.6, 0.0], [-0.1, 0.2, 0.5]],
        ]
    )
    targets = np.einsum("bij,bj->bi", true_weights[contexts], inputs)
    config = LocalPredictiveConfig(
        learning_rate=0.4,
        weight_decay=0.0,
        max_epochs=250,
        tolerance=1e-11,
        shuffle_batches=False,
    )
    model = LocalPredictiveModel(np.eye(3), n_contexts=2, config=config)
    model.fit_fixed_point(inputs, targets, contexts, max_iterations=250)

    predictions = model.predict(inputs, contexts)
    assert np.mean(np.square(predictions - targets)) < 1e-14
    np.testing.assert_allclose(model.weights_, true_weights, atol=2e-7)
    assert model.converged_
    assert model.n_epochs_ <= 250


def test_low_dimensional_feedback_constrains_the_entire_plastic_component() -> None:
    rng = np.random.default_rng(2)
    basis, _ = np.linalg.qr(rng.normal(size=(6, 2)), mode="reduced")
    inputs = rng.normal(size=(80, 6))
    targets = rng.normal(size=(80, 6))
    model = LocalPredictiveModel(
        basis,
        config=LocalPredictiveConfig(
            learning_rate=0.1,
            weight_decay=0.01,
            max_epochs=8,
            tolerance=0.0,
            shuffle_batches=False,
        ),
    )
    model.fit_fixed_point(inputs, targets, max_iterations=8)

    plastic = model.plastic_component[0]
    assert np.linalg.matrix_rank(plastic, tol=1e-10) <= 2
    complement_projector = np.eye(6) - basis @ basis.T
    np.testing.assert_allclose(complement_projector @ plastic, 0.0, atol=1e-12)


def test_minibatch_training_is_reproducible_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(9)
    inputs = rng.normal(size=(41, 4))
    targets = rng.normal(size=(41, 4))
    contexts = np.arange(41) % 2
    config = LocalPredictiveConfig(
        learning_rate=0.02,
        batch_size=7,
        max_epochs=4,
        tolerance=0.0,
        shuffle_batches=True,
        seed=12,
    )
    first = LocalPredictiveModel(np.eye(4), n_contexts=2, config=config)
    second = LocalPredictiveModel(np.eye(4), n_contexts=2, config=config)
    first.fit(inputs, targets, contexts)
    second.fit(inputs, targets, contexts)
    np.testing.assert_array_equal(first.weights_, second.weights_)
    assert first.plasticity_cost == second.plasticity_cost


def test_predict_and_rollout_select_context_without_learning() -> None:
    weights = np.array(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.5]],
        ]
    )
    model = LocalPredictiveModel(np.eye(2), n_contexts=2, initial_weights=weights)
    np.testing.assert_allclose(model.predict(np.array([1.0, 2.0]), 0), [2.0, 2.0])
    trajectory = model.rollout(np.array([1.0, 2.0]), 2, contexts=[0, 1])
    np.testing.assert_allclose(trajectory, [[1.0, 2.0], [2.0, 2.0], [2.0, 1.0]])
    np.testing.assert_allclose(model.rollout(np.array([1.0, 2.0]), 0), [[1.0, 2.0]])


def test_update_clipping_distinguishes_raw_and_applied_changes() -> None:
    model = LocalPredictiveModel(
        np.eye(2),
        config=LocalPredictiveConfig(
            learning_rate=1.0,
            max_update_fro_norm=0.05,
            max_epochs=1,
        ),
    )
    model.partial_fit(np.ones((2, 2)), 10.0 * np.ones((2, 2)))
    assert np.linalg.norm(model.raw_plastic_update[0]) > 0.05
    assert np.linalg.norm(model.applied_plastic_update[0]) == pytest.approx(0.05)


def test_shuffled_feedback_uses_and_records_block_local_source_times() -> None:
    embedding = np.eye(4, 2)
    feedback = make_feedback_subspace(embedding, 2, "shuffled", seed=6)
    config = LocalPredictiveConfig(
        learning_rate=0.1,
        batch_size=None,
        max_epochs=1,
        tolerance=0.0,
        shuffle_batches=False,
    )
    model = LocalPredictiveModel(feedback, config=config)
    inputs = np.array(
        [[1.0, 0.0, 0.5, -1.0], [0.0, 1.0, -0.5, 1.0],
         [0.5, 0.2, 1.0, 0.0], [-0.5, 1.0, 0.0, 0.3]]
    )
    targets = np.arange(16, dtype=float).reshape(4, 4) / 10.0
    block_ids = np.array([10, 10, 20, 20])
    permutation = np.array([1, 0, 3, 2])

    with pytest.raises(ValueError, match="requires a block-local"):
        model.fit(inputs, targets)
    model.fit(
        inputs,
        targets,
        feedback_permutation=permutation,
        block_ids=block_ids,
    )
    expected_error = targets[permutation]  # initial weights are exactly zero
    expected_projected = (expected_error @ feedback.basis) @ feedback.basis.T
    expected_raw = 0.1 * expected_projected.T @ inputs / inputs.shape[0]
    np.testing.assert_allclose(model.raw_plastic_update[0], expected_raw)
    np.testing.assert_array_equal(model.temporal_feedback_permutation, permutation)
    np.testing.assert_array_equal(model.feedback_block_ids_, block_ids)
    assert not model.feedback_permutation_.flags.writeable


def test_temporal_feedback_permutation_cannot_cross_blocks_or_contexts() -> None:
    model = LocalPredictiveModel(np.eye(2), n_contexts=2)
    samples = np.ones((4, 2))
    contexts = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="block boundaries"):
        model.fit(
            samples,
            samples,
            contexts,
            feedback_permutation=[2, 3, 0, 1],
            block_ids=[0, 0, 1, 1],
        )
    with pytest.raises(ValueError, match="context boundaries"):
        model.fit(
            samples,
            samples,
            contexts,
            feedback_permutation=[2, 3, 0, 1],
            block_ids=[0, 0, 0, 0],
        )


def test_model_validates_feedback_samples_contexts_and_rollout() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        LocalPredictiveModel(np.ones((3, 2)))
    with pytest.raises(ValueError, match="initial_weights"):
        LocalPredictiveModel(np.eye(2), initial_weights=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="at least 1"):
        LocalPredictiveModel(np.eye(2), n_contexts=0)

    model = LocalPredictiveModel(np.eye(2), n_contexts=2)
    samples = np.ones((3, 2))
    with pytest.raises(ValueError, match="contexts are required"):
        model.fit(samples, samples)
    with pytest.raises(TypeError, match="integers"):
        model.predict(samples, [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="lie in"):
        model.predict(samples, [0, 1, 2])
    with pytest.raises(ValueError, match="finite"):
        model.predict(np.array([[np.nan, 0.0]]), [0])
    with pytest.raises(ValueError, match="initial_activity"):
        model.rollout(np.ones(3), 2, 0)
    with pytest.raises(ValueError, match="at least 0"):
        model.rollout(np.ones(2), -1, 0)


@pytest.mark.parametrize(
    "kwargs, exception",
    [
        ({"learning_rate": 0.0}, ValueError),
        ({"weight_decay": -1.0}, ValueError),
        ({"batch_size": 0}, ValueError),
        ({"max_epochs": 0}, ValueError),
        ({"shuffle_batches": 1}, TypeError),
        ({"seed": -1}, ValueError),
    ],
)
def test_config_rejects_invalid_values(
    kwargs: dict[str, object], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        LocalPredictiveConfig(**kwargs)
