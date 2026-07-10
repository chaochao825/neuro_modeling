from __future__ import annotations

import numpy as np

from shared_dynamics_real_data.pipeline import SharedDynamicsPipeline
from shared_dynamics_real_data.preprocessing import (
    TrainOnlyPreprocessor,
    fit_controlled_basis,
)
from shared_dynamics_real_data.splits import TimeSegment


def _segment(context: str, values: np.ndarray, start: int = 0) -> TimeSegment:
    return TimeSegment(context, values, np.arange(start, start + len(values)))


def test_unit_selection_scaling_and_pca_ignore_test_sentinel() -> None:
    time = np.linspace(-1.0, 1.0, 24)
    train_values = np.column_stack([time, 2.0 * time, np.zeros_like(time)])
    train = (_segment("a", train_values), _segment("b", train_values[::-1]))
    sentinel = np.full((8, 3), 10_000.0)
    test = (_segment("a", sentinel, 100), _segment("b", sentinel, 200))

    pipeline = SharedDynamicsPipeline(
        "shared", latent_dim=1, max_units=3, random_state=7
    ).fit(train)
    train_reference_scale = pipeline.model_.rollout_reference_scale_
    pipeline.score(test)

    np.testing.assert_allclose(pipeline.preprocessor_.mean_, 0.0, atol=1e-12)
    basis = pipeline.model_.bases_["a"]
    assert abs(basis[2, 0]) < 1e-12
    # The test-only sentinel must not alter any train-fitted statistic.
    assert np.max(pipeline.preprocessor_.mean_) < 1.0
    assert pipeline.model_.rollout_reference_scale_ == train_reference_scale
    np.testing.assert_allclose(train_reference_scale, np.sqrt(2.0 / 3.0))


def test_unit_selection_uses_training_variance_only() -> None:
    train = np.column_stack(
        [np.linspace(0, 1, 20), np.linspace(0, 20, 20), np.ones(20)]
    )
    preprocessor = TrainOnlyPreprocessor(max_units=1).fit((_segment("x", train),))
    np.testing.assert_array_equal(preprocessor.unit_indices_, [1])


def test_all_basis_controls_are_orthonormal_and_seeded() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(60, 8))
    aligned = fit_controlled_basis(values, 3, control="aligned", random_state=11)
    for control in ("aligned", "random", "orthogonal", "shuffled"):
        first = fit_controlled_basis(values, 3, control=control, random_state=11)
        second = fit_controlled_basis(values, 3, control=control, random_state=11)
        np.testing.assert_allclose(first.T @ first, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(first, second, atol=0.0)
    orthogonal = fit_controlled_basis(
        values, 3, control="orthogonal", random_state=11
    )
    np.testing.assert_allclose(aligned.T @ orthogonal, 0.0, atol=1e-10)
