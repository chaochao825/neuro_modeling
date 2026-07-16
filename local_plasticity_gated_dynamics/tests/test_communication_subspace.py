from __future__ import annotations

import numpy as np
import pytest

from src.analysis.communication_subspace import (
    communication_subspace_overlap,
    compare_communication_subspaces,
    fit_train_communication_subspace,
)


def test_train_fitted_communication_subspace_predicts_heldout_data() -> None:
    rng = np.random.default_rng(21)
    source = rng.normal(size=(300, 4))
    coefficient = np.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, -1.5, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    target = source @ coefficient + 0.01 * rng.normal(size=(300, 3))
    fit = fit_train_communication_subspace(
        source[:220],
        target[:220],
        rank=2,
        ridge=1e-6,
        normalize=True,
        sample_ids=np.arange(220),
    )

    assert fit.rank_ == 2
    assert fit.n_train_samples_ == 220
    assert fit.fit_sample_ids_ == tuple(range(220))
    assert np.linalg.matrix_rank(fit.coefficients_) == 2
    assert fit.source_basis_.shape == (4, 2)
    assert fit.target_basis_.shape == (3, 2)
    assert fit.transform_source(source[220:]).shape == (80, 2)
    assert fit.transform_target(target[220:]).shape == (80, 2)
    assert fit.heldout_r2(source[220:], target[220:]) > 0.99
    assert not fit.coefficients_.flags.writeable


def test_communication_fit_does_not_borrow_heldout_statistics() -> None:
    rng = np.random.default_rng(8)
    source_train = rng.normal(size=(100, 3))
    target_train = np.column_stack(
        [
            source_train[:, 0] + source_train[:, 1],
            source_train[:, 1] - source_train[:, 0],
        ]
    )
    fit = fit_train_communication_subspace(
        source_train,
        target_train,
        rank=2,
        normalize=True,
    )
    source_mean = fit.source_mean_.copy()
    target_mean = fit.target_mean_.copy()
    fit.predict(np.full((4, 3), 1e9))

    np.testing.assert_array_equal(fit.source_mean_, source_mean)
    np.testing.assert_array_equal(fit.target_mean_, target_mean)
    np.testing.assert_allclose(source_mean, np.mean(source_train, axis=0))
    np.testing.assert_allclose(target_mean, np.mean(target_train, axis=0))


def test_communication_overlap_distinguishes_shared_and_orthogonal_routes() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(500, 4))
    first_target = source[:, [0]]
    shared_target = -2.0 * source[:, [0]]
    orthogonal_target = source[:, [1]]
    first = fit_train_communication_subspace(
        source,
        first_target,
        rank=1,
    )
    shared = fit_train_communication_subspace(
        source,
        shared_target,
        rank=1,
    )
    orthogonal = fit_train_communication_subspace(
        source,
        orthogonal_target,
        rank=1,
    )

    assert communication_subspace_overlap(first, shared) == pytest.approx(
        1.0, abs=1e-10
    )
    assert communication_subspace_overlap(first, orthogonal) == pytest.approx(
        0.0, abs=1e-10
    )
    comparison = compare_communication_subspaces(first, orthogonal)
    assert comparison.principal_angles_degrees[0] == pytest.approx(90.0, abs=1e-8)
    assert comparison.side == "source"


def test_communication_subspace_validation_fails_closed() -> None:
    source = np.arange(30, dtype=float).reshape(10, 3)
    target = source[:, :2]
    with pytest.raises(ValueError, match="share sample rows"):
        fit_train_communication_subspace(source, target[:-1], rank=1)
    with pytest.raises(ValueError, match="must not exceed"):
        fit_train_communication_subspace(source, target, rank=3)
    with pytest.raises(ValueError, match="predictive rank"):
        fit_train_communication_subspace(
            np.eye(4),
            np.ones((4, 2)),
            rank=1,
        )
