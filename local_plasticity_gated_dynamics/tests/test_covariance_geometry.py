from __future__ import annotations

import numpy as np
import pytest

from src.analysis.covariance_geometry import (
    bures_covariance_distance,
    compare_covariance_geometries,
    covariance_eigenspectrum,
    covariance_participation_ratio,
    covariance_principal_angles,
    fit_conditional_covariance_geometry,
    fit_covariance_geometry,
)


def test_covariance_spectrum_participation_ratio_and_bures_distance() -> None:
    first = np.diag([4.0, 1.0])
    second = np.diag([1.0, 9.0])
    np.testing.assert_allclose(
        covariance_eigenspectrum(first),
        [4.0, 1.0],
    )
    np.testing.assert_allclose(
        covariance_eigenspectrum(first, normalize=True),
        [0.8, 0.2],
    )
    assert covariance_participation_ratio(first) == pytest.approx(25.0 / 17.0)
    assert bures_covariance_distance(first, second, squared=True) == pytest.approx(5.0)
    assert bures_covariance_distance(first, second) == pytest.approx(np.sqrt(5.0))
    assert bures_covariance_distance(first, first) == pytest.approx(0.0)


def test_bures_distance_is_stable_for_nearly_low_rank_covariances() -> None:
    rng = np.random.default_rng(91)
    basis, _ = np.linalg.qr(rng.normal(size=(128, 8)))
    spectrum = np.geomspace(2e-2, 1e-12, 8)
    first = (basis * spectrum[np.newaxis, :]) @ basis.T
    perturbation = rng.normal(scale=1e-4, size=(128, 8))
    shifted_basis, _ = np.linalg.qr(basis + perturbation)
    second = (shifted_basis * spectrum[np.newaxis, :]) @ shifted_basis.T

    forward = bures_covariance_distance(first, second, squared=True)
    reverse = bures_covariance_distance(second, first, squared=True)

    assert np.isfinite(forward)
    assert forward >= 0.0
    assert forward == pytest.approx(reverse, rel=1e-7, abs=1e-14)


def test_covariance_principal_angles_detect_rotation() -> None:
    first = np.diag([5.0, 2.0, 0.5])
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = rotation @ first @ rotation.T
    # The leading two-dimensional plane is unchanged by an in-plane rotation.
    np.testing.assert_allclose(
        covariance_principal_angles(
            first,
            rotated,
            n_components=2,
        ),
        0.0,
        atol=1e-8,
    )
    orthogonal = np.diag([0.5, 2.0, 5.0])
    angle = covariance_principal_angles(
        first,
        orthogonal,
        n_components=1,
    )
    assert angle[0] == pytest.approx(90.0)


def test_covariance_geometry_fit_is_train_only_and_comparable() -> None:
    rng = np.random.default_rng(4)
    train = rng.normal(size=(80, 3)) @ np.diag([3.0, 1.0, 0.3])
    fit = fit_covariance_geometry(
        train,
        n_components=2,
        sample_ids=np.arange(80),
    )
    heldout = np.full((10, 3), 1e6)
    transformed = fit.transform(heldout)

    assert fit.n_train_samples_ == 80
    assert fit.fit_sample_ids_ == tuple(range(80))
    np.testing.assert_allclose(fit.mean_, np.mean(train, axis=0))
    assert transformed.shape == (10, 2)
    np.testing.assert_allclose(fit.mean_, np.mean(train, axis=0))
    assert not fit.covariance_.flags.writeable

    second = fit_covariance_geometry(train.copy(), n_components=2)
    comparison = compare_covariance_geometries(fit, second)
    np.testing.assert_allclose(comparison.principal_angles_degrees, 0.0, atol=1e-6)
    assert comparison.subspace_overlap == pytest.approx(1.0)
    assert comparison.bures_distance == pytest.approx(0.0)
    assert comparison.participation_ratio_a == pytest.approx(
        comparison.participation_ratio_b
    )


def test_conditional_covariance_uses_frozen_training_residualizer() -> None:
    rng = np.random.default_rng(12)
    nuisance = np.linspace(-2.0, 2.0, 100)[:, None]
    residual = rng.normal(scale=0.2, size=(100, 2))
    activity = np.column_stack([2.0 * nuisance[:, 0], -nuisance[:, 0]]) + residual
    fitted = fit_conditional_covariance_geometry(
        activity,
        nuisance,
        n_components=2,
        sample_ids=np.arange(100),
    )
    heldout_nuisance = np.array([[1.0], [-1.0]])
    heldout_activity = np.column_stack(
        [2.0 * heldout_nuisance[:, 0], -heldout_nuisance[:, 0]]
    )
    heldout_residual = fitted.residualize(heldout_activity, heldout_nuisance)

    assert fitted.geometry_.fit_sample_ids_ == tuple(range(100))
    assert np.max(np.abs(heldout_residual)) < 0.1
    assert fitted.geometry_.participation_ratio_ > 1.0


def test_covariance_geometry_rejects_indefinite_or_rank_deficient_inputs() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        covariance_eigenspectrum(np.diag([1.0, -0.1]))
    with pytest.raises(ValueError, match="insufficient positive"):
        covariance_principal_angles(
            np.diag([1.0, 0.0]),
            np.eye(2),
            n_components=2,
        )
