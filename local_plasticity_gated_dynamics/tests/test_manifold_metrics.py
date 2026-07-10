import numpy as np
import pytest

from src.analysis.manifold_metrics import (
    fit_train_pca,
    latent_r2,
    principal_angles,
    rollout_error,
    rollout_metrics,
    subspace_overlap,
)


def test_pca_normalization_is_frozen_from_training_data() -> None:
    train = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    fitted = fit_train_pca(
        train, n_components=1, normalize=True, sample_ids=["a", "b", "c", "d"]
    )
    fitted_mean = fitted.mean_.copy()
    fitted_scale = fitted.scale_.copy()

    test = np.array([[100.0, 200.0], [101.0, 202.0]])
    scores = fitted.transform(test)

    assert fitted.fit_sample_ids_ == ("a", "b", "c", "d")
    assert np.array_equal(fitted.mean_, fitted_mean)
    assert np.array_equal(fitted.scale_, fitted_scale)
    with pytest.raises(ValueError):
        fitted.mean_[0] = 999.0
    assert abs(float(np.mean(scores))) > 10.0
    reconstructed_train = fitted.inverse_transform(fitted.transform(train))
    assert np.allclose(reconstructed_train, train)


def test_principal_angles_and_projection_overlap() -> None:
    first = np.eye(4)[:, :2]
    same = first @ np.array([[0.0, -1.0], [1.0, 0.0]])
    orthogonal = np.eye(4)[:, 2:]

    assert np.allclose(principal_angles(first, same), 0.0, atol=1e-7)
    assert subspace_overlap(first, same) == pytest.approx(1.0)
    assert np.allclose(principal_angles(first, orthogonal), np.pi / 2.0)
    assert subspace_overlap(first, orthogonal) == pytest.approx(0.0)


def test_latent_r2_and_rollout_horizon_metrics() -> None:
    latent = np.arange(24, dtype=float).reshape(3, 4, 2)
    assert latent_r2(latent, latent) == pytest.approx(1.0)
    assert np.allclose(latent_r2(latent, latent, multioutput="raw_values"), 1.0)

    truth = np.zeros((2, 3, 2))
    prediction = np.broadcast_to(
        np.array([1.0, 2.0, 3.0])[None, :, None], truth.shape
    ).copy()
    summary = rollout_metrics(
        truth, prediction, train_reference=np.array([[-1.0, -1.0], [1.0, 1.0]])
    )
    assert summary.rmse == pytest.approx(np.sqrt(14.0 / 3.0))
    assert summary.mae == pytest.approx(2.0)
    assert np.allclose(summary.per_horizon_rmse, [1.0, 2.0, 3.0])
    assert summary.normalized_rmse == pytest.approx(summary.rmse)
    assert rollout_error(truth, prediction) == pytest.approx(summary.rmse)


def test_train_reference_is_required_for_normalized_rollout() -> None:
    values = np.zeros((3, 2))
    with pytest.raises(ValueError, match="train_reference"):
        rollout_error(values, values, normalized=True)
    with pytest.raises(ValueError, match="constant"):
        rollout_metrics(values, values, train_reference=np.ones((4, 2)))


def test_manifold_metrics_validate_shapes_and_rank() -> None:
    with pytest.raises(ValueError):
        fit_train_pca(np.ones((2, 3)), n_components=2)
    with pytest.raises(ValueError, match="linearly independent"):
        principal_angles(np.ones((3, 2)), np.eye(3)[:, :2])
    with pytest.raises(ValueError, match="identical shapes"):
        latent_r2(np.ones((3, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError, match="missing"):
        fit_train_pca(np.eye(3), n_components=1, sample_ids=[0, np.nan, 2])
