"""Contracts for the train-only coarse effective-control dynamics audit."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.effective_control_dynamics import (
    effective_control_dynamics_audit,
    ei_continuous_jacobian,
    ei_scalar_gain_jacobian_tangent,
    fit_soft_epoch_dynamics,
    projected_normal_linear_summary,
    pullback_rate_tangent_basis,
)
from src.models.ei_rate_network import EIRateNetwork


def _closed_epoch_features(
    n_trials: int, *, seed: int, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    state0 = np.array([[0.72, 0.08], [-0.06, 0.61]])
    state1 = np.array([[0.48, -0.16], [0.13, 0.79]])
    probability = (np.arange(n_trials) % 2).astype(float)
    latent = np.empty((n_trials, 3, 2), dtype=float)
    for trial in range(n_trials):
        latent[trial, 0] = rng.normal(size=2)
        operator = state1 if probability[trial] else state0
        latent[trial, 1] = latent[trial, 0] @ operator
        latent[trial, 2] = latent[trial, 1] @ operator
    states = latent @ basis.T
    return states.reshape(n_trials, -1), probability


def _stable_high_rank_network() -> EIRateNetwork:
    return EIRateNetwork(
        n_units=8,
        n_inputs=0,
        excitatory_fraction=0.75,
        connection_probability=1.0,
        tau_e=10.0,
        tau_i=10.0,
        dt=1.0,
        bulk_gain=0.05,
        inhibitory_gain=1.0,
        activation="tanh",
        allow_self_connections=True,
        normalize_fan_in_after_update=False,
        seed=9,
    )


def test_high_rank_physical_network_has_closed_lowdim_epoch_surrogate() -> None:
    basis, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(8, 2)))
    train_features, train_probability = _closed_epoch_features(80, seed=5, basis=basis)
    test_features, test_probability = _closed_epoch_features(30, seed=6, basis=basis)
    network = _stable_high_rank_network()
    gain_axis = np.linspace(-0.9, 0.9, network.n_units)
    mean_state = np.linspace(0.1, 0.3, network.n_units)

    audit = effective_control_dynamics_audit(
        train_features=train_features,
        test_features=test_features,
        train_state1_probability=train_probability,
        test_state1_probability=test_probability,
        network=network,
        mean_state=mean_state,
        gain_axis=gain_axis,
        gain_strength=0.4,
        latent_dim=2,
        ridge=1e-10,
        normal_horizon=1.0,
    )
    repeated = effective_control_dynamics_audit(
        train_features=train_features,
        test_features=test_features,
        train_state1_probability=train_probability,
        test_state1_probability=test_probability,
        network=network,
        mean_state=mean_state,
        gain_axis=gain_axis,
        gain_strength=0.4,
        latent_dim=2,
        ridge=1e-10,
        normal_horizon=1.0,
    )

    assert audit == repeated
    assert audit.physical_raw_rank == network.n_units
    assert audit.physical_effective_rank > 4.0
    assert audit.basis_dimension == 2
    assert audit.basis_explained_variance_fraction == pytest.approx(1.0)
    assert audit.heldout_normalized_closure_mse < 1e-16
    assert audit.heldout_basis_residual_fraction < 1e-24
    assert audit.declared_scalar_control_dimension == 1
    assert audit.operator_control_dimension == 1
    assert audit.operator_design_rank == audit.operator_design_columns
    # A scalar controller can change a full-rank Jacobian; these quantities
    # must not be conflated with its one-dimensional input space.
    assert audit.jacobian_tangent_raw_rank == network.n_units
    assert audit.same_state_delta_jacobian_raw_rank == network.n_units
    assert audit.local_state_tangent_dimension == 2
    assert audit.rate_to_state_pullback_residual < 1e-12
    assert audit.normal_stability_eligible
    assert audit.normal_dimension == network.n_units - 2
    assert audit.normal_local_max_real_part < 0.0
    assert audit.normal_local_decay_ratio < 1.0
    assert audit.preprocessing_fit_train_only
    assert not audit.heldout_used_for_fit
    assert "not_full_trajectory_lds" in audit.approximation_scope


def test_soft_epoch_fit_is_frozen_and_heldout_scoring_does_not_refit() -> None:
    basis, _ = np.linalg.qr(np.random.default_rng(10).normal(size=(6, 2)))
    train, probability = _closed_epoch_features(40, seed=11, basis=basis)
    heldout, heldout_probability = _closed_epoch_features(12, seed=12, basis=basis)
    fitted = fit_soft_epoch_dynamics(
        train,
        np.column_stack((1.0 - probability, probability)),
        n_units=6,
        latent_dim=2,
        ridge=1e-8,
        normalize_activity=True,
    )
    snapshots = tuple(
        value.copy()
        for value in (
            fitted.pca.mean_,
            fitted.pca.scale_,
            fitted.pca.components_,
            fitted.operator_state0,
            fitted.operator_state1,
            fitted.closure_variance,
        )
    )

    first = fitted.score(heldout, heldout_probability)
    shifted = fitted.score(heldout + 100.0, heldout_probability)
    for before, after in zip(
        snapshots,
        (
            fitted.pca.mean_,
            fitted.pca.scale_,
            fitted.pca.components_,
            fitted.operator_state0,
            fitted.operator_state1,
            fitted.closure_variance,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)
        assert not after.flags.writeable
    assert first.normalized_closure_mse < shifted.normalized_closure_mse
    assert fitted.preprocessing_fit_scope == "train_features_only"
    with pytest.raises(ValueError, match=r"3 \* n_units"):
        fitted.score(np.ones((2, 17)), np.full(2, 0.5))


def test_jacobian_tangent_matches_finite_difference() -> None:
    network = _stable_high_rank_network()
    state = np.linspace(0.08, 0.29, network.n_units)
    axis = np.linspace(-0.8, 0.7, network.n_units)
    epsilon = 1e-6
    analytic = ei_scalar_gain_jacobian_tangent(
        network,
        state,
        axis,
        gain_strength=0.35,
        belief=0.4,
    )
    high = ei_continuous_jacobian(
        network,
        state,
        axis,
        gain_strength=0.35,
        belief=0.4 + epsilon,
    )
    low = ei_continuous_jacobian(
        network,
        state,
        axis,
        gain_strength=0.35,
        belief=0.4 - epsilon,
    )
    np.testing.assert_allclose(
        analytic, (high - low) / (2.0 * epsilon), rtol=5e-7, atol=1e-11
    )


def test_projected_normal_decay_recovers_diagonal_linear_system() -> None:
    jacobian = np.diag([-0.1, -0.2, -1.0, -2.0])
    tangent_basis = np.eye(4)[:, :2]
    summary = projected_normal_linear_summary(jacobian, tangent_basis, horizon=0.5)
    assert summary.normal_dimension == 2
    assert summary.max_real_part == pytest.approx(-1.0)
    assert summary.decay_ratio == pytest.approx(np.exp(-0.5))
    assert summary.normal_to_tangent_coupling_norm == pytest.approx(0.0)


def test_rate_basis_is_pulled_back_into_preactivation_coordinates() -> None:
    network = _stable_high_rank_network()
    state = np.linspace(0.08, 0.29, network.n_units)
    axis = np.linspace(-0.8, 0.7, network.n_units)
    belief = 0.4
    strength = 0.35
    gain = 1.0 + strength * (2.0 * belief - 1.0) * axis
    derivative = gain * (1.0 - np.tanh(gain * state) ** 2)
    state_basis, _ = np.linalg.qr(
        np.random.default_rng(13).normal(size=(network.n_units, 2))
    )
    rate_basis = derivative[:, None] * state_basis
    pulled = pullback_rate_tangent_basis(
        network,
        state,
        axis,
        rate_basis,
        gain_strength=strength,
        belief=belief,
    )
    assert pulled.eligible
    assert pulled.tangent_dimension == 2
    assert pulled.relative_rate_reconstruction_residual < 1e-14
    expected_projector = state_basis @ state_basis.T
    actual_projector = pulled.preactivation_basis @ pulled.preactivation_basis.T
    np.testing.assert_allclose(actual_projector, expected_projector, atol=1e-12)

    rectified = EIRateNetwork(
        n_units=4,
        connection_probability=1.0,
        activation="rectified_tanh",
        allow_self_connections=True,
        seed=2,
    )
    with pytest.raises(ValueError, match="no locally representable"):
        pullback_rate_tangent_basis(
            rectified,
            -np.ones(4),
            np.linspace(-0.5, 0.5, 4),
            np.eye(4)[:, :1],
            gain_strength=0.2,
            belief=0.5,
        )


def test_effective_control_audit_rejects_invalid_feature_and_gate_shapes() -> None:
    network = _stable_high_rank_network()
    with pytest.raises(ValueError, match=r"3 \* n_units"):
        fit_soft_epoch_dynamics(
            np.ones((3, 8)),
            np.full(3, 0.5),
            n_units=8,
            latent_dim=2,
        )
    with pytest.raises(ValueError, match="sum to one"):
        fit_soft_epoch_dynamics(
            np.ones((3, 24)),
            np.ones((3, 2)),
            n_units=8,
            latent_dim=2,
        )
    with pytest.raises(ValueError, match="leave a normal space"):
        projected_normal_linear_summary(np.eye(2), np.eye(2), horizon=1.0)
    with pytest.raises(ValueError, match="gain_strength"):
        ei_continuous_jacobian(
            network,
            np.zeros(8),
            np.ones(8),
            gain_strength=1.0,
            belief=0.5,
        )
