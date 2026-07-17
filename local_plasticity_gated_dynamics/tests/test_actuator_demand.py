"""Tests for reachability-weighted actuator-demand theory."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.actuator_demand import (
    actuator_demand_from_gramians,
    actuator_demand_index,
    control_gramians,
    finite_horizon_local_demand,
    projected_transition_rank,
    state_input_cross_energy,
    transition_rank_requirement,
)


def test_scalar_infinite_horizon_demand_matches_analytic_solution() -> None:
    transition = np.array([[0.5]])
    input_matrix = np.array([[2.0]])
    observation = np.array([[3.0]])
    receipt = actuator_demand_index(
        transition,
        input_matrix,
        observation,
        np.array([[0.1]]),
        np.array([[0.2]]),
    )
    expected_state = 0.8
    expected_input = np.sqrt(12.0) * 0.2
    assert receipt.controllability_gramian[0, 0] == pytest.approx(16.0 / 3.0)
    assert receipt.observability_gramian[0, 0] == pytest.approx(12.0)
    assert receipt.state_demand == pytest.approx(expected_state)
    assert receipt.input_demand == pytest.approx(expected_input)
    assert receipt.state_fraction == pytest.approx(
        expected_state / (expected_state + expected_input)
    )
    assert receipt.projected_transition_rank == 1


def test_finite_horizon_gramians_use_registered_lags_only() -> None:
    transition = np.diag([0.5, 0.25])
    input_matrix = np.eye(2)
    observation = np.eye(2)
    controllability, observability = control_gramians(
        transition, input_matrix, observation, horizon=2
    )
    expected = np.eye(2) + transition @ transition.T
    np.testing.assert_allclose(controllability, expected, atol=1e-12)
    np.testing.assert_allclose(observability, expected, atol=1e-12)


def test_state_fraction_has_registered_endpoints_and_rejects_null_task() -> None:
    transition = np.diag([0.4, 0.6])
    input_matrix = np.eye(2)
    observation = np.eye(2)
    delta_a = np.diag([0.1, -0.1])
    delta_b = np.array([[0.2, 0.0], [0.0, -0.2]])
    state_only = actuator_demand_index(
        transition, input_matrix, observation, delta_a, np.zeros((2, 2))
    )
    input_only = actuator_demand_index(
        transition, input_matrix, observation, np.zeros((2, 2)), delta_b
    )
    assert state_only.state_fraction == pytest.approx(1.0)
    assert input_only.state_fraction == pytest.approx(0.0)
    with pytest.raises(ValueError, match="jointly zero"):
        actuator_demand_index(
            transition,
            input_matrix,
            observation,
            np.zeros((2, 2)),
            np.zeros((2, 2)),
        )


def test_demand_index_is_invariant_to_orthogonal_state_and_input_rotations() -> None:
    rng = np.random.default_rng(7)
    transition = np.diag([0.2, 0.4, 0.7])
    input_matrix = rng.normal(size=(3, 2))
    observation = rng.normal(size=(2, 3))
    delta_a = rng.normal(size=(3, 3)) * 0.05
    delta_b = rng.normal(size=(3, 2)) * 0.05
    state_rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    input_rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    input_moment = np.array([[1.4, 0.2], [0.2, 0.7]])
    original = actuator_demand_index(
        transition,
        input_matrix,
        observation,
        delta_a,
        delta_b,
        input_second_moment=input_moment,
        horizon=8,
    )
    rotated = actuator_demand_index(
        state_rotation @ transition @ state_rotation.T,
        state_rotation @ input_matrix @ input_rotation.T,
        observation @ state_rotation.T,
        state_rotation @ delta_a @ state_rotation.T,
        state_rotation @ delta_b @ input_rotation.T,
        input_second_moment=input_rotation @ input_moment @ input_rotation.T,
        horizon=8,
    )
    assert rotated.state_demand == pytest.approx(original.state_demand, abs=1e-12)
    assert rotated.input_demand == pytest.approx(original.input_demand, abs=1e-12)
    assert rotated.state_fraction == pytest.approx(
        original.state_fraction, abs=1e-12
    )
    assert (
        rotated.projected_transition_rank
        == original.projected_transition_rank
    )


def test_projected_rank_ignores_unreachable_or_unobservable_directions() -> None:
    delta_a = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    controllability = np.diag([2.0, 1.0, 0.0])
    observability = np.diag([1.0, 0.0, 3.0])
    assert projected_transition_rank(
        delta_a, controllability, observability
    ) == 1


def test_input_second_moment_enters_input_demand_and_both_gramians() -> None:
    transition = np.zeros((2, 2))
    input_matrix = np.eye(2)
    observation = np.eye(2)
    input_moment = np.diag([4.0, 1.0])
    receipt = actuator_demand_index(
        transition,
        input_matrix,
        observation,
        np.diag([0.5, 0.0]),
        np.eye(2),
        input_second_moment=input_moment,
    )
    np.testing.assert_allclose(receipt.controllability_gramian, input_moment)
    assert receipt.state_demand == pytest.approx(1.0)
    assert receipt.input_demand == pytest.approx(np.sqrt(5.0))
    assert receipt.state_energy_fraction == pytest.approx(1.0 / 6.0)


def test_precomputed_gramians_and_rank_tail_audit() -> None:
    transition = np.diag([0.2, 0.3, 0.4])
    controllability, observability = control_gramians(
        transition, np.eye(3), np.eye(3)
    )
    delta_a = np.diag([3.0, 2.0, 1.0])
    receipt = actuator_demand_from_gramians(
        delta_a,
        np.zeros((3, 2)),
        controllability,
        observability,
        input_second_moment=np.eye(2),
    )
    audit = transition_rank_requirement(
        delta_a,
        controllability,
        observability,
        candidate_ranks=(0, 1, 2, 3),
    )
    assert receipt.state_fraction == pytest.approx(1.0)
    assert audit.raw_rank == 3
    assert audit.projected_rank == 3
    assert audit.tail_energy_fractions[0] == pytest.approx(1.0)
    assert audit.tail_energy_fractions[-1] == pytest.approx(0.0)
    assert audit.energy_rank_99 == 3
    assert not audit.weighted_singular_values.flags.writeable


def test_cross_energy_detects_state_input_dependence() -> None:
    delta_a = np.eye(2)
    delta_b = np.eye(2)
    observability = np.diag([2.0, 1.0])
    cross = np.diag([0.25, 0.5])
    assert state_input_cross_energy(
        delta_a, delta_b, observability, cross
    ) == pytest.approx(2.0)


def test_finite_horizon_local_demand_uses_remaining_output_horizon() -> None:
    transition = np.array([[0.5]])
    observation = np.array([[1.0]])
    delta_a = np.array([[2.0]])
    delta_b = np.array([[3.0]])
    state_moments = np.array([[[1.0]], [[4.0]]])
    input_moments = np.array([[[1.0]], [[0.0]]])
    receipt = finite_horizon_local_demand(
        transition,
        observation,
        delta_a,
        delta_b,
        state_moments,
        input_moments,
    )
    # Remaining observability is 1.25 at t=0 and 1 at t=1; weights are 1/2.
    expected_state_energy = 0.5 * (1.25 * 4.0 * 1.0 + 1.0 * 4.0 * 4.0)
    expected_input_energy = 0.5 * (1.25 * 9.0)
    assert receipt.state_demand == pytest.approx(np.sqrt(expected_state_energy))
    assert receipt.input_demand == pytest.approx(np.sqrt(expected_input_energy))
    assert receipt.horizon == 2
    assert receipt.marginal_decomposition_valid


def test_finite_horizon_cross_moment_can_invalidate_marginal_chi() -> None:
    receipt = finite_horizon_local_demand(
        np.array([[0.0]]),
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([[[1.0]]]),
        np.array([[[1.0]]]),
        state_input_cross_moments=np.array([[[0.5]]]),
        cross_relative_tolerance=0.1,
    )
    assert receipt.cross_energy == pytest.approx(1.0)
    assert receipt.cross_relative_magnitude == pytest.approx(0.5)
    assert not receipt.marginal_decomposition_valid


def test_demand_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="strictly stable"):
        control_gramians(np.eye(2), np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="positive"):
        control_gramians(0.5 * np.eye(2), np.eye(2), np.eye(2), horizon=0)
    with pytest.raises(ValueError, match="input_demand must match"):
        actuator_demand_index(
            0.5 * np.eye(2),
            np.eye(2),
            np.eye(2),
            np.eye(2),
            np.ones((2, 1)),
        )
    with pytest.raises(ValueError, match="symmetric"):
        actuator_demand_index(
            0.5 * np.eye(2),
            np.eye(2),
            np.eye(2),
            np.eye(2),
            np.eye(2),
            input_second_moment=np.array([[1.0, 1.0], [0.0, 1.0]]),
        )
