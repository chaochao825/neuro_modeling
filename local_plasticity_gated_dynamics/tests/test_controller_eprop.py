"""Local diagonal eligibility contracts for the Exp23 controller."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import src.plasticity.controller_eprop as eprop_module
from src.models.ei_rate_network import EIRateNetwork
from src.plasticity.controller_eprop import (
    DiagonalControllerEPropRule,
    apply_gain_bounds,
    block_local_gain_axis_eprop_sensitivities,
    diagonal_eprop_gradient,
    diagonal_eprop_sensitivities,
    diagonal_gain_axis_eprop_sensitivities,
)
from src.plasticity.forward_sensitivity import (
    exact_forward_sensitivities,
    simulate_frozen_gain_axis_trajectory,
)


def test_diagonal_eprop_is_exact_when_state_jacobians_are_diagonal() -> None:
    jacobian_diagonal = np.array(
        [[0.8, 0.4, -0.2], [0.5, 0.7, 0.1], [0.9, -0.3, 0.6]]
    )
    jacobians = np.array([np.diag(values) for values in jacobian_diagonal])
    control = np.arange(27.0).reshape(3, 3, 3) / 20.0

    exact = exact_forward_sensitivities(jacobians, control)
    local = diagonal_eprop_sensitivities(jacobians, control)
    np.testing.assert_allclose(local.eligibilities, exact.sensitivities)
    np.testing.assert_array_equal(local.jacobian_diagonal, jacobian_diagonal)
    assert local.approximation == "state_jacobian_diagonal_only"
    assert not local.eligibilities.flags.writeable


def test_off_diagonal_recurrence_is_explicitly_omitted() -> None:
    jacobians = np.array(
        [
            [[0.5, 0.8], [0.2, 0.4]],
            [[0.6, -0.3], [0.7, 0.1]],
        ]
    )
    control = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.2, 0.1], [-0.1, 0.3]],
        ]
    )
    local = diagonal_eprop_sensitivities(jacobians, control)
    exact = exact_forward_sensitivities(jacobians, control)

    expected_first = control[0]
    expected_second = np.diag(jacobians[1])[:, None] * expected_first + control[1]
    np.testing.assert_allclose(local.eligibilities[1], expected_first)
    np.testing.assert_allclose(local.eligibilities[2], expected_second)
    assert not np.allclose(local.eligibilities[2], exact.sensitivities[2])


def test_compact_gain_axis_eprop_matches_materialized_diagonal_recurrence() -> None:
    jacobian_diagonal = np.array(
        [
            [0.9, 0.8, 0.3, 0.2],
            [0.7, 0.6, -0.1, 0.4],
            [0.5, 0.4, 0.2, 0.1],
        ]
    )
    rate_direct = np.array([[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]])
    control = np.zeros((3, 4, 2))
    control[:, 2, 0] = rate_direct[:, 0]
    control[:, 3, 1] = rate_direct[:, 1]
    materialized = diagonal_eprop_sensitivities(
        jacobian_diagonal,
        control,
    )
    compact = diagonal_gain_axis_eprop_sensitivities(
        jacobian_diagonal,
        rate_direct,
    )
    np.testing.assert_allclose(compact.eligibilities, materialized.eligibilities)
    assert compact.direct_control_structure == "rate_block_diagonal"
    assert not compact.rate_axis_direct_derivatives.flags.writeable


def test_block_local_drive_gain_retains_memory_without_self_connections() -> None:
    network = EIRateNetwork(
        6,
        n_inputs=0,
        connection_probability=0.8,
        activation="tanh",
        allow_self_connections=False,
        seed=101,
    )
    np.testing.assert_array_equal(
        np.diag(network.recurrent_weights),
        np.zeros(network.n_units),
    )
    drives = np.zeros((4, network.n_units), dtype=np.float64)
    drives[0] = np.linspace(0.1, 0.4, network.n_units)
    trajectory = simulate_frozen_gain_axis_trajectory(
        drives,
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(network.n_units),
        network.recurrent_weights,
        network.dt / network.time_constants,
        gain_min=0.5,
        gain_max=1.5,
        activation="tanh",
        gain_application="drive",
    )
    local = block_local_gain_axis_eprop_sensitivities(
        trajectory.local_jacobian_blocks,
        trajectory.state_axis_direct_derivatives,
        trajectory.rate_axis_direct_derivatives,
    )
    assert np.allclose(trajectory.state_axis_direct_derivatives[1:], 0.0)
    first_rate = np.diag(local.eligibilities[1, network.n_units :])
    later_rate = np.diag(local.eligibilities[2, network.n_units :])
    assert np.any(np.abs(first_rate) > 0.0)
    assert np.any(np.abs(later_rate) > 0.0)
    assert local.approximation == "per_unit_state_rate_block_jacobian"
    assert not local.eligibilities.flags.writeable


def test_learning_signal_contraction_matches_stateful_online_rule() -> None:
    rule = DiagonalControllerEPropRule(
        2,
        3,
        learning_rate=0.2,
        l2_decay=0.1,
    )
    jacobians = np.array(
        [
            [[0.6, 0.2], [-0.1, 0.4]],
            [[0.8, -0.3], [0.5, 0.2]],
        ]
    )
    control = np.array(
        [
            [[1.0, 0.0, 0.5], [0.2, -0.3, 0.1]],
            [[-0.4, 0.6, 0.2], [0.3, 0.5, -0.2]],
        ]
    )
    signals = np.array([[0.7, -0.2], [-0.1, 0.8]])

    first = rule.advance(jacobians[0], control[0], signals[0])
    second = rule.advance(jacobians[1], control[1], signals[1])
    batch = diagonal_eprop_sensitivities(jacobians, control)
    batch_gradient = diagonal_eprop_gradient(
        batch,
        np.vstack((np.zeros(2), signals)),
    )
    np.testing.assert_allclose(second.eligibility, batch.eligibilities[-1])
    np.testing.assert_allclose(
        second.accumulated_gradient,
        batch_gradient.gradient,
    )
    assert first.step_index == 1
    assert second.step_index == 2

    parameters = np.array([0.2, -0.1, 0.3])
    activity_gradient = np.array([0.05, 0.0, -0.02])
    proposal = rule.propose(
        parameters,
        activity_gradient=activity_gradient,
    )
    expected_gradient = (
        batch_gradient.gradient + 0.1 * parameters + activity_gradient
    )
    np.testing.assert_allclose(proposal.estimated_gradient, expected_gradient)
    np.testing.assert_allclose(proposal.raw_update, -0.2 * expected_gradient)
    assert proposal.raw_l1 == pytest.approx(np.sum(np.abs(proposal.raw_update)))
    assert proposal.raw_l2 == pytest.approx(np.linalg.norm(proposal.raw_update))
    assert proposal.steps_accumulated == 2
    assert not proposal.recurrent_parameters_trainable


def test_gain_bound_application_preserves_direction_with_one_global_scale() -> None:
    parameters = np.zeros(3)
    proposed = np.array([1.0, -2.0, 0.5])
    beliefs = np.array([-1.0, 0.0, 1.0])
    application = apply_gain_bounds(
        parameters,
        proposed,
        beliefs,
        gain_min=0.5,
        gain_max=1.5,
    )
    assert application.scale_factor == pytest.approx(0.25)
    np.testing.assert_allclose(application.applied_update, 0.25 * proposed)
    assert application.gain_min_observed == pytest.approx(0.5)
    assert application.gain_max_observed == pytest.approx(1.5)
    assert application.constrained
    assert not application.applied_update.flags.writeable

    unconstrained = apply_gain_bounds(
        parameters,
        0.1 * proposed,
        beliefs,
        gain_min=0.5,
        gain_max=1.5,
    )
    assert unconstrained.scale_factor == 1.0
    assert not unconstrained.constrained


def test_rejected_online_step_does_not_mutate_rule_state() -> None:
    rule = DiagonalControllerEPropRule(2, 2, learning_rate=0.1)
    before_trace = rule.eligibility
    before_gradient = rule.accumulated_gradient
    with pytest.raises(ValueError, match="learning_signal"):
        rule.advance(
            np.eye(2),
            np.ones((2, 2)),
            np.ones(3),
        )
    np.testing.assert_array_equal(rule.eligibility, before_trace)
    np.testing.assert_array_equal(rule.accumulated_gradient, before_gradient)
    assert rule.steps_accumulated == 0


def test_controller_eprop_module_has_no_gradient_engine_or_reverse_time_path() -> None:
    source = inspect.getsource(eprop_module)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert "autograd" not in source.lower()
    assert ".backward(" not in source
    assert "reversed(" not in source
