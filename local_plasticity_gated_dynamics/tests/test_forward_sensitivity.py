"""Exact forward-sensitivity contracts for the Exp23 controller audit."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import src.plasticity.forward_sensitivity as sensitivity_module
from src.models.ei_rate_network import EIRateNetwork
from src.plasticity.forward_sensitivity import (
    epoch_mean_readout_gradient,
    exact_forward_gradient,
    exact_forward_sensitivities,
    frozen_gain_axis_step_partials,
    simulate_frozen_gain_axis_trajectory,
)


def test_axis_trajectory_matches_ei_rate_network_forward_dynamics() -> None:
    network = EIRateNetwork(
        6,
        n_inputs=2,
        connection_probability=0.8,
        activation="rectified_tanh",
        seed=19,
    )
    rng = np.random.default_rng(27)
    inputs = rng.normal(scale=0.2, size=(7, 2))
    beliefs = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 0.8, 1.0])
    axis = rng.normal(scale=0.15, size=network.n_units)
    gains = np.clip(1.0 + beliefs[:, None] * axis[None, :], 0.5, 1.5)
    expected = network.run(inputs, gains=gains)

    observed = simulate_frozen_gain_axis_trajectory(
        inputs @ network.input_weights.T,
        beliefs,
        axis,
        network.recurrent_weights,
        network.dt / network.time_constants,
        gain_min=0.5,
        gain_max=1.5,
        activation="rectified_tanh",
    )
    np.testing.assert_allclose(observed.states, expected.x, atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(observed.rates, expected.rates, atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(observed.gains, gains)
    assert observed.augmented_state_jacobians is None
    assert observed.augmented_control_jacobians is None
    assert observed.augmented_sensitivities is None
    assert not observed.stored_augmented_partials


def test_exact_recurrence_and_loss_contraction_match_manual_linear_system() -> None:
    jacobians = np.array(
        [
            [[0.8, 0.2], [-0.1, 0.7]],
            [[0.6, -0.3], [0.4, 0.5]],
            [[0.9, 0.0], [0.2, 0.4]],
        ]
    )
    control = np.array(
        [
            [[1.0, 0.0], [0.0, 0.5]],
            [[0.2, -0.1], [0.3, 0.7]],
            [[-0.4, 0.2], [0.1, 0.6]],
        ]
    )
    initial = np.array([[0.1, 0.0], [0.0, -0.2]])
    result = exact_forward_sensitivities(
        jacobians,
        control,
        initial_sensitivity=initial,
    )

    expected = [initial]
    for state_jacobian, direct in zip(jacobians, control, strict=True):
        expected.append(state_jacobian @ expected[-1] + direct)
    np.testing.assert_allclose(result.sensitivities, np.stack(expected))
    assert result.max_recurrence_residual == 0.0
    assert result.n_steps == 3
    assert not result.sensitivities.flags.writeable

    state_loss_gradient = np.array(
        [[0.0, 0.0], [1.0, -0.5], [0.2, 0.4], [-0.3, 0.8]]
    )
    direct = np.array([0.25, -0.1])
    gradient = exact_forward_gradient(
        result,
        state_loss_gradient,
        direct_parameter_gradients=direct,
    )
    manual = sum(
        state_loss_gradient[time] @ expected[time]
        for time in range(len(expected))
    ) + direct
    np.testing.assert_allclose(gradient.gradient, manual)
    np.testing.assert_allclose(
        gradient.gradient,
        gradient.state_contribution + gradient.direct_contribution,
    )


def test_ei_step_augmented_partials_match_central_finite_differences() -> None:
    rng = np.random.default_rng(41)
    n_units = 4
    state = rng.normal(scale=0.2, size=n_units)
    rates = np.maximum(np.tanh(state), 0.0)
    axis = rng.normal(scale=0.15, size=n_units)
    weights = rng.normal(scale=0.2, size=(n_units, n_units))
    drive = rng.normal(scale=0.1, size=n_units)
    step_fraction = np.array([0.05, 0.08, 0.06, 0.1])
    belief = 0.7
    weights_before = weights.copy()

    analytic = frozen_gain_axis_step_partials(
        state,
        rates,
        axis,
        weights,
        drive,
        step_fraction,
        signed_belief=belief,
        gain_min=0.5,
        gain_max=1.5,
        activation="tanh",
    )

    def transition(
        state_value: np.ndarray,
        rate_value: np.ndarray,
        axis_value: np.ndarray,
    ) -> np.ndarray:
        result = frozen_gain_axis_step_partials(
            state_value,
            rate_value,
            axis_value,
            weights,
            drive,
            step_fraction,
            signed_belief=belief,
            gain_min=0.5,
            gain_max=1.5,
            activation="tanh",
        )
        return np.concatenate((result.next_state, result.next_rates))

    epsilon = 1e-6
    augmented = np.concatenate((state, rates))
    finite_state = np.empty((2 * n_units, 2 * n_units))
    for coordinate in range(2 * n_units):
        plus = augmented.copy()
        minus = augmented.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        finite_state[:, coordinate] = (
            transition(plus[:n_units], plus[n_units:], axis)
            - transition(minus[:n_units], minus[n_units:], axis)
        ) / (2.0 * epsilon)
    finite_axis = np.empty((2 * n_units, n_units))
    for coordinate in range(n_units):
        plus = axis.copy()
        minus = axis.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        finite_axis[:, coordinate] = (
            transition(state, rates, plus) - transition(state, rates, minus)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        analytic.state_jacobian,
        finite_state,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        analytic.control_jacobian,
        finite_axis,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_array_equal(weights, weights_before)
    assert not analytic.state_jacobian.flags.writeable
    assert not analytic.control_jacobian.flags.writeable


def test_drive_gain_step_partials_match_central_finite_differences() -> None:
    rng = np.random.default_rng(57)
    n_units = 4
    state = rng.normal(scale=0.15, size=n_units)
    rates = np.tanh(state)
    axis = rng.normal(scale=0.1, size=n_units)
    weights = rng.normal(scale=0.12, size=(n_units, n_units))
    np.fill_diagonal(weights, 0.0)
    drive = rng.normal(scale=0.2, size=n_units)
    step_fraction = np.array([0.08, 0.05, 0.09, 0.07])

    analytic = frozen_gain_axis_step_partials(
        state,
        rates,
        axis,
        weights,
        drive,
        step_fraction,
        signed_belief=0.65,
        gain_min=0.5,
        gain_max=1.5,
        activation="tanh",
        gain_application="drive",
    )

    def transition(
        state_value: np.ndarray,
        rate_value: np.ndarray,
        axis_value: np.ndarray,
    ) -> np.ndarray:
        result = frozen_gain_axis_step_partials(
            state_value,
            rate_value,
            axis_value,
            weights,
            drive,
            step_fraction,
            signed_belief=0.65,
            gain_min=0.5,
            gain_max=1.5,
            activation="tanh",
            gain_application="drive",
        )
        return np.concatenate((result.next_state, result.next_rates))

    epsilon = 1e-6
    augmented = np.concatenate((state, rates))
    finite_state = np.empty((2 * n_units, 2 * n_units))
    for coordinate in range(2 * n_units):
        plus = augmented.copy()
        minus = augmented.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        finite_state[:, coordinate] = (
            transition(plus[:n_units], plus[n_units:], axis)
            - transition(minus[:n_units], minus[n_units:], axis)
        ) / (2.0 * epsilon)
    finite_axis = np.empty((2 * n_units, n_units))
    for coordinate in range(n_units):
        plus = axis.copy()
        minus = axis.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        finite_axis[:, coordinate] = (
            transition(state, rates, plus) - transition(state, rates, minus)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        analytic.state_jacobian,
        finite_state,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        analytic.control_jacobian,
        finite_axis,
        rtol=2e-6,
        atol=2e-7,
    )
    assert analytic.gain_application == "drive"
    assert np.any(np.abs(analytic.control_jacobian[:n_units]) > 0.0)


def test_full_rate_sensitivity_and_epoch_readout_gradient_match_finite_difference() -> None:
    rng = np.random.default_rng(83)
    n_steps = 6
    n_units = 3
    weights = rng.normal(scale=0.15, size=(n_units, n_units))
    drives = rng.normal(scale=0.12, size=(n_steps, n_units))
    beliefs = np.array([-0.8, -0.8, 0.2, 0.2, 0.9, 0.9])
    axis = np.array([0.1, -0.15, 0.2])
    step_fraction = np.array([0.08, 0.06, 0.1])
    labels = np.array(
        ["sensory", "sensory", "delay", "delay", "response", "response"]
    )
    coefficients = rng.normal(size=(3, n_units))

    trajectory = simulate_frozen_gain_axis_trajectory(
        drives,
        beliefs,
        axis,
        weights,
        step_fraction,
        gain_min=0.5,
        gain_max=1.5,
        activation="tanh",
        store_augmented_partials=True,
    )
    audit = epoch_mean_readout_gradient(
        trajectory.rate_sensitivities,
        labels,
        coefficients,
    )

    def score(axis_value: np.ndarray) -> float:
        result = simulate_frozen_gain_axis_trajectory(
            drives,
            beliefs,
            axis_value,
            weights,
            step_fraction,
            gain_min=0.5,
            gain_max=1.5,
            activation="tanh",
        )
        post_rates = result.rates[1:]
        value = 0.0
        for index, epoch in enumerate(("sensory", "delay", "response")):
            value += float(coefficients[index] @ post_rates[labels == epoch].mean(0))
        return value

    epsilon = 1e-6
    finite = np.empty(n_units)
    for coordinate in range(n_units):
        plus = axis.copy()
        minus = axis.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        finite[coordinate] = (score(plus) - score(minus)) / (2.0 * epsilon)

    np.testing.assert_allclose(
        audit.score_gradient,
        finite,
        rtol=3e-6,
        atol=3e-7,
    )
    assert trajectory.augmented_state_jacobians is not None
    assert trajectory.augmented_control_jacobians is not None
    assert trajectory.augmented_sensitivities is not None
    generic = exact_forward_sensitivities(
        trajectory.augmented_state_jacobians,
        trajectory.augmented_control_jacobians,
    )
    np.testing.assert_allclose(
        trajectory.augmented_sensitivities,
        generic.sensitivities,
        rtol=2e-14,
        atol=2e-14,
    )
    assert trajectory.stored_augmented_partials
    assert trajectory.states.shape == (n_steps + 1, n_units)
    assert trajectory.rates.shape == (n_steps + 1, n_units)
    assert trajectory.rate_sensitivities.shape == (
        n_steps + 1,
        n_units,
        n_units,
    )
    np.testing.assert_array_equal(audit.epoch_counts, np.array([2, 2, 2]))
    assert not trajectory.rate_sensitivities.flags.writeable


def test_saturated_gain_has_zero_direct_axis_derivative() -> None:
    partials = frozen_gain_axis_step_partials(
        np.array([0.2, -0.1]),
        np.array([0.1, 0.0]),
        np.array([2.0, -2.0]),
        np.array([[0.3, -0.2], [0.1, 0.4]]),
        np.zeros(2),
        np.array([0.1, 0.2]),
        signed_belief=1.0,
        gain_min=0.5,
        gain_max=1.5,
        activation="tanh",
    )
    np.testing.assert_array_equal(partials.saturation_mask, np.ones(2, dtype=bool))
    np.testing.assert_array_equal(partials.gain_derivative, np.zeros(2))
    np.testing.assert_array_equal(partials.control_jacobian, np.zeros((4, 2)))


@pytest.mark.parametrize(
    ("state_jacobians", "control_jacobians", "message"),
    [
        (
            np.zeros((2, 2, 3)),
            np.zeros((2, 2, 1)),
            "square",
        ),
        (
            np.zeros((2, 2, 2)),
            np.zeros((3, 2, 1)),
            "align",
        ),
    ],
)
def test_exact_sensitivity_rejects_incompatible_shapes(
    state_jacobians: np.ndarray,
    control_jacobians: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        exact_forward_sensitivities(state_jacobians, control_jacobians)


def test_forward_sensitivity_uses_no_gradient_engine() -> None:
    source = inspect.getsource(sensitivity_module)
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
