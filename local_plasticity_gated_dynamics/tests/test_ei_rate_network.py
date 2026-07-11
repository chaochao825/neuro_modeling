"""Fast contracts for the NumPy Dale-constrained E/I network."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.models.ei_rate_network import EIRateNetwork, EIRateState, dale_project


def test_sparse_dale_columns_and_explicit_weight_components() -> None:
    network = EIRateNetwork(
        20,
        excitatory_fraction=0.8,
        connection_probability=0.35,
        seed=7,
    )

    assert network.n_excitatory == 16
    assert network.n_inhibitory == 4
    assert network.connectivity_mask.shape == (20, 20)
    assert not np.any(np.diag(network.connectivity_mask))
    assert np.any(network.connectivity_mask)
    assert np.any(~network.connectivity_mask)

    weights = network.recurrent_weights
    assert np.all(weights[~network.connectivity_mask] == 0.0)
    assert np.all(weights[:, network.excitatory_mask] >= 0.0)
    assert np.all(weights[:, network.inhibitory_mask] <= 0.0)
    np.testing.assert_allclose(
        weights,
        network.W_bulk + network.W_task + network.W_homeo + network.W_normalization,
    )
    for component in (
        network.W_bulk,
        network.W_task,
        network.W_homeo,
        network.W_normalization,
    ):
        with pytest.raises(ValueError, match="read-only"):
            component[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        network.fan_in_l1_target[0] = 1.0
    assert network.validate_dale()


def test_e_and_i_postsynaptic_units_use_distinct_time_constants() -> None:
    network = EIRateNetwork(
        10,
        connection_probability=0.0,
        tau_e=20.0,
        tau_i=10.0,
        dt=2.0,
        bulk_gain=0.0,
        seed=1,
    )
    x = np.ones(10)
    rates = np.zeros(10)
    step = network.step(EIRateState(x=x, rates=rates))

    np.testing.assert_allclose(step.state.x[network.excitatory_mask], 0.9)
    np.testing.assert_allclose(step.state.x[network.inhibitory_mask], 0.8)
    assert np.all(step.state.rates >= 0.0)


def test_component_updates_are_projected_against_effective_weights() -> None:
    network = EIRateNetwork(
        10,
        connection_probability=1.0,
        allow_self_connections=True,
        seed=11,
    )
    before = network.recurrent_weights
    target_fan_in = network.fan_in_l1_target.copy()
    violating = np.zeros((10, 10))
    violating[:, network.excitatory_mask] = -0.05
    violating[:, network.inhibitory_mask] = 0.05

    task_application = network.apply_task_update(violating)
    after_task = network.recurrent_weights
    np.testing.assert_allclose(after_task - before, task_application.total_update)
    assert np.linalg.norm(task_application.local_update) > 0.0
    np.testing.assert_allclose(network.W_task, task_application.local_update)
    assert network.validate_dale()
    assert np.all(after_task[:, network.excitatory_mask] >= 0.0)
    assert np.all(after_task[:, network.inhibitory_mask] <= 0.0)
    np.testing.assert_allclose(network.fan_in_l1_norms, target_fan_in)

    homeostatic = np.zeros((10, 10))
    homeostatic[np.ix_(network.excitatory_mask, network.inhibitory_mask)] = -0.01
    before_homeo = network.recurrent_weights
    homeostatic_application = network.apply_homeostatic_update(homeostatic)
    np.testing.assert_allclose(
        network.recurrent_weights - before_homeo,
        homeostatic_application.total_update,
    )
    np.testing.assert_allclose(
        network.recurrent_weights,
        network.W_bulk + network.W_task + network.W_homeo + network.W_normalization,
    )
    assert network.validate_dale()
    np.testing.assert_allclose(network.fan_in_l1_norms, target_fan_in)


def test_trusted_projected_update_matches_validated_public_path() -> None:
    public = EIRateNetwork(
        12,
        connection_probability=0.4,
        normalize_fan_in_after_update=True,
        seed=41,
    )
    trusted = copy.deepcopy(public)
    proposal = np.random.default_rng(3).normal(scale=0.01, size=(12, 12))
    before = public.recurrent_weights
    candidate = before + np.where(public.connectivity_mask, proposal, 0.0)
    projected = dale_project(
        candidate, public.connectivity_mask, public.presynaptic_signs
    )
    preprojected_update = projected - before

    expected = public.apply_task_update(proposal)
    actual = trusted._apply_projected_task_update(preprojected_update)

    np.testing.assert_allclose(actual.local_update, expected.local_update)
    np.testing.assert_allclose(
        actual.normalization_correction, expected.normalization_correction
    )
    np.testing.assert_allclose(actual.total_update, expected.total_update)
    np.testing.assert_allclose(trusted.recurrent_weights, public.recurrent_weights)
    np.testing.assert_allclose(
        trusted.recurrent_weights,
        trusted.W_bulk + trusted.W_task + trusted.W_homeo + trusted.W_normalization,
    )


def test_failed_normalization_is_atomic_across_components_and_cache() -> None:
    network = EIRateNetwork(8, connection_probability=1.0, seed=4)
    before_weights = network.recurrent_weights
    before_task = network.W_task
    before_normalization = network.W_normalization
    network._fan_in_l1_target = np.ones(3)

    with pytest.raises(ValueError):
        network.apply_task_update(np.full((8, 8), 1e-3))

    np.testing.assert_array_equal(network.recurrent_weights, before_weights)
    np.testing.assert_array_equal(network.W_task, before_task)
    np.testing.assert_array_equal(network.W_normalization, before_normalization)
    np.testing.assert_array_equal(
        network.recurrent_weights,
        network.W_bulk + network.W_task + network.W_homeo + network.W_normalization,
    )


def test_normalization_correction_does_not_pollute_local_components() -> None:
    network = EIRateNetwork(
        8,
        connection_probability=1.0,
        allow_self_connections=True,
        seed=23,
    )
    task_proposal = 0.001 * np.outer(
        np.linspace(1.0, 2.0, 8), network.presynaptic_signs
    )
    task = network.apply_task_update(task_proposal)
    assert np.linalg.matrix_rank(task.local_update, tol=1e-12) == 1
    np.testing.assert_allclose(network.W_task, task.local_update)
    np.testing.assert_allclose(network.W_normalization, task.normalization_correction)

    homeostatic_proposal = np.zeros((8, 8))
    homeostatic_proposal[
        np.ix_(network.excitatory_mask, network.inhibitory_mask)
    ] = -0.002
    before_homeostatic_component = network.W_homeo.copy()
    homeostatic = network.apply_homeostatic_update(homeostatic_proposal)
    local_delta = network.W_homeo - before_homeostatic_component
    np.testing.assert_allclose(local_delta, homeostatic.local_update)
    assert np.all(local_delta[:, network.excitatory_mask] == 0.0)
    assert np.any(
        homeostatic.normalization_correction[:, network.excitatory_mask] != 0.0
    )


def test_zero_bulk_reference_does_not_erase_task_learning() -> None:
    network = EIRateNetwork(
        8,
        connection_probability=1.0,
        allow_self_connections=True,
        bulk_gain=0.0,
        seed=29,
    )
    proposal = 0.01 * np.outer(np.ones(8), network.presynaptic_signs)
    application = network.apply_task_update(proposal)
    assert application.local_l1_cost > 0.0
    assert application.normalization_l1_cost == 0.0
    np.testing.assert_allclose(application.total_update, application.local_update)
    np.testing.assert_allclose(network.recurrent_weights, proposal)


def test_inhibitory_gain_scales_only_inhibitory_outgoing_columns() -> None:
    unit_gain = EIRateNetwork(
        10,
        connection_probability=1.0,
        allow_self_connections=True,
        inhibitory_gain=1.0,
        seed=19,
    )
    stronger_inhibition = EIRateNetwork(
        10,
        connection_probability=1.0,
        allow_self_connections=True,
        inhibitory_gain=3.0,
        seed=19,
    )
    np.testing.assert_allclose(
        stronger_inhibition.W_bulk[:, stronger_inhibition.excitatory_mask],
        unit_gain.W_bulk[:, unit_gain.excitatory_mask],
    )
    np.testing.assert_allclose(
        stronger_inhibition.W_bulk[:, stronger_inhibition.inhibitory_mask],
        3.0 * unit_gain.W_bulk[:, unit_gain.inhibitory_mask],
    )


def test_step_accepts_md_gain_and_run_is_time_major() -> None:
    network = EIRateNetwork(8, n_inputs=2, connection_probability=0.0, seed=3)
    state = network.initial_state()
    output = network.step(state, np.array([1.0, -1.0]), gain=np.linspace(0.5, 1.5, 8))
    assert output.state.x.shape == (8,)
    assert output.state.rates.shape == (8,)

    trajectory = network.run(np.zeros((4, 2)), gains=np.ones((4, 8)))
    assert trajectory.x.shape == (5, 8)
    assert trajectory.rates.shape == (5, 8)


def test_trial_batch_matches_independent_fine_step_runs() -> None:
    network = EIRateNetwork(
        8,
        n_inputs=2,
        connection_probability=0.25,
        tau_e=20.0,
        tau_i=10.0,
        dt=5.0,
        seed=31,
    )
    inputs = np.array(
        [
            [[1.0, -0.5], [0.0, 0.25]],
            [[-0.25, 0.75], [0.5, 0.0]],
        ]
    )
    gains = np.ones((2, 2, 8))
    gains[0, 1] = np.linspace(0.8, 1.2, 8)
    batched = network.run_trial_batch(inputs, gains=gains, substeps=2)

    assert batched.x.shape == (2, 3, 8)
    assert batched.rates.shape == (2, 3, 8)
    assert batched.integration_substeps == 2
    assert batched.substep_firing_sum > 0.0
    assert batched.substep_input_event_sum > 0.0
    assert batched.substep_recurrent_event_sum >= 0.0
    for trial in range(2):
        fine_inputs = np.repeat(inputs[trial], 2, axis=0)
        fine_gains = np.repeat(gains[trial], 2, axis=0)
        independent = network.run(fine_inputs, gains=fine_gains)
        np.testing.assert_allclose(batched.x[trial, 1:], independent.x[[2, 4]])
        np.testing.assert_allclose(batched.rates[trial, 1:], independent.rates[[2, 4]])


def test_trial_batch_rejects_misaligned_gains_and_invalid_substeps() -> None:
    network = EIRateNetwork(8, n_inputs=2, seed=5)
    inputs = np.zeros((3, 4, 2))
    with pytest.raises(ValueError, match="gains"):
        network.run_trial_batch(inputs, gains=np.ones((3, 4, 7)))
    with pytest.raises(ValueError, match="substeps"):
        network.run_trial_batch(inputs, substeps=0)


@pytest.mark.parametrize(
    "kwargs, exception",
    [
        ({"n_units": 1}, ValueError),
        ({"n_units": 8, "excitatory_fraction": 1.0}, ValueError),
        ({"n_units": 8, "connection_probability": 1.1}, ValueError),
        ({"n_units": 8, "tau_i": 0.0}, ValueError),
        ({"n_units": 8, "tau_i": 10.0, "dt": 11.0}, ValueError),
        ({"n_units": 8, "activation": "relu"}, ValueError),
        ({"n_units": 8, "inhibitory_gain": 0.0}, ValueError),
        ({"n_units": 8, "allow_self_connections": 1}, TypeError),
        ({"n_units": 8, "normalize_fan_in_after_update": 1}, TypeError),
    ],
)
def test_configuration_validation(
    kwargs: dict[str, object], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        EIRateNetwork(**kwargs)


def test_update_and_input_shape_validation() -> None:
    network = EIRateNetwork(8, n_inputs=2, seed=4)
    with pytest.raises(ValueError):
        network.apply_task_update(np.zeros((7, 7)))
    with pytest.raises(ValueError):
        network.step(network.initial_state(), np.zeros(3))
    with pytest.raises(ValueError):
        network.step(network.initial_state(), np.zeros(2), gain=np.ones(7))


def test_dale_project_rejects_nonbinary_connectivity_masks() -> None:
    from src.models.ei_rate_network import dale_project

    with pytest.raises(ValueError, match="binary"):
        dale_project(np.eye(2), np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([1, -1]))
