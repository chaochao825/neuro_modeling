"""Contracts for continuous receiver trajectories and train-only dynamics."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import exp19_belief_ei_effective_dynamics as exp19
from src.analysis.trajectory_control_dynamics import (
    belief_manifold_geometry,
    fit_trajectory_koopman,
    fixed_drive_attractor_probe,
    nonlinear_perturbation_recovery,
    persistence_trajectory_score,
)
from src.models.belief_gain import balanced_gain_axis
from src.models.context_belief import MDRecurrentBeliefGate
from src.models.ei_rate_network import EIRateNetwork
from src.tasks.hidden_context import (
    generate_hidden_context,
    make_hidden_context_random_tape,
)
from src.training.hidden_context_ei import simulate_receiver
from src.training.hidden_context_gate import split_hidden_context_dataset
from src.utils.reproducibility import derive_seed


def _smoke_receiver_inputs(seed: int = 0):
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "smoke"
        / "exp19_belief_ei_effective_dynamics.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    task = exp19._task_config(config)
    tape = make_hidden_context_random_tape(task, seed=seed)
    dataset = generate_hidden_context(task, seed=seed, random_tape=tape)
    splits = split_hidden_context_dataset(
        dataset,
        outer_test_fraction=float(config["outer_test_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        seed=seed,
    )
    network = exp19._network(config, task, seed)
    axis = balanced_gain_axis(
        network.excitatory_mask,
        seed=derive_seed(seed, "trajectory-test", "gain-axis"),
    )
    gate = MDRecurrentBeliefGate(
        seed=derive_seed(seed, "trajectory-test", "gate"),
        **dict(config["md_gate"]),
    ).fit(splits.train.gate)
    prediction = gate.predict(splits.train.gate)
    return config, splits.train, network, axis, prediction.context_probability


def test_full_substep_recording_preserves_legacy_receiver_outputs() -> None:
    config, train, network, axis, posterior = _smoke_receiver_inputs()
    common = dict(
        network=network,
        dataset=train,
        posterior_state1=posterior,
        gain_axis=axis,
        gain_strength=float(config["gain_strength"]),
        integration_substeps=int(config["integration_substeps"]),
        trial_batch_size=17,
        pathway_gating=True,
        population_gain=True,
    )
    legacy = simulate_receiver(**common)
    recorded = simulate_receiver(**common, record_substeps=True)

    np.testing.assert_array_equal(recorded.features, legacy.features)
    np.testing.assert_array_equal(recorded.mean_x, legacy.mean_x)
    np.testing.assert_array_equal(recorded.mean_gain, legacy.mean_gain)
    assert recorded.input_event_sum == legacy.input_event_sum
    assert recorded.recurrent_event_sum == legacy.recurrent_event_sum
    assert recorded.firing_sum == legacy.firing_sum
    assert recorded.receiver_fingerprint == legacy.receiver_fingerprint
    assert recorded.trajectory_sequence_scope == "trial_reset_state"
    assert recorded.full_x_trajectory is not None
    assert recorded.full_rate_trajectory is not None
    assert recorded.full_x_trajectory.shape[0] == train.task.trial_ids.size
    expected_states = (
        train.task.inputs.shape[1] * int(config["integration_substeps"]) + 1
    )
    assert recorded.full_x_trajectory.shape[1] == expected_states
    np.testing.assert_array_equal(recorded.full_x_trajectory[:, 0], 0.0)
    assert not recorded.full_x_trajectory.flags.writeable
    assert recorded.receiver_raw_sensory_inputs is recorded.raw_inputs


def test_episode_continuous_receiver_carries_state_and_records_applied_belief() -> None:
    config, train, network, axis, posterior = _smoke_receiver_inputs(seed=1)
    simulation = simulate_receiver(
        network,
        train,
        posterior,
        axis,
        gain_strength=float(config["gain_strength"]),
        integration_substeps=int(config["integration_substeps"]),
        trial_batch_size=2,
        pathway_gating=True,
        population_gain=True,
        record_substeps=True,
        continuous_episodes=True,
    )

    assert simulation.trajectory_sequence_scope == "episode_continuous_state"
    assert simulation.full_x_trajectory is not None
    assert simulation.trajectory_belief is not None
    assert simulation.trajectory_posterior is not None
    assert simulation.trajectory_population_gain_belief is not None
    assert simulation.trajectory_epoch is not None
    np.testing.assert_array_equal(simulation.full_x_trajectory[:, 0], 0.0)
    trial_fine_steps = train.task.inputs.shape[1] * int(config["integration_substeps"])
    assert np.any(np.abs(simulation.full_x_trajectory[:, trial_fine_steps]) > 0.0)
    cue = simulation.trajectory_epoch == "cue"
    np.testing.assert_array_equal(simulation.trajectory_belief[:, cue], 0.5)
    np.testing.assert_array_equal(
        simulation.trajectory_population_gain_belief[:, cue], 0.5
    )
    assert np.any(np.abs(simulation.trajectory_posterior[:, cue] - 0.5) > 1e-12)


def test_actuator_specific_belief_receipts_do_not_invent_population_gain() -> None:
    config, train, network, axis, posterior = _smoke_receiver_inputs(seed=2)
    pathway_only = simulate_receiver(
        network,
        train,
        posterior,
        axis,
        gain_strength=float(config["gain_strength"]),
        integration_substeps=int(config["integration_substeps"]),
        trial_batch_size=31,
        pathway_gating=True,
        population_gain=False,
        record_substeps=True,
    )
    assert pathway_only.trajectory_population_gain_belief is not None
    np.testing.assert_array_equal(pathway_only.trajectory_population_gain_belief, 0.5)
    assert pathway_only.trajectory_pathway_belief is not None
    assert pathway_only.trajectory_epoch is not None
    sensory = pathway_only.trajectory_epoch == "sensory"
    nonsensory = ~sensory
    assert np.any(
        np.abs(pathway_only.trajectory_pathway_belief[:, sensory] - 0.5) > 1e-12
    )
    np.testing.assert_array_equal(pathway_only.trajectory_belief[:, nonsensory], 0.5)


def _shared_controlled_system(
    *,
    seed: int,
    n_sequences: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_map = np.array(
        [[1.0, 0.0, 0.5, -0.25], [0.0, 1.0, -0.25, 0.5]],
        dtype=float,
    )
    rate_map = np.array(
        [[0.5, -0.25, 1.0, 0.0], [0.25, 0.5, 0.0, 1.0]],
        dtype=float,
    )
    state0 = np.array([[0.82, 0.00], [0.00, 0.35]])
    state1 = np.array([[0.20, -0.55], [0.55, 0.72]])
    input_map = np.array([[0.25, -0.10], [0.05, 0.20]])
    controls = rng.normal(scale=0.5, size=(n_sequences, n_steps, 2))
    belief = rng.integers(0, 2, size=(n_sequences, n_steps)).astype(float)
    latent = np.empty((n_sequences, n_steps + 1, 2), dtype=float)
    latent[:, 0] = rng.normal(scale=0.4, size=(n_sequences, 2))
    for time in range(n_steps):
        p = belief[:, time, None]
        next0 = latent[:, time] @ state0.T
        next1 = latent[:, time] @ state1.T
        latent[:, time + 1] = (
            (1.0 - p) * next0 + p * next1 + controls[:, time] @ input_map.T
        )
    x = latent @ x_map
    rates = latent @ rate_map
    epoch = np.resize(
        np.array(["cue", "sensory", "delay", "response"], dtype="U8"),
        n_steps,
    )
    return x, rates, controls, belief, epoch


def _operator_identifiability_system(
    *,
    seed: int,
    neutral_cue: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_sequences = 32
    n_steps = 16
    n_units = 8
    x_scale = np.geomspace(0.2, 3.0, n_units)
    x = rng.normal(size=(n_sequences, n_steps + 1, n_units)) * x_scale
    rates = np.maximum(np.tanh(x), 0.0)
    controls = rng.normal(size=(n_sequences, n_steps, 2))
    epoch = np.resize(
        np.array(["cue", "sensory", "delay", "response"], dtype="U8"),
        n_steps,
    )
    belief = rng.uniform(0.1, 0.9, size=(n_sequences, n_steps))
    cue = epoch == "cue"
    if neutral_cue:
        belief[:, cue] = 0.5
    else:
        cue_values = np.where(np.arange(n_sequences) % 2 == 0, 0.2, 0.8)
        belief[:, cue] = cue_values[:, None]
    return x, rates, controls, belief, epoch


def _negative_zero_weight_replay():
    n_units = 6
    n_sequences = 5
    n_steps = 6
    network = EIRateNetwork(
        n_units,
        n_inputs=2,
        excitatory_fraction=0.5,
        connection_probability=0.0,
        tau_e=20.0,
        tau_i=20.0,
        dt=5.0,
        input_scale=0.0,
        activation="rectified_tanh",
        seed=2,
    )
    rng = np.random.default_rng(91)
    controls = np.zeros((n_sequences, n_steps, 2), dtype=float)
    belief = np.full((n_sequences, n_steps), 0.5, dtype=float)
    epoch = np.resize(
        np.array(["sensory", "delay", "response"], dtype="U8"),
        n_steps,
    )
    x = np.empty((n_sequences, n_steps + 1, n_units), dtype=float)
    rates = np.empty_like(x)
    for sequence in range(n_sequences):
        state = network.initial_state(-1.0 - rng.uniform(size=n_units))
        x[sequence, 0] = state.x
        rates[sequence, 0] = state.rates
        for time in range(n_steps):
            state = network.step(
                state,
                controls[sequence, time],
                gain=np.ones(n_units),
            ).state
            x[sequence, time + 1] = state.x
            rates[sequence, time + 1] = state.rates
    fitted = fit_trajectory_koopman(
        x,
        rates,
        controls,
        belief,
        epoch,
        integration_substeps=1,
        latent_dim=2,
        normalize_activity=True,
        belief_conditioned=False,
        operator_mode="common",
    )
    return (
        network,
        fitted,
        x,
        rates,
        controls,
        belief,
        epoch,
        np.zeros(n_units, dtype=float),
    )


def test_neutral_cue_constraint_resolves_the_registered_rank_deficiency() -> None:
    trajectories = _operator_identifiability_system(seed=44, neutral_cue=True)
    generic = fit_trajectory_koopman(
        *trajectories,
        integration_substeps=1,
        latent_dim=4,
        ridge=1e-8,
        normalize_activity=True,
        belief_conditioned=True,
        operator_mode="full",
    )
    constrained = fit_trajectory_koopman(
        *trajectories,
        integration_substeps=1,
        latent_dim=4,
        ridge=1e-8,
        normalize_activity=True,
        belief_conditioned=True,
        operator_mode="full_shared_neutral_cue",
        shared_preprocessing=generic,
    )

    assert generic.operator_design_rank == 19
    assert generic.operator_design_columns == 20
    assert generic.operator_unconstrained_columns == 20
    assert generic.operator_constraint == "none"
    assert constrained.operator_design_rank == 19
    assert constrained.operator_design_columns == 19
    assert constrained.operator_unconstrained_columns == 20
    assert constrained.operator_constraint == "shared_neutral_cue_coefficient"
    cue_coefficient = constrained.latent_dim + constrained.n_inputs
    np.testing.assert_array_equal(
        constrained.operator_state0[cue_coefficient],
        constrained.operator_state1[cue_coefficient],
    )
    np.testing.assert_allclose(
        constrained.operator_state0,
        generic.operator_state0,
        atol=2e-7,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        constrained.operator_state1,
        generic.operator_state1,
        atol=2e-7,
        rtol=1e-7,
    )
    generic_score = generic.score(*trajectories)
    constrained_score = constrained.score(*trajectories)
    assert constrained_score.one_step_normalized_mse == pytest.approx(
        generic_score.one_step_normalized_mse,
        abs=1e-14,
    )
    assert constrained_score.rollout_normalized_rmse == pytest.approx(
        generic_score.rollout_normalized_rmse,
        abs=1e-14,
    )


def test_generic_full_is_identifiable_when_cue_belief_varies() -> None:
    trajectories = _operator_identifiability_system(seed=45, neutral_cue=False)
    fitted = fit_trajectory_koopman(
        *trajectories,
        integration_substeps=1,
        latent_dim=4,
        ridge=1e-8,
        normalize_activity=True,
        belief_conditioned=True,
        operator_mode="full",
    )

    assert fitted.operator_design_rank == 20
    assert fitted.operator_design_columns == 20
    assert fitted.operator_unconstrained_columns == 20
    assert fitted.operator_constraint == "none"
    with pytest.raises(ValueError, match="requires exact cue belief 0.5"):
        fit_trajectory_koopman(
            *trajectories,
            integration_substeps=1,
            latent_dim=4,
            normalize_activity=True,
            belief_conditioned=True,
            operator_mode="full_shared_neutral_cue",
        )


def test_physical_x_tangent_basis_uses_raw_coordinates_and_is_rotation_invariant() -> (
    None
):
    trajectories = _shared_controlled_system(
        seed=18,
        n_sequences=16,
        n_steps=8,
    )
    fitted = fit_trajectory_koopman(
        *trajectories,
        integration_substeps=1,
        latent_dim=2,
        normalize_activity=True,
    )
    n_units = fitted.n_units
    components = np.zeros((2, 2 * n_units), dtype=float)
    components[0, 0] = 0.5
    components[0, 2] = 0.5
    components[0, n_units] = np.sqrt(0.5)
    components[1, 1] = 0.5
    components[1, 3] = 0.5
    components[1, n_units + 1] = np.sqrt(0.5)
    scale = np.ones(2 * n_units, dtype=float)
    scale[:n_units] = np.array([1.0, 1.0, 4.0, 2.0])
    state_pca = replace(
        fitted.state_pca,
        components_=components,
        scale_=scale,
    )
    scaled = replace(fitted, state_pca=state_pca)

    geometry = scaled.physical_x_tangent_basis
    raw_joint = scale[:, None] * components.T
    expected_left, expected_singular, _ = np.linalg.svd(
        raw_joint[:n_units],
        full_matrices=False,
    )
    expected_projector = expected_left[:, :2] @ expected_left[:, :2].T
    actual_projector = geometry.basis @ geometry.basis.T
    np.testing.assert_allclose(actual_projector, expected_projector, atol=1e-12)
    np.testing.assert_allclose(
        geometry.singular_values,
        expected_singular,
        atol=1e-12,
    )
    assert geometry.x_block_energy_fraction == pytest.approx(
        np.sum(raw_joint[:n_units] ** 2) / np.sum(raw_joint**2)
    )
    standardized_left, _, _ = np.linalg.svd(
        components.T[:n_units],
        full_matrices=False,
    )
    standardized_projector = standardized_left[:, :2] @ standardized_left[:, :2].T
    assert not np.allclose(actual_projector, standardized_projector)

    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    rotated = replace(
        scaled,
        state_pca=replace(state_pca, components_=rotation @ components),
    )
    rotated_basis = rotated.physical_x_tangent_basis.basis
    np.testing.assert_allclose(
        rotated_basis @ rotated_basis.T,
        actual_projector,
        atol=1e-12,
    )


def test_rank_deficient_physical_x_projection_fails_closed() -> None:
    (
        network,
        fitted,
        x,
        rates,
        controls,
        belief,
        epoch,
        gain_axis,
    ) = _negative_zero_weight_replay()
    components = np.zeros_like(fitted.state_pca.components_)
    components[0, 0] = 1.0
    components[1, fitted.n_units] = 1.0
    rank_deficient = replace(
        fitted,
        state_pca=replace(fitted.state_pca, components_=components),
    )

    assert rank_deficient.physical_x_tangent_basis.rank == 1
    with pytest.raises(ValueError, match="physical-x projection.*rank-deficient"):
        nonlinear_perturbation_recovery(
            network,
            rank_deficient,
            x,
            rates,
            controls,
            belief,
            epoch,
            gain_axis,
            gain_strength=0.0,
            integration_substeps=1,
            horizon_steps=2,
            amplitudes=(0.001,),
            n_references=4,
            seed=7,
        )


def test_physical_x_recovery_covers_rectified_inactive_states_and_matches_leak() -> (
    None
):
    (
        network,
        fitted,
        x,
        rates,
        controls,
        belief,
        epoch,
        gain_axis,
    ) = _negative_zero_weight_replay()
    common = dict(
        gain_strength=0.0,
        integration_substeps=1,
        horizon_steps=2,
        amplitudes=(0.001, 0.002),
        n_references=25,
        seed=11,
        baseline_replay_tolerance=1e-12,
    )
    first = nonlinear_perturbation_recovery(
        network,
        fitted,
        x,
        rates,
        controls,
        belief,
        epoch,
        gain_axis,
        **common,
    )
    second = nonlinear_perturbation_recovery(
        network,
        fitted,
        x,
        rates,
        controls,
        belief,
        epoch,
        gain_axis,
        **common,
    )

    assert np.all(x < 0.0)
    np.testing.assert_array_equal(rates, 0.0)
    assert first == second
    assert first.tangent_basis_space == ("train_joint_state_pca_physical_x_projection")
    assert first.tangent_basis_rank == fitted.latent_dim
    assert first.candidate_reference_count == 20
    assert first.sampled_reference_count == 20
    assert first.planned_reference_count == 25
    assert first.eligible_reference_count == 20
    assert first.sampled_reference_fraction == pytest.approx(0.8)
    assert first.eligible_sampled_reference_fraction == pytest.approx(1.0)
    assert first.eligible_reference_fraction == pytest.approx(0.8)
    assert first.normal_perturbation_count == 80
    contraction = abs(1.0 - network.dt / 20.0) ** first.horizon_steps
    assert first.normal_endpoint_ratio_median == pytest.approx(
        contraction,
        abs=2e-13,
    )
    assert first.normal_endpoint_ratio_maximum == pytest.approx(
        contraction,
        abs=2e-13,
    )
    assert first.tangent_endpoint_ratio_median == pytest.approx(
        contraction,
        abs=2e-13,
    )
    assert first.initial_normal_purity_median == pytest.approx(1.0, abs=1e-13)
    assert first.initial_tangent_purity_median == pytest.approx(1.0, abs=1e-13)
    assert first.baseline_replay_max_abs_error == 0.0

    corrupted_x = np.array(x, copy=True)
    corrupted_x[0, 2, 0] += 1e-4
    with pytest.raises(RuntimeError, match="baseline replay differs"):
        nonlinear_perturbation_recovery(
            network,
            fitted,
            corrupted_x,
            rates,
            controls,
            belief,
            epoch,
            gain_axis,
            **common,
        )
    corrupted_rates = np.array(rates, copy=True)
    corrupted_rates[0, 2, 0] = 1e-4
    with pytest.raises(RuntimeError, match="baseline replay differs"):
        nonlinear_perturbation_recovery(
            network,
            fitted,
            x,
            corrupted_rates,
            controls,
            belief,
            epoch,
            gain_axis,
            **common,
        )


def test_belief_koopman_beats_common_and_persistence_on_heldout_sequences() -> None:
    train = _shared_controlled_system(seed=3, n_sequences=24, n_steps=16)
    test = _shared_controlled_system(seed=9, n_sequences=8, n_steps=16)
    belief_model = fit_trajectory_koopman(
        *train,
        integration_substeps=1,
        latent_dim=2,
        ridge=1e-10,
        normalize_activity=True,
        belief_conditioned=True,
    )
    common_model = fit_trajectory_koopman(
        *train,
        integration_substeps=1,
        latent_dim=2,
        ridge=1e-10,
        normalize_activity=True,
        belief_conditioned=False,
        shared_preprocessing=belief_model,
    )
    state_only_model = fit_trajectory_koopman(
        *train,
        integration_substeps=1,
        latent_dim=2,
        ridge=1e-10,
        normalize_activity=True,
        belief_conditioned=True,
        operator_mode="state_only",
        shared_preprocessing=belief_model,
    )
    belief_score = belief_model.score(*test)
    common_score = common_model.score(*test)
    persistence = persistence_trajectory_score(belief_model, *test)
    state_only_score = state_only_model.score(*test)

    assert belief_score.rollout_normalized_rmse < 1e-5
    assert belief_score.rollout_normalized_rmse < common_score.rollout_normalized_rmse
    assert belief_score.rollout_normalized_rmse < persistence.rollout_normalized_rmse
    assert state_only_score.rollout_normalized_rmse < 1e-5
    assert state_only_model.state_transition_delta_frobenius > 0.0
    assert state_only_model.exogenous_control_delta_frobenius == pytest.approx(0.0)
    assert belief_score.heldout_state_basis_residual_fraction < 1e-20
    assert belief_model.preprocessing_fit_scope.endswith(
        "training_sequences_substeps_only"
    )
    assert common_model.state_pca is belief_model.state_pca
    assert common_model.rate_pca is belief_model.rate_pca
    assert common_model.training_trajectory_fingerprint == (
        belief_model.training_trajectory_fingerprint
    )

    geometry = belief_manifold_geometry(
        belief_model,
        train[1],
        train[3],
        test[1],
        test[3],
        low_threshold=0.1,
        high_threshold=0.9,
    )
    assert geometry.eligible
    assert "not_proof_of_attractors" in geometry.interpretation


def test_randomized_pca_is_seeded_and_shared_receipt_fails_closed() -> None:
    train = _shared_controlled_system(seed=12, n_sequences=12, n_steps=8)
    first = fit_trajectory_koopman(
        *train,
        integration_substeps=1,
        latent_dim=2,
        pca_solver="randomized",
        pca_seed=17,
    )
    second = fit_trajectory_koopman(
        *train,
        integration_substeps=1,
        latent_dim=2,
        pca_solver="randomized",
        pca_seed=17,
    )
    np.testing.assert_array_equal(
        first.state_pca.components_, second.state_pca.components_
    )
    altered = list(train)
    altered[0] = np.array(train[0], copy=True)
    altered[0][0, 0, 0] += 1e-3
    with pytest.raises(ValueError, match="shared preprocessing"):
        fit_trajectory_koopman(
            *altered,
            integration_substeps=1,
            latent_dim=2,
            belief_conditioned=False,
            shared_preprocessing=first,
        )
    with pytest.raises(ValueError, match="pca_seed"):
        fit_trajectory_koopman(
            *train,
            integration_substeps=1,
            latent_dim=2,
            pca_solver="randomized",
            pca_seed=True,
        )


def test_belief_geometry_rejects_invalid_and_rank_deficient_groups() -> None:
    train = _shared_controlled_system(seed=14, n_sequences=12, n_steps=8)
    fitted = fit_trajectory_koopman(
        *train,
        integration_substeps=1,
        latent_dim=2,
    )
    with pytest.raises(ValueError, match="finite"):
        bad = np.array(train[3], copy=True)
        bad[0, 0] = np.nan
        belief_manifold_geometry(fitted, train[1], bad, train[1], train[3])
    with pytest.raises(ValueError, match="subspace_dim"):
        belief_manifold_geometry(
            fitted,
            train[1],
            train[3],
            train[1],
            train[3],
            subspace_dim=1.5,
        )
    zero_rates = np.zeros_like(train[1])
    degenerate = belief_manifold_geometry(
        fitted,
        zero_rates,
        train[3],
        zero_rates,
        train[3],
        low_threshold=0.1,
        high_threshold=0.9,
    )
    assert not degenerate.eligible
    assert "rank_deficient" in str(degenerate.ineligibility_reason)


def test_global_zero_contraction_is_not_belief_specific_attractor_support() -> None:
    network = EIRateNetwork(
        8,
        n_inputs=2,
        connection_probability=0.0,
        tau_e=20.0,
        tau_i=20.0,
        dt=5.0,
        input_scale=0.0,
        activation="tanh",
        seed=4,
    )
    vector = np.linspace(-0.5, 0.5, network.n_units)
    anchors = np.stack((vector, -vector, 2.0 * vector, -2.0 * vector))
    probe = fixed_drive_attractor_probe(
        network,
        anchors,
        np.linspace(-1.0, 1.0, network.n_units),
        gain_strength=0.5,
        raw_drive=(0.0, 0.0),
        horizon_steps=12,
        minimum_separation=1.0,
    )
    assert probe.both_conditions_contract
    assert not probe.separated_convergence
    assert probe.endpoint_centroid_separation < 1e-12
    assert probe.centroid_separation_over_initial_dispersion < 1e-12
