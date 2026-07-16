"""Contracts for continuous receiver trajectories and train-only dynamics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import exp19_belief_ei_effective_dynamics as exp19
from src.analysis.trajectory_control_dynamics import (
    belief_manifold_geometry,
    fit_trajectory_koopman,
    fixed_drive_attractor_probe,
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
