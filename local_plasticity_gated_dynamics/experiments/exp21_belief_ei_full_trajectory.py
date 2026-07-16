"""Audit belief-controlled E/I dynamics at every Euler substep.

Exp21 leaves Exp19 untouched.  Its primary receiver preserves Exp19's
every-trial zero-state reset while recording all 50 ms states.  A separately
labelled sensitivity carries state continuously within held-out episodes.
Both use the same frozen high-rank Dale E/I checkpoint, cue-only learned
belief gate, gain axis, readout, task split, and deterministic input tape.

The fitted models are controlled affine Koopman *predictors*, not
probabilistic LDS models.  A raw-input common/full comparison measures total
scalar-control prediction gain.  A routed-input common/state-plus-affine
comparison holds the pathway actuator in known exogenous inputs and tests
whether population belief improves latent state/affine prediction while input
and epoch coefficients remain shared.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp19_belief_ei_effective_dynamics as exp19
from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.trajectory_control_dynamics import (
    FittedTrajectoryKoopman,
    PerturbationEligibilityError,
    TrajectoryKoopmanScore,
    belief_manifold_geometry,
    fit_trajectory_koopman,
    fixed_drive_attractor_probe,
    nonlinear_perturbation_recovery,
    persistence_trajectory_score,
)
from src.models.belief_gain import balanced_gain_axis
from src.models.context_belief import MDRecurrentBeliefGate
from src.tasks.hidden_context import (
    generate_hidden_context,
    make_hidden_context_random_tape,
)
from src.training.hidden_context_ei import (
    ReceiverSimulation,
    evaluate_receiver_condition,
    fit_receiver_readout,
    simulate_receiver,
)
from src.training.hidden_context_gate import split_hidden_context_dataset
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


EXPERIMENT = "exp21_belief_ei_full_trajectory"
PROTOCOL_VERSION = "exp21_v2"
PERTURBATION_GEOMETRY = "joint_state_pca_physical_x_projection_v2"
PERTURBATION_REPORT_SCOPE = "physical_x_projection_finite_amplitude_recovery"
PERTURBATION_REFERENCE_FRACTION_POLICY = (
    "sampled_over_planned_eligible_sampled_over_sampled_eligible_reference_over_planned"
)
GATE_TIMING = exp19.GATE_TIMING
GATE_MODEL_NAME = exp19.GATE_MODEL_NAME
CONDITION = {
    "condition": "md_combined_intact_full_trajectory",
    "model_family": "frozen_high_rank_dale_ei",
    "controller_mode": "combined",
    "belief_intervention": "none",
    "trajectory_sampling": "euler_substep",
}


def _fingerprint(label: str, *values: object) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        else:
            digest.update(
                json.dumps(value, sort_keys=True, default=repr).encode("utf-8")
            )
        digest.update(b"\0")
    return digest.hexdigest()


def _protocol_id(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "config_path"}
    return _fingerprint("exp21-full-trajectory-protocol-v2", payload)


def _require_trajectory(
    simulation: ReceiverSimulation,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    values = (
        simulation.full_x_trajectory,
        simulation.full_rate_trajectory,
        simulation.receiver_raw_sensory_inputs,
        simulation.routed_inputs,
        simulation.trajectory_belief,
        simulation.trajectory_population_gain_belief,
        simulation.trajectory_epoch,
    )
    if any(value is None for value in values):
        raise RuntimeError("full trajectory recording contract is incomplete")
    return values  # type: ignore[return-value]


def _score_fields(
    prefix: str,
    score: TrajectoryKoopmanScore,
    *,
    trajectory_dt: float,
) -> dict[str, object]:
    per_horizon = np.asarray(
        score.rollout_per_horizon_normalized_rmse, dtype=np.float64
    )
    result: dict[str, object] = {
        f"{prefix}_one_step_normalized_mse": score.one_step_normalized_mse,
        f"{prefix}_one_step_raw_latent_mse": score.one_step_raw_latent_mse,
        f"{prefix}_rollout_normalized_rmse": score.rollout_normalized_rmse,
        f"{prefix}_rollout_endpoint_normalized_rmse": (
            score.rollout_endpoint_normalized_rmse
        ),
        f"{prefix}_rollout_auc_normalized_rmse": float(np.mean(per_horizon)),
        f"{prefix}_rollout_max_normalized_rmse": float(np.max(per_horizon)),
        f"{prefix}_rollout_per_horizon_normalized_rmse": per_horizon.tolist(),
        f"{prefix}_heldout_state_basis_residual_fraction": (
            score.heldout_state_basis_residual_fraction
        ),
        f"{prefix}_heldout_rate_basis_residual_fraction": (
            score.heldout_rate_basis_residual_fraction
        ),
        f"{prefix}_n_sequences": score.n_sequences,
        f"{prefix}_n_transitions": score.n_transitions,
        f"{prefix}_n_rollout_windows": score.n_rollout_windows,
        f"{prefix}_rollout_interpretation": score.interpretation,
    }
    for horizon_ms in (100.0, 250.0, 500.0, 1000.0):
        index = int(round(horizon_ms / trajectory_dt)) - 1
        result[f"{prefix}_rollout_rmse_{int(horizon_ms)}ms"] = (
            float(per_horizon[index]) if 0 <= index < per_horizon.size else None
        )
    return result


def _fit_dynamics_bundle(
    train: ReceiverSimulation,
    test: ReceiverSimulation,
    *,
    config: dict[str, Any],
    seed: int,
    prefix: str,
) -> tuple[dict[str, object], FittedTrajectoryKoopman]:
    (
        train_x,
        train_rates,
        train_raw,
        train_routed,
        train_control_belief,
        train_population_belief,
        train_epoch,
    ) = _require_trajectory(train)
    (
        test_x,
        test_rates,
        test_raw,
        test_routed,
        test_control_belief,
        test_population_belief,
        test_epoch,
    ) = _require_trajectory(test)
    options = dict(config["trajectory_dynamics"])
    substeps = int(config["integration_substeps"])
    scope = str(train.trajectory_sequence_scope)
    if scope != str(test.trajectory_sequence_scope):
        raise RuntimeError("train/test trajectory sequence scopes differ")
    pca_seed = derive_seed(seed, "exp21", prefix, "randomized-pca") % (2**32 - 1)
    common = dict(
        integration_substeps=substeps,
        latent_dim=int(options["latent_dim"]),
        ridge=float(options["ridge"]),
        normalize_activity=bool(options.get("normalize_activity", True)),
        sequence_scope=scope,
    )
    total_full = fit_trajectory_koopman(
        train_x,
        train_rates,
        train_raw,
        train_control_belief,
        train_epoch,
        belief_conditioned=True,
        operator_mode="full_shared_neutral_cue",
        pca_solver=str(options.get("pca_solver", "randomized")),
        pca_seed=pca_seed,
        **common,
    )
    raw_common = fit_trajectory_koopman(
        train_x,
        train_rates,
        train_raw,
        train_control_belief,
        train_epoch,
        belief_conditioned=False,
        operator_mode="common",
        shared_preprocessing=total_full,
        **common,
    )
    routed_common = fit_trajectory_koopman(
        train_x,
        train_rates,
        train_routed,
        train_population_belief,
        train_epoch,
        belief_conditioned=False,
        operator_mode="common",
        shared_preprocessing=total_full,
        **common,
    )
    routed_state = fit_trajectory_koopman(
        train_x,
        train_rates,
        train_routed,
        train_population_belief,
        train_epoch,
        belief_conditioned=True,
        operator_mode="state_only",
        shared_preprocessing=total_full,
        **common,
    )

    horizon = int(options["rollout_horizon_steps"])
    stride = int(options.get("rollout_stride_steps", horizon))
    score_options = {
        "sequence_scope": scope,
        "rollout_horizon_steps": horizon,
        "rollout_stride_steps": stride,
    }
    total_score = total_full.score(
        test_x,
        test_rates,
        test_raw,
        test_control_belief,
        test_epoch,
        **score_options,
    )
    raw_common_score = raw_common.score(
        test_x,
        test_rates,
        test_raw,
        test_control_belief,
        test_epoch,
        **score_options,
    )
    routed_common_score = routed_common.score(
        test_x,
        test_rates,
        test_routed,
        test_population_belief,
        test_epoch,
        **score_options,
    )
    routed_state_score = routed_state.score(
        test_x,
        test_rates,
        test_routed,
        test_population_belief,
        test_epoch,
        **score_options,
    )
    persistence_score = persistence_trajectory_score(
        total_full,
        test_x,
        test_rates,
        test_raw,
        test_control_belief,
        test_epoch,
        **score_options,
    )
    dt = float(test.trajectory_dt)
    metrics: dict[str, object] = {}
    for name, score in (
        ("total_full", total_score),
        ("raw_common", raw_common_score),
        ("routed_state_affine", routed_state_score),
        ("routed_common", routed_common_score),
        ("persistence", persistence_score),
    ):
        metrics.update(_score_fields(f"{prefix}_{name}", score, trajectory_dt=dt))
    metrics.update(
        {
            f"{prefix}_total_control_rollout_gain_vs_raw_common": (
                raw_common_score.rollout_normalized_rmse
                - total_score.rollout_normalized_rmse
            ),
            f"{prefix}_total_control_rollout_gain_vs_persistence": (
                persistence_score.rollout_normalized_rmse
                - total_score.rollout_normalized_rmse
            ),
            f"{prefix}_population_state_affine_rollout_gain_vs_routed_common": (
                routed_common_score.rollout_normalized_rmse
                - routed_state_score.rollout_normalized_rmse
            ),
            f"{prefix}_population_state_transition_delta_frobenius": (
                routed_state.state_transition_delta_frobenius
            ),
            f"{prefix}_population_affine_bias_delta_norm": (
                routed_state.affine_bias_delta_norm
            ),
            f"{prefix}_population_exogenous_control_delta_frobenius": (
                routed_state.exogenous_control_delta_frobenius
            ),
            f"{prefix}_state_affine_operator_design_rank": (
                routed_state.operator_design_rank
            ),
            f"{prefix}_state_affine_operator_design_columns": (
                routed_state.operator_design_columns
            ),
            f"{prefix}_state_affine_operator_design_full_rank": (
                routed_state.operator_design_rank
                == routed_state.operator_design_columns
            ),
            f"{prefix}_total_operator_design_rank": total_full.operator_design_rank,
            f"{prefix}_total_operator_design_columns": (
                total_full.operator_design_columns
            ),
            f"{prefix}_total_operator_design_full_rank": (
                total_full.operator_design_rank == total_full.operator_design_columns
            ),
            f"{prefix}_total_operator_unconstrained_columns": (
                total_full.operator_unconstrained_columns
            ),
            f"{prefix}_total_operator_identifiable_columns": (
                total_full.operator_design_columns
            ),
            f"{prefix}_total_operator_constraint": total_full.operator_constraint,
            f"{prefix}_total_operator_mode": total_full.operator_mode,
            f"{prefix}_total_operator_identifiability_fingerprint": _fingerprint(
                "exp21-total-operator-identifiability-v2",
                total_full.operator_mode,
                total_full.operator_unconstrained_columns,
                total_full.operator_design_columns,
                total_full.operator_constraint,
                total_full.training_trajectory_fingerprint,
            ),
            f"{prefix}_total_full_parameter_count": total_full.parameter_count,
            f"{prefix}_raw_common_parameter_count": raw_common.parameter_count,
            f"{prefix}_routed_common_parameter_count": (routed_common.parameter_count),
            f"{prefix}_routed_state_affine_parameter_count": (
                routed_state.parameter_count
            ),
            f"{prefix}_pca_solver": total_full.pca_solver,
            f"{prefix}_pca_seed": total_full.pca_seed,
            f"{prefix}_state_preprocessing_fingerprint": (
                total_full.state_preprocessing_fingerprint
            ),
            f"{prefix}_total_operator_train_fingerprint": (
                total_full.training_trajectory_fingerprint
            ),
            f"{prefix}_routed_state_affine_operator_train_fingerprint": (
                routed_state.training_trajectory_fingerprint
            ),
            f"{prefix}_paired_models_share_state_pca": bool(
                raw_common.state_pca is total_full.state_pca
                and routed_common.state_pca is total_full.state_pca
                and routed_state.state_pca is total_full.state_pca
            ),
            f"{prefix}_trajectory_sequence_scope": scope,
            f"{prefix}_train_trajectory_shape": list(train_x.shape),
            f"{prefix}_test_trajectory_shape": list(test_x.shape),
            f"{prefix}_trajectory_dt": dt,
            f"{prefix}_rollout_horizon_steps": min(horizon, test_x.shape[1] - 1),
            f"{prefix}_rollout_stride_steps": stride,
        }
    )
    return metrics, total_full


def _geometry_metrics(
    fitted: FittedTrajectoryKoopman,
    train: ReceiverSimulation,
    test: ReceiverSimulation,
    config: dict[str, Any],
    prefix: str,
) -> dict[str, object]:
    train_values = _require_trajectory(train)
    test_values = _require_trajectory(test)
    options = dict(config["belief_geometry"])
    summary = belief_manifold_geometry(
        fitted,
        train_values[1],
        train_values[4],
        test_values[1],
        test_values[4],
        low_threshold=float(options["low_threshold"]),
        high_threshold=float(options["high_threshold"]),
        subspace_dim=int(options["subspace_dim"]),
    )
    result = asdict(summary)
    return {f"{prefix}_belief_{key}": value for key, value in result.items()}


def _perturbation_metrics(
    fitted: FittedTrajectoryKoopman,
    network,
    train: ReceiverSimulation,
    test: ReceiverSimulation,
    gain_axis: np.ndarray,
    config: dict[str, Any],
    seed: int,
    prefix: str,
) -> dict[str, object]:
    test_values = _require_trajectory(test)
    train_x = _require_trajectory(train)[0]
    options = dict(config["perturbation"])
    geometry = str(options.get("geometry", ""))
    if geometry != PERTURBATION_GEOMETRY:
        raise ValueError(f"perturbation.geometry must equal {PERTURBATION_GEOMETRY!r}")
    train_scale = float(np.sqrt(np.mean(np.var(train_x[:, 1:], axis=(0, 1)))))
    fractions = tuple(float(item) for item in options["amplitude_fractions"])
    amplitudes = tuple(train_scale * item for item in fractions)
    try:
        summary = nonlinear_perturbation_recovery(
            network,
            fitted,
            test_values[0],
            test_values[1],
            test_values[3],
            test_values[5],
            test_values[6],
            gain_axis,
            gain_strength=float(config["gain_strength"]),
            integration_substeps=int(config["integration_substeps"]),
            horizon_steps=int(options["horizon_steps"]),
            amplitudes=amplitudes,
            n_references=int(options["n_references"]),
            seed=derive_seed(seed, "exp21", prefix, "nonlinear-perturbation"),
            baseline_replay_tolerance=float(
                options.get("baseline_replay_tolerance", 1e-10)
            ),
        )
        result = {
            f"{prefix}_perturbation_status": "complete",
            f"{prefix}_perturbation_error": None,
            f"{prefix}_perturbation_train_x_scale": train_scale,
            f"{prefix}_perturbation_amplitude_fractions": list(fractions),
            f"{prefix}_perturbation_geometry": geometry,
            f"{prefix}_perturbation_report_scope": PERTURBATION_REPORT_SCOPE,
            f"{prefix}_perturbation_reference_fraction_policy": (
                PERTURBATION_REFERENCE_FRACTION_POLICY
            ),
            **{
                f"{prefix}_perturbation_{key}": value
                for key, value in asdict(summary).items()
            },
        }
    except PerturbationEligibilityError as error:
        result = {
            f"{prefix}_perturbation_status": "ineligible",
            f"{prefix}_perturbation_error": (f"{type(error).__name__}: {error}"),
            f"{prefix}_perturbation_train_x_scale": train_scale,
            f"{prefix}_perturbation_amplitude_fractions": list(fractions),
            f"{prefix}_perturbation_geometry": geometry,
            f"{prefix}_perturbation_report_scope": PERTURBATION_REPORT_SCOPE,
            f"{prefix}_perturbation_reference_fraction_policy": (
                PERTURBATION_REFERENCE_FRACTION_POLICY
            ),
        }
    return result


def _attractor_metrics(
    network,
    train: ReceiverSimulation,
    gain_axis: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, object]:
    train_x = _require_trajectory(train)[0]
    options = dict(config["attractor_probe"])
    flattened = train_x[:, 1:].reshape(-1, network.n_units)
    count = min(int(options["n_anchors"]), flattened.shape[0])
    if count < 2:
        raise ValueError("attractor probe requires at least two training anchors")
    rng = np.random.default_rng(derive_seed(seed, "exp21", "attractor-anchors"))
    indices = np.sort(rng.choice(flattened.shape[0], size=count, replace=False))
    summary = fixed_drive_attractor_probe(
        network,
        flattened[indices],
        gain_axis,
        gain_strength=float(config["gain_strength"]),
        beliefs=tuple(float(item) for item in options["beliefs"]),
        raw_drive=tuple(float(item) for item in options["raw_drive"]),
        horizon_steps=int(options["horizon_steps"]),
        minimum_separation=float(options["minimum_separation"]),
        minimum_initial_scaled_separation=float(
            options.get("minimum_initial_scaled_separation", 0.05)
        ),
        population_gain=True,
        pathway_gating=True,
    )
    return {
        "attractor_anchor_fit_scope": "training_trajectory_only",
        "attractor_anchor_indices_fingerprint": _fingerprint(
            "exp21-attractor-anchor-indices-v2", indices
        ),
        **{f"attractor_{key}": value for key, value in asdict(summary).items()},
    }


def _simulate(
    network,
    dataset,
    posterior: np.ndarray,
    gain_axis: np.ndarray,
    config: dict[str, Any],
    *,
    continuous_episodes: bool,
) -> ReceiverSimulation:
    return simulate_receiver(
        network,
        dataset,
        posterior,
        gain_axis,
        gain_strength=float(config["gain_strength"]),
        integration_substeps=int(config["integration_substeps"]),
        trial_batch_size=int(config["trial_batch_size"]),
        pathway_gating=True,
        population_gain=True,
        record_substeps=True,
        continuous_episodes=continuous_episodes,
    )


def run_seed(config: dict[str, Any], seed: int, results_root: str | Path) -> Path:
    """Run one seed while retaining a failure row for the registered cell."""

    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": (
            "md_filtered_belief_full_substep_controlled_affine_koopman_audit_v2"
        ),
        "experiment_protocol_version": PROTOCOL_VERSION,
        "used_autograd": False,
        "used_bptt": False,
        "recurrent_learning": False,
        "full_trajectory_model": True,
        "full_trajectory_lds": False,
    }
    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
    ) as run:
        run.register_conditions([CONDITION])
        try:
            if config.get("gate_timing") != GATE_TIMING:
                raise ValueError(f"gate_timing must equal {GATE_TIMING!r}")
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
            initial_weights = network.recurrent_weights
            network_id = _fingerprint(
                "exp21-frozen-dale-ei-network-v2",
                initial_weights,
                network.input_weights,
                network.excitatory_mask,
                network.time_constants,
                network.dt,
                network.activation_name,
            )
            gain_axis = balanced_gain_axis(
                network.excitatory_mask,
                seed=derive_seed(seed, "exp19", "gain-axis"),
            )
            gain_axis_id = _fingerprint("exp21-exp19-matched-gain-axis-v2", gain_axis)
            gate = MDRecurrentBeliefGate(
                seed=derive_seed(seed, "exp19", "md-gate"),
                **dict(config["md_gate"]),
            ).fit(splits.train.gate)
            train_prediction = gate.predict(splits.train.gate)
            test_prediction = gate.predict(splits.test.gate)
            if (
                train_prediction.test_accessed_true_context
                or test_prediction.test_accessed_true_context
            ):
                raise RuntimeError("held-out belief gate accessed hidden context")
            gate_checkpoint_id = _fingerprint(
                "exp21-exp19-matched-gate-v2",
                train_prediction.parameters,
                train_prediction.fit_trial_ids,
                train_prediction.fit_episode_ids,
            )

            train_reset = _simulate(
                network,
                splits.train,
                train_prediction.context_probability,
                gain_axis,
                config,
                continuous_episodes=False,
            )
            test_reset = _simulate(
                network,
                splits.test,
                test_prediction.context_probability,
                gain_axis,
                config,
                continuous_episodes=False,
            )
            readout = fit_receiver_readout(
                train_reset,
                splits.train.task,
                alpha=float(config["readout_alpha"]),
            )
            primary_behavior = evaluate_receiver_condition(
                network=network,
                simulation=test_reset,
                readout=readout,
                prediction=test_prediction,
                dataset=splits.test,
                gate_model=GATE_MODEL_NAME,
                intervention="none",
                gate_checkpoint_id=gate_checkpoint_id,
                gain_axis_id=gain_axis_id,
                split_id=splits.fingerprint,
                network_init_id=network_id,
                gate_operations_per_trial=float(config["gate_compute"]["operations"]),
                gate_state_updates_per_trial=float(config["gate_compute"]["states"]),
            )
            metrics, primary_fitted = _fit_dynamics_bundle(
                train_reset,
                test_reset,
                config=config,
                seed=seed,
                prefix="trial_reset",
            )
            metrics.update(
                _geometry_metrics(
                    primary_fitted,
                    train_reset,
                    test_reset,
                    config,
                    "trial_reset",
                )
            )
            metrics.update(
                _perturbation_metrics(
                    primary_fitted,
                    network,
                    train_reset,
                    test_reset,
                    gain_axis,
                    config,
                    seed,
                    "trial_reset",
                )
            )
            metrics.update(
                _attractor_metrics(network, train_reset, gain_axis, config, seed)
            )
            primary_train_trajectory_id = train_reset.trajectory_fingerprint
            primary_test_trajectory_id = test_reset.trajectory_fingerprint
            primary_train_shape = list(_require_trajectory(train_reset)[0].shape)
            primary_test_shape = list(_require_trajectory(test_reset)[0].shape)

            del primary_fitted
            gc.collect()

            train_episode = _simulate(
                network,
                splits.train,
                train_prediction.context_probability,
                gain_axis,
                config,
                continuous_episodes=True,
            )
            test_episode = _simulate(
                network,
                splits.test,
                test_prediction.context_probability,
                gain_axis,
                config,
                continuous_episodes=True,
            )
            episode_behavior = evaluate_receiver_condition(
                network=network,
                simulation=test_episode,
                readout=readout,
                prediction=test_prediction,
                dataset=splits.test,
                gate_model=GATE_MODEL_NAME,
                intervention="none",
                gate_checkpoint_id=gate_checkpoint_id,
                gain_axis_id=gain_axis_id,
                split_id=splits.fingerprint,
                network_init_id=network_id,
                gate_operations_per_trial=float(config["gate_compute"]["operations"]),
                gate_state_updates_per_trial=float(config["gate_compute"]["states"]),
            )
            episode_metrics, episode_fitted = _fit_dynamics_bundle(
                train_episode,
                test_episode,
                config=config,
                seed=seed,
                prefix="episode_continuous",
            )
            metrics.update(episode_metrics)
            metrics.update(
                _geometry_metrics(
                    episode_fitted,
                    train_episode,
                    test_episode,
                    config,
                    "episode_continuous",
                )
            )
            metrics.update(
                _perturbation_metrics(
                    episode_fitted,
                    network,
                    train_episode,
                    test_episode,
                    gain_axis,
                    config,
                    seed,
                    "episode_continuous",
                )
            )
            if not np.array_equal(network.recurrent_weights, initial_weights):
                raise RuntimeError("frozen recurrent checkpoint changed")

            metrics.update(
                {
                    **primary_behavior,
                    "profile": str(config.get("profile", "unspecified")),
                    "experiment_protocol_version": PROTOCOL_VERSION,
                    "training_algorithm": run_config["training_algorithm"],
                    "used_autograd": False,
                    "used_bptt": False,
                    "recurrent_learning": False,
                    "full_trajectory_model": True,
                    "full_trajectory_lds": False,
                    "trajectory_model_family": (
                        "controlled_affine_koopman_predictor_not_probabilistic_lds"
                    ),
                    "primary_receiver_state_reset_scope": "every_trial_zero_state",
                    "sensitivity_receiver_state_reset_scope": "episode_start_only",
                    "primary_exp19_physical_semantics_preserved": True,
                    "episode_continuous_changes_receiver_state_policy": True,
                    "known_future_exogenous_controls_in_rollout": True,
                    "autonomous_rollout": False,
                    "preprocessing_fit_train_only": True,
                    "operator_fit_train_only": True,
                    "perturbation_basis_fit_train_only": True,
                    "heldout_neural_states_used_for_operator_fit": False,
                    "statistics_unit": "seed",
                    "split_unit": "episode",
                    "gate_fit_accessed_true_context": False,
                    "gate_test_accessed_true_context": False,
                    "true_context_access_scope": "evaluation_only",
                    "episode_continuous_behavior_accuracy": episode_behavior[
                        "behavior_accuracy"
                    ],
                    "episode_continuous_behavior_balanced_accuracy": (
                        episode_behavior["behavior_balanced_accuracy"]
                    ),
                    "episode_continuous_activity_participation_ratio": (
                        episode_behavior["activity_participation_ratio"]
                    ),
                    "trial_reset_train_trajectory_fingerprint": (
                        primary_train_trajectory_id
                    ),
                    "trial_reset_test_trajectory_fingerprint": (
                        primary_test_trajectory_id
                    ),
                    "trial_reset_train_trajectory_shape": primary_train_shape,
                    "trial_reset_test_trajectory_shape": primary_test_shape,
                    "episode_continuous_train_trajectory_fingerprint": (
                        train_episode.trajectory_fingerprint
                    ),
                    "episode_continuous_test_trajectory_fingerprint": (
                        test_episode.trajectory_fingerprint
                    ),
                    "trajectory_storage_scope": (
                        "in_memory_analysis_only_hash_and_shape_published"
                    ),
                    "network_init_id": network_id,
                    "gain_axis_id": gain_axis_id,
                    "gate_checkpoint_id": gate_checkpoint_id,
                    "readout_checkpoint_id": readout.checkpoint_id,
                    "split_id": splits.fingerprint,
                    "random_tape_id": tape.fingerprint,
                    "shared_noise_id": tape.fingerprint,
                    "receiver_noise_policy": "none_deterministic",
                    "experiment_protocol_id": _protocol_id(config),
                    "planned_condition_count": 1,
                    "planned_condition_grid_id": _fingerprint(
                        "exp21-planned-grid-v2", CONDITION
                    ),
                    "receiver_raw_sensory_input_policy": (
                        "two_sensory_channels_without_cue_channels"
                    ),
                    "total_control_model_input_policy": (
                        "raw_receiver_sensory_plus_scalar_control_interactions"
                    ),
                    "population_state_affine_model_input_policy": (
                        "already_routed_sensory_plus_population_gain_belief_"
                        "state_and_affine_bias_switch_input_and_epoch_shared"
                    ),
                    "latent_dim_selection": "registered_fixed_not_data_selected",
                    "nested_cv_latent_selection_used": False,
                    "readout_reused_across_state_policies": True,
                    "network_reused_across_state_policies": True,
                    "gate_reused_across_state_policies": True,
                    "gain_axis_reused_across_state_policies": True,
                }
            )
            run.record(metrics, **CONDITION)
        except Exception as error:
            run.mark_condition_failure(error, **CONDITION)
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "Full-substep belief E/I trajectory audit",
        "configs/formal/exp21_belief_ei_full_trajectory.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        path = run_seed(config, seed, args.results_root)
        print(path)


if __name__ == "__main__":
    main()
