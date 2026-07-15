"""Audit cue-filtered belief control in a high-rank frozen Dale E/I receiver.

This experiment is deliberately narrower than a full trajectory LDS claim.
One MD-like gate is fit from training cues only.  Because the synthetic task
has an explicit cue epoch before its sensory epoch, the gate consumes the
current cue and freezes ``p(z_t | cue_<=t)`` before controlling sensory,
delay, and response processing.  A single intact training simulation fits
both the task readout and a three-epoch reduced-dynamics surrogate.  Every
held-out intervention and mode ablation reuses those two frozen checkpoints.

The direct evidence-mixing row is an explicitly separate analytic baseline;
it is not presented as an E/I receiver or as evidence for reduced neural
dynamics.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.effective_control_dynamics import (
    FittedSoftEpochDynamics,
    ei_continuous_jacobian,
    ei_scalar_gain_jacobian_tangent,
    fit_soft_epoch_dynamics,
    projected_normal_linear_summary,
    pullback_rate_tangent_basis,
)
from src.analysis.gate_metrics import context_calibration_summary
from src.analysis.rank_stage_metrics import matrix_rank_summary
from src.models.belief_gain import balanced_gain_axis
from src.models.context_belief import (
    GatePrediction,
    MDRecurrentBeliefGate,
    deranged_trajectory_shuffle,
    episode_delay,
    neutral_clamp,
)
from src.models.ei_rate_network import EIRateNetwork
from src.tasks.hidden_context import (
    HiddenContextConfig,
    HiddenContextDataset,
    TaskLearningBatch,
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


APPROXIMATION_SCOPE = (
    "three_epoch_mean_rate_soft_operator_surrogate_not_full_trajectory_lds"
)
GATE_TIMING = "current_cue_filtered_after_cue_epoch_before_sensory"
GATE_MODEL_NAME = "md_recurrent_belief_filtered_posterior"


@dataclass(frozen=True, slots=True)
class _ConditionSpec:
    condition: str
    model_family: str
    controller_mode: str
    belief_intervention: str

    @property
    def uses_ei_receiver(self) -> bool:
        return self.model_family == "frozen_high_rank_dale_ei"

    @property
    def population_gain(self) -> bool:
        return self.controller_mode in {"combined", "population_only"}

    @property
    def pathway_gating(self) -> bool:
        return self.controller_mode in {"combined", "pathway_only"}


EI_CONDITION_SPECS = (
    _ConditionSpec(
        "md_combined_intact", "frozen_high_rank_dale_ei", "combined", "none"
    ),
    _ConditionSpec(
        "md_combined_clamp", "frozen_high_rank_dale_ei", "combined", "clamp"
    ),
    _ConditionSpec(
        "md_combined_delay", "frozen_high_rank_dale_ei", "combined", "delay"
    ),
    _ConditionSpec(
        "md_combined_shuffle", "frozen_high_rank_dale_ei", "combined", "shuffle"
    ),
    _ConditionSpec(
        "md_population_only",
        "frozen_high_rank_dale_ei",
        "population_only",
        "none",
    ),
    _ConditionSpec(
        "md_pathway_only",
        "frozen_high_rank_dale_ei",
        "pathway_only",
        "none",
    ),
    _ConditionSpec(
        "md_disconnected", "frozen_high_rank_dale_ei", "disconnected", "none"
    ),
)
DIRECT_BASELINE_SPEC = _ConditionSpec(
    "direct_evidence_mix",
    "direct_evidence_mix_ridge_baseline",
    "analytic_evidence_mix",
    "none",
)
ALL_CONDITION_SPECS = (*EI_CONDITION_SPECS, DIRECT_BASELINE_SPEC)


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
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _protocol_id(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "config_path"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(b"exp19-belief-ei-effective-v1\0" + encoded).hexdigest()


def _planned_conditions(config: dict[str, Any]) -> list[dict[str, object]]:
    delay = int(config["interventions"]["delay_trials"])
    if delay < 1:
        raise ValueError("interventions.delay_trials must be positive")
    result: list[dict[str, object]] = []
    for spec in ALL_CONDITION_SPECS:
        intervention = (
            f"delay_{delay}"
            if spec.belief_intervention == "delay"
            else spec.belief_intervention
        )
        result.append(
            {
                "condition": spec.condition,
                "model_family": spec.model_family,
                "controller_mode": spec.controller_mode,
                "belief_intervention": intervention,
                "gate_model": GATE_MODEL_NAME,
            }
        )
    if len(result) != len({str(item["condition"]) for item in result}):
        raise RuntimeError("Exp19 condition names must be unique")
    return result


def _task_config(config: dict[str, Any]) -> HiddenContextConfig:
    return HiddenContextConfig(
        **dict(config["task"]),
        cue_reliability=float(config["cue_reliability"]),
        context_hazard=float(config["context_hazard"]),
    )


def _network(
    config: dict[str, Any], task: HiddenContextConfig, seed: int
) -> EIRateNetwork:
    options = dict(config["network"])
    substeps = int(config["integration_substeps"])
    if substeps < 1:
        raise ValueError("integration_substeps must be positive")
    expected_dt = float(task.dt_ms) / substeps
    configured_dt = float(options.get("dt", expected_dt))
    if not np.isclose(configured_dt, expected_dt, atol=0.0, rtol=1e-12):
        raise ValueError(
            "network.dt * integration_substeps must equal the task time step"
        )
    options["dt"] = configured_dt
    return EIRateNetwork(
        n_inputs=2,
        seed=derive_seed(seed, "exp19", "network-init"),
        **options,
    )


def _task_targets(task: TaskLearningBatch) -> np.ndarray:
    response = np.flatnonzero(np.asarray(task.epoch) == "response")
    values = np.asarray(task.targets[:, response, 0], dtype=np.float64)
    if response.size == 0 or not np.all(task.loss_mask[:, response]):
        raise ValueError("the response epoch must be fully supervised")
    if not np.all(values == values[:, :1]):
        raise ValueError("response target must be constant within a trial")
    result = values[:, 0]
    if not np.isin(result, [-1.0, 1.0]).all():
        raise ValueError("Exp19 requires binary -1/+1 task targets")
    return result


def _balanced_accuracy(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.mean(predicted[target == label] == label)
                for label in np.unique(target)
            ]
        )
    )


def _dynamics_checkpoint_id(
    fitted: FittedSoftEpochDynamics, train_data_id: str
) -> str:
    return _fingerprint(
        "exp19-train-only-soft-epoch-dynamics-v1",
        train_data_id,
        fitted.pca.mean_,
        fitted.pca.scale_,
        fitted.pca.components_,
        fitted.operator_state0,
        fitted.operator_state1,
        fitted.closure_variance,
        fitted.ridge,
    )


def _prediction_for_spec(
    spec: _ConditionSpec,
    intact: GatePrediction,
    *,
    delay_trials: int,
    seed: int,
) -> GatePrediction:
    if spec.belief_intervention == "none":
        return intact
    if spec.belief_intervention == "clamp":
        return neutral_clamp(intact)
    if spec.belief_intervention == "delay":
        return episode_delay(intact, delay_trials)
    if spec.belief_intervention == "shuffle":
        return deranged_trajectory_shuffle(
            intact,
            seed=derive_seed(seed, "exp19", "test-trajectory-shuffle"),
        )
    raise ValueError(f"unknown belief intervention: {spec.belief_intervention}")


def _combined_control_trajectory_rank(simulation: ReceiverSimulation) -> int:
    blocks: list[np.ndarray] = []
    if simulation.pathway_gating:
        blocks.append(np.asarray(simulation.pathway_scales, dtype=np.float64))
    if simulation.population_gain:
        active = np.asarray(simulation.gain.active_time_mask, dtype=bool)
        blocks.append(np.mean(simulation.gain.gains[:, active], axis=1))
    if not blocks:
        return 0
    trajectory = np.concatenate(blocks, axis=1)
    centered = trajectory - np.mean(trajectory, axis=0, keepdims=True)
    return int(np.linalg.matrix_rank(centered, tol=1e-12))


def _geometry_metrics(
    *,
    network: EIRateNetwork,
    fitted: FittedSoftEpochDynamics,
    mean_state: np.ndarray,
    gain_axis: np.ndarray,
    gain_strength: float,
    population_gain: bool,
    normal_horizon: float,
    rate_derivative_tolerance: float = 1e-8,
    rate_pullback_residual_tolerance: float = 0.05,
) -> dict[str, object]:
    strength = float(gain_strength) if population_gain else 0.0
    derivative_tolerance = float(rate_derivative_tolerance)
    residual_tolerance = float(rate_pullback_residual_tolerance)
    if not np.isfinite(derivative_tolerance) or derivative_tolerance <= 0.0:
        raise ValueError("rate_derivative_tolerance must be positive and finite")
    if not np.isfinite(residual_tolerance) or residual_tolerance < 0.0:
        raise ValueError(
            "rate_pullback_residual_tolerance must be non-negative and finite"
        )
    tangent = ei_scalar_gain_jacobian_tangent(
        network,
        mean_state,
        gain_axis,
        gain_strength=strength,
        belief=0.5,
    )
    tangent_rank = matrix_rank_summary(tangent)
    jacobian_low = ei_continuous_jacobian(
        network,
        mean_state,
        gain_axis,
        gain_strength=strength,
        belief=0.0,
    )
    jacobian_high = ei_continuous_jacobian(
        network,
        mean_state,
        gain_axis,
        gain_strength=strength,
        belief=1.0,
    )
    delta_rank = matrix_rank_summary(jacobian_high - jacobian_low)
    result: dict[str, object] = {
        "jacobian_tangent_raw_rank": tangent_rank.numerical_rank,
        "jacobian_tangent_effective_rank": tangent_rank.effective_rank,
        "same_state_delta_jacobian_raw_rank": delta_rank.numerical_rank,
        "same_state_delta_jacobian_effective_rank": delta_rank.effective_rank,
        "local_state_tangent_dimension": 0,
        "rate_to_state_pullback_residual": None,
        "rate_to_state_derivative_active_count": 0,
        "normal_stability_eligible": False,
        "normal_stability_ineligibility_reason": None,
        "normal_dimension": None,
        "normal_local_decay_ratio": None,
        "normal_local_max_real_part": None,
        "normal_to_tangent_coupling_norm": None,
        "normal_horizon": float(normal_horizon),
        "jacobian_operating_state_scope": "intact_training_mean_state_only",
        "normal_basis_scope": (
            "local_pseudoinverse_preactivation_pullback_of_"
            "intact_train_epoch_rate_basis"
        ),
        "jacobian_scope": (
            "continuous_time_same_train_state_scalar_belief_gain_linearization"
        ),
    }
    try:
        pulled_back = pullback_rate_tangent_basis(
            network,
            mean_state,
            gain_axis,
            fitted.raw_activity_basis,
            gain_strength=strength,
            belief=0.5,
            derivative_tolerance=derivative_tolerance,
            residual_tolerance=residual_tolerance,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        state = np.asarray(mean_state, dtype=np.float64)
        gain = np.ones(network.n_units, dtype=np.float64)
        activated = np.tanh(gain * state)
        derivative = gain * (1.0 - activated * activated)
        if network.activation_name == "rectified_tanh":
            derivative = np.where(activated > 0.0, derivative, 0.0)
        result["rate_to_state_derivative_active_count"] = int(
            np.count_nonzero(np.abs(derivative) > derivative_tolerance)
        )
        result["normal_stability_ineligibility_reason"] = (
            f"rate_basis_pullback_ineligible:{type(error).__name__}:{error}"
        )
        return result
    result.update(
        local_state_tangent_dimension=pulled_back.tangent_dimension,
        rate_to_state_pullback_residual=(
            pulled_back.relative_rate_reconstruction_residual
        ),
        rate_to_state_derivative_active_count=pulled_back.derivative_active_count,
        normal_stability_eligible=pulled_back.eligible,
        normal_stability_ineligibility_reason=(
            None
            if pulled_back.eligible
            else "rate_basis_pullback_rank_or_residual_threshold_failed"
        ),
    )
    if not pulled_back.eligible:
        return result
    reference = ei_continuous_jacobian(
        network,
        mean_state,
        gain_axis,
        gain_strength=strength,
        belief=0.5,
    )
    try:
        normal = projected_normal_linear_summary(
            reference,
            pulled_back.preactivation_basis,
            horizon=float(normal_horizon),
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        result["normal_stability_eligible"] = False
        result["normal_stability_ineligibility_reason"] = (
            f"projected_normal_numerical_failure:{type(error).__name__}:{error}"
        )
        return result
    result.update(
        normal_dimension=normal.normal_dimension,
        normal_local_decay_ratio=normal.decay_ratio,
        normal_local_max_real_part=normal.max_real_part,
        normal_to_tangent_coupling_norm=normal.normal_to_tangent_coupling_norm,
        normal_horizon=normal.horizon,
    )
    return result


def _physical_metrics(network: EIRateNetwork) -> dict[str, object]:
    weights = network.recurrent_weights
    rank = matrix_rank_summary(weights)
    excitatory = network.excitatory_mask
    inhibitory = network.inhibitory_mask
    tolerance = 1e-12
    excitatory_violations = int(np.sum(weights[:, excitatory] < -tolerance))
    inhibitory_violations = int(np.sum(weights[:, inhibitory] > tolerance))
    return {
        "physical_raw_rank": rank.numerical_rank,
        "physical_effective_rank": rank.effective_rank,
        "physical_rank_threshold": rank.threshold,
        "physical_rank_fraction": rank.numerical_rank / network.n_units,
        "high_rank_physical_recurrent": rank.numerical_rank >= 0.9 * network.n_units,
        "physical_dale_valid": bool(network.validate_dale(atol=tolerance)),
        "dale_excitatory_violation_count": excitatory_violations,
        "dale_inhibitory_violation_count": inhibitory_violations,
        "dale_sign_convention": "outgoing_columns",
    }


def _shared_operator_metrics(fitted: FittedSoftEpochDynamics) -> dict[str, object]:
    delta = fitted.operator_state1 - fitted.operator_state0
    rank = matrix_rank_summary(delta.reshape(1, -1))
    return {
        "basis_dimension": fitted.latent_dim,
        "basis_explained_variance_fraction": float(
            np.sum(fitted.pca.explained_variance_ratio_)
        ),
        "operator_control_dimension": rank.numerical_rank,
        "operator_control_effective_dimension": rank.effective_rank,
        "operator_delta_frobenius_norm": float(np.linalg.norm(delta)),
        "operator_design_rank": fitted.operator_design_rank,
        "operator_design_columns": fitted.operator_design_columns,
        "n_train_dynamics_trials": fitted.n_train_trials,
    }


def _direct_evidence_feature(
    dataset: HiddenContextDataset, state1_probability: np.ndarray
) -> np.ndarray:
    probability = np.asarray(state1_probability, dtype=np.float64)
    n_trials = dataset.task.inputs.shape[0]
    if probability.shape != (n_trials,):
        raise ValueError("direct-mix beliefs must align with trials")
    sensory_mask = np.asarray(dataset.task.epoch) == "sensory"
    evidence = np.mean(dataset.task.inputs[:, sensory_mask, :2], axis=1)
    return (1.0 - probability) * evidence[:, 0] + probability * evidence[:, 1]


def _fit_direct_baseline(
    dataset: HiddenContextDataset,
    prediction: GatePrediction,
    *,
    alpha: float,
) -> tuple[float, float, float, float, str]:
    feature = _direct_evidence_feature(dataset, prediction.context_probability)
    target = _task_targets(dataset.task)
    mean = float(np.mean(feature))
    empirical_scale = float(np.std(feature))
    scale = empirical_scale if empirical_scale > 0.0 else 1.0
    standardized = ((feature - mean) / scale).reshape(-1, 1)
    model = Ridge(alpha=float(alpha), fit_intercept=True).fit(standardized, target)
    coefficient = float(np.asarray(model.coef_).reshape(-1)[0])
    intercept = float(model.intercept_)
    checkpoint = _fingerprint(
        "exp19-direct-evidence-mix-ridge-v1",
        dataset.task.fingerprint,
        prediction.fingerprint,
        mean,
        scale,
        coefficient,
        intercept,
        float(alpha),
    )
    return mean, scale, coefficient, intercept, checkpoint


def _evaluate_direct_baseline(
    *,
    train_fit: tuple[float, float, float, float, str],
    dataset: HiddenContextDataset,
    prediction: GatePrediction,
    gate_checkpoint_id: str,
    split_id: str,
    random_tape_id: str,
    protocol_id: str,
    config: dict[str, Any],
) -> dict[str, object]:
    mean, scale, coefficient, intercept, checkpoint = train_fit
    feature = _direct_evidence_feature(dataset, prediction.context_probability)
    scores = ((feature - mean) / scale) * coefficient + intercept
    predicted = np.where(scores >= 0.0, 1, -1)
    targets = _task_targets(dataset.task).astype(int)
    calibration = context_calibration_summary(
        prediction.context_probability,
        dataset.truth.hidden_states,
        n_bins=10,
        epsilon=1e-6,
    )
    gate_compute = dict(config["gate_compute"])
    analytic_operations = 4.0
    centered_belief = (
        prediction.context_probability - np.mean(prediction.context_probability)
    )[:, None]
    empirical_control_rank = int(np.linalg.matrix_rank(centered_belief, tol=1e-12))
    return {
        "status": "complete",
        "statistics_unit": "seed",
        "split_unit": "episode",
        "profile": str(config.get("profile", "unspecified")),
        "training_algorithm": "train_only_scalar_ridge_on_analytic_evidence_mix",
        "uses_ei_receiver": False,
        "direct_baseline_separate_from_ei": True,
        "behavior_accuracy": float(np.mean(predicted == targets)),
        "behavior_balanced_accuracy": _balanced_accuracy(predicted, targets),
        "context_nll": calibration.nll,
        "context_brier": calibration.brier,
        "context_ece": calibration.expected_calibration_error,
        "context_accuracy": calibration.accuracy,
        "activity_participation_ratio": None,
        "activity_metric_applicable": False,
        "heldout_normalized_closure_mse": None,
        "heldout_raw_latent_mse": None,
        "heldout_basis_residual_fraction": None,
        "coarse_dynamics_applicable": False,
        "approximation_scope": "not_applicable_direct_nonrecurrent_baseline",
        "physical_raw_rank": None,
        "physical_effective_rank": None,
        "physical_dale_valid": None,
        "declared_scalar_control_dimension": 1,
        "combined_effective_control_dimension": 1,
        "empirical_combined_control_trajectory_rank": empirical_control_rank,
        "population_gain_control_rank": 0,
        "pathway_control_rank": 0,
        "operator_control_dimension": None,
        "jacobian_tangent_raw_rank": None,
        "same_state_delta_jacobian_raw_rank": None,
        "normal_local_decay_ratio": None,
        "irrelevant_pathway_input_fraction": None,
        "gate_primitive_operations_per_trial": float(gate_compute["operations"]),
        "gate_state_updates_per_trial": float(gate_compute["states"]),
        "analytic_mix_operations_per_trial": analytic_operations,
        "end_to_end_compute_proxy_per_trial": float(
            gate_compute["operations"] + gate_compute["states"] + analytic_operations
        ),
        "energy_interpretation": (
            "declared_unitless_gate_plus_direct_mix_operations_not_atp"
        ),
        "gate_fit_accessed_true_context": False,
        "gate_test_accessed_true_context": False,
        "true_context_access_scope": "evaluation_only",
        "gate_timing": GATE_TIMING,
        "belief_timing": "filtered_posterior_after_current_cue",
        "current_cue_accessed_for_same_trial": True,
        "cue_available_before_receiver_control": True,
        "receiver_received_cue_channels": False,
        "receiver_control_epoch_scope": "not_applicable_direct_baseline",
        "preprocessing_fit_train_only": True,
        "readout_fit_train_only": True,
        "dynamics_fit_train_only": None,
        "dynamics_heldout_used_for_fit": None,
        "gate_checkpoint_id": gate_checkpoint_id,
        "belief_trajectory_id": prediction.fingerprint,
        "intact_belief_trajectory_id": prediction.fingerprint,
        "direct_baseline_checkpoint_id": checkpoint,
        "readout_checkpoint_id": None,
        "dynamics_checkpoint_id": None,
        "network_init_id": None,
        "split_id": split_id,
        "random_tape_id": random_tape_id,
        "shared_noise_id": random_tape_id,
        "receiver_noise_policy": "none_deterministic",
        "experiment_protocol_id": protocol_id,
        "test_trial_count": int(targets.size),
        "full_trajectory_lds": False,
    }


def run_seed(config: dict[str, Any], seed: int, results_root: str | Path) -> Path:
    """Run one paired seed and preserve every planned success or failure cell."""

    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "md_filtered_belief_frozen_high_rank_dale_ei",
        "used_autograd": False,
        "used_bptt": False,
        "parent_checkpoint": None,
        "recurrent_learning": False,
        "full_trajectory_lds": False,
    }
    with ExperimentRun(
        "exp19_belief_ei_effective_dynamics",
        seed,
        run_config,
        results_root=results_root,
    ) as run:
        try:
            planned = _planned_conditions(config)
        except Exception as error:
            planned = [{"condition": "setup"}]
            run.register_conditions(planned)
            run.mark_condition_failure(error, **planned[0])
            return run.path
        run.register_conditions(planned)
        planned_grid_id = _fingerprint(
            "exp19-exact-planned-grid-v1",
            tuple(
                tuple(sorted((str(key), repr(value)) for key, value in item.items()))
                for item in planned
            ),
        )
        dimensions = {str(item["condition"]): item for item in planned}
        emitted: set[str] = set()

        def emit_failure(name: str, error: BaseException) -> None:
            if name in emitted:
                raise RuntimeError(f"condition emitted twice: {name}")
            run.mark_condition_failure(error, **dimensions[name])
            emitted.add(name)

        def emit_success(name: str, metrics: dict[str, object]) -> None:
            if name in emitted:
                raise RuntimeError(f"condition emitted twice: {name}")
            run.record(metrics, **dimensions[name])
            emitted.add(name)

        try:
            if config.get("gate_timing") != GATE_TIMING:
                raise ValueError(
                    f"Exp19 gate_timing must equal {GATE_TIMING!r}"
                )
            task = _task_config(config)
            tape = make_hidden_context_random_tape(task, seed=seed)
            dataset = generate_hidden_context(task, seed=seed, random_tape=tape)
            splits = split_hidden_context_dataset(
                dataset,
                outer_test_fraction=float(config["outer_test_fraction"]),
                validation_fraction=float(config["validation_fraction"]),
                seed=seed,
            )
            network = _network(config, task, seed)
            initial_weights = network.recurrent_weights
            network_id = _fingerprint(
                "exp19-frozen-dale-ei-network-v1",
                initial_weights,
                network.input_weights,
                network.excitatory_mask,
            )
            gain_axis = balanced_gain_axis(
                network.excitatory_mask,
                seed=derive_seed(seed, "exp19", "gain-axis"),
            )
            gain_axis_id = _fingerprint(
                "exp19-balanced-ei-gain-axis-v1", gain_axis
            )
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
                raise RuntimeError("cue-filtered gate accessed hidden context")
            gate_checkpoint_id = _fingerprint(
                "exp19-md-current-cue-filtered-gate-v1",
                train_prediction.parameters,
                train_prediction.fit_trial_ids,
                train_prediction.fit_episode_ids,
                splits.train.gate.fingerprint,
            )
            train_simulation = simulate_receiver(
                network,
                splits.train,
                train_prediction.context_probability,
                gain_axis,
                gain_strength=float(config["gain_strength"]),
                integration_substeps=int(config["integration_substeps"]),
                trial_batch_size=int(config["trial_batch_size"]),
                pathway_gating=True,
                population_gain=True,
            )
            readout = fit_receiver_readout(
                train_simulation,
                splits.train.task,
                alpha=float(config["readout_alpha"]),
            )
            dynamics_options = dict(config["effective_dynamics"])
            fitted_dynamics = fit_soft_epoch_dynamics(
                train_simulation.features,
                train_prediction.context_probability,
                n_units=network.n_units,
                latent_dim=int(dynamics_options["latent_dim"]),
                ridge=float(dynamics_options["ridge"]),
                normalize_activity=bool(
                    dynamics_options.get("normalize_activity", False)
                ),
            )
            dynamics_checkpoint_id = _dynamics_checkpoint_id(
                fitted_dynamics, readout.train_data_id
            )
            protocol_id = _protocol_id(config)
            physical_metrics = _physical_metrics(network)
            operator_metrics = _shared_operator_metrics(fitted_dynamics)
            geometry_by_population_mode = {
                enabled: _geometry_metrics(
                    network=network,
                    fitted=fitted_dynamics,
                    mean_state=train_simulation.mean_x,
                    gain_axis=gain_axis,
                    gain_strength=float(config["gain_strength"]),
                    population_gain=enabled,
                    normal_horizon=float(dynamics_options["normal_horizon_ms"]),
                    rate_derivative_tolerance=float(
                        dynamics_options.get("rate_derivative_tolerance", 1e-8)
                    ),
                    rate_pullback_residual_tolerance=float(
                        dynamics_options.get(
                            "rate_pullback_residual_tolerance", 0.05
                        )
                    ),
                )
                for enabled in (False, True)
            }
            direct_fit = _fit_direct_baseline(
                splits.train,
                train_prediction,
                alpha=float(config["linear_mix_alpha"]),
            )
            gate_fit_metadata = gate.audit_metadata()
            gate_provenance_metrics = {
                "estimated_context_hazard": float(
                    gate_fit_metadata["estimated_context_hazard"]
                ),
                "estimated_cue_reliability": float(
                    gate_fit_metadata["estimated_cue_reliability"]
                ),
                "moment_anchor_identifiable": bool(
                    gate_fit_metadata["moment_anchor_identifiable"]
                ),
                "cue_signal_z_score": float(
                    gate_fit_metadata["cue_signal_z_score"]
                ),
                "local_gate_update_l1": float(
                    gate_fit_metadata["local_update_l1"]
                ),
                "gate_fit_trial_count": len(gate_fit_metadata["fit_trial_ids"]),
                "gate_fit_episode_count": len(
                    gate_fit_metadata["fit_episode_ids"]
                ),
                "gate_fit_observation_fingerprint": str(
                    gate_fit_metadata["fit_observation_fingerprint"]
                ),
            }
            episode_starts = np.asarray(splits.test.gate.episode_start, dtype=bool)
            start_deviation = float(
                np.max(
                    np.abs(
                        test_prediction.context_probability[episode_starts] - 0.5
                    )
                )
            )
        except Exception as error:
            for spec in ALL_CONDITION_SPECS:
                emit_failure(spec.condition, error)
            return run.path

        for spec in EI_CONDITION_SPECS:
            try:
                prediction = _prediction_for_spec(
                    spec,
                    test_prediction,
                    delay_trials=int(config["interventions"]["delay_trials"]),
                    seed=seed,
                )
                simulation = simulate_receiver(
                    network,
                    splits.test,
                    prediction.context_probability,
                    gain_axis,
                    gain_strength=float(config["gain_strength"]),
                    integration_substeps=int(config["integration_substeps"]),
                    trial_batch_size=int(config["trial_batch_size"]),
                    pathway_gating=spec.pathway_gating,
                    population_gain=spec.population_gain,
                )
                if not np.array_equal(network.recurrent_weights, initial_weights):
                    raise RuntimeError("frozen E/I recurrent weights changed")
                closure_probability = (
                    prediction.context_probability
                    if spec.controller_mode != "disconnected"
                    else np.full(prediction.trial_ids.size, 0.5)
                )
                closure = fitted_dynamics.score(
                    simulation.features, closure_probability
                )
                evaluator_intervention = (
                    spec.belief_intervention
                    if spec.belief_intervention != "none"
                    else "none"
                )
                metrics = evaluate_receiver_condition(
                    network=network,
                    simulation=simulation,
                    readout=readout,
                    prediction=prediction,
                    dataset=splits.test,
                    gate_model=GATE_MODEL_NAME,
                    intervention=evaluator_intervention,
                    gate_checkpoint_id=gate_checkpoint_id,
                    gain_axis_id=gain_axis_id,
                    split_id=splits.fingerprint,
                    network_init_id=network_id,
                    gate_operations_per_trial=float(
                        config["gate_compute"]["operations"]
                    ),
                    gate_state_updates_per_trial=float(
                        config["gate_compute"]["states"]
                    ),
                )
                controller_connected = spec.pathway_gating or spec.population_gain
                metrics.update(
                    profile=str(config.get("profile", "unspecified")),
                    training_algorithm=(
                        "md_filtered_belief_frozen_high_rank_dale_ei"
                    ),
                    uses_ei_receiver=True,
                    bridge_scope=(
                        "cue_filtered_belief_to_frozen_high_rank_dale_ei_"
                        "coarse_effective_dynamics"
                    ),
                    recurrent_learning=False,
                    gate_fit_supervision="none",
                    gate_received_true_q_h=False,
                    gate_timing=GATE_TIMING,
                    belief_timing="filtered_posterior_after_current_cue",
                    current_cue_accessed_for_same_trial=True,
                    cue_available_before_receiver_control=True,
                    receiver_received_cue_channels=False,
                    receiver_control_epoch_scope=(
                        "population_gain_sensory_delay_response;"
                        "pathway_input_nonzero_only_during_sensory"
                    ),
                    test_episode_start_abs_belief_from_half=start_deviation,
                    physical_recurrent_rank_claim=(
                        "high_rank_background_not_low_rank_physical_connectivity"
                    ),
                    declared_scalar_control_dimension=(
                        1 if controller_connected else 0
                    ),
                    combined_effective_control_dimension=(
                        1 if controller_connected else 0
                    ),
                    empirical_combined_control_trajectory_rank=(
                        _combined_control_trajectory_rank(simulation)
                    ),
                    controller_actuator_count=(
                        int(spec.pathway_gating) + int(spec.population_gain)
                    ),
                    heldout_normalized_closure_mse=(
                        closure.normalized_closure_mse
                    ),
                    heldout_raw_latent_mse=closure.raw_latent_mse,
                    heldout_basis_residual_fraction=(
                        closure.heldout_basis_residual_fraction
                    ),
                    n_test_dynamics_trials=closure.n_trials,
                    n_test_dynamics_transitions=closure.n_transitions,
                    approximation_scope=APPROXIMATION_SCOPE,
                    full_trajectory_lds=False,
                    coarse_dynamics_applicable=True,
                    closure_belief_input_policy=(
                        "neutral_no_control"
                        if spec.controller_mode == "disconnected"
                        else "applied_filtered_or_intervened_belief"
                    ),
                    preprocessing_fit_train_only=True,
                    readout_fit_train_only=True,
                    dynamics_fit_train_only=True,
                    dynamics_heldout_used_for_fit=False,
                    dynamics_reused_from_intact_train=True,
                    readout_reused_from_intact_train=True,
                    base_conditions_share_readout=True,
                    base_comparison_scope=(
                        "single_intact_train_fitted_checkpoint_fixed_test_ablation"
                    ),
                    intervention_postfit=(spec.condition != "md_combined_intact"),
                    intervention_reuses_intact_readout=True,
                    intervention_reuses_intact_receiver=True,
                    intervention_reuses_intact_gate_checkpoint=True,
                    intervention_reuses_intact_dynamics=True,
                    dynamics_checkpoint_id=dynamics_checkpoint_id,
                    random_tape_id=tape.fingerprint,
                    shared_noise_id=tape.fingerprint,
                    receiver_noise_policy="none_deterministic",
                    experiment_protocol_id=protocol_id,
                    planned_condition_grid_id=planned_grid_id,
                    planned_condition_count=len(planned),
                    observed_grid_cell_id=spec.condition,
                    network_n_units=network.n_units,
                    network_excitatory_fraction=network.excitatory_fraction,
                    **physical_metrics,
                    **operator_metrics,
                    **geometry_by_population_mode[spec.population_gain],
                    **gate_provenance_metrics,
                )
                emit_success(spec.condition, metrics)
            except Exception as error:
                emit_failure(spec.condition, error)

        try:
            metrics = _evaluate_direct_baseline(
                train_fit=direct_fit,
                dataset=splits.test,
                prediction=test_prediction,
                gate_checkpoint_id=gate_checkpoint_id,
                split_id=splits.fingerprint,
                random_tape_id=tape.fingerprint,
                protocol_id=protocol_id,
                config=config,
            )
            metrics.update(
                test_episode_start_abs_belief_from_half=start_deviation,
                gate_fit_supervision="none",
                gate_received_true_q_h=False,
                planned_condition_grid_id=planned_grid_id,
                planned_condition_count=len(planned),
                observed_grid_cell_id=DIRECT_BASELINE_SPEC.condition,
                **gate_provenance_metrics,
            )
            emit_success(DIRECT_BASELINE_SPEC.condition, metrics)
        except Exception as error:
            emit_failure(DIRECT_BASELINE_SPEC.condition, error)

        expected = {spec.condition for spec in ALL_CONDITION_SPECS}
        if emitted != expected:
            raise RuntimeError(
                f"planned/emitted mismatch: missing={sorted(expected - emitted)}; "
                f"extra={sorted(emitted - expected)}"
            )
        if not np.array_equal(network.recurrent_weights, initial_weights):
            raise RuntimeError("frozen E/I receiver changed during Exp19")
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "Cue-filtered belief to high-rank E/I effective dynamics audit",
        "configs/formal/exp19_belief_ei_effective_dynamics.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
