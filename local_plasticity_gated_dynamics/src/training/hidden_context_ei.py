"""Leakage-safe hidden-belief bridge into a frozen Dale E/I receiver.

This is an intentionally narrow mechanism test.  Both sensory streams enter
the same recurrent network unchanged; a causal belief controls only a rank-one
population gain axis.  Recurrent weights are frozen, so positive results test
the sufficiency of low-dimensional effective control and do *not* establish a
three-factor recurrent-plasticity mechanism.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge

from src.analysis.end_to_end_energy import end_to_end_compute_proxy
from src.analysis.gate_metrics import context_calibration_summary
from src.analysis.rank_metrics import effective_rank, participation_ratio
from src.analysis.switching_metrics import jacobian_spectrum_summary
from src.models.belief_gain import (
    BeliefGainTrajectory,
    belief_gain_trajectory,
)
from src.models.context_belief import (
    GatePrediction,
    LearnedSymmetricHMM,
    MDRecurrentBeliefGate,
    NoGate,
    OracleBayesianFilter,
    deranged_trajectory_shuffle,
    episode_delay,
    neutral_clamp,
)
from src.models.ei_rate_network import EIRateNetwork
from src.tasks.hidden_context import HiddenContextDataset, TaskLearningBatch
from src.training.hidden_context_gate import HiddenContextSplits
from src.utils.reproducibility import derive_seed


BRIDGE_BASE_GATES = (
    "oracle_bayes",
    "learned_hmm",
    "md_recurrent_belief",
    "no_gate",
)
BRIDGE_MD_INTERVENTIONS = ("clamp", "delay", "shuffle")


def _fingerprint(*values: object) -> str:
    digest = hashlib.sha256()
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


@dataclass(frozen=True)
class SplitGatePredictions:
    """One frozen gate checkpoint evaluated independently on every split."""

    gate_model: str
    train: GatePrediction
    dev: GatePrediction
    test: GatePrediction
    checkpoint_id: str
    fit_metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.gate_model not in BRIDGE_BASE_GATES:
            raise ValueError(f"unknown bridge gate: {self.gate_model}")
        for prediction in (self.train, self.dev, self.test):
            if not isinstance(prediction, GatePrediction):
                raise TypeError("all split predictions must be GatePrediction")
            if prediction.test_accessed_true_context:
                raise ValueError("bridge predictions must not access test context")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty")


def fit_gate_split_predictions(
    gate_model: str,
    splits: HiddenContextSplits,
    *,
    context_hazard: float,
    cue_reliability: float,
    config: Mapping[str, Any],
    seed: int,
) -> SplitGatePredictions:
    """Fit on train cues only, then run causal filtering on each split."""

    if gate_model not in BRIDGE_BASE_GATES:
        raise ValueError(f"unknown bridge gate: {gate_model}")
    options: dict[str, Any]
    if gate_model == "oracle_bayes":
        options = {
            "context_hazard": float(context_hazard),
            "cue_reliability": float(cue_reliability),
            "seed": derive_seed(seed, "p2-ei", gate_model),
        }
        model = OracleBayesianFilter(**options)
        supervision = "known_generative_params"
    elif gate_model == "learned_hmm":
        configured = dict(config.get("learned_hmm", {}))
        if "max_iterations" in configured:
            configured["max_iter"] = configured.pop("max_iterations")
        if "tolerance" in configured:
            configured["tol"] = configured.pop("tolerance")
        options = {
            "seed": derive_seed(seed, "p2-ei", gate_model),
            **configured,
        }
        model = LearnedSymmetricHMM(**options).fit(splits.train.gate)
        supervision = "none"
    elif gate_model == "md_recurrent_belief":
        options = {
            "seed": derive_seed(seed, "p2-ei", gate_model),
            **dict(config.get("md_gate", {})),
        }
        model = MDRecurrentBeliefGate(**options).fit(splits.train.gate)
        supervision = "none"
    else:
        options = {"seed": derive_seed(seed, "p2-ei", gate_model)}
        model = NoGate(**options)
        supervision = "none"

    predictions = tuple(
        model.predict(item.gate) for item in (splits.train, splits.dev, splits.test)
    )
    model_audit = model.audit_metadata() if hasattr(model, "audit_metadata") else {}
    fit_metadata = {
        **model_audit,
        "gate_fit_supervision": supervision,
        "gate_received_true_q_h": gate_model == "oracle_bayes",
        "gate_fit_accessed_task_target": False,
        "gate_test_accessed_task_target": False,
    }
    checkpoint = _fingerprint(
        "hidden-context-ei-gate-v1",
        gate_model,
        predictions[0].parameters,
        predictions[0].fit_trial_ids,
        splits.train.gate.fingerprint,
        options,
    )
    return SplitGatePredictions(
        gate_model=gate_model,
        train=predictions[0],
        dev=predictions[1],
        test=predictions[2],
        checkpoint_id=checkpoint,
        fit_metadata=fit_metadata,
    )


def intervene_on_test_prediction(
    fitted: SplitGatePredictions,
    intervention: str,
    *,
    delay_trials: int,
    seed: int,
) -> GatePrediction:
    """Branch a held-out MD trajectory without refitting receiver or readout."""

    if fitted.gate_model != "md_recurrent_belief":
        raise ValueError("bridge interventions require the intact MD-like gate")
    if intervention == "clamp":
        return neutral_clamp(fitted.test)
    if intervention == "delay":
        return episode_delay(fitted.test, int(delay_trials))
    if intervention == "shuffle":
        return deranged_trajectory_shuffle(
            fitted.test,
            seed=derive_seed(seed, "p2-ei", "test-trajectory-shuffle"),
        )
    raise ValueError(f"unknown bridge intervention: {intervention}")


@dataclass(frozen=True)
class ReceiverSimulation:
    """Readout features plus receiver-level audit aggregates."""

    features: np.ndarray
    mean_x: np.ndarray
    mean_gain: np.ndarray
    gain: BeliefGainTrajectory
    input_event_sum: float
    recurrent_event_sum: float
    firing_sum: float
    receiver_fingerprint: str


def _epoch_feature_indices(task: TaskLearningBatch) -> tuple[np.ndarray, ...]:
    indices = tuple(
        np.flatnonzero(np.asarray(task.epoch) == epoch)
        for epoch in ("sensory", "delay", "response")
    )
    if any(item.size == 0 for item in indices):
        raise ValueError("receiver feature epochs must all be populated")
    return indices


def simulate_receiver(
    network: EIRateNetwork,
    dataset: HiddenContextDataset,
    posterior_state1: np.ndarray,
    gain_axis: np.ndarray,
    *,
    gain_strength: float,
    integration_substeps: int,
    trial_batch_size: int,
) -> ReceiverSimulation:
    """Run unchanged sensory channels through one frozen receiver checkpoint."""

    if network.n_inputs != 2:
        raise ValueError("hidden-context E/I receiver must have exactly two inputs")
    if (
        isinstance(trial_batch_size, (bool, np.bool_))
        or not isinstance(trial_batch_size, (int, np.integer))
        or int(trial_batch_size) < 1
    ):
        raise ValueError("trial_batch_size must be a positive integer")
    posterior = np.asarray(posterior_state1, dtype=np.float64)
    n_trials = dataset.task.inputs.shape[0]
    if posterior.shape != (n_trials,):
        raise ValueError("posterior_state1 must align with receiver trials")
    gain = belief_gain_trajectory(
        posterior,
        dataset.task.epoch,
        gain_axis,
        strength=float(gain_strength),
        neutral_epochs=("cue",),
    )
    # Capability boundary: cue one-hot channels 2:4 are not passed to the
    # receiver.  Both sensory streams 0:2 are copied without belief mixing.
    sensory_inputs = np.asarray(dataset.task.inputs[:, :, :2], dtype=np.float64)
    epoch_indices = _epoch_feature_indices(dataset.task)
    feature_blocks: list[np.ndarray] = []
    mean_x_sum = np.zeros(network.n_units, dtype=np.float64)
    mean_gain_sum = np.zeros(network.n_units, dtype=np.float64)
    mean_count = 0
    input_events = 0.0
    recurrent_events = 0.0
    firing = 0.0
    batch = int(trial_batch_size)
    for start in range(0, n_trials, batch):
        stop = min(start + batch, n_trials)
        trajectory = network.run_trial_batch(
            sensory_inputs[start:stop],
            gains=gain.gains[start:stop],
            substeps=int(integration_substeps),
        )
        # History index 0 is the initial state; coarse task step t is stored at
        # t+1 and is the fixed sampling convention for every condition.
        saved_rates = trajectory.rates[:, 1:]
        features = np.concatenate(
            [np.mean(saved_rates[:, indices], axis=1) for indices in epoch_indices],
            axis=1,
        )
        feature_blocks.append(features)
        active = gain.active_time_mask
        saved_x = trajectory.x[:, 1:][:, active]
        mean_x_sum += np.sum(saved_x, axis=(0, 1))
        mean_gain_sum += np.sum(gain.gains[start:stop][:, active], axis=(0, 1))
        mean_count += saved_x.shape[0] * saved_x.shape[1]
        input_events += trajectory.substep_input_event_sum
        recurrent_events += trajectory.substep_recurrent_event_sum
        firing += trajectory.substep_firing_sum
    feature_matrix = np.concatenate(feature_blocks, axis=0)
    mean_x = mean_x_sum / mean_count
    mean_gain = mean_gain_sum / mean_count
    return ReceiverSimulation(
        features=feature_matrix,
        mean_x=mean_x,
        mean_gain=mean_gain,
        gain=gain,
        input_event_sum=float(input_events),
        recurrent_event_sum=float(recurrent_events),
        firing_sum=float(firing),
        receiver_fingerprint=_fingerprint(
            "frozen-ei-receiver-features-v1",
            dataset.task.fingerprint,
            network.recurrent_weights,
            network.input_weights,
            gain.fingerprint,
            int(integration_substeps),
            feature_matrix,
        ),
    )


def _task_targets(task: TaskLearningBatch) -> np.ndarray:
    response = np.flatnonzero(np.asarray(task.epoch) == "response")
    masked = task.loss_mask[:, response]
    if not np.all(masked):
        raise ValueError("all response steps must be supervised for bridge readout")
    values = task.targets[:, response, 0]
    if not np.all(values == values[:, :1]):
        raise ValueError("response target must be constant within each trial")
    targets = np.asarray(values[:, 0], dtype=np.float64)
    if not np.isin(targets, [-1.0, 1.0]).all():
        raise ValueError("bridge currently requires binary choice targets")
    return targets


@dataclass(frozen=True)
class ReceiverReadout:
    """Train-only standardizer and ridge readout."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    alpha: float
    train_data_id: str
    checkpoint_id: str

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError("readout features do not match fitted dimension")
        return (values - self.mean) / self.scale

    def scores(self, features: np.ndarray) -> np.ndarray:
        return self.transform(features) @ self.weights + self.intercept


def fit_receiver_readout(
    simulation: ReceiverSimulation,
    task: TaskLearningBatch,
    *,
    alpha: float,
) -> ReceiverReadout:
    """Fit all preprocessing and the readout on training trials only."""

    if not np.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise ValueError("readout alpha must be non-negative and finite")
    features = np.asarray(simulation.features, dtype=np.float64)
    targets = _task_targets(task)
    if features.shape[0] != targets.size:
        raise ValueError("readout features and targets disagree on trial count")
    mean = np.mean(features, axis=0)
    empirical_scale = np.std(features, axis=0)
    scale = np.where(empirical_scale > 0.0, empirical_scale, 1.0)
    standardized = (features - mean) / scale
    model = Ridge(alpha=float(alpha), fit_intercept=True).fit(standardized, targets)
    weights = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    train_data_id = _fingerprint(task.fingerprint, simulation.receiver_fingerprint)
    checkpoint = _fingerprint(
        "frozen-ei-ridge-readout-v1",
        mean,
        scale,
        weights,
        float(model.intercept_),
        float(alpha),
        train_data_id,
    )
    return ReceiverReadout(
        mean=mean,
        scale=scale,
        weights=weights,
        intercept=float(model.intercept_),
        alpha=float(alpha),
        train_data_id=train_data_id,
        checkpoint_id=checkpoint,
    )


def _balanced_accuracy(predicted: np.ndarray, target: np.ndarray) -> float:
    recalls = [
        np.mean(predicted[target == label] == label) for label in np.unique(target)
    ]
    return float(np.mean(recalls))


def _jacobian_metrics(
    network: EIRateNetwork, mean_x: np.ndarray, mean_gain: np.ndarray
) -> dict[str, object]:
    activated = np.tanh(mean_gain * mean_x)
    derivative = mean_gain * (1.0 - activated * activated)
    if network.activation_name == "rectified_tanh":
        derivative = np.where(activated > 0.0, derivative, 0.0)
    jacobian = -np.eye(network.n_units) + network.recurrent_weights @ np.diag(
        derivative
    )
    jacobian = jacobian / network.time_constants[:, np.newaxis]
    summary = jacobian_spectrum_summary(jacobian, dynamics="continuous")
    return {
        "jacobian_max_real_part": summary.max_real_part,
        "jacobian_spectral_radius": summary.spectral_radius,
        "jacobian_unstable_fraction": summary.unstable_fraction,
        "jacobian_stability_margin": summary.stability_margin,
    }


def evaluate_receiver_condition(
    *,
    network: EIRateNetwork,
    simulation: ReceiverSimulation,
    readout: ReceiverReadout,
    prediction: GatePrediction,
    dataset: HiddenContextDataset,
    gate_model: str,
    intervention: str,
    gate_checkpoint_id: str,
    gain_axis_id: str,
    split_id: str,
    network_init_id: str,
    gate_operations_per_trial: float,
    gate_state_updates_per_trial: float,
) -> dict[str, object]:
    """Freeze receiver output, then grant context truth only to metric code."""

    if not np.array_equal(prediction.trial_ids, dataset.gate.trial_ids):
        raise RuntimeError("prediction and held-out receiver trials are misaligned")
    if prediction.test_accessed_true_context:
        raise RuntimeError("held-out gate prediction accessed hidden context")
    scores = readout.scores(simulation.features)
    predicted = np.where(scores >= 0.0, 1, -1)
    targets = _task_targets(dataset.task).astype(int)
    standardized = readout.transform(simulation.features)
    calibration = context_calibration_summary(
        prediction.context_probability,
        dataset.truth.hidden_states,
        n_bins=10,
        epsilon=1e-6,
    )
    compute = end_to_end_compute_proxy(
        n_trials=targets.size,
        input_event_sum=simulation.input_event_sum,
        recurrent_event_sum=simulation.recurrent_event_sum,
        firing_sum=simulation.firing_sum,
        readout_features=standardized,
        readout_weights=readout.weights,
        gate_operations_per_trial=float(gate_operations_per_trial),
        gate_state_updates_per_trial=float(gate_state_updates_per_trial),
    )
    base_prediction = prediction.base_prediction_fingerprint
    intervention_postfit = intervention != "none"
    if intervention_postfit and not base_prediction:
        raise RuntimeError("intervention is missing its intact belief provenance")
    return {
        "status": "complete",
        "statistics_unit": "seed",
        "split_unit": "episode",
        "bridge_scope": "belief_gain_to_frozen_dale_ei_receiver",
        "receiver_gate_mode": "rank_one_gain_only",
        "sensory_input_policy": "both_streams_unchanged_no_cue_channels",
        "recurrent_learning": False,
        "three_factor_plasticity_claim_eligible": False,
        "behavior_accuracy": float(np.mean(predicted == targets)),
        "behavior_balanced_accuracy": _balanced_accuracy(predicted, targets),
        "context_nll": calibration.nll,
        "context_brier": calibration.brier,
        "context_ece": calibration.expected_calibration_error,
        "context_accuracy": calibration.accuracy,
        "activity_participation_ratio": participation_ratio(simulation.features),
        "recurrent_effective_rank": effective_rank(network.recurrent_weights),
        "effective_control_rank": simulation.gain.control_rank,
        "gain_min": float(np.min(simulation.gain.gains)),
        "gain_max": float(np.max(simulation.gain.gains)),
        "plasticity_l1_cost": 0.0,
        "input_weighted_events_per_trial": compute.input_weighted_events,
        "recurrent_weighted_events_per_trial": compute.recurrent_weighted_events,
        "receiver_firing_magnitude_per_trial": compute.receiver_firing_magnitude,
        "readout_weighted_events_per_trial": compute.readout_weighted_events,
        "gate_primitive_operations_per_trial": compute.gate_primitive_operations,
        "gate_state_updates_per_trial": compute.gate_state_updates,
        "end_to_end_compute_proxy_per_trial": compute.total_compute_proxy,
        "energy_interpretation": compute.interpretation,
        "intervention_postfit": intervention_postfit,
        "intervention_reuses_intact_readout": intervention_postfit,
        "intervention_reuses_intact_receiver": intervention_postfit,
        "intervention_reuses_intact_gate_checkpoint": intervention_postfit,
        "gate_fit_accessed_true_context": bool(prediction.fit_accessed_true_context),
        "gate_test_accessed_true_context": False,
        "true_context_access_scope": "evaluation_only",
        "preprocessing_fit_train_only": True,
        "readout_fit_train_only": True,
        "base_conditions_share_readout": False,
        "base_comparison_scope": "separately_train_optimized_pipeline_comparison",
        "efficiency_claim_eligible": False,
        "gate_compute_accounting": "declared_constant_primitive_estimate",
        "gate_checkpoint_id": gate_checkpoint_id,
        "belief_trajectory_id": prediction.fingerprint,
        "intact_belief_trajectory_id": base_prediction or prediction.fingerprint,
        "gain_trajectory_id": simulation.gain.fingerprint,
        "gain_axis_id": gain_axis_id,
        "receiver_simulation_id": simulation.receiver_fingerprint,
        "readout_checkpoint_id": readout.checkpoint_id,
        "readout_fit_data_id": readout.train_data_id,
        "network_init_id": network_init_id,
        "network_initialization_id": network_init_id,
        "split_id": split_id,
        "test_trial_count": int(targets.size),
        **_jacobian_metrics(network, simulation.mean_x, simulation.mean_gain),
    }


__all__ = [
    "BRIDGE_BASE_GATES",
    "BRIDGE_MD_INTERVENTIONS",
    "ReceiverReadout",
    "ReceiverSimulation",
    "SplitGatePredictions",
    "evaluate_receiver_condition",
    "fit_gate_split_predictions",
    "fit_receiver_readout",
    "intervene_on_test_prediction",
    "simulate_receiver",
]
