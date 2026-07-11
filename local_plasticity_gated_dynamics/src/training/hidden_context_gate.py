"""Leakage-safe fitting and evaluation for the hidden-context gate audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from src.analysis.gate_metrics import (
    context_calibration_summary,
    gate_energy_summary,
    gated_behavior_summary,
    switch_inference_summary,
)
from src.analysis.p2_protocol import p2_protocol_id
from src.models.context_belief import (
    GatePrediction,
    LearnedSymmetricHMM,
    MDRecurrentBeliefGate,
    NoGate,
    OracleBayesianFilter,
    SupervisedCueGate,
    deranged_trajectory_shuffle,
    episode_delay,
    neutral_clamp,
)
from src.tasks.hidden_context import HiddenContextDataset
from src.utils.reproducibility import derive_seed


BASE_GATES = (
    "oracle_bayes",
    "supervised_upper_bound",
    "learned_hmm",
    "md_recurrent_belief",
    "no_gate",
)
MD_INTERVENTIONS = ("clamp", "delay", "shuffle")


class ScientificallyInvalidCondition(ValueError):
    """Raised when a cell is computable but violates its evidence contract."""


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
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=repr,
                ).encode("utf-8")
            )
        digest.update(b"\0")
    return digest.hexdigest()


def _probability(value: object, *, name: str, lower: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar probability")
    result = float(value)
    if not np.isfinite(result) or not lower <= result <= 1.0:
        raise ValueError(f"{name} must lie in [{lower}, 1]")
    return result


@dataclass(frozen=True)
class HiddenGateCondition:
    """One pre-registered q/h/gate/intervention cell."""

    cue_reliability: float
    context_hazard: float
    gate_model: str
    intervention: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cue_reliability",
            _probability(self.cue_reliability, name="cue_reliability", lower=0.5),
        )
        object.__setattr__(
            self,
            "context_hazard",
            _probability(self.context_hazard, name="context_hazard", lower=0.0),
        )
        if self.gate_model not in BASE_GATES:
            raise ValueError(f"unknown gate_model: {self.gate_model}")
        allowed = (
            {"none", *MD_INTERVENTIONS}
            if self.gate_model == "md_recurrent_belief"
            else {"none"}
        )
        if self.intervention not in allowed:
            raise ValueError(
                f"intervention {self.intervention!r} is invalid for {self.gate_model}"
            )

    @property
    def name(self) -> str:
        q = f"{self.cue_reliability:.2f}".replace(".", "p")
        h = f"{self.context_hazard:.2f}".replace(".", "p")
        return f"q{q}__h{h}__{self.gate_model}__{self.intervention}"

    def as_dict(self) -> dict[str, object]:
        return {"condition": self.name, **asdict(self)}


def build_hidden_gate_conditions(
    config: Mapping[str, Any],
) -> list[HiddenGateCondition]:
    """Build the exact base-plus-MD-intervention grid without duplicates."""

    reliabilities = tuple(float(value) for value in config["cue_reliabilities"])
    hazards = tuple(float(value) for value in config["context_hazards"])
    if not reliabilities or len(reliabilities) != len(set(reliabilities)):
        raise ValueError("cue_reliabilities must be non-empty and unique")
    if not hazards or len(hazards) != len(set(hazards)):
        raise ValueError("context_hazards must be non-empty and unique")
    conditions: list[HiddenGateCondition] = []
    for reliability in reliabilities:
        for hazard in hazards:
            conditions.extend(
                HiddenGateCondition(reliability, hazard, gate) for gate in BASE_GATES
            )
            conditions.extend(
                HiddenGateCondition(
                    reliability,
                    hazard,
                    "md_recurrent_belief",
                    intervention,
                )
                for intervention in MD_INTERVENTIONS
            )
    if len(conditions) != len({item.name for item in conditions}):
        raise RuntimeError("hidden-context condition names are not unique")
    return conditions


@dataclass(frozen=True)
class HiddenContextSplits:
    """Whole-episode train/dev/test split with an immutable audit ID."""

    train: HiddenContextDataset
    dev: HiddenContextDataset
    test: HiddenContextDataset
    fingerprint: str

    def __post_init__(self) -> None:
        for item in (self.train, self.dev, self.test):
            if not isinstance(item, HiddenContextDataset):
                raise TypeError("all splits must be HiddenContextDataset instances")
        episode_sets = [
            set(item.gate.episode_ids.tolist())
            for item in (self.train, self.dev, self.test)
        ]
        if any(
            episode_sets[first] & episode_sets[second]
            for first, second in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("train/dev/test episodes must be disjoint")
        if not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise ValueError("split fingerprint must be non-empty")


def split_hidden_context_dataset(
    dataset: HiddenContextDataset,
    *,
    outer_test_fraction: float,
    validation_fraction: float,
    seed: int,
) -> HiddenContextSplits:
    """Split complete episodes; no time point or partial episode is sampled."""

    development, test = dataset.train_test_split(
        test_fraction=float(outer_test_fraction),
        seed=derive_seed(seed, "p2", "outer-episode-split"),
    )
    train, dev = development.train_test_split(
        test_fraction=float(validation_fraction),
        seed=derive_seed(seed, "p2", "inner-episode-split"),
    )
    split_id = _fingerprint(
        train.gate.episode_ids,
        dev.gate.episode_ids,
        test.gate.episode_ids,
        "whole_episode_train_dev_test_v1",
    )
    return HiddenContextSplits(train, dev, test, split_id)


@dataclass(frozen=True)
class FittedGate:
    """Frozen test prediction and the checkpoint from which it came."""

    gate_model: str
    prediction: GatePrediction
    checkpoint_id: str
    fit_metadata: dict[str, object]


def _gate_checkpoint_id(
    gate_model: str,
    prediction: GatePrediction,
    options: Mapping[str, Any],
) -> str:
    return _fingerprint(
        "hidden-context-gate-checkpoint-v1",
        gate_model,
        prediction.parameters,
        prediction.fit_trial_ids,
        prediction.fit_episode_ids,
        dict(options),
    )


def fit_hidden_gate(
    gate_model: str,
    splits: HiddenContextSplits,
    *,
    context_hazard: float,
    cue_reliability: float,
    config: Mapping[str, Any],
    seed: int,
) -> FittedGate:
    """Fit one gate using only its explicitly declared training capability."""

    if gate_model not in BASE_GATES:
        raise ValueError(f"unknown gate_model: {gate_model}")
    options: dict[str, Any]
    supervision: str
    if gate_model == "oracle_bayes":
        options = {
            "context_hazard": float(context_hazard),
            "cue_reliability": float(cue_reliability),
        }
        model = OracleBayesianFilter(**options)
        supervision = "known_generative_params"
    elif gate_model == "supervised_upper_bound":
        configured = dict(config.get("supervised_gate", {}))
        ridge = float(configured.pop("ridge", 1e-3))
        options = {
            "C": float(configured.pop("C", 1.0 / max(ridge, 1e-12))),
            "seed": derive_seed(seed, "p2", gate_model),
            **configured,
        }
        model = SupervisedCueGate(**options).fit_supervised(
            splits.train.gate,
            splits.train.truth.hidden_states,
        )
        supervision = "train_context_labels"
    elif gate_model == "learned_hmm":
        configured = dict(config.get("learned_hmm", {}))
        if "max_iterations" in configured:
            configured["max_iter"] = configured.pop("max_iterations")
        if "tolerance" in configured:
            configured["tol"] = configured.pop("tolerance")
        options = {
            "seed": derive_seed(seed, "p2", gate_model),
            **configured,
        }
        model = LearnedSymmetricHMM(**options).fit(splits.train.gate)
        supervision = "none"
    elif gate_model == "md_recurrent_belief":
        options = {
            "seed": derive_seed(seed, "p2", gate_model),
            **dict(config.get("md_gate", {})),
        }
        model = MDRecurrentBeliefGate(**options).fit(splits.train.gate)
        supervision = "none"
    else:
        options = {}
        model = NoGate()
        supervision = "none"

    prediction = model.predict(splits.test.gate)
    model_audit = model.audit_metadata() if hasattr(model, "audit_metadata") else {}
    metadata = {
        **model_audit,
        **prediction.audit_metadata(),
        "gate_fit_supervision": supervision,
        "gate_received_true_q_h": gate_model == "oracle_bayes",
    }
    checkpoint = _gate_checkpoint_id(gate_model, prediction, options)
    return FittedGate(gate_model, prediction, checkpoint, metadata)


def intervene_on_prediction(
    fitted: FittedGate,
    intervention: str,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> GatePrediction:
    """Apply a post-fit intervention without changing the intact checkpoint."""

    if fitted.gate_model != "md_recurrent_belief":
        raise ValueError("causal interventions are defined only for the MD-like gate")
    if intervention == "clamp":
        return neutral_clamp(fitted.prediction)
    if intervention == "delay":
        delay = int(dict(config.get("interventions", {})).get("delay_trials", 1))
        return episode_delay(fitted.prediction, delay)
    if intervention == "shuffle":
        return deranged_trajectory_shuffle(
            fitted.prediction,
            seed=derive_seed(seed, "p2", "belief-trajectory-shuffle"),
        )
    raise ValueError(f"unknown intervention: {intervention}")


def _sensory_evidence(dataset: HiddenContextDataset) -> np.ndarray:
    steps = dataset.config.epoch_steps
    start = steps["cue"]
    stop = start + steps["sensory"]
    evidence = dataset.task.inputs[:, start:stop, :2].sum(axis=1)
    return np.asarray(evidence / dataset.config.input_scale, dtype=np.float64)


def _stream_id(dataset: HiddenContextDataset, name: str) -> str:
    streams = dict(dataset.random_stream_fingerprints)
    if name not in streams:
        raise RuntimeError(f"missing random stream fingerprint: {name}")
    return streams[name]


def evaluate_gate_prediction(
    fitted: FittedGate,
    prediction: GatePrediction,
    splits: HiddenContextSplits,
    condition: HiddenGateCondition,
    *,
    config: Mapping[str, Any],
    profile: str,
    seed: int,
) -> dict[str, object]:
    """Freeze a prediction, then grant truth capability only to metric code."""

    if (
        prediction.trial_ids.shape != splits.test.gate.trial_ids.shape
        or not np.array_equal(prediction.trial_ids, splits.test.gate.trial_ids)
    ):
        raise RuntimeError("gate prediction is not aligned to held-out trial IDs")
    posterior = np.array(prediction.context_probability, dtype=np.float64, copy=True)
    posterior.setflags(write=False)

    calibration = context_calibration_summary(
        posterior,
        splits.test.truth.hidden_states,
        n_bins=10,
        epsilon=1e-6,
    )
    switch_options = dict(config.get("switch_metrics", {}))
    minimum_switches = int(switch_options.pop("minimum_eligible_switches", 20))
    switches = switch_inference_summary(
        posterior,
        splits.test.truth.hidden_states,
        splits.test.truth.episode_ids,
        **switch_options,
    )
    if switches.switch_count < minimum_switches:
        raise ScientificallyInvalidCondition(
            f"only {switches.switch_count} eligible switches; requires "
            f"{minimum_switches}"
        )
    behavior = gated_behavior_summary(
        posterior,
        _sensory_evidence(splits.test),
        splits.test.truth.choices,
    )
    energy = gate_energy_summary(posterior, splits.test.gate.episode_ids)

    is_supervised = condition.gate_model == "supervised_upper_bound"
    is_oracle = condition.gate_model == "oracle_bayes"
    is_intervention = condition.intervention != "none"
    fit_accessed_truth = bool(fitted.fit_metadata.get("gate_fit_accessed_true_context"))
    test_accessed_truth = bool(
        fitted.fit_metadata.get("gate_test_accessed_true_context")
    )
    if fit_accessed_truth != is_supervised or test_accessed_truth:
        raise RuntimeError("gate access audit disagrees with the registered capability")
    if (
        is_intervention
        and prediction.base_prediction_fingerprint != fitted.prediction.fingerprint
    ):
        raise RuntimeError("intervention did not branch from the intact MD prediction")

    episode_sets = [
        set(item.gate.episode_ids.tolist())
        for item in (splits.train, splits.dev, splits.test)
    ]
    disjoint = not any(
        episode_sets[first] & episode_sets[second]
        for first, second in ((0, 1), (0, 2), (1, 2))
    )
    readout_protocol_id = "fixed_belief_weighted_evidence_no_fit_v1"
    readout_id = _fingerprint(
        readout_protocol_id,
        splits.train.task.fingerprint,
    )
    belief_id = prediction.fingerprint
    hidden_state_id = splits.test.truth.fingerprint
    observation_id = splits.test.gate.fingerprint
    task_id = splits.test.task.fingerprint
    network_id = _fingerprint("gate-only-no-pfc-network", seed)
    provenance = {
        "hidden_context_task": True,
        "cue_encodes_observation_not_state": True,
        "gate_test_accessed_true_context": False,
        "gate_fit_accessed_true_context": is_supervised,
        "third_factor_accessed_true_context": False,
        "oracle_warm_start_used": False,
        "md_fit_used_context_bias": False,
        "gate_fit_accessed_task_target": False,
        "gate_test_accessed_task_target": False,
        "gate_test_future_observations_accessed": False,
        "gate_fit_used_batch_smoothing": condition.gate_model == "learned_hmm",
        "state_label_alignment_accessed_true_context": False,
        "test_switch_boundaries_accessed_by_model": False,
        "preprocessing_fit_train_only": True,
        "hyperparameters_preregistered": True,
        "dev_used_for_selection": False,
        "train_dev_test_episode_disjoint": disjoint,
        "belief_online_causal": True,
        "predictions_frozen_before_truth_scoring": not posterior.flags.writeable,
        "true_context_access_scope": (
            "train_gate_fit_and_evaluation" if is_supervised else "evaluation_only"
        ),
        "gate_fit_supervision": (
            "train_context_labels"
            if is_supervised
            else ("known_generative_params" if is_oracle else "none")
        ),
        "gate_received_true_q_h": is_oracle,
        "eligible_for_p2_support": not is_supervised,
        "intervention_postfit": is_intervention,
        "intervention_reuses_intact_checkpoint": is_intervention,
        "intervention_reuses_intact_readout": is_intervention,
        "intervention_permutation_accessed_true_context": False,
    }
    return {
        "status": "complete",
        "profile": str(profile),
        "training_algorithm": "hidden_context_gate_grid",
        "used_autograd": False,
        "statistics_unit": "seed",
        "split_unit": "episode",
        "gate_only_benchmark": True,
        "task_network_scope": "belief_gated_fixed_evidence_integrator",
        "context_nll": calibration.nll,
        "context_brier": calibration.brier,
        "context_ece": calibration.expected_calibration_error,
        "context_accuracy": calibration.accuracy,
        "switch_latency_trials": switches.mean_latency_trials,
        "switch_latency_median_trials": switches.median_latency_trials,
        "switch_censored_fraction": switches.censored_fraction,
        "false_switch_rate": switches.false_switch_rate,
        "missed_switch_rate": switches.missed_switch_rate,
        "behavior_accuracy": behavior.accuracy,
        "behavior_balanced_accuracy": behavior.balanced_accuracy,
        "energy_proxy_per_trial": energy.total_energy,
        "gate_state_energy": energy.state_energy,
        "gate_transition_energy": energy.transition_energy,
        "eligible_switch_count": switches.switch_count,
        "missed_switch_count": switches.missed_switch_count,
        "false_switch_count": switches.false_switch_count,
        "eligible_switch_trial_count": switches.eligible_trial_count,
        "latency_limit_trials": int(switch_options.get("max_latency", 5)),
        "latency_sustain_trials": int(switch_options.get("sustain_trials", 2)),
        "posterior_threshold": float(switch_options.get("posterior_threshold", 0.8)),
        "minimum_state_duration": int(switch_options.get("minimum_state_duration", 5)),
        "switch_tolerance_trials": int(switch_options.get("match_tolerance", 1)),
        "minimum_eligible_switches": minimum_switches,
        "delay_trials": int(
            dict(config.get("interventions", {})).get("delay_trials", 1)
        ),
        "outer_test_fraction": float(config.get("outer_test_fraction", np.nan)),
        "validation_fraction": float(config.get("validation_fraction", np.nan)),
        "p2_protocol_id": p2_protocol_id(config),
        "train_episode_count": len(episode_sets[0]),
        "dev_episode_count": len(episode_sets[1]),
        "test_episode_count": len(episode_sets[2]),
        "train_trial_count": int(splits.train.gate.trial_ids.size),
        "dev_trial_count": int(splits.dev.gate.trial_ids.size),
        "test_trial_count": int(splits.test.gate.trial_ids.size),
        "random_tape_id": splits.test.random_tape_fingerprint,
        "state_tape_id": hidden_state_id,
        "hidden_state_tape_id": hidden_state_id,
        "observation_tape_id": observation_id,
        "task_tape_id": task_id,
        "noise_tape_id": _stream_id(splits.test, "sensory_noise"),
        "network_init_id": network_id,
        "network_initialization_id": network_id,
        "split_id": splits.fingerprint,
        "readout_fit_data_id": splits.train.task.fingerprint,
        "readout_protocol_id": readout_protocol_id,
        "readout_id": readout_id,
        "checkpoint_id": fitted.checkpoint_id,
        "belief_trajectory_id": belief_id,
        "intact_belief_trajectory_id": fitted.prediction.fingerprint,
        "gate_parameters": list(prediction.parameters),
        "gate_learning_rule": fitted.fit_metadata.get(
            "local_update_rule",
            "batch_em"
            if condition.gate_model == "learned_hmm"
            else "fixed_or_supervised_comparator",
        ),
        "hmm_fit_converged": fitted.fit_metadata.get("em_converged"),
        "hmm_fit_iterations": fitted.fit_metadata.get("em_iterations"),
        "md_moment_anchor_identifiable": fitted.fit_metadata.get(
            "moment_anchor_identifiable"
        ),
        "estimated_context_hazard": fitted.fit_metadata.get("estimated_context_hazard"),
        "estimated_cue_reliability": fitted.fit_metadata.get(
            "estimated_cue_reliability"
        ),
        "md_two_slice_hazard": fitted.fit_metadata.get("two_slice_hazard"),
        "md_two_slice_reliability": fitted.fit_metadata.get("two_slice_reliability"),
        "md_moment_hazard": fitted.fit_metadata.get("moment_hazard"),
        "md_moment_reliability": fitted.fit_metadata.get("moment_reliability"),
        "md_moment_anchor_weight": fitted.fit_metadata.get("moment_anchor_weight"),
        **provenance,
    }


__all__ = [
    "BASE_GATES",
    "MD_INTERVENTIONS",
    "FittedGate",
    "HiddenContextSplits",
    "HiddenGateCondition",
    "ScientificallyInvalidCondition",
    "build_hidden_gate_conditions",
    "evaluate_gate_prediction",
    "fit_hidden_gate",
    "intervene_on_prediction",
    "split_hidden_context_dataset",
]
