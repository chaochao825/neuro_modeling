"""Closed-loop learning of a belief-controlled population-gain axis.

This experiment fixes a high-rank Dale E/I receiver, a cue-only MD-like
belief filter, and a train-only ridge readout.  Only the population gain
vector is adapted on development episodes.  The local condition uses a
forward diagonal e-prop trace; exact forward sensitivity and Torch BPTT train
the same axis and are explicit oracle/baseline conditions.

Every update is followed by a newly simulated episode.  No condition changes
the recurrent matrix, input matrix, readout, gate, split, trial order, or
random tape.  The held-out test split is used only after the axis is frozen.

For results-schema compatibility, ``current_off_policy`` names a frozen
zero-axis trajectory baseline.  Its proposal is the Exp23 per-unit
``[state, rate]`` block-local e-prop rule; it is not the distinct Exp22
eligibility-tape implementation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.linear_model import Ridge

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp19_belief_ei_effective_dynamics as exp19
from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.analysis.controller_gradient_audit import directional_cosine
from src.analysis.functional_budget_matching import (
    functional_state_displacement,
    match_functional_state_displacement,
)
from src.models.context_belief import MDRecurrentBeliefGate
from src.models.ei_rate_network import EIRateNetwork
from src.plasticity.controller_eprop import (
    apply_gain_bounds,
    block_local_gain_axis_eprop_sensitivities,
    diagonal_eprop_gradient,
)
from src.plasticity.forward_sensitivity import (
    epoch_mean_readout_gradient,
    simulate_frozen_gain_axis_trajectory,
)
from src.tasks.hidden_context import (
    HiddenContextConfig,
    HiddenContextDataset,
    TaskLearningBatch,
    generate_hidden_context,
    make_hidden_context_random_tape,
)
from src.training.hidden_context_ei import (
    ReceiverReadout,
)
from src.training.hidden_context_gate import HiddenContextSplits, split_hidden_context_dataset
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


EXPERIMENT = "exp23_closed_loop_local_controller"
TASKS = ("current", "delayed")
CONDITIONS = (
    "frozen",
    "current_off_policy",
    "random_update",
    "exact_forward_sensitivity",
    "bptt_axis_only",
    "local_eprop",
)
GradientMode = Literal["exact", "local"]
FROZEN_TRAJECTORY_BLOCK_LOCAL_METHOD = (
    "frozen_neutral_trajectory_block_local_eprop"
)


@dataclass(frozen=True, slots=True)
class AxisForward:
    features: np.ndarray
    scores: np.ndarray
    targets: np.ndarray
    loss: float
    task_loss: float
    rate_penalty: float
    axis_penalty: float
    rates: np.ndarray
    states: np.ndarray
    gains: np.ndarray
    saturation_fraction: float
    max_firing_rate: float
    mean_firing_rate: float
    gradient: np.ndarray | None
    exact_gradient: np.ndarray | None


@dataclass(frozen=True, slots=True)
class UpdateReceipt:
    axis: np.ndarray
    raw_update: np.ndarray
    applied_update: np.ndarray
    gain_scale: float
    functional_scale: float
    state_displacement: float
    loss_before: float
    loss_after: float
    improved: bool


@dataclass(frozen=True, slots=True)
class RecurrentWeightCheckpoint:
    """Independent bitwise receipt for the frozen recurrent matrix."""

    weights: np.ndarray
    fingerprint: str


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


def _capture_recurrent_weight_checkpoint(
    network: EIRateNetwork,
) -> RecurrentWeightCheckpoint:
    """Copy and hash recurrent weights so later in-place changes are detectable."""

    weights = np.array(
        network.recurrent_weights,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    weights.setflags(write=False)
    return RecurrentWeightCheckpoint(
        weights=weights,
        fingerprint=_fingerprint(
            "exp23-recurrent-weights-bitwise-v1",
            weights,
        ),
    )


def _assert_recurrent_weight_checkpoint_unchanged(
    network: EIRateNetwork,
    expected: RecurrentWeightCheckpoint,
    *,
    phase: str,
) -> RecurrentWeightCheckpoint:
    """Fail closed unless both the copied values and their bitwise hash match."""

    observed = _capture_recurrent_weight_checkpoint(network)
    if (
        observed.fingerprint != expected.fingerprint
        or not np.array_equal(observed.weights, expected.weights)
    ):
        raise RuntimeError(f"Exp23 recurrent weights changed during {phase}")
    return observed


def _condition_training_algorithm(condition: str) -> str:
    """Return an accurate method label while retaining the legacy condition key."""

    if condition == "current_off_policy":
        return FROZEN_TRAJECTORY_BLOCK_LOCAL_METHOD
    return condition


def _planned_conditions(config: dict[str, Any]) -> list[dict[str, object]]:
    del config
    return [
        {
            "condition": condition,
            "condition_method_label": _condition_training_algorithm(condition),
            "condition_key_is_legacy_alias": condition == "current_off_policy",
            "task_variant": task,
            "controller_parameterization": "population_gain_axis",
        }
        for task in TASKS
        for condition in CONDITIONS
    ]


def _delayed_task(dataset: HiddenContextDataset) -> HiddenContextDataset:
    """Reorder one task to cue, blank delay, sensory, response."""

    task = dataset.task
    epoch = np.asarray(task.epoch)
    ordered_indices = np.concatenate(
        [
            np.flatnonzero(epoch == name)
            for name in ("cue", "delay", "sensory", "response")
        ]
    )
    if ordered_indices.size != epoch.size or np.unique(ordered_indices).size != epoch.size:
        raise RuntimeError("hidden-context epochs cannot be reordered losslessly")
    reordered_inputs = np.asarray(task.inputs[:, ordered_indices], dtype=np.float64)
    delayed_mask = np.asarray(epoch[ordered_indices]) == "delay"
    reordered_inputs[:, delayed_mask, :2] = 0.0
    reordered_targets = np.asarray(task.targets[:, ordered_indices], dtype=np.float64)
    reordered_loss = np.asarray(task.loss_mask[:, ordered_indices], dtype=bool)
    reordered_loss[:, delayed_mask] = False
    delayed_task = TaskLearningBatch(
        inputs=reordered_inputs,
        targets=reordered_targets,
        loss_mask=reordered_loss,
        trial_ids=task.trial_ids,
        episode_ids=task.episode_ids,
        episode_trial_indices=task.episode_trial_indices,
        episode_start=task.episode_start,
        epoch=epoch[ordered_indices],
        time_ms=task.time_ms,
        config=task.config,
    )
    return replace(dataset, task=delayed_task)


def _task_dataset(
    config: dict[str, Any], seed: int, task_variant: str
) -> tuple[HiddenContextDataset, str]:
    task = exp19._task_config(config)
    tape = make_hidden_context_random_tape(task, seed=seed)
    dataset = generate_hidden_context(task, seed=seed, random_tape=tape)
    if task_variant == "current":
        return dataset, tape.fingerprint
    if task_variant == "delayed":
        return _delayed_task(dataset), tape.fingerprint
    raise ValueError(f"unknown task variant: {task_variant}")


def _network(
    config: dict[str, Any], task: HiddenContextConfig, seed: int
) -> EIRateNetwork:
    """Build the paired high-rank receiver with sensory and cue channels."""

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
        n_inputs=4,
        seed=derive_seed(seed, "exp19", "network-init"),
        **options,
    )


def _fit_gate(
    splits: HiddenContextSplits, config: dict[str, Any], seed: int
) -> tuple[MDRecurrentBeliefGate, dict[str, np.ndarray], str]:
    gate = MDRecurrentBeliefGate(
        seed=derive_seed(seed, "exp23", "md-gate"),
        **dict(config["md_gate"]),
    ).fit(splits.train.gate)
    predictions = {
        name: gate.predict(getattr(splits, name).gate).context_probability
        for name in ("train", "dev", "test")
    }
    audit_predictions = [
        gate.predict(getattr(splits, name).gate) for name in ("train", "dev", "test")
    ]
    if any(item.test_accessed_true_context for item in audit_predictions):
        raise RuntimeError("Exp23 belief gate accessed hidden context")
    checkpoint = _fingerprint(
        "exp23-cue-only-md-gate-v1",
        audit_predictions[0].parameters,
        audit_predictions[0].fit_trial_ids,
        splits.train.gate.fingerprint,
    )
    return gate, predictions, checkpoint


def _neutral_readout(
    network: EIRateNetwork,
    splits: HiddenContextSplits,
    train_posterior: np.ndarray,
    config: dict[str, Any],
    task_variant: str,
) -> ReceiverReadout:
    rates, _, _ = _simulate_axis_batch(
        network,
        splits.train,
        train_posterior,
        np.zeros(network.n_units, dtype=np.float64),
        config,
        task_variant,
    )
    features = _feature_matrix(rates, splits.train.task.epoch)
    targets = exp19._task_targets(splits.train.task)
    mean = np.mean(features, axis=0)
    empirical_scale = np.std(features, axis=0)
    scale = np.where(empirical_scale > 0.0, empirical_scale, 1.0)
    alpha = float(config["readout_alpha"])
    model = Ridge(alpha=alpha, fit_intercept=True).fit(
        (features - mean) / scale,
        targets,
    )
    weights = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    train_data_id = _fingerprint(
        "exp23-neutral-readout-train-data-v2",
        splits.train.task.fingerprint,
        features,
        task_variant,
    )
    checkpoint = _fingerprint(
        "exp23-neutral-current-gain-ridge-readout-v2",
        mean,
        scale,
        weights,
        float(model.intercept_),
        alpha,
        train_data_id,
    )
    return ReceiverReadout(
        mean=mean,
        scale=scale,
        weights=weights,
        intercept=float(model.intercept_),
        alpha=alpha,
        train_data_id=train_data_id,
        checkpoint_id=checkpoint,
    )


def _episode_indices(dataset: HiddenContextDataset) -> list[np.ndarray]:
    episode_ids = np.asarray(dataset.task.episode_ids, dtype=int)
    return [
        np.flatnonzero(episode_ids == episode)
        for episode in dict.fromkeys(episode_ids.tolist())
    ]


def _coarse_gains(
    dataset: HiddenContextDataset,
    posterior: np.ndarray,
    axis: np.ndarray,
    *,
    gain_min: float,
    gain_max: float,
    task_variant: str,
) -> np.ndarray:
    signed = _control_schedule(dataset, posterior, task_variant)
    raw = (
        1.0
        + signed[:, :, None]
        * np.asarray(axis, dtype=np.float64)[None, None, :]
    )
    return np.clip(raw, float(gain_min), float(gain_max))


def _control_schedule(
    dataset: HiddenContextDataset,
    posterior: np.ndarray,
    task_variant: str,
) -> np.ndarray:
    """Return the time-local scalar controller signal used by the receiver."""

    epoch = np.asarray(dataset.task.epoch)
    if task_variant == "current":
        active = epoch != "cue"
    elif task_variant == "delayed":
        # The controller may establish/maintain a state from the cue through
        # the blank delay, but it cannot directly gate the later sensory input.
        active = np.isin(epoch, ("cue", "delay"))
    else:
        raise ValueError(f"unknown task variant: {task_variant}")
    signed = 2.0 * np.asarray(posterior, dtype=np.float64) - 1.0
    return signed[:, None] * active[None, :]


def _routed_inputs(
    dataset: HiddenContextDataset,
    posterior: np.ndarray,
    task_variant: str,
) -> np.ndarray:
    inputs = np.asarray(dataset.task.inputs[:, :, :4], dtype=np.float64).copy()
    if task_variant == "current":
        scales = np.column_stack((1.0 - posterior, posterior))
        inputs[:, :, :2] *= scales[:, None, :]
        # Preserve the Exp19/22 current-task comparison: the receiver does not
        # receive an extra cue channel in this branch.
        inputs[:, :, 2:] = 0.0
    elif task_variant == "delayed":
        # Both sensory streams arrive with identical physical routing.  Only
        # the earlier cue channels and the learned controller can distinguish
        # context, eliminating the sensory-time routing shortcut.
        pass
    else:
        raise ValueError(f"unknown task variant: {task_variant}")
    return inputs


def _feature_matrix(rates: np.ndarray, epoch: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.mean(rates[:, np.asarray(epoch) == name], axis=1)
            for name in ("sensory", "delay", "response")
        ],
        axis=1,
    )


def _activation(
    state: np.ndarray, activation_name: str
) -> np.ndarray:
    activated = np.tanh(state)
    if activation_name == "rectified_tanh":
        return np.maximum(activated, 0.0)
    if activation_name == "tanh":
        return activated
    raise ValueError(f"unsupported activation: {activation_name}")


def _simulate_axis_batch(
    network: EIRateNetwork,
    dataset: HiddenContextDataset,
    posterior: np.ndarray,
    axis: np.ndarray,
    config: dict[str, Any],
    task_variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the drive-gain controller with trial-reset frozen E/I dynamics."""

    controller = dict(config["controller"])
    inputs = _routed_inputs(dataset, posterior, task_variant)
    drives = inputs @ network.input_weights.T
    gains = _coarse_gains(
        dataset,
        posterior,
        axis,
        gain_min=float(controller["gain_min"]),
        gain_max=float(controller["gain_max"]),
        task_variant=task_variant,
    )
    n_trials, n_steps = inputs.shape[:2]
    states = np.zeros((n_trials, n_steps, network.n_units), dtype=np.float64)
    rates = np.zeros_like(states)
    x = np.zeros((n_trials, network.n_units), dtype=np.float64)
    rate = np.zeros_like(x)
    step_fraction = network.dt / network.time_constants
    for time in range(n_steps):
        gain = gains[:, time]
        for _ in range(int(config["integration_substeps"])):
            total_drive = rate @ network.recurrent_weights.T + drives[:, time]
            x = x + step_fraction[None, :] * (
                -x + gain * total_drive
            )
            rate = _activation(x, network.activation_name)
        states[:, time] = x
        rates[:, time] = rate
    return rates, states, gains


def _forward_no_sensitivity(
    network: EIRateNetwork,
    dataset: HiddenContextDataset,
    posterior: np.ndarray,
    axis: np.ndarray,
    readout: ReceiverReadout,
    config: dict[str, Any],
    task_variant: str,
) -> AxisForward:
    controller = dict(config["controller"])
    rates, states, gains = _simulate_axis_batch(
        network,
        dataset,
        posterior,
        axis,
        config,
        task_variant,
    )
    features = _feature_matrix(rates, dataset.task.epoch)
    scores = readout.scores(features)
    targets = exp19._task_targets(dataset.task)
    task_loss = float(np.mean(np.logaddexp(0.0, -targets * scores)))
    rate_penalty = float(controller["rate_penalty"]) * float(np.mean(rates**2))
    axis_penalty = 0.5 * float(controller["axis_l2"]) * float(axis @ axis)
    return AxisForward(
        features=features,
        scores=scores,
        targets=targets,
        loss=task_loss + rate_penalty + axis_penalty,
        task_loss=task_loss,
        rate_penalty=rate_penalty,
        axis_penalty=axis_penalty,
        rates=rates,
        states=states,
        gains=gains,
        saturation_fraction=float(
            np.mean(
                np.isclose(gains, float(controller["gain_min"]))
                | np.isclose(gains, float(controller["gain_max"]))
            )
        ),
        max_firing_rate=float(np.max(rates)),
        mean_firing_rate=float(np.mean(rates)),
        gradient=None,
        exact_gradient=None,
    )


def _expanded_trial(
    network: EIRateNetwork,
    dataset: HiddenContextDataset,
    posterior: np.ndarray,
    trial: int,
    substeps: int,
    task_variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    routed = _routed_inputs(dataset, posterior, task_variant)[trial]
    coarse_drive = routed @ network.input_weights.T
    drives = np.repeat(coarse_drive, substeps, axis=0)
    schedule = _control_schedule(dataset, posterior, task_variant)[trial]
    beliefs = np.repeat(schedule, substeps)
    labels = np.full(drives.shape[0], "substep", dtype="U32")
    labels[substeps - 1 :: substeps] = np.asarray(dataset.task.epoch, dtype="U32")
    return drives, beliefs, labels


def _forward_with_gradient(
    network: EIRateNetwork,
    dataset: HiddenContextDataset,
    posterior: np.ndarray,
    axis: np.ndarray,
    readout: ReceiverReadout,
    config: dict[str, Any],
    *,
    mode: GradientMode,
    task_variant: str,
) -> AxisForward:
    controller = dict(config["controller"])
    substeps = int(config["integration_substeps"])
    n_trials = dataset.task.inputs.shape[0]
    n_units = network.n_units
    coefficients = (readout.weights / readout.scale).reshape(3, n_units)
    targets = exp19._task_targets(dataset.task)
    features: list[np.ndarray] = []
    rates_all: list[np.ndarray] = []
    states_all: list[np.ndarray] = []
    gains_all: list[np.ndarray] = []
    exact_score_gradients: list[np.ndarray] = []
    local_score_gradients: list[np.ndarray] = []
    exact_activity_gradient = np.zeros(n_units, dtype=np.float64)
    local_activity_gradient = np.zeros(n_units, dtype=np.float64)
    local_eligibilities: list[np.ndarray] = []
    time_weights: list[np.ndarray] = []
    activity_time_signals: list[np.ndarray] = []

    for trial in range(n_trials):
        drives, beliefs, labels = _expanded_trial(
            network,
            dataset,
            posterior,
            trial,
            substeps,
            task_variant,
        )
        trajectory = simulate_frozen_gain_axis_trajectory(
            drives,
            beliefs,
            axis,
            network.recurrent_weights,
            network.dt / network.time_constants,
            gain_min=float(controller["gain_min"]),
            gain_max=float(controller["gain_max"]),
            activation=network.activation_name,
            gain_application="drive",
        )
        coarse = np.arange(substeps, drives.shape[0] + 1, substeps)
        coarse_rates = np.asarray(trajectory.rates[coarse], dtype=np.float64)
        coarse_states = np.asarray(trajectory.states[coarse], dtype=np.float64)
        features.append(_feature_matrix(coarse_rates[None], dataset.task.epoch)[0])
        rates_all.append(coarse_rates)
        states_all.append(coarse_states)
        gains_all.append(np.asarray(trajectory.gains[substeps - 1 :: substeps]))
        exact_readout = epoch_mean_readout_gradient(
            trajectory.rate_sensitivities,
            labels,
            coefficients,
        )
        exact_score_gradients.append(np.asarray(exact_readout.score_gradient))
        time_weights.append(np.asarray(exact_readout.readout_time_weights))
        activity_signal = np.zeros_like(trajectory.rates[1:])
        activity_signal[coarse - 1] = (
            2.0
            * float(controller["rate_penalty"])
            * coarse_rates
            / (n_trials * coarse_rates.shape[0] * n_units)
        )
        activity_time_signals.append(activity_signal)
        exact_activity_gradient += (
            np.einsum(
                "tu,tup->p",
                activity_signal,
                trajectory.rate_sensitivities[1:],
                optimize=True,
            )
        )
        if mode == "local":
            eligibility = block_local_gain_axis_eprop_sensitivities(
                trajectory.local_jacobian_blocks,
                trajectory.state_axis_direct_derivatives,
                trajectory.rate_axis_direct_derivatives,
            )
            local_eligibilities.append(np.asarray(eligibility.eligibilities))
            local_readout = epoch_mean_readout_gradient(
                eligibility.eligibilities[:, n_units:, :],
                labels,
                coefficients,
            )
            local_score_gradients.append(np.asarray(local_readout.score_gradient))
            local_activity_gradient += (
                np.einsum(
                    "tu,tup->p",
                    activity_signal,
                    eligibility.eligibilities[1:, n_units:, :],
                    optimize=True,
                )
            )

    feature_matrix = np.stack(features)
    scores = readout.scores(feature_matrix)
    task_loss = float(np.mean(np.logaddexp(0.0, -targets * scores)))
    rates = np.stack(rates_all)
    states = np.stack(states_all)
    gains = np.stack(gains_all)
    rate_penalty = float(controller["rate_penalty"]) * float(np.mean(rates**2))
    axis_penalty = 0.5 * float(controller["axis_l2"]) * float(axis @ axis)
    margins = targets * scores
    score_error = -targets * np.exp(-np.logaddexp(0.0, margins))
    exact_gradient = np.mean(
        score_error[:, None] * np.stack(exact_score_gradients), axis=0
    )
    exact_gradient += exact_activity_gradient + float(controller["axis_l2"]) * axis

    if mode == "exact":
        gradient = exact_gradient
    else:
        # Recontract the explicitly local learning signal with the forward
        # diagonal eligibility receipt.  The readout feedback and task error
        # are neuron-local; belief sign is already in dF/da.
        gradient = np.zeros(n_units, dtype=np.float64)
        for trial in range(n_trials):
            signal = np.zeros(
                (local_eligibilities[trial].shape[0], 2 * n_units),
                dtype=np.float64,
            )
            signal[1:, n_units:] = (
                score_error[trial] * time_weights[trial] / n_trials
                + activity_time_signals[trial]
            )
            gradient += diagonal_eprop_gradient(
                local_eligibilities[trial], signal
            ).gradient
        gradient += float(controller["axis_l2"]) * axis
        if not np.allclose(
            gradient,
            np.mean(
                score_error[:, None] * np.stack(local_score_gradients), axis=0
            )
            + local_activity_gradient
            + float(controller["axis_l2"]) * axis,
            atol=1e-10,
            rtol=1e-8,
        ):
            raise RuntimeError("local e-prop learning-signal contraction mismatch")

    return AxisForward(
        features=feature_matrix,
        scores=scores,
        targets=targets,
        loss=task_loss + rate_penalty + axis_penalty,
        task_loss=task_loss,
        rate_penalty=rate_penalty,
        axis_penalty=axis_penalty,
        rates=rates,
        states=states,
        gains=gains,
        saturation_fraction=float(
            np.mean(
                np.isclose(gains, float(controller["gain_min"]))
                | np.isclose(gains, float(controller["gain_max"]))
            )
        ),
        max_firing_rate=float(np.max(rates)),
        mean_firing_rate=float(np.mean(rates)),
        gradient=np.asarray(gradient, dtype=np.float64),
        exact_gradient=np.asarray(exact_gradient, dtype=np.float64),
    )


def _bptt_gradient(
    network: EIRateNetwork,
    dataset: HiddenContextDataset,
    posterior: np.ndarray,
    axis: np.ndarray,
    readout: ReceiverReadout,
    config: dict[str, Any],
    task_variant: str,
) -> tuple[np.ndarray, float]:
    """Differentiate only the controller axis with Torch reverse mode."""

    controller = dict(config["controller"])
    dtype = torch.float64
    device = torch.device("cpu")
    parameter = torch.tensor(axis, dtype=dtype, device=device, requires_grad=True)
    weights = torch.tensor(network.recurrent_weights, dtype=dtype, device=device)
    input_weights = torch.tensor(network.input_weights, dtype=dtype, device=device)
    dt_over_tau = torch.tensor(
        network.dt / network.time_constants, dtype=dtype, device=device
    )
    inputs = torch.tensor(
        _routed_inputs(dataset, posterior, task_variant),
        dtype=dtype,
        device=device,
    )
    schedule = torch.tensor(
        _control_schedule(dataset, posterior, task_variant),
        dtype=dtype,
        device=device,
    )
    targets = torch.tensor(
        exp19._task_targets(dataset.task), dtype=dtype, device=device
    )
    x = torch.zeros((inputs.shape[0], network.n_units), dtype=dtype, device=device)
    rates = torch.zeros_like(x)
    coarse_rates: list[torch.Tensor] = []
    substeps = int(config["integration_substeps"])
    for time in range(inputs.shape[1]):
        gain = torch.clamp(
            1.0
            + schedule[:, time, None] * parameter[None, :],
            min=float(controller["gain_min"]),
            max=float(controller["gain_max"]),
        )
        drive = inputs[:, time] @ input_weights.T
        for _ in range(substeps):
            total_drive = rates @ weights.T + drive
            x = x + dt_over_tau[None, :] * (-x + gain * total_drive)
            activated = torch.tanh(x)
            rates = (
                torch.clamp(activated, min=0.0)
                if network.activation_name == "rectified_tanh"
                else activated
            )
        coarse_rates.append(rates)
    rate_tensor = torch.stack(coarse_rates, dim=1)
    feature_blocks = [
        rate_tensor[
            :, torch.tensor(np.asarray(dataset.task.epoch) == name, device=device)
        ].mean(dim=1)
        for name in ("sensory", "delay", "response")
    ]
    features = torch.cat(feature_blocks, dim=1)
    mean = torch.tensor(readout.mean, dtype=dtype, device=device)
    scale = torch.tensor(readout.scale, dtype=dtype, device=device)
    readout_weights = torch.tensor(readout.weights, dtype=dtype, device=device)
    scores = ((features - mean) / scale) @ readout_weights + float(readout.intercept)
    loss = torch.nn.functional.softplus(-targets * scores).mean()
    loss = (
        loss
        + float(controller["rate_penalty"]) * torch.mean(rate_tensor**2)
        + 0.5 * float(controller["axis_l2"]) * torch.sum(parameter**2)
    )
    loss.backward()
    if parameter.grad is None:
        raise RuntimeError("axis-only BPTT did not produce a gradient")
    return parameter.grad.detach().cpu().numpy(), float(loss.detach().cpu())


def _clip_l2(update: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(update))
    if norm == 0.0 or norm <= maximum:
        return np.asarray(update, dtype=np.float64)
    return np.asarray(update, dtype=np.float64) * (maximum / norm)


def _apply_update(
    network: EIRateNetwork,
    dataset: HiddenContextDataset,
    posterior: np.ndarray,
    readout: ReceiverReadout,
    config: dict[str, Any],
    axis: np.ndarray,
    raw_update: np.ndarray,
    task_variant: str,
) -> UpdateReceipt:
    controller = dict(config["controller"])
    clipped = _clip_l2(raw_update, float(controller["max_update_l2"]))
    signed = np.concatenate(
        (
            np.array([0.0]),
            2.0 * np.asarray(posterior, dtype=np.float64) - 1.0,
        )
    )
    bounded = apply_gain_bounds(
        axis,
        clipped,
        signed,
        gain_min=float(controller["gain_min"]),
        gain_max=float(controller["gain_max"]),
    )
    current = _forward_no_sensitivity(
        network,
        dataset,
        posterior,
        axis,
        readout,
        config,
        task_variant,
    )
    frozen = _forward_no_sensitivity(
        network,
        dataset,
        posterior,
        np.zeros_like(axis),
        readout,
        config,
        task_variant,
    )
    applied = np.asarray(bounded.applied_update, dtype=np.float64)
    functional_scale = 1.0
    maximum_displacement = float(controller["max_state_displacement"])
    candidate = current
    displacement = functional_state_displacement(
        current.states,
        frozen.states,
        exclude_initial=False,
    )
    for _ in range(int(controller["max_backtracks"]) + 1):
        candidate_axis = axis + functional_scale * applied
        candidate = _forward_no_sensitivity(
            network,
            dataset,
            posterior,
            candidate_axis,
            readout,
            config,
            task_variant,
        )
        displacement = functional_state_displacement(
            candidate.states,
            frozen.states,
            exclude_initial=False,
        )
        if displacement <= maximum_displacement:
            break
        functional_scale *= 0.5
    else:
        functional_scale = 0.0
        candidate_axis = axis.copy()
        candidate = current
        displacement = functional_state_displacement(
            current.states,
            frozen.states,
            exclude_initial=False,
        )
    final_update = functional_scale * applied
    final_axis = axis + final_update
    return UpdateReceipt(
        axis=np.asarray(final_axis, dtype=np.float64),
        raw_update=np.asarray(raw_update, dtype=np.float64),
        applied_update=np.asarray(final_update, dtype=np.float64),
        gain_scale=float(bounded.scale_factor),
        functional_scale=float(functional_scale),
        state_displacement=displacement,
        loss_before=current.loss,
        loss_after=candidate.loss,
        improved=bool(candidate.loss < current.loss),
    )


def _frozen_trajectory_block_local_updates(
    network: EIRateNetwork,
    splits: HiddenContextSplits,
    posterior: np.ndarray,
    readout: ReceiverReadout,
    config: dict[str, Any],
    task_variant: str,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    """Precompute Exp23 block-local proposals on zero-axis trajectories.

    This deliberately does not reuse the Exp22 eligibility-tape proposal.
    ``current_off_policy`` is retained only as a results-schema compatibility
    key; the explicit method receipt below is the normative description.
    """

    options = dict(config["off_policy"])
    budget_norm = str(options.get("budget_norm", "l1")).lower()
    if budget_norm not in {"l1", "l2"}:
        raise ValueError("off_policy.budget_norm must be 'l1' or 'l2'")
    budget_tolerance = float(options.get("budget_tolerance", 1e-9))
    if not np.isfinite(budget_tolerance) or budget_tolerance < 0.0:
        raise ValueError("off_policy.budget_tolerance must be non-negative")
    updates: dict[int, np.ndarray] = {}
    for indices in _episode_indices(splits.dev):
        episode = splits.dev.subset(indices)
        result = _forward_with_gradient(
            network,
            episode,
            posterior[indices],
            np.zeros(network.n_units, dtype=np.float64),
            readout,
            config,
            mode="local",
            task_variant=task_variant,
        )
        if result.gradient is None:
            raise RuntimeError("off-policy frozen trajectory returned no gradient")
        updates[int(episode.task.episode_ids[0])] = (
            -float(options["learning_rate"]) * result.gradient
        )

    def proposal_norm(value: np.ndarray) -> float:
        if budget_norm == "l1":
            return float(np.sum(np.abs(value)))
        return float(np.linalg.norm(value))
    raw_total = sum(proposal_norm(value) for value in updates.values())
    budget = float(options["total_budget"])
    if not np.isfinite(budget) or budget < 0.0:
        raise ValueError("off_policy.total_budget must be non-negative")
    scale = min(1.0, budget / raw_total) if raw_total > 0.0 else 0.0
    updates = {key: scale * value for key, value in updates.items()}
    applied_total = sum(proposal_norm(value) for value in updates.values())
    legacy_no_op_fields = sorted(
        set(options).intersection({"tau_eligibility_steps", "error_clip"})
    )
    return updates, {
        "off_policy_method": FROZEN_TRAJECTORY_BLOCK_LOCAL_METHOD,
        "off_policy_condition_key_is_legacy_alias": True,
        "off_policy_legacy_condition_key": "current_off_policy",
        "off_policy_exp22_proposal_reused": False,
        "off_policy_eligibility_approximation": (
            "per_unit_state_rate_block_jacobian"
        ),
        "off_policy_proposal_axis": "zero_population_gain_axis",
        "off_policy_proposal_trajectory_recomputed_after_each_update": False,
        "off_policy_proposal_budget_norm": budget_norm,
        "off_policy_proposal_budget_total": budget,
        "off_policy_proposal_budget_raw": raw_total,
        "off_policy_proposal_budget_applied": applied_total,
        "off_policy_proposal_budget_scale": scale,
        "off_policy_proposal_budget_attained": bool(
            abs(applied_total - budget) <= budget_tolerance
        ),
        "off_policy_proposal_budget_cap_respected": bool(
            applied_total <= budget + budget_tolerance
        ),
        "off_policy_legacy_no_op_config_fields": legacy_no_op_fields,
    }


def _balanced_accuracy(scores: np.ndarray, targets: np.ndarray) -> float:
    return exp19._balanced_accuracy(
        np.where(scores >= 0.0, 1, -1), targets.astype(int)
    )


def _switch_cost(
    scores: np.ndarray, dataset: HiddenContextDataset, *, post_switch_trials: int
) -> float:
    predicted = np.where(scores >= 0.0, 1, -1)
    target = exp19._task_targets(dataset.task).astype(int)
    switch = np.asarray(dataset.truth.switch_mask, dtype=bool)
    episode = np.asarray(dataset.truth.episode_ids, dtype=int)
    near = np.zeros(target.size, dtype=bool)
    for index in np.flatnonzero(switch):
        stop = min(target.size, index + int(post_switch_trials))
        selected = np.arange(index, stop)
        near[selected[episode[selected] == episode[index]]] = True
    if not np.any(near) or np.all(near):
        return float("nan")
    return float(np.mean(predicted[~near] == target[~near]) - np.mean(predicted[near] == target[near]))


def _trials_to_criterion(
    episode_ba: list[float],
    episode_trials: list[int],
    *,
    criterion: float,
    window: int,
) -> int | None:
    for stop in range(window, len(episode_ba) + 1):
        if float(np.mean(episode_ba[stop - window : stop])) >= criterion:
            return int(np.sum(episode_trials[:stop]))
    return None


def _jacobian_margin(
    network: EIRateNetwork, mean_x: np.ndarray, mean_gain: np.ndarray
) -> float:
    activated = np.tanh(mean_x)
    derivative = 1.0 - activated * activated
    if network.activation_name == "rectified_tanh":
        derivative = np.where(activated > 0.0, derivative, 0.0)
    jacobian = (
        -np.eye(network.n_units)
        + np.diag(mean_gain)
        @ network.recurrent_weights
        @ np.diag(derivative)
    ) / network.time_constants[:, None]
    return float(-np.max(np.real(np.linalg.eigvals(jacobian))))


def _bptt_selection_better(
    balanced_accuracy: float,
    task_loss: float,
    *,
    best_balanced_accuracy: float,
    best_task_loss: float,
) -> bool:
    """Use dev behavior first and dev task loss only as a deterministic tie-break."""

    tolerance = 1e-12
    return bool(
        balanced_accuracy > best_balanced_accuracy + tolerance
        or (
            abs(balanced_accuracy - best_balanced_accuracy) <= tolerance
            and task_loss < best_task_loss - tolerance
        )
    )


def _train_bptt_condition(
    *,
    network: EIRateNetwork,
    splits: HiddenContextSplits,
    predictions: dict[str, np.ndarray],
    readout: ReceiverReadout,
    config: dict[str, Any],
    task_variant: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Tune the axis-only BPTT upper bound on the development split only."""

    options = dict(config["bptt_optimizer"])
    learning_rates = [float(value) for value in options["learning_rates"]]
    if not learning_rates or any(value <= 0.0 for value in learning_rates):
        raise ValueError("bptt_optimizer.learning_rates must be positive")
    max_steps = int(options["max_steps"])
    patience = int(options["patience"])
    if max_steps < 1 or patience < 1:
        raise ValueError("BPTT max_steps and patience must be positive")
    beta1 = float(options.get("beta1", 0.9))
    beta2 = float(options.get("beta2", 0.999))
    epsilon = float(options.get("epsilon", 1e-8))
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0 and epsilon > 0.0):
        raise ValueError("invalid BPTT Adam hyperparameters")

    dataset = splits.dev
    posterior = predictions["dev"]
    zero_axis = np.zeros(network.n_units, dtype=np.float64)
    bptt_gradient, _ = _bptt_gradient(
        network,
        dataset,
        posterior,
        zero_axis,
        readout,
        config,
        task_variant,
    )
    exact = _forward_with_gradient(
        network,
        dataset,
        posterior,
        zero_axis,
        readout,
        config,
        mode="exact",
        task_variant=task_variant,
    )
    if exact.exact_gradient is None:
        raise RuntimeError("BPTT exact-gradient audit is unavailable")
    audit = directional_cosine(-bptt_gradient, -exact.exact_gradient)

    recurrent_checkpoint = _capture_recurrent_weight_checkpoint(network)
    candidates: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    total_candidate_updates = 0
    for learning_rate in learning_rates:
        axis = zero_axis.copy()
        first_moment = np.zeros_like(axis)
        second_moment = np.zeros_like(axis)
        start = _forward_no_sensitivity(
            network,
            dataset,
            posterior,
            axis,
            readout,
            config,
            task_variant,
        )
        best_axis = axis.copy()
        best_balanced_accuracy = _balanced_accuracy(start.scores, start.targets)
        best_task_loss = start.task_loss
        best_step = 0
        history = [best_balanced_accuracy]
        loss_history = [best_task_loss]
        receipts: list[UpdateReceipt] = []
        stale_steps = 0

        for step in range(1, max_steps + 1):
            gradient, _ = _bptt_gradient(
                network,
                dataset,
                posterior,
                axis,
                readout,
                config,
                task_variant,
            )
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
            corrected_first = first_moment / (1.0 - beta1**step)
            corrected_second = second_moment / (1.0 - beta2**step)
            raw_update = (
                -learning_rate
                * corrected_first
                / (np.sqrt(corrected_second) + epsilon)
            )
            receipt = _apply_update(
                network,
                dataset,
                posterior,
                readout,
                config,
                axis,
                raw_update,
                task_variant,
            )
            receipts.append(receipt)
            axis = receipt.axis
            current = _forward_no_sensitivity(
                network,
                dataset,
                posterior,
                axis,
                readout,
                config,
                task_variant,
            )
            balanced_accuracy = _balanced_accuracy(current.scores, current.targets)
            history.append(balanced_accuracy)
            loss_history.append(current.task_loss)
            if _bptt_selection_better(
                balanced_accuracy,
                current.task_loss,
                best_balanced_accuracy=best_balanced_accuracy,
                best_task_loss=best_task_loss,
            ):
                best_axis = axis.copy()
                best_balanced_accuracy = balanced_accuracy
                best_task_loss = current.task_loss
                best_step = step
                stale_steps = 0
            else:
                stale_steps += 1
            if stale_steps >= patience:
                break

        candidate = {
            "learning_rate": learning_rate,
            "axis": best_axis,
            "best_step": best_step,
            "best_balanced_accuracy": best_balanced_accuracy,
            "best_task_loss": best_task_loss,
            "history": history,
            "loss_history": loss_history,
            "receipts": receipts,
            "steps_run": len(receipts),
        }
        candidates.append(candidate)
        total_candidate_updates += len(receipts)
        if selected is None or _bptt_selection_better(
            best_balanced_accuracy,
            best_task_loss,
            best_balanced_accuracy=float(selected["best_balanced_accuracy"]),
            best_task_loss=float(selected["best_task_loss"]),
        ):
            selected = candidate

    if selected is None:
        raise RuntimeError("BPTT candidate search produced no result")
    final_recurrent_checkpoint = _assert_recurrent_weight_checkpoint_unchanged(
        network,
        recurrent_checkpoint,
        phase="BPTT tuning",
    )

    selected_best_step = int(selected["best_step"])
    selected_receipts = list(selected["receipts"])[:selected_best_step]
    selected_history = [
        float(value)
        for value in list(selected["history"])[: selected_best_step + 1]
    ]
    selected_loss_history = [
        float(value)
        for value in list(selected["loss_history"])[: selected_best_step + 1]
    ]
    selected_steps = selected_best_step
    trials_per_step = int(dataset.task.trial_ids.size)
    candidate_summaries = [
        {
            "learning_rate": float(candidate["learning_rate"]),
            "best_step": int(candidate["best_step"]),
            "steps_run": int(candidate["steps_run"]),
            "best_dev_balanced_accuracy": float(
                candidate["best_balanced_accuracy"]
            ),
            "best_dev_task_loss": float(candidate["best_task_loss"]),
        }
        for candidate in candidates
    ]
    return np.asarray(selected["axis"], dtype=np.float64), {
        "update_events": selected_steps,
        "episode_balanced_accuracy_history": selected_history,
        "bptt_selected_dev_task_loss_history": selected_loss_history,
        "trials_to_criterion": _trials_to_criterion(
            selected_history,
            [trials_per_step] * len(selected_history),
            criterion=float(config["controller"]["criterion"]),
            window=int(config["controller"]["criterion_window"]),
        ),
        "median_update_cosine_to_exact": audit.cosine if audit.eligible else None,
        "eligible_update_cosine_count": int(audit.eligible),
        "probability_update_decreases_loss": (
            float(np.mean([receipt.improved for receipt in selected_receipts]))
            if selected_receipts
            else None
        ),
        "plasticity_l1_path": float(
            sum(np.sum(np.abs(receipt.applied_update)) for receipt in selected_receipts)
        ),
        "plasticity_l2_path": float(
            sum(np.linalg.norm(receipt.applied_update) for receipt in selected_receipts)
        ),
        "maximum_training_state_displacement": float(
            max(
                (receipt.state_displacement for receipt in selected_receipts),
                default=0.0,
            )
        ),
        "bptt_exact_gradient_cosine": audit.cosine if audit.eligible else None,
        "bptt_optimizer": "deterministic_adam_axis_only",
        "bptt_selection_scope": "full_dev_split_only",
        "bptt_selection_rule": "dev_balanced_accuracy_then_task_loss",
        "bptt_step_zero_eligible": True,
        "bptt_selected_learning_rate": float(selected["learning_rate"]),
        "bptt_selected_step": selected_best_step,
        "bptt_selected_dev_balanced_accuracy": float(
            selected["best_balanced_accuracy"]
        ),
        "bptt_selected_dev_task_loss": float(selected["best_task_loss"]),
        "bptt_candidate_summaries": candidate_summaries,
        "bptt_candidate_update_events_total": total_candidate_updates,
        "recurrent_weights_bitwise_frozen": True,
        "recurrent_weights_initial_id": recurrent_checkpoint.fingerprint,
        "recurrent_weights_final_id": final_recurrent_checkpoint.fingerprint,
        "recurrent_weights_audit": "independent_copy_and_sha256",
    }


def _train_condition(
    condition: str,
    *,
    network: EIRateNetwork,
    splits: HiddenContextSplits,
    predictions: dict[str, np.ndarray],
    readout: ReceiverReadout,
    config: dict[str, Any],
    seed: int,
    task_variant: str,
) -> tuple[np.ndarray, dict[str, object]]:
    if condition == "bptt_axis_only":
        return _train_bptt_condition(
            network=network,
            splits=splits,
            predictions=predictions,
            readout=readout,
            config=config,
            task_variant=task_variant,
        )

    controller = dict(config["controller"])
    recurrent_checkpoint = _capture_recurrent_weight_checkpoint(network)
    axis = np.zeros(network.n_units, dtype=np.float64)
    episode_groups = _episode_indices(splits.dev)
    episode_ba: list[float] = []
    episode_trials: list[int] = []
    cosines: list[float] = []
    improvements: list[bool] = []
    update_l1 = 0.0
    update_l2_path = 0.0
    maximum_displacement = 0.0
    update_events = 0
    bptt_exact_cosines: list[float] = []
    off_policy, off_policy_audit = (
        _frozen_trajectory_block_local_updates(
            network,
            splits,
            predictions["dev"],
            readout,
            config,
            task_variant,
        )
        if condition == "current_off_policy"
        else ({}, {})
    )
    random_tape = np.random.default_rng(
        derive_seed(seed, "exp23", task_variant, "random-update")
    ).normal(
        size=(
            int(controller["n_passes"]) * len(episode_groups),
            network.n_units,
        )
    )
    random_index = 0

    for _pass in range(int(controller["n_passes"])):
        for indices in episode_groups:
            episode = splits.dev.subset(indices)
            posterior = predictions["dev"][indices]
            current = _forward_no_sensitivity(
                network,
                episode,
                posterior,
                axis,
                readout,
                config,
                task_variant,
            )
            episode_ba.append(_balanced_accuracy(current.scores, current.targets))
            episode_trials.append(int(indices.size))
            if condition == "frozen":
                continue
            exact: AxisForward | None = None
            if condition in {"exact_forward_sensitivity", "local_eprop"}:
                exact = _forward_with_gradient(
                    network,
                    episode,
                    posterior,
                    axis,
                    readout,
                    config,
                    mode="local" if condition == "local_eprop" else "exact",
                    task_variant=task_variant,
                )
                if exact.gradient is None or exact.exact_gradient is None:
                    raise RuntimeError("Exp23 gradient condition returned no gradient")
                gradient = exact.gradient
                reference_update = -float(controller["learning_rate"]) * exact.exact_gradient
            elif condition == "bptt_axis_only":
                gradient, _ = _bptt_gradient(
                    network,
                    episode,
                    posterior,
                    axis,
                    readout,
                    config,
                    task_variant,
                )
                # Direction audit against the exact forward oracle is done on
                # a pre-registered small subset to keep the baseline tractable.
                exact_audit = _forward_with_gradient(
                    network,
                    episode,
                    posterior,
                    axis,
                    readout,
                    config,
                    mode="exact",
                    task_variant=task_variant,
                )
                if exact_audit.exact_gradient is None:
                    raise RuntimeError("exact audit gradient is unavailable")
                audit = directional_cosine(-gradient, -exact_audit.exact_gradient)
                if audit.eligible:
                    bptt_exact_cosines.append(audit.cosine)
                reference_update = -float(controller["learning_rate"]) * exact_audit.exact_gradient
            elif condition == "random_update":
                direction = random_tape[random_index]
                random_index += 1
                norm = float(np.linalg.norm(direction))
                gradient = -direction / norm if norm > 0.0 else np.zeros_like(axis)
                exact_audit = _forward_with_gradient(
                    network,
                    episode,
                    posterior,
                    axis,
                    readout,
                    config,
                    mode="exact",
                    task_variant=task_variant,
                )
                if exact_audit.exact_gradient is None:
                    raise RuntimeError("random update exact audit is unavailable")
                reference_update = -float(controller["learning_rate"]) * exact_audit.exact_gradient
            elif condition == "current_off_policy":
                episode_id = int(episode.task.episode_ids[0])
                raw = (
                    off_policy[episode_id]
                    if _pass == 0
                    else np.zeros(network.n_units, dtype=np.float64)
                )
                gradient = -raw / max(float(controller["learning_rate"]), 1e-12)
                exact_audit = _forward_with_gradient(
                    network,
                    episode,
                    posterior,
                    axis,
                    readout,
                    config,
                    mode="exact",
                    task_variant=task_variant,
                )
                if exact_audit.exact_gradient is None:
                    raise RuntimeError("off-policy exact audit is unavailable")
                reference_update = -float(controller["learning_rate"]) * exact_audit.exact_gradient
            else:
                raise ValueError(f"unknown Exp23 condition: {condition}")

            raw_update = -float(controller["learning_rate"]) * np.asarray(gradient)
            receipt = _apply_update(
                network,
                episode,
                posterior,
                readout,
                config,
                axis,
                raw_update,
                task_variant,
            )
            audit = directional_cosine(receipt.applied_update, reference_update)
            if audit.eligible:
                cosines.append(audit.cosine)
            improvements.append(receipt.improved)
            update_l1 += float(np.sum(np.abs(receipt.applied_update)))
            update_l2_path += float(np.linalg.norm(receipt.applied_update))
            maximum_displacement = max(maximum_displacement, receipt.state_displacement)
            axis = receipt.axis
            update_events += 1

    final_recurrent_checkpoint = _assert_recurrent_weight_checkpoint_unchanged(
        network,
        recurrent_checkpoint,
        phase="axis learning",
    )
    return axis, {
        "update_events": update_events,
        "episode_balanced_accuracy_history": episode_ba,
        "trials_to_criterion": _trials_to_criterion(
            episode_ba,
            episode_trials,
            criterion=float(controller["criterion"]),
            window=int(controller["criterion_window"]),
        ),
        "median_update_cosine_to_exact": (
            float(np.median(cosines)) if cosines else None
        ),
        "eligible_update_cosine_count": len(cosines),
        "probability_update_decreases_loss": (
            float(np.mean(improvements)) if improvements else None
        ),
        "plasticity_l1_path": update_l1,
        "plasticity_l2_path": update_l2_path,
        "maximum_training_state_displacement": maximum_displacement,
        "bptt_exact_gradient_cosine": (
            float(np.median(bptt_exact_cosines)) if bptt_exact_cosines else None
        ),
        "recurrent_weights_bitwise_frozen": True,
        "recurrent_weights_initial_id": recurrent_checkpoint.fingerprint,
        "recurrent_weights_final_id": final_recurrent_checkpoint.fingerprint,
        "recurrent_weights_audit": "independent_copy_and_sha256",
        **off_policy_audit,
    }


def _ood_metrics(
    *,
    network: EIRateNetwork,
    gate: MDRecurrentBeliefGate,
    readout: ReceiverReadout,
    axis: np.ndarray,
    config: dict[str, Any],
    seed: int,
    task_variant: str,
) -> dict[str, object]:
    cells = list(config.get("ood_cells", []))
    scores: list[float] = []
    labels: list[str] = []
    for index, cell in enumerate(cells):
        task_options = dict(config["task"])
        task_options.update(
            cue_reliability=float(cell["cue_reliability"]),
            context_hazard=float(cell["context_hazard"]),
        )
        task = HiddenContextConfig(**task_options)
        ood_seed = derive_seed(seed, "exp23", "ood", task_variant, index)
        tape = make_hidden_context_random_tape(task, seed=ood_seed)
        dataset = generate_hidden_context(task, seed=ood_seed, random_tape=tape)
        if task_variant == "delayed":
            dataset = _delayed_task(dataset)
        posterior = gate.predict(dataset.gate).context_probability
        result = _forward_no_sensitivity(
            network,
            dataset,
            posterior,
            axis,
            readout,
            config,
            task_variant,
        )
        scores.append(_balanced_accuracy(result.scores, result.targets))
        labels.append(
            f"q{float(cell['cue_reliability']):.2f}_h"
            f"{float(cell['context_hazard']):.2f}"
        )
    return {
        "ood_balanced_accuracy_mean": float(np.mean(scores)) if scores else None,
        "ood_balanced_accuracy_by_cell": dict(zip(labels, scores, strict=True)),
        "ood_cells_used_for_tuning": False,
    }


def _match_axis_functional_budget(
    *,
    network: EIRateNetwork,
    dataset: HiddenContextDataset,
    posterior: np.ndarray,
    axis: np.ndarray,
    config: dict[str, Any],
    task_variant: str,
    frozen_condition: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    """Match state displacement and gate common rate/gain envelopes on dev.

    A single scalar axis multiplier can match one equality target.  Exp23
    therefore uses state displacement as that equality and treats rate cost,
    gain displacement, and the physical gain range as pre-registered upper
    envelopes.  The task-level cross-condition audit is applied after every
    non-frozen direction has been evaluated.
    """

    controller = dict(config["controller"])
    target = float(controller["matched_dev_state_displacement"])
    rate_limit = float(controller["matched_dev_rate_displacement_limit"])
    gain_displacement_limit_per_unit = float(
        controller["matched_dev_gain_displacement_limit_per_unit"]
    )
    gain_displacement_limit = (
        gain_displacement_limit_per_unit * network.n_units
    )
    gain_min = float(controller["gain_min"])
    gain_max = float(controller["gain_max"])

    def receipt(
        *,
        achieved: float,
        scale: float,
        maximum_scale: float,
        rate_displacement: float,
        gain_displacement: float,
        gain_min_observed: float,
        gain_max_observed: float,
        relative_error: float,
        state_converged: bool,
        evaluations: int,
    ) -> dict[str, object]:
        tolerance = float(
            controller.get("functional_match_absolute_tolerance", 1e-8)
        )
        state_valid = bool(state_converged)
        rate_valid = bool(rate_displacement <= rate_limit + tolerance)
        gain_range_valid = bool(
            gain_min_observed >= gain_min - tolerance
            and gain_max_observed <= gain_max + tolerance
        )
        gain_displacement_valid = bool(
            gain_displacement <= gain_displacement_limit + tolerance
        )
        gain_valid = gain_range_valid and gain_displacement_valid
        local_valid = state_valid and rate_valid and gain_valid
        return {
            "functional_budget_type": (
                "state_equality_plus_rate_gain_envelopes"
            ),
            "functional_budget_target": target,
            "functional_budget_achieved": achieved,
            "functional_budget_scale": scale,
            "functional_budget_maximum_scale": maximum_scale,
            "functional_budget_rate_displacement": rate_displacement,
            "functional_budget_rate_displacement_limit": rate_limit,
            "functional_budget_rate_envelope_fraction": (
                rate_displacement / rate_limit
                if rate_limit > 0.0
                else (0.0 if rate_displacement <= tolerance else float("inf"))
            ),
            "functional_budget_gain_displacement": gain_displacement,
            "functional_budget_gain_displacement_limit": (
                gain_displacement_limit
            ),
            "functional_budget_gain_displacement_limit_per_unit": (
                gain_displacement_limit_per_unit
            ),
            "functional_budget_gain_envelope_fraction": (
                gain_displacement / gain_displacement_limit
                if gain_displacement_limit > 0.0
                else (
                    0.0
                    if gain_displacement <= tolerance
                    else float("inf")
                )
            ),
            "functional_budget_gain_min_dev_observed": gain_min_observed,
            "functional_budget_gain_max_dev_observed": gain_max_observed,
            "functional_budget_gain_min_limit": gain_min,
            "functional_budget_gain_max_limit": gain_max,
            "functional_budget_relative_error": relative_error,
            "functional_budget_converged": state_converged,
            "functional_budget_state_valid": state_valid,
            "functional_budget_rate_valid": rate_valid,
            "functional_budget_gain_range_valid": gain_range_valid,
            "functional_budget_gain_displacement_valid": (
                gain_displacement_valid
            ),
            "functional_budget_gain_valid": gain_valid,
            "functional_budget_joint_local_valid": local_valid,
            "functional_budget_match_scope": (
                "dev_split_train_fitted_direction"
            ),
            "functional_budget_evaluations": evaluations,
            # Non-frozen rows remain fail-closed until the task-level audit
            # confirms that all registered directions were available.
            "functional_budget_cross_condition_complete": frozen_condition,
            "functional_budget_cross_condition_valid": frozen_condition,
            "joint_functional_budget_valid": (
                local_valid if frozen_condition else False
            ),
            "functional_budget_satisfied": (
                local_valid if frozen_condition else False
            ),
            "functional_budget_formal_ready": (
                local_valid if frozen_condition else False
            ),
        }

    if frozen_condition:
        frozen_receipt = receipt(
            achieved=0.0,
            scale=0.0,
            maximum_scale=0.0,
            rate_displacement=0.0,
            gain_displacement=0.0,
            gain_min_observed=1.0,
            gain_max_observed=1.0,
            relative_error=0.0,
            state_converged=True,
            evaluations=1,
        )
        frozen_receipt["functional_budget_target"] = 0.0
        return np.zeros_like(axis), frozen_receipt
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-14:
        return np.zeros_like(axis), receipt(
            achieved=0.0,
            scale=0.0,
            maximum_scale=0.0,
            rate_displacement=0.0,
            gain_displacement=0.0,
            gain_min_observed=1.0,
            gain_max_observed=1.0,
            relative_error=1.0,
            state_converged=False,
            evaluations=0,
        )

    direction = np.asarray(axis, dtype=np.float64) / norm
    headroom = min(gain_max - 1.0, 1.0 - gain_min)
    maximum_direction = float(np.max(np.abs(direction)))
    maximum_scale = (
        headroom / maximum_direction
        if headroom > 0.0 and maximum_direction > 0.0
        else 0.0
    )
    if maximum_scale <= 0.0:
        return np.zeros_like(axis), receipt(
            achieved=0.0,
            scale=0.0,
            maximum_scale=0.0,
            rate_displacement=0.0,
            gain_displacement=0.0,
            gain_min_observed=1.0,
            gain_max_observed=1.0,
            relative_error=1.0,
            state_converged=False,
            evaluations=0,
        )

    frozen_rates, frozen_states, frozen_gains = _simulate_axis_batch(
        network,
        dataset,
        posterior,
        np.zeros_like(axis),
        config,
        task_variant,
    )

    def rollout(scale: float) -> np.ndarray:
        _, states, _ = _simulate_axis_batch(
            network,
            dataset,
            posterior,
            scale * direction,
            config,
            task_variant,
        )
        return states

    match = match_functional_state_displacement(
        rollout,
        frozen_states,
        target_displacement=target,
        initial_scale=min(1.0, maximum_scale),
        max_scale=maximum_scale,
        relative_tolerance=float(
            controller.get("functional_match_relative_tolerance", 0.02)
        ),
        absolute_tolerance=float(
            controller.get("functional_match_absolute_tolerance", 1e-8)
        ),
        max_iterations=int(controller.get("functional_match_max_iterations", 50)),
        exclude_initial=False,
        raise_on_unreachable=False,
    )
    matched_rates, _, matched_gains = _simulate_axis_batch(
        network,
        dataset,
        posterior,
        match.scale * direction,
        config,
        task_variant,
    )
    rate_displacement = functional_state_displacement(
        matched_rates,
        frozen_rates,
        exclude_initial=False,
    )
    gain_displacement = functional_state_displacement(
        matched_gains,
        frozen_gains,
        exclude_initial=False,
    )
    return match.scale * direction, receipt(
        achieved=match.achieved_displacement,
        scale=match.scale,
        maximum_scale=maximum_scale,
        rate_displacement=rate_displacement,
        gain_displacement=gain_displacement,
        gain_min_observed=float(np.min(matched_gains)),
        gain_max_observed=float(np.max(matched_gains)),
        relative_error=match.relative_error,
        state_converged=match.converged,
        evaluations=match.n_evaluations,
    )


def _apply_cross_condition_functional_budget_audit(
    metrics_by_condition: dict[str, dict[str, object]],
    config: dict[str, Any],
) -> None:
    """Attach one fail-closed realized-budget audit to a paired task.

    Rate and gain remain envelopes, not equality targets.  Their
    cross-condition mismatch is therefore normalized by the corresponding
    registered envelope.  State mismatch is normalized by its equality
    target.  The audit mutates only the not-yet-emitted metric dictionaries.
    """

    controller = dict(config["controller"])
    active_conditions = tuple(
        condition for condition in CONDITIONS if condition != "frozen"
    )
    available = tuple(
        condition
        for condition in active_conditions
        if condition in metrics_by_condition
    )
    complete = len(available) == len(active_conditions)
    state_target = float(controller["matched_dev_state_displacement"])
    rate_limit = float(controller["matched_dev_rate_displacement_limit"])
    gain_limit = (
        float(
            next(iter(metrics_by_condition.values()))[
                "functional_budget_gain_displacement_limit"
            ]
        )
        if metrics_by_condition
        else float(controller["matched_dev_gain_displacement_limit_per_unit"])
    )
    state_tolerance = float(
        controller["cross_condition_state_relative_mismatch_tolerance"]
    )
    rate_tolerance = float(
        controller["cross_condition_rate_envelope_mismatch_tolerance"]
    )
    gain_tolerance = float(
        controller["cross_condition_gain_envelope_mismatch_tolerance"]
    )
    floor = float(
        controller.get("functional_match_absolute_tolerance", 1e-8)
    )

    def relative_range(
        field: str,
        denominator: float,
    ) -> float | None:
        if not complete:
            return None
        values = np.asarray(
            [
                float(metrics_by_condition[condition][field])
                for condition in active_conditions
            ],
            dtype=np.float64,
        )
        return float(
            (np.max(values) - np.min(values))
            / max(abs(denominator), floor)
        )

    state_mismatch = relative_range(
        "functional_budget_achieved",
        state_target,
    )
    rate_mismatch = relative_range(
        "functional_budget_rate_displacement",
        rate_limit,
    )
    gain_mismatch = relative_range(
        "functional_budget_gain_displacement",
        gain_limit,
    )
    state_mismatch_valid = bool(
        state_mismatch is not None and state_mismatch <= state_tolerance
    )
    rate_mismatch_valid = bool(
        rate_mismatch is not None and rate_mismatch <= rate_tolerance
    )
    gain_mismatch_valid = bool(
        gain_mismatch is not None and gain_mismatch <= gain_tolerance
    )
    all_local_valid = bool(
        complete
        and all(
            bool(
                metrics_by_condition[condition][
                    "functional_budget_joint_local_valid"
                ]
            )
            for condition in active_conditions
        )
    )
    cross_valid = bool(
        complete
        and all_local_valid
        and state_mismatch_valid
        and rate_mismatch_valid
        and gain_mismatch_valid
    )
    common = {
        "functional_budget_cross_condition_expected_count": len(
            active_conditions
        ),
        "functional_budget_cross_condition_available_count": len(available),
        "functional_budget_cross_condition_complete": complete,
        "functional_budget_all_nonfrozen_local_envelopes_valid": (
            all_local_valid
        ),
        "functional_budget_cross_condition_state_relative_mismatch": (
            state_mismatch
        ),
        "functional_budget_cross_condition_state_relative_mismatch_tolerance": (
            state_tolerance
        ),
        "functional_budget_cross_condition_state_mismatch_valid": (
            state_mismatch_valid
        ),
        "functional_budget_cross_condition_rate_envelope_mismatch": (
            rate_mismatch
        ),
        "functional_budget_cross_condition_rate_envelope_mismatch_tolerance": (
            rate_tolerance
        ),
        "functional_budget_cross_condition_rate_mismatch_valid": (
            rate_mismatch_valid
        ),
        "functional_budget_cross_condition_gain_envelope_mismatch": (
            gain_mismatch
        ),
        "functional_budget_cross_condition_gain_envelope_mismatch_tolerance": (
            gain_tolerance
        ),
        "functional_budget_cross_condition_gain_mismatch_valid": (
            gain_mismatch_valid
        ),
        "functional_budget_cross_condition_valid": cross_valid,
    }
    for condition, metrics in metrics_by_condition.items():
        metrics.update(common)
        local_valid = bool(metrics["functional_budget_joint_local_valid"])
        satisfied = local_valid and (
            condition == "frozen" or cross_valid
        )
        metrics["joint_functional_budget_valid"] = satisfied
        metrics["functional_budget_satisfied"] = satisfied
        metrics["functional_budget_formal_ready"] = satisfied


def _condition_metrics(
    condition: str,
    *,
    network: EIRateNetwork,
    splits: HiddenContextSplits,
    predictions: dict[str, np.ndarray],
    gate: MDRecurrentBeliefGate,
    readout: ReceiverReadout,
    config: dict[str, Any],
    seed: int,
    task_variant: str,
) -> dict[str, object]:
    natural_axis, training = _train_condition(
        condition,
        network=network,
        splits=splits,
        predictions=predictions,
        readout=readout,
        config=config,
        seed=seed,
        task_variant=task_variant,
    )
    axis, budget = _match_axis_functional_budget(
        network=network,
        dataset=splits.dev,
        posterior=predictions["dev"],
        axis=natural_axis,
        config=config,
        task_variant=task_variant,
        frozen_condition=condition == "frozen",
    )
    natural_test = _forward_no_sensitivity(
        network,
        splits.test,
        predictions["test"],
        natural_axis,
        readout,
        config,
        task_variant,
    )
    test = _forward_no_sensitivity(
        network,
        splits.test,
        predictions["test"],
        axis,
        readout,
        config,
        task_variant,
    )
    frozen = _forward_no_sensitivity(
        network,
        splits.test,
        predictions["test"],
        np.zeros_like(axis),
        readout,
        config,
        task_variant,
    )
    displacement = functional_state_displacement(
        test.states,
        frozen.states,
        exclude_initial=False,
    )
    rate_displacement = functional_state_displacement(
        test.rates,
        frozen.rates,
        exclude_initial=False,
    )
    gain_displacement = functional_state_displacement(
        test.gains,
        frozen.gains,
        exclude_initial=False,
    )
    delay_mask = np.asarray(splits.test.task.epoch) == "delay"
    delay_state_norm = (
        float(np.mean(np.linalg.norm(test.states[:, delay_mask], axis=-1)))
        if np.any(delay_mask)
        else float("nan")
    )
    delay_control_displacement = (
        functional_state_displacement(
            test.states[:, delay_mask],
            frozen.states[:, delay_mask],
            exclude_initial=False,
        )
        if np.any(delay_mask)
        else float("nan")
    )
    controller = dict(config["controller"])
    gate_audit = gate.audit_metadata()
    return {
        "status": "complete",
        "profile": str(config.get("profile", "unspecified")),
        "statistics_unit": "seed",
        "split_unit": "episode",
        "training_algorithm": _condition_training_algorithm(condition),
        "condition_schema_key": condition,
        "used_bptt": condition == "bptt_axis_only",
        "used_autograd": condition == "bptt_axis_only",
        "local_learning": condition == "local_eprop",
        "local_proposal_rule_used": condition
        in {"current_off_policy", "local_eprop"},
        "closed_loop_local_learning_claim_eligible": condition == "local_eprop",
        "closed_loop_learning": condition
        in {
            "random_update",
            "exact_forward_sensitivity",
            "bptt_axis_only",
            "local_eprop",
        },
        "off_policy_frozen_trajectory": condition == "current_off_policy",
        "off_policy_exp22_method_claimed": False,
        "dev_trajectory_recomputed_after_each_update": condition
        in {
            "random_update",
            "exact_forward_sensitivity",
            "bptt_axis_only",
            "local_eprop",
        },
        "behavior_accuracy": float(
            np.mean(np.where(test.scores >= 0.0, 1, -1) == test.targets)
        ),
        "behavior_balanced_accuracy": _balanced_accuracy(test.scores, test.targets),
        "natural_behavior_balanced_accuracy": _balanced_accuracy(
            natural_test.scores,
            natural_test.targets,
        ),
        "switch_cost": _switch_cost(
            test.scores,
            splits.test,
            post_switch_trials=int(controller["post_switch_trials"]),
        ),
        "test_loss": test.loss,
        "test_task_loss": test.task_loss,
        "max_firing_rate": test.max_firing_rate,
        "mean_firing_rate": test.mean_firing_rate,
        "gain_saturation_fraction": test.saturation_fraction,
        "gain_min_observed": float(np.min(test.gains)),
        "gain_max_observed": float(np.max(test.gains)),
        "jacobian_spectral_margin": _jacobian_margin(
            network,
            np.mean(test.states, axis=(0, 1)),
            np.mean(test.gains, axis=(0, 1)),
        ),
        "control_state_displacement": displacement,
        "control_rate_displacement": rate_displacement,
        "control_gain_displacement": gain_displacement,
        "delay_state_norm": delay_state_norm,
        "delay_control_state_displacement": delay_control_displacement,
        "delayed_sensory_routing_disabled": task_variant == "delayed",
        "controller_gain_application": "multiplicative_total_drive",
        "controller_active_epochs": (
            ["cue", "delay"]
            if task_variant == "delayed"
            else ["sensory", "delay", "response"]
        ),
        "local_eligibility_approximation": (
            "per_unit_state_rate_block_jacobian"
            if condition == "local_eprop"
            else None
        ),
        "control_axis_l1": float(np.sum(np.abs(axis))),
        "control_axis_l2": float(np.linalg.norm(axis)),
        "natural_control_axis_l1": float(np.sum(np.abs(natural_axis))),
        "natural_control_axis_l2": float(np.linalg.norm(natural_axis)),
        "control_axis_id": _fingerprint("exp23-axis-v1", axis),
        "functional_state_displacement_limit": float(
            controller["max_state_displacement"]
        ),
        "functional_budget_calibration": str(
            controller["functional_budget_calibration"]
        ),
        "functional_budget_fit_scope": (
            "development_split_only_with_train_fitted_direction"
        ),
        "functional_budget_targets_use_behavior": False,
        "functional_budget_equality_targets": [
            "state_displacement",
        ],
        "functional_budget_upper_bound_constraints": [
            "rate_displacement",
            "gain_displacement",
            "gain_physical_range",
        ],
        "all_functional_budget_terms_preregistered": True,
        "rate_gain_observables_reported_not_gated": False,
        "functional_displacement_definition": (
            "mean_over_trial_time_of_squared_state_l2_norm"
        ),
        "functional_budget_primary_metric": "state_displacement",
        "functional_secondary_budgets": [
            "rate_displacement",
            "gain_displacement",
        ],
        "absolute_performance_reported_separately": True,
        "relative_noninferiority_reported_separately": True,
        "recurrent_learning": False,
        "readout_fit_train_only": True,
        "readout_fit_scope": "training_split_only",
        "gate_fit_train_only": True,
        "gate_fit_scope": "training_split_only",
        "axis_fit_dev_only": True,
        "axis_fit_scope": "development_split_only",
        "test_used_for_axis_fit": False,
        "axis_selection_accessed_test": False,
        "gate_test_accessed_true_context": False,
        "third_factor_accessed_true_context": False,
        "hidden_context_access_audit_passed": True,
        "local_rule_autograd_free": condition == "local_eprop"
        and condition != "bptt_axis_only",
        "local_rule_bptt_free": condition == "local_eprop"
        and condition != "bptt_axis_only",
        "gate_moment_anchor_identifiable": bool(
            gate_audit["moment_anchor_identifiable"]
        ),
        "gate_estimated_context_hazard": float(
            gate_audit["estimated_context_hazard"]
        ),
        "gate_estimated_cue_reliability": float(
            gate_audit["estimated_cue_reliability"]
        ),
        "gate_mean_absolute_signed_belief_dev": float(
            np.mean(np.abs(2.0 * predictions["dev"] - 1.0))
        ),
        **budget,
        **training,
        **_ood_metrics(
            network=network,
            gate=gate,
            readout=readout,
            axis=axis,
            config=config,
            seed=seed,
            task_variant=task_variant,
        ),
    }


def run_seed(config: dict[str, Any], seed: int, results_root: str | Path) -> Path:
    """Run all paired Exp23 task/algorithm cells for one independent seed."""

    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "paired_closed_loop_axis_only_controller_audit",
        "recurrent_learning": False,
    }
    with ExperimentRun(EXPERIMENT, seed, run_config, results_root=results_root) as run:
        planned = _planned_conditions(config)
        run.register_conditions(planned)
        dimensions = {
            (str(item["task_variant"]), str(item["condition"])): item
            for item in planned
        }
        emitted: set[tuple[str, str]] = set()
        for task_variant in TASKS:
            task_cells = [
                (task_variant, condition)
                for condition in CONDITIONS
            ]
            try:
                dataset, tape_id = _task_dataset(config, seed, task_variant)
                splits = split_hidden_context_dataset(
                    dataset,
                    outer_test_fraction=float(config["outer_test_fraction"]),
                    validation_fraction=float(config["validation_fraction"]),
                    seed=seed,
                )
                network = _network(config, dataset.config, seed)
                initial_weights = network.recurrent_weights
                recurrent_frozen_hash = _fingerprint(
                    "exp23-frozen-recurrent-v1",
                    initial_weights,
                )
                recurrent_snapshot = network.recurrent_weights
                recurrent_snapshot.flat[0] = recurrent_snapshot.flat[0] + 1.0
                recurrent_copy_isolation_audit_passed = bool(
                    np.array_equal(network.recurrent_weights, initial_weights)
                    and not np.shares_memory(
                        recurrent_snapshot,
                        network.recurrent_weights,
                    )
                )
                if not recurrent_copy_isolation_audit_passed:
                    raise RuntimeError(
                        "Exp23 recurrent-weight snapshot is not copy isolated"
                    )
                network_id = _fingerprint(
                    "exp23-frozen-high-rank-dale-ei-v1",
                    initial_weights,
                    network.input_weights,
                    network.excitatory_mask,
                )
                gate, predictions, gate_id = _fit_gate(splits, config, seed)
                readout = _neutral_readout(
                    network,
                    splits,
                    predictions["train"],
                    config,
                    task_variant,
                )
                pairing_bundle_id = _fingerprint(
                    "exp23-paired-mechanism-bundle-v1",
                    tape_id,
                    splits.fingerprint,
                    network_id,
                    gate_id,
                    readout.checkpoint_id,
                    readout.train_data_id,
                )
                shared = {
                    "random_tape_id": tape_id,
                    "split_id": splits.fingerprint,
                    "network_init_id": network_id,
                    "gate_checkpoint_id": gate_id,
                    "readout_checkpoint_id": readout.checkpoint_id,
                    "readout_fit_data_id": readout.train_data_id,
                    "pairing_bundle_id": pairing_bundle_id,
                    "network_physical_rank": int(
                        np.linalg.matrix_rank(initial_weights)
                    ),
                    "network_unit_count": network.n_units,
                    "network_input_count": network.n_inputs,
                    "network_high_rank": np.linalg.matrix_rank(initial_weights)
                    >= 0.9 * network.n_units,
                    "paired_network_gate_readout_split_tape": True,
                    "recurrent_frozen_hash": recurrent_frozen_hash,
                    "recurrent_weights_snapshot_is_copy": True,
                    "recurrent_copy_isolation_audit_passed": (
                        recurrent_copy_isolation_audit_passed
                    ),
                    "statistics_unit": "seed",
                }
                task_metrics: dict[str, dict[str, object]] = {}
                task_errors: dict[str, Exception] = {}
                for condition in CONDITIONS:
                    try:
                        recurrent_before = network.recurrent_weights
                        recurrent_before_hash = _fingerprint(
                            "exp23-frozen-recurrent-v1",
                            recurrent_before,
                        )
                        metrics = _condition_metrics(
                            condition,
                            network=network,
                            splits=splits,
                            predictions=predictions,
                            gate=gate,
                            readout=readout,
                            config=config,
                            seed=seed,
                            task_variant=task_variant,
                        )
                        recurrent_after = network.recurrent_weights
                        recurrent_after_hash = _fingerprint(
                            "exp23-frozen-recurrent-v1",
                            recurrent_after,
                        )
                        recurrent_hash_audit_passed = bool(
                            np.array_equal(recurrent_before, initial_weights)
                            and np.array_equal(recurrent_after, initial_weights)
                            and recurrent_before_hash == recurrent_frozen_hash
                            and recurrent_after_hash == recurrent_frozen_hash
                        )
                        if not recurrent_hash_audit_passed:
                            raise RuntimeError(
                                "a paired Exp23 condition changed recurrent weights"
                            )
                        task_metrics[condition] = {
                            **metrics,
                            "recurrent_weights_hash_before_condition": (
                                recurrent_before_hash
                            ),
                            "recurrent_weights_hash_after_condition": (
                                recurrent_after_hash
                            ),
                            "recurrent_hash_audit_passed": (
                                recurrent_hash_audit_passed
                            ),
                        }
                    except Exception as error:
                        task_errors[condition] = error

                _apply_cross_condition_functional_budget_audit(
                    task_metrics,
                    config,
                )
                for condition in CONDITIONS:
                    key = (task_variant, condition)
                    if condition in task_errors:
                        run.mark_condition_failure(
                            task_errors[condition],
                            **dimensions[key],
                        )
                    else:
                        run.record(
                            {**shared, **task_metrics[condition]},
                            **dimensions[key],
                        )
                    emitted.add(key)
            except Exception as error:
                for key in task_cells:
                    if key not in emitted:
                        run.mark_condition_failure(error, **dimensions[key])
                        emitted.add(key)
        if emitted != set(dimensions):
            raise RuntimeError("Exp23 did not emit every registered condition")
        return run.path


def main(argv: list[str] | None = None) -> None:
    parser = basic_parser(
        "Closed-loop local gain-controller learning",
        "configs/smoke/exp23_closed_loop_local_controller.json",
    )
    args = parser.parse_args(argv)
    config = load_json_config(args.config)
    seeds = seed_list(args.seeds if args.seeds is not None else config["seeds"])
    for seed in seeds:
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
