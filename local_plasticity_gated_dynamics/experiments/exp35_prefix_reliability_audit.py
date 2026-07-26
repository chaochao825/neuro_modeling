"""Retrospective ORBIT baseline audit and controlled prefix-router falsification."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.analysis.orbit_streaming_metrics import (
    actuator_headroom,
    holm_adjust,
    paired_user_inference,
    reduce_to_user_accuracy,
    task_video_accuracy_rows,
)
from src.analysis.prefix_reliability_metrics import routing_stress_metrics
from src.data.orbit_streaming import (
    OrbitEmbeddingEpisode,
    OrbitEpisodeSamplingConfig,
    OrbitFeatureStore,
    validate_user_disjoint_stores,
)
from src.models.causal_consensus_gate import (
    CausalConsensusConfig,
    CausalConsensusGate,
    instantaneous_majority_predictions,
)
from src.models.prefix_reliability import (
    ActionCalibration,
    action_probabilities,
    fit_action_calibration,
    prefix_class_vote,
    prefix_probability_ensemble,
)
from src.models.streaming_fewshot_actuators import (
    ACTUATOR_NAMES,
    PersonalizedStreamingActuators,
    StreamingActuatorConfig,
    StreamingActuatorTrace,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


EXPERIMENT = "exp35_prefix_reliability_audit"
PROTOCOL_VERSION = "exp35_prefix_reliability_audit_v2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFIX_ACTION_CONDITIONS = tuple(
    f"prefix_{name}_probability" for name in ACTUATOR_NAMES
)
ORBIT_CONDITIONS = (
    *ACTUATOR_NAMES,
    "selection_fixed_best",
    "instantaneous_majority",
    "prefix_consistency",
    "lagged_prefix_consistency",
    "prefix_vote_equal",
    "current_probability_equal",
    *PREFIX_ACTION_CONDITIONS,
    "selection_prefix_single",
    "prefix_probability_equal",
    "current_calibrated_stack",
    "prefix_calibrated_stack",
    "oracle_action_per_frame",
)
PRIMARY_COMPARATORS = (
    "lagged_prefix_consistency",
    "prefix_vote_equal",
    "prefix_probability_equal",
    "prefix_calibrated_stack",
    "selection_prefix_single",
    "temporal",
)
STRESS_SCENARIOS = (
    "stable_correct_vs_stable_wrong",
    "stable_wrong_vs_noisy_correct",
    "within_stream_switch",
    "boundary_switch",
    "prefix_bias",
)
STRESS_CONDITIONS = (
    "prefix_consistency",
    "lagged_prefix_consistency",
    "prefix_vote_equal",
    "prefix_probability_equal",
    "prefix_calibrated_stack",
)


def _calibration_dict(calibration: ActionCalibration) -> dict[str, Any]:
    return {
        "temperatures": calibration.temperatures.tolist(),
        "stacking_weights": calibration.stacking_weights.tolist(),
        "action_nll": calibration.action_nll.tolist(),
        "ensemble_nll": calibration.ensemble_nll,
        "n_frames": calibration.n_frames,
    }


def _sampling_config(config: Mapping[str, Any]) -> OrbitEpisodeSamplingConfig:
    return OrbitEpisodeSamplingConfig(**dict(config["sampling"]))


def _actuator_config(config: Mapping[str, Any], *, retention: float) -> StreamingActuatorConfig:
    payload = dict(config["actuators"])
    payload["temporal_retention"] = float(retention)
    return StreamingActuatorConfig(**payload)


def _gate_config(config: Mapping[str, Any]) -> CausalConsensusConfig:
    payload = dict(config["gate"])
    payload["tie_break_order"] = tuple(payload["tie_break_order"])
    return CausalConsensusConfig(**payload)


def _users(
    store: OrbitFeatureStore, requested: Iterable[str] | None
) -> tuple[str, ...]:
    if requested is None:
        return store.users
    selected = tuple(map(str, requested))
    if not selected:
        return store.users
    if len(selected) != len(set(selected)):
        raise ValueError("requested users must be unique")
    missing = set(selected) - set(store.users)
    if missing:
        raise ValueError(f"requested users are absent: {sorted(missing)}")
    return selected


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp35 protocol version mismatch")
    profile = str(config.get("profile"))
    if profile not in {"smoke", "development", "retrospective"}:
        raise ValueError("profile must be smoke, development, or retrospective")
    if config.get("evidence_provenance") != "retrospective_mechanism_audit":
        raise ValueError("Exp35 evidence provenance must remain retrospective")
    for key in (
        "used_eval_labels_for_fit",
        "used_future_frames",
        "used_autograd",
        "used_bptt",
    ):
        if config.get(key) is not False:
            raise ValueError(f"Exp35 requires {key}=false")
    if config.get("used_selection_labels_for_calibration") is not True:
        raise ValueError("Exp35 must disclose validation-label calibration")
    if config.get("selection_split") not in {"train", "validation"}:
        raise ValueError("selection_split must be train or validation")
    if config.get("eval_split") not in {"validation", "test"}:
        raise ValueError("eval_split must be validation or test")
    if profile == "retrospective" and config.get("eval_split") != "test":
        raise ValueError("retrospective profile must evaluate the exposed test split")
    for name in ("selection_feature_root", "eval_feature_root"):
        if not isinstance(config.get(name), str) or not str(config[name]):
            raise ValueError(f"{name} must be a non-empty path")
    selection_users = set(map(str, config.get("selection_user_ids", [])))
    eval_users = set(map(str, config.get("eval_user_ids", [])))
    if config["selection_split"] == config["eval_split"]:
        if not selection_users or not eval_users or selection_users & eval_users:
            raise ValueError("same-split development needs explicit disjoint users")
    for key in ("n_selection_tasks_per_user", "n_eval_tasks_per_user"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    grid = np.asarray(config.get("temporal_retention_grid", []), dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("temporal_retention_grid needs at least two finite values")
    if np.any((grid < 0.0) | (grid >= 1.0)) or len(set(grid.tolist())) != grid.size:
        raise ValueError("temporal retentions must be unique values in [0, 1)")
    calibration = config.get("calibration", {})
    bounds = calibration.get("temperature_bounds", [])
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("calibration.temperature_bounds must contain two values")
    if float(calibration.get("stacking_l2", -1.0)) < 0.0:
        raise ValueError("calibration.stacking_l2 must be non-negative")
    stride = calibration.get("frame_stride", 1)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("calibration.frame_stride must be a positive integer")
    bootstrap = config.get("analysis", {}).get("bootstrap_samples")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 100:
        raise ValueError("analysis.bootstrap_samples must be an integer >= 100")
    _sampling_config(config)
    _actuator_config(config, retention=float(grid[0]))
    gate = _gate_config(config)
    if gate.delay_frames != 0 or gate.reset_each_frame:
        raise ValueError("the main Exp35 gate must be current-prefix and persistent")


def _sample_episode(
    store: OrbitFeatureStore,
    user_id: str,
    task_index: int,
    *,
    seed: int,
    phase: str,
    config: Mapping[str, Any],
) -> OrbitEmbeddingEpisode:
    return store.sample_episode(
        user_id,
        seed=derive_seed(seed, phase, user_id, task_index),
        task_index=task_index,
        config=_sampling_config(config),
    )


def _fit_base_actuators(
    episode: OrbitEmbeddingEpisode, config: Mapping[str, Any]
) -> PersonalizedStreamingActuators:
    first_retention = float(config["temporal_retention_grid"][0])
    return PersonalizedStreamingActuators.fit(
        episode.support,
        n_classes=episode.n_classes,
        config=_actuator_config(config, retention=first_retention),
    )


def _trace_with_retention(
    fitted: PersonalizedStreamingActuators,
    episode: OrbitEmbeddingEpisode,
    *,
    retention: float,
) -> StreamingActuatorTrace:
    configured = replace(
        fitted,
        config=replace(fitted.config, temporal_retention=float(retention)),
    )
    return configured.trace(episode.query_observation)


def _selection_fit(
    store: OrbitFeatureStore,
    users: tuple[str, ...],
    *,
    seed: int,
    config: Mapping[str, Any],
) -> tuple[float, int, int, ActionCalibration, dict[str, Any]]:
    grid = tuple(float(value) for value in config["temporal_retention_grid"])
    temporal_correct = np.zeros(len(grid), dtype=np.int64)
    temporal_frames = 0
    failures: list[dict[str, object]] = []
    for user_id in users:
        for task_index in range(int(config["n_selection_tasks_per_user"])):
            try:
                episode = _sample_episode(
                    store,
                    user_id,
                    task_index,
                    seed=seed,
                    phase="selection",
                    config=config,
                )
                fitted = _fit_base_actuators(episode, config)
                for index, retention in enumerate(grid):
                    trace = _trace_with_retention(
                        fitted, episode, retention=retention
                    )
                    temporal_correct[index] += int(
                        np.sum(
                            trace.predictions[:, ACTUATOR_NAMES.index("temporal")]
                            == episode.query_labels
                        )
                    )
                temporal_frames += int(episode.query_labels.size)
            except Exception as error:
                failures.append(
                    {
                        "stage": "retention_selection",
                        "user_id": user_id,
                        "task_index": task_index,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    if temporal_frames == 0:
        raise RuntimeError("temporal retention selection produced no frames")
    if failures and bool(config.get("require_complete_selection_split", False)):
        raise RuntimeError(
            f"retention selection lost {len(failures)} required user/task cells"
        )
    temporal_accuracy = temporal_correct / temporal_frames
    selected_retention = grid[int(np.argmax(temporal_accuracy))]

    action_correct = np.zeros(len(ACTUATOR_NAMES), dtype=np.int64)
    prefix_action_correct = np.zeros(len(ACTUATOR_NAMES), dtype=np.int64)
    action_frames = 0
    score_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    stride = int(config["calibration"].get("frame_stride", 1))
    for user_id in users:
        for task_index in range(int(config["n_selection_tasks_per_user"])):
            try:
                episode = _sample_episode(
                    store,
                    user_id,
                    task_index,
                    seed=seed,
                    phase="selection",
                    config=config,
                )
                fitted = _fit_base_actuators(episode, config)
                trace = _trace_with_retention(
                    fitted, episode, retention=selected_retention
                )
                action_correct += np.sum(
                    trace.predictions == episode.query_labels[:, None], axis=0
                )
                probabilities = action_probabilities(trace.scores)
                for action in range(len(ACTUATOR_NAMES)):
                    weights = np.zeros(len(ACTUATOR_NAMES), dtype=np.float64)
                    weights[action] = 1.0
                    prefix = prefix_probability_ensemble(
                        probabilities,
                        video_ids=trace.video_ids,
                        action_weights=weights,
                        retention=1.0,
                    )
                    prefix_action_correct[action] += int(
                        np.sum(prefix.predictions == episode.query_labels)
                    )
                action_frames += int(episode.query_labels.size)
                score_batches.append(np.asarray(trace.scores[::stride]))
                label_batches.append(np.asarray(episode.query_labels[::stride]))
            except Exception as error:
                failures.append(
                    {
                        "stage": "calibration",
                        "user_id": user_id,
                        "task_index": task_index,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    if action_frames == 0 or not score_batches:
        raise RuntimeError("action selection and calibration produced no frames")
    calibration_failures = [
        failure for failure in failures if failure["stage"] == "calibration"
    ]
    if calibration_failures and bool(
        config.get("require_complete_selection_split", False)
    ):
        raise RuntimeError(
            f"calibration lost {len(calibration_failures)} required user/task cells"
        )
    calibration = fit_action_calibration(
        score_batches,
        label_batches,
        temperature_bounds=tuple(
            map(float, config["calibration"]["temperature_bounds"])
        ),
        stacking_l2=float(config["calibration"]["stacking_l2"]),
    )
    fixed_action = int(np.argmax(action_correct))
    prefix_action = int(np.argmax(prefix_action_correct))
    audit = {
        "selected_temporal_retention": selected_retention,
        "temporal_retention_grid": list(grid),
        "temporal_correct": temporal_correct.tolist(),
        "temporal_accuracy": temporal_accuracy.tolist(),
        "temporal_selection_frames": temporal_frames,
        "fixed_action_correct": action_correct.tolist(),
        "fixed_action_accuracy": (action_correct / action_frames).tolist(),
        "fixed_action_frames": action_frames,
        "selected_fixed_action": fixed_action,
        "selected_fixed_name": ACTUATOR_NAMES[fixed_action],
        "prefix_action_correct": prefix_action_correct.tolist(),
        "prefix_action_accuracy": (prefix_action_correct / action_frames).tolist(),
        "selected_prefix_action": prefix_action,
        "selected_prefix_name": ACTUATOR_NAMES[prefix_action],
        "calibration": _calibration_dict(calibration),
        "calibration_frame_stride": stride,
        "failures": failures,
    }
    return selected_retention, fixed_action, prefix_action, calibration, audit


def _controller_state_bytes(
    condition: str, *, n_classes: int, fixed_action: int
) -> int:
    scalar_bytes = np.dtype(np.float64).itemsize
    if condition in {"prefix_consistency", "lagged_prefix_consistency"}:
        return len(ACTUATOR_NAMES) * n_classes * scalar_bytes
    if condition in {
        "prefix_vote_equal",
        "prefix_probability_equal",
        "prefix_calibrated_stack",
        "selection_prefix_single",
        *PREFIX_ACTION_CONDITIONS,
    }:
        return n_classes * scalar_bytes
    if condition == "temporal" or (
        condition == "selection_fixed_best"
        and ACTUATOR_NAMES[fixed_action] == "temporal"
    ):
        return n_classes * scalar_bytes
    return 0


def _condition_output(
    condition: str,
    *,
    episode: OrbitEmbeddingEpisode,
    trace: StreamingActuatorTrace,
    fixed_action: int,
    prefix_action: int,
    calibration: ActionCalibration,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, str]:
    n = episode.query_labels.size
    actions: np.ndarray | None = None
    state = np.zeros(n, dtype=np.float64)
    if condition in ACTUATOR_NAMES:
        action = ACTUATOR_NAMES.index(condition)
        actions = np.full(n, action, dtype=np.int64)
        predictions = trace.predictions[:, action]
        costs = trace.action_event_l1[:, action]
        scope = "selected_actuator"
    elif condition == "selection_fixed_best":
        actions = np.full(n, fixed_action, dtype=np.int64)
        predictions = trace.predictions[:, fixed_action]
        costs = trace.action_event_l1[:, fixed_action]
        scope = "selected_actuator"
    elif condition in {"prefix_consistency", "lagged_prefix_consistency"}:
        gate_config = _gate_config(config)
        if condition == "lagged_prefix_consistency":
            gate_config = replace(gate_config, delay_frames=1)
        selected = CausalConsensusGate(
            len(ACTUATOR_NAMES), episode.n_classes, config=gate_config
        ).trace(
            trace.predictions,
            video_ids=trace.video_ids,
            action_event_l1=trace.action_event_l1,
        )
        predictions = selected.predictions
        actions = selected.actions
        costs = selected.full_bank_event_l1
        state = selected.count_state_l1
        scope = "full_actuator_bank"
    elif condition == "instantaneous_majority":
        predictions, actions = instantaneous_majority_predictions(
            trace.predictions,
            n_classes=episode.n_classes,
            tie_break_order=_gate_config(config).tie_break_order,
        )
        costs = np.sum(trace.action_event_l1, axis=1)
        scope = "full_actuator_bank"
    elif condition == "prefix_vote_equal":
        ensemble = prefix_class_vote(
            trace.predictions,
            n_classes=episode.n_classes,
            video_ids=trace.video_ids,
            retention=1.0,
        )
        predictions = ensemble.predictions
        costs = np.sum(trace.action_event_l1, axis=1)
        state = ensemble.state_l1
        scope = "full_actuator_bank"
    elif condition in PREFIX_ACTION_CONDITIONS or condition == "selection_prefix_single":
        if condition == "selection_prefix_single":
            action = prefix_action
        else:
            action_name = condition.removeprefix("prefix_").removesuffix(
                "_probability"
            )
            action = ACTUATOR_NAMES.index(action_name)
        weights = np.zeros(len(ACTUATOR_NAMES), dtype=np.float64)
        weights[action] = 1.0
        ensemble = prefix_probability_ensemble(
            action_probabilities(trace.scores),
            video_ids=trace.video_ids,
            action_weights=weights,
            retention=1.0,
        )
        predictions = ensemble.predictions
        costs = trace.action_event_l1[:, action]
        state = ensemble.state_l1
        scope = "selected_actuator"
    elif condition in {
        "current_probability_equal",
        "prefix_probability_equal",
        "current_calibrated_stack",
        "prefix_calibrated_stack",
    }:
        calibrated = "calibrated" in condition
        probabilities = action_probabilities(
            trace.scores,
            temperatures=calibration.temperatures if calibrated else None,
        )
        ensemble = prefix_probability_ensemble(
            probabilities,
            video_ids=trace.video_ids,
            action_weights=calibration.stacking_weights if calibrated else None,
            retention=1.0 if condition.startswith("prefix") else 0.0,
        )
        predictions = ensemble.predictions
        costs = np.sum(trace.action_event_l1, axis=1)
        state = (
            ensemble.state_l1
            if condition.startswith("prefix")
            else np.zeros(n, dtype=np.float64)
        )
        scope = "full_actuator_bank"
    elif condition == "oracle_action_per_frame":
        correct = trace.predictions == episode.query_labels[:, None]
        actions = np.where(
            np.any(correct, axis=1), np.argmax(correct, axis=1), fixed_action
        ).astype(np.int64)
        predictions = trace.predictions[np.arange(n), actions]
        costs = trace.action_event_l1[np.arange(n), actions]
        scope = "label_oracle_full_bank"
    else:
        raise ValueError(f"unknown Exp35 condition: {condition}")
    return (
        np.asarray(predictions, dtype=np.int64),
        None if actions is None else np.asarray(actions, dtype=np.int64),
        np.asarray(costs, dtype=np.float64),
        np.asarray(state, dtype=np.float64),
        scope,
    )


def _evaluate_episode(
    episode: OrbitEmbeddingEpisode,
    *,
    selected_retention: float,
    fixed_action: int,
    prefix_action: int,
    calibration: ActionCalibration,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fitted = _fit_base_actuators(episode, config)
    trace = _trace_with_retention(
        fitted, episode, retention=selected_retention
    )
    headroom = actuator_headroom(episode.query_labels, trace.predictions)
    rows: list[pd.DataFrame] = []
    for condition in ORBIT_CONDITIONS:
        predictions, actions, costs, state, compute_scope = _condition_output(
            condition,
            episode=episode,
            trace=trace,
            fixed_action=fixed_action,
            prefix_action=prefix_action,
            calibration=calibration,
            config=config,
        )
        frame = task_video_accuracy_rows(
            user_id=episode.user_id,
            task_index=episode.task_index,
            condition=condition,
            labels=episode.query_labels,
            predictions=predictions,
            video_ids=episode.query_video_ids,
            selected_actions=actions,
        )
        for column, values in (
            ("mean_event_l1", costs),
            ("mean_controller_state_l1", state),
        ):
            mapping = {
                str(video_id): float(
                    np.mean(values[episode.query_video_ids == video_id])
                )
                for video_id in np.unique(episode.query_video_ids)
            }
            frame[column] = frame["video_id"].map(mapping)
        frame["controller_state_bytes"] = _controller_state_bytes(
            condition, n_classes=episode.n_classes, fixed_action=fixed_action
        )
        frame["compute_scope"] = compute_scope
        frame["status"] = "complete"
        frame["episode_fingerprint"] = episode.fingerprint
        frame["trace_fingerprint"] = trace.fingerprint
        rows.append(frame)
    diagnostic = {
        "user_id": episode.user_id,
        "task_index": episode.task_index,
        "n_classes": episode.n_classes,
        "best_fixed_accuracy": headroom.best_fixed_accuracy,
        "oracle_accuracy": headroom.oracle_accuracy,
        "oracle_gain": headroom.oracle_gain,
        "action_disagreement": headroom.action_disagreement,
        "support_write_l1": fitted.write_l1_cost,
        "support_write_l2": fitted.write_l2_cost,
        "n_excluded_query_videos": len(episode.excluded_query_video_ids),
        "excluded_query_video_ids": json.dumps(episode.excluded_query_video_ids),
    }
    return pd.concat(rows, ignore_index=True), diagnostic


def _scores_from_predictions(
    predictions: np.ndarray, *, n_classes: int, confidence: np.ndarray
) -> np.ndarray:
    n_frames, n_actions = predictions.shape
    scores = np.empty((n_frames, n_actions, n_classes), dtype=np.float64)
    for frame in range(n_frames):
        for action in range(n_actions):
            high = float(confidence[frame, action])
            low = (1.0 - high) / (n_classes - 1)
            scores[frame, action] = np.log(max(low, 1e-12))
            scores[frame, action, predictions[frame, action]] = np.log(
                max(high, 1e-12)
            )
    return scores


def _synthetic_calibration(seed: int) -> ActionCalibration:
    rng = np.random.default_rng(derive_seed(seed, "stress", "calibration"))
    labels = rng.integers(0, 2, size=400, dtype=np.int64)
    first_correct = rng.random(labels.size) < 0.9
    second_correct = rng.random(labels.size) < 0.3
    predictions = np.column_stack(
        (
            np.where(first_correct, labels, 1 - labels),
            np.where(second_correct, labels, 1 - labels),
        )
    ).astype(np.int64)
    confidence = np.column_stack(
        (np.full(labels.size, 0.8), np.full(labels.size, 0.95))
    )
    scores = _scores_from_predictions(
        predictions, n_classes=2, confidence=confidence
    )
    return fit_action_calibration(
        [scores], [labels], temperature_bounds=(0.05, 20.0), stacking_l2=1e-3
    )


def _stress_scenario(
    scenario: str, *, seed: int, n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None]:
    if n_frames < 16 or n_frames % 2:
        raise ValueError("stress n_frames must be an even integer >= 16")
    rng = np.random.default_rng(derive_seed(seed, "stress", scenario))
    half = n_frames // 2
    switch_index: int | None = None
    if scenario == "stable_correct_vs_stable_wrong":
        labels = np.zeros(n_frames, dtype=np.int64)
        predictions = np.column_stack((labels, np.ones(n_frames, dtype=np.int64)))
        videos = np.full(n_frames, "v0")
    elif scenario == "stable_wrong_vs_noisy_correct":
        labels = np.zeros(n_frames, dtype=np.int64)
        first = labels.copy()
        first[np.arange(4, n_frames, 5)] = 1
        predictions = np.column_stack((first, np.ones(n_frames, dtype=np.int64)))
        videos = np.full(n_frames, "v0")
    elif scenario == "within_stream_switch":
        labels = np.concatenate(
            (np.zeros(half, dtype=np.int64), np.ones(half, dtype=np.int64))
        )
        predictions = np.column_stack((labels, np.zeros(n_frames, dtype=np.int64)))
        videos = np.full(n_frames, "v0")
        switch_index = half
    elif scenario == "boundary_switch":
        labels = np.concatenate(
            (np.zeros(half, dtype=np.int64), np.ones(half, dtype=np.int64))
        )
        second = np.zeros(n_frames, dtype=np.int64)
        second[half:] = rng.integers(0, 2, size=half)
        predictions = np.column_stack((labels, second))
        videos = np.concatenate((np.full(half, "v0"), np.full(half, "v1")))
        switch_index = half
    elif scenario == "prefix_bias":
        labels = np.ones(n_frames, dtype=np.int64)
        first = labels.copy()
        first[: max(2, n_frames // 8)] = 0
        predictions = np.column_stack((first, np.zeros(n_frames, dtype=np.int64)))
        videos = np.full(n_frames, "v0")
    else:
        raise ValueError(f"unknown stress scenario: {scenario}")
    confidence = np.column_stack(
        (np.full(n_frames, 0.8), np.full(n_frames, 0.95))
    )
    scores = _scores_from_predictions(
        predictions, n_classes=2, confidence=confidence
    )
    return labels, predictions, scores, videos, switch_index


def run_stress_panel(
    *, seed: int, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, ActionCalibration]:
    calibration = _synthetic_calibration(seed)
    n_frames = int(config.get("stress", {}).get("n_frames", 64))
    stability = int(config.get("stress", {}).get("stability_frames", 3))
    rows: list[dict[str, object]] = []
    for scenario in STRESS_SCENARIOS:
        labels, action_predictions, scores, videos, switch_index = _stress_scenario(
            scenario, seed=seed, n_frames=n_frames
        )
        probabilities = action_probabilities(scores)
        calibrated_probabilities = action_probabilities(
            scores, temperatures=calibration.temperatures
        )
        gate_config = CausalConsensusConfig(tie_break_order=(1, 0))
        outputs: dict[str, np.ndarray] = {
            "prefix_consistency": CausalConsensusGate(
                2, 2, config=gate_config
            ).trace(action_predictions, video_ids=videos).predictions,
            "lagged_prefix_consistency": CausalConsensusGate(
                2, 2, config=replace(gate_config, delay_frames=1)
            ).trace(action_predictions, video_ids=videos).predictions,
            "prefix_vote_equal": prefix_class_vote(
                action_predictions,
                n_classes=2,
                video_ids=videos,
                retention=1.0,
            ).predictions,
            "prefix_probability_equal": prefix_probability_ensemble(
                probabilities, video_ids=videos, retention=1.0
            ).predictions,
            "prefix_calibrated_stack": prefix_probability_ensemble(
                calibrated_probabilities,
                video_ids=videos,
                action_weights=calibration.stacking_weights,
                retention=1.0,
            ).predictions,
        }
        stable_wrong = action_predictions[:, 1]
        for condition, output in outputs.items():
            metrics = routing_stress_metrics(
                labels,
                output,
                stable_wrong_predictions=stable_wrong,
                switch_index=switch_index,
                stability_frames=stability,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "condition": condition,
                    **asdict(metrics),
                    "status": "complete",
                }
            )
    return pd.DataFrame(rows), calibration


def _make_store(
    root: str,
    *,
    split: str,
    split_path: Path,
    complete: bool,
    cache: bool,
) -> OrbitFeatureStore:
    return OrbitFeatureStore(
        Path(root).expanduser(),
        split=split,
        official_splits_path=split_path,
        require_complete_split=complete,
        cache_videos=cache,
    )


def run_seed(config: Mapping[str, Any], *, seed: int, results_root: str | Path) -> Path:
    _validate_config(config)
    initialize_seed(seed)
    split_path = Path(str(config["official_splits_path"]))
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    selection_store = _make_store(
        str(config["selection_feature_root"]),
        split=str(config["selection_split"]),
        split_path=split_path,
        complete=bool(config.get("require_complete_selection_split", False)),
        cache=bool(config.get("cache_features_in_memory", False)),
    )
    if (
        config["selection_split"] == config["eval_split"]
        and Path(str(config["selection_feature_root"])).expanduser().resolve()
        == Path(str(config["eval_feature_root"])).expanduser().resolve()
    ):
        eval_store = selection_store
    else:
        eval_store = _make_store(
            str(config["eval_feature_root"]),
            split=str(config["eval_split"]),
            split_path=split_path,
            complete=bool(config.get("require_complete_eval_split", False)),
            cache=bool(config.get("cache_features_in_memory", False)),
        )
        validate_user_disjoint_stores((selection_store, eval_store))
    selection_users = _users(selection_store, config.get("selection_user_ids"))
    eval_users = _users(eval_store, config.get("eval_user_ids"))
    if set(selection_users) & set(eval_users):
        raise ValueError("selection and evaluation users must be disjoint")

    run_config = dict(config)
    run_config["selection_users"] = list(selection_users)
    run_config["eval_users"] = list(eval_users)
    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
        run_label=str(config["profile"]),
    ) as run:
        run.register_conditions(
            [
                {
                    "panel": "orbit",
                    "user_id": user_id,
                    "task_index": task_index,
                    "condition": condition,
                }
                for user_id in eval_users
                for task_index in range(int(config["n_eval_tasks_per_user"]))
                for condition in ORBIT_CONDITIONS
            ]
            + [
                {
                    "panel": "stress",
                    "scenario": scenario,
                    "condition": condition,
                }
                for scenario in STRESS_SCENARIOS
                for condition in STRESS_CONDITIONS
            ]
        )
        selected_retention, fixed_action, prefix_action, calibration, selection_audit = (
            _selection_fit(
                selection_store,
                selection_users,
                seed=seed,
                config=config,
            )
        )
        (run.path / "selection_calibration_audit.json").write_text(
            json.dumps(selection_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raw_frames: list[pd.DataFrame] = []
        diagnostics: list[dict[str, Any]] = []
        for user_id in eval_users:
            for task_index in range(int(config["n_eval_tasks_per_user"])):
                try:
                    episode = _sample_episode(
                        eval_store,
                        user_id,
                        task_index,
                        seed=seed,
                        phase="eval",
                        config=config,
                    )
                    rows, diagnostic = _evaluate_episode(
                        episode,
                        selected_retention=selected_retention,
                        fixed_action=fixed_action,
                        prefix_action=prefix_action,
                        calibration=calibration,
                        config=config,
                    )
                    raw_frames.append(rows)
                    diagnostics.append(diagnostic)
                    for row in rows.to_dict("records"):
                        dimensions = {
                            key: row.pop(key)
                            for key in ("user_id", "task_index", "video_id", "condition")
                        }
                        run.record(row, panel="orbit", **dimensions)
                except Exception as error:
                    for condition in ORBIT_CONDITIONS:
                        run.mark_condition_failure(
                            error,
                            panel="orbit",
                            user_id=user_id,
                            task_index=task_index,
                            video_id="unavailable",
                            condition=condition,
                        )
        if not raw_frames:
            raise RuntimeError("all Exp35 ORBIT evaluation tasks failed")
        raw = pd.concat(raw_frames, ignore_index=True)
        diagnostic_frame = pd.DataFrame(diagnostics)
        raw.to_csv(run.path / "raw_orbit_video_metrics.csv", index=False)
        diagnostic_frame.to_csv(run.path / "orbit_actuator_headroom.csv", index=False)
        user_rows = reduce_to_user_accuracy(raw)
        user_rows.to_csv(run.path / "orbit_user_metrics.csv", index=False)
        means = user_rows.groupby("condition")["user_video_mean_accuracy"].mean().to_dict()
        bootstrap = int(config["analysis"]["bootstrap_samples"])
        inference = [
            paired_user_inference(
                user_rows,
                method="prefix_consistency",
                comparator=comparator,
                bootstrap_samples=bootstrap,
                seed=derive_seed(seed, "inference", index),
            )
            for index, comparator in enumerate(PRIMARY_COMPARATORS)
        ]
        adjusted = holm_adjust(item.sign_flip_pvalue for item in inference)

        stress, stress_calibration = run_stress_panel(seed=seed, config=config)
        stress.to_csv(run.path / "stress_metrics.csv", index=False)
        for row in stress.to_dict("records"):
            dimensions = {key: row.pop(key) for key in ("scenario", "condition")}
            run.record(row, panel="stress", **dimensions)
        expected_pairs = {
            (user_id, task_index)
            for user_id in eval_users
            for task_index in range(int(config["n_eval_tasks_per_user"]))
        }
        observed_pairs = set(
            raw[["user_id", "task_index"]].itertuples(index=False, name=None)
        )
        eligible_baselines = {
            name: means[name] for name in PRIMARY_COMPARATORS if name in means
        }
        strongest_name = max(eligible_baselines, key=eligible_baselines.get)
        strongest_accuracy = float(eligible_baselines[strongest_name])
        consistency_accuracy = float(means["prefix_consistency"])
        stable_rows = stress.loc[
            stress["scenario"] == "stable_correct_vs_stable_wrong"
        ].set_index("condition")
        stable_consistency_accuracy = float(
            stable_rows.loc["prefix_consistency", "accuracy"]
        )
        wrong_lock = float(
            stable_rows.loc["prefix_consistency", "wrong_lock_fraction"]
        )
        correctness_interpretation = (
            "oppose"
            if stable_consistency_accuracy < 1.0 or wrong_lock > 0.0
            else "inconclusive"
        )
        comparative_conclusion = (
            "oppose"
            if consistency_accuracy <= strongest_accuracy
            else "inconclusive"
        )
        summary = {
            "experiment": EXPERIMENT,
            "protocol_version": PROTOCOL_VERSION,
            "profile": config["profile"],
            "evidence_provenance": config["evidence_provenance"],
            "seed": seed,
            "selected_temporal_retention": selected_retention,
            "selected_fixed_action": fixed_action,
            "selected_fixed_name": ACTUATOR_NAMES[fixed_action],
            "selected_prefix_action": prefix_action,
            "selected_prefix_name": ACTUATOR_NAMES[prefix_action],
            "calibration": _calibration_dict(calibration),
            "condition_user_mean_accuracy": means,
            "paired_user_inference": [asdict(item) for item in inference],
            "holm_adjusted_pvalues": adjusted.tolist(),
            "strongest_eligible_baseline": strongest_name,
            "strongest_eligible_baseline_accuracy": strongest_accuracy,
            "prefix_consistency_accuracy": consistency_accuracy,
            "comparative_conclusion": comparative_conclusion,
            "comparative_reason": (
                "prefix consistency did not exceed the strongest registered causal baseline"
                if comparative_conclusion == "oppose"
                else "positive retrospective evidence cannot confirm the title claim"
            ),
            "correctness_interpretation_conclusion": correctness_interpretation,
            "stable_wrong_control": {
                "prefix_consistency_accuracy": stable_consistency_accuracy,
                "wrong_lock_fraction": wrong_lock,
            },
            "stress_calibration": _calibration_dict(stress_calibration),
            "coverage": {
                "complete": observed_pairs == expected_pairs,
                "expected_users": len(eval_users),
                "observed_users": len(set(raw["user_id"])),
                "expected_user_tasks": len(expected_pairs),
                "observed_user_tasks": len(observed_pairs),
            },
            "feature_cache": {
                "selection": selection_store.cache_stats,
                "evaluation": eval_store.cache_stats,
            },
            "statistical_unit": "user",
            "claim_upgrade_allowed": False,
        }
        (run.path / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return run.path


def main() -> None:
    parser = basic_parser(
        __doc__ or EXPERIMENT,
        "configs/smoke/exp35_prefix_reliability_audit.json",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    seeds = seed_list(args.seeds if args.seeds is not None else config["seeds"])
    for seed in seeds:
        print(run_seed(config, seed=seed, results_root=args.results_root))


if __name__ == "__main__":
    main()
