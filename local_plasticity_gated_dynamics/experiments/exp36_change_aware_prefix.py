"""Prospective external audit of forgetting and causal change-aware prefix state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.analysis.change_point_metrics import change_point_metrics
from src.data.orbit_streaming import (
    OrbitEmbeddingEpisode,
    OrbitEpisodeSamplingConfig,
    OrbitFeatureStore,
    OrbitQueryObservation,
    validate_user_disjoint_stores,
)
from src.models.change_aware_prefix import (
    AccumulatorTrace,
    JSDChangeConfig,
    circularly_shift_resets,
    fixed_forgetting_accumulator,
    jsd_change_accumulator,
    scheduled_reset_accumulator,
    sliding_window_accumulator,
)
from src.models.prefix_reliability import action_probabilities
from src.models.streaming_fewshot_actuators import (
    PersonalizedStreamingActuators,
    StreamingActuatorConfig,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


EXPERIMENT = "exp36_change_aware_prefix"
PROTOCOL_VERSION = "exp36_change_aware_prefix_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    "current_frame",
    "cumulative",
    "fixed_forgetting",
    "sliding_window",
    "jsd_change_reset",
    "jsd_score_no_reset",
    "matched_shifted_reset",
    "oracle_change_reset",
)
PANELS = ("natural", "hidden_switch")


@dataclass(frozen=True, slots=True, eq=False)
class PreparedStream:
    unit_id: str
    task_index: int
    panel: str
    evidence: np.ndarray
    labels: np.ndarray
    stream_ids: np.ndarray
    switch_flags: np.ndarray
    source_video_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        evidence = np.asarray(self.evidence, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.int64)
        streams = np.asarray(self.stream_ids, dtype=str)
        switches = np.asarray(self.switch_flags, dtype=np.bool_)
        if self.panel not in PANELS:
            raise ValueError(f"unknown panel: {self.panel}")
        if evidence.ndim != 2 or evidence.shape[0] == 0 or evidence.shape[1] < 2:
            raise ValueError("prepared evidence must have shape [frame, class>=2]")
        if labels.shape != (evidence.shape[0],):
            raise ValueError("prepared labels must align with evidence")
        if streams.shape != labels.shape or switches.shape != labels.shape:
            raise ValueError("prepared stream metadata must align with labels")
        if not np.all(np.isfinite(evidence)) or np.any(evidence < 0.0):
            raise ValueError("prepared evidence must be finite and non-negative")
        if not np.allclose(np.sum(evidence, axis=1), 1.0, atol=1e-6):
            raise ValueError("prepared evidence rows must sum to one")
        if np.any(labels < 0) or np.any(labels >= evidence.shape[1]):
            raise ValueError("prepared labels fall outside the class range")
        if not self.source_video_ids:
            raise ValueError("prepared stream needs source video provenance")
        for name, value, dtype in (
            ("evidence", evidence, np.float64),
            ("labels", labels, np.int64),
            ("stream_ids", streams, str),
            ("switch_flags", switches, np.bool_),
        ):
            result = np.array(value, dtype=dtype, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_preregistration(config_path: Path) -> dict[str, Any]:
    """Verify the frozen external config and protocol before outcome access."""

    receipt_path = PROJECT_ROOT / "provenance/exp36_preregistration_receipt_20260726.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("Exp36 preregistration receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp36 receipt protocol mismatch")
    checks = (
        (PROJECT_ROOT / str(receipt["protocol_path"]), "protocol_sha256"),
        (config_path.resolve(), "config_sha256"),
        (PROJECT_ROOT / str(receipt["cohort_path"]), "cohort_sha256"),
    )
    for path, key in checks:
        if not path.is_file() or _sha256(path) != str(receipt[key]):
            raise ValueError(f"Exp36 preregistration hash mismatch: {path}")
    return receipt


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _validate_grid(value: Any, *, name: str, lower: float, upper: float) -> tuple[float, ...]:
    grid = np.asarray(value, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must be a non-empty finite grid")
    if np.any((grid < lower) | (grid > upper)) or len(set(grid.tolist())) != grid.size:
        raise ValueError(f"{name} lies outside [{lower}, {upper}] or has duplicates")
    return tuple(float(item) for item in grid)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp36 protocol version mismatch")
    if config.get("profile") not in {"smoke", "development", "prospective_external"}:
        raise ValueError("Exp36 profile must be smoke, development, or prospective_external")
    if config.get("evidence_provenance") not in {
        "development_scale_gate",
        "prospective_external_confirmation",
    }:
        raise ValueError("Exp36 evidence provenance is invalid")
    for key in ("used_external_labels_for_fit", "used_future_frames", "used_autograd", "used_bptt"):
        if config.get(key) is not False:
            raise ValueError(f"Exp36 requires {key}=false")
    if config.get("used_development_labels_for_selection") is not True:
        raise ValueError("Exp36 must disclose development-label selection")
    if config.get("profile") == "prospective_external":
        if config.get("evidence_provenance") != "prospective_external_confirmation":
            raise ValueError("prospective profile requires prospective provenance")
        if config.get("claim_upgrade_allowed") is not True:
            raise ValueError("prospective profile must state claim_upgrade_allowed=true")
        expected_collectors = tuple(f"P{index}" for index in range(1, 13))
        if tuple(config.get("external_collectors", ())) != expected_collectors:
            raise ValueError("prospective Exp36 requires frozen collectors P1--P12")
    for key in ("n_development_tasks_per_user", "n_external_tasks_per_collector"):
        _positive_int(config, key)
    _validate_grid(config.get("retention_grid"), name="retention_grid", lower=0.0, upper=1.0)
    windows = config.get("window_grid", ())
    if not windows or any(isinstance(value, bool) or int(value) < 1 for value in windows):
        raise ValueError("window_grid must contain positive integers")
    detector = config.get("detector_grid", {})
    _validate_grid(detector.get("fast_retention"), name="fast_retention", lower=0.0, upper=0.999999)
    _validate_grid(detector.get("jsd_threshold"), name="jsd_threshold", lower=0.0, upper=float(np.log(2.0)))
    for key in ("patience", "min_run_frames"):
        values = detector.get(key, ())
        if not values or any(isinstance(value, bool) or int(value) < 1 for value in values):
            raise ValueError(f"detector {key} must contain positive integers")
    stream = config.get("stream", {})
    for key in ("segments_per_stream", "segment_frames", "post_switch_window", "detection_tolerance"):
        _positive_int(stream, key)
    if int(stream["segments_per_stream"]) < 2:
        raise ValueError("hidden stream needs at least two segments")
    analysis = config.get("analysis", {})
    if int(analysis.get("bootstrap_samples", 0)) < 100:
        raise ValueError("analysis.bootstrap_samples must be at least 100")
    _sampling_config(config)
    StreamingActuatorConfig(**dict(config["actuators"]))


def _sampling_config(config: Mapping[str, Any]) -> OrbitEpisodeSamplingConfig:
    return OrbitEpisodeSamplingConfig(**dict(config["sampling"]))


def _fit_actuators(
    episode: OrbitEmbeddingEpisode, config: Mapping[str, Any]
) -> PersonalizedStreamingActuators:
    return PersonalizedStreamingActuators.fit(
        episode.support,
        n_classes=episode.n_classes,
        config=StreamingActuatorConfig(**dict(config["actuators"])),
    )


def _equal_probability_evidence(
    fitted: PersonalizedStreamingActuators, observation: OrbitQueryObservation
) -> np.ndarray:
    trace = fitted.trace(observation)
    probabilities = np.asarray(action_probabilities(trace.scores), dtype=np.float64)
    evidence = np.mean(probabilities, axis=1)
    evidence /= np.sum(evidence, axis=1, keepdims=True)
    return evidence


def _natural_stream(
    episode: OrbitEmbeddingEpisode, fitted: PersonalizedStreamingActuators
) -> PreparedStream:
    evidence = _equal_probability_evidence(fitted, episode.query_observation)
    return PreparedStream(
        unit_id=episode.user_id,
        task_index=episode.task_index,
        panel="natural",
        evidence=evidence,
        labels=episode.query_labels,
        stream_ids=episode.query_video_ids,
        switch_flags=np.zeros(episode.query_labels.size, dtype=np.bool_),
        source_video_ids=tuple(map(str, np.unique(episode.query_video_ids))),
    )


def build_hidden_switch_stream(
    episode: OrbitEmbeddingEpisode,
    fitted: PersonalizedStreamingActuators,
    *,
    seed: int,
    segments_per_stream: int,
    segment_frames: int,
) -> PreparedStream:
    """Concatenate real different-class segments while hiding their boundaries."""

    candidates: dict[int, list[np.ndarray]] = {}
    video_names: dict[tuple[int, int], str] = {}
    for video_id_raw in np.unique(episode.query_video_ids):
        video_id = str(video_id_raw)
        indices = np.flatnonzero(episode.query_video_ids == video_id)
        labels = np.unique(episode.query_labels[indices])
        if labels.size != 1:
            raise ValueError(f"source video {video_id} has multiple query labels")
        label = int(labels[0])
        if indices.size >= segment_frames:
            candidates.setdefault(label, []).append(indices)
            video_names[(label, len(candidates[label]) - 1)] = video_id
    eligible_labels = np.asarray(sorted(candidates), dtype=np.int64)
    if eligible_labels.size < segments_per_stream:
        raise ValueError(
            f"{episode.user_id} task {episode.task_index} has only "
            f"{eligible_labels.size} eligible switch classes"
        )
    rng = np.random.default_rng(
        derive_seed(seed, "exp36-hidden", episode.user_id, episode.task_index)
    )
    chosen_labels = rng.choice(
        eligible_labels, size=segments_per_stream, replace=False
    )
    embedding_segments: list[np.ndarray] = []
    label_segments: list[np.ndarray] = []
    source_videos: list[str] = []
    for label_raw in chosen_labels:
        label = int(label_raw)
        video_index = int(rng.integers(0, len(candidates[label])))
        indices = candidates[label][video_index]
        start = int(rng.integers(0, indices.size - segment_frames + 1))
        chosen = indices[start : start + segment_frames]
        embedding_segments.append(episode.query_embeddings[chosen])
        label_segments.append(np.full(segment_frames, label, dtype=np.int64))
        source_videos.append(video_names[(label, video_index)])
    embeddings = np.concatenate(embedding_segments, axis=0)
    labels = np.concatenate(label_segments)
    n_frames = labels.size
    hidden_id = f"hidden::{episode.user_id}::{episode.task_index}"
    observation = OrbitQueryObservation(
        embeddings=embeddings,
        video_ids=np.repeat(np.asarray([hidden_id]), n_frames),
        frame_indices=np.arange(n_frames, dtype=np.int64),
    )
    evidence = _equal_probability_evidence(fitted, observation)
    switches = np.zeros(n_frames, dtype=np.bool_)
    switches[np.arange(1, segments_per_stream) * segment_frames] = True
    return PreparedStream(
        unit_id=episode.user_id,
        task_index=episode.task_index,
        panel="hidden_switch",
        evidence=evidence,
        labels=labels,
        stream_ids=observation.video_ids,
        switch_flags=switches,
        source_video_ids=tuple(source_videos),
    )


def _sample_episode(
    store: OrbitFeatureStore,
    unit_id: str,
    task_index: int,
    *,
    seed: int,
    phase: str,
    config: Mapping[str, Any],
) -> OrbitEmbeddingEpisode:
    return store.sample_episode(
        unit_id,
        seed=derive_seed(seed, EXPERIMENT, phase, unit_id, task_index),
        task_index=task_index,
        config=_sampling_config(config),
    )


def _prepare_streams(
    store: OrbitFeatureStore,
    units: Iterable[str],
    *,
    n_tasks: int,
    seed: int,
    phase: str,
    config: Mapping[str, Any],
) -> list[PreparedStream]:
    stream_config = config["stream"]
    prepared: list[PreparedStream] = []
    for unit_id in units:
        for task_index in range(n_tasks):
            episode = _sample_episode(
                store,
                unit_id,
                task_index,
                seed=seed,
                phase=phase,
                config=config,
            )
            fitted = _fit_actuators(episode, config)
            prepared.append(_natural_stream(episode, fitted))
            prepared.append(
                build_hidden_switch_stream(
                    episode,
                    fitted,
                    seed=derive_seed(seed, phase, unit_id, task_index),
                    segments_per_stream=int(stream_config["segments_per_stream"]),
                    segment_frames=int(stream_config["segment_frames"]),
                )
            )
    return prepared


def _accuracy(trace: AccumulatorTrace, stream: PreparedStream) -> tuple[int, int]:
    return int(np.sum(trace.predictions == stream.labels)), int(stream.labels.size)


def _select_fixed_forgetting(
    streams: list[PreparedStream], grid: Iterable[float]
) -> tuple[float, list[dict[str, Any]]]:
    hidden = [stream for stream in streams if stream.panel == "hidden_switch"]
    rows: list[dict[str, Any]] = []
    for retention in grid:
        correct = frames = 0
        for stream in hidden:
            trace = fixed_forgetting_accumulator(
                stream.evidence,
                stream_ids=stream.stream_ids,
                retention=float(retention),
            )
            value, count = _accuracy(trace, stream)
            correct += value
            frames += count
        rows.append(
            {
                "candidate_type": "fixed_forgetting",
                "retention": float(retention),
                "hidden_switch_accuracy": correct / frames,
                "eligible": True,
            }
        )
    selected = max(rows, key=lambda row: (row["hidden_switch_accuracy"], row["retention"]))
    return float(selected["retention"]), rows


def _select_window(
    streams: list[PreparedStream], grid: Iterable[int]
) -> tuple[int, list[dict[str, Any]]]:
    hidden = [stream for stream in streams if stream.panel == "hidden_switch"]
    rows: list[dict[str, Any]] = []
    for window in grid:
        correct = frames = 0
        for stream in hidden:
            trace = sliding_window_accumulator(
                stream.evidence,
                stream_ids=stream.stream_ids,
                window_frames=int(window),
            )
            value, count = _accuracy(trace, stream)
            correct += value
            frames += count
        rows.append(
            {
                "candidate_type": "sliding_window",
                "window_frames": int(window),
                "hidden_switch_accuracy": correct / frames,
                "eligible": True,
            }
        )
    selected = max(rows, key=lambda row: (row["hidden_switch_accuracy"], row["window_frames"]))
    return int(selected["window_frames"]), rows


def _detector_grid(config: Mapping[str, Any]) -> Iterable[JSDChangeConfig]:
    grid = config["detector_grid"]
    for fast, threshold, patience, minimum in product(
        grid["fast_retention"],
        grid["jsd_threshold"],
        grid["patience"],
        grid["min_run_frames"],
    ):
        yield JSDChangeConfig(
            fast_retention=float(fast),
            jsd_threshold=float(threshold),
            patience=int(patience),
            min_run_frames=int(minimum),
        )


def _select_detector(
    streams: list[PreparedStream], config: Mapping[str, Any]
) -> tuple[JSDChangeConfig, list[dict[str, Any]]]:
    natural = [stream for stream in streams if stream.panel == "natural"]
    hidden = [stream for stream in streams if stream.panel == "hidden_switch"]
    cumulative_correct = cumulative_frames = 0
    for stream in natural:
        trace = fixed_forgetting_accumulator(
            stream.evidence, stream_ids=stream.stream_ids, retention=1.0
        )
        value, count = _accuracy(trace, stream)
        cumulative_correct += value
        cumulative_frames += count
    cumulative_accuracy = cumulative_correct / cumulative_frames
    constraints = config["selection_constraints"]
    rows: list[dict[str, Any]] = []
    for candidate in _detector_grid(config):
        natural_correct = natural_frames = natural_alarms = 0
        for stream in natural:
            trace = jsd_change_accumulator(
                stream.evidence, stream_ids=stream.stream_ids, config=candidate
            )
            value, count = _accuracy(trace, stream)
            natural_correct += value
            natural_frames += count
            natural_alarms += int(np.sum(trace.alarm_flags))
        hidden_correct = hidden_frames = 0
        delays: list[float] = []
        for stream in hidden:
            trace = jsd_change_accumulator(
                stream.evidence, stream_ids=stream.stream_ids, config=candidate
            )
            value, count = _accuracy(trace, stream)
            hidden_correct += value
            hidden_frames += count
            metrics = change_point_metrics(
                trace.predictions,
                stream.labels,
                alarm_flags=trace.alarm_flags,
                switch_flags=stream.switch_flags,
                post_switch_window=int(config["stream"]["post_switch_window"]),
                detection_tolerance=int(config["stream"]["detection_tolerance"]),
            )
            if np.isfinite(metrics.median_detection_delay):
                delays.append(metrics.median_detection_delay)
        natural_accuracy = natural_correct / natural_frames
        false_alarm_rate = 1000.0 * natural_alarms / natural_frames
        delay = float(np.mean(delays)) if delays else float("inf")
        eligible = (
            cumulative_accuracy - natural_accuracy
            <= float(constraints["max_natural_accuracy_loss"])
            and false_alarm_rate
            <= float(constraints["max_false_alarms_per_1000"])
        )
        rows.append(
            {
                "candidate_type": "jsd_change_reset",
                **asdict(candidate),
                "natural_accuracy": natural_accuracy,
                "natural_accuracy_loss": cumulative_accuracy - natural_accuracy,
                "natural_false_alarms_per_1000": false_alarm_rate,
                "hidden_switch_accuracy": hidden_correct / hidden_frames,
                "mean_detection_delay": delay if np.isfinite(delay) else None,
                "eligible": bool(eligible),
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        raise RuntimeError("no JSD detector satisfies the frozen validation constraints")

    def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        delay = float(row["mean_detection_delay"]) if row["mean_detection_delay"] is not None else float("inf")
        parameters = (
            float(row["fast_retention"]),
            float(row["jsd_threshold"]),
            int(row["patience"]),
            int(row["min_run_frames"]),
        )
        return (
            -float(row["hidden_switch_accuracy"]),
            delay,
            float(row["natural_false_alarms_per_1000"]),
            parameters,
        )

    selected = min(eligible_rows, key=selection_key)
    return (
        JSDChangeConfig(
            fast_retention=float(selected["fast_retention"]),
            jsd_threshold=float(selected["jsd_threshold"]),
            patience=int(selected["patience"]),
            min_run_frames=int(selected["min_run_frames"]),
        ),
        rows,
    )


def fit_development_selection(
    store: OrbitFeatureStore,
    *,
    seed: int,
    config: Mapping[str, Any],
) -> tuple[float, int, JSDChangeConfig, pd.DataFrame]:
    streams = _prepare_streams(
        store,
        store.users,
        n_tasks=int(config["n_development_tasks_per_user"]),
        seed=seed,
        phase="development",
        config=config,
    )
    retention, retention_rows = _select_fixed_forgetting(
        streams, config["retention_grid"]
    )
    window, window_rows = _select_window(streams, config["window_grid"])
    detector, detector_rows = _select_detector(streams, config)
    audit = pd.DataFrame([*retention_rows, *window_rows, *detector_rows])
    audit["selected"] = False
    audit.loc[
        (audit["candidate_type"] == "fixed_forgetting")
        & (audit["retention"] == retention),
        "selected",
    ] = True
    audit.loc[
        (audit["candidate_type"] == "sliding_window")
        & (audit["window_frames"] == window),
        "selected",
    ] = True
    detector_mask = audit["candidate_type"] == "jsd_change_reset"
    for key, value in asdict(detector).items():
        detector_mask &= audit[key] == value
    audit.loc[detector_mask, "selected"] = True
    return retention, window, detector, audit


def _condition_traces(
    stream: PreparedStream,
    *,
    fixed_retention: float,
    window_frames: int,
    detector: JSDChangeConfig,
    seed: int,
) -> dict[str, AccumulatorTrace]:
    current = fixed_forgetting_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, retention=0.0
    )
    cumulative = fixed_forgetting_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, retention=1.0
    )
    forgetting = fixed_forgetting_accumulator(
        stream.evidence,
        stream_ids=stream.stream_ids,
        retention=fixed_retention,
    )
    window = sliding_window_accumulator(
        stream.evidence,
        stream_ids=stream.stream_ids,
        window_frames=window_frames,
    )
    change = jsd_change_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, config=detector
    )
    score_only = jsd_change_accumulator(
        stream.evidence,
        stream_ids=stream.stream_ids,
        config=detector,
        enable_reset=False,
    )
    if stream.labels.size < 2:
        raise ValueError("matched shifted reset needs at least two frames")
    offset = 1 + derive_seed(
        seed, "matched-reset", stream.unit_id, stream.task_index, stream.panel
    ) % (stream.labels.size - 1)
    shifted = circularly_shift_resets(
        change.reset_flags,
        stream_ids=stream.stream_ids,
        offset=int(offset),
    )
    matched = scheduled_reset_accumulator(
        stream.evidence,
        stream_ids=stream.stream_ids,
        reset_schedule=shifted,
    )
    oracle = scheduled_reset_accumulator(
        stream.evidence,
        stream_ids=stream.stream_ids,
        reset_schedule=stream.switch_flags,
    )
    return {
        "current_frame": current,
        "cumulative": cumulative,
        "fixed_forgetting": forgetting,
        "sliding_window": window,
        "jsd_change_reset": change,
        "jsd_score_no_reset": score_only,
        "matched_shifted_reset": matched,
        "oracle_change_reset": oracle,
    }


def evaluate_prepared_stream(
    stream: PreparedStream,
    *,
    fixed_retention: float,
    window_frames: int,
    detector: JSDChangeConfig,
    seed: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    traces = _condition_traces(
        stream,
        fixed_retention=fixed_retention,
        window_frames=window_frames,
        detector=detector,
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    for condition, trace in traces.items():
        metrics = change_point_metrics(
            trace.predictions,
            stream.labels,
            alarm_flags=trace.alarm_flags,
            switch_flags=stream.switch_flags,
            post_switch_window=int(config["stream"]["post_switch_window"]),
            detection_tolerance=int(config["stream"]["detection_tolerance"]),
        )
        rows.append(
            {
                "unit_id": stream.unit_id,
                "task_index": stream.task_index,
                "panel": stream.panel,
                "condition": condition,
                **asdict(metrics),
                "n_resets": int(np.sum(trace.reset_flags)),
                "mean_state_l1": float(np.mean(trace.state_l1)),
                "source_video_ids": "|".join(stream.source_video_ids),
                "status": "complete",
            }
        )
    return pd.DataFrame(rows)


def _development_store(config: Mapping[str, Any]) -> OrbitFeatureStore:
    return OrbitFeatureStore(
        str(config["development_feature_root"]),
        split=str(config["development_split"]),
        official_splits_path=PROJECT_ROOT / str(config["development_splits_path"]),
        require_complete_split=bool(config.get("require_complete_development_split", False)),
        cache_videos=bool(config.get("cache_features_in_memory", False)),
    )


def _external_store(config: Mapping[str, Any]) -> OrbitFeatureStore:
    return OrbitFeatureStore(
        str(config["external_feature_root"]),
        split=str(config["external_split"]),
        external_cohort_path=PROJECT_ROOT / str(config["external_cohort_path"]),
        require_complete_split=bool(config.get("require_complete_external_cohort", False)),
        cache_videos=bool(config.get("cache_features_in_memory", False)),
    )


def run_development_seed(
    config: Mapping[str, Any], *, seed: int, results_root: str | Path
) -> Path:
    """Run the preregistered selection gate without opening external data."""

    validate_config(config)
    if config.get("profile") != "development":
        raise ValueError("run_development_seed requires profile=development")
    if config.get("claim_upgrade_allowed") is not False:
        raise ValueError("development runs cannot upgrade claims")
    initialize_seed(seed)
    store = _development_store(config)
    run_config = dict(config)
    run_config["development_users"] = list(store.users)
    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
        run_label="development",
    ) as run:
        run.register_conditions([{"stage": "selection"}])
        retention, window, detector, audit = fit_development_selection(
            store, seed=seed, config=config
        )
        audit.to_csv(run.path / "development_selection_audit.csv", index=False)
        eligible_detector = audit.loc[
            (audit["candidate_type"] == "jsd_change_reset")
            & audit["eligible"].astype(bool)
        ]
        selected = {
            "fixed_retention": retention,
            "window_frames": window,
            "detector": asdict(detector),
            "n_eligible_detector_candidates": int(len(eligible_detector)),
            "used_external_data": False,
        }
        (run.path / "selected_hyperparameters.json").write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run.record(
            {
                "status": "complete",
                "fixed_retention": retention,
                "window_frames": window,
                "n_eligible_detector_candidates": int(len(eligible_detector)),
            },
            stage="selection",
        )
        summary = {
            "experiment": EXPERIMENT,
            "protocol_version": PROTOCOL_VERSION,
            "profile": "development",
            "evidence_provenance": "development_scale_gate",
            "claim_upgrade_allowed": False,
            "seed": seed,
            "selected_hyperparameters": selected,
            "development_users": list(store.users),
            "feature_cache": store.cache_stats,
            "statistical_unit": "user",
        }
        (run.path / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return run.path


def run_seed(
    config: Mapping[str, Any], *, seed: int, results_root: str | Path
) -> Path:
    validate_config(config)
    initialize_seed(seed)
    development_store = _development_store(config)
    external_store = _external_store(config)
    validate_user_disjoint_stores((development_store, external_store))
    expected_collectors = tuple(map(str, config.get("external_collectors", ())))
    if expected_collectors and external_store.users != tuple(sorted(expected_collectors)):
        if set(external_store.users) != set(expected_collectors):
            raise ValueError("external feature store does not match frozen collectors")
    run_config = dict(config)
    run_config["development_users"] = list(development_store.users)
    run_config["external_collectors_observed"] = list(external_store.users)
    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
        run_label=str(config["profile"]),
    ) as run:
        n_tasks = int(config["n_external_tasks_per_collector"])
        run.register_conditions(
            [
                {
                    "collector_id": collector,
                    "task_index": task_index,
                    "panel": panel,
                    "condition": condition,
                }
                for collector in external_store.users
                for task_index in range(n_tasks)
                for panel in PANELS
                for condition in CONDITIONS
            ]
        )
        retention, window, detector, selection_audit = fit_development_selection(
            development_store, seed=seed, config=config
        )
        selection_audit.to_csv(run.path / "development_selection_audit.csv", index=False)
        selected_payload = {
            "fixed_retention": retention,
            "window_frames": window,
            "detector": asdict(detector),
            "development_users": list(development_store.users),
            "used_external_labels": False,
        }
        (run.path / "selected_hyperparameters.json").write_text(
            json.dumps(selected_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raw_frames: list[pd.DataFrame] = []
        for collector in external_store.users:
            for task_index in range(n_tasks):
                try:
                    episode = _sample_episode(
                        external_store,
                        collector,
                        task_index,
                        seed=seed,
                        phase="external",
                        config=config,
                    )
                    fitted = _fit_actuators(episode, config)
                    streams = (
                        _natural_stream(episode, fitted),
                        build_hidden_switch_stream(
                            episode,
                            fitted,
                            seed=derive_seed(seed, "external", collector, task_index),
                            segments_per_stream=int(config["stream"]["segments_per_stream"]),
                            segment_frames=int(config["stream"]["segment_frames"]),
                        ),
                    )
                    for stream in streams:
                        frame = evaluate_prepared_stream(
                            stream,
                            fixed_retention=retention,
                            window_frames=window,
                            detector=detector,
                            seed=seed,
                            config=config,
                        )
                        raw_frames.append(frame)
                        for row in frame.to_dict("records"):
                            dimensions = {
                                "collector_id": row.pop("unit_id"),
                                "task_index": row.pop("task_index"),
                                "panel": row.pop("panel"),
                                "condition": row.pop("condition"),
                            }
                            run.record(row, **dimensions)
                except Exception as error:
                    for panel in PANELS:
                        for condition in CONDITIONS:
                            run.mark_condition_failure(
                                error,
                                collector_id=collector,
                                task_index=task_index,
                                panel=panel,
                                condition=condition,
                            )
        if not raw_frames:
            raise RuntimeError("all Exp36 external tasks failed")
        raw = pd.concat(raw_frames, ignore_index=True)
        raw.to_csv(run.path / "external_task_metrics.csv", index=False)
        expected = {
            (collector, task, panel, condition)
            for collector in external_store.users
            for task in range(n_tasks)
            for panel in PANELS
            for condition in CONDITIONS
        }
        observed = set(
            raw[["unit_id", "task_index", "panel", "condition"]].itertuples(
                index=False, name=None
            )
        )
        condition_means = (
            raw.groupby(["panel", "condition"], as_index=False)["accuracy"]
            .mean()
            .to_dict("records")
        )
        summary = {
            "experiment": EXPERIMENT,
            "protocol_version": PROTOCOL_VERSION,
            "profile": config["profile"],
            "evidence_provenance": config["evidence_provenance"],
            "claim_upgrade_allowed": bool(config.get("claim_upgrade_allowed", False)),
            "seed": seed,
            "selected_hyperparameters": selected_payload,
            "coverage": {
                "complete": observed == expected,
                "expected_conditions": len(expected),
                "observed_conditions": len(observed),
                "expected_collectors": len(expected_collectors),
                "observed_collectors": len(set(raw["unit_id"])),
            },
            "condition_task_mean_accuracy": condition_means,
            "feature_cache": {
                "development": development_store.cache_stats,
                "external": external_store.cache_stats,
            },
            "statistical_unit": "collector",
        }
        (run.path / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return run.path


def main() -> None:
    parser = basic_parser(
        __doc__ or EXPERIMENT,
        "configs/prospective/exp36_change_aware_prefix.json",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_json_config(config_path)
    if config.get("profile") == "prospective_external":
        validate_preregistration(config_path)
    seeds = seed_list(args.seeds if args.seeds is not None else config["seeds"])
    for seed in seeds:
        if config.get("profile") == "development":
            path = run_development_seed(
                config, seed=seed, results_root=args.results_root
            )
        else:
            path = run_seed(config, seed=seed, results_root=args.results_root)
        print(path)


if __name__ == "__main__":
    main()
