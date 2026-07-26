"""Prospective staged Stream-51 test of causal continuous memory control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.analysis.memory_demand_metrics import (
    qualify_memory_demand,
    source_video_accuracy,
    switch_reachability,
)
from src.data.stream51_streaming import (
    Stream51FeatureStore,
    Stream51Stream,
    fit_stream51_vmf,
    make_stream51_hidden_streams,
    make_stream51_natural_streams,
)
from src.models.change_aware_prefix import (
    fixed_forgetting_accumulator,
    sliding_window_accumulator,
)
from src.models.embedding_evidence import VMFEvidenceModel
from src.models.soft_memory_controller import (
    ControllerStandardizer,
    SoftMemoryConfig,
    accumulate_with_retention,
    causal_control_features,
    controller_retention,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


EXPERIMENT = "exp38_stream51_soft_memory"
PROTOCOL_VERSION = "exp38_stream51_soft_memory_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    "current_frame",
    "cumulative",
    "fixed_forgetting",
    "sliding_window",
    "soft_memory",
    "hard_memory",
    "matched_shifted_memory",
    "oracle_memory",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_grid(
    value: Any, *, name: str, lower: float, upper: float
) -> tuple[float, ...]:
    grid = np.asarray(value, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    if np.any((grid < lower) | (grid > upper)):
        raise ValueError(f"{name} contains an out-of-range value")
    if len(set(grid.tolist())) != len(grid):
        raise ValueError(f"{name} contains duplicate values")
    return tuple(float(item) for item in grid)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp38 protocol mismatch")
    if config.get("profile") != "prospective_external_staged":
        raise ValueError("Exp38 requires the prospective staged profile")
    if config.get("evidence_provenance") != "prospective_stream51_video_holdout":
        raise ValueError("Exp38 evidence provenance mismatch")
    for key in (
        "used_external_labels_for_fit",
        "used_future_frames",
        "used_autograd",
        "used_bptt",
    ):
        if config.get(key) is not False:
            raise ValueError(f"Exp38 requires {key}=false")
    for key in (
        "used_support_labels_for_evidence_model",
        "used_development_labels_for_selection",
        "used_qualification_labels_for_gating",
    ):
        if config.get(key) is not True:
            raise ValueError(f"Exp38 requires {key}=true disclosure")
    if len(seed_list(config["seeds"])) != 5:
        raise ValueError("Exp38 requires five registered assembly seeds")
    if int(config.get("archive_content_length", -1)) != 11343409620:
        raise ValueError("Exp38 archive content length mismatch")
    if config.get("archive_sha256") != (
        "db2711e34130923147c69e203ebfde46c8d651846958d645359a0ceb4d910465"
    ):
        raise ValueError("Exp38 archive SHA-256 mismatch")
    if config.get("official_ordering_sha256") != (
        "8368c5920486e83746d03acd9d71e71fdc403065d8f5d74df6d8de275b426fd4"
    ):
        raise ValueError("Exp38 ordering SHA-256 mismatch")
    if config.get("official_repo_commit") != "8a066737ac8b3ac6f57987e6b3713ddcfbd1dcbf":
        raise ValueError("Exp38 upstream commit mismatch")
    encoder = config.get("encoder", {})
    for key in ("batch_size", "decode_workers"):
        if isinstance(encoder.get(key), bool) or int(encoder.get(key, 0)) < 1:
            raise ValueError(f"encoder {key} must be a positive integer")
    fraction = float(config.get("development_fit_fraction", np.nan))
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("development_fit_fraction must lie in (0, 1)")
    _finite_grid(
        config["temperature_grid"],
        name="temperature_grid",
        lower=1e-12,
        upper=1e6,
    )
    _finite_grid(
        config["retention_grid"],
        name="retention_grid",
        lower=0.0,
        upper=1.0,
    )
    windows = tuple(int(value) for value in config["window_grid"])
    if not windows or min(windows) < 1 or len(set(windows)) != len(windows):
        raise ValueError("window_grid must contain unique positive integers")
    controller = config["controller_grid"]
    floors = _finite_grid(
        controller["retention_floor"],
        name="controller retention_floor",
        lower=0.0,
        upper=1.0,
    )
    ceilings = _finite_grid(
        controller["retention_ceiling"],
        name="controller retention_ceiling",
        lower=0.0,
        upper=1.0,
    )
    if not any(floor < ceiling for floor in floors for ceiling in ceilings):
        raise ValueError("controller retention grid has no valid interval")
    _finite_grid(controller["gain"], name="controller gain", lower=1e-12, upper=1e3)
    _finite_grid(
        controller["threshold"],
        name="controller threshold",
        lower=-1e3,
        upper=1e3,
    )
    templates = np.asarray(controller["feature_templates"], dtype=np.float64)
    if (
        templates.ndim != 2
        or templates.shape[1] != 3
        or not np.all(np.isfinite(templates))
        or np.any(templates < 0.0)
        or np.any(np.sum(templates, axis=1) <= 0.0)
    ):
        raise ValueError("controller feature templates must be non-negative triples")
    task = config["task"]
    for key in (
        "natural_max_frames",
        "segment_frames",
        "videos_per_stream",
        "post_switch_window",
        "detection_tolerance",
        "refractory_frames",
        "min_run_frames",
    ):
        if isinstance(task.get(key), bool) or int(task.get(key, 0)) < 2:
            raise ValueError(f"task {key} must be an integer of at least two")
    if config["qualification"].get("require_all_seeds") is not True:
        raise ValueError("Exp38 external access requires all seeds to qualify")


def validate_preregistration(config_path: Path) -> dict[str, Any]:
    receipt_path = PROJECT_ROOT / "provenance/exp38_preregistration_receipt_20260726.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("Exp38 preregistration receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp38 preregistration version mismatch")
    checks = (
        (PROJECT_ROOT / str(receipt["protocol_path"]), "protocol_sha256"),
        (config_path.resolve(), "config_sha256"),
        (PROJECT_ROOT / str(receipt["cohort_path"]), "cohort_sha256"),
    )
    for path, key in checks:
        if not path.is_file() or _sha256(path) != str(receipt[key]):
            raise ValueError(f"Exp38 preregistration hash mismatch: {path}")
    return receipt


def validate_implementation_receipt() -> dict[str, Any]:
    path = PROJECT_ROOT / "provenance/exp38_implementation_receipt_20260726.json"
    if not path.is_file():
        raise FileNotFoundError("Exp38 implementation receipt is missing")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp38 implementation receipt mismatch")
    for relative, expected in receipt.get("files", {}).items():
        source = PROJECT_ROOT / str(relative)
        if not source.is_file() or _sha256(source) != str(expected):
            raise ValueError(f"Exp38 implementation hash mismatch: {source}")
    return receipt


def _store(config: Mapping[str, Any], *, include_external: bool) -> Stream51FeatureStore:
    # Qualification can see support and development only.  After qualification,
    # deployment parameters come from the frozen receipt, so the external stage
    # deliberately does not load development features again.
    splits = ("support", "external") if include_external else ("support", "development")
    return Stream51FeatureStore(
        config["feature_root"],
        cohort_path=PROJECT_ROOT / str(config["cohort_path"]),
        required_splits=splits,
        cache_in_memory=True,
    )


def partition_development_videos(
    store: Stream51FeatureStore, *, salt: str, fit_fraction: float
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Create a deterministic within-class fit/qualification partition."""

    if not isinstance(salt, str) or not salt:
        raise ValueError("development partition salt must be non-empty")
    fit_fraction = float(fit_fraction)
    if not np.isfinite(fit_fraction) or not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in (0, 1)")
    by_class: dict[int, list[str]] = {}
    for key in store.video_keys("development"):
        by_class.setdefault(store.class_id(key), []).append(key)
    fit: list[str] = []
    qualification: list[str] = []
    for class_id, keys in sorted(by_class.items()):
        ranked = sorted(
            keys,
            key=lambda key: hashlib.sha256(
                f"{salt}|{class_id}|{key}".encode()
            ).hexdigest(),
        )
        if len(ranked) < 2:
            raise ValueError(f"development class {class_id} has fewer than two videos")
        boundary = int(np.clip(np.floor(len(ranked) * fit_fraction), 1, len(ranked) - 1))
        fit.extend(ranked[:boundary])
        qualification.extend(ranked[boundary:])
    if set(fit) & set(qualification) or set(fit) | set(qualification) != set(
        store.video_keys("development")
    ):
        raise RuntimeError("development partition is not exhaustive and disjoint")
    return tuple(sorted(fit)), tuple(sorted(qualification))


def select_temperature(
    store: Stream51FeatureStore,
    model: VMFEvidenceModel,
    *,
    video_keys: Sequence[str],
    grid: Iterable[float],
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for temperature in map(float, grid):
        video_nll: list[float] = []
        video_accuracy: list[float] = []
        for key in video_keys:
            features = store.load(key)
            probabilities = model.probabilities(features, temperature=temperature)
            class_id = store.class_id(key)
            video_nll.append(
                -float(np.mean(np.log(np.maximum(probabilities[:, class_id], 1e-12))))
            )
            video_accuracy.append(
                float(np.mean(np.argmax(probabilities, axis=1) == class_id))
            )
        rows.append(
            {
                "temperature": temperature,
                "video_equal_nll": float(np.mean(video_nll)),
                "video_equal_accuracy": float(np.mean(video_accuracy)),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            float(row["video_equal_nll"]),
            -float(row["video_equal_accuracy"]),
            float(row["temperature"]),
        ),
    )
    for row in rows:
        row["selected"] = row is selected
    return float(selected["temperature"]), pd.DataFrame(rows)


def _make_streams(
    store: Stream51FeatureStore,
    model: VMFEvidenceModel,
    *,
    split: str,
    video_keys: Sequence[str],
    temperature: float,
    seed: int,
    task: Mapping[str, Any],
) -> tuple[tuple[Stream51Stream, ...], tuple[Stream51Stream, ...]]:
    natural = make_stream51_natural_streams(
        store,
        model,
        split=split,
        temperature=temperature,
        max_frames=int(task["natural_max_frames"]),
        video_keys=video_keys,
    )
    hidden = make_stream51_hidden_streams(
        store,
        model,
        split=split,
        seed=seed,
        temperature=temperature,
        segment_frames=int(task["segment_frames"]),
        videos_per_stream=int(task["videos_per_stream"]),
        video_keys=video_keys,
    )
    return natural, hidden


@dataclass(frozen=True, slots=True)
class PanelScore:
    video_equal_accuracy: float
    frame_accuracy: float
    video_equal_post_switch_accuracy: float
    n_videos: int
    n_frames: int


def _score_predictions(
    streams: Sequence[Stream51Stream], predictions: Sequence[np.ndarray], *, post_window: int
) -> tuple[PanelScore, pd.DataFrame]:
    if len(streams) != len(predictions) or not streams:
        raise ValueError("streams and predictions must be non-empty and aligned")
    rows: list[pd.DataFrame] = []
    total_correct = 0
    total_frames = 0
    for stream, prediction in zip(streams, predictions, strict=True):
        prediction = np.asarray(prediction, dtype=np.int64)
        rows.append(
            source_video_accuracy(
                prediction,
                stream.labels,
                source_video_ids=stream.source_video_ids,
                switch_flags=stream.switch_flags,
                post_switch_window=post_window,
            )
        )
        total_correct += int(np.sum(prediction == stream.labels))
        total_frames += len(prediction)
    frame = pd.concat(rows, ignore_index=True)
    if frame["source_video_id"].duplicated().any():
        raise ValueError("a source video appears more than once in one panel")
    post = frame["post_switch_accuracy"].dropna()
    score = PanelScore(
        video_equal_accuracy=float(frame["accuracy"].mean()),
        frame_accuracy=float(total_correct / total_frames),
        video_equal_post_switch_accuracy=(
            float(post.mean()) if not post.empty else float("nan")
        ),
        n_videos=int(len(frame)),
        n_frames=int(total_frames),
    )
    return score, frame


def _fixed_predictions(
    streams: Sequence[Stream51Stream], *, retention: float
) -> list[np.ndarray]:
    return [
        fixed_forgetting_accumulator(
            stream.evidence,
            stream_ids=stream.stream_ids,
            retention=float(retention),
        ).predictions
        for stream in streams
    ]


def _window_predictions(
    streams: Sequence[Stream51Stream], *, window_frames: int
) -> list[np.ndarray]:
    return [
        sliding_window_accumulator(
            stream.evidence,
            stream_ids=stream.stream_ids,
            window_frames=int(window_frames),
        ).predictions
        for stream in streams
    ]


def _oracle_reset_prediction(stream: Stream51Stream) -> np.ndarray:
    """Evaluation-only cumulative memory with perfect boundary resets."""

    retention = np.ones(len(stream.labels), dtype=np.float64)
    retention[stream.switch_flags] = 0.0
    return accumulate_with_retention(
        stream.evidence,
        stream_ids=stream.stream_ids,
        retention_value=retention,
    )[0]


def select_stationary_baselines(
    natural: Sequence[Stream51Stream],
    hidden: Sequence[Stream51Stream],
    *,
    config: Mapping[str, Any],
) -> tuple[float, int, pd.DataFrame]:
    post_window = int(config["task"]["post_switch_window"])
    current_natural, _ = _score_predictions(
        natural, _fixed_predictions(natural, retention=0.0), post_window=post_window
    )
    rows: list[dict[str, Any]] = []
    for retention in map(float, config["retention_grid"]):
        natural_score, _ = _score_predictions(
            natural,
            _fixed_predictions(natural, retention=retention),
            post_window=post_window,
        )
        hidden_score, _ = _score_predictions(
            hidden,
            _fixed_predictions(hidden, retention=retention),
            post_window=post_window,
        )
        rows.append(
            {
                "candidate_type": "fixed_forgetting",
                "retention": retention,
                "window_frames": None,
                "natural_accuracy": natural_score.video_equal_accuracy,
                "hidden_accuracy": hidden_score.video_equal_accuracy,
                "post_switch_accuracy": hidden_score.video_equal_post_switch_accuracy,
            }
        )
    for window in map(int, config["window_grid"]):
        natural_score, _ = _score_predictions(
            natural,
            _window_predictions(natural, window_frames=window),
            post_window=post_window,
        )
        hidden_score, _ = _score_predictions(
            hidden,
            _window_predictions(hidden, window_frames=window),
            post_window=post_window,
        )
        rows.append(
            {
                "candidate_type": "sliding_window",
                "retention": None,
                "window_frames": window,
                "natural_accuracy": natural_score.video_equal_accuracy,
                "hidden_accuracy": hidden_score.video_equal_accuracy,
                "post_switch_accuracy": hidden_score.video_equal_post_switch_accuracy,
            }
        )
    max_loss = float(config["selection_constraints"]["max_natural_accuracy_loss"])
    for row in rows:
        row["natural_accuracy_loss"] = (
            current_natural.video_equal_accuracy - float(row["natural_accuracy"])
        )
        row["eligible"] = bool(float(row["natural_accuracy_loss"]) <= max_loss)
        row["selected"] = False

    def choose(candidate_type: str) -> dict[str, Any]:
        candidates = [
            row
            for row in rows
            if row["candidate_type"] == candidate_type and bool(row["eligible"])
        ]
        if not candidates:
            raise RuntimeError(f"no eligible {candidate_type} candidate")
        return min(
            candidates,
            key=lambda row: (
                -float(row["hidden_accuracy"]),
                -float(row["post_switch_accuracy"]),
                float(row["natural_accuracy_loss"]),
                (
                    float(row["retention"])
                    if row["retention"] is not None
                    else int(row["window_frames"])
                ),
            ),
        )

    retention_row = choose("fixed_forgetting")
    window_row = choose("sliding_window")
    retention_row["selected"] = True
    window_row["selected"] = True
    return (
        float(retention_row["retention"]),
        int(window_row["window_frames"]),
        pd.DataFrame(rows),
    )


def controller_candidates(config: Mapping[str, Any]) -> tuple[SoftMemoryConfig, ...]:
    grid = config["controller_grid"]
    candidates: list[SoftMemoryConfig] = []
    seen: set[tuple[float, ...]] = set()
    for floor, ceiling, gain, threshold, template_value in product(
        grid["retention_floor"],
        grid["retention_ceiling"],
        grid["gain"],
        grid["threshold"],
        grid["feature_templates"],
    ):
        floor = float(floor)
        ceiling = float(ceiling)
        if floor >= ceiling:
            continue
        template = np.asarray(template_value, dtype=np.float64)
        template /= np.sum(template)
        gain = float(gain)
        weights = gain * template
        candidate = SoftMemoryConfig(
            retention_floor=floor,
            retention_ceiling=ceiling,
            bias=-gain * float(threshold),
            surprise_weight=float(weights[0]),
            entropy_weight=float(weights[1]),
            disagreement_weight=float(weights[2]),
            fast_retention=float(grid["fast_retention"]),
            slow_retention=float(grid["slow_retention"]),
            evidence_weight_floor=float(grid["evidence_weight_floor"]),
        )
        signature = tuple(float(value) for value in asdict(candidate).values())
        if signature not in seen:
            seen.add(signature)
            candidates.append(candidate)
    if not candidates:
        raise ValueError("controller grid produced no valid candidates")
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class CachedControlStream:
    stream: Stream51Stream
    raw_features: np.ndarray
    standardized_features: np.ndarray


def fit_control_standardizer(
    streams: Sequence[Stream51Stream], *, config: Mapping[str, Any]
) -> ControllerStandardizer:
    grid = config["controller_grid"]
    features = [
        causal_control_features(
            stream.evidence,
            stream_ids=stream.stream_ids,
            fast_retention=float(grid["fast_retention"]),
            slow_retention=float(grid["slow_retention"]),
            observation_log_likelihood=stream.observation_log_likelihood,
        ).raw_features
        for stream in streams
    ]
    return ControllerStandardizer.fit(np.concatenate(features))


def cache_control_streams(
    streams: Sequence[Stream51Stream],
    *,
    standardizer: ControllerStandardizer,
    config: Mapping[str, Any],
) -> tuple[CachedControlStream, ...]:
    grid = config["controller_grid"]
    cached: list[CachedControlStream] = []
    for stream in streams:
        raw = causal_control_features(
            stream.evidence,
            stream_ids=stream.stream_ids,
            fast_retention=float(grid["fast_retention"]),
            slow_retention=float(grid["slow_retention"]),
            observation_log_likelihood=stream.observation_log_likelihood,
        ).raw_features
        cached.append(
            CachedControlStream(
                stream=stream,
                raw_features=raw,
                standardized_features=standardizer.transform(raw),
            )
        )
    return tuple(cached)


def _candidate_tapes(
    cached: Sequence[CachedControlStream], *, candidate: SoftMemoryConfig, hard: bool = False
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    predictions: list[np.ndarray] = []
    retention_tapes: list[np.ndarray] = []
    risk_tapes: list[np.ndarray] = []
    for item in cached:
        retention, risk = controller_retention(
            item.standardized_features, config=candidate, hard=hard
        )
        weights = candidate.evidence_weight_floor + (
            1.0 - candidate.evidence_weight_floor
        ) * (1.0 - item.raw_features[:, 1])
        prediction, _, _ = accumulate_with_retention(
            item.stream.evidence,
            stream_ids=item.stream.stream_ids,
            retention_value=retention,
            evidence_weight_value=weights,
        )
        predictions.append(prediction)
        retention_tapes.append(retention)
        risk_tapes.append(risk)
    return predictions, retention_tapes, risk_tapes


def _combined_reachability(
    cached_groups: Sequence[Sequence[CachedControlStream]],
    risk_groups: Sequence[Sequence[np.ndarray]],
    *,
    task: Mapping[str, Any],
) -> Any:
    streams = [item.stream for group in cached_groups for item in group]
    risks = [risk for group in risk_groups for risk in group]
    return switch_reachability(
        np.concatenate(risks),
        switch_flags=np.concatenate([stream.switch_flags for stream in streams]),
        stream_ids=np.concatenate([stream.stream_ids for stream in streams]),
        threshold=0.5,
        detection_tolerance=int(task["detection_tolerance"]),
        refractory_frames=int(task["refractory_frames"]),
        min_run_frames=int(task["min_run_frames"]),
    )


def select_controller(
    natural: Sequence[Stream51Stream],
    hidden: Sequence[Stream51Stream],
    *,
    standardizer: ControllerStandardizer,
    reference_natural_accuracy: float,
    config: Mapping[str, Any],
) -> tuple[SoftMemoryConfig, bool, pd.DataFrame]:
    natural_cache = cache_control_streams(
        natural, standardizer=standardizer, config=config
    )
    hidden_cache = cache_control_streams(
        hidden, standardizer=standardizer, config=config
    )
    post_window = int(config["task"]["post_switch_window"])
    constraints = config["selection_constraints"]
    rows: list[dict[str, Any]] = []
    candidate_objects: list[SoftMemoryConfig] = []
    for index, candidate in enumerate(controller_candidates(config)):
        natural_predictions, natural_retention, natural_risk = _candidate_tapes(
            natural_cache, candidate=candidate
        )
        hidden_predictions, hidden_retention, hidden_risk = _candidate_tapes(
            hidden_cache, candidate=candidate
        )
        natural_score, _ = _score_predictions(
            natural, natural_predictions, post_window=post_window
        )
        hidden_score, _ = _score_predictions(
            hidden, hidden_predictions, post_window=post_window
        )
        reachability = _combined_reachability(
            (natural_cache, hidden_cache),
            (natural_risk, hidden_risk),
            task=config["task"],
        )
        natural_loss = reference_natural_accuracy - natural_score.video_equal_accuracy
        eligible = bool(
            natural_loss <= float(constraints["max_natural_accuracy_loss"])
            and np.isfinite(reachability.recall)
            and reachability.recall >= float(constraints["min_switch_recall"])
            and reachability.false_alarms_per_1000
            <= float(constraints["max_false_alarms_per_1000"])
            and np.isfinite(reachability.median_delay)
        )
        rows.append(
            {
                "candidate_index": index,
                **asdict(candidate),
                "natural_accuracy": natural_score.video_equal_accuracy,
                "natural_accuracy_loss": natural_loss,
                "hidden_accuracy": hidden_score.video_equal_accuracy,
                "post_switch_accuracy": hidden_score.video_equal_post_switch_accuracy,
                "reachability_auc": reachability.auc,
                "reachability_recall": reachability.recall,
                "false_alarms_per_1000": reachability.false_alarms_per_1000,
                "median_delay": reachability.median_delay,
                "mean_retention": float(
                    np.mean(np.concatenate([*natural_retention, *hidden_retention]))
                ),
                "eligible": eligible,
                "selected": False,
            }
        )
        candidate_objects.append(candidate)
    eligible_rows = [row for row in rows if bool(row["eligible"])]
    selection_pool = eligible_rows if eligible_rows else rows
    selected_row = min(
        selection_pool,
        key=lambda row: (
            -float(row["hidden_accuracy"]),
            -float(row["post_switch_accuracy"]),
            float(row["natural_accuracy_loss"]),
            -float(row["reachability_auc"])
            if np.isfinite(float(row["reachability_auc"]))
            else float("inf"),
            int(row["candidate_index"]),
        ),
    )
    selected_row["selected"] = True
    selected = candidate_objects[int(selected_row["candidate_index"])]
    return selected, bool(eligible_rows), pd.DataFrame(rows)


def _stationary_scores(
    natural: Sequence[Stream51Stream],
    hidden: Sequence[Stream51Stream],
    *,
    retention: float,
    window_frames: int,
    post_window: int,
) -> dict[str, PanelScore]:
    conditions = {
        "current_natural": (natural, _fixed_predictions(natural, retention=0.0)),
        "retention_natural": (
            natural,
            _fixed_predictions(natural, retention=retention),
        ),
        "window_natural": (
            natural,
            _window_predictions(natural, window_frames=window_frames),
        ),
        "retention_hidden": (
            hidden,
            _fixed_predictions(hidden, retention=retention),
        ),
        "window_hidden": (
            hidden,
            _window_predictions(hidden, window_frames=window_frames),
        ),
        "cumulative_hidden": (hidden, _fixed_predictions(hidden, retention=1.0)),
    }
    return {
        name: _score_predictions(streams, predictions, post_window=post_window)[0]
        for name, (streams, predictions) in conditions.items()
    }


def evaluate_qualification(
    natural: Sequence[Stream51Stream],
    hidden: Sequence[Stream51Stream],
    *,
    retention: float,
    window_frames: int,
    controller: SoftMemoryConfig,
    controller_fit_eligible: bool,
    standardizer: ControllerStandardizer,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    post_window = int(config["task"]["post_switch_window"])
    stationary = _stationary_scores(
        natural,
        hidden,
        retention=retention,
        window_frames=window_frames,
        post_window=post_window,
    )
    natural_cache = cache_control_streams(
        natural, standardizer=standardizer, config=config
    )
    hidden_cache = cache_control_streams(
        hidden, standardizer=standardizer, config=config
    )
    natural_soft, _, natural_risk = _candidate_tapes(
        natural_cache, candidate=controller
    )
    hidden_soft, _, hidden_risk = _candidate_tapes(hidden_cache, candidate=controller)
    soft_natural, soft_natural_rows = _score_predictions(
        natural, natural_soft, post_window=post_window
    )
    soft_hidden, soft_hidden_rows = _score_predictions(
        hidden, hidden_soft, post_window=post_window
    )
    oracle_predictions = [_oracle_reset_prediction(stream) for stream in hidden]
    oracle_hidden, _ = _score_predictions(
        hidden, oracle_predictions, post_window=post_window
    )
    reachability = _combined_reachability(
        (natural_cache, hidden_cache),
        (natural_risk, hidden_risk),
        task=config["task"],
    )
    best_accumulator_natural = max(
        stationary["retention_natural"].video_equal_accuracy,
        stationary["window_natural"].video_equal_accuracy,
    )
    best_fixed_hidden = max(
        stationary["retention_hidden"].video_equal_accuracy,
        stationary["window_hidden"].video_equal_accuracy,
    )
    best_fixed_post = max(
        stationary["retention_hidden"].video_equal_post_switch_accuracy,
        stationary["window_hidden"].video_equal_post_switch_accuracy,
    )
    thresholds = config["qualification"]
    qualification = qualify_memory_demand(
        current_frame_natural_accuracy=stationary[
            "current_natural"
        ].video_equal_accuracy,
        best_accumulator_natural_accuracy=best_accumulator_natural,
        best_fixed_hidden_accuracy=best_fixed_hidden,
        oracle_hidden_accuracy=oracle_hidden.video_equal_accuracy,
        best_fixed_post_switch_accuracy=best_fixed_post,
        cumulative_post_switch_accuracy=stationary[
            "cumulative_hidden"
        ].video_equal_post_switch_accuracy,
        reachability=reachability,
        stable_gain_mcid=float(thresholds["stable_gain_mcid"]),
        oracle_headroom_mcid=float(thresholds["oracle_headroom_mcid"]),
        cumulative_harm_mcid=float(thresholds["cumulative_harm_mcid"]),
        min_reachability_auc=float(thresholds["min_reachability_auc"]),
        min_reachability_recall=float(thresholds["min_reachability_recall"]),
        max_false_alarms_per_1000=float(
            thresholds["max_false_alarms_per_1000"]
        ),
        max_median_delay=float(thresholds["max_median_delay"]),
    )
    passed = bool(qualification.passed and controller_fit_eligible)
    summary = {
        **asdict(qualification),
        "controller_fit_eligible": bool(controller_fit_eligible),
        "passed": passed,
        "selected_retention": float(retention),
        "selected_window_frames": int(window_frames),
        "current_frame_natural_accuracy": stationary[
            "current_natural"
        ].video_equal_accuracy,
        "fixed_retention_natural_accuracy": stationary[
            "retention_natural"
        ].video_equal_accuracy,
        "sliding_window_natural_accuracy": stationary[
            "window_natural"
        ].video_equal_accuracy,
        "fixed_retention_hidden_accuracy": stationary[
            "retention_hidden"
        ].video_equal_accuracy,
        "sliding_window_hidden_accuracy": stationary[
            "window_hidden"
        ].video_equal_accuracy,
        "cumulative_hidden_accuracy": stationary[
            "cumulative_hidden"
        ].video_equal_accuracy,
        "soft_natural_accuracy": soft_natural.video_equal_accuracy,
        "soft_hidden_accuracy": soft_hidden.video_equal_accuracy,
        "soft_hidden_post_switch_accuracy": soft_hidden.video_equal_post_switch_accuracy,
        "oracle_hidden_accuracy": oracle_hidden.video_equal_accuracy,
    }
    qualification_rows = pd.concat(
        [
            soft_natural_rows.assign(panel="natural"),
            soft_hidden_rows.assign(panel="hidden_switch"),
        ],
        ignore_index=True,
    )
    return summary, qualification_rows


def run_qualification_seed(
    config: Mapping[str, Any], *, seed: int, results_root: str | Path
) -> Path:
    validate_config(config)
    initialize_seed(seed)
    store = _store(config, include_external=False)
    fit_keys, qualification_keys = partition_development_videos(
        store,
        salt=str(config["development_partition_salt"]),
        fit_fraction=float(config["development_fit_fraction"]),
    )
    model = fit_stream51_vmf(
        store,
        split="support",
        max_frames_per_video=int(config["encoder"]["support_frames_per_video"]),
    )
    with ExperimentRun(
        EXPERIMENT,
        seed,
        dict(config),
        results_root=results_root,
        run_label="qualification",
    ) as run:
        run.register_conditions([{"stage": "qualification", "seed": seed}])
        np.savez_compressed(
            run.path / "support_vmf_model.npz",
            directions=model.directions,
            concentration=np.array([model.concentration]),
            class_counts=model.class_counts,
        )
        temperature, temperature_audit = select_temperature(
            store,
            model,
            video_keys=fit_keys,
            grid=config["temperature_grid"],
        )
        fit_natural, fit_hidden = _make_streams(
            store,
            model,
            split="development",
            video_keys=fit_keys,
            temperature=temperature,
            seed=seed,
            task=config["task"],
        )
        retention, window, stationary_audit = select_stationary_baselines(
            fit_natural, fit_hidden, config=config
        )
        standardizer = fit_control_standardizer(
            (*fit_natural, *fit_hidden), config=config
        )
        stationary_fit_scores = _stationary_scores(
            fit_natural,
            fit_hidden,
            retention=retention,
            window_frames=window,
            post_window=int(config["task"]["post_switch_window"]),
        )
        reference_natural = max(
            stationary_fit_scores["retention_natural"].video_equal_accuracy,
            stationary_fit_scores["window_natural"].video_equal_accuracy,
        )
        controller, controller_eligible, controller_audit = select_controller(
            fit_natural,
            fit_hidden,
            standardizer=standardizer,
            reference_natural_accuracy=reference_natural,
            config=config,
        )
        qualification_natural, qualification_hidden = _make_streams(
            store,
            model,
            split="development",
            video_keys=qualification_keys,
            temperature=temperature,
            seed=seed,
            task=config["task"],
        )
        qualification, qualification_rows = evaluate_qualification(
            qualification_natural,
            qualification_hidden,
            retention=retention,
            window_frames=window,
            controller=controller,
            controller_fit_eligible=controller_eligible,
            standardizer=standardizer,
            config=config,
        )
        selected = {
            "temperature": temperature,
            "retention": retention,
            "window_frames": window,
            "controller": asdict(controller),
            "controller_standardizer": {
                "mean": standardizer.mean.tolist(),
                "scale": standardizer.scale.tolist(),
            },
            "controller_fit_eligible": controller_eligible,
            "fit_video_count": len(fit_keys),
            "qualification_video_count": len(qualification_keys),
        }
        temperature_audit.to_csv(run.path / "temperature_audit.csv", index=False)
        stationary_audit.to_csv(run.path / "stationary_audit.csv", index=False)
        controller_audit.to_csv(run.path / "controller_audit.csv", index=False)
        qualification_rows.to_csv(
            run.path / "qualification_video_metrics.csv", index=False
        )
        (run.path / "development_partition.json").write_text(
            json.dumps(
                {
                    "fit_video_keys": fit_keys,
                    "qualification_video_keys": qualification_keys,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (run.path / "selected_hyperparameters.json").write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {
            "experiment": EXPERIMENT,
            "protocol_version": PROTOCOL_VERSION,
            "stage": "qualification",
            "seed": seed,
            "statistical_unit": "source_video",
            "external_features_accessed": False,
            "selected_hyperparameters": selected,
            "qualification": qualification,
        }
        (run.path / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        run.record(
            {
                **qualification,
                "status": "complete",
                "external_features_accessed": False,
            },
            stage="qualification",
        )
        return run.path


def validate_qualification_receipt(
    path_value: str | Path, *, config: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Exp38 qualification receipt is missing")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp38 qualification receipt version mismatch")
    if receipt.get("all_registered_seeds_passed") is not True:
        raise PermissionError("Exp38 external stage remains locked")
    if tuple(receipt.get("seeds", ())) != tuple(seed_list(config["seeds"])):
        raise ValueError("Exp38 qualification receipt seed mismatch")
    seed_results = receipt.get("seed_results")
    if not isinstance(seed_results, list):
        raise ValueError("Exp38 qualification receipt has no seed results")
    observed = tuple(int(row.get("seed", -1)) for row in seed_results)
    if observed != tuple(seed_list(config["seeds"])):
        raise ValueError("Exp38 qualification seed results are incomplete")
    if not all(
        row.get("passed") is True
        and isinstance(row.get("qualification"), Mapping)
        and row["qualification"].get("passed") is True
        and isinstance(row.get("selected_hyperparameters"), Mapping)
        for row in seed_results
    ):
        raise ValueError("Exp38 qualification receipt contains an invalid seed result")
    preregistration = PROJECT_ROOT / "provenance/exp38_preregistration_receipt_20260726.json"
    implementation = PROJECT_ROOT / "provenance/exp38_implementation_receipt_20260726.json"
    for source, key in (
        (preregistration, "preregistration_receipt_sha256"),
        (implementation, "implementation_receipt_sha256"),
    ):
        if not source.is_file() or _sha256(source) != str(receipt.get(key)):
            raise ValueError("Exp38 qualification receipt provenance mismatch")
    return receipt


def qualification_parameters_for_seed(
    receipt: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    """Load, validate, and copy the exact parameters frozen at qualification."""

    matching = [
        row
        for row in receipt.get("seed_results", ())
        if int(row.get("seed", -1)) == int(seed)
    ]
    if len(matching) != 1:
        raise ValueError("qualification receipt does not identify one seed result")
    selected = matching[0].get("selected_hyperparameters")
    if not isinstance(selected, Mapping):
        raise ValueError("qualification receipt has no selected hyperparameters")
    required = {
        "temperature",
        "retention",
        "window_frames",
        "controller",
        "controller_standardizer",
        "controller_fit_eligible",
    }
    if not required <= set(selected):
        raise ValueError("qualification hyperparameters are incomplete")
    if selected["controller_fit_eligible"] is not True:
        raise PermissionError("qualification controller was not eligible")
    temperature = float(selected["temperature"])
    retention = float(selected["retention"])
    window_frames = int(selected["window_frames"])
    if (
        not np.isfinite(temperature)
        or temperature <= 0.0
        or not np.isfinite(retention)
        or not 0.0 <= retention <= 1.0
        or window_frames < 1
    ):
        raise ValueError("qualification scalar hyperparameters are invalid")
    controller = SoftMemoryConfig(**dict(selected["controller"]))
    standardizer_payload = selected["controller_standardizer"]
    if not isinstance(standardizer_payload, Mapping):
        raise ValueError("qualification controller standardizer is invalid")
    standardizer = ControllerStandardizer(
        mean=np.asarray(standardizer_payload["mean"], dtype=np.float64),
        scale=np.asarray(standardizer_payload["scale"], dtype=np.float64),
    )
    return {
        **dict(selected),
        "temperature": temperature,
        "retention": retention,
        "window_frames": window_frames,
        "controller": asdict(controller),
        "controller_standardizer": {
            "mean": standardizer.mean.tolist(),
            "scale": standardizer.scale.tolist(),
        },
    }


def _external_condition_predictions(
    stream: Stream51Stream,
    *,
    retention: float,
    window_frames: int,
    controller: SoftMemoryConfig,
    standardizer: ControllerStandardizer,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    current = fixed_forgetting_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, retention=0.0
    ).predictions
    cumulative = fixed_forgetting_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, retention=1.0
    ).predictions
    fixed = fixed_forgetting_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, retention=retention
    ).predictions
    window = sliding_window_accumulator(
        stream.evidence,
        stream_ids=stream.stream_ids,
        window_frames=window_frames,
    ).predictions
    raw = causal_control_features(
        stream.evidence,
        stream_ids=stream.stream_ids,
        fast_retention=controller.fast_retention,
        slow_retention=controller.slow_retention,
        observation_log_likelihood=stream.observation_log_likelihood,
    ).raw_features
    standardized = standardizer.transform(raw)
    soft_retention, soft_risk = controller_retention(
        standardized, config=controller
    )
    hard_retention, _ = controller_retention(
        standardized, config=controller, hard=True
    )
    weights = controller.evidence_weight_floor + (
        1.0 - controller.evidence_weight_floor
    ) * (1.0 - raw[:, 1])

    def accumulate(tape: np.ndarray) -> np.ndarray:
        return accumulate_with_retention(
            stream.evidence,
            stream_ids=stream.stream_ids,
            retention_value=tape,
            evidence_weight_value=weights,
        )[0]

    if len(soft_retention) < 2:
        raise ValueError("external stream is too short for timing intervention")
    offset = 1 + derive_seed(
        seed, "exp38-shift", stream.panel, stream.task_index
    ) % (len(soft_retention) - 1)
    shifted_retention = np.roll(soft_retention, int(offset))
    oracle_retention = np.ones(len(stream.labels), dtype=np.float64)
    oracle_retention[stream.switch_flags] = 0.0
    predictions = {
        "current_frame": current,
        "cumulative": cumulative,
        "fixed_forgetting": fixed,
        "sliding_window": window,
        "soft_memory": accumulate(soft_retention),
        "hard_memory": accumulate(hard_retention),
        "matched_shifted_memory": accumulate(shifted_retention),
        "oracle_memory": accumulate(oracle_retention),
    }
    summaries = {
        "soft_memory": {
            "mean_retention": float(np.mean(soft_retention)),
            "retention_std": float(np.std(soft_retention)),
            "mean_change_risk": float(np.mean(soft_risk)),
            "max_change_risk": float(np.max(soft_risk)),
        },
        "hard_memory": {
            "mean_retention": float(np.mean(hard_retention)),
            "retention_std": float(np.std(hard_retention)),
            "mean_change_risk": float(np.mean(soft_risk >= 0.5)),
            "max_change_risk": float(np.max(soft_risk)),
        },
        "matched_shifted_memory": {
            "mean_retention": float(np.mean(shifted_retention)),
            "retention_std": float(np.std(shifted_retention)),
            "mean_change_risk": float(np.mean(soft_risk)),
            "max_change_risk": float(np.max(soft_risk)),
        },
        "oracle_memory": {
            "mean_retention": float(np.mean(oracle_retention)),
            "retention_std": float(np.std(oracle_retention)),
            "mean_change_risk": float(np.mean(stream.switch_flags)),
            "max_change_risk": float(np.max(stream.switch_flags)),
        },
    }
    return predictions, summaries


def run_external_seed(
    config: Mapping[str, Any],
    *,
    seed: int,
    results_root: str | Path,
    qualification_receipt: str | Path,
) -> Path:
    validate_config(config)
    receipt = validate_qualification_receipt(qualification_receipt, config=config)
    selected = qualification_parameters_for_seed(receipt, seed=seed)
    initialize_seed(seed)
    store = _store(config, include_external=True)
    model = fit_stream51_vmf(
        store,
        split="support",
        max_frames_per_video=int(config["encoder"]["support_frames_per_video"]),
    )
    temperature = float(selected["temperature"])
    retention = float(selected["retention"])
    window = int(selected["window_frames"])
    controller = SoftMemoryConfig(**dict(selected["controller"]))
    standardizer = ControllerStandardizer(
        mean=np.asarray(
            selected["controller_standardizer"]["mean"], dtype=np.float64
        ),
        scale=np.asarray(
            selected["controller_standardizer"]["scale"], dtype=np.float64
        ),
    )
    external_keys = store.video_keys("external")
    natural, hidden = _make_streams(
        store,
        model,
        split="external",
        video_keys=external_keys,
        temperature=temperature,
        seed=seed,
        task=config["task"],
    )
    with ExperimentRun(
        EXPERIMENT,
        seed,
        dict(config),
        results_root=results_root,
        run_label="external",
    ) as run:
        run.register_conditions(
            [
                {"panel": panel, "condition": condition}
                for panel in ("natural", "hidden_switch")
                for condition in CONDITIONS
            ]
        )
        rows: list[pd.DataFrame] = []
        control_rows: list[dict[str, Any]] = []
        for panel, streams in (("natural", natural), ("hidden_switch", hidden)):
            predictions_by_condition: dict[str, list[np.ndarray]] = {
                condition: [] for condition in CONDITIONS
            }
            for stream in streams:
                predictions, summaries = _external_condition_predictions(
                    stream,
                    retention=retention,
                    window_frames=window,
                    controller=controller,
                    standardizer=standardizer,
                    seed=seed,
                )
                for condition, prediction in predictions.items():
                    predictions_by_condition[condition].append(prediction)
                for condition, summary in summaries.items():
                    control_rows.append(
                        {
                            "panel": panel,
                            "task_index": stream.task_index,
                            "condition": condition,
                            **summary,
                        }
                    )
            for condition, prediction_tapes in predictions_by_condition.items():
                score, video_rows = _score_predictions(
                    streams,
                    prediction_tapes,
                    post_window=int(config["task"]["post_switch_window"]),
                )
                video_rows = video_rows.assign(panel=panel, condition=condition)
                rows.append(video_rows)
                run.record(
                    {**asdict(score), "status": "complete"},
                    panel=panel,
                    condition=condition,
                )
        raw = pd.concat(rows, ignore_index=True)
        expected = {
            (video_key, panel, condition)
            for video_key in external_keys
            for panel in ("natural", "hidden_switch")
            for condition in CONDITIONS
        }
        observed = set(
            raw[["source_video_id", "panel", "condition"]].itertuples(
                index=False, name=None
            )
        )
        if observed != expected:
            raise RuntimeError("Exp38 external source-video coverage is incomplete")
        raw.to_csv(run.path / "external_video_metrics.csv", index=False)
        pd.DataFrame(control_rows).to_csv(
            run.path / "external_control_summary.csv", index=False
        )
        (run.path / "selected_hyperparameters.json").write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {
            "experiment": EXPERIMENT,
            "protocol_version": PROTOCOL_VERSION,
            "stage": "external",
            "seed": seed,
            "statistical_unit": "source_video",
            "external_features_accessed": True,
            "coverage": {
                "complete": True,
                "n_external_videos": len(external_keys),
                "expected_video_cells": len(expected),
                "observed_video_cells": len(observed),
            },
            "selected_hyperparameters": selected,
            "condition_means": raw.groupby(
                ["panel", "condition"], as_index=False
            )["accuracy"].mean().to_dict("records"),
        }
        (run.path / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return run.path


def main() -> None:
    parser = basic_parser(
        __doc__ or EXPERIMENT,
        "configs/prospective/exp38_stream51_soft_memory.json",
    )
    parser.add_argument(
        "--stage", choices=("qualification", "external"), default="qualification"
    )
    parser.add_argument("--qualification-receipt", default=None)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_json_config(config_path)
    validate_config(config)
    validate_preregistration(config_path)
    validate_implementation_receipt()
    seeds = seed_list(args.seeds if args.seeds is not None else config["seeds"])
    if args.stage == "external" and args.qualification_receipt is None:
        raise ValueError("--qualification-receipt is required for external stage")
    for seed in seeds:
        if args.stage == "qualification":
            path = run_qualification_seed(
                config, seed=seed, results_root=args.results_root
            )
        else:
            path = run_external_seed(
                config,
                seed=seed,
                results_root=args.results_root,
                qualification_receipt=args.qualification_receipt,
            )
        print(path)


if __name__ == "__main__":
    main()
