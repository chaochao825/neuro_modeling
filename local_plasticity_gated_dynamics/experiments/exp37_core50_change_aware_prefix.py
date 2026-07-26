"""Prospective CORe50 audit of Bayesian change-aware prefix accumulation."""

from __future__ import annotations

from dataclasses import asdict
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
from src.data.core50_streaming import (
    Core50FeatureStore,
    Core50Stream,
    fit_object_prototypes,
    prepare_core50_task,
)
from src.models.bocpd_prefix import BOCPDConfig, bocpd_prefix_accumulator
from src.models.change_aware_prefix import (
    AccumulatorTrace,
    circularly_shift_resets,
    fixed_forgetting_accumulator,
    scheduled_reset_accumulator,
    sliding_window_accumulator,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


EXPERIMENT = "exp37_core50_change_aware_prefix"
PROTOCOL_VERSION = "exp37_core50_change_aware_prefix_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANELS = ("natural", "hidden_switch")
CONDITIONS = (
    "current_frame",
    "cumulative",
    "fixed_forgetting",
    "sliding_window",
    "bocpd_change_reset",
    "bocpd_posterior",
    "bocpd_score_no_reset",
    "matched_shifted_reset",
    "oracle_change_reset",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_preregistration(config_path: Path) -> dict[str, Any]:
    receipt_path = PROJECT_ROOT / "provenance/exp37_preregistration_receipt_20260726.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp37 preregistration version mismatch")
    for path, key in (
        (PROJECT_ROOT / str(receipt["protocol_path"]), "protocol_sha256"),
        (config_path.resolve(), "config_sha256"),
        (PROJECT_ROOT / str(receipt["cohort_path"]), "cohort_sha256"),
    ):
        if not path.is_file() or _sha256(path) != str(receipt[key]):
            raise ValueError(f"Exp37 preregistration hash mismatch: {path}")
    return receipt


def _finite_unique_grid(
    value: Any, *, name: str, lower: float, upper: float
) -> tuple[float, ...]:
    grid = np.asarray(value, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    if np.any((grid < lower) | (grid > upper)) or len(set(grid.tolist())) != len(grid):
        raise ValueError(f"{name} has duplicate or out-of-range values")
    return tuple(float(item) for item in grid)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp37 protocol mismatch")
    if config.get("profile") != "prospective_external":
        raise ValueError("Exp37 v1 only permits prospective_external")
    if config.get("evidence_provenance") != "prospective_core50_session_holdout":
        raise ValueError("Exp37 evidence provenance mismatch")
    for key in ("used_external_labels_for_fit", "used_future_frames", "used_autograd", "used_bptt"):
        if config.get(key) is not False:
            raise ValueError(f"Exp37 requires {key}=false")
    if config.get("used_development_labels_for_selection") is not True:
        raise ValueError("Exp37 must disclose development label selection")
    if tuple(config.get("support_sessions", ())) != ("s1",):
        raise ValueError("Exp37 support session must be s1")
    if tuple(config.get("development_sessions", ())) != ("s2",):
        raise ValueError("Exp37 development session must be s2")
    if tuple(config.get("external_sessions", ())) != tuple(f"s{i}" for i in range(3, 12)):
        raise ValueError("Exp37 external sessions must be s3--s11")
    if tuple(config.get("objects", ())) != tuple(f"o{i}" for i in range(1, 51)):
        raise ValueError("Exp37 requires all 50 objects")
    if len(seed_list(config["seeds"])) != 5:
        raise ValueError("Exp37 requires five seeds")
    for key in (
        "n_external_tasks_per_session",
        "n_development_tasks",
        "prototype_stride",
        "max_prototype_frames_per_object",
    ):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    _finite_unique_grid(config["temperature_grid"], name="temperature_grid", lower=1e-12, upper=1e6)
    _finite_unique_grid(config["retention_grid"], name="retention_grid", lower=0.0, upper=1.0)
    if any(int(item) < 1 for item in config["window_grid"]):
        raise ValueError("window_grid must be positive")
    grid = config["bocpd_grid"]
    _finite_unique_grid(grid["hazard"], name="hazard", lower=1e-12, upper=1.0 - 1e-12)
    _finite_unique_grid(
        grid["prior_concentration"], name="prior_concentration", lower=1e-12, upper=1e6
    )
    _finite_unique_grid(grid["alarm_threshold"], name="alarm_threshold", lower=1e-12, upper=1.0)
    for key in ("min_run_frames",):
        if any(int(item) < 1 for item in grid[key]):
            raise ValueError(f"{key} must be positive")
    BOCPDConfig(
        hazard=float(grid["hazard"][0]),
        prior_concentration=float(grid["prior_concentration"][0]),
        alarm_threshold=float(grid["alarm_threshold"][0]),
        min_run_frames=int(grid["min_run_frames"][0]),
        max_run_length=int(grid["max_run_length"]),
    )
    stream = config["stream"]
    for key in (
        "n_classes",
        "segments_per_stream",
        "segment_frames",
        "natural_frames",
        "post_switch_window",
        "detection_tolerance",
    ):
        if isinstance(stream.get(key), bool) or int(stream.get(key, 0)) < 1:
            raise ValueError(f"stream.{key} must be positive")


def validate_feature_cache(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config["feature_root"])).expanduser().resolve()
    required = (
        root / "feature_manifest.csv",
        root / "failures.csv",
        root / "provenance.json",
        root / "acquisition_attestation.json",
        root / "schema_audit.csv",
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"CORe50 feature artifact missing: {path}")
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("CORe50 feature protocol mismatch")
    if provenance.get("schema_complete") is not True or int(
        provenance.get("n_completed_cells", -1)
    ) != 550:
        raise RuntimeError("CORe50 feature cache has incomplete schema")
    if int(provenance.get("n_failed_cells", -1)) != 0:
        raise RuntimeError("CORe50 feature cache retains failed cells")
    identity = str(provenance.get("encoder_identity", ""))
    if "efficientnet_b0" not in identity or "IMAGENET1K_V1" not in identity:
        raise RuntimeError("CORe50 feature cache uses the wrong encoder")
    failures = pd.read_csv(root / "failures.csv", keep_default_na=False)
    if len(failures):
        raise RuntimeError("CORe50 failures.csv is nonempty")
    manifest = pd.read_csv(root / "feature_manifest.csv", keep_default_na=False)
    if len(manifest) != 550:
        raise RuntimeError("CORe50 manifest must contain 550 cells")
    return {
        "root": str(root),
        "archive_sha256": str(provenance["archive_sha256"]),
        "archive_content_length": int(provenance["archive_content_length"]),
        "encoder_identity": identity,
        "n_cells": int(len(manifest)),
        "n_frames": int(pd.to_numeric(manifest["n_frames"], errors="raise").sum()),
    }


def _store(config: Mapping[str, Any]) -> Core50FeatureStore:
    sessions = (
        list(config["support_sessions"])
        + list(config["development_sessions"])
        + list(config["external_sessions"])
    )
    return Core50FeatureStore(
        Path(str(config["feature_root"])),
        expected_sessions=sessions,
        expected_objects=config["objects"],
        cache_in_memory=True,
    )


def _prototypes(store: Core50FeatureStore, config: Mapping[str, Any]) -> np.ndarray:
    return fit_object_prototypes(
        store,
        support_session="s1",
        object_ids=config["objects"],
        stride=int(config["prototype_stride"]),
        max_frames_per_object=int(config["max_prototype_frames_per_object"]),
    )


def _streams(
    store: Core50FeatureStore,
    prototypes: np.ndarray,
    *,
    session_id: str,
    seed: int,
    n_tasks: int,
    temperature: float,
    config: Mapping[str, Any],
) -> list[Core50Stream]:
    result: list[Core50Stream] = []
    for task_index in range(n_tasks):
        result.extend(
            prepare_core50_task(
                store,
                prototypes=prototypes,
                session_id=session_id,
                seed=seed,
                task_index=task_index,
                temperature=temperature,
                stream_config=config["stream"],
            )
        )
    return result


def _accuracy(trace: AccumulatorTrace, stream: Core50Stream) -> tuple[int, int]:
    return int(np.sum(trace.predictions == stream.labels)), int(len(stream.labels))


def select_temperature(
    store: Core50FeatureStore,
    prototypes: np.ndarray,
    *,
    seed: int,
    config: Mapping[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for temperature in config["temperature_grid"]:
        streams = _streams(
            store,
            prototypes,
            session_id="s2",
            seed=seed,
            n_tasks=int(config["n_development_tasks"]),
            temperature=float(temperature),
            config=config,
        )
        natural = [stream for stream in streams if stream.panel == "natural"]
        correct = frames = 0
        for stream in natural:
            trace = fixed_forgetting_accumulator(
                stream.evidence, stream_ids=stream.stream_ids, retention=0.0
            )
            value, count = _accuracy(trace, stream)
            correct += value
            frames += count
        rows.append(
            {
                "candidate_type": "temperature",
                "temperature": float(temperature),
                "natural_current_frame_accuracy": correct / frames,
                "eligible": True,
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            float(row["natural_current_frame_accuracy"]),
            -float(row["temperature"]),
        ),
    )
    return float(selected["temperature"]), rows


def _select_stationary(
    streams: list[Core50Stream], config: Mapping[str, Any]
) -> tuple[float, int, list[dict[str, Any]]]:
    hidden = [stream for stream in streams if stream.panel == "hidden_switch"]
    rows: list[dict[str, Any]] = []
    for retention in config["retention_grid"]:
        correct = frames = 0
        for stream in hidden:
            trace = fixed_forgetting_accumulator(
                stream.evidence, stream_ids=stream.stream_ids, retention=float(retention)
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
    retention_row = max(
        rows,
        key=lambda row: (float(row["hidden_switch_accuracy"]), float(row["retention"])),
    )
    for window in config["window_grid"]:
        correct = frames = 0
        for stream in hidden:
            trace = sliding_window_accumulator(
                stream.evidence, stream_ids=stream.stream_ids, window_frames=int(window)
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
    window_row = max(
        [row for row in rows if row["candidate_type"] == "sliding_window"],
        key=lambda row: (float(row["hidden_switch_accuracy"]), int(row["window_frames"])),
    )
    return float(retention_row["retention"]), int(window_row["window_frames"]), rows


def _bocpd_grid(config: Mapping[str, Any]) -> Iterable[BOCPDConfig]:
    grid = config["bocpd_grid"]
    for hazard, prior, threshold, minimum in product(
        grid["hazard"],
        grid["prior_concentration"],
        grid["alarm_threshold"],
        grid["min_run_frames"],
    ):
        yield BOCPDConfig(
            hazard=float(hazard),
            prior_concentration=float(prior),
            alarm_threshold=float(threshold),
            min_run_frames=int(minimum),
            max_run_length=int(grid["max_run_length"]),
        )


def _select_bocpd(
    streams: list[Core50Stream], config: Mapping[str, Any]
) -> tuple[BOCPDConfig, list[dict[str, Any]]]:
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
    for candidate in _bocpd_grid(config):
        natural_correct = natural_frames = natural_alarms = 0
        for stream in natural:
            trace = bocpd_prefix_accumulator(
                stream.evidence,
                stream_ids=stream.stream_ids,
                config=candidate,
                mode="hard_reset",
            )
            value, count = _accuracy(trace, stream)
            natural_correct += value
            natural_frames += count
            natural_alarms += int(np.sum(trace.alarm_flags))
        hidden_correct = hidden_frames = post_correct = post_frames = 0
        delays: list[float] = []
        for stream in hidden:
            trace = bocpd_prefix_accumulator(
                stream.evidence,
                stream_ids=stream.stream_ids,
                config=candidate,
                mode="hard_reset",
            )
            value, count = _accuracy(trace, stream)
            hidden_correct += value
            hidden_frames += count
            metric = change_point_metrics(
                trace.predictions,
                stream.labels,
                alarm_flags=trace.alarm_flags,
                switch_flags=stream.switch_flags,
                post_switch_window=int(config["stream"]["post_switch_window"]),
                detection_tolerance=int(config["stream"]["detection_tolerance"]),
            )
            switch_indices = np.flatnonzero(stream.switch_flags)
            mask = np.zeros(len(stream.labels), dtype=np.bool_)
            for switch in switch_indices:
                mask[switch : switch + int(config["stream"]["post_switch_window"])] = True
            post_correct += int(np.sum(trace.predictions[mask] == stream.labels[mask]))
            post_frames += int(np.sum(mask))
            if np.isfinite(metric.median_detection_delay):
                delays.append(metric.median_detection_delay)
        natural_accuracy = natural_correct / natural_frames
        false_alarm_rate = 1000.0 * natural_alarms / natural_frames
        delay = float(np.mean(delays)) if delays else float("inf")
        eligible = (
            cumulative_accuracy - natural_accuracy
            <= float(constraints["max_natural_accuracy_loss"])
            and false_alarm_rate <= float(constraints["max_false_alarms_per_1000"])
        )
        rows.append(
            {
                "candidate_type": "bocpd_change_reset",
                **asdict(candidate),
                "natural_accuracy": natural_accuracy,
                "natural_accuracy_loss": cumulative_accuracy - natural_accuracy,
                "natural_false_alarms_per_1000": false_alarm_rate,
                "hidden_switch_accuracy": hidden_correct / hidden_frames,
                "post_switch_accuracy": post_correct / post_frames,
                "mean_detection_delay": delay if np.isfinite(delay) else None,
                "eligible": bool(eligible),
            }
        )
    eligible_rows = [row for row in rows if bool(row["eligible"])]
    if not eligible_rows:
        raise RuntimeError("no BOCPD detector satisfies frozen development constraints")

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        delay = (
            float(row["mean_detection_delay"])
            if row["mean_detection_delay"] is not None
            else float("inf")
        )
        parameters = (
            float(row["hazard"]),
            float(row["prior_concentration"]),
            float(row["alarm_threshold"]),
            int(row["min_run_frames"]),
        )
        return (
            -float(row["hidden_switch_accuracy"]),
            -float(row["post_switch_accuracy"]),
            delay,
            float(row["natural_false_alarms_per_1000"]),
            parameters,
        )

    selected = min(eligible_rows, key=key)
    return BOCPDConfig(
        hazard=float(selected["hazard"]),
        prior_concentration=float(selected["prior_concentration"]),
        alarm_threshold=float(selected["alarm_threshold"]),
        min_run_frames=int(selected["min_run_frames"]),
        max_run_length=int(selected["max_run_length"]),
    ), rows


def fit_development_selection(
    store: Core50FeatureStore,
    prototypes: np.ndarray,
    *,
    seed: int,
    config: Mapping[str, Any],
) -> tuple[float, float, int, BOCPDConfig, pd.DataFrame]:
    temperature, temperature_rows = select_temperature(
        store, prototypes, seed=seed, config=config
    )
    streams = _streams(
        store,
        prototypes,
        session_id="s2",
        seed=seed,
        n_tasks=int(config["n_development_tasks"]),
        temperature=temperature,
        config=config,
    )
    retention, window, stationary_rows = _select_stationary(streams, config)
    detector, detector_rows = _select_bocpd(streams, config)
    audit = pd.DataFrame([*temperature_rows, *stationary_rows, *detector_rows])
    audit["selected"] = False
    audit.loc[
        (audit["candidate_type"] == "temperature")
        & (audit["temperature"] == temperature),
        "selected",
    ] = True
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
    detector_mask = audit["candidate_type"] == "bocpd_change_reset"
    for name, value in asdict(detector).items():
        detector_mask &= audit[name] == value
    audit.loc[detector_mask, "selected"] = True
    return temperature, retention, window, detector, audit


def _condition_traces(
    stream: Core50Stream,
    *,
    fixed_retention: float,
    window_frames: int,
    detector: BOCPDConfig,
    seed: int,
) -> dict[str, AccumulatorTrace]:
    current = fixed_forgetting_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, retention=0.0
    )
    cumulative = fixed_forgetting_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, retention=1.0
    )
    forgetting = fixed_forgetting_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, retention=fixed_retention
    )
    window = sliding_window_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, window_frames=window_frames
    )
    change = bocpd_prefix_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, config=detector, mode="hard_reset"
    )
    posterior = bocpd_prefix_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, config=detector, mode="posterior"
    )
    score = bocpd_prefix_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, config=detector, mode="score_only"
    )
    offset = 1 + derive_seed(
        seed, "exp37-matched-reset", stream.session_id, stream.task_index, stream.panel
    ) % (len(stream.labels) - 1)
    shifted = circularly_shift_resets(
        change.reset_flags, stream_ids=stream.stream_ids, offset=int(offset)
    )
    matched = scheduled_reset_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, reset_schedule=shifted
    )
    oracle = scheduled_reset_accumulator(
        stream.evidence, stream_ids=stream.stream_ids, reset_schedule=stream.switch_flags
    )
    return {
        "current_frame": current,
        "cumulative": cumulative,
        "fixed_forgetting": forgetting,
        "sliding_window": window,
        "bocpd_change_reset": change,
        "bocpd_posterior": posterior,
        "bocpd_score_no_reset": score,
        "matched_shifted_reset": matched,
        "oracle_change_reset": oracle,
    }


def evaluate_stream(
    stream: Core50Stream,
    *,
    fixed_retention: float,
    window_frames: int,
    detector: BOCPDConfig,
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
                "session_id": stream.session_id,
                "task_index": stream.task_index,
                "panel": stream.panel,
                "condition": condition,
                **asdict(metrics),
                "n_resets": int(np.sum(trace.reset_flags)),
                "mean_state_l1": float(np.mean(trace.state_l1)),
                "object_ids": "|".join(stream.object_ids),
                "source_cells": "|".join(stream.source_cells),
                "status": "complete",
            }
        )
    return pd.DataFrame(rows)


def run_seed(
    config: Mapping[str, Any], *, seed: int, results_root: str | Path
) -> Path:
    validate_config(config)
    initialize_seed(seed)
    feature_info = validate_feature_cache(config)
    store = _store(config)
    prototypes = _prototypes(store, config)
    with ExperimentRun(
        EXPERIMENT,
        seed,
        dict(config),
        results_root=results_root,
        run_label="prospective_external",
    ) as run:
        sessions = tuple(map(str, config["external_sessions"]))
        n_tasks = int(config["n_external_tasks_per_session"])
        run.register_conditions(
            [
                {
                    "session_id": session,
                    "task_index": task,
                    "panel": panel,
                    "condition": condition,
                }
                for session in sessions
                for task in range(n_tasks)
                for panel in PANELS
                for condition in CONDITIONS
            ]
        )
        temperature, retention, window, detector, audit = fit_development_selection(
            store, prototypes, seed=seed, config=config
        )
        audit.to_csv(run.path / "development_selection_audit.csv", index=False)
        selected = {
            "temperature": temperature,
            "fixed_retention": retention,
            "window_frames": window,
            "detector": asdict(detector),
            "development_sessions": ["s2"],
            "used_external_labels": False,
        }
        (run.path / "selected_hyperparameters.json").write_text(
            json.dumps(selected, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        frames: list[pd.DataFrame] = []
        for session in sessions:
            for task_index in range(n_tasks):
                try:
                    streams = prepare_core50_task(
                        store,
                        prototypes=prototypes,
                        session_id=session,
                        seed=seed,
                        task_index=task_index,
                        temperature=temperature,
                        stream_config=config["stream"],
                    )
                    for stream in streams:
                        frame = evaluate_stream(
                            stream,
                            fixed_retention=retention,
                            window_frames=window,
                            detector=detector,
                            seed=seed,
                            config=config,
                        )
                        frames.append(frame)
                        for row in frame.to_dict("records"):
                            dimensions = {
                                "session_id": row.pop("session_id"),
                                "task_index": row.pop("task_index"),
                                "panel": row.pop("panel"),
                                "condition": row.pop("condition"),
                            }
                            run.record(row, **dimensions)
                except Exception as error:  # noqa: BLE001 - retain every failed cell
                    for panel in PANELS:
                        for condition in CONDITIONS:
                            run.mark_condition_failure(
                                error,
                                session_id=session,
                                task_index=task_index,
                                panel=panel,
                                condition=condition,
                            )
        if not frames:
            raise RuntimeError("all Exp37 external tasks failed")
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(run.path / "external_task_metrics.csv", index=False)
        expected = {
            (session, task, panel, condition)
            for session in sessions
            for task in range(n_tasks)
            for panel in PANELS
            for condition in CONDITIONS
        }
        observed = set(
            raw[["session_id", "task_index", "panel", "condition"]].itertuples(
                index=False, name=None
            )
        )
        summary = {
            "experiment": EXPERIMENT,
            "protocol_version": PROTOCOL_VERSION,
            "profile": "prospective_external",
            "evidence_provenance": config["evidence_provenance"],
            "claim_upgrade_allowed": bool(config["claim_upgrade_allowed"]),
            "seed": seed,
            "selected_hyperparameters": selected,
            "coverage": {
                "complete": observed == expected,
                "expected_conditions": len(expected),
                "observed_conditions": len(observed),
                "expected_sessions": len(sessions),
                "observed_sessions": int(raw["session_id"].nunique()),
            },
            "condition_task_mean_accuracy": raw.groupby(
                ["panel", "condition"], as_index=False
            )["accuracy"].mean().to_dict("records"),
            "feature_cache": feature_info,
            "statistical_unit": "session",
        }
        (run.path / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return run.path


def main() -> None:
    parser = basic_parser(
        __doc__ or EXPERIMENT,
        "configs/prospective/exp37_core50_change_aware_prefix.json",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_json_config(config_path)
    validate_preregistration(config_path)
    seeds = seed_list(args.seeds if args.seeds is not None else config["seeds"])
    for seed in seeds:
        print(run_seed(config, seed=seed, results_root=args.results_root))


if __name__ == "__main__":
    main()
