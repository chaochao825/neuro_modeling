"""Family-aware, failure-preserving metrics for structured reasoning tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.structured_protocol import (
    PublicTask,
    StructuredDataset,
    StructuredProtocolError,
    VALID_SPLITS,
)


PredictionProvider = Mapping[str, Any] | Callable[[PublicTask], Any]
_ENDPOINTS = {
    "arc": ("all_query_exact", "exact"),
    "maze": ("path_valid", "path_optimal", "exact"),
    "sudoku": (
        "blank_cell_accuracy",
        "full_cell_accuracy",
        "clues_preserved",
        "rows_valid",
        "columns_valid",
        "boxes_valid",
        "valid_solution",
        "exact",
    ),
}


@dataclass(frozen=True, slots=True)
class StructuredReasoningEvaluation:
    """Per-task records plus a source-group-aware aggregate."""

    family: str
    split: str
    task_metrics: pd.DataFrame
    summary: Mapping[str, Any]


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("bootstrap_seed must be an integer")
    return int(seed)


def _validate_n_bootstrap(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("n_bootstrap must be an integer")
    if value < 1:
        raise ValueError("n_bootstrap must be positive")
    return int(value)


def _dependency_components(frame: pd.DataFrame) -> np.ndarray:
    parent = list(range(len(frame)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple[str, object], int] = {}
    for index, row in frame.iterrows():
        for kind in ("source_group", "augmentation_group"):
            key = (kind, row[kind])
            prior = seen.setdefault(key, index)
            left, right = find(index), find(prior)
            if left != right:
                parent[right] = left
    return np.asarray([find(index) for index in range(len(frame))], dtype=int)


def _cluster_bootstrap_mean(
    values: np.ndarray,
    components: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
) -> tuple[float, float]:
    n_bootstrap = _validate_n_bootstrap(n_bootstrap)
    unique_groups = np.asarray(sorted(set(components.tolist())), dtype=object)
    if unique_groups.size == 0:
        raise ValueError("cluster bootstrap requires groups")
    group_means = np.asarray(
        [values[components == group].mean() for group in unique_groups], dtype=float
    )
    rng = np.random.default_rng(_validate_seed(seed))
    sampled = rng.integers(0, len(group_means), size=(n_bootstrap, len(group_means)))
    means = group_means[sampled].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _prediction_for(
    provider: PredictionProvider, task: PublicTask
) -> tuple[object, str, str | None]:
    if isinstance(provider, Mapping):
        if task.task_id not in provider:
            return None, "missing", None
        prediction = provider[task.task_id]
        return prediction, "missing" if prediction is None else "returned", None
    if not callable(provider):
        raise TypeError("predictions must be a mapping or callable")
    try:
        prediction = provider(task)
    except Exception as error:  # A failed task remains an explicit denominator row.
        return None, "failed", type(error).__name__
    return prediction, "missing" if prediction is None else "returned", None


def evaluate_predictions(
    dataset: StructuredDataset,
    predictions: PredictionProvider,
    *,
    family: str | None = None,
    split: str = "test",
    bootstrap_seed: int,
    n_bootstrap: int = 10_000,
) -> StructuredReasoningEvaluation:
    """Evaluate every selected task; missing, malformed, and failed attempts stay in.

    Primary point estimates retain every task in the denominator.  Confidence
    intervals resample source groups, never individual augmented copies.
    """

    normalized_split = str(split).strip().lower()
    if normalized_split not in VALID_SPLITS:
        raise StructuredProtocolError(f"unknown split {split!r}")
    selected = dataset.for_split(normalized_split)
    if family is None:
        families = {task.family for task in selected}
        if len(families) != 1:
            raise StructuredProtocolError(
                "family must be supplied when a split has zero or multiple families"
            )
        normalized_family = next(iter(families))
    else:
        normalized_family = str(family).strip().lower()
        selected = tuple(task for task in selected if task.family == normalized_family)
    if normalized_family not in _ENDPOINTS:
        raise StructuredProtocolError(f"unsupported family {normalized_family!r}")
    if not selected:
        raise StructuredProtocolError(
            f"no {normalized_family!r} tasks in split {normalized_split!r}"
        )
    if isinstance(predictions, Mapping):
        known_ids = {task.task_id for task in selected}
        extras = set(predictions) - known_ids
        if extras:
            raise StructuredProtocolError(
                f"predictions contain unknown/unselected task IDs: {sorted(extras)!r}"
            )

    rows: list[dict[str, Any]] = []
    for task in selected:
        prediction, status, error_type = _prediction_for(predictions, task)
        metrics = dict(dataset.target_store.score(task, prediction))
        row: dict[str, Any] = {
            "task_id": task.task_id,
            "family": task.family,
            "split": task.split,
            "source_group": task.source_group,
            "augmentation_group": task.augmentation_group,
            "task_fingerprint": task.fingerprint,
            "prediction_status": status,
            "failure_type": error_type,
        }
        row.update(metrics)
        rows.append(row)
    frame = pd.DataFrame(rows)
    endpoints = _ENDPOINTS[normalized_family]
    summary: dict[str, Any] = {
        "status": "complete",
        "family": normalized_family,
        "split": normalized_split,
        "n_tasks": int(len(frame)),
        "n_source_groups": int(frame["source_group"].nunique()),
        "n_predictions_returned": int((frame["prediction_status"] == "returned").sum()),
        "n_missing": int((frame["prediction_status"] == "missing").sum()),
        "n_failed": int((frame["prediction_status"] == "failed").sum()),
        "denominator_includes_missing_and_failed": True,
        "statistics_unit": "source_augmentation_dependency_component",
        "bootstrap_seed": _validate_seed(bootstrap_seed),
        "n_bootstrap": _validate_n_bootstrap(n_bootstrap),
        "group_split_disjoint": True,
        "query_targets_exposed_to_model": False,
    }
    components = _dependency_components(frame)
    summary["n_dependency_components"] = int(len(set(components.tolist())))
    for endpoint in endpoints:
        values = frame[endpoint].fillna(False).to_numpy(dtype=float)
        low, high = _cluster_bootstrap_mean(
            values,
            components,
            seed=bootstrap_seed,
            n_bootstrap=n_bootstrap,
        )
        component_means = np.asarray(
            [values[components == item].mean() for item in sorted(set(components))]
        )
        summary[f"{endpoint}_rate"] = float(component_means.mean())
        summary[f"{endpoint}_task_rate"] = float(values.mean())
        summary[f"{endpoint}_ci_low"] = low
        summary[f"{endpoint}_ci_high"] = high
    primary_endpoint = {
        "arc": "all_query_exact",
        "maze": "path_optimal",
        "sudoku": "exact",
    }[normalized_family]
    summary["primary_endpoint"] = primary_endpoint
    summary["primary_value"] = summary[f"{primary_endpoint}_rate"]
    return StructuredReasoningEvaluation(
        family=normalized_family,
        split=normalized_split,
        task_metrics=frame,
        summary=summary,
    )


def evaluate_arc_predictions(
    dataset: StructuredDataset,
    predictions: PredictionProvider,
    *,
    split: str = "test",
    bootstrap_seed: int,
    n_bootstrap: int = 10_000,
) -> StructuredReasoningEvaluation:
    return evaluate_predictions(
        dataset,
        predictions,
        family="arc",
        split=split,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )


def evaluate_maze_predictions(
    dataset: StructuredDataset,
    predictions: PredictionProvider,
    *,
    split: str = "test",
    bootstrap_seed: int,
    n_bootstrap: int = 10_000,
) -> StructuredReasoningEvaluation:
    return evaluate_predictions(
        dataset,
        predictions,
        family="maze",
        split=split,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )


def evaluate_sudoku_predictions(
    dataset: StructuredDataset,
    predictions: PredictionProvider,
    *,
    split: str = "test",
    bootstrap_seed: int,
    n_bootstrap: int = 10_000,
) -> StructuredReasoningEvaluation:
    return evaluate_predictions(
        dataset,
        predictions,
        family="sudoku",
        split=split,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )


evaluate_structured_reasoning = evaluate_predictions
