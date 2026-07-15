"""Leakage-safe loader and exact-match scorer for official ARC JSON tasks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.data.structured_protocol import (
    PublicTask,
    StructuredDataset,
    StructuredProtocolError,
    VALID_SPLITS,
    build_structured_dataset,
    public_projection_sha256,
)


def _grid(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u"}:
        raise StructuredProtocolError(f"{name} must contain integer colors")
    array = np.asarray(raw, dtype=np.int16)
    if array.ndim != 2 or min(array.shape, default=0) < 1:
        raise StructuredProtocolError(f"{name} must be a non-empty 2D grid")
    if np.any((array < 0) | (array > 9)):
        raise StructuredProtocolError(f"{name} colors must be in [0, 9]")
    return array


def _pair(value: object, *, name: str, require_output: bool) -> tuple[np.ndarray, ...]:
    if not isinstance(value, Mapping) or "input" not in value:
        raise StructuredProtocolError(f"{name} must contain input")
    input_grid = _grid(value["input"], name=f"{name}.input")
    if not require_output:
        return (input_grid,)
    if "output" not in value:
        raise StructuredProtocolError(f"{name} must contain output for evaluation")
    return input_grid, _grid(value["output"], name=f"{name}.output")


def _normalize_split(value: object) -> str:
    if not isinstance(value, str) or value.strip().lower() not in VALID_SPLITS:
        raise StructuredProtocolError(
            f"ARC split must be one of {sorted(VALID_SPLITS)!r}"
        )
    return value.strip().lower()


def _infer_split(path: Path, root: Path, default: str | None) -> str:
    aliases = {
        "development": "validation",
        "evaluation": "test",
        "eval": "test",
        "test": "test",
        "train": "train",
        "training": "train",
        "validation": "validation",
        "val": "validation",
    }
    for part in reversed(path.relative_to(root).parts[:-1]):
        inferred = aliases.get(part.lower())
        if inferred is not None:
            return inferred
    if default is None:
        raise StructuredProtocolError(
            f"cannot infer split for {path}; pass split=... or use split directories"
        )
    return _normalize_split(default)


def _prediction_outputs(prediction: object, *, n_queries: int) -> tuple[object, ...]:
    if prediction is None:
        return ()
    if isinstance(prediction, Mapping):
        prediction = prediction.get("outputs", prediction.get("grids"))
        if prediction is None:
            return ()
    if isinstance(prediction, np.ndarray):
        if prediction.ndim == 2 and n_queries == 1:
            return (prediction,)
        if prediction.ndim == 3:
            return tuple(prediction[index] for index in range(prediction.shape[0]))
        return ()
    if not isinstance(prediction, Sequence) or isinstance(prediction, (str, bytes)):
        return ()
    if n_queries == 1:
        try:
            candidate = np.asarray(prediction)
        except (TypeError, ValueError):
            candidate = np.empty(0)
        if candidate.ndim == 2:
            return (prediction,)
    return tuple(prediction)


def _grid_comparison(candidate: object, expected: np.ndarray) -> tuple[bool, bool, float]:
    """Validate one submitted ARC grid before computing diagnostics."""

    try:
        values = np.asarray(candidate)
    except (TypeError, ValueError):
        return False, False, 0.0
    valid = (
        values.ndim == 2
        and values.dtype.kind in {"i", "u"}
        and not np.any((values < 0) | (values > 9))
    )
    if not valid:
        return False, False, 0.0
    shape_exact = values.shape == expected.shape
    exact = shape_exact and np.array_equal(values, expected)
    cell_accuracy = float(np.mean(values == expected)) if shape_exact else 0.0
    return bool(exact), bool(shape_exact), cell_accuracy


def score_arc_prediction(
    task: PublicTask, prediction: object, target: object
) -> Mapping[str, object]:
    """Require exact equality for every query in an ARC task."""

    del task
    targets = tuple(target) if isinstance(target, (tuple, list)) else ()
    outputs = _prediction_outputs(prediction, n_queries=len(targets))
    query_exact: list[bool] = []
    for index, expected in enumerate(targets):
        if index >= len(outputs):
            query_exact.append(False)
            continue
        exact, _shape_exact, _cell_accuracy = _grid_comparison(
            outputs[index], expected
        )
        query_exact.append(bool(exact))
    complete = len(outputs) == len(targets) and bool(targets)
    all_exact = complete and all(query_exact)
    return {
        "prediction_provided": prediction is not None,
        "n_queries": len(targets),
        "n_outputs_received": len(outputs),
        "query_exact_count": int(sum(query_exact)),
        "query_exact_fraction": (float(np.mean(query_exact)) if query_exact else 0.0),
        "all_query_exact": bool(all_exact),
        "exact": bool(all_exact),
    }


def score_arc_attempts(
    task: PublicTask,
    prediction: object,
    target: object,
    *,
    max_attempts: int = 2,
) -> Mapping[str, object]:
    """Score the official ARC protocol of at most two attempts per query.

    The preferred prediction representation is ``{"attempts": [a1, a2]}``,
    where each attempt has the same complete-task representation accepted by
    :func:`score_arc_prediction`.  A legacy single prediction is treated as
    one attempt.  A task is correct only when every query is solved by at
    least one registered attempt; failed or malformed attempts remain in the
    denominator.
    """

    del task
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError("max_attempts must be an integer")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    targets = tuple(target) if isinstance(target, (tuple, list)) else ()
    raw_attempts: object
    if isinstance(prediction, Mapping) and "attempts" in prediction:
        raw_attempts = prediction["attempts"]
    else:
        raw_attempts = (prediction,) if prediction is not None else ()
    if not isinstance(raw_attempts, Sequence) or isinstance(
        raw_attempts, (str, bytes, np.ndarray)
    ):
        attempts: tuple[object, ...] = ()
    else:
        attempts = tuple(raw_attempts)
    too_many_attempts = len(attempts) > max_attempts
    accepted_attempts = attempts[:max_attempts]
    parsed = tuple(
        _prediction_outputs(attempt, n_queries=len(targets))
        for attempt in accepted_attempts
    )
    malformed_attempts = tuple(len(outputs) != len(targets) for outputs in parsed)
    query_exact: list[bool] = []
    query_winning_attempt: list[int | None] = []
    query_shape_exact: list[bool] = []
    query_best_cell_accuracy: list[float] = []
    attempt_exact: list[list[bool]] = [list() for _ in parsed]
    for query_index, expected in enumerate(targets):
        solved = False
        winner: int | None = None
        shape_solved = False
        best_cell_accuracy = 0.0
        for attempt_index, outputs in enumerate(parsed):
            if query_index >= len(outputs):
                attempt_exact[attempt_index].append(False)
                continue
            exact, shape_exact, cell_accuracy = _grid_comparison(
                outputs[query_index], expected
            )
            attempt_exact[attempt_index].append(bool(exact))
            shape_solved |= bool(shape_exact)
            best_cell_accuracy = max(best_cell_accuracy, cell_accuracy)
            if exact and winner is None:
                solved = True
                winner = attempt_index + 1
        for attempt_index in range(len(parsed)):
            if len(attempt_exact[attempt_index]) <= query_index:
                attempt_exact[attempt_index].append(False)
        query_exact.append(solved)
        query_winning_attempt.append(winner)
        query_shape_exact.append(shape_solved)
        query_best_cell_accuracy.append(best_cell_accuracy)
    all_exact = (
        bool(targets)
        and all(query_exact)
        and not too_many_attempts
        and not any(malformed_attempts)
    )
    return {
        "prediction_provided": prediction is not None,
        "n_queries": len(targets),
        "n_attempts_received": len(attempts),
        "n_attempts_scored": len(accepted_attempts),
        "max_attempts": max_attempts,
        "too_many_attempts": too_many_attempts,
        "malformed_attempts": malformed_attempts,
        "query_exact_count": int(sum(query_exact)),
        "query_exact_fraction": (float(np.mean(query_exact)) if query_exact else 0.0),
        "query_exact": tuple(query_exact),
        "query_winning_attempt": tuple(query_winning_attempt),
        "query_shape_exact": tuple(query_shape_exact),
        "query_best_cell_accuracy": tuple(query_best_cell_accuracy),
        "shape_exact_fraction": (
            float(np.mean(query_shape_exact)) if query_shape_exact else 0.0
        ),
        "best_cell_accuracy": (
            float(np.mean(query_best_cell_accuracy))
            if query_best_cell_accuracy
            else 0.0
        ),
        "attempt_query_exact": tuple(tuple(row) for row in attempt_exact),
        "all_query_exact": all_exact,
        "exact": all_exact,
    }


def load_arc_directory(
    directory: str | Path,
    *,
    split: str | None = None,
    dataset_name: str = "ARC",
    dataset_revision: str = "unspecified",
    exclude_relative_paths: Sequence[str] = (),
    namespace_task_ids: bool = False,
    attempt_aware_scoring: bool = False,
) -> StructuredDataset:
    """Load official ARC JSON files while stripping every query output.

    Standard ``training``/``evaluation`` directory names are converted to
    ``train``/``test`` automatically.  A direct directory requires an explicit
    ``split`` argument.  Optional top-level ``source_group`` and
    ``augmentation_group`` metadata are honored and checked by the shared
    protocol for cross-split leakage.
    """

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise StructuredProtocolError("dataset_name must be a non-empty string")
    if not isinstance(dataset_revision, str) or not dataset_revision.strip():
        raise StructuredProtocolError("dataset_revision must be a non-empty string")
    if not isinstance(namespace_task_ids, (bool, np.bool_)):
        raise StructuredProtocolError("namespace_task_ids must be boolean")
    excluded: set[str] = set()
    for value in exclude_relative_paths:
        candidate = Path(str(value).replace("\\", "/"))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise StructuredProtocolError("ARC exclusions must be relative paths")
        excluded.add(candidate.as_posix())
    discovered = sorted(root.rglob("*.json"))
    discovered_relative = {
        path.relative_to(root).as_posix(): path for path in discovered
    }
    missing_exclusions = excluded - set(discovered_relative)
    if missing_exclusions:
        raise StructuredProtocolError(
            f"ARC exclusions do not exist: {sorted(missing_exclusions)!r}"
        )
    paths = [
        path
        for relative, path in sorted(discovered_relative.items())
        if relative not in excluded
    ]
    if not paths:
        raise StructuredProtocolError(f"no ARC JSON files found under {root}")
    tasks: list[PublicTask] = []
    targets: list[tuple[np.ndarray, ...]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise StructuredProtocolError(
                f"invalid ARC JSON {path}: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise StructuredProtocolError(f"ARC task {path} must be a JSON object")
        raw_train = payload.get("train")
        raw_test = payload.get("test")
        if (
            not isinstance(raw_train, list)
            or not raw_train
            or not isinstance(raw_test, list)
            or not raw_test
        ):
            raise StructuredProtocolError(
                f"ARC task {path} needs non-empty train and test lists"
            )
        demos: list[Mapping[str, np.ndarray]] = []
        for index, raw_pair in enumerate(raw_train):
            input_grid, output_grid = _pair(
                raw_pair, name=f"{path.name}.train[{index}]", require_output=True
            )
            demos.append({"input": input_grid, "output": output_grid})
        query_inputs: list[np.ndarray] = []
        query_targets: list[np.ndarray] = []
        for index, raw_pair in enumerate(raw_test):
            input_grid, output_grid = _pair(
                raw_pair, name=f"{path.name}.test[{index}]", require_output=True
            )
            query_inputs.append(input_grid)
            query_targets.append(output_grid)
        relative = path.relative_to(root).with_suffix("").as_posix()
        task_split = (
            _normalize_split(payload["split"])
            if "split" in payload
            else _infer_split(path, root, split)
        )
        public_digest = public_projection_sha256(
            {
                "family": "arc",
                "support_inputs": tuple(item["input"] for item in demos),
                "support_outputs": tuple(item["output"] for item in demos),
                "query_inputs": tuple(query_inputs),
            }
        )
        bare_task_id = str(payload.get("task_id", relative)).strip()
        task_id = (
            f"{dataset_name.strip()}@{dataset_revision.strip()}:"
            f"{bare_task_id}:{public_digest[:16]}"
            if namespace_task_ids
            else bare_task_id
        )
        # Target-free semantic grouping catches renamed copies across splits.
        source_group = str(payload.get("source_group", public_digest)).strip()
        augmentation_group = str(
            payload.get("augmentation_group", source_group)
        ).strip()
        tasks.append(
            PublicTask(
                task_id=task_id,
                family="arc",
                split=task_split,
                source_group=source_group,
                augmentation_group=augmentation_group,
                context={
                    "demonstrations": tuple(demos),
                    "support_inputs": tuple(item["input"] for item in demos),
                    "support_outputs": tuple(item["output"] for item in demos),
                },
                query={"inputs": tuple(query_inputs)},
                metadata={
                    "source_file": path.relative_to(root).as_posix(),
                    "source_format": "official_arc_json",
                    "source_dataset": dataset_name.strip(),
                    "source_version": dataset_revision.strip(),
                    "source_sha256": public_digest,
                    "source_hash_scope": "public_projection",
                    "excluded_relative_paths": tuple(sorted(excluded)),
                    "n_demonstrations": len(demos),
                    "n_queries": len(query_inputs),
                },
            )
        )
        targets.append(tuple(query_targets))
    if not isinstance(attempt_aware_scoring, (bool, np.bool_)):
        raise StructuredProtocolError("attempt_aware_scoring must be boolean")
    scorer = score_arc_attempts if attempt_aware_scoring else score_arc_prediction
    return build_structured_dataset(tasks, targets, scorer=scorer)


# Explicit alias used by callers that name the official benchmark family.
load_official_arc_directory = load_arc_directory
