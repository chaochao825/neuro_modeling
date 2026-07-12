"""JSONL/NPZ maze tasks scored by legality and shortest-path optimality."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.data.structured_protocol import (
    PublicTask,
    StructuredDataset,
    StructuredProtocolError,
    VALID_SPLITS,
    build_structured_dataset,
    public_projection_sha256,
)


def _maze(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "b"}:
        raise StructuredProtocolError(f"{name} must be an integer 2D array")
    array = np.asarray(raw, dtype=np.int16)
    if array.ndim != 2 or min(array.shape, default=0) < 1:
        raise StructuredProtocolError(f"{name} must be a non-empty 2D array")
    return array


def _coordinate(value: object, *, name: str) -> tuple[int, int]:
    raw = np.asarray(value)
    if raw.shape != (2,) or raw.dtype.kind not in {"i", "u"}:
        raise StructuredProtocolError(f"{name} must be an integer [row, column]")
    return int(raw[0]), int(raw[1])


def _split(value: object, *, default: str | None) -> str:
    candidate = default if value is None else value
    if not isinstance(candidate, str) or candidate.strip().lower() not in VALID_SPLITS:
        raise StructuredProtocolError(
            f"maze split must be one of {sorted(VALID_SPLITS)!r}"
        )
    return candidate.strip().lower()


def shortest_path_distance(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    wall_value: int = 1,
) -> int | None:
    """Return the unweighted four-neighbor distance, or ``None`` if unreachable."""

    rows, columns = maze.shape

    def open_cell(point: tuple[int, int]) -> bool:
        row, column = point
        return (
            0 <= row < rows
            and 0 <= column < columns
            and int(maze[row, column]) != wall_value
        )

    if not open_cell(start) or not open_cell(goal):
        return None
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    visited = {start}
    while queue:
        point, distance = queue.popleft()
        if point == goal:
            return distance
        row, column = point
        for neighbor in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if neighbor not in visited and open_cell(neighbor):
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def _parse_path(prediction: object) -> tuple[tuple[int, int], ...] | None:
    if prediction is None:
        return None
    if isinstance(prediction, Mapping):
        prediction = prediction.get("path")
        if prediction is None:
            return None
    try:
        raw = np.asarray(prediction)
    except (TypeError, ValueError):
        return None
    if raw.ndim != 2 or raw.shape[1:] != (2,) or raw.dtype.kind not in {"i", "u"}:
        return None
    return tuple((int(point[0]), int(point[1])) for point in raw)


def score_maze_prediction(
    task: PublicTask, prediction: object, target: object
) -> Mapping[str, object]:
    """Accept every legal shortest path instead of one privileged reference."""

    maze = np.asarray(task.query["grid"])
    start = tuple(task.query["start"])
    goal = tuple(task.query["goal"])
    wall_value = int(task.query["wall_value"])
    expected_distance = int(target["shortest_distance"])
    path = _parse_path(prediction)
    path_valid = path is not None and bool(path)
    if path_valid:
        path_valid = path[0] == start and path[-1] == goal
    if path_valid:
        rows, columns = maze.shape
        for point in path:
            row, column = point
            if (
                row < 0
                or row >= rows
                or column < 0
                or column >= columns
                or int(maze[row, column]) == wall_value
            ):
                path_valid = False
                break
    if path_valid:
        path_valid = all(
            abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1
            for left, right in zip(path, path[1:], strict=False)
        )
    path_length = len(path) - 1 if path is not None and path else None
    optimal = bool(path_valid and path_length == expected_distance)
    return {
        "prediction_provided": prediction is not None,
        "path_parseable": path is not None,
        "path_valid": bool(path_valid),
        "path_optimal": optimal,
        "path_length": path_length,
        "shortest_distance": expected_distance,
        "optimality_gap": (
            int(path_length - expected_distance)
            if path_valid and path_length is not None
            else None
        ),
        "exact": optimal,
    }


def _record_to_task(
    record: Mapping[str, Any],
    *,
    index: int,
    default_split: str | None,
    source_file: str,
    source_format: str,
) -> tuple[PublicTask, Mapping[str, int]]:
    maze_value = record.get("maze", record.get("grid"))
    if maze_value is None:
        raise StructuredProtocolError(f"maze record {index} is missing maze/grid")
    maze = _maze(maze_value, name=f"maze[{index}]")
    start = _coordinate(record.get("start"), name=f"start[{index}]")
    goal = _coordinate(record.get("goal"), name=f"goal[{index}]")
    wall_value = int(record.get("wall_value", 1))
    distance = shortest_path_distance(maze, start, goal, wall_value=wall_value)
    if distance is None:
        raise StructuredProtocolError(f"maze record {index} has no path to its goal")
    declared_distance = record.get("shortest_distance", record.get("distance"))
    if declared_distance is not None and int(declared_distance) != distance:
        raise StructuredProtocolError(
            f"maze record {index} shortest_distance disagrees with BFS"
        )
    task_id = str(record.get("task_id", f"maze_{index:06d}")).strip()
    public_digest = public_projection_sha256(
        {
            "family": "maze",
            "grid": maze,
            "start": start,
            "goal": goal,
            "wall_value": wall_value,
        }
    )
    source_group = str(record.get("source_group", public_digest)).strip()
    augmentation_group = str(record.get("augmentation_group", source_group)).strip()
    task = PublicTask(
        task_id=task_id,
        family="maze",
        split=_split(record.get("split"), default=default_split),
        source_group=source_group,
        augmentation_group=augmentation_group,
        context={"support_inputs": (), "support_outputs": ()},
        query={
            "grid": maze,
            "start": start,
            "goal": goal,
            "wall_value": wall_value,
        },
        metadata={
            "source_file": source_file,
            "source_format": source_format,
            "source_version": str(record.get("source_version", "maze_v1")),
            "source_repository": record.get("source_repository"),
            "source_record_path": record.get("source_record_path"),
            "source_record_sha256": record.get("source_record_sha256"),
            "source_license": record.get("source_license"),
            "split_provenance": record.get("split_provenance"),
            "parse_pipeline": record.get("parse_pipeline"),
            "source_sha256": public_digest,
            "source_hash_scope": "public_projection",
            "height": int(maze.shape[0]),
            "width": int(maze.shape[1]),
        },
    )
    return task, {"shortest_distance": distance}


def _jsonl_records(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise StructuredProtocolError(
                f"invalid maze JSONL line {line_number}: {error}"
            ) from error
        if not isinstance(value, Mapping):
            raise StructuredProtocolError(
                f"maze JSONL line {line_number} must be an object"
            )
        records.append(value)
    return records


def _npz_records(path: Path) -> list[Mapping[str, Any]]:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise StructuredProtocolError(
            f"cannot safely load maze NPZ: {error}"
        ) from error
    with archive:
        maze_key = "mazes" if "mazes" in archive else "maze"
        if maze_key not in archive or "starts" not in archive or "goals" not in archive:
            raise StructuredProtocolError("maze NPZ needs mazes, starts, and goals")
        mazes = archive[maze_key]
        starts = archive["starts"]
        goals = archive["goals"]
        if (
            mazes.ndim != 3
            or starts.shape != (len(mazes), 2)
            or goals.shape
            != (
                len(mazes),
                2,
            )
        ):
            raise StructuredProtocolError("maze NPZ array dimensions are inconsistent")

        def optional(name: str, index: int, default: Any) -> Any:
            if name not in archive:
                return default
            values = archive[name]
            if values.ndim == 0:
                return values.item()
            if len(values) != len(mazes):
                raise StructuredProtocolError(f"maze NPZ {name} length mismatch")
            value = values[index]
            return value.item() if isinstance(value, np.generic) else value

        def text(value: Any) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)

        records = []
        for index in range(len(mazes)):
            record: dict[str, Any] = {
                "maze": mazes[index],
                "start": starts[index],
                "goal": goals[index],
                "task_id": text(optional("task_ids", index, f"maze_{index:06d}")),
                "source_group": text(
                    optional("source_groups", index, f"maze_{index:06d}")
                ),
                "augmentation_group": text(
                    optional("augmentation_groups", index, f"maze_{index:06d}")
                ),
                "wall_value": optional("wall_values", index, 1),
            }
            for input_name, output_name in (
                ("splits", "split"),
                ("shortest_distances", "shortest_distance"),
            ):
                if input_name in archive:
                    value = optional(input_name, index, None)
                    record[output_name] = (
                        text(value) if input_name == "splits" else value
                    )
            records.append(record)
    return records


def load_maze_tasks(
    path: str | Path,
    *,
    split: str | None = "test",
) -> StructuredDataset:
    """Load maze records from JSONL or non-pickled NPZ arrays."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        records = _jsonl_records(source)
        source_format = "maze_jsonl_v1"
    elif suffix == ".npz":
        records = _npz_records(source)
        source_format = "maze_npz_v1"
    else:
        raise StructuredProtocolError("maze loader accepts only .jsonl or .npz")
    if not records:
        raise StructuredProtocolError("maze dataset must not be empty")
    task_target_pairs = [
        _record_to_task(
            record,
            index=index,
            default_split=split,
            source_file=source.name,
            source_format=source_format,
        )
        for index, record in enumerate(records)
    ]
    tasks, targets = zip(*task_target_pairs, strict=True)
    return build_structured_dataset(tasks, targets, scorer=score_maze_prediction)


load_maze_dataset = load_maze_tasks
