"""Deterministic, target-free proposal generators for structured tasks.

The generators are deliberately small and auditable.  They turn a public task
into executable predictions plus features computed from demonstrations or from
the prediction itself.  Hidden query targets are never accepted by this
module.  Consequently an experiment can measure both proposal coverage and
selection accuracy without confusing the two.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from src.data.structured_protocol import PublicTask


FEATURE_NAMES = (
    "support_exact_rate",
    "support_shape_rate",
    "support_cell_accuracy",
    "support_color_jaccard",
    "prediction_valid",
    "prediction_complete",
    "normalized_cost",
    "normalized_complexity",
    "input_height",
    "input_width",
    "input_density",
    "output_height_ratio",
    "output_width_ratio",
    "output_color_count",
    "path_reaches_goal",
    "path_length_ratio",
    "path_turn_ratio",
    "sudoku_filled_fraction",
    "sudoku_constraint_fraction",
    "bias",
)


class ProposalError(ValueError):
    """Raised when a public task cannot be converted into proposals."""


def _freeze_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, slots=True)
class ProposalBatch:
    """A target-free, immutable and compute-accounted proposal set."""

    task_id: str
    family: str
    candidate_ids: tuple[str, ...]
    features: np.ndarray
    outputs: tuple[Any, ...]
    compute_costs: np.ndarray
    generator_version: str = "structured_proposals_v1"

    def __post_init__(self) -> None:
        identifiers = tuple(str(item) for item in self.candidate_ids)
        if not identifiers or any(not item for item in identifiers):
            raise ProposalError("candidate_ids must be non-empty strings")
        if len(set(identifiers)) != len(identifiers):
            raise ProposalError("candidate_ids must be unique")
        features = np.asarray(self.features, dtype=float)
        costs = np.asarray(self.compute_costs, dtype=float)
        if features.shape != (len(identifiers), len(FEATURE_NAMES)):
            raise ProposalError(
                f"features must have shape (n_candidates, {len(FEATURE_NAMES)})"
            )
        if not np.isfinite(features).all():
            raise ProposalError("proposal features must be finite")
        if costs.shape != (len(identifiers),) or not np.isfinite(costs).all():
            raise ProposalError("compute_costs must be one finite value per proposal")
        if np.any(costs <= 0.0):
            raise ProposalError("compute costs must be positive")
        outputs = tuple(_freeze_output(item) for item in self.outputs)
        if len(outputs) != len(identifiers):
            raise ProposalError("outputs must match candidate_ids")
        object.__setattr__(self, "candidate_ids", identifiers)
        object.__setattr__(self, "features", _freeze_array(features, dtype=float))
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "compute_costs", _freeze_array(costs, dtype=float))

    @property
    def candidate_fingerprint(self) -> str:
        payload = {
            "task_id": self.task_id,
            "family": self.family,
            "candidate_ids": self.candidate_ids,
            "features": self.features,
            "outputs": self.outputs,
            "compute_costs": self.compute_costs,
            "generator_version": self.generator_version,
        }
        encoded = json.dumps(
            _jsonable(payload), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def matched_compute_budget(self) -> float:
        """Budget charged to every selector irrespective of its chosen output."""

        return float(np.sum(self.compute_costs))


def _freeze_output(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _freeze_array(value)
    if isinstance(value, Mapping):
        return {str(key): _freeze_output(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_output(item) for item in value)
    return value


def _grid(value: Any, *, name: str, max_value: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.size == 0:
        raise ProposalError(f"{name} must be a non-empty two-dimensional grid")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.integer
    ):
        raise ProposalError(f"{name} must contain integers")
    result = np.asarray(raw, dtype=int)
    if np.any(result < 0) or (max_value is not None and np.any(result > max_value)):
        raise ProposalError(f"{name} contains values outside the allowed range")
    return result


def _cell_accuracy(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 0.0
    return float(np.mean(left == right))


def _color_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a, b = set(np.unique(left).tolist()), set(np.unique(right).tolist())
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def _crop_nonzero(value: np.ndarray) -> np.ndarray:
    positions = np.argwhere(value != 0)
    if not len(positions):
        return value.copy()
    low, high = positions.min(axis=0), positions.max(axis=0) + 1
    return value[low[0] : high[0], low[1] : high[1]]


def _crop_color(value: np.ndarray, color: int) -> np.ndarray:
    positions = np.argwhere(value == color)
    if not len(positions):
        return value.copy()
    low, high = positions.min(axis=0), positions.max(axis=0) + 1
    return value[low[0] : high[0], low[1] : high[1]]


def _majority_downsample(value: np.ndarray, row_factor: int, col_factor: int) -> np.ndarray:
    if value.shape[0] % row_factor or value.shape[1] % col_factor:
        return value.copy()
    reshaped = value.reshape(
        value.shape[0] // row_factor,
        row_factor,
        value.shape[1] // col_factor,
        col_factor,
    )
    output = np.empty((reshaped.shape[0], reshaped.shape[2]), dtype=int)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            values, counts = np.unique(reshaped[row, :, col, :], return_counts=True)
            output[row, col] = int(values[np.argmax(counts)])
    return output


def _learn_color_map(
    transformed: Sequence[np.ndarray], targets: Sequence[np.ndarray]
) -> dict[int, int]:
    counts: dict[int, dict[int, int]] = {}
    for source, target in zip(transformed, targets, strict=True):
        if source.shape != target.shape:
            continue
        for source_value, target_value in zip(source.ravel(), target.ravel(), strict=True):
            by_target = counts.setdefault(int(source_value), {})
            by_target[int(target_value)] = by_target.get(int(target_value), 0) + 1
    return {
        source: max(by_target, key=lambda target: (by_target[target], -target))
        for source, by_target in counts.items()
    }


def _apply_color_map(value: np.ndarray, mapping: Mapping[int, int]) -> np.ndarray:
    output = value.copy()
    for source, target in mapping.items():
        output[value == int(source)] = int(target)
    return output


def _arc_feature_row(
    support_predictions: Sequence[np.ndarray],
    support_targets: Sequence[np.ndarray],
    query_input: np.ndarray,
    query_output: np.ndarray,
    *,
    cost: float,
    complexity: float,
) -> np.ndarray:
    exact = np.mean(
        [
            prediction.shape == target.shape and np.array_equal(prediction, target)
            for prediction, target in zip(
                support_predictions, support_targets, strict=True
            )
        ]
    )
    shape = np.mean(
        [
            prediction.shape == target.shape
            for prediction, target in zip(
                support_predictions, support_targets, strict=True
            )
        ]
    )
    cell = np.mean(
        [
            _cell_accuracy(prediction, target)
            for prediction, target in zip(
                support_predictions, support_targets, strict=True
            )
        ]
    )
    colors = np.mean(
        [
            _color_jaccard(prediction, target)
            for prediction, target in zip(
                support_predictions, support_targets, strict=True
            )
        ]
    )
    area = max(1, query_input.size)
    return np.asarray(
        [
            exact,
            shape,
            cell,
            colors,
            1.0,
            1.0,
            np.log1p(cost) / 10.0,
            complexity / 10.0,
            query_input.shape[0] / 30.0,
            query_input.shape[1] / 30.0,
            np.count_nonzero(query_input) / area,
            query_output.shape[0] / query_input.shape[0],
            query_output.shape[1] / query_input.shape[1],
            len(np.unique(query_output)) / 10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )


def generate_arc_proposals(task: PublicTask, *, max_candidates: int = 96) -> ProposalBatch:
    """Infer a compact program library from ARC demonstrations only."""

    if task.family != "arc":
        raise ProposalError("generate_arc_proposals requires an ARC task")
    support_inputs = tuple(
        _grid(item, name="ARC support input", max_value=9)
        for item in task.support_inputs
    )
    support_outputs = tuple(
        _grid(item, name="ARC support output", max_value=9)
        for item in task.support_outputs
    )
    query_inputs = tuple(
        _grid(item, name="ARC query input", max_value=9)
        for item in task.query_inputs
    )
    if not support_inputs or len(support_inputs) != len(support_outputs):
        raise ProposalError("ARC tasks require matching non-empty demonstrations")
    if not query_inputs:
        raise ProposalError("ARC tasks require at least one query input")

    transform_specs: list[tuple[str, float, Callable[[np.ndarray], np.ndarray]]] = [
        ("identity", 0.0, lambda value: value.copy()),
        ("rot90", 1.0, lambda value: np.rot90(value, 1)),
        ("rot180", 1.0, lambda value: np.rot90(value, 2)),
        ("rot270", 1.0, lambda value: np.rot90(value, 3)),
        ("flip_lr", 1.0, np.fliplr),
        ("flip_ud", 1.0, np.flipud),
        ("transpose", 1.0, np.transpose),
        ("anti_transpose", 2.0, lambda value: np.rot90(value.T, 2)),
        ("crop_nonzero", 2.0, _crop_nonzero),
    ]
    observed_colors = sorted(
        {
            int(color)
            for value in support_inputs
            for color in np.unique(value)
            if int(color) != 0
        }
    )
    for color in observed_colors:
        transform_specs.append(
            (
                f"crop_color_{color}",
                3.0,
                lambda value, selected=color: _crop_color(value, selected),
            )
        )
    for row_factor in range(1, 5):
        for col_factor in range(1, 5):
            if row_factor == col_factor == 1:
                continue
            transform_specs.append(
                (
                    f"repeat_{row_factor}x{col_factor}",
                    2.0 + 0.25 * (row_factor + col_factor),
                    lambda value, rf=row_factor, cf=col_factor: np.repeat(
                        np.repeat(value, rf, axis=0), cf, axis=1
                    ),
                )
            )
            transform_specs.append(
                (
                    f"downsample_{row_factor}x{col_factor}",
                    3.0 + 0.25 * (row_factor + col_factor),
                    lambda value, rf=row_factor, cf=col_factor: _majority_downsample(
                        value, rf, cf
                    ),
                )
            )

    records: list[tuple[str, tuple[np.ndarray, ...], np.ndarray, float]] = []
    seen_outputs: set[str] = set()
    for name, complexity, transform in transform_specs:
        transformed_support = tuple(np.asarray(transform(value), dtype=int) for value in support_inputs)
        transformed_query = tuple(np.asarray(transform(value), dtype=int) for value in query_inputs)
        variants: list[tuple[str, tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]] = [
            (name, transformed_support, transformed_query, complexity)
        ]
        mapping = _learn_color_map(transformed_support, support_outputs)
        if mapping and any(source != target for source, target in mapping.items()):
            mapped_support = tuple(
                _apply_color_map(value, mapping) for value in transformed_support
            )
            mapped_query = tuple(
                _apply_color_map(value, mapping) for value in transformed_query
            )
            encoded_map = "_".join(f"{a}to{b}" for a, b in sorted(mapping.items()))
            variants.append(
                (
                    f"{name}__map_{encoded_map}",
                    mapped_support,
                    mapped_query,
                    complexity + 1.0,
                )
            )
        for variant_name, support_prediction, query_prediction, variant_complexity in variants:
            digest = hashlib.sha256(
                json.dumps(_jsonable(query_prediction), sort_keys=True).encode("utf-8")
            ).hexdigest()
            if digest in seen_outputs:
                continue
            seen_outputs.add(digest)
            cost = float(
                sum(value.size for value in support_inputs)
                + sum(value.size for value in query_inputs)
            ) * (1.0 + variant_complexity)
            feature_rows = [
                _arc_feature_row(
                    support_prediction,
                    support_outputs,
                    query_input,
                    query_output,
                    cost=cost,
                    complexity=variant_complexity,
                )
                for query_input, query_output in zip(
                    query_inputs, query_prediction, strict=True
                )
            ]
            records.append(
                (
                    variant_name,
                    tuple(query_prediction),
                    np.mean(np.stack(feature_rows), axis=0),
                    cost,
                )
            )
    records.sort(key=lambda item: (-item[2][0], -item[2][2], item[0]))
    records = records[:max_candidates]
    return ProposalBatch(
        task.task_id,
        task.family,
        tuple(item[0] for item in records),
        np.stack([item[2] for item in records]),
        tuple(item[1] for item in records),
        np.asarray([item[3] for item in records]),
    )


def _maze_payload(task: PublicTask) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    payload: Any = task.query
    if "grid" not in payload and "maze" not in payload and task.query_inputs:
        first = task.query_inputs[0]
        payload = first if isinstance(first, Mapping) else {"grid": first}
    grid_value = payload.get("grid", payload.get("maze"))
    if grid_value is None:
        raise ProposalError("maze query requires grid/maze")
    grid = _grid(grid_value, name="maze grid")
    wall_value = int(payload.get("wall_value", task.metadata.get("wall_value", 1)))
    start_value = int(payload.get("start_value", task.metadata.get("start_value", 2)))
    goal_value = int(payload.get("goal_value", task.metadata.get("goal_value", 3)))
    start_raw = payload.get("start")
    goal_raw = payload.get("goal")
    start_positions = np.argwhere(grid == start_value)
    goal_positions = np.argwhere(grid == goal_value)
    start = (
        tuple(int(item) for item in start_raw)
        if start_raw is not None
        else tuple(int(item) for item in start_positions[0])
        if len(start_positions) == 1
        else None
    )
    goal = (
        tuple(int(item) for item in goal_raw)
        if goal_raw is not None
        else tuple(int(item) for item in goal_positions[0])
        if len(goal_positions) == 1
        else None
    )
    if start is None or goal is None or len(start) != 2 or len(goal) != 2:
        raise ProposalError("maze requires one start and one goal")
    traversable = grid != wall_value
    if not traversable[start] or not traversable[goal]:
        raise ProposalError("maze start/goal cannot be walls")
    return traversable, start, goal


def _neighbors(
    position: tuple[int, int], traversable: np.ndarray
) -> Iterable[tuple[int, int]]:
    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        row, col = position[0] + dr, position[1] + dc
        if (
            0 <= row < traversable.shape[0]
            and 0 <= col < traversable.shape[1]
            and traversable[row, col]
        ):
            yield row, col


def _reconstruct_path(
    parents: Mapping[tuple[int, int], tuple[int, int] | None],
    goal: tuple[int, int],
) -> np.ndarray:
    if goal not in parents:
        return np.empty((0, 2), dtype=int)
    path: list[tuple[int, int]] = []
    current: tuple[int, int] | None = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    return np.asarray(path[::-1], dtype=int)


def _search_maze(
    traversable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    algorithm: str,
) -> tuple[np.ndarray, int]:
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    expanded = 0
    if algorithm == "bfs":
        frontier: Any = deque([start])
        pop = frontier.popleft
        push = frontier.append
    elif algorithm == "dfs":
        frontier = [start]
        pop = frontier.pop
        push = frontier.append
    else:
        frontier_heap: list[tuple[float, int, tuple[int, int]]] = []
        serial = 0
        heapq.heappush(frontier_heap, (0.0, serial, start))
        distances = {start: 0}
        while frontier_heap:
            _, _, current = heapq.heappop(frontier_heap)
            expanded += 1
            if current == goal:
                return _reconstruct_path(parents, goal), expanded
            for neighbor in _neighbors(current, traversable):
                candidate_distance = distances[current] + 1
                if neighbor in distances and candidate_distance >= distances[neighbor]:
                    continue
                distances[neighbor] = candidate_distance
                parents[neighbor] = current
                heuristic = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                priority = heuristic + (candidate_distance if algorithm == "astar" else 0)
                serial += 1
                heapq.heappush(frontier_heap, (priority, serial, neighbor))
        return np.empty((0, 2), dtype=int), expanded
    while frontier:
        current = pop()
        expanded += 1
        if current == goal:
            return _reconstruct_path(parents, goal), expanded
        for neighbor in _neighbors(current, traversable):
            if neighbor not in parents:
                parents[neighbor] = current
                push(neighbor)
    return np.empty((0, 2), dtype=int), expanded


def _valid_maze_path(
    path: np.ndarray,
    traversable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    if path.ndim != 2 or path.shape[1:] != (2,) or len(path) == 0:
        return False
    if tuple(path[0]) != start or tuple(path[-1]) != goal:
        return False
    if np.any(path < 0) or np.any(path[:, 0] >= traversable.shape[0]) or np.any(
        path[:, 1] >= traversable.shape[1]
    ):
        return False
    if not np.all(traversable[path[:, 0], path[:, 1]]):
        return False
    return bool(np.all(np.abs(np.diff(path, axis=0)).sum(axis=1) == 1))


def generate_maze_proposals(task: PublicTask) -> ProposalBatch:
    """Run matched deterministic search policies and expose their accounting."""

    if task.family != "maze":
        raise ProposalError("generate_maze_proposals requires a maze task")
    traversable, start, goal = _maze_payload(task)
    records = []
    for algorithm, complexity in (("bfs", 1.0), ("astar", 2.0), ("greedy", 1.5), ("dfs", 1.0)):
        path, expanded = _search_maze(
            traversable, start, goal, algorithm=algorithm
        )
        valid = _valid_maze_path(path, traversable, start, goal)
        steps = max(0, len(path) - 1)
        turns = (
            int(np.sum(np.any(np.diff(path, axis=0)[1:] != np.diff(path, axis=0)[:-1], axis=1)))
            if len(path) > 2
            else 0
        )
        free = max(1, int(np.sum(traversable)))
        feature = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                valid,
                valid,
                np.log1p(expanded) / 10.0,
                complexity / 10.0,
                traversable.shape[0] / 128.0,
                traversable.shape[1] / 128.0,
                1.0 - np.mean(traversable),
                1.0,
                1.0,
                0.0,
                valid,
                steps / free,
                turns / max(1, steps),
                0.0,
                0.0,
                1.0,
            ],
            dtype=float,
        )
        records.append((algorithm, path, feature, float(max(1, expanded))))
    return ProposalBatch(
        task.task_id,
        task.family,
        tuple(item[0] for item in records),
        np.stack([item[2] for item in records]),
        tuple(item[1] for item in records),
        np.asarray([item[3] for item in records]),
    )


def _sudoku_constraint_fraction(grid: np.ndarray) -> float:
    groups = [*grid, *grid.T]
    box = int(round(np.sqrt(grid.shape[0])))
    if box * box == grid.shape[0]:
        groups.extend(
            grid[row : row + box, col : col + box].ravel()
            for row in range(0, grid.shape[0], box)
            for col in range(0, grid.shape[1], box)
        )
    valid = 0
    for values in groups:
        filled = values[values > 0]
        valid += int(len(filled) == len(np.unique(filled)))
    return valid / len(groups)


def _solve_sudoku(
    puzzle: np.ndarray, *, node_budget: int, use_mrv: bool
) -> tuple[np.ndarray, int, bool]:
    size = puzzle.shape[0]
    box = int(round(np.sqrt(size)))
    grid = puzzle.copy()
    nodes = 0

    def allowed(row: int, col: int) -> list[int]:
        occupied = set(grid[row]) | set(grid[:, col])
        if box * box == size:
            occupied |= set(
                grid[
                    (row // box) * box : (row // box + 1) * box,
                    (col // box) * box : (col // box + 1) * box,
                ].ravel()
            )
        return [digit for digit in range(1, size + 1) if digit not in occupied]

    def search() -> bool:
        nonlocal nodes
        if nodes >= node_budget:
            return False
        empty = np.argwhere(grid == 0)
        if not len(empty):
            return True
        choices = []
        for row_raw, col_raw in empty:
            row, col = int(row_raw), int(col_raw)
            values = allowed(row, col)
            if not values:
                return False
            choices.append((len(values), row, col, values))
            if not use_mrv:
                break
        _, row, col, values = min(choices) if use_mrv else choices[0]
        for digit in values:
            if nodes >= node_budget:
                break
            nodes += 1
            grid[row, col] = digit
            if search():
                return True
            grid[row, col] = 0
        return False

    solved = search()
    return grid, nodes, solved


def generate_sudoku_proposals(task: PublicTask) -> ProposalBatch:
    """Generate bounded-search Sudoku outputs without consulting the solution."""

    if task.family != "sudoku":
        raise ProposalError("generate_sudoku_proposals requires a Sudoku task")
    payload: Any = task.query
    if "grid" not in payload and "puzzle" not in payload and task.query_inputs:
        first = task.query_inputs[0]
        payload = first if isinstance(first, Mapping) else {"grid": first}
    puzzle_value = payload.get("grid", payload.get("puzzle"))
    if puzzle_value is None:
        raise ProposalError("Sudoku query requires grid/puzzle")
    puzzle = _grid(puzzle_value, name="Sudoku puzzle")
    if puzzle.shape[0] != puzzle.shape[1]:
        raise ProposalError("Sudoku grid must be square")
    size = puzzle.shape[0]
    box = int(round(np.sqrt(size)))
    if box * box != size or np.any(puzzle > size):
        raise ProposalError("Sudoku size must have integral boxes and valid digits")
    records = []
    for use_mrv, budget in (
        (False, 64),
        (True, 64),
        (True, 512),
        (True, 4096),
        (True, 100_000),
    ):
        output, nodes, solved = _solve_sudoku(
            puzzle, node_budget=budget, use_mrv=use_mrv
        )
        clue_ok = bool(np.all(output[puzzle > 0] == puzzle[puzzle > 0]))
        constraint = _sudoku_constraint_fraction(output)
        complete = bool(solved and np.all(output > 0) and clue_ok and constraint == 1.0)
        identifier = f"{'mrv' if use_mrv else 'sequential'}_budget_{budget}"
        feature = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                clue_ok and constraint == 1.0,
                complete,
                np.log1p(nodes) / 15.0,
                (1.0 if use_mrv else 0.5) + np.log10(budget) / 10.0,
                size / 16.0,
                size / 16.0,
                np.count_nonzero(puzzle) / puzzle.size,
                1.0,
                1.0,
                len(np.unique(output[output > 0])) / size,
                0.0,
                0.0,
                0.0,
                np.count_nonzero(output) / output.size,
                constraint,
                1.0,
            ],
            dtype=float,
        )
        records.append((identifier, output, feature, float(max(1, nodes))))
    return ProposalBatch(
        task.task_id,
        task.family,
        tuple(item[0] for item in records),
        np.stack([item[2] for item in records]),
        tuple(item[1] for item in records),
        np.asarray([item[3] for item in records]),
    )


def generate_structured_proposals(task: PublicTask, **kwargs: Any) -> ProposalBatch:
    """Dispatch to the registered target-free family generator."""

    if not isinstance(task, PublicTask):
        raise TypeError("task must be a PublicTask")
    if task.family == "arc":
        return generate_arc_proposals(task, **kwargs)
    if kwargs:
        raise TypeError(f"unexpected proposal options for {task.family}: {sorted(kwargs)}")
    if task.family == "maze":
        return generate_maze_proposals(task)
    if task.family == "sudoku":
        return generate_sudoku_proposals(task)
    raise ProposalError(f"no proposal generator registered for family {task.family!r}")
