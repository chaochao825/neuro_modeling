"""Leakage-safe Sudoku arrays for the micro recursive-reasoning baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.structured_protocol import PublicTask, StructuredDataset


@dataclass(frozen=True, eq=False)
class SupervisedSudokuArrays:
    """A trial-major supervised split available only from non-test tasks."""

    inputs: np.ndarray
    targets: np.ndarray
    task_ids: tuple[str, ...]
    source_groups: tuple[str, ...]
    name: str

    def __post_init__(self) -> None:
        inputs = np.asarray(self.inputs)
        targets = np.asarray(self.targets)
        if (
            inputs.ndim != 2
            or inputs.shape[1] != 81
            or targets.shape != inputs.shape
            or inputs.dtype.kind not in {"i", "u"}
            or targets.dtype.kind not in {"i", "u"}
        ):
            raise ValueError("Sudoku arrays must be integer [task, 81] matrices")
        if len(inputs) < 1:
            raise ValueError("Sudoku split must not be empty")
        if len(self.task_ids) != len(inputs) or len(self.source_groups) != len(inputs):
            raise ValueError("task IDs and source groups must align with arrays")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task IDs must be unique within a split")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be non-empty")
        inputs = np.asarray(inputs, dtype=np.int64).copy()
        targets = np.asarray(targets, dtype=np.int64).copy()
        if (
            np.any((inputs < 0) | (inputs > 9))
            or np.any((targets < 1) | (targets > 9))
            or np.any(np.all(inputs > 0, axis=1))
        ):
            raise ValueError("Sudoku tokens are invalid or a puzzle has no blanks")
        visible = inputs > 0
        if not np.array_equal(inputs[visible], targets[visible]):
            raise ValueError("targets must preserve visible Sudoku clues")
        inputs.setflags(write=False)
        targets.setflags(write=False)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "task_ids", tuple(str(value) for value in self.task_ids))
        object.__setattr__(
            self, "source_groups", tuple(str(value) for value in self.source_groups)
        )


def _arrays_from_tasks(
    dataset: StructuredDataset,
    tasks: Sequence[PublicTask],
    *,
    name: str,
) -> SupervisedSudokuArrays:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    task_ids: list[str] = []
    source_groups: list[str] = []
    for task in tasks:
        if task.family != "sudoku" or task.split == "test":
            raise ValueError("supervised arrays require non-test Sudoku tasks")
        view = dataset.target_store.training_view(task)
        inputs.append(np.asarray(task.query["grid"], dtype=np.int64).reshape(-1))
        targets.append(np.asarray(view.query_targets, dtype=np.int64).reshape(-1))
        task_ids.append(task.task_id)
        source_groups.append(task.source_group)
    return SupervisedSudokuArrays(
        np.stack(inputs),
        np.stack(targets),
        tuple(task_ids),
        tuple(source_groups),
        name,
    )


def split_sudoku_training_tasks(
    dataset: StructuredDataset,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[SupervisedSudokuArrays, SupervisedSudokuArrays]:
    """Split complete source groups within the public training partition."""

    if not isinstance(dataset, StructuredDataset):
        raise TypeError("dataset must be StructuredDataset")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    if not isinstance(validation_fraction, (int, float, np.integer, np.floating)):
        raise TypeError("validation_fraction must be numeric")
    validation_fraction = float(validation_fraction)
    if not np.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    tasks = dataset.for_split("train")
    if len(tasks) < 2 or any(task.family != "sudoku" for task in tasks):
        raise ValueError("at least two training Sudoku tasks are required")
    groups = tuple(dict.fromkeys(task.source_group for task in tasks))
    if len(groups) < 2:
        raise ValueError("at least two independent source groups are required")
    n_validation = min(
        max(1, int(round(validation_fraction * len(groups)))), len(groups) - 1
    )
    rng = np.random.default_rng(int(seed))
    validation_groups = {
        groups[index] for index in rng.permutation(len(groups))[:n_validation]
    }
    training_tasks = tuple(
        task for task in tasks if task.source_group not in validation_groups
    )
    validation_tasks = tuple(
        task for task in tasks if task.source_group in validation_groups
    )
    training = _arrays_from_tasks(dataset, training_tasks, name="inner_train")
    validation = _arrays_from_tasks(
        dataset, validation_tasks, name="inner_validation"
    )
    if set(training.source_groups) & set(validation.source_groups):
        raise AssertionError("source-group split unexpectedly overlaps")
    return training, validation


def _sudoku_permutation(
    puzzle: np.ndarray,
    solution: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    base = 3
    bands = rng.permutation(base)
    stacks = rng.permutation(base)
    rows = np.concatenate(
        [band * base + rng.permutation(base) for band in bands]
    )
    columns = np.concatenate(
        [stack * base + rng.permutation(base) for stack in stacks]
    )
    transformed_puzzle = puzzle[np.ix_(rows, columns)]
    transformed_solution = solution[np.ix_(rows, columns)]
    if bool(rng.integers(0, 2)):
        transformed_puzzle = transformed_puzzle.T
        transformed_solution = transformed_solution.T
    mapping = np.arange(10, dtype=np.int64)
    mapping[1:] = rng.permutation(np.arange(1, 10))
    return mapping[transformed_puzzle], mapping[transformed_solution]


def augment_sudoku_training(
    split: SupervisedSudokuArrays,
    *,
    augmentations_per_task: int,
    seed: int,
) -> SupervisedSudokuArrays:
    """Apply only label-preserving Sudoku symmetries to an inner-train split."""

    if not isinstance(split, SupervisedSudokuArrays):
        raise TypeError("split must be SupervisedSudokuArrays")
    if split.name != "inner_train":
        raise ValueError("augmentation is restricted to the inner training split")
    if isinstance(augmentations_per_task, (bool, np.bool_)) or not isinstance(
        augmentations_per_task, (int, np.integer)
    ):
        raise TypeError("augmentations_per_task must be an integer")
    if int(augmentations_per_task) < 0:
        raise ValueError("augmentations_per_task must be non-negative")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(int(seed))
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    task_ids: list[str] = []
    groups: list[str] = []
    for index, (puzzle, solution, task_id, group) in enumerate(
        zip(
            split.inputs,
            split.targets,
            split.task_ids,
            split.source_groups,
            strict=True,
        )
    ):
        inputs.append(puzzle)
        targets.append(solution)
        task_ids.append(f"{task_id}::augmentation=000")
        groups.append(group)
        puzzle_grid = puzzle.reshape(9, 9)
        solution_grid = solution.reshape(9, 9)
        for augmentation in range(1, int(augmentations_per_task) + 1):
            augmented_puzzle, augmented_solution = _sudoku_permutation(
                puzzle_grid, solution_grid, rng
            )
            inputs.append(augmented_puzzle.reshape(-1))
            targets.append(augmented_solution.reshape(-1))
            task_ids.append(f"{task_id}::augmentation={augmentation:03d}")
            groups.append(group)
    # Augmented IDs are intentionally unique while source groups remain tied
    # to the original independent puzzle.
    return SupervisedSudokuArrays(
        np.stack(inputs),
        np.stack(targets),
        tuple(task_ids),
        tuple(groups),
        "inner_train_augmented",
    )


def public_sudoku_test_inputs(
    dataset: StructuredDataset,
) -> tuple[np.ndarray, tuple[PublicTask, ...]]:
    """Expose model-visible test puzzles without ever returning test targets."""

    if not isinstance(dataset, StructuredDataset):
        raise TypeError("dataset must be StructuredDataset")
    tasks = dataset.for_split("test")
    if not tasks or any(task.family != "sudoku" for task in tasks):
        raise ValueError("dataset needs at least one Sudoku test task")
    inputs = np.stack(
        [np.asarray(task.query["grid"], dtype=np.int64).reshape(-1) for task in tasks]
    )
    inputs.setflags(write=False)
    return inputs, tasks
