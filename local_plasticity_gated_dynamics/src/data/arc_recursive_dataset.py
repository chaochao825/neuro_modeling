"""Leakage-safe ARC tensors and reversible test-time augmentations.

This module is intentionally independent from the older proposal-selection
pipeline.  It packs variable ARC grids for a direct grid decoder, keeps every
inner split at the task/source-group level, and exposes test demonstrations
without ever requesting a test target capability.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.structured_protocol import PublicTask, StructuredDataset


ARC_COLORS = 10
ARC_PAD_TOKEN = 10
ARC_TARGET_IGNORE = -100


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _arc_grid(value: object, *, name: str, max_grid_size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape, default=0) < 1:
        raise ValueError(f"{name} must be a non-empty 2D grid")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integer colors")
    array = np.asarray(array, dtype=np.int64)
    if max(array.shape) > max_grid_size:
        raise ValueError(f"{name} exceeds max_grid_size={max_grid_size}")
    if np.any((array < 0) | (array >= ARC_COLORS)):
        raise ValueError(f"{name} colors must lie in [0, 9]")
    return array


@dataclass(frozen=True, eq=False)
class ARCTransform:
    """One exactly invertible D4 plus color-permutation transformation."""

    rotation_quarters: int = 0
    reflect: bool = False
    color_permutation: tuple[int, ...] = tuple(range(ARC_COLORS))

    def __post_init__(self) -> None:
        rotation = _integer(
            self.rotation_quarters, name="rotation_quarters", minimum=0
        )
        if rotation > 3:
            raise ValueError("rotation_quarters must lie in [0, 3]")
        if not isinstance(self.reflect, (bool, np.bool_)):
            raise TypeError("reflect must be boolean")
        colors = tuple(int(value) for value in self.color_permutation)
        if len(colors) != ARC_COLORS or set(colors) != set(range(ARC_COLORS)):
            raise ValueError("color_permutation must be a permutation of 0..9")
        object.__setattr__(self, "rotation_quarters", rotation)
        object.__setattr__(self, "reflect", bool(self.reflect))
        object.__setattr__(self, "color_permutation", colors)

    @property
    def fingerprint(self) -> str:
        payload = bytes(
            (self.rotation_quarters, int(self.reflect), *self.color_permutation)
        )
        return hashlib.sha256(payload).hexdigest()

    def apply(self, grid: object) -> np.ndarray:
        values = np.asarray(grid)
        if values.ndim != 2 or values.dtype.kind not in {"i", "u"}:
            raise ValueError("grid must be a 2D integer array")
        if np.any((values < 0) | (values >= ARC_COLORS)):
            raise ValueError("grid colors must lie in [0, 9]")
        mapped = np.asarray(self.color_permutation, dtype=np.int64)[values]
        transformed = np.rot90(mapped, self.rotation_quarters)
        if self.reflect:
            transformed = np.fliplr(transformed)
        return np.ascontiguousarray(transformed, dtype=np.int64)

    def invert(self, grid: object) -> np.ndarray:
        values = np.asarray(grid)
        if values.ndim != 2 or values.dtype.kind not in {"i", "u"}:
            raise ValueError("grid must be a 2D integer array")
        if np.any((values < 0) | (values >= ARC_COLORS)):
            raise ValueError("grid colors must lie in [0, 9]")
        restored = np.fliplr(values) if self.reflect else values
        restored = np.rot90(restored, -self.rotation_quarters)
        inverse = np.empty(ARC_COLORS, dtype=np.int64)
        inverse[np.asarray(self.color_permutation, dtype=np.int64)] = np.arange(
            ARC_COLORS
        )
        return np.ascontiguousarray(inverse[restored], dtype=np.int64)


def seeded_arc_transforms(
    *,
    count: int,
    seed: int,
    include_identity: bool = True,
    permute_background: bool = False,
) -> tuple[ARCTransform, ...]:
    """Return deterministic, reversible augmentations with stable fingerprints."""

    count = _integer(count, name="count", minimum=1)
    seed = _integer(seed, name="seed", minimum=0)
    if not isinstance(include_identity, (bool, np.bool_)):
        raise TypeError("include_identity must be boolean")
    if not isinstance(permute_background, (bool, np.bool_)):
        raise TypeError("permute_background must be boolean")
    rng = np.random.default_rng(seed)
    transforms: list[ARCTransform] = []
    if include_identity:
        transforms.append(ARCTransform())
    seen = {item.fingerprint for item in transforms}
    max_unique = 8 * (3_628_800 if permute_background else 362_880)
    if count > max_unique:
        raise ValueError("requested more unique ARC transforms than exist")
    while len(transforms) < count:
        colors = np.arange(ARC_COLORS, dtype=np.int64)
        if permute_background:
            colors = rng.permutation(colors)
        else:
            colors[1:] = rng.permutation(colors[1:])
        candidate = ARCTransform(
            rotation_quarters=int(rng.integers(0, 4)),
            reflect=bool(rng.integers(0, 2)),
            color_permutation=tuple(int(value) for value in colors),
        )
        if candidate.fingerprint not in seen:
            seen.add(candidate.fingerprint)
            transforms.append(candidate)
    return tuple(transforms)


def pack_arc_grid(
    grid: object, *, max_grid_size: int, pad_value: int = ARC_PAD_TOKEN
) -> tuple[np.ndarray, tuple[int, int]]:
    """Top-left pack one grid into a fixed square token sequence."""

    max_grid_size = _integer(max_grid_size, name="max_grid_size", minimum=1)
    values = _arc_grid(grid, name="grid", max_grid_size=max_grid_size)
    packed = np.full((max_grid_size, max_grid_size), int(pad_value), dtype=np.int64)
    height, width = values.shape
    packed[:height, :width] = values
    return packed.reshape(-1), (int(height), int(width))


def pack_arc_target(
    grid: object, *, max_grid_size: int
) -> tuple[np.ndarray, tuple[int, int]]:
    return pack_arc_grid(
        grid, max_grid_size=max_grid_size, pad_value=ARC_TARGET_IGNORE
    )


def unpack_arc_grid(
    packed: object, shape: Sequence[int], *, max_grid_size: int
) -> np.ndarray:
    max_grid_size = _integer(max_grid_size, name="max_grid_size", minimum=1)
    values = np.asarray(packed)
    if values.shape != (max_grid_size * max_grid_size,):
        raise ValueError("packed grid has the wrong length")
    if len(shape) != 2:
        raise ValueError("shape must contain height and width")
    height = _integer(shape[0], name="height", minimum=1)
    width = _integer(shape[1], name="width", minimum=1)
    if height > max_grid_size or width > max_grid_size:
        raise ValueError("shape exceeds max_grid_size")
    return np.array(
        values.reshape(max_grid_size, max_grid_size)[:height, :width], copy=True
    )


@dataclass(frozen=True, eq=False)
class ARCGridExamples:
    """Pair-major tensors with task-level independence metadata."""

    inputs: np.ndarray
    targets: np.ndarray
    input_shapes: np.ndarray
    target_shapes: np.ndarray
    puzzle_indices: np.ndarray
    example_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    source_groups: tuple[str, ...]
    augmentation_groups: tuple[str, ...]
    transform_fingerprints: tuple[str, ...]
    max_grid_size: int
    name: str

    def __post_init__(self) -> None:
        size = _integer(self.max_grid_size, name="max_grid_size", minimum=1)
        inputs = np.asarray(self.inputs)
        targets = np.asarray(self.targets)
        input_shapes = np.asarray(self.input_shapes)
        target_shapes = np.asarray(self.target_shapes)
        puzzle_indices = np.asarray(self.puzzle_indices)
        count = inputs.shape[0] if inputs.ndim == 2 else 0
        if count < 1 or inputs.shape != (count, size * size):
            raise ValueError("inputs must be a non-empty packed grid matrix")
        if targets.shape != inputs.shape:
            raise ValueError("targets must align with inputs")
        if input_shapes.shape != (count, 2) or target_shapes.shape != (count, 2):
            raise ValueError("input/target shapes must be [example, 2]")
        if puzzle_indices.shape != (count,):
            raise ValueError("puzzle_indices must align with examples")
        metadata = (
            self.example_ids,
            self.task_ids,
            self.source_groups,
            self.augmentation_groups,
            self.transform_fingerprints,
        )
        if any(len(field) != count for field in metadata):
            raise ValueError("example metadata must align with arrays")
        if len(set(self.example_ids)) != count:
            raise ValueError("example_ids must be unique")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be non-empty")
        inputs = np.asarray(inputs, dtype=np.int64).copy()
        targets = np.asarray(targets, dtype=np.int64).copy()
        input_shapes = np.asarray(input_shapes, dtype=np.int64).copy()
        target_shapes = np.asarray(target_shapes, dtype=np.int64).copy()
        puzzle_indices = np.asarray(puzzle_indices, dtype=np.int64).copy()
        if np.any((inputs < 0) | (inputs > ARC_PAD_TOKEN)):
            raise ValueError("input tokens are invalid")
        valid_targets = targets != ARC_TARGET_IGNORE
        if np.any((targets[valid_targets] < 0) | (targets[valid_targets] >= 10)):
            raise ValueError("target colors are invalid")
        if np.any((input_shapes < 1) | (input_shapes > size)) or np.any(
            (target_shapes < 1) | (target_shapes > size)
        ):
            raise ValueError("grid shapes are invalid")
        if np.any(puzzle_indices < 0):
            raise ValueError("puzzle indices must be non-negative")
        for array in (inputs, targets, input_shapes, target_shapes, puzzle_indices):
            array.setflags(write=False)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "input_shapes", input_shapes)
        object.__setattr__(self, "target_shapes", target_shapes)
        object.__setattr__(self, "puzzle_indices", puzzle_indices)

    @property
    def n_puzzles(self) -> int:
        return int(np.max(self.puzzle_indices)) + 1


def split_arc_training_tasks(
    dataset: StructuredDataset, *, validation_fraction: float, seed: int
) -> tuple[tuple[PublicTask, ...], tuple[PublicTask, ...]]:
    """Split connected source/augmentation/content groups, never grid cells."""

    if not isinstance(dataset, StructuredDataset):
        raise TypeError("dataset must be StructuredDataset")
    seed = _integer(seed, name="seed", minimum=0)
    fraction = float(validation_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    tasks = dataset.for_split("train")
    if len(tasks) < 2 or any(task.family != "arc" for task in tasks):
        raise ValueError("at least two ARC training tasks are required")
    parent = list(range(len(tasks)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[tuple[str, str], int] = {}
    for index, task in enumerate(tasks):
        for key in (
            ("source", task.source_group),
            ("augmentation", task.augmentation_group),
            ("content", task.content_group),
        ):
            union(index, seen.setdefault(key, index))
    components: dict[int, list[int]] = {}
    for index in range(len(tasks)):
        components.setdefault(find(index), []).append(index)
    groups = tuple(sorted(components.values(), key=lambda item: tuple(item)))
    if len(groups) < 2:
        raise ValueError("at least two independent ARC components are required")
    n_validation = min(max(1, round(fraction * len(groups))), len(groups) - 1)
    rng = np.random.default_rng(seed)
    held_out = {
        task_index
        for group_index in rng.permutation(len(groups))[:n_validation]
        for task_index in groups[group_index]
    }
    training = tuple(task for index, task in enumerate(tasks) if index not in held_out)
    validation = tuple(task for index, task in enumerate(tasks) if index in held_out)
    for attribute in ("source_group", "augmentation_group", "content_group"):
        if {getattr(task, attribute) for task in training} & {
            getattr(task, attribute) for task in validation
        }:
            raise AssertionError(f"{attribute} unexpectedly overlaps")
    return training, validation


def _task_pairs(
    dataset: StructuredDataset | None,
    task: PublicTask,
    *,
    include_query_targets: bool,
) -> tuple[tuple[np.ndarray, np.ndarray, str], ...]:
    demonstrations = tuple(task.context.get("demonstrations", ()))
    pairs = [
        (
            np.asarray(item["input"], dtype=np.int64),
            np.asarray(item["output"], dtype=np.int64),
            f"support_{index:03d}",
        )
        for index, item in enumerate(demonstrations)
    ]
    if include_query_targets:
        if dataset is None or task.split == "test":
            raise ValueError("query targets require a non-test dataset capability")
        targets = tuple(dataset.target_store.training_view(task).query_targets)
        inputs = tuple(task.query.get("inputs", ()))
        if len(inputs) != len(targets):
            raise ValueError("ARC query inputs and targets do not align")
        pairs.extend(
            (
                np.asarray(input_grid, dtype=np.int64),
                np.asarray(output_grid, dtype=np.int64),
                f"query_{index:03d}",
            )
            for index, (input_grid, output_grid) in enumerate(
                zip(inputs, targets, strict=True)
            )
        )
    if not pairs:
        raise ValueError("ARC task has no public demonstrations")
    return tuple(pairs)


def build_arc_examples(
    dataset: StructuredDataset | None,
    tasks: Sequence[PublicTask],
    *,
    max_grid_size: int,
    augmentations_per_pair: int,
    seed: int,
    include_query_targets: bool,
    name: str,
) -> ARCGridExamples:
    """Build direct grid-to-grid examples from whole tasks.

    Test tasks are accepted only with ``include_query_targets=False``.  Their
    examples therefore contain public demonstration outputs and cannot expose
    a hidden query target through this API.
    """

    max_grid_size = _integer(max_grid_size, name="max_grid_size", minimum=1)
    augmentations = _integer(
        augmentations_per_pair, name="augmentations_per_pair", minimum=0
    )
    seed = _integer(seed, name="seed", minimum=0)
    tasks = tuple(tasks)
    if not tasks or any(task.family != "arc" for task in tasks):
        raise ValueError("tasks must be a non-empty ARC task sequence")
    if include_query_targets and any(task.split == "test" for task in tasks):
        raise ValueError("test query targets are unavailable")
    transforms = seeded_arc_transforms(
        count=augmentations + 1,
        seed=seed,
        include_identity=True,
    )
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    input_shapes: list[tuple[int, int]] = []
    target_shapes: list[tuple[int, int]] = []
    puzzle_indices: list[int] = []
    example_ids: list[str] = []
    task_ids: list[str] = []
    source_groups: list[str] = []
    augmentation_groups: list[str] = []
    fingerprints: list[str] = []
    for puzzle_index, task in enumerate(tasks):
        for input_grid, output_grid, pair_name in _task_pairs(
            dataset, task, include_query_targets=include_query_targets
        ):
            for transform_index, transform in enumerate(transforms):
                transformed_input = transform.apply(input_grid)
                transformed_output = transform.apply(output_grid)
                packed_input, input_shape = pack_arc_grid(
                    transformed_input, max_grid_size=max_grid_size
                )
                packed_target, target_shape = pack_arc_target(
                    transformed_output, max_grid_size=max_grid_size
                )
                inputs.append(packed_input)
                targets.append(packed_target)
                input_shapes.append(input_shape)
                target_shapes.append(target_shape)
                puzzle_indices.append(puzzle_index)
                example_ids.append(
                    f"{task.task_id}::{pair_name}::augmentation={transform_index:03d}"
                )
                task_ids.append(task.task_id)
                source_groups.append(task.source_group)
                augmentation_groups.append(task.augmentation_group)
                fingerprints.append(transform.fingerprint)
    return ARCGridExamples(
        inputs=np.stack(inputs),
        targets=np.stack(targets),
        input_shapes=np.asarray(input_shapes, dtype=np.int64),
        target_shapes=np.asarray(target_shapes, dtype=np.int64),
        puzzle_indices=np.asarray(puzzle_indices, dtype=np.int64),
        example_ids=tuple(example_ids),
        task_ids=tuple(task_ids),
        source_groups=tuple(source_groups),
        augmentation_groups=tuple(augmentation_groups),
        transform_fingerprints=tuple(fingerprints),
        max_grid_size=max_grid_size,
        name=name,
    )


def public_arc_support_examples(
    task: PublicTask,
    *,
    max_grid_size: int,
    augmentations_per_pair: int,
    seed: int,
) -> ARCGridExamples:
    """Return only the model-visible demonstrations for task-level TTA."""

    return build_arc_examples(
        None,
        (task,),
        max_grid_size=max_grid_size,
        augmentations_per_pair=augmentations_per_pair,
        seed=seed,
        include_query_targets=False,
        name="public_support_tta",
    )
