"""Capability-safe protocol shared by structured reasoning benchmarks.

The central invariant is that :class:`PublicTask` never carries a query
target.  Targets live in :class:`TargetStore`: train and validation tasks may
be converted to a :class:`TrainingTaskView`, while test targets can only be
consumed by a registered, family-specific scoring function.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


VALID_SPLITS = frozenset({"train", "validation", "test"})
_FORBIDDEN_QUERY_KEYS = frozenset(
    {
        "answer",
        "answers",
        "label",
        "labels",
        "output",
        "outputs",
        "shortest_distance",
        "shortest_path",
        "shortest_paths",
        "solution",
        "solutions",
        "target",
        "targets",
    }
)


class StructuredProtocolError(ValueError):
    """Raised when structured-task data violate the public protocol."""


class CapabilityError(PermissionError):
    """Raised when code requests a capability unavailable for a task split."""


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredProtocolError(f"{name} must be a non-empty string")
    return value.strip()


def freeze_value(value: Any) -> Any:
    """Recursively copy data into immutable containers and read-only arrays."""

    if isinstance(value, np.ndarray):
        frozen = np.array(value, copy=True)
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(freeze_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _target_bearing_key(key: str) -> bool:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key.strip())
    normalized = snake_case.lower()
    tokens = tuple(part for part in re.split(r"[^a-z0-9]+", normalized) if part)
    return normalized in _FORBIDDEN_QUERY_KEYS or any(
        token
        in {
            "answer",
            "answers",
            "label",
            "labels",
            "solution",
            "solutions",
            "target",
            "targets",
        }
        for token in tokens
    )


def _assert_no_hidden_target(
    value: Any, *, path: str, allow_demonstration_output: bool = False
) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            raw_name = str(raw_key).strip()
            key = raw_name.lower()
            allowed_output = allow_demonstration_output and (
                (path == "context" and key == "support_outputs")
                or (
                    re.fullmatch(r"context\.demonstrations\[\d+\]", path) is not None
                    and key == "output"
                )
            )
            if (
                _target_bearing_key(raw_name) or key in {"output", "outputs"}
            ) and not allowed_output:
                raise StructuredProtocolError(
                    f"{path} contains target-bearing key {raw_key!r}"
                )
            _assert_no_hidden_target(
                item,
                path=f"{path}.{raw_key}",
                allow_demonstration_output=allow_demonstration_output,
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_no_hidden_target(
                item,
                path=f"{path}[{index}]",
                allow_demonstration_output=allow_demonstration_output,
            )


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
    if isinstance(value, frozenset):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def public_projection_sha256(value: Any) -> str:
    """Hash an explicitly target-stripped public projection."""

    canonical = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicTask:
    """Target-free task object safe to hand to models and agents."""

    task_id: str
    family: str
    split: str
    source_group: str
    augmentation_group: str
    context: Mapping[str, Any]
    query: Mapping[str, Any]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        task_id = _nonempty_text(self.task_id, name="task_id")
        family = _nonempty_text(self.family, name="family").lower()
        split = _nonempty_text(self.split, name="split").lower()
        if split not in VALID_SPLITS:
            raise StructuredProtocolError(
                f"split must be one of {sorted(VALID_SPLITS)!r}"
            )
        source_group = _nonempty_text(self.source_group, name="source_group")
        augmentation_group = _nonempty_text(
            self.augmentation_group, name="augmentation_group"
        )
        if not isinstance(self.context, Mapping):
            raise StructuredProtocolError("context must be a mapping")
        if not isinstance(self.query, Mapping):
            raise StructuredProtocolError("query must be a mapping")
        metadata = {} if self.metadata is None else self.metadata
        if not isinstance(metadata, Mapping):
            raise StructuredProtocolError("metadata must be a mapping")
        _assert_no_hidden_target(
            self.context, path="context", allow_demonstration_output=True
        )
        _assert_no_hidden_target(self.query, path="query")
        _assert_no_hidden_target(metadata, path="metadata")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "source_group", source_group)
        object.__setattr__(self, "augmentation_group", augmentation_group)
        object.__setattr__(self, "context", freeze_value(self.context))
        object.__setattr__(self, "query", freeze_value(self.query))
        object.__setattr__(self, "metadata", freeze_value(metadata))

    @property
    def fingerprint(self) -> str:
        """Hash the complete public view (and therefore no hidden target)."""

        payload = {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "source_group": self.source_group,
            "augmentation_group": self.augmentation_group,
            "context": _jsonable(self.context),
            "query": _jsonable(self.query),
            "metadata": _jsonable(self.metadata),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def content_group(self) -> str:
        """Content-address the model-visible task, excluding IDs and metadata.

        Source identifiers are supplied by datasets and can therefore miss
        duplicate examples.  This independent digest makes an exact public
        task appearing under different IDs fail closed when it crosses a
        train/validation/test boundary.
        """

        return public_projection_sha256(
            {
                "family": self.family,
                "context": self.context,
                "query": self.query,
            }
        )

    @property
    def group_id(self) -> str:
        """Compatibility name for the source-level independence group."""

        return self.source_group

    @property
    def support_inputs(self) -> tuple[Any, ...]:
        explicit = self.context.get("support_inputs")
        if explicit is not None:
            return tuple(explicit)
        demonstrations = self.context.get("demonstrations", ())
        return tuple(item["input"] for item in demonstrations)

    @property
    def support_outputs(self) -> tuple[Any, ...]:
        explicit = self.context.get("support_outputs")
        if explicit is not None:
            return tuple(explicit)
        demonstrations = self.context.get("demonstrations", ())
        return tuple(item["output"] for item in demonstrations)

    @property
    def query_inputs(self) -> tuple[Any, ...]:
        values = self.query.get("inputs")
        if values is None:
            return (self.query,)
        return tuple(values)


@dataclass(frozen=True, slots=True)
class TrainingTaskView:
    """A supervised view that can exist only for train/validation tasks."""

    public_task: PublicTask
    target: Any

    def __post_init__(self) -> None:
        if self.public_task.split == "test":
            raise CapabilityError("test tasks cannot produce TrainingTaskView")
        object.__setattr__(self, "target", freeze_value(self.target))

    @property
    def query_targets(self) -> Any:
        """Explicit name used by supervised candidate generators."""

        return self.target


ScoreFunction = Callable[[PublicTask, Any, Any], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class _TargetRecord:
    public_task: PublicTask
    target: Any
    scorer: ScoreFunction


class TargetStore:
    """Opaque targets bound to exact public-task object identities.

    There is intentionally no ``get_target`` method.  Test labels can be used
    only by :meth:`score`; supervised views are restricted to train and
    validation.  Identity binding prevents a caller from copying a test task
    and changing its ``split`` field to obtain its target.
    """

    __slots__ = ("__records",)

    def __init__(self, records: Sequence[_TargetRecord]) -> None:
        normalized: dict[str, _TargetRecord] = {}
        for record in records:
            task_id = record.public_task.task_id
            if task_id in normalized:
                raise StructuredProtocolError(f"duplicate target for {task_id!r}")
            if not callable(record.scorer):
                raise StructuredProtocolError(f"scorer for {task_id!r} is not callable")
            normalized[task_id] = _TargetRecord(
                public_task=record.public_task,
                target=freeze_value(record.target),
                scorer=record.scorer,
            )
        if not normalized:
            raise StructuredProtocolError("TargetStore must not be empty")
        self.__records = MappingProxyType(normalized)

    def __len__(self) -> int:
        return len(self.__records)

    def __repr__(self) -> str:
        return f"TargetStore(n_tasks={len(self)})"

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self.__records)

    def _record_for(self, task: PublicTask) -> _TargetRecord:
        if not isinstance(task, PublicTask):
            raise TypeError("task must be a PublicTask")
        record = self.__records.get(task.task_id)
        if record is None:
            raise CapabilityError(f"no capability registered for {task.task_id!r}")
        if record.public_task is not task:
            raise CapabilityError(
                "task capability is identity-bound; copied or modified tasks are rejected"
            )
        return record

    def training_view(self, task: PublicTask) -> TrainingTaskView:
        """Return labels only for an identity-bound train/validation task."""

        record = self._record_for(task)
        if task.split == "test":
            raise CapabilityError("test targets are unavailable for training")
        return TrainingTaskView(public_task=task, target=record.target)

    get_training_view = training_view

    def score(self, task: PublicTask, prediction: Any) -> Mapping[str, Any]:
        """Score without exposing the stored target to the caller."""

        record = self._record_for(task)
        result = record.scorer(task, prediction, record.target)
        if not isinstance(result, Mapping):
            raise StructuredProtocolError("registered scorer must return a mapping")
        return freeze_value(result)

    evaluate = score


def validate_group_disjointness(tasks: Sequence[PublicTask]) -> None:
    """Reject declared or independently content-addressed split leakage."""

    memberships: dict[tuple[str, str], set[str]] = {}
    for task in tasks:
        for group_kind, group_id in (
            ("source_group", task.source_group),
            ("augmentation_group", task.augmentation_group),
            ("public_content", task.content_group),
        ):
            memberships.setdefault((group_kind, group_id), set()).add(task.split)
    leaking = {
        f"{kind}:{group_id}": sorted(splits)
        for (kind, group_id), splits in memberships.items()
        if len(splits) > 1
    }
    if leaking:
        raise StructuredProtocolError(
            f"source/augmentation/public-content group crosses splits: {leaking!r}"
        )


@dataclass(frozen=True, slots=True)
class StructuredDataset:
    """Immutable public tasks paired with a non-serializing target store."""

    tasks: tuple[PublicTask, ...]
    target_store: TargetStore

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        if not tasks:
            raise StructuredProtocolError("structured dataset must contain tasks")
        identifiers = [task.task_id for task in tasks]
        if len(identifiers) != len(set(identifiers)):
            raise StructuredProtocolError("task_id values must be globally unique")
        if set(identifiers) != set(self.target_store.task_ids):
            raise StructuredProtocolError("public tasks and TargetStore IDs differ")
        validate_group_disjointness(tasks)
        object.__setattr__(self, "tasks", tasks)

    def for_split(self, split: str) -> tuple[PublicTask, ...]:
        normalized = _nonempty_text(split, name="split").lower()
        if normalized not in VALID_SPLITS:
            raise StructuredProtocolError(f"unknown split {split!r}")
        return tuple(task for task in self.tasks if task.split == normalized)

    @property
    def families(self) -> frozenset[str]:
        return frozenset(task.family for task in self.tasks)

    @property
    def train_task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks if task.split == "train")

    @property
    def validation_task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks if task.split == "validation")

    @property
    def test_task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks if task.split == "test")

    @property
    def split_by_groups(self) -> Mapping[str, str]:
        """Source-group split assignment, validated to be one-to-one."""

        return MappingProxyType({task.source_group: task.split for task in self.tasks})


def build_structured_dataset(
    tasks: Sequence[PublicTask],
    targets: Sequence[Any],
    *,
    scorer: ScoreFunction | Mapping[str, ScoreFunction],
) -> StructuredDataset:
    """Bind public tasks to opaque targets and registered scorers."""

    public_tasks = tuple(tasks)
    hidden_targets = tuple(targets)
    if len(public_tasks) != len(hidden_targets):
        raise StructuredProtocolError("tasks and targets must have equal lengths")
    records: list[_TargetRecord] = []
    for task, target in zip(public_tasks, hidden_targets, strict=True):
        if isinstance(scorer, Mapping):
            try:
                family_scorer = scorer[task.family]
            except KeyError as error:
                raise StructuredProtocolError(
                    f"no scorer registered for family {task.family!r}"
                ) from error
        else:
            family_scorer = scorer
        records.append(_TargetRecord(task, target, family_scorer))
    return StructuredDataset(public_tasks, TargetStore(records))


def grouped_split_indices(
    source_groups: Sequence[str],
    augmentation_groups: Sequence[str],
    *,
    seed: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[str, ...]:
    """Assign connected source/augmentation components deterministically.

    Any records sharing either group are joined before shuffling, so neither
    family can cross the resulting train/validation/test partition.
    """

    source = tuple(_nonempty_text(item, name="source_group") for item in source_groups)
    augmentation = tuple(
        _nonempty_text(item, name="augmentation_group") for item in augmentation_groups
    )
    if len(source) != len(augmentation) or not source:
        raise StructuredProtocolError(
            "source_groups and augmentation_groups must have equal non-zero lengths"
        )
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if not 0.0 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must be in [0, 1]")
    if not 0.0 <= validation_fraction <= 1.0:
        raise ValueError("validation_fraction must be in [0, 1]")
    if train_fraction + validation_fraction > 1.0:
        raise ValueError("train and validation fractions must sum to at most one")

    parent = list(range(len(source)))

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
    for index, (source_id, augmentation_id) in enumerate(
        zip(source, augmentation, strict=True)
    ):
        for key in (("source", source_id), ("augmentation", augmentation_id)):
            previous = seen.setdefault(key, index)
            union(index, previous)

    components: dict[int, list[int]] = {}
    for index in range(len(source)):
        components.setdefault(find(index), []).append(index)
    ordered = sorted(components.values(), key=lambda values: tuple(values))
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(ordered))
    shuffled = [ordered[index] for index in permutation]
    n_components = len(shuffled)
    n_train = int(np.floor(train_fraction * n_components))
    n_validation = int(np.floor(validation_fraction * n_components))
    assignments = ["test"] * len(source)
    for rank, component in enumerate(shuffled):
        split = (
            "train"
            if rank < n_train
            else "validation"
            if rank < n_train + n_validation
            else "test"
        )
        for index in component:
            assignments[index] = split
    return tuple(assignments)


split_by_groups = grouped_split_indices
