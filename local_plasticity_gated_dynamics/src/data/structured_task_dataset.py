"""Frozen candidate/action tapes for secondary structured-task validation.

The adapter deliberately does not generate ARC candidates.  It consumes an
immutable tape whose candidate features and baseline scores were computed
before this experiment.  The only supervised target exposed by this module is
``exact_correct``; callers must restrict fitting to tasks assigned to the
router-training split.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROUTER_SPLITS = frozenset({"train", "test"})


class StructuredTaskDataError(RuntimeError):
    """Raised when a frozen structured-task tape violates its schema."""


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredTaskDataError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise StructuredTaskDataError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise StructuredTaskDataError(f"{name} must be a finite real number") from error
    if not np.isfinite(result):
        raise StructuredTaskDataError(f"{name} must be a finite real number")
    return result


def _partition_ids(
    value: object, *, task_id: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != {"train", "test"}:
        raise StructuredTaskDataError(
            f"task {task_id!r} partitions must contain exactly train and test"
        )
    normalized: dict[str, tuple[str, ...]] = {}
    for partition in ("train", "test"):
        identifiers = value[partition]
        if isinstance(identifiers, (str, bytes)) or not isinstance(
            identifiers, Sequence
        ):
            raise StructuredTaskDataError(
                f"task {task_id!r} {partition} partition must be an ID sequence"
            )
        values = tuple(
            _nonempty_string(identifier, name=f"{partition} example ID")
            for identifier in identifiers
        )
        if not values or len(set(values)) != len(values):
            raise StructuredTaskDataError(
                f"task {task_id!r} {partition} partition must be non-empty and unique"
            )
        normalized[partition] = values
    overlap = set(normalized["train"]) & set(normalized["test"])
    if overlap:
        raise StructuredTaskDataError(
            f"task {task_id!r} has train/test example leakage: {sorted(overlap)!r}"
        )
    return normalized["train"], normalized["test"]


def _aggregate_train_value(
    payload: Mapping[str, Any], *, singular: str, plural: str, candidate_id: str
) -> float | None:
    if singular in payload:
        return _finite_float(
            payload[singular], name=f"candidate {candidate_id!r} {singular}"
        )
    if plural not in payload:
        return None
    values = payload[plural]
    if isinstance(values, Mapping):
        if "train" not in values:
            raise StructuredTaskDataError(
                f"candidate {candidate_id!r} {plural} mapping must contain train"
            )
        values = values["train"]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise StructuredTaskDataError(
            f"candidate {candidate_id!r} {plural} must be a non-empty numeric sequence"
        )
    numeric = np.asarray(
        [
            _finite_float(value, name=f"candidate {candidate_id!r} {plural}")
            for value in values
        ],
        dtype=float,
    )
    if numeric.size == 0:
        raise StructuredTaskDataError(
            f"candidate {candidate_id!r} {plural} must not be empty"
        )
    # Only train-partition scores are collapsed into the frozen baseline score.
    return float(np.mean(numeric))


@dataclass(frozen=True)
class StructuredCandidate:
    candidate_id: str
    features: np.ndarray
    baseline_score: float
    exact_correct: bool
    compute_cost: float
    feature_provenance_hash: str
    feature_source_example_ids: tuple[str, ...]
    score_source_example_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        candidate_id = _nonempty_string(self.candidate_id, name="candidate_id")
        raw = np.asarray(self.features)
        if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
            raise StructuredTaskDataError(
                "candidate features must be real numeric values"
            )
        features = np.asarray(raw, dtype=float)
        if features.ndim != 1 or features.size == 0 or not np.isfinite(features).all():
            raise StructuredTaskDataError(
                "candidate features must be a non-empty finite one-dimensional array"
            )
        score = _finite_float(self.baseline_score, name="baseline_score")
        if not isinstance(self.exact_correct, (bool, np.bool_)):
            raise StructuredTaskDataError("exact_correct must be boolean")
        cost = _finite_float(self.compute_cost, name="compute_cost")
        if cost <= 0.0:
            raise StructuredTaskDataError("compute_cost must be positive")
        provenance = _nonempty_string(
            self.feature_provenance_hash, name="feature_provenance_hash"
        ).lower()
        if not _SHA256.fullmatch(provenance):
            raise StructuredTaskDataError(
                "feature_provenance_hash must be a SHA-256 hex digest"
            )
        source_groups: dict[str, tuple[str, ...]] = {}
        for name in ("feature_source_example_ids", "score_source_example_ids"):
            raw_sources = getattr(self, name)
            if isinstance(raw_sources, (str, bytes)) or not isinstance(
                raw_sources, Sequence
            ):
                raise StructuredTaskDataError(f"{name} must be an ID sequence")
            sources = tuple(_nonempty_string(value, name=name) for value in raw_sources)
            if not sources or len(set(sources)) != len(sources):
                raise StructuredTaskDataError(f"{name} must be non-empty and unique")
            source_groups[name] = sources
        features = features.copy()
        features.setflags(write=False)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "baseline_score", score)
        object.__setattr__(self, "exact_correct", bool(self.exact_correct))
        object.__setattr__(self, "compute_cost", cost)
        object.__setattr__(self, "feature_provenance_hash", provenance)
        for name, sources in source_groups.items():
            object.__setattr__(self, name, sources)


@dataclass(frozen=True)
class StructuredTask:
    task_id: str
    router_split: str
    train_example_ids: tuple[str, ...]
    test_example_ids: tuple[str, ...]
    candidates: tuple[StructuredCandidate, ...]
    provenance_hash: str

    def __post_init__(self) -> None:
        train_ids = set(self.train_example_ids)
        test_ids = set(self.test_example_ids)
        if not train_ids or not test_ids or train_ids & test_ids:
            raise StructuredTaskDataError(
                f"task {self.task_id!r} must have disjoint non-empty example partitions"
            )
        for candidate in self.candidates:
            for name in (
                "feature_source_example_ids",
                "score_source_example_ids",
            ):
                sources = set(getattr(candidate, name))
                if not sources <= train_ids:
                    raise StructuredTaskDataError(
                        f"task {self.task_id!r} candidate {candidate.candidate_id!r} "
                        f"{name} must be a subset of train example IDs"
                    )

    @property
    def candidate_covered(self) -> bool:
        return any(candidate.exact_correct for candidate in self.candidates)

    @property
    def candidate_fingerprint(self) -> str:
        # Correctness is intentionally omitted: this fingerprint describes the
        # candidate/action set available to every selector, not its hidden labels.
        payload = [
            {
                "candidate_id": candidate.candidate_id,
                "baseline_score": candidate.baseline_score,
                "features": candidate.features.tolist(),
                "compute_cost": candidate.compute_cost,
                "feature_provenance_hash": candidate.feature_provenance_hash,
                "feature_source_example_ids": candidate.feature_source_example_ids,
                "score_source_example_ids": candidate.score_source_example_ids,
            }
            for candidate in self.candidates
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class StructuredTaskDataset:
    tasks: tuple[StructuredTask, ...]
    schema_version: str
    tape_fingerprint: str
    frozen_before_evaluation: bool
    candidate_generator_commit: str
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.tasks:
            raise StructuredTaskDataError("structured-task tape must contain tasks")
        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise StructuredTaskDataError("task_id values must be globally unique")
        provenance_hashes = [task.provenance_hash for task in self.tasks]
        if len(set(provenance_hashes)) != len(provenance_hashes):
            raise StructuredTaskDataError(
                "task provenance_hash values must be globally unique across splits"
            )
        all_example_ids = [
            example_id
            for task in self.tasks
            for example_id in (*task.train_example_ids, *task.test_example_ids)
        ]
        if len(set(all_example_ids)) != len(all_example_ids):
            raise StructuredTaskDataError(
                "example IDs must be globally unique across task/router splits"
            )
        if {task.router_split for task in self.tasks} != _ROUTER_SPLITS:
            raise StructuredTaskDataError(
                "tape must contain at least one complete train task and one complete test task"
            )
        feature_dims = {
            candidate.features.size
            for task in self.tasks
            for candidate in task.candidates
        }
        if not feature_dims:
            raise StructuredTaskDataError("tape must contain at least one candidate")
        if len(feature_dims) != 1:
            raise StructuredTaskDataError("all candidate feature dimensions must match")
        if not _SHA256.fullmatch(self.tape_fingerprint):
            raise StructuredTaskDataError(
                "tape_fingerprint must be a SHA-256 hex digest"
            )
        if not isinstance(self.frozen_before_evaluation, (bool, np.bool_)):
            raise StructuredTaskDataError("frozen_before_evaluation must be boolean")
        if not _SHA256.fullmatch(str(self.candidate_generator_commit).lower()):
            raise StructuredTaskDataError(
                "candidate_generator_commit must be a SHA-256 hex digest"
            )

    @property
    def train_tasks(self) -> tuple[StructuredTask, ...]:
        return tuple(task for task in self.tasks if task.router_split == "train")

    @property
    def test_tasks(self) -> tuple[StructuredTask, ...]:
        return tuple(task for task in self.tasks if task.router_split == "test")

    @property
    def feature_dim(self) -> int:
        for task in self.tasks:
            if task.candidates:
                return int(task.candidates[0].features.size)
        raise AssertionError("validated dataset unexpectedly has no candidates")


def _candidate_from_payload(value: object, *, task_id: str) -> StructuredCandidate:
    if not isinstance(value, Mapping):
        raise StructuredTaskDataError(f"task {task_id!r} candidate must be an object")
    candidate_id = _nonempty_string(value.get("candidate_id"), name="candidate_id")
    if "features" not in value:
        raise StructuredTaskDataError(f"candidate {candidate_id!r} is missing features")
    score = _aggregate_train_value(
        value, singular="score", plural="scores", candidate_id=candidate_id
    )
    if score is None:
        score = _aggregate_train_value(
            value, singular="logit", plural="logits", candidate_id=candidate_id
        )
    if score is None:
        raise StructuredTaskDataError(
            f"candidate {candidate_id!r} requires score(s) or logit(s)"
        )
    if "exact_correct" not in value:
        raise StructuredTaskDataError(
            f"candidate {candidate_id!r} is missing exact_correct"
        )
    return StructuredCandidate(
        candidate_id=candidate_id,
        features=np.asarray(value["features"]),
        baseline_score=score,
        exact_correct=value["exact_correct"],
        compute_cost=value.get("compute_cost", 1.0),
        feature_provenance_hash=value.get("feature_provenance_hash", ""),
        feature_source_example_ids=tuple(value.get("feature_source_example_ids", ())),
        score_source_example_ids=tuple(value.get("score_source_example_ids", ())),
    )


def _task_from_payload(value: object) -> StructuredTask:
    if not isinstance(value, Mapping):
        raise StructuredTaskDataError("every tape task must be an object")
    task_id = _nonempty_string(value.get("task_id"), name="task_id")
    router_split = _nonempty_string(
        value.get("router_split", value.get("split")), name="router_split"
    ).lower()
    if router_split not in _ROUTER_SPLITS:
        raise StructuredTaskDataError(
            f"task {task_id!r} router_split must be train or test"
        )
    train_ids, test_ids = _partition_ids(value.get("partitions"), task_id=task_id)
    provenance_hash = _nonempty_string(
        value.get("provenance_hash"), name="provenance_hash"
    ).lower()
    if not _SHA256.fullmatch(provenance_hash):
        raise StructuredTaskDataError(
            f"task {task_id!r} provenance_hash must be a SHA-256 hex digest"
        )
    raw_candidates = value.get("candidates")
    if isinstance(raw_candidates, (str, bytes)) or not isinstance(
        raw_candidates, Sequence
    ):
        raise StructuredTaskDataError(f"task {task_id!r} candidates must be a list")
    candidates = tuple(
        _candidate_from_payload(candidate, task_id=task_id)
        for candidate in raw_candidates
    )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise StructuredTaskDataError(
            f"task {task_id!r} candidate_id values must be unique"
        )
    if sum(candidate.exact_correct for candidate in candidates) > 1:
        raise StructuredTaskDataError(
            f"task {task_id!r} has multiple exact-correct candidates"
        )
    return StructuredTask(
        task_id=task_id,
        router_split=router_split,
        train_example_ids=train_ids,
        test_example_ids=test_ids,
        candidates=candidates,
        provenance_hash=provenance_hash,
    )


def _load_payload(path: Path) -> tuple[str, list[object], bool, str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        tasks: list[object] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise StructuredTaskDataError(
                    f"invalid JSONL record at line {line_number}: {error}"
                ) from error
        return "1.0", tasks, False, "0" * 64
    if suffix != ".json":
        raise StructuredTaskDataError("structured-task tape must be .json or .jsonl")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise StructuredTaskDataError(f"invalid JSON tape: {error}") from error
    if isinstance(payload, list):
        return "1.0", payload, False, "0" * 64
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tasks"), list):
        raise StructuredTaskDataError(
            "JSON tape must be a task list or contain a tasks list"
        )
    frozen = payload.get("frozen_before_evaluation", False)
    if not isinstance(frozen, (bool, np.bool_)):
        raise StructuredTaskDataError("frozen_before_evaluation must be boolean")
    generator_commit = str(payload.get("candidate_generator_commit", "")).lower()
    if not _SHA256.fullmatch(generator_commit):
        raise StructuredTaskDataError(
            "candidate_generator_commit must be a SHA-256 hex digest"
        )
    return (
        str(payload.get("schema_version", "1.0")),
        payload["tasks"],
        bool(frozen),
        generator_commit,
    )


def load_structured_task_tape(path: str | Path) -> StructuredTaskDataset:
    """Load and validate a frozen JSON/JSONL candidate tape.

    Router train/test membership is assigned once per complete task.  Within a
    task, the required ``partitions`` field records the disjoint ARC-style
    demonstration (train) and query (test) example IDs.
    """

    tape_path = Path(path)
    if not tape_path.is_file():
        raise FileNotFoundError(tape_path)
    schema_version, raw_tasks, frozen, generator_commit = _load_payload(tape_path)
    tasks = tuple(_task_from_payload(task) for task in raw_tasks)
    canonical_payload = {
        "schema_version": schema_version,
        "frozen_before_evaluation": frozen,
        "candidate_generator_commit": generator_commit,
        "tasks": raw_tasks,
    }
    canonical = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return StructuredTaskDataset(
        tasks=tasks,
        schema_version=schema_version,
        tape_fingerprint=fingerprint,
        frozen_before_evaluation=frozen,
        candidate_generator_commit=generator_commit,
        source_path=tape_path.resolve(),
    )


def make_synthetic_structured_tape(
    *,
    n_train_tasks: int,
    n_test_tasks: int,
    n_candidates: int,
    feature_dim: int,
    missing_test_tasks: int = 0,
    seed: int,
) -> dict[str, object]:
    """Create a deterministic smoke fixture, never a scientific ARC dataset."""

    integer_values = {
        "n_train_tasks": n_train_tasks,
        "n_test_tasks": n_test_tasks,
        "n_candidates": n_candidates,
        "feature_dim": feature_dim,
        "missing_test_tasks": missing_test_tasks,
        "seed": seed,
    }
    for name, value in integer_values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
    if min(n_train_tasks, n_test_tasks, n_candidates, feature_dim) < 1:
        raise ValueError("task, candidate, and feature counts must be positive")
    if not 0 <= missing_test_tasks < n_test_tasks:
        raise ValueError("missing_test_tasks must be in [0, n_test_tasks)")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = np.random.default_rng(seed)
    tasks: list[dict[str, object]] = []
    total = n_train_tasks + n_test_tasks
    for index in range(total):
        split = "train" if index < n_train_tasks else "test"
        split_index = index if split == "train" else index - n_train_tasks
        task_id = f"synthetic_{split}_{split_index:04d}"
        is_missing = split == "test" and split_index < missing_test_tasks
        train_example_ids = [f"{task_id}:train:0", f"{task_id}:train:1"]
        candidates: list[dict[str, object]] = []
        if not is_missing:
            correct_index = int(rng.integers(0, n_candidates))
            for candidate_index in range(n_candidates):
                correct = candidate_index == correct_index
                features = rng.normal(0.0, 0.65, size=feature_dim)
                features[0] += 2.0 if correct else -0.5
                # Baseline ranking is deliberately noisy so the fixture tests
                # routing code paths without serving as scientific evidence.
                score = float(rng.normal(0.6 if correct else 0.0, 1.0))
                candidates.append(
                    {
                        "candidate_id": f"candidate_{candidate_index:02d}",
                        "scores": {"train": [score, score + float(rng.normal(0, 0.1))]},
                        "logits": {"train": [score]},
                        "features": features.tolist(),
                        "exact_correct": bool(correct),
                        "compute_cost": 1.0,
                        "feature_provenance_hash": hashlib.sha256(
                            f"synthetic-feature:{seed}:{task_id}:{candidate_index}".encode(
                                "utf-8"
                            )
                        ).hexdigest(),
                        "feature_source_example_ids": train_example_ids,
                        "score_source_example_ids": train_example_ids,
                    }
                )
        provenance = hashlib.sha256(
            f"synthetic-smoke:{seed}:{task_id}".encode("utf-8")
        ).hexdigest()
        tasks.append(
            {
                "task_id": task_id,
                "router_split": split,
                "partitions": {
                    "train": train_example_ids,
                    "test": [f"{task_id}:test:0"],
                },
                "candidates": candidates,
                "provenance_hash": provenance,
            }
        )
    return {
        "schema_version": "1.0",
        "fixture_only": True,
        "frozen_before_evaluation": False,
        "candidate_generator_commit": hashlib.sha256(
            f"synthetic-generator:{seed}".encode("utf-8")
        ).hexdigest(),
        "tasks": tasks,
    }
