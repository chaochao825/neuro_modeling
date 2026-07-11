"""Capability-safe contracts for structured candidate reasoners.

The public inference object intentionally contains candidate inputs and outputs,
but never correctness labels or query targets.  Labels live in the separate
``TrainingCandidateSet`` capability and are accepted only by ``fit``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


class StructuredReasonerError(ValueError):
    """Raised when a structured-reasoning contract is violated."""


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredReasonerError(f"{name} must be a non-empty string")
    return value.strip()


def _readonly_array(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
    dtype: np.dtype[Any] | type[Any] | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"O", "S", "U", "V"}:
        raise StructuredReasonerError(f"{name} must be numeric")
    array = np.asarray(raw, dtype=dtype).copy()
    if ndim is not None and array.ndim != ndim:
        raise StructuredReasonerError(f"{name} must have {ndim} dimensions")
    if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
        raise StructuredReasonerError(f"{name} must contain finite values")
    array.setflags(write=False)
    return array


def _freeze_payload(value: object) -> object:
    """Defensively freeze nested candidate outputs without interpreting them."""

    if isinstance(value, np.ndarray):
        result = value.copy()
        result.setflags(write=False)
        return result
    if isinstance(value, Mapping):
        return MappingProxyType(
            {copy.deepcopy(key): _freeze_payload(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_payload(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_payload(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class CandidateSet:
    """Model-visible candidates for one task, with no target-bearing fields."""

    task_id: str
    family: str
    candidate_ids: tuple[str, ...]
    features: np.ndarray
    candidate_outputs: tuple[object, ...]
    candidate_provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        task_id = _nonempty_text(self.task_id, name="task_id")
        family = _nonempty_text(self.family, name="family").lower()
        candidate_ids = tuple(
            _nonempty_text(value, name="candidate_id") for value in self.candidate_ids
        )
        if not candidate_ids:
            raise StructuredReasonerError("candidate_ids must not be empty")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise StructuredReasonerError("candidate_ids must be unique within a task")
        features = _readonly_array(
            self.features,
            name="features",
            ndim=2,
            dtype=float,
        )
        if features.shape[0] != len(candidate_ids) or features.shape[1] == 0:
            raise StructuredReasonerError(
                "features must have shape (number of candidates, positive feature dim)"
            )
        outputs = tuple(_freeze_payload(value) for value in self.candidate_outputs)
        provenance = tuple(
            _nonempty_text(value, name="candidate_provenance")
            for value in self.candidate_provenance
        )
        if len(outputs) != len(candidate_ids):
            raise StructuredReasonerError(
                "candidate_outputs must align one-to-one with candidate_ids"
            )
        if len(provenance) != len(candidate_ids):
            raise StructuredReasonerError(
                "candidate_provenance must align one-to-one with candidate_ids"
            )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "candidate_outputs", outputs)
        object.__setattr__(self, "candidate_provenance", provenance)

    @property
    def n_candidates(self) -> int:
        return len(self.candidate_ids)

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def outputs(self) -> tuple[object, ...]:
        """Concise read-only alias used by candidate generators."""

        return self.candidate_outputs

    @property
    def provenance(self) -> tuple[str, ...]:
        """Concise read-only alias used by experiment serializers."""

        return self.candidate_provenance


@dataclass(frozen=True)
class TrainingCandidateSet:
    """Training-only label capability wrapped around a public candidate set."""

    public: CandidateSet
    labels: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.public, CandidateSet):
            raise StructuredReasonerError("public must be a CandidateSet")
        labels = _readonly_array(self.labels, name="labels", ndim=1, dtype=float)
        if labels.shape != (self.public.n_candidates,):
            raise StructuredReasonerError("labels must align with candidate_ids")
        if np.any((labels < 0.0) | (labels > 1.0)):
            raise StructuredReasonerError("labels must lie in [0, 1]")
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True)
class ComputeBudget:
    """Hard inference budget shared across candidate selectors."""

    max_candidate_evaluations: int
    max_internal_steps: int

    def __post_init__(self) -> None:
        for name in ("max_candidate_evaluations", "max_internal_steps"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise StructuredReasonerError(f"{name} must be an integer")
            if value < 1:
                raise StructuredReasonerError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class ComputeReceipt:
    """Auditable charged compute for one solve call."""

    budget: ComputeBudget
    candidate_evaluations: int
    internal_steps: int
    fast_updates: int
    slow_updates: int
    trace_updates: int
    charged_units: int
    exhausted: bool

    def __post_init__(self) -> None:
        counts = (
            self.candidate_evaluations,
            self.internal_steps,
            self.fast_updates,
            self.slow_updates,
            self.trace_updates,
            self.charged_units,
        )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
            for value in counts
        ):
            raise StructuredReasonerError("compute receipt counts must be non-negative")
        if self.candidate_evaluations > self.budget.max_candidate_evaluations:
            raise StructuredReasonerError("candidate evaluations exceed the budget")
        if self.internal_steps > self.budget.max_internal_steps:
            raise StructuredReasonerError("internal steps exceed the budget")

    @property
    def within_budget(self) -> bool:
        return (
            self.candidate_evaluations <= self.budget.max_candidate_evaluations
            and self.internal_steps <= self.budget.max_internal_steps
        )


@dataclass(frozen=True)
class FitReceipt:
    """Training provenance, including an explicit BPTT declaration."""

    task_ids: tuple[str, ...]
    used_bptt: bool
    task_balanced: bool
    optimization: str
    epochs: int
    final_loss: float
    seed: int

    def __post_init__(self) -> None:
        task_ids = tuple(_nonempty_text(value, name="task_id") for value in self.task_ids)
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise StructuredReasonerError("fit task_ids must be non-empty and unique")
        if self.epochs < 1:
            raise StructuredReasonerError("epochs must be positive")
        if not np.isfinite(self.final_loss) or self.final_loss < 0.0:
            raise StructuredReasonerError("final_loss must be finite and non-negative")
        object.__setattr__(self, "task_ids", task_ids)


@dataclass(frozen=True)
class SolverOutput:
    """A selected public candidate plus dynamics and compute audit traces."""

    task_id: str
    selected_index: int
    selected_candidate_id: str
    selected_output: object
    scores: np.ndarray
    trace: np.ndarray
    bilinear_trace: np.ndarray
    receipt: ComputeReceipt

    def __post_init__(self) -> None:
        task_id = _nonempty_text(self.task_id, name="task_id")
        candidate_id = _nonempty_text(
            self.selected_candidate_id, name="selected_candidate_id"
        )
        scores = _readonly_array(self.scores, name="scores", ndim=1, dtype=float)
        trace = _readonly_array(self.trace, name="trace", ndim=2, dtype=float)
        bilinear = _readonly_array(
            self.bilinear_trace,
            name="bilinear_trace",
            ndim=2,
            dtype=float,
        )
        if not 0 <= self.selected_index < scores.size:
            raise StructuredReasonerError("selected_index is outside scores")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "selected_candidate_id", candidate_id)
        object.__setattr__(self, "selected_output", _freeze_payload(self.selected_output))
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "trace", trace)
        object.__setattr__(self, "bilinear_trace", bilinear)


@runtime_checkable
class StructuredReasoner(Protocol):
    """Common API for local main models and explicitly BPTT baselines."""

    used_bptt: bool

    def fit(self, tasks: Sequence[TrainingCandidateSet]) -> FitReceipt:
        """Fit using training-capability objects only."""

    def solve(
        self,
        candidates: CandidateSet,
        budget: ComputeBudget | None = None,
    ) -> SolverOutput:
        """Select from a public candidate set without target access."""
