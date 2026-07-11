"""Single-norm cumulative update-budget control for paired experiments.

The controller operates on the final local update direction, after sparsity and
Dale projection but before optional normalization corrections.  It only scales
an update toward zero, so a Dale-feasible update remains feasible along the
line segment from the current weights to the proposed weights.

Each controller matches exactly one path-length norm.  The other norm is
recorded as a diagnostic; a scalar rescaling cannot, in general, match both
L1 and L2 budgets simultaneously.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


Array = np.ndarray
BudgetNorm = Literal["l1", "l2"]


def _nonnegative_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _matrix_norms(update: Array) -> tuple[float, float]:
    l1 = float(np.sum(np.abs(update)))
    l2 = float(np.linalg.norm(update))
    if not np.isfinite(l1) or not np.isfinite(l2):
        raise ValueError("proposed_matrix norms must be finite")
    return l1, l2


@dataclass(frozen=True)
class UpdateBudgetSummary:
    """Immutable audit record for one selected-norm budget panel.

    Cumulative norms are path lengths: they sum each event's matrix norm rather
    than taking the norm of a potentially cancelling cumulative update matrix.
    ``final_shortfall`` is only defined after all planned event slots have been
    consumed.
    """

    selected_norm: BudgetNorm
    secondary_norm: BudgetNorm
    total_budget: float
    tolerance: float
    planned_events: int
    processed_events: int
    raw_nonzero_events: int
    applied_nonzero_events: int
    zero_proposal_events: int
    cumulative_raw_l1: float
    cumulative_raw_l2: float
    cumulative_applied_l1: float
    cumulative_applied_l2: float
    selected_applied: float
    remaining: float
    complete: bool
    attained: bool
    final_shortfall: float | None
    simultaneous_dual_norm_match: bool = False
    secondary_norm_is_diagnostic_only: bool = True

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable copy suitable for experiment metrics."""

        return asdict(self)


class UpdateBudgetController:
    """Allocate a cumulative L1 or L2 update budget across online events.

    Parameters
    ----------
    total_budget:
        Maximum cumulative path length in the selected norm.
    norm:
        The sole matched norm for this controller, either ``"l1"`` or ``"l2"``.
    planned_events:
        Number of update-event slots.  A zero proposal still consumes its slot
        and is explicitly recorded.
    tolerance:
        Absolute tolerance used only to classify final budget attainment.

    Notes
    -----
    At each event the controller allocates an equal share of the remaining
    budget over the remaining slots.  Unspent budget from a zero or insufficient
    proposal is carried forward.  Updates are never amplified: a formal target
    should therefore be calibrated no larger than the available proposal path.
    Any irrecoverable deficit is retained as ``final_shortfall``.

    Call :meth:`scale` with a sparse/Dale-applied local update, before fan-in or
    other normalization corrections.  Because the returned factor lies in
    ``[0, 1]``, the result stays on the feasible segment to the projected
    candidate.  The input array is never mutated.
    """

    def __init__(
        self,
        total_budget: float,
        norm: BudgetNorm,
        planned_events: int,
        tolerance: float = 1e-9,
    ) -> None:
        self.total_budget = _nonnegative_scalar(total_budget, name="total_budget")
        if not isinstance(norm, str):
            raise TypeError("norm must be a string")
        if norm not in {"l1", "l2"}:
            raise ValueError("norm must be 'l1' or 'l2'")
        self.norm: BudgetNorm = norm
        self.planned_events = _positive_int(planned_events, name="planned_events")
        self.tolerance = _nonnegative_scalar(tolerance, name="tolerance")

        self._shape: tuple[int, int] | None = None
        self._processed_events = 0
        self._raw_nonzero_events = 0
        self._applied_nonzero_events = 0
        self._zero_proposal_events = 0
        self._cumulative_raw_l1 = 0.0
        self._cumulative_raw_l2 = 0.0
        self._cumulative_applied_l1 = 0.0
        self._cumulative_applied_l2 = 0.0

    @property
    def processed_events(self) -> int:
        return self._processed_events

    @property
    def complete(self) -> bool:
        return self._processed_events == self.planned_events

    @property
    def selected_applied(self) -> float:
        if self.norm == "l1":
            return self._cumulative_applied_l1
        return self._cumulative_applied_l2

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self.selected_applied)

    @property
    def attained(self) -> bool:
        return self.complete and self.remaining <= self.tolerance

    def scale(self, proposed_matrix: Array) -> Array:
        """Return the budgeted update for the next planned event slot.

        The selected cumulative norm never exceeds ``total_budget`` apart from
        floating-point roundoff bounded by a final defensive rescaling.
        """

        if self.complete:
            raise RuntimeError("all planned update events have already been processed")
        raw_proposed = np.asarray(proposed_matrix)
        if np.iscomplexobj(raw_proposed):
            raise ValueError("proposed_matrix must be real-valued")
        proposed = np.asarray(raw_proposed, dtype=float)
        if proposed.ndim != 2 or proposed.shape[0] < 1 or proposed.shape[1] < 1:
            raise ValueError(
                "proposed_matrix must be a non-empty two-dimensional matrix"
            )
        if not np.all(np.isfinite(proposed)):
            raise ValueError("proposed_matrix must contain only finite values")
        if self._shape is None:
            self._shape = proposed.shape
        elif proposed.shape != self._shape:
            raise ValueError(
                f"proposed_matrix shape changed from {self._shape} to {proposed.shape}"
            )

        raw_l1, raw_l2 = _matrix_norms(proposed)
        selected_raw = raw_l1 if self.norm == "l1" else raw_l2
        self._cumulative_raw_l1 += raw_l1
        self._cumulative_raw_l2 += raw_l2
        if selected_raw == 0.0:
            self._zero_proposal_events += 1
        else:
            self._raw_nonzero_events += 1

        remaining_slots = self.planned_events - self._processed_events
        allocation = self.remaining / remaining_slots
        spend = min(selected_raw, allocation)
        if selected_raw == 0.0 or spend == 0.0:
            scaled = np.zeros_like(proposed)
        else:
            scaled = proposed * (spend / selected_raw)

        applied_l1, applied_l2 = _matrix_norms(scaled)
        selected_event = applied_l1 if self.norm == "l1" else applied_l2
        # Homogeneous matrix norms should already equal ``spend``.  This guard
        # prevents a one-ulp overshoot from accumulating past the hard budget.
        available = self.remaining
        if selected_event > available and selected_event > 0.0:
            scaled *= available / selected_event
            applied_l1, applied_l2 = _matrix_norms(scaled)
            selected_event = applied_l1 if self.norm == "l1" else applied_l2

        self._cumulative_applied_l1 += applied_l1
        self._cumulative_applied_l2 += applied_l2
        if selected_event > 0.0:
            self._applied_nonzero_events += 1
        self._processed_events += 1
        return scaled

    def summary(self) -> UpdateBudgetSummary:
        """Return the current immutable audit summary."""

        remaining = self.remaining
        complete = self.complete
        return UpdateBudgetSummary(
            selected_norm=self.norm,
            secondary_norm="l2" if self.norm == "l1" else "l1",
            total_budget=self.total_budget,
            tolerance=self.tolerance,
            planned_events=self.planned_events,
            processed_events=self._processed_events,
            raw_nonzero_events=self._raw_nonzero_events,
            applied_nonzero_events=self._applied_nonzero_events,
            zero_proposal_events=self._zero_proposal_events,
            cumulative_raw_l1=self._cumulative_raw_l1,
            cumulative_raw_l2=self._cumulative_raw_l2,
            cumulative_applied_l1=self._cumulative_applied_l1,
            cumulative_applied_l2=self._cumulative_applied_l2,
            selected_applied=self.selected_applied,
            remaining=remaining,
            complete=complete,
            attained=complete and remaining <= self.tolerance,
            final_shortfall=remaining if complete else None,
        )
