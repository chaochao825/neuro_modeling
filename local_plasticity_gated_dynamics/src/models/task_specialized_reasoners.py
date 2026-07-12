"""Target-free task adapters for Sudoku and ARC reasoning.

These models borrow task-design principles rather than implementations.  The
Sudoku adapter uses sparse positive candidate activity and local constraint
interactions; the ARC adapter uses slow operator-family beliefs and fast
within-family proposal selection.  Neither class is BDH or HRM, and neither
uses spiking dynamics or BPTT.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from src.data.structured_protocol import PublicTask
from src.tasks.structured_proposals import (
    FEATURE_NAMES,
    ProposalBatch,
    generate_arc_proposals,
)


class TaskSpecializedReasonerError(ValueError):
    """Raised when a task-specific model contract is violated."""


def _readonly(value: object, *, dtype: type = float) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _freeze_output(value: object) -> object:
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.setflags(write=False)
        return result
    if isinstance(value, tuple):
        return tuple(_freeze_output(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_output(item) for key, item in value.items()}
        )
    return value


@dataclass(frozen=True, slots=True)
class TaskSpecializedResult:
    """Target-free output and auditable internal computation receipt."""

    task_id: str
    family: str
    output: object
    state_trace: np.ndarray
    receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.task_id or self.family not in {"arc", "sudoku"}:
            raise TaskSpecializedReasonerError("invalid task result identity")
        trace = _readonly(self.state_trace)
        if trace.ndim != 2 or not np.isfinite(trace).all():
            raise TaskSpecializedReasonerError("state_trace must be finite and 2-D")
        receipt = dict(self.receipt)
        if receipt.get("used_bptt") is not False:
            raise TaskSpecializedReasonerError("task adapters must declare no BPTT")
        object.__setattr__(self, "state_trace", trace)
        object.__setattr__(self, "output", _freeze_output(self.output))
        object.__setattr__(self, "receipt", MappingProxyType(receipt))


_SUDOKU_UNITS = tuple(
    [tuple((row, col) for col in range(9)) for row in range(9)]
    + [tuple((row, col) for row in range(9)) for col in range(9)]
    + [
        tuple(
            (row, col)
            for row in range(box_row, box_row + 3)
            for col in range(box_col, box_col + 3)
        )
        for box_row in (0, 3, 6)
        for box_col in (0, 3, 6)
    ]
)
_SUDOKU_CELL_UNITS = {
    (row, col): tuple(unit for unit in _SUDOKU_UNITS if (row, col) in unit)
    for row in range(9)
    for col in range(9)
}
_SUDOKU_PEERS = {
    cell: frozenset(position for unit in units for position in unit if position != cell)
    for cell, units in _SUDOKU_CELL_UNITS.items()
}


def _sudoku_board(task: PublicTask) -> np.ndarray:
    if not isinstance(task, PublicTask) or task.family != "sudoku":
        raise TaskSpecializedReasonerError("Sudoku reasoner requires a Sudoku task")
    board = np.asarray(task.query.get("grid"))
    if board.shape != (9, 9) or board.dtype.kind not in {"i", "u"}:
        raise TaskSpecializedReasonerError("Sudoku query grid must be 9x9 integers")
    board = np.asarray(board, dtype=np.int8)
    if np.any((board < 0) | (board > 9)):
        raise TaskSpecializedReasonerError("Sudoku entries must be in [0, 9]")
    for unit in _SUDOKU_UNITS:
        values = [int(board[cell]) for cell in unit if board[cell] != 0]
        if len(values) != len(set(values)):
            raise TaskSpecializedReasonerError(
                "Sudoku clues violate a local constraint"
            )
    return board.copy()


def _candidate_activity(board: np.ndarray) -> tuple[np.ndarray, bool]:
    activity = np.zeros((9, 9, 9), dtype=bool)
    contradiction = any(
        len(values := [int(board[cell]) for cell in unit if board[cell] != 0])
        != len(set(values))
        for unit in _SUDOKU_UNITS
    )
    for row in range(9):
        for col in range(9):
            value = int(board[row, col])
            if value:
                activity[row, col, value - 1] = True
                continue
            forbidden = {int(board[cell]) for cell in _SUDOKU_PEERS[(row, col)]}
            allowed = [digit for digit in range(1, 10) if digit not in forbidden]
            if not allowed:
                contradiction = True
            else:
                activity[row, col, np.asarray(allowed) - 1] = True
    return activity, contradiction


def _activity_summary(board: np.ndarray, activity: np.ndarray) -> np.ndarray:
    unresolved = board == 0
    counts = activity.sum(axis=2)
    probabilities = activity / np.maximum(counts[..., None], 1)
    safe_probabilities = np.where(probabilities > 0, probabilities, 1.0)
    entropy = -np.sum(probabilities * np.log(safe_probabilities), axis=2)
    return np.asarray(
        [
            float(np.mean(unresolved)),
            float(np.mean(counts[unresolved])) if np.any(unresolved) else 1.0,
            float(np.mean(entropy[unresolved])) if np.any(unresolved) else 0.0,
        ]
    )


def _propagate_sudoku(
    initial: np.ndarray, *, max_steps: int
) -> tuple[np.ndarray, np.ndarray, int, int, int, bool]:
    board = initial.copy()
    trace: list[np.ndarray] = []
    assignments_total = 0
    update_rounds = 0
    contradiction = False
    for _ in range(max_steps):
        activity, contradiction = _candidate_activity(board)
        trace.append(_activity_summary(board, activity))
        if contradiction or not np.any(board == 0):
            break
        assignments: dict[tuple[int, int], int] = {}
        for row, col in np.argwhere(board == 0):
            digits = np.flatnonzero(activity[row, col]) + 1
            if len(digits) == 1:
                assignments[(int(row), int(col))] = int(digits[0])
        for unit in _SUDOKU_UNITS:
            for digit in range(1, 10):
                positions = [
                    cell
                    for cell in unit
                    if board[cell] == 0 and activity[cell][digit - 1]
                ]
                if len(positions) == 1:
                    cell = positions[0]
                    previous = assignments.get(cell)
                    if previous is not None and previous != digit:
                        contradiction = True
                        break
                    assignments[cell] = digit
            if contradiction:
                break
        if contradiction or not assignments:
            break
        for cell, digit in assignments.items():
            board[cell] = digit
        assignments_total += len(assignments)
        update_rounds += 1
    activity, final_contradiction = _candidate_activity(board)
    contradiction = contradiction or final_contradiction
    if not trace or not np.array_equal(trace[-1], _activity_summary(board, activity)):
        trace.append(_activity_summary(board, activity))
    return (
        board,
        np.stack(trace),
        len(trace),
        update_rounds,
        assignments_total,
        contradiction,
    )


class SudokuConstraintDynamics:
    """Sparse local constraint dynamics with an explicit optional branch budget."""

    used_bptt = False

    def __init__(self, *, max_steps: int = 128, branch_budget: int = 0) -> None:
        if max_steps < 1 or branch_budget < 0:
            raise TaskSpecializedReasonerError("invalid Sudoku compute budget")
        self.max_steps = int(max_steps)
        self.branch_budget = int(branch_budget)

    def solve(self, task: PublicTask) -> TaskSpecializedResult:
        initial = _sudoku_board(task)
        branch_counter = [0]
        trace_rows: list[np.ndarray] = []
        state_evaluations = 0
        local_update_rounds = 0
        local_assignments = 0

        def search(board: np.ndarray) -> tuple[np.ndarray, bool, bool]:
            nonlocal state_evaluations, local_assignments, local_update_rounds
            (
                propagated,
                trace,
                evaluations,
                update_rounds,
                assignments,
                contradiction,
            ) = _propagate_sudoku(board, max_steps=self.max_steps)
            trace_rows.extend(trace)
            state_evaluations += evaluations
            local_update_rounds += update_rounds
            local_assignments += assignments
            if contradiction:
                return propagated, False, True
            if not np.any(propagated == 0):
                return propagated, True, False
            if branch_counter[0] >= self.branch_budget:
                return propagated, False, False
            activity, _ = _candidate_activity(propagated)
            unresolved = np.argwhere(propagated == 0)
            row, col = min(
                unresolved,
                key=lambda cell: int(activity[int(cell[0]), int(cell[1])].sum()),
            )
            digits = np.flatnonzero(activity[int(row), int(col)]) + 1
            last = propagated
            all_tried = True
            all_contradictory = True
            for digit in digits:
                if branch_counter[0] >= self.branch_budget:
                    all_tried = False
                    break
                branch_counter[0] += 1
                proposal = propagated.copy()
                proposal[int(row), int(col)] = int(digit)
                last, solved, child_contradiction = search(proposal)
                if solved:
                    return last, True, False
                all_contradictory = all_contradictory and child_contradiction
            return last, False, all_tried and all_contradictory

        output, solved, contradiction = search(initial)
        return TaskSpecializedResult(
            task_id=task.task_id,
            family="sudoku",
            output=output,
            state_trace=np.stack(trace_rows),
            receipt={
                "architecture": "sparse_local_sudoku_constraint_dynamics_v1",
                "inspiration_scope": "task_design_only_not_bdh_reimplementation",
                "used_bptt": False,
                "spiking_required": False,
                "max_steps_per_propagation": self.max_steps,
                "branch_budget": self.branch_budget,
                "branches_used": branch_counter[0],
                "used_branch_search": branch_counter[0] > 0,
                "propagation_state_evaluations": state_evaluations,
                "local_update_rounds": local_update_rounds,
                "local_assignments": local_assignments,
                "solved": solved,
                "contradiction": contradiction,
            },
        )


def _arc_operator_family(candidate_id: str) -> str:
    base = candidate_id.split("__", 1)[0]
    if base.startswith("rot") or base in {
        "flip_lr",
        "flip_ud",
        "transpose",
        "anti_transpose",
    }:
        return "geometry"
    if base.startswith("repeat") or base.startswith("downsample"):
        return "rescale"
    if base.startswith("crop"):
        return "extract"
    return "identity"


class ARCSlowFastProgramReasoner:
    """Slow operator-family belief with fast demonstration-grounded selection."""

    used_bptt = False

    def __init__(
        self,
        *,
        max_candidates: int = 96,
        max_steps: int = 8,
        belief_decay: float = 0.5,
        halt_margin: float = 0.98,
    ) -> None:
        if max_candidates < 1 or max_steps < 1:
            raise TaskSpecializedReasonerError("ARC budgets must be positive")
        if not 0.0 <= belief_decay < 1.0 or not 0.0 < halt_margin <= 1.0:
            raise TaskSpecializedReasonerError("invalid ARC belief dynamics")
        self.max_candidates = int(max_candidates)
        self.max_steps = int(max_steps)
        self.belief_decay = float(belief_decay)
        self.halt_margin = float(halt_margin)

    def solve(self, task: PublicTask) -> TaskSpecializedResult:
        if not isinstance(task, PublicTask) or task.family != "arc":
            raise TaskSpecializedReasonerError("ARC reasoner requires an ARC task")
        proposals: ProposalBatch = generate_arc_proposals(
            task, max_candidates=self.max_candidates
        )
        feature = {name: index for index, name in enumerate(FEATURE_NAMES)}
        local_evidence = (
            6.0 * proposals.features[:, feature["support_exact_rate"]]
            + 2.0 * proposals.features[:, feature["support_cell_accuracy"]]
            + proposals.features[:, feature["support_shape_rate"]]
            + 0.5 * proposals.features[:, feature["support_color_jaccard"]]
            - 0.1 * proposals.features[:, feature["normalized_complexity"]]
        )
        families = tuple(
            _arc_operator_family(value) for value in proposals.candidate_ids
        )
        family_names = tuple(sorted(set(families)))
        family_evidence = np.asarray(
            [
                float(np.max(local_evidence[np.asarray(families) == family]))
                for family in family_names
            ]
        )
        slow_state = np.zeros(len(family_names), dtype=float)
        trace = []
        selected = 0
        stable_steps = 0
        for _ in range(self.max_steps):
            slow_state = self.belief_decay * slow_state + family_evidence
            slow_probability = np.exp(slow_state - np.max(slow_state))
            slow_probability /= slow_probability.sum()
            family_bonus = np.asarray(
                [
                    np.log(slow_probability[family_names.index(family)] + 1e-12)
                    for family in families
                ]
            )
            scores = local_evidence + family_bonus
            new_selected = int(np.argmax(scores))
            stable_steps = stable_steps + 1 if new_selected == selected else 0
            selected = new_selected
            ordered = np.sort(slow_probability)
            margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 1.0
            trace.append(
                np.concatenate(
                    [slow_probability, [margin, float(selected), float(stable_steps)]]
                )
            )
            if margin >= self.halt_margin and stable_steps >= 1:
                break
        return TaskSpecializedResult(
            task_id=task.task_id,
            family="arc",
            output=proposals.outputs[selected],
            state_trace=np.stack(trace),
            receipt={
                "architecture": "arc_slow_fast_program_belief_v1",
                "inspiration_scope": "task_design_only_not_hrm_reimplementation",
                "used_bptt": False,
                "spiking_required": False,
                "candidate_generator_version": proposals.generator_version,
                "candidate_fingerprint": proposals.candidate_fingerprint,
                "candidate_count": len(proposals.candidate_ids),
                "operator_family_count": len(family_names),
                "reasoning_steps": len(trace),
                "selected_candidate_id": proposals.candidate_ids[selected],
                "selected_operator_family": families[selected],
                "halted_early": len(trace) < self.max_steps,
            },
        )


__all__ = [
    "ARCSlowFastProgramReasoner",
    "SudokuConstraintDynamics",
    "TaskSpecializedReasonerError",
    "TaskSpecializedResult",
]
