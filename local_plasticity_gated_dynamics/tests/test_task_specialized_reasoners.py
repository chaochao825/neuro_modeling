from __future__ import annotations

import numpy as np
import pytest

from src.data.structured_protocol import PublicTask
from src.models.task_specialized_reasoners import (
    ARCSlowFastProgramReasoner,
    SudokuConstraintDynamics,
    TaskSpecializedReasonerError,
)


def _sudoku_solution() -> np.ndarray:
    return np.asarray(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [4, 5, 6, 7, 8, 9, 1, 2, 3],
            [7, 8, 9, 1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7, 8, 9, 1],
            [5, 6, 7, 8, 9, 1, 2, 3, 4],
            [8, 9, 1, 2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8, 9, 1, 2],
            [6, 7, 8, 9, 1, 2, 3, 4, 5],
            [9, 1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=np.int8,
    )


def _sudoku_task(grid: np.ndarray) -> PublicTask:
    return PublicTask(
        task_id="sudoku-public",
        family="sudoku",
        split="test",
        source_group="sudoku-source",
        augmentation_group="sudoku-source",
        context={"rules": "standard_9x9"},
        query={"grid": grid},
    )


def _arc_task() -> tuple[PublicTask, np.ndarray]:
    support_a = np.asarray([[1, 0, 0], [2, 0, 0]])
    support_b = np.asarray([[0, 3], [0, 0], [0, 0]])
    query = np.asarray([[1, 2, 0], [0, 0, 0]])
    task = PublicTask(
        task_id="arc-rotate",
        family="arc",
        split="test",
        source_group="arc-source",
        augmentation_group="arc-source",
        context={
            "support_inputs": (support_a, support_b),
            "support_outputs": (np.rot90(support_a), np.rot90(support_b)),
        },
        query={"inputs": (query,)},
    )
    return task, np.rot90(query)


def test_sudoku_local_constraint_dynamics_solves_without_target_or_bptt() -> None:
    solution = _sudoku_solution()
    puzzle = solution.copy()
    puzzle[0, 0] = 0
    result = SudokuConstraintDynamics(branch_budget=0).solve(_sudoku_task(puzzle))

    np.testing.assert_array_equal(result.output, solution)
    assert result.receipt["solved"] is True
    assert result.receipt["used_branch_search"] is False
    assert result.receipt["used_bptt"] is False
    assert result.receipt["spiking_required"] is False
    assert result.state_trace.shape[1] == 3
    assert not result.state_trace.flags.writeable
    assert isinstance(result.output, np.ndarray)
    assert not result.output.flags.writeable


def test_sudoku_branch_search_is_explicit_and_budgeted() -> None:
    empty = np.zeros((9, 9), dtype=np.int8)
    no_branch = SudokuConstraintDynamics(branch_budget=0).solve(_sudoku_task(empty))
    one_branch = SudokuConstraintDynamics(branch_budget=1).solve(_sudoku_task(empty))

    assert no_branch.receipt["branches_used"] == 0
    assert no_branch.receipt["solved"] is False
    assert one_branch.receipt["branches_used"] == 1
    assert one_branch.receipt["branches_used"] <= one_branch.receipt["branch_budget"]
    assert one_branch.receipt["used_branch_search"] is True
    assert one_branch.receipt["solved"] is False
    assert one_branch.receipt["local_update_rounds"] <= (
        one_branch.receipt["max_steps_per_propagation"]
        * (one_branch.receipt["branches_used"] + 1)
    )


def test_sudoku_rejects_inconsistent_public_clues() -> None:
    broken = np.zeros((9, 9), dtype=np.int8)
    broken[0, :2] = 1
    with pytest.raises(TaskSpecializedReasonerError, match="clues violate"):
        SudokuConstraintDynamics().solve(_sudoku_task(broken))


def test_sudoku_never_marks_a_duplicate_unit_as_solved() -> None:
    puzzle = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 7, 0, 0],
            [0, 7, 0, 0, 2, 0, 0, 0, 9],
            [0, 0, 0, 0, 0, 0, 0, 5, 0],
            [0, 9, 0, 0, 0, 0, 0, 0, 0],
            [7, 0, 0, 2, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 5, 0, 0, 0],
            [0, 6, 9, 0, 0, 0, 8, 0, 7],
            [5, 1, 0, 0, 0, 0, 0, 9, 0],
            [8, 3, 0, 0, 0, 0, 4, 0, 2],
        ],
        dtype=np.int8,
    )
    result = SudokuConstraintDynamics(branch_budget=64).solve(_sudoku_task(puzzle))
    valid = False
    if result.receipt["solved"]:
        board = np.asarray(result.output)
        expected = set(range(1, 10))
        units = [*board, *board.T]
        units.extend(
            board[row : row + 3, col : col + 3].ravel()
            for row in (0, 3, 6)
            for col in (0, 3, 6)
        )
        valid = all(set(int(value) for value in unit) == expected for unit in units)
    assert result.receipt["solved"] is False or valid
    exhausted = SudokuConstraintDynamics(branch_budget=256).solve(_sudoku_task(puzzle))
    assert exhausted.receipt["branches_used"] <= 256
    assert exhausted.receipt["solved"] is False
    assert exhausted.receipt["contradiction"] is True


def test_arc_slow_fast_reasoner_selects_demo_consistent_program() -> None:
    task, expected = _arc_task()
    result = ARCSlowFastProgramReasoner(max_steps=5).solve(task)

    assert isinstance(result.output, tuple)
    np.testing.assert_array_equal(result.output[0], expected)
    assert result.receipt["selected_candidate_id"] == "rot90"
    assert result.receipt["selected_operator_family"] == "geometry"
    assert result.receipt["used_bptt"] is False
    assert result.receipt["spiking_required"] is False
    assert 1 <= result.receipt["reasoning_steps"] <= 5
    assert result.state_trace.shape[0] == result.receipt["reasoning_steps"]


def test_task_specific_reasoners_reject_the_wrong_family() -> None:
    arc, _ = _arc_task()
    sudoku = _sudoku_task(_sudoku_solution())
    with pytest.raises(TaskSpecializedReasonerError, match="Sudoku task"):
        SudokuConstraintDynamics().solve(arc)
    with pytest.raises(TaskSpecializedReasonerError, match="ARC task"):
        ARCSlowFastProgramReasoner().solve(sudoku)
