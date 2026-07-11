import numpy as np

from src.data.structured_protocol import PublicTask
from src.tasks.structured_proposals import (
    FEATURE_NAMES,
    generate_arc_proposals,
    generate_maze_proposals,
    generate_sudoku_proposals,
)


def test_arc_proposals_use_support_only_and_include_learned_recoloring() -> None:
    task = PublicTask(
        task_id="arc-recolor",
        family="arc",
        split="test",
        source_group="arc-recolor",
        augmentation_group="arc-recolor",
        context={
            "demonstrations": [
                {
                    "input": [[0, 1], [1, 0]],
                    "output": [[0, 2], [2, 0]],
                }
            ]
        },
        query={"inputs": [[[1, 0], [0, 1]]]},
    )
    proposals = generate_arc_proposals(task)
    assert proposals.features.shape[1] == len(FEATURE_NAMES)
    assert not proposals.features.flags.writeable
    assert any(
        np.array_equal(candidate[0], np.array([[2, 0], [0, 2]]))
        for candidate in proposals.outputs
    )
    assert proposals.matched_compute_budget > 0
    first = proposals.candidate_fingerprint
    assert first == generate_arc_proposals(task).candidate_fingerprint


def test_maze_proposals_include_legal_shortest_path() -> None:
    grid = np.array(
        [
            [2, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 3],
        ]
    )
    task = PublicTask(
        "maze-a",
        "maze",
        "test",
        "maze-a",
        "maze-a",
        context={},
        query={"grid": grid},
    )
    proposals = generate_maze_proposals(task)
    bfs = proposals.outputs[proposals.candidate_ids.index("bfs")]
    assert tuple(bfs[0]) == (0, 0)
    assert tuple(bfs[-1]) == (3, 3)
    assert np.all(np.abs(np.diff(bfs, axis=0)).sum(axis=1) == 1)
    assert proposals.features[proposals.candidate_ids.index("bfs"), 14] == 1.0


def test_sudoku_proposals_solve_without_target_access() -> None:
    puzzle = np.array(
        [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]
    )
    task = PublicTask(
        "sudoku-a",
        "sudoku",
        "test",
        "sudoku-a",
        "sudoku-a",
        context={},
        query={"grid": puzzle},
    )
    proposals = generate_sudoku_proposals(task)
    complete = [
        output
        for output, feature in zip(proposals.outputs, proposals.features, strict=True)
        if feature[5] == 1.0
    ]
    assert complete
    solution = complete[-1]
    assert np.all(solution[puzzle > 0] == puzzle[puzzle > 0])
    assert all(set(row) == set(range(1, 10)) for row in solution)

