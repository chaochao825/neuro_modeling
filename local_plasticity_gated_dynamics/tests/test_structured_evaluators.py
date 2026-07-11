"""Loader and family-level scoring tests for ARC, maze, and Sudoku."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.analysis.structured_reasoning_metrics import (
    evaluate_arc_predictions,
    evaluate_maze_predictions,
    evaluate_sudoku_predictions,
)
from src.data.arc_tasks import load_arc_directory
from src.data.maze_tasks import load_maze_tasks
from src.data.sudoku_tasks import load_sudoku_tasks
from src.tasks.structured_proposals import generate_structured_proposals


_SOLUTION = (
    "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
)
_PUZZLE = (
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
)


def _arc_payload(offset: int = 0) -> dict[str, object]:
    return {
        "train": [
            {
                "input": [[offset, 1], [1, offset]],
                "output": [[1, offset], [offset, 1]],
            }
        ],
        "test": [
            {"input": [[offset]], "output": [[1]]},
            {"input": [[1]], "output": [[offset]]},
        ],
    }


def test_arc_loader_hides_query_outputs_and_requires_all_query_exact(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.json").write_text(json.dumps(_arc_payload()), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(_arc_payload(2)), encoding="utf-8")
    dataset = load_arc_directory(tmp_path, split="test")
    assert dataset.families == {"arc"}
    assert "output" not in dataset.tasks[0].query
    assert len(dataset.tasks[0].query_inputs) == 2
    assert dataset.tasks[0].metadata["source_format"] == "official_arc_json"
    assert len(dataset.tasks[0].metadata["source_sha256"]) == 64
    assert generate_structured_proposals(dataset.tasks[0]).outputs
    evaluation = evaluate_arc_predictions(
        dataset,
        {"a": [[[1]], [[0]]]},
        bootstrap_seed=5,
        n_bootstrap=100,
    )
    assert evaluation.summary["n_tasks"] == 2
    assert evaluation.summary["n_missing"] == 1
    assert evaluation.summary["all_query_exact_rate"] == 0.5
    assert evaluation.summary["denominator_includes_missing_and_failed"]

    partial = evaluate_arc_predictions(
        dataset,
        {"a": [[[1]], [[9]]], "b": [[[1]], [[2]]]},
        bootstrap_seed=5,
        n_bootstrap=100,
    )
    a_row = partial.task_metrics.set_index("task_id").loc["a"]
    assert a_row["query_exact_fraction"] == 0.5
    assert not bool(a_row["all_query_exact"])


def test_maze_jsonl_accepts_distinct_legal_shortest_paths_and_keeps_failures(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mazes.jsonl"
    records = [
        {
            "task_id": "m0",
            "maze": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "start": [0, 0],
            "goal": [2, 2],
        },
        {
            "task_id": "m1",
            "maze": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "start": [0, 0],
            "goal": [2, 2],
        },
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    dataset = load_maze_tasks(source)
    assert generate_structured_proposals(dataset.tasks[0]).outputs
    first_path = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]

    def predictor(task):
        if task.task_id == "m1":
            raise RuntimeError("solver failed")
        return first_path

    evaluation = evaluate_maze_predictions(
        dataset,
        predictor,
        bootstrap_seed=11,
        n_bootstrap=100,
    )
    assert evaluation.summary["path_valid_rate"] == 0.5
    assert evaluation.summary["path_optimal_rate"] == 0.5
    assert evaluation.summary["n_failed"] == 1

    alternative = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    alternate_score = dataset.target_store.score(dataset.tasks[0], alternative)
    assert alternate_score["path_optimal"]


def test_maze_npz_loads_without_pickle_and_freezes_arrays(tmp_path: Path) -> None:
    path = tmp_path / "mazes.npz"
    np.savez(
        path,
        mazes=np.zeros((1, 2, 2), dtype=np.int8),
        starts=np.array([[0, 0]], dtype=np.int8),
        goals=np.array([[1, 1]], dtype=np.int8),
        task_ids=np.array(["npz0"]),
        shortest_distances=np.array([2]),
    )
    dataset = load_maze_tasks(path)
    maze = dataset.tasks[0].query["grid"]
    assert not maze.flags.writeable
    assert dataset.tasks[0].metadata["source_format"] == "maze_npz_v1"


def test_sudoku_csv_checks_constraints_clues_exact_and_missing_denominator(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sudoku.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "puzzle", "solution"])
        writer.writeheader()
        writer.writerow({"task_id": "s0", "puzzle": _PUZZLE, "solution": _SOLUTION})
        writer.writerow({"task_id": "s1", "puzzle": _PUZZLE, "solution": _SOLUTION})
    dataset = load_sudoku_tasks(path)
    assert generate_structured_proposals(dataset.tasks[0]).outputs
    assert "solution" not in dataset.tasks[0].query
    evaluation = evaluate_sudoku_predictions(
        dataset,
        {"s0": _SOLUTION},
        bootstrap_seed=3,
        n_bootstrap=100,
    )
    assert evaluation.summary["n_tasks"] == 2
    assert evaluation.summary["clues_preserved_rate"] == 0.5
    assert evaluation.summary["valid_solution_rate"] == 0.5
    assert evaluation.summary["exact_rate"] == 0.5

    bad = "1" * 81
    score = dataset.target_store.score(dataset.tasks[0], bad)
    assert not score["clues_preserved"]
    assert not score["rows_valid"]
    assert not score["exact"]


def test_sudoku_jsonl_loader(tmp_path: Path) -> None:
    path = tmp_path / "sudoku.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "json0",
                "puzzle": _PUZZLE,
                "solution": _SOLUTION,
                "source_version": "fixture-1",
            }
        ),
        encoding="utf-8",
    )
    dataset = load_sudoku_tasks(path, split="validation")
    assert dataset.validation_task_ids == ("json0",)
    assert dataset.target_store.training_view(dataset.tasks[0]).target.shape == (9, 9)


def test_public_fingerprints_are_invariant_to_hidden_target_mutations(
    tmp_path: Path,
) -> None:
    arc_dir = tmp_path / "arc"
    arc_dir.mkdir()
    arc_path = arc_dir / "task.json"
    first_arc = _arc_payload()
    arc_path.write_text(json.dumps(first_arc), encoding="utf-8")
    arc_before = load_arc_directory(arc_dir, split="test").tasks[0]
    first_arc["test"][0]["output"] = [[9]]
    arc_path.write_text(json.dumps(first_arc), encoding="utf-8")
    arc_after = load_arc_directory(arc_dir, split="test").tasks[0]
    assert arc_before.fingerprint == arc_after.fingerprint

    maze_path = tmp_path / "maze.jsonl"
    maze_record = {
        "task_id": "m",
        "maze": [[0, 0], [0, 0]],
        "start": [0, 0],
        "goal": [1, 1],
        "shortest_paths": [[[0, 0], [1, 0], [1, 1]]],
    }
    maze_path.write_text(json.dumps(maze_record), encoding="utf-8")
    maze_before = load_maze_tasks(maze_path).tasks[0]
    maze_record["shortest_paths"] = [[[0, 0], [0, 1], [1, 1]]]
    maze_path.write_text(json.dumps(maze_record), encoding="utf-8")
    maze_after = load_maze_tasks(maze_path).tasks[0]
    assert maze_before.fingerprint == maze_after.fingerprint

    sudoku_path = tmp_path / "sudoku.jsonl"
    sudoku_record = {"task_id": "s", "puzzle": "0" * 81, "solution": _SOLUTION}
    sudoku_path.write_text(json.dumps(sudoku_record), encoding="utf-8")
    sudoku_before = load_sudoku_tasks(sudoku_path).tasks[0]
    swapped = _SOLUTION.translate(str.maketrans("12", "21"))
    sudoku_record["solution"] = swapped
    sudoku_path.write_text(json.dumps(sudoku_record), encoding="utf-8")
    sudoku_after = load_sudoku_tasks(sudoku_path).tasks[0]
    assert sudoku_before.fingerprint == sudoku_after.fingerprint
