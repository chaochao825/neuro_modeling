"""CSV/JSONL Sudoku loader with clue, constraint, and exact scoring."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.data.structured_protocol import (
    PublicTask,
    StructuredDataset,
    StructuredProtocolError,
    VALID_SPLITS,
    build_structured_dataset,
    public_projection_sha256,
)


_SEPARATORS = re.compile(r"[\s,;|]+")
_DIGITS = frozenset(range(1, 10))


def _board(value: object, *, name: str, allow_blanks: bool) -> np.ndarray:
    if isinstance(value, str):
        compact = _SEPARATORS.sub("", value.strip())
        if len(compact) != 81 or any(
            character not in "0123456789." for character in compact
        ):
            raise StructuredProtocolError(f"{name} must contain exactly 81 digits/dots")
        raw = np.asarray(
            [0 if character == "." else int(character) for character in compact]
        ).reshape(9, 9)
    else:
        raw = np.asarray(value)
    if raw.shape != (9, 9) or raw.dtype.kind not in {"i", "u"}:
        raise StructuredProtocolError(f"{name} must be a 9x9 integer board")
    board = np.asarray(raw, dtype=np.int8)
    minimum = 0 if allow_blanks else 1
    if np.any((board < minimum) | (board > 9)):
        raise StructuredProtocolError(f"{name} entries must be in [{minimum}, 9]")
    return board


def _constraint_metrics(board: np.ndarray) -> tuple[bool, bool, bool]:
    rows = all(set(int(value) for value in row) == _DIGITS for row in board)
    columns = all(
        set(int(value) for value in board[:, index]) == _DIGITS for index in range(9)
    )
    boxes = all(
        set(
            int(value)
            for value in board[
                box_row : box_row + 3,
                box_column : box_column + 3,
            ].ravel()
        )
        == _DIGITS
        for box_row in (0, 3, 6)
        for box_column in (0, 3, 6)
    )
    return rows, columns, boxes


def _prediction_board(prediction: object) -> np.ndarray | None:
    if prediction is None:
        return None
    if isinstance(prediction, Mapping):
        prediction = prediction.get("board", prediction.get("solution"))
        if prediction is None:
            return None
    try:
        return _board(prediction, name="prediction", allow_blanks=False)
    except (StructuredProtocolError, TypeError, ValueError):
        return None


def score_sudoku_prediction(
    task: PublicTask, prediction: object, target: object
) -> Mapping[str, object]:
    """Check clues, all row/column/box constraints, and exact solution."""

    puzzle = np.asarray(task.query["grid"])
    expected = np.asarray(target)
    board = _prediction_board(prediction)
    parseable = board is not None
    clue_mask = puzzle != 0
    clues_preserved = bool(
        parseable and np.array_equal(board[clue_mask], puzzle[clue_mask])
    )
    if board is None:
        rows_valid = columns_valid = boxes_valid = False
    else:
        rows_valid, columns_valid, boxes_valid = _constraint_metrics(board)
    valid = bool(
        parseable and clues_preserved and rows_valid and columns_valid and boxes_valid
    )
    exact = bool(valid and np.array_equal(board, expected))
    return {
        "prediction_provided": prediction is not None,
        "board_parseable": parseable,
        "clues_preserved": clues_preserved,
        "rows_valid": bool(rows_valid),
        "columns_valid": bool(columns_valid),
        "boxes_valid": bool(boxes_valid),
        "valid_solution": valid,
        "exact": exact,
    }


def _records(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(record) for record in csv.DictReader(handle)]
    if path.suffix.lower() != ".jsonl":
        raise StructuredProtocolError("Sudoku loader accepts only .csv or .jsonl")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise StructuredProtocolError(
                f"invalid Sudoku JSONL line {line_number}: {error}"
            ) from error
        if not isinstance(record, Mapping):
            raise StructuredProtocolError(
                f"Sudoku JSONL line {line_number} must be an object"
            )
        records.append(record)
    return records


def _first(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name not in record:
            continue
        value = record[name]
        if value is not None and not (isinstance(value, str) and not value.strip()):
            return value
    return None


def load_sudoku_tasks(
    path: str | Path,
    *,
    split: str | None = "test",
) -> StructuredDataset:
    """Load Sudoku puzzles while keeping solutions exclusively in TargetStore."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    records = _records(source)
    if not records:
        raise StructuredProtocolError("Sudoku dataset must not be empty")
    tasks: list[PublicTask] = []
    targets: list[np.ndarray] = []
    default_split = split
    for index, record in enumerate(records):
        puzzle_value = _first(record, ("puzzle", "clues", "board", "input", "q"))
        solution_value = _first(record, ("solution", "answer", "target", "output", "a"))
        if puzzle_value is None or solution_value is None:
            raise StructuredProtocolError(
                f"Sudoku record {index} needs puzzle/clues and solution/answer"
            )
        puzzle = _board(puzzle_value, name=f"puzzle[{index}]", allow_blanks=True)
        solution = _board(solution_value, name=f"solution[{index}]", allow_blanks=False)
        rows_valid, columns_valid, boxes_valid = _constraint_metrics(solution)
        if not (rows_valid and columns_valid and boxes_valid):
            raise StructuredProtocolError(f"solution[{index}] violates Sudoku rules")
        clue_mask = puzzle != 0
        if not np.array_equal(puzzle[clue_mask], solution[clue_mask]):
            raise StructuredProtocolError(
                f"solution[{index}] does not preserve puzzle clues"
            )
        task_id = str(record.get("task_id", f"sudoku_{index:06d}")).strip()
        public_digest = public_projection_sha256({"family": "sudoku", "puzzle": puzzle})
        declared_digest = record.get("puzzle_sha256")
        if declared_digest is not None and str(declared_digest) != public_digest:
            raise StructuredProtocolError(
                f"puzzle_sha256[{index}] disagrees with model-visible puzzle"
            )
        source_group = str(
            record.get("source_group", record.get("source", public_digest))
        ).strip()
        augmentation_group = str(record.get("augmentation_group", source_group)).strip()
        raw_split = record.get("split", default_split)
        if (
            not isinstance(raw_split, str)
            or raw_split.strip().lower() not in VALID_SPLITS
        ):
            raise StructuredProtocolError(
                f"Sudoku split must be one of {sorted(VALID_SPLITS)!r}"
            )
        tasks.append(
            PublicTask(
                task_id=task_id,
                family="sudoku",
                split=raw_split.strip().lower(),
                source_group=source_group,
                augmentation_group=augmentation_group,
                context={"support_inputs": (), "support_outputs": ()},
                query={"grid": puzzle},
                metadata={
                    "source_file": source.name,
                    "source_format": (
                        "sudoku_csv_v1"
                        if source.suffix.lower() == ".csv"
                        else "sudoku_jsonl_v1"
                    ),
                    "source_version": str(record.get("source_version", "sudoku_v1")),
                    "source_repository": record.get("source_repository"),
                    "source_record_path": record.get("source_record_path"),
                    "source_record_sha256": record.get("source_record_sha256"),
                    "source_license": record.get("source_license"),
                    "split_provenance": record.get("split_provenance"),
                    "source_sha256": public_digest,
                    "source_hash_scope": "public_projection",
                    "n_clues": int(clue_mask.sum()),
                    "difficulty_rating": record.get("rating"),
                },
            )
        )
        targets.append(solution)
    return build_structured_dataset(tasks, targets, scorer=score_sudoku_prediction)


load_sudoku_dataset = load_sudoku_tasks
