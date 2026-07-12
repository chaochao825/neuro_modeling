from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.common import load_json_config
from experiments.exp15_task_specialized_reasoning import _bootstrap_accuracy, run_seed


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_exp15_bootstrap_weights_source_groups_not_tasks() -> None:
    estimate, low, high = _bootstrap_accuracy(
        np.asarray([1.0, 0.0, 0.0]),
        ("two-task-group", "two-task-group", "one-task-group"),
        n_bootstrap=200,
        seed=7,
    )
    assert estimate == pytest.approx(0.25)
    assert 0.0 <= low <= estimate <= high <= 0.5


@pytest.mark.parametrize(
    ("config_path", "expected_conditions", "expected_tasks"),
    [
        (
            "configs/smoke/exp15_task_specialized_arc.json",
            {"arc_slow_fast_program"},
            4,
        ),
        (
            "configs/smoke/exp15_task_specialized_sudoku.json",
            {"sudoku_local_no_branch", "sudoku_local_bounded_branch"},
            4,
        ),
    ],
)
def test_exp15_smoke_is_target_safe_and_never_promotes_an_advantage_claim(
    tmp_path: Path,
    config_path: str,
    expected_conditions: set[str],
    expected_tasks: int,
) -> None:
    config = load_json_config(config_path)
    config["n_bootstrap"] = 100
    path = run_seed(config, 0, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    records = _records(path)
    aggregates = [row for row in records if row.get("stage") == "aggregate"]
    task_rows = [row for row in records if row.get("stage") == "task_test"]

    assert status["status"] == "complete"
    assert {row["condition"] for row in aggregates} == expected_conditions
    assert len(task_rows) == expected_tasks * len(expected_conditions)
    assert all(row["used_bptt"] is False for row in records)
    assert all(row["spiking_required"] is False for row in records)
    assert all(row["core_claim_eligible"] is False for row in aggregates)
    assert all(row["claim_conclusion"] == "inconclusive" for row in aggregates)
    assert all(
        row["matched_advantage_comparator_registered"] is False for row in aggregates
    )
    assert all(row["fixture_only"] is True for row in aggregates)
    assert all(row["source_manifest_verified"] is False for row in aggregates)
    assert all(row["statistics_unit"] == "source_group" for row in aggregates)
    assert all(0.0 <= row["functional_success_rate"] <= 1.0 for row in aggregates)
