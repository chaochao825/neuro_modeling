from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.common import load_json_config
from experiments.exp13_structured_reasoning import run_seed
from src.data.arc_tasks import load_arc_directory
from src.data.structured_protocol import StructuredProtocolError


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_exp13_smoke_runs_matched_target_safe_panel(tmp_path: Path) -> None:
    config = load_json_config("configs/smoke/exp13_structured_reasoning.json")
    path = run_seed(config, 0, str(tmp_path / "results"))
    records = _records(path)
    assert len(records) == 6
    assert {record["status"] for record in records} == {"complete"}
    by_condition = {record["condition"]: record for record in records}
    assert by_condition["gru_bptt"]["used_bptt"] is True
    assert by_condition["hierarchical_local"]["used_bptt"] is False
    assert by_condition["hierarchical_local"]["spiking_model"] is False
    assert by_condition["hierarchical_local"]["control_dim"] == 3
    assert by_condition["flat_local"]["control_dim"] == 0
    assert by_condition["flat_local"]["control_operator_rank"] == 0
    assert by_condition["candidate_oracle"]["selection_accessed_query_target"] is True
    assert all(
        record["query_targets_exposed_to_solver"] is False for record in records
    )
    assert all(record["fixture_only"] is True for record in records)

    task_metrics = pd.read_csv(path / "task_metrics.csv.gz")
    fingerprint_counts = task_metrics.groupby("task_id")[
        "candidate_fingerprint"
    ].nunique()
    assert fingerprint_counts.eq(1).all()
    assert task_metrics.groupby("task_id")["n_candidates"].nunique().eq(1).all()
    assert (path / "public_task_provenance.json").is_file()
    assert (path / "fit_receipts.json").is_file()


def test_formal_unlicensed_dataset_stays_fail_closed(tmp_path: Path) -> None:
    config = load_json_config(
        "configs/formal/exp13_structured_reasoning_maze.json"
    )
    path = run_seed(config, 0, str(tmp_path / "results"))
    records = _records(path)
    assert len(records) == len(config["conditions"])
    assert {record["status"] for record in records} == {"failed"}
    assert all("license_status" in record["error"] for record in records)


def test_arc_duplicate_is_detected_and_explicit_exclusion_is_namespaced(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation"
    training.mkdir()
    evaluation.mkdir()
    duplicate = {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[1]], "output": [[0]]}],
    }
    unique = {
        "train": [{"input": [[2]], "output": [[3]]}],
        "test": [{"input": [[3]], "output": [[2]]}],
    }
    (training / "original.json").write_text(json.dumps(duplicate), encoding="utf-8")
    (evaluation / "renamed.json").write_text(json.dumps(duplicate), encoding="utf-8")
    (evaluation / "unique.json").write_text(json.dumps(unique), encoding="utf-8")
    with pytest.raises(StructuredProtocolError, match="crosses splits"):
        load_arc_directory(tmp_path)

    dataset = load_arc_directory(
        tmp_path,
        dataset_name="ARC-X",
        dataset_revision="deadbeef",
        exclude_relative_paths=("evaluation/renamed.json",),
        namespace_task_ids=True,
    )
    assert len(dataset.for_split("train")) == 1
    assert len(dataset.for_split("test")) == 1
    assert all(task.task_id.startswith("ARC-X@deadbeef:") for task in dataset.tasks)
    assert all(
        task.metadata["excluded_relative_paths"] == ("evaluation/renamed.json",)
        for task in dataset.tasks
    )

