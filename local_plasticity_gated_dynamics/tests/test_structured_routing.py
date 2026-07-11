"""Task-safe contracts for the secondary structured routing benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import experiments.exp12_structured_routing as exp12
from experiments.common import load_json_config
from experiments.exp12_structured_routing import run_seed
from src.analysis.structured_routing_metrics import (
    ROUTING_CONDITIONS,
    assert_matched_routing_contract,
    evaluate_structured_routing,
)
from src.data.structured_task_dataset import (
    StructuredTaskDataError,
    load_structured_task_tape,
    make_synthetic_structured_tape,
)


def _dataset(tmp_path: Path):
    payload = make_synthetic_structured_tape(
        n_train_tasks=8,
        n_test_tasks=5,
        n_candidates=4,
        feature_dim=4,
        missing_test_tasks=1,
        seed=9,
    )
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_structured_task_tape(path)


def test_frozen_tape_keeps_complete_tasks_and_missing_candidate_failures(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    assert len(dataset.train_tasks) == 8
    assert len(dataset.test_tasks) == 5
    assert not (
        {task.task_id for task in dataset.train_tasks}
        & {task.task_id for task in dataset.test_tasks}
    )
    assert sum(not task.candidates for task in dataset.test_tasks) == 1
    for task in dataset.tasks:
        assert not (set(task.train_example_ids) & set(task.test_example_ids))


def test_tape_fails_closed_on_cross_split_provenance_or_example_reuse(
    tmp_path: Path,
) -> None:
    payload = make_synthetic_structured_tape(
        n_train_tasks=2,
        n_test_tasks=2,
        n_candidates=2,
        feature_dim=2,
        seed=4,
    )
    payload["tasks"][2]["provenance_hash"] = payload["tasks"][0]["provenance_hash"]
    path = tmp_path / "duplicate_provenance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StructuredTaskDataError, match="provenance_hash"):
        load_structured_task_tape(path)

    payload = make_synthetic_structured_tape(
        n_train_tasks=2,
        n_test_tasks=2,
        n_candidates=2,
        feature_dim=2,
        seed=5,
    )
    payload["tasks"][2]["partitions"]["train"][0] = payload["tasks"][0]["partitions"][
        "train"
    ][0]
    path = tmp_path / "duplicate_example.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StructuredTaskDataError, match="globally unique"):
        load_structured_task_tape(path)

    payload = make_synthetic_structured_tape(
        n_train_tasks=2,
        n_test_tasks=2,
        n_candidates=2,
        feature_dim=2,
        seed=6,
    )
    candidate = payload["tasks"][0]["candidates"][0]
    candidate["feature_source_example_ids"] = payload["tasks"][0]["partitions"]["test"]
    path = tmp_path / "feature_test_leak.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StructuredTaskDataError, match="subset of train"):
        load_structured_task_tape(path)


def test_tape_fingerprint_binds_freeze_and_generator_header(tmp_path: Path) -> None:
    payload = make_synthetic_structured_tape(
        n_train_tasks=2,
        n_test_tasks=2,
        n_candidates=2,
        feature_dim=2,
        seed=11,
    )
    path = tmp_path / "header_bound.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    baseline = load_structured_task_tape(path).tape_fingerprint

    payload["frozen_before_evaluation"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    frozen = load_structured_task_tape(path).tape_fingerprint
    assert frozen != baseline

    payload["candidate_generator_commit"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    generator_changed = load_structured_task_tape(path).tape_fingerprint
    assert generator_changed not in {baseline, frozen}

    payload["schema_version"] = "2.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    schema_changed = load_structured_task_tape(path).tape_fingerprint
    assert schema_changed not in {baseline, frozen, generator_changed}


def test_shared_router_is_train_only_and_all_conditions_are_budget_matched(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    evaluations = evaluate_structured_routing(
        dataset,
        router_dim=2,
        seed=3,
        n_bootstrap=100,
    )
    assert {item.condition for item in evaluations} == set(ROUTING_CONDITIONS)
    assert_matched_routing_contract(evaluations)
    for evaluation in evaluations:
        assert evaluation.summary["all_preprocessing_fit_on_train_tasks"]
        assert not evaluation.summary["fit_accessed_test_exact_correct"]
        assert evaluation.summary["statistics_unit"] == "task"
        assert evaluation.summary["candidate_coverage"] == 0.8
        assert len(evaluation.task_metrics) == 5
        assert (
            evaluation.task_metrics["matched_compute_budget"].to_numpy()
            == evaluations[0].task_metrics["matched_compute_budget"].to_numpy()
        ).all()
    oracle = next(
        item for item in evaluations if item.condition == "per_task_oracle_ceiling"
    )
    assert oracle.summary["selection_accessed_test_exact_correct"]
    shared = next(item for item in evaluations if item.condition == "shared_router")
    assert not shared.summary["selection_accessed_test_exact_correct"]


def test_exp12_smoke_marks_fixture_as_non_scientific(tmp_path: Path) -> None:
    config = load_json_config("configs/smoke/exp12_structured_routing.json")
    path = run_seed(config, 0, str(tmp_path))
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert {record["status"] for record in records} == {"complete"}
    assert all(record["fixture_only"] for record in records)
    assert all(not record["scientific_evidence_eligible"] for record in records)
    assert all(not record["candidate_generation_frozen"] for record in records)
    assert all(not record["input_feature_provenance_validated"] for record in records)
    assert all(
        record["provenance_validation_level"] == "schema_attestation_only"
        for record in records
    )
    assert all(not record["efficiency_claim_eligible"] for record in records)
    assert all(not record["neural_evidence_claim"] for record in records)
    task_rows = pd.read_csv(path / "task_metrics.csv")
    assert len(task_rows) == 18
    assert set(task_rows["task_status"]) == {"complete", "missing_candidate_set"}


def test_formal_self_attested_tape_remains_scientifically_ineligible(
    tmp_path: Path,
) -> None:
    payload = make_synthetic_structured_tape(
        n_train_tasks=8,
        n_test_tasks=5,
        n_candidates=4,
        feature_dim=4,
        seed=13,
    )
    payload["frozen_before_evaluation"] = True
    tape = tmp_path / "self_attested.json"
    tape.write_text(json.dumps(payload), encoding="utf-8")
    config = load_json_config("configs/formal/exp12_structured_routing.json")
    config.update(
        tape_path=str(tape),
        router_dim=2,
        n_bootstrap=20,
        evidence_contract={
            "minimum_test_tasks": 1,
            "minimum_candidate_coverage": 0.0,
        },
    )
    path = run_seed(config, 0, str(tmp_path / "runs"))
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert all(record["schema_contract_eligible"] for record in records)
    assert all(not record["scientific_evidence_eligible"] for record in records)
    assert all(not record["input_feature_provenance_recomputed"] for record in records)


def test_exp12_exception_path_does_not_duplicate_completed_condition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = exp12.evaluate_structured_routing

    def inject_second_record_collision(*args, **kwargs):
        evaluations = original(*args, **kwargs)
        evaluations[1].summary["routing_condition"] = "overlap"
        return evaluations

    monkeypatch.setattr(
        exp12,
        "evaluate_structured_routing",
        inject_second_record_collision,
    )
    config = load_json_config("configs/smoke/exp12_structured_routing.json")
    path = exp12.run_seed(config, 0, str(tmp_path))
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert len({record["condition"] for record in records}) == 3
    statuses = {record["condition"]: record["status"] for record in records}
    assert statuses == {
        "no_router": "complete",
        "shared_router": "failed",
        "per_task_oracle_ceiling": "failed",
    }
