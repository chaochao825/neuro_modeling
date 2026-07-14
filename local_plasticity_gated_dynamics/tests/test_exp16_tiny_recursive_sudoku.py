from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from experiments.common import load_json_config
from experiments.exp16_tiny_recursive_sudoku import (
    CONDITIONS,
    _validate_calibration_freeze,
    calibration_candidate_sha256,
    calibration_environment_sha256,
    run_seed,
)
from figures.exp16_tiny_recursive_plot import plot_exp16
from scripts.summarize_exp16_tiny_recursive import (
    EXPERIMENT,
    latest_attempt_metrics,
    load_published_snapshot,
    publish_snapshot,
)
from src.utils.artifacts import ExperimentRun


def _rows(path):
    return [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_exp16_smoke_retains_matched_baseline_receipts_and_test_safety(
    tmp_path,
) -> None:
    config = load_json_config("configs/smoke/exp16_tiny_recursive_sudoku.json")
    config["seeds"] = [0]
    config["n_bootstrap"] = 100
    config["augmentations_per_task"] = 0
    config["synthetic_fixture"].update(
        n_train_tasks=8,
        n_test_tasks=3,
        clue_fraction=0.85,
    )
    config["model"].update(
        hidden_size=8,
        num_heads=2,
        layers=1,
        high_cycles=1,
        low_cycles=1,
        supervision_steps=2,
    )
    config["training"].update(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        device="cpu",
    )
    run_path = run_seed(config, 0, tmp_path)
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    planned = json.loads(
        (run_path / "planned_conditions.json").read_text(encoding="utf-8")
    )
    assert {row["condition"] for row in planned} == set(CONDITIONS)

    rows = _rows(run_path)
    aggregates = {
        row["condition"]: row for row in rows if row.get("stage") == "aggregate"
    }
    assert set(aggregates) == set(CONDITIONS)
    recursive = aggregates["micro_trm_bptt"]
    flat = aggregates["single_state_core_call_matched"]
    assert recursive["parameter_count"] == flat["parameter_count"]
    assert (
        recursive["nominal_core_calls_per_evaluation"]
        == flat["nominal_core_calls_per_evaluation"]
    )
    assert recursive["optimizer_steps"] == flat["optimizer_steps"]
    assert recursive["initialization_sha256"] == flat["initialization_sha256"]
    assert recursive["used_bptt"] is True
    assert recursive["eligible_for_local_initialization"] is False
    assert recursive["claim_conclusion"] == "inconclusive"
    assert recursive["strict_deterministic_algorithms"] is True
    assert recursive["attention_backend"] == "cpu_default"
    assert recursive["loss_scope"] == "blank_only"
    assert 0.0 <= recursive["blank_cell_accuracy"] <= 1.0
    assert 0.0 <= recursive["selected_validation_blank_cell_accuracy"] <= 1.0

    comparison = next(row for row in rows if row.get("stage") == "comparison")
    assert comparison["all_matching_gates_passed"] is True
    assert comparison["statistics_unit"] == "seed"
    assert comparison["claim_scope"] == "computational_baseline_only"
    assert comparison["claim_conclusion"] == "inconclusive"

    provenance = json.loads(
        (run_path / "source_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["test_targets_exposed_to_fit"] is False
    assert provenance["inner_groups_disjoint"] is True
    receipts = json.loads((run_path / "fit_receipts.json").read_text(encoding="utf-8"))
    assert all(
        receipt["test_data_used_for_fit"] is False for receipt in receipts.values()
    )
    assert all(
        (run_path / receipt["checkpoint_path"]).is_file()
        for receipt in receipts.values()
    )

    outputs = publish_snapshot([run_path], tmp_path, prefix="exp16_test")
    assert all(path.is_file() for path in outputs.values())
    figures = plot_exp16(tmp_path, prefix="exp16_test")
    assert all(path.is_file() for path in figures.values())
    report = outputs["report"].read_text(encoding="utf-8")
    assert "not an official HRM/TRM reproduction" in report
    assert "Conclusion: **inconclusive**" in report
    assert "formal promotion is disabled" in report
    assert "synthetic Sudoku fixture" in report
    with pytest.raises(FileExistsError, match="publication is immutable"):
        publish_snapshot([run_path], tmp_path, prefix="exp16_test")
    with pytest.raises(FileExistsError, match="figures are immutable"):
        plot_exp16(tmp_path, prefix="exp16_test")
    with pytest.raises(ValueError, match="duplicate Exp16 run directory"):
        publish_snapshot([run_path, run_path], tmp_path, prefix="exp16_duplicate_run")

    conditions = pd.read_csv(outputs["conditions"])
    conditions.loc[0, "scoped_raw_sha256"] = "0" * 64
    conditions.to_csv(outputs["conditions"], index=False)
    with pytest.raises(ValueError, match="raw binding is invalid"):
        load_published_snapshot(tmp_path, prefix="exp16_test")


def test_exp16_failed_run_is_publishable_but_never_promoted(tmp_path) -> None:
    config = load_json_config("configs/smoke/exp16_tiny_recursive_sudoku.json")
    config["seeds"] = [0]
    config["synthetic_fixture"].update(n_train_tasks=1, n_test_tasks=1)
    run_path = run_seed(config, 0, tmp_path / "runs_only")
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete_with_failures"
    assert (run_path / "fit_receipts.json").is_file()

    outputs = publish_snapshot(
        [run_path], tmp_path / "publication", prefix="exp16_failed"
    )
    comparison = pd.read_csv(outputs["comparison"])
    manifest = pd.read_csv(outputs["manifest"])
    assert comparison.empty
    assert manifest.loc[0, "run_status"] == "complete_with_failures"
    assert int(manifest.loc[0, "condition_failures"]) == 2
    report = outputs["report"].read_text(encoding="utf-8")
    assert "No complete paired comparison" in report


def test_exp16_training_setup_failure_is_recorded_for_both_conditions(
    tmp_path, monkeypatch
) -> None:
    config = load_json_config("configs/smoke/exp16_tiny_recursive_sudoku.json")

    def fail_setup(_config):
        raise RuntimeError("deliberate setup failure")

    monkeypatch.setattr(
        "experiments.exp16_tiny_recursive_sudoku._training_config", fail_setup
    )
    run_path = run_seed(config, 0, tmp_path)
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    rows = _rows(run_path)
    failures = [row for row in rows if row.get("stage") == "training_setup"]
    assert status["status"] == "complete_with_failures"
    assert {row["condition"] for row in failures} == set(CONDITIONS)
    assert json.loads((run_path / "fit_receipts.json").read_text("utf-8")) == {}


def test_exp16_confirmation_freeze_is_executable_and_hash_bound(
    tmp_path, monkeypatch
) -> None:
    config = load_json_config("configs/smoke/exp16_tiny_recursive_sudoku.json")
    candidate_sha256 = calibration_candidate_sha256(config)
    decision = {
        "status": "frozen_validation_only",
        "all_freeze_gates_passed": True,
        "enough_seeds": True,
        "all_runs_clean": True,
        "all_candidates_complete": True,
        "all_git_clean": True,
        "selected_candidate": "blank_reference",
        "selected_candidate_config_sha256": candidate_sha256,
        "submitted_seeds": [10, 11, 12],
        "test_data_used_for_fit_or_selection": False,
        "test_prediction_array_requested": False,
        "public_test_prediction_adapter_called": False,
        "hidden_target_scorer_called": False,
        "confirmation_test_still_required": True,
        "git_commit": "calibration-commit",
        "require_clean_git": True,
        "calibration_code_sha256": "c" * 64,
        "calibration_environment_sha256": calibration_environment_sha256(),
    }
    decision_path = tmp_path / "freeze.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    decision_sha256 = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    config.update(
        require_calibration_freeze=True,
        calibration_freeze={
            "freeze_decision_path": str(decision_path),
            "freeze_decision_sha256": decision_sha256,
            "selected_candidate": "blank_reference",
            "selected_candidate_config_sha256": candidate_sha256,
            "selection_seeds": [10, 11, 12],
        },
    )
    monkeypatch.setattr(
        "experiments.exp16_tiny_recursive_sudoku.calibration_code_sha256",
        lambda: "c" * 64,
    )
    monkeypatch.setattr(
        "experiments.exp16_tiny_recursive_sudoku._git_state",
        lambda: ("confirmation-commit", False),
    )
    receipt = _validate_calibration_freeze(config, confirmation_seed=20)
    assert receipt["validated"] is True
    assert receipt["calibration_git_commit"] == "calibration-commit"
    assert receipt["git_commit"] == "confirmation-commit"

    for gate in (
        "enough_seeds",
        "all_runs_clean",
        "all_candidates_complete",
        "test_data_used_for_fit_or_selection",
        "test_prediction_array_requested",
        "public_test_prediction_adapter_called",
        "hidden_target_scorer_called",
    ):
        mutated = dict(decision)
        mutated[gate] = not decision[gate]
        decision_path.write_text(json.dumps(mutated), encoding="utf-8")
        config["calibration_freeze"]["freeze_decision_sha256"] = hashlib.sha256(
            decision_path.read_bytes()
        ).hexdigest()
        with pytest.raises(ValueError, match="gates are not satisfied"):
            _validate_calibration_freeze(config, confirmation_seed=20)

    mutated = dict(decision)
    mutated["all_git_clean"] = False
    decision_path.write_text(json.dumps(mutated), encoding="utf-8")
    config["calibration_freeze"]["freeze_decision_sha256"] = hashlib.sha256(
        decision_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="not git-clean"):
        _validate_calibration_freeze(config, confirmation_seed=20)

    mutated = dict(decision)
    mutated["calibration_environment_sha256"] = "0" * 64
    decision_path.write_text(json.dumps(mutated), encoding="utf-8")
    config["calibration_freeze"]["freeze_decision_sha256"] = hashlib.sha256(
        decision_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="software environment"):
        _validate_calibration_freeze(config, confirmation_seed=20)

    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    config["calibration_freeze"]["freeze_decision_sha256"] = hashlib.sha256(
        decision_path.read_bytes()
    ).hexdigest()

    config["training"]["epochs"] += 1
    with pytest.raises(ValueError, match="frozen candidate"):
        _validate_calibration_freeze(config, confirmation_seed=20)

    config["training"]["epochs"] -= 1
    config["validation_fraction"] = 0.49
    with pytest.raises(ValueError, match="frozen candidate"):
        _validate_calibration_freeze(config, confirmation_seed=20)


def test_exp16_invalid_freeze_fails_before_dataset_or_test_access(tmp_path) -> None:
    config = load_json_config("configs/smoke/exp16_tiny_recursive_sudoku.json")
    config.update(
        require_calibration_freeze=True,
        calibration_freeze={"freeze_decision_path": str(tmp_path / "missing.json")},
    )
    run_path = run_seed(config, 20, tmp_path / "runs")
    rows = _rows(run_path)
    assert {row["stage"] for row in rows} == {"calibration_freeze"}
    assert not (run_path / "source_provenance.json").exists()
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete_with_failures"


@pytest.mark.parametrize("evidence_stage", ["frozen_confirmation", "retry_pilot"])
def test_exp16_confirmation_stage_cannot_disable_freeze_gate(
    evidence_stage,
) -> None:
    config = load_json_config("configs/smoke/exp16_tiny_recursive_sudoku.json")
    config["evidence_stage"] = evidence_stage
    config["require_calibration_freeze"] = False
    with pytest.raises(ValueError, match="requires a calibration freeze"):
        _validate_calibration_freeze(config, confirmation_seed=20)


def test_real_confirmation_cannot_bypass_formal_data_validation(tmp_path) -> None:
    config = load_json_config("configs/formal/exp17_tiny_recursive_calibration.json")
    formal_sha256 = calibration_candidate_sha256(config)
    config.update(
        profile="retry_pilot",
        require_calibration_freeze=True,
        calibration_freeze={"freeze_decision_path": str(tmp_path / "unused.json")},
    )
    assert calibration_candidate_sha256(config) != formal_sha256
    with pytest.raises(ValueError, match="formal data validation"):
        _validate_calibration_freeze(config, confirmation_seed=2000)


def test_exp16_empty_failed_attempt_and_latest_retry_selection_are_fail_closed(
    tmp_path,
) -> None:
    try:
        with ExperimentRun(
            EXPERIMENT,
            7,
            {"profile": "smoke"},
            results_root=tmp_path / "runs_only",
        ) as run:
            run.register_conditions(
                [
                    {"condition": condition, "reasoning_mode": mode}
                    for condition, mode in CONDITIONS.items()
                ]
            )
            empty_failed_run = run.path
            raise RuntimeError("failure before the first metric")
    except RuntimeError:
        pass

    outputs = publish_snapshot(
        [empty_failed_run], tmp_path / "publication", prefix="exp16_empty_failed"
    )
    raw = pd.read_csv(outputs["raw"])
    manifest = pd.read_csv(outputs["manifest"])
    assert raw.loc[0, "stage"] == "run_status"
    assert raw.loc[0, "status"] == "failed"
    assert bool(manifest.loc[0, "selected_for_descriptive_summary"])

    retry_manifest = pd.DataFrame(
        [
            {
                "seed": 0,
                "started_at": "2026-01-01T00:00:00Z",
                "published_run_path": "old",
                "run_id": "old-success",
                "selected_for_descriptive_summary": False,
            },
            {
                "seed": 0,
                "started_at": "2026-01-02T00:00:00Z",
                "published_run_path": "new",
                "run_id": "new-failure",
                "selected_for_descriptive_summary": True,
            },
        ]
    )
    retry_raw = pd.DataFrame(
        [
            {"run_id": "old-success", "stage": "aggregate"},
            {"run_id": "new-failure", "stage": "run_status"},
        ]
    )
    selected = latest_attempt_metrics(retry_raw, retry_manifest)
    assert selected["run_id"].tolist() == ["new-failure"]
