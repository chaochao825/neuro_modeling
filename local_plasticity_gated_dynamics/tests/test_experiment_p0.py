from __future__ import annotations

import json
from pathlib import Path

import experiments.exp07_mechanism_identifiability as exp07
from experiments.common import load_json_config
from src.baselines.tuned_recurrent import RefitFailedError


PROJECT = Path(__file__).resolve().parents[1]


def test_exp07_formal_contract_has_thirty_independent_seeds() -> None:
    config = load_json_config("configs/formal/exp07_mechanism_identifiability.json")
    assert config["seeds"] == list(range(30))
    assert config["budget_norms"] == ["l1", "l2"]
    assert config["baseline"]["hidden_sizes"]
    assert len(config["baseline"]["learning_rates"]) >= 2
    assert config["task"]["n_trials"] == 160
    assert config["test_fraction"] == 0.25
    assert config["validation_fraction"] == 1 / 3


def test_exp07_smoke_writes_complete_paired_grid(tmp_path: Path) -> None:
    config = load_json_config("configs/smoke/exp07_mechanism_identifiability.json")
    path = exp07.run_seed(config, 0, str(tmp_path))
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 34
    assert {record["condition"] for record in records} >= {
        "task-homeostasis-normalization__aligned__l1",
        "task-homeostasis-normalization__shuffled__l2",
        "task-homeostasis__aligned__l1",
        "task-normalization__aligned__l1",
        "homeostasis-normalization__aligned__l1",
        "frozen-recurrent__aligned__l1",
        "tuned-bptt",
        "tuned-gru",
    }
    local = [record for record in records if record["condition"].startswith("task-")]
    assert local
    assert {record["status"] for record in records} == {"complete"}
    assert all(
        record["budget_match_valid"] is True
        for record in records
        if "budget_match_valid" in record
    )
    for field in (
        "initialization_id",
        "readout_training_id",
        "gate_id",
        "trial_order_id",
        "noise_id",
        "homeostasis_signal_id",
        "feedback_encoder_id",
    ):
        assert len({record[field] for record in local}) == 1
    assert {record["actual_outer_test_fraction"] for record in records} == {
        config["test_fraction"]
    }
    assert {record["actual_inner_validation_fraction"] for record in records} == {
        config["validation_fraction"]
    }
    baselines = [
        record for record in records if record["condition"].startswith("tuned-")
    ]
    assert all(record["test_data_used_for_selection"] is False for record in baselines)
    assert all(record["tuning_audit"]["candidate_audits"] for record in baselines)


def test_exp07_retains_budget_shortfall_and_condition_exception_dimensions(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_json_config("configs/smoke/exp07_mechanism_identifiability.json")
    config = json.loads(json.dumps(config))
    config["baseline"].update(
        {
            "hidden_sizes": [4],
            "learning_rates": [0.001],
            "rate_leaks": [0.5],
            "max_epochs": 1,
            "patience": 1,
        }
    )
    original = exp07.run_mechanism_condition
    original_baseline = exp07._baseline_metrics

    class FakeRefitAudit:
        error = "intentional refit failure"

        @staticmethod
        def to_dict() -> dict[str, object]:
            return {
                "status": "failed",
                "error_type": "RuntimeError",
                "error": "intentional refit failure",
                "test_data_used_for_refit": False,
            }

    def injected_failure(train, test, resources, condition, training):
        if condition.name == "task-only__random__l1":
            raise RuntimeError("intentional condition failure")
        result = original(train, test, resources, condition, training)
        if condition.name == "task-only__aligned__l1":
            result.metrics.update(
                status="invalid",
                budget_match_valid=False,
                failure_reason="intentional_update_budget_shortfall",
            )
        return result

    def injected_baseline(*args, **kwargs):
        if kwargs["cell_type"] == "rate_rnn":
            raise RefitFailedError(FakeRefitAudit())
        return original_baseline(*args, **kwargs)

    monkeypatch.setattr(exp07, "run_mechanism_condition", injected_failure)
    monkeypatch.setattr(exp07, "_baseline_metrics", injected_baseline)
    path = exp07.run_seed(config, 0, str(tmp_path))
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lookup = {record["condition"]: record for record in records}
    shortfall = lookup["task-only__aligned__l1"]
    exception = lookup["task-only__random__l1"]
    refit_failure = lookup["tuned-bptt"]
    assert shortfall["status"] == "failed"
    assert shortfall["budget_match_valid"] is False
    assert shortfall["task_budget_total_budget"] > 0.0
    assert shortfall["failure_reason"] == "intentional_update_budget_shortfall"
    assert exception["status"] == "failed"
    assert exception["error"] == "intentional condition failure"
    assert refit_failure["status"] == "failed"
    assert refit_failure["error_type"] == "RefitFailedError"
    assert refit_failure["baseline_audit"] == FakeRefitAudit.to_dict()
    assert refit_failure["test_data_used_for_selection"] is False
    for record in (shortfall, exception):
        assert record["mechanism"] == "task-only"
        assert record["budget_norm"] == "l1"
        assert record["task_plasticity_enabled"] is True
        assert record["homeostasis_enabled"] is False
        assert record["normalization_enabled"] is False
    planned = json.loads((path / "planned_conditions.json").read_text(encoding="utf-8"))
    planned_lookup = {record["condition"]: record for record in planned}
    assert planned_lookup["task-only__random__l1"]["feedback_mode"] == "random"
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete_with_failures"
    assert status["condition_failures"] == 3
