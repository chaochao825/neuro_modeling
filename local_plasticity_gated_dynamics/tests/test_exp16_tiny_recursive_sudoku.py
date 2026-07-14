from __future__ import annotations

import json

from experiments.common import load_json_config
from experiments.exp16_tiny_recursive_sudoku import CONDITIONS, run_seed
from figures.exp16_tiny_recursive_plot import plot_exp16
from scripts.summarize_exp16_tiny_recursive import publish_snapshot


def _rows(path):
    return [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_exp16_smoke_retains_matched_baseline_receipts_and_test_safety(tmp_path) -> None:
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
    )
    config["training"].update(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        auxiliary_loss_weight=0.0,
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
    flat = aggregates["flat_shared_compute_matched"]
    assert recursive["parameter_count"] == flat["parameter_count"]
    assert recursive["core_calls_per_forward"] == flat["core_calls_per_forward"]
    assert recursive["optimizer_steps"] == flat["optimizer_steps"]
    assert recursive["initialization_sha256"] == flat["initialization_sha256"]
    assert recursive["used_bptt"] is True
    assert recursive["eligible_for_local_initialization"] is False
    assert recursive["claim_conclusion"] == "inconclusive"

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
    receipts = json.loads(
        (run_path / "fit_receipts.json").read_text(encoding="utf-8")
    )
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
