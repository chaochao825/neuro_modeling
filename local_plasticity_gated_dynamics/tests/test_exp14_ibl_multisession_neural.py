from __future__ import annotations

import json
from pathlib import Path

from experiments.exp14_ibl_multisession_neural import (
    _synthetic_prepared_sessions,
    panel_claim_scope,
    run_seed,
)


def _config(profile: str = "smoke") -> dict[str, object]:
    return {
        "profile": profile,
        "data_mode": "synthetic_smoke"
        if profile == "smoke"
        else "frozen_compact_cache",
        "views": ["stimulus_pre"],
        "panels": ["primary_past_safe"],
        "synthetic_sessions": 6,
        "synthetic_trials": 80,
        "synthetic_time_bins": 5,
        "minimum_trials": 50,
        "minimum_blocks": 5,
        "planned_sessions": 6,
        "planned_animals": 3,
        "outer_test_fraction": 0.2,
        "inner_validation_fraction": 0.25,
        "latent_dims": [1],
        "ridges": [0.1],
        "min_units_per_region": 2,
        "minimum_region_sessions": 5,
        "max_units_per_region": 2,
        "macro_region_mapping_path": "configs/exp14_allen_macro_region_mapping_v1.json",
        "expected_macro_region_mapping_sha256": "3bac702ed6b3ee5c21acbbfd929b077baa63226369ca8e1bef0b6faeb487fc23",
        "expected_macro_region_mapping_schema": "exp14_allen_macro_region_mapping_v1",
        "expected_macro_region_source_ontology_sha256": "63654b8d35c7c1b5665636b645da774776ee8263658192f5dca1e815095e9147",
        "expected_macro_region_source_provenance_sha256": "a01b7fa535e6de437ac46e8cf9de68a87d6a9b5587d055a3935476d956109fdc",
        "macro_region_mapping_formal_compact_manifest_sha256": "a5acb134ae4b34f47db150948a7f7ab58e8eb85e204fb981e0ca744eba328a09",
        "learned_hmm": {
            "max_iter": 60,
            "n_restarts": 1,
            "min_emission_gap": 0.1,
            "require_converged": False,
            "require_identifiable": False,
        },
        "n_bootstrap": 100,
    }


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_exp14_smoke_runs_nested_paired_count_models(tmp_path: Path) -> None:
    path = run_seed(_config(), 3, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    records = _records(path)
    outer = [record for record in records if record.get("stage") == "outer_test"]
    assert len(outer) == 6 * 3
    assert {record["model_family"] for record in outer} == {
        "common",
        "shared",
        "full",
    }
    assert all(record["preprocessing_fit_train_only"] for record in outer)
    assert all(not record["test_context_observed"] for record in outer)
    assert all(not record["counts_residualized_before_poisson"] for record in outer)
    assert len({record["comparison_preprocessing_sha256"] for record in outer}) == 1
    assert all(
        record["region_anchor_policy"] == "fixed_region_order_union" for record in outer
    )
    assert all(
        record["region_imputation_strategy"] == "pooled_training_fold_region_mean"
        for record in outer
    )
    assert all(record["all_complete_sessions_retained"] for record in outer)
    assert all(record["n_sessions_retained"] == 6 for record in outer)
    assert all(len(record["retained_session_ids"]) == 6 for record in outer)
    assert all(len(record["region_session_coverage"]) == 3 for record in outer)
    assert all(record["n_anchor_regions_missing"] == 0 for record in outer)
    assert all(
        record["macro_region_mapping_schema"] == "exp14_allen_macro_region_mapping_v1"
        for record in outer
    )
    assert all(
        record["macro_region_mapping_sha256"]
        == "3bac702ed6b3ee5c21acbbfd929b077baa63226369ca8e1bef0b6faeb487fc23"
        for record in outer
    )
    assert all(
        not record["macro_region_behavior_or_model_selected"] for record in outer
    )
    planned = json.loads((path / "planned_conditions.json").read_text(encoding="utf-8"))
    planned_grid = {
        (
            row["session_id"],
            row["view"],
            row["panel"],
            row["model_family"],
        )
        for row in planned
    }
    observed_grid = {
        (
            row["session_id"],
            row["view"],
            row["panel"],
            row["model_family"],
        )
        for row in outer
    }
    assert observed_grid == planned_grid
    summary = [
        record
        for record in records
        if record.get("stage") == "animal_session_comparison"
    ]
    assert len(summary) == 1
    assert summary[0]["core_conclusion"] in {"support", "oppose", "inconclusive"}
    assert not summary[0]["full_latent_lds"]
    assert summary[0]["claim_scope"] == "registered_primary"
    assert summary[0]["causal_timing_eligible"]
    assert summary[0]["minimum_region_sessions"] == 5
    assert summary[0]["all_complete_sessions_retained"]


def test_only_stimulus_primary_panel_is_core_claim_eligible() -> None:
    assert (
        panel_claim_scope("stimulus_pre", "primary_past_safe") == "registered_primary"
    )
    assert panel_claim_scope("movement_pre", "primary_past_safe") == "sensitivity_only"
    assert (
        panel_claim_scope("stimulus_pre", "full_trial_sensitivity")
        == "sensitivity_only"
    )


def test_formal_profile_never_falls_back_to_synthetic_data(tmp_path: Path) -> None:
    path = run_seed(_config("formal"), 0, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete_with_failures"
    records = _records(path)
    assert len(records) == 3
    assert all(record["status"] == "failed" for record in records)
    assert all("frozen compact neural cache" in record["error"] for record in records)


def test_formal_profile_rejects_direct_prepared_session_injection(
    tmp_path: Path,
) -> None:
    smoke = _config()
    prepared = _synthetic_prepared_sessions(smoke, 0)
    path = run_seed(_config("formal"), 0, tmp_path, prepared_sessions=prepared)
    records = _records(path)
    assert len(records) == 3
    assert all(record["status"] == "failed" for record in records)
    assert all(
        "rejects direct prepared_sessions" in record["error"] for record in records
    )
