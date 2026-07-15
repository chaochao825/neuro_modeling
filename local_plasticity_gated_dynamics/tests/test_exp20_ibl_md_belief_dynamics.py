from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp14_ibl_multisession_neural import _synthetic_prepared_sessions
from experiments.exp20_ibl_md_belief_dynamics import (
    MODEL_CONDITIONS,
    evaluate_prepared_sessions,
    run_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(name: str = "smoke") -> dict[str, object]:
    path = PROJECT_ROOT / "configs" / name / "exp20_ibl_md_belief_dynamics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _small_config() -> dict[str, object]:
    config = _config()
    config.update(latent_dims=[1], ridges=[0.1], n_bootstrap=100)
    return config


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_exp20_smoke_is_exact_paired_predictive_prior_grid(tmp_path: Path) -> None:
    path = run_seed(_small_config(), 4, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"

    records = _records(path)
    outer = [item for item in records if item.get("stage") == "outer_test"]
    assert len(outer) == 6 * len(MODEL_CONDITIONS)
    planned = json.loads((path / "planned_conditions.json").read_text(encoding="utf-8"))
    planned_grid = {(item["session_id"], item["condition"]) for item in planned}
    observed_grid = {(item["session_id"], item["condition"]) for item in outer}
    assert observed_grid == planned_grid
    assert all(item["split_unit"] == "contiguous_true_probabilityLeft_block" for item in outer)
    assert all(not item["gate_received_probability_left"] for item in outer)
    assert all(not item["belief_uses_current_trial_stimulus"] for item in outer)
    assert all(not item["belief_uses_future_trials"] for item in outer)
    assert all(not item["belief_accessed_true_context"] for item in outer)
    assert all(not item["full_latent_lds"] for item in outer)

    by_session: dict[str, list[dict[str, object]]] = {}
    for item in outer:
        by_session.setdefault(str(item["session_id"]), []).append(item)
    intervention_conditions = {
        "md_shared",
        "md_clamp",
        "md_delay_1",
        "md_delay_5",
        "md_shuffle",
    }
    for rows in by_session.values():
        interventions = [
            item for item in rows if item["condition"] in intervention_conditions
        ]
        assert len({item["fit_fingerprint"] for item in interventions}) == 1
        assert len({item["belief_checkpoint_sha256"] for item in interventions}) == 1
        assert all(
            item["all_model_parameters_frozen_for_intervention"]
            for item in interventions
            if item["condition"] != "md_shared"
        )
        assert all(item["evaluated_heldout_belief_sha256"] for item in interventions)

    summary = [item for item in records if item.get("stage") == "cohort_summary"]
    assert len(summary) == 1
    assert summary[0]["core_conclusion"] in {"support", "oppose", "inconclusive"}
    assert not summary[0]["truth_used_by_gate_or_model"]


def test_exp20_true_blocks_only_define_whole_block_boundaries() -> None:
    config = _small_config()
    prepared = _synthetic_prepared_sessions(config, 2)
    result = evaluate_prepared_sessions(prepared, config=config, seed=2)
    for session_id, split in result.splits.items():
        blocks = np.asarray(split.ordered_block_ids)
        boundary = len(split.train_trial_ids)
        assert blocks[boundary - 1] != blocks[boundary]
        assert np.array_equal(
            blocks,
            result.bundles[session_id].true_block_ids[: len(blocks)],
        )
        receipt = next(
            item.belief_receipt
            for item in result.md_sessions
            if item.session_id == session_id
        )
        assert receipt.input_columns == ("stimulus_side_lag1",)
        assert not receipt.accessed_true_context


def test_exp20_formal_missing_cache_fails_closed_without_synthetic_fallback(
    tmp_path: Path,
) -> None:
    config = _config("formal")
    config["compact_cache_manifest"] = str(tmp_path / "missing.csv")
    path = run_seed(config, 0, tmp_path / "results")
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete_with_failures"
    records = _records(path)
    assert len(records) == 1
    assert records[0]["condition"] == "cohort_load"
    assert records[0]["stage"] == "data_loading"
    assert records[0]["status"] == "failed"
