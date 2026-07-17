from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import exp30_associative_actuator_trend as exp30
from experiments.common import load_json_config
from figures.exp30_associative_actuator_trend_plot import plot_exp30
from scripts.summarize_exp30 import summarize_records, validate_panel_contract


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return load_json_config(
        ROOT / "configs" / "smoke" / "exp30_associative_actuator_trend.json"
    )


def _records(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )


def test_exp30_config_is_trend_only_and_formal_scale_is_preregistered() -> None:
    smoke = _config()
    formal = load_json_config(
        ROOT / "configs" / "formal" / "exp30_associative_actuator_trend.json"
    )
    assert smoke["seeds"] == [9100, 9101, 9102, 9103, 9104]
    assert formal["seeds"] == list(range(30))
    assert set(smoke["seeds"]).isdisjoint(formal["seeds"])
    assert smoke["profile"] == "smoke"
    assert formal["profile"] == "formal"
    assert smoke["used_autograd"] is False
    assert smoke["used_bptt"] is False
    assert len(exp30._planned_conditions(smoke)) == 4 * len(exp30.MODES)


def test_one_seed_is_paired_query_rms_matched_and_shows_registered_reversal(
    tmp_path: Path,
) -> None:
    path = exp30.run_seed(_config(), 9100, tmp_path, run_label="pytest")
    records = _records(path)
    planned = json.loads(
        (path / "planned_conditions.json").read_text(encoding="utf-8")
    )
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    run_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert len(records) == len(planned) == 32
    assert status["status"] == "complete"
    evidence = run_config["evidence_provenance"]
    assert manifest["evidence_provenance"] == evidence
    assert evidence["schema_version"] == exp30.EVIDENCE_SCHEMA_VERSION
    assert set(evidence["critical_file_sha256"]) == set(exp30.CRITICAL_CODE_FILES)
    assert all(len(value) == 64 for value in evidence["critical_file_sha256"].values())
    assert len(evidence["source_config_sha256"]) == 64
    assert set(records["status"]) == {"complete"}
    assert records["statistics_unit"].eq("seed").all()
    assert records["split_unit"].eq("block").all()
    assert (~records["time_points_randomly_split"]).all()
    assert records["fixed_high_rank_carrier"].all()
    assert records["carrier_shared_across_modes_and_demands"].all()
    assert records["motif_dictionary_shared_across_demands"].all()
    assert records["readout_shared_across_modes"].all()
    assert records["control_gain_train_fitted_per_mode_and_demand"].all()
    assert records["functional_budget_scope"].eq("query_output_rms_only").all()
    assert (~records["write_energy_budget_matched_across_all_modes"]).all()
    assert records["carrier_bridge_is_identity_calibrated"].all()
    assert (~records["carrier_dynamics_contribute_to_task_solution"]).all()
    assert records["combined_composes_actuator_outputs"].all()
    assert (~records["combined_oracle_target_access"]).all()
    assert records["functional_budget_valid"].all()
    assert records["associative_shuffled_write_budget_equal"].all()
    assert records["carrier_fingerprint"].nunique() == 1
    assert records["train_split_fingerprint"].nunique() == 1
    assert records["test_split_fingerprint"].nunique() == 1
    pivot = records.pivot(
        index="memory_demand",
        columns="actuator_mode",
        values="test_normalized_score",
    )
    assert pivot.loc[0.0, "routing"] - pivot.loc[0.0, "associative"] > 1.0
    assert pivot.loc[1.0, "associative"] - pivot.loc[1.0, "routing"] > 1.0
    assert pivot.loc[1.0, "associative"] - pivot.loc[1.0, "associative_shuffled"] > 1.0
    assert float(pivot["matched"].mean() - pivot["fixed_best"].mean()) > 0.3
    advantage = pivot["associative"] - pivot["routing"]
    assert float(pd.Series(advantage.index).corr(advantage, method="spearman")) > 0.9
    assert records["carrier_rank"].eq(64).all()
    assert np.max(records["carrier_bridge_reconstruction_error"]) < 1e-12

    replicated = pd.concat(
        [records.assign(seed=seed) for seed in range(5)], ignore_index=True
    )
    conditions, seeds, summary = summarize_records(replicated)
    assert summary["trend_classification"] == "trend-positive"
    assert summary["claim_classification"] == "inconclusive"
    assert summary["positive_seed_fraction"] == 1.0
    assert len(seeds) == 5
    output = tmp_path / "figure"
    output.mkdir()
    conditions.to_csv(output / "conditions.csv", index=False)
    seeds.to_csv(output / "seed_summary.csv", index=False)
    png, pdf = plot_exp30(output)
    assert png.stat().st_size > 10_000
    assert pdf.stat().st_size > 1_000

    publishable = pd.concat(
        [
            records.assign(seed=seed, run_git_dirty=False)
            for seed in _config()["seeds"]
        ],
        ignore_index=True,
    )
    validate_panel_contract(publishable, _config())
    with pytest.raises(RuntimeError, match="missing or adds"):
        validate_panel_contract(publishable.iloc[:-1].copy(), _config())
    dirty = publishable.copy()
    dirty.loc[0, "run_git_dirty"] = True
    with pytest.raises(RuntimeError, match="clean Git"):
        validate_panel_contract(dirty, _config())
    mixed_commit = publishable.copy()
    mixed_commit.loc[0, "run_git_commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="run_git_commit"):
        validate_panel_contract(mixed_commit, _config())
