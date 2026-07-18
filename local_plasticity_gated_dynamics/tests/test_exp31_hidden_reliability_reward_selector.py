from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments import exp31_hidden_reliability_reward_selector as exp31
from experiments.common import load_json_config
from figures.exp31_hidden_reliability_reward_selector_plot import plot_exp31
from scripts.summarize_exp31 import summarize_records


ROOT = Path(__file__).resolve().parents[1]


def _tiny_config() -> dict[str, object]:
    config = load_json_config(
        ROOT
        / "configs"
        / "smoke"
        / "exp31_hidden_reliability_reward_selector.json"
    )
    config["seeds"] = [9200]
    config["task"] = {
        "n_train_blocks_per_cell": 2,
        "n_test_blocks_per_cell": 2,
        "trials_per_block": 24,
        "probe_trials": 8,
        "key_dim": 4,
        "load_values": [2, 4],
        "distractor_write_values": [0, 2],
        "direct_reliabilities": [0.6, 0.9],
        "distractor_strength": 1.0,
    }
    return config


def _records(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )


def test_exp31_configs_freeze_reward_only_protocol_before_scale() -> None:
    smoke = load_json_config(
        ROOT
        / "configs"
        / "smoke"
        / "exp31_hidden_reliability_reward_selector.json"
    )
    formal = load_json_config(
        ROOT
        / "configs"
        / "formal"
        / "exp31_hidden_reliability_reward_selector.json"
    )
    assert smoke["seeds"] == [9200, 9201, 9202, 9203, 9204]
    assert formal["seeds"] == list(range(30))
    assert set(smoke["seeds"]).isdisjoint(formal["seeds"])
    assert smoke["task"] == formal["task"]
    assert smoke["selector"] == formal["selector"]
    assert smoke["used_autograd"] is False
    assert smoke["used_bptt"] is False


def test_one_exp31_seed_is_paired_reward_only_and_failure_preserving(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    path = exp31.run_seed(config, 9200, tmp_path, run_label="pytest")
    records = _records(path)
    planned = json.loads(
        (path / "planned_conditions.json").read_text(encoding="utf-8")
    )
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert len(records) == len(planned) == 2 * 2 * 2 * 2 * len(exp31.MODES)
    assert set(records["status"]) == {"complete"}
    assert records["statistics_unit"].eq("seed").all()
    assert records["split_unit"].eq("whole_block").all()
    assert (~records["time_points_randomly_split"]).all()
    assert (~records["task_target_is_explicit_actuator_mixture"]).all()
    assert (~records["mode_or_demand_specific_gain_fitted"]).all()
    assert (~records["high_rank_carrier_present"]).all()
    assert records["probe_cost_in_primary"].all()
    assert records["primary_scope"].eq(
        "full_test_block_including_probe_cost"
    ).all()
    local = records[records["actuator_mode"] == "reward_only_local"]
    oracle = records[records["actuator_mode"] == "oracle_hidden_train_map"]
    assert local["selector_received_executed_scalar_reward"].all()
    assert (~local["selector_received_true_context"]).all()
    assert (~local["selector_received_unexecuted_reward"]).all()
    assert (~local["selector_received_candidate_utility_vector"]).all()
    assert (~local["selector_received_prospective_descriptor"]).all()
    assert local["reward_only_interface_audit"].all()
    assert oracle["selector_received_true_context"].all()
    assert records["associative_query_shuffled_write_budget_exact"].all()
    assert records["functional_output_magnitude_matched"].all()
    assert records.groupby("block_id")["test_block_fingerprint"].nunique().max() == 1
    assert records["train_split_fingerprint"].nunique() == 1

    conditions, seeds, summary = summarize_records(records, config)
    assert not conditions.empty
    assert len(seeds) == 1
    assert summary["claim_classification"] == "inconclusive"
    output = tmp_path / "figure"
    output.mkdir()
    conditions.to_csv(output / "conditions.csv", index=False)
    seeds.to_csv(output / "seed_summary.csv", index=False)
    png, pdf = plot_exp31(output)
    assert png.stat().st_size > 10_000
    assert pdf.stat().st_size > 1_000
