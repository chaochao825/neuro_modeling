from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import exp32_persistent_sparse_feedback as exp32
from experiments.common import load_json_config
from figures.exp32_persistent_sparse_feedback_plot import plot_exp32
from scripts import summarize_exp32 as exp32_summary
from scripts.summarize_exp32 import (
    load_panel,
    package_run_receipts,
    summarize_records,
    validate_panel_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _tiny_config() -> dict[str, object]:
    config = load_json_config(
        ROOT / "configs" / "smoke" / "exp32_persistent_sparse_feedback.json"
    )
    config["seeds"] = [9300]
    config["task"] = {
        "n_trials": 128,
        "key_dim": 4,
        "direct_reliabilities": [0.6, 0.9],
        "load_values": [2, 4],
        "distractor_write_values": [0, 2],
        "distractor_strength": 1.0,
        "hazards": [0.05],
        "feedback_fractions": [0.125],
        "feedback_delays": [4],
    }
    return config


def _records(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )


def test_exp32_configs_freeze_smoke_before_formal_scale() -> None:
    smoke = load_json_config(
        ROOT / "configs" / "smoke" / "exp32_persistent_sparse_feedback.json"
    )
    formal = load_json_config(
        ROOT / "configs" / "formal" / "exp32_persistent_sparse_feedback.json"
    )
    assert smoke["seeds"] == [9300, 9301, 9302, 9303, 9304]
    assert formal["seeds"] == list(range(30))
    assert formal["formal_authorized"] is False
    assert set(smoke["seeds"]).isdisjoint(formal["seeds"])
    assert smoke["task"] == formal["task"]
    assert smoke["selector"] == formal["selector"]
    assert smoke["used_autograd"] is False
    assert smoke["used_bptt"] is False


def test_exp32_boundary_config_uses_new_seeds_without_retuning() -> None:
    smoke = load_json_config(
        ROOT / "configs" / "smoke" / "exp32_persistent_sparse_feedback.json"
    )
    boundary = load_json_config(
        ROOT / "configs" / "formal" / "exp32_evidence_per_dwell_boundary.json"
    )
    assert boundary["formal_authorized"] is True
    assert set(boundary["seeds"]).isdisjoint(smoke["seeds"])
    assert boundary["selector"] == smoke["selector"]
    for field in (
        "key_dim",
        "direct_reliabilities",
        "load_values",
        "distractor_write_values",
        "distractor_strength",
    ):
        assert boundary["task"][field] == smoke["task"][field]
    assert boundary["task"]["n_trials"] == 4096
    assert boundary["analysis"]["primary_hazard"] == 0.01
    assert boundary["analysis"]["claim_family"] == "feedback_memory_timescale_phase"
    exp32._validate_config(boundary)


def test_exp32_critical_hashes_cover_runtime_and_formal_inference_helpers() -> None:
    assert {
        "experiments/common.py",
        "src/analysis/hidden_selector_metrics.py",
        "src/utils/artifacts.py",
        "src/utils/reproducibility.py",
    } <= set(exp32.CRITICAL_CODE_FILES)


def test_formal_payload_digest_rejects_task_selector_and_analysis_mutation() -> None:
    original = load_json_config(
        ROOT / "configs" / "formal" / "exp32_evidence_per_dwell_boundary.json"
    )
    mutations = (
        ("selector", "alpha", 0.99),
        ("task", "n_trials", 8192),
        ("analysis", "primary_mcid", 0.0),
        (
            "analysis",
            "holm_family",
            [
                "evidence_response_slope",
                "persistent_over_opposite_eligibility",
                "persistent_over_train_fixed",
            ],
        ),
    )
    original_digest = exp32._canonical_formal_payload_digest(original)
    assert original_digest == original["authorized_formal_payload_sha256"]
    for section, field, value in mutations:
        changed = copy.deepcopy(original)
        changed[section][field] = value
        assert exp32._canonical_formal_payload_digest(changed) != original_digest
        error_pattern = "Holm family" if field == "holm_family" else "payload"
        with pytest.raises(ValueError, match=error_pattern):
            exp32._validate_config(changed)


def test_formal_checkout_binding_is_strict_but_archive_policy_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_commit = "a" * 40
    run_tree = "b" * 40
    raw = pd.DataFrame(
        {
            "run_git_commit": [run_commit],
            "run_git_tree": [run_tree],
            "run_git_dirty": [False],
        }
    )
    provenance = {"git": {"commit": run_commit, "tree": run_tree, "dirty": False}}
    monkeypatch.setattr(
        exp32_summary,
        "_git_checkout_identity",
        lambda: {"commit": "c" * 40, "tree": "d" * 40, "dirty": False},
    )
    with pytest.raises(RuntimeError, match="current checkout commit/tree mismatch"):
        exp32_summary._validate_checkout_binding(
            raw, provenance, profile="formal", checkout_policy="live"
        )
    archived = exp32_summary._validate_checkout_binding(
        raw, provenance, profile="formal", checkout_policy="archived"
    )
    assert archived["archived_reanalysis"] is True
    assert archived["current_checkout_matches_run"] is False
    assert archived["current_checkout_reproducibility_claimed"] is False

    changed = raw.copy()
    changed["run_git_tree"] = "e" * 40
    with pytest.raises(RuntimeError, match="disagrees with source provenance"):
        exp32_summary._validate_checkout_binding(
            changed, provenance, profile="formal", checkout_policy="archived"
        )


def test_unauthorized_formal_exp32_fails_closed() -> None:
    formal = load_json_config(
        ROOT / "configs" / "formal" / "exp32_persistent_sparse_feedback.json"
    )
    try:
        exp32._validate_config(formal)
    except ValueError as error:
        assert "fail-closed" in str(error)
    else:
        raise AssertionError("unauthorized Exp32 formal config must fail closed")


def test_one_exp32_seed_is_paired_reward_only_and_failure_preserving(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    path = exp32.run_seed(config, 9300, tmp_path, run_label="pytest")
    records = _records(path)
    planned = json.loads((path / "planned_conditions.json").read_text("utf-8"))
    assert len(records) == len(planned) == len(exp32.MODES)
    assert records["status"].eq("complete").all()
    assert records["statistics_unit"].eq("seed").all()
    assert records["split_unit"].eq("whole_independent_stream").all()
    assert (~records["time_points_randomly_split"]).all()
    assert (~records["controller_reset_at_switch"]).all()
    assert (~records["switch_times_exposed_to_selector"]).all()
    assert (~records["task_target_is_explicit_actuator_mixture"]).all()
    assert (~records["high_rank_carrier_present"]).all()
    assert records["exploration_and_switch_cost_in_primary"].all()
    local = records[records["selector_mode"] == "persistent_rpe_local"].iloc[0]
    oracle = records[records["selector_mode"] == "oracle_hidden_state"].iloc[0]
    assert local["selector_received_executed_scalar_reward"]
    assert not local["selector_received_true_context"]
    assert not local["selector_received_unexecuted_reward"]
    assert local["reward_only_interface_audit"]
    assert local["selector_internal_state_dimension"] == 2
    assert local["selector_control_dimension"] == 1
    assert local["belief_metric_semantics"] == "action_one_policy_probability_proxy"
    assert local["delayed_rpe_reference"].startswith("delivery_time")
    opposite = records[records["selector_mode"] == "credit_shuffled_local"].iloc[0]
    assert opposite["credit_intervention_semantics"] == "opposite_action_eligibility"
    assert oracle["selector_received_true_context"]
    assert (
        records.groupby(["hazard", "feedback_fraction", "feedback_delay"])[
            "test_pool_fingerprint"
        ]
        .nunique()
        .max()
        == 1
    )
    assert records["feedback_pending_count"].eq(0).all()

    conditions, seeds, summary = summarize_records(records, config)
    assert not conditions.empty
    assert len(seeds) == 1
    assert summary["claim_classification"] == "inconclusive"
    output = tmp_path / "figure"
    output.mkdir()
    records.to_csv(output / "raw_metrics.csv.gz", index=False, compression="gzip")
    seeds.to_csv(output / "seed_summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    png, pdf = plot_exp32(output)
    assert png.stat().st_size > 10_000
    assert pdf.stat().st_size > 1_000


def test_boundary_summary_pairs_slow_and_fast_hazards(tmp_path: Path) -> None:
    config = _tiny_config()
    config["task"]["hazards"] = [0.01, 0.05]
    config["analysis"].update(
        {
            "claim_family": "evidence_per_dwell_boundary",
            "primary_hazard": 0.01,
            "boundary_reference_hazard": 0.05,
            "boundary_interaction_mcid": 0.03,
        }
    )
    path = exp32.run_seed(config, 9300, tmp_path, run_label="pytest-boundary")
    records = _records(path)
    conditions, seeds, summary = summarize_records(records, config)
    assert set(conditions["expected_feedback_per_dwell"]) == {2.5, 12.5}
    assert "reference_persistent_minus_fixed" in seeds
    assert "evidence_per_dwell_boundary_interaction" in seeds
    assert summary["claim_family"] == "evidence_per_dwell_boundary"
    assert summary["primary_expected_feedback_per_dwell"] == 12.5
    assert summary["boundary_reference_expected_feedback_per_dwell"] == 2.5


def test_pairing_contract_detects_cross_fraction_and_hazard_mutations(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    config["task"]["hazards"] = [0.01, 0.05]
    config["task"]["feedback_fractions"] = [0.5, 0.125]
    config["task"]["feedback_delays"] = [0, 4]
    path = exp32.run_seed(config, 9300, tmp_path, run_label="pytest-pairing")
    records = _records(path)
    validate_panel_contract(records, config)

    changed = records.copy()
    mask = changed["feedback_fraction"].eq(0.125)
    changed.loc[mask, "train_pool_fingerprint"] = "f" * 64
    with pytest.raises(RuntimeError, match="seed-level paired tape"):
        validate_panel_contract(changed, config)

    changed = records.copy()
    mask = changed["feedback_fraction"].eq(0.125) & changed["hazard"].eq(0.01)
    changed.loc[mask, "state_tape_fingerprint"] = "e" * 64
    with pytest.raises(RuntimeError, match="hazard-level paired tape"):
        validate_panel_contract(changed, config)

    changed = records.copy()
    mask = changed["hazard"].eq(0.05) & changed["feedback_fraction"].eq(0.125)
    changed.loc[mask, "feedback_tape_fingerprint"] = "d" * 64
    with pytest.raises(RuntimeError, match="feedback tape changed"):
        validate_panel_contract(changed, config)


def test_audit_failure_cannot_produce_directional_formal_conclusion(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    path = exp32.run_seed(config, 9300, tmp_path, run_label="pytest-invalid-audit")
    base = _records(path)
    copies = []
    for seed in range(30):
        frame = base.copy()
        frame["seed"] = seed
        frame.loc[
            frame["selector_mode"].eq("train_fixed_best"), "full_stream_accuracy"
        ] = 0.8
        frame.loc[
            frame["selector_mode"].eq("persistent_rpe_local"), "full_stream_accuracy"
        ] = 0.4
        frame.loc[
            frame["selector_mode"].eq("credit_shuffled_local"), "full_stream_accuracy"
        ] = 0.7
        copies.append(frame)
    records = pd.concat(copies, ignore_index=True)
    records.loc[
        records["selector_mode"].eq("persistent_rpe_local"),
        "selector_received_true_context",
    ] = True
    config["profile"] = "formal"
    config["seeds"] = list(range(30))
    config["analysis"].update(
        {
            "bootstrap_samples": 1000,
            "permutation_samples": 1000,
            "statistics_seed": 999,
            "holm_family": list(exp32._expected_holm_family("original_primary")),
        }
    )
    _, _, summary = summarize_records(records, config)
    assert summary["access_and_pairing_audit_passed"] is False
    assert summary["claim_classification"] == "inconclusive"


def test_seed_setup_failure_is_retained_for_every_planned_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config()

    def fail_setup(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic setup failure")

    monkeypatch.setattr(exp32, "_potential_outcomes", fail_setup)
    path = exp32.run_seed(config, 9300, tmp_path, run_label="pytest-setup-fail")
    records = _records(path)
    planned = json.loads((path / "planned_conditions.json").read_text("utf-8"))
    status = json.loads((path / "status.json").read_text("utf-8"))
    assert len(records) == len(planned) == len(exp32.MODES)
    assert records["status"].eq("failed").all()
    assert status["status"] == "complete_with_failures"

    raw, manifest = load_panel(tmp_path, config, run_label="pytest-setup-fail")
    conditions, seeds, summary = summarize_records(raw, config)
    assert len(conditions) == len(exp32.MODES)
    assert conditions["status"].eq("failed").all()
    assert seeds.empty
    assert summary["panel_complete"] is False
    assert summary["access_and_pairing_audit_passed"] is False
    assert summary["claim_classification"] == "inconclusive"
    assert summary["scale_decision"] == "scale-not-authorized"
    assert summary["n_failed_rows"] == len(planned)

    output = tmp_path / "packaged"
    output.mkdir()
    receipts = package_run_receipts(manifest, output)
    relative_paths = receipts["relative_path"].astype(str)
    assert relative_paths.str.contains("\\\\").sum() == 0
    assert relative_paths.str.endswith("run_log.txt").sum() == 1
    assert not relative_paths.str.endswith("run.log").any()
    assert all((output / relative).is_file() for relative in relative_paths)


def test_phase_summary_uses_seed_level_evidence_and_iso_lambda_claims() -> None:
    config = load_json_config(
        ROOT / "configs" / "formal" / "exp32_evidence_per_dwell_boundary.json"
    )
    config["analysis"]["bootstrap_samples"] = 2000
    config["analysis"]["permutation_samples"] = 2000
    tau_q = -1.0 / np.log(config["selector"]["retention"])
    rows = []
    adaptive = {
        "cumulative_sample_average",
        "persistent_rpe_local",
        "credit_shuffled_local",
        "bayes_reward_filter",
    }
    for seed in config["seeds"]:
        for hazard in config["task"]["hazards"]:
            for fraction in config["task"]["feedback_fractions"]:
                for delay in config["task"]["feedback_delays"]:
                    log_lambda = np.log2(fraction / hazard)
                    log_chi = np.log2(hazard * tau_q)
                    kappa = (1.0 - 2.0 * hazard) ** (delay + 1)
                    gain = (
                        0.04
                        + 0.01 * log_lambda
                        - 0.03 * log_chi
                        - 0.005 * (-np.log2(kappa))
                    )
                    accuracies = {
                        "train_fixed_best": 0.70,
                        "matched_random": 0.60,
                        "cumulative_sample_average": 0.70 + 0.5 * gain,
                        "persistent_rpe_local": 0.70 + gain,
                        "credit_shuffled_local": 0.66 + gain,
                        "no_feedback_local": 0.60,
                        "bayes_reward_filter": 0.70 + 0.8 * gain,
                        "oracle_hidden_state": 0.95,
                    }
                    for mode in exp32.MODES:
                        rows.append(
                            {
                                "seed": seed,
                                "status": "complete",
                                "hazard": hazard,
                                "feedback_fraction": fraction,
                                "feedback_delay": delay,
                                "selector_mode": mode,
                                "full_stream_accuracy": accuracies[mode],
                                "dynamic_regret_to_hidden_oracle": 0.95
                                - accuracies[mode],
                                "median_switch_latency": 2.0,
                                "false_switch_rate": 0.01,
                                "context_nll": 0.6,
                                "context_brier": 0.2,
                                "reward_only_interface_audit": True,
                                "selector_received_true_context": mode
                                == "oracle_hidden_state",
                                "selector_received_unexecuted_reward": False,
                                "both_actuator_winners_present": True,
                                "feedback_pending_count": 0,
                                "feedback_available_count": 16,
                                "feedback_delivered_count": 16
                                if mode in adaptive
                                else 0,
                                "time_points_randomly_split": False,
                                "controller_reset_at_switch": False,
                                "switch_times_exposed_to_selector": False,
                                "feedback_schedule_nested_audit": True,
                                "no_feedback_matches_random_action_tape": True,
                                "selector_update_l1": (
                                    1.2
                                    if mode == "credit_shuffled_local"
                                    else 1.0
                                    if mode == "persistent_rpe_local"
                                    else 0.0
                                ),
                            }
                        )
    raw = pd.DataFrame(rows)
    _, seeds, summary = summarize_records(raw, config)
    assert len(seeds) == 30
    assert summary["main_controller_claim_classification"] == "support"
    assert summary["timescale_structure_claim_classification"] == "support"
    assert summary["claim_classification"] == "support"
    assert summary["mean_evidence_response_slope"] == pytest.approx(0.01)
    assert summary["mean_iso_lambda_slow_minus_fast"] > 0.06

    changed = copy.deepcopy(config)
    changed["analysis"]["holm_family"] = list(
        reversed(changed["analysis"]["holm_family"])
    )
    with pytest.raises(RuntimeError, match="configured Holm family"):
        summarize_records(raw, changed)
