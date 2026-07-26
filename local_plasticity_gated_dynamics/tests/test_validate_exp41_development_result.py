from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.exp41_matched_identifiability import execute
from scripts.validate_exp41_development_result import _strict_json, validate_result


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/development/exp41_matched_identifiability_probe_v1.json"
REGISTERED_RESULT = ROOT / "results/exp41_matched_identifiability_development_v1"
DECISION_PATH = ROOT / "provenance/exp41_development_decision_20260727.json"


def _small_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["stream"] = {
        "block_length": 16,
        "blocks_per_sequence": 4,
        "n_sequences": 1,
        "initial_state_variance": 4.0,
    }
    config["selection"]["fixed_jump_grid"] = {
        "process_variance": [0.02],
        "observation_variance": [0.02],
    }
    config["selection"]["online_em_process_rate_grid"] = [0.1]
    config["selection"]["online_em_observation_rate_grid"] = [0.1]
    config["selection"]["autocovariance_decay_grid"] = [0.99]
    config["selection"]["autocovariance_prior_mass_grid"] = [2.0]
    config["selection"]["total_variance_decay_grid"] = [0.99]
    config["selection"]["total_variance_prior_mass_grid"] = [2.0]
    config["selection"]["total_variance_q_fraction_grid"] = [0.5]
    config["selection"]["imm_switch_grid"] = [0.01]
    config["analysis"]["late_window"] = 4
    config["analysis"]["bootstrap_samples"] = 50
    return config


def test_validator_replays_complete_development_artifact_and_fails_on_tamper(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_small_config()), encoding="utf-8")
    output = tmp_path / "result"
    execute(config_path, output)

    receipt = validate_result(output, require_clean_run_start=False)
    assert receipt["audit_status"] == "passed"
    assert receipt["claim_eligible"] is False
    assert receipt["budget_matched"] is False
    assert receipt["n_seeds"] == 8
    assert receipt["artifact_hash_check"] == "passed"
    assert receipt["source_commit_check"] == "passed"
    assert receipt["selection_argmin_replay"] == "passed"

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["verdict"] = "support"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["summary.json"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="claim boundary"):
        validate_result(output, require_clean_run_start=False)


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"status": "complete", "status": "failed"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        _strict_json(path)


def test_registered_development_result_and_stop_decision_are_hash_bound() -> None:
    receipt = validate_result(REGISTERED_RESULT)
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))

    assert receipt["audit_status"] == "passed"
    assert receipt["run_start_clean"] is True
    assert receipt["run_start_commit"] == decision["run_start_commit"]
    assert receipt["n_seeds"] == 8
    assert receipt["verdict"] == "inconclusive"
    for relative, expected in decision["result_hashes"].items():
        observed = hashlib.sha256((REGISTERED_RESULT / relative).read_bytes()).hexdigest()
        assert observed == expected
    replay = decision["replay_receipt"]
    replay_path = ROOT / replay["path"]
    assert hashlib.sha256(replay_path.read_bytes()).hexdigest() == replay["sha256"]
    assert decision["evidence_scope"] == "outcome_exposed_development_only"
    assert decision["claim_eligible"] is False
    assert decision["execution"]["reserved_formal_seeds_accessed"] is False
    assert decision["execution"]["budget_matched"] is False
    assert decision["stop_decision"] == {
        "advance_exp41_to_formal": False,
        "execute_exp42_under_current_entry_gate": False,
        "retune_on_development_seeds": False,
        "reason": (
            "The deployable autocovariance controller separates matched Q/R "
            "regimes but does not beat the reduced total-variance control, is "
            "worse than current online-EM, and has transition-window harm."
        ),
    }
    endpoints = decision["descriptive_endpoints"][
        "nll_gain_baseline_minus_autocov"
    ]
    assert endpoints["h_plus_total_variance"]["mean"] == pytest.approx(
        -0.010722277767784522
    )
    assert endpoints["current_online_em"]["positive_seeds"] == 0
    classifications = {
        row["claim_id"]: row["conclusion"]
        for row in decision["claim_classification"]
    }
    assert classifications == {
        "fast_post_transition_adaptation": "oppose",
        "late_regime_adaptation": "inconclusive",
        "matched_qr_statistical_discrimination": "support",
        "predictive_utility_beyond_current_online_em": "oppose",
        "predictive_utility_beyond_total_variance": "oppose",
        "real_behavior_or_neural_utility": "inconclusive",
    }
