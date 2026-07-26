from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import exp41_matched_identifiability as exp41
from experiments.exp41_matched_identifiability import (
    METHODS,
    execute,
    run_seed,
    summarize,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/development/exp41_matched_identifiability_probe_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _small_config() -> dict:
    config = _config()
    config["stream"] = {
        "block_length": 16,
        "blocks_per_sequence": 4,
        "n_sequences": 2,
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
    config["analysis"]["bootstrap_samples"] = 100
    return config


def test_registered_config_is_strictly_development_only() -> None:
    config = _config()
    validate_config(config)
    assert config["profile"] == "development_matched_identifiability_probe"
    assert config["claim_upgrade_allowed"] is False
    assert config["seeds"] == list(range(41000, 41008))
    assert config["generator_hazard"] == 0.0
    assert config["filter_hazard_floor"] > 0.0
    assert config["used_autograd"] is False
    assert config["used_bptt"] is False
    fractions = config["selection"]["total_variance_q_fraction_grid"]
    assert any(value == pytest.approx(1.0 / 24.0) for value in fractions)
    assert any(value == pytest.approx(1.0 / 12.0) for value in fractions)
    assert any(value == pytest.approx(2.0 / 3.0) for value in fractions)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("profile", "formal", "development-only"),
        ("protocol_version", "changed", "protocol_version"),
        ("claim_upgrade_allowed", True, "claim upgrade"),
        ("seeds", [41000], "development seeds"),
        ("generator_hazard", 0.01, "H=0"),
        ("used_autograd", True, "autograd"),
    ),
)
def test_config_rejects_claim_upgrade_formal_or_semantic_drift(
    field: str, value: object, message: str
) -> None:
    config = _config()
    config[field] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        (None, "seeds", [41000.0, *range(41001, 41008)]),
        ("stream", "block_length", 16.5),
        ("stream", "n_sequences", True),
        ("analysis", "transition_windows", [1.0, 4, 8, 16]),
        ("analysis", "late_window", 4.5),
        ("analysis", "bootstrap_samples", True),
        ("analysis", "statistics_seed", 4.5),
    ),
)
def test_config_rejects_boolean_or_noninteger_integer_fields(
    section: str | None, field: str, value: object
) -> None:
    config = _config()
    target = config if section is None else config[section]
    target[field] = value
    with pytest.raises(ValueError, match="integer|seed"):
        validate_config(config)


def test_run_seed_uses_independent_fit_test_tapes_and_complete_pairing() -> None:
    config = _small_config()
    blocks, audit, metadata = run_seed(config, 41000)

    assert set(blocks["method"]) == set(METHODS)
    assert blocks.groupby("method").size().nunique() == 1
    assert blocks["test_tape_digest"].nunique() == 1
    assert metadata["fit_tape_digest"] != metadata["test_tape_digest"]
    assert audit["data_split"].eq("fit").all()
    assert audit["fit_tape_digest"].nunique() == 1
    assert audit["fit_tape_digest"].iat[0] == metadata["fit_tape_digest"]
    privileged = audit.loc[
        audit["selection_family"].isin(
            ("generator_supported_seen_regime_imm", "dynamic_qr_oracle")
        )
    ]
    deployable = audit.loc[~audit.index.isin(privileged.index)]
    assert privileged["uses_true_parameters"].all()
    assert not deployable["uses_true_parameters"].any()
    assert metadata["all_methods_share_test_tape"] is True
    assert metadata["generator_hazard"] == 0.0
    assert metadata["filter_hazard_floor"] == config["filter_hazard_floor"]

    first_blocks = blocks["block_within_sequence"].eq(0)
    assert not blocks.loc[first_blocks, "transition_eligible"].any()
    for window in (1, 4, 8, 16):
        endpoint = f"transition_nll_{window}"
        assert blocks.loc[first_blocks, endpoint].isna().all()
        assert np.all(np.isfinite(blocks.loc[~first_blocks, endpoint]))
    assert np.all(np.isfinite(blocks["late_nll"]))
    assert np.all(np.isfinite(blocks["q_absolute_log_error"]))
    assert np.all(np.isfinite(blocks["r_absolute_log_error"]))
    assert np.all(blocks["invalid_rows"].eq(0))
    assert {
        "mean_direct_gamma0_estimate",
        "mean_direct_gamma1_estimate",
        "direct_gamma0_absolute_error",
        "direct_gamma1_absolute_error",
        "q_clipping_fraction",
        "r_clipping_fraction",
        "parameter_saturation_fraction",
        "parameter_update_l1",
        "parameter_update_l2",
        "parameter_update_count",
    } <= set(blocks.columns)


def test_summary_uses_seed_unit_and_limits_matched_pair_claim() -> None:
    config = _small_config()
    panels = [run_seed(config, seed)[0] for seed in (41000, 41001)]
    seed_metrics, comparisons, separation, summary = summarize(
        pd.concat(panels, ignore_index=True), config=config
    )

    assert seed_metrics["seed"].nunique() == 2
    assert summary["statistics_unit"] == "seed"
    assert summary["claim_eligible"] is False
    assert summary["verdict"] == "inconclusive"
    assert summary["cross_loading_diagonal_dominance_claimed"] is False
    assert summary["budget_matched"] is False
    assert summary["development_go_gate_satisfied"] is False
    assert summary["tied_qr_executed_separately"] is False
    assert "same one-scalar" in summary["tied_qr_equivalence_note"]
    assert "tied_qr" not in set(seed_metrics["method"])
    assert set(separation["method"]) == set(METHODS)
    assert set(summary["matched_pair_separation_by_method"]) == set(METHODS)
    assert summary["matched_pair_go_diagnostic_method"] == "autocov_factorized"
    assert summary["method_roles"]["generator_supported_seen_regime_imm"].startswith(
        "privileged"
    )
    assert {
        "late_nll",
        "mean_q_absolute_log_error",
        "mean_r_absolute_log_error",
        "parameter_update_l1",
        "parameter_update_l2",
        "parameter_update_count",
    } <= set(seed_metrics.columns)
    assert set(separation["pair_id"]) == {"m06", "m12"}
    assert {
        "q_estimate_q_dominant_minus_r_dominant",
        "r_estimate_r_dominant_minus_q_dominant",
    } <= set(separation.columns)
    assert set(comparisons["baseline"]) == set(METHODS) - {"autocov_factorized"}


def test_execute_writes_complete_atomic_development_artifact(tmp_path: Path) -> None:
    config = _small_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "result"

    summary = execute(config_path, output)

    assert summary["claim_eligible"] is False
    required = {
        "config.json",
        "environment.json",
        "planned_conditions.json",
        "block_metrics.csv",
        "selection_audit.csv",
        "seed_metrics.csv",
        "comparisons.csv",
        "matched_pair_separation.csv",
        "summary.json",
        "report.md",
        "failures.json",
        "failed_seeds.csv",
        "status.json",
        "run.log",
        "manifest.json",
    }
    assert required <= {path.name for path in output.iterdir()}
    assert not list(output.rglob("*.tmp"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert required - {"manifest.json"} <= set(manifest["artifacts"])
    assert "src/utils/reproducibility.py" in manifest["source_sha256"]
    assert len(manifest["implementation_sha256"]) == 64
    environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
    assert environment["development_run_environment"] is True
    assert {"numpy", "scipy", "pandas"} <= set(environment["packages"])
    assert {"commit", "tree", "dirty"} <= set(environment["git"])
    assert manifest["git"] == environment["git"]
    assert manifest["git_snapshot_role"] == "run_start_before_output_mutation"
    assert {"commit", "tree", "dirty"} <= set(manifest["finalization_git"])
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "privileged generator-supported references" in report
    assert "budgets are **not matched**" in report
    assert "same one-scalar parameterization" in report
    assert pd.read_csv(output / "failed_seeds.csv").empty


def test_execute_retains_failed_seed_and_finalizes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "failed-result"
    original = exp41.run_seed

    def fail_one(config, seed):
        if seed == config["seeds"][0]:
            raise RuntimeError("injected failure")
        return original(config, seed)

    monkeypatch.setattr(exp41, "run_seed", fail_one)
    with pytest.raises(RuntimeError, match="seed failed"):
        execute(config_path, output)

    failures = json.loads((output / "failures.json").read_text(encoding="utf-8"))
    assert failures == [
        {
            "error": "injected failure",
            "error_type": "RuntimeError",
            "seed": config["seeds"][0],
            "status": "failed",
        }
    ]
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert manifest["status"] == "failed"
    assert not list(output.rglob("*.tmp"))
