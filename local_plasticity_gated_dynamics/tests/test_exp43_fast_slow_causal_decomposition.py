from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.exp43_fast_slow_causal_decomposition import (
    METHODS,
    execute,
    run_seed,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "development"
    / "exp43_fast_slow_causal_decomposition_probe_v1.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _smoke_config() -> dict[str, object]:
    config = deepcopy(_config())
    config["seeds"] = [43000]
    config["stream"] = {
        "block_lengths": [64],
        "blocks_per_sequence": 4,
        "n_sequences": 2,
        "jump_variance": 4.0,
    }
    config["selection"] = {
        "hazard_rate_grid": [0.02],
        "process_rate_grid": [0.1],
        "observation_rate_grid": [0.2],
        "fixed_process_grid": [0.02],
        "fixed_observation_grid": [0.04],
        "total_variance_decay_grid": [0.99],
        "total_variance_prior_mass_grid": [4.0],
        "total_variance_q_fraction_grid": [0.5],
        "imm_switch_grid": [0.01],
    }
    config["analysis"]["required_positive_seeds"] = 1
    return config


def test_exp43_config_is_development_only_and_reserves_formal_seeds() -> None:
    config = _config()
    validate_config(config)
    assert config["claim_upgrade_allowed"] is False
    assert set(config["seeds"]).isdisjoint(config["reserved_formal_seeds"])

    invalid = deepcopy(config)
    invalid["claim_upgrade_allowed"] = True
    with pytest.raises(ValueError, match="cannot upgrade"):
        validate_config(invalid)
    invalid = deepcopy(config)
    invalid["seeds"] = [43100]
    with pytest.raises(ValueError, match="overlap"):
        validate_config(invalid)


def test_exp43_seed_executes_complete_paired_panel_without_test_selection() -> None:
    config = _smoke_config()
    seed, blocks, events, regimes, audit, metadata = run_seed(config, 43000)
    assert tuple(seed["method"]) == METHODS
    assert set(blocks["method"]) == set(METHODS)
    assert set(events["method"]) == set(METHODS)
    assert set(regimes["method"]) == set(METHODS)
    assert audit["selected"].sum() == 4
    assert not any("test" in column.lower() for column in audit.columns)
    assert metadata["fit_tape_digest"] != metadata["test_tape_digest"]
    assert metadata["claim_upgrade_allowed"] is False


def test_exp43_execute_preserves_traceability_and_never_upgrades_claim(
    tmp_path: Path,
) -> None:
    config = _smoke_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "result"
    summary = execute(config_path, output)

    assert summary["verdict"] == "inconclusive_development_only"
    assert summary["claim_upgrade_allowed"] is False
    for name in (
        "config.json",
        "environment.json",
        "planned_conditions.json",
        "method_budget.csv",
        "seed_metrics.csv",
        "block_metrics.csv",
        "event_window_metrics.csv",
        "regime_window_metrics.csv",
        "selection_audit.csv",
        "comparisons.csv",
        "failures.json",
        "summary.json",
        "report.md",
        "run.log",
        "status.json",
        "manifest.json",
    ):
        assert (output / name).is_file(), name
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["n_failed_seeds"] == 0
    metrics = pd.read_csv(output / "seed_metrics.csv")
    assert set(metrics["method"]) == set(METHODS)


def test_exp43_refuses_to_overwrite_an_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        execute(CONFIG_PATH, output)
