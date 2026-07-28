from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.exp43_fast_slow_causal_decomposition import execute
from scripts.validate_exp43_development_result import validate_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "development"
    / "exp43_fast_slow_causal_decomposition_probe_v1.json"
)


def _smoke_config() -> dict[str, object]:
    config = deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
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


def test_exp43_validator_replays_complete_smoke_result(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_smoke_config()), encoding="utf-8")
    output = tmp_path / "result"
    execute(config_path, output)
    receipt = validate_result(output)
    assert receipt["status"] == "pass"
    assert receipt["summary_replay"] == "pass"
    assert receipt["formal_seeds_accessed"] is False
    assert receipt["claim_upgrade_allowed"] is False


def test_exp43_validator_rejects_tampered_aggregate(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_smoke_config()), encoding="utf-8")
    output = tmp_path / "result"
    execute(config_path, output)
    metrics_path = output / "seed_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics.loc[0, "overall_nll"] += 1.0
    metrics.to_csv(metrics_path, index=False)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_result(output)


def test_exp43_validator_rejects_missing_artifact(tmp_path: Path) -> None:
    output = tmp_path / "incomplete"
    output.mkdir()
    with pytest.raises(ValueError, match="missing required"):
        validate_result(output)
