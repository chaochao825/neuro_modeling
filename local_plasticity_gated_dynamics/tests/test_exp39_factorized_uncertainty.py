from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import exp39_factorized_uncertainty as exp39
from experiments.exp39_factorized_uncertainty import (
    METHODS,
    run_seed,
    summarize,
    validate_config,
    validate_implementation_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> dict:
    config = json.loads(
        (
            ROOT
            / "configs/development/exp39_factorized_uncertainty_dev_v2.json"
        ).read_text(encoding="utf-8")
    )
    config["seeds"] = [390]
    config["stream"] = {
        "block_length": 8,
        "blocks_per_sequence": 8,
        "n_sequences": 4,
        "jump_variance": 4.0,
    }
    config["selection"]["ema_alpha_grid"] = [0.1]
    config["selection"]["window_grid"] = [4]
    config["selection"]["fixed_jump_grid"] = {
        "hazard": [0.01],
        "process_variance": [0.01],
        "observation_variance": [0.04],
    }
    config["selection"]["hazard_adaptation_rate_grid"] = [0.01]
    config["selection"]["process_adaptation_rate_grid"] = [0.2]
    config["selection"]["observation_adaptation_rate_grid"] = [0.2]
    config["selection"]["imm_switch_grid"] = [0.015625]
    config["analysis"]["recovery_window"] = 2
    for key in list(config["analysis"]["acceptance"]):
        if "positive_seeds" in key:
            config["analysis"]["acceptance"][key] = 1
    return config


def test_run_seed_preserves_pairing_and_complete_factorial() -> None:
    config = _small_config()
    validate_config(config, formal=False)
    blocks, audit, metadata = run_seed(config, 390)
    assert set(blocks["method"]) == set(METHODS)
    assert blocks.groupby("method")["test_tape_digest"].nunique().eq(1).all()
    assert blocks["test_tape_digest"].nunique() == 1
    assert set(blocks["cell"]) == {
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
    }
    assert blocks.groupby("method").size().nunique() == 1
    assert audit.groupby("selection_family")["selected"].sum().eq(1).all()
    assert metadata["controller_state_dimension"] == 3
    assert metadata["seen_imm_modes"] == 4
    assert metadata["oracle_factorial_imm_modes"] == 8


def test_summary_uses_seed_as_unit_and_reports_holm_family() -> None:
    config = _small_config()
    blocks, _, _ = run_seed(config, 390)
    seeds, comparisons, tracking, summary = summarize(blocks, config=config)
    assert seeds["seed"].nunique() == 1
    assert summary["statistics_unit"] == "seed"
    assert summary["multiplicity"]["method"] == "Holm"
    assert len(summary["multiplicity"]["family"]) == 5
    assert set(tracking["factor"]) == {"h", "q", "r"}
    assert np.all(np.isfinite(tracking["log_parameter_correlation"]))
    assert {"comparison", "factor"} <= set(comparisons.columns)


def test_formal_validation_is_fail_closed() -> None:
    config = _small_config()
    config["claim_upgrade_allowed"] = True
    with pytest.raises(ValueError, match="30 seeds"):
        validate_config(config, formal=True)


def test_receipt_mismatch_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("current\n", encoding="utf-8")
    receipt = {
        "protocol_version": "unit",
        "files": {str(tracked.relative_to(tmp_path)): "0" * 64},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    config = {
        "protocol_version": "unit",
        "implementation_receipt_path": "receipt.json",
    }
    monkeypatch.setattr(exp39, "PROJECT_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="mismatch"):
        validate_implementation_receipt(config)
