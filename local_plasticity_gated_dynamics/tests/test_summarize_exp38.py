from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.common import load_json_config
from scripts.summarize_exp38 import build_qualification_receipt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return deepcopy(
        load_json_config(
            PROJECT_ROOT / "configs/prospective/exp38_stream51_soft_memory.json"
        )
    )


def _qualification_run(root: Path, seed: int, *, passed: bool) -> Path:
    run = root / f"seed_{seed}"
    run.mkdir(parents=True)
    summary = {
        "protocol_version": "exp38_stream51_soft_memory_v1",
        "stage": "qualification",
        "seed": seed,
        "external_features_accessed": False,
        "qualification": {
            "passed": passed,
            "stable_accumulation_gate": passed,
            "oracle_headroom_gate": passed,
            "cumulative_harm_gate": passed,
            "reachability_gate": passed,
        },
        "selected_hyperparameters": {"temperature": 1.0},
    }
    (run / "summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    return run


def test_qualification_receipt_requires_all_registered_seeds(tmp_path: Path) -> None:
    config = _config()
    runs = [
        _qualification_run(tmp_path / "runs", seed, passed=True)
        for seed in config["seeds"]
    ]
    preregistration = tmp_path / "preregistration.json"
    implementation = tmp_path / "implementation.json"
    preregistration.write_text("{}", encoding="utf-8")
    implementation.write_text("{}", encoding="utf-8")
    output = tmp_path / "qualification_receipt.json"
    payload = build_qualification_receipt(
        runs,
        config=config,
        output_path=output,
        preregistration_receipt_path=preregistration,
        implementation_receipt_path=implementation,
    )
    assert payload["all_registered_seeds_passed"] is True
    assert payload["external_stage_authorized"] is True
    assert payload["external_outcomes_inspected"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_one_failed_seed_keeps_external_stage_locked(tmp_path: Path) -> None:
    config = _config()
    runs = [
        _qualification_run(
            tmp_path / "runs", seed, passed=index != 2
        )
        for index, seed in enumerate(config["seeds"])
    ]
    preregistration = tmp_path / "preregistration.json"
    implementation = tmp_path / "implementation.json"
    preregistration.write_text("{}", encoding="utf-8")
    implementation.write_text("{}", encoding="utf-8")
    payload = build_qualification_receipt(
        runs,
        config=config,
        output_path=tmp_path / "receipt.json",
        preregistration_receipt_path=preregistration,
        implementation_receipt_path=implementation,
    )
    assert payload["all_registered_seeds_passed"] is False
    assert payload["external_stage_authorized"] is False


def test_missing_registered_seed_is_invalid_not_inconclusive(tmp_path: Path) -> None:
    config = _config()
    runs = [
        _qualification_run(tmp_path / "runs", seed, passed=True)
        for seed in config["seeds"][:-1]
    ]
    preregistration = tmp_path / "preregistration.json"
    implementation = tmp_path / "implementation.json"
    preregistration.write_text("{}", encoding="utf-8")
    implementation.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="registered seeds"):
        build_qualification_receipt(
            runs,
            config=config,
            output_path=tmp_path / "receipt.json",
            preregistration_receipt_path=preregistration,
            implementation_receipt_path=implementation,
        )
