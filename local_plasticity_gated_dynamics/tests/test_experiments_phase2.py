from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from experiments.common import load_json_config
from experiments.exp02_context_ei_oracle_gate import run_seed as run_exp02_seed
from experiments.exp03_context_ei_learned_gate import run_seed as run_exp03_seed


@pytest.mark.parametrize(
    ("config_path", "runner", "expected_experiment"),
    [
        (
            "configs/smoke/exp02_context_ei_oracle_gate.json",
            run_exp02_seed,
            "exp02_context_ei_oracle_gate",
        ),
        (
            "configs/smoke/exp03_context_ei_learned_gate.json",
            run_exp03_seed,
            "exp03_context_ei_learned_gate",
        ),
    ],
)
def test_phase2_smoke_retains_every_required_cell(
    tmp_path: Path, config_path: str, runner, expected_experiment: str
) -> None:
    config = load_json_config(config_path)
    run_path = runner(config, 0, str(tmp_path))
    planned = json.loads((run_path / "planned_conditions.json").read_text())
    records = [
        json.loads(line)
        for line in (run_path / "metrics.jsonl").read_text().splitlines()
    ]

    required_names = {
        "local",
        "bptt",
        "readout-only",
        "no-gate",
        "no-homeostasis",
        "full-feedback",
        "shuffled-feedback",
        "separate-network",
    }
    expected_cells = sum(
        len(item.get("conditions", required_names)) for item in config["architectures"]
    )
    assert len(planned) == expected_cells
    assert len(records) == len(planned)
    assert {record["condition"] for record in records} == required_names
    assert {record["status"] for record in records} == {"complete"}
    assert all(record["experiment"] == expected_experiment for record in records)
    assert all(record["used_autograd"] is False for record in records if record["condition"] != "bptt")
    assert all(record["used_autograd"] is True for record in records if record["condition"] == "bptt")
    assert all(
        record["raw_update_effective_rank"] == 0.0
        for record in records
        if record["condition"] == "readout-only"
    )
    assert all(
        record["gate_context_accuracy"] is None
        for record in records
        if record["condition"] == "no-gate"
    )
    assert all(
        record["shared_coordinate_metrics_applicable"] is False
        for record in records
        if record["condition"] == "separate-network"
    )
    assert all(
        record["total_update_effective_rank"] is None
        and record["plasticity_update_energy"] is None
        and "isolated baseline" in record["plasticity_update_energy_reason"]
        for record in records
        if record["condition"] == "bptt"
    )
    for architecture in {record["architecture"] for record in records}:
        architecture_records = [
            record for record in records if record["architecture"] == architecture
        ]
        assert len({record["initialization_id"] for record in architecture_records}) == 1
        assert len({record["initialization_seed"] for record in architecture_records}) == 1
    assert all(
        record["model_kind"] == "ei"
        and record["homeostasis_control_interpretation"] == "applicable_ei_ablation"
        for record in records
        if record["condition"] == "no-homeostasis"
    )
    separate_records = [
        record for record in records if record["condition"] == "separate-network"
    ]
    assert all(
        len(record["activity_participation_ratio_by_network"]) == 2
        and record["activity_dimension_scope"]
        == "mean_of_within_network_context_dimensions"
        for record in separate_records
    )
    status = json.loads((run_path / "status.json").read_text())
    assert status["status"] == "complete"


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/formal/exp02_context_ei_oracle_gate.json",
        "configs/formal/exp03_context_ei_learned_gate.json",
    ],
)
def test_formal_phase2_grid_has_required_scale_seeds_and_ei_controls(
    config_path: str,
) -> None:
    config = load_json_config(config_path)
    assert config["seeds"] == list(range(20))
    architectures = config["architectures"]
    assert any(
        item["kind"] == "non_dale" and item["n_units"] == 256
        for item in architectures
    )
    non_dale = next(item for item in architectures if item["kind"] == "non_dale")
    assert "no-homeostasis" not in non_dale["conditions"]
    ei = [item for item in architectures if item["kind"] == "ei"]
    assert all(item["n_units"] == 512 for item in ei)
    assert {round(1.0 - item["excitatory_fraction"], 1) for item in ei} == {
        0.1,
        0.2,
        0.3,
    }
    assert len({round(item["inhibitory_gain"], 6) for item in ei}) == 3


def test_phase2_task_generation_failure_is_retained_for_every_cell(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(
        load_json_config("configs/smoke/exp02_context_ei_oracle_gate.json")
    )
    config["task"]["n_trials"] = 3
    run_path = run_exp02_seed(config, 0, str(tmp_path))
    planned = json.loads((run_path / "planned_conditions.json").read_text())
    records = [
        json.loads(line)
        for line in (run_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == len(planned)
    assert {record["status"] for record in records} == {"failed"}
    status = json.loads((run_path / "status.json").read_text())
    assert status["status"] == "complete_with_failures"


def test_phase2_setup_validation_is_preserved_as_an_artifact(tmp_path: Path) -> None:
    config = copy.deepcopy(
        load_json_config("configs/smoke/exp02_context_ei_oracle_gate.json")
    )
    config["architectures"] = []
    run_path = run_exp02_seed(config, 0, str(tmp_path))
    records = [
        json.loads(line)
        for line in (run_path / "metrics.jsonl").read_text().splitlines()
    ]

    assert (run_path / "config.json").exists()
    assert (run_path / "run.log").exists()
    assert len(records) == 1
    assert records[0]["condition"] == "setup"
    assert records[0]["status"] == "failed"
    status = json.loads((run_path / "status.json").read_text())
    assert status["status"] == "complete_with_failures"
