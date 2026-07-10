import json
from pathlib import Path

from experiments.exp00_fixed_point import run_seed as run_exp00_seed
from experiments.exp01_feedback_dimension_sweep import run_seed as run_exp01_seed
from experiments.common import load_json_config


def test_exp00_smoke_writes_complete_artifacts(tmp_path: Path) -> None:
    config = load_json_config("configs/smoke/exp00_fixed_point.json")
    run_path = run_exp00_seed(config, 0, str(tmp_path))
    record = json.loads((run_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["learned_effective_rank"] <= config["latent_dim"] + 1e-6
    assert (run_path / "config.json").is_file()


def test_exp01_smoke_retains_every_planned_cell(tmp_path: Path) -> None:
    config = load_json_config("configs/smoke/exp01_feedback_dimension_sweep.json")
    run_path = run_exp01_seed(config, 0, str(tmp_path))
    planned = json.loads((run_path / "planned_conditions.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == len(planned)
    assert {record["status"] for record in records} <= {"complete", "invalid", "failed"}
    assert any(record.get("feedback_mode") == "aligned" for record in records)


def test_exp01_grid_configuration_failure_is_retained_inside_artifact(
    tmp_path: Path,
) -> None:
    config = load_json_config("configs/smoke/exp01_feedback_dimension_sweep.json")
    del config["feedback_dims"]
    run_path = run_exp01_seed(config, 0, str(tmp_path))
    record = json.loads((run_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["condition"] == "grid_setup"
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete_with_failures"
