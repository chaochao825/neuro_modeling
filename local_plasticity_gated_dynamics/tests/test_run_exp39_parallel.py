from __future__ import annotations

import json
from pathlib import Path

from scripts.run_exp39_parallel import parallel_execute


ROOT = Path(__file__).resolve().parents[1]


def test_parallel_wrapper_preserves_complete_seed_artifacts(
    tmp_path: Path,
) -> None:
    config = json.loads(
        (
            ROOT
            / "configs/development/exp39_factorized_uncertainty_dev_v2.json"
        ).read_text(encoding="utf-8")
    )
    config["seeds"] = [399]
    config["stream"].update(
        block_length=4, blocks_per_sequence=8, n_sequences=4
    )
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
    config["analysis"]["bootstrap_samples"] = 100
    for key in tuple(config["analysis"]["acceptance"]):
        if "positive_seeds" in key:
            config["analysis"]["acceptance"][key] = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "output"
    summary = parallel_execute(config_path, output, workers=1)
    assert summary["n_complete_seeds"] == 1
    assert json.loads((output / "failures.json").read_text()) == []
    assert (output / "seed_399/block_metrics.csv").stat().st_size > 0
    amendment = json.loads((output / "execution_amendment.json").read_text())
    assert amendment["scientific_functions_changed"] is False
