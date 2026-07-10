import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from experiments.common import load_json_config
from experiments.exp05_sequence_real_data import run_seed


def _fixture_session(path: Path) -> None:
    path.mkdir(parents=True)
    n_trials, n_units = 24, 6
    rule_pattern = np.array(["forward", "forward", "backward", "backward"])
    rank_pattern = np.array([1, 2, 1, 2])
    operation_pattern = np.array(["hold", "sort", "sort", "hold"])
    item_patterns = [
        np.array(values)
        for values in (
            ["a", "b", "b", "a"],
            ["a", "b", "b", "a"],
            ["a", "b", "a", "b"],
            ["a", "b", "a", "b"],
            ["a", "b", "b", "a"],
            ["a", "b", "a", "b"],
        )
    ]
    pd.DataFrame(
        {
            "block": np.repeat(np.arange(6), 4),
            "rule": np.tile(rule_pattern, 6),
            "item": np.concatenate(item_patterns),
            "rank": np.tile(rank_pattern, 6),
            "operation": np.tile(operation_pattern, 6),
            "choice": np.tile([0, 1, 1, 0], 6),
        }
    ).to_csv(path / "trials.csv", index=False)
    pd.DataFrame({"unit": range(n_units), "channel": range(n_units)}).to_csv(
        path / "units.csv", index=False
    )
    rng = np.random.default_rng(0)
    spikes = np.empty((n_trials, n_units), dtype=object)
    for trial in range(n_trials):
        for unit in range(n_units):
            count = 2 + ((trial + unit) % 3)
            spikes[trial, unit] = np.sort(rng.uniform(0.0, 0.2, size=count))
    savemat(path / "spikes.mat", {"spikes": spikes})


def test_sequence_experiment_runs_on_fixture_and_missing_data_is_retained(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _fixture_session(data_root / "session-a")
    config = load_json_config("configs/smoke/exp05_sequence_real_data.json")
    config["data_root"] = str(data_root)
    path = run_seed(config, 0, str(tmp_path / "results"))
    records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    assert any(record.get("model_family") == "shared" for record in records)
    assert all(record["status"] == "complete" for record in records)

    missing = dict(config, data_root=str(tmp_path / "absent"))
    missing_path = run_seed(missing, 0, str(tmp_path / "results"))
    missing_record = json.loads((missing_path / "metrics.jsonl").read_text())
    assert missing_record["status"] == "failed"
    assert missing_record["aggregation_level"] == "session"
