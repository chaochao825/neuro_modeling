import json

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from shared_dynamics_real_data.run_visual_context import (
    _attempted_data_manifest,
    _data_manifest,
    _parse_seed_override,
    _planned_conditions,
    _source_fingerprint,
    _unit_subset,
    run_experiment,
    validate_config,
)
from shared_dynamics_real_data.build_report import collect_runs, _latest_panel


def _config() -> dict:
    return {
        "profile": "smoke",
        "seeds": [0, 1],
        "n_units": 8,
        "n_splits": 2,
        "purge": 1,
        "latent_dims": [1, 2],
        "ridge": 0.001,
        "variance_floor": 0.0001,
        "top_k": 2,
        "model_specs": [
            {
                "model": "common",
                "family": "common",
                "basis_control": "aligned",
            },
            {
                "model": "shared",
                "family": "shared",
                "basis_control": "aligned",
            },
        ],
    }


def test_planned_cells_are_complete_and_unique() -> None:
    config = validate_config(_config())
    planned = _planned_conditions(config["seeds"], config)
    assert len(planned) == 2 * 2 * 2 * 2
    keys = {
        (row["computational_seed"], row["fold"], row["latent_dim"], row["model"])
        for row in planned
    }
    assert len(keys) == len(planned)


def test_unit_subsets_and_seed_override_are_deterministic() -> None:
    first = _unit_subset(100, 12, seed=7)
    second = _unit_subset(100, 12, seed=7)
    assert (first == second).all()
    assert len(set(first.tolist())) == 12
    assert _parse_seed_override("4,9", [0]) == [4, 9]
    with pytest.raises(ValueError, match="distinct"):
        _parse_seed_override("4,4", [0])
    assert _source_fingerprint() == _source_fingerprint()


def test_config_rejects_duplicate_model_names() -> None:
    config = _config()
    config["model_specs"].append(dict(config["model_specs"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_config(config)


def test_run_experiment_retains_complete_and_failed_cells(tmp_path) -> None:
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir()
    rng = np.random.default_rng(42)
    # MATLAB files are unit x time. Both visual contexts share four units.
    for filename in (
        "data_mouse_visual_responding.mat",
        "data_mouse_visual_spontaneous.mat",
    ):
        savemat(filename := data_root / filename, {"X": rng.normal(size=(4, 24))})
        assert filename.is_file()
    config = _config()
    config.update(
        {
            "seeds": [0],
            "n_units": 4,
            "latent_dims": [1, 5],  # d=5 must fail, and the raw row must remain.
            "model_specs": [config["model_specs"][0]],
            "top_k": 1,
        }
    )
    config = validate_config(config)

    run_dir = run_experiment(
        config, data_root=data_root, results_root=results_root, seeds=[0]
    )

    for name in (
        "config.json",
        "environment.json",
        "planned_conditions.json",
        "data_manifest.json",
        "metrics.jsonl",
        "metrics.csv",
        "run.log",
        "status.json",
        "unit_subsets.json",
    ):
        assert (run_dir / name).is_file()
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert len(metrics) == 4
    assert (metrics["status"] == "complete").sum() == 2
    assert (metrics["status"] == "failed").sum() == 2
    assert metrics.loc[metrics["status"] == "failed", "error_type"].notna().all()
    assert metrics["analysis_fingerprint"].nunique() == 1
    assert metrics["data_fingerprint"].nunique() == 1
    resolved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    assert resolved["analysis_fingerprint"] == metrics["analysis_fingerprint"].iloc[0]
    assert manifest["data_fingerprint"] == metrics["data_fingerprint"].iloc[0]
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["planned_cells"] == status["recorded_cells"] == 4
    assert status["status"] == "complete_with_failures"

    loaded = {
        "visual_responding": np.zeros((24, 4), dtype=np.float32),
        "visual_spontaneous": np.zeros((24, 4), dtype=np.float32),
    }
    assert _attempted_data_manifest(data_root)["data_fingerprint"] == _data_manifest(
        data_root, loaded
    )["data_fingerprint"]


def test_top_level_data_failure_retains_aggregable_failed_panel(tmp_path) -> None:
    config = validate_config({**_config(), "seeds": [0]})
    results_root = tmp_path / "results"
    with pytest.raises(RuntimeError, match="artifacts retained"):
        run_experiment(
            config,
            data_root=tmp_path / "missing_data",
            results_root=results_root,
            seeds=[0],
        )

    run_dir = next((results_root / "runs").iterdir())
    manifest = json.loads(
        (run_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["load_status"] == "attempted"
    assert len(manifest["data_fingerprint"]) == 64
    latest = _latest_panel(collect_runs(results_root), "smoke")
    assert len(latest) == 2 * 2 * 2
    assert set(latest["status"]) == {"missing"}
    assert latest["data_fingerprint"].notna().all()
