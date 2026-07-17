"""Contracts for the preregistered Exp26 actuator phase diagram."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import exp26_actuator_phase_diagram as exp26
from experiments.common import load_json_config


ROOT = Path(__file__).resolve().parents[1]


def _config(profile: str = "smoke") -> dict[str, object]:
    return load_json_config(
        ROOT / "configs" / profile / "exp26_actuator_phase_diagram.json"
    )


def _records(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )


def test_exp26_manifest_and_seed_contract_are_preregistered() -> None:
    smoke = _config()
    formal = _config("formal")
    assert smoke["seeds"] == [9000, 9001]
    assert formal["seeds"] == list(range(30))
    assert set(smoke["seeds"]).isdisjoint(formal["seeds"])
    assert smoke["dev_only"] is True
    assert formal["dev_only"] is False
    assert len(exp26._manifest(smoke)) == 24
    assert len(exp26._manifest(formal)) == 88
    assert len(exp26._planned_conditions(smoke)) == 24 * 5
    assert len(exp26._planned_conditions(formal)) == 88 * 5
    assert formal["used_autograd"] is False
    assert formal["used_bptt"] is False


def test_registered_moments_encode_input_epoch_delay_and_noise() -> None:
    config = _config()
    carrier = exp26.make_carrier(exp26._carrier_config(config), 9000)
    dataset_config = exp26._dataset_config(config)
    state, inputs = exp26._registered_second_moments(
        carrier,
        dataset_config,
        delay=3,
        noise_std=0.3,
    )
    assert state.shape[0] == dataset_config.input_steps + 3
    assert inputs.shape[0] == state.shape[0]
    np.testing.assert_allclose(
        inputs[: dataset_config.input_steps],
        np.broadcast_to(np.eye(4), (dataset_config.input_steps, 4, 4)),
    )
    np.testing.assert_array_equal(inputs[dataset_config.input_steps :], 0.0)
    np.testing.assert_array_equal(state[0], 0.0)
    assert np.trace(state[-1]) > 0.0


def test_one_smoke_seed_is_paired_budget_matched_and_failure_retaining(
    tmp_path: Path,
) -> None:
    config = _config()
    path = exp26.run_seed(config, 9000, tmp_path)
    planned = json.loads(
        (path / "planned_conditions.json").read_text(encoding="utf-8")
    )
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    records = _records(path)
    assert len(planned) == 120
    assert len(records) == len(planned)
    assert status["status"] == "complete"
    assert set(records["status"]) == {"complete"}
    assert set(records["actuator_mode"]) == set(exp26.MODES)
    assert records["statistics_unit"].eq("seed").all()
    assert records["split_unit"].eq("block").all()
    assert (~records["time_points_randomly_split"]).all()
    assert records["readout_fit_train_only"].all()
    assert records["readout_shared_across_modes"].all()
    assert records["paired_noise_across_modes"].all()
    assert records["demand_marginal_decomposition_valid"].all()
    assert records["generator_state_input_cross_moment_zero_by_construction"].all()
    assert (~records["amplitudes_equalized_by_demand"]).all()
    assert (~records["effective_corrections_dale_constrained"]).all()
    assert records["functional_budget_valid"].all()
    active = records[records["actuator_mode"] != "frozen"]
    assert active["functional_budget_l2_relative_error"].max() <= 1e-8
    assert records.groupby("generator_id")["training_fingerprint"].nunique().max() == 1
    assert records.groupby("generator_id")["test_tape_fingerprint"].nunique().max() == 1
    assert records.groupby("generator_id")["base_recurrent_fingerprint"].nunique().max() == 1
    generators = records.drop_duplicates("generator_id")
    middle = generators[generators["alpha"] == 0.5]
    assert np.max(np.abs(middle["chi"] - middle["alpha"])) > 0.2


def test_smoke_endpoint_roles_follow_task_demand_without_manual_alignment(
    tmp_path: Path,
) -> None:
    records = _records(exp26.run_seed(_config(), 9000, tmp_path))
    pivot = records.pivot(
        index=["generator_id", "alpha", "chi"],
        columns="actuator_mode",
        values="test_balanced_accuracy",
    ).reset_index()
    input_cells = pivot[pivot["alpha"] == 0.0]
    state_cells = pivot[pivot["alpha"] == 1.0]
    np.testing.assert_allclose(input_cells["routing"], 1.0)
    np.testing.assert_allclose(state_cells["low_rank"], 1.0)
    assert float(np.mean(input_cells["routing"] - input_cells["low_rank"])) >= 0.4
    assert float(
        np.mean(state_cells["low_rank"] - state_cells[["routing", "gain"]].max(axis=1))
    ) >= 0.3
    advantage = pivot["low_rank"] - pivot[["routing", "gain"]].max(axis=1)
    assert float(pd.Series(pivot["chi"]).corr(pd.Series(advantage), method="spearman")) > 0.7
