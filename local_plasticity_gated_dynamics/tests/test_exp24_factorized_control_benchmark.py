"""Contracts for the Exp24 oracle actuator-isolation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import exp24_factorized_control_benchmark as exp24


def _config(profile: str = "smoke") -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / profile
        / "exp24_factorized_control_benchmark.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_exp24_grid_and_formal_seed_contract() -> None:
    planned = exp24._planned_conditions()
    assert len(planned) == 10
    assert {row["task"] for row in planned} == {
        "routing_dominant",
        "dynamics_dominant",
    }
    assert {row["condition"] for row in planned} == {
        "frozen",
        "routing",
        "gain",
        "low_rank",
        "rgl",
    }
    assert all(row["control_dim"] == 2 for row in planned)
    formal = _config("formal")
    assert formal["seeds"] == list(range(30))
    assert formal["training_algorithm"] == "oracle_factorized_actuator_isolation"
    assert formal["used_autograd"] is False
    assert _config()["seeds"] == list(range(5))


def test_exp24_tasks_encode_distinct_computations() -> None:
    config = _config()
    routing = exp24._dataset(config, "routing_dominant", 7)
    dynamics = exp24._dataset(config, "dynamics_dominant", 7)
    assert routing.train.inputs.shape == dynamics.train.inputs.shape
    assert routing.train.contexts.shape == dynamics.train.contexts.shape
    assert set(routing.train.contexts.ravel()) == {0, 1}
    assert set(dynamics.train.contexts.ravel()) == {0, 1}
    assert (routing.ood.switch_steps >= 0).all()
    assert (dynamics.ood.switch_steps >= 0).all()
    # Dynamics inputs are generated independently of context; context enters
    # the registered target recurrence and oracle controller, not the drive law.
    assert config["task"]["dynamics_rho"] > 0


def test_routing_task_exposes_identity_to_input_and_gain_but_not_late_recurrence() -> None:
    config = _config()
    controller, _ = exp24._controller(config, "routing_dominant", 3)
    half = controller.n_units // 2
    assert np.all(controller.input_weights[:half, 1] == 0.0)
    assert np.all(controller.input_weights[half:, 0] == 0.0)
    assert np.all(controller.input_weights[:half, 0] > 0.0)
    assert np.all(controller.input_weights[half:, 1] > 0.0)
    dataset = exp24._dataset(config, "routing_dominant", 3)
    sensory_steps = int(config["task"]["routing_sensory_steps"])
    assert np.max(
        np.abs(dataset.train.inputs[:, :-sensory_steps])
    ) < np.max(np.abs(dataset.train.inputs[:, -sensory_steps:]))
    controls = np.array([[1.0, 0.0], [0.0, 1.0]])
    routing_scales = 1.0 + controls @ controller.routing_axes.T
    assert routing_scales[0, 0] > routing_scales[0, 1]
    assert routing_scales[1, 1] > routing_scales[1, 0]
    gain = 1.0 + controls @ controller.gain_axes.T
    assert np.mean(gain[0, :half]) > np.mean(gain[0, half:])
    assert np.mean(gain[1, half:]) > np.mean(gain[1, :half])


def test_exp24_batched_rollout_matches_factorized_controller_semantics() -> None:
    config = _config()
    dataset = exp24._dataset(config, "dynamics_dominant", 2)
    controller, _ = exp24._controller(config, "dynamics_dominant", 2)
    split = exp24.BenchmarkSplit(
        inputs=dataset.train.inputs[:1],
        contexts=dataset.train.contexts[:1],
        labels=dataset.train.labels[:1],
        instantaneous_labels=dataset.train.instantaneous_labels[:1],
        block_ids=dataset.train.block_ids[:1],
        switch_steps=dataset.train.switch_steps[:1],
    )
    batched = exp24._rollout_split(controller, split, "rgl", scale=0.4)
    controls = 0.4 * exp24._oracle_controls(split.contexts)[0]
    reference = controller.rollout(
        np.zeros(controller.n_units),
        split.inputs[0],
        mode="rgl",
        controls=controls,
    )
    np.testing.assert_allclose(batched.states[0], reference.states, atol=1e-12)
    np.testing.assert_allclose(batched.rates[0], reference.rates, atol=1e-12)
    np.testing.assert_allclose(
        batched.gains[0],
        reference.gain_history,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        batched.routing_scales[0],
        reference.routing_scale_history,
        atol=1e-12,
    )


def test_exp24_smoke_is_paired_train_safe_and_failure_retaining(
    tmp_path: Path,
) -> None:
    path = exp24.run_seed(_config(), 0, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    records = _records(path)
    planned = json.loads(
        (path / "planned_conditions.json").read_text(encoding="utf-8")
    )

    assert len(planned) == 10
    assert len(records) == 10
    assert status["status"] in {"complete", "complete_with_failures"}
    assert {(row["task"], row["condition"]) for row in records} == {
        (task, mode) for task in exp24.TASKS for mode in exp24.MODES
    }
    for task in exp24.TASKS:
        task_rows = [row for row in records if row["task"] == task]
        successful = [row for row in task_rows if row["status"] == "complete"]
        assert successful
        base_receipts = {
            row["base_recurrent_fingerprint"]
            for row in successful
            if "base_recurrent_fingerprint" in row
        }
        data_receipts = {
            row["dataset_fingerprint"]
            for row in successful
            if "dataset_fingerprint" in row
        }
        split_receipts = {
            row["block_split_fingerprint"]
            for row in successful
            if "block_split_fingerprint" in row
        }
        assert len(base_receipts) == 1
        assert len(data_receipts) == 1
        assert len(split_receipts) == 1
        for row in successful:
            assert row["control_dim"] == 2
            assert row["oracle_controller"]
            assert row["oracle_actuator_isolation"]
            assert not row["local_learning_enabled"]
            assert not row["used_bptt"]
            assert not row["used_autograd"]
            assert row["split_unit"] == "block"
            assert not row["time_points_randomly_split"]
            assert row["readout_fit_train_only"]
            assert row["dynamics_fit_train_only"]
            assert row["covariance_preprocessing_fit_train_only"]
            assert row["communication_subspace_fit_train_only"]
            assert row["normal_tangent_basis_fit_train_only"]
            assert not row["parameter_norm_budget_used"]
            assert row["all_functional_budget_terms_preregistered"]
            assert not row[
                "rate_gain_event_observables_reported_not_jointly_matched"
            ]
            assert row["joint_functional_budget_valid"]
            assert row["functional_budget_state_valid"]
            assert row["functional_budget_rate_valid"]
            assert row["functional_budget_gain_valid"]
            assert row["functional_budget_event_valid"]
            assert row["controlled_rollout_uses_future_inputs"]
            assert row["controlled_rollout_uses_future_oracle_controls"]
            assert not row["autonomous_rollout"]
            assert row["base_is_high_rank"]
            assert row["base_dale_columns"]
            assert "test_balanced_accuracy" in row
            assert "switch_latency_steps" in row
            assert "readout_interference" in row
            assert "delta_A_frobenius" in row
            assert "delta_B_frobenius" in row
            assert "delta_c_l2" in row
            assert "controlled_rollout_normalized_rmse" in row
            assert "normal_perturbation_endpoint_ratio" in row
            assert "conditional_covariance_bures_distance" in row
            assert "communication_source_overlap" in row
            assert "communication_heldout_r2_mean" in row


def test_exp24_five_seed_smoke_separates_actuator_roles() -> None:
    config = _config()
    routing_advantages: list[float] = []
    gain_advantages: list[float] = []
    low_rank_advantages: list[float] = []
    rgl_advantages: list[float] = []
    for seed in config["seeds"]:
        for task in exp24.TASKS:
            (
                dataset,
                controller,
                dale_signs,
                frozen,
                state_target,
                rate_target,
                raw_displacements,
            ) = exp24._task_setup(config, task, seed)
            metrics = {}
            for mode in exp24.MODES:
                row, valid = exp24._condition_metrics(
                    config,
                    task=task,
                    mode=mode,
                    seed=seed,
                    dataset=dataset,
                    controller=controller,
                    dale_signs=dale_signs,
                    frozen_train=frozen,
                    target_displacement=state_target,
                    target_rate_change=rate_target,
                    raw_displacements=raw_displacements,
                )
                assert valid
                metrics[mode] = row
            if task == "routing_dominant":
                routing_advantages.append(
                    metrics["routing"]["test_balanced_accuracy"]
                    - metrics["low_rank"]["test_balanced_accuracy"]
                )
                gain_advantages.append(
                    metrics["gain"]["test_balanced_accuracy"]
                    - metrics["low_rank"]["test_balanced_accuracy"]
                )
            else:
                low_rank_advantages.append(
                    metrics["low_rank"]["test_balanced_accuracy"]
                    - metrics["routing"]["test_balanced_accuracy"]
                )
                rgl_advantages.append(
                    metrics["rgl"]["test_balanced_accuracy"]
                    - metrics["routing"]["test_balanced_accuracy"]
                )
    assert min(routing_advantages) > 0.0
    assert np.median(routing_advantages) >= 0.10
    assert min(gain_advantages) > 0.0
    assert np.median(gain_advantages) >= 0.10
    assert min(low_rank_advantages) > 0.0
    assert np.median(low_rank_advantages) >= 0.15
    assert min(rgl_advantages) > 0.0
    assert np.median(rgl_advantages) >= 0.10


def test_exp24_setup_failure_registers_every_planned_cell(tmp_path: Path) -> None:
    config = _config()
    config["task"]["n_train_blocks"] = 3
    path = exp24.run_seed(config, 4, tmp_path)
    records = _records(path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert len(records) == 10
    assert all(row["status"] == "failed" for row in records)
    assert all(row["error_type"] == "ValueError" for row in records)
    assert status["status"] == "complete_with_failures"
