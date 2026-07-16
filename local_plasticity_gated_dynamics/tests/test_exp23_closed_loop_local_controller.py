"""Contracts for the paired closed-loop controller experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import exp23_closed_loop_local_controller as exp23


def _config() -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "smoke"
        / "exp23_closed_loop_local_controller.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_delayed_variant_is_cue_blank_delay_sensory_response() -> None:
    config = _config()
    current, tape = exp23._task_dataset(config, 0, "current")
    delayed, delayed_tape = exp23._task_dataset(config, 0, "delayed")

    assert tape == delayed_tape
    assert np.array_equal(current.gate.cue_observations, delayed.gate.cue_observations)
    assert list(dict.fromkeys(delayed.task.epoch.tolist())) == [
        "cue",
        "delay",
        "sensory",
        "response",
    ]
    delay = delayed.task.epoch == "delay"
    assert np.all(delayed.task.inputs[:, delay, :2] == 0.0)
    assert not np.any(delayed.task.loss_mask[:, delay])

    posterior = np.linspace(0.1, 0.9, delayed.task.trial_ids.size)
    reversed_posterior = 1.0 - posterior
    delayed_inputs = exp23._routed_inputs(delayed, posterior, "delayed")
    delayed_reversed = exp23._routed_inputs(
        delayed,
        reversed_posterior,
        "delayed",
    )
    np.testing.assert_array_equal(
        delayed_inputs[:, :, :2],
        delayed_reversed[:, :, :2],
    )
    np.testing.assert_array_equal(
        delayed_inputs[:, :, :2],
        delayed.task.inputs[:, :, :2],
    )

    network = exp23._network(config, delayed.config, 0)
    axis = np.linspace(-0.2, 0.2, network.n_units)
    controlled_rates, controlled_states, gains = exp23._simulate_axis_batch(
        network,
        delayed,
        posterior,
        axis,
        config,
        "delayed",
    )
    _, frozen_states, _ = exp23._simulate_axis_batch(
        network,
        delayed,
        posterior,
        np.zeros_like(axis),
        config,
        "delayed",
    )
    sensory = delayed.task.epoch == "sensory"
    assert np.all(gains[:, sensory] == 1.0)
    assert np.any(np.abs(controlled_states[:, delay]) > 0.0)
    assert np.any(
        np.abs(controlled_states[:, delay] - frozen_states[:, delay]) > 0.0
    )
    assert np.any(np.abs(controlled_rates[:, delay]) > 0.0)


def test_exp23_smoke_is_closed_loop_paired_and_recurrent_frozen(
    tmp_path: Path,
) -> None:
    path = exp23.run_seed(_config(), 0, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    planned = json.loads((path / "planned_conditions.json").read_text(encoding="utf-8"))
    records = _records(path)

    assert status["status"] == "complete"
    assert len(planned) == len(records) == 12
    assert all(row["status"] == "complete" for row in records)
    assert {str(row["condition"]) for row in records} == set(exp23.CONDITIONS)
    assert {str(row["task_variant"]) for row in records} == set(exp23.TASKS)

    for task in exp23.TASKS:
        rows = [row for row in records if row["task_variant"] == task]
        for field in (
            "random_tape_id",
            "split_id",
            "network_init_id",
            "gate_checkpoint_id",
            "readout_checkpoint_id",
            "readout_fit_data_id",
            "pairing_bundle_id",
            "recurrent_frozen_hash",
            "recurrent_weights_hash_before_condition",
            "recurrent_weights_hash_after_condition",
            "recurrent_weights_initial_id",
            "recurrent_weights_final_id",
        ):
            assert len({str(row[field]) for row in rows}) == 1
        assert all(row["paired_network_gate_readout_split_tape"] for row in rows)
        assert all(row["recurrent_weights_snapshot_is_copy"] for row in rows)
        assert all(row["recurrent_copy_isolation_audit_passed"] for row in rows)
        assert all(row["recurrent_hash_audit_passed"] for row in rows)
        assert all(row["recurrent_weights_bitwise_frozen"] for row in rows)
        assert all(
            row["recurrent_weights_audit"] == "independent_copy_and_sha256"
            for row in rows
        )
        assert all(
            row["recurrent_weights_audit"] == "independent_copy_and_sha256"
            for row in rows
        )
        assert all(
            row["recurrent_weights_initial_id"] == row["recurrent_weights_final_id"]
            for row in rows
        )
        assert all(not row["recurrent_learning"] for row in rows)
        assert all(row["readout_fit_train_only"] for row in rows)
        assert all(row["readout_fit_scope"] == "training_split_only" for row in rows)
        assert all(row["gate_fit_train_only"] for row in rows)
        assert all(row["gate_fit_scope"] == "training_split_only" for row in rows)
        assert all(row["axis_fit_dev_only"] for row in rows)
        assert all(row["axis_fit_scope"] == "development_split_only" for row in rows)
        assert all(not row["test_used_for_axis_fit"] for row in rows)
        assert all(not row["axis_selection_accessed_test"] for row in rows)
        assert all(not row["gate_test_accessed_true_context"] for row in rows)
        assert all(not row["third_factor_accessed_true_context"] for row in rows)
        assert all(row["hidden_context_access_audit_passed"] for row in rows)
        assert all(row["gate_moment_anchor_identifiable"] for row in rows)
        assert all(
            float(row["gate_mean_absolute_signed_belief_dev"]) > 0.10
            for row in rows
        )
        assert all(row["statistics_unit"] == "seed" for row in rows)
        assert all(row["split_unit"] == "episode" for row in rows)

    by_key = {
        (str(row["task_variant"]), str(row["condition"])): row for row in records
    }
    for task in exp23.TASKS:
        frozen = by_key[(task, "frozen")]
        off_policy = by_key[(task, "current_off_policy")]
        local = by_key[(task, "local_eprop")]
        exact = by_key[(task, "exact_forward_sensitivity")]
        bptt = by_key[(task, "bptt_axis_only")]
        random = by_key[(task, "random_update")]

        assert not frozen["closed_loop_learning"]
        assert frozen["update_events"] == 0
        assert off_policy["off_policy_frozen_trajectory"]
        assert not off_policy["closed_loop_learning"]
        assert not off_policy["dev_trajectory_recomputed_after_each_update"]
        assert off_policy["condition_schema_key"] == "current_off_policy"
        assert off_policy["condition_key_is_legacy_alias"]
        assert (
            off_policy["condition_method_label"]
            == exp23.FROZEN_TRAJECTORY_BLOCK_LOCAL_METHOD
        )
        assert (
            off_policy["training_algorithm"]
            == exp23.FROZEN_TRAJECTORY_BLOCK_LOCAL_METHOD
        )
        assert off_policy["local_proposal_rule_used"]
        assert not off_policy["local_learning"]
        assert not off_policy["closed_loop_local_learning_claim_eligible"]
        assert not off_policy["off_policy_exp22_method_claimed"]
        assert not off_policy["off_policy_exp22_proposal_reused"]
        assert (
            off_policy["off_policy_eligibility_approximation"]
            == "per_unit_state_rate_block_jacobian"
        )
        assert (
            off_policy["off_policy_proposal_axis"]
            == "zero_population_gain_axis"
        )
        assert not off_policy[
            "off_policy_proposal_trajectory_recomputed_after_each_update"
        ]
        assert off_policy["off_policy_proposal_budget_cap_respected"]
        assert set(off_policy["off_policy_legacy_no_op_config_fields"]) == {
            "error_clip",
            "tau_eligibility_steps",
        }
        for row in (local, exact, bptt, random):
            assert row["closed_loop_learning"]
            assert row["dev_trajectory_recomputed_after_each_update"]
            assert int(row["update_events"]) > 0
        assert local["local_learning"]
        assert local["closed_loop_local_learning_claim_eligible"]
        assert (
            local["local_eligibility_approximation"]
            == "per_unit_state_rate_block_jacobian"
        )
        assert not local["used_autograd"]
        assert not local["used_bptt"]
        assert local["local_rule_autograd_free"]
        assert local["local_rule_bptt_free"]
        assert not exact["used_autograd"]
        assert bptt["used_autograd"]
        assert bptt["used_bptt"]
        assert float(exact["median_update_cosine_to_exact"]) > 0.999999
        assert float(bptt["bptt_exact_gradient_cosine"]) > 0.999999
        assert bptt["bptt_optimizer"] == "deterministic_adam_axis_only"
        assert bptt["bptt_selection_scope"] == "full_dev_split_only"
        assert bptt["bptt_step_zero_eligible"]
        assert float(bptt["bptt_selected_learning_rate"]) in {
            0.001,
            0.003,
            0.01,
            0.03,
        }
        selected_step = int(bptt["bptt_selected_step"])
        assert int(bptt["update_events"]) == selected_step
        assert (
            len(bptt["episode_balanced_accuracy_history"])
            == selected_step + 1
        )
        assert (
            len(bptt["bptt_selected_dev_task_loss_history"])
            == selected_step + 1
        )
        selected_candidate = next(
            candidate
            for candidate in bptt["bptt_candidate_summaries"]
            if float(candidate["learning_rate"])
            == float(bptt["bptt_selected_learning_rate"])
        )
        assert int(selected_candidate["best_step"]) == selected_step
        assert int(selected_candidate["steps_run"]) >= selected_step
        if selected_step == 0:
            assert float(bptt["plasticity_l1_path"]) == 0.0
            assert float(bptt["plasticity_l2_path"]) == 0.0
            assert float(bptt["maximum_training_state_displacement"]) == 0.0
            assert bptt["probability_update_decreases_loss"] is None
        for row in (off_policy, local, exact, bptt, random):
            assert float(row["functional_budget_scale"]) <= float(
                row["functional_budget_maximum_scale"]
            )
            assert 0.5 <= float(row["gain_min_observed"]) <= 1.5
            assert 0.5 <= float(row["gain_max_observed"]) <= 1.5
            assert (
                row["functional_budget_type"]
                == "state_equality_plus_rate_gain_envelopes"
            )
            assert row["functional_budget_state_valid"]
            assert row["functional_budget_rate_valid"]
            assert row["functional_budget_gain_valid"]
            assert row["functional_budget_joint_local_valid"]
            assert row["functional_budget_cross_condition_complete"]
            assert row["functional_budget_cross_condition_valid"]
            assert row["joint_functional_budget_valid"]
            assert row["functional_budget_satisfied"]
            assert row["functional_budget_formal_ready"]
            assert float(row["functional_budget_rate_displacement"]) <= float(
                row["functional_budget_rate_displacement_limit"]
            ) + 1e-8
            assert float(row["functional_budget_gain_displacement"]) <= float(
                row["functional_budget_gain_displacement_limit"]
            ) + 1e-8
            assert float(
                row[
                    "functional_budget_cross_condition_state_relative_mismatch"
                ]
            ) <= float(
                row[
                    "functional_budget_cross_condition_state_relative_mismatch_tolerance"
                ]
            )
            assert float(
                row[
                    "functional_budget_cross_condition_rate_envelope_mismatch"
                ]
            ) <= float(
                row[
                    "functional_budget_cross_condition_rate_envelope_mismatch_tolerance"
                ]
            )
            assert float(
                row[
                    "functional_budget_cross_condition_gain_envelope_mismatch"
                ]
            ) <= float(
                row[
                    "functional_budget_cross_condition_gain_envelope_mismatch_tolerance"
                ]
            )


def test_exp23_setup_failure_retains_all_registered_cells(tmp_path: Path) -> None:
    config = _config()
    config["network"]["dt"] = 25.0
    path = exp23.run_seed(config, 3, tmp_path)
    records = _records(path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))

    assert status["status"] == "complete_with_failures"
    assert len(records) == 12
    assert all(row["status"] == "failed" for row in records)
    assert all("network.dt * integration_substeps" in str(row["error"]) for row in records)


def test_cross_condition_budget_audit_is_joint_and_fail_closed() -> None:
    config = _config()
    target = float(config["controller"]["matched_dev_state_displacement"])
    rate_limit = float(
        config["controller"]["matched_dev_rate_displacement_limit"]
    )
    gain_limit = (
        float(
            config["controller"][
                "matched_dev_gain_displacement_limit_per_unit"
            ]
        )
        * int(config["network"]["n_units"])
    )

    def rows(gain_values: list[float]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {
            "frozen": {
                "functional_budget_joint_local_valid": True,
                "functional_budget_achieved": 0.0,
                "functional_budget_rate_displacement": 0.0,
                "functional_budget_gain_displacement": 0.0,
                "functional_budget_gain_displacement_limit": gain_limit,
            }
        }
        for index, condition in enumerate(exp23.CONDITIONS[1:]):
            result[condition] = {
                "functional_budget_joint_local_valid": True,
                "functional_budget_achieved": target * (
                    0.99 + 0.005 * index
                ),
                "functional_budget_rate_displacement": rate_limit * (
                    0.30 + 0.02 * index
                ),
                "functional_budget_gain_displacement": gain_values[index],
                "functional_budget_gain_displacement_limit": gain_limit,
            }
        return result

    matched = rows([0.30 * gain_limit] * (len(exp23.CONDITIONS) - 1))
    exp23._apply_cross_condition_functional_budget_audit(matched, config)
    assert all(
        bool(matched[condition]["functional_budget_satisfied"])
        for condition in exp23.CONDITIONS
    )
    assert matched["local_eprop"]["functional_budget_cross_condition_valid"]

    mismatched = rows(
        [0.0, 0.0, 0.0, 0.0, 0.90 * gain_limit]
    )
    exp23._apply_cross_condition_functional_budget_audit(
        mismatched,
        config,
    )
    assert not mismatched["frozen"][
        "functional_budget_cross_condition_gain_mismatch_valid"
    ]
    assert all(
        not bool(mismatched[condition]["functional_budget_satisfied"])
        for condition in exp23.CONDITIONS[1:]
    )
    # The zero-control reference remains locally valid; it is not one of the
    # active directions whose comparability claim is being gated.
    assert mismatched["frozen"]["functional_budget_satisfied"]

    incomplete = rows([0.0] * (len(exp23.CONDITIONS) - 1))
    del incomplete["random_update"]
    exp23._apply_cross_condition_functional_budget_audit(incomplete, config)
    assert not incomplete["local_eprop"][
        "functional_budget_cross_condition_complete"
    ]
    assert not incomplete["local_eprop"]["functional_budget_satisfied"]
