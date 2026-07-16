"""Audit scalar-belief E/I gain-axis proposals from local three-factor signals.

The MD-like belief gate and one neutral receiver readout are fitted on whole
training episodes.  Neuron-local eligibility traces and gain-axis proposals
use only one frozen neutral-gain development trajectory.  This is therefore an
off-policy proposal/alignment audit, not closed-loop online local learning:
development states and errors are not recomputed after each axis update.  Test
episodes freeze the gate, readout, axis, and high-rank Dale E/I recurrent
checkpoint.  No BPTT, autograd, recurrent-weight learning, homeostasis, or
test-time fitting is used.

L1 and L2 are separate matched-budget panels.  The non-selected norm is always
diagnostic.  Only the oracle-third-factor upper bound may access development
context truth; orthogonal feedback is truth-free.  Every test-time gain
trajectory is controlled by the learned belief posterior.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp19_belief_ei_effective_dynamics as exp19
from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.models.context_belief import MDRecurrentBeliefGate
from src.plasticity.gain_axis import (
    apply_gain_axis_path_budget,
    make_deranged_shuffle,
    make_signed_permutation,
    orthogonal_component,
)
from src.tasks.hidden_context import (
    generate_hidden_context,
    make_hidden_context_random_tape,
)
from src.training.hidden_context_ei import (
    evaluate_receiver_condition,
    fit_receiver_readout,
    simulate_receiver,
)
from src.training.hidden_context_gain_axis import (
    GainAxisProposalTape,
    build_gain_axis_local_tape,
    make_gain_axis_proposal_tape,
)
from src.training.hidden_context_gate import split_hidden_context_dataset
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


EXPERIMENT = "exp22_hidden_context_local_gain_axis"
GATE_TIMING = exp19.GATE_TIMING
GATE_MODEL_NAME = exp19.GATE_MODEL_NAME
FEEDBACK_CONDITIONS = (
    "aligned_local",
    "oracle_third_factor",
    "random_signed_feedback",
    "shuffled_feedback",
    "orthogonal_feedback",
)


@dataclass(frozen=True, slots=True)
class _LearnedAxis:
    coefficient: np.ndarray
    axis: np.ndarray
    strength: float
    budget: dict[str, object]
    proposal_fingerprint: str
    axis_fingerprint: str


def _fingerprint(label: str, *values: object) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        else:
            digest.update(
                json.dumps(value, sort_keys=True, default=repr).encode("utf-8")
            )
        digest.update(b"\0")
    return digest.hexdigest()


def _planned_conditions(config: dict[str, Any]) -> list[dict[str, object]]:
    del config
    result = [
        {
            "condition": "frozen_zero",
            "feedback_condition": "frozen",
            "budget_norm": "none",
            "third_factor_source": "none",
        }
    ]
    for norm in ("l1", "l2"):
        for feedback in FEEDBACK_CONDITIONS:
            source = (
                "development_hidden_context_upper_bound"
                if feedback == "oracle_third_factor"
                else "learned_belief"
            )
            result.append(
                {
                    "condition": f"{feedback}_{norm}",
                    "feedback_condition": feedback,
                    "budget_norm": norm,
                    "third_factor_source": source,
                }
            )
    return result


def _orthogonalized_feedback(
    aligned: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    aligned_array = np.asarray(aligned, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if (
        aligned_array.ndim != 2
        or aligned_array.shape != candidate_array.shape
        or not np.all(np.isfinite(aligned_array))
        or not np.all(np.isfinite(candidate_array))
    ):
        raise ValueError("feedback matrices must be finite with matching 2D shape")
    result = np.empty_like(aligned_array)
    for epoch in range(aligned_array.shape[0]):
        aligned_row = aligned_array[epoch]
        projected = orthogonal_component(candidate_array[epoch], aligned_row)
        aligned_norm = float(np.linalg.norm(aligned_row))
        projected_norm = float(np.linalg.norm(projected))
        if aligned_norm == 0.0 or projected_norm == 0.0:
            raise RuntimeError("orthogonal feedback construction is degenerate")
        result[epoch] = projected * (aligned_norm / projected_norm)
        angle = _angle_degrees(aligned_row, result[epoch])
        if angle is None or not np.isclose(angle, 90.0, atol=1e-8, rtol=0.0):
            raise RuntimeError("orthogonal epoch feedback failed its 90-degree audit")
    frozen = np.array(result, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


def _condition_proposal(
    feedback: str,
    proposals: dict[str, GainAxisProposalTape],
) -> tuple[GainAxisProposalTape, bool]:
    if feedback not in FEEDBACK_CONDITIONS:
        raise ValueError(f"unknown feedback condition: {feedback}")
    if feedback not in proposals:
        raise RuntimeError(f"{feedback} proposal tape is unavailable")
    return proposals[feedback], feedback == "oracle_third_factor"


def _learn_axis(
    proposals: np.ndarray,
    *,
    norm: str,
    total_budget: float,
    tolerance: float,
    proposal_fingerprint: str,
) -> _LearnedAxis:
    if not 0.0 < float(total_budget) < 1.0:
        raise ValueError("gain-axis budget must lie strictly between zero and one")
    application = apply_gain_axis_path_budget(
        proposals,
        total_budget=float(total_budget),
        norm=norm,
        tolerance=float(tolerance),
    )
    coefficient = np.sum(application.applied_updates, axis=0)
    if np.max(np.abs(coefficient)) > float(total_budget) + float(tolerance):
        raise RuntimeError("budgeted coefficient violates positivity bound")
    n_events = int(proposals.shape[0])
    scaled_down = (
        application.raw_nonzero_events
        if application.scale_factor < 1.0 - float(tolerance)
        else 0
    )
    zero_scaled = (
        application.raw_nonzero_events if application.scale_factor == 0.0 else 0
    )
    selected_raw = application.raw_path_l1 if norm == "l1" else application.raw_path_l2
    budget_record = {
        "selected_norm": norm,
        "secondary_norm": "l2" if norm == "l1" else "l1",
        "total_budget": float(total_budget),
        "tolerance": float(tolerance),
        "planned_events": n_events,
        "processed_events": n_events,
        "raw_nonzero_events": application.raw_nonzero_events,
        "applied_nonzero_events": application.applied_nonzero_events,
        "zero_proposal_events": application.zero_proposal_events,
        "cumulative_raw_l1": application.raw_path_l1,
        "cumulative_raw_l2": application.raw_path_l2,
        "cumulative_applied_l1": application.applied_path_l1,
        "cumulative_applied_l2": application.applied_path_l2,
        "selected_raw": selected_raw,
        "selected_applied": application.selected_applied,
        "remaining": application.final_shortfall,
        "complete": True,
        "attained": application.attained,
        "final_shortfall": application.final_shortfall,
        "simultaneous_dual_norm_match": False,
        "secondary_norm_is_diagnostic_only": True,
        "scaled_down_event_count": scaled_down,
        "zero_scale_event_count": zero_scaled,
        "minimum_positive_scale_factor": (
            application.scale_factor if application.scale_factor > 0.0 else None
        ),
        "maximum_scale_factor": application.scale_factor,
        "global_scale_factor": application.scale_factor,
        "scaling_policy": (
            "single_global_downscale_preserves_event_relative_magnitude"
        ),
        "path_budget_application_id": application.fingerprint,
    }
    axis = coefficient / float(total_budget)
    fingerprint = _fingerprint(
        "exp22-budgeted-gain-axis-v1",
        norm,
        float(total_budget),
        proposal_fingerprint,
        coefficient,
        axis,
        budget_record,
    )
    frozen_axis = np.array(axis, dtype=np.float64, copy=True)
    frozen_axis.setflags(write=False)
    frozen_coefficient = np.array(coefficient, dtype=np.float64, copy=True)
    frozen_coefficient.setflags(write=False)
    return _LearnedAxis(
        coefficient=frozen_coefficient,
        axis=frozen_axis,
        strength=float(total_budget),
        budget=budget_record,
        proposal_fingerprint=proposal_fingerprint,
        axis_fingerprint=fingerprint,
    )


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    cosine = float(np.clip((left @ right) / (left_norm * right_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def run_seed(config: dict[str, Any], seed: int, results_root: str | Path) -> Path:
    """Run all paired local-axis cells for one independent seed."""

    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": (
            "dev_frozen_trajectory_three_factor_gain_axis_proposal_audit"
        ),
        "used_autograd": False,
        "used_bptt": False,
        "recurrent_learning": False,
        "gain_axis_learning": True,
    }
    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
    ) as run:
        try:
            planned = _planned_conditions(config)
        except Exception as error:
            planned = [{"condition": "setup"}]
            run.register_conditions(planned)
            run.mark_condition_failure(error, **planned[0])
            return run.path
        run.register_conditions(planned)
        dimensions = {str(item["condition"]): item for item in planned}
        emitted: set[str] = set()

        def fail(name: str, error: BaseException) -> None:
            if name in emitted:
                raise RuntimeError(f"condition emitted twice: {name}")
            run.mark_condition_failure(error, **dimensions[name])
            emitted.add(name)

        def succeed(name: str, metrics: dict[str, object]) -> None:
            if name in emitted:
                raise RuntimeError(f"condition emitted twice: {name}")
            run.record(metrics, **dimensions[name])
            emitted.add(name)

        try:
            if config.get("gate_timing") != GATE_TIMING:
                raise ValueError(f"gate_timing must equal {GATE_TIMING!r}")
            task = exp19._task_config(config)
            tape = make_hidden_context_random_tape(task, seed=seed)
            dataset = generate_hidden_context(task, seed=seed, random_tape=tape)
            splits = split_hidden_context_dataset(
                dataset,
                outer_test_fraction=float(config["outer_test_fraction"]),
                validation_fraction=float(config["validation_fraction"]),
                seed=seed,
            )
            network = exp19._network(config, task, seed)
            initial_weights = network.recurrent_weights
            network_id = _fingerprint(
                "exp22-frozen-dale-ei-network-v1",
                initial_weights,
                network.input_weights,
                network.excitatory_mask,
            )
            zero_axis = np.zeros(network.n_units, dtype=np.float64)
            gate = MDRecurrentBeliefGate(
                seed=derive_seed(seed, "exp19", "md-gate"),
                **dict(config["md_gate"]),
            ).fit(splits.train.gate)
            train_prediction = gate.predict(splits.train.gate)
            dev_prediction = gate.predict(splits.dev.gate)
            test_prediction = gate.predict(splits.test.gate)
            if any(
                prediction.test_accessed_true_context
                for prediction in (
                    train_prediction,
                    dev_prediction,
                    test_prediction,
                )
            ):
                raise RuntimeError("belief gate accessed hidden context")
            gate_checkpoint_id = _fingerprint(
                "exp22-train-cue-only-gate-v1",
                train_prediction.parameters,
                train_prediction.fit_trial_ids,
                train_prediction.fit_episode_ids,
            )
            neutral_common = dict(
                gain_axis=zero_axis,
                gain_strength=0.0,
                integration_substeps=int(config["integration_substeps"]),
                trial_batch_size=int(config["trial_batch_size"]),
                pathway_gating=True,
                population_gain=True,
            )
            train_neutral = simulate_receiver(
                network,
                splits.train,
                train_prediction.context_probability,
                **neutral_common,
            )
            readout = fit_receiver_readout(
                train_neutral,
                splits.train.task,
                alpha=float(config["readout_alpha"]),
            )
            dev_neutral = simulate_receiver(
                network,
                splits.dev,
                dev_prediction.context_probability,
                record_substeps=True,
                **neutral_common,
            )
            options = dict(config["gain_axis_learning"])
            budgets = dict(options["budgets"])
            if set(budgets) != {"l1", "l2"}:
                raise ValueError(
                    "gain_axis_learning.budgets must define exactly l1 and l2"
                )
            aligned_local_tape = build_gain_axis_local_tape(
                network,
                dev_neutral,
                splits.dev.task,
                readout,
                integration_substeps=int(config["integration_substeps"]),
                tau_eligibility_steps=float(options["tau_eligibility_steps"]),
            )
            aligned_feedback = np.asarray(
                aligned_local_tape.feedback_coefficients,
                dtype=np.float64,
            )
            condition_local_tapes = {
                "aligned_local": aligned_local_tape,
                "oracle_third_factor": aligned_local_tape,
            }
            learned_third_factor = 2.0 * dev_prediction.context_probability - 1.0
            learned_third_factor_id = _fingerprint(
                "exp22-learned-third-factor-v1",
                aligned_local_tape.trial_ids,
                learned_third_factor,
            )
            condition_proposals = {
                "aligned_local": make_gain_axis_proposal_tape(
                    aligned_local_tape,
                    learned_third_factor,
                    learning_rate=float(options["learning_rate"]),
                    error_clip=float(options["error_clip"]),
                )
            }
            feedback_transform_ids = {
                "aligned_local": "identity_readout_aligned_local_feedback",
                "oracle_third_factor": ("identity_readout_aligned_oracle_third_factor"),
            }
            feedback_angles = {
                "aligned_local": 0.0,
                "oracle_third_factor": 0.0,
            }
            feedback_coefficient_angles = {
                "aligned_local": 0.0,
                "oracle_third_factor": 0.0,
            }
            feedback_epoch_angles = {
                "aligned_local": (0.0, 0.0, 0.0),
                "oracle_third_factor": (0.0, 0.0, 0.0),
            }
            aligned_feedback_id = _fingerprint(
                "exp22-feedback-coefficients-v1",
                aligned_local_tape.feedback_policy,
                aligned_local_tape.feedback_coefficients,
            )
            feedback_coefficient_ids = {
                "aligned_local": aligned_feedback_id,
                "oracle_third_factor": aligned_feedback_id,
            }
            aligned_schedule_id = _fingerprint(
                "exp22-feedback-schedule-v1",
                aligned_local_tape.feedback_schedule,
            )
            feedback_schedule_ids = {
                "aligned_local": aligned_schedule_id,
                "oracle_third_factor": aligned_schedule_id,
            }
            condition_setup_errors: dict[str, BaseException] = {}
            planned_grid_id = _fingerprint("exp22-planned-grid-v1", tuple(planned))
            protocol_id = _fingerprint(
                "exp22-protocol-v1",
                {key: value for key, value in config.items() if key != "config_path"},
            )
            shared = {
                "profile": str(config.get("profile", "unspecified")),
                "training_algorithm": run_config["training_algorithm"],
                "used_autograd": False,
                "used_bptt": False,
                "recurrent_learning": False,
                "homeostasis_learning": False,
                "normalization_learning": False,
                "receiver_noise_policy": "none_deterministic",
                "shared_noise_id": tape.fingerprint,
                "random_tape_id": tape.fingerprint,
                "network_init_id": network_id,
                "gate_checkpoint_id": gate_checkpoint_id,
                "readout_checkpoint_id": readout.checkpoint_id,
                "split_id": splits.fingerprint,
                "dev_neutral_trajectory_id": dev_neutral.trajectory_fingerprint,
                "aligned_dev_local_tape_id": aligned_local_tape.fingerprint,
                "dev_local_tape_id": aligned_local_tape.fingerprint,
                "learned_third_factor_id": learned_third_factor_id,
                "dev_trial_order_id": _fingerprint(
                    "exp22-dev-trial-order-v1", aligned_local_tape.trial_ids
                ),
                "planned_condition_grid_id": planned_grid_id,
                "planned_condition_count": len(planned),
                "experiment_protocol_id": protocol_id,
                "statistics_unit": "seed",
                "split_unit": "episode",
                "gate_fit_train_only": True,
                "readout_fit_train_only": True,
                "gain_axis_fit_dev_only": True,
                "test_used_for_axis_fit": False,
                "gain_axis_learning_closed_loop": False,
                "dev_trajectory_recomputed_after_each_update": False,
                "off_policy_frozen_trajectory_proposal_audit": True,
                "proposal_learning_rate": float(options["learning_rate"]),
                "proposal_scale_predeclared_in_config": True,
                "proposal_scale_fit_on_test": False,
                "budget_controller_can_amplify_proposals": False,
                "budget_controller_used": False,
                "budget_matcher_can_amplify_proposals": False,
                "budget_preserves_event_relative_magnitude": True,
                "train_dev_test_episode_disjoint": True,
                "gate_fit_accessed_true_context": False,
                "gate_test_accessed_true_context": False,
                "test_gain_control_source": "learned_belief_posterior",
                "local_feedback_scope": (
                    "condition_specific_fixed_feedback_coefficient_times_"
                    "same_neuron_local_state_and_activation_derivative"
                ),
                "gain_axis_three_factor_rule_used_for_eligibility": True,
                "fixed_readout_feedback_coefficients_used": True,
                "feedback_transform_applied_before_local_eligibility": True,
                "proposal_coordinate_permutation_after_eligibility": False,
                "weight_transport_free_claim": False,
                "generic_recurrent_three_factor_claim_eligible": False,
                "closed_loop_local_plasticity_claim_eligible": False,
                "base_conditions_share_readout": True,
                "base_comparison_scope": (
                    "single_neutral_train_readout_paired_dev_axis_test_evaluation"
                ),
            }
        except Exception as error:
            for item in planned:
                fail(str(item["condition"]), error)
            return run.path

        try:
            frozen_simulation = simulate_receiver(
                network,
                splits.test,
                test_prediction.context_probability,
                zero_axis,
                gain_strength=0.0,
                integration_substeps=int(config["integration_substeps"]),
                trial_batch_size=int(config["trial_batch_size"]),
                pathway_gating=True,
                population_gain=True,
            )
            frozen_metrics = evaluate_receiver_condition(
                network=network,
                simulation=frozen_simulation,
                readout=readout,
                prediction=test_prediction,
                dataset=splits.test,
                gate_model=GATE_MODEL_NAME,
                intervention="none",
                gate_checkpoint_id=gate_checkpoint_id,
                gain_axis_id=_fingerprint("exp22-frozen-zero-axis-v1", zero_axis),
                split_id=splits.fingerprint,
                network_init_id=network_id,
                gate_operations_per_trial=float(config["gate_compute"]["operations"]),
                gate_state_updates_per_trial=float(config["gate_compute"]["states"]),
            )
            frozen_metrics.update(
                shared,
                gain_axis_learning=False,
                gain_axis_three_factor_rule_used_for_eligibility=False,
                fixed_readout_feedback_coefficients_used=False,
                feedback_transform_applied_before_local_eligibility=False,
                budget_preserves_event_relative_magnitude=False,
                gain_axis_local_plasticity_claim_eligible=False,
                gain_axis_off_policy_proposal_claim_eligible=False,
                frozen_zero_update_budget_baseline=True,
                budget_selected_norm="none",
                budget_total=0.0,
                budget_attained=True,
                budget_selected_raw=0.0,
                budget_global_scale_factor=None,
                budget_scaling_policy="none_zero_update_baseline",
                budget_path_application_id=None,
                budget_simultaneous_dual_norm_match=False,
                plasticity_l1_cost=0.0,
                plasticity_l2_cost=0.0,
                dev_truth_accessed_for_axis=False,
                axis_test_truth_accessed=False,
                gain_axis_coefficient_l1=0.0,
                gain_axis_coefficient_l2=0.0,
                gain_axis_angle_to_oracle_degrees=None,
                condition_dev_local_tape_id=None,
                condition_local_feedback_scope=None,
                feedback_coefficients_id=None,
                feedback_schedule_id=None,
                condition_third_factor_id=None,
                feedback_policy="frozen_zero_feedback",
                feedback_angle_to_aligned_degrees=None,
                feedback_coefficient_angle_to_aligned_degrees=None,
                feedback_sensory_angle_to_aligned_degrees=None,
                feedback_delay_angle_to_aligned_degrees=None,
                feedback_response_angle_to_aligned_degrees=None,
                feedback_coefficient_l1=0.0,
                feedback_coefficient_l2=0.0,
                gain_axis_feedback_transform_id="frozen_zero_feedback",
            )
            succeed("frozen_zero", frozen_metrics)
        except Exception as error:
            fail("frozen_zero", error)

        def register_feedback_condition(
            feedback: str,
            feedback_coefficients: np.ndarray,
            *,
            feedback_policy: str,
            transform_id: str,
        ) -> None:
            local_tape = build_gain_axis_local_tape(
                network,
                dev_neutral,
                splits.dev.task,
                readout,
                integration_substeps=int(config["integration_substeps"]),
                tau_eligibility_steps=float(options["tau_eligibility_steps"]),
                feedback_coefficients=feedback_coefficients,
                feedback_policy=feedback_policy,
            )
            condition_local_tapes[feedback] = local_tape
            condition_proposals[feedback] = make_gain_axis_proposal_tape(
                local_tape,
                learned_third_factor,
                learning_rate=float(options["learning_rate"]),
                error_clip=float(options["error_clip"]),
            )
            feedback_transform_ids[feedback] = transform_id
            feedback_angles[feedback] = _angle_degrees(
                aligned_local_tape.feedback_schedule.reshape(-1),
                local_tape.feedback_schedule.reshape(-1),
            )
            feedback_coefficient_angles[feedback] = _angle_degrees(
                aligned_feedback.reshape(-1),
                local_tape.feedback_coefficients.reshape(-1),
            )
            feedback_epoch_angles[feedback] = tuple(
                _angle_degrees(
                    aligned_feedback[epoch],
                    local_tape.feedback_coefficients[epoch],
                )
                for epoch in range(aligned_feedback.shape[0])
            )
            feedback_coefficient_ids[feedback] = _fingerprint(
                "exp22-feedback-coefficients-v1",
                local_tape.feedback_policy,
                local_tape.feedback_coefficients,
            )
            feedback_schedule_ids[feedback] = _fingerprint(
                "exp22-feedback-schedule-v1",
                local_tape.feedback_schedule,
            )

        try:
            random_transform = make_signed_permutation(
                network.n_units,
                seed=derive_seed(seed, "exp22", "random-feedback"),
                group_labels=network.excitatory_mask,
                sign_flips=True,
            )
            random_feedback = np.stack(
                [random_transform.apply(row) for row in aligned_feedback],
                axis=0,
            )
            register_feedback_condition(
                "random_signed_feedback",
                random_feedback,
                feedback_policy="random_signed_readout_feedback",
                transform_id=random_transform.fingerprint,
            )
        except Exception as error:
            condition_setup_errors["random_signed_feedback"] = error

        try:
            shuffled_transform = make_deranged_shuffle(
                network.n_units,
                seed=derive_seed(seed, "exp22", "shuffled-feedback"),
                group_labels=network.excitatory_mask,
            )
            shuffled_feedback = np.stack(
                [shuffled_transform.apply(row) for row in aligned_feedback],
                axis=0,
            )
            register_feedback_condition(
                "shuffled_feedback",
                shuffled_feedback,
                feedback_policy="deranged_shuffled_readout_feedback",
                transform_id=shuffled_transform.fingerprint,
            )
        except Exception as error:
            condition_setup_errors["shuffled_feedback"] = error

        try:
            orthogonal_candidate_transform = make_signed_permutation(
                network.n_units,
                seed=derive_seed(seed, "exp22", "orthogonal-feedback-candidate"),
                group_labels=network.excitatory_mask,
                deranged=True,
                sign_flips=True,
            )
            orthogonal_candidate = np.stack(
                [orthogonal_candidate_transform.apply(row) for row in aligned_feedback],
                axis=0,
            )
            orthogonal_feedback = _orthogonalized_feedback(
                aligned_feedback,
                orthogonal_candidate,
            )
            register_feedback_condition(
                "orthogonal_feedback",
                orthogonal_feedback,
                feedback_policy="orthogonalized_readout_feedback",
                transform_id=_fingerprint(
                    "exp22-orthogonalized-feedback-v1",
                    orthogonal_candidate_transform.fingerprint,
                    aligned_feedback,
                    orthogonal_feedback,
                ),
            )
            if not np.isclose(
                feedback_angles["orthogonal_feedback"],
                90.0,
                atol=1e-8,
                rtol=0.0,
            ) or not all(
                angle is not None and np.isclose(angle, 90.0, atol=1e-8, rtol=0.0)
                for angle in feedback_epoch_angles["orthogonal_feedback"]
            ):
                raise RuntimeError(
                    "time-expanded orthogonal feedback failed its angle audit"
                )
        except Exception as error:
            condition_setup_errors["orthogonal_feedback"] = error
            condition_local_tapes.pop("orthogonal_feedback", None)
            condition_proposals.pop("orthogonal_feedback", None)

        oracle_proposal_error: BaseException | None = None
        oracle_third_factor_id: str | None = None
        try:
            oracle_third_factor = (
                2.0 * splits.dev.truth.hidden_states.astype(float) - 1.0
            )
            oracle_third_factor_id = _fingerprint(
                "exp22-oracle-third-factor-v1",
                aligned_local_tape.trial_ids,
                oracle_third_factor,
            )
            condition_proposals["oracle_third_factor"] = make_gain_axis_proposal_tape(
                condition_local_tapes["oracle_third_factor"],
                oracle_third_factor,
                learning_rate=float(options["learning_rate"]),
                error_clip=float(options["error_clip"]),
            )
        except Exception as error:
            # Oracle-only diagnostics must never select availability of the
            # learned/frozen/random/shuffled/orthogonal cells.
            condition_proposals.pop("oracle_third_factor", None)
            oracle_proposal_error = error

        tolerance = float(options.get("budget_tolerance", 1e-9))
        for norm in ("l1", "l2"):
            learned_by_feedback: dict[str, _LearnedAxis] = {}
            truth_access: dict[str, bool] = {}
            proposal_ids: dict[str, str] = {}
            for feedback in FEEDBACK_CONDITIONS:
                name = f"{feedback}_{norm}"
                try:
                    if feedback in condition_setup_errors:
                        raise condition_setup_errors[feedback]
                    if (
                        feedback == "oracle_third_factor"
                        and oracle_proposal_error is not None
                    ):
                        raise oracle_proposal_error
                    proposal_tape, accessed_truth = _condition_proposal(
                        feedback,
                        condition_proposals,
                    )
                    learned_by_feedback[feedback] = _learn_axis(
                        np.asarray(proposal_tape.proposals, dtype=np.float64),
                        norm=norm,
                        total_budget=float(budgets[norm]),
                        tolerance=tolerance,
                        proposal_fingerprint=proposal_tape.fingerprint,
                    )
                    truth_access[feedback] = accessed_truth
                    proposal_ids[feedback] = proposal_tape.fingerprint
                except Exception as error:
                    fail(name, error)

            oracle_axis = learned_by_feedback.get("oracle_third_factor")
            for feedback in FEEDBACK_CONDITIONS:
                name = f"{feedback}_{norm}"
                if name in emitted:
                    continue
                learned_axis = learned_by_feedback[feedback]
                local_tape = condition_local_tapes[feedback]
                try:
                    simulation = simulate_receiver(
                        network,
                        splits.test,
                        test_prediction.context_probability,
                        learned_axis.axis,
                        gain_strength=learned_axis.strength,
                        integration_substeps=int(config["integration_substeps"]),
                        trial_batch_size=int(config["trial_batch_size"]),
                        pathway_gating=True,
                        population_gain=True,
                    )
                    metrics = evaluate_receiver_condition(
                        network=network,
                        simulation=simulation,
                        readout=readout,
                        prediction=test_prediction,
                        dataset=splits.test,
                        gate_model=GATE_MODEL_NAME,
                        intervention="none",
                        gate_checkpoint_id=gate_checkpoint_id,
                        gain_axis_id=learned_axis.axis_fingerprint,
                        split_id=splits.fingerprint,
                        network_init_id=network_id,
                        gate_operations_per_trial=float(
                            config["gate_compute"]["operations"]
                        ),
                        gate_state_updates_per_trial=float(
                            config["gate_compute"]["states"]
                        ),
                    )
                    budget = learned_axis.budget
                    angle = (
                        _angle_degrees(
                            learned_axis.coefficient,
                            oracle_axis.coefficient,
                        )
                        if oracle_axis is not None
                        else None
                    )
                    feedback_coefficients = np.asarray(
                        local_tape.feedback_coefficients,
                        dtype=np.float64,
                    )
                    metrics.update(
                        shared,
                        gain_axis_learning=True,
                        gain_axis_local_plasticity_claim_eligible=False,
                        gain_axis_off_policy_proposal_claim_eligible=bool(
                            feedback == "aligned_local"
                            and budget["attained"]
                            and not truth_access[feedback]
                        ),
                        frozen_zero_update_budget_baseline=False,
                        budget_selected_norm=norm,
                        budget_total=float(budgets[norm]),
                        budget_attained=bool(budget["attained"]),
                        budget_final_shortfall=budget["final_shortfall"],
                        budget_selected_raw=budget["selected_raw"],
                        budget_selected_applied=budget["selected_applied"],
                        budget_global_scale_factor=budget["global_scale_factor"],
                        budget_scaling_policy=budget["scaling_policy"],
                        budget_path_application_id=(
                            budget["path_budget_application_id"]
                        ),
                        budget_cumulative_raw_l1=budget["cumulative_raw_l1"],
                        budget_cumulative_raw_l2=budget["cumulative_raw_l2"],
                        budget_cumulative_applied_l1=(budget["cumulative_applied_l1"]),
                        budget_cumulative_applied_l2=(budget["cumulative_applied_l2"]),
                        budget_planned_events=budget["planned_events"],
                        budget_processed_events=budget["processed_events"],
                        budget_raw_nonzero_events=budget["raw_nonzero_events"],
                        budget_applied_nonzero_events=(
                            budget["applied_nonzero_events"]
                        ),
                        budget_zero_proposal_events=budget["zero_proposal_events"],
                        budget_scaled_down_event_count=(
                            budget["scaled_down_event_count"]
                        ),
                        budget_zero_scale_event_count=(
                            budget["zero_scale_event_count"]
                        ),
                        budget_minimum_positive_scale_factor=(
                            budget["minimum_positive_scale_factor"]
                        ),
                        budget_maximum_scale_factor=budget["maximum_scale_factor"],
                        budget_remaining=budget["remaining"],
                        budget_tolerance=budget["tolerance"],
                        budget_secondary_norm_is_diagnostic_only=(
                            budget["secondary_norm_is_diagnostic_only"]
                        ),
                        budget_simultaneous_dual_norm_match=False,
                        plasticity_l1_cost=budget["cumulative_applied_l1"],
                        plasticity_l2_cost=budget["cumulative_applied_l2"],
                        gain_axis_coefficient_l1=float(
                            np.sum(np.abs(learned_axis.coefficient))
                        ),
                        gain_axis_coefficient_l2=float(
                            np.linalg.norm(learned_axis.coefficient)
                        ),
                        gain_axis_coefficient_max_abs=float(
                            np.max(np.abs(learned_axis.coefficient))
                        ),
                        gain_axis_angle_to_oracle_degrees=angle,
                        gain_axis_proposal_id=proposal_ids[feedback],
                        gain_axis_feedback_transform_id=(
                            feedback_transform_ids[feedback]
                        ),
                        gain_axis_checkpoint_id=learned_axis.axis_fingerprint,
                        condition_dev_local_tape_id=local_tape.fingerprint,
                        condition_local_feedback_scope=(
                            local_tape.local_feedback_scope
                        ),
                        feedback_coefficients_id=(feedback_coefficient_ids[feedback]),
                        feedback_schedule_id=feedback_schedule_ids[feedback],
                        condition_third_factor_id=(
                            oracle_third_factor_id
                            if feedback == "oracle_third_factor"
                            else learned_third_factor_id
                        ),
                        feedback_policy=local_tape.feedback_policy,
                        feedback_angle_to_aligned_degrees=feedback_angles[feedback],
                        feedback_coefficient_angle_to_aligned_degrees=(
                            feedback_coefficient_angles[feedback]
                        ),
                        feedback_sensory_angle_to_aligned_degrees=(
                            feedback_epoch_angles[feedback][0]
                        ),
                        feedback_delay_angle_to_aligned_degrees=(
                            feedback_epoch_angles[feedback][1]
                        ),
                        feedback_response_angle_to_aligned_degrees=(
                            feedback_epoch_angles[feedback][2]
                        ),
                        feedback_coefficient_l1=float(
                            np.sum(np.abs(feedback_coefficients))
                        ),
                        feedback_coefficient_l2=float(
                            np.linalg.norm(feedback_coefficients)
                        ),
                        dev_truth_accessed_for_axis=truth_access[feedback],
                        axis_test_truth_accessed=False,
                        recurrent_weights_bitwise_frozen=bool(
                            np.array_equal(network.recurrent_weights, initial_weights)
                        ),
                    )
                    succeed(name, metrics)
                except Exception as error:
                    fail(name, error)

        if not np.array_equal(network.recurrent_weights, initial_weights):
            raise RuntimeError("frozen recurrent checkpoint changed during Exp22")
        expected = {str(item["condition"]) for item in planned}
        if emitted != expected:
            raise RuntimeError(
                f"planned/emitted mismatch: missing={sorted(expected - emitted)}"
            )
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "Local hidden-context gain-axis learning audit",
        "configs/formal/exp22_hidden_context_local_gain_axis.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        print(run_seed(config, seed, args.results_root))


if __name__ == "__main__":
    main()
