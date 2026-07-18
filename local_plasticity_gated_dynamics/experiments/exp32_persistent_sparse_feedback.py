"""Persistent reward-only actuator belief in a continuous hidden HMM stream.

Two fixed Exp31 actuator motifs generate paired potential outcomes.  The local
controller is not reset at hidden switches and receives only sparse delayed
reward from its executed action.  A known-hazard Bayesian reward filter and a
true-state oracle are explicitly labelled comparators; neither is the local
main method.  There is no autograd or BPTT.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.sparse_feedback_metrics import (
    binary_belief_scores,
    post_switch_cost,
    stream_accuracy,
    switch_diagnostics,
)
from src.models.capacity_limited_associative_actuator import (
    CapacityLimitedAssociativeActuator,
)
from src.plasticity.sparse_reward_belief import (
    BayesianDelayedRewardFilter,
    PersistentRewardBeliefSelector,
    RewardBeliefReceipt,
)
from src.tasks.hidden_reliability_association import (
    HiddenReliabilityBlockSpec,
    HiddenReliabilityTaskConfig,
    materialize_hidden_reliability_block,
)
from src.tasks.nonstationary_actuator_stream import (
    NonstationaryActuatorStreamConfig,
    make_nested_feedback_tapes,
    make_stream_tape,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


EXPERIMENT = "exp32_persistent_sparse_feedback"
PROTOCOL_VERSION = "exp32_persistent_sparse_reward_v1"
EVIDENCE_SCHEMA_VERSION = "exp32_evidence_v2"
SUPPORTED_EVIDENCE_SCHEMA_VERSIONS = {"exp32_evidence_v1", EVIDENCE_SCHEMA_VERSION}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "train_fixed_best",
    "matched_random",
    "cumulative_sample_average",
    "persistent_rpe_local",
    "credit_shuffled_local",
    "no_feedback_local",
    "bayes_reward_filter",
    "oracle_hidden_state",
)
CRITICAL_CODE_FILES = (
    "experiments/common.py",
    "experiments/exp32_persistent_sparse_feedback.py",
    "src/tasks/nonstationary_actuator_stream.py",
    "src/tasks/hidden_reliability_association.py",
    "src/models/capacity_limited_associative_actuator.py",
    "src/plasticity/sparse_reward_belief.py",
    "src/analysis/hidden_selector_metrics.py",
    "src/analysis/sparse_feedback_metrics.py",
    "src/utils/artifacts.py",
    "src/utils/reproducibility.py",
    "scripts/summarize_exp32.py",
    "figures/exp32_persistent_sparse_feedback_plot.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(*arrays: np.ndarray, labels: tuple[object, ...]) -> str:
    digest = hashlib.sha256(repr(labels).encode("utf-8"))
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_checkout_identity() -> dict[str, object]:
    """Return the checked-out Git object identity without hiding lookup failure."""

    git: dict[str, object] = {"commit": None, "tree": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        git = {"commit": commit, "tree": tree, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return git


def _source_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    files = {name: _sha256(PROJECT_ROOT / name) for name in CRITICAL_CODE_FILES}
    config_path = Path(str(config.get("config_path", "")))
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "critical_file_sha256": files,
        "source_config_sha256": (
            _sha256(config_path) if config_path.is_file() else None
        ),
        "formal_authorization_receipt_sha256": _formal_authorization_digest(config),
        "git": _git_checkout_identity(),
    }


def _canonical_formal_payload_digest(config: Mapping[str, Any]) -> str:
    """Hash every field allowed to affect the confirmatory run or inference."""

    payload = {
        "protocol_version": config.get("protocol_version"),
        "analysis_protocol_version": config.get("analysis_protocol_version"),
        "seeds": list(seed_list(config["seeds"])),
        "training_algorithm": config.get("training_algorithm"),
        "used_autograd": config.get("used_autograd"),
        "used_bptt": config.get("used_bptt"),
        "task": config.get("task"),
        "selector": config.get("selector"),
        "analysis": config.get("analysis"),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _formal_authorization_digest(config: Mapping[str, Any]) -> str | None:
    """Validate and bind the explicit development-to-confirmation transition."""

    if config.get("profile") != "formal":
        return None
    if config.get("formal_authorized") is not True:
        raise ValueError("formal Exp32 is fail-closed without authorization")
    relative = Path(str(config.get("formal_authorization_receipt", "")))
    if not relative.parts or relative.is_absolute():
        raise ValueError("formal authorization receipt must be a project-relative path")
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()) or not path.is_file():
        raise ValueError(
            "formal authorization receipt is outside the project or missing"
        )
    observed = _sha256(path)
    expected = str(config.get("formal_authorization_receipt_sha256", "")).lower()
    if observed != expected:
        raise ValueError("formal authorization receipt SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("authorization_status")
        != "independent_boundary_confirmation_authorized"
    ):
        raise ValueError("formal authorization receipt does not authorize confirmation")
    if payload.get("authorized_analysis_protocol") != config.get(
        "analysis_protocol_version"
    ):
        raise ValueError("formal authorization analysis protocol mismatch")
    if payload.get("generative_protocol") != PROTOCOL_VERSION:
        raise ValueError("formal authorization generative protocol mismatch")
    if payload.get("original_primary_scale_decision") != "scale-not-authorized":
        raise ValueError(
            "formal receipt must retain the failed original scale decision"
        )
    if payload.get("controller_parameters_unchanged") is not True:
        raise ValueError("formal receipt does not freeze controller parameters")
    formal_payload_digest = _canonical_formal_payload_digest(config)
    if (
        formal_payload_digest
        != str(config.get("authorized_formal_payload_sha256", "")).lower()
    ):
        raise ValueError("formal config payload does not match its authorized digest")
    if (
        formal_payload_digest
        != str(payload.get("authorized_formal_payload_sha256", "")).lower()
    ):
        raise ValueError("formal payload does not match the transition receipt")
    smoke_seeds = {int(value) for value in payload.get("development_smoke_seeds", [])}
    formal_seeds = {int(value) for value in seed_list(config["seeds"])}
    if smoke_seeds & formal_seeds:
        raise ValueError("formal seeds overlap the development smoke panel")
    if formal_seeds != {
        int(value) for value in payload.get("independent_formal_seeds", [])
    }:
        raise ValueError("formal seeds do not match the authorization receipt")
    for path_key, sha_key in (
        ("source_summary_path", "source_summary_sha256"),
        ("source_raw_metrics_path", "source_raw_metrics_sha256"),
        ("source_smoke_config_path", "source_smoke_config_sha256"),
    ):
        source = (PROJECT_ROOT / str(payload[path_key])).resolve()
        if not source.is_relative_to(PROJECT_ROOT.resolve()) or not source.is_file():
            raise ValueError(f"authorization source is missing: {path_key}")
        if _sha256(source) != str(payload[sha_key]).lower():
            raise ValueError(f"authorization source hash mismatch: {path_key}")
    return observed


def _task_config(config: Mapping[str, Any]) -> NonstationaryActuatorStreamConfig:
    payload = dict(config["task"])
    for name in (
        "direct_reliabilities",
        "load_values",
        "distractor_write_values",
        "hazards",
        "feedback_fractions",
        "feedback_delays",
    ):
        payload[name] = tuple(payload[name])
    return NonstationaryActuatorStreamConfig(**payload)


def _expected_holm_family(claim_family: str) -> tuple[str, ...]:
    """Return the exact registered family consumed by formal inference."""

    base = (
        "persistent_over_train_fixed",
        "persistent_over_opposite_eligibility",
    )
    if claim_family == "original_primary":
        return base
    if claim_family == "evidence_per_dwell_boundary":
        return (*base, "evidence_per_dwell_boundary_interaction")
    if claim_family == "feedback_memory_timescale_phase":
        return (*base, "evidence_response_slope")
    raise ValueError(f"unknown Exp32 claim_family: {claim_family}")


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("config protocol_version does not match Exp32")
    task = _task_config(config)
    if not seed_list(config["seeds"]):
        raise ValueError("Exp32 requires at least one seed")
    if bool(config.get("used_autograd")) or bool(config.get("used_bptt")):
        raise ValueError("Exp32 local controller cannot use autograd or BPTT")
    formal = config.get("profile") == "formal"
    if formal and config.get("formal_authorized") is not True:
        _formal_authorization_digest(config)
    selector = dict(config["selector"])
    if not 0.0 < float(selector["alpha"]) <= 1.0:
        raise ValueError("selector alpha must lie in (0, 1]")
    if not 0.0 <= float(selector["retention"]) <= 1.0:
        raise ValueError("selector retention must lie in [0, 1]")
    analysis = dict(config["analysis"])
    if float(analysis["primary_hazard"]) not in task.hazards:
        raise ValueError("primary hazard is not registered")
    if float(analysis["primary_feedback_fraction"]) not in task.feedback_fractions:
        raise ValueError("primary feedback fraction is not registered")
    if int(analysis["primary_feedback_delay"]) not in task.feedback_delays:
        raise ValueError("primary feedback delay is not registered")
    claim_family = str(analysis.get("claim_family", "original_primary"))
    if claim_family not in {
        "original_primary",
        "evidence_per_dwell_boundary",
        "feedback_memory_timescale_phase",
    }:
        raise ValueError(f"unknown Exp32 claim_family: {claim_family}")
    if formal:
        configured_holm_family = tuple(
            str(value) for value in analysis.get("holm_family", ())
        )
        expected_holm_family = _expected_holm_family(claim_family)
        if configured_holm_family != expected_holm_family:
            raise ValueError(
                "formal Exp32 Holm family must exactly match the registered "
                f"contrasts: {expected_holm_family}"
            )
    if claim_family == "evidence_per_dwell_boundary":
        reference_hazard = float(analysis["boundary_reference_hazard"])
        if reference_hazard not in task.hazards:
            raise ValueError("boundary reference hazard is not registered")
        if reference_hazard <= float(analysis["primary_hazard"]):
            raise ValueError("boundary reference hazard must exceed the primary hazard")
        if float(analysis["boundary_interaction_mcid"]) < 0.0:
            raise ValueError("boundary interaction MCID must be non-negative")
    if claim_family == "feedback_memory_timescale_phase":
        for hazard_name in ("iso_lambda_slow_hazard", "iso_lambda_fast_hazard"):
            if float(analysis[hazard_name]) not in task.hazards:
                raise ValueError(f"{hazard_name} is not registered")
        if float(analysis["iso_lambda_slow_hazard"]) >= float(
            analysis["iso_lambda_fast_hazard"]
        ):
            raise ValueError("iso-lambda slow hazard must be smaller than fast hazard")
        for name in (
            "evidence_slope_mcid",
            "timescale_structure_mcid",
        ):
            if float(analysis[name]) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for lambda_value in analysis["iso_lambda_values"]:
            for hazard_name in ("iso_lambda_slow_hazard", "iso_lambda_fast_hazard"):
                fraction = float(lambda_value) * float(analysis[hazard_name])
                if not any(
                    np.isclose(fraction, value) for value in task.feedback_fractions
                ):
                    raise ValueError("iso-lambda contrast is not present in the grid")
        if float(analysis["delay_probe_hazard"]) not in task.hazards:
            raise ValueError("delay probe hazard is not registered")
        for fraction in analysis["delay_probe_feedback_fractions"]:
            if float(fraction) not in task.feedback_fractions:
                raise ValueError("delay probe feedback fraction is not registered")
        for delay_name in ("delay_probe_short", "delay_probe_long"):
            if int(analysis[delay_name]) not in task.feedback_delays:
                raise ValueError(f"{delay_name} is not registered")
    if formal:
        _formal_authorization_digest(config)


def _condition_dimensions(
    *, hazard: float, feedback_fraction: float, feedback_delay: int, mode: str
) -> dict[str, object]:
    return {
        "hazard": float(hazard),
        "feedback_fraction": float(feedback_fraction),
        "feedback_delay": int(feedback_delay),
        "condition": mode,
        "selector_mode": mode,
        "protocol_version": PROTOCOL_VERSION,
    }


def _planned_conditions(config: Mapping[str, Any]) -> list[dict[str, object]]:
    task = _task_config(config)
    return [
        _condition_dimensions(
            hazard=hazard,
            feedback_fraction=fraction,
            feedback_delay=delay,
            mode=mode,
        )
        for hazard in task.hazards
        for fraction in task.feedback_fractions
        for delay in task.feedback_delays
        for mode in MODES
    ]


def _base_hidden_config(
    task: NonstationaryActuatorStreamConfig,
) -> HiddenReliabilityTaskConfig:
    return HiddenReliabilityTaskConfig(
        n_train_blocks_per_cell=2,
        n_test_blocks_per_cell=2,
        trials_per_block=task.n_trials,
        probe_trials=2,
        key_dim=task.key_dim,
        load_values=task.load_values,
        distractor_write_values=task.distractor_write_values,
        direct_reliabilities=task.direct_reliabilities,
        distractor_strength=task.distractor_strength,
    )


def _state_spec(
    task: NonstationaryActuatorStreamConfig,
    *,
    root_seed: int,
    split: str,
    state: int,
) -> HiddenReliabilityBlockSpec:
    if split not in {"train", "test"} or state not in {0, 1}:
        raise ValueError("split/state is invalid")
    # State 0: reliable route and high associative pressure.  State 1: weak
    # route and low associative pressure.  The actuator outputs, not an
    # explicit target mixture, determine the realized reward gaps.
    if state == 0:
        reliability = task.direct_reliabilities[1]
        load = task.load_values[1]
        distractors = task.distractor_write_values[1]
    else:
        reliability = task.direct_reliabilities[0]
        load = task.load_values[0]
        distractors = task.distractor_write_values[0]
    return HiddenReliabilityBlockSpec(
        root_seed=root_seed,
        split=split,
        block_id=(0 if split == "train" else 2) + state,
        cell_rep=0,
        direct_reliability=reliability,
        association_load=load,
        distractor_writes=distractors,
        block_seed=derive_seed(root_seed, EXPERIMENT, split, "state", state),
    )


def _potential_outcomes(
    task: NonstationaryActuatorStreamConfig,
    *,
    seed: int,
    split: str,
) -> tuple[np.ndarray, str, tuple[str, str]]:
    hidden_config = _base_hidden_config(task)
    actuator = CapacityLimitedAssociativeActuator(
        key_dim=task.key_dim, distractor_strength=task.distractor_strength
    )
    rewards = np.empty((2, task.n_trials, 2), dtype=np.float64)
    block_digests: list[str] = []
    for state in range(2):
        block = materialize_hidden_reliability_block(
            hidden_config,
            _state_spec(task, root_seed=seed, split=split, state=state),
        )
        outputs = actuator.evaluate(block)
        if not outputs.update_budget_exact:
            raise RuntimeError("the associative potential-outcome budget is invalid")
        rewards[state, :, 0] = outputs.routing == block.targets
        rewards[state, :, 1] = outputs.associative == block.targets
        block_digests.append(block.fingerprint)
    digest = _array_digest(rewards, labels=(seed, split, *block_digests))
    rewards.setflags(write=False)
    return rewards, digest, (block_digests[0], block_digests[1])


def _train_dictionary(train_rewards: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    counts = np.sum(train_rewards, axis=1)
    emission = (counts + 1.0) / (train_rewards.shape[1] + 2.0)
    winners = np.argmax(emission, axis=1).astype(np.int64)
    global_action = int(np.argmax(np.mean(train_rewards, axis=(0, 1))))
    if not np.array_equal(winners, np.array([0, 1], dtype=np.int64)):
        raise RuntimeError(
            "registered state cells did not naturally produce both actuator winners"
        )
    return global_action, winners, emission


def _run_adaptive(
    selector: PersistentRewardBeliefSelector | BayesianDelayedRewardFilter,
    potential_rewards: np.ndarray,
    feedback_available: np.ndarray,
    action_uniforms: np.ndarray,
    *,
    delay: int,
    withhold_feedback: bool = False,
) -> tuple[np.ndarray, RewardBeliefReceipt]:
    rewards = np.empty(potential_rewards.shape[0], dtype=np.float64)
    for trial, uniform in enumerate(action_uniforms):
        selector.advance(trial)
        action = selector.choose(float(uniform))
        reward = float(potential_rewards[trial, action])
        rewards[trial] = reward
        selector.schedule_feedback(
            source_trial=trial,
            executed_action=action,
            reward=reward,
            available=bool(feedback_available[trial]) and not withhold_feedback,
            delay=delay,
        )
    receipt = selector.receipt()
    if receipt.pending_feedback_count:
        raise RuntimeError("registered feedback did not arrive before stream end")
    return rewards, receipt


def _mode_outputs(
    *,
    mode: str,
    selector_config: Mapping[str, Any],
    seed: int,
    hazard: float,
    potential_rewards: np.ndarray,
    train_fixed_action: int,
    state_winners: np.ndarray,
    emission: np.ndarray,
    hidden_states: np.ndarray,
    feedback_available: np.ndarray,
    action_uniforms: np.ndarray,
    delay: int,
) -> tuple[np.ndarray, np.ndarray | None, RewardBeliefReceipt | None, np.ndarray]:
    n_trials = potential_rewards.shape[0]
    if mode == "train_fixed_best":
        actions = np.full(n_trials, train_fixed_action, dtype=np.int64)
        rewards = potential_rewards[np.arange(n_trials), actions]
        return rewards, None, None, actions
    if mode == "matched_random":
        actions = (action_uniforms < 0.5).astype(np.int64)
        rewards = potential_rewards[np.arange(n_trials), actions]
        return rewards, None, None, actions
    if mode == "oracle_hidden_state":
        actions = state_winners[hidden_states]
        rewards = potential_rewards[np.arange(n_trials), actions]
        return rewards, hidden_states.astype(np.float64), None, actions

    common = {
        "temperature": float(selector_config["temperature"]),
        "q_prior": float(selector_config["q_prior"]),
        "seed": derive_seed(seed, EXPERIMENT, "common-action-rng", hazard),
    }
    if mode == "cumulative_sample_average":
        selector = PersistentRewardBeliefSelector(
            alpha=float(selector_config["alpha"]),
            retention=1.0,
            temperature=float(selector_config["cumulative_temperature"]),
            q_prior=common["q_prior"],
            seed=common["seed"],
            update_mode="sample_average",
        )
        rewards, receipt = _run_adaptive(
            selector,
            potential_rewards,
            feedback_available,
            action_uniforms,
            delay=delay,
        )
    elif mode in {"persistent_rpe_local", "credit_shuffled_local", "no_feedback_local"}:
        selector = PersistentRewardBeliefSelector(
            alpha=float(selector_config["alpha"]),
            retention=float(selector_config["retention"]),
            temperature=common["temperature"],
            q_prior=common["q_prior"],
            seed=common["seed"],
            update_mode="fixed_rpe",
            credit_assignment=(
                "opposite" if mode == "credit_shuffled_local" else "executed"
            ),
        )
        rewards, receipt = _run_adaptive(
            selector,
            potential_rewards,
            feedback_available,
            action_uniforms,
            delay=delay,
            withhold_feedback=mode == "no_feedback_local",
        )
    elif mode == "bayes_reward_filter":
        selector = BayesianDelayedRewardFilter(
            emission,
            hazard=hazard,
            temperature=float(selector_config["bayes_temperature"]),
            seed=common["seed"],
        )
        rewards, receipt = _run_adaptive(
            selector,
            potential_rewards,
            feedback_available,
            action_uniforms,
            delay=delay,
        )
    else:
        raise ValueError(f"unknown Exp32 mode: {mode}")
    return (
        rewards,
        np.asarray(receipt.belief_probabilities),
        receipt,
        np.asarray(receipt.actions),
    )


def run_seed(
    config: dict[str, Any],
    seed: int,
    results_root: str | Path,
    *,
    run_label: str | None = None,
) -> Path:
    _validate_config(config)
    if seed not in seed_list(config["seeds"]):
        raise ValueError("seed is not registered by the selected Exp32 config")
    initialize_seed(seed)
    task = _task_config(config)
    provenance = _source_provenance(config)
    if (
        config.get("profile") == "formal"
        and provenance["git"].get("dirty") is not False
    ):
        raise RuntimeError("formal Exp32 requires a clean Git-bound worktree")
    run_config = {key: value for key, value in config.items() if key != "config_path"}
    run_config["evidence_provenance"] = provenance

    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
        run_label=run_label,
    ) as run:
        run.register_conditions(_planned_conditions(config))
        try:
            train_rewards, train_pool_digest, train_block_digests = _potential_outcomes(
                task, seed=seed, split="train"
            )
            test_rewards, test_pool_digest, test_block_digests = _potential_outcomes(
                task, seed=seed, split="test"
            )
            fixed_action, state_winners, emission = _train_dictionary(train_rewards)
            action_rng = np.random.default_rng(
                derive_seed(seed, EXPERIMENT, "paired-action-uniforms")
            )
            action_uniforms = action_rng.random(task.n_trials)
            action_uniform_digest = _array_digest(
                action_uniforms, labels=(seed, "paired-action-uniforms")
            )
            nested_feedback = make_nested_feedback_tapes(task, seed=seed)
            feedback_schedule_nested_audit = all(
                not np.any(nested_feedback[smaller][0] & ~nested_feedback[larger][0])
                for larger, smaller in zip(
                    task.feedback_fractions[:-1],
                    task.feedback_fractions[1:],
                    strict=True,
                )
            )
            if not feedback_schedule_nested_audit:
                raise RuntimeError("feedback schedules failed the nested-tape audit")
        except Exception as error:
            for dimensions in _planned_conditions(config):
                run.record_failed_condition(
                    {
                        "failure_stage": "seed_setup",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                        "run_git_commit": provenance["git"]["commit"],
                        "run_git_tree": provenance["git"]["tree"],
                        "run_git_dirty": provenance["git"]["dirty"],
                    },
                    **dimensions,
                )
            return run.path
        for hazard in task.hazards:
            for fraction in task.feedback_fractions:
                for delay in task.feedback_delays:
                    tape = make_stream_tape(
                        task,
                        seed=seed,
                        hazard=hazard,
                        feedback_fraction=fraction,
                        feedback_delay=delay,
                    )
                    trials = np.arange(task.n_trials)
                    potential = test_rewards[tape.hidden_states, trials]
                    selected_pool_digest = _array_digest(
                        potential,
                        labels=(seed, hazard, test_pool_digest, "selected-outcomes"),
                    )
                    oracle_actions = state_winners[tape.hidden_states]
                    oracle_rewards = potential[trials, oracle_actions]
                    for mode in MODES:
                        dimensions = _condition_dimensions(
                            hazard=hazard,
                            feedback_fraction=fraction,
                            feedback_delay=delay,
                            mode=mode,
                        )
                        try:
                            rewards, belief, receipt, actions = _mode_outputs(
                                mode=mode,
                                selector_config=config["selector"],
                                seed=seed,
                                hazard=hazard,
                                potential_rewards=potential,
                                train_fixed_action=fixed_action,
                                state_winners=state_winners,
                                emission=emission,
                                hidden_states=tape.hidden_states,
                                feedback_available=tape.feedback_available,
                                action_uniforms=action_uniforms,
                                delay=delay,
                            )
                            accuracy = stream_accuracy(rewards)
                            oracle_accuracy = stream_accuracy(oracle_rewards)
                            metrics: dict[str, Any] = {
                                "status": "complete",
                                "profile": config["profile"],
                                "statistics_unit": "seed",
                                "nested_unit": "continuous_trial_stream",
                                "split_unit": "whole_independent_stream",
                                "time_points_randomly_split": False,
                                "training_algorithm": (
                                    config["training_algorithm"]
                                    if mode == "persistent_rpe_local"
                                    else mode
                                ),
                                "used_autograd": False,
                                "used_bptt": False,
                                "controller_reset_at_switch": False,
                                "switch_times_exposed_to_selector": False,
                                "task_target_is_explicit_actuator_mixture": False,
                                "fixed_actuator_motifs_across_all_conditions": True,
                                "high_rank_carrier_present": False,
                                "carrier_dynamics_claimed": False,
                                "primary_scope": "full_continuous_test_stream",
                                "exploration_and_switch_cost_in_primary": True,
                                "selector_received_true_context": mode
                                == "oracle_hidden_state",
                                "selector_received_executed_scalar_reward": mode
                                in {
                                    "cumulative_sample_average",
                                    "persistent_rpe_local",
                                    "credit_shuffled_local",
                                    "bayes_reward_filter",
                                },
                                "selector_received_unexecuted_reward": False,
                                "selector_received_test_candidate_utility_vector": False,
                                "selector_received_switch_time": False,
                                "bayes_received_registered_hazard": mode
                                == "bayes_reward_filter",
                                "selector_used_train_state_labels": mode
                                in {"bayes_reward_filter", "oracle_hidden_state"},
                                "selector_used_train_counterfactual_outcomes": mode
                                in {
                                    "train_fixed_best",
                                    "bayes_reward_filter",
                                    "oracle_hidden_state",
                                },
                                "reward_only_interface_audit": mode
                                not in {
                                    "cumulative_sample_average",
                                    "persistent_rpe_local",
                                    "credit_shuffled_local",
                                    "bayes_reward_filter",
                                }
                                or (
                                    receipt is not None
                                    and receipt.used_true_context is False
                                    and receipt.used_counterfactual_reward is False
                                    and receipt.used_autograd is False
                                    and receipt.used_bptt is False
                                ),
                                "credit_assignment_is_executed_action": mode
                                != "credit_shuffled_local",
                                "credit_intervention_semantics": (
                                    "opposite_action_eligibility"
                                    if mode == "credit_shuffled_local"
                                    else "executed_action_eligibility"
                                ),
                                "train_fixed_best_action": int(fixed_action),
                                "train_state_winner_actions": state_winners.tolist(),
                                "train_emission_probabilities": emission.tolist(),
                                "both_actuator_winners_present": bool(
                                    np.array_equal(state_winners, np.array([0, 1]))
                                ),
                                "full_stream_accuracy": accuracy,
                                "oracle_hidden_accuracy": oracle_accuracy,
                                "dynamic_regret_to_hidden_oracle": oracle_accuracy
                                - accuracy,
                                "post_switch_cost_8": post_switch_cost(
                                    tape.hidden_states, rewards, window=8
                                ),
                                "hidden_switch_count": int(
                                    np.count_nonzero(tape.switch_mask)
                                ),
                                "action_one_fraction": float(np.mean(actions == 1)),
                                "feedback_available_count": int(
                                    np.count_nonzero(tape.feedback_available)
                                ),
                                "feedback_delivered_count": (
                                    int(receipt.delivered_rewards.size)
                                    if receipt is not None
                                    else 0
                                ),
                                "feedback_pending_count": (
                                    int(receipt.pending_feedback_count)
                                    if receipt is not None
                                    else 0
                                ),
                                "selector_update_l1": (
                                    float(receipt.cumulative_update_l1)
                                    if receipt is not None
                                    else 0.0
                                ),
                                "selector_update_l2": (
                                    float(receipt.cumulative_update_l2)
                                    if receipt is not None
                                    else 0.0
                                ),
                                "selector_update_budget_semantics": (
                                    "reward_driven_updates_only"
                                    if receipt is not None
                                    else "not_applicable"
                                ),
                                "selector_retention_l1": (
                                    float(receipt.cumulative_retention_l1)
                                    if receipt is not None
                                    else 0.0
                                ),
                                "selector_retention_l2": (
                                    float(receipt.cumulative_retention_l2)
                                    if receipt is not None
                                    else 0.0
                                ),
                                "selector_total_state_update_l1": (
                                    float(receipt.cumulative_total_state_update_l1)
                                    if receipt is not None
                                    else 0.0
                                ),
                                "selector_total_state_update_l2": (
                                    float(receipt.cumulative_total_state_update_l2)
                                    if receipt is not None
                                    else 0.0
                                ),
                                "selector_belief_dimension": (
                                    int(receipt.internal_state_dimension)
                                    if receipt is not None
                                    else 0
                                ),
                                "selector_internal_state_dimension": (
                                    int(receipt.internal_state_dimension)
                                    if receipt is not None
                                    else 0
                                ),
                                "selector_control_dimension": (
                                    int(receipt.control_dimension)
                                    if receipt is not None
                                    else 0
                                ),
                                "belief_metric_semantics": (
                                    receipt.belief_semantics
                                    if receipt is not None
                                    else "not_applicable"
                                ),
                                "delayed_rpe_reference": (
                                    receipt.rpe_reference
                                    if receipt is not None
                                    else "not_applicable"
                                ),
                                "feedback_schedule_nested_audit": feedback_schedule_nested_audit,
                                "no_feedback_matches_random_action_tape": (
                                    mode != "no_feedback_local"
                                    or np.array_equal(
                                        actions,
                                        (action_uniforms < 0.5).astype(np.int64),
                                    )
                                ),
                                "train_pool_fingerprint": train_pool_digest,
                                "test_pool_fingerprint": test_pool_digest,
                                "train_block_fingerprints": train_block_digests,
                                "test_block_fingerprints": test_block_digests,
                                "selected_potential_outcome_fingerprint": selected_pool_digest,
                                "state_tape_fingerprint": tape.state_fingerprint,
                                "feedback_tape_fingerprint": tape.feedback_fingerprint,
                                "action_uniform_tape_fingerprint": action_uniform_digest,
                                "selector_observation_fingerprint": (
                                    receipt.observation_fingerprint
                                    if receipt is not None
                                    else None
                                ),
                                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                                "run_git_commit": provenance["git"]["commit"],
                                "run_git_tree": provenance["git"]["tree"],
                                "run_git_dirty": provenance["git"]["dirty"],
                            }
                            if belief is not None:
                                metrics.update(
                                    binary_belief_scores(tape.hidden_states, belief)
                                )
                                metrics.update(
                                    switch_diagnostics(tape.hidden_states, belief)
                                )
                            else:
                                metrics.update(
                                    {
                                        "context_nll": None,
                                        "context_brier": None,
                                        "belief_state_accuracy": None,
                                        "median_switch_latency": None,
                                        "mean_switch_latency": None,
                                        "false_switch_rate": None,
                                        "belief_transition_count": None,
                                    }
                                )
                            run.record(metrics, **dimensions)
                        except Exception as error:
                            run.mark_condition_failure(error, **dimensions)
        return run.path


def _selected_seeds(config: dict[str, Any], override: str | None) -> Iterable[int]:
    return seed_list(override if override is not None else config["seeds"])


def main() -> None:
    parser = basic_parser(
        "Exp32 persistent sparse-feedback reward belief",
        "configs/smoke/exp32_persistent_sparse_feedback.json",
    )
    parser.add_argument("--run-label", help="path-safe panel label")
    args = parser.parse_args()
    config = load_json_config(args.config)
    for seed in _selected_seeds(config, args.seeds):
        path = run_seed(config, seed, args.results_root, run_label=args.run_label)
        print(path)


if __name__ == "__main__":
    main()
