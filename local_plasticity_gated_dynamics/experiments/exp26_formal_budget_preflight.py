"""Train-only functional-budget preflight for formal Exp26.

This audit enumerates every registered seed, generator and active actuator
mode before the expensive behavioral run.  It calls the same
``fit_task_matched_actuator`` entry point as Exp26 using training trajectories
only.  Validation/test arrays, rollouts, readouts and behavior metrics are
never accessed.

The registered ``max_scale`` is not changed.  When that bound blocks a fit, a
second diagnostic call through the same fit function uses the largest finite
floating-point bound solely to recover the required scale.  The condition is
still reported as unreachable under the registered configuration.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp26_actuator_phase_diagram as exp26
from experiments.common import load_json_config
from src.analysis.actuator_manifest import GeneratorCell, manifest_hash
from src.models.task_matched_actuators import ActuatorFitError
from src.tasks import actuator_matching as actuator_task
from src.tasks.actuator_matching import ActuatorTaskSpec, ActuatorTaskSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp26_formal_budget_preflight"
SCHEMA_VERSION = "exp26_budget_preflight_v2_observed_bound"
ACTIVE_MODES = tuple(mode for mode in exp26.MODES if mode != "frozen")
REQUIRED_SCALE_MATCH_RTOL = 1e-10
REQUIRED_SCALE_MATCH_ATOL = 1e-12
_MAX_SCALE_ERROR = "functional-budget scale is non-finite or exceeds max_scale"
_CRITICAL_CODE_FILES = (
    "experiments/exp26_formal_budget_preflight.py",
    "experiments/exp26_actuator_phase_diagram.py",
    "src/models/task_matched_actuators.py",
    "src/tasks/actuator_matching.py",
    "src/analysis/actuator_manifest.py",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def registered_config_sha256(config: Mapping[str, Any]) -> str:
    """Reuse the Exp26 canonical hash excluding runtime/provenance fields."""

    return exp26.canonical_config_sha256(config)


def _next_power_of_two(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("ceiling input must be positive and finite")
    return float(2.0 ** int(np.ceil(np.log2(value))))


def _git_output(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _worktree_content_sha256() -> tuple[str | None, bool | None]:
    status = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    if status is None:
        return None, None
    digest = hashlib.sha256()
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    digest.update(tracked_diff)
    untracked = _git_output("ls-files", "--others", "--exclude-standard") or ""
    for relative in sorted(line for line in untracked.splitlines() if line):
        path = PROJECT_ROOT / relative
        if path.is_file():
            encoded = relative.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            digest.update(path.read_bytes())
    return digest.hexdigest(), bool(status)


def code_tree_provenance() -> dict[str, object]:
    """Bind the receipt to both git state and all critical source bytes."""

    file_hashes: dict[str, str] = {}
    digest = hashlib.sha256()
    for relative in _CRITICAL_CODE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"critical Exp26 code file is missing: {path}")
        payload = path.read_bytes()
        file_hash = _sha256_bytes(payload)
        file_hashes[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    worktree_hash, dirty = _worktree_content_sha256()
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_tree": _git_output("rev-parse", "HEAD^{tree}"),
        "git_dirty": dirty,
        "worktree_content_sha256": worktree_hash,
        "critical_code_sha256": digest.hexdigest(),
        "critical_file_sha256": file_hashes,
        "fit_entrypoint": (
            "src.models.task_matched_actuators.fit_task_matched_actuator"
        ),
    }


def _paired_code_tree_provenance(
    start: Mapping[str, object],
    end: Mapping[str, object],
) -> dict[str, object]:
    compared = (
        "git_commit",
        "git_tree",
        "worktree_content_sha256",
        "critical_code_sha256",
    )
    result = dict(start)
    result["stable_during_run"] = all(start.get(key) == end.get(key) for key in compared)
    result["end_snapshot"] = dict(end)
    return result


def _validated_seed(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("formal seeds must be integers")
    result = int(value)
    if result < 0:
        raise ValueError("formal seeds must be non-negative")
    return result


def validate_formal_contract(config: Mapping[str, Any]) -> tuple[int, ...]:
    """Fail closed unless this is the registered 30-seed formal profile."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if config.get("profile") != "formal" or config.get("dev_only") is not False:
        raise ValueError("preflight requires profile='formal' and dev_only=false")
    raw_seeds = config.get("seeds")
    if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes)):
        raise TypeError("formal seeds must be a sequence")
    seeds = tuple(_validated_seed(value) for value in raw_seeds)
    if len(seeds) != 30 or len(set(seeds)) != 30:
        raise ValueError("formal preflight requires exactly 30 independent seeds")
    if seeds != tuple(range(30)):
        raise ValueError("formal preflight requires the registered seeds 0--29")
    if config.get("used_autograd") is not False or config.get("used_bptt") is not False:
        raise ValueError("formal Exp26 preflight requires non-BPTT fitting")
    if tuple(ACTIVE_MODES) != ("routing", "gain", "low_rank", "rgl"):
        raise RuntimeError("the registered Exp26 active-mode set changed")
    return seeds


def validate_registered_budget_policy(
    config: Mapping[str, Any],
    seeds: Sequence[int],
    cells: Sequence[GeneratorCell],
) -> float:
    """Verify the frozen train-only ceiling derivation and panel size."""

    policy = config.get("budget_preflight")
    if not isinstance(policy, Mapping):
        raise ValueError("formal Exp26 requires a registered budget_preflight policy")
    if (
        policy.get("receipt_schema") != SCHEMA_VERSION
        or policy.get("fit_scope") != "training_blocks_only"
        or policy.get("rounding_rule") != "next_power_of_two"
        or policy.get("validation_test_behavior_used") is not False
        or policy.get("validation_test_rollout_used") is not False
    ):
        raise ValueError("formal Exp26 budget preflight schema or scope is invalid")
    revision = policy.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("formal Exp26 budget preflight revision is missing")
    required_max = float(policy.get("required_scale_max", float("nan")))
    headroom = float(policy.get("headroom_multiplier", float("nan")))
    if not np.isfinite(required_max) or required_max <= 0.0:
        raise ValueError("registered required_scale_max must be positive and finite")
    if not np.isfinite(headroom) or headroom < 1.0:
        raise ValueError("registered headroom_multiplier must be finite and at least one")
    expected_fits = len(seeds) * len(cells) * len(ACTIVE_MODES)
    if int(policy.get("n_registered_active_fits", -1)) != expected_fits:
        raise ValueError("registered active-fit count differs from the formal panel")
    expected_ceiling = _next_power_of_two(headroom * required_max)
    configured_ceiling = float(config["actuator"]["max_scale"])
    if configured_ceiling != expected_ceiling:
        raise ValueError(
            "formal max_scale does not match the registered train-only policy"
        )
    return expected_ceiling


@dataclass(frozen=True)
class PreflightRecord:
    seed: int
    generator_id: str
    generator_split: str
    alpha: float
    transition_rank: int
    input_rank: int
    delay: int
    noise_std: float
    rotation_seed: int
    actuator_mode: str
    registered_max_scale: float
    required_budget_scale: float | None
    max_scale_exceeded: bool | None
    reachable_under_registered_max_scale: bool
    fit_status: str
    diagnostic_refit_used: bool
    target_l2_rms: float | None
    raw_current_l2_rms: float | None
    budget_relative_error: float | None
    train_split_fingerprint: str | None
    training_fingerprint: str | None
    process_noise_fingerprint: str | None
    validation_rollout_accessed: bool = False
    test_rollout_accessed: bool = False
    validation_behavior_accessed: bool = False
    test_behavior_accessed: bool = False
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class TrainOnlyTask:
    """Capability object containing no validation or test split."""

    spec: ActuatorTaskSpec
    train: ActuatorTaskSplit


def _base_record(
    *,
    seed: int,
    cell: GeneratorCell,
    mode: str,
    max_scale: float,
) -> dict[str, object]:
    return {
        "seed": int(seed),
        "generator_id": cell.generator_id,
        "generator_split": cell.generator_split,
        "alpha": float(cell.alpha),
        "transition_rank": int(cell.transition_rank),
        "input_rank": int(cell.input_rank),
        "delay": int(cell.delay),
        "noise_std": float(cell.noise_std),
        "rotation_seed": int(cell.rotation_seed),
        "actuator_mode": mode,
        "registered_max_scale": float(max_scale),
    }


def _failure_record(
    *,
    seed: int,
    cell: GeneratorCell,
    mode: str,
    max_scale: float,
    status: str,
    error: BaseException,
) -> PreflightRecord:
    return PreflightRecord(
        **_base_record(seed=seed, cell=cell, mode=mode, max_scale=max_scale),
        required_budget_scale=None,
        max_scale_exceeded=None,
        reachable_under_registered_max_scale=False,
        fit_status=status,
        diagnostic_refit_used=False,
        target_l2_rms=None,
        raw_current_l2_rms=None,
        budget_relative_error=None,
        train_split_fingerprint=None,
        training_fingerprint=None,
        process_noise_fingerprint=None,
        error_type=type(error).__name__,
        error_message=str(error),
    )


def _task_training_data(
    config: Mapping[str, Any],
    carrier: object,
    cell: GeneratorCell,
    *,
    seed: int,
) -> TrainOnlyTask:
    task = dict(config["task"])
    spec = exp26.make_task_spec(
        carrier,
        alpha=cell.alpha,
        rA=cell.transition_rank,
        rB=cell.input_rank,
        delay=cell.delay,
        noise=cell.noise_std,
        rotation_seed=cell.rotation_seed,
        generator_id=cell.generator_id,
        delta_a_log10_range=tuple(task["delta_a_log10_range"]),
        delta_b_log10_range=tuple(task["delta_b_log10_range"]),
        stability_limit=float(task["stability_limit"]),
    )
    train = actuator_task.make_actuator_matching_train_split(
        spec,
        exp26._dataset_config(config),
        seed=seed,
    )
    return TrainOnlyTask(spec=spec, train=train)


def _fit_record(
    config: Mapping[str, Any],
    dataset: TrainOnlyTask,
    *,
    seed: int,
    cell: GeneratorCell,
    mode: str,
) -> PreflightRecord:
    registered = exp26._actuator_config(config)
    # Intentional capability boundary: only the training split is obtained.
    train = dataset.train
    arguments = (
        train.target_states,
        train.inputs,
        exp26._context_tape(train),
        dataset.spec.carrier.a0,
        dataset.spec.carrier.b0,
    )
    keywords = {
        "mode": mode,
        "process_noise": train.noise,
        "config": registered,
    }
    diagnostic = False
    try:
        actuator = exp26.fit_task_matched_actuator(*arguments, **keywords)
        status = "complete"
    except ActuatorFitError as error:
        if str(error) != _MAX_SCALE_ERROR:
            failed = _failure_record(
                seed=seed,
                cell=cell,
                mode=mode,
                max_scale=registered.max_scale,
                status="fit_error",
                error=error,
            )
            values = asdict(failed)
            values["train_split_fingerprint"] = train.fingerprint
            return PreflightRecord(**values)
        diagnostic = True
        diagnostic_config = replace(
            registered,
            max_scale=float(np.finfo(np.float64).max),
        )
        try:
            actuator = exp26.fit_task_matched_actuator(
                *arguments,
                **{**keywords, "config": diagnostic_config},
            )
        except Exception as diagnostic_error:
            failed = _failure_record(
                seed=seed,
                cell=cell,
                mode=mode,
                max_scale=registered.max_scale,
                status="diagnostic_fit_error",
                error=diagnostic_error,
            )
            values = asdict(failed)
            values["diagnostic_refit_used"] = True
            values["train_split_fingerprint"] = train.fingerprint
            return PreflightRecord(**values)
        status = "blocked_max_scale"
    except Exception as error:
        failed = _failure_record(
            seed=seed,
            cell=cell,
            mode=mode,
            max_scale=registered.max_scale,
            status="fit_error",
            error=error,
        )
        values = asdict(failed)
        values["train_split_fingerprint"] = train.fingerprint
        return PreflightRecord(**values)

    receipt = actuator.receipt
    required = float(receipt.budget_scale)
    exceeded = bool(required > registered.max_scale)
    if diagnostic and not exceeded:
        raise RuntimeError(
            "diagnostic max-scale recovery disagrees with the registered gate"
        )
    if not diagnostic and exceeded:
        raise RuntimeError("registered fit accepted a scale above max_scale")
    return PreflightRecord(
        **_base_record(
            seed=seed,
            cell=cell,
            mode=mode,
            max_scale=registered.max_scale,
        ),
        required_budget_scale=required,
        max_scale_exceeded=exceeded,
        reachable_under_registered_max_scale=not exceeded,
        fit_status=status,
        diagnostic_refit_used=diagnostic,
        target_l2_rms=float(receipt.target_l2_rms),
        raw_current_l2_rms=float(receipt.raw_current_l2_rms),
        budget_relative_error=float(receipt.budget_l2_relative_error),
        train_split_fingerprint=train.fingerprint,
        training_fingerprint=receipt.training_fingerprint,
        process_noise_fingerprint=receipt.process_noise_fingerprint,
    )


def audit_seed(
    config: Mapping[str, Any],
    seed: int,
    cells: Sequence[GeneratorCell],
) -> list[PreflightRecord]:
    """Audit one independent seed without evaluating held-out behavior."""

    seed = _validated_seed(seed)
    registered = exp26._actuator_config(config)
    carrier = exp26.make_carrier(exp26._carrier_config(config), seed)
    records: list[PreflightRecord] = []
    for cell in cells:
        try:
            dataset = _task_training_data(config, carrier, cell, seed=seed)
        except Exception as error:
            records.extend(
                _failure_record(
                    seed=seed,
                    cell=cell,
                    mode=mode,
                    max_scale=registered.max_scale,
                    status="setup_error",
                    error=error,
                )
                for mode in ACTIVE_MODES
            )
            continue
        records.extend(
            _fit_record(config, dataset, seed=seed, cell=cell, mode=mode)
            for mode in ACTIVE_MODES
        )
    return records


def _worker(payload: tuple[dict[str, Any], int, tuple[GeneratorCell, ...]]) -> list[PreflightRecord]:
    config, seed, cells = payload
    return audit_seed(config, seed, cells)


def audit_cells(
    config: Mapping[str, Any],
    seeds: Iterable[int],
    cells: Sequence[GeneratorCell],
    *,
    workers: int = 1,
) -> list[PreflightRecord]:
    """Audit explicitly supplied seeds/cells; useful for unit and smoke runs."""

    resolved_seeds = tuple(_validated_seed(seed) for seed in seeds)
    if not resolved_seeds or len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("seeds must be non-empty and unique")
    resolved_cells = tuple(cells)
    if not resolved_cells or not all(isinstance(cell, GeneratorCell) for cell in cells):
        raise ValueError("cells must be a non-empty GeneratorCell sequence")
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer")
    if workers < 1:
        raise ValueError("workers must be positive")
    payloads = [
        (dict(config), seed, resolved_cells)
        for seed in resolved_seeds
    ]
    if workers == 1:
        batches = (_worker(payload) for payload in payloads)
        records = [record for batch in batches for record in batch]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batches = executor.map(_worker, payloads)
            records = [record for batch in batches for record in batch]
    records.sort(key=lambda item: (item.seed, item.generator_id, item.actuator_mode))
    expected = len(resolved_seeds) * len(resolved_cells) * len(ACTIVE_MODES)
    if len(records) != expected:
        raise RuntimeError("preflight did not retain every planned condition")
    return records


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("q50", "q90", "q95", "q99", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _optional_finite_positive(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) and result > 0.0 else None


def _optional_positive_integer(value: object) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        return None
    result = int(value)
    return result if result > 0 else None


def _budget_policy_bindings(
    config: Mapping[str, Any] | None,
    records: Sequence[PreflightRecord],
    *,
    observed_required_scale_max: float | None,
    expected_panel_size: int,
    expected_seed_count: int | None = None,
    expected_generator_count: int | None = None,
) -> dict[str, object]:
    """Recompute and bind the frozen observed-max budget policy."""

    policy_value = config.get("budget_preflight") if config is not None else None
    policy = policy_value if isinstance(policy_value, Mapping) else {}
    actuator_value = config.get("actuator") if config is not None else None
    actuator = actuator_value if isinstance(actuator_value, Mapping) else {}
    registered_required_max = _optional_finite_positive(
        policy.get("required_scale_max")
    )
    headroom = _optional_finite_positive(policy.get("headroom_multiplier"))
    registered_panel_size = _optional_positive_integer(
        policy.get("n_registered_active_fits")
    )
    configured_max_scale = _optional_finite_positive(actuator.get("max_scale"))
    revision = policy.get("revision")
    registered_receipt_schema = policy.get("receipt_schema")
    rounding_rule = policy.get("rounding_rule")
    policy_contract_valid = bool(
        isinstance(revision, str)
        and revision
        and registered_receipt_schema == SCHEMA_VERSION
        and policy.get("fit_scope") == "training_blocks_only"
        and rounding_rule == "next_power_of_two"
        and policy.get("validation_test_behavior_used") is False
        and policy.get("validation_test_rollout_used") is False
        and registered_required_max is not None
        and headroom is not None
        and headroom >= 1.0
        and registered_panel_size is not None
        and configured_max_scale is not None
    )
    if (
        observed_required_scale_max is not None
        and registered_required_max is not None
    ):
        absolute_error: float | None = abs(
            observed_required_scale_max - registered_required_max
        )
        observed_max_matches = bool(
            np.isclose(
                observed_required_scale_max,
                registered_required_max,
                rtol=REQUIRED_SCALE_MATCH_RTOL,
                atol=REQUIRED_SCALE_MATCH_ATOL,
            )
        )
    else:
        absolute_error = None
        observed_max_matches = False
    try:
        rounded_policy_ceiling = (
            _next_power_of_two(headroom * registered_required_max)
            if headroom is not None and registered_required_max is not None
            else None
        )
    except ValueError:
        rounded_policy_ceiling = None
    try:
        observed_rounded_ceiling = (
            _next_power_of_two(headroom * observed_required_scale_max)
            if headroom is not None and observed_required_scale_max is not None
            else None
        )
    except ValueError:
        observed_rounded_ceiling = None
    record_ceiling_binding_valid = bool(
        configured_max_scale is not None
        and all(
            record.registered_max_scale == configured_max_scale
            for record in records
        )
    )
    ceiling_binding_valid = bool(
        configured_max_scale is not None
        and rounded_policy_ceiling == configured_max_scale
        and observed_rounded_ceiling == configured_max_scale
        and record_ceiling_binding_valid
    )
    observed_panel_size = len(records)
    observed_seeds = {record.seed for record in records}
    observed_generators = {record.generator_id for record in records}
    observed_modes = {record.actuator_mode for record in records}
    observed_condition_keys = {
        (record.seed, record.generator_id, record.actuator_mode)
        for record in records
    }
    resolved_seed_count = (
        len(observed_seeds) if expected_seed_count is None else expected_seed_count
    )
    resolved_generator_count = (
        len(observed_generators)
        if expected_generator_count is None
        else expected_generator_count
    )
    panel_cartesian_complete = bool(
        len(observed_condition_keys) == observed_panel_size
        and len(observed_seeds) == resolved_seed_count
        and len(observed_generators) == resolved_generator_count
        and observed_modes == set(ACTIVE_MODES)
        and observed_panel_size
        == resolved_seed_count * resolved_generator_count * len(ACTIVE_MODES)
    )
    panel_binding_valid = bool(
        observed_panel_size == expected_panel_size == registered_panel_size
        and panel_cartesian_complete
    )
    all_required_scales_observed = all(
        record.required_budget_scale is not None for record in records
    )
    policy_valid = bool(
        policy_contract_valid
        and observed_max_matches
        and ceiling_binding_valid
        and panel_binding_valid
        and all_required_scales_observed
    )
    return {
        "policy_revision": revision if isinstance(revision, str) else None,
        "registered_receipt_schema": (
            registered_receipt_schema
            if isinstance(registered_receipt_schema, str)
            else None
        ),
        "receipt_schema_matches": registered_receipt_schema == SCHEMA_VERSION,
        "policy_contract_valid": policy_contract_valid,
        "policy_valid": policy_valid,
        "observed_required_scale_max": observed_required_scale_max,
        "registered_required_scale_max": registered_required_max,
        "observed_max_matches": observed_max_matches,
        "required_scale_match_rtol": REQUIRED_SCALE_MATCH_RTOL,
        "required_scale_match_atol": REQUIRED_SCALE_MATCH_ATOL,
        "required_scale_match_absolute_error": absolute_error,
        "registered_headroom_multiplier": headroom,
        "rounding_rule": rounding_rule,
        "rounded_policy_ceiling": rounded_policy_ceiling,
        "observed_rounded_policy_ceiling": observed_rounded_ceiling,
        "configured_max_scale": configured_max_scale,
        "record_ceiling_binding_valid": record_ceiling_binding_valid,
        "ceiling_binding_valid": ceiling_binding_valid,
        "observed_panel_size": observed_panel_size,
        "observed_unique_condition_count": len(observed_condition_keys),
        "observed_seed_count": len(observed_seeds),
        "expected_seed_count": resolved_seed_count,
        "observed_generator_count": len(observed_generators),
        "expected_generator_count": resolved_generator_count,
        "observed_modes": sorted(observed_modes),
        "panel_cartesian_complete": panel_cartesian_complete,
        "registered_panel_size": registered_panel_size,
        "expected_panel_size": expected_panel_size,
        "panel_binding_valid": panel_binding_valid,
        "all_required_scales_observed": all_required_scales_observed,
    }


def summarize_records(
    records: Sequence[PreflightRecord],
    *,
    config_sha256: str,
    receipt_manifest_hash: str,
    provenance: Mapping[str, object],
    config: Mapping[str, Any] | None = None,
    expected_panel_size: int | None = None,
    expected_seed_count: int | None = None,
    expected_generator_count: int | None = None,
) -> dict[str, object]:
    if not records:
        raise ValueError("records must be non-empty")
    required = [
        float(record.required_budget_scale)
        for record in records
        if record.required_budget_scale is not None
    ]
    blockers = [
        record for record in records if not record.reachable_under_registered_max_scale
    ]
    worst = max(
        (record for record in records if record.required_budget_scale is not None),
        key=lambda item: float(item.required_budget_scale),
        default=None,
    )
    by_mode = {
        mode: _quantiles(
            [
                float(record.required_budget_scale)
                for record in records
                if record.actuator_mode == mode
                and record.required_budget_scale is not None
            ]
        )
        for mode in ACTIVE_MODES
    }
    by_seed = {
        str(seed): _quantiles(
            [
                float(record.required_budget_scale)
                for record in records
                if record.seed == seed and record.required_budget_scale is not None
            ]
        )
        for seed in sorted({record.seed for record in records})
    }
    end_provenance = provenance.get("end_snapshot")
    provenance_clean = bool(
        provenance.get("git_dirty") is False
        and isinstance(end_provenance, Mapping)
        and end_provenance.get("git_dirty") is False
    )
    provenance_stable = bool(provenance.get("stable_during_run", False))
    scale_quantiles = _quantiles(required)
    observed_required_scale_max = scale_quantiles["max"]
    resolved_expected_panel = (
        len(records) if expected_panel_size is None else int(expected_panel_size)
    )
    bindings = _budget_policy_bindings(
        config,
        records,
        observed_required_scale_max=observed_required_scale_max,
        expected_panel_size=resolved_expected_panel,
        expected_seed_count=expected_seed_count,
        expected_generator_count=expected_generator_count,
    )
    zero_blockers = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_schema": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "canonical_config_sha256": config_sha256,
        "manifest_hash": receipt_manifest_hash,
        "code_tree_provenance": dict(provenance),
        "git_commit": provenance.get("git_commit"),
        "git_tree": provenance.get("git_tree"),
        "fit_scope": "training_blocks_only",
        "fit_entrypoint_matches_exp26": True,
        "validation_rollout_accessed": False,
        "test_rollout_accessed": False,
        "validation_behavior_accessed": False,
        "test_behavior_accessed": False,
        "registered_active_modes": list(ACTIVE_MODES),
        "n_records": len(records),
        "n_seeds": len({record.seed for record in records}),
        "n_generators": len({record.generator_id for record in records}),
        "n_modes": len({record.actuator_mode for record in records}),
        "n_required_scales_recovered": len(required),
        "n_unreachable": len(blockers),
        "n_max_scale_blockers": sum(
            record.max_scale_exceeded is True for record in records
        ),
        "n_other_failures": sum(
            record.max_scale_exceeded is None for record in blockers
        ),
        "all_reachable_under_registered_max_scale": zero_blockers,
        "provenance_stable_during_run": provenance_stable,
        "provenance_clean": provenance_clean,
        "preflight_passed": bool(
            zero_blockers
            and provenance_stable
            and provenance_clean
            and bindings["policy_valid"]
        ),
        "required_budget_scale_quantiles": scale_quantiles,
        "required_budget_scale_quantiles_by_mode": by_mode,
        "required_budget_scale_quantiles_by_seed": by_seed,
        "worst_condition": asdict(worst) if worst is not None else None,
        "policy_required_scale_max": bindings["registered_required_scale_max"],
        "derived_ceiling": bindings["rounded_policy_ceiling"],
        **bindings,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_receipt(
    output_dir: str | Path,
    config: Mapping[str, Any],
    cells: Sequence[GeneratorCell],
    records: Sequence[PreflightRecord],
    *,
    provenance: Mapping[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    """Persist raw cells and summary before returning pass/fail status."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=False)
    config_hash = registered_config_sha256(config)
    receipt_manifest_hash = manifest_hash(cells)
    if provenance is None:
        snapshot = code_tree_provenance()
        provenance = _paired_code_tree_provenance(snapshot, snapshot)
    summary = summarize_records(
        records,
        config_sha256=config_hash,
        receipt_manifest_hash=receipt_manifest_hash,
        provenance=provenance,
        config=config,
        expected_panel_size=(
            len(config.get("seeds", ())) * len(cells) * len(ACTIVE_MODES)
        ),
        expected_seed_count=len(config.get("seeds", ())),
        expected_generator_count=len(cells),
    )
    _write_json(path / "config.json", dict(config))
    (path / "config.sha256").write_text(config_hash + "\n", encoding="ascii")
    _write_json(path / "generator_manifest.json", [asdict(cell) for cell in cells])
    (path / "manifest.sha256").write_text(
        receipt_manifest_hash + "\n", encoding="ascii"
    )
    with (path / "preflight_cells.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True, allow_nan=False))
            handle.write("\n")
    _write_json(path / "preflight_summary.json", summary)
    _write_json(path / "code_tree_provenance.json", provenance)
    return path, summary


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    # ``results/runs`` is the repository's ignored artifact staging area.  A
    # receipt written to an unignored path would make the otherwise-clean tree
    # dirty before the formal runner can verify the receipt's Git identity.
    return PROJECT_ROOT / "results" / "runs" / EXPERIMENT / stamp


def run_formal_preflight(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    workers: int = 1,
) -> tuple[Path, dict[str, object]]:
    seeds = validate_formal_contract(config)
    cells = exp26._manifest(config)
    provenance_start = code_tree_provenance()
    records = audit_cells(config, seeds, cells, workers=workers)
    provenance_end = code_tree_provenance()
    provenance = _paired_code_tree_provenance(provenance_start, provenance_end)
    return write_receipt(
        output_dir,
        config,
        cells,
        records,
        provenance=provenance,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/formal/exp26_actuator_phase_diagram.json"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_json_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    path, summary = run_formal_preflight(
        config,
        output_dir,
        workers=args.workers,
    )
    print(path)
    if not summary.get(
        "preflight_passed",
        summary["all_reachable_under_registered_max_scale"],
    ):
        print(
            "formal Exp26 preflight failed: "
            f"{summary['n_unreachable']} unreachable conditions; "
            "code/tree stable="
            f"{summary.get('provenance_stable_during_run', 'unknown')}; "
            "policy valid="
            f"{summary.get('policy_valid', 'unknown')}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_MODES",
    "PreflightRecord",
    "audit_cells",
    "audit_seed",
    "code_tree_provenance",
    "main",
    "registered_config_sha256",
    "run_formal_preflight",
    "summarize_records",
    "validate_formal_contract",
    "validate_registered_budget_policy",
    "write_receipt",
]
