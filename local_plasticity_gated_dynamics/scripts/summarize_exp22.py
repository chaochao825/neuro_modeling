"""Publish the Exp22 off-policy gain-axis proposal matched-budget audit.

Exp22 derives three-factor proposals from a frozen development trajectory,
then evaluates the resulting postsynaptic gain axis with a frozen high-rank
Dale E/I recurrent matrix.  This is an off-policy proposal-alignment audit,
not closed-loop online local learning and not recurrent plasticity.  L1 and
L2 budgets are separate panels, failed cells are retained, and the independent
seed is the only inferential unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp22_hidden_context_local_gain_axis import (  # noqa: E402
    EXPERIMENT,
    _planned_conditions,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/formal/exp22_hidden_context_local_gain_axis.json"
)
DEFAULT_PREFIX = "exp22_hidden_context_local_gain_axis_formal"
TERMINAL_RUN_STATUS = {"complete", "complete_with_failures"}
ROW_STATUS = {"complete", "failed", "invalid"}
PANELS = ("l1", "l2")
CORE_COMPARATORS = (
    "frozen_zero",
    "random_signed_feedback",
    "shuffled_feedback",
)
ORTHOGONAL = "orthogonal_feedback"
ORACLE = "oracle_third_factor"
PAIRING_IDS = (
    "network_init_id",
    "gate_checkpoint_id",
    "readout_checkpoint_id",
    "split_id",
    "random_tape_id",
    "shared_noise_id",
    "dev_neutral_trajectory_id",
    "dev_trial_order_id",
    "learned_third_factor_id",
    "planned_condition_grid_id",
    "experiment_protocol_id",
)
PRIMARY_FAMILY = "Holm(exp22_primary_seed_sign_family)"
NEGATIVE_CONTROL_FAMILY = "Holm(exp22_orthogonal_seed_sign_family)"
ORACLE_FAMILY = "Holm(exp22_oracle_margin_seed_sign_family)"
ABSOLUTE_FAMILY = "Holm(exp22_absolute_balanced_accuracy_seed_sign_family)"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ENVIRONMENT_PACKAGES = (
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "torch",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain a non-empty JSONL object stream")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_sha256(environment: Mapping[str, Any]) -> str:
    """Validate and hash a declared Python 3.11 scientific environment."""

    if not isinstance(environment, Mapping):
        raise ValueError("formal run environment must be a JSON object")
    python = environment.get("python")
    platform = environment.get("platform")
    executable = environment.get("executable")
    packages = environment.get("packages")
    if not isinstance(python, str) or re.match(r"^3\.11\.", python) is None:
        raise ValueError("formal run environment must use Python 3.11")
    if not isinstance(platform, str) or not platform.strip():
        raise ValueError("formal run environment lacks platform provenance")
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("formal run environment lacks executable provenance")
    if not isinstance(packages, Mapping):
        raise ValueError("formal run environment lacks package provenance")
    missing = [name for name in _ENVIRONMENT_PACKAGES if not packages.get(name)]
    if missing:
        raise ValueError(f"formal run environment lacks packages: {missing}")
    payload = {
        "python": python,
        "platform": platform,
        "executable": executable,
        "packages": {name: str(packages[name]) for name in _ENVIRONMENT_PACKAGES},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_publication_provenance(
    provenance: Mapping[str, object],
) -> dict[str, str]:
    if not isinstance(provenance, Mapping):
        raise ValueError("Exp22 publication provenance must be a mapping")
    commit = str(provenance.get("analysis_git_commit", ""))
    script_sha256 = str(provenance.get("analysis_script_sha256", ""))
    python = provenance.get("analysis_python")
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("Exp22 publication provenance lacks a valid Git commit")
    if _DIGEST.fullmatch(script_sha256) is None:
        raise ValueError("Exp22 publication provenance lacks a script digest")
    if not isinstance(python, str) or re.match(r"^3\.11\.", python) is None:
        raise ValueError("Exp22 publication analysis must use Python 3.11")
    return {
        "analysis_git_commit": commit,
        "analysis_script_sha256": script_sha256,
        "analysis_python": python,
    }


def _analysis_provenance() -> dict[str, object]:
    """Bind a formal snapshot to clean Python 3.11 analysis code."""

    if sys.version_info[:2] != (3, 11):
        raise ValueError("formal snapshot analysis must use Python 3.11")
    repository = PROJECT_ROOT.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("formal snapshot lacks Git analysis provenance") from error
    if _COMMIT.fullmatch(commit) is None or status:
        raise ValueError("formal snapshot analysis requires a clean Git commit")
    return {
        "analysis_git_commit": commit,
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "analysis_python": sys.version,
    }


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    payload = _json(config_path)
    if not isinstance(payload, dict):
        raise ValueError("Exp22 config must be a JSON object")
    payload["config_path"] = str(config_path.resolve())
    return payload


def _planned_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    raw = config.get("seeds")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("Exp22 config seeds must be a sequence")
    result: list[int] = []
    for value in raw:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError("Exp22 seeds must be non-negative integers")
        result.append(int(value))
    if not result or len(set(result)) != len(result):
        raise ValueError("Exp22 seeds must be unique and non-empty")
    return tuple(result)


def _require_formal_registration(config: Mapping[str, Any]) -> None:
    if config.get("profile") != "formal":
        raise ValueError("Exp22 publication requires profile=formal")
    if _planned_seeds(config) != tuple(range(30)):
        raise ValueError(
            "Exp22 publication requires the 30 registered independent seeds"
        )
    network = config.get("network")
    if not isinstance(network, Mapping) or network.get("n_units") != 512:
        raise ValueError("Exp22 publication requires the registered N=512 network")


def _thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    raw = config.get("registered_claim_thresholds")
    required = {
        "minimum_aligned_gain_vs_frozen",
        "minimum_aligned_gain_vs_random",
        "minimum_aligned_gain_vs_shuffled",
        "maximum_aligned_oracle_gap",
        "minimum_budget_valid_fraction",
        "minimum_aligned_absolute_balanced_accuracy",
    }
    if not isinstance(raw, Mapping) or not required <= set(raw):
        raise ValueError("Exp22 config lacks registered claim thresholds")
    result: dict[str, float] = {}
    for name in required:
        value = raw[name]
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise ValueError(f"{name} must be a finite scalar")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{name} must be finite")
        result[name] = number
    if result["maximum_aligned_oracle_gap"] < 0.0:
        raise ValueError("maximum_aligned_oracle_gap must be non-negative")
    fraction = result["minimum_budget_valid_fraction"]
    if not 0.0 < fraction <= 1.0:
        raise ValueError("minimum_budget_valid_fraction must lie in (0, 1]")
    absolute = result["minimum_aligned_absolute_balanced_accuracy"]
    if not 0.0 <= absolute <= 1.0:
        raise ValueError(
            "minimum_aligned_absolute_balanced_accuracy must lie in [0, 1]"
        )
    return result


def _expected_run_config(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "seed": int(seed),
        **dict(config),
        "training_algorithm": (
            "dev_frozen_trajectory_three_factor_gain_axis_proposal_audit"
        ),
        "used_autograd": False,
        "used_bptt": False,
        "recurrent_learning": False,
        "gain_axis_learning": True,
    }


def _configs_match(observed: object, expected: Mapping[str, Any]) -> bool:
    if not isinstance(observed, Mapping):
        return False
    left, right = dict(observed), dict(expected)
    left_path = left.pop("config_path", None)
    right_path = right.pop("config_path", None)
    if left != right:
        return False
    if left_path is None or right_path is None:
        return left_path == right_path
    return Path(str(left_path)).name == Path(str(right_path)).name


def _expected_grid(config: Mapping[str, Any]) -> list[dict[str, object]]:
    return _planned_conditions(dict(config))


def _planned_without_index(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ValueError("planned_conditions.json must be a list of objects")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        row = dict(item)
        if row.pop("condition_index", None) != index:
            raise ValueError("planned condition indexes must be ordered")
        result.append(row)
    return result


def _latest_terminal_attempt(
    results_root: Path,
    *,
    seed: int,
    expected_config: Mapping[str, Any],
) -> Path:
    seed_root = results_root / "runs" / EXPERIMENT / f"seed_{seed:04d}"
    for attempt in sorted(seed_root.glob("*"), reverse=True):
        config_path = attempt / "config.json"
        status_path = attempt / "status.json"
        if (
            not config_path.is_file()
            or not status_path.is_file()
            or not _configs_match(_json(config_path), expected_config)
        ):
            continue
        status = _json(status_path)
        if isinstance(status, Mapping) and status.get("status") in TERMINAL_RUN_STATUS:
            return attempt
    raise FileNotFoundError(
        f"no terminal exact-config Exp22 attempt exists for seed={seed}"
    )


def _validate_attempt(
    attempt: Path,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    required = (
        "config.json",
        "environment.json",
        "manifest.json",
        "metrics.jsonl",
        "planned_conditions.json",
        "status.json",
    )
    missing = [name for name in required if not (attempt / name).is_file()]
    if missing:
        raise FileNotFoundError(f"seed {seed} attempt lacks artifacts: {missing}")
    if not _configs_match(
        _json(attempt / "config.json"), _expected_run_config(config, seed)
    ):
        raise ValueError(f"seed {seed} run config differs from Exp22 config")

    status = _json(attempt / "status.json")
    manifest = _json(attempt / "manifest.json")
    environment = _json(attempt / "environment.json")
    environment_sha256 = _environment_sha256(environment)
    git = environment.get("git")
    if (
        not isinstance(status, Mapping)
        or status.get("status") not in TERMINAL_RUN_STATUS
        or not isinstance(manifest, Mapping)
        or manifest.get("status") != status.get("status")
        or manifest.get("experiment") != EXPERIMENT
        or int(manifest.get("seed", -1)) != seed
        or manifest.get("profile") != "formal"
    ):
        raise ValueError(f"seed {seed} status/manifest contract is malformed")
    if (
        not isinstance(git, Mapping)
        or _COMMIT.fullmatch(str(git.get("commit", ""))) is None
        or git.get("dirty") is not False
    ):
        raise ValueError(f"seed {seed} formal attempt lacks a clean Git receipt")

    expected = _expected_grid(config)
    observed = _planned_without_index(_json(attempt / "planned_conditions.json"))
    if observed != expected:
        raise ValueError(f"seed {seed} planned Exp22 grid is malformed")
    rows = _records(attempt / "metrics.jsonl")
    if len(rows) != len(expected):
        raise ValueError(f"seed {seed} must retain all 11 Exp22 cells")
    expected_by_name = {str(item["condition"]): item for item in expected}
    names = [str(row.get("condition")) for row in rows]
    if len(set(names)) != len(names) or set(names) != set(expected_by_name):
        raise ValueError(f"seed {seed} has a duplicate or incomplete Exp22 grid")
    for row in rows:
        name = str(row.get("condition"))
        if (
            row.get("experiment") != EXPERIMENT
            or int(row.get("seed", -1)) != seed
            or row.get("run_id") != manifest.get("run_id")
            or row.get("status") not in ROW_STATUS
        ):
            raise ValueError(f"seed {seed} raw Exp22 identity is malformed")
        for field, value in expected_by_name[name].items():
            if row.get(field) != value:
                raise ValueError(f"seed {seed} condition {name} differs on {field}")
    failures = sum(row["status"] == "failed" for row in rows)
    invalid = sum(row["status"] == "invalid" for row in rows)
    expected_status = "complete_with_failures" if failures or invalid else "complete"
    if (
        status.get("status") != expected_status
        or int(status.get("condition_failures", failures)) != failures
        or int(status.get("condition_invalid", invalid)) != invalid
    ):
        raise ValueError(f"seed {seed} failure receipts disagree with raw rows")
    return [
        {
            **row,
            "attempt_path": str(attempt.resolve()),
            "run_status": str(status["status"]),
            "run_git_commit": str(git["commit"]),
            "environment_sha256": environment_sha256,
        }
        for row in rows
    ]


def collect_registered_runs(
    results_root: str | Path, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Collect every registered cell from one terminal attempt per seed."""

    _require_formal_registration(config)
    root = Path(results_root)
    rows: list[dict[str, Any]] = []
    for seed in _planned_seeds(config):
        attempt = _latest_terminal_attempt(
            root,
            seed=seed,
            expected_config=_expected_run_config(config, seed),
        )
        rows.extend(_validate_attempt(attempt, config=config, seed=seed))
    raw = (
        pd.DataFrame(rows)
        .sort_values(["seed", "condition"], kind="stable")
        .reset_index(drop=True)
    )
    validate_raw_frame(raw, config)
    if raw["run_git_commit"].astype(str).nunique(dropna=False) != 1:
        raise ValueError("Exp22 formal seeds use mixed Git commits")
    if raw["environment_sha256"].astype(str).nunique(dropna=False) != 1:
        raise ValueError("Exp22 formal seeds use mixed software environments")
    return raw


def _strict_bool(value: object, expected: bool) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value) is expected


def validate_raw_frame(raw: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Validate raw identity and the retained 11-cell factorial structure."""

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Exp22 raw table must be a non-empty DataFrame")
    required = {
        "experiment",
        "seed",
        "condition",
        "feedback_condition",
        "budget_norm",
        "third_factor_source",
        "status",
    }
    if not required <= set(raw):
        raise ValueError(
            f"Exp22 raw table lacks columns: {sorted(required - set(raw))}"
        )
    seeds = pd.to_numeric(raw["seed"], errors="coerce")
    if not np.isfinite(seeds).all() or not np.equal(seeds, np.floor(seeds)).all():
        raise ValueError("Exp22 seed values must be integers")
    observed_seeds = set(seeds.astype(int))
    if observed_seeds != set(_planned_seeds(config)):
        raise ValueError("Exp22 raw table does not retain every registered seed")
    if not raw["experiment"].eq(EXPERIMENT).all():
        raise ValueError("Exp22 raw table contains another experiment")
    if not raw["status"].isin(ROW_STATUS).all():
        raise ValueError("Exp22 raw table contains an unknown row status")
    duplicate = raw.assign(_seed=seeds.astype(int)).duplicated(["_seed", "condition"])
    if bool(duplicate.any()):
        raise ValueError("Exp22 raw table contains duplicate seed-condition rows")

    expected = _expected_grid(config)
    expected_by_name = {str(item["condition"]): item for item in expected}
    for seed, group in raw.assign(_seed=seeds.astype(int)).groupby("_seed", sort=True):
        if len(group) != len(expected) or set(group["condition"]) != set(
            expected_by_name
        ):
            raise ValueError(f"seed {seed} lacks the complete 11-cell grid")
        for row in group.to_dict(orient="records"):
            planned = expected_by_name[str(row["condition"])]
            for field, value in planned.items():
                if row.get(field) != value:
                    raise ValueError(
                        f"seed {seed} condition {row['condition']} "
                        f"differs on planned field {field}"
                    )


def _bool_field(row: Mapping[str, Any], field: str, expected: bool) -> bool:
    return _strict_bool(row.get(field), expected)


def _finite_number(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _seed_eligibility(raw: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Return panel/role capability receipts for every independent seed.

    Oracle or orthogonal failures do not invalidate the non-oracle primary
    panel.  The complete 11-cell *planned* grid is still required structurally
    by :func:`validate_raw_frame`; this function determines which observed
    comparisons are scientifically eligible.
    """

    budgets = dict(config["gain_axis_learning"]["budgets"])
    tolerance = float(config["gain_axis_learning"].get("budget_tolerance", 1e-9))
    common_true = (
        "base_conditions_share_readout",
        "gate_fit_train_only",
        "readout_fit_train_only",
        "gain_axis_fit_dev_only",
        "train_dev_test_episode_disjoint",
        "proposal_scale_predeclared_in_config",
        "off_policy_frozen_trajectory_proposal_audit",
    )
    learned_true = (
        "fixed_readout_feedback_coefficients_used",
        "gain_axis_three_factor_rule_used_for_eligibility",
        "feedback_transform_applied_before_local_eligibility",
        "budget_preserves_event_relative_magnitude",
    )
    common_false = (
        "used_autograd",
        "used_bptt",
        "recurrent_learning",
        "homeostasis_learning",
        "normalization_learning",
        "gate_fit_accessed_true_context",
        "gate_test_accessed_true_context",
        "axis_test_truth_accessed",
        "test_used_for_axis_fit",
        "proposal_scale_fit_on_test",
        "budget_controller_can_amplify_proposals",
        "budget_controller_used",
        "budget_matcher_can_amplify_proposals",
        "budget_simultaneous_dual_norm_match",
        "generic_recurrent_three_factor_claim_eligible",
        "gain_axis_local_plasticity_claim_eligible",
        "gain_axis_learning_closed_loop",
        "dev_trajectory_recomputed_after_each_update",
        "proposal_coordinate_permutation_after_eligibility",
        "closed_loop_local_plasticity_claim_eligible",
        "weight_transport_free_claim",
    )

    def assess(
        by_condition: Mapping[str, Mapping[str, Any]],
        *,
        panel: str,
        role: str,
    ) -> tuple[bool, str]:
        reasons: list[str] = []
        required = [
            "frozen_zero",
            f"aligned_local_{panel}",
            f"random_signed_feedback_{panel}",
            f"shuffled_feedback_{panel}",
        ]
        if role == "upper_bound":
            required.append(f"{ORACLE}_{panel}")
        elif role == "negative_control":
            required.append(f"{ORTHOGONAL}_{panel}")
        rows = [by_condition.get(name, {}) for name in required]
        learned_rows = [
            by_condition.get(name, {}) for name in required if name != "frozen_zero"
        ]
        if any(row.get("status") != "complete" for row in rows):
            reasons.append(f"incomplete_{role}_{panel}_cells")

        for field in PAIRING_IDS:
            values = [row.get(field) for row in rows]
            if (
                any(value is None or str(value) == "" for value in values)
                or len({str(value) for value in values}) != 1
            ):
                reasons.append(f"unshared_{field}")
        if any(
            not _bool_field(row, field, True) for row in rows for field in common_true
        ):
            reasons.append("required_train_dev_pairing_capability_missing")
        if any(
            not _bool_field(row, field, True)
            for row in learned_rows
            for field in learned_true
        ):
            reasons.append("required_local_proposal_capability_missing")
        if any(
            not _bool_field(row, field, False) for row in rows for field in common_false
        ):
            reasons.append("truth_bptt_or_learning_scope_violation")
        if any(row.get("statistics_unit") != "seed" for row in rows):
            reasons.append("statistics_unit_not_seed")
        if any(
            row.get("test_gain_control_source") != "learned_belief_posterior"
            for row in rows
        ):
            reasons.append("test_gain_not_driven_by_learned_belief")

        frozen = by_condition.get("frozen_zero", {})
        if (
            not _bool_field(frozen, "budget_attained", True)
            or not _bool_field(frozen, "gain_axis_learning", False)
            or _finite_number(frozen.get("budget_total")) != 0.0
            or frozen.get("budget_selected_norm") != "none"
            or _finite_number(frozen.get("budget_selected_raw")) != 0.0
            or not _missing(frozen.get("budget_global_scale_factor"))
            or frozen.get("budget_scaling_policy") != "none_zero_update_baseline"
            or not _missing(frozen.get("budget_path_application_id"))
            or not _bool_field(frozen, "frozen_zero_update_budget_baseline", True)
        ):
            reasons.append("frozen_control_contract_failed")

        learned_names = [name for name in required if name != "frozen_zero"]
        for name in learned_names:
            row = by_condition.get(name, {})
            total = _finite_number(row.get("budget_total"))
            raw_selected = _finite_number(row.get("budget_selected_raw"))
            applied = _finite_number(row.get("budget_selected_applied"))
            scale = _finite_number(row.get("budget_global_scale_factor"))
            target = float(budgets[panel])
            if (
                not _bool_field(row, "gain_axis_learning", True)
                or not _bool_field(row, "budget_attained", True)
                or not _bool_field(
                    row, "budget_secondary_norm_is_diagnostic_only", True
                )
                or not _bool_field(row, "recurrent_weights_bitwise_frozen", True)
                or row.get("budget_selected_norm") != panel
                or total is None
                or raw_selected is None
                or applied is None
                or scale is None
                or not np.isclose(total, target, atol=tolerance, rtol=0.0)
                or not np.isclose(applied, target, atol=tolerance, rtol=0.0)
                or raw_selected + tolerance < applied
                or not 0.0 <= scale <= 1.0
                or not np.isclose(
                    raw_selected * scale,
                    applied,
                    atol=max(tolerance, 1e-10),
                    rtol=1e-9,
                )
                or row.get("budget_scaling_policy")
                != ("single_global_downscale_preserves_event_relative_magnitude")
                or row.get("budget_path_application_id") in {None, ""}
                or not _bool_field(row, "frozen_zero_update_budget_baseline", False)
            ):
                reasons.append(f"unattained_or_mismatched_{panel}_budget")
                break
            feedback = str(row.get("feedback_condition", ""))
            other_panel = "l2" if panel == "l1" else "l1"
            counterpart = by_condition.get(f"{feedback}_{other_panel}", {})
            if counterpart.get("status") == "complete":
                for field in (
                    "condition_dev_local_tape_id",
                    "feedback_coefficients_id",
                    "feedback_policy",
                ):
                    left = row.get(field)
                    right = counterpart.get(field)
                    if (
                        left is None
                        or right is None
                        or str(left) == ""
                        or str(left) != str(right)
                    ):
                        reasons.append(f"{feedback}_cross_panel_{field}_not_reused")

        aligned_row = by_condition.get(f"aligned_local_{panel}", {})
        oracle_row = by_condition.get(f"{ORACLE}_{panel}", {})
        if role == "upper_bound" and (
            aligned_row.get("condition_dev_local_tape_id") is None
            or aligned_row.get("condition_dev_local_tape_id")
            != oracle_row.get("condition_dev_local_tape_id")
        ):
            reasons.append("oracle_did_not_reuse_aligned_local_tape")
        if role == "upper_bound":
            for field in ("feedback_coefficients_id", "feedback_policy"):
                if aligned_row.get(field) is None or aligned_row.get(
                    field
                ) != oracle_row.get(field):
                    reasons.append(f"oracle_did_not_reuse_aligned_{field}")
        policy_rows = [
            by_condition.get(f"{feedback}_{panel}", {})
            for feedback in (
                "aligned_local",
                "random_signed_feedback",
                "shuffled_feedback",
                ORTHOGONAL,
            )
        ]
        for field in (
            "condition_dev_local_tape_id",
            "feedback_coefficients_id",
            "feedback_policy",
        ):
            available = [
                str(row.get(field))
                for row in policy_rows
                if row.get("status") == "complete"
            ]
            if len(available) != len(set(available)):
                reasons.append(f"feedback_specific_{field}_not_independent")

        aligned_feedback_l2 = _finite_number(aligned_row.get("feedback_coefficient_l2"))
        for name in learned_names:
            coefficient_l2 = _finite_number(
                by_condition.get(name, {}).get("feedback_coefficient_l2")
            )
            if (
                aligned_feedback_l2 is None
                or coefficient_l2 is None
                or not np.isclose(
                    coefficient_l2,
                    aligned_feedback_l2,
                    atol=max(tolerance, 1e-10),
                    rtol=1e-9,
                )
            ):
                reasons.append("feedback_l2_magnitude_not_matched")
                break
        if role == "negative_control":
            orthogonal = by_condition.get(f"{ORTHOGONAL}_{panel}", {})
            for field in (
                "feedback_angle_to_aligned_degrees",
                "feedback_coefficient_angle_to_aligned_degrees",
                "feedback_sensory_angle_to_aligned_degrees",
                "feedback_delay_angle_to_aligned_degrees",
                "feedback_response_angle_to_aligned_degrees",
            ):
                angle = _finite_number(orthogonal.get(field))
                if angle is None or not np.isclose(angle, 90.0, atol=1e-8, rtol=0.0):
                    reasons.append(f"orthogonal_{field}_not_90_degrees")

        aligned = by_condition.get(f"aligned_local_{panel}", {})
        if not _bool_field(
            aligned, "gain_axis_off_policy_proposal_claim_eligible", True
        ):
            reasons.append(f"aligned_{panel}_off_policy_scope_ineligible")
        for name in learned_names:
            row = by_condition.get(name, {})
            expected = name == f"aligned_local_{panel}"
            if not _bool_field(
                row, "gain_axis_off_policy_proposal_claim_eligible", expected
            ):
                reasons.append("off_policy_claim_scope_malformed")
                break

        for name in required:
            row = by_condition.get(name, {})
            feedback = str(row.get("feedback_condition", ""))
            expected_truth = feedback == ORACLE
            if not _bool_field(row, "dev_truth_accessed_for_axis", expected_truth):
                reasons.append("development_truth_scope_violation")
                break
            condition_third_factor = row.get("condition_third_factor_id")
            learned_third_factor = row.get("learned_third_factor_id")
            if feedback == "frozen":
                if not _missing(condition_third_factor):
                    reasons.append("frozen_condition_third_factor_not_empty")
                    break
            elif expected_truth:
                if (
                    condition_third_factor in {None, ""}
                    or learned_third_factor in {None, ""}
                    or condition_third_factor == learned_third_factor
                ):
                    reasons.append("oracle_third_factor_receipt_malformed")
                    break
            elif (
                condition_third_factor in {None, ""}
                or condition_third_factor != learned_third_factor
            ):
                reasons.append("learned_third_factor_not_shared")
                break
        unique = tuple(dict.fromkeys(reasons))
        return not unique, ";".join(unique)

    receipts: list[dict[str, object]] = []
    for seed, group in raw.groupby("seed", sort=True):
        by_condition = {
            str(row["condition"]): row for row in group.to_dict(orient="records")
        }
        for panel in PANELS:
            for role in ("primary", "upper_bound", "negative_control"):
                eligible, reason = assess(by_condition, panel=panel, role=role)
                receipts.append(
                    {
                        "seed": int(seed),
                        "panel": panel,
                        "eligibility_role": role,
                        "eligible": eligible,
                        "ineligibility_reason": reason,
                    }
                )
    return pd.DataFrame(receipts).sort_values(
        ["seed", "panel", "eligibility_role"], kind="stable"
    )


def _bootstrap_interval(
    values: np.ndarray, *, n_bootstrap: int, seed: int
) -> tuple[float, float]:
    if n_bootstrap < 100:
        raise ValueError("Exp22 seed bootstrap requires at least 100 draws")
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(n_bootstrap), values.size), replace=True)
    low, high = np.quantile(np.mean(draws, axis=1), [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_p(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > np.finfo(np.float64).eps]
    if nonzero.size == 0:
        return 1.0
    positives = int(np.count_nonzero(nonzero > 0.0))
    smaller = min(positives, int(nonzero.size) - positives)
    tail = sum(math.comb(int(nonzero.size), k) for k in range(smaller + 1))
    return float(min(1.0, 2.0 * tail / (2 ** int(nonzero.size))))


def _exact_sign_p_greater(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > np.finfo(np.float64).eps]
    if nonzero.size == 0:
        return 1.0
    positives = int(np.count_nonzero(nonzero > 0.0))
    tail = sum(
        math.comb(int(nonzero.size), k) for k in range(positives, int(nonzero.size) + 1)
    )
    return float(tail / (2 ** int(nonzero.size)))


def _holm(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    running = 0.0
    ordered = finite[np.argsort(p[finite])]
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (len(finite) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def _paired_effect(
    complete: pd.DataFrame,
    eligible_seeds: set[int],
    *,
    panel: str,
    comparator: str,
) -> np.ndarray:
    aligned = f"aligned_local_{panel}"
    comparator_condition = (
        comparator if comparator == "frozen_zero" else f"{comparator}_{panel}"
    )
    selected = complete.loc[
        complete["seed"].astype(int).isin(eligible_seeds)
        & complete["condition"].isin([aligned, comparator_condition]),
        ["seed", "condition", "behavior_balanced_accuracy"],
    ].copy()
    selected["behavior_balanced_accuracy"] = pd.to_numeric(
        selected["behavior_balanced_accuracy"], errors="coerce"
    )
    pivot = selected.pivot(
        index="seed",
        columns="condition",
        values="behavior_balanced_accuracy",
    )
    if {aligned, comparator_condition} - set(pivot):
        return np.empty(0, dtype=np.float64)
    difference = pivot[aligned] - pivot[comparator_condition]
    difference = difference.replace([np.inf, -np.inf], np.nan).dropna()
    return difference.to_numpy(dtype=np.float64)


def _effect_row(
    *,
    panel: str,
    comparator: str,
    values: np.ndarray,
    threshold: float,
    n_planned: int,
    n_eligible: int,
    minimum_valid_fraction: float,
    n_bootstrap: int,
    index: int,
    family: str,
    role: str,
) -> dict[str, Any]:
    estimate = float(np.mean(values)) if values.size else float("nan")
    low, high = _bootstrap_interval(
        values,
        n_bootstrap=n_bootstrap,
        seed=220_022 + 104729 * index,
    )
    p_value = _exact_sign_p(values - threshold) if values.size else float("nan")
    return {
        "experiment": EXPERIMENT,
        "proposition": (
            "off_policy_gain_axis_proposal_behavior_advantage"
            if role == "primary"
            else "orthogonal_feedback_negative_control"
        ),
        "panel": panel,
        "comparison": f"aligned_local_{panel}_vs_{comparator}",
        "effect_definition": ("aligned_minus_comparator_heldout_balanced_accuracy"),
        "inference_unit": "seed",
        "multiplicity_family": family,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "holm_adjusted_p": float("nan"),
        "n_complete": int(values.size),
        "n_eligible": int(n_eligible),
        "n_planned": int(n_planned),
        "eligible_fraction": n_eligible / n_planned,
        "minimum_eligible_fraction": minimum_valid_fraction,
        "threshold": threshold,
        "conclusion": "inconclusive",
        "claim_scope": (
            "held-out behavior after a development-trajectory off-policy "
            "three-factor proposal; not closed-loop local learning and "
            "recurrent weights are bitwise frozen"
            if role == "primary"
            else (
                "fixed-feedback 90-degree orthogonal negative control; "
                "diagnostic only, truth-free, and never sufficient for the "
                "main claim"
            )
        ),
        "control_role": role,
    }


def _classify_holm_rows(
    rows: list[dict[str, Any]], *, minimum_valid_fraction: float
) -> None:
    adjusted = _holm([float(row["p_value"]) for row in rows])
    for row, p_holm in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = float(p_holm)
        sufficiently_complete = (
            row["eligible_fraction"] >= minimum_valid_fraction
            and row["n_complete"] == row["n_eligible"]
            and row["n_eligible"] > 0
        )
        if sufficiently_complete and row["ci_low"] > row["threshold"] and p_holm < 0.05:
            row["conclusion"] = "support"
        elif (
            sufficiently_complete
            and row["ci_high"] < row["threshold"]
            and p_holm < 0.05
        ):
            row["conclusion"] = "oppose"


def _oracle_row(
    complete: pd.DataFrame,
    eligible_seeds: set[int],
    *,
    panel: str,
    n_planned: int,
    n_eligible: int,
    minimum_valid_fraction: float,
    maximum_gap: float,
    n_bootstrap: int,
    index: int,
) -> dict[str, Any]:
    values = _paired_effect(complete, eligible_seeds, panel=panel, comparator=ORACLE)
    estimate = float(np.mean(values)) if values.size else float("nan")
    low, high = _bootstrap_interval(
        values,
        n_bootstrap=n_bootstrap,
        seed=2_220_022 + 104729 * index,
    )
    lower_margin = -float(maximum_gap)
    p_value = (
        _exact_sign_p_greater(values - lower_margin) if values.size else float("nan")
    )
    return {
        "experiment": EXPERIMENT,
        "proposition": "oracle_third_factor_upper_bound_gap",
        "panel": panel,
        "comparison": f"aligned_local_{panel}_vs_{ORACLE}",
        "effect_definition": "aligned_minus_oracle_heldout_balanced_accuracy",
        "inference_unit": "seed",
        "multiplicity_family": ORACLE_FAMILY,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "holm_adjusted_p": float("nan"),
        "n_complete": int(values.size),
        "n_eligible": int(n_eligible),
        "n_planned": int(n_planned),
        "eligible_fraction": n_eligible / n_planned,
        "minimum_eligible_fraction": minimum_valid_fraction,
        "threshold": lower_margin,
        "conclusion": "inconclusive",
        "claim_scope": (
            "development-hidden-context oracle third factor is an upper bound; "
            "this margin result cannot establish the local mechanism alone"
        ),
        "control_role": "upper_bound",
    }


def _classify_oracle_rows(
    rows: list[dict[str, Any]], *, minimum_valid_fraction: float
) -> None:
    adjusted = _holm([float(row["p_value"]) for row in rows])
    for row, p_holm in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = float(p_holm)
        enough = (
            row["eligible_fraction"] >= minimum_valid_fraction
            and row["n_complete"] == row["n_eligible"]
            and row["n_eligible"] > 0
        )
        if enough and row["ci_low"] > row["threshold"] and p_holm < 0.05:
            row["conclusion"] = "support"
        elif enough and row["ci_high"] < row["threshold"]:
            row["conclusion"] = "oppose"


def _joint_row(
    panel: str,
    primary: Sequence[Mapping[str, Any]],
    absolute_balanced: Mapping[str, Any],
    oracle_margin: Mapping[str, Any],
    *,
    n_planned: int,
    n_eligible: int,
    minimum_valid_fraction: float,
) -> dict[str, Any]:
    by_comparison = {
        str(row["comparison"]).rsplit("_vs_", 1)[-1]: row for row in primary
    }
    conclusions = {
        name: str(by_comparison[name]["conclusion"]) for name in CORE_COMPARATORS
    }
    absolute_conclusion = str(absolute_balanced["conclusion"])
    oracle_conclusion = str(oracle_margin["conclusion"])
    enough = n_eligible / n_planned >= minimum_valid_fraction
    if (
        enough
        and all(value == "support" for value in conclusions.values())
        and absolute_conclusion == "support"
        and oracle_conclusion == "support"
    ):
        conclusion = "support"
    elif enough and (
        any(value == "oppose" for value in conclusions.values())
        or absolute_conclusion == "oppose"
        or oracle_conclusion == "oppose"
    ):
        conclusion = "oppose"
    else:
        conclusion = "inconclusive"
    return {
        "experiment": EXPERIMENT,
        "proposition": "joint_off_policy_proposal_alignment_specificity",
        "panel": panel,
        "comparison": (
            f"aligned_local_{panel}_joint_vs_frozen_random_shuffled_and_oracle_margin"
        ),
        "effect_definition": (
            "all_three_registered_aligned_superiority_contrasts_plus_absolute_"
            "balanced_accuracy_and_registered_oracle_noninferiority_margin"
        ),
        "inference_unit": "seed",
        "multiplicity_family": PRIMARY_FAMILY,
        "estimate": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": float("nan"),
        "holm_adjusted_p": float("nan"),
        "n_complete": int(n_eligible),
        "n_eligible": int(n_eligible),
        "n_planned": int(n_planned),
        "eligible_fraction": n_eligible / n_planned,
        "minimum_eligible_fraction": minimum_valid_fraction,
        "threshold": (
            "aligned exceeds frozen, random-signed, and shuffled registered "
            "thresholds and independently meets the registered absolute "
            "balanced-accuracy threshold and registered oracle noninferiority "
            "margin; frozen and shuffled must both support"
        ),
        "conclusion": conclusion,
        "claim_scope": (
            "joint held-out behavior specificity for a frozen-trajectory "
            "off-policy gain-axis proposal that is also within the registered "
            "oracle margin; explicitly not closed-loop local learning or "
            "recurrent plasticity"
        ),
        "control_role": "joint_primary",
    }


def _absolute_row(
    complete: pd.DataFrame,
    eligible_seeds: set[int],
    *,
    panel: str,
    metric: str,
    n_planned: int,
    n_bootstrap: int,
    index: int,
    threshold: float | None,
) -> dict[str, Any]:
    selected = complete.loc[
        complete["seed"].astype(int).isin(eligible_seeds)
        & complete["condition"].eq(f"aligned_local_{panel}"),
        ["seed", metric],
    ].copy()
    values = pd.to_numeric(selected[metric], errors="coerce")
    values = (
        values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
    )
    estimate = float(np.mean(values)) if values.size else float("nan")
    low, high = _bootstrap_interval(
        values,
        n_bootstrap=n_bootstrap,
        seed=22_220_022 + 104729 * index,
    )
    registered = threshold is not None
    p_value = (
        _exact_sign_p(values - float(threshold))
        if registered and values.size
        else float("nan")
    )
    return {
        "experiment": EXPERIMENT,
        "proposition": (
            "aligned_absolute_balanced_accuracy"
            if registered
            else "aligned_absolute_accuracy_descriptive"
        ),
        "panel": panel,
        "comparison": f"aligned_local_{panel}_{metric}",
        "effect_definition": f"aligned_heldout_{metric}",
        "inference_unit": "seed",
        "multiplicity_family": (
            ABSOLUTE_FAMILY
            if registered
            else "none_descriptive_no_registered_threshold"
        ),
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "holm_adjusted_p": float("nan"),
        "n_complete": int(values.size),
        "n_eligible": int(len(eligible_seeds)),
        "n_planned": int(n_planned),
        "eligible_fraction": len(eligible_seeds) / n_planned,
        "minimum_eligible_fraction": float("nan"),
        "threshold": (float(threshold) if registered else "none_registered"),
        "conclusion": "inconclusive",
        "claim_scope": (
            "absolute held-out aligned balanced accuracy with an independent "
            "registered threshold; never combined with relative performance "
            "through an OR rule"
            if registered
            else (
                "absolute held-out aligned accuracy reported separately from "
                "relative contrasts; descriptive because no accuracy threshold "
                "was registered"
            )
        ),
        "control_role": (
            "absolute_registered" if registered else "absolute_descriptive"
        ),
    }


def summarize_formal_runs(
    raw: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    """Summarize separate L1/L2 seed-level matched-budget panels."""

    validate_raw_frame(raw, config)
    thresholds = _thresholds(config)
    receipts = _seed_eligibility(raw, config)
    n_planned = len(_planned_seeds(config))
    minimum_fraction = thresholds["minimum_budget_valid_fraction"]
    complete = raw.loc[raw["status"].eq("complete")].copy()

    primary_rows: list[dict[str, Any]] = []
    orthogonal_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    absolute_rows: list[dict[str, Any]] = []
    threshold_by_comparator = {
        "frozen_zero": thresholds["minimum_aligned_gain_vs_frozen"],
        "random_signed_feedback": thresholds["minimum_aligned_gain_vs_random"],
        "shuffled_feedback": thresholds["minimum_aligned_gain_vs_shuffled"],
    }
    index = 0
    for panel in PANELS:
        primary_eligible_seeds = set(
            receipts.loc[
                receipts["panel"].eq(panel)
                & receipts["eligibility_role"].eq("primary")
                & receipts["eligible"],
                "seed",
            ].astype(int)
        )
        n_primary_eligible = len(primary_eligible_seeds)
        for comparator in CORE_COMPARATORS:
            values = _paired_effect(
                complete,
                primary_eligible_seeds,
                panel=panel,
                comparator=comparator,
            )
            primary_rows.append(
                _effect_row(
                    panel=panel,
                    comparator=comparator,
                    values=values,
                    threshold=threshold_by_comparator[comparator],
                    n_planned=n_planned,
                    n_eligible=n_primary_eligible,
                    minimum_valid_fraction=minimum_fraction,
                    n_bootstrap=n_bootstrap,
                    index=index,
                    family=PRIMARY_FAMILY,
                    role="primary",
                )
            )
            index += 1
        orthogonal_eligible_seeds = set(
            receipts.loc[
                receipts["panel"].eq(panel)
                & receipts["eligibility_role"].eq("negative_control")
                & receipts["eligible"],
                "seed",
            ].astype(int)
        )
        values = _paired_effect(
            complete,
            orthogonal_eligible_seeds,
            panel=panel,
            comparator=ORTHOGONAL,
        )
        orthogonal_rows.append(
            _effect_row(
                panel=panel,
                comparator=ORTHOGONAL,
                values=values,
                threshold=0.0,
                n_planned=n_planned,
                n_eligible=len(orthogonal_eligible_seeds),
                minimum_valid_fraction=minimum_fraction,
                n_bootstrap=n_bootstrap,
                index=index,
                family=NEGATIVE_CONTROL_FAMILY,
                role="negative_control",
            )
        )
        index += 1
        oracle_eligible_seeds = set(
            receipts.loc[
                receipts["panel"].eq(panel)
                & receipts["eligibility_role"].eq("upper_bound")
                & receipts["eligible"],
                "seed",
            ].astype(int)
        )
        oracle_rows.append(
            _oracle_row(
                complete,
                oracle_eligible_seeds,
                panel=panel,
                n_planned=n_planned,
                n_eligible=len(oracle_eligible_seeds),
                minimum_valid_fraction=minimum_fraction,
                maximum_gap=thresholds["maximum_aligned_oracle_gap"],
                n_bootstrap=n_bootstrap,
                index=index,
            )
        )
        index += 1
        for metric in ("behavior_accuracy", "behavior_balanced_accuracy"):
            absolute_threshold = (
                thresholds["minimum_aligned_absolute_balanced_accuracy"]
                if metric == "behavior_balanced_accuracy"
                else None
            )
            absolute_rows.append(
                _absolute_row(
                    complete,
                    primary_eligible_seeds,
                    panel=panel,
                    metric=metric,
                    n_planned=n_planned,
                    n_bootstrap=n_bootstrap,
                    index=index,
                    threshold=absolute_threshold,
                )
            )
            index += 1

    _classify_holm_rows(primary_rows, minimum_valid_fraction=minimum_fraction)
    _classify_holm_rows(orthogonal_rows, minimum_valid_fraction=minimum_fraction)
    _classify_oracle_rows(oracle_rows, minimum_valid_fraction=minimum_fraction)
    _classify_holm_rows(
        [row for row in absolute_rows if row["control_role"] == "absolute_registered"],
        minimum_valid_fraction=minimum_fraction,
    )
    joint_rows = [
        _joint_row(
            panel,
            [row for row in primary_rows if row["panel"] == panel],
            next(
                row
                for row in absolute_rows
                if row["panel"] == panel
                and row["control_role"] == "absolute_registered"
            ),
            next(row for row in oracle_rows if row["panel"] == panel),
            n_planned=n_planned,
            n_eligible=int(
                receipts.loc[
                    receipts["panel"].eq(panel)
                    & receipts["eligibility_role"].eq("primary"),
                    "eligible",
                ].sum()
            ),
            minimum_valid_fraction=minimum_fraction,
        )
        for panel in PANELS
    ]
    summary = pd.DataFrame(
        [
            *primary_rows,
            *oracle_rows,
            *orthogonal_rows,
            *absolute_rows,
            *joint_rows,
        ]
    )
    summary["retained_failed_or_invalid_cell_count"] = int(
        np.count_nonzero(~raw["status"].eq("complete"))
    )
    summary["retained_failed_or_invalid_seed_count"] = int(
        raw.loc[~raw["status"].eq("complete"), "seed"].nunique()
    )
    summary["ineligible_seed_count"] = int(
        receipts.loc[
            receipts["eligibility_role"].eq("primary") & ~receipts["eligible"],
            "seed",
        ].nunique()
    )
    return summary


def _csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column] = result[column].map(
            lambda value: (
                json.dumps(value, sort_keys=True, ensure_ascii=False)
                if isinstance(value, (dict, list, tuple))
                else value
            )
        )
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            return ""
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join((header, rule, *body))


def _plot(
    summary: pd.DataFrame,
    *,
    png: Path,
    pdf: Path,
) -> None:
    labels = {
        "frozen_zero": "Frozen zero axis",
        "random_signed_feedback": "Random signed",
        "shuffled_feedback": "Shuffled",
        ORACLE: "Oracle upper bound",
        ORTHOGONAL: "Orthogonal (90 deg)",
    }
    colors = {
        "support": "#0072B2",
        "oppose": "#D55E00",
        "inconclusive": "#7F7F7F",
    }
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
        for panel_index, (axis, panel) in enumerate(zip(axes, PANELS, strict=True)):
            panel_rows = summary.loc[
                summary["panel"].eq(panel)
                & summary["control_role"].isin(
                    ["primary", "upper_bound", "negative_control"]
                )
            ].copy()
            order = [
                "frozen_zero",
                "random_signed_feedback",
                "shuffled_feedback",
                ORACLE,
                ORTHOGONAL,
            ]
            panel_rows["_comparator"] = panel_rows["comparison"].map(
                lambda value: str(value).rsplit("_vs_", 1)[-1]
            )
            panel_rows["_order"] = panel_rows["_comparator"].map(order.index)
            panel_rows = panel_rows.sort_values("_order", kind="stable")
            y = np.arange(len(panel_rows))
            for row_index, row in enumerate(panel_rows.itertuples()):
                center = float(row.estimate)
                low = float(row.ci_low)
                high = float(row.ci_high)
                if np.isfinite([center, low, high]).all():
                    axis.errorbar(
                        center,
                        y[row_index],
                        xerr=np.asarray(
                            [
                                [max(0.0, center - low)],
                                [max(0.0, high - center)],
                            ]
                        ),
                        fmt="s" if row.control_role != "primary" else "o",
                        color=colors[str(row.conclusion)],
                        capsize=3,
                        markersize=5,
                    )
            axis.set_yticks(
                y,
                [labels[str(value)] for value in panel_rows["_comparator"].tolist()],
            )
            axis.set_ylim(len(panel_rows) - 0.5, -0.5)
            axis.axvline(0.0, color="0.25", linewidth=0.8, linestyle="--")
            axis.set_xlabel(
                "Aligned - comparator held-out balanced accuracy (95% seed CI)"
            )
            axis.set_title(
                f"{chr(ord('a') + panel_index)}  {panel.upper()} budget panel",
                loc="left",
            )
            axis.spines[["top", "right"]].set_visible(False)
            joint = summary.loc[
                summary["panel"].eq(panel) & summary["control_role"].eq("joint_primary")
            ].iloc[0]
            axis.text(
                0.0,
                -0.22,
                (
                    f"Joint local-axis claim: "
                    f"{str(joint['conclusion']).upper()}  "
                    f"({int(joint['n_eligible'])}/{int(joint['n_planned'])} "
                    "eligible seeds)"
                ),
                transform=axis.transAxes,
                fontsize=8,
            )
        figure.savefig(
            png,
            dpi=300,
            bbox_inches="tight",
            metadata={"Software": "Exp22 deterministic matplotlib summary"},
        )
        figure.savefig(
            pdf,
            bbox_inches="tight",
            metadata={
                "Creator": "Exp22 deterministic matplotlib summary",
                "Producer": "Matplotlib",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(figure)


def write_snapshot_artifacts(
    raw: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    prefix: str = DEFAULT_PREFIX,
    n_bootstrap: int = 5000,
    publication_provenance: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write deterministic standalone raw, summary, report, and figures."""

    if prefix == DEFAULT_PREFIX:
        _require_formal_registration(config)
        if publication_provenance is None:
            raise ValueError(
                "the formal Exp22 prefix requires validated publication provenance"
            )
    validate_raw_frame(raw, config)
    raw_sorted = raw.sort_values(["seed", "condition"], kind="stable").reset_index(
        drop=True
    )
    receipts = _seed_eligibility(raw_sorted, config)
    receipt_wide_rows: list[dict[str, object]] = []
    for seed, group in receipts.groupby("seed", sort=True):
        item: dict[str, object] = {"seed": int(seed)}
        for receipt in group.to_dict(orient="records"):
            receipt_prefix = f"summary_{receipt['panel']}_{receipt['eligibility_role']}"
            item[f"{receipt_prefix}_eligible"] = receipt["eligible"]
            item[f"{receipt_prefix}_ineligibility_reason"] = receipt[
                "ineligibility_reason"
            ]
        receipt_wide_rows.append(item)
    raw_published = raw_sorted.merge(
        pd.DataFrame(receipt_wide_rows), on="seed", how="left"
    )
    summary = summarize_formal_runs(raw_sorted, config, n_bootstrap=n_bootstrap)
    provenance_lines = [
        "- Publication provenance: unvalidated non-formal helper snapshot."
    ]
    if publication_provenance is not None:
        validated_provenance = _validate_publication_provenance(publication_provenance)
        if (
            "run_git_commit" not in raw_sorted
            or "environment_sha256" not in raw_sorted
            or raw_sorted["run_git_commit"].astype(str).nunique(dropna=False) != 1
            or raw_sorted["environment_sha256"].astype(str).nunique(dropna=False) != 1
        ):
            raise ValueError("Exp22 formal raw provenance is incomplete")
        for field, value in validated_provenance.items():
            summary[field] = value
        provenance_lines = [
            (
                "- Raw-run commit: "
                f"`{raw_sorted['run_git_commit'].astype(str).iloc[0]}`."
            ),
            (
                "- Raw software-environment SHA-256: "
                f"`{raw_sorted['environment_sha256'].astype(str).iloc[0]}`."
            ),
            (
                "- Analysis commit: "
                f"`{validated_provenance['analysis_git_commit']}`; "
                "analysis-script SHA-256: "
                f"`{validated_provenance['analysis_script_sha256']}`."
            ),
            f"- Analysis Python: `{validated_provenance['analysis_python']}`.",
        ]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": target / f"{prefix}_raw.csv.gz",
        "summary": target / f"{prefix}_summary.csv",
        "report": target / f"{prefix}_report.md",
        "png": target / f"{prefix}.png",
        "pdf": target / f"{prefix}.pdf",
    }
    _csv_safe(raw_published).to_csv(
        paths["raw"],
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    _csv_safe(summary).to_csv(paths["summary"], index=False)

    failures = raw_sorted.loc[~raw_sorted["status"].eq("complete")].copy()
    failure_columns = [
        field
        for field in (
            "seed",
            "condition",
            "status",
            "error_type",
            "error",
            "reason",
        )
        if field in failures
    ]
    ineligible = receipts.loc[~receipts["eligible"]]
    report = [
        "# Exp22: local hidden-context gain-axis audit",
        "",
        (
            "This standalone snapshot reports development-only postsynaptic "
            "gain-axis proposals derived off-policy from a frozen neutral "
            "trajectory and evaluated on a frozen high-rank Dale E/I receiver."
        ),
        "",
        f"- Registered independent seeds: {len(_planned_seeds(config))}",
        (
            "- Eligible primary paired seeds: "
            + ", ".join(
                f"{panel.upper()}="
                f"{int(receipts.loc[receipts['panel'].eq(panel) & receipts['eligibility_role'].eq('primary'), 'eligible'].sum())}"
                for panel in PANELS
            )
        ),
        (
            "- Retained failed/invalid cells: "
            f"{int(np.count_nonzero(~raw_sorted['status'].eq('complete')))}"
        ),
        *provenance_lines,
        (
            "- Eligibility requires the structurally complete 11-cell grid, "
            "shared base network/gate/readout/split/neutral-trajectory receipts, "
            "reused feedback-specific tapes across L1/L2, attained selected-norm "
            "budgets, and no test truth, BPTT, autograd, or recurrent learning."
        ),
        (
            "- L1 and L2 are separate panels. Aligned, random-signed, shuffled, "
            "orthogonal, and oracle proposal cells are selected-norm "
            "budget-matched within each panel; frozen_zero has a zero update "
            "budget and is an on/off baseline, not a matched-budget cell. The "
            "unselected norm is diagnostic only."
        ),
        (
            "- Oracle-third-factor and truth-free 90-degree orthogonal cells "
            "are respectively an upper bound and a negative control. Neither "
            "can establish the main proposal-alignment claim."
        ),
        (
            "- Recurrent weights are frozen. Proposals are computed off-policy "
            "from frozen development trajectories, so no conclusion is a claim "
            "of online local plasticity, recurrent plasticity, or "
            "weight-transport freedom."
        ),
        (
            "- Exp22 contains no tuned BPTT, GRU, full-feedback, or recurrent-"
            "plasticity on/off baseline. It therefore cannot close the P0 "
            "acceptance criteria."
        ),
        (
            "- Aligned absolute accuracy and balanced accuracy are reported "
            "with seed-level confidence intervals separately from relative "
            "contrasts. Balanced accuracy has its own registered threshold and "
            "must support, together with all three relative primary contrasts, "
            "and the registered oracle noninferiority margin for the joint claim "
            "to support. Raw accuracy has no registered threshold and remains "
            "descriptive/inconclusive."
        ),
        "",
        "## Conclusions",
        "",
        _markdown_table(
            summary[
                [
                    "proposition",
                    "panel",
                    "comparison",
                    "estimate",
                    "ci_low",
                    "ci_high",
                    "n_eligible",
                    "n_planned",
                    "conclusion",
                    "claim_scope",
                ]
            ]
        ),
        "",
        "## Ineligible seeds",
        "",
        (_markdown_table(ineligible) if len(ineligible) else "None."),
        "",
        "## Retained failed or invalid cells",
        "",
        (_markdown_table(failures[failure_columns]) if len(failures) else "None."),
        "",
        "## Classification rule",
        "",
        textwrap.fill(
            "Held-out balanced-accuracy contrasts are paired within seed. The "
            "three primary comparisons per panel use deterministic seed "
            "bootstrap intervals, exact sign tests, and one Holm family across "
            "both panels. The joint claim can support only when the aligned "
            "off-policy proposal supports all comparisons against frozen, "
            "random-signed, and shuffled feedback, meets the registered absolute "
            "balanced-accuracy threshold, and supports the registered oracle "
            "noninferiority margin; in particular, failure to beat frozen or "
            "shuffled can never support it. Oracle and orthogonal failures do not "
            "invalidate otherwise eligible non-oracle comparisons, but an "
            "unavailable or inconclusive oracle margin leaves the joint claim "
            "inconclusive. These controls remain bounded diagnostics and cannot "
            "establish the mechanism alone. Failed and scientifically ineligible "
            "seeds remain visible in the raw snapshot. The registered absolute "
            "balanced-accuracy threshold, relative primary claims, and oracle "
            "margin are joined through AND, never OR.",
            width=96,
        ),
        "",
    ]
    paths["report"].write_text("\n".join(report), encoding="utf-8")
    _plot(summary, png=paths["png"], pdf=paths["pdf"])
    return paths


def publish_snapshot(
    results_root: str | Path,
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    prefix: str = DEFAULT_PREFIX,
    n_bootstrap: int = 5000,
) -> dict[str, Path]:
    _require_formal_registration(config)
    provenance = _analysis_provenance()
    raw = collect_registered_runs(results_root, config)
    return write_snapshot_artifacts(
        raw,
        config,
        output_dir=output_dir,
        prefix=prefix,
        n_bootstrap=n_bootstrap,
        publication_provenance=provenance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    config = _load_config(args.config)
    paths = publish_snapshot(
        args.results_root,
        config,
        output_dir=args.output_dir,
        prefix=args.prefix,
        n_bootstrap=args.n_bootstrap,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
