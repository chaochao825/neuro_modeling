"""Preregistered feasibility-aware confirmatory Exp26 source panel.

The evaluation seeds 60--89 are a fresh panel.  This module performs no
selector fitting and no statistical inference.  Every registered actuator
cell is retained.  An active actuator that cannot satisfy the preregistered
functional budget or stability constraint is terminally ``infeasible`` and
deploys the exact same-cell frozen utility; it is never dropped or imputed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp26_actuator_phase_diagram as exp26
from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.actuator_manifest import GeneratorCell, manifest_hash
from src.models.task_matched_actuators import ActuatorFitError
from src.utils.artifacts import ExperimentRun


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp29_confirmatory_source_panel"
PROFILE = "confirmatory_test"
PROTOCOL_VERSION = "exp29_confirmatory_source_v1"
EVIDENCE_SCHEMA_VERSION = "exp29_confirmatory_source_evidence_v1"
REQUIRED_RUN_LABEL = "exp29-confirmatory-source-v1"
REGISTERED_CONFIG_CANONICAL_SHA256 = (
    "70db02c9e578ace8a1719e4ff6c71c07048a1836f2dafc9fcd845ad5b0bd9e14"
)
REGISTERED_CONFIG_HASH_SENTINEL = "<EXP29_REGISTERED_CONFIG_CANONICAL_SHA256>"
IMPLEMENTATION_HASH_SCHEME = "runner_config_literal_sentinel_v1"
RUNNER_RELATIVE_PATH = "experiments/exp29_confirmatory_source_panel.py"
PACKAGER_RELATIVE_PATH = "scripts/package_exp29_confirmatory_source_panel.py"
ADAPTER_RELATIVE_PATH = "src/data/exp29_feasibility_selector_dataset.py"
IMPLEMENTATION_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    PACKAGER_RELATIVE_PATH,
    ADAPTER_RELATIVE_PATH,
)
EXPECTED_SEEDS = tuple(range(60, 90))
META_TRAINING_SEEDS = tuple(range(30))
EXPECTED_GENERATORS = 88
EXPECTED_MODES = tuple(exp26.MODES)
EXPECTED_ROWS_PER_SEED = EXPECTED_GENERATORS * len(EXPECTED_MODES)
EXPECTED_PANEL_ROWS = len(EXPECTED_SEEDS) * EXPECTED_ROWS_PER_SEED
MAX_SCALE = 256.0
SOURCE_CONFIG_RELATIVE_PATH = "configs/formal/exp26_actuator_phase_diagram.json"
SOURCE_PREFLIGHT_RELATIVE_PATH = (
    "results/exp26_actuator_matching_formal_v2_e08beaf/preflight"
)
SOURCE_PUBLISHED_MANIFEST_RELATIVE_PATH = (
    "results/exp26_actuator_matching_formal_v2_e08beaf/MANIFEST.sha256"
)
GIT_OBJECT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_RUNTIME_CONFIG_KEYS = {
    "config_path",
    "evidence_provenance",
    "experiment",
    "run_label",
    "seed",
}
_COPIED_SOURCE_KEYS = (
    "training_algorithm",
    "used_autograd",
    "used_bptt",
    "manifest",
    "carrier",
    "task",
    "analysis",
)
_EXPECTED_FEASIBILITY_POLICY = {
    "version": "exp29_frozen_fallback_v1",
    "terminal_statuses": ["complete", "infeasible", "failed", "invalid"],
    "infeasible_causes": [
        "budget_scale_above_cap",
        "degenerate_actuator",
        "budget_mismatch",
        "effective_instability",
    ],
    "infeasible_rows_retained": True,
    "infeasible_rows_deleted_or_imputed": False,
    "deployment_fallback_mode": "frozen",
    "infeasible_selected_utility": "same_cell_same_split_frozen_utility",
    "selector_candidate_modes": ["routing", "gain", "low_rank"],
    "nonselector_control_mode": "rgl",
    "oracle_choice_set": "frozen_plus_feasible_selector_candidate_modes",
    "matched_budget_support_requires_feasible_active_row": True,
    "infeasible_rate_reporting": "seed_by_actuator_family",
    "unconditional_inference": True,
}


@dataclass(frozen=True)
class SourceContract:
    config_sha256: str
    config_file_sha256: str
    source_contract_sha256: str
    source_config_sha256: str
    source_config_file_sha256: str
    source_manifest_sha256: str
    source_preflight_receipt_sha256: str
    source_critical_code_sha256: str
    implementation_file_sha256: Mapping[str, str]
    implementation_contract_sha256: str
    current_git: Mapping[str, object]
    runtime_versions: Mapping[str, str]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_runner_source_sha256(path: Path) -> str:
    """Hash runner source after replacing only its registered-config literal."""

    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read Exp29 runner source {path}") from error
    pattern = re.compile(
        r"(?P<prefix>REGISTERED_CONFIG_CANONICAL_SHA256\s*=\s*\(\s*[\"'])"
        r"(?P<digest>[0-9a-f]{64})"
        r"(?P<suffix>[\"']\s*\))"
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError("Exp29 runner must contain one registered-config literal")
    match = matches[0]
    normalized = (
        source[: match.start("digest")]
        + REGISTERED_CONFIG_HASH_SENTINEL
        + source[match.end("digest") :]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def exp29_implementation_sha256() -> dict[str, str]:
    """Return the registered normalized/raw implementation source digests."""

    return {
        RUNNER_RELATIVE_PATH: _normalized_runner_source_sha256(
            PROJECT_ROOT / RUNNER_RELATIVE_PATH
        ),
        PACKAGER_RELATIVE_PATH: _file_sha256(PROJECT_ROOT / PACKAGER_RELATIVE_PATH),
        ADAPTER_RELATIVE_PATH: _file_sha256(PROJECT_ROOT / ADAPTER_RELATIVE_PATH),
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    return {
        str(key): value
        for key, value in config.items()
        if str(key) not in _RUNTIME_CONFIG_KEYS
    }


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    return _canonical_sha256(canonical_config_payload(config))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read registered JSON object {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"registered JSON must be an object: {path}")
    return dict(value)


def _exact_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _config_source_file(config: Mapping[str, Any]) -> Path:
    value = config.get("config_path")
    if not isinstance(value, str) or not value:
        raise ValueError("confirmatory config requires config_path provenance")
    path = Path(value)
    source = _read_object(path)
    if canonical_config_payload(source) != canonical_config_payload(config):
        raise ValueError("runtime confirmatory config differs from its file")
    return path


def _validate_git(value: Mapping[str, object]) -> dict[str, object]:
    commit = value.get("commit")
    tree = value.get("tree")
    if (
        not isinstance(commit, str)
        or not GIT_OBJECT_PATTERN.fullmatch(commit)
        or not isinstance(tree, str)
        or not GIT_OBJECT_PATTERN.fullmatch(tree)
        or value.get("dirty") is not False
    ):
        raise ValueError("confirmatory attempts require a clean identifiable git tree")
    return {"commit": commit, "tree": tree, "dirty": False}


def _validate_runtime(
    config: Mapping[str, Any], value: Mapping[str, object]
) -> dict[str, str]:
    policy = config.get("runtime_policy")
    required = ["numpy", "scipy", "pandas", "scikit_learn", "statsmodels"]
    if (
        not isinstance(policy, Mapping)
        or policy.get("python_major_minor") != "3.11"
        or policy.get("required_distributions") != required
        or policy.get("exact_match_across_seeds") is not True
    ):
        raise ValueError("confirmatory runtime policy is not registered")
    python = value.get("python")
    if not isinstance(python, str) or not python.startswith("3.11."):
        raise ValueError("confirmatory source requires Python 3.11")
    resolved = {"python": python}
    for name in required:
        version = value.get(name)
        if not isinstance(version, str) or not version:
            raise ValueError(f"confirmatory runtime lacks {name}")
        resolved[name] = version
    return resolved


def validate_source_contract(
    config: Mapping[str, Any],
    *,
    current_git: Mapping[str, object] | None = None,
    runtime_versions: Mapping[str, object] | None = None,
) -> SourceContract:
    """Validate the immutable preregistration without accessing seeds 60--89."""

    if canonical_config_sha256(config) != REGISTERED_CONFIG_CANONICAL_SHA256:
        raise ValueError("confirmatory config hash is not registered")
    if (
        config.get("profile") != PROFILE
        or config.get("dev_only") is not False
        or config.get("protocol_version") != PROTOCOL_VERSION
        or config.get("required_run_label") != REQUIRED_RUN_LABEL
        or config.get("evidence_role") != "confirmatory_test_source_only"
        or config.get("standalone_inference_permitted") is not False
        or config.get("training_algorithm")
        != "train_only_closed_form_task_matched_actuators"
        or config.get("used_autograd") is not False
        or config.get("used_bptt") is not False
    ):
        raise ValueError("confirmatory source profile/protocol is invalid")
    seeds = tuple(_exact_integer(seed, name="seed") for seed in config["seeds"])
    preregistration = config.get("preregistration")
    if (
        seeds != EXPECTED_SEEDS
        or not isinstance(preregistration, Mapping)
        or tuple(preregistration.get("meta_training_seeds", ())) != META_TRAINING_SEEDS
        or tuple(preregistration.get("evaluation_seeds", ())) != EXPECTED_SEEDS
        or preregistration.get(
            "evaluation_seed_performance_or_reachability_read_before_registration"
        )
        is not False
        or preregistration.get("ceiling_retuning_after_registration_permitted")
        is not False
        or preregistration.get("selective_rerun_permitted") is not False
        or preregistration.get("missing_or_failed_attempt_replacement_permitted")
        is not False
        or preregistration.get("max_scale_selection")
        != "fixed_256_before_seed_60_89_access"
        or preregistration.get("statistics_unit") != "seed"
        or preregistration.get("primary_scope")
        != "unconditional_all_registered_heldout_cells"
    ):
        raise ValueError("confirmatory seed-access preregistration is invalid")
    if config.get("feasibility_policy") != _EXPECTED_FEASIBILITY_POLICY:
        raise ValueError("confirmatory feasibility/fallback policy is invalid")

    binding = config.get("source_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("confirmatory source lacks source_binding")
    if (
        binding.get("source_config_relative_path") != SOURCE_CONFIG_RELATIVE_PATH
        or binding.get("source_preflight_relative_path")
        != SOURCE_PREFLIGHT_RELATIVE_PATH
        or binding.get("source_published_manifest_relative_path")
        != SOURCE_PUBLISHED_MANIFEST_RELATIVE_PATH
    ):
        raise ValueError("confirmatory source artifact paths are not registered")
    source_config_path = PROJECT_ROOT / SOURCE_CONFIG_RELATIVE_PATH
    source_config = _read_object(source_config_path)
    source_file_sha = _file_sha256(source_config_path)
    source_sha = exp26.canonical_config_sha256(source_config)
    if source_file_sha != binding.get(
        "source_config_file_sha256"
    ) or source_sha != binding.get("source_config_canonical_sha256"):
        raise ValueError("frozen Exp26 config binding is invalid")
    for key in _COPIED_SOURCE_KEYS:
        if config.get(key) != source_config.get(key):
            raise ValueError(f"confirmatory {key} differs from frozen Exp26")
    source_actuator = source_config.get("actuator")
    actuator = config.get("actuator")
    if not isinstance(source_actuator, Mapping) or not isinstance(actuator, Mapping):
        raise ValueError("confirmatory actuator contract is malformed")
    expected_actuator = dict(source_actuator)
    expected_actuator["max_scale"] = MAX_SCALE
    if dict(actuator) != expected_actuator:
        raise ValueError("confirmatory actuator differs beyond frozen cap 256")

    cells = exp26._manifest(config)
    receipt = manifest_hash(cells)
    if (
        len(cells) != EXPECTED_GENERATORS
        or len({cell.generator_id for cell in cells}) != EXPECTED_GENERATORS
        or receipt != binding.get("source_manifest_sha256")
        or receipt != source_config["manifest"]["expected_hash"]
    ):
        raise ValueError("confirmatory generator manifest is invalid")
    preflight_path = PROJECT_ROOT / SOURCE_PREFLIGHT_RELATIVE_PATH
    preflight_receipt = exp26._preflight_receipt_sha256(preflight_path)
    summary = _read_object(preflight_path / "preflight_summary.json")
    code_tree = _read_object(preflight_path / "code_tree_provenance.json")
    if (
        preflight_receipt != binding.get("source_preflight_receipt_sha256")
        or summary.get("schema_version") != binding.get("source_preflight_schema")
        or summary.get("preflight_passed") is not True
        or summary.get("canonical_config_sha256") != source_sha
        or summary.get("manifest_hash") != receipt
        or int(summary.get("n_seeds", -1)) != 30
        or int(summary.get("n_generators", -1)) != EXPECTED_GENERATORS
    ):
        raise ValueError("frozen Exp26 preflight binding is invalid")
    critical = binding.get("critical_file_sha256")
    if not isinstance(critical, Mapping):
        raise ValueError("confirmatory source lacks critical file hashes")
    observed = {
        str(relative): _file_sha256(PROJECT_ROOT / str(relative))
        for relative in critical
    }
    if (
        dict(critical) != observed
        or code_tree.get("critical_file_sha256") != observed
        or code_tree.get("critical_code_sha256")
        != binding.get("source_preflight_critical_code_sha256")
        or code_tree.get("git_commit") != binding.get("source_preflight_git_commit")
        or code_tree.get("git_tree") != binding.get("source_preflight_git_tree")
        or code_tree.get("stable_during_run") is not True
        or code_tree.get("git_dirty") is not False
    ):
        raise ValueError("critical Exp26 code binding is invalid")
    implementation = binding.get("exp29_implementation_sha256")
    observed_implementation = exp29_implementation_sha256()
    if (
        binding.get("exp29_implementation_hash_scheme") != IMPLEMENTATION_HASH_SCHEME
        or not isinstance(implementation, Mapping)
        or set(implementation) != set(IMPLEMENTATION_RELATIVE_PATHS)
        or dict(implementation) != observed_implementation
    ):
        raise ValueError("registered Exp29 implementation binding is invalid")
    implementation_contract_sha256 = _canonical_sha256(observed_implementation)
    budget = binding.get("budget_policy")
    if (
        not isinstance(budget, Mapping)
        or budget.get("selection_scope") != "preregistered_without_seed_60_89_access"
        or budget.get("evaluation_panel_refit_permitted") is not False
        or float(budget.get("max_scale", -1.0)) != MAX_SCALE
        or budget.get("no_further_ceiling_amendment") is not True
    ):
        raise ValueError("confirmatory fixed-ceiling policy is invalid")
    if _file_sha256(
        PROJECT_ROOT / SOURCE_PUBLISHED_MANIFEST_RELATIVE_PATH
    ) != binding.get("source_published_manifest_file_sha256"):
        raise ValueError("published Exp26 manifest binding is invalid")

    config_path = _config_source_file(config)
    git = _validate_git(exp26.git_identity() if current_git is None else current_git)
    runtime = _validate_runtime(
        config,
        exp26.scientific_runtime_versions()
        if runtime_versions is None
        else runtime_versions,
    )
    return SourceContract(
        config_sha256=canonical_config_sha256(config),
        config_file_sha256=_file_sha256(config_path),
        source_contract_sha256=_canonical_sha256(binding),
        source_config_sha256=source_sha,
        source_config_file_sha256=source_file_sha,
        source_manifest_sha256=receipt,
        source_preflight_receipt_sha256=preflight_receipt,
        source_critical_code_sha256=str(
            binding["source_preflight_critical_code_sha256"]
        ),
        implementation_file_sha256=observed_implementation,
        implementation_contract_sha256=implementation_contract_sha256,
        current_git=git,
        runtime_versions=runtime,
    )


def build_evidence_provenance(
    contract: SourceContract, *, run_label: str
) -> dict[str, object]:
    if run_label != REQUIRED_RUN_LABEL:
        raise ValueError(f"confirmatory source requires {REQUIRED_RUN_LABEL!r}")
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_role": "confirmatory_test_source_only",
        "standalone_inference_permitted": False,
        "registered_config_sha256": contract.config_sha256,
        "registered_config_file_sha256": contract.config_file_sha256,
        "source_contract_sha256": contract.source_contract_sha256,
        "source_exp26_config_sha256": contract.source_config_sha256,
        "source_exp26_config_file_sha256": contract.source_config_file_sha256,
        "source_exp26_manifest_sha256": contract.source_manifest_sha256,
        "source_exp26_preflight_receipt_sha256": (
            contract.source_preflight_receipt_sha256
        ),
        "source_exp26_critical_code_sha256": contract.source_critical_code_sha256,
        "exp29_implementation_hash_scheme": IMPLEMENTATION_HASH_SCHEME,
        "exp29_implementation_sha256": dict(contract.implementation_file_sha256),
        "exp29_implementation_contract_sha256": (
            contract.implementation_contract_sha256
        ),
        "feasibility_policy_sha256": _canonical_sha256(_EXPECTED_FEASIBILITY_POLICY),
        "evaluation_seeds_sha256": _canonical_sha256(list(EXPECTED_SEEDS)),
        "run_git": dict(contract.current_git),
        "runtime_versions": dict(contract.runtime_versions),
        "run_label": run_label,
    }


def evidence_row_fields(provenance: Mapping[str, Any]) -> dict[str, object]:
    git = provenance["run_git"]
    versions = provenance["runtime_versions"]
    if not isinstance(git, Mapping) or not isinstance(versions, Mapping):
        raise ValueError("confirmatory source provenance is malformed")
    return {
        "source_panel_evidence_schema": provenance["schema_version"],
        "source_panel_protocol_version": provenance["protocol_version"],
        "registered_config_sha256": provenance["registered_config_sha256"],
        "registered_config_file_sha256": provenance["registered_config_file_sha256"],
        "source_contract_sha256": provenance["source_contract_sha256"],
        "source_exp26_config_sha256": provenance["source_exp26_config_sha256"],
        "source_exp26_config_file_sha256": provenance[
            "source_exp26_config_file_sha256"
        ],
        "source_exp26_manifest_sha256": provenance["source_exp26_manifest_sha256"],
        "source_exp26_preflight_receipt_sha256": provenance[
            "source_exp26_preflight_receipt_sha256"
        ],
        "source_exp26_critical_code_sha256": provenance[
            "source_exp26_critical_code_sha256"
        ],
        "exp29_implementation_contract_sha256": provenance[
            "exp29_implementation_contract_sha256"
        ],
        "exp29_runner_source_sha256": provenance["exp29_implementation_sha256"][
            RUNNER_RELATIVE_PATH
        ],
        "exp29_packager_source_sha256": provenance["exp29_implementation_sha256"][
            PACKAGER_RELATIVE_PATH
        ],
        "exp29_adapter_source_sha256": provenance["exp29_implementation_sha256"][
            ADAPTER_RELATIVE_PATH
        ],
        "feasibility_policy_sha256": provenance["feasibility_policy_sha256"],
        "evaluation_seeds_sha256": provenance["evaluation_seeds_sha256"],
        "run_git_commit": git["commit"],
        "run_git_tree": git["tree"],
        "run_git_dirty": git["dirty"],
        "run_python_version": versions["python"],
        "run_numpy_version": versions["numpy"],
        "run_scipy_version": versions["scipy"],
        "run_scikit_learn_version": versions["scikit_learn"],
        "run_pandas_version": versions["pandas"],
        "run_statsmodels_version": versions["statsmodels"],
        "run_label": provenance["run_label"],
        "source_only": True,
        "standalone_inference_permitted": False,
    }


def planned_conditions(config: Mapping[str, Any]) -> list[dict[str, object]]:
    return exp26._planned_conditions(config)


def _dimensions(
    cell: GeneratorCell,
    *,
    mode: str,
    manifest_receipt: str,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "generator_id": cell.generator_id,
        "generator_split": cell.generator_split,
        "alpha": cell.alpha,
        "transition_rank": cell.transition_rank,
        "input_rank": cell.input_rank,
        "delay": cell.delay,
        "noise_std": cell.noise_std,
        "rotation_seed": cell.rotation_seed,
        "actuator_mode": mode,
        "condition": mode,
        "manifest_hash": manifest_receipt,
        **evidence,
    }


def _deployment_metrics(
    metrics: Mapping[str, object],
    *,
    selected_mode: str,
    fallback_applied: bool,
    feasible: bool,
) -> dict[str, object]:
    result = dict(metrics)
    result.update(
        {
            "actuator_feasible": feasible,
            "deployment_available": True,
            "deployment_mode": "frozen" if fallback_applied else selected_mode,
            "deployment_fallback_mode": "frozen",
            "deployment_fallback_applied": fallback_applied,
            "deployment_validation_balanced_accuracy": float(
                metrics["validation_balanced_accuracy"]
            ),
            "deployment_test_balanced_accuracy": float(
                metrics["test_balanced_accuracy"]
            ),
            "matched_budget_support_eligible": bool(
                feasible and selected_mode != "frozen"
            ),
            "unconditional_cell_retained": True,
        }
    )
    return result


def _classify_infeasible(error: ActuatorFitError) -> str:
    message = str(error).lower()
    if "exceeds max_scale" in message or "non-finite" in message:
        return "budget_scale_above_cap"
    if "degenerate" in message:
        return "degenerate_actuator"
    return "budget_mismatch"


def _fallback_metrics(
    frozen: Mapping[str, object],
    *,
    mode: str,
    reason: str,
    error: BaseException | None = None,
    functional_budget_valid: bool = False,
) -> dict[str, object]:
    copied = {
        name: frozen[name]
        for name in (
            "experiment_protocol_version",
            "statistics_unit",
            "split_unit",
            "time_points_randomly_split",
            "profile",
            "dev_only",
            "training_algorithm",
            "used_autograd",
            "used_bptt",
            "chi",
            "state_demand",
            "input_demand",
            "target_train_balanced_accuracy",
            "target_validation_balanced_accuracy",
            "target_test_balanced_accuracy",
            "train_balanced_accuracy",
            "validation_balanced_accuracy",
            "test_balanced_accuracy",
            "dataset_fingerprint",
            "train_split_fingerprint",
            "validation_split_fingerprint",
            "test_split_fingerprint",
        )
    }
    copied.update(
        {
            "status": "infeasible",
            "actuator_feasible": False,
            "infeasible_reason": reason,
            "deployment_available": True,
            "deployment_mode": "frozen",
            "deployment_fallback_mode": "frozen",
            "deployment_fallback_applied": True,
            "deployment_validation_balanced_accuracy": float(
                frozen["validation_balanced_accuracy"]
            ),
            "deployment_test_balanced_accuracy": float(
                frozen["test_balanced_accuracy"]
            ),
            "fallback_frozen_validation_balanced_accuracy": float(
                frozen["validation_balanced_accuracy"]
            ),
            "fallback_frozen_test_balanced_accuracy": float(
                frozen["test_balanced_accuracy"]
            ),
            "fallback_frozen_correction_fingerprint": frozen["correction_fingerprint"],
            "functional_budget_applicable": True,
            "functional_budget_valid": functional_budget_valid,
            "matched_budget_support_eligible": False,
            "unconditional_cell_retained": True,
            "requested_actuator_mode": mode,
        }
    )
    if error is not None:
        copied["error_type"] = type(error).__name__
        copied["error"] = str(error)
    return copied


def _record_setup_failure(
    run: ExperimentRun,
    cell: GeneratorCell,
    error: BaseException,
    *,
    receipt: str,
    evidence: Mapping[str, object],
) -> None:
    for mode in EXPECTED_MODES:
        run.mark_condition_failure(
            error,
            **_dimensions(
                cell,
                mode=mode,
                manifest_receipt=receipt,
                evidence=evidence,
            ),
        )


def run_seed(
    config: dict[str, Any],
    seed: int,
    results_root: str | Path,
    *,
    run_label: str,
) -> Path:
    """Run one untouched confirmatory seed and retain every registered cell."""

    contract = validate_source_contract(config)
    requested_seed = _exact_integer(seed, name="requested seed")
    if requested_seed not in EXPECTED_SEEDS:
        raise ValueError("requested confirmatory seed is not in 60--89")
    if run_label != REQUIRED_RUN_LABEL:
        raise ValueError(f"confirmatory source requires {REQUIRED_RUN_LABEL!r}")
    cells = exp26._manifest(config)
    receipt = manifest_hash(cells)
    initialize_seed(requested_seed)
    provenance = build_evidence_provenance(contract, run_label=run_label)
    evidence = evidence_row_fields(provenance)
    run_config = {**config, "evidence_provenance": provenance}
    with ExperimentRun(
        EXPERIMENT,
        requested_seed,
        run_config,
        results_root=results_root,
        run_label=run_label,
    ) as run:
        run.register_conditions(planned_conditions(run_config))
        try:
            carrier = exp26.make_carrier(
                exp26._carrier_config(run_config), requested_seed
            )
        except Exception as error:
            for cell in cells:
                _record_setup_failure(
                    run,
                    cell,
                    error,
                    receipt=receipt,
                    evidence=evidence,
                )
        else:
            moment_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
            for cell in cells:
                try:
                    setup = exp26._setup_generator(
                        run_config,
                        carrier,
                        cell,
                        seed=requested_seed,
                        moment_cache=moment_cache,
                    )
                    frozen, frozen_valid = exp26._condition_metrics(
                        run_config,
                        setup,
                        mode="frozen",
                    )
                    if not frozen_valid:
                        raise RuntimeError("registered frozen actuator is unavailable")
                except Exception as error:
                    _record_setup_failure(
                        run,
                        cell,
                        error,
                        receipt=receipt,
                        evidence=evidence,
                    )
                    continue
                frozen_metrics = _deployment_metrics(
                    frozen,
                    selected_mode="frozen",
                    fallback_applied=False,
                    feasible=True,
                )
                run.record(
                    frozen_metrics,
                    **_dimensions(
                        cell,
                        mode="frozen",
                        manifest_receipt=receipt,
                        evidence=evidence,
                    ),
                )
                for mode in EXPECTED_MODES[1:]:
                    dimensions = _dimensions(
                        cell,
                        mode=mode,
                        manifest_receipt=receipt,
                        evidence=evidence,
                    )
                    try:
                        metrics, valid = exp26._condition_metrics(
                            run_config,
                            setup,
                            mode=mode,
                        )
                    except ActuatorFitError as error:
                        run.record(
                            _fallback_metrics(
                                frozen,
                                mode=mode,
                                reason=_classify_infeasible(error),
                                error=error,
                            ),
                            **dimensions,
                        )
                    except Exception as error:
                        payload = _fallback_metrics(
                            frozen,
                            mode=mode,
                            reason="unexpected_execution_failure",
                            error=error,
                        )
                        payload["failure_reason"] = "unexpected_execution_failure"
                        run.record_failed_condition(payload, **dimensions)
                    else:
                        if valid:
                            run.record(
                                _deployment_metrics(
                                    metrics,
                                    selected_mode=mode,
                                    fallback_applied=False,
                                    feasible=True,
                                ),
                                **dimensions,
                            )
                        else:
                            reason = (
                                "budget_mismatch"
                                if not bool(metrics["functional_budget_valid"])
                                else "effective_instability"
                            )
                            run.record(
                                _fallback_metrics(
                                    frozen,
                                    mode=mode,
                                    reason=reason,
                                    functional_budget_valid=bool(
                                        metrics["functional_budget_valid"]
                                    ),
                                ),
                                **dimensions,
                            )

        end_git = exp26.git_identity()
        end_runtime = exp26.scientific_runtime_versions()
        if any(
            end_git.get(name) != contract.current_git.get(name)
            for name in ("commit", "tree", "dirty")
        ) or dict(end_runtime) != dict(contract.runtime_versions):
            raise RuntimeError(
                "confirmatory source git/runtime identity changed during seed run"
            )
        return run.path


def _selected_seeds(config: Mapping[str, Any], override: str | None) -> Iterable[int]:
    selected = seed_list(override if override is not None else config["seeds"])
    if any(seed not in EXPECTED_SEEDS for seed in selected):
        raise ValueError("seed override must be a subset of 60--89")
    return selected


def main() -> None:
    parser = basic_parser(
        "Exp29 preregistered feasibility-aware confirmatory source panel",
        "configs/formal/exp29_confirmatory_source_panel.json",
    )
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    config = load_json_config(args.config)
    if args.run_label != REQUIRED_RUN_LABEL:
        parser.error(f"--run-label must equal {REQUIRED_RUN_LABEL}")
    for seed in _selected_seeds(config, args.seeds):
        print(
            run_seed(
                config,
                seed,
                args.results_root,
                run_label=args.run_label,
            )
        )


if __name__ == "__main__":
    main()
