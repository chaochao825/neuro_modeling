"""Hash-locked independent Exp26 source panel for downstream Exp28 tests.

This module deliberately performs no selector fitting or statistical inference.
It reuses the frozen Exp26 generator and task-matched actuator implementation on
independent carrier/data seeds 30--59.  The old formal configuration, generator
manifest, train-only budget preflight, and every critical Exp26 implementation
file are hash bound before an attempt may start.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp26_actuator_phase_diagram as exp26
from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.analysis.actuator_manifest import GeneratorCell, manifest_hash
from src.utils.artifacts import ExperimentRun


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp28_exp26_independent_source_panel"
PROTOCOL_VERSION = "exp28_exp26_independent_source_v1"
EVIDENCE_SCHEMA_VERSION = "exp28_independent_source_evidence_v1"
AMENDED_PROTOCOL_VERSION = (
    "exp28_exp26_independent_source_v1_ceiling_amendment_1"
)
AMENDED_EVIDENCE_SCHEMA_VERSION = (
    "exp28_independent_source_evidence_v1_ceiling_amendment_1"
)
PROFILE = "independent_test"
EXPECTED_SEEDS = tuple(range(30, 60))
EXPECTED_GENERATORS = 88
EXPECTED_MODES = tuple(exp26.MODES)
EXPECTED_ROWS_PER_SEED = EXPECTED_GENERATORS * len(EXPECTED_MODES)
EXPECTED_PANEL_ROWS = len(EXPECTED_SEEDS) * EXPECTED_ROWS_PER_SEED
REQUIRED_RUN_LABEL = "exp28-exp26-independent-source-v1"
AMENDED_REQUIRED_RUN_LABEL = (
    "exp28-exp26-independent-source-v1-ceiling-amendment-1"
)
FROZEN_MAX_SCALE = 128.0
AMENDED_MAX_SCALE = 256.0
AMENDMENT_ID = "exp28_ceiling_reachability_amendment_1"
AMENDED_CONFIG_CANONICAL_SHA256 = (
    "0ec0ff7c1079df793063b0bb37f52b9c823b8fff96a8572e021f51438788c03b"
)
SOURCE_CONFIG_RELATIVE_PATH = "configs/formal/exp26_actuator_phase_diagram.json"
SOURCE_PREFLIGHT_RELATIVE_PATH = (
    "results/exp26_actuator_matching_formal_v2_e08beaf/preflight"
)
SOURCE_PUBLISHED_MANIFEST_RELATIVE_PATH = (
    "results/exp26_actuator_matching_formal_v2_e08beaf/MANIFEST.sha256"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
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
_ROW_EVIDENCE_FIELDS = (
    "source_panel_evidence_schema",
    "independent_config_sha256",
    "independent_config_file_sha256",
    "source_contract_sha256",
    "source_exp26_config_sha256",
    "source_exp26_config_file_sha256",
    "source_exp26_manifest_sha256",
    "source_exp26_preflight_receipt_sha256",
    "source_exp26_critical_code_sha256",
    "run_git_commit",
    "run_git_tree",
    "run_git_dirty",
    "run_python_version",
    "run_numpy_version",
    "run_scipy_version",
    "run_scikit_learn_version",
    "run_pandas_version",
    "run_statsmodels_version",
    "run_label",
    "source_only",
    "standalone_inference_permitted",
)


@dataclass(frozen=True)
class SourceContract:
    """Validated immutable identities needed by one independent attempt."""

    independent_config_sha256: str
    independent_config_file_sha256: str
    source_contract_sha256: str
    source_config_sha256: str
    source_config_file_sha256: str
    source_manifest_sha256: str
    source_preflight_receipt_sha256: str
    source_critical_code_sha256: str
    current_git: Mapping[str, object]
    runtime_versions: Mapping[str, str]
    protocol_version: str = PROTOCOL_VERSION
    evidence_schema_version: str = EVIDENCE_SCHEMA_VERSION
    required_run_label: str = REQUIRED_RUN_LABEL
    functional_budget_max_scale: float = FROZEN_MAX_SCALE
    protocol_amendment: Mapping[str, object] | None = None
    protocol_amendment_sha256: str | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only attempt-specific provenance from an Exp28 config."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    return {
        str(key): value
        for key, value in config.items()
        if str(key) not in _RUNTIME_CONFIG_KEYS
    }


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    return _canonical_sha256(canonical_config_payload(config))


_EXPECTED_AMENDMENT: dict[str, object] = {
    "amendment_id": AMENDMENT_ID,
    "amendment_status": "transparent_post_hoc_protocol_amendment",
    "amendment_scope": "independent_test_functional_budget_ceiling_only",
    "trigger_class": "reachability_only",
    "decision_rule": "next_power_of_two_strictly_above_previous_ceiling",
    "previous_max_scale": FROZEN_MAX_SCALE,
    "amended_max_scale": AMENDED_MAX_SCALE,
    "performance_metrics_inspected": True,
    "performance_metrics_inspection_timing": (
        "after_deterministic_amendment_decision_draft"
    ),
    "performance_metrics_used_for_amendment": False,
    "confirmatory_independence_restored": False,
    "selective_rerun_permitted": False,
    "further_ceiling_amendments_permitted": False,
    "rerun_scope": "all_30_seeds_x_88_generators_x_5_modes",
    "trigger_evidence": {
        "source_package_locator": (
            "results/exp28_independent_source_ceiling128_invalid_v3_e650dd1"
        ),
        "source_package_archive_file": (
            "exp28_source_v3_e650dd1_invalid_package.tar.gz"
        ),
        "source_package_archive_sha256": (
            "107039999aaa271d27aafd97746d74e76eb341be15247d4a84994a884a7d7c10"
        ),
        "source_panel_receipt_file_sha256": (
            "d01352232ac22d35153ebd74e4916bbb8e9b598ddcb622b269fa73d6733be96d"
        ),
        "source_panel_receipt_payload_sha256": (
            "816ce6f72a5bd4d02b3f18eda06fc1d280de0e0d9d90f75becbcbc93ccc10084"
        ),
        "source_panel_conclusion_file_sha256": (
            "07ea53eb824d77f1d53699f73d1cc158b038c735fb0e67c3cf68833716660ff0"
        ),
        "source_panel_raw_metrics_sha256": (
            "55e825667f003d5f80b489eb13cdecb57dcb4a40aab10b0c68250e69b37632fe"
        ),
        "run_git_commit": "e650dd17554d62124f714e52a1ab7d171fa3f2b2",
        "run_git_tree": "30d7939386c4ec218c5b74ef92e35f645b571224",
        "observed_row_count": 13200,
        "observed_complete_rows": 13199,
        "observed_failed_rows": 1,
        "failing_cell": {
            "seed": 52,
            "generator_id": "d65464a1a0917550a226",
            "actuator_mode": "routing",
            "alpha": 1.0,
            "transition_rank": 1,
            "input_rank": 2,
            "delay": 12,
            "noise_std": 0.3,
            "error_type": "ActuatorFitError",
            "error": (
                "functional-budget scale is non-finite or exceeds max_scale"
            ),
        },
    },
    "unchanged_components": [
        "meta_source",
        "generator_manifest",
        "task",
        "carrier",
        "noise",
        "trial_order",
        "actuator_modes",
        "performance_endpoints",
        "performance_acceptance_criteria",
    ],
}


def _registered_protocol(config: Mapping[str, Any]) -> dict[str, object]:
    """Resolve either frozen-v1 or its explicit reachability amendment."""

    version = config.get("protocol_version")
    amendment = config.get("protocol_amendment")
    if version == PROTOCOL_VERSION:
        if amendment is not None:
            raise ValueError("frozen-v1 protocol must not contain an amendment")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "required_run_label": REQUIRED_RUN_LABEL,
            "max_scale": FROZEN_MAX_SCALE,
            "protocol_amendment": None,
            "protocol_amendment_sha256": None,
        }
    if version != AMENDED_PROTOCOL_VERSION:
        raise ValueError("independent source protocol is not registered")
    if not isinstance(amendment, Mapping) or dict(amendment) != _EXPECTED_AMENDMENT:
        raise ValueError("independent source ceiling amendment is not registered")
    if canonical_config_sha256(config) != AMENDED_CONFIG_CANONICAL_SHA256:
        raise ValueError("independent source amended config hash is not registered")
    return {
        "protocol_version": AMENDED_PROTOCOL_VERSION,
        "evidence_schema_version": AMENDED_EVIDENCE_SCHEMA_VERSION,
        "required_run_label": AMENDED_REQUIRED_RUN_LABEL,
        "max_scale": AMENDED_MAX_SCALE,
        "protocol_amendment": dict(amendment),
        "protocol_amendment_sha256": _canonical_sha256(amendment),
    }


def _validate_amendment_trigger_artifact(amendment: Mapping[str, Any]) -> None:
    """Verify the preserved ceiling-128 failure without reading performance fields."""

    trigger = amendment.get("trigger_evidence")
    if not isinstance(trigger, Mapping):
        raise ValueError("ceiling amendment trigger evidence is malformed")
    locator = trigger.get("source_package_locator")
    archive_name = trigger.get("source_package_archive_file")
    if not isinstance(locator, str) or not isinstance(archive_name, str):
        raise ValueError("ceiling amendment trigger paths are malformed")
    evidence_dir = (PROJECT_ROOT / locator).resolve()
    try:
        evidence_dir.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("ceiling amendment trigger escapes the project") from error
    receipt_path = evidence_dir / "source_panel_receipt.json"
    conclusion_path = evidence_dir / "conclusion.json"
    archive_path = evidence_dir / archive_name
    if (
        _file_sha256(receipt_path)
        != trigger.get("source_panel_receipt_file_sha256")
        or _file_sha256(conclusion_path)
        != trigger.get("source_panel_conclusion_file_sha256")
        or _file_sha256(archive_path)
        != trigger.get("source_package_archive_sha256")
    ):
        raise ValueError("ceiling amendment trigger artifact hash is invalid")

    receipt = _read_object(receipt_path)
    conclusion = _read_object(conclusion_path)
    receipt_payload = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_payload_sha256"
    }
    coverage = receipt.get("coverage")
    attempts = receipt.get("attempts")
    run_provenance = receipt.get("run_provenance")
    if (
        receipt.get("receipt_payload_sha256")
        != trigger.get("source_panel_receipt_payload_sha256")
        or _canonical_sha256(receipt_payload)
        != trigger.get("source_panel_receipt_payload_sha256")
        or receipt.get("raw_metrics_sha256")
        != trigger.get("source_panel_raw_metrics_sha256")
        or receipt.get("raw_metrics_row_count") != 13200
        or not isinstance(coverage, Mapping)
        or coverage.get("observed_row_count") != 13200
        or coverage.get("row_status_counts")
        != {"complete": 13199, "failed": 1, "invalid": 0}
        or coverage.get("cartesian_complete") is not True
        or coverage.get("source_panel_valid") is not False
        or coverage.get("all_functional_budgets_valid") is not False
        or not isinstance(attempts, list)
        or len(attempts) != 30
        or not isinstance(run_provenance, Mapping)
        or run_provenance.get("run_git")
        != {
            "commit": trigger.get("run_git_commit"),
            "tree": trigger.get("run_git_tree"),
            "dirty": False,
        }
        or conclusion.get("coverage") != coverage
        or conclusion.get("source_panel_valid") is not False
        or conclusion.get("conclusion") != "inconclusive"
        or conclusion.get("raw_metrics_sha256")
        != trigger.get("source_panel_raw_metrics_sha256")
    ):
        raise ValueError("ceiling amendment trigger receipt is invalid")
    failed_attempts = [
        attempt
        for attempt in attempts
        if isinstance(attempt, Mapping)
        and int(attempt.get("observed_failed_rows", 0)) != 0
    ]
    if (
        len(failed_attempts) != 1
        or failed_attempts[0].get("seed") != 52
        or failed_attempts[0].get("observed_row_count") != 440
        or failed_attempts[0].get("observed_complete_rows") != 439
        or failed_attempts[0].get("observed_failed_rows") != 1
        or failed_attempts[0].get("observed_invalid_rows") != 0
        or failed_attempts[0].get("run_status") != "complete_with_failures"
    ):
        raise ValueError("ceiling amendment trigger failure count is invalid")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        receipt_member = archive.extractfile("package/source_panel_receipt.json")
        conclusion_member = archive.extractfile("package/conclusion.json")
        raw_member = archive.extractfile("package/raw_metrics.jsonl")
        if receipt_member is None or conclusion_member is None or raw_member is None:
            raise ValueError("ceiling amendment trigger archive is incomplete")
        if (
            receipt_member.read() != receipt_path.read_bytes()
            or conclusion_member.read() != conclusion_path.read_bytes()
        ):
            raise ValueError("ceiling amendment trigger archive metadata differs")
        raw_digest = hashlib.sha256()
        failed_rows: list[dict[str, Any]] = []
        row_count = 0
        for raw_line in raw_member:
            raw_digest.update(raw_line)
            row_count += 1
            if b'"status":"failed"' in raw_line:
                value = json.loads(raw_line)
                if not isinstance(value, dict):
                    raise ValueError("ceiling amendment failed row is malformed")
                failed_rows.append(value)
    failing_cell = trigger.get("failing_cell")
    if not isinstance(failing_cell, Mapping):
        raise ValueError("ceiling amendment failing-cell identity is malformed")
    if (
        row_count != 13200
        or raw_digest.hexdigest() != trigger.get("source_panel_raw_metrics_sha256")
        or len(failed_rows) != 1
        or any(failed_rows[0].get(key) != value for key, value in failing_cell.items())
    ):
        raise ValueError("ceiling amendment unique failed-cell identity is invalid")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read hash-bound JSON object {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"hash-bound JSON must be an object: {path}")
    return dict(value)


def _config_source_file(config: Mapping[str, Any]) -> Path:
    value = config.get("config_path")
    if not isinstance(value, str) or not value:
        raise ValueError("independent source config requires config_path provenance")
    path = Path(value)
    if not path.is_file():
        raise ValueError("independent source config_path is not a regular file")
    source = _read_object(path)
    if canonical_config_payload(source) != canonical_config_payload(config):
        raise ValueError("runtime independent source config differs from its file")
    return path


def _exact_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_runtime_policy(
    config: Mapping[str, Any],
    runtime_versions: Mapping[str, object],
) -> dict[str, str]:
    policy = config.get("runtime_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("independent source config lacks runtime_policy")
    expected_python = policy.get("python_major_minor")
    python_version = runtime_versions.get("python")
    if (
        expected_python != "3.11"
        or not isinstance(python_version, str)
        or not python_version.startswith("3.11.")
    ):
        raise ValueError("independent source panel requires Python 3.11")
    expected_distributions = [
        "numpy",
        "scipy",
        "pandas",
        "scikit_learn",
        "statsmodels",
    ]
    if (
        policy.get("required_distributions")
        != ["numpy", "scipy", "pandas", "scikit_learn", "statsmodels"]
        or policy.get("exact_match_across_seeds") is not True
    ):
        raise ValueError("independent source runtime policy is not registered")
    resolved: dict[str, str] = {"python": python_version}
    for name in expected_distributions:
        value = runtime_versions.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"independent source runtime lacks {name}")
        resolved[name] = value
    return resolved


def _validate_current_git(current_git: Mapping[str, object]) -> dict[str, object]:
    commit = current_git.get("commit")
    tree = current_git.get("tree")
    dirty = current_git.get("dirty")
    if (
        not isinstance(commit, str)
        or not GIT_OBJECT_PATTERN.fullmatch(commit)
        or not isinstance(tree, str)
        or not GIT_OBJECT_PATTERN.fullmatch(tree)
        or dirty is not False
    ):
        raise ValueError(
            "independent source attempts require a clean identifiable git tree"
        )
    return {"commit": commit, "tree": tree, "dirty": False}


def validate_source_contract(
    config: Mapping[str, Any],
    *,
    current_git: Mapping[str, object] | None = None,
    runtime_versions: Mapping[str, object] | None = None,
) -> SourceContract:
    """Fail closed unless Exp28 is an exact independent replay of Exp26."""

    protocol = _registered_protocol(config)
    if protocol["protocol_amendment"] is not None:
        amendment = protocol["protocol_amendment"]
        if not isinstance(amendment, Mapping):
            raise ValueError("independent source amendment is malformed")
        _validate_amendment_trigger_artifact(amendment)
    if (
        config.get("profile") != PROFILE
        or config.get("dev_only") is not False
        or config.get("protocol_version") != protocol["protocol_version"]
        or config.get("required_run_label") != protocol["required_run_label"]
        or config.get("evidence_role") != "independent_test_source_only"
        or config.get("standalone_inference_permitted") is not False
    ):
        raise ValueError("independent source profile/protocol contract is invalid")
    seeds_value = config.get("seeds")
    if not isinstance(seeds_value, Sequence) or isinstance(
        seeds_value, (str, bytes)
    ):
        raise ValueError("independent source seeds must be a sequence")
    seeds = tuple(_exact_integer(seed, name="seed") for seed in seeds_value)
    if seeds != EXPECTED_SEEDS:
        raise ValueError("independent source panel requires exact seeds 30--59")
    if (
        config.get("training_algorithm")
        != "train_only_closed_form_task_matched_actuators"
        or config.get("used_autograd") is not False
        or config.get("used_bptt") is not False
    ):
        raise ValueError("independent source panel must retain non-BPTT Exp26 fitting")

    binding = config.get("source_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("independent source config lacks source_binding")
    if (
        binding.get("source_config_relative_path") != SOURCE_CONFIG_RELATIVE_PATH
        or binding.get("source_preflight_relative_path")
        != SOURCE_PREFLIGHT_RELATIVE_PATH
        or binding.get("source_published_manifest_relative_path")
        != SOURCE_PUBLISHED_MANIFEST_RELATIVE_PATH
    ):
        raise ValueError("independent source artifact paths are not registered")

    source_config_path = PROJECT_ROOT / SOURCE_CONFIG_RELATIVE_PATH
    source_config = _read_object(source_config_path)
    source_config_file_sha = _file_sha256(source_config_path)
    source_config_sha = exp26.canonical_config_sha256(source_config)
    if (
        source_config_file_sha != binding.get("source_config_file_sha256")
        or source_config_sha != binding.get("source_config_canonical_sha256")
    ):
        raise ValueError("old Exp26 formal config hash binding is invalid")
    for key in _COPIED_SOURCE_KEYS:
        if config.get(key) != source_config.get(key):
            raise ValueError(f"independent source {key} differs from frozen Exp26")
    actuator = config.get("actuator")
    source_actuator = source_config.get("actuator")
    if not isinstance(actuator, Mapping) or not isinstance(source_actuator, Mapping):
        raise ValueError("independent source actuator contract is malformed")
    expected_actuator = dict(source_actuator)
    expected_actuator["max_scale"] = protocol["max_scale"]
    if dict(actuator) != expected_actuator:
        raise ValueError(
            "independent source actuator differs beyond the registered ceiling"
        )

    cells = exp26._manifest(config)
    source_manifest_sha = manifest_hash(cells)
    if (
        len(cells) != EXPECTED_GENERATORS
        or len({cell.generator_id for cell in cells}) != EXPECTED_GENERATORS
        or source_manifest_sha != binding.get("source_manifest_sha256")
        or source_manifest_sha != source_config["manifest"]["expected_hash"]
    ):
        raise ValueError("old Exp26 generator manifest hash binding is invalid")

    preflight_path = PROJECT_ROOT / SOURCE_PREFLIGHT_RELATIVE_PATH
    receipt_sha = exp26._preflight_receipt_sha256(preflight_path)
    preflight_summary = _read_object(preflight_path / "preflight_summary.json")
    code_tree = _read_object(preflight_path / "code_tree_provenance.json")
    if (
        receipt_sha != binding.get("source_preflight_receipt_sha256")
        or preflight_summary.get("schema_version")
        != binding.get("source_preflight_schema")
        or preflight_summary.get("preflight_passed") is not True
        or preflight_summary.get("provenance_clean") is not True
        or preflight_summary.get("provenance_stable_during_run") is not True
        or preflight_summary.get("canonical_config_sha256") != source_config_sha
        or preflight_summary.get("manifest_hash") != source_manifest_sha
        or float(preflight_summary.get("derived_ceiling", -1.0)) != 128.0
        or int(preflight_summary.get("n_seeds", -1)) != 30
        or int(preflight_summary.get("n_generators", -1)) != EXPECTED_GENERATORS
    ):
        raise ValueError("old Exp26 train-only preflight binding is invalid")
    critical_file_hashes = binding.get("critical_file_sha256")
    if not isinstance(critical_file_hashes, Mapping):
        raise ValueError("independent source lacks critical Exp26 file hashes")
    observed_file_hashes = {
        str(relative): _file_sha256(PROJECT_ROOT / str(relative))
        for relative in critical_file_hashes
    }
    if (
        dict(critical_file_hashes) != observed_file_hashes
        or code_tree.get("critical_file_sha256") != observed_file_hashes
        or code_tree.get("critical_code_sha256")
        != binding.get("source_preflight_critical_code_sha256")
        or code_tree.get("git_commit") != binding.get("source_preflight_git_commit")
        or code_tree.get("git_tree") != binding.get("source_preflight_git_tree")
        or code_tree.get("stable_during_run") is not True
        or code_tree.get("git_dirty") is not False
    ):
        raise ValueError("critical Exp26 implementation hash binding is invalid")
    budget_policy = binding.get("budget_policy")
    if (
        not isinstance(budget_policy, Mapping)
        or budget_policy.get("selection_scope")
        != "exp26_seeds_0_29_training_blocks_only"
        or budget_policy.get("independent_panel_refit_permitted") is not False
        or float(budget_policy.get("frozen_max_scale", -1.0)) != 128.0
        or not np.isclose(
            float(budget_policy.get("source_required_scale_max", np.nan)),
            float(preflight_summary.get("registered_required_scale_max", np.nan)),
            rtol=1e-10,
            atol=1e-12,
        )
    ):
        raise ValueError("independent source frozen budget policy is invalid")

    published_manifest = PROJECT_ROOT / SOURCE_PUBLISHED_MANIFEST_RELATIVE_PATH
    if (
        _file_sha256(published_manifest)
        != binding.get("source_published_manifest_file_sha256")
    ):
        raise ValueError("published Exp26 evidence manifest hash is invalid")

    config_path = _config_source_file(config)
    run_git = _validate_current_git(
        exp26.git_identity() if current_git is None else current_git
    )
    versions = _validate_runtime_policy(
        config,
        exp26.scientific_runtime_versions()
        if runtime_versions is None
        else runtime_versions,
    )
    return SourceContract(
        independent_config_sha256=canonical_config_sha256(config),
        independent_config_file_sha256=_file_sha256(config_path),
        source_contract_sha256=_canonical_sha256(binding),
        source_config_sha256=source_config_sha,
        source_config_file_sha256=source_config_file_sha,
        source_manifest_sha256=source_manifest_sha,
        source_preflight_receipt_sha256=receipt_sha,
        source_critical_code_sha256=str(
            binding["source_preflight_critical_code_sha256"]
        ),
        current_git=run_git,
        runtime_versions=versions,
        protocol_version=str(protocol["protocol_version"]),
        evidence_schema_version=str(protocol["evidence_schema_version"]),
        required_run_label=str(protocol["required_run_label"]),
        functional_budget_max_scale=float(protocol["max_scale"]),
        protocol_amendment=protocol["protocol_amendment"],
        protocol_amendment_sha256=(
            None
            if protocol["protocol_amendment_sha256"] is None
            else str(protocol["protocol_amendment_sha256"])
        ),
    )


def build_evidence_provenance(
    contract: SourceContract,
    *,
    run_label: str,
) -> dict[str, object]:
    """Create one identity shared by the config, manifest, and every row."""

    if run_label != contract.required_run_label:
        raise ValueError(
            f"registered independent source requires {contract.required_run_label!r}"
        )
    provenance: dict[str, object] = {
        "schema_version": contract.evidence_schema_version,
        "evidence_role": "independent_test_source_only",
        "standalone_inference_permitted": False,
        "independent_config_sha256": contract.independent_config_sha256,
        "independent_config_file_sha256": contract.independent_config_file_sha256,
        "source_contract_sha256": contract.source_contract_sha256,
        "source_exp26_config_sha256": contract.source_config_sha256,
        "source_exp26_config_file_sha256": contract.source_config_file_sha256,
        "source_exp26_manifest_sha256": contract.source_manifest_sha256,
        "source_exp26_preflight_receipt_sha256": (
            contract.source_preflight_receipt_sha256
        ),
        "source_exp26_critical_code_sha256": contract.source_critical_code_sha256,
        "run_git": dict(contract.current_git),
        "runtime_versions": dict(contract.runtime_versions),
        "run_label": run_label,
    }
    if contract.protocol_amendment is not None:
        provenance.update(
            {
                "protocol_version": contract.protocol_version,
                "functional_budget_max_scale": (
                    contract.functional_budget_max_scale
                ),
                "protocol_amendment": dict(contract.protocol_amendment),
                "protocol_amendment_sha256": (
                    contract.protocol_amendment_sha256
                ),
            }
        )
    return provenance


def evidence_row_fields(provenance: Mapping[str, Any]) -> dict[str, object]:
    git = provenance["run_git"]
    versions = provenance["runtime_versions"]
    if not isinstance(git, Mapping) or not isinstance(versions, Mapping):
        raise ValueError("independent source provenance is malformed")
    fields: dict[str, object] = {
        "source_panel_evidence_schema": provenance["schema_version"],
        "independent_config_sha256": provenance["independent_config_sha256"],
        "independent_config_file_sha256": provenance[
            "independent_config_file_sha256"
        ],
        "source_contract_sha256": provenance["source_contract_sha256"],
        "source_exp26_config_sha256": provenance["source_exp26_config_sha256"],
        "source_exp26_config_file_sha256": provenance[
            "source_exp26_config_file_sha256"
        ],
        "source_exp26_manifest_sha256": provenance[
            "source_exp26_manifest_sha256"
        ],
        "source_exp26_preflight_receipt_sha256": provenance[
            "source_exp26_preflight_receipt_sha256"
        ],
        "source_exp26_critical_code_sha256": provenance[
            "source_exp26_critical_code_sha256"
        ],
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
    amendment = provenance.get("protocol_amendment")
    if amendment is not None:
        if not isinstance(amendment, Mapping):
            raise ValueError("independent source amendment provenance is malformed")
        trigger = amendment.get("trigger_evidence")
        if not isinstance(trigger, Mapping):
            raise ValueError("independent source amendment trigger is malformed")
        fields.update(
            {
                "source_panel_protocol_version": provenance["protocol_version"],
                "functional_budget_max_scale": provenance[
                    "functional_budget_max_scale"
                ],
                "protocol_amendment_id": amendment["amendment_id"],
                "protocol_amendment_sha256": provenance[
                    "protocol_amendment_sha256"
                ],
                "protocol_amendment_trigger_class": amendment["trigger_class"],
                "protocol_amendment_performance_metrics_inspected": amendment[
                    "performance_metrics_inspected"
                ],
                "protocol_amendment_trigger_receipt_file_sha256": trigger[
                    "source_panel_receipt_file_sha256"
                ],
                "protocol_amendment_trigger_raw_metrics_sha256": trigger[
                    "source_panel_raw_metrics_sha256"
                ],
            }
        )
    return fields


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


def _record_all_failed(
    run: ExperimentRun,
    cells: Sequence[GeneratorCell],
    error: BaseException,
    *,
    receipt: str,
    evidence: Mapping[str, object],
) -> None:
    for cell in cells:
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
    """Run one independent source seed without fitting any selector."""

    contract = validate_source_contract(config)
    if run_label != contract.required_run_label:
        raise ValueError(
            f"formal independent source requires {contract.required_run_label!r}"
        )
    requested_seed = _exact_integer(seed, name="requested seed")
    if requested_seed not in EXPECTED_SEEDS:
        raise ValueError("requested independent source seed is not in 30--59")
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
            _record_all_failed(
                run,
                cells,
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
                except Exception as error:
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
                    continue
                for mode in EXPECTED_MODES:
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
                        # ``source_only`` and
                        # ``standalone_inference_permitted`` are immutable
                        # evidence dimensions.  Do not duplicate them in the
                        # metric payload: ExperimentRun correctly rejects any
                        # metric/dimension overlap.
                        if "source_panel_protocol_version" not in dimensions:
                            metrics["source_panel_protocol_version"] = (
                                contract.protocol_version
                            )
                        if valid:
                            run.record(metrics, **dimensions)
                        else:
                            failure_reasons = []
                            if not metrics["functional_budget_valid"]:
                                failure_reasons.append("functional_budget")
                            if not metrics["effective_dynamics_strictly_stable"]:
                                failure_reasons.append("effective_stability")
                            metrics["failure_reason"] = ",".join(failure_reasons)
                            run.record_failed_condition(metrics, **dimensions)
                    except Exception as error:
                        run.mark_condition_failure(error, **dimensions)

        end_git = exp26.git_identity()
        end_runtime = exp26.scientific_runtime_versions()
        if any(
            end_git.get(name) != contract.current_git.get(name)
            for name in ("commit", "tree", "dirty")
        ) or dict(end_runtime) != dict(contract.runtime_versions):
            raise RuntimeError(
                "independent source git/runtime identity changed during seed run"
            )
        return run.path


def _selected_seeds(config: Mapping[str, Any], override: str | None) -> Iterable[int]:
    selected = seed_list(override if override is not None else config["seeds"])
    if any(seed not in EXPECTED_SEEDS for seed in selected):
        raise ValueError("seed override must be a subset of 30--59")
    return selected


def main() -> None:
    parser = basic_parser(
        "Exp28 independent Exp26 source-only panel",
        "configs/formal/exp28_exp26_independent_source_panel.json",
    )
    parser.add_argument(
        "--run-label",
        required=True,
        help="must equal the run label registered by the selected config",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    try:
        required_run_label = str(_registered_protocol(config)["required_run_label"])
    except ValueError as error:
        parser.error(str(error))
    if args.run_label != required_run_label:
        parser.error(f"--run-label must equal {required_run_label}")
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
