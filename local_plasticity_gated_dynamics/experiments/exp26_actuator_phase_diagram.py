"""Continuous task--actuator matching phase diagram on a high-rank E/I base.

Exp26 replaces the two manually aligned Exp24 endpoints with independently
rotated, rank-controlled linear task generators.  Every actuator family is
fit on the same training blocks and matched to the same training functional
current budget.  Validation and test blocks are complete held-out trials;
time points are never randomly reassigned.

The registered prospective coordinate is a finite-horizon local-injection
demand fraction, ``chi``.  It uses the actual context differences
``alpha * delta_A`` and ``(1-alpha) * delta_B`` and explicitly audits the
state--input cross term.  RGL is retained only as a combined-actuator ceiling;
the primary label compares the best single input family with low-rank
recurrent control.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.actuator_demand import (
    control_gramians,
    finite_horizon_local_demand,
    transition_rank_requirement,
)
from src.analysis.actuator_manifest import (
    GeneratorCell,
    manifest_hash,
    select_generator_manifest,
)
from src.models.factorized_controller import ActuatorMode
from src.models.task_matched_actuators import (
    TaskMatchedActuator,
    TaskMatchedActuatorConfig,
    TaskMatchedRollout,
    fit_task_matched_actuator,
)
from src.tasks.actuator_matching import (
    ActuatorCarrier,
    ActuatorDatasetConfig,
    ActuatorMatchingDataset,
    ActuatorTaskSplit,
    CarrierConfig,
    make_carrier,
    make_dataset,
    make_task_spec,
)
from src.utils.artifacts import ExperimentRun


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp26_actuator_phase_diagram"
PROTOCOL_VERSION = "exp26_preregistered_v2_train_only_budget_bound"
MODES = ("frozen", "routing", "gain", "low_rank", "rgl")
PRIMARY_SINGLE_FAMILY_MODES = ("routing", "gain", "low_rank")
EVIDENCE_SCHEMA_VERSION = "exp26_formal_evidence_v1"
BUDGET_PREFLIGHT_SCHEMA_VERSION = "exp26_budget_preflight_v2_observed_bound"
BUDGET_PREFLIGHT_ACTIVE_MODES = tuple(mode for mode in MODES if mode != "frozen")
BUDGET_PREFLIGHT_FILES = (
    "code_tree_provenance.json",
    "config.json",
    "config.sha256",
    "generator_manifest.json",
    "manifest.sha256",
    "preflight_cells.jsonl",
    "preflight_summary.json",
)
BUDGET_PREFLIGHT_CRITICAL_CODE_FILES = (
    "experiments/exp26_formal_budget_preflight.py",
    "experiments/exp26_actuator_phase_diagram.py",
    "src/models/task_matched_actuators.py",
    "src/tasks/actuator_matching.py",
    "src/analysis/actuator_manifest.py",
)
RUNTIME_VERSION_DISTRIBUTIONS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit_learn": "scikit-learn",
    "pandas": "pandas",
    "statsmodels": "statsmodels",
}
_RUNTIME_CONFIG_KEYS = {
    "config_path",
    "evidence_provenance",
    "experiment",
    "run_label",
    "seed",
}


def canonical_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the scientific config without path/run-specific provenance."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    return {
        str(key): value
        for key, value in config.items()
        if str(key) not in _RUNTIME_CONFIG_KEYS
    }


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash canonical JSON, independent of path and platform line endings."""

    payload = canonical_config_payload(config)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_runtime_versions() -> dict[str, str | None]:
    """Return the exact Python and registered scientific-stack versions."""

    result: dict[str, str | None] = {"python": platform.python_version()}
    for label, distribution in RUNTIME_VERSION_DISTRIBUTIONS.items():
        try:
            result[label] = version(distribution)
        except PackageNotFoundError:
            result[label] = None
    return result


def git_identity(repository: Path = PROJECT_ROOT) -> dict[str, object]:
    """Return commit/tree/dirty receipts, or explicit unavailable values."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tree": None, "dirty": None}
    return {"commit": commit or None, "tree": tree or None, "dirty": bool(status)}


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Exp26 preflight JSON {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Exp26 preflight JSON must be an object: {path}")
    return dict(value)


def _json_array(path: Path) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Exp26 preflight JSON {path}") from error
    if not isinstance(value, list):
        raise ValueError(f"Exp26 preflight JSON must be an array: {path}")
    return value


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read Exp26 preflight cells {path}") from error
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Exp26 preflight cells must be non-empty JSONL")
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid Exp26 preflight cell at line {line_number}"
            ) from error
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Exp26 preflight cell {line_number} must be an object"
            )
        rows.append(dict(value))
    return rows


def _sha256_text(path: Path) -> str:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read Exp26 preflight hash {path}") from error
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid Exp26 preflight SHA-256 in {path}")
    return value


def _preflight_receipt_sha256(directory: Path) -> str:
    """Hash every immutable machine-readable file in a preflight receipt."""

    digest = hashlib.sha256()
    for name in BUDGET_PREFLIGHT_FILES:
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Exp26 preflight receipt lacks regular file {name}")
        payload = path.read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _critical_code_sha256(file_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative in BUDGET_PREFLIGHT_CRITICAL_CODE_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hashes[relative]))
    return digest.hexdigest()


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _exact_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _same_float(first: object, second: object, *, name: str) -> float:
    resolved = _finite_float(first, name=name)
    expected = _finite_float(second, name=f"expected {name}")
    if not np.isclose(resolved, expected, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Exp26 preflight {name} is inconsistent")
    return resolved


def _next_power_of_two(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Exp26 preflight policy ceiling input is invalid")
    return float(2.0 ** int(np.ceil(np.log2(value))))


def validate_budget_preflight_receipt(
    config: Mapping[str, Any],
    cells: Sequence[GeneratorCell],
    receipt_path: str | Path,
    *,
    current_git: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate a complete clean train-only receipt before a formal run starts."""

    if config.get("profile") != "formal" or config.get("dev_only") is not False:
        raise ValueError("budget preflight receipts are only valid for formal Exp26")
    seeds_value = config.get("seeds")
    if not isinstance(seeds_value, Sequence) or isinstance(
        seeds_value, (str, bytes)
    ):
        raise ValueError("formal Exp26 seeds must be a sequence")
    seeds = tuple(
        _exact_integer(seed, name="formal seed") for seed in seeds_value
    )
    if seeds != tuple(range(30)):
        raise ValueError("formal Exp26 requires the registered seeds 0--29")
    if len(cells) != 88 or len({cell.generator_id for cell in cells}) != 88:
        raise ValueError("formal Exp26 requires 88 unique registered generators")

    directory = Path(receipt_path)
    if not directory.is_dir():
        raise ValueError("formal Exp26 requires a preflight receipt directory")
    receipt_sha256 = _preflight_receipt_sha256(directory)
    receipt_config = _json_object(directory / "config.json")
    summary = _json_object(directory / "preflight_summary.json")
    code_tree = _json_object(directory / "code_tree_provenance.json")
    manifest_rows = _json_array(directory / "generator_manifest.json")
    raw_rows = _jsonl_objects(directory / "preflight_cells.jsonl")

    registered_config_sha = canonical_config_sha256(config)
    receipt_config_sha = canonical_config_sha256(receipt_config)
    if (
        receipt_config_sha != registered_config_sha
        or _sha256_text(directory / "config.sha256") != registered_config_sha
        or summary.get("config_sha256") != registered_config_sha
        or summary.get("canonical_config_sha256") != registered_config_sha
    ):
        raise ValueError("Exp26 preflight config SHA binding is invalid")

    expected_manifest_rows = [asdict(cell) for cell in cells]
    current_manifest_sha = manifest_hash(cells)
    if (
        manifest_rows != expected_manifest_rows
        or _sha256_text(directory / "manifest.sha256") != current_manifest_sha
        or summary.get("manifest_hash") != current_manifest_sha
        or config.get("manifest", {}).get("expected_hash") != current_manifest_sha
    ):
        raise ValueError("Exp26 preflight manifest binding is invalid")

    if (
        summary.get("schema_version") != BUDGET_PREFLIGHT_SCHEMA_VERSION
        or summary.get("receipt_schema") != BUDGET_PREFLIGHT_SCHEMA_VERSION
        or summary.get("registered_receipt_schema")
        != BUDGET_PREFLIGHT_SCHEMA_VERSION
        or summary.get("receipt_schema_matches") is not True
    ):
        raise ValueError("Exp26 preflight receipt schema is not registered")
    if summary.get("code_tree_provenance") != code_tree:
        raise ValueError("Exp26 preflight code-tree receipt is inconsistent")

    recorded_file_hashes = code_tree.get("critical_file_sha256")
    if not isinstance(recorded_file_hashes, Mapping):
        raise ValueError("Exp26 preflight critical-code hashes are missing")
    current_file_hashes = {
        relative: _file_sha256(PROJECT_ROOT / relative)
        for relative in BUDGET_PREFLIGHT_CRITICAL_CODE_FILES
    }
    if (
        dict(recorded_file_hashes) != current_file_hashes
        or code_tree.get("critical_code_sha256")
        != _critical_code_sha256(current_file_hashes)
    ):
        raise ValueError("Exp26 preflight critical-code hash binding is invalid")

    run_git = dict(git_identity() if current_git is None else current_git)
    commit = run_git.get("commit")
    tree = run_git.get("tree")
    if (
        not isinstance(commit, str)
        or not commit
        or not isinstance(tree, str)
        or not tree
        or run_git.get("dirty") is not False
    ):
        raise ValueError("formal Exp26 must start from a clean identifiable git tree")
    end_snapshot = code_tree.get("end_snapshot")
    if (
        code_tree.get("stable_during_run") is not True
        or code_tree.get("git_dirty") is not False
        or code_tree.get("git_commit") != commit
        or code_tree.get("git_tree") != tree
        or not isinstance(end_snapshot, Mapping)
        or end_snapshot.get("git_dirty") is not False
        or end_snapshot.get("git_commit") != commit
        or end_snapshot.get("git_tree") != tree
        or end_snapshot.get("critical_code_sha256")
        != code_tree.get("critical_code_sha256")
        or end_snapshot.get("worktree_content_sha256")
        != code_tree.get("worktree_content_sha256")
        or summary.get("git_commit") != commit
        or summary.get("git_tree") != tree
        or summary.get("provenance_clean") is not True
        or summary.get("provenance_stable_during_run") is not True
    ):
        raise ValueError("Exp26 preflight is not clean, stable, and git-identical")

    expected_keys = {
        (seed, cell.generator_id, mode)
        for seed in seeds
        for cell in cells
        for mode in BUDGET_PREFLIGHT_ACTIVE_MODES
    }
    observed_keys: list[tuple[int, str, str]] = []
    required_scales: list[float] = []
    cell_by_id = {cell.generator_id: asdict(cell) for cell in cells}
    configured_ceiling = _finite_float(
        config.get("actuator", {}).get("max_scale"),
        name="configured max_scale",
    )
    for row_number, row in enumerate(raw_rows, start=1):
        seed = _exact_integer(row.get("seed"), name=f"cell {row_number} seed")
        generator_id = row.get("generator_id")
        mode = row.get("actuator_mode")
        if not isinstance(generator_id, str) or not isinstance(mode, str):
            raise ValueError("Exp26 preflight cell identity is malformed")
        expected_cell = cell_by_id.get(generator_id)
        if expected_cell is None or any(
            row.get(name) != value for name, value in expected_cell.items()
        ):
            raise ValueError("Exp26 preflight generator metadata is inconsistent")
        observed_keys.append((seed, generator_id, mode))
        required = _finite_float(
            row.get("required_budget_scale"),
            name=f"cell {row_number} required scale",
        )
        if required <= 0.0:
            raise ValueError("Exp26 preflight required scales must be positive")
        required_scales.append(required)
        _same_float(
            row.get("registered_max_scale"),
            configured_ceiling,
            name=f"cell {row_number} registered max_scale",
        )
        if (
            row.get("fit_status") != "complete"
            or row.get("reachable_under_registered_max_scale") is not True
            or row.get("max_scale_exceeded") is not False
            or row.get("diagnostic_refit_used") is not False
            or row.get("validation_rollout_accessed") is not False
            or row.get("test_rollout_accessed") is not False
            or row.get("validation_behavior_accessed") is not False
            or row.get("test_behavior_accessed") is not False
        ):
            raise ValueError(
                "Exp26 preflight contains a failed, unreachable, or held-out cell"
            )
    if len(observed_keys) != len(expected_keys) or set(observed_keys) != expected_keys:
        raise ValueError("Exp26 preflight does not cover the complete formal panel")
    if len(set(observed_keys)) != len(observed_keys):
        raise ValueError("Exp26 preflight contains duplicate panel cells")

    expected_panel_size = len(expected_keys)
    count_bindings = {
        "n_records": expected_panel_size,
        "n_seeds": len(seeds),
        "n_generators": len(cells),
        "n_modes": len(BUDGET_PREFLIGHT_ACTIVE_MODES),
        "n_required_scales_recovered": expected_panel_size,
        "observed_panel_size": expected_panel_size,
        "registered_panel_size": expected_panel_size,
        "expected_panel_size": expected_panel_size,
        "n_unreachable": 0,
        "n_max_scale_blockers": 0,
        "n_other_failures": 0,
    }
    if any(
        _exact_integer(summary.get(name), name=f"summary {name}") != expected
        for name, expected in count_bindings.items()
    ):
        raise ValueError("Exp26 preflight panel counts are inconsistent")
    if summary.get("registered_active_modes") != list(
        BUDGET_PREFLIGHT_ACTIVE_MODES
    ):
        raise ValueError("Exp26 preflight actuator modes are inconsistent")
    if (
        summary.get("fit_scope") != "training_blocks_only"
        or summary.get("fit_entrypoint_matches_exp26") is not True
        or summary.get("validation_rollout_accessed") is not False
        or summary.get("test_rollout_accessed") is not False
        or summary.get("validation_behavior_accessed") is not False
        or summary.get("test_behavior_accessed") is not False
    ):
        raise ValueError("Exp26 preflight violates the train-only capability boundary")

    policy = config.get("budget_preflight")
    if not isinstance(policy, Mapping):
        raise ValueError("formal Exp26 lacks a registered budget policy")
    policy_required = _finite_float(
        policy.get("required_scale_max"), name="policy required scale max"
    )
    headroom = _finite_float(
        policy.get("headroom_multiplier"), name="policy headroom multiplier"
    )
    if policy.get("receipt_schema") != BUDGET_PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("formal Exp26 budget policy lacks the v2 receipt schema")
    if (
        policy.get("rounding_rule") != "next_power_of_two"
        or _exact_integer(
            policy.get("n_registered_active_fits"),
            name="registered active fit count",
        )
        != expected_panel_size
    ):
        raise ValueError("formal Exp26 budget policy is inconsistent")
    observed_max = max(required_scales)
    derived_ceiling = _next_power_of_two(headroom * policy_required)
    observed_ceiling = _next_power_of_two(headroom * observed_max)
    _same_float(
        observed_max,
        policy_required,
        name="observed required scale max",
    )
    for name, expected in {
        "observed_required_scale_max": observed_max,
        "registered_required_scale_max": policy_required,
        "policy_required_scale_max": policy_required,
        "registered_headroom_multiplier": headroom,
        "rounded_policy_ceiling": derived_ceiling,
        "observed_rounded_policy_ceiling": observed_ceiling,
        "derived_ceiling": derived_ceiling,
        "configured_max_scale": configured_ceiling,
    }.items():
        _same_float(summary.get(name), expected, name=f"summary {name}")
    if derived_ceiling != configured_ceiling or observed_ceiling != configured_ceiling:
        raise ValueError("Exp26 preflight policy ceiling differs from max_scale")
    required_true = (
        "policy_contract_valid",
        "policy_valid",
        "observed_max_matches",
        "record_ceiling_binding_valid",
        "ceiling_binding_valid",
        "panel_binding_valid",
        "all_required_scales_observed",
        "all_reachable_under_registered_max_scale",
        "preflight_passed",
    )
    if any(summary.get(name) is not True for name in required_true):
        raise ValueError("Exp26 preflight did not pass every registered gate")

    return {
        "required": True,
        "receipt_schema": BUDGET_PREFLIGHT_SCHEMA_VERSION,
        "receipt_sha256": receipt_sha256,
        "preflight_passed": True,
        "registered_config_sha256": registered_config_sha,
        "manifest_sha256": current_manifest_sha,
        "observed_required_scale_max": observed_max,
        "policy_required_scale_max": policy_required,
        "derived_ceiling": derived_ceiling,
        "provenance_clean": True,
        "provenance_stable_during_run": True,
        "git_commit": commit,
        "git_tree": tree,
    }


def build_evidence_provenance(
    config: Mapping[str, Any],
    *,
    manifest_sha256: str,
    budget_preflight: Mapping[str, object] | None = None,
    run_git: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind a run to its source config, analysis contract, code, and runtime."""

    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp26 config protocol_version is not registered")
    config_path_value = config.get("config_path")
    if not isinstance(config_path_value, str) or not config_path_value:
        raise ValueError("Exp26 evidence runs require config_path provenance")
    config_path = Path(config_path_value)
    try:
        source = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Exp26 source config {config_path}") from error
    if not isinstance(source, Mapping):
        raise ValueError("Exp26 source config must be a JSON object")
    if canonical_config_payload(source) != canonical_config_payload(config):
        raise ValueError("runtime Exp26 config differs from its source config")
    analysis = config.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("Exp26 config requires an analysis mapping")
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "canonical_config_sha256": canonical_config_sha256(config),
        "source_config_file_sha256": _file_sha256(config_path),
        "manifest_sha256": str(manifest_sha256),
        "analysis": {
            "tie_margin": float(analysis["tie_margin"]),
            "bootstrap_samples": int(analysis["bootstrap_samples"]),
            "permutation_samples": int(analysis["permutation_samples"]),
            "statistics_seed": int(analysis["statistics_seed"]),
        },
        "budget_preflight": (
            dict(budget_preflight) if budget_preflight is not None else None
        ),
        "git": dict(git_identity() if run_git is None else run_git),
        "runtime_versions": scientific_runtime_versions(),
    }


def evidence_row_fields(
    provenance: Mapping[str, Any],
    *,
    run_label: str | None,
) -> dict[str, object]:
    """Flatten the immutable provenance receipt into every metrics row."""

    analysis = dict(provenance["analysis"])
    git = dict(provenance["git"])
    versions = dict(provenance["runtime_versions"])
    budget_value = provenance.get("budget_preflight")
    budget = dict(budget_value) if isinstance(budget_value, Mapping) else None
    return {
        "evidence_schema_version": provenance["schema_version"],
        "formal_config_sha256": provenance["canonical_config_sha256"],
        "source_config_file_sha256": provenance["source_config_file_sha256"],
        "registered_manifest_sha256": provenance["manifest_sha256"],
        "registered_tie_margin": analysis["tie_margin"],
        "registered_bootstrap_samples": analysis["bootstrap_samples"],
        "registered_permutation_samples": analysis["permutation_samples"],
        "registered_statistics_seed": analysis["statistics_seed"],
        "run_git_commit": git["commit"],
        "run_git_tree": git["tree"],
        "run_git_dirty": git["dirty"],
        "run_python_version": versions["python"],
        "run_numpy_version": versions["numpy"],
        "run_scipy_version": versions["scipy"],
        "run_scikit_learn_version": versions["scikit_learn"],
        "run_pandas_version": versions["pandas"],
        "run_statsmodels_version": versions["statsmodels"],
        "run_label": run_label,
        "preflight_required": bool(budget is not None and budget["required"]),
        "preflight_passed": (
            bool(budget["preflight_passed"]) if budget is not None else None
        ),
        "preflight_receipt_sha256": (
            budget["receipt_sha256"] if budget is not None else None
        ),
        "preflight_git_commit": (
            budget["git_commit"] if budget is not None else None
        ),
        "preflight_git_tree": budget["git_tree"] if budget is not None else None,
    }


@dataclass(frozen=True)
class SharedReadout:
    """One train-fitted readout shared by all paired actuator modes."""

    mean: FloatArray
    scale: FloatArray
    model: Ridge

    def predict(self, features: FloatArray) -> IntArray:
        values = np.asarray(features, dtype=np.float64)
        normalized = (values - self.mean) / self.scale
        decision = np.asarray(self.model.predict(normalized), dtype=np.float64)
        return np.where(decision >= 0.0, 1, -1).astype(np.int64)


@dataclass(frozen=True)
class GeneratorSetup:
    cell: GeneratorCell
    dataset: ActuatorMatchingDataset
    readout: SharedReadout
    frozen_train: TaskMatchedRollout
    frozen_validation: TaskMatchedRollout
    frozen_test: TaskMatchedRollout
    target_train_scale: float
    demand_metrics: Mapping[str, object]


def _manifest(config: Mapping[str, Any]) -> tuple[GeneratorCell, ...]:
    options = dict(config["manifest"])
    cells = select_generator_manifest(
        options["grid"],
        per_alpha_per_split=int(options["per_alpha_per_split"]),
        selection_seed=int(options["selection_seed"]),
    )
    observed = manifest_hash(cells)
    expected = str(options["expected_hash"])
    if observed != expected:
        raise RuntimeError(
            f"Exp26 manifest hash mismatch: expected {expected}, observed {observed}"
        )
    return cells


def _planned_conditions(config: Mapping[str, Any]) -> list[dict[str, object]]:
    cells = _manifest(config)
    receipt = manifest_hash(cells)
    return [
        {
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
            "manifest_hash": receipt,
        }
        for cell in cells
        for mode in MODES
    ]


def _carrier_config(config: Mapping[str, Any]) -> CarrierConfig:
    return CarrierConfig(**dict(config["carrier"]))


def _dataset_config(config: Mapping[str, Any]) -> ActuatorDatasetConfig:
    options = dict(config["task"])
    return ActuatorDatasetConfig(
        n_train_blocks=int(options["n_train_blocks"]),
        n_validation_blocks=int(options["n_validation_blocks"]),
        n_test_blocks=int(options["n_test_blocks"]),
        trials_per_block=int(options["trials_per_block"]),
        input_steps=int(options["input_steps"]),
        input_std=float(options["input_std"]),
    )


def _actuator_config(config: Mapping[str, Any]) -> TaskMatchedActuatorConfig:
    options = dict(config["actuator"])
    return TaskMatchedActuatorConfig(
        rank_a=int(options["rank_a_capacity"]),
        rank_b=int(options["rank_b_capacity"]),
        ridge=float(options["ridge"]),
        max_scale=float(options["max_scale"]),
        degeneracy_tolerance=float(options["degeneracy_tolerance"]),
        budget_relative_tolerance=float(options["budget_relative_tolerance"]),
        context_center_tolerance=float(options["context_center_tolerance"]),
    )


def _control_observable(
    target_states: FloatArray,
    frozen_states: FloatArray,
    observation: FloatArray,
) -> FloatArray:
    return (target_states[:, -1] - frozen_states[:, -1]) @ observation.T


def _control_labels(features: FloatArray) -> IntArray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("control features must have shape [trial, output]")
    if np.any(values[:, 0] == 0.0):
        raise RuntimeError("control-induced target behavior has an exact zero margin")
    return np.where(values[:, 0] > 0.0, 1, -1).astype(np.int64)


def _fit_shared_readout(
    split: ActuatorTaskSplit,
    frozen: TaskMatchedRollout,
    observation: FloatArray,
    *,
    ridge: float,
) -> SharedReadout:
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("analysis.readout_ridge must be finite and non-negative")
    features = _control_observable(
        split.target_states,
        frozen.states,
        observation,
    )
    labels = _control_labels(features)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    model = Ridge(alpha=ridge).fit((features - mean) / scale, labels)
    mean.setflags(write=False)
    scale.setflags(write=False)
    return SharedReadout(mean=mean, scale=scale, model=model)


def _balanced_accuracy(labels: IntArray, predictions: IntArray) -> float:
    return float(balanced_accuracy_score(labels, predictions))


def _context_tape(split: ActuatorTaskSplit) -> FloatArray:
    return np.broadcast_to(
        split.contexts[:, np.newaxis],
        split.inputs.shape[:2],
    ).astype(np.float64, copy=True)


def _rollout(
    actuator: TaskMatchedActuator,
    split: ActuatorTaskSplit,
) -> TaskMatchedRollout:
    return actuator.rollout(
        split.target_states[:, 0],
        split.inputs,
        _context_tape(split),
        process_noise=split.noise,
    )


def _registered_second_moments(
    carrier: ActuatorCarrier,
    dataset_config: ActuatorDatasetConfig,
    *,
    delay: int,
    noise_std: float,
) -> tuple[FloatArray, FloatArray]:
    """Return analytic baseline moments under the registered white tapes."""

    horizon = dataset_config.input_steps + int(delay)
    if horizon < 1:
        raise ValueError("registered task horizon must be positive")
    n_state = carrier.config.n_neurons
    n_input = carrier.config.n_inputs
    state_moments = np.empty((horizon, n_state, n_state), dtype=np.float64)
    input_moments = np.zeros((horizon, n_input, n_input), dtype=np.float64)
    occupancy = np.zeros((n_state, n_state), dtype=np.float64)
    noise_covariance = float(noise_std) ** 2 * np.eye(n_state)
    for step in range(horizon):
        state_moments[step] = occupancy
        if step < dataset_config.input_steps:
            input_moments[step] = (
                dataset_config.input_std**2 * np.eye(n_input)
            )
        occupancy = (
            carrier.a0 @ occupancy @ carrier.a0.T
            + carrier.b0 @ input_moments[step] @ carrier.b0.T
            + noise_covariance
        )
        occupancy = 0.5 * (occupancy + occupancy.T)
    return state_moments, input_moments


def _empirical_cross_moments(split: ActuatorTaskSplit) -> FloatArray:
    states = np.asarray(split.target_states[:, :-1], dtype=np.float64)
    inputs = np.asarray(split.inputs, dtype=np.float64)
    return np.einsum("bti,btj->tij", states, inputs) / states.shape[0]


def _demand_metrics(
    config: Mapping[str, Any],
    carrier: ActuatorCarrier,
    dataset: ActuatorMatchingDataset,
    moments: tuple[FloatArray, FloatArray],
) -> dict[str, object]:
    analysis = dict(config["analysis"])
    spec = dataset.spec
    actual_delta_a = spec.alpha * spec.delta_a
    actual_delta_b = (1.0 - spec.alpha) * spec.delta_b
    state_moments, input_moments = moments
    demand = finite_horizon_local_demand(
        carrier.a0,
        carrier.c,
        actual_delta_a,
        actual_delta_b,
        state_moments,
        input_moments,
        cross_relative_tolerance=float(analysis["cross_relative_tolerance"]),
    )
    empirical_cross = finite_horizon_local_demand(
        carrier.a0,
        carrier.c,
        actual_delta_a,
        actual_delta_b,
        state_moments,
        input_moments,
        state_input_cross_moments=_empirical_cross_moments(dataset.train),
        cross_relative_tolerance=float(analysis["cross_relative_tolerance"]),
    )
    average_input_moment = np.mean(input_moments, axis=0)
    controllability, observability = control_gramians(
        carrier.a0,
        carrier.b0,
        carrier.c,
        input_second_moment=average_input_moment,
        horizon=dataset.train.n_steps,
    )
    rank = transition_rank_requirement(
        actual_delta_a,
        controllability,
        observability,
        candidate_ranks=(0, 1, 2, 4, 8),
        support_rtol=float(analysis["support_rtol"]),
        rank_rtol=float(analysis["rank_rtol"]),
    )
    return {
        "chi": demand.state_fraction,
        "chi_energy": demand.state_energy_fraction,
        "state_demand": demand.state_demand,
        "input_demand": demand.input_demand,
        "demand_horizon": demand.horizon,
        "demand_definition": "finite_horizon_local_injection_train_law",
        "demand_uses_actual_context_difference": True,
        "demand_input_second_moment_included": True,
        "demand_cross_energy": demand.cross_energy,
        "demand_cross_relative_magnitude": demand.cross_relative_magnitude,
        "demand_marginal_decomposition_valid": (
            demand.marginal_decomposition_valid
        ),
        "generator_state_input_cross_moment_zero_by_construction": True,
        "empirical_train_cross_energy": empirical_cross.cross_energy,
        "empirical_train_cross_relative_magnitude": (
            empirical_cross.cross_relative_magnitude
        ),
        "empirical_train_cross_within_tolerance": (
            empirical_cross.marginal_decomposition_valid
        ),
        "chi_minus_alpha": demand.state_fraction - spec.alpha,
        "chi_is_not_defined_by_alpha": True,
        "delta_a_independent_amplitude": spec.delta_a_amplitude,
        "delta_b_independent_amplitude": spec.delta_b_amplitude,
        "amplitudes_equalized_by_demand": False,
        "transition_rank_raw": rank.raw_rank,
        "transition_rank_projected": rank.projected_rank,
        "transition_energy_rank_99": rank.energy_rank_99,
        "transition_energy_rank_999": rank.energy_rank_999,
        "transition_rank_candidates": list(rank.candidate_ranks),
        "transition_rank_tail_energy_fractions": list(
            rank.tail_energy_fractions
        ),
    }


def _target_train_scale(dataset: ActuatorMatchingDataset) -> float:
    states = np.asarray(dataset.train.target_states[:, 1:], dtype=np.float64)
    centered = states - np.mean(states, axis=(0, 1), keepdims=True)
    scale = float(np.sqrt(np.mean(centered * centered)))
    if scale <= 1e-12:
        raise RuntimeError("training target state scale is degenerate")
    return scale


def _setup_generator(
    config: Mapping[str, Any],
    carrier: ActuatorCarrier,
    cell: GeneratorCell,
    *,
    seed: int,
    moment_cache: dict[tuple[int, float], tuple[FloatArray, FloatArray]],
) -> GeneratorSetup:
    task = dict(config["task"])
    spec = make_task_spec(
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
    dataset_config = _dataset_config(config)
    dataset = make_dataset(spec, dataset_config, seed=seed)
    cache_key = (cell.delay, cell.noise_std)
    if cache_key not in moment_cache:
        moment_cache[cache_key] = _registered_second_moments(
            carrier,
            dataset_config,
            delay=cell.delay,
            noise_std=cell.noise_std,
        )
    frozen_actuator = fit_task_matched_actuator(
        dataset.train.target_states,
        dataset.train.inputs,
        _context_tape(dataset.train),
        carrier.a0,
        carrier.b0,
        mode=ActuatorMode.FROZEN,
        process_noise=dataset.train.noise,
        config=_actuator_config(config),
    )
    frozen_train = _rollout(frozen_actuator, dataset.train)
    frozen_validation = _rollout(frozen_actuator, dataset.validation)
    frozen_test = _rollout(frozen_actuator, dataset.test)
    readout = _fit_shared_readout(
        dataset.train,
        frozen_train,
        carrier.c,
        ridge=float(dict(config["analysis"])["readout_ridge"]),
    )
    return GeneratorSetup(
        cell=cell,
        dataset=dataset,
        readout=readout,
        frozen_train=frozen_train,
        frozen_validation=frozen_validation,
        frozen_test=frozen_test,
        target_train_scale=_target_train_scale(dataset),
        demand_metrics=_demand_metrics(
            config, carrier, dataset, moment_cache[cache_key]
        ),
    )


def _cosine(first: FloatArray, second: FloatArray) -> float:
    first_flat = np.asarray(first, dtype=np.float64).ravel()
    second_flat = np.asarray(second, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(first_flat) * np.linalg.norm(second_flat))
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(first_flat, second_flat) / denominator)


def _normalized_state_error(
    target: FloatArray,
    prediction: FloatArray,
    *,
    train_scale: float,
) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)) / train_scale)


def _output_r2(
    target: FloatArray,
    prediction: FloatArray,
    observation: FloatArray,
) -> float:
    target_output = np.einsum("bti,oi->bto", target, observation).reshape(-1)
    predicted_output = np.einsum(
        "bti,oi->bto", prediction, observation
    ).reshape(-1)
    return float(r2_score(target_output, predicted_output))


def _effective_spectral_radii(actuator: TaskMatchedActuator) -> tuple[float, float]:
    gain_a = actuator.gain[:, np.newaxis] * actuator.baseline_a
    radii = []
    for context in (-1.0, 1.0):
        effective = actuator.baseline_a + context * (
            actuator.delta_a + gain_a
        )
        radii.append(float(np.max(np.abs(np.linalg.eigvals(effective)))))
    return float(radii[0]), float(radii[1])


def _split_metrics(
    setup: GeneratorSetup,
    split: ActuatorTaskSplit,
    rollout: TaskMatchedRollout,
    frozen: TaskMatchedRollout,
    *,
    prefix: str,
) -> dict[str, object]:
    target = np.asarray(split.target_states, dtype=np.float64)
    prediction = np.asarray(rollout.states, dtype=np.float64)
    target_control = _control_observable(
        target,
        frozen.states,
        setup.dataset.spec.carrier.c,
    )
    predicted_control = _control_observable(
        prediction,
        frozen.states,
        setup.dataset.spec.carrier.c,
    )
    target_labels = _control_labels(target_control)
    labels = setup.readout.predict(predicted_control)
    absolute_predictions = np.where(
        prediction[:, -1] @ setup.dataset.spec.carrier.c[0] >= 0.0,
        1,
        -1,
    ).astype(np.int64)
    delay_start = split.input_steps + 1
    delay_target = target[:, delay_start:]
    delay_prediction = prediction[:, delay_start:]
    if delay_target.shape[1] == 0:
        delay_error: float | None = None
    else:
        delay_error = _normalized_state_error(
            delay_target,
            delay_prediction,
            train_scale=setup.target_train_scale,
        )
    return {
        f"{prefix}_balanced_accuracy": _balanced_accuracy(target_labels, labels),
        f"{prefix}_behavior_endpoint": "control_induced_observable_sign",
        f"{prefix}_absolute_observation_balanced_accuracy": _balanced_accuracy(
            split.labels, absolute_predictions
        ),
        f"{prefix}_state_normalized_rmse": _normalized_state_error(
            target,
            prediction,
            train_scale=setup.target_train_scale,
        ),
        f"{prefix}_zero_input_normalized_rmse": delay_error,
        f"{prefix}_output_r2": _output_r2(
            target,
            prediction,
            setup.dataset.spec.carrier.c,
        ),
        f"{prefix}_correction_event_proxy_mean": float(
            np.mean(rollout.event_proxy_by_step)
        ),
        f"{prefix}_tape_fingerprint": rollout.tape_fingerprint,
    }


def _condition_metrics(
    config: Mapping[str, Any],
    setup: GeneratorSetup,
    *,
    mode: str,
) -> tuple[dict[str, object], bool]:
    dataset = setup.dataset
    contexts = _context_tape(dataset.train)
    actuator = fit_task_matched_actuator(
        dataset.train.target_states,
        dataset.train.inputs,
        contexts,
        dataset.spec.carrier.a0,
        dataset.spec.carrier.b0,
        mode=mode,
        process_noise=dataset.train.noise,
        config=_actuator_config(config),
    )
    train = _rollout(actuator, dataset.train)
    validation = _rollout(actuator, dataset.validation)
    test = _rollout(actuator, dataset.test)
    radii = _effective_spectral_radii(actuator)
    budget_applicable = mode != ActuatorMode.FROZEN.value
    budget_valid = bool(
        not budget_applicable
        or actuator.receipt.budget_l2_relative_error
        <= _actuator_config(config).budget_relative_tolerance
    )
    stable = bool(max(radii) < 1.0)
    target_train_ba = _balanced_accuracy(
        _control_labels(
            _control_observable(
                dataset.train.target_states,
                setup.frozen_train.states,
                dataset.spec.carrier.c,
            )
        ),
        setup.readout.predict(
            _control_observable(
                dataset.train.target_states,
                setup.frozen_train.states,
                dataset.spec.carrier.c,
            )
        ),
    )
    target_validation_ba = _balanced_accuracy(
        _control_labels(
            _control_observable(
                dataset.validation.target_states,
                setup.frozen_validation.states,
                dataset.spec.carrier.c,
            )
        ),
        setup.readout.predict(
            _control_observable(
                dataset.validation.target_states,
                setup.frozen_validation.states,
                dataset.spec.carrier.c,
            )
        ),
    )
    target_test_ba = _balanced_accuracy(
        _control_labels(
            _control_observable(
                dataset.test.target_states,
                setup.frozen_test.states,
                dataset.spec.carrier.c,
            )
        ),
        setup.readout.predict(
            _control_observable(
                dataset.test.target_states,
                setup.frozen_test.states,
                dataset.spec.carrier.c,
            )
        ),
    )
    metrics: dict[str, object] = {
        "status": "complete",
        "experiment_protocol_version": PROTOCOL_VERSION,
        "statistics_unit": "seed",
        "split_unit": "block",
        "time_points_randomly_split": False,
        "profile": str(config["profile"]),
        "dev_only": bool(config.get("dev_only", False)),
        "training_algorithm": str(config["training_algorithm"]),
        "used_autograd": False,
        "used_bptt": False,
        "local_learning_enabled": False,
        "oracle_task_matched_actuator_ceiling": True,
        "selector_learning_enabled": False,
        "rgl_is_combined_ceiling_only": True,
        "optimal_label_uses_single_family_only": True,
        "shared_scalar_centered_context_control": True,
        "belief_dimension": 1,
        "operator_rank_separate_from_belief_dimension": True,
        "frozen_high_rank_dale_compatible_base": True,
        "effective_corrections_dale_constrained": False,
        "base_recurrent_rank": int(
            np.linalg.matrix_rank(dataset.spec.carrier.a0)
        ),
        "base_recurrent_spectral_radius": dataset.spec.carrier.spectral_radius,
        "base_recurrent_fingerprint": dataset.spec.carrier.fingerprint,
        "task_spec_fingerprint": dataset.spec.fingerprint,
        "dataset_fingerprint": dataset.fingerprint,
        "train_split_fingerprint": dataset.train.fingerprint,
        "validation_split_fingerprint": dataset.validation.fingerprint,
        "test_split_fingerprint": dataset.test.fingerprint,
        "paired_base_across_modes": True,
        "paired_data_across_modes": True,
        "paired_initial_state_across_modes": True,
        "paired_context_across_modes": True,
        "paired_noise_across_modes": True,
        "readout_fit_train_only": True,
        "readout_shared_across_modes": True,
        "primary_behavior_is_control_induced_causal_contrast": True,
        "absolute_behavior_reported_secondary": True,
        "target_train_balanced_accuracy": target_train_ba,
        "target_validation_balanced_accuracy": target_validation_ba,
        "target_test_balanced_accuracy": target_test_ba,
        "functional_budget_type": "train_teacher_forced_correction_current_l2_rms",
        "functional_budget_fit_scope": "training_blocks_only",
        "functional_budget_target_uses_behavior": False,
        "functional_budget_applicable": budget_applicable,
        "functional_budget_valid": budget_valid,
        "functional_budget_target_l2_rms": actuator.receipt.target_l2_rms,
        "functional_budget_matched_l2_rms": (
            actuator.receipt.matched_current_l2_rms
        ),
        "functional_budget_l2_relative_error": (
            actuator.receipt.budget_l2_relative_error
        ),
        "functional_current_l1_reported_not_matched": True,
        "functional_current_l1_mean": actuator.receipt.matched_current_l1_mean,
        "budget_scale": actuator.receipt.budget_scale,
        "teacher_forced_error_rms": actuator.receipt.teacher_forced_error_rms,
        "teacher_forced_explained_fraction": (
            actuator.receipt.teacher_forced_explained_fraction
        ),
        "actuator_rank_a_capacity": actuator.receipt.rank_a_limit,
        "actuator_rank_b_capacity": actuator.receipt.rank_b_limit,
        "fitted_recurrent_rank": actuator.receipt.recurrent_rank,
        "fitted_input_rank": actuator.receipt.input_rank,
        "fitted_gain_rank": actuator.receipt.gain_rank,
        "raw_recurrent_fit_rank": actuator.receipt.raw_recurrent_rank,
        "raw_input_fit_rank": actuator.receipt.raw_input_rank,
        "recurrent_task_alignment_cosine": _cosine(
            actuator.delta_a,
            0.5 * dataset.spec.alpha * dataset.spec.delta_a,
        ),
        "input_task_alignment_cosine": _cosine(
            actuator.delta_b,
            0.5 * (1.0 - dataset.spec.alpha) * dataset.spec.delta_b,
        ),
        "effective_context_minus_spectral_radius": radii[0],
        "effective_context_plus_spectral_radius": radii[1],
        "effective_dynamics_strictly_stable": stable,
        "training_fingerprint": actuator.receipt.training_fingerprint,
        "training_noise_fingerprint": actuator.receipt.process_noise_fingerprint,
        "correction_fingerprint": actuator.receipt.correction_fingerprint,
        **setup.demand_metrics,
        **_split_metrics(
            setup,
            dataset.train,
            train,
            setup.frozen_train,
            prefix="train",
        ),
        **_split_metrics(
            setup,
            dataset.validation,
            validation,
            setup.frozen_validation,
            prefix="validation",
        ),
        **_split_metrics(
            setup,
            dataset.test,
            test,
            setup.frozen_test,
            prefix="test",
        ),
    }
    return metrics, bool(budget_valid and stable)


def _dimensions(
    cell: GeneratorCell,
    *,
    mode: str,
    receipt: str,
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
        "manifest_hash": receipt,
        **evidence,
    }


def run_seed(
    config: dict[str, Any],
    seed: int,
    results_root: str | Path,
    *,
    run_label: str | None = None,
    preflight_receipt: str | Path | None = None,
) -> Path:
    if config.get("profile") == "formal" and (
        not isinstance(run_label, str) or not run_label
    ):
        raise ValueError("formal Exp26 requires a non-empty shared run_label")
    registered_seeds = config.get("seeds")
    if not isinstance(registered_seeds, Sequence) or isinstance(
        registered_seeds, (str, bytes)
    ):
        raise ValueError("Exp26 config seeds must be a sequence")
    resolved_seeds = tuple(
        _exact_integer(value, name="registered seed") for value in registered_seeds
    )
    requested_seed = _exact_integer(seed, name="requested seed")
    if requested_seed not in resolved_seeds:
        raise ValueError("requested Exp26 seed is not registered in the config")
    seed = requested_seed
    cells = _manifest(config)
    receipt = manifest_hash(cells)
    run_git = git_identity()
    if config.get("profile") == "formal":
        if preflight_receipt is None:
            raise ValueError("formal Exp26 requires --preflight-receipt")
        budget_preflight = validate_budget_preflight_receipt(
            config,
            cells,
            preflight_receipt,
            current_git=run_git,
        )
    else:
        if preflight_receipt is not None:
            raise ValueError("preflight receipts are only accepted for formal Exp26")
        budget_preflight = None
    initialize_seed(seed)
    provenance = build_evidence_provenance(
        config,
        manifest_sha256=receipt,
        budget_preflight=budget_preflight,
        run_git=run_git,
    )
    evidence = evidence_row_fields(provenance, run_label=run_label)
    run_config = {**config, "evidence_provenance": provenance}
    carrier = make_carrier(_carrier_config(run_config), seed)
    moment_cache: dict[tuple[int, float], tuple[FloatArray, FloatArray]] = {}
    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
        run_label=run_label,
    ) as run:
        run.register_conditions(_planned_conditions(run_config))
        for cell in cells:
            try:
                setup = _setup_generator(
                    run_config,
                    carrier,
                    cell,
                    seed=seed,
                    moment_cache=moment_cache,
                )
            except Exception as error:
                for mode in MODES:
                    run.mark_condition_failure(
                        error,
                        **_dimensions(
                            cell,
                            mode=mode,
                            receipt=receipt,
                            evidence=evidence,
                        ),
                    )
                continue
            for mode in MODES:
                dimensions = _dimensions(
                    cell,
                    mode=mode,
                    receipt=receipt,
                    evidence=evidence,
                )
                try:
                    metrics, valid = _condition_metrics(
                        run_config,
                        setup,
                        mode=mode,
                    )
                    if valid:
                        run.record(metrics, **dimensions)
                    else:
                        failed = []
                        if not metrics["functional_budget_valid"]:
                            failed.append("functional_budget")
                        if not metrics["effective_dynamics_strictly_stable"]:
                            failed.append("effective_stability")
                        metrics["failure_reason"] = ",".join(failed)
                        run.record_failed_condition(metrics, **dimensions)
                except Exception as error:
                    run.mark_condition_failure(error, **dimensions)
        if config.get("profile") == "formal":
            end_git = git_identity()
            if any(
                end_git.get(name) != run_git.get(name)
                for name in ("commit", "tree", "dirty")
            ):
                raise RuntimeError(
                    "formal Exp26 git identity changed during the seed run"
                )
        return run.path


def _selected_seeds(config: dict[str, Any], override: str | None) -> Iterable[int]:
    return seed_list(override if override is not None else config["seeds"])


def main() -> None:
    parser = basic_parser(
        "Exp26 preregistered task-actuator matching phase diagram",
        "configs/formal/exp26_actuator_phase_diagram.json",
    )
    parser.add_argument(
        "--run-label",
        help="path-safe label shared by every seed in one immutable run panel",
    )
    parser.add_argument(
        "--preflight-receipt",
        type=Path,
        help=(
            "clean v2 train-only budget-preflight receipt directory; required "
            "for the formal profile"
        ),
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    if config.get("profile") == "formal" and args.preflight_receipt is None:
        parser.error("formal Exp26 requires --preflight-receipt")
    if config.get("profile") == "formal" and not args.run_label:
        parser.error("formal Exp26 requires --run-label")
    for seed in _selected_seeds(config, args.seeds):
        path = run_seed(
            config,
            seed,
            args.results_root,
            run_label=args.run_label,
            preflight_receipt=args.preflight_receipt,
        )
        print(path)


if __name__ == "__main__":
    main()
