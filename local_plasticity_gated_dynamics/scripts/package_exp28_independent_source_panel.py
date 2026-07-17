"""Collect and hash-bind the source-only Exp28 independent Exp26 panel.

The package is intentionally non-inferential.  Even a complete valid panel is
labelled ``inconclusive`` until a separately preregistered downstream selector
analysis consumes it.  Missing seeds and scientific failures are retained;
duplicate attempts, unexpected cells, or broken provenance fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import exp26_actuator_phase_diagram as exp26  # noqa: E402
from experiments.common import load_json_config  # noqa: E402
from experiments.exp28_exp26_independent_source_panel import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    EXPECTED_MODES,
    EXPECTED_PANEL_ROWS,
    EXPECTED_ROWS_PER_SEED,
    EXPECTED_SEEDS,
    EXPERIMENT,
    PROFILE,
    REQUIRED_RUN_LABEL,
    _ROW_EVIDENCE_FIELDS,
    _canonical_sha256,
    _dimensions,
    _file_sha256,
    build_evidence_provenance,
    canonical_config_payload,
    evidence_row_fields,
    planned_conditions,
    validate_source_contract,
)
from src.analysis.actuator_manifest import manifest_hash  # noqa: E402


SOURCE_PACKAGE_SCHEMA_VERSION = "exp28_independent_source_package_v1"
TERMINAL_STATUSES = {"complete", "complete_with_failures"}
ROW_STATUSES = {"complete", "failed", "invalid"}
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "formal"
    / "exp28_exp26_independent_source_panel.json"
)
ATTEMPT_FILES = (
    "config.json",
    "environment.json",
    "manifest.json",
    "metrics.jsonl",
    "planned_conditions.json",
    "status.json",
)


@dataclass(frozen=True)
class AttemptReceipt:
    seed: int
    path: str
    run_status: str
    run_id: str
    planned_coverage_valid: bool
    observed_row_count: int
    observed_complete_rows: int
    observed_failed_rows: int
    observed_invalid_rows: int
    file_sha256: Mapping[str, str]


@dataclass(frozen=True)
class PanelCollection:
    """One non-selective view of every attempt and every raw row."""

    rows: tuple[Mapping[str, Any], ...]
    attempts: tuple[AttemptReceipt, ...]
    config: Mapping[str, Any]
    config_sha256: str
    config_file_sha256: str
    source_contract: Mapping[str, Any]
    provenance_identity: Mapping[str, Any] | None


@dataclass(frozen=True)
class SourcePanelPackage:
    """Validated downstream interface; rows are in deterministic panel order."""

    receipt: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    receipt_payload_sha256: str
    receipt_file_sha256: str
    raw_metrics_sha256: str


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read independent source artifact {path}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read independent source rows {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank independent source row at {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid independent source row at {path}:{line_number}"
            ) from error
        if not isinstance(value, Mapping):
            raise ValueError(
                f"independent source row must be an object at {path}:{line_number}"
            )
        rows.append(dict(value))
    return rows


def _strict_true(value: object) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _attempt_candidates(
    results_root: Path,
    *,
    run_label: str,
) -> dict[int, list[Path]]:
    base = results_root / "runs" / EXPERIMENT
    if not base.exists():
        return {}
    candidates: dict[int, list[Path]] = {}
    for metrics_path in sorted(base.glob("seed_*/*/metrics.jsonl")):
        attempt = metrics_path.parent
        config_path = attempt / "config.json"
        if not config_path.is_file():
            raise ValueError(f"attempt with metrics lacks config.json: {attempt}")
        value = _read_json(config_path)
        if not isinstance(value, Mapping):
            raise ValueError(f"attempt config is not an object: {config_path}")
        if value.get("experiment") != EXPERIMENT:
            raise ValueError(f"wrong experiment identity in {config_path}")
        if value.get("run_label") != run_label:
            continue
        seed_value = value.get("seed")
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise ValueError(f"invalid seed in {config_path}")
        seed = int(seed_value)
        if seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected independent source seed {seed}")
        if attempt.parent.name != f"seed_{seed:04d}":
            raise ValueError(f"attempt path seed disagrees with {config_path}")
        candidates.setdefault(seed, []).append(attempt)
    return candidates


def _expected_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"condition_index": index, **row}
        for index, row in enumerate(planned_conditions(config))
    ]


def _planned_coverage_valid(path: Path, config: Mapping[str, Any]) -> bool:
    value = _read_json(path)
    return bool(isinstance(value, list) and value == _expected_plan(config))


def _environment_matches(
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> bool:
    versions = provenance.get("runtime_versions")
    run_git = provenance.get("run_git")
    packages = environment.get("packages")
    environment_git = environment.get("git")
    if not all(
        isinstance(value, Mapping)
        for value in (versions, run_git, packages, environment_git)
    ):
        return False
    package_map = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "statsmodels": "statsmodels",
    }
    return bool(
        all(
            packages.get(distribution) == versions.get(label)
            for label, distribution in package_map.items()
        )
        and str(environment.get("python", "")).startswith(
            f"{versions.get('python')} "
        )
        and dict(environment_git) == dict(run_git)
    )


def _validate_attempt(
    attempt: Path,
    *,
    seed: int,
    registered_config: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
) -> tuple[AttemptReceipt, list[dict[str, Any]]]:
    required = [name for name in ATTEMPT_FILES if not (attempt / name).is_file()]
    if required:
        raise ValueError(f"independent source attempt lacks files {required}: {attempt}")
    config = _read_json(attempt / "config.json")
    status = _read_json(attempt / "status.json")
    manifest = _read_json(attempt / "manifest.json")
    environment = _read_json(attempt / "environment.json")
    if not all(
        isinstance(value, Mapping)
        for value in (config, status, manifest, environment)
    ):
        raise ValueError(f"malformed independent source metadata: {attempt}")
    if canonical_config_payload(config) != canonical_config_payload(
        registered_config
    ):
        raise ValueError(f"attempt config differs from registration: {attempt}")
    run_status = str(status.get("status", "missing"))
    run_id = manifest.get("run_id")
    if (
        config.get("experiment") != EXPERIMENT
        or config.get("seed") != seed
        or config.get("run_label") != REQUIRED_RUN_LABEL
        or config.get("profile") != PROFILE
        or status.get("seed") != seed
        or status.get("run_label") != REQUIRED_RUN_LABEL
        or manifest.get("experiment") != EXPERIMENT
        or manifest.get("seed") != seed
        or manifest.get("run_label") != REQUIRED_RUN_LABEL
        or manifest.get("profile") != PROFILE
        or manifest.get("status") != run_status
        or not isinstance(run_id, str)
        or not run_id
    ):
        raise ValueError(f"attempt identity is inconsistent: {attempt}")
    provenance = config.get("evidence_provenance")
    if (
        not isinstance(provenance, Mapping)
        or dict(provenance) != dict(expected_provenance)
        or manifest.get("evidence_provenance") != provenance
        or provenance.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or not _environment_matches(environment, provenance)
    ):
        raise ValueError(f"attempt evidence provenance is inconsistent: {attempt}")

    cells = exp26._manifest(registered_config)
    receipt = manifest_hash(cells)
    expected_row_evidence = {
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
        "run_git_commit": provenance["run_git"]["commit"],
        "run_git_tree": provenance["run_git"]["tree"],
        "run_git_dirty": provenance["run_git"]["dirty"],
        "run_python_version": provenance["runtime_versions"]["python"],
        "run_numpy_version": provenance["runtime_versions"]["numpy"],
        "run_scipy_version": provenance["runtime_versions"]["scipy"],
        "run_scikit_learn_version": provenance["runtime_versions"][
            "scikit_learn"
        ],
        "run_pandas_version": provenance["runtime_versions"]["pandas"],
        "run_statsmodels_version": provenance["runtime_versions"]["statsmodels"],
        "run_label": REQUIRED_RUN_LABEL,
        "source_only": True,
        "standalone_inference_permitted": False,
    }
    expected_dimensions = {
        (cell.generator_id, mode): _dimensions(
            cell,
            mode=mode,
            manifest_receipt=receipt,
            evidence=expected_row_evidence,
        )
        for cell in cells
        for mode in EXPECTED_MODES
    }
    raw_rows = _read_jsonl(attempt / "metrics.jsonl")
    observed_keys: list[tuple[str, str]] = []
    normalized_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if (
            row.get("experiment") != EXPERIMENT
            or row.get("seed") != seed
            or row.get("run_id") != run_id
            or row.get("status") not in ROW_STATUSES
        ):
            raise ValueError(f"malformed independent source metric row: {attempt}")
        key = (str(row.get("generator_id")), str(row.get("actuator_mode")))
        dimensions = expected_dimensions.get(key)
        if dimensions is None or any(row.get(name) != value for name, value in dimensions.items()):
            raise ValueError(f"unexpected/tampered independent source cell: {attempt}")
        observed_keys.append(key)
        row_profile = row.get("profile")
        if row_profile not in (None, PROFILE):
            raise ValueError(f"independent source row has wrong profile: {attempt}")
        normalized_rows.append(
            {
                **row,
                "profile": PROFILE,
                "_attempt_path": str(attempt.resolve()),
                "_run_status": run_status,
                "_effective_status": (
                    str(row["status"])
                    if run_status in TERMINAL_STATUSES
                    else "failed"
                ),
            }
        )
    if len(observed_keys) != len(set(observed_keys)):
        raise ValueError(f"duplicate independent source cells: {attempt}")
    counts = {status: 0 for status in ROW_STATUSES}
    for row in raw_rows:
        counts[str(row["status"])] += 1
    files = {name: _file_sha256(attempt / name) for name in ATTEMPT_FILES}
    return (
        AttemptReceipt(
            seed=seed,
            path=str(attempt.resolve()),
            run_status=run_status,
            run_id=run_id,
            planned_coverage_valid=_planned_coverage_valid(
                attempt / "planned_conditions.json", registered_config
            ),
            observed_row_count=len(raw_rows),
            observed_complete_rows=counts["complete"],
            observed_failed_rows=counts["failed"],
            observed_invalid_rows=counts["invalid"],
            file_sha256=files,
        ),
        normalized_rows,
    )


def collect_source_panel(
    results_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    run_label: str = REQUIRED_RUN_LABEL,
    current_git: Mapping[str, object] | None = None,
    runtime_versions: Mapping[str, object] | None = None,
) -> PanelCollection:
    """Collect all matching attempts without selecting by success or recency."""

    if run_label != REQUIRED_RUN_LABEL:
        raise ValueError(f"run_label must equal {REQUIRED_RUN_LABEL}")
    registered_config = load_json_config(config_path)
    contract = validate_source_contract(
        registered_config,
        current_git=current_git,
        runtime_versions=runtime_versions,
    )
    provenance = build_evidence_provenance(contract, run_label=run_label)
    candidates = _attempt_candidates(Path(results_root), run_label=run_label)
    duplicates = {
        seed: [str(path) for path in paths]
        for seed, paths in candidates.items()
        if len(paths) > 1
    }
    if duplicates:
        raise ValueError(
            "multiple independent source attempts found; refusing favourable "
            f"selection: {duplicates}"
        )
    attempts: list[AttemptReceipt] = []
    rows: list[dict[str, Any]] = []
    for seed in sorted(candidates):
        attempt_receipt, attempt_rows = _validate_attempt(
            candidates[seed][0],
            seed=seed,
            registered_config=registered_config,
            expected_provenance=provenance,
        )
        attempts.append(attempt_receipt)
        rows.extend(attempt_rows)
    cells = exp26._manifest(registered_config)
    cell_order = {cell.generator_id: index for index, cell in enumerate(cells)}
    mode_order = {mode: index for index, mode in enumerate(EXPECTED_MODES)}
    rows.sort(
        key=lambda row: (
            int(row["seed"]),
            cell_order[str(row["generator_id"])],
            mode_order[str(row["actuator_mode"])],
        )
    )
    for index, row in enumerate(rows):
        row["source_panel_row_index"] = index
    return PanelCollection(
        rows=tuple(rows),
        attempts=tuple(attempts),
        config=registered_config,
        config_sha256=contract.independent_config_sha256,
        config_file_sha256=contract.independent_config_file_sha256,
        source_contract=dict(registered_config["source_binding"]),
        provenance_identity=provenance if attempts else None,
    )


def panel_coverage(collection: PanelCollection) -> dict[str, Any]:
    expected_keys = {
        (seed, cell.generator_id, mode)
        for seed in EXPECTED_SEEDS
        for cell in exp26._manifest(collection.config)
        for mode in EXPECTED_MODES
    }
    observed_keys = [
        (int(row["seed"]), str(row["generator_id"]), str(row["actuator_mode"]))
        for row in collection.rows
    ]
    status_counts = {status: 0 for status in ROW_STATUSES}
    for row in collection.rows:
        status_counts[str(row["status"])] += 1
    observed_seeds = tuple(attempt.seed for attempt in collection.attempts)
    row_count_complete = len(collection.rows) == EXPECTED_PANEL_ROWS
    cartesian_complete = bool(
        row_count_complete
        and len(observed_keys) == len(set(observed_keys))
        and set(observed_keys) == expected_keys
    )
    terminal_complete = bool(
        observed_seeds == EXPECTED_SEEDS
        and all(attempt.run_status in TERMINAL_STATUSES for attempt in collection.attempts)
    )
    planned_complete = bool(
        len(collection.attempts) == len(EXPECTED_SEEDS)
        and all(
            attempt.planned_coverage_valid
            and attempt.observed_row_count == EXPECTED_ROWS_PER_SEED
            for attempt in collection.attempts
        )
    )
    all_budget_valid = bool(
        collection.rows
        and all(_strict_true(row.get("functional_budget_valid")) for row in collection.rows)
    )
    all_stable = bool(
        collection.rows
        and all(
            _strict_true(row.get("effective_dynamics_strictly_stable"))
            for row in collection.rows
        )
    )
    no_failed_rows = status_counts["failed"] == 0 and status_counts["invalid"] == 0
    source_panel_valid = bool(
        cartesian_complete
        and terminal_complete
        and planned_complete
        and all_budget_valid
        and all_stable
        and no_failed_rows
    )
    return {
        "expected_seeds": list(EXPECTED_SEEDS),
        "observed_seeds": list(observed_seeds),
        "missing_seeds": sorted(set(EXPECTED_SEEDS) - set(observed_seeds)),
        "expected_seed_count": len(EXPECTED_SEEDS),
        "observed_seed_count": len(observed_seeds),
        "expected_generators_per_seed": 88,
        "expected_modes_per_generator": len(EXPECTED_MODES),
        "expected_rows_per_seed": EXPECTED_ROWS_PER_SEED,
        "expected_row_count": EXPECTED_PANEL_ROWS,
        "observed_row_count": len(collection.rows),
        "row_count_complete": row_count_complete,
        "cartesian_complete": cartesian_complete,
        "terminal_attempts_complete": terminal_complete,
        "planned_coverage_complete": planned_complete,
        "all_functional_budgets_valid": all_budget_valid,
        "all_effective_dynamics_stable": all_stable,
        "row_status_counts": status_counts,
        "all_failures_retained": True,
        "source_panel_valid": source_panel_valid,
    }


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for row in rows
    ).encode("utf-8")


_COMPLETE_SOURCE_REQUIRED_FIELDS = {
    "run_id",
    "experiment",
    "seed",
    "generator_id",
    "generator_split",
    "alpha",
    "transition_rank",
    "input_rank",
    "delay",
    "noise_std",
    "rotation_seed",
    "actuator_mode",
    "condition",
    "manifest_hash",
    "profile",
    "status",
    "chi",
    "state_demand",
    "input_demand",
    "validation_balanced_accuracy",
    "test_balanced_accuracy",
    "functional_budget_valid",
    "effective_dynamics_strictly_stable",
    "source_panel_row_index",
    "_run_status",
    "_effective_status",
    *_ROW_EVIDENCE_FIELDS,
}


def _validate_package_attempts(
    receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[AttemptReceipt, ...]:
    value = receipt.get("attempts")
    if not isinstance(value, list):
        raise ValueError("independent source attempts receipt must be a list")
    attempts: list[AttemptReceipt] = []
    seen_seeds: set[int] = set()
    seen_run_ids: set[str] = set()
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ValueError("independent source attempt receipt is malformed")
        try:
            seed = int(entry["seed"])
            observed_row_count = int(entry["observed_row_count"])
            observed_complete_rows = int(entry["observed_complete_rows"])
            observed_failed_rows = int(entry["observed_failed_rows"])
            observed_invalid_rows = int(entry["observed_invalid_rows"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("independent source attempt counts are malformed") from error
        run_id = entry.get("run_id")
        path = entry.get("path")
        run_status = entry.get("run_status")
        planned_valid = entry.get("planned_coverage_valid")
        hashes = entry.get("file_sha256")
        if (
            seed not in EXPECTED_SEEDS
            or seed in seen_seeds
            or not isinstance(run_id, str)
            or not run_id
            or run_id in seen_run_ids
            or not isinstance(path, str)
            or not path
            or not isinstance(run_status, str)
            or not isinstance(planned_valid, bool)
            or not isinstance(hashes, Mapping)
            or set(hashes) != set(ATTEMPT_FILES)
            or any(
                not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for digest in hashes.values()
            )
            or min(
                observed_row_count,
                observed_complete_rows,
                observed_failed_rows,
                observed_invalid_rows,
            )
            < 0
            or observed_row_count
            != observed_complete_rows + observed_failed_rows + observed_invalid_rows
        ):
            raise ValueError("independent source attempt receipt is invalid")
        seen_seeds.add(seed)
        seen_run_ids.add(run_id)
        seed_rows = [row for row in rows if row.get("seed") == seed]
        counts = {status: 0 for status in ROW_STATUSES}
        for row in seed_rows:
            status = str(row.get("status"))
            if status not in counts:
                raise ValueError("independent source row status is invalid")
            counts[status] += 1
            expected_effective = (
                status if run_status in TERMINAL_STATUSES else "failed"
            )
            if (
                row.get("run_id") != run_id
                or row.get("_run_status") != run_status
                or row.get("_effective_status") != expected_effective
            ):
                raise ValueError("independent source row/attempt identity disagrees")
        if (
            len(seed_rows) != observed_row_count
            or counts["complete"] != observed_complete_rows
            or counts["failed"] != observed_failed_rows
            or counts["invalid"] != observed_invalid_rows
        ):
            raise ValueError("independent source attempt counts disagree with raw rows")
        attempts.append(
            AttemptReceipt(
                seed=seed,
                path=path,
                run_status=run_status,
                run_id=run_id,
                planned_coverage_valid=planned_valid,
                observed_row_count=observed_row_count,
                observed_complete_rows=observed_complete_rows,
                observed_failed_rows=observed_failed_rows,
                observed_invalid_rows=observed_invalid_rows,
                file_sha256=dict(hashes),
            )
        )
    row_seeds = {int(row["seed"]) for row in rows}
    if row_seeds - seen_seeds:
        raise ValueError("independent source rows lack an attempt receipt")
    return tuple(sorted(attempts, key=lambda attempt: attempt.seed))


def _validate_raw_panel_schema(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    cells = exp26._manifest(config)
    if (
        len(cells) != 88
        or len({cell.generator_id for cell in cells}) != 88
        or {cell.generator_split for cell in cells} != {"discovery", "heldout"}
    ):
        raise ValueError("registered independent generator schema is invalid")
    receipt = manifest_hash(cells)
    expected_evidence = evidence_row_fields(provenance)
    expected_dimensions = {
        (seed, cell.generator_id, mode): _dimensions(
            cell,
            mode=mode,
            manifest_receipt=receipt,
            evidence=expected_evidence,
        )
        for seed in EXPECTED_SEEDS
        for cell in cells
        for mode in EXPECTED_MODES
    }
    seen: set[tuple[int, str, str]] = set()
    for index, row in enumerate(rows):
        try:
            seed = int(row["seed"])
            key = (seed, str(row["generator_id"]), str(row["actuator_mode"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("independent source raw cell identity is malformed") from error
        dimensions = expected_dimensions.get(key)
        if (
            key in seen
            or dimensions is None
            or any(row.get(name) != value for name, value in dimensions.items())
            or row.get("experiment") != EXPERIMENT
            or row.get("profile") != PROFILE
            or row.get("source_panel_row_index") != index
            or row.get("status") not in ROW_STATUSES
        ):
            raise ValueError("independent source raw Cartesian schema is invalid")
        seen.add(key)
        if row.get("status") == "complete":
            missing = _COMPLETE_SOURCE_REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(
                    "complete independent source row lacks fields: "
                    f"{sorted(missing)}"
                )
            numeric_fields = (
                "alpha",
                "transition_rank",
                "input_rank",
                "delay",
                "noise_std",
                "chi",
                "state_demand",
                "input_demand",
                "validation_balanced_accuracy",
                "test_balanced_accuracy",
            )
            try:
                numeric = np.asarray([float(row[name]) for name in numeric_fields])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "complete independent source row has non-numeric fields"
                ) from error
            if (
                not np.all(np.isfinite(numeric))
                or not _strict_true(row.get("functional_budget_valid"))
                or not _strict_true(row.get("effective_dynamics_strictly_stable"))
            ):
                raise ValueError(
                    "complete independent source row violates budget/stability schema"
                )


def _recompute_package_coverage(
    rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[AttemptReceipt],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    observed_seeds = tuple(attempt.seed for attempt in attempts)
    observed_keys = [
        (int(row["seed"]), str(row["generator_id"]), str(row["actuator_mode"]))
        for row in rows
    ]
    expected_keys = {
        (seed, cell.generator_id, mode)
        for seed in EXPECTED_SEEDS
        for cell in exp26._manifest(config)
        for mode in EXPECTED_MODES
    }
    status_counts = {status: 0 for status in ROW_STATUSES}
    for row in rows:
        status_counts[str(row["status"])] += 1
    row_count_complete = len(rows) == EXPECTED_PANEL_ROWS
    cartesian_complete = bool(
        row_count_complete
        and len(observed_keys) == len(set(observed_keys))
        and set(observed_keys) == expected_keys
    )
    terminal_complete = bool(
        observed_seeds == EXPECTED_SEEDS
        and all(attempt.run_status in TERMINAL_STATUSES for attempt in attempts)
    )
    planned_complete = bool(
        len(attempts) == len(EXPECTED_SEEDS)
        and all(
            attempt.planned_coverage_valid
            and attempt.observed_row_count == EXPECTED_ROWS_PER_SEED
            for attempt in attempts
        )
    )
    all_budget_valid = bool(
        rows and all(_strict_true(row.get("functional_budget_valid")) for row in rows)
    )
    all_stable = bool(
        rows
        and all(
            _strict_true(row.get("effective_dynamics_strictly_stable")) for row in rows
        )
    )
    source_panel_valid = bool(
        cartesian_complete
        and terminal_complete
        and planned_complete
        and all_budget_valid
        and all_stable
        and status_counts["failed"] == 0
        and status_counts["invalid"] == 0
    )
    return {
        "expected_seeds": list(EXPECTED_SEEDS),
        "observed_seeds": list(observed_seeds),
        "missing_seeds": sorted(set(EXPECTED_SEEDS) - set(observed_seeds)),
        "expected_seed_count": len(EXPECTED_SEEDS),
        "observed_seed_count": len(observed_seeds),
        "expected_generators_per_seed": 88,
        "expected_modes_per_generator": len(EXPECTED_MODES),
        "expected_rows_per_seed": EXPECTED_ROWS_PER_SEED,
        "expected_row_count": EXPECTED_PANEL_ROWS,
        "observed_row_count": len(rows),
        "row_count_complete": row_count_complete,
        "cartesian_complete": cartesian_complete,
        "terminal_attempts_complete": terminal_complete,
        "planned_coverage_complete": planned_complete,
        "all_functional_budgets_valid": all_budget_valid,
        "all_effective_dynamics_stable": all_stable,
        "row_status_counts": status_counts,
        "all_failures_retained": True,
        "source_panel_valid": source_panel_valid,
    }


def write_source_panel_package(
    collection: PanelCollection,
    output_dir: str | Path,
) -> Path:
    """Write an immutable raw package; never make a scientific claim."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite source panel package {output}")
    output.mkdir(parents=True)
    raw_payload = _canonical_jsonl(collection.rows)
    raw_path = output / "raw_metrics.jsonl"
    raw_path.write_bytes(raw_payload)
    raw_sha = hashlib.sha256(raw_payload).hexdigest()
    coverage = panel_coverage(collection)
    receipt_payload: dict[str, Any] = {
        "schema_version": SOURCE_PACKAGE_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "profile": PROFILE,
        "evidence_role": "independent_test_source_only",
        "conclusion": "inconclusive",
        "standalone_inference_performed": False,
        "standalone_inference_permitted": False,
        "reason": (
            "source-only independent panel; downstream inference is separately "
            "preregistered"
        ),
        "registered_config_sha256": collection.config_sha256,
        "registered_config_file_sha256": collection.config_file_sha256,
        "source_contract": dict(collection.source_contract),
        "source_contract_sha256": _canonical_sha256(collection.source_contract),
        "run_provenance": collection.provenance_identity,
        "coverage": coverage,
        "raw_metrics_file": raw_path.name,
        "raw_metrics_sha256": raw_sha,
        "raw_metrics_row_count": len(collection.rows),
        "attempts": [asdict(attempt) for attempt in collection.attempts],
    }
    receipt_payload_sha = _canonical_sha256(receipt_payload)
    receipt = {
        **receipt_payload,
        "receipt_payload_sha256": receipt_payload_sha,
    }
    (output / "source_panel_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    conclusion = {
        "profile": PROFILE,
        "conclusion": "inconclusive",
        "evidence_role": "independent_test_source_only",
        "standalone_inference_performed": False,
        "source_panel_valid": coverage["source_panel_valid"],
        "coverage": coverage,
        "source_panel_receipt_payload_sha256": receipt_payload_sha,
        "raw_metrics_sha256": raw_sha,
        "reason": receipt_payload["reason"],
    }
    (output / "conclusion.json").write_text(
        json.dumps(conclusion, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Exp28 independent Exp26 source panel",
        "",
        "This artifact is source-only and has conclusion **inconclusive**. No ",
        "selector or standalone hypothesis test was fit while packaging it.",
        "",
        f"- Expected rows: {EXPECTED_PANEL_ROWS}",
        f"- Observed rows: {coverage['observed_row_count']}",
        f"- Complete Cartesian panel: {coverage['cartesian_complete']}",
        f"- All functional budgets valid: {coverage['all_functional_budgets_valid']}",
        f"- All effective dynamics stable: {coverage['all_effective_dynamics_stable']}",
        f"- Source panel valid: {coverage['source_panel_valid']}",
        f"- Raw SHA-256: `{raw_sha}`",
        "",
        "All observed failed and invalid rows are retained in `raw_metrics.jsonl`.",
    ]
    (output / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return output


def load_source_panel_package(
    package_dir: str | Path,
    *,
    require_complete: bool = True,
) -> SourcePanelPackage:
    """Rebuild every package invariant without trusting its self-declarations."""

    directory = Path(package_dir)
    receipt_path = directory / "source_panel_receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt_file_sha = hashlib.sha256(receipt_bytes).hexdigest()
    receipt = _read_json(receipt_path)
    conclusion = _read_json(directory / "conclusion.json")
    if not isinstance(receipt, Mapping) or not isinstance(conclusion, Mapping):
        raise ValueError("independent source package metadata is malformed")
    recorded_receipt_sha = receipt.get("receipt_payload_sha256")
    payload = {
        key: value for key, value in receipt.items() if key != "receipt_payload_sha256"
    }
    if (
        receipt.get("schema_version") != SOURCE_PACKAGE_SCHEMA_VERSION
        or receipt.get("profile") != PROFILE
        or receipt.get("conclusion") != "inconclusive"
        or receipt.get("evidence_role") != "independent_test_source_only"
        or receipt.get("standalone_inference_performed") is not False
        or receipt.get("standalone_inference_permitted") is not False
        or recorded_receipt_sha != _canonical_sha256(payload)
        or conclusion.get("profile") != PROFILE
        or conclusion.get("conclusion") != "inconclusive"
        or conclusion.get("source_panel_receipt_payload_sha256")
        != recorded_receipt_sha
        or conclusion.get("raw_metrics_sha256") != receipt.get("raw_metrics_sha256")
        or conclusion.get("coverage") != receipt.get("coverage")
        or conclusion.get("source_panel_valid")
        != receipt.get("coverage", {}).get("source_panel_valid")
    ):
        raise ValueError("independent source package receipt is invalid")
    raw_name = receipt.get("raw_metrics_file")
    if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
        raise ValueError("independent source raw path is invalid")
    raw_path = directory / raw_name
    raw_bytes = raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != receipt.get("raw_metrics_sha256"):
        raise ValueError("independent source raw metrics hash is invalid")
    rows = _read_jsonl(raw_path)
    if raw_bytes != _canonical_jsonl(rows):
        raise ValueError("independent source raw metrics are not canonical JSONL")
    if len(rows) != receipt.get("raw_metrics_row_count"):
        raise ValueError("independent source raw row count is invalid")

    registered_config = load_json_config(DEFAULT_CONFIG)
    run_provenance = receipt.get("run_provenance")
    if not isinstance(run_provenance, Mapping):
        raise ValueError("independent source package lacks run provenance")
    run_git = run_provenance.get("run_git")
    runtime_versions = run_provenance.get("runtime_versions")
    if not isinstance(run_git, Mapping) or not isinstance(runtime_versions, Mapping):
        raise ValueError("independent source package run provenance is malformed")
    contract = validate_source_contract(
        registered_config,
        current_git=run_git,
        runtime_versions=runtime_versions,
    )
    expected_provenance = build_evidence_provenance(
        contract, run_label=REQUIRED_RUN_LABEL
    )
    source_contract = receipt.get("source_contract")
    if (
        receipt.get("registered_config_sha256")
        != contract.independent_config_sha256
        or receipt.get("registered_config_file_sha256")
        != contract.independent_config_file_sha256
        or not isinstance(source_contract, Mapping)
        or dict(source_contract) != dict(registered_config["source_binding"])
        or receipt.get("source_contract_sha256")
        != contract.source_contract_sha256
        or dict(run_provenance) != expected_provenance
    ):
        raise ValueError("independent source known config/contract binding is invalid")

    _validate_raw_panel_schema(
        rows,
        config=registered_config,
        provenance=run_provenance,
    )
    attempts = _validate_package_attempts(receipt, rows)
    recomputed_coverage = _recompute_package_coverage(
        rows,
        attempts,
        config=registered_config,
    )
    if (
        receipt.get("coverage") != recomputed_coverage
        or conclusion.get("coverage") != recomputed_coverage
        or conclusion.get("source_panel_valid")
        != recomputed_coverage["source_panel_valid"]
    ):
        raise ValueError("independent source declared coverage is not reproducible")
    if require_complete and recomputed_coverage["source_panel_valid"] is not True:
        raise ValueError("independent source panel is incomplete or scientifically invalid")
    return SourcePanelPackage(
        receipt=dict(receipt),
        rows=tuple(rows),
        receipt_payload_sha256=str(recorded_receipt_sha),
        receipt_file_sha256=receipt_file_sha,
        raw_metrics_sha256=str(receipt["raw_metrics_sha256"]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Package the source-only Exp28 independent Exp26 panel"
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-label", default=REQUIRED_RUN_LABEL)
    args = parser.parse_args(argv)
    collection = collect_source_panel(
        args.results_root,
        config_path=args.config,
        run_label=args.run_label,
    )
    output = write_source_panel_package(collection, args.output_dir)
    print(output)


if __name__ == "__main__":
    main()
