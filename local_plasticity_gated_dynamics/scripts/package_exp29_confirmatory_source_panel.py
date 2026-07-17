"""Collect, validate, and hash-bind the Exp29 confirmatory source panel."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import exp26_actuator_phase_diagram as exp26  # noqa: E402
from experiments.common import load_json_config  # noqa: E402
from experiments.exp29_confirmatory_source_panel import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    EXPECTED_GENERATORS,
    EXPECTED_MODES,
    EXPECTED_PANEL_ROWS,
    EXPECTED_ROWS_PER_SEED,
    EXPECTED_SEEDS,
    EXPERIMENT,
    PROFILE,
    PROTOCOL_VERSION,
    REQUIRED_RUN_LABEL,
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


SOURCE_PACKAGE_SCHEMA_VERSION = "exp29_confirmatory_source_package_v1"
STATISTICS_UNIT = "seed"
TERMINAL_RUN_STATUSES = {"complete", "complete_with_failures"}
ROW_STATUSES = {"complete", "infeasible", "failed", "invalid"}
ACTIVE_MODES = tuple(mode for mode in EXPECTED_MODES if mode != "frozen")
INFEASIBLE_REASONS = {
    "budget_scale_above_cap",
    "degenerate_actuator",
    "budget_mismatch",
    "effective_instability",
}
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "formal" / "exp29_confirmatory_source_panel.json"
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
    row_status_counts: Mapping[str, int]
    file_sha256: Mapping[str, str]


@dataclass(frozen=True)
class PanelCollection:
    rows: tuple[Mapping[str, Any], ...]
    attempts: tuple[AttemptReceipt, ...]
    config: Mapping[str, Any]
    config_sha256: str
    config_file_sha256: str
    source_contract: Mapping[str, Any]
    provenance_identity: Mapping[str, Any] | None


@dataclass(frozen=True)
class SourcePanelPackage:
    receipt: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    receipt_payload_sha256: str
    receipt_file_sha256: str
    conclusion_file_sha256: str
    raw_metrics_sha256: str


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read confirmatory artifact {path}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read confirmatory rows {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank confirmatory row at {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid confirmatory row at {path}:{line_number}"
            ) from error
        if not isinstance(value, Mapping):
            raise ValueError("confirmatory metric row must be an object")
        rows.append(dict(value))
    return rows


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for row in rows
    ).encode("utf-8")


def _strict_bool(value: object, expected: bool) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value) is expected


def _same_number(first: object, second: object) -> bool:
    try:
        left = float(first)
        right = float(second)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(left) and np.isfinite(right) and left == right)


def _attempt_candidates(results_root: Path, *, run_label: str) -> dict[int, list[Path]]:
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
            raise ValueError("confirmatory attempt config is malformed")
        if value.get("experiment") != EXPERIMENT:
            raise ValueError("wrong experiment identity in confirmatory results root")
        if value.get("run_label") != run_label:
            continue
        seed = value.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("confirmatory attempt seed is malformed")
        if seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected confirmatory seed {seed}")
        if attempt.parent.name != f"seed_{seed:04d}":
            raise ValueError("confirmatory attempt path and seed disagree")
        candidates.setdefault(seed, []).append(attempt)
    return candidates


def _expected_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"condition_index": index, **condition}
        for index, condition in enumerate(planned_conditions(config))
    ]


def _environment_matches(
    environment: Mapping[str, Any], provenance: Mapping[str, Any]
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
        and str(environment.get("python", "")).startswith(f"{versions.get('python')} ")
        and dict(environment_git) == dict(run_git)
    )


def _validate_attempt(
    attempt: Path,
    *,
    seed: int,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[AttemptReceipt, list[dict[str, Any]]]:
    missing = [name for name in ATTEMPT_FILES if not (attempt / name).is_file()]
    if missing:
        raise ValueError(f"confirmatory attempt lacks files {missing}: {attempt}")
    attempt_config = _read_json(attempt / "config.json")
    status = _read_json(attempt / "status.json")
    manifest = _read_json(attempt / "manifest.json")
    environment = _read_json(attempt / "environment.json")
    if not all(
        isinstance(value, Mapping)
        for value in (attempt_config, status, manifest, environment)
    ):
        raise ValueError("confirmatory attempt metadata is malformed")
    if canonical_config_payload(attempt_config) != canonical_config_payload(config):
        raise ValueError("confirmatory attempt differs from registered config")
    run_status = str(status.get("status", "missing"))
    run_id = manifest.get("run_id")
    if (
        attempt_config.get("experiment") != EXPERIMENT
        or attempt_config.get("seed") != seed
        or attempt_config.get("run_label") != REQUIRED_RUN_LABEL
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
        raise ValueError("confirmatory attempt identity is inconsistent")
    attempt_provenance = attempt_config.get("evidence_provenance")
    if (
        not isinstance(attempt_provenance, Mapping)
        or dict(attempt_provenance) != dict(provenance)
        or manifest.get("evidence_provenance") != attempt_provenance
        or not _environment_matches(environment, provenance)
    ):
        raise ValueError("confirmatory attempt provenance is inconsistent")

    cells = exp26._manifest(config)
    manifest_receipt = manifest_hash(cells)
    evidence = evidence_row_fields(provenance)
    expected_dimensions = {
        (cell.generator_id, mode): _dimensions(
            cell,
            mode=mode,
            manifest_receipt=manifest_receipt,
            evidence=evidence,
        )
        for cell in cells
        for mode in EXPECTED_MODES
    }
    raw_rows = _read_jsonl(attempt / "metrics.jsonl")
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    counts = {status_name: 0 for status_name in ROW_STATUSES}
    for row in raw_rows:
        key = (str(row.get("generator_id")), str(row.get("actuator_mode")))
        dimensions = expected_dimensions.get(key)
        row_status = str(row.get("status"))
        if (
            row.get("experiment") != EXPERIMENT
            or row.get("seed") != seed
            or row.get("run_id") != run_id
            or row_status not in ROW_STATUSES
            or row.get("statistics_unit") != STATISTICS_UNIT
            or key in seen
            or dimensions is None
            or any(row.get(name) != value for name, value in dimensions.items())
        ):
            raise ValueError(f"unexpected/tampered confirmatory row: {attempt}")
        seen.add(key)
        counts[row_status] += 1
        rows.append(
            {
                **row,
                "profile": PROFILE,
                "_attempt_path": str(attempt.resolve()),
                "_run_status": run_status,
                "_effective_status": (
                    row_status if run_status in TERMINAL_RUN_STATUSES else "failed"
                ),
            }
        )
    expected_run_status = (
        "complete_with_failures"
        if counts["failed"] or counts["invalid"]
        else "complete"
    )
    if run_status != expected_run_status:
        raise ValueError("confirmatory run status disagrees with terminal rows")
    plan_valid = _read_json(attempt / "planned_conditions.json") == _expected_plan(
        config
    )
    files = {name: _file_sha256(attempt / name) for name in ATTEMPT_FILES}
    return (
        AttemptReceipt(
            seed=seed,
            path=str(attempt.resolve()),
            run_status=run_status,
            run_id=run_id,
            planned_coverage_valid=plan_valid,
            observed_row_count=len(raw_rows),
            row_status_counts=counts,
            file_sha256=files,
        ),
        rows,
    )


def collect_source_panel(
    results_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    run_label: str = REQUIRED_RUN_LABEL,
    current_git: Mapping[str, object] | None = None,
    runtime_versions: Mapping[str, object] | None = None,
) -> PanelCollection:
    config = load_json_config(config_path)
    contract = validate_source_contract(
        config,
        current_git=current_git,
        runtime_versions=runtime_versions,
    )
    if run_label != REQUIRED_RUN_LABEL:
        raise ValueError(f"run_label must equal {REQUIRED_RUN_LABEL}")
    provenance = build_evidence_provenance(contract, run_label=run_label)
    candidates = _attempt_candidates(Path(results_root), run_label=run_label)
    duplicates = {
        seed: [str(path) for path in paths]
        for seed, paths in candidates.items()
        if len(paths) > 1
    }
    if duplicates:
        raise ValueError(
            "multiple confirmatory attempts found; selective rerun is forbidden: "
            f"{duplicates}"
        )
    attempts: list[AttemptReceipt] = []
    rows: list[dict[str, Any]] = []
    for seed in sorted(candidates):
        receipt, attempt_rows = _validate_attempt(
            candidates[seed][0],
            seed=seed,
            config=config,
            provenance=provenance,
        )
        attempts.append(receipt)
        rows.extend(attempt_rows)
    cells = exp26._manifest(config)
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
        config=config,
        config_sha256=contract.config_sha256,
        config_file_sha256=contract.config_file_sha256,
        source_contract=dict(config["source_binding"]),
        provenance_identity=provenance if attempts else None,
    )


def _validate_cell_semantics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if any(row.get("statistics_unit") != STATISTICS_UNIT for row in rows):
        raise ValueError("confirmatory statistics unit must be seed")
    grouped: dict[tuple[int, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["generator_id"]))
        grouped.setdefault(key, {})[str(row["actuator_mode"])] = row
    infeasible_counts = {
        seed: {mode: 0 for mode in ACTIVE_MODES} for seed in EXPECTED_SEEDS
    }
    feasible_active_rows = 0
    for key, modes in grouped.items():
        if set(modes) != set(EXPECTED_MODES):
            raise ValueError(f"confirmatory cell lacks registered modes: {key}")
        frozen = modes["frozen"]
        if (
            frozen.get("status") != "complete"
            or not _strict_bool(frozen.get("actuator_feasible"), True)
            or not _strict_bool(frozen.get("deployment_available"), True)
            or not _strict_bool(frozen.get("deployment_fallback_applied"), False)
            or frozen.get("deployment_mode") != "frozen"
            or not _strict_bool(frozen.get("matched_budget_support_eligible"), False)
            or not _strict_bool(frozen.get("unconditional_cell_retained"), True)
            or any(
                not np.isfinite(float(frozen.get(name, np.nan)))
                or not 0.0 <= float(frozen.get(name, np.nan)) <= 1.0
                for name in (
                    "validation_balanced_accuracy",
                    "test_balanced_accuracy",
                )
            )
        ):
            raise ValueError("registered frozen fallback is not deployable")
        frozen_validation = frozen.get("validation_balanced_accuracy")
        frozen_test = frozen.get("test_balanced_accuracy")
        for mode in ACTIVE_MODES:
            row = modes[mode]
            status = row.get("status")
            if (
                any(
                    not _same_number(row.get(name), frozen.get(name))
                    for name in ("chi", "state_demand", "input_demand")
                )
                or any(
                    row.get(name) != frozen.get(name)
                    for name in (
                        "dataset_fingerprint",
                        "train_split_fingerprint",
                        "validation_split_fingerprint",
                        "test_split_fingerprint",
                    )
                )
                or not _strict_bool(row.get("deployment_available"), True)
                or not _strict_bool(row.get("unconditional_cell_retained"), True)
                or any(
                    not np.isfinite(float(row.get(name, np.nan)))
                    or not 0.0 <= float(row.get(name, np.nan)) <= 1.0
                    for name in (
                        "validation_balanced_accuracy",
                        "test_balanced_accuracy",
                    )
                )
            ):
                raise ValueError("confirmatory actuator family is not paired")
            if status == "complete":
                if (
                    not _strict_bool(row.get("actuator_feasible"), True)
                    or not _strict_bool(row.get("functional_budget_valid"), True)
                    or not _strict_bool(
                        row.get("effective_dynamics_strictly_stable"), True
                    )
                    or not _strict_bool(
                        row.get("matched_budget_support_eligible"), True
                    )
                    or not _strict_bool(row.get("deployment_fallback_applied"), False)
                    or row.get("deployment_mode") != mode
                ):
                    raise ValueError(
                        "feasible active row violates matched-budget policy"
                    )
                feasible_active_rows += 1
            elif status == "infeasible":
                if (
                    row.get("infeasible_reason") not in INFEASIBLE_REASONS
                    or not _strict_bool(row.get("actuator_feasible"), False)
                    or not _strict_bool(
                        row.get("matched_budget_support_eligible"), False
                    )
                    or not _strict_bool(row.get("deployment_fallback_applied"), True)
                    or row.get("deployment_mode") != "frozen"
                    or not _same_number(
                        row.get("validation_balanced_accuracy"), frozen_validation
                    )
                    or not _same_number(row.get("test_balanced_accuracy"), frozen_test)
                    or not _same_number(
                        row.get("deployment_validation_balanced_accuracy"),
                        frozen_validation,
                    )
                    or not _same_number(
                        row.get("deployment_test_balanced_accuracy"), frozen_test
                    )
                    or not _same_number(
                        row.get("fallback_frozen_validation_balanced_accuracy"),
                        frozen_validation,
                    )
                    or not _same_number(
                        row.get("fallback_frozen_test_balanced_accuracy"), frozen_test
                    )
                    or row.get("fallback_frozen_correction_fingerprint")
                    != frozen.get("correction_fingerprint")
                ):
                    raise ValueError(
                        "infeasible row does not use exact frozen fallback"
                    )
                budget_valid = row.get("functional_budget_valid")
                if row.get("infeasible_reason") == "effective_instability":
                    if not _strict_bool(budget_valid, True):
                        raise ValueError(
                            "instability-only infeasibility must preserve valid budget"
                        )
                elif not _strict_bool(budget_valid, False):
                    raise ValueError(
                        "fit/budget infeasibility cannot claim a matched budget"
                    )
                infeasible_counts[key[0]][mode] += 1
            elif status not in {"failed", "invalid"}:
                raise ValueError("confirmatory row has unknown terminal status")
    rates = {
        str(seed): {
            mode: infeasible_counts[seed][mode] / EXPECTED_GENERATORS
            for mode in ACTIVE_MODES
        }
        for seed in EXPECTED_SEEDS
    }
    family_rates = {
        mode: sum(infeasible_counts[seed][mode] for seed in EXPECTED_SEEDS)
        / (len(EXPECTED_SEEDS) * EXPECTED_GENERATORS)
        for mode in ACTIVE_MODES
    }
    return {
        "feasible_active_row_count": feasible_active_rows,
        "matched_budget_support_row_count": feasible_active_rows,
        "infeasible_rate_by_seed_family": rates,
        "infeasible_rate_by_family": family_rates,
    }


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
    counts = {status: 0 for status in ROW_STATUSES}
    for row in collection.rows:
        counts[str(row["status"])] += 1
    observed_seeds = tuple(attempt.seed for attempt in collection.attempts)
    cartesian = bool(
        len(collection.rows) == EXPECTED_PANEL_ROWS
        and len(observed_keys) == len(set(observed_keys))
        and set(observed_keys) == expected_keys
    )
    attempts_complete = bool(
        observed_seeds == EXPECTED_SEEDS
        and all(
            attempt.run_status in TERMINAL_RUN_STATUSES
            and attempt.planned_coverage_valid
            and attempt.observed_row_count == EXPECTED_ROWS_PER_SEED
            for attempt in collection.attempts
        )
    )
    semantics: dict[str, Any] = {
        "feasible_active_row_count": 0,
        "matched_budget_support_row_count": 0,
        "infeasible_rate_by_seed_family": {},
        "infeasible_rate_by_family": {},
    }
    semantics_valid = False
    if cartesian:
        try:
            semantics = _validate_cell_semantics(collection.rows)
            semantics_valid = True
        except ValueError:
            semantics_valid = False
    valid = bool(
        cartesian
        and attempts_complete
        and semantics_valid
        and counts["failed"] == 0
        and counts["invalid"] == 0
    )
    return {
        "expected_seeds": list(EXPECTED_SEEDS),
        "observed_seeds": list(observed_seeds),
        "missing_seeds": sorted(set(EXPECTED_SEEDS) - set(observed_seeds)),
        "expected_seed_count": len(EXPECTED_SEEDS),
        "observed_seed_count": len(observed_seeds),
        "expected_generators_per_seed": EXPECTED_GENERATORS,
        "expected_modes_per_generator": len(EXPECTED_MODES),
        "expected_rows_per_seed": EXPECTED_ROWS_PER_SEED,
        "expected_row_count": EXPECTED_PANEL_ROWS,
        "observed_row_count": len(collection.rows),
        "cartesian_complete": cartesian,
        "terminal_attempts_complete": attempts_complete,
        "feasibility_semantics_valid": semantics_valid,
        "row_status_counts": counts,
        "all_registered_rows_retained": True,
        "selective_rerun_permitted": False,
        "statistics_unit": STATISTICS_UNIT,
        **semantics,
        "source_panel_valid": valid,
    }


def _validate_raw_schema(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    cells = exp26._manifest(config)
    receipt = manifest_hash(cells)
    evidence = evidence_row_fields(provenance)
    expected = {
        (seed, cell.generator_id, mode): _dimensions(
            cell,
            mode=mode,
            manifest_receipt=receipt,
            evidence=evidence,
        )
        for seed in EXPECTED_SEEDS
        for cell in cells
        for mode in EXPECTED_MODES
    }
    seen: set[tuple[int, str, str]] = set()
    for index, row in enumerate(rows):
        if row.get("statistics_unit") != STATISTICS_UNIT:
            raise ValueError("confirmatory statistics unit must be seed")
        key = (
            int(row.get("seed", -1)),
            str(row.get("generator_id")),
            str(row.get("actuator_mode")),
        )
        dimensions = expected.get(key)
        if (
            key in seen
            or dimensions is None
            or any(row.get(name) != value for name, value in dimensions.items())
            or row.get("experiment") != EXPERIMENT
            or row.get("profile") != PROFILE
            or row.get("source_panel_row_index") != index
            or row.get("status") not in ROW_STATUSES
        ):
            raise ValueError("confirmatory raw Cartesian schema is invalid")
        seen.add(key)


def write_source_panel_package(
    collection: PanelCollection, output_dir: str | Path
) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite confirmatory package {output}")
    output.mkdir(parents=True)
    raw_payload = _canonical_jsonl(collection.rows)
    raw_path = output / "raw_metrics.jsonl"
    raw_path.write_bytes(raw_payload)
    raw_sha = hashlib.sha256(raw_payload).hexdigest()
    coverage = panel_coverage(collection)
    receipt_payload = {
        "schema_version": SOURCE_PACKAGE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "profile": PROFILE,
        "statistics_unit": STATISTICS_UNIT,
        "evidence_role": "confirmatory_test_source_only",
        "conclusion": "inconclusive",
        "standalone_inference_performed": False,
        "standalone_inference_permitted": False,
        "reason": "source-only panel; inference is separately preregistered",
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
    receipt_sha = _canonical_sha256(receipt_payload)
    receipt = {**receipt_payload, "receipt_payload_sha256": receipt_sha}
    receipt_path = output / "source_panel_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    conclusion = {
        "profile": PROFILE,
        "statistics_unit": STATISTICS_UNIT,
        "conclusion": "inconclusive",
        "evidence_role": "confirmatory_test_source_only",
        "standalone_inference_performed": False,
        "source_panel_valid": coverage["source_panel_valid"],
        "coverage": coverage,
        "source_panel_receipt_payload_sha256": receipt_sha,
        "raw_metrics_sha256": raw_sha,
        "reason": receipt_payload["reason"],
    }
    (output / "conclusion.json").write_text(
        json.dumps(conclusion, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        "\n".join(
            [
                "# Exp29 confirmatory source panel",
                "",
                "This package is source-only; its conclusion is **inconclusive**.",
                "No selector was fitted and no hypothesis test was run here.",
                f"Statistical unit: **{STATISTICS_UNIT}**.",
                "",
                f"- Expected rows: {EXPECTED_PANEL_ROWS}",
                f"- Observed rows: {coverage['observed_row_count']}",
                f"- Infeasible rows retained: {coverage['row_status_counts']['infeasible']}",
                f"- Feasibility semantics valid: {coverage['feasibility_semantics_valid']}",
                f"- Source panel valid: {coverage['source_panel_valid']}",
                f"- Raw SHA-256: `{raw_sha}`",
                "",
                "Matched-budget support is restricted to feasible active rows. ",
                "All inference remains unconditional over registered cells.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _validate_attempt_receipts(
    receipt: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[AttemptReceipt, ...]:
    raw_attempts = receipt.get("attempts")
    if not isinstance(raw_attempts, list):
        raise ValueError("confirmatory package attempts must be a list")
    attempts: list[AttemptReceipt] = []
    for value in raw_attempts:
        if not isinstance(value, Mapping):
            raise ValueError("confirmatory attempt receipt is malformed")
        if value.get("planned_coverage_valid") is not True:
            raise ValueError("confirmatory attempt plan receipt is malformed")
        attempt = AttemptReceipt(
            seed=int(value["seed"]),
            path=str(value["path"]),
            run_status=str(value["run_status"]),
            run_id=str(value["run_id"]),
            planned_coverage_valid=bool(value["planned_coverage_valid"]),
            observed_row_count=int(value["observed_row_count"]),
            row_status_counts=dict(value["row_status_counts"]),
            file_sha256=dict(value["file_sha256"]),
        )
        seed_rows = [row for row in rows if int(row["seed"]) == attempt.seed]
        counts = {status: 0 for status in ROW_STATUSES}
        for row in seed_rows:
            counts[str(row["status"])] += 1
        expected_run_status = (
            "complete_with_failures"
            if counts["failed"] or counts["invalid"]
            else "complete"
        )
        if (
            attempt.seed not in EXPECTED_SEEDS
            or attempt.run_status not in TERMINAL_RUN_STATUSES
            or attempt.run_status != expected_run_status
            or not attempt.run_id
            or attempt.planned_coverage_valid is not True
            or set(attempt.file_sha256) != set(ATTEMPT_FILES)
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in attempt.file_sha256.values()
            )
            or any(row.get("run_id") != attempt.run_id for row in seed_rows)
            or any(row.get("_run_status") != attempt.run_status for row in seed_rows)
            or any(
                row.get("_effective_status") != row.get("status") for row in seed_rows
            )
            or any(row.get("_attempt_path") != attempt.path for row in seed_rows)
            or attempt.observed_row_count != len(seed_rows)
            or dict(attempt.row_status_counts) != counts
        ):
            raise ValueError("confirmatory attempt counts disagree with raw rows")
        attempts.append(attempt)
    if tuple(attempt.seed for attempt in attempts) != tuple(
        sorted(attempt.seed for attempt in attempts)
    ):
        raise ValueError("confirmatory attempt receipts are not seed ordered")
    return tuple(attempts)


def _coverage_from_package(
    rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[AttemptReceipt],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    collection = PanelCollection(
        rows=tuple(rows),
        attempts=tuple(attempts),
        config=config,
        config_sha256="unused",
        config_file_sha256="unused",
        source_contract={},
        provenance_identity=None,
    )
    return panel_coverage(collection)


def load_source_panel_package(
    package_dir: str | Path,
    *,
    require_complete: bool = True,
    config_path: str | Path = DEFAULT_CONFIG,
) -> SourcePanelPackage:
    directory = Path(package_dir)
    receipt_path = directory / "source_panel_receipt.json"
    conclusion_path = directory / "conclusion.json"
    receipt_bytes = receipt_path.read_bytes()
    conclusion_bytes = conclusion_path.read_bytes()
    receipt = _read_json(receipt_path)
    conclusion = _read_json(conclusion_path)
    if not isinstance(receipt, Mapping) or not isinstance(conclusion, Mapping):
        raise ValueError("confirmatory package metadata is malformed")
    config = load_json_config(config_path)
    run_provenance = receipt.get("run_provenance")
    if not isinstance(run_provenance, Mapping):
        raise ValueError("confirmatory package lacks run provenance")
    git = run_provenance.get("run_git")
    runtime = run_provenance.get("runtime_versions")
    if not isinstance(git, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("confirmatory package run provenance is malformed")
    contract = validate_source_contract(
        config, current_git=git, runtime_versions=runtime
    )
    expected_provenance = build_evidence_provenance(
        contract, run_label=REQUIRED_RUN_LABEL
    )
    payload = {
        key: value for key, value in receipt.items() if key != "receipt_payload_sha256"
    }
    payload_sha = receipt.get("receipt_payload_sha256")
    if (
        receipt.get("schema_version") != SOURCE_PACKAGE_SCHEMA_VERSION
        or receipt.get("protocol_version") != PROTOCOL_VERSION
        or receipt.get("experiment") != EXPERIMENT
        or receipt.get("profile") != PROFILE
        or receipt.get("statistics_unit") != STATISTICS_UNIT
        or receipt.get("conclusion") != "inconclusive"
        or receipt.get("standalone_inference_performed") is not False
        or receipt.get("standalone_inference_permitted") is not False
        or payload_sha != _canonical_sha256(payload)
        or receipt.get("registered_config_sha256") != contract.config_sha256
        or receipt.get("registered_config_file_sha256") != contract.config_file_sha256
        or receipt.get("source_contract") != config.get("source_binding")
        or receipt.get("source_contract_sha256")
        != _canonical_sha256(config["source_binding"])
        or dict(run_provenance) != expected_provenance
        or conclusion.get("profile") != PROFILE
        or conclusion.get("statistics_unit") != STATISTICS_UNIT
        or conclusion.get("conclusion") != "inconclusive"
        or conclusion.get("source_panel_receipt_payload_sha256") != payload_sha
        or conclusion.get("coverage") != receipt.get("coverage")
        or conclusion.get("source_panel_valid")
        != receipt.get("coverage", {}).get("source_panel_valid")
    ):
        raise ValueError("confirmatory source package receipt is invalid")
    raw_name = receipt.get("raw_metrics_file")
    if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
        raise ValueError("confirmatory raw metrics path is invalid")
    raw_bytes = (directory / raw_name).read_bytes()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    if raw_sha != receipt.get("raw_metrics_sha256"):
        raise ValueError("confirmatory raw metrics hash is invalid")
    rows = _read_jsonl(directory / raw_name)
    if raw_bytes != _canonical_jsonl(rows) or len(rows) != receipt.get(
        "raw_metrics_row_count"
    ):
        raise ValueError("confirmatory raw metrics encoding/count is invalid")
    _validate_raw_schema(rows, config=config, provenance=run_provenance)
    attempts = _validate_attempt_receipts(receipt, rows)
    coverage = _coverage_from_package(rows, attempts, config)
    if coverage != receipt.get("coverage") or coverage != conclusion.get("coverage"):
        raise ValueError("confirmatory coverage is not reproducible")
    if conclusion.get("raw_metrics_sha256") != raw_sha:
        raise ValueError("confirmatory conclusion raw hash is invalid")
    if require_complete and coverage["source_panel_valid"] is not True:
        raise ValueError("confirmatory source panel is incomplete or invalid")
    return SourcePanelPackage(
        receipt=dict(receipt),
        rows=tuple(rows),
        receipt_payload_sha256=str(payload_sha),
        receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        conclusion_file_sha256=hashlib.sha256(conclusion_bytes).hexdigest(),
        raw_metrics_sha256=raw_sha,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Package the preregistered Exp29 confirmatory source panel"
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
    print(write_source_panel_package(collection, args.output_dir))


if __name__ == "__main__":
    main()
