"""Read-only validation for the reviewed exp14 IBL compact neural cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibl_neural_cache import (  # noqa: E402
    CompactNeuralCohort,
    IBLNeuralCacheError,
    load_compact_neural_cohort,
)

DEFAULT_CONFIG = Path("configs/formal/exp14_ibl_multisession_neural.json")
_REQUIRED_CONFIG_FIELDS = (
    "profile",
    "data_mode",
    "compact_cache_manifest",
    "expected_source_manifest_sha256",
    "expected_acquisition_bundle_sha256",
    "expected_bwm_repository_commit",
    "expected_compact_manifest_sha256",
    "expected_compact_bundle_sha256",
    "planned_sessions",
    "planned_animals",
)


class CompactValidationError(ValueError):
    """Raised when the formal config cannot identify one reviewed cache."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_from_project(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=True)


def _load_formal_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = _resolve_from_project(path)
    if not config_path.is_file():
        raise CompactValidationError("formal config path is not a file")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompactValidationError("formal config is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise CompactValidationError("formal config must be a JSON object")
    missing = sorted(set(_REQUIRED_CONFIG_FIELDS).difference(raw))
    if missing:
        raise CompactValidationError(f"formal config is missing fields: {missing}")
    if raw["profile"] != "formal":
        raise CompactValidationError("validator accepts only profile='formal'")
    if raw["data_mode"] != "frozen_compact_cache":
        raise CompactValidationError(
            "validator accepts only data_mode='frozen_compact_cache'"
        )
    for field in ("planned_sessions", "planned_animals"):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CompactValidationError(f"{field} must be a positive integer")
    for field in _REQUIRED_CONFIG_FIELDS[2:8]:
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise CompactValidationError(f"{field} must be a non-empty string")
    return config_path, raw


def _view_report(cohort: CompactNeuralCohort) -> dict[str, dict[str, object]]:
    if not cohort.sessions:
        return {}
    view_names = set(cohort.sessions[0].count_views)
    if any(set(session.count_views) != view_names for session in cohort.sessions):
        raise CompactValidationError("sessions do not expose one shared view family")
    report: dict[str, dict[str, object]] = {}
    for view in sorted(view_names):
        shapes = [session.count_views[view].shape for session in cohort.sessions]
        report[view] = {
            "trial_records": int(sum(shape[0] for shape in shapes)),
            "valid_trials": int(
                sum(session.valid_masks[view].sum() for session in cohort.sessions)
            ),
            "time_bins": sorted({int(shape[1]) for shape in shapes}),
            "total_binned_spikes": int(
                sum(session.count_views[view].sum() for session in cohort.sessions)
            ),
        }
    return report


def validate_compact(
    config: str | Path = DEFAULT_CONFIG,
    *,
    compact_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Validate config and all hash-bound cache artifacts without writing files."""

    config_path, values = _load_formal_config(config)
    if compact_manifest is None:
        manifest_path = _resolve_from_project(values["compact_cache_manifest"])
    else:
        manifest_path = _resolve_from_project(compact_manifest)
    if not manifest_path.is_file():
        raise CompactValidationError("compact manifest path is not a file")

    cohort = load_compact_neural_cohort(
        manifest_path,
        expected_source_manifest_sha256=values["expected_source_manifest_sha256"],
        expected_acquisition_bundle_sha256=values["expected_acquisition_bundle_sha256"],
        expected_bwm_repository_commit=values["expected_bwm_repository_commit"],
        expected_compact_manifest_sha256=values["expected_compact_manifest_sha256"],
        expected_compact_bundle_sha256=values["expected_compact_bundle_sha256"],
        expected_sessions=values["planned_sessions"],
        minimum_animals=values["planned_animals"],
    )
    status_counts = Counter(item.status for item in cohort.dispositions)
    validation_counts = Counter(
        item.acquisition_validation_status for item in cohort.complete_dispositions
    )
    animals = sorted({session.animal_id for session in cohort.sessions})
    eids = sorted(session.eid for session in cohort.sessions)
    regions = sorted(
        {
            str(region)
            for session in cohort.sessions
            for region in session.regions.tolist()
        }
    )
    return {
        "schema_version": "exp14_ibl_compact_validation_report_v1",
        "status": "valid",
        "offline_only": True,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "compact_manifest_path": str(manifest_path),
        "compact_manifest_sha256": cohort.compact_manifest_sha256,
        "compact_bundle_sha256": cohort.compact_bundle_sha256,
        "evidence_scope": cohort.evidence_scope,
        "expected": {
            "sessions": values["planned_sessions"],
            "minimum_animals": values["planned_animals"],
            "source_manifest_sha256": values["expected_source_manifest_sha256"],
            "acquisition_bundle_sha256": values["expected_acquisition_bundle_sha256"],
            "bwm_repository_commit": values["expected_bwm_repository_commit"],
        },
        "observed": {
            "dispositions": len(cohort.dispositions),
            "complete_sessions": len(cohort.sessions),
            "unique_eids": len(set(eids)),
            "unique_animals": len(animals),
            "total_units": int(
                sum(len(session.unit_ids) for session in cohort.sessions)
            ),
            "unique_region_labels": len(regions),
            "status_counts": dict(sorted(status_counts.items())),
            "acquisition_validation_counts": dict(sorted(validation_counts.items())),
            "views": _view_report(cohort),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="formal exp14 JSON config, relative to the project root by default",
    )
    parser.add_argument(
        "--compact-manifest",
        help="optional read-only local manifest override; no files are copied",
    )
    parser.add_argument("--indent", type=int, default=2, choices=range(0, 9))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = validate_compact(args.config, compact_manifest=args.compact_manifest)
    except (CompactValidationError, IBLNeuralCacheError, OSError) as error:
        payload = {
            "status": "invalid",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
