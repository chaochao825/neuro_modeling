"""Offline-only loader for reviewed exp14 compact neural artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.ibl_multisession import PreparedIBLNeuralSession


class IBLNeuralCacheError(RuntimeError):
    """Raised when a compact cache is missing, altered, or under-specified."""


_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_ACQUISITION_APPROVAL_SHA256 = (
    "64e0b1ead4175b2007b0c4aa01524e9e9db4beebde63f016cf46e245f0da4287"
)
_BIN_SIZE_S = 0.02
_WINDOW_S = (-0.5, 0.0)
_N_TIME_BINS = 25
_EXPECTED_TIME_AXIS = np.arange(-0.5, 0.0, 0.02, dtype=np.float64)
_MASK_PROTOCOL = "audited_pinned_bwm_formula_min_rt_0p08_max_rt_2_default_nan_v2"
_BWM_LOADER_SHA256 = "c2e570c62cd0e047303c97d7999711b659a1c37eaa56dd2740ffff2c81f85321"
_POSTPROCESS_LAUNCHER_SHA256 = (
    "fe8a30ab705d7e0c8474e574fcae36ce68f510d2d53de81003edd3a3f05f2837"
)
_POSTPROCESS_SCRIPT_SHA256 = (
    "951a516ae7eb2aa024f37e1890d6afd34dd19378905b90744dfadb32e7a76a17"
)
_COMPACT_SCHEMA_SHA256 = (
    "f29c5506be93393499b90535c401bec1c82e0754737733897716cd4e2fade39d"
)
_QC_SORTING_RULE = "probe_name_then_pid_then_cluster_id_v1"
_QC_DATASET_NAMES = (
    "passingSpikes.table.pqt",
    "clusters.metrics.pqt",
    "clusters.channels.npy",
    "channels.brainLocationIds_ccf_2017.npy",
)
_COMPACT_REQUIRED_DATASET_NAMES = tuple(
    sorted(
        (
            "_ibl_trials.table.pqt",
            "_ibl_wheel.position.npy",
            "_ibl_wheel.timestamps.npy",
            "channels.brainLocationIds_ccf_2017.npy",
            "clusters.channels.npy",
            "clusters.metrics.pqt",
            "passingSpikes.table.pqt",
        )
    )
)
_CAMERA_RULE = "first_intact_pair_in_left_right_body_priority"
_CAMERA_VIEWS = ("left", "right", "body")
_ACQUISITION_VALIDATION_STATUSES = {
    "official_pinned_load_good_units_passed",
    "official_good_unit_loader_failed_retained",
    "official_good_unit_loader_skipped_missing_inventory_retained",
    "official_good_unit_loader_mixed_nonpass_retained",
}
_EXTERNAL_ATLAS_PATHS = (
    "/home/wangmeiqi/Downloads/ONE/openalyx.internationalbrainlab.org/"
    "histology/ATLAS/Needles/Allen/average_template_25.nrrd",
    "/home/wangmeiqi/Downloads/ONE/openalyx.internationalbrainlab.org/"
    "histology/ATLAS/Needles/Allen/annotation_25.nrrd",
)
_COMPLETE_METADATA_KEYS = {
    "schema_version",
    "candidate_rank",
    "status",
    "error_type",
    "error",
    "eid",
    "animal_id",
    "source_manifest_sha256",
    "acquisition_bundle_sha256",
    "acquisition_approval_sha256",
    "bwm_repository_commit",
    "bwm_loader_sha256",
    "bwm_loader_relative_path",
    "postprocess_launcher_sha256",
    "spike_sorting_revision",
    "unit_qc_threshold",
    "unit_qc_applied",
    "acquisition_validation_status",
    "unit_qc_method",
    "unit_qc_equivalence_sha256",
    "unit_qc_equivalence",
    "acquisition_official_loader_receipts",
    "acquisition_dataset_failure_receipts",
    "auxiliary_dataset_failure_policy",
    "compact_required_input_bundle",
    "compact_required_input_bundle_sha256",
    "acquisition_session_status_sha256",
    "acquisition_external_side_effect",
    "postprocess_input_policy",
    "region_mapping_input",
    "trial_dataset_uuid",
    "trial_dataset_revision",
    "trial_dataset_md5",
    "official_bwm_mask_sha256",
    "official_bwm_mask_evidence",
    "download_summary_sha256",
    "download_log_sha256",
    "session_status_bundle_sha256",
    "bin_size_s",
    "window_s",
    "n_time_bins",
    "window_semantics",
    "time_axis_semantics",
    "trial_count",
    "official_bwm_mask_count",
    "unit_count",
    "motion_energy_proxy",
    "wheel_displacement",
    "cv_block_policy",
    "observed_good_spike_support_s",
    "acquisition_session_status",
    "created_at_utc",
}
_COMPACT_BUNDLE_KEYS = {
    "schema_version",
    "created_at_utc",
    "compact_manifest_sha256",
    "artifact_sha256",
    "source_manifest_sha256",
    "acquisition_bundle_sha256",
    "acquisition_approval_sha256",
    "download_summary_sha256",
    "download_log_sha256",
    "session_status_bundle_sha256",
    "acquisition_external_side_effect",
    "postprocess_script_sha256",
    "postprocess_launcher_sha256",
    "compact_schema_sha256",
    "bwm_loader_sha256",
    "bwm_loader_relative_path",
    "region_mapping_path",
    "region_mapping_sha256",
    "region_mapping_provenance_path",
    "region_mapping_provenance_sha256",
    "iblatlas_version",
    "iblatlas_source_commit",
    "iblatlas_regions_source_sha256",
    "complete_sessions",
    "failed_sessions",
}
_REGION_MAPPING_SHA256 = (
    "63654b8d35c7c1b5665636b645da774776ee8263658192f5dca1e815095e9147"
)
_REGION_MAPPING_PROVENANCE_SHA256 = (
    "a01b7fa535e6de437ac46e8cf9de68a87d6a9b5587d055a3935476d956109fdc"
)
_REGIONS_SOURCE_SHA256 = (
    "cdfe3e5c8ed350af182b14f7ce627096484a529da9837264f53f3472319dcc63"
)
_REGION_MAPPING_COLUMNS = (
    "id",
    "atlas_id",
    "name",
    "acronym",
    "st_level",
    "ontology_id",
    "hemisphere_id",
    "weight",
    "parent_structure_id",
    "depth",
    "graph_id",
    "graph_order",
    "structure_id_path",
    "color_hex_triplet",
    "neuro_name_structure_id",
    "neuro_name_structure_id_path",
    "failed",
    "sphinx_id",
    "structure_name_facet",
    "failed_facet",
    "safe_name",
)
_PUBLIC_TRIAL_COLUMNS = (
    "trial_id",
    "stimulus",
    "stimulus_side",
    "choice",
    "wheel",
    "reward",
    "reaction_time",
    "stim_on",
    "first_movement",
    "timing_valid",
    "official_bwm_mask",
    "block_id",
    "motion_energy_proxy",
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IBLNeuralCacheError(f"{name} must be a non-empty string")
    return value.strip()


def _require_hash(value: object, *, name: str, pattern: re.Pattern[str]) -> str:
    text = _require_text(value, name=name)
    if pattern.fullmatch(text) is None:
        raise IBLNeuralCacheError(f"{name} has an invalid digest")
    return text


def _require_uuid(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    try:
        parsed = uuid.UUID(text)
    except (ValueError, AttributeError) as error:
        raise IBLNeuralCacheError(f"{name} must be a canonical UUID") from error
    if str(parsed) != text:
        raise IBLNeuralCacheError(f"{name} must be a canonical lowercase UUID")
    return text


def _require_json_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IBLNeuralCacheError(f"{name} must be an integer >= {minimum}")
    return value


def _require_utc_timestamp(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise IBLNeuralCacheError(f"{name} is not a valid ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise IBLNeuralCacheError(f"{name} must include an explicit UTC offset")
    return text


def _strict_bool(value: object, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise IBLNeuralCacheError(f"{name} must be an explicit binary value")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root: Path, value: object, expected_sha256: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or relative == Path("."):
        raise IBLNeuralCacheError("compact paths must be non-empty and relative")
    try:
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise IBLNeuralCacheError(
            f"compact file is missing or escapes root: {relative}"
        ) from error
    if not path.is_file():
        raise IBLNeuralCacheError(f"compact artifact is not a file: {relative}")
    expected = _require_hash(
        expected_sha256, name=f"{relative} SHA-256", pattern=_HEX_64
    )
    if _sha256(path) != expected:
        raise IBLNeuralCacheError(f"compact artifact SHA-256 mismatch: {relative}")
    return path


@dataclass(frozen=True, slots=True)
class CompactSessionDisposition:
    candidate_rank: int
    eid: str
    animal_id: str
    status: str
    error_type: str
    error: str
    source_manifest_sha256: str
    acquisition_bundle_sha256: str
    bwm_repository_commit: str
    spike_sorting_revision: str
    unit_qc_threshold: float
    unit_qc_applied: bool
    acquisition_validation_status: str
    unit_qc_method: str
    unit_qc_equivalence_sha256: str
    trial_dataset_uuid: str
    trial_dataset_revision: str
    trial_dataset_md5: str
    official_bwm_mask_sha256: str
    download_summary_sha256: str
    session_status_bundle_sha256: str
    region_mapping_path: str
    region_mapping_sha256: str
    region_mapping_provenance_path: str
    region_mapping_provenance_sha256: str
    iblatlas_version: str
    iblatlas_source_commit: str
    iblatlas_regions_source_sha256: str


@dataclass(frozen=True, slots=True)
class CompactNeuralCohort:
    dispositions: tuple[CompactSessionDisposition, ...]
    sessions: tuple[PreparedIBLNeuralSession, ...]
    compact_manifest_sha256: str
    compact_bundle_sha256: str
    evidence_scope: str = "reviewed_offline_compact_ibl_counts"

    @property
    def complete_dispositions(self) -> tuple[CompactSessionDisposition, ...]:
        return tuple(item for item in self.dispositions if item.status == "complete")


_MANIFEST_COLUMNS = (
    "candidate_rank",
    "eid",
    "animal_id",
    "status",
    "error_type",
    "error",
    "npz_path",
    "npz_sha256",
    "stimulus_trials_path",
    "stimulus_trials_sha256",
    "movement_trials_path",
    "movement_trials_sha256",
    "metadata_path",
    "metadata_sha256",
    "source_manifest_sha256",
    "acquisition_bundle_sha256",
    "bwm_repository_commit",
    "spike_sorting_revision",
    "unit_qc_threshold",
    "unit_qc_applied",
    "acquisition_validation_status",
    "unit_qc_method",
    "unit_qc_equivalence_sha256",
    "trial_dataset_uuid",
    "trial_dataset_revision",
    "trial_dataset_md5",
    "official_bwm_mask_sha256",
    "download_summary_sha256",
    "session_status_bundle_sha256",
    "region_mapping_path",
    "region_mapping_sha256",
    "region_mapping_provenance_path",
    "region_mapping_provenance_sha256",
    "iblatlas_version",
    "iblatlas_source_commit",
    "iblatlas_regions_source_sha256",
)


def _expect_exact_keys(
    value: object, expected: set[str], *, name: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise IBLNeuralCacheError(f"{name} has an unexpected schema")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    names = {
        "stimulus_pre_counts",
        "movement_pre_counts",
        "stimulus_pre_valid",
        "movement_pre_valid",
        "stimulus_pre_time",
        "movement_pre_time",
        "unit_ids",
        "regions",
        "trial_ids",
    }
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != names:
                raise IBLNeuralCacheError(
                    "compact NPZ must contain exactly the reviewed nine arrays"
                )
            arrays = {name: np.array(payload[name], copy=True) for name in names}
    except (OSError, ValueError) as error:
        raise IBLNeuralCacheError("compact NPZ cannot be loaded safely") from error

    trial_ids = arrays["trial_ids"]
    unit_ids = arrays["unit_ids"]
    regions = arrays["regions"]
    if (
        trial_ids.dtype != np.dtype("int64")
        or trial_ids.ndim != 1
        or len(trial_ids) == 0
        or not np.array_equal(trial_ids, np.arange(len(trial_ids), dtype=np.int64))
    ):
        raise IBLNeuralCacheError(
            "compact trial_ids must be sequential int64 values starting at zero"
        )
    if (
        unit_ids.dtype.kind != "U"
        or regions.dtype.kind != "U"
        or unit_ids.ndim != 1
        or regions.shape != unit_ids.shape
        or len(unit_ids) == 0
        or len(set(unit_ids.tolist())) != len(unit_ids)
        or np.any(np.char.str_len(unit_ids) == 0)
        or np.any(np.char.str_len(regions) == 0)
    ):
        raise IBLNeuralCacheError(
            "compact unit_ids/regions must be strict non-empty Unicode vectors"
        )
    expected_shape = (len(trial_ids), _N_TIME_BINS, len(unit_ids))
    for view in ("stimulus_pre", "movement_pre"):
        counts = arrays[f"{view}_counts"]
        valid = arrays[f"{view}_valid"]
        axis = arrays[f"{view}_time"]
        if counts.dtype != np.dtype("int32") or counts.shape != expected_shape:
            raise IBLNeuralCacheError(f"{view} counts must be int32 trial x 25 x unit")
        if np.any(counts < 0):
            raise IBLNeuralCacheError(f"{view} counts must be non-negative")
        if valid.dtype != np.dtype("bool") or valid.shape != (len(trial_ids),):
            raise IBLNeuralCacheError(f"{view} valid mask must be a bool vector")
        if np.any(counts[~valid] != 0):
            raise IBLNeuralCacheError(
                f"{view} counts outside the reviewed valid mask must be zero"
            )
        if (
            axis.dtype != np.dtype("float64")
            or axis.shape != (_N_TIME_BINS,)
            or not np.array_equal(axis, _EXPECTED_TIME_AXIS)
        ):
            raise IBLNeuralCacheError(
                f"{view} time axis must be exact float64 left bin edges"
            )
    return arrays


def _json_int_list(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list):
        raise IBLNeuralCacheError(f"{name} must be a JSON integer list")
    return [
        _require_json_int(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _validate_dataset_identifier(
    value: object, *, name: str, expected_dataset_name: str | None = None
) -> Mapping[str, Any]:
    dataset = _expect_exact_keys(value, {"dataset_uuid", "name", "md5"}, name=name)
    _require_uuid(dataset["dataset_uuid"], name=f"{name}.dataset_uuid")
    dataset_name = _require_text(dataset["name"], name=f"{name}.name")
    if expected_dataset_name is not None and dataset_name != expected_dataset_name:
        raise IBLNeuralCacheError(f"{name} has the wrong frozen dataset name")
    _require_hash(dataset["md5"], name=f"{name}.md5", pattern=_HEX_32)
    return dataset


def _dataset_identity(value: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(value["dataset_uuid"]),
        str(value["name"]),
        str(value["md5"]),
    )


def _load_region_mapping(
    metadata: Mapping[str, Any], *, root: Path
) -> Mapping[int, str]:
    declaration = _expect_exact_keys(
        metadata.get("region_mapping_input"),
        {
            "schema_version",
            "mapping_relative_path",
            "mapping_sha256",
            "provenance_relative_path",
            "provenance_sha256",
            "source_package",
            "source_version",
            "source_commit",
            "source_commit_status",
            "regions_source_sha256",
        },
        name="region_mapping_input",
    )
    expected = {
        "schema_version": "ibl_exp14_region_mapping_input_v1",
        "mapping_relative_path": "provenance/iblatlas_allen_structure_tree.csv",
        "mapping_sha256": _REGION_MAPPING_SHA256,
        "provenance_relative_path": "provenance/region_mapping_provenance.json",
        "provenance_sha256": _REGION_MAPPING_PROVENANCE_SHA256,
        "source_package": "iblatlas",
        "source_version": "1.1.0",
        "source_commit": None,
        "source_commit_status": "unavailable_pypi_distribution_direct_url_absent",
        "regions_source_sha256": _REGIONS_SOURCE_SHA256,
    }
    if declaration != expected:
        raise IBLNeuralCacheError("region mapping input is not the reviewed resource")
    mapping_path = _safe_file(
        root, declaration["mapping_relative_path"], declaration["mapping_sha256"]
    )
    _safe_file(
        root,
        declaration["provenance_relative_path"],
        declaration["provenance_sha256"],
    )
    try:
        frame = pd.read_csv(mapping_path)
    except (OSError, pd.errors.ParserError) as error:
        raise IBLNeuralCacheError(
            "reviewed region mapping CSV cannot be parsed"
        ) from error
    if tuple(str(column) for column in frame.columns) != _REGION_MAPPING_COLUMNS:
        raise IBLNeuralCacheError("reviewed region mapping CSV schema is wrong")
    ids = _integer_csv_column(frame, "id")
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise IBLNeuralCacheError(
            "reviewed region mapping IDs must be non-empty/unique"
        )
    if frame["acronym"].isna().any():
        raise IBLNeuralCacheError("reviewed region mapping contains a null acronym")
    acronyms = frame["acronym"].astype(str).str.strip()
    if (acronyms.str.len() == 0).any():
        raise IBLNeuralCacheError("reviewed region mapping contains an empty acronym")
    return dict(zip(ids.tolist(), acronyms.tolist(), strict=True))


def _validate_unit_qc(
    metadata: Mapping[str, Any],
    *,
    expected_sha256: str,
    unit_ids: np.ndarray,
    regions: np.ndarray,
    region_mapping: Mapping[int, str],
) -> tuple[tuple[tuple[str, str], ...], tuple[Mapping[str, Any], ...]]:
    payload = _expect_exact_keys(
        metadata.get("unit_qc_equivalence"),
        {"sorting_rule", "probes"},
        name="unit_qc_equivalence",
    )
    if payload["sorting_rule"] != _QC_SORTING_RULE:
        raise IBLNeuralCacheError("unit QC sorting rule is not the reviewed rule")
    probes = payload["probes"]
    if not isinstance(probes, list) or not probes:
        raise IBLNeuralCacheError("unit_qc_equivalence.probes must be non-empty")
    if _canonical_sha256(payload) != expected_sha256:
        raise IBLNeuralCacheError("unit QC equivalence canonical hash mismatch")

    expected_units: list[str] = []
    expected_regions: list[str] = []
    probe_keys: list[tuple[str, str]] = []
    qc_datasets: list[Mapping[str, Any]] = []
    for index, raw_probe in enumerate(probes):
        name = f"unit_qc_equivalence.probes[{index}]"
        probe = _expect_exact_keys(
            raw_probe,
            {
                "pid",
                "probe_name",
                "passing_cluster_ids",
                "metrics_label_ge_1_cluster_ids",
                "cluster_channel_ccf_acronym",
                "input_datasets",
            },
            name=name,
        )
        pid = _require_uuid(probe["pid"], name=f"{name}.pid")
        probe_name = _require_text(probe["probe_name"], name=f"{name}.probe_name")
        probe_keys.append((probe_name, pid))
        passing = _json_int_list(
            probe["passing_cluster_ids"], name=f"{name}.passing_cluster_ids"
        )
        metrics = _json_int_list(
            probe["metrics_label_ge_1_cluster_ids"],
            name=f"{name}.metrics_label_ge_1_cluster_ids",
        )
        if passing != sorted(set(passing)) or passing != metrics or not passing:
            raise IBLNeuralCacheError(
                "passingSpikes and metrics label>=1 cluster IDs must be equal and sorted"
            )
        mapping = probe["cluster_channel_ccf_acronym"]
        if not isinstance(mapping, list) or len(mapping) != len(passing):
            raise IBLNeuralCacheError("unit QC cluster-to-region mapping is incomplete")
        mapped_clusters: list[int] = []
        for map_index, entry in enumerate(mapping):
            if not isinstance(entry, list) or len(entry) != 4:
                raise IBLNeuralCacheError(
                    "unit QC cluster-to-region entries must be four-element lists"
                )
            cluster = _require_json_int(
                entry[0], name=f"{name}.mapping[{map_index}].cluster"
            )
            _require_json_int(entry[1], name=f"{name}.mapping[{map_index}].channel")
            if isinstance(entry[2], bool) or not isinstance(entry[2], int):
                raise IBLNeuralCacheError(
                    "unit QC cluster-to-region CCF ID must be an integer"
                )
            ccf_id = int(entry[2])
            acronym = _require_text(
                entry[3], name=f"{name}.mapping[{map_index}].acronym"
            )
            if region_mapping.get(abs(ccf_id)) != acronym:
                raise IBLNeuralCacheError(
                    "unit QC acronym disagrees with reviewed frozen region mapping"
                )
            mapped_clusters.append(cluster)
            expected_units.append(f"{pid}:{cluster}")
            expected_regions.append(acronym)
        if mapped_clusters != passing:
            raise IBLNeuralCacheError(
                "unit QC cluster-to-region mapping is not sorted/aligned"
            )
        datasets = probe["input_datasets"]
        if not isinstance(datasets, list) or len(datasets) != len(_QC_DATASET_NAMES):
            raise IBLNeuralCacheError("unit QC input dataset bundle is incomplete")
        for dataset_index, (dataset, dataset_name) in enumerate(
            zip(datasets, _QC_DATASET_NAMES, strict=True)
        ):
            qc_datasets.append(
                _validate_dataset_identifier(
                    dataset,
                    name=f"{name}.input_datasets[{dataset_index}]",
                    expected_dataset_name=dataset_name,
                )
            )
    if probe_keys != sorted(probe_keys) or len(set(probe_keys)) != len(probe_keys):
        raise IBLNeuralCacheError("unit QC probes are not uniquely sorted")
    if not np.array_equal(unit_ids, np.asarray(expected_units, dtype=unit_ids.dtype)):
        raise IBLNeuralCacheError("compact unit IDs are not bound to reviewed QC")
    if not np.array_equal(regions, np.asarray(expected_regions, dtype=regions.dtype)):
        raise IBLNeuralCacheError(
            "compact regions are not bound to reviewed QC mapping"
        )
    if len({_dataset_identity(item) for item in qc_datasets}) != len(qc_datasets):
        raise IBLNeuralCacheError("unit QC input dataset identities must be unique")
    return tuple(probe_keys), tuple(qc_datasets)


def _validate_official_mask(
    metadata: Mapping[str, Any],
    *,
    row: Mapping[str, object],
    trial_ids: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    evidence = _expect_exact_keys(
        metadata.get("official_bwm_mask_evidence"),
        {
            "protocol",
            "bwm_loader_sha256",
            "runtime_brainwidemap_imported",
            "no_truncation_precondition",
            "trial_ids",
            "mask",
            "selected_trial_ids",
            "trial_dataset",
        },
        name="official_bwm_mask_evidence",
    )
    if (
        evidence["protocol"] != _MASK_PROTOCOL
        or evidence["bwm_loader_sha256"] != _BWM_LOADER_SHA256
        or evidence["runtime_brainwidemap_imported"] is not False
    ):
        raise IBLNeuralCacheError("official BWM mask protocol is not pinned")
    ids = _json_int_list(evidence["trial_ids"], name="official mask trial_ids")
    mask_values = _json_int_list(evidence["mask"], name="official mask bits")
    selected = _json_int_list(
        evidence["selected_trial_ids"], name="official mask selected_trial_ids"
    )
    if any(item not in {0, 1} for item in mask_values):
        raise IBLNeuralCacheError("official BWM mask must contain only 0/1")
    if ids != trial_ids.tolist() or len(mask_values) != len(ids):
        raise IBLNeuralCacheError("official BWM mask trial IDs do not align with NPZ")
    expected_selected = [
        item for item, keep in zip(ids, mask_values, strict=True) if keep
    ]
    if selected != expected_selected:
        raise IBLNeuralCacheError("official BWM selected trial IDs disagree with mask")
    no_truncation = _expect_exact_keys(
        evidence["no_truncation_precondition"],
        {
            "n_trials",
            "n_trials_strictly_greater_than_400",
            "easy_trial_definition",
            "easy_trial_count",
            "easy_performance",
            "easy_performance_strictly_greater_than_0p9",
            "passed",
        },
        name="official mask no_truncation_precondition",
    )
    n_trials = _require_json_int(
        no_truncation["n_trials"], name="official mask no-truncation n_trials"
    )
    easy_count = _require_json_int(
        no_truncation["easy_trial_count"],
        name="official mask no-truncation easy_trial_count",
        minimum=1,
    )
    try:
        easy_performance = float(no_truncation["easy_performance"])
    except (TypeError, ValueError) as error:
        raise IBLNeuralCacheError(
            "official mask no-truncation performance must be numeric"
        ) from error
    if (
        n_trials != len(trial_ids)
        or n_trials <= 400
        or no_truncation["n_trials_strictly_greater_than_400"] is not True
        or no_truncation["easy_trial_definition"] != "abs(signed_contrast_percent)>=50"
        or easy_count > n_trials
        or not np.isfinite(easy_performance)
        or not 0.9 < easy_performance <= 1.0
        or no_truncation["easy_performance_strictly_greater_than_0p9"] is not True
        or no_truncation["passed"] is not True
    ):
        raise IBLNeuralCacheError(
            "official BWM mask lacks a valid full-table no-truncation proof"
        )
    dataset = _expect_exact_keys(
        evidence["trial_dataset"],
        {"dataset_uuid", "revision", "md5"},
        name="official mask trial_dataset",
    )
    if (
        _require_uuid(
            dataset["dataset_uuid"], name="official mask trial_dataset.dataset_uuid"
        )
        != str(row["trial_dataset_uuid"])
        or _require_text(
            dataset["revision"], name="official mask trial_dataset.revision"
        )
        != str(row["trial_dataset_revision"])
        or _require_hash(
            dataset["md5"], name="official mask trial_dataset.md5", pattern=_HEX_32
        )
        != str(row["trial_dataset_md5"])
    ):
        raise IBLNeuralCacheError("official BWM mask trial dataset binding is wrong")
    expected_hash = _require_hash(
        row["official_bwm_mask_sha256"],
        name="official_bwm_mask_sha256",
        pattern=_HEX_64,
    )
    if _canonical_sha256(evidence) != expected_hash:
        raise IBLNeuralCacheError("official BWM mask canonical hash mismatch")
    if metadata.get("official_bwm_mask_count") != int(sum(mask_values)):
        raise IBLNeuralCacheError("official BWM mask count disagrees with metadata")
    return np.asarray(mask_values, dtype=bool), no_truncation


def _validate_support(
    metadata: Mapping[str, Any], probe_keys: Sequence[tuple[str, str]]
) -> tuple[tuple[float, float], ...]:
    raw_support = metadata.get("observed_good_spike_support_s")
    if not isinstance(raw_support, list) or len(raw_support) != len(probe_keys):
        raise IBLNeuralCacheError("observed good-spike support is incomplete")
    support: list[tuple[float, float]] = []
    for index, (raw_item, (probe_name, pid)) in enumerate(
        zip(raw_support, probe_keys, strict=True)
    ):
        name = f"observed_good_spike_support_s[{index}]"
        item = _expect_exact_keys(
            raw_item, {"pid", "probe_name", "start", "stop", "semantics"}, name=name
        )
        if item["pid"] != pid or item["probe_name"] != probe_name:
            raise IBLNeuralCacheError("good-spike support probe order/binding is wrong")
        if (
            item["semantics"]
            != "conservative_observed_good_spike_support_not_recording_interval"
        ):
            raise IBLNeuralCacheError("good-spike support semantics are not explicit")
        try:
            start, stop = float(item["start"]), float(item["stop"])
        except (TypeError, ValueError) as error:
            raise IBLNeuralCacheError(
                "good-spike support bounds must be numeric"
            ) from error
        if not np.isfinite([start, stop]).all() or start >= stop:
            raise IBLNeuralCacheError("good-spike support bounds are invalid")
        support.append((start, stop))
    return tuple(support)


def _validate_motion_proxy(
    metadata: Mapping[str, Any],
) -> tuple[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    motion = _expect_exact_keys(
        metadata.get("motion_energy_proxy"),
        {"stimulus_pre", "movement_pre"},
        name="motion_energy_proxy",
    )
    camera_view: str | None = None
    datasets_by_view: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for view in ("stimulus_pre", "movement_pre"):
        item = _expect_exact_keys(
            motion[view],
            {
                "proxy",
                "camera_view",
                "window_s",
                "window_semantics",
                "aggregation",
                "time_dataset",
                "energy_dataset",
            },
            name=f"motion_energy_proxy.{view}",
        )
        if (
            item["proxy"] != "camera_roi_motion_energy_mean"
            or item["camera_view"] not in _CAMERA_VIEWS
        ):
            raise IBLNeuralCacheError("motion-energy proxy declaration is invalid")
        if (
            item["window_s"] != list(_WINDOW_S)
            or item["window_semantics"] != "half-open_[event+start,event+stop)"
            or item["aggregation"] != "finite_sample_arithmetic_mean"
        ):
            raise IBLNeuralCacheError("motion-energy proxy window is not pinned")
        declared_view = str(item["camera_view"])
        if camera_view is None:
            camera_view = declared_view
        elif declared_view != camera_view:
            raise IBLNeuralCacheError(
                "motion-energy views must use the same selected camera pair"
            )
        time_dataset = _validate_dataset_identifier(
            item["time_dataset"],
            name=f"motion_energy_proxy.{view}.time_dataset",
            expected_dataset_name=f"_ibl_{declared_view}Camera.times.npy",
        )
        energy_dataset = _validate_dataset_identifier(
            item["energy_dataset"],
            name=f"motion_energy_proxy.{view}.energy_dataset",
            expected_dataset_name=f"{declared_view}Camera.ROIMotionEnergy.npy",
        )
        datasets_by_view.append((time_dataset, energy_dataset))
    if datasets_by_view[0] != datasets_by_view[1] or camera_view is None:
        raise IBLNeuralCacheError(
            "motion-energy dataset identities must be shared across views"
        )
    return camera_view, datasets_by_view[0]


def _validate_wheel_proxy(
    metadata: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    wheel = _expect_exact_keys(
        metadata.get("wheel_displacement"),
        {
            "proxy",
            "start_event",
            "stop_event",
            "window_semantics",
            "aggregation",
            "time_dataset",
            "position_dataset",
        },
        name="wheel_displacement",
    )
    if {
        "proxy": wheel["proxy"],
        "start_event": wheel["start_event"],
        "stop_event": wheel["stop_event"],
        "window_semantics": wheel["window_semantics"],
        "aggregation": wheel["aggregation"],
    } != {
        "proxy": "total_absolute_wheel_displacement",
        "start_event": "stimOn_times",
        "stop_event": "response_times",
        "window_semantics": "half-open_[start_event,stop_event)",
        "aggregation": "sum_absolute_first_differences",
    }:
        raise IBLNeuralCacheError("wheel-displacement proxy declaration is invalid")
    time_dataset = _validate_dataset_identifier(
        wheel["time_dataset"],
        name="wheel_displacement.time_dataset",
        expected_dataset_name="_ibl_wheel.timestamps.npy",
    )
    position_dataset = _validate_dataset_identifier(
        wheel["position_dataset"],
        name="wheel_displacement.position_dataset",
        expected_dataset_name="_ibl_wheel.position.npy",
    )
    return time_dataset, position_dataset


def _validate_acquisition_loader_receipts(
    metadata: Mapping[str, Any],
    *,
    probe_keys: Sequence[tuple[str, str]],
    qc_method: str,
) -> tuple[str, tuple[str, ...]]:
    if "offline_official_loader_attempt" in metadata:
        raise IBLNeuralCacheError(
            "postprocessing must not perform a second official-loader attempt"
        )
    receipts = metadata.get("acquisition_official_loader_receipts")
    if not isinstance(receipts, list) or len(receipts) != len(probe_keys):
        raise IBLNeuralCacheError("acquisition official-loader receipts are incomplete")
    events: list[str] = []
    run_ids: set[str] = set()
    common_keys = {
        "run_id",
        "timestamp_utc",
        "eid",
        "pid",
        "probe_name",
        "revision",
        "qc",
        "event",
    }
    for index, (raw_receipt, (probe_name, pid)) in enumerate(
        zip(receipts, probe_keys, strict=True)
    ):
        name = f"acquisition_official_loader_receipts[{index}]"
        receipt = _expect_exact_keys(
            raw_receipt, {"source_record", "source_record_sha256"}, name=name
        )
        record = receipt["source_record"]
        if not isinstance(record, dict):
            raise IBLNeuralCacheError(f"{name}.source_record must be a JSON object")
        event = str(record.get("event", ""))
        if event == "official_good_unit_loader_passed":
            expected_keys = common_keys | {"unit_count", "spike_count"}
        elif event == "official_good_unit_loader_failed":
            expected_keys = common_keys | {"error_type", "error", "traceback"}
        elif event == "official_good_unit_loader_skipped_missing_inventory":
            expected_keys = common_keys | {"missing_dataset_uuids"}
        else:
            raise IBLNeuralCacheError(
                "acquisition official-loader receipt event is invalid"
            )
        record = _expect_exact_keys(record, expected_keys, name=f"{name}.source_record")
        if (
            record["eid"] != metadata.get("eid")
            or record["pid"] != pid
            or record["probe_name"] != probe_name
        ):
            raise IBLNeuralCacheError(
                "acquisition official-loader receipt identity/binding is wrong"
            )
        run_ids.add(_require_text(record["run_id"], name=f"{name}.run_id"))
        if (
            record["revision"] != "2024-05-06"
            or isinstance(record["qc"], bool)
            or not isinstance(record["qc"], (int, float))
            or float(record["qc"]) != 1.0
        ):
            raise IBLNeuralCacheError(
                "acquisition official-loader receipt revision/QC is not pinned"
            )
        _require_utc_timestamp(record["timestamp_utc"], name=f"{name}.timestamp_utc")
        declared_record_sha = _require_hash(
            receipt["source_record_sha256"],
            name=f"{name}.source_record_sha256",
            pattern=_HEX_64,
        )
        if _canonical_sha256(record) != declared_record_sha:
            raise IBLNeuralCacheError(
                "acquisition official-loader source-record hash mismatch"
            )
        if event == "official_good_unit_loader_failed":
            for field in ("error_type", "error", "traceback"):
                _require_text(record[field], name=f"{name}.{field}")
        elif event == "official_good_unit_loader_passed":
            _require_json_int(
                record["unit_count"], name=f"{name}.unit_count", minimum=0
            )
            _require_json_int(
                record["spike_count"], name=f"{name}.spike_count", minimum=0
            )
        else:
            missing = record["missing_dataset_uuids"]
            if not isinstance(missing, list) or not missing:
                raise IBLNeuralCacheError(
                    "skipped official-loader receipt must retain missing UUIDs"
                )
            parsed_missing = [
                _require_uuid(item, name=f"{name}.missing_dataset_uuids[{item_index}]")
                for item_index, item in enumerate(missing)
            ]
            if len(set(parsed_missing)) != len(parsed_missing):
                raise IBLNeuralCacheError(
                    "skipped official-loader missing UUIDs must be unique"
                )
        events.append(event)
    if len(run_ids) != 1:
        raise IBLNeuralCacheError(
            "acquisition official-loader receipts must bind one acquisition run"
        )
    if qc_method != "passing_spikes_metrics_label_equivalence_v1":
        raise IBLNeuralCacheError(
            "complete compact sessions must use independently recomputed equivalence QC"
        )
    event_set = set(events)
    if event_set == {"official_good_unit_loader_passed"}:
        validation_status = "official_pinned_load_good_units_passed"
    elif event_set == {"official_good_unit_loader_failed"}:
        validation_status = "official_good_unit_loader_failed_retained"
    elif event_set == {"official_good_unit_loader_skipped_missing_inventory"}:
        validation_status = (
            "official_good_unit_loader_skipped_missing_inventory_retained"
        )
    else:
        validation_status = "official_good_unit_loader_mixed_nonpass_retained"
    if metadata.get("acquisition_validation_status") != validation_status:
        raise IBLNeuralCacheError(
            "acquisition validation status disagrees with raw loader receipts"
        )
    return next(iter(run_ids)), tuple(events)


def _validate_dataset_failure_receipts(
    metadata: Mapping[str, Any], *, acquisition_run_id: str
) -> tuple[Mapping[str, Any], ...]:
    wrappers = metadata.get("acquisition_dataset_failure_receipts")
    if not isinstance(wrappers, list):
        raise IBLNeuralCacheError(
            "acquisition dataset-failure receipts must be a JSON list"
        )
    record_keys = {
        "run_id",
        "timestamp_utc",
        "eid",
        "dataset_uuid",
        "category",
        "name",
        "collection",
        "human_revision",
        "event",
        "error_type",
        "error",
        "traceback",
    }
    records: list[Mapping[str, Any]] = []
    for index, raw_wrapper in enumerate(wrappers):
        name = f"acquisition_dataset_failure_receipts[{index}]"
        wrapper = _expect_exact_keys(
            raw_wrapper, {"source_record", "source_record_sha256"}, name=name
        )
        record = _expect_exact_keys(
            wrapper["source_record"], record_keys, name=f"{name}.source_record"
        )
        if (
            record["run_id"] != acquisition_run_id
            or record["eid"] != metadata.get("eid")
            or record["event"] != "dataset_failed"
        ):
            raise IBLNeuralCacheError(
                "acquisition dataset-failure receipt identity is inconsistent"
            )
        _require_utc_timestamp(
            record["timestamp_utc"], name=f"{name}.source_record.timestamp_utc"
        )
        _require_uuid(record["dataset_uuid"], name=f"{name}.source_record.dataset_uuid")
        for field in (
            "category",
            "name",
            "collection",
            "human_revision",
            "error_type",
            "error",
            "traceback",
        ):
            _require_text(record[field], name=f"{name}.source_record.{field}")
        declared = _require_hash(
            wrapper["source_record_sha256"],
            name=f"{name}.source_record_sha256",
            pattern=_HEX_64,
        )
        if _canonical_sha256(record) != declared:
            raise IBLNeuralCacheError(
                "acquisition dataset-failure source-record hash mismatch"
            )
        records.append(record)
    dataset_uuids = [str(record["dataset_uuid"]) for record in records]
    if dataset_uuids != sorted(dataset_uuids) or len(set(dataset_uuids)) != len(
        dataset_uuids
    ):
        raise IBLNeuralCacheError(
            "acquisition dataset-failure receipts must be uniquely UUID-sorted"
        )
    return tuple(records)


def _validate_required_input_bundle(
    metadata: Mapping[str, Any],
    *,
    camera_view: str,
    expected_datasets: Sequence[Mapping[str, Any]],
    dataset_failures: Sequence[Mapping[str, Any]],
) -> None:
    policy = _expect_exact_keys(
        metadata.get("auxiliary_dataset_failure_policy"),
        {
            "classification",
            "required_exact_names",
            "camera_rule",
            "selected_camera_view",
            "required_failure_count",
            "auxiliary_failure_count",
            "auxiliary_failure_dataset_uuids",
        },
        name="auxiliary_dataset_failure_policy",
    )
    failure_uuids = [str(record["dataset_uuid"]) for record in dataset_failures]
    if (
        policy["classification"] != "auxiliary_nonblocking_not_read"
        or policy["required_exact_names"] != list(_COMPACT_REQUIRED_DATASET_NAMES)
        or policy["camera_rule"] != _CAMERA_RULE
        or policy["selected_camera_view"] != camera_view
        or policy["required_failure_count"] != 0
        or policy["auxiliary_failure_count"] != len(dataset_failures)
        or policy["auxiliary_failure_dataset_uuids"] != failure_uuids
        or any(
            str(record["name"]) in _COMPACT_REQUIRED_DATASET_NAMES
            for record in dataset_failures
        )
    ):
        raise IBLNeuralCacheError(
            "required/auxiliary acquisition failure policy is inconsistent"
        )

    payload = _expect_exact_keys(
        metadata.get("compact_required_input_bundle"),
        {
            "protocol",
            "required_exact_names",
            "camera_rule",
            "selected_camera_view",
            "datasets",
        },
        name="compact_required_input_bundle",
    )
    if (
        payload["protocol"] != "compact_required_input_bundle_v1"
        or payload["required_exact_names"] != list(_COMPACT_REQUIRED_DATASET_NAMES)
        or payload["camera_rule"] != _CAMERA_RULE
        or payload["selected_camera_view"] != camera_view
    ):
        raise IBLNeuralCacheError("compact required-input policy is not pinned")
    datasets = payload["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise IBLNeuralCacheError("compact required-input bundle is empty")
    actual_identities: list[tuple[str, str, str]] = []
    actual_uuids: list[str] = []
    for index, raw_dataset in enumerate(datasets):
        name = f"compact_required_input_bundle.datasets[{index}]"
        dataset = _expect_exact_keys(
            raw_dataset, {"dataset_uuid", "name", "md5", "roles"}, name=name
        )
        identity = _validate_dataset_identifier(
            {key: dataset[key] for key in ("dataset_uuid", "name", "md5")},
            name=name,
        )
        roles = dataset["roles"]
        if not isinstance(roles, list) or not roles:
            raise IBLNeuralCacheError(f"{name}.roles must be a non-empty list")
        normalized_roles = [
            _require_text(role, name=f"{name}.roles[{role_index}]")
            for role_index, role in enumerate(roles)
        ]
        if normalized_roles != sorted(set(normalized_roles)):
            raise IBLNeuralCacheError(f"{name}.roles must be uniquely sorted")
        actual_identities.append(_dataset_identity(identity))
        actual_uuids.append(str(identity["dataset_uuid"]))
    if actual_uuids != sorted(actual_uuids) or len(set(actual_uuids)) != len(
        actual_uuids
    ):
        raise IBLNeuralCacheError(
            "compact required-input datasets must be uniquely UUID-sorted"
        )
    expected_identities = sorted(_dataset_identity(item) for item in expected_datasets)
    if sorted(actual_identities) != expected_identities:
        raise IBLNeuralCacheError(
            "compact required-input tracker differs from independently derived inputs"
        )
    if set(actual_uuids) & set(failure_uuids):
        raise IBLNeuralCacheError("a failed dataset is declared as a compact input")
    expected_sha = _require_hash(
        metadata.get("compact_required_input_bundle_sha256"),
        name="compact_required_input_bundle_sha256",
        pattern=_HEX_64,
    )
    if _canonical_sha256(payload) != expected_sha:
        raise IBLNeuralCacheError("compact required-input canonical hash mismatch")


def _validate_external_side_effect_payload(
    raw_evidence: object, *, expected_log_sha256: object
) -> str:
    evidence = _expect_exact_keys(
        raw_evidence,
        {
            "observed",
            "scope",
            "evidence_log_relative_path",
            "evidence_log_sha256",
            "observed_markers",
            "external_paths",
            "used_by_compact",
            "receipts_authoritative_for_qc",
            "policy",
        },
        name="acquisition_external_side_effect",
    )
    log_sha = _require_hash(
        evidence["evidence_log_sha256"],
        name="acquisition external-side-effect log SHA-256",
        pattern=_HEX_64,
    )
    expected_paths = list(_EXTERNAL_ATLAS_PATHS)
    if evidence != {
        "observed": True,
        "scope": "staging_external_atlas_cache",
        "evidence_log_relative_path": "logs/download.log",
        "evidence_log_sha256": log_sha,
        "observed_markers": [f"Downloading: {path}" for path in expected_paths],
        "external_paths": expected_paths,
        "used_by_compact": False,
        "receipts_authoritative_for_qc": False,
        "policy": "blocked_as_input",
    }:
        raise IBLNeuralCacheError(
            "acquisition external side effect is not explicitly blocked as input"
        )
    expected_log_sha = _require_hash(
        expected_log_sha256, name="download_log_sha256", pattern=_HEX_64
    )
    if expected_log_sha != log_sha:
        raise IBLNeuralCacheError(
            "download-log hash disagrees with external side-effect evidence"
        )
    return log_sha


def _validate_external_side_effect(metadata: Mapping[str, Any]) -> str:
    return _validate_external_side_effect_payload(
        metadata.get("acquisition_external_side_effect"),
        expected_log_sha256=metadata.get("download_log_sha256"),
    )


def _integer_csv_column(table: pd.DataFrame, column: str) -> np.ndarray:
    try:
        numeric = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise IBLNeuralCacheError(f"CSV {column} must be integer-valued") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise IBLNeuralCacheError(f"CSV {column} must be finite integer-valued")
    return numeric.astype(np.int64)


def _binary_csv_column(table: pd.DataFrame, column: str) -> np.ndarray:
    values: list[bool] = []
    for value in table[column].tolist():
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
            continue
        normalized = str(value).strip().lower()
        if normalized in {"1", "1.0", "true"}:
            values.append(True)
        elif normalized in {"0", "0.0", "false"}:
            values.append(False)
        else:
            raise IBLNeuralCacheError(f"CSV {column} must contain only binary values")
    return np.asarray(values, dtype=bool)


def _validate_trial_table(
    path: Path,
    *,
    view: str,
    trial_ids: np.ndarray,
    official_mask: np.ndarray,
    valid: np.ndarray,
    support: Sequence[tuple[float, float]],
    no_truncation: Mapping[str, Any],
) -> pd.DataFrame:
    try:
        table = pd.read_csv(path, float_precision="round_trip")
    except (OSError, pd.errors.ParserError) as error:
        raise IBLNeuralCacheError(
            f"{view} public trial CSV cannot be parsed"
        ) from error
    if tuple(str(column) for column in table.columns) != _PUBLIC_TRIAL_COLUMNS:
        raise IBLNeuralCacheError(
            f"{view} CSV must contain exactly the reviewed public columns; "
            "probability/context fields are forbidden"
        )
    ids = _integer_csv_column(table, "trial_id")
    if not np.array_equal(ids, trial_ids):
        raise IBLNeuralCacheError(f"{view} CSV trial IDs do not match NPZ")
    blocks = _integer_csv_column(table, "block_id")
    if not np.array_equal(blocks, trial_ids // 50):
        raise IBLNeuralCacheError(f"{view} CSV block_id is not trial_id // 50")
    csv_official = _binary_csv_column(table, "official_bwm_mask")
    csv_timing = _binary_csv_column(table, "timing_valid")
    side = _binary_csv_column(table, "stimulus_side")
    try:
        stimulus = pd.to_numeric(table["stimulus"], errors="raise").to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as error:
        raise IBLNeuralCacheError("CSV stimulus must be numeric") from error
    side_conflict = ((stimulus < 0.0) & ~side) | ((stimulus > 0.0) & side)
    if not np.isfinite(stimulus).all() or np.any(side_conflict):
        raise IBLNeuralCacheError(
            "CSV stimulus_side must use left=1/right=0; zero-contrast side is preserved"
        )
    try:
        reward = pd.to_numeric(table["reward"], errors="raise").to_numpy(dtype=float)
        reaction_time = pd.to_numeric(table["reaction_time"], errors="raise").to_numpy(
            dtype=float
        )
        stim_on = pd.to_numeric(table["stim_on"], errors="coerce").to_numpy(dtype=float)
        first_movement = pd.to_numeric(
            table["first_movement"], errors="coerce"
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise IBLNeuralCacheError(
            "CSV reward/reaction/event columns must be numeric"
        ) from error
    finite_timing = np.isfinite(stim_on) & np.isfinite(first_movement)
    if np.isinf(reward).any() or not np.allclose(
        reaction_time[finite_timing],
        (first_movement - stim_on)[finite_timing],
        rtol=0.0,
        atol=1e-12,
    ):
        raise IBLNeuralCacheError("CSV reward/reaction-time evidence is inconsistent")
    easy = np.abs(stimulus * 100.0) >= 50.0
    easy_count = int(easy.sum())
    easy_performance = float(np.mean(reward[easy] == 1.0)) if easy_count else np.nan
    if (
        len(table) <= 400
        or easy_count != no_truncation["easy_trial_count"]
        or not np.isclose(
            easy_performance,
            float(no_truncation["easy_performance"]),
            rtol=0.0,
            atol=1e-15,
        )
        or not easy_performance > 0.9
    ):
        raise IBLNeuralCacheError(
            "CSV does not reproduce the full-table no-truncation evidence"
        )
    if not np.array_equal(csv_official, official_mask):
        raise IBLNeuralCacheError(
            f"{view} CSV official BWM mask disagrees with evidence"
        )
    if not np.array_equal(csv_timing, valid):
        raise IBLNeuralCacheError(f"{view} CSV timing_valid disagrees with NPZ")
    events = stim_on if view == "stimulus_pre" else first_movement
    expected_valid = official_mask & np.isfinite(events)
    for start, stop in support:
        expected_valid &= events + _WINDOW_S[0] >= start
        expected_valid &= events + _WINDOW_S[1] <= stop
    if not np.array_equal(valid, expected_valid):
        raise IBLNeuralCacheError(
            f"{view} valid mask violates event/support/official-mask formula"
        )
    return table


def _validate_metadata_contract(
    metadata: object,
    *,
    row: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    root: Path,
) -> tuple[
    np.ndarray,
    tuple[tuple[float, float], ...],
    Mapping[str, Any],
    Mapping[str, str],
]:
    if not isinstance(metadata, dict):
        raise IBLNeuralCacheError("compact metadata must be a JSON object")
    if set(metadata) != _COMPLETE_METADATA_KEYS:
        raise IBLNeuralCacheError("compact complete-session metadata schema is not v2")
    required_metadata = {
        "candidate_rank": int(row["candidate_rank"]),
        "eid": str(row["eid"]),
        "animal_id": str(row["animal_id"]),
        "status": "complete",
        "error_type": "",
        "error": "",
        "source_manifest_sha256": str(row["source_manifest_sha256"]),
        "acquisition_bundle_sha256": str(row["acquisition_bundle_sha256"]),
        "bwm_repository_commit": str(row["bwm_repository_commit"]),
        "spike_sorting_revision": str(row["spike_sorting_revision"]),
        "unit_qc_threshold": 1.0,
        "unit_qc_applied": True,
        "acquisition_validation_status": str(row["acquisition_validation_status"]),
        "unit_qc_method": str(row["unit_qc_method"]),
        "unit_qc_equivalence_sha256": str(row["unit_qc_equivalence_sha256"]),
        "trial_dataset_uuid": str(row["trial_dataset_uuid"]),
        "trial_dataset_revision": str(row["trial_dataset_revision"]),
        "trial_dataset_md5": str(row["trial_dataset_md5"]),
        "official_bwm_mask_sha256": str(row["official_bwm_mask_sha256"]),
        "download_summary_sha256": str(row["download_summary_sha256"]),
        "session_status_bundle_sha256": str(row["session_status_bundle_sha256"]),
    }
    if any(
        key not in metadata or metadata[key] != value
        for key, value in required_metadata.items()
    ):
        raise IBLNeuralCacheError("compact metadata disagrees with its manifest row")
    if metadata.get("acquisition_approval_sha256") != _ACQUISITION_APPROVAL_SHA256:
        raise IBLNeuralCacheError(
            "compact metadata lacks the reviewed acquisition approval"
        )
    if (
        metadata.get("schema_version") != "ibl_exp14_compact_session_v2"
        or metadata.get("bwm_loader_sha256") != _BWM_LOADER_SHA256
        or metadata.get("bwm_loader_relative_path") != "provenance/bwm_loading.py"
        or metadata.get("postprocess_launcher_sha256") != _POSTPROCESS_LAUNCHER_SHA256
    ):
        raise IBLNeuralCacheError(
            "compact metadata is not bound to the reviewed producer inputs"
        )
    _require_utc_timestamp(metadata.get("created_at_utc"), name="created_at_utc")
    if metadata.get("acquisition_session_status") not in {
        "passed",
        "failed_retained",
    }:
        raise IBLNeuralCacheError("acquisition session status is invalid")
    if (
        metadata.get("bin_size_s") != _BIN_SIZE_S
        or metadata.get("window_s") != list(_WINDOW_S)
        or metadata.get("n_time_bins") != _N_TIME_BINS
        or metadata.get("window_semantics") != "half-open_[start,stop)"
        or metadata.get("time_axis_semantics") != "left_bin_edges"
    ):
        raise IBLNeuralCacheError(
            "compact metadata has an unreviewed bin/window contract"
        )
    n_trials = len(arrays["trial_ids"])
    n_units = len(arrays["unit_ids"])
    if metadata.get("trial_count") != n_trials or metadata.get("unit_count") != n_units:
        raise IBLNeuralCacheError(
            "compact metadata trial/unit counts disagree with NPZ"
        )
    cv = _expect_exact_keys(
        metadata.get("cv_block_policy"),
        {"method", "block_size_trials", "is_true_context"},
        name="cv_block_policy",
    )
    if (
        cv["method"] != "fixed_trial_id_chunks_v1"
        or cv["block_size_trials"] != 50
        or cv["is_true_context"] is not False
    ):
        raise IBLNeuralCacheError("compact CV blocks must be fixed non-context chunks")
    input_policy = _expect_exact_keys(
        metadata.get("postprocess_input_policy"),
        {
            "one_api_called",
            "network_access",
            "external_atlas_cache_used",
            "data_inputs",
        },
        name="postprocess_input_policy",
    )
    if input_policy != {
        "one_api_called": False,
        "network_access": False,
        "external_atlas_cache_used": False,
        "data_inputs": "review_bound_staging_files_only",
    }:
        raise IBLNeuralCacheError(
            "compact postprocessing must be offline and staging-bound"
        )
    _require_hash(
        metadata.get("acquisition_session_status_sha256"),
        name="acquisition_session_status_sha256",
        pattern=_HEX_64,
    )
    _require_hash(
        metadata.get("download_summary_sha256"),
        name="download_summary_sha256",
        pattern=_HEX_64,
    )
    _require_hash(
        metadata.get("session_status_bundle_sha256"),
        name="session_status_bundle_sha256",
        pattern=_HEX_64,
    )
    log_sha = _validate_external_side_effect(metadata)
    camera_view, motion_datasets = _validate_motion_proxy(metadata)
    wheel_datasets = _validate_wheel_proxy(metadata)
    region_mapping = _load_region_mapping(metadata, root=root)
    qc_sha = _require_hash(
        row["unit_qc_equivalence_sha256"],
        name="unit_qc_equivalence_sha256",
        pattern=_HEX_64,
    )
    probe_keys, qc_datasets = _validate_unit_qc(
        metadata,
        expected_sha256=qc_sha,
        unit_ids=arrays["unit_ids"],
        regions=arrays["regions"],
        region_mapping=region_mapping,
    )
    acquisition_run_id, loader_events = _validate_acquisition_loader_receipts(
        metadata, probe_keys=probe_keys, qc_method=str(row["unit_qc_method"])
    )
    dataset_failures = _validate_dataset_failure_receipts(
        metadata, acquisition_run_id=acquisition_run_id
    )
    expected_acquisition_status = (
        "passed"
        if not dataset_failures
        and set(loader_events) == {"official_good_unit_loader_passed"}
        else "failed_retained"
    )
    if metadata.get("acquisition_session_status") != expected_acquisition_status:
        raise IBLNeuralCacheError(
            "acquisition session status disagrees with retained raw receipts"
        )
    trial_dataset = {
        "dataset_uuid": str(row["trial_dataset_uuid"]),
        "name": "_ibl_trials.table.pqt",
        "md5": str(row["trial_dataset_md5"]),
    }
    expected_datasets = (
        *qc_datasets,
        trial_dataset,
        *wheel_datasets,
        *motion_datasets,
    )
    if len({_dataset_identity(item) for item in expected_datasets}) != len(
        expected_datasets
    ):
        raise IBLNeuralCacheError(
            "independently derived compact input identities are not unique"
        )
    _validate_required_input_bundle(
        metadata,
        camera_view=camera_view,
        expected_datasets=expected_datasets,
        dataset_failures=dataset_failures,
    )
    support = _validate_support(metadata, probe_keys)
    official_mask, no_truncation = _validate_official_mask(
        metadata, row=row, trial_ids=arrays["trial_ids"]
    )
    return (
        official_mask,
        support,
        no_truncation,
        {
            "download_log_sha256": log_sha,
            "postprocess_launcher_sha256": str(metadata["postprocess_launcher_sha256"]),
        },
    )


def _load_session(
    row: Mapping[str, object], root: Path
) -> tuple[PreparedIBLNeuralSession, Mapping[str, str]]:
    npz_path = _safe_file(root, row["npz_path"], row["npz_sha256"])
    stimulus_path = _safe_file(
        root, row["stimulus_trials_path"], row["stimulus_trials_sha256"]
    )
    movement_path = _safe_file(
        root, row["movement_trials_path"], row["movement_trials_sha256"]
    )
    metadata_path = _safe_file(root, row["metadata_path"], row["metadata_sha256"])
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IBLNeuralCacheError("compact metadata JSON cannot be parsed") from error
    arrays = _load_npz(npz_path)
    official_mask, support, no_truncation, producer_evidence = (
        _validate_metadata_contract(metadata, row=row, arrays=arrays, root=root)
    )
    stimulus = _validate_trial_table(
        stimulus_path,
        view="stimulus_pre",
        trial_ids=arrays["trial_ids"],
        official_mask=official_mask,
        valid=arrays["stimulus_pre_valid"],
        support=support,
        no_truncation=no_truncation,
    )
    movement = _validate_trial_table(
        movement_path,
        view="movement_pre",
        trial_ids=arrays["trial_ids"],
        official_mask=official_mask,
        valid=arrays["movement_pre_valid"],
        support=support,
        no_truncation=no_truncation,
    )
    return (
        PreparedIBLNeuralSession(
            eid=str(row["eid"]),
            animal_id=str(row["animal_id"]),
            count_views={
                "stimulus_pre": arrays["stimulus_pre_counts"],
                "movement_pre": arrays["movement_pre_counts"],
            },
            valid_masks={
                "stimulus_pre": arrays["stimulus_pre_valid"],
                "movement_pre": arrays["movement_pre_valid"],
            },
            time_axes={
                "stimulus_pre": arrays["stimulus_pre_time"],
                "movement_pre": arrays["movement_pre_time"],
            },
            unit_ids=arrays["unit_ids"],
            regions=arrays["regions"],
            view_trial_tables={"stimulus_pre": stimulus, "movement_pre": movement},
            current_trial_ids=arrays["trial_ids"],
        ),
        producer_evidence,
    )


def _validate_compact_bundle(
    root: Path,
    *,
    frame: pd.DataFrame,
    manifest_sha256: str,
    expected_compact_bundle_sha256: str,
    expected_source_manifest_sha256: str,
    expected_acquisition_bundle_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    bundle_path = _safe_file(
        root,
        "compact_bundle.json",
        expected_compact_bundle_sha256,
    )
    try:
        raw_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IBLNeuralCacheError("compact bundle JSON cannot be parsed") from error
    bundle = _expect_exact_keys(raw_bundle, _COMPACT_BUNDLE_KEYS, name="compact_bundle")
    _require_utc_timestamp(
        bundle["created_at_utc"], name="compact_bundle.created_at_utc"
    )
    fixed = {
        "schema_version": "ibl_exp14_compact_bundle_v2",
        "compact_manifest_sha256": manifest_sha256,
        "source_manifest_sha256": expected_source_manifest_sha256,
        "acquisition_bundle_sha256": expected_acquisition_bundle_sha256,
        "acquisition_approval_sha256": _ACQUISITION_APPROVAL_SHA256,
        "postprocess_script_sha256": _POSTPROCESS_SCRIPT_SHA256,
        "postprocess_launcher_sha256": _POSTPROCESS_LAUNCHER_SHA256,
        "compact_schema_sha256": _COMPACT_SCHEMA_SHA256,
        "bwm_loader_sha256": _BWM_LOADER_SHA256,
        "bwm_loader_relative_path": "provenance/bwm_loading.py",
        "region_mapping_path": "provenance/iblatlas_allen_structure_tree.csv",
        "region_mapping_sha256": _REGION_MAPPING_SHA256,
        "region_mapping_provenance_path": ("provenance/region_mapping_provenance.json"),
        "region_mapping_provenance_sha256": _REGION_MAPPING_PROVENANCE_SHA256,
        "iblatlas_version": "1.1.0",
        "iblatlas_source_commit": ("unavailable_pypi_distribution_direct_url_absent"),
        "iblatlas_regions_source_sha256": _REGIONS_SOURCE_SHA256,
    }
    if any(bundle.get(key) != value for key, value in fixed.items()):
        raise IBLNeuralCacheError(
            "compact bundle is not bound to the reviewed producer/schema/provenance"
        )
    _safe_file(root, bundle["bwm_loader_relative_path"], bundle["bwm_loader_sha256"])
    _safe_file(root, bundle["region_mapping_path"], bundle["region_mapping_sha256"])
    _safe_file(
        root,
        bundle["region_mapping_provenance_path"],
        bundle["region_mapping_provenance_sha256"],
    )
    summary_sha = _require_hash(
        bundle["download_summary_sha256"],
        name="compact_bundle.download_summary_sha256",
        pattern=_HEX_64,
    )
    status_sha = _require_hash(
        bundle["session_status_bundle_sha256"],
        name="compact_bundle.session_status_bundle_sha256",
        pattern=_HEX_64,
    )
    if set(frame["download_summary_sha256"].astype(str)) != {summary_sha} or set(
        frame["session_status_bundle_sha256"].astype(str)
    ) != {status_sha}:
        raise IBLNeuralCacheError(
            "compact bundle dynamic acquisition hashes disagree with the manifest"
        )
    _validate_external_side_effect_payload(
        bundle["acquisition_external_side_effect"],
        expected_log_sha256=bundle["download_log_sha256"],
    )
    complete = frame.loc[frame["status"].astype(str) == "complete"]
    failed = frame.loc[frame["status"].astype(str) == "failed"]
    if (
        _require_json_int(
            bundle["complete_sessions"], name="compact_bundle.complete_sessions"
        )
        != len(complete)
        or _require_json_int(
            bundle["failed_sessions"], name="compact_bundle.failed_sessions"
        )
        != len(failed)
        or len(complete) + len(failed) != len(frame)
    ):
        raise IBLNeuralCacheError(
            "compact bundle session counts disagree with manifest"
        )
    artifact_hashes = bundle["artifact_sha256"]
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(
        complete["eid"].astype(str)
    ):
        raise IBLNeuralCacheError(
            "compact bundle artifact map does not match complete sessions"
        )
    artifact_columns = {
        "npz": "npz_sha256",
        "stimulus": "stimulus_trials_sha256",
        "movement": "movement_trials_sha256",
        "metadata": "metadata_sha256",
    }
    for row in complete.to_dict(orient="records"):
        eid = str(row["eid"])
        artifacts = _expect_exact_keys(
            artifact_hashes[eid], set(artifact_columns), name=f"artifact_sha256.{eid}"
        )
        for artifact, column in artifact_columns.items():
            declared = _require_hash(
                artifacts[artifact],
                name=f"artifact_sha256.{eid}.{artifact}",
                pattern=_HEX_64,
            )
            if declared != str(row[column]):
                raise IBLNeuralCacheError(
                    "compact bundle artifact hash disagrees with manifest"
                )
    return bundle, expected_compact_bundle_sha256


def load_compact_neural_cohort(
    manifest_csv: str | Path,
    *,
    expected_source_manifest_sha256: str,
    expected_acquisition_bundle_sha256: str,
    expected_bwm_repository_commit: str,
    expected_compact_manifest_sha256: str,
    expected_compact_bundle_sha256: str,
    expected_sessions: int = 20,
    minimum_animals: int = 5,
) -> CompactNeuralCohort:
    """Load only hash-bound compact files; never import ONE or access a network."""

    expected_source_manifest_sha256 = _require_hash(
        expected_source_manifest_sha256,
        name="expected source manifest SHA-256",
        pattern=_HEX_64,
    )
    expected_acquisition_bundle_sha256 = _require_hash(
        expected_acquisition_bundle_sha256,
        name="expected acquisition bundle SHA-256",
        pattern=_HEX_64,
    )
    expected_bwm_repository_commit = _require_hash(
        expected_bwm_repository_commit,
        name="expected BWM repository commit",
        pattern=_HEX_40,
    )
    expected_compact_manifest_sha256 = _require_hash(
        expected_compact_manifest_sha256,
        name="expected compact manifest SHA-256",
        pattern=_HEX_64,
    )
    expected_compact_bundle_sha256 = _require_hash(
        expected_compact_bundle_sha256,
        name="expected compact bundle SHA-256",
        pattern=_HEX_64,
    )
    if expected_sessions < 1 or minimum_animals < 1:
        raise IBLNeuralCacheError("cohort count requirements must be positive")
    path = Path(manifest_csv).resolve(strict=True)
    manifest_sha256 = _sha256(path)
    if manifest_sha256 != expected_compact_manifest_sha256:
        raise IBLNeuralCacheError(
            "compact manifest does not match the reviewed root hash"
        )
    root = path.parent.resolve(strict=True)
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as error:
        raise IBLNeuralCacheError("compact manifest cannot be parsed") from error
    if tuple(str(column) for column in frame.columns) != _MANIFEST_COLUMNS:
        raise IBLNeuralCacheError(
            "compact manifest must contain exactly the reviewed ordered columns"
        )
    try:
        rank_values = pd.to_numeric(frame["candidate_rank"], errors="raise").to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as error:
        raise IBLNeuralCacheError("candidate_rank must be integer-valued") from error
    if (
        not np.isfinite(rank_values).all()
        or not np.equal(rank_values, np.floor(rank_values)).all()
    ):
        raise IBLNeuralCacheError("candidate_rank must be integer-valued")
    ranks = pd.Series(rank_values.astype(int), index=frame.index)
    if len(frame) != expected_sessions or ranks.duplicated().any() or (ranks < 0).any():
        raise IBLNeuralCacheError(
            "compact manifest must retain the reviewed cohort row count and unique "
            "non-negative source ranks"
        )
    frame = frame.assign(candidate_rank=ranks).sort_values(
        "candidate_rank", kind="mergesort"
    )
    eids = frame["eid"].astype(str)
    animals_in_manifest = frame["animal_id"].astype(str)
    if eids.duplicated().any() or (eids.str.strip().str.len() == 0).any():
        raise IBLNeuralCacheError("compact manifest contains duplicate EIDs")
    if (animals_in_manifest.str.strip().str.len() == 0).any():
        raise IBLNeuralCacheError("compact manifest contains an empty animal ID")
    source_hashes = set(frame["source_manifest_sha256"].astype(str))
    bwm_commits = set(frame["bwm_repository_commit"].astype(str))
    bundle_hashes = set(frame["acquisition_bundle_sha256"].astype(str))
    if source_hashes != {expected_source_manifest_sha256}:
        raise IBLNeuralCacheError("compact cohort source-manifest binding is wrong")
    if bwm_commits != {expected_bwm_repository_commit}:
        raise IBLNeuralCacheError("compact cohort BWM commit binding is wrong")
    if bundle_hashes != {expected_acquisition_bundle_sha256}:
        raise IBLNeuralCacheError("compact cohort acquisition-bundle binding is wrong")
    if (
        len(set(frame["download_summary_sha256"].astype(str))) != 1
        or len(set(frame["session_status_bundle_sha256"].astype(str))) != 1
    ):
        raise IBLNeuralCacheError(
            "compact cohort must share one acquisition summary/status bundle"
        )
    _, compact_bundle_sha256 = _validate_compact_bundle(
        root,
        frame=frame,
        manifest_sha256=manifest_sha256,
        expected_compact_bundle_sha256=expected_compact_bundle_sha256,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_acquisition_bundle_sha256=expected_acquisition_bundle_sha256,
    )

    dispositions = []
    sessions = []
    producer_evidence_rows: list[Mapping[str, str]] = []
    for row in frame.to_dict(orient="records"):
        status = str(row["status"])
        qc_applied = _strict_bool(row["unit_qc_applied"], name="unit_qc_applied")
        try:
            qc_threshold = float(row["unit_qc_threshold"])
        except (TypeError, ValueError) as error:
            raise IBLNeuralCacheError("unit_qc_threshold must be numeric") from error
        _require_hash(
            row["download_summary_sha256"],
            name="download_summary_sha256",
            pattern=_HEX_64,
        )
        _require_hash(
            row["session_status_bundle_sha256"],
            name="session_status_bundle_sha256",
            pattern=_HEX_64,
        )
        expected_region_manifest = {
            "region_mapping_path": "provenance/iblatlas_allen_structure_tree.csv",
            "region_mapping_sha256": _REGION_MAPPING_SHA256,
            "region_mapping_provenance_path": (
                "provenance/region_mapping_provenance.json"
            ),
            "region_mapping_provenance_sha256": (_REGION_MAPPING_PROVENANCE_SHA256),
            "iblatlas_version": "1.1.0",
            "iblatlas_source_commit": (
                "unavailable_pypi_distribution_direct_url_absent"
            ),
            "iblatlas_regions_source_sha256": _REGIONS_SOURCE_SHA256,
        }
        if any(
            str(row[key]) != value for key, value in expected_region_manifest.items()
        ):
            raise IBLNeuralCacheError(
                "compact manifest does not bind the reviewed region resource"
            )
        disposition = CompactSessionDisposition(
            candidate_rank=int(row["candidate_rank"]),
            eid=str(row["eid"]),
            animal_id=str(row["animal_id"]),
            status=status,
            error_type=str(row["error_type"]),
            error=str(row["error"]),
            source_manifest_sha256=str(row["source_manifest_sha256"]),
            acquisition_bundle_sha256=str(row["acquisition_bundle_sha256"]),
            bwm_repository_commit=str(row["bwm_repository_commit"]),
            spike_sorting_revision=str(row["spike_sorting_revision"]),
            unit_qc_threshold=qc_threshold,
            unit_qc_applied=qc_applied,
            acquisition_validation_status=str(row["acquisition_validation_status"]),
            unit_qc_method=str(row["unit_qc_method"]),
            unit_qc_equivalence_sha256=str(row["unit_qc_equivalence_sha256"]),
            trial_dataset_uuid=str(row["trial_dataset_uuid"]),
            trial_dataset_revision=str(row["trial_dataset_revision"]),
            trial_dataset_md5=str(row["trial_dataset_md5"]),
            official_bwm_mask_sha256=str(row["official_bwm_mask_sha256"]),
            download_summary_sha256=str(row["download_summary_sha256"]),
            session_status_bundle_sha256=str(row["session_status_bundle_sha256"]),
            region_mapping_path=str(row["region_mapping_path"]),
            region_mapping_sha256=str(row["region_mapping_sha256"]),
            region_mapping_provenance_path=str(row["region_mapping_provenance_path"]),
            region_mapping_provenance_sha256=str(
                row["region_mapping_provenance_sha256"]
            ),
            iblatlas_version=str(row["iblatlas_version"]),
            iblatlas_source_commit=str(row["iblatlas_source_commit"]),
            iblatlas_regions_source_sha256=str(row["iblatlas_regions_source_sha256"]),
        )
        dispositions.append(disposition)
        if status == "complete":
            if (
                disposition.error_type
                or disposition.error
                or not qc_applied
                or qc_threshold != 1.0
                or disposition.spike_sorting_revision != "2024-05-06"
            ):
                raise IBLNeuralCacheError("completed session violates frozen unit QC")
            if disposition.unit_qc_method != (
                "passing_spikes_metrics_label_equivalence_v1"
            ):
                raise IBLNeuralCacheError(
                    "completed session has an unreviewed QC method"
                )
            if (
                disposition.acquisition_validation_status
                not in _ACQUISITION_VALIDATION_STATUSES
            ):
                raise IBLNeuralCacheError(
                    "completed session acquisition-validation status is invalid"
                )
            _require_hash(
                disposition.unit_qc_equivalence_sha256,
                name="unit_qc_equivalence_sha256",
                pattern=_HEX_64,
            )
            _require_uuid(disposition.trial_dataset_uuid, name="trial_dataset_uuid")
            _require_text(
                disposition.trial_dataset_revision, name="trial_dataset_revision"
            )
            _require_hash(
                disposition.trial_dataset_md5,
                name="trial_dataset_md5",
                pattern=_HEX_32,
            )
            _require_hash(
                disposition.official_bwm_mask_sha256,
                name="official_bwm_mask_sha256",
                pattern=_HEX_64,
            )
            session, producer_evidence = _load_session(row, root)
            sessions.append(session)
            producer_evidence_rows.append(producer_evidence)
        elif status == "failed":
            if not disposition.error_type or not disposition.error:
                raise IBLNeuralCacheError(
                    "failed session must retain its failure reason"
                )
        else:
            raise IBLNeuralCacheError("compact status must be complete or failed")
    animals = {session.animal_id for session in sessions}
    if producer_evidence_rows:
        if {row["postprocess_launcher_sha256"] for row in producer_evidence_rows} != {
            _POSTPROCESS_LAUNCHER_SHA256
        } or len({row["download_log_sha256"] for row in producer_evidence_rows}) != 1:
            raise IBLNeuralCacheError(
                "complete sessions do not share the reviewed launcher/download log"
            )
    if len(sessions) > expected_sessions or len(animals) < min(
        minimum_animals, len(sessions)
    ):
        raise IBLNeuralCacheError(
            "compact complete-session/animal counts are inconsistent"
        )
    return CompactNeuralCohort(
        dispositions=tuple(dispositions),
        sessions=tuple(sessions),
        compact_manifest_sha256=manifest_sha256,
        compact_bundle_sha256=compact_bundle_sha256,
    )


__all__ = [
    "CompactNeuralCohort",
    "CompactSessionDisposition",
    "IBLNeuralCacheError",
    "load_compact_neural_cohort",
]
