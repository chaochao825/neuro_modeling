from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data.ibl_neural_cache as neural_cache
from src.data.ibl_neural_cache import (
    IBLNeuralCacheError,
    load_compact_neural_cohort,
)


SOURCE_SHA = "a" * 64
BUNDLE_SHA = "b" * 64
BWM_COMMIT = "c" * 40
APPROVAL_SHA = "64e0b1ead4175b2007b0c4aa01524e9e9db4beebde63f016cf46e245f0da4287"
DOWNLOAD_SHA = "e" * 64
STATUS_SHA = "f" * 64
DOWNLOAD_LOG_SHA = "d" * 64
TRIAL_REVISION = "2024-05-06"


def test_consumer_binds_exact_frozen_postprocess_launcher() -> None:
    assert neural_cache._POSTPROCESS_LAUNCHER_SHA256 == (
        "fe8a30ab705d7e0c8474e574fcae36ce68f510d2d53de81003edd3a3f05f2837"
    )
    assert neural_cache._POSTPROCESS_SCRIPT_SHA256 == (
        "951a516ae7eb2aa024f37e1890d6afd34dd19378905b90744dfadb32e7a76a17"
    )
    assert neural_cache._COMPACT_SCHEMA_SHA256 == (
        "f29c5506be93393499b90535c401bec1c82e0754737733897716cd4e2fade39d"
    )


@pytest.fixture(autouse=True)
def _reviewed_region_mapping_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = tmp_path / "provenance"
    provenance.mkdir(exist_ok=True)
    mapping_path = provenance / "iblatlas_allen_structure_tree.csv"
    rows = []
    for region_id, acronym in ((0, "void"), (500, "MOs"), (600, "MD")):
        row = {column: 0 for column in neural_cache._REGION_MAPPING_COLUMNS}
        row.update(id=region_id, name=acronym, acronym=acronym, safe_name=acronym)
        rows.append(row)
    pd.DataFrame(rows, columns=neural_cache._REGION_MAPPING_COLUMNS).to_csv(
        mapping_path, index=False
    )
    provenance_path = provenance / "region_mapping_provenance.json"
    provenance_path.write_text(
        json.dumps({"fixture": "reviewed frozen mapping"}), encoding="utf-8"
    )
    bwm_loader_path = provenance / "bwm_loading.py"
    bwm_loader_path.write_text("# reviewed BWM loader fixture\n", encoding="utf-8")
    monkeypatch.setattr(neural_cache, "_REGION_MAPPING_SHA256", _sha(mapping_path))
    monkeypatch.setattr(
        neural_cache, "_REGION_MAPPING_PROVENANCE_SHA256", _sha(provenance_path)
    )
    monkeypatch.setattr(neural_cache, "_BWM_LOADER_SHA256", _sha(bwm_loader_path))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _uuid(value: int) -> str:
    return f"00000000-0000-0000-0000-{value:012d}"


def _unit_qc_payload(rank: int) -> tuple[dict[str, object], str]:
    pid = _uuid(100 + rank)
    probe = {
        "pid": pid,
        "probe_name": "probe00",
        "passing_cluster_ids": [0, 1],
        "metrics_label_ge_1_cluster_ids": [0, 1],
        "cluster_channel_ccf_acronym": [
            [0, 4, 500, "MOs"],
            [1, 8, 600, "MD"],
        ],
        "input_datasets": [
            {
                "dataset_uuid": _uuid(200 + rank * 10 + index),
                "name": name,
                "md5": f"{index + 1:x}" * 32,
            }
            for index, name in enumerate(
                (
                    "passingSpikes.table.pqt",
                    "clusters.metrics.pqt",
                    "clusters.channels.npy",
                    "channels.brainLocationIds_ccf_2017.npy",
                )
            )
        ],
    }
    return {
        "sorting_rule": "probe_name_then_pid_then_cluster_id_v1",
        "probes": [probe],
    }, pid


def _mask_evidence(trial_uuid: str, n_trials: int) -> dict[str, object]:
    mask = np.resize(np.asarray([0, 1, 1, 1, 1, 0], dtype=int), n_trials).tolist()
    trial_ids = list(range(n_trials))
    stimulus = np.resize(np.asarray([-1.0, 1.0, 0.0, -0.5, 0.5, 0.0]), n_trials)
    easy_count = int((np.abs(stimulus * 100.0) >= 50.0).sum())
    return {
        "protocol": "audited_pinned_bwm_formula_min_rt_0p08_max_rt_2_default_nan_v2",
        "bwm_loader_sha256": neural_cache._BWM_LOADER_SHA256,
        "runtime_brainwidemap_imported": False,
        "no_truncation_precondition": {
            "n_trials": n_trials,
            "n_trials_strictly_greater_than_400": True,
            "easy_trial_definition": "abs(signed_contrast_percent)>=50",
            "easy_trial_count": easy_count,
            "easy_performance": 1.0,
            "easy_performance_strictly_greater_than_0p9": True,
            "passed": True,
        },
        "trial_ids": trial_ids,
        "mask": mask,
        "selected_trial_ids": [
            trial_id for trial_id, keep in zip(trial_ids, mask, strict=True) if keep
        ],
        "trial_dataset": {
            "dataset_uuid": trial_uuid,
            "revision": TRIAL_REVISION,
            "md5": "9" * 32,
        },
    }


def _trial_table(n_trials: int, valid: np.ndarray) -> pd.DataFrame:
    stimulus = np.resize(np.asarray([-1.0, 1.0, 0.0, -0.5, 0.5, 0.0]), n_trials)
    return pd.DataFrame(
        {
            "trial_id": np.arange(n_trials, dtype=np.int64),
            "stimulus": stimulus,
            # Existing exp11/HMM convention: left=1, right=0.  Zero-contrast
            # trials preserve the original finite side instead of inferring it.
            "stimulus_side": np.resize(np.asarray([1, 0, 1, 1, 0, 0]), n_trials),
            "choice": np.where(np.arange(n_trials) % 2, 1, -1),
            "wheel": 0.2,
            "reward": 1,
            "reaction_time": 0.2,
            "stim_on": 1.0 + np.arange(n_trials),
            "first_movement": 1.2 + np.arange(n_trials),
            "timing_valid": valid,
            "official_bwm_mask": valid,
            "block_id": np.arange(n_trials, dtype=np.int64) // 50,
            "motion_energy_proxy": 0.1,
        }
    )


def _complete_row(root: Path, rank: int) -> dict[str, object]:
    eid = f"eid-{rank}"
    n_trials, n_units = 401, 2
    unit_qc, pid = _unit_qc_payload(rank)
    qc_sha = _canonical_sha(unit_qc)
    trial_uuid = _uuid(300 + rank)
    mask_evidence = _mask_evidence(trial_uuid, n_trials)
    mask_sha = _canonical_sha(mask_evidence)
    official_mask = np.asarray(mask_evidence["mask"], dtype=bool)
    acquisition_source_record = {
        "run_id": "reviewed-acquisition-run",
        "timestamp_utc": "2026-07-12T00:00:00+00:00",
        "eid": eid,
        "pid": pid,
        "probe_name": "probe00",
        "revision": "2024-05-06",
        "qc": 1.0,
        "event": "official_good_unit_loader_failed",
        "error_type": "AttributeError",
        "error": "offline loader failed",
        "traceback": "retained acquisition review trace",
    }
    acquisition_receipt = {
        "source_record": acquisition_source_record,
        "source_record_sha256": _canonical_sha(acquisition_source_record),
    }
    stimulus_counts = np.ones((n_trials, 25, n_units), dtype=np.int32)
    movement_counts = np.full((n_trials, 25, n_units), 2, dtype=np.int32)
    stimulus_counts[~official_mask] = 0
    movement_counts[~official_mask] = 0
    npz = root / f"{eid}.npz"
    np.savez_compressed(
        npz,
        stimulus_pre_counts=stimulus_counts,
        movement_pre_counts=movement_counts,
        stimulus_pre_valid=official_mask,
        movement_pre_valid=official_mask,
        stimulus_pre_time=np.arange(-0.5, 0.0, 0.02, dtype=np.float64),
        movement_pre_time=np.arange(-0.5, 0.0, 0.02, dtype=np.float64),
        unit_ids=np.asarray([f"{pid}:0", f"{pid}:1"]),
        regions=np.asarray(["MOs", "MD"]),
        trial_ids=np.arange(n_trials, dtype=np.int64),
    )
    stimulus = root / f"{eid}-stimulus.csv"
    movement = root / f"{eid}-movement.csv"
    _trial_table(n_trials, official_mask).to_csv(stimulus, index=False)
    _trial_table(n_trials, official_mask).assign(motion_energy_proxy=1.1).to_csv(
        movement, index=False
    )
    motion_time = {
        "dataset_uuid": _uuid(400 + rank * 10),
        "name": "_ibl_leftCamera.times.npy",
        "md5": "a" * 32,
    }
    motion_energy = {
        "dataset_uuid": _uuid(401 + rank * 10),
        "name": "leftCamera.ROIMotionEnergy.npy",
        "md5": "b" * 32,
    }
    wheel_time = {
        "dataset_uuid": _uuid(402 + rank * 10),
        "name": "_ibl_wheel.timestamps.npy",
        "md5": "c" * 32,
    }
    wheel_position = {
        "dataset_uuid": _uuid(403 + rank * 10),
        "name": "_ibl_wheel.position.npy",
        "md5": "d" * 32,
    }
    trial_dataset = {
        "dataset_uuid": trial_uuid,
        "name": "_ibl_trials.table.pqt",
        "md5": "9" * 32,
    }
    qc_datasets = unit_qc["probes"][0]["input_datasets"]
    required_datasets = []
    for dataset, roles in (
        (trial_dataset, ["session:trials_table"]),
        (wheel_time, ["session:wheel_timestamps"]),
        (wheel_position, ["session:wheel_position"]),
        (
            motion_time,
            ["movement_pre:camera_times", "stimulus_pre:camera_times"],
        ),
        (
            motion_energy,
            ["movement_pre:motion_energy", "stimulus_pre:motion_energy"],
        ),
        *(
            (
                dataset,
                [f"probe:{pid}:{dataset['name']}"],
            )
            for dataset in qc_datasets
        ),
    ):
        required_datasets.append({**dataset, "roles": sorted(roles)})
    required_datasets.sort(key=lambda item: item["dataset_uuid"])
    required_input_bundle = {
        "protocol": "compact_required_input_bundle_v1",
        "required_exact_names": list(neural_cache._COMPACT_REQUIRED_DATASET_NAMES),
        "camera_rule": neural_cache._CAMERA_RULE,
        "selected_camera_view": "left",
        "datasets": required_datasets,
    }
    external_paths = list(neural_cache._EXTERNAL_ATLAS_PATHS)
    external_side_effect = {
        "observed": True,
        "scope": "staging_external_atlas_cache",
        "evidence_log_relative_path": "logs/download.log",
        "evidence_log_sha256": DOWNLOAD_LOG_SHA,
        "observed_markers": [f"Downloading: {path}" for path in external_paths],
        "external_paths": external_paths,
        "used_by_compact": False,
        "receipts_authoritative_for_qc": False,
        "policy": "blocked_as_input",
    }
    metadata = root / f"{eid}.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "ibl_exp14_compact_session_v2",
                "candidate_rank": rank,
                "eid": eid,
                "animal_id": f"animal-{rank}",
                "status": "complete",
                "error_type": "",
                "error": "",
                "source_manifest_sha256": SOURCE_SHA,
                "acquisition_bundle_sha256": BUNDLE_SHA,
                "acquisition_approval_sha256": APPROVAL_SHA,
                "bwm_repository_commit": BWM_COMMIT,
                "bwm_loader_sha256": neural_cache._BWM_LOADER_SHA256,
                "bwm_loader_relative_path": "provenance/bwm_loading.py",
                "postprocess_launcher_sha256": neural_cache._POSTPROCESS_LAUNCHER_SHA256,
                "spike_sorting_revision": "2024-05-06",
                "unit_qc_threshold": 1.0,
                "unit_qc_applied": True,
                "acquisition_validation_status": "official_good_unit_loader_failed_retained",
                "unit_qc_method": "passing_spikes_metrics_label_equivalence_v1",
                "unit_qc_equivalence_sha256": qc_sha,
                "unit_qc_equivalence": unit_qc,
                "acquisition_official_loader_receipts": [acquisition_receipt],
                "acquisition_dataset_failure_receipts": [],
                "auxiliary_dataset_failure_policy": {
                    "classification": "auxiliary_nonblocking_not_read",
                    "required_exact_names": list(
                        neural_cache._COMPACT_REQUIRED_DATASET_NAMES
                    ),
                    "camera_rule": neural_cache._CAMERA_RULE,
                    "selected_camera_view": "left",
                    "required_failure_count": 0,
                    "auxiliary_failure_count": 0,
                    "auxiliary_failure_dataset_uuids": [],
                },
                "compact_required_input_bundle": required_input_bundle,
                "compact_required_input_bundle_sha256": _canonical_sha(
                    required_input_bundle
                ),
                "acquisition_session_status_sha256": "8" * 64,
                "acquisition_external_side_effect": external_side_effect,
                "postprocess_input_policy": {
                    "one_api_called": False,
                    "network_access": False,
                    "external_atlas_cache_used": False,
                    "data_inputs": "review_bound_staging_files_only",
                },
                "region_mapping_input": {
                    "schema_version": "ibl_exp14_region_mapping_input_v1",
                    "mapping_relative_path": "provenance/iblatlas_allen_structure_tree.csv",
                    "mapping_sha256": neural_cache._REGION_MAPPING_SHA256,
                    "provenance_relative_path": "provenance/region_mapping_provenance.json",
                    "provenance_sha256": neural_cache._REGION_MAPPING_PROVENANCE_SHA256,
                    "source_package": "iblatlas",
                    "source_version": "1.1.0",
                    "source_commit": None,
                    "source_commit_status": "unavailable_pypi_distribution_direct_url_absent",
                    "regions_source_sha256": neural_cache._REGIONS_SOURCE_SHA256,
                },
                "trial_dataset_uuid": trial_uuid,
                "trial_dataset_revision": TRIAL_REVISION,
                "trial_dataset_md5": "9" * 32,
                "official_bwm_mask_sha256": mask_sha,
                "official_bwm_mask_evidence": mask_evidence,
                "download_summary_sha256": DOWNLOAD_SHA,
                "download_log_sha256": DOWNLOAD_LOG_SHA,
                "session_status_bundle_sha256": STATUS_SHA,
                "bin_size_s": 0.02,
                "window_s": [-0.5, 0.0],
                "n_time_bins": 25,
                "window_semantics": "half-open_[start,stop)",
                "time_axis_semantics": "left_bin_edges",
                "trial_count": n_trials,
                "official_bwm_mask_count": int(official_mask.sum()),
                "unit_count": n_units,
                "motion_energy_proxy": {
                    view: {
                        "proxy": "camera_roi_motion_energy_mean",
                        "camera_view": "left",
                        "window_s": [-0.5, 0.0],
                        "window_semantics": "half-open_[event+start,event+stop)",
                        "aggregation": "finite_sample_arithmetic_mean",
                        "time_dataset": motion_time,
                        "energy_dataset": motion_energy,
                    }
                    for view in ("stimulus_pre", "movement_pre")
                },
                "wheel_displacement": {
                    "proxy": "total_absolute_wheel_displacement",
                    "start_event": "stimOn_times",
                    "stop_event": "response_times",
                    "window_semantics": "half-open_[start_event,stop_event)",
                    "aggregation": "sum_absolute_first_differences",
                    "time_dataset": wheel_time,
                    "position_dataset": wheel_position,
                },
                "cv_block_policy": {
                    "method": "fixed_trial_id_chunks_v1",
                    "block_size_trials": 50,
                    "is_true_context": False,
                },
                "observed_good_spike_support_s": [
                    {
                        "pid": pid,
                        "probe_name": "probe00",
                        "start": 0.0,
                        "stop": 1000.0,
                        "semantics": "conservative_observed_good_spike_support_not_recording_interval",
                    }
                ],
                "acquisition_session_status": "failed_retained",
                "created_at_utc": "2026-07-12T03:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return {
        "candidate_rank": rank,
        "eid": eid,
        "animal_id": f"animal-{rank}",
        "status": "complete",
        "error_type": "",
        "error": "",
        "npz_path": npz.name,
        "npz_sha256": _sha(npz),
        "stimulus_trials_path": stimulus.name,
        "stimulus_trials_sha256": _sha(stimulus),
        "movement_trials_path": movement.name,
        "movement_trials_sha256": _sha(movement),
        "metadata_path": metadata.name,
        "metadata_sha256": _sha(metadata),
        "source_manifest_sha256": SOURCE_SHA,
        "acquisition_bundle_sha256": BUNDLE_SHA,
        "bwm_repository_commit": BWM_COMMIT,
        "spike_sorting_revision": "2024-05-06",
        "unit_qc_threshold": 1.0,
        "unit_qc_applied": True,
        "acquisition_validation_status": "official_good_unit_loader_failed_retained",
        "unit_qc_method": "passing_spikes_metrics_label_equivalence_v1",
        "unit_qc_equivalence_sha256": qc_sha,
        "trial_dataset_uuid": trial_uuid,
        "trial_dataset_revision": TRIAL_REVISION,
        "trial_dataset_md5": "9" * 32,
        "official_bwm_mask_sha256": mask_sha,
        "download_summary_sha256": DOWNLOAD_SHA,
        "session_status_bundle_sha256": STATUS_SHA,
        "region_mapping_path": "provenance/iblatlas_allen_structure_tree.csv",
        "region_mapping_sha256": neural_cache._REGION_MAPPING_SHA256,
        "region_mapping_provenance_path": "provenance/region_mapping_provenance.json",
        "region_mapping_provenance_sha256": neural_cache._REGION_MAPPING_PROVENANCE_SHA256,
        "iblatlas_version": "1.1.0",
        "iblatlas_source_commit": "unavailable_pypi_distribution_direct_url_absent",
        "iblatlas_regions_source_sha256": neural_cache._REGIONS_SOURCE_SHA256,
    }


def _write_manifest(root: Path, rows: list[dict[str, object]]) -> Path:
    manifest = root / "compact_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _write_bundle(root: Path, rows: list[dict[str, object]], manifest: Path) -> Path:
    complete = [row for row in rows if row["status"] == "complete"]
    failed = [row for row in rows if row["status"] == "failed"]
    if not complete:
        raise AssertionError("bundle fixture needs one complete session")
    first_metadata = json.loads(
        (root / str(complete[0]["metadata_path"])).read_text(encoding="utf-8")
    )
    artifact_sha256 = {
        str(row["eid"]): {
            "npz": str(row["npz_sha256"]),
            "stimulus": str(row["stimulus_trials_sha256"]),
            "movement": str(row["movement_trials_sha256"]),
            "metadata": str(row["metadata_sha256"]),
        }
        for row in complete
    }
    bundle_payload = {
        "schema_version": "ibl_exp14_compact_bundle_v2",
        "created_at_utc": "2026-07-12T03:15:00+00:00",
        "compact_manifest_sha256": _sha(manifest),
        "artifact_sha256": artifact_sha256,
        "source_manifest_sha256": SOURCE_SHA,
        "acquisition_bundle_sha256": BUNDLE_SHA,
        "acquisition_approval_sha256": APPROVAL_SHA,
        "download_summary_sha256": DOWNLOAD_SHA,
        "download_log_sha256": DOWNLOAD_LOG_SHA,
        "session_status_bundle_sha256": STATUS_SHA,
        "acquisition_external_side_effect": first_metadata[
            "acquisition_external_side_effect"
        ],
        "postprocess_script_sha256": neural_cache._POSTPROCESS_SCRIPT_SHA256,
        "postprocess_launcher_sha256": neural_cache._POSTPROCESS_LAUNCHER_SHA256,
        "compact_schema_sha256": neural_cache._COMPACT_SCHEMA_SHA256,
        "bwm_loader_sha256": neural_cache._BWM_LOADER_SHA256,
        "bwm_loader_relative_path": "provenance/bwm_loading.py",
        "region_mapping_path": "provenance/iblatlas_allen_structure_tree.csv",
        "region_mapping_sha256": neural_cache._REGION_MAPPING_SHA256,
        "region_mapping_provenance_path": "provenance/region_mapping_provenance.json",
        "region_mapping_provenance_sha256": (
            neural_cache._REGION_MAPPING_PROVENANCE_SHA256
        ),
        "iblatlas_version": "1.1.0",
        "iblatlas_source_commit": ("unavailable_pypi_distribution_direct_url_absent"),
        "iblatlas_regions_source_sha256": neural_cache._REGIONS_SOURCE_SHA256,
        "complete_sessions": len(complete),
        "failed_sessions": len(failed),
    }
    bundle = root / "compact_bundle.json"
    bundle.write_text(json.dumps(bundle_payload), encoding="utf-8")
    return bundle


def _load(root: Path, rows: list[dict[str, object]]):
    manifest = _write_manifest(root, rows)
    bundle = _write_bundle(root, rows, manifest)
    return _load_existing(root, rows, manifest, bundle)


def _load_existing(
    root: Path,
    rows: list[dict[str, object]],
    manifest: Path,
    bundle: Path,
):
    return load_compact_neural_cohort(
        manifest,
        expected_source_manifest_sha256=SOURCE_SHA,
        expected_acquisition_bundle_sha256=BUNDLE_SHA,
        expected_bwm_repository_commit=BWM_COMMIT,
        expected_compact_manifest_sha256=_sha(manifest),
        expected_compact_bundle_sha256=_sha(bundle),
        expected_sessions=len(rows),
        minimum_animals=1,
    )


def _rewrite_npz(
    root: Path,
    row: dict[str, object],
    mutate,
) -> None:
    path = root / str(row["npz_path"])
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)
    row["npz_sha256"] = _sha(path)


def _rewrite_metadata(root: Path, row: dict[str, object], mutate) -> None:
    path = root / str(row["metadata_path"])
    metadata = json.loads(path.read_text(encoding="utf-8"))
    mutate(metadata)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    row["metadata_sha256"] = _sha(path)


def _rewrite_csv(root: Path, row: dict[str, object], artifact: str, mutate) -> None:
    path_key = f"{artifact}_trials_path"
    sha_key = f"{artifact}_trials_sha256"
    path = root / str(row[path_key])
    table = pd.read_csv(path)
    mutate(table)
    table.to_csv(path, index=False, float_format="%.17g")
    row[sha_key] = _sha(path)


def test_compact_cache_loads_only_hash_bound_offline_artifacts(tmp_path: Path) -> None:
    # Source candidate ranks are deliberately non-contiguous in the reviewed
    # cohort; the trusted manifest hash binds the exact rank/EID mapping.
    rows = [_complete_row(tmp_path, rank) for rank in (0, 3)]
    manifest = tmp_path / "compact_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    bundle = _write_bundle(tmp_path, rows, manifest)
    reviewed_manifest_sha = _sha(manifest)
    cohort = load_compact_neural_cohort(
        manifest,
        expected_source_manifest_sha256=SOURCE_SHA,
        expected_acquisition_bundle_sha256=BUNDLE_SHA,
        expected_bwm_repository_commit=BWM_COMMIT,
        expected_compact_manifest_sha256=reviewed_manifest_sha,
        expected_compact_bundle_sha256=_sha(bundle),
        expected_sessions=2,
        minimum_animals=2,
    )
    assert len(cohort.dispositions) == 2
    assert [item.candidate_rank for item in cohort.dispositions] == [0, 3]
    assert len(cohort.sessions) == 2
    assert cohort.sessions[0].count_views["stimulus_pre"].dtype == np.int64
    assert (
        cohort.sessions[0].trial_table("movement_pre").loc[0, "motion_energy_proxy"]
        == 1.1
    )
    assert len(cohort.compact_manifest_sha256) == 64
    assert cohort.compact_bundle_sha256 == _sha(bundle)

    first_npz = tmp_path / str(rows[0]["npz_path"])
    with first_npz.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(IBLNeuralCacheError, match="SHA-256 mismatch"):
        load_compact_neural_cohort(
            manifest,
            expected_source_manifest_sha256=SOURCE_SHA,
            expected_acquisition_bundle_sha256=BUNDLE_SHA,
            expected_bwm_repository_commit=BWM_COMMIT,
            expected_compact_manifest_sha256=reviewed_manifest_sha,
            expected_compact_bundle_sha256=_sha(bundle),
            expected_sessions=2,
            minimum_animals=2,
        )

    rewritten = pd.read_csv(manifest, keep_default_na=False)
    rewritten.loc[0, "npz_sha256"] = _sha(first_npz)
    rewritten.to_csv(manifest, index=False)
    with pytest.raises(IBLNeuralCacheError, match="reviewed root hash"):
        load_compact_neural_cohort(
            manifest,
            expected_source_manifest_sha256=SOURCE_SHA,
            expected_acquisition_bundle_sha256=BUNDLE_SHA,
            expected_bwm_repository_commit=BWM_COMMIT,
            expected_compact_manifest_sha256=reviewed_manifest_sha,
            expected_compact_bundle_sha256=_sha(bundle),
            expected_sessions=2,
            minimum_animals=2,
        )


def test_compact_cache_retains_failed_session_rows(tmp_path: Path) -> None:
    complete = _complete_row(tmp_path, 0)
    failed = {
        **complete,
        "candidate_rank": 1,
        "eid": "eid-failed",
        "animal_id": "animal-failed",
        "status": "failed",
        "error_type": "DownloadError",
        "error": "missing frozen dataset",
        "npz_path": "",
        "npz_sha256": "",
        "stimulus_trials_path": "",
        "stimulus_trials_sha256": "",
        "movement_trials_path": "",
        "movement_trials_sha256": "",
        "metadata_path": "",
        "metadata_sha256": "",
    }
    manifest = tmp_path / "compact_manifest.csv"
    pd.DataFrame([complete, failed]).to_csv(manifest, index=False)
    bundle = _write_bundle(tmp_path, [complete, failed], manifest)
    cohort = load_compact_neural_cohort(
        manifest,
        expected_source_manifest_sha256=SOURCE_SHA,
        expected_acquisition_bundle_sha256=BUNDLE_SHA,
        expected_bwm_repository_commit=BWM_COMMIT,
        expected_compact_manifest_sha256=_sha(manifest),
        expected_compact_bundle_sha256=_sha(bundle),
        expected_sessions=2,
        minimum_animals=1,
    )
    assert len(cohort.sessions) == 1
    assert cohort.dispositions[1].status == "failed"
    assert cohort.dispositions[1].error == "missing frozen dataset"


def test_compact_cache_rejects_public_csv_context_leak(tmp_path: Path) -> None:
    row = _complete_row(tmp_path, 0)
    _rewrite_csv(
        tmp_path,
        row,
        "stimulus",
        lambda table: table.__setitem__("probability_left", 0.8),
    )
    with pytest.raises(IBLNeuralCacheError, match="probability/context fields"):
        _load(tmp_path, [row])


@pytest.mark.parametrize("failure", ["shape", "dtype", "time"])
def test_compact_cache_rejects_npz_shape_dtype_or_time_tamper(
    tmp_path: Path, failure: str
) -> None:
    row = _complete_row(tmp_path, 0)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        if failure == "shape":
            arrays["stimulus_pre_counts"] = arrays["stimulus_pre_counts"][:, :-1]
        elif failure == "dtype":
            arrays["stimulus_pre_counts"] = arrays["stimulus_pre_counts"].astype(
                np.int64
            )
        else:
            arrays["movement_pre_time"][0] += 1e-6

    _rewrite_npz(tmp_path, row, mutate)
    with pytest.raises(IBLNeuralCacheError, match="counts must|time axis"):
        _load(tmp_path, [row])


def test_compact_cache_rejects_trial_id_block_and_side_contracts(
    tmp_path: Path,
) -> None:
    row = _complete_row(tmp_path, 0)
    _rewrite_csv(
        tmp_path,
        row,
        "movement",
        lambda table: table.__setitem__("block_id", 7),
    )
    with pytest.raises(IBLNeuralCacheError, match="block_id"):
        _load(tmp_path, [row])

    row = _complete_row(tmp_path, 0)
    _rewrite_csv(
        tmp_path,
        row,
        "stimulus",
        lambda table: table.__setitem__("stimulus_side", 0),
    )
    with pytest.raises(IBLNeuralCacheError, match="left=1/right=0"):
        _load(tmp_path, [row])


def test_compact_cache_recomputes_qc_equivalence_and_binds_units(
    tmp_path: Path,
) -> None:
    row = _complete_row(tmp_path, 0)

    def mutate(metadata: dict[str, object]) -> None:
        payload = metadata["unit_qc_equivalence"]
        assert isinstance(payload, dict)
        probes = payload["probes"]
        assert isinstance(probes, list)
        probes[0]["metrics_label_ge_1_cluster_ids"] = [0]
        digest = _canonical_sha(payload)
        metadata["unit_qc_equivalence_sha256"] = digest
        row["unit_qc_equivalence_sha256"] = digest

    _rewrite_metadata(tmp_path, row, mutate)
    with pytest.raises(IBLNeuralCacheError, match="cluster IDs must be equal"):
        _load(tmp_path, [row])


def test_compact_cache_recomputes_and_cross_checks_official_mask(
    tmp_path: Path,
) -> None:
    row = _complete_row(tmp_path, 0)

    def mutate(metadata: dict[str, object]) -> None:
        evidence = metadata["official_bwm_mask_evidence"]
        assert isinstance(evidence, dict)
        mask = evidence["mask"]
        assert isinstance(mask, list)
        mask[0] = 1
        evidence["selected_trial_ids"] = [
            trial_id
            for trial_id, keep in zip(evidence["trial_ids"], mask, strict=True)
            if keep
        ]
        digest = _canonical_sha(evidence)
        metadata["official_bwm_mask_sha256"] = digest
        metadata["official_bwm_mask_count"] = int(sum(mask))
        row["official_bwm_mask_sha256"] = digest

    _rewrite_metadata(tmp_path, row, mutate)
    with pytest.raises(IBLNeuralCacheError, match="CSV official BWM mask"):
        _load(tmp_path, [row])


def test_compact_cache_rejects_true_context_blocks_and_invalid_mask_formula(
    tmp_path: Path,
) -> None:
    row = _complete_row(tmp_path, 0)

    def leak_context(metadata: dict[str, object]) -> None:
        policy = metadata["cv_block_policy"]
        assert isinstance(policy, dict)
        policy["is_true_context"] = True

    _rewrite_metadata(tmp_path, row, leak_context)
    with pytest.raises(IBLNeuralCacheError, match="non-context chunks"):
        _load(tmp_path, [row])

    row = _complete_row(tmp_path, 0)

    def alter_valid(arrays: dict[str, np.ndarray]) -> None:
        arrays["stimulus_pre_valid"][1] = False
        arrays["stimulus_pre_counts"][1] = 0

    _rewrite_npz(tmp_path, row, alter_valid)

    def alter_timing_valid(table: pd.DataFrame) -> None:
        table.loc[1, "timing_valid"] = False

    _rewrite_csv(tmp_path, row, "stimulus", alter_timing_valid)
    with pytest.raises(
        IBLNeuralCacheError, match="event/support/official-mask formula"
    ):
        _load(tmp_path, [row])


def test_compact_cache_rejects_counts_outside_valid_mask(tmp_path: Path) -> None:
    row = _complete_row(tmp_path, 0)

    def add_invalid_activity(arrays: dict[str, np.ndarray]) -> None:
        arrays["movement_pre_counts"][0, 0, 0] = 1

    _rewrite_npz(tmp_path, row, add_invalid_activity)
    with pytest.raises(IBLNeuralCacheError, match="outside the reviewed valid mask"):
        _load(tmp_path, [row])


def test_compact_cache_recomputes_acquisition_receipt_hash(tmp_path: Path) -> None:
    row = _complete_row(tmp_path, 0)

    def mutate(metadata: dict[str, object]) -> None:
        receipts = metadata["acquisition_official_loader_receipts"]
        assert isinstance(receipts, list)
        receipts[0]["source_record"]["error"] = "rewritten after acquisition"

    _rewrite_metadata(tmp_path, row, mutate)
    with pytest.raises(IBLNeuralCacheError, match="source-record hash mismatch"):
        _load(tmp_path, [row])


def test_compact_cache_binds_qc_acronyms_to_frozen_region_mapping(
    tmp_path: Path,
) -> None:
    row = _complete_row(tmp_path, 0)

    def mutate_metadata(metadata: dict[str, object]) -> None:
        payload = metadata["unit_qc_equivalence"]
        assert isinstance(payload, dict)
        probes = payload["probes"]
        assert isinstance(probes, list)
        probes[0]["cluster_channel_ccf_acronym"][0][3] = "VISp"
        digest = _canonical_sha(payload)
        metadata["unit_qc_equivalence_sha256"] = digest
        row["unit_qc_equivalence_sha256"] = digest

    def mutate_npz(arrays: dict[str, np.ndarray]) -> None:
        arrays["regions"] = np.asarray(["VISp", "MD"])

    _rewrite_metadata(tmp_path, row, mutate_metadata)
    _rewrite_npz(tmp_path, row, mutate_npz)
    with pytest.raises(IBLNeuralCacheError, match="frozen region mapping"):
        _load(tmp_path, [row])


@pytest.mark.parametrize(
    ("event", "event_fields", "validation_status"),
    [
        (
            "official_good_unit_loader_passed",
            {"unit_count": 2, "spike_count": 100},
            "official_pinned_load_good_units_passed",
        ),
        (
            "official_good_unit_loader_skipped_missing_inventory",
            {"missing_dataset_uuids": [_uuid(999)]},
            "official_good_unit_loader_skipped_missing_inventory_retained",
        ),
    ],
)
def test_compact_cache_accepts_variable_raw_loader_receipt_schemas(
    tmp_path: Path,
    event: str,
    event_fields: dict[str, object],
    validation_status: str,
) -> None:
    row = _complete_row(tmp_path, 0)

    def mutate(metadata: dict[str, object]) -> None:
        wrappers = metadata["acquisition_official_loader_receipts"]
        assert isinstance(wrappers, list)
        original = wrappers[0]["source_record"]
        common = {
            key: original[key]
            for key in (
                "run_id",
                "timestamp_utc",
                "eid",
                "pid",
                "probe_name",
                "revision",
                "qc",
            )
        }
        record = {**common, "event": event, **event_fields}
        wrappers[0] = {
            "source_record": record,
            "source_record_sha256": _canonical_sha(record),
        }
        metadata["acquisition_validation_status"] = validation_status
        metadata["acquisition_session_status"] = (
            "passed"
            if event == "official_good_unit_loader_passed"
            else "failed_retained"
        )

    row["acquisition_validation_status"] = validation_status
    _rewrite_metadata(tmp_path, row, mutate)
    assert len(_load(tmp_path, [row]).sessions) == 1


def test_compact_cache_rejects_required_input_identity_rewrite(tmp_path: Path) -> None:
    row = _complete_row(tmp_path, 0)

    def mutate(metadata: dict[str, object]) -> None:
        payload = metadata["compact_required_input_bundle"]
        assert isinstance(payload, dict)
        datasets = payload["datasets"]
        assert isinstance(datasets, list)
        datasets[0]["md5"] = "0" * 32
        metadata["compact_required_input_bundle_sha256"] = _canonical_sha(payload)

    _rewrite_metadata(tmp_path, row, mutate)
    with pytest.raises(IBLNeuralCacheError, match="independently derived inputs"):
        _load(tmp_path, [row])


def test_compact_cache_retains_auxiliary_dataset_failure_receipt(
    tmp_path: Path,
) -> None:
    row = _complete_row(tmp_path, 0)

    def mutate(metadata: dict[str, object]) -> None:
        record = {
            "run_id": "reviewed-acquisition-run",
            "timestamp_utc": "2026-07-12T00:01:00+00:00",
            "eid": "eid-0",
            "dataset_uuid": _uuid(998),
            "category": "good_unit_spike_sorting",
            "name": "clusters.waveforms.npy",
            "collection": "alf/probe00/pykilosort",
            "human_revision": "2024-05-06",
            "event": "dataset_failed",
            "error_type": "SSLError",
            "error": "retained auxiliary download failure",
            "traceback": "retained trace",
        }
        metadata["acquisition_dataset_failure_receipts"] = [
            {
                "source_record": record,
                "source_record_sha256": _canonical_sha(record),
            }
        ]
        policy = metadata["auxiliary_dataset_failure_policy"]
        assert isinstance(policy, dict)
        policy["auxiliary_failure_count"] = 1
        policy["auxiliary_failure_dataset_uuids"] = [_uuid(998)]

    _rewrite_metadata(tmp_path, row, mutate)
    assert len(_load(tmp_path, [row]).sessions) == 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("used_by_compact", True, "blocked as input"),
        ("receipts_authoritative_for_qc", True, "blocked as input"),
        ("policy", "allowed_as_input", "blocked as input"),
    ],
)
def test_compact_cache_rejects_external_side_effect_as_input(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    row = _complete_row(tmp_path, 0)

    def mutate(metadata: dict[str, object]) -> None:
        evidence = metadata["acquisition_external_side_effect"]
        assert isinstance(evidence, dict)
        evidence[field] = value

    _rewrite_metadata(tmp_path, row, mutate)
    with pytest.raises(IBLNeuralCacheError, match=match):
        _load(tmp_path, [row])


def test_compact_cache_recomputes_no_truncation_evidence(tmp_path: Path) -> None:
    row = _complete_row(tmp_path, 0)

    def mutate(metadata: dict[str, object]) -> None:
        evidence = metadata["official_bwm_mask_evidence"]
        assert isinstance(evidence, dict)
        no_truncation = evidence["no_truncation_precondition"]
        assert isinstance(no_truncation, dict)
        no_truncation["easy_performance"] = 0.95
        digest = _canonical_sha(evidence)
        metadata["official_bwm_mask_sha256"] = digest
        row["official_bwm_mask_sha256"] = digest

    _rewrite_metadata(tmp_path, row, mutate)
    with pytest.raises(IBLNeuralCacheError, match="does not reproduce"):
        _load(tmp_path, [row])


def test_compact_cache_rejects_unreviewed_v2_metadata_key(tmp_path: Path) -> None:
    row = _complete_row(tmp_path, 0)
    _rewrite_metadata(
        tmp_path,
        row,
        lambda metadata: metadata.__setitem__("unreviewed_input", "hidden"),
    )
    with pytest.raises(IBLNeuralCacheError, match="metadata schema is not v2"):
        _load(tmp_path, [row])


def test_compact_cache_round_trips_producer_17g_event_times(tmp_path: Path) -> None:
    row = _complete_row(tmp_path, 0)

    def mutate_table(table: pd.DataFrame) -> None:
        table.loc[1, "stim_on"] = 4128.028166666667
        table.loc[1, "first_movement"] = 4128.1818000050225
        table.loc[1, "reaction_time"] = 0.1536333383555757

    def mutate_metadata(metadata: dict[str, object]) -> None:
        supports = metadata["observed_good_spike_support_s"]
        assert isinstance(supports, list)
        supports[0]["stop"] = 5000.0

    _rewrite_csv(tmp_path, row, "stimulus", mutate_table)
    _rewrite_metadata(tmp_path, row, mutate_metadata)
    path = tmp_path / str(row["stimulus_trials_path"])
    default = pd.read_csv(path)
    round_trip = pd.read_csv(path, float_precision="round_trip")
    default_error = float(
        default.loc[1, "reaction_time"]
        - (default.loc[1, "first_movement"] - default.loc[1, "stim_on"])
    )
    round_trip_error = float(
        round_trip.loc[1, "reaction_time"]
        - (round_trip.loc[1, "first_movement"] - round_trip.loc[1, "stim_on"])
    )
    assert abs(default_error) > 1e-12
    assert round_trip_error == 0.0
    assert len(_load(tmp_path, [row]).sessions) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("postprocess_script_sha256", "1" * 64),
        ("compact_schema_sha256", "2" * 64),
        ("postprocess_launcher_sha256", "3" * 64),
    ],
)
def test_compact_cache_rejects_rehashed_unreviewed_bundle_lineage(
    tmp_path: Path, field: str, value: str
) -> None:
    row = _complete_row(tmp_path, 0)
    rows = [row]
    manifest = _write_manifest(tmp_path, rows)
    bundle = _write_bundle(tmp_path, rows, manifest)
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    payload[field] = value
    bundle.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IBLNeuralCacheError, match="reviewed producer/schema"):
        _load_existing(tmp_path, rows, manifest, bundle)


def test_compact_cache_rejects_rehashed_bundle_artifact_mismatch(
    tmp_path: Path,
) -> None:
    row = _complete_row(tmp_path, 0)
    rows = [row]
    manifest = _write_manifest(tmp_path, rows)
    bundle = _write_bundle(tmp_path, rows, manifest)
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    payload["artifact_sha256"]["eid-0"]["npz"] = "4" * 64
    bundle.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IBLNeuralCacheError, match="artifact hash disagrees"):
        _load_existing(tmp_path, rows, manifest, bundle)


def test_compact_cache_hashes_copied_bwm_provenance(tmp_path: Path) -> None:
    row = _complete_row(tmp_path, 0)
    rows = [row]
    manifest = _write_manifest(tmp_path, rows)
    bundle = _write_bundle(tmp_path, rows, manifest)
    bwm_loader = tmp_path / "provenance" / "bwm_loading.py"
    bwm_loader.write_text("# rewritten after compact build\n", encoding="utf-8")
    with pytest.raises(IBLNeuralCacheError, match="SHA-256 mismatch"):
        _load_existing(tmp_path, rows, manifest, bundle)
