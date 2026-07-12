#!/usr/bin/env python3
"""Candidate-only offline exp14 compactor; execution requires a second review gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "one"
MANIFESTS = ROOT / "manifests"
STATUS = ROOT / "status"
DOWNLOAD_LOG = ROOT / "logs" / "download.log"
OUTPUT = ROOT / "compact_v1"
SCHEMA = MANIFESTS / "compact_schema.json"
POSTPROCESS_LAUNCHER = ROOT / "scripts" / "launch_postprocess.sh"
REGION_MAPPING = ROOT / "frozen_sources" / "iblatlas_allen_structure_tree.csv"
REGION_MAPPING_PROVENANCE = MANIFESTS / "region_mapping_provenance.json"
POSTPROCESS_REVIEW = STATUS / "POSTPROCESS_REVIEW_APPROVED.json"
ACQUISITION_APPROVAL = STATUS / "REVIEW_APPROVED.json"
ACQUISITION_RUN_ID = "ibl-exp14-20260712T004706Z-67048f63"
ACQUISITION_APPROVAL_SHA256 = "64e0b1ead4175b2007b0c4aa01524e9e9db4beebde63f016cf46e245f0da4287"
ACQUISITION_BUNDLE_SHA256 = "0c3e12126d2151e78382e41f0ef9ed6023ffb67fa4ad7e285f1bc570a9c8f3cd"
SOURCE_MANIFEST_SHA256 = "112c84ad93eee49186ab117343ebebb4921d2f1bcea57a9c9326ca38d337a0e6"
BWM_COMMIT = "118fc36cb3602934466ad2c6087c2b3b441f9f1f"
BWM_LOADER_SHA256 = "c2e570c62cd0e047303c97d7999711b659a1c37eaa56dd2740ffff2c81f85321"
POSTPROCESS_LAUNCHER_SHA256 = "fe8a30ab705d7e0c8474e574fcae36ce68f510d2d53de81003edd3a3f05f2837"
PINNED_BWM_LOADER = ROOT / "frozen_sources" / "bwm_loading.py"
REGION_MAPPING_SHA256 = "63654b8d35c7c1b5665636b645da774776ee8263658192f5dca1e815095e9147"
REGION_MAPPING_PROVENANCE_SHA256 = "a01b7fa535e6de437ac46e8cf9de68a87d6a9b5587d055a3935476d956109fdc"
IBLATLAS_VERSION = "1.1.0"
IBLATLAS_REGIONS_SOURCE_SHA256 = "cdfe3e5c8ed350af182b14f7ce627096484a529da9837264f53f3472319dcc63"
SPIKE_SORTING_REVISION = "2024-05-06"
UNIT_QC_THRESHOLD = 1.0
BIN_SIZE = 0.020
WINDOW = (-0.5, 0.0)
CAMERA_PRIORITY = ("left", "right", "body")
COMPACT_REQUIRED_DATASET_NAMES = frozenset({
    "_ibl_trials.table.pqt",
    "_ibl_wheel.timestamps.npy",
    "_ibl_wheel.position.npy",
    "passingSpikes.table.pqt",
    "clusters.metrics.pqt",
    "clusters.channels.npy",
    "channels.brainLocationIds_ccf_2017.npy",
})


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: object) -> None:
    if not inside(path, OUTPUT):
        raise RuntimeError(f"output escaped compact root: {path}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def cache_path(row: dict[str, str]) -> Path:
    prefix = "https://ibl.flatironinstitute.org/public/"
    data_url = row["data_url"]
    if not data_url.startswith(prefix) or "?" in data_url or "\\" in data_url:
        raise RuntimeError(f"unexpected frozen dataset URL: {row['dataset_uuid']}")
    parts = data_url[len(prefix):].split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise RuntimeError(f"unsafe frozen dataset URL path: {row['dataset_uuid']}")
    uuid_token = f".{uuid.UUID(row['dataset_uuid'])}."
    if parts[-1].count(uuid_token) != 1:
        raise RuntimeError(f"dataset UUID filename binding is invalid: {row['dataset_uuid']}")
    parts[-1] = parts[-1].replace(uuid_token, ".", 1)
    path = CACHE.joinpath(*parts)
    if not inside(path, CACHE):
        raise RuntimeError(f"dataset path escaped cache: {path}")
    return path


def checked_dataset(row: dict[str, str], tracker: dict[str, dict[str, object]],
                    role: str) -> Path:
    if not role:
        raise RuntimeError("compact dataset role is required")
    path = cache_path(row)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(row["file_size"]):
        raise RuntimeError(f"dataset size mismatch: {row['dataset_uuid']}")
    if hash_file(path, "md5").lower() != row["md5"].lower():
        raise RuntimeError(f"dataset MD5 mismatch: {row['dataset_uuid']}")
    identity: dict[str, object] = {
        "dataset_uuid": row["dataset_uuid"],
        "name": row["name"],
        "md5": row["md5"],
        "roles": [],
    }
    existing = tracker.setdefault(row["dataset_uuid"], identity)
    if any(existing.get(key) != identity[key] for key in ("dataset_uuid", "name", "md5")):
        raise RuntimeError(f"dataset identity drift in compact tracker: {row['dataset_uuid']}")
    roles = existing.get("roles")
    if not isinstance(roles, list):
        raise RuntimeError("compact tracker roles are invalid")
    if role not in roles:
        roles.append(role)
    return path


def verify_manifest_bundle() -> None:
    if hash_file(MANIFESTS / "manifest_hashes.json") != ACQUISITION_BUNDLE_SHA256:
        raise RuntimeError("acquisition bundle SHA mismatch")
    hashes = read_json(MANIFESTS / "manifest_hashes.json")
    for name, expected in hashes.items():
        if hash_file(MANIFESTS / name) != expected:
            raise RuntimeError(f"acquisition manifest drift: {name}")
    if hash_file(ACQUISITION_APPROVAL) != ACQUISITION_APPROVAL_SHA256:
        raise RuntimeError("acquisition approval SHA mismatch")
    approval = read_json(ACQUISITION_APPROVAL)
    if approval.get("decision") != "approved":
        raise RuntimeError("acquisition review is not approved")
    if hash_file(PINNED_BWM_LOADER) != BWM_LOADER_SHA256:
        raise RuntimeError("pinned BWM loader drift")
    if hash_file(POSTPROCESS_LAUNCHER) != POSTPROCESS_LAUNCHER_SHA256:
        raise RuntimeError("postprocess launcher SHA mismatch")


def completed_acquisition() -> tuple[dict[str, object], str]:
    summary_path = STATUS / f"download_summary_{ACQUISITION_RUN_ID}.json"
    if not summary_path.is_file():
        raise RuntimeError("acquisition is not complete")
    summary = read_json(summary_path)
    if summary.get("status") not in {"completed", "completed_with_failures"}:
        raise RuntimeError(f"invalid acquisition status: {summary.get('status')}")
    totals = summary.get("totals", {})
    accounted = int(totals.get("dataset_downloaded_verified", 0))
    accounted += int(totals.get("dataset_cache_hit_verified", 0))
    accounted += int(totals.get("dataset_failed", 0))
    if accounted != 727:
        raise RuntimeError(f"acquisition did not account for all 727 datasets: {accounted}")
    return summary, hash_file(summary_path)


def acquisition_external_side_effect() -> tuple[dict[str, object], str]:
    if not DOWNLOAD_LOG.is_file():
        raise RuntimeError("acquisition download log is absent")
    external_paths = [
        "/home/wangmeiqi/Downloads/ONE/openalyx.internationalbrainlab.org/"
        "histology/ATLAS/Needles/Allen/average_template_25.nrrd",
        "/home/wangmeiqi/Downloads/ONE/openalyx.internationalbrainlab.org/"
        "histology/ATLAS/Needles/Allen/annotation_25.nrrd",
    ]
    markers = [f"Downloading: {path}" for path in external_paths]
    log_text = DOWNLOAD_LOG.read_text(encoding="utf-8", errors="replace")
    missing = [marker for marker in markers if marker not in log_text]
    if missing:
        raise RuntimeError("acquisition external-atlas side-effect markers are absent")
    log_sha = hash_file(DOWNLOAD_LOG)
    evidence = {
        "observed": True,
        "scope": "staging_external_atlas_cache",
        "evidence_log_relative_path": str(DOWNLOAD_LOG.relative_to(ROOT)),
        "evidence_log_sha256": log_sha,
        "observed_markers": markers,
        "external_paths": external_paths,
        "used_by_compact": False,
        "receipts_authoritative_for_qc": False,
        "policy": "blocked_as_input",
    }
    return evidence, log_sha


def session_status_bundle(sessions: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    records = []
    for session in sorted(sessions, key=lambda row: int(row["candidate_rank"])):
        matches = list((STATUS / "sessions").glob(
            f"{ACQUISITION_RUN_ID}_*_{session['eid']}.json"
        ))
        if len(matches) != 1:
            raise RuntimeError(f"expected one acquisition status for {session['eid']}")
        records.append({"eid": session["eid"],
                        "relative_path": str(matches[0].relative_to(ROOT)),
                        "sha256": hash_file(matches[0])})
    return canonical_sha256(records), records


def load_region_mapping() -> tuple[dict[int, str], dict[str, object]]:
    if hash_file(REGION_MAPPING) != REGION_MAPPING_SHA256:
        raise RuntimeError("frozen iblatlas region mapping SHA mismatch")
    if hash_file(REGION_MAPPING_PROVENANCE) != REGION_MAPPING_PROVENANCE_SHA256:
        raise RuntimeError("region mapping provenance SHA mismatch")
    provenance = read_json(REGION_MAPPING_PROVENANCE)
    expected = {
        "schema_version": "ibl_exp14_region_mapping_input_v1",
        "source_package": "iblatlas",
        "source_version": IBLATLAS_VERSION,
        "mapping_sha256": REGION_MAPPING_SHA256,
        "regions_source_sha256": IBLATLAS_REGIONS_SOURCE_SHA256,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise RuntimeError("region mapping provenance content mismatch")
    table = pd.read_csv(REGION_MAPPING)
    if not {"id", "acronym"} <= set(table):
        raise RuntimeError("frozen region mapping lacks id/acronym columns")
    ids_numeric = pd.to_numeric(table["id"], errors="raise").to_numpy(dtype=np.float64)
    if (not np.isfinite(ids_numeric).all() or
            not np.array_equal(ids_numeric, ids_numeric.astype(np.int64))):
        raise RuntimeError("frozen region mapping IDs are not finite integers")
    ids = ids_numeric.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("frozen region mapping IDs are not unique")
    if table["acronym"].isna().any():
        raise RuntimeError("frozen region mapping contains null acronyms")
    acronyms = table["acronym"].astype(str).to_numpy()
    if any(not value.strip() for value in acronyms):
        raise RuntimeError("frozen region mapping contains empty acronyms")
    lookup = {int(region_id): str(acronym)
              for region_id, acronym in zip(ids, acronyms, strict=True)}
    metadata = {
        "schema_version": "ibl_exp14_region_mapping_input_v1",
        "mapping_relative_path": "provenance/iblatlas_allen_structure_tree.csv",
        "mapping_sha256": REGION_MAPPING_SHA256,
        "provenance_relative_path": "provenance/region_mapping_provenance.json",
        "provenance_sha256": REGION_MAPPING_PROVENANCE_SHA256,
        "source_package": "iblatlas",
        "source_version": IBLATLAS_VERSION,
        "source_commit": None,
        "source_commit_status": "unavailable_pypi_distribution_direct_url_absent",
        "regions_source_sha256": IBLATLAS_REGIONS_SOURCE_SHA256,
    }
    return lookup, metadata


def acquisition_loader_receipts(state: dict[str, object], probes: list[dict[str, str]],
                                eid: str) -> list[dict[str, object]]:
    raw_receipts = state.get("probe_validations")
    if not isinstance(raw_receipts, list):
        raise RuntimeError(f"acquisition probe validations missing for {eid}")
    expected = sorted((probe["probe_name"], probe["pid"]) for probe in probes)
    actual = sorted((str(item.get("probe_name", "")), str(item.get("pid", "")))
                    for item in raw_receipts)
    if actual != expected:
        raise RuntimeError(f"acquisition probe validation identity mismatch for {eid}")
    receipts = []
    for raw in sorted(raw_receipts,
                      key=lambda item: (str(item.get("probe_name", "")),
                                        str(item.get("pid", "")))):
        if (raw.get("eid") != eid or raw.get("run_id") != ACQUISITION_RUN_ID or
                raw.get("revision") != SPIKE_SORTING_REVISION or
                float(raw.get("qc", float("nan"))) != UNIT_QC_THRESHOLD):
            raise RuntimeError(f"acquisition loader receipt provenance mismatch for {eid}")
        event_name = str(raw.get("event", ""))
        base_keys = {"run_id", "timestamp_utc", "eid", "pid", "probe_name",
                     "revision", "qc", "event"}
        if event_name == "official_good_unit_loader_passed":
            if set(raw) != base_keys | {"unit_count", "spike_count"}:
                raise RuntimeError(f"official loader pass receipt schema mismatch for {eid}")
            unit_count = int(raw["unit_count"])
            spike_count = int(raw["spike_count"])
            if unit_count < 0 or spike_count < 0:
                raise RuntimeError(f"negative official loader counts for {eid}")
        elif event_name == "official_good_unit_loader_failed":
            if set(raw) != base_keys | {"error_type", "error", "traceback"}:
                raise RuntimeError(f"official loader failure receipt schema mismatch for {eid}")
            error_type = str(raw["error_type"])
            error = str(raw["error"])
            trace = str(raw["traceback"])
            if not error_type or not error or not trace:
                raise RuntimeError(f"incomplete official loader failure receipt for {eid}")
        elif event_name == "official_good_unit_loader_skipped_missing_inventory":
            if set(raw) != base_keys | {"missing_dataset_uuids"}:
                raise RuntimeError(f"official loader skipped receipt schema mismatch for {eid}")
            missing = raw["missing_dataset_uuids"]
            if (not isinstance(missing, list) or not missing or
                    any(str(uuid.UUID(str(value))) != str(value) for value in missing)):
                raise RuntimeError(f"official loader skipped receipt UUIDs invalid for {eid}")
        else:
            raise RuntimeError(f"unexpected official loader receipt event for {eid}: {event_name}")
        if not isinstance(raw.get("timestamp_utc"), str) or not raw["timestamp_utc"]:
            raise RuntimeError(f"official loader receipt timestamp missing for {eid}")
        source_record = dict(raw)
        receipts.append({"source_record": source_record,
                         "source_record_sha256": canonical_sha256(source_record)})
    return receipts


def acquisition_dataset_failure_receipts(
        state: dict[str, object], session_rows: list[dict[str, str]]) -> tuple[
            list[dict[str, object]], list[dict[str, object]], list[dict[str, object]],
            list[dict[str, str]], str]:
    raw_datasets = state.get("datasets")
    if not isinstance(raw_datasets, list):
        raise RuntimeError("acquisition dataset receipts are missing")
    failures = sorted((dict(item) for item in raw_datasets if item.get("event") == "dataset_failed"),
                      key=lambda item: str(item.get("dataset_uuid", "")))
    inventory_by_uuid = {row["dataset_uuid"]: row for row in session_rows}
    wrappers = []
    for raw in failures:
        dataset_uuid = str(raw.get("dataset_uuid", ""))
        if dataset_uuid not in inventory_by_uuid:
            raise RuntimeError(f"dataset failure is absent from frozen inventory: {dataset_uuid}")
        inventory_row = inventory_by_uuid[dataset_uuid]
        if raw.get("name") != inventory_row["name"]:
            raise RuntimeError(f"dataset failure name mismatch: {dataset_uuid}")
        wrappers.append({"source_record": raw,
                         "source_record_sha256": canonical_sha256(raw)})
    failed_uuids = {item["source_record"]["dataset_uuid"] for item in wrappers}
    usable_rows = [row for row in session_rows if row["dataset_uuid"] not in failed_uuids]
    required = [item for item in wrappers
                if item["source_record"]["name"] in COMPACT_REQUIRED_DATASET_NAMES]
    selected_camera = ""
    for view in CAMERA_PRIORITY:
        time_rows = [row for row in usable_rows if row["name"] == f"_ibl_{view}Camera.times.npy"]
        energy_rows = [row for row in usable_rows if row["name"] == f"{view}Camera.ROIMotionEnergy.npy"]
        if len(time_rows) == len(energy_rows) == 1:
            selected_camera = view
            break
    if not selected_camera:
        raise RuntimeError("no intact reviewed camera time/motion-energy pair remains")
    auxiliary = [item for item in wrappers if item not in required]
    return wrappers, required, auxiliary, usable_rows, selected_camera


def expected_required_input_identities(rows: list[dict[str, str]],
                                       probes: list[dict[str, str]],
                                       selected_camera: str) -> list[dict[str, str]]:
    required_rows = [
        one_row(rows, name="_ibl_trials.table.pqt"),
        one_row(rows, name="_ibl_wheel.timestamps.npy"),
        one_row(rows, name="_ibl_wheel.position.npy"),
        one_row(rows, name=f"_ibl_{selected_camera}Camera.times.npy"),
        one_row(rows, name=f"{selected_camera}Camera.ROIMotionEnergy.npy"),
    ]
    for probe in sorted(probes, key=lambda row: (row["probe_name"], row["pid"])):
        probe_rows = [row for row in rows if row["pid"] == probe["pid"]]
        for name in ("passingSpikes.table.pqt", "clusters.metrics.pqt",
                     "clusters.channels.npy", "channels.brainLocationIds_ccf_2017.npy"):
            required_rows.append(one_row(probe_rows, name=name))
    identities = sorted((dataset_identity(row) for row in required_rows),
                        key=lambda item: item["dataset_uuid"])
    if len({item["dataset_uuid"] for item in identities}) != len(identities):
        raise RuntimeError("compact required input bundle contains duplicate UUIDs")
    return identities


def validate_inputs(require_postprocess_review: bool) -> dict[str, object]:
    if ROOT.resolve() != Path("/home/spco/sow_linear/ibl_neural_exp14_staging"):
        raise RuntimeError(f"unexpected staging root: {ROOT}")
    verify_manifest_bundle()
    sessions = read_csv(MANIFESTS / "selected_sessions.csv")
    probes = read_csv(MANIFESTS / "selected_probes.csv")
    inventory = read_csv(MANIFESTS / "dataset_inventory.csv")
    if len(sessions) != 20 or len({row["subject"] for row in sessions}) != 20:
        raise RuntimeError("expected 20 unique-animal sessions")
    if len(probes) != 35 or len(inventory) != 727:
        raise RuntimeError("probe/inventory cardinality mismatch")
    summary, summary_sha = completed_acquisition()
    side_effect_evidence, download_log_sha = acquisition_external_side_effect()
    status_bundle_sha, status_records = session_status_bundle(sessions)
    region_lookup, region_mapping_input = load_region_mapping()
    if require_postprocess_review:
        if not POSTPROCESS_REVIEW.is_file():
            raise RuntimeError("postprocess review marker is absent")
        marker = read_json(POSTPROCESS_REVIEW)
        expected = {
            "decision": "approved",
            "postprocess_script_sha256": hash_file(Path(__file__)),
            "postprocess_launcher_sha256": POSTPROCESS_LAUNCHER_SHA256,
            "compact_schema_sha256": hash_file(SCHEMA),
            "acquisition_approval_sha256": ACQUISITION_APPROVAL_SHA256,
            "acquisition_bundle_sha256": ACQUISITION_BUNDLE_SHA256,
            "bwm_loader_sha256": BWM_LOADER_SHA256,
            "region_mapping_sha256": REGION_MAPPING_SHA256,
            "region_mapping_provenance_sha256": REGION_MAPPING_PROVENANCE_SHA256,
            "download_summary_sha256": summary_sha,
            "download_log_sha256": download_log_sha,
            "session_status_bundle_sha256": status_bundle_sha,
        }
        if any(marker.get(key) != value for key, value in expected.items()):
            raise RuntimeError("postprocess approval is stale")
    return {"summary": summary, "sessions": sessions, "probes": probes, "inventory": inventory,
            "download_summary_sha256": summary_sha,
            "download_log_sha256": download_log_sha,
            "session_status_bundle_sha256": status_bundle_sha,
            "session_status_records": status_records,
            "region_lookup": region_lookup,
            "region_mapping_input": region_mapping_input,
            "acquisition_external_side_effect": side_effect_evidence}


def one_row(rows: Iterable[dict[str, str]], *, name: str) -> dict[str, str]:
    selected = [row for row in rows if row["name"] == name]
    if len(selected) != 1:
        raise RuntimeError(f"expected one {name}, found {len(selected)}")
    return selected[0]


def dataset_identity(row: dict[str, str]) -> dict[str, str]:
    return {"dataset_uuid": row["dataset_uuid"], "name": row["name"], "md5": row["md5"]}


def probe_equivalence(probe: dict[str, str], rows: list[dict[str, str]],
                      region_lookup: dict[int, str],
                      tracker: dict[str, dict[str, object]]) -> dict[str, object]:
    passing_row = one_row(rows, name="passingSpikes.table.pqt")
    metrics_row = one_row(rows, name="clusters.metrics.pqt")
    cluster_channel_row = one_row(rows, name="clusters.channels.npy")
    channel_region_row = one_row(rows, name="channels.brainLocationIds_ccf_2017.npy")
    prefix = f"probe:{probe['probe_name']}:{probe['pid']}"
    passing = pd.read_parquet(checked_dataset(passing_row, tracker, f"{prefix}:passing_spikes"))
    metrics = pd.read_parquet(checked_dataset(metrics_row, tracker, f"{prefix}:cluster_metrics"))
    cluster_channels = np.load(
        checked_dataset(cluster_channel_row, tracker, f"{prefix}:cluster_channels"),
        allow_pickle=False,
    )
    channel_regions = np.load(
        checked_dataset(channel_region_row, tracker, f"{prefix}:channel_ccf_ids"),
        allow_pickle=False,
    )
    required_passing = {"times", "clusters"}
    required_metrics = {"cluster_id", "label"}
    if not required_passing <= set(passing) or not required_metrics <= set(metrics):
        raise RuntimeError(f"good-unit tables have unexpected schema for {probe['pid']}")
    passing_ids = np.sort(passing["clusters"].astype(np.int64).unique())
    metric_ids = np.sort(metrics.loc[metrics["label"] >= UNIT_QC_THRESHOLD, "cluster_id"].astype(np.int64).unique())
    if not np.array_equal(passing_ids, metric_ids):
        raise RuntimeError(f"passingSpikes/metrics label equivalence failed: {probe['pid']}")
    if passing_ids.size == 0 or passing_ids.min() < 0 or passing_ids.max() >= len(cluster_channels):
        raise RuntimeError(f"cluster-channel index is invalid: {probe['pid']}")
    channel_indices = np.asarray(cluster_channels[passing_ids], dtype=np.int64)
    if channel_indices.min() < 0 or channel_indices.max() >= len(channel_regions):
        raise RuntimeError(f"channel-region index is invalid: {probe['pid']}")
    ccf_ids = np.asarray(channel_regions[channel_indices], dtype=np.int64)
    missing_ccf_ids = sorted({abs(int(value)) for value in ccf_ids
                              if abs(int(value)) not in region_lookup})
    if missing_ccf_ids:
        raise RuntimeError(
            f"pinned region mapping lacks CCF ids for {probe['pid']}: {missing_ccf_ids[:8]}"
        )
    acronyms = np.asarray([region_lookup[abs(int(value))] for value in ccf_ids], dtype=str)
    mapping = [[int(cluster), int(channel), int(ccf), str(acronym)]
               for cluster, channel, ccf, acronym in zip(
                   passing_ids, channel_indices, ccf_ids, acronyms, strict=True)]
    source_rows = [passing_row, metrics_row, cluster_channel_row, channel_region_row]
    evidence = {
        "pid": probe["pid"],
        "probe_name": probe["probe_name"],
        "passing_cluster_ids": passing_ids.tolist(),
        "metrics_label_ge_1_cluster_ids": metric_ids.tolist(),
        "cluster_channel_ccf_acronym": mapping,
        "input_datasets": [{"dataset_uuid": row["dataset_uuid"], "name": row["name"],
                            "md5": row["md5"]} for row in source_rows],
    }
    clusters = passing["clusters"].to_numpy(dtype=np.int64)
    local_indices = np.searchsorted(passing_ids, clusters)
    if (np.any(local_indices < 0) or np.any(local_indices >= len(passing_ids)) or
            not np.array_equal(passing_ids[local_indices], clusters)):
        raise RuntimeError(f"passingSpikes contains a non-good cluster: {probe['pid']}")
    times = passing["times"].to_numpy(dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise RuntimeError(f"passing spike times invalid: {probe['pid']}")
    return {"evidence": evidence, "times": times, "local_indices": local_indices,
            "unit_ids": np.asarray([f"{probe['pid']}:{value}" for value in passing_ids], dtype=str),
            "regions": acronyms,
            "observed_good_spike_support": (float(times[0]), float(times[-1]))}


def independent_mask_formula(trials: pd.DataFrame) -> np.ndarray:
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType"]
    if any(name not in trials for name in required):
        raise RuntimeError("trial table lacks official BWM mask fields")
    rt = trials["firstMovement_times"].to_numpy(float) - trials["stimOn_times"].to_numpy(float)
    mask = np.isfinite(rt) & (rt >= 0.08) & (rt <= 2.0)
    for name in required:
        mask &= np.isfinite(trials[name].to_numpy(float))
    return mask


def audited_official_mask(trials: pd.DataFrame, eid: str,
                          trial_row: dict[str, str]) -> tuple[np.ndarray, dict[str, object]]:
    del eid  # Session identity is bound by the caller and trial dataset UUID below.
    left = trials["contrastLeft"].to_numpy(dtype=np.float64)
    right = trials["contrastRight"].to_numpy(dtype=np.float64)
    feedback = trials["feedbackType"].to_numpy(dtype=np.float64)
    signed_contrast_percent = (
        np.nan_to_num(right, nan=0.0) - np.nan_to_num(left, nan=0.0)
    ) * 100.0
    easy = np.abs(signed_contrast_percent) >= 50.0
    easy_count = int(easy.sum())
    if easy_count == 0:
        raise RuntimeError("BWM no-truncation audit has no easy trials")
    easy_performance = float(np.mean(feedback[easy] == 1.0))
    no_truncation_passed = len(trials) > 400 and easy_performance > 0.9
    if not no_truncation_passed:
        raise RuntimeError(
            "full trial table fails intended BWM no-truncation precondition: "
            f"n_trials={len(trials)}, easy_performance={easy_performance}"
        )
    mask = independent_mask_formula(trials)
    trial_ids = np.arange(len(trials), dtype=np.int64)
    payload = {
        "protocol": "audited_pinned_bwm_formula_min_rt_0p08_max_rt_2_default_nan_v2",
        "bwm_loader_sha256": BWM_LOADER_SHA256,
        "runtime_brainwidemap_imported": False,
        "no_truncation_precondition": {
            "n_trials": int(len(trials)),
            "n_trials_strictly_greater_than_400": bool(len(trials) > 400),
            "easy_trial_definition": "abs(signed_contrast_percent)>=50",
            "easy_trial_count": easy_count,
            "easy_performance": easy_performance,
            "easy_performance_strictly_greater_than_0p9": bool(easy_performance > 0.9),
            "passed": True,
        },
        "trial_ids": trial_ids.tolist(),
        "mask": mask.astype(np.uint8).tolist(),
        "selected_trial_ids": trial_ids[mask].tolist(),
        "trial_dataset": {"dataset_uuid": trial_row["dataset_uuid"],
                          "revision": trial_row["human_revision"],
                          "md5": trial_row["md5"]},
    }
    return mask, {"sha256": canonical_sha256(payload), "payload": payload}


def wheel_displacement(times: np.ndarray, position: np.ndarray,
                       starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    if times.ndim != 1 or position.shape != times.shape or np.any(np.diff(times) < 0):
        raise RuntimeError("wheel arrays are invalid")
    result = np.full(starts.shape, np.nan, dtype=np.float64)
    for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            continue
        left, right = np.searchsorted(times, [start, stop], side="left")
        values = position[left:right]
        if len(values) >= 2:
            result[index] = float(np.abs(np.diff(values)).sum())
    return result


def motion_proxy(rows: list[dict[str, str]], events: np.ndarray,
                 tracker: dict[str, dict[str, object]],
                 analysis_view: str) -> tuple[np.ndarray, dict[str, object]]:
    for view in CAMERA_PRIORITY:
        time_rows = [row for row in rows if row["name"] == f"_ibl_{view}Camera.times.npy"]
        energy_rows = [row for row in rows if row["name"] == f"{view}Camera.ROIMotionEnergy.npy"]
        if len(time_rows) == len(energy_rows) == 1:
            times = np.load(
                checked_dataset(time_rows[0], tracker, f"{analysis_view}:camera_times"),
                allow_pickle=False,
            ).astype(np.float64)
            energy = np.load(
                checked_dataset(energy_rows[0], tracker, f"{analysis_view}:motion_energy"),
                allow_pickle=False,
            ).astype(np.float64)
            if times.ndim != 1 or energy.shape != times.shape or np.any(np.diff(times) <= 0):
                raise RuntimeError(f"camera motion-energy arrays invalid: {view}")
            proxy = np.full(events.shape, np.nan, dtype=np.float64)
            for index, event in enumerate(events):
                if not np.isfinite(event):
                    continue
                left = np.searchsorted(times, event + WINDOW[0], side="left")
                right = np.searchsorted(times, event + WINDOW[1], side="left")
                values = energy[left:right]
                values = values[np.isfinite(values)]
                if values.size:
                    proxy[index] = float(values.mean())
            metadata = {
                "proxy": "camera_roi_motion_energy_mean",
                "camera_view": view,
                "window_s": list(WINDOW),
                "window_semantics": "half-open_[event+start,event+stop)",
                "aggregation": "finite_sample_arithmetic_mean",
                "time_dataset": dataset_identity(time_rows[0]),
                "energy_dataset": dataset_identity(energy_rows[0]),
            }
            return proxy, metadata
    raise RuntimeError("no reviewed camera time/motion-energy pair is available")


def trial_tables(trials: pd.DataFrame, rows: list[dict[str, str]], eid: str,
                 trial_row: dict[str, str],
                 tracker: dict[str, dict[str, object]]) -> tuple[dict[str, pd.DataFrame], np.ndarray,
                                                     dict[str, object], dict[str, object],
                                                     dict[str, object]]:
    n = len(trials)
    def values(name: str) -> np.ndarray:
        if name not in trials:
            raise RuntimeError(f"trials missing {name}")
        return trials[name].to_numpy(dtype=np.float64)
    stim_on, movement, response = values("stimOn_times"), values("firstMovement_times"), values("response_times")
    left, right = values("contrastLeft"), values("contrastRight")
    exactly_one_side = np.logical_xor(np.isfinite(left), np.isfinite(right))
    if not exactly_one_side.all() or np.any(np.isinf(left) | np.isinf(right)):
        raise RuntimeError("each trial must have exactly one finite stimulus side")
    stimulus = np.nan_to_num(right, nan=0.0) - np.nan_to_num(left, nan=0.0)
    # Preserve the presented side even for zero-contrast trials and match the
    # existing exp11/HMM convention: left=1, right=0.
    stimulus_side = np.isfinite(left).astype(np.int64)
    wheel_time_row = one_row(rows, name="_ibl_wheel.timestamps.npy")
    wheel_position_row = one_row(rows, name="_ibl_wheel.position.npy")
    wheel_time = np.load(
        checked_dataset(wheel_time_row, tracker, "session:wheel_timestamps"),
        allow_pickle=False,
    )
    wheel_position = np.load(
        checked_dataset(wheel_position_row, tracker, "session:wheel_position"),
        allow_pickle=False,
    )
    wheel = wheel_displacement(wheel_time.astype(float), wheel_position.astype(float), stim_on, response)
    wheel_metadata = {
        "proxy": "total_absolute_wheel_displacement",
        "start_event": "stimOn_times",
        "stop_event": "response_times",
        "window_semantics": "half-open_[start_event,stop_event)",
        "aggregation": "sum_absolute_first_differences",
        "time_dataset": dataset_identity(wheel_time_row),
        "position_dataset": dataset_identity(wheel_position_row),
    }
    mask, mask_evidence = audited_official_mask(trials, eid, trial_row)
    base = pd.DataFrame({"trial_id": np.arange(n, dtype=np.int64), "stimulus": stimulus,
                         "stimulus_side": stimulus_side,
                         "choice": values("choice"), "wheel": wheel, "reward": values("feedbackType"),
                         "reaction_time": movement - stim_on,
                         "stim_on": stim_on, "first_movement": movement,
                         "timing_valid": np.isfinite(stim_on) & np.isfinite(movement) & (movement >= stim_on),
                         "official_bwm_mask": mask,
                         "block_id": np.arange(n, dtype=np.int64) // 50})
    tables, proxies = {}, {}
    for view, events in (("stimulus_pre", stim_on), ("movement_pre", movement)):
        motion_energy, metadata = motion_proxy(rows, events, tracker, view)
        table = base.copy(deep=True)
        table["motion_energy_proxy"] = motion_energy
        tables[view] = table
        proxies[view] = metadata
    return tables, mask, proxies, mask_evidence, wheel_metadata


def bin_counts(times: np.ndarray, unit_indices: np.ndarray, events: np.ndarray,
               official: np.ndarray, observed_supports: list[tuple[float, float]],
               n_units: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.arange(WINDOW[0], WINDOW[1] + BIN_SIZE / 2, BIN_SIZE, dtype=np.float64)
    counts = np.zeros((len(events), len(edges) - 1, n_units), dtype=np.int32)
    valid = np.isfinite(events) & official
    for start, stop in observed_supports:
        valid &= (events + WINDOW[0] >= start) & (events + WINDOW[1] <= stop)
    for trial in np.flatnonzero(valid):
        event = events[trial]
        left = np.searchsorted(times, event + WINDOW[0], side="left")
        right = np.searchsorted(times, event + WINDOW[1], side="left")
        relative = times[left:right] - event
        bins = np.searchsorted(edges, relative, side="right") - 1
        keep = (bins >= 0) & (bins < len(edges) - 1)
        np.add.at(counts[trial], (bins[keep], unit_indices[left:right][keep]), 1)
    return counts, valid.astype(bool), edges[:-1]


def acquisition_state(eid: str) -> dict[str, object]:
    matches = list((STATUS / "sessions").glob(f"{ACQUISITION_RUN_ID}_*_{eid}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one acquisition session status for {eid}")
    return read_json(matches[0])


def process_session(session: dict[str, str], probes: list[dict[str, str]],
                    inventory: list[dict[str, str]], work: Path,
                    dynamic_bindings: dict[str, str], region_lookup: dict[int, str],
                    region_mapping_input: dict[str, object],
                    session_status_sha256: str,
                    external_side_effect: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    eid = session["eid"]
    probes = sorted(probes, key=lambda row: (row["probe_name"], row["pid"]))
    session_rows = [row for row in inventory if row["eid"] == eid]
    state = acquisition_state(eid)
    (dataset_failure_receipts, required_dataset_failures, auxiliary_dataset_failures,
     usable_session_rows, selected_camera_view) = acquisition_dataset_failure_receipts(
        state, session_rows
    )
    if required_dataset_failures:
        failed_names = [item["source_record"]["name"] for item in required_dataset_failures]
        raise RuntimeError(f"compact-required acquisition dataset failures retained: {failed_names}")
    expected_required_inputs = expected_required_input_identities(
        usable_session_rows, probes, selected_camera_view
    )
    input_tracker: dict[str, dict[str, object]] = {}
    loader_receipts = acquisition_loader_receipts(state, probes, eid)
    loader_events = {row["source_record"]["event"] for row in loader_receipts}
    if loader_events == {"official_good_unit_loader_passed"}:
        acquisition_validation_status = "official_pinned_load_good_units_passed"
    elif loader_events == {"official_good_unit_loader_failed"}:
        acquisition_validation_status = "official_good_unit_loader_failed_retained"
    elif loader_events == {"official_good_unit_loader_skipped_missing_inventory"}:
        acquisition_validation_status = "official_good_unit_loader_skipped_missing_inventory_retained"
    else:
        acquisition_validation_status = "official_good_unit_loader_mixed_nonpass_retained"
    equivalents, all_times, all_indices, unit_ids, regions, observed_supports = [], [], [], [], [], []
    offset = 0
    for probe in probes:
        probe_rows = [row for row in usable_session_rows if row["pid"] == probe["pid"]]
        result = probe_equivalence(probe, probe_rows, region_lookup, input_tracker)
        equivalents.append(result["evidence"])
        all_times.append(result["times"])
        all_indices.append(result["local_indices"] + offset)
        unit_ids.append(result["unit_ids"])
        regions.append(result["regions"])
        observed_supports.append(result["observed_good_spike_support"])
        offset += len(result["unit_ids"])
    method = "passing_spikes_metrics_label_equivalence_v1"
    equivalence_payload = {
        "sorting_rule": "probe_name_then_pid_then_cluster_id_v1",
        "probes": equivalents,
    }
    equivalence_sha = canonical_sha256(equivalence_payload)
    times = np.concatenate(all_times)
    indices = np.concatenate(all_indices).astype(np.int64)
    order = np.argsort(times, kind="stable")
    times, indices = times[order], indices[order]
    unit_ids_array = np.concatenate(unit_ids).astype(str)
    regions_array = np.concatenate(regions).astype(str)
    trial_row = one_row(usable_session_rows, name="_ibl_trials.table.pqt")
    trials = pd.read_parquet(
        checked_dataset(trial_row, input_tracker, "session:trials_table")
    )
    tables, official, proxy_metadata, mask_evidence, wheel_metadata = trial_tables(
        trials, usable_session_rows, eid, trial_row, input_tracker
    )
    if any(value["camera_view"] != selected_camera_view
           for value in proxy_metadata.values()):
        raise RuntimeError("camera priority classification drift")
    tracked_inputs = []
    for item in sorted(input_tracker.values(), key=lambda value: str(value["dataset_uuid"])):
        tracked = dict(item)
        tracked["roles"] = sorted(str(role) for role in tracked["roles"])
        tracked_inputs.append(tracked)
    tracked_identities = [{key: item[key] for key in ("dataset_uuid", "name", "md5")}
                          for item in tracked_inputs]
    if tracked_identities != expected_required_inputs:
        raise RuntimeError("compact required input tracker differs from preregistered bundle")
    required_input_payload = {
        "protocol": "compact_required_input_bundle_v1",
        "required_exact_names": sorted(COMPACT_REQUIRED_DATASET_NAMES),
        "camera_rule": "first_intact_pair_in_left_right_body_priority",
        "selected_camera_view": selected_camera_view,
        "datasets": tracked_inputs,
    }
    required_input_sha = canonical_sha256(required_input_payload)
    expected_mask = int(float(session["official_bwm_mask_trial_count"]))
    if int(official.sum()) != expected_mask:
        raise RuntimeError(f"official mask count mismatch: {official.sum()} != {expected_mask}")
    stim_events = trials["stimOn_times"].to_numpy(float)
    move_events = trials["firstMovement_times"].to_numpy(float)
    stim_counts, stim_valid, stim_time = bin_counts(
        times, indices, stim_events, official, observed_supports, offset
    )
    move_counts, move_valid, move_time = bin_counts(
        times, indices, move_events, official, observed_supports, offset
    )
    tables["stimulus_pre"]["timing_valid"] = stim_valid
    tables["movement_pre"]["timing_valid"] = move_valid
    npz = work / "counts.npz"
    np.savez_compressed(npz, stimulus_pre_counts=stim_counts, movement_pre_counts=move_counts,
                        stimulus_pre_valid=stim_valid, movement_pre_valid=move_valid,
                        stimulus_pre_time=stim_time, movement_pre_time=move_time,
                        unit_ids=unit_ids_array, regions=regions_array,
                        trial_ids=np.arange(len(trials), dtype=np.int64))
    stimulus_csv, movement_csv = work / "stimulus_pre_trials.csv", work / "movement_pre_trials.csv"
    tables["stimulus_pre"].to_csv(stimulus_csv, index=False, float_format="%.17g")
    tables["movement_pre"].to_csv(movement_csv, index=False, float_format="%.17g")
    metadata = {"schema_version": "ibl_exp14_compact_session_v2",
                "candidate_rank": int(session["candidate_rank"]),
                "status": "complete", "error_type": "", "error": "", "eid": eid,
                "animal_id": session["subject"], "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "acquisition_bundle_sha256": ACQUISITION_BUNDLE_SHA256,
                "acquisition_approval_sha256": ACQUISITION_APPROVAL_SHA256,
                 "bwm_repository_commit": BWM_COMMIT, "bwm_loader_sha256": BWM_LOADER_SHA256,
                 "bwm_loader_relative_path": "provenance/bwm_loading.py",
                 "postprocess_launcher_sha256": POSTPROCESS_LAUNCHER_SHA256,
                 "spike_sorting_revision": SPIKE_SORTING_REVISION,
                "unit_qc_threshold": UNIT_QC_THRESHOLD, "unit_qc_applied": True,
                "acquisition_validation_status": acquisition_validation_status,
                "unit_qc_method": method, "unit_qc_equivalence_sha256": equivalence_sha,
                 "unit_qc_equivalence": equivalence_payload,
                 "acquisition_official_loader_receipts": loader_receipts,
                 "acquisition_dataset_failure_receipts": dataset_failure_receipts,
                 "auxiliary_dataset_failure_policy": {
                     "classification": "auxiliary_nonblocking_not_read",
                     "required_exact_names": sorted(COMPACT_REQUIRED_DATASET_NAMES),
                     "camera_rule": "first_intact_pair_in_left_right_body_priority",
                     "selected_camera_view": selected_camera_view,
                     "required_failure_count": 0,
                     "auxiliary_failure_count": len(auxiliary_dataset_failures),
                     "auxiliary_failure_dataset_uuids": [
                         item["source_record"]["dataset_uuid"]
                         for item in auxiliary_dataset_failures
                     ],
                 },
                 "compact_required_input_bundle": required_input_payload,
                 "compact_required_input_bundle_sha256": required_input_sha,
                 "acquisition_session_status_sha256": session_status_sha256,
                 "acquisition_external_side_effect": external_side_effect,
                 "postprocess_input_policy": {
                     "one_api_called": False,
                     "network_access": False,
                     "external_atlas_cache_used": False,
                     "data_inputs": "review_bound_staging_files_only",
                 },
                 "region_mapping_input": region_mapping_input,
                "trial_dataset_uuid": trial_row["dataset_uuid"],
                "trial_dataset_revision": trial_row["human_revision"],
                "trial_dataset_md5": trial_row["md5"],
                "official_bwm_mask_sha256": mask_evidence["sha256"],
                "official_bwm_mask_evidence": mask_evidence["payload"],
                 "download_summary_sha256": dynamic_bindings["download_summary_sha256"],
                 "download_log_sha256": dynamic_bindings["download_log_sha256"],
                 "session_status_bundle_sha256": dynamic_bindings["session_status_bundle_sha256"],
                "bin_size_s": BIN_SIZE, "window_s": list(WINDOW), "n_time_bins": 25,
                "window_semantics": "half-open_[start,stop)",
                "time_axis_semantics": "left_bin_edges",
                "trial_count": len(trials), "official_bwm_mask_count": int(official.sum()),
                 "unit_count": int(offset), "motion_energy_proxy": proxy_metadata,
                 "wheel_displacement": wheel_metadata,
                "cv_block_policy": {"method": "fixed_trial_id_chunks_v1",
                                    "block_size_trials": 50, "is_true_context": False},
                "observed_good_spike_support_s": [
                    {"pid": probe["pid"], "probe_name": probe["probe_name"],
                     "start": support[0], "stop": support[1],
                     "semantics": "conservative_observed_good_spike_support_not_recording_interval"}
                    for probe, support in zip(probes, observed_supports, strict=True)
                ],
                "acquisition_session_status": state["status"], "created_at_utc": now()}
    metadata_path = work / "metadata.json"
    write_json(metadata_path, metadata)
    row = {"candidate_rank": int(session["candidate_rank"]), "eid": eid,
           "animal_id": session["subject"], "status": "complete", "error_type": "", "error": "",
           "npz_path": "", "npz_sha256": "", "stimulus_trials_path": "",
           "stimulus_trials_sha256": "", "movement_trials_path": "", "movement_trials_sha256": "",
           "metadata_path": "", "metadata_sha256": "", "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
           "acquisition_bundle_sha256": ACQUISITION_BUNDLE_SHA256, "bwm_repository_commit": BWM_COMMIT,
           "spike_sorting_revision": SPIKE_SORTING_REVISION, "unit_qc_threshold": UNIT_QC_THRESHOLD,
           "unit_qc_applied": True, "acquisition_validation_status": acquisition_validation_status,
           "unit_qc_method": method, "unit_qc_equivalence_sha256": equivalence_sha,
           "trial_dataset_uuid": trial_row["dataset_uuid"],
           "trial_dataset_revision": trial_row["human_revision"],
           "trial_dataset_md5": trial_row["md5"],
           "official_bwm_mask_sha256": mask_evidence["sha256"],
           "download_summary_sha256": dynamic_bindings["download_summary_sha256"],
           "session_status_bundle_sha256": dynamic_bindings["session_status_bundle_sha256"],
           "region_mapping_path": "provenance/iblatlas_allen_structure_tree.csv",
           "region_mapping_sha256": REGION_MAPPING_SHA256,
           "region_mapping_provenance_path": "provenance/region_mapping_provenance.json",
           "region_mapping_provenance_sha256": REGION_MAPPING_PROVENANCE_SHA256,
           "iblatlas_version": IBLATLAS_VERSION,
           "iblatlas_source_commit": "unavailable_pypi_distribution_direct_url_absent",
           "iblatlas_regions_source_sha256": IBLATLAS_REGIONS_SOURCE_SHA256}
    return row, {"npz": npz, "stimulus": stimulus_csv, "movement": movement_csv,
                 "metadata": metadata_path}


def manifest_fields() -> list[str]:
    return ["candidate_rank", "eid", "animal_id", "status", "error_type", "error",
            "npz_path", "npz_sha256", "stimulus_trials_path", "stimulus_trials_sha256",
            "movement_trials_path", "movement_trials_sha256", "metadata_path", "metadata_sha256",
            "source_manifest_sha256", "acquisition_bundle_sha256", "bwm_repository_commit",
            "spike_sorting_revision", "unit_qc_threshold", "unit_qc_applied",
            "acquisition_validation_status", "unit_qc_method", "unit_qc_equivalence_sha256",
            "trial_dataset_uuid", "trial_dataset_revision", "trial_dataset_md5",
            "official_bwm_mask_sha256", "download_summary_sha256",
            "session_status_bundle_sha256", "region_mapping_path", "region_mapping_sha256",
            "region_mapping_provenance_path", "region_mapping_provenance_sha256",
            "iblatlas_version", "iblatlas_source_commit", "iblatlas_regions_source_sha256"]


def run(frozen: dict[str, object]) -> dict[str, object]:
    if OUTPUT.exists():
        raise RuntimeError(f"compact output already exists; fail-closed: {OUTPUT}")
    OUTPUT.mkdir(mode=0o750)
    (OUTPUT / ".work").mkdir()
    (OUTPUT / "sessions").mkdir()
    (OUTPUT / "failed").mkdir()
    provenance_dir = OUTPUT / "provenance"
    provenance_dir.mkdir()
    mapping_copy = provenance_dir / "iblatlas_allen_structure_tree.csv"
    provenance_copy = provenance_dir / "region_mapping_provenance.json"
    bwm_loader_copy = provenance_dir / "bwm_loading.py"
    shutil.copyfile(REGION_MAPPING, mapping_copy)
    shutil.copyfile(REGION_MAPPING_PROVENANCE, provenance_copy)
    shutil.copyfile(PINNED_BWM_LOADER, bwm_loader_copy)
    if (hash_file(mapping_copy) != REGION_MAPPING_SHA256 or
            hash_file(provenance_copy) != REGION_MAPPING_PROVENANCE_SHA256 or
            hash_file(bwm_loader_copy) != BWM_LOADER_SHA256):
        raise RuntimeError("compact frozen provenance copy failed verification")
    probes_by_eid: dict[str, list[dict[str, str]]] = {}
    for probe in frozen["probes"]:
        probes_by_eid.setdefault(probe["eid"], []).append(probe)
    rows, artifact_hashes = [], {}
    dynamic_bindings = {
        "download_summary_sha256": frozen["download_summary_sha256"],
        "download_log_sha256": frozen["download_log_sha256"],
        "session_status_bundle_sha256": frozen["session_status_bundle_sha256"],
    }
    status_sha_by_eid = {record["eid"]: record["sha256"]
                         for record in frozen["session_status_records"]}
    for session in sorted(frozen["sessions"], key=lambda row: int(row["candidate_rank"])):
        eid = session["eid"]
        work = OUTPUT / ".work" / f"{int(session['candidate_rank']):03d}_{eid}"
        destination = OUTPUT / "sessions" / eid
        try:
            work.mkdir()
            row, artifacts = process_session(
                session, probes_by_eid[eid], frozen["inventory"], work,
                dynamic_bindings, frozen["region_lookup"], frozen["region_mapping_input"],
                status_sha_by_eid[eid], frozen["acquisition_external_side_effect"]
            )
            os.replace(work, destination)
            moved = {name: destination / path.name for name, path in artifacts.items()}
            row.update(npz_path=str(moved["npz"].relative_to(OUTPUT)), npz_sha256=hash_file(moved["npz"]),
                       stimulus_trials_path=str(moved["stimulus"].relative_to(OUTPUT)),
                       stimulus_trials_sha256=hash_file(moved["stimulus"]),
                       movement_trials_path=str(moved["movement"].relative_to(OUTPUT)),
                       movement_trials_sha256=hash_file(moved["movement"]),
                       metadata_path=str(moved["metadata"].relative_to(OUTPUT)),
                       metadata_sha256=hash_file(moved["metadata"]))
            artifact_hashes[eid] = {name: hash_file(path) for name, path in moved.items()}
        except Exception as error:
            error_record = {"eid": eid, "error_type": type(error).__name__, "error": str(error),
                            "traceback": traceback.format_exc(limit=12), "timestamp_utc": now()}
            validation_status = "acquisition_status_unreadable_retained"
            try:
                state = acquisition_state(eid)
                dataset_failed = any(item["event"] == "dataset_failed" for item in state.get("datasets", []))
                loader_failed = any(item.get("event") != "official_good_unit_loader_passed"
                                    for item in state.get("probe_validations", []))
                if dataset_failed:
                    validation_status = "acquisition_dataset_failed_retained"
                elif loader_failed:
                    validation_status = "official_good_unit_loader_failed_retained"
                else:
                    validation_status = "acquisition_failure_retained"
            except Exception as state_error:
                error_record["acquisition_state_error"] = {
                    "error_type": type(state_error).__name__, "error": str(state_error)
                }
            row = {field: "" for field in manifest_fields()}
            row.update(candidate_rank=int(session["candidate_rank"]), eid=eid, animal_id=session["subject"],
                       status="failed", error_type=type(error).__name__, error=str(error),
                       source_manifest_sha256=SOURCE_MANIFEST_SHA256,
                       acquisition_bundle_sha256=ACQUISITION_BUNDLE_SHA256,
                       bwm_repository_commit=BWM_COMMIT, spike_sorting_revision=SPIKE_SORTING_REVISION,
                       unit_qc_threshold=UNIT_QC_THRESHOLD, unit_qc_applied=False,
                       acquisition_validation_status=validation_status,
                       unit_qc_method="none", unit_qc_equivalence_sha256="",
                       trial_dataset_uuid=session["dataset_uuid"],
                       trial_dataset_revision=session["dataset_revision"],
                       trial_dataset_md5=session["dataset_hash"],
                       official_bwm_mask_sha256="",
                       download_summary_sha256=dynamic_bindings["download_summary_sha256"],
                       session_status_bundle_sha256=dynamic_bindings["session_status_bundle_sha256"],
                       region_mapping_path="provenance/iblatlas_allen_structure_tree.csv",
                       region_mapping_sha256=REGION_MAPPING_SHA256,
                       region_mapping_provenance_path="provenance/region_mapping_provenance.json",
                       region_mapping_provenance_sha256=REGION_MAPPING_PROVENANCE_SHA256,
                       iblatlas_version=IBLATLAS_VERSION,
                       iblatlas_source_commit="unavailable_pypi_distribution_direct_url_absent",
                       iblatlas_regions_source_sha256=IBLATLAS_REGIONS_SOURCE_SHA256)
            retention_errors = []
            try:
                location = work if work.exists() else destination if destination.exists() else None
                if location is not None:
                    error_path = location / "error.json"
                    if not error_path.exists():
                        write_json(error_path, error_record)
                    failed_target = OUTPUT / "failed" / f"{int(session['candidate_rank']):03d}_{eid}"
                    if location != failed_target and not failed_target.exists():
                        os.replace(location, failed_target)
            except Exception as retention_error:
                retention_errors.append(f"{type(retention_error).__name__}: {retention_error}")
            if retention_errors:
                row["error"] = f"{row['error']} | retention_error={' ; '.join(retention_errors)}"
        rows.append(row)
    if len(rows) != 20 or len({row["eid"] for row in rows}) != 20:
        raise RuntimeError("compact failure retention did not preserve all 20 sessions")
    manifest = OUTPUT / "compact_manifest.csv"
    with manifest.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields())
        writer.writeheader(); writer.writerows(rows)
    bundle = {"schema_version": "ibl_exp14_compact_bundle_v2", "created_at_utc": now(),
              "compact_manifest_sha256": hash_file(manifest), "artifact_sha256": artifact_hashes,
              "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
              "acquisition_bundle_sha256": ACQUISITION_BUNDLE_SHA256,
              "acquisition_approval_sha256": ACQUISITION_APPROVAL_SHA256,
              "download_summary_sha256": dynamic_bindings["download_summary_sha256"],
              "download_log_sha256": dynamic_bindings["download_log_sha256"],
              "session_status_bundle_sha256": dynamic_bindings["session_status_bundle_sha256"],
              "acquisition_external_side_effect": frozen["acquisition_external_side_effect"],
              "postprocess_script_sha256": hash_file(Path(__file__)),
              "postprocess_launcher_sha256": POSTPROCESS_LAUNCHER_SHA256,
              "compact_schema_sha256": hash_file(SCHEMA),
              "bwm_loader_sha256": BWM_LOADER_SHA256,
              "bwm_loader_relative_path": str(bwm_loader_copy.relative_to(OUTPUT)),
              "region_mapping_path": str(mapping_copy.relative_to(OUTPUT)),
              "region_mapping_sha256": hash_file(mapping_copy),
              "region_mapping_provenance_path": str(provenance_copy.relative_to(OUTPUT)),
              "region_mapping_provenance_sha256": hash_file(provenance_copy),
              "iblatlas_version": IBLATLAS_VERSION,
              "iblatlas_source_commit": "unavailable_pypi_distribution_direct_url_absent",
              "iblatlas_regions_source_sha256": IBLATLAS_REGIONS_SOURCE_SHA256,
              "complete_sessions": sum(row["status"] == "complete" for row in rows),
              "failed_sessions": sum(row["status"] == "failed" for row in rows)}
    write_json(OUTPUT / "compact_bundle.json", bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-inputs-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        frozen = validate_inputs(require_postprocess_review=args.execute)
        if args.validate_inputs_only:
            print(json.dumps({"status": "inputs_valid", "sessions": len(frozen["sessions"]),
                              "probes": len(frozen["probes"]), "datasets": len(frozen["inventory"]),
                              "postprocess_review_required": True}, sort_keys=True))
            return 0
        print(json.dumps(run(frozen), sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "error": str(error), "traceback": traceback.format_exc(limit=12)},
                         sort_keys=True), file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
