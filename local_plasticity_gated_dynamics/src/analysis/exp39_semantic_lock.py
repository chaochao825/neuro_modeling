"""Independent integrity and semantic-lock checks for frozen Exp39 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from experiments.exp39_factorized_uncertainty import (
    _levels,
    _stream_config,
    summarize,
)
from src.analysis.exp39_claim_boundary import (
    cellwise_utility,
    cross_loading,
    headroom_retention,
    timing_utility,
)
from src.tasks.factorized_uncertainty import (
    UncertaintyTape,
    all_factorial_cells,
    generate_uncertainty_tape,
)


_MANIFEST_LINE = re.compile(r"(?P<digest>[0-9a-f]{64})  (?P<path>.+)")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def portable_tape_fingerprint(
    tape: UncertaintyTape, *, decimals: int = 10
) -> str:
    """Hash a tape after removing disclosed cross-platform float tail noise.

    NumPy's normal transform can differ by a few ULP across operating systems
    even at the same NumPy version.  Exact artifact bytes remain protected by
    the frozen manifests.  This separate fingerprint binds regenerated tapes
    to their scientific values at a fixed decimal precision without claiming
    that the original execution environment was recorded.
    """

    if isinstance(decimals, bool) or not 0 <= int(decimals) <= 15:
        raise ValueError("decimals must be an integer in [0, 15]")
    decimals = int(decimals)
    hasher = hashlib.sha256()
    fields = (
        "observations",
        "latent",
        "hazard",
        "process_variance",
        "observation_variance",
        "jump_flags",
        "sequence_ids",
        "block_ids",
        "cells",
    )
    for field in fields:
        value = np.asarray(getattr(tape, field))
        hasher.update(field.encode("ascii"))
        hasher.update(np.asarray(value.shape, dtype="<i8").tobytes())
        if np.issubdtype(value.dtype, np.floating):
            canonical = np.round(value.astype(np.float64), decimals=decimals)
            canonical[canonical == 0.0] = 0.0
            hasher.update(canonical.astype("<f8", copy=False).tobytes())
        elif np.issubdtype(value.dtype, np.integer):
            hasher.update(value.astype("<i8", copy=False).tobytes())
        elif np.issubdtype(value.dtype, np.bool_):
            hasher.update(value.astype(np.uint8, copy=False).tobytes())
        else:
            for item in value.astype(str):
                encoded = item.encode("utf-8")
                hasher.update(len(encoded).to_bytes(8, "little"))
                hasher.update(encoded)
    return hasher.hexdigest()


def _aggregate_tape_fingerprints(
    values: Sequence[tuple[int, str]],
) -> str:
    hasher = hashlib.sha256()
    for seed, digest in values:
        hasher.update(int(seed).to_bytes(8, "little", signed=False))
        hasher.update(bytes.fromhex(digest))
    return hasher.hexdigest()


def _inside(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _resolve_inside(root: Path, relative: object, *, label: str) -> Path:
    value = Path(str(relative))
    if value.is_absolute() or ".." in value.parts:
        raise RuntimeError(f"unsafe {label} path: {relative}")
    resolved = (root / value).resolve()
    if not _inside(root, resolved):
        raise RuntimeError(f"{label} path leaves project root: {relative}")
    return resolved


def verify_sha256_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    required_entry_count: int | None = None,
) -> dict[str, str]:
    """Strictly parse and verify a sha256sum-style manifest."""

    root = project_root.resolve()
    manifest = manifest_path.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            raise ValueError(f"blank manifest line {line_number}")
        match = _MANIFEST_LINE.fullmatch(raw_line)
        if match is None:
            raise ValueError(f"invalid manifest line {line_number}")
        relative = match.group("path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        if relative in entries:
            raise ValueError(f"duplicate manifest path: {relative}")
        path = (root / relative_path).resolve()
        if not _inside(root, path):
            raise ValueError(f"manifest path leaves project root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = match.group("digest")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"manifest digest mismatch: {relative}")
        entries[relative] = expected
    if not entries:
        raise ValueError("manifest must contain at least one entry")
    if required_entry_count is not None and len(entries) != required_entry_count:
        raise RuntimeError(
            f"manifest has {len(entries)} entries; expected {required_entry_count}"
        )
    return entries


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _verify_hash_receipt(project_root: Path, contract: Mapping[str, Any]) -> int:
    receipt_path = _resolve_inside(
        project_root, contract["path"], label="hash receipt"
    )
    if not _inside(project_root, receipt_path) or not receipt_path.is_file():
        raise RuntimeError(f"unsafe or missing hash receipt: {contract['path']}")
    if sha256_file(receipt_path) != str(contract["sha256"]):
        raise RuntimeError(f"hash receipt changed: {contract['path']}")
    receipt = _load_json(receipt_path)
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"hash receipt has no files: {contract['path']}")
    for relative, expected in files.items():
        path = (project_root / str(relative)).resolve()
        if not _inside(project_root, path) or not path.is_file():
            raise RuntimeError(f"unsafe or missing receipt file: {relative}")
        if sha256_file(path) != str(expected):
            raise RuntimeError(f"hash receipt mismatch: {relative}")
    return len(files)


def _assert_close(observed: float, expected: float, *, label: str) -> None:
    if not np.isclose(float(observed), float(expected), rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"semantic fingerprint mismatch for {label}: "
            f"observed={observed!r}, expected={expected!r}"
        )


def _assert_json_close(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise RuntimeError(f"JSON keys differ for {label}")
        for key in expected:
            _assert_json_close(
                observed[key], expected[key], label=f"{label}.{key}"
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise RuntimeError(f"JSON list shape differs for {label}")
        for index, (observed_item, expected_item) in enumerate(
            zip(observed, expected, strict=True)
        ):
            _assert_json_close(
                observed_item, expected_item, label=f"{label}[{index}]"
            )
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if observed != expected or type(observed) is not type(expected):
            raise RuntimeError(f"JSON value differs for {label}")
        return
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        _assert_close(float(observed), float(expected), label=label)
        return
    if observed != expected:
        raise RuntimeError(f"JSON value differs for {label}")


def _assert_aggregate_matches_shards(
    result_dir: Path,
    *,
    seeds: Sequence[int],
    filename: str,
    sort_columns: Sequence[str],
    dtype: Mapping[str, Any] | None = None,
) -> int:
    shards = []
    for seed in seeds:
        path = result_dir / f"seed_{int(seed)}" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append(pd.read_csv(path, dtype=dtype))
    concatenated = pd.concat(shards, ignore_index=True).sort_values(
        list(sort_columns), kind="stable"
    ).reset_index(drop=True)
    aggregate_path = result_dir / filename
    aggregate = pd.read_csv(aggregate_path, dtype=dtype).sort_values(
        list(sort_columns), kind="stable"
    ).reset_index(drop=True)
    try:
        assert_frame_equal(concatenated, aggregate, check_exact=True)
    except AssertionError as error:
        raise RuntimeError(
            f"{filename} does not exactly equal its per-seed shards"
        ) from error
    return int(len(aggregate))


def validate_exp39_semantic_lock(
    project_root: Path,
    result_dir: Path,
    lock_path: Path,
) -> dict[str, Any]:
    """Verify byte integrity, shard lineage, and disclosed numeric boundaries."""

    project = project_root.resolve()
    result = result_dir.resolve()
    lock = _load_json(lock_path.resolve())
    if lock.get("lock_version") != "exp39_semantic_lock_v1":
        raise RuntimeError("unsupported Exp39 semantic-lock version")
    if lock.get("claim_upgrade_allowed") is not False:
        raise RuntimeError("semantic audit must remain claim-ineligible")
    expected_result_path = _resolve_inside(
        project, lock.get("result_relative_path"), label="formal result"
    )
    if result != expected_result_path:
        raise RuntimeError("result_dir is not the formal result bound by the lock")
    manifests = lock.get("manifests")
    if not isinstance(manifests, dict):
        raise RuntimeError("semantic lock lacks manifest contracts")
    verified_manifests: dict[str, int] = {}
    for name in ("formal_artifacts", "publication_amendment"):
        contract = manifests.get(name)
        if not isinstance(contract, dict):
            raise RuntimeError(f"semantic lock lacks {name} manifest")
        path = _resolve_inside(project, contract["path"], label=name)
        if sha256_file(path) != str(contract["sha256"]):
            raise RuntimeError(f"{name} manifest file was changed")
        entries = verify_sha256_manifest(
            path,
            project_root=project,
            required_entry_count=int(contract["entry_count"]),
        )
        verified_manifests[name] = len(entries)
    receipt_contracts = lock.get("hash_receipts")
    if not isinstance(receipt_contracts, dict):
        raise RuntimeError("semantic lock lacks source receipt contracts")
    verified_receipts = {
        name: _verify_hash_receipt(project, contract)
        for name, contract in receipt_contracts.items()
    }

    config = _load_json(result / "config.json")
    seeds = tuple(map(int, config["seeds"]))
    expected = lock["expected_result"]
    if tuple(seeds) != tuple(map(int, expected["seeds"])):
        raise RuntimeError("formal seed list differs from the semantic lock")
    if sha256_file(result / "config.json") != str(expected["config_sha256"]):
        raise RuntimeError("formal result config differs from the semantic lock")
    block_rows = _assert_aggregate_matches_shards(
        result,
        seeds=seeds,
        filename="block_metrics.csv",
        sort_columns=("seed", "method", "block_id"),
        dtype={"cell": str},
    )
    selection_rows = _assert_aggregate_matches_shards(
        result,
        seeds=seeds,
        filename="selection_audit.csv",
        sort_columns=("seed", "selection_family", "selection_nll"),
    )
    if block_rows != int(expected["n_block_rows"]):
        raise RuntimeError("formal block-row count differs from the semantic lock")
    if selection_rows != int(expected["n_selection_rows"]):
        raise RuntimeError("formal selection-row count differs from the semantic lock")

    exact_tape_digest_mismatches: list[dict[str, Any]] = []
    portable_fingerprints: dict[str, list[tuple[int, str]]] = {
        "fit": [],
        "test": [],
    }
    portable_contract = expected.get("portable_tape_semantic_fingerprint")
    if not isinstance(portable_contract, dict):
        raise RuntimeError("semantic lock lacks portable tape fingerprints")
    decimals = int(portable_contract.get("quantization_decimals", -1))

    blocks = pd.read_csv(result / "block_metrics.csv", dtype={"cell": str})
    selection = pd.read_csv(result / "selection_audit.csv")
    selected_flags = selection["selected"]
    if not pd.api.types.is_bool_dtype(selected_flags.dtype):
        normalized = selected_flags.astype(str).str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise RuntimeError("selection flags are not strict booleans")
        selected_flags = normalized.eq("true")
    for (seed, family), candidates in selection.groupby(
        ["seed", "selection_family"], sort=True
    ):
        flags = selected_flags.loc[candidates.index]
        if int(flags.sum()) != 1:
            raise RuntimeError(f"seed {seed}/{family} lacks one selected candidate")
        selected_nll = float(candidates.loc[flags, "selection_nll"].item())
        minimum_nll = float(candidates["selection_nll"].min())
        _assert_close(
            selected_nll,
            minimum_nll,
            label=f"selection_argmin:{seed}:{family}",
        )

    levels = _levels(config)
    stream = _stream_config(config)
    for seed in seeds:
        metadata = _load_json(result / f"seed_{seed}" / "metadata.json")
        fit_tape = generate_uncertainty_tape(
            seed=seed,
            split="fit_single_factor",
            cells=tuple(config["partitions"]["fit_cells"]),
            levels=levels,
            config=stream,
        )
        test_tape = generate_uncertainty_tape(
            seed=seed,
            split="test_full_factorial",
            cells=all_factorial_cells(),
            levels=levels,
            config=stream,
        )
        for split, tape in (("fit", fit_tape), ("test", test_tape)):
            expected_digest = metadata.get(f"{split}_tape_digest")
            if not isinstance(expected_digest, str) or len(expected_digest) != 64:
                raise RuntimeError(f"seed {seed} has an invalid {split}-tape digest")
            if expected_digest != tape.digest:
                exact_tape_digest_mismatches.append(
                    {
                        "seed": seed,
                        "split": split,
                        "stored": expected_digest,
                        "regenerated": tape.digest,
                    }
                )
            portable_fingerprints[split].append(
                (seed, portable_tape_fingerprint(tape, decimals=decimals))
            )
        seed_blocks = blocks.loc[blocks["seed"] == seed]
        if set(seed_blocks["test_tape_digest"].astype(str)) != {
            metadata["test_tape_digest"]
        }:
            raise RuntimeError(f"seed {seed} block rows use a different test tape")
    portable_aggregate = {
        split: _aggregate_tape_fingerprints(values)
        for split, values in portable_fingerprints.items()
    }
    for split, observed in portable_aggregate.items():
        if observed != portable_contract.get(f"{split}_aggregate_sha256"):
            raise RuntimeError(
                f"portable {split}-tape semantic fingerprint changed"
            )

    replay_seed_metrics, replay_comparisons, replay_tracking, replay_summary = (
        summarize(blocks, config=config)
    )
    replay_tables = {
        "seed_metrics.csv": replay_seed_metrics,
        "comparisons_and_clamps.csv": replay_comparisons,
        "parameter_tracking.csv": replay_tracking,
    }
    for filename, replay_frame in replay_tables.items():
        stored_frame = pd.read_csv(result / filename)
        try:
            assert_frame_equal(
                stored_frame.reset_index(drop=True),
                replay_frame.reset_index(drop=True),
                check_exact=False,
                check_dtype=False,
                rtol=0.0,
                atol=1e-12,
            )
        except AssertionError as error:
            raise RuntimeError(f"summary-derived table changed: {filename}") from error
    stored_summary = _load_json(result / "summary.json")
    _assert_json_close(stored_summary, replay_summary, label="full_summary_replay")
    _, cells = cellwise_utility(blocks)
    _, loadings = cross_loading(blocks)
    timing = timing_utility(blocks)
    headroom = headroom_retention(blocks)
    for item in expected["cellwise_nll_gain"]:
        observed = cells.loc[
            (cells["comparison"] == item["comparison"])
            & (cells["cell"] == item["cell"])
        ]
        if len(observed) != 1:
            raise RuntimeError("semantic cell fingerprint is not uniquely addressable")
        _assert_close(
            observed["mean_nll_gain"].item(),
            item["value"],
            label=f"{item['comparison']}:{item['cell']}",
        )
        if int(observed["positive_seeds"].item()) != int(item["positive_seeds"]):
            raise RuntimeError("semantic cell positive-seed count changed")
    for item in expected["cross_loading"]:
        observed = loadings.loc[
            (loadings["estimated_factor"] == item["estimated_factor"])
            & (loadings["true_factor"] == item["true_factor"])
        ]
        if len(observed) != 1:
            raise RuntimeError("semantic loading fingerprint is not unique")
        _assert_close(
            observed["mean_log_response"].item(),
            item["value"],
            label=f"loading:{item['estimated_factor']}<-{item['true_factor']}",
        )
    for item in expected["timing_nll_gain"]:
        observed = timing.loc[
            (timing["panel"] == item["panel"])
            & (timing["comparison"] == item["comparison"])
            & (timing["endpoint"] == item["endpoint"])
        ]
        if len(observed) != 1:
            raise RuntimeError("semantic timing fingerprint is not unique")
        _assert_close(
            observed["mean_nll_gain"].item(),
            item["value"],
            label=(
                f"timing:{item['panel']}:{item['comparison']}:"
                f"{item['endpoint']}"
            ),
        )
    for name, value in expected["headroom"].items():
        _assert_close(headroom[name], value, label=f"headroom:{name}")

    return {
        "audit_status": "passed",
        "lock_version": lock["lock_version"],
        "claim_upgrade_allowed": False,
        "formal_result_introduced_by_commit": lock[
            "formal_result_introduced_by_commit"
        ],
        "n_seeds": len(seeds),
        "n_block_rows": block_rows,
        "n_selection_rows": selection_rows,
        "verified_manifest_entries": verified_manifests,
        "verified_hash_receipt_files": verified_receipts,
        "aggregate_shard_identity": "passed",
        "fit_test_tape_semantic_replay": "passed",
        "fit_test_tape_exact_replay": (
            "passed"
            if not exact_tape_digest_mismatches
            else "platform_float_tail_variance_disclosed"
        ),
        "exact_tape_digest_mismatches": exact_tape_digest_mismatches,
        "portable_tape_semantic_fingerprint": portable_aggregate,
        "selection_argmin_replay": "passed",
        "full_summary_replay": "passed",
        "summary_derived_tables_replay": "passed",
        "numeric_semantic_fingerprint": "passed",
        "claim_boundary": lock["claim_boundary"],
    }


__all__ = [
    "sha256_file",
    "portable_tape_fingerprint",
    "validate_exp39_semantic_lock",
    "verify_sha256_manifest",
]
