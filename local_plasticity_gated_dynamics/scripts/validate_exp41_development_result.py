#!/usr/bin/env python3
"""Independent fail-closed audit for an Exp41 development artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp41_matched_identifiability import (
    EXPERIMENT,
    FROZEN_EXP39_DEPENDENCY,
    METHODS,
    PROFILE,
    _stream_config,
    summarize,
    validate_config,
)
from src.tasks.matched_uncertainty import generate_matched_uncertainty_tape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVILEGED = {
    "generator_supported_seen_regime_imm",
    "dynamic_qr_oracle",
}
REQUIRED_ROOT_FILES = {
    "block_metrics.csv",
    "comparisons.csv",
    "config.json",
    "environment.json",
    "failed_seeds.csv",
    "failures.json",
    "manifest.json",
    "matched_pair_separation.csv",
    "planned_conditions.json",
    "report.md",
    "run.log",
    "seed_metrics.csv",
    "selection_audit.csv",
    "status.json",
    "summary.json",
}
SOURCE_FILES = {
    FROZEN_EXP39_DEPENDENCY,
    "src/models/autocovariance_uncertainty_filter.py",
    "src/tasks/matched_uncertainty.py",
    "src/utils/reproducibility.py",
    "experiments/exp41_matched_identifiability.py",
}


def _strict_payload(path: Path) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs
    )


def _strict_json(path: Path) -> dict[str, Any]:
    payload = _strict_payload(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _safe_artifact_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
        raise RuntimeError(f"unsafe manifest artifact path: {relative}")
    unresolved = root / candidate
    resolved = unresolved.resolve()
    if root not in resolved.parents or unresolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"missing, linked, or out-of-tree artifact: {relative}")
    return resolved


def _git_blob(commit: str, project_relative: str) -> bytes:
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError(f"invalid run-start commit: {commit}")
    repository = Path(
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    prefix = PROJECT_ROOT.resolve().relative_to(repository).as_posix()
    object_path = f"{prefix}/{project_relative}" if prefix != "." else project_relative
    return subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{object_path}"],
        check=True,
        capture_output=True,
    ).stdout


def _assert_frame(actual: pd.DataFrame, expected: pd.DataFrame, *, name: str) -> None:
    try:
        assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise RuntimeError(f"{name} does not replay") from error


def _assert_nested_close(actual: Any, expected: Any, *, path: str = "summary") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"{path} keys do not replay")
        for key in expected:
            _assert_nested_close(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"{path} list does not replay")
        for index, value in enumerate(expected):
            _assert_nested_close(actual[index], value, path=f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not np.isclose(
            float(actual), expected, rtol=1e-12, atol=1e-12
        ):
            raise RuntimeError(f"{path} numeric value does not replay")
        return
    if actual != expected:
        raise RuntimeError(f"{path} does not replay: {actual!r} != {expected!r}")


def _boolean_column(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.map({"True": True, "False": False, "true": True, "false": False})
    if mapped.isna().any():
        raise RuntimeError(f"{name} is not a strict boolean column")
    return mapped.astype(bool)


def _candidate_count(config: dict[str, Any]) -> int:
    selection = config["selection"]
    fixed = selection["fixed_jump_grid"]
    return (
        len(fixed["process_variance"]) * len(fixed["observation_variance"])
        + len(selection["online_em_process_rate_grid"])
        * len(selection["online_em_observation_rate_grid"])
        + len(selection["total_variance_decay_grid"])
        * len(selection["total_variance_prior_mass_grid"])
        * len(selection["total_variance_q_fraction_grid"])
        + len(selection["autocovariance_decay_grid"])
        * len(selection["autocovariance_prior_mass_grid"])
        + len(selection["imm_switch_grid"])
        + 1
    )


def validate_result(
    result_dir: Path,
    *,
    require_clean_run_start: bool = True,
) -> dict[str, Any]:
    """Validate completeness, provenance, causal pairing, and replay semantics."""

    root = result_dir.resolve()
    if not root.is_dir():
        raise RuntimeError(f"missing result directory: {root}")
    missing = sorted(name for name in REQUIRED_ROOT_FILES if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"development package is missing files: {missing}")
    if list(root.rglob("*.tmp")):
        raise RuntimeError("development package contains unfinished temporary files")

    config = _strict_json(root / "config.json")
    validate_config(config)
    seeds = tuple(map(int, config["seeds"]))
    environment = _strict_json(root / "environment.json")
    planned = _strict_json(root / "planned_conditions.json")
    status = _strict_json(root / "status.json")
    summary = _strict_json(root / "summary.json")
    manifest = _strict_json(root / "manifest.json")

    if status != {
        "claim_upgrade_allowed": False,
        "development_only": True,
        "n_complete_seeds": len(seeds),
        "n_failed_seeds": 0,
        "n_planned_seeds": len(seeds),
        "status": "complete",
        "verdict": "inconclusive",
    }:
        raise RuntimeError("root status is not a complete claim-ineligible development run")
    failures = _strict_payload(root / "failures.json")
    if failures != [] or not pd.read_csv(root / "failed_seeds.csv").empty:
        raise RuntimeError("development package contains failed seeds")
    if summary.get("claim_eligible") is not False or summary.get("verdict") != "inconclusive":
        raise RuntimeError("summary crossed the development claim boundary")
    if summary.get("budget_matched") is not False or summary.get(
        "development_go_gate_satisfied"
    ) is not False:
        raise RuntimeError("unmatched-budget development artifact reported a go decision")
    if planned.get("methods") != list(METHODS) or planned.get("seeds") != list(seeds):
        raise RuntimeError("planned method or seed panel differs from the config")
    if planned.get("budget_matched") is not False or planned.get(
        "claim_upgrade_allowed"
    ) is not False:
        raise RuntimeError("planned conditions crossed the development boundary")

    if manifest.get("experiment") != EXPERIMENT or manifest.get("profile") != PROFILE:
        raise RuntimeError("manifest experiment/profile mismatch")
    if manifest.get("status") != "complete" or manifest.get(
        "claim_upgrade_allowed"
    ) is not False:
        raise RuntimeError("manifest is not a complete claim-ineligible run")
    if manifest.get("git") != environment.get("git"):
        raise RuntimeError("manifest and run-start Git snapshots disagree")
    if require_clean_run_start and manifest["git"].get("dirty") is not False:
        raise RuntimeError("run-start Git snapshot was dirty")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("manifest artifacts must be an object")
    actual_relative = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(artifacts) != actual_relative:
        raise RuntimeError("manifest artifact inventory is incomplete or contains extras")
    for relative, expected_hash in artifacts.items():
        path = _safe_artifact_path(root, relative)
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"artifact hash mismatch: {relative}")

    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != SOURCE_FILES:
        raise RuntimeError("manifest source inventory is incomplete")
    commit = str(manifest["git"].get("commit", ""))
    for relative, expected_hash in source_hashes.items():
        if _sha256_bytes(_git_blob(commit, relative)) != expected_hash:
            raise RuntimeError(f"run-start source hash mismatch: {relative}")
    implementation_hash = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if manifest.get("implementation_sha256") != implementation_hash:
        raise RuntimeError("implementation hash does not bind the source inventory")

    root_blocks = pd.read_csv(root / "block_metrics.csv")
    root_selection = pd.read_csv(root / "selection_audit.csv")
    shard_blocks: list[pd.DataFrame] = []
    shard_selection: list[pd.DataFrame] = []
    stream = _stream_config(config)
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        seed_status = _strict_json(seed_root / "status.json")
        if seed_status != {"seed": seed, "status": "complete"}:
            raise RuntimeError(f"seed {seed} is not complete")
        metadata = _strict_json(seed_root / "metadata.json")
        if metadata.get("fit_test_tapes_independent") is not True or metadata.get(
            "all_methods_share_test_tape"
        ) is not True:
            raise RuntimeError(f"seed {seed} lacks paired independent tape evidence")
        fit_tape = generate_matched_uncertainty_tape(
            seed=seed, split="fit_matched_qr", config=stream
        )
        test_tape = generate_matched_uncertainty_tape(
            seed=seed, split="test_matched_qr", config=stream
        )
        if metadata.get("fit_tape_digest") != fit_tape.digest:
            raise RuntimeError(f"seed {seed} fit tape does not replay")
        if metadata.get("test_tape_digest") != test_tape.digest:
            raise RuntimeError(f"seed {seed} test tape does not replay")
        seed_blocks = pd.read_csv(seed_root / "block_metrics.csv")
        seed_selection = pd.read_csv(seed_root / "selection_audit.csv")
        if set(seed_blocks["test_tape_digest"]) != {test_tape.digest}:
            raise RuntimeError(f"seed {seed} methods do not share the replayed test tape")
        if set(seed_selection["fit_tape_digest"]) != {fit_tape.digest}:
            raise RuntimeError(f"seed {seed} selection did not use the replayed fit tape")
        shard_blocks.append(seed_blocks)
        shard_selection.append(seed_selection)

    _assert_frame(root_blocks, pd.concat(shard_blocks, ignore_index=True), name="block metrics")
    _assert_frame(
        root_selection,
        pd.concat(shard_selection, ignore_index=True),
        name="selection audit",
    )
    expected_blocks = len(seeds) * len(METHODS) * stream.n_sequences * stream.blocks_per_sequence
    if len(root_blocks) != expected_blocks:
        raise RuntimeError(f"expected {expected_blocks} block rows, found {len(root_blocks)}")
    if root_blocks.duplicated(["seed", "method", "block_id"]).any():
        raise RuntimeError("duplicate seed/method/block rows")
    if set(root_blocks["method"]) != set(METHODS) or set(root_blocks["seed"]) != set(seeds):
        raise RuntimeError("block method or seed coverage is incomplete")
    if not root_blocks.groupby("seed")["test_tape_digest"].nunique().eq(1).all():
        raise RuntimeError("methods were not paired on one test tape per seed")
    if int(root_blocks["invalid_rows"].sum()) != 0:
        raise RuntimeError("block metrics contain invalid rows")

    if len(root_selection) != len(seeds) * _candidate_count(config):
        raise RuntimeError("selection candidate coverage is incomplete")
    selected_mask = _boolean_column(root_selection["selected"], name="selected")
    privileged_mask = _boolean_column(
        root_selection["uses_true_parameters"], name="uses_true_parameters"
    )
    if set(root_selection["selection_family"]) != set(METHODS):
        raise RuntimeError("selection families are incomplete")
    if not root_selection.loc[selected_mask].groupby(
        ["seed", "selection_family"]
    ).size().eq(1).all():
        raise RuntimeError("each seed/family must select exactly one candidate")
    if set(root_selection.loc[privileged_mask, "selection_family"]) != PRIVILEGED:
        raise RuntimeError("truth-supported selection methods are mislabeled")
    if root_selection.loc[~privileged_mask, "selection_family"].isin(PRIVILEGED).any():
        raise RuntimeError("a privileged selection row was labeled deployable")
    for (_, _), frame in root_selection.groupby(["seed", "selection_family"], sort=False):
        expected = frame.sort_values(["selection_nll", "candidate"], kind="stable").iloc[0]
        observed = frame.loc[selected_mask.loc[frame.index]].iloc[0]
        if observed["candidate"] != expected["candidate"]:
            raise RuntimeError("selected hyperparameter is not the fit-tape argmin")

    replay_seed, replay_comparisons, replay_separation, replay_summary = summarize(
        root_blocks, config=config
    )
    _assert_frame(pd.read_csv(root / "seed_metrics.csv"), replay_seed, name="seed metrics")
    _assert_frame(
        pd.read_csv(root / "comparisons.csv"), replay_comparisons, name="comparisons"
    )
    _assert_frame(
        pd.read_csv(root / "matched_pair_separation.csv"),
        replay_separation,
        name="matched-pair separation",
    )
    _assert_nested_close(summary, replay_summary)

    return {
        "audit_status": "passed",
        "protocol_version": config["protocol_version"],
        "development_only": True,
        "claim_eligible": False,
        "verdict": "inconclusive",
        "budget_matched": False,
        "run_start_commit": commit,
        "run_start_clean": manifest["git"].get("dirty") is False,
        "n_seeds": len(seeds),
        "n_block_rows": len(root_blocks),
        "n_selection_rows": len(root_selection),
        "methods": list(METHODS),
        "statistics_unit": "seed",
        "artifact_hash_check": "passed",
        "source_commit_check": "passed",
        "tape_replay_check": "passed",
        "aggregate_shard_identity": "passed",
        "selection_argmin_replay": "passed",
        "summary_replay_check": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-dirty-run-start",
        action="store_true",
        help="Only for unit-test/smoke artifacts; never use for the registered run.",
    )
    args = parser.parse_args()
    result = validate_result(
        args.result_dir,
        require_clean_run_start=not args.allow_dirty_run_start,
    )
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(destination)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
