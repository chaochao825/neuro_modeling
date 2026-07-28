"""Replay and validate an Exp43 development-only result directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from experiments.exp43_fast_slow_causal_decomposition import (
    EXPERIMENT,
    METHODS,
    ORACLE_METHODS,
    SOURCE_FILES,
    summarize,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _assert_nested_close(actual: Any, expected: Any, *, path: str = "summary") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{path} mapping keys differ during replay")
        for key in expected:
            _assert_nested_close(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} list shape differs during replay")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_nested_close(
                actual_item, expected_item, path=f"{path}[{index}]"
            )
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not np.isclose(
            float(actual), float(expected), rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"{path} differs during numerical replay")
        return
    if actual != expected:
        raise ValueError(f"{path} differs during replay: {actual!r} != {expected!r}")


def _sorted(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.sort_values(list(columns)).reset_index(drop=True)


def validate_result(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    required = (
        "config.json",
        "environment.json",
        "planned_conditions.json",
        "method_budget.csv",
        "seed_metrics.csv",
        "block_metrics.csv",
        "event_window_metrics.csv",
        "regime_window_metrics.csv",
        "selection_audit.csv",
        "comparisons.csv",
        "failures.json",
        "failed_seeds.csv",
        "summary.json",
        "report.md",
        "run.log",
        "status.json",
        "manifest.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise ValueError(f"missing required Exp43 artifacts: {missing}")

    config = _read_json(output / "config.json")
    validate_config(config)
    status = _read_json(output / "status.json")
    summary = _read_json(output / "summary.json")
    environment = _read_json(output / "environment.json")
    planned = _read_json(output / "planned_conditions.json")
    manifest = _read_json(output / "manifest.json")
    failures = json.loads((output / "failures.json").read_text(encoding="utf-8"))

    if status.get("status") != "complete" or status.get("n_failed_seeds") != 0:
        raise ValueError("Exp43 status is not complete with zero failed seeds")
    if status.get("claim_upgrade_allowed") is not False:
        raise ValueError("Exp43 status illegally permits a claim upgrade")
    if summary.get("verdict") != "inconclusive_development_only":
        raise ValueError("Exp43 summary has an invalid development verdict")
    if summary.get("claim_upgrade_allowed") is not False:
        raise ValueError("Exp43 summary illegally permits a claim upgrade")
    if planned.get("methods") != list(METHODS):
        raise ValueError("planned method panel differs from the registered panel")
    if planned.get("reserved_formal_seeds") != list(range(43100, 43130)):
        raise ValueError("reserved formal seeds are incomplete")
    if failures != [] or not pd.read_csv(output / "failed_seeds.csv").empty:
        raise ValueError("failure artifacts are inconsistent with complete status")
    if environment.get("git", {}).get("dirty") is not False:
        raise ValueError("Exp43 run did not start from a clean git worktree")

    artifact_hashes = manifest.get("artifacts")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("manifest artifact hashes are missing")
    for relative, expected_hash in artifact_hashes.items():
        path = output / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"artifact hash mismatch: {relative}")
    source_hashes = manifest.get("source_sha256")
    expected_source_hashes = {
        relative: _sha256(PROJECT_ROOT / relative) for relative in SOURCE_FILES
    }
    if source_hashes != expected_source_hashes:
        raise ValueError("source hashes differ from the executed implementation")

    seed_metrics = pd.read_csv(output / "seed_metrics.csv")
    block_metrics = pd.read_csv(output / "block_metrics.csv")
    event_metrics = pd.read_csv(output / "event_window_metrics.csv")
    regime_metrics = pd.read_csv(output / "regime_window_metrics.csv")
    selection_audit = pd.read_csv(output / "selection_audit.csv")
    comparisons = pd.read_csv(output / "comparisons.csv")
    expected_panel = {
        (int(seed), method) for seed in config["seeds"] for method in METHODS
    }
    observed_panel = set(
        zip(seed_metrics["seed"].astype(int), seed_metrics["method"], strict=True)
    )
    if observed_panel != expected_panel or seed_metrics.duplicated(
        ["seed", "method"]
    ).any():
        raise ValueError("aggregate seed panel is incomplete or duplicated")
    privilege = seed_metrics.groupby("method")["privileged"].unique().to_dict()
    for method in METHODS:
        expected_privilege = method in ORACLE_METHODS
        observed_privilege = [
            bool(value) for value in privilege.get(method, np.asarray([])).tolist()
        ]
        if observed_privilege != [expected_privilege]:
            raise ValueError(f"privilege annotation is wrong for {method}")
    if any("test" in column.lower() for column in selection_audit.columns):
        raise ValueError("selection audit contains a test-derived column")
    selected_counts = selection_audit.groupby("seed")["selected"].sum()
    if not np.all(selected_counts == 4):
        raise ValueError("each seed must select exactly four fit-only configurations")

    per_seed_frames: dict[str, list[pd.DataFrame]] = {
        "seed_metrics.csv": [],
        "block_metrics.csv": [],
        "event_window_metrics.csv": [],
        "regime_window_metrics.csv": [],
        "selection_audit.csv": [],
    }
    digests: set[str] = set()
    for seed in map(int, config["seeds"]):
        seed_dir = output / f"seed_{seed}"
        seed_status = _read_json(seed_dir / "status.json")
        metadata = _read_json(seed_dir / "metadata.json")
        if seed_status != {"seed": seed, "status": "complete"}:
            raise ValueError(f"seed {seed} status is invalid")
        fit_digest = str(metadata.get("fit_tape_digest", ""))
        test_digest = str(metadata.get("test_tape_digest", ""))
        if len(fit_digest) != 64 or len(test_digest) != 64 or fit_digest == test_digest:
            raise ValueError(f"seed {seed} has invalid fit/test tape digests")
        if fit_digest in digests or test_digest in digests:
            raise ValueError("tape digest was reused across a fit/test seed")
        digests.update((fit_digest, test_digest))
        if metadata.get("claim_upgrade_allowed") is not False:
            raise ValueError(f"seed {seed} metadata permits a claim upgrade")
        for name in per_seed_frames:
            per_seed_frames[name].append(pd.read_csv(seed_dir / name))

    aggregate_frames = {
        "seed_metrics.csv": (seed_metrics, ("seed", "method")),
        "block_metrics.csv": (block_metrics, ("seed", "method", "block_id")),
        "event_window_metrics.csv": (
            event_metrics,
            ("seed", "method", "event_index", "window"),
        ),
        "regime_window_metrics.csv": (
            regime_metrics,
            ("seed", "method", "block_id", "window"),
        ),
        "selection_audit.csv": (
            selection_audit,
            ("seed", "selection_family", "fit_nll"),
        ),
    }
    for name, (aggregate, sort_columns) in aggregate_frames.items():
        reconstructed = pd.concat(per_seed_frames[name], ignore_index=True)
        assert_frame_equal(
            _sorted(aggregate, sort_columns),
            _sorted(reconstructed, sort_columns),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    replay_comparisons, replay_summary = summarize(
        seed_metrics,
        block_metrics,
        event_metrics,
        regime_metrics,
        config=config,
    )
    assert_frame_equal(
        _sorted(comparisons, ("contrast", "endpoint")),
        _sorted(replay_comparisons, ("contrast", "endpoint")),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    _assert_nested_close(summary, replay_summary)

    formal_dirs = [
        path.name
        for seed in range(43100, 43130)
        if (path := output / f"seed_{seed}").exists()
    ]
    if formal_dirs:
        raise ValueError(f"reserved formal seed artifacts exist: {formal_dirs}")
    return {
        "validator": "validate_exp43_development_result_v1",
        "status": "pass",
        "experiment": EXPERIMENT,
        "result_dir": str(output),
        "n_seeds": len(config["seeds"]),
        "n_methods": len(METHODS),
        "summary_replay": "pass",
        "aggregate_replay": "pass",
        "artifact_hashes": "pass",
        "source_hashes": "pass",
        "formal_seeds_accessed": False,
        "claim_upgrade_allowed": False,
        "implementation_sha256": manifest["implementation_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()
    receipt = validate_result(args.result_dir)
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        if args.receipt.exists():
            raise FileExistsError(args.receipt)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
