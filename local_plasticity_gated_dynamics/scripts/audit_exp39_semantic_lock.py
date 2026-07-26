#!/usr/bin/env python3
"""Replay Exp39 and verify the additive byte/numeric semantic lock."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_exp39_result import validate_result
from src.analysis.exp39_semantic_lock import validate_exp39_semantic_lock


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _replay_environment() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for distribution in ("numpy", "scipy", "pandas"):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None
    return {
        "scope": "current_replay_only_not_original_formal_execution",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
    }


def audit(
    result_dir: Path,
    lock_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    replay = validate_result(result_dir.resolve())
    semantic = validate_exp39_semantic_lock(
        project_root.resolve(), result_dir.resolve(), lock_path.resolve()
    )
    return {
        **semantic,
        "replay_environment": _replay_environment(),
        "replay_environment_is_original_formal_environment": False,
        "frozen_replay": {
            "audit_status": replay["audit_status"],
            "paired_tape_check": replay["paired_tape_check"],
            "selection_replay_check": replay["selection_replay_check"],
            "summary_replay_check": replay["summary_replay_check"],
            "registered_verdict": replay["verdict"],
            "registered_joint_gate_passed": replay["joint_gate_passed"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        default=PROJECT_ROOT / "provenance/exp39_semantic_lock_20260727.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.result_dir, args.lock)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
