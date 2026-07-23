"""Recheck deleted branch tips against the recorded consolidation commit."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_evidence_views import load_branch_history, load_historical_git_objects


FIELDS = (
    "branch",
    "tip_sha",
    "target_sha",
    "reachable",
    "unique_commits",
    "deleted_path_count",
    "deleted_paths",
    "materialization_status",
)


def _git(repo_root: Path, args: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def audit(project_root: Path) -> list[dict[str, str]]:
    project_root = project_root.resolve()
    repo_root = project_root.parent
    receipt = json.loads(
        (project_root / "provenance" / "consolidation_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    target = receipt["consolidation_commit"]
    _git(repo_root, ["cat-file", "-e", f"{target}^{{commit}}"])
    objects = load_historical_git_objects(project_root)
    materialized = {
        f"{project_root.name}/{row['path']}": row["materialized_path"]
        for row in objects
    }

    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for branch in load_branch_history(project_root):
        tip = branch["tip_sha"]
        reachable = (
            subprocess.run(
                ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", tip, target],
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
        unique = int(_git(repo_root, ["rev-list", "--count", tip, "--not", target]))
        deleted = _git(
            repo_root,
            ["diff", "--diff-filter=D", "--name-only", f"{tip}..{target}"],
        ).splitlines()
        missing = [path for path in deleted if path not in materialized]
        if missing:
            status = "missing"
        elif deleted:
            status = "materialized_archive"
        else:
            status = "not_needed"
        if not reachable or unique != 0 or missing:
            failures.append(branch["branch"])
        rows.append(
            {
                "branch": branch["branch"],
                "tip_sha": tip,
                "target_sha": target,
                "reachable": str(reachable).lower(),
                "unique_commits": str(unique),
                "deleted_path_count": str(len(deleted)),
                "deleted_paths": ";".join(deleted),
                "materialization_status": status,
            }
        )
    if failures:
        raise RuntimeError(f"branch consolidation audit failed: {failures}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = audit(args.project_root)
    output = args.output or args.project_root / "results" / "history" / "branch_reachability.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
