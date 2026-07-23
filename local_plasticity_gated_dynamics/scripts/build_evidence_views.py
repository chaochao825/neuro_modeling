"""Validate the evidence registry and build disjoint current/history views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import tarfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


EXPERIMENT_FIELDS = (
    "experiment_id",
    "title",
    "track",
    "disposition",
    "conclusion",
    "current_successor",
    "canonical_evidence",
    "reason",
)
BRANCH_FIELDS = (
    "branch",
    "tip_sha",
    "tip_date",
    "relationship_to_current",
    "commits_added_after_tip",
    "disposition",
    "scope",
    "snapshot_path",
)
GIT_OBJECT_FIELDS = (
    "branch",
    "tip_sha",
    "path",
    "blob_sha",
    "bytes",
    "materialized_path",
    "archive_member",
    "archive_format",
    "retention",
)
CURRENT_DISPOSITIONS = {"current_core", "current_foundation", "current_open"}
ALLOWED_DISPOSITIONS = CURRENT_DISPOSITIONS | {"historical_only"}
ALLOWED_CONCLUSIONS = {"support", "oppose", "inconclusive", "mixed"}
EXPECTED_BRANCHES = {
    "main",
    "agent/real-lowdim-validation",
    "agent/effective-control-p0",
    "agent/hidden-context-p2",
    "agent/ei-realdata-system",
    "agent/integrated-tiny-hrm",
    "agent/exp16-learning-retry",
    "agent/exp18-arc-recursive",
    "agent/belief-ei-block-switch",
    "agent/exp26-actuator-matching",
}


def _read_csv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            raise ValueError(
                f"{path} has fields {reader.fieldnames}; expected {list(expected_fields)}"
            )
        return [{key: value.strip() for key, value in row.items()} for row in reader]


def load_registry(project_root: Path) -> list[dict[str, str]]:
    """Load and validate the experiment registry."""

    project_root = project_root.resolve()
    rows = _read_csv(
        project_root / "provenance" / "experiment_registry.csv", EXPERIMENT_FIELDS
    )
    ids = [row["experiment_id"] for row in rows]
    expected_ids = [f"exp{index:02d}" for index in range(34)]
    if ids != expected_ids:
        raise ValueError(f"experiment IDs must be exactly {expected_ids}; got {ids}")
    if len(ids) != len(set(ids)):
        raise ValueError("experiment registry contains duplicate IDs")

    script_ids = sorted(
        {
            match.group(1)
            for path in (project_root / "experiments").glob("exp[0-9][0-9]_*.py")
            if (match := re.match(r"(exp\d{2})_", path.name))
        }
    )
    if script_ids != expected_ids:
        raise ValueError(
            f"experiment entry points do not cover exp00-exp33: {script_ids}"
        )

    for row in rows:
        experiment_id = row["experiment_id"]
        if row["disposition"] not in ALLOWED_DISPOSITIONS:
            raise ValueError(
                f"invalid disposition for {experiment_id}: {row['disposition']}"
            )
        if row["conclusion"] not in ALLOWED_CONCLUSIONS:
            raise ValueError(
                f"invalid conclusion for {experiment_id}: {row['conclusion']}"
            )
        if row["disposition"] == "historical_only" and not row["reason"]:
            raise ValueError(f"historical experiment {experiment_id} requires a reason")
        successor = row["current_successor"]
        if successor not in {"", "none"} and successor not in expected_ids:
            raise ValueError(f"unknown successor for {experiment_id}: {successor}")
        evidence = (project_root / row["canonical_evidence"]).resolve()
        if project_root not in evidence.parents or not evidence.is_file():
            raise ValueError(
                f"missing or out-of-tree evidence for {experiment_id}: {evidence}"
            )
    return rows


def load_branch_history(project_root: Path) -> list[dict[str, str]]:
    """Load and validate the audited remote-branch inventory."""

    project_root = project_root.resolve()
    rows = _read_csv(project_root / "provenance" / "branch_history.csv", BRANCH_FIELDS)
    names = {row["branch"] for row in rows}
    if names != EXPECTED_BRANCHES:
        raise ValueError(f"branch inventory mismatch: {sorted(names)}")
    for row in rows:
        if not re.fullmatch(r"[0-9a-f]{40}", row["tip_sha"]):
            raise ValueError(f"invalid tip SHA for {row['branch']}")
        relation = row["relationship_to_current"]
        if relation not in {"current_base", "ancestor"}:
            raise ValueError(
                f"invalid branch relationship for {row['branch']}: {relation}"
            )
        if relation == "ancestor" and row["disposition"] != "historical_snapshot":
            raise ValueError(
                f"ancestor branch lacks historical disposition: {row['branch']}"
            )
        snapshot = row["snapshot_path"]
        if relation == "ancestor":
            snapshot_path = (project_root / snapshot).resolve()
            if project_root not in snapshot_path.parents or not snapshot_path.is_dir():
                raise ValueError(
                    f"missing branch snapshot for {row['branch']}: {snapshot_path}"
                )
            expected = {"project_README.md", "report.md", "summary.csv"}
            if not expected.issubset({path.name for path in snapshot_path.iterdir()}):
                raise ValueError(f"incomplete branch snapshot for {row['branch']}")
    return rows


def load_historical_git_objects(project_root: Path) -> list[dict[str, str]]:
    """Validate historical-only Git objects and their materialized archives."""

    rows = _read_csv(
        project_root / "provenance" / "historical_git_objects.csv",
        GIT_OBJECT_FIELDS,
    )
    branch_names = {row["branch"] for row in load_branch_history(project_root)}
    for row in rows:
        if row["branch"] not in branch_names:
            raise ValueError(f"historical object uses unknown branch: {row['branch']}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["tip_sha"]):
            raise ValueError(f"invalid object tip SHA: {row['tip_sha']}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["blob_sha"]):
            raise ValueError(f"invalid blob SHA: {row['blob_sha']}")
        if int(row["bytes"]) <= 0:
            raise ValueError(f"invalid object byte count: {row['bytes']}")
        archive_path = (project_root / row["materialized_path"]).resolve()
        if project_root not in archive_path.parents or not archive_path.is_file():
            raise ValueError(f"missing materialized historical object: {archive_path}")
        if row["archive_format"] != "tar.gz":
            raise ValueError(
                f"unsupported historical archive format: {row['archive_format']}"
            )
        with tarfile.open(archive_path, mode="r:gz") as archive:
            member = archive.getmember(row["archive_member"])
            if not member.isfile() or member.size != int(row["bytes"]):
                raise ValueError(
                    f"historical archive member mismatch: {row['archive_member']}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(
                    f"cannot read historical archive member: {row['archive_member']}"
                )
            digest = hashlib.sha1(f"blob {member.size}\0".encode("ascii"))
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != row["blob_sha"]:
            raise ValueError(
                f"materialized archive does not match Git blob: {archive_path}"
            )
    return rows


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, str]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _experiment_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    lines = [
        "| Experiment | Track | Disposition | Conclusion | Successor | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        evidence = row["canonical_evidence"]
        successor = row["current_successor"] or "none"
        evidence_link = (
            f"../{evidence.removeprefix('results/')}"
            if evidence.startswith("results/")
            else f"../../{evidence}"
        )
        lines.append(
            f"| {row['experiment_id']} {row['title']} | {row['track']} | "
            f"`{row['disposition']}` | **{row['conclusion']}** | {successor} | "
            f"[{Path(evidence).name}]({evidence_link}) |"
        )
    return lines


def _snapshot_manifest(project_root: Path) -> list[dict[str, str]]:
    snapshot_root = project_root / "results" / "history" / "branch_snapshots"
    manifest: list[dict[str, str]] = []
    preserved = [
        candidate for candidate in snapshot_root.rglob("*") if candidate.is_file()
    ]
    preserved.extend(
        project_root / "results" / "history" / name
        for name in (
            "actuator_matching_critical_audit_20260718.md",
            "project_README_pre_consolidation.md",
        )
    )
    for path in sorted(preserved):
        if not path.is_file():
            raise ValueError(f"missing preserved historical artifact: {path}")
        data = path.read_bytes()
        manifest.append(
            {
                "snapshot_path": path.relative_to(project_root).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": str(len(data)),
            }
        )
    return manifest


def _split_aggregate_claims(
    project_root: Path,
    current_ids: set[str],
    historical_ids: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]], tuple[str, ...]]:
    """Split the legacy mixed aggregate without changing any claim row."""

    aggregate = project_root / "results" / "summary.csv"
    with aggregate.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not fields or "experiment" not in fields:
        raise ValueError(f"legacy aggregate lacks an experiment field: {aggregate}")

    current: list[dict[str, str]] = []
    history: list[dict[str, str]] = []
    for row in rows:
        match = re.match(r"(exp\d{2})", row["experiment"])
        if not match:
            raise ValueError(f"cannot classify aggregate row: {row['experiment']}")
        experiment_id = match.group(1)
        if experiment_id in current_ids:
            current.append(row)
        elif experiment_id in historical_ids:
            history.append(row)
        else:
            raise ValueError(
                f"aggregate row references an unregistered experiment: {experiment_id}"
            )
    return current, history, fields


def _filter_claim_file(
    path: Path, experiment_ids: set[str]
) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not fields or "experiment" not in fields:
        raise ValueError(f"claim file lacks an experiment field: {path}")
    selected = []
    for row in rows:
        match = re.match(r"(exp\d{2})", row["experiment"])
        if match and match.group(1) in experiment_ids:
            selected.append(row)
    if not selected:
        raise ValueError(f"claim file has no requested experiment rows: {path}")
    return selected, fields


def build_views(project_root: Path, output_root: Path | None = None) -> None:
    """Validate provenance and write deterministic current/history indexes."""

    project_root = project_root.resolve()
    output_root = (output_root or project_root / "results").resolve()
    registry = load_registry(project_root)
    branches = load_branch_history(project_root)
    historical_objects = load_historical_git_objects(project_root)
    current = [row for row in registry if row["disposition"] in CURRENT_DISPOSITIONS]
    history = [row for row in registry if row["disposition"] == "historical_only"]
    current_ids = {row["experiment_id"] for row in current}
    historical_ids = {row["experiment_id"] for row in history}
    if current_ids & historical_ids:
        raise ValueError("current and historical experiment views overlap")

    current_root = output_root / "current"
    history_root = output_root / "history"
    _write_csv(current_root / "experiments.csv", current, EXPERIMENT_FIELDS)
    _write_csv(history_root / "experiments.csv", history, EXPERIMENT_FIELDS)
    _write_csv(history_root / "branches.csv", branches, BRANCH_FIELDS)
    _write_csv(
        history_root / "git_objects.csv",
        historical_objects,
        GIT_OBJECT_FIELDS,
    )
    current_claims, historical_claims, claim_fields = _split_aggregate_claims(
        project_root, current_ids, historical_ids
    )
    _write_csv(current_root / "claims.csv", current_claims, claim_fields)
    _write_csv(history_root / "claims.csv", historical_claims, claim_fields)
    foundation_claims, foundation_fields = _filter_claim_file(
        project_root
        / "results"
        / "history"
        / "branch_snapshots"
        / "hidden-context-p2"
        / "summary.csv",
        {"exp08", "exp09"},
    )
    _write_csv(
        current_root / "foundation_claims.csv",
        foundation_claims,
        foundation_fields,
    )
    _write_csv(
        history_root / "snapshot_manifest.csv",
        _snapshot_manifest(project_root),
        ("snapshot_path", "sha256", "bytes"),
    )

    current_lines = [
        "# Current evidence view",
        "",
        "This view contains only evidence that remains part of the active theory:",
        "high-rank physical E/I substrates, low-dimensional credit/effective control,",
        "hidden belief inference, and task-matched actuator selection. Superseded or",
        "rejected methods are excluded and live only in the historical view.",
        "",
        "A `support` label is scoped to the registered experiment; it is not support",
        "for the complete theory. The active real-neural endpoint (Exp25) remains",
        "fail-closed and inconclusive.",
        "",
        "`claims.csv` is the lossless current-only split of the legacy mixed",
        "`results/summary.csv`; no historical claim row is present in it.",
        "`foundation_claims.csv` extracts the still-active Exp08/09 rows from",
        "their hash-bound ancestor snapshot without importing superseded claims.",
        "",
        *_experiment_table(current),
        "",
        "Generated by `scripts/build_evidence_views.py` from the provenance registry.",
    ]
    (current_root / "README.md").write_text(
        "\n".join(current_lines) + "\n", encoding="utf-8", newline="\n"
    )

    history_lines = [
        "# Historical evidence view",
        "",
        "This directory is the only presentation surface for superseded, rejected,",
        "abandoned, or exploratory proposals. Conclusions are preserved exactly at the",
        "experiment level: a once-positive result can remain `support` while its",
        "disposition is `historical_only`. It must not be promoted into the current",
        "method claim.",
        "",
        "The code entry points remain in `experiments/` for reproducibility. Branch",
        "snapshots preserve each ancestor branch's README, report, and summary without",
        "rewriting their original claims. `snapshot_manifest.csv` binds those files by",
        "SHA-256. No failed or negative result was deleted.",
        "The only ancestor-tip result absent from the current tree is indexed",
        "in `git_objects.csv` and materialized as a compressed historical archive",
        "whose decompressed bytes are checked against the original Git blob SHA.",
        "`branch_reachability.csv` is the executable audit receipt showing that",
        "every deleted branch tip contributes zero commits outside consolidated main.",
        "",
        "`claims.csv` retains every historical row found in the legacy mixed",
        "aggregate, including all failed Exp23 controller rows.",
        "The pre-consolidation critical audit is preserved as",
        "`actuator_matching_critical_audit_20260718.md`; the current copy omits",
        "the rejected Exp23 mechanism and abandoned Exp32-v1 configuration.",
        "The complete pre-consolidation project narrative and reproduction",
        "commands are preserved as `project_README_pre_consolidation.md`.",
        "",
        *_experiment_table(history),
        "",
        "Generated by `scripts/build_evidence_views.py` from the provenance registry.",
    ]
    (history_root / "README.md").write_text(
        "\n".join(history_lines) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    build_views(args.project_root, args.output_root)


if __name__ == "__main__":
    main()
