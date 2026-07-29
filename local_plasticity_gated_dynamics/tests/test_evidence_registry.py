from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_evidence_views import (
    CURRENT_DISPOSITIONS,
    build_views,
    load_branch_history,
    load_historical_git_objects,
    load_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_registry_is_complete_disjoint_and_evidence_bound() -> None:
    rows = load_registry(PROJECT_ROOT)
    assert [row["experiment_id"] for row in rows] == [
        f"exp{index:02d}" for index in range(45)
    ]
    current = {
        row["experiment_id"]
        for row in rows
        if row["disposition"] in CURRENT_DISPOSITIONS
    }
    history = {
        row["experiment_id"] for row in rows if row["disposition"] == "historical_only"
    }
    assert current.isdisjoint(history)
    assert current | history == {f"exp{index:02d}" for index in range(45)}


def test_rejected_abandoned_and_superseded_work_is_historical_only() -> None:
    registry = {row["experiment_id"]: row for row in load_registry(PROJECT_ROOT)}
    dispositions = {
        experiment_id: row["disposition"] for experiment_id, row in registry.items()
    }
    for experiment_id in (
        "exp00",
        "exp04",
        "exp13",
        "exp16",
        "exp18",
        "exp22",
        "exp23",
        "exp28",
        "exp30",
        "exp33",
        "exp34",
        "exp35",
        "exp36",
        "exp37",
        "exp38",
        "exp40",
        "exp41",
        "exp42",
        "exp43",
        "exp44",
    ):
        assert dispositions[experiment_id] == "historical_only"
    for experiment_id in (
        "exp08",
        "exp09",
        "exp21",
        "exp24",
        "exp25",
        "exp26",
        "exp29",
        "exp31",
        "exp32",
        "exp39",
    ):
        assert dispositions[experiment_id] in CURRENT_DISPOSITIONS
    assert registry["exp37"]["disposition"] == "historical_only"
    assert registry["exp37"]["conclusion"] == "oppose"
    assert registry["exp37"]["canonical_evidence"] == (
        "results/exp37_core50_change_aware_prefix_confirmation/report.md"
    )
    assert registry["exp41"]["conclusion"] == "mixed"
    assert registry["exp41"]["canonical_evidence"] == (
        "results/history/exp41_matched_qr_development_20260727.md"
    )
    assert registry["exp43"]["conclusion"] == "mixed"
    assert registry["exp43"]["canonical_evidence"] == (
        "results/history/exp43_fast_slow_causal_decomposition_development_20260728.md"
    )
    assert registry["exp44"]["disposition"] == "historical_only"
    assert registry["exp44"]["conclusion"] == "oppose"
    assert registry["exp44"]["canonical_evidence"] == (
        "results/exp44_piray_daw_qr_behavior_development_v1/report.md"
    )


def test_every_ancestor_branch_has_a_hash_bound_snapshot() -> None:
    rows = load_branch_history(PROJECT_ROOT)
    ancestors = [row for row in rows if row["relationship_to_current"] == "ancestor"]
    assert len(ancestors) == 9
    for row in ancestors:
        snapshot = PROJECT_ROOT / row["snapshot_path"]
        assert {"project_README.md", "report.md", "summary.csv"}.issubset(
            {path.name for path in snapshot.iterdir()}
        )
    objects = load_historical_git_objects(PROJECT_ROOT)
    assert [(row["path"], row["bytes"]) for row in objects] == [
        ("results/raw_metrics.csv", "29417930")
    ]
    assert objects[0]["archive_format"] == "tar.gz"
    assert (PROJECT_ROOT / objects[0]["materialized_path"]).is_file()


def test_generated_views_match_committed_indexes(tmp_path: Path) -> None:
    build_views(PROJECT_ROOT, tmp_path)
    for relative in (
        Path("current/README.md"),
        Path("current/claims.csv"),
        Path("current/experiments.csv"),
        Path("current/foundation_claims.csv"),
        Path("history/README.md"),
        Path("history/claims.csv"),
        Path("history/experiments.csv"),
        Path("history/git_objects.csv"),
        Path("history/branches.csv"),
        Path("history/snapshot_manifest.csv"),
    ):
        assert (tmp_path / relative).read_bytes() == (
            PROJECT_ROOT / "results" / relative
        ).read_bytes()

    with (tmp_path / "history" / "snapshot_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 30
    assert all(len(row["sha256"]) == 64 and int(row["bytes"]) > 0 for row in manifest)

    with (tmp_path / "current" / "claims.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        current_claims = list(csv.DictReader(handle))
    with (tmp_path / "history" / "claims.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        historical_claims = list(csv.DictReader(handle))
    assert current_claims
    assert historical_claims
    assert all(not row["experiment"].startswith("exp23_") for row in current_claims)
    historical_ids = {
        row["experiment_id"]
        for row in load_registry(PROJECT_ROOT)
        if row["disposition"] == "historical_only"
    }
    assert all(
        any(
            row["experiment"].startswith(f"{experiment_id}_")
            for experiment_id in historical_ids
        )
        for row in historical_claims
    )
    exp37_claims = [
        row
        for row in historical_claims
        if row["experiment"] == "exp37_core50_change_aware_prefix"
    ]
    assert {row["claim_id"] for row in exp37_claims} >= {
        "exp37_over_fixed_forgetting_switch",
        "exp37_over_sliding_window_switch",
        "exp37_joint_change_aware_prefix",
    }
    assert next(
        row
        for row in exp37_claims
        if row["claim_id"] == "exp37_joint_change_aware_prefix"
    )["conclusion"] == "oppose"
    exp44_claims = [
        row
        for row in historical_claims
        if row["experiment"] == "exp44_piray_daw_qr_behavior"
    ]
    assert {row["claim_id"] for row in exp44_claims} >= {
        "exp44_nll_vs_fixed",
        "exp44_nll_vs_total_uncertainty",
        "exp44_joint_development_gate",
        "exp44_experiment2_confirmation",
        "exp44_conditional_popgym_scale",
    }
    assert next(
        row
        for row in exp44_claims
        if row["claim_id"] == "exp44_joint_development_gate"
    )["conclusion"] == "oppose"

    with (tmp_path / "current" / "foundation_claims.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        foundation_claims = list(csv.DictReader(handle))
    assert {row["experiment"] for row in foundation_claims} == {"exp08", "exp09"}


def test_deleted_branch_tips_are_fully_reachable_and_materialized() -> None:
    with (PROJECT_ROOT / "results" / "history" / "branch_reachability.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert all(row["reachable"] == "true" for row in rows)
    assert all(row["unique_commits"] == "0" for row in rows)
    deleted = [row for row in rows if row["deleted_path_count"] != "0"]
    assert len(deleted) == 1
    assert deleted[0]["branch"] == "agent/real-lowdim-validation"
    assert deleted[0]["materialization_status"] == "materialized_archive"
