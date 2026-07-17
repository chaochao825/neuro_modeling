"""Contracts for the standalone Exp26 phase-diagram summarizer."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import exp26_actuator_phase_diagram as exp26
from scripts.summarize_exp26 import (
    EXPERIMENT,
    collect_metrics,
    write_summary_artifacts,
)


MODES = ("frozen", "routing", "gain", "low_rank", "rgl")
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = {
    profile: ROOT / "configs" / profile / "exp26_actuator_phase_diagram.json"
    for profile in ("smoke", "formal")
}


def _registered_config(profile: str) -> dict[str, object]:
    path = CONFIG_PATHS[profile].resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config["config_path"] = str(path)
    return config


def _budget_receipt(
    config: dict[str, object],
    evidence: dict[str, object],
) -> dict[str, object]:
    git = evidence["git"]
    return {
        "required": True,
        "receipt_schema": "exp26_budget_preflight_v2_observed_bound",
        "receipt_sha256": "a" * 64,
        "preflight_passed": True,
        "registered_config_sha256": exp26.canonical_config_sha256(config),
        "manifest_sha256": config["manifest"]["expected_hash"],
        "observed_required_scale_max": config["budget_preflight"][
            "required_scale_max"
        ],
        "policy_required_scale_max": config["budget_preflight"][
            "required_scale_max"
        ],
        "derived_ceiling": config["actuator"]["max_scale"],
        "provenance_clean": True,
        "provenance_stable_during_run": True,
        "git_commit": git["commit"],
        "git_tree": git["tree"],
    }


def _budget_row_fields(evidence: dict[str, object]) -> dict[str, object]:
    receipt = evidence["budget_preflight"]
    if receipt is None:
        return {
            "preflight_required": False,
            "preflight_passed": None,
            "preflight_receipt_sha256": None,
            "preflight_git_commit": None,
            "preflight_git_tree": None,
        }
    return {
        "preflight_required": receipt["required"],
        "preflight_passed": receipt["preflight_passed"],
        "preflight_receipt_sha256": receipt["receipt_sha256"],
        "preflight_git_commit": receipt["git_commit"],
        "preflight_git_tree": receipt["git_tree"],
    }


def _rows(
    seed: int,
    *,
    config: dict[str, object],
    evidence: dict[str, object],
    run_id: str,
    run_label: str,
    retain_rgl_failure: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cells = exp26._manifest(config)
    chi_by_generator: dict[str, float] = {}
    for split_index, split in enumerate(("discovery", "heldout")):
        split_cells = [cell for cell in cells if cell.generator_split == split]
        chi_grid = np.linspace(0.05, 0.95, len(split_cells))
        permutation = np.random.default_rng(2601 + split_index).permutation(
            len(split_cells)
        )
        for index, cell in enumerate(split_cells):
            chi_by_generator[cell.generator_id] = float(chi_grid[permutation[index]])

    first_heldout = next(
        cell.generator_id
        for cell in cells
        if cell.generator_split == "heldout"
    )
    receipt_fields = exp26.evidence_row_fields(evidence, run_label=run_label)
    budget_fields = _budget_row_fields(evidence)
    for plan in exp26._planned_conditions(config):
        mode = str(plan["actuator_mode"])
        chi = chi_by_generator[str(plan["generator_id"])]
        values = {
            "frozen": 0.50,
            "routing": 0.90 - 0.35 * chi,
            "gain": 0.88 - 0.35 * chi,
            "low_rank": 0.55 + 0.35 * chi,
            "rgl": 0.92,
        }
        failed = bool(
            retain_rgl_failure
            and plan["generator_id"] == first_heldout
            and mode == "rgl"
        )
        rows.append(
            {
                "run_id": run_id,
                "experiment": EXPERIMENT,
                "seed": seed,
                **plan,
                "chi": chi,
                "delay_steps": int(plan["delay"]),
                "validation_balanced_accuracy": float(values[mode]),
                "test_balanced_accuracy": float(values[mode]),
                "energy_proxy": 1.0 + chi,
                "status": "failed" if failed else "complete",
                "functional_budget_valid": not failed,
                **receipt_fields,
                **budget_fields,
                **(
                    {"failure_reason": "retained RGL ceiling failure"}
                    if failed
                    else {}
                ),
            }
        )
    return rows


def _write_attempt(
    root: Path,
    *,
    seed: int,
    label: str,
    profile: str = "smoke",
    retain_rgl_failure: bool = False,
) -> Path:
    attempt = (
        root
        / "results"
        / "runs"
        / EXPERIMENT
        / f"seed_{seed:04d}"
        / f"20260717T000000.000000Z_{label}"
    )
    attempt.mkdir(parents=True)
    config = _registered_config(profile)
    receipt = exp26.manifest_hash(exp26._manifest(config))
    evidence = exp26.build_evidence_provenance(config, manifest_sha256=receipt)
    if profile == "formal":
        # Represents a completed clean run while the test worktree is dirty.
        evidence = copy.deepcopy(evidence)
        evidence["git"]["dirty"] = False
        evidence["budget_preflight"] = _budget_receipt(config, evidence)
    else:
        evidence["budget_preflight"] = None
    run_id = f"run-{profile}-{seed}-{label}"
    rows = _rows(
        seed,
        config=config,
        evidence=evidence,
        run_id=run_id,
        run_label=label,
        retain_rgl_failure=retain_rgl_failure,
    )
    artifact_config = {
        "experiment": EXPERIMENT,
        "seed": seed,
        "run_label": label,
        **config,
        "evidence_provenance": evidence,
    }
    failures = sum(row["status"] == "failed" for row in rows)
    status = "complete_with_failures" if failures else "complete"
    (attempt / "config.json").write_text(
        json.dumps(artifact_config), encoding="utf-8"
    )
    (attempt / "status.json").write_text(
        json.dumps(
            {
                "status": status,
                "seed": seed,
                "run_label": label,
                "condition_failures": failures,
                "condition_invalid": 0,
            }
        ),
        encoding="utf-8",
    )
    (attempt / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "experiment": EXPERIMENT,
                "seed": seed,
                "run_label": label,
                "profile": profile,
                "status": status,
                "evidence_provenance": evidence,
            }
        ),
        encoding="utf-8",
    )
    versions = evidence["runtime_versions"]
    git = evidence["git"]
    (attempt / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "packages": {
                    "numpy": versions["numpy"],
                    "scipy": versions["scipy"],
                    "pandas": versions["pandas"],
                    "scikit-learn": versions["scikit_learn"],
                    "statsmodels": versions["statsmodels"],
                },
                "git": {
                    "commit": git["commit"],
                    "tree": git["tree"],
                    "dirty": git["dirty"],
                },
            }
        ),
        encoding="utf-8",
    )
    (attempt / "planned_conditions.json").write_text(
        json.dumps(
            [
                {"condition_index": index, **row}
                for index, row in enumerate(exp26._planned_conditions(config))
            ]
        ),
        encoding="utf-8",
    )
    (attempt / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return attempt


def _update_budget_receipt(
    attempt: Path,
    *,
    receipt_updates: dict[str, object],
    row_updates: dict[str, object] | None = None,
) -> None:
    for name in ("config.json", "manifest.json"):
        path = attempt / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["evidence_provenance"]["budget_preflight"].update(
            receipt_updates
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
    if row_updates:
        metrics_path = attempt / "metrics.jsonl"
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        for row in rows:
            row.update(row_updates)
        metrics_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


def test_exp26_summarizer_help_works_outside_repository(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_exp26.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--results-root" in completed.stdout
    assert "--run-label" in completed.stdout
    assert "--skip-plots" in completed.stdout


def test_smoke_snapshot_is_seed_level_dev_only_and_retains_failures(
    tmp_path: Path,
) -> None:
    for seed in (9000, 9001):
        _write_attempt(
            tmp_path,
            seed=seed,
            label="registered",
            retain_rgl_failure=seed == 9001,
        )
    paths = write_summary_artifacts(
        tmp_path / "results",
        output_dir=tmp_path / "summary",
        profile="smoke",
        run_label="registered",
        skip_plots=True,
    )
    assert set(paths) == {
        "raw",
        "summary",
        "seed_endpoints",
        "conclusion",
        "report",
    }
    assert all(path.is_file() for path in paths.values())

    raw = pd.read_csv(paths["raw"])
    summary = pd.read_csv(paths["summary"])
    endpoints = pd.read_csv(paths["seed_endpoints"])
    conclusion = json.loads(paths["conclusion"].read_text(encoding="utf-8"))
    report = paths["report"].read_text(encoding="utf-8")

    assert len(raw) == 240
    assert raw["statistics_unit"].isna().all() if "statistics_unit" in raw else True
    assert len(raw.loc[raw["status"].eq("failed")]) == 1
    assert raw.loc[raw["status"].eq("failed"), "failure_reason"].iloc[0] == (
        "retained RGL ceiling failure"
    )
    assert set(summary["statistics_unit"]) == {"seed"}
    assert summary["n_seed"].max() == 2
    assert summary["n_seed"].max() < raw["generator_id"].nunique()
    assert set(endpoints["seed"]) == {9000, 9001}

    assert conclusion["profile"] == "smoke"
    assert conclusion["evidence_scope"] == "development_only"
    assert conclusion["dev_only"] is True
    assert conclusion["confirmatory_eligible"] is False
    assert conclusion["conclusion"] == "inconclusive"
    assert conclusion["statistics_unit"] == "seed"
    assert conclusion["coverage"]["failed_row_count"] == 1
    assert conclusion["coverage"]["missing_seed_count"] == 0
    assert "DEVELOPMENT ONLY" in report
    assert "three co-primary" in report
    assert "Holm" in report
    assert "χ versus raw α" in report
    assert "RGL is a descriptive composite ceiling" in report
    assert "Failed rows retained: 1" in report
    assert conclusion["registered_config_sha256"] == exp26.canonical_config_sha256(
        _registered_config("smoke")
    )
    analysis = _registered_config("smoke")["analysis"]
    assert conclusion["registered_analysis"] == {
        name: analysis[name]
        for name in (
            "tie_margin",
            "bootstrap_samples",
            "permutation_samples",
            "statistics_seed",
        )
    }
    assert conclusion["run_provenance"]["run_label"] == "registered"
    assert conclusion["run_provenance"]["budget_preflight"] is None
    assert conclusion["run_provenance"]["runtime_versions"] == (
        exp26.scientific_runtime_versions()
    )


def test_multiple_runs_require_explicit_run_label(tmp_path: Path) -> None:
    for label in ("first", "second"):
        _write_attempt(
            tmp_path,
            seed=9000,
            label=label,
        )
    with pytest.raises(ValueError, match="multiple Exp26 attempts"):
        collect_metrics(tmp_path / "results", profile="smoke")

    selected = collect_metrics(
        tmp_path / "results", profile="smoke", run_label="second"
    )
    assert len(selected.attempts) == 1
    assert selected.attempts[0].run_label == "second"
    assert set(selected.raw["_run_label"]) == {"second"}


def test_formal_profile_registration_is_exactly_thirty_seeds(
    tmp_path: Path,
) -> None:
    _write_attempt(
        tmp_path,
        seed=0,
        label="formal",
        profile="formal",
    )
    paths = write_summary_artifacts(
        tmp_path / "results",
        output_dir=tmp_path / "formal-summary",
        profile="formal",
        run_label="formal",
        skip_plots=True,
    )
    conclusion = json.loads(paths["conclusion"].read_text(encoding="utf-8"))
    assert conclusion["coverage"]["expected_seeds"] == list(range(30))
    assert conclusion["coverage"]["missing_seed_count"] == 29
    assert conclusion["complete_primary_coverage"] is False
    assert conclusion["conclusion"] == "inconclusive"
    receipt = conclusion["run_provenance"]["budget_preflight"]
    assert receipt["receipt_schema"] == (
        "exp26_budget_preflight_v2_observed_bound"
    )
    assert receipt["preflight_passed"] is True
    assert receipt["receipt_sha256"] == "a" * 64


def test_formal_summary_requires_explicit_shared_run_label(tmp_path: Path) -> None:
    _write_attempt(
        tmp_path,
        seed=0,
        label="formal",
        profile="formal",
    )

    with pytest.raises(ValueError, match="requires explicit --run-label"):
        collect_metrics(tmp_path / "results", profile="formal")


def test_formal_budget_preflight_receipt_is_required(tmp_path: Path) -> None:
    attempt = _write_attempt(
        tmp_path,
        seed=0,
        label="formal",
        profile="formal",
    )
    for name in ("config.json", "manifest.json"):
        path = attempt / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["evidence_provenance"].pop("budget_preflight")
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lacks budget_preflight binding"):
        collect_metrics(
            tmp_path / "results",
            profile="formal",
            run_label="formal",
        )


def test_formal_budget_receipt_sha_must_match_across_seeds(tmp_path: Path) -> None:
    for seed in (0, 1):
        _write_attempt(
            tmp_path,
            seed=seed,
            label="formal",
            profile="formal",
        )
    attempt = next(
        (tmp_path / "results" / "runs" / EXPERIMENT / "seed_0001").iterdir()
    )
    replacement_sha = "d" * 64
    _update_budget_receipt(
        attempt,
        receipt_updates={"receipt_sha256": replacement_sha},
        row_updates={"preflight_receipt_sha256": replacement_sha},
    )

    with pytest.raises(ValueError, match="inconsistent run provenance"):
        collect_metrics(
            tmp_path / "results",
            profile="formal",
            run_label="formal",
        )


@pytest.mark.parametrize(
    "receipt_updates,match",
    [
        ({"preflight_passed": False}, "identity is invalid"),
        ({"provenance_clean": False}, "cleanliness/stability binding is invalid"),
        (
            {"provenance_stable_during_run": False},
            "cleanliness/stability binding is invalid",
        ),
    ],
)
def test_formal_budget_receipt_requires_passed_clean_stable_preflight(
    tmp_path: Path,
    receipt_updates: dict[str, object],
    match: str,
) -> None:
    attempt = _write_attempt(
        tmp_path,
        seed=0,
        label="formal",
        profile="formal",
    )
    _update_budget_receipt(attempt, receipt_updates=receipt_updates)

    with pytest.raises(ValueError, match=match):
        collect_metrics(
            tmp_path / "results",
            profile="formal",
            run_label="formal",
        )


@pytest.mark.parametrize(
    "receipt_name,row_name",
    [
        ("git_commit", "preflight_git_commit"),
        ("git_tree", "preflight_git_tree"),
    ],
)
def test_formal_budget_receipt_git_must_match_main_panel(
    tmp_path: Path,
    receipt_name: str,
    row_name: str,
) -> None:
    attempt = _write_attempt(
        tmp_path,
        seed=0,
        label="formal",
        profile="formal",
    )
    replacement = "e" * 40
    _update_budget_receipt(
        attempt,
        receipt_updates={receipt_name: replacement},
        row_updates={row_name: replacement},
    )

    with pytest.raises(ValueError, match="cleanliness/stability binding is invalid"):
        collect_metrics(
            tmp_path / "results",
            profile="formal",
            run_label="formal",
        )


def test_smoke_budget_preflight_must_be_null(tmp_path: Path) -> None:
    attempt = _write_attempt(
        tmp_path,
        seed=9000,
        label="registered",
    )
    for name in ("config.json", "manifest.json"):
        path = attempt / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["evidence_provenance"]["budget_preflight"] = {}
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="smoke evidence must use"):
        collect_metrics(
            tmp_path / "results",
            profile="smoke",
            run_label="registered",
        )


def test_attempt_config_must_equal_current_registered_config(tmp_path: Path) -> None:
    attempt = _write_attempt(
        tmp_path,
        seed=9000,
        label="registered",
    )
    config_path = attempt / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["carrier"]["n_neurons"] += 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the current registered"):
        collect_metrics(
            tmp_path / "results",
            profile="smoke",
            run_label="registered",
        )


def test_runtime_provenance_mismatch_fails_closed(tmp_path: Path) -> None:
    attempt = _write_attempt(
        tmp_path,
        seed=9000,
        label="registered",
    )
    config_path = attempt / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["evidence_provenance"]["runtime_versions"]["numpy"] = "0.0.invalid"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path = attempt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence_provenance"]["runtime_versions"]["numpy"] = "0.0.invalid"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific runtime versions are invalid"):
        collect_metrics(
            tmp_path / "results",
            profile="smoke",
            run_label="registered",
        )


def test_cross_seed_provenance_must_be_identical(tmp_path: Path) -> None:
    for seed in (9000, 9001):
        _write_attempt(
            tmp_path,
            seed=seed,
            label="registered",
        )
    attempt = next(
        (tmp_path / "results" / "runs" / EXPERIMENT / "seed_9001").iterdir()
    )
    replacement_tree = "b" * 40
    config_path = attempt / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["evidence_provenance"]["git"]["tree"] = replacement_tree
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path = attempt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence_provenance"]["git"]["tree"] = replacement_tree
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    environment_path = attempt / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["git"]["tree"] = replacement_tree
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    metrics_path = attempt / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    for row in rows:
        row["run_git_tree"] = replacement_tree
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inconsistent run provenance"):
        collect_metrics(
            tmp_path / "results",
            profile="smoke",
            run_label="registered",
        )


def test_cross_seed_raw_config_hash_may_differ_when_canonical_config_matches(
    tmp_path: Path,
) -> None:
    for seed in (9000, 9001):
        _write_attempt(
            tmp_path,
            seed=seed,
            label="registered",
        )
    attempt = next(
        (tmp_path / "results" / "runs" / EXPERIMENT / "seed_9001").iterdir()
    )
    replacement_hash = "c" * 64
    config_path = attempt / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    original_hash = config["evidence_provenance"]["source_config_file_sha256"]
    assert original_hash != replacement_hash
    config["evidence_provenance"]["source_config_file_sha256"] = replacement_hash
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path = attempt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence_provenance"]["source_config_file_sha256"] = replacement_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metrics_path = attempt / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    for row in rows:
        row["source_config_file_sha256"] = replacement_hash
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    collection = collect_metrics(
        tmp_path / "results",
        profile="smoke",
        run_label="registered",
    )
    assert collection.source_config_file_sha256_by_seed == {
        9000: original_hash,
        9001: replacement_hash,
    }
    observed = collection.raw.groupby("seed")[
        "source_config_file_sha256"
    ].unique()
    assert observed.loc[9000].tolist() == [original_hash]
    assert observed.loc[9001].tolist() == [replacement_hash]


def test_analysis_cli_values_are_equality_assertions(tmp_path: Path) -> None:
    for seed in (9000, 9001):
        _write_attempt(
            tmp_path,
            seed=seed,
            label="registered",
        )

    with pytest.raises(ValueError, match="differs from the registered value"):
        write_summary_artifacts(
            tmp_path / "results",
            output_dir=tmp_path / "summary",
            profile="smoke",
            run_label="registered",
            skip_plots=True,
            tie_margin=0.02,
        )
