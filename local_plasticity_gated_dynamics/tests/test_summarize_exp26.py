"""Contracts for the standalone Exp26 phase-diagram summarizer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.summarize_exp26 import (
    EXPERIMENT,
    collect_metrics,
    write_summary_artifacts,
)


MODES = ("frozen", "routing", "gain", "low_rank", "rgl")


def _rows(seed: int, *, retain_rgl_failure: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    chi_grid = np.linspace(0.05, 0.95, 10)
    alpha_grid = np.linspace(0.05, 0.95, 10)
    for split in ("discovery", "heldout"):
        for index, chi in enumerate(chi_grid):
            alpha = alpha_grid[(3 * index) % len(alpha_grid)]
            values = {
                "frozen": 0.50,
                "routing": 0.90 - 0.35 * chi,
                "gain": 0.88 - 0.35 * chi,
                "low_rank": 0.55 + 0.35 * chi,
                "rgl": 0.92,
            }
            for mode in MODES:
                failed = bool(
                    retain_rgl_failure
                    and split == "heldout"
                    and index == 0
                    and mode == "rgl"
                )
                rows.append(
                    {
                        "run_id": f"run-{seed}",
                        "experiment": EXPERIMENT,
                        "seed": seed,
                        "generator_id": f"{split}-{index}",
                        "generator_split": split,
                        "actuator_mode": mode,
                        "condition": mode,
                        "alpha": float(alpha),
                        "chi": float(chi),
                        "transition_rank": 2,
                        "input_rank": 1,
                        "delay_steps": 0,
                        "noise_std": 0.01,
                        "validation_balanced_accuracy": float(values[mode]),
                        "test_balanced_accuracy": float(values[mode]),
                        "energy_proxy": 1.0 + 0.1 * index,
                        "status": "failed" if failed else "complete",
                        "functional_budget_valid": not failed,
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
    rows: list[dict[str, object]],
    profile: str = "smoke",
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
    config = {
        "experiment": EXPERIMENT,
        "seed": seed,
        "profile": profile,
        "dev_only": profile == "smoke",
        "seeds": [9000, 9001] if profile == "smoke" else list(range(30)),
        "run_label": label,
    }
    failures = sum(row["status"] == "failed" for row in rows)
    status = "complete_with_failures" if failures else "complete"
    (attempt / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    (attempt / "status.json").write_text(
        json.dumps(
            {
                "status": status,
                "condition_failures": failures,
                "condition_invalid": 0,
            }
        ),
        encoding="utf-8",
    )
    (attempt / "planned_conditions.json").write_text(
        json.dumps(
            [
                {
                    "condition_index": index,
                    "generator_id": row["generator_id"],
                    "generator_split": row["generator_split"],
                    "actuator_mode": row["actuator_mode"],
                    "alpha": row["alpha"],
                    "transition_rank": row["transition_rank"],
                    "input_rank": row["input_rank"],
                    "delay": row["delay_steps"],
                    "noise_std": row["noise_std"],
                }
                for index, row in enumerate(rows)
            ]
        ),
        encoding="utf-8",
    )
    (attempt / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return attempt


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
            rows=_rows(seed, retain_rgl_failure=seed == 9001),
        )
    paths = write_summary_artifacts(
        tmp_path / "results",
        output_dir=tmp_path / "summary",
        profile="smoke",
        run_label="registered",
        skip_plots=True,
        bootstrap_samples=200,
        permutation_samples=200,
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

    assert len(raw) == 200
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


def test_multiple_runs_require_explicit_run_label(tmp_path: Path) -> None:
    for label in ("first", "second"):
        _write_attempt(
            tmp_path,
            seed=9000,
            label=label,
            rows=_rows(9000),
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
        rows=_rows(0),
        profile="formal",
    )
    paths = write_summary_artifacts(
        tmp_path / "results",
        output_dir=tmp_path / "formal-summary",
        profile="formal",
        run_label="formal",
        skip_plots=True,
        bootstrap_samples=100,
        permutation_samples=100,
    )
    conclusion = json.loads(paths["conclusion"].read_text(encoding="utf-8"))
    assert conclusion["coverage"]["expected_seeds"] == list(range(30))
    assert conclusion["coverage"]["missing_seed_count"] == 29
    assert conclusion["complete_primary_coverage"] is False
    assert conclusion["conclusion"] == "inconclusive"
