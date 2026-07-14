"""Freeze an Exp17 candidate from validation-only multi-seed receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


EXPERIMENT = "exp17_tiny_recursive_calibration"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)

    def cell(value: object) -> str:
        if isinstance(value, float):
            value = f"{value:.6g}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _metric_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(row.get("experiment") != EXPERIMENT for row in rows):
        raise ValueError(f"{run_dir} contains non-Exp17 metrics")
    if any(row.get("stage") == "task_test" for row in rows):
        raise ValueError("Exp17 calibration artifacts must not contain test rows")
    if any(row.get("test_data_used_for_fit_or_selection") is True for row in rows):
        raise ValueError("Exp17 calibration artifacts report test access")
    if any(row.get("hidden_target_scorer_called") is True for row in rows):
        raise ValueError("Exp17 calibration artifacts report hidden-target scoring")
    if any(row.get("test_prediction_array_requested") is True for row in rows):
        raise ValueError("Exp17 calibration artifacts report test prediction access")
    return rows


def summarize_runs(run_dirs: Iterable[str | Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = tuple(Path(path).resolve() for path in run_dirs)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("run directories must be non-empty and unique")
    all_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    minimum_seeds = 1
    require_clean_git: bool | None = None
    expected_candidates: set[str] | None = None
    for run_dir in paths:
        config = _read_json(run_dir / "config.json")
        status = _read_json(run_dir / "status.json")
        provenance = _read_json(run_dir / "source_provenance.json")
        environment = _read_json(run_dir / "environment.json")
        if config.get("test_access_forbidden") is not True:
            raise ValueError("run config does not enforce test-free calibration")
        if provenance.get("test_data_used_for_fit_or_selection") is not False:
            raise ValueError("run provenance does not certify test-free calibration")
        contract = config.get("selection_contract", {})
        run_requires_clean_git = bool(contract.get("require_clean_git", False))
        require_clean_git = (
            run_requires_clean_git
            if require_clean_git is None
            else require_clean_git
        )
        if run_requires_clean_git != require_clean_git:
            raise ValueError("clean-git requirements differ across Exp17 seeds")
        minimum_seeds = max(
            minimum_seeds, int(contract.get("minimum_confirmation_seeds", 1))
        )
        planned = json.loads(
            (run_dir / "planned_conditions.json").read_text(encoding="utf-8")
        )
        if not isinstance(planned, list):
            raise ValueError("planned_conditions.json must contain a list")
        names = {str(row["condition"]) for row in planned}
        expected_candidates = names if expected_candidates is None else expected_candidates
        if names != expected_candidates:
            raise ValueError("planned candidates differ across Exp17 seeds")
        rows = _metric_rows(run_dir)
        all_rows.extend(rows)
        git = environment.get("git", {})
        if not isinstance(git, dict):
            raise ValueError("run environment lacks a git provenance mapping")
        manifests.append(
            {
                "seed": int(status["seed"]),
                "run_id": next((row["run_id"] for row in rows), None),
                "run_path": str(run_dir),
                "run_status": status["status"],
                "condition_failures": int(status.get("condition_failures", 0)),
                "semantic_config_sha256": config["semantic_config_sha256"],
                "calibration_code_sha256": config["calibration_code_sha256"],
                "formal_data_validation_required": config.get("profile") == "formal",
                "git_commit": git.get("commit"),
                "git_dirty": git.get("dirty"),
                "metrics_sha256": _sha256(run_dir / "metrics.jsonl"),
                "receipts_sha256": _sha256(run_dir / "fit_receipts.json"),
                "test_prediction_array_requested": False,
                "hidden_target_scorer_called": False,
            }
        )
    manifest = pd.DataFrame(manifests).sort_values("seed", ignore_index=True)
    if manifest["seed"].duplicated().any():
        raise ValueError("cross-seed freeze requires one run per seed")
    if manifest["semantic_config_sha256"].nunique() != 1:
        raise ValueError("semantic Exp17 configs differ across submitted seeds")
    if manifest["calibration_code_sha256"].nunique() != 1:
        raise ValueError("calibration code identity differs across submitted seeds")
    if manifest["formal_data_validation_required"].nunique() != 1:
        raise ValueError("formal data validation modes differ across Exp17 seeds")
    if (
        manifest["git_commit"].isna().any()
        or manifest["git_commit"].nunique() != 1
    ):
        raise ValueError("Exp17 seeds must share one known git commit")
    candidate_columns = [
        "candidate",
        "seed",
        "selected_validation_blank_cell_accuracy",
        "selected_validation_exact_accuracy",
        "selected_train_blank_cell_accuracy",
        "parameter_count",
        "optimizer_steps",
        "candidate_config_sha256",
    ]
    candidate_rows = pd.DataFrame(
        [row for row in all_rows if row.get("stage") == "calibration_candidate"],
        columns=candidate_columns,
    )
    summary_rows: list[dict[str, Any]] = []
    for candidate in sorted(expected_candidates or ()):
        frame = candidate_rows[candidate_rows["candidate"] == candidate]
        candidate_hashes = frame["candidate_config_sha256"].dropna().unique()
        hash_consistent = len(candidate_hashes) == 1
        summary_rows.append(
            {
                "candidate": candidate,
                "n_seeds_complete": int(frame["seed"].nunique()) if len(frame) else 0,
                "mean_validation_blank_cell_accuracy": float(
                    frame["selected_validation_blank_cell_accuracy"].mean()
                )
                if len(frame)
                else float("nan"),
                "mean_validation_exact_accuracy": float(
                    frame["selected_validation_exact_accuracy"].mean()
                )
                if len(frame)
                else float("nan"),
                "mean_train_blank_cell_accuracy": float(
                    frame["selected_train_blank_cell_accuracy"].mean()
                )
                if len(frame)
                else float("nan"),
                "mean_parameter_count": float(frame["parameter_count"].mean())
                if len(frame)
                else float("nan"),
                "mean_optimizer_steps": float(frame["optimizer_steps"].mean())
                if len(frame)
                else float("nan"),
                "candidate_config_sha256": (
                    str(candidate_hashes[0]) if hash_consistent else None
                ),
                "candidate_config_hash_consistent": hash_consistent,
                "complete_on_all_submitted_seeds": bool(
                    len(frame) and frame["seed"].nunique() == manifest["seed"].nunique()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    enough_seeds = manifest["seed"].nunique() >= minimum_seeds
    all_runs_clean = bool(
        (manifest["run_status"] == "complete").all()
        and (manifest["condition_failures"] == 0).all()
    )
    all_git_clean = bool((manifest["git_dirty"] == False).all())  # noqa: E712
    all_candidates_complete = bool(
        len(summary) == len(expected_candidates or ())
        and summary["complete_on_all_submitted_seeds"].all()
        and summary["candidate_config_hash_consistent"].all()
    )
    freeze_gates_passed = bool(
        enough_seeds
        and all_runs_clean
        and all_candidates_complete
        and (not require_clean_git or all_git_clean)
    )
    eligible = summary.copy() if freeze_gates_passed else summary.iloc[0:0].copy()
    if eligible.empty:
        decision: dict[str, Any] = {
            "status": "insufficient_validation_evidence",
            "selected_candidate": None,
            "minimum_confirmation_seeds": minimum_seeds,
            "submitted_seeds": manifest["seed"].tolist(),
            "enough_seeds": enough_seeds,
            "all_runs_clean": all_runs_clean,
            "require_clean_git": bool(require_clean_git),
            "all_git_clean": all_git_clean,
            "all_candidates_complete": all_candidates_complete,
            "all_freeze_gates_passed": False,
            "git_commit": str(manifest["git_commit"].iloc[0]),
            "calibration_code_sha256": str(
                manifest["calibration_code_sha256"].iloc[0]
            ),
            "dataset_adapter_loaded_test_records": True,
            "formal_data_validation_required": bool(
                manifest["formal_data_validation_required"].iloc[0]
            ),
            "test_prediction_array_requested": False,
            "hidden_target_scorer_called": False,
            "formal_claim_promotion_enabled": False,
            "claim_conclusion": "inconclusive",
        }
    else:
        eligible = eligible.sort_values(
            [
                "mean_validation_blank_cell_accuracy",
                "mean_validation_exact_accuracy",
                "mean_parameter_count",
                "candidate",
            ],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        selected = eligible.iloc[0]
        decision = {
            "status": "frozen_validation_only",
            "selected_candidate": str(selected["candidate"]),
            "selected_candidate_config_sha256": str(
                selected["candidate_config_sha256"]
            ),
            "primary_metric": "mean_validation_blank_cell_accuracy",
            "primary_metric_value": float(
                selected["mean_validation_blank_cell_accuracy"]
            ),
            "minimum_confirmation_seeds": minimum_seeds,
            "submitted_seeds": manifest["seed"].tolist(),
            "enough_seeds": enough_seeds,
            "all_runs_clean": all_runs_clean,
            "require_clean_git": bool(require_clean_git),
            "all_git_clean": all_git_clean,
            "all_candidates_complete": all_candidates_complete,
            "all_freeze_gates_passed": True,
            "git_commit": str(manifest["git_commit"].iloc[0]),
            "calibration_code_sha256": str(
                manifest["calibration_code_sha256"].iloc[0]
            ),
            "dataset_adapter_loaded_test_records": True,
            "formal_data_validation_required": bool(
                manifest["formal_data_validation_required"].iloc[0]
            ),
            "test_prediction_array_requested": False,
            "hidden_target_scorer_called": False,
            "confirmation_test_still_required": True,
            "formal_claim_promotion_enabled": False,
            "claim_conclusion": "inconclusive",
        }
    return summary, {"decision": decision, "manifest": manifest}


def publish_summary(
    run_dirs: Iterable[str | Path], output_dir: str | Path, *, prefix: str
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidates": output / f"{prefix}_candidates.csv",
        "manifest": output / f"{prefix}_run_manifest.csv",
        "decision": output / f"{prefix}_freeze_decision.json",
        "report": output / f"{prefix}_report.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"calibration publication is immutable: {existing[0]}")
    candidates, payload = summarize_runs(run_dirs)
    manifest: pd.DataFrame = payload["manifest"]
    decision: dict[str, Any] = payload["decision"]
    candidates.to_csv(paths["candidates"], index=False)
    manifest.to_csv(paths["manifest"], index=False)
    paths["decision"].write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Exp17 tiny-recursive calibration",
        "",
        "This is a train/inner-validation-only calibration artifact. The dataset "
        "adapter loaded its opaque capability store, but no test prediction array "
        "was requested and the hidden-target scorer was not called.",
        "",
        f"- Status: `{decision['status']}`",
        f"- Selected candidate: `{decision['selected_candidate']}`",
        f"- Submitted seeds: `{decision['submitted_seeds']}`",
        "- Confirmation on an independently frozen test panel is still required.",
        "- Claim conclusion: **inconclusive**.",
        "",
        "## Validation-only candidate summary",
        "",
        _markdown_table(candidates),
        "",
    ]
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--prefix", default="exp17_tiny_recursive_calibration")
    args = parser.parse_args()
    outputs = publish_summary(args.run_dir, args.output_dir, prefix=args.prefix)
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
