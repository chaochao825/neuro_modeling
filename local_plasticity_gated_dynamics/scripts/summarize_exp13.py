"""Build a fail-closed formal exp13 snapshot from immutable seed runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import load_json_config  # noqa: E402
from src.analysis.structured_benchmark import STRUCTURED_CONDITIONS  # noqa: E402
from src.analysis.structured_formal import summarize_structured_formal  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _metric_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} contains a non-object record")
    return records


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_attempt_config(
    formal_config: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    """Reproduce the exact immutable config written by ``run_seed``."""

    run_config = {
        **formal_config,
        "training_algorithm": "matched_hybrid_structured_candidate_selection",
        "used_autograd": "baseline_only",
        "parent_checkpoint": None,
        "spiking_model": False,
        "neural_evidence_claim": False,
    }
    return {
        "experiment": "exp13_structured_reasoning",
        "seed": int(seed),
        **run_config,
    }


def _latest_matching_attempt(
    results_root: Path,
    *,
    seed: int,
    expected_config: dict[str, Any],
) -> Path:
    seed_root = (
        results_root / "runs" / "exp13_structured_reasoning" / f"seed_{seed:04d}"
    )
    for attempt in sorted(seed_root.glob("*"), reverse=True):
        config_path = attempt / "config.json"
        if not config_path.is_file():
            continue
        config = _read_json(config_path)
        if config == expected_config:
            return attempt
    raise FileNotFoundError(
        "no exp13 attempt exactly matches the complete registered formal config "
        f"for seed={seed}"
    )


def collect_formal_runs(
    results_root: Path,
    *,
    expected_seeds: list[int],
    family: str,
    revision: str,
    source_manifest_sha256: str,
    dataset_name: str,
    test_split_role: str,
    formal_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registered_data = formal_config.get("data")
    if (
        formal_config.get("profile") != "formal"
        or formal_config.get("family") != family
        or formal_config.get("hyperparameters_frozen_before_test") is not True
        or not isinstance(registered_data, dict)
        or registered_data.get("revision") != revision
        or registered_data.get("manifest_sha256") != source_manifest_sha256
        or registered_data.get("dataset_name") != dataset_name
        or registered_data.get("test_split_role") != test_split_role
    ):
        raise ValueError("collector arguments disagree with the complete formal config")
    formal_config_sha256 = _canonical_sha256(formal_config)
    raw_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    for seed in expected_seeds:
        expected_config = _expected_attempt_config(formal_config, seed=seed)
        attempt = _latest_matching_attempt(
            results_root,
            seed=seed,
            expected_config=expected_config,
        )
        status = _read_json(attempt / "status.json")
        environment = _read_json(attempt / "environment.json")
        manifest = _read_json(attempt / "manifest.json")
        planned = json.loads(
            (attempt / "planned_conditions.json").read_text(encoding="utf-8")
        )
        metrics = _metric_records(attempt / "metrics.jsonl")
        git = environment.get("git", {})
        if status.get("status") != "complete":
            raise ValueError(f"latest seed {seed} attempt is not complete")
        if not isinstance(git, dict) or git.get("dirty") is not False:
            raise ValueError(f"latest seed {seed} attempt is not from a clean worktree")
        planned_conditions = {str(item.get("condition")) for item in planned}
        if planned_conditions != set(STRUCTURED_CONDITIONS):
            raise ValueError(f"seed {seed} has an incomplete planned condition family")
        if (
            len(metrics) != len(STRUCTURED_CONDITIONS)
            or {str(item.get("condition")) for item in metrics}
            != set(STRUCTURED_CONDITIONS)
            or {str(item.get("status")) for item in metrics} != {"complete"}
        ):
            raise ValueError(f"seed {seed} metric records are incomplete")
        by_condition = {str(item["condition"]): item for item in metrics}
        if not all(bool(item.get("formal_evidence_eligible")) for item in metrics):
            raise ValueError(f"seed {seed} contains an evidence-ineligible condition")
        if not all(
            item.get("hyperparameters_frozen_before_test") is True for item in metrics
        ):
            raise ValueError(f"seed {seed} did not freeze hyperparameters before test")
        if {str(item.get("test_split_role")) for item in metrics} != {test_split_role}:
            raise ValueError(f"seed {seed} metric records use a different split role")
        if {str(item.get("source_manifest_sha256")) for item in metrics} != {
            source_manifest_sha256
        }:
            raise ValueError(
                f"seed {seed} metric records use a different source manifest"
            )
        if {str(item.get("source_revision")) for item in metrics} != {revision}:
            raise ValueError(f"seed {seed} metric records use a different revision")
        if {str(item.get("dataset_name")) for item in metrics} != {dataset_name}:
            raise ValueError(f"seed {seed} metric records use a different dataset")
        task_path = attempt / "task_metrics.csv.gz"
        task = pd.read_csv(task_path)
        if set(task["condition"].astype(str)) != set(STRUCTURED_CONDITIONS):
            raise ValueError(f"seed {seed} task panel is incomplete")
        for condition, record in by_condition.items():
            mask = task["condition"].astype(str) == condition
            for column in (
                "parameter_count",
                "trainable_parameter_count",
                "used_bptt",
                "control_dim",
                "control_operator_rank",
                "formal_evidence_eligible",
                "core_support_eligible",
                "hyperparameters_frozen_before_test",
                "test_split_role",
            ):
                task.loc[mask, column] = record[column]
        task["run_id"] = str(manifest["run_id"])
        task["run_git_commit"] = str(git["commit"])
        task["run_git_dirty"] = bool(git["dirty"])
        task["source_revision"] = revision
        task["source_manifest_sha256"] = source_manifest_sha256
        task["dataset_name"] = dataset_name
        task["formal_config_sha256"] = formal_config_sha256
        raw_frames.append(task)
        run_rows.append(
            {
                "seed": seed,
                "run_id": manifest["run_id"],
                "attempt_name": attempt.name,
                "git_commit": git["commit"],
                "git_dirty": git["dirty"],
                "status": status["status"],
                "source_manifest_sha256": source_manifest_sha256,
                "source_revision": revision,
                "dataset_name": dataset_name,
                "test_split_role": test_split_role,
                "formal_config_sha256": formal_config_sha256,
                "config_sha256": _sha256(attempt / "config.json"),
                "environment_sha256": _sha256(attempt / "environment.json"),
                "metrics_sha256": _sha256(attempt / "metrics.jsonl"),
                "task_metrics_sha256": _sha256(task_path),
                "manifest_sha256": _sha256(attempt / "manifest.json"),
            }
        )
    return pd.concat(raw_frames, ignore_index=True), pd.DataFrame(run_rows)


def _write_report(
    path: Path,
    conditions: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    revision: str,
    raw_sha256: str,
    run_manifest_sha256: str,
    family: str,
    dataset_name: str,
) -> None:
    core = comparisons.set_index("comparison").loc["hierarchical_vs_flat"]
    lines = [
        f"# Exp13 {family.upper()} formal report",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Dataset revision: `{revision}`",
        f"- Raw task panel SHA-256: `{raw_sha256}`",
        f"- Clean-run manifest SHA-256: `{run_manifest_sha256}`",
        "- Statistical unit: source/augmentation dependency component; seeds are nested within task.",
        "- Scope: matched hybrid proposal selection only; no neural or biological claim.",
        f"- Candidate coverage gate: {float(core['candidate_coverage']):.4f} "
        f"(required {float(core['minimum_candidate_coverage']):.4f}; "
        f"passed={bool(core['coverage_gate_passed'])}).",
        "",
        "## Absolute performance",
        "",
        "| Condition | Exact accuracy | 95% CI | Coverage | Parameters | BPTT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in conditions.to_dict("records"):
        lines.append(
            f"| {row['condition']} | {row['exact_accuracy']:.4f} | "
            f"[{row['exact_accuracy_ci_low']:.4f}, {row['exact_accuracy_ci_high']:.4f}] | "
            f"{row['candidate_coverage']:.4f} | {int(row['parameter_count'])} | "
            f"{bool(row['used_bptt'])} |"
        )
    lines.extend(
        [
            "",
            "## Registered paired comparisons",
            "",
            "| Comparison | Estimate | 95% CI | Holm p | Conclusion |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in comparisons.to_dict("records"):
        lines.append(
            f"| {row['comparison']} | {row['estimate']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{row['wilcoxon_p_holm']:.4g} | {row['conclusion']} |"
        )
    lines.extend(
        [
            "",
            "## Core conclusion",
            "",
            f"`hierarchical_local > flat_local`: **{core['conclusion']}** "
            f"(difference {core['estimate']:.4f}, 95% CI "
            f"[{core['ci_low']:.4f}, {core['ci_high']:.4f}]).",
            "",
            "This conclusion cannot be promoted to end-to-end neural reasoning: the same "
            "deterministic program/search proposal library is supplied to every selector. "
            f"{family.upper()} does not replace the pending multi-session neural-activity "
            "experiment.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/formal/exp13_structured_reasoning_arc.json",
    )
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-prefix")
    parser.add_argument("--n-bootstrap", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    args = parser.parse_args()

    config = load_json_config(args.config)
    data = dict(config["data"])
    expected_seeds = [int(seed) for seed in config["seeds"]]
    results_root = Path(args.results_root)
    raw, run_manifest = collect_formal_runs(
        results_root,
        expected_seeds=expected_seeds,
        family=str(config["family"]),
        revision=str(data["revision"]),
        source_manifest_sha256=str(data["manifest_sha256"]),
        dataset_name=str(data["dataset_name"]),
        test_split_role=str(data["test_split_role"]),
        formal_config=config,
    )
    family = str(config["family"]).strip().lower()
    if family not in {"arc", "maze", "sudoku"}:
        raise ValueError("formal exp13 family must be arc, maze, or sudoku")
    prefix = str(args.output_prefix or f"exp13_{family}_formal")
    raw_path = results_root / f"{prefix}_raw.csv.gz"
    run_manifest_path = results_root / f"{prefix}_run_manifest.csv"
    raw.to_csv(
        raw_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    run_manifest.to_csv(run_manifest_path, index=False)
    raw_sha = _sha256(raw_path)
    run_manifest_sha = _sha256(run_manifest_path)
    conditions, comparisons = summarize_structured_formal(
        raw,
        expected_seeds=expected_seeds,
        seed=int(args.bootstrap_seed),
        n_bootstrap=int(args.n_bootstrap),
        minimum_candidate_coverage=float(config.get("minimum_candidate_coverage", 0.9)),
        task_family=family,
        test_split_role=str(data["test_split_role"]),
    )
    bindings = {
        "task_family": family,
        "dataset_name": data["dataset_name"],
        "source_revision": data["revision"],
        "source_manifest_sha256": data["manifest_sha256"],
        "test_split_role": data["test_split_role"],
        "formal_config_sha256": _canonical_sha256(config),
        "scoped_raw_sha256": raw_sha,
        "run_manifest_sha256": run_manifest_sha,
        "run_git_commit": run_manifest["git_commit"].iloc[0],
        "run_git_dirty": False,
    }
    for key, value in bindings.items():
        conditions[key] = value
        comparisons[key] = value
    condition_path = results_root / f"{prefix}_conditions.csv"
    comparison_path = results_root / f"{prefix}_comparisons.csv"
    conditions.to_csv(condition_path, index=False)
    comparisons.to_csv(comparison_path, index=False)
    _write_report(
        results_root / f"{prefix}_report.md",
        conditions,
        comparisons,
        revision=str(data["revision"]),
        raw_sha256=raw_sha,
        run_manifest_sha256=run_manifest_sha,
        family=family,
        dataset_name=str(data["dataset_name"]),
    )


if __name__ == "__main__":
    main()
