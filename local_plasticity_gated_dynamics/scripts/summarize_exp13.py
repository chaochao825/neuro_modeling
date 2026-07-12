"""Build a fail-closed formal ARC exp13 snapshot from immutable seed runs."""

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


def _latest_matching_attempt(
    results_root: Path,
    *,
    seed: int,
    family: str,
    revision: str,
) -> Path:
    seed_root = (
        results_root
        / "runs"
        / "exp13_structured_reasoning"
        / f"seed_{seed:04d}"
    )
    for attempt in sorted(seed_root.glob("*"), reverse=True):
        config_path = attempt / "config.json"
        if not config_path.is_file():
            continue
        config = _read_json(config_path)
        data = config.get("data", {})
        if (
            config.get("profile") == "formal"
            and config.get("family") == family
            and isinstance(data, dict)
            and data.get("revision") == revision
        ):
            return attempt
    raise FileNotFoundError(
        f"no formal {family} exp13 attempt for seed={seed}, revision={revision}"
    )


def collect_formal_runs(
    results_root: Path,
    *,
    expected_seeds: list[int],
    family: str,
    revision: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    for seed in expected_seeds:
        attempt = _latest_matching_attempt(
            results_root, seed=seed, family=family, revision=revision
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
            ):
                task.loc[mask, column] = record[column]
        task["run_id"] = str(manifest["run_id"])
        task["run_git_commit"] = str(git["commit"])
        task["run_git_dirty"] = bool(git["dirty"])
        task["source_revision"] = revision
        raw_frames.append(task)
        run_rows.append(
            {
                "seed": seed,
                "run_id": manifest["run_id"],
                "attempt_name": attempt.name,
                "git_commit": git["commit"],
                "git_dirty": git["dirty"],
                "status": status["status"],
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
) -> None:
    core = comparisons.set_index("comparison").loc["hierarchical_vs_flat"]
    lines = [
        "# Exp13 ARC formal report",
        "",
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
            "deterministic program proposal library is supplied to every selector. ARC does "
            "not replace the pending multi-session neural-activity experiment.",
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
    parser.add_argument("--output-prefix", default="exp13_arc_formal")
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
    )
    prefix = str(args.output_prefix)
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
        minimum_candidate_coverage=float(
            config.get("minimum_candidate_coverage", 0.9)
        ),
    )
    bindings = {
        "source_revision": data["revision"],
        "source_manifest_sha256": data["manifest_sha256"],
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
    )


if __name__ == "__main__":
    main()
