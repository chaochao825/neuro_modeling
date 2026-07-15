"""Publish an immutable, exact-run-list snapshot for Exp18 ARC results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXPERIMENT = "exp18_arc_recursive_baseline"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_run(run_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = _json_object(run_dir / "config.json")
    status = _json_object(run_dir / "status.json")
    manifest = _json_object(run_dir / "manifest.json")
    if config.get("experiment") != EXPERIMENT or manifest.get("experiment") != EXPERIMENT:
        raise ValueError(f"{run_dir} is not an {EXPERIMENT} run")
    run_id = str(manifest.get("run_id", ""))
    if not run_id:
        raise ValueError(f"{run_dir} has no run_id")
    seed = int(config["seed"])
    if int(status.get("seed", -1)) != seed or int(manifest.get("seed", -1)) != seed:
        raise ValueError(f"{run_dir} has inconsistent seed provenance")
    if status.get("status") not in {"complete", "complete_with_failures"}:
        raise ValueError(f"{run_dir} is partial or failed: {status.get('status')!r}")
    planned_path = run_dir / "planned_conditions.json"
    planned = json.loads(planned_path.read_text(encoding="utf-8"))
    if not isinstance(planned, list) or not planned:
        raise ValueError(f"{run_dir} has no planned condition/task panel")
    planned_keys: list[tuple[str, str]] = []
    for row in planned:
        if not isinstance(row, dict):
            raise ValueError(f"{run_dir} has malformed planned conditions")
        condition = row.get("condition")
        task_id = row.get("task_id")
        if not isinstance(condition, str) or not isinstance(task_id, str):
            raise ValueError(f"{run_dir} planned panel lacks condition/task IDs")
        planned_keys.append((condition, task_id))
    if len(planned_keys) != len(set(planned_keys)):
        raise ValueError(f"{run_dir} planned panel contains duplicates")
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in metrics:
        if row.get("run_id") != run_id:
            raise ValueError(f"{run_dir} mixes run IDs")
        if int(row.get("seed", -1)) != seed:
            raise ValueError(f"{run_dir} mixes independent seeds")
        row["published_run_path"] = str(run_dir.resolve())
    if not metrics:
        raise ValueError(f"{run_dir} completed without metrics")
    task_rows = [row for row in metrics if row.get("level") == "task"]
    observed_keys = [
        (str(row.get("condition")), str(row.get("task_id"))) for row in task_rows
    ]
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != set(
        planned_keys
    ):
        raise ValueError(f"{run_dir} task metrics do not match the planned panel")
    if any(row.get("query_targets_used") is not False for row in task_rows):
        raise ValueError(f"{run_dir} does not prove target-free task inference")
    planned_conditions = {condition for condition, _task_id in planned_keys}
    summaries = [row for row in metrics if row.get("level") == "condition_summary"]
    if (
        {str(row.get("condition")) for row in summaries} != planned_conditions
        or len(summaries) != len(planned_conditions)
    ):
        raise ValueError(f"{run_dir} condition summaries are incomplete")
    for summary in summaries:
        condition = str(summary["condition"])
        expected_count = sum(key[0] == condition for key in planned_keys)
        if (
            int(summary.get("n_tasks", -1)) != expected_count
            or summary.get("query_targets_used") is not False
        ):
            raise ValueError(f"{run_dir} has an invalid condition summary")
    provenance_path = run_dir / "source_provenance.json"
    fit_path = run_dir / "fit_receipts.json"
    provenance = _json_object(provenance_path)
    data_config = dict(config.get("data", {}))
    if (
        provenance.get("test_query_targets_used_for_fit_or_tta") is not False
        or provenance.get("inner_validation_query_targets_used_for_selection")
        is not False
        or not bool(data_config.get("attempt_aware_scoring", False))
    ):
        raise ValueError(f"{run_dir} provenance does not prove target isolation")
    if str(config.get("profile")) == "formal":
        expected_source = {
            "dataset_name": data_config.get("dataset_name"),
            "source_revision": data_config.get("revision"),
            "manifest_sha256": data_config.get("manifest_sha256"),
        }
        if any(provenance.get(key) != value for key, value in expected_source.items()):
            raise ValueError(f"{run_dir} config and source provenance are not bound")
        if (
            provenance.get("source_manifest_verified") is not True
            or provenance.get("source_acquisition_verified") is not True
        ):
            raise ValueError(f"{run_dir} formal source verification is incomplete")
    receipt = {
        "run_id": run_id,
        "seed": seed,
        "profile": str(config.get("profile", "unspecified")),
        "evidence_stage": str(config.get("evidence_stage", "unspecified")),
        "run_status": str(status.get("status")),
        "condition_failures": int(status.get("condition_failures", 0)),
        "condition_invalid": int(status.get("condition_invalid", 0)),
        "published_run_path": str(run_dir.resolve()),
        "config_sha256": _sha256(run_dir / "config.json"),
        "metrics_sha256": _sha256(run_dir / "metrics.jsonl"),
        "planned_conditions_sha256": _sha256(planned_path),
        "fit_receipts_sha256": _sha256(fit_path) if fit_path.is_file() else None,
        "source_provenance_sha256": (
            _sha256(provenance_path) if provenance_path.is_file() else None
        ),
        "formal_claim_promotion_enabled": bool(
            config.get("formal_claim_promotion_enabled", False)
        ),
        "protocol": str(config.get("protocol", "unspecified")),
        "planned_task_conditions": len(planned_keys),
        "query_targets_verified_false": True,
        "dataset_name": str(provenance.get("dataset_name", "unspecified")),
        "dataset_revision": str(provenance.get("source_revision", "unspecified")),
        "source_manifest_sha256": provenance.get("manifest_sha256"),
    }
    return metrics, receipt


def _seed_level_comparison(comparisons: pd.DataFrame) -> dict[str, object]:
    if comparisons.empty:
        return {
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_seeds": 0,
            "conclusion": "inconclusive",
        }
    values = comparisons["pass_at_2_difference"].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("comparison contains non-finite differences")
    if len(values) < 2:
        low = high = float(values.mean())
    else:
        rng = np.random.default_rng(18)
        indices = rng.integers(0, len(values), size=(10_000, len(values)))
        draws = values[indices].mean(axis=1)
        low, high = (float(value) for value in np.quantile(draws, [0.025, 0.975]))
    estimate = float(values.mean())
    conclusion = "support" if low > 0 else "oppose" if high < 0 else "inconclusive"
    # One seed can never promote a mechanism claim even when its task bootstrap
    # happens to exclude zero.
    if len(values) < 3:
        conclusion = "inconclusive"
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "n_seeds": len(values),
        "conclusion": conclusion,
    }


def publish_snapshot(
    run_dirs: Sequence[str | Path],
    results_root: str | Path,
    *,
    prefix: str = "exp18_arc_recursive_canary",
) -> dict[str, Path]:
    """Publish only explicitly supplied run directories; never auto-select."""

    if not run_dirs:
        raise ValueError("at least one exact run directory is required")
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be one path component")
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": root / f"{prefix}_raw.csv.gz",
        "conditions": root / f"{prefix}_conditions.csv",
        "comparison": root / f"{prefix}_comparison.csv",
        "manifest": root / f"{prefix}_run_manifest.csv",
        "report": root / f"{prefix}_report.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Exp18 snapshots are immutable; choose another prefix: "
            + ", ".join(str(path) for path in existing)
        )
    all_metrics: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in run_dirs:
        run_dir = Path(value)
        metrics, receipt = _load_run(run_dir)
        run_id = str(receipt["run_id"])
        if run_id in seen:
            raise ValueError("duplicate Exp18 run supplied")
        seen.add(run_id)
        all_metrics.extend(metrics)
        receipts.append(receipt)
    raw = pd.DataFrame(all_metrics)
    manifest = pd.DataFrame(receipts).sort_values(["seed", "run_id"])
    if manifest["seed"].duplicated().any():
        raise ValueError("Exp18 snapshots require one exact run per independent seed")
    dataset_keys = set(
        zip(
            manifest["dataset_name"].astype(str),
            manifest["dataset_revision"].astype(str),
            strict=True,
        )
    )
    if len(dataset_keys) != 1:
        raise ValueError("Exp18 snapshots cannot mix ARC datasets or revisions")
    if not manifest["query_targets_verified_false"].all():
        raise ValueError("Exp18 snapshot contains an unverified target-access run")
    condition_rows = raw.loc[raw.get("level", pd.Series(dtype=str)).eq("condition_summary")].copy()
    comparison_rows = raw.loc[raw.get("level", pd.Series(dtype=str)).eq("registered_comparison")].copy()
    if not condition_rows.empty:
        condition_rows = condition_rows.sort_values(["seed", "condition"])
    if not comparison_rows.empty:
        comparison_rows = comparison_rows.sort_values(["seed", "comparison"])
    seed_summary = _seed_level_comparison(comparison_rows)
    raw.to_csv(paths["raw"], index=False, compression="gzip")
    condition_rows.to_csv(paths["conditions"], index=False)
    comparison_rows.to_csv(paths["comparison"], index=False)
    manifest.to_csv(paths["manifest"], index=False)
    lines = [
        "# Exp18 ARC recursive baseline report",
        "",
        f"- Exact runs: {len(manifest)}",
        f"- Seeds: {', '.join(str(value) for value in manifest['seed'].tolist())}",
        f"- Dataset/revision: {manifest['dataset_name'].iloc[0]} / "
        f"{manifest['dataset_revision'].iloc[0]}",
        f"- Protocols: {', '.join(sorted(set(manifest['protocol'].astype(str))))}",
        f"- Run failures/invalid: {int(manifest['condition_failures'].sum())}/"
        f"{int(manifest['condition_invalid'].sum())}",
        "- Test query targets used for fit/TTA: false by protocol",
        "- Official/private leaderboard score: false",
        "",
        "## Registered recursive comparison",
        "",
        f"- Mean pass@2 difference: {seed_summary['estimate']}",
        f"- Seed-bootstrap 95% CI: [{seed_summary['ci_low']}, {seed_summary['ci_high']}]",
        f"- Independent seeds: {seed_summary['n_seeds']}",
        f"- Conclusion: **{seed_summary['conclusion']}**",
        "",
        "This is an independent micro-TRM-like BPTT baseline. It is not the official "
        "7M TRM reproduction and cannot by itself support the local-learning claim.",
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="exp18_arc_recursive_canary")
    args = parser.parse_args()
    paths = publish_snapshot(args.runs, args.results_root, prefix=args.prefix)
    print("\n".join(f"{name}: {path}" for name, path in paths.items()))


if __name__ == "__main__":
    main()
