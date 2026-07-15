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
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in metrics:
        if row.get("run_id") != run_id:
            raise ValueError(f"{run_dir} mixes run IDs")
        row["published_run_path"] = str(run_dir.resolve())
    if not metrics:
        metrics.append(
            {
                "run_id": run_id,
                "experiment": EXPERIMENT,
                "seed": int(config["seed"]),
                "level": "run_status",
                "status": status.get("status"),
                "published_run_path": str(run_dir.resolve()),
            }
        )
    provenance_path = run_dir / "source_provenance.json"
    fit_path = run_dir / "fit_receipts.json"
    receipt = {
        "run_id": run_id,
        "seed": int(config["seed"]),
        "profile": str(config.get("profile", "unspecified")),
        "evidence_stage": str(config.get("evidence_stage", "unspecified")),
        "run_status": str(status.get("status")),
        "condition_failures": int(status.get("condition_failures", 0)),
        "condition_invalid": int(status.get("condition_invalid", 0)),
        "published_run_path": str(run_dir.resolve()),
        "config_sha256": _sha256(run_dir / "config.json"),
        "metrics_sha256": _sha256(run_dir / "metrics.jsonl"),
        "fit_receipts_sha256": _sha256(fit_path) if fit_path.is_file() else None,
        "source_provenance_sha256": (
            _sha256(provenance_path) if provenance_path.is_file() else None
        ),
        "formal_claim_promotion_enabled": bool(
            config.get("formal_claim_promotion_enabled", False)
        ),
        "protocol": str(config.get("protocol", "unspecified")),
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

