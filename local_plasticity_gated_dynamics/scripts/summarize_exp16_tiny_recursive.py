"""Publish an exact-run-list snapshot for the Exp16 micro-TRM-like audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp16_tiny_recursive_sudoku import CONDITIONS  # noqa: E402


EXPERIMENT = "exp16_tiny_recursive_sudoku"
COMPARISON = "micro_trm_minus_flat_compute_matched"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _bootstrap_mean(
    values: np.ndarray, *, draws: int = 10_000, seed: int = 16
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _load_run(run_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = _read_json(run_dir / "config.json")
    status = _read_json(run_dir / "status.json")
    manifest = _read_json(run_dir / "manifest.json")
    if config.get("experiment") != EXPERIMENT or manifest.get("experiment") != EXPERIMENT:
        raise ValueError(f"{run_dir} is not an {EXPERIMENT} run")
    planned = json.loads((run_dir / "planned_conditions.json").read_text("utf-8"))
    if {row["condition"] for row in planned} != set(CONDITIONS):
        raise ValueError("Exp16 publication requires the two frozen conditions")
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    relative = "/".join(run_dir.parts[-4:])
    for row in metrics:
        row["published_run_path"] = relative
    environment = _read_json(run_dir / "environment.json")
    git = environment.get("git") if isinstance(environment.get("git"), dict) else {}
    receipt = {
        "seed": int(config["seed"]),
        "profile": str(config.get("profile", "unspecified")),
        "run_status": str(status.get("status")),
        "condition_failures": int(status.get("condition_failures", 0)),
        "published_run_path": relative,
        "metrics_sha256": _sha256(run_dir / "metrics.jsonl"),
        "config_sha256": _sha256(run_dir / "config.json"),
        "fit_receipts_sha256": _sha256(run_dir / "fit_receipts.json"),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
    }
    return metrics, receipt


def publish_snapshot(
    run_dirs: Sequence[str | Path],
    results_root: str | Path,
    *,
    prefix: str = "exp16_tiny_recursive_smoke",
) -> dict[str, Path]:
    """Publish only the explicitly supplied immutable runs; never auto-select."""

    if not run_dirs:
        raise ValueError("at least one exact run directory is required")
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    for value in run_dirs:
        metrics, receipt = _load_run(Path(value))
        all_metrics.extend(metrics)
        manifests.append(receipt)
    manifest = pd.DataFrame(manifests).sort_values("seed").reset_index(drop=True)
    if manifest["seed"].duplicated().any():
        raise ValueError("published run list must contain one run per seed")
    raw = pd.DataFrame(all_metrics)
    aggregates = raw.loc[raw["stage"].eq("aggregate")].copy()
    if not aggregates.empty:
        aggregates = aggregates.sort_values(["seed", "condition"])
    comparisons = raw.loc[raw["stage"].eq("comparison")].copy()
    if not comparisons.empty:
        comparisons = comparisons.sort_values("seed")

    condition_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        rows = aggregates.loc[aggregates["condition"].eq(condition)]
        if rows.empty:
            condition_rows.append(
                {
                    "condition": condition,
                    "n_complete_seeds": 0,
                    "n_planned_seeds": len(manifest),
                    "conclusion": "inconclusive",
                }
            )
            continue
        mean, low, high = _bootstrap_mean(rows["exact_accuracy"].to_numpy(float))
        condition_rows.append(
            {
                "condition": condition,
                "n_complete_seeds": len(rows),
                "n_planned_seeds": len(manifest),
                "mean_exact_accuracy": mean,
                "seed_bootstrap_ci_low": low,
                "seed_bootstrap_ci_high": high,
                "mean_parameter_count": float(rows["parameter_count"].mean()),
                "mean_core_calls": float(rows["core_calls_per_forward"].mean()),
                "conclusion": "inconclusive",
            }
        )
    condition_summary = pd.DataFrame(condition_rows)

    comparison_rows: list[dict[str, object]] = []
    if not comparisons.empty:
        values = comparisons["estimate"].to_numpy(float)
        estimate, low, high = _bootstrap_mean(values)
        nonzero = int(np.count_nonzero(values))
        p_value = (
            1.0
            if nonzero == 0
            else float(
                wilcoxon(
                    values,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                ).pvalue
            )
        )
        matching = all(
            bool(comparisons[column].all())
            for column in (
                "parameter_count_matched",
                "initialization_matched",
                "optimizer_steps_matched",
                "core_calls_matched",
                "training_examples_and_order_matched",
                "validation_and_test_panels_matched",
            )
        )
        formal = bool(
            len(comparisons) >= 30
            and comparisons["formal_data_eligible"].all()
            and manifest["profile"].eq("formal").all()
            and matching
        )
        conclusion = "inconclusive"
        if formal and p_value < 0.05 and low > 0.0:
            conclusion = "support"
        elif formal and p_value < 0.05 and high < 0.0:
            conclusion = "oppose"
        comparison_rows.append(
            {
                "comparison": COMPARISON,
                "n_complete_seeds": len(comparisons),
                "n_planned_seeds": len(manifest),
                "estimate": estimate,
                "seed_bootstrap_ci_low": low,
                "seed_bootstrap_ci_high": high,
                "wilcoxon_p": p_value,
                "wilcoxon_p_holm": p_value,
                "n_nonzero_seeds": nonzero,
                "all_matching_gates_passed": matching,
                "formal_claim_eligible": formal,
                "conclusion": conclusion,
            }
        )
    comparison_summary = pd.DataFrame(comparison_rows)

    paths = {
        "raw": root / f"{prefix}_raw.csv.gz",
        "conditions": root / f"{prefix}_conditions.csv",
        "comparison": root / f"{prefix}_comparison.csv",
        "manifest": root / f"{prefix}_run_manifest.csv",
        "report": root / f"{prefix}_report.md",
    }
    raw.to_csv(paths["raw"], index=False, compression="gzip", lineterminator="\n")
    condition_summary.to_csv(paths["conditions"], index=False, lineterminator="\n")
    comparison_summary.to_csv(paths["comparison"], index=False, lineterminator="\n")
    manifest.to_csv(paths["manifest"], index=False, lineterminator="\n")
    comparison_text = (
        "No complete paired comparison was available."
        if comparison_summary.empty
        else (
            f"The seed-macro exact-accuracy difference was "
            f"{float(comparison_summary.iloc[0]['estimate']):.4f} "
            f"[{float(comparison_summary.iloc[0]['seed_bootstrap_ci_low']):.4f}, "
            f"{float(comparison_summary.iloc[0]['seed_bootstrap_ci_high']):.4f}]. "
            f"Conclusion: **{comparison_summary.iloc[0]['conclusion']}**."
        )
    )
    paths["report"].write_text(
        "\n".join(
            [
                "# Exp16 micro-TRM-like Sudoku audit",
                "",
                "This is an independently written, baseline-only small model. It uses "
                "BPTT and is not an official HRM/TRM reproduction or local-learning evidence.",
                "",
                f"Published seeds: {len(manifest)}. {comparison_text}",
                "",
                "The smoke profile is systems evidence only. A claim requires the frozen "
                "formal public-data profile, 30 independent training seeds, and every "
                "matching gate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="exp16_tiny_recursive_smoke")
    args = parser.parse_args(argv)
    outputs = publish_snapshot(
        args.run_dir,
        args.results_root,
        prefix=args.prefix,
    )
    from figures.exp16_tiny_recursive_plot import plot_exp16

    outputs.update(plot_exp16(Path(args.results_root), prefix=args.prefix))
    print(json.dumps({key: str(value) for key, value in outputs.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
