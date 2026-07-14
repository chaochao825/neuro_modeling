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
COMPARISON = "micro_trm_minus_single_state_core_call_matched"


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
    if (
        config.get("experiment") != EXPERIMENT
        or manifest.get("experiment") != EXPERIMENT
    ):
        raise ValueError(f"{run_dir} is not an {EXPERIMENT} run")
    planned = json.loads((run_dir / "planned_conditions.json").read_text("utf-8"))
    if {row["condition"] for row in planned} != set(CONDITIONS):
        raise ValueError("Exp16 publication requires the two frozen conditions")
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{run_dir} has no valid run_id")
    relative = "/".join(run_dir.parts[-4:])
    for row in metrics:
        if row.get("run_id") != run_id:
            raise ValueError(f"{run_dir} contains a metric from another run")
        row["published_run_path"] = relative
    if not metrics:
        # Preserve an attempt even when setup failed before the first metric
        # could be recorded. This row is descriptive provenance, not a score.
        metrics.append(
            {
                "run_id": run_id,
                "experiment": EXPERIMENT,
                "seed": int(config["seed"]),
                "stage": "run_status",
                "status": str(status.get("status")),
                "published_run_path": relative,
            }
        )
    environment = _read_json(run_dir / "environment.json")
    git = environment.get("git") if isinstance(environment.get("git"), dict) else {}
    fit_receipts_path = run_dir / "fit_receipts.json"
    receipt = {
        "run_id": run_id,
        "seed": int(config["seed"]),
        "profile": str(config.get("profile", "unspecified")),
        "run_status": str(status.get("status")),
        "started_at": str(status.get("started_at", manifest.get("started_at", ""))),
        "condition_failures": int(status.get("condition_failures", 0)),
        "condition_invalid": int(status.get("condition_invalid", 0)),
        "published_run_path": relative,
        "metrics_sha256": _sha256(run_dir / "metrics.jsonl"),
        "config_sha256": _sha256(run_dir / "config.json"),
        "registered_config_sha256": config.get("registered_config_sha256"),
        "fit_receipts_sha256": (
            _sha256(fit_receipts_path) if fit_receipts_path.is_file() else None
        ),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
    }
    return metrics, receipt


def _mark_latest_attempts(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark exactly the latest explicitly published attempt for every seed."""

    required = {"seed", "started_at", "published_run_path", "run_id"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Exp16 manifest is missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("Exp16 manifest must contain at least one attempt")
    ordered = manifest.sort_values(
        ["seed", "started_at", "published_run_path"], kind="stable"
    )
    latest_indices = set(ordered.groupby("seed", sort=False).tail(1).index)
    marked = manifest.copy()
    marked["selected_for_descriptive_summary"] = marked.index.to_series().map(
        lambda index: index in latest_indices
    )
    selected = marked.loc[marked["selected_for_descriptive_summary"]].copy()
    return marked, selected.sort_values("seed").reset_index(drop=True)


def latest_attempt_metrics(raw: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Return metrics from the same latest-per-seed attempts used in summaries."""

    expected, selected = _mark_latest_attempts(
        manifest.drop(columns=["selected_for_descriptive_summary"], errors="ignore")
    )
    if "selected_for_descriptive_summary" in manifest:
        actual = (
            manifest["selected_for_descriptive_summary"]
            .astype(str)
            .str.lower()
            .eq("true")
            .to_numpy()
        )
        wanted = expected["selected_for_descriptive_summary"].to_numpy(bool)
        if not np.array_equal(actual, wanted):
            raise ValueError("Exp16 manifest latest-attempt selection is invalid")
    if "run_id" not in raw:
        raise ValueError("Exp16 raw snapshot has no run_id column")
    selected_run_ids = set(selected["run_id"].astype(str))
    return raw.loc[raw["run_id"].astype(str).isin(selected_run_ids)].copy()


def publish_snapshot(
    run_dirs: Sequence[str | Path],
    results_root: str | Path,
    *,
    prefix: str = "exp16_tiny_recursive_smoke",
) -> dict[str, Path]:
    """Publish only the explicitly supplied immutable runs; never auto-select."""

    if not run_dirs:
        raise ValueError("at least one exact run directory is required")
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be one non-empty path component")
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
            "Exp16 publication is immutable; choose a new prefix: "
            + ", ".join(str(path) for path in existing)
        )
    all_metrics: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    for value in run_dirs:
        metrics, receipt = _load_run(Path(value))
        all_metrics.extend(metrics)
        manifests.append(receipt)
    manifest = pd.DataFrame(manifests).sort_values("seed").reset_index(drop=True)
    raw = pd.DataFrame(all_metrics)
    # All attempts remain in raw/manifest. Inference uses the latest explicitly
    # supplied attempt per seed, so a latest failure cannot fall back to an
    # earlier success.
    manifest, selected_manifest = _mark_latest_attempts(manifest)
    selected_raw = latest_attempt_metrics(raw, manifest)
    aggregates = selected_raw.loc[selected_raw["stage"].eq("aggregate")].copy()
    if not aggregates.empty:
        aggregates = aggregates.sort_values(["seed", "condition"])
    comparisons = selected_raw.loc[selected_raw["stage"].eq("comparison")].copy()
    if not comparisons.empty:
        comparisons = comparisons.sort_values("seed")

    condition_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        rows = (
            aggregates
            if aggregates.empty
            else aggregates.loc[aggregates["condition"].eq(condition)]
        )
        if rows.empty:
            condition_rows.append(
                {
                    "condition": condition,
                    "n_complete_seeds": 0,
                    "n_planned_seeds": len(selected_manifest),
                    "n_published_attempts": len(manifest),
                    "mean_exact_accuracy": np.nan,
                    "seed_bootstrap_ci_low": np.nan,
                    "seed_bootstrap_ci_high": np.nan,
                    "mean_parameter_count": np.nan,
                    "mean_nominal_core_calls": np.nan,
                    "conclusion": "inconclusive",
                }
            )
            continue
        mean, low, high = _bootstrap_mean(rows["exact_accuracy"].to_numpy(float))
        condition_rows.append(
            {
                "condition": condition,
                "n_complete_seeds": len(rows),
                "n_planned_seeds": len(selected_manifest),
                "n_published_attempts": len(manifest),
                "mean_exact_accuracy": mean,
                "seed_bootstrap_ci_low": low,
                "seed_bootstrap_ci_high": high,
                "mean_parameter_count": float(rows["parameter_count"].mean()),
                "mean_nominal_core_calls": float(
                    rows["nominal_core_calls_per_evaluation"].mean()
                ),
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
                "nominal_core_calls_matched",
                "training_data_matched",
                "validation_data_matched",
                "epoch_permutations_matched",
                "test_panel_fingerprints_matched",
            )
        )
        # Formal promotion is intentionally unavailable in this first additive
        # implementation. A future publisher must bind a canonical run
        # inventory and recompute task/source-group/seed metrics from raw rows.
        formal = False
        conclusion = "inconclusive"
        comparison_rows.append(
            {
                "comparison": COMPARISON,
                "n_complete_seeds": len(comparisons),
                "n_planned_seeds": len(selected_manifest),
                "n_published_attempts": len(manifest),
                "estimate": estimate,
                "seed_bootstrap_ci_low": low,
                "seed_bootstrap_ci_high": high,
                "wilcoxon_p": p_value,
                "wilcoxon_p_holm": p_value,
                "n_nonzero_seeds": nonzero,
                "all_matching_gates_passed": matching,
                "formal_claim_eligible": formal,
                "formal_ineligibility_reason": (
                    "pilot_only_publisher_raw_recompute_and_canonical_inventory_pending"
                ),
                "conclusion": conclusion,
            }
        )
    comparison_columns = [
        "comparison",
        "n_complete_seeds",
        "n_planned_seeds",
        "n_published_attempts",
        "estimate",
        "seed_bootstrap_ci_low",
        "seed_bootstrap_ci_high",
        "wilcoxon_p",
        "wilcoxon_p_holm",
        "n_nonzero_seeds",
        "all_matching_gates_passed",
        "formal_claim_eligible",
        "formal_ineligibility_reason",
        "conclusion",
    ]
    comparison_summary = pd.DataFrame(comparison_rows, columns=comparison_columns)

    raw.to_csv(
        paths["raw"],
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        lineterminator="\n",
    )
    manifest.to_csv(paths["manifest"], index=False, lineterminator="\n")
    raw_sha256 = _sha256(paths["raw"])
    manifest_sha256 = _sha256(paths["manifest"])
    config_hashes = ";".join(
        sorted(set(manifest["registered_config_sha256"].dropna().astype(str)))
    )
    for frame in (condition_summary, comparison_summary):
        frame["scoped_raw_sha256"] = raw_sha256
        frame["run_manifest_sha256"] = manifest_sha256
        frame["registered_config_sha256_values"] = config_hashes
        frame["publisher_scope"] = "pilot_only_formal_promotion_disabled"
    condition_summary.to_csv(paths["conditions"], index=False, lineterminator="\n")
    comparison_summary.to_csv(paths["comparison"], index=False, lineterminator="\n")
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
                f"Published attempts: {len(manifest)} across "
                f"{len(selected_manifest)} seeds. {comparison_text}",
                "",
                "This publisher is intentionally pilot-only: formal promotion is disabled "
                "until a canonical all-attempt inventory and raw task-level recomputation "
                "are implemented. The public Sudoku V2 test panel is non-OOD.",
                "",
                f"Raw SHA-256: `{raw_sha256}`; run-manifest SHA-256: "
                f"`{manifest_sha256}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def load_published_snapshot(
    results_root: str | Path,
    *,
    prefix: str = "exp16_tiny_recursive_smoke",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a scoped snapshot only after checking its raw/manifest bindings."""

    root = Path(results_root)
    raw_path = root / f"{prefix}_raw.csv.gz"
    conditions = pd.read_csv(root / f"{prefix}_conditions.csv")
    comparison = pd.read_csv(root / f"{prefix}_comparison.csv")
    manifest_path = root / f"{prefix}_run_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    raw = pd.read_csv(raw_path)
    latest_attempt_metrics(raw, manifest)
    raw_hash = _sha256(raw_path)
    manifest_hash = _sha256(manifest_path)
    for name, frame in (("conditions", conditions), ("comparison", comparison)):
        if frame.empty and name == "comparison":
            continue
        if set(frame["scoped_raw_sha256"].astype(str)) != {raw_hash}:
            raise ValueError(f"Exp16 {name} raw binding is invalid")
        if set(frame["run_manifest_sha256"].astype(str)) != {manifest_hash}:
            raise ValueError(f"Exp16 {name} manifest binding is invalid")
        if set(frame["publisher_scope"].astype(str)) != {
            "pilot_only_formal_promotion_disabled"
        }:
            raise ValueError(f"Exp16 {name} publisher scope is invalid")
    return raw, conditions, comparison, manifest


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
    print(
        json.dumps({key: str(value) for key, value in outputs.items()}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
