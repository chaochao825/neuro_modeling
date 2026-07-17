"""Summarize an explicitly labeled Exp30 panel at the seed level."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config, seed_list
from experiments.exp30_associative_actuator_trend import (
    EVIDENCE_SCHEMA_VERSION,
    EXPERIMENT,
    MODES,
    PROTOCOL_VERSION,
    _mu_values,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _eligible_run_dirs(
    results_root: Path,
    *,
    seeds: list[int],
    profile: str,
    run_label: str,
) -> list[Path]:
    selected: list[Path] = []
    for seed in seeds:
        seed_root = results_root / "runs" / EXPERIMENT / f"seed_{seed:04d}"
        candidates: list[Path] = []
        if seed_root.exists():
            for path in sorted(item for item in seed_root.iterdir() if item.is_dir()):
                status_path = path / "status.json"
                manifest_path = path / "manifest.json"
                if not status_path.exists() or not manifest_path.exists():
                    continue
                status = _read_json(status_path)
                manifest = _read_json(manifest_path)
                if (
                    manifest.get("profile") == profile
                    and manifest.get("run_label") == run_label
                    and status.get("status") in {"complete", "complete_with_failures"}
                ):
                    candidates.append(path)
        if len(candidates) != 1:
            raise RuntimeError(
                f"seed {seed} has {len(candidates)} eligible runs for label "
                f"{run_label!r}; expected exactly one"
            )
        selected.append(candidates[0])
    return selected


def load_panel(
    results_root: Path,
    config: dict[str, Any],
    *,
    run_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = seed_list(config["seeds"])
    run_dirs = _eligible_run_dirs(
        results_root,
        seeds=seeds,
        profile=str(config["profile"]),
        run_label=run_label,
    )
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    panel_provenance: dict[str, Any] | None = None
    expected_rows = len(_mu_values(config)) * len(MODES)
    for path in run_dirs:
        run_config = _read_json(path / "config.json")
        observed_static = {
            key: value
            for key, value in run_config.items()
            if key
            not in {"experiment", "seed", "run_label", "evidence_provenance"}
        }
        expected_static = {
            key: value for key, value in config.items() if key != "config_path"
        }
        if observed_static != expected_static:
            raise RuntimeError(f"{path} does not match the selected static config")
        provenance = run_config.get("evidence_provenance")
        if not isinstance(provenance, dict):
            raise RuntimeError(f"{path} has no evidence provenance binding")
        if panel_provenance is None:
            panel_provenance = provenance
        elif provenance != panel_provenance:
            raise RuntimeError("Exp30 panel mixes source or Git provenance")
        rows = pd.read_json(path / "metrics.jsonl", lines=True)
        if len(rows) != expected_rows:
            raise RuntimeError(f"{path} has {len(rows)} rows; expected {expected_rows}")
        frames.append(rows)
        status = _read_json(path / "status.json")
        manifests.append(
            {
                "seed": int(status["seed"]),
                "run_path": str(path.resolve()),
                "run_status": str(status["status"]),
                "condition_failures": int(status["condition_failures"]),
                "condition_invalid": int(status["condition_invalid"]),
            }
        )
    raw = pd.concat(frames, ignore_index=True)
    if set(raw["seed"].astype(int)) != set(seeds):
        raise RuntimeError("raw panel seed set differs from the registered config")
    validate_panel_contract(raw, config)
    return raw, pd.DataFrame(manifests).sort_values("seed").reset_index(drop=True)


def validate_panel_contract(raw: pd.DataFrame, config: dict[str, Any]) -> None:
    """Fail closed on heterogeneous, incomplete, dirty, or unpaired panels."""

    required = {
        "seed",
        "status",
        "memory_demand",
        "actuator_mode",
        "protocol_version",
        "evidence_schema_version",
        "run_git_commit",
        "run_git_tree",
        "run_git_dirty",
        "train_split_fingerprint",
        "test_split_fingerprint",
        "carrier_fingerprint",
        "functional_budget_valid",
        "associative_shuffled_write_budget_equal",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"missing Exp30 contract columns: {sorted(missing)}")
    seeds = seed_list(config["seeds"])
    expected_keys = {
        (seed, mu, mode)
        for seed in seeds
        for mu in _mu_values(config)
        for mode in MODES
    }
    observed_keys = list(
        zip(
            raw["seed"].astype(int),
            raw["memory_demand"].astype(float),
            raw["actuator_mode"].astype(str),
            strict=True,
        )
    )
    if len(observed_keys) != len(set(observed_keys)):
        raise RuntimeError("Exp30 panel contains duplicate seed/demand/mode cells")
    if set(observed_keys) != expected_keys:
        raise RuntimeError("Exp30 panel is missing or adds seed/demand/mode cells")
    if set(raw["protocol_version"].astype(str)) != {PROTOCOL_VERSION}:
        raise RuntimeError("Exp30 panel mixes protocol versions")
    if set(raw["evidence_schema_version"].astype(str)) != {
        EVIDENCE_SCHEMA_VERSION
    }:
        raise RuntimeError("Exp30 panel mixes evidence schemas")
    if not raw["run_git_dirty"].eq(False).all():
        raise RuntimeError("Exp30 publishable panel must come from clean Git runs")
    for column in ("run_git_commit", "run_git_tree"):
        values = raw[column].astype(str)
        if values.nunique() != 1 or len(values.iloc[0]) != 40:
            raise RuntimeError(f"Exp30 panel mixes or lacks one bound {column}")
        try:
            int(values.iloc[0], 16)
        except ValueError as error:
            raise RuntimeError(f"Exp30 {column} is not hexadecimal") from error
    if not raw["functional_budget_valid"].eq(True).all():
        raise RuntimeError("Exp30 panel contains an invalid query-output RMS budget")
    if not raw["associative_shuffled_write_budget_equal"].eq(True).all():
        raise RuntimeError("Exp30 associative/shuffled write budgets differ")
    for column in (
        "train_split_fingerprint",
        "test_split_fingerprint",
        "carrier_fingerprint",
    ):
        if raw.groupby("seed")[column].nunique().max() != 1:
            raise RuntimeError(f"Exp30 {column} is not paired within seed")


def summarize_records(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {
        "seed",
        "status",
        "memory_demand",
        "actuator_mode",
        "test_normalized_score",
        "test_sign_accuracy",
        "paired_matched_minus_fixed_score",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"missing Exp30 raw columns: {sorted(missing)}")
    if set(raw["status"]) != {"complete"}:
        raise RuntimeError("Exp30 summary retains failures but cannot promote a trend")
    conditions = (
        raw.groupby(["memory_demand", "actuator_mode"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_score=("test_normalized_score", "mean"),
            sd_score=("test_normalized_score", "std"),
            mean_sign_accuracy=("test_sign_accuracy", "mean"),
        )
        .sort_values(["memory_demand", "actuator_mode"])
        .reset_index(drop=True)
    )
    score_pivot = raw.pivot(
        index=["seed", "memory_demand"],
        columns="actuator_mode",
        values="test_normalized_score",
    ).reset_index()
    per_seed_rows: list[dict[str, Any]] = []
    for seed, frame in score_pivot.groupby("seed", sort=True):
        ordered = frame.sort_values("memory_demand")
        advantage = ordered["associative"] - ordered["routing"]
        crossover = float(
            pd.Series(ordered["memory_demand"]).corr(
                pd.Series(advantage), method="spearman"
            )
        )
        low = ordered.iloc[0]
        high = ordered.iloc[-1]
        per_seed_rows.append(
            {
                "seed": int(seed),
                "routing_macro_score": float(ordered["routing"].mean()),
                "low_rank_macro_score": float(ordered["low_rank"].mean()),
                "associative_macro_score": float(ordered["associative"].mean()),
                "fixed_best_macro_score": float(ordered["fixed_best"].mean()),
                "matched_macro_score": float(ordered["matched"].mean()),
                "combined_macro_score": float(ordered["combined"].mean()),
                "matched_minus_fixed": float(
                    ordered["matched"].mean() - ordered["fixed_best"].mean()
                ),
                "demand_advantage_spearman": crossover,
                "routing_low_endpoint_win": bool(low["routing"] > low["associative"]),
                "associative_high_endpoint_win": bool(
                    high["associative"] > high["routing"]
                ),
                "associative_over_shuffled_high": float(
                    high["associative"] - high["associative_shuffled"]
                ),
            }
        )
    seeds = pd.DataFrame(per_seed_rows)
    positive = (
        (seeds["demand_advantage_spearman"] > 0.8)
        & (seeds["matched_minus_fixed"] > 0.0)
        & seeds["routing_low_endpoint_win"]
        & seeds["associative_high_endpoint_win"]
        & (seeds["associative_over_shuffled_high"] > 0.0)
    )
    positive_fraction = float(np.mean(positive))
    profile_values = set(raw["profile"].astype(str)) if "profile" in raw else set()
    profile = next(iter(profile_values)) if len(profile_values) == 1 else "unknown"
    trend = "trend-positive" if positive_fraction >= 0.8 else "trend-not-established"
    summary = {
        "experiment": EXPERIMENT,
        "profile": profile,
        "n_seeds": int(seeds["seed"].nunique()),
        "trend_classification": trend,
        "claim_classification": "inconclusive" if profile != "formal" else "pending-formal-statistics",
        "positive_seed_fraction": positive_fraction,
        "mean_matched_minus_fixed": float(seeds["matched_minus_fixed"].mean()),
        "mean_demand_advantage_spearman": float(
            seeds["demand_advantage_spearman"].mean()
        ),
        "mean_associative_over_shuffled_high": float(
            seeds["associative_over_shuffled_high"].mean()
        ),
        "strong_baseline_claimed": False,
        "learned_memory_selector_claimed": False,
    }
    return conditions, seeds, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_run_receipts(manifest: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Copy the small immutable per-seed receipt set into the scoped bundle."""

    required_files = (
        "config.json",
        "environment.json",
        "status.json",
        "manifest.json",
        "planned_conditions.json",
        "metrics.jsonl",
        "run.log",
    )
    rows: list[dict[str, object]] = []
    receipt_root = output / "run_receipts"
    receipt_root.mkdir(parents=True, exist_ok=False)
    for record in manifest.itertuples(index=False):
        source = Path(str(record.run_path))
        destination = receipt_root / f"seed_{int(record.seed):04d}"
        destination.mkdir(parents=True, exist_ok=False)
        for name in required_files:
            source_file = source / name
            if not source_file.exists():
                raise FileNotFoundError(f"missing Exp30 receipt: {source_file}")
            destination_file = destination / name
            shutil.copy2(source_file, destination_file)
            rows.append(
                {
                    "seed": int(record.seed),
                    "relative_path": str(destination_file.relative_to(output)),
                    "size_bytes": destination_file.stat().st_size,
                    "sha256": _sha256(destination_file),
                }
            )
    return pd.DataFrame(rows).sort_values(["seed", "relative_path"]).reset_index(
        drop=True
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Exp30 associative-actuator trend",
            "",
            f"- Profile: `{summary['profile']}` ({summary['n_seeds']} seed replicates).",
            f"- Exploratory trend: **{summary['trend_classification']}**.",
            f"- Formal claim classification: **{summary['claim_classification']}**.",
            f"- Positive seed fraction: {summary['positive_seed_fraction']:.3f}.",
            f"- Mean matched-minus-fixed score: {summary['mean_matched_minus_fixed']:.4f}.",
            f"- Mean demand/advantage Spearman: {summary['mean_demand_advantage_spearman']:.4f}.",
            f"- Mean high-demand associative-minus-shuffled score: {summary['mean_associative_over_shuffled_high']:.4f}.",
            "",
            "This constructed positive-control panel tests a crossover and specificity",
            "sanity trend under a fixed identity-calibrated carrier bridge and fixed",
            "actuator dictionary. It is not a strong-baseline",
            "comparison, an observation-only learned selector, real-data evidence, or",
            "formal support for the full Actuator Matching Principle.",
            "",
        ]
    )


def _bundle_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Exp30 associative-actuator trend: development evidence",
            "",
            "This scoped bundle contains the complete five-seed development panel",
            "for `exp30_trend_v1` (seeds 9100--9104). All 160 planned condition",
            "rows completed; no failed or invalid row was removed. `run_receipts/`",
            "preserves each seed's config, environment, planned conditions, raw",
            "JSONL metrics, status, manifest, and log. `receipt_manifest.csv` binds",
            "those files by SHA-256.",
            "",
            f"The exploratory gate is **{summary['trend_classification']}**:",
            "",
            f"- positive seed fraction: {summary['positive_seed_fraction']:.3f};",
            f"- mean demand/advantage Spearman: {summary['mean_demand_advantage_spearman']:.4f};",
            f"- mean matched-minus-fixed score: {summary['mean_matched_minus_fixed']:+.4f};",
            f"- mean high-demand associative-minus-shuffled score: {summary['mean_associative_over_shuffled_high']:+.4f}.",
            "",
            f"The scientific classification remains **{summary['claim_classification']}**.",
            "This positive-control sanity result licenses scale; it is not a",
            "carrier-dynamics result or a strong-baseline",
            "claim, an observation-only learned selector, real-data evidence, or",
            "formal support for the complete Actuator Matching Principle.",
            "",
            "Files: `raw_metrics.csv.gz`, `conditions.csv`, `seed_summary.csv`,",
            "`summary.csv`, `summary.json`, `report.md`, `run_manifest.csv`,",
            "`receipt_manifest.csv`, `run_receipts/`, and the data-bound PNG/PDF.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/smoke/exp30_associative_actuator_trend.json",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-label", required=True)
    parser.add_argument(
        "--output-dir", default="results/exp30_associative_actuator_trend_smoke"
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    raw, manifest = load_panel(
        Path(args.results_root), config, run_label=args.run_label
    )
    conditions, seeds, summary = summarize_records(raw)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    raw.to_csv(output / "raw_metrics.csv.gz", index=False, compression="gzip")
    manifest.to_csv(output / "run_manifest.csv", index=False)
    receipts = package_run_receipts(manifest, output)
    receipts.to_csv(output / "receipt_manifest.csv", index=False)
    conditions.to_csv(output / "conditions.csv", index=False)
    seeds.to_csv(output / "seed_summary.csv", index=False)
    pd.DataFrame([summary]).to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    (output / "README.md").write_text(
        _bundle_readme(summary), encoding="utf-8"
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
