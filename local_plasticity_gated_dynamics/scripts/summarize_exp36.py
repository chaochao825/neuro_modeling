#!/usr/bin/env python3
"""Fail-closed collector-level summary for prospective Exp36 confirmation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config, seed_list
from experiments.exp36_change_aware_prefix import (
    CONDITIONS,
    EXPERIMENT,
    PANELS,
    PROTOCOL_VERSION,
    validate_preregistration,
)
from src.analysis.orbit_streaming_metrics import holm_adjust, paired_user_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_COMPARISONS = (
    {
        "comparison": "change_reset_over_cumulative_switch",
        "panel": "hidden_switch",
        "metric": "accuracy",
        "method": "jsd_change_reset",
        "comparator": "cumulative",
    },
    {
        "comparison": "change_reset_over_fixed_forgetting_switch",
        "panel": "hidden_switch",
        "metric": "accuracy",
        "method": "jsd_change_reset",
        "comparator": "fixed_forgetting",
    },
    {
        "comparison": "change_reset_over_fixed_forgetting_post_switch",
        "panel": "hidden_switch",
        "metric": "post_switch_accuracy",
        "method": "jsd_change_reset",
        "comparator": "fixed_forgetting",
    },
    {
        "comparison": "change_reset_over_cumulative_natural",
        "panel": "natural",
        "metric": "accuracy",
        "method": "jsd_change_reset",
        "comparator": "cumulative",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def eligible_run_dirs(
    results_root: Path, *, seeds: Iterable[int], profile: str
) -> list[Path]:
    selected: list[Path] = []
    for seed in seeds:
        root = results_root / "runs" / EXPERIMENT / f"seed_{int(seed):04d}"
        candidates: list[Path] = []
        if root.is_dir():
            for path in sorted(item for item in root.iterdir() if item.is_dir()):
                status_path = path / "status.json"
                manifest_path = path / "manifest.json"
                if not status_path.is_file() or not manifest_path.is_file():
                    continue
                status = _read_json(status_path)
                manifest = _read_json(manifest_path)
                if (
                    manifest.get("profile") == profile
                    and manifest.get("run_label") == profile
                    and status.get("status") in {"complete", "complete_with_failures"}
                ):
                    candidates.append(path)
        if len(candidates) != 1:
            raise RuntimeError(
                f"seed {seed} has {len(candidates)} eligible Exp36 {profile} runs; "
                "summarize an isolated results root"
            )
        selected.append(candidates[0])
    return selected


def validate_external_feature_cache(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config["external_feature_root"])).expanduser().resolve()
    manifest_path = root / "feature_manifest.csv"
    provenance_path = root / "provenance_external.json"
    failure_path = root / "failures_external.csv"
    for path in (manifest_path, provenance_path, failure_path):
        if not path.is_file():
            raise RuntimeError(f"external feature-cache artifact missing: {path}")
    provenance = _read_json(provenance_path)
    expected_md5 = str(config["external_archive_md5"]).lower()
    if provenance.get("source_archive_md5") != expected_md5:
        raise RuntimeError("external feature cache lacks the frozen archive MD5")
    if provenance.get("split") != "external" or int(
        provenance.get("n_failures", -1)
    ) != 0:
        raise RuntimeError("external feature provenance reports an invalid extraction")
    identity = str(provenance.get("encoder_identity", ""))
    if "efficientnet_b0" not in identity or "IMAGENET1K_V1" not in identity:
        raise RuntimeError("external feature cache uses the wrong frozen encoder")
    if failure_path.read_text(encoding="utf-8").strip():
        failures = pd.read_csv(failure_path)
        if len(failures):
            raise RuntimeError("external feature cache retains failed videos")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "split",
        "user_id",
        "video_id",
        "feature_path",
        "n_frames",
        "feature_dim",
    }
    if not required <= set(manifest.columns):
        raise RuntimeError("external feature manifest misses required columns")
    collectors = tuple(map(str, config["external_collectors"]))
    if set(manifest["user_id"].astype(str)) != set(collectors):
        raise RuntimeError("external feature manifest has incomplete collector coverage")
    if set(manifest["split"].astype(str)) != {"external"}:
        raise RuntimeError("external feature manifest contains another split")
    if manifest["video_id"].duplicated().any() or len(manifest) != int(
        provenance.get("n_planned_videos", -1)
    ):
        raise RuntimeError("external feature manifest has invalid video coverage")
    if (pd.to_numeric(manifest["n_frames"], errors="raise") < 1).any():
        raise RuntimeError("external feature manifest contains an empty video")
    missing_files = [
        value
        for value in manifest["feature_path"].astype(str)
        if not (root / value).is_file()
    ]
    if missing_files:
        raise RuntimeError(f"external cache misses {len(missing_files)} feature files")
    return {
        "root": str(root),
        "source_archive_md5": expected_md5,
        "encoder_identity": identity,
        "n_videos": int(len(manifest)),
        "n_collectors": int(manifest["user_id"].nunique()),
        "n_frames": int(pd.to_numeric(manifest["n_frames"]).sum()),
    }


def load_panel(
    run_dirs: Iterable[Path], *, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_frames: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for path in run_dirs:
        status = _read_json(path / "status.json")
        observed_config = _read_json(path / "config.json")
        summary = _read_json(path / "summary.json")
        if observed_config.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"{path} has a different Exp36 protocol")
        if observed_config.get("evidence_provenance") != config.get(
            "evidence_provenance"
        ):
            raise RuntimeError(f"{path} has different evidence provenance")
        for key in ("used_external_labels_for_fit", "used_future_frames"):
            if observed_config.get(key) is not False:
                raise RuntimeError(f"{path} violates the frozen causal-data gate")
        if summary.get("coverage", {}).get("complete") is not True:
            raise RuntimeError(f"{path} has incomplete registered coverage")
        if summary.get("claim_upgrade_allowed") is not True:
            raise RuntimeError(f"{path} is not a prospective confirmation run")
        if int(status.get("condition_failures", 0)) or int(
            status.get("condition_invalid", 0)
        ):
            raise RuntimeError(f"{path} retains failed or invalid conditions")
        seed = int(status["seed"])
        raw = pd.read_csv(path / "external_task_metrics.csv")
        raw["seed"] = seed
        raw["run_path"] = str(path.resolve())
        raw_frames.append(raw)
        selected = _read_json(path / "selected_hyperparameters.json")
        if selected.get("used_external_labels") is not False:
            raise RuntimeError(f"{path} selected parameters using external labels")
        selections.append(
            {
                "seed": seed,
                "fixed_retention": float(selected["fixed_retention"]),
                "window_frames": int(selected["window_frames"]),
                **dict(selected["detector"]),
                "run_path": str(path.resolve()),
            }
        )
        manifests.append(
            {
                "seed": seed,
                "run_path": str(path.resolve()),
                "run_status": status["status"],
                "condition_failures": int(status.get("condition_failures", 0)),
                "condition_invalid": int(status.get("condition_invalid", 0)),
            }
        )
    if not raw_frames:
        raise RuntimeError("no Exp36 runs were loaded")
    return (
        pd.concat(raw_frames, ignore_index=True),
        pd.DataFrame(selections),
        pd.DataFrame(manifests),
    )


def validate_raw_panel(raw: pd.DataFrame, *, config: Mapping[str, Any]) -> None:
    required = {
        "seed",
        "unit_id",
        "task_index",
        "panel",
        "condition",
        "accuracy",
        "post_switch_accuracy",
        "false_alarms_per_1000",
        "median_detection_delay",
        "n_frames",
        "n_switches",
        "n_alarms",
        "n_matched_switches",
        "source_video_ids",
        "status",
    }
    if not required <= set(raw.columns):
        raise RuntimeError(f"raw Exp36 panel misses {sorted(required - set(raw.columns))}")
    collectors = tuple(map(str, config["external_collectors"]))
    seeds = tuple(seed_list(config["seeds"]))
    n_tasks = int(config["n_external_tasks_per_collector"])
    expected = {
        (seed, collector, task, panel, condition)
        for seed in seeds
        for collector in collectors
        for task in range(n_tasks)
        for panel in PANELS
        for condition in CONDITIONS
    }
    observed = set(
        raw[["seed", "unit_id", "task_index", "panel", "condition"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected or len(raw) != len(expected):
        raise RuntimeError(
            f"raw Exp36 coverage mismatch: observed {len(observed)} of {len(expected)}"
        )
    if raw.duplicated(
        ["seed", "unit_id", "task_index", "panel", "condition"]
    ).any():
        raise RuntimeError("raw Exp36 panel contains duplicate registered cells")
    if not raw["status"].eq("complete").all():
        raise RuntimeError("raw Exp36 panel contains incomplete rows")
    accuracy = pd.to_numeric(raw["accuracy"], errors="raise").to_numpy(float)
    if not np.isfinite(accuracy).all() or np.any((accuracy < 0.0) | (accuracy > 1.0)):
        raise RuntimeError("raw Exp36 accuracy is invalid")
    hidden_post = pd.to_numeric(
        raw.loc[raw["panel"] == "hidden_switch", "post_switch_accuracy"],
        errors="raise",
    ).to_numpy(float)
    if not np.isfinite(hidden_post).all() or np.any(
        (hidden_post < 0.0) | (hidden_post > 1.0)
    ):
        raise RuntimeError("hidden-switch post-switch accuracy is invalid")
    for column in ("n_frames", "n_switches", "n_alarms", "n_matched_switches"):
        values = pd.to_numeric(raw[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise RuntimeError(f"raw Exp36 {column} is invalid")
    if (pd.to_numeric(raw["n_frames"], errors="raise") < 1).any():
        raise RuntimeError("raw Exp36 contains an empty stream")
    if raw["source_video_ids"].fillna("").astype(str).str.len().eq(0).any():
        raise RuntimeError("raw Exp36 loses source-video provenance")


def _finite_mean(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def reduce_to_collectors(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (collector, panel, condition), frame in raw.groupby(
        ["unit_id", "panel", "condition"], sort=True
    ):
        n_frames = int(pd.to_numeric(frame["n_frames"]).sum())
        n_switches = int(pd.to_numeric(frame["n_switches"]).sum())
        n_alarms = int(pd.to_numeric(frame["n_alarms"]).sum())
        n_matched = int(pd.to_numeric(frame["n_matched_switches"]).sum())
        delays = pd.to_numeric(
            frame["median_detection_delay"], errors="coerce"
        ).to_numpy(float)
        delays = delays[np.isfinite(delays)]
        rows.append(
            {
                "collector_id": str(collector),
                "panel": str(panel),
                "condition": str(condition),
                "accuracy": _finite_mean(frame["accuracy"]),
                "post_switch_accuracy": _finite_mean(frame["post_switch_accuracy"]),
                "detection_precision": (
                    float(n_matched / n_alarms) if n_alarms else float("nan")
                ),
                "detection_recall": (
                    float(n_matched / n_switches) if n_switches else float("nan")
                ),
                "false_alarms_per_1000": (
                    1000.0 * (n_alarms - n_matched) / n_frames
                ),
                "median_detection_delay": (
                    float(np.median(delays)) if delays.size else float("nan")
                ),
                "n_frames_nested": n_frames,
                "n_switches_nested": n_switches,
                "n_alarms_nested": n_alarms,
                "n_matched_switches_nested": n_matched,
                "n_seed_task_rows": int(len(frame)),
                "n_seeds": int(frame["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _paired_comparison(
    collectors: pd.DataFrame,
    *,
    specification: Mapping[str, str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    panel = collectors.loc[collectors["panel"] == specification["panel"]].copy()
    panel = panel[["collector_id", "condition", specification["metric"]]].rename(
        columns={
            "collector_id": "user_id",
            specification["metric"]: "user_video_mean_accuracy",
        }
    )
    if panel["user_video_mean_accuracy"].isna().any():
        raise RuntimeError(f"comparison {specification['comparison']} contains NaN")
    result = paired_user_inference(
        panel,
        method=specification["method"],
        comparator=specification["comparator"],
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "comparison": specification["comparison"],
        "panel": specification["panel"],
        "metric": specification["metric"],
        **asdict(result),
    }


def summarize_panel(
    raw: pd.DataFrame, *, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_raw_panel(raw, config=config)
    collectors = reduce_to_collectors(raw)
    expected_collector_rows = (
        len(config["external_collectors"]) * len(PANELS) * len(CONDITIONS)
    )
    if len(collectors) != expected_collector_rows:
        raise RuntimeError("collector reduction lost registered cells")
    analysis = config["analysis"]
    comparisons = [
        _paired_comparison(
            collectors,
            specification=specification,
            bootstrap_samples=int(analysis["bootstrap_samples"]),
            seed=int(analysis["statistics_seed"]) + index,
        )
        for index, specification in enumerate(PRIMARY_COMPARISONS)
    ]
    adjusted = holm_adjust(item["sign_flip_pvalue"] for item in comparisons)
    for index, value in enumerate(adjusted):
        comparisons[index]["holm_adjusted_pvalue"] = float(value)
    comparison_frame = pd.DataFrame(comparisons)
    by_name = comparison_frame.set_index("comparison").to_dict("index")
    means = {
        f"{panel}::{condition}": float(value)
        for (panel, condition), value in collectors.groupby(
            ["panel", "condition"], sort=True
        )["accuracy"].mean().items()
    }
    detector_hidden = collectors.loc[
        (collectors["panel"] == "hidden_switch")
        & (collectors["condition"] == "jsd_change_reset")
    ]
    detector_natural = collectors.loc[
        (collectors["panel"] == "natural")
        & (collectors["condition"] == "jsd_change_reset")
    ]
    collector_delays = detector_hidden["median_detection_delay"].to_numpy(float)
    collector_delays = np.where(np.isfinite(collector_delays), collector_delays, np.inf)
    cohort_delay = float(np.median(collector_delays))
    natural_false_alarms = float(detector_natural["false_alarms_per_1000"].mean())
    first = by_name["change_reset_over_cumulative_switch"]
    second = by_name["change_reset_over_fixed_forgetting_switch"]
    third = by_name["change_reset_over_fixed_forgetting_post_switch"]
    fourth = by_name["change_reset_over_cumulative_natural"]
    alpha = float(analysis["alpha"])
    causal_change = means["hidden_switch::jsd_change_reset"]
    score_only = means["hidden_switch::jsd_score_no_reset"]
    shifted = means["hidden_switch::matched_shifted_reset"]
    gates = {
        "complete_collector_coverage": int(collectors["collector_id"].nunique())
        == len(config["external_collectors"]),
        "hidden_gain_over_cumulative": (
            first["mean_difference"] >= float(analysis["hidden_switch_mcid"])
            and first["ci_low"] > 0.0
            and first["holm_adjusted_pvalue"] <= alpha
        ),
        "hidden_gain_over_fixed_forgetting": (
            second["mean_difference"] > 0.0
            and second["ci_low"] > 0.0
            and second["holm_adjusted_pvalue"] <= alpha
        ),
        "natural_noninferiority": (
            fourth["mean_difference"] >= -0.01
            and fourth["ci_low"]
            > -float(analysis["natural_noninferiority_margin"])
        ),
        "detection_delay": cohort_delay <= float(analysis["max_median_delay"]),
        "false_alarm_rate": natural_false_alarms
        <= float(analysis["max_false_alarms_per_1000"]),
        "timing_controls": causal_change > score_only and causal_change > shifted,
    }
    fixed_matches_or_beats = second["mean_difference"] <= 0.0
    significantly_negative = any(
        item["holm_adjusted_pvalue"] <= alpha and item["ci_high"] < 0.0
        for item in (first, second, third)
    ) or fourth["ci_high"] < -float(analysis["natural_noninferiority_margin"])
    conclusion = (
        "support"
        if all(gates.values())
        else "oppose"
        if significantly_negative
        else "inconclusive"
    )
    summary = {
        "experiment": EXPERIMENT,
        "protocol_version": PROTOCOL_VERSION,
        "profile": config["profile"],
        "evidence_provenance": config["evidence_provenance"],
        "claim_upgrade_allowed": True,
        "n_seeds": int(raw["seed"].nunique()),
        "n_collectors": int(collectors["collector_id"].nunique()),
        "n_tasks_per_collector_per_seed": int(config["n_external_tasks_per_collector"]),
        "statistical_unit": "collector",
        "nested_observations": "algorithmic seeds and tasks averaged within collector",
        "condition_collector_mean_accuracy": means,
        "comparisons": comparisons,
        "cohort_median_detection_delay": (
            cohort_delay if np.isfinite(cohort_delay) else None
        ),
        "natural_false_alarms_per_1000": natural_false_alarms,
        "timing_control_accuracies": {
            "jsd_change_reset": causal_change,
            "jsd_score_no_reset": score_only,
            "matched_shifted_reset": shifted,
        },
        "registered_diagnostics": {
            "post_switch_gain_over_fixed_forgetting": bool(
                third["mean_difference"] > 0.0
                and third["ci_low"] > 0.0
                and third["holm_adjusted_pvalue"] <= alpha
            ),
            "detector_recall_collector_mean": float(
                detector_hidden["detection_recall"].mean()
            ),
        },
        "support_gates": {key: bool(value) for key, value in gates.items()},
        "fixed_forgetting_matches_or_beats_detector": bool(fixed_matches_or_beats),
        "stop_rule_triggered": bool(fixed_matches_or_beats),
        "conclusion": conclusion,
    }
    return collectors, comparison_frame, summary


def _report(summary: Mapping[str, Any]) -> str:
    delay = summary["cohort_median_detection_delay"]
    delay_text = "not estimable" if delay is None else f"{float(delay):.3f} frames"
    lines = [
        "# Exp36 Change-Aware Prefix Confirmation",
        "",
        "Evidence status: prospectively frozen external confirmation.",
        "",
        f"Independent unit: collector (n={summary['n_collectors']}); seeds and tasks were averaged within collector.",
        "",
        "## Primary registered comparisons",
        "",
    ]
    for item in summary["comparisons"]:
        lines.append(
            f"- {item['comparison']}: {item['mean_difference']:+.4f} "
            f"(95% collector bootstrap {item['ci_low']:+.4f}, "
            f"{item['ci_high']:+.4f}; Holm p={item['holm_adjusted_pvalue']:.6g})."
        )
    lines.extend(
        [
            "",
            "## Operational diagnostics",
            "",
            f"- Cohort median detection delay: {delay_text}.",
            f"- Natural false alarms: {summary['natural_false_alarms_per_1000']:.3f} per 1,000 frames.",
            f"- Frozen stop rule triggered: {summary['stop_rule_triggered']}.",
            "",
            "## Preregistered verdict",
            "",
            f"**{str(summary['conclusion']).upper()}**",
            "",
        ]
    )
    for gate, passed in summary["support_gates"].items():
        lines.append(f"- {gate}: {'pass' if passed else 'fail'}")
    lines.extend(
        [
            "",
            "This verdict is bounded to frozen EfficientNet-B0 probability evidence on ORBIT-India; it is not a novelty, SOTA, or biological claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/prospective/exp36_change_aware_prefix.json"
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    validate_preregistration(config_path)
    config = load_json_config(config_path)
    results_root = Path(args.results_root).expanduser().resolve()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_root / "exp36_change_aware_prefix_confirmation"
    )
    output.mkdir(parents=True, exist_ok=False)
    feature_cache = validate_external_feature_cache(config)
    runs = eligible_run_dirs(
        results_root,
        seeds=seed_list(config["seeds"]),
        profile=str(config["profile"]),
    )
    raw, selections, manifest = load_panel(runs, config=config)
    collectors, comparisons, summary = summarize_panel(raw, config=config)
    summary["feature_cache"] = feature_cache
    summary["selection_stability"] = selections.to_dict("records")
    raw.to_csv(
        output / "raw_external_task_panel.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    collectors.to_csv(output / "collector_panel.csv", index=False)
    comparisons.to_csv(output / "primary_collector_comparisons.csv", index=False)
    selections.to_csv(output / "selection_stability.csv", index=False)
    manifest.to_csv(output / "run_manifest.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
