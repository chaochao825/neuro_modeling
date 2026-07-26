#!/usr/bin/env python3
"""Fail-closed session-level summary for prospective Exp37."""

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
from experiments.exp37_core50_change_aware_prefix import (
    CONDITIONS,
    EXPERIMENT,
    PANELS,
    PROTOCOL_VERSION,
    validate_feature_cache,
    validate_preregistration,
)
from src.analysis.orbit_streaming_metrics import holm_adjust, paired_user_inference


PRIMARY_COMPARISONS = (
    {
        "comparison": "change_reset_over_cumulative_switch",
        "panel": "hidden_switch",
        "metric": "accuracy",
        "method": "bocpd_change_reset",
        "comparator": "cumulative",
    },
    {
        "comparison": "change_reset_over_fixed_forgetting_switch",
        "panel": "hidden_switch",
        "metric": "accuracy",
        "method": "bocpd_change_reset",
        "comparator": "fixed_forgetting",
    },
    {
        "comparison": "change_reset_over_sliding_window_switch",
        "panel": "hidden_switch",
        "metric": "accuracy",
        "method": "bocpd_change_reset",
        "comparator": "sliding_window",
    },
    {
        "comparison": "change_reset_over_cumulative_natural",
        "panel": "natural",
        "metric": "accuracy",
        "method": "bocpd_change_reset",
        "comparator": "cumulative",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def eligible_run_dirs(results_root: Path, *, seeds: Iterable[int]) -> list[Path]:
    selected: list[Path] = []
    for seed in seeds:
        root = results_root / "runs" / EXPERIMENT / f"seed_{int(seed):04d}"
        candidates: list[Path] = []
        if root.is_dir():
            for path in sorted(item for item in root.iterdir() if item.is_dir()):
                if not (path / "status.json").is_file() or not (path / "manifest.json").is_file():
                    continue
                status = _read_json(path / "status.json")
                manifest = _read_json(path / "manifest.json")
                if (
                    manifest.get("profile") == "prospective_external"
                    and manifest.get("run_label") == "prospective_external"
                    and status.get("status") in {"complete", "complete_with_failures"}
                ):
                    candidates.append(path)
        if len(candidates) != 1:
            raise RuntimeError(
                f"seed {seed} has {len(candidates)} eligible Exp37 runs; use an isolated root"
            )
        selected.append(candidates[0])
    return selected


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
            raise RuntimeError(f"{path} uses another Exp37 protocol")
        for key in ("used_external_labels_for_fit", "used_future_frames"):
            if observed_config.get(key) is not False:
                raise RuntimeError(f"{path} violates the causal data gate")
        if summary.get("coverage", {}).get("complete") is not True:
            raise RuntimeError(f"{path} has incomplete registered coverage")
        if summary.get("claim_upgrade_allowed") is not True:
            raise RuntimeError(f"{path} cannot upgrade the claim")
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
            raise RuntimeError(f"{path} selected parameters on external labels")
        selections.append(
            {
                "seed": seed,
                "temperature": float(selected["temperature"]),
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
                "run_status": str(status["status"]),
                "condition_failures": int(status.get("condition_failures", 0)),
                "condition_invalid": int(status.get("condition_invalid", 0)),
            }
        )
    if not raw_frames:
        raise RuntimeError("no Exp37 runs loaded")
    return (
        pd.concat(raw_frames, ignore_index=True),
        pd.DataFrame(selections),
        pd.DataFrame(manifests),
    )


def validate_raw_panel(raw: pd.DataFrame, *, config: Mapping[str, Any]) -> None:
    required = {
        "seed",
        "session_id",
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
        "object_ids",
        "source_cells",
        "status",
    }
    if not required <= set(raw.columns):
        raise RuntimeError(f"raw Exp37 panel misses {sorted(required-set(raw.columns))}")
    seeds = tuple(seed_list(config["seeds"]))
    sessions = tuple(map(str, config["external_sessions"]))
    n_tasks = int(config["n_external_tasks_per_session"])
    expected = {
        (seed, session, task, panel, condition)
        for seed in seeds
        for session in sessions
        for task in range(n_tasks)
        for panel in PANELS
        for condition in CONDITIONS
    }
    observed = set(
        raw[["seed", "session_id", "task_index", "panel", "condition"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected or len(raw) != len(expected):
        raise RuntimeError(
            f"raw Exp37 coverage mismatch: observed {len(observed)} of {len(expected)}"
        )
    if raw.duplicated(
        ["seed", "session_id", "task_index", "panel", "condition"]
    ).any():
        raise RuntimeError("raw Exp37 panel contains duplicate cells")
    if not raw["status"].eq("complete").all():
        raise RuntimeError("raw Exp37 panel contains incomplete rows")
    accuracy = pd.to_numeric(raw["accuracy"], errors="raise").to_numpy(float)
    if not np.isfinite(accuracy).all() or np.any((accuracy < 0.0) | (accuracy > 1.0)):
        raise RuntimeError("raw Exp37 accuracy is invalid")
    hidden_post = pd.to_numeric(
        raw.loc[raw["panel"] == "hidden_switch", "post_switch_accuracy"],
        errors="raise",
    ).to_numpy(float)
    if not np.isfinite(hidden_post).all() or np.any(
        (hidden_post < 0.0) | (hidden_post > 1.0)
    ):
        raise RuntimeError("raw Exp37 hidden post-switch accuracy is invalid")
    for column in ("n_frames", "n_switches", "n_alarms", "n_matched_switches"):
        values = pd.to_numeric(raw[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise RuntimeError(f"raw Exp37 {column} is invalid")
    if (pd.to_numeric(raw["n_frames"], errors="raise") < 1).any():
        raise RuntimeError("raw Exp37 contains an empty stream")
    for column in ("object_ids", "source_cells"):
        if raw[column].fillna("").astype(str).str.len().eq(0).any():
            raise RuntimeError(f"raw Exp37 loses {column} provenance")


def _finite_mean(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def reduce_to_sessions(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (session, panel, condition), frame in raw.groupby(
        ["session_id", "panel", "condition"], sort=True
    ):
        n_frames = int(pd.to_numeric(frame["n_frames"]).sum())
        n_switches = int(pd.to_numeric(frame["n_switches"]).sum())
        n_alarms = int(pd.to_numeric(frame["n_alarms"]).sum())
        n_matched = int(pd.to_numeric(frame["n_matched_switches"]).sum())
        delays = pd.to_numeric(frame["median_detection_delay"], errors="coerce").to_numpy(float)
        delays = delays[np.isfinite(delays)]
        rows.append(
            {
                "session_id": str(session),
                "panel": str(panel),
                "condition": str(condition),
                "accuracy": _finite_mean(frame["accuracy"]),
                "post_switch_accuracy": _finite_mean(frame["post_switch_accuracy"]),
                "detection_precision": float(n_matched / n_alarms) if n_alarms else float("nan"),
                "detection_recall": float(n_matched / n_switches) if n_switches else float("nan"),
                "false_alarms_per_1000": 1000.0 * (n_alarms - n_matched) / n_frames,
                "median_detection_delay": float(np.median(delays)) if delays.size else float("nan"),
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
    sessions: pd.DataFrame,
    *,
    specification: Mapping[str, str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    panel = sessions.loc[sessions["panel"] == specification["panel"]].copy()
    panel = panel[["session_id", "condition", specification["metric"]]].rename(
        columns={
            "session_id": "user_id",
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
    return {"comparison": specification["comparison"], **specification, **asdict(result)}


def summarize_panel(
    raw: pd.DataFrame, *, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_raw_panel(raw, config=config)
    sessions = reduce_to_sessions(raw)
    expected_rows = len(config["external_sessions"]) * len(PANELS) * len(CONDITIONS)
    if len(sessions) != expected_rows:
        raise RuntimeError("session reduction lost registered cells")
    analysis = config["analysis"]
    comparisons = [
        _paired_comparison(
            sessions,
            specification=specification,
            bootstrap_samples=int(analysis["bootstrap_samples"]),
            seed=int(analysis["statistics_seed"]) + index,
        )
        for index, specification in enumerate(PRIMARY_COMPARISONS)
    ]
    adjusted = holm_adjust(item["sign_flip_pvalue"] for item in comparisons)
    for item, value in zip(comparisons, adjusted, strict=True):
        item["holm_adjusted_pvalue"] = float(value)
    comparison_frame = pd.DataFrame(comparisons)
    by_name = comparison_frame.set_index("comparison").to_dict("index")
    means = {
        f"{panel}::{condition}": float(value)
        for (panel, condition), value in sessions.groupby(
            ["panel", "condition"], sort=True
        )["accuracy"].mean().items()
    }
    detector_hidden = sessions.loc[
        (sessions["panel"] == "hidden_switch")
        & (sessions["condition"] == "bocpd_change_reset")
    ]
    detector_natural = sessions.loc[
        (sessions["panel"] == "natural")
        & (sessions["condition"] == "bocpd_change_reset")
    ]
    delays = detector_hidden["median_detection_delay"].to_numpy(float)
    delays = np.where(np.isfinite(delays), delays, np.inf)
    cohort_delay = float(np.median(delays))
    natural_false_alarms = float(detector_natural["false_alarms_per_1000"].mean())
    cumulative = by_name["change_reset_over_cumulative_switch"]
    fixed = by_name["change_reset_over_fixed_forgetting_switch"]
    window = by_name["change_reset_over_sliding_window_switch"]
    natural = by_name["change_reset_over_cumulative_natural"]
    alpha = float(analysis["alpha"])
    hard_accuracy = means["hidden_switch::bocpd_change_reset"]
    score_accuracy = means["hidden_switch::bocpd_score_no_reset"]
    shifted_accuracy = means["hidden_switch::matched_shifted_reset"]
    gates = {
        "complete_session_coverage": int(sessions["session_id"].nunique())
        == len(config["external_sessions"]),
        "hidden_gain_over_cumulative": (
            cumulative["mean_difference"] >= float(analysis["hidden_switch_mcid"])
            and cumulative["ci_low"] > 0.0
            and cumulative["holm_adjusted_pvalue"] <= alpha
        ),
        "hidden_gain_over_fixed_forgetting": (
            fixed["mean_difference"] > 0.0
            and fixed["ci_low"] > 0.0
            and fixed["holm_adjusted_pvalue"] <= alpha
        ),
        "hidden_gain_over_sliding_window": (
            window["mean_difference"] > 0.0
            and window["ci_low"] > 0.0
            and window["holm_adjusted_pvalue"] <= alpha
        ),
        "natural_noninferiority": (
            natural["mean_difference"]
            >= -float(analysis["max_natural_point_loss"])
            and natural["ci_low"] > -float(analysis["natural_noninferiority_margin"])
        ),
        "detection_delay": cohort_delay <= float(analysis["max_median_delay"]),
        "false_alarm_rate": natural_false_alarms
        <= float(analysis["max_false_alarms_per_1000"]),
        "timing_controls": hard_accuracy > score_accuracy
        and hard_accuracy > shifted_accuracy,
    }
    stationary_matches = fixed["mean_difference"] <= 0.0 or window["mean_difference"] <= 0.0
    significantly_negative = any(
        item["holm_adjusted_pvalue"] <= alpha and item["ci_high"] < 0.0
        for item in (cumulative, fixed, window)
    ) or natural["ci_high"] < -float(analysis["natural_noninferiority_margin"])
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
        "n_sessions": int(sessions["session_id"].nunique()),
        "n_tasks_per_session_per_seed": int(config["n_external_tasks_per_session"]),
        "statistical_unit": "session",
        "nested_observations": "algorithmic seeds tasks objects and frames averaged within session",
        "condition_session_mean_accuracy": means,
        "comparisons": comparisons,
        "cohort_median_detection_delay": cohort_delay if np.isfinite(cohort_delay) else None,
        "natural_false_alarms_per_1000": natural_false_alarms,
        "timing_control_accuracies": {
            "bocpd_change_reset": hard_accuracy,
            "bocpd_score_no_reset": score_accuracy,
            "matched_shifted_reset": shifted_accuracy,
        },
        "registered_diagnostics": {
            "bocpd_posterior_hidden_accuracy": means["hidden_switch::bocpd_posterior"],
            "oracle_hidden_accuracy": means["hidden_switch::oracle_change_reset"],
            "detector_recall_session_mean": float(detector_hidden["detection_recall"].mean()),
        },
        "support_gates": {key: bool(value) for key, value in gates.items()},
        "stationary_forgetting_matches_or_beats_detector": bool(stationary_matches),
        "stop_rule_triggered": bool(stationary_matches),
        "conclusion": conclusion,
    }
    return sessions, comparison_frame, summary


def _report(summary: Mapping[str, Any]) -> str:
    delay = summary["cohort_median_detection_delay"]
    delay_text = "not estimable" if delay is None else f"{float(delay):.3f} frames"
    lines = [
        "# Exp37 Bayesian Change-Aware Prefix Confirmation",
        "",
        "Evidence status: prospectively frozen session-held CORe50 evaluation.",
        "",
        f"Independent unit: session (n={summary['n_sessions']}); seeds, tasks, objects, and frames were averaged within session.",
        "",
        "## Primary registered comparisons",
        "",
    ]
    for item in summary["comparisons"]:
        lines.append(
            f"- {item['comparison']}: {item['mean_difference']:+.4f} "
            f"(95% session bootstrap {item['ci_low']:+.4f}, {item['ci_high']:+.4f}; "
            f"Holm p={item['holm_adjusted_pvalue']:.6g})."
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
            "This verdict concerns temporal decision state under frozen CORe50 evidence. It is not an official continual-learning, SOTA, or biological claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/prospective/exp37_core50_change_aware_prefix.json"
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
        else results_root / "exp37_core50_change_aware_prefix_confirmation"
    )
    output.mkdir(parents=True, exist_ok=False)
    feature_cache = validate_feature_cache(config)
    runs = eligible_run_dirs(results_root, seeds=seed_list(config["seeds"]))
    raw, selections, manifest = load_panel(runs, config=config)
    sessions, comparisons, summary = summarize_panel(raw, config=config)
    summary["feature_cache"] = feature_cache
    summary["selection_stability"] = selections.to_dict("records")
    raw.to_csv(
        output / "raw_external_task_panel.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    sessions.to_csv(output / "session_panel.csv", index=False)
    comparisons.to_csv(output / "primary_session_comparisons.csv", index=False)
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
