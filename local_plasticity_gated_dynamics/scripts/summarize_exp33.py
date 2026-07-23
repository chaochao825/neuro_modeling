"""Aggregate Exp33 without treating frames, videos, tasks, or seeds as users."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config, seed_list
from experiments.exp33_orbit_streaming_fewshot import (
    EVALUATION_CONDITIONS,
    EXPERIMENT,
    PROTOCOL_VERSION,
)
from src.analysis.orbit_streaming_metrics import (
    holm_adjust,
    paired_user_inference,
    reduce_to_user_accuracy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def eligible_run_dirs(
    results_root: Path,
    *,
    seeds: Iterable[int],
    profile: str,
) -> list[Path]:
    selected: list[Path] = []
    for seed in seeds:
        root = results_root / "runs" / EXPERIMENT / f"seed_{int(seed):04d}"
        candidates = []
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
                f"seed {seed} has {len(candidates)} eligible Exp33 {profile} runs; "
                "pass an isolated results root before summarizing"
            )
        selected.append(candidates[0])
    return selected


def load_panel(
    run_dirs: Iterable[Path], *, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_frames: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    expected_conditions = set(EVALUATION_CONDITIONS)
    for path in run_dirs:
        status = _read_json(path / "status.json")
        observed_config = _read_json(path / "config.json")
        if observed_config.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"{path} has a different Exp33 protocol")
        if observed_config.get("profile") != config.get("profile"):
            raise RuntimeError(f"{path} has a different profile")
        seed = int(status["seed"])
        raw = pd.read_csv(path / "raw_video_metrics.csv")
        required = {
            "user_id",
            "task_index",
            "video_id",
            "condition",
            "frame_accuracy",
            "status",
        }
        if not required <= set(raw.columns):
            raise RuntimeError(f"{path} raw metrics miss registered columns")
        if set(raw["condition"]) != expected_conditions:
            raise RuntimeError(f"{path} does not contain every Exp33 condition")
        if not raw["status"].eq("complete").all():
            raise RuntimeError(f"{path} contains failed evaluated videos")
        duplicates = raw.duplicated(["user_id", "task_index", "video_id", "condition"])
        if duplicates.any():
            raise RuntimeError(f"{path} contains duplicate task/video conditions")
        raw["seed"] = seed
        raw["run_path"] = str(path.resolve())
        raw_frames.append(raw)
        diagnostic = pd.read_csv(path / "actuator_headroom.csv")
        diagnostic["seed"] = seed
        diagnostics.append(diagnostic)
        manifests.append(
            {
                "seed": seed,
                "run_path": str(path.resolve()),
                "run_status": status["status"],
                "condition_failures": int(status["condition_failures"]),
                "condition_invalid": int(status["condition_invalid"]),
            }
        )
    panel = pd.concat(raw_frames, ignore_index=True)
    if (
        panel.groupby(["seed", "user_id", "task_index", "video_id"])[
            "episode_fingerprint"
        ]
        .nunique()
        .max()
        != 1
    ):
        raise RuntimeError("Exp33 conditions do not share the same episode tape")
    if (
        panel.groupby(["seed", "user_id", "task_index", "video_id"])[
            "trace_fingerprint"
        ]
        .nunique()
        .max()
        != 1
    ):
        raise RuntimeError("Exp33 conditions do not share the same actuator trace")
    return (
        panel,
        pd.concat(diagnostics, ignore_index=True),
        pd.DataFrame(manifests).sort_values("seed"),
    )


def summarize_panel(
    raw: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    per_seed_users = []
    for seed, frame in raw.groupby("seed", sort=True):
        reduced = reduce_to_user_accuracy(frame)
        reduced["seed"] = int(seed)
        per_seed_users.append(reduced)
    seed_user = pd.concat(per_seed_users, ignore_index=True)
    # Algorithmic seeds are repeated fits on the same participants.  Average
    # them within participant before any inferential comparison.
    user_panel = (
        seed_user.groupby(["user_id", "condition"], as_index=False)[
            "user_video_mean_accuracy"
        ]
        .mean()
        .sort_values(["user_id", "condition"])
        .reset_index(drop=True)
    )
    analysis = dict(config["analysis"])
    bootstrap_samples = int(analysis.get("bootstrap_samples", 20_000))
    statistics_seed = int(analysis.get("statistics_seed", 33033))
    comparisons = []
    for index, comparator in enumerate(("train_fixed_best", "credit_shuffled_local")):
        inference = paired_user_inference(
            user_panel,
            method="reward_only_local",
            comparator=comparator,
            bootstrap_samples=bootstrap_samples,
            seed=statistics_seed + index,
        )
        comparisons.append(inference)
    adjusted = holm_adjust(item.sign_flip_pvalue for item in comparisons)
    comparison_rows = pd.DataFrame(
        [
            {
                **asdict(item),
                "holm_adjusted_pvalue": float(adjusted[index]),
            }
            for index, item in enumerate(comparisons)
        ]
    )
    headroom_user = (
        diagnostics.groupby("user_id", as_index=False)[
            ["oracle_gain", "action_disagreement"]
        ]
        .mean()
        .sort_values("user_id")
    )
    mean_headroom = float(headroom_user["oracle_gain"].mean())
    mean_disagreement = float(headroom_user["action_disagreement"].mean())
    fixed = comparisons[0]
    shuffled = comparisons[1]
    headroom_gate = mean_headroom >= float(
        analysis["minimum_oracle_headroom"]
    ) and mean_disagreement >= float(analysis["minimum_action_disagreement"])
    trend_gate = fixed.mean_difference > 0.0 and shuffled.mean_difference > 0.0
    scale_decision = (
        "scale-authorized" if headroom_gate and trend_gate else "scale-not-authorized"
    )
    profile = str(config["profile"])
    conclusion = "inconclusive"
    reason = "development panel is used only for a scale decision"
    if profile == "formal":
        minimum = float(analysis["minimum_accuracy_gain"])
        significance = bool(
            np.all(adjusted <= 0.05) and fixed.ci_low > 0.0 and shuffled.ci_low > 0.0
        )
        if fixed.mean_difference >= minimum and headroom_gate and significance:
            conclusion = "support"
            reason = "all preregistered user-level gates passed"
        elif fixed.ci_high < minimum or not headroom_gate:
            conclusion = "oppose"
            reason = "registered performance or headroom gate failed"
        else:
            reason = "formal user-level interval did not resolve the claim"
    summary = {
        "experiment": EXPERIMENT,
        "protocol_version": PROTOCOL_VERSION,
        "profile": profile,
        "n_seeds": int(raw["seed"].nunique()),
        "n_users": int(user_panel["user_id"].nunique()),
        "mean_oracle_headroom": mean_headroom,
        "mean_action_disagreement": mean_disagreement,
        "local_vs_train_fixed": asdict(fixed),
        "local_vs_credit_shuffled": asdict(shuffled),
        "holm_adjusted_pvalues": adjusted.tolist(),
        "scale_decision": scale_decision,
        "claim_classification": conclusion,
        "claim_reason": reason,
    }
    return comparison_rows, {
        "summary": summary,
        "user_panel": user_panel,
        "headroom_user": headroom_user,
    }


def _report(summary: dict[str, Any], condition_means: pd.Series) -> str:
    fixed = summary["local_vs_train_fixed"]
    shuffled = summary["local_vs_credit_shuffled"]
    lines = [
        "# Exp33 ORBIT streaming few-shot report",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Users (statistical unit): {summary['n_users']}",
        f"- Algorithmic seeds averaged within user: {summary['n_seeds']}",
        f"- Scale decision: **{summary['scale_decision']}**",
        f"- Claim classification: **{summary['claim_classification']}**",
        "",
        "## Held-out accuracy",
        "",
        "| Condition | Mean user/video accuracy |",
        "|---|---:|",
    ]
    for condition in EVALUATION_CONDITIONS:
        lines.append(f"| {condition} | {float(condition_means[condition]):.4f} |")
    lines.extend(
        [
            "",
            "## Registered comparisons",
            "",
            (
                "- Reward-only local minus train-selected fixed: "
                f"{fixed['mean_difference']:+.4f} "
                f"(95% user bootstrap {fixed['ci_low']:+.4f}, "
                f"{fixed['ci_high']:+.4f})."
            ),
            (
                "- Reward-only local minus credit-shuffled local: "
                f"{shuffled['mean_difference']:+.4f} "
                f"(95% user bootstrap {shuffled['ci_low']:+.4f}, "
                f"{shuffled['ci_high']:+.4f})."
            ),
            f"- Mean oracle headroom: {summary['mean_oracle_headroom']:.4f}.",
            f"- Mean action disagreement: {summary['mean_action_disagreement']:.4f}.",
            "",
            "This report does not treat frames, videos, tasks, or repeated seeds as independent participants. Development results cannot support a confirmatory claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/smoke/exp33_orbit_streaming_fewshot.json"
    )
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_json_config(args.config)
    results_root = Path(args.results_root).expanduser().resolve()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_root / "exp33_orbit_streaming_fewshot"
    )
    output.mkdir(parents=True, exist_ok=True)
    runs = eligible_run_dirs(
        results_root,
        seeds=seed_list(config["seeds"]),
        profile=str(config["profile"]),
    )
    raw, diagnostics, manifest = load_panel(runs, config=config)
    comparisons, payload = summarize_panel(raw, diagnostics, config=config)
    user_panel = payload["user_panel"]
    headroom_user = payload["headroom_user"]
    summary = payload["summary"]
    raw.to_csv(output / "raw_video_panel.csv", index=False)
    diagnostics.to_csv(output / "headroom_panel.csv", index=False)
    user_panel.to_csv(output / "user_panel.csv", index=False)
    headroom_user.to_csv(output / "headroom_user.csv", index=False)
    comparisons.to_csv(output / "paired_user_comparisons.csv", index=False)
    manifest.to_csv(output / "run_manifest.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    means = user_panel.groupby("condition")["user_video_mean_accuracy"].mean()
    (output / "report.md").write_text(_report(summary, means), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
