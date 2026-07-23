"""Aggregate Exp34 at the user level and freeze its formal scale receipt."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config, seed_list
from experiments.exp34_orbit_causal_consensus import (
    AUTHORIZATION_SCHEMA,
    EVALUATION_CONDITIONS,
    EXPERIMENT,
    PROTOCOL_VERSION,
    _canonical_sha256,
    feature_store_hashes,
    formal_config_fingerprint,
    implementation_hashes,
)
from src.analysis.orbit_streaming_metrics import (
    holm_adjust,
    paired_user_inference,
    reduce_to_user_accuracy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARATORS = (
    "selection_fixed_best",
    "memoryless_consensus",
    "instantaneous_majority",
    "delayed_consensus",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
                f"seed {seed} has {len(candidates)} eligible Exp34 {profile} runs; "
                "pass an isolated results root before summarizing"
            )
        selected.append(candidates[0])
    return selected


def load_panel(
    run_dirs: Iterable[Path], *, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_frames: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    expected_conditions = set(EVALUATION_CONDITIONS)
    for path in run_dirs:
        status = _read_json(path / "status.json")
        observed_config = _read_json(path / "config.json")
        if observed_config.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"{path} has a different Exp34 protocol")
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
            "episode_fingerprint",
            "trace_fingerprint",
        }
        if not required <= set(raw.columns):
            raise RuntimeError(f"{path} raw metrics miss registered columns")
        if set(raw["condition"]) != expected_conditions:
            raise RuntimeError(f"{path} does not contain every Exp34 condition")
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
    tape_groups = panel.groupby(["seed", "user_id", "task_index", "video_id"])
    if tape_groups["episode_fingerprint"].nunique().max() != 1:
        raise RuntimeError("Exp34 conditions do not share the same episode tape")
    if tape_groups["trace_fingerprint"].nunique().max() != 1:
        raise RuntimeError("Exp34 conditions do not share the same actuator trace")
    return (
        panel,
        pd.concat(diagnostics, ignore_index=True),
        pd.DataFrame(manifests).sort_values("seed"),
    )


def summarize_panel(
    raw: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    per_seed_users = []
    for seed, frame in raw.groupby("seed", sort=True):
        reduced = reduce_to_user_accuracy(frame)
        reduced["seed"] = int(seed)
        per_seed_users.append(reduced)
    seed_user = pd.concat(per_seed_users, ignore_index=True)
    # Seeds are repeated algorithmic runs on the same people.  They are
    # averaged inside user before any uncertainty calculation.
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
    statistics_seed = int(analysis.get("statistics_seed", 34034))
    comparisons = []
    for index, comparator in enumerate(COMPARATORS):
        comparisons.append(
            paired_user_inference(
                user_panel,
                method="causal_consensus",
                comparator=comparator,
                bootstrap_samples=bootstrap_samples,
                seed=statistics_seed + index,
            )
        )
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
    fixed, memoryless, majority, delayed = comparisons
    retained = fixed.mean_difference / mean_headroom if mean_headroom > 0.0 else 0.0
    minimum_gain = float(analysis["minimum_accuracy_gain"])
    minimum_retained = float(analysis["minimum_retained_oracle_headroom"])
    causal_gate = bool(
        fixed.mean_difference >= minimum_gain
        and memoryless.mean_difference > 0.0
        and majority.mean_difference > 0.0
        and delayed.mean_difference > 0.0
        and fixed.ci_low > 0.0
        and memoryless.ci_low > 0.0
        and majority.ci_low > 0.0
        and delayed.ci_low > 0.0
        and retained >= minimum_retained
        and mean_disagreement > 0.0
    )
    profile = str(config["profile"])
    scale_decision = "scale-authorized" if causal_gate else "scale-not-authorized"
    conclusion = "inconclusive"
    reason = "development users authorize scale only; they cannot support the claim"
    if profile == "formal":
        scale_decision = "not-applicable"
        significance = bool(
            adjusted[0] <= 0.05
            and adjusted[1] <= 0.05
            and adjusted[2] <= 0.05
            and adjusted[3] <= 0.05
            and fixed.ci_low > 0.0
            and memoryless.ci_low > 0.0
            and majority.ci_low > 0.0
            and delayed.ci_low > 0.0
        )
        if (
            fixed.mean_difference >= minimum_gain
            and retained >= minimum_retained
            and significance
        ):
            conclusion = "support"
            reason = "registered test-user task and causal-memory gates passed"
        elif (
            fixed.ci_high < minimum_gain
            or memoryless.ci_high <= 0.0
            or majority.ci_high <= 0.0
            or delayed.ci_high <= 0.0
        ):
            conclusion = "oppose"
            reason = "registered test-user task or causal-memory effect was absent"
        else:
            reason = "formal user-level uncertainty did not resolve the claim"
    means = user_panel.groupby("condition")["user_video_mean_accuracy"].mean()
    official_means = raw.groupby("condition")["frame_accuracy"].mean()
    summary = {
        "experiment": EXPERIMENT,
        "protocol_version": PROTOCOL_VERSION,
        "profile": profile,
        "n_seeds": int(raw["seed"].nunique()),
        "n_users": int(user_panel["user_id"].nunique()),
        "condition_user_mean_accuracy": {
            condition: float(means[condition]) for condition in EVALUATION_CONDITIONS
        },
        "official_task_video_mean_accuracy": {
            condition: float(official_means[condition])
            for condition in EVALUATION_CONDITIONS
        },
        "mean_oracle_headroom": mean_headroom,
        "mean_action_disagreement": mean_disagreement,
        "retained_oracle_headroom_fraction": retained,
        "comparisons": [asdict(item) for item in comparisons],
        "holm_adjusted_pvalues": adjusted.tolist(),
        "scale_decision": scale_decision,
        "claim_classification": conclusion,
        "claim_reason": reason,
        "statistical_unit": "user",
        "algorithmic_seed_handling": "averaged within user",
    }
    return comparison_rows, {
        "summary": summary,
        "user_panel": user_panel,
        "headroom_user": headroom_user,
    }


def build_authorization_receipt(
    summary: Mapping[str, Any],
    *,
    formal_config: Mapping[str, Any],
    development_summary_path: Path,
    run_manifest_path: Path,
) -> dict[str, Any]:
    if summary.get("profile") == "formal":
        raise ValueError("formal results cannot authorize themselves")
    if summary.get("scale_decision") != "scale-authorized":
        raise ValueError("Exp34 scale gate did not authorize formal evaluation")
    receipt: dict[str, Any] = {
        "schema": AUTHORIZATION_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "authorized": True,
        "scale_decision": "scale-authorized",
        "n_development_users": int(summary["n_users"]),
        "n_development_seeds": int(summary["n_seeds"]),
        "development_summary_sha256": _file_sha256(development_summary_path),
        "development_run_manifest_sha256": _file_sha256(run_manifest_path),
        "formal_config_fingerprint": formal_config_fingerprint(formal_config),
        "implementation_sha256": implementation_hashes(),
        "feature_store_sha256": feature_store_hashes(formal_config),
        "registered_gate_results": {
            "minimum_accuracy_gain": float(
                formal_config["analysis"]["minimum_accuracy_gain"]
            ),
            "minimum_retained_oracle_headroom": float(
                formal_config["analysis"]["minimum_retained_oracle_headroom"]
            ),
            "observed_retained_oracle_headroom": float(
                summary["retained_oracle_headroom_fraction"]
            ),
            "comparisons": summary["comparisons"],
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Exp34 ORBIT causal-consensus report",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Users (statistical unit): {summary['n_users']}",
        f"- Algorithmic seeds averaged within user: {summary['n_seeds']}",
        f"- Scale decision: **{summary['scale_decision']}**",
        f"- Claim classification: **{summary['claim_classification']}**",
        "",
        "## User-level accuracy",
        "",
        "| Condition | User-equal mean | Official task-video mean |",
        "|---|---:|---:|",
    ]
    means = summary["condition_user_mean_accuracy"]
    official = summary["official_task_video_mean_accuracy"]
    for condition in EVALUATION_CONDITIONS:
        lines.append(
            f"| {condition} | {float(means[condition]):.4f} | "
            f"{float(official[condition]):.4f} |"
        )
    lines.extend(["", "## Registered causal comparisons", ""])
    for item in summary["comparisons"]:
        lines.append(
            f"- Causal consensus minus {item['comparator']}: "
            f"{item['mean_difference']:+.4f} "
            f"(95% user bootstrap {item['ci_low']:+.4f}, "
            f"{item['ci_high']:+.4f}; positive users "
            f"{item['positive_users']}/{item['n_users']})."
        )
    lines.extend(
        [
            f"- Mean per-frame oracle headroom: {summary['mean_oracle_headroom']:.4f}.",
            "- Retained oracle headroom fraction: "
            f"{summary['retained_oracle_headroom_fraction']:.3f}.",
            f"- Mean actuator disagreement: {summary['mean_action_disagreement']:.4f}.",
            "",
            "The gate uses no query labels or future frames, but computes the full four-actuator bank. Its consensus signal is specific to ORBIT's one-object-per-video structure. Development users only authorize a frozen test run and never count as confirmatory evidence.",
            "The official-style point estimate flattens task/video samples for benchmark comparability; all hypothesis uncertainty still uses user as the independent unit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/smoke/exp34_orbit_causal_consensus.json"
    )
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--formal-config",
        default="configs/formal/exp34_orbit_causal_consensus.json",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    results_root = Path(args.results_root).expanduser().resolve()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_root / "exp34_orbit_causal_consensus"
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
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    if summary["scale_decision"] == "scale-authorized":
        formal_config = load_json_config(args.formal_config)
        receipt = build_authorization_receipt(
            summary,
            formal_config=formal_config,
            development_summary_path=summary_path,
            run_manifest_path=output / "run_manifest.csv",
        )
        (output / "scale_authorization.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(output)


if __name__ == "__main__":
    main()
