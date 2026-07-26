"""Create fail-closed qualification receipts or external Exp38 summaries."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config, seed_list
from experiments.exp38_stream51_soft_memory import PROTOCOL_VERSION
from src.analysis.orbit_streaming_metrics import holm_adjust, paired_user_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_summary(path_value: str | Path, *, stage: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Exp38 run summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol_version") != PROTOCOL_VERSION or summary.get("stage") != stage:
        raise ValueError(f"Exp38 run stage/version mismatch: {path}")
    return path, summary


def build_qualification_receipt(
    run_paths: Sequence[str | Path],
    *,
    config: Mapping[str, Any],
    output_path: str | Path,
    preregistration_receipt_path: str | Path,
    implementation_receipt_path: str | Path,
) -> dict[str, Any]:
    registered = tuple(seed_list(config["seeds"]))
    summaries: dict[int, tuple[Path, dict[str, Any]]] = {}
    for value in run_paths:
        path, summary = _run_summary(value, stage="qualification")
        seed = int(summary["seed"])
        if seed in summaries:
            raise ValueError(f"duplicate Exp38 qualification seed: {seed}")
        if summary.get("external_features_accessed") is not False:
            raise ValueError("qualification run accessed external features")
        summaries[seed] = (path, summary)
    if tuple(sorted(summaries)) != tuple(sorted(registered)):
        raise ValueError("Exp38 qualification run set does not match registered seeds")
    preregistration = Path(preregistration_receipt_path).expanduser().resolve()
    implementation = Path(implementation_receipt_path).expanduser().resolve()
    if not preregistration.is_file() or not implementation.is_file():
        raise FileNotFoundError("Exp38 frozen receipts must exist before qualification")
    seed_rows: list[dict[str, Any]] = []
    for seed in registered:
        path, summary = summaries[seed]
        qualification = dict(summary["qualification"])
        seed_rows.append(
            {
                "seed": seed,
                "passed": bool(qualification["passed"]),
                "run_path": str(path),
                "summary_sha256": _sha256(path / "summary.json"),
                "qualification": qualification,
                "selected_hyperparameters": summary["selected_hyperparameters"],
            }
        )
    all_passed = bool(all(bool(row["passed"]) for row in seed_rows))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "receipt_type": "exp38_qualification_gate",
        "seeds": list(registered),
        "require_all_seeds": True,
        "all_registered_seeds_passed": all_passed,
        "external_stage_authorized": all_passed,
        "external_outcomes_inspected": False,
        "preregistration_receipt_path": str(preregistration),
        "preregistration_receipt_sha256": _sha256(preregistration),
        "implementation_receipt_path": str(implementation),
        "implementation_receipt_sha256": _sha256(implementation),
        "seed_results": seed_rows,
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _external_unit_table(run_paths: Sequence[str | Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for value in run_paths:
        path, summary = _run_summary(value, stage="external")
        seed = int(summary["seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate Exp38 external seed: {seed}")
        seen_seeds.add(seed)
        frame_path = path / "external_video_metrics.csv"
        if not frame_path.is_file():
            raise FileNotFoundError(f"Exp38 external metrics are missing: {frame_path}")
        frame = pd.read_csv(frame_path, keep_default_na=False)
        required = {
            "source_video_id",
            "panel",
            "condition",
            "accuracy",
            "post_switch_accuracy",
        }
        if not required <= set(frame.columns):
            raise ValueError("Exp38 external metrics schema is incomplete")
        frame["seed"] = seed
        frames.append(frame)
        summaries.append(summary)
    raw = pd.concat(frames, ignore_index=True)
    raw["accuracy"] = pd.to_numeric(raw["accuracy"], errors="raise")
    if not np.isfinite(raw["accuracy"]).all():
        raise ValueError("Exp38 external accuracy must be finite")
    units = (
        raw.groupby(
            ["source_video_id", "panel", "condition"], as_index=False, sort=True
        )["accuracy"]
        .mean()
        .rename(columns={"accuracy": "seed_mean_accuracy"})
    )
    seed_counts = raw.groupby(["source_video_id", "panel", "condition"])["seed"].nunique()
    if seed_counts.nunique() != 1 or int(seed_counts.iloc[0]) != len(seen_seeds):
        raise ValueError("Exp38 source videos are not complete across seeds")
    return units, summaries


def summarize_external(
    run_paths: Sequence[str | Path],
    *,
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    registered = tuple(seed_list(config["seeds"]))
    units, run_summaries = _external_unit_table(run_paths)
    observed_seeds = tuple(sorted(int(summary["seed"]) for summary in run_summaries))
    if observed_seeds != tuple(sorted(registered)):
        raise ValueError("Exp38 external run set does not match registered seeds")
    analysis = config["analysis"]
    comparisons_spec = (
        ("soft_over_fixed_forgetting_hidden", "hidden_switch", "soft_memory", "fixed_forgetting"),
        ("soft_over_sliding_window_hidden", "hidden_switch", "soft_memory", "sliding_window"),
        ("soft_over_hard_hidden", "hidden_switch", "soft_memory", "hard_memory"),
        (
            "soft_over_shifted_timing_hidden",
            "hidden_switch",
            "soft_memory",
            "matched_shifted_memory",
        ),
        ("soft_noninferior_current_natural", "natural", "soft_memory", "current_frame"),
    )
    comparisons: list[dict[str, Any]] = []
    for index, (name, panel, method, comparator) in enumerate(comparisons_spec):
        subset = units[units["panel"] == panel].rename(
            columns={
                "source_video_id": "user_id",
                "seed_mean_accuracy": "user_video_mean_accuracy",
            }
        )
        inference = paired_user_inference(
            subset[["user_id", "condition", "user_video_mean_accuracy"]],
            method=method,
            comparator=comparator,
            bootstrap_samples=int(analysis["bootstrap_samples"]),
            seed=int(analysis["statistics_seed"]) + index,
        )
        comparisons.append({"comparison": name, "panel": panel, **asdict(inference)})
    adjusted = holm_adjust(item["sign_flip_pvalue"] for item in comparisons)
    for item, value in zip(comparisons, adjusted, strict=True):
        item["holm_adjusted_pvalue"] = float(value)
    by_name = {str(item["comparison"]): item for item in comparisons}
    fixed = by_name["soft_over_fixed_forgetting_hidden"]
    window = by_name["soft_over_sliding_window_hidden"]
    natural = by_name["soft_noninferior_current_natural"]
    alpha = float(analysis["alpha"])
    mcid = float(analysis["hidden_switch_mcid"])
    margin = float(analysis["natural_noninferiority_margin"])
    support = bool(
        all(
            float(item["mean_difference"]) >= mcid
            and float(item["ci_low"]) > 0.0
            and float(item["holm_adjusted_pvalue"]) <= alpha
            for item in (fixed, window)
        )
        and float(natural["ci_low"]) > -margin
    )
    oppose = bool(
        not support
        and (
            float(fixed["ci_high"]) < mcid or float(window["ci_high"]) < mcid
        )
    )
    verdict = "support" if support else "oppose" if oppose else "inconclusive"
    means = (
        units.groupby(["panel", "condition"], as_index=False)["seed_mean_accuracy"]
        .mean()
        .rename(columns={"seed_mean_accuracy": "video_equal_accuracy"})
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    units.to_csv(output / "external_video_seed_mean_metrics.csv", index=False)
    means.to_csv(output / "condition_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "comparisons.csv", index=False)
    summary = {
        "experiment": "exp38_stream51_soft_memory",
        "protocol_version": PROTOCOL_VERSION,
        "stage": "external_summary",
        "verdict": verdict,
        "statistical_unit": "source_video",
        "seeds_averaged_within_video": True,
        "n_seeds": len(registered),
        "n_external_videos": int(units["source_video_id"].nunique()),
        "coverage_complete": True,
        "primary_mcid": mcid,
        "natural_noninferiority_margin": margin,
        "comparisons": comparisons,
        "condition_means": means.to_dict("records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Exp38 Stream-51 causal soft-memory report",
        "",
        f"Verdict: **{verdict.upper()}**.",
        "",
        "The independent unit is the source video; five assembly seeds were averaged within video before paired inference.",
        "",
        "## Registered comparisons",
        "",
    ]
    for item in comparisons:
        report.append(
            f"- `{item['comparison']}`: {item['mean_difference']:+.4f} "
            f"(95% video bootstrap {item['ci_low']:+.4f}, {item['ci_high']:+.4f}; "
            f"Holm p={item['holm_adjusted_pvalue']:.6g})."
        )
    report.extend(
        [
            "",
            "Support requires the registered practical gain against both fixed-time baselines plus natural-video non-inferiority; low controller dimension or switch recall alone is not sufficient.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("qualification", "external"), required=True)
    parser.add_argument("--run-path", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preregistration-receipt",
        type=Path,
        default=PROJECT_ROOT / "provenance/exp38_preregistration_receipt_20260726.json",
    )
    parser.add_argument(
        "--implementation-receipt",
        type=Path,
        default=PROJECT_ROOT / "provenance/exp38_implementation_receipt_20260726.json",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    if args.stage == "qualification":
        payload = build_qualification_receipt(
            args.run_path,
            config=config,
            output_path=args.output,
            preregistration_receipt_path=args.preregistration_receipt,
            implementation_receipt_path=args.implementation_receipt,
        )
    else:
        payload = summarize_external(
            args.run_path, config=config, output_dir=args.output
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
