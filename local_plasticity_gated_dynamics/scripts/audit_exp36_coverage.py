#!/usr/bin/env python3
"""Outcome-blind schema and run-coverage audit for invalid Exp36-v1 runs.

This audit deliberately does not load ``external_task_metrics.csv`` and never
reads accuracy or detector-utility fields from ``metrics.jsonl``.  It exists to
record whether the frozen inferential panel could be instantiated, not to turn
an incomplete panel into an exploratory result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config, seed_list
from experiments.exp36_change_aware_prefix import EXPERIMENT, PROTOCOL_VERSION
from scripts.summarize_exp36 import eligible_run_dirs, validate_external_feature_cache


OUTCOME_FIELDS = frozenset(
    {
        "accuracy",
        "post_switch_accuracy",
        "detection_precision",
        "detection_recall",
        "false_alarms_per_1000",
        "median_detection_delay",
        "mean_state_l1",
    }
)
FAILURE_FIELDS = (
    "seed",
    "collector_id",
    "task_index",
    "panel",
    "condition",
    "status",
    "error_type",
    "error",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _failure_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Yield only failed records without deserializing completed outcomes."""

    status_pattern = re.compile(r'"status"\s*:\s*"failed"')
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not status_pattern.search(line):
                continue
            record = json.loads(line)
            leaked = OUTCOME_FIELDS.intersection(record)
            # Failed records must not carry outcomes.  Failing here prevents a
            # future artifact change from silently widening this audit's scope.
            if leaked:
                raise RuntimeError(
                    f"failed record unexpectedly contains outcome fields: {sorted(leaked)}"
                )
            yield {field: record.get(field) for field in FAILURE_FIELDS}


def collector_schema_coverage(feature_manifest: Path) -> pd.DataFrame:
    manifest = pd.read_csv(feature_manifest, keep_default_na=False)
    required = {"user_id", "object_name", "video_type", "video_id", "n_frames"}
    if not required <= set(manifest.columns):
        raise RuntimeError(
            f"feature manifest misses {sorted(required - set(manifest.columns))}"
        )
    if manifest["video_id"].duplicated().any():
        raise RuntimeError("feature manifest contains duplicate video IDs")
    rows: list[dict[str, Any]] = []
    for collector, group in manifest.groupby("user_id", sort=True):
        object_types = (
            group.groupby("object_name")["video_type"].agg(lambda x: frozenset(map(str, x)))
        )
        complete = [
            str(name)
            for name, types in object_types.items()
            if {"clean", "clutter"} <= set(types)
        ]
        incomplete = [str(name) for name in object_types.index if str(name) not in complete]
        rows.append(
            {
                "collector_id": str(collector),
                "n_objects": int(len(object_types)),
                "n_complete_clean_clutter_objects": int(len(complete)),
                "n_incomplete_objects": int(len(incomplete)),
                "incomplete_objects": "|".join(sorted(incomplete)),
                "n_videos": int(len(group)),
                "n_frames": int(pd.to_numeric(group["n_frames"], errors="raise").sum()),
                # The frozen sampler draws from every named object.  Merely
                # having four complete pairs is insufficient when another
                # candidate object lacks either clean or clutter video.
                "eligible_for_registered_sampling": bool(
                    len(complete) >= 4 and not incomplete
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_exp36_coverage(
    *, results_root: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Exp36 protocol mismatch")
    if config.get("profile") != "prospective_external":
        raise RuntimeError("coverage audit requires prospective_external profile")
    seeds = tuple(seed_list(config["seeds"]))
    run_dirs = eligible_run_dirs(results_root, seeds=seeds, profile="prospective_external")
    feature_info = validate_external_feature_cache(config)
    schema = collector_schema_coverage(
        Path(str(config["external_feature_root"])).expanduser().resolve()
        / "feature_manifest.csv"
    )

    run_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        status = _read_json(run_dir / "status.json")
        summary = _read_json(run_dir / "summary.json")
        coverage = summary.get("coverage", {})
        run_rows.append(
            {
                "seed": int(status["seed"]),
                "run_path": str(run_dir.resolve()),
                "run_status": str(status["status"]),
                "condition_failures": int(status.get("condition_failures", 0)),
                "condition_invalid": int(status.get("condition_invalid", 0)),
                "expected_collectors": int(coverage.get("expected_collectors", -1)),
                "observed_collectors": int(coverage.get("observed_collectors", -1)),
                "expected_conditions": int(coverage.get("expected_conditions", -1)),
                "observed_conditions": int(coverage.get("observed_conditions", -1)),
                "coverage_complete": bool(coverage.get("complete", False)),
            }
        )
        failed_rows.extend(_failure_rows(run_dir / "metrics.jsonl"))

    runs = pd.DataFrame(run_rows).sort_values("seed", ignore_index=True)
    failures = pd.DataFrame(failed_rows, columns=FAILURE_FIELDS)
    if runs["coverage_complete"].any():
        raise RuntimeError("invalid-panel audit received a coverage-complete run")
    if failures.empty:
        raise RuntimeError("incomplete Exp36 panel has no auditable failure records")
    if int(runs["condition_failures"].sum()) != len(failures):
        raise RuntimeError("failure records do not match status counters")

    grouped = (
        failures.groupby(
            ["collector_id", "error_type", "error"], dropna=False, sort=True
        )
        .size()
        .rename("n_failed_cells")
        .reset_index()
    )
    failed_collectors = sorted(map(str, failures["collector_id"].unique()))
    eligible = sorted(
        schema.loc[
            schema["eligible_for_registered_sampling"], "collector_id"
        ].astype(str)
    )
    registered = list(map(str, config["external_collectors"]))
    observed_counts = Counter(runs["observed_collectors"].astype(int))
    summary = {
        "experiment": EXPERIMENT,
        "protocol_version": PROTOCOL_VERSION,
        "audit_type": "outcome_blind_schema_and_coverage",
        "status": "invalid",
        "conclusion": "inconclusive",
        "claim_upgrade_allowed": False,
        "accuracy_fields_read": False,
        "reason": (
            "The frozen 12-collector ORBIT-India cohort cannot instantiate the "
            "registered four-class paired clean/clutter panel."
        ),
        "n_seeds": int(len(runs)),
        "registered_collectors": registered,
        "n_registered_collectors": int(len(registered)),
        "schema_eligible_collectors": eligible,
        "n_schema_eligible_collectors": int(len(eligible)),
        "failed_collectors": failed_collectors,
        "n_failed_collectors": int(len(failed_collectors)),
        "condition_failures_total": int(len(failures)),
        "condition_failures_per_seed": {
            str(row.seed): int(row.condition_failures)
            for row in runs.itertuples(index=False)
        },
        "observed_collectors_per_seed": {
            str(key): int(value) for key, value in sorted(observed_counts.items())
        },
        "feature_cache": feature_info,
        "next_action": (
            "Do not summarize the four surviving collectors as confirmation; "
            "register a new schema-compatible dataset before outcome access."
        ),
    }
    return runs, grouped, schema, summary


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Exp36-v1 outcome-blind validity audit",
                "",
                "**Conclusion: inconclusive (invalid registered panel).**",
                "",
                str(summary["reason"]),
                "",
                f"- Seeds completed: {summary['n_seeds']}",
                f"- Registered collectors: {summary['n_registered_collectors']}",
                f"- Schema-eligible collectors: {summary['n_schema_eligible_collectors']}",
                f"- Failed registered cells: {summary['condition_failures_total']}",
                "- Accuracy fields inspected by this audit: no",
                "- Claim upgrade allowed: no",
                "",
                "The failed runs and all incomplete conditions are retained. The four "
                "surviving collectors are not analyzed as a prospective confirmation.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_json_config(args.config)
    runs, failures, schema, summary = audit_exp36_coverage(
        results_root=args.results_root.resolve(), config=config
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs.to_csv(output / "run_coverage.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    failures.to_csv(output / "failure_groups.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    schema.to_csv(
        output / "collector_schema_coverage.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(output / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
