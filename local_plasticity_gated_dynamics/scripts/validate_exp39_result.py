#!/usr/bin/env python3
"""Independent completeness and replay audit for the formal Exp39 package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp39_factorized_uncertainty import (
    METHODS,
    _read_json,
    _write_json,
    summarize,
    validate_config,
    validate_implementation_receipt,
)


def validate_result(result_dir: Path) -> dict[str, Any]:
    root = result_dir.resolve()
    required = {
        "config.json",
        "planned_conditions.json",
        "execution_amendment.json",
        "failures.json",
        "block_metrics.csv",
        "selection_audit.csv",
        "seed_metrics.csv",
        "comparisons_and_clamps.csv",
        "parameter_tracking.csv",
        "summary.json",
        "report.md",
        "status.json",
        "run.log",
        "launcher.log",
        "launcher.status",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"formal package is missing files: {missing}")
    config = _read_json(root / "config.json")
    validate_config(config, formal=True)
    receipt = validate_implementation_receipt(config)
    failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))
    if failures != []:
        raise RuntimeError("formal package contains failed seeds")
    launcher_status = (root / "launcher.status").read_text(encoding="utf-8").strip()
    if launcher_status != "0":
        raise RuntimeError(f"parallel launcher status is {launcher_status}")
    seeds = tuple(map(int, config["seeds"]))
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        status = _read_json(seed_root / "status.json")
        if status != {"seed": seed, "status": "complete"}:
            raise RuntimeError(f"seed {seed} is not complete")
        for name in ("block_metrics.csv", "selection_audit.csv", "metadata.json"):
            if not (seed_root / name).is_file():
                raise RuntimeError(f"seed {seed} lacks {name}")

    blocks = pd.read_csv(root / "block_metrics.csv", dtype={"cell": str})
    expected_blocks = (
        len(seeds)
        * len(METHODS)
        * int(config["stream"]["n_sequences"])
        * int(config["stream"]["blocks_per_sequence"])
    )
    if len(blocks) != expected_blocks:
        raise RuntimeError(
            f"expected {expected_blocks} block rows, found {len(blocks)}"
        )
    if set(blocks["seed"]) != set(seeds) or set(blocks["method"]) != set(METHODS):
        raise RuntimeError("seed or method coverage is incomplete")
    duplicate_key = ["seed", "method", "block_id"]
    if blocks.duplicated(duplicate_key).any():
        raise RuntimeError("duplicate seed/method/block metric rows")
    per_method = blocks.groupby(["seed", "method"]).size()
    expected_per_method = (
        int(config["stream"]["n_sequences"])
        * int(config["stream"]["blocks_per_sequence"])
    )
    if not per_method.eq(expected_per_method).all():
        raise RuntimeError("method-level block coverage is incomplete")
    if not blocks.groupby("seed")["test_tape_digest"].nunique().eq(1).all():
        raise RuntimeError("methods did not share one test tape per seed")
    expected_cell_count = (
        int(config["stream"]["n_sequences"])
        * int(config["stream"]["blocks_per_sequence"])
        // 8
    )
    cell_counts = (
        blocks.loc[blocks["method"] == "factorized"]
        .groupby(["seed", "cell"])
        .size()
    )
    if not cell_counts.eq(expected_cell_count).all():
        raise RuntimeError("factorial cell coverage is unbalanced")

    selection = pd.read_csv(root / "selection_audit.csv")
    selected = selection.loc[selection["selected"].astype(bool)]
    required_families = {"fixed", "factorized", "seen_imm", "oracle_imm"}
    if set(selected["selection_family"]) != required_families:
        raise RuntimeError("selection families are incomplete")
    if not selected.groupby(["seed", "selection_family"]).size().eq(1).all():
        raise RuntimeError("each seed/family must select exactly one candidate")

    replay_seed, _, replay_tracking, replay_summary = summarize(
        blocks, config=config
    )
    stored_seed = pd.read_csv(root / "seed_metrics.csv")
    assert_frame_equal(
        replay_seed.reset_index(drop=True),
        stored_seed.reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    stored_tracking = pd.read_csv(root / "parameter_tracking.csv")
    assert_frame_equal(
        replay_tracking.reset_index(drop=True),
        stored_tracking.reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    stored_summary = _read_json(root / "summary.json")
    if stored_summary["verdict"] != replay_summary["verdict"]:
        raise RuntimeError("stored and replayed verdicts disagree")
    if stored_summary["joint_gate_passed"] != replay_summary["joint_gate_passed"]:
        raise RuntimeError("stored and replayed joint gates disagree")
    for family in ("best_fixed", "seen_mode_imm"):
        observed = stored_summary["utility_gates"][family]["mean_nll_gain"]
        replayed = replay_summary["utility_gates"][family]["mean_nll_gain"]
        if not np.isclose(observed, replayed, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"stored and replayed {family} gains disagree")
    amendment = _read_json(root / "execution_amendment.json")
    if amendment["scientific_functions_changed"] is not False:
        raise RuntimeError("execution amendment changed scientific functions")
    if amendment["classification_outcomes_inspected_before_amendment"] is not False:
        raise RuntimeError("execution amendment was not outcome-blind")
    return {
        "audit_status": "passed",
        "protocol_version": config["protocol_version"],
        "implementation_frozen_at": receipt["frozen_at"],
        "n_seeds": len(seeds),
        "n_block_rows": len(blocks),
        "n_failed_seeds": 0,
        "methods": list(METHODS),
        "statistics_unit": "seed",
        "paired_tape_check": "passed",
        "selection_replay_check": "passed",
        "summary_replay_check": "passed",
        "verdict": stored_summary["verdict"],
        "joint_gate_passed": stored_summary["joint_gate_passed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_result(args.result_dir)
    destination = (
        args.output.resolve()
        if args.output is not None
        else args.result_dir.resolve() / "audit_receipt.json"
    )
    if destination.exists():
        raise FileExistsError(destination)
    _write_json(destination, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
