#!/usr/bin/env python3
"""Outcome-blind CPU orchestration for the hash-frozen Exp39 seed function."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp39_factorized_uncertainty import (
    METHODS,
    _read_json,
    _report,
    _write_json,
    run_seed,
    summarize,
    validate_config,
    validate_implementation_receipt,
)


def _one_seed(
    config: Mapping[str, Any], seed: int
) -> tuple[int, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    blocks, audit, metadata = run_seed(config, seed)
    return seed, blocks, audit, metadata


def parallel_execute(
    config_path: Path, output_dir: Path, *, workers: int
) -> dict[str, Any]:
    """Execute independent frozen seeds in parallel; retain every failure."""

    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    config = _read_json(config_path)
    formal = bool(config["claim_upgrade_allowed"])
    validate_config(config, formal=formal)
    if formal:
        validate_implementation_receipt(config)
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config.json")
    _write_json(
        output / "execution_amendment.json",
        {
            "classification_outcomes_inspected_before_amendment": False,
            "scientific_functions_changed": False,
            "frozen_seed_function": (
                "experiments.exp39_factorized_uncertainty.run_seed"
            ),
            "frozen_summary_function": (
                "experiments.exp39_factorized_uncertainty.summarize"
            ),
            "workers": int(workers),
            "reason": "Dispatch independent formal seeds across CPU workers",
        },
    )
    _write_json(
        output / "planned_conditions.json",
        {
            "methods": list(METHODS),
            "fit_cells": list(config["partitions"]["fit_cells"]),
            "heldout_composition_cells": list(
                config["partitions"]["heldout_composition_cells"]
            ),
            "seeds": list(map(int, config["seeds"])),
        },
    )
    logger = logging.getLogger(f"exp39.parallel.{output.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    logger.addHandler(handler)
    blocks: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {
                pool.submit(_one_seed, config, int(seed)): int(seed)
                for seed in config["seeds"]
            }
            for future in as_completed(futures):
                seed = futures[future]
                seed_dir = output / f"seed_{seed}"
                seed_dir.mkdir()
                try:
                    _, block, audit, metadata = future.result()
                    block.to_csv(seed_dir / "block_metrics.csv", index=False)
                    audit.to_csv(seed_dir / "selection_audit.csv", index=False)
                    _write_json(seed_dir / "metadata.json", metadata)
                    _write_json(
                        seed_dir / "status.json",
                        {"seed": seed, "status": "complete"},
                    )
                    blocks.append(block)
                    audits.append(audit)
                    logger.info("seed %s complete", seed)
                except Exception as error:
                    failure = {
                        "seed": seed,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    failures.append(failure)
                    _write_json(seed_dir / "status.json", failure)
                    logger.exception("seed %s failed", seed)
        _write_json(output / "failures.json", failures)
        if failures or len(blocks) != len(config["seeds"]):
            raise RuntimeError(
                "one or more parallel Exp39 seeds failed; summary is claim-ineligible"
            )
        block_metrics = pd.concat(blocks, ignore_index=True).sort_values(
            ["seed", "method", "block_id"], kind="stable"
        )
        selection_audit = pd.concat(audits, ignore_index=True).sort_values(
            ["seed", "selection_family", "selection_nll"], kind="stable"
        )
        block_metrics.to_csv(output / "block_metrics.csv", index=False)
        selection_audit.to_csv(output / "selection_audit.csv", index=False)
        seed_metrics, comparisons, tracking, summary = summarize(
            block_metrics, config=config
        )
        seed_metrics.to_csv(output / "seed_metrics.csv", index=False)
        comparisons.to_csv(
            output / "comparisons_and_clamps.csv", index=False
        )
        tracking.to_csv(output / "parameter_tracking.csv", index=False)
        _write_json(output / "summary.json", summary)
        (output / "report.md").write_text(_report(summary), encoding="utf-8")
        _write_json(output / "status.json", {"status": "complete", **summary})
        return summary
    except Exception as error:
        _write_json(
            output / "status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    summary = parallel_execute(
        args.config.resolve(), args.output_dir, workers=args.workers
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
