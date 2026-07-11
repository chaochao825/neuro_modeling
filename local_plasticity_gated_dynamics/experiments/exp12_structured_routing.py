"""Secondary ARC-style routing benchmark on frozen candidate/action tapes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.structured_routing_metrics import (
    ROUTING_CONDITIONS,
    evaluate_structured_routing,
)
from src.data.structured_task_dataset import (
    load_structured_task_tape,
    make_synthetic_structured_tape,
)
from src.utils.artifacts import ExperimentRun


def _load_or_make_tape(config: dict[str, Any], run_path: Path):
    fixture = config.get("synthetic_fixture")
    tape_path = config.get("tape_path")
    if fixture is not None and tape_path is not None:
        raise ValueError("configure exactly one of tape_path or synthetic_fixture")
    if fixture is not None:
        if str(config.get("profile")) != "smoke":
            raise ValueError(
                "synthetic structured tapes are allowed only in smoke runs"
            )
        payload = make_synthetic_structured_tape(**dict(fixture))
        path = run_path / "synthetic_fixture_not_scientific.json"
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return load_structured_task_tape(path), True
    if not isinstance(tape_path, str) or not tape_path:
        raise ValueError("a frozen tape_path is required outside smoke fixtures")
    return load_structured_task_tape(tape_path), False


def run_seed(config: dict[str, Any], seed: int, results_root: str) -> Path:
    """Fit one train-task router and score every complete held-out task."""

    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "train_task_only_shared_candidate_router",
        "used_autograd": False,
        "parent_checkpoint": None,
        "neural_evidence_claim": False,
    }
    with ExperimentRun(
        "exp12_structured_routing",
        seed,
        run_config,
        results_root=results_root,
    ) as run:
        recorded_conditions: set[str] = set()
        planned = [
            {"condition": condition, "routing_condition": condition}
            for condition in ROUTING_CONDITIONS
        ]
        run.register_conditions(planned)
        try:
            dataset, fixture_only = _load_or_make_tape(config, run.path)
            tape_provenance = {
                "schema_version": dataset.schema_version,
                "tape_fingerprint": dataset.tape_fingerprint,
                "frozen_before_evaluation": dataset.frozen_before_evaluation,
                "candidate_generator_commit": dataset.candidate_generator_commit,
                "source_name": (
                    dataset.source_path.name
                    if dataset.source_path is not None
                    else None
                ),
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "router_split": task.router_split,
                        "task_provenance_hash": task.provenance_hash,
                        "candidate_fingerprint": task.candidate_fingerprint,
                        "train_example_ids": list(task.train_example_ids),
                        "test_example_ids": list(task.test_example_ids),
                    }
                    for task in dataset.tasks
                ],
            }
            (run.path / "structured_tape_provenance.json").write_text(
                json.dumps(tape_provenance, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            evaluations = evaluate_structured_routing(
                dataset,
                router_dim=int(config["router_dim"]),
                seed=seed,
                n_bootstrap=int(config["n_bootstrap"]),
                regularization_c=float(config.get("regularization_c", 1.0)),
            )
            task_metrics = pd.concat(
                [evaluation.task_metrics for evaluation in evaluations],
                ignore_index=True,
            )
            task_metrics.insert(0, "seed", seed)
            task_metrics.insert(0, "tape_fingerprint", dataset.tape_fingerprint)
            task_metrics.to_csv(run.path / "task_metrics.csv", index=False)
            for evaluation in evaluations:
                dimensions = {
                    "condition": evaluation.condition,
                    "routing_condition": evaluation.condition,
                }
                metrics = dict(evaluation.summary)
                # These are ExperimentRun dimensions, not duplicate metric keys.
                metrics.pop("condition", None)
                contract = dict(config.get("evidence_contract", {}))
                minimum_test_tasks = int(contract.get("minimum_test_tasks", 100))
                minimum_coverage = float(
                    contract.get("minimum_candidate_coverage", 0.9)
                )
                schema_contract_eligible = (
                    str(config.get("profile", "")) == "formal"
                    and not fixture_only
                    and dataset.frozen_before_evaluation
                    and int(metrics["n_test_tasks"]) >= minimum_test_tasks
                    and float(metrics["candidate_coverage"]) >= minimum_coverage
                )
                metrics.update(
                    profile=str(config.get("profile", "unspecified")),
                    training_algorithm="train_task_only_shared_candidate_router",
                    fixture_only=fixture_only,
                    schema_contract_eligible=schema_contract_eligible,
                    scientific_evidence_eligible=False,
                    scientific_evidence_ineligibility_reason=(
                        "external_feature_extractor_and_candidate_generator_"
                        "provenance_not_independently_recomputed"
                    ),
                    minimum_test_tasks=minimum_test_tasks,
                    minimum_candidate_coverage=minimum_coverage,
                    neural_evidence_claim=False,
                    candidate_tape_source=(
                        "synthetic_smoke_fixture"
                        if fixture_only
                        else str(dataset.source_path)
                    ),
                )
                run.record(metrics, **dimensions)
                recorded_conditions.add(evaluation.condition)
        except Exception as error:
            for dimensions in planned:
                if str(dimensions["condition"]) not in recorded_conditions:
                    run.mark_condition_failure(error, **dimensions)
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "Structured routing benchmark",
        "configs/smoke/exp12_structured_routing.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
