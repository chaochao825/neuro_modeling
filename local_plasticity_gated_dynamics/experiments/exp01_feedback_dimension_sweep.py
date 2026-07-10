"""Phase-1 feedback dimension/geometry/noise/decay sweep."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    build_phase1_conditions,
    evaluate_phase1_condition,
    initialize_seed,
    load_json_config,
    make_phase1_dataset,
    seed_list,
)
from src.utils.artifacts import ExperimentRun


def run_seed(config: dict, seed: int, results_root: str) -> Path:
    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "local_predictive_fixed_point_iteration",
        "used_autograd": False,
        "parent_checkpoint": None,
    }
    with ExperimentRun(
        "exp01_feedback_dimension_sweep", seed, run_config, results_root=results_root
    ) as run:
        try:
            conditions = build_phase1_conditions(config)
        except Exception as error:
            run.register_conditions([{"condition": "grid_setup"}])
            run.mark_condition_failure(error, condition="grid_setup")
            return run.path
        datasets = {}
        run.register_conditions([condition.as_dict() for condition in conditions])
        for condition in conditions:
            dimensions = condition.as_dict()
            try:
                key = condition.activity_noise_std
                if key not in datasets:
                    datasets[key] = make_phase1_dataset(config, condition, seed)
                metrics, _ = evaluate_phase1_condition(
                    datasets[key], condition, config, seed=seed
                )
                if metrics["status"] == "complete":
                    run.record(metrics, **dimensions)
                else:
                    run.record_failed_condition(metrics, **dimensions)
            except ValueError as error:
                message = str(error)
                if condition.feedback_mode == "orthogonal" and "maximum" in message:
                    run.mark_condition_invalid(message, **dimensions)
                else:
                    run.mark_condition_failure(error, **dimensions)
            except Exception as error:
                run.mark_condition_failure(error, **dimensions)
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "feedback sweep", "configs/formal/exp01_feedback_dimension_sweep.json"
    ).parse_args()
    config = load_json_config(args.config)
    seeds = seed_list(args.seeds or config["seeds"])
    for seed in seeds:
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
