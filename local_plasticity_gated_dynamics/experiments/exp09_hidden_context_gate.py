"""P2 hidden-HMM context inference and frozen-gate intervention audit."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.p2_protocol import validate_formal_p2_protocol
from src.tasks.hidden_context import (
    HiddenContextConfig,
    generate_hidden_context,
    make_hidden_context_random_tape,
)
from src.training.hidden_context_gate import (
    BASE_GATES,
    HiddenGateCondition,
    ScientificallyInvalidCondition,
    build_hidden_gate_conditions,
    evaluate_gate_prediction,
    fit_hidden_gate,
    intervene_on_prediction,
    split_hidden_context_dataset,
)
from src.utils.artifacts import ExperimentRun


FORMAL_Q = (0.55, 0.70, 0.85, 1.0)
FORMAL_H = (0.01, 0.05, 0.10, 0.20)
FORMAL_SEEDS = tuple(range(30))


def _validate_registered_config(config: dict[str, Any]) -> None:
    if str(config.get("profile")) != "formal":
        return
    validate_formal_p2_protocol(config)


def _condition_groups(
    conditions: list[HiddenGateCondition],
) -> dict[tuple[float, float], list[HiddenGateCondition]]:
    groups: dict[tuple[float, float], list[HiddenGateCondition]] = {}
    for condition in conditions:
        groups.setdefault(
            (condition.cue_reliability, condition.context_hazard), []
        ).append(condition)
    return groups


def run_seed(config: dict[str, Any], seed: int, results_root: str) -> Path:
    """Run all 128 paired cells for one independent seed."""

    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "hidden_context_gate_grid",
        "used_autograd": False,
        "parent_checkpoint": None,
    }
    with ExperimentRun(
        "exp09_hidden_context_gate",
        seed,
        run_config,
        results_root=results_root,
    ) as run:
        try:
            _validate_registered_config(config)
            conditions = build_hidden_gate_conditions(config)
            if len(conditions) != 128:
                raise ValueError(
                    "registered hidden-context grid must contain 128 cells per seed"
                )
            planned = [condition.as_dict() for condition in conditions]
            run.register_conditions(planned)
        except Exception as error:
            run.register_conditions([{"condition": "setup"}])
            run.mark_condition_failure(error, condition="setup")
            return run.path

        try:
            task_options = dict(config["task"])
            tape_config = HiddenContextConfig(
                **task_options,
                cue_reliability=conditions[0].cue_reliability,
                context_hazard=conditions[0].context_hazard,
            )
            random_tape = make_hidden_context_random_tape(tape_config, seed=seed)
        except Exception as error:
            for dimensions in planned:
                run.mark_condition_failure(error, **dimensions)
            return run.path

        emitted: set[str] = set()

        def record_failure(
            condition: HiddenGateCondition,
            error: BaseException,
            *,
            invalid: bool = False,
        ) -> None:
            dimensions = condition.as_dict()
            if condition.name in emitted:
                raise RuntimeError(f"condition emitted twice: {condition.name}")
            if invalid:
                run.mark_condition_invalid(str(error), **dimensions)
            else:
                run.mark_condition_failure(error, **dimensions)
            emitted.add(condition.name)

        def record_success(
            condition: HiddenGateCondition, metrics: dict[str, object]
        ) -> None:
            if condition.name in emitted:
                raise RuntimeError(f"condition emitted twice: {condition.name}")
            run.record(metrics, **condition.as_dict())
            emitted.add(condition.name)

        for (reliability, hazard), cells in _condition_groups(conditions).items():
            try:
                cell_config = replace(
                    tape_config,
                    cue_reliability=reliability,
                    context_hazard=hazard,
                )
                dataset = generate_hidden_context(
                    cell_config,
                    seed=seed,
                    random_tape=random_tape,
                )
                splits = split_hidden_context_dataset(
                    dataset,
                    outer_test_fraction=float(config["outer_test_fraction"]),
                    validation_fraction=float(config["validation_fraction"]),
                    seed=seed,
                )
            except Exception as error:
                for condition in cells:
                    record_failure(condition, error)
                continue

            base_cells = {
                condition.gate_model: condition
                for condition in cells
                if condition.intervention == "none"
            }
            intervention_cells = {
                condition.intervention: condition
                for condition in cells
                if condition.intervention != "none"
            }
            for gate_model in BASE_GATES:
                condition = base_cells[gate_model]
                try:
                    fitted = fit_hidden_gate(
                        gate_model,
                        splits,
                        context_hazard=hazard,
                        cue_reliability=reliability,
                        config=config,
                        seed=seed,
                    )
                except Exception as error:
                    record_failure(condition, error)
                    if gate_model == "md_recurrent_belief":
                        for dependent in intervention_cells.values():
                            record_failure(dependent, error)
                    continue

                try:
                    metrics = evaluate_gate_prediction(
                        fitted,
                        fitted.prediction,
                        splits,
                        condition,
                        config=config,
                        profile=str(config.get("profile", "unspecified")),
                        seed=seed,
                    )
                    record_success(condition, metrics)
                except ScientificallyInvalidCondition as error:
                    record_failure(condition, error, invalid=True)
                except Exception as error:
                    record_failure(condition, error)

                if gate_model != "md_recurrent_belief":
                    continue
                for intervention, dependent in intervention_cells.items():
                    try:
                        altered = intervene_on_prediction(
                            fitted,
                            intervention,
                            config=config,
                            seed=seed,
                        )
                        metrics = evaluate_gate_prediction(
                            fitted,
                            altered,
                            splits,
                            dependent,
                            config=config,
                            profile=str(config.get("profile", "unspecified")),
                            seed=seed,
                        )
                        record_success(dependent, metrics)
                    except ScientificallyInvalidCondition as error:
                        record_failure(dependent, error, invalid=True)
                    except Exception as error:
                        record_failure(dependent, error)

        expected = {condition.name for condition in conditions}
        if emitted != expected:
            missing = sorted(expected - emitted)
            extra = sorted(emitted - expected)
            raise RuntimeError(
                f"planned/emitted grid mismatch; missing={missing}; extra={extra}"
            )
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "P2 hidden-context gate audit",
        "configs/formal/exp09_hidden_context_gate.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
