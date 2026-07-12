"""Evaluate task-specialized, target-free ARC and Sudoku dynamics.

Exp15 is an additive feasibility audit.  It does not rewrite Exp13 and cannot
promote an advantage claim until a matched-compute comparator is registered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (  # noqa: E402
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from experiments.exp13_structured_reasoning import _load_dataset  # noqa: E402
from src.models.task_specialized_reasoners import (  # noqa: E402
    ARCSlowFastProgramReasoner,
    SudokuConstraintDynamics,
)
from src.utils.artifacts import ExperimentRun  # noqa: E402
from src.utils.reproducibility import derive_seed  # noqa: E402


_CONDITIONS = {
    "arc": ("arc_slow_fast_program",),
    "sudoku": ("sudoku_local_no_branch", "sudoku_local_bounded_branch"),
}


def _bootstrap_accuracy(
    exact: np.ndarray,
    groups: tuple[str, ...],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    if (
        exact.ndim != 1
        or not len(exact)
        or len(groups) != len(exact)
        or n_bootstrap < 100
    ):
        raise ValueError("group bootstrap requires data and at least 100 draws")
    unique_groups = tuple(sorted(set(groups)))
    group_values = np.asarray(
        [float(np.mean(exact[np.asarray(groups) == group])) for group in unique_groups]
    )
    rng = np.random.default_rng(seed)
    draws = np.mean(
        group_values[
            rng.integers(
                0,
                len(group_values),
                size=(n_bootstrap, len(group_values)),
            )
        ],
        axis=1,
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(np.mean(group_values)), float(low), float(high)


def _solver(condition: str, config: Mapping[str, Any]):
    model = dict(config.get("model", {}))
    if condition == "arc_slow_fast_program":
        return ARCSlowFastProgramReasoner(
            max_candidates=int(model.get("max_arc_candidates", 96)),
            max_steps=int(model.get("max_reasoning_steps", 8)),
            belief_decay=float(model.get("belief_decay", 0.5)),
            halt_margin=float(model.get("halt_margin", 0.98)),
        )
    if condition == "sudoku_local_no_branch":
        return SudokuConstraintDynamics(
            max_steps=int(model.get("max_propagation_steps", 128)), branch_budget=0
        )
    if condition == "sudoku_local_bounded_branch":
        return SudokuConstraintDynamics(
            max_steps=int(model.get("max_propagation_steps", 128)),
            branch_budget=int(model.get("branch_budget", 256)),
        )
    raise ValueError(f"unknown Exp15 condition {condition!r}")


def run_seed(config: Mapping[str, Any], seed: int, results_root: str | Path) -> Path:
    initialize_seed(seed)
    family = str(config["family"]).strip().lower()
    if family not in _CONDITIONS:
        raise ValueError("Exp15 supports only ARC and Sudoku")
    conditions = tuple(config.get("conditions", _CONDITIONS[family]))
    if not conditions or not set(conditions).issubset(_CONDITIONS[family]):
        raise ValueError("Exp15 conditions do not match the task family")
    run_config = {
        **dict(config),
        "training_algorithm": "target_free_task_specialized_dynamics",
        "used_autograd": False,
        "used_bptt": False,
        "spiking_required": False,
        "reference_scope": "task_design_only_not_bdh_or_hrm_reimplementation",
    }
    with ExperimentRun(
        "exp15_task_specialized_reasoning", seed, run_config, results_root=results_root
    ) as run:
        run.register_conditions(
            [
                {"condition": condition, "task_family": family}
                for condition in conditions
            ]
        )
        try:
            dataset, fixture_only, provenance = _load_dataset(dict(config), run.path)
            test_tasks = dataset.for_split("test")
            if len(test_tasks) < 1:
                raise ValueError("Exp15 requires test tasks")
            (run.path / "source_provenance.json").write_text(
                json.dumps(
                    {
                        **provenance,
                        "fixture_only": fixture_only,
                        "n_test_tasks": len(test_tasks),
                        "test_task_fingerprints": {
                            task.task_id: task.fingerprint for task in test_tasks
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except Exception as error:
            for condition in conditions:
                run.mark_condition_failure(
                    error, condition=condition, task_family=family, stage="dataset"
                )
            return run.path

        for condition in conditions:
            try:
                solver = _solver(condition, config)
                task_rows = []
                for task in test_tasks:
                    result = solver.solve(task)
                    score = dict(dataset.target_store.score(task, result.output))
                    row = {
                        "task_id": task.task_id,
                        "source_group": task.source_group,
                        "public_fingerprint": task.fingerprint,
                        "exact": bool(score.get("exact", False)),
                        "valid_solution": bool(score.get("valid_solution", False)),
                        "state_steps": int(result.state_trace.shape[0]),
                        **dict(result.receipt),
                    }
                    task_rows.append(row)
                    run.record(
                        row,
                        stage="task_test",
                        condition=condition,
                        task_family=family,
                        statistics_unit="task",
                    )
                exact = np.asarray([row["exact"] for row in task_rows], dtype=float)
                source_groups = tuple(str(row["source_group"]) for row in task_rows)
                estimate, ci_low, ci_high = _bootstrap_accuracy(
                    exact,
                    source_groups,
                    n_bootstrap=int(config.get("n_bootstrap", 1000)),
                    seed=derive_seed(seed, "exp15", family, condition, "bootstrap"),
                )
                functional = np.asarray(
                    [
                        row["valid_solution"] if family == "sudoku" else row["exact"]
                        for row in task_rows
                    ],
                    dtype=float,
                )
                functional_estimate, functional_low, functional_high = (
                    _bootstrap_accuracy(
                        functional,
                        source_groups,
                        n_bootstrap=int(config.get("n_bootstrap", 1000)),
                        seed=derive_seed(
                            seed, "exp15", family, condition, "functional_bootstrap"
                        ),
                    )
                )
                source_manifest_verified = family == "sudoku" and provenance.get(
                    "preparation_manifest_status"
                ) in {"complete", "complete_with_exclusions"}
                eligible = (
                    str(config.get("profile")) == "formal"
                    and not fixture_only
                    and provenance.get("license_status") == "verified"
                    and source_manifest_verified
                    and len(test_tasks) >= int(config.get("minimum_test_tasks", 1))
                )
                run.record(
                    {
                        "status": "complete",
                        "n_tasks": len(task_rows),
                        "n_independent_source_groups": len(set(source_groups)),
                        "exact_accuracy": estimate,
                        "exact_accuracy_ci_low": ci_low,
                        "exact_accuracy_ci_high": ci_high,
                        "functional_success_rate": functional_estimate,
                        "functional_success_ci_low": functional_low,
                        "functional_success_ci_high": functional_high,
                        "functional_success_definition": (
                            "valid_constraint_solution"
                            if family == "sudoku"
                            else "all_query_exact"
                        ),
                        "mean_state_steps": float(
                            np.mean([row["state_steps"] for row in task_rows])
                        ),
                        "formal_data_eligible": eligible,
                        "source_manifest_verified": source_manifest_verified,
                        "formal_data_ineligibility_reason": (
                            None
                            if eligible
                            else (
                                "synthetic_fixture"
                                if fixture_only
                                else "arc_tree_manifest_not_yet_verified"
                                if family == "arc"
                                else "formal_source_contract_not_met"
                            )
                        ),
                        "matched_advantage_comparator_registered": False,
                        "core_claim_eligible": False,
                        "claim_conclusion": "inconclusive",
                        "fixture_only": fixture_only,
                        "used_bptt": False,
                        "spiking_required": False,
                    },
                    stage="aggregate",
                    condition=condition,
                    task_family=family,
                    statistics_unit="source_group",
                )
            except Exception as error:
                run.mark_condition_failure(
                    error,
                    condition=condition,
                    task_family=family,
                    stage="evaluation",
                )
        return run.path


def main() -> None:
    args = basic_parser(
        "Task-specialized ARC/Sudoku reasoning audit",
        "configs/smoke/exp15_task_specialized_arc.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
