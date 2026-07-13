"""Evaluate task-specialized, target-free ARC and Sudoku dynamics.

Exp15 is an additive feasibility audit.  It does not rewrite Exp13.  ARC uses
a registered flat selector on the identical proposal panel and charged compute
budget; low proposal coverage remains a fail-closed advantage gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import wilcoxon

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
    ARCFlatProgramReasoner,
    ARCSlowFastProgramReasoner,
    SudokuConstraintDynamics,
)
from src.tasks.structured_proposals import generate_arc_proposals  # noqa: E402
from src.utils.artifacts import ExperimentRun  # noqa: E402
from src.utils.reproducibility import derive_seed  # noqa: E402


_CONDITIONS = {
    "arc": ("arc_slow_fast_program", "arc_flat_program_matched"),
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


def _paired_group_comparison(
    candidate_rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    if n_bootstrap < 100:
        raise ValueError("paired group bootstrap requires at least 100 draws")
    candidate = {str(row["task_id"]): row for row in candidate_rows}
    reference = {str(row["task_id"]): row for row in reference_rows}
    if not candidate or set(candidate) != set(reference):
        raise ValueError("paired ARC comparison requires an identical task panel")

    group_differences: dict[str, list[float]] = {}
    candidate_coverage: dict[str, list[float]] = {}
    fingerprints_match = True
    coverage_match = True
    charge_match = True
    for task_id in sorted(candidate):
        left = candidate[task_id]
        right = reference[task_id]
        left_group = str(left["source_group"])
        if left_group != str(right["source_group"]):
            raise ValueError("paired ARC source groups differ across conditions")
        group_differences.setdefault(left_group, []).append(
            float(bool(left["exact"])) - float(bool(right["exact"]))
        )
        candidate_coverage.setdefault(left_group, []).append(
            float(bool(left["candidate_covered"]))
        )
        coverage_match &= bool(left["candidate_covered"]) == bool(
            right["candidate_covered"]
        )
        fingerprints_match &= str(left["candidate_fingerprint"]) == str(
            right["candidate_fingerprint"]
        )
        charge_match &= bool(
            np.isclose(
                float(left["charged_compute_units"]),
                float(right["charged_compute_units"]),
                rtol=0.0,
                atol=1e-12,
            )
        )

    groups = tuple(sorted(group_differences))
    differences = np.asarray(
        [float(np.mean(group_differences[group])) for group in groups]
    )
    coverage = np.asarray(
        [float(np.mean(candidate_coverage[group])) for group in groups]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(groups), size=(n_bootstrap, len(groups)), endpoint=False
    )
    bootstrap = np.mean(differences[indices], axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    nonzero = int(np.count_nonzero(differences))
    p_value = (
        1.0
        if nonzero == 0
        else float(
            wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    )
    return {
        "estimate": float(np.mean(differences)),
        "ci_low": float(low),
        "ci_high": float(high),
        "wilcoxon_p": p_value,
        "wilcoxon_p_holm": p_value,
        "n_independent_source_groups": len(groups),
        "n_nonzero_source_groups": nonzero,
        "candidate_coverage": float(np.mean(coverage)),
        "candidate_fingerprints_matched": fingerprints_match,
        "candidate_coverage_matched": coverage_match,
        "charged_compute_matched": charge_match,
    }


def _solver(condition: str, config: Mapping[str, Any]):
    model = dict(config.get("model", {}))
    if condition == "arc_slow_fast_program":
        return ARCSlowFastProgramReasoner(
            max_candidates=int(model.get("max_arc_candidates", 96)),
            max_steps=int(model.get("max_reasoning_steps", 8)),
            belief_decay=float(model.get("belief_decay", 0.5)),
            halt_margin=float(model.get("halt_margin", 0.98)),
            family_evidence_top_k=int(model.get("family_evidence_top_k", 3)),
            family_belief_gain=float(model.get("family_belief_gain", 1.0)),
        )
    if condition == "arc_flat_program_matched":
        return ARCFlatProgramReasoner(
            max_candidates=int(model.get("max_arc_candidates", 96)),
            max_steps=int(model.get("max_reasoning_steps", 8)),
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

        source_manifest_verified = (
            bool(
                provenance.get("source_manifest_verified")
                and provenance.get("source_acquisition_verified")
            )
            if family == "arc"
            else provenance.get("preparation_manifest_status")
            in {"complete", "complete_with_exclusions"}
        )
        formal_data_eligible = (
            str(config.get("profile")) == "formal"
            and not fixture_only
            and provenance.get("license_status") == "verified"
            and source_manifest_verified
            and provenance.get("test_split_role") in {"ood", "non_ood"}
            and len(test_tasks) >= int(config.get("minimum_test_tasks", 1))
        )
        if formal_data_eligible:
            formal_data_ineligibility_reason = None
        elif fixture_only:
            formal_data_ineligibility_reason = "synthetic_fixture"
        elif not source_manifest_verified:
            formal_data_ineligibility_reason = "source_manifest_not_verified"
        elif provenance.get("license_status") != "verified":
            formal_data_ineligibility_reason = "license_not_verified"
        elif len(test_tasks) < int(config.get("minimum_test_tasks", 1)):
            formal_data_ineligibility_reason = "minimum_test_tasks_not_met"
        else:
            formal_data_ineligibility_reason = "formal_source_contract_not_met"

        comparison_config = config.get("registered_comparison")
        matched_comparator_registered = bool(
            family == "arc"
            and isinstance(comparison_config, Mapping)
            and str(comparison_config.get("name", "")).strip()
            and comparison_config.get("candidate") == "arc_slow_fast_program"
            and comparison_config.get("reference") == "arc_flat_program_matched"
            and {
                "arc_slow_fast_program",
                "arc_flat_program_matched",
            }.issubset(conditions)
        )
        completed_task_rows: dict[str, list[dict[str, object]]] = {}
        for condition in conditions:
            try:
                solver = _solver(condition, config)
                task_rows: list[dict[str, object]] = []
                for task in test_tasks:
                    candidate_covered = False
                    if family == "arc":
                        proposals = generate_arc_proposals(
                            task,
                            max_candidates=int(
                                dict(config.get("model", {})).get(
                                    "max_arc_candidates", 96
                                )
                            ),
                        )
                        result = solver.solve(task, proposals=proposals)
                        if (
                            result.receipt.get("candidate_fingerprint")
                            != proposals.candidate_fingerprint
                        ):
                            raise RuntimeError(
                                "ARC solver receipt differs from the shared proposal panel"
                            )
                        candidate_covered = any(
                            bool(
                                dataset.target_store.score(task, output).get(
                                    "exact", False
                                )
                            )
                            for output in proposals.outputs
                        )
                    else:
                        result = solver.solve(task)
                    score = dict(dataset.target_store.score(task, result.output))
                    row = {
                        "task_id": task.task_id,
                        "source_group": task.source_group,
                        "public_fingerprint": task.fingerprint,
                        "exact": bool(score.get("exact", False)),
                        "valid_solution": bool(score.get("valid_solution", False)),
                        "candidate_covered": candidate_covered,
                        "candidate_coverage_is_oracle_diagnostic": family == "arc",
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
                completed_task_rows[condition] = task_rows
                exact = np.asarray([row["exact"] for row in task_rows], dtype=float)
                source_groups = tuple(str(row["source_group"]) for row in task_rows)
                accuracy_bootstrap_seed = derive_seed(
                    seed, "exp15", family, condition, "bootstrap"
                )
                estimate, ci_low, ci_high = _bootstrap_accuracy(
                    exact,
                    source_groups,
                    n_bootstrap=int(config.get("n_bootstrap", 1000)),
                    seed=accuracy_bootstrap_seed,
                )
                functional = np.asarray(
                    [
                        row["valid_solution"] if family == "sudoku" else row["exact"]
                        for row in task_rows
                    ],
                    dtype=float,
                )
                functional_bootstrap_seed = derive_seed(
                    seed, "exp15", family, condition, "functional_bootstrap"
                )
                functional_estimate, functional_low, functional_high = (
                    _bootstrap_accuracy(
                        functional,
                        source_groups,
                        n_bootstrap=int(config.get("n_bootstrap", 1000)),
                        seed=functional_bootstrap_seed,
                    )
                )
                compute_keys = (
                    "measured_compute_units",
                    "charged_compute_units",
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
                        "accuracy_bootstrap_seed": accuracy_bootstrap_seed,
                        "functional_bootstrap_seed": functional_bootstrap_seed,
                        "n_bootstrap": int(config.get("n_bootstrap", 1000)),
                        "mean_state_steps": float(
                            np.mean([row["state_steps"] for row in task_rows])
                        ),
                        **{
                            f"mean_{key}": float(
                                np.mean([float(row[key]) for row in task_rows])
                            )
                            for key in compute_keys
                            if all(key in row for row in task_rows)
                        },
                        "candidate_coverage": (
                            float(
                                np.mean(
                                    [
                                        bool(row["candidate_covered"])
                                        for row in task_rows
                                    ]
                                )
                            )
                            if family == "arc"
                            else None
                        ),
                        "formal_data_eligible": formal_data_eligible,
                        "source_manifest_verified": source_manifest_verified,
                        "source_acquisition_verified": bool(
                            provenance.get("source_acquisition_verified")
                        ),
                        "formal_data_ineligibility_reason": (
                            formal_data_ineligibility_reason
                        ),
                        "matched_advantage_comparator_registered": (
                            matched_comparator_registered
                        ),
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

        if matched_comparator_registered:
            assert isinstance(comparison_config, Mapping)
            comparison_name = str(comparison_config["name"])
            candidate_name = str(comparison_config["candidate"])
            reference_name = str(comparison_config["reference"])
            try:
                comparison_bootstrap_seed = derive_seed(
                    seed, "exp15", family, comparison_name, "paired_bootstrap"
                )
                comparison = _paired_group_comparison(
                    completed_task_rows[candidate_name],
                    completed_task_rows[reference_name],
                    n_bootstrap=int(config.get("n_bootstrap", 1000)),
                    seed=comparison_bootstrap_seed,
                )
                minimum_coverage = float(
                    comparison_config.get("minimum_candidate_coverage", 0.9)
                )
                alpha = float(comparison_config.get("alpha", 0.05))
                if not 0.0 <= minimum_coverage <= 1.0 or not 0.0 < alpha < 1.0:
                    raise ValueError("invalid registered ARC comparison thresholds")
                coverage_gate_passed = bool(
                    float(comparison["candidate_coverage"]) >= minimum_coverage
                )
                core_claim_eligible = bool(
                    formal_data_eligible
                    and provenance.get("test_split_role") == "ood"
                    and comparison["candidate_fingerprints_matched"]
                    and comparison["candidate_coverage_matched"]
                    and comparison["charged_compute_matched"]
                    and coverage_gate_passed
                )
                if (
                    core_claim_eligible
                    and float(comparison["wilcoxon_p_holm"]) < alpha
                    and float(comparison["ci_low"]) > 0.0
                ):
                    conclusion = "support"
                elif (
                    core_claim_eligible
                    and float(comparison["wilcoxon_p_holm"]) < alpha
                    and float(comparison["ci_high"]) < 0.0
                ):
                    conclusion = "oppose"
                else:
                    conclusion = "inconclusive"
                run.record(
                    {
                        "status": "complete",
                        "comparison": comparison_name,
                        "candidate": candidate_name,
                        "reference": reference_name,
                        "estimand": "candidate_exact_accuracy_minus_reference",
                        **comparison,
                        "multiple_comparison_family": (
                            "exp15_arc_one_registered_comparison"
                        ),
                        "bootstrap_seed": comparison_bootstrap_seed,
                        "n_bootstrap": int(config.get("n_bootstrap", 1000)),
                        "minimum_candidate_coverage": minimum_coverage,
                        "coverage_gate_passed": coverage_gate_passed,
                        "formal_data_eligible": formal_data_eligible,
                        "registered_ood_split": (
                            provenance.get("test_split_role") == "ood"
                        ),
                        "matched_advantage_comparator_registered": True,
                        "core_claim_eligible": core_claim_eligible,
                        "claim_conclusion": conclusion,
                        "used_bptt": False,
                        "spiking_required": False,
                    },
                    stage="comparison",
                    condition=comparison_name,
                    task_family=family,
                    statistics_unit="source_group",
                )
            except Exception as error:
                run.mark_condition_failure(
                    error,
                    condition=comparison_name,
                    task_family=family,
                    stage="comparison",
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
