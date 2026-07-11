"""Matched proposal-selection benchmark used by ``exp13``.

This module joins the capability-safe task protocol, reproducible proposal
generators, local fast/slow controllers, and the isolated GRU/BPTT baseline.
The proposal panel is materialized once per task and reused byte-for-byte by
every condition.  The resulting pipeline is an auditable hybrid solver; it is
not a reproduction of HRM or CTM and it is not a proposal-free neural solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.structured_reasoning_metrics import evaluate_predictions
from src.baselines.structured_baseline import SmallGRUBPTTBaseline
from src.data.structured_protocol import PublicTask, StructuredDataset
from src.models.shared_gated_reasoner import HierarchicalCandidateController
from src.models.structured_reasoner import (
    CandidateSet,
    ComputeBudget,
    FitReceipt,
    SolverOutput,
    TrainingCandidateSet,
)
from src.tasks.structured_proposals import (
    FEATURE_NAMES,
    ProposalBatch,
    generate_structured_proposals,
)


STRUCTURED_CONDITIONS = (
    "support_heuristic",
    "flat_local",
    "hierarchical_local",
    "trace_local",
    "gru_bptt",
    "candidate_oracle",
)


@dataclass(frozen=True, slots=True)
class CandidatePanel:
    """One matched, target-free proposal batch per successfully parsed task."""

    candidates: Mapping[str, CandidateSet]
    proposal_batches: Mapping[str, ProposalBatch]
    failures: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class StructuredConditionResult:
    condition: str
    task_metrics: pd.DataFrame
    summary: Mapping[str, Any]
    fit_receipt: FitReceipt | None


def proposal_batch_to_candidate_set(batch: ProposalBatch) -> CandidateSet:
    """Bind each candidate to generator version and whole-panel fingerprint."""

    if not isinstance(batch, ProposalBatch):
        raise TypeError("batch must be a ProposalBatch")
    fingerprint = batch.candidate_fingerprint
    provenance = tuple(
        f"{batch.generator_version}:panel={fingerprint}:candidate={candidate_id}"
        for candidate_id in batch.candidate_ids
    )
    return CandidateSet(
        task_id=batch.task_id,
        family=batch.family,
        candidate_ids=batch.candidate_ids,
        features=batch.features,
        candidate_outputs=batch.outputs,
        candidate_provenance=provenance,
    )


def build_candidate_panel(
    dataset: StructuredDataset,
    *,
    max_arc_candidates: int = 96,
) -> CandidatePanel:
    """Generate each public task once; retain task-level generation failures."""

    candidates: dict[str, CandidateSet] = {}
    batches: dict[str, ProposalBatch] = {}
    failures: dict[str, str] = {}
    for task in dataset.tasks:
        try:
            options = (
                {"max_candidates": int(max_arc_candidates)}
                if task.family == "arc"
                else {}
            )
            batch = generate_structured_proposals(task, **options)
            if batch.task_id != task.task_id or batch.family != task.family:
                raise RuntimeError("proposal batch identity differs from public task")
            batches[task.task_id] = batch
            candidates[task.task_id] = proposal_batch_to_candidate_set(batch)
        except Exception as error:
            failures[task.task_id] = f"{type(error).__name__}: {error}"
    return CandidatePanel(candidates, batches, failures)


def _exact(metrics: Mapping[str, Any]) -> float:
    return float(bool(metrics.get("exact", False)))


def training_candidate_sets(
    dataset: StructuredDataset,
    panel: CandidatePanel,
    *,
    splits: Sequence[str] = ("train",),
) -> tuple[TrainingCandidateSet, ...]:
    """Create labels only after obtaining a non-test training capability."""

    allowed = frozenset(str(split).lower() for split in splits)
    if "test" in allowed:
        raise ValueError("test labels cannot be requested for fitting")
    result: list[TrainingCandidateSet] = []
    for task in dataset.tasks:
        if task.split not in allowed or task.task_id not in panel.candidates:
            continue
        # This call is the explicit capability check.  The target itself is not
        # passed to the reasoner; family scorers emit one correctness bit per
        # already materialized proposal.
        dataset.target_store.training_view(task)
        public = panel.candidates[task.task_id]
        labels = np.asarray(
            [
                _exact(dataset.target_store.score(task, output))
                for output in public.candidate_outputs
            ],
            dtype=float,
        )
        result.append(TrainingCandidateSet(public, labels))
    if not result:
        raise ValueError("no successfully generated non-test tasks are available to fit")
    return tuple(result)


def _support_heuristic_scores(candidates: CandidateSet) -> np.ndarray:
    feature = candidates.features
    # All terms are target-free.  ARC relies on demonstration fit; Maze and
    # Sudoku additionally expose legality/completeness of the proposed output.
    score = (
        4.0 * feature[:, 0]
        + 2.0 * feature[:, 2]
        + feature[:, 4]
        + feature[:, 5]
        + feature[:, 14]
        + feature[:, 18]
        - 0.05 * feature[:, 6]
        - 0.02 * feature[:, 7]
        - 0.1 * feature[:, 15]
    )
    return np.asarray(score, dtype=float)


def _heuristic_select(candidates: CandidateSet) -> tuple[int, np.ndarray]:
    scores = _support_heuristic_scores(candidates)
    return int(np.argmax(scores)), scores


def _oracle_select(
    dataset: StructuredDataset,
    task: PublicTask,
    candidates: CandidateSet,
) -> tuple[int, np.ndarray]:
    scores = np.asarray(
        [
            _exact(dataset.target_store.score(task, output))
            for output in candidates.candidate_outputs
        ],
        dtype=float,
    )
    return int(np.argmax(scores)), scores


def _main_parameter_counts(model: HierarchicalCandidateController) -> tuple[int, int]:
    arrays = (
        model.input_to_fast,
        model.fast_recurrent,
        model.fast_to_slow,
        model.slow_recurrent,
        model.slow_to_control,
        model.control_to_fast,
    )
    fixed = int(sum(array.size for array in arrays))
    trainable = int(0 if model.readout_ is None else model.readout_.size)
    return fixed + trainable, trainable


def _baseline_parameter_count(model: SmallGRUBPTTBaseline) -> int:
    return int(sum(parameter.numel() for parameter in model.network.parameters()))


def _make_reasoner(
    condition: str,
    *,
    feature_dim: int,
    seed: int,
    model_config: Mapping[str, Any],
) -> HierarchicalCandidateController | SmallGRUBPTTBaseline | None:
    if condition in {"support_heuristic", "candidate_oracle"}:
        return None
    if condition == "gru_bptt":
        return SmallGRUBPTTBaseline(
            feature_dim=feature_dim,
            hidden_dim=int(model_config.get("gru_hidden_dim", 16)),
            epochs=int(model_config.get("gru_epochs", 30)),
            learning_rate=float(model_config.get("gru_learning_rate", 1e-2)),
            seed=seed,
        )
    mode = {
        "flat_local": "flat",
        "hierarchical_local": "hierarchical",
        "trace_local": "trace",
    }.get(condition)
    if mode is None:
        raise ValueError(f"unknown structured condition {condition!r}")
    return HierarchicalCandidateController(
        feature_dim=feature_dim,
        fast_dim=int(model_config.get("fast_dim", 24)),
        slow_dim=int(model_config.get("slow_dim", 12)),
        control_dim=int(model_config.get("control_dim", 4)),
        mode=mode,
        cycles=int(model_config.get("cycles", 3)),
        fast_steps_per_cycle=int(model_config.get("fast_steps_per_cycle", 2)),
        trace_pairs=int(model_config.get("trace_pairs", 8)),
        trace_decay=float(model_config.get("trace_decay", 0.9)),
        ridge=float(model_config.get("ridge", 1e-3)),
        seed=seed,
    )


def _candidate_coverage(
    dataset: StructuredDataset,
    task: PublicTask,
    candidates: CandidateSet,
) -> bool:
    return any(
        _exact(dataset.target_store.score(task, output)) > 0.0
        for output in candidates.candidate_outputs
    )


def run_structured_condition(
    dataset: StructuredDataset,
    panel: CandidatePanel,
    *,
    condition: str,
    seed: int,
    model_config: Mapping[str, Any],
    n_bootstrap: int,
    fit_splits: Sequence[str] = ("train",),
) -> StructuredConditionResult:
    """Fit and evaluate one selector on the exact same materialized panel."""

    if condition not in STRUCTURED_CONDITIONS:
        raise ValueError(f"condition must be one of {STRUCTURED_CONDITIONS}")
    families = dataset.families
    if len(families) != 1:
        raise ValueError("one exp13 condition must contain exactly one task family")
    family = next(iter(families))
    feature_dim = len(FEATURE_NAMES)
    reasoner = _make_reasoner(
        condition,
        feature_dim=feature_dim,
        seed=int(seed),
        model_config=model_config,
    )
    fit_receipt: FitReceipt | None = None
    if reasoner is not None:
        training = training_candidate_sets(dataset, panel, splits=fit_splits)
        fit_receipt = reasoner.fit(training)

    max_candidates = max(
        (candidate.n_candidates for candidate in panel.candidates.values()),
        default=1,
    )
    max_internal_steps = int(
        model_config.get(
            "max_internal_steps",
            max(
                max_candidates,
                getattr(reasoner, "required_internal_steps", 1),
            ),
        )
    )
    predictions: dict[str, Any] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    for task in dataset.for_split("test"):
        public = panel.candidates.get(task.task_id)
        batch = panel.proposal_batches.get(task.task_id)
        if public is None or batch is None:
            diagnostic_rows.append(
                {
                    "task_id": task.task_id,
                    "selector_status": "proposal_generation_failed",
                    "proposal_failure": panel.failures.get(task.task_id),
                }
            )
            continue
        solver_output: SolverOutput | None = None
        if condition == "support_heuristic":
            selected, scores = _heuristic_select(public)
        elif condition == "candidate_oracle":
            selected, scores = _oracle_select(dataset, task, public)
        else:
            if reasoner is None:
                raise RuntimeError("reasoner condition was not instantiated")
            solver_output = reasoner.solve(
                public,
                ComputeBudget(
                    max_candidate_evaluations=max_candidates,
                    max_internal_steps=max_internal_steps,
                ),
            )
            selected, scores = solver_output.selected_index, solver_output.scores
        predictions[task.task_id] = public.candidate_outputs[selected]
        receipt = None if solver_output is None else solver_output.receipt
        diagnostic_rows.append(
            {
                "task_id": task.task_id,
                "selector_status": "complete",
                "proposal_failure": None,
                "candidate_fingerprint": batch.candidate_fingerprint,
                "n_candidates": public.n_candidates,
                "candidate_covered": _candidate_coverage(dataset, task, public),
                "selected_index": selected,
                "selected_candidate_id": public.candidate_ids[selected],
                "selected_score": float(scores[selected]),
                "candidate_generation_compute": batch.matched_compute_budget,
                "selected_candidate_compute": float(batch.compute_costs[selected]),
                "candidate_evaluations": (
                    public.n_candidates
                    if receipt is None
                    else receipt.candidate_evaluations
                ),
                "measured_internal_steps": 0 if receipt is None else receipt.internal_steps,
                "charged_candidate_evaluations": max_candidates,
                "charged_internal_steps": max_internal_steps,
                "compute_matched_by_padding": True,
                "state_trace_dimension": (
                    0 if solver_output is None else solver_output.trace.shape[1]
                ),
                "bilinear_trace_dimension": (
                    0
                    if solver_output is None
                    else solver_output.bilinear_trace.shape[1]
                ),
                "bilinear_trace_energy": (
                    0.0
                    if solver_output is None
                    else float(np.sum(np.square(solver_output.bilinear_trace)))
                ),
            }
        )

    evaluation = evaluate_predictions(
        dataset,
        predictions,
        family=family,
        split="test",
        bootstrap_seed=int(seed),
        n_bootstrap=int(n_bootstrap),
    )
    diagnostics = pd.DataFrame(diagnostic_rows)
    task_metrics = evaluation.task_metrics.merge(
        diagnostics, on="task_id", how="left", validate="one_to_one"
    )
    task_metrics["condition"] = condition
    task_metrics["seed"] = int(seed)
    summary = dict(evaluation.summary)
    completed = task_metrics["selector_status"].fillna("") == "complete"
    covered = task_metrics["candidate_covered"].fillna(False).astype(bool)
    parameter_count = 0
    trainable_parameter_count = 0
    control_rank = 0
    control_dim = 0
    if isinstance(reasoner, HierarchicalCandidateController):
        parameter_count, trainable_parameter_count = _main_parameter_counts(reasoner)
        if reasoner.mode != "flat":
            control_rank = int(np.linalg.matrix_rank(reasoner.control_operator))
            control_dim = int(reasoner.control_dim)
    elif isinstance(reasoner, SmallGRUBPTTBaseline):
        parameter_count = trainable_parameter_count = _baseline_parameter_count(reasoner)
    summary.update(
        condition=condition,
        proposal_generator="structured_proposals_v1",
        proposal_pipeline_recomputed=True,
        candidate_set_matched=True,
        candidate_coverage=float(covered.mean()),
        covered_task_accuracy=(
            float(task_metrics.loc[covered, "exact"].astype(float).mean())
            if covered.any()
            else float("nan")
        ),
        proposal_generation_failure_rate=float((~completed).mean()),
        charged_candidate_evaluations=max_candidates,
        charged_internal_steps=max_internal_steps,
        compute_matched_by_padding=True,
        measured_compute_matched=False,
        efficiency_claim_eligible=False,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        used_bptt=bool(False if reasoner is None else reasoner.used_bptt),
        fit_task_count=(0 if fit_receipt is None else len(fit_receipt.task_ids)),
        fit_task_ids=([] if fit_receipt is None else list(fit_receipt.task_ids)),
        fit_task_balanced=(None if fit_receipt is None else fit_receipt.task_balanced),
        control_dim=control_dim,
        control_operator_rank=control_rank,
        spiking_model=False,
        query_targets_exposed_to_solver=False,
        selection_accessed_query_target=condition == "candidate_oracle",
        official_hrm_reproduction=False,
        official_ctm_reproduction=False,
        interpretation_scope="hybrid_structured_proposal_selection_only",
        neural_evidence_claim=False,
        biological_mechanism_claim_eligible=False,
    )
    return StructuredConditionResult(
        condition=condition,
        task_metrics=task_metrics,
        summary=summary,
        fit_receipt=fit_receipt,
    )


def assert_matched_candidate_panel(results: Sequence[StructuredConditionResult]) -> None:
    """Verify candidate identity and charged budgets across every condition."""

    if not results:
        raise ValueError("results must not be empty")
    columns = [
        "task_id",
        "candidate_fingerprint",
        "n_candidates",
        "candidate_generation_compute",
        "charged_candidate_evaluations",
        "charged_internal_steps",
    ]
    reference: pd.DataFrame | None = None
    for result in results:
        current = result.task_metrics.reindex(columns=columns).sort_values("task_id")
        current = current.reset_index(drop=True)
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise ValueError(
                f"condition {result.condition!r} did not use the matched candidate panel"
            )


def fit_receipt_dict(receipt: FitReceipt | None) -> Mapping[str, Any] | None:
    """JSON-ready fit provenance for experiment artifacts."""

    return None if receipt is None else asdict(receipt)
