"""Multi-session IBL hidden-belief conditional Poisson dynamics audit.

The primary model is a continuous/rate abstraction.  It uses no spiking
mechanism beyond the observed spike counts and no BPTT.  The endpoint is an
exact one-step conditional Poisson likelihood, not a latent Poisson LDS
marginal likelihood.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    PROJECT_ROOT,
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.ibl_neural_metrics import (
    IBLNeuralComparison,
    compare_count_families,
)
from src.data.ibl_multisession import (
    ChronologicalBlockSplit,
    PreparedIBLNeuralSession,
    chronological_outer_inner_splits,
)
from src.data.ibl_neural_cache import load_compact_neural_cohort
from src.data.ibl_neural_panel import (
    MACRO_REGION_MAPPING_SCHEMA,
    MACRO_REGION_SOURCE_ONTOLOGY_SHA256,
    MACRO_REGION_SOURCE_PROVENANCE_SHA256,
    IBLNeuralPanelInput,
    build_model_session,
    load_allen_macro_region_mapping,
    prepare_neural_panel_input,
    union_region_anchors,
)
from src.models.hierarchical_count_dynamics import (
    HierarchicalCountDynamics,
    HierarchicalCountScore,
    NeuralCountSession,
    TrialBlockSplit,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


FAMILIES = ("common", "shared", "full")


@dataclass(frozen=True, slots=True)
class NestedCandidateRecord:
    latent_dim: int
    ridge: float
    status: str
    animal_mean_validation_nll: float
    error_type: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PanelEvaluation:
    view: str
    panel: str
    selected_latent_dim: int
    selected_ridge: float
    common_regions: tuple[str, ...]
    minimum_region_sessions: int
    region_session_coverage: tuple[Mapping[str, object], ...]
    complete_session_ids: tuple[str, ...]
    retained_session_ids: tuple[str, ...]
    macro_region_mapping_receipt: Mapping[str, object]
    nested_candidates: tuple[NestedCandidateRecord, ...]
    scores: Mapping[str, HierarchicalCountScore]
    comparison: IBLNeuralComparison
    comparison_preprocessing_sha256: str
    session_receipts: tuple[Mapping[str, object], ...]
    causal_timing_eligible: bool
    nested_selection_objective: str = (
        "mean_animal_validation_nll_across_common_shared_full"
    )
    likelihood_kind: str = "one_step_conditional_poisson"
    full_latent_lds: bool = False
    region_anchor_policy: str = "fixed_region_order_union"
    region_imputation_strategy: str = "pooled_training_fold_region_mean"


def _animal_mean_nll(score: HierarchicalCountScore) -> float:
    by_animal: dict[str, list[float]] = {}
    for record in score.per_session:
        by_animal.setdefault(record.animal_id, []).append(record.nll_per_count)
    return float(np.mean([np.mean(by_animal[animal]) for animal in sorted(by_animal)]))


def _preprocessing_fingerprint(model: HierarchicalCountDynamics) -> str:
    arrays = (
        model.scaler_mean_,
        model.scaler_scale_,
        model.pca_mean_,
        model.pca_components_,
        model.region_train_observation_counts_,
    ) + tuple(
        model.observation_matrices_[key] for key in sorted(model.observation_matrices_)
    )
    if any(array is None for array in arrays):
        raise RuntimeError("model preprocessing is incomplete")
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    digest.update(model.region_imputation_strategy.encode("ascii"))
    for session_id in sorted(model.session_region_presence_):
        digest.update(session_id.encode("utf-8"))
        digest.update(
            np.asarray(
                model.session_region_presence_[session_id], dtype=np.uint8
            ).tobytes()
        )
    return digest.hexdigest()


def panel_claim_scope(view: str, panel: str) -> str:
    return (
        "registered_primary"
        if view == "stimulus_pre" and panel == "primary_past_safe"
        else "sensitivity_only"
    )


def _build_sessions(
    panels: Sequence[IBLNeuralPanelInput],
    splits: Mapping[str, ChronologicalBlockSplit],
    *,
    common_regions: Sequence[str],
    config: Mapping[str, Any],
    seed: int,
) -> tuple[
    tuple[NeuralCountSession, ...],
    dict[str, TrialBlockSplit],
    tuple[Mapping[str, object], ...],
]:
    built = []
    receipts = []
    for panel in panels:
        item = build_model_session(
            panel,
            splits[panel.session_id],
            common_regions=common_regions,
            max_units_per_region=int(config["max_units_per_region"]),
            min_units_per_region=int(config["min_units_per_region"]),
            hmm_options=dict(config["learned_hmm"]),
            seed=derive_seed(seed, "exp14", panel.session_id, "belief"),
        )
        built.append(item)
        receipts.append(
            {
                "session_id": panel.session_id,
                "animal_id": panel.animal_id,
                "split_fingerprint": item.split.split_fingerprint,
                "belief_checkpoint_sha256": item.session.belief_receipt.checkpoint_sha256,
                "belief_trajectory_sha256": item.session.belief_receipt.belief_sha256,
                "complete_case_mask_sha256": item.complete_case_receipt.mask_sha256,
                "n_complete_trials": len(item.complete_case_receipt.kept_trial_ids),
                "n_excluded_trials": len(item.complete_case_receipt.excluded_trial_ids),
                "selected_unit_ids": item.selected_unit_ids,
                "anchor_regions_present": item.present_anchor_regions,
                "anchor_regions_missing": item.missing_anchor_regions,
                "n_anchor_regions_present": len(item.present_anchor_regions),
                "n_anchor_regions_missing": len(item.missing_anchor_regions),
                "hmm_fit_converged": item.hmm_checkpoint["converged"],
                "hmm_state_identifiable": item.hmm_checkpoint["identifiable"],
            }
        )
    model_splits = {item.session.session_id: item.split for item in built}
    return tuple(item.session for item in built), model_splits, tuple(receipts)


def _fit_score(
    family: str,
    sessions: Sequence[NeuralCountSession],
    splits: Mapping[str, TrialBlockSplit],
    *,
    common_regions: Sequence[str],
    latent_dim: int,
    ridge: float,
    seed: int,
) -> tuple[HierarchicalCountDynamics, HierarchicalCountScore]:
    model = HierarchicalCountDynamics(
        family,
        common_regions=common_regions,
        latent_dim=latent_dim,
        ridge=ridge,
        seed=seed,
    ).fit(sessions, splits)
    return model, model.score(sessions, splits)


def evaluate_prepared_panel(
    prepared_sessions: Sequence[PreparedIBLNeuralSession],
    *,
    view: str,
    panel: str,
    config: Mapping[str, Any],
    seed: int,
) -> PanelEvaluation:
    """Run nested selection and a paired common/shared/full outer test."""

    mapping_path = Path(str(config["macro_region_mapping_path"]))
    if not mapping_path.is_absolute():
        mapping_path = PROJECT_ROOT / mapping_path
    formal_mapping_manifest_sha256 = str(
        config["macro_region_mapping_formal_compact_manifest_sha256"]
    )
    source_ontology_path = None
    formal_profile = (
        str(config.get("profile")) == "formal"
        and str(config.get("data_mode")) == "frozen_compact_cache"
    )
    if formal_profile:
        if formal_mapping_manifest_sha256 != str(
            config["expected_compact_manifest_sha256"]
        ):
            raise ValueError(
                "macro-region artifact is not bound to the formal compact manifest"
            )
        compact_manifest = Path(str(config["compact_cache_manifest"]))
        if not compact_manifest.is_absolute():
            compact_manifest = PROJECT_ROOT / compact_manifest
        source_ontology_path = (
            compact_manifest.parent / "provenance" / "iblatlas_allen_structure_tree.csv"
        )
    macro_mapping = load_allen_macro_region_mapping(
        mapping_path,
        expected_sha256=str(config["expected_macro_region_mapping_sha256"]),
        expected_compact_manifest_sha256=formal_mapping_manifest_sha256,
        source_ontology_path=source_ontology_path,
    )
    if (
        str(config["expected_macro_region_mapping_schema"])
        != MACRO_REGION_MAPPING_SCHEMA
        or str(config["expected_macro_region_source_ontology_sha256"])
        != MACRO_REGION_SOURCE_ONTOLOGY_SHA256
        or str(config["expected_macro_region_source_provenance_sha256"])
        != MACRO_REGION_SOURCE_PROVENANCE_SHA256
    ):
        raise ValueError("macro-region config does not bind the reviewed mapping")
    all_acronyms = tuple(
        str(acronym)
        for session in prepared_sessions
        for acronym in session.regions.tolist()
    )
    macro_mapping.validate_acronym_scope(
        all_acronyms, require_exact_formal_scope=formal_profile
    )
    panels = tuple(
        prepare_neural_panel_input(
            session,
            view=view,
            panel=panel,
            minimum_trials=int(config["minimum_trials"]),
            minimum_blocks=int(config["minimum_blocks"]),
            macro_region_mapping=macro_mapping,
        )
        for session in prepared_sessions
    )
    if len({item.session_id for item in panels}) != len(panels):
        raise ValueError("prepared session IDs must be unique")
    anchor_audit = union_region_anchors(
        panels,
        min_units_per_region=int(config["min_units_per_region"]),
        minimum_region_sessions=int(config["minimum_region_sessions"]),
    )
    common_regions = anchor_audit.regions
    outer_splits = {}
    inner_splits = {}
    for item in panels:
        outer, inner = chronological_outer_inner_splits(
            item.trial_ids,
            item.block_ids,
            outer_test_fraction=float(config["outer_test_fraction"]),
            inner_validation_fraction=float(config["inner_validation_fraction"]),
        )
        outer_splits[item.session_id] = outer
        inner_splits[item.session_id] = inner

    inner_sessions, inner_model_splits, _ = _build_sessions(
        panels,
        inner_splits,
        common_regions=common_regions,
        config=config,
        seed=derive_seed(seed, "exp14", view, panel, "inner"),
    )
    expected_session_ids = tuple(item.session_id for item in panels)
    if tuple(item.session_id for item in inner_sessions) != expected_session_ids:
        raise RuntimeError("inner representation did not retain every complete session")
    candidate_records = []
    successful: list[tuple[float, int, float]] = []
    latent_dims = sorted(
        {
            int(value)
            for value in config["latent_dims"]
            if 1 <= int(value) <= len(common_regions)
        }
    )
    if not latent_dims:
        raise ValueError("no configured latent dimension fits common region anchors")
    for latent_dim in latent_dims:
        for ridge in sorted({float(value) for value in config["ridges"]}):
            try:
                family_objectives = []
                for family in FAMILIES:
                    _, score = _fit_score(
                        family,
                        inner_sessions,
                        inner_model_splits,
                        common_regions=common_regions,
                        latent_dim=latent_dim,
                        ridge=ridge,
                        seed=derive_seed(
                            seed,
                            "exp14",
                            "inner_candidate",
                            latent_dim,
                            ridge,
                            family,
                        ),
                    )
                    family_objectives.append(_animal_mean_nll(score))
                objective = float(np.mean(family_objectives))
                successful.append((objective, latent_dim, ridge))
                candidate_records.append(
                    NestedCandidateRecord(latent_dim, ridge, "complete", objective)
                )
            except Exception as error:
                candidate_records.append(
                    NestedCandidateRecord(
                        latent_dim,
                        ridge,
                        "failed",
                        float("nan"),
                        type(error).__name__,
                        str(error),
                    )
                )
    if not successful:
        raise RuntimeError("all nested latent/ridge candidates failed")
    _, selected_dim, selected_ridge = min(successful)

    outer_sessions, outer_model_splits, receipts = _build_sessions(
        panels,
        outer_splits,
        common_regions=common_regions,
        config=config,
        seed=derive_seed(seed, "exp14", view, panel, "outer"),
    )
    retained_session_ids = tuple(item.session_id for item in outer_sessions)
    if retained_session_ids != expected_session_ids:
        raise RuntimeError("outer representation did not retain every complete session")
    models = {}
    scores = {}
    for family in FAMILIES:
        model, score = _fit_score(
            family,
            outer_sessions,
            outer_model_splits,
            common_regions=common_regions,
            latent_dim=selected_dim,
            ridge=selected_ridge,
            seed=derive_seed(seed, "exp14", view, panel, family),
        )
        models[family], scores[family] = model, score
    fingerprints = {
        family: _preprocessing_fingerprint(model) for family, model in models.items()
    }
    if len(set(fingerprints.values())) != 1:
        raise RuntimeError(
            "paired model families did not share preprocessing/observation parameters"
        )
    comparison = compare_count_families(
        scores,
        planned_sessions=int(config["planned_sessions"]),
        planned_animals=int(config["planned_animals"]),
        n_bootstrap=int(config["n_bootstrap"]),
        seed=derive_seed(seed, "exp14", view, panel, "bootstrap"),
    )
    return PanelEvaluation(
        view=view,
        panel=panel,
        selected_latent_dim=selected_dim,
        selected_ridge=selected_ridge,
        common_regions=common_regions,
        minimum_region_sessions=anchor_audit.minimum_region_sessions,
        region_session_coverage=anchor_audit.coverage_records(),
        complete_session_ids=expected_session_ids,
        retained_session_ids=retained_session_ids,
        macro_region_mapping_receipt=macro_mapping.receipt(),
        nested_candidates=tuple(candidate_records),
        scores=scores,
        comparison=comparison,
        comparison_preprocessing_sha256=next(iter(fingerprints.values())),
        session_receipts=receipts,
        causal_timing_eligible=all(item.causal_timing_eligible for item in panels),
    )


def _synthetic_prepared_sessions(config: Mapping[str, Any], seed: int):
    """Generate non-evidence smoke data; never used by the formal profile."""

    from src.data.ibl_multisession import PreparedIBLNeuralSession

    rng = np.random.default_rng(seed)
    result = []
    n_sessions = int(config.get("synthetic_sessions", 6))
    n_trials = int(config.get("synthetic_trials", 80))
    n_time = int(config.get("synthetic_time_bins", 6))
    block_size = 10
    regions = np.asarray(["MOs1", "VISp4", "VAL", "LP", "CP", "ACB"])
    for session_index in range(n_sessions):
        blocks = np.arange(n_trials) // block_size
        probability = np.where(blocks % 2 == 0, 0.8, 0.2)
        stimulus_side = rng.binomial(1, probability)
        table = pd.DataFrame(
            {
                "stimulus": np.where(stimulus_side, 1.0, -1.0),
                "stimulus_side": stimulus_side,
                "choice": np.where(rng.random(n_trials) > 0.5, 1, -1),
                "reward": np.where(rng.random(n_trials) > 0.3, 1, -1),
                "reaction_time": rng.uniform(0.2, 0.8, n_trials),
                "wheel": rng.uniform(0.0, 2.0, n_trials),
                "motion_energy_proxy": rng.uniform(0.0, 1.0, n_trials),
                "probability_left": probability,
                "block_id": blocks,
            }
        )
        counts = np.empty((n_trials, n_time, len(regions)), dtype=np.int64)
        for trial in range(n_trials):
            state = 1.0 if probability[trial] > 0.5 else -1.0
            for time in range(n_time):
                slope = (
                    state
                    * (time - (n_time - 1) / 2)
                    * np.linspace(-0.1, 0.1, len(regions))
                )
                counts[trial, time] = rng.poisson(np.exp(1.8 + slope))
        result.append(
            PreparedIBLNeuralSession(
                eid=f"synthetic-{session_index}",
                animal_id=f"synthetic-animal-{session_index // 2}",
                count_views={"stimulus_pre": counts, "movement_pre": counts + 1},
                valid_masks={
                    "stimulus_pre": np.ones(n_trials, dtype=bool),
                    "movement_pre": np.ones(n_trials, dtype=bool),
                },
                time_axes={
                    "stimulus_pre": np.linspace(-0.5, -0.1, n_time),
                    "movement_pre": np.linspace(-0.5, -0.1, n_time),
                },
                regions=regions,
                unit_ids=np.asarray(
                    [
                        f"synthetic-{session_index}:u{index}"
                        for index in range(len(regions))
                    ]
                ),
                view_trial_tables={
                    "stimulus_pre": table,
                    "movement_pre": table.assign(
                        motion_energy_proxy=table["motion_energy_proxy"] + 0.1
                    ),
                },
                current_trial_ids=np.arange(n_trials),
            )
        )
    return tuple(result)


def _record_evaluation(
    run: ExperimentRun,
    result: PanelEvaluation,
    *,
    session_provenance: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    claim_scope = panel_claim_scope(result.view, result.panel)
    all_sessions_retained = result.retained_session_ids == result.complete_session_ids
    if not all_sessions_retained:
        raise RuntimeError("evaluation receipt lost one or more complete sessions")
    anchor_metrics = {
        "region_anchor_policy": result.region_anchor_policy,
        "region_imputation_strategy": result.region_imputation_strategy,
        "minimum_region_sessions": result.minimum_region_sessions,
        "region_session_coverage": result.region_session_coverage,
        "n_complete_sessions_input": len(result.complete_session_ids),
        "n_sessions_retained": len(result.retained_session_ids),
        "complete_session_ids": result.complete_session_ids,
        "retained_session_ids": result.retained_session_ids,
        "all_complete_sessions_retained": all_sessions_retained,
        **dict(result.macro_region_mapping_receipt),
    }
    for candidate in result.nested_candidates:
        run.record(
            asdict(candidate),
            stage="nested_selection",
            view=result.view,
            panel=result.panel,
        )
    receipts = {str(item["session_id"]): item for item in result.session_receipts}
    for family, score in result.scores.items():
        for metrics in score.per_session:
            metric_values = asdict(metrics)
            metric_values.pop("session_id")
            metric_values.pop("animal_id")
            receipt_values = dict(receipts[metrics.session_id])
            receipt_values.pop("session_id")
            receipt_values.pop("animal_id")
            provenance_values = (
                {}
                if session_provenance is None
                else dict(session_provenance.get(metrics.session_id, {}))
            )
            run.record(
                {
                    **metric_values,
                    "status": "complete",
                    "selected_latent_dim": result.selected_latent_dim,
                    "selected_ridge": result.selected_ridge,
                    "common_regions": result.common_regions,
                    **anchor_metrics,
                    "parameter_count": score.parameter_count,
                    "likelihood_kind": score.likelihood_kind,
                    "full_latent_lds": score.full_latent_lds,
                    "preprocessing_fit_train_only": True,
                    "hidden_context_inference": True,
                    "test_context_observed": False,
                    "condition_schedule_used_for_split_only": True,
                    "nuisance_as_log_rate_controls": True,
                    "counts_residualized_before_poisson": False,
                    "claim_scope": claim_scope,
                    "causal_timing_eligible": result.causal_timing_eligible,
                    "comparison_preprocessing_sha256": result.comparison_preprocessing_sha256,
                    **receipt_values,
                    **provenance_values,
                },
                stage="outer_test",
                view=result.view,
                panel=result.panel,
                model_family=family,
                session_id=metrics.session_id,
                animal_id=metrics.animal_id,
                aggregation_level="session",
                statistics_unit="session_nested_within_animal",
            )
    conclusion_field = (
        {"core_conclusion": result.comparison.conclusion}
        if claim_scope == "registered_primary"
        else {"sensitivity_conclusion": result.comparison.conclusion}
    )
    run.record(
        {
            "status": "complete",
            "selected_latent_dim": result.selected_latent_dim,
            "selected_ridge": result.selected_ridge,
            "common_regions": result.common_regions,
            **anchor_metrics,
            "nested_selection_objective": result.nested_selection_objective,
            "comparison": asdict(result.comparison),
            "claim_scope": claim_scope,
            "causal_timing_eligible": result.causal_timing_eligible,
            "core_claim_eligible": claim_scope == "registered_primary",
            **conclusion_field,
            "likelihood_kind": result.likelihood_kind,
            "full_latent_lds": result.full_latent_lds,
        },
        stage="animal_session_comparison",
        view=result.view,
        panel=result.panel,
        aggregation_level="animal_with_session_nested",
    )


def run_seed(
    config: Mapping[str, Any],
    seed: int,
    results_root: str | Path,
    *,
    prepared_sessions: Sequence[PreparedIBLNeuralSession] | None = None,
) -> Path:
    initialize_seed(seed)
    profile = str(config.get("profile", "unspecified"))
    data_mode = str(config.get("data_mode", "frozen_compact_cache"))
    run_config = {
        **dict(config),
        "training_algorithm": "train_only_nested_hidden_belief_count_dynamics",
        "used_autograd": False,
        "parent_checkpoint": None,
        "spiking_mechanism_required": False,
        "hrm_ctm_scope": "continuous_inspiration_not_reproduction",
    }
    with ExperimentRun(
        "exp14_ibl_multisession_neural", seed, run_config, results_root=results_root
    ) as run:
        if profile == "formal" and prepared_sessions is not None:
            error = RuntimeError(
                "formal exp14 rejects direct prepared_sessions injection; "
                "a verified CompactNeuralCohort capability is required"
            )
            run.register_conditions(
                [
                    {
                        "view": "discovery",
                        "panel": "capability_validation",
                        "model_family": family,
                    }
                    for family in FAMILIES
                ]
            )
            for family in FAMILIES:
                run.mark_condition_failure(
                    error,
                    view="discovery",
                    panel="capability_validation",
                    model_family=family,
                    aggregation_level="session",
                )
            return run.path
        compact_cohort = None
        if prepared_sessions is None:
            if data_mode == "synthetic_smoke" and profile == "smoke":
                prepared_sessions = _synthetic_prepared_sessions(config, seed)
            elif data_mode == "frozen_compact_cache" and profile == "formal":
                try:
                    manifest = Path(str(config["compact_cache_manifest"]))
                    if not manifest.is_absolute():
                        manifest = PROJECT_ROOT / manifest
                    compact_cohort = load_compact_neural_cohort(
                        manifest,
                        expected_source_manifest_sha256=str(
                            config["expected_source_manifest_sha256"]
                        ),
                        expected_acquisition_bundle_sha256=str(
                            config["expected_acquisition_bundle_sha256"]
                        ),
                        expected_bwm_repository_commit=str(
                            config["expected_bwm_repository_commit"]
                        ),
                        expected_compact_manifest_sha256=str(
                            config["expected_compact_manifest_sha256"]
                        ),
                        expected_compact_bundle_sha256=str(
                            config["expected_compact_bundle_sha256"]
                        ),
                        expected_sessions=int(config["planned_sessions"]),
                        minimum_animals=int(config["planned_animals"]),
                    )
                    prepared_sessions = compact_cohort.sessions
                except Exception as error:
                    cache_error = RuntimeError(
                        "formal exp14 frozen compact neural cache validation failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    run.register_conditions(
                        [
                            {
                                "view": "discovery",
                                "panel": "frozen_cache",
                                "model_family": family,
                            }
                            for family in FAMILIES
                        ]
                    )
                    for family in FAMILIES:
                        run.mark_condition_failure(
                            cache_error,
                            view="discovery",
                            panel="frozen_cache",
                            model_family=family,
                            aggregation_level="session",
                        )
                    return run.path
            else:
                error = RuntimeError(
                    "formal exp14 requires the reviewed frozen compact neural cache; "
                    "synthetic fallback is forbidden"
                )
                run.register_conditions(
                    [
                        {
                            "view": "discovery",
                            "panel": "frozen_cache",
                            "model_family": family,
                        }
                        for family in FAMILIES
                    ]
                )
                for family in FAMILIES:
                    run.mark_condition_failure(
                        error,
                        view="discovery",
                        panel="frozen_cache",
                        model_family=family,
                        aggregation_level="session",
                    )
                return run.path
        views = tuple(config.get("views", ("stimulus_pre",)))
        panels = tuple(config.get("panels", ("primary_past_safe",)))
        if compact_cohort is None:
            planned_session_ids = tuple(session.eid for session in prepared_sessions)
        else:
            planned_session_ids = tuple(
                item.eid for item in compact_cohort.dispositions
            )
        planned = [
            {
                "session_id": session_id,
                "view": view,
                "panel": panel,
                "model_family": family,
            }
            for session_id in planned_session_ids
            for view in views
            for panel in panels
            for family in FAMILIES
        ]
        run.register_conditions(planned)
        provenance: dict[str, dict[str, object]] = {}
        if compact_cohort is not None:
            for disposition in compact_cohort.dispositions:
                provenance[disposition.eid] = {
                    "source_manifest_sha256": disposition.source_manifest_sha256,
                    "acquisition_bundle_sha256": disposition.acquisition_bundle_sha256,
                    "bwm_repository_commit": disposition.bwm_repository_commit,
                    "spike_sorting_revision": disposition.spike_sorting_revision,
                    "unit_qc_threshold": disposition.unit_qc_threshold,
                    "unit_qc_applied": disposition.unit_qc_applied,
                    "acquisition_validation_status": disposition.acquisition_validation_status,
                    "unit_qc_method": disposition.unit_qc_method,
                    "unit_qc_equivalence_sha256": disposition.unit_qc_equivalence_sha256,
                    "compact_manifest_sha256": compact_cohort.compact_manifest_sha256,
                    "compact_bundle_sha256": compact_cohort.compact_bundle_sha256,
                }
                if disposition.status == "failed":
                    error = RuntimeError(
                        f"{disposition.error_type}: {disposition.error}"
                    )
                    for view in views:
                        for panel in panels:
                            for family in FAMILIES:
                                run.mark_condition_failure(
                                    error,
                                    session_id=disposition.eid,
                                    animal_id=disposition.animal_id,
                                    view=view,
                                    panel=panel,
                                    model_family=family,
                                    aggregation_level="session",
                                )
        for view in views:
            for panel in panels:
                try:
                    result = evaluate_prepared_panel(
                        prepared_sessions,
                        view=view,
                        panel=panel,
                        config=config,
                        seed=seed,
                    )
                    _record_evaluation(run, result, session_provenance=provenance)
                except Exception as error:
                    for session in prepared_sessions:
                        for family in FAMILIES:
                            run.mark_condition_failure(
                                error,
                                session_id=session.eid,
                                animal_id=session.animal_id,
                                view=view,
                                panel=panel,
                                model_family=family,
                                aggregation_level="session",
                            )
        return run.path


def main() -> None:
    parser = basic_parser(
        "Run the multi-session IBL conditional count-dynamics audit",
        "configs/smoke/exp14_ibl_multisession_neural.json",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    seeds = seed_list(args.seeds if args.seeds is not None else config["seeds"])
    for seed in seeds:
        path = run_seed(config, seed, args.results_root)
        print(path)


if __name__ == "__main__":
    main()
