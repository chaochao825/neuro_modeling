"""Connect the validated MD-like belief to real IBL block-switching dynamics.

The registered primary endpoint remains a train-only, teacher-forced
conditional Poisson score.  This experiment adds two things without rewriting
Exp14: an MD recurrent *predictive-prior* gate suitable for stimulus-pre data,
and fixed-checkpoint held-out belief interventions.  ``probabilityLeft`` is a
hash-bound sidecar used only for whole-block splitting and post-fit scoring.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    PROJECT_ROOT,
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from experiments.exp14_ibl_multisession_neural import (
    FAMILIES,
    _animal_mean_nll,
    _fit_score,
    _preprocessing_fingerprint,
    _synthetic_prepared_sessions,
)
from src.analysis.gate_metrics import (
    context_calibration_summary,
    switch_inference_summary,
)
from src.analysis.ibl_belief_dynamics_metrics import (
    BeliefDynamicsContrast,
    compare_belief_dynamics_conditions,
)
from src.analysis.ibl_neural_metrics import compare_count_families
from src.data.ibl_block_sidecar import IBLBlockTruth, load_ibl_block_truth
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


INTERVENTIONS = ("md_shared", "md_clamp", "md_delay_1", "md_delay_5", "md_shuffle")
MODEL_CONDITIONS = ("common", "md_shared", "md_full", "hmm_shared", *INTERVENTIONS[1:])


@dataclass(frozen=True, slots=True)
class PanelTruthBundle:
    panel: IBLNeuralPanelInput
    probability_left: np.ndarray
    true_block_ids: np.ndarray
    behavior_choice: np.ndarray
    truth_provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Exp20Evaluation:
    scores: Mapping[str, HierarchicalCountScore]
    models: Mapping[str, HierarchicalCountDynamics]
    md_sessions: tuple[NeuralCountSession, ...]
    hmm_sessions: tuple[NeuralCountSession, ...]
    splits: Mapping[str, TrialBlockSplit]
    bundles: Mapping[str, PanelTruthBundle]
    selected_latent_dim: int
    selected_ridge: float
    common_regions: tuple[str, ...]
    nested_candidates: tuple[Mapping[str, object], ...]
    comparison: object
    belief_contrasts: tuple[BeliefDynamicsContrast, ...]
    heldout_beliefs: Mapping[str, Mapping[str, np.ndarray]]


def true_block_ids(probability_left: Sequence[float]) -> np.ndarray:
    """Label contiguous probabilityLeft runs without exposing labels to a gate."""

    probability = np.asarray(probability_left, dtype=float)
    if probability.ndim != 1 or probability.size < 2 or not np.isfinite(probability).all():
        raise ValueError("probability_left must be a finite trial vector")
    blocks = np.zeros(probability.size, dtype=np.int64)
    blocks[1:] = np.cumsum(~np.isclose(probability[1:], probability[:-1], atol=1e-9))
    blocks.setflags(write=False)
    return blocks


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256(b"exp20-heldout-belief-v1\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _synthetic_truth(prepared: PreparedIBLNeuralSession, view: str) -> IBLBlockTruth:
    table = prepared.trial_table(view)
    if "probability_left" not in table:
        raise ValueError("synthetic smoke table lacks probability_left")
    trial_ids = np.asarray(prepared.current_trial_ids, dtype=np.int64)
    probability = table["probability_left"].to_numpy(dtype=float)
    switches = np.zeros(len(probability), dtype=bool)
    switches[1:] = ~np.isclose(probability[1:], probability[:-1], atol=1e-9)
    return IBLBlockTruth(
        trial_ids=trial_ids,
        probability_left=probability,
        block_switch=switches,
        official_bwm_mask=np.ones(len(probability), dtype=bool),
    )


def _align_truth(panel: IBLNeuralPanelInput, truth: IBLBlockTruth) -> tuple[np.ndarray, np.ndarray]:
    lookup = {int(value): index for index, value in enumerate(truth.trial_ids.tolist())}
    try:
        positions = np.asarray([lookup[int(value)] for value in panel.trial_ids], dtype=int)
    except KeyError as error:
        raise ValueError("neural panel trial is absent from block truth sidecar") from error
    probability = np.asarray(truth.probability_left[positions], dtype=float)
    return probability, true_block_ids(probability)


def _macro_mapping(config: Mapping[str, Any]):
    mapping_path = Path(str(config["macro_region_mapping_path"]))
    if not mapping_path.is_absolute():
        mapping_path = PROJECT_ROOT / mapping_path
    formal = str(config.get("profile")) == "formal"
    source_ontology = None
    if formal:
        if str(config["macro_region_mapping_formal_compact_manifest_sha256"]) != str(
            config["expected_compact_manifest_sha256"]
        ):
            raise ValueError(
                "macro-region artifact is not bound to the formal compact manifest"
            )
        compact_manifest = Path(str(config["compact_cache_manifest"]))
        if not compact_manifest.is_absolute():
            compact_manifest = PROJECT_ROOT / compact_manifest
        source_ontology = (
            compact_manifest.parent / "provenance" / "iblatlas_allen_structure_tree.csv"
        )
    mapping = load_allen_macro_region_mapping(
        mapping_path,
        expected_sha256=str(config["expected_macro_region_mapping_sha256"]),
        expected_compact_manifest_sha256=str(
            config["macro_region_mapping_formal_compact_manifest_sha256"]
        ),
        source_ontology_path=source_ontology,
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
    return mapping


def _prepare_bundles(
    prepared_sessions: Sequence[PreparedIBLNeuralSession],
    *,
    config: Mapping[str, Any],
) -> tuple[PanelTruthBundle, ...]:
    view = str(config["view"])
    panel_name = str(config["panel"])
    mapping = _macro_mapping(config)
    formal = str(config.get("profile")) == "formal"
    mapping.validate_acronym_scope(
        tuple(
            str(acronym)
            for prepared in prepared_sessions
            for acronym in prepared.regions.tolist()
        ),
        require_exact_formal_scope=formal,
    )
    if formal:
        manifest = Path(str(config["behavior_truth_manifest"]))
        if not manifest.is_absolute():
            manifest = PROJECT_ROOT / manifest
    bundles = []
    for prepared in prepared_sessions:
        panel = prepare_neural_panel_input(
            prepared,
            view=view,
            panel=panel_name,
            minimum_trials=int(config["minimum_trials"]),
            minimum_blocks=2,
            macro_region_mapping=mapping,
        )
        if formal:
            truth, provenance = load_ibl_block_truth(
                manifest,
                prepared,
                expected_manifest_sha256=str(
                    config["expected_behavior_truth_manifest_sha256"]
                ),
            )
        else:
            truth = _synthetic_truth(prepared, view)
            provenance = {
                "scope": "evaluation_only",
                "source": "synthetic_smoke",
            }
        probability, blocks = _align_truth(panel, truth)
        table = prepared.trial_table(view)
        row_by_trial = {
            int(trial_id): index
            for index, trial_id in enumerate(prepared.current_trial_ids.tolist())
        }
        choice = np.asarray(
            [table.iloc[row_by_trial[int(trial_id)]]["choice"] for trial_id in panel.trial_ids],
            dtype=float,
        )
        choice.setflags(write=False)
        if np.unique(blocks).size < int(config["minimum_blocks"]):
            raise ValueError(f"session {panel.session_id} has too few true context blocks")
        bundles.append(
            PanelTruthBundle(panel, probability, blocks, choice, provenance)
        )
    return tuple(bundles)


def _make_splits(
    bundles: Sequence[PanelTruthBundle], config: Mapping[str, Any]
) -> tuple[dict[str, ChronologicalBlockSplit], dict[str, ChronologicalBlockSplit]]:
    outer, inner = {}, {}
    for bundle in bundles:
        first, second = chronological_outer_inner_splits(
            bundle.panel.trial_ids,
            bundle.true_block_ids,
            outer_test_fraction=float(config["outer_test_fraction"]),
            inner_validation_fraction=float(config["inner_validation_fraction"]),
        )
        outer[bundle.panel.session_id] = first
        inner[bundle.panel.session_id] = second
    return outer, inner


def _build_sessions(
    bundles: Sequence[PanelTruthBundle],
    splits: Mapping[str, ChronologicalBlockSplit],
    *,
    common_regions: Sequence[str],
    gate_options: Mapping[str, object],
    config: Mapping[str, Any],
    seed: int,
) -> tuple[tuple[NeuralCountSession, ...], dict[str, TrialBlockSplit]]:
    built = []
    for bundle in bundles:
        item = build_model_session(
            bundle.panel,
            splits[bundle.panel.session_id],
            common_regions=common_regions,
            max_units_per_region=int(config["max_units_per_region"]),
            min_units_per_region=int(config["min_units_per_region"]),
            hmm_options=gate_options,
            seed=derive_seed(seed, "exp20", bundle.panel.session_id, "belief"),
            split_block_ids=bundle.true_block_ids,
        )
        built.append(item)
    return (
        tuple(item.session for item in built),
        {item.session.session_id: item.split for item in built},
    )


def _belief_overrides(
    sessions: Sequence[NeuralCountSession],
    splits: Mapping[str, TrialBlockSplit],
    *,
    delay: int | None = None,
    clamp: bool = False,
    shuffle_seed: int | None = None,
) -> dict[str, np.ndarray]:
    result = {}
    for session in sessions:
        positions = {value: index for index, value in enumerate(session.trial_ids.tolist())}
        test_positions = np.asarray(
            [positions[value] for value in splits[session.session_id].test_trial_ids], dtype=int
        )
        if clamp:
            beliefs = np.full((test_positions.size, 2), 0.5)
        elif delay is not None:
            if delay < 1:
                raise ValueError("delay must be positive")
            delayed = np.full_like(session.beliefs, 0.5)
            delayed[delay:] = session.beliefs[:-delay]
            beliefs = delayed[test_positions]
        elif shuffle_seed is not None:
            intact = session.beliefs[test_positions]
            if len(intact) < 2:
                raise ValueError("belief shuffle requires two held-out trials")
            rng = np.random.default_rng(
                derive_seed(shuffle_seed, "exp20", session.session_id, "shuffle")
            )
            shift = int(rng.integers(1, len(intact)))
            beliefs = np.roll(intact, shift, axis=0)
        else:
            beliefs = session.beliefs[test_positions]
        frozen = np.array(beliefs, dtype=float, copy=True)
        frozen.setflags(write=False)
        result[session.session_id] = frozen
    return result


def _assert_paired_gate_sessions(
    md_sessions: Sequence[NeuralCountSession],
    hmm_sessions: Sequence[NeuralCountSession],
) -> None:
    """Ensure gate comparisons differ only in their belief trajectories."""

    if tuple(item.session_id for item in md_sessions) != tuple(
        item.session_id for item in hmm_sessions
    ):
        raise RuntimeError("MD and HMM representations changed session identity/order")
    for md_session, hmm_session in zip(md_sessions, hmm_sessions, strict=True):
        immutable_pairs = (
            (md_session.counts, hmm_session.counts),
            (md_session.trial_ids, hmm_session.trial_ids),
            (md_session.controls, hmm_session.controls),
        )
        if md_session.unit_regions != hmm_session.unit_regions or any(
            not np.array_equal(left, right) for left, right in immutable_pairs
        ):
            raise RuntimeError(
                "MD and HMM representations changed neural data/preprocessing inputs"
            )


def _operator_metrics(model: HierarchicalCountDynamics) -> dict[str, object]:
    state0 = np.asarray(model.transition_matrices_["state_0"])
    state1 = np.asarray(model.transition_matrices_["state_1"])
    delta = state1[:, : model.latent_dim] - state0[:, : model.latent_dim]
    return {
        "shared_basis_dim": model.latent_dim,
        "belief_operator_family_dimension": int(not np.allclose(delta, 0.0)),
        "state_operator_delta_rank": int(np.linalg.matrix_rank(delta)),
        "state0_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvals(state0[:, : model.latent_dim])))
        ),
        "state1_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvals(state1[:, : model.latent_dim])))
        ),
    }


def evaluate_prepared_sessions(
    prepared_sessions: Sequence[PreparedIBLNeuralSession],
    *,
    config: Mapping[str, Any],
    seed: int,
) -> Exp20Evaluation:
    if tuple(sorted(int(value) for value in config["interventions"]["delay_trials"])) != (
        1,
        5,
    ):
        raise ValueError("Exp20 preregisters exactly the 1-trial and 5-trial delays")
    prepared_ids = tuple(item.eid for item in prepared_sessions)
    if not prepared_ids or len(set(prepared_ids)) != len(prepared_ids):
        raise ValueError("prepared session IDs must be non-empty and unique")
    bundles = _prepare_bundles(prepared_sessions, config=config)
    panels = tuple(item.panel for item in bundles)
    anchor_audit = union_region_anchors(
        panels,
        min_units_per_region=int(config["min_units_per_region"]),
        minimum_region_sessions=int(config["minimum_region_sessions"]),
    )
    common_regions = anchor_audit.regions
    outer, inner = _make_splits(bundles, config)
    inner_sessions, inner_model_splits = _build_sessions(
        bundles,
        inner,
        common_regions=common_regions,
        gate_options=dict(config["md_gate"]),
        config=config,
        seed=derive_seed(seed, "exp20", "inner"),
    )
    candidates = []
    successful = []
    for latent_dim in sorted({int(value) for value in config["latent_dims"]}):
        if not 1 <= latent_dim <= len(common_regions):
            continue
        for ridge in sorted({float(value) for value in config["ridges"]}):
            try:
                objectives = []
                for family in FAMILIES:
                    _, score = _fit_score(
                        family,
                        inner_sessions,
                        inner_model_splits,
                        common_regions=common_regions,
                        latent_dim=latent_dim,
                        ridge=ridge,
                        seed=derive_seed(seed, "exp20", "inner", latent_dim, ridge, family),
                    )
                    objectives.append(_animal_mean_nll(score))
                objective = float(np.mean(objectives))
                successful.append((objective, latent_dim, ridge))
                candidates.append(
                    {"latent_dim": latent_dim, "ridge": ridge, "status": "complete", "objective": objective}
                )
            except Exception as error:
                candidates.append(
                    {
                        "latent_dim": latent_dim,
                        "ridge": ridge,
                        "status": "failed",
                        "objective": float("nan"),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    if not successful:
        raise RuntimeError("all nested Exp20 hyperparameter candidates failed")
    _, selected_dim, selected_ridge = min(successful)

    md_sessions, model_splits = _build_sessions(
        bundles,
        outer,
        common_regions=common_regions,
        gate_options=dict(config["md_gate"]),
        config=config,
        seed=derive_seed(seed, "exp20", "outer", "md"),
    )
    hmm_sessions, hmm_splits = _build_sessions(
        bundles,
        outer,
        common_regions=common_regions,
        gate_options=dict(config["hmm_baseline"]),
        config=config,
        seed=derive_seed(seed, "exp20", "outer", "hmm"),
    )
    if model_splits != hmm_splits:
        raise RuntimeError("MD and HMM gates received different outer splits")
    _assert_paired_gate_sessions(md_sessions, hmm_sessions)
    models, scores = {}, {}
    for family in FAMILIES:
        model, score = _fit_score(
            family,
            md_sessions,
            model_splits,
            common_regions=common_regions,
            latent_dim=selected_dim,
            ridge=selected_ridge,
            seed=derive_seed(seed, "exp20", family),
        )
        models[f"md_{family}"] = model
        score_key = "common" if family == "common" else f"md_{family}"
        scores[score_key] = score
    shared = models["md_shared"]
    shared_fit_fingerprint = shared.fit_fingerprint_
    heldout_beliefs = {
        "md_shared": _belief_overrides(md_sessions, model_splits),
        "md_full": _belief_overrides(md_sessions, model_splits),
        "hmm_shared": _belief_overrides(hmm_sessions, model_splits),
        "md_clamp": _belief_overrides(md_sessions, model_splits, clamp=True),
    }
    scores["md_clamp"] = shared.score_heldout_belief_counterfactual(
        md_sessions, model_splits, heldout_beliefs["md_clamp"]
    )
    for delay in tuple(int(value) for value in config["interventions"]["delay_trials"]):
        condition = f"md_delay_{delay}"
        heldout_beliefs[condition] = _belief_overrides(
            md_sessions, model_splits, delay=delay
        )
        scores[condition] = shared.score_heldout_belief_counterfactual(
            md_sessions,
            model_splits,
            heldout_beliefs[condition],
        )
    heldout_beliefs["md_shuffle"] = _belief_overrides(
        md_sessions, model_splits, shuffle_seed=seed
    )
    scores["md_shuffle"] = shared.score_heldout_belief_counterfactual(
        md_sessions,
        model_splits,
        heldout_beliefs["md_shuffle"],
    )
    if shared.fit_fingerprint_ != shared_fit_fingerprint:
        raise RuntimeError("held-out belief interventions mutated the shared checkpoint")
    hmm_model, hmm_score = _fit_score(
        "shared",
        hmm_sessions,
        hmm_splits,
        common_regions=common_regions,
        latent_dim=selected_dim,
        ridge=selected_ridge,
        seed=derive_seed(seed, "exp20", "hmm_shared"),
    )
    models["hmm_shared"] = hmm_model
    scores["hmm_shared"] = hmm_score
    md_preprocessing = {
        _preprocessing_fingerprint(models[f"md_{family}"]) for family in FAMILIES
    }
    if len(md_preprocessing) != 1 or _preprocessing_fingerprint(hmm_model) not in md_preprocessing:
        raise RuntimeError("paired gates/model families did not share count preprocessing")
    comparison = compare_count_families(
        {
            "common": scores["common"],
            "shared": scores["md_shared"],
            "full": scores["md_full"],
        },
        planned_sessions=int(config["planned_sessions"]),
        planned_animals=int(config["planned_animals"]),
        n_bootstrap=int(config["n_bootstrap"]),
        seed=derive_seed(seed, "exp20", "bootstrap"),
    )
    comparator_conditions = (
        "common",
        "hmm_shared",
        "md_clamp",
        "md_delay_1",
        "md_delay_5",
        "md_shuffle",
    )
    belief_contrasts = compare_belief_dynamics_conditions(
        {
            condition: scores[condition]
            for condition in ("md_shared", *comparator_conditions)
        },
        intact_condition="md_shared",
        comparator_conditions=comparator_conditions,
        planned_sessions=int(config["planned_sessions"]),
        planned_animals=int(config["planned_animals"]),
        n_bootstrap=int(config["n_bootstrap"]),
        seed=derive_seed(seed, "exp20", "belief_contrast_bootstrap"),
    )
    return Exp20Evaluation(
        scores=scores,
        models=models,
        md_sessions=md_sessions,
        hmm_sessions=hmm_sessions,
        splits=model_splits,
        bundles={item.panel.session_id: item for item in bundles},
        selected_latent_dim=selected_dim,
        selected_ridge=selected_ridge,
        common_regions=common_regions,
        nested_candidates=tuple(candidates),
        comparison=comparison,
        belief_contrasts=belief_contrasts,
        heldout_beliefs=heldout_beliefs,
    )


def _condition_beliefs(
    result: Exp20Evaluation, condition: str
) -> Mapping[str, np.ndarray] | None:
    if condition == "common":
        return None
    try:
        return result.heldout_beliefs[condition]
    except KeyError as error:
        raise ValueError(f"unknown condition {condition!r}") from error


def _causal_choice_history_bias(choice: Sequence[float], *, alpha: float) -> np.ndarray:
    """Return a descriptive p(left) proxy using previous choices only."""

    smoothing = float(alpha)
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("behavior_bias_alpha must lie in (0, 1]")
    values = np.asarray(choice, dtype=float)
    if values.ndim != 1:
        raise ValueError("behavior choices must form a vector")
    probability = np.empty(len(values), dtype=float)
    state = 0.5
    for index, value in enumerate(values):
        probability[index] = state
        if np.isclose(value, -1.0):
            state = (1.0 - smoothing) * state + smoothing
        elif np.isclose(value, 1.0):
            state = (1.0 - smoothing) * state
    probability.setflags(write=False)
    return probability


def _switch_metrics(
    posterior: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    *,
    prefix: str,
) -> dict[str, object]:
    values: dict[str, object] = {
        f"{prefix}_switch_latency_trials": float("nan"),
        f"{prefix}_false_switch_rate": float("nan"),
        f"{prefix}_switch_metric_status": "no_eligible_switch",
    }
    try:
        switch = switch_inference_summary(
            posterior,
            labels,
            episode_ids,
            max_latency=10,
            minimum_state_duration=5,
        )
    except ValueError:
        return values
    return {
        f"{prefix}_switch_latency_trials": switch.mean_latency_trials,
        f"{prefix}_false_switch_rate": switch.false_switch_rate,
        f"{prefix}_switch_metric_status": "complete",
    }


def _truth_metrics(
    result: Exp20Evaluation,
    session: NeuralCountSession,
    beliefs: np.ndarray | None,
    *,
    behavior_bias_alpha: float,
) -> dict[str, object]:
    bundle = result.bundles[session.session_id]
    truth_by_id = dict(
        zip(
            bundle.panel.trial_ids.tolist(),
            bundle.probability_left.tolist(),
            strict=True,
        )
    )
    test_ids = result.splits[session.session_id].test_trial_ids
    probability = np.asarray([truth_by_id[int(value)] for value in test_ids], dtype=float)
    binary = np.isclose(probability, 0.2) | np.isclose(probability, 0.8)
    binary_positions = np.flatnonzero(binary)
    labels = (probability[binary] > 0.5).astype(int)
    if not len(labels):
        raise ValueError("held-out suffix contains no binary biased-block trials")
    episodes = np.zeros(len(binary_positions), dtype=int)
    if len(binary_positions) > 1:
        episodes[1:] = np.cumsum(np.diff(binary_positions) > 1)

    behavior_full = _causal_choice_history_bias(
        bundle.behavior_choice,
        alpha=behavior_bias_alpha,
    )
    behavior_by_id = dict(
        zip(bundle.panel.trial_ids.tolist(), behavior_full.tolist(), strict=True)
    )
    behavior_posterior = np.asarray(
        [behavior_by_id[int(value)] for value in test_ids], dtype=float
    )[binary]
    behavior_calibration = context_calibration_summary(
        behavior_posterior,
        labels,
        n_bins=10,
        epsilon=1e-9,
    )
    behavior_switch = _switch_metrics(
        behavior_posterior,
        labels,
        episodes,
        prefix="behavior_bias",
    )
    result_metrics: dict[str, object] = {
        "binary_context_test_trials": int(len(labels)),
        "behavior_bias_definition": "causal_ewma_of_previous_left_choices",
        "behavior_bias_alpha": float(behavior_bias_alpha),
        "behavior_bias_context_nll": behavior_calibration.nll,
        "behavior_bias_context_brier": behavior_calibration.brier,
        "behavior_bias_context_ece": behavior_calibration.expected_calibration_error,
        **behavior_switch,
    }
    if beliefs is None:
        return {
            **result_metrics,
            "context_nll": float("nan"),
            "context_brier": float("nan"),
            "context_ece": float("nan"),
            "belief_switch_latency_trials": float("nan"),
            "belief_false_switch_rate": float("nan"),
            "belief_switch_metric_status": "not_applicable",
            "belief_minus_behavior_switch_latency_trials": float("nan"),
        }

    posterior = np.asarray(beliefs, dtype=float)[binary, 1]
    calibration = context_calibration_summary(
        posterior,
        labels,
        n_bins=10,
        epsilon=1e-9,
    )
    belief_switch = _switch_metrics(
        posterior,
        labels,
        episodes,
        prefix="belief",
    )
    belief_latency = float(belief_switch["belief_switch_latency_trials"])
    behavior_latency = float(
        behavior_switch["behavior_bias_switch_latency_trials"]
    )
    return {
        **result_metrics,
        "context_nll": calibration.nll,
        "context_brier": calibration.brier,
        "context_ece": calibration.expected_calibration_error,
        **belief_switch,
        "belief_minus_behavior_switch_latency_trials": (
            belief_latency - behavior_latency
            if np.isfinite(belief_latency) and np.isfinite(behavior_latency)
            else float("nan")
        ),
    }


def _record_evaluation(
    run: ExperimentRun,
    result: Exp20Evaluation,
    *,
    behavior_bias_alpha: float,
) -> None:
    for candidate in result.nested_candidates:
        run.record(
            candidate,
            stage="nested_selection",
            condition="md_gate_common_shared_full_mean",
        )
    for contrast in result.belief_contrasts:
        run.record(
            asdict(contrast),
            stage="animal_session_belief_contrast",
            condition=contrast.comparison,
        )
    session_maps = {
        condition: {item.session_id: item for item in score.per_session}
        for condition, score in result.scores.items()
    }
    md_by_id = {item.session_id: item for item in result.md_sessions}
    hmm_by_id = {item.session_id: item for item in result.hmm_sessions}
    operator = _operator_metrics(result.models["md_shared"])
    for condition in MODEL_CONDITIONS:
        score_key = condition
        beliefs_by_session = _condition_beliefs(result, condition)
        score = result.scores[score_key]
        for session_id, metric in session_maps[score_key].items():
            gate_session = (
                hmm_by_id[session_id]
                if condition == "hmm_shared"
                else md_by_id[session_id]
            )
            gate_receipt = None if condition == "common" else gate_session.belief_receipt
            truth = _truth_metrics(
                result,
                gate_session,
                None if beliefs_by_session is None else beliefs_by_session[session_id],
                behavior_bias_alpha=behavior_bias_alpha,
            )
            model_key = (
                "hmm_shared"
                if condition == "hmm_shared"
                else "md_common"
                if condition == "common"
                else "md_full"
                if condition == "md_full"
                else "md_shared"
            )
            evaluated_belief_sha256 = (
                None
                if beliefs_by_session is None
                else _array_sha256(beliefs_by_session[session_id])
            )
            run.record(
                {
                    "status": "complete",
                    "statistics_unit": "animal_with_session_nested",
                    "view": "stimulus_pre",
                    "panel": "primary_past_safe",
                    "selected_latent_dim": result.selected_latent_dim,
                    "selected_ridge": result.selected_ridge,
                    "common_regions": result.common_regions,
                    "nll_per_count": metric.nll_per_count,
                    "pseudo_r2": metric.pseudo_r2,
                    "closure_mse": metric.closure_mse,
                    "parameter_count": score.parameter_count,
                    "fit_fingerprint": result.models[model_key].fit_fingerprint_,
                    "belief_checkpoint_sha256": (
                        None if gate_receipt is None else gate_receipt.checkpoint_sha256
                    ),
                    "source_belief_trajectory_sha256": (
                        None if gate_receipt is None else gate_receipt.belief_sha256
                    ),
                    "evaluated_heldout_belief_sha256": evaluated_belief_sha256,
                    "belief_fit_method": (
                        "none" if gate_receipt is None else gate_receipt.method
                    ),
                    "belief_uses_current_trial_stimulus": (
                        False
                        if gate_receipt is None
                        else gate_receipt.uses_current_trial_stimulus
                    ),
                    "belief_uses_future_trials": (
                        False if gate_receipt is None else gate_receipt.uses_future_trials
                    ),
                    "belief_accessed_true_context": (
                        False
                        if gate_receipt is None
                        else gate_receipt.accessed_true_context
                    ),
                    "belief_intervention_postfit": condition in INTERVENTIONS[1:],
                    "all_model_parameters_frozen_for_intervention": condition in INTERVENTIONS[1:],
                    "gate_model": (
                        "none"
                        if condition == "common"
                        else "learned_categorical_hmm"
                        if condition == "hmm_shared"
                        else "md_recurrent_belief_predictive_prior"
                    ),
                    "gate_received_probability_left": False,
                    "probability_left_access_scope": "whole_block_split_and_postfit_evaluation_only",
                    "preprocessing_fit_train_only": True,
                    "split_unit": "contiguous_true_probabilityLeft_block",
                    "full_latent_lds": False,
                    "model_scope": (
                        "teacher_forced_one_step_conditional_poisson_"
                        "shared_basis_dynamics_not_full_lds"
                    ),
                    "likelihood_kind": score.likelihood_kind,
                    "truth_sidecar_provenance": result.bundles[session_id].truth_provenance,
                    **truth,
                    **(operator if condition != "common" else {}),
                },
                session_id=session_id,
                animal_id=metric.animal_id,
                condition=condition,
                stage="outer_test",
            )
    run.record(
        {
            "status": "complete",
            "statistics_unit": "animal_with_session_nested",
            "selected_latent_dim": result.selected_latent_dim,
            "selected_ridge": result.selected_ridge,
            "nested_candidates": result.nested_candidates,
            "common_regions": result.common_regions,
            "comparison": asdict(result.comparison),
            "belief_contrasts": [asdict(item) for item in result.belief_contrasts],
            "core_conclusion": result.comparison.conclusion,
            "truth_used_by_gate_or_model": False,
            "full_latent_lds": False,
            **operator,
        },
        stage="cohort_summary",
        condition="md_shared_vs_common_full",
    )


def _load_prepared(config: Mapping[str, Any], seed: int):
    if str(config.get("profile")) == "smoke" and str(config.get("data_mode")) == "synthetic_smoke":
        return _synthetic_prepared_sessions(config, seed)
    if str(config.get("profile")) != "formal" or str(config.get("data_mode")) != "frozen_compact_cache":
        raise RuntimeError("Exp20 permits only synthetic smoke or verified formal compact data")
    manifest = Path(str(config["compact_cache_manifest"]))
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    cohort = load_compact_neural_cohort(
        manifest,
        expected_source_manifest_sha256=str(config["expected_source_manifest_sha256"]),
        expected_acquisition_bundle_sha256=str(config["expected_acquisition_bundle_sha256"]),
        expected_bwm_repository_commit=str(config["expected_bwm_repository_commit"]),
        expected_compact_manifest_sha256=str(config["expected_compact_manifest_sha256"]),
        expected_compact_bundle_sha256=str(config["expected_compact_bundle_sha256"]),
        expected_sessions=int(config["planned_sessions"]),
        minimum_animals=int(config["planned_animals"]),
    )
    return cohort.sessions


def run_seed(
    config: Mapping[str, Any], seed: int, results_root: str | Path
) -> Path:
    initialize_seed(seed)
    run_config = {
        **dict(config),
        "training_algorithm": "md_predictive_prior_shared_count_dynamics",
        "used_autograd": False,
        "used_bptt": False,
        "recurrent_ei_mechanism_claim_eligible": False,
    }
    with ExperimentRun(
        "exp20_ibl_md_belief_dynamics", seed, run_config, results_root=results_root
    ) as run:
        try:
            prepared = _load_prepared(config, seed)
        except Exception as error:
            planned = [{"condition": "cohort_load", "stage": "data_loading"}]
            run.register_conditions(planned)
            run.mark_condition_failure(error, **planned[0])
            return run.path
        planned_ids = [str(item.eid) for item in prepared]
        planned = [
            {"session_id": session_id, "condition": condition, "stage": "outer_test"}
            for session_id in planned_ids
            for condition in MODEL_CONDITIONS
        ]
        run.register_conditions(planned)
        try:
            result = evaluate_prepared_sessions(prepared, config=config, seed=seed)
            _record_evaluation(
                run,
                result,
                behavior_bias_alpha=float(config.get("behavior_bias_alpha", 0.1)),
            )
        except Exception as error:
            for item in planned:
                run.mark_condition_failure(error, **item)
        return run.path


def main() -> None:
    parser = basic_parser(
        __doc__ or "IBL MD belief dynamics",
        "configs/smoke/exp20_ibl_md_belief_dynamics.json",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
