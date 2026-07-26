"""IBL hidden-block utility of factorized causal controller states."""

from __future__ import annotations

import sys
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
from experiments.exp11_ibl_behavior_belief import (
    _configured_session_specs,
    _load_configured_sessions,
)
from src.analysis.ibl_behavior_metrics import (
    binary_context_metrics,
    oracle_ceiling_beliefs,
)
from src.analysis.ibl_state_utility import (
    FACTORIZED_STATE_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    MEAN_STATE_FEATURE_NAMES,
    belief_mean_features,
    causal_history_features,
    evaluate_choice_readout,
    factorized_clamp_features,
    factorized_state_features,
)
from src.data.ibl_behavior import (
    IBLBehaviorDataError,
    IBLBehaviorSession,
    LearnedCategoricalHMM,
    contiguous_block_split,
)
from src.models.ibl_run_length_observer import (
    RunLengthCandidate,
    SemiMarkovBlockObserver,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


CONDITIONS = (
    "history_only",
    "learned_hmm_mean",
    "semimarkov_mean",
    "semimarkov_release",
    "semimarkov_concentration",
    "factorized_state",
    "oracle_context_mean",
)
NONFACTORIZED_BASELINES = (
    "history_only",
    "learned_hmm_mean",
    "semimarkov_mean",
)
RELEASE_FEATURE_NAMES = MEAN_STATE_FEATURE_NAMES + (
    "release_probability",
    "prior_x_release",
)
CONCENTRATION_FEATURE_NAMES = MEAN_STATE_FEATURE_NAMES + (
    "run_length_concentration",
    "prior_x_concentration",
)


def _feature_columns(features: np.ndarray, names: Sequence[str]) -> np.ndarray:
    indices = [FACTORIZED_STATE_FEATURE_NAMES.index(name) for name in names]
    result = np.array(features[:, indices], copy=True)
    result.setflags(write=False)
    return result


def _candidate_grid(config: Mapping[str, Any]) -> tuple[RunLengthCandidate, ...]:
    raw = config.get("run_length_candidates", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("run_length_candidates must be a non-empty sequence")
    candidates = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise ValueError("every run-length candidate must be an object")
        candidates.append(
            RunLengthCandidate(
                min_run=int(value["min_run"]),
                max_run=int(value["max_run"]),
                hazard_scale=float(value["hazard_scale"]),
            )
        )
    return tuple(candidates)


def _make_condition_inputs(
    session: IBLBehaviorSession,
    train_indices: np.ndarray,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[
    dict[str, tuple[np.ndarray, tuple[str, ...], np.ndarray, dict[str, object]]],
    dict[str, Exception],
]:
    inputs: dict[
        str, tuple[np.ndarray, tuple[str, ...], np.ndarray, dict[str, object]]
    ] = {}
    failures: dict[str, Exception] = {}
    observations = session.observations
    uniform = np.full((session.trial_ids.size, 2), 0.5)
    inputs["history_only"] = (
        causal_history_features(session),
        HISTORY_FEATURE_NAMES,
        uniform,
        {
            "state_dimension": 0,
            "gate_fit_trial_ids": [],
            "gate_fit_supervision": "none",
        },
    )

    try:
        hmm_options = dict(config.get("learned_hmm", {}))
        hmm_options["seed"] = derive_seed(seed, "exp40", session.eid, "hmm")
        hmm = LearnedCategoricalHMM(**hmm_options).fit(observations, train_indices)
        prediction = hmm.predict(observations)
        inputs["learned_hmm_mean"] = (
            belief_mean_features(session, prediction.beliefs),
            MEAN_STATE_FEATURE_NAMES,
            prediction.beliefs,
            {
                "state_dimension": 1,
                "gate_fit_trial_ids": prediction.fit_trial_ids.tolist(),
                "belief_trajectory_id": prediction.fingerprint,
                "hmm_train_log_likelihood": hmm.train_log_likelihood_,
                "hmm_fit_converged": hmm.converged_,
                "hmm_state_separation_identifiable": hmm.identifiable_,
                "hmm_emission_gap": hmm.emission_gap_,
                "gate_fit_supervision": "stimulus_side_only_unsupervised",
            },
        )
    except Exception as error:
        failures["learned_hmm_mean"] = error

    try:
        observer_options = dict(config.get("run_length_observer", {}))
        observer = SemiMarkovBlockObserver(
            candidates=_candidate_grid(config), **observer_options
        ).fit(observations, train_indices)
        prediction = observer.predict(observations)
        factorized = factorized_state_features(session, prediction)
        shared = {
            "gate_fit_trial_ids": prediction.fit_trial_ids.tolist(),
            "belief_trajectory_id": prediction.fingerprint,
            "run_length_min": prediction.candidate.min_run,
            "run_length_max": prediction.candidate.max_run,
            "run_length_hazard_scale": prediction.candidate.hazard_scale,
            "run_length_release_window": prediction.release_window,
            "run_length_train_stimulus_nll": prediction.train_stimulus_nll,
            "gate_fit_supervision": "stimulus_side_only_task_structured",
        }
        inputs["semimarkov_mean"] = (
            factorized[:, : len(MEAN_STATE_FEATURE_NAMES)],
            MEAN_STATE_FEATURE_NAMES,
            prediction.beliefs,
            {**shared, "state_dimension": 1},
        )
        inputs["semimarkov_release"] = (
            _feature_columns(factorized, RELEASE_FEATURE_NAMES),
            RELEASE_FEATURE_NAMES,
            prediction.beliefs,
            {**shared, "state_dimension": 2},
        )
        inputs["semimarkov_concentration"] = (
            _feature_columns(factorized, CONCENTRATION_FEATURE_NAMES),
            CONCENTRATION_FEATURE_NAMES,
            prediction.beliefs,
            {**shared, "state_dimension": 2},
        )
        inputs["factorized_state"] = (
            factorized,
            FACTORIZED_STATE_FEATURE_NAMES,
            prediction.beliefs,
            {**shared, "state_dimension": 3},
        )
    except Exception as error:
        for condition in (
            "semimarkov_mean",
            "semimarkov_release",
            "semimarkov_concentration",
            "factorized_state",
        ):
            failures[condition] = error

    try:
        oracle = oracle_ceiling_beliefs(session.context_labels)
        inputs["oracle_context_mean"] = (
            belief_mean_features(session, oracle),
            MEAN_STATE_FEATURE_NAMES,
            oracle,
            {
                "state_dimension": 1,
                "gate_fit_trial_ids": [],
                "belief_trajectory_id": "evaluation_truth_ceiling",
                "gate_fit_supervision": "evaluation_truth_only",
            },
        )
    except Exception as error:
        failures["oracle_context_mean"] = error
    return inputs, failures


def _flatten_metrics(evaluation: Any) -> dict[str, object]:
    payload: dict[str, object] = {
        "selected_regularization_c": evaluation.selected_c,
        "dev_regularization_selection_scope": evaluation.dev_selection_scope,
        "dev_selection_nll": evaluation.dev_selection_nll,
        "dev_selection_trial_count": evaluation.dev_selection_trial_count,
        "behavior_feature_count": evaluation.feature_count,
        "behavior_parameter_count": evaluation.feature_count + 1,
        "behavior_feature_names": list(evaluation.feature_names),
        "behavior_readout_fit_trial_ids": evaluation.fit_trial_ids.tolist(),
        "behavior_dev_selection_trial_ids": evaluation.dev_trial_ids.tolist(),
        "behavior_test_trial_ids": evaluation.test_trial_ids.tolist(),
    }
    for subset, metrics in evaluation.metrics.items():
        prefix = f"test_{subset}"
        payload.update(
            {
                f"{prefix}_choice_nll": metrics.nll,
                f"{prefix}_choice_brier": metrics.brier,
                f"{prefix}_choice_accuracy": metrics.accuracy,
                f"{prefix}_choice_count": metrics.n_trials,
            }
        )
    for intervention, subsets in evaluation.intervention_metrics.items():
        for subset, metrics in subsets.items():
            prefix = f"intervention_{intervention}_{subset}"
            payload.update(
                {
                    f"{prefix}_choice_nll": metrics.nll,
                    f"{prefix}_choice_brier": metrics.brier,
                    f"{prefix}_choice_accuracy": metrics.accuracy,
                    f"{prefix}_choice_count": metrics.n_trials,
                }
            )
    if evaluation.dev_selection_scope == "low_contrast":
        payload["dev_low_contrast_selection_nll"] = evaluation.dev_selection_nll
        payload["dev_low_contrast_selection_trial_count"] = (
            evaluation.dev_selection_trial_count
        )
    return payload


def _session_dimensions(
    session_id: str,
    animal_id: str,
    condition: str,
    provenance: Mapping[str, str],
) -> dict[str, object]:
    return {
        "condition": condition,
        "eid": session_id,
        "session_id": session_id,
        "animal_id": animal_id,
        "statistics_unit": "animal",
        "aggregation_level": "session_then_animal",
        **{
            name: value for name, value in provenance.items() if value not in {None, ""}
        },
    }


def _write_predictions(
    run_path: Path,
    session: IBLBehaviorSession,
    condition: str,
    evaluation: Any,
) -> str:
    directory = run_path / "raw_predictions"
    directory.mkdir(exist_ok=True)
    path = directory / f"{session.eid}_{condition}.npz"
    test_lookup = {int(value): index for index, value in enumerate(session.trial_ids)}
    positions = np.array(
        [test_lookup[int(value)] for value in evaluation.test_trial_ids], dtype=int
    )
    np.savez_compressed(
        path,
        trial_ids=evaluation.test_trial_ids,
        choice_left=session.choice_left[positions],
        signed_contrast=session.signed_contrast[positions],
        context_labels=session.context_labels[positions],
        probabilities=evaluation.test_probabilities,
    )
    return str(path.relative_to(run_path))


def run_seed(
    config: dict[str, Any],
    seed: int,
    results_root: str,
    *,
    sessions: Sequence[IBLBehaviorSession] | None = None,
) -> Path:
    """Run the complete paired session panel and preserve every failure."""

    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "causal_semimarkov_state_plus_frozen_logistic_v1",
        "used_autograd": False,
        "data_access": "local_trial_tables_only",
        "true_context_available_to_candidate": False,
    }
    with ExperimentRun(
        "exp40_ibl_state_utility", seed, run_config, results_root=results_root
    ) as run:
        if sessions is None:
            specs = _configured_session_specs(config)
            candidates = _load_configured_sessions(specs)
        else:
            if not sessions or not all(
                isinstance(item, IBLBehaviorSession) for item in sessions
            ):
                raise TypeError("sessions must contain IBLBehaviorSession values")
            candidates = [
                (
                    item.eid,
                    item.animal_id,
                    item,
                    {"cohort_id": "injected_test_session"},
                )
                for item in sessions
            ]
        if not candidates:
            raise IBLBehaviorDataError("no local IBL sessions are configured")
        planned = [
            _session_dimensions(eid, animal, condition, provenance)
            for eid, animal, _, provenance in candidates
            for condition in CONDITIONS
        ]
        run.register_conditions(planned)
        if config.get("cohort_manifest"):
            manifest = Path(str(config["cohort_manifest"]))
            if not manifest.is_absolute():
                manifest = PROJECT_ROOT / manifest
            (run.path / "cohort_manifest.csv").write_bytes(manifest.read_bytes())

        readout_options = dict(config.get("choice_readout", {}))
        split_options = dict(config.get("split", {}))
        baseline_policy = str(config.get("primary_baseline_condition", "dev_best"))
        if baseline_policy not in {*NONFACTORIZED_BASELINES, "dev_best"}:
            raise ValueError("primary_baseline_condition is not registered")

        for eid, animal_id, loaded, provenance in candidates:
            if isinstance(loaded, Exception):
                for condition in CONDITIONS:
                    run.mark_condition_failure(
                        loaded,
                        **_session_dimensions(eid, animal_id, condition, provenance),
                    )
                continue
            session = loaded
            try:
                split = contiguous_block_split(session.block_ids, **split_options)
                inputs, construction_failures = _make_condition_inputs(
                    session,
                    split.train_indices,
                    config=config,
                    seed=seed,
                )
            except Exception as error:
                for condition in CONDITIONS:
                    run.mark_condition_failure(
                        error,
                        **_session_dimensions(eid, animal_id, condition, provenance),
                    )
                continue

            evaluations: dict[str, Any] = {}
            records: dict[str, dict[str, object]] = {}
            for condition in CONDITIONS:
                dimensions = _session_dimensions(eid, animal_id, condition, provenance)
                if condition in construction_failures:
                    run.mark_condition_failure(
                        construction_failures[condition], **dimensions
                    )
                    continue
                features, names, beliefs, gate_provenance = inputs[condition]
                interventions: dict[str, np.ndarray] = {}
                if condition == "factorized_state":
                    fit = np.concatenate([split.train_indices, split.dev_indices])
                    interventions = {
                        "clamp_release": factorized_clamp_features(
                            features, fit_indices=fit, clamp="release"
                        ),
                        "clamp_concentration": factorized_clamp_features(
                            features, fit_indices=fit, clamp="concentration"
                        ),
                    }
                try:
                    evaluation = evaluate_choice_readout(
                        session,
                        split,
                        features,
                        condition=condition,
                        feature_names=names,
                        seed=derive_seed(
                            seed, "exp40", session.eid, condition, "readout"
                        ),
                        interventions=interventions,
                        **readout_options,
                    )
                    context_test = split.test_indices[
                        session.context_score_mask[split.test_indices]
                    ]
                    context = binary_context_metrics(
                        beliefs,
                        session.context_labels,
                        indices=context_test,
                    )
                    evaluations[condition] = evaluation
                    records[condition] = {
                        "status": "complete",
                        "profile": str(config.get("profile", "unspecified")),
                        "behavior_only_benchmark": True,
                        "neural_activity_analyzed": False,
                        "split_unit": "contiguous_probabilityLeft_block",
                        "split_is_chronological": True,
                        "train_dev_test_blocks_disjoint": True,
                        "preprocessing_fit_train_dev_only": True,
                        "dev_used_for_regularization_selection": True,
                        "test_used_for_selection": False,
                        "gate_uses_stimulus_side_only": condition
                        != "oracle_context_mean",
                        "gate_uses_current_trial_stimulus": False
                        if condition != "oracle_context_mean"
                        else None,
                        "gate_uses_future_stimuli": False
                        if condition != "oracle_context_mean"
                        else None,
                        "gate_accessed_probabilityLeft": condition
                        == "oracle_context_mean",
                        "truth_used_for_fold_grouping": True,
                        "oracle_is_evaluation_only": condition == "oracle_context_mean",
                        "context_nll": context.nll,
                        "context_brier": context.brier,
                        "context_accuracy": context.accuracy,
                        "context_trial_count": context.n_trials,
                        "session_trial_count": int(session.trial_ids.size),
                        "session_block_count": session.n_blocks,
                        "valid_choice_count": int(np.sum(session.choice_valid)),
                        "train_block_count": int(split.train_block_ids.size),
                        "dev_block_count": int(split.dev_block_ids.size),
                        "test_block_count": int(split.test_block_ids.size),
                        "split_id": split.fingerprint,
                        "observation_tape_id": session.observations.fingerprint,
                        "raw_prediction_artifact": _write_predictions(
                            run.path, session, condition, evaluation
                        ),
                        **gate_provenance,
                        **_flatten_metrics(evaluation),
                    }
                except Exception as error:
                    run.mark_condition_failure(error, **dimensions)

            completed_baselines = [
                name for name in NONFACTORIZED_BASELINES if name in evaluations
            ]
            if baseline_policy == "dev_best":
                selected_baseline = (
                    min(
                        completed_baselines,
                        key=lambda name: (
                            evaluations[name].dev_selection_nll,
                            NONFACTORIZED_BASELINES.index(name),
                        ),
                    )
                    if completed_baselines
                    else None
                )
            else:
                selected_baseline = (
                    baseline_policy if baseline_policy in evaluations else None
                )
            baseline_nll = (
                evaluations[selected_baseline].metrics["low_contrast"].nll
                if selected_baseline is not None
                else float("nan")
            )
            candidate_nll = (
                evaluations["factorized_state"].metrics["low_contrast"].nll
                if "factorized_state" in evaluations
                else float("nan")
            )
            oracle_nll = (
                evaluations["oracle_context_mean"].metrics["low_contrast"].nll
                if "oracle_context_mean" in evaluations
                else float("nan")
            )
            for condition, record in records.items():
                record.update(
                    {
                        "primary_baseline_policy": baseline_policy,
                        "selected_primary_baseline": selected_baseline,
                        "primary_baseline_test_low_contrast_nll": baseline_nll,
                        "factorized_test_low_contrast_nll": candidate_nll,
                        "factorized_nll_gain_vs_primary_baseline": baseline_nll
                        - candidate_nll,
                        "oracle_nll_headroom_from_factorized": candidate_nll
                        - oracle_nll,
                    }
                )
                if condition == "factorized_state":
                    intact = evaluations[condition].metrics["low_contrast"].nll
                    for intervention, values in evaluations[
                        condition
                    ].intervention_metrics.items():
                        record[f"{intervention}_low_contrast_nll_harm"] = (
                            values["low_contrast"].nll - intact
                        )
                run.record(
                    record,
                    **_session_dimensions(eid, animal_id, condition, provenance),
                )
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "IBL state utility experiment",
        "configs/development/exp40_ibl_state_utility_exposed.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
