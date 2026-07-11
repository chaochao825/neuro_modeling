"""Leakage-safe, behavior-only IBL hidden-block belief benchmark."""

from __future__ import annotations

import hashlib
import sys
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
from src.analysis.ibl_behavior_metrics import (
    binary_context_metrics,
    fit_behavior_logistic,
    oracle_ceiling_beliefs,
)
from src.data.ibl_behavior import (
    ExponentialHistoryBelief,
    IBLBehaviorDataError,
    IBLBehaviorSession,
    LearnedCategoricalHMM,
    NoMemoryBelief,
    contiguous_block_split,
    load_ibl_behavior_table,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


CONDITIONS = (
    "no_memory",
    "exponential_history",
    "learned_categorical_hmm",
    "oracle_ceiling",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_dimensions(
    eid: str,
    animal_id: str,
    condition: str,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    dimensions = {
        "condition": condition,
        "eid": eid,
        "session_id": eid,
        "animal_id": animal_id,
        "aggregation_level": "session",
        "statistics_unit": "session",
    }
    for name in (
        "cohort_id",
        "cohort_manifest_sha256",
        "compact_table_sha256",
        "dataset_uuid",
        "dataset_revision",
        "dataset_hash",
        "dataset_qc",
        "bwm_repository_commit",
    ):
        if provenance is not None and provenance.get(name) not in {None, ""}:
            dimensions[name] = provenance[name]
    return dimensions


def _configured_session_specs(config: Mapping[str, Any]) -> list[dict[str, str]]:
    manifest_value = config.get("cohort_manifest")
    raw = config.get("sessions", [])
    if str(config.get("profile", "")) == "formal" and not manifest_value:
        raise ValueError("formal exp11 requires a frozen cohort_manifest")
    if manifest_value and raw:
        raise ValueError("configure either cohort_manifest or sessions, not both")
    if manifest_value:
        manifest_path = Path(str(manifest_value))
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(f"cohort manifest does not exist: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        required = {
            "eid",
            "subject",
            "status",
            "compact_table",
            "cohort_id",
            "compact_table_sha256",
            "dataset_uuid",
            "dataset_revision",
            "dataset_hash",
            "dataset_qc",
            "bwm_repository_commit",
        }
        missing = sorted(required - set(manifest.columns))
        if missing:
            raise ValueError(f"cohort manifest is missing columns: {missing}")
        eligible = manifest[manifest["status"].astype(str).eq("eligible")].copy()
        requirements = dict(config.get("cohort_requirements", {}))
        minimum_sessions = int(requirements.get("minimum_sessions", 20))
        minimum_animals = int(requirements.get("minimum_animals", 5))
        if len(eligible) < minimum_sessions:
            raise ValueError(
                f"cohort has {len(eligible)} eligible sessions; requires "
                f"{minimum_sessions}"
            )
        animal_count = eligible["subject"].astype(str).nunique()
        if animal_count < minimum_animals:
            raise ValueError(
                f"cohort has {animal_count} eligible animals; requires "
                f"{minimum_animals}"
            )
        invalid_qc = ~eligible["dataset_qc"].astype(str).str.upper().isin(
            ["PASS", "WARNING"]
        )
        if invalid_qc.any():
            raise ValueError("eligible cohort rows must have dataset QC PASS/WARNING")
        manifest_sha256 = _file_sha256(manifest_path)
        raw = [
            {
                "path": str((manifest_path.parent / str(row.compact_table)).resolve()),
                "eid": str(row.eid),
                "animal_id": str(row.subject),
                "cohort_id": str(row.cohort_id),
                "cohort_manifest_sha256": manifest_sha256,
                "compact_table_sha256": str(row.compact_table_sha256),
                "dataset_uuid": str(row.dataset_uuid),
                "dataset_revision": str(row.dataset_revision),
                "dataset_hash": str(row.dataset_hash),
                "dataset_qc": str(row.dataset_qc),
                "bwm_repository_commit": str(row.bwm_repository_commit),
            }
            for row in eligible.itertuples(index=False)
        ]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("sessions must be a sequence of local table specifications")
    specs: list[dict[str, str]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ValueError(f"sessions[{index}] must be an object")
        spec = {
            name: str(value.get(name, ""))
            for name in (
                "path",
                "eid",
                "animal_id",
                "cohort_id",
                "cohort_manifest_sha256",
                "compact_table_sha256",
                "dataset_uuid",
                "dataset_revision",
                "dataset_hash",
                "dataset_qc",
                "bwm_repository_commit",
            )
        }
        if any(not spec[name] for name in ("path", "eid", "animal_id")):
            raise ValueError(f"sessions[{index}] requires path, eid, and animal_id")
        path = Path(spec["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        spec["path"] = str(path.resolve())
        specs.append(spec)
    if len({spec["eid"] for spec in specs}) != len(specs):
        raise ValueError("configured eids must be unique")
    return specs


def _load_configured_sessions(
    specs: Sequence[Mapping[str, str]],
) -> list[tuple[str, str, IBLBehaviorSession | Exception, dict[str, str]]]:
    loaded: list[tuple[str, str, IBLBehaviorSession | Exception, dict[str, str]]] = []
    for spec in specs:
        eid = str(spec["eid"])
        animal_id = str(spec["animal_id"])
        provenance = {
            name: str(spec.get(name, ""))
            for name in (
                "cohort_id",
                "cohort_manifest_sha256",
                "compact_table_sha256",
                "dataset_uuid",
                "dataset_revision",
                "dataset_hash",
                "dataset_qc",
                "bwm_repository_commit",
            )
        }
        try:
            path = Path(spec["path"])
            expected_hash = provenance["compact_table_sha256"]
            if expected_hash and _file_sha256(path) != expected_hash:
                raise IBLBehaviorDataError(
                    f"compact table SHA-256 mismatch for session {eid}"
                )
            session: IBLBehaviorSession | Exception = load_ibl_behavior_table(
                path, eid=eid, animal_id=animal_id
            )
        except Exception as error:
            session = error
        loaded.append((eid, animal_id, session, provenance))
    return loaded


def _make_predictions(
    session: IBLBehaviorSession,
    train_indices: np.ndarray,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, dict[str, object]]], dict[str, Exception]]:
    observations = session.observations
    predictions: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
    failures: dict[str, Exception] = {}

    try:
        prediction = NoMemoryBelief().predict(observations)
        predictions["no_memory"] = (
            prediction.beliefs,
            {
                "gate_fit_trial_ids": prediction.fit_trial_ids.tolist(),
                "belief_trajectory_id": prediction.fingerprint,
            },
        )
    except Exception as error:
        failures["no_memory"] = error

    try:
        options = dict(config.get("exponential_history", {}))
        model = ExponentialHistoryBelief(**options).fit(observations, train_indices)
        prediction = model.predict(observations)
        predictions["exponential_history"] = (
            prediction.beliefs,
            {
                "gate_fit_trial_ids": prediction.fit_trial_ids.tolist(),
                "belief_trajectory_id": prediction.fingerprint,
                "selected_decay": model.decay_,
                "train_stimulus_predictive_nll": model.train_predictive_nll_,
            },
        )
    except Exception as error:
        failures["exponential_history"] = error

    try:
        options = dict(config.get("learned_hmm", {}))
        options["seed"] = derive_seed(seed, "exp11", session.eid, "hmm")
        model = LearnedCategoricalHMM(**options).fit(observations, train_indices)
        prediction = model.predict(observations)
        predictions["learned_categorical_hmm"] = (
            prediction.beliefs,
            {
                "gate_fit_trial_ids": prediction.fit_trial_ids.tolist(),
                "belief_trajectory_id": prediction.fingerprint,
                "hmm_initial": model.initial_.tolist(),
                "hmm_transition": model.transition_.tolist(),
                "hmm_left_emission": model.emission_[:, 1].tolist(),
                "hmm_train_log_likelihood": model.train_log_likelihood_,
                "hmm_iterations": model.n_iterations_,
                "hmm_fit_converged": model.converged_,
                "hmm_state_order": "ascending_learned_left_emission",
                "hmm_min_emission_gap": model.emission_gap_,
                "hmm_minimum_required_emission_gap": model.min_emission_gap,
                "hmm_state_separation_identifiable": model.identifiable_,
                "known_context_rate_initialization_used": True,
                "gate_fit_supervision": "task_informed_unsupervised_stimulus_only",
            },
        )
    except Exception as error:
        failures["learned_categorical_hmm"] = error

    # This upper ceiling exists only inside evaluation code.  It is never a
    # learned gate and is explicitly excluded from mechanism-support claims.
    try:
        predictions["oracle_ceiling"] = (
            oracle_ceiling_beliefs(session.context_labels),
            {
                "gate_fit_trial_ids": [],
                "belief_trajectory_id": "evaluation_truth_ceiling",
            },
        )
    except Exception as error:
        failures["oracle_ceiling"] = error
    return predictions, failures


def _evaluate_condition(
    session: IBLBehaviorSession,
    condition: str,
    beliefs: np.ndarray,
    gate_provenance: Mapping[str, object],
    split: Any,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, object]:
    context_indices = split.test_indices[session.context_score_mask[split.test_indices]]
    context = binary_context_metrics(
        beliefs,
        session.context_labels,
        indices=context_indices,
        epsilon=float(config.get("context_epsilon", 1e-9)),
        n_bins=int(config.get("calibration_bins", 10)),
    )
    behavior_options = dict(config.get("behavior_model", {}))
    behavior = fit_behavior_logistic(
        session,
        beliefs,
        split,
        seed=derive_seed(seed, "exp11", session.eid, condition, "readout"),
        **behavior_options,
    )
    oracle = condition == "oracle_ceiling"
    learned_hmm_valid = bool(
        gate_provenance.get("hmm_fit_converged", False)
        and gate_provenance.get("hmm_state_separation_identifiable", False)
    )
    condition_roles = {
        "no_memory": "uniform_belief_strong_causal_history_control",
        "exponential_history": "causal_history_belief_control",
        "learned_categorical_hmm": "task_informed_unsupervised_learned_gate",
        "oracle_ceiling": "evaluation_only_truth_ceiling",
    }
    return {
        "status": "complete",
        "profile": str(config.get("profile", "unspecified")),
        "training_algorithm": "ibl_causal_hidden_block_belief_v1",
        "used_autograd": False,
        "behavior_only_benchmark": True,
        "neural_activity_analyzed": False,
        "split_unit": "contiguous_probabilityLeft_block",
        "split_is_chronological": True,
        "train_dev_test_blocks_disjoint": True,
        "preprocessing_fit_train_only": True,
        "dev_used_for_selection": False,
        "frozen_parameters_online_filter_state": not oracle,
        "online_state_updates_from_dev_and_past_test_observations": not oracle,
        "algorithmic_seed_nested_within_session": True,
        "algorithmic_seed_is_statistical_unit": False,
        "session_seed_aggregation_required": True,
        "primary_context_state_count": 2,
        "unbiased_0p5_role": "initial_burn_in_excluded_from_context_scoring",
        "unbiased_0p5_used_as_hidden_state": False,
        "condition_role": condition_roles[condition],
        "no_memory_scope": "belief_gate_only_not_behavior_history",
        "gate_uses_stimulus_side_only": not oracle,
        "gate_uses_current_trial_stimulus": False if not oracle else None,
        "gate_uses_future_stimuli": False if not oracle else None,
        "gate_reset_at_true_boundaries": False if not oracle else None,
        "gate_fit_accessed_probabilityLeft": None if oracle else False,
        "gate_test_accessed_probabilityLeft": False if not oracle else True,
        "behavior_readout_fit_accessed_probabilityLeft": oracle,
        "behavior_readout_test_accessed_probabilityLeft": oracle,
        "truth_used_for_fold_grouping": True,
        "probabilityLeft_access_scope": (
            "evaluation_ceiling_and_grouped_scoring"
            if oracle
            else "grouped_split_and_frozen_prediction_scoring_only"
        ),
        "eligible_for_context_inference_support": condition == "learned_categorical_hmm"
        and learned_hmm_valid,
        "eligible_for_behavior_pipeline_evaluation": condition
        == "learned_categorical_hmm",
        "oracle_is_evaluation_only": oracle,
        "oracle_behavior_ceiling_uses_true_context": oracle,
        "context_nll": context.nll,
        "context_brier": context.brier,
        "context_accuracy": context.accuracy,
        "context_ece": context.expected_calibration_error,
        "context_test_trial_count": context.n_trials,
        "behavior_log_loss": behavior.log_loss,
        "behavior_accuracy": behavior.accuracy,
        "behavior_balanced_accuracy": behavior.balanced_accuracy,
        "behavior_roc_auc": behavior.roc_auc,
        "behavior_mcfadden_pseudo_r2": behavior.mcfadden_pseudo_r2,
        "behavior_feature_count": behavior.feature_count,
        "behavior_readout_fit_trial_ids": behavior.fit_trial_ids.tolist(),
        "behavior_test_trial_ids": behavior.test_trial_ids.tolist(),
        "session_trial_count": int(session.trial_ids.size),
        "session_block_count": session.n_blocks,
        "valid_choice_count": int(np.sum(session.choice_valid)),
        "official_bwm_mask_present": session.official_bwm_mask_present,
        "official_bwm_analysis_trial_count": int(np.sum(session.analysis_mask)),
        "unbiased_burn_in_trial_count": int(
            np.count_nonzero(session.context_labels < 0)
        ),
        "source_trial_indices_contiguous": bool(
            np.all(np.diff(session.source_trial_indices) == 1)
        ),
        "train_trial_count": int(split.train_indices.size),
        "dev_trial_count": int(split.dev_indices.size),
        "test_trial_count": int(split.test_indices.size),
        "train_block_count": int(split.train_block_ids.size),
        "dev_block_count": int(split.dev_block_ids.size),
        "test_block_count": int(split.test_block_ids.size),
        "split_id": split.fingerprint,
        "observation_tape_id": session.observations.fingerprint,
        **dict(gate_provenance),
    }


def run_seed(
    config: dict[str, Any],
    seed: int,
    results_root: str,
    *,
    sessions: Sequence[IBLBehaviorSession] | None = None,
) -> Path:
    """Run all registered session/condition cells, preserving every failure."""

    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "ibl_causal_hidden_block_belief_v1",
        "used_autograd": False,
        "parent_checkpoint": None,
        "data_access": "local_trial_tables_only",
    }
    with ExperimentRun(
        "exp11_ibl_behavior_belief", seed, run_config, results_root=results_root
    ) as run:
        registered = False
        planned_dimensions: list[dict[str, object]] = []
        try:
            if sessions is None:
                specs = _configured_session_specs(config)
                candidates = _load_configured_sessions(specs)
            else:
                if not sessions:
                    raise ValueError("injected sessions cannot be empty")
                if not all(isinstance(item, IBLBehaviorSession) for item in sessions):
                    raise TypeError("every injected session must be IBLBehaviorSession")
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
                raise IBLBehaviorDataError(
                    "no local IBL behavior tables configured; network download is disabled"
                )
            if len({eid for eid, _, _, _ in candidates}) != len(candidates):
                raise ValueError("session eids must be unique")
            planned_dimensions = [
                _session_dimensions(eid, animal_id, condition, provenance)
                for eid, animal_id, _, provenance in candidates
                for condition in CONDITIONS
            ]
            run.register_conditions(planned_dimensions)
            registered = True
            split_options = dict(config.get("split", {}))
            if config.get("cohort_manifest"):
                manifest_path = Path(str(config["cohort_manifest"]))
                if not manifest_path.is_absolute():
                    manifest_path = PROJECT_ROOT / manifest_path
                snapshot = run.path / "cohort_manifest.csv"
                snapshot.write_bytes(manifest_path.read_bytes())
                (run.path / "cohort_manifest.sha256").write_text(
                    _file_sha256(snapshot) + "\n", encoding="ascii"
                )
        except Exception as error:
            if registered:
                for dimensions in planned_dimensions:
                    run.mark_condition_failure(error, **dimensions)
            else:
                run.register_conditions([{"condition": "setup"}])
                run.mark_condition_failure(error, condition="setup")
            return run.path

        for eid, animal_id, candidate, cohort_provenance in candidates:
            if isinstance(candidate, Exception):
                for condition in CONDITIONS:
                    run.mark_condition_failure(
                        candidate,
                        **_session_dimensions(
                            eid, animal_id, condition, cohort_provenance
                        ),
                    )
                continue
            session = candidate
            try:
                if (
                    str(config.get("profile", "")) == "formal"
                    and not session.official_bwm_mask_present
                ):
                    raise IBLBehaviorDataError(
                        "formal exp11 requires the official BWM trial mask"
                    )
                split = contiguous_block_split(session.block_ids, **split_options)
            except Exception as error:
                for condition in CONDITIONS:
                    run.mark_condition_failure(
                        error,
                        **_session_dimensions(
                            eid, animal_id, condition, cohort_provenance
                        ),
                    )
                continue

            predictions, failures = _make_predictions(
                session,
                split.train_indices,
                config=config,
                seed=seed,
            )
            evaluated: dict[str, dict[str, object]] = {}
            for condition in CONDITIONS:
                if condition in failures:
                    continue
                try:
                    beliefs, gate_provenance = predictions[condition]
                    evaluated[condition] = _evaluate_condition(
                        session,
                        condition,
                        beliefs,
                        gate_provenance,
                        split,
                        config=config,
                        seed=seed,
                    )
                except Exception as error:
                    failures[condition] = error

            baseline_loss = (
                float(evaluated["no_memory"]["behavior_log_loss"])
                if "no_memory" in evaluated
                else np.nan
            )
            for condition in CONDITIONS:
                dimensions = _session_dimensions(
                    eid, animal_id, condition, cohort_provenance
                )
                if condition in failures:
                    run.mark_condition_failure(failures[condition], **dimensions)
                    continue
                metrics = evaluated[condition]
                condition_loss = float(metrics["behavior_log_loss"])
                metrics["behavior_log_loss_gain_vs_no_memory"] = (
                    baseline_loss - condition_loss
                    if np.isfinite(baseline_loss)
                    else np.nan
                )
                run.record(metrics, **dimensions)
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "IBL behavior hidden-block belief benchmark",
        "configs/smoke/exp11_ibl_behavior_belief.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
