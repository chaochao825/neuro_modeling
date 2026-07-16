"""Fail-closed real-data comparison for shared compositional dynamics.

Exp25 consumes only a byte-verified canonical conversion of the official
CompositionalTasks Figshare release.  There is deliberately no synthetic,
download, or inferred-schema fallback in this experiment entry point.

The five registered model families share the same outer folds, past-only
belief estimator, nested dimension candidates, and session data:

``common``
    Common state and input dynamics.
``input-gated``
    Belief-dependent input routing with common state dynamics.
``state-gated``
    Belief-dependent low-rank state dynamics with common input routing.
``fully-gated``
    Belief-dependent input and state dynamics.
``separate-task``
    Fit-only task-specific operators; held-out task truth is never materialized
    and scoring uses the same past-only belief probabilities as shared models.

All neural preprocessing, observation models, dynamics, null rates, and belief
classifier parameters are fit inside the applicable training fold.  The
reported likelihood is exact conditional one-step Poisson likelihood, not a
full marginal PLDS likelihood and not an autonomous rollout score.

The current encoder fits PCA separately within each session.  Those latent
coordinates are not aligned or identifiable across sessions, so a verified
multi-session dataset is retained but every would-be shared-dynamics
comparison is marked scientifically invalid.  No held-out likelihood is
reported until a train-only shared basis or explicit identifiable alignment is
implemented.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.compositional_tasks_loader as compositional
from experiments.common import (
    PROJECT_ROOT,
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.models.belief_controlled_plds import (
    DEFAULT_LATENT_DIMENSIONS,
    LATENT_COORDINATE_SYSTEM,
    SESSION_BASIS_ALIGNMENT,
    SHARED_CROSS_SESSION_DYNAMICS_IDENTIFIABLE,
    BeliefControlledCountSession,
    BeliefControlledPLDS,
    TrialFold,
    select_latent_dimension,
)
from src.utils.artifacts import ExperimentRun


EXPERIMENT = "exp25_compositional_tasks_real"
PROTOCOL_VERSION = "exp25_official_canonical_v2_fold_safe_fail_closed_basis"
FAMILIES = (
    "common",
    "input-gated",
    "state-gated",
    "fully-gated",
    "separate-task",
)
PROTOCOLS = (
    "leave-one-block-out",
    "leave-one-composition-out",
    "unseen-stimulus-action-composition",
    "cross-session-transfer",
)
_HELDOUT_TASK_SENTINEL = "__HELDOUT_TASK_TRUTH_NOT_MATERIALIZED__"
_CROSS_SESSION_UNSUPPORTED = (
    "cross-session transfer is scientifically ineligible in the current "
    "implementation: BeliefControlledPLDS has session-specific observation "
    "matrices but no hierarchical train-session-to-unseen-session observation "
    "map. Fitting that map on held-out neural counts would violate train-only "
    "preprocessing, so this protocol fails closed."
)
_UNALIGNED_SHARED_DYNAMICS = (
    "shared multi-session dynamics are scientifically ineligible in the current "
    "implementation: every session uses an independently fit train-only PCA "
    "coordinate system, but no train-only shared basis or identifiable alignment "
    "maps those coordinates into a common latent space. Pooling them would make "
    "the shared operator coordinate-dependent, so all multi-session model "
    "comparisons fail closed."
)


@dataclass(frozen=True, slots=True)
class _FoldSpec:
    protocol: str
    fold_id: str
    heldout_values: tuple[object, ...]
    train_indices_by_session: Mapping[str, tuple[int, ...]]
    test_indices_by_session: Mapping[str, tuple[int, ...]]

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.train_indices_by_session))


@dataclass(frozen=True, slots=True)
class _NestedSelection:
    selected_dimension: int | None
    candidates: tuple[dict[str, object], ...]
    inner_fold_ids: tuple[str, ...]


def _planned_conditions() -> list[dict[str, object]]:
    return [
        {
            "condition": f"{protocol}:{family}",
            "protocol": protocol,
            "model_family": family,
            "evaluation_level": "animal_session",
        }
        for protocol in PROTOCOLS
        for family in FAMILIES
    ]


def _resolve_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_config(config: Mapping[str, Any]) -> None:
    if str(config.get("data_mode")) != "official_canonical_only":
        raise ValueError(
            "Exp25 accepts only data_mode='official_canonical_only'; "
            "synthetic and inferred-schema fallbacks are forbidden"
        )
    protocols = tuple(str(value) for value in config.get("protocols", PROTOCOLS))
    if len(set(protocols)) != len(protocols) or set(protocols) != set(PROTOCOLS):
        raise ValueError(f"protocols must contain exactly {PROTOCOLS}")
    dimensions = tuple(int(value) for value in config["candidate_latent_dims"])
    if dimensions != DEFAULT_LATENT_DIMENSIONS:
        raise ValueError(
            "candidate_latent_dims must be exactly [2, 4, 8, 16] "
            "for the registered nested-CV comparison"
        )
    if int(config.get("minimum_sessions", 2)) < 1:
        raise ValueError("minimum_sessions must be positive")
    if int(config.get("minimum_animals", 2)) < 1:
        raise ValueError("minimum_animals must be positive")
    for key in ("max_outer_folds", "max_inner_folds"):
        value = config.get(key)
        if value is not None and int(value) < 1:
            raise ValueError(f"{key} must be null or a positive integer")
    belief = dict(config["belief"])
    if not belief.get("cue_columns") or not belief.get("behavior_columns"):
        raise ValueError("belief cue_columns and behavior_columns must be non-empty")
    model = dict(config["model"])
    if int(model.get("gate_rank", 2)) < 1:
        raise ValueError("model gate_rank must be positive")
    if int(model.get("max_irls", 40)) < 1:
        raise ValueError("model max_irls must be positive")


def _load_verified_dataset(
    config: Mapping[str, Any],
) -> compositional.CompositionalDataset:
    root = _resolve_path(config["data_root"])
    manifest = _resolve_path(config["manifest_path"])
    digest = str(config["expected_manifest_sha256"])
    source = compositional.validate_official_compositional_source(root)
    if not source.source_verified:
        raise RuntimeError("official Figshare source verification did not complete")
    if not manifest.is_file():
        raise RuntimeError(
            "official Figshare source bytes were verified, but canonical "
            "trial-level neural counts are absent. Exp25 requires a reviewed "
            "hash-pinned trials.csv, units.csv, session NPZ bundle, conversion "
            f"file, and manifest; missing {manifest}"
        )
    if digest == "0" * 64:
        raise RuntimeError(
            "expected_manifest_sha256 is an unresolved all-zero placeholder; "
            "the canonical conversion must be reviewed and hash-pinned"
        )
    dataset = compositional.load_compositional_tasks(
        root,
        manifest,
        expected_manifest_sha256=digest,
    )
    n_sessions = len(dataset.sessions)
    n_animals = len({session.animal_id for session in dataset.sessions})
    if n_sessions < int(config.get("minimum_sessions", 2)):
        raise RuntimeError(
            f"canonical cohort has {n_sessions} sessions, below the registered minimum"
        )
    if n_animals < int(config.get("minimum_animals", 2)):
        raise RuntimeError(
            f"canonical cohort has {n_animals} animals, below the registered minimum"
        )
    return dataset


def _session_row_indices(
    dataset: compositional.CompositionalDataset,
) -> dict[str, np.ndarray]:
    lookup = {
        (str(row.session_id), str(row.trial_id)): int(index)
        for index, row in dataset.trials.iterrows()
    }
    result: dict[str, np.ndarray] = {}
    for session in dataset.sessions:
        try:
            indices = np.asarray(
                [
                    lookup[(session.session_id, str(trial_id))]
                    for trial_id in session.trial_ids
                ],
                dtype=int,
            )
        except KeyError as error:
            raise RuntimeError(
                "canonical session arrays and trial table are not aligned"
            ) from error
        result[session.session_id] = indices
    return result


def _fold_from_indices(
    dataset: compositional.CompositionalDataset,
    *,
    protocol: str,
    fold_number: int,
    heldout_values: tuple[object, ...],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> _FoldSpec | None:
    trials = dataset.trials
    train_set = set(np.asarray(train_indices, dtype=int).tolist())
    test_set = set(np.asarray(test_indices, dtype=int).tolist())
    train_by_session: dict[str, tuple[int, ...]] = {}
    test_by_session: dict[str, tuple[int, ...]] = {}
    for session in dataset.sessions:
        rows = trials.index[
            trials["session_id"].astype(str).eq(session.session_id)
        ].to_numpy(dtype=int)
        train = tuple(int(value) for value in rows if int(value) in train_set)
        test = tuple(int(value) for value in rows if int(value) in test_set)
        if train and test:
            train_by_session[session.session_id] = train
            test_by_session[session.session_id] = test
    if not train_by_session:
        return None
    return _FoldSpec(
        protocol=protocol,
        fold_id=f"{protocol}-fold-{fold_number:03d}",
        heldout_values=heldout_values,
        train_indices_by_session=train_by_session,
        test_indices_by_session=test_by_session,
    )


def _unseen_pair_splits(
    dataset: compositional.CompositionalDataset,
) -> list[tuple[tuple[object, ...], np.ndarray, np.ndarray]]:
    trials = dataset.trials
    pairs = list(
        zip(
            trials["stimulus_id"].astype(str),
            trials["action_id"].astype(str),
            strict=True,
        )
    )
    unique = sorted(set(pairs), key=repr)
    if len(unique) < 2:
        raise compositional.CompositionalTasksDataError(
            "unseen stimulus-action CV requires at least two complete combinations"
        )
    result = []
    for heldout in unique:
        mask = np.asarray([value == heldout for value in pairs], dtype=bool)
        result.append(
            (
                (heldout,),
                np.flatnonzero(~mask),
                np.flatnonzero(mask),
            )
        )
    return result


def _outer_folds(
    dataset: compositional.CompositionalDataset,
    protocol: str,
    *,
    max_folds: int | None,
) -> tuple[_FoldSpec, ...]:
    if protocol == "leave-one-block-out":
        raw = [
            (
                split.heldout_values,
                split.train_indices,
                split.test_indices,
            )
            for split in compositional.leave_one_block_out_splits(dataset.trials)
        ]
    elif protocol == "leave-one-composition-out":
        raw = [
            (
                split.heldout_values,
                split.train_indices,
                split.test_indices,
            )
            for split in compositional.leave_one_composition_out_splits(
                dataset.trials
            )
        ]
    elif protocol == "unseen-stimulus-action-composition":
        raw = _unseen_pair_splits(dataset)
    else:
        raise ValueError(f"unsupported fold construction protocol: {protocol}")
    folds = []
    for index, (heldout, train, test) in enumerate(raw):
        fold = _fold_from_indices(
            dataset,
            protocol=protocol,
            fold_number=index,
            heldout_values=tuple(heldout),
            train_indices=np.asarray(train, dtype=int),
            test_indices=np.asarray(test, dtype=int),
        )
        if fold is not None:
            folds.append(fold)
    if max_folds is not None:
        folds = folds[:max_folds]
    if not folds:
        raise RuntimeError(f"{protocol} produced no complete train/test session folds")
    return tuple(folds)


def _inner_folds(
    dataset: compositional.CompositionalDataset,
    outer: _FoldSpec,
    *,
    max_folds: int | None,
) -> tuple[_FoldSpec, ...]:
    blocks_by_session: dict[str, tuple[str, ...]] = {}
    for session_id, indices in outer.train_indices_by_session.items():
        rows = dataset.trials.iloc[list(indices)].copy()
        rows["_block"] = rows["block_id"].astype(str)
        rows["_order"] = rows["trial_order"].astype(float)
        block_order = (
            rows.groupby("_block", sort=False)["_order"].min().sort_values()
        )
        blocks = tuple(str(value) for value in block_order.index)
        if len(blocks) < 2:
            raise RuntimeError(
                f"nested block CV needs at least two outer-train blocks in {session_id}"
            )
        blocks_by_session[session_id] = blocks
    count = min(len(value) for value in blocks_by_session.values())
    if max_folds is not None:
        count = min(count, max_folds)
    result = []
    for fold_index in range(count):
        train_by_session: dict[str, tuple[int, ...]] = {}
        test_by_session: dict[str, tuple[int, ...]] = {}
        heldout: list[object] = []
        for session_id, indices in outer.train_indices_by_session.items():
            heldout_block = blocks_by_session[session_id][fold_index]
            heldout.append((session_id, heldout_block))
            rows = dataset.trials.iloc[list(indices)]
            mask = rows["block_id"].astype(str).eq(heldout_block).to_numpy()
            array = np.asarray(indices, dtype=int)
            train = tuple(array[~mask].tolist())
            test = tuple(array[mask].tolist())
            if not train or not test:
                raise RuntimeError("inner block fold is empty")
            train_by_session[session_id] = train
            test_by_session[session_id] = test
        result.append(
            _FoldSpec(
                protocol=f"{outer.protocol}:nested-block",
                fold_id=f"{outer.fold_id}-inner-{fold_index:02d}",
                heldout_values=tuple(heldout),
                train_indices_by_session=train_by_session,
                test_indices_by_session=test_by_session,
            )
        )
    return tuple(result)


def _trial_folds(
    dataset: compositional.CompositionalDataset,
    fold: _FoldSpec,
) -> dict[str, TrialFold]:
    trials = dataset.trials
    result = {}
    for session_id in fold.session_ids:
        train_ids = tuple(
            trials.iloc[index]["trial_id"]
            for index in fold.train_indices_by_session[session_id]
        )
        test_ids = tuple(
            trials.iloc[index]["trial_id"]
            for index in fold.test_indices_by_session[session_id]
        )
        result[session_id] = TrialFold(train_ids, test_ids)
    return result


def _allowed_trial_ids(
    dataset: compositional.CompositionalDataset,
    outer: _FoldSpec,
) -> dict[str, tuple[object, ...]]:
    return {
        session_id: tuple(
            dataset.trials.iloc[index]["trial_id"]
            for index in outer.train_indices_by_session[session_id]
        )
        for session_id in outer.session_ids
    }


def _build_controlled_sessions(
    dataset: compositional.CompositionalDataset,
    *,
    selected_session_ids: Sequence[str],
    fit_indices_by_session: Mapping[str, Sequence[int]],
    belief_config: Mapping[str, Any],
) -> tuple[
    tuple[BeliefControlledCountSession, ...],
    tuple[BeliefControlledCountSession, ...],
    dict[str, object],
]:
    session_ids = tuple(selected_session_ids)
    fit_indices = np.asarray(
        sorted(
            {
                int(index)
                for session_id in session_ids
                for index in fit_indices_by_session[session_id]
            }
        ),
        dtype=int,
    )
    estimator = compositional.PastOnlyBeliefEstimator(
        cue_columns=tuple(str(value) for value in belief_config["cue_columns"]),
        behavior_columns=tuple(
            str(value) for value in belief_config["behavior_columns"]
        ),
        numeric_columns=tuple(
            str(value) for value in belief_config.get("numeric_columns", ())
        ),
        group_columns=("session_id",),
        order_column="trial_order",
        ridge=float(belief_config.get("ridge", 1.0)),
        temperature=float(belief_config.get("temperature", 1.0)),
    ).fit(
        dataset.trials,
        fit_indices,
        label_column=str(belief_config.get("fit_label_column", "composition_id")),
    )
    row_indices = _session_row_indices(dataset)
    session_lookup = {session.session_id: session for session in dataset.sessions}
    fit_set = set(fit_indices.tolist())
    fitted: list[BeliefControlledCountSession] = []
    receipts = []
    label_column = str(belief_config.get("fit_label_column", "composition_id"))
    for session_id in session_ids:
        source = session_lookup[session_id]
        rows = row_indices[session_id]
        trajectory = estimator.predict(dataset.trials, rows)
        task_ids = np.full(
            source.counts.shape[0],
            _HELDOUT_TASK_SENTINEL,
            dtype=object,
        )
        for position, row_index in enumerate(rows):
            if int(row_index) in fit_set:
                task_ids[position] = dataset.trials.iloc[int(row_index)][label_column]
        fitted.append(
            BeliefControlledCountSession(
                session_id=source.session_id,
                animal_id=source.animal_id,
                counts=source.counts,
                inputs=source.inputs,
                beliefs=trajectory.probabilities,
                belief_labels=trajectory.classes,
                trial_ids=source.trial_ids.astype(object),
                belief_receipt=trajectory.receipt,
                task_ids=task_ids,
            )
        )
        receipts.append(trajectory.receipt)
    fitted_tuple = tuple(fitted)
    truth_free = tuple(replace(session, task_ids=None) for session in fitted_tuple)
    audit = {
        "belief_fit_trial_count": int(fit_indices.size),
        "belief_classes": [repr(value) for value in estimator.classes_],
        "belief_source_columns": list(estimator.source_columns),
        "belief_feature_lag_trials": 1,
        "belief_fit_history_scope": "training_rows_only_within_group",
        "belief_prediction_history_scope": (
            "all_causally_prior_rows_within_group"
        ),
        "belief_fit_preprocessing_heldout_independent": all(
            receipt.fit_preprocessing_heldout_independent
            for receipt in receipts
        ),
        "belief_uses_current_trial_fields": any(
            receipt.uses_current_trial_fields for receipt in receipts
        ),
        "belief_uses_future_trials": any(
            receipt.uses_future_trials for receipt in receipts
        ),
        "belief_accessed_test_truth": any(
            receipt.accessed_test_truth for receipt in receipts
        ),
        "belief_checkpoint_sha256": estimator.checkpoint_sha256_,
        "heldout_task_truth_materialized": False,
    }
    return fitted_tuple, truth_free, audit


def _model_options(config: Mapping[str, Any]) -> dict[str, object]:
    model = dict(config["model"])
    return {
        "gate_rank": int(model.get("gate_rank", 2)),
        "ridge": float(model.get("ridge", 1e-3)),
        "poisson_ridge": float(model.get("poisson_ridge", 1e-3)),
        "max_irls": int(model.get("max_irls", 40)),
    }


def _select_dimension(
    dataset: compositional.CompositionalDataset,
    outer: _FoldSpec,
    *,
    family: str,
    config: Mapping[str, Any],
) -> _NestedSelection:
    dimensions = tuple(int(value) for value in config["candidate_latent_dims"])
    inner_specs = _inner_folds(
        dataset,
        outer,
        max_folds=(
            None
            if config.get("max_inner_folds") is None
            else int(config["max_inner_folds"])
        ),
    )
    by_dimension: dict[int, list[float | None]] = {
        dimension: [] for dimension in dimensions
    }
    errors: dict[int, list[str | None]] = {
        dimension: [] for dimension in dimensions
    }
    for inner in inner_specs:
        try:
            sessions, _, _ = _build_controlled_sessions(
                dataset,
                selected_session_ids=outer.session_ids,
                fit_indices_by_session=inner.train_indices_by_session,
                belief_config=dict(config["belief"]),
            )
            selection = select_latent_dimension(
                family,
                sessions,
                [_trial_folds(dataset, inner)],
                candidate_dimensions=dimensions,
                allowed_trial_ids=_allowed_trial_ids(dataset, outer),
                **_model_options(config),
            )
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            for dimension in dimensions:
                by_dimension[dimension].append(None)
                errors[dimension].append(reason)
            continue
        candidates = {
            candidate.latent_dim: candidate for candidate in selection.candidates
        }
        for dimension in dimensions:
            candidate = candidates[dimension]
            by_dimension[dimension].append(candidate.mean_log_likelihood)
            errors[dimension].append(
                None
                if candidate.eligible
                else next(
                    (
                        value
                        for value in candidate.fold_errors
                        if value is not None
                    ),
                    "ineligible nested dimension",
                )
            )
    records = []
    eligible = []
    for dimension in dimensions:
        values = by_dimension[dimension]
        complete = bool(values) and all(value is not None for value in values)
        mean_score = (
            float(np.mean([float(value) for value in values]))
            if complete
            else None
        )
        records.append(
            {
                "latent_dim": dimension,
                "inner_mean_log_likelihoods": values,
                "inner_errors": errors[dimension],
                "mean_log_likelihood": mean_score,
                "eligible": complete,
            }
        )
        if mean_score is not None:
            eligible.append((dimension, mean_score))
    selected = (
        min(eligible, key=lambda item: (-item[1], item[0]))[0]
        if eligible
        else None
    )
    return _NestedSelection(
        selected_dimension=selected,
        candidates=tuple(records),
        inner_fold_ids=tuple(fold.fold_id for fold in inner_specs),
    )


def _source_receipt_metrics(
    dataset: compositional.CompositionalDataset,
) -> dict[str, object]:
    receipt = dataset.receipt
    return {
        "official_source_verified": receipt.source_verified,
        "canonical_conversion_verified": receipt.canonical_verified,
        "canonical_manifest_sha256": receipt.manifest_sha256,
        "canonical_manifest_schema": receipt.manifest_schema,
        "canonical_conversion_code_sha256": receipt.conversion_code_sha256,
        "official_file_md5": dict(receipt.official_file_md5),
        "canonical_file_sha256": dict(receipt.canonical_file_sha256),
        "figshare_doi": compositional.FIGSHARE_DOI,
        "code_doi": compositional.CODE_DOI,
    }


def _evaluate_outer_fold(
    dataset: compositional.CompositionalDataset,
    outer: _FoldSpec,
    *,
    family: str,
    config: Mapping[str, Any],
) -> dict[str, object]:
    selection = _select_dimension(dataset, outer, family=family, config=config)
    if selection.selected_dimension is None:
        raise RuntimeError(
            "all registered nested latent dimensions failed; "
            f"candidates={selection.candidates!r}"
        )
    fit_sessions, score_sessions, belief_audit = _build_controlled_sessions(
        dataset,
        selected_session_ids=outer.session_ids,
        fit_indices_by_session=outer.train_indices_by_session,
        belief_config=dict(config["belief"]),
    )
    folds = _trial_folds(dataset, outer)
    model = BeliefControlledPLDS(
        family,
        selection.selected_dimension,
        **_model_options(config),
    ).fit(fit_sessions, folds)
    score = model.score(score_sessions, folds)
    animal_lookup = {
        session.session_id: session.animal_id for session in score_sessions
    }
    per_session = [
        {
            **asdict(value),
            "animal_id": animal_lookup[value.session_id],
            "heldout_log_likelihood_gain_vs_null": (
                value.log_likelihood - value.null_log_likelihood
            ),
        }
        for value in score.per_session
    ]
    if score.heldout_truth_used:
        raise RuntimeError("model score reports held-out task-truth access")
    if (
        belief_audit["belief_uses_current_trial_fields"]
        or belief_audit["belief_uses_future_trials"]
        or belief_audit["belief_accessed_test_truth"]
    ):
        raise RuntimeError("belief audit failed its past-only causal contract")
    return {
        "status": "complete",
        "selected_latent_dim": selection.selected_dimension,
        "candidate_latent_dims": list(DEFAULT_LATENT_DIMENSIONS),
        "nested_dimension_candidates": list(selection.candidates),
        "nested_inner_fold_ids": list(selection.inner_fold_ids),
        "nested_selection_scope": "outer_training_blocks_only",
        "heldout_log_likelihood": score.log_likelihood,
        "heldout_null_log_likelihood": score.null_log_likelihood,
        "heldout_log_likelihood_gain_vs_null": (
            score.log_likelihood - score.null_log_likelihood
        ),
        "heldout_mean_log_likelihood": score.mean_log_likelihood,
        "heldout_nll_per_count": score.nll_per_count,
        "heldout_bits_per_spike": score.bits_per_spike,
        "heldout_observation_count": score.n_observations,
        "heldout_spike_count": score.n_spikes,
        "parameter_count": score.parameter_count,
        "parameter_breakdown": model.parameter_breakdown(),
        "likelihood_kind": score.likelihood_kind,
        "full_marginal_plds": score.full_marginal_plds,
        "heldout_truth_used": score.heldout_truth_used,
        "one_step_conditional_prediction": True,
        "autonomous_forecast": False,
        "train_only_preprocessing": True,
        "time_points_randomly_split": False,
        "outer_split_unit": (
            "complete_trial_blocks"
            if outer.protocol == "leave-one-block-out"
            else "complete_trials_grouped_by_registered_composition"
        ),
        "heldout_values": [repr(value) for value in outer.heldout_values],
        "session_ids": list(outer.session_ids),
        "animal_ids": sorted(set(animal_lookup.values())),
        "independent_statistical_units": ["animal", "session"],
        "neuron_as_independent_repeat": False,
        "time_bin_as_independent_repeat": False,
        "per_session": per_session,
        **belief_audit,
        **_source_receipt_metrics(dataset),
    }


def _aggregate_successes(
    successes: Sequence[dict[str, object]],
    *,
    planned_folds: int,
    failed_folds: int,
) -> dict[str, object]:
    total_ll = float(
        sum(float(value["heldout_log_likelihood"]) for value in successes)
    )
    total_null = float(
        sum(float(value["heldout_null_log_likelihood"]) for value in successes)
    )
    observations = int(
        sum(int(value["heldout_observation_count"]) for value in successes)
    )
    spikes = int(sum(int(value["heldout_spike_count"]) for value in successes))
    bits = (
        (total_ll - total_null) / (spikes * np.log(2.0))
        if spikes > 0
        else float("nan")
    )
    animals = sorted(
        {
            str(animal)
            for value in successes
            for animal in value["animal_ids"]  # type: ignore[index]
        }
    )
    sessions = sorted(
        {
            str(session)
            for value in successes
            for session in value["session_ids"]  # type: ignore[index]
        }
    )
    return {
        "status": "complete" if failed_folds == 0 else "complete_with_failures",
        "outer_folds_planned": planned_folds,
        "outer_folds_complete": len(successes),
        "outer_folds_failed": failed_folds,
        "heldout_log_likelihood": total_ll,
        "heldout_null_log_likelihood": total_null,
        "heldout_log_likelihood_gain_vs_null": total_ll - total_null,
        "heldout_mean_log_likelihood": total_ll / observations,
        "heldout_nll_per_count": -total_ll / observations,
        "heldout_bits_per_spike": float(bits),
        "heldout_observation_count": observations,
        "heldout_spike_count": spikes,
        "parameter_count_by_fold": [
            int(value["parameter_count"]) for value in successes
        ],
        "selected_latent_dim_by_fold": [
            int(value["selected_latent_dim"]) for value in successes
        ],
        "animal_ids": animals,
        "session_ids": sessions,
        "n_animals": len(animals),
        "n_sessions": len(sessions),
        "independent_statistical_units": ["animal", "session"],
        "absolute_performance_reported_separately": True,
        "relative_null_gain_reported_separately": True,
        "train_only_preprocessing": True,
        "heldout_truth_used": False,
        "one_step_conditional_prediction": True,
        "autonomous_forecast": False,
    }


def _data_failure(error: BaseException) -> RuntimeError:
    return RuntimeError(
        "Exp25 official/canonical data validation failed closed; no synthetic "
        f"or inferred-schema substitute was used: {type(error).__name__}: {error}"
    )


def run_seed(
    config: Mapping[str, Any],
    seed: int,
    results_root: str | Path,
) -> Path:
    """Run all registered real-data families while retaining every failure."""

    initialize_seed(seed)
    run_config = {
        **dict(config),
        "protocol_version": PROTOCOL_VERSION,
        "training_algorithm": (
            "nested_group_cv_past_only_belief_conditional_poisson_dynamics"
        ),
        "used_autograd": False,
        "parent_checkpoint": None,
        "synthetic_fallback": False,
        "test_task_truth_available_to_controller": False,
    }
    planned = _planned_conditions()
    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
    ) as run:
        run.register_conditions(planned)
        try:
            _validate_config(config)
            dataset = _load_verified_dataset(config)
        except Exception as error:
            failure = _data_failure(error)
            for condition in planned:
                run.mark_condition_failure(
                    failure,
                    **condition,
                    stage="official_canonical_data_validation",
                )
            return run.path

        if len(dataset.sessions) > 1:
            source_metrics = _source_receipt_metrics(dataset)
            animals = sorted({session.animal_id for session in dataset.sessions})
            sessions = sorted(session.session_id for session in dataset.sessions)
            for condition in planned:
                protocol = str(condition["protocol"])
                reason = _UNALIGNED_SHARED_DYNAMICS
                if protocol == "cross-session-transfer":
                    reason = f"{reason} {_CROSS_SESSION_UNSUPPORTED}"
                run.mark_condition_invalid(
                    reason,
                    **condition,
                    stage="latent_coordinate_identifiability",
                    latent_coordinate_system=LATENT_COORDINATE_SYSTEM,
                    session_basis_alignment=SESSION_BASIS_ALIGNMENT,
                    shared_cross_session_dynamics_identifiable=(
                        SHARED_CROSS_SESSION_DYNAMICS_IDENTIFIABLE
                    ),
                    shared_cross_session_dynamics_claimed=False,
                    heldout_likelihood_comparison_valid=False,
                    train_only_shared_basis_implemented=False,
                    unseen_session_observation_map_implemented=False,
                    session_ids=sessions,
                    animal_ids=animals,
                    n_sessions=len(sessions),
                    n_animals=len(animals),
                    independent_statistical_units=["animal", "session"],
                    **source_metrics,
                )
            return run.path

        max_outer = (
            None
            if config.get("max_outer_folds") is None
            else int(config["max_outer_folds"])
        )
        protocols = tuple(str(value) for value in config["protocols"])
        for protocol in protocols:
            if protocol == "cross-session-transfer":
                for family in FAMILIES:
                    run.mark_condition_invalid(
                        _CROSS_SESSION_UNSUPPORTED,
                        condition=f"{protocol}:{family}",
                        protocol=protocol,
                        model_family=family,
                        evaluation_level="animal_session",
                        stage="outer_cv",
                    )
                continue
            try:
                folds = _outer_folds(
                    dataset,
                    protocol,
                    max_folds=max_outer,
                )
            except Exception as error:
                for family in FAMILIES:
                    run.mark_condition_failure(
                        error,
                        condition=f"{protocol}:{family}",
                        protocol=protocol,
                        model_family=family,
                        evaluation_level="animal_session",
                        stage="outer_cv_construction",
                    )
                continue
            for family in FAMILIES:
                successes: list[dict[str, object]] = []
                failed = 0
                for fold in folds:
                    dimensions = {
                        "condition": f"{protocol}:{family}",
                        "protocol": protocol,
                        "model_family": family,
                        "evaluation_level": "animal_session",
                        "record_type": "outer_fold",
                        "fold_id": fold.fold_id,
                    }
                    try:
                        metrics = _evaluate_outer_fold(
                            dataset,
                            fold,
                            family=family,
                            config=config,
                        )
                    except Exception as error:
                        failed += 1
                        run.mark_condition_failure(
                            error,
                            **dimensions,
                            stage="nested_fit_and_heldout_score",
                        )
                    else:
                        successes.append(metrics)
                        run.record(metrics, **dimensions)
                aggregate_dimensions = {
                    "condition": f"{protocol}:{family}",
                    "protocol": protocol,
                    "model_family": family,
                    "evaluation_level": "animal_session",
                    "record_type": "protocol_aggregate",
                }
                if not successes:
                    run.record_failed_condition(
                        {
                            "failure_reason": (
                                "no outer fold completed exact held-out scoring"
                            ),
                            "outer_folds_planned": len(folds),
                            "outer_folds_complete": 0,
                            "outer_folds_failed": failed,
                            "candidate_latent_dims": list(
                                DEFAULT_LATENT_DIMENSIONS
                            ),
                            "synthetic_fallback": False,
                        },
                        **aggregate_dimensions,
                    )
                    continue
                aggregate = _aggregate_successes(
                    successes,
                    planned_folds=len(folds),
                    failed_folds=failed,
                )
                if failed:
                    run.record_failed_condition(
                        {
                            **aggregate,
                            "failure_reason": (
                                "one or more registered outer folds failed; "
                                "successful fold metrics are retained"
                            ),
                        },
                        **aggregate_dimensions,
                    )
                else:
                    run.record(aggregate, **aggregate_dimensions)
        return run.path


def main(argv: list[str] | None = None) -> None:
    parser = basic_parser(
        "Run fail-closed CompositionalTasks shared-dynamics validation",
        "configs/smoke/exp25_compositional_tasks_real.json",
    )
    args = parser.parse_args(argv)
    config = load_json_config(args.config)
    seeds = seed_list(args.seeds if args.seeds is not None else config["seeds"])
    for seed in seeds:
        print(run_seed(config, seed, args.results_root))


if __name__ == "__main__":
    main()
