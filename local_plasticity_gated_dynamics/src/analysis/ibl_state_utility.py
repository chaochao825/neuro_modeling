"""Leakage-safe behavioral utility tests for causal IBL controller states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from src.data.ibl_behavior import (
    ContiguousBlockSplit,
    IBLBehaviorDataError,
    IBLBehaviorSession,
    causal_exponential_trace,
)
from src.models.ibl_run_length_observer import RunLengthPrediction


Array = np.ndarray

HISTORY_FEATURE_NAMES = (
    "current_signed_contrast",
    "current_absolute_contrast",
    "past_stimulus_trace_fast",
    "past_stimulus_trace_slow",
    "previous_choice",
    "previous_rewarded_choice",
)
MEAN_STATE_FEATURE_NAMES = HISTORY_FEATURE_NAMES + ("prior_log_odds",)
FACTORIZED_STATE_FEATURE_NAMES = MEAN_STATE_FEATURE_NAMES + (
    "release_probability",
    "run_length_concentration",
    "prior_x_release",
    "prior_x_concentration",
)


def _readonly_matrix(value: object, *, n_trials: int) -> Array:
    result = np.array(value, dtype=float, order="C", copy=True)
    if (
        result.ndim != 2
        or result.shape[0] != n_trials
        or result.shape[1] < 1
        or not np.isfinite(result).all()
    ):
        raise IBLBehaviorDataError(
            "features must be a finite matrix with one row per trial"
        )
    result.setflags(write=False)
    return result


def causal_history_features(session: IBLBehaviorSession) -> Array:
    """Return the preregistered strong history-only feature family."""

    if not isinstance(session, IBLBehaviorSession):
        raise TypeError("session must be an IBLBehaviorSession")
    choice_signed = 2.0 * session.choice_left.astype(float) - 1.0
    choice_signed = np.where(session.choice_valid, choice_signed, 0.0)
    previous_choice = np.zeros(session.trial_ids.size, dtype=float)
    previous_rewarded = np.zeros(session.trial_ids.size, dtype=float)
    previous_choice[1:] = choice_signed[:-1]
    previous_rewarded[1:] = choice_signed[:-1] * session.feedback_correct[:-1].astype(
        float
    )
    features = np.column_stack(
        [
            session.signed_contrast,
            np.abs(session.signed_contrast),
            causal_exponential_trace(session.stimulus_side, 0.8),
            causal_exponential_trace(session.stimulus_side, 0.95),
            previous_choice,
            previous_rewarded,
        ]
    )
    return _readonly_matrix(features, n_trials=session.trial_ids.size)


def belief_mean_features(
    session: IBLBehaviorSession,
    beliefs: object,
    *,
    epsilon: float = 1e-6,
) -> Array:
    """Add only the directional prior mean to the history baseline."""

    probability = np.asarray(beliefs, dtype=float)
    if (
        probability.shape != (session.trial_ids.size, 2)
        or not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or not np.allclose(probability.sum(axis=1), 1.0)
    ):
        raise IBLBehaviorDataError("beliefs must be trial-aligned probabilities")
    if not np.isfinite(epsilon) or not 0.0 < float(epsilon) < 0.5:
        raise ValueError("epsilon must lie in (0,0.5)")
    high = np.clip(probability[:, 1], float(epsilon), 1.0 - float(epsilon))
    prior_log_odds = np.log(high) - np.log1p(-high)
    return _readonly_matrix(
        np.column_stack([causal_history_features(session), prior_log_odds]),
        n_trials=session.trial_ids.size,
    )


def factorized_state_features(
    session: IBLBehaviorSession,
    prediction: RunLengthPrediction,
) -> Array:
    """Expand a three-dimensional causal state for a linear choice readout.

    The state itself is ``(prior log odds, release probability, run-length
    concentration)``.  Two fixed interactions permit release and precision to
    modulate the signed prior without introducing a nonlinear learned gate.
    """

    if not isinstance(prediction, RunLengthPrediction):
        raise TypeError("prediction must be a RunLengthPrediction")
    if not np.array_equal(prediction.trial_ids, session.trial_ids):
        raise IBLBehaviorDataError("prediction and session trial IDs differ")
    mean_features = belief_mean_features(session, prediction.beliefs)
    prior = mean_features[:, -1]
    release = prediction.release_probability
    concentration = 1.0 / prediction.run_length_ess
    return _readonly_matrix(
        np.column_stack(
            [
                mean_features,
                release,
                concentration,
                prior * release,
                prior * concentration,
            ]
        ),
        n_trials=session.trial_ids.size,
    )


def factorized_clamp_features(
    features: object,
    *,
    fit_indices: object,
    clamp: str,
) -> Array:
    """Clamp one controller coordinate while preserving a frozen readout."""

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(FACTORIZED_STATE_FEATURE_NAMES):
        raise IBLBehaviorDataError("expected the factorized feature contract")
    fit = np.asarray(fit_indices, dtype=int)
    if (
        fit.ndim != 1
        or fit.size == 0
        or np.any(fit < 0)
        or np.any(fit >= values.shape[0])
    ):
        raise IBLBehaviorDataError("fit_indices must select feature rows")
    result = np.array(values, copy=True)
    prior_index = FACTORIZED_STATE_FEATURE_NAMES.index("prior_log_odds")
    if clamp == "release":
        state_index = FACTORIZED_STATE_FEATURE_NAMES.index("release_probability")
        interaction_index = FACTORIZED_STATE_FEATURE_NAMES.index("prior_x_release")
    elif clamp == "concentration":
        state_index = FACTORIZED_STATE_FEATURE_NAMES.index("run_length_concentration")
        interaction_index = FACTORIZED_STATE_FEATURE_NAMES.index(
            "prior_x_concentration"
        )
    else:
        raise ValueError("clamp must be 'release' or 'concentration'")
    fixed = float(np.mean(result[fit, state_index]))
    result[:, state_index] = fixed
    result[:, interaction_index] = result[:, prior_index] * fixed
    return _readonly_matrix(result, n_trials=result.shape[0])


@dataclass(frozen=True)
class ChoiceSubsetMetrics:
    nll: float
    brier: float
    accuracy: float
    n_trials: int


@dataclass(frozen=True)
class ChoiceReadoutEvaluation:
    condition: str
    selected_c: float
    dev_selection_scope: str
    dev_selection_nll: float
    dev_selection_trial_count: int
    feature_count: int
    feature_names: tuple[str, ...]
    fit_trial_ids: Array
    dev_trial_ids: Array
    test_trial_ids: Array
    test_probabilities: Array
    metrics: Mapping[str, ChoiceSubsetMetrics]
    intervention_metrics: Mapping[str, Mapping[str, ChoiceSubsetMetrics]]

    def __post_init__(self) -> None:
        for name in (
            "fit_trial_ids",
            "dev_trial_ids",
            "test_trial_ids",
            "test_probabilities",
        ):
            dtype = float if name == "test_probabilities" else int
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            if value.ndim != 1:
                raise IBLBehaviorDataError(f"{name} must be one-dimensional")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.test_probabilities.shape != self.test_trial_ids.shape:
            raise IBLBehaviorDataError("test probabilities must align with test IDs")


def _subset_metrics(
    labels: Array, probability: Array, mask: Array
) -> ChoiceSubsetMetrics:
    y = labels[mask]
    p = probability[mask]
    if y.size == 0:
        return ChoiceSubsetMetrics(
            nll=float("nan"),
            brier=float("nan"),
            accuracy=float("nan"),
            n_trials=0,
        )
    return ChoiceSubsetMetrics(
        nll=float(log_loss(y, p, labels=[0, 1])),
        brier=float(np.mean((p - y) ** 2)),
        accuracy=float(np.mean((p >= 0.5).astype(int) == y)),
        n_trials=int(y.size),
    )


def _metric_masks(signed_contrast: Array, threshold: float) -> dict[str, Array]:
    absolute = np.abs(signed_contrast)
    return {
        "all": np.ones(absolute.size, dtype=bool),
        "low_contrast": absolute <= float(threshold) + 1e-12,
        "zero_contrast": absolute <= 1e-12,
    }


def evaluate_choice_readout(
    session: IBLBehaviorSession,
    split: ContiguousBlockSplit,
    features: object,
    *,
    condition: str,
    feature_names: Sequence[str],
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    dev_selection_scope: str = "low_contrast",
    low_contrast_threshold: float = 0.0625,
    min_dev_low_contrast_trials: int = 8,
    min_test_low_contrast_trials: int = 8,
    max_iter: int = 2000,
    seed: int = 0,
    interventions: Mapping[str, object] | None = None,
) -> ChoiceReadoutEvaluation:
    """Select regularization on dev, refit on train+dev, and score test.

    Every scaler is fit only on the rows used to fit its corresponding logistic
    model.  Test probabilities are never used for hyperparameter or condition
    selection.  Intervention matrices use the same final scaler and readout.
    """

    if not isinstance(session, IBLBehaviorSession):
        raise TypeError("session must be an IBLBehaviorSession")
    if not isinstance(split, ContiguousBlockSplit):
        raise TypeError("split must be a ContiguousBlockSplit")
    values = _readonly_matrix(features, n_trials=session.trial_ids.size)
    names = tuple(str(name) for name in feature_names)
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names must uniquely name every feature column")
    candidates = tuple(float(value) for value in c_grid)
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or any(not np.isfinite(value) or value <= 0.0 for value in candidates)
    ):
        raise ValueError("c_grid must contain unique positive values")
    if not np.isfinite(low_contrast_threshold) or low_contrast_threshold < 0.0:
        raise ValueError("low_contrast_threshold cannot be negative")
    if dev_selection_scope not in {"low_contrast", "all"}:
        raise ValueError("dev_selection_scope must be 'low_contrast' or 'all'")

    eligible = session.choice_valid & session.analysis_mask
    train = split.train_indices[eligible[split.train_indices]]
    dev = split.dev_indices[eligible[split.dev_indices]]
    test = split.test_indices[eligible[split.test_indices]]
    dev_low = dev[
        np.abs(session.signed_contrast[dev]) <= float(low_contrast_threshold) + 1e-12
    ]
    test_low = test[
        np.abs(session.signed_contrast[test]) <= float(low_contrast_threshold) + 1e-12
    ]
    if dev_low.size < int(min_dev_low_contrast_trials):
        raise IBLBehaviorDataError("too few low-contrast choices in dev split")
    if test_low.size < int(min_test_low_contrast_trials):
        raise IBLBehaviorDataError("too few low-contrast choices in test split")
    if np.unique(session.choice_left[train]).size != 2:
        raise IBLBehaviorDataError("training choices must contain both classes")
    selection_trials = dev_low if dev_selection_scope == "low_contrast" else dev

    dev_scores: list[float] = []
    for candidate in candidates:
        scaler = StandardScaler().fit(values[train])
        model = LogisticRegression(
            C=candidate,
            solver="lbfgs",
            max_iter=int(max_iter),
            random_state=int(seed),
        ).fit(scaler.transform(values[train]), session.choice_left[train])
        probability = model.predict_proba(scaler.transform(values[selection_trials]))[
            :, 1
        ]
        dev_scores.append(
            float(
                log_loss(
                    session.choice_left[selection_trials],
                    probability,
                    labels=[0, 1],
                )
            )
        )
    selected_index = min(
        range(len(candidates)), key=lambda index: (dev_scores[index], candidates[index])
    )
    selected_c = candidates[selected_index]

    fit = np.concatenate([train, dev])
    if np.unique(session.choice_left[fit]).size != 2:
        raise IBLBehaviorDataError("final fit choices must contain both classes")
    scaler = StandardScaler().fit(values[fit])
    model = LogisticRegression(
        C=selected_c,
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(seed),
    ).fit(scaler.transform(values[fit]), session.choice_left[fit])
    probability = model.predict_proba(scaler.transform(values[test]))[:, 1]
    masks = _metric_masks(session.signed_contrast[test], low_contrast_threshold)
    metrics = {
        name: _subset_metrics(session.choice_left[test], probability, mask)
        for name, mask in masks.items()
    }

    intervention_metrics: dict[str, dict[str, ChoiceSubsetMetrics]] = {}
    for name, alternate in dict(interventions or {}).items():
        alternate_values = _readonly_matrix(alternate, n_trials=session.trial_ids.size)
        if alternate_values.shape != values.shape:
            raise IBLBehaviorDataError("intervention feature shape changed")
        alternate_probability = model.predict_proba(
            scaler.transform(alternate_values[test])
        )[:, 1]
        intervention_metrics[str(name)] = {
            subset: _subset_metrics(
                session.choice_left[test], alternate_probability, mask
            )
            for subset, mask in masks.items()
        }
    return ChoiceReadoutEvaluation(
        condition=str(condition),
        selected_c=selected_c,
        dev_selection_scope=dev_selection_scope,
        dev_selection_nll=dev_scores[selected_index],
        dev_selection_trial_count=int(selection_trials.size),
        feature_count=int(values.shape[1]),
        feature_names=names,
        fit_trial_ids=session.trial_ids[fit],
        dev_trial_ids=session.trial_ids[selection_trials],
        test_trial_ids=session.trial_ids[test],
        test_probabilities=probability,
        metrics=metrics,
        intervention_metrics=intervention_metrics,
    )


__all__ = [
    "FACTORIZED_STATE_FEATURE_NAMES",
    "HISTORY_FEATURE_NAMES",
    "MEAN_STATE_FEATURE_NAMES",
    "ChoiceReadoutEvaluation",
    "ChoiceSubsetMetrics",
    "belief_mean_features",
    "causal_history_features",
    "evaluate_choice_readout",
    "factorized_clamp_features",
    "factorized_state_features",
]
