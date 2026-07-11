"""Session-level scoring for the leakage-safe IBL behavior benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.data.ibl_behavior import (
    ContiguousBlockSplit,
    IBLBehaviorDataError,
    IBLBehaviorSession,
    causal_exponential_trace,
)


Array = np.ndarray


def _validated_beliefs(value: object, n_trials: int) -> Array:
    beliefs = np.asarray(value, dtype=float)
    if beliefs.shape != (n_trials, 2):
        raise ValueError("beliefs must have shape (n_trials,2)")
    if (
        not np.isfinite(beliefs).all()
        or np.any(beliefs < 0.0)
        or not np.allclose(beliefs.sum(axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("beliefs must be finite probabilities summing to one")
    return beliefs


def oracle_ceiling_beliefs(context_labels: object) -> Array:
    """Construct an explicitly evaluation-only, truth-reading upper ceiling."""

    labels = np.asarray(context_labels, dtype=int)
    if labels.ndim != 1 or labels.size == 0 or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("context_labels must be a non-empty vector in {-1,0,1}")
    result = np.full((labels.size, 2), 0.5, dtype=float)
    biased = labels >= 0
    result[biased] = np.eye(2, dtype=float)[labels[biased]]
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ContextMetricSummary:
    nll: float
    brier: float
    accuracy: float
    expected_calibration_error: float
    n_trials: int


def binary_context_metrics(
    beliefs: object,
    context_labels: object,
    *,
    indices: object,
    epsilon: float = 1e-9,
    n_bins: int = 10,
) -> ContextMetricSummary:
    """Score frozen beliefs; this is the only non-ceiling truth capability."""

    labels = np.asarray(context_labels, dtype=int)
    probability = _validated_beliefs(beliefs, labels.size)
    selected = np.asarray(indices, dtype=int)
    if (
        selected.ndim != 1
        or selected.size == 0
        or np.any(selected < 0)
        or np.any(selected >= labels.size)
    ):
        raise ValueError("indices must select at least one valid trial")
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("context_labels must lie in {-1,0,1}")
    if not 0.0 < float(epsilon) < 1.0 or int(n_bins) < 1:
        raise ValueError("epsilon must lie in (0,1) and n_bins must be positive")
    y = labels[selected]
    if np.any(y < 0):
        raise ValueError("indices must exclude unbiased 0.5 burn-in trials")
    p = probability[selected]
    one_hot = np.eye(2)[y]
    nll = -float(np.mean(np.log(np.clip(p[np.arange(y.size), y], epsilon, 1.0))))
    brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
    prediction = np.argmax(p, axis=1)
    confidence = np.max(p, axis=1)
    correct = prediction == y
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence >= lower) & (
            confidence <= upper if upper == 1.0 else confidence < upper
        )
        if np.any(in_bin):
            ece += float(np.mean(in_bin)) * abs(
                float(np.mean(correct[in_bin])) - float(np.mean(confidence[in_bin]))
            )
    return ContextMetricSummary(
        nll=nll,
        brier=brier,
        accuracy=float(np.mean(correct)),
        expected_calibration_error=float(ece),
        n_trials=int(y.size),
    )


# Backward-compatible name for earlier three-state pilot callers.  The current
# implementation is explicitly binary and masks the 0.5 burn-in state.
multiclass_context_metrics = binary_context_metrics


BEHAVIOR_FEATURE_NAMES = (
    "current_signed_contrast",
    "current_absolute_contrast",
    "past_stimulus_trace_fast",
    "past_stimulus_trace_slow",
    "previous_choice",
    "previous_rewarded_choice",
    "belief_high",
)


def causal_behavior_features(session: IBLBehaviorSession, beliefs: object) -> Array:
    """Build current-stimulus plus strictly past history/readout features.

    Histories run continuously across true block and fold boundaries.  The final
    low-context belief is omitted because the two probabilities sum to one.
    """

    if not isinstance(session, IBLBehaviorSession):
        raise TypeError("session must be an IBLBehaviorSession")
    p = _validated_beliefs(beliefs, session.trial_ids.size)
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
            p[:, 1],
        ]
    )
    if not np.isfinite(features).all():
        raise IBLBehaviorDataError("behavior features must be finite")
    features.setflags(write=False)
    return features


@dataclass(frozen=True)
class BehaviorMetricSummary:
    log_loss: float
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    mcfadden_pseudo_r2: float
    fit_trial_ids: Array
    test_trial_ids: Array
    feature_count: int

    def __post_init__(self) -> None:
        fit_ids = np.array(self.fit_trial_ids, dtype=int, copy=True)
        test_ids = np.array(self.test_trial_ids, dtype=int, copy=True)
        fit_ids.setflags(write=False)
        test_ids.setflags(write=False)
        object.__setattr__(self, "fit_trial_ids", fit_ids)
        object.__setattr__(self, "test_trial_ids", test_ids)


def fit_behavior_logistic(
    session: IBLBehaviorSession,
    beliefs: object,
    split: ContiguousBlockSplit,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 0,
) -> BehaviorMetricSummary:
    """Fit scaler and logistic readout on train trials, evaluate test trials."""

    if not isinstance(session, IBLBehaviorSession):
        raise TypeError("session must be an IBLBehaviorSession")
    if not isinstance(split, ContiguousBlockSplit):
        raise TypeError("split must be a ContiguousBlockSplit")
    if not np.isfinite(C) or float(C) <= 0.0 or int(max_iter) < 1:
        raise ValueError("C and max_iter must be positive")
    features = causal_behavior_features(session, beliefs)
    train_mask = session.choice_valid & session.analysis_mask
    test_mask = session.choice_valid & session.analysis_mask
    train = split.train_indices[train_mask[split.train_indices]]
    test = split.test_indices[test_mask[split.test_indices]]
    if train.size < 4 or test.size < 2:
        raise IBLBehaviorDataError("too few valid choices in train or test split")
    y_train = session.choice_left[train]
    y_test = session.choice_left[test]
    if np.unique(y_train).size != 2 or np.unique(y_test).size != 2:
        raise IBLBehaviorDataError(
            "behavior train and test splits must each contain both choices"
        )
    scaler = StandardScaler().fit(features[train])
    model = LogisticRegression(
        C=float(C),
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(seed),
    ).fit(scaler.transform(features[train]), y_train)
    probability = model.predict_proba(scaler.transform(features[test]))[:, 1]
    prediction = (probability >= 0.5).astype(int)
    model_loss = float(log_loss(y_test, probability, labels=[0, 1]))
    train_rate = float(np.clip(np.mean(y_train), 1e-9, 1.0 - 1e-9))
    null_probability = np.full(y_test.size, train_rate)
    null_loss = float(log_loss(y_test, null_probability, labels=[0, 1]))
    return BehaviorMetricSummary(
        log_loss=model_loss,
        accuracy=float(accuracy_score(y_test, prediction)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, prediction)),
        roc_auc=float(roc_auc_score(y_test, probability)),
        mcfadden_pseudo_r2=float(1.0 - model_loss / null_loss),
        fit_trial_ids=session.trial_ids[train],
        test_trial_ids=session.trial_ids[test],
        feature_count=int(features.shape[1]),
    )


__all__ = [
    "BEHAVIOR_FEATURE_NAMES",
    "BehaviorMetricSummary",
    "ContextMetricSummary",
    "causal_behavior_features",
    "binary_context_metrics",
    "fit_behavior_logistic",
    "multiclass_context_metrics",
    "oracle_ceiling_beliefs",
]
