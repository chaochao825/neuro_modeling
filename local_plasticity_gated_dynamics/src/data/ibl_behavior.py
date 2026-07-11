"""Local-table IBL behavior data and causal binary hidden-block belief models.

The gate-facing capability in this module contains stimulus sides and trial IDs
only.  ``probabilityLeft`` is retained on :class:`IBLBehaviorSession` for
held-out scoring, but is never accepted by the learned gate APIs.  Every belief
at trial ``t`` is a predictive prior based on stimulus sides through ``t - 1``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp


Array = np.ndarray
IBL_CONTEXT_LEVELS = np.array([0.2, 0.5, 0.8], dtype=float)
IBL_CONTEXT_LEVELS.setflags(write=False)
IBL_BIASED_CONTEXT_LEVELS = np.array([0.2, 0.8], dtype=float)
IBL_BIASED_CONTEXT_LEVELS.setflags(write=False)


class IBLBehaviorDataError(ValueError):
    """Raised when a local IBL trial table violates the benchmark contract."""


def _readonly_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype | type,
    ndim: int = 1,
) -> Array:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.ndim != ndim:
        raise IBLBehaviorDataError(f"{name} must be {ndim}-dimensional")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise IBLBehaviorDataError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _fingerprint(*arrays: Array, tag: str) -> str:
    digest = hashlib.sha256(tag.encode("utf-8"))
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _context_labels(probability_left: Array, tolerance: float) -> Array:
    distance = np.abs(probability_left[:, None] - IBL_CONTEXT_LEVELS[None, :])
    labels = np.argmin(distance, axis=1)
    if np.any(distance[np.arange(probability_left.size), labels] > tolerance):
        observed = np.unique(probability_left).tolist()
        raise IBLBehaviorDataError(
            "probabilityLeft must match the public IBL levels 0.2, 0.5, 0.8; "
            f"observed={observed}"
        )
    # The unbiased 0.5 block is an initial burn-in in the biased-block task,
    # not a third long-lived hidden state.  It receives label -1 and is masked
    # from primary context scoring; 0.2/0.8 map to binary labels 0/1.
    return np.where(labels == 1, -1, np.where(labels == 0, 0, 1)).astype(int)


def contiguous_block_ids(values: object) -> Array:
    """Assign a new block ID whenever a context value changes."""

    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise IBLBehaviorDataError("probabilityLeft must be a non-empty vector")
    if not np.isfinite(probabilities).all():
        raise IBLBehaviorDataError("probabilityLeft must be finite")
    changes = np.ones(probabilities.size, dtype=bool)
    if probabilities.size > 1:
        changes[1:] = ~np.isclose(
            probabilities[1:], probabilities[:-1], rtol=0.0, atol=1e-8
        )
    result = np.cumsum(changes, dtype=int) - 1
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class IBLBehaviorObservations:
    """Minimal capability exposed to hidden-block gates."""

    trial_ids: Array
    stimulus_side: Array

    def __post_init__(self) -> None:
        trial_ids = _readonly_array(self.trial_ids, name="trial_ids", dtype=int)
        stimulus_side = _readonly_array(
            self.stimulus_side, name="stimulus_side", dtype=int
        )
        if trial_ids.size == 0 or stimulus_side.shape != trial_ids.shape:
            raise IBLBehaviorDataError(
                "trial_ids and stimulus_side must be matching non-empty vectors"
            )
        if np.unique(trial_ids).size != trial_ids.size or np.any(trial_ids < 0):
            raise IBLBehaviorDataError("trial_ids must be unique and non-negative")
        if not np.isin(stimulus_side, [0, 1]).all():
            raise IBLBehaviorDataError("stimulus_side must be binary")
        object.__setattr__(self, "trial_ids", trial_ids)
        object.__setattr__(self, "stimulus_side", stimulus_side)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.trial_ids,
            self.stimulus_side,
            tag="ibl-behavior-observations-v1",
        )


@dataclass(frozen=True)
class IBLBehaviorSession:
    """One local IBL session with evaluation truth kept outside gate input."""

    eid: str
    animal_id: str
    trial_ids: Array
    stimulus_side: Array
    signed_contrast: Array
    choice_left: Array
    choice_valid: Array
    feedback_correct: Array
    probability_left: Array
    context_labels: Array
    block_ids: Array
    source_trial_indices: Array
    analysis_mask: Array
    context_score_mask: Array
    official_bwm_mask_present: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.eid, str) or not self.eid:
            raise IBLBehaviorDataError("eid must be a non-empty string")
        if not isinstance(self.animal_id, str) or not self.animal_id:
            raise IBLBehaviorDataError("animal_id must be a non-empty string")
        trial_ids = _readonly_array(self.trial_ids, name="trial_ids", dtype=int)
        n_trials = trial_ids.size
        if n_trials < 3 or np.unique(trial_ids).size != n_trials:
            raise IBLBehaviorDataError(
                "a session needs at least three trials with unique trial_ids"
            )
        fields: dict[str, Array] = {
            "stimulus_side": _readonly_array(
                self.stimulus_side, name="stimulus_side", dtype=int
            ),
            "signed_contrast": _readonly_array(
                self.signed_contrast, name="signed_contrast", dtype=float
            ),
            "choice_left": _readonly_array(
                self.choice_left, name="choice_left", dtype=int
            ),
            "choice_valid": _readonly_array(
                self.choice_valid, name="choice_valid", dtype=bool
            ),
            "feedback_correct": _readonly_array(
                self.feedback_correct, name="feedback_correct", dtype=bool
            ),
            "probability_left": _readonly_array(
                self.probability_left, name="probability_left", dtype=float
            ),
            "context_labels": _readonly_array(
                self.context_labels, name="context_labels", dtype=int
            ),
            "block_ids": _readonly_array(self.block_ids, name="block_ids", dtype=int),
            "source_trial_indices": _readonly_array(
                self.source_trial_indices, name="source_trial_indices", dtype=int
            ),
            "analysis_mask": _readonly_array(
                self.analysis_mask, name="analysis_mask", dtype=bool
            ),
            "context_score_mask": _readonly_array(
                self.context_score_mask, name="context_score_mask", dtype=bool
            ),
        }
        if any(value.shape != (n_trials,) for value in fields.values()):
            raise IBLBehaviorDataError("all session arrays must match trial_ids")
        if not np.isin(fields["stimulus_side"], [0, 1]).all():
            raise IBLBehaviorDataError("stimulus_side must be binary")
        if not np.isin(fields["choice_left"], [0, 1]).all():
            raise IBLBehaviorDataError("choice_left must be binary")
        if not np.isin(fields["context_labels"], [-1, 0, 1]).all():
            raise IBLBehaviorDataError("context_labels must be in {-1,0,1}")
        expected_labels = _context_labels(fields["probability_left"], 1e-6)
        if not np.array_equal(fields["context_labels"], expected_labels):
            raise IBLBehaviorDataError("context_labels disagree with probability_left")
        expected_blocks = contiguous_block_ids(fields["probability_left"])
        if not np.array_equal(fields["block_ids"], expected_blocks):
            raise IBLBehaviorDataError(
                "block_ids must be the contiguous probabilityLeft blocks"
            )
        source = fields["source_trial_indices"]
        if np.any(source < 0) or np.any(np.diff(source) != 1):
            raise IBLBehaviorDataError(
                "source_trial_indices must be consecutive; missing-observation "
                "transitions are not implemented and must not be stitched"
            )
        expected_context_mask = (fields["context_labels"] >= 0) & fields[
            "analysis_mask"
        ]
        if not np.array_equal(fields["context_score_mask"], expected_context_mask):
            raise IBLBehaviorDataError(
                "context_score_mask must select analyzed 0.2/0.8 trials only"
            )
        if not isinstance(self.official_bwm_mask_present, (bool, np.bool_)):
            raise IBLBehaviorDataError("official_bwm_mask_present must be boolean")
        object.__setattr__(self, "trial_ids", trial_ids)
        for name, value in fields.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "official_bwm_mask_present", bool(self.official_bwm_mask_present)
        )

    @property
    def observations(self) -> IBLBehaviorObservations:
        """Return the truth-free gate capability."""

        return IBLBehaviorObservations(self.trial_ids, self.stimulus_side)

    @property
    def n_blocks(self) -> int:
        return int(np.unique(self.block_ids).size)


def _choice_columns(values: Array) -> tuple[Array, Array]:
    numeric = np.asarray(values, dtype=float)
    valid = np.isin(numeric, [-1.0, 1.0])
    # In the public IBL convention choice=+1 denotes a leftward choice and
    # choice=-1 denotes a rightward choice.
    left = np.zeros(numeric.size, dtype=int)
    left[valid] = (numeric[valid] == 1.0).astype(int)
    return left, valid


def load_ibl_behavior_table(
    table: str | Path | pd.DataFrame | Mapping[str, object],
    *,
    eid: str,
    animal_id: str,
    context_tolerance: float = 1e-6,
) -> IBLBehaviorSession:
    """Load a local CSV/Parquet/DataFrame trial table without network access."""

    if isinstance(table, pd.DataFrame):
        frame = table.copy(deep=True)
    elif isinstance(table, Mapping):
        frame = pd.DataFrame(dict(table))
    else:
        path = Path(table)
        if not path.exists() or not path.is_file():
            raise IBLBehaviorDataError(f"local trial table does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".parquet", ".pqt"}:
            frame = pd.read_parquet(path)
        else:
            raise IBLBehaviorDataError("local trial table must be CSV or Parquet")
    required = {
        "contrastLeft",
        "contrastRight",
        "choice",
        "feedbackType",
        "probabilityLeft",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise IBLBehaviorDataError(f"trial table is missing columns: {missing}")
    if len(frame) < 3:
        raise IBLBehaviorDataError("trial table contains too few trials")

    left = pd.to_numeric(frame["contrastLeft"], errors="coerce").to_numpy(float)
    right = pd.to_numeric(frame["contrastRight"], errors="coerce").to_numpy(float)
    left_present = np.isfinite(left)
    right_present = np.isfinite(right)
    if not np.all(left_present ^ right_present):
        raise IBLBehaviorDataError(
            "every trial must have exactly one finite contrast side"
        )
    stimulus_side = left_present.astype(int)
    strength = np.where(left_present, left, right)
    if np.any(strength < 0.0):
        raise IBLBehaviorDataError("stimulus contrasts must be non-negative")
    signed_contrast = np.where(left_present, strength, -strength)

    choice_left, choice_valid = _choice_columns(frame["choice"].to_numpy())
    probability_left = pd.to_numeric(
        frame["probabilityLeft"], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(probability_left).all():
        raise IBLBehaviorDataError("probabilityLeft must be complete and finite")
    labels = _context_labels(probability_left, float(context_tolerance))
    feedback = pd.to_numeric(frame["feedbackType"], errors="coerce").to_numpy(float)
    feedback_correct = np.isfinite(feedback) & (feedback > 0.0)
    if "source_trial_index" in frame:
        source_numeric = pd.to_numeric(
            frame["source_trial_index"], errors="coerce"
        ).to_numpy(float)
        if (
            not np.isfinite(source_numeric).all()
            or not np.equal(source_numeric, np.floor(source_numeric)).all()
        ):
            raise IBLBehaviorDataError(
                "source_trial_index must contain finite integers"
            )
        source_trial_indices = source_numeric.astype(int)
    else:
        source_trial_indices = np.arange(len(frame), dtype=int)
    if np.any(source_trial_indices < 0) or np.any(np.diff(source_trial_indices) != 1):
        raise IBLBehaviorDataError(
            "source_trial_index contains gaps; this HMM cannot stitch missing trials"
        )
    official_mask_present = "official_bwm_mask" in frame
    if official_mask_present:
        raw_mask = frame["official_bwm_mask"]
        if raw_mask.dtype.kind == "b":
            analysis_mask = raw_mask.to_numpy(bool)
        else:
            normalized = raw_mask.astype(str).str.strip().str.lower()
            if not normalized.isin(["true", "false", "1", "0"]).all():
                raise IBLBehaviorDataError(
                    "official_bwm_mask must contain only booleans"
                )
            analysis_mask = normalized.isin(["true", "1"]).to_numpy(bool)
    else:
        analysis_mask = np.ones(len(frame), dtype=bool)
    trial_ids = source_trial_indices.copy()
    return IBLBehaviorSession(
        eid=eid,
        animal_id=animal_id,
        trial_ids=trial_ids,
        stimulus_side=stimulus_side,
        signed_contrast=signed_contrast,
        choice_left=choice_left,
        choice_valid=choice_valid,
        feedback_correct=feedback_correct,
        probability_left=probability_left,
        context_labels=labels,
        block_ids=contiguous_block_ids(probability_left),
        source_trial_indices=source_trial_indices,
        analysis_mask=analysis_mask,
        context_score_mask=(labels >= 0) & analysis_mask,
        official_bwm_mask_present=official_mask_present,
    )


@dataclass(frozen=True)
class ContiguousBlockSplit:
    """Chronological train/dev/test split whose boundaries fall between blocks."""

    train_indices: Array
    dev_indices: Array
    test_indices: Array
    train_block_ids: Array
    dev_block_ids: Array
    test_block_ids: Array
    fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "train_indices",
            "dev_indices",
            "test_indices",
            "train_block_ids",
            "dev_block_ids",
            "test_block_ids",
        ):
            value = _readonly_array(getattr(self, name), name=name, dtype=int)
            if value.size == 0:
                raise IBLBehaviorDataError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        index_parts = (self.train_indices, self.dev_indices, self.test_indices)
        if any(
            np.intersect1d(index_parts[i], index_parts[j]).size
            for i, j in ((0, 1), (0, 2), (1, 2))
        ):
            raise IBLBehaviorDataError("train/dev/test trial indices must be disjoint")
        block_parts = (self.train_block_ids, self.dev_block_ids, self.test_block_ids)
        if any(
            np.intersect1d(block_parts[i], block_parts[j]).size
            for i, j in ((0, 1), (0, 2), (1, 2))
        ):
            raise IBLBehaviorDataError("train/dev/test blocks must be disjoint")
        if not (
            self.train_indices[-1] < self.dev_indices[0]
            and self.dev_indices[-1] < self.test_indices[0]
        ):
            raise IBLBehaviorDataError("splits must be chronological and contiguous")
        if not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise IBLBehaviorDataError("split fingerprint must be non-empty")


def contiguous_block_split(
    block_ids: object,
    *,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    min_blocks: int = 5,
) -> ContiguousBlockSplit:
    """Split an ordered session only at true block boundaries.

    Truth is used here solely to define grouped evaluation folds.  Gate state is
    never reset at any of these true boundaries.
    """

    blocks = np.asarray(block_ids, dtype=int)
    if blocks.ndim != 1 or blocks.size < 3 or np.any(blocks < 0):
        raise IBLBehaviorDataError("block_ids must be a non-negative trial vector")
    starts = np.ones(blocks.size, dtype=bool)
    starts[1:] = blocks[1:] != blocks[:-1]
    ordered = blocks[starts]
    if np.unique(ordered).size != ordered.size:
        raise IBLBehaviorDataError("each block must occupy one contiguous segment")
    if ordered.size < int(min_blocks) or ordered.size < 3:
        raise IBLBehaviorDataError(
            f"session has {ordered.size} blocks; requires at least {max(3, int(min_blocks))}"
        )
    for name, value in (
        ("test_fraction", test_fraction),
        ("validation_fraction", validation_fraction),
    ):
        if not np.isfinite(value) or not 0.0 < float(value) < 1.0:
            raise IBLBehaviorDataError(f"{name} must lie in (0,1)")
    n_blocks = ordered.size
    n_test = max(1, int(np.ceil(float(test_fraction) * n_blocks)))
    n_dev = max(1, int(np.ceil(float(validation_fraction) * n_blocks)))
    if n_test + n_dev >= n_blocks:
        raise IBLBehaviorDataError("split fractions leave no training blocks")
    n_train = n_blocks - n_dev - n_test
    train_blocks = ordered[:n_train]
    dev_blocks = ordered[n_train : n_train + n_dev]
    test_blocks = ordered[n_train + n_dev :]
    train = np.flatnonzero(np.isin(blocks, train_blocks))
    dev = np.flatnonzero(np.isin(blocks, dev_blocks))
    test = np.flatnonzero(np.isin(blocks, test_blocks))
    split_id = _fingerprint(
        train,
        dev,
        test,
        tag="ibl-behavior-chronological-block-split-v1",
    )
    return ContiguousBlockSplit(
        train,
        dev,
        test,
        train_blocks,
        dev_blocks,
        test_blocks,
        split_id,
    )


def causal_exponential_trace(stimulus_side: object, decay: float) -> Array:
    """Return a signed trace using observations strictly before each trial."""

    sides = np.asarray(stimulus_side, dtype=int)
    if sides.ndim != 1 or sides.size == 0 or not np.isin(sides, [0, 1]).all():
        raise IBLBehaviorDataError("stimulus_side must be a non-empty binary vector")
    if not np.isfinite(decay) or not 0.0 <= float(decay) < 1.0:
        raise IBLBehaviorDataError("decay must lie in [0,1)")
    signed = 2.0 * sides.astype(float) - 1.0
    trace = np.empty(sides.size, dtype=float)
    state = 0.0
    for index in range(sides.size):
        trace[index] = state
        state = float(decay) * state + (1.0 - float(decay)) * signed[index]
    trace.setflags(write=False)
    return trace


@dataclass(frozen=True)
class IBLBeliefPrediction:
    """Binary biased-block causal belief trajectory with fit provenance."""

    beliefs: Array
    trial_ids: Array
    model_name: str
    fit_trial_ids: Array
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        beliefs = _readonly_array(self.beliefs, name="beliefs", dtype=float, ndim=2)
        trial_ids = _readonly_array(self.trial_ids, name="trial_ids", dtype=int)
        fit_ids = _readonly_array(self.fit_trial_ids, name="fit_trial_ids", dtype=int)
        if beliefs.shape != (trial_ids.size, 2):
            raise IBLBehaviorDataError("beliefs must have shape (n_trials,2)")
        if np.any(beliefs < 0.0) or not np.allclose(beliefs.sum(axis=1), 1.0):
            raise IBLBehaviorDataError(
                "belief rows must be non-negative and sum to one"
            )
        if not isinstance(self.model_name, str) or not self.model_name:
            raise IBLBehaviorDataError("model_name must be non-empty")
        if np.unique(fit_ids).size != fit_ids.size:
            raise IBLBehaviorDataError("fit_trial_ids must be unique")
        object.__setattr__(self, "beliefs", beliefs)
        object.__setattr__(self, "trial_ids", trial_ids)
        object.__setattr__(self, "fit_trial_ids", fit_ids)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.beliefs,
            self.trial_ids,
            self.fit_trial_ids,
            tag=f"ibl-belief-{self.model_name}-v1",
        )


def _validate_training_prefix(indices: object, n_trials: int) -> Array:
    train = np.asarray(indices, dtype=int)
    if (
        train.ndim != 1
        or train.size < 3
        or np.any(train < 0)
        or np.any(train >= n_trials)
    ):
        raise IBLBehaviorDataError("train_indices must select at least three trials")
    if not np.array_equal(train, np.arange(train.size)):
        raise IBLBehaviorDataError(
            "train_indices must be the chronological session prefix"
        )
    result = train.copy()
    result.setflags(write=False)
    return result


class NoMemoryBelief:
    """Uniform context prior that consumes no trial observation."""

    def predict(self, observations: IBLBehaviorObservations) -> IBLBeliefPrediction:
        if not isinstance(observations, IBLBehaviorObservations):
            raise TypeError("observations must be IBLBehaviorObservations")
        return IBLBeliefPrediction(
            np.full((observations.trial_ids.size, 2), 0.5),
            observations.trial_ids,
            "no_memory",
            np.empty(0, dtype=int),
            (("uses_current_stimulus", False),),
        )


class ExponentialHistoryBelief:
    """Causal exponential history with decay chosen on train observations only."""

    def __init__(
        self,
        *,
        decays: Sequence[float] = (0.5, 0.8, 0.9, 0.95, 0.98),
        belief_width: float = 0.12,
    ) -> None:
        self.decays = tuple(float(value) for value in decays)
        if not self.decays or len(set(self.decays)) != len(self.decays):
            raise IBLBehaviorDataError("decays must be non-empty and unique")
        if any(not 0.0 <= value < 1.0 for value in self.decays):
            raise IBLBehaviorDataError("every decay must lie in [0,1)")
        self.belief_width = float(belief_width)
        if not np.isfinite(self.belief_width) or self.belief_width <= 0.0:
            raise IBLBehaviorDataError("belief_width must be positive")
        self._fitted = False

    def fit(
        self,
        observations: IBLBehaviorObservations,
        train_indices: object,
    ) -> "ExponentialHistoryBelief":
        if not isinstance(observations, IBLBehaviorObservations):
            raise TypeError("observations must be IBLBehaviorObservations")
        train = _validate_training_prefix(train_indices, observations.trial_ids.size)
        targets = observations.stimulus_side[train].astype(float)
        scores: list[float] = []
        for decay in self.decays:
            trace = causal_exponential_trace(observations.stimulus_side, decay)
            probability = np.clip((trace[train] + 1.0) / 2.0, 1e-6, 1.0 - 1e-6)
            score = -float(
                np.mean(
                    targets * np.log(probability)
                    + (1.0 - targets) * np.log(1.0 - probability)
                )
            )
            scores.append(score)
        self.decay_ = self.decays[int(np.argmin(scores))]
        self.train_predictive_nll_ = float(min(scores))
        self.fit_trial_ids_ = observations.trial_ids[train].copy()
        self._fitted = True
        return self

    def predict(self, observations: IBLBehaviorObservations) -> IBLBeliefPrediction:
        if not self._fitted:
            raise RuntimeError("ExponentialHistoryBelief must be fit first")
        if not isinstance(observations, IBLBehaviorObservations):
            raise TypeError("observations must be IBLBehaviorObservations")
        trace = causal_exponential_trace(observations.stimulus_side, self.decay_)
        estimated_rate = (trace + 1.0) / 2.0
        logits = (
            -0.5
            * (
                (estimated_rate[:, None] - IBL_BIASED_CONTEXT_LEVELS[None, :])
                / self.belief_width
            )
            ** 2
        )
        beliefs = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        # No evidence exists before the first trial.
        beliefs[0] = 0.5
        return IBLBeliefPrediction(
            beliefs,
            observations.trial_ids,
            "exponential_history",
            self.fit_trial_ids_,
            (
                ("decay", self.decay_),
                ("belief_width", self.belief_width),
                ("train_predictive_nll", self.train_predictive_nll_),
                ("uses_current_stimulus", False),
            ),
        )


def _normalize_rows(values: Array) -> Array:
    result = np.clip(np.asarray(values, dtype=float), 1e-12, None)
    return result / result.sum(axis=1, keepdims=True)


def _forward_backward(
    observations: Array,
    initial: Array,
    transition: Array,
    emission: Array,
) -> tuple[float, Array, Array]:
    n_trials = observations.size
    n_states = int(np.asarray(initial).size)
    if transition.shape != (n_states, n_states) or emission.shape != (n_states, 2):
        raise IBLBehaviorDataError("HMM parameter shapes are inconsistent")
    log_initial = np.log(np.clip(initial, 1e-300, None))
    log_transition = np.log(np.clip(transition, 1e-300, None))
    log_emission = np.log(np.clip(emission, 1e-300, None))
    alpha = np.empty((n_trials, n_states), dtype=float)
    alpha[0] = log_initial + log_emission[:, observations[0]]
    for trial in range(1, n_trials):
        alpha[trial] = log_emission[:, observations[trial]] + logsumexp(
            alpha[trial - 1][:, None] + log_transition, axis=0
        )
    likelihood = float(logsumexp(alpha[-1]))
    beta = np.zeros((n_trials, n_states), dtype=float)
    for trial in range(n_trials - 2, -1, -1):
        beta[trial] = logsumexp(
            log_transition
            + log_emission[:, observations[trial + 1]][None, :]
            + beta[trial + 1][None, :],
            axis=1,
        )
    log_gamma = alpha + beta - likelihood
    gamma = np.exp(log_gamma)
    xi_sum = np.zeros((n_states, n_states), dtype=float)
    for trial in range(n_trials - 1):
        log_xi = (
            alpha[trial][:, None]
            + log_transition
            + log_emission[:, observations[trial + 1]][None, :]
            + beta[trial + 1][None, :]
            - likelihood
        )
        xi_sum += np.exp(log_xi)
    return likelihood, gamma, xi_sum


class LearnedCategoricalHMM:
    """Task-informed unsupervised HMM with learned, emission-ordered states.

    Public IBL context rates initialize EM but no per-trial ``probabilityLeft``
    labels are accepted.  This distinction is retained in prediction metadata.
    """

    def __init__(
        self,
        *,
        max_iter: int = 200,
        tolerance: float = 1e-6,
        pseudocount: float = 0.1,
        n_restarts: int = 4,
        min_emission_gap: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.pseudocount = float(pseudocount)
        self.n_restarts = int(n_restarts)
        self.min_emission_gap = float(min_emission_gap)
        self.seed = int(seed)
        if self.max_iter < 1 or self.n_restarts < 1:
            raise IBLBehaviorDataError("max_iter and n_restarts must be positive")
        if (
            self.tolerance <= 0.0
            or self.pseudocount <= 0.0
            or not 0.0 < self.min_emission_gap < 1.0
        ):
            raise IBLBehaviorDataError("tolerance and pseudocount must be positive")
        self._fitted = False

    def _fit_restart(
        self, y: Array, restart: int
    ) -> tuple[float, Array, Array, Array, int, bool]:
        rng = np.random.default_rng(self.seed + 104729 * restart)
        rates = np.clip(
            IBL_BIASED_CONTEXT_LEVELS + rng.normal(0.0, 0.035, size=2),
            0.03,
            0.97,
        )
        emission = np.column_stack([1.0 - rates, rates])
        stay = float(rng.uniform(0.94, 0.995))
        transition = np.full((2, 2), 1.0 - stay)
        np.fill_diagonal(transition, stay)
        initial = np.full(2, 0.5)
        previous = -np.inf
        likelihood = -np.inf
        converged = False
        for iteration in range(1, self.max_iter + 1):
            likelihood, gamma, xi_sum = _forward_backward(
                y, initial, transition, emission
            )
            initial = (gamma[0] + self.pseudocount) / (
                gamma[0].sum() + 2.0 * self.pseudocount
            )
            transition = _normalize_rows(xi_sum + self.pseudocount)
            emission_counts = np.full((2, 2), self.pseudocount, dtype=float)
            for symbol in (0, 1):
                emission_counts[:, symbol] += gamma[y == symbol].sum(axis=0)
            emission = _normalize_rows(emission_counts)
            # Pseudocounts maximize a regularized objective, so the raw data
            # likelihood may move down by a tiny amount near the MAP optimum.
            if np.isfinite(previous) and abs(likelihood - previous) <= self.tolerance:
                converged = True
                break
            previous = likelihood
        final_likelihood, _, _ = _forward_backward(y, initial, transition, emission)
        return final_likelihood, initial, transition, emission, iteration, converged

    def fit(
        self,
        observations: IBLBehaviorObservations,
        train_indices: object,
    ) -> "LearnedCategoricalHMM":
        if not isinstance(observations, IBLBehaviorObservations):
            raise TypeError("observations must be IBLBehaviorObservations")
        train = _validate_training_prefix(train_indices, observations.trial_ids.size)
        y = observations.stimulus_side[train]
        candidates = [
            self._fit_restart(y, restart) for restart in range(self.n_restarts)
        ]
        likelihood, initial, transition, emission, iterations, converged = max(
            candidates, key=lambda item: item[0]
        )
        order = np.argsort(emission[:, 1])
        self.initial_ = np.array(initial[order], copy=True)
        self.transition_ = np.array(transition[np.ix_(order, order)], copy=True)
        self.emission_ = np.array(emission[order], copy=True)
        self.fit_trial_ids_ = observations.trial_ids[train].copy()
        self.train_log_likelihood_ = float(likelihood)
        self.n_iterations_ = int(iterations)
        self.converged_ = bool(converged)
        self.emission_gap_ = float(np.diff(self.emission_[:, 1])[0])
        self.identifiable_ = self.emission_gap_ >= self.min_emission_gap
        for value in (
            self.initial_,
            self.transition_,
            self.emission_,
            self.fit_trial_ids_,
        ):
            value.setflags(write=False)
        self._fitted = True
        return self

    def predict(self, observations: IBLBehaviorObservations) -> IBLBeliefPrediction:
        if not self._fitted:
            raise RuntimeError("LearnedCategoricalHMM must be fit first")
        if not isinstance(observations, IBLBehaviorObservations):
            raise TypeError("observations must be IBLBehaviorObservations")
        beliefs = np.empty((observations.trial_ids.size, 2), dtype=float)
        prior = np.array(self.initial_, copy=True)
        for trial, symbol in enumerate(observations.stimulus_side):
            # Store p(z_t | y_<t), then consume y_t for the next trial only.
            beliefs[trial] = prior
            posterior = prior * self.emission_[:, int(symbol)]
            posterior /= posterior.sum()
            prior = posterior @ self.transition_
        return IBLBeliefPrediction(
            beliefs,
            observations.trial_ids,
            "learned_categorical_hmm",
            self.fit_trial_ids_,
            (
                ("train_log_likelihood", self.train_log_likelihood_),
                ("n_iterations", self.n_iterations_),
                ("n_restarts", self.n_restarts),
                ("em_converged", self.converged_),
                ("emission_gap", self.emission_gap_),
                ("minimum_emission_gap", self.min_emission_gap),
                ("state_identifiable", self.identifiable_),
                ("uses_current_stimulus", False),
                ("state_order", "ascending_learned_left_emission"),
                ("known_context_rate_initialization_used", True),
                ("gate_fit_supervision", "task_informed_unsupervised_stimulus_only"),
            ),
        )


__all__ = [
    "IBL_BIASED_CONTEXT_LEVELS",
    "IBL_CONTEXT_LEVELS",
    "ContiguousBlockSplit",
    "ExponentialHistoryBelief",
    "IBLBehaviorDataError",
    "IBLBehaviorObservations",
    "IBLBehaviorSession",
    "IBLBeliefPrediction",
    "LearnedCategoricalHMM",
    "NoMemoryBelief",
    "causal_exponential_trace",
    "contiguous_block_ids",
    "contiguous_block_split",
    "load_ibl_behavior_table",
]
