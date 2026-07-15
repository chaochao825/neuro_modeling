"""Leakage-safe hierarchical one-step Poisson count dynamics.

This module is deliberately *not* a latent Poisson LDS: it does not define a
latent process-noise distribution, perform filtering/smoothing, or report a
marginal sequence likelihood.  It is a teacher-forced one-step conditional
model.  Region-averaged activity is projected through train-only scaling/PCA,
closed-form ridge dynamics predict the next latent coordinate, and a
session-specific log-rate map produces conditional Poisson rates.

Beliefs must be past-only soft two-state probabilities supplied by the caller.
Optional generic nuisance controls must likewise be past-safe.  The model has
no API for privileged state labels.  Every transition remains inside a whole
trial selected by an explicit chronological train/test trial-ID split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy.special import gammaln


Array = np.ndarray
Family = Literal["common", "shared", "full"]
_FAMILIES = frozenset({"common", "shared", "full"})
_SHA256 = frozenset("0123456789abcdef")


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _freeze(array: Array) -> Array:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _array_sha256(array: Array) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_sha256(value: object, *, name: str) -> str:
    text = _text(value, name=name).lower()
    if len(text) != 64 or any(character not in _SHA256 for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _trial_id_tuple(
    values: Sequence[object] | Array, *, name: str
) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of trial IDs")
    identifiers = tuple(values)
    if not identifiers:
        raise ValueError(f"{name} must not be empty")
    try:
        unique = set(identifiers)
    except TypeError as error:
        raise TypeError(f"{name} values must be hashable") from error
    if len(unique) != len(identifiers):
        raise ValueError(f"{name} must not contain duplicates")
    return identifiers


@dataclass(frozen=True, slots=True)
class BeliefFitReceipt:
    """Auditable contract for a train-only, past-only belief trajectory."""

    method: str
    fit_trial_ids: tuple[object, ...]
    observation_fit_trial_ids: tuple[object, ...]
    input_columns: tuple[str, ...]
    uses_current_trial_stimulus: bool
    uses_future_trials: bool
    accessed_true_context: bool
    checkpoint_sha256: str
    belief_sha256: str

    def __post_init__(self) -> None:
        method = _text(self.method, name="belief method")
        fit_ids = _trial_id_tuple(self.fit_trial_ids, name="belief fit_trial_ids")
        observation_fit_ids = _trial_id_tuple(
            self.observation_fit_trial_ids,
            name="belief observation_fit_trial_ids",
        )
        columns = tuple(
            _text(column, name="belief input column") for column in self.input_columns
        )
        if columns != ("stimulus_side_lag1",):
            raise ValueError(
                "belief input_columns must be exactly ('stimulus_side_lag1',)"
            )
        flags = (
            self.uses_current_trial_stimulus,
            self.uses_future_trials,
            self.accessed_true_context,
        )
        if not all(isinstance(value, (bool, np.bool_)) for value in flags):
            raise TypeError("belief provenance flags must be booleans")
        if any(bool(value) for value in flags):
            raise ValueError(
                "belief receipt must exclude current/future stimulus and true context"
            )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "fit_trial_ids", fit_ids)
        object.__setattr__(self, "observation_fit_trial_ids", observation_fit_ids)
        object.__setattr__(self, "input_columns", columns)
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _validate_sha256(self.checkpoint_sha256, name="checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "belief_sha256",
            _validate_sha256(self.belief_sha256, name="belief_sha256"),
        )

    @classmethod
    def bind(
        cls,
        beliefs: Array,
        *,
        method: str,
        fit_trial_ids: Sequence[object],
        observation_fit_trial_ids: Sequence[object] | None = None,
        checkpoint_payload: object,
    ) -> "BeliefFitReceipt":
        """Bind beliefs to a serializable checkpoint without accepting truth labels."""

        encoded = json.dumps(
            checkpoint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return cls(
            method=method,
            fit_trial_ids=tuple(fit_trial_ids),
            observation_fit_trial_ids=tuple(
                fit_trial_ids
                if observation_fit_trial_ids is None
                else observation_fit_trial_ids
            ),
            input_columns=("stimulus_side_lag1",),
            uses_current_trial_stimulus=False,
            uses_future_trials=False,
            accessed_true_context=False,
            checkpoint_sha256=hashlib.sha256(encoded).hexdigest(),
            belief_sha256=_array_sha256(np.asarray(beliefs, dtype=float)),
        )


@dataclass(frozen=True, slots=True)
class NeuralCountSession:
    """Immutable multi-trial neural counts and target-free causal covariates.

    ``counts`` has shape ``[trial, time, unit]``.  ``beliefs`` has shape
    ``[trial, 2]`` and must contain strictly soft probabilities.  ``controls``
    is optional and has shape ``[trial, nuisance_feature]``; it is used only in
    the session observation map, never to fit the shared latent basis.
    """

    session_id: str
    animal_id: str
    counts: Array
    unit_regions: tuple[str, ...]
    beliefs: Array
    trial_ids: Array
    belief_receipt: BeliefFitReceipt
    controls: Array | None = None

    def __post_init__(self) -> None:
        session_id = _text(self.session_id, name="session_id")
        animal_id = _text(self.animal_id, name="animal_id")
        raw_counts = np.asarray(self.counts)
        if (
            raw_counts.ndim != 3
            or min(raw_counts.shape, default=0) < 1
            or raw_counts.shape[1] < 2
        ):
            raise ValueError("counts must have shape [trial, time>=2, unit]")
        if raw_counts.dtype.kind not in {"i", "u"} or np.any(raw_counts < 0):
            raise ValueError("counts must contain non-negative integers")
        counts = np.asarray(raw_counts, dtype=np.int64)
        regions = tuple(
            _text(value, name="unit region").lower() for value in self.unit_regions
        )
        if len(regions) != counts.shape[2]:
            raise ValueError("unit_regions must provide one label per unit")
        beliefs = np.asarray(self.beliefs, dtype=float)
        if beliefs.shape != (counts.shape[0], 2) or not np.isfinite(beliefs).all():
            raise ValueError("beliefs must be finite with shape [trial, 2]")
        if np.any((beliefs <= 0.0) | (beliefs >= 1.0)):
            raise ValueError("beliefs must be strictly soft probabilities")
        if not np.allclose(beliefs.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("belief rows must sum to one")
        raw_ids = np.asarray(self.trial_ids, dtype=object)
        if raw_ids.ndim != 1 or raw_ids.shape[0] != counts.shape[0]:
            raise ValueError("trial_ids must be a vector with one ID per trial")
        _trial_id_tuple(raw_ids.tolist(), name="trial_ids")
        if not isinstance(self.belief_receipt, BeliefFitReceipt):
            raise TypeError("belief_receipt must be a BeliefFitReceipt")
        if self.belief_receipt.belief_sha256 != _array_sha256(beliefs):
            raise ValueError("belief receipt does not bind the supplied beliefs")
        positions = {value: index for index, value in enumerate(raw_ids.tolist())}
        if any(item not in positions for item in self.belief_receipt.fit_trial_ids):
            raise ValueError("belief fit_trial_ids must belong to this session")
        fit_positions = [positions[item] for item in self.belief_receipt.fit_trial_ids]
        if fit_positions != list(range(len(fit_positions))):
            raise ValueError("belief fit_trial_ids must be a chronological prefix")
        controls: Array | None
        if self.controls is None:
            controls = None
        else:
            controls = np.asarray(self.controls, dtype=float)
            if (
                controls.ndim != 2
                or controls.shape[0] != counts.shape[0]
                or controls.shape[1] < 1
                or not np.isfinite(controls).all()
            ):
                raise ValueError(
                    "controls must be finite with shape [trial, positive feature dim]"
                )
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "animal_id", animal_id)
        object.__setattr__(self, "counts", _freeze(counts))
        object.__setattr__(self, "unit_regions", regions)
        object.__setattr__(self, "beliefs", _freeze(beliefs))
        object.__setattr__(self, "trial_ids", _freeze(raw_ids))
        object.__setattr__(
            self, "controls", None if controls is None else _freeze(controls)
        )

    @property
    def n_trials(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_time(self) -> int:
        return int(self.counts.shape[1])

    @property
    def n_units(self) -> int:
        return int(self.counts.shape[2])

    @property
    def control_dim(self) -> int:
        return 0 if self.controls is None else int(self.controls.shape[1])


@dataclass(frozen=True, slots=True)
class TrialBlockSplit:
    """Disjoint whole-trial IDs, supplied in their chronological order."""

    train_trial_ids: tuple[object, ...]
    test_trial_ids: tuple[object, ...]
    ordered_trial_ids: tuple[object, ...]
    ordered_block_ids: tuple[object, ...]
    split_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        train = _trial_id_tuple(self.train_trial_ids, name="train_trial_ids")
        test = _trial_id_tuple(self.test_trial_ids, name="test_trial_ids")
        ordered = _trial_id_tuple(self.ordered_trial_ids, name="ordered_trial_ids")
        blocks = tuple(self.ordered_block_ids)
        if len(blocks) != len(ordered):
            raise ValueError("ordered block IDs must align with ordered trial IDs")
        try:
            hash(tuple(blocks))
        except TypeError as error:
            raise TypeError("ordered block IDs must be hashable") from error
        try:
            overlap = set(train) & set(test)
        except TypeError as error:
            raise TypeError("trial split IDs must be hashable") from error
        if overlap:
            raise ValueError(
                f"train/test trial IDs overlap: {sorted(map(str, overlap))}"
            )
        if train + test != ordered:
            raise ValueError(
                "train/test IDs must completely cover one chronological prefix/suffix"
            )
        if blocks[len(train) - 1] == blocks[len(train)]:
            raise ValueError("train/test boundary must not split a whole block")
        completed: set[object] = set()
        previous = blocks[0]
        for block in blocks[1:]:
            if block == previous:
                continue
            completed.add(previous)
            if block in completed:
                raise ValueError("block IDs must form contiguous runs")
            previous = block
        payload = json.dumps(
            {
                "trial_ids": [str(value) for value in ordered],
                "block_ids": [str(value) for value in blocks],
                "n_train": len(train),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        object.__setattr__(self, "train_trial_ids", train)
        object.__setattr__(self, "test_trial_ids", test)
        object.__setattr__(self, "ordered_trial_ids", ordered)
        object.__setattr__(self, "ordered_block_ids", blocks)
        object.__setattr__(
            self, "split_fingerprint", hashlib.sha256(payload).hexdigest()
        )

    @classmethod
    def chronological_holdout(
        cls,
        session: NeuralCountSession,
        *,
        block_ids: Sequence[object],
        n_train: int,
    ) -> "TrialBlockSplit":
        if isinstance(n_train, (bool, np.bool_)) or not isinstance(
            n_train, (int, np.integer)
        ):
            raise TypeError("n_train must be an integer")
        if not 1 <= n_train < session.n_trials:
            raise ValueError("n_train must leave non-empty train and test blocks")
        ids = tuple(session.trial_ids.tolist())
        blocks = tuple(block_ids)
        return cls(ids[: int(n_train)], ids[int(n_train) :], ids, blocks)


@dataclass(frozen=True, slots=True)
class SessionCountPrediction:
    session_id: str
    trial_ids: tuple[object, ...]
    latent_prediction: Array
    rates: Array

    def __post_init__(self) -> None:
        latent = np.asarray(self.latent_prediction, dtype=float)
        rates = np.asarray(self.rates, dtype=float)
        if latent.ndim != 3 or rates.ndim != 3:
            raise ValueError("prediction arrays must have [trial, transition, feature]")
        if latent.shape[:2] != rates.shape[:2] or latent.shape[0] != len(
            self.trial_ids
        ):
            raise ValueError("prediction arrays and trial IDs must align")
        if not np.isfinite(latent).all() or not np.isfinite(rates).all():
            raise ValueError("predictions must be finite")
        if np.any(rates <= 0.0):
            raise ValueError("Poisson rates must be positive")
        object.__setattr__(self, "latent_prediction", _freeze(latent))
        object.__setattr__(self, "rates", _freeze(rates))


@dataclass(frozen=True, slots=True)
class SessionCountMetrics:
    session_id: str
    animal_id: str
    n_transitions: int
    n_count_observations: int
    log_likelihood: float
    null_log_likelihood: float
    saturated_log_likelihood: float
    nll_per_count: float
    pseudo_r2: float
    closure_mse: float


@dataclass(frozen=True, slots=True)
class HierarchicalCountScore:
    family: Family
    per_session: tuple[SessionCountMetrics, ...]
    parameter_count: int
    nll_per_count: float
    pseudo_r2: float
    closure_mse: float
    likelihood_kind: str = "one_step_conditional_poisson"
    full_latent_lds: bool = False
    pooled_metric_for_registered_inference: bool = False
    registered_inference_unit: str = "animal_with_session_nested"


def poisson_log_likelihood(counts: Array, rates: Array) -> float:
    """Exact independent Poisson log likelihood, including ``gammaln(y+1)``."""

    raw_counts = np.asarray(counts)
    raw_rates = np.asarray(rates, dtype=float)
    if raw_counts.dtype.kind not in {"i", "u"} or np.any(raw_counts < 0):
        raise ValueError("counts must contain non-negative integers")
    try:
        observations, means = np.broadcast_arrays(raw_counts, raw_rates)
    except ValueError as error:
        raise ValueError("counts and rates must be broadcast-compatible") from error
    if not np.isfinite(means).all() or np.any(means <= 0.0):
        raise ValueError("rates must be finite and strictly positive")
    values = observations.astype(float, copy=False)
    return float(np.sum(values * np.log(means) - means - gammaln(values + 1.0)))


def _saturated_poisson_log_likelihood(counts: Array) -> float:
    values = np.asarray(counts, dtype=float)
    positive = values > 0.0
    terms = -values - gammaln(values + 1.0)
    terms[positive] += values[positive] * np.log(values[positive])
    return float(np.sum(terms))


def _ridge_fit(
    design: Array,
    target: Array,
    ridge: float,
    *,
    unpenalized: Sequence[int] = (-1,),
) -> Array:
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    penalty = ridge * np.eye(x.shape[1], dtype=float)
    for index in unpenalized:
        penalty[index, index] = 0.0
    gram = x.T @ x + penalty
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


@dataclass(frozen=True)
class _Transitions:
    region_current: Array
    region_following: Array
    following_counts: Array
    beliefs: Array
    controls: Array
    n_trials: int
    n_steps: int


class HierarchicalCountDynamics:
    """Union-region-anchored common/shared/full conditional count dynamics.

    A session may omit any subset of the configured anatomical anchors as long
    as it contains at least one.  Structurally missing region means are imputed
    with statistics fitted only on the pooled training transitions, which maps
    missing values to zero after standardization.  Session-level presence masks
    are retained as diagnostics and are never inferred from held-out counts.
    """

    used_bptt = False
    likelihood_kind = "one_step_conditional_poisson"
    full_latent_lds = False
    region_imputation_strategy = "pooled_training_fold_region_mean"

    def __init__(
        self,
        family: Family,
        *,
        common_regions: Sequence[str],
        latent_dim: int,
        ridge: float = 1e-3,
        rate_clip: tuple[float, float] = (-12.0, 12.0),
        seed: int = 0,
    ) -> None:
        if family not in _FAMILIES:
            raise ValueError(f"family must be one of {sorted(_FAMILIES)}")
        regions = tuple(
            _text(value, name="common region").lower() for value in common_regions
        )
        if not regions or len(set(regions)) != len(regions):
            raise ValueError("common_regions must be non-empty and unique")
        if isinstance(latent_dim, (bool, np.bool_)) or not isinstance(
            latent_dim, (int, np.integer)
        ):
            raise TypeError("latent_dim must be an integer")
        if not 1 <= latent_dim <= len(regions):
            raise ValueError("latent_dim must lie in [1, number of common regions]")
        if not np.isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")
        if (
            len(rate_clip) != 2
            or not np.isfinite(rate_clip).all()
            or rate_clip[0] >= rate_clip[1]
        ):
            raise ValueError("rate_clip must be an increasing finite pair")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.family: Family = family
        self.common_regions = regions
        self.latent_dim = int(latent_dim)
        self.ridge = float(ridge)
        self.rate_clip = (float(rate_clip[0]), float(rate_clip[1]))
        self.seed = int(seed)

        self.scaler_mean_: Array | None = None
        self.scaler_scale_: Array | None = None
        self.pca_mean_: Array | None = None
        self.pca_components_: Array | None = None
        self.region_train_observation_counts_: Array | None = None
        self.observation_matrices_: dict[str, Array] = {}
        self.null_log_rates_: dict[str, Array] = {}
        self.transition_matrices_: dict[str, Array] = {}
        self.session_animals_: dict[str, str] = {}
        self.session_regions_: dict[str, tuple[str, ...]] = {}
        self.session_region_presence_: dict[str, tuple[bool, ...]] = {}
        self.session_control_dims_: dict[str, int] = {}
        self.train_trial_ids_: dict[str, tuple[object, ...]] = {}
        self.split_fingerprints_: dict[str, str] = {}
        self.belief_receipts_: dict[str, BeliefFitReceipt] = {}
        self.fit_fingerprint_: str | None = None

    @staticmethod
    def _session_map(
        sessions: Sequence[NeuralCountSession],
    ) -> dict[str, NeuralCountSession]:
        values = tuple(sessions)
        if not values or not all(
            isinstance(item, NeuralCountSession) for item in values
        ):
            raise TypeError("sessions must contain NeuralCountSession objects")
        result = {item.session_id: item for item in values}
        if len(result) != len(values):
            raise ValueError("session_id values must be unique")
        return result

    @staticmethod
    def _validate_split_map(
        sessions: Mapping[str, NeuralCountSession],
        splits: Mapping[str, TrialBlockSplit],
    ) -> None:
        if not isinstance(splits, Mapping):
            raise TypeError("splits must map session IDs to TrialBlockSplit")
        if set(splits) != set(sessions):
            raise ValueError("split/session IDs must match exactly")
        if not all(isinstance(value, TrialBlockSplit) for value in splits.values()):
            raise TypeError("every split must be a TrialBlockSplit")

    @staticmethod
    def _indices(
        session: NeuralCountSession,
        identifiers: Sequence[object],
        *,
        name: str,
    ) -> Array:
        lookup = {
            value: index for index, value in enumerate(session.trial_ids.tolist())
        }
        try:
            missing = [value for value in identifiers if value not in lookup]
        except TypeError as error:
            raise TypeError(f"{name} trial IDs must be hashable") from error
        if missing:
            raise ValueError(f"{name} contains IDs absent from {session.session_id!r}")
        positions = np.asarray([lookup[value] for value in identifiers], dtype=int)
        if positions.size > 1 and np.any(np.diff(positions) <= 0):
            raise ValueError(f"{name} IDs must follow chronological session order")
        return positions

    def _validate_session_regions(
        self, session: NeuralCountSession
    ) -> tuple[bool, ...]:
        labels = set(session.unit_regions)
        presence = tuple(region in labels for region in self.common_regions)
        if not any(presence):
            raise ValueError(
                f"session {session.session_id!r} contains no configured anchor region"
            )
        return presence

    def _region_values(self, session: NeuralCountSession, indices: Array) -> Array:
        selected = session.counts[indices]
        values = np.full(
            (len(indices), session.n_time, len(self.common_regions)),
            dtype=float,
            fill_value=np.nan,
        )
        labels = np.asarray(session.unit_regions, dtype=object)
        for region_index, region in enumerate(self.common_regions):
            unit_mask = labels == region
            if np.any(unit_mask):
                values[:, :, region_index] = selected[:, :, unit_mask].mean(axis=2)
        return np.log1p(values)

    def _transitions(self, session: NeuralCountSession, indices: Array) -> _Transitions:
        region = self._region_values(session, indices)
        n_trials, n_time, _ = region.shape
        n_steps = n_time - 1
        beliefs = np.repeat(session.beliefs[indices], n_steps, axis=0)
        if session.controls is None:
            controls = np.empty((n_trials * n_steps, 0), dtype=float)
        else:
            controls = np.repeat(session.controls[indices], n_steps, axis=0)
        return _Transitions(
            region_current=region[:, :-1].reshape(-1, region.shape[-1]),
            region_following=region[:, 1:].reshape(-1, region.shape[-1]),
            following_counts=session.counts[indices, 1:].reshape(-1, session.n_units),
            beliefs=beliefs,
            controls=controls,
            n_trials=n_trials,
            n_steps=n_steps,
        )

    def _fit_preprocessing(self, transitions: Sequence[_Transitions]) -> None:
        values = np.concatenate(
            [
                np.concatenate((item.region_current, item.region_following), axis=0)
                for item in transitions
            ],
            axis=0,
        )
        if values.shape[0] < self.latent_dim:
            raise ValueError("too few training transitions for latent_dim")
        if np.isinf(values).any():
            raise ValueError("region activity must not contain infinite values")
        observed = np.isfinite(values)
        observation_counts = observed.sum(axis=0)
        if np.any(observation_counts < 1):
            missing = tuple(
                self.common_regions[index]
                for index in np.flatnonzero(observation_counts < 1)
            )
            raise ValueError(
                f"union anchors lack training-fold observations: {missing!r}"
            )
        mean = np.divide(
            np.where(observed, values, 0.0).sum(axis=0),
            observation_counts,
        )
        centered = np.where(observed, values - mean, 0.0)
        scale = np.sqrt(
            np.divide((centered * centered).sum(axis=0), observation_counts)
        )
        scale[scale < 1e-8] = 1.0
        imputed = np.where(observed, values, mean)
        standardized = (imputed - mean) / scale
        pca_mean = standardized.mean(axis=0)
        _, _, right = np.linalg.svd(standardized - pca_mean, full_matrices=False)
        components = right[: self.latent_dim].copy()
        # Fix the arbitrary SVD sign for deterministic serialized parameters.
        for index in range(len(components)):
            anchor = int(np.argmax(np.abs(components[index])))
            if components[index, anchor] < 0.0:
                components[index] *= -1.0
        self.scaler_mean_ = _freeze(mean)
        self.scaler_scale_ = _freeze(scale)
        self.pca_mean_ = _freeze(pca_mean)
        self.pca_components_ = _freeze(components)
        self.region_train_observation_counts_ = _freeze(
            observation_counts.astype(np.int64)
        )

    def _latent(self, region_values: Array) -> Array:
        if (
            self.scaler_mean_ is None
            or self.scaler_scale_ is None
            or self.pca_mean_ is None
            or self.pca_components_ is None
        ):
            raise RuntimeError("fit must be called before latent projection")
        values = np.asarray(region_values, dtype=float)
        if values.shape[-1] != len(self.common_regions):
            raise ValueError("region_values do not match the configured anchor basis")
        if np.isinf(values).any():
            raise ValueError("region_values must not contain infinite values")
        imputed = np.where(np.isnan(values), self.scaler_mean_, values)
        if not np.isfinite(imputed).all():
            raise ValueError("region_values contain unsupported non-finite values")
        standardized = (imputed - self.scaler_mean_) / self.scaler_scale_
        return (standardized - self.pca_mean_) @ self.pca_components_.T

    @staticmethod
    def _base_design(latent: Array) -> Array:
        return np.column_stack((latent, np.ones(len(latent))))

    def _switch_design(self, latent: Array, beliefs: Array) -> Array:
        base = self._base_design(latent)
        return np.concatenate(
            [beliefs[:, state, None] * base for state in range(2)], axis=1
        )

    def fit(
        self,
        sessions: Sequence[NeuralCountSession],
        splits: Mapping[str, TrialBlockSplit],
    ) -> "HierarchicalCountDynamics":
        """Fit every parameter from train trial transitions only."""

        by_id = self._session_map(sessions)
        self._validate_split_map(by_id, splits)
        training: dict[str, _Transitions] = {}
        for session_id, session in by_id.items():
            self._validate_session_regions(session)
            split = splits[session_id]
            if split.ordered_trial_ids != tuple(session.trial_ids.tolist()):
                raise ValueError(
                    "split ordered_trial_ids must exactly bind the session trial order"
                )
            if session.belief_receipt.fit_trial_ids != split.train_trial_ids:
                raise ValueError(
                    "belief fit_trial_ids must exactly match the model train block"
                )
            train_indices = self._indices(
                session, split.train_trial_ids, name="train_trial_ids"
            )
            test_indices = self._indices(
                session, split.test_trial_ids, name="test_trial_ids"
            )
            if int(train_indices[-1]) >= int(test_indices[0]):
                raise ValueError(
                    "the chronological train block must precede the test block"
                )
            training[session_id] = self._transitions(session, train_indices)

        self._fit_preprocessing(tuple(training.values()))
        latent_pairs: dict[str, tuple[Array, Array]] = {}
        for session_id, item in training.items():
            latent_pairs[session_id] = (
                self._latent(item.region_current),
                self._latent(item.region_following),
            )

        self.transition_matrices_ = {}
        if self.family == "common":
            current = np.concatenate(
                [latent_pairs[key][0] for key in sorted(latent_pairs)], axis=0
            )
            following = np.concatenate(
                [latent_pairs[key][1] for key in sorted(latent_pairs)], axis=0
            )
            coefficient = _ridge_fit(self._base_design(current), following, self.ridge)
            self.transition_matrices_["common"] = _freeze(coefficient.T)
        elif self.family == "shared":
            design = np.concatenate(
                [
                    self._switch_design(latent_pairs[key][0], training[key].beliefs)
                    for key in sorted(latent_pairs)
                ],
                axis=0,
            )
            following = np.concatenate(
                [latent_pairs[key][1] for key in sorted(latent_pairs)], axis=0
            )
            block = self.latent_dim + 1
            coefficient = _ridge_fit(
                design,
                following,
                self.ridge,
                unpenalized=(block - 1, 2 * block - 1),
            )
            for state in range(2):
                self.transition_matrices_[f"state_{state}"] = _freeze(
                    coefficient[state * block : (state + 1) * block].T
                )
        else:
            for session_id in sorted(latent_pairs):
                design = self._switch_design(
                    latent_pairs[session_id][0], training[session_id].beliefs
                )
                block = self.latent_dim + 1
                coefficient = _ridge_fit(
                    design,
                    latent_pairs[session_id][1],
                    self.ridge,
                    unpenalized=(block - 1, 2 * block - 1),
                )
                for state in range(2):
                    self.transition_matrices_[f"{session_id}:state_{state}"] = _freeze(
                        coefficient[state * block : (state + 1) * block].T
                    )

        self.observation_matrices_ = {}
        self.null_log_rates_ = {}
        self.session_animals_ = {}
        self.session_regions_ = {}
        self.session_region_presence_ = {}
        self.session_control_dims_ = {}
        self.train_trial_ids_ = {}
        for session_id in sorted(by_id):
            session = by_id[session_id]
            item = training[session_id]
            following_latent = latent_pairs[session_id][1]
            observation_design = np.column_stack(
                (following_latent, item.controls, np.ones(len(following_latent)))
            )
            log_counts = np.log(item.following_counts + 0.5)
            coefficient = _ridge_fit(observation_design, log_counts, self.ridge)
            self.observation_matrices_[session_id] = _freeze(coefficient.T)
            mean_rate = np.maximum(item.following_counts.mean(axis=0), 1e-8)
            self.null_log_rates_[session_id] = _freeze(np.log(mean_rate))
            self.session_animals_[session_id] = session.animal_id
            self.session_regions_[session_id] = session.unit_regions
            self.session_region_presence_[session_id] = self._validate_session_regions(
                session
            )
            self.session_control_dims_[session_id] = session.control_dim
            self.train_trial_ids_[session_id] = tuple(
                splits[session_id].train_trial_ids
            )
            self.split_fingerprints_[session_id] = splits[session_id].split_fingerprint
            self.belief_receipts_[session_id] = session.belief_receipt

        digest = hashlib.sha256()
        for array in self._parameter_arrays():
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        self.fit_fingerprint_ = digest.hexdigest()
        return self

    def _parameter_arrays(self) -> tuple[Array, ...]:
        if self.pca_components_ is None:
            raise RuntimeError("model is not fitted")
        preprocessing = (
            self.scaler_mean_,
            self.scaler_scale_,
            self.pca_mean_,
            self.pca_components_,
        )
        assert all(value is not None for value in preprocessing)
        return (
            tuple(value for value in preprocessing if value is not None)
            + tuple(
                self.observation_matrices_[key]
                for key in sorted(self.observation_matrices_)
            )
            + tuple(self.null_log_rates_[key] for key in sorted(self.null_log_rates_))
            + tuple(
                self.transition_matrices_[key]
                for key in sorted(self.transition_matrices_)
            )
        )

    def parameter_count(self) -> int:
        """Count scaler, PCA, session observations, and dynamics parameters.

        The train-only null-rate comparator used solely to define pseudo-R2 is
        intentionally not charged to the predictive model.
        """

        if self.pca_components_ is None:
            raise RuntimeError("model is not fitted")
        preprocessing = (
            self.scaler_mean_,
            self.scaler_scale_,
            self.pca_mean_,
            self.pca_components_,
        )
        assert all(value is not None for value in preprocessing)
        arrays = tuple(value for value in preprocessing if value is not None)
        arrays += tuple(self.observation_matrices_.values())
        arrays += tuple(self.transition_matrices_.values())
        return int(sum(array.size for array in arrays))

    def _validate_scoring_session(self, session: NeuralCountSession) -> None:
        if session.session_id not in self.observation_matrices_:
            raise ValueError(f"unseen session {session.session_id!r}")
        if session.animal_id != self.session_animals_[session.session_id]:
            raise ValueError("animal_id differs from the fitted session")
        if session.unit_regions != self.session_regions_[session.session_id]:
            raise ValueError("unit regions/order differ from the fitted session")
        if session.control_dim != self.session_control_dims_[session.session_id]:
            raise ValueError("control dimension differs from the fitted session")
        if session.belief_receipt != self.belief_receipts_[session.session_id]:
            raise ValueError("belief receipt differs from the fitted session")
        self._validate_session_regions(session)

    def _validate_scoring_split(self, session_id: str, split: TrialBlockSplit) -> None:
        expected = self.split_fingerprints_.get(session_id)
        if expected is None:
            raise ValueError(f"unseen session {session_id!r}")
        if split.split_fingerprint != expected:
            raise ValueError("scoring split differs from the fitted split")

    def _predict_latent(self, session_id: str, current: Array, beliefs: Array) -> Array:
        base = self._base_design(current)
        if self.family == "common":
            return base @ self.transition_matrices_["common"].T
        if self.family == "shared":
            matrices = [
                self.transition_matrices_[f"state_{state}"] for state in range(2)
            ]
        else:
            matrices = [
                self.transition_matrices_[f"{session_id}:state_{state}"]
                for state in range(2)
            ]
        return sum(
            beliefs[:, state, None] * (base @ matrices[state].T) for state in range(2)
        )

    @staticmethod
    def _heldout_belief_override(value: Array, *, n_trials: int) -> Array:
        beliefs = np.asarray(value, dtype=float)
        if beliefs.shape != (n_trials, 2) or not np.isfinite(beliefs).all():
            raise ValueError(
                "held-out belief override must be finite with shape [heldout_trial, 2]"
            )
        if np.any((beliefs <= 0.0) | (beliefs >= 1.0)):
            raise ValueError("held-out belief override must remain strictly soft")
        if not np.allclose(beliefs.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("held-out belief override rows must sum to one")
        return beliefs

    def _predict_one(
        self,
        session: NeuralCountSession,
        identifiers: Sequence[object],
        *,
        heldout_beliefs: Array | None = None,
    ) -> tuple[SessionCountPrediction, _Transitions, Array]:
        self._validate_scoring_session(session)
        indices = self._indices(session, identifiers, name="test_trial_ids")
        item = self._transitions(session, indices)
        if heldout_beliefs is not None:
            override = self._heldout_belief_override(
                heldout_beliefs, n_trials=len(indices)
            )
            item = _Transitions(
                region_current=item.region_current,
                region_following=item.region_following,
                following_counts=item.following_counts,
                beliefs=np.repeat(override, item.n_steps, axis=0),
                controls=item.controls,
                n_trials=item.n_trials,
                n_steps=item.n_steps,
            )
        current = self._latent(item.region_current)
        following = self._latent(item.region_following)
        latent_prediction = self._predict_latent(
            session.session_id, current, item.beliefs
        )
        observation_design = np.column_stack(
            (latent_prediction, item.controls, np.ones(len(latent_prediction)))
        )
        log_rates = (
            observation_design @ self.observation_matrices_[session.session_id].T
        )
        rates = np.exp(np.clip(log_rates, *self.rate_clip))
        prediction = SessionCountPrediction(
            session_id=session.session_id,
            trial_ids=tuple(identifiers),
            latent_prediction=latent_prediction.reshape(
                item.n_trials, item.n_steps, self.latent_dim
            ),
            rates=rates.reshape(item.n_trials, item.n_steps, session.n_units),
        )
        return prediction, item, following

    def predict(
        self,
        sessions: Sequence[NeuralCountSession],
        splits: Mapping[str, TrialBlockSplit],
    ) -> Mapping[str, SessionCountPrediction]:
        """Predict held-out next-bin rates conditioned on held-out current bins."""

        by_id = self._session_map(sessions)
        self._validate_split_map(by_id, splits)
        predictions: dict[str, SessionCountPrediction] = {}
        for session_id in sorted(by_id):
            self._validate_scoring_split(session_id, splits[session_id])
            prediction, _, _ = self._predict_one(
                by_id[session_id], splits[session_id].test_trial_ids
            )
            predictions[session_id] = prediction
        return predictions

    @staticmethod
    def _pseudo_r2(model_ll: float, null_ll: float, saturated_ll: float) -> float:
        denominator = saturated_ll - null_ll
        if denominator <= 1e-12:
            return float("nan")
        return float(1.0 - (saturated_ll - model_ll) / denominator)

    def _score(
        self,
        sessions: Sequence[NeuralCountSession],
        splits: Mapping[str, TrialBlockSplit],
        *,
        heldout_beliefs: Mapping[str, Array] | None,
    ) -> HierarchicalCountScore:
        by_id = self._session_map(sessions)
        self._validate_split_map(by_id, splits)
        if heldout_beliefs is not None:
            if self.family == "common":
                raise ValueError(
                    "common dynamics do not consume beliefs and cannot be belief-intervened"
                )
            if set(heldout_beliefs) != set(by_id):
                raise ValueError(
                    "held-out belief overrides must cover exactly the scored sessions"
                )
        metrics: list[SessionCountMetrics] = []
        total_model = total_null = total_saturated = 0.0
        total_count_observations = 0
        total_closure_squared = 0.0
        total_latent_values = 0
        for session_id in sorted(by_id):
            session = by_id[session_id]
            self._validate_scoring_split(session_id, splits[session_id])
            prediction, item, following_latent = self._predict_one(
                session,
                splits[session_id].test_trial_ids,
                heldout_beliefs=(
                    None
                    if heldout_beliefs is None
                    else heldout_beliefs[session_id]
                ),
            )
            observed = item.following_counts.reshape(prediction.rates.shape)
            model_ll = poisson_log_likelihood(observed, prediction.rates)
            null_rates = np.exp(self.null_log_rates_[session_id])
            null_ll = poisson_log_likelihood(observed, null_rates)
            saturated_ll = _saturated_poisson_log_likelihood(observed)
            latent_flat = prediction.latent_prediction.reshape(-1, self.latent_dim)
            closure = float(np.mean(np.square(latent_flat - following_latent)))
            n_observations = int(observed.size)
            metrics.append(
                SessionCountMetrics(
                    session_id=session_id,
                    animal_id=session.animal_id,
                    n_transitions=int(item.n_trials * item.n_steps),
                    n_count_observations=n_observations,
                    log_likelihood=model_ll,
                    null_log_likelihood=null_ll,
                    saturated_log_likelihood=saturated_ll,
                    nll_per_count=-model_ll / n_observations,
                    pseudo_r2=self._pseudo_r2(model_ll, null_ll, saturated_ll),
                    closure_mse=closure,
                )
            )
            total_model += model_ll
            total_null += null_ll
            total_saturated += saturated_ll
            total_count_observations += n_observations
            total_closure_squared += float(
                np.sum(np.square(latent_flat - following_latent))
            )
            total_latent_values += following_latent.size
        return HierarchicalCountScore(
            family=self.family,
            per_session=tuple(metrics),
            parameter_count=self.parameter_count(),
            nll_per_count=-total_model / total_count_observations,
            pseudo_r2=self._pseudo_r2(total_model, total_null, total_saturated),
            closure_mse=total_closure_squared / total_latent_values,
        )

    def score(
        self,
        sessions: Sequence[NeuralCountSession],
        splits: Mapping[str, TrialBlockSplit],
    ) -> HierarchicalCountScore:
        """Score exact teacher-forced held-out one-step Poisson likelihoods."""

        return self._score(sessions, splits, heldout_beliefs=None)

    def score_heldout_belief_counterfactual(
        self,
        sessions: Sequence[NeuralCountSession],
        splits: Mapping[str, TrialBlockSplit],
        heldout_beliefs: Mapping[str, Array],
    ) -> HierarchicalCountScore:
        """Score a post-fit belief intervention with every parameter frozen.

        Each override must align exactly with that session's chronological test
        trials.  Training beliefs, preprocessing, latent basis, dynamics,
        observation maps, and count targets are never refit.  This API is for
        fixed-checkpoint clamp/delay/shuffle audits and deliberately has no
        route for replacing train-fold beliefs or privileged context labels.
        """

        if not isinstance(heldout_beliefs, Mapping):
            raise TypeError("heldout_beliefs must be a session-to-array mapping")
        return self._score(
            sessions,
            splits,
            heldout_beliefs=heldout_beliefs,
        )
