"""Deterministic belief-controlled Poisson latent dynamics for Exp25.

This is a compact, leakage-auditable comparison model rather than a claim of
exact marginal PLDS inference.  Each session receives a train-only log-count
encoder and a session-specific Poisson observation matrix.  Because those
encoders are independently fit PCA coordinate systems, the current
implementation is scientifically eligible only for a single session.  It
fails closed before fitting a shared operator across multiple unaligned,
non-identifiable session coordinates.  Conditional Poisson scoring uses
held-out next bins; current held-out counts may encode the current latent
state, while held-out target bins are never fit.

The five nested comparison families are:

``common``
    Shared state and input operators.
``input-gated``
    Belief modulates only the input operator.
``state-gated``
    Belief modulates only rank-constrained internal state operators.
``fully-gated``
    Both input and rank-constrained state operators are modulated.
``separate-task``
    Fit-only task labels estimate independent full operators and
    task-specific session observations.  Held-out prediction still mixes them
    using past-only beliefs; the scoring API never reads held-out task truth.

All scaling, latent bases, dynamics, emissions, and null rates are fit from
whole training trials supplied by :class:`TrialFold`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy.special import gammaln


Array = np.ndarray
Family = Literal[
    "common",
    "input-gated",
    "state-gated",
    "fully-gated",
    "separate-task",
]
DEFAULT_LATENT_DIMENSIONS = (2, 4, 8, 16)
_FAMILY_ALIASES: Mapping[str, Family] = {
    "common": "common",
    "input-gated": "input-gated",
    "input_gated": "input-gated",
    "state-gated": "state-gated",
    "state_gated": "state-gated",
    "fully-gated": "fully-gated",
    "fully_gated": "fully-gated",
    "separate-task": "separate-task",
    "separate_task": "separate-task",
}
_FORBIDDEN_BELIEF_SOURCES = frozenset(
    {
        "composition_id",
        "context",
        "context_id",
        "ground_truth",
        "target",
        "target_id",
        "task_id",
        "true_context",
        "true_task",
    }
)
LATENT_COORDINATE_SYSTEM = "independent_session_train_only_pca"
SESSION_BASIS_ALIGNMENT = "not_implemented"
SHARED_CROSS_SESSION_DYNAMICS_IDENTIFIABLE = False


class UnalignedSessionLatentCoordinatesError(ValueError):
    """Raised when independent session PCA coordinates would be pooled."""


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _freeze(value: Array, *, dtype: np.dtype | type | None = None) -> Array:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _label_key(value: object) -> tuple[str, str, object]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        raise ValueError("task/belief labels cannot be missing")
    if isinstance(value, tuple):
        return "builtins", "tuple", tuple(_label_key(item) for item in value)
    if isinstance(value, (float, complex)) and not np.isfinite(value):
        raise ValueError("task/belief labels must be finite")
    try:
        hash(value)
    except TypeError as error:
        raise ValueError("task/belief labels must be hashable") from error
    value_type = type(value)
    return value_type.__module__, value_type.__qualname__, value


def _id_tuple(values: Sequence[object] | Array, *, name: str) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(values)
    try:
        unique = set(result)
    except TypeError as error:
        raise TypeError(f"{name} must contain hashable values") from error
    if not result or len(unique) != len(result):
        raise ValueError(f"{name} must be non-empty and contain no duplicates")
    return result


def _array_sha256(value: Array) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CausalBeliefReceipt:
    """Minimal interoperable receipt for a strictly past-only belief tape."""

    evaluated_trial_keys: tuple[tuple[str, str], ...]
    source_columns: tuple[str, ...]
    feature_lag_trials: int = 1
    uses_current_trial_fields: bool = False
    uses_future_trials: bool = False
    accessed_test_truth: bool = False


def _validate_belief_receipt(
    receipt: object,
    *,
    session_id: str,
    trial_ids: Array,
) -> None:
    if receipt is None:
        raise ValueError("belief_receipt is required; unproven beliefs fail closed")
    for name in (
        "uses_current_trial_fields",
        "uses_future_trials",
        "accessed_test_truth",
    ):
        value = getattr(receipt, name, None)
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"belief receipt is missing boolean {name}")
        if bool(value):
            raise ValueError("belief receipt permits current/future/test-truth leakage")
    lag = getattr(receipt, "feature_lag_trials", None)
    if (
        isinstance(lag, (bool, np.bool_))
        or not isinstance(lag, (int, np.integer))
        or int(lag) < 1
    ):
        raise ValueError("belief receipt must certify a positive trial lag")
    source_columns = getattr(receipt, "source_columns", None)
    if (
        isinstance(source_columns, (str, bytes))
        or not source_columns
        or any(not isinstance(value, str) or not value.strip() for value in source_columns)
    ):
        raise ValueError("belief receipt must list non-empty causal source columns")
    if {value.strip().lower() for value in source_columns} & _FORBIDDEN_BELIEF_SOURCES:
        raise ValueError("belief receipt lists privileged task/context truth sources")
    keys = getattr(receipt, "evaluated_trial_keys", None)
    expected = tuple((session_id, str(trial_id)) for trial_id in trial_ids)
    if keys is None or tuple(keys) != expected:
        raise ValueError("belief receipt does not bind this session trial order")


@dataclass(frozen=True, slots=True)
class BeliefControlledCountSession:
    """Immutable neural-count session and truth-free held-out controller inputs."""

    session_id: str
    animal_id: str
    counts: Array
    inputs: Array
    beliefs: Array
    belief_labels: tuple[object, ...]
    trial_ids: Array
    belief_receipt: object
    task_ids: Array | None = None

    def __post_init__(self) -> None:
        session_id = _text(self.session_id, name="session_id")
        animal_id = _text(self.animal_id, name="animal_id")
        raw_counts = np.asarray(self.counts)
        if (
            raw_counts.ndim != 3
            or raw_counts.shape[1] < 2
            or raw_counts.shape[2] < 1
            or raw_counts.dtype.kind not in {"i", "u"}
            or np.any(raw_counts < 0)
        ):
            raise ValueError(
                "counts must be non-negative integer [trial,time>=2,unit] data"
            )
        counts = np.asarray(raw_counts, dtype=np.int64)
        inputs = np.asarray(self.inputs, dtype=float)
        if (
            inputs.ndim != 3
            or inputs.shape[:2] != counts.shape[:2]
            or inputs.shape[2] < 1
            or not np.isfinite(inputs).all()
        ):
            raise ValueError("inputs must be finite [trial,time,input] data")
        labels = tuple(self.belief_labels)
        if len(labels) < 2 or len({_label_key(value) for value in labels}) != len(labels):
            raise ValueError("belief_labels must contain at least two unique labels")
        beliefs = np.asarray(self.beliefs, dtype=float)
        if beliefs.ndim == 2 and beliefs.shape == (counts.shape[0], len(labels)):
            beliefs = np.repeat(beliefs[:, None, :], counts.shape[1], axis=1)
        if (
            beliefs.shape != (*counts.shape[:2], len(labels))
            or not np.isfinite(beliefs).all()
            or np.any((beliefs <= 0.0) | (beliefs >= 1.0))
            or not np.allclose(beliefs.sum(axis=2), 1.0, atol=1e-9, rtol=0.0)
        ):
            raise ValueError(
                "beliefs must be strictly soft [trial,time,belief] probabilities"
            )
        raw_ids = np.asarray(self.trial_ids, dtype=object)
        if raw_ids.shape != (counts.shape[0],):
            raise ValueError("trial_ids must contain one ID per trial")
        _id_tuple(raw_ids.tolist(), name="trial_ids")
        _validate_belief_receipt(
            self.belief_receipt, session_id=session_id, trial_ids=raw_ids
        )
        task_ids: Array | None = None
        if self.task_ids is not None:
            task_ids = np.asarray(self.task_ids, dtype=object)
            if task_ids.shape != (counts.shape[0],):
                raise ValueError("task_ids must contain one fit-only label per trial")
            for value in task_ids:
                _label_key(value)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "animal_id", animal_id)
        object.__setattr__(self, "belief_labels", labels)
        object.__setattr__(self, "counts", _freeze(counts))
        object.__setattr__(self, "inputs", _freeze(inputs, dtype=float))
        object.__setattr__(self, "beliefs", _freeze(beliefs, dtype=float))
        object.__setattr__(self, "trial_ids", _freeze(raw_ids))
        if task_ids is not None:
            object.__setattr__(self, "task_ids", _freeze(task_ids))

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
    def input_dim(self) -> int:
        return int(self.inputs.shape[2])

    @property
    def belief_dim(self) -> int:
        return int(self.beliefs.shape[2])


@dataclass(frozen=True, slots=True)
class TrialFold:
    """Whole-trial train/test IDs for one session."""

    train_trial_ids: tuple[object, ...]
    test_trial_ids: tuple[object, ...]

    def __post_init__(self) -> None:
        train = _id_tuple(self.train_trial_ids, name="train_trial_ids")
        test = _id_tuple(self.test_trial_ids, name="test_trial_ids")
        if set(train) & set(test):
            raise ValueError("train and test trial IDs overlap")
        object.__setattr__(self, "train_trial_ids", train)
        object.__setattr__(self, "test_trial_ids", test)


def _positions(session: BeliefControlledCountSession, ids: Sequence[object]) -> Array:
    lookup = {value: index for index, value in enumerate(session.trial_ids.tolist())}
    if any(value not in lookup for value in ids):
        raise ValueError(f"split contains unknown trial IDs for {session.session_id}")
    return np.asarray([lookup[value] for value in ids], dtype=int)


@dataclass(frozen=True, slots=True)
class _SessionParameters:
    mean: Array
    scale: Array
    components: Array
    observation: Array
    intercept: Array
    null_rate: Array
    train_trial_ids: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class SessionPoissonPrediction:
    session_id: str
    trial_ids: tuple[object, ...]
    observed: Array
    rates: Array
    latent_current: Array
    latent_prediction: Array

    def __post_init__(self) -> None:
        for name in ("observed", "rates", "latent_current", "latent_prediction"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class SessionPoissonScore:
    session_id: str
    log_likelihood: float
    null_log_likelihood: float
    n_observations: int
    n_spikes: int
    bits_per_spike: float


@dataclass(frozen=True, slots=True)
class PoissonHeldoutScore:
    log_likelihood: float
    null_log_likelihood: float
    nll_per_count: float
    mean_log_likelihood: float
    n_observations: int
    n_spikes: int
    bits_per_spike: float
    parameter_count: int
    per_session: tuple[SessionPoissonScore, ...]
    likelihood_kind: str = "one_step_conditional_poisson_train_only_latent_encoder"
    full_marginal_plds: bool = False
    heldout_truth_used: bool = False


def poisson_log_likelihood(counts: Array, rates: Array) -> float:
    observed = np.asarray(counts)
    predicted = np.asarray(rates, dtype=float)
    if (
        observed.shape != predicted.shape
        or observed.dtype.kind not in {"i", "u"}
        or np.any(observed < 0)
    ):
        raise ValueError("counts and rates must have matching non-negative count shape")
    if not np.isfinite(predicted).all() or np.any(predicted <= 0.0):
        raise ValueError("Poisson rates must be finite and positive")
    return float(
        np.sum(observed * np.log(predicted) - predicted - gammaln(observed + 1.0))
    )


def _ridge(design: Array, target: Array, alpha: float, *, intercept: bool) -> Array:
    gram = design.T @ design
    regularizer = alpha * np.eye(gram.shape[0], dtype=float)
    if intercept:
        regularizer[-1, -1] = 0.0
    return np.linalg.pinv(gram + regularizer) @ design.T @ target


def _truncate_rank(matrix: Array, rank: int) -> Array:
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    retained = min(rank, singular.size)
    return (left[:, :retained] * singular[:retained]) @ right[:retained]


def _poisson_irls(
    latent: Array,
    counts: Array,
    *,
    ridge: float,
    max_iter: int,
) -> tuple[Array, Array]:
    design = np.column_stack([latent, np.ones(latent.shape[0])])
    target = np.asarray(counts, dtype=float)
    initial = _ridge(
        design,
        np.log(target + 0.5),
        max(ridge, 1e-8),
        intercept=True,
    )
    coefficients = initial.copy()
    penalty = ridge * np.eye(design.shape[1], dtype=float)
    penalty[-1, -1] = 0.0
    for _ in range(max_iter):
        eta = np.clip(design @ coefficients, -12.0, 12.0)
        mean = np.exp(eta)
        maximum_step = 0.0
        for unit in range(target.shape[1]):
            weighted_design = design * mean[:, [unit]]
            hessian = design.T @ weighted_design + penalty
            gradient = (
                design.T @ (target[:, unit] - mean[:, unit])
                - penalty @ coefficients[:, unit]
            )
            step = np.linalg.pinv(hessian) @ gradient
            coefficients[:, unit] += step
            maximum_step = max(maximum_step, float(np.max(np.abs(step))))
        if maximum_step < 1e-7:
            break
    return coefficients[:-1], coefficients[-1]


class BeliefControlledPLDS:
    """Nested, deterministic single-session Poisson latent comparison.

    A multi-session call fails closed until a train-only shared latent basis or
    an identifiable session-alignment model is implemented.
    """

    def __init__(
        self,
        family: Family | str,
        latent_dim: int,
        *,
        gate_rank: int = 2,
        ridge: float = 1e-3,
        poisson_ridge: float = 1e-3,
        max_irls: int = 40,
    ) -> None:
        if not isinstance(family, str) or family not in _FAMILY_ALIASES:
            raise ValueError(f"family must be one of {sorted(_FAMILY_ALIASES)}")
        if (
            isinstance(latent_dim, (bool, np.bool_))
            or not isinstance(latent_dim, (int, np.integer))
            or int(latent_dim) < 1
        ):
            raise ValueError("latent_dim must be a positive integer")
        if (
            isinstance(gate_rank, (bool, np.bool_))
            or not isinstance(gate_rank, (int, np.integer))
            or int(gate_rank) < 1
        ):
            raise ValueError("gate_rank must be a positive integer")
        if (
            not np.isfinite(ridge)
            or ridge < 0
            or not np.isfinite(poisson_ridge)
            or poisson_ridge < 0
        ):
            raise ValueError("ridge penalties must be finite and non-negative")
        if (
            isinstance(max_irls, (bool, np.bool_))
            or not isinstance(max_irls, (int, np.integer))
            or int(max_irls) < 1
        ):
            raise ValueError("max_irls must be a positive integer")
        self.family: Family = _FAMILY_ALIASES[family]
        self.latent_dim = int(latent_dim)
        self.gate_rank = min(int(gate_rank), self.latent_dim)
        self.ridge = float(ridge)
        self.poisson_ridge = float(poisson_ridge)
        self.max_irls = int(max_irls)
        self._fitted = False

    def _validate_collection(
        self,
        sessions: Sequence[BeliefControlledCountSession],
        folds: Mapping[str, TrialFold],
        *,
        require_task_ids: bool,
    ) -> tuple[BeliefControlledCountSession, ...]:
        values = tuple(sessions)
        if not values:
            raise ValueError("sessions must not be empty")
        if not all(isinstance(value, BeliefControlledCountSession) for value in values):
            raise TypeError("sessions must contain BeliefControlledCountSession objects")
        ids = tuple(value.session_id for value in values)
        if len(set(ids)) != len(ids) or set(folds) != set(ids):
            raise ValueError("folds must cover each unique session exactly")
        input_dims = {value.input_dim for value in values}
        belief_labels = {tuple(map(_label_key, value.belief_labels)) for value in values}
        if len(input_dims) != 1 or len(belief_labels) != 1:
            raise ValueError(
                "sessions must share input dimension and ordered belief labels"
            )
        for session in values:
            fold = folds[session.session_id]
            if not isinstance(fold, TrialFold):
                raise TypeError("folds must map session IDs to TrialFold")
            covered = set(fold.train_trial_ids) | set(fold.test_trial_ids)
            if not covered <= set(session.trial_ids.tolist()):
                raise ValueError("fold contains trial IDs absent from its session")
            if (
                require_task_ids
                and self.family == "separate-task"
                and session.task_ids is None
            ):
                raise ValueError("separate-task fit requires fit-only task_ids")
        return values

    def _fit_session_encoder(
        self,
        session: BeliefControlledCountSession,
        train: Array,
    ) -> tuple[Array, Array, Array, Array]:
        train_counts = session.counts[train].reshape(-1, session.n_units)
        transformed = np.log1p(train_counts.astype(float))
        mean = transformed.mean(axis=0)
        scale = transformed.std(axis=0, ddof=0)
        scale[scale < 1e-8] = 1.0
        standardized = (transformed - mean) / scale
        _, singular, right = np.linalg.svd(standardized, full_matrices=False)
        tolerance = (
            singular[0]
            * max(standardized.shape)
            * np.finfo(singular.dtype).eps
        )
        maximum_dim = min(session.n_units, int(np.count_nonzero(singular > tolerance)))
        if self.latent_dim > maximum_dim:
            raise ValueError(
                f"latent_dim={self.latent_dim} exceeds train rank "
                f"{maximum_dim} in session {session.session_id}"
            )
        components = right[: self.latent_dim].copy()
        latent = standardized @ components.T
        anchors = np.column_stack(
            [
                session.inputs[train].reshape(-1, session.input_dim),
                session.beliefs[train, :, 1:].reshape(
                    -1, session.belief_dim - 1
                ),
                np.tile(
                    np.linspace(-1.0, 1.0, session.n_time),
                    len(train),
                )[:, None],
            ]
        )
        anchors -= anchors.mean(axis=0, keepdims=True)
        for component in range(self.latent_dim):
            centered = latent[:, component] - latent[:, component].mean()
            covariance = centered @ anchors
            anchor = int(np.argmax(np.abs(covariance)))
            if abs(covariance[anchor]) > 1e-10:
                sign = 1.0 if covariance[anchor] >= 0.0 else -1.0
            else:
                loading = int(np.argmax(np.abs(components[component])))
                sign = 1.0 if components[component, loading] >= 0.0 else -1.0
            components[component] *= sign
            latent[:, component] *= sign
        return mean, scale, components, latent

    @staticmethod
    def _encode(
        counts: Array,
        parameters: _SessionParameters,
    ) -> Array:
        values = np.log1p(np.asarray(counts, dtype=float))
        return ((values - parameters.mean) / parameters.scale) @ parameters.components.T

    def _training_transitions(
        self,
        sessions: Sequence[BeliefControlledCountSession],
        folds: Mapping[str, TrialFold],
        encoders: Mapping[str, tuple[Array, Array, Array, Array]],
    ) -> tuple[Array, Array, Array, Array, Array | None]:
        currents: list[Array] = []
        followings: list[Array] = []
        inputs: list[Array] = []
        beliefs: list[Array] = []
        tasks: list[Array] = []
        label_lookup = {
            _label_key(value): index for index, value in enumerate(self.belief_labels_)
        }
        for session in sessions:
            train = _positions(session, folds[session.session_id].train_trial_ids)
            mean, scale, components, _ = encoders[session.session_id]
            all_latent = (
                (np.log1p(session.counts[train].astype(float)) - mean) / scale
            ) @ components.T
            currents.append(all_latent[:, :-1].reshape(-1, self.latent_dim))
            followings.append(all_latent[:, 1:].reshape(-1, self.latent_dim))
            inputs.append(
                session.inputs[train, :-1].reshape(-1, session.input_dim)
            )
            beliefs.append(
                session.beliefs[train, :-1].reshape(-1, session.belief_dim)
            )
            if self.family == "separate-task":
                assert session.task_ids is not None
                one_hot = np.zeros((len(train), session.belief_dim), dtype=float)
                for row, value in enumerate(session.task_ids[train]):
                    key = _label_key(value)
                    if key not in label_lookup:
                        raise ValueError(
                            "fit-only task label is absent from belief label order"
                        )
                    one_hot[row, label_lookup[key]] = 1.0
                tasks.append(np.repeat(one_hot, session.n_time - 1, axis=0))
        return (
            np.concatenate(currents),
            np.concatenate(followings),
            np.concatenate(inputs),
            np.concatenate(beliefs),
            None if not tasks else np.concatenate(tasks),
        )

    def _dynamics_design(
        self,
        latent: Array,
        inputs: Array,
        beliefs: Array,
        task_one_hot: Array | None,
    ) -> Array:
        contrast = beliefs[:, 1:]
        ones = np.ones((latent.shape[0], 1), dtype=float)
        if self.family == "common":
            return np.column_stack([latent, inputs, ones])
        if self.family == "input-gated":
            input_delta = np.concatenate(
                [inputs * contrast[:, [index]] for index in range(contrast.shape[1])],
                axis=1,
            )
            return np.column_stack([latent, inputs, input_delta, ones])
        if self.family == "state-gated":
            state_delta = np.concatenate(
                [latent * contrast[:, [index]] for index in range(contrast.shape[1])],
                axis=1,
            )
            return np.column_stack([latent, state_delta, inputs, ones])
        if self.family == "fully-gated":
            state_delta = np.concatenate(
                [latent * contrast[:, [index]] for index in range(contrast.shape[1])],
                axis=1,
            )
            input_delta = np.concatenate(
                [inputs * contrast[:, [index]] for index in range(contrast.shape[1])],
                axis=1,
            )
            return np.column_stack(
                [latent, state_delta, inputs, input_delta, ones]
            )
        if task_one_hot is None:
            raise ValueError("separate-task dynamics require fit-only task labels")
        state = np.concatenate(
            [
                latent * task_one_hot[:, [index]]
                for index in range(task_one_hot.shape[1])
            ],
            axis=1,
        )
        routed_input = np.concatenate(
            [
                inputs * task_one_hot[:, [index]]
                for index in range(task_one_hot.shape[1])
            ],
            axis=1,
        )
        return np.column_stack([state, routed_input, task_one_hot])

    def _parse_dynamics(self, coefficients: Array) -> None:
        d = self.latent_dim
        p = self.input_dim_
        k = self.belief_dim_
        position = 0
        if self.family == "separate-task":
            state_rows = k * d
            input_rows = k * p
            self.state_base_ = coefficients[position : position + state_rows].reshape(
                k, d, d
            )
            position += state_rows
            self.input_base_ = coefficients[position : position + input_rows].reshape(
                k, p, d
            )
            position += input_rows
            self.intercept_ = coefficients[position : position + k]
            self.state_delta_ = np.empty((0, d, d))
            self.input_delta_ = np.empty((0, p, d))
            return
        self.state_base_ = coefficients[position : position + d]
        position += d
        if self.family in {"state-gated", "fully-gated"}:
            state_rows = (k - 1) * d
            raw = coefficients[position : position + state_rows].reshape(k - 1, d, d)
            self.state_delta_ = np.stack(
                [_truncate_rank(matrix, self.gate_rank) for matrix in raw]
            )
            position += state_rows
        else:
            self.state_delta_ = np.empty((0, d, d))
        self.input_base_ = coefficients[position : position + p]
        position += p
        if self.family in {"input-gated", "fully-gated"}:
            input_rows = (k - 1) * p
            self.input_delta_ = coefficients[
                position : position + input_rows
            ].reshape(k - 1, p, d)
            position += input_rows
        else:
            self.input_delta_ = np.empty((0, p, d))
        self.intercept_ = coefficients[position]

    def _predict_latent(
        self,
        latent: Array,
        inputs: Array,
        beliefs: Array,
    ) -> Array:
        if self.family == "separate-task":
            state_prediction = np.einsum(
                "nk,nkd->nd",
                beliefs,
                np.einsum("nd,kde->nke", latent, self.state_base_),
            )
            input_prediction = np.einsum(
                "nk,nkd->nd",
                beliefs,
                np.einsum("np,kpd->nkd", inputs, self.input_base_),
            )
            return (
                state_prediction
                + input_prediction
                + beliefs @ self.intercept_
            )
        prediction = latent @ self.state_base_ + inputs @ self.input_base_
        contrast = beliefs[:, 1:]
        for index, weight in enumerate(contrast.T):
            if self.state_delta_.size:
                prediction += (latent @ self.state_delta_[index]) * weight[:, None]
            if self.input_delta_.size:
                prediction += (inputs @ self.input_delta_[index]) * weight[:, None]
        return prediction + self.intercept_

    def fit(
        self,
        sessions: Sequence[BeliefControlledCountSession],
        folds: Mapping[str, TrialFold],
    ) -> "BeliefControlledPLDS":
        """Fit every data-dependent quantity only on each fold's train trials."""

        values = self._validate_collection(sessions, folds, require_task_ids=True)
        if len(values) > 1:
            raise UnalignedSessionLatentCoordinatesError(
                "shared cross-session dynamics are scientifically ineligible: "
                "each session uses an independently fit train-only PCA coordinate "
                "system, but no train-only shared basis or identifiable session "
                "alignment is implemented"
            )
        self._fitted = False
        self.input_dim_ = values[0].input_dim
        self.belief_dim_ = values[0].belief_dim
        self.belief_labels_ = values[0].belief_labels
        encoders: dict[str, tuple[Array, Array, Array, Array]] = {}
        for session in values:
            train = _positions(session, folds[session.session_id].train_trial_ids)
            encoders[session.session_id] = self._fit_session_encoder(session, train)
        current, following, inputs, beliefs, task_one_hot = self._training_transitions(
            values, folds, encoders
        )
        design = self._dynamics_design(
            current, inputs, beliefs, task_one_hot
        )
        coefficients = _ridge(
            design, following, self.ridge, intercept=self.family != "separate-task"
        )
        self._parse_dynamics(coefficients)

        parameters: dict[str, _SessionParameters] = {}
        label_lookup = {
            _label_key(value): index for index, value in enumerate(self.belief_labels_)
        }
        for session in values:
            train_ids = folds[session.session_id].train_trial_ids
            train = _positions(session, train_ids)
            mean, scale, components, latent = encoders[session.session_id]
            flat_counts = session.counts[train].reshape(-1, session.n_units)
            if self.family == "separate-task":
                assert session.task_ids is not None
                observations: list[Array] = []
                intercepts: list[Array] = []
                for label in self.belief_labels_:
                    selected_trials = np.asarray(
                        [
                            _label_key(value) == _label_key(label)
                            for value in session.task_ids[train]
                        ],
                        dtype=bool,
                    )
                    selected = np.repeat(selected_trials, session.n_time)
                    if np.count_nonzero(selected) <= self.latent_dim:
                        raise ValueError(
                            f"session {session.session_id} lacks train samples "
                            f"for separate task {label!r}"
                        )
                    observation, intercept = _poisson_irls(
                        latent[selected],
                        flat_counts[selected],
                        ridge=self.poisson_ridge,
                        max_iter=self.max_irls,
                    )
                    observations.append(observation)
                    intercepts.append(intercept)
                observation_value = np.stack(observations)
                intercept_value = np.stack(intercepts)
            else:
                observation_value, intercept_value = _poisson_irls(
                    latent,
                    flat_counts,
                    ridge=self.poisson_ridge,
                    max_iter=self.max_irls,
                )
            # The explicit lookup audit prevents a silent disagreement between
            # fit labels and the controller's ordered probability coordinates.
            if self.family == "separate-task" and session.task_ids is not None:
                for value in session.task_ids[train]:
                    if _label_key(value) not in label_lookup:
                        raise ValueError(
                            "fit-only task label is absent from belief label order"
                        )
            parameters[session.session_id] = _SessionParameters(
                mean=_freeze(mean),
                scale=_freeze(scale),
                components=_freeze(components),
                observation=_freeze(observation_value),
                intercept=_freeze(intercept_value),
                null_rate=_freeze(
                    np.maximum(flat_counts.mean(axis=0), 1e-8), dtype=float
                ),
                train_trial_ids=tuple(train_ids),
            )
        self.session_parameters_ = parameters
        self.session_unit_counts_ = {
            session.session_id: session.n_units for session in values
        }
        self.fit_trial_ids_ = {
            session_id: tuple(fold.train_trial_ids)
            for session_id, fold in folds.items()
        }
        self._fitted = True
        self.fit_fingerprint_ = self._fingerprint()
        return self

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.family.encode("ascii"))
        digest.update(str(self.latent_dim).encode("ascii"))
        for array in (
            self.state_base_,
            self.state_delta_,
            self.input_base_,
            self.input_delta_,
            self.intercept_,
        ):
            digest.update(_array_sha256(np.asarray(array)).encode("ascii"))
        for session_id in sorted(self.session_parameters_):
            digest.update(session_id.encode("utf-8"))
            parameters = self.session_parameters_[session_id]
            for array in (
                parameters.mean,
                parameters.scale,
                parameters.components,
                parameters.observation,
                parameters.intercept,
                parameters.null_rate,
            ):
                digest.update(_array_sha256(array).encode("ascii"))
        return digest.hexdigest()

    def _validate_scoring(
        self,
        sessions: Sequence[BeliefControlledCountSession],
        folds: Mapping[str, TrialFold],
    ) -> tuple[BeliefControlledCountSession, ...]:
        if not self._fitted:
            raise RuntimeError("model must be fit before prediction")
        values = self._validate_collection(sessions, folds, require_task_ids=False)
        if set(self.session_parameters_) != {value.session_id for value in values}:
            raise ValueError(
                "session-specific observation matrices cannot score unseen sessions"
            )
        for session in values:
            if session.n_units != self.session_unit_counts_[session.session_id]:
                raise ValueError("held-out session unit count differs from fit")
            if (
                session.input_dim != self.input_dim_
                or session.belief_labels != self.belief_labels_
            ):
                raise ValueError("held-out input/belief schema differs from fit")
        return values

    def predict(
        self,
        sessions: Sequence[BeliefControlledCountSession],
        folds: Mapping[str, TrialFold],
    ) -> dict[str, SessionPoissonPrediction]:
        """Predict next-bin rates without reading held-out task labels."""

        values = self._validate_scoring(sessions, folds)
        predictions: dict[str, SessionPoissonPrediction] = {}
        for session in values:
            parameters = self.session_parameters_[session.session_id]
            test_ids = folds[session.session_id].test_trial_ids
            test = _positions(session, test_ids)
            current_counts = session.counts[test, :-1].reshape(-1, session.n_units)
            observed = session.counts[test, 1:].reshape(-1, session.n_units)
            latent = self._encode(current_counts, parameters)
            inputs = session.inputs[test, :-1].reshape(-1, self.input_dim_)
            beliefs = session.beliefs[test, :-1].reshape(-1, self.belief_dim_)
            latent_prediction = self._predict_latent(latent, inputs, beliefs)
            if self.family == "separate-task":
                logits = np.einsum(
                    "nd,kdu->nku", latent_prediction, parameters.observation
                ) + parameters.intercept[None, :, :]
                log_rate = np.einsum("nk,nku->nu", beliefs, logits)
            else:
                log_rate = (
                    latent_prediction @ parameters.observation
                    + parameters.intercept
                )
            rates = np.exp(np.clip(log_rate, -12.0, 12.0))
            predictions[session.session_id] = SessionPoissonPrediction(
                session_id=session.session_id,
                trial_ids=tuple(test_ids),
                observed=observed,
                rates=rates,
                latent_current=latent,
                latent_prediction=latent_prediction,
            )
        return predictions

    def score(
        self,
        sessions: Sequence[BeliefControlledCountSession],
        folds: Mapping[str, TrialFold],
    ) -> PoissonHeldoutScore:
        """Return exact held-out conditional Poisson log likelihood."""

        predictions = self.predict(sessions, folds)
        per_session: list[SessionPoissonScore] = []
        total_ll = 0.0
        total_null = 0.0
        total_observations = 0
        total_spikes = 0
        for session_id in sorted(predictions):
            prediction = predictions[session_id]
            parameters = self.session_parameters_[session_id]
            null_rates = np.broadcast_to(
                parameters.null_rate, prediction.observed.shape
            )
            log_likelihood = poisson_log_likelihood(
                prediction.observed, prediction.rates
            )
            null_log_likelihood = poisson_log_likelihood(
                prediction.observed, null_rates
            )
            n_spikes = int(prediction.observed.sum())
            gain = log_likelihood - null_log_likelihood
            bits = (
                gain / (n_spikes * np.log(2.0))
                if n_spikes > 0
                else float("nan")
            )
            n_observations = int(prediction.observed.size)
            per_session.append(
                SessionPoissonScore(
                    session_id=session_id,
                    log_likelihood=log_likelihood,
                    null_log_likelihood=null_log_likelihood,
                    n_observations=n_observations,
                    n_spikes=n_spikes,
                    bits_per_spike=float(bits),
                )
            )
            total_ll += log_likelihood
            total_null += null_log_likelihood
            total_observations += n_observations
            total_spikes += n_spikes
        total_bits = (
            (total_ll - total_null) / (total_spikes * np.log(2.0))
            if total_spikes > 0
            else float("nan")
        )
        return PoissonHeldoutScore(
            log_likelihood=float(total_ll),
            null_log_likelihood=float(total_null),
            nll_per_count=float(-total_ll / total_observations),
            mean_log_likelihood=float(total_ll / total_observations),
            n_observations=total_observations,
            n_spikes=total_spikes,
            bits_per_spike=float(total_bits),
            parameter_count=self.parameter_count(),
            per_session=tuple(per_session),
        )

    def parameter_breakdown(self) -> dict[str, int]:
        if not self._fitted:
            raise RuntimeError("model must be fit before parameter counting")
        d, p, k = self.latent_dim, self.input_dim_, self.belief_dim_
        common = d * d + p * d + d
        rank_delta = self.gate_rank * (2 * d - self.gate_rank)
        if self.family == "common":
            dynamics = common
        elif self.family == "input-gated":
            dynamics = common + (k - 1) * p * d
        elif self.family == "state-gated":
            dynamics = common + (k - 1) * rank_delta
        elif self.family == "fully-gated":
            dynamics = common + (k - 1) * (rank_delta + p * d)
        else:
            dynamics = k * (d * d + p * d + d)
        preprocessing = 0
        observation = 0
        for session_id, parameters in self.session_parameters_.items():
            units = self.session_unit_counts_[session_id]
            preprocessing += 2 * units + units * d
            multiplier = k if self.family == "separate-task" else 1
            observation += multiplier * units * (d + 1)
            if parameters.components.shape != (d, units):
                raise RuntimeError("stored preprocessing shape is inconsistent")
        return {
            "dynamics": int(dynamics),
            "session_preprocessing": int(preprocessing),
            "session_observation": int(observation),
        }

    def parameter_count(self) -> int:
        return int(sum(self.parameter_breakdown().values()))

    def effective_state_operators(self) -> dict[object, Array]:
        """Return one fitted operator per belief coordinate for auditing."""

        if not self._fitted:
            raise RuntimeError("model must be fit before operator inspection")
        if self.family == "separate-task":
            return {
                label: self.state_base_[index].copy()
                for index, label in enumerate(self.belief_labels_)
            }
        result = {self.belief_labels_[0]: np.asarray(self.state_base_).copy()}
        for index, label in enumerate(self.belief_labels_[1:]):
            matrix = np.asarray(self.state_base_).copy()
            if self.state_delta_.size:
                matrix += self.state_delta_[index]
            result[label] = matrix
        return result


@dataclass(frozen=True, slots=True)
class DimensionCandidateResult:
    latent_dim: int
    fold_mean_log_likelihoods: tuple[float | None, ...]
    fold_errors: tuple[str | None, ...]
    mean_log_likelihood: float | None
    eligible: bool


@dataclass(frozen=True, slots=True)
class DimensionSelectionResult:
    selected_dimension: int | None
    candidates: tuple[DimensionCandidateResult, ...]
    candidate_dimensions: tuple[int, ...]
    selection_scope: str = "nested_group_cv_training_scope_only"


def select_latent_dimension(
    family: Family | str,
    sessions: Sequence[BeliefControlledCountSession],
    inner_folds: Sequence[Mapping[str, TrialFold]],
    *,
    candidate_dimensions: Sequence[int] = DEFAULT_LATENT_DIMENSIONS,
    allowed_trial_ids: Mapping[str, Sequence[object]] | None = None,
    gate_rank: int = 2,
    ridge: float = 1e-3,
    poisson_ridge: float = 1e-3,
    max_irls: int = 40,
) -> DimensionSelectionResult:
    """Nested-CV hook that retains every failed/ineligible dimension.

    ``allowed_trial_ids`` should be the outer-fold training IDs.  When supplied,
    every inner train/validation ID is checked against that scope before any
    model is fit.
    """

    dimensions = tuple(int(value) for value in candidate_dimensions)
    if (
        not dimensions
        or len(set(dimensions)) != len(dimensions)
        or any(value not in DEFAULT_LATENT_DIMENSIONS for value in dimensions)
    ):
        raise ValueError(
            f"candidate dimensions must be unique values from "
            f"{DEFAULT_LATENT_DIMENSIONS}"
        )
    folds = tuple(inner_folds)
    if not folds:
        raise ValueError("inner_folds must not be empty")
    session_ids = {session.session_id for session in sessions}
    if allowed_trial_ids is not None and set(allowed_trial_ids) != session_ids:
        raise ValueError("allowed_trial_ids must cover every session exactly")
    for fold in folds:
        if set(fold) != session_ids:
            raise ValueError("each inner fold must cover every session exactly")
        if allowed_trial_ids is not None:
            for session_id, trial_fold in fold.items():
                allowed = set(allowed_trial_ids[session_id])
                used = set(trial_fold.train_trial_ids) | set(
                    trial_fold.test_trial_ids
                )
                if not used <= allowed:
                    raise ValueError(
                        "inner fold accesses trial IDs outside outer training scope"
                    )

    candidates: list[DimensionCandidateResult] = []
    for latent_dim in dimensions:
        scores: list[float | None] = []
        errors: list[str | None] = []
        for fold in folds:
            try:
                model = BeliefControlledPLDS(
                    family,
                    latent_dim,
                    gate_rank=gate_rank,
                    ridge=ridge,
                    poisson_ridge=poisson_ridge,
                    max_irls=max_irls,
                ).fit(sessions, fold)
                score = model.score(sessions, fold)
            except (ValueError, TypeError, np.linalg.LinAlgError) as error:
                scores.append(None)
                errors.append(f"{type(error).__name__}: {error}")
            else:
                scores.append(score.mean_log_likelihood)
                errors.append(None)
        eligible = all(value is not None for value in scores)
        mean_score = (
            float(np.mean([float(value) for value in scores]))
            if eligible
            else None
        )
        candidates.append(
            DimensionCandidateResult(
                latent_dim=latent_dim,
                fold_mean_log_likelihoods=tuple(scores),
                fold_errors=tuple(errors),
                mean_log_likelihood=mean_score,
                eligible=eligible,
            )
        )
    eligible_candidates = [
        candidate for candidate in candidates if candidate.mean_log_likelihood is not None
    ]
    selected = (
        min(
            eligible_candidates,
            key=lambda candidate: (
                -float(candidate.mean_log_likelihood),
                candidate.latent_dim,
            ),
        ).latent_dim
        if eligible_candidates
        else None
    )
    return DimensionSelectionResult(
        selected_dimension=selected,
        candidates=tuple(candidates),
        candidate_dimensions=dimensions,
    )


__all__ = [
    "DEFAULT_LATENT_DIMENSIONS",
    "LATENT_COORDINATE_SYSTEM",
    "SESSION_BASIS_ALIGNMENT",
    "SHARED_CROSS_SESSION_DYNAMICS_IDENTIFIABLE",
    "BeliefControlledCountSession",
    "BeliefControlledPLDS",
    "CausalBeliefReceipt",
    "DimensionCandidateResult",
    "DimensionSelectionResult",
    "PoissonHeldoutScore",
    "SessionPoissonPrediction",
    "SessionPoissonScore",
    "TrialFold",
    "UnalignedSessionLatentCoordinatesError",
    "poisson_log_likelihood",
    "select_latent_dimension",
]
