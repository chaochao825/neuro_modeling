"""Leakage-safe common, shared-basis, and full switching linear dynamics.

These are observed-state conditional linear-Gaussian models. They deliberately
do not call a PCA+regression pipeline a latent Kalman likelihood. All reported
likelihoods are one-step conditional Gaussian likelihoods on held-out complete
trial/block transitions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from sklearn.decomposition import PCA
from scipy.special import logsumexp

from src.utils.splits import grouped_train_test_split


Array = np.ndarray
Family = Literal["common", "shared", "full"]


def _as_2d_float(name: str, value: Array, rows: int | None = None) -> Array:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} row count does not match states")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_label_vector(name: str, value: Iterable[object], rows: int) -> Array:
    """Preserve tuple-valued labels instead of expanding them into 2-D rows."""

    if isinstance(value, np.ndarray):
        array = value.copy()
    else:
        items = list(value)
        array = np.empty(len(items), dtype=object)
        array[:] = items
    if array.ndim != 1 or array.shape[0] != rows:
        raise ValueError(f"{name} must be a vector matching transitions")
    return array


@dataclass(frozen=True)
class TransitionDataset:
    """One-step transitions with provenance for complete trial/block splits."""

    current: Array
    following: Array
    conditions: Array
    groups: Array
    controls: Array | None = None
    trial_ids: Array | None = None
    time_indices: Array | None = None

    def __post_init__(self) -> None:
        current = _as_2d_float("current", self.current).copy()
        following = _as_2d_float("following", self.following, current.shape[0]).copy()
        if current.shape != following.shape:
            raise ValueError("current and following must have identical shape")
        if current.shape[0] < 2 or current.shape[1] < 1:
            raise ValueError("at least two transitions and one feature are required")
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "following", following)
        for name in ("conditions", "groups"):
            array = _as_label_vector(name, getattr(self, name), current.shape[0])
            for value in array:
                _condition_key(value)
            object.__setattr__(self, name, array)
        if self.controls is not None:
            object.__setattr__(
                self,
                "controls",
                _as_2d_float("controls", self.controls, current.shape[0]).copy(),
            )
        for name in ("trial_ids", "time_indices"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value).copy()
            if array.ndim != 1 or array.shape[0] != current.shape[0]:
                raise ValueError(f"{name} must match the number of transitions")
            if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
                array.dtype, np.integer
            ) or np.any(array < 0):
                raise ValueError(f"{name} must contain non-negative integers")
            object.__setattr__(self, name, array)
        # Frozen dataclasses do not freeze ndarray buffers by themselves.
        for name in (
            "current",
            "following",
            "conditions",
            "groups",
            "controls",
            "trial_ids",
            "time_indices",
        ):
            value = getattr(self, name)
            if value is not None:
                value.setflags(write=False)

    @property
    def n_samples(self) -> int:
        return int(self.current.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.current.shape[1])

    def subset(self, indices: Sequence[int] | Array) -> "TransitionDataset":
        raw = np.asarray(indices)
        if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
            raw.dtype, np.integer
        ):
            raise TypeError("indices must contain integers")
        idx = raw.astype(int, copy=False)
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= self.n_samples):
            raise ValueError("indices are out of range")
        if np.unique(idx).size != idx.size:
            raise ValueError("indices must not contain duplicates")
        return TransitionDataset(
            current=self.current[idx],
            following=self.following[idx],
            conditions=self.conditions[idx],
            groups=self.groups[idx],
            controls=None if self.controls is None else self.controls[idx],
            trial_ids=None if self.trial_ids is None else self.trial_ids[idx],
            time_indices=None if self.time_indices is None else self.time_indices[idx],
        )

    def train_test_split(
        self,
        *,
        test_fraction: float = 0.2,
        seed: int,
        require_condition_coverage: bool = True,
    ) -> tuple["TransitionDataset", "TransitionDataset"]:
        if not isinstance(require_condition_coverage, (bool, np.bool_)):
            raise TypeError("require_condition_coverage must be boolean")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        all_conditions = {_condition_key(value) for value in self.conditions}
        for attempt in range(100):
            train, test = grouped_train_test_split(
                self.groups, test_fraction=test_fraction, seed=int(seed) + attempt
            )
            if not require_condition_coverage:
                return self.subset(train), self.subset(test)
            train_conditions = {_condition_key(value) for value in self.conditions[train]}
            test_conditions = {_condition_key(value) for value in self.conditions[test]}
            if train_conditions == all_conditions and test_conditions == all_conditions:
                return self.subset(train), self.subset(test)
        raise ValueError(
            "could not split whole groups while retaining every condition in train and test"
        )

    @classmethod
    def from_trials(
        cls,
        activity: Array,
        *,
        conditions: Array,
        groups: Iterable[object],
        controls: Array | None = None,
        valid_mask: Array | None = None,
    ) -> "TransitionDataset":
        """Build transitions without ever crossing a trial boundary."""

        values = np.asarray(activity, dtype=float)
        if values.ndim != 3:
            raise ValueError("activity must have shape [trial, time, feature]")
        if not np.isfinite(values).all():
            raise ValueError("activity contains non-finite values")
        n_trials, n_time, _ = values.shape
        if n_time < 2:
            raise ValueError("trials need at least two time bins")
        group_array = _as_label_vector("groups", groups, n_trials)

        if isinstance(conditions, np.ndarray):
            condition_array = conditions.copy()
        else:
            condition_array = _as_label_vector("conditions", conditions, n_trials)
        if condition_array.shape == (n_trials,):
            condition_array = np.repeat(condition_array[:, None], n_time, axis=1)
        if condition_array.shape != (n_trials, n_time):
            raise ValueError("conditions must be trial-level or [trial, time]")

        if valid_mask is None:
            mask = np.ones((n_trials, n_time), dtype=bool)
        else:
            raw_mask = np.asarray(valid_mask)
            if raw_mask.shape != (n_trials, n_time):
                raise ValueError("valid_mask must have shape [trial, time]")
            if raw_mask.dtype != bool and not np.isin(raw_mask, [0, 1]).all():
                raise ValueError("valid_mask must contain only boolean/0/1 values")
            mask = raw_mask.astype(bool, copy=False)

        control_array: Array | None
        if controls is None:
            control_array = None
        else:
            raw_controls = np.asarray(controls, dtype=float)
            if raw_controls.ndim == 2 and raw_controls.shape[0] == n_trials:
                raw_controls = np.repeat(raw_controls[:, None, :], n_time, axis=1)
            if raw_controls.ndim != 3 or raw_controls.shape[:2] != (n_trials, n_time):
                raise ValueError("controls must be trial-level or [trial, time, control]")
            control_array = raw_controls

        currents: list[Array] = []
        followings: list[Array] = []
        condition_rows: list[object] = []
        group_rows: list[object] = []
        control_rows: list[Array] = []
        trial_rows: list[int] = []
        time_rows: list[int] = []
        for trial in range(n_trials):
            for time in range(n_time - 1):
                if not (mask[trial, time] and mask[trial, time + 1]):
                    continue
                currents.append(values[trial, time])
                followings.append(values[trial, time + 1])
                condition_rows.append(condition_array[trial, time])
                group_rows.append(group_array[trial])
                if control_array is not None:
                    control_rows.append(control_array[trial, time])
                trial_rows.append(trial)
                time_rows.append(time)
        if len(currents) < 2:
            raise ValueError("valid_mask leaves fewer than two transitions")
        return cls(
            current=np.stack(currents),
            following=np.stack(followings),
            conditions=_as_label_vector("conditions", condition_rows, len(condition_rows)),
            groups=_as_label_vector("groups", group_rows, len(group_rows)),
            controls=None if control_array is None else np.stack(control_rows),
            trial_ids=np.asarray(trial_rows, dtype=int),
            time_indices=np.asarray(time_rows, dtype=int),
        )


@dataclass(frozen=True)
class DynamicsScore:
    log_likelihood: float
    mean_log_likelihood: float
    nll_per_scalar: float
    mse: float
    r2: float
    n_transitions: int


def _ridge_coefficients(design: Array, target: Array, alpha: float) -> Array:
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("ridge alpha must be non-negative and finite")
    gram = design.T @ design
    regularizer = alpha * np.eye(gram.shape[0], dtype=float)
    try:
        return np.linalg.solve(gram + regularizer, design.T @ target)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram + regularizer) @ design.T @ target


def _stabilize_covariance(matrix: Array, floor: float = 1e-10) -> Array:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(symmetric)
    clipped = np.clip(values, floor, 1e8)
    return (vectors * clipped) @ vectors.T


def _condition_key(value: object) -> tuple[str, str, object]:
    """Keep type information so integer 1 never aliases string '1'."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        raise ValueError("condition/group labels cannot be missing")
    if isinstance(value, tuple):
        encoded_tuple = tuple(_condition_key(item) for item in value)
        return "builtins", "tuple", encoded_tuple
    if isinstance(value, (float, complex)) and not np.isfinite(value):
        raise ValueError("condition/group labels must be finite")
    try:
        if not bool(value == value):
            raise ValueError("condition/group labels cannot be missing")
    except (TypeError, ValueError):
        raise ValueError("condition/group labels must be scalar and non-missing") from None
    try:
        hash(value)
    except TypeError as error:
        raise ValueError("conditions must contain hashable scalar labels") from error
    value_type = type(value)
    return value_type.__module__, value_type.__qualname__, value


class SwitchingLinearDynamics:
    """Nested common/shared/full observed-state dynamics estimator."""

    def __init__(
        self,
        family: Family,
        *,
        latent_dim: int | None = None,
        ridge: float = 1e-4,
        variance_floor: float = 1e-6,
    ) -> None:
        if not isinstance(family, str) or family not in {"common", "shared", "full"}:
            raise ValueError("family must be common, shared, or full")
        if family == "shared":
            if (
                isinstance(latent_dim, (bool, np.bool_))
                or not isinstance(latent_dim, (int, np.integer))
                or latent_dim < 1
            ):
                raise ValueError("shared dynamics require an integer latent_dim >= 1")
            latent_dim = int(latent_dim)
        elif latent_dim is not None:
            raise ValueError("latent_dim is only valid for shared dynamics")
        if (
            isinstance(ridge, (bool, np.bool_))
            or not np.isfinite(ridge)
            or ridge < 0
            or isinstance(variance_floor, (bool, np.bool_))
            or not np.isfinite(variance_floor)
            or variance_floor <= 0
        ):
            raise ValueError(
                "ridge must be finite and non-negative; variance_floor must be finite and positive"
            )
        self.family = family
        self.latent_dim = latent_dim
        self.ridge = float(ridge)
        self.variance_floor = float(variance_floor)
        self._fitted = False

    def _encode_conditions(self, conditions: Array, *, allow_unseen: bool = False) -> Array:
        indices = np.empty(len(conditions), dtype=int)
        for row, condition in enumerate(conditions):
            key = _condition_key(condition)
            if key not in self.condition_to_index_:
                if allow_unseen:
                    indices[row] = -1
                    continue
                raise ValueError(f"unseen condition {condition!r}")
            indices[row] = self.condition_to_index_[key]
        return indices

    def _scale_x(self, x: Array) -> Array:
        return (x - self.mean_) / self.scale_

    def _unscale_y(self, y: Array) -> Array:
        return y * self.scale_ + self.mean_

    def fit(self, train: TransitionDataset) -> "SwitchingLinearDynamics":
        """Fit every data-dependent object exclusively on the supplied train set."""

        if not isinstance(train, TransitionDataset):
            raise TypeError("train must be a TransitionDataset")
        self._fitted = False
        x = _as_2d_float("current", train.current)
        y = _as_2d_float("following", train.following, x.shape[0])
        if self.family == "shared" and int(self.latent_dim or 0) > x.shape[1]:
            raise ValueError("latent_dim cannot exceed the observed feature count")
        self.n_features_ = x.shape[1]
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0, ddof=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        xs = self._scale_x(x)
        ys = self._scale_x(y)

        condition_labels: list[object] = []
        self.condition_to_index_: dict[tuple[str, str, object], int] = {}
        for condition in train.conditions:
            key = _condition_key(condition)
            if key not in self.condition_to_index_:
                self.condition_to_index_[key] = len(condition_labels)
                condition_labels.append(condition.item() if isinstance(condition, np.generic) else condition)
        self.conditions_ = tuple(condition_labels)
        if len(set(self.conditions_)) != len(self.conditions_):
            raise ValueError(
                "condition labels collide as Python mapping keys; use one consistent label type"
            )
        condition_idx = self._encode_conditions(train.conditions)
        k_conditions = len(condition_labels)
        one_hot = np.eye(k_conditions, dtype=float)[condition_idx]

        if train.controls is None:
            controls = np.empty((train.n_samples, 0), dtype=float)
            self.control_mean_ = np.empty(0)
            self.control_scale_ = np.empty(0)
        else:
            controls = _as_2d_float("controls", train.controls, train.n_samples)
            self.control_mean_ = controls.mean(axis=0)
            self.control_scale_ = controls.std(axis=0, ddof=0)
            self.control_scale_[self.control_scale_ < 1e-12] = 1.0
            controls = (controls - self.control_mean_) / self.control_scale_
        self.n_controls_ = controls.shape[1]

        # Nuisance inputs and condition intercepts are identical across families.
        # They are fitted jointly with the dynamic terms: sequentially removing
        # nuisance from Y alone is biased whenever current state and controls
        # are correlated.
        nuisance_design = np.column_stack([controls, one_hot])

        if self.family == "common":
            joint = np.column_stack([xs, nuisance_design])
            coefficients = _ridge_coefficients(joint, ys, self.ridge)
            self.dynamic_coef_ = coefficients[: self.n_features_]
            self.nuisance_coef_ = coefficients[self.n_features_ :]
        elif self.family == "full":
            block_design = np.concatenate(
                [xs * one_hot[:, [index]] for index in range(k_conditions)], axis=1
            )
            joint = np.column_stack([block_design, nuisance_design])
            coefficients = _ridge_coefficients(joint, ys, self.ridge)
            dynamic_rows = k_conditions * self.n_features_
            self.dynamic_coef_ = coefficients[:dynamic_rows].reshape(
                k_conditions, self.n_features_, self.n_features_
            )
            self.nuisance_coef_ = coefficients[dynamic_rows:]
        else:
            self.pca_ = PCA(n_components=int(self.latent_dim), svd_solver="full")
            self.pca_.fit(xs)
            self.basis_ = self.pca_.components_.T
            z_current = xs @ self.basis_
            block_design = np.concatenate(
                [z_current * one_hot[:, [index]] for index in range(k_conditions)], axis=1
            )
            # In-basis outputs contain both switching dynamics and nuisance;
            # solve those terms jointly. Orthogonal outputs can only be
            # nuisance-driven. The decomposition is exact for an orthonormal
            # basis and preserves the same ridge penalty.
            latent_joint = np.column_stack([block_design, nuisance_design])
            latent_coefficients = _ridge_coefficients(
                latent_joint, ys @ self.basis_, self.ridge
            )
            dynamic_rows = k_conditions * int(self.latent_dim)
            self.dynamic_coef_ = latent_coefficients[:dynamic_rows].reshape(
                k_conditions, int(self.latent_dim), int(self.latent_dim)
            )
            nuisance_parallel = latent_coefficients[dynamic_rows:] @ self.basis_.T
            target_orthogonal = ys - (ys @ self.basis_) @ self.basis_.T
            nuisance_orthogonal = _ridge_coefficients(
                nuisance_design, target_orthogonal, self.ridge
            )
            self.nuisance_coef_ = nuisance_parallel + nuisance_orthogonal

        prediction_standardized = self._predict_standardized(train)
        residual = ys - prediction_standardized
        self.noise_variance_ = np.maximum(
            np.mean(residual**2, axis=0), self.variance_floor
        )
        fingerprint = "\x1f".join(
            repr(_condition_key(group)) for group in np.asarray(train.groups)
        ).encode("utf-8")
        self.train_group_fingerprint_ = hashlib.sha256(fingerprint).hexdigest()
        self._fitted = True
        return self

    def _predict_standardized(self, data: TransitionDataset) -> Array:
        xs = self._scale_x(_as_2d_float("current", data.current))
        condition_idx = self._encode_conditions(data.conditions)
        one_hot = np.eye(len(self.conditions_), dtype=float)[condition_idx]
        if data.controls is None:
            if self.n_controls_:
                raise ValueError("model was fit with controls but prediction data has none")
            controls = np.empty((data.n_samples, 0), dtype=float)
        else:
            controls = _as_2d_float("controls", data.controls, data.n_samples)
            if controls.shape[1] != self.n_controls_:
                raise ValueError("control feature count differs from training")
            controls = (controls - self.control_mean_) / self.control_scale_
        nuisance = np.column_stack([controls, one_hot]) @ self.nuisance_coef_

        if self.family == "common":
            dynamic = xs @ self.dynamic_coef_
        elif self.family == "full":
            dynamic = np.empty_like(xs)
            for index in range(len(self.conditions_)):
                rows = condition_idx == index
                dynamic[rows] = xs[rows] @ self.dynamic_coef_[index]
        else:
            z = xs @ self.basis_
            latent_prediction = np.empty_like(z)
            for index in range(len(self.conditions_)):
                rows = condition_idx == index
                latent_prediction[rows] = z[rows] @ self.dynamic_coef_[index]
            dynamic = latent_prediction @ self.basis_.T
        return nuisance + dynamic

    def predict(self, data: TransitionDataset) -> Array:
        if not self._fitted:
            raise RuntimeError("model must be fit before prediction")
        if data.n_features != self.n_features_:
            raise ValueError("feature count differs from training")
        return self._unscale_y(self._predict_standardized(data))

    def score(self, data: TransitionDataset) -> DynamicsScore:
        if not self._fitted:
            raise RuntimeError("model must be fit before scoring")
        if data.n_features != self.n_features_:
            raise ValueError("feature count differs from training")
        observed_standardized = self._scale_x(data.following)
        predicted_standardized = self._predict_standardized(data)
        residual = observed_standardized - predicted_standardized
        log_terms = np.log(2.0 * np.pi * self.noise_variance_)[None, :] + (
            residual**2 / self.noise_variance_[None, :]
        )
        standardized_log_likelihood = float(-0.5 * np.sum(log_terms))
        # Density transformation back to original activity coordinates.
        log_likelihood = standardized_log_likelihood - data.n_samples * float(
            np.sum(np.log(self.scale_))
        )
        prediction = self._unscale_y(predicted_standardized)
        residual_original = data.following - prediction
        mse = float(np.mean(residual_original**2))
        denominator = float(np.sum((data.following - data.following.mean(axis=0)) ** 2))
        r2 = 1.0 - float(np.sum(residual_original**2)) / denominator if denominator > 0 else 0.0
        scalar_count = data.n_samples * data.n_features
        return DynamicsScore(
            log_likelihood=log_likelihood,
            mean_log_likelihood=log_likelihood / data.n_samples,
            nll_per_scalar=-log_likelihood / scalar_count,
            mse=mse,
            r2=r2,
            n_transitions=data.n_samples,
        )

    def transition_matrices(self) -> dict[object, Array]:
        """Return matrices in standardized row-vector convention."""

        if not self._fitted:
            raise RuntimeError("model must be fit first")
        if self.family == "common":
            return {label: self.dynamic_coef_.copy() for label in self.conditions_}
        if self.family == "full":
            return {
                label: self.dynamic_coef_[index].copy()
                for index, label in enumerate(self.conditions_)
            }
        return {
            label: self.basis_ @ self.dynamic_coef_[index] @ self.basis_.T
            for index, label in enumerate(self.conditions_)
        }

    def parameter_count(self) -> int:
        if not self._fitted:
            raise RuntimeError("model must be fit first")
        n = self.n_features_
        k = len(self.conditions_)
        nuisance = self.n_controls_ * n + k * n
        noise = n
        if self.family == "common":
            dynamics = n * n
        elif self.family == "full":
            dynamics = k * n * n
        else:
            d = int(self.latent_dim)
            basis_degrees = n * d - d * (d + 1) // 2
            dynamics = basis_degrees + k * d * d
        return int(nuisance + noise + dynamics)


class CommonDynamics(SwitchingLinearDynamics):
    def __init__(self, *, ridge: float = 1e-4, variance_floor: float = 1e-6) -> None:
        super().__init__("common", ridge=ridge, variance_floor=variance_floor)


class SharedBasisSwitchingDynamics(SwitchingLinearDynamics):
    def __init__(
        self, latent_dim: int, *, ridge: float = 1e-4, variance_floor: float = 1e-6
    ) -> None:
        super().__init__(
            "shared", latent_dim=latent_dim, ridge=ridge, variance_floor=variance_floor
        )


class FullSwitchingDynamics(SwitchingLinearDynamics):
    def __init__(self, *, ridge: float = 1e-4, variance_floor: float = 1e-6) -> None:
        super().__init__("full", ridge=ridge, variance_floor=variance_floor)


def retained_switching_gain(
    common_nll: float, shared_nll: float, full_nll: float
) -> float:
    """Fraction of the full-vs-common NLL improvement retained by shared basis."""

    values = np.asarray([common_nll, shared_nll, full_nll], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("NLL values must be finite")
    denominator = common_nll - full_nll
    if abs(denominator) < 1e-15:
        return float("nan")
    return float((common_nll - shared_nll) / denominator)


@dataclass(frozen=True)
class LDSSequenceDataset:
    """Complete sequences for marginal Kalman likelihood evaluation."""

    observations: tuple[Array, ...]
    conditions: tuple[Array, ...]
    groups: Array

    def __post_init__(self) -> None:
        observations = tuple(np.asarray(value, dtype=float).copy() for value in self.observations)
        conditions = tuple(np.asarray(value).copy() for value in self.conditions)
        groups = np.asarray(self.groups).copy()
        if not observations or len(observations) != len(conditions):
            raise ValueError("observations and conditions must contain matching sequences")
        n_features = observations[0].shape[1] if observations[0].ndim == 2 else -1
        for observation, condition in zip(observations, conditions, strict=True):
            if observation.ndim != 2 or observation.shape[0] < 2 or observation.shape[1] != n_features:
                raise ValueError("every observation sequence must be [time>=2, common feature]")
            if condition.shape != (observation.shape[0],):
                raise ValueError("each condition vector must match its observation sequence")
            if not np.isfinite(observation).all():
                raise ValueError("observations contain non-finite values")
            observation.setflags(write=False)
            condition.setflags(write=False)
        if groups.shape != (len(observations),):
            raise ValueError("groups must contain one identifier per complete sequence")
        groups.setflags(write=False)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "groups", groups)

    @property
    def n_features(self) -> int:
        return int(self.observations[0].shape[1])

    @property
    def n_observations(self) -> int:
        return int(sum(sequence.shape[0] for sequence in self.observations))

    def subset(self, indices: Sequence[int] | Array) -> "LDSSequenceDataset":
        idx = np.asarray(indices, dtype=int)
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= len(self.observations)):
            raise ValueError("sequence indices are out of range")
        return LDSSequenceDataset(
            tuple(self.observations[index] for index in idx),
            tuple(self.conditions[index] for index in idx),
            self.groups[idx],
        )


@dataclass(frozen=True)
class LDSScore:
    log_likelihood: float
    nll_per_scalar: float
    n_sequences: int
    n_observations: int


@dataclass(frozen=True)
class HiddenContextFilterResult:
    log_likelihood: float
    state_mean: Array
    context_probability: Array
    condition_labels: tuple[object, ...]


class SwitchingLDS:
    """Two-stage fitted linear-Gaussian state-space model with Kalman scoring.

    `common` uses one observation basis and one transition. `shared` uses a
    shared observation basis with condition-specific latent transitions.
    `full` allows both observation bases and transitions to vary by condition.
    PCA initializes the observation model from training sequences only; the
    reported held-out score is the true marginal Kalman log likelihood of the
    resulting state-space model, not a one-step regression likelihood.
    """

    def __init__(
        self,
        family: Family,
        latent_dim: int,
        *,
        ridge: float = 1e-3,
        variance_floor: float = 1e-5,
    ) -> None:
        if family not in {"common", "shared", "full"}:
            raise ValueError("family must be common, shared, or full")
        if latent_dim < 1 or ridge < 0 or variance_floor <= 0:
            raise ValueError("latent_dim/floor must be positive and ridge non-negative")
        self.family = family
        self.latent_dim = int(latent_dim)
        self.ridge = float(ridge)
        self.variance_floor = float(variance_floor)
        self._fitted = False

    def _condition_index(self, value: object) -> int:
        key = _condition_key(value)
        if key not in self.condition_to_index_:
            raise ValueError(f"unseen condition {value!r}")
        return self.condition_to_index_[key]

    def fit(self, train: LDSSequenceDataset) -> "SwitchingLDS":
        if self.latent_dim > train.n_features:
            raise ValueError("latent_dim cannot exceed observation dimension")
        flat = np.concatenate(train.observations, axis=0)
        self.n_features_ = train.n_features
        self.global_mean_ = flat.mean(axis=0)
        self.global_scale_ = flat.std(axis=0, ddof=0)
        self.global_scale_[self.global_scale_ < 1e-12] = 1.0
        standardized = [(sequence - self.global_mean_) / self.global_scale_ for sequence in train.observations]
        labels: list[object] = []
        self.condition_to_index_ = {}
        for condition_sequence in train.conditions:
            for condition in condition_sequence:
                key = _condition_key(condition)
                if key not in self.condition_to_index_:
                    self.condition_to_index_[key] = len(labels)
                    labels.append(condition.item() if isinstance(condition, np.generic) else condition)
        self.conditions_ = tuple(labels)
        k_count = len(labels)

        self.observation_basis_: list[Array] = []
        self.observation_mean_: list[Array] = []
        self.observation_variance_: list[Array] = []
        latent_sequences: list[Array] = []
        if self.family in {"common", "shared"}:
            pca = PCA(n_components=self.latent_dim, svd_solver="full").fit(
                np.concatenate(standardized, axis=0)
            )
            basis = pca.components_.T
            mean = np.zeros(self.n_features_)
            residual = np.concatenate(standardized, axis=0) - (
                np.concatenate(standardized, axis=0) @ basis
            ) @ basis.T
            variance = np.maximum(np.mean(residual**2, axis=0), self.variance_floor)
            self.observation_basis_ = [basis]
            self.observation_mean_ = [mean]
            self.observation_variance_ = [variance]
            latent_sequences = [sequence @ basis for sequence in standardized]
        else:
            flat_conditions = np.concatenate(train.conditions)
            flat_standardized = np.concatenate(standardized, axis=0)
            for label in self.conditions_:
                mask = np.array([_condition_key(value) == _condition_key(label) for value in flat_conditions])
                values = flat_standardized[mask]
                if values.shape[0] <= self.latent_dim:
                    raise ValueError("full LDS has too few training observations for a condition")
                mean = values.mean(axis=0)
                pca = PCA(n_components=self.latent_dim, svd_solver="full").fit(values - mean)
                basis = pca.components_.T
                residual = values - mean - ((values - mean) @ basis) @ basis.T
                self.observation_basis_.append(basis)
                self.observation_mean_.append(mean)
                self.observation_variance_.append(
                    np.maximum(np.mean(residual**2, axis=0), self.variance_floor)
                )
            for sequence, conditions in zip(standardized, train.conditions, strict=True):
                latent = np.empty((sequence.shape[0], self.latent_dim))
                for time, condition in enumerate(conditions):
                    index = self._condition_index(condition)
                    latent[time] = (
                        sequence[time] - self.observation_mean_[index]
                    ) @ self.observation_basis_[index]
                latent_sequences.append(latent)

        transition_count = 1 if self.family == "common" else k_count
        x_rows: list[list[Array]] = [[] for _ in range(transition_count)]
        y_rows: list[list[Array]] = [[] for _ in range(transition_count)]
        initial = []
        for latent, conditions in zip(latent_sequences, train.conditions, strict=True):
            initial.append(latent[0])
            for time in range(latent.shape[0] - 1):
                index = 0 if self.family == "common" else self._condition_index(conditions[time])
                x_rows[index].append(latent[time])
                y_rows[index].append(latent[time + 1])
        self.transition_: list[Array] = []
        self.process_variance_: list[Array] = []
        for inputs, targets in zip(x_rows, y_rows, strict=True):
            if len(inputs) < self.latent_dim:
                raise ValueError("too few within-sequence transitions for LDS fit")
            x = np.asarray(inputs)
            y = np.asarray(targets)
            coefficients = _ridge_coefficients(x, y, self.ridge)
            residual = y - x @ coefficients
            transition = coefficients.T
            spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
            if spectral_radius >= 0.999:
                transition = transition * (0.999 / spectral_radius)
            self.transition_.append(transition)
            self.process_variance_.append(
                np.maximum(np.mean(residual**2, axis=0), self.variance_floor)
            )
        initial_array = np.asarray(initial)
        self.initial_mean_ = initial_array.mean(axis=0)
        if initial_array.shape[0] > 1:
            covariance = np.cov(initial_array, rowvar=False)
            if np.ndim(covariance) == 0:
                covariance = np.array([[float(covariance)]])
        else:
            covariance = np.eye(self.latent_dim)
        self.initial_covariance_ = covariance + self.variance_floor * np.eye(self.latent_dim)
        self._fitted = True
        return self

    def _observation_parameters(self, condition: object) -> tuple[Array, Array, Array]:
        index = 0 if self.family in {"common", "shared"} else self._condition_index(condition)
        return (
            self.observation_basis_[index],
            self.observation_mean_[index],
            self.observation_variance_[index],
        )

    def _transition_parameters(self, condition: object) -> tuple[Array, Array]:
        index = 0 if self.family == "common" else self._condition_index(condition)
        return self.transition_[index], self.process_variance_[index]

    def filter_sequence(self, observations: Array, conditions: Array) -> tuple[float, Array]:
        if not self._fitted:
            raise RuntimeError("LDS must be fit first")
        y = np.asarray(observations, dtype=float)
        c = np.asarray(conditions)
        if y.ndim != 2 or y.shape[1] != self.n_features_ or c.shape != (y.shape[0],):
            raise ValueError("held-out sequence shape differs from training")
        y = (y - self.global_mean_) / self.global_scale_
        mean = self.initial_mean_.copy()
        covariance = self.initial_covariance_.copy()
        posterior = np.empty((y.shape[0], self.latent_dim))
        log_likelihood = 0.0
        identity = np.eye(self.latent_dim)
        for time in range(y.shape[0]):
            basis, observation_mean, observation_variance = self._observation_parameters(c[time])
            innovation = y[time] - observation_mean - basis @ mean
            innovation_covariance = _stabilize_covariance(
                basis @ covariance @ basis.T + np.diag(observation_variance),
                floor=self.variance_floor,
            )
            sign, logdet = np.linalg.slogdet(innovation_covariance)
            if sign <= 0:
                raise FloatingPointError("non-positive innovation covariance")
            solved = np.linalg.solve(innovation_covariance, innovation)
            log_likelihood += -0.5 * (
                self.n_features_ * np.log(2.0 * np.pi) + logdet + innovation @ solved
            )
            gain = np.linalg.solve(innovation_covariance, basis @ covariance).T
            mean = mean + gain @ innovation
            covariance = (identity - gain @ basis) @ covariance @ (identity - gain @ basis).T + (
                gain * observation_variance[None, :]
            ) @ gain.T
            covariance = _stabilize_covariance(covariance)
            posterior[time] = mean
            if time < y.shape[0] - 1:
                transition, process_variance = self._transition_parameters(c[time])
                mean = transition @ mean
                covariance = _stabilize_covariance(
                    transition @ covariance @ transition.T + np.diag(process_variance)
                )
        # Standardization density Jacobian.
        log_likelihood -= y.shape[0] * float(np.sum(np.log(self.global_scale_)))
        return float(log_likelihood), posterior

    def score(self, test: LDSSequenceDataset) -> LDSScore:
        likelihood = 0.0
        observations = 0
        for sequence, conditions in zip(test.observations, test.conditions, strict=True):
            value, _ = self.filter_sequence(sequence, conditions)
            likelihood += value
            observations += sequence.shape[0]
        return LDSScore(
            log_likelihood=float(likelihood),
            nll_per_scalar=float(-likelihood / (observations * self.n_features_)),
            n_sequences=len(test.observations),
            n_observations=observations,
        )

    def filter_hidden_context_sequence(
        self,
        observations: Array,
        *,
        stay_probability: float = 0.95,
    ) -> HiddenContextFilterResult:
        """Causally infer an unobserved discrete context with an IMM filter.

        The true held-out condition sequence is not an input. A fixed sticky
        Markov prior is preregistered through `stay_probability`; Gaussian
        mixtures are moment-matched for each destination context at every
        step. This avoids circular context decoding from an oracle-conditioned
        transition schedule while retaining a tractable switching-LDS filter.
        """

        if not self._fitted:
            raise RuntimeError("LDS must be fit first")
        values = np.asarray(observations, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_ or values.shape[0] < 1:
            raise ValueError("observations must be [time>=1, fitted feature]")
        if not np.isfinite(values).all():
            raise ValueError("observations contain non-finite values")
        if not 0.0 < stay_probability < 1.0:
            raise ValueError("stay_probability must lie in (0, 1)")
        values = (values - self.global_mean_) / self.global_scale_
        k_count = 1 if self.family == "common" else len(self.conditions_)
        if k_count == 1:
            condition = self.conditions_[0]
            likelihood, posterior = self.filter_sequence(
                observations, np.repeat(condition, observations.shape[0])
            )
            return HiddenContextFilterResult(
                likelihood,
                posterior,
                np.ones((observations.shape[0], 1)),
                (condition,),
            )
        off_diagonal = (1.0 - stay_probability) / (k_count - 1)
        discrete_transition = np.full((k_count, k_count), off_diagonal)
        np.fill_diagonal(discrete_transition, stay_probability)
        context_probability = np.full(k_count, 1.0 / k_count)
        means = np.repeat(self.initial_mean_[None, :], k_count, axis=0)
        covariances = np.repeat(self.initial_covariance_[None, :, :], k_count, axis=0)
        probability_history = np.empty((values.shape[0], k_count))
        state_history = np.empty((values.shape[0], self.latent_dim))
        identity = np.eye(self.latent_dim)
        total_log_likelihood = 0.0
        for time, observation in enumerate(values):
            if time == 0:
                prior_probability = context_probability.copy()
                mixed_means = means.copy()
                mixed_covariances = covariances.copy()
            else:
                source_means = np.empty_like(means)
                source_covariances = np.empty_like(covariances)
                for source in range(k_count):
                    transition, process_variance = self._transition_parameters(
                        self.conditions_[source]
                    )
                    source_means[source] = transition @ means[source]
                    source_covariances[source] = _stabilize_covariance(
                        transition @ covariances[source] @ transition.T
                        + np.diag(process_variance)
                    )
                prior_probability = context_probability @ discrete_transition
                mixed_means = np.empty_like(means)
                mixed_covariances = np.empty_like(covariances)
                for destination in range(k_count):
                    weights = context_probability * discrete_transition[:, destination]
                    weights /= max(float(weights.sum()), np.finfo(float).tiny)
                    mixed_mean = np.sum(weights[:, None] * source_means, axis=0)
                    mixed_covariance = np.zeros((self.latent_dim, self.latent_dim))
                    for source in range(k_count):
                        difference = source_means[source] - mixed_mean
                        mixed_covariance += weights[source] * (
                            source_covariances[source] + np.outer(difference, difference)
                        )
                    mixed_means[destination] = mixed_mean
                    mixed_covariances[destination] = _stabilize_covariance(mixed_covariance)

            log_weights = np.empty(k_count)
            updated_means = np.empty_like(means)
            updated_covariances = np.empty_like(covariances)
            for context_index, condition in enumerate(self.conditions_):
                basis, observation_mean, observation_variance = self._observation_parameters(
                    condition
                )
                innovation = observation - observation_mean - basis @ mixed_means[context_index]
                innovation_covariance = _stabilize_covariance(
                    basis @ mixed_covariances[context_index] @ basis.T
                    + np.diag(observation_variance),
                    floor=self.variance_floor,
                )
                sign, logdet = np.linalg.slogdet(innovation_covariance)
                if sign <= 0:
                    raise FloatingPointError("non-positive hidden-filter covariance")
                solved = np.linalg.solve(innovation_covariance, innovation)
                log_weights[context_index] = np.log(
                    max(prior_probability[context_index], np.finfo(float).tiny)
                ) - 0.5 * (
                    self.n_features_ * np.log(2.0 * np.pi)
                    + logdet
                    + innovation @ solved
                )
                gain = np.linalg.solve(
                    innovation_covariance, basis @ mixed_covariances[context_index]
                ).T
                updated_means[context_index] = (
                    mixed_means[context_index] + gain @ innovation
                )
                updated_covariances[context_index] = _stabilize_covariance(
                    (identity - gain @ basis)
                    @ mixed_covariances[context_index]
                    @ (identity - gain @ basis).T
                    + (gain * observation_variance[None, :]) @ gain.T
                )
            step_log_likelihood = float(logsumexp(log_weights))
            total_log_likelihood += step_log_likelihood
            context_probability = np.exp(log_weights - step_log_likelihood)
            means = updated_means
            covariances = updated_covariances
            probability_history[time] = context_probability
            state_history[time] = np.sum(context_probability[:, None] * means, axis=0)
        total_log_likelihood -= values.shape[0] * float(np.sum(np.log(self.global_scale_)))
        return HiddenContextFilterResult(
            log_likelihood=float(total_log_likelihood),
            state_mean=state_history,
            context_probability=probability_history,
            condition_labels=self.conditions_,
        )

    def score_hidden_context(
        self,
        test: LDSSequenceDataset,
        *,
        stay_probability: float = 0.95,
    ) -> LDSScore:
        likelihood = 0.0
        observations = 0
        for sequence in test.observations:
            result = self.filter_hidden_context_sequence(
                sequence, stay_probability=stay_probability
            )
            likelihood += result.log_likelihood
            observations += sequence.shape[0]
        return LDSScore(
            log_likelihood=float(likelihood),
            nll_per_scalar=float(-likelihood / (observations * self.n_features_)),
            n_sequences=len(test.observations),
            n_observations=observations,
        )

    def parameter_count(self) -> int:
        if not self._fitted:
            raise RuntimeError("LDS must be fit first")
        n, d, k = self.n_features_, self.latent_dim, len(self.conditions_)
        # The train-fitted global centering and scaling vectors are part of the
        # fitted model and must be included in absolute parameter comparisons.
        global_normalization = 2 * n
        basis_degrees = n * d - d * (d + 1) // 2
        initial = d + d * (d + 1) // 2
        if self.family == "common":
            return int(
                global_normalization + basis_degrees + n + d * d + d + initial
            )
        if self.family == "shared":
            return int(
                global_normalization
                + basis_degrees
                + n
                + k * (d * d + d)
                + initial
            )
        return int(
            global_normalization
            + k * (basis_degrees + n + d * d + d)
            + (k - 1) * n
            + initial
        )
