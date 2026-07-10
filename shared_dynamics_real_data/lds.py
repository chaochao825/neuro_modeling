"""Shared-basis linear-Gaussian state-space models.

Held-out marginal likelihood uses the matrix determinant lemma and Woodbury
identity.  Its largest observation-space object is ``N x d``; an ``N x N``
innovation covariance is never materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve, subspace_angles

from .preprocessing import BasisControl, fit_controlled_basis
from .splits import TimeSegment


Family = Literal["common", "shared", "separate"]


def _positive_definite(matrix: np.ndarray, floor: float) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def woodbury_gaussian_logpdf(
    residual: np.ndarray,
    observation: np.ndarray,
    latent_covariance: np.ndarray,
    observation_variance: np.ndarray,
    *,
    observation_information: np.ndarray | None = None,
    weighted_observation_transpose: np.ndarray | None = None,
) -> float:
    """Log N(residual; 0, diag(R) + C P C.T) without an ``N x N`` matrix."""

    r = np.asarray(residual, dtype=float)
    c = np.asarray(observation, dtype=float)
    p = np.asarray(latent_covariance, dtype=float)
    diag_r = np.asarray(observation_variance, dtype=float)
    if r.ndim != 1 or c.ndim != 2 or c.shape[0] != r.size:
        raise ValueError("residual/observation shapes do not match")
    if p.shape != (c.shape[1], c.shape[1]) or diag_r.shape != r.shape:
        raise ValueError("covariance shapes do not match")
    if not np.isfinite(r).all() or not np.isfinite(c).all() or not np.isfinite(p).all():
        raise ValueError("inputs must be finite")
    if not np.isfinite(diag_r).all() or np.any(diag_r <= 0):
        raise ValueError("observation variances must be finite and positive")

    information = (
        c.T @ (c / diag_r[:, None])
        if observation_information is None
        else np.asarray(observation_information, dtype=float)
    )
    weighted_transpose = (
        c.T / diag_r[None, :]
        if weighted_observation_transpose is None
        else np.asarray(weighted_observation_transpose, dtype=float)
    )
    if information.shape != p.shape or weighted_transpose.shape != c.T.shape:
        raise ValueError("cached observation information has the wrong shape")
    p = _positive_definite(p, 1e-12)
    latent_chol = np.linalg.cholesky(p)
    small = np.eye(c.shape[1]) + latent_chol.T @ information @ latent_chol
    factor = cho_factor(small, lower=True, check_finite=False)
    weighted_r = r / diag_r
    projected = latent_chol.T @ (weighted_transpose @ r)
    correction = float(projected @ cho_solve(factor, projected, check_finite=False))
    quadratic = max(0.0, float(r @ weighted_r) - correction)
    logdet = float(np.log(diag_r).sum()) + 2.0 * float(
        np.log(np.diag(factor[0])).sum()
    )
    return -0.5 * (r.size * np.log(2.0 * np.pi) + logdet + quadratic)


def effective_rank(matrix: np.ndarray, *, tolerance: float = 1e-12) -> float:
    """Shannon effective rank of normalized singular values.

    Squared singular values are reserved for the separate top-k energy metric.
    """

    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    if not singular.size or singular[0] == 0:
        return 0.0
    singular = singular[singular > tolerance * singular[0]]
    probabilities = singular / singular.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


@dataclass(frozen=True)
class LDSScores:
    marginal_log_likelihood: float
    nll_per_scalar: float
    likelihood_coordinate: str
    standardized_marginal_log_likelihood: float
    standardized_nll_per_scalar: float
    one_step_r2: float
    rollout_nrmse: float
    prediction_metric_coordinate: str
    parameter_count: int
    effective_rank: float
    subspace_angle_degrees: float
    n_observations: int
    n_sequences: int


def _fit_affine_dynamics(
    current: np.ndarray,
    following: np.ndarray,
    *,
    ridge: float,
    variance_floor: float,
    max_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack([current, np.ones(current.shape[0])])
    gram = design.T @ design
    regularizer = np.eye(design.shape[1]) * ridge
    regularizer[-1, -1] = 0.0
    try:
        coefficients = np.linalg.solve(gram + regularizer, design.T @ following)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram + regularizer) @ design.T @ following
    transition = coefficients[:-1].T
    offset = coefficients[-1]
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if radius > max_radius:
        transition *= max_radius / radius
        # Refit the affine term after changing A; otherwise stabilization
        # creates a systematic residual that variance() would silently drop.
        offset = np.mean(following - current @ transition.T, axis=0)
    residual = following - (current @ transition.T + offset)
    process_variance = np.maximum(np.mean(residual**2, axis=0), variance_floor)
    return transition, offset, process_variance


class SharedBasisLDS:
    """Common, shared-basis switching, or separate-basis switching LDS."""

    def __init__(
        self,
        family: Family,
        latent_dim: int,
        *,
        basis_control: BasisControl = "aligned",
        random_state: int = 0,
        ridge: float = 1e-4,
        variance_floor: float = 1e-5,
        max_radius: float = 0.995,
    ) -> None:
        if family not in {"common", "shared", "separate"}:
            raise ValueError("family must be common, shared, or separate")
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, (int, np.integer)):
            raise TypeError("latent_dim must be an integer")
        if int(latent_dim) < 1:
            raise ValueError("latent_dim must be positive")
        if basis_control not in {"aligned", "random", "orthogonal", "shuffled"}:
            raise ValueError("unknown basis control")
        if isinstance(random_state, bool) or not isinstance(
            random_state, (int, np.integer)
        ):
            raise TypeError("random_state must be an integer")
        if (
            not np.isfinite(ridge)
            or ridge < 0
            or not np.isfinite(variance_floor)
            or variance_floor <= 0
            or not np.isfinite(max_radius)
            or not 0 < max_radius <= 1
        ):
            raise ValueError("invalid ridge, variance_floor, or max_radius")
        self.family = family
        self.latent_dim = int(latent_dim)
        self.basis_control = basis_control
        self.random_state = int(random_state)
        self.ridge = float(ridge)
        self.variance_floor = float(variance_floor)
        self.max_radius = float(max_radius)

    def fit(self, segments: Sequence[TimeSegment]) -> "SharedBasisLDS":
        if not segments:
            raise ValueError("training segments cannot be empty")
        contexts = tuple(sorted({segment.context for segment in segments}))
        if any(not any(s.context == c and s.values.shape[0] >= 2 for s in segments) for c in contexts):
            raise ValueError("every context needs at least one train transition")
        n_features_set = {segment.values.shape[1] for segment in segments}
        if len(n_features_set) != 1:
            raise ValueError("segments have inconsistent feature counts")
        n_features = n_features_set.pop()
        if self.latent_dim > min(n_features, sum(s.values.shape[0] for s in segments)):
            raise ValueError("latent_dim exceeds available train dimensions")

        self.contexts_ = contexts
        self.n_features_ = n_features
        pooled = np.vstack([segment.values for segment in segments])
        centered_train = pooled - pooled.mean(axis=0, keepdims=True)
        self.rollout_reference_scale_ = max(
            float(np.sqrt(np.mean(centered_train**2))),
            np.finfo(float).eps,
        )
        if self.family == "separate":
            self.bases_ = {
                context: fit_controlled_basis(
                    np.vstack([s.values for s in segments if s.context == context]),
                    self.latent_dim,
                    control=self.basis_control,
                    random_state=self.random_state + index,
                )
                for index, context in enumerate(contexts)
            }
        else:
            shared_basis = fit_controlled_basis(
                pooled,
                self.latent_dim,
                control=self.basis_control,
                random_state=self.random_state,
            )
            self.bases_ = {context: shared_basis for context in contexts}

        if self.family == "separate":
            self.observation_means_ = {
                context: np.vstack(
                    [s.values for s in segments if s.context == context]
                ).mean(axis=0)
                for context in contexts
            }
        else:
            # TrainOnlyPreprocessor makes the pooled mean zero. Store it
            # explicitly to keep the observation equation numerically exact.
            pooled_mean = pooled.mean(axis=0)
            self.observation_means_ = {
                context: pooled_mean for context in contexts
            }
        latent_segments = {
            id(segment): (
                segment.values - self.observation_means_[segment.context]
            )
            @ self.bases_[segment.context]
            for segment in segments
        }
        transitions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for context in contexts:
            current = []
            following = []
            for segment in segments:
                if segment.context != context or segment.values.shape[0] < 2:
                    continue
                latent = latent_segments[id(segment)]
                current.append(latent[:-1])
                following.append(latent[1:])
            transitions[context] = (np.vstack(current), np.vstack(following))

        self.transitions_: dict[str, np.ndarray] = {}
        self.offsets_: dict[str, np.ndarray] = {}
        self.process_variances_: dict[str, np.ndarray] = {}
        if self.family == "common":
            current = np.vstack([transitions[c][0] for c in contexts])
            following = np.vstack([transitions[c][1] for c in contexts])
            fitted = _fit_affine_dynamics(
                current,
                following,
                ridge=self.ridge,
                variance_floor=self.variance_floor,
                max_radius=self.max_radius,
            )
            for context in contexts:
                self.transitions_[context], self.offsets_[context], self.process_variances_[context] = fitted
        else:
            for context in contexts:
                fitted = _fit_affine_dynamics(
                    *transitions[context],
                    ridge=self.ridge,
                    variance_floor=self.variance_floor,
                    max_radius=self.max_radius,
                )
                self.transitions_[context], self.offsets_[context], self.process_variances_[context] = fitted

        self.observation_variances_: dict[str, np.ndarray] = {}
        if self.family == "separate":
            for context in contexts:
                matrix = np.vstack([s.values for s in segments if s.context == context])
                basis = self.bases_[context]
                centered = matrix - self.observation_means_[context]
                residual = centered - (centered @ basis) @ basis.T
                self.observation_variances_[context] = np.maximum(
                    np.mean(residual**2, axis=0), self.variance_floor
                )
        else:
            basis = self.bases_[contexts[0]]
            centered = pooled - self.observation_means_[contexts[0]]
            residual = centered - (centered @ basis) @ basis.T
            variance = np.maximum(
                np.mean(residual**2, axis=0), self.variance_floor
            )
            self.observation_variances_ = {context: variance for context in contexts}
        # These two O(d^2) / O(Nd) terms are reused at every Kalman update.
        self.observation_information_ = {
            context: self.bases_[context].T
            @ (
                self.bases_[context]
                / self.observation_variances_[context][:, None]
            )
            for context in contexts
        }
        self.weighted_observation_transposes_ = {
            context: self.bases_[context].T
            / self.observation_variances_[context][None, :]
            for context in contexts
        }
        self._fitted = True
        return self

    def _check_context(self, context: str) -> None:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("model must be fitted first")
        if context not in self.contexts_:
            raise ValueError(f"unseen context {context!r}")

    def _filter_update(
        self,
        observation_value: np.ndarray,
        context: str,
        prior_mean: np.ndarray,
        prior_covariance: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        basis = self.bases_[context]
        diag_r = self.observation_variances_[context]
        residual = (
            observation_value
            - self.observation_means_[context]
            - basis @ prior_mean
        )
        logpdf = woodbury_gaussian_logpdf(
            residual,
            basis,
            prior_covariance,
            diag_r,
            observation_information=self.observation_information_[context],
            weighted_observation_transpose=self.weighted_observation_transposes_[context],
        )
        prior_covariance = _positive_definite(prior_covariance, self.variance_floor)
        prior_factor = cho_factor(prior_covariance, lower=True, check_finite=False)
        prior_precision = cho_solve(
            prior_factor, np.eye(self.latent_dim), check_finite=False
        )
        posterior_precision = prior_precision + self.observation_information_[context]
        posterior_factor = cho_factor(
            posterior_precision, lower=True, check_finite=False
        )
        innovation_information = self.weighted_observation_transposes_[context] @ residual
        posterior_covariance = cho_solve(
            posterior_factor, np.eye(self.latent_dim), check_finite=False
        )
        posterior_mean = prior_mean + posterior_covariance @ innovation_information
        return logpdf, posterior_mean, posterior_covariance

    def score(self, segments: Sequence[TimeSegment]) -> LDSScores:
        if not segments:
            raise ValueError("test segments cannot be empty")
        total_logpdf = 0.0
        n_observations = 0
        one_targets: list[np.ndarray] = []
        one_predictions: list[np.ndarray] = []
        rollout_targets: list[np.ndarray] = []
        rollout_predictions: list[np.ndarray] = []
        for segment in segments:
            self._check_context(segment.context)
            if segment.values.shape[1] != self.n_features_:
                raise ValueError("test segment has the wrong feature count")
            context = segment.context
            basis = self.bases_[context]
            transition = self.transitions_[context]
            offset = self.offsets_[context]
            q = self.process_variances_[context]
            mean = np.zeros(self.latent_dim)
            covariance = np.eye(self.latent_dim)
            rollout_mean: np.ndarray | None = None
            for time, value in enumerate(segment.values):
                if time > 0:
                    one_predictions.append(
                        self.observation_means_[context] + basis @ mean
                    )
                    one_targets.append(value)
                logpdf, posterior_mean, posterior_covariance = self._filter_update(
                    value, context, mean, covariance
                )
                total_logpdf += logpdf
                n_observations += 1
                if time == 0:
                    rollout_mean = transition @ posterior_mean + offset
                elif rollout_mean is not None:
                    rollout_predictions.append(
                        self.observation_means_[context] + basis @ rollout_mean
                    )
                    rollout_targets.append(value)
                    rollout_mean = transition @ rollout_mean + offset
                mean = transition @ posterior_mean + offset
                covariance = transition @ posterior_covariance @ transition.T + np.diag(q)

        if not one_targets or not rollout_targets:
            raise ValueError("test data need at least two points per sequence")
        one_y = np.vstack(one_targets)
        one_hat = np.vstack(one_predictions)
        denominator = float(np.sum((one_y - one_y.mean(axis=0)) ** 2))
        one_step_r2 = 1.0 - float(np.sum((one_y - one_hat) ** 2)) / max(
            denominator, np.finfo(float).eps
        )
        rollout_y = np.vstack(rollout_targets)
        rollout_hat = np.vstack(rollout_predictions)
        rmse = float(np.sqrt(np.mean((rollout_y - rollout_hat) ** 2)))
        rollout_nrmse = rmse / self.rollout_reference_scale_
        return LDSScores(
            marginal_log_likelihood=float(total_logpdf),
            nll_per_scalar=float(-total_logpdf / (n_observations * self.n_features_)),
            likelihood_coordinate="train_standardized_selected_units",
            standardized_marginal_log_likelihood=float(total_logpdf),
            standardized_nll_per_scalar=float(
                -total_logpdf / (n_observations * self.n_features_)
            ),
            one_step_r2=float(one_step_r2),
            rollout_nrmse=float(rollout_nrmse),
            prediction_metric_coordinate="train_standardized_selected_units",
            parameter_count=self.parameter_count(),
            effective_rank=self.mean_effective_rank(),
            subspace_angle_degrees=self.subspace_angle_degrees(),
            n_observations=n_observations,
            n_sequences=len(segments),
        )

    def parameter_count(self) -> int:
        """Count continuous fitted degrees of freedom.

        Bases use the Stiefel-manifold dimension ``N*d-d*(d+1)/2``.  The
        shared train-fitted mean and scale contribute ``2*N``.  Initial latent
        mean/covariance are fixed to zero/identity and therefore add no fitted
        parameters.  Discrete unit indices are not continuous parameters.
        """

        if not getattr(self, "_fitted", False):
            raise RuntimeError("model must be fitted first")
        n, d, k = self.n_features_, self.latent_dim, len(self.contexts_)
        # A seeded random control basis is fixed before seeing activity and has
        # no fitted continuous degrees of freedom. PCA-derived aligned,
        # orthogonal, and row-shuffled bases remain train-fitted.
        basis_dof = (
            0
            if self.basis_control == "random"
            else n * d - d * (d + 1) // 2
        )
        preprocessing = 2 * n
        dynamic_per = d * d + 2 * d  # A, affine offset, diagonal Q
        if self.family == "common":
            return int(preprocessing + basis_dof + dynamic_per + n)
        if self.family == "shared":
            return int(preprocessing + basis_dof + k * dynamic_per + n)
        context_means = (k - 1) * n
        return int(
            preprocessing + k * (basis_dof + dynamic_per + n) + context_means
        )

    def mean_effective_rank(self) -> float:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("model must be fitted first")
        return float(
            np.mean([effective_rank(self.transitions_[c]) for c in self.contexts_])
        )

    def mean_topk_singular_energy(self, k: int) -> float:
        """Mean fraction of transition singular energy captured by top ``k``."""

        if not getattr(self, "_fitted", False):
            raise RuntimeError("model must be fitted first")
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer")
        if not 1 <= int(k) <= self.latent_dim:
            raise ValueError("k must lie between one and latent_dim")
        k = int(k)
        fractions = []
        for context in self.contexts_:
            energy = np.linalg.svd(
                self.transitions_[context], compute_uv=False
            ) ** 2
            fractions.append(float(energy[:k].sum() / max(energy.sum(), 1e-15)))
        return float(np.mean(fractions))

    def subspace_angle_degrees(self) -> float:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("model must be fitted first")
        if self.family != "separate" or len(self.contexts_) < 2:
            return 0.0
        angles = [
            float(np.max(subspace_angles(self.bases_[first], self.bases_[second])))
            for first, second in combinations(self.contexts_, 2)
        ]
        return float(np.degrees(np.mean(angles)))
