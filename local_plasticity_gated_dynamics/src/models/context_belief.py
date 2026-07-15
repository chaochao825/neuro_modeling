"""Leakage-safe causal belief gates for binary hidden-context tasks.

Every unsupervised gate in this module accepts only the observation capability
defined by :class:`src.tasks.hidden_context.GateObservationBatch`.  The module
deliberately uses structural validation instead of importing the task class so
that gate code cannot acquire evaluation truth through a richer dataset object.

The supervised gate is an explicitly labelled upper bound.  It receives hidden
labels only through ``fit_supervised``; its prediction API is observation-only.
No implementation uses PyTorch, autograd, or back-propagation through time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression


Array = np.ndarray
_EMPTY_INT = np.empty(0, dtype=int)
_EMPTY_INT.setflags(write=False)


def _validated_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validated_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        valid = result >= minimum if minimum_inclusive else result > minimum
        if not valid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}")
    if maximum is not None:
        valid = result <= maximum if maximum_inclusive else result < maximum
        if not valid:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {operator} {maximum}")
    return result


def _readonly_int_vector(
    value: object,
    *,
    name: str,
    length: int | None = None,
    allow_negative_one: bool = False,
) -> Array:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one dimensional")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    result = np.array(raw, dtype=int, order="C", copy=True)
    if length is not None and result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    minimum = -1 if allow_negative_one else 0
    if np.any(result < minimum):
        raise ValueError(f"{name} values must be at least {minimum}")
    result.setflags(write=False)
    return result


def _readonly_beliefs(value: object, *, length: int) -> Array:
    beliefs = np.array(value, dtype=float, order="C", copy=True)
    if beliefs.shape != (length, 2):
        raise ValueError(f"beliefs must have shape ({length}, 2)")
    if not np.isfinite(beliefs).all() or np.any(beliefs < 0.0):
        raise ValueError("beliefs must be finite and non-negative")
    totals = beliefs.sum(axis=1)
    if not np.allclose(totals, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("each belief row must sum to one")
    beliefs /= totals[:, None]
    beliefs.setflags(write=False)
    return beliefs


def _parameter_tuple(value: object) -> tuple[tuple[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("parameters must be a sequence of name/value pairs")
    try:
        pairs = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("parameters must be a sequence of name/value pairs") from error
    result: list[tuple[str, Any]] = []
    names: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("each parameter must be a name/value pair")
        name, raw = pair
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("parameter names must be unique non-empty strings")
        if isinstance(raw, np.generic):
            raw = raw.item()
        if not isinstance(raw, (str, bool, int, float, type(None))):
            raise TypeError("parameter values must be scalar primitives")
        if isinstance(raw, float) and not np.isfinite(raw):
            raise ValueError("numeric parameter values must be finite")
        names.add(name)
        result.append((name, raw))
    return tuple(result)


@dataclass(frozen=True)
class _ObservationView:
    cues: Array
    trial_ids: Array
    episode_ids: Array
    trial_in_episode: Array
    episode_start: Array

    @property
    def episode_slices(self) -> tuple[slice, ...]:
        starts = np.flatnonzero(self.episode_start)
        stops = np.concatenate([starts[1:], np.array([self.cues.size])])
        return tuple(
            slice(int(start), int(stop))
            for start, stop in zip(starts, stops, strict=True)
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for array in (
            self.cues,
            self.trial_ids,
            self.episode_ids,
            self.trial_in_episode,
            self.episode_start,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()


def _attribute(value: object, primary: str, alias: str | None = None) -> object:
    if hasattr(value, primary):
        return getattr(value, primary)
    if alias is not None and hasattr(value, alias):
        return getattr(value, alias)
    alternatives = f" or {alias}" if alias is not None else ""
    raise TypeError(f"observations must expose {primary}{alternatives}")


def _observation_view(observations: object) -> _ObservationView:
    """Validate only the structural observation capability used by gates."""

    cues = _readonly_int_vector(
        _attribute(observations, "cue_observations", "cues"),
        name="cue_observations",
    )
    if cues.size == 0:
        raise ValueError("cue_observations cannot be empty")
    if not np.isin(cues, [0, 1]).all():
        raise ValueError("cue_observations must be binary")
    n_trials = cues.size
    episode_ids = _readonly_int_vector(
        _attribute(observations, "episode_ids"),
        name="episode_ids",
        length=n_trials,
    )

    if hasattr(observations, "trial_ids"):
        trial_ids = _readonly_int_vector(
            getattr(observations, "trial_ids"),
            name="trial_ids",
            length=n_trials,
        )
    else:
        trial_ids = _readonly_int_vector(
            np.arange(n_trials), name="trial_ids", length=n_trials
        )
    if np.unique(trial_ids).size != n_trials:
        raise ValueError("trial_ids must be unique")

    if hasattr(observations, "episode_trial_indices"):
        raw_positions = getattr(observations, "episode_trial_indices")
    elif hasattr(observations, "trial_in_episode"):
        raw_positions = getattr(observations, "trial_in_episode")
    else:
        raw_positions = None

    starts = np.ones(n_trials, dtype=bool)
    if n_trials > 1:
        starts[1:] = episode_ids[1:] != episode_ids[:-1]
    ordered_episodes = episode_ids[starts]
    if np.unique(ordered_episodes).size != ordered_episodes.size:
        raise ValueError("each episode must occupy one contiguous segment")

    derived_positions = np.empty(n_trials, dtype=int)
    start_indices = np.flatnonzero(starts)
    stop_indices = np.concatenate([start_indices[1:], np.array([n_trials])])
    for start, stop in zip(start_indices, stop_indices, strict=True):
        derived_positions[start:stop] = np.arange(stop - start)

    if raw_positions is None:
        positions = _readonly_int_vector(
            derived_positions, name="trial_in_episode", length=n_trials
        )
    else:
        positions = _readonly_int_vector(
            raw_positions, name="trial_in_episode", length=n_trials
        )
        if not np.array_equal(positions, derived_positions):
            raise ValueError(
                "trial_in_episode must be contiguous and zero based within episodes"
            )

    if hasattr(observations, "episode_start"):
        raw_start = np.asarray(getattr(observations, "episode_start"))
        if raw_start.shape != (n_trials,) or (
            raw_start.dtype != bool and not np.isin(raw_start, [0, 1]).all()
        ):
            raise TypeError("episode_start must be a boolean trial vector")
        if not np.array_equal(raw_start.astype(bool), starts):
            raise ValueError("episode_start disagrees with episode_ids")
    starts = np.array(starts, dtype=bool, copy=True)
    starts.setflags(write=False)
    return _ObservationView(cues, trial_ids, episode_ids, positions, starts)


@dataclass(frozen=True)
class GatePrediction:
    """A causal belief trajectory plus explicit fit and intervention provenance."""

    beliefs: Array
    trial_ids: Array
    episode_ids: Array
    gate_name: str
    fit_trial_ids: Array = field(default_factory=lambda: _EMPTY_INT.copy())
    fit_episode_ids: Array = field(default_factory=lambda: _EMPTY_INT.copy())
    fit_accessed_true_context: bool = False
    test_accessed_true_context: bool = False
    intervention: str = "none"
    source_trial_ids: Array | None = None
    source_episode_ids: Array | None = None
    base_prediction_fingerprint: str | None = None
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.gate_name, str) or not self.gate_name:
            raise ValueError("gate_name must be a non-empty string")
        if not isinstance(self.intervention, str) or not self.intervention:
            raise ValueError("intervention must be a non-empty string")
        for name in ("fit_accessed_true_context", "test_accessed_true_context"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean")
        trial_ids = _readonly_int_vector(self.trial_ids, name="trial_ids")
        if trial_ids.size == 0 or np.unique(trial_ids).size != trial_ids.size:
            raise ValueError("trial_ids must be non-empty and unique")
        n_trials = trial_ids.size
        episode_ids = _readonly_int_vector(
            self.episode_ids, name="episode_ids", length=n_trials
        )
        beliefs = _readonly_beliefs(self.beliefs, length=n_trials)
        fit_trial_ids = _readonly_int_vector(self.fit_trial_ids, name="fit_trial_ids")
        fit_episode_ids = _readonly_int_vector(
            self.fit_episode_ids, name="fit_episode_ids"
        )
        if np.unique(fit_trial_ids).size != fit_trial_ids.size:
            raise ValueError("fit_trial_ids must be unique")
        if np.unique(fit_episode_ids).size != fit_episode_ids.size:
            raise ValueError("fit_episode_ids must be unique")
        source_trial_ids = _readonly_int_vector(
            trial_ids if self.source_trial_ids is None else self.source_trial_ids,
            name="source_trial_ids",
            length=n_trials,
            allow_negative_one=True,
        )
        source_episode_ids = _readonly_int_vector(
            episode_ids if self.source_episode_ids is None else self.source_episode_ids,
            name="source_episode_ids",
            length=n_trials,
            allow_negative_one=True,
        )
        if self.base_prediction_fingerprint is not None and (
            not isinstance(self.base_prediction_fingerprint, str)
            or not self.base_prediction_fingerprint
        ):
            raise ValueError("base_prediction_fingerprint must be None or non-empty")
        object.__setattr__(self, "beliefs", beliefs)
        object.__setattr__(self, "trial_ids", trial_ids)
        object.__setattr__(self, "episode_ids", episode_ids)
        object.__setattr__(self, "fit_trial_ids", fit_trial_ids)
        object.__setattr__(self, "fit_episode_ids", fit_episode_ids)
        object.__setattr__(self, "source_trial_ids", source_trial_ids)
        object.__setattr__(self, "source_episode_ids", source_episode_ids)
        object.__setattr__(self, "parameters", _parameter_tuple(self.parameters))
        object.__setattr__(
            self, "fit_accessed_true_context", bool(self.fit_accessed_true_context)
        )
        object.__setattr__(
            self, "test_accessed_true_context", bool(self.test_accessed_true_context)
        )

    @property
    def context_probability(self) -> Array:
        result = self.beliefs[:, 1].copy()
        result.setflags(write=False)
        return result

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for scalar in (
            self.gate_name,
            self.intervention,
            self.fit_accessed_true_context,
            self.test_accessed_true_context,
            self.base_prediction_fingerprint,
            self.parameters,
        ):
            digest.update(repr(scalar).encode("utf-8"))
            digest.update(b"\0")
        for array in (
            self.beliefs,
            self.trial_ids,
            self.episode_ids,
            self.fit_trial_ids,
            self.fit_episode_ids,
            self.source_trial_ids,
            self.source_episode_ids,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "gate_fit_accessed_true_context": self.fit_accessed_true_context,
            "gate_test_accessed_true_context": self.test_accessed_true_context,
            "fit_accessed_true_context": self.fit_accessed_true_context,
            "test_accessed_true_context": self.test_accessed_true_context,
            "fit_trial_ids": self.fit_trial_ids.tolist(),
            "fit_episode_ids": self.fit_episode_ids.tolist(),
            "intervention": self.intervention,
            "base_prediction_fingerprint": self.base_prediction_fingerprint,
            "prediction_fingerprint": self.fingerprint,
            **dict(self.parameters),
        }


def _safe_log(probabilities: Array) -> Array:
    result = np.full_like(np.asarray(probabilities, dtype=float), -np.inf)
    positive = probabilities > 0.0
    np.log(probabilities, out=result, where=positive)
    return result


def _row_normalize(matrix: Array) -> Array:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (2, 2) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("a transition or emission matrix must be finite 2-by-2")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("a transition or emission matrix cannot have an empty row")
    return values / totals


def _transition_matrix(hazard: float) -> Array:
    return np.array([[1.0 - hazard, hazard], [hazard, 1.0 - hazard]])


def _emission_matrix(reliability: float) -> Array:
    return np.array(
        [[reliability, 1.0 - reliability], [1.0 - reliability, reliability]]
    )


def _posterior(prior: Array, emission: Array, cue: int, temperature: float) -> Array:
    # ``inverse_temperature`` is a cue-likelihood gain.  Recurrent priors retain
    # their Bayesian scale, matching the causal two-slice rule used in MD fit.
    log_weight = _safe_log(prior) + temperature * _safe_log(emission[:, cue])
    normalizer = float(logsumexp(log_weight))
    if not np.isfinite(normalizer):
        raise ValueError("the observed cue has zero probability under the gate model")
    return np.exp(log_weight - normalizer)


def _causal_filter(
    view: _ObservationView,
    *,
    transition: Array,
    emission: Array,
    inverse_temperature: float = 1.0,
) -> Array:
    transition = _row_normalize(transition)
    emission = _row_normalize(emission)
    temperature = _validated_real(
        inverse_temperature,
        name="inverse_temperature",
        minimum=0.0,
        minimum_inclusive=False,
    )
    beliefs = np.empty((view.cues.size, 2), dtype=float)
    for episode in view.episode_slices:
        belief = np.full(2, 0.5)
        for local_index, index in enumerate(range(episode.start, episode.stop)):
            prior = belief if local_index == 0 else transition.T @ belief
            belief = _posterior(prior, emission, int(view.cues[index]), temperature)
            beliefs[index] = belief
    return beliefs


def _causal_predictive_prior(
    view: _ObservationView,
    *,
    transition: Array,
    emission: Array,
    inverse_temperature: float = 1.0,
) -> Array:
    """Return the predictive prior ``p(z_t | y_{<t})``.

    The value emitted
    for trial ``t`` is frozen before that trial's cue is incorporated.  The
    current cue updates a private posterior used only to form trial ``t+1``'s
    predictive prior.  Every episode therefore starts from exactly ``[.5,
    .5]`` and cannot inherit evidence from the preceding episode.
    """

    transition = _row_normalize(transition)
    emission = _row_normalize(emission)
    temperature = _validated_real(
        inverse_temperature,
        name="inverse_temperature",
        minimum=0.0,
        minimum_inclusive=False,
    )
    beliefs = np.empty((view.cues.size, 2), dtype=float)
    for episode in view.episode_slices:
        posterior = np.full(2, 0.5)
        for local_index, index in enumerate(range(episode.start, episode.stop)):
            prior = (
                np.full(2, 0.5)
                if local_index == 0
                else transition.T @ posterior
            )
            beliefs[index] = prior
            posterior = _posterior(
                prior,
                emission,
                int(view.cues[index]),
                temperature,
            )
    return beliefs


def _prediction(
    view: _ObservationView,
    beliefs: Array,
    *,
    gate_name: str,
    fit_trial_ids: Array = _EMPTY_INT,
    fit_episode_ids: Array = _EMPTY_INT,
    fit_accessed_true_context: bool = False,
    parameters: tuple[tuple[str, Any], ...] = (),
) -> GatePrediction:
    return GatePrediction(
        beliefs=beliefs,
        trial_ids=view.trial_ids,
        episode_ids=view.episode_ids,
        gate_name=gate_name,
        fit_trial_ids=fit_trial_ids,
        fit_episode_ids=fit_episode_ids,
        fit_accessed_true_context=fit_accessed_true_context,
        test_accessed_true_context=False,
        parameters=parameters,
    )


class OracleBayesianFilter:
    """Causal Bayes filter that knows HMM parameters but never latent states."""

    def __init__(
        self,
        context_hazard: float,
        cue_reliability: float,
        *,
        seed: int = 0,
    ) -> None:
        self.context_hazard = _validated_real(
            context_hazard,
            name="context_hazard",
            minimum=0.0,
            maximum=1.0,
        )
        self.cue_reliability = _validated_real(
            cue_reliability,
            name="cue_reliability",
            minimum=0.5,
            maximum=1.0,
        )
        self.seed = _validated_integer(seed, name="seed")

    def predict(self, observations: object) -> GatePrediction:
        view = _observation_view(observations)
        beliefs = _causal_filter(
            view,
            transition=_transition_matrix(self.context_hazard),
            emission=_emission_matrix(self.cue_reliability),
        )
        return _prediction(
            view,
            beliefs,
            gate_name="oracle_bayesian_filter",
            parameters=(
                ("context_hazard", self.context_hazard),
                ("cue_reliability", self.cue_reliability),
                ("seed", self.seed),
            ),
        )


def _expectation_statistics(
    view: _ObservationView, hazard: float, reliability: float
) -> tuple[float, float, float, float, float]:
    transition = _transition_matrix(hazard)
    emission = _emission_matrix(reliability)
    log_likelihood = 0.0
    expected_switches = 0.0
    transition_count = 0.0
    expected_matches = 0.0
    observation_count = 0.0
    for episode in view.episode_slices:
        cues = view.cues[episode]
        length = cues.size
        alpha = np.empty((length, 2), dtype=float)
        scales = np.empty(length, dtype=float)
        weighted = 0.5 * emission[:, int(cues[0])]
        scales[0] = weighted.sum()
        if scales[0] <= 0.0:
            return -np.inf, 0.0, 0.0, 0.0, 0.0
        alpha[0] = weighted / scales[0]
        for time in range(1, length):
            weighted = (transition.T @ alpha[time - 1]) * emission[:, int(cues[time])]
            scales[time] = weighted.sum()
            if scales[time] <= 0.0:
                return -np.inf, 0.0, 0.0, 0.0, 0.0
            alpha[time] = weighted / scales[time]
        log_likelihood += float(np.log(scales).sum())

        beta = np.ones((length, 2), dtype=float)
        for time in range(length - 2, -1, -1):
            beta[time] = transition @ (
                emission[:, int(cues[time + 1])] * beta[time + 1]
            )
            beta[time] /= scales[time + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        expected_matches += float(gamma[np.arange(length), cues].sum())
        observation_count += float(length)

        for time in range(length - 1):
            following = emission[:, int(cues[time + 1])] * beta[time + 1]
            xi = alpha[time, :, None] * transition * following[None, :]
            total = xi.sum()
            if total <= 0.0:
                raise RuntimeError("forward-backward produced an empty transition")
            xi /= total
            expected_switches += float(xi[0, 1] + xi[1, 0])
            transition_count += 1.0
    return (
        log_likelihood,
        expected_switches,
        transition_count,
        expected_matches,
        observation_count,
    )


class LearnedSymmetricHMM:
    """Unsupervised two-state symmetric HMM learned from cue episodes only."""

    def __init__(
        self,
        *,
        max_iter: int = 100,
        tol: float = 1e-8,
        initial_hazards: Sequence[float] = (0.02, 0.1, 0.25),
        initial_reliabilities: Sequence[float] = (0.6, 0.8, 0.95),
        min_probability: float = 1e-6,
        pseudocount: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.max_iter = _validated_integer(max_iter, name="max_iter", minimum=1)
        self.tol = _validated_real(tol, name="tol", minimum=0.0)
        self.min_probability = _validated_real(
            min_probability,
            name="min_probability",
            minimum=0.0,
            maximum=0.25,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )
        self.seed = _validated_integer(seed, name="seed")
        self.pseudocount = _validated_real(pseudocount, name="pseudocount", minimum=0.0)
        if isinstance(initial_hazards, (str, bytes)) or isinstance(
            initial_reliabilities, (str, bytes)
        ):
            raise TypeError("HMM initializations must be numeric sequences")
        self.initial_hazards = tuple(
            _validated_real(
                value,
                name="initial_hazard",
                minimum=self.min_probability,
                maximum=0.5 - self.min_probability,
            )
            for value in initial_hazards
        )
        self.initial_reliabilities = tuple(
            _validated_real(
                value,
                name="initial_reliability",
                minimum=0.5 + self.min_probability,
                maximum=1.0 - self.min_probability,
            )
            for value in initial_reliabilities
        )
        if not self.initial_hazards or not self.initial_reliabilities:
            raise ValueError("at least one HMM initialization is required")
        self._fitted = False

    def fit(self, observations: object) -> "LearnedSymmetricHMM":
        view = _observation_view(observations)
        transition_count = sum(
            episode.stop - episode.start - 1 for episode in view.episode_slices
        )
        if transition_count < 1:
            raise ValueError(
                "HMM fitting requires at least one within-episode transition"
            )

        best: tuple[float, float, float, tuple[float, ...], bool] | None = None
        for initial_hazard in self.initial_hazards:
            for initial_reliability in self.initial_reliabilities:
                hazard = initial_hazard
                reliability = initial_reliability
                history: list[float] = []
                converged = False
                for _ in range(self.max_iter):
                    statistics = _expectation_statistics(view, hazard, reliability)
                    current_ll, switches, transitions, matches, observations_n = (
                        statistics
                    )
                    if not np.isfinite(current_ll):
                        break
                    if not history:
                        history.append(float(current_ll))
                    candidate_hazard = float(
                        np.clip(
                            (switches + self.pseudocount)
                            / (transitions + 2.0 * self.pseudocount),
                            self.min_probability,
                            0.5 - self.min_probability,
                        )
                    )
                    candidate_reliability = float(
                        np.clip(
                            (matches + self.pseudocount)
                            / (observations_n + 2.0 * self.pseudocount),
                            0.5 + self.min_probability,
                            1.0 - self.min_probability,
                        )
                    )
                    candidate_ll = _expectation_statistics(
                        view, candidate_hazard, candidate_reliability
                    )[0]
                    if candidate_ll < current_ll - 1e-7 * (1.0 + abs(current_ll)):
                        raise RuntimeError("HMM EM likelihood decreased unexpectedly")
                    history.append(float(candidate_ll))
                    hazard = candidate_hazard
                    reliability = candidate_reliability
                    if abs(candidate_ll - current_ll) <= self.tol * (
                        1.0 + abs(current_ll)
                    ):
                        converged = True
                        break
                if not history:
                    continue
                candidate = (
                    history[-1],
                    hazard,
                    reliability,
                    tuple(history),
                    converged,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            raise RuntimeError("every HMM initialization had zero cue likelihood")

        self.context_hazard_ = float(best[1])
        self.cue_reliability_ = float(best[2])
        self.log_likelihood_history_ = best[3]
        self.converged_ = bool(best[4])
        self.fit_trial_ids_ = view.trial_ids.copy()
        self.fit_episode_ids_ = np.unique(view.episode_ids)
        self.fit_observation_fingerprint_ = view.fingerprint
        self.fit_accessed_true_context_ = False
        self._fitted = True
        return self

    def predict(self, observations: object) -> GatePrediction:
        if not self._fitted:
            raise RuntimeError("LearnedSymmetricHMM must be fit before prediction")
        view = _observation_view(observations)
        beliefs = _causal_filter(
            view,
            transition=_transition_matrix(self.context_hazard_),
            emission=_emission_matrix(self.cue_reliability_),
        )
        return _prediction(
            view,
            beliefs,
            gate_name="learned_symmetric_hmm",
            fit_trial_ids=self.fit_trial_ids_,
            fit_episode_ids=self.fit_episode_ids_,
            parameters=(
                ("estimated_context_hazard", self.context_hazard_),
                ("estimated_cue_reliability", self.cue_reliability_),
                ("fit_converged", self.converged_),
                ("pseudocount", self.pseudocount),
                ("fit_observation_fingerprint", self.fit_observation_fingerprint_),
                ("seed", self.seed),
            ),
        )

    def audit_metadata(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("LearnedSymmetricHMM is not fit")
        return {
            "gate_fit_accessed_true_context": False,
            "gate_test_accessed_true_context": False,
            "fit_trial_ids": self.fit_trial_ids_.tolist(),
            "fit_episode_ids": self.fit_episode_ids_.tolist(),
            "fit_observation_fingerprint": self.fit_observation_fingerprint_,
            "estimated_context_hazard": self.context_hazard_,
            "estimated_cue_reliability": self.cue_reliability_,
            "pseudocount": self.pseudocount,
            "em_converged": self.converged_,
            "em_iterations": len(self.log_likelihood_history_) - 1,
        }


def _causal_hebbian_moment_anchor(
    view: _ObservationView,
    *,
    z_threshold: float = 3.0,
    probability_floor: float = 1e-6,
) -> tuple[float, float, float, float, float, bool]:
    """Estimate symmetric HMM parameters from causal lag-1/lag-2 products.

    For signed cues ``x``, the symmetric binary HMM obeys
    ``E[x_t x_(t-k)] = (2q-1)^2 (1-2h)^k``.  Both products can be accumulated
    online with one- and two-trial eligibility traces.  When the lag-2 signal
    is not distinguishable from sampling noise, the honest estimate is a
    neutral gate rather than an overconfident extrapolation.
    """

    max_lag = 5
    lag_sums = np.zeros(max_lag, dtype=float)
    lag_counts = np.zeros(max_lag, dtype=int)
    signed = 2.0 * view.cues.astype(float) - 1.0
    for episode in view.episode_slices:
        values = signed[episode]
        for lag in range(1, min(max_lag, values.size - 1) + 1):
            lag_sums[lag - 1] += float(np.dot(values[lag:], values[:-lag]))
            lag_counts[lag - 1] += values.size - lag
    if lag_counts[1] < 1:
        raise ValueError("MD moment anchor requires at least one two-trial cue lag")
    valid = lag_counts > 0
    correlations = lag_sums[valid] / lag_counts[valid]
    lag1 = float(correlations[0])
    lag2 = float(correlations[1])
    standard_error = np.sqrt(max(1.0 - lag1**2, probability_floor) / lag_counts[0])
    z_score = lag1 / standard_error
    identifiable = bool(lag1 > 0.0 and z_score >= z_threshold)
    if not identifiable:
        return (
            0.5 - probability_floor,
            0.5 + probability_floor,
            lag1,
            lag2,
            z_score,
            False,
        )
    preceding = correlations[:-1]
    following = correlations[1:]
    persistence = float(
        np.clip(
            np.dot(preceding, following)
            / max(float(np.dot(preceding, preceding)), probability_floor),
            probability_floor,
            1.0 - probability_floor,
        )
    )
    powers = persistence ** np.arange(1, correlations.size + 1)
    signal_squared = float(
        np.clip(
            np.dot(correlations, powers) / np.dot(powers, powers),
            0.0,
            1.0,
        )
    )
    hazard = float(
        np.clip(
            0.5 * (1.0 - persistence),
            probability_floor,
            0.5 - probability_floor,
        )
    )
    reliability = float(
        np.clip(
            0.5 * (1.0 + np.sqrt(signal_squared)),
            0.5 + probability_floor,
            1.0 - probability_floor,
        )
    )
    return hazard, reliability, lag1, lag2, z_score, True


class MDRecurrentBeliefGate:
    """Cue-only recurrent belief circuit with causal local soft-count learning."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.02,
        inverse_temperature: float = 1.0,
        pseudocount: float = 0.01,
        n_passes: int = 1,
        initial_persistence: float = 0.9,
        initial_reliability: float = 0.75,
        recurrent_smoothing: float | None = None,
        seed: int = 0,
    ) -> None:
        self.learning_rate = _validated_real(
            learning_rate,
            name="learning_rate",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        self.inverse_temperature = _validated_real(
            inverse_temperature,
            name="inverse_temperature",
            minimum=0.0,
            minimum_inclusive=False,
        )
        self.pseudocount = _validated_real(
            pseudocount,
            name="pseudocount",
            minimum=0.0,
            minimum_inclusive=False,
        )
        self.n_passes = _validated_integer(n_passes, name="n_passes", minimum=1)
        persistence = (
            initial_persistence
            if recurrent_smoothing is None
            else _validated_real(
                recurrent_smoothing,
                name="recurrent_smoothing",
                minimum=0.5,
                maximum=1.0,
                maximum_inclusive=False,
            )
        )
        self.initial_persistence = _validated_real(
            persistence,
            name="initial_persistence",
            minimum=0.5,
            maximum=1.0,
            maximum_inclusive=False,
        )
        self.initial_reliability = _validated_real(
            initial_reliability,
            name="initial_reliability",
            minimum=0.5,
            maximum=1.0,
        )
        self.seed = _validated_integer(seed, name="seed")
        self.recurrent_smoothing = self.initial_persistence
        self._fitted = False

    def fit(self, observations: object) -> "MDRecurrentBeliefGate":
        view = _observation_view(observations)
        hazard = 1.0 - self.initial_persistence
        reliability = self.initial_reliability
        transition = _transition_matrix(hazard)
        emission = _emission_matrix(reliability)
        # Symmetric Beta pseudo-counts anchor state labels to cue semantics while
        # keeping both HMM parameters learnable from causal observations alone.
        switch_count = self.pseudocount * hazard
        stay_count = self.pseudocount * (1.0 - hazard)
        match_count = self.pseudocount * reliability
        mismatch_count = self.pseudocount * (1.0 - reliability)
        total_update_l1 = 0.0
        eta = self.learning_rate
        probability_floor = 1e-6

        for _ in range(self.n_passes):
            for episode in view.episode_slices:
                previous_belief: Array | None = None
                episode_switches = 0.0
                episode_transitions = 0.0
                episode_matches = 0.0
                episode_observations = 0.0
                for index in range(episode.start, episode.stop):
                    cue = int(view.cues[index])
                    tempered_emission = emission[:, cue] ** self.inverse_temperature
                    if previous_belief is None:
                        belief = 0.5 * tempered_emission
                        normalizer = float(belief.sum())
                        if normalizer <= 0.0:
                            raise ValueError(
                                "MD belief update encountered a zero-probability cue"
                            )
                        belief /= normalizer
                    else:
                        # Causal two-slice posterior:
                        # p(z[t-1], z[t] | y[:t]) is a local eligibility term
                        # formed from the previous belief, recurrent transition,
                        # and current cue likelihood.  Summing over the previous
                        # state yields the current filtered belief.
                        joint = (
                            previous_belief[:, None]
                            * transition
                            * tempered_emission[None, :]
                        )
                        normalizer = float(joint.sum())
                        if normalizer <= 0.0:
                            raise ValueError(
                                "MD two-slice update encountered an empty posterior"
                            )
                        joint /= normalizer
                        belief = joint.sum(axis=0)
                        episode_switches += float(joint[0, 1] + joint[1, 0])
                        episode_transitions += 1.0
                    episode_matches += float(belief[cue])
                    episode_observations += 1.0
                    previous_belief = belief

                switch_count += episode_switches
                stay_count += episode_transitions - episode_switches
                match_count += episode_matches
                mismatch_count += episode_observations - episode_matches
                target_hazard = float(
                    np.clip(
                        switch_count / (switch_count + stay_count),
                        probability_floor,
                        0.5 - probability_floor,
                    )
                )
                target_reliability = float(
                    np.clip(
                        match_count / (match_count + mismatch_count),
                        0.5 + probability_floor,
                        1.0 - probability_floor,
                    )
                )
                old_transition = transition
                old_emission = emission
                hazard = (1.0 - eta) * hazard + eta * target_hazard
                reliability = (1.0 - eta) * reliability + eta * target_reliability
                transition = _transition_matrix(hazard)
                emission = _emission_matrix(reliability)
                total_update_l1 += float(
                    np.abs(transition - old_transition).sum()
                    + np.abs(emission - old_emission).sum()
                )

        (
            moment_hazard,
            moment_reliability,
            self.cue_lag1_correlation_,
            self.cue_lag2_correlation_,
            self.cue_signal_z_score_,
            self.moment_anchor_identifiable_,
        ) = _causal_hebbian_moment_anchor(view)
        self.two_slice_hazard_ = float(hazard)
        self.two_slice_reliability_ = float(reliability)
        self.moment_hazard_ = float(moment_hazard)
        self.moment_reliability_ = float(moment_reliability)
        self.moment_anchor_weight_ = 0.8 if self.moment_anchor_identifiable_ else 1.0
        if self.moment_anchor_identifiable_:
            final_hazard = (
                1.0 - self.moment_anchor_weight_
            ) * hazard + self.moment_anchor_weight_ * moment_hazard
            final_reliability = (
                1.0 - self.moment_anchor_weight_
            ) * reliability + self.moment_anchor_weight_ * moment_reliability
        else:
            # Weak cue autocorrelation cannot identify h and q separately.  A
            # neutral gate is the conservative local estimate and avoids
            # turning initialization bias into confident context predictions.
            final_hazard = moment_hazard
            final_reliability = moment_reliability
        anchored_transition = _transition_matrix(final_hazard)
        anchored_emission = _emission_matrix(final_reliability)
        total_update_l1 += float(
            np.abs(anchored_transition - transition).sum()
            + np.abs(anchored_emission - emission).sum()
        )
        transition = anchored_transition
        emission = anchored_emission

        self.transition_ = transition.copy()
        self.emission_ = emission.copy()
        self.transition_.setflags(write=False)
        self.emission_.setflags(write=False)
        self.local_update_l1_ = total_update_l1
        self.fit_trial_ids_ = view.trial_ids.copy()
        self.fit_episode_ids_ = np.unique(view.episode_ids)
        self.fit_observation_fingerprint_ = view.fingerprint
        self.fit_accessed_true_context_ = False
        self._fitted = True
        return self

    def _prediction_parameters(self) -> tuple[tuple[str, Any], ...]:
        """Return immutable fit provenance shared by both timing interfaces."""

        return (
            ("learning_rate", self.learning_rate),
            ("inverse_temperature", self.inverse_temperature),
            ("pseudocount", self.pseudocount),
            ("recurrent_smoothing", self.recurrent_smoothing),
            ("n_passes", self.n_passes),
            ("local_update_l1", self.local_update_l1_),
            ("estimated_context_hazard", float(self.transition_[0, 1])),
            ("estimated_cue_reliability", float(self.emission_[0, 0])),
            (
                "local_update_rule",
                "causal_two_slice_with_hebbian_moment_shrinkage",
            ),
            ("cue_lag1_correlation", self.cue_lag1_correlation_),
            ("cue_lag2_correlation", self.cue_lag2_correlation_),
            ("cue_signal_z_score", self.cue_signal_z_score_),
            ("moment_anchor_identifiable", self.moment_anchor_identifiable_),
            ("moment_anchor_weight", self.moment_anchor_weight_),
            ("two_slice_hazard", self.two_slice_hazard_),
            ("two_slice_reliability", self.two_slice_reliability_),
            ("moment_hazard", self.moment_hazard_),
            ("moment_reliability", self.moment_reliability_),
            ("fit_observation_fingerprint", self.fit_observation_fingerprint_),
            ("seed", self.seed),
        )

    def predict(self, observations: object) -> GatePrediction:
        if not self._fitted:
            raise RuntimeError("MDRecurrentBeliefGate must be fit before prediction")
        view = _observation_view(observations)
        beliefs = _causal_filter(
            view,
            transition=self.transition_,
            emission=self.emission_,
            inverse_temperature=self.inverse_temperature,
        )
        return _prediction(
            view,
            beliefs,
            gate_name="md_recurrent_belief",
            fit_trial_ids=self.fit_trial_ids_,
            fit_episode_ids=self.fit_episode_ids_,
            parameters=self._prediction_parameters(),
        )

    def predict_prior(self, observations: object) -> GatePrediction:
        """Predict context before the current trial's cue is available.

        The returned row for trial ``t`` is ``p(z_t | y_{<t})``.  Cue ``y_t``
        is consumed only after that row is frozen and can first affect trial
        ``t+1``.  This is the timing-safe interface for pre-stimulus analyses;
        :meth:`predict` intentionally retains its existing current-cue
        posterior semantics.
        """

        if not self._fitted:
            raise RuntimeError("MDRecurrentBeliefGate must be fit before prediction")
        view = _observation_view(observations)
        beliefs = _causal_predictive_prior(
            view,
            transition=self.transition_,
            emission=self.emission_,
            inverse_temperature=self.inverse_temperature,
        )
        return _prediction(
            view,
            beliefs,
            gate_name="md_recurrent_belief_predictive_prior",
            fit_trial_ids=self.fit_trial_ids_,
            fit_episode_ids=self.fit_episode_ids_,
            parameters=self._prediction_parameters()
            + (
                ("belief_timing", "predictive_prior_before_current_cue"),
                ("observation_window", "strictly_before_current_trial"),
                ("current_cue_accessed_for_same_trial", False),
                ("future_cues_accessed", False),
            ),
        )

    def predictive_prior(self, observations: object) -> GatePrediction:
        """Compatibility alias for :meth:`predict_prior`."""

        return self.predict_prior(observations)

    def audit_metadata(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("MDRecurrentBeliefGate is not fit")
        return {
            "gate_fit_accessed_true_context": False,
            "gate_test_accessed_true_context": False,
            "fit_trial_ids": self.fit_trial_ids_.tolist(),
            "fit_episode_ids": self.fit_episode_ids_.tolist(),
            "fit_observation_fingerprint": self.fit_observation_fingerprint_,
            "local_update_l1": self.local_update_l1_,
            "estimated_context_hazard": float(self.transition_[0, 1]),
            "estimated_cue_reliability": float(self.emission_[0, 0]),
            "local_update_rule": "causal_two_slice_with_hebbian_moment_shrinkage",
            "cue_lag1_correlation": self.cue_lag1_correlation_,
            "cue_lag2_correlation": self.cue_lag2_correlation_,
            "cue_signal_z_score": self.cue_signal_z_score_,
            "moment_anchor_identifiable": self.moment_anchor_identifiable_,
            "moment_anchor_weight": self.moment_anchor_weight_,
            "two_slice_hazard": self.two_slice_hazard_,
            "two_slice_reliability": self.two_slice_reliability_,
            "moment_hazard": self.moment_hazard_,
            "moment_reliability": self.moment_reliability_,
        }


def _causal_cue_features(view: _ObservationView, decays: tuple[float, ...]) -> Array:
    features = np.empty((view.cues.size, 1 + len(decays)), dtype=float)
    signed = 2.0 * view.cues.astype(float) - 1.0
    for episode in view.episode_slices:
        traces = np.zeros(len(decays), dtype=float)
        decay_values = np.asarray(decays, dtype=float)
        for index in range(episode.start, episode.stop):
            traces = decay_values * traces + (1.0 - decay_values) * signed[index]
            features[index, 0] = signed[index]
            features[index, 1:] = traces
    return features


class SupervisedCueGate:
    """Explicit supervised upper bound with observation-only prediction."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        trace_decays: Sequence[float] = (0.5, 0.8, 0.95),
        max_iter: int = 1000,
        seed: int = 0,
    ) -> None:
        self.C = _validated_real(C, name="C", minimum=0.0, minimum_inclusive=False)
        if isinstance(trace_decays, (str, bytes)):
            raise TypeError("trace_decays must be a numeric sequence")
        self.trace_decays = tuple(
            _validated_real(
                value,
                name="trace_decay",
                minimum=0.0,
                maximum=1.0,
                maximum_inclusive=False,
            )
            for value in trace_decays
        )
        if not self.trace_decays or len(set(self.trace_decays)) != len(
            self.trace_decays
        ):
            raise ValueError("trace_decays must be non-empty and unique")
        self.max_iter = _validated_integer(max_iter, name="max_iter", minimum=1)
        self.seed = _validated_integer(seed, name="seed")
        self._fitted = False

    def fit_supervised(
        self, observations: object, hidden_states: object
    ) -> "SupervisedCueGate":
        view = _observation_view(observations)
        labels = _readonly_int_vector(
            hidden_states, name="hidden_states", length=view.cues.size
        )
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("hidden_states must be binary")
        if np.unique(labels).size != 2:
            raise ValueError("supervised fitting requires both hidden states")
        features = _causal_cue_features(view, self.trace_decays)
        classifier = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        classifier.fit(features, labels)
        if not np.array_equal(classifier.classes_, np.array([0, 1])):
            raise RuntimeError("supervised classifier did not retain binary labels")
        self.classifier_ = classifier
        self.fit_trial_ids_ = view.trial_ids.copy()
        self.fit_episode_ids_ = np.unique(view.episode_ids)
        self.fit_observation_fingerprint_ = view.fingerprint
        self.fit_accessed_true_context_ = True
        self._fitted = True
        return self

    def predict(self, observations: object) -> GatePrediction:
        if not self._fitted:
            raise RuntimeError("SupervisedCueGate must be fit before prediction")
        view = _observation_view(observations)
        features = _causal_cue_features(view, self.trace_decays)
        beliefs = self.classifier_.predict_proba(features)
        return _prediction(
            view,
            beliefs,
            gate_name="supervised_cue_upper_bound",
            fit_trial_ids=self.fit_trial_ids_,
            fit_episode_ids=self.fit_episode_ids_,
            fit_accessed_true_context=True,
            parameters=(
                ("C", self.C),
                ("feature_count", features.shape[1]),
                ("fit_observation_fingerprint", self.fit_observation_fingerprint_),
                ("seed", self.seed),
            ),
        )

    def audit_metadata(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("SupervisedCueGate is not fit")
        return {
            "evidence_role": "supervised_upper_bound",
            "gate_fit_accessed_true_context": True,
            "gate_test_accessed_true_context": False,
            "fit_trial_ids": self.fit_trial_ids_.tolist(),
            "fit_episode_ids": self.fit_episode_ids_.tolist(),
            "fit_observation_fingerprint": self.fit_observation_fingerprint_,
        }


class NoGate:
    """Neutral no-context control."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = _validated_integer(seed, name="seed")

    def predict(self, observations: object) -> GatePrediction:
        view = _observation_view(observations)
        return _prediction(
            view,
            np.full((view.cues.size, 2), 0.5),
            gate_name="no_gate",
            parameters=(("seed", self.seed),),
        )


def _intervention_base(prediction: GatePrediction) -> str:
    return prediction.base_prediction_fingerprint or prediction.fingerprint


def neutral_clamp(prediction: GatePrediction) -> GatePrediction:
    """Clamp every trial to the neutral belief without refitting the gate."""

    if not isinstance(prediction, GatePrediction):
        raise TypeError("prediction must be a GatePrediction")
    return replace(
        prediction,
        beliefs=np.full_like(prediction.beliefs, 0.5),
        intervention="neutral_clamp",
        source_trial_ids=np.full(prediction.trial_ids.size, -1, dtype=int),
        source_episode_ids=np.full(prediction.episode_ids.size, -1, dtype=int),
        base_prediction_fingerprint=_intervention_base(prediction),
    )


def _prediction_episode_slices(prediction: GatePrediction) -> tuple[slice, ...]:
    starts = np.ones(prediction.episode_ids.size, dtype=bool)
    if starts.size > 1:
        starts[1:] = prediction.episode_ids[1:] != prediction.episode_ids[:-1]
    ordered = prediction.episode_ids[starts]
    if np.unique(ordered).size != ordered.size:
        raise ValueError("prediction episodes must be contiguous")
    indices = np.flatnonzero(starts)
    stops = np.concatenate([indices[1:], np.array([starts.size])])
    return tuple(
        slice(int(start), int(stop)) for start, stop in zip(indices, stops, strict=True)
    )


def episode_delay(prediction: GatePrediction, delay_trials: int) -> GatePrediction:
    """Delay beliefs within each episode, padding only with neutral beliefs."""

    if not isinstance(prediction, GatePrediction):
        raise TypeError("prediction must be a GatePrediction")
    delay = _validated_integer(delay_trials, name="delay_trials")
    beliefs = np.full_like(prediction.beliefs, 0.5)
    source_trial_ids = np.full(prediction.trial_ids.size, -1, dtype=int)
    source_episode_ids = np.full(prediction.episode_ids.size, -1, dtype=int)
    for episode in _prediction_episode_slices(prediction):
        length = episode.stop - episode.start
        if delay < length:
            destination = slice(episode.start + delay, episode.stop)
            source = slice(episode.start, episode.stop - delay)
            beliefs[destination] = prediction.beliefs[source]
            source_trial_ids[destination] = prediction.source_trial_ids[source]
            source_episode_ids[destination] = prediction.source_episode_ids[source]
    return replace(
        prediction,
        beliefs=beliefs,
        intervention=f"delay_{delay}",
        source_trial_ids=source_trial_ids,
        source_episode_ids=source_episode_ids,
        base_prediction_fingerprint=_intervention_base(prediction),
    )


def deranged_trajectory_shuffle(
    prediction: GatePrediction, *, seed: int
) -> GatePrediction:
    """Derange complete equal-length trajectories across test episodes."""

    if not isinstance(prediction, GatePrediction):
        raise TypeError("prediction must be a GatePrediction")
    seed = _validated_integer(seed, name="seed")
    episodes = _prediction_episode_slices(prediction)
    if len(episodes) < 2:
        raise ValueError("trajectory shuffle requires at least two episodes")
    lengths = np.asarray([item.stop - item.start for item in episodes], dtype=int)
    beliefs = np.empty_like(prediction.beliefs)
    source_trial_ids = np.empty_like(prediction.trial_ids)
    source_episode_ids = np.empty_like(prediction.episode_ids)
    rng = np.random.default_rng(seed)
    for length in np.unique(lengths):
        group = np.flatnonzero(lengths == length)
        if group.size < 2:
            raise ValueError(
                "every episode-length stratum needs at least two trajectories"
            )
        shift = int(rng.integers(1, group.size))
        source_group = np.roll(group, -shift)
        for destination_index, source_index in zip(group, source_group, strict=True):
            destination = episodes[int(destination_index)]
            source = episodes[int(source_index)]
            beliefs[destination] = prediction.beliefs[source]
            source_trial_ids[destination] = prediction.source_trial_ids[source]
            source_episode_ids[destination] = prediction.source_episode_ids[source]
            destination_episode = prediction.episode_ids[destination.start]
            if np.any(source_episode_ids[destination] == destination_episode):
                raise RuntimeError("trajectory shuffle produced an episode fixed point")
    return replace(
        prediction,
        beliefs=beliefs,
        intervention="trajectory_shuffle",
        source_trial_ids=source_trial_ids,
        source_episode_ids=source_episode_ids,
        base_prediction_fingerprint=_intervention_base(prediction),
    )


# Explicit semantic aliases used by training code and tests.
clamp_belief_neutral = neutral_clamp
delay_belief_by_episode = episode_delay
shuffle_belief_trajectories = deranged_trajectory_shuffle


__all__ = [
    "GatePrediction",
    "LearnedSymmetricHMM",
    "MDRecurrentBeliefGate",
    "NoGate",
    "OracleBayesianFilter",
    "SupervisedCueGate",
    "clamp_belief_neutral",
    "delay_belief_by_episode",
    "deranged_trajectory_shuffle",
    "episode_delay",
    "neutral_clamp",
    "shuffle_belief_trajectories",
]
