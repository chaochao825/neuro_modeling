"""Claim-ineligible diagnostics for memory-update algebra and reachability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _probabilities(value: ArrayLike, *, name: str = "probabilities") -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError(f"{name} must have shape [frame, class>=2]")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(np.sum(result, axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{name} rows must sum to one")
    return result


def _aligned_strings(value: ArrayLike, *, n_frames: int, name: str) -> NDArray[np.str_]:
    result = np.asarray(value, dtype=str)
    if result.shape != (n_frames,) or np.any(np.char.str_len(result) == 0):
        raise ValueError(f"{name} must be an aligned non-empty string vector")
    return result


def _aligned_flags(value: ArrayLike | None, *, n_frames: int) -> NDArray[np.bool_]:
    if value is None:
        return np.zeros(n_frames, dtype=np.bool_)
    result = np.asarray(value, dtype=np.bool_)
    if result.shape != (n_frames,):
        raise ValueError("reset flags must align with frames")
    return result


def _unit_interval(value: float, *, name: str, closed: bool = True) -> float:
    result = float(value)
    upper_ok = result <= 1.0 if closed else result < 1.0
    if not np.isfinite(result) or result < 0.0 or not upper_ok:
        boundary = "[0, 1]" if closed else "[0, 1)"
        raise ValueError(f"{name} must lie in {boundary}")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class BeliefTrace:
    """Causal class-belief tape."""

    probabilities: FloatArray
    predictions: IntArray

    def __post_init__(self) -> None:
        probabilities = _probabilities(self.probabilities)
        predictions = np.asarray(self.predictions, dtype=np.int64)
        if predictions.shape != (probabilities.shape[0],):
            raise ValueError("predictions must align with probabilities")
        if np.any(predictions < 0) or np.any(predictions >= probabilities.shape[1]):
            raise ValueError("predictions fall outside the class range")
        probabilities = np.array(probabilities, copy=True)
        predictions = np.array(predictions, copy=True)
        probabilities.setflags(write=False)
        predictions.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "predictions", predictions)


@dataclass(frozen=True, slots=True, eq=False)
class OracleWriteTrace:
    """Label-revealed one-step write target for a fixed causal state tape."""

    targets: NDArray[np.bool_]
    log_memory_mass: FloatArray
    keep_nll: FloatArray
    write_nll: FloatArray

    def __post_init__(self) -> None:
        targets = np.asarray(self.targets, dtype=np.bool_)
        arrays = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (self.log_memory_mass, self.keep_nll, self.write_nll)
        )
        if targets.ndim != 1 or targets.size == 0:
            raise ValueError("targets must be a non-empty vector")
        if any(value.shape != targets.shape for value in arrays):
            raise ValueError("oracle-write arrays must align")
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("oracle-write arrays must be finite")
        for name, value in (
            ("targets", targets),
            ("log_memory_mass", arrays[0]),
            ("keep_nll", arrays[1]),
            ("write_nll", arrays[2]),
        ):
            frozen = np.array(value, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)


def direct_alpha_filter(
    evidence_value: ArrayLike,
    *,
    stream_ids: ArrayLike,
    alpha: float,
    reset_flags: ArrayLike | None = None,
) -> BeliefTrace:
    """Filter normalized evidence with a state-invariant effective learning rate."""

    evidence = _probabilities(evidence_value, name="evidence")
    streams = _aligned_strings(
        stream_ids, n_frames=evidence.shape[0], name="stream_ids"
    )
    resets = _aligned_flags(reset_flags, n_frames=evidence.shape[0])
    learning_rate = _unit_interval(alpha, name="alpha")
    output = np.empty_like(evidence)
    state = np.full(evidence.shape[1], 1.0 / evidence.shape[1], dtype=np.float64)
    previous_stream = ""
    for index, stream_raw in enumerate(streams):
        stream = str(stream_raw)
        new_stream = stream != previous_stream
        if new_stream or resets[index]:
            state = np.array(evidence[index], copy=True)
        else:
            state = (1.0 - learning_rate) * state + learning_rate * evidence[index]
            state /= np.sum(state)
        output[index] = state
        previous_stream = stream
    return BeliefTrace(
        probabilities=output,
        predictions=np.argmax(output, axis=1).astype(np.int64),
    )


def likelihood_hmm_filter(
    log_likelihood_value: ArrayLike,
    *,
    stream_ids: ArrayLike,
    hazard: float,
    temperature: float,
    reset_flags: ArrayLike | None = None,
) -> BeliefTrace:
    """Accumulate class likelihoods with a symmetric causal HMM transition."""

    log_likelihood = np.asarray(log_likelihood_value, dtype=np.float64)
    if (
        log_likelihood.ndim != 2
        or log_likelihood.shape[0] == 0
        or log_likelihood.shape[1] < 2
        or not np.all(np.isfinite(log_likelihood))
    ):
        raise ValueError("log_likelihood must have shape [frame, class>=2]")
    streams = _aligned_strings(
        stream_ids, n_frames=log_likelihood.shape[0], name="stream_ids"
    )
    resets = _aligned_flags(reset_flags, n_frames=log_likelihood.shape[0])
    transition_hazard = _unit_interval(hazard, name="hazard", closed=False)
    evidence_temperature = float(temperature)
    if not np.isfinite(evidence_temperature) or evidence_temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    n_classes = log_likelihood.shape[1]
    uniform = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    state = uniform.copy()
    output = np.empty_like(log_likelihood)
    previous_stream = ""
    for index, stream_raw in enumerate(streams):
        stream = str(stream_raw)
        if stream != previous_stream or resets[index]:
            prior = uniform
        else:
            prior = (1.0 - transition_hazard) * state + transition_hazard * uniform
        log_state = np.log(np.maximum(prior, 1e-300)) + (
            log_likelihood[index] / evidence_temperature
        )
        log_state -= logsumexp(log_state)
        state = np.exp(log_state)
        output[index] = state
        previous_stream = stream
    return BeliefTrace(
        probabilities=output,
        predictions=np.argmax(output, axis=1).astype(np.int64),
    )


def source_video_belief_metrics(
    probabilities_value: ArrayLike,
    labels_value: ArrayLike,
    *,
    source_video_ids: ArrayLike,
    switch_flags: ArrayLike,
    post_switch_window: int,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Return one NLL/accuracy row per source video."""

    probabilities = _probabilities(probabilities_value)
    labels = np.asarray(labels_value)
    if labels.dtype.kind not in {"i", "u"} or labels.shape != (
        probabilities.shape[0],
    ):
        raise ValueError("labels must be an aligned integer vector")
    labels = np.asarray(labels, dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels fall outside the class range")
    sources = _aligned_strings(
        source_video_ids,
        n_frames=probabilities.shape[0],
        name="source_video_ids",
    )
    switches = _aligned_flags(switch_flags, n_frames=probabilities.shape[0])
    if isinstance(post_switch_window, bool) or int(post_switch_window) < 1:
        raise ValueError("post_switch_window must be a positive integer")
    post_switch_window = int(post_switch_window)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    frame_nll = -np.log(
        np.maximum(probabilities[np.arange(len(labels)), labels], epsilon)
    )
    predictions = np.argmax(probabilities, axis=1)
    rows: list[dict[str, object]] = []
    for source in dict.fromkeys(sources.tolist()):
        indices = np.flatnonzero(sources == source)
        if indices.size == 0 or np.any(np.diff(indices) != 1):
            raise ValueError("source videos must occupy contiguous frame blocks")
        local_switch = bool(switches[indices[0]])
        post_indices = indices[:post_switch_window] if local_switch else np.array([])
        rows.append(
            {
                "source_video_id": str(source),
                "n_frames": int(indices.size),
                "accuracy": float(np.mean(predictions[indices] == labels[indices])),
                "nll": float(np.mean(frame_nll[indices])),
                "post_switch_nll": (
                    float(np.mean(frame_nll[post_indices]))
                    if post_indices.size
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_video_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    required = {"source_video_id", "n_frames", "accuracy", "nll", "post_switch_nll"}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError("video metric frame has an invalid schema")
    if frame["source_video_id"].duplicated().any():
        raise ValueError("source videos must be unique")
    post = frame["post_switch_nll"].dropna()
    return {
        "n_videos": int(len(frame)),
        "n_frames": int(frame["n_frames"].sum()),
        "video_equal_accuracy": float(frame["accuracy"].mean()),
        "video_equal_nll": float(frame["nll"].mean()),
        "video_equal_post_switch_nll": (
            float(post.mean()) if not post.empty else float("nan")
        ),
    }


def oracle_write_targets(
    evidence_value: ArrayLike,
    labels_value: ArrayLike,
    *,
    stream_ids: ArrayLike,
    retention: float,
    epsilon: float = 1e-12,
) -> OracleWriteTrace:
    """Construct a label-revealed keep-versus-write target on a fixed tape."""

    evidence = _probabilities(evidence_value, name="evidence")
    labels = np.asarray(labels_value)
    if labels.dtype.kind not in {"i", "u"} or labels.shape != (evidence.shape[0],):
        raise ValueError("labels must be an aligned integer vector")
    labels = np.asarray(labels, dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= evidence.shape[1]):
        raise ValueError("labels fall outside the class range")
    streams = _aligned_strings(
        stream_ids, n_frames=evidence.shape[0], name="stream_ids"
    )
    keep = _unit_interval(retention, name="retention")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    targets = np.empty(evidence.shape[0], dtype=np.bool_)
    log_mass = np.empty(evidence.shape[0], dtype=np.float64)
    keep_nll = np.empty(evidence.shape[0], dtype=np.float64)
    write_nll = np.empty(evidence.shape[0], dtype=np.float64)
    state = np.zeros(evidence.shape[1], dtype=np.float64)
    mass = 0.0
    previous_stream = ""
    uniform = np.full(evidence.shape[1], 1.0 / evidence.shape[1])
    for index, stream_raw in enumerate(streams):
        stream = str(stream_raw)
        if stream != previous_stream:
            state.fill(0.0)
            mass = 0.0
        previous_probability = state / mass if mass > 0.0 else uniform
        class_id = int(labels[index])
        keep_nll[index] = -np.log(max(previous_probability[class_id], epsilon))
        write_nll[index] = -np.log(max(evidence[index, class_id], epsilon))
        targets[index] = bool(mass == 0.0 or write_nll[index] < keep_nll[index])
        log_mass[index] = np.log1p(mass)
        state = keep * state + evidence[index]
        mass = keep * mass + 1.0
        previous_stream = stream
    return OracleWriteTrace(
        targets=targets,
        log_memory_mass=log_mass,
        keep_nll=keep_nll,
        write_nll=write_nll,
    )


def fit_oracle_write_probe(
    features_value: ArrayLike, targets_value: ArrayLike, *, seed: int
) -> Pipeline:
    """Fit a deterministic train-only logistic probe."""

    features = np.asarray(features_value, dtype=np.float64)
    targets = np.asarray(targets_value, dtype=np.bool_)
    if features.ndim != 2 or features.shape[0] < 4 or features.shape[1] < 1:
        raise ValueError("features must have shape [sample>=4, dimension>=1]")
    if targets.shape != (features.shape[0],) or np.unique(targets).size != 2:
        raise ValueError("targets must align and contain both classes")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must be finite")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=int(seed),
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(features, targets.astype(np.int64))
    return model


def grouped_binary_metrics(
    targets_value: ArrayLike, scores_value: ArrayLike, *, group_ids: ArrayLike
) -> dict[str, float | int]:
    """Average Brier and within-video AUC without treating frames as replicates."""

    targets = np.asarray(targets_value, dtype=np.bool_)
    scores = np.asarray(scores_value, dtype=np.float64)
    if targets.ndim != 1 or targets.size == 0 or scores.shape != targets.shape:
        raise ValueError("targets and scores must be aligned vectors")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must be finite probabilities")
    groups = _aligned_strings(group_ids, n_frames=targets.size, name="group_ids")
    aucs: list[float] = []
    briers: list[float] = []
    for group in dict.fromkeys(groups.tolist()):
        mask = groups == group
        group_targets = targets[mask].astype(np.int64)
        group_scores = scores[mask]
        briers.append(float(np.mean((group_scores - group_targets) ** 2)))
        if np.unique(group_targets).size == 2:
            aucs.append(float(roc_auc_score(group_targets, group_scores)))
    return {
        "n_groups": int(len(briers)),
        "n_auc_groups": int(len(aucs)),
        "video_equal_brier": float(np.mean(briers)),
        "video_equal_auc": float(np.mean(aucs)) if aucs else float("nan"),
    }


__all__ = [
    "BeliefTrace",
    "OracleWriteTrace",
    "direct_alpha_filter",
    "fit_oracle_write_probe",
    "grouped_binary_metrics",
    "likelihood_hmm_filter",
    "oracle_write_targets",
    "source_video_belief_metrics",
    "summarize_video_metrics",
]
