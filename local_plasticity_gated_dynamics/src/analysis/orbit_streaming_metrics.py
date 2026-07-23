"""ORBIT metrics with user-level statistical inference.

Raw frame predictions are first reduced within task/video.  The public ORBIT
leaderboard endpoint is the mean of those task-video frame accuracies.  Formal
comparisons additionally aggregate within user and treat users—not frames,
videos, or neurons—as independent units.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _paired_vectors(
    labels: ArrayLike, predictions: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(labels, dtype=np.int64)
    output = np.asarray(predictions, dtype=np.int64)
    if target.ndim != 1 or target.size == 0 or output.shape != target.shape:
        raise ValueError(
            "labels and predictions must be equally shaped non-empty vectors"
        )
    if np.any(target < 0) or np.any(output < 0):
        raise ValueError("labels and predictions must be non-negative")
    return target, output


def frame_accuracy(labels: ArrayLike, predictions: ArrayLike) -> float:
    target, output = _paired_vectors(labels, predictions)
    return float(np.mean(target == output))


def task_video_accuracy_rows(
    *,
    user_id: str,
    task_index: int,
    condition: str,
    labels: ArrayLike,
    predictions: ArrayLike,
    video_ids: ArrayLike,
    selected_actions: ArrayLike | None = None,
) -> pd.DataFrame:
    """Return one raw metric row per task/video, the public scoring atom."""

    target, output = _paired_vectors(labels, predictions)
    videos = np.asarray(video_ids, dtype=str)
    if videos.shape != target.shape or np.any(np.char.str_len(videos) == 0):
        raise ValueError("video_ids must provide one non-empty id per frame")
    actions = None
    if selected_actions is not None:
        actions = np.asarray(selected_actions, dtype=np.int64)
        if actions.shape != target.shape or np.any(actions < 0):
            raise ValueError(
                "selected_actions must align with frames and be non-negative"
            )
    rows: list[dict[str, object]] = []
    for video_id in dict.fromkeys(videos.tolist()):
        mask = videos == video_id
        row: dict[str, object] = {
            "user_id": str(user_id),
            "task_index": int(task_index),
            "video_id": str(video_id),
            "condition": str(condition),
            "n_frames": int(np.sum(mask)),
            "frame_accuracy": float(np.mean(target[mask] == output[mask])),
        }
        if actions is not None:
            values, counts = np.unique(actions[mask], return_counts=True)
            for action, count in zip(values, counts, strict=True):
                row[f"action_{int(action)}_fraction"] = float(count / np.sum(mask))
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class ActuatorHeadroom:
    per_action_accuracy: FloatArray
    best_action: int
    best_fixed_accuracy: float
    oracle_accuracy: float
    oracle_gain: float
    action_disagreement: float


def actuator_headroom(
    labels: ArrayLike, action_predictions: ArrayLike
) -> ActuatorHeadroom:
    target = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(action_predictions, dtype=np.int64)
    if target.ndim != 1 or target.size == 0:
        raise ValueError("labels must be a non-empty vector")
    if (
        predictions.ndim != 2
        or predictions.shape[0] != target.size
        or predictions.shape[1] < 2
    ):
        raise ValueError("action_predictions must have shape [frame, action>=2]")
    if np.any(target < 0) or np.any(predictions < 0):
        raise ValueError("labels and action predictions must be non-negative")
    accuracy = np.mean(predictions == target[:, None], axis=0)
    best = int(np.argmax(accuracy))
    oracle_correct = np.any(predictions == target[:, None], axis=1)
    disagreement = np.any(predictions != predictions[:, :1], axis=1)
    best_accuracy = float(accuracy[best])
    oracle_accuracy = float(np.mean(oracle_correct))
    result = ActuatorHeadroom(
        per_action_accuracy=np.array(accuracy, dtype=np.float64),
        best_action=best,
        best_fixed_accuracy=best_accuracy,
        oracle_accuracy=oracle_accuracy,
        oracle_gain=oracle_accuracy - best_accuracy,
        action_disagreement=float(np.mean(disagreement)),
    )
    result.per_action_accuracy.setflags(write=False)
    return result


def reduce_to_user_accuracy(raw_video_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "condition", "frame_accuracy"}
    if not required <= set(raw_video_rows.columns):
        raise ValueError(
            f"raw rows missing columns: {sorted(required - set(raw_video_rows.columns))}"
        )
    frame = raw_video_rows.copy()
    frame["frame_accuracy"] = pd.to_numeric(frame["frame_accuracy"], errors="raise")
    if not np.isfinite(frame["frame_accuracy"]).all():
        raise ValueError("frame_accuracy must be finite")
    return (
        frame.groupby(["user_id", "condition"], as_index=False, sort=True)[
            "frame_accuracy"
        ]
        .mean()
        .rename(columns={"frame_accuracy": "user_video_mean_accuracy"})
    )


@dataclass(frozen=True, slots=True)
class PairedUserInference:
    method: str
    comparator: str
    n_users: int
    mean_difference: float
    ci_low: float
    ci_high: float
    positive_users: int
    sign_flip_pvalue: float


def _exact_sign_flip_pvalue(differences: FloatArray) -> float:
    observed = abs(float(np.mean(differences)))
    n = differences.size
    if n <= 20:
        exceed = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            statistic = abs(float(np.mean(differences * np.asarray(signs))))
            exceed += int(statistic >= observed - 1e-15)
            total += 1
        return float(exceed / total)
    rng = np.random.default_rng(0)
    signs = rng.choice((-1.0, 1.0), size=(100_000, n))
    null = np.abs(np.mean(signs * differences[None, :], axis=1))
    return float((1 + np.sum(null >= observed)) / (null.size + 1))


def paired_user_inference(
    user_rows: pd.DataFrame,
    *,
    method: str,
    comparator: str,
    bootstrap_samples: int = 20_000,
    seed: int = 0,
) -> PairedUserInference:
    if isinstance(bootstrap_samples, bool) or int(bootstrap_samples) < 100:
        raise ValueError("bootstrap_samples must be an integer >= 100")
    required = {"user_id", "condition", "user_video_mean_accuracy"}
    if not required <= set(user_rows.columns):
        raise ValueError(
            f"user rows missing columns: {sorted(required - set(user_rows.columns))}"
        )
    wide = user_rows.pivot(
        index="user_id", columns="condition", values="user_video_mean_accuracy"
    )
    if method not in wide or comparator not in wide:
        raise ValueError("method and comparator must both occur for every paired user")
    paired = wide[[method, comparator]].dropna()
    if paired.empty:
        raise ValueError("no paired users remain")
    differences = np.asarray(paired[method] - paired[comparator], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, differences.size, size=(int(bootstrap_samples), differences.size)
    )
    boot = np.mean(differences[indices], axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return PairedUserInference(
        method=method,
        comparator=comparator,
        n_users=int(differences.size),
        mean_difference=float(np.mean(differences)),
        ci_low=float(low),
        ci_high=float(high),
        positive_users=int(np.sum(differences > 0.0)),
        sign_flip_pvalue=_exact_sign_flip_pvalue(differences),
    )


def holm_adjust(pvalues: Iterable[float]) -> FloatArray:
    """Return Holm-adjusted p-values without treating observations as tests."""

    raw = np.asarray(tuple(pvalues), dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("pvalues must be a non-empty finite vector")
    if np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("pvalues must lie in [0, 1]")
    order = np.argsort(raw)
    adjusted_sorted = np.maximum.accumulate(
        (raw.size - np.arange(raw.size)) * raw[order]
    )
    adjusted = np.empty_like(raw)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    adjusted.setflags(write=False)
    return adjusted
