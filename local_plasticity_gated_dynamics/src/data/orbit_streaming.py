"""Fail-closed ORBIT manifests and embedding episodes.

The deployable model receives labelled clean-video support embeddings and an
unlabelled causal query stream.  Query labels remain in the episode object for
evaluation, but :attr:`OrbitEmbeddingEpisode.query_observation` deliberately
returns a label-free view.  Sampling is by user/task/video and chronological
frame order is restored after random frame selection.

This module has no dependency on the upstream ORBIT implementation.  The
official user split copied into ``data/orbit_official_splits.json`` is checked
whenever a store is opened.  Raw-image feature extraction lives in
``scripts/prepare_orbit_features.py`` so model and metric tests can use small
synthetic feature stores without downloading the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal, Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
SplitName = Literal["train", "validation", "test"]

FEATURE_MANIFEST_COLUMNS = (
    "split",
    "user_id",
    "object_name",
    "video_type",
    "video_id",
    "feature_path",
    "n_frames",
    "feature_dim",
    "source_fingerprint",
)


def _readonly(value: ArrayLike, *, dtype: np.dtype | type) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _string_array(value: ArrayLike, *, name: str) -> NDArray[np.str_]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    result = np.asarray(array, dtype=str)
    if np.any(np.char.str_len(result) == 0):
        raise ValueError(f"{name} cannot contain empty strings")
    result = result.copy()
    result.setflags(write=False)
    return result


def _hash_payload(label: str, *values: object) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def derive_orbit_seed(seed: int, *parts: object) -> int:
    """Return a stable 32-bit seed independent of Python hash randomisation."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    digest = hashlib.sha256(str(int(seed)).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(repr(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:4], "little", signed=False)


def load_official_orbit_splits(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Load and validate the user-disjoint official ORBIT benchmark split."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"official ORBIT split file not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if set(payload) != {"train", "validation", "test"}:
        raise ValueError("official ORBIT split must contain train/validation/test")
    result: dict[str, tuple[str, ...]] = {}
    all_users: list[str] = []
    for split in ("train", "validation", "test"):
        values = tuple(str(value) for value in payload[split])
        if not values or len(values) != len(set(values)):
            raise ValueError(f"ORBIT {split} users must be non-empty and unique")
        result[split] = values
        all_users.extend(values)
    if len(all_users) != len(set(all_users)):
        raise ValueError("ORBIT train/validation/test users must be disjoint")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class OrbitSupportSet:
    embeddings: FloatArray
    labels: IntArray
    video_ids: NDArray[np.str_]
    frame_indices: IntArray

    def __post_init__(self) -> None:
        embeddings = np.asarray(self.embeddings, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.int64)
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        video_ids = _string_array(self.video_ids, name="support.video_ids")
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("support embeddings must have shape [sample, feature]")
        n = embeddings.shape[0]
        if (
            labels.shape != (n,)
            or frame_indices.shape != (n,)
            or video_ids.shape != (n,)
        ):
            raise ValueError("support arrays must have the same sample dimension")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("support embeddings must be finite")
        if np.any(labels < 0) or np.any(frame_indices < 0):
            raise ValueError("support labels and frame indices must be non-negative")
        object.__setattr__(self, "embeddings", _readonly(embeddings, dtype=np.float64))
        object.__setattr__(self, "labels", _readonly(labels, dtype=np.int64))
        object.__setattr__(self, "video_ids", video_ids)
        object.__setattr__(
            self, "frame_indices", _readonly(frame_indices, dtype=np.int64)
        )


@dataclass(frozen=True, slots=True, eq=False)
class OrbitQueryObservation:
    """Label-free causal input passed to a deployable recognizer."""

    embeddings: FloatArray
    video_ids: NDArray[np.str_]
    frame_indices: IntArray

    def __post_init__(self) -> None:
        embeddings = np.asarray(self.embeddings, dtype=np.float64)
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        video_ids = _string_array(self.video_ids, name="query.video_ids")
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("query embeddings must have shape [frame, feature]")
        n = embeddings.shape[0]
        if video_ids.shape != (n,) or frame_indices.shape != (n,):
            raise ValueError("query observation arrays must share the frame dimension")
        if not np.all(np.isfinite(embeddings)) or np.any(frame_indices < 0):
            raise ValueError("query observations must be finite and non-negative")
        for video_id in np.unique(video_ids):
            indices = frame_indices[video_ids == video_id]
            if np.any(np.diff(indices) < 0):
                raise ValueError("query frames must be chronological within each video")
        object.__setattr__(self, "embeddings", _readonly(embeddings, dtype=np.float64))
        object.__setattr__(self, "video_ids", video_ids)
        object.__setattr__(
            self, "frame_indices", _readonly(frame_indices, dtype=np.int64)
        )


@dataclass(frozen=True, slots=True, eq=False)
class OrbitEmbeddingEpisode:
    split: SplitName
    user_id: str
    task_index: int
    class_names: tuple[str, ...]
    support: OrbitSupportSet
    query_embeddings: FloatArray
    query_labels: IntArray
    query_video_ids: NDArray[np.str_]
    query_frame_indices: IntArray
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if not self.user_id or not self.class_names:
            raise ValueError("episode user and class names must be non-empty")
        if len(self.class_names) != len(set(self.class_names)):
            raise ValueError("class names must be unique")
        if isinstance(self.task_index, bool) or int(self.task_index) < 0:
            raise ValueError("task_index must be a non-negative integer")
        query = OrbitQueryObservation(
            self.query_embeddings,
            self.query_video_ids,
            self.query_frame_indices,
        )
        labels = np.asarray(self.query_labels, dtype=np.int64)
        if labels.shape != (query.embeddings.shape[0],):
            raise ValueError("query labels must have one value per query frame")
        n_classes = len(self.class_names)
        if np.any(labels < 0) or np.any(labels >= n_classes):
            raise ValueError("query labels fall outside the episode class range")
        if np.any(self.support.labels >= n_classes):
            raise ValueError("support labels fall outside the episode class range")
        if self.support.embeddings.shape[1] != query.embeddings.shape[1]:
            raise ValueError("support and query feature dimensions must match")
        expected = set(range(n_classes))
        if set(np.unique(self.support.labels).tolist()) != expected:
            raise ValueError("every episode class must have support samples")
        if set(np.unique(labels).tolist()) != expected:
            raise ValueError("every episode class must have query samples")
        fingerprint = self.fingerprint or _hash_payload(
            "orbit-embedding-episode-v1",
            self.split,
            self.user_id,
            int(self.task_index),
            self.class_names,
            self.support.embeddings,
            self.support.labels,
            self.support.video_ids,
            self.support.frame_indices,
            query.embeddings,
            labels,
            query.video_ids,
            query.frame_indices,
        )
        object.__setattr__(self, "task_index", int(self.task_index))
        object.__setattr__(self, "query_embeddings", query.embeddings)
        object.__setattr__(self, "query_labels", _readonly(labels, dtype=np.int64))
        object.__setattr__(self, "query_video_ids", query.video_ids)
        object.__setattr__(self, "query_frame_indices", query.frame_indices)
        object.__setattr__(self, "fingerprint", fingerprint)

    @property
    def query_observation(self) -> OrbitQueryObservation:
        return OrbitQueryObservation(
            self.query_embeddings,
            self.query_video_ids,
            self.query_frame_indices,
        )

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


@dataclass(frozen=True, slots=True)
class OrbitEpisodeSamplingConfig:
    support_stride: int = 30
    max_support_frames_per_video: int = 200
    query_frames_per_video: int = 200
    min_query_frames_per_video: int = 50
    max_frames_per_video: int = 1000
    support_video_limit: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "support_stride",
            "max_support_frames_per_video",
            "query_frames_per_video",
            "min_query_frames_per_video",
            "max_frames_per_video",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.support_video_limit is not None:
            value = self.support_video_limit
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(
                    "support_video_limit must be None or a positive integer"
                )
            object.__setattr__(self, "support_video_limit", int(value))
        if self.min_query_frames_per_video > self.max_frames_per_video:
            raise ValueError("minimum query frames cannot exceed the frame cap")


class OrbitFeatureStore:
    """Read a resumable per-video feature cache and enforce official users."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: SplitName,
        official_splits_path: str | Path,
        require_complete_split: bool = False,
        cache_videos: bool = False,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        manifest_path = self.root / "feature_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"ORBIT feature manifest not found: {manifest_path}"
            )
        frame = pd.read_csv(manifest_path, keep_default_na=False)
        missing = set(FEATURE_MANIFEST_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"feature manifest missing columns: {sorted(missing)}")
        frame = frame.loc[frame["split"].astype(str) == split].copy()
        if frame.empty:
            raise ValueError(f"feature manifest contains no {split} rows")
        for column in (
            "user_id",
            "object_name",
            "video_type",
            "video_id",
            "feature_path",
        ):
            frame[column] = frame[column].astype(str)
            if (frame[column].str.len() == 0).any():
                raise ValueError(f"feature manifest contains empty {column}")
        if not frame["video_type"].isin(["clean", "clutter"]).all():
            raise ValueError("feature manifest video_type must be clean or clutter")
        for column in ("n_frames", "feature_dim"):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
            if (frame[column] <= 0).any():
                raise ValueError(f"feature manifest {column} must be positive")
        if frame["feature_dim"].nunique() != 1:
            raise ValueError("all videos in a feature store must share feature_dim")
        if frame.duplicated(
            ["split", "user_id", "object_name", "video_type", "video_id"]
        ).any():
            raise ValueError("feature manifest contains duplicate videos")
        splits = load_official_orbit_splits(official_splits_path)
        expected = set(splits[split])
        observed = set(frame["user_id"])
        if not observed <= expected:
            raise ValueError(f"feature store has users outside official {split} split")
        if require_complete_split and observed != expected:
            missing_users = sorted(expected - observed)
            raise ValueError(
                f"incomplete official {split} split; missing {missing_users}"
            )
        for relative in frame["feature_path"]:
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise FileNotFoundError(
                    f"feature file is missing or escapes store: {relative}"
                )
        self.frame = frame.sort_values(
            ["user_id", "object_name", "video_type", "video_id"]
        ).reset_index(drop=True)
        self.feature_dim = int(self.frame["feature_dim"].iloc[0])
        self.users = tuple(sorted(observed))
        self.cache_videos = bool(cache_videos)
        self._video_cache: dict[str, tuple[FloatArray, IntArray, BoolArray]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def cache_nbytes(self) -> int:
        """Bytes held by the optional immutable in-process video cache."""

        return int(
            sum(
                array.nbytes
                for values in self._video_cache.values()
                for array in values
            )
        )

    @property
    def cache_stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self.cache_videos,
            "hits": int(self.cache_hits),
            "misses": int(self.cache_misses),
            "videos": len(self._video_cache),
            "nbytes": self.cache_nbytes,
        }

    def _load_video(
        self, row: Mapping[str, object]
    ) -> tuple[FloatArray, IntArray, BoolArray]:
        relative = str(row["feature_path"])
        if self.cache_videos and relative in self._video_cache:
            self.cache_hits += 1
            return self._video_cache[relative]
        self.cache_misses += 1
        path = (self.root / relative).resolve()
        with np.load(path, allow_pickle=False) as payload:
            required = {"embeddings", "frame_indices", "object_present"}
            if not required <= set(payload.files):
                raise ValueError(f"feature file lacks required arrays: {path}")
            embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
            frame_indices = np.asarray(payload["frame_indices"], dtype=np.int64)
            object_present = np.asarray(payload["object_present"], dtype=np.bool_)
        n = int(row["n_frames"])
        if embeddings.shape != (n, self.feature_dim):
            raise ValueError(f"feature shape mismatch in {path}")
        if frame_indices.shape != (n,) or object_present.shape != (n,):
            raise ValueError(f"metadata shape mismatch in {path}")
        if not np.all(np.isfinite(embeddings)) or np.any(np.diff(frame_indices) < 0):
            raise ValueError(f"non-finite or unsorted features in {path}")
        result = (
            _readonly(embeddings, dtype=np.float64),
            _readonly(frame_indices, dtype=np.int64),
            _readonly(object_present, dtype=np.bool_),
        )
        if self.cache_videos:
            self._video_cache[relative] = result
        return result

    def sample_episode(
        self,
        user_id: str,
        *,
        seed: int,
        task_index: int,
        config: OrbitEpisodeSamplingConfig | None = None,
    ) -> OrbitEmbeddingEpisode:
        if user_id not in self.users:
            raise KeyError(f"user {user_id!r} is not in this {self.split} store")
        if (
            isinstance(task_index, bool)
            or not isinstance(task_index, (int, np.integer))
            or int(task_index) < 0
        ):
            raise ValueError("task_index must be a non-negative integer")
        cfg = config or OrbitEpisodeSamplingConfig()
        user_rows = self.frame.loc[self.frame["user_id"] == user_id]
        class_names = tuple(sorted(user_rows["object_name"].unique().tolist()))
        support_embeddings: list[np.ndarray] = []
        support_labels: list[np.ndarray] = []
        support_video_ids: list[np.ndarray] = []
        support_frame_indices: list[np.ndarray] = []
        query_embeddings: list[np.ndarray] = []
        query_labels: list[np.ndarray] = []
        query_video_ids: list[np.ndarray] = []
        query_frame_indices: list[np.ndarray] = []

        for label, object_name in enumerate(class_names):
            object_rows = user_rows.loc[user_rows["object_name"] == object_name]
            clean = object_rows.loc[object_rows["video_type"] == "clean"]
            clutter = object_rows.loc[object_rows["video_type"] == "clutter"]
            if clean.empty or clutter.empty:
                raise ValueError(
                    f"{user_id}/{object_name} lacks clean or clutter videos"
                )
            clean = clean.sort_values("video_id")
            if (
                cfg.support_video_limit is not None
                and len(clean) > cfg.support_video_limit
            ):
                rng = np.random.default_rng(
                    derive_orbit_seed(
                        seed,
                        self.split,
                        user_id,
                        task_index,
                        object_name,
                        "support-videos",
                    )
                )
                positions = np.sort(
                    rng.choice(len(clean), size=cfg.support_video_limit, replace=False)
                )
                clean = clean.iloc[positions]
            for _, row in clean.iterrows():
                embedding, frame_index, present = self._load_video(row)
                candidate = np.flatnonzero(present)[: cfg.max_frames_per_video]
                chosen = candidate[:: cfg.support_stride][
                    : cfg.max_support_frames_per_video
                ]
                if chosen.size == 0:
                    raise ValueError(
                        f"support video has no valid frames: {row['video_id']}"
                    )
                support_embeddings.append(embedding[chosen])
                support_labels.append(np.full(chosen.size, label, dtype=np.int64))
                support_video_ids.append(
                    np.repeat(
                        np.asarray([str(row["video_id"])], dtype=str), chosen.size
                    )
                )
                support_frame_indices.append(frame_index[chosen])

            for _, row in clutter.sort_values("video_id").iterrows():
                embedding, frame_index, present = self._load_video(row)
                candidate = np.flatnonzero(present)[: cfg.max_frames_per_video]
                if candidate.size < cfg.min_query_frames_per_video:
                    raise ValueError(
                        f"query video {row['video_id']} has only {candidate.size} valid frames"
                    )
                n_query = min(cfg.query_frames_per_video, candidate.size)
                rng = np.random.default_rng(
                    derive_orbit_seed(
                        seed,
                        self.split,
                        user_id,
                        task_index,
                        row["video_id"],
                        "query-frames",
                    )
                )
                # The official protocol samples frames randomly.  Sorting the
                # sampled source positions restores chronological order for a
                # causal temporal model without changing the sampled set.
                chosen = np.sort(rng.choice(candidate, size=n_query, replace=False))
                query_embeddings.append(embedding[chosen])
                query_labels.append(np.full(chosen.size, label, dtype=np.int64))
                query_video_ids.append(
                    np.repeat(
                        np.asarray([str(row["video_id"])], dtype=str), chosen.size
                    )
                )
                query_frame_indices.append(frame_index[chosen])

        support = OrbitSupportSet(
            embeddings=np.concatenate(support_embeddings, axis=0),
            labels=np.concatenate(support_labels),
            video_ids=np.concatenate(support_video_ids),
            frame_indices=np.concatenate(support_frame_indices),
        )
        return OrbitEmbeddingEpisode(
            split=self.split,
            user_id=user_id,
            task_index=int(task_index),
            class_names=class_names,
            support=support,
            query_embeddings=np.concatenate(query_embeddings, axis=0),
            query_labels=np.concatenate(query_labels),
            query_video_ids=np.concatenate(query_video_ids),
            query_frame_indices=np.concatenate(query_frame_indices),
        )


def validate_user_disjoint_stores(stores: Iterable[OrbitFeatureStore]) -> None:
    """Reject any repeated user across supplied feature stores."""

    seen: set[str] = set()
    for store in stores:
        overlap = seen & set(store.users)
        if overlap:
            raise ValueError(f"ORBIT feature stores share users: {sorted(overlap)}")
        seen.update(store.users)
