"""Outcome-blind Stream-51 video splits and frozen-feature streaming tasks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from src.models.embedding_evidence import VMFEvidenceModel
from src.utils.reproducibility import derive_seed


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Stream51FrameRecord:
    """One official ordering row with its source-video identifiers."""

    path: str
    class_id: int
    class_name: str
    clip_id: int
    video_id: int
    frame_id: int

    @property
    def video_key(self) -> str:
        return (
            f"c{self.class_id:02d}_clip{self.clip_id:03d}_"
            f"video{self.video_id:03d}"
        )


def parse_stream51_ordering_line(line: str) -> Stream51FrameRecord:
    """Parse one official ``path class_id`` ordering row."""

    fields = str(line).strip().rsplit(maxsplit=1)
    if len(fields) != 2:
        raise ValueError("Stream-51 ordering rows must contain path and class")
    path_text, class_text = fields
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
        raise ValueError("Stream-51 path must be a safe train/class/file path")
    if path.parts[0] != "train" or path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("Stream-51 training path has unexpected schema")
    try:
        class_id = int(class_text)
    except ValueError as error:
        raise ValueError("Stream-51 class id must be an integer") from error
    folder_fields = path.parts[1].split("-", maxsplit=1)
    if len(folder_fields) != 2 or not folder_fields[1]:
        raise ValueError("Stream-51 class folder has unexpected schema")
    if int(folder_fields[0]) - 1 != class_id:
        raise ValueError("Stream-51 class folder and label disagree")
    stem_fields = path.stem.split("_")
    if len(stem_fields) != 3 or not all(field.isdigit() for field in stem_fields):
        raise ValueError("Stream-51 frame name has unexpected schema")
    clip_id, video_id, frame_id = map(int, stem_fields)
    if min(class_id, clip_id, video_id, frame_id) < 0:
        raise ValueError("Stream-51 identifiers must be non-negative")
    return Stream51FrameRecord(
        path=path.as_posix(),
        class_id=class_id,
        class_name=folder_fields[1],
        clip_id=clip_id,
        video_id=video_id,
        frame_id=frame_id,
    )


def read_stream51_ordering(path: str | Path) -> tuple[Stream51FrameRecord, ...]:
    ordering_path = Path(path).expanduser().resolve()
    if not ordering_path.is_file():
        raise FileNotFoundError(f"Stream-51 ordering is missing: {ordering_path}")
    records = tuple(
        parse_stream51_ordering_line(line)
        for line in ordering_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records:
        raise ValueError("Stream-51 ordering is empty")
    paths = [record.path for record in records]
    if len(set(paths)) != len(paths):
        raise ValueError("Stream-51 ordering contains duplicate frames")
    return records


def stream51_video_table(
    records: Iterable[Stream51FrameRecord],
) -> pd.DataFrame:
    rows = [
        {
            "video_key": record.video_key,
            "class_id": record.class_id,
            "class_name": record.class_name,
            "clip_id": record.clip_id,
            "video_id": record.video_id,
            "frame_id": record.frame_id,
            "path": record.path,
        }
        for record in records
    ]
    if not rows:
        raise ValueError("records must be non-empty")
    frame = pd.DataFrame(rows)
    class_counts = frame.groupby("class_id")["class_name"].nunique()
    if (class_counts != 1).any():
        raise ValueError("class ids do not map uniquely to class names")
    video_classes = frame.groupby("video_key")["class_id"].nunique()
    if (video_classes != 1).any():
        raise ValueError("video keys cross class boundaries")
    duplicate_frames = frame.duplicated(["video_key", "frame_id"])
    if duplicate_frames.any():
        raise ValueError("video contains duplicate frame identifiers")
    return frame


def assign_stream51_video_splits(
    records: Iterable[Stream51FrameRecord],
    *,
    salt: str,
    support_fraction: float = 0.4,
    development_fraction: float = 0.3,
    minimum_videos_per_split: int = 3,
) -> dict[str, object]:
    """Assign whole source videos before model outcomes are available."""

    if not isinstance(salt, str) or not salt:
        raise ValueError("split salt must be non-empty")
    support_fraction = float(support_fraction)
    development_fraction = float(development_fraction)
    if (
        not np.isfinite(support_fraction)
        or not np.isfinite(development_fraction)
        or support_fraction <= 0.0
        or development_fraction <= 0.0
        or support_fraction + development_fraction >= 1.0
    ):
        raise ValueError("split fractions must be positive and sum below one")
    if isinstance(minimum_videos_per_split, bool) or not isinstance(
        minimum_videos_per_split, (int, np.integer)
    ):
        raise TypeError("minimum_videos_per_split must be an integer")
    minimum = int(minimum_videos_per_split)
    if minimum < 1:
        raise ValueError("minimum_videos_per_split must be positive")
    frame = stream51_video_table(records)
    videos = (
        frame.groupby(
            ["video_key", "class_id", "class_name", "clip_id", "video_id"],
            as_index=False,
        )
        .agg(
            n_available_frames=("frame_id", "size"),
            first_frame_id=("frame_id", "min"),
            last_frame_id=("frame_id", "max"),
        )
        .sort_values(["class_id", "video_key"])
    )
    assignments: list[dict[str, object]] = []
    for class_id, group in videos.groupby("class_id", sort=True):
        group_rows = group.to_dict("records")
        if len(group_rows) < 3 * minimum:
            raise ValueError(
                f"class {class_id} has fewer than {3 * minimum} videos"
            )
        ranked = sorted(
            group_rows,
            key=lambda row: hashlib.sha256(
                f"{salt}|{class_id}|{row['video_key']}".encode()
            ).hexdigest(),
        )
        n_support = max(minimum, int(np.floor(len(ranked) * support_fraction)))
        n_development = max(
            minimum, int(np.floor(len(ranked) * development_fraction))
        )
        n_external = len(ranked) - n_support - n_development
        if n_external < minimum:
            raise ValueError(f"class {class_id} cannot satisfy external minimum")
        boundaries = (n_support, n_support + n_development)
        for index, row in enumerate(ranked):
            split = (
                "support"
                if index < boundaries[0]
                else "development" if index < boundaries[1] else "external"
            )
            assignments.append({**row, "split": split})
    assignments.sort(key=lambda row: (str(row["split"]), str(row["video_key"])))
    counts = pd.DataFrame(assignments).groupby("split").size().to_dict()
    return {
        "schema_version": "stream51_video_split_v1",
        "split_salt": salt,
        "support_fraction": support_fraction,
        "development_fraction": development_fraction,
        "minimum_videos_per_class_per_split": minimum,
        "n_classes": int(videos["class_id"].nunique()),
        "n_videos": int(len(videos)),
        "split_counts": {str(key): int(value) for key, value in counts.items()},
        "videos": assignments,
    }


@dataclass(frozen=True, slots=True, eq=False)
class Stream51Stream:
    split: str
    task_index: int
    panel: str
    evidence: FloatArray
    observation_log_likelihood: FloatArray
    labels: NDArray[np.int64]
    stream_ids: NDArray[np.str_]
    switch_flags: NDArray[np.bool_]
    source_video_ids: NDArray[np.str_]

    def __post_init__(self) -> None:
        evidence = np.asarray(self.evidence, dtype=np.float64)
        log_likelihood = np.asarray(
            self.observation_log_likelihood, dtype=np.float64
        )
        labels = np.asarray(self.labels, dtype=np.int64)
        streams = np.asarray(self.stream_ids, dtype=str)
        switches = np.asarray(self.switch_flags, dtype=np.bool_)
        sources = np.asarray(self.source_video_ids, dtype=str)
        if self.split not in {"development", "external"}:
            raise ValueError("query split must be development or external")
        if self.panel not in {"natural", "hidden_switch"}:
            raise ValueError("panel must be natural or hidden_switch")
        if evidence.ndim != 2 or evidence.shape[0] == 0 or evidence.shape[1] < 2:
            raise ValueError("evidence must have shape [frame, class>=2]")
        if log_likelihood.shape != evidence.shape:
            raise ValueError("observation log likelihood must match evidence")
        for name, value in (
            ("labels", labels),
            ("streams", streams),
            ("switches", switches),
            ("sources", sources),
        ):
            if value.shape != (evidence.shape[0],):
                raise ValueError(f"{name} must align with evidence")
        if not np.all(np.isfinite(evidence)) or np.any(evidence < 0.0):
            raise ValueError("evidence must be finite and non-negative")
        if not np.allclose(evidence.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("evidence rows must sum to one")
        if not np.all(np.isfinite(log_likelihood)):
            raise ValueError("observation log likelihood must be finite")
        if np.any(labels < 0) or np.any(labels >= evidence.shape[1]):
            raise ValueError("labels fall outside evidence classes")
        if np.any(np.char.str_len(streams) == 0) or np.any(
            np.char.str_len(sources) == 0
        ):
            raise ValueError("stream and source identifiers must be non-empty")
        for name, value, dtype in (
            ("evidence", evidence, np.float64),
            ("observation_log_likelihood", log_likelihood, np.float64),
            ("labels", labels, np.int64),
            ("stream_ids", streams, str),
            ("switch_flags", switches, np.bool_),
            ("source_video_ids", sources, str),
        ):
            result = np.array(value, dtype=dtype, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)


class Stream51FeatureStore:
    """Fail-closed video-level cache whose required splits are explicit."""

    def __init__(
        self,
        root: str | Path,
        *,
        cohort_path: str | Path,
        required_splits: Sequence[str],
        cache_in_memory: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        cohort_file = Path(cohort_path).expanduser().resolve()
        if not cohort_file.is_file():
            raise FileNotFoundError(f"Stream-51 cohort is missing: {cohort_file}")
        cohort = json.loads(cohort_file.read_text(encoding="utf-8"))
        if cohort.get("schema_version") != "stream51_video_split_v1":
            raise ValueError("Stream-51 cohort schema mismatch")
        video_rows = pd.DataFrame(cohort.get("videos", []))
        required_video_columns = {
            "video_key",
            "class_id",
            "class_name",
            "split",
            "n_available_frames",
        }
        if video_rows.empty or not required_video_columns <= set(video_rows.columns):
            raise ValueError("Stream-51 cohort video table is incomplete")
        if video_rows["video_key"].duplicated().any():
            raise ValueError("Stream-51 cohort contains duplicate videos")
        splits = tuple(map(str, required_splits))
        if not splits or len(set(splits)) != len(splits):
            raise ValueError("required_splits must be unique and non-empty")
        if not set(splits) <= {"support", "development", "external"}:
            raise ValueError("required_splits contains an unknown split")
        manifest_path = self.root / "feature_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Stream-51 feature manifest is missing: {manifest_path}"
            )
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        required_manifest_columns = {
            "video_key",
            "class_id",
            "split",
            "feature_path",
            "n_frames",
            "feature_dim",
        }
        if not required_manifest_columns <= set(manifest.columns):
            raise ValueError("Stream-51 feature manifest is incomplete")
        if manifest["video_key"].duplicated().any():
            raise ValueError("Stream-51 feature manifest contains duplicate videos")
        expected_rows = video_rows[video_rows["split"].isin(splits)].copy()
        observed_rows = manifest[manifest["split"].isin(splits)].copy()
        expected = set(expected_rows["video_key"].astype(str))
        observed = set(observed_rows["video_key"].astype(str))
        if observed != expected:
            raise ValueError("Stream-51 feature cache does not cover required splits")
        cohort_map = video_rows.set_index("video_key")
        for row in observed_rows.itertuples(index=False):
            expected_row = cohort_map.loc[str(row.video_key)]
            if int(row.class_id) != int(expected_row.class_id) or str(row.split) != str(
                expected_row.split
            ):
                raise ValueError("Stream-51 manifest disagrees with frozen cohort")
        dims = pd.to_numeric(observed_rows["feature_dim"], errors="raise")
        frames = pd.to_numeric(observed_rows["n_frames"], errors="raise")
        if dims.empty or dims.nunique() != 1 or int(dims.iloc[0]) < 2:
            raise ValueError("Stream-51 feature dimensions are invalid")
        if (frames < 2).any():
            raise ValueError("Stream-51 videos require at least two cached frames")
        missing = [
            str(path)
            for path in observed_rows["feature_path"]
            if not (self.root / str(path)).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Stream-51 cache misses {len(missing)} feature videos"
            )
        self.required_splits = splits
        self.feature_dim = int(dims.iloc[0])
        self.n_classes = int(cohort["n_classes"])
        self._metadata = {
            str(row.video_key): {
                "class_id": int(row.class_id),
                "class_name": str(row.class_name),
                "split": str(row.split),
            }
            for row in expected_rows.itertuples(index=False)
        }
        self._paths = {
            str(row.video_key): self.root / str(row.feature_path)
            for row in observed_rows.itertuples(index=False)
        }
        self._cache: dict[str, FloatArray] | None = {} if cache_in_memory else None

    def video_keys(self, split: str) -> tuple[str, ...]:
        split = str(split)
        if split not in self.required_splits:
            raise KeyError(f"split is not loaded: {split}")
        return tuple(
            sorted(
                key
                for key, metadata in self._metadata.items()
                if metadata["split"] == split
            )
        )

    def class_id(self, video_key: str) -> int:
        try:
            return int(self._metadata[str(video_key)]["class_id"])
        except KeyError as error:
            raise KeyError(f"unknown Stream-51 video: {video_key}") from error

    def load(self, video_key: str) -> FloatArray:
        key = str(video_key)
        if key not in self._paths:
            raise KeyError(f"unknown Stream-51 video: {key}")
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        with np.load(self._paths[key], allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float64)
        if (
            features.ndim != 2
            or features.shape[0] < 2
            or features.shape[1] != self.feature_dim
            or not np.all(np.isfinite(features))
        ):
            raise ValueError(f"invalid Stream-51 feature video: {key}")
        features.setflags(write=False)
        if self._cache is not None:
            self._cache[key] = features
        return features


def fit_stream51_vmf(
    store: Stream51FeatureStore,
    *,
    split: str = "support",
    max_frames_per_video: int | None = None,
) -> VMFEvidenceModel:
    if split not in store.required_splits:
        raise KeyError(f"split is not loaded: {split}")
    features: list[FloatArray] = []
    labels: list[NDArray[np.int64]] = []
    for video_key in store.video_keys(split):
        video = store.load(video_key)
        if max_frames_per_video is not None:
            if isinstance(max_frames_per_video, bool) or int(max_frames_per_video) < 1:
                raise ValueError("max_frames_per_video must be positive")
            video = video[: int(max_frames_per_video)]
        features.append(video)
        labels.append(
            np.full(video.shape[0], store.class_id(video_key), dtype=np.int64)
        )
    return VMFEvidenceModel.fit(
        np.concatenate(features),
        np.concatenate(labels),
        n_classes=store.n_classes,
    )


def _subsample_video(
    features: FloatArray, *, n_frames: int, seed: int, namespace: str
) -> FloatArray:
    if isinstance(n_frames, bool) or not isinstance(n_frames, (int, np.integer)):
        raise TypeError("n_frames must be an integer")
    n_frames = int(n_frames)
    if n_frames < 2:
        raise ValueError("n_frames must be at least two")
    if len(features) <= n_frames:
        return features
    rng = np.random.default_rng(derive_seed(seed, namespace, len(features), n_frames))
    start = int(rng.integers(0, len(features) - n_frames + 1))
    return features[start : start + n_frames]


def make_stream51_natural_streams(
    store: Stream51FeatureStore,
    model: VMFEvidenceModel,
    *,
    split: str,
    temperature: float,
    max_frames: int,
    video_keys: Sequence[str] | None = None,
) -> tuple[Stream51Stream, ...]:
    streams: list[Stream51Stream] = []
    selected = tuple(video_keys) if video_keys is not None else store.video_keys(split)
    if not selected or not set(selected) <= set(store.video_keys(split)):
        raise ValueError("video_keys must be a non-empty subset of the requested split")
    for task_index, video_key in enumerate(selected):
        features = store.load(video_key)
        if len(features) > int(max_frames):
            indices = np.linspace(0, len(features) - 1, int(max_frames)).round().astype(int)
            features = features[indices]
        log_likelihood = model.relative_log_likelihood(features)
        evidence = model.probabilities(features, temperature=temperature)
        class_id = store.class_id(video_key)
        n_frames = len(features)
        streams.append(
            Stream51Stream(
                split=split,
                task_index=task_index,
                panel="natural",
                evidence=evidence,
                observation_log_likelihood=log_likelihood,
                labels=np.full(n_frames, class_id, dtype=np.int64),
                stream_ids=np.repeat(f"{split}:natural:{video_key}", n_frames),
                switch_flags=np.zeros(n_frames, dtype=np.bool_),
                source_video_ids=np.repeat(video_key, n_frames),
            )
        )
    return tuple(streams)


def _alternating_video_order(
    store: Stream51FeatureStore,
    *,
    split: str,
    seed: int,
    video_keys: Sequence[str] | None = None,
) -> list[str]:
    rng = np.random.default_rng(derive_seed(seed, "stream51-hidden-order", split))
    remaining = (
        list(video_keys) if video_keys is not None else list(store.video_keys(split))
    )
    if not remaining or not set(remaining) <= set(store.video_keys(split)):
        raise ValueError("video_keys must be a non-empty subset of the requested split")
    rng.shuffle(remaining)
    ordered: list[str] = []
    while remaining:
        previous_class = store.class_id(ordered[-1]) if ordered else None
        candidate_index = next(
            (
                index
                for index, key in enumerate(remaining)
                if store.class_id(key) != previous_class
            ),
            0,
        )
        ordered.append(remaining.pop(candidate_index))
    return ordered


def make_stream51_hidden_streams(
    store: Stream51FeatureStore,
    model: VMFEvidenceModel,
    *,
    split: str,
    seed: int,
    temperature: float,
    segment_frames: int,
    videos_per_stream: int,
    video_keys: Sequence[str] | None = None,
) -> tuple[Stream51Stream, ...]:
    if isinstance(videos_per_stream, bool) or int(videos_per_stream) < 2:
        raise ValueError("videos_per_stream must be at least two")
    if isinstance(segment_frames, bool) or int(segment_frames) < 2:
        raise ValueError("segment_frames must be at least two")
    videos_per_stream = int(videos_per_stream)
    segment_frames = int(segment_frames)
    ordered = _alternating_video_order(
        store, split=split, seed=seed, video_keys=video_keys
    )
    streams: list[Stream51Stream] = []
    chunks = [
        ordered[start : start + videos_per_stream]
        for start in range(0, len(ordered), videos_per_stream)
    ]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2].extend(chunks.pop())
    if not chunks or len(chunks[0]) < 2:
        raise ValueError("hidden stream requires at least two videos")
    for task_index, video_keys in enumerate(chunks):
        feature_parts: list[FloatArray] = []
        label_parts: list[NDArray[np.int64]] = []
        source_parts: list[NDArray[np.str_]] = []
        switch_parts: list[NDArray[np.bool_]] = []
        for segment_index, video_key in enumerate(video_keys):
            features = _subsample_video(
                store.load(video_key),
                n_frames=segment_frames,
                seed=derive_seed(seed, task_index, segment_index, video_key),
                namespace="stream51-segment",
            )
            n_frames = len(features)
            switches = np.zeros(n_frames, dtype=np.bool_)
            if segment_index > 0:
                switches[0] = True
            feature_parts.append(features)
            label_parts.append(
                np.full(n_frames, store.class_id(video_key), dtype=np.int64)
            )
            source_parts.append(np.repeat(video_key, n_frames))
            switch_parts.append(switches)
        concatenated = np.concatenate(feature_parts)
        log_likelihood = model.relative_log_likelihood(concatenated)
        evidence = model.probabilities(concatenated, temperature=temperature)
        stream_name = f"{split}:hidden:{seed}:{task_index}"
        streams.append(
            Stream51Stream(
                split=split,
                task_index=task_index,
                panel="hidden_switch",
                evidence=evidence,
                observation_log_likelihood=log_likelihood,
                labels=np.concatenate(label_parts),
                stream_ids=np.repeat(stream_name, len(concatenated)),
                switch_flags=np.concatenate(switch_parts),
                source_video_ids=np.concatenate(source_parts),
            )
        )
    return tuple(streams)


__all__ = [
    "Stream51FeatureStore",
    "Stream51FrameRecord",
    "Stream51Stream",
    "assign_stream51_video_splits",
    "fit_stream51_vmf",
    "make_stream51_hidden_streams",
    "make_stream51_natural_streams",
    "parse_stream51_ordering_line",
    "read_stream51_ordering",
    "stream51_video_table",
]
