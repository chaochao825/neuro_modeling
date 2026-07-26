"""Fail-closed CORe50 feature store and deterministic streaming tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

from src.utils.reproducibility import derive_seed


FloatArray = NDArray[np.float64]


def _natural_index(value: str) -> tuple[str, int]:
    prefix = value.rstrip("0123456789")
    suffix = value[len(prefix) :]
    return prefix, int(suffix) if suffix else -1


@dataclass(frozen=True, slots=True, eq=False)
class Core50Stream:
    session_id: str
    task_index: int
    panel: str
    evidence: FloatArray
    labels: NDArray[np.int64]
    stream_ids: NDArray[np.str_]
    switch_flags: NDArray[np.bool_]
    object_ids: tuple[str, ...]
    source_cells: tuple[str, ...]

    def __post_init__(self) -> None:
        evidence = np.asarray(self.evidence, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.int64)
        streams = np.asarray(self.stream_ids, dtype=str)
        switches = np.asarray(self.switch_flags, dtype=np.bool_)
        if self.panel not in {"natural", "hidden_switch"}:
            raise ValueError("panel must be natural or hidden_switch")
        if evidence.ndim != 2 or evidence.shape[0] == 0 or evidence.shape[1] < 2:
            raise ValueError("evidence must have shape [frame, class>=2]")
        if labels.shape != (evidence.shape[0],):
            raise ValueError("labels must align with evidence")
        if streams.shape != labels.shape or switches.shape != labels.shape:
            raise ValueError("stream metadata must align with evidence")
        if not np.all(np.isfinite(evidence)) or np.any(evidence < 0.0):
            raise ValueError("evidence must be finite and non-negative")
        if not np.allclose(np.sum(evidence, axis=1), 1.0, atol=1e-6):
            raise ValueError("evidence rows must sum to one")
        if np.any(labels < 0) or np.any(labels >= evidence.shape[1]):
            raise ValueError("labels fall outside task classes")
        if len(self.object_ids) != evidence.shape[1]:
            raise ValueError("object_ids must define every task class")
        for name, value, dtype in (
            ("evidence", evidence, np.float64),
            ("labels", labels, np.int64),
            ("stream_ids", streams, str),
            ("switch_flags", switches, np.bool_),
        ):
            result = np.array(value, dtype=dtype, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)


class Core50FeatureStore:
    """Validated session/object feature cells with optional memory cache."""

    def __init__(
        self,
        root: Path,
        *,
        expected_sessions: Iterable[str],
        expected_objects: Iterable[str],
        cache_in_memory: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "feature_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"CORe50 feature manifest missing: {manifest_path}")
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        required = {
            "session_id",
            "object_id",
            "feature_path",
            "n_frames",
            "feature_dim",
        }
        if not required <= set(manifest.columns):
            raise ValueError(f"feature manifest misses {sorted(required-set(manifest.columns))}")
        sessions = tuple(sorted(map(str, expected_sessions), key=_natural_index))
        objects = tuple(sorted(map(str, expected_objects), key=_natural_index))
        expected = {(session, object_id) for session in sessions for object_id in objects}
        observed = set(
            manifest[["session_id", "object_id"]].astype(str).itertuples(index=False, name=None)
        )
        if observed != expected or len(manifest) != len(expected):
            raise ValueError("CORe50 feature manifest has incomplete or duplicate schema")
        dims = pd.to_numeric(manifest["feature_dim"], errors="raise")
        frames = pd.to_numeric(manifest["n_frames"], errors="raise")
        if dims.nunique() != 1 or int(dims.iloc[0]) < 1 or (frames < 1).any():
            raise ValueError("CORe50 feature manifest has invalid shapes")
        missing = [
            value
            for value in manifest["feature_path"].astype(str)
            if not (self.root / value).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"CORe50 cache misses {len(missing)} feature cells")
        self.sessions = sessions
        self.objects = objects
        self.feature_dim = int(dims.iloc[0])
        self._paths = {
            (str(row.session_id), str(row.object_id)): self.root / str(row.feature_path)
            for row in manifest.itertuples(index=False)
        }
        self._cache: dict[tuple[str, str], FloatArray] | None = (
            {} if cache_in_memory else None
        )

    def load(self, session_id: str, object_id: str) -> FloatArray:
        key = (str(session_id), str(object_id))
        if key not in self._paths:
            raise KeyError(f"unknown CORe50 feature cell: {key}")
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        with np.load(self._paths[key], allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float64)
        if (
            features.ndim != 2
            or features.shape[0] < 1
            or features.shape[1] != self.feature_dim
            or not np.all(np.isfinite(features))
        ):
            raise ValueError(f"invalid CORe50 feature cell: {key}")
        features.setflags(write=False)
        if self._cache is not None:
            self._cache[key] = features
        return features


def fit_object_prototypes(
    store: Core50FeatureStore,
    *,
    support_session: str,
    object_ids: Iterable[str],
    stride: int,
    max_frames_per_object: int,
) -> FloatArray:
    if isinstance(stride, bool) or int(stride) < 1:
        raise ValueError("stride must be positive")
    if isinstance(max_frames_per_object, bool) or int(max_frames_per_object) < 1:
        raise ValueError("max_frames_per_object must be positive")
    prototypes: list[FloatArray] = []
    for object_id in map(str, object_ids):
        features = store.load(str(support_session), object_id)[:: int(stride)][
            : int(max_frames_per_object)
        ]
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized = features / np.maximum(norms, 1e-12)
        prototype = np.mean(normalized, axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        prototypes.append(prototype)
    result = np.asarray(prototypes, dtype=np.float64)
    result.setflags(write=False)
    return result


def cosine_probability_evidence(
    features_value: ArrayLike,
    prototypes_value: ArrayLike,
    *,
    temperature: float,
) -> FloatArray:
    features = np.asarray(features_value, dtype=np.float64)
    prototypes = np.asarray(prototypes_value, dtype=np.float64)
    scale = float(temperature)
    if (
        features.ndim != 2
        or prototypes.ndim != 2
        or features.shape[1] != prototypes.shape[1]
        or prototypes.shape[0] < 2
    ):
        raise ValueError("features and prototypes must be aligned matrices")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(prototypes)):
        raise ValueError("features and prototypes must be finite")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("temperature must be finite and positive")
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    prototype_norm = prototypes / np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
    )
    logits = scale * (normalized @ prototype_norm.T)
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    probabilities.setflags(write=False)
    return probabilities


def prepare_core50_task(
    store: Core50FeatureStore,
    *,
    prototypes: ArrayLike,
    session_id: str,
    seed: int,
    task_index: int,
    temperature: float,
    stream_config: Mapping[str, int],
) -> tuple[Core50Stream, Core50Stream]:
    n_classes = int(stream_config["n_classes"])
    segments = int(stream_config["segments_per_stream"])
    segment_frames = int(stream_config["segment_frames"])
    natural_frames = int(stream_config["natural_frames"])
    if n_classes < 2 or segments < 2 or segment_frames < 1 or natural_frames < segment_frames:
        raise ValueError("invalid CORe50 stream configuration")
    all_prototypes = np.asarray(prototypes, dtype=np.float64)
    if all_prototypes.shape != (len(store.objects), store.feature_dim):
        raise ValueError("prototype bank does not match store")
    rng = np.random.default_rng(
        derive_seed(int(seed), "core50-task", str(session_id), int(task_index))
    )
    indices = np.sort(rng.choice(len(store.objects), size=n_classes, replace=False))
    object_ids = tuple(store.objects[int(index)] for index in indices)
    task_prototypes = all_prototypes[indices]
    pools: list[FloatArray] = []
    sources: list[str] = []
    for object_id in object_ids:
        features = store.load(str(session_id), object_id)
        if len(features) < natural_frames:
            raise ValueError(f"{session_id}/{object_id} has insufficient frames")
        start = int(rng.integers(0, len(features) - natural_frames + 1))
        pools.append(
            cosine_probability_evidence(
                features[start : start + natural_frames],
                task_prototypes,
                temperature=temperature,
            )
        )
        sources.append(f"{session_id}/{object_id}:{start}:{start+natural_frames}")

    natural_evidence = np.concatenate(pools, axis=0)
    natural_labels = np.repeat(np.arange(n_classes, dtype=np.int64), natural_frames)
    natural_streams = np.concatenate(
        [np.repeat(f"{session_id}:{task_index}:{object_id}", natural_frames) for object_id in object_ids]
    )
    natural = Core50Stream(
        session_id=str(session_id),
        task_index=int(task_index),
        panel="natural",
        evidence=natural_evidence,
        labels=natural_labels,
        stream_ids=natural_streams,
        switch_flags=np.zeros(len(natural_labels), dtype=np.bool_),
        object_ids=object_ids,
        source_cells=tuple(sources),
    )

    sequence = [int(rng.integers(0, n_classes))]
    for _ in range(1, segments):
        candidates = [index for index in range(n_classes) if index != sequence[-1]]
        sequence.append(int(rng.choice(candidates)))
    hidden_parts: list[FloatArray] = []
    hidden_sources: list[str] = []
    for segment_index, class_index in enumerate(sequence):
        offset = int(rng.integers(0, natural_frames - segment_frames + 1))
        hidden_parts.append(pools[class_index][offset : offset + segment_frames])
        hidden_sources.append(
            f"{session_id}/{object_ids[class_index]}:pool-offset-{offset}:segment-{segment_index}"
        )
    hidden_evidence = np.concatenate(hidden_parts, axis=0)
    hidden_labels = np.repeat(np.asarray(sequence, dtype=np.int64), segment_frames)
    switches = np.zeros(len(hidden_labels), dtype=np.bool_)
    switches[np.arange(1, segments) * segment_frames] = True
    hidden = Core50Stream(
        session_id=str(session_id),
        task_index=int(task_index),
        panel="hidden_switch",
        evidence=hidden_evidence,
        labels=hidden_labels,
        stream_ids=np.repeat(f"{session_id}:{task_index}:hidden", len(hidden_labels)),
        switch_flags=switches,
        object_ids=object_ids,
        source_cells=tuple(hidden_sources),
    )
    return natural, hidden


__all__ = [
    "Core50FeatureStore",
    "Core50Stream",
    "cosine_probability_evidence",
    "fit_object_prototypes",
    "prepare_core50_task",
]
