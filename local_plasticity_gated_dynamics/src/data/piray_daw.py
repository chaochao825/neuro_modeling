"""Fail-closed loader for the Piray--Daw 2024 behavioral dataset.

Only the released ``data.mat`` files are read.  Pre-fitted ``model_*.mat``
artifacts are deliberately outside this interface so deployable methods cannot
silently consume participant-fitted or privileged signals.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


DATA_ARCHIVE_MD5 = "59cdae0b66f5868ae9268729fa863d0e"
DATA_MAT_SHA256 = {
    1: "dbb6446bc412f5fcf883dcb4b0a950e74b3109814001f7c0ccbe968d0a197271",
    2: "b6d74c615b7ec8f4ff5e9c4ccb058ce89e47568be6f22dfbbfc0f3ec3c65a0f5",
}
EXPECTED_PARTICIPANTS = {1: 223, 2: 420}
EXPECTED_TRIALS = 50
EXPECTED_BLOCKS = 4


def _digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _readonly(value: Any, *, dtype: np.dtype[Any] | type) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PirayDawDataset:
    """Canonical participant-by-trial representation of one experiment."""

    experiment: int
    bucket: np.ndarray
    response_time: np.ndarray
    randomization_order: np.ndarray
    age: np.ndarray
    gender: tuple[str, ...]
    bag: np.ndarray
    bird: np.ndarray
    true_process_variance: np.ndarray
    true_observation_variance: np.ndarray
    source_path: Path
    source_sha256: str

    def __post_init__(self) -> None:
        if self.experiment not in (1, 2):
            raise ValueError("experiment must be 1 or 2")
        typed_arrays = {
            "bucket": (self.bucket, np.float64),
            "response_time": (self.response_time, np.float64),
            "randomization_order": (self.randomization_order, np.int64),
            "age": (self.age, np.float64),
            "bag": (self.bag, np.float64),
            "bird": (self.bird, np.float64),
            "true_process_variance": (self.true_process_variance, np.float64),
            "true_observation_variance": (
                self.true_observation_variance,
                np.float64,
            ),
        }
        for name, (value, dtype) in typed_arrays.items():
            object.__setattr__(self, name, _readonly(value, dtype=dtype))
        object.__setattr__(self, "gender", tuple(map(str, self.gender)))
        object.__setattr__(self, "source_path", Path(self.source_path))
        expected_n = EXPECTED_PARTICIPANTS[self.experiment]
        expected_cube = (expected_n, EXPECTED_TRIALS, EXPECTED_BLOCKS)
        if np.asarray(self.bucket).shape != expected_cube:
            raise ValueError(f"bucket must have shape {expected_cube}")
        if np.asarray(self.response_time).shape != expected_cube:
            raise ValueError(f"response_time must have shape {expected_cube}")
        if np.asarray(self.randomization_order).shape != (
            expected_n,
            EXPECTED_BLOCKS,
        ):
            raise ValueError("randomization_order has an invalid shape")
        if np.asarray(self.age).shape != (expected_n,):
            raise ValueError("age has an invalid shape")
        if len(self.gender) != expected_n:
            raise ValueError("gender has an invalid length")
        for name in ("bag", "bird"):
            if np.asarray(getattr(self, name)).shape != (
                EXPECTED_TRIALS,
                EXPECTED_BLOCKS,
            ):
                raise ValueError(f"{name} has an invalid shape")
        for name in ("true_process_variance", "true_observation_variance"):
            if np.asarray(getattr(self, name)).shape != (EXPECTED_BLOCKS,):
                raise ValueError(f"{name} has an invalid shape")
        numeric = (
            self.bucket,
            self.response_time,
            self.randomization_order,
            self.bag,
            self.bird,
            self.true_process_variance,
            self.true_observation_variance,
        )
        if not all(np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("core released arrays must be finite")
        if np.any(np.isinf(self.age)):
            raise ValueError("age may be missing but cannot be infinite")
        expected_permutation = np.arange(EXPECTED_BLOCKS)
        if not all(
            np.array_equal(np.sort(row), expected_permutation)
            for row in self.randomization_order
        ):
            raise ValueError("each randomization_order row must be a 0..3 permutation")
        cells = set(
            zip(
                self.true_process_variance.tolist(),
                self.true_observation_variance.tolist(),
                strict=True,
            )
        )
        if len(cells) != EXPECTED_BLOCKS:
            raise ValueError("the four Q/R factorial cells must be unique")
        expected_cells = {(4.0, 16.0), (49.0, 16.0), (4.0, 64.0), (49.0, 64.0)}
        if cells != expected_cells:
            raise ValueError("Q/R labels must match the released 2x2 factorial design")

    @property
    def n_participants(self) -> int:
        return int(self.bucket.shape[0])

    def long_frame(self, *, include_hidden_bird: bool = False) -> pd.DataFrame:
        """Return one row per participant/block/trial without reordering trials.

        Hidden bird position is evaluation-only and omitted by default.
        """

        participant, trial, block = np.indices(self.bucket.shape)
        frame = pd.DataFrame(
            {
                "experiment": self.experiment,
                "participant_id": participant.ravel(),
                "block_id": block.ravel(),
                "trial": trial.ravel(),
                "bucket": self.bucket.ravel(),
                "response_time": self.response_time.ravel(),
                "bag": np.broadcast_to(self.bag, self.bucket.shape).ravel(),
                "true_process_variance": np.broadcast_to(
                    self.true_process_variance[None, None, :], self.bucket.shape
                ).ravel(),
                "true_observation_variance": np.broadcast_to(
                    self.true_observation_variance[None, None, :], self.bucket.shape
                ).ravel(),
            }
        )
        if include_hidden_bird:
            frame["bird"] = np.broadcast_to(self.bird, self.bucket.shape).ravel()
        return frame


def _participant_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        participants = [value]
    elif isinstance(value, np.ndarray):
        participants = list(value.reshape(-1))
    else:
        participants = list(value)
    if not participants or not all(isinstance(row, dict) for row in participants):
        raise ValueError("data must contain participant structs")
    return participants


def load_piray_daw(
    root: str | Path,
    *,
    experiment: int,
    allow_confirmation: bool = False,
    verify_hashes: bool = True,
) -> PirayDawDataset:
    """Load one released experiment with provenance and schema checks.

    Experiment 2 is a locked confirmatory cohort.  Callers must opt in
    explicitly after a protocol authorizes its use.
    """

    if experiment not in (1, 2):
        raise ValueError("experiment must be 1 or 2")
    if experiment == 2 and not allow_confirmation:
        raise PermissionError(
            "Experiment 2 is confirmation-locked; pass allow_confirmation=True "
            "only from an authorized immutable confirmation configuration"
        )
    base = Path(root).expanduser().resolve()
    archive = base / "data.zip"
    source = base / f"experiment{experiment}" / "data.mat"
    if not archive.is_file() or not source.is_file():
        raise FileNotFoundError(
            f"expected hash-bound data.zip and {source.relative_to(base)} under {base}"
        )
    if verify_hashes:
        archive_md5 = _digest(archive, "md5")
        if archive_md5 != DATA_ARCHIVE_MD5:
            raise ValueError(
                f"data.zip MD5 mismatch: expected {DATA_ARCHIVE_MD5}, got {archive_md5}"
            )
        source_sha256 = _digest(source, "sha256")
        if source_sha256 != DATA_MAT_SHA256[experiment]:
            raise ValueError(
                "extracted data.mat SHA256 mismatch: "
                f"expected {DATA_MAT_SHA256[experiment]}, got {source_sha256}"
            )
    else:
        source_sha256 = _digest(source, "sha256")

    payload = loadmat(source, simplify_cells=True)
    if "meta_data" not in payload or "data" not in payload:
        raise ValueError("data.mat must contain meta_data and data")
    metadata = payload["meta_data"]
    if not isinstance(metadata, dict):
        raise ValueError("meta_data must be a MATLAB struct")
    required_meta = {"true_vol", "true_sto", "bird", "bag"}
    if not required_meta.issubset(metadata):
        raise ValueError(f"meta_data is missing {sorted(required_meta - set(metadata))}")

    participants = _participant_list(payload["data"])
    if len(participants) != EXPECTED_PARTICIPANTS[experiment]:
        raise ValueError(
            f"expected {EXPECTED_PARTICIPANTS[experiment]} participants, "
            f"found {len(participants)}"
        )
    required_participant = {
        "bucket",
        "response_time",
        "randomization_order",
        "age",
        "gender",
    }
    for index, participant in enumerate(participants):
        missing = required_participant - set(participant)
        if missing:
            raise ValueError(f"participant {index} is missing {sorted(missing)}")

    bucket = _readonly(
        np.stack([participant["bucket"] for participant in participants]),
        dtype=np.float64,
    )
    response_time = _readonly(
        np.stack([participant["response_time"] for participant in participants]),
        dtype=np.float64,
    )
    randomization_order = _readonly(
        np.stack([participant["randomization_order"] for participant in participants]),
        dtype=np.int64,
    )
    age = _readonly([participant["age"] for participant in participants], dtype=np.float64)
    gender = tuple(str(participant["gender"]) for participant in participants)
    return PirayDawDataset(
        experiment=experiment,
        bucket=bucket,
        response_time=response_time,
        randomization_order=randomization_order,
        age=age,
        gender=gender,
        bag=_readonly(metadata["bag"], dtype=np.float64),
        bird=_readonly(metadata["bird"], dtype=np.float64),
        true_process_variance=_readonly(metadata["true_vol"], dtype=np.float64),
        true_observation_variance=_readonly(metadata["true_sto"], dtype=np.float64),
        source_path=source,
        source_sha256=source_sha256,
    )


__all__ = [
    "DATA_ARCHIVE_MD5",
    "DATA_MAT_SHA256",
    "EXPECTED_BLOCKS",
    "EXPECTED_PARTICIPANTS",
    "EXPECTED_TRIALS",
    "PirayDawDataset",
    "load_piray_daw",
]
