"""Fail-closed split/evaluation-only IBL block-truth sidecar.

The public neural cache deliberately excludes ``probabilityLeft``.  This
module binds a prepared neural session back to the independently frozen Exp11
behavior cohort only after verifying both file hashes and every trial identity
field shared by the two artifacts.  The returned capability has no cue or
observation adapter.  It is restricted to whole-block split construction and
post-fit evaluation; it must never enter gate or neural-model inputs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from src.data.ibl_multisession import PreparedIBLNeuralSession


Array = np.ndarray
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTEXT_LEVELS = np.asarray([0.2, 0.5, 0.8], dtype=np.float64)
_MANIFEST_COLUMNS = {
    "eid",
    "subject",
    "status",
    "compact_table",
    "compact_table_sha256",
}
_COMPACT_COLUMNS = {
    "source_trial_index",
    "contrastLeft",
    "contrastRight",
    "choice",
    "probabilityLeft",
    "official_bwm_mask",
}
_NEURAL_ALIGNMENT_COLUMNS = {
    "trial_id",
    "stimulus_side",
    "choice",
    "official_bwm_mask",
}


class IBLBlockSidecarError(ValueError):
    """Raised when frozen behavior and neural artifacts cannot be proven equal."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, *, name: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256.fullmatch(text):
        raise IBLBlockSidecarError(f"{name} must be a lowercase 64-hex SHA-256")
    return text


def _text(value: object, *, name: str) -> str:
    if pd.isna(value):
        raise IBLBlockSidecarError(f"{name} must be non-empty")
    text = str(value).strip()
    if not text:
        raise IBLBlockSidecarError(f"{name} must be non-empty")
    return text


def _integer_column(values: object, *, name: str) -> Array:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    if (
        numeric.ndim != 1
        or numeric.size == 0
        or not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
        or np.any(numeric < 0.0)
    ):
        raise IBLBlockSidecarError(
            f"{name} must contain non-negative finite integers"
        )
    result = numeric.astype(np.int64)
    if len(np.unique(result)) != len(result):
        raise IBLBlockSidecarError(f"{name} must contain unique values")
    return result


def _binary_column(values: object, *, name: str) -> Array:
    series = pd.Series(values)
    if series.dtype.kind == "b":
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
        raise IBLBlockSidecarError(f"{name} must contain only binary values")
    return normalized.isin(["true", "1", "1.0"]).to_numpy(dtype=bool)


def _numeric_column(values: object, *, name: str, allow_nan: bool) -> Array:
    result = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    if result.ndim != 1 or result.size == 0 or np.isinf(result).any():
        raise IBLBlockSidecarError(f"{name} must be a numeric vector without infinity")
    if not allow_nan and not np.isfinite(result).all():
        raise IBLBlockSidecarError(f"{name} must be complete and finite")
    return result


def _same_numeric(left: Array, right: Array) -> bool:
    if left.shape != right.shape:
        return False
    return bool(np.all((left == right) | (np.isnan(left) & np.isnan(right))))


def _alignment_fingerprint(*arrays: Array) -> str:
    digest = hashlib.sha256(b"ibl-block-sidecar-alignment-v1\0")
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _freeze(value: object, *, dtype: object) -> Array:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class IBLBlockTruth:
    """Split/evaluation-only true block labels aligned to one prepared session.

    This capability intentionally does not expose ``cue_observations``, an
    ``observations`` conversion, or any gate/model construction method.
    """

    trial_ids: Array
    probability_left: Array
    block_switch: Array
    official_bwm_mask: Array

    def __post_init__(self) -> None:
        trial_ids = _integer_column(self.trial_ids, name="trial_ids")
        probability = np.asarray(self.probability_left, dtype=float)
        switches = np.asarray(self.block_switch)
        official = np.asarray(self.official_bwm_mask)
        n_trials = len(trial_ids)
        if probability.shape != (n_trials,) or not np.isfinite(probability).all():
            raise IBLBlockSidecarError(
                "probability_left must be a matching complete finite vector"
            )
        close = np.isclose(
            probability[:, None],
            _CONTEXT_LEVELS[None, :],
            atol=1e-6,
            rtol=0.0,
        )
        if not np.all(np.sum(close, axis=1) == 1):
            raise IBLBlockSidecarError(
                "probability_left must use only IBL levels 0.2, 0.5, and 0.8"
            )
        probability = _CONTEXT_LEVELS[np.argmax(close, axis=1)]
        if switches.shape != (n_trials,) or switches.dtype.kind != "b":
            raise IBLBlockSidecarError("block_switch must be a matching boolean vector")
        expected_switch = np.zeros(n_trials, dtype=bool)
        expected_switch[1:] = probability[1:] != probability[:-1]
        if not np.array_equal(switches, expected_switch):
            raise IBLBlockSidecarError(
                "block_switch must mark exact probabilityLeft changes"
            )
        if official.shape != (n_trials,) or official.dtype.kind != "b":
            raise IBLBlockSidecarError(
                "official_bwm_mask must be a matching boolean vector"
            )
        object.__setattr__(self, "trial_ids", _freeze(trial_ids, dtype=np.int64))
        object.__setattr__(
            self, "probability_left", _freeze(probability, dtype=np.float64)
        )
        object.__setattr__(self, "block_switch", _freeze(switches, dtype=bool))
        object.__setattr__(
            self, "official_bwm_mask", _freeze(official, dtype=bool)
        )

    @property
    def fingerprint(self) -> str:
        return _alignment_fingerprint(
            self.trial_ids,
            self.probability_left,
            self.block_switch,
            self.official_bwm_mask,
        )


def _manifest_row(
    manifest: pd.DataFrame,
    prepared: PreparedIBLNeuralSession,
) -> pd.Series:
    missing = sorted(_MANIFEST_COLUMNS - set(manifest.columns))
    if missing:
        raise IBLBlockSidecarError(f"cohort manifest is missing columns: {missing}")
    matching = manifest.loc[manifest["eid"].astype(str).eq(prepared.eid)]
    if len(matching) != 1:
        raise IBLBlockSidecarError(
            "cohort manifest must contain exactly one row for the prepared EID"
        )
    row = matching.iloc[0]
    if _text(row["subject"], name="manifest subject") != prepared.animal_id:
        raise IBLBlockSidecarError(
            "cohort manifest animal does not match the prepared neural session"
        )
    if _text(row["status"], name="manifest status") != "eligible":
        raise IBLBlockSidecarError("prepared EID is not eligible in the cohort manifest")
    if "eligible" in manifest.columns:
        eligible = _binary_column([row["eligible"]], name="manifest eligible")[0]
        if not bool(eligible):
            raise IBLBlockSidecarError(
                "cohort manifest status and eligible flag disagree"
            )
    return row


def _compact_path(manifest_path: Path, value: object) -> tuple[Path, str]:
    relative_text = _text(value, name="compact_table")
    relative = Path(relative_text)
    if relative.is_absolute():
        raise IBLBlockSidecarError("compact_table must be relative to the manifest")
    root = manifest_path.parent.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise IBLBlockSidecarError("compact_table escapes the frozen cohort directory")
    if not resolved.is_file():
        raise IBLBlockSidecarError("compact_table does not exist")
    return resolved, relative.as_posix()


def _validate_neural_view(
    prepared: PreparedIBLNeuralSession,
    *,
    view: str,
    trial_ids: Array,
    stimulus_side: Array,
    choice: Array,
    official_mask: Array,
) -> None:
    table = prepared.trial_table(view)
    missing = sorted(_NEURAL_ALIGNMENT_COLUMNS - set(table.columns))
    if missing:
        raise IBLBlockSidecarError(
            f"{view} neural trial table is missing alignment columns: {missing}"
        )
    neural_ids = _integer_column(table["trial_id"], name=f"{view} trial_id")
    if not np.array_equal(neural_ids, trial_ids):
        raise IBLBlockSidecarError(
            f"{view} trial_id does not align with source_trial_index"
        )
    neural_side = _binary_column(
        table["stimulus_side"], name=f"{view} stimulus_side"
    ).astype(np.int64)
    if not np.array_equal(neural_side, stimulus_side):
        raise IBLBlockSidecarError(
            f"{view} stimulus_side disagrees with compact contrasts"
        )
    neural_choice = _numeric_column(
        table["choice"], name=f"{view} choice", allow_nan=True
    )
    if not _same_numeric(neural_choice, choice):
        raise IBLBlockSidecarError(f"{view} choice disagrees with compact trials")
    neural_official = _binary_column(
        table["official_bwm_mask"], name=f"{view} official_bwm_mask"
    )
    if not np.array_equal(neural_official, official_mask):
        raise IBLBlockSidecarError(
            f"{view} official_bwm_mask disagrees with compact trials"
        )


def load_ibl_block_truth(
    manifest_path: str | Path,
    prepared: PreparedIBLNeuralSession,
    *,
    expected_manifest_sha256: str,
) -> tuple[IBLBlockTruth, Mapping[str, object]]:
    """Load hash-bound block truth after complete cross-artifact alignment.

    The read-only capability may define whole-block train/validation/test splits
    and may score frozen predictions.  It must never be supplied to a gate or
    neural-model fit/predict method.
    """

    if not isinstance(prepared, PreparedIBLNeuralSession):
        raise TypeError("prepared must be a PreparedIBLNeuralSession")
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise IBLBlockSidecarError("cohort manifest does not exist")
    expected_manifest_hash = _sha256(
        expected_manifest_sha256, name="expected_manifest_sha256"
    )
    actual_manifest_hash = _file_sha256(path)
    if actual_manifest_hash != expected_manifest_hash:
        raise IBLBlockSidecarError("cohort manifest SHA-256 mismatch")
    try:
        manifest = pd.read_csv(path, float_precision="round_trip")
    except (OSError, pd.errors.ParserError) as error:
        raise IBLBlockSidecarError("cohort manifest cannot be parsed") from error
    row = _manifest_row(manifest, prepared)
    compact_path, compact_relative = _compact_path(path, row["compact_table"])
    expected_compact_hash = _sha256(
        row["compact_table_sha256"], name="compact_table_sha256"
    )
    actual_compact_hash = _file_sha256(compact_path)
    if actual_compact_hash != expected_compact_hash:
        raise IBLBlockSidecarError("compact table SHA-256 mismatch")
    try:
        compact = pd.read_csv(compact_path, float_precision="round_trip")
    except (OSError, pd.errors.ParserError) as error:
        raise IBLBlockSidecarError("compact trial table cannot be parsed") from error
    missing = sorted(_COMPACT_COLUMNS - set(compact.columns))
    if missing:
        raise IBLBlockSidecarError(f"compact table is missing columns: {missing}")

    trial_ids = _integer_column(
        compact["source_trial_index"], name="source_trial_index"
    )
    prepared_ids = _integer_column(
        prepared.current_trial_ids, name="prepared current_trial_ids"
    )
    if not np.array_equal(trial_ids, prepared_ids):
        raise IBLBlockSidecarError(
            "source_trial_index does not align with prepared current_trial_ids"
        )
    left = _numeric_column(compact["contrastLeft"], name="contrastLeft", allow_nan=True)
    right = _numeric_column(
        compact["contrastRight"], name="contrastRight", allow_nan=True
    )
    left_present = np.isfinite(left)
    right_present = np.isfinite(right)
    if not np.all(left_present ^ right_present):
        raise IBLBlockSidecarError(
            "compact contrasts must identify exactly one stimulus side per trial"
        )
    stimulus_side = left_present.astype(np.int64)
    choice = _numeric_column(compact["choice"], name="compact choice", allow_nan=True)
    official_mask = _binary_column(
        compact["official_bwm_mask"], name="compact official_bwm_mask"
    )
    probability = _numeric_column(
        compact["probabilityLeft"], name="probabilityLeft", allow_nan=False
    )

    for view in ("stimulus_pre", "movement_pre"):
        _validate_neural_view(
            prepared,
            view=view,
            trial_ids=trial_ids,
            stimulus_side=stimulus_side,
            choice=choice,
            official_mask=official_mask,
        )
    switches = np.zeros(len(probability), dtype=bool)
    switches[1:] = ~np.isclose(
        probability[1:], probability[:-1], atol=1e-6, rtol=0.0
    )
    truth = IBLBlockTruth(
        trial_ids=trial_ids,
        probability_left=probability,
        block_switch=switches,
        official_bwm_mask=official_mask,
    )
    alignment_hash = _alignment_fingerprint(
        trial_ids,
        stimulus_side,
        choice,
        official_mask,
    )
    provenance: dict[str, object] = {
        "capability": "ibl_block_truth_sidecar",
        "access_scope": "whole_block_split_and_postfit_evaluation_only",
        "eligible_for_whole_block_split": True,
        "eligible_for_postfit_evaluation": True,
        "eligible_for_gate_input": False,
        "eligible_for_model_input": False,
        "eid": prepared.eid,
        "animal_id": prepared.animal_id,
        "cohort_manifest_path": str(path),
        "cohort_manifest_sha256": actual_manifest_hash,
        "compact_table": compact_relative,
        "compact_table_sha256": actual_compact_hash,
        "trial_count": int(len(trial_ids)),
        "source_trial_index_policy": "exact_prepared_current_trial_id_match",
        "aligned_neural_views": ("stimulus_pre", "movement_pre"),
        "alignment_fingerprint": alignment_hash,
        "truth_fingerprint": truth.fingerprint,
    }
    for key in (
        "cohort_id",
        "dataset_uuid",
        "dataset_revision",
        "dataset_hash",
        "dataset_qc",
        "bwm_repository_commit",
    ):
        if key in row.index and not pd.isna(row[key]) and str(row[key]).strip():
            provenance[key] = str(row[key]).strip()
    return truth, MappingProxyType(provenance)


__all__ = [
    "IBLBlockSidecarError",
    "IBLBlockTruth",
    "load_ibl_block_truth",
]
