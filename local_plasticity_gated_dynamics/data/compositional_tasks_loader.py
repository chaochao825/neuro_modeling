"""Fail-closed loader and split utilities for the CompositionalTasks release.

The loader deliberately has no download fallback and no synthetic-data path.
It first verifies the reviewed Nature/Figshare provenance and all six official
MAT files, then verifies a hash-bound canonical conversion before exposing
neural counts.  A caller must pin the canonical manifest SHA-256 explicitly.

Canonical manifest schema
-------------------------
The UTF-8 JSON manifest has exactly four top-level fields:

``schema_version``
    ``"compositional_tasks_canonical_manifest_v1"``.
``source``
    The exact publication, Figshare, license, and code provenance constants
    declared below.
``official_files``
    The exact six reviewed Figshare file records, stored under ``raw/``.
``canonical``
    Hash/size descriptors for ``trials.csv``, ``units.csv``, and one ``.npz``
    asset per session.  It also verifies the exact conversion-code file.

Each session asset contains integer ``counts[trial,time,unit]``, finite
``inputs[trial,time,input]``, string ``trial_ids[trial]``, and string
``unit_ids[unit]`` arrays.  The tables and arrays must agree exactly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


Array = np.ndarray
PAPER_URL = "https://www.nature.com/articles/s41586-025-09805-2"
FIGSHARE_ARTICLE_ID = 30276238
FIGSHARE_DOI = "10.6084/m9.figshare.30276238.v1"
FIGSHARE_LICENSE = "CC BY 4.0"
CODE_DOI = "10.5281/zenodo.17274345"
FIGSHARE_DOWNLOAD_TEMPLATE = "https://ndownloader.figshare.com/files/{file_id}"
MANIFEST_SCHEMA = "compositional_tasks_canonical_manifest_v1"

_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_TRIAL_COLUMNS = frozenset(
    {
        "animal_id",
        "session_id",
        "trial_id",
        "trial_order",
        "block_id",
        "composition_id",
        "cue",
        "behavior",
        "stimulus_id",
        "action_id",
    }
)
_REQUIRED_UNIT_COLUMNS = frozenset({"session_id", "unit_id", "region"})
_FORBIDDEN_BELIEF_COLUMNS = frozenset(
    {
        "composition_id",
        "context",
        "context_id",
        "ground_truth",
        "target",
        "target_id",
        "task_id",
        "true_context",
        "true_task",
    }
)
_START_TOKEN = "__PAST_ONLY_SEQUENCE_START__"


class CompositionalTasksDataError(ValueError):
    """Raised when public-data bytes, schema, or causal contracts are invalid."""


@dataclass(frozen=True, slots=True)
class OfficialFileSpec:
    name: str
    file_id: int
    size: int
    md5: str

    @property
    def download_url(self) -> str:
        return FIGSHARE_DOWNLOAD_TEMPLATE.format(file_id=self.file_id)

    def manifest_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": f"raw/{self.name}",
            "file_id": self.file_id,
            "size": self.size,
            "md5": self.md5,
            "download_url": self.download_url,
        }


OFFICIAL_FILE_SPECS: Mapping[str, OfficialFileSpec] = MappingProxyType(
    {
        item.name: item
        for item in (
            OfficialFileSpec(
                "BhvData.mat",
                58487176,
                1_486_766,
                "e1510efe3ea89e66d9c16767754f93fa",
            ),
            OfficialFileSpec(
                "PFC_ClassifierData.mat",
                58487158,
                2_559_898,
                "04e42460ccae245ec16fcd9801ddea4a",
            ),
            OfficialFileSpec(
                "PFC_ClassifierLearningData.mat",
                58487155,
                1_732_080,
                "aa1232b20f6d4a66185899ea00c73c38",
            ),
            OfficialFileSpec(
                "PFC_CompressionData.mat",
                58487140,
                683_254,
                "c5e25e3507bcbec75d51695e4810806f",
            ),
            OfficialFileSpec(
                "DynamicTransformationData.mat",
                58487173,
                21_835_139,
                "d73c9aeea6a61268ee880b192c38006b",
            ),
            OfficialFileSpec(
                "GLMdata.mat",
                58487164,
                288_180_371,
                "ee2eb0897709364cf906c42ebf701ed4a",
            ),
        )
    }
)

OFFICIAL_SOURCE_PROVENANCE: Mapping[str, object] = MappingProxyType(
    {
        "paper_url": PAPER_URL,
        "figshare_article_id": FIGSHARE_ARTICLE_ID,
        "figshare_doi": FIGSHARE_DOI,
        "license": FIGSHARE_LICENSE,
        "code_doi": CODE_DOI,
    }
)


def _file_digest(path: Path, algorithm: Literal["md5", "sha256"]) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompositionalTasksDataError(f"{name} must be a non-empty string")
    return value.strip()


def _safe_file(root: Path, record: Mapping[str, object], *, name: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise CompositionalTasksDataError(
            f"{name} descriptor must contain exactly path/sha256/bytes"
        )
    relative = Path(_text(record["path"], name=f"{name} path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise CompositionalTasksDataError(f"{name} path must remain under data root")
    digest = str(record["sha256"])
    if _HEX64.fullmatch(digest) is None:
        raise CompositionalTasksDataError(f"{name} SHA-256 is invalid")
    try:
        expected_bytes = int(record["bytes"])
    except (TypeError, ValueError) as error:
        raise CompositionalTasksDataError(f"{name} bytes must be an integer") from error
    if expected_bytes < 1:
        raise CompositionalTasksDataError(f"{name} bytes must be positive")
    path = root / relative
    root_resolved = root.resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root_resolved)
    ):
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_bytes:
        raise CompositionalTasksDataError(f"{name} byte size mismatch")
    if _file_digest(path, "sha256") != digest:
        raise CompositionalTasksDataError(f"{name} SHA-256 mismatch")
    return path


def _official_records() -> list[dict[str, object]]:
    return [
        OFFICIAL_FILE_SPECS[name].manifest_record()
        for name in sorted(OFFICIAL_FILE_SPECS)
    ]


@dataclass(frozen=True, slots=True)
class CompositionalSourceReceipt:
    manifest_sha256: str
    manifest_schema: str
    official_file_md5: Mapping[str, str]
    official_file_bytes: Mapping[str, int]
    canonical_file_sha256: Mapping[str, str]
    conversion_code_sha256: str
    source_verified: bool = True
    canonical_verified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "official_file_md5", MappingProxyType(dict(self.official_file_md5))
        )
        object.__setattr__(
            self,
            "official_file_bytes",
            MappingProxyType(
                {key: int(value) for key, value in self.official_file_bytes.items()}
            ),
        )
        object.__setattr__(
            self,
            "canonical_file_sha256",
            MappingProxyType(dict(self.canonical_file_sha256)),
        )


@dataclass(frozen=True, slots=True)
class OfficialSourceReceipt:
    source_provenance: Mapping[str, object]
    official_file_md5: Mapping[str, str]
    official_file_bytes: Mapping[str, int]
    source_verified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_provenance",
            MappingProxyType(dict(self.source_provenance)),
        )
        object.__setattr__(
            self,
            "official_file_md5",
            MappingProxyType(dict(self.official_file_md5)),
        )
        object.__setattr__(
            self,
            "official_file_bytes",
            MappingProxyType(
                {key: int(value) for key, value in self.official_file_bytes.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class CompositionalSession:
    session_id: str
    animal_id: str
    counts: Array
    inputs: Array
    trial_ids: Array
    unit_ids: Array
    unit_regions: tuple[str, ...]

    def __post_init__(self) -> None:
        session_id = _text(self.session_id, name="session_id")
        animal_id = _text(self.animal_id, name="animal_id")
        counts = np.asarray(self.counts)
        inputs = np.asarray(self.inputs, dtype=float)
        trial_ids = np.asarray(self.trial_ids, dtype=str)
        unit_ids = np.asarray(self.unit_ids, dtype=str)
        if (
            counts.ndim != 3
            or counts.shape[1] < 2
            or counts.shape[2] < 1
            or counts.dtype.kind not in {"i", "u"}
            or np.any(counts < 0)
        ):
            raise CompositionalTasksDataError(
                "counts must be non-negative integer [trial,time>=2,unit] data"
            )
        if (
            inputs.ndim != 3
            or inputs.shape[:2] != counts.shape[:2]
            or inputs.shape[2] < 1
            or not np.isfinite(inputs).all()
        ):
            raise CompositionalTasksDataError(
                "inputs must be finite [trial,time,input] data aligned to counts"
            )
        if trial_ids.shape != (counts.shape[0],) or len(set(trial_ids)) != len(
            trial_ids
        ):
            raise CompositionalTasksDataError(
                "trial_ids must be a unique vector aligned to counts"
            )
        if unit_ids.shape != (counts.shape[2],) or len(set(unit_ids)) != len(unit_ids):
            raise CompositionalTasksDataError(
                "unit_ids must be a unique vector aligned to counts"
            )
        regions = tuple(_text(value, name="unit region") for value in self.unit_regions)
        if len(regions) != counts.shape[2]:
            raise CompositionalTasksDataError(
                "unit_regions must provide one region per unit"
            )
        frozen_arrays = {
            "counts": np.asarray(counts, dtype=np.int64).copy(),
            "inputs": inputs.copy(),
            "trial_ids": trial_ids.copy(),
            "unit_ids": unit_ids.copy(),
        }
        for name, array in frozen_arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "animal_id", animal_id)
        object.__setattr__(self, "unit_regions", regions)


@dataclass(frozen=True, slots=True)
class CompositionalDataset:
    sessions: tuple[CompositionalSession, ...]
    trials: pd.DataFrame
    units: pd.DataFrame
    receipt: CompositionalSourceReceipt

    def __post_init__(self) -> None:
        if not self.sessions:
            raise CompositionalTasksDataError("dataset must contain at least one session")
        session_ids = tuple(session.session_id for session in self.sessions)
        if len(set(session_ids)) != len(session_ids):
            raise CompositionalTasksDataError("session IDs must be unique")
        object.__setattr__(self, "trials", self.trials.copy(deep=True))
        object.__setattr__(self, "units", self.units.copy(deep=True))


def _validate_table(
    frame: pd.DataFrame,
    *,
    required: frozenset[str],
    table_name: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CompositionalTasksDataError(
            f"{table_name} is missing required columns: {missing}"
        )
    if frame.empty:
        raise CompositionalTasksDataError(f"{table_name} must not be empty")
    required_frame = frame[list(sorted(required))]
    if required_frame.isna().any().any():
        raise CompositionalTasksDataError(
            f"{table_name} subject/session/block/composition schema contains missing values"
        )
    if required_frame.map(lambda value: not str(value).strip()).any().any():
        raise CompositionalTasksDataError(
            f"{table_name} subject/session/block/composition schema contains empty values"
        )


def _validate_official_source(root: Path, records: object) -> tuple[dict[str, str], dict[str, int]]:
    if not isinstance(records, list):
        raise CompositionalTasksDataError("official_files must be a reviewed list")
    if records != _official_records():
        raise CompositionalTasksDataError(
            "official_files differ from the reviewed Figshare contract"
        )
    raw_root = root / "raw"
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    expected_names = set(OFFICIAL_FILE_SPECS)
    discovered = {
        path.name
        for path in raw_root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if discovered != expected_names:
        raise CompositionalTasksDataError(
            "raw source coverage mismatch; "
            f"missing={sorted(expected_names - discovered)!r}, "
            f"extra={sorted(discovered - expected_names)!r}"
        )
    digests: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for name, spec in OFFICIAL_FILE_SPECS.items():
        path = raw_root / name
        if path.is_symlink() or path.stat().st_size != spec.size:
            raise CompositionalTasksDataError(
                f"official source size mismatch: {name}"
            )
        digest = _file_digest(path, "md5")
        if digest != spec.md5:
            raise CompositionalTasksDataError(
                f"official source MD5 mismatch: {name}"
            )
        digests[name] = digest
        sizes[name] = path.stat().st_size
    return digests, sizes


def validate_official_compositional_source(
    data_root: str | Path,
) -> OfficialSourceReceipt:
    """Verify the exact six-file Figshare release before canonical conversion."""

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digests, sizes = _validate_official_source(root, _official_records())
    return OfficialSourceReceipt(
        source_provenance=OFFICIAL_SOURCE_PROVENANCE,
        official_file_md5=digests,
        official_file_bytes=sizes,
    )


def load_compositional_tasks(
    data_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
) -> CompositionalDataset:
    """Load only a byte-verified official source plus canonical conversion."""

    root = Path(data_root)
    manifest = Path(manifest_path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not manifest.is_file() or manifest.is_symlink():
        raise FileNotFoundError(manifest)
    if _HEX64.fullmatch(str(expected_manifest_sha256)) is None:
        raise CompositionalTasksDataError(
            "expected_manifest_sha256 must be lowercase SHA-256"
        )
    actual_manifest_sha = _file_digest(manifest, "sha256")
    if actual_manifest_sha != expected_manifest_sha256:
        raise CompositionalTasksDataError("canonical manifest SHA-256 mismatch")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompositionalTasksDataError(
            "canonical manifest must be valid UTF-8 JSON"
        ) from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "source",
        "official_files",
        "canonical",
    }:
        raise CompositionalTasksDataError("canonical manifest top-level schema is wrong")
    if payload["schema_version"] != MANIFEST_SCHEMA:
        raise CompositionalTasksDataError("canonical manifest version is wrong")
    if payload["source"] != dict(OFFICIAL_SOURCE_PROVENANCE):
        raise CompositionalTasksDataError(
            "manifest provenance differs from the reviewed publication release"
        )
    official_md5, official_bytes = _validate_official_source(
        root, payload["official_files"]
    )

    canonical = payload["canonical"]
    if not isinstance(canonical, Mapping) or set(canonical) != {
        "trials",
        "units",
        "sessions",
        "conversion_code",
    }:
        raise CompositionalTasksDataError("canonical conversion schema is wrong")
    conversion_path = _safe_file(
        root, canonical["conversion_code"], name="conversion_code"
    )
    conversion_sha = _file_digest(conversion_path, "sha256")
    trials_path = _safe_file(root, canonical["trials"], name="trials")
    units_path = _safe_file(root, canonical["units"], name="units")
    try:
        trials = pd.read_csv(trials_path, keep_default_na=False)
        units = pd.read_csv(units_path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise CompositionalTasksDataError("canonical CSV tables cannot be parsed") from error
    _validate_table(trials, required=_REQUIRED_TRIAL_COLUMNS, table_name="trials")
    _validate_table(units, required=_REQUIRED_UNIT_COLUMNS, table_name="units")
    if trials.duplicated(["session_id", "trial_id"]).any():
        raise CompositionalTasksDataError("trial IDs must be unique within session")
    numeric_order = pd.to_numeric(trials["trial_order"], errors="coerce")
    if numeric_order.isna().any() or (numeric_order < 0).any():
        raise CompositionalTasksDataError(
            "trial_order must contain non-negative numbers"
        )
    if trials.assign(_order=numeric_order).duplicated(
        ["session_id", "_order"]
    ).any():
        raise CompositionalTasksDataError("trial_order must be unique within session")
    if units.duplicated(["session_id", "unit_id"]).any():
        raise CompositionalTasksDataError("unit IDs must be unique within session")

    session_records = canonical["sessions"]
    if not isinstance(session_records, list) or not session_records:
        raise CompositionalTasksDataError(
            "canonical sessions must be a non-empty list"
        )
    expected_session_schema = {
        "session_id",
        "animal_id",
        "asset",
        "counts_key",
        "inputs_key",
        "trial_ids_key",
        "unit_ids_key",
    }
    sessions: list[CompositionalSession] = []
    canonical_hashes = {
        "trials": str(canonical["trials"]["sha256"]),
        "units": str(canonical["units"]["sha256"]),
    }
    seen_sessions: set[str] = set()
    table_sessions = set(trials["session_id"].astype(str))
    if table_sessions != set(units["session_id"].astype(str)):
        raise CompositionalTasksDataError(
            "trials and units must cover exactly the same sessions"
        )
    for index, raw_record in enumerate(session_records):
        if not isinstance(raw_record, Mapping) or set(raw_record) != expected_session_schema:
            raise CompositionalTasksDataError(
                f"canonical session record {index} has the wrong schema"
            )
        session_id = _text(raw_record["session_id"], name="session_id")
        animal_id = _text(raw_record["animal_id"], name="animal_id")
        if session_id in seen_sessions:
            raise CompositionalTasksDataError("canonical session IDs are duplicated")
        seen_sessions.add(session_id)
        path = _safe_file(root, raw_record["asset"], name=f"session {session_id}")
        canonical_hashes[f"session:{session_id}"] = str(
            raw_record["asset"]["sha256"]
        )
        keys = {
            role: _text(raw_record[f"{role}_key"], name=f"{role}_key")
            for role in ("counts", "inputs", "trial_ids", "unit_ids")
        }
        try:
            with np.load(path, allow_pickle=False) as bundle:
                if set(bundle.files) != set(keys.values()):
                    raise CompositionalTasksDataError(
                        f"session {session_id} asset keys differ from manifest"
                    )
                arrays = {role: bundle[key].copy() for role, key in keys.items()}
        except (OSError, ValueError) as error:
            raise CompositionalTasksDataError(
                f"session {session_id} asset cannot be parsed"
            ) from error
        trial_rows = (
            trials.loc[trials["session_id"].astype(str).eq(session_id)]
            .assign(_order=lambda frame: pd.to_numeric(frame["trial_order"]))
            .sort_values("_order", kind="mergesort")
        )
        unit_rows = units.loc[units["session_id"].astype(str).eq(session_id)]
        if trial_rows.empty or unit_rows.empty:
            raise CompositionalTasksDataError(
                f"session {session_id} is absent from canonical tables"
            )
        animals = set(trial_rows["animal_id"].astype(str))
        if animals != {animal_id}:
            raise CompositionalTasksDataError(
                f"session {session_id} animal provenance is inconsistent"
            )
        trial_ids = np.asarray(arrays["trial_ids"], dtype=str)
        unit_ids = np.asarray(arrays["unit_ids"], dtype=str)
        if not np.array_equal(
            trial_ids, trial_rows["trial_id"].astype(str).to_numpy()
        ):
            raise CompositionalTasksDataError(
                f"session {session_id} trial table/array order differs"
            )
        unit_lookup = unit_rows.assign(
            _unit_id=unit_rows["unit_id"].astype(str)
        ).set_index("_unit_id", drop=False)
        if set(unit_ids) != set(unit_lookup.index):
            raise CompositionalTasksDataError(
                f"session {session_id} unit table/array coverage differs"
            )
        regions = tuple(
            str(unit_lookup.loc[unit_id, "region"]).strip() for unit_id in unit_ids
        )
        sessions.append(
            CompositionalSession(
                session_id=session_id,
                animal_id=animal_id,
                counts=arrays["counts"],
                inputs=arrays["inputs"],
                trial_ids=trial_ids,
                unit_ids=unit_ids,
                unit_regions=regions,
            )
        )
    if seen_sessions != table_sessions:
        raise CompositionalTasksDataError(
            "canonical manifest/table session coverage differs"
        )
    receipt = CompositionalSourceReceipt(
        manifest_sha256=actual_manifest_sha,
        manifest_schema=MANIFEST_SCHEMA,
        official_file_md5=official_md5,
        official_file_bytes=official_bytes,
        canonical_file_sha256=canonical_hashes,
        conversion_code_sha256=conversion_sha,
    )
    return CompositionalDataset(tuple(sessions), trials, units, receipt)


@dataclass(frozen=True, slots=True)
class GroupedTrialSplit:
    kind: Literal["block", "session", "animal", "composition"]
    heldout_values: tuple[object, ...]
    train_indices: Array
    test_indices: Array

    def __post_init__(self) -> None:
        train = np.asarray(self.train_indices, dtype=int).copy()
        test = np.asarray(self.test_indices, dtype=int).copy()
        if (
            train.ndim != 1
            or test.ndim != 1
            or min(train.size, test.size) < 1
            or np.intersect1d(train, test).size
        ):
            raise CompositionalTasksDataError(
                "grouped split must contain disjoint non-empty train/test indices"
            )
        train.setflags(write=False)
        test.setflags(write=False)
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "test_indices", test)


def _leave_one_group_out(
    trials: pd.DataFrame,
    groups: Sequence[object],
    *,
    kind: Literal["block", "session", "animal", "composition"],
) -> tuple[GroupedTrialSplit, ...]:
    if len(groups) != len(trials):
        raise ValueError("groups must match trials")
    keys = list(groups)
    try:
        unique = sorted(set(keys), key=repr)
    except TypeError as error:
        raise CompositionalTasksDataError("split labels must be hashable") from error
    if len(unique) < 2:
        raise CompositionalTasksDataError(
            f"{kind} split requires at least two complete groups"
        )
    result = []
    for heldout in unique:
        mask = np.asarray([value == heldout for value in keys], dtype=bool)
        result.append(
            GroupedTrialSplit(
                kind=kind,
                heldout_values=(heldout,),
                train_indices=np.flatnonzero(~mask),
                test_indices=np.flatnonzero(mask),
            )
        )
    return tuple(result)


def leave_one_block_out_splits(
    trials: pd.DataFrame,
) -> tuple[GroupedTrialSplit, ...]:
    """Leave out one whole session/block pair at a time."""

    _validate_table(trials, required=_REQUIRED_TRIAL_COLUMNS, table_name="trials")
    groups = list(
        zip(
            trials["session_id"].astype(str),
            trials["block_id"].astype(str),
            strict=True,
        )
    )
    return _leave_one_group_out(trials, groups, kind="block")


def leave_one_session_out_splits(
    trials: pd.DataFrame,
) -> tuple[GroupedTrialSplit, ...]:
    _validate_table(trials, required=_REQUIRED_TRIAL_COLUMNS, table_name="trials")
    return _leave_one_group_out(
        trials, trials["session_id"].astype(str).tolist(), kind="session"
    )


def leave_one_animal_out_splits(
    trials: pd.DataFrame,
) -> tuple[GroupedTrialSplit, ...]:
    _validate_table(trials, required=_REQUIRED_TRIAL_COLUMNS, table_name="trials")
    return _leave_one_group_out(
        trials, trials["animal_id"].astype(str).tolist(), kind="animal"
    )


def leave_one_composition_out_splits(
    trials: pd.DataFrame,
) -> tuple[GroupedTrialSplit, ...]:
    _validate_table(trials, required=_REQUIRED_TRIAL_COLUMNS, table_name="trials")
    return _leave_one_group_out(
        trials, trials["composition_id"].astype(str).tolist(), kind="composition"
    )


def _indices(values: Iterable[int], *, n_rows: int, name: str) -> Array:
    raw = np.asarray(list(values))
    if raw.ndim != 1 or np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.integer
    ):
        raise TypeError(f"{name} must contain integer row positions")
    result = raw.astype(int, copy=False)
    if (
        result.size < 1
        or np.any(result < 0)
        or np.any(result >= n_rows)
        or np.unique(result).size != result.size
    ):
        raise ValueError(f"{name} must contain unique in-range row positions")
    return result


def _label_key(value: object) -> tuple[str, str]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        raise CompositionalTasksDataError("belief labels must be finite and non-missing")
    return type(value).__qualname__, repr(value)


@dataclass(frozen=True, slots=True)
class BeliefProvenanceReceipt:
    fit_trial_keys: tuple[tuple[str, str], ...]
    evaluated_trial_keys: tuple[tuple[str, str], ...]
    source_columns: tuple[str, ...]
    fit_label_column: str
    feature_lag_trials: int
    uses_current_trial_fields: bool
    uses_future_trials: bool
    accessed_test_truth: bool
    fit_design_sha256: str
    checkpoint_sha256: str
    fit_history_scope: str
    prediction_history_scope: str
    fit_preprocessing_heldout_independent: bool


@dataclass(frozen=True, slots=True)
class BeliefTrajectory:
    probabilities: Array
    trial_keys: tuple[tuple[str, str], ...]
    classes: tuple[object, ...]
    receipt: BeliefProvenanceReceipt

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=float).copy()
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != len(self.trial_keys)
            or probabilities.shape[1] != len(self.classes)
            or not np.isfinite(probabilities).all()
            or np.any((probabilities <= 0.0) | (probabilities >= 1.0))
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
        ):
            raise CompositionalTasksDataError(
                "beliefs must be finite strictly-soft row probabilities"
            )
        probabilities.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)


class PastOnlyBeliefEstimator:
    """Train-only ridge-softmax belief model using strictly lag-one fields.

    Training composition/task labels fit the classifier, but ``predict`` does
    not read that column.  Fit-time history is constructed only from the
    supplied training rows, so a held-out row cannot change lag features,
    categories, numeric normalization, or the fitted checkpoint.  Prediction
    history may use all causally prior rows in the supplied frame, while every
    cue and behavior feature remains shifted by one row within session.
    """

    def __init__(
        self,
        *,
        cue_columns: Sequence[str],
        behavior_columns: Sequence[str],
        numeric_columns: Sequence[str] = (),
        group_columns: Sequence[str] = ("session_id",),
        order_column: str = "trial_order",
        ridge: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        cue = tuple(_text(value, name="cue column") for value in cue_columns)
        behavior = tuple(
            _text(value, name="behavior column") for value in behavior_columns
        )
        numeric = tuple(_text(value, name="numeric column") for value in numeric_columns)
        groups = tuple(_text(value, name="group column") for value in group_columns)
        source = cue + behavior
        if not source or len(set(source)) != len(source):
            raise ValueError("cue/behavior columns must be non-empty and unique")
        if not set(numeric) <= set(source):
            raise ValueError("numeric_columns must be a subset of source columns")
        if set(source) & _FORBIDDEN_BELIEF_COLUMNS:
            raise ValueError("belief sources cannot include truth/context/task columns")
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("group_columns must be non-empty and unique")
        if (
            not np.isfinite(ridge)
            or ridge < 0
            or not np.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("ridge and temperature must be finite and valid")
        self.cue_columns = cue
        self.behavior_columns = behavior
        self.numeric_columns = numeric
        self.group_columns = groups
        self.order_column = _text(order_column, name="order_column")
        self.ridge = float(ridge)
        self.temperature = float(temperature)
        self._fitted = False

    @property
    def source_columns(self) -> tuple[str, ...]:
        return self.cue_columns + self.behavior_columns

    def _validate_frame(self, frame: pd.DataFrame, *, include_label: str | None) -> None:
        required = {
            *self.source_columns,
            *self.group_columns,
            self.order_column,
            "session_id",
            "trial_id",
        }
        if include_label is not None:
            required.add(include_label)
        missing = sorted(required - set(frame))
        if missing:
            raise CompositionalTasksDataError(
                f"belief frame is missing columns: {missing}"
            )
        if frame[list(required)].isna().any().any():
            raise CompositionalTasksDataError(
                "belief source/order/group/label columns cannot be missing"
            )
        order = pd.to_numeric(frame[self.order_column], errors="coerce")
        if order.isna().any():
            raise CompositionalTasksDataError("belief trial order must be numeric")

    def _lagged(self, frame: pd.DataFrame) -> dict[str, Array]:
        n_rows = len(frame)
        result: dict[str, Array] = {}
        group_values = [
            tuple(row)
            for row in frame[list(self.group_columns)].itertuples(index=False, name=None)
        ]
        order = pd.to_numeric(frame[self.order_column]).to_numpy(dtype=float)
        grouped: dict[tuple[object, ...], list[int]] = {}
        for index, group in enumerate(group_values):
            grouped.setdefault(group, []).append(index)
        for column in self.source_columns:
            if column in self.numeric_columns:
                values: Array = np.zeros(n_rows, dtype=float)
            else:
                values = np.empty(n_rows, dtype=object)
                values[:] = _START_TOKEN
            source = frame[column].to_numpy(copy=False)
            for positions in grouped.values():
                sorted_positions = sorted(positions, key=lambda item: (order[item], item))
                for previous, current in zip(
                    sorted_positions[:-1], sorted_positions[1:], strict=True
                ):
                    values[current] = source[previous]
            if column in self.numeric_columns:
                try:
                    values = np.asarray(values, dtype=float)
                except (TypeError, ValueError) as error:
                    raise CompositionalTasksDataError(
                        f"numeric belief source {column!r} is not numeric"
                    ) from error
                if not np.isfinite(values).all():
                    raise CompositionalTasksDataError(
                        f"numeric belief source {column!r} is not finite"
                    )
            result[column] = values
        return result

    def _trial_keys(
        self, frame: pd.DataFrame, indices: Array
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (
                str(frame.iloc[index]["session_id"]),
                str(frame.iloc[index]["trial_id"]),
            )
            for index in indices
        )

    def _fit_schema(self, lagged: Mapping[str, Array], train: Array) -> None:
        self.categories_: dict[str, tuple[str, ...]] = {}
        self.numeric_mean_: dict[str, float] = {}
        self.numeric_scale_: dict[str, float] = {}
        for column in self.source_columns:
            values = lagged[column][train]
            if column in self.numeric_columns:
                mean = float(np.mean(values.astype(float)))
                scale = float(np.std(values.astype(float), ddof=0))
                self.numeric_mean_[column] = mean
                self.numeric_scale_[column] = scale if scale >= 1e-12 else 1.0
            else:
                self.categories_[column] = tuple(
                    sorted({str(value) for value in values}, key=repr)
                )

    def _design(self, lagged: Mapping[str, Array]) -> Array:
        columns: list[Array] = [np.ones((len(next(iter(lagged.values()))), 1))]
        for name in self.source_columns:
            values = lagged[name]
            if name in self.numeric_columns:
                columns.append(
                    (
                        (values.astype(float) - self.numeric_mean_[name])
                        / self.numeric_scale_[name]
                    )[:, None]
                )
            else:
                text = np.asarray([str(value) for value in values], dtype=object)
                categories = self.categories_[name]
                columns.append(
                    np.column_stack([text == category for category in categories]).astype(
                        float
                    )
                )
        return np.column_stack(columns)

    def fit(
        self,
        trials: pd.DataFrame,
        train_indices: Iterable[int],
        *,
        label_column: str,
    ) -> "PastOnlyBeliefEstimator":
        label = _text(label_column, name="label_column")
        if label in self.source_columns:
            raise ValueError("fit label cannot also be a belief source")
        train = _indices(train_indices, n_rows=len(trials), name="train_indices")
        fit_frame = trials.iloc[train].reset_index(drop=True)
        self._validate_frame(fit_frame, include_label=label)
        lagged = self._lagged(fit_frame)
        labels = fit_frame[label].to_numpy(copy=False)
        classes = tuple(
            value
            for _, value in sorted(
                {_label_key(value): value for value in labels}.items(),
                key=lambda item: item[0],
            )
        )
        if len(classes) < 2:
            raise CompositionalTasksDataError(
                "belief training requires at least two task classes"
            )
        class_index = {_label_key(value): index for index, value in enumerate(classes)}
        fit_rows = np.arange(train.size, dtype=int)
        self._fit_schema(lagged, fit_rows)
        fit_design = self._design(lagged)
        target = np.zeros((train.size, len(classes)), dtype=float)
        for row, value in enumerate(labels):
            target[row, class_index[_label_key(value)]] = 1.0
        gram = fit_design.T @ fit_design
        regularizer = self.ridge * np.eye(gram.shape[0], dtype=float)
        regularizer[0, 0] = 0.0
        self.coefficients_ = np.linalg.pinv(gram + regularizer) @ fit_design.T @ target
        self.classes_ = classes
        self.label_column_ = label
        self.fit_indices_ = train.copy()
        self.fit_trial_keys_ = self._trial_keys(trials, train)
        self.fit_design_sha256_ = hashlib.sha256(
            np.ascontiguousarray(fit_design).tobytes()
        ).hexdigest()
        checkpoint = {
            "classes": [repr(value) for value in classes],
            "source_columns": self.source_columns,
            "numeric_columns": self.numeric_columns,
            "categories": self.categories_,
            "numeric_mean": self.numeric_mean_,
            "numeric_scale": self.numeric_scale_,
            "coefficients_sha256": hashlib.sha256(
                np.ascontiguousarray(self.coefficients_).tobytes()
            ).hexdigest(),
            "fit_trial_keys": self.fit_trial_keys_,
            "lag": 1,
            "fit_history_scope": "training_rows_only_within_group",
            "prediction_history_scope": "all_causally_prior_rows_within_group",
            "fit_preprocessing_heldout_independent": True,
        }
        self.checkpoint_sha256_ = _canonical_json_sha256(checkpoint)
        self._fitted = True
        return self

    def predict(
        self,
        trials: pd.DataFrame,
        indices: Iterable[int] | None = None,
    ) -> BeliefTrajectory:
        if not self._fitted:
            raise RuntimeError("belief estimator must be fit before predict")
        # The fit label is intentionally not required or read here.
        self._validate_frame(trials, include_label=None)
        selected = (
            np.arange(len(trials), dtype=int)
            if indices is None
            else _indices(indices, n_rows=len(trials), name="indices")
        )
        logits = self._design(self._lagged(trials))[selected] @ self.coefficients_
        logits = logits / self.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        probability = np.exp(np.clip(logits, -700.0, 0.0))
        probability = np.clip(probability, 1e-12, None)
        probability /= probability.sum(axis=1, keepdims=True)
        probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
        probability /= probability.sum(axis=1, keepdims=True)
        evaluated = self._trial_keys(trials, selected)
        receipt = BeliefProvenanceReceipt(
            fit_trial_keys=self.fit_trial_keys_,
            evaluated_trial_keys=evaluated,
            source_columns=self.source_columns,
            fit_label_column=self.label_column_,
            feature_lag_trials=1,
            uses_current_trial_fields=False,
            uses_future_trials=False,
            accessed_test_truth=False,
            fit_design_sha256=self.fit_design_sha256_,
            checkpoint_sha256=self.checkpoint_sha256_,
            fit_history_scope="training_rows_only_within_group",
            prediction_history_scope="all_causally_prior_rows_within_group",
            fit_preprocessing_heldout_independent=True,
        )
        return BeliefTrajectory(
            probabilities=probability,
            trial_keys=evaluated,
            classes=self.classes_,
            receipt=receipt,
        )


__all__ = [
    "CODE_DOI",
    "FIGSHARE_ARTICLE_ID",
    "FIGSHARE_DOI",
    "FIGSHARE_LICENSE",
    "MANIFEST_SCHEMA",
    "OFFICIAL_FILE_SPECS",
    "OFFICIAL_SOURCE_PROVENANCE",
    "PAPER_URL",
    "BeliefProvenanceReceipt",
    "BeliefTrajectory",
    "CompositionalDataset",
    "CompositionalSession",
    "CompositionalSourceReceipt",
    "CompositionalTasksDataError",
    "GroupedTrialSplit",
    "OfficialSourceReceipt",
    "OfficialFileSpec",
    "PastOnlyBeliefEstimator",
    "leave_one_animal_out_splits",
    "leave_one_block_out_splits",
    "leave_one_composition_out_splits",
    "leave_one_session_out_splits",
    "load_compositional_tasks",
    "validate_official_compositional_source",
]
