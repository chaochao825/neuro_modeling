"""Fail-closed Exp26-to-Exp27 selector dataset contracts.

The selector is deliberately trained on *discovery* generators from other
network seeds and evaluated on *held-out* generators from the outer seed.  No
utility, normalization statistic, or task row from the outer seed enters the
training fold.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CANDIDATE_MODES = ("routing", "gain", "low_rank")
EXPECTED_PANEL_MODES = ("frozen", *CANDIDATE_MODES, "rgl")
RAW_FEATURE_NAMES = (
    "chi",
    "log_state_demand",
    "log_input_demand",
    "log2_transition_rank",
    "log2_input_rank",
    "scaled_delay",
    "log_noise",
)
NORMALIZED_FEATURE_NAMES = (*RAW_FEATURE_NAMES, "bias")
DEMAND_FLOOR = 1e-12
NOISE_FLOOR = 1e-12
DELAY_SCALE_STEPS = 4.0
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

_REQUIRED_COLUMNS = {
    "seed",
    "generator_id",
    "generator_split",
    "actuator_mode",
    "chi",
    "state_demand",
    "input_demand",
    "transition_rank",
    "input_rank",
    "delay",
    "noise_std",
    "alpha",
    "validation_balanced_accuracy",
    "test_balanced_accuracy",
    "status",
    "functional_budget_valid",
    "profile",
    "manifest_hash",
    "formal_config_sha256",
    "registered_manifest_sha256",
}
_MODE_INVARIANT_COLUMNS = (
    "chi",
    "state_demand",
    "input_demand",
    "transition_rank",
    "input_rank",
    "delay",
    "noise_std",
    "alpha",
    "generator_split",
    "manifest_hash",
    "formal_config_sha256",
    "registered_manifest_sha256",
)
_GENERATOR_INVARIANT_COLUMNS = (
    "alpha",
    "transition_rank",
    "input_rank",
    "delay",
    "noise_std",
    "generator_split",
    "manifest_hash",
    "formal_config_sha256",
    "registered_manifest_sha256",
)


def _readonly_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any] | type[Any],
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _string_tuple(
    value: Sequence[object], *, name: str, length: int
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(str(item) for item in value)
    if len(result) != length:
        raise ValueError(f"{name} must have length {length}")
    if any(not item for item in result):
        raise ValueError(f"{name} values must be non-empty")
    return result


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_conclusion(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read a valid conclusion JSON from {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("conclusion must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _conclusion_raw_sha256(conclusion: Mapping[str, Any]) -> str | None:
    direct = conclusion.get("raw_metrics_sha256")
    if direct is not None:
        return _require_sha256(direct, name="conclusion raw_metrics_sha256")
    artifacts = conclusion.get("artifacts")
    if (
        isinstance(artifacts, Mapping)
        and artifacts.get("raw_metrics_sha256") is not None
    ):
        return _require_sha256(
            artifacts["raw_metrics_sha256"],
            name="conclusion artifacts.raw_metrics_sha256",
        )
    return None


def _validate_bool_series(series: pd.Series, *, name: str) -> np.ndarray:
    values: list[bool] = []
    for item in series.tolist():
        if isinstance(item, (bool, np.bool_)):
            values.append(bool(item))
        elif isinstance(item, str) and item.strip().lower() in {"true", "false"}:
            values.append(item.strip().lower() == "true")
        else:
            raise ValueError(f"{name} must contain explicit booleans")
    return np.asarray(values, dtype=bool)


def _single_string(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"{column} must be identical across the complete panel")
    return values[0]


def _validate_mode_invariance(
    group: pd.DataFrame, *, seed: int, generator_id: str
) -> None:
    for column in _MODE_INVARIANT_COLUMNS:
        values = group[column].to_numpy()
        first = values[0]
        if pd.isna(first) or np.any(pd.isna(values)):
            raise ValueError(f"{column} contains missing values")
        if not all(item == first for item in values[1:]):
            raise ValueError(
                f"mode-invariant feature {column} differs for seed={seed}, "
                f"generator={generator_id}"
            )


def _feature_row(row: pd.Series) -> np.ndarray:
    state_demand = float(row["state_demand"])
    input_demand = float(row["input_demand"])
    transition_rank_value = float(row["transition_rank"])
    input_rank_value = float(row["input_rank"])
    if not transition_rank_value.is_integer() or not input_rank_value.is_integer():
        raise ValueError("task ranks must be exact integers")
    transition_rank = int(transition_rank_value)
    input_rank = int(input_rank_value)
    delay = float(row["delay"])
    noise = float(row["noise_std"])
    chi = float(row["chi"])
    if state_demand < 0.0 or input_demand < 0.0:
        raise ValueError("demand values must be non-negative")
    if transition_rank < 1 or input_rank < 1:
        raise ValueError("task ranks must be positive")
    if delay < 0.0 or noise < 0.0:
        raise ValueError("delay and noise must be non-negative")
    if not 0.0 <= chi <= 1.0:
        raise ValueError("chi must lie in [0, 1]")
    return np.asarray(
        [
            chi,
            np.log(max(state_demand, DEMAND_FLOOR)),
            np.log(max(input_demand, DEMAND_FLOOR)),
            np.log2(transition_rank),
            np.log2(input_rank),
            delay / DELAY_SCALE_STEPS,
            np.log(max(noise, NOISE_FLOOR)),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Exp26SelectorSource:
    """Immutable, one-row-per-seed/generator Exp26 selector source."""

    profile: str
    conclusion: str
    raw_metrics_sha256: str
    conclusion_sha256: str
    config_sha256: str
    manifest_sha256: str
    candidate_modes: tuple[str, ...]
    seeds: np.ndarray
    generator_ids: tuple[str, ...]
    generator_splits: tuple[str, ...]
    alpha: np.ndarray
    transition_rank: np.ndarray
    input_rank: np.ndarray
    delay: np.ndarray
    noise_std: np.ndarray
    raw_features: np.ndarray
    validation_utilities: np.ndarray
    test_utilities: np.ndarray

    def __post_init__(self) -> None:
        if self.profile not in {"formal", "smoke"}:
            raise ValueError("profile must be 'formal' or 'smoke'")
        if self.conclusion not in {"support", "oppose", "inconclusive"}:
            raise ValueError("conclusion is not a registered ternary conclusion")
        for name in (
            "raw_metrics_sha256",
            "conclusion_sha256",
            "config_sha256",
            "manifest_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), name=name)
            )
        if tuple(self.candidate_modes) != CANDIDATE_MODES:
            raise ValueError(f"candidate_modes must equal {CANDIDATE_MODES}")
        seeds = _readonly_array(self.seeds, name="seeds", dtype=np.int64, ndim=1)
        n_samples = seeds.shape[0]
        if n_samples == 0:
            raise ValueError("selector source must contain samples")
        generator_ids = _string_tuple(
            self.generator_ids, name="generator_ids", length=n_samples
        )
        generator_splits = _string_tuple(
            self.generator_splits, name="generator_splits", length=n_samples
        )
        if set(generator_splits) != {"discovery", "heldout"}:
            raise ValueError("source must contain discovery and heldout generators")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "generator_ids", generator_ids)
        object.__setattr__(self, "generator_splits", generator_splits)
        vector_specs = (
            ("alpha", np.float64),
            ("transition_rank", np.int64),
            ("input_rank", np.int64),
            ("delay", np.float64),
            ("noise_std", np.float64),
        )
        for name, dtype in vector_specs:
            value = _readonly_array(getattr(self, name), name=name, dtype=dtype, ndim=1)
            if value.shape != (n_samples,):
                raise ValueError(f"{name} must have one value per sample")
            object.__setattr__(self, name, value)
        features = _readonly_array(
            self.raw_features, name="raw_features", dtype=np.float64, ndim=2
        )
        validation = _readonly_array(
            self.validation_utilities,
            name="validation_utilities",
            dtype=np.float64,
            ndim=2,
        )
        test = _readonly_array(
            self.test_utilities, name="test_utilities", dtype=np.float64, ndim=2
        )
        if features.shape != (n_samples, len(RAW_FEATURE_NAMES)):
            raise ValueError("raw_features has an invalid shape")
        expected_utilities = (n_samples, len(CANDIDATE_MODES))
        if validation.shape != expected_utilities or test.shape != expected_utilities:
            raise ValueError("utility matrices have an invalid shape")
        if np.any((validation < 0.0) | (validation > 1.0)) or np.any(
            (test < 0.0) | (test > 1.0)
        ):
            raise ValueError("balanced-accuracy utilities must lie in [0, 1]")
        object.__setattr__(self, "raw_features", features)
        object.__setattr__(self, "validation_utilities", validation)
        object.__setattr__(self, "test_utilities", test)

    @property
    def unique_seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.unique(self.seeds))


@dataclass(frozen=True)
class SelectorFold:
    """One immutable outer-seed leave-one-seed-out selector fold."""

    outer_seed: int
    candidate_modes: tuple[str, ...]
    feature_names: tuple[str, ...]
    train_seeds: np.ndarray
    train_generator_ids: tuple[str, ...]
    train_raw_features: np.ndarray
    train_utilities: np.ndarray
    test_seeds: np.ndarray
    test_generator_ids: tuple[str, ...]
    test_raw_features: np.ndarray
    test_utilities: np.ndarray
    test_unseen_composition: np.ndarray
    test_composition_overlap: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.outer_seed, (bool, np.bool_)) or not isinstance(
            self.outer_seed, (int, np.integer)
        ):
            raise TypeError("outer_seed must be an integer")
        if tuple(self.candidate_modes) != CANDIDATE_MODES:
            raise ValueError(f"candidate_modes must equal {CANDIDATE_MODES}")
        if tuple(self.feature_names) != RAW_FEATURE_NAMES:
            raise ValueError(f"feature_names must equal {RAW_FEATURE_NAMES}")
        for prefix in ("train", "test"):
            seeds = _readonly_array(
                getattr(self, f"{prefix}_seeds"),
                name=f"{prefix}_seeds",
                dtype=np.int64,
                ndim=1,
            )
            n_samples = seeds.shape[0]
            if n_samples == 0:
                raise ValueError(f"{prefix} fold must be non-empty")
            identifiers = _string_tuple(
                getattr(self, f"{prefix}_generator_ids"),
                name=f"{prefix}_generator_ids",
                length=n_samples,
            )
            features = _readonly_array(
                getattr(self, f"{prefix}_raw_features"),
                name=f"{prefix}_raw_features",
                dtype=np.float64,
                ndim=2,
            )
            utilities = _readonly_array(
                getattr(self, f"{prefix}_utilities"),
                name=f"{prefix}_utilities",
                dtype=np.float64,
                ndim=2,
            )
            if features.shape != (n_samples, len(RAW_FEATURE_NAMES)):
                raise ValueError(f"{prefix}_raw_features has an invalid shape")
            if utilities.shape != (n_samples, len(CANDIDATE_MODES)):
                raise ValueError(f"{prefix}_utilities has an invalid shape")
            object.__setattr__(self, f"{prefix}_seeds", seeds)
            object.__setattr__(self, f"{prefix}_generator_ids", identifiers)
            object.__setattr__(self, f"{prefix}_raw_features", features)
            object.__setattr__(self, f"{prefix}_utilities", utilities)
        if int(self.outer_seed) in set(int(item) for item in self.train_seeds):
            raise ValueError("outer seed leaked into selector training samples")
        if set(int(item) for item in self.test_seeds) != {int(self.outer_seed)}:
            raise ValueError("test samples must come only from the outer seed")
        unseen = _readonly_array(
            self.test_unseen_composition,
            name="test_unseen_composition",
            dtype=bool,
            ndim=1,
        )
        overlap = _readonly_array(
            self.test_composition_overlap,
            name="test_composition_overlap",
            dtype=bool,
            ndim=1,
        )
        if (
            unseen.shape != self.test_seeds.shape
            or overlap.shape != self.test_seeds.shape
        ):
            raise ValueError("composition flags must have one value per test sample")
        if not np.array_equal(overlap, ~unseen):
            raise ValueError("composition overlap must be the complement of unseen")
        object.__setattr__(self, "test_unseen_composition", unseen)
        object.__setattr__(self, "test_composition_overlap", overlap)


@dataclass(frozen=True)
class SelectorFeatureNormalizer:
    """Training-discovery-only standardization receipt."""

    mean: np.ndarray
    scale: np.ndarray
    n_fit_samples: int

    def __post_init__(self) -> None:
        mean = _readonly_array(self.mean, name="mean", dtype=np.float64, ndim=1)
        scale = _readonly_array(self.scale, name="scale", dtype=np.float64, ndim=1)
        if mean.shape != (len(RAW_FEATURE_NAMES),) or scale.shape != mean.shape:
            raise ValueError("normalizer vectors have an invalid shape")
        if np.any(scale <= 0.0):
            raise ValueError("normalizer scales must be positive")
        if isinstance(self.n_fit_samples, (bool, np.bool_)) or not isinstance(
            self.n_fit_samples, (int, np.integer)
        ):
            raise TypeError("n_fit_samples must be an integer")
        if int(self.n_fit_samples) < 1:
            raise ValueError("n_fit_samples must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "n_fit_samples", int(self.n_fit_samples))

    @classmethod
    def fit(cls, train_raw_features: object) -> "SelectorFeatureNormalizer":
        features = _readonly_array(
            train_raw_features,
            name="train_raw_features",
            dtype=np.float64,
            ndim=2,
        )
        if features.shape[0] < 1 or features.shape[1] != len(RAW_FEATURE_NAMES):
            raise ValueError("train_raw_features has an invalid shape")
        scale = np.std(features, axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(
            mean=np.mean(features, axis=0),
            scale=scale,
            n_fit_samples=features.shape[0],
        )

    def transform(self, raw_features: object) -> np.ndarray:
        features = _readonly_array(
            raw_features, name="raw_features", dtype=np.float64, ndim=2
        )
        if features.shape[1:] != (len(RAW_FEATURE_NAMES),):
            raise ValueError("raw_features has an invalid shape")
        normalized = (features - self.mean) / self.scale
        result = np.column_stack((normalized, np.ones(features.shape[0])))
        result.setflags(write=False)
        return result

    def fit_transform(self, train_raw_features: object) -> np.ndarray:
        return self.transform(train_raw_features)

    @property
    def fit_fingerprint(self) -> str:
        """Stable audit fingerprint of training-only fitted statistics."""

        payload = {
            "feature_names": RAW_FEATURE_NAMES,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "n_fit_samples": self.n_fit_samples,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def center(self) -> np.ndarray:
        """Read-only alias used by serialized normalization receipts."""

        return self.mean

    def to_dict(self) -> dict[str, object]:
        """Return the fixed, directly JSON-serializable fit receipt."""

        return {
            "feature_names": list(RAW_FEATURE_NAMES),
            "center": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fingerprint": self.fit_fingerprint,
            "fit_scope": "training_discovery_only",
            "train_n": self.n_fit_samples,
        }


def load_exp26_selector_source(
    raw_metrics_path: str | Path,
    conclusion_path: str | Path,
    *,
    expected_profile: str,
    expected_raw_sha256: str | None = None,
    require_support: bool = True,
) -> Exp26SelectorSource:
    """Load a hash-bound, complete Exp26 panel for selector learning.

    A raw hash must be supplied either by ``expected_raw_sha256`` or by the
    conclusion JSON.  If both are present they must agree.  Formal selector
    learning defaults to requiring the registered Exp26 conclusion
    ``"support"``; smoke fixtures may explicitly set ``require_support=False``.
    """

    if expected_profile not in {"formal", "smoke"}:
        raise ValueError("expected_profile must be 'formal' or 'smoke'")
    if not isinstance(require_support, bool):
        raise TypeError("require_support must be a bool")
    raw_path = Path(raw_metrics_path)
    conclusion_file = Path(conclusion_path)
    if not raw_path.is_file():
        raise ValueError(f"raw metrics file does not exist: {raw_path}")
    if raw_path.suffix not in {".csv", ".gz"}:
        raise ValueError("raw metrics must be a .csv or .csv.gz file")
    conclusion, conclusion_sha = _read_conclusion(conclusion_file)
    bound_sha = _conclusion_raw_sha256(conclusion)
    if expected_raw_sha256 is not None:
        expected_raw_sha256 = _require_sha256(
            expected_raw_sha256, name="expected_raw_sha256"
        )
        if bound_sha is not None and bound_sha != expected_raw_sha256:
            raise ValueError("explicit and conclusion raw-metrics hashes disagree")
        bound_sha = expected_raw_sha256
    if bound_sha is None:
        raise ValueError("raw metrics SHA-256 is not bound by argument or conclusion")
    observed_raw_sha = _file_sha256(raw_path)
    if observed_raw_sha != bound_sha:
        raise ValueError("raw metrics SHA-256 does not match the registered digest")

    profile = conclusion.get("profile")
    if profile != expected_profile:
        raise ValueError("conclusion profile does not match expected_profile")
    result = conclusion.get("conclusion")
    if result not in {"support", "oppose", "inconclusive"}:
        raise ValueError("conclusion is not a registered ternary conclusion")
    if require_support and result != "support":
        raise ValueError("Exp26 must conclude support before selector learning")
    if conclusion.get("complete_primary_coverage") is not True:
        raise ValueError("Exp26 conclusion lacks complete primary coverage")
    if expected_profile == "formal" and result == "support":
        if conclusion.get("confirmatory_eligible") is not True:
            raise ValueError("formal support is not confirmatory eligible")
        if conclusion.get("dev_only") is not False:
            raise ValueError("formal support cannot be development-only")

    config_sha = _require_sha256(
        conclusion.get("registered_config_sha256"),
        name="conclusion registered_config_sha256",
    )
    manifest_sha = _require_sha256(
        conclusion.get("registered_manifest_sha256"),
        name="conclusion registered_manifest_sha256",
    )
    coverage = conclusion.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("conclusion coverage must be a mapping")
    expected_seed_values = coverage.get("expected_seeds")
    if not isinstance(expected_seed_values, list) or not expected_seed_values:
        raise ValueError("conclusion coverage must register expected seeds")
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in expected_seed_values
    ):
        raise ValueError("registered expected seeds must be integers")
    expected_seeds = tuple(sorted(int(item) for item in expected_seed_values))
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("registered expected seeds contain duplicates")

    try:
        frame = pd.read_csv(raw_path, compression="infer")
    except Exception as error:  # pandas parser errors vary by engine/version
        raise ValueError(f"cannot read raw metrics: {error}") from error
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"raw metrics lack required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("raw metrics are empty")
    if int(coverage.get("raw_row_count", -1)) != len(frame):
        raise ValueError("raw row count disagrees with conclusion coverage")
    required_frame = frame.loc[:, sorted(_REQUIRED_COLUMNS)]
    if required_frame.isna().any(axis=None):
        missing_columns = required_frame.columns[required_frame.isna().any()].tolist()
        raise ValueError(f"raw metrics contain missing values: {missing_columns}")
    numeric_columns = (
        "seed",
        "chi",
        "state_demand",
        "input_demand",
        "transition_rank",
        "input_rank",
        "delay",
        "noise_std",
        "alpha",
        "validation_balanced_accuracy",
        "test_balanced_accuracy",
    )
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{column} must contain finite numeric values")
        frame[column] = values
    if np.any(frame["seed"].to_numpy() % 1 != 0):
        raise ValueError("seed values must be integers")
    frame["seed"] = frame["seed"].astype(np.int64)
    observed_seeds = tuple(sorted(int(value) for value in frame["seed"].unique()))
    if observed_seeds != expected_seeds:
        raise ValueError("raw seed panel does not match conclusion coverage")
    if int(conclusion.get("n_seeds", -1)) != len(expected_seeds):
        raise ValueError("conclusion n_seeds disagrees with the registered panel")
    if _single_string(frame, "profile") != expected_profile:
        raise ValueError("raw profile does not match expected_profile")
    if _single_string(frame, "manifest_hash") != manifest_sha:
        raise ValueError("raw manifest hash disagrees with conclusion")
    if _single_string(frame, "registered_manifest_sha256") != manifest_sha:
        raise ValueError("raw registered manifest hash disagrees with conclusion")
    if _single_string(frame, "formal_config_sha256") != config_sha:
        raise ValueError("raw config hash disagrees with conclusion")
    if set(frame["status"].astype(str)) != {"complete"}:
        raise ValueError("selector source contains failed or invalid rows")
    if not np.all(
        _validate_bool_series(
            frame["functional_budget_valid"], name="functional_budget_valid"
        )
    ):
        raise ValueError("selector source contains functional-budget-invalid rows")

    duplicate = frame.duplicated(["seed", "generator_id", "actuator_mode"], keep=False)
    if bool(duplicate.any()):
        raise ValueError("duplicate seed × generator × mode rows are not allowed")
    grouped = frame.groupby(["seed", "generator_id"], sort=True, observed=True)
    expected_mode_set = set(EXPECTED_PANEL_MODES)
    first_seed_generators: set[str] | None = None
    metadata_by_generator: dict[str, tuple[object, ...]] = {}
    sample_rows: list[tuple[int, str, pd.Series, np.ndarray, np.ndarray]] = []
    for (seed_value, generator_value), group in grouped:
        seed = int(seed_value)
        generator_id = str(generator_value)
        modes = set(group["actuator_mode"].astype(str))
        if modes != expected_mode_set or len(group) != len(EXPECTED_PANEL_MODES):
            raise ValueError(
                f"incomplete actuator panel for seed={seed}, generator={generator_id}"
            )
        _validate_mode_invariance(group, seed=seed, generator_id=generator_id)
        indexed = group.set_index(group["actuator_mode"].astype(str), drop=False)
        reference = indexed.loc[CANDIDATE_MODES[0]]
        metadata = tuple(reference[column] for column in _GENERATOR_INVARIANT_COLUMNS)
        if (
            generator_id in metadata_by_generator
            and metadata_by_generator[generator_id] != metadata
        ):
            raise ValueError("generator metadata differs across network seeds")
        metadata_by_generator[generator_id] = metadata
        validation = np.asarray(
            [
                indexed.loc[mode, "validation_balanced_accuracy"]
                for mode in CANDIDATE_MODES
            ],
            dtype=np.float64,
        )
        test = np.asarray(
            [indexed.loc[mode, "test_balanced_accuracy"] for mode in CANDIDATE_MODES],
            dtype=np.float64,
        )
        sample_rows.append((seed, generator_id, reference, validation, test))
    for seed in expected_seeds:
        generators = set(
            str(item)
            for item in frame.loc[frame["seed"] == seed, "generator_id"].unique()
        )
        if first_seed_generators is None:
            first_seed_generators = generators
        elif generators != first_seed_generators:
            raise ValueError("generator panel differs across network seeds")
    assert first_seed_generators is not None
    expected_raw_rows = (
        len(expected_seeds) * len(first_seed_generators) * len(EXPECTED_PANEL_MODES)
    )
    if len(frame) != expected_raw_rows:
        raise ValueError("raw panel is not a complete seed × generator × mode product")
    if int(coverage.get("primary_row_count", -1)) != (
        len(expected_seeds) * len(first_seed_generators) * 4
    ):
        raise ValueError("primary row coverage disagrees with the complete panel")
    if int(coverage.get("rgl_ceiling_row_count", -1)) != (
        len(expected_seeds) * len(first_seed_generators)
    ):
        raise ValueError("RGL ceiling coverage disagrees with the complete panel")

    split_order = {"discovery": 0, "heldout": 1}
    sample_rows.sort(
        key=lambda item: (
            item[0],
            split_order.get(str(item[2]["generator_split"]), 99),
            item[1],
        )
    )
    splits = tuple(str(item[2]["generator_split"]) for item in sample_rows)
    if set(splits) != {"discovery", "heldout"}:
        raise ValueError("generator_split must contain discovery and heldout only")
    if any(split not in split_order for split in splits):
        raise ValueError("unknown generator split")
    return Exp26SelectorSource(
        profile=expected_profile,
        conclusion=str(result),
        raw_metrics_sha256=observed_raw_sha,
        conclusion_sha256=conclusion_sha,
        config_sha256=config_sha,
        manifest_sha256=manifest_sha,
        candidate_modes=CANDIDATE_MODES,
        seeds=np.asarray([item[0] for item in sample_rows]),
        generator_ids=tuple(item[1] for item in sample_rows),
        generator_splits=splits,
        alpha=np.asarray([float(item[2]["alpha"]) for item in sample_rows]),
        transition_rank=np.asarray(
            [int(item[2]["transition_rank"]) for item in sample_rows]
        ),
        input_rank=np.asarray([int(item[2]["input_rank"]) for item in sample_rows]),
        delay=np.asarray([float(item[2]["delay"]) for item in sample_rows]),
        noise_std=np.asarray([float(item[2]["noise_std"]) for item in sample_rows]),
        raw_features=np.vstack([_feature_row(item[2]) for item in sample_rows]),
        validation_utilities=np.vstack([item[3] for item in sample_rows]),
        test_utilities=np.vstack([item[4] for item in sample_rows]),
    )


def build_outer_seed_loso(
    source: Exp26SelectorSource,
    outer_seed: int,
) -> SelectorFold:
    """Create a leakage-safe outer-seed LOSO selector fold."""

    if not isinstance(source, Exp26SelectorSource):
        raise TypeError("source must be an Exp26SelectorSource")
    if isinstance(outer_seed, (bool, np.bool_)) or not isinstance(
        outer_seed, (int, np.integer)
    ):
        raise TypeError("outer_seed must be an integer")
    outer = int(outer_seed)
    if outer not in source.unique_seeds:
        raise ValueError("outer_seed is not present in source")
    if len(source.unique_seeds) < 2:
        raise ValueError("LOSO requires at least two independent seeds")
    splits = np.asarray(source.generator_splits, dtype=object)
    train_mask = (source.seeds != outer) & (splits == "discovery")
    test_mask = (source.seeds == outer) & (splits == "heldout")
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("LOSO train or test partition is empty")
    discovery_mask = splits == "discovery"
    discovery_compositions = {
        (
            float(source.alpha[index]),
            int(source.transition_rank[index]),
            int(source.input_rank[index]),
        )
        for index in np.flatnonzero(discovery_mask)
    }
    test_indices = np.flatnonzero(test_mask)
    unseen = np.asarray(
        [
            (
                float(source.alpha[index]),
                int(source.transition_rank[index]),
                int(source.input_rank[index]),
            )
            not in discovery_compositions
            for index in test_indices
        ],
        dtype=bool,
    )
    train_indices = np.flatnonzero(train_mask)
    return SelectorFold(
        outer_seed=outer,
        candidate_modes=CANDIDATE_MODES,
        feature_names=RAW_FEATURE_NAMES,
        train_seeds=source.seeds[train_indices],
        train_generator_ids=tuple(
            source.generator_ids[index] for index in train_indices
        ),
        train_raw_features=source.raw_features[train_indices],
        train_utilities=source.validation_utilities[train_indices],
        test_seeds=source.seeds[test_indices],
        test_generator_ids=tuple(source.generator_ids[index] for index in test_indices),
        test_raw_features=source.raw_features[test_indices],
        test_utilities=source.test_utilities[test_indices],
        test_unseen_composition=unseen,
        test_composition_overlap=~unseen,
    )


def build_three_step_cues(normalized_features: object) -> np.ndarray:
    """Mask normalized features into demand, rank, and timing cue steps.

    The bias appears only in the final timing step.  Consequently, summing the
    three cues exactly reconstructs the normalized eight-feature vector and
    cannot accidentally triple the intercept.
    """

    features = _readonly_array(
        normalized_features,
        name="normalized_features",
        dtype=np.float64,
        ndim=2,
    )
    if features.shape[1:] != (len(NORMALIZED_FEATURE_NAMES),):
        raise ValueError("normalized_features must have eight columns")
    if not np.allclose(features[:, -1], 1.0, rtol=0.0, atol=0.0):
        raise ValueError("normalized_features must contain an explicit unit bias")
    cues = np.zeros((features.shape[0], 3, features.shape[1]), dtype=np.float64)
    cues[:, 0, 0:3] = features[:, 0:3]
    cues[:, 1, 3:5] = features[:, 3:5]
    cues[:, 2, 5:8] = features[:, 5:8]
    cues.setflags(write=False)
    return cues


__all__ = [
    "CANDIDATE_MODES",
    "EXPECTED_PANEL_MODES",
    "RAW_FEATURE_NAMES",
    "NORMALIZED_FEATURE_NAMES",
    "Exp26SelectorSource",
    "SelectorFeatureNormalizer",
    "SelectorFold",
    "build_outer_seed_loso",
    "build_three_step_cues",
    "load_exp26_selector_source",
]
