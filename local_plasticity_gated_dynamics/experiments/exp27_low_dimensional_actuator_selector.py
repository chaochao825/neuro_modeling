"""Nested-seed selector over the immutable Exp26 actuator dictionary.

The carrier, readout, actuator bases, trajectories, and candidate utilities
are read-only Exp26 evidence.  For each outer seed, selectors see only other
seeds' discovery-composition validation utilities.  They are evaluated once
on the outer seed's held-out-composition test utilities.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from experiments.exp26_actuator_phase_diagram import (
    canonical_config_sha256,
    git_identity,
)
from src.data.actuator_selector_dataset import (
    SelectorFeatureNormalizer,
    build_outer_seed_loso,
    build_three_step_cues,
    load_exp26_selector_source,
)
from src.models.actuator_selector import GRUSelectorBaseline
from src.plasticity.selector_three_factor import LocalThreeFactorSelector
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp27_low_dimensional_actuator_selector"
PROTOCOL_VERSION = "exp27_preregistered_v1_frozen_exp26_dictionary"
CANDIDATE_MODES = ("routing", "gain", "low_rank")
SELECTORS = ("oracle", "gru_bptt", "local_three_factor", "fixed_best")
TRAINING_SCOPE = "other_seed_discovery_validation_only"
TEST_SCOPE = "outer_seed_heldout_test_only"
SOURCE_RECEIPT_SCHEMA = "exp27_frozen_exp26_source_v1"
_SOURCE_HASH_REGISTRY = {
    "formal": {
        "expected_raw_metrics_sha256": "b3ef5e22c241f832b1fd50254f87e3890ec45057bfeda3a784cbd218623a1193",
        "expected_conclusion_sha256": "2038127ac875f9faae94b305343415b8fb3a794f9ea032f017401e432fa9d40f",
        "expected_preflight_receipt_sha256": "bad665691233c9611fcdcce897c642d517a938b78adbabadee783c5e8cb1a671",
        "expected_config_sha256": "07ad3f16d9de6b5906155d95f215e9434e478ca992fd023adfabcd21a0005ecf",
        "expected_manifest_sha256": "a1c17a1e88c731f6678760865cf51d7236ae771bf839645c401e5cff8798ebfa",
    },
    "smoke": {
        "expected_raw_metrics_sha256": "e5dfd3ba9ea26b7b4319de910a0724b40a631f0f591f585f3b0c09033250700c",
        "expected_conclusion_sha256": "7a0f1dd04fb7d8ac88e05ea9ab4eff614f0b6bfc68873c4c7dd78a1058c96d1b",
        "expected_preflight_receipt_sha256": None,
        "expected_config_sha256": "583f42e522cbbc9ad42a36434ab6305ecdf715db62e368ed3eb306d429f96084",
        "expected_manifest_sha256": "9d334f0e3de86843b5e61b8f70cd7466c6933d47dcf50464fd51bd9a65519d5d",
    },
}

_SOURCE_REQUIRED_COLUMNS = {
    "seed",
    "generator_id",
    "generator_split",
    "actuator_mode",
    "alpha",
    "chi",
    "transition_rank",
    "input_rank",
    "delay",
    "noise_std",
    "state_demand",
    "input_demand",
    "validation_balanced_accuracy",
    "test_balanced_accuracy",
    "formal_config_sha256",
    "registered_manifest_sha256",
    "experiment_protocol_version",
    "run_label",
    "preflight_receipt_sha256",
    "functional_budget_valid",
    "effective_dynamics_strictly_stable",
    "readout_fit_train_only",
    "selector_learning_enabled",
    "used_autograd",
    "used_bptt",
    "status",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Exp27 source paths must be non-empty strings")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _expected_sha(value: object, *, name: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_seed(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_config(config: Mapping[str, Any]) -> tuple[int, ...]:
    profile = config.get("profile")
    if profile not in {"smoke", "formal"}:
        raise ValueError("Exp27 profile must be smoke or formal")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp27 protocol version mismatch")
    if bool(config.get("dev_only")) != (profile == "smoke"):
        raise ValueError("Exp27 dev_only/profile contract mismatch")
    if tuple(config.get("dictionary", ())) != CANDIDATE_MODES:
        raise ValueError("Exp27 frozen actuator dictionary/order mismatch")
    if tuple(config.get("selectors", ())) != SELECTORS:
        raise ValueError("Exp27 selector registry/order mismatch")
    if config.get("used_autograd") is not False or config.get("used_bptt") is not False:
        raise ValueError("Exp27 main local method must prohibit autograd and BPTT")
    expected_features = (
        "chi",
        "state_demand",
        "input_demand",
        "transition_rank",
        "input_rank",
        "delay",
        "noise_std",
    )
    if tuple(config.get("feature_columns", ())) != expected_features:
        raise ValueError("Exp27 registered selector feature contract mismatch")
    transform = config.get("feature_transform")
    if not isinstance(transform, Mapping):
        raise ValueError("Exp27 feature transform must be registered")
    if tuple(transform.get("cues", ())) != ("demand", "ranks", "timing_noise"):
        raise ValueError("Exp27 must use the three registered selector cues")
    if transform.get("fit_scope") != TRAINING_SCOPE:
        raise ValueError("Exp27 normalization must be nested-training-only")
    if transform.get("include_bias") is not True:
        raise ValueError("Exp27 transformed features must include a bias")
    expected_transforms = {
        "chi": "identity",
        "state_demand": "log",
        "input_demand": "log",
        "transition_rank": "log2",
        "input_rank": "log2",
        "delay": "scaled",
        "noise_std": "log",
    }
    if transform.get("raw_to_transformed") != expected_transforms:
        raise ValueError("Exp27 registered feature transformations mismatch")
    seeds_value = config.get("seeds")
    if not isinstance(seeds_value, Sequence) or isinstance(seeds_value, (str, bytes)):
        raise ValueError("Exp27 seeds must be a sequence")
    seeds = tuple(_exact_seed(value, name="registered seed") for value in seeds_value)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Exp27 seeds must be unique")
    expected_seeds = tuple(range(30)) if profile == "formal" else (9000, 9001)
    if seeds != expected_seeds:
        raise ValueError("Exp27 profile seed registry mismatch")
    analysis = config.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("Exp27 analysis settings must be registered")
    if bool(analysis.get("force_inconclusive")) != (profile == "smoke"):
        raise ValueError("Exp27 smoke must be forced inconclusive")
    expected_analysis = {
        "bootstrap_samples": 20000 if profile == "formal" else 200,
        "permutation_samples": 100000 if profile == "formal" else 500,
        "confidence": 0.95,
        "statistics_seed": 2701,
        "force_inconclusive": profile == "smoke",
    }
    if analysis != expected_analysis:
        raise ValueError("Exp27 analysis settings are not preregistered")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("Exp27 evaluation settings must be registered")
    expected_evaluation = {
        "train_generator_split": "discovery",
        "train_endpoint": "validation_balanced_accuracy",
        "test_generator_split": "heldout",
        "test_endpoint": "test_balanced_accuracy",
        "outer_unit": "seed",
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"Exp27 evaluation contract mismatch: {key}")
    if tuple(evaluation.get("tie_break_order", ())) != CANDIDATE_MODES:
        raise ValueError("Exp27 candidate tie-break order mismatch")
    if evaluation.get("local_oracle_gain_fraction_threshold") != 0.8:
        raise ValueError("Exp27 local selector threshold mismatch")
    expected_counts = (44, 43) if profile == "formal" else (12, 4)
    observed_counts = (
        evaluation.get("expected_heldout_generators_per_seed"),
        evaluation.get("expected_strict_unseen_per_seed"),
    )
    if observed_counts != expected_counts:
        raise ValueError("Exp27 held-out/strict-unseen count registry mismatch")
    expected_local = {
        "epochs": 200 if profile == "formal" else 20,
        "learning_rate": 0.05,
        "temperature": 1.0,
        "teacher_temperature": 0.05,
        "l2": 0.0001,
        "eligibility_decay": 0.8,
        "belief_retention": 0.8,
    }
    if config.get("local_selector") != expected_local:
        raise ValueError("Exp27 local-selector hyperparameters are not registered")
    expected_gru = {
        "hidden_dim": 8,
        "epochs": 200 if profile == "formal" else 20,
        "learning_rate": 0.02,
        "weight_decay": 0.0001,
        "teacher_temperature": 0.05,
        "device": "cpu",
        "deterministic": True,
    }
    if config.get("gru_selector") != expected_gru:
        raise ValueError("Exp27 GRU baseline hyperparameters are not registered")
    source = _source_config(config)
    for key, expected in _SOURCE_HASH_REGISTRY[profile].items():
        if source.get(key) != expected:
            raise ValueError(f"Exp27 frozen source registry mismatch: {key}")
    expected_source_metadata = {
        "expected_run_label": f"exp26-{profile}-v2",
        "expected_protocol_version": ("exp26_preregistered_v2_train_only_budget_bound"),
        "required_conclusion": "support" if profile == "formal" else "inconclusive",
        "required_profile": profile,
    }
    for key, expected in expected_source_metadata.items():
        if source.get(key) != expected:
            raise ValueError(f"Exp27 frozen source metadata mismatch: {key}")
    if profile == "formal" and config.get("required_run_label") != "exp27-formal-v1":
        raise ValueError("Exp27 formal run-label registry mismatch")
    return seeds


def _source_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    source = config.get("source_exp26")
    if not isinstance(source, Mapping):
        raise ValueError("Exp27 requires a frozen Exp26 source contract")
    return source


def _registered_source_provenance(config: Mapping[str, Any]) -> dict[str, object]:
    """Return the immutable receipt promised by config before source I/O."""

    source = _source_config(config)
    return {
        "schema_version": SOURCE_RECEIPT_SCHEMA,
        "raw_metrics_path": str(_resolve_source_path(source.get("raw_metrics_path"))),
        "conclusion_path": str(_resolve_source_path(source.get("conclusion_path"))),
        "raw_metrics_sha256": _expected_sha(
            source.get("expected_raw_metrics_sha256"),
            name="expected raw metrics hash",
        ),
        "conclusion_sha256": _expected_sha(
            source.get("expected_conclusion_sha256"),
            name="expected conclusion hash",
        ),
        "preflight_receipt_sha256": _expected_sha(
            source.get("expected_preflight_receipt_sha256"),
            name="expected Exp26 preflight receipt hash",
            optional=True,
        ),
        "source_config_sha256": _expected_sha(
            source.get("expected_config_sha256"),
            name="expected Exp26 config hash",
        ),
        "source_manifest_sha256": _expected_sha(
            source.get("expected_manifest_sha256"),
            name="expected Exp26 manifest hash",
        ),
        "source_run_label": source.get("expected_run_label"),
        "source_protocol_version": source.get("expected_protocol_version"),
        "source_profile": source.get("required_profile"),
        "source_conclusion": source.get("required_conclusion"),
    }


def _peek_source_rows(config: Mapping[str, Any], seed: int) -> pd.DataFrame:
    """Read only the fields needed to register a failure-retaining plan."""

    source = _source_config(config)
    path = _resolve_source_path(source.get("raw_metrics_path"))
    columns = [
        "seed",
        "generator_id",
        "generator_split",
        "alpha",
        "transition_rank",
        "input_rank",
        "delay",
        "noise_std",
    ]
    rows = pd.read_csv(path, usecols=columns)
    rows = rows[
        (rows["seed"].astype(int) == seed)
        & (rows["generator_split"].astype(str) == "heldout")
    ].drop_duplicates("generator_id")
    if rows.empty:
        raise ValueError("Exp27 source has no outer-seed held-out generators")
    return rows.sort_values("generator_id", kind="stable").reset_index(drop=True)


def _planned_conditions(peek: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    conditions: list[dict[str, object]] = []
    for row in peek.itertuples(index=False):
        for selector in SELECTORS:
            conditions.append(
                {
                    "outer_seed": seed,
                    "source_seed": seed,
                    "generator_split": "heldout",
                    "generator_id": str(row.generator_id),
                    "alpha": float(row.alpha),
                    "transition_rank": int(row.transition_rank),
                    "input_rank": int(row.input_rank),
                    "delay": int(row.delay),
                    "noise": float(row.noise_std),
                    "selector": selector,
                }
            )
    return conditions


def _all_boolean(rows: pd.DataFrame, column: str, expected: bool) -> bool:
    values = rows[column]
    if values.isna().any():
        return False

    def parse(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError(f"Exp27 source {column} contains a non-boolean literal")

    normalized = values.map(parse)
    return bool((normalized == expected).all())


def _singleton_text(rows: pd.DataFrame, column: str) -> str:
    values = rows[column].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"Exp27 source {column} must have one value")
    return values[0]


def _validate_frozen_source(
    config: Mapping[str, Any],
) -> tuple[object, pd.DataFrame, dict[str, object]]:
    source_config = _source_config(config)
    raw_path = _resolve_source_path(source_config.get("raw_metrics_path"))
    conclusion_path = _resolve_source_path(source_config.get("conclusion_path"))
    expected_raw = _expected_sha(
        source_config.get("expected_raw_metrics_sha256"),
        name="expected raw metrics hash",
    )
    expected_conclusion = _expected_sha(
        source_config.get("expected_conclusion_sha256"),
        name="expected conclusion hash",
    )
    expected_config = _expected_sha(
        source_config.get("expected_config_sha256"),
        name="expected Exp26 config hash",
    )
    expected_manifest = _expected_sha(
        source_config.get("expected_manifest_sha256"),
        name="expected Exp26 manifest hash",
    )
    expected_preflight = _expected_sha(
        source_config.get("expected_preflight_receipt_sha256"),
        name="expected Exp26 preflight receipt hash",
        optional=True,
    )
    observed_raw = _file_sha256(raw_path)
    observed_conclusion = _file_sha256(conclusion_path)
    if observed_raw != expected_raw:
        raise ValueError("Exp27 frozen Exp26 raw-metrics SHA-256 mismatch")
    if observed_conclusion != expected_conclusion:
        raise ValueError("Exp27 frozen Exp26 conclusion SHA-256 mismatch")
    try:
        conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Exp27 cannot read the Exp26 conclusion") from error
    if not isinstance(conclusion, Mapping):
        raise ValueError("Exp27 Exp26 conclusion must be a JSON object")
    expected_profile = source_config.get("required_profile")
    if conclusion.get("profile") != expected_profile:
        raise ValueError("Exp27 Exp26 conclusion profile mismatch")
    if conclusion.get("conclusion") != source_config.get("required_conclusion"):
        raise ValueError("Exp27 Exp26 conclusion classification mismatch")
    if conclusion.get("registered_config_sha256") != expected_config:
        raise ValueError("Exp27 Exp26 conclusion config receipt mismatch")
    if conclusion.get("registered_manifest_sha256") != expected_manifest:
        raise ValueError("Exp27 Exp26 conclusion manifest receipt mismatch")

    rows = pd.read_csv(raw_path)
    missing = _SOURCE_REQUIRED_COLUMNS - set(rows.columns)
    if missing:
        raise ValueError(f"Exp27 source schema lacks columns: {sorted(missing)}")
    registered_seeds = tuple(int(value) for value in config["seeds"])
    observed_seeds = tuple(sorted(rows["seed"].astype(int).unique().tolist()))
    if observed_seeds != registered_seeds:
        raise ValueError("Exp27 source seed panel mismatch")
    if _singleton_text(rows, "formal_config_sha256") != expected_config:
        raise ValueError("Exp27 raw source config receipt mismatch")
    if _singleton_text(rows, "registered_manifest_sha256") != expected_manifest:
        raise ValueError("Exp27 raw source manifest receipt mismatch")
    if _singleton_text(rows, "experiment_protocol_version") != source_config.get(
        "expected_protocol_version"
    ):
        raise ValueError("Exp27 raw source protocol mismatch")
    if _singleton_text(rows, "run_label") != source_config.get("expected_run_label"):
        raise ValueError("Exp27 raw source run-label mismatch")
    if expected_preflight is None:
        if rows["preflight_receipt_sha256"].notna().any():
            raise ValueError("Exp27 smoke source unexpectedly carries preflight")
    elif _singleton_text(rows, "preflight_receipt_sha256") != expected_preflight:
        raise ValueError("Exp27 raw source preflight receipt mismatch")
    for column, expected in (
        ("functional_budget_valid", True),
        ("effective_dynamics_strictly_stable", True),
        ("readout_fit_train_only", True),
        ("selector_learning_enabled", False),
        ("used_autograd", False),
        ("used_bptt", False),
    ):
        if not _all_boolean(rows, column, expected):
            raise ValueError(f"Exp27 source invariant failed: {column}")
    if set(rows["status"].astype(str)) != {"complete"}:
        raise ValueError("Exp27 source includes incomplete candidate rows")

    candidates = rows[rows["actuator_mode"].isin(CANDIDATE_MODES)].copy()
    counts = candidates.groupby(["seed", "generator_id"], observed=True).size()
    if counts.empty or not counts.eq(len(CANDIDATE_MODES)).all():
        raise ValueError("Exp27 source lacks a complete three-actuator dictionary")
    if candidates.duplicated(["seed", "generator_id", "actuator_mode"]).any():
        raise ValueError("Exp27 source duplicates a seed/generator/actuator cell")
    if set(candidates["actuator_mode"].astype(str)) != set(CANDIDATE_MODES):
        raise ValueError("Exp27 source actuator dictionary mismatch")
    numeric_columns = [
        *config["feature_columns"],
        "alpha",
        "validation_balanced_accuracy",
        "test_balanced_accuracy",
    ]
    numeric = candidates[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Exp27 source contains non-finite registered values")
    for endpoint in ("validation_balanced_accuracy", "test_balanced_accuracy"):
        values = numeric[endpoint].to_numpy(dtype=np.float64)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"Exp27 source {endpoint} lies outside [0, 1]")
    metadata_columns = [
        "generator_split",
        "alpha",
        *config["feature_columns"],
    ]
    metadata_unique = candidates.groupby(["seed", "generator_id"], observed=True)[
        metadata_columns
    ].nunique(dropna=False)
    if (metadata_unique > 1).any(axis=None):
        raise ValueError("Exp27 source generator metadata vary across actuators")
    generator_splits = candidates.groupby("generator_id", observed=True)[
        "generator_split"
    ].nunique()
    if not generator_splits.eq(1).all():
        raise ValueError("Exp27 source generator split varies across seeds")
    discovery_ids = set(
        candidates.loc[candidates["generator_split"] == "discovery", "generator_id"]
    )
    heldout_ids = set(
        candidates.loc[candidates["generator_split"] == "heldout", "generator_id"]
    )
    if not discovery_ids or not heldout_ids or discovery_ids & heldout_ids:
        raise ValueError("Exp27 source generator IDs are not block-disjoint")

    loaded_source = load_exp26_selector_source(
        raw_path,
        conclusion_path,
        expected_profile=str(expected_profile),
        expected_raw_sha256=expected_raw,
        require_support=expected_profile == "formal",
    )
    receipt: dict[str, object] = {
        **_registered_source_provenance(config),
        "candidate_rows": int(candidates.shape[0]),
        "source_seeds": list(registered_seeds),
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt)
    return loaded_source, candidates, receipt


def _metadata_for_test(
    candidates: pd.DataFrame,
    *,
    seed: int,
    generator_ids: Sequence[object],
) -> pd.DataFrame:
    rows = candidates[
        (candidates["seed"].astype(int) == seed)
        & (candidates["generator_split"].astype(str) == "heldout")
        & (candidates["actuator_mode"].astype(str) == CANDIDATE_MODES[0])
    ].set_index("generator_id", drop=False)
    ordered_ids = [str(value) for value in generator_ids]
    rows.index = rows.index.astype(str)
    missing = set(ordered_ids) - set(rows.index)
    if missing:
        raise ValueError(
            f"Exp27 fold test metadata missing generators: {sorted(missing)}"
        )
    return rows.loc[ordered_ids].reset_index(drop=True)


def _construct_registered(cls: type, options: Mapping[str, object]) -> object:
    """Fail closed rather than silently ignoring a frozen hyperparameter."""

    signature = inspect.signature(cls)
    unsupported = set(options) - set(signature.parameters)
    if unsupported:
        raise TypeError(
            f"{cls.__name__} lacks registered parameters: {sorted(unsupported)}"
        )
    return cls(**options)


def _receipt_value(receipt: object, name: str) -> object:
    if isinstance(receipt, Mapping):
        return receipt.get(name)
    return getattr(receipt, name, None)


def _receipt_costs(receipt: object) -> tuple[float, float]:
    l1 = _receipt_value(receipt, "cumulative_update_l1")
    l2 = _receipt_value(receipt, "cumulative_update_l2")
    if l1 is None or l2 is None:
        raise ValueError("trained selector receipt lacks cumulative update costs")
    result = float(l1), float(l2)
    if not np.isfinite(result).all() or min(result) < 0.0:
        raise ValueError("selector update costs must be finite and non-negative")
    return result


def _jsonable(value: object) -> object:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Exp27 receipts cannot contain non-finite floats")
    return value


def _payload_sha256(value: object) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_fingerprint(label: str, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _decision_fingerprint(
    generator_ids: Sequence[object], probabilities: np.ndarray
) -> str:
    identifiers = [str(value) for value in generator_ids]
    encoded_ids = json.dumps(
        identifiers, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(identifiers), len(CANDIDATE_MODES)):
        raise ValueError("Exp27 decision fingerprint shape mismatch")
    digest = hashlib.sha256(b"exp27-selector-test-decisions-v1")
    digest.update(len(encoded_ids).to_bytes(8, "little"))
    digest.update(encoded_ids)
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _normalizer_receipt(
    raw: np.ndarray,
    transformed: np.ndarray,
    *,
    feature_names: Sequence[str],
    center: np.ndarray,
    scale: np.ndarray,
    train_n: int,
    core_fit_fingerprint: str,
) -> dict[str, object]:
    raw = np.asarray(raw, dtype=np.float64)
    transformed = np.asarray(transformed, dtype=np.float64)
    if raw.ndim != 2 or transformed.ndim != 2 or raw.shape[0] != transformed.shape[0]:
        raise ValueError("Exp27 normalizer receipt arrays have incompatible shapes")
    receipt = {
        "schema_version": "exp27_selector_normalizer_v1",
        "algorithm": "train_fold_feature_transform_and_standardization",
        "feature_names": list(feature_names),
        "center": np.asarray(center, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
        "fingerprint": core_fit_fingerprint,
        "fit_scope": TRAINING_SCOPE,
        "train_n": int(train_n),
        "n_fit_examples": int(raw.shape[0]),
        "n_raw_features": int(raw.shape[1]),
        "n_transformed_features": int(transformed.shape[1]),
        "raw_feature_mean": np.mean(raw, axis=0),
        "raw_feature_std": np.std(raw, axis=0),
        "raw_feature_min": np.min(raw, axis=0),
        "raw_feature_max": np.max(raw, axis=0),
        "transformed_feature_mean": np.mean(transformed, axis=0),
        "transformed_feature_std": np.std(transformed, axis=0),
        "independent_array_fingerprint": _array_fingerprint(
            "exp27-normalizer-training-fold-v1", raw, transformed
        ),
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt)
    return receipt


def _model_receipt(
    selector: str,
    *,
    hyperparameters: Mapping[str, object],
    fit_receipt: object | None,
    test_generator_ids: Sequence[object] | None,
    test_probabilities: np.ndarray | None,
    error: Exception | None,
) -> dict[str, object]:
    if error is None:
        if test_generator_ids is None or test_probabilities is None:
            raise ValueError("completed selector receipt requires test decisions")
        identifiers = [str(value) for value in test_generator_ids]
        probabilities = np.asarray(test_probabilities, dtype=np.float64)
        if not np.all(np.isfinite(probabilities)) or not np.allclose(
            np.sum(probabilities, axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("selector receipt test probabilities are invalid")
        decision_fingerprint: str | None = _decision_fingerprint(
            identifiers, probabilities
        )
        probability_payload: object = probabilities
    else:
        identifiers = []
        probability_payload = None
        decision_fingerprint = None
    payload: dict[str, object] = {
        "selector": selector,
        "algorithm": (
            "local_three_factor_eligibility_selector"
            if selector == "local_three_factor"
            else "deterministic_cpu_gru_bptt_baseline"
        ),
        "used_autograd": selector == "gru_bptt",
        "used_bptt": selector == "gru_bptt",
        "hyperparameters": dict(hyperparameters),
        "status": "complete" if error is None else "failed",
        "fit_receipt": _jsonable(fit_receipt) if fit_receipt is not None else None,
        "test_generator_ids": identifiers,
        "test_probabilities": _jsonable(probability_payload),
        "test_decision_fingerprint": decision_fingerprint,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
    }
    payload["receipt_sha256"] = _payload_sha256(payload)
    return payload


def _fit_learned_selectors(
    fold: object,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, tuple[float, float]],
    dict[str, object],
    dict[str, object],
    dict[str, Exception],
]:
    normalizer = SelectorFeatureNormalizer.fit(fold.train_raw_features)
    train_features = normalizer.transform(fold.train_raw_features)
    test_features = normalizer.transform(fold.test_raw_features)
    train_cues = build_three_step_cues(train_features)
    test_cues = build_three_step_cues(test_features)
    if train_cues.ndim != 3 or test_cues.ndim != 3:
        raise ValueError("Exp27 selector cues must be trial x cue x feature tensors")
    input_dim = int(train_cues.shape[-1])
    probabilities: dict[str, np.ndarray] = {}
    costs: dict[str, tuple[float, float]] = {}
    errors: dict[str, Exception] = {}
    fit_receipts: dict[str, object | None] = {}
    resolved_options: dict[str, dict[str, object]] = {}
    local_options = dict(config["local_selector"])
    local_options.update(
        input_dim=input_dim,
        shuffle_seed=derive_seed(seed, "exp27", "local_selector"),
    )
    resolved_options["local_three_factor"] = dict(local_options)
    try:
        local = _construct_registered(LocalThreeFactorSelector, local_options)
        local_receipt = local.fit(train_cues, fold.train_utilities)
        local_probabilities = np.asarray(
            local.predict_proba(test_cues), dtype=np.float64
        )
        if _receipt_value(local_receipt, "used_autograd") is not False:
            raise RuntimeError("Exp27 local selector receipt must prohibit autograd")
        if _receipt_value(local_receipt, "used_bptt") is not False:
            raise RuntimeError("Exp27 local selector receipt must prohibit BPTT")
        probabilities["local_three_factor"] = local_probabilities
        costs["local_three_factor"] = _receipt_costs(local_receipt)
        fit_receipts["local_three_factor"] = local_receipt
    except Exception as error:
        errors["local_three_factor"] = error
        fit_receipts["local_three_factor"] = None

    gru_options = dict(config["gru_selector"])
    gru_options.update(
        input_dim=input_dim,
        seed=derive_seed(seed, "exp27", "gru_selector"),
    )
    resolved_options["gru_bptt"] = dict(gru_options)
    try:
        gru = _construct_registered(GRUSelectorBaseline, gru_options)
        gru_receipt = gru.fit(train_cues, fold.train_utilities)
        gru_probabilities = np.asarray(gru.predict_proba(test_cues), dtype=np.float64)
        if _receipt_value(gru_receipt, "used_autograd") is not True:
            raise RuntimeError("Exp27 GRU baseline must disclose autograd")
        if _receipt_value(gru_receipt, "used_bptt") is not True:
            raise RuntimeError("Exp27 GRU baseline must disclose BPTT")
        probabilities["gru_bptt"] = gru_probabilities
        costs["gru_bptt"] = _receipt_costs(gru_receipt)
        fit_receipts["gru_bptt"] = gru_receipt
    except Exception as error:
        errors["gru_bptt"] = error
        fit_receipts["gru_bptt"] = None
    expected_shape = (fold.test_utilities.shape[0], len(CANDIDATE_MODES))
    for name, values in list(probabilities.items()):
        if values.shape != expected_shape:
            raise ValueError(f"Exp27 {name} probability shape mismatch")
        if not np.isfinite(values).all():
            raise ValueError(f"Exp27 {name} probabilities are non-finite")
    normalizer_receipt = _normalizer_receipt(
        np.asarray(fold.train_raw_features, dtype=np.float64),
        np.asarray(train_features, dtype=np.float64),
        feature_names=(*fold.feature_names, "bias"),
        center=np.asarray(normalizer.mean, dtype=np.float64),
        scale=np.asarray(normalizer.scale, dtype=np.float64),
        train_n=int(normalizer.n_fit_samples),
        core_fit_fingerprint=str(normalizer.fit_fingerprint),
    )
    selector_receipts: dict[str, object] = {
        "schema_version": "exp27_selector_training_receipts_v1",
        "outer_seed": seed,
        "training_scope": TRAINING_SCOPE,
        **{
            name: _model_receipt(
                name,
                hyperparameters=resolved_options[name],
                fit_receipt=fit_receipts[name],
                test_generator_ids=fold.test_generator_ids,
                test_probabilities=probabilities.get(name),
                error=errors.get(name),
            )
            for name in ("local_three_factor", "gru_bptt")
        },
    }
    selector_receipts["receipt_sha256"] = _payload_sha256(selector_receipts)
    return probabilities, costs, selector_receipts, normalizer_receipt, errors


def _selector_flags(selector: str) -> dict[str, bool]:
    return {
        "used_autograd": selector == "gru_bptt",
        "used_bptt": selector == "gru_bptt",
        "local_learning_main_model": selector == "local_three_factor",
        "bptt_baseline_isolated": selector == "gru_bptt",
        "selector_uses_outer_test_for_selection": selector == "oracle",
    }


def _one_hot(indices: np.ndarray) -> np.ndarray:
    result = np.zeros((indices.size, len(CANDIDATE_MODES)), dtype=np.float64)
    result[np.arange(indices.size), indices] = 1.0
    return result


def _row_metrics(
    *,
    selector: str,
    selected_index: int,
    probabilities: np.ndarray,
    utilities: np.ndarray,
    oracle_index: int,
    fixed_index: int,
    plasticity_cost: tuple[float, float],
) -> dict[str, object]:
    selected_utility = float(utilities[selected_index])
    oracle_utility = float(utilities[oracle_index])
    fixed_utility = float(utilities[fixed_index])
    denominator = oracle_utility - fixed_utility
    recovered = (
        (selected_utility - fixed_utility) / denominator
        if denominator > 1e-12
        else None
    )
    return {
        "status": "complete",
        "mode_selected": CANDIDATE_MODES[selected_index],
        "target_mode": CANDIDATE_MODES[oracle_index],
        "oracle_mode": CANDIDATE_MODES[oracle_index],
        "fixed_best_mode": CANDIDATE_MODES[fixed_index],
        "utility": selected_utility,
        "candidate_routing_utility": float(utilities[0]),
        "candidate_gain_utility": float(utilities[1]),
        "candidate_low_rank_utility": float(utilities[2]),
        "oracle_utility": oracle_utility,
        "fixed_best_utility": fixed_utility,
        "selection_correct": selected_index == oracle_index,
        "regret": oracle_utility - selected_utility,
        "oracle_gain_over_fixed": denominator,
        "oracle_gain_fraction_applicable": denominator > 1e-12,
        "oracle_gain_recovered_fraction": recovered,
        "beats_fixed": selected_utility > fixed_utility,
        "plasticity_l1": plasticity_cost[0],
        "plasticity_l2": plasticity_cost[1],
        "selection_probability_routing": float(probabilities[0]),
        "selection_probability_gain": float(probabilities[1]),
        "selection_probability_low_rank": float(probabilities[2]),
        **_selector_flags(selector),
    }


def _failure_metrics(selector: str, reason: str) -> dict[str, object]:
    return {
        "failure_reason": reason,
        "mode_selected": None,
        "target_mode": None,
        "oracle_mode": None,
        "fixed_best_mode": None,
        "utility": None,
        "candidate_routing_utility": None,
        "candidate_gain_utility": None,
        "candidate_low_rank_utility": None,
        "oracle_utility": None,
        "fixed_best_utility": None,
        "selection_correct": None,
        "plasticity_l1": None,
        "plasticity_l2": None,
        **_selector_flags(selector),
    }


def _audit_outer_fold(
    fold: object,
    candidates: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    seed: int,
    registered_seeds: Sequence[int],
) -> tuple[
    pd.DataFrame,
    tuple[int, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    metadata = _metadata_for_test(
        candidates,
        seed=seed,
        generator_ids=fold.test_generator_ids,
    )
    train_seeds = tuple(sorted(set(int(value) for value in fold.train_seeds)))
    if seed in train_seeds or set(train_seeds) != set(registered_seeds) - {seed}:
        raise RuntimeError("Exp27 LOSO fold leaked or omitted source seeds")
    if not np.array_equal(
        np.asarray(fold.test_seeds, dtype=np.int64),
        np.full(len(metadata), seed, dtype=np.int64),
    ):
        raise RuntimeError("Exp27 test fold is not confined to the outer seed")
    unseen = np.asarray(fold.test_unseen_composition, dtype=bool)
    overlap = np.asarray(fold.test_composition_overlap, dtype=bool)
    if unseen.shape != (len(metadata),) or not np.array_equal(unseen, ~overlap):
        raise RuntimeError("Exp27 composition-overlap audit is inconsistent")
    evaluation = config["evaluation"]
    if len(metadata) != int(evaluation["expected_heldout_generators_per_seed"]):
        raise RuntimeError("Exp27 held-out generator count mismatch")
    if int(np.sum(unseen)) != int(evaluation["expected_strict_unseen_per_seed"]):
        raise RuntimeError("Exp27 strict-unseen generator count mismatch")
    test_utilities = np.asarray(fold.test_utilities, dtype=np.float64)
    train_utilities = np.asarray(fold.train_utilities, dtype=np.float64)
    expected_test_shape = (len(metadata), len(CANDIDATE_MODES))
    if test_utilities.shape != expected_test_shape:
        raise RuntimeError("Exp27 test utility matrix shape mismatch")
    if train_utilities.ndim != 2 or train_utilities.shape[1] != len(CANDIDATE_MODES):
        raise RuntimeError("Exp27 train utility matrix shape mismatch")
    if not np.isfinite(train_utilities).all() or not np.isfinite(test_utilities).all():
        raise RuntimeError("Exp27 fold contains non-finite utilities")
    return metadata, train_seeds, unseen, overlap, train_utilities, test_utilities


def _dimension_fields(
    metadata: Mapping[str, object],
    *,
    seed: int,
    selector: str,
    strict_unseen: bool | None,
    run_label: str | None,
    source_receipt: Mapping[str, object] | None,
    config_sha256: str,
    run_git: Mapping[str, object],
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL_VERSION,
        "run_label": run_label,
        "outer_seed": seed,
        "source_seed": seed,
        "generator_split": "heldout",
        "generator_id": str(metadata["generator_id"]),
        "alpha": float(metadata["alpha"]),
        "transition_rank": int(metadata["transition_rank"]),
        "input_rank": int(metadata["input_rank"]),
        "delay": int(metadata["delay"]),
        "noise": float(metadata.get("noise", metadata.get("noise_std"))),
        "noise_std": float(metadata.get("noise_std", metadata.get("noise"))),
        "strict_unseen_composition": strict_unseen,
        "selector": selector,
        "training_scope": TRAINING_SCOPE,
        "test_scope": TEST_SCOPE,
        "statistics_unit": "seed",
        "time_points_randomly_split": False,
        "carrier_frozen": True,
        "actuator_dictionary_frozen": True,
        "actuator_basis_trained": False,
        "readout_trained": False,
        "exp27_config_sha256": config_sha256,
        "run_git_commit": run_git.get("commit"),
        "run_git_tree": run_git.get("tree"),
        "run_git_dirty": run_git.get("dirty"),
        "source_raw_metrics_sha256": (
            source_receipt.get("raw_metrics_sha256") if source_receipt else None
        ),
        "source_conclusion_sha256": (
            source_receipt.get("conclusion_sha256") if source_receipt else None
        ),
        "source_preflight_receipt_sha256": (
            source_receipt.get("preflight_receipt_sha256") if source_receipt else None
        ),
        "source_receipt_sha256": (
            source_receipt.get("receipt_sha256") if source_receipt else None
        ),
        "source_config_sha256": (
            source_receipt.get("source_config_sha256") if source_receipt else None
        ),
        "source_manifest_sha256": (
            source_receipt.get("source_manifest_sha256") if source_receipt else None
        ),
        "source_run_label": (
            source_receipt.get("source_run_label") if source_receipt else None
        ),
        "source_protocol_version": (
            source_receipt.get("source_protocol_version") if source_receipt else None
        ),
        "source_profile": (
            source_receipt.get("source_profile") if source_receipt else None
        ),
        "source_conclusion": (
            source_receipt.get("source_conclusion") if source_receipt else None
        ),
    }


def run_seed(
    config: dict[str, Any],
    seed: int,
    results_root: str | Path,
    *,
    run_label: str | None = None,
) -> Path:
    seeds = _validate_config(config)
    seed = _exact_seed(seed, name="requested seed")
    if seed not in seeds:
        raise ValueError("requested Exp27 seed is not registered")
    profile = str(config["profile"])
    if profile == "formal":
        required_label = config.get("required_run_label")
        if run_label != required_label:
            raise ValueError("formal Exp27 requires its registered shared run_label")
    peek = _peek_source_rows(config, seed)
    plan = _planned_conditions(peek, seed)
    run_git = git_identity()
    registered_source = _registered_source_provenance(config)
    config_sha256 = canonical_config_sha256(config)
    provenance = {
        "schema_version": "exp27_selector_evidence_v1",
        "canonical_config_sha256": config_sha256,
        "git": run_git,
        "run_label": run_label,
        "source": registered_source,
    }
    run_config = {**config, "evidence_provenance": provenance}
    initialize_seed(seed)
    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
        run_label=run_label,
    ) as run:
        run.register_conditions(plan)
        if profile == "formal" and run_git.get("dirty") is not False:
            error = RuntimeError("formal Exp27 requires a clean Git worktree")
            for condition in plan:
                dimensions = _dimension_fields(
                    condition,
                    seed=seed,
                    selector=str(condition["selector"]),
                    strict_unseen=None,
                    run_label=run_label,
                    source_receipt=None,
                    config_sha256=config_sha256,
                    run_git=run_git,
                )
                run.record_failed_condition(
                    _failure_metrics(str(condition["selector"]), str(error)),
                    **dimensions,
                )
            raise error
        try:
            source, candidates, source_receipt = _validate_frozen_source(config)
            fold = build_outer_seed_loso(source, outer_seed=seed)
            (
                metadata,
                train_seeds,
                unseen,
                overlap,
                train_utilities,
                test_utilities,
            ) = _audit_outer_fold(
                fold,
                candidates,
                config,
                seed=seed,
                registered_seeds=seeds,
            )
        except Exception as error:
            for condition in plan:
                dimensions = _dimension_fields(
                    condition,
                    seed=seed,
                    selector=str(condition["selector"]),
                    strict_unseen=None,
                    run_label=run_label,
                    source_receipt=None,
                    config_sha256=config_sha256,
                    run_git=run_git,
                )
                run.record_failed_condition(
                    _failure_metrics(str(condition["selector"]), str(error)),
                    **dimensions,
                )
            raise
        (run.path / "source_receipt.json").write_text(
            json.dumps(source_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        learned_probabilities: dict[str, np.ndarray] = {}
        learned_costs: dict[str, tuple[float, float]] = {}
        learned_errors: dict[str, Exception] = {}
        selector_receipts: dict[str, object]
        normalizer_receipt: dict[str, object]
        try:
            (
                learned_probabilities,
                learned_costs,
                selector_receipts,
                normalizer_receipt,
                learned_errors,
            ) = _fit_learned_selectors(fold, config, seed=seed)
        except Exception as error:
            learned_errors = {
                "local_three_factor": error,
                "gru_bptt": error,
            }
            normalizer_receipt = {
                "schema_version": "exp27_selector_normalizer_v1",
                "fit_scope": TRAINING_SCOPE,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            normalizer_receipt["receipt_sha256"] = _payload_sha256(normalizer_receipt)
            selector_receipts = {
                "schema_version": "exp27_selector_training_receipts_v1",
                "outer_seed": seed,
                "training_scope": TRAINING_SCOPE,
                **{
                    name: _model_receipt(
                        name,
                        hyperparameters=dict(
                            config[
                                f"{'local' if name == 'local_three_factor' else 'gru'}_selector"
                            ]
                        ),
                        fit_receipt=None,
                        test_generator_ids=None,
                        test_probabilities=None,
                        error=error,
                    )
                    for name in ("local_three_factor", "gru_bptt")
                },
            }
            selector_receipts["receipt_sha256"] = _payload_sha256(selector_receipts)
        normalizer_path = run.path / "normalizer_receipt.json"
        normalizer_path.write_text(
            json.dumps(_jsonable(normalizer_receipt), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selector_receipts_path = run.path / "selector_training_receipts.json"
        selector_receipts_path.write_text(
            json.dumps(_jsonable(selector_receipts), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        normalizer_file_sha256 = _file_sha256(normalizer_path)
        selector_receipts_file_sha256 = _file_sha256(selector_receipts_path)
        if not normalizer_file_sha256 or not selector_receipts_file_sha256:
            raise RuntimeError("Exp27 receipt file hashing failed")
        oracle_indices = np.argmax(test_utilities, axis=1)
        fixed_index = int(np.argmax(np.mean(train_utilities, axis=0)))
        train_mean_utilities = np.mean(train_utilities, axis=0)
        fixed_indices = np.full(len(metadata), fixed_index, dtype=np.int64)
        selector_probabilities = {
            "oracle": _one_hot(oracle_indices),
            "fixed_best": _one_hot(fixed_indices),
            **learned_probabilities,
        }
        selector_costs = {
            "oracle": (0.0, 0.0),
            "fixed_best": (0.0, 0.0),
            **learned_costs,
        }
        for index, row in metadata.iterrows():
            row_mapping = row.to_dict()
            for selector in SELECTORS:
                dimensions = _dimension_fields(
                    row_mapping,
                    seed=seed,
                    selector=selector,
                    strict_unseen=bool(unseen[index]),
                    run_label=run_label,
                    source_receipt=source_receipt,
                    config_sha256=config_sha256,
                    run_git=run_git,
                )
                common = {
                    "profile": profile,
                    "dev_only": bool(config["dev_only"]),
                    "force_inconclusive": bool(
                        config["analysis"]["force_inconclusive"]
                    ),
                    "training_source_seeds": list(train_seeds),
                    "n_training_source_seeds": len(train_seeds),
                    "training_source_seed_count": len(train_seeds),
                    "n_training_examples": int(train_utilities.shape[0]),
                    "feature_columns": list(config["feature_columns"]),
                    "normalization_fit_scope": TRAINING_SCOPE,
                    "composition_overlap_secondary": bool(overlap[index]),
                    "primary_endpoint_eligible": bool(unseen[index]),
                    "train_mean_candidate_routing_utility": float(
                        train_mean_utilities[0]
                    ),
                    "train_mean_candidate_gain_utility": float(train_mean_utilities[1]),
                    "train_mean_candidate_low_rank_utility": float(
                        train_mean_utilities[2]
                    ),
                    "outer_seed_excluded_from_training": True,
                    "train_split": "discovery",
                    "train_endpoint": "validation_balanced_accuracy",
                    "test_endpoint": "test_balanced_accuracy",
                    "selector_teacher_signal": (
                        "full_three_candidate_validation_utility_vector"
                        if selector in {"local_three_factor", "gru_bptt"}
                        else "not_applicable"
                    ),
                    "selector_teacher_counterfactual": selector
                    in {"local_three_factor", "gru_bptt"},
                    "selector_teacher_is_scalar_reward": False,
                    "hidden_belief_inference_enabled": False,
                    "task_descriptor_cues_privileged": True,
                    "update_budget_matched_across_selectors": False,
                }
                common.update(
                    {
                        "selector_training_receipts_sha256": selector_receipts[
                            "receipt_sha256"
                        ],
                        "selector_model_receipt_sha256": (
                            selector_receipts[selector]["receipt_sha256"]
                            if selector in {"local_three_factor", "gru_bptt"}
                            else None
                        ),
                        "normalizer_receipt_sha256": normalizer_receipt[
                            "receipt_sha256"
                        ],
                        "normalizer_fit_fingerprint": normalizer_receipt.get(
                            "fingerprint"
                        ),
                    }
                )
                if selector in learned_errors:
                    failed = {
                        **_failure_metrics(selector, str(learned_errors[selector])),
                        **common,
                    }
                    run.record_failed_condition(failed, **dimensions)
                    continue
                probabilities = selector_probabilities[selector][index]
                selected_index = int(np.argmax(probabilities))
                metrics = {
                    **_row_metrics(
                        selector=selector,
                        selected_index=selected_index,
                        probabilities=probabilities,
                        utilities=test_utilities[index],
                        oracle_index=int(oracle_indices[index]),
                        fixed_index=fixed_index,
                        plasticity_cost=selector_costs[selector],
                    ),
                    **common,
                }
                run.record(metrics, **dimensions)
        if profile == "formal":
            end_git = git_identity()
            if any(
                end_git.get(key) != run_git.get(key)
                for key in ("commit", "tree", "dirty")
            ):
                raise RuntimeError("formal Exp27 Git identity changed during run")
        return run.path


def _selected_seeds(config: Mapping[str, Any], override: str | None) -> Iterable[int]:
    return seed_list(override if override is not None else config["seeds"])


def main() -> None:
    parser = basic_parser(
        "Exp27 frozen-dictionary low-dimensional actuator selector",
        "configs/formal/exp27_low_dimensional_actuator_selector.json",
    )
    parser.add_argument(
        "--run-label",
        help="registered label shared across every formal outer-seed run",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    if config.get("profile") == "formal" and args.run_label != config.get(
        "required_run_label"
    ):
        parser.error("formal Exp27 requires the registered shared --run-label")
    for seed in _selected_seeds(config, args.seeds):
        print(run_seed(config, seed, args.results_root, run_label=args.run_label))


if __name__ == "__main__":
    main()
