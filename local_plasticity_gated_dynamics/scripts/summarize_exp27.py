"""Fail-closed collector and seed-level report for Exp27 selector evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp26_actuator_phase_diagram import (  # noqa: E402
    canonical_config_sha256,
    git_identity,
)
from experiments.exp27_low_dimensional_actuator_selector import (  # noqa: E402
    CANDIDATE_MODES,
    EXPERIMENT,
    PROTOCOL_VERSION,
    SELECTORS,
    _audit_outer_fold,
    _decision_fingerprint,
    _fit_learned_selectors,
    _validate_config,
    _validate_frozen_source,
)
from scripts.plot_exp27 import plot_selector_evidence  # noqa: E402
from src.analysis.actuator_selector_metrics import (  # noqa: E402
    ActuatorSelectorConclusion,
    summarize_actuator_selector,
    validate_selector_records,
)
from src.data.actuator_selector_dataset import (  # noqa: E402
    RAW_FEATURE_NAMES,
    SelectorFeatureNormalizer,
    build_outer_seed_loso,
)


DEFAULT_CONFIG_PATHS = {
    profile: PROJECT_ROOT
    / "configs"
    / profile
    / "exp27_low_dimensional_actuator_selector.json"
    for profile in ("smoke", "formal")
}
PLAN_KEYS = (
    "outer_seed",
    "source_seed",
    "generator_split",
    "generator_id",
    "alpha",
    "transition_rank",
    "input_rank",
    "delay",
    "noise",
    "selector",
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt_sha256(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    return _canonical_sha256(payload)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _matching_attempt(seed_root: Path, *, run_label: str) -> Path:
    if not seed_root.is_dir():
        raise ValueError(f"missing Exp27 seed directory: {seed_root}")
    matches: list[Path] = []
    for path in sorted(item for item in seed_root.iterdir() if item.is_dir()):
        status_path = path / "status.json"
        if not status_path.is_file():
            continue
        status = _read_json(status_path)
        if isinstance(status, Mapping) and status.get("run_label") == run_label:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"{seed_root} has {len(matches)} attempts labelled {run_label!r}; expected one"
        )
    return matches[0]


def _validate_self_hashed_receipt(
    value: object,
    *,
    name: str,
) -> tuple[Mapping[str, Any], str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    observed = value.get("receipt_sha256")
    expected = _receipt_sha256(value)
    if observed != expected:
        raise ValueError(f"{name} self-hash mismatch")
    return value, expected


def _validate_training_receipt(
    receipt: object,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[str, dict[str, str]]:
    value, digest = _validate_self_hashed_receipt(
        receipt, name="selector_training_receipts.json"
    )
    if value.get("schema_version") != "exp27_selector_training_receipts_v1":
        raise ValueError("selector training receipt schema mismatch")
    if value.get("outer_seed") != seed:
        raise ValueError("selector training receipt outer seed mismatch")
    if value.get("training_scope") != "other_seed_discovery_validation_only":
        raise ValueError("selector training receipt scope mismatch")
    contracts = {
        "local_three_factor": (False, False, config["local_selector"]),
        "gru_bptt": (True, True, config["gru_selector"]),
    }
    model_digests: dict[str, str] = {}
    for selector, (used_autograd, used_bptt, hyperparameters) in contracts.items():
        entry = value.get(selector)
        if not isinstance(entry, Mapping) or entry.get("status") != "complete":
            raise ValueError(f"{selector} training receipt is not complete")
        if entry.get("used_autograd") is not used_autograd:
            raise ValueError(f"{selector} autograd disclosure mismatch")
        if entry.get("used_bptt") is not used_bptt:
            raise ValueError(f"{selector} BPTT disclosure mismatch")
        observed_hyperparameters = entry.get("hyperparameters")
        if not isinstance(observed_hyperparameters, Mapping):
            raise ValueError(f"{selector} hyperparameters are missing")
        for key, expected in hyperparameters.items():
            if observed_hyperparameters.get(key) != expected:
                raise ValueError(f"{selector} frozen hyperparameter mismatch: {key}")
        fit_receipt = entry.get("fit_receipt")
        if not isinstance(fit_receipt, Mapping):
            raise ValueError(f"{selector} fit receipt is missing")
        if fit_receipt.get("used_autograd") is not used_autograd:
            raise ValueError(f"{selector} fit-receipt autograd mismatch")
        if fit_receipt.get("used_bptt") is not used_bptt:
            raise ValueError(f"{selector} fit-receipt BPTT mismatch")
        nested_digest = entry.get("receipt_sha256")
        nested_payload = dict(entry)
        nested_payload.pop("receipt_sha256", None)
        if nested_digest != _canonical_sha256(nested_payload):
            raise ValueError(f"{selector} model receipt self-hash mismatch")
        model_digests[selector] = str(nested_digest)
    return digest, model_digests


def _validate_normalizer_receipt(
    receipt: object,
    *,
    config: Mapping[str, Any],
) -> tuple[str, str]:
    value, digest = _validate_self_hashed_receipt(
        receipt, name="normalizer_receipt.json"
    )
    if value.get("schema_version") != "exp27_selector_normalizer_v1":
        raise ValueError("normalizer receipt schema mismatch")
    if value.get("fit_scope") != "other_seed_discovery_validation_only":
        raise ValueError("normalizer fit scope mismatch")
    if not isinstance(value.get("train_n"), int) or value["train_n"] < 1:
        raise ValueError("normalizer train_n must be a positive integer")
    feature_names = value.get("feature_names")
    expected_feature_names = [*RAW_FEATURE_NAMES, "bias"]
    if feature_names != expected_feature_names:
        raise ValueError("normalizer receipt must describe seven features plus bias")
    center = np.asarray(value.get("center"), dtype=np.float64)
    scale = np.asarray(value.get("scale"), dtype=np.float64)
    if center.shape != (7,) or scale.shape != (7,):
        raise ValueError("normalizer center/scale shape mismatch")
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("normalizer center/scale are non-finite")
    if np.any(scale <= 0.0):
        raise ValueError("normalizer scales must be positive")
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("normalizer fingerprint must be a SHA-256")
    expected_fingerprint = _canonical_sha256(
        {
            "feature_names": list(RAW_FEATURE_NAMES),
            "mean": center.tolist(),
            "scale": scale.tolist(),
            "n_fit_samples": value["train_n"],
        }
    )
    if fingerprint != expected_fingerprint:
        raise ValueError("normalizer fit fingerprint is inconsistent")
    if tuple(config["feature_columns"]) != (
        "chi",
        "state_demand",
        "input_demand",
        "transition_rank",
        "input_rank",
        "delay",
        "noise_std",
    ):
        raise ValueError("registered feature order changed")
    return digest, fingerprint


def _literal_boolean_column(
    frame: pd.DataFrame,
    column: str,
    *,
    expected: np.ndarray,
) -> None:
    if column not in frame:
        raise ValueError(f"Exp27 reconstructed fold field is missing: {column}")
    observed = frame[column].tolist()
    if not all(isinstance(value, (bool, np.bool_)) for value in observed):
        raise ValueError(f"Exp27 reconstructed fold field is not boolean: {column}")
    if not np.array_equal(np.asarray(observed, dtype=bool), expected):
        raise ValueError(f"Exp27 reconstructed fold mismatch: {column}")


def _exact_numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    expected: np.ndarray,
) -> None:
    if column not in frame:
        raise ValueError(f"Exp27 reconstructed fold field is missing: {column}")
    observed = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    if observed.shape != expected_values.shape or not np.array_equal(
        observed, expected_values
    ):
        raise ValueError(f"Exp27 reconstructed fold mismatch: {column}")


def _validate_reconstructed_fold(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    seed: int,
    seeds: tuple[int, ...],
    frozen_source: object,
    frozen_candidates: pd.DataFrame,
    training_receipt: Mapping[str, Any],
    normalizer_receipt: Mapping[str, Any],
    training_digest: str,
    normalizer_digest: str,
) -> None:
    """Recompute a complete outer fold from the immutable Exp26 source.

    Row-level provenance strings are not sufficient evidence that a metric was
    actually derived from the registered source.  This audit therefore rebuilds
    the fold, normalization, and deterministic learned selectors and requires
    exact agreement with both the raw rows and their serialized receipts.
    """

    fold = build_outer_seed_loso(frozen_source, outer_seed=seed)
    (
        metadata,
        train_seeds,
        unseen,
        overlap,
        train_utilities,
        test_utilities,
    ) = _audit_outer_fold(
        fold,
        frozen_candidates,
        config,
        seed=seed,
        registered_seeds=seeds,
    )
    expected_ids = tuple(str(value) for value in metadata["generator_id"])
    row_ids = frame["generator_id"].astype(str).tolist()
    id_to_index = {identifier: index for index, identifier in enumerate(expected_ids)}
    if len(id_to_index) != len(expected_ids) or any(
        identifier not in id_to_index for identifier in row_ids
    ):
        raise ValueError("Exp27 rows differ from reconstructed source generators")
    generator_counts = frame["generator_id"].astype(str).value_counts()
    if set(generator_counts.index) != set(expected_ids) or not bool(
        generator_counts.eq(len(SELECTORS)).all()
    ):
        raise ValueError("Exp27 reconstructed generator coverage mismatch")
    row_indices = np.asarray([id_to_index[identifier] for identifier in row_ids])

    metadata_contract = {
        "alpha": metadata["alpha"].to_numpy(dtype=np.float64),
        "transition_rank": metadata["transition_rank"].to_numpy(dtype=np.float64),
        "input_rank": metadata["input_rank"].to_numpy(dtype=np.float64),
        "delay": metadata["delay"].to_numpy(dtype=np.float64),
        "noise": metadata["noise_std"].to_numpy(dtype=np.float64),
        "noise_std": metadata["noise_std"].to_numpy(dtype=np.float64),
    }
    for column, source_values in metadata_contract.items():
        _exact_numeric_column(
            frame,
            column,
            expected=np.asarray(source_values)[row_indices],
        )
    _literal_boolean_column(
        frame,
        "strict_unseen_composition",
        expected=unseen[row_indices],
    )
    _literal_boolean_column(
        frame,
        "primary_endpoint_eligible",
        expected=unseen[row_indices],
    )
    _literal_boolean_column(
        frame,
        "composition_overlap_secondary",
        expected=overlap[row_indices],
    )

    candidate_columns = [f"candidate_{mode}_utility" for mode in CANDIDATE_MODES]
    observed_test_utilities = (
        frame[candidate_columns]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float64)
    )
    if not np.array_equal(observed_test_utilities, test_utilities[row_indices]):
        raise ValueError("Exp27 rows differ from reconstructed source test utilities")

    expected_train_seeds = tuple(int(value) for value in train_seeds)
    for value in frame["training_source_seeds"]:
        if not isinstance(value, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in value
        ):
            raise ValueError("Exp27 training_source_seeds must contain exact integers")
        if tuple(value) != expected_train_seeds:
            raise ValueError("Exp27 rows differ from reconstructed training source seeds")
    expected_train_n = int(train_utilities.shape[0])
    for column, expected_value in (
        ("n_training_examples", expected_train_n),
        ("n_training_source_seeds", len(expected_train_seeds)),
        ("training_source_seed_count", len(expected_train_seeds)),
    ):
        _exact_numeric_column(
            frame,
            column,
            expected=np.full(frame.shape[0], expected_value, dtype=np.float64),
        )
    train_mean_columns = [
        f"train_mean_candidate_{mode}_utility" for mode in CANDIDATE_MODES
    ]
    observed_train_means = (
        frame[train_mean_columns]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float64)
    )
    expected_train_means = np.broadcast_to(
        np.mean(train_utilities, axis=0), observed_train_means.shape
    )
    if not np.array_equal(observed_train_means, expected_train_means):
        raise ValueError("Exp27 rows differ from reconstructed training utility means")

    independent_normalizer = SelectorFeatureNormalizer.fit(fold.train_raw_features)
    receipt_center = np.asarray(normalizer_receipt.get("center"), dtype=np.float64)
    receipt_scale = np.asarray(normalizer_receipt.get("scale"), dtype=np.float64)
    if not np.array_equal(receipt_center, independent_normalizer.mean):
        raise ValueError("Exp27 normalizer center differs from reconstructed fold")
    if not np.array_equal(receipt_scale, independent_normalizer.scale):
        raise ValueError("Exp27 normalizer scale differs from reconstructed fold")
    if normalizer_receipt.get("fingerprint") != independent_normalizer.fit_fingerprint:
        raise ValueError("Exp27 normalizer fingerprint differs from reconstructed fold")

    (
        reconstructed_probabilities,
        _,
        reconstructed_training_receipt,
        reconstructed_normalizer_receipt,
        reconstruction_errors,
    ) = _fit_learned_selectors(fold, config, seed=seed)
    if reconstruction_errors:
        details = ", ".join(
            f"{name}: {type(error).__name__}" for name, error in reconstruction_errors.items()
        )
        raise ValueError(f"Exp27 selector replay failed: {details}")
    if reconstructed_training_receipt.get("receipt_sha256") != training_digest:
        raise ValueError("Exp27 selector training receipt differs from deterministic replay")
    if reconstructed_normalizer_receipt.get("receipt_sha256") != normalizer_digest:
        raise ValueError("Exp27 normalizer receipt differs from deterministic replay")

    train_mean = np.mean(train_utilities, axis=0)
    oracle_indices = np.argmax(test_utilities, axis=1)
    fixed_index = int(np.argmax(train_mean))
    oracle_probabilities = np.zeros_like(test_utilities)
    oracle_probabilities[np.arange(test_utilities.shape[0]), oracle_indices] = 1.0
    fixed_probabilities = np.zeros_like(test_utilities)
    fixed_probabilities[:, fixed_index] = 1.0
    expected_probabilities = {
        "oracle": oracle_probabilities,
        "fixed_best": fixed_probabilities,
        **reconstructed_probabilities,
    }
    probability_columns = [
        "selection_probability_routing",
        "selection_probability_gain",
        "selection_probability_low_rank",
    ]
    for selector in SELECTORS:
        selector_rows = frame[frame["selector"] == selector].copy()
        selector_rows["generator_id"] = selector_rows["generator_id"].astype(str)
        selector_rows = selector_rows.set_index("generator_id").loc[list(expected_ids)]
        observed_probabilities = selector_rows[probability_columns].to_numpy(
            dtype=np.float64
        )
        expected = expected_probabilities[selector]
        if not np.array_equal(observed_probabilities, expected):
            raise ValueError(
                f"Exp27 {selector} decisions differ from deterministic replay"
            )
        if selector in {"local_three_factor", "gru_bptt"}:
            expected_fingerprint = _decision_fingerprint(expected_ids, expected)
            if (
                training_receipt[selector].get("test_decision_fingerprint")
                != expected_fingerprint
            ):
                raise ValueError(
                    f"Exp27 {selector} decision fingerprint differs from replay"
                )


@dataclass(frozen=True)
class AttemptAudit:
    seed: int
    path: str
    planned_conditions: int
    observed_rows: int
    run_git_commit: str
    run_git_tree: str
    source_receipt_sha256: str
    training_receipt_sha256: str
    normalizer_receipt_sha256: str
    normalizer_fit_fingerprint: str


@dataclass(frozen=True)
class SelectorCollection:
    raw: pd.DataFrame
    config: Mapping[str, Any]
    config_sha256: str
    source_receipt: Mapping[str, Any]
    source_receipt_sha256: str
    attempts: tuple[AttemptAudit, ...]
    run_git_commit: str
    run_git_tree: str
    runtime_identity: Mapping[str, Any]
    run_label: str


def collect_exp27_runs(
    results_root: str | Path,
    *,
    config_path: str | Path,
    run_label: str,
) -> SelectorCollection:
    """Collect exactly one explicit attempt per registered outer seed."""

    if not isinstance(run_label, str) or not run_label:
        raise ValueError("run_label must be explicit and non-empty")
    config_value = _read_json(Path(config_path))
    if not isinstance(config_value, Mapping):
        raise ValueError("Exp27 config must be a JSON object")
    config = dict(config_value)
    seeds = _validate_config(config)
    if config["profile"] == "formal" and run_label != config["required_run_label"]:
        raise ValueError("formal run label differs from registration")
    config_sha = canonical_config_sha256(config)
    frozen_source, frozen_candidates, expected_source_receipt = (
        _validate_frozen_source(config)
    )
    root = Path(results_root) / "runs" / EXPERIMENT
    frames: list[pd.DataFrame] = []
    audits: list[AttemptAudit] = []
    source_identity: Mapping[str, Any] | None = None
    source_digest: str | None = None
    run_git_identity: tuple[str, str] | None = None
    runtime_identity: Mapping[str, Any] | None = None
    for seed in seeds:
        attempt = _matching_attempt(root / f"seed_{seed:04d}", run_label=run_label)
        status = _read_json(attempt / "status.json")
        manifest = _read_json(attempt / "manifest.json")
        run_config = _read_json(attempt / "config.json")
        environment = _read_json(attempt / "environment.json")
        planned = _read_json(attempt / "planned_conditions.json")
        metrics = _read_jsonl(attempt / "metrics.jsonl")
        source_receipt = _read_json(attempt / "source_receipt.json")
        training_receipt = _read_json(attempt / "selector_training_receipts.json")
        normalizer_receipt = _read_json(attempt / "normalizer_receipt.json")
        if not all(
            isinstance(value, Mapping) for value in (status, manifest, run_config)
        ):
            raise ValueError(f"Exp27 attempt metadata is malformed: {attempt}")
        if status.get("status") != "complete" or manifest.get("status") != "complete":
            raise ValueError(f"Exp27 attempt is not complete: {attempt}")
        if (
            status.get("condition_failures") != 0
            or status.get("condition_invalid") != 0
        ):
            raise ValueError(f"Exp27 attempt contains failed conditions: {attempt}")
        if status.get("seed") != seed or manifest.get("seed") != seed:
            raise ValueError("Exp27 attempt seed metadata mismatch")
        if (
            status.get("run_label") != run_label
            or manifest.get("run_label") != run_label
        ):
            raise ValueError("Exp27 attempt run-label metadata mismatch")
        for key, expected in config.items():
            if run_config.get(key) != expected:
                raise ValueError(f"Exp27 run config differs from registration: {key}")
        provenance = run_config.get("evidence_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Exp27 evidence provenance is missing")
        if provenance.get("canonical_config_sha256") != config_sha:
            raise ValueError("Exp27 canonical config receipt mismatch")
        if provenance.get("run_label") != run_label:
            raise ValueError("Exp27 provenance run label mismatch")
        run_git = provenance.get("git")
        environment_git = (
            environment.get("git") if isinstance(environment, Mapping) else None
        )
        if not isinstance(run_git, Mapping) or run_git != environment_git:
            raise ValueError("Exp27 run/environment Git provenance mismatch")
        if config["profile"] == "formal" and run_git.get("dirty") is not False:
            raise ValueError("formal Exp27 provenance is dirty")
        commit, tree = run_git.get("commit"), run_git.get("tree")
        if not isinstance(commit, str) or not isinstance(tree, str):
            raise ValueError("Exp27 Git commit/tree are missing")
        if run_git_identity is None:
            run_git_identity = (commit, tree)
        elif run_git_identity != (commit, tree):
            raise ValueError("Exp27 Git commit/tree changed across outer seeds")
        packages = environment.get("packages")
        if not isinstance(packages, Mapping) or not isinstance(
            environment.get("python"), str
        ):
            raise ValueError("Exp27 runtime provenance is incomplete")
        current_runtime = {
            "python": environment["python"],
            "packages": dict(packages),
        }
        if runtime_identity is None:
            runtime_identity = current_runtime
        elif runtime_identity != current_runtime:
            raise ValueError("Exp27 Python/package runtime changed across outer seeds")

        source_receipt, observed_source_digest = _validate_self_hashed_receipt(
            source_receipt, name="source_receipt.json"
        )
        if source_receipt != expected_source_receipt:
            raise ValueError(
                "Exp27 attempt source receipt differs from the registered locked source"
            )
        if source_identity is None:
            source_identity = source_receipt
            source_digest = observed_source_digest
        elif source_receipt != source_identity:
            raise ValueError("Exp27 source receipt changed across outer seeds")
        provenance_source = provenance.get("source")
        if not isinstance(provenance_source, Mapping):
            raise ValueError("Exp27 registered source provenance is missing")
        if not set(provenance_source).issubset(source_receipt):
            raise ValueError("Exp27 source receipt lacks registered provenance")
        if provenance_source != {key: source_receipt[key] for key in provenance_source}:
            raise ValueError("Exp27 run provenance/source receipt mismatch")
        training_digest, model_digests = _validate_training_receipt(
            training_receipt, config=config, seed=seed
        )
        normalizer_digest, normalizer_fingerprint = _validate_normalizer_receipt(
            normalizer_receipt, config=config
        )

        expected_rows = int(
            config["evaluation"]["expected_heldout_generators_per_seed"]
        ) * len(SELECTORS)
        if not isinstance(planned, list) or len(planned) != expected_rows:
            raise ValueError("Exp27 planned-condition count mismatch")
        if len(metrics) != expected_rows:
            raise ValueError("Exp27 raw row count differs from registered plan")
        normalized_plan: list[dict[str, Any]] = []
        for index, row in enumerate(planned):
            if not isinstance(row, Mapping) or row.get("condition_index") != index:
                raise ValueError("Exp27 planned-condition indexes are not contiguous")
            normalized_plan.append({key: row.get(key) for key in PLAN_KEYS})
        normalized_metrics = [
            {key: row.get(key) for key in PLAN_KEYS} for row in metrics
        ]
        if {_canonical_sha256(row) for row in normalized_plan} != {
            _canonical_sha256(row) for row in normalized_metrics
        }:
            raise ValueError("Exp27 raw rows do not cover the registered plan")
        if len({_canonical_sha256(row) for row in normalized_metrics}) != expected_rows:
            raise ValueError("Exp27 raw selector conditions are duplicated")
        frame = pd.DataFrame(metrics)
        row_bindings: tuple[tuple[str, object], ...] = (
            ("protocol", PROTOCOL_VERSION),
            ("run_label", run_label),
            ("profile", config["profile"]),
            ("dev_only", bool(config["dev_only"])),
            ("exp27_config_sha256", config_sha),
            ("run_git_commit", commit),
            ("run_git_tree", tree),
            ("run_git_dirty", run_git.get("dirty")),
            ("source_raw_metrics_sha256", source_receipt["raw_metrics_sha256"]),
            ("source_conclusion_sha256", source_receipt["conclusion_sha256"]),
            ("source_receipt_sha256", observed_source_digest),
            ("source_config_sha256", source_receipt["source_config_sha256"]),
            ("source_manifest_sha256", source_receipt["source_manifest_sha256"]),
            ("source_run_label", source_receipt["source_run_label"]),
            ("source_protocol_version", source_receipt["source_protocol_version"]),
            ("source_profile", source_receipt["source_profile"]),
            ("source_conclusion", source_receipt["source_conclusion"]),
        )
        for column, expected in row_bindings:
            if column not in frame:
                raise ValueError(f"Exp27 row provenance is missing: {column}")
            observed_values = frame[column].tolist()
            if not all(value == expected for value in observed_values):
                raise ValueError(f"Exp27 row provenance mismatch: {column}")
        expected_preflight = source_receipt["preflight_receipt_sha256"]
        if expected_preflight is None:
            if frame["source_preflight_receipt_sha256"].notna().any():
                raise ValueError("Exp27 smoke rows unexpectedly claim a preflight")
        elif not bool(
            (frame["source_preflight_receipt_sha256"] == expected_preflight).all()
        ):
            raise ValueError("Exp27 row source-preflight binding failed")
        for column, expected in (
            ("selector_training_receipts_sha256", training_digest),
            ("normalizer_receipt_sha256", normalizer_digest),
            ("normalizer_fit_fingerprint", normalizer_fingerprint),
            ("source_receipt_sha256", observed_source_digest),
        ):
            if (
                column not in frame
                or not bool(frame[column].notna().all())
                or not bool((frame[column].astype(str) == expected).all())
            ):
                raise ValueError(f"Exp27 row receipt binding failed: {column}")
        if set(frame["source_raw_metrics_sha256"].dropna().astype(str)) != {
            str(source_receipt["raw_metrics_sha256"])
        }:
            raise ValueError("Exp27 rows are not bound to the source receipt")
        model_hash_rows = frame[frame["selector"].isin(model_digests)]
        for selector, expected in model_digests.items():
            observed = model_hash_rows.loc[
                model_hash_rows["selector"] == selector,
                "selector_model_receipt_sha256",
            ]
            if not bool(observed.notna().all()) or not bool(
                (observed.astype(str) == expected).all()
            ):
                raise ValueError(f"Exp27 row/model receipt binding failed: {selector}")
        probability_columns = [
            "selection_probability_routing",
            "selection_probability_gain",
            "selection_probability_low_rank",
        ]
        missing_probability = set(probability_columns) - set(frame.columns)
        if missing_probability:
            raise ValueError("Exp27 selector probabilities are missing")
        row_probabilities = (
            frame[probability_columns]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float64)
        )
        if not np.all(np.isfinite(row_probabilities)) or not np.allclose(
            np.sum(row_probabilities, axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("Exp27 row selector probabilities are invalid")
        expected_modes = np.asarray(CANDIDATE_MODES)[
            np.argmax(row_probabilities, axis=1)
        ]
        if not np.array_equal(frame["mode_selected"].to_numpy(), expected_modes):
            raise ValueError("Exp27 row probabilities do not select mode_selected")
        for selector in model_digests:
            selector_rows = frame[frame["selector"] == selector]
            model_receipt = training_receipt[selector]
            receipt_ids = model_receipt.get("test_generator_ids")
            receipt_probabilities = np.asarray(
                model_receipt.get("test_probabilities"), dtype=np.float64
            )
            row_ids = selector_rows["generator_id"].astype(str).tolist()
            row_selector_probabilities = selector_rows[probability_columns].to_numpy(
                dtype=np.float64
            )
            if receipt_ids != row_ids or not np.array_equal(
                receipt_probabilities, row_selector_probabilities
            ):
                raise ValueError(f"{selector} receipt does not bind row decisions")
            expected_decision_fingerprint = _decision_fingerprint(
                row_ids, row_selector_probabilities
            )
            if (
                model_receipt.get("test_decision_fingerprint")
                != expected_decision_fingerprint
            ):
                raise ValueError(f"{selector} decision fingerprint mismatch")
        reference_hashes = frame.loc[
            frame["selector"].isin({"oracle", "fixed_best"}),
            "selector_model_receipt_sha256",
        ]
        if reference_hashes.notna().any():
            raise ValueError("oracle/fixed rows must not claim a learned-model receipt")
        training_examples = set(
            pd.to_numeric(frame["n_training_examples"], errors="raise").astype(int)
        )
        if training_examples != {int(normalizer_receipt["train_n"])}:
            raise ValueError("normalizer and row training counts differ")
        if set(
            pd.to_numeric(frame["training_source_seed_count"], errors="raise").astype(
                int
            )
        ) != {len(seeds) - 1}:
            raise ValueError("row LOSO training-seed count mismatch")
        for selector in model_digests:
            fit_receipt = training_receipt[selector]["fit_receipt"]
            if not isinstance(fit_receipt, Mapping):
                raise ValueError(f"{selector} fit receipt is missing")
            selector_rows = frame[frame["selector"] == selector]
            if int(fit_receipt.get("n_samples", -1)) not in training_examples:
                raise ValueError(f"{selector} fit/row training counts differ")
            for column, receipt_key in (
                ("plasticity_l1", "cumulative_update_l1"),
                ("plasticity_l2", "cumulative_update_l2"),
            ):
                expected_cost = float(fit_receipt[receipt_key])
                observed_cost = selector_rows[column].to_numpy(dtype=np.float64)
                if not np.allclose(observed_cost, expected_cost, rtol=0.0, atol=1e-12):
                    raise ValueError(f"{selector} row/update-cost binding failed")
        _validate_reconstructed_fold(
            frame,
            config=config,
            seed=seed,
            seeds=seeds,
            frozen_source=frozen_source,
            frozen_candidates=frozen_candidates,
            training_receipt=training_receipt,
            normalizer_receipt=normalizer_receipt,
            training_digest=training_digest,
            normalizer_digest=normalizer_digest,
        )
        frame["_attempt_path"] = str(attempt.resolve())
        frame["_run_git_commit"] = commit
        frame["_run_git_tree"] = tree
        frames.append(frame)
        audits.append(
            AttemptAudit(
                seed=seed,
                path=str(attempt.resolve()),
                planned_conditions=expected_rows,
                observed_rows=len(metrics),
                run_git_commit=commit,
                run_git_tree=tree,
                source_receipt_sha256=observed_source_digest,
                training_receipt_sha256=training_digest,
                normalizer_receipt_sha256=normalizer_digest,
                normalizer_fit_fingerprint=normalizer_fingerprint,
            )
        )
    if (
        source_identity is None
        or source_digest is None
        or run_git_identity is None
        or runtime_identity is None
    ):
        raise ValueError("Exp27 collection is empty")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    validate_selector_records(
        raw,
        expected_seeds=seeds,
        expected_primary_generators_per_seed=int(
            config["evaluation"]["expected_strict_unseen_per_seed"]
        ),
    )
    if config["profile"] == "formal":
        current_git = git_identity()
        if (
            current_git.get("dirty") is not False
            or (current_git.get("commit"), current_git.get("tree")) != run_git_identity
        ):
            raise ValueError(
                "formal Exp27 summary must run clean on the evidence commit/tree"
            )
    return SelectorCollection(
        raw=raw,
        config=config,
        config_sha256=config_sha,
        source_receipt=source_identity,
        source_receipt_sha256=source_digest,
        attempts=tuple(audits),
        run_git_commit=run_git_identity[0],
        run_git_tree=run_git_identity[1],
        runtime_identity=runtime_identity,
        run_label=run_label,
    )


def _summary_table(conclusion: ActuatorSelectorConclusion) -> pd.DataFrame:
    endpoints = pd.DataFrame(asdict(item) for item in conclusion.seed_endpoints)
    metrics = (
        "routing_utility",
        "gain_utility",
        "low_rank_utility",
        "fixed_best_utility",
        "oracle_utility",
        "gru_bptt_utility",
        "local_three_factor_utility",
        "local_minus_fixed_best",
        "oracle_minus_fixed_best",
        "local_noninferiority_contrast",
        "local_selection_accuracy",
        "gru_selection_accuracy",
        "local_update_l1",
        "local_update_l2",
        "gru_update_l1",
        "gru_update_l2",
    )
    rows = []
    for metric in metrics:
        values = endpoints[metric].to_numpy(dtype=np.float64)
        rows.append(
            {
                "metric": metric,
                "statistics_unit": "outer_seed",
                "n": values.size,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def _report_markdown(
    collection: SelectorCollection,
    conclusion: ActuatorSelectorConclusion,
    *,
    raw_metrics_sha256: str,
) -> str:
    primary_rows = []
    for item in conclusion.primary_contrasts:
        primary_rows.append(
            "| {name} | {mean:.6f} | [{lower:.6f}, {upper:.6f}] | {p:.6g} | {holm:.6g} | {opp_holm:.6g} |".format(
                name=item.name,
                mean=item.mean,
                lower=item.lower_confidence,
                upper=item.upper_confidence,
                p=item.p_value,
                holm=item.p_value_holm,
                opp_holm=item.opposition_p_value_holm,
            )
        )
    endpoints = pd.DataFrame(asdict(item) for item in conclusion.seed_endpoints)
    utility_rows = []
    for label, column in (
        ("Fixed best", "fixed_best_utility"),
        ("Local three-factor", "local_three_factor_utility"),
        ("GRU-BPTT", "gru_bptt_utility"),
        ("Oracle ceiling", "oracle_utility"),
    ):
        values = endpoints[column].to_numpy(dtype=np.float64)
        utility_rows.append(
            f"| {label} | {np.mean(values):.6f} | {np.std(values, ddof=1):.6f} |"
        )
    return f"""# Exp27 low-dimensional actuator selector

## Conclusion

**{conclusion.conclusion.upper()}** — {conclusion.reason}.

The confirmatory unit is the outer network seed (`n={conclusion.n_seeds}`),
and only strict-unseen `(alpha, transition_rank, input_rank)` compositions enter
the primary endpoints. Generator cells are paired prediction targets, not
independent replicates.

## Registered primary endpoints

| Endpoint | Seed mean | 95% seed-bootstrap CI | Positive p | Positive Holm p | Negative Holm p |
|---|---:|---:|---:|---:|---:|
{chr(10).join(primary_rows)}

Support requires both the local-vs-fixed gain and the 0.8-oracle
non-inferiority contrast to have a positive lower confidence bound and
Holm-adjusted `p < 0.05`.

## Held-out utility

| Policy | Seed mean | Seed SD |
|---|---:|---:|
{chr(10).join(utility_rows)}

## Provenance and coverage

- Profile: `{collection.config["profile"]}`; run label: `{collection.run_label}`.
- Git commit/tree: `{collection.run_git_commit}` / `{collection.run_git_tree}`.
- Python runtime: `{str(collection.runtime_identity["python"]).split()[0]}`.
- Canonical Exp27 config SHA-256: `{collection.config_sha256}`.
- Frozen Exp26 raw SHA-256: `{collection.source_receipt["raw_metrics_sha256"]}`.
- Frozen Exp26 conclusion SHA-256: `{collection.source_receipt["conclusion_sha256"]}`.
- Collected Exp27 raw-metrics SHA-256: `{raw_metrics_sha256}`.
- Complete outer seeds: {len(collection.attempts)}; every planned selector row was retained.
- Local main method: no autograd and no BPTT. GRU-BPTT is an isolated baseline.

## Interpretation boundary

Exp27 selects among frozen, task-matched **actuator-family policies**; it does
not learn recurrent weights or prove a single global biophysical motif
dictionary. Its prospective demand cues are generator-provided task
descriptors, not an online hidden-state belief inferred from observations.
The local third factor is the full three-candidate validation-utility vector,
not a scalar reward from only the selected action, and its update-cost proxy is
not budget-matched to the differently parameterized GRU. Accordingly, a
support result establishes supervised unseen-composition family selection,
not hidden-context inference, de-novo motif formation, or an independent
temporal-credit mechanism.

The 30 LOSO endpoints reuse highly overlapping meta-training seed sets, so
their seed bootstrap and sign-flip inference is a cross-fitted, conditional
analysis rather than 30 independently trained selectors. A fixed meta-train
versus independent test-seed split remains the appropriate sensitivity check.
Also, "strict unseen" is registered only for the `(alpha, transition rank,
input rank)` triple, not every delay/noise/rotation coordinate.
"""


def write_exp27_summary(
    collection: SelectorCollection,
    output_dir: str | Path,
    *,
    make_figure: bool = True,
) -> ActuatorSelectorConclusion:
    """Write the complete summary bundle from an already audited collection."""

    config = collection.config
    analysis = config["analysis"]
    conclusion = summarize_actuator_selector(
        collection.raw,
        expected_seeds=tuple(int(value) for value in config["seeds"]),
        expected_primary_generators_per_seed=int(
            config["evaluation"]["expected_strict_unseen_per_seed"]
        ),
        noninferiority_fraction=float(
            config["evaluation"]["local_oracle_gain_fraction_threshold"]
        ),
        bootstrap_samples=int(analysis["bootstrap_samples"]),
        permutation_samples=int(analysis["permutation_samples"]),
        confidence=float(analysis["confidence"]),
        random_seed=int(analysis["statistics_seed"]),
        force_inconclusive=bool(analysis["force_inconclusive"]),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    endpoints = pd.DataFrame(asdict(item) for item in conclusion.seed_endpoints)
    summary = _summary_table(conclusion)
    raw_path = output / "raw_metrics.csv.gz"
    collection.raw.to_csv(
        raw_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    raw_metrics_sha256 = _file_sha256(raw_path)
    endpoints.to_csv(output / "seed_endpoints.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    payload = {
        **conclusion.to_dict(),
        "profile": config["profile"],
        "dev_only": bool(config["dev_only"]),
        "confirmatory_eligible": bool(
            config["profile"] == "formal" and not config["dev_only"]
        ),
        "protocol_version": PROTOCOL_VERSION,
        "run_label": collection.run_label,
        "canonical_config_sha256": collection.config_sha256,
        "source_receipt_sha256": collection.source_receipt_sha256,
        "run_git_commit": collection.run_git_commit,
        "run_git_tree": collection.run_git_tree,
        "runtime_identity": collection.runtime_identity,
        "raw_metrics_sha256": raw_metrics_sha256,
        "inference_scope": (
            "cross_fitted_outer_seed_endpoints_with_overlapping_training_folds"
        ),
        "attempts": [asdict(item) for item in collection.attempts],
    }
    _write_json(output / "conclusion.json", payload)
    _write_json(
        output / "provenance.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "run_label": collection.run_label,
            "canonical_config_sha256": collection.config_sha256,
            "source_receipt": collection.source_receipt,
            "source_receipt_sha256": collection.source_receipt_sha256,
            "run_git_commit": collection.run_git_commit,
            "run_git_tree": collection.run_git_tree,
            "runtime_identity": collection.runtime_identity,
            "raw_metrics_sha256": raw_metrics_sha256,
            "inference_scope": (
                "cross_fitted_outer_seed_endpoints_with_overlapping_training_folds"
            ),
            "attempts": [asdict(item) for item in collection.attempts],
        },
    )
    (output / "report.md").write_text(
        _report_markdown(
            collection,
            conclusion,
            raw_metrics_sha256=raw_metrics_sha256,
        ),
        encoding="utf-8",
    )
    if make_figure:
        plot_selector_evidence(endpoints, output / "exp27_selector_evidence")
    return conclusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--profile", choices=("smoke", "formal"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()
    config_path = args.config or DEFAULT_CONFIG_PATHS[args.profile]
    collection = collect_exp27_runs(
        args.results_root,
        config_path=config_path,
        run_label=args.run_label,
    )
    if collection.config["profile"] != args.profile:
        raise ValueError("requested profile differs from registered config")
    conclusion = write_exp27_summary(
        collection, args.output_dir, make_figure=not args.skip_figure
    )
    print(args.output_dir)
    print(conclusion.conclusion)


if __name__ == "__main__":
    main()
