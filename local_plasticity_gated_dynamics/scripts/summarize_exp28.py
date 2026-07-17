"""Audit and replay the non-confirmatory Exp28 amended sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp26_actuator_phase_diagram import (  # noqa: E402
    canonical_config_sha256,
    git_identity,
)
from experiments.exp28_independent_actuator_selector import (  # noqa: E402
    CANDIDATE_MODES,
    CONFIRMATORY_ELIGIBLE,
    DECISION_RECEIPT_SCHEMA,
    EXPECTED_EVALUATION_SEEDS,
    EXPECTED_META_SEEDS,
    EXPERIMENT,
    FIT_RECEIPT_SCHEMA,
    INFERENCE_SCOPE,
    INFERENCE_STATUS,
    PROTOCOL_VERSION,
    REQUIRED_RUN_LABEL,
    SELECTORS,
    SOURCE_RECEIPT_SCHEMA,
    _decision_fingerprint,
    _file_sha256,
    _fit_frozen_selectors,
    _load_sources,
    _payload_sha256,
    _planned_conditions,
    _semantic_records,
    _validate_config,
)
from scripts.plot_exp27 import plot_selector_evidence  # noqa: E402
from src.analysis.actuator_phase_statistics import holm_adjust  # noqa: E402
from src.analysis.actuator_selector_metrics import (  # noqa: E402
    ActuatorSelectorConclusion,
    SeedSelectorEndpoint,
    _contrast_summary,
    _finite_vector,
    _one_sided_sign_flip,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "formal" / "exp28_independent_actuator_selector.json"
)
PLAN_KEYS = (
    "selector_fit_seed",
    "evaluation_seed",
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
PROBABILITY_COLUMNS = (
    "selection_probability_routing",
    "selection_probability_gain",
    "selection_probability_low_rank",
)
CANDIDATE_UTILITY_COLUMNS = tuple(
    f"candidate_{mode}_utility" for mode in CANDIDATE_MODES
)
TRAIN_UTILITY_COLUMNS = tuple(
    f"train_mean_candidate_{mode}_utility" for mode in CANDIDATE_MODES
)


@dataclass(frozen=True)
class Exp28Collection:
    raw: pd.DataFrame
    config: Mapping[str, Any]
    config_sha256: str
    attempt_path: str
    run_git_commit: str
    run_git_tree: str
    runtime_identity: Mapping[str, Any]
    source_receipt: Mapping[str, Any]
    source_receipt_file_sha256: str
    fit_receipt: Mapping[str, Any]
    fit_receipt_file_sha256: str
    decision_receipt: Mapping[str, Any]
    decision_receipt_file_sha256: str
    run_label: str


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank Exp28 row at {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid Exp28 JSON row at line {line_number}") from error
        if not isinstance(value, dict):
            raise ValueError("every Exp28 metric row must be a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _matching_attempt(
    results_root: Path,
    *,
    selector_fit_seed: int,
    run_label: str,
) -> Path:
    seed_root = results_root / "runs" / EXPERIMENT / f"seed_{selector_fit_seed:04d}"
    if not seed_root.is_dir():
        raise ValueError(f"missing Exp28 fit-seed directory: {seed_root}")
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
            f"Exp28 has {len(matches)} attempts labelled {run_label!r}; expected one"
        )
    return matches[0]


def _validate_self_hash(
    value: object,
    *,
    name: str,
    schema: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    payload = dict(value)
    observed = payload.pop("receipt_sha256", None)
    if value.get("schema_version") != schema or observed != _payload_sha256(payload):
        raise ValueError(f"{name} schema or self-hash mismatch")
    return value


def _strict_boolean(series: pd.Series, *, name: str) -> pd.Series:
    if not bool(series.map(lambda value: isinstance(value, (bool, np.bool_))).all()):
        raise TypeError(f"{name} must contain literal booleans")
    return series.astype(bool)


def validate_independent_selector_records(
    records: pd.DataFrame,
    *,
    selector_fit_seed: int,
    expected_primary_generators_per_seed: int,
) -> pd.DataFrame:
    """Validate the fixed-fit independent panel and return strict-unseen rows."""

    required = {
        "seed",
        "selector_fit_seed",
        "evaluation_seed",
        "outer_seed",
        "source_seed",
        "generator_id",
        "generator_split",
        "strict_unseen_composition",
        "primary_endpoint_eligible",
        "directional_sensitivity_eligible",
        "confirmatory_endpoint_eligible",
        "composition_overlap_secondary",
        "selector",
        "mode_selected",
        "oracle_mode",
        "fixed_best_mode",
        "utility",
        "status",
        "plasticity_l1",
        "plasticity_l2",
        "training_source_seed_count",
        "evaluation_seed_excluded_from_training",
        "independent_rows_used_for_fit",
        "selector_fit_count",
        "train_split",
        "train_endpoint",
        "test_endpoint",
        *PROBABILITY_COLUMNS,
        *CANDIDATE_UTILITY_COLUMNS,
        *TRAIN_UTILITY_COLUMNS,
    }
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Exp28 selector rows lack columns: {sorted(missing)}")
    values = records.copy()
    integer_columns = (
        "seed",
        "selector_fit_seed",
        "evaluation_seed",
        "outer_seed",
        "source_seed",
        "training_source_seed_count",
        "independent_rows_used_for_fit",
        "selector_fit_count",
    )
    for column in integer_columns:
        numeric = pd.to_numeric(values[column], errors="raise")
        if not np.all(np.equal(numeric, np.floor(numeric))):
            raise ValueError(f"{column} must contain exact integers")
        values[column] = numeric.astype(np.int64)
    if set(values["seed"]) != {selector_fit_seed} or set(
        values["selector_fit_seed"]
    ) != {selector_fit_seed}:
        raise ValueError("Exp28 rows are not bound to the registered fit seed")
    evaluation = values["evaluation_seed"]
    if set(int(seed) for seed in evaluation.unique()) != set(EXPECTED_EVALUATION_SEEDS):
        raise ValueError("Exp28 independent evaluation seed coverage mismatch")
    if not bool((values["outer_seed"] == evaluation).all()) or not bool(
        (values["source_seed"] == evaluation).all()
    ):
        raise ValueError("Exp28 outer/source seed differs from evaluation seed")
    if not bool(values["status"].eq("complete").all()):
        raise ValueError("Exp28 amended-sensitivity summary requires all rows complete")
    if not bool(values["generator_split"].eq("heldout").all()):
        raise ValueError("Exp28 may evaluate only heldout generator rows")
    if not bool(values["train_split"].eq("discovery").all()):
        raise ValueError("Exp28 selector fit must use discovery generators")
    if not bool(
        values["train_endpoint"].eq("validation_balanced_accuracy").all()
    ) or not bool(values["test_endpoint"].eq("test_balanced_accuracy").all()):
        raise ValueError("Exp28 train/test endpoint binding mismatch")
    if set(values["training_source_seed_count"]) != {len(EXPECTED_META_SEEDS)}:
        raise ValueError("Exp28 meta-training seed count must be 30")
    if set(values["independent_rows_used_for_fit"]) != {0}:
        raise ValueError("independent rows entered the selector fit")
    if set(values["selector_fit_count"]) != {1}:
        raise ValueError("Exp28 selectors were not fitted exactly once")
    excluded = _strict_boolean(
        values["evaluation_seed_excluded_from_training"],
        name="evaluation_seed_excluded_from_training",
    )
    if not bool(excluded.all()):
        raise ValueError("independent evaluation seed entered selector training")
    unseen = _strict_boolean(
        values["strict_unseen_composition"], name="strict_unseen_composition"
    )
    primary = _strict_boolean(
        values["primary_endpoint_eligible"], name="primary_endpoint_eligible"
    )
    directional = _strict_boolean(
        values["directional_sensitivity_eligible"],
        name="directional_sensitivity_eligible",
    )
    confirmatory = _strict_boolean(
        values["confirmatory_endpoint_eligible"],
        name="confirmatory_endpoint_eligible",
    )
    overlap = _strict_boolean(
        values["composition_overlap_secondary"],
        name="composition_overlap_secondary",
    )
    if (
        not np.array_equal(unseen.to_numpy(), primary.to_numpy())
        or not np.array_equal(unseen.to_numpy(), directional.to_numpy())
        or bool(confirmatory.any())
        or not np.array_equal(overlap.to_numpy(), (~unseen).to_numpy())
    ):
        raise ValueError("Exp28 composition eligibility flags are inconsistent")
    values["strict_unseen_composition"] = unseen

    group_keys = ["evaluation_seed", "generator_id"]
    counts = values.groupby(group_keys, sort=False)["selector"].agg(["size", "nunique"])
    if not bool(
        (
            (counts["size"] == len(SELECTORS)) & (counts["nunique"] == len(SELECTORS))
        ).all()
    ):
        raise ValueError("each Exp28 cell must contain exactly four selector rows")
    selector_sets = values.groupby(group_keys, sort=False)["selector"].agg(set)
    if not bool(selector_sets.map(lambda item: item == set(SELECTORS)).all()):
        raise ValueError("Exp28 selector panel differs from registration")
    heldout_counts = values.groupby("evaluation_seed")["generator_id"].nunique()
    if set(heldout_counts) != {44}:
        raise ValueError("Exp28 heldout generator count differs from registration")
    primary_mask = values["strict_unseen_composition"].to_numpy(dtype=bool)
    primary_counts = (
        values.loc[primary_mask].groupby("evaluation_seed")["generator_id"].nunique()
    )
    if set(primary_counts) != {expected_primary_generators_per_seed}:
        raise ValueError("Exp28 strict-unseen generator count mismatch")

    numeric_columns = (
        "utility",
        "plasticity_l1",
        "plasticity_l2",
        *PROBABILITY_COLUMNS,
        *CANDIDATE_UTILITY_COLUMNS,
        *TRAIN_UTILITY_COLUMNS,
    )
    for column in numeric_columns:
        values[column] = pd.to_numeric(values[column], errors="raise")
    numeric = values[list(numeric_columns)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Exp28 utilities, probabilities, or costs are non-finite")
    utility_columns = ("utility", *CANDIDATE_UTILITY_COLUMNS, *TRAIN_UTILITY_COLUMNS)
    utilities = values[list(utility_columns)].to_numpy(dtype=np.float64)
    if np.any((utilities < 0.0) | (utilities > 1.0)):
        raise ValueError("Exp28 balanced-accuracy utilities must lie in [0, 1]")
    if np.any(
        values[["plasticity_l1", "plasticity_l2"]].to_numpy(dtype=np.float64) < 0.0
    ):
        raise ValueError("Exp28 selector update costs must be non-negative")
    probabilities = values[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Exp28 selector probabilities are invalid")
    selected_indices = np.argmax(probabilities, axis=1)
    registered_modes = np.asarray(CANDIDATE_MODES, dtype=object)
    if not np.array_equal(
        values["mode_selected"].to_numpy(), registered_modes[selected_indices]
    ):
        raise ValueError("Exp28 probabilities do not select mode_selected")
    candidate = values[list(CANDIDATE_UTILITY_COLUMNS)].to_numpy(dtype=np.float64)
    selected_utility = candidate[np.arange(len(values)), selected_indices]
    if not np.allclose(
        values["utility"].to_numpy(dtype=np.float64),
        selected_utility,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Exp28 selected utility does not match candidate utility")
    paired = (
        "oracle_mode",
        "fixed_best_mode",
        *CANDIDATE_UTILITY_COLUMNS,
        *TRAIN_UTILITY_COLUMNS,
    )
    if bool(
        (
            values.groupby(group_keys, sort=False)[list(paired)].nunique(dropna=False)
            != 1
        ).any(axis=None)
    ):
        raise ValueError("Exp28 paired candidate data differ across selectors")
    oracle_rows = values[values["selector"] == "oracle"]
    expected_oracle = registered_modes[
        np.argmax(oracle_rows[list(CANDIDATE_UTILITY_COLUMNS)].to_numpy(), axis=1)
    ]
    if not np.array_equal(oracle_rows["mode_selected"].to_numpy(), expected_oracle):
        raise ValueError("Exp28 oracle violates registered tie breaking")
    fixed_rows = values[values["selector"] == "fixed_best"]
    if fixed_rows["fixed_best_mode"].nunique() != 1 or not bool(
        (fixed_rows["mode_selected"] == fixed_rows["fixed_best_mode"]).all()
    ):
        raise ValueError("Exp28 fixed-best policy is not globally frozen")
    train_means = fixed_rows.iloc[0][list(TRAIN_UTILITY_COLUMNS)].to_numpy(
        dtype=np.float64
    )
    if (
        fixed_rows["fixed_best_mode"].iloc[0]
        != registered_modes[int(np.argmax(train_means))]
    ):
        raise ValueError("Exp28 fixed-best mode is not the meta-train argmax")
    primary_rows = values.loc[primary_mask].copy()
    return primary_rows.sort_values(
        ["evaluation_seed", "generator_id", "selector"], kind="mergesort"
    ).reset_index(drop=True)


def _independent_seed_endpoints(
    primary: pd.DataFrame,
    *,
    noninferiority_fraction: float,
) -> tuple[SeedSelectorEndpoint, ...]:
    endpoints: list[SeedSelectorEndpoint] = []
    for seed, seed_frame in primary.groupby("evaluation_seed", sort=True):
        policy = seed_frame.pivot(
            index="generator_id", columns="selector", values="utility"
        )
        cell = seed_frame.groupby("generator_id", sort=True).first()
        fixed_gain = policy["oracle"] - policy["fixed_best"]
        local_gain = policy["local_three_factor"] - policy["fixed_best"]
        gru_gain = policy["gru_bptt"] - policy["fixed_best"]
        applicable = fixed_gain > 1e-12
        oracle_gain_mean = float(np.mean(fixed_gain))
        local_gain_mean = float(np.mean(local_gain))
        gru_gain_mean = float(np.mean(gru_gain))
        local_rows = seed_frame[seed_frame["selector"] == "local_three_factor"]
        gru_rows = seed_frame[seed_frame["selector"] == "gru_bptt"]
        endpoints.append(
            SeedSelectorEndpoint(
                seed=int(seed),
                strict_unseen_generators=int(policy.shape[0]),
                fixed_best_mode=str(cell["fixed_best_mode"].iloc[0]),
                routing_utility=float(np.mean(cell["candidate_routing_utility"])),
                gain_utility=float(np.mean(cell["candidate_gain_utility"])),
                low_rank_utility=float(np.mean(cell["candidate_low_rank_utility"])),
                fixed_best_utility=float(np.mean(policy["fixed_best"])),
                oracle_utility=float(np.mean(policy["oracle"])),
                gru_bptt_utility=float(np.mean(policy["gru_bptt"])),
                local_three_factor_utility=float(np.mean(policy["local_three_factor"])),
                local_minus_fixed_best=local_gain_mean,
                oracle_minus_fixed_best=oracle_gain_mean,
                gru_minus_fixed_best=gru_gain_mean,
                local_noninferiority_contrast=float(
                    local_gain_mean - noninferiority_fraction * oracle_gain_mean
                ),
                local_regret=float(
                    np.mean(policy["oracle"] - policy["local_three_factor"])
                ),
                gru_regret=float(np.mean(policy["oracle"] - policy["gru_bptt"])),
                local_selection_accuracy=float(
                    np.mean(
                        local_rows["mode_selected"].to_numpy()
                        == local_rows["oracle_mode"].to_numpy()
                    )
                ),
                gru_selection_accuracy=float(
                    np.mean(
                        gru_rows["mode_selected"].to_numpy()
                        == gru_rows["oracle_mode"].to_numpy()
                    )
                ),
                local_recovered_oracle_gain=float(
                    local_gain_mean / oracle_gain_mean
                    if oracle_gain_mean > 1e-12
                    else float("nan")
                ),
                gru_recovered_oracle_gain=float(
                    gru_gain_mean / oracle_gain_mean
                    if oracle_gain_mean > 1e-12
                    else float("nan")
                ),
                positive_oracle_gain_cells=int(np.sum(applicable)),
                local_cell_recovery_mean=float(
                    np.mean(
                        (local_gain[applicable] / fixed_gain[applicable]).to_numpy()
                    )
                    if bool(applicable.any())
                    else float("nan")
                ),
                gru_cell_recovery_mean=float(
                    np.mean((gru_gain[applicable] / fixed_gain[applicable]).to_numpy())
                    if bool(applicable.any())
                    else float("nan")
                ),
                local_update_l1=float(np.mean(local_rows["plasticity_l1"])),
                local_update_l2=float(np.mean(local_rows["plasticity_l2"])),
                gru_update_l1=float(np.mean(gru_rows["plasticity_l1"])),
                gru_update_l2=float(np.mean(gru_rows["plasticity_l2"])),
            )
        )
    return tuple(endpoints)


def summarize_independent_selector(
    records: pd.DataFrame,
    *,
    selector_fit_seed: int,
    expected_primary_generators_per_seed: int,
    noninferiority_fraction: float,
    bootstrap_samples: int,
    permutation_samples: int,
    confidence: float,
    random_seed: int,
) -> ActuatorSelectorConclusion:
    primary_frame = validate_independent_selector_records(
        records,
        selector_fit_seed=selector_fit_seed,
        expected_primary_generators_per_seed=expected_primary_generators_per_seed,
    )
    endpoints = _independent_seed_endpoints(
        primary_frame, noninferiority_fraction=noninferiority_fraction
    )
    endpoint_map = {
        "local_noninferiority_contrast": _finite_vector(
            [item.local_noninferiority_contrast for item in endpoints],
            name="local_noninferiority_contrast",
        ),
        "local_minus_fixed_best": _finite_vector(
            [item.local_minus_fixed_best for item in endpoints],
            name="local_minus_fixed_best",
        ),
    }
    positive_p = np.asarray(
        [
            _one_sided_sign_flip(
                values,
                seed=random_seed + 100 + index,
                samples=permutation_samples,
            )
            for index, values in enumerate(endpoint_map.values())
        ]
    )
    negative_p = np.asarray(
        [
            _one_sided_sign_flip(
                -values,
                seed=random_seed + 200 + index,
                samples=permutation_samples,
            )
            for index, values in enumerate(endpoint_map.values())
        ]
    )
    positive_holm = holm_adjust(positive_p)
    negative_holm = holm_adjust(negative_p)
    primary = tuple(
        _contrast_summary(
            name,
            values,
            bootstrap_seed=random_seed + index,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            p_value=float(positive_p[index]),
            p_value_holm=float(positive_holm[index]),
            opposition_p_value=float(negative_p[index]),
            opposition_p_value_holm=float(negative_holm[index]),
            confirmatory=False,
        )
        for index, (name, values) in enumerate(endpoint_map.items())
    )
    descriptive_values = {
        "local_minus_routing": np.asarray(
            [
                item.local_three_factor_utility - item.routing_utility
                for item in endpoints
            ]
        ),
        "local_minus_gain": np.asarray(
            [item.local_three_factor_utility - item.gain_utility for item in endpoints]
        ),
        "local_minus_low_rank": np.asarray(
            [
                item.local_three_factor_utility - item.low_rank_utility
                for item in endpoints
            ]
        ),
        "local_minus_gru_bptt": np.asarray(
            [
                item.local_three_factor_utility - item.gru_bptt_utility
                for item in endpoints
            ]
        ),
    }
    descriptive = tuple(
        _contrast_summary(
            name,
            _finite_vector(values, name=name),
            bootstrap_seed=random_seed + 20 + index,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            p_value=None,
            p_value_holm=None,
            opposition_p_value=None,
            opposition_p_value_holm=None,
            confirmatory=False,
        )
        for index, (name, values) in enumerate(descriptive_values.items())
    )
    complete = len(endpoints) == len(EXPECTED_EVALUATION_SEEDS)
    return ActuatorSelectorConclusion(
        conclusion="inconclusive",
        reason=(
            "post-hoc ceiling-amended source cannot restore confirmatory "
            "independence; overall classification is forced inconclusive"
        ),
        statistics_unit="post_hoc_amended_evaluation_seed",
        n_seeds=len(endpoints),
        strict_unseen_only=True,
        complete_primary_coverage=complete,
        noninferiority_fraction=noninferiority_fraction,
        primary_contrasts=primary,
        descriptive_contrasts=descriptive,
        seed_endpoints=endpoints,
    )


def directional_sensitivity_classification(
    conclusion: ActuatorSelectorConclusion,
) -> dict[str, object]:
    """Classify direction descriptively without changing overall conclusion."""

    primary = conclusion.primary_contrasts
    complete = conclusion.complete_primary_coverage
    supported = complete and all(
        item.lower_confidence > 0.0
        and item.p_value_holm is not None
        and item.p_value_holm < 0.05
        for item in primary
    )
    opposed = complete and any(
        item.upper_confidence <= 0.0
        and item.opposition_p_value_holm is not None
        and item.opposition_p_value_holm < 0.05
        for item in primary
    )
    if supported:
        classification = "support"
        reason = "both directional seed-level thresholds passed"
    elif opposed:
        classification = "oppose"
        reason = "at least one directional seed-level claim was contradicted"
    else:
        classification = "inconclusive"
        reason = "the two directional seed-level thresholds were not both met"
    return {
        "classification": classification,
        "reason": reason,
        "non_confirmatory": True,
        "inference_status": INFERENCE_STATUS,
    }


def collect_exp28(
    results_root: str | Path,
    *,
    config: Mapping[str, Any],
    run_label: str,
) -> Exp28Collection:
    _validate_config(config)
    if run_label != REQUIRED_RUN_LABEL:
        raise ValueError("Exp28 collector requires the registered run label")
    fit_seed = int(config["selector_fit_seed"])
    attempt = _matching_attempt(
        Path(results_root), selector_fit_seed=fit_seed, run_label=run_label
    )
    status = _read_json(attempt / "status.json")
    if not isinstance(status, Mapping) or status.get("status") != "complete":
        raise ValueError("Exp28 amended-sensitivity attempt is not fully complete")
    stored_config = _read_json(attempt / "config.json")
    if not isinstance(stored_config, Mapping):
        raise ValueError("Exp28 stored config is malformed")
    for key, expected in config.items():
        if stored_config.get(key) != expected:
            raise ValueError(f"Exp28 stored config differs at {key}")
    if (
        stored_config.get("seed") != fit_seed
        or stored_config.get("run_label") != run_label
    ):
        raise ValueError("Exp28 stored fit seed or run label mismatch")
    config_sha = canonical_config_sha256(config)
    provenance = stored_config.get("evidence_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("schema_version")
        != "exp28_amended_sensitivity_selector_evidence_v1"
        or provenance.get("canonical_config_sha256") != config_sha
        or provenance.get("inference_scope") != INFERENCE_SCOPE
        or provenance.get("inference_status") != INFERENCE_STATUS
        or provenance.get("confirmatory_eligible") is not CONFIRMATORY_ELIGIBLE
    ):
        raise ValueError("Exp28 stored evidence provenance mismatch")
    run_git = provenance.get("git")
    if not isinstance(run_git, Mapping) or run_git.get("dirty") is not False:
        raise ValueError("Exp28 evidence Git identity is dirty or missing")
    commit = run_git.get("commit")
    tree = run_git.get("tree")
    if not isinstance(commit, str) or not isinstance(tree, str):
        raise ValueError("Exp28 evidence Git commit/tree is missing")
    environment = _read_json(attempt / "environment.json")
    if not isinstance(environment, Mapping) or environment.get("git") != run_git:
        raise ValueError("Exp28 environment Git identity mismatch")

    source_path = attempt / "source_package_receipt.json"
    fit_path = attempt / "selector_fit_receipt.json"
    decision_path = attempt / "decision_receipt.json"
    source_receipt = _validate_self_hash(
        _read_json(source_path), name=source_path.name, schema=SOURCE_RECEIPT_SCHEMA
    )
    fit_receipt = _validate_self_hash(
        _read_json(fit_path), name=fit_path.name, schema=FIT_RECEIPT_SCHEMA
    )
    decision_receipt = _validate_self_hash(
        _read_json(decision_path),
        name=decision_path.name,
        schema=DECISION_RECEIPT_SCHEMA,
    )
    source_file_sha = _file_sha256(source_path)
    fit_file_sha = _file_sha256(fit_path)
    decision_file_sha = _file_sha256(decision_path)
    if decision_receipt.get("selector_fit_receipt_sha256") != fit_receipt.get(
        "receipt_sha256"
    ):
        raise ValueError("Exp28 decision receipt is not bound to the fit receipt")
    if (
        fit_receipt.get("selector_fit_seed") != fit_seed
        or fit_receipt.get("fit_count") != 1
    ):
        raise ValueError("Exp28 fit receipt does not prove one registered fit")
    if fit_receipt.get("independent_rows_used_for_fit") != 0:
        raise ValueError("Exp28 fit receipt admits independent data leakage")
    for selector in ("local_three_factor", "gru_bptt"):
        model_receipt = fit_receipt.get(selector)
        if not isinstance(model_receipt, Mapping):
            raise ValueError(f"Exp28 {selector} fit receipt is missing")
        model_payload = dict(model_receipt)
        model_digest = model_payload.pop("receipt_sha256", None)
        if model_digest != _payload_sha256(model_payload):
            raise ValueError(f"Exp28 {selector} fit receipt self-hash mismatch")
    decision_selectors = decision_receipt.get("selectors")
    if not isinstance(decision_selectors, Mapping) or set(decision_selectors) != set(
        SELECTORS
    ):
        raise ValueError("Exp28 decision receipt selector registry mismatch")
    for selector in SELECTORS:
        selector_receipt = decision_selectors[selector]
        if not isinstance(selector_receipt, Mapping):
            raise ValueError(f"Exp28 {selector} decision receipt is missing")
        selector_payload = dict(selector_receipt)
        selector_digest = selector_payload.pop("receipt_sha256", None)
        if selector_digest != _payload_sha256(selector_payload):
            raise ValueError(f"Exp28 {selector} decision receipt self-hash mismatch")

    plans = _read_json(attempt / "planned_conditions.json")
    metrics = _read_jsonl(attempt / "metrics.jsonl")
    if not isinstance(plans, list):
        raise ValueError("Exp28 planned conditions must be a JSON list")
    expected_plan = _planned_conditions(config)
    normalized_plan = [
        {key: row.get(key) for key in PLAN_KEYS}
        for row in plans
        if isinstance(row, Mapping)
    ]
    if normalized_plan != expected_plan:
        raise ValueError("Exp28 planned conditions differ from registration")
    if [row.get("condition_index") for row in plans] != list(range(len(plans))):
        raise ValueError("Exp28 condition indexes are not contiguous")
    if len(metrics) != len(expected_plan):
        raise ValueError("Exp28 observed row count differs from plan")
    normalized_metrics = [{key: row.get(key) for key in PLAN_KEYS} for row in metrics]
    if normalized_metrics != expected_plan:
        raise ValueError("Exp28 metric ordering/coverage differs from plan")

    meta_training, folds, replay_source = _load_sources(config)
    replay_fit = _fit_frozen_selectors(meta_training, folds, config)
    if _payload_sha256(source_receipt) != _payload_sha256(replay_source):
        raise ValueError("Exp28 source/package receipt replay mismatch")
    if _payload_sha256(fit_receipt) != _payload_sha256(replay_fit.fit_receipt):
        raise ValueError("Exp28 frozen selector fit replay mismatch")
    if _payload_sha256(decision_receipt) != _payload_sha256(
        replay_fit.decision_receipt
    ):
        raise ValueError("Exp28 all-test decision replay mismatch")
    replay_records = _semantic_records(
        meta_training,
        folds,
        replay_fit,
        config,
        run_label=run_label,
        config_sha256=config_sha,
        run_git=run_git,
        source_receipt=replay_source,
        source_receipt_file_sha256=source_file_sha,
        fit_receipt_file_sha256=fit_file_sha,
        decision_receipt_file_sha256=decision_file_sha,
    )
    expected_semantic = [
        {**dimensions, **row_metrics} for row_metrics, dimensions in replay_records
    ]
    observed_semantic = [
        {
            key: value
            for key, value in row.items()
            if key not in {"run_id", "experiment", "seed", "recorded_at"}
        }
        for row in metrics
    ]
    if [_payload_sha256(row) for row in observed_semantic] != [
        _payload_sha256(row) for row in expected_semantic
    ]:
        raise ValueError("Exp28 deterministic semantic-row replay mismatch")

    frame = pd.DataFrame(metrics)
    row_bindings = {
        "protocol": PROTOCOL_VERSION,
        "run_label": run_label,
        "selector_fit_seed": fit_seed,
        "exp28_config_sha256": config_sha,
        "run_git_commit": commit,
        "run_git_tree": tree,
        "run_git_dirty": False,
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_receipt_file_sha256": source_file_sha,
        "selector_fit_receipt_sha256": fit_receipt["receipt_sha256"],
        "selector_fit_receipt_file_sha256": fit_file_sha,
        "decision_receipt_sha256": decision_receipt["receipt_sha256"],
        "decision_receipt_file_sha256": decision_file_sha,
        "inference_scope": INFERENCE_SCOPE,
        "inference_status": INFERENCE_STATUS,
        "confirmatory_eligible": CONFIRMATORY_ELIGIBLE,
        "force_inconclusive": True,
        "profile": "post_hoc_sensitivity",
    }
    for column, expected in row_bindings.items():
        if column not in frame or not bool(
            frame[column].map(lambda value: value == expected).all()
        ):
            raise ValueError(f"Exp28 row provenance mismatch: {column}")
    for selector in SELECTORS:
        selector_receipt = decision_receipt["selectors"][selector]
        selector_rows = frame[frame["selector"] == selector]
        row_seeds = selector_rows["evaluation_seed"].to_numpy(dtype=np.int64)
        row_ids = selector_rows["generator_id"].astype(str).tolist()
        row_probabilities = selector_rows[list(PROBABILITY_COLUMNS)].to_numpy(
            dtype=np.float64
        )
        if (
            selector_receipt["evaluation_seeds"] != row_seeds.tolist()
            or selector_receipt["generator_ids"] != row_ids
            or not np.array_equal(
                np.asarray(selector_receipt["probabilities"], dtype=np.float64),
                row_probabilities,
            )
        ):
            raise ValueError(f"Exp28 {selector} decisions are not receipt-bound")
        if selector_receipt["decision_fingerprint"] != _decision_fingerprint(
            row_seeds, row_ids, row_probabilities
        ):
            raise ValueError(f"Exp28 {selector} decision fingerprint mismatch")
    validate_independent_selector_records(
        frame,
        selector_fit_seed=fit_seed,
        expected_primary_generators_per_seed=int(
            config["evaluation"]["expected_strict_unseen_per_seed"]
        ),
    )
    current_git = git_identity()
    if current_git.get("dirty") is not False or (
        current_git.get("commit"),
        current_git.get("tree"),
    ) != (commit, tree):
        raise ValueError(
            "Exp28 amended-sensitivity summary must run clean on evidence commit/tree"
        )
    return Exp28Collection(
        raw=frame,
        config=dict(config),
        config_sha256=config_sha,
        attempt_path=str(attempt.resolve()),
        run_git_commit=commit,
        run_git_tree=tree,
        runtime_identity=dict(environment),
        source_receipt=dict(source_receipt),
        source_receipt_file_sha256=source_file_sha,
        fit_receipt=dict(fit_receipt),
        fit_receipt_file_sha256=fit_file_sha,
        decision_receipt=dict(decision_receipt),
        decision_receipt_file_sha256=decision_file_sha,
        run_label=run_label,
    )


def _summary_table(conclusion: ActuatorSelectorConclusion) -> pd.DataFrame:
    endpoints = pd.DataFrame(asdict(item) for item in conclusion.seed_endpoints)
    columns = (
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
    )
    return pd.DataFrame(
        {
            "metric": column,
            "statistics_unit": "post_hoc_amended_evaluation_seed",
            "n": len(endpoints),
            "mean": float(np.mean(endpoints[column])),
            "std": float(np.std(endpoints[column], ddof=1)),
            "min": float(np.min(endpoints[column])),
            "max": float(np.max(endpoints[column])),
        }
        for column in columns
    )


def _report(
    collection: Exp28Collection,
    conclusion: ActuatorSelectorConclusion,
    *,
    raw_sha256: str,
) -> str:
    directional = directional_sensitivity_classification(conclusion)
    contrast_rows = "\n".join(
        "| {name} | {mean:.6f} | [{lower:.6f}, {upper:.6f}] | {holm:.6g} | {negative:.6g} |".format(
            name=item.name,
            mean=item.mean,
            lower=item.lower_confidence,
            upper=item.upper_confidence,
            holm=item.p_value_holm,
            negative=item.opposition_p_value_holm,
        )
        for item in conclusion.primary_contrasts
    )
    return f"""# Exp28 post-hoc amended actuator-selector sensitivity

## Overall conclusion

**{conclusion.conclusion.upper()}** — {conclusion.reason}.

One selector set was fitted once on Exp26 seeds `0--29`; the 30 disjoint
seeds `30--59` are the sensitivity-analysis statistical units. The source
ceiling was amended from `128` to `256` after the original panel was inspected,
so confirmatory independence cannot be restored.

Directional sensitivity: **{str(directional["classification"]).upper()}** —
{directional["reason"]}. This label is descriptive and cannot change the
overall `INCONCLUSIVE` classification.

| Directional endpoint | Seed mean | 95% bootstrap CI | Positive Holm p | Negative Holm p |
|---|---:|---:|---:|---:|
{contrast_rows}

Directional support requires both positive lower bounds and both positive
Holm-adjusted `p < 0.05`. The inference status is `{INFERENCE_STATUS}`.

## Audit

- Amended evaluation seeds: `{conclusion.n_seeds}`.
- Selector fit count: `1`; registered fit seed: `{collection.config["selector_fit_seed"]}`.
- Git commit/tree: `{collection.run_git_commit}` / `{collection.run_git_tree}`.
- Config SHA-256: `{collection.config_sha256}`.
- Source/package receipt SHA-256: `{collection.source_receipt["receipt_sha256"]}`.
- Protocol-amendment SHA-256: `{collection.source_receipt["independent_package"]["protocol_amendment_sha256"]}`.
- Frozen-fit receipt SHA-256: `{collection.fit_receipt["receipt_sha256"]}`.
- All-test decision receipt SHA-256: `{collection.decision_receipt["receipt_sha256"]}`.
- Collected raw SHA-256: `{raw_sha256}`.
- Collector result: exact source, fit, decision, and semantic-row replay passed.

## Interpretation boundary

This analysis tests supervised task-demand-to-family selection over frozen,
task-matched actuator policies. It does not establish hidden belief inference,
de-novo motif formation, scalar-reward-only learning, or update-budget
equivalence between the local and GRU selectors. Inference is conditional on
one fixed meta-training panel and optimizer root seed `2801`; it does not
marginalize meta-panel or optimizer-seed uncertainty.

The original ceiling-128 frozen source protocol is classified **OPPOSE**
because one of 13,200 registered cells failed. Consequently, the corresponding
selector formal analysis is **INCONCLUSIVE**. A future confirmatory test is
reserved for wholly fresh seeds `60--89` under the separate namespace
`exp28_fresh_independent_confirmatory_v2_seeds60_89`; those seeds are not part
of this amended sensitivity analysis.
"""


def write_exp28_summary(
    collection: Exp28Collection,
    output_dir: str | Path,
    *,
    make_figure: bool = True,
) -> ActuatorSelectorConclusion:
    config = collection.config
    analysis = config["analysis"]
    conclusion = summarize_independent_selector(
        collection.raw,
        selector_fit_seed=int(config["selector_fit_seed"]),
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
    )
    if conclusion.conclusion != "inconclusive":
        raise RuntimeError("amended Exp28 overall conclusion must be inconclusive")
    directional = directional_sensitivity_classification(conclusion)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    endpoints = pd.DataFrame(asdict(item) for item in conclusion.seed_endpoints)
    raw_path = output / "raw_metrics.csv.gz"
    collection.raw.to_csv(
        raw_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    raw_sha = _file_sha256(raw_path)
    endpoints.to_csv(output / "seed_endpoints.csv", index=False)
    _summary_table(conclusion).to_csv(output / "summary.csv", index=False)
    payload = {
        **conclusion.to_dict(),
        "profile": "post_hoc_sensitivity",
        "confirmatory_eligible": CONFIRMATORY_ELIGIBLE,
        "overall_classification_forced": True,
        "force_inconclusive": True,
        "directional_sensitivity": directional,
        "inference_status": INFERENCE_STATUS,
        "protocol_version": PROTOCOL_VERSION,
        "run_label": collection.run_label,
        "canonical_config_sha256": collection.config_sha256,
        "run_git_commit": collection.run_git_commit,
        "run_git_tree": collection.run_git_tree,
        "source_receipt_sha256": collection.source_receipt["receipt_sha256"],
        "selector_fit_receipt_sha256": collection.fit_receipt["receipt_sha256"],
        "decision_receipt_sha256": collection.decision_receipt["receipt_sha256"],
        "raw_metrics_sha256": raw_sha,
        "inference_scope": INFERENCE_SCOPE,
        "historical_frozen_source_classification": "oppose",
        "historical_selector_formal_classification": "inconclusive",
        "future_confirmatory_reservation": config["future_confirmatory_reservation"],
        "attempt_path": collection.attempt_path,
    }
    _write_json(output / "conclusion.json", payload)
    _write_json(
        output / "provenance.json",
        {
            **payload,
            "source_receipt": collection.source_receipt,
            "fit_receipt_file_sha256": collection.fit_receipt_file_sha256,
            "decision_receipt_file_sha256": collection.decision_receipt_file_sha256,
            "runtime_identity": collection.runtime_identity,
        },
    )
    (output / "report.md").write_text(
        _report(collection, conclusion, raw_sha256=raw_sha), encoding="utf-8"
    )
    if make_figure:
        plot_selector_evidence(
            endpoints,
            output / "exp28_amended_sensitivity_evidence",
            title=(
                "Exp28 post-hoc amended actuator-selector sensitivity "
                f"(NON-CONFIRMATORY; n={endpoints.shape[0]} seeds)"
            ),
            contrast_title="B  Directional non-inferiority sensitivity",
        )
    return conclusion


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-label", default=REQUIRED_RUN_LABEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args(argv)
    config_value = _read_json(args.config)
    if not isinstance(config_value, Mapping):
        raise ValueError("Exp28 config must be a JSON object")
    collection = collect_exp28(
        args.results_root, config=config_value, run_label=args.run_label
    )
    conclusion = write_exp28_summary(
        collection, args.output_dir, make_figure=not args.skip_figure
    )
    print(
        json.dumps(
            {
                "overall": conclusion.to_dict(),
                "directional_sensitivity": directional_sensitivity_classification(
                    conclusion
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
