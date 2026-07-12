"""Build and validate the fail-closed formal exp14 neural-data snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import load_json_config  # noqa: E402
from experiments.exp14_ibl_multisession_neural import (  # noqa: E402
    FAMILIES,
    panel_claim_scope,
)
from src.analysis.ibl_neural_metrics import compare_count_families  # noqa: E402
from src.data.ibl_neural_panel import (  # noqa: E402
    MACRO_REGION_MAPPING_SCHEMA,
    MACRO_REGION_SOURCE_ONTOLOGY_SHA256,
    MACRO_REGION_SOURCE_PROVENANCE_SHA256,
    REGION_ORDER,
    AllenMacroRegionMapping,
    load_allen_macro_region_mapping,
)
from src.models.hierarchical_count_dynamics import (  # noqa: E402
    HierarchicalCountScore,
    SessionCountMetrics,
)
from src.utils.reproducibility import derive_seed  # noqa: E402

DEFAULT_PREFIX = "exp14_ibl_multisession_neural_formal"
REGISTERED_CONFIG_PATH = "configs/formal/exp14_ibl_multisession_neural.json"
REGISTERED_HMM_RESTART_SELECTION_POLICY = (
    "eligible_converged_identifiable_then_likelihood"
)
_HASH_FIELDS = (
    "expected_source_manifest_sha256",
    "expected_acquisition_bundle_sha256",
    "expected_compact_manifest_sha256",
    "expected_compact_bundle_sha256",
    "expected_macro_region_mapping_sha256",
    "expected_macro_region_source_ontology_sha256",
    "expected_macro_region_source_provenance_sha256",
    "macro_region_mapping_formal_compact_manifest_sha256",
)
_PAIRED_HMM_RECEIPT_FIELDS = (
    "belief_checkpoint_sha256",
    "belief_trajectory_sha256",
    "hmm_fit_converged",
    "hmm_state_identifiable",
    "hmm_restart_selection_policy",
    "hmm_selected_restart",
    "hmm_eligible_restart_count",
    "hmm_eligible_restart_fallback",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _portable_formal_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove host checkout paths while preserving every scientific field."""

    payload = dict(config)
    payload["config_path"] = REGISTERED_CONFIG_PATH
    return payload


def _portable_formal_config_sha256(config: Mapping[str, Any]) -> str:
    return _canonical_sha256(_portable_formal_config(config))


def _registered_formal_json_sha256() -> str:
    return _sha256(PROJECT_ROOT / REGISTERED_CONFIG_PATH)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(item, dict) for item in records):
        raise ValueError(f"{path} contains a non-object record")
    return records


def _require_digest(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _expected_attempt_config(
    formal_config: Mapping[str, Any], *, seed: int = 0
) -> dict[str, Any]:
    return {
        "experiment": "exp14_ibl_multisession_neural",
        "seed": int(seed),
        **dict(formal_config),
        "training_algorithm": "train_only_nested_hidden_belief_count_dynamics",
        "used_autograd": False,
        "parent_checkpoint": None,
        "spiking_mechanism_required": False,
        "hrm_ctm_scope": "continuous_inspiration_not_reproduction",
    }


def _latest_matching_attempt(
    results_root: Path, *, expected_config: Mapping[str, Any]
) -> Path:
    seed_root = results_root / "runs" / "exp14_ibl_multisession_neural" / "seed_0000"
    for attempt in sorted(seed_root.glob("*"), reverse=True):
        path = attempt / "config.json"
        if path.is_file() and _read_json(path) == dict(expected_config):
            return attempt
    raise FileNotFoundError(
        "no exp14 seed=0 attempt exactly matches the complete formal config"
    )


def _validate_formal_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("profile") != "formal"
        or config.get("data_mode") != "frozen_compact_cache"
        or [int(seed) for seed in config.get("seeds", [])] != [0]
        or int(config.get("planned_sessions", 0)) != 20
        or int(config.get("planned_animals", 0)) != 5
    ):
        raise ValueError(
            "exp14 snapshot requires the registered formal seed=0 cohort contract"
        )
    for field in _HASH_FIELDS:
        _require_digest(config.get(field), field)
    commit = str(config.get("expected_bwm_repository_commit", ""))
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError(
            "expected_bwm_repository_commit must be a lowercase Git commit"
        )
    if int(config.get("n_bootstrap", 0)) < 100:
        raise ValueError("formal exp14 requires at least 100 bootstrap draws")
    if int(config.get("minimum_region_sessions", 0)) != 5:
        raise ValueError("formal exp14 minimum_region_sessions must equal 5")
    _formal_hmm_contract(config)
    _formal_macro_mapping(config)


def _formal_hmm_contract(config: Mapping[str, Any]) -> tuple[str, int]:
    options = config.get("learned_hmm")
    if not isinstance(options, Mapping):
        raise ValueError("formal exp14 requires a learned_hmm mapping")
    policy = options.get("restart_selection_policy")
    n_restarts = options.get("n_restarts")
    if (
        policy != REGISTERED_HMM_RESTART_SELECTION_POLICY
        or isinstance(n_restarts, bool)
        or not isinstance(n_restarts, int)
        or n_restarts < 1
        or options.get("require_converged") is not True
        or options.get("require_identifiable") is not True
    ):
        raise ValueError(
            "formal exp14 requires eligible-first HMM restart selection and "
            "converged identifiable fits"
        )
    return str(policy), int(n_restarts)


def _validate_hmm_restart_receipt(
    record: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    policy, n_restarts = _formal_hmm_contract(config)

    def exact_bool(value: object, expected: bool) -> bool:
        return isinstance(value, (bool, np.bool_)) and bool(value) is expected

    def receipt_int(value: object, name: str) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer receipt")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be an integer receipt") from error
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be an integer receipt")
        return int(numeric)

    selected = receipt_int(record.get("hmm_selected_restart"), "hmm_selected_restart")
    eligible_count = receipt_int(
        record.get("hmm_eligible_restart_count"), "hmm_eligible_restart_count"
    )
    if (
        record.get("hmm_restart_selection_policy") != policy
        or not exact_bool(record.get("hmm_fit_converged"), True)
        or not exact_bool(record.get("hmm_state_identifiable"), True)
        or not exact_bool(record.get("hmm_eligible_restart_fallback"), False)
        or not 0 <= selected < n_restarts
        or not 1 <= eligible_count <= n_restarts
    ):
        raise ValueError(
            "outer-test HMM receipt violates the registered eligible-restart contract"
        )


def _validate_paired_hmm_receipts(outer_frame: pd.DataFrame) -> None:
    missing = set(_PAIRED_HMM_RECEIPT_FIELDS) - set(outer_frame.columns)
    if missing:
        raise ValueError(f"outer-test rows lack paired HMM receipts: {sorted(missing)}")
    for field in ("belief_checkpoint_sha256", "belief_trajectory_sha256"):
        for value in outer_frame[field].tolist():
            _require_digest(value, field)
    group_keys = ["session_id", "view", "panel"]
    grouped = outer_frame.groupby(group_keys, dropna=False)
    if grouped.size().ne(len(FAMILIES)).any():
        raise ValueError("outer-test HMM receipts lack one paired model family")
    for field in _PAIRED_HMM_RECEIPT_FIELDS:
        if grouped[field].nunique(dropna=False).ne(1).any():
            raise ValueError("paired model families disagree on their HMM fit receipt")


def _formal_macro_mapping(config: Mapping[str, Any]) -> AllenMacroRegionMapping:
    if (
        config.get("macro_region_mapping_path")
        != "configs/exp14_allen_macro_region_mapping_v1.json"
        or config.get("expected_macro_region_mapping_schema")
        != MACRO_REGION_MAPPING_SCHEMA
        or config.get("expected_macro_region_source_ontology_sha256")
        != MACRO_REGION_SOURCE_ONTOLOGY_SHA256
        or config.get("expected_macro_region_source_provenance_sha256")
        != MACRO_REGION_SOURCE_PROVENANCE_SHA256
        or config.get("macro_region_mapping_formal_compact_manifest_sha256")
        != config.get("expected_compact_manifest_sha256")
    ):
        raise ValueError("formal exp14 macro-region mapping registration is invalid")
    return load_allen_macro_region_mapping(
        PROJECT_ROOT / str(config["macro_region_mapping_path"]),
        expected_sha256=str(config["expected_macro_region_mapping_sha256"]),
        expected_compact_manifest_sha256=str(
            config["macro_region_mapping_formal_compact_manifest_sha256"]
        ),
    )


def _registered_formal_config() -> dict[str, Any]:
    return load_json_config(REGISTERED_CONFIG_PATH)


def _condition_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(
        str(row[key]) for key in ("session_id", "view", "panel", "model_family")
    )  # type: ignore[return-value]


def _common_regions(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                "common_regions must be canonical JSON or a sequence"
            ) from error
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("common_regions must be a non-empty sequence")
    result = tuple(str(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError("common_regions must be unique")
    return result


def _json_container(value: object, *, name: str) -> object:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} must be canonical JSON or a container") from error
    if not isinstance(value, (list, tuple, dict)):
        raise ValueError(f"{name} must be a container")
    return value


def _validate_region_receipts(
    outer_frame: pd.DataFrame,
    comparison_records: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> None:
    mapping_receipt = dict(_formal_macro_mapping(config).receipt())
    comparison_by_pair = {
        (str(row["view"]), str(row["panel"])): row for row in comparison_records
    }
    expected_sessions = int(config["planned_sessions"])
    for pair, group in outer_frame.groupby(["view", "panel"]):
        key = (str(pair[0]), str(pair[1]))
        comparison = comparison_by_pair[key]
        session_ids = tuple(sorted(group["session_id"].astype(str).unique()))
        if len(session_ids) != expected_sessions:
            raise ValueError("region receipt does not cover all planned sessions")
        scalar_expected = {
            "region_anchor_policy": "fixed_region_order_union",
            "region_imputation_strategy": "pooled_training_fold_region_mean",
            "minimum_region_sessions": int(config["minimum_region_sessions"]),
            "n_complete_sessions_input": expected_sessions,
            "n_sessions_retained": expected_sessions,
            "all_complete_sessions_retained": True,
        }
        for field, expected in scalar_expected.items():
            if comparison.get(field) != expected or any(
                value != expected for value in group[field].tolist()
            ):
                raise ValueError(f"region receipt field {field} violates registration")
        for field, expected in mapping_receipt.items():
            comparison_value = comparison.get(field)
            outer_values = group[field].tolist()
            if isinstance(expected, dict):
                expected_json = json.dumps(expected, sort_keys=True)
                comparison_valid = (
                    json.dumps(
                        _json_container(comparison_value, name=field), sort_keys=True
                    )
                    == expected_json
                )
                outer_valid = all(
                    json.dumps(_json_container(value, name=field), sort_keys=True)
                    == expected_json
                    for value in outer_values
                )
            else:
                comparison_valid = comparison_value == expected
                outer_valid = all(value == expected for value in outer_values)
            if not comparison_valid or not outer_valid:
                raise ValueError(
                    f"macro-region mapping receipt field {field} violates registration"
                )
        complete_ids = tuple(
            str(item)
            for item in _json_container(
                comparison.get("complete_session_ids"), name="complete_session_ids"
            )
        )
        retained_ids = tuple(
            str(item)
            for item in _json_container(
                comparison.get("retained_session_ids"), name="retained_session_ids"
            )
        )
        if complete_ids != retained_ids or set(complete_ids) != set(session_ids):
            raise ValueError("complete/retained session IDs disagree with outer rows")
        for field, expected_ids in (
            ("complete_session_ids", complete_ids),
            ("retained_session_ids", retained_ids),
        ):
            observed = {
                tuple(str(item) for item in _json_container(value, name=field))
                for value in group[field]
            }
            if observed != {expected_ids}:
                raise ValueError(f"outer/comparison {field} receipts disagree")
        common = _common_regions(comparison.get("common_regions"))
        order_indices = [REGION_ORDER.index(region) for region in common]
        if order_indices != sorted(order_indices):
            raise ValueError("common regions do not follow fixed REGION_ORDER")
        coverage = _json_container(
            comparison.get("region_session_coverage"),
            name="region_session_coverage",
        )
        if not isinstance(coverage, (list, tuple)) or [
            str(item.get("region")) for item in coverage if isinstance(item, dict)
        ] != list(common):
            raise ValueError("region coverage order differs from common regions")
        normalized_coverage = json.dumps(coverage, sort_keys=True)
        for value in group["region_session_coverage"]:
            observed = _json_container(value, name="region_session_coverage")
            if json.dumps(observed, sort_keys=True) != normalized_coverage:
                raise ValueError("outer/comparison region coverage receipts disagree")
        missing_by_region: dict[str, set[str]] = {}
        for item in coverage:
            if not isinstance(item, dict):
                raise ValueError("region coverage item must be an object")
            region = str(item["region"])
            present = int(item["n_sessions_present"])
            missing = int(item["n_sessions_missing"])
            missing_id_values = tuple(
                str(value)
                for value in _json_container(
                    item["missing_session_ids"], name="missing_session_ids"
                )
            )
            missing_ids = set(missing_id_values)
            if (
                present < int(config["minimum_region_sessions"])
                or present + missing != expected_sessions
                or len(missing_ids) != len(missing_id_values)
                or len(missing_ids) != missing
                or not missing_ids <= set(session_ids)
                or not np.isclose(
                    float(item["session_fraction_present"]),
                    present / expected_sessions,
                )
            ):
                raise ValueError("region coverage count/fraction/missing IDs disagree")
            missing_by_region[region] = missing_ids
        observed_missing: dict[str, set[str]] = {region: set() for region in common}
        for session_id, session_rows in group.groupby("session_id"):
            present_values = {
                _common_regions(value)
                for value in session_rows["anchor_regions_present"]
            }
            missing_values = {
                tuple(
                    str(item)
                    for item in _json_container(value, name="anchor_regions_missing")
                )
                for value in session_rows["anchor_regions_missing"]
            }
            if len(present_values) != 1 or len(missing_values) != 1:
                raise ValueError("per-session anchor receipts differ across families")
            present_regions = next(iter(present_values))
            missing_regions = next(iter(missing_values))
            if (
                len(set(missing_regions)) != len(missing_regions)
                or set(present_regions).isdisjoint(missing_regions) is False
                or set(present_regions) | set(missing_regions) != set(common)
                or set(session_rows["n_anchor_regions_present"].astype(int))
                != {len(present_regions)}
                or set(session_rows["n_anchor_regions_missing"].astype(int))
                != {len(missing_regions)}
            ):
                raise ValueError(
                    "per-session present/missing anchor receipt is invalid"
                )
            for region in missing_regions:
                observed_missing[region].add(str(session_id))
        if observed_missing != missing_by_region:
            raise ValueError("coverage missing IDs disagree with per-session receipts")


def _animal_mean(frame: pd.DataFrame, column: str) -> float:
    values = frame.groupby("animal_id", sort=True)[column].mean()
    return float(values.mean())


def _summarize_conditions(outer_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (view, panel, family), group in outer_frame.groupby(
        ["view", "panel", "model_family"], sort=True
    ):
        counts = group["parameter_count"].astype(int).unique()
        if len(counts) != 1:
            raise ValueError("a model family has inconsistent parameter counts")
        rows.append(
            {
                "view": view,
                "panel": panel,
                "model_family": family,
                "claim_scope": panel_claim_scope(str(view), str(panel)),
                "n_sessions": group["session_id"].nunique(),
                "n_animals": group["animal_id"].nunique(),
                "animal_mean_nll_per_count": _animal_mean(group, "nll_per_count"),
                "animal_mean_pseudo_r2": _animal_mean(group, "pseudo_r2"),
                "animal_mean_closure_mse": _animal_mean(group, "closure_mse"),
                "parameter_count": int(counts[0]),
            }
        )
    return pd.DataFrame(rows)


def _flatten_comparison(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    nested_selection_valid: bool = True,
) -> dict[str, Any]:
    comparison = record.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError(
            "animal/session comparison record lacks its registered comparison"
        )
    intervals = {}
    for name in ("shared_vs_common", "full_vs_common", "retention_margin"):
        interval = comparison.get(name)
        if not isinstance(interval, dict):
            raise ValueError(f"comparison lacks {name}")
        if interval.get("n_sessions") is None or interval.get("n_animals") is None:
            raise ValueError(f"comparison {name} lacks cohort counts")
        intervals[name] = interval
    if (
        len({int(item["n_sessions"]) for item in intervals.values()}) != 1
        or len({int(item["n_animals"]) for item in intervals.values()}) != 1
    ):
        raise ValueError("comparison intervals disagree on statistical units")
    view, panel = str(record["view"]), str(record["panel"])
    scope = panel_claim_scope(view, panel)
    if (
        record.get("status") != "complete"
        or record.get("aggregation_level") != "animal_with_session_nested"
        or record.get("claim_scope") != scope
        or record.get("core_claim_eligible") is not (scope == "registered_primary")
        or record.get("nested_selection_objective")
        != "mean_animal_validation_nll_across_common_shared_full"
        or comparison.get("inference_unit") != "animal_with_session_nested"
        or record.get("likelihood_kind") != "one_step_conditional_poisson"
        or record.get("full_latent_lds") is not False
        or (
            scope == "registered_primary"
            and record.get("causal_timing_eligible") is not True
        )
    ):
        raise ValueError("comparison violates its animal-with-session-nested scope")
    n_sessions = int(intervals["shared_vs_common"]["n_sessions"])
    n_animals = int(intervals["shared_vs_common"]["n_animals"])
    shared = intervals["shared_vs_common"]
    full = intervals["full_vs_common"]
    retention = intervals["retention_margin"]
    complete = (
        bool(comparison.get("complete_cohort"))
        and n_sessions >= int(config["planned_sessions"])
        and n_animals >= int(config["planned_animals"])
    )
    fewer = bool(comparison.get("shared_has_fewer_parameters")) and int(
        comparison["shared_parameter_count"]
    ) < int(comparison["full_parameter_count"])
    retained_ratio = float(comparison["retained_full_gain_ratio"])
    common_regions = _common_regions(record.get("common_regions"))
    region_coverage = _json_container(
        record.get("region_session_coverage"), name="region_session_coverage"
    )
    if not isinstance(region_coverage, (list, tuple)):
        raise ValueError("region_session_coverage must be a sequence")
    support = (
        complete
        and float(shared["ci_low"]) > 0
        and float(shared["holm_adjusted_p"]) < 0.05
        and float(full["ci_low"]) > 0
        and float(full["holm_adjusted_p"]) < 0.05
        and bool(comparison.get("retention_defined"))
        and retained_ratio >= 0.9
        and float(retention["ci_low"]) >= 0
        and float(retention["holm_adjusted_p"]) < 0.05
        and fewer
        and comparison.get("conclusion") == "support"
    )
    oppose = (
        complete
        and comparison.get("conclusion") == "oppose"
        and (
            (float(shared["ci_high"]) < 0 and float(shared["holm_adjusted_p"]) < 0.05)
            or (
                bool(comparison.get("retention_defined"))
                and float(retention["ci_high"]) < 0
                and float(retention["holm_adjusted_p"]) < 0.05
            )
        )
    )
    conclusion = "support" if support else "oppose" if oppose else "inconclusive"
    if comparison.get("conclusion") in {
        "support",
        "oppose",
    } and conclusion != comparison.get("conclusion"):
        raise ValueError(
            "stored comparison conclusion fails the registered exp14 gates"
        )
    result: dict[str, Any] = {
        "view": view,
        "panel": panel,
        "claim_scope": scope,
        "core_claim_eligible": scope == "registered_primary" and nested_selection_valid,
        "n_sessions": n_sessions,
        "n_animals": n_animals,
        "complete_cohort": complete,
        "shared_parameter_count": int(comparison["shared_parameter_count"]),
        "full_parameter_count": int(comparison["full_parameter_count"]),
        "shared_has_fewer_parameters": fewer,
        "retained_full_gain_ratio": retained_ratio,
        "retention_defined": bool(comparison.get("retention_defined")),
        "panel_conclusion": conclusion,
        "core_conclusion": conclusion
        if scope == "registered_primary" and nested_selection_valid
        else "inconclusive",
        "nested_selection_valid": nested_selection_valid,
        "likelihood_kind": record.get("likelihood_kind"),
        "full_latent_lds": bool(record.get("full_latent_lds")),
        "causal_timing_eligible": bool(record.get("causal_timing_eligible")),
        "common_regions_json": json.dumps(list(common_regions), separators=(",", ":")),
        "region_session_coverage_json": json.dumps(
            region_coverage, sort_keys=True, separators=(",", ":")
        ),
        "minimum_region_sessions": int(record.get("minimum_region_sessions", -1)),
        "region_anchor_policy": str(record.get("region_anchor_policy")),
        "region_imputation_strategy": str(record.get("region_imputation_strategy")),
        "macro_region_mapping_schema": record.get("macro_region_mapping_schema"),
        "macro_region_mapping_sha256": record.get("macro_region_mapping_sha256"),
        "macro_region_source_ontology_sha256": record.get(
            "macro_region_source_ontology_sha256"
        ),
        "macro_region_source_provenance_sha256": record.get(
            "macro_region_source_provenance_sha256"
        ),
        "macro_region_formal_acronym_count": int(
            record.get("macro_region_formal_acronym_count", 0)
        ),
        "macro_region_formal_acronyms_sha256": record.get(
            "macro_region_formal_acronyms_sha256"
        ),
        "macro_region_mapping_formal_compact_manifest_sha256": config.get(
            "macro_region_mapping_formal_compact_manifest_sha256"
        ),
        "hmm_restart_selection_policy": _formal_hmm_contract(config)[0],
        "hmm_n_restarts": _formal_hmm_contract(config)[1],
    }
    for name, interval in intervals.items():
        for field in (
            "estimate",
            "ci_low",
            "ci_high",
            "bootstrap_p_two_sided",
            "holm_adjusted_p",
        ):
            result[f"{name}_{field}"] = float(interval[field])
    return result


def _recompute_comparison(
    outer_frame: pd.DataFrame,
    *,
    view: str,
    panel: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    scoped = outer_frame.loc[
        (outer_frame["view"].astype(str) == view)
        & (outer_frame["panel"].astype(str) == panel)
    ].copy()
    if set(scoped["model_family"].astype(str)) != set(FAMILIES):
        raise ValueError("outer rows lack a common/shared/full family")
    family_sessions = {
        family: set(
            scoped.loc[
                scoped["model_family"].astype(str) == family, "session_id"
            ].astype(str)
        )
        for family in FAMILIES
    }
    if len({frozenset(value) for value in family_sessions.values()}) != 1 or len(
        next(iter(family_sessions.values()))
    ) != int(config["planned_sessions"]):
        raise ValueError("outer model families do not score the same 20 sessions")
    if scoped.duplicated(["session_id", "model_family"]).any():
        raise ValueError("outer rows duplicate a session/model-family pair")
    animal_map = scoped.groupby("session_id")["animal_id"].nunique()
    if animal_map.ne(1).any() or scoped["animal_id"].astype(str).nunique() < int(
        config["planned_animals"]
    ):
        raise ValueError("outer rows have an invalid session-to-animal map")
    required = {
        "n_transitions",
        "n_count_observations",
        "log_likelihood",
        "null_log_likelihood",
        "saturated_log_likelihood",
        "nll_per_count",
        "pseudo_r2",
        "closure_mse",
        "parameter_count",
    }
    missing = sorted(required - set(scoped.columns))
    if missing:
        raise ValueError(f"outer rows lack score fields: {missing}")
    scores: dict[str, HierarchicalCountScore] = {}
    for family in FAMILIES:
        rows = scoped.loc[scoped["model_family"].astype(str) == family].sort_values(
            "session_id"
        )
        parameter_counts = rows["parameter_count"].astype(int).unique()
        if len(parameter_counts) != 1:
            raise ValueError("outer rows have inconsistent family parameter counts")
        per_session = tuple(
            SessionCountMetrics(
                session_id=str(row.session_id),
                animal_id=str(row.animal_id),
                n_transitions=int(row.n_transitions),
                n_count_observations=int(row.n_count_observations),
                log_likelihood=float(row.log_likelihood),
                null_log_likelihood=float(row.null_log_likelihood),
                saturated_log_likelihood=float(row.saturated_log_likelihood),
                nll_per_count=float(row.nll_per_count),
                pseudo_r2=float(row.pseudo_r2),
                closure_mse=float(row.closure_mse),
            )
            for row in rows.itertuples(index=False)
        )
        scores[family] = HierarchicalCountScore(
            family=family,
            per_session=per_session,
            parameter_count=int(parameter_counts[0]),
            nll_per_count=float(rows["nll_per_count"].mean()),
            pseudo_r2=float(rows["pseudo_r2"].mean()),
            closure_mse=float(rows["closure_mse"].mean()),
        )
    result = compare_count_families(
        scores,
        planned_sessions=int(config["planned_sessions"]),
        planned_animals=int(config["planned_animals"]),
        n_bootstrap=int(config["n_bootstrap"]),
        seed=derive_seed(0, "exp14", view, panel, "bootstrap"),
    )
    return asdict(result)


def _assert_nested_close(
    stored: Any, recomputed: Any, *, path: str = "comparison"
) -> None:
    if isinstance(recomputed, dict):
        if not isinstance(stored, dict) or set(stored) != set(recomputed):
            raise ValueError(f"{path} structure differs from recomputed comparison")
        for key, value in recomputed.items():
            _assert_nested_close(stored[key], value, path=f"{path}.{key}")
        return
    if isinstance(recomputed, float):
        stored_is_serialized_nan = (
            isinstance(stored, str) and stored == "nan" and np.isnan(recomputed)
        )
        if not stored_is_serialized_nan and (
            not isinstance(stored, (int, float))
            or not np.isclose(
                float(stored), recomputed, rtol=1e-12, atol=1e-12, equal_nan=True
            )
        ):
            raise ValueError(f"{path} differs from recomputed comparison")
        return
    if stored != recomputed:
        raise ValueError(f"{path} differs from recomputed comparison")


def _nested_selection_valid(
    metrics: list[dict[str, Any]],
    outer_frame: pd.DataFrame,
    comparison_records: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> bool:
    nested = [row for row in metrics if row.get("stage") == "nested_selection"]
    comparisons = {
        (str(row["view"]), str(row["panel"])): row for row in comparison_records
    }
    expected_grid = set()
    for view in map(str, config["views"]):
        for panel in map(str, config["panels"]):
            regions = _common_regions(
                comparisons[(view, panel)].get("common_regions", ())
            )
            n_regions = len(regions)
            valid_dims = [
                int(dim) for dim in config["latent_dims"] if 1 <= int(dim) <= n_regions
            ]
            expected_grid.update(
                (view, panel, dim, float(ridge))
                for dim in valid_dims
                for ridge in config["ridges"]
            )
    observed_grid = {
        (
            str(row.get("view")),
            str(row.get("panel")),
            int(row.get("latent_dim", -1)),
            float(row.get("ridge", "nan")),
        )
        for row in nested
    }
    if len(nested) != len(expected_grid) or observed_grid != expected_grid:
        return False
    for view in map(str, config["views"]):
        for panel in map(str, config["panels"]):
            rows = [
                row
                for row in nested
                if str(row.get("view")) == view and str(row.get("panel")) == panel
            ]
            if any(
                str(row.get("status")) not in {"complete", "failed"} for row in rows
            ):
                return False
            successful = []
            for row in rows:
                if row.get("status") == "complete":
                    objective = float(row.get("animal_mean_validation_nll", "nan"))
                    if not np.isfinite(objective):
                        return False
                    successful.append(
                        (objective, int(row["latent_dim"]), float(row["ridge"]))
                    )
            if not successful:
                return False
            _, selected_dim, selected_ridge = min(successful)
            scoped = outer_frame.loc[
                (outer_frame["view"].astype(str) == view)
                & (outer_frame["panel"].astype(str) == panel)
            ]
            if set(scoped["selected_latent_dim"].astype(int)) != {
                selected_dim
            } or not np.allclose(
                scoped["selected_ridge"].astype(float), selected_ridge
            ):
                raise ValueError(
                    "outer rows disagree with nested-selected hyperparameters"
                )
            comparison = comparisons[(view, panel)]
            if int(
                comparison.get("selected_latent_dim", -1)
            ) != selected_dim or not np.isclose(
                float(comparison.get("selected_ridge", "nan")), selected_ridge
            ):
                raise ValueError(
                    "animal/session comparison disagrees with nested selection"
                )
    return True


def collect_formal_run(
    results_root: Path, formal_config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect exactly one complete clean formal run and retain every metric row."""

    _validate_formal_config(formal_config)
    attempt = _latest_matching_attempt(
        results_root, expected_config=_expected_attempt_config(formal_config)
    )
    status = _read_json(attempt / "status.json")
    environment = _read_json(attempt / "environment.json")
    manifest = _read_json(attempt / "manifest.json")
    planned = _read_json(attempt / "planned_conditions.json")
    metrics = _read_records(attempt / "metrics.jsonl")
    git = environment.get("git", {})
    if status.get("status") != "complete" or manifest.get("status") != "complete":
        raise ValueError("formal exp14 snapshot requires status complete")
    if (
        not isinstance(git, dict)
        or git.get("dirty") is not False
        or not git.get("commit")
    ):
        raise ValueError("formal exp14 snapshot requires a clean Git worktree")
    if manifest.get("run_id") is None or any(
        row.get("run_id") != manifest["run_id"] for row in metrics
    ):
        raise ValueError("metric run IDs do not bind to the immutable manifest")
    expected_views = {str(item) for item in formal_config["views"]}
    expected_panels = {str(item) for item in formal_config["panels"]}
    if not isinstance(planned, list) or len(planned) != int(
        formal_config["planned_sessions"]
    ) * len(expected_views) * len(expected_panels) * len(FAMILIES):
        raise ValueError("planned exp14 condition family has the wrong size")
    planned_keys = {_condition_key(row) for row in planned}
    session_ids = {key[0] for key in planned_keys}
    expected_keys = {
        (session, view, panel, family)
        for session in session_ids
        for view in expected_views
        for panel in expected_panels
        for family in FAMILIES
    }
    if (
        len(session_ids) != int(formal_config["planned_sessions"])
        or planned_keys != expected_keys
    ):
        raise ValueError("planned exp14 condition family is incomplete")
    outer = [row for row in metrics if row.get("stage") == "outer_test"]
    if (
        len(outer) != len(planned_keys)
        or {_condition_key(row) for row in outer} != planned_keys
        or any(row.get("status") != "complete" for row in outer)
    ):
        raise ValueError(
            "planned failed, missing, or duplicate outer-test conditions are retained but ineligible"
        )
    source = str(formal_config["expected_source_manifest_sha256"])
    acquisition = str(formal_config["expected_acquisition_bundle_sha256"])
    compact = str(formal_config["expected_compact_manifest_sha256"])
    compact_bundle = str(formal_config["expected_compact_bundle_sha256"])
    bwm = str(formal_config["expected_bwm_repository_commit"])
    for row in outer:
        if (
            row.get("source_manifest_sha256") != source
            or row.get("acquisition_bundle_sha256") != acquisition
            or row.get("compact_manifest_sha256") != compact
            or row.get("compact_bundle_sha256") != compact_bundle
            or row.get("bwm_repository_commit") != bwm
            or row.get("aggregation_level") != "session"
            or row.get("statistics_unit") != "session_nested_within_animal"
            or row.get("preprocessing_fit_train_only") is not True
            or row.get("hidden_context_inference") is not True
            or row.get("test_context_observed") is not False
            or row.get("condition_schedule_used_for_split_only") is not True
            or row.get("nuisance_as_log_rate_controls") is not True
            or row.get("counts_residualized_before_poisson") is not False
            or row.get("likelihood_kind") != "one_step_conditional_poisson"
            or row.get("full_latent_lds") is not False
            or row.get("claim_scope")
            != panel_claim_scope(str(row.get("view")), str(row.get("panel")))
            or (
                row.get("claim_scope") == "registered_primary"
                and row.get("causal_timing_eligible") is not True
            )
        ):
            raise ValueError(
                "outer-test row violates provenance, leakage, or inference contract"
            )
        _require_digest(
            row.get("comparison_preprocessing_sha256"),
            "comparison_preprocessing_sha256",
        )
        _validate_hmm_restart_receipt(row, formal_config)
    comparison_records = [
        row for row in metrics if row.get("stage") == "animal_session_comparison"
    ]
    expected_pairs = {
        (view, panel) for view in expected_views for panel in expected_panels
    }
    if (
        len(comparison_records) != len(expected_pairs)
        or {(str(row.get("view")), str(row.get("panel"))) for row in comparison_records}
        != expected_pairs
    ):
        raise ValueError("animal/session comparison family is incomplete")
    outer_frame = pd.DataFrame(outer)
    _validate_paired_hmm_receipts(outer_frame)
    if outer_frame.groupby("session_id")["animal_id"].nunique().ne(1).any():
        raise ValueError("a session maps to different animals across views or panels")
    preprocessing_counts = outer_frame.groupby(["view", "panel"])[
        "comparison_preprocessing_sha256"
    ].nunique()
    if preprocessing_counts.ne(1).any():
        raise ValueError("a view/panel uses multiple preprocessing fingerprints")
    comparison_by_pair = {
        (str(row["view"]), str(row["panel"])): row for row in comparison_records
    }
    for pair, group in outer_frame.groupby(["view", "panel"]):
        outer_regions = {
            _common_regions(value) for value in group["common_regions"].tolist()
        }
        comparison_regions = _common_regions(
            comparison_by_pair[(str(pair[0]), str(pair[1]))].get("common_regions")
        )
        if outer_regions != {comparison_regions}:
            raise ValueError("outer and comparison common_regions disagree")
    _validate_region_receipts(outer_frame, comparison_records, formal_config)
    nested_valid = _nested_selection_valid(
        metrics, outer_frame, comparison_records, formal_config
    )
    for record in comparison_records:
        recomputed = _recompute_comparison(
            outer_frame,
            view=str(record["view"]),
            panel=str(record["panel"]),
            config=formal_config,
        )
        _assert_nested_close(record.get("comparison"), recomputed)
    comparisons = pd.DataFrame(
        [
            _flatten_comparison(row, formal_config, nested_selection_valid=nested_valid)
            for row in comparison_records
        ]
    )
    conditions = _summarize_conditions(outer_frame)
    raw = pd.DataFrame(metrics)
    for column in raw.columns:
        if raw[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            raw[column] = raw[column].map(
                lambda value: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
            )
    macro_mapping = _formal_macro_mapping(formal_config)
    macro_receipt = dict(macro_mapping.receipt())
    hmm_policy, hmm_n_restarts = _formal_hmm_contract(formal_config)
    run_manifest = pd.DataFrame(
        [
            {
                "seed": 0,
                "run_id": manifest["run_id"],
                "attempt_name": attempt.name,
                "status": status["status"],
                "git_commit": git["commit"],
                "git_dirty": False,
                "formal_config_sha256": _portable_formal_config_sha256(formal_config),
                "registered_formal_json_sha256": _registered_formal_json_sha256(),
                "source_manifest_sha256": source,
                "acquisition_bundle_sha256": acquisition,
                "compact_manifest_sha256": compact,
                "compact_bundle_sha256": compact_bundle,
                "bwm_repository_commit": bwm,
                "planned_sessions": int(formal_config["planned_sessions"]),
                "planned_animals": int(formal_config["planned_animals"]),
                "n_bootstrap": int(formal_config["n_bootstrap"]),
                "hmm_restart_selection_policy": hmm_policy,
                "hmm_n_restarts": hmm_n_restarts,
                "minimum_region_sessions": int(
                    formal_config["minimum_region_sessions"]
                ),
                "macro_region_mapping_path": formal_config["macro_region_mapping_path"],
                "macro_region_mapping_schema": macro_receipt[
                    "macro_region_mapping_schema"
                ],
                "macro_region_mapping_sha256": macro_receipt[
                    "macro_region_mapping_sha256"
                ],
                "macro_region_source_ontology_sha256": macro_receipt[
                    "macro_region_source_ontology_sha256"
                ],
                "macro_region_source_provenance_sha256": macro_receipt[
                    "macro_region_source_provenance_sha256"
                ],
                "macro_region_mapping_formal_compact_manifest_sha256": formal_config[
                    "macro_region_mapping_formal_compact_manifest_sha256"
                ],
                "macro_region_formal_acronym_count": macro_receipt[
                    "macro_region_formal_acronym_count"
                ],
                "macro_region_formal_acronyms_sha256": macro_receipt[
                    "macro_region_formal_acronyms_sha256"
                ],
                "views_json": json.dumps(list(formal_config["views"])),
                "panels_json": json.dumps(list(formal_config["panels"])),
                "latent_dims_json": json.dumps(list(formal_config["latent_dims"])),
                "ridges_json": json.dumps(list(formal_config["ridges"])),
                **{
                    f"{name.removesuffix('.json').replace('.', '_')}_sha256": _sha256(
                        attempt / name
                    )
                    for name in (
                        "config.json",
                        "environment.json",
                        "status.json",
                        "planned_conditions.json",
                        "metrics.jsonl",
                        "manifest.json",
                        "run.log",
                    )
                },
            }
        ]
    )
    return raw, conditions, comparisons, run_manifest


def _paths(results_root: Path, prefix: str) -> dict[str, Path]:
    return {
        name: results_root / f"{prefix}_{name}{suffix}"
        for name, suffix in {
            "raw": ".csv.gz",
            "conditions": ".csv",
            "comparisons": ".csv",
            "run_manifest": ".csv",
            "report": ".md",
        }.items()
    }


def load_validated_exp14_snapshot(
    results_root: Path, *, prefix: str = DEFAULT_PREFIX
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = _paths(results_root, prefix)
    required = [
        paths[name] for name in ("raw", "conditions", "comparisons", "run_manifest")
    ]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("exp14 formal snapshot is absent or partially present")
    binding_dtypes = {
        column: "string"
        for column in (
            "expected_source_manifest_sha256",
            "expected_acquisition_bundle_sha256",
            "expected_compact_manifest_sha256",
            "expected_compact_bundle_sha256",
            "expected_bwm_repository_commit",
            "formal_config_sha256",
            "registered_formal_json_sha256",
            "scoped_raw_sha256",
            "run_manifest_sha256",
            "run_git_commit",
            "macro_region_mapping_sha256",
            "macro_region_source_ontology_sha256",
            "macro_region_source_provenance_sha256",
            "macro_region_mapping_formal_compact_manifest_sha256",
            "macro_region_formal_acronyms_sha256",
            "hmm_restart_selection_policy",
        )
    }
    conditions, comparisons = (
        pd.read_csv(paths["conditions"], dtype=binding_dtypes),
        pd.read_csv(paths["comparisons"], dtype=binding_dtypes),
    )
    raw, run_manifest = (
        pd.read_csv(
            paths["raw"],
            low_memory=False,
            dtype={
                "source_manifest_sha256": "string",
                "acquisition_bundle_sha256": "string",
                "compact_manifest_sha256": "string",
                "compact_bundle_sha256": "string",
                "bwm_repository_commit": "string",
                "comparison_preprocessing_sha256": "string",
                "macro_region_mapping_sha256": "string",
                "macro_region_source_ontology_sha256": "string",
                "macro_region_source_provenance_sha256": "string",
                "macro_region_formal_acronyms_sha256": "string",
                "hmm_restart_selection_policy": "string",
            },
        ),
        pd.read_csv(
            paths["run_manifest"],
            dtype={
                "source_manifest_sha256": "string",
                "acquisition_bundle_sha256": "string",
                "compact_manifest_sha256": "string",
                "compact_bundle_sha256": "string",
                "bwm_repository_commit": "string",
                "formal_config_sha256": "string",
                "registered_formal_json_sha256": "string",
                "git_commit": "string",
                "macro_region_mapping_sha256": "string",
                "macro_region_source_ontology_sha256": "string",
                "macro_region_source_provenance_sha256": "string",
                "macro_region_mapping_formal_compact_manifest_sha256": "string",
                "macro_region_formal_acronyms_sha256": "string",
            },
        ),
    )
    raw_sha, manifest_sha = _sha256(paths["raw"]), _sha256(paths["run_manifest"])
    for frame, name in ((conditions, "conditions"), (comparisons, "comparisons")):
        if set(frame["scoped_raw_sha256"].astype(str)) != {raw_sha} or set(
            frame["run_manifest_sha256"].astype(str)
        ) != {manifest_sha}:
            raise ValueError(f"exp14 {name} does not bind its raw/run manifest")
    registered_snapshot = prefix == DEFAULT_PREFIX
    expected_scope = "registered_core" if registered_snapshot else "exploratory_only"
    for frame, name in ((conditions, "conditions"), (comparisons, "comparisons")):
        if set(frame["snapshot_scope"].astype(str)) != {expected_scope}:
            raise ValueError(f"exp14 {name} has an invalid snapshot scope")
    if (
        len(run_manifest) != 1
        or int(run_manifest.iloc[0]["seed"]) != 0
        or str(run_manifest.iloc[0]["status"]) != "complete"
        or str(run_manifest.iloc[0]["git_dirty"]).lower() != "false"
    ):
        raise ValueError(
            "exp14 snapshot run manifest is not a clean complete seed=0 run"
        )
    binding_map = {
        "expected_source_manifest_sha256": "source_manifest_sha256",
        "expected_acquisition_bundle_sha256": "acquisition_bundle_sha256",
        "expected_compact_manifest_sha256": "compact_manifest_sha256",
        "expected_compact_bundle_sha256": "compact_bundle_sha256",
        "expected_bwm_repository_commit": "bwm_repository_commit",
        "formal_config_sha256": "formal_config_sha256",
        "registered_formal_json_sha256": "registered_formal_json_sha256",
        "run_git_commit": "git_commit",
        "macro_region_mapping_sha256": "macro_region_mapping_sha256",
        "macro_region_source_ontology_sha256": "macro_region_source_ontology_sha256",
        "macro_region_source_provenance_sha256": "macro_region_source_provenance_sha256",
        "macro_region_mapping_schema": "macro_region_mapping_schema",
        "macro_region_mapping_formal_compact_manifest_sha256": (
            "macro_region_mapping_formal_compact_manifest_sha256"
        ),
        "macro_region_formal_acronyms_sha256": ("macro_region_formal_acronyms_sha256"),
        "macro_region_formal_acronym_count": "macro_region_formal_acronym_count",
        "hmm_restart_selection_policy": "hmm_restart_selection_policy",
        "hmm_n_restarts": "hmm_n_restarts",
    }
    for output_column, manifest_column in binding_map.items():
        expected_value = str(run_manifest.iloc[0][manifest_column])
        for frame, name in ((conditions, "conditions"), (comparisons, "comparisons")):
            if set(frame[output_column].astype(str)) != {expected_value}:
                raise ValueError(
                    f"exp14 {name} {output_column} disagrees with run manifest"
                )
    if registered_snapshot:
        registered = _registered_formal_config()
        registered_mapping = _formal_macro_mapping(registered)
        expected_bindings = {
            "formal_config_sha256": _portable_formal_config_sha256(registered),
            "registered_formal_json_sha256": _registered_formal_json_sha256(),
            "expected_source_manifest_sha256": registered[
                "expected_source_manifest_sha256"
            ],
            "expected_acquisition_bundle_sha256": registered[
                "expected_acquisition_bundle_sha256"
            ],
            "expected_compact_manifest_sha256": registered[
                "expected_compact_manifest_sha256"
            ],
            "expected_compact_bundle_sha256": registered[
                "expected_compact_bundle_sha256"
            ],
            "expected_bwm_repository_commit": registered[
                "expected_bwm_repository_commit"
            ],
            "macro_region_mapping_sha256": registered[
                "expected_macro_region_mapping_sha256"
            ],
            "macro_region_source_ontology_sha256": registered[
                "expected_macro_region_source_ontology_sha256"
            ],
            "macro_region_source_provenance_sha256": registered[
                "expected_macro_region_source_provenance_sha256"
            ],
            "macro_region_mapping_schema": registered[
                "expected_macro_region_mapping_schema"
            ],
            "macro_region_mapping_formal_compact_manifest_sha256": registered[
                "macro_region_mapping_formal_compact_manifest_sha256"
            ],
            "macro_region_formal_acronym_count": (
                registered_mapping.formal_compact_acronym_count
            ),
            "macro_region_formal_acronyms_sha256": (
                registered_mapping.formal_compact_acronyms_sha256
            ),
            "hmm_restart_selection_policy": registered["learned_hmm"][
                "restart_selection_policy"
            ],
            "hmm_n_restarts": registered["learned_hmm"]["n_restarts"],
        }
        for column, expected_value in expected_bindings.items():
            if set(comparisons[column].astype(str)) != {str(expected_value)}:
                raise ValueError(
                    "default exp14 snapshot differs from the frozen registered config"
                )
    elif comparisons["core_conclusion"].astype(str).ne("inconclusive").any():
        raise ValueError("exploratory exp14 snapshot promoted a core conclusion")
    expected = {
        ("stimulus_pre", "primary_past_safe"),
        ("stimulus_pre", "full_trial_sensitivity"),
        ("movement_pre", "primary_past_safe"),
        ("movement_pre", "full_trial_sensitivity"),
    }
    observed = set(
        zip(
            comparisons["view"].astype(str),
            comparisons["panel"].astype(str),
            strict=True,
        )
    )
    if observed != expected or len(comparisons) != 4:
        raise ValueError("exp14 comparison snapshot lacks a registered view/panel")
    primary = comparisons.loc[
        (comparisons["view"] == "stimulus_pre")
        & (comparisons["panel"] == "primary_past_safe")
    ]
    if len(primary) != 1 or primary.iloc[0]["claim_scope"] != "registered_primary":
        raise ValueError(
            "exp14 snapshot lacks its unique registered primary comparison"
        )
    if (
        comparisons.loc[
            comparisons["claim_scope"] != "registered_primary", "core_conclusion"
        ]
        .astype(str)
        .ne("inconclusive")
        .any()
    ):
        raise ValueError("an exp14 sensitivity panel was promoted to a core conclusion")
    outer = raw.loc[raw["stage"].astype(str) == "outer_test"].copy()
    if len(outer) != 240 or outer["status"].astype(str).ne("complete").any():
        raise ValueError(
            "exp14 raw snapshot does not preserve 240 complete planned cells"
        )
    expected_outer_bindings = {
        "source_manifest_sha256": str(run_manifest.iloc[0]["source_manifest_sha256"]),
        "acquisition_bundle_sha256": str(
            run_manifest.iloc[0]["acquisition_bundle_sha256"]
        ),
        "compact_manifest_sha256": str(run_manifest.iloc[0]["compact_manifest_sha256"]),
        "compact_bundle_sha256": str(run_manifest.iloc[0]["compact_bundle_sha256"]),
        "bwm_repository_commit": str(run_manifest.iloc[0]["bwm_repository_commit"]),
    }
    for record in outer.to_dict("records"):
        if any(
            record.get(key) != value for key, value in expected_outer_bindings.items()
        ):
            raise ValueError("exp14 raw outer provenance disagrees with run manifest")
        scope = panel_claim_scope(str(record["view"]), str(record["panel"]))
        if (
            record.get("preprocessing_fit_train_only") is not True
            or record.get("hidden_context_inference") is not True
            or record.get("test_context_observed") is not False
            or record.get("condition_schedule_used_for_split_only") is not True
            or record.get("nuisance_as_log_rate_controls") is not True
            or record.get("counts_residualized_before_poisson") is not False
            or record.get("likelihood_kind") != "one_step_conditional_poisson"
            or record.get("full_latent_lds") is not False
            or record.get("claim_scope") != scope
            or (
                scope == "registered_primary"
                and record.get("causal_timing_eligible") is not True
            )
        ):
            raise ValueError("exp14 raw outer row violates its mechanism scope")
        _require_digest(
            record.get("comparison_preprocessing_sha256"),
            "comparison_preprocessing_sha256",
        )
        _validate_hmm_restart_receipt(
            record,
            {
                "learned_hmm": {
                    "restart_selection_policy": str(
                        run_manifest.iloc[0]["hmm_restart_selection_policy"]
                    ),
                    "n_restarts": int(run_manifest.iloc[0]["hmm_n_restarts"]),
                    "require_converged": True,
                    "require_identifiable": True,
                }
            },
        )
    if outer.groupby("session_id")["animal_id"].nunique().ne(1).any():
        raise ValueError("a raw session maps to multiple animals")
    _validate_paired_hmm_receipts(outer)
    if (
        outer.groupby(["view", "panel"])["comparison_preprocessing_sha256"]
        .nunique()
        .ne(1)
        .any()
    ):
        raise ValueError("raw view/panel preprocessing fingerprints disagree")
    recomputed_conditions = _summarize_conditions(outer)
    base_condition_columns = list(recomputed_conditions.columns)
    pd.testing.assert_frame_equal(
        conditions[base_condition_columns]
        .sort_values(["view", "panel", "model_family"])
        .reset_index(drop=True),
        recomputed_conditions.sort_values(
            ["view", "panel", "model_family"]
        ).reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    comparison_records = raw.loc[
        raw["stage"].astype(str) == "animal_session_comparison"
    ]
    if len(comparison_records) != 4:
        raise ValueError("exp14 raw snapshot lacks four animal/session comparisons")
    comparison_regions = {
        (str(row["view"]), str(row["panel"])): _common_regions(
            row.get("common_regions")
        )
        for row in comparison_records.to_dict("records")
    }
    for pair, group in outer.groupby(["view", "panel"]):
        regions = {_common_regions(value) for value in group["common_regions"]}
        if regions != {comparison_regions[(str(pair[0]), str(pair[1]))]}:
            raise ValueError("raw outer/comparison common_regions disagree")
    recomputed_rows = []
    recompute_config = {
        "planned_sessions": int(run_manifest.iloc[0]["planned_sessions"]),
        "planned_animals": int(run_manifest.iloc[0]["planned_animals"]),
        "n_bootstrap": int(run_manifest.iloc[0]["n_bootstrap"]),
        "minimum_region_sessions": int(run_manifest.iloc[0]["minimum_region_sessions"]),
        "macro_region_mapping_path": str(
            run_manifest.iloc[0]["macro_region_mapping_path"]
        ),
        "expected_macro_region_mapping_sha256": str(
            run_manifest.iloc[0]["macro_region_mapping_sha256"]
        ),
        "expected_macro_region_mapping_schema": str(
            run_manifest.iloc[0]["macro_region_mapping_schema"]
        ),
        "expected_macro_region_source_ontology_sha256": str(
            run_manifest.iloc[0]["macro_region_source_ontology_sha256"]
        ),
        "expected_macro_region_source_provenance_sha256": str(
            run_manifest.iloc[0]["macro_region_source_provenance_sha256"]
        ),
        "macro_region_mapping_formal_compact_manifest_sha256": str(
            run_manifest.iloc[0]["macro_region_mapping_formal_compact_manifest_sha256"]
        ),
        "expected_compact_manifest_sha256": str(
            run_manifest.iloc[0]["compact_manifest_sha256"]
        ),
        "learned_hmm": {
            "restart_selection_policy": str(
                run_manifest.iloc[0]["hmm_restart_selection_policy"]
            ),
            "n_restarts": int(run_manifest.iloc[0]["hmm_n_restarts"]),
            "require_converged": True,
            "require_identifiable": True,
        },
    }
    if (
        recompute_config["planned_sessions"] != 20
        or recompute_config["planned_animals"] < 5
        or recompute_config["n_bootstrap"] < 100
    ):
        raise ValueError("exp14 run manifest violates the formal inference contract")
    _validate_region_receipts(
        outer, comparison_records.to_dict("records"), recompute_config
    )
    nested_valid = _nested_selection_valid(
        raw.to_dict("records"),
        outer,
        comparison_records.to_dict("records"),
        {
            **recompute_config,
            "views": json.loads(str(run_manifest.iloc[0]["views_json"])),
            "panels": json.loads(str(run_manifest.iloc[0]["panels_json"])),
            "latent_dims": json.loads(str(run_manifest.iloc[0]["latent_dims_json"])),
            "ridges": json.loads(str(run_manifest.iloc[0]["ridges_json"])),
        },
    )
    for record in comparison_records.to_dict("records"):
        value = record.get("comparison")
        if not isinstance(value, str):
            raise ValueError("exp14 raw comparison payload is not canonical JSON")
        record["comparison"] = json.loads(value)
        independently_recomputed = _recompute_comparison(
            outer,
            view=str(record["view"]),
            panel=str(record["panel"]),
            config=recompute_config,
        )
        _assert_nested_close(record["comparison"], independently_recomputed)
        recomputed_rows.append(
            _flatten_comparison(
                record,
                recompute_config,
                nested_selection_valid=nested_valid,
            )
        )
    recomputed_comparisons = pd.DataFrame(recomputed_rows)
    if not registered_snapshot:
        recomputed_comparisons["core_claim_eligible"] = False
        recomputed_comparisons["core_conclusion"] = "inconclusive"
    base_comparison_columns = list(recomputed_comparisons.columns)
    pd.testing.assert_frame_equal(
        comparisons[base_comparison_columns]
        .sort_values(["view", "panel"])
        .reset_index(drop=True),
        recomputed_comparisons.sort_values(["view", "panel"]).reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    return conditions, comparisons, raw, run_manifest


def _write_report(
    path: Path, conditions: pd.DataFrame, comparisons: pd.DataFrame
) -> None:
    primary = comparisons.loc[comparisons["claim_scope"] == "registered_primary"].iloc[
        0
    ]
    common_regions = tuple(json.loads(str(primary["common_regions_json"])))
    region_coverage = json.loads(str(primary["region_session_coverage_json"]))
    lines = [
        "# Exp14 IBL multi-session neural audit",
        "",
        "The endpoint is held-out one-step conditional Poisson likelihood, not a full latent-LDS marginal likelihood.",
        "",
        f"- Registered primary: `stimulus_pre / primary_past_safe` — **{primary['core_conclusion']}**.",
        f"- Cohort: {int(primary['n_sessions'])} sessions nested within {int(primary['n_animals'])} animals.",
        f"- Raw SHA-256: `{primary['scoped_raw_sha256']}`",
        f"- Run-manifest SHA-256: `{primary['run_manifest_sha256']}`",
        f"- Compact manifest SHA-256: `{primary['expected_compact_manifest_sha256']}`",
        f"- Compact bundle SHA-256: `{primary['expected_compact_bundle_sha256']}`",
        f"- Registered formal JSON SHA-256: `{primary['registered_formal_json_sha256']}`",
        f"- Portable formal-config SHA-256: `{primary['formal_config_sha256']}`",
        f"- Snapshot scope: `{primary['snapshot_scope']}`",
        f"- Anatomical anchor policy: `{primary['region_anchor_policy']}`; "
        f"minimum session coverage={int(primary['minimum_region_sessions'])}.",
        f"- Allen macro mapping: `{primary['macro_region_mapping_schema']}`; "
        f"artifact SHA-256=`{primary['macro_region_mapping_sha256']}`.",
        f"- Allen ontology source SHA-256: "
        f"`{primary['macro_region_source_ontology_sha256']}`.",
        f"- Allen ontology provenance SHA-256: "
        f"`{primary['macro_region_source_provenance_sha256']}`.",
        f"- Macro mapping compact scope: "
        f"`{primary['macro_region_mapping_formal_compact_manifest_sha256']}`; "
        f"acronyms={int(primary['macro_region_formal_acronym_count'])}; "
        f"acronym-set SHA-256=`{primary['macro_region_formal_acronyms_sha256']}`.",
        f"- Missing-region handling: `{primary['region_imputation_strategy']}` "
        "(training folds only).",
        f"- Shared anchor basis: `{', '.join(common_regions)}`.",
        f"- HMM restart selection: `{primary['hmm_restart_selection_policy']}`; "
        f"registered restarts={int(primary['hmm_n_restarts'])}; every retained "
        "outer fit had at least one converged, identifiable eligible restart.",
        "",
        "## Anatomical anchor coverage",
        "",
        "| Region | Sessions present | Sessions missing | Fraction present |",
        "|---|---:|---:|---:|",
        *[
            f"| {item['region']} | {int(item['n_sessions_present'])} | "
            f"{int(item['n_sessions_missing'])} | "
            f"{float(item['session_fraction_present']):.3f} |"
            for item in region_coverage
        ],
        "",
        "## Absolute model views",
        "",
        "| View | Panel | Scope | Family | Animal-mean NLL/count | Pseudo-R² | Parameters |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in conditions.to_dict("records"):
        lines.append(
            f"| {row['view']} | {row['panel']} | {row['claim_scope']} | {row['model_family']} | {row['animal_mean_nll_per_count']:.6f} | {row['animal_mean_pseudo_r2']:.6f} | {int(row['parameter_count'])} |"
        )
    lines += [
        "",
        "## Animal-primary paired comparisons",
        "",
        "| View | Panel | Scope | Common - shared NLL/count (positive favors shared) [95% CI] | 90% retention ratio | Panel conclusion | Core conclusion |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in comparisons.to_dict("records"):
        lines.append(
            f"| {row['view']} | {row['panel']} | {row['claim_scope']} | {row['shared_vs_common_estimate']:.6f} [{row['shared_vs_common_ci_low']:.6f}, {row['shared_vs_common_ci_high']:.6f}] | {row['retained_full_gain_ratio']:.4f} | {row['panel_conclusion']} | {row['core_conclusion']} |"
        )
    lines += [
        "",
        "Only the registered stimulus-pre/past-safe panel can update the core claim. Movement-pre and full-trial-covariate panels are sensitivity analyses even when their panel-level result is conclusive.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def build_snapshot(
    results_root: Path, config: Mapping[str, Any], *, prefix: str = DEFAULT_PREFIX
) -> None:
    matches_registered = _portable_formal_config(config) == _portable_formal_config(
        _registered_formal_config()
    )
    registered = matches_registered and prefix == DEFAULT_PREFIX
    if prefix == DEFAULT_PREFIX and not matches_registered:
        raise ValueError(
            "default exp14 snapshot must exactly match the frozen registered config"
        )
    raw, conditions, comparisons, run_manifest = collect_formal_run(
        results_root, config
    )
    if not registered:
        comparisons["core_claim_eligible"] = False
        comparisons["core_conclusion"] = "inconclusive"
    paths = _paths(results_root, prefix)
    raw.to_csv(
        paths["raw"],
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    run_manifest.to_csv(paths["run_manifest"], index=False, lineterminator="\n")
    bindings = {
        "expected_source_manifest_sha256": config["expected_source_manifest_sha256"],
        "expected_acquisition_bundle_sha256": config[
            "expected_acquisition_bundle_sha256"
        ],
        "expected_compact_manifest_sha256": config["expected_compact_manifest_sha256"],
        "expected_compact_bundle_sha256": config["expected_compact_bundle_sha256"],
        "expected_bwm_repository_commit": config["expected_bwm_repository_commit"],
        "macro_region_mapping_sha256": config["expected_macro_region_mapping_sha256"],
        "macro_region_source_ontology_sha256": config[
            "expected_macro_region_source_ontology_sha256"
        ],
        "macro_region_source_provenance_sha256": config[
            "expected_macro_region_source_provenance_sha256"
        ],
        "macro_region_mapping_schema": config["expected_macro_region_mapping_schema"],
        "macro_region_mapping_formal_compact_manifest_sha256": config[
            "macro_region_mapping_formal_compact_manifest_sha256"
        ],
        "macro_region_formal_acronyms_sha256": run_manifest.iloc[0][
            "macro_region_formal_acronyms_sha256"
        ],
        "macro_region_formal_acronym_count": int(
            run_manifest.iloc[0]["macro_region_formal_acronym_count"]
        ),
        "formal_config_sha256": _portable_formal_config_sha256(config),
        "registered_formal_json_sha256": _registered_formal_json_sha256(),
        "scoped_raw_sha256": _sha256(paths["raw"]),
        "run_manifest_sha256": _sha256(paths["run_manifest"]),
        "run_git_commit": run_manifest.iloc[0]["git_commit"],
        "run_git_dirty": False,
        "snapshot_scope": "registered_core" if registered else "exploratory_only",
        "hmm_restart_selection_policy": config["learned_hmm"][
            "restart_selection_policy"
        ],
        "hmm_n_restarts": int(config["learned_hmm"]["n_restarts"]),
    }
    for key, value in bindings.items():
        conditions[key] = value
        comparisons[key] = value
    conditions.to_csv(paths["conditions"], index=False, lineterminator="\n")
    comparisons.to_csv(paths["comparisons"], index=False, lineterminator="\n")
    _write_report(paths["report"], conditions, comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/formal/exp14_ibl_multisession_neural.json"
    )
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()
    build_snapshot(
        Path(args.results_root), load_json_config(args.config), prefix=args.prefix
    )


if __name__ == "__main__":
    main()
