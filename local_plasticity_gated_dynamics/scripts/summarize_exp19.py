"""Publish a fail-closed Exp19 high-rank E/I belief-control snapshot.

The inferential unit for behavior is the independent seed.  The physical E/I,
coarse closure, and local normal-stability rows are registered-threshold audits,
not additional behavioral tests.  The reduced model is explicitly a
three-epoch mean-rate surrogate and is never labelled a full trajectory LDS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import load_json_config  # noqa: E402
from experiments.exp19_belief_ei_effective_dynamics import (  # noqa: E402
    ALL_CONDITION_SPECS,
    EI_CONDITION_SPECS,
    GATE_TIMING,
    _planned_conditions,
)

EXPERIMENT = "exp19_belief_ei_effective_dynamics"
INTACT = "md_combined_intact"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/formal/exp19_belief_ei_effective_dynamics.json"
DEFAULT_PREFIX = "exp19_belief_ei_effective_dynamics_formal"
TERMINAL_COMPLETE = {"complete", "complete_with_failures"}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ENVIRONMENT_PACKAGES = (
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "torch",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must be a non-empty JSONL object stream")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_sha256(environment: Mapping[str, Any]) -> str:
    """Validate Python 3.11 and hash the declared scientific software stack."""

    if not isinstance(environment, Mapping):
        raise ValueError("formal run environment must be a JSON object")
    python = environment.get("python")
    packages = environment.get("packages")
    if not isinstance(python, str) or re.match(r"^3\.11\.", python) is None:
        raise ValueError("formal run environment must use Python 3.11")
    if not isinstance(packages, Mapping):
        raise ValueError("formal run environment lacks package provenance")
    missing = [name for name in _ENVIRONMENT_PACKAGES if not packages.get(name)]
    if missing:
        raise ValueError(f"formal run environment lacks packages: {missing}")
    payload = {
        "python": python,
        "packages": {name: str(packages[name]) for name in _ENVIRONMENT_PACKAGES},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _analysis_provenance() -> dict[str, object]:
    """Bind the generated snapshot to clean Python 3.11 analysis code."""

    if sys.version_info[:2] != (3, 11):
        raise ValueError("formal snapshot analysis must use Python 3.11")
    repository = PROJECT_ROOT.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("formal snapshot lacks Git analysis provenance") from error
    if _COMMIT.fullmatch(commit) is None or status:
        raise ValueError("formal snapshot analysis requires a clean Git commit")
    return {
        "analysis_git_commit": commit,
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "analysis_python": sys.version,
    }


def _expected_run_config(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "seed": int(seed),
        **dict(config),
        "training_algorithm": "md_filtered_belief_frozen_high_rank_dale_ei",
        "used_autograd": False,
        "used_bptt": False,
        "parent_checkpoint": None,
        "recurrent_learning": False,
        "full_trajectory_lds": False,
    }


def _configs_match(observed: object, expected: Mapping[str, Any]) -> bool:
    """Compare the full contract while allowing checkout-root relocation."""

    if not isinstance(observed, Mapping):
        return False
    left, right = dict(observed), dict(expected)
    left_path = left.pop("config_path", None)
    right_path = right.pop("config_path", None)
    if left != right:
        return False
    if left_path is None or right_path is None:
        return left_path == right_path
    return Path(str(left_path)).name == Path(str(right_path)).name


def _latest_complete_attempt(
    results_root: Path, *, seed: int, expected_config: Mapping[str, Any]
) -> Path:
    seed_root = results_root / "runs" / EXPERIMENT / f"seed_{seed:04d}"
    matching_nonterminal = False
    for attempt in sorted(seed_root.glob("*"), reverse=True):
        config_path = attempt / "config.json"
        status_path = attempt / "status.json"
        if not config_path.is_file() or not _configs_match(
            _json(config_path), expected_config
        ):
            continue
        if not status_path.is_file():
            matching_nonterminal = True
            continue
        status = _json(status_path)
        if isinstance(status, Mapping) and status.get("status") in TERMINAL_COMPLETE:
            return attempt
        matching_nonterminal = True
    detail = " (only non-complete matching attempts exist)" if matching_nonterminal else ""
    raise FileNotFoundError(
        f"no complete {EXPERIMENT} attempt matches the registered config for "
        f"seed={seed}{detail}"
    )


def _planned_without_index(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError("planned_conditions.json must contain a list of objects")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        row = dict(item)
        if row.pop("condition_index", None) != index:
            raise ValueError("planned condition indexes must be contiguous and ordered")
        rows.append(row)
    return rows


def _require_digest(value: object, field: str) -> str:
    text = str(value)
    if _DIGEST.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _validate_success_contract(rows: Sequence[Mapping[str, Any]], seed: int) -> None:
    complete = [row for row in rows if row.get("status") == "complete"]
    if not complete:
        return
    by_condition = {str(row["condition"]): row for row in complete}
    common_fields = (
        "gate_checkpoint_id",
        "split_id",
        "random_tape_id",
        "shared_noise_id",
        "experiment_protocol_id",
        "planned_condition_grid_id",
    )
    for field in common_fields:
        values = {_require_digest(row.get(field), field) for row in complete}
        if len(values) != 1:
            raise ValueError(f"seed {seed} does not share {field} across complete rows")

    ei_names = {spec.condition for spec in EI_CONDITION_SPECS}
    ei_rows = [row for row in complete if str(row["condition"]) in ei_names]
    for row in ei_rows:
        for field in (
            "readout_checkpoint_id",
            "dynamics_checkpoint_id",
            "network_init_id",
            "gain_axis_id",
        ):
            _require_digest(row.get(field), field)
        if row.get("full_trajectory_lds") is not False:
            raise ValueError("Exp19 must remain outside full-trajectory LDS scope")
        if row.get("preprocessing_fit_train_only") is not True:
            raise ValueError("Exp19 preprocessing must be fit on training data only")
        if row.get("readout_reused_from_intact_train") is not True:
            raise ValueError("E/I comparison did not reuse the intact readout")
        if row.get("dynamics_reused_from_intact_train") is not True:
            raise ValueError("E/I comparison did not reuse intact dynamics")
        if row.get("gate_fit_accessed_true_context") is not False:
            raise ValueError("Exp19 gate fit accessed hidden context")
        if row.get("gate_test_accessed_true_context") is not False:
            raise ValueError("Exp19 gate test accessed hidden context")
        if (
            row.get("gate_timing") != GATE_TIMING
            or row.get("current_cue_accessed_for_same_trial") is not True
            or row.get("cue_available_before_receiver_control") is not True
            or row.get("receiver_received_cue_channels") is not False
        ):
            raise ValueError("Exp19 cue-epoch timing/capability contract failed")
    for field in (
        "readout_checkpoint_id",
        "dynamics_checkpoint_id",
        "network_init_id",
        "gain_axis_id",
    ):
        if len({_require_digest(row.get(field), field) for row in ei_rows}) > 1:
            raise ValueError(f"seed {seed} E/I cells disagree on {field}")

    ablations = [row for row in ei_rows if row["condition"] != INTACT]
    for row in ablations:
        if not all(
            row.get(field) is True
            for field in (
                "intervention_postfit",
                "intervention_reuses_intact_readout",
                "intervention_reuses_intact_receiver",
                "intervention_reuses_intact_gate_checkpoint",
                "intervention_reuses_intact_dynamics",
            )
        ):
            raise ValueError("Exp19 E/I ablation is not a fixed-checkpoint comparison")

    direct = by_condition.get("direct_evidence_mix")
    if direct is not None:
        _require_digest(direct.get("direct_baseline_checkpoint_id"), "direct checkpoint")
        if direct.get("direct_baseline_separate_from_ei") is not True:
            raise ValueError("direct evidence row is not marked as a separate baseline")


def _validate_attempt(
    attempt: Path,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = (
        "config.json",
        "environment.json",
        "manifest.json",
        "metrics.jsonl",
        "planned_conditions.json",
        "status.json",
    )
    missing = [name for name in required if not (attempt / name).is_file()]
    if missing:
        raise FileNotFoundError(f"seed {seed} attempt lacks artifacts: {missing}")
    expected_config = _expected_run_config(config, seed)
    if not _configs_match(_json(attempt / "config.json"), expected_config):
        raise ValueError(f"seed {seed} run config differs from the registered config")
    status = _json(attempt / "status.json")
    manifest = _json(attempt / "manifest.json")
    environment = _json(attempt / "environment.json")
    environment_sha256 = _environment_sha256(environment)
    if status.get("status") not in TERMINAL_COMPLETE:
        raise ValueError(f"seed {seed} attempt is not complete")
    if manifest.get("status") != status.get("status"):
        raise ValueError(f"seed {seed} manifest/status disagree")
    if (
        manifest.get("experiment") != EXPERIMENT
        or int(manifest.get("seed", -1)) != seed
        or manifest.get("profile") != "formal"
    ):
        raise ValueError(f"seed {seed} manifest violates the formal identity contract")
    git = environment.get("git")
    if (
        not isinstance(git, Mapping)
        or _COMMIT.fullmatch(str(git.get("commit", ""))) is None
        or git.get("dirty") is not False
    ):
        raise ValueError(f"seed {seed} formal attempt must have a clean Git receipt")

    expected_planned = _planned_conditions(dict(config))
    observed_planned = _planned_without_index(_json(attempt / "planned_conditions.json"))
    if observed_planned != expected_planned:
        raise ValueError(f"seed {seed} planned grid differs from the registered grid")
    rows = _records(attempt / "metrics.jsonl")
    if len(rows) != len(expected_planned):
        raise ValueError(f"seed {seed} must retain exactly one row per planned cell")
    expected_by_name = {str(item["condition"]): item for item in expected_planned}
    if {str(row.get("condition")) for row in rows} != set(expected_by_name):
        raise ValueError(f"seed {seed} observed condition grid is incomplete")
    if len({str(row.get("condition")) for row in rows}) != len(rows):
        raise ValueError(f"seed {seed} contains duplicate condition rows")
    run_id = str(manifest.get("run_id", ""))
    for row in rows:
        condition = str(row["condition"])
        if (
            row.get("run_id") != run_id
            or row.get("experiment") != EXPERIMENT
            or int(row.get("seed", -1)) != seed
            or row.get("status") not in {"complete", "failed", "invalid"}
        ):
            raise ValueError(f"seed {seed} metric identity/status is malformed")
        for field, value in expected_by_name[condition].items():
            if row.get(field) != value:
                raise ValueError(
                    f"seed {seed} condition {condition} differs on planned field {field}"
                )
    failures = sum(row["status"] == "failed" for row in rows)
    invalid = sum(row["status"] == "invalid" for row in rows)
    expected_status = "complete_with_failures" if failures or invalid else "complete"
    if (
        status.get("status") != expected_status
        or int(status.get("condition_failures", -1)) != failures
        or int(status.get("condition_invalid", -1)) != invalid
    ):
        raise ValueError(f"seed {seed} failure receipts disagree with raw rows")
    _validate_success_contract(rows, seed)
    provenance = {
        "seed": seed,
        "run_id": run_id,
        "attempt": str(attempt.resolve()),
        "run_status": status["status"],
        "git_commit": str(git["commit"]),
        "environment_sha256": environment_sha256,
        "config_sha256": _sha256(attempt / "config.json"),
        "metrics_sha256": _sha256(attempt / "metrics.jsonl"),
        "planned_conditions_sha256": _sha256(attempt / "planned_conditions.json"),
    }
    return rows, provenance


def collect_formal_runs(
    results_root: str | Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect the latest complete exact-config attempt for every registered seed."""

    if config.get("profile") != "formal":
        raise ValueError("Exp19 publication requires a formal config")
    seeds = [int(value) for value in config.get("seeds", [])]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Exp19 formal seeds must be unique and non-empty")
    thresholds = config.get("registered_claim_thresholds")
    required_thresholds = {
        "minimum_physical_rank_fraction",
        "minimum_operator_delta_frobenius_norm",
        "maximum_normalized_closure_mse",
        "maximum_basis_residual_fraction",
        "minimum_gate_identifiable_fraction",
        "minimum_normal_stability_eligible_fraction",
        "maximum_normal_local_decay_ratio",
        "maximum_normal_local_max_real_part",
    }
    if not isinstance(thresholds, Mapping) or not required_thresholds <= set(thresholds):
        raise ValueError("Exp19 config lacks the complete registered claim thresholds")

    all_rows: list[dict[str, Any]] = []
    receipts = []
    root = Path(results_root)
    for seed in seeds:
        expected = _expected_run_config(config, seed)
        attempt = _latest_complete_attempt(root, seed=seed, expected_config=expected)
        rows, receipt = _validate_attempt(attempt, config=config, seed=seed)
        for row in rows:
            all_rows.append({**row, **receipt})
        receipts.append(receipt)
    receipt_frame = pd.DataFrame(receipts)
    _validate_cross_seed_receipts(receipt_frame)
    return pd.DataFrame(all_rows), receipt_frame


def _validate_cross_seed_receipts(receipts: pd.DataFrame) -> None:
    """Refuse a publication snapshot assembled from different code states."""

    if receipts.empty or "git_commit" not in receipts:
        raise ValueError("Exp19 cross-seed Git receipts are incomplete")
    commits = receipts["git_commit"].astype(str)
    if commits.nunique(dropna=False) != 1:
        raise ValueError("Exp19 formal seeds use mixed Git commits")
    if "environment_sha256" not in receipts:
        raise ValueError("Exp19 cross-seed environment receipts are incomplete")
    environments = receipts["environment_sha256"].astype(str)
    if environments.nunique(dropna=False) != 1:
        raise ValueError("Exp19 formal seeds use mixed software environments")


def _bootstrap_interval(
    values: np.ndarray, *, n_bootstrap: int, seed: int
) -> tuple[float, float]:
    if n_bootstrap < 100:
        raise ValueError("paired seed bootstrap requires at least 100 draws")
    rng = np.random.default_rng(seed)
    draws = np.mean(
        rng.choice(values, size=(n_bootstrap, len(values)), replace=True), axis=1
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _wilcoxon_p(values: np.ndarray) -> float:
    if not np.any(values != 0.0):
        return 1.0
    return float(
        wilcoxon(
            values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


def _holm(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    running = 0.0
    for rank, index in enumerate(finite[np.argsort(p[finite])]):
        running = max(running, min(1.0, (len(finite) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def _numeric(frame: pd.DataFrame, field: str) -> np.ndarray:
    if field not in frame:
        return np.empty(0, dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)


def summarize_formal_runs(
    raw: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    """Create seed-primary inference plus registered mechanism threshold audits."""

    planned_seeds = tuple(int(value) for value in config["seeds"])
    comparators = [spec.condition for spec in ALL_CONDITION_SPECS if spec.condition != INTACT]
    complete = raw.loc[raw["status"] == "complete"].copy()
    behavior_rows: list[dict[str, Any]] = []
    raw_p = []
    for index, comparator in enumerate(comparators):
        paired = complete.loc[
            complete["condition"].isin([INTACT, comparator]),
            ["seed", "condition", "behavior_balanced_accuracy"],
        ].copy()
        paired["behavior_balanced_accuracy"] = pd.to_numeric(
            paired["behavior_balanced_accuracy"], errors="coerce"
        )
        pivot = paired.pivot(index="seed", columns="condition", values="behavior_balanced_accuracy")
        if {INTACT, comparator} <= set(pivot):
            difference = (pivot[INTACT] - pivot[comparator]).dropna().to_numpy(dtype=float)
        else:
            difference = np.empty(0, dtype=float)
        if difference.size:
            estimate = float(np.mean(difference))
            low, high = _bootstrap_interval(
                difference,
                n_bootstrap=n_bootstrap,
                seed=190_019 + 104729 * index,
            )
            p_value = _wilcoxon_p(difference)
        else:
            estimate = low = high = p_value = float("nan")
        raw_p.append(p_value)
        behavior_rows.append(
            {
                "experiment": EXPERIMENT,
                "proposition": (
                    "separate_train_only_baseline"
                    if comparator == "direct_evidence_mix"
                    else "heldout_behavior_fixed_checkpoint"
                ),
                "comparison": f"{INTACT}_vs_{comparator}",
                "effect_definition": "intact_minus_comparator_balanced_accuracy",
                "inference_unit": "seed",
                "multiplicity_family": "exp19_paired_behavior_wilcoxon",
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "p_value": p_value,
                "holm_adjusted_p": float("nan"),
                "n_complete": int(difference.size),
                "n_planned": len(planned_seeds),
                "threshold": 0.0,
                "conclusion": "inconclusive",
                "claim_scope": (
                    "heldout task behavior; separately train-fitted scalar ridge baseline"
                    if comparator == "direct_evidence_mix"
                    else (
                        "architectural ablation with fixed intact checkpoints; not "
                        "input-charge-matched for pathway-specific causal attribution"
                        if comparator in {"md_population_only", "md_disconnected"}
                        else "heldout task behavior; fixed intact readout/dynamics checkpoints"
                    )
                ),
            }
        )
    adjusted = _holm(raw_p)
    for row, p_holm in zip(behavior_rows, adjusted, strict=True):
        row["holm_adjusted_p"] = float(p_holm)
        full = row["n_complete"] == row["n_planned"]
        if full and row["ci_low"] > 0.0 and p_holm < 0.05:
            row["conclusion"] = "support"
        elif full and row["ci_high"] < 0.0 and p_holm < 0.05:
            row["conclusion"] = "oppose"

    thresholds = dict(config["registered_claim_thresholds"])
    intact = complete.loc[complete["condition"] == INTACT].copy()
    full_intact = set(intact["seed"].astype(int)) == set(planned_seeds)

    def threshold_row(
        proposition: str,
        passed: bool,
        *,
        estimate: float,
        threshold: str,
        scope: str,
    ) -> dict[str, Any]:
        return {
            "experiment": EXPERIMENT,
            "proposition": proposition,
            "comparison": "registered_threshold_audit",
            "effect_definition": scope,
            "inference_unit": "seed",
            "multiplicity_family": "none_registered_threshold",
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "holm_adjusted_p": float("nan"),
            "n_complete": int(intact["seed"].nunique()),
            "n_planned": len(planned_seeds),
            "threshold": threshold,
            "conclusion": (
                "support" if full_intact and passed else "oppose" if full_intact else "inconclusive"
            ),
            "claim_scope": scope,
        }

    rank = _numeric(intact, "physical_rank_fraction")
    dale = intact.get("physical_dale_valid", pd.Series(dtype=bool)).eq(True).to_numpy()
    rank_threshold = float(thresholds["minimum_physical_rank_fraction"])
    physical_pass = bool(
        len(rank) == len(planned_seeds)
        and np.isfinite(rank).all()
        and np.all(rank >= rank_threshold)
        and len(dale) == len(planned_seeds)
        and np.all(dale)
    )
    mechanism_rows = [
        threshold_row(
            "high_rank_dale_physical_background",
            physical_pass,
            estimate=float(np.nanmin(rank)) if rank.size else float("nan"),
            threshold=f"rank_fraction>={rank_threshold}; Dale_valid=true for every seed",
            scope="physical recurrent matrix only; descriptive mechanism audit",
        )
    ]

    identifiable = intact.get(
        "moment_anchor_identifiable", pd.Series(dtype=bool)
    ).eq(True).to_numpy()
    identifiable_fraction = (
        float(np.mean(identifiable)) if len(identifiable) else float("nan")
    )
    min_identifiable = float(thresholds["minimum_gate_identifiable_fraction"])
    mechanism_rows.append(
        threshold_row(
            "filtered_gate_moment_anchor_identifiability",
            bool(np.isfinite(identifiable_fraction) and identifiable_fraction >= min_identifiable),
            estimate=identifiable_fraction,
            threshold=f"identifiable_seed_fraction>={min_identifiable}",
            scope="training-cue moment-anchor identifiability; no hidden-state labels used",
        )
    )

    declared = _numeric(intact, "declared_scalar_control_dimension")
    effective = _numeric(intact, "combined_effective_control_dimension")
    empirical = _numeric(intact, "empirical_combined_control_trajectory_rank")
    operator = _numeric(intact, "operator_control_dimension")
    operator_delta = _numeric(intact, "operator_delta_frobenius_norm")
    min_operator_delta = float(
        thresholds["minimum_operator_delta_frobenius_norm"]
    )
    control_pass = bool(
        len(declared) == len(planned_seeds)
        and np.all(declared == 1)
        and np.all(effective == 1)
        and np.isfinite(empirical).all()
        and np.all(empirical == 1)
        and np.isfinite(operator).all()
        and np.all(operator == 1)
        and np.isfinite(operator_delta).all()
        and np.all(operator_delta >= min_operator_delta)
    )
    mechanism_rows.append(
        threshold_row(
            "scalar_effective_control",
            control_pass,
            estimate=(
                float(np.nanmin(operator_delta))
                if operator_delta.size
                else float("nan")
            ),
            threshold=(
                "declared=effective=empirical=operator dimension=1; "
                f"operator_delta_frobenius_norm>={min_operator_delta} every seed"
            ),
            scope=(
                "nonzero scalar control/operator family; behavior benefit is a "
                "separate paired-seed claim"
            ),
        )
    )

    closure = _numeric(intact, "heldout_normalized_closure_mse")
    residual = _numeric(intact, "heldout_basis_residual_fraction")
    max_closure = float(thresholds["maximum_normalized_closure_mse"])
    max_residual = float(thresholds["maximum_basis_residual_fraction"])
    closure_pass = bool(
        len(closure) == len(planned_seeds)
        and np.isfinite(closure).all()
        and np.isfinite(residual).all()
        and np.all(closure < max_closure)
        and np.all(residual < max_residual)
    )
    mechanism_rows.append(
        threshold_row(
            "coarse_shared_dynamics_closure",
            closure_pass,
            estimate=float(np.nanmax(closure)) if closure.size else float("nan"),
            threshold=f"closure<{max_closure}; basis_residual<{max_residual} for every seed",
            scope=(
                "three-epoch mean-rate soft-operator surrogate; explicitly not a full LDS"
            ),
        )
    )

    eligible = intact.get(
        "normal_stability_eligible", pd.Series(dtype=bool)
    ).eq(True).to_numpy()
    eligible_fraction = float(np.mean(eligible)) if len(eligible) else float("nan")
    decay = _numeric(intact.loc[intact.get("normal_stability_eligible", False).eq(True)], "normal_local_decay_ratio") if "normal_stability_eligible" in intact else np.empty(0)
    max_real = _numeric(intact.loc[intact.get("normal_stability_eligible", False).eq(True)], "normal_local_max_real_part") if "normal_stability_eligible" in intact else np.empty(0)
    min_eligible = float(thresholds["minimum_normal_stability_eligible_fraction"])
    max_decay = float(thresholds["maximum_normal_local_decay_ratio"])
    max_real_threshold = float(thresholds["maximum_normal_local_max_real_part"])
    normal_pass = bool(
        np.isfinite(eligible_fraction)
        and eligible_fraction >= min_eligible
        and decay.size > 0
        and np.isfinite(decay).all()
        and np.isfinite(max_real).all()
        and np.all(decay < max_decay)
        and np.all(max_real < max_real_threshold)
    )
    mechanism_rows.append(
        threshold_row(
            "local_normal_perturbation_decay",
            normal_pass,
            estimate=eligible_fraction,
            threshold=(
                f"eligible_fraction>={min_eligible}; decay<{max_decay}; "
                f"max_real_part<{max_real_threshold}"
            ),
            scope="local linearization at intact training mean state only",
        )
    )
    return pd.DataFrame([*behavior_rows, *mechanism_rows])


def _csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(
            lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
            if isinstance(value, (dict, list, tuple))
            else value
        )
    return safe


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render Markdown without pandas' optional tabulate dependency."""

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join((header, rule, *body))


def _plot(summary: pd.DataFrame, raw: pd.DataFrame, png: Path, pdf: Path) -> None:
    behavior = summary.loc[
        summary["multiplicity_family"] == "exp19_paired_behavior_wilcoxon"
    ].copy()
    audits = summary.loc[summary["multiplicity_family"] == "none_registered_threshold"]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.5),
        gridspec_kw={"width_ratios": (1.05, 1.0)},
        constrained_layout=True,
    )
    if len(behavior):
        y = np.arange(len(behavior))
        estimate = behavior["estimate"].to_numpy(dtype=float)
        low = behavior["ci_low"].to_numpy(dtype=float)
        high = behavior["ci_high"].to_numpy(dtype=float)
        labels = behavior["comparison"].str.replace(f"{INTACT}_vs_", "")
        labels = labels.mask(
            labels.eq("direct_evidence_mix"),
            "direct_evidence_mix (separate fit)",
        )
        for index, (center, lower, upper, proposition) in enumerate(
            zip(estimate, low, high, behavior["proposition"], strict=True)
        ):
            separate = proposition == "separate_train_only_baseline"
            axes[0].errorbar(
                center,
                y[index],
                xerr=np.asarray([[center - lower], [upper - center]]),
                fmt="s" if separate else "o",
                color="#b2182b" if separate else "#2166ac",
                capsize=3,
            )
        axes[0].set_yticks(y, labels)
        axes[0].axvline(0.0, color="0.25", linewidth=0.8, linestyle="--")
        axes[0].set_xlabel("Intact - comparator balanced accuracy (95% CI)")
    else:
        axes[0].text(0.5, 0.5, "No complete paired behavior rows", ha="center")
    axes[0].set_title("a  Held-out behavior (seed is the unit)", loc="left")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].axis("off")
    audit_text = [
        f"{row.proposition}: {str(row.conclusion).upper()}\n"
        + textwrap.fill(
            str(row.threshold),
            width=52,
            initial_indent="  ",
            subsequent_indent="  ",
        )
        for row in audits.itertuples()
    ]
    failure_count = int((raw["status"] != "complete").sum())
    axes[1].text(
        0.0,
        1.0,
        "b  Registered mechanism boundaries\n\n"
        + "\n\n".join(audit_text)
        + f"\n\nRetained failed/invalid cells: {failure_count}",
        va="top",
        fontsize=8.5,
    )
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def publish_snapshot(
    results_root: str | Path,
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    prefix: str = DEFAULT_PREFIX,
    n_bootstrap: int = 5000,
) -> dict[str, Path]:
    analysis = _analysis_provenance()
    raw, receipts = collect_formal_runs(results_root, config)
    summary = summarize_formal_runs(raw, config, n_bootstrap=n_bootstrap)
    raw = raw.assign(**analysis)
    summary = summary.assign(
        raw_run_git_commit=receipts["git_commit"].iloc[0], **analysis
    )
    receipts = receipts.assign(**analysis)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": target / f"{prefix}_raw.csv.gz",
        "summary": target / f"{prefix}_summary.csv",
        "receipts": target / f"{prefix}_run_receipts.csv",
        "report": target / f"{prefix}_report.md",
        "png": target / f"{prefix}.png",
        "pdf": target / f"{prefix}.pdf",
    }
    _csv_safe(raw).to_csv(paths["raw"], index=False, compression="gzip")
    _csv_safe(summary).to_csv(paths["summary"], index=False)
    receipts.to_csv(paths["receipts"], index=False)
    failures = int((raw["status"] != "complete").sum())
    report = [
        "# Exp19: belief-gated high-rank E/I effective dynamics",
        "",
        "This snapshot uses seed-level paired inference. Every registered cell, including failures and invalid cells, is retained in the scoped raw table.",
        "",
        "The physical recurrent matrix audit, scalar-control audit, coarse closure audit, and local normal audit are distinct claims. The reduced model is a three-epoch mean-rate soft-operator surrogate, not a full trajectory LDS.",
        "",
        f"- Registered seeds: {len(config['seeds'])}",
        f"- Retained failed/invalid cells: {failures}",
        "- Behavioral multiplicity: paired Wilcoxon tests, one Holm family; 95% seed bootstrap CIs.",
        "- Mechanism classification: preregistered thresholds from the formal config; no neuron or time-bin pseudo-replication.",
        "- Population-only and disconnected are architectural, not input-charge-matched pathway controls; primary clamp/delay/shuffle belief interventions preserve the complementary pathway coefficient sum.",
        f"- Raw-run commit: `{receipts['git_commit'].iloc[0]}`; analysis commit: `{analysis['analysis_git_commit']}`.",
        "",
        "## Conclusions",
        "",
        _markdown_table(
            summary[
                [
                    "proposition",
                    "comparison",
                    "estimate",
                    "n_complete",
                    "conclusion",
                    "claim_scope",
                ]
            ]
        ),
        "",
        "## Provenance",
        "",
        _markdown_table(receipts),
        "",
    ]
    paths["report"].write_text("\n".join(report), encoding="utf-8")
    _plot(summary, raw, paths["png"], paths["pdf"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    config = load_json_config(args.config)
    publish_snapshot(
        args.results_root,
        config,
        output_dir=args.output_dir,
        prefix=args.prefix,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
