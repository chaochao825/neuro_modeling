"""Publish a fail-closed real-IBL Exp20 belief-dynamics snapshot.

The registered neural endpoint is an animal/session-primary, teacher-forced
one-step conditional Poisson score.  It is not a full latent LDS and cannot
identify an E/I or Dale mechanism.  ``probabilityLeft`` is accepted only as a
hash-bound whole-block split and post-fit evaluation sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import load_json_config  # noqa: E402
from experiments.exp20_ibl_md_belief_dynamics import (  # noqa: E402
    INTERVENTIONS,
    MODEL_CONDITIONS,
)


EXPERIMENT = "exp20_ibl_md_belief_dynamics"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/formal/exp20_ibl_md_belief_dynamics.json"
DEFAULT_PREFIX = "exp20_ibl_md_belief_dynamics_formal"
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
_CONTRASTS = (
    "md_shared_vs_common",
    "md_shared_vs_hmm_shared",
    "md_shared_vs_md_clamp",
    "md_shared_vs_md_delay_1",
    "md_shared_vs_md_delay_5",
    "md_shared_vs_md_shuffle",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, Any]]:
    result = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not result or not all(isinstance(item, dict) for item in result):
        raise ValueError(f"{path} must contain a non-empty JSONL object stream")
    return result


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
        "training_algorithm": "md_predictive_prior_shared_count_dynamics",
        "used_autograd": False,
        "used_bptt": False,
        "recurrent_ei_mechanism_claim_eligible": False,
    }


def _configs_match(observed: object, expected: Mapping[str, Any]) -> bool:
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


def _planned_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError("planned_conditions.json must contain an object list")
    result = []
    for index, item in enumerate(value):
        row = dict(item)
        if row.pop("condition_index", None) != index:
            raise ValueError("planned condition indexes must be contiguous")
        result.append(row)
    return result


def _latest_attempt(
    results_root: Path,
    *,
    seed: int,
    expected_config: Mapping[str, Any],
) -> Path:
    root = results_root / "runs" / EXPERIMENT / f"seed_{seed:04d}"
    nonterminal = False
    for attempt in sorted(root.glob("*"), reverse=True):
        config_path = attempt / "config.json"
        status_path = attempt / "status.json"
        if not config_path.is_file() or not _configs_match(
            _json(config_path), expected_config
        ):
            continue
        if not status_path.is_file():
            nonterminal = True
            continue
        status = _json(status_path)
        if isinstance(status, Mapping) and status.get("status") in {
            "complete",
            "complete_with_failures",
        }:
            return attempt
        nonterminal = True
    suffix = " (only nonterminal matches exist)" if nonterminal else ""
    raise FileNotFoundError(f"no complete exact-config Exp20 seed={seed} run{suffix}")


def _validate_outer_contract(
    outer: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> None:
    planned_sessions = int(config["planned_sessions"])
    planned_animals = int(config["planned_animals"])
    complete = [item for item in outer if item.get("status") == "complete"]
    if not complete:
        return
    if len(complete) != planned_sessions * len(MODEL_CONDITIONS):
        raise ValueError("a partially complete Exp20 outer grid cannot be published")
    sessions = {str(item["session_id"]) for item in complete}
    animals = {str(item["animal_id"]) for item in complete}
    if len(sessions) != planned_sessions or len(animals) < planned_animals:
        raise ValueError("Exp20 complete cohort size violates the formal contract")
    if {
        (str(item["session_id"]), str(item["condition"])) for item in complete
    } != {(session, condition) for session in sessions for condition in MODEL_CONDITIONS}:
        raise ValueError("Exp20 outer session/condition grid is incomplete")

    required_false = (
        "gate_received_probability_left",
        "belief_uses_current_trial_stimulus",
        "belief_uses_future_trials",
        "belief_accessed_true_context",
        "full_latent_lds",
    )
    for item in complete:
        if any(item.get(field) is not False for field in required_false):
            raise ValueError("Exp20 timing/truth/full-LDS capability contract failed")
        if (
            item.get("statistics_unit") != "animal_with_session_nested"
            or item.get("preprocessing_fit_train_only") is not True
            or item.get("split_unit") != "contiguous_true_probabilityLeft_block"
            or item.get("probability_left_access_scope")
            != "whole_block_split_and_postfit_evaluation_only"
            or item.get("likelihood_kind") != "one_step_conditional_poisson"
        ):
            raise ValueError("Exp20 statistical/preprocessing/model scope is malformed")
        provenance = item.get("truth_sidecar_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Exp20 row lacks a hash-bound truth sidecar receipt")
        if (
            provenance.get("access_scope")
            != "whole_block_split_and_postfit_evaluation_only"
            or provenance.get("eligible_for_whole_block_split") is not True
            or provenance.get("eligible_for_postfit_evaluation") is not True
            or provenance.get("eligible_for_gate_input") is not False
            or provenance.get("eligible_for_model_input") is not False
            or _DIGEST.fullmatch(str(provenance.get("cohort_manifest_sha256", "")))
            is None
        ):
            raise ValueError("Exp20 truth sidecar exceeded its evaluation capability")

    by_session: dict[str, dict[str, Mapping[str, Any]]] = {}
    for item in complete:
        by_session.setdefault(str(item["session_id"]), {})[str(item["condition"])] = item
    for condition_rows in by_session.values():
        fixed = [condition_rows[name] for name in INTERVENTIONS]
        digest_fields = (
            "fit_fingerprint",
            "belief_checkpoint_sha256",
            "source_belief_trajectory_sha256",
            "evaluated_heldout_belief_sha256",
        )
        if any(
            _DIGEST.fullmatch(str(item.get(field, ""))) is None
            for item in fixed
            for field in digest_fields
        ):
            raise ValueError("Exp20 fixed intervention provenance is incomplete")
        if len({str(item["fit_fingerprint"]) for item in fixed}) != 1:
            raise ValueError("Exp20 belief interventions changed the neural checkpoint")
        if len({str(item["belief_checkpoint_sha256"]) for item in fixed}) != 1:
            raise ValueError("Exp20 belief interventions changed the gate checkpoint")
        if len({str(item["source_belief_trajectory_sha256"]) for item in fixed}) != 1:
            raise ValueError("Exp20 belief interventions changed the source trajectory")
        for item in fixed[1:]:
            if (
                item.get("belief_intervention_postfit") is not True
                or item.get("all_model_parameters_frozen_for_intervention") is not True
            ):
                raise ValueError("Exp20 intervention is not a frozen post-fit branch")


def collect_formal_run(
    results_root: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect and validate the unique registered formal Exp20 run."""

    if config.get("profile") != "formal" or list(config.get("seeds", [])) != [0]:
        raise ValueError("Exp20 publication requires the registered formal seed=0 config")
    seed = 0
    root = Path(results_root)
    attempt = _latest_attempt(
        root,
        seed=seed,
        expected_config=_expected_run_config(config, seed),
    )
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
        raise FileNotFoundError(f"Exp20 run lacks required artifacts: {missing}")
    if not _configs_match(
        _json(attempt / "config.json"), _expected_run_config(config, seed)
    ):
        raise ValueError("Exp20 run config differs from the formal registration")
    status = _json(attempt / "status.json")
    manifest = _json(attempt / "manifest.json")
    environment = _json(attempt / "environment.json")
    environment_sha256 = _environment_sha256(environment)
    git = environment.get("git")
    if (
        not isinstance(git, Mapping)
        or _COMMIT.fullmatch(str(git.get("commit", ""))) is None
        or git.get("dirty") is not False
    ):
        raise ValueError("Exp20 formal snapshot requires a clean Git receipt")
    if (
        manifest.get("experiment") != EXPERIMENT
        or manifest.get("profile") != "formal"
        or int(manifest.get("seed", -1)) != seed
        or manifest.get("status") != status.get("status")
    ):
        raise ValueError("Exp20 manifest/status identity is invalid")

    planned = _planned_rows(_json(attempt / "planned_conditions.json"))
    expected_count = int(config["planned_sessions"]) * len(MODEL_CONDITIONS)
    session_ids = {str(item.get("session_id")) for item in planned}
    if (
        len(planned) != expected_count
        or len(session_ids) != int(config["planned_sessions"])
        or {str(item.get("condition")) for item in planned} != set(MODEL_CONDITIONS)
        or any(item.get("stage") != "outer_test" for item in planned)
        or len(
            {(str(item.get("session_id")), str(item.get("condition"))) for item in planned}
        )
        != expected_count
    ):
        raise ValueError("Exp20 planned outer grid violates the formal contract")

    rows = _records(attempt / "metrics.jsonl")
    run_id = str(manifest.get("run_id", ""))
    if any(
        item.get("run_id") != run_id
        or item.get("experiment") != EXPERIMENT
        or int(item.get("seed", -1)) != seed
        for item in rows
    ):
        raise ValueError("Exp20 raw record identity is inconsistent")
    outer = [item for item in rows if item.get("stage") == "outer_test"]
    planned_grid = {
        (str(item["session_id"]), str(item["condition"])) for item in planned
    }
    observed_grid = {
        (str(item.get("session_id")), str(item.get("condition"))) for item in outer
    }
    if len(outer) != expected_count or observed_grid != planned_grid:
        raise ValueError("Exp20 observed outer grid differs from planned conditions")
    failures = sum(item.get("status") == "failed" for item in outer)
    invalid = sum(item.get("status") == "invalid" for item in outer)
    expected_status = "complete_with_failures" if failures or invalid else "complete"
    if (
        status.get("status") != expected_status
        or int(status.get("condition_failures", -1)) != failures
        or int(status.get("condition_invalid", -1)) != invalid
    ):
        raise ValueError("Exp20 failure counts disagree with the outer grid")
    _validate_outer_contract(outer, config=config)

    complete_outer = all(item.get("status") == "complete" for item in outer)
    contrasts = [
        item for item in rows if item.get("stage") == "animal_session_belief_contrast"
    ]
    summaries = [item for item in rows if item.get("stage") == "cohort_summary"]
    if complete_outer:
        if (
            len(contrasts) != len(_CONTRASTS)
            or {str(item.get("comparison")) for item in contrasts} != set(_CONTRASTS)
            or len(summaries) != 1
        ):
            raise ValueError("Exp20 complete run lacks registered cohort inference")
        for item in contrasts:
            if (
                item.get("inference_unit") != "animal_with_session_nested"
                or int(item.get("n_sessions", -1)) != int(config["planned_sessions"])
                or int(item.get("n_animals", -1)) < int(config["planned_animals"])
                or item.get("conclusion") not in {"support", "oppose", "inconclusive"}
            ):
                raise ValueError("Exp20 stored belief contrast is malformed")
        summary = summaries[0]
        comparison = summary.get("comparison")
        if (
            not isinstance(comparison, Mapping)
            or summary.get("core_conclusion")
            not in {"support", "oppose", "inconclusive"}
            or summary.get("truth_used_by_gate_or_model") is not False
            or summary.get("full_latent_lds") is not False
        ):
            raise ValueError("Exp20 common/shared/full cohort summary is malformed")

    receipt = {
        "seed": seed,
        "run_id": run_id,
        "attempt": str(attempt.resolve()),
        "run_status": str(status["status"]),
        "git_commit": str(git["commit"]),
        "environment_sha256": environment_sha256,
        "config_sha256": _sha256(attempt / "config.json"),
        "metrics_sha256": _sha256(attempt / "metrics.jsonl"),
        "planned_conditions_sha256": _sha256(attempt / "planned_conditions.json"),
    }
    enriched = [{**item, **receipt} for item in rows]
    return pd.DataFrame(enriched), pd.DataFrame([receipt])


def _finite_mean_interval(values: Sequence[float]) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(20_020)
    draws = np.mean(rng.choice(array, size=(5000, len(array))), axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(np.mean(array)), float(low), float(high), int(len(array))


def summarize_formal_run(raw: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Use only stored animal/session inference; add one descriptive timing row."""

    contrast = raw.loc[raw.get("stage").eq("animal_session_belief_contrast")].copy()
    rows: list[dict[str, Any]] = []
    for item in contrast.to_dict("records"):
        rows.append(
            {
                "experiment": EXPERIMENT,
                "proposition": "belief_condition_neural_prediction",
                "comparison": item["comparison"],
                "effect_definition": item["effect_definition"],
                "inference_unit": item["inference_unit"],
                "multiplicity_family": "Holm(exp20_belief_condition_family)",
                "estimate": item["estimate"],
                "ci_low": item["ci_low"],
                "ci_high": item["ci_high"],
                "p_value": item["bootstrap_p_two_sided"],
                "holm_adjusted_p": item["holm_adjusted_p"],
                "n_sessions": item["n_sessions"],
                "n_animals": item["n_animals"],
                "conclusion": item["conclusion"],
                "claim_scope": "teacher-forced one-step conditional Poisson prediction",
            }
        )

    cohort = raw.loc[raw.get("stage").eq("cohort_summary")]
    if len(cohort) == 1:
        item = cohort.iloc[0]
        comparison = item["comparison"]
        shared = comparison["shared_vs_common"]
        rows.append(
            {
                "experiment": EXPERIMENT,
                "proposition": "shared_basis_joint_registered_claim",
                "comparison": "md_shared_vs_common_and_full",
                "effect_definition": "common_minus_shared_nll_with_90pct_full_gain_retention",
                "inference_unit": "animal_with_session_nested",
                "multiplicity_family": "Holm(exp20_common_shared_full_family)",
                "estimate": shared["estimate"],
                "ci_low": shared["ci_low"],
                "ci_high": shared["ci_high"],
                "p_value": shared["bootstrap_p_two_sided"],
                "holm_adjusted_p": shared["holm_adjusted_p"],
                "n_sessions": shared["n_sessions"],
                "n_animals": shared["n_animals"],
                "conclusion": item["core_conclusion"],
                "claim_scope": (
                    "joint shared-vs-common, full gain, retention, and parameter-count gate"
                ),
            }
        )

    outer = raw.loc[
        raw.get("stage").eq("outer_test") & raw.get("condition").eq("md_shared")
    ].copy()
    complete = outer.loc[outer.get("status").eq("complete")]
    timing = pd.to_numeric(
        complete.get("belief_minus_behavior_switch_latency_trials"), errors="coerce"
    ).to_numpy(dtype=float)
    estimate, low, high, n_timing = _finite_mean_interval(timing)
    rows.append(
        {
            "experiment": EXPERIMENT,
            "proposition": "belief_vs_behavior_bias_switch_timing_descriptive",
            "comparison": "md_belief_minus_causal_choice_history_bias_latency",
            "effect_definition": "negative_values_mean_belief_switches_first",
            "inference_unit": "animal_session",
            "multiplicity_family": "descriptive_no_registered_test",
            "estimate": estimate,
            "ci_low": low,
            "ci_high": high,
            "p_value": float("nan"),
            "holm_adjusted_p": float("nan"),
            "n_sessions": n_timing,
            "n_animals": n_timing,
            "conclusion": "inconclusive",
            "claim_scope": (
                "descriptive causal EWMA choice-bias proxy; not a neural-latent lead claim"
            ),
        }
    )

    expected_outer = int(config["planned_sessions"]) * len(MODEL_CONDITIONS)
    leakage_pass = bool(
        len(raw.loc[raw.get("stage").eq("outer_test")]) == expected_outer
        and len(complete) == int(config["planned_sessions"])
        and complete.get("gate_received_probability_left").eq(False).all()
        and complete.get("belief_uses_current_trial_stimulus").eq(False).all()
        and complete.get("belief_uses_future_trials").eq(False).all()
        and complete.get("belief_accessed_true_context").eq(False).all()
    )
    rows.append(
        {
            "experiment": EXPERIMENT,
            "proposition": "past_only_truth_capability_contract",
            "comparison": "registered_threshold_audit",
            "effect_definition": "probabilityLeft excluded from gate/model; current/future cue excluded",
            "inference_unit": "session_contract",
            "multiplicity_family": "none_contract_audit",
            "estimate": float(leakage_pass),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "holm_adjusted_p": float("nan"),
            "n_sessions": int(config["planned_sessions"]) if leakage_pass else len(complete),
            "n_animals": int(config["planned_animals"]) if leakage_pass else 0,
            "conclusion": "support" if leakage_pass else "inconclusive",
            "claim_scope": "data/timing capability audit only",
        }
    )
    return pd.DataFrame(rows)


def _csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column] = result[column].map(
            lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
            if isinstance(value, (dict, list, tuple))
            else value
        )
    return result


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


def _plot(raw: pd.DataFrame, summary: pd.DataFrame, *, png: Path, pdf: Path) -> None:
    outer = raw.loc[
        raw.get("stage").eq("outer_test") & raw.get("status").eq("complete")
    ].copy()
    figure, axes = plt.subplots(
        2, 2, figsize=(12.0, 8.0), constrained_layout=True
    )
    order = list(MODEL_CONDITIONS)
    nll = [
        pd.to_numeric(
            outer.loc[outer["condition"].eq(condition), "nll_per_count"],
            errors="coerce",
        ).mean()
        for condition in order
    ]
    axes[0, 0].bar(np.arange(len(order)), nll, color="#496A81")
    axes[0, 0].set_xticks(np.arange(len(order)), order, rotation=40, ha="right")
    axes[0, 0].set_ylabel("NLL / count (lower is better)")
    axes[0, 0].set_title("a  Real IBL held-out prediction", loc="left")

    effects = summary.loc[
        summary["proposition"].eq("belief_condition_neural_prediction")
    ].copy()
    y = np.arange(len(effects))
    estimate = pd.to_numeric(effects["estimate"], errors="coerce").to_numpy()
    low = pd.to_numeric(effects["ci_low"], errors="coerce").to_numpy()
    high = pd.to_numeric(effects["ci_high"], errors="coerce").to_numpy()
    axes[0, 1].errorbar(
        estimate,
        y,
        xerr=np.vstack((estimate - low, high - estimate)),
        fmt="o",
        color="#B65C45",
        capsize=3,
    )
    axes[0, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_yticks(y, effects["comparison"])
    axes[0, 1].set_xlabel("comparator NLL - MD shared NLL")
    axes[0, 1].set_title("b  Animal/session nested contrasts", loc="left")

    gate_conditions = [
        "md_shared",
        "hmm_shared",
        "md_clamp",
        "md_delay_1",
        "md_delay_5",
        "md_shuffle",
    ]
    context = [
        pd.to_numeric(
            outer.loc[outer["condition"].eq(condition), "context_nll"],
            errors="coerce",
        ).mean()
        for condition in gate_conditions
    ]
    axes[1, 0].bar(np.arange(len(gate_conditions)), context, color="#5A8F62")
    axes[1, 0].set_xticks(
        np.arange(len(gate_conditions)), gate_conditions, rotation=40, ha="right"
    )
    axes[1, 0].set_ylabel("Context NLL")
    axes[1, 0].set_title("c  Past-only belief evaluation", loc="left")

    model_conditions = ["common", "md_shared", "md_full"]
    parameters = [
        pd.to_numeric(
            outer.loc[outer["condition"].eq(condition), "parameter_count"],
            errors="coerce",
        ).mean()
        for condition in model_conditions
    ]
    axes[1, 1].bar(np.arange(3), parameters, color="#7A6A9A")
    axes[1, 1].set_xticks(np.arange(3), model_conditions)
    axes[1, 1].set_ylabel("Parameters")
    axes[1, 1].set_title("d  Shared vs full complexity", loc="left")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
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
    output_dir: str | Path | None = None,
    prefix: str = DEFAULT_PREFIX,
) -> Mapping[str, Path]:
    root = Path(results_root)
    target = root if output_dir is None else Path(output_dir)
    analysis = _analysis_provenance()
    raw, receipts = collect_formal_run(root, config)
    summary = summarize_formal_run(raw, config)
    raw = raw.assign(**analysis)
    summary = summary.assign(
        raw_run_git_commit=receipts["git_commit"].iloc[0], **analysis
    )
    receipts = receipts.assign(**analysis)
    paths = {
        "raw": target / f"{prefix}_raw.csv.gz",
        "summary": target / f"{prefix}_summary.csv",
        "receipts": target / f"{prefix}_run_manifest.csv",
        "report": target / f"{prefix}_report.md",
        "png": target / f"{prefix}.png",
        "pdf": target / f"{prefix}.pdf",
    }
    target.mkdir(parents=True, exist_ok=True)
    _csv_safe(raw).to_csv(paths["raw"], index=False, compression="gzip")
    _csv_safe(summary).to_csv(paths["summary"], index=False)
    receipts.to_csv(paths["receipts"], index=False)
    _plot(raw, summary, png=paths["png"], pdf=paths["pdf"])
    report = [
        "# Exp20 real IBL belief-gated dynamics",
        "",
        "- Data: hash-bound frozen compact IBL neural/behavior cohorts.",
        "- Inference: animal with session nested; never neuron or time bin.",
        "- Model: teacher-forced one-step conditional Poisson shared-basis dynamics, not a full LDS.",
        "- Mechanism boundary: these recordings do not establish E/I, Dale, or recurrent plasticity.",
        "- Truth capability: probabilityLeft is restricted to whole-block splitting and post-fit evaluation.",
        f"- Raw-run commit: `{receipts['git_commit'].iloc[0]}`; analysis commit: `{analysis['analysis_git_commit']}`.",
        "",
        _markdown_table(
            summary[
                [
                    "proposition",
                    "comparison",
                    "estimate",
                    "ci_low",
                    "ci_high",
                    "conclusion",
                    "claim_scope",
                ]
            ]
        ),
        "",
    ]
    paths["report"].write_text("\n".join(report), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()
    config = load_json_config(args.config)
    publish_snapshot(
        args.results_root,
        config,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
