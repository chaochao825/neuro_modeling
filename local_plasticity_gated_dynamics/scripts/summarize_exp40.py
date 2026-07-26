"""Summarize the registered and assay-probe Exp40 runs without cherry-picking."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.ibl_state_utility_summary import (  # noqa: E402
    exp40_animal_effects,
    exp40_claims,
    exp40_condition_summary,
)


EXPERIMENT = "exp40_ibl_state_utility"
PRIMARY_PROFILE = "development_posthoc_exposed_cohort"
PROBE_PROFILE = "development_posthoc_assay_probe"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(run_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    metrics_path = run_path / "metrics.jsonl"
    config_path = run_path / "config.json"
    status_path = run_path / "status.json"
    planned_path = run_path / "planned_conditions.json"
    for path in (metrics_path, config_path, status_path, planned_path):
        if not path.is_file():
            raise FileNotFoundError(f"Exp40 run artifact is missing: {path}")
    frame = pd.DataFrame(
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    planned = json.loads(planned_path.read_text(encoding="utf-8"))
    if len(frame) != len(planned):
        raise ValueError("Exp40 metrics do not cover every planned condition")
    return frame, {
        "run_path": str(run_path),
        "run_name": run_path.name,
        "profile": str(config.get("profile", "")),
        "status": str(status.get("status", "")),
        "condition_failures": int(status.get("condition_failures", -1)),
        "metrics_rows": int(len(frame)),
        "metrics_sha256": _sha256(metrics_path),
        "config_sha256": _sha256(config_path),
        "planned_conditions_sha256": _sha256(planned_path),
    }


def _latest_profile_run(results_root: Path, profile: str) -> Path:
    experiment_root = results_root / "runs" / EXPERIMENT
    candidates = []
    for config_path in experiment_root.glob("seed_*/*/config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if str(config.get("profile", "")) == profile:
            candidates.append(config_path.parent)
    if not candidates:
        raise FileNotFoundError(f"no Exp40 run has profile {profile}")
    return max(candidates, key=lambda path: path.name)


def _report(
    condition_summary: pd.DataFrame,
    effects: pd.DataFrame,
    claims: pd.DataFrame,
    receipt: dict[str, object],
) -> str:
    eligible = effects.loc[effects["endpoint_status"].astype(str).eq("complete")]
    failed = effects.loc[~effects["endpoint_status"].astype(str).eq("complete")]
    registered = condition_summary.loc[
        condition_summary["analysis"].astype(str).eq("registered_readout")
    ].set_index("condition")
    probe = condition_summary.loc[
        condition_summary["analysis"].astype(str).eq("assay_probe")
    ].set_index("condition")
    claim_rows = claims.set_index("claim")

    lines = [
        "# Exp40 IBL factorized-state utility audit",
        "",
        "## Decision",
        "",
        "The outcome-exposed 30-animal cohort is a post-hoc development panel, not "
        "confirmatory evidence. The development gate **did not pass**. A disjoint "
        "new cohort was therefore neither frozen nor opened, and neural analysis "
        "remains locked.",
        "",
        f"All 30 animals and all 210 planned condition cells were retained. "
        f"{len(eligible)} animals formed a complete paired endpoint panel; "
        f"{len(failed)} failed symmetrically across all seven conditions because "
        "their chronological test fold contained fewer than eight low-contrast "
        "choices.",
        "",
        "## Registered development readout",
        "",
        "| Condition | Parameters | Mean low-contrast choice NLL | Mean context NLL |",
        "|---|---:|---:|---:|",
    ]
    order = [
        "history_only",
        "learned_hmm_mean",
        "semimarkov_mean",
        "semimarkov_release",
        "semimarkov_concentration",
        "factorized_state",
        "oracle_context_mean",
    ]
    for condition in order:
        row = registered.loc[condition]
        lines.append(
            f"| {condition} | {int(row.behavior_parameter_count)} | "
            f"{row.mean_test_low_contrast_choice_nll:.6f} | "
            f"{row.mean_context_nll:.6f} |"
        )
    utility = claim_rows.loc["any_behavioral_utility"]
    meaningful = claim_rows.loc["meaningful_behavioral_utility"]
    context = claim_rows.loc["context_decoding_gain"]
    release = claim_rows.loc["release_actuator_contribution"]
    precision = claim_rows.loc["precision_actuator_contribution"]
    lines.extend(
        [
            "",
            "Semi-Markov context decoding improved over the learned HMM by "
            f"{context.estimate:.6f} nats/trial "
            f"(95% animal bootstrap [{context.ci_low:.6f}, "
            f"{context.ci_high:.6f}]); the bounded development conclusion is "
            f"**{context.conclusion}**.",
            "",
            "The factorized state did not convert that decoding gain into held-out "
            "choice utility. Dev-selected baseline minus factorized NLL was "
            f"{utility.estimate:.6f} [{utility.ci_low:.6f}, "
            f"{utility.ci_high:.6f}], positive in "
            f"{int(utility.positive_animals)}/{int(utility.n_animals)} animals. "
            f"Any positive gain is **{utility.conclusion}**; a registered 0.005 "
            f"nats/trial meaningful gain is **{meaningful.conclusion}**.",
            "",
            f"Release clamp harm was {release.estimate:.6f} "
            f"[{release.ci_low:.6f}, {release.ci_high:.6f}] "
            f"(**{release.conclusion}**). Precision clamp harm was "
            f"{precision.estimate:.6f} [{precision.ci_low:.6f}, "
            f"{precision.ci_high:.6f}] (**{precision.conclusion}**).",
            "",
            "## Post-outcome assay probe",
            "",
            "After inspecting the registered development result, one bounded probe "
            "selected regularization on all dev choices and added stronger "
            "regularization. It changed no observer state, task endpoint, test fold, "
            "or baseline family. Mean factorized low-contrast NLL changed from "
            f"{registered.loc['factorized_state'].mean_test_low_contrast_choice_nll:.6f} "
            f"to {probe.loc['factorized_state'].mean_test_low_contrast_choice_nll:.6f}. "
            "Its paired gain over the dev-selected baseline remained negative "
            f"({eligible.probe_gain_selected_baseline_minus_factorized.mean():.6f} "
            "nats/trial). This probe diagnoses readout variance but cannot rescue or "
            "confirm the hypothesis.",
            "",
            "## Interpretation boundary",
            "",
            "The supported decoding result establishes only that known task duration "
            "structure helps recover the experimenter's block label. It does not "
            "show that release probability or run-length precision improves animal "
            "choice prediction, implements a biological actuator, or recovers an "
            "independently identifiable sensory-noise state. The truth-context "
            "condition is an evaluation diagnostic, not a behavioral upper bound: "
            "animals act on subjective beliefs rather than the experimenter's label.",
            "",
            "The next admissible confirmation task must independently manipulate "
            "environmental volatility and observation noise and must first show "
            "held-out utility on an outcome-blind development gate. Scaling to IBL "
            "neural data is not unlocked by context NLL alone.",
            "",
            "## Artifact receipt",
            "",
            f"- Registered run: `{receipt['primary']['run_path']}`",
            f"- Assay-probe run: `{receipt['probe']['run_path']}`",
            f"- Registered metrics SHA-256: `{receipt['primary']['metrics_sha256']}`",
            f"- Probe metrics SHA-256: `{receipt['probe']['metrics_sha256']}`",
            "- Statistical unit: animal; one session per animal in this cohort.",
            "- Multiplicity: Holm across the five bounded development claims.",
            "- Confirmatory status: inconclusive/not run on new data.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(
    *,
    results_root: Path,
    output_root: Path,
    primary_run: Path | None = None,
    probe_run: Path | None = None,
    n_resamples: int = 20_000,
) -> dict[str, Path]:
    primary_path = primary_run or _latest_profile_run(results_root, PRIMARY_PROFILE)
    probe_path = probe_run or _latest_profile_run(results_root, PROBE_PROFILE)
    primary, primary_receipt = _load_run(primary_path)
    probe, probe_receipt = _load_run(probe_path)
    if primary_receipt["profile"] != PRIMARY_PROFILE:
        raise ValueError(
            "primary run profile is not the registered development profile"
        )
    if probe_receipt["profile"] != PROBE_PROFILE:
        raise ValueError("probe run profile is not the bounded assay profile")
    effects = exp40_animal_effects(primary, probe)
    claims = exp40_claims(effects, n_resamples=n_resamples)
    conditions = exp40_condition_summary(primary, probe)
    receipt = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "selection_policy": "latest_attempt_with_exact_profile_including_failures",
        "primary": primary_receipt,
        "probe": probe_receipt,
        "planned_animals": int(effects["animal_id"].nunique()),
        "complete_endpoint_animals": int(
            effects["endpoint_status"].astype(str).eq("complete").sum()
        ),
        "new_disjoint_cohort_frozen": False,
        "new_disjoint_cohort_outcomes_opened": False,
        "neural_analysis_unlocked": False,
        "development_gate": "failed",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "condition_summary": output_root
        / "exp40_ibl_state_utility_condition_summary.csv",
        "animal_effects": output_root / "exp40_ibl_state_utility_animal_effects.csv",
        "claims": output_root / "exp40_ibl_state_utility_claims.csv",
        "receipt": output_root / "exp40_ibl_state_utility_receipt.json",
        "report": output_root / "exp40_ibl_state_utility_report.md",
    }
    conditions.to_csv(paths["condition_summary"], index=False)
    effects.to_csv(paths["animal_effects"], index=False)
    claims.to_csv(paths["claims"], index=False)
    paths["receipt"].write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(
        _report(conditions, effects, claims, receipt), encoding="utf-8"
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--primary-run", type=Path)
    parser.add_argument("--probe-run", type=Path)
    parser.add_argument("--n-resamples", type=int, default=20_000)
    args = parser.parse_args()
    paths = summarize(
        results_root=args.results_root,
        output_root=args.output_root,
        primary_run=args.primary_run,
        probe_run=args.probe_run,
        n_resamples=args.n_resamples,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
