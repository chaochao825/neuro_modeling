"""Aggregate immutable visual-context runs without treating folds as replicates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared_dynamics_real_data.run_visual_context import _source_fingerprint


PANEL_COLUMNS = ["analysis_fingerprint", "data_fingerprint"]
CELL_KEY = [
    "profile",
    "computational_seed",
    "fold",
    "latent_dim",
    "model",
]
SCORE_METRICS = [
    "nll_per_scalar",
    "standardized_nll_per_scalar",
    "one_step_r2",
    "rollout_nrmse",
    "parameter_count",
    "effective_rank",
    "top_k_singular_energy",
    "subspace_angle_degrees",
]
SEED_KEYS = [
    "analysis_fingerprint",
    "data_fingerprint",
    "dataset_pair",
    "profile",
    "computational_seed",
    "latent_dim",
    "model",
]
COMPARISON_COLUMNS = [
    "analysis_fingerprint",
    "data_fingerprint",
    "dataset_pair",
    "profile",
    "computational_seed",
    "latent_dim",
    "shared_minus_common_nll",
    "shared_minus_separate_nll",
    "retained_switching_gain",
    "random_minus_shared_nll",
    "orthogonal_minus_shared_nll",
    "shuffled_minus_shared_nll",
    "shared_parameter_fraction_of_separate",
]


def collect_runs(results_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in sorted((results_root / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        planned_path = run_dir / "planned_conditions.json"
        if not config_path.is_file() or not planned_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        planned = pd.DataFrame(
            json.loads(planned_path.read_text(encoding="utf-8"))
        )
        if planned.empty or not set(CELL_KEY[1:]) <= set(planned.columns):
            raise ValueError(f"invalid planned_conditions in {run_dir}")
        semantic_key = ["computational_seed", "fold", "latent_dim", "model"]
        if planned.duplicated(semantic_key).any():
            raise ValueError(f"duplicate planned condition in {run_dir}")
        plan_fields = {"seeds", "n_splits", "latent_dims", "model_specs"}
        if plan_fields <= set(config):
            expected_plan = {
                (int(seed), int(fold), int(latent_dim), str(spec["model"]))
                for seed in config["seeds"]
                for fold in range(int(config["n_splits"]))
                for latent_dim in config["latent_dims"]
                for spec in config["model_specs"]
            }
            observed_plan = set(
                planned[semantic_key].itertuples(index=False, name=None)
            )
            if observed_plan != expected_plan:
                raise ValueError(f"planned_conditions do not match config in {run_dir}")
        status_path = run_dir / "status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else {}
        )
        manifest_path = run_dir / "data_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        metrics_path = run_dir / "metrics.csv"
        try:
            recorded = (
                pd.read_csv(metrics_path)
                if metrics_path.is_file() and metrics_path.stat().st_size > 0
                else pd.DataFrame()
            )
        except pd.errors.EmptyDataError:
            recorded = pd.DataFrame()
        if recorded.empty and (run_dir / "metrics.jsonl").is_file():
            jsonl_rows = []
            for line in (run_dir / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines():
                if line.strip():
                    jsonl_rows.append(json.loads(line))
            recorded = pd.DataFrame(jsonl_rows)
        if not recorded.empty:
            missing_recorded_keys = set(semantic_key) - set(recorded.columns)
            if missing_recorded_keys:
                raise ValueError(
                    f"recorded metrics lack keys {sorted(missing_recorded_keys)} "
                    f"in {run_dir}"
                )
            if recorded.duplicated(semantic_key).any():
                raise ValueError(f"duplicate recorded condition in {run_dir}")
            planned_keys = set(
                planned[semantic_key].itertuples(index=False, name=None)
            )
            recorded_keys = set(
                recorded[semantic_key].itertuples(index=False, name=None)
            )
            if not recorded_keys <= planned_keys:
                raise ValueError(f"recorded condition absent from plan in {run_dir}")
            recorded["run_id"] = run_dir.name
            recorded["run_source_fingerprint"] = config.get(
                "source_fingerprint", ""
            )
            frames.append(recorded)

        if recorded.empty:
            missing = planned.copy()
        else:
            observed_key = recorded[semantic_key].drop_duplicates()
            missing = planned.merge(
                observed_key,
                on=semantic_key,
                how="left",
                indicator=True,
                validate="one_to_one",
            )
            missing = missing.loc[missing["_merge"] == "left_only"].drop(
                columns="_merge"
            )
        if not missing.empty:
            missing["run_id"] = run_dir.name
            missing["profile"] = config.get("profile", "unknown")
            missing["analysis_fingerprint"] = config.get(
                "analysis_fingerprint", ""
            )
            missing["run_source_fingerprint"] = config.get(
                "source_fingerprint", ""
            )
            missing["data_fingerprint"] = manifest.get("data_fingerprint", np.nan)
            missing["dataset_pair"] = manifest.get(
                "dataset_pair", "visual_responding__visual_spontaneous"
            )
            missing["configured_seed_universe"] = json.dumps(
                config.get("configured_seeds", config.get("seeds", [])),
                separators=(",", ":"),
            )
            missing["n_units"] = config.get("n_units", np.nan)
            missing["purge"] = config.get("purge", np.nan)
            missing["status"] = "missing"
            missing["error_type"] = status.get(
                "top_level_error_type", "MissingCell"
            ) or "MissingCell"
            missing["error_message"] = status.get(
                "top_level_error_message",
                "planned cell absent from metrics.csv",
            ) or "planned cell absent from metrics.csv"
            frames.append(missing)
    if not frames:
        raise FileNotFoundError(
            f"no readable run artifacts beneath {results_root / 'runs'}"
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def _latest_panel(
    raw: pd.DataFrame,
    profile: str,
    *,
    analysis_fingerprint: str | None = None,
) -> pd.DataFrame:
    data = raw.loc[raw["profile"] == profile].copy()
    if data.empty:
        raise ValueError(f"no run rows for profile {profile!r}")
    missing_panel = set(PANEL_COLUMNS) - set(data.columns)
    if missing_panel:
        raise ValueError(
            f"metrics lack provenance fingerprints: {sorted(missing_panel)}"
        )
    available = sorted(data["analysis_fingerprint"].dropna().unique())
    if analysis_fingerprint is None:
        if len(available) != 1:
            raise ValueError(
                "profile contains multiple analysis fingerprints; pass one explicitly: "
                + ", ".join(available)
            )
        analysis_fingerprint = available[0]
    data = data.loc[data["analysis_fingerprint"] == analysis_fingerprint].copy()
    if data.empty:
        raise ValueError(f"unknown analysis fingerprint {analysis_fingerprint!r}")
    missing = set(CELL_KEY) - set(data.columns)
    if missing:
        raise ValueError(f"metrics are missing keys: {sorted(missing)}")
    data = data.sort_values("run_id").drop_duplicates(CELL_KEY, keep="last")
    data_versions = data["data_fingerprint"].dropna().unique()
    if len(data_versions) != 1 or data["data_fingerprint"].isna().any():
        raise ValueError(
            "latest analysis panel contains multiple or missing data fingerprints"
        )
    if "configured_seed_universe" in data:
        universes = data["configured_seed_universe"].dropna().astype(str).unique()
        if len(universes) != 1:
            raise ValueError("analysis panel has inconsistent configured seed universes")
        configured_seeds = [int(value) for value in json.loads(universes[0])]
        observed_seeds = set(data["computational_seed"].astype(int))
        missing_seeds = sorted(set(configured_seeds) - observed_seeds)
        unexpected_seeds = sorted(observed_seeds - set(configured_seeds))
        if unexpected_seeds:
            raise ValueError(f"panel contains non-configured seeds: {unexpected_seeds}")
        if missing_seeds:
            templates = data.sort_values("run_id").drop_duplicates(
                ["fold", "latent_dim", "model"]
            )
            synthesized = []
            for seed in missing_seeds:
                for row in templates.to_dict(orient="records"):
                    row.update(
                        {
                            "computational_seed": seed,
                            "run_id": "missing_configured_seed",
                            "status": "missing",
                            "error_type": "MissingSeed",
                            "error_message": "configured seed has no run artifact",
                        }
                    )
                    for metric in SCORE_METRICS:
                        row[metric] = np.nan
                    synthesized.append(row)
            data = pd.concat([data, pd.DataFrame(synthesized)], ignore_index=True)
    return data.sort_values(CELL_KEY).reset_index(drop=True)


def _seed_level(latest: pd.DataFrame) -> pd.DataFrame:
    panel_keys = [
        "analysis_fingerprint",
        "data_fingerprint",
        "dataset_pair",
        "profile",
        "computational_seed",
        "latent_dim",
    ]
    expected_models = int(latest["model"].nunique())
    expected_folds = int(latest["fold"].nunique())
    expected_cells = expected_models * expected_folds
    valid = (
        latest.assign(_complete=latest["status"].eq("complete"))
        .groupby(panel_keys, as_index=False)
        .agg(
            _cells=("model", "size"),
            _complete_cells=("_complete", "sum"),
            _model_count=("model", "nunique"),
            _fold_count=("fold", "nunique"),
        )
    )
    valid = valid.loc[
        (valid["_cells"] == expected_cells)
        & (valid["_complete_cells"] == expected_cells)
        & (valid["_model_count"] == expected_models)
        & (valid["_fold_count"] == expected_folds),
        panel_keys,
    ]
    complete = latest.merge(valid, on=panel_keys, how="inner", validate="many_to_one")
    if complete.empty:
        return pd.DataFrame(columns=SEED_KEYS + SCORE_METRICS)
    # Folds are correlated resampling diagnostics. Aggregate them inside each
    # computational seed before any seed-robustness summary.
    return (
        complete.groupby(SEED_KEYS, as_index=False)[SCORE_METRICS]
        .mean()
        .sort_values(["computational_seed", "latent_dim", "model"])
    )


def _model_summary(seed_level: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "analysis_fingerprint",
        "data_fingerprint",
        "dataset_pair",
        "profile",
        "latent_dim",
        "model",
    ]
    output_columns = keys + [
        f"{metric}_{statistic}"
        for metric in (
            "nll_per_scalar",
            "one_step_r2",
            "rollout_nrmse",
            "parameter_count",
            "effective_rank",
            "top_k_singular_energy",
            "subspace_angle_degrees",
        )
        for statistic in ("mean", "std", "median", "min", "max")
    ] + ["n_computational_seeds"]
    if seed_level.empty:
        return pd.DataFrame(columns=output_columns)
    metrics = [
        "nll_per_scalar",
        "one_step_r2",
        "rollout_nrmse",
        "parameter_count",
        "effective_rank",
        "top_k_singular_energy",
        "subspace_angle_degrees",
    ]
    grouped = seed_level.groupby(keys)
    pieces = []
    for metric in metrics:
        part = grouped[metric].agg(["mean", "std", "median", "min", "max"]).reset_index()
        part = part.rename(
            columns={name: f"{metric}_{name}" for name in ("mean", "std", "median", "min", "max")}
        )
        pieces.append(part)
    result = pieces[0]
    for part in pieces[1:]:
        result = result.merge(
            part,
            on=keys,
            validate="one_to_one",
        )
    counts = grouped.size().rename("n_computational_seeds").reset_index()
    return result.merge(
        counts,
        on=keys,
        validate="one_to_one",
    )


def _comparisons(seed_level: pd.DataFrame) -> pd.DataFrame:
    if seed_level.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    index = [
        "analysis_fingerprint",
        "data_fingerprint",
        "dataset_pair",
        "profile",
        "computational_seed",
        "latent_dim",
    ]
    nll = seed_level.pivot(
        index=index,
        columns="model",
        values="nll_per_scalar",
    )
    params = seed_level.pivot(
        index=index,
        columns="model",
        values="parameter_count",
    )
    required = ["common", "shared", "separate", "random", "orthogonal", "shuffled"]
    nll = nll.reindex(columns=required)
    params = params.reindex(index=nll.index, columns=required)
    out = nll.reset_index()[index].copy()
    out["shared_minus_common_nll"] = nll["shared"].to_numpy() - nll["common"].to_numpy()
    out["shared_minus_separate_nll"] = nll["shared"].to_numpy() - nll["separate"].to_numpy()
    denominator = nll["common"].to_numpy() - nll["separate"].to_numpy()
    numerator = nll["common"].to_numpy() - nll["shared"].to_numpy()
    out["retained_switching_gain"] = np.where(
        denominator > 0,
        numerator / denominator,
        np.nan,
    )
    for control in ("random", "orthogonal", "shuffled"):
        out[f"{control}_minus_shared_nll"] = (
            nll[control].to_numpy() - nll["shared"].to_numpy()
        )
    out["shared_parameter_fraction_of_separate"] = (
        params["shared"].to_numpy() / params["separate"].to_numpy()
    )
    return out.reindex(columns=COMPARISON_COLUMNS)


def _claim_rows(
    comparisons: pd.DataFrame,
    seed_level: pd.DataFrame,
    latest: pd.DataFrame,
) -> pd.DataFrame:
    claim_columns = [
        "claim_id",
        "criterion",
        "target_dim",
        "descriptive_estimate",
        "descriptive_direction",
        "stats_unit",
        "n_biological_units",
        "n_computational_seeds",
        "n_failed_cells",
        "conclusion",
        "reason",
    ]
    dims = sorted(seed_level["latent_dim"].unique()) if not seed_level.empty else []
    # These are explicitly rank-4 claims; do not silently substitute a
    # different tested dimension when d=4 is absent.
    target_dim = 4.0 if 4 in dims else np.nan
    max_dim = max(dims) if dims else np.nan
    at_target = comparisons.loc[comparisons["latent_dim"] == target_dim].copy()
    gain_panel = at_target.dropna(
        subset=[
            "retained_switching_gain",
            "shared_parameter_fraction_of_separate",
        ]
    )
    gain_pair_panel = at_target.dropna(
        subset=[
            "shared_minus_common_nll",
            "shared_minus_separate_nll",
            "shared_parameter_fraction_of_separate",
        ]
    )
    control_columns = [
        "random_minus_shared_nll",
        "orthogonal_minus_shared_nll",
        "shuffled_minus_shared_nll",
    ]
    control_panel = at_target.dropna(subset=control_columns)
    switching_panel = at_target.dropna(subset=["shared_minus_common_nll"])
    shared = seed_level.loc[seed_level["model"] == "shared"].copy()
    shared_target = shared.loc[shared["latent_dim"] == target_dim].dropna(
        subset=["one_step_r2", "rollout_nrmse"]
    )
    shared_index = [
        "analysis_fingerprint",
        "data_fingerprint",
        "dataset_pair",
        "profile",
        "computational_seed",
    ]
    if shared.empty or not np.isfinite(target_dim) or not np.isfinite(max_dim):
        paired_dim = pd.DataFrame(columns=["target", "max"])
    else:
        by_dim = shared.pivot(
            index=shared_index, columns="latent_dim", values="nll_per_scalar"
        )
        if target_dim == max_dim or target_dim not in by_dim or max_dim not in by_dim:
            paired_dim = pd.DataFrame(columns=["target", "max"])
        else:
            paired_dim = by_dim[[target_dim, max_dim]].dropna().rename(
                columns={target_dim: "target", max_dim: "max"}
            )
    failed = int((latest["status"] != "complete").sum())
    gain_is_fully_defined = (
        not gain_pair_panel.empty and len(gain_panel) == len(gain_pair_panel)
    )
    gain_estimate = (
        min(
            float(gain_panel["retained_switching_gain"].median()) - 0.95,
            1.0
            - float(gain_panel["shared_parameter_fraction_of_separate"].median()),
        )
        if gain_is_fully_defined
        else np.nan
    )
    gain_direction = (
        "unavailable"
        if not gain_is_fully_defined
        else "support"
        if gain_estimate >= 0.0
        else "oppose"
    )
    control_min = (
        control_panel[control_columns].min(axis=1) if not control_panel.empty else pd.Series(dtype=float)
    )
    control_estimate = float(control_min.median()) if not control_min.empty else np.nan
    control_direction = (
        "unavailable"
        if control_panel.empty
        else "support"
        if control_estimate > 0.0
        else "oppose"
    )
    rank_difference = (
        paired_dim["target"] - paired_dim["max"]
        if not paired_dim.empty
        else pd.Series(dtype=float)
    )
    rank_estimate = float(rank_difference.median()) if not rank_difference.empty else np.nan
    rank_direction = (
        "unavailable"
        if rank_difference.empty
        else "support"
        if rank_estimate <= 0.01
        else "oppose"
    )
    switching_estimate = (
        float(switching_panel["shared_minus_common_nll"].median())
        if not switching_panel.empty
        else np.nan
    )
    switching_direction = (
        "unavailable"
        if switching_panel.empty
        else "support"
        if switching_estimate < 0.0
        else "oppose"
    )
    predictive_estimate = (
        min(
            float(shared_target["one_step_r2"].median()),
            1.0 - float(shared_target["rollout_nrmse"].median()),
        )
        if not shared_target.empty
        else np.nan
    )
    predictive_direction = (
        "unavailable"
        if shared_target.empty
        else "support"
        if predictive_estimate > 0.0
        else "oppose"
    )
    records = [
        {
            "claim_id": "R0_switching_improves_common",
            "criterion": "shared context transitions have lower NLL than common transition",
            "target_dim": target_dim,
            "descriptive_estimate": switching_estimate,
            "descriptive_direction": switching_direction,
            "n_computational_seeds": int(
                switching_panel["computational_seed"].nunique()
            ),
        },
        {
            "claim_id": "R1_shared_retains_separate_gain",
            "criterion": "min(median retained gain-0.95, 1-median parameter fraction) >=0",
            "target_dim": target_dim,
            "descriptive_estimate": gain_estimate,
            "descriptive_direction": gain_direction,
            "n_computational_seeds": int(
                gain_pair_panel["computational_seed"].nunique()
            ),
            "unavailable_reason": (
                "paired models are complete, but separate does not improve "
                "common (common-minus-separate NLL <=0), so there is no "
                "positive switching gain to retain"
                if not gain_pair_panel.empty and not gain_is_fully_defined
                else "no complete paired computational seeds for this criterion"
            ),
        },
        {
            "claim_id": "R2_aligned_beats_basis_controls",
            "criterion": "median per-seed minimum control-minus-shared NLL >0",
            "target_dim": target_dim,
            "descriptive_estimate": control_estimate,
            "descriptive_direction": control_direction,
            "n_computational_seeds": int(control_panel["computational_seed"].nunique()),
        },
        {
            "claim_id": "R3_d4_nll_vs_highest_tested_dimension",
            "criterion": (
                "d=4 shared NLL no more than 0.01 above highest tested "
                f"dimension (d={int(max_dim) if np.isfinite(max_dim) else 'NA'}); "
                "not an intrinsic-rank estimate"
            ),
            "target_dim": target_dim,
            "descriptive_estimate": rank_estimate,
            "descriptive_direction": rank_direction,
            "n_computational_seeds": int(len(paired_dim)),
        },
        {
            "claim_id": "R4_shared_has_absolute_predictive_signal",
            "criterion": "min(median one-step R2, 1-median rollout NRMSE) >0",
            "target_dim": target_dim,
            "descriptive_estimate": predictive_estimate,
            "descriptive_direction": predictive_direction,
            "n_computational_seeds": int(
                shared_target["computational_seed"].nunique()
            ),
        },
    ]
    for record in records:
        unavailable_reason = record.pop(
            "unavailable_reason", "no complete paired computational seeds for this criterion"
        )
        reason = (
            unavailable_reason + "; "
            if record["descriptive_direction"] == "unavailable"
            else ""
        )
        reason += (
            "only one aligned recording pair; seeds/folds/neurons are not "
            "biological replicates"
        )
        record.update(
            {
                "stats_unit": "recording_pair",
                "n_biological_units": 1,
                "n_failed_cells": failed,
                # One responding/spontaneous population cannot support
                # population-level inference regardless of seed robustness.
                "conclusion": "inconclusive",
                "reason": reason,
            }
        )
    return pd.DataFrame(records).reindex(columns=claim_columns)


def _fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.5g}"


def write_report(
    path: Path,
    *,
    profile: str,
    raw: pd.DataFrame,
    latest: pd.DataFrame,
    model_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    claims: pd.DataFrame,
) -> None:
    complete = int((latest["status"] == "complete").sum())
    failed = int((latest["status"] != "complete").sum())
    lines = [
        "# Shared-basis real-data report",
        "",
        f"Profile: `{profile}`. Latest planned panel: {len(latest)} cells; complete: {complete}; failed: {failed}.",
        "",
        "The analysis uses the upstream-described same-neuron visual responding/spontaneous pair; row alignment cannot be independently verified because the MAT files contain no unit identifiers. Contiguous blocks are held out with a purge gap. Unit selection, scaling, PCA bases, transition/noise parameters, and rollout normalization are fit on training blocks only.",
        "",
        "Computational seeds sample matched neuron subsets. Folds, seeds, time bins, and neurons are not biological replicates. Consequently every population-level claim remains `inconclusive` until independent sessions/animals are available.",
        "",
        "The direction column is a deterministic audit of this single recording pair, not an inferential result. In particular, a positive shared-minus-common NLL opposes a switching advantage, while a non-positive absolute-signal margin opposes positive one-step R2 together with rollout error below the training-dispersion scale.",
        "",
        "## Three-way claim audit",
        "",
        "| Claim | Criterion | Descriptive estimate | Direction within this recording | Formal conclusion |",
        "|---|---|---:|---|---|",
    ]
    for row in claims.itertuples(index=False):
        lines.append(
            f"| {row.claim_id} | {row.criterion} | {_fmt(float(row.descriptive_estimate))} | {row.descriptive_direction} | **{row.conclusion}** |"
        )
    unavailable = claims.loc[
        claims["descriptive_direction"] == "unavailable"
    ]
    if not unavailable.empty:
        lines.extend(["", "Unavailable descriptive criteria:"])
        for row in unavailable.itertuples(index=False):
            lines.append(f"- `{row.claim_id}`: {row.reason}.")
    lines.extend(
        [
            "",
            "## Model summary",
            "",
            "Values aggregate folds inside each computational seed and then summarize seed robustness.",
            "",
            "| d | Model | NLL/scalar mean | one-step R2 mean | rollout NRMSE mean | parameters median | effective rank mean | top-k energy mean | seeds |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in model_summary.sort_values(["latent_dim", "model"]).itertuples(index=False):
        lines.append(
            f"| {int(row.latent_dim)} | {row.model} | {_fmt(row.nll_per_scalar_mean)} | {_fmt(row.one_step_r2_mean)} | {_fmt(row.rollout_nrmse_mean)} | {_fmt(row.parameter_count_median)} | {_fmt(row.effective_rank_mean)} | {_fmt(row.top_k_singular_energy_mean)} | {int(row.n_computational_seeds)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The Gaussian LDS likelihood is a predictive population score for binarized calcium-event vectors; it is not a Bernoulli spike likelihood or a causal recurrent-connectivity estimate.",
            "- Responding and spontaneous are separate recordings without within-recording switch timestamps. This tests cross-context parameter sharing, not natural fast switching.",
            "- The source files contain no trial, behavior, animal, E/I, or anatomical-coordinate metadata. Those claims are not tested here.",
            "- `minimal_computation_python` estimates equal-time direct dependencies and minimal input count; it is reported separately from latent rank.",
            "",
            "## Artifacts",
            "",
            "- `results/raw_metrics.csv`: all attempts, including failures.",
            "- `results/latest_metrics.csv`: latest complete/failed planned cells for the selected profile.",
            "- `results/model_summary.csv`: fold-within-seed aggregation.",
            "- `results/comparisons.csv`: paired computational robustness contrasts.",
            "- `results/summary.csv`: formal three-category claim table.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(
    results_root: Path,
    *,
    profile: str,
    analysis_fingerprint: str | None = None,
) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    raw = collect_runs(results_root)
    latest = _latest_panel(
        raw, profile, analysis_fingerprint=analysis_fingerprint
    )
    seed_level = _seed_level(latest)
    model_summary = _model_summary(seed_level)
    comparisons = _comparisons(seed_level)
    claims = _claim_rows(comparisons, seed_level, latest)
    raw.to_csv(results_root / "raw_metrics.csv", index=False)
    latest.to_csv(results_root / "latest_metrics.csv", index=False)
    seed_level.to_csv(results_root / "seed_level_metrics.csv", index=False)
    model_summary.to_csv(results_root / "model_summary.csv", index=False)
    comparisons.to_csv(results_root / "comparisons.csv", index=False)
    claims.to_csv(results_root / "summary.csv", index=False)
    selected_fingerprint = str(latest["analysis_fingerprint"].iloc[0])
    selected_data = str(latest["data_fingerprint"].iloc[0])
    run_sources = sorted(
        str(value)
        for value in latest.get("run_source_fingerprint", pd.Series(dtype=str))
        .dropna()
        .unique()
        if str(value)
    )
    (results_root / "panel.json").write_text(
        pd.Series(
            {
                "profile": profile,
                "analysis_fingerprint": selected_fingerprint,
                "data_fingerprint": selected_data,
                "run_source_fingerprints": run_sources,
                "report_source_fingerprint": _source_fingerprint(),
                "latest_cells": len(latest),
            }
        ).to_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    write_report(
        results_root / "report.md",
        profile=profile,
        raw=raw,
        latest=latest,
        model_summary=model_summary,
        comparisons=comparisons,
        claims=claims,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--profile", default="formal")
    parser.add_argument("--analysis-fingerprint", default=None)
    args = parser.parse_args()
    build(
        args.results_root,
        profile=args.profile,
        analysis_fingerprint=args.analysis_fingerprint,
    )


if __name__ == "__main__":
    main()
