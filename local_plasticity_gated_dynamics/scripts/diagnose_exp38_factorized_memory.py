#!/usr/bin/env python3
"""Post-hoc Exp38 audit of direct learning rates and likelihood algebra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config, seed_list
from experiments.exp38_stream51_soft_memory import (
    _make_streams,
    _store,
    validate_config,
    validate_implementation_receipt,
    validate_preregistration,
)
from src.analysis.factorized_memory_diagnostic import (
    BeliefTrace,
    direct_alpha_filter,
    fit_oracle_write_probe,
    grouped_binary_metrics,
    likelihood_hmm_filter,
    oracle_write_targets,
    source_video_belief_metrics,
    summarize_video_metrics,
)
from src.data.stream51_streaming import Stream51Stream, fit_stream51_vmf
from src.models.soft_memory_controller import (
    accumulate_with_retention,
    causal_control_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPHA_GRID = (0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
HAZARD_GRID = (0.0, 1 / 192, 1 / 96, 1 / 48, 1 / 24, 1 / 12, 1 / 6, 1 / 3)
STREAM51_RETAIN_MIN_SEEDS = 4
STREAM51_RETAIN_MIN_MEAN_NLL_GAIN = 0.005


def _one_path(paths: Sequence[Path], *, label: str) -> Path:
    selected = tuple(paths)
    if len(selected) != 1:
        raise RuntimeError(f"expected one {label}, found {len(selected)}")
    return selected[0]


def _run_path(results_root: Path, *, seed: int) -> Path:
    return _one_path(
        tuple(
            results_root.glob(
                "qualification_runs/runs/exp38_stream51_soft_memory/"
                f"seed_{seed}/*_qualification"
            )
        ),
        label=f"qualification run for seed {seed}",
    )


def _load_run_contract(results_root: Path, *, seed: int) -> dict[str, Any]:
    run = _run_path(results_root, seed=seed)
    selected = json.loads(
        (run / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    partition = json.loads(
        (run / "development_partition.json").read_text(encoding="utf-8")
    )
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("external_features_accessed") is not False:
        raise RuntimeError("diagnostic encountered external feature access")
    return {"run": run, "selected": selected, "partition": partition}


def _posterior_ema(stream: Stream51Stream, *, retention: float) -> BeliefTrace:
    predictions, probabilities, _ = accumulate_with_retention(
        stream.evidence,
        stream_ids=stream.stream_ids,
        retention_value=np.full(len(stream.labels), float(retention)),
    )
    return BeliefTrace(probabilities=probabilities, predictions=predictions)


def _score_streams(
    streams: Sequence[Stream51Stream],
    trace_factory: Callable[[Stream51Stream], BeliefTrace],
    *,
    post_switch_window: int,
    seed: int,
    condition: str,
    panel: str,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for stream in streams:
        trace = trace_factory(stream)
        frame = source_video_belief_metrics(
            trace.probabilities,
            stream.labels,
            source_video_ids=stream.source_video_ids,
            switch_flags=stream.switch_flags,
            post_switch_window=post_switch_window,
        )
        frame.insert(0, "panel", panel)
        frame.insert(0, "condition", condition)
        frame.insert(0, "seed", int(seed))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["source_video_id"].duplicated().any():
        raise RuntimeError("source video was scored more than once in one panel")
    return summarize_video_metrics(combined), combined


def _select_scalar(
    streams: Sequence[Stream51Stream],
    *,
    grid: Sequence[float],
    trace_factory: Callable[[Stream51Stream, float], BeliefTrace],
    post_switch_window: int,
    seed: int,
    family: str,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for candidate in map(float, grid):
        summary, _ = _score_streams(
            streams,
            lambda stream, value=candidate: trace_factory(stream, value),
            post_switch_window=post_switch_window,
            seed=seed,
            condition=family,
            panel="fit_hidden",
        )
        rows.append(
            {
                "seed": int(seed),
                "family": family,
                "candidate": candidate,
                **summary,
                "selected": False,
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            float(row["video_equal_nll"]),
            -float(row["video_equal_accuracy"]),
            float(row["candidate"]),
        ),
    )
    selected["selected"] = True
    return float(selected["candidate"]), rows


def _probe_tape(
    streams: Sequence[Stream51Stream],
    *,
    retention: float,
    fast_retention: float,
    slow_retention: float,
) -> dict[str, np.ndarray]:
    features: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for stream in streams:
        control = causal_control_features(
            stream.evidence,
            stream_ids=stream.stream_ids,
            fast_retention=fast_retention,
            slow_retention=slow_retention,
            observation_log_likelihood=stream.observation_log_likelihood,
        )
        oracle = oracle_write_targets(
            stream.evidence,
            stream.labels,
            stream_ids=stream.stream_ids,
            retention=retention,
        )
        features.append(control.raw_features)
        masses.append(oracle.log_memory_mass[:, None])
        targets.append(oracle.targets)
        groups.append(stream.source_video_ids)
    raw = np.concatenate(features)
    mass = np.concatenate(masses)
    return {
        "three": raw,
        "three_plus_log_mass": np.concatenate([raw, mass], axis=1),
        "targets": np.concatenate(targets),
        "groups": np.concatenate(groups),
    }


def stream51_retention_gate(seed_metrics: pd.DataFrame) -> dict[str, Any]:
    """Apply the protocol's method-specific, seed-level reuse gate."""

    required = {"seed", "condition", "video_equal_nll"}
    if not required <= set(seed_metrics.columns):
        raise ValueError("seed metrics lack required columns")
    pivot = seed_metrics.pivot(
        index="seed", columns="condition", values="video_equal_nll"
    )
    methods: dict[str, Any] = {}
    for method in ("direct_alpha", "likelihood_hmm"):
        if not {"posterior_ema", method} <= set(pivot.columns):
            raise ValueError(f"seed metrics lack {method}")
        gains = pivot["posterior_ema"] - pivot[method]
        methods[method] = {
            "seed_gains": {str(int(key)): float(value) for key, value in gains.items()},
            "n_positive_seeds": int(np.sum(gains > 0.0)),
            "mean_nll_gain": float(np.mean(gains)),
            "passed": bool(
                np.sum(gains > 0.0) >= STREAM51_RETAIN_MIN_SEEDS
                and np.mean(gains) >= STREAM51_RETAIN_MIN_MEAN_NLL_GAIN
            ),
        }
    retained = any(value["passed"] for value in methods.values())
    return {
        "stream51_retained": bool(retained),
        "minimum_positive_seeds": STREAM51_RETAIN_MIN_SEEDS,
        "minimum_mean_nll_gain": STREAM51_RETAIN_MIN_MEAN_NLL_GAIN,
        "methods": methods,
    }


def diagnose(
    config_path: Path, results_root: Path
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Run the frozen revealed-data diagnostic without external access."""

    validate_preregistration(config_path)
    validate_implementation_receipt()
    config = load_json_config(config_path)
    validate_config(config)
    if "external" in str(config["feature_root"]).lower():
        raise ValueError("diagnostic feature root unexpectedly names external data")
    store = _store(config, include_external=False)
    model = fit_stream51_vmf(
        store,
        split="support",
        max_frames_per_video=int(config["encoder"]["support_frames_per_video"]),
    )
    metric_rows: list[dict[str, Any]] = []
    per_video_rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    post_window = int(config["task"]["post_switch_window"])
    fast_retention = float(config["controller_grid"]["fast_retention"])
    slow_retention = float(config["controller_grid"]["slow_retention"])

    for seed in seed_list(config["seeds"]):
        contract = _load_run_contract(results_root, seed=seed)
        selected = contract["selected"]
        partition = contract["partition"]
        temperature = float(selected["temperature"])
        retention = float(selected["retention"])
        fit_keys = tuple(map(str, partition["fit_video_keys"]))
        qualification_keys = tuple(map(str, partition["qualification_video_keys"]))
        fit_natural, fit_hidden = _make_streams(
            store,
            model,
            split="development",
            video_keys=fit_keys,
            temperature=temperature,
            seed=seed,
            task=config["task"],
        )
        qualification_natural, qualification_hidden = _make_streams(
            store,
            model,
            split="development",
            video_keys=qualification_keys,
            temperature=temperature,
            seed=seed,
            task=config["task"],
        )
        del fit_natural, qualification_natural
        alpha, alpha_audit = _select_scalar(
            fit_hidden,
            grid=ALPHA_GRID,
            trace_factory=lambda stream, value: direct_alpha_filter(
                stream.evidence, stream_ids=stream.stream_ids, alpha=value
            ),
            post_switch_window=post_window,
            seed=seed,
            family="direct_alpha",
        )
        hazard, hazard_audit = _select_scalar(
            fit_hidden,
            grid=HAZARD_GRID,
            trace_factory=lambda stream, value: likelihood_hmm_filter(
                stream.observation_log_likelihood,
                stream_ids=stream.stream_ids,
                hazard=value,
                temperature=temperature,
            ),
            post_switch_window=post_window,
            seed=seed,
            family="likelihood_hmm",
        )
        selection_rows.extend(alpha_audit)
        selection_rows.extend(hazard_audit)
        conditions: dict[str, Callable[[Stream51Stream], BeliefTrace]] = {
            "posterior_ema": lambda stream: _posterior_ema(
                stream, retention=retention
            ),
            "direct_alpha": lambda stream: direct_alpha_filter(
                stream.evidence, stream_ids=stream.stream_ids, alpha=alpha
            ),
            "true_switch_direct_alpha": lambda stream: direct_alpha_filter(
                stream.evidence,
                stream_ids=stream.stream_ids,
                alpha=alpha,
                reset_flags=stream.switch_flags,
            ),
            "likelihood_hmm": lambda stream: likelihood_hmm_filter(
                stream.observation_log_likelihood,
                stream_ids=stream.stream_ids,
                hazard=hazard,
                temperature=temperature,
            ),
            "true_switch_likelihood_reset": lambda stream: likelihood_hmm_filter(
                stream.observation_log_likelihood,
                stream_ids=stream.stream_ids,
                hazard=0.0,
                temperature=temperature,
                reset_flags=stream.switch_flags,
            ),
        }
        for condition, factory in conditions.items():
            score, frame = _score_streams(
                qualification_hidden,
                factory,
                post_switch_window=post_window,
                seed=seed,
                condition=condition,
                panel="qualification_hidden",
            )
            metric_rows.append(
                {
                    "seed": int(seed),
                    "condition": condition,
                    "selected_retention": retention,
                    "selected_alpha": alpha,
                    "selected_hazard": hazard,
                    "temperature": temperature,
                    **score,
                }
            )
            per_video_rows.append(frame)

        fit_probe = _probe_tape(
            fit_hidden,
            retention=retention,
            fast_retention=fast_retention,
            slow_retention=slow_retention,
        )
        qualification_probe = _probe_tape(
            qualification_hidden,
            retention=retention,
            fast_retention=fast_retention,
            slow_retention=slow_retention,
        )
        for feature_set in ("three", "three_plus_log_mass"):
            probe = fit_oracle_write_probe(
                fit_probe[feature_set], fit_probe["targets"], seed=seed
            )
            scores = probe.predict_proba(qualification_probe[feature_set])[:, 1]
            probe_rows.append(
                {
                    "seed": int(seed),
                    "feature_set": feature_set,
                    "n_fit_frames": int(len(fit_probe["targets"])),
                    "n_qualification_frames": int(
                        len(qualification_probe["targets"])
                    ),
                    "qualification_write_rate": float(
                        np.mean(qualification_probe["targets"])
                    ),
                    **grouped_binary_metrics(
                        qualification_probe["targets"],
                        scores,
                        group_ids=qualification_probe["groups"],
                    ),
                }
            )

    seed_metrics = pd.DataFrame(metric_rows)
    gate = stream51_retention_gate(seed_metrics)
    probes = pd.DataFrame(probe_rows)
    mass_pivot = probes.pivot(
        index="seed", columns="feature_set", values="video_equal_auc"
    )
    mass_auc_gain = (
        mass_pivot["three_plus_log_mass"] - mass_pivot["three"]
    )
    summary = {
        "analysis_status": "post_hoc_revealed_qualification_diagnostic_only",
        "claim_upgrade_allowed": False,
        "external_features_accessed": False,
        "external_outcomes_accessed": False,
        "seeds": list(seed_list(config["seeds"])),
        "alpha_grid": list(ALPHA_GRID),
        "hazard_grid": list(HAZARD_GRID),
        "stream51_reuse_gate": gate,
        "oracle_write_probe": {
            "mean_three_feature_auc": float(
                probes.loc[
                    probes["feature_set"] == "three", "video_equal_auc"
                ].mean()
            ),
            "mean_three_plus_mass_auc": float(
                probes.loc[
                    probes["feature_set"] == "three_plus_log_mass",
                    "video_equal_auc",
                ].mean()
            ),
            "mean_mass_auc_gain": float(mass_auc_gain.mean()),
        },
        "factorized_h_q_r_claim": "inconclusive_not_tested",
        "interpretation": (
            "This diagnostic can retire or retain the revealed Stream-51 task "
            "for development. It cannot validate h/Q/R or unlock external data."
        ),
    }
    return summary, {
        "seed_metrics": seed_metrics,
        "per_video_metrics": pd.concat(per_video_rows, ignore_index=True),
        "selection_audit": pd.DataFrame(selection_rows),
        "probe_metrics": probes,
    }


def _report(summary: Mapping[str, Any], frames: Mapping[str, pd.DataFrame]) -> str:
    gate = summary["stream51_reuse_gate"]
    probes = summary["oracle_write_probe"]
    metrics = frames["seed_metrics"]
    condition_means = metrics.groupby("condition")[
        ["video_equal_nll", "video_equal_accuracy", "video_equal_post_switch_nll"]
    ].mean()
    lines = [
        "# Exp38 Post-Hoc Factorized-Memory Diagnostic",
        "",
        "Status: **claim-ineligible revealed-data diagnostic**.",
        "External Stream-51 features/outcomes accessed: **no**.",
        "",
        "## Result",
        "",
        (
            "The Stream-51 development task was **retained** for further method "
            "development."
            if gate["stream51_retained"]
            else "The Stream-51 splice task was **retired** for the successor method."
        ),
        "",
        "| Condition | Mean video-equal NLL | Mean accuracy | Mean post-switch NLL |",
        "|---|---:|---:|---:|",
    ]
    for condition, row in condition_means.iterrows():
        lines.append(
            f"| {condition} | {row['video_equal_nll']:.6f} | "
            f"{row['video_equal_accuracy']:.6f} | "
            f"{row['video_equal_post_switch_nll']:.6f} |"
        )
    lines.extend(["", "## Registered task-reuse gate", ""])
    for method, result in gate["methods"].items():
        lines.append(
            f"- `{method}`: mean NLL gain {result['mean_nll_gain']:.6f}; "
            f"positive in {result['n_positive_seeds']}/5 seeds; "
            f"gate {'PASS' if result['passed'] else 'FAIL'}."
        )
    lines.extend(
        [
            "",
            "## Oracle-write reachability",
            "",
            f"- Three causal statistics: mean video-equal AUC "
            f"{probes['mean_three_feature_auc']:.6f}.",
            f"- Adding log memory mass: mean AUC "
            f"{probes['mean_three_plus_mass_auc']:.6f}; gain "
            f"{probes['mean_mass_auc_gain']:.6f}.",
            "",
            "These probes use label-revealed oracle targets and cannot become a "
            "deployable or confirmatory result.",
            "",
            "## Conclusion",
            "",
            "The factorized h/Q/R claim remains **inconclusive/not tested**. "
            "Stream-51 has no orthogonal observation-noise and drift manipulation. "
            "Only a separately frozen synthetic factorial can test identifiability.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prospective/exp38_stream51_soft_memory.json"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/exp38_stream51_soft_memory_prospective_v1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    summary, frames = diagnose(
        args.config.expanduser().resolve(), args.results_root.expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostic_config.json").write_text(
        json.dumps(
            {
                "analysis_status": summary["analysis_status"],
                "claim_upgrade_allowed": False,
                "external_access_allowed": False,
                "alpha_grid": list(ALPHA_GRID),
                "hazard_grid": list(HAZARD_GRID),
                "stream51_reuse_gate": {
                    "minimum_positive_seeds": STREAM51_RETAIN_MIN_SEEDS,
                    "minimum_mean_nll_gain": STREAM51_RETAIN_MIN_MEAN_NLL_GAIN,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "report.md").write_text(
        _report(summary, frames), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
