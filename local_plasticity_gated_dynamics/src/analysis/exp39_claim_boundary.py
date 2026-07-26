"""Post-hoc claim-boundary diagnostics for the frozen Exp39 result.

This module only reads already-produced block and selection tables.  It never
reruns, changes, or selects the Exp39 controller.  Every aggregate keeps the
formal seed as the independent unit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FACTORS = ("h", "q", "r")
HELDOUT_CELLS = ("011", "101", "110", "111")
UTILITY_BASELINES = ("selected_fixed", "seen_mode_imm")
ORACLE_METHODS = ("oracle_factorial_imm", "oracle_dynamic")


def _strict_boolean(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{name} must contain only true/false values")
    return normalized.eq("true")


def validate_block_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the frozen block-level diagnostic table."""

    required = {
        "seed",
        "method",
        "sequence_id",
        "block_id",
        "cell",
        "h_high",
        "q_high",
        "r_high",
        "heldout_composition",
        "mean_nll",
        "latent_mse",
        "early_nll",
        "late_nll",
        "mean_h_estimate",
        "mean_q_estimate",
        "mean_r_estimate",
        "true_h",
        "true_q",
        "true_r",
        "test_tape_digest",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"block metrics are missing {sorted(missing)}")
    if frame.empty:
        raise ValueError("block metrics must be non-empty")
    result = frame.copy()
    result["cell"] = result["cell"].astype(str).str.zfill(3)
    if not result["cell"].str.fullmatch(r"[01]{3}").all():
        raise ValueError("cell must be a three-bit code")
    result["heldout_composition"] = _strict_boolean(
        result["heldout_composition"], name="heldout_composition"
    )
    expected_heldout = result["cell"].isin(HELDOUT_CELLS)
    if not result["heldout_composition"].equals(expected_heldout):
        raise ValueError("heldout labels disagree with the registered cell split")
    bit_columns = ("h_high", "q_high", "r_high")
    expected_bits = np.asarray(
        [[int(bit) for bit in cell] for cell in result["cell"]], dtype=np.int64
    )
    actual_bits = result.loc[:, bit_columns].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_bits, expected_bits):
        raise ValueError("factor indicators disagree with cell labels")
    numeric = (
        "mean_nll",
        "latent_mse",
        "early_nll",
        "late_nll",
        "mean_h_estimate",
        "mean_q_estimate",
        "mean_r_estimate",
        "true_h",
        "true_q",
        "true_r",
    )
    values = result.loc[:, numeric].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("diagnostic metrics must be finite")
    if np.any(result[[f"mean_{name}_estimate" for name in FACTORS]] <= 0.0):
        raise ValueError("parameter estimates must be positive before log analysis")
    if np.any(result[[f"true_{name}" for name in FACTORS]] <= 0.0):
        raise ValueError("true parameters must be positive before log analysis")
    key = ["seed", "method", "block_id"]
    if result.duplicated(key).any():
        raise ValueError("seed/method/block rows must be unique")
    if not result.groupby("seed")["test_tape_digest"].nunique().eq(1).all():
        raise ValueError("all methods must share one test tape within each seed")
    return result


def cellwise_utility(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return seed-level and cell-level NLL gains on unseen compositions."""

    blocks = validate_block_metrics(frame)
    methods = ("factorized", *UTILITY_BASELINES)
    heldout = blocks.loc[
        blocks["heldout_composition"] & blocks["method"].isin(methods)
    ]
    grouped = (
        heldout.groupby(["seed", "cell", "method"], as_index=False)["mean_nll"]
        .mean()
        .pivot(index=["seed", "cell"], columns="method", values="mean_nll")
    )
    if grouped.isna().any().any():
        raise ValueError("cell-wise method coverage is incomplete")
    if set(grouped.index.get_level_values("cell")) != set(HELDOUT_CELLS):
        raise ValueError("held-out cell coverage is incomplete")
    rows: list[dict[str, Any]] = []
    for baseline in UTILITY_BASELINES:
        gain = grouped[baseline] - grouped["factorized"]
        for (seed, cell), value in gain.items():
            rows.append(
                {
                    "seed": int(seed),
                    "cell": str(cell),
                    "comparison": f"{baseline}_minus_factorized_nll",
                    "nll_gain": float(value),
                }
            )
    seed_level = pd.DataFrame(rows).sort_values(
        ["comparison", "cell", "seed"], kind="stable", ignore_index=True
    )
    summary = (
        seed_level.groupby(["comparison", "cell"], as_index=False)
        .agg(
            mean_nll_gain=("nll_gain", "mean"),
            positive_seeds=("nll_gain", lambda value: int(np.sum(value > 0.0))),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["comparison", "cell"], kind="stable", ignore_index=True)
    )
    return seed_level, summary


def cross_loading(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regress each log controller coordinate on all three true factor bits."""

    blocks = validate_block_metrics(frame)
    factorized = blocks.loc[blocks["method"] == "factorized"]
    rows: list[dict[str, Any]] = []
    design_columns = [f"{factor}_high" for factor in FACTORS]
    for seed, seed_frame in factorized.groupby("seed", sort=True):
        design = np.column_stack(
            [
                np.ones(len(seed_frame), dtype=np.float64),
                seed_frame[design_columns].to_numpy(dtype=np.float64),
            ]
        )
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError(f"seed {seed} lacks a full-rank factorial design")
        for estimated_factor in FACTORS:
            response = np.log(
                seed_frame[f"mean_{estimated_factor}_estimate"].to_numpy(float)
            )
            coefficients = np.linalg.lstsq(design, response, rcond=None)[0][1:]
            for true_factor, value in zip(FACTORS, coefficients, strict=True):
                rows.append(
                    {
                        "seed": int(seed),
                        "estimated_factor": estimated_factor,
                        "true_factor": true_factor,
                        "log_response": float(value),
                        "is_diagonal": estimated_factor == true_factor,
                    }
                )
    seed_level = pd.DataFrame(rows).sort_values(
        ["estimated_factor", "true_factor", "seed"],
        kind="stable",
        ignore_index=True,
    )
    summaries: list[dict[str, Any]] = []
    for estimated_factor, estimate_frame in seed_level.groupby(
        "estimated_factor", sort=True
    ):
        means = estimate_frame.groupby("true_factor")["log_response"].mean()
        diagonal = abs(float(means.loc[estimated_factor]))
        largest_off_diagonal = float(
            means.drop(index=estimated_factor).abs().max()
        )
        for true_factor in FACTORS:
            values = estimate_frame.loc[
                estimate_frame["true_factor"] == true_factor, "log_response"
            ]
            summaries.append(
                {
                    "estimated_factor": str(estimated_factor),
                    "true_factor": true_factor,
                    "mean_log_response": float(values.mean()),
                    "sd_log_response": float(values.std(ddof=1)),
                    "diagonal_margin": diagonal - largest_off_diagonal,
                    "diagonal_exceeds_all_off_diagonal": bool(
                        diagonal > largest_off_diagonal
                    ),
                    "n_seeds": int(values.size),
                }
            )
    return seed_level, pd.DataFrame(summaries)


def timing_utility(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize frozen early/late NLL, disclosing sequence initialization."""

    blocks = validate_block_metrics(frame)
    methods = ("factorized", *UTILITY_BASELINES)
    relevant = blocks.loc[
        blocks["method"].isin(methods)
    ].copy()
    relevant["sequence_initial_block"] = relevant["block_id"].eq(
        relevant.groupby(["seed", "method", "sequence_id"])[
            "block_id"
        ].transform("min")
    )
    heldout = relevant.loc[
        relevant["heldout_composition"]
    ]
    rows: list[dict[str, Any]] = []
    panels = {
        "all_blocks_including_sequence_initialization": heldout,
        "transition_blocks_only": heldout.loc[
            ~heldout["sequence_initial_block"]
        ],
    }
    for panel, panel_frame in panels.items():
        grouped = panel_frame.groupby(["seed", "method"])[
            ["early_nll", "late_nll"]
        ].mean()
        for baseline in UTILITY_BASELINES:
            baseline_values = grouped.xs(baseline, level="method")
            factorized = grouped.xs("factorized", level="method")
            if not baseline_values.index.equals(factorized.index):
                raise ValueError("timing comparison seed coverage is not paired")
            for endpoint in ("early_nll", "late_nll"):
                gain = baseline_values[endpoint] - factorized[endpoint]
                rows.append(
                    {
                        "panel": panel,
                        "comparison": f"{baseline}_minus_factorized_nll",
                        "endpoint": endpoint,
                        "mean_nll_gain": float(gain.mean()),
                        "positive_seeds": int(np.sum(gain > 0.0)),
                        "n_seeds": int(gain.size),
                    }
                )
    return pd.DataFrame(rows)


def headroom_retention(frame: pd.DataFrame) -> dict[str, float]:
    """Compute aggregate oracle-headroom fractions without per-block inference."""

    blocks = validate_block_metrics(frame)
    methods = ("factorized", "selected_fixed", "seen_mode_imm", *ORACLE_METHODS)
    heldout = blocks.loc[
        blocks["heldout_composition"] & blocks["method"].isin(methods)
    ]
    seed_method = heldout.groupby(["seed", "method"])["mean_nll"].mean().unstack()
    if seed_method[list(methods)].isna().any().any():
        raise ValueError("oracle headroom method coverage is incomplete")
    result: dict[str, float] = {}
    for oracle in ORACLE_METHODS:
        fixed_numerator = float(
            (seed_method["selected_fixed"] - seed_method["factorized"]).mean()
        )
        fixed_denominator = float(
            (seed_method["selected_fixed"] - seed_method[oracle]).mean()
        )
        seen_numerator = float(
            (seed_method["seen_mode_imm"] - seed_method["factorized"]).mean()
        )
        seen_denominator = float(
            (seed_method["seen_mode_imm"] - seed_method[oracle]).mean()
        )
        if fixed_denominator <= 0.0 or seen_denominator <= 0.0:
            raise ValueError("oracle does not define positive headroom")
        result[f"fixed_to_{oracle}_headroom_retained"] = (
            fixed_numerator / fixed_denominator
        )
        result[f"seen_imm_to_{oracle}_gap_closed"] = (
            seen_numerator / seen_denominator
        )
    return result


def selected_timescales(selection_frame: pd.DataFrame) -> pd.DataFrame:
    """Count the formally selected factorized adaptation rates."""

    required = {
        "seed",
        "selection_family",
        "selected",
        "candidate_h",
        "candidate_q",
        "candidate_r",
    }
    missing = required - set(selection_frame.columns)
    if missing:
        raise ValueError(f"selection audit is missing {sorted(missing)}")
    selected = _strict_boolean(selection_frame["selected"], name="selected")
    factorized = selection_frame.loc[
        selection_frame["selection_family"].eq("factorized") & selected
    ].copy()
    if factorized.empty or not factorized.groupby("seed").size().eq(1).all():
        raise ValueError("each seed must have one selected factorized candidate")
    rows: list[dict[str, Any]] = []
    for factor, column in (
        ("h", "candidate_h"),
        ("q", "candidate_q"),
        ("r", "candidate_r"),
    ):
        counts = factorized[column].astype(float).value_counts().sort_index()
        for rate, count in counts.items():
            rows.append(
                {
                    "factor": factor,
                    "adaptation_rate": float(rate),
                    "approximate_steps": float(1.0 / rate),
                    "selected_seeds": int(count),
                    "n_seeds": int(factorized["seed"].nunique()),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "FACTORS",
    "HELDOUT_CELLS",
    "cellwise_utility",
    "cross_loading",
    "headroom_retention",
    "selected_timescales",
    "timing_utility",
    "validate_block_metrics",
]
