"""Reanalyse the immutable Exp23 formal-v2 archive without rerunning seeds."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.hidden_selector_metrics import paired_bootstrap_interval
from scripts.summarize_exp23_exp25 import _exp23_mechanism_audit_ready


EXPECTED_ARCHIVE_SHA256 = (
    "20f0ef9229ddbedadf634506137df5da4f73c48e7962c934fb6d600e4cb33c40"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG_PATH = (
    PROJECT_ROOT / "configs/formal/exp23_closed_loop_local_controller.json"
)
EXPECTED_FORMAL_CONFIG_SHA256 = (
    "c708be09a4fd83541fb0b20db97966ed37f56ecdd64e6ee966d10bf20f1e9966"
)
TASKS = ("current", "delayed")
CONDITIONS = (
    "frozen",
    "local_eprop",
    "exact_forward_sensitivity",
    "bptt_axis_only",
    "random_update",
    "current_off_policy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_lf(path: Path, text: str) -> None:
    """Write portable, hash-stable UTF-8 text without OS newline translation."""

    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def _write_csv_lf(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n")


def load_archive(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha = _sha256(path)
    if observed_sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            "Exp23 probe archive SHA-256 does not match the frozen source"
        )
    rows: list[dict[str, Any]] = []
    metric_members = 0
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith("/metrics.jsonl"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read {member.name}")
            metric_members += 1
            text = io.TextIOWrapper(stream, encoding="utf-8")
            rows.extend(json.loads(line) for line in text if line.strip())
    raw = pd.DataFrame(rows)
    if metric_members != 30 or len(raw) != 360:
        raise RuntimeError(
            f"Exp23 archive contains {metric_members} metric files and {len(raw)} rows; "
            "expected 30 and 360"
        )
    expected = {
        (seed, task, condition)
        for seed in range(30)
        for task in TASKS
        for condition in CONDITIONS
    }
    observed = set(
        zip(
            raw["seed"].astype(int),
            raw["task_variant"].astype(str),
            raw["condition"].astype(str),
            strict=True,
        )
    )
    if observed != expected or len(raw) != len(observed):
        raise RuntimeError("Exp23 archive is not the complete paired 30-seed grid")
    if not raw["status"].eq("complete").all():
        raise RuntimeError("Exp23 frozen archive includes non-complete rows")
    return raw


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.Series(
        np.divide(
            numerator.to_numpy(float),
            denominator.to_numpy(float),
            out=np.full(len(numerator), np.nan, dtype=np.float64),
            where=np.abs(denominator.to_numpy(float)) > 0.0,
        ),
        index=numerator.index,
    )


def _bootstrap(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    return paired_bootstrap_interval(values, n_resamples=100_000, seed=seed)


def _formal_eligibility_and_oppose(
    raw: pd.DataFrame,
) -> tuple[bool, bool, dict[str, Any]]:
    """Reverify the archived formal gates and one registered oppose component."""

    if (
        not FORMAL_CONFIG_PATH.is_file()
        or _sha256(FORMAL_CONFIG_PATH) != EXPECTED_FORMAL_CONFIG_SHA256
    ):
        return False, False, {"failure": "frozen formal config hash mismatch"}
    required = {
        "functional_budget_satisfied",
        "gate_moment_anchor_identifiable",
        "gate_mean_absolute_signed_belief_dev",
        "random_tape_id",
        "split_id",
        "network_init_id",
        "gate_checkpoint_id",
        "readout_checkpoint_id",
        "readout_fit_data_id",
        "pairing_bundle_id",
    }
    if not required <= set(raw):
        return False, False, {"failure": "archive lacks formal audit receipts"}
    config = json.loads(FORMAL_CONFIG_PATH.read_text(encoding="utf-8"))
    planned = tuple(int(value) for value in config["seeds"])
    thresholds = dict(config["registered_claim_thresholds"])
    task_audits: dict[str, Any] = {}
    ready_all = True
    opposed_any = False
    for task_index, task in enumerate(TASKS):
        selected = raw[raw["task_variant"].eq(task)].copy()
        complete_grid = bool(
            len(selected) == len(planned) * len(CONDITIONS)
            and selected["status"].eq("complete").all()
            and set(selected["seed"].astype(int)) == set(planned)
        )
        nonfrozen = selected[selected["condition"].ne("frozen")]
        budget_ready = bool(
            len(nonfrozen) == len(planned) * (len(CONDITIONS) - 1)
            and nonfrozen["functional_budget_satisfied"].map(bool).all()
        )
        local = selected[selected["condition"].eq("local_eprop")]
        gate_ready = bool(
            len(local) == len(planned)
            and float(np.mean(local["gate_moment_anchor_identifiable"].map(bool)))
            >= float(thresholds["minimum_gate_identifiable_fraction"])
            and pd.to_numeric(
                local["gate_mean_absolute_signed_belief_dev"], errors="coerce"
            )
            .notna()
            .all()
            and float(
                pd.to_numeric(
                    local["gate_mean_absolute_signed_belief_dev"], errors="coerce"
                ).min()
            )
            >= float(thresholds["minimum_mean_absolute_signed_belief"])
        )
        mechanism_ready = _exp23_mechanism_audit_ready(selected, planned)
        task_ready = bool(
            complete_grid
            and budget_ready
            and gate_ready
            and mechanism_ready
            and len(planned) >= int(thresholds["minimum_formal_seeds"])
        )
        pivot = selected.pivot(
            index="seed", columns="condition", values="behavior_balanced_accuracy"
        ).reindex(planned)
        gain_vs_random = (pivot["local_eprop"] - pivot["random_update"]).to_numpy(float)
        ci = _bootstrap(gain_vs_random, seed=23200 + task_index)
        opposed = bool(
            task_ready and ci[1] < float(thresholds["minimum_local_gain_vs_random"])
        )
        ready_all = ready_all and task_ready
        opposed_any = opposed_any or opposed
        task_audits[task] = {
            "formal_ready": task_ready,
            "complete_grid": complete_grid,
            "functional_budget_ready": budget_ready,
            "gate_ready": gate_ready,
            "mechanism_audit_ready": mechanism_ready,
            "local_minus_random_mean": float(np.mean(gain_vs_random)),
            "local_minus_random_ci": list(ci),
            "registered_threshold": float(thresholds["minimum_local_gain_vs_random"]),
            "registered_component_opposed": opposed,
        }
    return ready_all, opposed_any, task_audits


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = raw.copy()
    work["axis_rescale_ratio"] = _safe_ratio(
        work["control_axis_l2"], work["natural_control_axis_l2"]
    )
    conditions = (
        work.groupby(["task_variant", "condition"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_matched_ba=("behavior_balanced_accuracy", "mean"),
            mean_natural_ba=("natural_behavior_balanced_accuracy", "mean"),
            mean_test_task_loss=("test_task_loss", "mean"),
            mean_matched_axis_l2=("control_axis_l2", "mean"),
            mean_natural_axis_l2=("natural_control_axis_l2", "mean"),
            median_axis_rescale_ratio=("axis_rescale_ratio", "median"),
            min_axis_rescale_ratio=("axis_rescale_ratio", "min"),
            max_axis_rescale_ratio=("axis_rescale_ratio", "max"),
            mean_update_cosine=("median_update_cosine_to_exact", "mean"),
        )
        .sort_values(["task_variant", "condition"])
        .reset_index(drop=True)
    )
    seed_rows: list[dict[str, Any]] = []
    task_summary: dict[str, Any] = {}
    for task_index, task in enumerate(TASKS):
        frame = work[work["task_variant"] == task].set_index(["seed", "condition"])

        def values(column: str, condition: str) -> np.ndarray:
            return (
                frame.xs(condition, level="condition")[column]
                .sort_index()
                .to_numpy(float)
            )

        matched_frozen = values("behavior_balanced_accuracy", "frozen")
        natural_frozen = values("natural_behavior_balanced_accuracy", "frozen")
        loss_frozen = values("test_task_loss", "frozen")
        gains: dict[str, np.ndarray] = {}
        natural_gains: dict[str, np.ndarray] = {}
        for condition in (
            "local_eprop",
            "exact_forward_sensitivity",
            "bptt_axis_only",
            "random_update",
        ):
            gains[condition] = (
                values("behavior_balanced_accuracy", condition) - matched_frozen
            )
            natural_gains[condition] = (
                values("natural_behavior_balanced_accuracy", condition) - natural_frozen
            )
        local_loss = values("test_task_loss", "local_eprop") - loss_frozen
        local_cosine = values("median_update_cosine_to_exact", "local_eprop")
        cosine_rank = pd.Series(local_cosine).rank()
        gain_rank = pd.Series(gains["local_eprop"]).rank()
        cosine_gain_spearman: float | None = (
            float(cosine_rank.corr(gain_rank))
            if cosine_rank.nunique(dropna=True) > 1
            and gain_rank.nunique(dropna=True) > 1
            else None
        )
        for seed in range(30):
            seed_rows.append(
                {
                    "seed": seed,
                    "task_variant": task,
                    "local_matched_gain": gains["local_eprop"][seed],
                    "local_natural_gain": natural_gains["local_eprop"][seed],
                    "exact_matched_gain": gains["exact_forward_sensitivity"][seed],
                    "exact_natural_gain": natural_gains["exact_forward_sensitivity"][
                        seed
                    ],
                    "bptt_matched_gain": gains["bptt_axis_only"][seed],
                    "bptt_natural_gain": natural_gains["bptt_axis_only"][seed],
                    "local_test_loss_change": local_loss[seed],
                    "local_update_cosine": local_cosine[seed],
                }
            )
        local_ci = _bootstrap(gains["local_eprop"], seed=23000 + task_index)
        bptt_natural_ci = _bootstrap(
            natural_gains["bptt_axis_only"], seed=23100 + task_index
        )
        exact_natural_ci = _bootstrap(
            natural_gains["exact_forward_sensitivity"], seed=23150 + task_index
        )
        task_summary[task] = {
            "frozen_balanced_accuracy": float(np.mean(matched_frozen)),
            "local_matched_gain": float(np.mean(gains["local_eprop"])),
            "local_matched_gain_ci": list(local_ci),
            "local_natural_gain": float(np.mean(natural_gains["local_eprop"])),
            "local_natural_positive_seed_fraction": float(
                np.mean(natural_gains["local_eprop"] > 0.0)
            ),
            "exact_matched_gain": float(np.mean(gains["exact_forward_sensitivity"])),
            "exact_natural_gain": float(
                np.mean(natural_gains["exact_forward_sensitivity"])
            ),
            "exact_natural_gain_ci": list(exact_natural_ci),
            "bptt_matched_gain": float(np.mean(gains["bptt_axis_only"])),
            "bptt_natural_gain": float(np.mean(natural_gains["bptt_axis_only"])),
            "bptt_natural_gain_ci": list(bptt_natural_ci),
            "bptt_natural_ci_upper_below_registered_mcid": bool(
                bptt_natural_ci[1] < 0.03
            ),
            "local_test_loss_change": float(np.mean(local_loss)),
            "cosine_gain_spearman": cosine_gain_spearman,
        }
        for condition in (
            "local_eprop",
            "exact_forward_sensitivity",
            "bptt_axis_only",
        ):
            ratios = work[
                (work["task_variant"] == task) & (work["condition"] == condition)
            ]["axis_rescale_ratio"].to_numpy(float)
            task_summary[task][f"{condition}_median_axis_rescale_ratio"] = float(
                np.nanmedian(ratios)
            )
            task_summary[task][f"{condition}_min_axis_rescale_ratio"] = float(
                np.nanmin(ratios)
            )
            task_summary[task][f"{condition}_max_axis_rescale_ratio"] = float(
                np.nanmax(ratios)
            )
    formal_ready, registered_component_opposed, formal_audit = (
        _formal_eligibility_and_oppose(raw)
    )
    seeds = pd.DataFrame(seed_rows)
    summary = {
        "experiment": "exp23_failure_probe",
        "source_experiment": "exp23_closed_loop_local_controller",
        "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "source_rows": int(len(raw)),
        "statistics_unit": "seed",
        "n_seeds": 30,
        "registered_mcid": 0.03,
        "formal_eligibility_reverified": formal_ready,
        "registered_opposed_component_reverified": registered_component_opposed,
        "formal_audit": formal_audit,
        "existing_claim_remains": (
            "oppose_current_drive_gain_axis_rule_and_budget_combination"
            if formal_ready and registered_component_opposed
            else "inconclusive_original_oppose_not_reestablished_by_probe"
        ),
        "probe_classification": "limited_tested_axis_headroom_and_matched_vs_natural_rescale_sensitive",
        "general_local_learning_conclusion": "inconclusive",
        "task_summary": task_summary,
    }
    return conditions, seeds, summary


def _report(summary: dict[str, Any]) -> str:
    current = summary["task_summary"]["current"]
    delayed = summary["task_summary"]["delayed"]
    return "\n".join(
        [
            "# Exp23 failure probe",
            "",
            "This is a fixed reanalysis of the immutable 30-seed formal-v2 archive;",
            "no seed, scale or condition was rerun or selected.",
            "",
            f"- Current local matched gain: {current['local_matched_gain']:+.5f}; natural gain: {current['local_natural_gain']:+.5f}.",
            f"- Delayed local matched gain: {delayed['local_matched_gain']:+.5f}; natural gain: {delayed['local_natural_gain']:+.5f}.",
            f"- Delayed BPTT natural gain: {delayed['bptt_natural_gain']:+.5f}, 95% seed-bootstrap CI [{delayed['bptt_natural_gain_ci'][0]:+.5f}, {delayed['bptt_natural_gain_ci'][1]:+.5f}].",
            f"- Delayed exact-forward natural gain: {delayed['exact_natural_gain']:+.5f}, 95% seed-bootstrap CI [{delayed['exact_natural_gain_ci'][0]:+.5f}, {delayed['exact_natural_gain_ci'][1]:+.5f}].",
            f"- Delayed local median matched/natural axis ratio: {delayed['local_eprop_median_axis_rescale_ratio']:.1f}x.",
            f"- Current local task-loss change: {current['local_test_loss_change']:+.5f} while balanced accuracy did not improve.",
            "",
            f"Formal eligibility reverified: {summary['formal_eligibility_reverified']}; registered oppose component reverified: {summary['registered_opposed_component_reverified']}.",
            "The original `oppose` result remains valid only when both audits above",
            "are true, and only for the registered drive-gain",
            "axis, local rule and state-displacement protocol.  It should not be",
            "generalized to all e-prop or local plasticity.  In this post-hoc",
            "diagnostic, the tested natural BPTT mean CI upper bound remained below",
            "the local rule's registered 0.03 MCID, suggesting limited headroom for",
            "this axis, optimizer and protocol rather than proving an impossibility.",
            "Post-training state-displacement matching strongly amplified",
            "the weakest delayed local direction.  A future probe must use an ex-ante",
            "direction-by-scale curve and separately match cumulative plasticity L1/L2.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", default="results/exp23_failure_probe")
    args = parser.parse_args()
    archive = Path(args.archive)
    raw = load_archive(archive)
    conditions, seeds, summary = summarize(raw)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    packaged_archive = output / "source_formal_v2_archive.tar.gz"
    packaged_config = output / "source_formal_config.json"
    shutil.copy2(archive, packaged_archive)
    shutil.copy2(FORMAL_CONFIG_PATH, packaged_config)
    _write_csv_lf(conditions, output / "condition_summary.csv")
    _write_csv_lf(seeds, output / "seed_contrasts.csv")
    _write_text_lf(
        output / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text_lf(output / "report.md", _report(summary))
    _write_text_lf(
        output / "source_provenance.json",
        json.dumps(
            {
                "source_archive_package_path": packaged_archive.name,
                "source_archive_location_informational": str(archive.resolve()),
                "source_archive_sha256": _sha256(archive),
                "expected_archive_sha256": EXPECTED_ARCHIVE_SHA256,
                "metric_rows": len(raw),
                "metric_seed_count": int(raw["seed"].nunique()),
                "analysis_script": str(Path(__file__).resolve()),
                "analysis_script_sha256": _sha256(Path(__file__)),
                "formal_config_package_path": packaged_config.name,
                "formal_config_location_informational": str(
                    FORMAL_CONFIG_PATH.resolve()
                ),
                "formal_config_sha256": _sha256(FORMAL_CONFIG_PATH),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
