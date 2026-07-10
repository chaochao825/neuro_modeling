"""Aggregate immutable runs into raw_metrics.csv, summary.csv, and report.md."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.claims import evaluate_core_claims  # noqa: E402


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def collect_runs(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    runs = []
    for status_path in sorted((results_root / "runs").glob("**/status.json")):
        run_dir = status_path.parent
        status = json.loads(status_path.read_text(encoding="utf-8"))
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        run_status = status.get("status", manifest.get("status", "unknown"))
        # Older interrupted artifacts predate persisted start timestamps.  The
        # immutable run directory begins with the same UTC timestamp, so it is
        # a stable fallback for latest-attempt ordering.
        run_started_at = (
            status.get("started_at")
            or manifest.get("started_at")
            or run_dir.name.split("_", maxsplit=1)[0]
        )
        planned_path = run_dir / "planned_conditions.json"
        n_planned = len(json.loads(planned_path.read_text(encoding="utf-8"))) if planned_path.exists() else 0
        runs.append(
            {
                "run_id": manifest.get("run_id"),
                "experiment": config.get("experiment"),
                "seed": config.get("seed"),
                "profile": config.get("profile", "unspecified"),
                "status": run_status,
                "started_at": run_started_at,
                "ended_at": status.get("ended_at", manifest.get("ended_at")),
                "n_planned": n_planned,
                "condition_failures": status.get("condition_failures", 0),
                "condition_invalid": status.get("condition_invalid", 0),
                "path": str(run_dir.resolve()),
            }
        )
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                record.setdefault("profile", config.get("profile", "unspecified"))
                record.setdefault("run_path", str(run_dir.resolve()))
                record.setdefault("run_status", run_status)
                record.setdefault("run_started_at", run_started_at)
                records.append({key: _csv_value(value) for key, value in record.items()})
        # A top-level failure or an interrupted nonterminal run may occur after
        # some conditions were streamed.  Materialize the run state so an
        # empty latest attempt cannot silently fall back to an older success;
        # claims additionally invalidate every row sharing this run_id.
        if run_status not in {"complete", "complete_with_failures"}:
            run_failure = {
                "run_id": manifest.get("run_id"),
                "experiment": config.get("experiment"),
                "seed": config.get("seed"),
                "recorded_at": status.get("ended_at") or run_started_at,
                "profile": config.get("profile", "unspecified"),
                "run_path": str(run_dir.resolve()),
                "run_status": run_status,
                "run_started_at": run_started_at,
                "status": "failed",
                "error_type": status.get(
                    "error_type",
                    "IncompleteRun" if run_status == "running" else "RunFailure",
                ),
                "error": status.get(
                    "error",
                    "nonterminal run artifact" if run_status == "running" else "top-level run failure",
                ),
                "run_level_failure": True,
            }
            records.append(
                {key: _csv_value(value) for key, value in run_failure.items()}
            )
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(runs)


def _format_number(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)


def write_report(results_root: Path, raw: pd.DataFrame, runs: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Local Plasticity to Gated Low-Dimensional Dynamics",
        "",
        "This report is generated from immutable run artifacts. Failed and invalid conditions are included; only formal-profile independent units can support or oppose a core claim.",
        "",
        "## Run coverage (all immutable attempts)",
        "",
        "Retries and interrupted attempts remain listed here. These are attempt counts, not unique-seed coverage; core-claim sample sizes use only the latest formal attempt for each experiment and seed.",
        "",
        "| Experiment | Profile | Attempts | Clean complete | Complete with failures | Failed/partial | Planned attempt-cells |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if runs.empty:
        lines.append("| none | — | 0 | 0 | 0 | 0 | 0 |")
    else:
        for (experiment, profile), group in runs.groupby(["experiment", "profile"], dropna=False):
            clean = int(group["status"].eq("complete").sum())
            with_failures = int(group["status"].eq("complete_with_failures").sum())
            failed = int(
                (~group["status"].isin(["complete", "complete_with_failures"])).sum()
            )
            lines.append(
                f"| {experiment} | {profile} | {len(group)} | {clean} | {with_failures} | {failed} | {int(group['n_planned'].sum())} |"
            )
    lines += [
        "",
        "## Core proposition audit",
        "",
        "| Claim | Criterion | n complete/planned | Estimate [95% CI] | Conclusion |",
        "|---|---|---:|---:|---|",
    ]
    for row in summary.to_dict("records"):
        interval = f"{_format_number(row['estimate'])} [{_format_number(row['ci_low'])}, {_format_number(row['ci_high'])}]"
        lines.append(
            f"| {row['claim_id']} | {row['criterion']} | {row['n_complete']}/{row['n_planned']} | {interval} | **{row['conclusion']}** |"
        )
    lines += ["", "### Evidence details", ""]
    for row in summary.to_dict("records"):
        note = str(row.get("note") or "—").replace("\n", " ")
        lines.append(f"- `{row['claim_id']}` (failed={row['n_failed']}): {note}")
    lines += [
        "",
        "## Interpretation safeguards",
        "",
        "- BPTT is isolated as a performance baseline; local-learning models do not import autograd or optimizers and cannot load BPTT checkpoints.",
        "- PCA, normalization, nuisance regression, subspaces, and dynamics are fit on training trials/blocks only.",
        "- Time points never cross trial/block splits. Symmetric smoothing is visualization-only; predictive likelihood uses causal smoothing/raw counts.",
        "- Inference units are seeds, sessions, or animals. Neurons are never treated as independent replicates.",
        "- IBL latent/behavior lead–lag is descriptive system-level evidence and is not interpreted as causal gating.",
        "",
        "## External-data status",
        "",
        "The referenced Zenodo sequence-memory record currently reports `access_right=restricted`. Missing access is retained as a failed session-level artifact and makes the corresponding claims inconclusive; it is never replaced by synthetic evidence.",
        "",
        "## Generated artifacts",
        "",
        "- `results/raw_metrics.csv`: every raw metric row, including failed and invalid conditions.",
        "- `results/runs.csv`: run status and planned-cell coverage.",
        "- `results/summary.csv`: one row per pre-registered core claim.",
        "- `results/core_results.pdf` and `results/phase_models.pdf`: script-generated data figures when applicable.",
        "",
    ]
    (results_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    raw, runs = collect_runs(results_root)
    raw.to_csv(results_root / "raw_metrics.csv", index=False)
    runs.to_csv(results_root / "runs.csv", index=False)
    summary = pd.DataFrame([result.to_dict() for result in evaluate_core_claims(raw)])
    summary.to_csv(results_root / "summary.csv", index=False)
    write_report(results_root, raw, runs, summary)
    if args.plots:
        for script in ("core_results_plot.py", "phase_models_plot.py"):
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "figures" / script), "--results-root", str(results_root)],
                check=True,
                cwd=PROJECT_ROOT,
            )


if __name__ == "__main__":
    main()
