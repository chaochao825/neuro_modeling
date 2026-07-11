"""Aggregate immutable runs into compressed raw metrics, claims, and a report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.claims import evaluate_core_claims  # noqa: E402


MAX_PUBLISHED_RAW_BYTES = 95 * 1024 * 1024
PORTABLE_RUNS_ROOT = "${CORE_PROJECT_ROOT}/results/runs"
REDACTED_HOST_TEXT = "${REDACTED_HOST_TEXT}"
_PORTABLE_SEGMENT = re.compile(r"[A-Za-z0-9._@+=-]+\Z")
_HOST_ABSOLUTE_PATHS = (
    re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]", flags=re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_\\])\\\\[^\\/\s]+[\\/]"),
    re.compile(r"(?<![A-Za-z0-9_:])//[^/\s]+/"),
    re.compile(r"(?<![A-Za-z0-9_:$}/.~])/(?!/)[^\s/]+"),
)
_HOST_PATH_AUDIT = (
    re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]+", flags=re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_:])(?:\\{2,}|/{2})[^\\/\s]+[\\/]+"),
    re.compile(r"(?<![A-Za-z0-9_:$}/.~])/(?!/)[A-Za-z0-9._~-]+"),
)


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            _redact_nested_host_paths(value),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    return value


def _portable_run_path(value: object) -> object:
    """Remove machine-specific prefixes from a published run path.

    Run artifacts remain in ignored local ``results/runs`` directories.  The
    compact snapshots retain their relative artifact location beneath a
    symbolic project root, so moving a snapshot between Windows and POSIX
    hosts cannot publish either host's absolute checkout path.  An unusual
    legacy path that has no recognizable ``runs`` component is represented by
    a deterministic digest instead of leaking or silently coalescing it.
    """

    if value is None or (
        not isinstance(value, (dict, list, tuple)) and bool(pd.isna(value))
    ):
        return value
    original = str(value).strip()
    if not original:
        return original
    normalized = original.replace("\\", "/")
    folded = normalized.casefold()
    portable_folded = PORTABLE_RUNS_ROOT.casefold()
    is_host_absolute = normalized.startswith("/") or bool(
        re.match(r"[A-Za-z]:/", normalized)
    )

    suffix: str | None = None
    if folded == portable_folded:
        suffix = ""
    elif folded.startswith(portable_folded + "/"):
        suffix = normalized[len(PORTABLE_RUNS_ROOT) + 1 :]
    elif is_host_absolute:
        for marker in ("/results/runs/",):
            offset = folded.rfind(marker)
            if offset >= 0:
                suffix = normalized[offset + len(marker) :]
                break
        if suffix is None and folded.endswith("/results/runs"):
            suffix = ""
    else:
        for marker in ("results/runs/", "runs/"):
            if folded.startswith(marker):
                suffix = normalized[len(marker) :]
                break
        if suffix is None and folded in ("results/runs", "runs"):
            suffix = ""

    parts = [] if suffix is None else [part for part in suffix.split("/") if part]
    if suffix is not None and all(
        part not in {".", ".."} and _PORTABLE_SEGMENT.fullmatch(part) for part in parts
    ):
        return PORTABLE_RUNS_ROOT + (f"/{'/'.join(parts)}" if parts else "")

    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:24]
    return f"{PORTABLE_RUNS_ROOT}/_sanitized/{digest}"


def _portable_discovered_run_path(run_dir: Path, results_root: Path) -> str:
    """Build a portable path directly from a discovered artifact directory."""

    try:
        relative = run_dir.relative_to(results_root / "runs")
    except ValueError as error:
        raise ValueError("discovered run directory escaped results/runs") from error
    return str(_portable_run_path(f"runs/{relative.as_posix()}"))


def _redact_host_text(text: str) -> str:
    """Replace a path-bearing text unit without retaining ambiguous fragments."""

    if not any(pattern.search(text) for pattern in _HOST_ABSOLUTE_PATHS):
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"{REDACTED_HOST_TEXT}/{digest}"


def _redact_nested_host_paths(value: object) -> object:
    """Recursively redact before JSON escaping can obscure UNC markers."""

    if isinstance(value, Path):
        return _redact_host_text(str(value))
    if isinstance(value, str):
        return _redact_host_text(value)
    if isinstance(value, dict):
        return {
            str(_redact_nested_host_paths(key)): _redact_nested_host_paths(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_nested_host_paths(item) for item in value]
    return value


def _redact_host_paths(value: object) -> object:
    """Redact scalar or compound host paths at the compact publication boundary."""

    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(decoded, (dict, list)):
                redacted = _redact_nested_host_paths(decoded)
                return json.dumps(
                    redacted,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
    redacted = _redact_nested_host_paths(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(redacted, ensure_ascii=False, sort_keys=True, default=str)
    return redacted


def _assert_no_host_paths(frame: pd.DataFrame) -> None:
    """Fail closed if a compact frame still contains a host absolute path."""

    for column in frame:
        for index, value in frame[column].items():
            if not isinstance(value, str) and isinstance(
                value, (Path, dict, list, tuple)
            ):
                value = json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, str) and any(
                pattern.search(value) for pattern in _HOST_PATH_AUDIT
            ):
                raise ValueError(
                    "absolute host path remained in compact snapshot at "
                    f"column {column!r}, row {index!r}"
                )


def _sanitize_compact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing neither host paths nor host-path fragments."""

    sanitized = frame.copy()
    for column in ("run_path", "path"):
        if column in sanitized:
            sanitized[column] = sanitized[column].map(_portable_run_path)
    for column in sanitized.columns.difference(["run_path", "path"]):
        sanitized[column] = sanitized[column].map(_redact_host_paths)
    _assert_no_host_paths(sanitized)
    return sanitized


def collect_runs(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    runs = []
    for status_path in sorted((results_root / "runs").glob("**/status.json")):
        run_dir = status_path.parent
        portable_run_path = _portable_discovered_run_path(run_dir, results_root)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        config_path = run_dir / "config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        manifest_path = run_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
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
        n_planned = (
            len(json.loads(planned_path.read_text(encoding="utf-8")))
            if planned_path.exists()
            else 0
        )
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
                "path": portable_run_path,
            }
        )
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                record.setdefault("profile", config.get("profile", "unspecified"))
                record["run_path"] = portable_run_path
                record.setdefault("run_status", run_status)
                record.setdefault("run_started_at", run_started_at)
                records.append(
                    {key: _csv_value(value) for key, value in record.items()}
                )
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
                "run_path": portable_run_path,
                "run_status": run_status,
                "run_started_at": run_started_at,
                "status": "failed",
                "error_type": status.get(
                    "error_type",
                    "IncompleteRun" if run_status == "running" else "RunFailure",
                ),
                "error": status.get(
                    "error",
                    "nonterminal run artifact"
                    if run_status == "running"
                    else "top-level run failure",
                ),
                "run_level_failure": True,
            }
            records.append(
                {key: _csv_value(value) for key, value in run_failure.items()}
            )
    return _sanitize_compact_frame(
        pd.DataFrame.from_records(records)
    ), _sanitize_compact_frame(pd.DataFrame.from_records(runs))


def _read_compact_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, float_precision="round_trip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _raw_snapshot_path(results_root: Path) -> Path:
    """Prefer the lossless compressed snapshot over the legacy plain CSV."""

    compressed = results_root / "raw_metrics.csv.gz"
    if compressed.exists():
        if compressed.stat().st_size == 0:
            raise ValueError("authoritative raw_metrics.csv.gz is empty")
        return compressed
    return results_root / "raw_metrics.csv"


def write_compact_raw(results_root: Path, raw: pd.DataFrame) -> None:
    """Write an authoritative deterministic gzip plus a local plotting cache."""

    raw = _sanitize_compact_frame(raw)
    compressed = results_root / "raw_metrics.csv.gz"
    staged = results_root / "raw_metrics.csv.gz.tmp"
    raw.to_csv(
        staged,
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    if staged.stat().st_size >= MAX_PUBLISHED_RAW_BYTES:
        raise ValueError(
            "compressed raw snapshot exceeds the 95 MiB publication safety limit"
        )
    staged.replace(compressed)
    # Figure scripts deliberately consume a plain CSV. It is reproducible from
    # the tracked gzip snapshot and ignored by git to stay below host limits.
    raw.to_csv(results_root / "raw_metrics.csv", index=False, lineterminator="\n")


def write_compact_runs(results_root: Path, runs: pd.DataFrame) -> None:
    """Write run coverage with the same portable-path publication boundary."""

    runs = _sanitize_compact_frame(runs)
    runs.to_csv(results_root / "runs.csv", index=False, lineterminator="\n")


def _identity_token(value: object) -> str | None:
    """Normalize scalar identifiers without turning missing values into text."""

    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else repr(numeric)
    token = str(value).strip()
    return token or None


def _run_identity(frame: pd.DataFrame) -> pd.Series:
    """Return nullable run keys shared by raw-metric and run-summary tables."""

    identities = pd.Series(pd.NA, index=frame.index, dtype="string")
    if frame.empty:
        return identities
    if "run_id" in frame:
        for index, value in frame["run_id"].items():
            token = _identity_token(value)
            if token is not None:
                identities.at[index] = f"run:{token}"

    start_column = next(
        (name for name in ("run_started_at", "started_at") if name in frame),
        None,
    )
    if start_column is None:
        return identities
    for index in identities.loc[identities.isna()].index:
        experiment = _identity_token(
            frame.at[index, "experiment"] if "experiment" in frame else None
        )
        seed = _identity_token(frame.at[index, "seed"] if "seed" in frame else None)
        started = _identity_token(frame.at[index, start_column])
        if experiment is not None and seed is not None and started is not None:
            identities.at[index] = f"legacy:{experiment}:{seed}:{started}"
    return identities


def merge_compact_snapshot(
    results_root: Path,
    discovered_raw: pd.DataFrame,
    discovered_runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge new immutable artifacts without erasing compact-only history.

    The repository intentionally tracks compact CSV snapshots while omitting
    older timestamped ``results/runs`` directories.  Rebuilding from only the
    currently present directories would therefore erase valid historical rows.
    For every newly discovered ``run_id`` we replace its prior compact rows as
    one unit (so a running/failed attempt can later become complete), retain all
    undiscovered run IDs, and append the freshly collected records.  Replacing
    by run rather than by timestamp also preserves experiments that emit
    multiple condition rows with the same ``recorded_at`` value.
    """

    if not isinstance(results_root, Path):
        raise TypeError("results_root must be a pathlib.Path")
    existing_raw = _sanitize_compact_frame(
        _read_compact_csv(_raw_snapshot_path(results_root))
    )
    existing_runs = _sanitize_compact_frame(
        _read_compact_csv(results_root / "runs.csv")
    )
    discovered_raw = _sanitize_compact_frame(discovered_raw)
    discovered_runs = _sanitize_compact_frame(discovered_runs)
    discovered_run_keys = _run_identity(discovered_runs)
    discovered_raw_keys = _run_identity(discovered_raw)
    if discovered_run_keys.isna().any() or discovered_raw_keys.isna().any():
        raise ValueError(
            "every discovered artifact row needs a run_id or stable "
            "experiment/seed/start-time provenance"
        )
    duplicate_runs = discovered_run_keys.loc[discovered_run_keys.duplicated(False)]
    if not duplicate_runs.empty:
        raise ValueError(
            "multiple discovered run directories share one run identity: "
            + ", ".join(sorted(set(duplicate_runs.astype(str).tolist())))
        )
    discovered_ids = set(discovered_run_keys.dropna().astype(str).tolist())
    discovered_ids.update(discovered_raw_keys.dropna().astype(str).tolist())

    def without_replaced(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not discovered_ids:
            return frame
        identities = _run_identity(frame)
        return frame.loc[~identities.isin(discovered_ids)].copy()

    retained_raw = without_replaced(existing_raw)
    retained_runs = without_replaced(existing_runs)
    raw = pd.concat([retained_raw, discovered_raw], ignore_index=True, sort=False)
    runs = pd.concat([retained_runs, discovered_runs], ignore_index=True, sort=False)
    # Exact duplicate removal is only a fallback for truly unidentified legacy
    # rows; records within a run are deliberately never keyed by timestamp.
    raw = raw.drop_duplicates(keep="last", ignore_index=True)
    runs = runs.drop_duplicates(keep="last", ignore_index=True)
    return raw, runs


def _format_number(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)


def write_report(
    results_root: Path, raw: pd.DataFrame, runs: pd.DataFrame, summary: pd.DataFrame
) -> None:
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
        for (experiment, profile), group in runs.groupby(
            ["experiment", "profile"], dropna=False
        ):
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
        "- Tuned BPTT rate-RNN and GRU baselines are isolated; local-learning models do not import autograd/optimizers and cannot load baseline checkpoints.",
        "- Absolute accuracy, BPTT non-inferiority, and GRU non-inferiority are independent claims and are never merged into one decision.",
        "- P0 non-inferiority means retaining at least 90% of a tuned baseline, not parity or outperformance; accuracy intervals are seed-level statements, not guarantees for every seed.",
        "- Legacy exp03 is a supervised/oracle-warm-start MD upper bound: its cue, gate fit, and recurrent third factor do not satisfy the hidden-context contract, so it cannot support P2.",
        "- A low matrix/tangent rank without improved held-out behavior or prediction cannot support the revised mechanism.",
        "- P0 L1 and L2 budget panels are matched separately; the non-selected norm is diagnostic and no simultaneous dual-norm match is claimed.",
        "- P0 task+homeostasis has one matched task component plus one matched homeostasis component, so its total component budget is twice homeostasis-only; normalization corrections are reported outside those selected component budgets.",
        "- The P0 homeostasis control is yoked inhibitory strengthening, not closed-loop E/I stability evidence; formal normal-perturbation decay, Lyapunov, and closure-error gates remain pending P4.",
        "- P1 cross-parameterization budgets are descriptive and unmatched; physical-rank versus credit-tangent results cannot rank parameterizations by task performance.",
        "- P2 learned-HMM and MD-like gates receive cue observations rather than realized context. Learned-HMM fitting uses legal train-episode batch smoothing, while every held-out belief trajectory is causal and frozen before truth scoring.",
        "- P2 supervised context inference is an explicitly ineligible upper bound. The oracle filter knows q/h but never receives realized state or switch boundaries.",
        "- P2 q/h cells are paired within seed and then equally averaged; post-fit clamp, delay, and shuffle interventions reuse the intact MD checkpoint and readout.",
        "- The P2 MD candidate is specifically causal two-slice local soft counts with Hebbian lag-1--5 moment shrinkage; it is not evidence for a pure soft-count learner.",
        "- P2_overall is a gate-only belief/effective-control stage gate. It cannot support coupled N=256/N=512 PFC/E/I dynamics, recurrent three-factor credit assignment, or homeostasis.",
        "- P2 energy_proxy_per_trial measures belief confidence and trajectory change, not physical energy consumption; P2i is diagnostic and excluded from P2_overall.",
        "- Nominal feedback dimension is an upper bound on the empirical projected signal span; it is not reported as an automatically realized exact rank.",
        "- PCA, normalization, nuisance regression, subspaces, and dynamics are fit on training trials/blocks only.",
        "- Time points never cross trial/block splits. Symmetric smoothing is visualization-only; predictive likelihood uses causal smoothing/raw counts.",
        "- Inference units are seeds, sessions, or animals. Neurons are never treated as independent replicates.",
        "- IBL latent/behavior lead–lag is descriptive system-level evidence and is not interpreted as causal gating.",
        "- IBL support requires a stimulus-pre primary panel with at least 5 animals/20 sessions, explicit unit-QC/context-coverage/nested-CV provenance, hierarchical observations, and parameter counts that include preprocessing.",
        "",
        "## External-data status",
        "",
        "The referenced Zenodo sequence-memory record currently reports `access_right=restricted`. Missing access is retained as a failed session-level artifact and makes the corresponding claims inconclusive; it is never replaced by synthetic evidence.",
        "",
        "## Generated artifacts",
        "",
        "- `results/raw_metrics.csv.gz`: lossless raw metric snapshot, including failed and invalid conditions; the uncompressed CSV is a reproducible local plotting cache.",
        "- `results/runs.csv`: run status and planned-cell coverage.",
        "- `results/summary.csv`: one row per pre-registered core claim.",
        "- `results/core_results.pdf`, `results/phase_models.pdf`, and `results/hidden_context.pdf`: script-generated data figures when applicable.",
        "",
    ]
    (results_root / "report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    discovered_raw, discovered_runs = collect_runs(results_root)
    raw, runs = merge_compact_snapshot(results_root, discovered_raw, discovered_runs)
    write_compact_raw(results_root, raw)
    write_compact_runs(results_root, runs)
    summary = pd.DataFrame([result.to_dict() for result in evaluate_core_claims(raw)])
    summary.to_csv(results_root / "summary.csv", index=False, lineterminator="\n")
    write_report(results_root, raw, runs, summary)
    if args.plots:
        for script in (
            "core_results_plot.py",
            "phase_models_plot.py",
            "hidden_context_plot.py",
        ):
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "figures" / script),
                    "--results-root",
                    str(results_root),
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )


if __name__ == "__main__":
    main()
