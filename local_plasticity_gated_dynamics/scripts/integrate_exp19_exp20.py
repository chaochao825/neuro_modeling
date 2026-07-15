"""Append validated Exp19/Exp20 scoped claims to the project report artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = (
    "claim_id",
    "experiment",
    "metric",
    "comparison",
    "stats_unit",
    "n_planned",
    "n_complete",
    "n_failed",
    "estimate",
    "ci_low",
    "ci_high",
    "effect_size",
    "p_value",
    "multiplicity_method",
    "conclusion",
    "criterion",
    "note",
)
EXPERIMENTS = {
    "exp19_belief_ei_effective_dynamics",
    "exp20_ibl_md_belief_dynamics",
}
START = "<!-- exp19-exp20:start -->"
END = "<!-- exp19-exp20:end -->"


def _slug(value: object) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    if not result:
        raise ValueError("claim proposition cannot produce an empty ID")
    return result


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _claim_rows(scoped: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    rows = []
    for index, item in scoped.reset_index(drop=True).iterrows():
        experiment = str(item["experiment"])
        if experiment not in EXPERIMENTS:
            raise ValueError(f"unexpected scoped experiment {experiment!r}")
        planned = int(
            item.get(
                "n_planned",
                item.get("n_sessions", 0),
            )
        )
        complete = int(
            item.get(
                "n_complete",
                item.get("n_sessions", 0),
            )
        )
        p_holm = _finite(item.get("holm_adjusted_p"))
        p_raw = _finite(item.get("p_value"))
        rows.append(
            {
                "claim_id": f"{prefix}_{index:02d}_{_slug(item['proposition'])}",
                "experiment": experiment,
                "metric": str(item["proposition"]),
                "comparison": str(item["comparison"]),
                "stats_unit": str(item.get("inference_unit", "unspecified")),
                "n_planned": planned,
                "n_complete": complete,
                "n_failed": max(0, planned - complete),
                "estimate": _finite(item.get("estimate")),
                "ci_low": _finite(item.get("ci_low")),
                "ci_high": _finite(item.get("ci_high")),
                "effect_size": _finite(item.get("estimate")),
                "p_value": p_holm if np.isfinite(p_holm) else p_raw,
                "multiplicity_method": str(
                    item.get("multiplicity_family", "none_scoped_audit")
                ),
                "conclusion": str(item["conclusion"]),
                "criterion": str(
                    item.get("threshold", item.get("effect_definition", "scoped"))
                ),
                "note": str(item.get("claim_scope", "")),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without pandas' optional tabulate dependency."""

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


def integrate(results_root: str | Path) -> tuple[Path, Path]:
    root = Path(results_root)
    summary_path = root / "summary.csv"
    report_path = root / "report.md"
    exp19_path = root / "exp19_belief_ei_effective_dynamics_formal_summary.csv"
    exp20_path = root / "exp20_ibl_md_belief_dynamics_formal_summary.csv"
    required = (summary_path, report_path, exp19_path, exp20_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"integration inputs are incomplete: {missing}")

    existing = pd.read_csv(summary_path)
    if tuple(existing.columns) != SUMMARY_COLUMNS:
        raise ValueError("results/summary.csv schema differs from the registered report")
    exp19 = pd.read_csv(exp19_path)
    exp20 = pd.read_csv(exp20_path)
    scoped = pd.concat((exp19, exp20), ignore_index=True, sort=False)
    if set(scoped["experiment"].astype(str)) != EXPERIMENTS:
        raise ValueError("scoped summaries do not contain exactly Exp19 and Exp20")
    appended = pd.concat(
        (
            _claim_rows(exp19, prefix="E19"),
            _claim_rows(exp20, prefix="E20"),
        ),
        ignore_index=True,
    )
    retained = existing.loc[~existing["experiment"].isin(EXPERIMENTS)]
    combined = pd.concat((retained, appended), ignore_index=True)
    if combined["claim_id"].duplicated().any():
        raise ValueError("integrated claim IDs are not unique")
    combined.to_csv(summary_path, index=False, lineterminator="\n")

    report = report_path.read_text(encoding="utf-8")
    if START in report or END in report:
        if report.count(START) != 1 or report.count(END) != 1:
            raise ValueError("existing Exp19/Exp20 report markers are malformed")
        before, remainder = report.split(START, 1)
        _, after = remainder.split(END, 1)
        report = before.rstrip() + "\n\n" + after.lstrip()
    table = _markdown_table(
        appended[
            ["claim_id", "metric", "comparison", "estimate", "conclusion", "note"]
        ]
    )
    section = "\n".join(
        (
            START,
            "## Exp19/Exp20 belief-to-dynamics extension",
            "",
            table,
            "",
            "Exp19 is a frozen high-rank Dale E/I sufficiency audit with a train-only "
            "three-epoch surrogate; it is not recurrent-plasticity or full-LDS evidence. "
            "Exp20 is a real-IBL teacher-forced conditional Poisson analysis; "
            "probabilityLeft is evaluation/split-only and the recordings cannot identify E/I.",
            "",
            "Generated scoped artifacts: `exp19_belief_ei_effective_dynamics_formal_*` "
            "and `exp20_ibl_md_belief_dynamics_formal_*`.",
            END,
        )
    )
    report_path.write_text(report.rstrip() + "\n\n" + section + "\n", encoding="utf-8")
    return summary_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()
    integrate(args.results_root)


if __name__ == "__main__":
    main()
