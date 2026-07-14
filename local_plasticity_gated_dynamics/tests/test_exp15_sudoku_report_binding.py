from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_report import _exp15_sudoku_report_lines


PROJECT = Path(__file__).resolve().parents[1]


def _copy_snapshot(target: Path) -> None:
    for name in ("exp15_formal_summary.csv", "exp15_formal_run_manifest.csv"):
        shutil.copy2(PROJECT / "results" / name, target / name)


def test_exp15_sudoku_report_is_bound_to_reviewed_summary_and_manifest(
    tmp_path,
) -> None:
    _copy_snapshot(tmp_path)
    lines = _exp15_sudoku_report_lines(tmp_path)
    assert any("Sudoku engineering audit" in line for line in lines)
    assert any("75.00%" in line for line in lines)
    assert all("**support**" not in line for line in lines)

    summary_path = tmp_path / "exp15_formal_summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[summary["family"].eq("sudoku"), "conclusion"] = "support"
    summary.to_csv(summary_path, index=False)
    with pytest.raises(ValueError, match="reviewed snapshot"):
        _exp15_sudoku_report_lines(tmp_path)
