from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.build_report import _exp16_pilot_report_lines


PREFIX = "exp16_tiny_recursive_smoke_3seed"
BOUND_SUFFIXES = (
    "conditions.csv",
    "comparison.csv",
    "raw.csv.gz",
    "run_manifest.csv",
)


def test_exp16_report_uses_only_the_reviewed_clean_snapshot(tmp_path: Path) -> None:
    source = Path("results")
    for suffix in BOUND_SUFFIXES:
        name = f"{PREFIX}_{suffix}"
        shutil.copy2(source / name, tmp_path / name)

    lines = _exp16_pilot_report_lines(tmp_path)
    rendered = "\n".join(lines)
    assert "pilot only" in rendered
    assert "0/8 held-out fixture puzzles" in rendered
    assert "no evidence for a recursive-state advantage" in rendered

    conditions = tmp_path / f"{PREFIX}_conditions.csv"
    conditions.write_text(
        conditions.read_text(encoding="utf-8").replace(
            ",0.0,0.0,0.0,", ",0.1,0.0,0.0,", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs from snapshot"):
        _exp16_pilot_report_lines(tmp_path)
