from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.build_report import _exp16_pilot_report_lines, _exp16_retry_report_lines


PREFIX = "exp16_tiny_recursive_smoke_3seed"
BOUND_SUFFIXES = (
    "conditions.csv",
    "comparison.csv",
    "raw.csv.gz",
    "run_manifest.csv",
)
RETRY_PREFIX = "exp16_tiny_recursive_retry_3seed"
RETRY_FILES = (
    f"{RETRY_PREFIX}_conditions.csv",
    f"{RETRY_PREFIX}_comparison.csv",
    f"{RETRY_PREFIX}_raw.csv.gz",
    f"{RETRY_PREFIX}_run_manifest.csv",
    "exp17_wichtounet_3seed_candidates.csv",
    "exp17_wichtounet_3seed_freeze_decision.json",
    "exp17_wichtounet_3seed_run_manifest.csv",
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


def test_exp16_retry_report_binds_calibration_and_confirmation(tmp_path: Path) -> None:
    source = Path("results")
    project = tmp_path / "project"
    results = project / "results"
    config_dir = project / "configs" / "formal"
    results.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    for name in RETRY_FILES:
        shutil.copy2(source / name, results / name)
    config_name = "exp16_tiny_recursive_sudoku_retry_pilot.json"
    shutil.copy2(Path("configs/formal") / config_name, config_dir / config_name)

    rendered = "\n".join(_exp16_retry_report_lines(results))
    assert "train/inner-validation only" in rendered
    assert "All three seed differences were negative" in rendered
    assert "opposes a tiny recursive-state advantage" in rendered
    assert "remains **inconclusive**" in rendered

    conditions = results / f"{RETRY_PREFIX}_conditions.csv"
    hidden_conditions = results / f"{RETRY_PREFIX}_conditions.missing"
    conditions.rename(hidden_conditions)
    with pytest.raises(ValueError, match="artifact set is incomplete"):
        _exp16_retry_report_lines(results)
    hidden_conditions.rename(conditions)

    config_path = config_dir / config_name
    original_config = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        original_config.replace("blank_low_diversity", "blank_high_diversity", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config differs from reviewed snapshot"):
        _exp16_retry_report_lines(results)
    config_path.write_text(original_config, encoding="utf-8")

    comparison = results / f"{RETRY_PREFIX}_comparison.csv"
    comparison.write_text(
        comparison.read_text(encoding="utf-8").replace("-0.009302", "0.009302"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs from snapshot"):
        _exp16_retry_report_lines(results)
