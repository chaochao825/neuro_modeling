from __future__ import annotations

from pathlib import Path

from figures.exp39_factorized_uncertainty_plot import render


ROOT = Path(__file__).resolve().parents[1]


def test_exp39_figure_renders_from_development_artifacts(tmp_path: Path) -> None:
    result = ROOT / "results/development/exp39_factorized_uncertainty_dev_v2"
    if not result.exists():
        return
    paths = render(result, tmp_path / "exp39")
    assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
    assert all(path.stat().st_size > 1_000 for path in paths)
