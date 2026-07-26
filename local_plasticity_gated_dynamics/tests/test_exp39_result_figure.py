from __future__ import annotations

from pathlib import Path

from figures.exp39_factorized_uncertainty_result_plot import render


ROOT = Path(__file__).resolve().parents[1]


def test_exp39_publication_figure_renders_from_formal_data(
    tmp_path: Path,
) -> None:
    result = ROOT / "results/exp39_factorized_uncertainty_prospective_v1"
    if not result.exists():
        return
    outputs = render(result, tmp_path / "formal")
    assert {path.suffix for path in outputs} == {".pdf", ".svg", ".png"}
    assert all(path.stat().st_size > 1_000 for path in outputs)
