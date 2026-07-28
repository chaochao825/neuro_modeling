from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from figures.exp43_fast_slow_causal_decomposition_plot import (
    cell_gains,
    learned_qr_by_cell,
    render,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp43_fast_slow_causal_decomposition_development_v1"


def test_exp43_cell_110_failure_is_preserved_in_figure_data() -> None:
    block = pd.read_csv(RESULT / "block_metrics.csv", dtype={"cell": str})
    gains = cell_gains(block)
    cell_110 = gains.loc[gains["cell"] == "110"]
    assert cell_110["total_variance"].mean() < 0.0
    assert cell_110["seen_imm"].mean() < 0.0


def test_exp43_cell_110_qr_miscalibration_is_preserved() -> None:
    block = pd.read_csv(RESULT / "block_metrics.csv", dtype={"cell": str})
    estimates = learned_qr_by_cell(block)
    cell_110 = estimates.loc[estimates["cell"] == "110"]
    assert np.isclose(cell_110["true_q"].iloc[0], 0.04)
    assert np.isclose(cell_110["true_r"].iloc[0], 0.01)
    assert cell_110["learned_q"].mean() < cell_110["true_q"].iloc[0]
    assert cell_110["learned_r"].mean() > cell_110["true_r"].iloc[0]


def test_exp43_development_figure_renders_from_validated_data(
    tmp_path: Path,
) -> None:
    outputs = render(RESULT, tmp_path / "exp43")
    assert {path.suffix for path in outputs} == {".pdf", ".svg", ".png"}
    assert all(path.stat().st_size > 1_000 for path in outputs)
