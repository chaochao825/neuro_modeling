from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.plot_exp27 import plot_selector_evidence


def _endpoints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed": [0, 1, 2],
            "routing_utility": [0.65, 0.66, 0.64],
            "gain_utility": [0.60, 0.61, 0.62],
            "low_rank_utility": [0.72, 0.70, 0.71],
            "fixed_best_utility": [0.72, 0.70, 0.71],
            "oracle_utility": [0.88, 0.86, 0.87],
            "gru_bptt_utility": [0.84, 0.82, 0.83],
            "local_three_factor_utility": [0.86, 0.84, 0.85],
            "local_minus_fixed_best": [0.14, 0.14, 0.14],
            "oracle_minus_fixed_best": [0.16, 0.16, 0.16],
            "local_selection_accuracy": [0.90, 0.88, 0.89],
            "gru_selection_accuracy": [0.86, 0.84, 0.85],
        }
    )


def test_plot_exp27_writes_all_registered_formats(tmp_path: Path) -> None:
    outputs = plot_selector_evidence(_endpoints(), tmp_path / "selector")

    assert [path.suffix for path in outputs] == [".png", ".pdf", ".svg"]
    assert all(path.exists() and path.stat().st_size > 1_000 for path in outputs)


def test_plot_exp27_accepts_an_explicit_nonconfirmatory_title(tmp_path: Path) -> None:
    title = "Exp28 post-hoc sensitivity (NON-CONFIRMATORY; n=3 seeds)"
    contrast_title = "B  Directional non-inferiority sensitivity"
    outputs = plot_selector_evidence(
        _endpoints(),
        tmp_path / "sensitivity",
        title=title,
        contrast_title=contrast_title,
    )

    svg = outputs[2].read_text(encoding="utf-8")
    assert title in svg
    assert contrast_title in svg
    assert "Exp27 frozen-family" not in svg


def test_plot_exp27_rejects_pseudoreplicated_or_incomplete_table(
    tmp_path: Path,
) -> None:
    duplicated = pd.concat([_endpoints(), _endpoints().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per seed"):
        plot_selector_evidence(duplicated, tmp_path / "duplicate")

    with pytest.raises(ValueError, match="lacks columns"):
        plot_selector_evidence(
            _endpoints().drop(columns="oracle_utility"), tmp_path / "bad"
        )
