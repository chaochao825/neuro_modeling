from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import figures.exp26_actuator_phase_diagram_plot as exp26_plot


def _bound_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for seed in range(3):
        for split, generator_offset in (("discovery", 0), ("heldout", 10)):
            for generator in range(4):
                chi = 0.15 + 0.22 * generator
                routing = 0.78 - 0.18 * chi + 0.002 * seed
                gain = 0.74 - 0.12 * chi + 0.002 * seed
                low_rank = 0.62 + 0.30 * chi + 0.002 * seed
                values = {
                    "frozen": 0.55,
                    "routing": routing,
                    "gain": gain,
                    "low_rank": low_rank,
                    # This ceiling must never enter the panel-A advantage.
                    "rgl": 0.99,
                }
                for mode, accuracy in values.items():
                    rows.append(
                        {
                            "seed": seed,
                            "generator_id": f"g{generator_offset + generator}",
                            "generator_split": split,
                            "actuator_mode": mode,
                            "chi": chi,
                            "alpha": generator / 3,
                            "transition_rank": (1, 2, 4, 8)[generator],
                            "input_rank": (1, 2, 4, 1)[generator],
                            "delay": generator % 2,
                            "noise_std": 0.01 * (generator % 2),
                            "test_balanced_accuracy": accuracy,
                            "functional_budget_valid": True,
                            "status": "complete",
                        }
                    )
    endpoints = pd.DataFrame(
        {
            "seed": [0, 1, 2],
            "spearman_rho": [0.80, 0.75, 0.85],
            "classifier_balanced_accuracy": [0.75, 0.80, 0.85],
            "classifier_auroc": [0.85, 0.90, 0.88],
            "chi_minus_alpha_auroc": [0.08, 0.12, 0.10],
            "discovery_threshold": [0.46, 0.50, 0.54],
        }
    )
    return pd.DataFrame(rows), endpoints


def test_exp26_plot_writes_nonempty_pdf_and_svg_from_cli(tmp_path: Path) -> None:
    metrics, endpoints = _bound_frames()
    metrics_path = tmp_path / "metrics.csv"
    endpoints_path = tmp_path / "seed_endpoints.csv"
    output_dir = tmp_path / "figures"
    metrics.to_csv(metrics_path, index=False)
    endpoints.to_csv(endpoints_path, index=False)

    paths = exp26_plot.main(
        [
            "--metrics",
            str(metrics_path),
            "--seed-endpoints",
            str(endpoints_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert [path.suffix for path in paths] == [".pdf", ".svg"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_exp26_advantage_excludes_rgl_and_uses_strict_heldout_pairs() -> None:
    metrics, endpoints = _bound_frames()
    clean, _ = exp26_plot._validated_inputs(metrics, endpoints)
    advantage = exp26_plot.heldout_advantage(clean)
    first = advantage.iloc[0]

    expected = first["low_rank"] - max(first["routing"], first["gain"])
    assert first["advantage"] == pytest.approx(expected)
    assert first["advantage"] != pytest.approx(first["low_rank"] - 0.99)
    assert set(advantage["generator_split"]) == {"heldout"}


def test_exp26_plot_has_no_heldout_estimation_api_or_overall_title() -> None:
    source = inspect.getsource(exp26_plot)
    forbidden = ("np.polyfit", "linregress", "curve_fit", "regplot")
    assert not any(token in source for token in forbidden)

    metrics, endpoints = _bound_frames()
    figure = exp26_plot.make_figure(metrics, endpoints)
    assert figure._suptitle is None
    assert all(axis.get_title() == "" for axis in figure.axes)
    assert all(not axis.spines["top"].get_visible() for axis in figure.axes)
    assert all(not axis.spines["right"].get_visible() for axis in figure.axes)
    plt.close(figure)


@pytest.mark.parametrize("which", ["metrics", "seed_endpoints"])
def test_exp26_plot_fails_closed_for_empty_inputs(which: str) -> None:
    metrics, endpoints = _bound_frames()
    if which == "metrics":
        metrics = metrics.iloc[:0]
    else:
        endpoints = endpoints.iloc[:0]
    with pytest.raises(ValueError, match="empty"):
        exp26_plot.make_figure(metrics, endpoints)


def test_exp26_plot_fails_closed_for_missing_columns() -> None:
    metrics, endpoints = _bound_frames()
    with pytest.raises(ValueError, match="missing required columns"):
        exp26_plot.make_figure(metrics.drop(columns="functional_budget_valid"), endpoints)
