from __future__ import annotations

import pandas as pd

from scripts.diagnose_exp38_factorized_memory import stream51_retention_gate


def _metrics(direct_gain: float, hmm_gain: float) -> pd.DataFrame:
    rows = []
    for seed in range(5):
        baseline = 1.0
        rows.extend(
            [
                {
                    "seed": seed,
                    "condition": "posterior_ema",
                    "video_equal_nll": baseline,
                },
                {
                    "seed": seed,
                    "condition": "direct_alpha",
                    "video_equal_nll": baseline - direct_gain,
                },
                {
                    "seed": seed,
                    "condition": "likelihood_hmm",
                    "video_equal_nll": baseline - hmm_gain,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_stream51_gate_requires_effect_size_and_seed_consistency() -> None:
    failed_effect = stream51_retention_gate(_metrics(0.001, -0.1))
    assert not failed_effect["stream51_retained"]
    passed = stream51_retention_gate(_metrics(0.01, -0.1))
    assert passed["stream51_retained"]
    assert passed["methods"]["direct_alpha"]["passed"]
