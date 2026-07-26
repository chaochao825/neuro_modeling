# Exp41 Matched-Q/R Identifiability Probe

Verdict: **INCONCLUSIVE (development-only diagnostic)**.

The generator has exactly H=0. All numerical jump filters use the disclosed hazard floor `0.0001` because the frozen Exp39 jump step requires a strictly positive hazard.
Fit and test tapes are independently generated. Hyperparameters see only the fit tape, and every method is evaluated on the same test tape.
`generator_supported_seen_regime_imm` and `dynamic_qr_oracle` are privileged generator-supported references, not deployable or fair baselines.

## Descriptive utility

- selected_fixed_jump: baseline-minus-autocov NLL +0.003886; positive in 4/8 seeds.
- current_online_em: baseline-minus-autocov NLL -0.021208; positive in 0/8 seeds.
- h_plus_total_variance: baseline-minus-autocov NLL -0.010722; positive in 3/8 seeds.
- generator_supported_seen_regime_imm: baseline-minus-autocov NLL -0.059558; positive in 0/8 seeds.
- dynamic_qr_oracle: baseline-minus-autocov NLL -0.074846; positive in 0/8 seeds.

## Matched-pair separation

- m06: Q separation +0.023179 (8/8 positive); R separation +0.010180 (8/8 positive).
- m12: Q separation +0.030732 (8/8 positive); R separation +0.015605 (8/8 positive).

## Claim boundary

The matched panel intentionally anticorrelates Q and R within each equal-marginal pair; it tests separation, not a diagonal-dominance cross-loading claim.
At H=0 with a fixed Q/R allocation, tied Q/R and h-plus-total-variance are the same one-scalar parameterization; only h_plus_total_variance is executed.
Functional/update budgets are **not matched**. L1, L2, and update counts are diagnostics only, so the development go gate is forced to FAIL and inference remains inconclusive.
Transition endpoints use the first 1/4/8/16 samples after a block transition and exclude the first block of every sequence.
This artifact cannot upgrade claims, cannot change Exp39, and does not constitute a formal-seed result.
