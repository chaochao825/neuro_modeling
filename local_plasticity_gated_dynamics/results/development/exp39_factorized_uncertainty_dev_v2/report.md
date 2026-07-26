# Exp39 Factorized-Uncertainty Result

Verdict: **OPPOSE**.
Joint preregistered gate: **FAIL**.

The primary panel contains only pairwise and triple h/Q/R combinations absent from fitting.
The eight-mode factorial IMM and time-varying oracle receive privileged generator support and are upper bounds, not baselines the method is required to beat.

## Utility

- best_fixed: mean NLL gain +0.288798; positive in 8/8 seeds; FAIL.
- seen_mode_imm: mean NLL gain +0.041772; positive in 8/8 seeds; FAIL.

## Selective clamps

- h: high-factor penalty +0.022943; low-factor penalty -0.009587; selectivity +0.032530; FAIL.
- q: high-factor penalty +0.041844; low-factor penalty +0.011970; selectivity +0.029874; FAIL.
- r: high-factor penalty +0.151198; low-factor penalty +0.097556; selectivity +0.053642; FAIL.

## Scope

Passing would support only compositional synthetic identifiability. Failure keeps IBL behavior and neural analysis locked. Low state dimension alone is never counted as support.
