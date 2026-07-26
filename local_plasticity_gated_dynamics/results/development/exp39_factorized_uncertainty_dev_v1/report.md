# Exp39 Factorized-Uncertainty Result

Verdict: **OPPOSE**.
Joint preregistered gate: **FAIL**.

The primary panel contains only pairwise and triple h/Q/R combinations absent from fitting.
The eight-mode factorial IMM and time-varying oracle receive privileged generator support and are upper bounds, not baselines the method is required to beat.

## Utility

- best_fixed: mean NLL gain +0.248542; positive in 8/8 seeds; FAIL.
- seen_mode_imm: mean NLL gain +0.001515; positive in 4/8 seeds; FAIL.

## Selective clamps

- h: high-factor penalty +0.020208; low-factor penalty -0.018085; selectivity +0.038292; FAIL.
- q: high-factor penalty +0.028366; low-factor penalty -0.007943; selectivity +0.036308; FAIL.
- r: high-factor penalty +0.104807; low-factor penalty +0.088133; selectivity +0.016675; FAIL.

## Scope

Passing would support only compositional synthetic identifiability. Failure keeps IBL behavior and neural analysis locked. Low state dimension alone is never counted as support.
