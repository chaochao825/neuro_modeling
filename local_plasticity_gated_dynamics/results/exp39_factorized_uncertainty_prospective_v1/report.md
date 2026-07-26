# Exp39 Factorized-Uncertainty Result

Verdict: **SUPPORT**.
Joint preregistered gate: **PASS**.

The primary panel contains only pairwise and triple h/Q/R combinations absent from fitting.
The eight-mode factorial IMM and time-varying oracle receive privileged generator support and are upper bounds, not baselines the method is required to beat.

## Utility

- best_fixed: mean NLL gain +0.290580; positive in 30/30 seeds; Holm p=4.65661e-09; PASS.
- seen_mode_imm: mean NLL gain +0.048008; positive in 30/30 seeds; Holm p=4.65661e-09; PASS.

## Selective clamps

- h: high-factor penalty +0.026064; low-factor penalty -0.011033; selectivity +0.037097; Holm p=4.65661e-09; PASS.
- q: high-factor penalty +0.041750; low-factor penalty +0.015068; selectivity +0.026683; Holm p=4.21517e-06; PASS.
- r: high-factor penalty +0.150800; low-factor penalty +0.098430; selectivity +0.052370; Holm p=4.65661e-09; PASS.

## Scope

Passing would support only compositional synthetic identifiability. Failure keeps IBL behavior and neural analysis locked. Low state dimension alone is never counted as support.
