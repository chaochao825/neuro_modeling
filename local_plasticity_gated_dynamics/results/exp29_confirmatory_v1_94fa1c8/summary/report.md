# Exp29 one-shot confirmatory actuator selector

## Conclusion

**SUPPORT** — both unconditional seed-level confirmatory endpoints passed.

Evidence validity: `true`. The statistical
unit is the evaluation seed (`n=30`). Every one of the 44
registered heldout cells per seed enters both primary endpoints, including
cells with infeasible active actuators. Infeasible selections receive their
same-cell frozen utility; they do not support a matched-budget mechanism claim.

| Endpoint | Seed mean | 95% bootstrap CI | Positive Holm p | Negative Holm p |
|---|---:|---:|---:|---:|
| local_noninferiority_contrast | 0.009964 | [0.007113, 0.012734] | 1.99998e-05 | 1 |
| local_minus_fixed_best | 0.100260 | [0.095376, 0.104656] | 1.99998e-05 | 1 |

## Audit boundary

- Meta fitting data: Exp26 seeds `0--29`, discovery/validation only.
- Confirmatory evaluation data: Exp29 seeds `60--89`, heldout/test only.
- Selector fit count: `1`; root fit seed: `2801`.
- Source rows used for fitting: `0`.
- Attempt path: `/home/spco/sow_linear/exp29_confirmatory_selector_v1_94fa1c8/runs/exp29_confirmatory_actuator_selector/seed_2801/20260717T113245.535597Z_exp29-confirmatory-selector-v1`.
- Git commit/tree: `94fa1c86e210e5bddd4e0ac7332577c07923cfca` / `f95d2ba7180ec93c57f27dc257987ae8df13ae8c`.
- Config SHA-256: `a5b5d6993ac8f0482dce6e8a08fc0428e8b5044cb9acf0a9b2bc277c464358f5`.
- Re-run or replacement attempts are forbidden.

The oracle is a test-aware feasible-plus-frozen ceiling. GRU-BPTT is an
isolated baseline; the local selector uses neither autograd nor BPTT. A support
classification concerns unconditional deployed selector utility over the
frozen dictionary. It does not convert infeasible fallback rows into
matched-budget evidence, establish hidden-context inference, or show online
scalar-reward learning.
