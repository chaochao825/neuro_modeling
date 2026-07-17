# Exp28 post-hoc amended actuator-selector sensitivity

## Overall conclusion

**INCONCLUSIVE** — post-hoc ceiling-amended source cannot restore confirmatory independence; overall classification is forced inconclusive.

One selector set was fitted once on Exp26 seeds `0--29`; the 30 disjoint
seeds `30--59` are the sensitivity-analysis statistical units. The source
ceiling was amended from `128` to `256` after the original panel was inspected,
so confirmatory independence cannot be restored.

Directional sensitivity: **SUPPORT** —
both directional seed-level thresholds passed. This label is descriptive and cannot change the
overall `INCONCLUSIVE` classification.

| Directional endpoint | Seed mean | 95% bootstrap CI | Positive Holm p | Negative Holm p |
|---|---:|---:|---:|---:|
| local_noninferiority_contrast | 0.011854 | [0.009507, 0.014165] | 1.99998e-05 | 1 |
| local_minus_fixed_best | 0.101583 | [0.096471, 0.106751] | 1.99998e-05 | 1 |

Directional support requires both positive lower bounds and both positive
Holm-adjusted `p < 0.05`. The inference status is `post_hoc_amended_sensitivity_non_confirmatory`.

## Audit

- Amended evaluation seeds: `30`.
- Selector fit count: `1`; registered fit seed: `2801`.
- Git commit/tree: `07b5ce23108ca179e1291393fccc12bd9227238e` / `ca01b07349067d5faed5540493ba2d4689dac4c1`.
- Config SHA-256: `964f9b13a1643daab2a7a9e0c35ddbbcc25f35c13196ed00ec21ca4666bfcdf6`.
- Source/package receipt SHA-256: `4922f552c65c598d1e745ada0dbfc78d2eb3315b9ced45b61cc3185cb50eafa9`.
- Protocol-amendment SHA-256: `d91f01144078f1b6af2198f50cbc59381f393165e2bbf1d9795c9365c304089f`.
- Frozen-fit receipt SHA-256: `f8c3e5d97867eb18f7818512f535d3420182b9f9e3ab64d7492527d2f2953192`.
- All-test decision receipt SHA-256: `da16ab30fb55bf8ab9575e0bd942c6528c1584f7d46346bc435bee8b66a39a93`.
- Collected raw SHA-256: `7ae3560acf1e714edd29f30a5112bdd0879519110804467a96d21b5e4838913d`.
- Collector result: exact source, fit, decision, and semantic-row replay passed.

## Interpretation boundary

This analysis tests supervised task-demand-to-family selection over frozen,
task-matched actuator policies. It does not establish hidden belief inference,
de-novo motif formation, scalar-reward-only learning, or update-budget
equivalence between the local and GRU selectors. Inference is conditional on
one fixed meta-training panel and optimizer root seed `2801`; it does not
marginalize meta-panel or optimizer-seed uncertainty.

The original ceiling-128 frozen source protocol is classified **OPPOSE**
because one of 13,200 registered cells failed. Consequently, the corresponding
selector formal analysis is **INCONCLUSIVE**. A future confirmatory test is
reserved for wholly fresh seeds `60--89` under the separate namespace
`exp28_fresh_independent_confirmatory_v2_seeds60_89`; those seeds are not part
of this amended sensitivity analysis.
