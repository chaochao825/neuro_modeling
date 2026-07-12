# Exp15 task-specialized reasoning audit

Both clean formal runs used commit `f61f26b5d424d637da27ef7ff6c56ee1c372072b`, Python 3.11, target-isolated public task capabilities, no BPTT, and no required spiking mechanism. Statistical resampling used source groups, not cells or internal reasoning steps.

## ARC-AGI-1

The slow/fast program adapter solved 1 of 399 de-duplicated OOD evaluation tasks: exact accuracy 0.2506% (95% source-group bootstrap CI 0–0.7519%). This does not improve the earlier finite-proposal ARC audit in a practically meaningful way. Moreover, the actual ARC directory-tree manifest is not yet verified, so `source_manifest_verified=false` and `formal_data_eligible=false` regardless of the configured revision string. The conclusion is **inconclusive**.

## Sudoku V2

On 28 de-duplicated public puzzles, pure local row/column/box constraint dynamics solved 75.0% exactly (95% CI 57.14–89.29%). Enabling an explicitly charged bounded search of up to 256 branches solved 100%; mean state evaluations increased from 7.32 to 11.43. The bounded-search result must not be attributed to local dynamics alone. The source preparation manifest is verified, but the test split is non-OOD and no matched-compute advantage comparator is registered. Both conclusions are **inconclusive**.

## Interpretation

The result supports the engineering value of task-specific adapters, not the scientific claim that BDH, HRM, local plasticity, or low-dimensional neural control is superior. Exp15 is deliberately additive and does not alter the published Exp13 or Exp14 conclusions.
