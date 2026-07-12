# Exp13 SUDOKU formal report

- Dataset: `wichtounet/sudoku_dataset-v2`
- Dataset revision: `0db6de036b80b4e8e4574abe6e15026331bd5c2c`
- Raw task panel SHA-256: `f37abb5bf569b28a9d715d1b3eb935c069d95739d04819ff49e198755ef0c648`
- Clean-run manifest SHA-256: `9535115f78f30f400502f1b6827ba6b96225d3cbcfb0bcdd39338ef0b0c94b3b`
- Bootstrap draws: 10000 (registered config).
- Test split role: `non_ood`; registered OOD core eligibility=False.
- Statistical unit: source/augmentation dependency component; seed replicates are averaged within task before component-level inference.
- Scope: matched hybrid proposal selection only; no neural or biological claim.
- Candidate coverage gate: 1.0000 (required 0.9000; passed=True).

## Absolute performance

| Condition | Exact accuracy | 95% CI | Coverage | Parameters | BPTT |
|---|---:|---:|---:|---:|---:|
| support_heuristic | 1.0000 | [1.0000, 1.0000] | 1.0000 | 0 | False |
| flat_local | 1.0000 | [1.0000, 1.0000] | 1.0000 | 1657 | False |
| hierarchical_local | 1.0000 | [1.0000, 1.0000] | 1.0000 | 1673 | False |
| trace_local | 1.0000 | [1.0000, 1.0000] | 1.0000 | 1681 | False |
| gru_bptt | 1.0000 | [1.0000, 1.0000] | 1.0000 | 1841 | True |
| candidate_oracle | 1.0000 | [1.0000, 1.0000] | 1.0000 | 0 | False |

## Registered paired comparisons

| Comparison | Estimate | 95% CI | Holm p | Conclusion |
|---|---:|---:|---:|---|
| hierarchical_vs_flat | 0.0000 | [0.0000, 0.0000] | 1 | inconclusive |
| trace_vs_flat | 0.0000 | [0.0000, 0.0000] | 1 | inconclusive |
| hierarchical_vs_support_heuristic | 0.0000 | [0.0000, 0.0000] | 1 | inconclusive |
| hierarchical_vs_gru_bptt | 0.0000 | [0.0000, 0.0000] | 1 | inconclusive |
| hierarchical_retains_90pct_gru | 0.1000 | [0.1000, 0.1000] | 7.279e-07 | inconclusive |
| trace_vs_hierarchical | 0.0000 | [0.0000, 0.0000] | 1 | inconclusive |

## Core conclusion

`hierarchical_local > flat_local`: **inconclusive** (difference 0.0000, 95% CI [0.0000, 0.0000]).

This conclusion cannot be promoted to end-to-end neural reasoning: the same deterministic program/search proposal library is supplied to every selector. SUDOKU does not replace the pending multi-session neural-activity experiment.


The test split is registered as `non_ood`. Therefore even a statistically positive retention margin cannot be promoted to core support; it remains an exploratory, ceiling-sensitive result.