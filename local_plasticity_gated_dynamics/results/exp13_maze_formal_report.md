# Exp13 MAZE formal report

- Dataset: `albertoRodriguez97/MazeBench`
- Dataset revision: `a71a2d1e0931c79f74cd91c5accd13f164d34c73`
- Raw task panel SHA-256: `ce270abdeff30be94f152d575ec05d26d46c49dc8755c3bdea3cb85bb46875f2`
- Clean-run manifest SHA-256: `44dedd8c58237d7624e4a7c34ac22260d20fdca88454b75ded8bd42f30a56022`
- Bootstrap draws: 10000 (registered config).
- Test split role: `ood`; registered OOD core eligibility=True.
- Statistical unit: source/augmentation dependency component; seed replicates are averaged within task before component-level inference.
- Scope: matched hybrid proposal selection only; no neural or biological claim.
- Candidate coverage gate: 1.0000 (required 0.9000; passed=True).

## Absolute performance

| Condition | Exact accuracy | 95% CI | Coverage | Parameters | BPTT |
|---|---:|---:|---:|---:|---:|
| support_heuristic | 0.8889 | [0.7222, 1.0000] | 1.0000 | 0 | False |
| flat_local | 0.9944 | [0.9852, 1.0000] | 1.0000 | 1657 | False |
| hierarchical_local | 0.9907 | [0.9741, 1.0000] | 1.0000 | 1673 | False |
| trace_local | 0.9889 | [0.9704, 1.0000] | 1.0000 | 1681 | False |
| gru_bptt | 1.0000 | [1.0000, 1.0000] | 1.0000 | 1841 | True |
| candidate_oracle | 1.0000 | [1.0000, 1.0000] | 1.0000 | 0 | False |

## Registered paired comparisons

| Comparison | Estimate | 95% CI | Holm p | Conclusion |
|---|---:|---:|---:|---|
| hierarchical_vs_flat | -0.0037 | [-0.0204, 0.0074] | 1 | inconclusive |
| trace_vs_flat | -0.0056 | [-0.0222, 0.0056] | 1 | inconclusive |
| hierarchical_vs_support_heuristic | 0.1019 | [0.0000, 0.2574] | 0.8986 | inconclusive |
| hierarchical_vs_gru_bptt | -0.0093 | [-0.0259, 0.0000] | 0.8986 | inconclusive |
| hierarchical_retains_90pct_gru | 0.0907 | [0.0741, 0.1000] | 0.000352 | support |
| trace_vs_hierarchical | -0.0019 | [-0.0056, 0.0000] | 0.9519 | inconclusive |

## Core conclusion

`hierarchical_local > flat_local`: **inconclusive** (difference -0.0037, 95% CI [-0.0204, 0.0074]).

This conclusion cannot be promoted to end-to-end neural reasoning: the same deterministic program/search proposal library is supplied to every selector. MAZE does not replace the pending multi-session neural-activity experiment.
