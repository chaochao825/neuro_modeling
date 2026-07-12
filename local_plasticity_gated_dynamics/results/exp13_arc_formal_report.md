# Exp13 ARC formal report

- Dataset revision: `399030444e0ab0cc8b4e199870fb20b863846f34`
- Raw task panel SHA-256: `0865d15177be359194b454ca8c94620df34e3ddf449b096b5e796847e0533b4d`
- Clean-run manifest SHA-256: `7e46d4e0b62106e20b047f89ea25903619806f738683fc1be38d6ae4a87e8ead`
- Statistical unit: source/augmentation dependency component; seeds are nested within task.
- Scope: matched hybrid proposal selection only; no neural or biological claim.
- Candidate coverage gate: 0.0125 (required 0.9000; passed=False).

## Absolute performance

| Condition | Exact accuracy | 95% CI | Coverage | Parameters | BPTT |
|---|---:|---:|---:|---:|---:|
| support_heuristic | 0.0050 | [0.0000, 0.0125] | 0.0125 | 0 | False |
| flat_local | 0.0030 | [0.0000, 0.0085] | 0.0125 | 1657 | False |
| hierarchical_local | 0.0030 | [0.0000, 0.0085] | 0.0125 | 1673 | False |
| trace_local | 0.0034 | [0.0000, 0.0094] | 0.0125 | 1681 | False |
| gru_bptt | 0.0050 | [0.0000, 0.0125] | 0.0125 | 1841 | True |
| candidate_oracle | 0.0125 | [0.0025, 0.0251] | 0.0125 | 0 | False |

## Registered paired comparisons

| Comparison | Estimate | 95% CI | Holm p | Conclusion |
|---|---:|---:|---:|---|
| hierarchical_vs_flat | 0.0000 | [-0.0003, 0.0003] | 1 | inconclusive |
| trace_vs_flat | 0.0004 | [0.0000, 0.0011] | 1 | inconclusive |
| hierarchical_vs_support_heuristic | -0.0020 | [-0.0070, 0.0010] | 1 | inconclusive |
| hierarchical_vs_gru_bptt | -0.0020 | [-0.0070, 0.0010] | 1 | inconclusive |
| hierarchical_retains_90pct_gru | -0.0015 | [-0.0063, 0.0013] | 1 | inconclusive |
| trace_vs_hierarchical | 0.0004 | [0.0000, 0.0011] | 1 | inconclusive |

## Core conclusion

`hierarchical_local > flat_local`: **inconclusive** (difference 0.0000, 95% CI [-0.0003, 0.0003]).

This conclusion cannot be promoted to end-to-end neural reasoning: the same deterministic program proposal library is supplied to every selector. ARC does not replace the pending multi-session neural-activity experiment.
