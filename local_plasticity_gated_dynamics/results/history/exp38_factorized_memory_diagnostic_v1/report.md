# Exp38 Post-Hoc Factorized-Memory Diagnostic

Status: **claim-ineligible revealed-data diagnostic**.
External Stream-51 features/outcomes accessed: **no**.

## Result

The Stream-51 splice task was **retired** for the successor method.

| Condition | Mean video-equal NLL | Mean accuracy | Mean post-switch NLL |
|---|---:|---:|---:|
| direct_alpha | 0.904145 | 0.782759 | 0.943526 |
| likelihood_hmm | 0.974929 | 0.803257 | 0.989660 |
| posterior_ema | 0.907958 | 0.777011 | 0.943890 |
| true_switch_direct_alpha | 0.877958 | 0.786830 | 0.880679 |
| true_switch_likelihood_reset | 2.401651 | 0.814990 | 1.794592 |

## Registered task-reuse gate

- `direct_alpha`: mean NLL gain 0.003813; positive in 5/5 seeds; gate FAIL.
- `likelihood_hmm`: mean NLL gain -0.066971; positive in 0/5 seeds; gate FAIL.

## Oracle-write reachability

- Three causal statistics: mean video-equal AUC 0.700658.
- Adding log memory mass: mean AUC 0.702022; gain 0.001364.

These probes use label-revealed oracle targets and cannot become a deployable or confirmatory result.

## Conclusion

The factorized h/Q/R claim remains **inconclusive/not tested**. Stream-51 has no orthogonal observation-noise and drift manipulation. Only a separately frozen synthetic factorial can test identifiability.
