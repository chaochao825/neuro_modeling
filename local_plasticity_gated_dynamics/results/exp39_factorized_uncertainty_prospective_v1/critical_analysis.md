# Exp39 Critical Interpretation

Status: post-outcome interpretation bound to the frozen formal package. The
registered joint verdict remains **support**, but the mechanistic conclusion is
narrower than the gate label alone suggests.

## What is supported

On pairwise and triple h/Q/R combinations absent from fitting, the
three-state factorized filter achieved mean predictive NLL 0.841257. The
fit-supported four-mode IMM reached 0.889265 and the selected stationary
filter reached 1.131837.

- Factorized minus selected fixed gain: +0.290580 nats, 95% seed-bootstrap
  interval [0.280122, 0.300594], positive in 30/30 seeds.
- Factorized minus seen-mode IMM gain: +0.048008 nats, interval
  [0.042136, 0.054094], positive in 30/30 seeds.
- Every utility and clamp test remains significant in the frozen five-test
  Holm family.
- h/Q/R clamp selectivity was +0.037097, +0.026683, and +0.052370 nats.

This supports a functional statement: a continuously factorized controller can
use the same three local states across unseen combinations more effectively
than selecting among joint modes observed during fitting.

## What is not supported

### The states are not calibrated parameter estimates

Block-level log-parameter correlations were 0.089 for h, 0.312 for Q, and
0.868 for R. Mean absolute log errors were 1.554, 1.219, and 0.647. Thus only
the observation-noise coordinate clearly tracked its generating quantity.
Selective clamp effects establish functional dependence; they do not prove
that h and Q are quantitatively identified.

The method should therefore be described as a factor-indexed functional state,
not as accurate online recovery of the true h/Q/R parameters.

### The advantage is not uniform across unseen cells

Factorized NLL minus seen-mode IMM, expressed as a gain where positive is
better, was:

| Held-out cell | Seen IMM minus factorized NLL |
|---|---:|
| 011 | +0.042614 |
| 101 | +0.020959 |
| 110 | -0.004334 |
| 111 | +0.132794 |

The average effect is robust across seeds but is largest when observation noise
is high and in the triple combination. The h+Q-only cell 110 does not improve.
The formal claim is therefore an aggregate unseen-composition result, not a
claim of universal cell-wise dominance.

### It does not switch faster

Immediately after a block transition, factorized NLL was 1.059377 versus
0.972679 for seen-mode IMM. Late in the block the ordering reversed:
0.796637 versus 0.878553. The advantage reflects adaptation to a composition
after evidence accumulates, not lower switch latency.

### It remains below privileged upper bounds

The eight-mode generator-supported IMM reached 0.764398 and the dynamic truth
filter reached 0.724684. The factorized method retained about 79% of the
eight-mode oracle gain over the selected fixed filter and remained 0.076859
nats worse than that upper bound. This is expected for an unprivileged
three-state learner, but it rules out oracle-equivalence language.

### In-distribution performance trades off

On the four modes present during fitting, seen-mode IMM outperformed the
factorized filter in every cell. The new method is useful for compositional
coverage, not as a universally superior stationary or in-distribution filter.

## Consequence for real data

Exp39 unlocks a new real-data attempt; it does not validate one. The previously
analyzed 30-session IBL cohort is outcome-exposed and cannot become
confirmatory evidence. Moreover, its categorical HMM already improved context
NLL without improving held-out choice log-loss. Any Exp40 claim therefore
requires a newly frozen session cohort and behavioral predictive utility; real
neural data remains locked until that behavioral gate passes.

