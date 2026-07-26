# Exp41 matched-Q/R development decision

Status: **historical mixed development result; no formal advancement**.

Exp41 was frozen before outcome inspection at commit
`0f9fbc861cc21799752daa4706c8a05eb7e1128d` and tag
`exp41-dev-v1-preoutcome-20260727`. All eight disclosed development seeds
completed, no reserved formal seed was accessed, and the independent
[replay receipt](../../provenance/exp41_development_replay_receipt_210_20260727.json)
passed artifact/source hashes, clean run-start provenance, tape regeneration,
fit-only argmin selection, seed/aggregate identity, and summary replay. The
complete raw artifact is retained at
[results/exp41_matched_identifiability_development_v1](../exp41_matched_identifiability_development_v1/report.md).

## Decision

The experiment supports statistical discrimination of matched Q/R regimes,
but opposes the utility claim for the tested controller:

\[
\boxed{\text{Q/R information is identifiable, but the current slow estimator
does not turn it into timely predictive utility.}}
\]

Therefore Exp41 does not advance to formal seeds, the same development seeds
must not be retuned, and Exp42 remains unexecuted under its registered entry
gate. The machine-readable decision is
[`exp41_development_decision_20260727.json`](../../provenance/exp41_development_decision_20260727.json).

## Aggregate held-out utility

Positive values mean the named baseline has higher NLL than autocovariance,
so positive favors autocovariance.

| Baseline minus autocovariance | Mean NLL gain | Positive seeds | Development interpretation |
|---|---:|---:|---|
| selected fixed | +0.003886 | 4/8 | inconclusive; interval crosses zero |
| current online-EM | **-0.021208** | 0/8 | opposes improvement |
| h + total variance | **-0.010722** | 3/8 | opposes reduced-factor superiority |
| generator-supported IMM | -0.059558 | 0/8 | privileged reference remains better |
| dynamic Q/R oracle | -0.074846 | 0/8 | substantial inference/control headroom remains |

Latent MSE is also not rescued: autocovariance is 0.019494 versus 0.019189 for
total variance and 0.018358 for current online-EM. Thus the NLL failure is not
merely an unfavorable variance calibration with better latent estimates.

## Identifiability succeeds

The equal-marginal pairs have the same \(Q+2R\), so lag-zero variance alone
cannot separate them. The autocovariance estimator nevertheless orders both Q
and R correctly in both pairs in 8/8 seeds:

| Pair | Q separation | Fraction of true Q difference | R separation | Fraction of true R difference |
|---|---:|---:|---:|---:|
| m06 | +0.023179 | 61.8% | +0.010180 | 54.3% |
| m12 | +0.030732 | 43.9% | +0.015605 | 44.6% |

Its mean absolute log errors for Q/R are 0.807/0.397, better than total
variance at 1.205/0.527. This supports the lagged-covariance identity as an
estimation mechanism. It does not establish downstream utility.

## Timing explains the failure

Against the reduced total-variance controller, baseline-minus-autocovariance
NLL is negative at every registered transition window:

| Window | 1 | 4 | 8 | 16 | Late 16 |
|---|---:|---:|---:|---:|---:|
| total variance minus autocovariance | -0.042843 | -0.077890 | -0.090838 | -0.136108 | +0.016334 |

The selected autocovariance decay is at least 0.98 in every seed; prior mass 16
is selected in 7/8 seeds. The method therefore pays a transition cost and only
becomes useful late. Its 9.23% mean parameter-saturation fraction, driven
mainly by 8.65% Q clipping, is an additional robustness warning.

Cell-wise utility is also heterogeneous. Relative to total variance,
autocovariance improves only `m06_r_dominant` on average (+0.019773, 6/8). It
loses on `m06_q_dominant` (-0.019342), `m12_q_dominant` (-0.031239), and
`m12_r_dominant` (-0.012081). Aggregate or parameter-recovery plots must not
hide this pattern.

## Claim classification

| Claim | Conclusion |
|---|---|
| Matched-Q/R statistical discrimination | **Support**, development scope |
| Predictive utility beyond total variance | **Oppose** |
| Predictive utility beyond current online-EM | **Oppose** |
| Fast post-transition adaptation | **Oppose** |
| Late-regime adaptation | **Inconclusive**; descriptive positive only |
| Matched-budget efficiency | **Inconclusive**; budgets were measured, not matched |
| Real behavior or neural utility | **Inconclusive / untested** |

The correct surviving insight is not a successful three-factor controller. It
is the separation between **statistical identifiability** and **control
utility**: a slow local sufficient statistic can recover the sign of Q/R
differences while degrading predictions at the times when adaptation matters.
Any future fast-event/slow-uncertainty study must use a new prospective
contract and untouched development data; it cannot be presented as a rescue
of this run.
