# Actuator Matching Principle

> Flexible computation need not retrain a full network for every task. A
> system can reuse a fixed high-dimensional carrier and a small set of control
> motifs, selecting the actuator that best matches a task's demand on input
> mapping, internal dynamics, or associative memory.

This is the project's organizing hypothesis. It does **not** assert that one
actuator is universally best, that physical recurrent connectivity must be low
rank, or that a small fixed state can losslessly replace arbitrary history or a
general KV cache.

## Operational statement

For a task demand

\[
d_\tau=(\Delta B_\tau,\Delta A_\tau,\Delta M_\tau),
\]

let \(\mathcal T_k\) be the realizable tangent space of actuator family \(k\)
and let \(C_k\) and \(S_k\) denote control cost and stability penalty. The
working prediction is

\[
k^*(\tau)=\arg\min_k
\left[
\min_{v\in\mathcal T_k}\lVert d_\tau-v\rVert^2_{W_\tau}
+\lambda C_k+\mu S_k
\right].
\]

Input routing, population gain, low-rank internal operators, and associative
write/retrieve states are different actuator families. Their usefulness should
reverse as task demand moves between their realizable computations. A selector
is useful only if it improves held-out behavior or prediction over a globally
fixed actuator; low matrix rank alone is not evidence.

## Evidence ledger

| Component | Current evidence | Classification |
|---|---|---|
| High-rank physical updates can coexist with low-dimensional credit | Exp08 rank-stage audit | support for the revised mechanism framing |
| Hidden belief can be inferred without true-context access | Exp09 hidden-HMM gate; Exp11 real IBL behavior is mixed | support in synthetic data; bounded real-behavior evidence |
| Belief modulates effective dynamics on a frozen high-rank E/I receiver | Exp10 bridge and Exp21 full-trajectory audit | support with fixed receiver and scoped closure endpoints |
| Input-versus-internal demand predicts actuator-family advantage | Exp26, 30 independent seeds; held-out demand AUROC 0.9467 and Spearman 0.7605 | support for the synthetic special case |
| Low-dimensional task-descriptor selector improves over one fixed family | Exp29 independent one-shot confirmation; local-minus-fixed utility +0.1003, 95% CI [0.0954, 0.1047] | support for task-matched family policies |
| Executed scalar reward selects fixed motifs under hidden reliability | Exp31, 30 seeds; local-minus-fixed accuracy +0.0472, 95% CI [0.0459, 0.0485] | support for a reset-block two-actuator controller |
| Reward belief persists in a continuous hidden stream | Exp32 main cell; local-minus-fixed +0.0435, 95% CI [0.0345, 0.0529] | support for the bounded slow-switch endpoint |
| Performance follows the registered feedback-memory timescale structure | Exp32 iso-lambda effect +0.0119, below the 0.02 MCID | inconclusive; the Exp32 joint claim is not promoted |
| Reward-only motif selection improves real personalized video recognition | Exp33 had 0.106 oracle headroom but local-minus-fixed was -0.0316 across three development seeds | inconclusive and superseded; the bounded reward-only transfer failed |
| Label-free causal consensus selects motifs for unseen-user video | Exp34 is the active failure-driven successor and awaits eligible validation/formal results | inconclusive / open |
| Participating E/I carrier improves Exp31/32 behavior | Exp31/32 use fixed motifs without a participating E/I carrier | inconclusive / untested |
| Shared model wins on multi-animal neural data | Exp25 fails closed because eligible canonical neural inputs are absent | inconclusive / open |

The full Actuator Matching Principle is therefore **partially supported but
formally inconclusive**. The synthetic family-matching and bounded reward-only
controller layers support. Carrier participation, broad motif reuse, and real
neural generalization remain unverified.

## Evidence boundary

Current and historical evidence are intentionally disjoint. The
[current registry](../results/current/README.md) contains only still-active
foundations, core experiments, and open endpoints. The
[historical registry](../results/history/README.md) contains the original
physical-low-rank interpretation, the rejected rate-matched phase gate, the
failed Exp23 local gain-axis combination, HRM/ARC/Sudoku explorations, and
development panels superseded by independent confirmation.

This separation changes no original statistic. In particular, a historical
positive result remains positive in its archived report, but it no longer
supports the current theory. Failed cells and inconclusive attempts remain
hash-bound and auditable.

## Non-claims

- Exp29 uses prospective generator descriptors and a full candidate-utility
  teacher; it is not a reward-only or observation-only result.
- Exp31 selects only two fixed synthetic motifs and resets its controller at
  block boundaries.
- Exp32 removes block resets but its action values are policy proxies rather
  than calibrated context posteriors; its opposite-credit intervention is not
  update-budget matched.
- Exp31/32 do not contain a participating high-rank E/I carrier.
- The current shared neural model has not beaten common dynamics across the
  required animals and sessions.
- The current results do not show end-to-end SOTA, universal network reuse,
  replacement of KV cache, or a complete biological implementation.
