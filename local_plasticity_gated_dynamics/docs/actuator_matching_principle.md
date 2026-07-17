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
| Input-versus-internal demand predicts actuator-family advantage | Exp26, 30 independent seeds; held-out demand AUROC 0.9467 and Spearman 0.7605 | support for the synthetic special case |
| Low-dimensional task-descriptor selector improves over one fixed family | Exp29 independent one-shot confirmation; local-minus-fixed utility +0.1003, 95% CI [0.0954, 0.1047] | support for task-matched family policies |
| Associative demand produces routing-to-memory crossover | Exp30 constructed positive-control panel; 5/5 positive seeds, mean demand/advantage Spearman 1.0 | sanity trend-positive, formally inconclusive |
| Fixed motif parameters generalize broadly across unrelated tasks | Exp30 fixes its analytic motifs across the demand sweep, but Exp26/29 fit task-specific family policies | inconclusive |
| Selector infers demand from observations or scalar reward | Current selector receives prospective descriptors and candidate utilities | inconclusive / planned |
| Real neural or natural-task validation of the complete principle | Existing Exp25 and multi-session neural comparisons do not support this chain | inconclusive |

The full Actuator Matching Principle is therefore **partially supported but
formally inconclusive**. The immutable Exp26 and Exp29 evidence packages are not
rewritten by the Exp30 extension.

## Trend-first expansion

Exp30 adds a positive-control sanity test for the associative-memory axis
without introducing a broad baseline race. It uses one frozen high-rank
Dale-compatible carrier and one fixed scalar readout. The carrier bridge is
calibrated to transmit a scalar identity, so its high-rank dynamics do not
solve the task. A direct cue and a trial-local key--value retrieval are mixed by a
registered memory-demand coordinate. Routing, a one-dimensional compressive
state, associative outer-product memory, and a key--value-shuffled causal
control share trials, noise, carrier, and readout. Each mode/demand cell gets a
training-only gain that matches query-output RMS. This is not a matched
write/state/energy budget across all actuator families; only associative and
shuffled memory have exactly matched write L1/L2.

The five development seeds passed the registered trend gate. This licenses a
larger delay/load/demand sweep; it does not license a formal support claim. The
next scale step should retain only minimal mechanism controls: frozen, fixed
single actuator, shuffled memory, demand-matched oracle, and eventually a
learned observation-only selector. GRU/BPTT or sequence-model comparisons can
be added after the crossover and selector effects are stable at formal scale.

## Non-claims

- Exp30 is a constructed capability test, not a real-data result or a strong
  sequence-model benchmark.
- Because the target is the registered sum of a direct and a retrievable
  component, the crossover is a pipeline sanity check, not independent
  evidence that the carrier discovers the decomposition.
- Its matched selector reads the explicit demand coordinate; it is neither a
  hidden-belief estimator nor a locally learned four-actuator controller.
- The associative state is reset between trials and does not provide arbitrary
  long-history exact recall.
- The current results do not show end-to-end SOTA, universal network reuse,
  replacement of KV cache, or a complete biological implementation.
