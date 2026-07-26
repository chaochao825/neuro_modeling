# Topic Brief

## Working title

**Factorized Uncertainty for Compositional Memory Adaptation**

This is a pre-evidence successor question. `Actuator Matching Principle`
remains a broader programme and is not an established contribution.

## Paper question

Can a three-scalar causal state separately estimate abrupt hazard \(h_t\),
gradual process volatility \(Q_t\), and observation noise \(R_t\), then
compose factors that were observed only in isolation during fitting and
improve held-out predictive NLL over the best fixed filter and a seen-mode
interacting multiple-model baseline without test-time gradients?

Here, causal means current-and-past-only execution plus registered clamp
interventions. It does not mean causal discovery from observational data.

## Active evidence status

One bounded formal result now supports this question. Exp35--Exp38 are
historical failure-boundary evidence, archived in
`results/history/adaptive_memory_exp35_exp38_20260726.md`. They motivate the
separation of change and noise but cannot verify it.

The claim-ineligible Exp38 algebra diagnostic is complete: direct alpha
improved NLL in 5/5 seeds but its mean 0.003813-nat gain missed the frozen
0.005 reuse threshold, and likelihood-HMM was worse. Stream-51 is retired for
the successor.

Eight disclosed Exp39 development seeds selected factor-specific local update
timescales. Development v2 was scale evidence only. The hash-frozen 30-seed
formal run then improved unseen-composition NLL over selected fixed by
0.290580 nats and over seen-mode IMM by 0.048008 nats, both in 30/30 seeds.
All three clamp-selectivity gates passed the frozen Holm family.

The result is functionally positive but mechanistically mixed. Log-parameter
correlations were only 0.089 for h and 0.312 for Q, versus 0.868 for R.
Cell 110 did not improve and early post-switch NLL was worse than seen IMM.
Accordingly the paper may claim aggregate compositional predictive utility and
functional clamp selectivity, but not calibrated h/Q/R identification,
universal cell-wise dominance, or faster switching.

The active sequence is gated:

1. the completed claim-ineligible Exp38 diagnostic retires the semantic-splice
   video task without testing h/Q/R;
2. the completed 30-seed synthetic factorial fits only baseline and isolated
   h/Q/R elevations, then supports aggregate prediction on unseen pairwise and
   triple compositions;
3. a new, outcome-blind IBL cohort is now eligible, but the exposed historical
   cohort cannot provide confirmation and behavioral NLL must improve;
4. neural decoding and E/I interpretation remain locked behind behavior.

## Active hypothesis and stop rule

The candidate method is

\[
(h_t,Q_t,R_t)\rightarrow
\text{state-aware Bayesian gain or directional delta update}.
\]

The state dimension, Kalman filtering, IMM, input-dependent memory, and
directional forgetting are not claimed novel. The only eligible contribution
is a tightly constrained causal-state comparison under matched data and strong
baselines. A privileged eight-mode IMM and a time-varying oracle are reported
as upper bounds and are not targets to beat. Exp39 cleared the frozen
synthetic gate. The next stop rule is behavioral: failure to improve held-out
choice prediction on a newly frozen multi-animal IBL cohort prevents all
neural claims.

## Archived Exp35 paper contract (historical, not current)

Everything below this heading records the superseded consistency-audit paper
contract. It remains reproducible but does not define the active method claim.

### Historical scope

The empirical task is ORBIT clutter-video recognition. Labelled clean support
videos personalize a frozen recognizer for one held-out user, while each query
video is processed without future frames or query labels. Prototype,
support-derived feature gain, associative delta memory, and temporal evidence
accumulation share the same encoder, support set, query frames, order, and
noise realization.

Exp34 is retained as the motivating positive result. Exp35 is the decisive
same-tape audit. It compares the prefix-consistency selector against lagged
selection, cumulative class vote, equal and calibrated probability
accumulation, a validation-selected single prefix operator, and the strongest
fixed temporal operator. It also instantiates stable-wrong, noisy-correct,
prefix-bias, within-stream-switch, and boundary-switch controls.

The following topics are outside the paper: physical recurrent-matrix rank,
participating Dale E/I execution, biological MD/ACC identity, local recurrent
plasticity, shared neural dynamics, ARC, maze, Sudoku, HRM, and general
sequence-model replacement. They remain auditable programme-level work.

## Verified result

Exp35 completed five fixed seeds, all 17 ORBIT test users, and every registered
condition with zero failed or invalid rows. Seeds were averaged within user
before paired inference.

- Equal-weight prefix probability accumulation reached 81.90% user-equal
  accuracy; prefix consistency reached 75.92%.
- Prefix consistency was 5.99 percentage points worse than equal prefix
  probability accumulation (95% user-bootstrap interval −9.44 to −3.20;
  Holm-adjusted p=0.00055).
- It was also worse than cumulative prefix vote, a validation-calibrated
  prefix stack, and a validation-selected single prefix operator by 5.42--5.76
  points. Its comparison with the fixed temporal operator was inconclusive.
- In the exact stable-correct versus stable-wrong intervention, prefix
  consistency selected the stable wrong operator throughout: accuracy 0 and
  wrong-lock fraction 1.

The registered comparative and consistency-as-correctness claims are therefore
**opposed**. Because the test split had already been inspected, Exp35 is a
retrospective mechanism audit and cannot upgrade a positive claim.

## Surviving insight

The exploratory decomposition shows a large, consistent benefit from
accumulating class probabilities over the prefix within each operator:
+13.39, +12.38, +13.93, and +5.10 percentage points for prototype, gain,
delta, and temporal respectively. The incremental value of pooling multiple
operators over a validation-selected single prefix operator is inconclusive
(+0.57 points; interval crosses zero). Thus the evidence favors **temporal
evidence accumulation within one-object videos**, not heterogeneous actuator
routing.

This decomposition is post-hoc and must remain labelled exploratory until a
new cohort is frozen prospectively.

## Prospective follow-up boundary

Exp36 attempted the untouched-cohort extension first, but ORBIT-India failed
the frozen schema gate before outcome inference. Exp37 then froze a balanced
CORe50 contract before acquisition completed and executed all 40,500 planned
condition cells across five seeds and nine held-out sessions.

The registered explicit-change result is **oppose**. Hard-reset categorical
BOCPD reached 39.84% hidden-switch accuracy, exactly matching cumulative
accumulation, versus 93.13% for validation-selected retention zero and 92.21%
for a selected two-frame window. Oracle reset reached 95.28%. A post-hoc,
development-only audit found maximum change posterior 0.008529 against the
minimum frozen alarm threshold 0.2. This diagnoses the tested detector's scale
mismatch and bounds the rejection; it cannot rescue the prospective result or
support general change-point claims.

## Prospective successor result: continuous memory control

Exp38 is a new, outcome-locked Stream-51 study rather than a repair of Exp37.
It asks whether a three-statistic causal state can continuously control a
single memory retention on a frozen high-dimensional embedding.  Support
videos fit a vMF evidence model, development-fit videos select calibration and
controller parameters, a disjoint development holdout must pass five conjunctive
memory-demand and reachability gates, and only then may 381 external source
videos be featurized and scored.  The source video is the statistical unit.

The qualification result is **oppose** for the registered joint readiness gate:
0/5 seeds passed. Oracle reset headroom and cumulative post-switch harm passed
in 5/5 seeds, but stable accumulation cleared its MCID in 2/5 and causal
reachability in only 1/5. The untouched external split therefore remains
unfeaturized and the main held-out-utility claim is **inconclusive**. On the
qualification holdout, soft retention exceeded the stronger fixed baseline by
only +0.0019 on average across assembly seeds, far below the external MCID.
The stop rule ends this adaptive-memory method line without threshold retuning,
GRU/BPTT rescue, E/I extension, or representation tuning. Exp35 remains the
current ORBIT paper conclusion; Exp38 is a separate prospective stop result.

## Paper and venue decision

- Do not submit the current work as an ICLR method paper or claim a new
  reliability router.
- A credible route is a TMLR-style mechanism audit or negative-result paper
  centered on information boundaries, stable-error lock-in, and the difference
  between prediction concentration and correctness.
- An external cohort is useful only for a newly frozen study of causal prefix
  aggregation and failure boundaries. It must not be presented as a rescue of
  the rejected consistency selector.
- Do not invest in HMM, Hedge, GRU, sparse execution, or participating E/I for
  this paper unless a new research question is registered first; the simple
  equal-prefix baseline already resolves the current claim.

## Reproducibility constraints

- Python 3.11 and explicit recorded seeds.
- User, session, animal, or independent seed is the statistical unit.
- Splits occur by user, trial, or block, never by randomly sampled time point.
- Preprocessing, selection, dimensionality reduction, and calibration are fit
  inside training/development folds.
- Query labels and future frames are unavailable to deployable methods.
- BPTT and GRU are baselines only; no local-learning claim may use them.
- Failed, invalid, opposing, and inconclusive conditions remain first-class
  evidence.

The Exp00--Exp35 lineage is recorded in
`docs/experiment_lineage_diagnostic_audit_20260726.md`; Exp36--Exp37 are added
to `docs/current_evidence_critical_audit.md`. Exp35, not Exp34, defines the
current ORBIT paper conclusion; Exp37 and Exp38 are separate prospective stop
results for hard-reset and continuous-retention extensions respectively.
