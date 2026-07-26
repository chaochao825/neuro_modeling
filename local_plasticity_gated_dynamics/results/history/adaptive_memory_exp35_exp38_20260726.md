# Adaptive-Memory Lineage: Exp35--Exp38

Status: **historical evidence; no claim upgrade allowed**  
Archived on: 2026-07-26  
Successor question: factorized uncertainty state \((h_t,Q_t,R_t)\), not another
scalar-retention controller.

## Why this line is historical

Exp35--Exp38 progressively tested whether causal prefix statistics could select
or continuously control a memory horizon on frozen visual representations. The
line established memory headroom but did not establish a useful adaptive
controller. Its implementations, failed gates, raw metrics, logs, receipts,
and untouched-test decisions remain reproducible; none is current positive
method evidence.

| Experiment | What was tested | Verified result | Diagnosis and stop boundary |
|---|---|---|---|
| Exp35 | Prefix consistency versus matched prefix vote/probability accumulation on ORBIT | **Oppose**: equal prefix probabilities reached 0.8190 versus 0.7592; stable-wrong accuracy was 0 | Concentration is not correctness. The positive accumulation decomposition is retrospective and constant-object-specific. |
| Exp36 | Prospective change-aware prefix accumulation on ORBIT-India | **Inconclusive/invalid schema**: only 4/12 collectors could instantiate every frozen task | Outcomes were not inspected. The surviving subset was not analyzed. |
| Exp37 | Categorical BOCPD hard reset on CORe50 | **Oppose**: hard reset 0.3984 versus fixed forgetting 0.9313 and window 0.9221 | The frozen detector never reached its alarm grid. A post-hoc scale audit diagnoses but cannot rescue the verdict. |
| Exp38 | Three-statistic continuous scalar retention on Stream-51 | **Qualification opposed; external inconclusive**: joint gate 0/5, reachability 1/5 | Oracle headroom and switch harm passed 5/5, but soft utility averaged only +0.0019 over the stronger fixed baseline. All 381 external videos remain unfeaturized. |
| Exp38 post-hoc diagnostic | Direct learning rate, likelihood-HMM algebra, and oracle-write reachability on the already revealed qualification split | **Stream-51 retired; h/Q/R not tested** | Direct \(\alpha\) reduced video-equal NLL in 5/5 seeds but by only 0.003813 nats on average, below the frozen 0.005 reuse threshold. Likelihood-HMM worsened NLL by 0.066971 nats. Oracle-write AUC was 0.700658 from the three statistics and 0.702022 after adding log memory mass. |

## Mechanistic lessons retained

1. Adaptive memory has an oracle upper bound on the tested streams, but oracle
   headroom is not evidence that a deployable actuator can reach it.
2. Exp38 hid the boundary flag, not the semantic consequence of the boundary:
   adjacent segments were forced to have different visible object classes.
   The task therefore did not require a genuinely latent context inferred from
   history.
3. Averaging normalized class posteriors has no general Bayesian evidence
   interpretation. Conditionally independent observations require likelihood
   accumulation; correlated observations require an explicit effective-sample
   correction.
4. Observation unreliability and environmental change have opposite normative
   effects on learning rate. Exp38 combined surprise, entropy, and fast/slow
   disagreement into one change score and wrote every frame with full weight.
5. Retention \(\lambda_t\) is not a state-invariant learning-rate actuator:
   \(\alpha_t=(\lambda_tN_{t-1}+1)^{-1}\). The same \(\lambda_t\) has a
   different effect at different accumulated memory mass \(N_{t-1}\).
6. A global scalar erases every class and embedding direction together. It
   cannot express observation-gated directional updates.

## Artifacts preserved

- Exp35 report: `results/exp35_prefix_reliability_audit_retrospective_v2/report.md`
- Exp36 outcome-blind audit: `results/exp36_v1_invalid_schema_audit/report.md`
- Exp37 report and raw external panel:
  `results/exp37_core50_change_aware_prefix_confirmation/`
- Exp38 report, all five failed qualification runs, logs, frozen contracts, and
  byte manifest: `results/exp38_stream51_soft_memory_prospective_v1/`
- Claim-ineligible Exp38 diagnostic, raw per-video metrics, all scalar-selection
  rows, and run log:
  `results/history/exp38_factorized_memory_diagnostic_v1/`
- The first diagnostic launch failed before data access because four frozen
  tests differed from their implementation receipt by one terminal newline.
  Its error log is preserved under
  `results/history/exp38_factorized_memory_diagnostic_failures/`; the files
  were restored to their registered SHA-256 values before the successful run.
- Experiment entry points remain in `experiments/exp35_*.py` through
  `experiments/exp38_*.py`.

The Exp38 artifact manifest contains 105 hash-checked files. The external
feature manifest had zero rows when this line was archived.

The diagnostic did not clear its method-specific reuse gate. Stream-51's
semantic-splice task is therefore retired for the successor method. This is a
task decision, not evidence against a factorized \(h/Q/R\) state: the dataset
does not independently manipulate jump hazard, gradual drift, and observation
noise.

## What may and may not carry forward

May carry forward:

- a frozen high-dimensional representation;
- a causal low-dimensional state with no test-time gradient;
- train-only calibration and trial/block/video-level splits;
- held-out predictive utility as the primary endpoint;
- fail-closed qualification and preserved negative results.

May not carry forward as a positive claim:

- prefix consistency as reliability;
- categorical BOCPD hard reset;
- surprise/entropy/disagreement mapped to one scalar retention;
- Stream-51 semantic splice boundaries as a latent-context task;
- E/I, MD/PFC, neuromodulation, or neural decoding before behavioral utility.

## Successor hypothesis

The only eligible computational successor separates abrupt changes, gradual
process drift, and observation noise:

\[
(h_t,Q_t,R_t)\longrightarrow
\text{Bayesian effective learning rate or directional delta update}.
\]

The state dimension alone is not novel. Joint volatility/noise estimation,
IMM filtering, probabilistic prototype tracking, input-dependent memory, and
directional forgetting are established. A future claim requires stable
held-out NLL improvement over the best fixed filter and a strong IMM baseline,
plus distinct preregistered consequences of clamping \(h\), \(Q\), and \(R\).

## Successor development record: Exp39

These rows are retained development evidence and cannot support the formal
claim.

| Attempt | Result | Decision |
|---|---|---|
| Dev v1, seeds 39000--39007 | Factorized over selected fixed: +0.248542 nats, 8/8. Over seen-mode IMM: +0.001515 nats, 4/8. Clamp selectivity was positive in 7/8 h, 8/8 Q, and 6/8 R seeds. | Did not freeze. The local online-EM process update was too slow: mean low/high Q estimates were only 0.01816/0.02176 for true 0.0025/0.04. |
| Factor-specific beta probe | The best disclosed development region used slow h and faster Q/R updates; raw rows remain in results/development/exp39_beta_probe_v1.csv. | Mechanistic correction only: separate factor timescales instead of changing the generator, endpoint, baseline, or gate. |
| Dev v2, same eight development seeds | Factorized over selected fixed: +0.288798 nats, 8/8. Over seen-mode IMM: +0.041772 nats, 8/8. h/Q/R clamp selectivity was +0.032530/+0.029874/+0.053642 nats and positive in 8/8 for every factor. | Authorized a new, untouched 30-seed formal run. Development v2 remains scale-only. |

The formal question was narrowed to compositional uncertainty generalization:
fit conditions contain baseline and one-factor elevations only; pairwise and
triple elevations are held out. The required IMM contains only the four fit
modes. An eight-mode generator-supported IMM and a dynamic truth filter are
reported as privileged upper bounds, preventing an invalid claim that an
approximate three-state filter should beat an oracle Bayes model.

The first formal launch was serial. It was stopped outcome-blind after writing
only completion statuses, and its incomplete directory is preserved under
results/history/exp39_serial_execution_attempt_incomplete_20260726/. The
execution-only amendment dispatches the unchanged hash-frozen seed function to
six CPU workers; no scientific code, seed, tape, threshold, or output was
changed.
