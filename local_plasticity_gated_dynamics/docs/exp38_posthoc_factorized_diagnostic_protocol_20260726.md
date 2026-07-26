# Exp38 Post-Hoc Factorized-Memory Diagnostic Protocol

Status: **revealed-data diagnostic only**  
Claim upgrade allowed: **no**  
External Stream-51 access allowed: **no**

## Question

Did Exp38 leave a recoverable algebraic or actuator-parameterization gap on its
already revealed qualification videos, even though its registered controller
family failed?

This diagnostic does not test the new \((h,Q,R)\) hypothesis. Stream-51 does
not orthogonally manipulate abrupt hazard, gradual process noise, and
observation noise, so those states are not identifiable from this panel.

## Frozen data boundary

- Reuse the exact Exp38 support/development feature cache, vMF model,
  temperature, five assembly seeds, and per-seed fit/qualification video keys.
- Hyperparameters are selected on the original development-fit videos only.
- All reported comparisons are evaluated on the disjoint, already revealed
  qualification videos.
- The 381 external videos remain unfeaturized and unscored.
- Source video remains the grouping unit; frames are never treated as
  independent replicates.

## Diagnostic conditions

1. **Posterior EMA:** the selected Exp38 fixed-retention recursion.
2. **Direct-alpha posterior filter:**
   \(p_t=(1-\alpha)p_{t-1}+\alpha q_t\), with \(\alpha\) selected on fit
   videos. This directly controls effective learning rate rather than memory
   mass-dependent retention.
3. **True-switch direct alpha:** set \(\alpha_t=1\) at revealed boundaries and
   use the fit-selected \(\alpha\) otherwise. This is an oracle diagnostic,
   not deployable evidence.
4. **Likelihood HMM:** accumulate the frozen vMF relative log likelihood with a
   fit-selected symmetric transition hazard. This tests correct likelihood
   algebra against linear posterior averaging.
5. **True-switch likelihood reset:** reset the likelihood belief to uniform at
   revealed boundaries. This is a second oracle diagnostic.
6. **One-step oracle write target:** for each frame, choose between keeping the
   previous belief and replacing it with current evidence according to the
   lower label-revealed one-step NLL. A train-only logistic probe predicts this
   target from surprise, entropy, disagreement, with and without log memory
   mass. This measures recoverable ranking signal only.

## Selection and endpoints

- Direct-alpha grid: `[0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]`.
- HMM hazard grid: `[0.0, 1/192, 1/96, 1/48, 1/24, 1/12, 1/6, 1/3]`.
- Select each candidate by source-video-equal hidden-stream NLL on fit videos;
  ties prefer the simpler/lower-update candidate.
- Primary descriptive endpoint: source-video-equal qualification NLL.
- Secondary endpoints: source-video-equal accuracy, post-switch NLL, oracle
  write AUC and Brier score, and the gain from adding log memory mass.
- Summaries are paired by the five registered assembly seeds. No p-value from
  frames or neurons is permitted.

## Interpretation gate

- The diagnostic may justify retiring or retaining Stream-51 as a development
  task. It cannot support external generalization or the factorized controller.
- Retain Stream-51 only if direct alpha or likelihood HMM improves qualification
  NLL over posterior EMA in at least 4/5 seeds and the mean improvement is at
  least 0.005 nats per frame.
- Otherwise retire this task construction. Do not retune Exp38 and do not open
  its external split.
- A new synthetic \((h,Q,R)\) factorial remains scientifically eligible because
  it supplies the orthogonal interventions absent from Stream-51; it must be
  frozen as a separate prospective experiment before outcomes are generated.

