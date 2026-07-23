# Exp33: causal actuator matching on ORBIT

## Question

Can a low-dimensional controller, learned only from the reward of the action it
actually executed, select among reusable few-shot computation motifs and
improve personalized object recognition for unseen people?

This experiment is the first end-to-end task gate for the Actuator Matching
Principle. It intentionally tests the task-space claim before adding a Dale E/I
carrier. A biological carrier cannot rescue a selector that has no measurable
headroom on real observations.

## Public task and split

The primary endpoint is ORBIT clean-support/clutter-query video evaluation.
The official user-disjoint split is copied into
`data/orbit_official_splits.json`. Test videos are processed causally: a
prediction may depend on the current and preceding frames of the same video,
never on future frames or query labels. The official random frame subset is
sorted by original frame index before a temporal actuator sees it.

Development uses four validation users for controller fitting and two held-out
validation users for evaluation. It can only produce an `inconclusive` claim.
The formal configuration is fail-closed until the development headroom,
causality, and cache-provenance gates pass; it then uses all train users for
fitting and all 17 test users for inference, with 50 tasks per test user.

## Paired computation motifs

All conditions share the exact frozen encoder, support frames, query frames,
class order, and task sampling.

- `prototype`: cosine nearest prototype.
- `gain`: support-only diagonal discriminability gain plus prototypes.
- `delta`: support-written class-by-feature delta memory.
- `temporal`: causal leaky accumulation of prototype evidence, reset at every
  video boundary.
- `train_fixed_best`: one actuator chosen only on fitting users.
- `reward_only_local`: a local contextual reward predictor updated only for the
  executed actuator.
- `credit_shuffled_local`: identical controller and tape, but reward eligibility
  is assigned to the next actuator.
- `oracle_per_frame`: diagnostic upper bound that sees query labels; it is never
  a deployable method.

The controller context contains only prototype confidence/margin, embedding
change, video boundary and elapsed time, plus support dispersion/count. There
is no true task-demand label and no counterfactual reward in the main update.
The controller receipt explicitly records `used_autograd=false` and
`used_bptt=false`.

## Metrics and statistical unit

Raw rows preserve every task/video condition, including failures. Frame
accuracy is first averaged within task/video, then within user. Formal
bootstrap and paired randomization use users—not frames—as independent units.
The report also includes action disagreement, per-frame oracle headroom,
actuator use, selected-event cost, support-write cost, and local update L1/L2.

The claim is not supported by a low-rank or low-dimensional diagnostic alone.
It requires registered held-out task improvement over `train_fixed_best`, plus
credit specificity relative to `credit_shuffled_local`. A validation-only run
remains inconclusive even if the numerical trend is positive.

## Scale gate

1. Cache provenance, annotation completeness, chronological ordering, and
   train/evaluation user separation must pass.
2. Mean actuator disagreement must be at least 0.05 and mean per-frame oracle
   headroom at least 0.01 on the held-out validation users.
3. `reward_only_local` should have a positive trend over `train_fixed_best` and
   `credit_shuffled_local`.
4. Only then download/encode train and test, authorize the frozen formal JSON,
   and run the public 17-user protocol.

If the headroom gate fails, the result rejects these actuator definitions on
this encoder; it does not justify scaling an E/I realization.
