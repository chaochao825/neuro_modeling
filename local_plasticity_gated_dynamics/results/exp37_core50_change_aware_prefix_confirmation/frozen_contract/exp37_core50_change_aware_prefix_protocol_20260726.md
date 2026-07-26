# Exp37 Prospective Protocol: Bayesian Change-Aware Prefix Accumulation on CORe50

Frozen: 2026-07-26T07:07:28Z, before downloading, extracting, embedding, or
evaluating CORe50. Only the official dataset description, directory schema,
license, archive URL, and HTTP headers were inspected. No image, embedding,
prediction, or task outcome was inspected.

Protocol version: `exp37_core50_change_aware_prefix_v1`.

## Question and scope

Does a causal prefix-probability state benefit from explicit Bayesian
change-point inference when an object changes inside an otherwise uninterrupted
visual stream, beyond both unbounded accumulation and validation-selected
stationary forgetting?

This is a prospective decision-utility test, not a claim that probability
averaging, exponential forgetting, or Bayesian online change-point detection
(BOCPD) is novel. It does not evaluate continual parameter learning and must not
be presented as an official CORe50 continual-learning benchmark result.

## Dataset freeze and outcome-blind schema gate

- Source: official CORe50 128x128 archive,
  `http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip`.
- Frozen HTTP metadata: content length `5892103007`, ETag
  `"03beef164c9d21:0"`, last modified `Wed, 10 May 2017 08:10:54 GMT`.
- License stated by the official project: CC BY 4.0.
- Expected schema: sessions `s1`--`s11`, objects `o1`--`o50`, with at least one
  image in every session/object cell.

Before feature extraction, an outcome-blind validator must confirm all 550
session/object cells, unique image paths, nonempty files, and an archive SHA256.
Schema failure invalidates v1; it may not be repaired by dropping sessions or
objects. The observed SHA256 is an acquisition attestation, not a value learned
from outcomes.

## Locked data roles and representation

- `s1`: support only; forms one frozen prototype per object.
- `s2`: development only; selects temperature and temporal hyperparameters.
- `s3`--`s11`: nine locked evaluation sessions; no labels from these sessions
  may fit, calibrate, select, stop, or modify a method.
- Independent inferential unit: evaluation session. Seeds, sampled tasks,
  objects, and frames are nested within session.
- Encoder: frozen torchvision EfficientNet-B0
  `IMAGENET1K_V1`; no fine-tuning or external-session adaptation.
- Prototype: normalized mean of at most 60 temporally ordered `s1` embeddings
  per object, selected with a fixed stride of 5.
- Frame evidence: temperature-scaled softmax of cosine similarity to the four
  task prototypes. Temperature is selected on `s2` only.

## Frozen task construction

For each seed and session, sample 50 four-object tasks without replacement
inside a task. A deterministic function of seed, session, task, and object
chooses contiguous frame segments; source order is never shuffled within a
segment.

Each task yields paired panels from exactly the same selected objects and frame
evidence:

1. **Natural:** four separate constant-object streams of 128 consecutive
   frames. Observable stream boundaries reset every method. This measures
   stationary accuracy and false alarms.
2. **Hidden switch:** six 32-frame segments are concatenated, adjacent segments
   always use different selected objects, and the model receives a single
   constant stream ID. Five class changes are therefore hidden from the model.

All failed tasks and cells are retained. A missing session/object cell,
insufficient frames, nonfinite feature, or incomplete condition panel
invalidates the confirmatory cohort rather than reducing it.

## Conditions

Every condition consumes identical causal frame-probability vectors.

1. `current_frame`: current evidence only.
2. `cumulative`: unbounded prefix sum.
3. `fixed_forgetting`: exponential state with retention selected on `s2`.
4. `sliding_window`: exact finite window selected on `s2`.
5. `bocpd_change_reset`: a categorical BOCPD run-length posterior with
   constant hazard and symmetric Dirichlet prior. Soft class probabilities are
   fractional categorical counts. When posterior change probability crosses
   the selected threshold after the minimum run, cumulative task state is
   replaced by current evidence.
6. `bocpd_posterior`: posterior-weighted expected within-run counts from the
   same BOCPD filter; no thresholded reset is used for its decision.
7. `bocpd_score_no_reset`: computes the identical BOCPD score and alarms while
   leaving cumulative task state unchanged.
8. `matched_shifted_reset`: applies the same number of resets as
   `bocpd_change_reset` at a deterministic nonzero circular shift.
9. `oracle_change_reset`: resets at the true hidden changes; evaluation-only
   upper reference.

BOCPD state is causal, truncated at 128 frames, renormalized after truncation,
and never receives the true class, future evidence, source boundary, or session
label.

## Frozen development grids and selection

- Evidence temperature: `[1, 2, 5, 10, 20]`.
- Fixed retention: `[0, 0.5, 0.7, 0.85, 0.93, 0.97, 0.99, 1]`.
- Sliding window: `[2, 4, 8, 16, 32, 64]`.
- BOCPD hazard: `[0.0078125, 0.015625, 0.03125, 0.0625]`.
- Symmetric prior concentration: `[0.1, 1, 5]` total counts.
- Alarm threshold: `[0.2, 0.4, 0.6, 0.8]`.
- Minimum run: `[2, 4, 8]` frames.

Temperature maximizes mean `s2` current-frame natural accuracy. Fixed
forgetting and sliding window separately maximize `s2` hidden-switch accuracy;
larger memory wins exact ties.

BOCPD hard-reset candidates must have natural accuracy loss versus cumulative
no worse than one percentage point and at most five false alarms per 1,000
natural frames on `s2`. Among eligible candidates, maximize hidden-switch
accuracy, then post-switch accuracy, then minimize delay and false alarms, then
use lexicographic parameter order. If none is eligible, the external BOCPD
claim is fail-closed and inconclusive; constraints may not be relaxed.

## External inference

Run seeds `13700`--`13704`, 50 tasks per evaluation session. Average seeds,
tasks, and the four natural streams within each of the nine sessions before
inference. All nine sessions and all registered conditions are required.

The primary Holm family contains:

1. hidden-switch accuracy: `bocpd_change_reset - cumulative`;
2. hidden-switch accuracy: `bocpd_change_reset - fixed_forgetting`;
3. hidden-switch accuracy: `bocpd_change_reset - sliding_window`;
4. natural accuracy: `bocpd_change_reset - cumulative`.

Use paired session bootstrap confidence intervals and exact paired sign-flip
tests. Frame, object, task, and seed are never treated as independent repeats.

## Claim gates

The bounded claim is `support` only if every condition holds:

- exact nine-session, five-seed, 50-task, two-panel, nine-condition coverage;
- hidden-switch gain over cumulative is at least 3 percentage points, its 95%
  session-bootstrap interval excludes zero, and Holm p <= 0.05;
- hidden-switch gains over both validation-selected stationary baselines are
  positive, their intervals exclude zero, and Holm p <= 0.05;
- natural accuracy loss versus cumulative is no worse than one point and the
  lower 95% interval is above -2 points;
- median detection delay is at most eight frames and false alarms are at most
  five per 1,000 natural frames;
- hard reset exceeds both score-only and matched-shifted reset descriptively.

`bocpd_posterior`, current-frame, and oracle conditions are registered
diagnostics, not alternative paths through the joint support gate. A
significantly negative registered effect is `oppose`; all other failures are
`inconclusive`. No OR rule may promote the claim.

## Stop rules and non-claims

- If either stationary forgetting baseline matches or beats hard-reset BOCPD,
  conclude that explicit change detection is unnecessary at this scale and
  stop tuning this controller on CORe50.
- Low detector delay or low false-alarm rate without decision gain is not
  support.
- External outcomes cannot change grids, task construction, exclusions,
  multiplicity, thresholds, or prose claims.
- No SOTA, continual-learning, biological, neural-data, E/I, local-plasticity,
  compute, or universal concept-drift claim is permitted.
