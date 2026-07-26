# Exp36 Prospective Protocol: Change-Aware Prefix Accumulation

Frozen: 2026-07-26T05:54:33Z, before downloading, extracting, embedding, or
evaluating ORBIT-India. Public metadata and file size were inspected; no image,
collector-level outcome, or model output was inspected.

Protocol version: `exp36_change_aware_prefix_v1`.

## Question

Does causal class-probability accumulation need forgetting or an online change
detector when the represented object can change inside one uninterrupted
stream?

Exp35 suggests that unbounded accumulation is useful when one query video
contains one object, but it fails after an unannounced within-stream switch.
Exp36 tests a new proposition. It does not attempt to rescue prefix-consistency
routing and does not claim that temporal averaging is novel.

## Evidence tiers

1. **Development only:** original ORBIT validation users. Hyperparameters and
   candidate selection use query labels only in this tier.
2. **Prospective external confirmation:** all 12 ORBIT-India collectors. No
   external label may fit, select, calibrate, stop, or modify a method.
3. **Controlled analytic tests:** synthetic categorical streams verify exact
   reset, false-alarm, and delay semantics. They cannot support the real-data
   claim.

The original ORBIT test users are prohibited from Exp36 tuning and primary
inference because they were exposed in Exp34/35.

## Data and representation

- ORBIT-India Zenodo record: `10.5281/zenodo.12608444`, version 1.
- Frozen archive name: `ORBIT-India dataset.zip`.
- Published MD5: `af4d05689f556ca477acc09a73bb3d3d`.
- Independent unit: collector, never frame, video, neuron, or algorithmic seed.
- Support: every eligible clean video for that collector, sampled with the
  frozen stride and cap below. Clean-frame issue annotations are not used.
- Query: clutter-video frames for evaluation. `object_not_present_issue` may be
  used only to apply the documented query eligibility rule; query labels remain
  metric-only.
- Encoder: frozen torchvision EfficientNet-B0 IMAGENET1K_V1, identical to
  Exp35. No external representation training or fine-tuning.
- Action evidence: equal-weight mean of the four Exp35 action probability
  vectors. The action bank, score conversion, and weights are frozen. This
  experiment changes only temporal state handling.

## Stream construction

Each sampled collector episode yields two paired panels.

### Natural constant-object videos

Every clutter video remains a separate stream with its observable video reset.
This panel measures ordinary accuracy and false resets. A change-aware method
must not obtain its switched-stream gain by degrading the natural regime.

### Hidden-switch streams

For each task, clutter-video segments from different object classes are
concatenated into one stream. The model receives one constant stream ID and no
video boundary or true class. Source order inside each segment is preserved.
Four segments of 32 frames are used where coverage permits, yielding three
hidden changes; unused videos remain recorded. Pairing and segment choice are
deterministic functions of the frozen seed, collector, and task index. A task
that cannot provide four eligible different-class segments is retained as
invalid rather than silently dropped.

## Conditions

All conditions consume the same framewise probability evidence.

1. `current_frame`: no temporal state.
2. `cumulative`: unbounded prefix sum, reset only at an observable stream
   boundary.
3. `fixed_forgetting`: exponentially discounted evidence; retention selected
   on ORBIT validation.
4. `sliding_window`: exact finite window; width selected on ORBIT validation.
5. `jsd_change_reset`: cumulative task state plus a two-timescale detector.
   It raises an alarm only when the fast and pre-update task distributions have
   different modal classes and Jensen-Shannon divergence exceeds the frozen
   threshold for the frozen patience. On alarm, the task state is replaced by
   the fast state. No label or future frame is used.
6. `jsd_score_no_reset`: identical detector computation without changing the
   task state.
7. `matched_shifted_reset`: uses the same number of resets as
   `jsd_change_reset`, circularly shifted by a deterministic nonzero offset.
   This nondeployable intervention tests reset timing rather than reset count.
8. `oracle_change_reset`: resets at the true hidden switch. It is an
   evaluation-only ceiling and never deployable.

## Frozen development grids and selection

- Fixed retention: `[0.0, 0.5, 0.7, 0.85, 0.93, 0.97, 0.99, 1.0]`.
- Sliding window: `[4, 8, 16, 32, 64]` frames.
- Detector fast retention: `[0.0, 0.5, 0.8, 0.9]`.
- Detector JSD threshold: `[0.01, 0.02, 0.05, 0.10, 0.20]`.
- Detector patience: `[1, 2, 3]` frames.
- Detector minimum run: `[4, 8]` frames.

Fixed forgetting and sliding window maximize validation hidden-switch
accuracy, with larger retention/window winning exact ties.

The detector is selected only among candidates satisfying both validation
constraints:

- natural-video accuracy loss versus cumulative no worse than 1 percentage
  point;
- false alarms no greater than 5 per 1,000 natural-video frames.

Among eligible candidates, maximize validation hidden-switch accuracy, then
minimize mean detection delay, then false alarms, then choose the lexicographic
parameter tuple. If no detector is eligible, the detector condition is
fail-closed and the external detector claim is `inconclusive`; constraints may
not be relaxed.

## External evaluation and statistics

Run five fixed algorithmic seeds: `13600` through `13604`, with 50 sampled
tasks per collector. Average seeds and tasks within collector before inference.
All 12 documented collectors are required. Missing collectors, incomplete
condition coverage, checksum mismatch, feature failures, or nonfinite metrics
invalidate the prospective claim rather than reducing the cohort.

Primary registered comparisons form one Holm family:

1. hidden-switch accuracy: `jsd_change_reset - cumulative`;
2. hidden-switch accuracy: `jsd_change_reset - fixed_forgetting`;
3. post-switch frames 1--16: `jsd_change_reset - fixed_forgetting`;
4. natural-video accuracy: `jsd_change_reset - cumulative`.

Inference uses paired collector bootstrap intervals and exact paired sign-flip
tests. Algorithmic seeds, tasks, frames, and videos are nested observations.

## Claim gates

The bounded external claim is `support` only if all conditions hold:

- all 12 collectors and every registered condition complete;
- mean hidden-switch gain over cumulative is at least 3 percentage points,
  its 95% collector-bootstrap interval excludes zero, and Holm p is at most
  0.05;
- mean hidden-switch gain over fixed forgetting is positive, its interval
  excludes zero, and Holm p is at most 0.05;
- natural-video accuracy loss versus cumulative is no worse than 1 percentage
  point and the lower 95% interval is above -2 percentage points;
- median detector delay is at most 8 frames;
- natural-video false alarms are at most 5 per 1,000 frames;
- `jsd_change_reset` exceeds both `jsd_score_no_reset` and
  `matched_shifted_reset` descriptively, so a score-only or reset-count
  explanation is not sufficient.

If a registered effect is significantly negative, classify the corresponding
claim `oppose`. Otherwise classify it `inconclusive`. No OR rule may promote
the joint claim.

## Stop rules

- If fixed forgetting matches or beats the detector, stop method development
  and report that stationary exponential memory is sufficient at this scale.
- If the detector violates the natural-video or false-alarm constraint on
  validation, do not evaluate a relaxed version externally.
- If ORBIT-India extraction or collector coverage is incompatible with the
  frozen protocol, preserve the failure and revise only under a versioned v2;
  v1 can never regain confirmatory status.
- External results may not change grids, episode construction, eligibility,
  action weights, thresholds, endpoints, multiplicity, or prose claims.

## Non-claims

- no SOTA, novelty of EMA/change-point detection, or general concept-drift
  claim;
- no compute advantage unless measured separately;
- no inference from frames or videos as independent replicates;
- no E/I, local-plasticity, neural-data, or biological claim;
- no claim that ORBIT-India natural videos contain within-video object changes;
  hidden-switch streams are explicitly constructed from real segments.
