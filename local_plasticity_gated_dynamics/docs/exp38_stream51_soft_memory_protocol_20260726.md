# Exp38 prospective protocol: causal soft memory on Stream-51

Status: **prospective, staged, outcome locked**.  This protocol defines a new
question after Exp37 opposed categorical BOCPD plus hard reset on CORe50.  It
does not amend, rerun, or reinterpret Exp37.

Outcome-blind amendment at `2026-07-26T11:06:26Z`: before any classification,
qualification, or external outcome was inspected, the oracle was made
controller-independent and fixed to cumulative retention with an exact reset
at true switches.  The earlier hash-bound draft is retained under ignored
`trash/20260726-192000-exp38-pre-oracle-amendment/`.

Outcome-blind schema amendment at `2026-07-26T11:09:13Z`: the first feature
attempt stopped before creating an embedding because some official tracker
bounds extend slightly outside the JPEG canvas (63 right-edge, 23 bottom-edge,
and 7 left-edge coordinate events; these sets need not be disjoint).  The validator now follows the
official loader and requires the box to be non-empty after clipping.  The
failed log is retained, and this did not expose a classification or task
outcome.

Outcome-blind execution amendment at `2026-07-26T11:17:56Z`: after 90 of 755
support/development videos had been cached, measured throughput was 15.2
videos/min while the A800 was idle.  The attempt-2 process was stopped between
atomic video writes and retained.  Eight ordered CPU decode workers now feed
the unchanged deterministic transform; a unit test verifies bit-identical
frame order and tensor output against serial decoding.  No classification or
task outcome had been computed.

## Registered question

With a fixed ImageNet representation and no test-time gradient updates, can a
three-dimensional causal control state continuously regulate evidence
retention so that it uses history within stable videos and forgets history
after hidden source-video switches better than every development-selected
fixed time scale?

The memory is

\[
S_t=\lambda_tS_{t-1}+w_tq_t,\qquad
N_t=\lambda_tN_{t-1}+w_t,\qquad
\hat p_t=S_t/(N_t+\epsilon).
\]

The deployed controller receives only predictive surprise, normalized
observation entropy, and fast/slow belief disagreement.  Its scalar risk is a
logistic function of these three development-standardized quantities, and
retention interpolates continuously between frozen lower and upper bounds.
The primary model fixes \(w_t=1\), so the primary contrast isolates memory
horizon rather than adding a confidence-gain mechanism.

## Dataset and untouched split

- Source: official Stream-51 training videos and `instance_seed10` ordering at
  upstream commit `8a066737ac8b3ac6f57987e6b3713ddcfbd1dcbf`.
- The archive identity, HTTP headers, upstream commit, ordering hash, cohort
  hash, config hash, and implementation hashes are recorded before model
  outcomes are read.
- Full archive CRC validation passed before freezing; the registered archive
  SHA-256 is `db2711e34130923147c69e203ebfde46c8d651846958d645359a0ceb4d910465`.
- The split unit is the full `(class, clip, video)` source video.  Frames are
  never independently split.
- Within every class, videos are SHA-256 ranked using the frozen salt
  `exp38-stream51-soft-memory-v1`; 40% are support, 30% development, and the
  remainder external, with at least three videos in each split.
- The resulting outcome-blind cohort contains 434 support, 321 development,
  and 381 external videos across all 51 classes.
- Development videos are divided again, within class and by a second frozen
  hash salt, into controller-fit and qualification-holdout halves.  Evidence
  calibration, normalization, time-scale selection, and controller selection
  use controller-fit videos only.  Qualification-holdout videos decide whether
  external analysis is allowed.
- External membership and labels may be present in the frozen schema, but no
  external embeddings, probabilities, accuracy, or controller statistics may
  be generated before qualification passes for every registered seed.
- The external stage loads each seed's exact temperature, stationary
  time scales, three-dimensional controller, and normalization from the
  qualification receipt.  It reads support and external features only and
  cannot rerun development selection.

This is a task-specific held-out-video protocol, not the official static
Stream-51 novelty-test leaderboard.  We will not describe its numbers as
official Stream-51 state of the art.

## Frozen representation and evidence model

The carrier is torchvision EfficientNet-B0 with `IMAGENET1K_V1` weights.  It is
never updated.  Object bounding boxes are cropped with the official 1.1
padding rule before the weight-specified deterministic preprocessing.  Frames
are sampled uniformly within a source video, up to 32 per support video and 64
per development/external video.

Only support videos fit a shared-concentration von Mises--Fisher observation
model on unit-normalized embeddings.  Class directions and concentration are
therefore independent of development and external outcomes.  A temperature
is selected by video-equal development-fit negative log likelihood.  Relative
class-conditional log likelihood, rather than fractional categorical counts,
drives predictive surprise.

## Causal tasks

`natural`: each query source video is evaluated as its own observable stream;
memory resets only at that observable video boundary.

`hidden_switch`: full, temporally ordered clips from different held-out source
videos are concatenated in groups of six.  Adjacent clips have different
classes, the boundary flag is withheld from deployable methods, and every
source video is used once per seed.  Up to 24 consecutive cached frames are
used per segment.  True switch flags are evaluation-only.

The source video is the independent real-data unit.  Seeds vary only stream
assembly and are averaged within video before inference.  Neurons, frames, and
time bins are not repetitions.

## Selection and baselines

All methods receive identical frozen embeddings, evidence, order, and stream
boundaries.  Development-fit selection includes:

- current frame (`retention=0`);
- cumulative (`retention=1`);
- the frozen retention grid;
- sliding windows of 2, 4, 8, 16, and 32 frames;
- the registered low-dimensional soft-controller grid.

External conditions are current frame, cumulative, selected fixed retention,
selected sliding window, continuous soft retention, the same risk thresholded
to a hard low/high retention, a circularly shifted retention-timing control,
and evaluation-only oracle cumulative retention (`lambda=1` within a segment,
`lambda=0` exactly at a true switch).  The shifted and oracle conditions are
causal-mechanism controls, not deployable methods.  No GRU,
BPTT, E/I network, or representation fine-tuning may be introduced after the
gate.

Controller candidates must satisfy the frozen natural-video loss constraint.
Reachability is reported for every candidate; a silent controller cannot win
by obtaining zero false alarms.  Selection and all ties use deterministic
lexicographic rules recorded in the implementation.

## Qualification gate before external access

On qualification-holdout videos, all of the following must hold jointly for
every seed:

1. the better selected stationary accumulator improves natural-video
   accuracy over current-frame by at least 0.02;
2. oracle cumulative-with-perfect-reset retention improves hidden-switch accuracy over the better
   selected fixed time scale by at least 0.02;
3. the better fixed method improves post-switch accuracy over cumulative by at
   least 0.03;
4. selected causal risk has switch AUC at least 0.65, recall at least 0.30,
   at most 30 false alarms per 1000 eligible frames, finite median delay, and
   median delay no larger than eight frames;
5. a controller candidate satisfied the development-fit reachability and
   natural-accuracy eligibility rules.

There is no `OR` success clause.  Failure of any gate writes a complete
qualification result, leaves external features locked, and yields an
`inconclusive` task/controller verdict rather than a positive or negative
external claim.

## External endpoints and verdict

The primary endpoint is video-equal hidden-switch accuracy of soft retention
minus the stronger of selected fixed retention and selected sliding window.
The registered minimum important difference is 0.02.  Natural-video accuracy
must also be non-inferior to current-frame with margin 0.01.  Secondary
endpoints are post-switch accuracy, switch latency, hard-versus-soft utility,
timing-shuffle utility, oracle headroom, retention distribution, control-state
dimension, and state/update cost.

Paired bootstrap inference resamples source videos after averaging seeds
within video.  The five listed comparisons form one Holm family.  Support
requires all of: positive primary point effect at least 0.02, lower 95%
bootstrap bound above zero, Holm-adjusted significance, and registered natural
non-inferiority.  Oppose requires a completed eligible external experiment and
a primary upper 95% bound below 0.02.  Otherwise the verdict is inconclusive.

No result based only on detector recall, low state dimension, or memory-state
rank counts as support.  Held-out task utility is mandatory.

## Stop rule

CORe50 is closed to retuning for this claim.  If Exp38 passes task
qualification but soft retention does not exceed the best fixed time scale on
the untouched Stream-51 videos, this adaptive-memory computation line stops as
a paper-method claim.  Neural, E/I, ARC, Sudoku, and HRM extensions remain out
of scope until this behavioral computation gate is supported.
