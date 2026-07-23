# Exp34: causal consensus belief on ORBIT

Exp33 established substantial real-video actuator disagreement and oracle
headroom, but its reward-only controller did not generalize from four fitting
users to two held-out validation users. Exp34 retains that failed result and
tests a narrower failure-driven hypothesis: when each query video contains one
personalized object, can a label-free belief over the temporal persistence of
candidate predictions choose the useful actuator online?

At frame `t`, every actuator emits a class prediction from the current and past
frames only. For actuator `a`, the gate updates an action-by-class count state
and defines belief as the largest class count divided by all counts for that
action. It emits the current prediction of the actuator with greatest belief.
Ties use the frozen order temporal, gain, prototype, delta. Counts reset at
video boundaries. Query labels, future frames, object-presence annotations,
autograd, and BPTT are unavailable.

The full actuator bank is evaluated on every frame, so Exp34 reports its full
bank arithmetic/event proxy rather than pretending to achieve sparse-execution
savings. The main claim concerns held-out accuracy and causal state, not energy
efficiency.

Development selects one fixed-actuator comparator on four validation users and
evaluates two disjoint validation users. It compares four fixed motifs,
selection-fixed, a state-free instantaneous majority ensemble, causal
consensus, a count-reset-every-frame intervention, an eight-frame delayed
update, and a label oracle. The formal configuration is
fail-closed; after authorization it selects the fixed comparator on all six
validation users and evaluates 50 tasks for each of 17 untouched test users.

Inference first averages frames within task/video, repeated algorithmic seeds
within user, and finally uses user as the independent unit. Support requires
consensus to beat the validation-selected fixed actuator, state-free
instantaneous majority, memoryless reset, and delayed-state interventions
under one Holm family. A validation result is always `inconclusive`, but may
authorize scale if its effect and retained oracle headroom clear their
registered thresholds.

## Invalidated v1 development result

The three-seed development panel completed on the two held-out validation
users. Causal consensus reached 0.7388 mean task-video accuracy versus 0.6419
for both the validation-selected fixed temporal actuator and the memoryless
intervention: a +0.0968 difference (user bootstrap interval +0.0527 to
+0.1409). Delaying belief updates by eight frames reduced the gain to +0.0421,
and the gate retained 91.3% of the available per-frame oracle headroom. Both
users improved in all three registered comparisons.

This panel is **invalid** for protocol authorization. The v1 feature pipeline
used the `object_not_present_issue` annotation to filter clean support frames,
whereas ORBIT permits that annotation only for clutter-query sampling and
forbids extra clean-video annotations during personalization. The artifacts and
receipt are retained, but the formal launch was stopped before any test user
was evaluated.

The annotation-safe feature rule makes all clean frames support-eligible, uses
object-presence filtering only for clutter query, and stores validation and
test features in separate cache roots.

## Annotation-safe development result and formal authorization

The corrected three-seed panel reached 0.7374 user-equal accuracy on the same
two held-out development users. It exceeded the validation-selected temporal
actuator by 0.0954, the memoryless reset by 0.0954, the state-free majority
ensemble by 0.0887, and the eight-frame delayed gate by 0.0434. Both users
improved in all four comparisons, and the gate retained 88.6% of oracle
headroom. Because only two development users are involved, this result remains
`inconclusive`; it authorizes but does not substitute for the frozen test run.

The authorization receipt binds the exact implementation plus the separate
six-user validation and 17-user test feature manifests. No test-user result
was inspected before this receipt was frozen.

## Invalid formal-v2 coverage attempt

The first authorized formal attempt used the annotation-safe feature stores,
but the episode loader treated an official video-level exclusion as a fatal
user-level error. ORBIT requires clutter videos with fewer than 50 valid query
frames to be excluded. Instead, the run dropped all tasks for P204 and P901,
leaving only 15/17 users. An early aggregator did not enforce the registered
coverage set and returned a nominally positive result. That result is
**invalid** and contributes no inferential evidence. The incomplete raw run,
hashes, and failure explanation remain in
[`results/exp34_orbit_causal_consensus_formal_v2_incomplete_invalid`](../results/exp34_orbit_causal_consensus_formal_v2_incomplete_invalid/INVALID_COVERAGE.md).

The correction changed only protocol compliance: skip and record a clutter
video below the official 50-valid-frame minimum, then fail unless every seed,
user, task, and remaining video is complete. It also added an explicit
coverage regression test. The active protocol is
`exp34_orbit_causal_consensus_v3_official_video_exclusion`.

## Corrected formal-v3 result

The corrected run completed 5/5 seeds, 17/17 test users, and all 4,250 planned
seed-user-task episodes with no failed or invalid condition. Exactly three
short clutter videos were excluded and recorded: two for P204 and one for
P901. Algorithmic seeds were averaged within user before inference.

Causal consensus reached 0.7189 user-equal accuracy. It exceeded the
validation-selected fixed gain actuator by +0.0293 (95% user-bootstrap CI
[+0.0155, +0.0437]; Holm p=0.00189), the memoryless reset by +0.0157
([+0.0039, +0.0279]; Holm p=0.02545), instantaneous majority by +0.0253
([+0.0150, +0.0362]; Holm p=0.000183), and the eight-frame delayed gate by
+0.0066 ([+0.0043, +0.0088]; Holm p=0.000275). The four effects were positive
for 15/17, 11/17, 16/17, and 15/17 users, respectively. The gate retained
53.6% of the available oracle headroom and had 0.3006 mean actuator
disagreement. Thus the corrected, registered task-and-causal-state gate is
**support**.

The official-style task-video point estimate was 67.43%. It is effectively
tied with the official EfficientNet-B0 cosine ProtoNet reference (67.48%) and
well below the reported ViT-B/32 reference (75.38%). These independently
published numbers use different representation-training budgets, so they are
descriptive context rather than paired non-inferiority or SOTA evidence. See
the [benchmark context](../results/exp34_orbit_causal_consensus_formal_v3_official_video_exclusion/benchmark_context.md),
[formal report](../results/exp34_orbit_causal_consensus_formal_v3_official_video_exclusion/report.md),
and [publication figure](../figures/exp34_orbit_causal_consensus_formal_v3_official_video_exclusion.pdf).

There is one unavoidable evidence boundary. Fifteen test users were exposed
by the invalid formal-v2 attempt before the coverage defect was found. The v3
correction was protocol-driven and did not tune the gate, but it reused the
same public test split. Therefore the corrected within-dataset mechanism
contrast supports C0/C1, while a strictly untouched prospective confirmation
is **inconclusive** and requires a new dataset or independently frozen
replication. No compute-efficiency, E/I-carrier, neural-data, or general-SOTA
claim follows from Exp34.
