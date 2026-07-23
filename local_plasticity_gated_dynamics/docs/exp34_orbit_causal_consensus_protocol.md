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
selection-fixed, causal consensus, a count-reset-every-frame intervention, an
eight-frame delayed update, and a label oracle. The formal configuration is
fail-closed; after authorization it selects the fixed comparator on all six
validation users and evaluates 50 tasks for each of 17 untouched test users.

Inference first averages frames within task/video, repeated algorithmic seeds
within user, and finally uses user as the independent unit. Support requires
consensus to beat both the validation-selected fixed actuator and the
memoryless intervention under Holm correction. A validation result is always
`inconclusive`, but may authorize scale if its effect and retained oracle
headroom clear their registered thresholds.

## Frozen development result and scale decision

The three-seed development panel completed on the two held-out validation
users. Causal consensus reached 0.7388 mean task-video accuracy versus 0.6419
for both the validation-selected fixed temporal actuator and the memoryless
intervention: a +0.0968 difference (user bootstrap interval +0.0527 to
+0.1409). Delaying belief updates by eight frames reduced the gain to +0.0421,
and the gate retained 91.3% of the available per-frame oracle headroom. Both
users improved in all three registered comparisons.

This panel is **inconclusive** because it has only two development users and
its exact sign-flip p-values are 0.5. It nevertheless passed the predeclared
scale gate. The hash-bound authorization receipt is saved with the development
artifacts, so formal evaluation can run once on all 17 untouched test users.
