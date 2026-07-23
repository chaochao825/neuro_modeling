# Critical evidence audit and scaling priorities

This audit contains only the active evidence surface. Matrix rank alone, a
positive constructed endpoint, or an oracle controller is never counted as
support for the full Actuator Matching Principle. Superseded and rejected work,
including the failed Exp23 gain-axis combination and exploratory Exp30 panel,
is indexed only in `results/history/README.md`.

| Evidence | Current conclusion | Main limitation |
|---|---|---|
| Exp08 credit/rank-stage audit | **Support**, revised rank interpretation | Low-dimensional credit does not imply a low-rank physical update after masking Dale constraints and normalization |
| Exp09 hidden-HMM belief gate | **Support**, leakage-safe synthetic inference | Does not identify a biological MD implementation |
| Exp21 belief-controlled E/I trajectories | **Support**, bounded frozen-receiver dynamics | Registered d=4 and conditioned rollouts are mechanism audits rather than nested-CV probabilistic LDS evidence |
| Exp24 factorized endpoints | **Support**, narrow synthetic capability result | Tasks and actuator axes are hand aligned; true context is available |
| Exp26 demand geometry | **Support**, synthetic actuator-family geometry | Actuator parameters are refit from each task's target trajectory |
| Exp29 selector | **Support**, descriptor-driven meta-selection | Inputs are privileged generator descriptors and training uses a full candidate-utility teacher, not scalar bandit feedback |
| Exp31 hidden-reliability selector | **Support**, narrow reward-only controller result | Fixed synthetic motifs; test-time scalar feedback is available; no participating high-rank carrier or real neural data |
| Exp32 persistent sparse-feedback selector | Main endpoint **support**; registered joint claim **inconclusive** | Slow-switch primary supports but the iso-lambda timescale effect misses its MCID; no participating E/I carrier |
| Exp34 ORBIT causal consensus | Corrected task/causal-state claim **support**; overall evidence **mixed** | 17/17-user effects survive fixed, memoryless, majority, and delay controls, but full-bank compute has no efficiency gain and the corrected run followed an invalid 15-user test exposure |
| Exp25 real compositional panel | **Inconclusive** | Canonical neural inputs are unavailable and the loader correctly fails closed |

Exp31's full-block reward-only advantage over the train-selected fixed actuator
is +0.0472 (95% seed-bootstrap interval [0.0459, 0.0485]), including the forced
probe cost. Exp32 then removes block resets and supports its bounded slow-switch
main endpoint: local-minus-fixed is +0.0435 (95% interval [0.0345, 0.0529]),
with 28/30 positive seeds. The stronger iso-lambda slow-minus-fast effect is
only +0.0119 and does not clear the registered 0.02 MCID, so the joint Exp32
claim remains inconclusive.

Exp34 provides the first complete real-task effect. Across 17 ORBIT test users,
causal consensus exceeded validation-fixed by +0.0293 (95% user-bootstrap CI
[+0.0155, +0.0437]), memoryless by +0.0157, state-free majority by +0.0253,
and an eight-frame delay by +0.0066; all four survived Holm correction. The
official-style task-video accuracy was 67.43%, essentially the published
EfficientNet-B0 cosine ProtoNet reference of 67.48%, rather than a new SOTA.
The result is classified mixed globally: the corrected mechanism contrast
supports, while strict untouched-first-look confirmation is lost because a
coverage bug exposed 15 test users before the correction.

Scaling priority is therefore:

1. rerun the paired actuator audit with a stronger official backbone and one
   shared task tape, including trained same-budget ProtoNet/CNAP-style
   comparators, to separate controller value from representation quality;
2. replace full-bank execution with a registered sparse/early-exit controller
   and require matched accuracy at lower measured MAC/event cost;
3. realize the surviving motifs inside a stable participating E/I carrier and
   require gain retention, low closure error, normal contraction, and matched
   control budget;
4. test the decomposition on a new prospectively frozen real task and on
   eligible multi-session neural data at animal/session inference level.

Increasing carrier neuron count while the carrier does no computation has no
scientific value. Likewise, quoting independently trained leaderboard numbers
cannot replace same-tape paired baselines; representation scaling and
mechanism scaling must remain separately identifiable.
