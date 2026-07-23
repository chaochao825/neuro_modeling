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
| Exp25 real compositional panel | **Inconclusive** | Canonical neural inputs are unavailable and the loader correctly fails closed |

Exp31's full-block reward-only advantage over the train-selected fixed actuator
is +0.0472 (95% seed-bootstrap interval [0.0459, 0.0485]), including the forced
probe cost. Exp32 then removes block resets and supports its bounded slow-switch
main endpoint: local-minus-fixed is +0.0435 (95% interval [0.0345, 0.0529]),
with 28/30 positive seeds. The stronger iso-lambda slow-minus-fast effect is
only +0.0119 and does not clear the registered 0.02 MCID, so the joint Exp32
claim remains inconclusive.

Scaling priority is therefore:

1. place the frozen motifs and persistent reward-only controller inside a
   stable participating E/I carrier that changes held-out utility and passes
   closure and normal-perturbation tests;
2. calibrate controller belief and match intervention update budgets while
   retaining executed-reward-only access;
3. test the frozen motif/controller decomposition on real multi-session
   block-switching neural and behavioral data.

Increasing carrier neuron count while the carrier does no computation has no
scientific value. Adding GRU/BPTT or broad SOTA baselines before carrier
participation and controller calibration are identified would test model
capacity before mechanism identifiability and is therefore lower priority.
