# Critical evidence audit and scaling priorities

This audit separates verified behavior from mechanism interpretation. Matrix
rank alone, a positive constructed endpoint, or an oracle controller is never
counted as support for the full Actuator Matching Principle.

| Evidence | Current conclusion | Main limitation |
|---|---|---|
| Exp23 closed-loop local gain controller | **Oppose** the current rule/axis/budget combination | Gradient cosine was positive but held-out behavior did not improve; this does not reject every local rule |
| Exp24 factorized endpoints | **Support**, narrow synthetic capability result | Tasks and actuator axes are hand aligned; true context is available |
| Exp26 demand geometry | **Support**, synthetic actuator-family geometry | Actuator parameters are refit from each task's target trajectory |
| Exp29 selector | **Support**, descriptor-driven meta-selection | Inputs are privileged generator descriptors and training uses a full candidate-utility teacher, not scalar bandit feedback |
| Exp30 associative trend | **Inconclusive**, positive-control trend | Target is an explicit actuator mixture, retrieval is exact, demand is visible, gains are fitted per cell, carrier is an identity bridge |
| Exp31 hidden-reliability selector | **Support**, narrow reward-only controller result | Fixed synthetic motifs; test-time scalar feedback is available; no participating high-rank carrier or real neural data |
| Exp25 real compositional panel | **Inconclusive** | Canonical neural inputs are unavailable and the loader correctly fails closed |

Exp31 has now passed its frozen five-seed scale gate and one-shot 30-seed
formal panel. The full-block reward-only advantage over the train-selected
fixed actuator is +0.0472 (95% seed-bootstrap interval [0.0459, 0.0485]), even
after charging the forced-probe cost. In this constructed setting, this removes
the privileged-descriptor access used by Exp29, but it does not establish
descriptor-free selection beyond a controlled synthetic two-actuator task.
The selector receives feedback labels on half of each block, resets between
blocks, and has no more than two fixed arms.

Scaling priority is therefore:

1. replace the non-participating identity carrier with
   a stable E/I carrier that changes held-out utility and passes closure and
   normal-perturbation tests;
2. add belief retention, feedback scarcity, and hidden-state hazard across
   blocks while retaining executed-reward-only access;
3. test the frozen motif/controller decomposition on real
   block-switching neural and behavioral data.

Increasing carrier neuron count while the carrier does no computation has no
scientific value. Adding GRU/BPTT or broad SOTA baselines before the reward-
access shortcut is removed would test model capacity before mechanism
identifiability and is therefore lower priority.
