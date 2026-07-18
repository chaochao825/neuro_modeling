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
| Exp25 real compositional panel | **Inconclusive** | Canonical neural inputs are unavailable and the loader correctly fails closed |

The highest-information next step is Exp31, not a 30-seed repetition of Exp30.
It removes the explicit target decomposition and asks whether an executed-
reward-only local controller can exploit a genuine routing/memory crossover
created by hidden cue reliability and dense-memory interference.

Scaling priority is therefore:

1. independent seeds, whole blocks, association load, distractor delay, and
   feedback scarcity for Exp31;
2. only after Exp31 passes, replace the non-participating identity carrier with
   a stable E/I carrier that changes held-out utility and passes closure and
   normal-perturbation tests;
3. then add belief retention and hidden-state hazard across blocks;
4. finally test the frozen motif/controller decomposition on real
   block-switching neural and behavioral data.

Increasing carrier neuron count while the carrier does no computation has no
scientific value. Adding GRU/BPTT or broad SOTA baselines before the reward-
access shortcut is removed would test model capacity before mechanism
identifiability and is therefore lower priority.
