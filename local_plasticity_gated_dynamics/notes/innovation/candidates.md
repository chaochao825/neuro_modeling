# Innovation Candidates

| ID | Candidate | Gap vs prior work | Evidence plan | Risks / objections | Status |
|---|---|---|---|---|---|
| I1 | Label-free causal consensus over reusable few-shot motifs for personalized ORBIT recognition | Existing ORBIT methods emphasize prototype/meta-learning, sampling, or parameter adaptation; temporal adaptation work does not test this heterogeneous full-bank controller | Official CLU-VE, held-out users, validation-fixed comparator, memoryless/majority/delay interventions | Single-object video assumption; full-bank compute; only 17 test users; corrected run followed invalid partial exposure | bounded-support / scale representation next |
| I2 | A multi-executor ARC/Sudoku solver | Current repository already shows that search and rule libraries dominate these tasks, making actuator credit hard to identify | Exact-match ARC/Sudoku with solver ablations | Mechanism can become a wrapper around symbolic search; weak link to E/I dynamics | rejected-primary |
| I3 | Counterfactual physical video reasoning with motif routing | Different queries naturally demand memory, dynamics, or intervention operators | CLEVRER object-state then raw-video pipeline | Perception and language confounds; expensive baselines; not yet supported by current controller | deferred |
| I4 | A universal alternative to attention/SSM memory | Gated delta and test-time-learning models motivate adaptive state updates | Language/retrieval scaling | Far beyond repository evidence and compute; no reason to expect universal replacement | rejected |
| I5 | Neural population explanation of actuator routing | Existing IBL and compositional-task tracks can test shared bases and context switching | Multi-session shared-basis state-space analysis | Correlational neural evidence cannot validate task performance or local credit assignment | secondary-only |

## Selection Notes

I1 is the primary contribution because it has a public end-to-end input/output benchmark, standard metrics, an exact causal information boundary, strong accessible baselines, and demand regimes that can distinguish the proposed mechanisms. I3 is the next task only if I1 passes. I2 and I4 are retained as rejected historical directions rather than silently removed.
