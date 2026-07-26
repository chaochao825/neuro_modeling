# Critical evidence audit and scaling priorities

This audit contains only the active evidence surface. Matrix rank alone, a
positive constructed endpoint, or an oracle controller is never counted as
support for the full Actuator Matching Principle. Superseded and rejected work,
including the failed Exp23 gain-axis combination and exploratory Exp30 panel,
is indexed only in `results/history/README.md`.

The experiment-level diagnosis is broader than this active view. See
`docs/experiment_lineage_diagnostic_audit_20260726.md` for Exp00--Exp35 and
`docs/iclr_reviewer_response_and_revision_plan_20260726.md` for the revised
paper boundary. Exp35 supplies the decisive evidence for the current paper's
negative title claim; Exp34 is now historical motivation.

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
| Exp34 ORBIT causal consensus | **Historical positive**, superseded | Its bounded gain over weak controls motivated Exp35, but the method is defeated by stronger same-tape causal baselines |
| Exp35 prefix reliability audit | Comparative claim **oppose**; consistency-as-correctness **oppose** | Equal prefix probability beats consistency by 5.99 points; stable-wrong control locks the selector; positive prefix-accumulation decomposition is retrospective and exploratory |
| Exp36 ORBIT-India change-aware prefix | **Inconclusive**, historical schema failure | Only 4/12 collectors instantiate every frozen task; the audit was outcome-blind and did not analyze the surviving subset |
| Exp37 CORe50 change-aware prefix | **Oppose**, prospective controller stop result | Hard-reset BOCPD never alarms and trails selected forgetting by 53.29 points; post-hoc analysis shows the frozen score/threshold scales were mismatched |
| Exp38 Stream-51 soft memory | Joint qualification **oppose**; external utility **inconclusive** | Oracle headroom and switch harm pass, but stable gain passes 2/5 seeds and causal reachability 1/5; external videos remain untouched |
| Exp25 real compositional panel | **Inconclusive** | Canonical neural inputs are unavailable and the loader correctly fails closed |

Exp31's full-block reward-only advantage over the train-selected fixed actuator
is +0.0472 (95% seed-bootstrap interval [0.0459, 0.0485]), including the forced
probe cost. Exp32 then removes block resets and supports its bounded slow-switch
main endpoint: local-minus-fixed is +0.0435 (95% interval [0.0345, 0.0529]),
with 28/30 positive seeds. The stronger iso-lambda slow-minus-fast effect is
only +0.0119 and does not clear the registered 0.02 MCID, so the joint Exp32
claim remains inconclusive.

Exp34 first appeared to provide a real-task routing effect: across 17 users it
beat validation-fixed by +0.0293, temporal/reset by +0.0157, state-free
majority by +0.0253, and an eight-frame delay by +0.0066. Those comparisons
remain historically valid, but they did not isolate routing from simple
prefix aggregation.

Exp35 performed that audit on the same information boundary. Equal-weight
prefix probability accumulation reached 0.8190 user-equal accuracy versus
0.7592 for prefix consistency. The paired difference was -0.0599 (95% user
bootstrap [-0.0944, -0.0320], Holm p=0.00055). Prefix vote, a calibrated
prefix stack, and a validation-selected single prefix operator also beat the
router. Its difference from fixed temporal was inconclusive. In the exact
stable-wrong control, the selector had accuracy zero and wrong-lock fraction
one. Both the comparative and reliability interpretations are therefore
opposed.

The surviving positive result is explicitly exploratory: accumulating class
probabilities across a video prefix improves each individual operator by
5.10--13.93 points, whereas the bank's increment over a selected single prefix
operator is +0.57 points with an interval crossing zero. The current evidence
supports temporal evidence accumulation under a constant-object-video
assumption, not heterogeneous actuator matching.

Exp37 then tests the accumulation extension prospectively on a newly acquired,
session-held CORe50 panel. All 40,500 registered cells complete, but hard-reset
BOCPD is identical to cumulative accumulation (0.3984 hidden-switch accuracy)
and is decisively below selected retention zero (0.9313; difference -0.5329,
95% session bootstrap [-0.5432, -0.5220], Holm p=0.015625) and a selected
two-frame window (0.9221; difference -0.5238, [-0.5334, -0.5141], Holm
p=0.015625). Oracle reset reaches 0.9528, so the task contains usable switch
information. A post-hoc development-only audit finds maximum change posterior
0.008529 versus the frozen minimum threshold 0.2. This is both a valid negative
result for the registered controller and a method-scale defect that prevents a
general rejection of change-point inference.

Exp38 then tests the narrower continuous-retention successor on a newly frozen
Stream-51 source-video split. The support/development cache audit passes for
755 videos and 34,250 frames without generating an external feature. The task
does require adaptive memory: oracle headroom is 0.0273--0.0558 and cumulative
post-switch harm is 0.7115--0.7356, both passing in 5/5 seeds. However, stable
accumulation passes its 0.02 MCID in only 2/5 seeds and the joint causal
reachability gate in only 1/5. Soft retention's descriptive qualification
advantage over the stronger fixed baseline is +0.0019 on average, well below
the external MCID. The all-seed gate therefore fails 0/5. This opposes the
registered readiness claim, leaves the external utility claim inconclusive,
and prevents any access to the 381 external videos.

Priority is therefore:

1. write the complete negative mechanism audit and retain Exp34 as motivating
   history rather than positive title evidence;
2. stop HMM/Hedge/GRU, sparse-routing, and E/I extensions intended to rescue
   prefix consistency;
3. honor both prospective stop rules: do not retune categorical BOCPD on
   CORe50 or the three-statistic soft controller on Stream-51, and do not open
   Exp38 external data after failed qualification;
4. retain simple forgetting/current-frame evidence as the real-task reference,
   rather than treating explicit change detection as automatically superior;
5. treat participating E/I and multi-session neural validation as separate
   paper contracts.

Increasing carrier neuron count while the carrier does no computation has no
scientific value. Likewise, quoting independently trained leaderboard numbers
cannot replace same-tape paired baselines; representation scaling and
mechanism scaling must remain separately identifiable.
