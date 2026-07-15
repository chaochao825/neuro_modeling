# Exp20 real IBL belief-gated dynamics

- Data: hash-bound frozen compact IBL neural/behavior cohorts.
- Inference: animal with session nested; never neuron or time bin.
- Model: teacher-forced one-step conditional Poisson shared-basis dynamics, not a full LDS.
- Mechanism boundary: these recordings do not establish E/I, Dale, or recurrent plasticity.
- Truth capability: probabilityLeft is restricted to whole-block splitting and post-fit evaluation.
- Raw-run commit: `4a6ae28e68574d1caedbaf4694fbc67bcd51d6c7`; analysis commit: `fff4640269e779770ea38e9b9de3667a64af2d9f`.

| proposition | comparison | estimate | ci_low | ci_high | conclusion | claim_scope |
| --- | --- | --- | --- | --- | --- | --- |
| belief_condition_neural_prediction | md_shared_vs_common | -4.175279188634029e-06 | -0.00016035446166155128 | 0.000124871266864309 | inconclusive | teacher-forced one-step conditional Poisson prediction |
| belief_condition_neural_prediction | md_shared_vs_hmm_shared | -3.600151883663138e-05 | -8.519520754405597e-05 | 7.956944796679004e-06 | inconclusive | teacher-forced one-step conditional Poisson prediction |
| belief_condition_neural_prediction | md_shared_vs_md_clamp | -7.763186133463718e-06 | -0.0001578228324080333 | 0.0001244804301584785 | inconclusive | teacher-forced one-step conditional Poisson prediction |
| belief_condition_neural_prediction | md_shared_vs_md_delay_1 | 0.00010430375897710342 | -3.1208260210631443e-06 | 0.00031422545811857965 | inconclusive | teacher-forced one-step conditional Poisson prediction |
| belief_condition_neural_prediction | md_shared_vs_md_delay_5 | 0.00026609667213824963 | -1.2845847231167384e-05 | 0.0007967004443565726 | inconclusive | teacher-forced one-step conditional Poisson prediction |
| belief_condition_neural_prediction | md_shared_vs_md_shuffle | 0.00020342701051165868 | -6.97016003904885e-06 | 0.0006112595515742966 | inconclusive | teacher-forced one-step conditional Poisson prediction |
| shared_basis_joint_registered_claim | md_shared_vs_common_and_full | -4.175279188634029e-06 | -0.00016904106635822494 | 0.00012309128649215284 | inconclusive | joint shared-vs-common, full gain, retention, and parameter-count gate |
| belief_vs_behavior_bias_switch_timing_descriptive | md_belief_minus_causal_choice_history_bias_latency | -3.9708333333333337 | -5.8125 | -2.0458333333333334 | inconclusive | descriptive causal EWMA choice-bias proxy; not a neural-latent lead claim |
| past_only_truth_capability_contract | registered_threshold_audit | 1.0 |  |  | support | data/timing capability audit only |
