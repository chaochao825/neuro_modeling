# Local Plasticity to Gated Low-Dimensional Dynamics

This report combines the compact immutable core-run snapshot with separately hash-bound scoped publications. Failed and invalid conditions are retained in their owning track; only formal-profile independent units can support or oppose a registered claim.

## Compact core run coverage

This table covers the compact Exp00--11 run snapshot. Retries and interrupted attempts remain listed here. Exp13--16 use separately validated raw tables and run manifests in their scoped sections below. These are attempt counts, not unique-seed coverage; claim sample sizes use only the eligible formal attempt for each experiment and independent unit.

| Experiment | Profile | Attempts | Clean complete | Complete with failures | Failed/partial | Planned attempt-cells |
|---|---:|---:|---:|---:|---:|---:|
| exp00_fixed_point | formal | 20 | 20 | 0 | 0 | 20 |
| exp00_fixed_point | smoke | 1 | 1 | 0 | 0 | 1 |
| exp01_feedback_dimension_sweep | formal | 20 | 0 | 20 | 0 | 960 |
| exp01_feedback_dimension_sweep | smoke | 1 | 1 | 0 | 0 | 12 |
| exp02_context_ei_oracle_gate | formal | 32 | 20 | 0 | 12 | 608 |
| exp02_context_ei_oracle_gate | smoke | 1 | 1 | 0 | 0 | 15 |
| exp03_context_ei_learned_gate | formal | 29 | 20 | 0 | 9 | 551 |
| exp03_context_ei_learned_gate | smoke | 1 | 1 | 0 | 0 | 15 |
| exp04_phase_gating | formal | 20 | 20 | 0 | 0 | 80 |
| exp05_sequence_real_data | formal | 2 | 0 | 2 | 0 | 2 |
| exp06_ibl_context_switch | formal | 8 | 4 | 2 | 2 | 12 |
| exp07_mechanism_identifiability | formal | 30 | 30 | 0 | 0 | 1020 |
| exp08_rank_stage_validation | formal | 30 | 30 | 0 | 0 | 4410 |
| exp09_hidden_context_gate | formal | 30 | 30 | 0 | 0 | 3840 |
| exp10_hidden_context_ei_bridge | formal | 120 | 90 | 0 | 30 | 3360 |
| exp10_hidden_context_ei_bridge | smoke | 60 | 60 | 0 | 0 | 420 |
| exp11_ibl_behavior_belief | formal | 1 | 1 | 0 | 0 | 120 |

## Registered core and scoped proposition audit

| Claim | Criterion | n complete/planned | Estimate [95% CI] | Conclusion |
|---|---|---:|---:|---|
| A1_rank_matches_feedback | 95% CI of effective-rank minus 4 lies within [-0.5, 0.5] | 20/20 | -1.812e-05 [-2.059e-05, -1.591e-05] | **support** |
| A2_d4_r2_noninferior_full | one-sided non-inferiority margin is -0.01 latent R2 | 20/20 | 5.038e-06 [-3.905e-06, 1.633e-05] | **support** |
| A3_alignment_is_necessary | aligned latent-R2 advantage CI lower bound is at least 0.10 | 20/20 | 0.9727 [0.9718, 0.9736] | **support** |
| B1a_local_absolute_accuracy | local absolute accuracy-minus-0.85 CI is reported independently | 20/20 | -0.1747 [-0.2303, -0.1194] | **oppose** |
| B1b_local_relative_noninferiority | local relative non-inferiority to 90% of paired BPTT is reported independently of absolute performance | 20/20 | 0.1263 [0.05168, 0.1959] | **support** |
| B2_gate_reduces_switch_cost | hidden-context gate improves switch cost without true-context access in the cue, gate fit/test, oracle warm start, or recurrent third factor | 0/20 | — [—, —] | **inconclusive** |
| P2a_hmm_context_nll | learned HMM improves context NLL by at least 0.02 nats/trial | 30/30 | 0.3275 [0.3189, 0.3357] | **support** |
| P2b_md_context_nll | MD belief improves context NLL by at least 0.02 nats/trial | 30/30 | 0.335 [0.3325, 0.3376] | **support** |
| P2c_md_context_brier | MD belief improves Brier score by at least 0.01 | 30/30 | 0.1293 [0.1284, 0.1303] | **support** |
| P2d_md_calibration | MD ECE upper CI <=0.05; ECE lower CI >=0.10 opposes | 30/30 | -0.01761 [-0.01906, -0.01605] | **support** |
| P2e_md_switch_latency | MD excess switch latency upper CI <=1 trial | 30/30 | -0.2495 [-0.2713, -0.2283] | **support** |
| P2f_md_false_switch | MD excess false-switch-rate upper CI <=0.01 | 30/30 | 0.0007485 [0.000535, 0.001001] | **support** |
| P2g_md_behavior | MD gate improves held-out balanced accuracy by at least 0.02 | 30/30 | 0.1343 [0.1332, 0.1354] | **support** |
| P2h_md_retains_oracle_gain | MD retains at least 90% of the paired oracle behavioral gain | 30/30 | 0.008053 [0.007645, 0.008453] | **support** |
| P2i_md_energy | MD energy upper ratio CI <=1.10 | 30/30 | 0.67 [0.6657, 0.675] | **oppose** |
| P2j_clamp_causal | post-fit clamp reduces balanced accuracy by at least 0.01 | 30/30 | 0.1343 [0.1332, 0.1354] | **support** |
| P2k_delay_causal | post-fit one-trial delay at h=0.10/0.20 reduces balanced accuracy by at least 0.01 | 30/30 | 0.03645 [0.0354, 0.03748] | **support** |
| P2l_shuffle_causal | post-fit trajectory shuffle reduces balanced accuracy by at least 0.01 | 30/30 | 0.1368 [0.1343, 0.1392] | **support** |
| B3_homeostasis_stabilizes | removing inhibitory homeostasis increases Jacobian instability | 20/20 | -0.03029 [-0.03384, -0.02665] | **oppose** |
| B4_local_rank_below_full_feedback | local three-factor update rank is lower than full-feedback rank | 20/20 | -0.1266 [-0.5359, 0.3157] | **inconclusive** |
| C1_phase_effect_survives_rate_match | exactly matched in-phase accuracy advantage CI exceeds 0.02 | 20/20 | 0 [0, 0] | **oppose** |
| D1_shared_basis_near_full | retained switching gain >= 0.90 and shared parameters < full | 0/2 | — [—, —] | **inconclusive** |
| D2_unseen_sequence_generalization | shared held-out NLL is below full LDS on unseen combinations | 0/2 | — [—, —] | **inconclusive** |
| E1_ibl_shared_switching | stimulus-pre hierarchical shared model improves on common, retains >=0.90 of full gain, and uses fewer counted parameters | 0/1 | — [—, —] | **inconclusive** |
| E2_latent_precedes_behavior_bias | independent-unit bootstrap CI of latent lead is above zero | 1/1 | — [—, —] | **inconclusive** |
| P0a_aligned_task_improves_prediction_vs_frozen | aligned task plasticity lowers held-out prediction MSE versus a bitwise-frozen recurrent network in both L1/L2 panels | 30/30 | 0.002431 [0.001238, 0.006549] | **support** |
| P0b_aligned_task_beats_shuffled | aligned feedback lowers held-out prediction MSE versus shuffled feedback under separately exact L1 and L2 task budgets | 30/30 | 0.001212 [0.0006801, 0.002995] | **support** |
| P0c_aligned_adds_value_over_matched_homeostasis | adding aligned task plasticity improves held-out prediction over the same-budget homeostasis-only control in both panels | 30/30 | 0.002435 [0.001249, 0.006574] | **support** |
| P0d_local_absolute_accuracy | absolute accuracy >=0.85 independently in both L1/L2 panels | 30/30 | 0.05667 [0.02917, 0.08167] | **support** |
| P0e_local_noninferior_tuned_bptt | relative non-inferiority to 90% of tuned BPTT independently in both L1/L2 panels and independently of absolute accuracy | 30/30 | 0.07492 [0.03458, 0.1174] | **support** |
| P0f_local_noninferior_tuned_gru | relative non-inferiority to 90% of tuned GRU independently in both L1/L2 panels and independently of absolute accuracy | 30/30 | 0.05842 [0.02192, 0.09917] | **support** |
| P1a_masked_outer_product_identity | M⊙uv^T equals diag(u)Mdiag(v) to <=1e-12 max residual | 30/30 | 0 [0, 0] | **support** |
| P1b_credit_tangent_respects_feedback_bound | instantaneous credit tangent does not exceed feedback dimension within 0.5 numerical-dimension tolerance | 30/30 | 0 [0, 0] | **support** |
| P1c_highrank_physical_update_coexists_with_lowdim_credit | masked physical numerical rank exceeds credit tangent dimension; this theoretical claim does not imply held-out task support | 30/30 | 124 [124, 124] | **support** |
| P0_overall | support iff every Holm-adjusted P0a--P0f claim supports; oppose iff at least one opposes; otherwise inconclusive | 30/30 | — [—, —] | **support** |
| P2_overall | support iff every critical Holm-adjusted P2 claim supports; oppose iff at least one opposes; otherwise inconclusive | 30/30 | — [—, —] | **support** |
| S1_exp10_hmm_context_inference | Holm p<0.05 and seed-macro bootstrap CI excludes zero after equal averaging across four q/h cells | 30/30 | 0.2777 [0.2696, 0.286] | **support** |
| S2_exp10_md_context_inference | Holm p<0.05 and seed-macro bootstrap CI excludes zero after equal averaging across four q/h cells | 30/30 | 0.2583 [0.2455, 0.2701] | **support** |
| S3_exp10_hmm_functional_pipeline | Holm p<0.05 and seed-macro bootstrap CI excludes zero after equal averaging across four q/h cells | 30/30 | 0.09999 [0.09621, 0.1038] | **support** |
| S4_exp10_md_functional_pipeline | Holm p<0.05 and seed-macro bootstrap CI excludes zero after equal averaging across four q/h cells | 30/30 | 0.09639 [0.092, 0.1009] | **support** |
| S5_exp10_md_retains_oracle_gain | Holm p<0.05 and seed-macro bootstrap CI excludes zero after equal averaging across four q/h cells | 30/30 | 0.005702 [0.003435, 0.007813] | **support** |
| S6_exp10_md_clamp_counterfactual | Holm p<0.05 and seed-macro bootstrap CI excludes zero after equal averaging across four q/h cells | 30/30 | 0.0899 [0.08662, 0.09337] | **support** |
| S7_exp10_md_delay_counterfactual | Holm p<0.05 and seed-macro bootstrap CI excludes zero after equal averaging across four q/h cells | 30/30 | 0.02226 [0.01986, 0.02472] | **support** |
| S8_exp10_md_shuffle_counterfactual | Holm p<0.05 and seed-macro bootstrap CI excludes zero after equal averaging across four q/h cells | 30/30 | 0.09923 [0.09425, 0.1045] | **support** |
| R1_ibl_hmm_context_inference | complete planned cohort plus Holm p<0.05 and animal-primary hierarchical CI excluding zero | 30/30 | 0.3768 [0.3313, 0.4178] | **support** |
| R2_ibl_history_context_inference | complete planned cohort plus Holm p<0.05 and animal-primary hierarchical CI excluding zero | 30/30 | -0.5649 [-0.7772, -0.3784] | **oppose** |
| R3_ibl_hmm_behavior_prediction | complete planned cohort plus Holm p<0.05 and animal-primary hierarchical CI excluding zero | 30/30 | -0.001087 [-0.003275, 0.0007699] | **inconclusive** |
| R4_ibl_history_behavior_prediction | complete planned cohort plus Holm p<0.05 and animal-primary hierarchical CI excluding zero | 30/30 | -0.003753 [-0.00724, -0.000773] | **inconclusive** |
| T1_arc_hierarchical_vs_flat | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 399/399 | 1.739e-20 [-0.0002506, 0.0002506] | **inconclusive** |
| T2_arc_trace_vs_flat | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 399/399 | 0.0004177 [0, 0.001086] | **inconclusive** |
| T3_arc_hierarchical_vs_heuristic | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 399/399 | -0.002005 [-0.007018, 0.001003] | **inconclusive** |
| T4_arc_hierarchical_vs_gru | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 399/399 | -0.002005 [-0.007018, 0.001003] | **inconclusive** |
| T5_arc_hierarchical_90pct_gru | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero for the candidate - 0.9*reference non-inferiority margin | 399/399 | -0.001504 [-0.006266, 0.001253] | **inconclusive** |
| T6_arc_trace_increment | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 399/399 | 0.0004177 [0, 0.001086] | **inconclusive** |
| M1_maze_hierarchical_vs_flat | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 18/18 | -0.003704 [-0.02037, 0.007407] | **inconclusive** |
| M2_maze_trace_vs_flat | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 18/18 | -0.005556 [-0.02222, 0.005556] | **inconclusive** |
| M3_maze_hierarchical_vs_heuristic | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 18/18 | 0.1019 [0, 0.2574] | **inconclusive** |
| M4_maze_hierarchical_vs_gru | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 18/18 | -0.009259 [-0.02593, 0] | **inconclusive** |
| M5_maze_hierarchical_90pct_gru | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero for the candidate - 0.9*reference non-inferiority margin | 18/18 | 0.09074 [0.07407, 0.1] | **support** |
| M6_maze_trace_increment | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 18/18 | -0.001852 [-0.005556, 0] | **inconclusive** |
| N1_sudoku_hierarchical_vs_flat | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 28/28 | 0 [0, 0] | **inconclusive** |
| N2_sudoku_trace_vs_flat | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 28/28 | 0 [0, 0] | **inconclusive** |
| N3_sudoku_hierarchical_vs_heuristic | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 28/28 | 0 [0, 0] | **inconclusive** |
| N4_sudoku_hierarchical_vs_gru | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 28/28 | 0 [0, 0] | **inconclusive** |
| N5_sudoku_hierarchical_90pct_gru | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero for the candidate - 0.9*reference non-inferiority margin | 28/28 | 0.1 [0.1, 0.1] | **inconclusive** |
| N6_sudoku_trace_increment | registered OOD split and candidate coverage gate plus Holm p<0.05 and task-component bootstrap CI excluding zero | 28/28 | 0 [0, 0] | **inconclusive** |
| U1_ibl_shared_neural_dynamics | registered stimulus-pre/past-safe panel; >=20 complete sessions and >=5 animals; Holm-significant shared>common; shared retains >=90% of significant full gain; fewer parameters | 20/20 | -0.0009953 [-0.003061, 0.0001569] | **inconclusive** |
| V1_exp15_arc_slow_fast_vs_flat | verified OOD source, identical candidates, matched charged compute, >=90% candidate coverage, paired CI and Holm p<0.05 | 399/399 | 0 [0, 0] | **inconclusive** |

### Evidence details

- `A1_rank_matches_feedback` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; two-one-sided margin tests awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.53674316406e-07); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `A2_d4_r2_noninferior_full` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.53674316406e-07); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `A3_alignment_is_necessary` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.53674316406e-07); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `B1a_local_absolute_accuracy` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided oppose-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=6.99728965543e-05); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `B1b_local_relative_noninferiority` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=0.0024299621582); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `B2_gate_reduces_switch_cost` (failed=0): legacy exp02/03 lacks leakage-free hidden-context provenance; exp03 is a supervised/oracle-warm-start upper bound and the legacy no-gate third factor receives true context
- `P2a_hmm_context_nll` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.31322574615e-10); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P2b_md_context_nll` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.31322574615e-10); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P2c_md_context_brier` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.31322574615e-10); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P2d_md_calibration` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.31322574615e-10); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P2e_md_switch_latency` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.31322574615e-10); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P2f_md_false_switch` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=8.666533221e-07); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P2g_md_behavior` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.31322574615e-10); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P2h_md_retains_oracle_gain` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.31322574615e-10); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P2i_md_energy` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided oppose-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.31322574615e-10); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P2j_clamp_causal` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.31322574615e-10); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P2k_delay_causal` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.31322574615e-10); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P2l_shuffle_causal` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.31322574615e-10); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `B3_homeostasis_stabilizes` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided oppose-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=9.53674316406e-07); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `B4_local_rank_below_full_feedback` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; two-sided zero-difference diagnostic awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=0.521673202515); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `C1_phase_effect_survives_rate_match` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided oppose-margin test awaits full-family Holm adjustment; all exact matching flags true and source fingerprints identical within seed; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=3.87210821552e-06); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `D1_shared_basis_near_full` (failed=1): complete common/shared/full formal panel unavailable
- `D2_unseen_sequence_generalization` (failed=1): complete unseen-combination shared/full panel unavailable
- `E1_ibl_shared_switching` (failed=1): strict eligible cohort has 0 animals/0 sessions (minimum 5/20); missing provenance field hierarchical_observation_model; missing provenance field nested_cv_latent_dimension; missing provenance field unit_qc_applied; missing provenance field context_coverage_valid; missing provenance field parameter_count_includes_preprocessing; missing provenance field hidden_context_inference; missing provenance field test_context_observed; missing provenance field belief_filter_used_true_block_boundaries; condition_schedule_observed is not uniformly False; parameter_count changes across folds or is missing; complete common/shared/full stimulus-pre panel unavailable
- `E2_latent_precedes_behavior_bias` (failed=0): hierarchical_observation_model is not uniformly True; nested_cv_latent_dimension is not uniformly True; unit_qc_applied is not uniformly True; context_coverage_valid is not uniformly True; parameter_count_includes_preprocessing is not uniformly True; hidden_context_inference is not uniformly True; test_context_observed is not uniformly False; belief_filter_used_true_block_boundaries is not uniformly False; behavior_bias_used_true_block_boundaries is not uniformly False; lead records do not exactly match the strict E1 session/animal cohort; the strict E1 model cohort or method provenance is invalid
- `P0a_aligned_task_improves_prediction_vs_frozen` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.0024310070015474325, CI=[0.0012379993876758281, 0.003979095880244444], raw_p=5.304813385009766e-06; l2: conclusion=support, n=30, estimate=0.004107516535009095, CI=[0.002218271399833035, 0.00654866282385085], raw_p=4.610046744346619e-06; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=5.30481338501e-06); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0b_aligned_task_beats_shuffled` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.001211568177126122, CI=[0.000680089378008703, 0.0018322525995008252], raw_p=2.9867514967918396e-06; l2: conclusion=support, n=30, estimate=0.0020118299553517874, CI=[0.0011746631922471406, 0.002994505864573219], raw_p=1.6195699572563171e-06; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=2.98675149679e-06); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0c_aligned_adds_value_over_matched_homeostasis` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.0024354150219953246, CI=[0.001248913662966034, 0.004012353270735189], raw_p=3.997236490249634e-06; l2: conclusion=support, n=30, estimate=0.004104866967973079, CI=[0.002230666236489589, 0.0065737887957078266], raw_p=2.9867514967918396e-06; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=3.99723649025e-06); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0d_local_absolute_accuracy` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.056666666666666685, CI=[0.02997916666666677, 0.08166666666666669], raw_p=0.00028261244544550504; l2: conclusion=support, n=30, estimate=0.05666666666666667, CI=[0.02916666666666669, 0.08166666666666669], raw_p=0.000305714097041996; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=0.000305714097042); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0e_local_noninferior_tuned_bptt` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.07491666666666666, CI=[0.034581249999999994, 0.11733958333333325], raw_p=0.00047877202703846004; l2: conclusion=support, n=30, estimate=0.07491666666666666, CI=[0.03491250000000002, 0.11742291666666656], raw_p=0.00047895487883489285; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=0.000478954878835); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0f_local_noninferior_tuned_gru` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.05841666666666667, CI=[0.021916666666666685, 0.09916874999999999], raw_p=0.003719669399051919; l2: conclusion=support, n=30, estimate=0.05841666666666667, CI=[0.022831250000000032, 0.09733541666666666], raw_p=0.003244611805976201; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=0.00371966939905); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P1a_masked_outer_product_identity` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=2.16023152891e-08); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P1b_credit_tangent_respects_feedback_bound` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=2.16023152891e-08); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P1c_highrank_physical_update_coexists_with_lowdim_credit` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 34 registered claims (raw Wilcoxon p=2.16023152891e-08); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0_overall` (failed=0): non-inferential stage gate; P0a_aligned_task_improves_prediction_vs_frozen=support; P0b_aligned_task_beats_shuffled=support; P0c_aligned_adds_value_over_matched_homeostasis=support; P0d_local_absolute_accuracy=support; P0e_local_noninferior_tuned_bptt=support; P0f_local_noninferior_tuned_gru=support
- `P2_overall` (failed=0): non-inferential P2 stage gate; P2b_md_context_nll=support; P2c_md_context_brier=support; P2d_md_calibration=support; P2e_md_switch_latency=support; P2f_md_false_switch=support; P2g_md_behavior=support; P2h_md_retains_oracle_gain=support; P2j_clamp_causal=support; P2k_delay_causal=support; P2l_shuffle_causal=support; strict panel issues: none
- `S1_exp10_hmm_context_inference` (failed=0): scope=simulated_hidden_context_inference; detailed conclusion=support_simulated_hidden_context_inference; q/h-cell mean range=[0.099677, 0.454324]; frozen recurrent; separately refit base readouts; no biological-mechanism, recurrent-plasticity, or efficiency claim; protocol=3a2abc0021fe97db655430ca94986700880e898f980c0ded7e7d33f1c069ad5e; scoped raw sha256=5c2f37e12864a73e5d2202dbceb077f3caae147fa94c6ce94b3309f2656c9749; clean-run manifest sha256=b0e29f5053a37593a197832ee12adc93ccb80fb55bd65003f20f90fff67aba94; run git commit=52fdcaa1e55ae0e0510ecca553c5acf6a4358072
- `S2_exp10_md_context_inference` (failed=0): scope=simulated_hidden_context_inference; detailed conclusion=support_simulated_hidden_context_inference; q/h-cell mean range=[0.0474984, 0.450007]; frozen recurrent; separately refit base readouts; no biological-mechanism, recurrent-plasticity, or efficiency claim; protocol=3a2abc0021fe97db655430ca94986700880e898f980c0ded7e7d33f1c069ad5e; scoped raw sha256=5c2f37e12864a73e5d2202dbceb077f3caae147fa94c6ce94b3309f2656c9749; clean-run manifest sha256=b0e29f5053a37593a197832ee12adc93ccb80fb55bd65003f20f90fff67aba94; run git commit=52fdcaa1e55ae0e0510ecca553c5acf6a4358072
- `S3_exp10_hmm_functional_pipeline` (failed=0): scope=separately_refit_functional_pipeline; detailed conclusion=support_functional_pipeline_formal; q/h-cell mean range=[0.0489918, 0.146146]; frozen recurrent; separately refit base readouts; no biological-mechanism, recurrent-plasticity, or efficiency claim; protocol=3a2abc0021fe97db655430ca94986700880e898f980c0ded7e7d33f1c069ad5e; scoped raw sha256=5c2f37e12864a73e5d2202dbceb077f3caae147fa94c6ce94b3309f2656c9749; clean-run manifest sha256=b0e29f5053a37593a197832ee12adc93ccb80fb55bd65003f20f90fff67aba94; run git commit=52fdcaa1e55ae0e0510ecca553c5acf6a4358072
- `S4_exp10_md_functional_pipeline` (failed=0): scope=separately_refit_functional_pipeline; detailed conclusion=support_functional_pipeline_formal; q/h-cell mean range=[0.0407393, 0.145414]; frozen recurrent; separately refit base readouts; no biological-mechanism, recurrent-plasticity, or efficiency claim; protocol=3a2abc0021fe97db655430ca94986700880e898f980c0ded7e7d33f1c069ad5e; scoped raw sha256=5c2f37e12864a73e5d2202dbceb077f3caae147fa94c6ce94b3309f2656c9749; clean-run manifest sha256=b0e29f5053a37593a197832ee12adc93ccb80fb55bd65003f20f90fff67aba94; run git commit=52fdcaa1e55ae0e0510ecca553c5acf6a4358072
- `S5_exp10_md_retains_oracle_gain` (failed=0): scope=separately_refit_noninferiority_margin; detailed conclusion=support_macro_average_90pct_oracle_gain_margin; q/h-cell mean range=[-0.00607921, 0.0131486]; frozen recurrent; separately refit base readouts; no biological-mechanism, recurrent-plasticity, or efficiency claim; protocol=3a2abc0021fe97db655430ca94986700880e898f980c0ded7e7d33f1c069ad5e; scoped raw sha256=5c2f37e12864a73e5d2202dbceb077f3caae147fa94c6ce94b3309f2656c9749; clean-run manifest sha256=b0e29f5053a37593a197832ee12adc93ccb80fb55bd65003f20f90fff67aba94; run git commit=52fdcaa1e55ae0e0510ecca553c5acf6a4358072
- `S6_exp10_md_clamp_counterfactual` (failed=0): scope=fixed_checkpoint_within_model_counterfactual; detailed conclusion=support_within_model_counterfactual; q/h-cell mean range=[0.0361962, 0.138036]; frozen recurrent; separately refit base readouts; no biological-mechanism, recurrent-plasticity, or efficiency claim; protocol=3a2abc0021fe97db655430ca94986700880e898f980c0ded7e7d33f1c069ad5e; scoped raw sha256=5c2f37e12864a73e5d2202dbceb077f3caae147fa94c6ce94b3309f2656c9749; clean-run manifest sha256=b0e29f5053a37593a197832ee12adc93ccb80fb55bd65003f20f90fff67aba94; run git commit=52fdcaa1e55ae0e0510ecca553c5acf6a4358072
- `S7_exp10_md_delay_counterfactual` (failed=0): scope=fixed_checkpoint_within_model_counterfactual; detailed conclusion=support_within_model_counterfactual; q/h-cell mean range=[0.00731382, 0.0490158]; frozen recurrent; separately refit base readouts; no biological-mechanism, recurrent-plasticity, or efficiency claim; protocol=3a2abc0021fe97db655430ca94986700880e898f980c0ded7e7d33f1c069ad5e; scoped raw sha256=5c2f37e12864a73e5d2202dbceb077f3caae147fa94c6ce94b3309f2656c9749; clean-run manifest sha256=b0e29f5053a37593a197832ee12adc93ccb80fb55bd65003f20f90fff67aba94; run git commit=52fdcaa1e55ae0e0510ecca553c5acf6a4358072
- `S8_exp10_md_shuffle_counterfactual` (failed=0): scope=fixed_checkpoint_within_model_counterfactual; detailed conclusion=support_within_model_counterfactual; q/h-cell mean range=[0.0401489, 0.150213]; frozen recurrent; separately refit base readouts; no biological-mechanism, recurrent-plasticity, or efficiency claim; protocol=3a2abc0021fe97db655430ca94986700880e898f980c0ded7e7d33f1c069ad5e; scoped raw sha256=5c2f37e12864a73e5d2202dbceb077f3caae147fa94c6ce94b3309f2656c9749; clean-run manifest sha256=b0e29f5053a37593a197832ee12adc93ccb80fb55bd65003f20f90fff67aba94; run git commit=52fdcaa1e55ae0e0510ecca553c5acf6a4358072
- `R1_ibl_hmm_context_inference` (failed=0): IBL trial-table behavior only; no neural activity or shared neural dynamics; planned/paired sessions=30/30; invalid HMM fits=0; latest run status=complete; source run id=c9ed2f4f-52bb-4f84-abb0-df5321d10e07; cohort manifest sha256=112c84ad93eee49186ab117343ebebb4921d2f1bcea57a9c9326ca38d337a0e6; scoped raw sha256=49b9ac6b8a48a7824cdf7288d878e6353fbb0e574b065b9bf46aad828730e67a
- `R2_ibl_history_context_inference` (failed=0): IBL trial-table behavior only; no neural activity or shared neural dynamics; planned/paired sessions=30/30; invalid HMM fits=0; latest run status=complete; source run id=c9ed2f4f-52bb-4f84-abb0-df5321d10e07; cohort manifest sha256=112c84ad93eee49186ab117343ebebb4921d2f1bcea57a9c9326ca38d337a0e6; scoped raw sha256=49b9ac6b8a48a7824cdf7288d878e6353fbb0e574b065b9bf46aad828730e67a
- `R3_ibl_hmm_behavior_prediction` (failed=0): IBL trial-table behavior only; no neural activity or shared neural dynamics; planned/paired sessions=30/30; invalid HMM fits=0; latest run status=complete; source run id=c9ed2f4f-52bb-4f84-abb0-df5321d10e07; cohort manifest sha256=112c84ad93eee49186ab117343ebebb4921d2f1bcea57a9c9326ca38d337a0e6; scoped raw sha256=49b9ac6b8a48a7824cdf7288d878e6353fbb0e574b065b9bf46aad828730e67a
- `R4_ibl_history_behavior_prediction` (failed=0): IBL trial-table behavior only; no neural activity or shared neural dynamics; planned/paired sessions=30/30; invalid HMM fits=0; latest run status=complete; source run id=c9ed2f4f-52bb-4f84-abb0-df5321d10e07; cohort manifest sha256=112c84ad93eee49186ab117343ebebb4921d2f1bcea57a9c9326ca38d337a0e6; scoped raw sha256=49b9ac6b8a48a7824cdf7288d878e6353fbb0e574b065b9bf46aad828730e67a
- `T1_arc_hierarchical_vs_flat` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, flat_local=1657 total/25 trainable; test_split_role=ood; 30 seeds; coverage=0.0125 vs required 0.9000; revision=399030444e0ab0cc8b4e199870fb20b863846f34; clean commit=4f77d1abee9f6213638a65c1e4c4525fcee83f1a; raw sha256=0865d15177be359194b454ca8c94620df34e3ddf449b096b5e796847e0533b4d; run manifest sha256=7e46d4e0b62106e20b047f89ea25903619806f738683fc1be38d6ae4a87e8ead
- `T2_arc_trace_vs_flat` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters trace_local=1681 total/49 trainable, flat_local=1657 total/25 trainable; test_split_role=ood; 30 seeds; coverage=0.0125 vs required 0.9000; revision=399030444e0ab0cc8b4e199870fb20b863846f34; clean commit=4f77d1abee9f6213638a65c1e4c4525fcee83f1a; raw sha256=0865d15177be359194b454ca8c94620df34e3ddf449b096b5e796847e0533b4d; run manifest sha256=7e46d4e0b62106e20b047f89ea25903619806f738683fc1be38d6ae4a87e8ead
- `T3_arc_hierarchical_vs_heuristic` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, support_heuristic=0 total/0 trainable; test_split_role=ood; 30 seeds; coverage=0.0125 vs required 0.9000; revision=399030444e0ab0cc8b4e199870fb20b863846f34; clean commit=4f77d1abee9f6213638a65c1e4c4525fcee83f1a; raw sha256=0865d15177be359194b454ca8c94620df34e3ddf449b096b5e796847e0533b4d; run manifest sha256=7e46d4e0b62106e20b047f89ea25903619806f738683fc1be38d6ae4a87e8ead
- `T4_arc_hierarchical_vs_gru` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, gru_bptt=1841 total/1841 trainable; test_split_role=ood; 30 seeds; coverage=0.0125 vs required 0.9000; revision=399030444e0ab0cc8b4e199870fb20b863846f34; clean commit=4f77d1abee9f6213638a65c1e4c4525fcee83f1a; raw sha256=0865d15177be359194b454ca8c94620df34e3ddf449b096b5e796847e0533b4d; run manifest sha256=7e46d4e0b62106e20b047f89ea25903619806f738683fc1be38d6ae4a87e8ead
- `T5_arc_hierarchical_90pct_gru` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, gru_bptt=1841 total/1841 trainable; test_split_role=ood; 30 seeds; coverage=0.0125 vs required 0.9000; revision=399030444e0ab0cc8b4e199870fb20b863846f34; clean commit=4f77d1abee9f6213638a65c1e4c4525fcee83f1a; raw sha256=0865d15177be359194b454ca8c94620df34e3ddf449b096b5e796847e0533b4d; run manifest sha256=7e46d4e0b62106e20b047f89ea25903619806f738683fc1be38d6ae4a87e8ead
- `T6_arc_trace_increment` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters trace_local=1681 total/49 trainable, hierarchical_local=1673 total/41 trainable; test_split_role=ood; 30 seeds; coverage=0.0125 vs required 0.9000; revision=399030444e0ab0cc8b4e199870fb20b863846f34; clean commit=4f77d1abee9f6213638a65c1e4c4525fcee83f1a; raw sha256=0865d15177be359194b454ca8c94620df34e3ddf449b096b5e796847e0533b4d; run manifest sha256=7e46d4e0b62106e20b047f89ea25903619806f738683fc1be38d6ae4a87e8ead
- `M1_maze_hierarchical_vs_flat` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, flat_local=1657 total/25 trainable; test_split_role=ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=a71a2d1e0931c79f74cd91c5accd13f164d34c73; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=ce270abdeff30be94f152d575ec05d26d46c49dc8755c3bdea3cb85bb46875f2; run manifest sha256=44dedd8c58237d7624e4a7c34ac22260d20fdca88454b75ded8bd42f30a56022
- `M2_maze_trace_vs_flat` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters trace_local=1681 total/49 trainable, flat_local=1657 total/25 trainable; test_split_role=ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=a71a2d1e0931c79f74cd91c5accd13f164d34c73; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=ce270abdeff30be94f152d575ec05d26d46c49dc8755c3bdea3cb85bb46875f2; run manifest sha256=44dedd8c58237d7624e4a7c34ac22260d20fdca88454b75ded8bd42f30a56022
- `M3_maze_hierarchical_vs_heuristic` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, support_heuristic=0 total/0 trainable; test_split_role=ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=a71a2d1e0931c79f74cd91c5accd13f164d34c73; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=ce270abdeff30be94f152d575ec05d26d46c49dc8755c3bdea3cb85bb46875f2; run manifest sha256=44dedd8c58237d7624e4a7c34ac22260d20fdca88454b75ded8bd42f30a56022
- `M4_maze_hierarchical_vs_gru` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, gru_bptt=1841 total/1841 trainable; test_split_role=ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=a71a2d1e0931c79f74cd91c5accd13f164d34c73; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=ce270abdeff30be94f152d575ec05d26d46c49dc8755c3bdea3cb85bb46875f2; run manifest sha256=44dedd8c58237d7624e4a7c34ac22260d20fdca88454b75ded8bd42f30a56022
- `M5_maze_hierarchical_90pct_gru` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, gru_bptt=1841 total/1841 trainable; test_split_role=ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=a71a2d1e0931c79f74cd91c5accd13f164d34c73; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=ce270abdeff30be94f152d575ec05d26d46c49dc8755c3bdea3cb85bb46875f2; run manifest sha256=44dedd8c58237d7624e4a7c34ac22260d20fdca88454b75ded8bd42f30a56022
- `M6_maze_trace_increment` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters trace_local=1681 total/49 trainable, hierarchical_local=1673 total/41 trainable; test_split_role=ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=a71a2d1e0931c79f74cd91c5accd13f164d34c73; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=ce270abdeff30be94f152d575ec05d26d46c49dc8755c3bdea3cb85bb46875f2; run manifest sha256=44dedd8c58237d7624e4a7c34ac22260d20fdca88454b75ded8bd42f30a56022
- `N1_sudoku_hierarchical_vs_flat` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, flat_local=1657 total/25 trainable; test_split_role=non_ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=0db6de036b80b4e8e4574abe6e15026331bd5c2c; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=f37abb5bf569b28a9d715d1b3eb935c069d95739d04819ff49e198755ef0c648; run manifest sha256=9535115f78f30f400502f1b6827ba6b96225d3cbcfb0bcdd39338ef0b0c94b3b
- `N2_sudoku_trace_vs_flat` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters trace_local=1681 total/49 trainable, flat_local=1657 total/25 trainable; test_split_role=non_ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=0db6de036b80b4e8e4574abe6e15026331bd5c2c; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=f37abb5bf569b28a9d715d1b3eb935c069d95739d04819ff49e198755ef0c648; run manifest sha256=9535115f78f30f400502f1b6827ba6b96225d3cbcfb0bcdd39338ef0b0c94b3b
- `N3_sudoku_hierarchical_vs_heuristic` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, support_heuristic=0 total/0 trainable; test_split_role=non_ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=0db6de036b80b4e8e4574abe6e15026331bd5c2c; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=f37abb5bf569b28a9d715d1b3eb935c069d95739d04819ff49e198755ef0c648; run manifest sha256=9535115f78f30f400502f1b6827ba6b96225d3cbcfb0bcdd39338ef0b0c94b3b
- `N4_sudoku_hierarchical_vs_gru` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, gru_bptt=1841 total/1841 trainable; test_split_role=non_ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=0db6de036b80b4e8e4574abe6e15026331bd5c2c; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=f37abb5bf569b28a9d715d1b3eb935c069d95739d04819ff49e198755ef0c648; run manifest sha256=9535115f78f30f400502f1b6827ba6b96225d3cbcfb0bcdd39338ef0b0c94b3b
- `N5_sudoku_hierarchical_90pct_gru` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters hierarchical_local=1673 total/41 trainable, gru_bptt=1841 total/1841 trainable; test_split_role=non_ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=0db6de036b80b4e8e4574abe6e15026331bd5c2c; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=f37abb5bf569b28a9d715d1b3eb935c069d95739d04819ff49e198755ef0c648; run manifest sha256=9535115f78f30f400502f1b6827ba6b96225d3cbcfb0bcdd39338ef0b0c94b3b
- `N6_sudoku_trace_increment` (failed=0): Hybrid selector over one shared, target-free proposal library only; no neural/biological claim and no end-to-end efficiency claim; selector-level parameters trace_local=1681 total/49 trainable, hierarchical_local=1673 total/41 trainable; test_split_role=non_ood; 30 seeds; coverage=1.0000 vs required 0.9000; revision=0db6de036b80b4e8e4574abe6e15026331bd5c2c; clean commit=8fba6178e8de0a3a11d3e3dee985d7d3efd6a83d; raw sha256=f37abb5bf569b28a9d715d1b3eb935c069d95739d04819ff49e198755ef0c648; run manifest sha256=9535115f78f30f400502f1b6827ba6b96225d3cbcfb0bcdd39338ef0b0c94b3b
- `U1_ibl_shared_neural_dynamics` (failed=0): One-step conditional Poisson likelihood, not a full latent LDS; sessions nested within animals; sensitivity views cannot update this claim; compact manifest sha256=a5acb134ae4b34f47db150948a7f7ab58e8eb85e204fb981e0ca744eba328a09; compact bundle sha256=f6cd351717986ede771a6bbbe755edeb3c30ef4bda48e86c8471bcf4364a41a4; registered formal JSON sha256=7717227beca59f6a39286a5bfc6ded10ed7b0896ce6f49409542f1743bf79680; portable formal-config sha256=106b99b861b0de82bfd020c23881e14a861b09ddc50f70752badb56b7a104913; macro mapping exp14_allen_macro_region_mapping_v1 sha256=3bac702ed6b3ee5c21acbbfd929b077baa63226369ca8e1bef0b6faeb487fc23; ontology/provenance sha256=63654b8d35c7c1b5665636b645da774776ee8263658192f5dca1e815095e9147/a01b7fa535e6de437ac46e8cf9de68a87d6a9b5587d055a3935476d956109fdc; mapping compact scope=a5acb134ae4b34f47db150948a7f7ab58e8eb85e204fb981e0ca744eba328a09; acronym count/hash=168/53c5a0cb3591749aad8fc2848b2627dfe93a4a747fa391e4ec68094211b8369b; shared/full/retention Holm p=1/1/1; classification trigger=none; raw sha256=8c14068e5855dc16fe3039ac86465fe9588ccda697b86bfce4340364f6de0ccd; clean commit=b68a1e7638636811116bb352330fb4a1fe2d5a75
- `V1_exp15_arc_slow_fast_vs_flat` (failed=0): Task-specialized finite-program selector only; no neural or biological claim; slow/fast and flat exact accuracy=0.00250627/0.00250627; coverage=0.01253133 vs required 0.9000; compute is an audited abstract operation proxy, not FLOPs/time/energy; source manifest=76e2360f6673093730676345fd3db8bf289be3f58179c002980a4e91ae0d9cda; clean commit=cbec277503d02844729d8fea5648a9e34e2ce44b; raw sha256=2353bf280aa5ded0fd3c1d202fe5eedf796c0de07091afffa4db0178548c9fcf

## P2 formal diagnostics

These are descriptive seed-level diagnostics. Each base-gate entry first averages the 16 q/h cells within a complete seed, then averages those seed macros. Therefore a macro average does not assert that the result holds in every q/h cell.
Fit counts below audit seed-by-q/h cells descriptively; they are not independent inferential replicates. Core-claim inference remains at the seed level.

### Base-gate macro averages

| Base gate | Complete seed macros | NLL | Brier | ECE | Latency | False switch | Behavior | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle Bayes | 30 | 0.3383 | 0.1135 | 0.01635 | 3.292 | 0.004706 | 0.891 | 0.9755 |
| Supervised upper bound (ineligible) | 30 | 0.3419 | 0.1147 | 0.01984 | 3.255 | 0.004784 | 0.8903 | 0.9765 |
| Learned HMM | 30 | 0.3656 | 0.1222 | 0.04195 | 3.057 | 0.01163 | 0.8922 | 1.045 |
| MD recurrent belief | 30 | 0.3582 | 0.1207 | 0.03239 | 3.042 | 0.005454 | 0.885 | 0.9772 |
| No gate | 30 | 0.6931 | 0.25 | 0.02883 | 6 | 0 | 0.7507 | 0.5 |

### Fit and identifiability diagnostics

- Learned-HMM convergence: 369/480 reported fits converged; EM iterations: mean 51.09, median 43, range 1–100.
- All finite held-out HMM scores remain in the preregistered P2a seed macro whether or not EM met its tolerance; non-converged fits are retained as a sensitivity caveat, not silently dropped.

| MD cue band | Identifiable / reported fits | Identifiable rate | Neutral fallback among non-identifiable |
|---|---:|---:|---:|
| q = 0.55 (weak cue) | 2/120 | 0.01667 | 118/118 |
| q >= 0.70 | 360/360 | 1 | unavailable |

The weak-cue safeguard returns neutral parameter estimates (q̂≈0.5, ĥ≈0.5) whenever the MD moment anchor is not identifiable; the final column audits that fallback in the observed formal fits.

### MD q/h-cell range

Each endpoint below is first averaged across seeds within a q/h cell. The extrema expose cell heterogeneity hidden by the macro average.

| Endpoint | Minimum cell mean (q, h) | Maximum cell mean (q, h) |
|---|---:|---:|
| Context NLL | 0.002941 (q=1, h=0.01) | 0.6964 (q=0.55, h=0.1) |
| Context Brier | 0.0006441 (q=1, h=0.01) | 0.2513 (q=0.55, h=0.1) |
| Context ECE | 0.002548 (q=1, h=0.01) | 0.07085 (q=0.55, h=0.01) |
| Switch latency (trials) | 0 (q=1, h=0.05) | 6 (q=0.55, h=0.05) |
| False-switch rate | 0 (q=0.55, h=0.05) | 0.01804 (q=0.7, h=0.1) |
| Behavior balanced accuracy | 0.7504 (q=0.55, h=0.1) | 0.999 (q=1, h=0.01) |
| Energy proxy / trial | 0.5 (q=0.55, h=0.05) | 1.348 (q=1, h=0.2) |

### P2i energy-ratio interpretation

P2i is registered on the log(MD/no-gate energy) scale. Exponentiating the summary estimate and CI gives an energy ratio of 1.954 [1.946, 1.964].


## Incremental exp10 bridge pilot (not formal)

This N=32 pilot uses 30 independent seeds and is reported separately from the registered N=256 formal grid. Base gates use separately fitted readouts, so their differences concern whole functional pipelines, not a fixed-readout gate effect. They are ineligible for biological-mechanism, recurrent-plasticity, or efficiency claims. Clamp/delay/shuffle are fixed-checkpoint within-model counterfactuals; all three are inconclusive.

| Comparison | Scope | Paired balanced-accuracy difference [95% seed-bootstrap CI] | Holm p | Conclusion |
|---|---|---:|---:|---|
| oracle_vs_no_gate | separately_refit_readout_functional_pipeline | 0.0358 [0.0183, 0.0532] | 0.001917 | **descriptive_ceiling_support** |
| hmm_vs_no_gate | separately_refit_readout_functional_pipeline | 0.0252 [0.0075, 0.0438] | 0.03203 | **functional_pipeline_support_pilot** |
| md_vs_no_gate | separately_refit_readout_functional_pipeline | 0.0033 [0.0000, 0.0093] | 0.3147 | **inconclusive_functional_pipeline_pilot** |
| md_vs_clamp | fixed_receiver_readout_within_model_counterfactual | 0.0031 [0.0000, 0.0089] | 0.3147 | **inconclusive_within_model_counterfactual** |
| md_vs_delay | fixed_receiver_readout_within_model_counterfactual | 0.0015 [-0.0021, 0.0066] | 0.9809 | **inconclusive_within_model_counterfactual** |
| md_vs_shuffle | fixed_receiver_readout_within_model_counterfactual | 0.0018 [-0.0003, 0.0058] | 0.9809 | **inconclusive_within_model_counterfactual** |

## exp10 N=256 bridge formal grid

Thirty seeds are paired within each of four q/h cells and then equally macro-averaged within seed. Base-gate behavior comparisons use separately fitted readouts and therefore support only whole functional pipelines. Clamp/delay/shuffle reuse the intact MD-like receiver and readout as within-model counterfactuals. Recurrent weights are frozen; no row is eligible for biological-mechanism, three-factor-plasticity, or efficiency claims.

The scoped rows are bound to clean Git commit `52fdcaa1e55ae0e0510ecca553c5acf6a4358072` (`dirty=false`), clean-run manifest `b0e29f5053a37593a197832ee12adc93ccb80fb55bd65003f20f90fff67aba94`, and scoped raw snapshot `5c2f37e12864a73e5d2202dbceb077f3caae147fa94c6ce94b3309f2656c9749`. The run manifest records per-seed run IDs plus SHA-256 values for config, planned conditions, status, manifest, environment, metrics, and run log artifacts.

| Comparison | Scope | Seed-macro difference [95% CI] | q/h-cell mean range | exp10-family Holm p | Conclusion |
|---|---|---:|---:|---:|---|
| hmm_context_vs_no_gate | simulated_hidden_context_inference | 0.2777 [0.2696, 0.2860] | [0.0997, 0.4543] | 1.676e-08 | **support_simulated_hidden_context_inference** |
| md_context_vs_no_gate | simulated_hidden_context_inference | 0.2583 [0.2455, 0.2701] | [0.0475, 0.4500] | 1.676e-08 | **support_simulated_hidden_context_inference** |
| hmm_behavior_vs_no_gate | separately_refit_functional_pipeline | 0.1000 [0.0962, 0.1038] | [0.0490, 0.1461] | 1.676e-08 | **support_functional_pipeline_formal** |
| md_behavior_vs_no_gate | separately_refit_functional_pipeline | 0.0964 [0.0920, 0.1009] | [0.0407, 0.1454] | 1.676e-08 | **support_functional_pipeline_formal** |
| oracle_behavior_vs_no_gate | descriptive_oracle_ceiling | 0.1008 [0.0971, 0.1044] | [0.0520, 0.1470] | 1.676e-08 | **descriptive_oracle_ceiling_support** |
| md_retains_90pct_oracle_gain | separately_refit_noninferiority_margin | 0.0057 [0.0034, 0.0078] | [-0.0061, 0.0131] | 4.408e-05 | **support_macro_average_90pct_oracle_gain_margin** |
| md_vs_clamp | fixed_checkpoint_within_model_counterfactual | 0.0899 [0.0866, 0.0934] | [0.0362, 0.1380] | 1.676e-08 | **support_within_model_counterfactual** |
| md_vs_delay | fixed_checkpoint_within_model_counterfactual | 0.0223 [0.0199, 0.0247] | [0.0073, 0.0490] | 1.676e-08 | **support_within_model_counterfactual** |
| md_vs_shuffle | fixed_checkpoint_within_model_counterfactual | 0.0992 [0.0942, 0.1045] | [0.0401, 0.1502] | 1.676e-08 | **support_within_model_counterfactual** |

The MD-like 90%-of-oracle margin supports only the predeclared seed-macro average: at least one q/h cell has a negative mean margin, so no every-cell retention claim is made.

## exp11 IBL hidden-block benchmark (behavior only)

This section analyzes trial-table behavior only: no spikes, neural activity, or shared neural dynamics are fit. Conclusions use animal-primary inference with sessions nested within animal, preserve failed/missing conditions, and are bound to cohort manifest `112c84ad93eee49186ab117343ebebb4921d2f1bcea57a9c9326ca38d337a0e6`.

Difference is reference minus candidate, so positive values favor the candidate. Holm correction is across the four exp11 behavior-only claims, separately from the legacy core-claim family.

| Claim | planned / paired sessions | animals | animal-mean difference (positive = better) [hierarchical 95% CI] | exp11-family Holm p | Conclusion |
|---|---:|---:|---:|---:|---|
| hmm_context_nll_gain | 30 / 30 | 30 | 0.3768 [0.3313, 0.4178] | 1.49e-08 | **support** |
| history_context_nll_gain | 30 / 30 | 30 | -0.5649 [-0.7772, -0.3784] | 2.498e-06 | **oppose** |
| hmm_behavior_log_loss_gain | 30 / 30 | 30 | -0.001087 [-0.003275, 0.0007699] | 0.9838 | **inconclusive** |
| history_behavior_log_loss_gain | 30 / 30 | 30 | -0.003753 [-0.00724, -0.000773] | 0.2806 | **inconclusive** |

## exp13 public ARC hybrid-solver audit

All hybrid selectors receive one shared, target-free proposal library. The candidate oracle accesses labels only after proposal generation and defines proposal coverage. Parameter counts below describe the selector only: they exclude proposal-library construction and solver cost, so this panel makes no end-to-end efficiency claim and no neural or biological claim.

Dataset `ARC-AGI-1` uses `test_split_role=ood` with 30 independent seeds. Candidate coverage is 0.01253 against the registered 0.9 gate. The validated scoped raw/run SHA-256 values are `0865d15177be359194b454ca8c94620df34e3ddf449b096b5e796847e0533b4d` / `7e46d4e0b62106e20b047f89ea25903619806f738683fc1be38d6ae4a87e8ead`.

### Absolute exact accuracy

| Selector | Exact accuracy [95% CI] | Coverage | Selector parameters (trainable) |
|---|---:|---:|---:|
| support_heuristic | 0.005013 [0, 0.01253] | 0.01253 | 0 (0) |
| flat_local | 0.003008 [0, 0.008521] | 0.01253 | 1657 (25) |
| hierarchical_local | 0.003008 [0, 0.008521] | 0.01253 | 1673 (41) |
| trace_local | 0.003425 [0, 0.009357] | 0.01253 | 1681 (49) |
| gru_bptt | 0.005013 [0, 0.01253] | 0.01253 | 1841 (1841) |
| candidate_oracle | 0.01253 [0.002506, 0.02506] | 0.01253 | 0 (0) |

### Registered selector comparisons

The statistics unit is the source/augmentation dependency component with seeds nested within task.

| Comparison | Task-component contrast [95% CI] | Holm p | Coverage gate | Conclusion |
|---|---:|---:|---:|---|
| hierarchical_vs_flat | 1.739e-20 [-0.0002506, 0.0002506] | 1 | False | **inconclusive** |
| trace_vs_flat | 0.0004177 [0, 0.001086] | 1 | False | **inconclusive** |
| hierarchical_vs_support_heuristic | -0.002005 [-0.007018, 0.001003] | 1 | False | **inconclusive** |
| hierarchical_vs_gru_bptt | -0.002005 [-0.007018, 0.001003] | 1 | False | **inconclusive** |
| hierarchical_retains_90pct_gru | -0.001504 [-0.006266, 0.001253] | 1 | False | **inconclusive** |
| trace_vs_hierarchical | 0.0004177 [0, 0.001086] | 1 | False | **inconclusive** |

This is a validated hybrid proposal-selection audit, not a proposal-free HRM/CTM reproduction. It cannot establish shared neural dynamics or a biological mechanism.

## exp13 public Maze hybrid-solver audit

All hybrid selectors receive one shared, target-free proposal library. The candidate oracle accesses labels only after proposal generation and defines proposal coverage. Parameter counts below describe the selector only: they exclude proposal-library construction and solver cost, so this panel makes no end-to-end efficiency claim and no neural or biological claim.

Dataset `albertoRodriguez97/MazeBench` uses `test_split_role=ood` with 30 independent seeds. Candidate coverage is 1 against the registered 0.9 gate. The validated scoped raw/run SHA-256 values are `ce270abdeff30be94f152d575ec05d26d46c49dc8755c3bdea3cb85bb46875f2` / `44dedd8c58237d7624e4a7c34ac22260d20fdca88454b75ded8bd42f30a56022`.

### Absolute exact accuracy

| Selector | Exact accuracy [95% CI] | Coverage | Selector parameters (trainable) |
|---|---:|---:|---:|
| support_heuristic | 0.8889 [0.7222, 1] | 1 | 0 (0) |
| flat_local | 0.9944 [0.9852, 1] | 1 | 1657 (25) |
| hierarchical_local | 0.9907 [0.9741, 1] | 1 | 1673 (41) |
| trace_local | 0.9889 [0.9704, 1] | 1 | 1681 (49) |
| gru_bptt | 1 [1, 1] | 1 | 1841 (1841) |
| candidate_oracle | 1 [1, 1] | 1 | 0 (0) |

### Registered selector comparisons

The statistics unit is the source/augmentation dependency component with seeds nested within task.

| Comparison | Task-component contrast [95% CI] | Holm p | Coverage gate | Conclusion |
|---|---:|---:|---:|---|
| hierarchical_vs_flat | -0.003704 [-0.02037, 0.007407] | 1 | True | **inconclusive** |
| trace_vs_flat | -0.005556 [-0.02222, 0.005556] | 1 | True | **inconclusive** |
| hierarchical_vs_support_heuristic | 0.1019 [0, 0.2574] | 0.8986 | True | **inconclusive** |
| hierarchical_vs_gru_bptt | -0.009259 [-0.02593, 0] | 0.8986 | True | **inconclusive** |
| hierarchical_retains_90pct_gru | 0.09074 [0.07407, 0.1] | 0.000352 | True | **support** |
| trace_vs_hierarchical | -0.001852 [-0.005556, 0] | 0.9519 | True | **inconclusive** |

This is a validated hybrid proposal-selection audit, not a proposal-free HRM/CTM reproduction. It cannot establish shared neural dynamics or a biological mechanism.

## exp13 public Sudoku hybrid-solver audit

All hybrid selectors receive one shared, target-free proposal library. The candidate oracle accesses labels only after proposal generation and defines proposal coverage. Parameter counts below describe the selector only: they exclude proposal-library construction and solver cost, so this panel makes no end-to-end efficiency claim and no neural or biological claim.

Dataset `wichtounet/sudoku_dataset-v2` uses `test_split_role=non_ood` with 30 independent seeds. Candidate coverage is 1 against the registered 0.9 gate. The validated scoped raw/run SHA-256 values are `f37abb5bf569b28a9d715d1b3eb935c069d95739d04819ff49e198755ef0c648` / `9535115f78f30f400502f1b6827ba6b96225d3cbcfb0bcdd39338ef0b0c94b3b`.

This is not a registered OOD split. Consequently, even a significant numerical margin remains core-ineligible and is reported as **inconclusive**; it is not upgraded to support.

### Absolute exact accuracy

| Selector | Exact accuracy [95% CI] | Coverage | Selector parameters (trainable) |
|---|---:|---:|---:|
| support_heuristic | 1 [1, 1] | 1 | 0 (0) |
| flat_local | 1 [1, 1] | 1 | 1657 (25) |
| hierarchical_local | 1 [1, 1] | 1 | 1673 (41) |
| trace_local | 1 [1, 1] | 1 | 1681 (49) |
| gru_bptt | 1 [1, 1] | 1 | 1841 (1841) |
| candidate_oracle | 1 [1, 1] | 1 | 0 (0) |

### Registered selector comparisons

The statistics unit is the source/augmentation dependency component with seeds nested within task.

| Comparison | Task-component contrast [95% CI] | Holm p | Coverage gate | Conclusion |
|---|---:|---:|---:|---|
| hierarchical_vs_flat | 0 [0, 0] | 1 | True | **inconclusive** |
| trace_vs_flat | 0 [0, 0] | 1 | True | **inconclusive** |
| hierarchical_vs_support_heuristic | 0 [0, 0] | 1 | True | **inconclusive** |
| hierarchical_vs_gru_bptt | 0 [0, 0] | 1 | True | **inconclusive** |
| hierarchical_retains_90pct_gru | 0.1 [0.1, 0.1] | 7.279e-07 | True | **inconclusive** |
| trace_vs_hierarchical | 0 [0, 0] | 1 | True | **inconclusive** |

This is a validated hybrid proposal-selection audit, not a proposal-free HRM/CTM reproduction. It cannot establish shared neural dynamics or a biological mechanism.

## exp14 IBL multi-session neural audit

The registered endpoint is held-out one-step conditional Poisson likelihood. Inference is animal-primary with sessions nested within animal; this is not a full latent-LDS marginal likelihood.

| View | Panel | Scope | Common - shared NLL/count (positive favors shared) [95% CI] | Retained full gain | Scoped conclusion |
|---|---|---|---:|---:|---|
| stimulus_pre | primary_past_safe | registered_primary | -0.0009953 [-0.003061, 0.0001569] | — | **inconclusive** |
| stimulus_pre | full_trial_sensitivity | sensitivity_only | -0.0006052 [-0.001943, 0.0001545] | — | **inconclusive** |
| movement_pre | primary_past_safe | sensitivity_only | 6.898e-05 [-0.0001002, 0.0002695] | — | **inconclusive** |
| movement_pre | full_trial_sensitivity | sensitivity_only | 1.618e-05 [-6.146e-05, 0.0001114] | — | **inconclusive** |

Only `stimulus_pre / primary_past_safe` updates the core claim. Movement-pre and full-trial-covariate results remain sensitivity-only.

## exp15 verified-source ARC task-specialization audit

All 800 ARC-AGI-1 JSON files and the Apache-2.0 license were verified against source manifest `76e2360f6673093730676345fd3db8bf289be3f58179c002980a4e91ae0d9cda`. The held-out evaluation panel contains 399 source groups after the registered cross-split duplicate exclusion. Query targets are used only for scoring and the explicitly labeled candidate-coverage oracle.

| Condition | Exact tasks [95% source-group CI] | Candidate coverage | Mean measured / charged abstract proxy |
|---|---:|---:|---:|
| Slow/fast family belief | 0.2506% [0.0000%, 0.7519%] | 1.2531% | 132616.40 / 133187.06 |
| Flat matched selector | 0.2506% [0.0000%, 0.7519%] | 1.2531% | 132324.23 / 133187.06 |

The registered slow/fast-minus-flat difference is 0.0000 percentage points (95% paired source-group CI 0.0000 to 0.0000; Holm p=1). Candidate coverage is 1.2531% versus the registered 90.0% gate. The scoped conclusion is **inconclusive**; this is not evidence for hierarchical advantage.

Charged compute is exactly paired by a preregistered abstract operation proxy. It is not a FLOP, wall-clock, energy, or end-to-end efficiency measurement. This task adapter is not a BDH/HRM reproduction and cannot substitute for neural shared-dynamics evidence.

Trusted raw SHA-256: `2353bf280aa5ded0fd3c1d202fe5eedf796c0de07091afffa4db0178548c9fcf`; run-manifest SHA-256: `fc4c6a177d2c08a81dcac6e0075f6bff8fa60f22937f63b29c715c83d8d6a066`; clean run commit: `cbec277503d02844729d8fea5648a9e34e2ce44b`.

## exp15 Sudoku engineering audit (report-only)

This non-OOD, unmatched-mechanism panel is retained for engineering visibility and does not append a global claim. The bounded-search condition adds a distinct search mechanism and cannot be attributed to local dynamics alone.

| Condition | Exact accuracy [95% source-group CI] | Mean state evaluations | Conclusion |
|---|---:|---:|---|
| Local constraints + bounded search | 100.00% [100.00%, 100.00%] | 11.43 | **inconclusive** |
| Local constraints only | 75.00% [57.14%, 89.29%] | 7.32 | **inconclusive** |

Both rows remain **inconclusive** for the repository's mechanism claims despite the engineering utility of bounded search.

## exp16 micro-TRM-like Sudoku smoke (pilot only)

This is a strict-deterministic, clean-commit engineering smoke on an auditable synthetic fixture. It is not an official HRM/TRM reproduction, public-task validation, local-learning result, or formal claim.

| Condition | Exact accuracy [95% seed-bootstrap CI] | Parameters | Nominal core calls | Conclusion |
|---|---:|---:|---:|---|
| micro-TRM-like two-state | 0.00% [0.00%, 0.00%] | 11776 | 16 | **inconclusive** |
| single-state core-call matched | 0.00% [0.00%, 0.00%] | 11776 | 16 | **inconclusive** |

The paired exact-accuracy difference was 0.0000 [0.0000, 0.0000] (Wilcoxon p=1). Both models solved 0/8 held-out fixture puzzles exactly in every seed, so this run provides no evidence for a recursive-state advantage.

Trusted scoped raw SHA-256: `fd72d108ee2bba5a0dff94c0fd3dbd64de43b56293b5b5d4b06bf2676e409c5e`; run-manifest SHA-256: `9db321bb555fc61f7fac27121de695cca3d97793c5f64d6708dd180155d7131a`; clean run commit: `e1f80c74e3a29e4fec7f0cdca9e9c725e784d63c`.

## Interpretation safeguards

- Tuned BPTT rate-RNN and GRU baselines are isolated; local-learning models do not import autograd/optimizers and cannot load baseline checkpoints.
- Absolute accuracy, BPTT non-inferiority, and GRU non-inferiority are independent claims and are never merged into one decision.
- P0 non-inferiority means retaining at least 90% of a tuned baseline, not parity or outperformance; accuracy intervals are seed-level statements, not guarantees for every seed.
- Legacy exp03 is a supervised/oracle-warm-start MD upper bound: its cue, gate fit, and recurrent third factor do not satisfy the hidden-context contract, so it cannot support P2.
- A low matrix/tangent rank without improved held-out behavior or prediction cannot support the revised mechanism.
- P0 L1 and L2 budget panels are matched separately; the non-selected norm is diagnostic and no simultaneous dual-norm match is claimed.
- P0 task+homeostasis has one matched task component plus one matched homeostasis component, so its total component budget is twice homeostasis-only; normalization corrections are reported outside those selected component budgets.
- The P0 homeostasis control is yoked inhibitory strengthening, not closed-loop E/I stability evidence; formal normal-perturbation decay, Lyapunov, and closure-error gates remain pending P4.
- P1 cross-parameterization budgets are descriptive and unmatched; physical-rank versus credit-tangent results cannot rank parameterizations by task performance.
- P2 learned-HMM and MD-like gates receive cue observations rather than realized context. Learned-HMM fitting uses legal train-episode batch smoothing, while every held-out belief trajectory is past-only and frozen before truth scoring.
- P2 supervised context inference is an explicitly ineligible upper bound. The oracle filter knows q/h but never receives realized state or switch boundaries.
- P2 q/h cells are paired within seed and then equally averaged; post-fit clamp, delay, and shuffle within-model counterfactuals reuse the intact MD checkpoint and readout. They are not biological causal evidence.
- The P2 MD candidate is specifically past-only two-slice local soft counts with Hebbian lag-1--5 moment shrinkage; it is not evidence for a pure soft-count learner.
- P2_overall is a gate-only belief/effective-control stage gate. It cannot support coupled N=256/N=512 PFC/E/I dynamics, recurrent three-factor credit assignment, or homeostasis.
- P2 energy_proxy_per_trial measures belief confidence and trajectory change, not physical energy consumption; P2i is diagnostic and excluded from P2_overall.
- Nominal feedback dimension is an upper bound on the empirical projected signal span; it is not reported as an automatically realized exact rank.
- PCA, normalization, nuisance regression, subspaces, and dynamics are fit on training trials/blocks only.
- Time points never cross trial/block splits. Symmetric smoothing is visualization-only; predictive likelihood uses causal smoothing/raw counts.
- Inference units are seeds, sessions, or animals. Neurons are never treated as independent replicates.
- IBL latent/behavior lead–lag is descriptive system-level evidence and is not interpreted as biological causal gating.
- Strict IBL neural/shared-dynamics P6 support (distinct from exp11 behavior-only inference) requires a stimulus-pre primary panel with at least 5 animals/20 sessions, explicit unit-QC/context-coverage/nested-CV provenance, hierarchical observations, and parameter counts that include preprocessing.
- Exp13 ARC, Maze, and Sudoku panels are public structured-task hybrid proposal selectors over shared proposal libraries. Their HRM/CTM-inspired mechanisms, selector accuracy, and candidate oracle cannot establish shared neural dynamics, a biological mechanism, or end-to-end computational efficiency.
- The exp13 Sudoku test split is `non_ood`; every Sudoku comparison therefore remains core-ineligible/inconclusive even when its numerical non-inferiority margin is significant.
- Exp15 ARC compares slow/fast family belief with a flat selector on identical candidates and matched charged abstract compute. The proxy is not FLOPs/time/energy, and the 1.2531% candidate coverage fails the 90% claim gate; the zero paired gain is inconclusive rather than support.
- Exp16 is an isolated global-BPTT micro-TRM-like smoke baseline. Its synthetic fixture, three seeds, zero exact accuracy, and pilot-only publisher cannot support an HRM/TRM reproduction, computational advantage, local-learning mechanism, or biological claim.

## External-data status

The referenced Zenodo sequence-memory record currently reports `access_right=restricted`. Missing access is retained as a failed session-level artifact and makes the corresponding claims inconclusive; it is never replaced by synthetic evidence.

## Generated artifacts

- `results/raw_metrics.csv.gz`: lossless compact Exp00--11 raw metric snapshot, including failed and invalid conditions; later scoped tracks retain separate hash-bound raw tables and run manifests. The uncompressed CSV is a reproducible local plotting cache.
- `results/runs.csv`: run status and planned-cell coverage.
- `results/summary.csv`: registered core claims plus explicitly scoped incremental claims.
- `results/exp10_bridge_formal_raw.csv.gz`, `results/exp10_bridge_formal_summary.csv`, and `results/exp10_bridge_formal_run_manifest.csv`: 30-seed N=256 formal bridge rows, seed-macro conclusions, and the clean per-run provenance/hash inventory.
- `results/exp11_ibl_behavior_real_raw.csv.gz` and `results/exp11_ibl_behavior_real_summary.csv`: behavior-only session rows and animal-primary conclusions.
- `results/exp11_ibl_behavior_cohort_{config,manifest,summary}`: frozen public-session selection, exclusions, and dataset provenance; raw trial tables are not published.
- `results/exp13_{arc,maze,sudoku}_formal_{raw,conditions,comparisons,run_manifest,report}`: public structured-task rows, task-primary statistics, provenance binding, and family-scoped interpretation.
- `results/exp14_ibl_multisession_neural_formal_{raw,conditions,comparisons,run_manifest,report}`: hash-bound multi-session neural snapshot and animal-primary inference.
- `results/exp15_arc_matched_formal_{raw,conditions,comparison,run_manifest,report}`: verified-source ARC task rows, registered paired task-primary comparison, and immutable publication bindings.
- `results/exp15_formal_{summary,run_manifest,report}`: reviewed legacy task-specialization snapshot containing the report-only Sudoku engineering audit.
- `results/exp16_tiny_recursive_smoke_3seed_{raw,conditions,comparison,run_manifest,report}`: strict-deterministic, clean-commit, synthetic-fixture smoke snapshot; formal promotion is disabled.
- `results/core_results.pdf`, `results/phase_models.pdf`, `results/hidden_context.pdf`, `results/exp10_bridge_pilot.pdf`, `results/exp10_bridge_formal.pdf`, `results/exp11_ibl_behavior_real.pdf`, `results/exp13_{arc,maze,sudoku}_formal.pdf`, `results/exp14_ibl_multisession_neural_formal.pdf`, `results/exp15_arc_matched_formal.pdf`, and `results/exp16_tiny_recursive_smoke_3seed.pdf`: script-generated data figures when applicable.
