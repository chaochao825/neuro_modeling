# Local Plasticity to Gated Low-Dimensional Dynamics

This report is generated from immutable run artifacts. Failed and invalid conditions are included; only formal-profile independent units can support or oppose a core claim.

## Run coverage (all immutable attempts)

Retries and interrupted attempts remain listed here. These are attempt counts, not unique-seed coverage; core-claim sample sizes use only the latest formal attempt for each experiment and seed.

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

## Core proposition audit

| Claim | Criterion | n complete/planned | Estimate [95% CI] | Conclusion |
|---|---|---:|---:|---|
| A1_rank_matches_feedback | 95% CI of effective-rank minus 4 lies within [-0.5, 0.5] | 20/20 | -1.812e-05 [-2.059e-05, -1.591e-05] | **support** |
| A2_d4_r2_noninferior_full | one-sided non-inferiority margin is -0.01 latent R2 | 20/20 | 5.038e-06 [-3.905e-06, 1.633e-05] | **support** |
| A3_alignment_is_necessary | aligned latent-R2 advantage CI lower bound is at least 0.10 | 20/20 | 0.9727 [0.9718, 0.9736] | **support** |
| B1a_local_absolute_accuracy | local absolute accuracy-minus-0.85 CI is reported independently | 20/20 | -0.1747 [-0.2303, -0.1194] | **oppose** |
| B1b_local_relative_noninferiority | local relative non-inferiority to 90% of paired BPTT is reported independently of absolute performance | 20/20 | 0.1263 [0.05168, 0.1959] | **support** |
| B2_gate_reduces_switch_cost | hidden-context gate improves switch cost without true-context access in the cue, gate fit/test, oracle warm start, or recurrent third factor | 0/20 | — [—, —] | **inconclusive** |
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

### Evidence details

- `A1_rank_matches_feedback` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; two-one-sided margin tests awaits full-family Holm adjustment; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=9.53674316406e-07); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `A2_d4_r2_noninferior_full` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=9.53674316406e-07); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `A3_alignment_is_necessary` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=9.53674316406e-07); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `B1a_local_absolute_accuracy` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided oppose-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=6.99728965543e-05); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `B1b_local_relative_noninferiority` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=0.0024299621582); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `B2_gate_reduces_switch_cost` (failed=0): legacy exp02/03 lacks leakage-free hidden-context provenance; exp03 is a supervised/oracle-warm-start upper bound and the legacy no-gate third factor receives true context
- `B3_homeostasis_stabilizes` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided oppose-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=9.53674316406e-07); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `B4_local_rank_below_full_feedback` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; two-sided zero-difference diagnostic awaits full-family Holm adjustment; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=0.521673202515); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `C1_phase_effect_survives_rate_match` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided oppose-margin test awaits full-family Holm adjustment; all exact matching flags true and source fingerprints identical within seed; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=3.87210821552e-06); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `D1_shared_basis_near_full` (failed=1): complete common/shared/full formal panel unavailable
- `D2_unseen_sequence_generalization` (failed=1): complete unseen-combination shared/full panel unavailable
- `E1_ibl_shared_switching` (failed=1): strict eligible cohort has 0 animals/0 sessions (minimum 5/20); missing provenance field hierarchical_observation_model; missing provenance field nested_cv_latent_dimension; missing provenance field unit_qc_applied; missing provenance field context_coverage_valid; missing provenance field parameter_count_includes_preprocessing; missing provenance field hidden_context_inference; missing provenance field test_context_observed; missing provenance field belief_filter_used_true_block_boundaries; condition_schedule_observed is not uniformly False; parameter_count changes across folds or is missing; complete common/shared/full stimulus-pre panel unavailable
- `E2_latent_precedes_behavior_bias` (failed=0): hierarchical_observation_model is not uniformly True; nested_cv_latent_dimension is not uniformly True; unit_qc_applied is not uniformly True; context_coverage_valid is not uniformly True; parameter_count_includes_preprocessing is not uniformly True; hidden_context_inference is not uniformly True; test_context_observed is not uniformly False; belief_filter_used_true_block_boundaries is not uniformly False; behavior_bias_used_true_block_boundaries is not uniformly False; lead records do not exactly match the strict E1 session/animal cohort; the strict E1 model cohort or method provenance is invalid
- `P0a_aligned_task_improves_prediction_vs_frozen` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.0024310070015474325, CI=[0.0012379993876758281, 0.003979095880244444], raw_p=5.304813385009766e-06; l2: conclusion=support, n=30, estimate=0.004107516535009095, CI=[0.002218271399833035, 0.00654866282385085], raw_p=4.610046744346619e-06; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=5.30481338501e-06); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0b_aligned_task_beats_shuffled` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.001211568177126122, CI=[0.000680089378008703, 0.0018322525995008252], raw_p=2.9867514967918396e-06; l2: conclusion=support, n=30, estimate=0.0020118299553517874, CI=[0.0011746631922471406, 0.002994505864573219], raw_p=1.6195699572563171e-06; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=2.98675149679e-06); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0c_aligned_adds_value_over_matched_homeostasis` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.0024354150219953246, CI=[0.001248913662966034, 0.004012353270735189], raw_p=3.997236490249634e-06; l2: conclusion=support, n=30, estimate=0.004104866967973079, CI=[0.002230666236489589, 0.0065737887957078266], raw_p=2.9867514967918396e-06; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=3.99723649025e-06); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0d_local_absolute_accuracy` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.056666666666666685, CI=[0.02997916666666677, 0.08166666666666669], raw_p=0.00028261244544550504; l2: conclusion=support, n=30, estimate=0.05666666666666667, CI=[0.02916666666666669, 0.08166666666666669], raw_p=0.000305714097041996; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=0.000305714097042); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0e_local_noninferior_tuned_bptt` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.07491666666666666, CI=[0.034581249999999994, 0.11733958333333325], raw_p=0.00047877202703846004; l2: conclusion=support, n=30, estimate=0.07491666666666666, CI=[0.03491250000000002, 0.11742291666666656], raw_p=0.00047895487883489285; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=0.000478954878835); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0f_local_noninferior_tuned_gru` (failed=0): intersection-union across separately matched L1/L2 panels; raw joint p=max(panel p) awaits full-family Holm adjustment; panel audit: l1: conclusion=support, n=30, estimate=0.05841666666666667, CI=[0.021916666666666685, 0.09916874999999999], raw_p=0.003719669399051919; l2: conclusion=support, n=30, estimate=0.05841666666666667, CI=[0.022831250000000032, 0.09733541666666666], raw_p=0.003244611805976201; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=0.00371966939905); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P1a_masked_outer_product_identity` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=2.16023152891e-08); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P1b_credit_tangent_respects_feedback_bound` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=2.16023152891e-08); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P1c_highrank_physical_update_coexists_with_lowdim_credit` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; one-sided support-margin test awaits full-family Holm adjustment; p_value is Holm-adjusted across all 22 registered claims (raw Wilcoxon p=2.16023152891e-08); a directional bootstrap criterion can support/oppose only when Holm p<=0.05
- `P0_overall` (failed=0): non-inferential stage gate; P0a_aligned_task_improves_prediction_vs_frozen=support; P0b_aligned_task_beats_shuffled=support; P0c_aligned_adds_value_over_matched_homeostasis=support; P0d_local_absolute_accuracy=support; P0e_local_noninferior_tuned_bptt=support; P0f_local_noninferior_tuned_gru=support

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
- Nominal feedback dimension is an upper bound on the empirical projected signal span; it is not reported as an automatically realized exact rank.
- PCA, normalization, nuisance regression, subspaces, and dynamics are fit on training trials/blocks only.
- Time points never cross trial/block splits. Symmetric smoothing is visualization-only; predictive likelihood uses causal smoothing/raw counts.
- Inference units are seeds, sessions, or animals. Neurons are never treated as independent replicates.
- IBL latent/behavior lead–lag is descriptive system-level evidence and is not interpreted as causal gating.
- IBL support requires a stimulus-pre primary panel with at least 5 animals/20 sessions, explicit unit-QC/context-coverage/nested-CV provenance, hierarchical observations, and parameter counts that include preprocessing.

## External-data status

The referenced Zenodo sequence-memory record currently reports `access_right=restricted`. Missing access is retained as a failed session-level artifact and makes the corresponding claims inconclusive; it is never replaced by synthetic evidence.

## Generated artifacts

- `results/raw_metrics.csv.gz`: lossless raw metric snapshot, including failed and invalid conditions; the uncompressed CSV is a reproducible local plotting cache.
- `results/runs.csv`: run status and planned-cell coverage.
- `results/summary.csv`: one row per pre-registered core claim.
- `results/core_results.pdf` and `results/phase_models.pdf`: script-generated data figures when applicable.
