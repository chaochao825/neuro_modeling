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

## Core proposition audit

| Claim | Criterion | n complete/planned | Estimate [95% CI] | Conclusion |
|---|---|---:|---:|---|
| A1_rank_matches_feedback | 95% CI of effective-rank minus 4 lies within [-0.5, 0.5] | 20/20 | -1.812e-05 [-2.059e-05, -1.591e-05] | **support** |
| A2_d4_r2_noninferior_full | one-sided non-inferiority margin is -0.01 latent R2 | 20/20 | 5.038e-06 [-3.905e-06, 1.633e-05] | **support** |
| A3_alignment_is_necessary | aligned latent-R2 advantage CI lower bound is at least 0.10 | 20/20 | 0.9727 [0.9718, 0.9736] | **support** |
| B1_local_reaches_task_threshold | local accuracy >= 0.85 OR >= 90% of paired BPTT (95% CI) | 20/20 | 0.1263 [0.05168, 0.1959] | **support** |
| B2_gate_reduces_switch_cost | no-gate switch cost exceeds gated-local switch cost (paired CI) | 20/20 | -0.01 [-0.065, 0.04] | **inconclusive** |
| B3_homeostasis_stabilizes | removing inhibitory homeostasis increases Jacobian instability | 20/20 | -0.03029 [-0.03384, -0.02665] | **oppose** |
| B4_local_rank_below_full_feedback | local three-factor update rank is lower than full-feedback rank | 20/20 | -0.1266 [-0.5359, 0.3157] | **inconclusive** |
| C1_phase_effect_survives_rate_match | exactly matched in-phase accuracy advantage CI exceeds 0.02 | 20/20 | 0 [0, 0] | **oppose** |
| D1_shared_basis_near_full | retained switching gain >= 0.95 and shared parameters < full | 0/2 | — [—, —] | **inconclusive** |
| D2_unseen_sequence_generalization | shared held-out NLL is below full LDS on unseen combinations | 0/2 | — [—, —] | **inconclusive** |
| E1_ibl_shared_switching | shared improves on common and retains >= 0.95 of full-model gain | 1/2 | — [—, —] | **inconclusive** |
| E2_latent_precedes_behavior_bias | independent-unit bootstrap CI of latent lead is above zero | 1/2 | — [—, —] | **inconclusive** |

### Evidence details

- `A1_rank_matches_feedback` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; p_value awaits full-family Holm adjustment; p_value is Holm-adjusted across all 12 registered claims (raw Wilcoxon p=1.90734863281e-06); bootstrap criterion determines the three-way conclusion
- `A2_d4_r2_noninferior_full` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; p_value awaits full-family Holm adjustment; p_value is Holm-adjusted across all 12 registered claims (raw Wilcoxon p=0.75616645813); bootstrap criterion determines the three-way conclusion
- `A3_alignment_is_necessary` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; p_value awaits full-family Holm adjustment; p_value is Holm-adjusted across all 12 registered claims (raw Wilcoxon p=1.90734863281e-06); bootstrap criterion determines the three-way conclusion
- `B1_local_reaches_task_threshold` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; p_value awaits full-family Holm adjustment; absolute accuracy-minus-0.85 CI [-0.23032, -0.119367]; p_value is Holm-adjusted across all 12 registered claims (raw Wilcoxon p=0.00485992431641); bootstrap criterion determines the three-way conclusion
- `B2_gate_reduces_switch_cost` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; p_value awaits full-family Holm adjustment; p_value is Holm-adjusted across all 12 registered claims (raw Wilcoxon p=1); bootstrap criterion determines the three-way conclusion
- `B3_homeostasis_stabilizes` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; p_value awaits full-family Holm adjustment; p_value is Holm-adjusted across all 12 registered claims (raw Wilcoxon p=1.90734863281e-06); bootstrap criterion determines the three-way conclusion
- `B4_local_rank_below_full_feedback` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; p_value awaits full-family Holm adjustment; p_value is Holm-adjusted across all 12 registered claims (raw Wilcoxon p=0.521673202515); bootstrap criterion determines the three-way conclusion
- `C1_phase_effect_survives_rate_match` (failed=0): paired 95% bootstrap CI at the declared independent-unit level; p_value awaits full-family Holm adjustment; all exact matching flags true and source fingerprints identical within seed; p_value is Holm-adjusted across all 12 registered claims (raw Wilcoxon p=1); bootstrap criterion determines the three-way conclusion
- `D1_shared_basis_near_full` (failed=1): complete common/shared/full formal panel unavailable
- `D2_unseen_sequence_generalization` (failed=1): complete unseen-combination shared/full panel unavailable
- `E1_ibl_shared_switching` (failed=0): requires at least 2 complete independent units; only 1/2 planned units are complete
- `E2_latent_precedes_behavior_bias` (failed=0): requires at least 2 complete independent units; only 1/2 planned units are complete; descriptive temporal association, not causal

## Interpretation safeguards

- BPTT is isolated as a performance baseline; local-learning models do not import autograd or optimizers and cannot load BPTT checkpoints.
- PCA, normalization, nuisance regression, subspaces, and dynamics are fit on training trials/blocks only.
- Time points never cross trial/block splits. Symmetric smoothing is visualization-only; predictive likelihood uses causal smoothing/raw counts.
- Inference units are seeds, sessions, or animals. Neurons are never treated as independent replicates.
- IBL latent/behavior lead–lag is descriptive system-level evidence and is not interpreted as causal gating.

## External-data status

The referenced Zenodo sequence-memory record currently reports `access_right=restricted`. Missing access is retained as a failed session-level artifact and makes the corresponding claims inconclusive; it is never replaced by synthetic evidence.

## Generated artifacts

- `results/raw_metrics.csv`: every raw metric row, including failed and invalid conditions.
- `results/runs.csv`: run status and planned-cell coverage.
- `results/summary.csv`: one row per pre-registered core claim.
- `results/core_results.pdf` and `results/phase_models.pdf`: script-generated data figures when applicable.
