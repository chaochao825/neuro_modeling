# Exp23-25 formal evidence summary

This report is fail-closed. It reads only attempts whose saved `config.json` declares `profile=formal`; smoke and pilot attempts are ignored even when their numerical metrics are favorable.

Every registered condition is represented in `summary.csv`. Missing, failed, and invalid cells are retained and prevent formal support in the affected AND gate.

All formal joint claims use AND, never OR.
Exp23 and Exp24 component inference use Holm correction; Exp23 formal readiness requires frozen-recurrent hash/copy receipts.

## Core conclusions

| claim_id | stats_unit | n_planned | n_complete | n_failed | n_invalid | n_sessions | conclusion | criterion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp23_joint_both_tasks | seed | 30 | 30 | 0 | 0 |  | oppose | both task-specific Holm-corrected IUT claims must support (AND) |
| exp24_joint_task_dependent_actuator_specialization | seed | 30 | 30 | 0 | 0 |  | support | all four Holm-corrected direction claims must support (intersection-union AND) |
| exp25_joint_reusable_shared_belief_dynamics | animal (sessions nested) | 2 | 0 | 2 | 0 | 0 | inconclusive | every Exp25 component must support (AND) |

## Component claims

| claim_id | scope | comparison | estimate | ci_low | ci_high | threshold | p_value | p_adjusted | multiplicity_method | n_complete | n_sessions | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp23_current_fraction_of_bptt_gain | current | (local - frozen) / (BPTT-axis - frozen) | -1.57423 |  |  | 0.6 |  |  | Holm within the four registered components for this task | 30 |  | inconclusive |
| exp23_current_gain_vs_frozen | current | local_eprop - frozen held-out balanced accuracy | -0.00125944 | -0.00862349 | 0.00586396 | 0.03 | 1 | 1 | Holm within the four registered components for this task | 30 |  | oppose |
| exp23_current_gain_vs_random | current | local_eprop - random_update held-out balanced accuracy | 0.000922474 | -0.00756733 | 0.00984358 | 0.03 | 1 | 1 | Holm within the four registered components for this task | 30 |  | oppose |
| exp23_current_joint_closed_loop_local_controller | current | all four Exp23 registered local-controller criteria |  |  |  |  |  |  | intersection-union AND over four Holm-corrected components | 30 |  | oppose |
| exp23_current_median_update_cosine | current | local update cosine with exact forward sensitivity | 0.742116 | 0.68334 | 0.758098 | 0 | 9.31323e-10 | 3.72529e-09 | Holm within the four registered components for this task | 30 |  | support |
| exp23_delayed_fraction_of_bptt_gain | delayed | (local - frozen) / (BPTT-axis - frozen) | -2.80475 |  |  | 0.6 |  |  | Holm within the four registered components for this task | 30 |  | inconclusive |
| exp23_delayed_gain_vs_frozen | delayed | local_eprop - frozen held-out balanced accuracy | -0.0113327 | -0.0229988 | -0.00158255 | 0.03 | 1 | 1 | Holm within the four registered components for this task | 30 |  | oppose |
| exp23_delayed_gain_vs_random | delayed | local_eprop - random_update held-out balanced accuracy | -0.0101892 | -0.0226971 | 0.000926516 | 0.03 | 1 | 1 | Holm within the four registered components for this task | 30 |  | oppose |
| exp23_delayed_joint_closed_loop_local_controller | delayed | all four Exp23 registered local-controller criteria |  |  |  |  |  |  | intersection-union AND over four Holm-corrected components | 30 |  | oppose |
| exp23_delayed_median_update_cosine | delayed | local update cosine with exact forward sensitivity | 0.392427 | 0.317478 | 0.467238 | 0 | 9.31323e-10 | 3.72529e-09 | Holm within the four registered components for this task | 30 |  | support |
| exp24_dynamics_prefers_low_rank_to_routing | dynamics_dominant | low_rank - routing | 0.185382 | 0.176921 | 0.193759 | 0 | 0.00019996 | 0.00079984 | Holm across the four registered Exp24 actuator comparisons | 30 |  | support |
| exp24_dynamics_prefers_rgl_to_routing | dynamics_dominant | rgl - routing | 0.128947 | 0.120678 | 0.137344 | 0 | 0.00019996 | 0.00079984 | Holm across the four registered Exp24 actuator comparisons | 30 |  | support |
| exp24_routing_prefers_gain_to_low_rank | routing_dominant | gain - low_rank | 0.13083 | 0.120767 | 0.140245 | 0 | 0.00019996 | 0.00079984 | Holm across the four registered Exp24 actuator comparisons | 30 |  | support |
| exp24_routing_prefers_routing_to_low_rank | routing_dominant | routing - low_rank | 0.130564 | 0.1206 | 0.139951 | 0 | 0.00019996 | 0.00079984 | Holm across the four registered Exp24 actuator comparisons | 30 |  | support |
| exp25_cross_session_fully_vs_common | cross-session-transfer | fully-gated - common held-out mean log likelihood in cross-session transfer |  |  |  | 0 |  |  |  | 0 | 0 | inconclusive |
| exp25_fully_gated_vs_common | implemented_outer_protocols | fully-gated - common held-out mean log likelihood |  |  |  | 0 |  |  |  | 0 | 0 | inconclusive |
| exp25_fully_retains_90pct_separate_gain | implemented_outer_protocols | (fully-common) - 0.9 * (separate-task-common) held-out gain |  |  |  | 0 |  |  |  | 0 | 0 | inconclusive |
| exp25_fully_uses_fewer_parameters | implemented_outer_protocols | fully-gated - separate-task parameter count |  |  |  | 0 |  |  |  | 0 | 0 | inconclusive |
| exp25_input_gain_exceeds_state_gain | implemented_outer_protocols | input-gated - state-gated held-out mean log likelihood |  |  |  | 0 |  |  |  | 0 | 0 | inconclusive |
| exp25_unseen_composition_shared_vs_separate | unseen-stimulus-action-composition | fully-gated - separate-task held-out mean log likelihood on unseen composition |  |  |  | 0 |  |  |  | 0 | 0 | inconclusive |

## Retained failed, invalid, or missing conditions

| experiment | scope | condition | status | n_units | unit_ids | note |
| --- | --- | --- | --- | --- | --- | --- |
| exp25_compositional_tasks_real |  | cross-session-transfer:common | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | cross-session-transfer:fully-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | cross-session-transfer:input-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | cross-session-transfer:separate-task | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | cross-session-transfer:state-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | leave-one-block-out:common | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | leave-one-block-out:fully-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | leave-one-block-out:input-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | leave-one-block-out:separate-task | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | leave-one-block-out:state-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | leave-one-composition-out:common | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | leave-one-composition-out:fully-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | leave-one-composition-out:input-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | leave-one-composition-out:separate-task | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | leave-one-composition-out:state-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | unseen-stimulus-action-composition:common | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | unseen-stimulus-action-composition:fully-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | unseen-stimulus-action-composition:input-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | unseen-stimulus-action-composition:separate-task | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |
| exp25_compositional_tasks_real |  | unseen-stimulus-action-composition:state-gated | failed | 1 | seed:0 | Exp25 official/canonical data validation failed closed; no synthetic or inferred-schema substitute was used: RuntimeError: official Figshare source bytes were verified, but canonical trial-level neural counts are absent. Exp25 requires a reviewed hash-pinned trials.csv, units.csv, session NPZ bundle, conversion file, and manifest; missing /home/spco/sow_linear/neuro_modeling_exp21_exp22_771526e/local_plasticity_gated_dynamics/data/compositional_tasks/official_canonical_v1/canonical_manifest.json |

## Interpretation boundary

Exp23 and Exp24 use seed as the independent unit. Exp25 first normalizes likelihood within held-
out session, then averages sessions within animal and bootstraps animals; neuron and time bin
are never replicates. Exp24 is an oracle actuator-isolation benchmark and does not itself
establish local controller learning. Exp25 scores exact one-step conditional Poisson likelihood
rather than a full marginal PLDS likelihood or autonomous forecast. A currently invalid cross-
session transfer condition cannot support the real-data joint claim. Exp23/24 mean-effect
p-values use one-sided paired sign-flip tests, the median-cosine component uses an exact sign
test, and all four use Holm correction within each task family; task and cross-task conclusions
are conservative intersection-union AND gates, never OR. Exp23 formal readiness also requires
explicit pairing IDs, frozen-recurrent hash/copy receipts, train/dev/test separation, no true-
context access, and local-eprop no-autograd/BPTT receipts for every registered seed. The Exp23
formal-v2 conclusion is limited to the registered matched state-displacement budget of 0.001 and
the implemented local-eprop rule; it does not reject all budgets or all local rules. That fixed
target was selected without behavior, loss, test, or OOD fields from the retained v1
development-reachability receipt. Exact config matching excludes the superseded 0.002 attempts
from v2 inference while leaving their raw artifacts intact.

## Exp31 formal update: hidden demand with executed-reward-only learning

Exp31 removes the strongest Exp30 shortcuts: the target is one queried random
value rather than an explicit routing/memory mixture; dense keys create natural
capacity limits; no mode-by-demand gain is fitted; and the local selector sees
only the scalar reward from the actuator it executed. The primary endpoint
charges the complete forced-exploration prefix.

| claim | estimate | 95% seed-bootstrap CI | Holm p | conclusion |
| --- | ---: | ---: | ---: | --- |
| reward-only local minus train-fixed full-block accuracy | +0.04721 | [0.04591, 0.04853] | 0.000040 | support |
| hidden-reliability crossover | +0.39609 | [0.38594, 0.40608] | 0.000040 | support |
| associative minus identical-write query-shuffled | +0.34670 | [0.34010, 0.35330] | 0.000040 | support |
| 25% oracle-gain retention margin | +0.02229 | [0.02122, 0.02338] | 0.000040 | support |

All 30 formal seeds and all 22,680 registered condition rows completed. Mean
selector choice accuracy was 0.9497; mean oracle gain retained was 0.4732;
query-shuffled accuracy was 0.4993; and associative accuracy decreased with
interference pressure (mean seed Spearman -0.9708).

The joint Exp31 conclusion is **support**, but only for a synthetic two-actuator
controller-identifiability claim. Exp31 contains no participating high-rank E/I
carrier, no neural recordings, and no strong task-model baseline. The
controller also receives labels on 64/128 trials, resets at every block, and
selects between only two fixed motifs. It therefore
does not yet support the full Actuator Matching Principle. The next
high-information experiment is to place the frozen motifs and reward-only
controller inside a genuinely participating stable E/I carrier, then test
closure, normal perturbation decay, and held-out utility before scaling neuron
count or moving to real block-switching data.
