# Exp23-25 formal evidence summary

> **Legacy mixed aggregate.** This file is retained byte-for-byte at the claim
> level because it was the published Exp23--25 report and later received Exp31
> and Exp32 appendices. It is not the current evidence surface. Exp23 and its
> rejected gain-axis rule are classified only as historical. Use
> [`current/README.md`](current/README.md) and
> [`current/claims.csv`](current/claims.csv) for active evidence, or
> [`history/README.md`](history/README.md) and
> [`history/claims.csv`](history/claims.csv) for superseded and failed work.

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

## Exp32 formal update: persistent reward-only control without block reset

The preregistered Exp32-v1 smoke primary failed at hazard 0.05, feedback 1/8
and delay 4: local minus train-fixed was only +0.00352 and only 3/5 seeds were
positive. That failure, all 1,920 rows, and every run receipt remain published;
the v1 formal launch stayed unauthorized.

An independently seeded v2 panel then tested a new feedback--memory-timescale
claim without changing the controller. All 30 seeds and all 10,800 planned
rows completed from clean commit `49aaaf3`. At the frozen slow-switch primary
cell, persistent local control exceeded train-fixed by +0.04349 (95% seed
bootstrap CI +0.03446 to +0.05289; 28/30 positive), exceeded opposite-action
eligibility by +0.08241 (CI +0.07667 to +0.08779), and had a +0.01011 accuracy
response per doubling of expected feedback per dwell (CI +0.00962 to +0.01062).
All three passed their 0.02/0.02/0.005 MCIDs and Holm correction, so the bounded
main-controller claim is **support**.

The stronger phase-diagram claim did not pass. On two exact iso-lambda lines,
the preregistered slow-minus-fast effect was +0.01195 (CI +0.00320 to +0.02089),
below the 0.02 structural MCID with one-sided p=0.955. That layer is
**inconclusive**, so the registered joint Exp32 result is also
**inconclusive**. The opposite-credit intervention is not update-budget
matched (mean L1 ratio 1.215), the controller's scores are action-policy
proxies rather than calibrated context posteriors, and neither an E/I carrier
nor neural data participates in Exp32.

The fixed Exp23 probe further narrows the earlier negative result: delayed
local gain was -0.01133 under matched state displacement but +0.00056 at its
natural scale, while the matching procedure amplified the local axis by a
median 83.4x. Natural-scale delayed BPTT gained +0.01796 (CI +0.01162 to
+0.02444), still below the local endpoint's 0.03 MCID. This opposes only the
tested drive-gain axis, local rule and matching protocol; it does not oppose
local learning in general.

See `results/actuator_matching_critical_audit_20260718.md` for the consolidated
support/oppose/inconclusive ledger, frontier comparison and next scale gate.

## Exp34 corrected formal update: causal motif consensus on ORBIT

The annotation-safe v3 formal panel completed all 5 seeds, 17 official test
users, and 4,250 planned seed-user-task episodes. Three clutter videos with
fewer than 50 valid frames were excluded exactly as required by the official
protocol; no remaining condition failed or was invalid. Algorithmic seeds were
averaged within user before paired inference.

| comparison | user-equal gain | 95% user-bootstrap CI | Holm p | conclusion |
| --- | ---: | ---: | ---: | --- |
| causal minus validation-fixed | +0.02929 | [+0.01549, +0.04368] | 0.001892 | support |
| causal minus memoryless reset | +0.01568 | [+0.00386, +0.02787] | 0.025452 | support |
| causal minus instantaneous majority | +0.02525 | [+0.01500, +0.03619] | 0.000183 | support |
| causal minus eight-frame delay | +0.00657 | [+0.00433, +0.00876] | 0.000275 | support |

The corrected joint task-and-causal-state claim is **support**. Causal
consensus reached 0.7189 user-equal accuracy, retained 53.6% of oracle
headroom, and used no query labels, future frames, autograd, or BPTT. Its
official-style task-video point estimate was 67.43%, essentially tied with the
published EfficientNet-B0 cosine ProtoNet 67.48%; it is not a SOTA claim. The
full four-actuator bank is evaluated, so no efficiency claim is supported.

The overall Exp34 evidence is **mixed** for protocol reasons. A preceding
formal-v2 attempt exposed 15 test users before a missing-user coverage defect
was identified. That result is invalid and retained with hashes. V3 repaired
only official short-video exclusion and strict expected-set checks, but reused
the same public test split. Consequently the corrected within-dataset
mechanism contrast supports, whereas strict untouched prospective
confirmation remains inconclusive pending a new frozen replication.

## Exp35 retrospective update: prefix reliability audit

Exp35 executes the stronger same-tape audit requested by the Exp34 stop rule.
The frozen v2 panel completed all five seeds, all 17 users, and all 19
conditions with zero failed or invalid conditions. Seeds were averaged within
user before paired inference. Validation users alone selected the temporal
retention, the single prefix operator, action temperatures, and stacking
weights; no evaluation label was used by a deployable condition.

| condition | user-equal accuracy |
| --- | ---: |
| equal prefix probability accumulation | **0.81905** |
| prefix prototype probability | 0.81683 |
| validation-calibrated prefix stack | 0.81681 |
| cumulative prefix vote | 0.81581 |
| validation-selected single prefix operator | 0.81336 |
| fixed temporal | 0.76011 |
| prefix consistency | **0.75917** |
| instantaneous majority | 0.70984 |

| registered comparison | consistency-minus-control | 95% user-bootstrap CI | Holm p | conclusion |
| --- | ---: | ---: | ---: | --- |
| lagged consistency | +0.00070 | [+0.00040, +0.00101] | 0.001221 | support, narrow self-inclusion effect |
| cumulative prefix vote | -0.05664 | [-0.09071, -0.02906] | 0.001587 | oppose |
| equal prefix probability | -0.05988 | [-0.09444, -0.03197] | 0.000549 | oppose |
| calibrated prefix stack | -0.05764 | [-0.09247, -0.02992] | 0.001587 | oppose |
| selected single prefix operator | -0.05419 | [-0.08681, -0.02637] | 0.002747 | oppose |
| fixed temporal | -0.00095 | [-0.00833, +0.00728] | 0.819824 | inconclusive |

The comparative routing claim is **oppose**. In the exact stable-correct versus
stable-wrong intervention, both actions have concentration one; the frozen tie
rule selects the wrong action throughout, giving accuracy 0 and wrong-lock
fraction 1. The consistency-as-correctness interpretation is also **oppose**.

An explicitly exploratory decomposition gives prefix-probability gains of
+0.1339, +0.1238, +0.1393, and +0.0510 for prototype, gain, delta, and temporal
respectively. The equal prefix bank exceeds the validation-selected single
prefix operator by only +0.0057, with a 95% interval [-0.0043, +0.0161]. Thus
the surviving signal is temporal evidence accumulation within constant-object
videos; heterogeneous-bank value remains inconclusive.

Exp35 is retrospective because the ORBIT test split had already been exposed.
No positive observation can upgrade a claim. The result supersedes Exp34 as
the paper-level verdict, moves Exp34 to historical motivation, and triggers the
registered stop rule against adding HMM, Hedge, GRU, sparse execution, or E/I
to rescue the router. The canonical package is
`results/exp35_prefix_reliability_audit_retrospective_v2/`.

## Exp36 prospective validity update: ORBIT-India schema incompatibility

Exp36-v1 was frozen before ORBIT-India download and completed all five planned
seed processes, but the registered inferential panel is invalid. Only 4 of 12
collectors could instantiate every four-class clean/clutter task. The other
eight collectors generated exactly 6,400 retained failed condition cells per
seed (32,000 total): some collectors had fewer than four classes and others
contained at least one object without the required clean/clutter pair.

This determination is outcome-blind. The validity audit parsed only status,
error, and schema fields; it did not inspect accuracy, post-switch accuracy, or
detector-utility fields. The four surviving collectors were deliberately not
summarized as a prospective result. Therefore the Exp36 change-aware prefix
claim is **inconclusive**, v1 is historical-only, and the failure is evidence
about dataset--protocol compatibility rather than evidence for or against the
controller.

The canonical audit is `results/exp36_v1_invalid_schema_audit/`. Its successor,
Exp37 freezes a balanced CORe50 session/object schema and a Bayesian online
change-point controller before data acquisition completes.

## Exp37 prospective update: change-aware prefix accumulation on CORe50

Exp37 was frozen before the CORe50 archive completed downloading, before image
extraction or embedding, and before any task outcome was available. The
acquired archive matched the published MD5 and the frozen byte length; the
outcome-blind schema audit found exactly 11 sessions, 50 objects, 550
session/object cells, and 164,866 images. Frozen EfficientNet-B0 features then
completed all 550 cells with zero failures.

The external panel contains all five seeds, all nine held-out sessions
(`s3`--`s11`), 50 tasks per session, two panels, and nine conditions: 40,500
registered condition cells with no failures or exclusions. Seeds, tasks,
objects, and frames were averaged inside session before inference.

| registered comparison | session-mean difference | 95% session-bootstrap CI | Holm p | conclusion |
| --- | ---: | ---: | ---: | --- |
| hard-reset BOCPD minus cumulative, hidden switch | +0.0000 | [+0.0000, +0.0000] | 1.000000 | inconclusive |
| hard-reset BOCPD minus selected fixed forgetting | -0.5329 | [-0.5432, -0.5220] | 0.015625 | oppose |
| hard-reset BOCPD minus selected sliding window | -0.5238 | [-0.5334, -0.5141] | 0.015625 | oppose |
| hard-reset BOCPD minus cumulative, natural | +0.0000 | [+0.0000, +0.0000] | 1.000000 | noninferiority gate passes |

Hard-reset BOCPD reached 0.3984 hidden-switch accuracy, exactly matching
unbounded cumulative accumulation because it never reset. Validation-selected
retention zero (current-frame evidence) reached 0.9313, the two-frame sliding
window reached 0.9221, and oracle change reset reached 0.9528. The registered
joint conclusion is therefore **oppose**, and the frozen stop rule against
further tuning this controller on CORe50 is triggered.

A separately labeled post-hoc, development-only diagnostic explains the
failure without changing the verdict. Across 48,000 `s2` hidden-stream frames,
the maximum change posterior was 0.008529; the minimum frozen alarm threshold
was 0.2, so no candidate in the 144-point BOCPD grid could alarm. This is a
threshold/model-scale defect in the tested detector and means Exp37 does not
refute all possible change-point methods. It does refute the registered claim
that this BOCPD hard-reset controller adds decision utility beyond simple
forgetting on this task.

The canonical 61-file package plus the non-overwriting interpretive figure
amendment is `results/exp37_core50_change_aware_prefix_confirmation/`; both
artifact SHA256 manifests are retained. The result is not an official CORe50
continual-learning, SOTA, biological, or general concept-drift claim.

## Exp38 prospective update: continuous memory control on Stream-51

Exp38 froze the official 11.34 GB Stream-51 archive identity, source-video
split, EfficientNet-B0 representation, support-only vMF evidence model,
development selection, five assembly seeds, and four conjunctive readiness
gates before any task outcome. The complete archive CRC passed. The final
support/development cache contains 755 source videos and 34,250 finite
1,280-dimensional frame embeddings with zero failed videos and zero external
rows. A reference-loader bbox schema failure and a later manifest-writer race
were retained as separate logs and repaired before qualification; neither
exposed classification outcomes.

The registered joint qualification is **oppose**: 0/5 seeds passed. Oracle
cumulative memory with perfect reset exceeded the stronger fixed time scale
in 5/5 seeds (headroom 0.0273--0.0558), and cumulative post-switch harm passed
in 5/5 (0.7115--0.7356). Stable accumulation cleared the 0.02 MCID in only
2/5 seeds; the other three reached 0.01952. Causal risk AUC was 0.746--0.831,
but the operational reachability gate passed only 1/5 because recall was
0.172--0.359 and one seed also exceeded the false-alarm ceiling.

On the disjoint qualification holdout, soft retention minus the stronger fixed
forgetting/window baseline averaged only +0.0019 across assembly seeds (range
-0.0038 to +0.0077), descriptively far below the external MCID of 0.02. This
does not replace the untouched external endpoint. Because the frozen all-seed
gate failed, all 381 external videos remain unfeaturized and unscored; the main
external utility claim is **inconclusive**. The stop rule prohibits post-hoc
threshold tuning or GRU/BPTT, E/I, and encoder rescue. The canonical package is
`results/exp38_stream51_soft_memory_prospective_v1/`.

## Exp38 post-hoc diagnostic and historical disposition

The attachment-motivated diagnostic tested whether the failed scalar soft
memory controller had nevertheless exposed a reusable uncertainty signal. It
is explicitly post-hoc and cannot rescue Exp38. Directly selected scalar
retention improved qualification NLL by only +0.003813 nats (5/5 seeds), below
the frozen +0.005 reuse threshold. Replacing the causal controller with a
likelihood HMM reduced NLL by -0.066971 nats relative to the selected fixed
memory (0/5 seeds positive). Adding cumulative log evidence to the three
instantaneous risk statistics increased oracle write-harm AUC by only
+0.001364 (0.700658 to 0.702022).

The diagnostic verdict is therefore **oppose** for both scalar-controller
reuse routes. Stream-51 is retired as a test bed for factorized uncertainty:
its semantic splice construction does not independently identify abrupt jump
hazard, gradual process drift, and observation noise. The complete Exp35--38
decision trail, including failed launches and supersession rules, is archived
in `results/history/adaptive_memory_exp35_exp38_20260726.md`; the raw diagnostic
package is `results/history/exp38_factorized_memory_diagnostic_v1/`.

## Exp39 prospective update: factorized uncertainty composition

Exp39 separated the successor question from Stream-51 using a synthetic jump
diffusion in which hazard, process variance, and observation variance are
orthogonally controlled. Fitting used only the baseline and three single-factor
cells (`000`, `100`, `010`, and `001`). Pairwise and triple cells (`110`,
`101`, `011`, and `111`) were untouched until the 30-seed formal run. The
controller maintains three causal, factor-indexed local states and performs no
test-time gradient update or BPTT. It was compared with a selected fixed
filter and an IMM containing only the four fitting modes; an eight-mode IMM
and a dynamic truth filter were retained as privileged upper bounds.

| Method | Mean held-out predictive NLL |
|---|---:|
| Dynamic truth filter | 0.724684 |
| Eight-mode oracle-supported IMM | 0.764398 |
| Factorized controller | 0.841257 |
| Four seen-mode IMM | 0.889265 |
| Selected fixed filter | 1.131837 |

The registered joint gate is **support**. Relative to the selected fixed
filter, the NLL gain was +0.290580 nats with a 95% seed-bootstrap interval of
[0.280122, 0.300594], positive in 30/30 seeds. Relative to the seen-mode IMM,
the gain was +0.048008 [0.042136, 0.054094], also positive in 30/30 seeds.
All five registered utility and clamp tests passed their Holm-corrected family.
High-minus-low clamp selectivity was +0.037097, +0.026683, and +0.052370 nats
for the h-, Q-, and R-indexed states, respectively.

This supports only a functional compositional statement: separate causal
states can reuse fitting-time evidence across unseen uncertainty combinations
more effectively than selecting among joint modes observed during fitting.
It does not establish calibrated recovery of the generating parameters.
Block-level log-parameter correlations were 0.089, 0.312, and 0.868 for h, Q,
and R, so parameter identification is **oppose**. Performance was not uniform:
the factorized controller did not beat the seen-mode IMM in held-out cell
`110`. It was also worse immediately after switches (NLL 1.059377 versus
0.972679), although better late in blocks (0.796637 versus 0.878553). Claims of
faster switching, universal cell-wise dominance, or oracle equivalence are
therefore rejected.

The independently replayed package contains 30/30 completed seeds, 15,360
block rows, zero failed conditions, paired-tape verification, selection replay,
summary replay, and SHA-256 manifests. The canonical report is
`results/exp39_factorized_uncertainty_prospective_v1/report.md`; bounded
post-outcome interpretation is in its `critical_analysis.md` companion.

## Exp40 post-hoc IBL state-utility audit

Exp40 used the existing outcome-exposed 30-session/30-animal IBL cohort only as
a development gate. It did not port the synthetic \((h,Q,R)\) labels literally:
IBL does not independently manipulate a time-varying sensory-noise parameter.
Instead, a task-structured causal observer exposed prior log odds, posterior
recent-change probability, and run-length concentration. It received only past
stimulus sides; `probabilityLeft` remained evaluation-only. Whole blocks were
split chronologically, and every scaler/readout was fitted inside train/dev.

All 210 planned cells were retained. Twenty-seven animals formed a complete
seven-condition endpoint panel; three failed every condition symmetrically
because their test folds contained fewer than eight low-contrast choices. The
semi-Markov state improved context NLL over the learned HMM by +0.071391
nats/trial [0.049444, 0.095949], so structured block decoding is **support** at
the post-hoc development tier.

That decoding gain did not produce behavioral utility. Dev-selected baseline
minus factorized-state low-contrast NLL was -0.010723 [-0.022459, 0.001010],
positive in 9/27 animals. Any positive gain is **inconclusive**, while the
0.005-nat meaningful-utility claim is **oppose** after Holm correction. Release
clamp harm was +0.001786 [-0.004956, 0.009137] (**inconclusive**); precision
clamp harm was -0.005114 [-0.009175, -0.001242] (**oppose**).

A single result-revealed assay probe selected regularization on all dev trials
and reduced readout variance, but its factorized gain remained negative
(-0.003617). It cannot replace the registered development result. The disjoint
IBL cohort was therefore neither frozen nor opened, and neural analysis remains
locked. Confirmatory real-data evidence is **inconclusive/not run**. The report,
animal effects, failure rows, figure, and run hashes are preserved under
`results/exp40_ibl_state_utility_*`; the full lineage record is
`results/history/exp40_ibl_factorized_state_development_20260726.md`.
