# Historical evidence view

This directory is the only presentation surface for superseded, rejected,
abandoned, or exploratory proposals. Conclusions are preserved exactly at the
experiment level: a once-positive result can remain `support` while its
disposition is `historical_only`. It must not be promoted into the current
method claim.

The code entry points remain in `experiments/` for reproducibility. Branch
snapshots preserve each ancestor branch's README, report, and summary without
rewriting their original claims. `snapshot_manifest.csv` binds those files by
SHA-256. No failed or negative result was deleted.
The only ancestor-tip result absent from the current tree is indexed
in `git_objects.csv` and materialized as a compressed historical archive
whose decompressed bytes are checked against the original Git blob SHA.
`branch_reachability.csv` is the executable audit receipt showing that
every deleted branch tip contributes zero commits outside consolidated main.

`claims.csv` retains every historical row found in the legacy mixed
aggregate, including all failed Exp23 controller rows.
The pre-consolidation critical audit is preserved as
`actuator_matching_critical_audit_20260718.md`; the current copy omits
the rejected Exp23 mechanism and abandoned Exp32-v1 configuration.
The complete pre-consolidation project narrative and reproduction
commands are preserved as `project_README_pre_consolidation.md`.

| Experiment | Track | Disposition | Conclusion | Successor | Evidence |
|---|---|---|---|---|---|
| exp00 Local predictive fixed point | physical_low_rank | `historical_only` | **support** | exp08 | [project_README.md](../history/branch_snapshots/real-lowdim-validation/project_README.md) |
| exp01 Feedback dimension and geometry sweep | physical_low_rank | `historical_only` | **support** | exp08 | [report.md](../history/branch_snapshots/real-lowdim-validation/report.md) |
| exp02 Oracle context E/I gate | legacy_context_gate | `historical_only` | **mixed** | exp09 | [report.md](../history/branch_snapshots/real-lowdim-validation/report.md) |
| exp03 Supervised Hebbian context gate | legacy_context_gate | `historical_only` | **mixed** | exp09 | [report.md](../history/branch_snapshots/real-lowdim-validation/report.md) |
| exp04 Rate-matched phase gate | phase_gate | `historical_only` | **oppose** | none | [report.md](../history/branch_snapshots/real-lowdim-validation/report.md) |
| exp05 Public sequence-memory shared dynamics | legacy_real_data | `historical_only` | **inconclusive** | exp25 | [report.md](../history/branch_snapshots/real-lowdim-validation/report.md) |
| exp06 IBL context-switch pilot | legacy_real_data | `historical_only` | **inconclusive** | exp11 | [report.md](../history/branch_snapshots/real-lowdim-validation/report.md) |
| exp07 Paired mechanism-identifiability audit | credit_assignment | `historical_only` | **support** | exp08 | [report.md](../history/branch_snapshots/effective-control-p0/report.md) |
| exp12 Structured routing interface | reasoning_exploration | `historical_only` | **inconclusive** | exp13 | [exp12_structured_routing.py](../../experiments/exp12_structured_routing.py) |
| exp13 ARC maze and Sudoku structured reasoning | reasoning_exploration | `historical_only` | **mixed** | none | [exp13_arc_formal_report.md](../exp13_arc_formal_report.md) |
| exp14 IBL multi-session neural audit | legacy_real_data | `historical_only` | **inconclusive** | exp25 | [exp14_ibl_multisession_neural_formal_report.md](../exp14_ibl_multisession_neural_formal_report.md) |
| exp15 Task-specialized ARC and Sudoku adapters | reasoning_exploration | `historical_only` | **inconclusive** | none | [exp15_formal_report.md](../exp15_formal_report.md) |
| exp16 Tiny recursive Sudoku baseline | reasoning_exploration | `historical_only` | **inconclusive** | none | [exp16_tiny_recursive_retry_3seed_report.md](../exp16_tiny_recursive_retry_3seed_report.md) |
| exp17 Tiny recursive calibration | reasoning_exploration | `historical_only` | **inconclusive** | none | [exp17_wichtounet_3seed_report.md](../exp17_wichtounet_3seed_report.md) |
| exp18 ARC recursive canary baseline | reasoning_exploration | `historical_only` | **inconclusive** | none | [exp18_arc1_canary_seed3000_20task_report.md](../exp18_arc1_canary_seed3000_20task_report.md) |
| exp19 Belief-controlled E/I checkpoint audit | belief_ei | `historical_only` | **mixed** | exp21 | [exp19_belief_ei_effective_dynamics_formal_report.md](../exp19_belief_ei_effective_dynamics_formal_report.md) |
| exp20 IBL belief-conditioned neural prediction | legacy_real_data | `historical_only` | **inconclusive** | exp25 | [exp20_ibl_md_belief_dynamics_formal_report.md](../exp20_ibl_md_belief_dynamics_formal_report.md) |
| exp22 Off-policy local gain-axis proposals | controller_learning | `historical_only` | **mixed** | exp23 | [exp22_hidden_context_local_gain_axis_formal_report.md](../exp22_hidden_context_local_gain_axis_formal_report.md) |
| exp23 Closed-loop local population-gain controller | controller_learning | `historical_only` | **oppose** | exp24 | [report.md](../exp23_failure_probe/report.md) |
| exp27 Descriptor actuator selector development panel | selector_learning | `historical_only` | **support** | exp29 | [report.md](../exp27_actuator_selector_formal_v1_c445191/formal/report.md) |
| exp28 Independent selector sensitivity and source amendment | selector_learning | `historical_only` | **inconclusive** | exp29 | [report.md](../exp28_selector_sensitivity_v2_07b5ce2/summary/report.md) |
| exp30 Associative actuator trend panel | associative_control | `historical_only` | **inconclusive** | exp31 | [report.md](../exp30_associative_actuator_trend_smoke_v1/report.md) |
| exp33 ORBIT reward-only streaming selector | real_task | `historical_only` | **inconclusive** | exp34 | [report.md](../exp33_orbit_streaming_fewshot_smoke_v1_failed/report.md) |

Generated by `scripts/build_evidence_views.py` from the provenance registry.
