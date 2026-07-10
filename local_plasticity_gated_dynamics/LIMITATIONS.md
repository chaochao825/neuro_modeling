# Known limitations of the current formal results

The committed summary is an immutable snapshot of the completed runs, not a claim that every mechanism is identified.

This monorepo carries the compact evidence snapshot (`raw_metrics.csv`,
`runs.csv`, `summary.csv`, the generated report, and PNG/PDF figures). The
timestamped `results/runs/` directories and per-attempt logs remain in the
source experiment workspace and are intentionally not duplicated here; the
committed configs and scripts regenerate that layout.

- Phase 1 is the clean constructive test: the aligned rank-4 update is rank matched and non-inferior to full feedback on the synthetic latent task.
- In Phase 2, derivative modulation, sparse masks, Dale projection, and fan-in normalization do not preserve the algebraic rank bound of the raw outer-product rule. The final E/I update is therefore not low rank even when activity remains low dimensional.
- Local, full-feedback, and shuffled-feedback task accuracies are indistinguishable in the completed conditions. These runs do not identify feedback alignment as the cause of behavior.
- Homeostatic updates dominate task-plastic updates in cumulative L1 budget. The existing homeostasis ablation is not budget matched and its stability result opposes the preregistered direction.
- The learned MD gate is a supervised Hebbian context classifier because its fit and modulatory features use true context information.
- B1 passes only the relative `>=90% BPTT` branch; it fails the absolute accuracy threshold. BPTT has not yet been tuned as a strong upper-bound baseline.
- The sequence-memory dataset was unavailable. The IBL result contains one session/animal and is descriptive only.

See `../docs/integrated_method_audit_zh.md` for the cross-workstream audit and the next falsification tests.
