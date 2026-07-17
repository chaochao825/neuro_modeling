# Exp27 frozen-family actuator selector evidence

This bundle preserves the complete smoke and formal evidence generated on the
clean 210-server Git commit `c44519135e48521ef3742fd63081ee34846936e2`
(tree `f10986b5ad25cf721659bc3f2013b165a728a8b3`). The frozen Exp26 source
raw-metrics digest is
`b3ef5e22c241f832b1fd50254f87e3890ec45057bfeda3a784cbd218623a1193`.

## Registered outcome

The 30-seed formal profile concluded **support** for the scoped, conditional
claim that a locally trained task-descriptor selector chooses among frozen,
task-matched actuator-family policies better than the best fixed family on
strict-unseen `(alpha, transition rank, input rank)` compositions.

- Local three-factor held-out balanced accuracy: `0.858002`.
- Fixed-best held-out balanced accuracy: `0.749919`.
- Oracle ceiling: `0.866885`.
- Local minus fixed: `0.108083`, 95% seed-bootstrap CI
  `[0.101914, 0.114382]`, Holm-adjusted positive `p=1.99998e-05`.
- Local 80%-oracle-gain non-inferiority contrast: `0.014511`, 95% CI
  `[0.012111, 0.016844]`, Holm-adjusted positive `p=1.99998e-05`.
- Local selector accuracy: `0.914729`; GRU-BPTT selector accuracy:
  `0.915504`.

Coverage is complete: 30 seeds, 5,280 retained selector rows, 5,160
strict-unseen primary rows, and zero failed rows. The smoke profile retains 96
rows and is forced `inconclusive` by registration.

## Interpretation boundary

This is not evidence for online hidden-belief inference. Selector inputs are
privileged prospective task-demand descriptors, and the local third factor is
the complete validation-utility vector for all three candidate families. The
30 leave-one-seed-out fits share most meta-training seeds, so the statistical
result is explicitly conditional and cross-fitted, not an independent
30-selector replication. A fixed meta-train panel (seeds 0--29) versus new
independent test seeds (30--59) is the registered follow-up. Local and GRU
update-cost proxies are descriptive and not budget matched.

## Contents

- `formal/` and `smoke/`: conclusion, report, summary, seed endpoints, raw
  metrics, provenance, and PNG/PDF/SVG figures.
- `raw_runs_exp27_formal_v1.tar.gz`: all 30 formal attempts plus per-seed and
  panel logs (`SHA-256 8f6622b3bf145e2ce0901c4a246fcd25e743fbc8bdc5a5f7f8f67544496fb0fe`).
- `raw_runs_exp27_smoke_v1.tar.gz`: both smoke attempts
  (`SHA-256 58d250273269892b0eb4e40d11114496fa6cb5cbe2e6cfe6cefc813f709037a8`).
- `configs/`: exact formal and smoke configurations.
- `logs/`: the formal parallel launcher.
- `MANIFEST.sha256`: SHA-256 digest for every other file in this bundle.

The formal collector independently reloaded the hash-locked Exp26 panel,
rebuilt every outer fold and normalizer, and deterministically replayed all
local and GRU fits before accepting the conclusion. The implementation passed
`1024` tests locally and on the 210 server; Ruff also passed on both systems.
