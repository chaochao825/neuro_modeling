# Exp28 amended actuator-selector sensitivity

This bundle is the replay-verified result from Git commit
`07b5ce23108ca179e1291393fccc12bd9227238e` (tree
`ca01b07349067d5faed5540493ba2d4689dac4c1`). One Local three-factor
selector and one GRU-BPTT baseline were fitted once with root seed 2801 on the
frozen Exp26 seeds 0--29 meta panel, then evaluated without refitting on the
ceiling-amended seeds 30--59 panel.

The overall three-class conclusion is **inconclusive** because the ceiling was
amended after inspection of the original independent panel. The separately
labelled directional sensitivity is **support**:

- Local utility: 0.855241; fixed-best utility: 0.753658; oracle: 0.865819.
- Local minus fixed-best: 0.101583, 95% seed-bootstrap CI
  [0.096471, 0.106751], one-sided Holm p = 1.99998e-5.
- Local 80%-oracle non-inferiority contrast: 0.011854, 95% CI
  [0.009507, 0.014165], one-sided Holm p = 1.99998e-5.
- Local versus GRU-BPTT: +0.004780, descriptive 95% CI
  [0.002544, 0.006936].
- Statistical unit: 30 post-hoc amended evaluation seeds; 43 strict-unseen
  generators were averaged within each seed.

This does not establish a confirmatory independent result, hidden belief
inference, de-novo actuator formation, scalar-reward-only learning, or matched
Local/GRU update budgets. The fully untouched confirmatory panel is separately
reserved for seeds 60--89.

`exp28_selector_sensitivity_v2_07b5ce2_full.tar.gz` preserves the complete raw
run, config, environment, planned conditions, fit/decision receipts, logs, and
summary. Its SHA-256 is
`e689111ca660e0dc8995d2a171aff5f75eef0b2d43126c8c3ebdb219836262b6`.
All exported artifact hashes are listed in `MANIFEST.sha256`.
