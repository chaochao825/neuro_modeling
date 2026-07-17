# Exp26 formal-v2 evidence bundle

This directory is the immutable publication bundle for the preregistered
task--actuator matching phase diagram.  The formal panel was generated on Git
commit `e08beaf9f51aacaaa80d42b2755c60d4080364bb` (tree
`c91f11863a654c7588b0d7bde5eb1e1f43e5c774`) with run label
`exp26-formal-v2`.  The corresponding local Windows commit has the same tree
but a different commit identifier because the first patch was replayed on the
210 GitHub worktree.

## Registered conclusion

**Support.** All 30 independent seeds and all 13,200 planned rows completed;
there were no missing, failed, invalid, duplicate, budget-invalid, or unstable
cells.  On held-out task generators:

- seed-level Spearman rho: mean 0.7605, 95% CI [0.7443, 0.7757];
- threshold classifier balanced accuracy: mean 0.8572, 95% CI
  [0.8375, 0.8758];
- threshold classifier AUROC: mean 0.9467, 95% CI [0.9365, 0.9560].

Each co-primary one-sided permutation test had raw `p=9.9999e-06` and
Holm-adjusted `p=2.99997e-05`.  The preregistered incremental gate also passed:
Gramian demand `chi` exceeded raw `alpha` by 0.1148 AUROC, 95% CI
[0.1009, 0.1286], `p=9.9999e-06`.

This supports the narrow claim that prospective task-demand geometry predicts
whether input routing/population gain or low-rank recurrent control is the
better single actuator family.  It does not show that a biological local rule
learned the selector; RGL remains a descriptive composite ceiling.

## Frozen preflight

The formal runner consumed the exact seven-file receipt under `preflight/`.
It covers `30 seeds x 88 generators x 4 active modes = 10,560` train-only
fits, with zero blockers.  Its aggregate receipt SHA-256 is
`bad665691233c9611fcdcce897c642d517a938b78adbabadee783c5e8cb1a671`.
The observed required scale maximum was `90.09208739042319`, matching the
registered `90.09208739042293` within the frozen numerical tolerance and
giving the registered next-power-of-two ceiling of 128.

The receipt retains its original remote `config_path` because that byte string
is part of the aggregate receipt hash used by every formal row.

## Contents

- `formal/`: conclusion, report, seed endpoints, aggregate summary, compressed
  raw metrics, and publication PDF/SVG;
- `smoke/`: the disjoint seed-9000/9001 pipeline check, permanently forced to
  `inconclusive`;
- `preflight/`: the complete clean train-only budget receipt;
- `logs/`: launcher, per-seed logs, exit receipt, and summary/preflight logs;
- `raw_runs_exp26_formal_v2.tar.gz`: exact ignored run artifacts for smoke and
  formal panels; SHA-256
  `4d7bec13533bed6c8f07a659b164ef8e4fc5f1727655c6a4ba43582ce596d368`;
- `MANIFEST.sha256`: hashes for the 60 unarchived generated evidence files.

For repository hygiene, the two publication SVGs differ from the downloaded
Matplotlib files only by removal of trailing spaces; `MANIFEST.sha256` records
the committed SVG bytes. PDFs, metrics, receipts, logs, and raw archives are
byte-identical to the verified 210 bundle.

Runtime: Python 3.11.15, NumPy 2.3.5, SciPy 1.17.1, pandas 3.0.1,
scikit-learn 1.8.0, and statsmodels 0.14.6.  The statistical unit is the seed;
generator, neuron, trial, and time point are not treated as independent
replicates.
