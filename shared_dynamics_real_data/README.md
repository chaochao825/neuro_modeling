# Shared-basis dynamics on public recordings

This subproject tests whether two recordings from the same 11,445 visual-cortex neurons can be described by a shared low-dimensional observation basis with context-specific dynamics.

The source files are the visual responding and visual spontaneous matrices from the upstream `Minimal_computation` release. The comparison relies on the upstream guarantee that rows refer to the same neurons in the same order; the MAT files do not carry independent unit identifiers with which to re-verify that alignment. They contain binary `X[neuron, time]` only and no trials, image identity, behavior, animal identity, E/I labels, anatomical coordinates, or within-recording switch timestamps.

## Models

- `common`: one training-fitted PCA basis and one affine latent transition shared across contexts.
- `shared`: one basis with a context-specific affine transition and process noise.
- `separate`: context-specific low-rank bases, means, transitions, and observation noise. This is a restricted low-rank comparator, not a full observation-space LDS.
- `random`: shared dynamics in a seeded random orthonormal basis.
- `orthogonal`: shared dynamics in PCA directions orthogonal to the aligned leading basis.
- `shuffled`: rows of the aligned basis are permuted across neurons. This is a neuron-alignment control, not a temporal shuffle.

Held-out marginal Gaussian LDS likelihood is computed with a diagonal observation covariance and the Woodbury identity, so scoring never constructs an `N x N` innovation covariance. Reported NLL includes the scale Jacobian and is in the selected original-neuron coordinates. The Gaussian score is a population prediction metric for binarized calcium events; it is not a Bernoulli spike likelihood or a causal connectivity estimate.

## Leakage and statistics contract

- Each recording is split into contiguous chronological blocks with a purge gap.
- No lag transition crosses a held-out or purged gap.
- Unit selection, mean/scale, PCA, dynamics, observation/process noise, and rollout normalization are fit on training blocks only.
- Computational seeds sample matched neuron subsets and test robustness. Folds, seeds, time bins, and neurons are not treated as biological replicates.
- With one aligned recording pair, all population-level claims remain `inconclusive` even when within-recording directions are favorable.
- Every attempted seed/fold/dimension/model cell is written to immutable JSONL/CSV artifacts, including failures.

## Reproduce on Python 3.11

From the repository root:

```bash
python -m pip install -e ./shared_dynamics_real_data
python -m pytest -q shared_dynamics_real_data/tests

python -m shared_dynamics_real_data.run_visual_context \
  --config shared_dynamics_real_data/configs/smoke.json \
  --data-root minimal_computation_original

python -m shared_dynamics_real_data.build_report --profile smoke
python -m shared_dynamics_real_data.figures.plot_visual_context
```

The formal config uses 20 deterministic unit-subset seeds, three contiguous folds, 128 aligned units, and latent dimensions `1,2,4,8,16,32`. Parallel seed groups may write separate immutable run directories; `build_report.py` combines only identical analysis and data fingerprints and refuses ambiguous panels.

## Formal result snapshot

All 2,160 planned cells completed. At `d=4`, the median paired shared-minus-common NLL was `+0.000203` (switching shared was slightly worse), while the shared model's mean one-step R2 was `-0.001567` and mean rollout NRMSE was `1.013948`. The separately computed median-based joint absolute-signal margin was `-0.013727`. Aligned shared dynamics descriptively beat random, orthogonal, and neuron-alignment-shuffled controls at `d=4`; the median across seeds of the minimum control-minus-shared gap was `+0.001340`.

These directions do not establish a population-level advantage: there is only one aligned recording pair. The formal conclusion for every real-data claim remains `inconclusive`. See `results/summary.csv` and `results/report.md` for the complete three-way audit.

At `d=4`, common also dominates shared on all four reported prediction/complexity axes: NLL `-0.239844` vs `-0.239648` (lower is better), one-step R2 `-0.000743` vs `-0.001567`, rollout NRMSE `1.013199` vs `1.013948`, and 910 vs 934 fitted parameters. Thus the aligned-basis control result does not demonstrate a switching/shared advantage. The `d=4` versus `d=32` comparison is only a comparison of two tested hyperparameters, not an estimate that the data have intrinsic rank four.

Formal outputs are:

- `results/raw_metrics.csv`
- `results/latest_metrics.csv`
- `results/model_summary.csv`
- `results/comparisons.csv`
- `results/summary.csv`
- `results/report.md`
- `results/runs/`: path-sanitized immutable configs, manifests, raw JSONL/CSV metrics, statuses, and logs for all 21 shards (20 formal plus one smoke run)
- `figures/visual_context_shared_dynamics.png`
- `figures/visual_context_shared_dynamics.pdf`
