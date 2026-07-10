# Neural Multiscale Tests

> Scope: `synthetic_calibration / single_seed`. The committed H1--H5 labels
> and figures are pipeline-calibration artifacts, not inferential evidence.
> Equal-time Minimal-computation results are methodologically separate and do
> not establish temporal history dependence or anatomical locality.

Reproducible simulation and analysis pipeline for separating five competing
mechanistic claims in neural population dynamics:

- H1: history dependence and local coupling
- H2: near-critical or critical-initialized eigenspectra
- H3: oscillatory synchrony requires extra phase and complex-mode evidence
- H4: avalanche criticality requires more than a log-log line
- H5: energy and wiring constraints can shape sparse near-critical dynamics

The default run is intentionally self-contained. It executes synthetic models
and writes a decision matrix without downloading public datasets. Public data
scripts accept local matrices or dataset-specific exports and apply the same
analysis functions.

## Quick Start

```bash
python run_simulations.py --quick --seed 7
python -m unittest discover -s tests
```

Outputs are written under `reports/` and `figures/`:

- `reports/summary.json`
- `reports/report.md`
- `reports/decision_matrix.json`
- root-level `report.md` and `decision_matrix.json` mirrors

## Public Data Interface

```bash
python fit_public_data.py --spike-matrix path/to/spikes.npy --bin-ms 10
python fit_public_data.py --demo-synthetic
```

The public registry documents Allen, IBL, Steinmetz, Stringer, and
Buzsaki/CRCNS targets. Heavy downloads are not performed by default.

