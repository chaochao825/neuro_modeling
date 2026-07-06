"""Registry for public neural datasets.

The pipeline avoids implicit heavy downloads. Each registry entry documents the
expected local export format consumed by fit_public_data.py.
"""

from __future__ import annotations

from typing import Dict


PUBLIC_DATASETS: Dict[str, Dict[str, str]] = {
    "allen_visual_coding": {
        "modality": "2p / Neuropixels",
        "expected_local_format": "time x units spike/count matrix in npy, npz, or csv plus optional behavior covariates",
        "notes": "Use bins of 1, 5, 10, 20 ms for spikes; report deconvolved and raw traces for calcium.",
    },
    "stringer_2019_visual_cortex": {
        "modality": "calcium",
        "expected_local_format": "trial/time x cells response matrix and stimulus labels",
        "notes": "Use cross-validated PCA/SVCA and shuffle controls.",
    },
    "ibl_brainwide_map": {
        "modality": "Neuropixels",
        "expected_local_format": "binned spikes with trial, wheel, lick, reward, pupil/running covariates when available",
        "notes": "State conditioning is required before interpreting correlations.",
    },
    "steinmetz_2019": {
        "modality": "Neuropixels",
        "expected_local_format": "time x units binned spikes with brain region labels",
        "notes": "Compare task engagement, choice, action, and region-specific fits.",
    },
    "buzsaki_crcns": {
        "modality": "extracellular spikes / LFP",
        "expected_local_format": "binned spikes and optional LFP/theta phase",
        "notes": "Control sleep/wake and theta phase for hippocampal analyses.",
    },
}


def describe_registry() -> Dict[str, Dict[str, str]]:
    return PUBLIC_DATASETS

