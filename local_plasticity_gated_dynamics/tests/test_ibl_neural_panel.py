from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.data.ibl_multisession import (
    PreparedIBLNeuralSession,
    chronological_outer_inner_splits,
)
from src.data.ibl_neural_panel import (
    DEFAULT_MACRO_REGION_MAPPING_PATH,
    DEFAULT_MACRO_REGION_MAPPING_SHA256,
    IBLMacroRegionMappingError,
    broad_region,
    build_model_session,
    common_region_anchors,
    default_allen_macro_region_mapping,
    load_allen_macro_region_mapping,
    prepare_neural_panel_input,
    union_region_anchors,
)


def _prepared(seed: int = 4) -> PreparedIBLNeuralSession:
    rng = np.random.default_rng(seed)
    n_trials, n_time = 96, 6
    block_ids = np.repeat(np.arange(8), n_trials // 8)
    probability_left = np.where(block_ids % 2 == 0, 0.8, 0.2)
    stimulus_side = rng.binomial(1, probability_left)
    signed = np.where(stimulus_side == 1, 1.0, -1.0)
    table = pd.DataFrame(
        {
            "stimulus": signed,
            "stimulus_side": stimulus_side,
            "choice": np.where(rng.random(n_trials) < 0.5, -1, 1),
            "reward": np.where(rng.random(n_trials) < 0.7, 1, -1),
            "reaction_time": rng.uniform(0.2, 0.8, n_trials),
            "wheel": rng.uniform(0.0, 2.0, n_trials),
            "motion_energy_proxy": rng.uniform(0.0, 1.0, n_trials),
            "probability_left": probability_left,
            "stim_on": np.arange(n_trials, dtype=float),
            "first_movement": np.arange(n_trials, dtype=float) + 0.4,
            "timing_valid": True,
            "block_id": block_ids,
        }
    )
    regions = np.asarray(
        ["MOs1", "VISp4", "ORBvl1", "VAL", "LP", "VPL", "CP", "ACB", "STR"]
    )
    counts = np.empty((n_trials, n_time, len(regions)), dtype=np.int64)
    for trial in range(n_trials):
        state = 1 if probability_left[trial] > 0.5 else -1
        for time in range(n_time):
            modulation = state * (time - 2.5) * np.linspace(-0.08, 0.08, len(regions))
            counts[trial, time] = rng.poisson(np.exp(1.8 + modulation))
    movement = table.copy(deep=True)
    movement["motion_energy_proxy"] += 10.0
    return PreparedIBLNeuralSession(
        eid=f"eid-{seed}",
        animal_id=f"animal-{seed}",
        count_views={"stimulus_pre": counts, "movement_pre": counts + 1},
        valid_masks={
            "stimulus_pre": np.ones(n_trials, dtype=bool),
            "movement_pre": np.ones(n_trials, dtype=bool),
        },
        time_axes={
            "stimulus_pre": np.linspace(-0.5, -0.1, n_time),
            "movement_pre": np.linspace(-0.5, -0.1, n_time),
        },
        regions=regions,
        unit_ids=np.asarray([f"u{index}" for index in range(len(regions))]),
        view_trial_tables={"stimulus_pre": table, "movement_pre": movement},
        current_trial_ids=np.arange(n_trials),
    )


def _outer(panel):
    outer, _ = chronological_outer_inner_splits(panel.trial_ids, panel.block_ids)
    return outer


def test_allen_ancestor_mapping_handles_prefix_collisions_and_fiber_tracts() -> None:
    expected = {
        "POST": "hippocampus",
        "VPL": "thalamus",
        "VPM": "thalamus",
        "PAR": "hippocampus",
        "PAA": "other",
        "PPN": "midbrain",
        "APN": "midbrain",
        "MB": "midbrain",
        "ZI": "other",
        "fiber tracts": "other",
        "cc": "other",
        "fa": "other",
        "never-an-allen-acronym": "other",
    }
    mapping = default_allen_macro_region_mapping()
    assert {label: broad_region(label, mapping) for label in expected} == expected
    assert (
        mapping.selection_policy == "frozen_anatomy_only_no_behavior_or_model_outcomes"
    )
    assert mapping.source_package == "iblatlas"
    assert mapping.source_version == "1.1.0"


def test_allen_macro_mapping_is_hash_bound_and_formal_scope_is_fail_closed() -> None:
    mapping = load_allen_macro_region_mapping(
        DEFAULT_MACRO_REGION_MAPPING_PATH,
        expected_sha256=DEFAULT_MACRO_REGION_MAPPING_SHA256,
        expected_compact_manifest_sha256=(
            "a5acb134ae4b34f47db150948a7f7ab58e8eb85e204fb981e0ca744eba328a09"
        ),
    )
    mapping.validate_acronym_scope(
        tuple(acronym for acronym, *_ in mapping.entries),
        require_exact_formal_scope=True,
    )
    with np.testing.assert_raises_regex(IBLMacroRegionMappingError, "wrong SHA-256"):
        load_allen_macro_region_mapping(
            DEFAULT_MACRO_REGION_MAPPING_PATH,
            expected_sha256="0" * 64,
            expected_compact_manifest_sha256=mapping.formal_compact_manifest_sha256,
        )
    with np.testing.assert_raises_regex(IBLMacroRegionMappingError, "scope differs"):
        mapping.validate_acronym_scope(
            tuple(acronym for acronym, *_ in mapping.entries[:-1]),
            require_exact_formal_scope=True,
        )


def test_panel_complete_case_and_past_only_belief_receipts() -> None:
    panel = prepare_neural_panel_input(
        _prepared(),
        view="stimulus_pre",
        panel="primary_past_safe",
        minimum_trials=60,
        minimum_blocks=5,
    )
    assert panel.trial_ids[0] == 1
    assert panel.complete_case_receipt.excluded_trial_ids == (0,)
    assert panel.causal_timing_eligible
    anchors = common_region_anchors((panel,), min_units_per_region=2)
    assert anchors == ("cortex", "thalamus", "striatum")
    built = build_model_session(
        panel,
        _outer(panel),
        common_regions=anchors,
        max_units_per_region=2,
        min_units_per_region=2,
        hmm_options={"max_iter": 100, "n_restarts": 2},
        seed=7,
    )
    assert built.session.belief_receipt.method == "learned_categorical_hmm_past_only"
    assert built.session.belief_receipt.fit_trial_ids == built.split.train_trial_ids
    assert not built.session.belief_receipt.accessed_true_context
    assert len(built.selected_unit_ids) == 6
    assert built.split.ordered_trial_ids == tuple(panel.trial_ids)


def test_union_anchors_are_fixed_order_and_do_not_drop_disjoint_sessions() -> None:
    first = prepare_neural_panel_input(
        _prepared(31),
        view="stimulus_pre",
        panel="primary_past_safe",
        minimum_trials=60,
        minimum_blocks=5,
    )
    second = prepare_neural_panel_input(
        _prepared(32),
        view="stimulus_pre",
        panel="primary_past_safe",
        minimum_trials=60,
        minimum_blocks=5,
    )
    first = replace(
        first,
        unit_regions=("cortex", "cortex") + ("other",) * (first.counts.shape[2] - 2),
    )
    second = replace(
        second,
        unit_regions=("thalamus", "thalamus")
        + ("other",) * (second.counts.shape[2] - 2),
    )

    audit = union_region_anchors(
        (first, second),
        min_units_per_region=2,
        minimum_region_sessions=1,
    )
    assert audit.regions == ("cortex", "thalamus")
    assert audit.region_session_counts == (1, 1)
    assert audit.region_session_fractions == (0.5, 0.5)
    assert audit.region_missing_session_ids == (
        (second.session_id,),
        (first.session_id,),
    )
    assert (
        tuple(record["region"] for record in audit.coverage_records()) == audit.regions
    )

    first_built = build_model_session(
        first,
        _outer(first),
        common_regions=audit.regions,
        max_units_per_region=2,
        min_units_per_region=2,
        hmm_options={
            "max_iter": 100,
            "n_restarts": 2,
            "require_converged": False,
            "require_identifiable": False,
        },
        seed=33,
    )
    second_built = build_model_session(
        second,
        _outer(second),
        common_regions=audit.regions,
        max_units_per_region=2,
        min_units_per_region=2,
        hmm_options={
            "max_iter": 100,
            "n_restarts": 2,
            "require_converged": False,
            "require_identifiable": False,
        },
        seed=34,
    )
    assert first_built.present_anchor_regions == ("cortex",)
    assert first_built.missing_anchor_regions == ("thalamus",)
    assert second_built.present_anchor_regions == ("thalamus",)
    assert second_built.missing_anchor_regions == ("cortex",)
    assert (
        len(first_built.selected_unit_ids) == len(second_built.selected_unit_ids) == 2
    )


def test_union_anchor_coverage_threshold_is_explicit_and_fail_closed() -> None:
    panel = prepare_neural_panel_input(
        _prepared(35),
        view="stimulus_pre",
        panel="primary_past_safe",
        minimum_trials=60,
        minimum_blocks=5,
    )
    with np.testing.assert_raises_regex(ValueError, "minimum_region_sessions"):
        union_region_anchors(
            (panel,), min_units_per_region=2, minimum_region_sessions=0
        )
    with np.testing.assert_raises_regex(ValueError, "minimum_region_sessions"):
        union_region_anchors(
            (panel,), min_units_per_region=2, minimum_region_sessions=2
        )
    unrepresented = replace(
        panel, unit_regions=("other",) * panel.counts.shape[2], session_id="other-only"
    )
    with np.testing.assert_raises_regex(ValueError, "without eligible units"):
        union_region_anchors(
            (panel, unrepresented),
            min_units_per_region=2,
            minimum_region_sessions=1,
        )


def test_probability_truth_mutation_does_not_change_gate_or_model_input() -> None:
    original = _prepared(8)
    mutated_tables = {
        name: table.copy(deep=True)
        for name, table in original.view_trial_tables.items()
    }
    for table in mutated_tables.values():
        table["probability_left"] = np.linspace(0.01, 0.99, len(table))
    mutated = replace(original, view_trial_tables=mutated_tables)
    kwargs = {
        "view": "stimulus_pre",
        "panel": "primary_past_safe",
        "minimum_trials": 60,
        "minimum_blocks": 5,
    }
    first = prepare_neural_panel_input(original, **kwargs)
    second = prepare_neural_panel_input(mutated, **kwargs)
    np.testing.assert_array_equal(first.stimulus_side, second.stimulus_side)
    np.testing.assert_array_equal(first.controls, second.controls)
    split = _outer(first)
    anchors = common_region_anchors((first,), min_units_per_region=2)
    build_kwargs = {
        "common_regions": anchors,
        "max_units_per_region": 2,
        "min_units_per_region": 2,
        "hmm_options": {
            "max_iter": 100,
            "n_restarts": 2,
            "require_converged": False,
            "require_identifiable": False,
        },
        "seed": 12,
    }
    a = build_model_session(first, split, **build_kwargs)
    b = build_model_session(second, split, **build_kwargs)
    np.testing.assert_array_equal(a.session.beliefs, b.session.beliefs)
    assert a.session.belief_receipt == b.session.belief_receipt


def test_heldout_stimulus_updates_online_belief_but_not_fitted_checkpoint() -> None:
    panel = prepare_neural_panel_input(
        _prepared(10),
        view="stimulus_pre",
        panel="primary_past_safe",
        minimum_trials=60,
        minimum_blocks=5,
    )
    split = _outer(panel)
    test_start = len(split.train_ids)
    changed_side = panel.stimulus_side.copy()
    changed_side[test_start] = 1 - changed_side[test_start]
    changed_gate = panel.gate_stimulus_side.copy()
    changed_trial_id = int(panel.trial_ids[test_start])
    changed_gate_position = int(
        np.flatnonzero(panel.gate_trial_ids == changed_trial_id)[0]
    )
    changed_gate[changed_gate_position] = 1 - changed_gate[changed_gate_position]
    changed = replace(
        panel,
        stimulus_side=changed_side,
        gate_stimulus_side=changed_gate,
    )
    anchors = common_region_anchors((panel,), min_units_per_region=2)
    kwargs = {
        "common_regions": anchors,
        "max_units_per_region": 2,
        "min_units_per_region": 2,
        "hmm_options": {
            "max_iter": 100,
            "n_restarts": 2,
            "require_converged": False,
            "require_identifiable": False,
        },
        "seed": 5,
    }
    first = build_model_session(panel, split, **kwargs)
    second = build_model_session(changed, split, **kwargs)
    assert first.hmm_checkpoint == second.hmm_checkpoint
    np.testing.assert_array_equal(
        first.session.beliefs[: test_start + 1],
        second.session.beliefs[: test_start + 1],
    )
    assert not np.array_equal(
        first.session.beliefs[test_start + 1 :],
        second.session.beliefs[test_start + 1 :],
    )


def test_gate_consumes_stimuli_from_neural_incomplete_trials() -> None:
    panel = prepare_neural_panel_input(
        _prepared(12),
        view="stimulus_pre",
        panel="primary_past_safe",
        minimum_trials=60,
        minimum_blocks=5,
    )
    assert panel.trial_ids[0] == 1 and panel.gate_trial_ids[0] == 0
    changed_gate = panel.gate_stimulus_side.copy()
    changed_gate[0] = 1 - changed_gate[0]
    changed = replace(panel, gate_stimulus_side=changed_gate)
    split = _outer(panel)
    anchors = common_region_anchors((panel,), min_units_per_region=2)
    kwargs = {
        "common_regions": anchors,
        "max_units_per_region": 2,
        "min_units_per_region": 2,
        "hmm_options": {
            "max_iter": 100,
            "n_restarts": 2,
            "require_converged": False,
            "require_identifiable": False,
        },
        "seed": 6,
    }
    first = build_model_session(panel, split, **kwargs)
    second = build_model_session(changed, split, **kwargs)
    assert first.session.belief_receipt.observation_fit_trial_ids[0] == 0
    assert not np.array_equal(first.session.beliefs[0], second.session.beliefs[0])


def test_full_trial_panel_is_explicitly_timing_ineligible() -> None:
    panel = prepare_neural_panel_input(
        _prepared(11),
        view="movement_pre",
        panel="full_trial_sensitivity",
        minimum_trials=60,
        minimum_blocks=5,
    )
    assert not panel.causal_timing_eligible
    assert panel.complete_case_receipt.nuisance_scope == "full_trial_sensitivity"


def test_gate_tape_rejects_gaps_and_reordering() -> None:
    panel = prepare_neural_panel_input(
        _prepared(13),
        view="stimulus_pre",
        panel="primary_past_safe",
        minimum_trials=60,
        minimum_blocks=5,
    )
    gap = panel.gate_trial_ids.copy()
    gap[10:] += 1
    with np.testing.assert_raises_regex(ValueError, "consecutive"):
        replace(panel, gate_trial_ids=gap)
    reordered = panel.gate_trial_ids.copy()
    reordered[[10, 11]] = reordered[[11, 10]]
    with np.testing.assert_raises_regex(ValueError, "consecutive"):
        replace(panel, gate_trial_ids=reordered)
