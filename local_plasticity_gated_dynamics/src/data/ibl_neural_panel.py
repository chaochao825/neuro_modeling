"""Capability-safe adapters from cached IBL counts to exp14 model inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from src.data.ibl_behavior import IBLBehaviorObservations, LearnedCategoricalHMM
from src.data.ibl_multisession import (
    ChronologicalBlockSplit,
    CompleteCaseReceipt,
    IBLMultiSessionError,
    PreparedIBLNeuralSession,
    complete_case_trial_mask,
    full_trial_sensitivity_nuisance_table,
    past_safe_nuisance_table,
)
from src.models.context_belief import MDRecurrentBeliefGate
from src.models.hierarchical_count_dynamics import (
    BeliefFitReceipt,
    NeuralCountSession,
    TrialBlockSplit,
)
from src.tasks.hidden_context import GateObservationBatch


Array = np.ndarray
REGION_ORDER = (
    "cortex",
    "thalamus",
    "striatum",
    "hippocampus",
    "midbrain",
    "hindbrain",
)
MACRO_REGION_MAPPING_SCHEMA = "exp14_allen_macro_region_mapping_v1"
MACRO_REGION_SOURCE_ONTOLOGY_SHA256 = (
    "63654b8d35c7c1b5665636b645da774776ee8263658192f5dca1e815095e9147"
)
MACRO_REGION_SOURCE_PROVENANCE_SHA256 = (
    "a01b7fa535e6de437ac46e8cf9de68a87d6a9b5587d055a3935476d956109fdc"
)
MACRO_REGION_ANCESTOR_IDS = (
    ("cortex", 315),
    ("thalamus", 549),
    ("striatum", 477),
    ("hippocampus", 1089),
    ("midbrain", 313),
    ("hindbrain", 1065),
)
DEFAULT_MACRO_REGION_MAPPING_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "exp14_allen_macro_region_mapping_v1.json"
)
DEFAULT_MACRO_REGION_MAPPING_SHA256 = (
    "3bac702ed6b3ee5c21acbbfd929b077baa63226369ca8e1bef0b6faeb487fc23"
)


class IBLMacroRegionMappingError(ValueError):
    """Raised when the frozen Allen-derived macro mapping is not auditable."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _acronym_set_sha256(acronyms: Sequence[str]) -> str:
    encoded = json.dumps(
        sorted(set(acronyms)), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class AllenMacroRegionMapping:
    """Hash-bound exact acronym mapping derived from Allen ancestor paths."""

    artifact_sha256: str
    source_ontology_sha256: str
    source_provenance_sha256: str
    source_package: str
    source_version: str
    derivation_method: str
    selection_policy: str
    formal_compact_manifest_sha256: str
    formal_compact_acronym_count: int
    formal_compact_acronyms_sha256: str
    entries: tuple[tuple[str, int, str, int | None], ...]
    regression_sentinels: tuple[tuple[str, str], ...]
    _lookup: Mapping[str, str] = field(repr=False, compare=False)

    def classify(self, acronym: object) -> str:
        """Classify one exact Allen acronym; unknown labels are always ``other``."""

        return self._lookup.get(str(acronym).strip(), "other")

    def validate_acronym_scope(
        self, acronyms: Sequence[str], *, require_exact_formal_scope: bool
    ) -> None:
        values = tuple(str(value).strip() for value in acronyms)
        if any(not value for value in values):
            raise IBLMacroRegionMappingError("macro-region acronym scope is empty")
        if not require_exact_formal_scope:
            return
        unknown = tuple(sorted(set(values) - set(self._lookup)))
        if unknown:
            raise IBLMacroRegionMappingError(
                f"formal compact contains acronyms absent from frozen mapping: {unknown!r}"
            )
        unique_count = len(set(values))
        if (
            unique_count != self.formal_compact_acronym_count
            or _acronym_set_sha256(values) != self.formal_compact_acronyms_sha256
        ):
            raise IBLMacroRegionMappingError(
                "formal compact acronym scope differs from the frozen anatomical scope"
            )

    def receipt(self) -> Mapping[str, object]:
        return {
            "macro_region_mapping_schema": MACRO_REGION_MAPPING_SCHEMA,
            "macro_region_mapping_sha256": self.artifact_sha256,
            "macro_region_source_ontology_sha256": self.source_ontology_sha256,
            "macro_region_source_provenance_sha256": self.source_provenance_sha256,
            "macro_region_source_package": self.source_package,
            "macro_region_source_version": self.source_version,
            "macro_region_derivation_method": self.derivation_method,
            "macro_region_ancestor_ids": dict(MACRO_REGION_ANCESTOR_IDS),
            "macro_region_selection_policy": self.selection_policy,
            "macro_region_behavior_or_model_selected": False,
            "macro_region_unknown_policy": "other",
            "macro_region_fiber_tract_policy": "other",
            "macro_region_formal_acronym_count": self.formal_compact_acronym_count,
            "macro_region_formal_acronyms_sha256": self.formal_compact_acronyms_sha256,
        }


def _validate_mapping_against_source(
    mapping: AllenMacroRegionMapping, source_ontology_path: Path
) -> None:
    if (
        not source_ontology_path.is_file()
        or _file_sha256(source_ontology_path) != mapping.source_ontology_sha256
    ):
        raise IBLMacroRegionMappingError(
            "Allen source ontology is absent or has the wrong SHA-256"
        )
    provenance_path = source_ontology_path.with_name("region_mapping_provenance.json")
    if (
        not provenance_path.is_file()
        or _file_sha256(provenance_path) != mapping.source_provenance_sha256
    ):
        raise IBLMacroRegionMappingError(
            "Allen source ontology provenance is absent or has the wrong SHA-256"
        )
    with source_ontology_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "acronym", "structure_id_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise IBLMacroRegionMappingError(
                "Allen source ontology schema is incomplete"
            )
        rows_by_acronym: dict[str, list[Mapping[str, str]]] = {}
        for row in reader:
            rows_by_acronym.setdefault(str(row["acronym"]), []).append(row)
    ancestor_ids = dict(MACRO_REGION_ANCESTOR_IDS)
    for acronym, structure_id, macro_region, matched_ancestor in mapping.entries:
        rows = rows_by_acronym.get(acronym, [])
        if len(rows) != 1:
            raise IBLMacroRegionMappingError(
                f"Allen source ontology does not uniquely bind acronym {acronym!r}"
            )
        row = rows[0]
        try:
            row_id = int(row["id"])
            path_ids = {
                int(value)
                for value in str(row["structure_id_path"]).strip("/").split("/")
                if value
            }
        except (TypeError, ValueError) as error:
            raise IBLMacroRegionMappingError(
                f"Allen source ontology row is invalid for {acronym!r}"
            ) from error
        matches = tuple(
            (name, ancestor)
            for name, ancestor in ancestor_ids.items()
            if ancestor in path_ids
        )
        derived_region, derived_ancestor = matches[0] if matches else ("other", None)
        if len(matches) > 1 or (
            row_id,
            derived_region,
            derived_ancestor,
        ) != (structure_id, macro_region, matched_ancestor):
            raise IBLMacroRegionMappingError(
                f"frozen macro mapping disagrees with Allen ancestry for {acronym!r}"
            )
    for acronym, expected_region in mapping.regression_sentinels:
        rows = rows_by_acronym.get(acronym, [])
        if len(rows) != 1:
            raise IBLMacroRegionMappingError(
                f"Allen source ontology does not uniquely bind sentinel {acronym!r}"
            )
        try:
            path_ids = {
                int(value)
                for value in str(rows[0]["structure_id_path"]).strip("/").split("/")
                if value
            }
        except (TypeError, ValueError) as error:
            raise IBLMacroRegionMappingError(
                f"Allen source ontology sentinel path is invalid for {acronym!r}"
            ) from error
        matches = tuple(
            name for name, ancestor in ancestor_ids.items() if ancestor in path_ids
        )
        derived_region = matches[0] if matches else "other"
        if len(matches) > 1 or derived_region != expected_region:
            raise IBLMacroRegionMappingError(
                f"Allen source ontology sentinel classification is wrong for {acronym!r}"
            )


def load_allen_macro_region_mapping(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_compact_manifest_sha256: str,
    source_ontology_path: str | Path | None = None,
) -> AllenMacroRegionMapping:
    """Load and strictly validate the tracked anatomical mapping artifact."""

    artifact_path = Path(path)
    if not artifact_path.is_file() or _file_sha256(artifact_path) != expected_sha256:
        raise IBLMacroRegionMappingError(
            "macro-region mapping artifact is absent or has the wrong SHA-256"
        )
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IBLMacroRegionMappingError(
            "macro-region mapping artifact is not valid JSON"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "source_ontology",
        "derivation",
        "scope",
        "regression_sentinels",
        "entries",
    }:
        raise IBLMacroRegionMappingError(
            "macro-region mapping top-level schema is wrong"
        )
    if payload.get("schema_version") != MACRO_REGION_MAPPING_SCHEMA:
        raise IBLMacroRegionMappingError("macro-region mapping schema version is wrong")
    source = payload.get("source_ontology")
    derivation = payload.get("derivation")
    scope = payload.get("scope")
    sentinels = payload.get("regression_sentinels")
    raw_entries = payload.get("entries")
    if not all(
        isinstance(value, dict) for value in (source, derivation, scope, sentinels)
    ):
        raise IBLMacroRegionMappingError("macro-region mapping metadata is malformed")
    assert isinstance(source, dict)
    assert isinstance(derivation, dict)
    assert isinstance(scope, dict)
    assert isinstance(sentinels, dict)
    if (
        source.get("sha256") != MACRO_REGION_SOURCE_ONTOLOGY_SHA256
        or source.get("provenance_sha256") != MACRO_REGION_SOURCE_PROVENANCE_SHA256
        or source.get("source_package") != "iblatlas"
        or source.get("source_version") != "1.1.0"
        or derivation.get("method")
        != "exact_acronym_then_structure_id_path_ancestor_membership_v1"
        or derivation.get("ancestor_ids") != dict(MACRO_REGION_ANCESTOR_IDS)
        or derivation.get("unknown_acronym_policy") != "other"
        or derivation.get("fiber_tract_policy") != "other"
        or derivation.get("selection_policy")
        != "frozen_anatomy_only_no_behavior_or_model_outcomes"
        or scope.get("formal_compact_manifest_sha256")
        != expected_compact_manifest_sha256
    ):
        raise IBLMacroRegionMappingError(
            "macro-region source, derivation, or compact binding is wrong"
        )
    if not isinstance(raw_entries, list) or not raw_entries:
        raise IBLMacroRegionMappingError("macro-region mapping entries are missing")
    entries: list[tuple[str, int, str, int | None]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict) or set(entry) != {
            "acronym",
            "structure_id",
            "macro_region",
            "matched_ancestor_id",
        }:
            raise IBLMacroRegionMappingError(
                f"macro-region mapping entry {index} has the wrong schema"
            )
        acronym = str(entry["acronym"]).strip()
        macro_region = str(entry["macro_region"])
        structure_id = entry["structure_id"]
        matched = entry["matched_ancestor_id"]
        if (
            not acronym
            or isinstance(structure_id, (bool, np.bool_))
            or not isinstance(structure_id, (int, np.integer))
            or macro_region not in {*REGION_ORDER, "other"}
            or (
                matched is not None
                and (
                    isinstance(matched, (bool, np.bool_))
                    or not isinstance(matched, (int, np.integer))
                )
            )
        ):
            raise IBLMacroRegionMappingError(
                f"macro-region mapping entry {index} has invalid values"
            )
        expected_ancestor = dict(MACRO_REGION_ANCESTOR_IDS).get(macro_region)
        if (None if matched is None else int(matched)) != expected_ancestor:
            raise IBLMacroRegionMappingError(
                f"macro-region mapping entry {index} has an invalid ancestor"
            )
        entries.append((acronym, int(structure_id), macro_region, expected_ancestor))
    if tuple(acronym for acronym, *_ in entries) != tuple(
        sorted(acronym for acronym, *_ in entries)
    ) or len({acronym for acronym, *_ in entries}) != len(entries):
        raise IBLMacroRegionMappingError(
            "macro-region mapping acronyms must be unique and sorted"
        )
    expected_sentinels = {
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
    }
    if sentinels != expected_sentinels:
        raise IBLMacroRegionMappingError("macro-region regression sentinels are wrong")
    lookup = {acronym: macro_region for acronym, _, macro_region, _ in entries}
    if any(
        lookup.get(key, "other") != value for key, value in expected_sentinels.items()
    ):
        raise IBLMacroRegionMappingError(
            "macro-region entries disagree with regression sentinels"
        )
    try:
        acronym_count = int(scope["formal_compact_acronym_count"])
        acronym_sha256 = str(scope["formal_compact_acronyms_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise IBLMacroRegionMappingError(
            "macro-region scope receipt is invalid"
        ) from error
    if (
        acronym_count != len(entries)
        or _acronym_set_sha256(tuple(lookup)) != acronym_sha256
    ):
        raise IBLMacroRegionMappingError(
            "macro-region entry set does not bind the formal acronym scope"
        )
    mapping = AllenMacroRegionMapping(
        artifact_sha256=expected_sha256,
        source_ontology_sha256=str(source["sha256"]),
        source_provenance_sha256=str(source["provenance_sha256"]),
        source_package=str(source["source_package"]),
        source_version=str(source["source_version"]),
        derivation_method=str(derivation["method"]),
        selection_policy=str(derivation["selection_policy"]),
        formal_compact_manifest_sha256=str(scope["formal_compact_manifest_sha256"]),
        formal_compact_acronym_count=acronym_count,
        formal_compact_acronyms_sha256=acronym_sha256,
        entries=tuple(entries),
        regression_sentinels=tuple(sorted(expected_sentinels.items())),
        _lookup=MappingProxyType(lookup),
    )
    if source_ontology_path is not None:
        _validate_mapping_against_source(mapping, Path(source_ontology_path))
    return mapping


@lru_cache(maxsize=1)
def default_allen_macro_region_mapping() -> AllenMacroRegionMapping:
    return load_allen_macro_region_mapping(
        DEFAULT_MACRO_REGION_MAPPING_PATH,
        expected_sha256=DEFAULT_MACRO_REGION_MAPPING_SHA256,
        expected_compact_manifest_sha256=(
            "a5acb134ae4b34f47db150948a7f7ab58e8eb85e204fb981e0ca744eba328a09"
        ),
    )


def broad_region(
    acronym: object, mapping: AllenMacroRegionMapping | None = None
) -> str:
    """Map one exact acronym through the frozen Allen-ancestor artifact."""

    resource = default_allen_macro_region_mapping() if mapping is None else mapping
    return resource.classify(acronym)


def _freeze(array: Array, *, dtype: object | None = None) -> Array:
    value = np.array(array, dtype=dtype, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True, slots=True)
class IBLNeuralPanelInput:
    session_id: str
    animal_id: str
    counts: Array
    unit_ids: Array
    unit_regions: tuple[str, ...]
    trial_ids: Array
    stimulus_side: Array
    gate_trial_ids: Array
    gate_stimulus_side: Array
    block_ids: Array
    controls: Array
    complete_case_receipt: CompleteCaseReceipt
    view: str
    panel: str
    causal_timing_eligible: bool

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts)
        trial_ids = np.asarray(self.trial_ids)
        stimulus = np.asarray(self.stimulus_side)
        gate_trial_ids = np.asarray(self.gate_trial_ids)
        gate_stimulus = np.asarray(self.gate_stimulus_side)
        blocks = np.asarray(self.block_ids)
        controls = np.asarray(self.controls, dtype=float)
        units = np.asarray(self.unit_ids).astype(str)
        if counts.ndim != 3 or counts.dtype.kind not in {"i", "u"}:
            raise IBLMultiSessionError(
                "panel counts must be integer trial x time x unit"
            )
        if np.any(counts < 0) or counts.shape[1] < 2:
            raise IBLMultiSessionError(
                "panel counts must be non-negative with >=2 bins"
            )
        n_trials = counts.shape[0]
        if (
            trial_ids.shape != (n_trials,)
            or stimulus.shape != (n_trials,)
            or blocks.shape != (n_trials,)
            or controls.ndim != 2
            or controls.shape[0] != n_trials
        ):
            raise IBLMultiSessionError("panel trial arrays must align")
        if trial_ids.dtype.kind not in {"i", "u"} or np.any(trial_ids < 0):
            raise IBLMultiSessionError("panel trial IDs must be non-negative integers")
        if len(np.unique(trial_ids)) != n_trials or not np.isin(stimulus, [0, 1]).all():
            raise IBLMultiSessionError(
                "trial IDs must be unique and stimulus side binary"
            )
        if (
            gate_trial_ids.ndim != 1
            or gate_trial_ids.dtype.kind not in {"i", "u"}
            or np.any(gate_trial_ids < 0)
            or len(np.unique(gate_trial_ids)) != len(gate_trial_ids)
            or (
                len(gate_trial_ids) > 1
                and np.any(np.diff(gate_trial_ids.astype(np.int64)) != 1)
            )
            or gate_stimulus.shape != gate_trial_ids.shape
            or not np.isin(gate_stimulus, [0, 1]).all()
        ):
            raise IBLMultiSessionError(
                "gate history must be a consecutive integer-ID binary-stimulus tape"
            )
        gate_lookup = {int(value): index for index, value in enumerate(gate_trial_ids)}
        try:
            selected_gate_positions = np.asarray(
                [gate_lookup[int(value)] for value in trial_ids], dtype=int
            )
        except KeyError as error:
            raise IBLMultiSessionError(
                "neural-complete trial IDs must belong to the gate tape"
            ) from error
        if np.any(np.diff(selected_gate_positions) <= 0) or not np.array_equal(
            stimulus, gate_stimulus[selected_gate_positions]
        ):
            raise IBLMultiSessionError(
                "neural-complete stimuli must preserve the gate-tape order"
            )
        if not np.isfinite(controls).all():
            raise IBLMultiSessionError("panel controls must be complete and finite")
        if units.shape != (counts.shape[2],) or len(self.unit_regions) != len(units):
            raise IBLMultiSessionError("panel unit metadata must align")
        object.__setattr__(self, "counts", _freeze(counts, dtype=np.int64))
        object.__setattr__(self, "unit_ids", _freeze(units))
        object.__setattr__(self, "trial_ids", _freeze(trial_ids, dtype=np.int64))
        object.__setattr__(self, "stimulus_side", _freeze(stimulus, dtype=np.int64))
        object.__setattr__(
            self, "gate_trial_ids", _freeze(gate_trial_ids, dtype=np.int64)
        )
        object.__setattr__(
            self, "gate_stimulus_side", _freeze(gate_stimulus, dtype=np.int64)
        )
        object.__setattr__(self, "block_ids", _freeze(blocks))
        object.__setattr__(self, "controls", _freeze(controls, dtype=float))


def prepare_neural_panel_input(
    prepared: PreparedIBLNeuralSession,
    *,
    view: str,
    panel: str,
    minimum_trials: int,
    minimum_blocks: int,
    macro_region_mapping: AllenMacroRegionMapping | None = None,
) -> IBLNeuralPanelInput:
    """Apply one frozen complete-case mask to every panel capability."""

    if view not in {"stimulus_pre", "movement_pre"}:
        raise IBLMultiSessionError("unknown neural view")
    table = prepared.trial_table(view)
    if "stimulus_side" not in table or "block_id" not in table:
        raise IBLMultiSessionError("trial table lacks stimulus_side or block_id")
    if panel == "primary_past_safe":
        nuisance = past_safe_nuisance_table(table, view=view)
    elif panel == "full_trial_sensitivity":
        nuisance = full_trial_sensitivity_nuisance_table(table)
    else:
        raise IBLMultiSessionError("unknown nuisance panel")
    mask, receipt = complete_case_trial_mask(
        nuisance,
        valid_mask=prepared.valid_masks[view],
        trial_ids=prepared.current_trial_ids,
    )
    if int(mask.sum()) < minimum_trials:
        raise IBLMultiSessionError("too few complete trials for exp14")
    blocks = table.loc[mask, "block_id"].to_numpy()
    if len(np.unique(blocks)) < minimum_blocks:
        raise IBLMultiSessionError("too few complete chronological blocks for exp14")
    mapping = (
        default_allen_macro_region_mapping()
        if macro_region_mapping is None
        else macro_region_mapping
    )
    regions = tuple(broad_region(value, mapping) for value in prepared.regions)
    return IBLNeuralPanelInput(
        session_id=prepared.eid,
        animal_id=prepared.animal_id,
        counts=prepared.count_views[view][mask],
        unit_ids=prepared.unit_ids,
        unit_regions=regions,
        trial_ids=np.asarray(prepared.current_trial_ids[mask], dtype=np.int64),
        stimulus_side=table.loc[mask, "stimulus_side"].to_numpy(dtype=int),
        gate_trial_ids=np.asarray(prepared.current_trial_ids, dtype=np.int64),
        gate_stimulus_side=table["stimulus_side"].to_numpy(dtype=int),
        block_ids=blocks,
        controls=nuisance.loc[mask].to_numpy(dtype=float),
        complete_case_receipt=receipt,
        view=view,
        panel=panel,
        causal_timing_eligible=bool(
            nuisance.attrs.get("eligible_for_prestim_causal_timing", False)
        ),
    )


def common_region_anchors(
    sessions: Sequence[IBLNeuralPanelInput], *, min_units_per_region: int
) -> tuple[str, ...]:
    """Return the legacy intersection of regions represented in every session.

    Exp14 uses :func:`union_region_anchors` instead.  This compatibility helper
    remains useful for single-session fixtures and for explicitly requested
    complete-case anatomical intersections.
    """

    if not sessions or min_units_per_region < 1:
        raise ValueError("sessions and a positive unit threshold are required")
    return union_region_anchors(
        sessions,
        min_units_per_region=min_units_per_region,
        minimum_region_sessions=len(sessions),
    ).regions


@dataclass(frozen=True, slots=True)
class RegionAnchorAudit:
    """Immutable coverage receipt for a deterministic union anchor basis."""

    regions: tuple[str, ...]
    n_sessions: int
    minimum_region_sessions: int
    min_units_per_region: int
    region_session_counts: tuple[int, ...]
    region_session_fractions: tuple[float, ...]
    region_missing_session_ids: tuple[tuple[str, ...], ...]

    def coverage_records(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                "region": region,
                "n_sessions_present": count,
                "session_fraction_present": fraction,
                "n_sessions_missing": self.n_sessions - count,
                "missing_session_ids": missing,
            }
            for region, count, fraction, missing in zip(
                self.regions,
                self.region_session_counts,
                self.region_session_fractions,
                self.region_missing_session_ids,
                strict=True,
            )
        )


def union_region_anchors(
    sessions: Sequence[IBLNeuralPanelInput],
    *,
    min_units_per_region: int,
    minimum_region_sessions: int,
) -> RegionAnchorAudit:
    """Select a fixed-order union basis without dropping sparse sessions.

    Coverage depends only on immutable anatomical unit metadata.  A region is
    retained when at least ``minimum_region_sessions`` sessions contain
    ``min_units_per_region`` units.  Sessions may therefore omit retained
    regions; the downstream model imputes those structural missing values from
    training-fold statistics.
    """

    values = tuple(sessions)
    if not values or min_units_per_region < 1:
        raise ValueError("sessions and a positive unit threshold are required")
    if (
        isinstance(minimum_region_sessions, (bool, np.bool_))
        or not isinstance(minimum_region_sessions, (int, np.integer))
        or not 1 <= int(minimum_region_sessions) <= len(values)
    ):
        raise ValueError(
            "minimum_region_sessions must be an integer in [1, number of sessions]"
        )
    session_ids = tuple(session.session_id for session in values)
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("session IDs must be unique for region coverage")

    regions: list[str] = []
    counts: list[int] = []
    fractions: list[float] = []
    missing_ids: list[tuple[str, ...]] = []
    qualifying_by_session: dict[str, set[str]] = {
        session_id: set() for session_id in session_ids
    }
    for region in REGION_ORDER:
        present = tuple(
            session.session_id
            for session in values
            if sum(label == region for label in session.unit_regions)
            >= min_units_per_region
        )
        if len(present) < int(minimum_region_sessions):
            continue
        present_set = set(present)
        regions.append(region)
        counts.append(len(present))
        fractions.append(len(present) / len(values))
        missing_ids.append(
            tuple(
                session_id
                for session_id in session_ids
                if session_id not in present_set
            )
        )
        for session_id in present:
            qualifying_by_session[session_id].add(region)
    if not regions:
        raise IBLMultiSessionError(
            "no anatomical region satisfies the preregistered union coverage"
        )
    unrepresented = tuple(
        session_id
        for session_id in session_ids
        if not qualifying_by_session[session_id]
    )
    if unrepresented:
        raise IBLMultiSessionError(
            "union anchors would leave complete sessions without eligible units: "
            f"{unrepresented!r}"
        )
    return RegionAnchorAudit(
        regions=tuple(regions),
        n_sessions=len(values),
        minimum_region_sessions=int(minimum_region_sessions),
        min_units_per_region=int(min_units_per_region),
        region_session_counts=tuple(counts),
        region_session_fractions=tuple(fractions),
        region_missing_session_ids=tuple(missing_ids),
    )


def _positions(session: IBLNeuralPanelInput, identifiers: Sequence[object]) -> Array:
    lookup = {int(value): index for index, value in enumerate(session.trial_ids)}
    try:
        positions = np.asarray([lookup[int(value)] for value in identifiers], dtype=int)
    except (KeyError, TypeError, ValueError) as error:
        raise IBLMultiSessionError("split IDs are absent from the panel") from error
    if len(positions) < 1 or np.any(np.diff(positions) <= 0):
        raise IBLMultiSessionError("split IDs must retain chronological order")
    return positions


def _select_train_units(
    session: IBLNeuralPanelInput,
    train_ids: Sequence[object],
    *,
    common_regions: Sequence[str],
    max_units_per_region: int,
    min_units_per_region: int,
) -> Array:
    if max_units_per_region < min_units_per_region:
        raise ValueError("max_units_per_region must cover min_units_per_region")
    train = _positions(session, train_ids)
    mean_count = session.counts[train].mean(axis=(0, 1))
    labels = np.asarray(session.unit_regions, dtype=object)
    selected: list[int] = []
    for region in common_regions:
        candidates = np.flatnonzero(labels == region)
        if len(candidates) < min_units_per_region:
            continue
        identifiers = session.unit_ids[candidates].astype(str)
        order = np.lexsort((identifiers, -mean_count[candidates]))
        selected.extend(candidates[order[:max_units_per_region]].tolist())
    if not selected:
        raise IBLMultiSessionError(
            f"session {session.session_id!r} has no eligible union-anchor units"
        )
    result = np.asarray(selected, dtype=int)
    result.setflags(write=False)
    return result


def _past_only_beliefs(
    session: IBLNeuralPanelInput,
    fit_trial_ids: Sequence[object],
    *,
    hmm_options: Mapping[str, object],
    seed: int,
) -> tuple[Array, BeliefFitReceipt, Mapping[str, object]]:
    fit_ids = tuple(fit_trial_ids)
    positions = _positions(session, fit_ids)
    if not np.array_equal(positions, np.arange(len(positions))):
        raise IBLMultiSessionError("belief fitting trials must be a prefix")
    gate_lookup = {
        int(value): index for index, value in enumerate(session.gate_trial_ids)
    }
    try:
        gate_fit_stop = gate_lookup[int(fit_ids[-1])] + 1
        selected_gate_positions = np.asarray(
            [gate_lookup[int(value)] for value in session.trial_ids], dtype=int
        )
    except KeyError as error:
        raise IBLMultiSessionError(
            "model trial IDs are absent from gate history"
        ) from error
    options = dict(hmm_options)
    model_name = str(options.pop("model", "learned_categorical_hmm"))
    if model_name == "md_recurrent_predictive":
        require_identifiable = bool(options.pop("require_identifiable", False))
        # The local recurrent estimator has no iterative convergence state: a
        # completed fit is its convergence receipt.  Accept the common exp14
        # option so paired configurations need not carry model-specific keys.
        options.pop("require_converged", None)
        options["seed"] = int(seed)
        episode_positions = np.arange(len(session.gate_trial_ids), dtype=int)
        episode_starts = np.zeros(len(session.gate_trial_ids), dtype=bool)
        episode_starts[0] = True
        gate_tape = GateObservationBatch(
            cue_observations=session.gate_stimulus_side,
            trial_ids=session.gate_trial_ids,
            episode_ids=np.zeros(len(session.gate_trial_ids), dtype=int),
            episode_trial_indices=episode_positions,
            episode_start=episode_starts,
        )
        training_tape = gate_tape.subset(np.arange(gate_fit_stop, dtype=int))
        md_gate = MDRecurrentBeliefGate(**options).fit(training_tape)
        if require_identifiable and not md_gate.moment_anchor_identifiable_:
            raise IBLMultiSessionError(
                "MD recurrent predictive belief is not identifiable"
            )
        prediction = md_gate.predictive_prior(gate_tape)
        selected_beliefs = prediction.beliefs[selected_gate_positions]
        audit = md_gate.audit_metadata()
        checkpoint = {
            "model": model_name,
            "initial": [0.5, 0.5],
            "transition": md_gate.transition_.tolist(),
            "emission": md_gate.emission_.tolist(),
            "fit_observation_fingerprint": training_tape.fingerprint,
            "fit_trial_ids": audit["fit_trial_ids"],
            "fit_episode_ids": audit["fit_episode_ids"],
            "local_update_l1": audit["local_update_l1"],
            "estimated_context_hazard": audit["estimated_context_hazard"],
            "estimated_cue_reliability": audit["estimated_cue_reliability"],
            "moment_anchor_identifiable": audit["moment_anchor_identifiable"],
            "predictive_prior": True,
            "uses_current_trial_stimulus": False,
            "uses_future_trials": False,
            "accessed_true_context": False,
            "converged": True,
            "identifiable": bool(md_gate.moment_anchor_identifiable_),
            "restart_selection_policy": "not_applicable_local_recurrent",
            "selected_restart": None,
            "eligible_restart_count": 1,
            "eligible_restart_fallback": False,
        }
        receipt = BeliefFitReceipt.bind(
            selected_beliefs,
            method="md_recurrent_belief_predictive_prior",
            fit_trial_ids=fit_ids,
            observation_fit_trial_ids=tuple(
                int(value) for value in session.gate_trial_ids[:gate_fit_stop]
            ),
            checkpoint_payload=checkpoint,
        )
        return selected_beliefs, receipt, checkpoint
    if model_name != "learned_categorical_hmm":
        raise IBLMultiSessionError(f"unknown past-only belief model {model_name!r}")
    observations = IBLBehaviorObservations(
        session.gate_trial_ids, session.gate_stimulus_side
    )
    require_converged = bool(options.pop("require_converged", True))
    require_identifiable = bool(options.pop("require_identifiable", True))
    options["seed"] = int(seed)
    gate_fit_positions = np.arange(gate_fit_stop, dtype=int)
    hmm = LearnedCategoricalHMM(**options).fit(observations, gate_fit_positions)
    prediction = hmm.predict(observations)
    if (require_converged and not hmm.converged_) or (
        require_identifiable and not hmm.identifiable_
    ):
        raise IBLMultiSessionError("past-only HMM is not converged and identifiable")
    training_observations = IBLBehaviorObservations(
        session.gate_trial_ids[:gate_fit_stop],
        session.gate_stimulus_side[:gate_fit_stop],
    )
    checkpoint = {
        "initial": hmm.initial_.tolist(),
        "transition": hmm.transition_.tolist(),
        "emission": hmm.emission_.tolist(),
        "fit_observation_fingerprint": training_observations.fingerprint,
        "train_log_likelihood": hmm.train_log_likelihood_,
        "iterations": hmm.n_iterations_,
        "converged": hmm.converged_,
        "identifiable": hmm.identifiable_,
        "restart_selection_policy": hmm.restart_selection_policy,
        "selected_restart": hmm.selected_restart_,
        "eligible_restart_count": hmm.n_eligible_restarts_,
        "eligible_restart_fallback": hmm.eligible_restart_fallback_,
    }
    selected_beliefs = prediction.beliefs[selected_gate_positions]
    receipt = BeliefFitReceipt.bind(
        selected_beliefs,
        method="learned_categorical_hmm_past_only",
        fit_trial_ids=fit_ids,
        observation_fit_trial_ids=tuple(
            int(value) for value in session.gate_trial_ids[:gate_fit_stop]
        ),
        checkpoint_payload=checkpoint,
    )
    return selected_beliefs, receipt, checkpoint


@dataclass(frozen=True, slots=True)
class BuiltNeuralSession:
    session: NeuralCountSession
    split: TrialBlockSplit
    selected_unit_ids: tuple[str, ...]
    present_anchor_regions: tuple[str, ...]
    missing_anchor_regions: tuple[str, ...]
    hmm_checkpoint: Mapping[str, object]
    complete_case_receipt: CompleteCaseReceipt


def build_model_session(
    panel: IBLNeuralPanelInput,
    split: ChronologicalBlockSplit,
    *,
    common_regions: Sequence[str],
    max_units_per_region: int,
    min_units_per_region: int,
    hmm_options: Mapping[str, object],
    seed: int,
    split_block_ids: Sequence[object] | None = None,
) -> BuiltNeuralSession:
    """Fit the hidden gate and unit selection using this split's train prefix.

    ``split_block_ids`` is an optional evaluation-owned schedule used only to
    certify whole-block cross-validation.  It is never placed in the
    :class:`GateObservationBatch`, neural covariates, or observation model.
    """

    split_ids = tuple(int(value) for value in split.train_ids + split.heldout_ids)
    tape_positions = _positions(panel, split_ids)
    if not np.array_equal(tape_positions, np.arange(len(tape_positions))):
        raise IBLMultiSessionError(
            "nested split must completely cover a chronological panel prefix"
        )
    if split_block_ids is None:
        split_blocks = np.asarray(panel.block_ids)
    else:
        split_blocks = np.asarray(split_block_ids)
        if split_blocks.shape != panel.block_ids.shape:
            raise IBLMultiSessionError(
                "split_block_ids must align with every analysis panel trial"
            )
    analysis_split_blocks = split_blocks[tape_positions]
    analysis_panel = replace(
        panel,
        counts=panel.counts[tape_positions],
        trial_ids=panel.trial_ids[tape_positions],
        stimulus_side=panel.stimulus_side[tape_positions],
        block_ids=panel.block_ids[tape_positions],
        controls=panel.controls[tape_positions],
    )
    all_ids = tuple(int(value) for value in analysis_panel.trial_ids)
    selected = _select_train_units(
        analysis_panel,
        split.train_ids,
        common_regions=common_regions,
        max_units_per_region=max_units_per_region,
        min_units_per_region=min_units_per_region,
    )
    beliefs, belief_receipt, checkpoint = _past_only_beliefs(
        analysis_panel,
        split.train_ids,
        hmm_options=hmm_options,
        seed=seed,
    )
    neural = NeuralCountSession(
        session_id=analysis_panel.session_id,
        animal_id=analysis_panel.animal_id,
        counts=analysis_panel.counts[:, :, selected],
        unit_regions=tuple(analysis_panel.unit_regions[index] for index in selected),
        beliefs=beliefs,
        trial_ids=analysis_panel.trial_ids,
        belief_receipt=belief_receipt,
        controls=analysis_panel.controls,
    )
    model_split = TrialBlockSplit(
        tuple(int(value) for value in split.train_ids),
        tuple(int(value) for value in split.heldout_ids),
        all_ids,
        tuple(analysis_split_blocks.tolist()),
    )
    return BuiltNeuralSession(
        session=neural,
        split=model_split,
        selected_unit_ids=tuple(
            str(analysis_panel.unit_ids[index]) for index in selected
        ),
        present_anchor_regions=tuple(
            region for region in common_regions if region in set(neural.unit_regions)
        ),
        missing_anchor_regions=tuple(
            region
            for region in common_regions
            if region not in set(neural.unit_regions)
        ),
        hmm_checkpoint=checkpoint,
        complete_case_receipt=panel.complete_case_receipt,
    )


__all__ = [
    "AllenMacroRegionMapping",
    "BuiltNeuralSession",
    "DEFAULT_MACRO_REGION_MAPPING_PATH",
    "DEFAULT_MACRO_REGION_MAPPING_SHA256",
    "IBLNeuralPanelInput",
    "IBLMacroRegionMappingError",
    "MACRO_REGION_ANCESTOR_IDS",
    "MACRO_REGION_MAPPING_SCHEMA",
    "MACRO_REGION_SOURCE_ONTOLOGY_SHA256",
    "MACRO_REGION_SOURCE_PROVENANCE_SHA256",
    "REGION_ORDER",
    "RegionAnchorAudit",
    "broad_region",
    "build_model_session",
    "common_region_anchors",
    "default_allen_macro_region_mapping",
    "load_allen_macro_region_mapping",
    "prepare_neural_panel_input",
    "union_region_anchors",
]
