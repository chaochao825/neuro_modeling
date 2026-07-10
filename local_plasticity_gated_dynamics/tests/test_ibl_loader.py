from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.data.ibl_loader import (
    CachedIBLSessionSource,
    IBLDataError,
    IBLDependencyError,
    OneAPISource,
    ProbeSpikes,
    TrialNuisanceResidualizer,
    build_trial_covariates,
    contiguous_context_blocks,
    event_aligned_spike_counts,
    load_ibl_trial_data,
)


def _write_array(path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values))


def _symlink_or_skip(link: Path, target: Path, *, directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symbolic links unavailable: {error}")


def _cached_alf_fixture(tmp_path):
    cache = tmp_path / "cache"
    session = cache / "lab" / "Subjects" / "mouse-cache" / "2024-01-02" / "001"
    alf = session / "alf"
    alf.mkdir(parents=True)
    base_response = np.array([1.2, 2.3])
    _write_array(alf / "_ibl_trials.response_times.npy", base_response)
    table = pd.DataFrame(
        {
            "stimOn_times": [1.0, 2.0],
            "firstMovement_times": [1.1, 2.2],
            "contrastLeft": [0.5, np.nan],
            "contrastRight": [np.nan, 0.25],
            "choice": [-1, 1],
            "feedbackType": [1, -1],
            "probabilityLeft": [0.8, 0.2],
        }
    )
    old_revision = alf / "#2024-01-01#"
    old_revision.mkdir()
    old_table = table.copy()
    old_table["choice"] = [1, 1]
    old_table.to_parquet(old_revision / "_ibl_trials.table.pqt")
    latest_revision = alf / "#2024-02-01#"
    latest_revision.mkdir()
    table.to_parquet(latest_revision / "_ibl_trials.table.pqt")

    _write_array(alf / "_ibl_wheel.timestamps.npy", np.linspace(0.0, 3.0, 31))
    _write_array(alf / "_ibl_wheel.position.npy", np.linspace(0.0, 1.0, 31))
    camera_times = np.linspace(0.0, 3.0, 31)
    _write_array(alf / "_ibl_leftCamera.times.npy", camera_times)
    pd.DataFrame(
        {
            "paw_x": camera_times,
            "paw_y": np.zeros(camera_times.size),
            "paw_likelihood": np.ones(camera_times.size),
        }
    ).to_parquet(alf / "_ibl_leftCamera.dlc.pqt")

    base_probe = alf / "probe00"
    _write_array(base_probe / "spikes.times.npy", [0.1])
    _write_array(base_probe / "spikes.clusters.npy", [0])
    _write_array(base_probe / "clusters.channels.npy", [0])
    sorter = alf / "probe00" / "pykilosort"
    for revision, times in (("#2024-01-01#", [0.2, 0.3]), ("#2024-02-01#", [0.4, 0.5])):
        directory = sorter / revision
        _write_array(directory / "spikes.times.npy", times)
        _write_array(directory / "spikes.clusters.npy", [0, 1])
        _write_array(directory / "clusters.channels.npy", [10, 11])
    probe01 = alf / "probe01"
    _write_array(probe01 / "spikes.times.npy", [0.6])
    _write_array(probe01 / "spikes.clusters.npy", [2])
    _write_array(probe01 / "clusters.cluster_id.npy", [2])
    return cache, session.relative_to(cache)


def _trials():
    return {
        "stimOn_times": np.array([1.0, 2.0, 3.0, 4.0]),
        "firstMovement_times": np.array([1.3, 2.4, 3.2, 4.5]),
        "contrastLeft": np.array([0.5, np.nan, 0.25, np.nan]),
        "contrastRight": np.array([np.nan, 0.5, np.nan, 0.25]),
        "choice": np.array([-1, 1, -1, 1]),
        "feedbackType": np.array([1, -1, 1, 1]),
        "probabilityLeft": np.array([0.8, 0.8, 0.2, 0.2]),
    }


class FakeSource:
    def search_sessions(self, *, limit):
        return ["fake"][:limit]

    def load_trials(self, eid):
        return _trials()

    def load_probe_spikes(self, eid):
        times = np.array([0.8, 0.9, 1.8, 1.95, 2.8, 3.9, 4.4])
        clusters = np.array([0, 1, 0, 1, 0, 1, 0])
        return [ProbeSpikes("probe00", times, clusters, np.array([0, 1]), np.array(["VISp", "MOs"]))]

    def load_wheel(self, eid):
        return {"timestamps": np.linspace(0, 5, 101), "position": np.linspace(0, 1, 101)}

    def load_pose_summary(self, eid, events, *, window_s=(-0.5, 0.0)):
        return np.arange(len(events), dtype=float)

    def session_details(self, eid):
        return {"subject": "mouse-1"}


def test_context_blocks_are_contiguous() -> None:
    values = [0.5, 0.5, 0.2, 0.2, 0.5]
    assert np.array_equal(contiguous_context_blocks(values), [0, 0, 1, 1, 2])
    with pytest.raises(ValueError, match="finite"):
        contiguous_context_blocks([0.5, np.nan])


def test_event_aligned_counts_use_only_pre_event_window() -> None:
    times = np.array([0.6, 0.9, 1.0, 1.1])
    clusters = np.array([0, 1, 0, 1])
    counts, axis, valid = event_aligned_spike_counts(
        times, clusters, np.array([1.0, np.nan]), n_units=2, window_s=(-0.5, 0.0)
    )
    assert counts.shape == (2, 25, 2)
    assert counts[0].sum() == 2
    assert counts[1].sum() == 0
    assert np.array_equal(valid, [True, False])
    assert axis[-1] < 0.0


def test_fake_source_loads_both_preregistered_views() -> None:
    data = load_ibl_trial_data(FakeSource(), "fake")
    assert data.animal_id == "mouse-1"
    assert set(data.activity) == {"stimulus_pre", "movement_pre"}
    assert set(data.view_covariates) == {"stimulus_pre", "movement_pre"}
    assert data.activity["stimulus_pre"].shape == (4, 25, 2)
    assert data.unit_ids.shape == data.regions.shape == (2,)
    np.testing.assert_array_equal(data.unit_ids, ["probe00:0", "probe00:1"])
    assert not data.activity["stimulus_pre"].flags.writeable
    with pytest.raises(TypeError):
        data.activity["extra"] = np.empty(0)  # type: ignore[index]


def test_session_loader_aligns_pose_to_each_neural_view() -> None:
    class RecordingSource(FakeSource):
        def __init__(self):
            self.pose_events = []

        def load_pose_summary(self, eid, events, *, window_s=(-0.5, 0.0)):
            self.pose_events.append((np.array(events, copy=True), window_s))
            return np.asarray(events, dtype=float)

    source = RecordingSource()
    data = load_ibl_trial_data(source, "fake", pre_window_s=(-0.2, 0.0))
    assert len(source.pose_events) == 2
    np.testing.assert_array_equal(source.pose_events[0][0], _trials()["stimOn_times"])
    np.testing.assert_array_equal(
        source.pose_events[1][0], _trials()["firstMovement_times"]
    )
    assert source.pose_events[0][1] == source.pose_events[1][1] == (-0.2, 0.0)
    np.testing.assert_array_equal(
        data.view_covariates["stimulus_pre"]["pose"], _trials()["stimOn_times"]
    )
    np.testing.assert_array_equal(
        data.view_covariates["movement_pre"]["pose"],
        _trials()["firstMovement_times"],
    )


def test_residualizer_is_train_fit_and_requires_complete_nuisances() -> None:
    covariates = build_trial_covariates(
        _trials(),
        wheel={"timestamps": np.linspace(0, 5, 101), "position": np.linspace(0, 1, 101)},
        pose_summary=np.arange(4, dtype=float),
    )
    activity = np.arange(4 * 3, dtype=float).reshape(4, 3)
    model = TrialNuisanceResidualizer(
        ["stimulus", "choice", "wheel", "reward", "reaction_time", "pose"]
    ).fit(covariates.iloc[:3], activity[:3], sample_ids=[0, 1, 2])
    residual = model.transform(covariates.iloc[3:], activity[3:])
    assert residual.shape == (1, 3)
    assert not model.coefficients_.flags.writeable
    assert not model.fit_sample_ids_.flags.writeable
    tuple_ids = [("session", index) for index in range(3)]
    model.fit(covariates.iloc[:3], activity[:3], sample_ids=tuple_ids)
    assert model.fit_sample_ids_[0] == ("session", 0)
    broken = covariates.copy()
    broken.loc[0, "pose"] = np.nan
    with pytest.raises(IBLDataError, match="missing values"):
        TrialNuisanceResidualizer(["pose"]).fit(broken, activity)


def test_probe_spikes_copy_inputs_and_event_binner_rejects_unsorted_times() -> None:
    times = np.array([0.2, 0.1])
    clusters = np.array([1, 0])
    probe = ProbeSpikes(
        "alf/probe00", times, clusters, np.array([0, 1]), np.array(["A", "B"])
    )
    times[:] = 99.0
    np.testing.assert_array_equal(probe.times, [0.1, 0.2])
    np.testing.assert_array_equal(probe.clusters, [0, 1])
    assert not probe.times.flags.writeable
    with pytest.raises(ValueError, match="sorted"):
        event_aligned_spike_counts(
            np.array([0.2, 0.1]),
            np.array([0, 1]),
            np.array([0.3]),
            n_units=2,
        )


def test_trial_covariate_schema_rejects_inconsistent_lengths() -> None:
    broken = _trials()
    broken["choice"] = np.array([1, -1])
    with pytest.raises(IBLDataError, match="inconsistent length"):
        build_trial_covariates(broken)
    both_sides = _trials()
    both_sides["contrastRight"][0] = 0.5
    with pytest.raises(IBLDataError, match="exactly one finite"):
        build_trial_covariates(both_sides)


def test_early_movement_trial_is_explicitly_marked_for_complete_case_filtering() -> None:
    early = _trials()
    early["firstMovement_times"][1] = early["stimOn_times"][1] - 0.1
    covariates = build_trial_covariates(early)
    assert not bool(covariates.loc[1, "timing_valid"])
    assert np.isnan(covariates.loc[1, "reaction_time"])
    assert bool(covariates.loc[0, "timing_valid"])


def test_direct_one_fallback_deduplicates_sorter_collections() -> None:
    class FakeOne:
        def __init__(self):
            self.loaded: list[tuple[str, str]] = []

        def load_object(self, eid, obj, *, collection, attribute=None):
            self.loaded.append((obj, collection))
            if obj == "spikes":
                return {"times": np.array([0.1, 0.2]), "clusters": np.array([0, 1])}
            return {
                "cluster_id": np.array([0, 1]),
                "acronyms": np.array(["VISp", "MOs"]),
            }

    source = OneAPISource.__new__(OneAPISource)
    source.one = FakeOne()
    probes = source._load_probe_spikes_direct(
        "eid", ["alf/probe00", "alf/probe00/pykilosort"]
    )
    assert len(probes) == 1
    assert probes[0].collection == "alf/probe00/pykilosort"
    assert set(source.one.loaded) == {
        ("spikes", "alf/probe00/pykilosort"),
        ("clusters", "alf/probe00/pykilosort"),
    }


def test_session_loader_requires_animal_for_statistical_provenance() -> None:
    class MissingAnimalSource(FakeSource):
        def session_details(self, eid):
            return {}

    with pytest.raises(IBLDataError, match="animal-level"):
        load_ibl_trial_data(MissingAnimalSource(), "fake")


def test_one_search_uses_current_dataset_files_and_checks_pose_separately() -> None:
    class SearchOne:
        def __init__(self):
            self.kwargs = None

        def search(self, **kwargs):
            self.kwargs = kwargs
            return ["a", "b", "c"]

        def list_datasets(self, eid):
            camera = "leftCamera" if eid != "a" else "unrelated"
            return [
                f"alf/_ibl_{camera}.times.npy",
                f"alf/_ibl_{camera}.dlc.pqt",
            ]

    source = OneAPISource.__new__(OneAPISource)
    source.one = SearchOne()
    assert source.search_sessions(limit=2) == ["b", "c"]
    assert "datasets" in source.one.kwargs
    assert "dataset" not in source.one.kwargs
    assert source.one.kwargs["datasets"] == [
        "_ibl_trials.table.pqt",
        "spikes.times.npy",
        "spikes.clusters.npy",
        "_ibl_leftCamera.dlc.pqt",
    ]


def test_one_source_uses_configurable_public_auth_without_retaining_credentials(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    class FakeOne:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    one_package = types.ModuleType("one")
    one_api = types.ModuleType("one.api")
    one_api.ONE = FakeOne
    monkeypatch.setitem(sys.modules, "one", one_package)
    monkeypatch.setitem(sys.modules, "one.api", one_api)

    source = OneAPISource(cache_dir=tmp_path)
    assert captured["username"] == "intbrainlab"
    assert captured["password"] == "international"
    assert set(vars(source)) == {"one"}
    with pytest.raises(ValueError, match="both"):
        OneAPISource(cache_dir=tmp_path, username="private", password=None)


def test_one_pose_summary_is_derived_from_camera_dlc_and_network_errors_propagate() -> None:
    class PoseOne:
        def load_object(self, eid, obj, **kwargs):
            if obj == "wheel":
                raise RuntimeError("network down")
            times = np.arange(0.0, 3.0, 0.1)
            return {
                "times": times,
                "dlc": pd.DataFrame(
                    {
                        "paw_r_x": times,
                        "paw_r_y": np.zeros(times.size),
                        "paw_r_likelihood": np.ones(times.size),
                    }
                ),
            }

    source = OneAPISource.__new__(OneAPISource)
    source.one = PoseOne()
    pose = source.load_pose_summary("eid", np.array([1.0, 2.0]))
    np.testing.assert_allclose(pose, [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    with pytest.raises(RuntimeError, match="network down"):
        source.load_wheel("eid")


def test_cached_source_discovers_latest_trials_and_local_covariates(tmp_path) -> None:
    cache, relative = _cached_alf_fixture(tmp_path)
    source = CachedIBLSessionSource(
        cache_dir=cache, session_paths={"eid-local": relative}
    )
    assert source.search_sessions(limit=1) == ["eid-local"]
    trials = source.load_trials("eid-local")
    np.testing.assert_array_equal(trials["choice"], [-1, 1])
    np.testing.assert_array_equal(trials["response_times"], [1.2, 2.3])
    wheel = source.load_wheel("eid-local")
    assert wheel is not None
    assert set(wheel) == {"position", "timestamps"}
    pose = source.load_pose_summary("eid-local", np.array([1.0, 2.0]))
    assert pose is not None
    np.testing.assert_allclose(pose, np.repeat(1.0 / np.sqrt(2.0), 2))
    assert source.session_details("eid-local") == {"subject": "mouse-cache"}


def test_cached_source_imports_alf_only_when_data_are_loaded(
    monkeypatch, tmp_path
) -> None:
    cache, relative = _cached_alf_fixture(tmp_path)

    def unavailable():
        raise IBLDependencyError("optional dependency unavailable")

    monkeypatch.setattr(CachedIBLSessionSource, "_alf_io", staticmethod(unavailable))
    source = CachedIBLSessionSource(
        cache_dir=cache, session_paths={"eid-local": relative}
    )
    assert source.search_sessions(limit=1) == ["eid-local"]
    assert source.session_details("eid-local") == {"subject": "mouse-cache"}
    with pytest.raises(IBLDependencyError, match="optional dependency"):
        source.load_trials("eid-local")


def test_cached_source_selects_one_deepest_latest_collection_per_probe(tmp_path) -> None:
    cache, relative = _cached_alf_fixture(tmp_path)
    source = CachedIBLSessionSource(
        cache_dir=cache, session_paths={"eid-local": relative}
    )
    probes = source.load_probe_spikes("eid-local")
    assert [probe.collection for probe in probes] == [
        "alf/probe00/pykilosort",
        "alf/probe01",
    ]
    np.testing.assert_array_equal(probes[0].times, [0.4, 0.5])
    np.testing.assert_array_equal(probes[0].unit_ids, [0, 1])
    np.testing.assert_array_equal(probes[1].unit_ids, [2])


def test_cached_source_rejects_escape_and_unknown_session(tmp_path) -> None:
    cache, relative = _cached_alf_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(IBLDataError, match="escapes"):
        CachedIBLSessionSource(
            cache_dir=cache, session_paths={"escaped": Path("..") / "outside"}
        )
    with pytest.raises(ValueError, match="relative"):
        CachedIBLSessionSource(
            cache_dir=cache, session_paths={"absolute": outside.resolve()}
        )
    source = CachedIBLSessionSource(
        cache_dir=cache, session_paths={"eid-local": relative}
    )
    with pytest.raises(IBLDataError, match="not present"):
        source.load_trials("unknown")


def test_cached_source_reports_missing_required_local_objects(tmp_path) -> None:
    cache = tmp_path / "cache"
    session = cache / "lab" / "Subjects" / "mouse" / "2024-01-02" / "001"
    (session / "alf").mkdir(parents=True)
    relative = session.relative_to(cache)
    source = CachedIBLSessionSource(
        cache_dir=cache, session_paths={"missing": relative}
    )
    with pytest.raises(IBLDataError, match="_ibl_trials.table.pqt"):
        source.load_trials("missing")
    with pytest.raises(IBLDataError, match="spike sorting"):
        source.load_probe_spikes("missing")
    with pytest.raises(ValueError, match="1-5"):
        CachedIBLSessionSource(
            cache_dir=cache,
            session_paths={str(index): relative for index in range(6)},
        )


def test_cached_source_rejects_alf_directory_symlink_escape(tmp_path) -> None:
    cache = tmp_path / "cache"
    session = cache / "lab" / "Subjects" / "mouse" / "2024-01-02" / "001"
    session.mkdir(parents=True)
    outside_alf = tmp_path / "outside-alf"
    outside_alf.mkdir()
    _symlink_or_skip(session / "alf", outside_alf, directory=True)
    source = CachedIBLSessionSource(
        cache_dir=cache, session_paths={"escaped": session.relative_to(cache)}
    )
    with pytest.raises(IBLDataError, match="escapes"):
        source.load_trials("escaped")


def test_cached_source_rejects_revision_and_dataset_symlink_escapes(tmp_path) -> None:
    cache, relative = _cached_alf_fixture(tmp_path)
    alf = cache / relative / "alf"
    outside_revision = tmp_path / "outside-revision"
    outside_revision.mkdir()
    pd.DataFrame({"choice": [1]}).to_parquet(
        outside_revision / "_ibl_trials.table.pqt"
    )
    _symlink_or_skip(alf / "#2099-01-01#", outside_revision, directory=True)
    source = CachedIBLSessionSource(
        cache_dir=cache, session_paths={"escaped": relative}
    )
    with pytest.raises(IBLDataError, match="revision directory.*escapes"):
        source.load_trials("escaped")

    isolated = cache / "lab" / "Subjects" / "mouse-file" / "2024-01-03" / "001"
    isolated_alf = isolated / "alf"
    isolated_alf.mkdir(parents=True)
    outside_table = tmp_path / "outside-table.pqt"
    pd.DataFrame({"choice": [1]}).to_parquet(outside_table)
    _symlink_or_skip(
        isolated_alf / "_ibl_trials.table.pqt", outside_table, directory=False
    )
    file_source = CachedIBLSessionSource(
        cache_dir=cache, session_paths={"file-escaped": isolated.relative_to(cache)}
    )
    with pytest.raises(IBLDataError, match="ALF dataset.*escapes"):
        file_source.load_trials("file-escaped")


def test_cached_source_rejects_probe_file_symlink_escape(tmp_path) -> None:
    cache, relative = _cached_alf_fixture(tmp_path)
    alf = cache / relative / "alf"
    probe = alf / "probe02"
    probe.mkdir()
    outside_times = tmp_path / "outside-spikes.times.npy"
    _write_array(outside_times, [0.7])
    _symlink_or_skip(probe / "spikes.times.npy", outside_times, directory=False)
    _write_array(probe / "spikes.clusters.npy", [0])
    source = CachedIBLSessionSource(
        cache_dir=cache, session_paths={"escaped": relative}
    )
    with pytest.raises(IBLDataError, match="spikes.times dataset.*escapes"):
        source.load_probe_spikes("escaped")
