import copy
from pathlib import Path

from bxjudge.two_bey_integration import reconstruct_round_tracks
from bxjudge.two_bey_tracking import load_policy
from bxjudge.types import Detection, Track


def t(tid, frames, x, cls="bey"):
    return Track(
        tid, cls, [Detection(f, cls, (x + f, 10, x + f + 10, 20), 0.9) for f in frames]
    )


def test_round_reconstruction_produces_two_raw_tracks_with_collision_free_ids():
    tracks = [
        t(1, [0, 1], 10),
        t(3, [3, 4], 10),
        t(2, range(5), 100),
        t(4, range(5), 50, "launcher"),
    ]
    original = copy.deepcopy(tracks)
    features = {
        (d.frame, d.box): ([1.0, 0.0] if tr.track_id in (1, 3) else [0.0, 1.0])
        for tr in tracks
        if tr.cls == "bey"
        for d in tr.dets
    }
    result = reconstruct_round_tracks(
        tracks,
        0,
        4,
        30,
        (200, 100),
        features,
        load_policy(Path("config/two_bey_tracking.yaml")),
        100,
    )
    assert tracks == original
    assert {tr.track_id for tr in result.tracks} == {101, 102}
    assert len(result.tracks) == 2
    assert result.members == {101: [1, 3], 102: [2]}
    assert [d.frame for d in result.tracks[0].dets] == [0, 1, 3, 4]
    assert 2 not in [d.frame for d in result.tracks[0].dets]
    assert result.record["max_physical_tops_per_frame"] == 2
    assert result.record["raw_input_bey_observations"] == 9
    assert all(d.cls == "bey" for tr in result.tracks for d in tr.dets)


def test_a_new_round_has_fresh_physical_ownership():
    tracks = [t(3, [10, 11], 10)]
    result = reconstruct_round_tracks(
        tracks,
        10,
        11,
        30,
        (200, 100),
        {},
        load_policy(Path("config/two_bey_tracking.yaml")),
        200,
    )
    assert result.tracks == [] and result.members == {}
    assert result.record["segment_status"] == "pair_not_confirmed"
    assert result.record["tentative_entities"] == 1
    assert [d["native_id"] for d in result.record["unassigned"]] == [3, 3]
    assert all(d["reason"] == "pair_not_confirmed" for d in result.record["unassigned"])


def test_two_sequential_objects_never_observed_together_do_not_confirm_a_battle():
    result = reconstruct_round_tracks(
        [t(1, [0, 1], 10), t(2, [5, 6], 100)],
        0,
        6,
        30,
        (200, 100),
        {},
        load_policy(Path("config/two_bey_tracking.yaml")),
        100,
    )
    assert result.tracks == []
    assert result.record["pair_confirmed_at_frame"] is None
    assert (
        result.record["raw_input_bey_observations"]
        == len(result.record["unassigned"])
        == 4
    )


def test_pair_confirmation_requires_consecutive_actual_source_frames():
    tracks = [t(1, [0, 2, 4], 10), t(2, [0, 2, 4], 100)]
    r = reconstruct_round_tracks(
        tracks,
        0,
        4,
        30,
        (200, 100),
        {},
        load_policy(Path("config/two_bey_tracking.yaml")),
        100,
    )
    assert r.tracks == [] and r.record["pair_confirmed_at_frame"] is None
    assert len(r.record["hypothesis_observations"]) == 6


def test_cli_forwards_the_explicit_two_bey_policy(tmp_path, monkeypatch):
    from bxjudge import pipeline
    from bxjudge.cli import main

    seen = {}
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **kw: seen.update(kw))
    assert (
        main(
            [
                "run",
                str(tmp_path),
                "--two-bey-tracking",
                "config/two_bey_tracking.yaml",
                "--dynamic-calibration",
                "config/dynamic_calibration.yaml",
                "--no-vlm",
            ]
        )
        == 0
    )
    assert seen["two_bey_config"] == Path("config/two_bey_tracking.yaml")


def test_two_bey_mode_requires_recorded_dynamic_geometry(tmp_path, th, rules):
    import pytest

    from bxjudge.pipeline import judge_video
    from tests.helpers import shipped_autocalib
    from tests.test_pipeline import info

    with pytest.raises(ValueError, match="requires dynamic"):
        judge_video(
            info(),
            tmp_path,
            lambda *a: [],
            rules,
            th,
            out_dir=tmp_path / "out",
            autocalib=shipped_autocalib(),
            two_bey_config=Path("config/two_bey_tracking.yaml"),
        )
    assert not (tmp_path / "out").exists()


def pipeline_setup(monkeypatch, tmp_path, tracks):
    from bxjudge import pipeline
    from bxjudge.autocalib import Alignment
    from bxjudge.dynamic_calibration import DynamicCalibration, GeometrySample
    from bxjudge.identity import IdentityAnchors
    from tests.test_pipeline import calibrated

    calibrated(monkeypatch, tmp_path, tracks=tracks)
    eye = Alignment(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), 1.0, 0.0, 0.0, 1.0, 0.0
    )

    def update(self, frame, image):
        self.samples[frame] = GeometrySample(frame, eye, 0, "tracking")
        return self.samples[frame]

    monkeypatch.setattr(DynamicCalibration, "update", update)
    monkeypatch.setattr(
        pipeline, "anchor_sides", lambda *a, **k: IdentityAnchors({1: "X", 2: "B"}, {})
    )
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"synthetic-source-only")
    return source


def invoke_two_pipeline(source, tmp_path, rules, th):
    from bxjudge import pipeline
    from bxjudge.dynamic_calibration import DEFAULT_DYNAMIC_CONFIG
    from tests.helpers import shipped_autocalib
    from tests.test_pipeline import info

    return pipeline.judge_video(
        info(source),
        tmp_path,
        lambda *a: [],
        rules,
        th,
        out_dir=tmp_path / "out",
        autocalib=shipped_autocalib(),
        dynamic_config=DEFAULT_DYNAMIC_CONFIG,
        two_bey_config=Path("config/two_bey_tracking.yaml"),
    )


def test_single_top_candidate_preserves_population_without_publishing_a_third_id(
    tmp_path, monkeypatch, th, rules
):
    import csv
    import json

    from bxjudge.rules import is_not_a_round
    from tests.helpers import make_track

    tracks = [make_track(1, "bey", {f: (350, 360) for f in range(35, 75)})]
    source = pipeline_setup(monkeypatch, tmp_path, tracks)
    [r] = invoke_two_pipeline(source, tmp_path, rules, th)
    assert r.needs_review and r.finish is None and r.winner is None
    assert not is_not_a_round(r.review_reasons)
    assert r.launch is None and r.motion_evidence is None
    assert "two_bey_singleton_segment" in r.review_reasons
    data = json.loads((tmp_path / "out" / r.geometry_evidence["path"]).read_bytes())[
        "two_bey_tracking"
    ]["rounds"]["1"]
    assert data["observed_entities"] == 0 and data["tentative_entities"] == 1
    assert len(data["hypothesis_observations"]) == len(data["unassigned"]) == 40
    assert data["observations"] == [] and data["members"] == {}
    csv_path = next((tmp_path / "out/trajectories").glob("*.csv"))
    assert not any(d["cls"] == "bey" for d in csv.DictReader(csv_path.open()))


def test_unassigned_unknown_native_observations_still_block_a_winner(
    tmp_path, monkeypatch, th, rules
):
    from bxjudge import pipeline
    from bxjudge.finish import FinishResult
    from tests.helpers import make_track

    tracks = [
        make_track(tid, "bey", {f: (x, 360) for f in range(35, 75)})
        for tid, x in [(1, 350), (2, 900), (3, 700)]
    ]
    source = pipeline_setup(monkeypatch, tmp_path, tracks)
    monkeypatch.setattr(
        pipeline, "detect_finish", lambda *a: FinishResult("over", "B", 70, 0.95, [])
    )
    [r] = invoke_two_pipeline(source, tmp_path, rules, th)
    assert r.finish is None and r.winner is None
    assert "two_bey_unassigned_identity_unknown" in r.review_reasons


def test_original_pre_phase_tracks_survive_excluded_held_processing(
    tmp_path, monkeypatch, th, rules
):
    import json

    from tests.helpers import make_track

    tracks = [
        make_track(tid, "bey", {f: (x, 360) for f in range(35, 75)})
        for tid, x in [(1, 350), (2, 900)]
    ]
    tracks += [
        make_track(4, "bey", {f: (700, 300) for f in range(35, 75)}),
        make_track(5, "launcher", {f: (700, 300) for f in range(35, 75)}),
    ]
    source = pipeline_setup(monkeypatch, tmp_path, tracks)
    [r] = invoke_two_pipeline(source, tmp_path, rules, th)
    record = json.loads((tmp_path / "out" / r.geometry_evidence["path"]).read_bytes())
    assert 4 in record["phase"]["excluded_held"]
    raw = {t["track_id"]: t for t in record["two_bey_tracking"]["raw_source_tracks"]}
    assert len(raw[4]["dets"]) == 40
    assert set(raw) == {1, 2, 4, 5}


def test_no_round_still_saves_geometry_and_all_source_observations(
    tmp_path, monkeypatch, th, rules
):
    import json

    from tests.helpers import make_track

    tracks = [
        make_track(4, "bey", {f: (700, 300) for f in range(35, 75)}),
        make_track(5, "launcher", {f: (700, 300) for f in range(35, 75)}),
    ]
    source = pipeline_setup(monkeypatch, tmp_path, tracks)
    [r] = invoke_two_pipeline(source, tmp_path, rules, th)
    assert "no_round" in r.review_reasons
    assert r.geometry_evidence is not None
    record = json.loads((tmp_path / "out" / r.geometry_evidence["path"]).read_bytes())
    assert record["two_bey_tracking"]["rounds"] == {}
    assert {t["track_id"] for t in record["two_bey_tracking"]["raw_source_tracks"]} == {
        4,
        5,
    }


def test_held_prefix_is_preserved_for_tracking_but_not_sent_as_free_motion(
    tmp_path, monkeypatch, th, rules
):
    import json

    from bxjudge import pipeline
    from tests.helpers import make_track

    tracks = [
        make_track(1, "bey", {f: (350, 360) for f in range(35, 75)}),
        make_track(2, "bey", {f: (900, 360) for f in range(35, 75)}),
        make_track(
            5,
            "launcher",
            {f: ((350, 360) if f < 46 else (100, 100)) for f in range(35, 75)},
        ),
    ]
    source = pipeline_setup(monkeypatch, tmp_path, tracks)
    seen = []
    original = pipeline.analyze_launch

    def analyze(current, *args, **kwargs):
        seen.extend(current)
        return original(current, *args, **kwargs)

    monkeypatch.setattr(pipeline, "analyze_launch", analyze)
    [r] = invoke_two_pipeline(source, tmp_path, rules, th)
    record = json.loads((tmp_path / "out" / r.geometry_evidence["path"]).read_bytes())
    phase_release = record["phase"]["releases"]["1"]
    rows = record["two_bey_tracking"]["rounds"]["1"]["observations"]
    assert min(d["frame"] for d in rows if d["native_id"] == 1) == 35
    x = next(
        t
        for t in seen
        if t.center_at(phase_release) is not None
        and t.center_at(phase_release)[0] < 500
    )
    assert x.first_frame() == phase_release
