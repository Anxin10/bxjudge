import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bxjudge import pipeline
from bxjudge.dynamic_calibration import DEFAULT_DYNAMIC_CONFIG
from tests.helpers import cal_with, make_track, shipped_autocalib, stadium_scene
from tests.test_pipeline import _snapshot, info


@pytest.mark.parametrize('two_bey',[False,True])
def test_dynamic_pipeline_uses_reference_coordinates_and_keeps_raw_evidence(
    tmp_path, monkeypatch, th, rules,two_bey
):
    ref, c = stadium_scene(size=(640, 360))
    cal_with(ref, tmp_path)
    c.ref_frame = str(tmp_path / "ref.png")
    cv2.imwrite(c.ref_frame, ref)
    frame, _ = stadium_scene(size=(640, 360), shift=(12, -8))
    monkeypatch.setattr(
        pipeline, "iter_frames", lambda *a, **k: iter((f, frame) for f in range(80))
    )
    monkeypatch.setattr(pipeline, "select_snapshot", lambda *a, video: _snapshot(c))
    monkeypatch.setattr(pipeline, "write_preview", lambda *a: None)
    monkeypatch.setattr(
        pipeline,
        "iter_stadium_displacement",
        lambda frames, *a, **k: ((f, img, 5.0) for f, img in frames),
    )
    tracks = [
        make_track(1, "bey", {f: (250 + 12, 220 - 8) for f in range(35, 75)}),
        make_track(2, "bey", {f: (400 + 12, 220 - 8) for f in range(35, 75)}),
    ]

    def tracking(frames, *a, **k):
        list(frames)
        return tracks

    monkeypatch.setattr(pipeline, "track_frames", tracking)
    observed = []
    original = pipeline.analyze_launch

    def analyze(current, *a, **k):
        observed.extend(current)
        return original(current, *a, **k)

    monkeypatch.setattr(pipeline, "analyze_launch", analyze)
    # Even an apparently confident downstream event must not choose the
    # winner while a real observed top has no launch identity anchor.
    from bxjudge.finish import FinishResult

    monkeypatch.setattr(
        pipeline, "detect_finish", lambda *a: FinishResult("over", "B", 70, 0.95, [])
    )
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source bytes")
    [r] = pipeline.judge_video(
        info(source),
        tmp_path,
        lambda *a: [],
        rules,
        th,
        out_dir=tmp_path / "out",
        autocalib=shipped_autocalib(),
        dynamic_config=DEFAULT_DYNAMIC_CONFIG,
        **({'two_bey_config':__import__('pathlib').Path('config/two_bey_tracking.yaml')} if two_bey else {}),
    )
    assert np.linalg.norm(np.array(observed[0].center_at(35)) - [250, 220]) < 1
    assert r.geometry_evidence["coordinates"] == "reference_pixels"
    record = tmp_path / "out" / r.geometry_evidence["path"]
    assert record.is_file()
    data = json.loads(record.read_text())
    assert data["calibration_id"] == r.calibration_id
    assert data["identity"]["sides"] == {}
    for track in observed:
        assert f'identity_unanchored@{track.track_id}' in r.review_reasons
    if two_bey:
        tracking=data['two_bey_tracking']['rounds']['1']
        assert tracking['configured_slots']==2 and tracking['observed_entities']==2
        assert tracking['interpolation_used_for_judging'] is False
        assert len(observed)==2
        assert 'two_bey_tracking_requires_review' in r.review_reasons
    assert r.needs_review and r.finish is None and r.winner is None
    values = r.motion_evidence["displacements"]
    if two_bey:
        # Observations without confirmed releases preserve their source boxes
        # but do not define an evidentiary touch window or a landing deadline.
        assert data["launch_timing"]["round_releases"]["1"] == {}
        assert len(data["two_bey_tracking"]["raw_source_tracks"]) == 2
        assert r.motion_evidence["landing_deadlines"] == {"X": None, "B": None}
        assert r.motion_evidence["window"] == [0, 0] and values == {}
        assert r.launch.landed == {"X": None, "B": None}
    else:
        # This mode measures a provisional launch window; undecoded frames
        # remain unknown.
        assert all(v == (5.0 if int(f) < 80 else None) for f, v in values.items())
        assert 5.0 in values.values()
    csv = next((tmp_path / "out/trajectories").glob("*.csv")).read_text()
    assert "262.0,212.0" in csv


def test_missing_geometry_cannot_look_like_a_confident_exit(
    tmp_path, monkeypatch, th, rules
):
    from tests.test_pipeline import _kw, calibrated

    tracks = [
        make_track(1, "bey", {f: (350, 360) for f in range(120)}),
        make_track(2, "bey", {f: (900, 360) for f in range(120)}),
    ]
    calibrated(monkeypatch, tmp_path, tracks=tracks)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source bytes")
    [r] = pipeline.judge_video(
        info(source),
        tmp_path,
        lambda *a: [],
        rules,
        th,
        **_kw(tmp_path),
        dynamic_config=DEFAULT_DYNAMIC_CONFIG,
    )
    assert r.needs_review
    assert "calibration_geometry_missing" in r.review_reasons
    assert r.finish is None and r.winner is None


def test_evidence_frame_uses_recorded_dynamic_geometry_and_hides_unknown(
    tmp_path, monkeypatch, th
):
    from dataclasses import asdict

    from bxjudge import review_ui
    from bxjudge.autocalib import Alignment
    from tests.helpers import make_calibration

    c = make_calibration()
    image = np.full((360, 640, 3), 128, np.uint8)
    monkeypatch.setattr(
        review_ui, "probe", lambda *a: type("Info", (), {"fps": 30, "duration": 1.0})()
    )
    monkeypatch.setattr(
        review_ui,
        "iter_frames",
        lambda *a, **k: ((f, image.copy()) for f in range(2)),
    )
    a = asdict(
        Alignment(
            ((1.0, 0.0, 12.0), (0.0, 1.0, -4.0), (0.0, 0.0, 1.0)),
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
        )
    )
    record = {
        "reference_size": [640, 360],
        "samples": {
            0: {"frame": 0, "alignment": a, "status": "tracking"},
            1: {"frame": 1, "alignment": None},
        },
    }
    rgb, _, _ = review_ui.evidence_frame(
        tmp_path / "x.mp4", 0, c, th, geometry_record=record
    )
    # y=20 stadium edge moves to16; inspect green relative to red, not raw BGR order.
    assert rgb[16, 250, 1] > rgb[16, 250, 0]
    rgb, _, _ = review_ui.evidence_frame(
        tmp_path / "x.mp4", 1 / 30, c, th, geometry_record=record
    )
    assert tuple(rgb[16, 250]) == (128, 128, 128)


@pytest.mark.parametrize("gap_kind", ["lost", "bridged"])
def test_missing_geometry_after_pocket_onset_cannot_create_terminal_entry(
    tmp_path, monkeypatch, th, rules, gap_kind
):
    from bxjudge.autocalib import Alignment
    from bxjudge.dynamic_calibration import DynamicCalibration, GeometrySample
    from tests.test_pipeline import _kw, calibrated

    eye = Alignment(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), 1.0, 0.0, 0.0, 1.0, 0.0
    )
    tracks = [
        make_track(
            1, "bey", {f: ((150, 500) if f == 20 else (350, 360)) for f in range(40)}
        ),
        make_track(2, "bey", {f: (900, 360) for f in range(40)}),
    ]
    calibrated(monkeypatch, tmp_path, tracks=tracks)

    def update(self, frame, image):
        sample = GeometrySample(
            frame,
            eye if frame <= 20 or gap_kind == "bridged" else None,
            0,
            "tracking" if frame <= 20 else gap_kind,
        )
        self.samples[frame] = sample
        self.diagnostics[frame] = {
            "tracking": {"reason": None if frame <= 20 else "synthetic_geometry_missing"},
            "relocation": None,
            "reason": None if frame <= 20 else "synthetic_geometry_missing",
        }
        return sample

    monkeypatch.setattr(DynamicCalibration, "update", update)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    [r] = pipeline.judge_video(
        info(source),
        tmp_path,
        lambda *a: [],
        rules,
        th,
        **_kw(tmp_path),
        dynamic_config=DEFAULT_DYNAMIC_CONFIG,
    )
    assert "calibration_geometry_missing" in r.review_reasons
    assert r.finish is None and r.winner is None


def test_reviewed_static_row_does_not_force_dynamic_rejudge_forever(
    tmp_path, monkeypatch, rules
):
    from tests.test_pipeline import _clip_setup, _confirm_reviewed, _JudgeSpy, _run

    videos, calib, out = _clip_setup(tmp_path, shift=(0, 0), seconds=2)
    _run(videos, calib, out, rules)
    _confirm_reviewed(out, "clip.mp4", 1, rules)
    original = (out / "rounds.json").read_bytes()
    spy = _JudgeSpy(monkeypatch)
    _run(videos, calib, out, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG)
    assert spy.videos == {"clip.mp4"}
    assert (out / "rounds.json").read_bytes() == original
    spy.videos.clear()
    _run(videos, calib, out, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG)
    assert spy.videos == set()
    assert (out / "rounds.json").read_bytes() == original


def test_relocation_before_launch_still_requires_candidate_review(
    tmp_path, monkeypatch, th, rules
):
    from bxjudge.autocalib import Alignment
    from bxjudge.dynamic_calibration import DynamicCalibration, GeometrySample
    from tests.test_pipeline import _kw, calibrated

    eye = Alignment(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), 1.0, 0.0, 0.0, 1.0, 0.0
    )
    tracks = [
        make_track(1, "bey", {f: (350, 360) for f in range(35, 75)}),
        make_track(2, "bey", {f: (900, 360) for f in range(35, 75)}),
    ]
    calibrated(monkeypatch, tmp_path, tracks=tracks)

    def update(self, frame, image):
        if frame == 10:
            self.segments.append(
                {
                    "id": 1,
                    "start_frame": 10,
                    "kind": "relocated",
                    "homography": np.eye(3).tolist(),
                    "confirmed_frames": [5, 10],
                    "score": 0.9,
                }
            )
        sample = GeometrySample(frame, eye, 1 if frame >= 10 else 0, "tracking")
        self.samples[frame] = sample
        self.diagnostics[frame] = {
            "tracking": {"reason": None}, "relocation": None, "reason": None,
        }
        return sample

    monkeypatch.setattr(DynamicCalibration, "update", update)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    [r] = pipeline.judge_video(
        info(source),
        tmp_path,
        lambda *a: [],
        rules,
        th,
        **_kw(tmp_path),
        dynamic_config=DEFAULT_DYNAMIC_CONFIG,
    )
    assert "calibration_relocated" in r.review_reasons


def test_raw_motion_preparation_overlaps_dynamic_alignment(tmp_path, monkeypatch, th, rules):
    from threading import Event, get_ident

    raw_second = Event()
    alignment_first = Event()
    raw_threads = set()
    geometry_threads = set()
    ref, c = stadium_scene(size=(640, 360))
    c.ref_frame = str(tmp_path / "reference.png")
    assert cv2.imwrite(c.ref_frame, ref)
    monkeypatch.setattr(
        pipeline, "iter_frames", lambda *a, **k: iter((f, ref) for f in range(40))
    )
    monkeypatch.setattr(pipeline, "select_snapshot", lambda *a, video: _snapshot(c))
    monkeypatch.setattr(pipeline, "write_preview", lambda *a: None)

    def raw(frames, *args, **kwargs):
        for f, image in frames:
            raw_threads.add(get_ident())
            if f == 1:
                raw_second.set()
                assert alignment_first.wait(2)
            yield f, image, 0.0

    original = pipeline.DynamicCalibration.update

    def geometry(self, f, image):
        geometry_threads.add(get_ident())
        if f == 0:
            alignment_first.set()
            assert raw_second.wait(2)
        return original(self, f, image)

    monkeypatch.setattr(pipeline, "iter_stadium_displacement", raw)
    monkeypatch.setattr(pipeline.DynamicCalibration, "update", geometry)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"synthetic source identity")
    [result] = pipeline.judge_video(
        info(source), tmp_path, lambda *a: [], rules, th,
        out_dir=tmp_path / "out", autocalib=shipped_autocalib(),
        dynamic_config=DEFAULT_DYNAMIC_CONFIG,
    )
    assert "analysis_error" not in result.review_reasons
    assert raw_threads and geometry_threads and raw_threads.isdisjoint(geometry_threads)
    record = json.loads(next((tmp_path / "out/geometry").glob("*.json")).read_text())
    assert [s["frame"] for s in record["samples"]] == list(range(40))


# Codex task-7 speed re-review (2026-10-29): the raw-motion stage closes its
# measurement generator, so an error raised on close is forwarded, never dropped
# as an ignored finalizer exception.
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_raw_motion_stage_closes_its_measurement_explicitly(tmp_path, monkeypatch, th, rules):
    import gc

    ref, c = stadium_scene(size=(640, 360))
    c.ref_frame = str(tmp_path / "reference.png")
    assert cv2.imwrite(c.ref_frame, ref)
    monkeypatch.setattr(
        pipeline, "iter_frames", lambda *a, **k: iter((f, ref) for f in range(40))
    )
    monkeypatch.setattr(pipeline, "select_snapshot", lambda *a, video: _snapshot(c))
    monkeypatch.setattr(pipeline, "write_preview", lambda *a: None)
    closed = []

    def raw(frames, *args, **kwargs):
        try:
            for f, image in frames:
                yield f, image, 0.0
        except GeneratorExit:
            closed.append(True)
            raise RuntimeError("decode failed") from None

    def detector(image, f):
        if f == 3:
            raise ValueError("detector stopped")
        return []

    monkeypatch.setattr(pipeline, "iter_stadium_displacement", raw)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"synthetic source identity")
    # The detector failure is the primary error; the close error rides as its cause
    # chain inside prefetched, never as an ignored finalizer exception.
    with pytest.raises(ValueError, match="detector stopped"):
        pipeline.judge_video(
            info(source), tmp_path, detector, rules, th,
            out_dir=tmp_path / "out", autocalib=shipped_autocalib(),
            dynamic_config=DEFAULT_DYNAMIC_CONFIG,
        )
    gc.collect()
    assert closed == [True]


# ---- keypoint stadiums and camera modes (plan 2026-11-12-dynamic-calibration, Task 6) ----


def _camera_policy():
    """The shipped camera policy with a 1 s refind interval (these tests count refind frames;
    the shipped interval is a calibrated value)."""
    from bxjudge.camera_motion import load_camera_policy

    policy = load_camera_policy()
    policy["handheld"] = {**policy["handheld"], "refind_s": 1.0, "max_unchecked_s": 3.0}
    return policy


class _FakeLocator:
    """Locates the stadium exactly where the calibration puts it, every time."""

    last_reason = None

    def __init__(self, c):
        self.c = c
        self.calls = 0

    def locate(self, image):
        from bxjudge.stadium_locator import LocatedStadium, alignment_from_homography

        self.calls += 1
        return LocatedStadium(np.eye(3), alignment_from_homography(self.c, np.eye(3), 0.9), None)


def _scene_setup(tmp_path, monkeypatch, *, select=True):
    """80 still frames of the synthetic stadium, two tops in the round, a confident finish."""
    from bxjudge.finish import FinishResult

    ref, c = stadium_scene(size=(640, 360))
    c.ref_frame = str(tmp_path / "ref.png")
    cv2.imwrite(c.ref_frame, ref)
    monkeypatch.setattr(pipeline, "iter_frames", lambda *a, **k: iter((f, ref) for f in range(80)))
    monkeypatch.setattr(pipeline, "select_snapshot",
                        lambda *a, video: _snapshot(c) if select else None)
    monkeypatch.setattr(pipeline, "write_preview", lambda *a: None)
    monkeypatch.setattr(pipeline, "iter_stadium_displacement",
                        lambda frames, *a, **k: ((f, img, 0.0) for f, img in frames))
    tracks = [
        make_track(1, "bey", {f: (250, 220) for f in range(35, 75)}),
        make_track(2, "bey", {f: (400, 220) for f in range(35, 75)}),
    ]

    def tracking(frames, *a, **k):
        list(frames)
        return tracks

    monkeypatch.setattr(pipeline, "track_frames", tracking)
    monkeypatch.setattr(pipeline, "detect_finish",
                        lambda *a: FinishResult("over", "B", 70, 0.95, []))
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source bytes")
    return c, source


def _tools():
    """``StadiumTools`` stand-in: only the finder config is read by the pipeline itself."""
    from types import SimpleNamespace

    from bxjudge.stadium_finder import load_stadium_finder_config

    return SimpleNamespace(cfg=load_stadium_finder_config())


def _judge(source, tmp_path, th, rules, **kw):
    return pipeline.judge_video(info(source), tmp_path, lambda *a: [], rules, th,
                                out_dir=tmp_path / "out", autocalib=shipped_autocalib(), **kw)


def _row(r):
    from dataclasses import asdict

    return asdict(r)


class _Found:
    """The parts of a ``FoundStadium`` the pipeline records."""

    def __init__(self, method="keypoints", frame_index=42, agreeing=2, visibility=None):
        from types import SimpleNamespace

        self.method, self.frame_index, self.agreeing_frames = method, frame_index, agreeing
        self.zone_visibility = visibility or {"over_left": 1.0, "over_right": 1.0,
                                              "xtreme": 1.0}
        self.fit = SimpleNamespace(coverage=0.9, precision=0.8, keypoint_error_px=3.0,
                                   keypoint_rail_shift_px=9.0)


def test_without_stadium_keypoints_results_are_identical(tmp_path, monkeypatch, th, rules):
    _, source = _scene_setup(tmp_path, monkeypatch)
    plain = _judge(source, tmp_path, th, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG)
    off = _judge(source, tmp_path, th, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG,
                 stadium_tools=None, camera_policy=None)
    assert [_row(r) for r in plain] == [_row(r) for r in off]
    assert plain[0].geometry_evidence == off[0].geometry_evidence
    assert "camera" not in plain[0].motion_evidence


def test_keypoint_snapshot_is_the_third_choice(tmp_path, monkeypatch, th, rules):
    from dataclasses import replace as dc_replace

    c, source = _scene_setup(tmp_path, monkeypatch, select=False)
    kp = dc_replace(_snapshot(c, sid="feedfacefeedface"), kind="keypoints", source_frame=42,
                    agreeing_frames=2)
    calls = []

    def fake_snapshot(video, out_dir, tools, autocalib, vetoed, **k):
        calls.append(k)
        return kp, _Found(visibility={"over_left": 1.0, "over_right": 1.0, "xtreme": 0.5})

    monkeypatch.setattr(pipeline, "keypoint_snapshot", fake_snapshot)
    monkeypatch.setattr(pipeline.KeypointLocator, "for_snapshot",
                        classmethod(lambda cls, *a: None))
    chosen = []
    [r] = _judge(source, tmp_path, th, rules, stadium_tools=_tools(), on_snapshot=chosen.append)
    assert chosen == [kp] and r.calibration_id == kp.id
    assert calls == [{"hwaccel": th.decode_hwaccel, "allow_single": False,
                      "keypoint_errors": []}]
    assert "zone_out_of_frame@xtreme" in r.review_reasons and r.needs_review
    assert r.motion_evidence["stadium_keypoints"] == {
        "snapshot_id": kp.id, "method": "keypoints", "frame_index": 42, "agreeing_frames": 2,
        "coverage": 0.9, "precision": 0.8, "keypoint_error_px": 3.0,
        "keypoint_rail_shift_px": 9.0}
    # Neither route finds a stadium: recalibrate, saying the keypoints failed too.
    monkeypatch.setattr(pipeline, "keypoint_snapshot", lambda *a, **k: (None, None))
    [r] = _judge(source, tmp_path, th, rules, stadium_tools=_tools())
    assert r.review_reasons == ["recalibrate", "stadium_unlocated"]

    def failing(*a, keypoint_errors=None, **k):
        keypoint_errors.append("frame 0: RuntimeError: CUDA out of memory")
        return None, None

    monkeypatch.setattr(pipeline, "keypoint_snapshot", failing)
    [r] = _judge(source, tmp_path, th, rules, stadium_tools=_tools())
    assert r.review_reasons == ["recalibrate", "stadium_unlocated", "keypoint_inference_error"]


def _handheld(monkeypatch, *, locator=True):
    from bxjudge.camera_motion import CameraMotion

    monkeypatch.setattr(pipeline, "measure_video_motion",
                        lambda *a, **k: CameraMotion(60.0, 9))
    monkeypatch.setattr(pipeline.KeypointLocator, "for_snapshot", classmethod(
        lambda cls, tools, snap, c, ref: _FakeLocator(c) if locator else None))


def test_handheld_round_marks_touch_unknown_and_notes_camera(tmp_path, monkeypatch, th, rules):
    from bxjudge.camera_motion import load_camera_policy

    _, source = _scene_setup(tmp_path, monkeypatch)
    _handheld(monkeypatch)

    def no_motion(*a, **k):
        raise AssertionError("stadium displacement is camera motion in handheld mode")

    monkeypatch.setattr(pipeline, "iter_stadium_displacement", no_motion)
    [r] = _judge(source, tmp_path, th, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG,
                 stadium_tools=_tools(), camera_policy=_camera_policy())
    assert r.launch.touch == {"X": "unknown", "B": "unknown"}
    assert "camera_handheld" in r.review_reasons
    assert "stadium_motion_unknown@camera_handheld" in r.review_reasons and r.needs_review
    assert set(r.motion_evidence["displacements"].values()) <= {None}
    assert r.motion_evidence["camera"] == {"mode": "handheld", "source": "measured",
                                           "px_per_s": 60.0, "pairs": 9}
    data = json.loads((tmp_path / "out" / r.geometry_evidence["path"]).read_text())
    assert data["mode"] == "handheld" and data["camera"]["mode"] == "handheld"
    assert [x["frame"] for x in data["refinds"]] == [30, 60, 79]  # 30 fps; finish() at 79
    assert all(s["alignment"] is not None for s in data["samples"])


def test_fixed_camera_keeps_the_dynamic_tracker(tmp_path, monkeypatch, th, rules):
    from bxjudge.camera_motion import CameraMotion, load_camera_policy

    _, source = _scene_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "measure_video_motion", lambda *a, **k: CameraMotion(4.0, 9))
    made = []
    monkeypatch.setattr(pipeline.KeypointLocator, "for_snapshot", classmethod(
        lambda cls, tools, snap, c, ref: made.append(_FakeLocator(c)) or made[-1]))
    [r] = _judge(source, tmp_path, th, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG,
                 stadium_tools=_tools(), camera_policy=_camera_policy())
    assert "camera_handheld" not in r.review_reasons
    assert r.motion_evidence["camera"]["mode"] == "fixed"
    data = json.loads((tmp_path / "out" / r.geometry_evidence["path"]).read_text())
    assert "mode" not in data and data["camera"]["px_per_s"] == 4.0
    assert len(made) == 1  # handed to DynamicCalibration for relocation


def test_handheld_without_locatable_reference_is_recalibrate(tmp_path, monkeypatch, th, rules):
    from bxjudge.camera_motion import load_camera_policy

    _, source = _scene_setup(tmp_path, monkeypatch)
    _handheld(monkeypatch, locator=False)
    [r] = _judge(source, tmp_path, th, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG,
                 stadium_tools=_tools(), camera_policy=_camera_policy())
    assert r.review_reasons == ["recalibrate", "handheld_reference_unlocated"]
    assert r.geometry_evidence is None


def test_handheld_without_dynamic_calibration_is_recalibrate(tmp_path, monkeypatch, th, rules):
    from bxjudge.camera_motion import load_camera_policy

    _, source = _scene_setup(tmp_path, monkeypatch)
    _handheld(monkeypatch)
    [r] = _judge(source, tmp_path, th, rules, stadium_tools=_tools(),
                 camera_policy=_camera_policy())
    assert r.review_reasons == ["recalibrate", "handheld_requires_dynamic_calibration"]


def test_handheld_single_fit_start_is_confirmed_before_any_zone(tmp_path, monkeypatch, th,
                                                                 rules):
    from dataclasses import replace as dc_replace

    from bxjudge.camera_motion import load_camera_policy

    c, source = _scene_setup(tmp_path, monkeypatch, select=False)
    _handheld(monkeypatch)
    kp = dc_replace(_snapshot(c, sid="feedfacefeedface"), kind="keypoints", source_frame=12,
                    agreeing_frames=1)
    calls = []

    def fake_snapshot(*a, **k):
        calls.append(k)
        return kp, _Found(method="keypoints_single", frame_index=12, agreeing=1)

    monkeypatch.setattr(pipeline, "keypoint_snapshot", fake_snapshot)
    [r] = _judge(source, tmp_path, th, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG,
                 stadium_tools=_tools(), camera_policy=_camera_policy())
    assert calls[0]["allow_single"] is True
    data = json.loads((tmp_path / "out" / r.geometry_evidence["path"]).read_text())
    assert data["segments"][0]["kind"] == "unconfirmed_snapshot"
    samples = {s["frame"]: s for s in data["samples"]}
    assert all(samples[f]["alignment"] is None for f in range(30))
    assert samples[30]["status"] == "refound"
    assert "calibration_relocated" in r.review_reasons  # the round (35-74) uses segment 1


def test_stadium_keypoint_cli_flags_reach_the_batch(tmp_path, monkeypatch):
    from bxjudge.cli import main

    captured = {}
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **kw: captured.update(kw))
    assert main(["run", str(tmp_path), "--no-vlm"]) == 0
    assert captured["stadium_keypoints"] is False
    custom = tmp_path / "camera.yaml"
    assert main(["run", str(tmp_path), "--no-vlm", "--stadium-keypoints",
                 "--camera-mode", str(custom)]) == 0
    assert captured["stadium_keypoints"] is True and captured["camera_config"] == custom


def test_batch_loads_the_keypoint_tools_once_and_hands_them_to_every_video(
        tmp_path, monkeypatch, th, rules):
    from bxjudge.camera_motion import load_camera_policy
    from tests.test_observed_timing import Batch

    tools = object()
    loads = []
    monkeypatch.setattr(pipeline, "load_stadium_tools",
                        lambda device=None: loads.append(device) or tools)
    for enabled in (False, True):
        root = tmp_path / str(enabled)
        root.mkdir()
        batch = Batch(root, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
        fake_judge = pipeline.judge_video
        seen = []

        def judge(info, *a, fake_judge=fake_judge, seen=seen, **k):
            seen.append((k.get("stadium_tools"), k.get("camera_policy")))
            return fake_judge(info, *a, **k)

        monkeypatch.setattr(pipeline, "judge_video", judge)
        batch.run(**({"stadium_keypoints": True} if enabled else {}))
        if enabled:
            assert len(loads) == 1
            assert seen == [(tools, load_camera_policy())] * 2
        else:
            assert loads == [] and seen == [(None, None)] * 2


# ---- Codex review R1 (Task 6) ---------------------------------------------------------


def test_a_preselected_single_fit_snapshot_still_needs_confirmation(tmp_path, monkeypatch, th,
                                                                    rules):
    """The batch may hand over a stored keypoints snapshot (no FoundStadium): its stored
    origin decides the unconfirmed start, as for a fresh one (Codex M4)."""
    from dataclasses import replace as dc_replace

    from bxjudge.camera_motion import load_camera_policy

    c, source = _scene_setup(tmp_path, monkeypatch, select=False)
    _handheld(monkeypatch)
    kp = dc_replace(_snapshot(c, sid="feedfacefeedface", score=0.77), kind="keypoints",
                    source_frame=12, agreeing_frames=1)
    monkeypatch.setattr(pipeline, "keypoint_snapshot",
                        lambda *a, **k: pytest.fail("a preselected snapshot is not re-found"))
    [r] = _judge(source, tmp_path, th, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG,
                 stadium_tools=_tools(), camera_policy=_camera_policy(), snapshot=kp)
    data = json.loads((tmp_path / "out" / r.geometry_evidence["path"]).read_text())
    assert data["segments"][0]["kind"] == "unconfirmed_snapshot"
    assert all(s["alignment"] is None for s in data["samples"] if s["frame"] < 30)
    assert r.motion_evidence["stadium_keypoints"] == {
        "snapshot_id": kp.id, "method": "keypoints_single", "frame_index": 12,
        "agreeing_frames": 1, "coverage": 0.77, "precision": None, "keypoint_error_px": None,
        "keypoint_rail_shift_px": None}


def test_a_failed_camera_measurement_is_unmeasured_not_an_analysis_error(tmp_path, monkeypatch,
                                                                          th, rules):
    from bxjudge.camera_motion import load_camera_policy

    _, source = _scene_setup(tmp_path, monkeypatch)
    _handheld(monkeypatch)

    def broken(*a, **k):
        raise cv2.error("feature detection failed")

    monkeypatch.setattr(pipeline, "measure_video_motion", broken)
    [r] = _judge(source, tmp_path, th, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG,
                 stadium_tools=_tools(), camera_policy=_camera_policy())
    camera = r.motion_evidence["camera"]
    assert (camera["mode"], camera["source"], camera["px_per_s"]) == \
        ("handheld", "unmeasured", None)
    assert camera["error"].startswith("error:")


def _batch_judged(tmp_path, monkeypatch, th, rules):
    from tests.test_observed_timing import Batch

    batch = Batch(tmp_path, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
    monkeypatch.setattr(pipeline, "load_stadium_tools",
                        lambda device=None: __import__("types").SimpleNamespace(
                            cfg=None, finder=__import__("types").SimpleNamespace(
                                weights_sha256=["w1"], reference=None),
                            model=__import__("types").SimpleNamespace(source_sha256="m")))
    return batch


def test_turning_keypoint_stadiums_on_or_changing_them_rejudges_cached_videos(
        tmp_path, monkeypatch, th, rules):
    """The analysis signature covers the keypoint route even without dynamic calibration
    (Codex High), and its model, weights and camera policy (Codex M3)."""
    import yaml

    batch = _batch_judged(tmp_path, monkeypatch, th, rules)
    batch.run()
    assert len(batch.judged) == 2
    batch.run()
    assert batch.judged == []  # cached
    batch.run(stadium_keypoints=True)
    assert len(batch.judged) == 2
    batch.run(stadium_keypoints=True)
    assert batch.judged == []
    custom = tmp_path / "camera.yaml"
    doc = yaml.safe_load(Path("config/camera_mode.yaml").read_text())
    doc["detection"]["handheld_px_per_s"] = 25.0
    custom.write_text(yaml.safe_dump(doc))
    batch.run(stadium_keypoints=True, camera_config=custom)
    assert len(batch.judged) == 2
    from types import SimpleNamespace

    monkeypatch.setattr(pipeline, "load_stadium_tools", lambda device=None: SimpleNamespace(
        cfg=None, finder=SimpleNamespace(weights_sha256=["w2"], reference=None),
        model=SimpleNamespace(source_sha256="m")))
    batch.run(stadium_keypoints=True, camera_config=custom)
    assert len(batch.judged) == 2  # new keypoint weights


def test_batch_preselection_reuses_a_recorded_keypoint_snapshot(tmp_path, monkeypatch):
    """A cached video judged with a keypoints snapshot keeps it as the current choice (no
    endless rejudging); a vetoed recorded snapshot gets a fresh keypoint find (Codex M2)."""
    from bxjudge.calibration_store import save_snapshot

    img, c = stadium_scene(size=(640, 360))
    out = tmp_path / "out"
    kp = save_snapshot(out, "keypoints", c, img, score=0.9, homography=np.eye(3).tolist(),
                       source_frame=3, agreeing_frames=2, autocalib=shipped_autocalib())
    manual = save_snapshot(out, "manual", c, img, score=1.0, autocalib=shipped_autocalib())
    called = []
    monkeypatch.setattr(pipeline, "keypoint_snapshot",
                        lambda *a, **k: called.append(k) or (kp, None))
    info_ = info(tmp_path / "clip.mp4")
    pick = pipeline._keypoint_preselection
    th_ = __import__("bxjudge.config", fromlist=["x"]).load_thresholds()
    assert pick(out, kp.id, set(), info_, object(), shipped_autocalib(), th_, reuse=True) == kp
    assert called == []
    assert pick(out, manual.id, set(), info_, object(), shipped_autocalib(), th_, reuse=True) is None
    assert pick(out, None, set(), info_, object(), shipped_autocalib(), th_, reuse=True) is None
    assert pick(out, "0" * 16, set(), info_, object(), shipped_autocalib(), th_, reuse=True) is None
    assert called == []
    assert pick(out, manual.id, {manual.id}, info_, object(), shipped_autocalib(), th_, reuse=True) == kp
    assert called == [{"hwaccel": th_.decode_hwaccel}]


# ---- Codex re-review R2 (Task 6) ------------------------------------------------------


def test_a_recorded_keypoint_snapshot_is_reused_only_under_the_same_signature(tmp_path,
                                                                              monkeypatch):
    from bxjudge.calibration_store import save_snapshot
    from bxjudge.config import load_thresholds

    img, c = stadium_scene(size=(640, 360))
    out = tmp_path / "out"
    kp = save_snapshot(out, "keypoints", c, img, score=0.9, homography=np.eye(3).tolist(),
                       source_frame=3, agreeing_frames=2, autocalib=shipped_autocalib())
    monkeypatch.setattr(pipeline, "keypoint_snapshot",
                        lambda *a, **k: pytest.fail("no fresh find for a non-vetoed video"))
    args = (out, kp.id, set(), info(tmp_path / "clip.mp4"), object(), shipped_autocalib(),
            load_thresholds())
    assert pipeline._keypoint_preselection(*args, reuse=True) == kp
    # A changed keypoint route (model, weights, finder config): the stored snapshot is not
    # current any more; judge_video selects afresh.
    assert pipeline._keypoint_preselection(*args, reuse=False) is None


def test_a_vetoed_video_keeps_its_rows_when_the_alternative_gives_no_round(tmp_path,
                                                                           monkeypatch, th,
                                                                           rules):
    """T4-3: a veto alone never turns judged rows into a recalibrate placeholder, also when a
    keypoint alternative was found but judging with it still gives recalibrate (Codex High)."""
    from tests.test_observed_timing import Batch

    batch = Batch(tmp_path, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
    batch.calibration_id = "a" * 16
    batch.run()
    before = [r for r in json.loads((batch.out / "rounds.json").read_text())
              if r["video"] == "a.mp4"]
    assert before and before[0]["calibration_id"] == "a" * 16
    monkeypatch.setattr(pipeline, "vetoed_ids", lambda out_dir: {"a" * 16})
    monkeypatch.setattr(pipeline, "processed_calibration",
                        lambda db, p: "a" * 16 if p.name == "a.mp4" else None)
    batch.due = True  # an alternative exists, so the video is judged again

    def placeholder(info_, *a, **k):
        batch.judged.append(info_.path.name)
        r = pipeline._review(info_.path.name, info_.duration, "recalibrate")
        r.review_reasons.append("handheld_requires_dynamic_calibration")
        return [r]

    monkeypatch.setattr(pipeline, "judge_video", placeholder)
    batch.run(limit=1)
    # It was judged, so it uses the --limit quota (C-X9): b.mp4 waits for the next run.
    assert batch.judged == ["a.mp4"]
    assert [r for r in json.loads((batch.out / "rounds.json").read_text())
            if r["video"] == "a.mp4"] == before
    # R68: its latest analysis is the recalibrate it just gave (the review page warns).
    assert batch.timings()["a.mp4"]["analysis_status"] == "recalibrate"



# ---- final whole-branch review ----------------------------------------------------------


def test_unknown_geometry_reaches_the_launch_analysis(tmp_path, monkeypatch, th, rules):
    """Frames without geometry are handed to analyze_launch so a top not seen landing there
    is not a launch miss (final review High)."""
    from dataclasses import replace as dc_replace

    from bxjudge.camera_motion import load_camera_policy

    c, source = _scene_setup(tmp_path, monkeypatch, select=False)
    _handheld(monkeypatch)
    kp = dc_replace(_snapshot(c, sid="feedfacefeedface"), kind="keypoints", source_frame=12,
                    agreeing_frames=1)
    seen = {}
    original = pipeline.analyze_launch

    def analyze(*a, **k):
        seen.update(k)
        return original(*a, **k)

    monkeypatch.setattr(pipeline, "analyze_launch", analyze)
    [r] = _judge(source, tmp_path, th, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG,
                 stadium_tools=_tools(), camera_policy=_camera_policy(), snapshot=kp)
    assert set(range(30)) <= seen["unobserved_frames"]
    assert seen["round_start"] == 35  # the round's own start frame
    assert not set(range(30, 80)) & seen["unobserved_frames"]
    data = json.loads((tmp_path / "out" / r.geometry_evidence["path"]).read_text())
    assert data["fps"] == 30  # the review page can place unknown spans without motion evidence


def test_without_the_flag_launch_analysis_gets_no_unobserved_frames(tmp_path, monkeypatch, th,
                                                                    rules):
    _, source = _scene_setup(tmp_path, monkeypatch)
    seen = {}
    original = pipeline.analyze_launch
    monkeypatch.setattr(pipeline, "analyze_launch",
                        lambda *a, **k: seen.update(k) or original(*a, **k))
    [r] = _judge(source, tmp_path, th, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG)
    assert "unobserved_frames" not in seen
    data = json.loads((tmp_path / "out" / r.geometry_evidence["path"]).read_text())
    assert "fps" not in data


def test_a_handheld_video_without_any_stadium_says_so(tmp_path, monkeypatch, th, rules):
    from bxjudge.camera_motion import load_camera_policy

    _, source = _scene_setup(tmp_path, monkeypatch, select=False)
    _handheld(monkeypatch)
    monkeypatch.setattr(pipeline, "keypoint_snapshot", lambda *a, **k: (None, None))
    [r] = _judge(source, tmp_path, th, rules, dynamic_config=DEFAULT_DYNAMIC_CONFIG,
                 stadium_tools=_tools(), camera_policy=_camera_policy())
    assert r.review_reasons == ["recalibrate", "stadium_unlocated", "camera_handheld"]
    # The camera measurement that led there is kept with the row.
    assert r.motion_evidence == {"camera": {"mode": "handheld", "source": "measured",
                                            "px_per_s": 60.0, "pairs": 9}}
