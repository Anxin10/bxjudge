import json

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
    # Missing identity extends the launch window to its configured cap. Frames
    # beyond the decoded clip remain unknown, never manufactured as no motion.
    values = r.motion_evidence["displacements"]
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
