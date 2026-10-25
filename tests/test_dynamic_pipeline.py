import json

import cv2
import numpy as np

from bxjudge import pipeline
from bxjudge.dynamic_calibration import DEFAULT_DYNAMIC_CONFIG
from tests.helpers import cal_with, make_track, shipped_autocalib, stadium_scene
from tests.test_pipeline import _snapshot, info


def test_dynamic_pipeline_uses_reference_coordinates_and_keeps_raw_evidence(
    tmp_path, monkeypatch, th, rules
):
    ref, c = stadium_scene(size=(640, 360))
    cal_with(ref, tmp_path)
    c.ref_frame = str(tmp_path / "ref.png")
    cv2.imwrite(c.ref_frame, ref)
    frame, _ = stadium_scene(size=(640, 360), shift=(12, -8))
    monkeypatch.setattr(
        pipeline, "iter_frames", lambda *a, **k: iter((f, frame) for f in range(80))
    )
    monkeypatch.setattr(pipeline, "select_snapshot", lambda *a: _snapshot(c))
    monkeypatch.setattr(pipeline, "write_preview", lambda *a: None)
    monkeypatch.setattr(
        pipeline,
        "iter_stadium_displacement",
        lambda frames, *a: ((f, img, 5.0) for f, img in frames),
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
    )
    assert np.linalg.norm(np.array(observed[0].center_at(35)) - [250, 220]) < 1
    assert r.geometry_evidence["coordinates"] == "reference_pixels"
    record = tmp_path / "out" / r.geometry_evidence["path"]
    assert record.is_file()
    data = json.loads(record.read_text())
    assert data["calibration_id"] == r.calibration_id
    assert set(r.motion_evidence["displacements"].values()) == {5.0}
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
            0: {"frame": 0, "alignment": a},
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


def test_missing_geometry_after_pocket_onset_cannot_create_terminal_entry(
    tmp_path, monkeypatch, th, rules
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
            eye if frame <= 20 else None,
            0,
            "tracking" if frame <= 20 else "lost",
        )
        self.samples[frame] = sample
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
