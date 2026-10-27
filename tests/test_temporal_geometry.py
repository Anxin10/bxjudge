import numpy as np
import pytest

from bxjudge.temporal_geometry import TemporalRimTracker, load_temporal_policy
from bxjudge.trajectory_video import load_video_policy
from tests.helpers import stadium_scene


def tracker():
    ref, c = stadium_scene(size=(640, 360))
    return (
        TemporalRimTracker(
            c, ref, 30, load_video_policy()["stadium_follow"], load_temporal_policy()
        ),
        ref,
        c,
    )


def test_chained_observations_need_template_closure_before_becoming_valid():
    obj, ref, _c = tracker()
    obj.observe(0, ref, [])
    obj.anchor(0, np.eye(3), 0)
    moved, _ = stadium_scene(size=(640, 360), shift=(2, 0))
    proposal = obj.observe(1, moved, [])
    assert proposal is not None
    assert obj.confirm(1, proposal, 0) == [1]


def test_template_disagreement_invalidates_the_whole_pending_chain():
    obj, ref, _c = tracker()
    obj.observe(0, ref, [])
    obj.anchor(0, np.eye(3), 0)
    moved, _ = stadium_scene(size=(640, 360), shift=(2, 0))
    assert obj.observe(1, moved, []) is not None
    wrong = np.eye(3)
    wrong[0, 2] = 30
    assert obj.confirm(1, wrong, 0) == []


def test_masked_full_foreground_and_flat_frame_cannot_supply_temporal_geometry():
    obj, ref, _c = tracker()
    obj.observe(0, ref, [])
    obj.anchor(0, np.eye(3), 0)
    assert obj.observe(1, ref, [(0, 0, 640, 360)]) is None
    assert obj.observe(2, np.full_like(ref, 128), []) is None


def test_source_time_limit_and_nonconsecutive_frame_break_the_chain():
    obj, ref, _c = tracker()
    obj.observe(0, ref, [])
    obj.anchor(0, np.eye(3), 0)
    assert obj.observe(2, ref, []) is None


def test_missing_temporal_policy_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("reanchor_s: 1\n")
    import pytest

    with pytest.raises(ValueError):
        load_temporal_policy(p)


def test_closed_template_chain_fills_only_observed_supported_frames(monkeypatch):
    from bxjudge.dynamic_calibration import DynamicCalibration, load_dynamic_policy

    ref, c = stadium_scene(size=(640, 360))
    d = DynamicCalibration(
        c, ref, 30, load_video_policy()["stadium_follow"], load_dynamic_policy()
    )
    d.temporal.cfg = {**d.temporal.cfg, "reanchor_s": 1 / 30}
    real = d.anchor.align
    d.update(0, ref)
    calls = iter([None, None, True, True, True])
    monkeypatch.setattr(
        d.anchor, "align", lambda image: real(image) if next(calls) else None
    )
    d.update(1, ref)
    d.update(2, ref)
    assert d.samples[1].alignment is None
    d.update(3, ref)
    d.update(4, ref)
    d.update(5, ref)
    assert d.samples[1].status == "temporal_verified"
    assert d.samples[1].alignment is not None


def test_bridged_geometry_never_enters_side_specific_judging():
    from bxjudge.autocalib import Alignment
    from bxjudge.dynamic_calibration import GeometrySample, normalize_tracks
    from tests.helpers import make_track

    a = Alignment(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), 1.0, 0.0, 0.0, 1.0, 0.0
    )
    samples = {
        0: GeometrySample(0, a, 0, "tracking"),
        1: GeometrySample(1, a, 0, "bridged"),
        2: GeometrySample(2, a, 0, "tracking"),
    }
    [track] = normalize_tracks(
        [make_track(1, "bey", {0: (100, 100), 1: (100, 100), 2: (100, 100)})], samples
    )
    assert set(track.centers()) == {0, 2}


def test_coverage_excludes_estimates_from_verified_frame_count():
    from bxjudge.geometry_coverage import measure_window

    result = measure_window(
        [{"frame": 0, "alignment": {"homography": []}, "status": "bridged"}], 0, 0
    )
    assert result["valid_frames"] == 0


def test_template_region_is_not_assumed_to_be_the_physical_ellipse():
    ref, c = stadium_scene(size=(640, 360))
    c.lid = c.stadium
    obj = TemporalRimTracker(
        c, ref, 30, load_video_policy()["stadium_follow"], load_temporal_policy()
    )
    assert obj.reference_ellipse is not None


@pytest.mark.parametrize("recorded", [True, False])
def test_animation_never_labels_bridged_geometry_as_verified(monkeypatch, recorded):
    from dataclasses import asdict

    from bxjudge import trajectory_video as video
    from bxjudge.autocalib import Alignment
    from bxjudge.dynamic_calibration import DynamicCalibration, GeometrySample

    ref, c = stadium_scene(size=(640, 360))
    a = Alignment(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), 1.0, 0.0, 0.0, 1.0, 0.0
    )
    sample = GeometrySample(0, a, 0, "bridged")
    monkeypatch.setattr(DynamicCalibration, "update", lambda *args: sample)
    labels = []
    monkeypatch.setattr(
        video.plot, "label", lambda canvas, text, *args: labels.append(text)
    )
    row = {
        "t_launch": 0.0,
        "t_end": 0.0,
        "finish": None,
        "landing_frame": {"X": None, "B": None},
    }
    points = [video.Observation("X", 1, "bey", 0, 0.0, 200.0, 200.0)]
    record = (
        {"reference_size": [640, 360], "samples": {0: asdict(sample)}}
        if recorded
        else None
    )
    list(
        video.animation_frames(
            row,
            c,
            points,
            load_video_policy(),
            coordinate_size=(640, 360),
            reference=ref,
            background_frames=iter([ref]),
            recorded_geometry=record,
            source_fps=30,
        )
    )
    assert any("zones hidden" in text for text in labels)
    assert not any("geometry:" in text or "Follow preview:" in text for text in labels)


def test_directly_verified_frame_is_not_withheld_until_a_future_checkpoint():
    from bxjudge.dynamic_calibration import DynamicCalibration, load_dynamic_policy

    ref, c = stadium_scene(size=(640, 360))
    d = DynamicCalibration(
        c, ref, 30, load_video_policy()["stadium_follow"], load_dynamic_policy()
    )
    assert d.update(0, ref).alignment is not None
    # Reference evidence for the current frame is available now. Do not turn it
    # into an estimated gap just because the periodic flow anchor is not due.
    assert d.update(1, ref).alignment is not None


def test_bridge_never_covers_a_chain_template_contradiction(monkeypatch):
    from bxjudge.dynamic_calibration import DynamicCalibration, load_dynamic_policy

    ref, c = stadium_scene(size=(640, 360))
    d = DynamicCalibration(
        c, ref, 30, load_video_policy()["stadium_follow"], load_dynamic_policy()
    )
    d.update(0, ref)
    real = d.temporal.confirm

    def confirm(frame, matrix, segment):
        if frame == 1:
            d.temporal.reject("temporal_template_disagreement")
            return []
        return real(frame, matrix, segment)

    monkeypatch.setattr(d.temporal, "confirm", confirm)
    d.update(1, ref)
    assert d.update(2, ref).alignment is not None
    assert d.samples[1].alignment is None
    assert d.samples[1].status == "lost"
    assert "bridge_reason" not in d.diagnostics[1]


def test_recorded_calibration_ignores_bridged_samples():
    from bxjudge.dynamic_calibration import recorded_calibration

    _ref, c = stadium_scene(size=(640, 360))
    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    full = {"homography": eye, "score": 1.0, "shift_frac": 0.0,
            "rotation_deg": 0.0, "scale": 1.0, "perspective": 0.0}
    record = {
        "reference_size": [640, 360],
        "samples": {
            0: {"alignment": full, "status": "tracking"},
            1: {"alignment": full, "status": "bridged"},
        },
    }
    assert recorded_calibration(c, record, 0, (640, 360)) is not None
    assert recorded_calibration(c, record, 1, (640, 360)) is None


def test_motion_occlusion_still_exports_the_motion_mask():
    obj, ref, _c = tracker()
    obj.observe(0, ref, [])
    obj.anchor(0, np.eye(3), 0)
    bright = np.clip(ref.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    assert obj.observe(1, bright, []) is None
    assert obj.last_diagnostic["reason"] == "temporal_motion_occlusion"
    assert np.count_nonzero(obj.foreground_mask) > 0
