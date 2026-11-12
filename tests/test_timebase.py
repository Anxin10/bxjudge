from pathlib import Path

import numpy as np
import pytest

from bxjudge.timebase import Timeline, span_at_least
from bxjudge.video_anomalies import AnomalyPolicy, detect_anomalies


def test_physical_span_counts_intervals_not_samples():
    stamps = tuple(i / 120 for i in range(7))
    assert not span_at_least(stamps, 0, 5, 50)
    assert span_at_least(stamps, 0, 6, 50)


def test_vfr_uses_source_pts_and_nominal_slow_motion_ratio():
    clock = Timeline((2.0, 2.033, 2.067, 2.367), 240, 30)
    assert clock.physical_s(0) == 0
    assert clock.elapsed_ms(2, 3) == pytest.approx(37.5)
    assert clock.playback_s(3) == pytest.approx(0.367)


@pytest.mark.parametrize("tag", [None, "120", "nan"])
def test_reject_missing_or_conflicting_capture_metadata(tag):
    from bxjudge.timebase import source_rates

    metadata = {
        "streams": [{"r_frame_rate": "30/1"}],
        "format": {"tags": {} if tag is None else {"com.android.capture.fps": tag}},
    }
    with pytest.raises(ValueError, match="capture.fps"):
        source_rates(Path("/videos/20261001/a.mp4"), metadata, {"20261001": (240, 30)})


def test_adaptive_anomalies_keep_motion_and_timestamp_evidence_separate():
    policy = AnomalyPolicy(3, 10, 0.25, 4, 6, 160)
    pts = np.arange(23, dtype=float) / 120
    pts[14:] += 0.1
    mean = np.full(22, 2.0)
    global_change = np.full(22, 1.0)
    mean[4:8] = 0.0
    mean[18] = 30
    global_change[18] = 30
    mean[10] = 20  # local motion, insufficient global change
    events = detect_anomalies(pts, mean, global_change, policy)
    assert [(e.kind, e.first_frame, e.last_frame) for e in events] == [
        ("freeze", 4, 8),
        ("timestamp_gap", 13, 14),
        ("content_jump", 18, 19),
    ]


def test_static_noisy_recording_is_not_a_freeze():
    policy = AnomalyPolicy(3, 10, 0.25, 4, 6, 160)
    differences = np.linspace(0.6, 1.4, 39)
    assert detect_anomalies(np.arange(40) / 120, differences, differences, policy) == []


def test_video_anomaly_note_preserves_decision_and_records_stop_overlap():
    from bxjudge.video_anomalies import Anomaly, annotate_round
    from tests.helpers import rr

    row = rr("spin", "X")
    original = (row.winner, row.finish, row.points, row.needs_review)
    clock = Timeline(tuple(i / 120 for i in range(20)), 120, 120)
    annotate_round(row, clock, [Anomaly("freeze", 3, 10)], 0, 19, [("B", "still", 5, 11)])
    assert (row.winner, row.finish, row.points, row.needs_review) == original
    assert "video_anomaly:freeze@3-10" in row.review_reasons
    event, = row.motion_evidence["video_anomalies"]
    assert event["overlapping_stop_evidence"] == [
        {"side": "B", "kind": "still", "first_frame": 5, "last_frame": 11}
    ]


@pytest.mark.parametrize("avg_fps,capture,playback,held,settle,jitter", [
    (30, 240, 30, 6, 6, 2),
    (23.976, 240, 30, 5, 6, 2),
    (120, 120, 120, 3, 3, 1),
])
def test_video_policy_converts_sample_counts(avg_fps, capture, playback, held, settle, jitter):
    from bxjudge.launch_phase import launch_sample_count
    from bxjudge.timebase import JudgmentTime, load_physical_time

    clock = Timeline((0, 0.033, 0.5), capture, playback)
    timing = JudgmentTime(clock, **load_physical_time())
    assert launch_sample_count(timing.values["held_s"], avg_fps) == held
    assert timing.values["settle_frames"] == settle
    assert timing.values["pocket_corner_jitter_frames"] == jitter
    assert timing.values["fall_speed_px"] == 1440 / capture
