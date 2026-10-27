import json

import numpy as np

from bxjudge.dynamic_calibration import DynamicCalibration, load_dynamic_policy
from bxjudge.stadium_follow import StadiumFollower
from bxjudge.trajectory_video import load_video_policy
from tests.helpers import stadium_scene


def test_rejection_records_actual_gate_and_does_not_keep_previous_success():
    reference, calibration = stadium_scene(size=(640, 360))
    follower = StadiumFollower(
        calibration, reference, load_video_policy()["stadium_follow"]
    )
    assert follower.align(reference) is not None
    assert follower.last_diagnostic["reason"] is None
    assert follower.align(np.full_like(reference, 128)) is None
    assert follower.last_diagnostic["reason"] == "frame_features_missing"
    assert follower.align(reference) is not None
    assert follower.last_diagnostic["reason"] is None


def test_bounds_failure_records_measurements_and_exact_policy_limits():
    reference, calibration = stadium_scene(size=(640, 360))
    moved, _ = stadium_scene(size=(640, 360), shift=(90, 0))
    cfg = load_video_policy()["stadium_follow"]
    follower = StadiumFollower(calibration, reference, cfg)
    assert follower.align(moved) is None
    diagnostic = follower.last_diagnostic
    assert diagnostic["reason"] == "transform_out_of_bounds"
    assert diagnostic["shift_frac"] > cfg["max_shift_frac"]
    assert diagnostic["limits"]["max_shift_frac"] == cfg["max_shift_frac"]
    json.dumps(diagnostic, allow_nan=False)


def test_each_source_frame_keeps_gate_and_confirmation_separate():
    reference, calibration = stadium_scene(size=(640, 360))
    tracker = DynamicCalibration(
        calibration,
        reference,
        30,
        load_video_policy()["stadium_follow"],
        load_dynamic_policy(),
    )
    tracker.update(0, reference)
    tracker.update(1, np.full_like(reference, 128))
    tracker.update(2, reference)
    tracker.update(3, reference)
    tracker.update(4, reference)
    record = tracker.record()
    values = record["samples"]
    assert [s["frame"] for s in values] == list(range(5))
    assert values[0]["rejection_reason"] is None
    assert values[1]["rejection_reason"] == "frame_features_missing"
    assert values[1]["diagnostics"]["tracking"]["reason"] == "frame_features_missing"
    assert values[2]["rejection_reason"] == "recovery_confirmation_pending"
    assert values[2]["diagnostics"]["tracking"]["reason"] is None
    assert values[2]["status"] == "bridged"
    assert values[2]["diagnostics"]["bridge_reason"] == "bounded_bridge_not_judging_evidence"
    assert values[4]["rejection_reason"] is None
    assert values[4]["alignment"] is not None
    assert all(s["diagnostics"]["relocation"] is None for s in values)
    json.dumps(record, allow_nan=False)
