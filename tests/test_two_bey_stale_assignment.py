"""Position/appearance assignment after a long loss is not player evidence."""

from pathlib import Path

import pytest

from bxjudge.two_bey_tracking import TwoBeyTracker, load_policy

POLICY = load_policy(Path("config/two_bey_tracking.yaml"))


def detection(native, x, appearance=(1.0, 0.0)):
    return {"native_id": native, "box": [x, 10, x + 10, 20],
            "score": 0.9, "appearance": appearance}


@pytest.mark.parametrize("native", [1, 27])
@pytest.mark.parametrize("appearance", [(1.0, 0.0), None])
def test_ordinary_stale_assignment_keeps_space_but_loses_player(native, appearance):
    tracker = TwoBeyTracker((200, 100), 30, POLICY)
    tracker.update(0, [detection(1, 10), detection(2, 100, (0.0, 1.0))])
    for f in range(1, 538):
        tracker.update(f, [detection(2, 100, (0.0, 1.0))])
    result = tracker.update(538, [detection(native, 12, appearance),
                                  detection(2, 100, (0.0, 1.0))])
    [returned] = [d for d in result["observed"] if d["entity_id"] == 1]
    assert returned["native_id"] == native
    assert returned["identity_valid"] is False
    assert tracker.identity_unknown_since == {1: 538}
    assert tracker.reacquisitions == [
        {"frame": 538, "entity_id": 1, "native_id": native, "unobserved_frames": 538}
    ]
    assert tracker.native_anchor_owner == {1: 1, 2: 2}
    later = tracker.update(539, [detection(native, 13, appearance)])
    assert later["observed"][0]["identity_valid"] is False


@pytest.mark.parametrize("gap,valid", [(29, True), (30, False)])
def test_ordinary_assignment_uses_existing_source_frame_gap_boundary(gap, valid):
    tracker = TwoBeyTracker((200, 100), 30, POLICY)
    tracker.update(0, [detection(1, 10), detection(2, 100, (0.0, 1.0))])
    # An update jump must use source time, not the number of update calls.
    result = tracker.update(gap, [detection(1, 10), detection(2, 100, (0.0, 1.0))])
    assert all(d["identity_valid"] is valid for d in result["observed"])


def test_validated_appearance_reanchor_can_resolve_a_long_merge():
    tracker = TwoBeyTracker((300, 100), 30, POLICY)
    tracker.update(0, [detection(1, 60), detection(2, 240, (0.0, 1.0))])
    for f in range(1, 41):
        tracker.update(f, [{"native_id": 1, "box": [55, 5, 255, 25],
                            "score": 0.9, "appearance": (0.5, 0.5)}])
    split = tracker.update(41, [detection(2, 60), detection(1, 240, (0.0, 1.0))])
    assert tracker.merge_episodes[-1]["identity_recovered_by"] == ["appearance"]
    assert all(d["identity_valid"] for d in split["observed"])
    assert tracker.identity_unknown_since == {}
    assert tracker.reacquisitions == []


def test_stale_new_native_is_not_imported_into_round_player_anchors():
    from bxjudge.two_bey_integration import reconstruct_round_tracks
    from bxjudge.types import Detection, Track

    def track(native, frames, x):
        return Track(native, "bey", [Detection(f, "bey", (x, 10, x + 10, 20), 0.9)
                                      for f in frames])

    tracks = [track(1, range(2), 10), track(2, range(40), 100),
              track(27, range(35, 40), 12)]
    features = {(d.frame, d.box): (1.0, 0.0) if t.track_id != 2 else (0.0, 1.0)
                for t in tracks for d in t.dets}
    physical = reconstruct_round_tracks(tracks, 0, 39, 30, (200, 100), features, POLICY, 100)
    assert physical.members == {101: [1], 102: [2]}
    assert physical.record["identity_unknown_since"] == {"101": 35}
    assert all(not d["identity_valid"] for d in physical.record["observations"]
               if d["native_id"] == 27)
