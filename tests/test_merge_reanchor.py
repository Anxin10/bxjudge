import json
from pathlib import Path

import pytest

from bxjudge.two_bey_tracking import TwoBeyTracker, identity_unknown_at, load_policy

A, B, MIX = (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)


def box(tid, cx, appearance):
    return {
        "native_id": tid,
        "box": [cx - 10, 90, cx + 10, 110],
        "score": 0.9,
        "appearance": None if appearance is None else list(appearance),
    }


def merged(tid, lo, hi, appearance=MIX):
    return {
        "native_id": tid,
        "box": [lo - 10, 90, hi + 10, 110],
        "score": 0.9,
        "appearance": None if appearance is None else list(appearance),
    }


def run(merge_frames, split_appearance=(A, B), appearance=(A, B)):
    """Top A (native 1) moves right, B (native 2) left; they cross while merged.

    After the split ByteTrack swaps the native IDs, as on 210958 f2828."""
    t = TwoBeyTracker((300, 200), 30, load_policy(Path("config/two_bey_tracking.yaml")))
    out = {}
    first = 8
    split = first + merge_frames
    for f in range(split + 4):
        a, b = 60 + 10 * f, 240 - 10 * f
        if f < first:
            ds = [box(1, a, appearance[0]), box(2, b, appearance[1])]
        elif f < split:
            ds = [merged(1, min(a, b), max(a, b), None if appearance[0] is None else MIX)]
        else:
            # Physical A now carries native 2 and vice versa.
            ds = [box(2, a, split_appearance[0]), box(1, b, split_appearance[1])]
        out[f] = t.update(f, ds)["observed"]
    return t, out, split


def position(rows, entity):
    [r] = [r for r in rows if r["entity_id"] == entity]
    return (r["box"][0] + r["box"][2]) / 2


def test_short_merge_with_distinct_tops_recovers_the_physical_identities():
    t, out, split = run(2)
    [episode] = t.merge_episodes
    assert episode["decision"] == "recovered"
    assert episode["identity_recovered_by"] == ["appearance", "motion"]
    assert episode["native_ids_swapped"] is True
    for f in range(split, split + 4):
        # Entity 1 is top A (moving right), despite the native-ID swap.
        assert position(out[f], 1) == 60 + 10 * f
        assert all(r["identity_valid"] for r in out[f])
    assert t.identity_unknown_since == {}
    assert t.identity_unknown_intervals == {1: [[8, split - 1]], 2: [[8, split - 1]]}
    assert all(not r["identity_valid"] for f in range(8, split) for r in out[f])


def test_identical_tops_after_a_long_merge_stay_unknown():
    t, out, split = run(10, split_appearance=(A, A), appearance=(A, A))
    [episode] = t.merge_episodes
    assert episode["decision"] == "unknown"
    assert episode["votes"] == {"motion": None, "appearance": None}
    assert all(not r["identity_valid"] for r in out[split + 3])
    assert set(t.identity_unknown_since) == {1, 2}


def test_conflicting_motion_and_appearance_leave_identity_unknown():
    t, out, split = run(2, split_appearance=(B, A))
    [episode] = t.merge_episodes
    assert set(episode["votes"].values()) == {"identity", "exchange"}
    assert episode["decision"] == "unknown"
    assert all(not r["identity_valid"] for r in out[split + 3])


def test_motion_alone_may_recover_when_appearance_is_absent():
    t, out, split = run(2, split_appearance=(None, None), appearance=(None, None))
    [episode] = t.merge_episodes
    assert episode["identity_recovered_by"] == ["motion"]
    assert position(out[split + 1], 1) == 60 + 10 * (split + 1)


def test_player_anchor_members_do_not_follow_a_reanchored_native_id():
    t, _, _ = run(2)
    # ByteTrack's IDs moved to the other top; anchors keyed by them must not.
    assert t.native_owners == {1: 2, 2: 1}
    assert t.native_anchor_owner == {1: 1, 2: 2}


def test_real_210958_merge_is_never_resolved_as_not_swapped():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/210958-merge-2796-2840.json").read_bytes()
    )
    t = TwoBeyTracker(
        tuple(fixture["frame_size"]),
        fixture["fps"],
        load_policy(Path("config/two_bey_tracking.yaml")),
    )
    for frame in fixture["observations"]:
        t.update(frame["frame"], frame["detections"])
    assert t.merge_episodes or t.pending_merge
    assert all(
        e["decision"] == "unknown" or e["native_ids_swapped"] is True
        for e in t.merge_episodes
    )


def test_unknown_lookup_covers_closed_spans_and_open_tails():
    record = {
        "identity_unknown_since": {"7": 50},
        "identity_unknown_intervals": {"8": [[10, 12]]},
    }
    assert [identity_unknown_at(record, 8, f) for f in (9, 10, 12, 13)] == [
        False, True, True, False,
    ]
    assert identity_unknown_at(record, 7, 49) is False
    assert identity_unknown_at(record, 7, 50) is True
    # No compatibility default: an incomplete record is an error.
    with pytest.raises(KeyError):
        identity_unknown_at({"identity_unknown_since": {}}, 8, 11)


def test_trajectory_clears_the_side_only_inside_a_recovered_span(tmp_path):
    import csv

    import cv2

    from bxjudge.output import write_trajectory
    from bxjudge.types import RoundResult
    from tests.helpers import make_track, stadium_scene

    image, calibration = stadium_scene(size=(640, 360))
    path = tmp_path / "reference.png"
    assert cv2.imwrite(str(path), image)
    calibration.ref_frame = str(path)
    track = make_track(1, "bey", {f: (300 + f, 200) for f in range(6)})
    result = RoundResult("clip.mp4", 1, 0, 1, None, None, None, 0, 0.0, True, [])
    csv_path, _ = write_trajectory(
        tmp_path, result, {"X": track}, [], calibration, 30, {},
        unknown_side_intervals={1: [[2, 3]]},
    )
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["side"] for r in rows] == ["X", "X", "", "", "X", "X"]


def test_overlapped_pre_merge_crops_do_not_become_reanchor_evidence():
    # Tops touching before the merge mix colours (IoU 1/3 here). Only crops
    # that overlap nothing may describe a top for re-anchoring.
    t = TwoBeyTracker((300, 200), 30, load_policy(Path("config/two_bey_tracking.yaml")))
    t.update(0, [box(1, 60, A), box(2, 240, B)])
    t.update(1, [box(1, 62, A), box(2, 238, B)])
    for f in range(2, 62):
        t.update(f, [box(1, 140, (0.2, 0.8)), box(2, 150, (0.8, 0.2))])
    for f in range(62, 72):
        t.update(f, [merged(1, 140, 150)])
    rows = t.update(72, [box(2, 118, A), box(1, 172, B)])["observed"]
    [episode] = t.merge_episodes
    assert episode["decision"] == "recovered"
    assert episode["native_ids_swapped"] is True
    assert position(rows, 1) == 118


def test_split_boxes_unlike_both_tops_are_never_given_a_player():
    t = TwoBeyTracker((300, 200), 30, load_policy(Path("config/two_bey_tracking.yaml")))
    a, b = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    for f in range(8):
        t.update(f, [box(1, 60 + 10 * f, a), box(2, 240 - 10 * f, b)])
    for f in range(8, 20):
        t.update(f, [merged(1, 140, 160, (0.5, 0.5, 0.0))])
    rows = t.update(20, [box(1, 130, (0.04, 0, 0.96)), box(2, 170, (0, 0.04, 0.96))])
    [episode] = t.merge_episodes
    assert episode["decision"] == "unknown"
    assert episode["reason"] == "split_outside_gates"
    assert not any(r["identity_valid"] for r in rows["observed"])


def test_the_creation_crop_counts_as_clean_reanchor_evidence():
    # Only frame 0 is unoccluded; the tops then touch until the merge and come
    # out on swapped sides. Motion alone says "kept", appearance disagrees.
    t = TwoBeyTracker((300, 200), 30, load_policy(Path("config/two_bey_tracking.yaml")))
    t.update(0, [box(1, 60, A), box(2, 240, B)])
    for f in range(1, 61):
        t.update(f, [box(1, 140, (0.2, 0.8)), box(2, 150, (0.8, 0.2))])
    for f in (61, 62):
        t.update(f, [merged(1, 140, 150)])
    rows = t.update(63, [box(1, 120, B), box(2, 170, A)])["observed"]
    [episode] = t.merge_episodes
    assert episode["decision"] == "unknown"
    assert episode["reason"] == "conflicting_cues"
    assert not any(r["identity_valid"] for r in rows)
