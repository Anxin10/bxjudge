import copy
from pathlib import Path

import numpy as np
import pytest

from bxjudge.two_bey_tracking import TwoBeyTracker, continuous_points, load_policy


def det(tid, x, color=(1.0, 0.0)):
    return {
        "native_id": tid,
        "box": [x, 10, x + 10, 20],
        "score": 0.9,
        "appearance": list(color),
    }


def tracker():
    return TwoBeyTracker(
        (200, 100), 30, load_policy(Path("config/two_bey_tracking.yaml"))
    )


def test_reappearing_top_keeps_one_identity_despite_native_id_change():
    t = tracker()
    rows = []
    for frame, ds in [
        (0, [det(10, 10)]),
        (1, [det(10, 12)]),
        (2, []),
        (3, [det(99, 16)]),
    ]:
        rows.extend(t.update(frame, ds)["observed"])
    assert {r["entity_id"] for r in rows} == {1}
    assert [r["frame"] for r in rows] == [0, 1, 3]
    assert t.entity_count == 1


def test_duplicate_boxes_do_not_create_a_second_physical_top():
    t = tracker()
    result = t.update(0, [det(1, 10), det(2, 10.5)])
    assert len(result["observed"]) == 1
    assert len(result["unassigned"]) == 1
    assert result["unassigned"][0]["reason"] == "duplicate_box"


def test_two_targets_persist_through_crossing_and_third_detection_is_audited():
    t = tracker()
    t.update(0, [det(10, 10), det(20, 90, (0, 1))])
    t.update(1, [det(10, 30), det(20, 70, (0, 1))])
    t.update(2, [det(10, 45), det(20, 55, (0, 1))])
    result = t.update(3, [det(30, 65), det(40, 35, (0, 1)), det(50, 150, (0.5, 0.5))])
    positions = {r["entity_id"]: r["box"][0] for r in result["observed"]}
    assert positions == {1: 65, 2: 35}
    assert len(result["unassigned"]) == 1 and t.entity_count == 2


def test_no_observation_is_assigned_to_both_targets_and_source_not_modified():
    t = tracker()
    ds = [det(10, 10), det(20, 90, (0, 1))]
    original = copy.deepcopy(ds)
    result = t.update(0, ds)
    assert ds == original
    assert len({r["native_id"] for r in result["observed"]}) == 2
    assert tracker().entity_count == 0
    with pytest.raises(ValueError):
        t.update(0, ds)


def test_missing_frames_are_explicit_interpolation_never_observed():
    rows = [
        {"entity_id": 1, "frame": 1, "box": [0, 0, 10, 10], "native_id": 4},
        {"entity_id": 1, "frame": 4, "box": [6, 3, 16, 13], "native_id": 8},
    ]
    points = continuous_points(rows)
    assert [p["frame"] for p in points] == [1, 2, 3, 4]
    assert [p["kind"] for p in points] == [
        "observed",
        "interpolated",
        "interpolated",
        "observed",
    ]
    assert points[1]["box"] == [2, 1, 12, 11]
    assert points[1]["adjudication_evidence"] is False
    assert points[0]["adjudication_evidence"] is True


def test_invalid_nonfinite_boxes_and_duplicate_native_ids_are_rejected():
    t = tracker()
    with pytest.raises(ValueError):
        t.update(0, [dict(det(1, 0), box=[0, 0, np.nan, 1])])
    with pytest.raises(ValueError):
        t.update(0, [det(1, 0), det(1, 100)])


def test_a_duplicate_native_id_cannot_later_spawn_a_false_second_top():
    t = tracker()
    t.update(0, [det(1, 10)])
    t.update(1, [det(1, 12), det(9, 12.5)])
    result = t.update(2, [det(9, 30)])
    assert t.entity_count == 1
    assert result["observed"][0]["entity_id"] == 1
    result = t.update(3, [det(9, 32), det(20, 100, (0, 1))])
    assert [(d["native_id"], d["entity_id"]) for d in result["observed"]] == [
        (9, 1),
        (20, 2),
    ]


def test_known_distinct_objects_are_not_collapsed_when_their_boxes_overlap():
    t = tracker()
    t.update(0, [det(1, 10), det(2, 100, (0, 1))])
    result = t.update(1, [det(1, 50), det(2, 51, (0, 1))])
    assert len(result["observed"]) == 2
    assert {d["entity_id"] for d in result["observed"]} == {1, 2}


def test_a_native_id_is_not_silently_reassigned_to_the_other_physical_top():
    t = tracker()
    t.update(0, [det(1, 10), det(2, 100, (0, 1))])
    result = t.update(1, [det(1, 100, (0, 1)), det(2, 10)])
    assert result["observed"] == []
    assert len(result["unassigned"]) == 2


def test_bad_yaml_policy_is_a_named_user_error(tmp_path):
    p = tmp_path / "bad-two-bey.yaml"
    p.write_text("max_targets: [2\n")
    with pytest.raises(ValueError, match="bad-two-bey.yaml"):
        load_policy(p)


def test_invalid_descriptor_update_leaves_time_motion_and_owners_unchanged():
    t = tracker()
    t.update(0, [det(1, 10)])
    previous = t.previous
    owners = dict(t.native_owners)
    state = t.slots[0]["kalman"].statePost.copy()
    with pytest.raises(ValueError, match="dimensions"):
        t.update(1, [det(1, 11, (1, 0, 0))])
    assert t.previous == previous
    assert t.native_owners == owners
    np.testing.assert_array_equal(t.slots[0]["kalman"].statePost, state)
    assert t.update(1, [det(1, 11)])["observed"][0]["entity_id"] == 1


def test_rejected_native_owner_does_not_contaminate_a_correct_new_alias():
    t = tracker()
    t.update(0, [det(1, 10), det(2, 100, (0, 1))])
    bad = dict(det(1, 100, (0, 1)), score=0.8)
    t.update(1, [bad, det(99, 100, (0, 1))])
    result = t.update(2, [det(1, 10), det(99, 102, (0, 1))])
    assert {(r["native_id"], r["entity_id"]) for r in result["observed"]} == {
        (1, 1),
        (99, 2),
    }
    assert t.native_owners[1] == 1 and t.native_owners[99] == 2


def test_replay_uses_explicit_coordinate_manifest_and_preserves_input_hashes(tmp_path):
    import json

    from scripts.reconstruct_two_beys import build
    from tests.test_reid_appearance import fixture, sha

    m, labels, annotations, _ = fixture(tmp_path)
    rounds = tmp_path / "rounds.json"
    rounds.write_text(
        json.dumps(
            [{"video": "synthetic.mp4", "round": 1, "t_launch": 0, "t_end": 0.8}]
        )
    )
    ref = tmp_path / "ref.json"
    ref.write_text("[]")
    pointer = tmp_path / "pointer.json"
    pointer.write_text(
        json.dumps({"reference": str(ref), "sha256": sha(ref), "revision": "a" * 64})
    )
    args = (
        tmp_path,
        rounds,
        pointer,
        Path("config/two_bey_tracking.yaml"),
        m,
        labels,
        annotations,
    )
    before = {str(p): sha(p) for p in (rounds, pointer, m, labels, annotations)}
    report = build(*args, tmp_path / "result")
    r = report["rounds"][0]
    assert r["configured_entity_slots"] == 2 and r["observed_entity_count"] == 1
    assert r["coordinate_size"] == [64, 48]
    assert sum(p["kind"] == "interpolated" for p in r["points"]) == 1
    assert r["manual_exclusions"] == []
    assert report["applied_to_referee"] is False
    assert all(sha(p) == digest for p, digest in before.items())
    from scripts.render_two_beys import build as render

    preview = render(
        tmp_path / "result/report.json",
        Path("config/trajectory_video.yaml"),
        tmp_path / "preview",
    )
    assert preview["previews"][0]["frame_count"] == 9
    assert preview["previews"][0]["entity_count"] == 1
    assert (tmp_path / "preview/index.html").exists()
    assert sha(tmp_path / "preview/reconstruction.json") == sha(
        tmp_path / "result/report.json"
    )
    from scripts.build_workbench import validate_two_bey_set

    validate_two_bey_set(tmp_path / "preview")
    (tmp_path / "preview" / preview["previews"][0]["path"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA"):
        validate_two_bey_set(tmp_path / "preview")
    with pytest.raises(ValueError, match="new directory"):
        build(*args, tmp_path / "result")


def _det_at(tid, cx, cy, size=12, color=(1.0, 0.0)):
    half = size / 2
    return {
        "native_id": tid,
        "box": [float(cx - half), float(cy - half), float(cx + half), float(cy + half)],
        "score": 0.9,
        "appearance": list(color),
    }


def test_circular_arc_prediction_reacquires_curved_orbit_after_occlusion():
    policy = load_policy(Path("config/two_bey_tracking.yaml"))
    policy["position_gate_ratio"] = 0.12
    policy["circular_prediction_enabled"] = True
    t = TwoBeyTracker((300, 300), 30, policy)
    cx, cy, r, omega = 150.0, 150.0, 80.0, 0.45
    for f in range(6):
        theta = f * omega
        res = t.update(f, [_det_at(10, cx + r * np.cos(theta), cy + r * np.sin(theta))])
        assert len(res["observed"]) == 1
    for f in range(6, 9):
        res_gap = t.update(f, [])
        assert res_gap["observed"] == []
    theta_9 = 9 * omega
    res_9 = t.update(9, [_det_at(99, cx + r * np.cos(theta_9), cy + r * np.sin(theta_9))])
    assert len(res_9["observed"]) == 1
    assert res_9["observed"][0]["entity_id"] == 1
    assert t.entity_count == 1


def test_circular_prediction_falls_back_cleanly_on_collinear_motion():
    policy = load_policy(Path("config/two_bey_tracking.yaml"))
    t = TwoBeyTracker((300, 200), 30, policy)
    for f in range(6):
        t.update(f, [_det_at(10, 40.0 + f * 12.0, 100.0)])
    t.update(6, [])
    res = t.update(7, [_det_at(20, 40.0 + 7 * 12.0, 100.0)])
    assert len(res["observed"]) == 1
    assert res["observed"][0]["entity_id"] == 1
    assert t.entity_count == 1


def test_rim_velocity_clamp_constrains_outward_prediction_to_arena_circle():
    policy = load_policy(Path("config/two_bey_tracking.yaml"))
    t = TwoBeyTracker((300, 300), 30, policy, arena_circle=(150.0, 150.0, 80.0))
    # Moving radially outward toward the right rim (150 + 80 = 230)
    for f, x in enumerate([190.0, 205.0, 220.0, 228.0]):
        t.update(f, [_det_at(10, x, 150.0)])
    # Predict 2 frames ahead while missing; clamped to rim, so a reappearing
    # detection just inside the rim at x=228 matches without drifting to x=255+
    t.update(4, [])
    res = t.update(5, [_det_at(10, 228.0, 150.0)])
    assert len(res["observed"]) == 1
    assert res["observed"][0]["entity_id"] == 1

