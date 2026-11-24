import pytest

from bxjudge.identity import anchor_sides
from tests.helpers import make_calibration, make_track


def test_short_missing_left_launcher_prefix_is_unknown_not_observed_free_motion():
    tracks = [
        make_track(1, "bey", {f: (200, 80) for f in range(8)}),
        make_track(2, "launcher", {f: (440, 40) for f in range(8)}, size=60),
        make_track(3, "launcher", {f: (200, 40) for f in range(2, 8)}, size=60),
    ]
    result = anchor_sides(
        tracks, [], make_calibration(), required_frames=4, fps=30, body_margin_px=12
    )
    assert result.sides == {1: "X"}
    assert result.evidence[1][0]["frame"] == 2


def test_duplicate_right_launcher_components_do_not_hide_separated_left_and_right_bodies():
    tracks = [
        make_track(1, "bey", {f: (440, 80) for f in range(6)}),
        make_track(2, "launcher", {f: (200, 40) for f in range(6)}, size=60),
        make_track(3, "launcher", {f: (440, 40) for f in range(6)}, size=60),
        make_track(4, "launcher", {f: (440, 50) for f in range(6)}, size=30),
    ]
    result = anchor_sides(
        tracks, [], make_calibration(), required_frames=4, fps=30, body_margin_px=12
    )
    assert result.sides == {1: "B"}


def test_long_unknown_prefix_cannot_turn_later_pickup_into_launch_origin():
    tracks = [
        make_track(
            1,
            "bey",
            {
                **{f: (500, 220) for f in range(40)},
                **{f: (200, 80) for f in range(40, 48)},
            },
        ),
        make_track(2, "launcher", {f: (440, 40) for f in range(48)}, size=60),
        make_track(3, "launcher", {f: (200, 40) for f in range(40, 48)}, size=60),
    ]
    assert (
        anchor_sides(
            tracks, [], make_calibration(), required_frames=4, fps=30, body_margin_px=12
        ).sides
        == {}
    )


def test_real_210958_launch_bodies_anchor_x_and_b_before_the_confirmed_swap():
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from bxjudge.dynamic_calibration import normalize_tracks
    from bxjudge.types import Calibration, Detection, Track

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/210958-launch-context.json").read_bytes()
    )
    calibration = Calibration(**fixture["calibration"])
    tracks = [
        Track(
            t["track_id"],
            t["cls"],
            [
                Detection(d["frame"], d["cls"], tuple(d["box"]), d["score"])
                for d in t["dets"]
            ],
        )
        for t in fixture["raw_source_tracks"]
    ]
    samples = {
        s["frame"]: SimpleNamespace(
            status=s["status"],
            alignment=SimpleNamespace(**s["alignment"]) if s["alignment"] else None,
        )
        for s in fixture["samples"]
    }
    result = anchor_sides(
        tracks,
        normalize_tracks(tracks, samples),
        calibration,
        required_frames=7,
        body_margin_px=12,
        fps=30,
    )
    assert result.sides == {5: "B", 6: "X"}
    assert {d["frame"] for d in result.evidence[6]} == set(range(2228, 2235))


@pytest.mark.parametrize("fps,expected", [(30, {}), (120, {1: "X"})])
def test_missing_launcher_wait_is_source_time_bounded(fps, expected):
    tracks = [
        make_track(1, "bey", {f: (200, 80) for f in range(48)}),
        make_track(2, "launcher", {f: (440, 40) for f in range(48)}, size=60),
        make_track(3, "launcher", {f: (200, 40) for f in range(40, 48)}, size=60),
    ]
    assert (
        anchor_sides(
            tracks,
            [],
            make_calibration(),
            required_frames=4,
            fps=fps,
            body_margin_px=12,
        ).sides
        == expected
    )


def test_identity_policy_missing_fields_are_not_filled(tmp_path):
    from bxjudge.identity import load_identity_policy

    path = tmp_path / "policy.yaml"
    path.write_text("context_wait_s: .75\n")
    with pytest.raises(ValueError):
        load_identity_policy(path)


def test_sparse_unknown_prefix_cannot_skip_the_source_time_deadline():
    tracks = [
        make_track(1, "bey", {0: (200, 80), **{f: (200, 80) for f in range(40, 48)}}),
        make_track(2, "launcher", {f: (440, 40) for f in range(48)}, size=60),
        make_track(3, "launcher", {f: (200, 40) for f in range(40, 48)}, size=60),
    ]
    assert (
        anchor_sides(
            tracks, [], make_calibration(), required_frames=4, fps=30, body_margin_px=12
        ).sides
        == {}
    )


def test_wide_body_spanning_two_players_cannot_merge_them_into_one_side():
    from bxjudge.identity import _body_pair

    boxes = [
        (1, (0, 0, 100, 100)),
        (2, (150, 0, 250, 100)),
        (3, (0, 0, 250, 100)),
        (4, (300, 0, 400, 100)),
    ]
    assert _body_pair(boxes, 0.8) == {}


def _raised_launch_cal():
    c = make_calibration()
    c.launch_left = [(40, 40), (320, 40), (320, 120), (40, 120)]
    c.launch_right = [(320, 40), (600, 40), (600, 120), (320, 120)]
    return c


def test_e1b_disabled_by_default_preserves_legacy_top_clipped_abort():
    from bxjudge.identity import load_identity_policy

    p = load_identity_policy()
    assert p["prelaunch_setup_anchor_enabled"] is False
    c = _raised_launch_cal()
    # Frame 0..3 top-clipped at y=10 (box y0=0..20, above launch_left y>=40), no launcher until f=50
    bey_pos = {**{f: (200, 10) for f in range(4)}, **{f: (200, 80) for f in range(4, 56)}}
    tracks = [
        make_track(1, "bey", bey_pos, size=20),
        make_track(2, "launcher", {f: (200, 60) for f in range(50, 56)}, size=50),
    ]
    p_short = {**p, "context_wait_s": 0.09375}
    res = anchor_sides(tracks, tracks, c, required_frames=4, body_margin_px=12, fps=120, policy=p_short)
    assert res.sides == {}


def test_e1b_top_border_prelaunch_entry_and_setup_wait_anchors_cleanly():
    from bxjudge.identity import load_identity_policy

    c = _raised_launch_cal()
    bey_pos = {**{f: (200, 10) for f in range(4)}, **{f: (200, 80) for f in range(4, 56)}}
    tracks = [
        make_track(1, "bey", bey_pos, size=20),
        make_track(2, "launcher", {f: (200, 60) for f in range(50, 56)}, size=50),
    ]
    p_on = {**load_identity_policy(), "context_wait_s": 0.09375, "prelaunch_setup_anchor_enabled": True, "prelaunch_setup_wait_s": 1.5}
    res = anchor_sides(tracks, tracks, c, required_frames=4, body_margin_px=12, fps=120, policy=p_on)
    assert res.sides == {1: "X"}


def test_e1b_never_anchors_once_bey_entered_battle_area_even_within_setup_wait():
    from bxjudge.identity import load_identity_policy

    c = _raised_launch_cal()
    # Frame 0 starts in battle_area (200, 220), then jumps to launcher at frame 1..8
    bey_pos = {0: (200, 220), **{f: (200, 80) for f in range(1, 9)}}
    tracks = [
        make_track(1, "bey", bey_pos, size=20),
        make_track(2, "launcher", {f: (200, 60) for f in range(1, 9)}, size=50),
    ]
    p_on = {**load_identity_policy(), "prelaunch_setup_anchor_enabled": True, "prelaunch_setup_wait_s": 1.5}
    res = anchor_sides(tracks, tracks, c, required_frames=4, body_margin_px=12, fps=120, policy=p_on)
    assert res.sides == {}


def test_e1b_prebattle_unanchored_setup_gap_allows_opponent_inference_but_anchored_gap_blocks():
    from pathlib import Path
    from bxjudge.identity import IdentityAnchors
    from bxjudge.two_bey_integration import infer_opponent_side, reconstruct_round_tracks
    from bxjudge.two_bey_tracking import identity_unknown_at, load_policy
    from bxjudge.types import Track

    c = make_calibration()
    pol = {**load_policy(Path("config/two_bey_tracking.yaml")), "reacquire_min_gap_s": 0.125}
    # Slot 1 (native 1) continuous from 0..65, anchored X at 0..3; enters battle_area at 60..65.
    # Slot 2 (native 2 at 0..2, gap 22 frames, native 3 at 25..65) unanchored before battle_start_frame=60.
    t1 = make_track(1, "bey", {**{f: (200, 80) for f in range(60)}, **{f: (200, 220) for f in range(60, 66)}})
    t2 = make_track(2, "bey", {f: (440, 80) for f in range(3)})
    t3 = make_track(3, "bey", {**{f: (440, 80) for f in range(25, 60)}, **{f: (440, 220) for f in range(60, 66)}})
    phys = reconstruct_round_tracks(
        [t1, t2, t3], 0, 65, 120.0, c.frame_size, {}, pol, 100,
        battle_start_frame=60, anchored_native_ids={1},
    )
    reg = [Track(t.track_id, t.cls, [d for d in t.dets if not identity_unknown_at(phys.record, t.track_id, d.frame)]) for t in phys.tracks]
    anchors = IdentityAnchors({1: "X"}, {1: [{"frame": f} for f in range(4)]})
    proof = infer_opponent_side(anchors, phys, reg, c, required_frames=3)
    assert proof is not None and proof["side"] == "B"

    # Anti-swap safety: if the ANCHORED slot itself had a gap before battle_start_frame=60, it MUST become unknown!
    t1_a = make_track(1, "bey", {f: (200, 80) for f in range(4)})
    t1_b = make_track(4, "bey", {**{f: (200, 80) for f in range(30, 60)}, **{f: (200, 220) for f in range(60, 66)}})
    t2_cont = make_track(2, "bey", {**{f: (440, 80) for f in range(60)}, **{f: (440, 220) for f in range(60, 66)}})
    phys_bad = reconstruct_round_tracks(
        [t1_a, t1_b, t2_cont], 0, 65, 120.0, c.frame_size, {}, pol, 100,
        battle_start_frame=60, anchored_native_ids={1},
    )
    reg_bad = [Track(t.track_id, t.cls, [d for d in t.dets if not identity_unknown_at(phys_bad.record, t.track_id, d.frame)]) for t in phys_bad.tracks]
    assert infer_opponent_side(anchors, phys_bad, reg_bad, c, required_frames=3) is None

