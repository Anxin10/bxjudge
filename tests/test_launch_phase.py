from bxjudge.launch_phase import load_launch_policy, prepare_launch_tracks
from tests.helpers import make_track


def test_held_wait_is_trimmed_only_after_observed_separation():
    bey = make_track(
        1,
        "bey",
        {
            **{f: (100, 90) for f in range(100)},
            **{f: (100, 160 + f - 100) for f in range(100, 110)},
        },
    )
    launcher = make_track(2, "launcher", {f: (100, 70) for f in range(110)}, size=40)
    result = prepare_launch_tracks([bey, launcher], 30, load_launch_policy())
    track = next(t for t in result.tracks if t.cls == "bey")
    assert track.first_frame() == 100
    assert result.releases[1] == 100
    assert len(track.dets) == 10
    assert bey.first_frame() == 0


def test_launcher_disappearance_is_not_observed_release():
    bey = make_track(1, "bey", {f: (100, 90 if f < 20 else 180) for f in range(30)})
    launcher = make_track(2, "launcher", {f: (100, 70) for f in range(20)}, size=40)
    result = prepare_launch_tracks([bey, launcher], 30, load_launch_policy())
    assert 1 not in result.releases
    assert "launch_release_unobserved@1" in result.notes


def test_unique_short_fragment_gap_links_actual_observations_only():
    a = make_track(1, "bey", {0: (100, 180), 1: (105, 180), 2: (110, 180)})
    b = make_track(2, "bey", {4: (120, 180), 5: (125, 180), 6: (130, 180)})
    result = prepare_launch_tracks([a, b], 30, load_launch_policy())
    assert len(result.tracks) == 1
    assert list(result.tracks[0].centers()) == [0, 1, 2, 4, 5, 6]
    assert result.members[1] == [1, 2]


def test_crossing_candidates_or_overlapping_tracks_are_not_joined():
    a = make_track(1, "bey", {0: (100, 180), 1: (105, 180), 2: (110, 180)})
    b = make_track(2, "bey", {4: (120, 180), 5: (125, 180), 6: (130, 180)})
    c = make_track(3, "bey", {4: (121, 180), 5: (126, 180), 6: (131, 180)})
    result = prepare_launch_tracks([a, b, c], 30, load_launch_policy())
    assert len(result.tracks) == 3
    assert not any(len(m) > 1 for m in result.members.values())


def test_held_track_cannot_extend_round_or_link_to_unobserved_release():
    held = make_track(1, "bey", {f: (100, 90) for f in range(30)})
    launcher = make_track(9, "launcher", {f: (100, 70) for f in range(30)}, size=40)
    free = make_track(2, "bey", {f: (200 + f, 220) for f in range(35, 45)})
    result = prepare_launch_tracks([held, launcher, free], 30, load_launch_policy())
    assert {t.track_id for t in result.tracks if t.cls == "bey"} == {2}
    assert result.excluded_held == [1]


def test_landing_deadline_begins_after_supported_release_not_long_mount(th):
    from bxjudge.launch_check import analyze_launch
    from tests.helpers import make_calibration

    bey = make_track(
        1,
        "bey",
        {
            **{f: (200, 40) for f in range(300)},
            **{f: (200, 180) for f in range(300, 330)},
        },
    )
    other = make_track(
        2,
        "bey",
        {
            **{f: (440, 40) for f in range(300)},
            **{f: (440, 180) for f in range(300, 330)},
        },
    )
    launchers = [
        make_track(3, "launcher", {f: (200, 20) for f in range(330)}, size=40),
        make_track(4, "launcher", {f: (440, 20) for f in range(330)}, size=40),
    ]
    result = prepare_launch_tracks([bey, other, *launchers], 30, load_launch_policy())
    by = [t for t in result.tracks if t.cls == "bey"]
    analysis = analyze_launch(
        by,
        launchers,
        {f: 0 for f in range(330)},
        make_calibration(),
        th,
        30,
        sides_override={1: "X", 2: "B"},
    )
    assert analysis.check.landed == {"X": True, "B": True}
    assert analysis.check.landing_frame == {"X": 300, "B": 300}
    assert analysis.window[0] == 285


def test_launcher_reappearing_after_support_gap_cannot_reset_release_deadline():
    bey = make_track(1, "bey", {f: (100, 90 if f < 20 else 180) for f in range(250)})
    body = make_track(
        9, "launcher", {f: (100, 70) for f in [*range(20), *range(200, 250)]}, size=40
    )
    result = prepare_launch_tracks([bey, body], 30, load_launch_policy())
    assert 1 not in result.releases
    assert next(t for t in result.tracks if t.cls == "bey").first_frame() == 0
    assert "launch_release_unobserved@1" in result.notes


def test_opposite_observed_velocity_refuses_fragment_join():
    a = make_track(10, "bey", {0: (100, 180), 1: (110, 180), 2: (120, 180)})
    b = make_track(11, "bey", {4: (140, 180), 5: (130, 180), 6: (120, 180)})
    result = prepare_launch_tracks([a, b], 30, load_launch_policy())
    assert result.members == {10: [10], 11: [11]}


def test_bey_gap_records_later_box_clearance_without_resetting_release():
    frames = [f for f in range(40) if f != 20]
    bey = make_track(1, "bey", {f: (100, 90 if f < 25 else 180) for f in frames})
    body = make_track(9, "launcher", {f: (100, 70) for f in range(40)}, size=40)
    result = prepare_launch_tracks([bey, body], 30, load_launch_policy())
    assert result.releases == {}
    assert next(t for t in result.tracks if t.cls == "bey").first_frame() == 0
    assert 20 not in next(t for t in result.tracks if t.cls == "bey").centers()
    evidence = result.record()["diagnostics"]["1"]
    assert evidence["interruption"] == {
        "reason": "bey_observation_gap",
        "frame": 20,
        "next_bey_frame": 21,
    }
    assert evidence["box_clear_runs"] == [
        {
            "body_track_id": 9,
            "first_frame": 25,
            "confirmed_at_frame": 27,
            "last_frame": 39,
        }
    ]


def test_missing_body_records_geometric_evidence_without_delayed_release():
    bey = make_track(1, "bey", {f: (100, 90 if f < 20 else 180) for f in range(250)})
    body = make_track(
        9, "launcher", {f: (100, 70) for f in [*range(20), *range(200, 250)]}, size=40
    )
    result = prepare_launch_tracks([bey, body], 30, load_launch_policy())
    assert result.releases == {}
    assert next(t for t in result.tracks if t.cls == "bey").first_frame() == 0
    assert result.diagnostics[1]["interruption"] == {
        "reason": "launcher_observation_missing",
        "frame": 20,
    }
    assert result.diagnostics[1]["box_clear_runs"][0]["first_frame"] == 200
    assert "launch_release_unobserved@1" in result.notes


def test_ambiguous_mounting_does_not_choose_a_body_for_clearance():
    bey = make_track(1, "bey", {f: (100, 90 if f < 20 else 180) for f in range(40)})
    bodies = [
        make_track(tid, "launcher", {f: (100, 70) for f in range(40)}, size=40)
        for tid in (8, 9)
    ]
    result = prepare_launch_tracks([bey, *bodies], 30, load_launch_policy())
    evidence = result.diagnostics[1]
    assert evidence["body_track_id"] is None
    assert evidence["interruption"] == {
        "reason": "prefix_launcher_ambiguous",
        "frame": 0,
    }
    assert evidence["box_clear_runs"] == []
    assert result.releases == {}


def test_clearance_confirmation_needs_actual_consecutive_observations():
    # Two clear observations on either side of a missing image cannot make a
    # three-frame confirmed geometric run; no point is inserted at frame22.
    bey = make_track(
        1, "bey", {f: (100, 90 if f < 20 else 180) for f in [*range(22), 23, 24]}
    )
    body = make_track(9, "launcher", {f: (100, 70) for f in range(25)}, size=40)
    result = prepare_launch_tracks([bey, body], 30, load_launch_policy())
    assert result.releases == {}
    assert result.diagnostics[1]["box_clear_runs"] == []



def test_same_frame_handoff_keeps_original_observations_without_duplicate_frame():
    a = make_track(1, "bey", {f: (100 + f, 180) for f in range(5)}, size=40)
    b = make_track(2, "bey", {f: (100 + f, 180) for f in range(4, 9)}, size=40)
    result = prepare_launch_tracks([a, b], 30, load_launch_policy())
    assert result.members == {1: [1, 2]}
    assert [d.frame for d in result.tracks[0].dets] == list(range(9))
    assert result.tracks[0].dets[4] is a.dets[4]
    assert len(a.dets) == len(b.dets) == 5


def test_duplicate_overlap_keeps_replaced_raw_observations_as_evidence():
    a = make_track(1, "bey", {f: (100 + f, 180) for f in range(7)}, size=40)
    b = make_track(2, "bey", {f: (101 + f, 180) for f in range(4, 11)}, size=40)
    result = prepare_launch_tracks([a, b], 30, load_launch_policy())
    assert result.members == {1: [1, 2]}
    assert len(result.tracks[0].dets) == 11
    join = result.record()["joins"][0]
    assert (join["from_track"], join["to_track"]) == (1, 2)
    assert [d["frame"] for d in join["replaced_observations"]] == [4, 5, 6]
    assert join["replaced_observations"][0]["box"] == b.dets[0].box


def test_overlapping_different_objects_and_missing_overlap_evidence_stay_separate():
    a = make_track(1, "bey", {f: (100 + f, 180) for f in range(7)}, size=40)
    b = make_track(2, "bey", {f: (100 + f, 210) for f in range(4, 11)}, size=40)
    missing = make_track(3, "bey", {f: (100 + f, 180) for f in (4, 6, 7, 8)}, size=40)
    for other in (b, missing):
        result = prepare_launch_tracks([a, other], 30, load_launch_policy())
        assert len(result.tracks) == 2


def test_fragment_continuation_never_combines_conflicting_side_hints():
    a = make_track(1, "bey", {f: (100 + f, 180) for f in range(5)}, size=40)
    b = make_track(2, "bey", {f: (100 + f, 180) for f in range(4, 9)}, size=40)
    result = prepare_launch_tracks([a, b], 30, load_launch_policy(), side_hints={1: "X", 2: "B"})
    assert result.members == {1: [1], 2: [2]}


def test_ambiguous_overlap_successors_and_full_lifetime_overlap_are_not_joined():
    a = make_track(1, "bey", {f: (100 + f, 180) for f in range(7)}, size=40)
    b = make_track(2, "bey", {f: (100 + f, 180) for f in range(4, 11)}, size=40)
    c = make_track(3, "bey", {f: (101 + f, 180) for f in range(4, 11)}, size=40)
    assert len(prepare_launch_tracks([a, b, c], 30, load_launch_policy()).tracks) == 3
    same_span = make_track(4, "bey", {f: (100 + f, 180) for f in range(7)}, size=40)
    assert len(prepare_launch_tracks([a, same_span], 30, load_launch_policy()).tracks) == 2


def test_same_frame_handoff_chain_is_order_independent_and_never_adds_observations():
    tracks = [make_track(i + 1, "bey", {f: (100 + f, 180) for f in range(i*4, i*4+5)}, size=40) for i in range(3)]
    result = prepare_launch_tracks(tracks[::-1], 30, load_launch_policy())
    assert result.members == {1: [1, 2, 3]}
    assert [d.frame for d in result.tracks[0].dets] == list(range(13))



def test_unknown_middle_fragment_cannot_bridge_opposite_known_sides():
    tracks = [make_track(i + 1, "bey", {f: (100 + f, 180) for f in range(i*4, i*4+5)}, size=40) for i in range(3)]
    result = prepare_launch_tracks(tracks, 30, load_launch_policy(), side_hints={1: "X", 3: "B"})
    assert result.members == {1: [1, 2], 3: [3]}
    assert len(result.joins) == 1



def test_single_box_jitter_does_not_reverse_the_short_motion_history():
    a = make_track(1, "bey", {0:(100,180),1:(102,180),2:(104,180),3:(106,180),4:(105,180)})
    b = make_track(2, "bey", {6:(109,180),7:(111,180),8:(113,180),9:(115,180),10:(117,180)})
    result = prepare_launch_tracks([a,b],30,load_launch_policy())
    assert result.members == {1:[1,2]}
    assert [d.frame for d in result.tracks[0].dets] == [0,1,2,3,4,6,7,8,9,10]
    assert result.joins[0]["motion"]["from_frames"] == [0,1,2,3,4]
    assert result.joins[0]["motion"]["from_velocity_px_s"] == [60.0,0.0]


def test_overlap_uses_actual_shared_position_instead_of_backward_extrapolation():
    # Stable two-pixel/frame motion with an endpoint box displaced by 10px.
    # Three actually shared frames agree within24px; backwards extrapolation
    # from the shifted tail incorrectly places the first match over24px away.
    a = make_track(1,"bey",{0:(90,180),1:(92,180),2:(94,180),3:(96,180),4:(108,180)},size=160)
    b = make_track(2,"bey",{2:(70,180),3:(72,180),4:(74,180),5:(76,180),6:(78,180)},size=160)
    result = prepare_launch_tracks([a,b],30,load_launch_policy())
    assert result.members == {1:[1,2]}
    assert result.joins[0]["motion"]["position_basis"] == "observed_shared_frame"
    assert result.joins[0]["motion"]["position_error_px"] == 24.0


def test_stationary_top_is_not_joined_to_a_different_moving_top():
    a=make_track(1,"bey",{f:(100,180) for f in range(5)},size=60)
    b=make_track(2,"bey",{f:(100+f*2,180) for f in range(4,9)},size=60)
    result=prepare_launch_tracks([a,b],30,load_launch_policy())
    assert result.members == {1:[1],2:[2]}


def test_reliable_opposite_motion_and_long_occlusion_remain_unjoined():
    a=make_track(1,"bey",{f:(100+f*2,180) for f in range(5)},size=60)
    b=make_track(2,"bey",{f:(116-(f-4)*2,180) for f in range(4,9)},size=60)
    far=make_track(3,"bey",{f:(100+f*2,180) for f in range(40,45)},size=60)
    for other in (b,far):
        result=prepare_launch_tracks([a,other],30,load_launch_policy())
        assert len(result.tracks)==2



def test_velocity_history_is_bounded_by_source_time_and_does_not_fill_gaps():
    from bxjudge.launch_phase import _endpoint_motion
    a=make_track(1,"bey",{0:(900,180),10:(100,180),12:(104,180),13:(106,180),14:(108,180)})
    velocity, frames = _endpoint_motion(a.dets, 5, tail=True)
    assert velocity == (2.0,0.0)
    assert frames == [10,12,13,14]
    assert [d.frame for d in a.dets] == [0,10,12,13,14]


def test_velocity_window_policy_requires_explicit_integer_budget(tmp_path):
    import pytest
    import yaml

    from bxjudge.launch_phase import load_launch_policy
    base=load_launch_policy()
    for invalid in (True, 1, 2.5, 0, -1):
        path=tmp_path/"policy.yaml"
        path.write_text(yaml.safe_dump({**base,"link_velocity_frames":invalid}))
        with pytest.raises(ValueError):
            load_launch_policy(path)
