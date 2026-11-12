import math
from dataclasses import replace

import pytest

from bxjudge.launch_check import (
    assign_sides, check_launch, detect_air_collision, detect_touch, impact_frames,
)
from tests.helpers import make_calibration, make_track, parabola_then_orbit, fall_then_rest


def crossing_tracks(bounce=True, at=10):
    x, b = {}, {}
    for f in range(at + 10):
        offset = abs(f - at) if bounce else at - f
        x[f] = (300 - 12 * offset, 150)
        b[f] = (300 + 12 * offset, 150)
    return make_track(1, 'bey', x), make_track(2, 'bey', b)


def pair(n=120):
    return [make_track(1, 'bey', parabola_then_orbit(30, n=n)),
            make_track(2, 'bey', parabola_then_orbit(32, start=(500, 20), n=n))]


def pulse(at=10, n=60):
    return {f: 5.0 if f == at else 0.0 for f in range(n)}


def test_assign_first_centers_and_stadium_centroid():
    tracks = [make_track(1, 'bey', {1: (100, 50), 3: (500, 50)}),
              make_track(2, 'bey', {0: (500, 50)}),
              make_track(3, 'bey', {0: (250, 200)}),
              make_track(4, 'bey', {0: (400, 200)})]
    assert assign_sides(tracks, make_calibration()) == {1: 'X', 2: 'B', 3: 'X', 4: 'B'}


def test_empty_or_duplicate_track_id_does_not_invent_mapping():
    tracks = [make_track(1, 'bey', {}), make_track(2, 'bey', {0: (100, 50)}),
              make_track(2, 'bey', {0: (500, 50)})]
    assert assign_sides(tracks, make_calibration()) == {}


def test_left_body_at_rim_and_supplied_launcher_side(th):
    launcher = make_track(9, 'launcher', {10: (100, 22)})
    assert detect_touch(pulse(), [launcher], {9: 'X'}, set(), (0, 30), make_calibration(), th)[0] == {'X': 'yes', 'B': 'no'}
    assert detect_touch(pulse(), [launcher], {9: 'B'}, set(), (0, 30), make_calibration(), th)[0] == {'X': 'no', 'B': 'yes'}


def test_landing_shake_exclusion_is_inclusive(th):
    touch, notes = detect_touch(pulse(), [], {}, {13}, (0, 30), make_calibration(), th)
    assert touch == {'X': 'no', 'B': 'no'}
    assert 'shake_by_bey_impact@10' in notes


def test_shake_without_body_or_unattributed_body_is_unknown(th):
    for launchers in ([], [make_track(9, 'launcher', {10: (100, 22)})],
                      [make_track(9, 'ripcord', {10: (100, 22)})]):
        touch, notes = detect_touch(pulse(), launchers, {}, set(), (0, 30), make_calibration(), th)
        assert touch == {'X': 'unknown', 'B': 'unknown'}
        assert 'shake_cause_unknown@10' in notes


def test_visible_body_away_rim_is_no_and_string_cannot_add_touch(th):
    launchers = [make_track(9, 'launcher', {10: (150, -10)}),
                 make_track(10, 'ripcord', {10: (100, 22)})]
    touch, notes = detect_touch(pulse(), launchers, {9: 'X', 10: 'B'}, set(), (0, 30), make_calibration(), th)
    assert touch == {'X': 'no', 'B': 'no'}
    assert 'shake_not_by_launcher_body@10' in notes


def test_shake_outside_window_ignored_and_only_onset_attributed(th):
    launcher = make_track(9, 'launcher', {11: (100, 22)})
    assert detect_touch(pulse(50), [], {}, set(), (0, 30), make_calibration(), th)[0] == {'X': 'no', 'B': 'no'}
    disp = pulse()
    disp[11] = 5.0
    assert detect_touch(disp, [launcher], {9: 'X'}, set(), (0, 30), make_calibration(), th)[0] == {'X': 'unknown', 'B': 'unknown'}


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_motion_never_means_no_touch(value, th):
    disp = {f: 0.0 for f in range(30)}
    disp[10] = value
    touch, notes = detect_touch(disp, [], {}, set(), (0, 29), make_calibration(), th)
    assert touch == {'X': 'unknown', 'B': 'unknown'}
    assert 'stadium_motion_unknown@10' in notes


def test_impacts_landing_box_contact_and_abrupt_rim(th):
    x, b = crossing_tracks()
    impacts = impact_frames([x, b], {1: 30, 2: None}, make_calibration(), th)
    assert 10 in impacts and 30 in impacts
    rim = make_track(3, 'bey', {0: (42, 100), 1: (42, 110), 2: (42, 100)})
    assert 1 in impact_frames([rim], {}, make_calibration(), th)


def test_tiny_jitter_and_unobserved_turn_are_not_impacts(th):
    for centers in ({0: (42, 100), 1: (42.01, 100), 2: (42, 100)},
                    {0: (42, 100), 2: (42, 120), 4: (42, 100)}):
        assert impact_frames([make_track(1, 'bey', centers)], {}, make_calibration(), th) == set()


def test_air_yes_maybe_no_and_pre_first_landing(th):
    assert detect_air_collision(*crossing_tracks(), 40, 42, th) == 'yes'
    assert detect_air_collision(*crossing_tracks(False), 40, 42, th) == 'maybe'
    assert detect_air_collision(*crossing_tracks(at=45), 40, 50, th) == 'no'
    a = make_track(1, 'bey', {f: (100 + f, 100) for f in range(20)})
    b = make_track(2, 'bey', {f: (500 + f, 100) for f in range(20)})
    assert detect_air_collision(a, b, None, None, th) == 'no'
    assert detect_air_collision(*crossing_tracks(), None, 40, th) == 'yes'


def test_air_jitter_is_not_both_abrupt_and_gaps_are_not_motion_evidence(th):
    a = make_track(1, 'bey', {0: (300, 150), 1: (300.01, 150), 2: (300, 150)})
    b = make_track(2, 'bey', {0: (300, 150), 1: (299.99, 150), 2: (300, 150)})
    assert detect_air_collision(a, b, None, None, th) == 'maybe'
    a, b = crossing_tracks()
    a.dets = [d for d in a.dets if d.frame != 9]
    b.dets = [d for d in b.dets if d.frame != 9]
    assert detect_air_collision(a, b, 40, 42, th) == 'maybe'


def _separated_tracks(a_frames, b_frames):
    a = make_track(1, 'bey', {f: (100 + f, 100) for f in a_frames})
    b = make_track(2, 'bey', {f: (500 + f, 100) for f in b_frames})
    return a, b


def test_air_no_requires_observation_coverage_of_airborne_interval(th):
    # X is unobserved for half of the airborne period [0, 42): absence of
    # evidence must not become a confident "no" (R31).
    a, b = _separated_tracks(range(21), range(42))
    assert detect_air_collision(a, b, 42, 42, th) == 'maybe'
    # A leading gap is equally missing evidence.
    a, b = _separated_tracks(range(21, 42), range(42))
    assert detect_air_collision(a, b, 42, 42, th) == 'maybe'
    # Without any landing cutoff the interval is the span of observations.
    a, b = _separated_tracks(range(20), range(60))
    assert detect_air_collision(a, b, None, None, th) == 'maybe'


def test_air_small_unobserved_stretch_within_tolerance_is_still_no(th):
    a, b = _separated_tracks(range(40), range(42))  # 2 of 42 frames missing
    assert detect_air_collision(a, b, 42, 42, th) == 'no'
    # The tolerance is configuration, not a code constant.
    strict = replace(th, air_evidence_max_missing_fraction=0.0)
    assert detect_air_collision(a, b, 42, 42, strict) == 'maybe'
    lenient = replace(th, air_evidence_max_missing_fraction=0.6)
    a, b = _separated_tracks(range(21), range(42))
    assert detect_air_collision(a, b, 42, 42, lenient) == 'no'


def test_check_launch_air_gap_is_maybe_with_note_not_confident_no(th):
    # X falls and is observed for frames 0-20, lost for 21-42, re-detected at rest
    # from 43; B lands at 42. Any collision in X's gap is invisible.
    x = {f: (150 + 100 * f / 40, -100 + 290 * f / 40) for f in range(21)}
    x.update({f: (250, 190) for f in range(43, 100)})
    b = {f: (500 - 150 * f / 41, -100 + 290 * f / 41) for f in range(42)}
    b.update({f: (350, 190) for f in range(42, 100)})
    lc, _ = check_launch([make_track(1, 'bey', x), make_track(2, 'bey', b)], [],
                         {f: 0.0 for f in range(200)}, make_calibration(), th, 60)
    assert lc.landing_frame == {'X': 44, 'B': 42}
    assert lc.air_collision == 'maybe'
    assert 'air_evidence_incomplete@X' in lc.notes
    assert 'air_evidence_incomplete@B' not in lc.notes


def test_landing_zone_recorded(th):
    tracks = pair()
    tracks[1] = make_track(2, 'bey', fall_then_rest(32, (560, 60), (500, -200)))
    lc, sides = check_launch(tracks, [], {f: 0.0 for f in range(200)}, make_calibration(), th, 60)
    assert sides == {1: 'X', 2: 'B'}
    assert lc.landing_zone == {'X': 'battle_area', 'B': 'lid'}
    assert lc.landed == {'X': True, 'B': False}


def test_lid_then_battle_settle_in_window_wins(th):
    centers = {f: (80, 60) for f in range(10)}
    centers.update({f: (250, 180) for f in range(10, 40)})
    tracks = [make_track(1, 'bey', centers), make_track(2, 'bey', {f: (500, 100) for f in range(40)})]
    lc, _ = check_launch(tracks, [], {f: 0.0 for f in range(200)}, make_calibration(), th, 60)
    assert lc.landed == {'X': True, 'B': True}
    assert lc.landing_zone['X'] == 'battle_area' and lc.landing_frame['X'] == 11


def test_missing_landing_cap_is_configured_and_rejects_late_battle_settle(th):
    centers = {0: (80, 0), 1: (80, 60)}
    centers.update({f: (80, 60) for f in range(2, 40)})
    centers.update({f: (250, 180) for f in range(40, 100)})
    tracks = [make_track(1, 'bey', centers), make_track(2, 'bey', {f: (500, 100) for f in range(100)})]
    short = replace(th, launch_missing_landing_s=0.5)
    lc, _ = check_launch(tracks, [], pulse(50, 200), make_calibration(), short, 60)
    assert lc.landing_zone['X'] == 'lid' and not lc.landed['X']
    assert lc.touch == {'X': 'no', 'B': 'no'}
    lc, _ = check_launch(tracks, [], pulse(50, 200), make_calibration(), th, 60)
    assert lc.landed['X'] and lc.touch == {'X': 'unknown', 'B': 'unknown'}


def test_check_launch_uses_both_launcher_launch_regions(th):
    tracks = pair()
    launchers = [make_track(9, 'launcher', {0: (100, 30), 5: (100, 22)}),
                 make_track(10, 'launcher', {0: (500, 30), 8: (500, 22)})]
    disp = pulse(5, 200)
    disp[8] = 5.0
    lc, _ = check_launch(tracks, launchers, disp, make_calibration(), th, 60)
    assert lc.touch == {'X': 'yes', 'B': 'yes'}


def test_launch_window_post_boundary_and_pre_boundary(th):
    tracks = pair()
    for t in tracks:
        for d in t.dets:
            d.frame += 40
    lc, _ = check_launch(tracks, [], pulse(9, 200), make_calibration(), th, 60)
    assert lc.touch == {'X': 'no', 'B': 'no'}
    lc, _ = check_launch(tracks, [], pulse(10, 200), make_calibration(), th, 60)
    assert lc.touch == {'X': 'unknown', 'B': 'unknown'}
    lc, _ = check_launch(tracks, [], pulse(110, 200), make_calibration(), th, 60)
    assert lc.touch == {'X': 'no', 'B': 'no'}


@pytest.mark.parametrize('tracks', [[], [make_track(1, 'bey', {0: (100, 50)})],
    [make_track(1, 'bey', {0: (100, 50)}), make_track(2, 'bey', {0: (110, 50)})]])
def test_missing_or_duplicate_side_requires_review(tracks, th):
    lc, _ = check_launch(tracks, [], {}, make_calibration(), th, 60)
    assert lc.air_collision == 'maybe'
    assert 'unknown' in lc.touch.values()
    assert any('track_evidence' in note for note in lc.notes)

def test_air_turn_evidence_cannot_use_post_landing_frame(th):
    # The only direction reversal is at the last airborne frame. Its outgoing
    # velocity belongs to the landing frame, so it cannot prove an air collision.
    a, b = crossing_tracks(at=10)
    assert detect_air_collision(a, b, 11, 40, th) == 'maybe'


def test_body_box_distance_uses_inclusive_configured_margin(th):
    body = make_track(9, 'launcher', {10: (100, -5)})  # box bottom5, rim20: gap15
    args = (pulse(), [body], {9: 'X'}, set(), (0, 30), make_calibration())
    assert detect_touch(*args, th)[0]['X'] == 'yes'
    assert detect_touch(*args, replace(th, rim_margin_px=14))[0] == {'X': 'no', 'B': 'no'}


def test_unknown_motion_and_unattributed_shakes_do_not_erase_known_touch(th):
    body = make_track(9, 'launcher', {10: (100, 22)})
    disp = pulse()
    disp[20] = float('nan')
    disp[25] = 5.0
    touch, _ = detect_touch(disp, [body], {9: 'X'}, set(), (0, 30), make_calibration(), th)
    assert touch == {'X': 'yes', 'B': 'unknown'}


def test_duplicate_launcher_id_cannot_create_a_known_side_touch(th):
    bodies = [make_track(9, 'launcher', {10: (100, 22)}),
              make_track(9, 'launcher', {10: (500, 22)})]
    touch, _ = detect_touch(pulse(), bodies, {9: 'X'}, set(), (0, 30), make_calibration(), th)
    assert touch == {'X': 'unknown', 'B': 'unknown'}


def test_never_settled_is_none_and_failed_landing(th):
    tracks = [make_track(1, 'bey', {f: (100, 50 + 10 * f) for f in range(10)}),
              make_track(2, 'bey', {f: (500, 50 + 10 * f) for f in range(10)})]
    lc, _ = check_launch(tracks, [], pulse(170, 200), make_calibration(), th, 60)
    assert lc.landing_frame == {'X': None, 'B': None}
    assert lc.landing_zone == {'X': None, 'B': None}
    assert lc.landed == {'X': False, 'B': False}
    assert lc.touch == {'X': 'unknown', 'B': 'unknown'}


def test_late_battle_settle_cannot_retroactively_extend_launch(th):
    tracks = []
    for tid, startx in ((1, 80), (2, 560)):
        centers = {0: (startx, 0), 1: (startx, 60)}
        centers.update({f: (startx, 60) for f in range(2, 190)})
        centers.update({f: (250 + tid * 10, 180) for f in range(190, 220)})
        tracks.append(make_track(tid, 'bey', centers))
    lc, _ = check_launch(tracks, [], pulse(200, 240), make_calibration(), th, 60)
    assert lc.landed == {'X': False, 'B': False}
    assert lc.landing_zone == {'X': 'lid', 'B': 'lid'}
    assert lc.touch == {'X': 'no', 'B': 'no'}
    assert 'landing_timing_unknown@X:191' in lc.notes
    assert 'landing_timing_unknown@B:191' in lc.notes


def test_full_settle_evidence_must_fit_deadline(th):
    tracks = [make_track(1, 'bey', {f: (250, 180) for f in range(6)}),
              make_track(2, 'bey', {f: (500, 100) for f in range(6)})]
    lc, _ = check_launch(tracks, [], {f: 0.0 for f in range(10)}, make_calibration(), replace(th, launch_missing_landing_s=0.05), 60)
    assert lc.landed == {'X': False, 'B': False}


def test_meaningful_motion_and_shake_thresholds_are_configured(th):
    a, b = crossing_tracks()
    assert detect_air_collision(a, b, 40, 42, replace(th, abrupt_min_speed_px=100)) == 'maybe'
    body = make_track(9, 'launcher', {10: (100, 22)})
    assert detect_touch(pulse(), [body], {9: 'X'}, set(), (0, 30), make_calibration(), replace(th, shake_px=5))[0] == {'X': 'no', 'B': 'no'}


def test_duplicate_frame_track_is_not_arbitrarily_assigned():
    track = make_track(1, 'bey', {0: (100, 50)})
    track.dets.extend(make_track(1, 'bey', {0: (500, 50)}).dets)
    assert assign_sides([track], make_calibration()) == {}

def test_each_earlier_settle_suppresses_shake_before_selected_battle_landing(th):
    centers = {0: (80, 0), 1: (80, 60)}
    centers.update({f: (80, 60) for f in range(2, 10)})
    centers.update({f: (250, 180) for f in range(10, 40)})
    tracks = [make_track(1, 'bey', centers),
              make_track(2, 'bey', {f: (500, 100) for f in range(40)})]
    body = make_track(9, 'launcher', {2: (100, 22)})
    lc, _ = check_launch(tracks, [body], pulse(2, 200), make_calibration(), th, 60)
    assert lc.landing_frame['X'] == 11
    assert lc.touch == {'X': 'no', 'B': 'no'}
    assert 'shake_by_bey_impact@2' in lc.notes


def test_air_cutoff_requires_battle_landing_not_lid_settle(th):
    a, b = crossing_tracks(at=20)
    for track in (a, b):
        # Fast initial vertical fall then a lid-only path and bounce.
        for d in track.dets:
            d.box = (d.box[0], -10 if d.frame == 0 else 40,
                     d.box[2], 10 if d.frame == 0 else 60)
    lc, _ = check_launch([a, b], [], {f: 0.0 for f in range(200)}, make_calibration(), th, 60)
    assert lc.landing_zone == {'X': 'lid', 'B': 'lid'}
    assert lc.landed == {'X': False, 'B': False}
    assert lc.air_collision == 'yes'

@pytest.mark.parametrize('missing', [set(range(1, 30)), {0}, {15}, {29}])
def test_partial_motion_coverage_leaves_both_sides_unknown(missing, th):
    disp = {f: 0.0 for f in range(30) if f not in missing}
    touch, notes = detect_touch(disp, [], {}, set(), (0, 30), make_calibration(), th)
    assert touch == {'X': 'unknown', 'B': 'unknown'}
    assert 'stadium_motion_missing' in notes


@pytest.mark.parametrize('missing', [5, 20])
def test_missing_motion_preserves_affirmative_touch(missing, th):
    disp = pulse(n=30)
    del disp[missing]
    body = make_track(9, 'launcher', {10: (100, 22)})
    touch, notes = detect_touch(disp, [body], {9: 'X'}, set(), (0, 30), make_calibration(), th)
    assert touch == {'X': 'yes', 'B': 'unknown'}
    assert 'stadium_motion_missing' in notes


@pytest.mark.parametrize(('at', 'expected'), [
    (9, {'X': 'no', 'B': 'no'}),
    (10, {'X': 'unknown', 'B': 'unknown'}),
    (29, {'X': 'unknown', 'B': 'unknown'}),
    (30, {'X': 'no', 'B': 'no'}),
])
def test_touch_window_is_start_inclusive_end_exclusive(at, expected, th):
    touch, _ = detect_touch(pulse(at, 31), [], {}, set(), (10, 30), make_calibration(), th)
    assert touch == expected


def test_end_frame_is_not_required_motion_evidence(th):
    touch, notes = detect_touch({f: 0.0 for f in range(30)}, [], {}, set(),
                                (0, 30), make_calibration(), th)
    assert touch == {'X': 'no', 'B': 'no'}
    assert notes == []


@pytest.mark.parametrize(('at', 'expected'), [
    (31, {'X': 'unknown', 'B': 'unknown'}),
    (32, {'X': 'no', 'B': 'no'}),
])
def test_successful_launch_touch_window_excludes_post_end_frame(at, expected, th):
    # Separate floor rests at frame 2; the configured 0.5 s interval ends at 32.
    tracks = [make_track(tid, 'bey', fall_then_rest(2, (x, 180), (x, 0)))
              for tid, x in ((1, 150), (2, 500))]
    lc, _ = check_launch(tracks, [], pulse(at, 33), make_calibration(), th, 60)
    assert lc.landing_frame == {'X': 2, 'B': 2}
    assert lc.touch == expected


def test_post_deadline_physical_settle_still_excludes_touch(th):
    x = {0: (150, 0), 1: (150, 100)}
    x.update({f: (150, 100) for f in range(2, 181)})
    x.update({f: (150, 120) for f in range(181, 221)})
    b = {0: (560, 0), 1: (560, 60)}
    b.update({f: (560, 60) for f in range(2, 170)})
    b.update({f: (500, 180) for f in range(170, 221)})
    tracks = [make_track(1, 'bey', x), make_track(2, 'bey', b)]
    body = make_track(9, 'launcher', {182: (100, 22)})
    # Both successful launches extend [0, 201), past the eligibility cap 180.
    lc, _ = check_launch(tracks, [body], pulse(182, 201), make_calibration(), th, 60)
    assert lc.landed == {'X': True, 'B': True}
    assert lc.landing_frame == {'X': 2, 'B': 171}
    assert lc.touch == {'X': 'no', 'B': 'no'}
    assert 'shake_by_bey_impact@182' in lc.notes


def test_late_physical_floor_still_cuts_off_ground_collision(th):
    tracks = []
    for tid, startx, floorx, direction in ((1, 80, 180, -1), (2, 560, 420, 1)):
        centers = {0: (startx, 0), 1: (startx, 60)}
        centers.update({f: (startx, 60) for f in range(2, 190)})
        centers.update({f: (floorx, 180) for f in range(190, 200)})
        centers.update({f: (300 + direction * 12 * abs(f - 210), 180)
                        for f in range(200, 231)})
        tracks.append(make_track(tid, 'bey', centers))
    lc, _ = check_launch(tracks, [], pulse(250, 180), make_calibration(), th, 60)
    assert lc.landed == {'X': False, 'B': False}
    assert lc.landing_frame == {'X': 2, 'B': 2}
    assert lc.landing_zone == {'X': 'lid', 'B': 'lid'}
    assert 'landing_timing_unknown@X:191' in lc.notes
    assert 'landing_timing_unknown@B:191' in lc.notes
    assert lc.air_collision == 'maybe'

def test_motion_unknown_reason_makes_touch_unknown_without_scanning(th, monkeypatch):
    """A handheld camera moves the rim the shake detector measures: touch is unknown, with
    one note naming why, and the shake scan never runs (plan 2026-11-12 Task 6)."""
    import bxjudge.launch_check as lc_mod
    from bxjudge.launch_check import analyze_launch

    tracks = pair()
    baseline = analyze_launch(tracks, [], {f: 0.0 for f in range(200)}, make_calibration(),
                              th, 60).check
    assert baseline.touch == {"X": "no", "B": "no"}

    def no_scan(*a, **k):
        raise AssertionError("detect_touch must not run")

    monkeypatch.setattr(lc_mod, "detect_touch", no_scan)
    got = analyze_launch(tracks, [], {f: math.nan for f in range(200)}, make_calibration(), th,
                         60, motion_unknown_reason="camera_handheld").check
    assert got.touch == {"X": "unknown", "B": "unknown"}
    assert [n for n in got.notes if n.startswith("stadium_motion")] == \
        ["stadium_motion_unknown@camera_handheld"]
    assert (got.landed, got.landing_frame, got.landing_zone) == \
        (baseline.landed, baseline.landing_frame, baseline.landing_zone)


def test_unobserved_frames_in_the_landing_window_make_a_miss_unknown(th):
    """A top not seen landing while the stadium geometry was unknown is not a launch miss:
    landing is unknown (redo without a penalty), never not_landed (final review High)."""
    from bxjudge.launch_check import analyze_launch

    tracks = pair()
    tracks[1] = make_track(2, "bey", fall_then_rest(32, (560, 60), (500, -200)))  # B on the lid
    disp = {f: 0.0 for f in range(200)}
    plain = analyze_launch(tracks, [], disp, make_calibration(), th, 60).check
    assert plain.landed == {"X": True, "B": False}
    seen = analyze_launch(tracks, [], disp, make_calibration(), th, 60,
                          unobserved_frames={round(th.launch_missing_landing_s * 60) + 5}).check
    assert seen.landed == plain.landed  # gap outside the landing window: unchanged
    gap = analyze_launch(tracks, [], disp, make_calibration(), th, 60,
                         unobserved_frames={20, 21, 22}).check
    assert gap.landed == {"X": True, "B": None}
    assert "landing_observation_incomplete@B" in gap.notes


def test_a_side_whose_observations_were_all_dropped_is_unknown_not_a_miss(th):
    """Every observation of B fell in frames without geometry: no track at all is left for B.
    Its landing is unknown (redo, no miss counted), never not_landed (final re-review High)."""
    from bxjudge.launch_check import analyze_launch

    x_only = [pair()[0]]
    disp = {f: 0.0 for f in range(200)}
    plain = analyze_launch(x_only, [], disp, make_calibration(), th, 60).check
    assert plain.landed["B"] is False  # without geometry information: as before
    gap = analyze_launch(x_only, [], disp, make_calibration(), th, 60,
                         unobserved_frames=set(range(5, 40))).check
    assert gap.landed["B"] is None and "landing_observation_incomplete@B" in gap.notes


def test_unknown_frames_before_the_first_kept_observation_still_count(th):
    """The round starts at frame 5; frames 5-39 had no geometry, so every B observation and
    X's early frames were dropped and X's first kept observation is at 40. The landing window
    starts at the round start, not at that first kept observation (final re-review High)."""
    from bxjudge.launch_check import analyze_launch

    x = make_track(1, "bey", {f + 40: c for f, c in pair()[0].centers().items()})
    disp = {f: 0.0 for f in range(300)}
    unknown = set(range(5, 40))
    late = analyze_launch([x], [], disp, make_calibration(), th, 60,
                          unobserved_frames=unknown).check
    assert late.landed["B"] is False  # the window measured from frame 40 misses the gap
    got = analyze_launch([x], [], disp, make_calibration(), th, 60,
                         unobserved_frames=unknown, round_start=5).check
    assert got.landed["B"] is None and "landing_observation_incomplete@B" in got.notes


def test_a_round_with_every_observation_dropped_is_unknown_on_both_sides(th):
    from bxjudge.launch_check import analyze_launch

    start = round(th.launch_missing_landing_s * 60) + 100
    disp = {f: 0.0 for f in range(start + 400)}
    got = analyze_launch([], [], disp, make_calibration(), th, 60,
                         unobserved_frames=set(range(start, start + 50)),
                         round_start=start).check
    assert got.landed == {"X": None, "B": None}
    assert {"landing_observation_incomplete@X", "landing_observation_incomplete@B"} <= set(
        got.notes)
