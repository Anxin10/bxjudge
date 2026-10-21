from dataclasses import replace

import pytest

from bxjudge.landing import find_landing_frame, find_settle, find_settles
from tests.helpers import (
    fall_then_rest,
    make_calibration,
    make_track,
    parabola_then_orbit,
)


def test_parabola_then_orbit_lands_near_truth(th):
    t = make_track(1, "bey", parabola_then_orbit(land_at=30))
    assert abs(find_landing_frame(t, make_calibration(), th) - 30) <= 2
    assert find_settle(t, make_calibration(), th)[1] == "battle_area"


def test_parabola_other_landing_time(th):
    t = make_track(1, "bey", parabola_then_orbit(land_at=45))
    assert abs(find_landing_frame(t, make_calibration(), th) - 45) <= 2


def test_projected_inside_while_falling_not_landed(th):
    centers = {f: (300, 70 + 10 * f) for f in range(15)}  # projected in battle_area, still falling fast
    centers.update({f: (300 + f, 220) for f in range(15, 40)})
    assert find_landing_frame(make_track(1, "bey", centers), make_calibration(), th) >= 15


def test_settles_on_lid_not_landed(th):
    centers = {f: (80, 30 + 3 * f) for f in range(10)}  # reaches the lid-only area
    centers.update({f: (80, 60) for f in range(10, 40)})
    t = make_track(1, "bey", centers)
    assert find_settle(t, make_calibration(), th)[1] == "lid"
    assert find_landing_frame(t, make_calibration(), th) is None


def test_never_enters_battle_area(th):
    centers = {f: (20, 10 + f) for f in range(40)}
    assert find_landing_frame(make_track(1, "bey", centers), make_calibration(), th) is None


def test_fall_then_rest_in_battle_area(th):
    t = make_track(1, "bey", fall_then_rest(20, (300, 200), (300, 20)))
    assert find_settle(t, make_calibration(), th) == (20, "battle_area")
    assert find_landing_frame(t, make_calibration(), th) == 20


def test_falls_into_over_zone_not_landed(th):
    t = make_track(1, "bey", fall_then_rest(20, (75, 250), (75, 100)))
    assert find_settle(t, make_calibration(), th) == (20, "over_left")
    assert find_landing_frame(t, make_calibration(), th) is None


def test_detection_gap_during_settling_still_lands(th):
    # gap rule: the window may skip frames; consecutive missing detections inside it are fine
    for missing in ({35}, {31, 32}, {30}, {33, 34}):
        centers = {f: p for f, p in parabola_then_orbit(land_at=30).items() if f not in missing}
        t = make_track(1, "bey", centers)
        got = find_landing_frame(t, make_calibration(), th)
        assert got is not None and abs(got - 30) <= 3, (missing, got)


def test_long_blackout_inside_window_does_not_settle(th):
    centers = fall_then_rest(15, (300, 200), (300, 20), n=18)  # frames 0..17
    centers.update({f: (300, 200) for f in range(40, 60)})  # 22-frame blackout, then at rest
    t = make_track(1, "bey", centers)
    assert find_settle(t, make_calibration(), th) == (40, "battle_area")



def test_zone_change_inside_window_does_not_settle(th):
    # slow drift lid -> battle_area along y=100, crossing x=120 at frame 5: a window must be one zone
    t = make_track(1, "bey", {f: (110 + 2.0 * f, 100) for f in range(30)})
    assert find_settle(t, make_calibration(), th) == (5, "battle_area")


def test_settle_episodes_follow_lid_rim_then_battle_area(th):
    # Each transition includes a fast vertical fall followed by a full quiet window.
    n = th.settle_frames
    centers = {0: (80, 0), 1: (80, 30), 2: (80, 60)}
    centers.update({f: (80, 60) for f in range(3, n + 3)})
    centers.update({f: (80, 150) for f in range(n + 3, 2 * n + 4)})
    centers.update({f: (300, 220) for f in range(2 * n + 4, 3 * n + 5)})
    t = make_track(1, "bey", centers)
    c = make_calibration()

    assert find_settles(t, c, th) == [
        (3, "lid"), (n + 4, "stadium_other"), (2 * n + 5, "battle_area")
    ]
    assert find_settle(t, c, th) == (3, "lid")
    assert find_landing_frame(t, c, th) == 2 * n + 5


def test_continued_settle_emits_one_episode(th):
    t = make_track(1, "bey", fall_then_rest(20, (300, 200), (300, 20)))
    assert find_settles(t, make_calibration(), th) == [(20, "battle_area")]


def test_fast_vertical_motion_splits_same_zone_settles(th):
    n = th.settle_frames
    centers = {0: (300, 60), 1: (300, 100)}
    centers.update({f: (300, 100) for f in range(2, n + 2)})
    centers.update({f: (300, 180) for f in range(n + 2, 2 * n + 3)})
    t = make_track(1, "bey", centers)
    assert find_settles(t, make_calibration(), th) == [
        (2, "battle_area"), (n + 3, "battle_area")
    ]


def test_lid_then_projected_battle_area_still_falling_is_not_landing(th):
    n = th.settle_frames
    centers = {0: (80, 0), 1: (80, 60)}
    centers.update({f: (80, 60) for f in range(2, n + 2)})
    centers.update({n + 2 + i: (300, 80 + 10 * i) for i in range(n)})
    t = make_track(1, "bey", centers)
    assert find_settles(t, make_calibration(), th) == [(2, "lid")]
    assert find_landing_frame(t, make_calibration(), th) is None


def test_zone_change_splits_settles_without_fast_vertical_motion(th):
    n = th.settle_frames
    centers = {f: (80, 100) for f in range(n)}
    centers.update({f: (300, 100) for f in range(n, 2 * n)})
    centers.update({f: (80, 100) for f in range(2 * n, 3 * n)})
    assert find_settles(make_track(1, "bey", centers), make_calibration(), th) == [
        (0, "lid"), (n, "battle_area"), (2 * n, "lid")
    ]


def test_long_detection_gap_splits_settles(th):
    n = th.settle_frames
    centers = {f: (300, 200) for f in range(n)}
    centers.update({f: (300, 200) for f in range(3 * n, 4 * n)})
    assert find_settles(make_track(1, "bey", centers), make_calibration(), th) == [
        (0, "battle_area"), (3 * n, "battle_area")
    ]


@pytest.mark.parametrize("centers", [{}, {0: (300, 200)}])
def test_track_without_full_settle_window_has_no_episodes(centers, th):
    t = make_track(1, "bey", centers)
    c = make_calibration()
    assert find_settles(t, c, th) == []
    assert find_settle(t, c, th) == (None, None)
    assert find_landing_frame(t, c, th) is None


def test_short_gap_does_not_repeat_settle_episode(th):
    centers = {f: p for f, p in fall_then_rest(20, (300, 200), (300, 20)).items() if f != 30}
    assert find_settles(make_track(1, "bey", centers), make_calibration(), th) == [
        (20, "battle_area")
    ]


def test_episode_duration_and_vertical_speed_use_configured_thresholds(th):
    c = make_calibration()
    t = make_track(1, "bey", {f: (300, 100 + 2 * f) for f in range(5)})
    assert find_settles(t, c, replace(th, settle_frames=5, fall_speed_px=2)) == [
        (0, "battle_area")
    ]
    assert find_settles(t, c, replace(th, settle_frames=6, fall_speed_px=2)) == []
    assert find_settles(t, c, replace(th, settle_frames=5, fall_speed_px=1)) == []
