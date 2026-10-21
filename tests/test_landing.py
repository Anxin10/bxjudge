from bxjudge.landing import find_landing_frame, find_settle
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
