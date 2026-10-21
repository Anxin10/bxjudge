from bxjudge.geometry import point_in_polygon, zone_of, boxes_touch, is_abrupt_change, velocities
from tests.helpers import make_calibration


def test_point_in_polygon():
    c = make_calibration()
    assert point_in_polygon((300, 150), c.battle_area)
    assert not point_in_polygon((50, 10), c.battle_area)


def test_zone_of():
    c = make_calibration()
    assert zone_of((300, 200), c) == "battle_area"
    assert zone_of((300, 100), c) == "battle_area"      # 與蓋子重疊處以戰鬥區為準
    assert zone_of((80, 60), c) == "lid"
    assert zone_of((320, 320), c) == "xtreme"
    assert zone_of((70, 250), c) == "over_left" and zone_of((560, 250), c) == "over_right"
    assert zone_of((150, 320), c) == "stadium_other" and zone_of((5, 5), c) == "outside"


def test_boxes_touch_with_margin():
    assert boxes_touch((0,0,10,10), (10,0,20,10))
    assert not boxes_touch((0,0,10,10), (15,0,25,10))
    assert boxes_touch((0,0,10,10), (15,0,25,10), margin=5)


def test_abrupt_change():
    assert is_abrupt_change((10,0), (0,10), angle_deg=45, speed_ratio=0.5)      # 90° 轉向
    assert is_abrupt_change((10,0), (4,0), angle_deg=45, speed_ratio=0.5)       # 速度降 60%
    assert not is_abrupt_change((10,0), (9,1), angle_deg=45, speed_ratio=0.5)


def test_velocities():
    v = velocities({0:(0,0), 1:(3,4), 2:(6,8)})
    assert v[1] == (3,4) and v[2] == (3,4)
