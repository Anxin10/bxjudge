import dataclasses

import numpy as np

from bxjudge.geometry import box_center, polygon_mask
from bxjudge.types import LaunchCheck, RoundResult
from tests.helpers import make_calibration, make_track


def test_track_accessors():
    t = make_track(1, "bey", {5: (10, 10), 6: (12, 10), 8: (20, 10)}, size=4.0)
    assert t.first_frame() == 5 and t.last_frame() == 8
    assert t.box_at(6) == (10.0, 8.0, 14.0, 12.0)
    assert t.box_at(7) is None and t.center_at(7) is None
    assert t.center_at(8) == (20.0, 10.0)
    assert t.centers() == {5: (10.0, 10.0), 6: (12.0, 10.0), 8: (20.0, 10.0)}
    assert box_center(t.box_at(5)) == (10.0, 10.0)


def test_round_result_asdict():
    lc = LaunchCheck(
        touch={"X": "no", "B": "unknown"},
        landed={"X": True, "B": True},
        landing_frame={"X": 10, "B": None},
        landing_zone={"X": "battle_area", "B": None},
        air_collision="no",
        notes=[],
    )
    r = RoundResult("v.mp4", 1, 0.0, 5.0, lc, "spin", "X", 1, 0.9, False, [])
    d = dataclasses.asdict(r)
    assert d["launch"]["landing_zone"]["X"] == "battle_area" and d["reviewed"] is False


def test_polygon_mask():
    c = make_calibration()
    m = polygon_mask(c.battle_area, c.frame_size)
    assert m.shape == (360, 640) and m.dtype == np.uint8
    assert m[150, 300] == 255 and m[10, 10] == 0
