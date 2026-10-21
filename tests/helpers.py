"""Shared test helpers for the whole project (later tasks add more here)."""
from __future__ import annotations

from bxjudge.types import Box, Calibration, Detection, Point, Polygon, Track


def rect(x1: float, y1: float, x2: float, y2: float) -> Polygon:
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def make_track(
    track_id: int,
    cls: str,
    centers: dict[int, Point],
    size: float = 20.0,
    score: float = 0.9,
) -> Track:
    half = size / 2.0
    dets = []
    for frame in sorted(centers):
        cx, cy = centers[frame]
        box: Box = (cx - half, cy - half, cx + half, cy + half)
        dets.append(Detection(frame=frame, cls=cls, box=box, score=score))
    return Track(track_id=track_id, cls=cls, dets=dets)


def make_calibration() -> Calibration:
    """640x360 rectangular stand-in calibration."""
    return Calibration(
        name="test",
        frame_size=(640, 360),
        stadium=rect(40, 20, 600, 340),
        lid=rect(60, 25, 580, 120),
        battle_area=rect(120, 60, 520, 280),
        over_left=rect(40, 200, 110, 300),
        over_right=rect(530, 200, 600, 300),
        xtreme=rect(280, 290, 360, 340),
        launch_left=rect(40, 0, 320, 120),
        launch_right=rect(320, 0, 600, 120),
        ref_frame="",
    )
