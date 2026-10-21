"""Pure geometry helpers: zones, boxes, velocities."""
from __future__ import annotations

import math

import cv2
import numpy as np

from bxjudge.types import Box, Calibration, Point, Polygon, Zone

_EPS = 1e-9


def _contour(poly: Polygon) -> np.ndarray:
    return np.array(poly, dtype=np.float32).reshape(-1, 1, 2)


def point_in_polygon(p: Point, poly: Polygon) -> bool:
    """True if ``p`` is inside or on the boundary of ``poly``."""
    return cv2.pointPolygonTest(_contour(poly), (float(p[0]), float(p[1])), False) >= 0


def zone_of(p: Point, c: Calibration) -> Zone:
    """Classify a point. Order matters: xtreme, over_left, over_right, battle_area,
    lid, stadium_other, outside (battle_area wins where the lid overlaps it)."""
    if point_in_polygon(p, c.xtreme):
        return "xtreme"
    if point_in_polygon(p, c.over_left):
        return "over_left"
    if point_in_polygon(p, c.over_right):
        return "over_right"
    if point_in_polygon(p, c.battle_area):
        return "battle_area"
    if point_in_polygon(p, c.lid):
        return "lid"
    if point_in_polygon(p, c.stadium):
        return "stadium_other"
    return "outside"


def box_center(b: Box) -> Point:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def boxes_touch(a: Box, b: Box, margin: float = 0.0) -> bool:
    """True if the gap between the boxes is <= ``margin`` on both axes (edges touching count)."""
    gap_x = max(a[0] - b[2], b[0] - a[2])
    gap_y = max(a[1] - b[3], b[1] - a[3])
    return gap_x <= margin and gap_y <= margin


def polygon_mask(poly: Polygon, size: tuple[int, int]) -> np.ndarray:
    """Filled mask for ``poly``; ``size`` is (width, height) like ``Calibration.frame_size``.
    Returns a uint8 array of shape (height, width): 255 inside, 0 outside."""
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(np.array(poly, dtype=np.float64)).astype(np.int32)], 255)
    return mask


# Implementation constants for the rim region (not rule thresholds), in px at a 320 px wide image;
# callers working at another width scale them by width/320. Shared by calibration similarity and
# stadium-motion so both measure the same region the same way.
RIM_BLUR_SIGMA = 1.0
RIM_MASK_MARGIN = 3


def rim_mask(c: Calibration, size: tuple[int, int], margin: int) -> np.ndarray:
    """Stadium minus battle_area (rim + lid region) as a bool array of shape (height, width),
    at ``size`` = (width, height). The polygons are rasterised at ``c.frame_size`` and resized.
    The stadium is eroded and the battle_area dilated by ``margin`` px (at ``size``) so
    blur / gradient / alignment bleed from outside the region cannot enter it. Shared by
    calibration similarity and stadium-motion measurement so both look at the same region."""
    k = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)

    def small(poly: Polygon) -> np.ndarray:
        return cv2.resize(polygon_mask(poly, c.frame_size), size, interpolation=cv2.INTER_NEAREST)

    return (cv2.erode(small(c.stadium), k) > 0) & (cv2.dilate(small(c.battle_area), k) == 0)


def velocities(centers: dict[int, Point]) -> dict[int, tuple[float, float]]:
    """Per-frame velocity keyed by the later frame of each pair of neighbouring samples.
    If frames are missing between two samples the difference is divided by the frame gap."""
    frames = sorted(centers)
    out: dict[int, tuple[float, float]] = {}
    for prev, cur in zip(frames, frames[1:]):
        gap = cur - prev
        out[cur] = (
            (centers[cur][0] - centers[prev][0]) / gap,
            (centers[cur][1] - centers[prev][1]) / gap,
        )
    return out


def is_abrupt_change(
    v_before: tuple[float, float],
    v_after: tuple[float, float],
    angle_deg: float,
    speed_ratio: float,
) -> bool:
    """True if heading turns by >= ``angle_deg`` (both speeds non-zero) or speed changes by
    >= ``speed_ratio`` of the larger of the two speeds (so 0 -> moving or moving -> 0 counts)."""
    s0 = math.hypot(*v_before)
    s1 = math.hypot(*v_after)
    big = max(s0, s1)
    if big < _EPS:
        return False
    if abs(s1 - s0) / big >= speed_ratio:
        return True
    if s0 < _EPS or s1 < _EPS:
        return False
    cross = v_before[0] * v_after[1] - v_before[1] * v_after[0]
    dot = v_before[0] * v_after[0] + v_before[1] * v_after[1]
    return math.degrees(abs(math.atan2(cross, dot))) >= angle_deg
