"""Frame-preserving launch evidence in the caller's inference-pixel coordinates.

Tracks and calibration polygons must use the same coordinates. Launcher detections
represent the body only; ripcord/string tracks never provide touch evidence.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from bxjudge.config import Thresholds
from bxjudge.geometry import boxes_touch, is_abrupt_change, point_in_polygon, velocities
from bxjudge.landing import find_settles
from bxjudge.types import (
    AirState,
    Box,
    Calibration,
    LaunchCheck,
    Point,
    Side,
    TouchState,
    Track,
)

_SIDES: tuple[Side, Side] = ('X', 'B')


def _launch_side(p: Point, c: Calibration) -> Side | None:
    left, right = point_in_polygon(p, c.launch_left), point_in_polygon(p, c.launch_right)
    if left == right:
        return None
    return 'X' if left else 'B'


def assign_sides(bey_tracks: list[Track], c: Calibration) -> dict[int, Side]:
    """Assign each unique nonempty bey ID from its first observed center."""
    counts = Counter(t.track_id for t in bey_tracks)
    moments = cv2.moments(np.asarray(c.stadium, dtype=np.float32))
    if moments['m00'] == 0:
        raise ValueError('stadium must have positive area')
    mid_x = moments['m10'] / moments['m00']
    result = {}
    for t in bey_tracks:
        if (t.cls != 'bey' or not t.dets or counts[t.track_id] != 1
                or len({d.frame for d in t.dets}) != len(t.dets)):
            continue
        p = t.center_at(t.first_frame())
        result[t.track_id] = _launch_side(p, c) or ('X' if p[0] < mid_x else 'B')
    return result


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    u = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2)) if length2 else 0.0
    return math.hypot(p[0] - a[0] - u * dx, p[1] - a[1] - u * dy)


def _box_rim_distance(box: Box, c: Calibration) -> float:
    """Minimum Euclidean distance between a body rectangle and stadium boundary.

    A stadium segment intersecting the rectangle has distance zero. Otherwise
    the nearest pair includes a segment endpoint or a rectangle corner.
    """
    x0, y0, x1, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    distance = math.inf
    for a, b in zip(c.stadium, c.stadium[1:] + c.stadium[:1]):
        low, high = 0.0, 1.0
        for origin, delta, start, end in ((a[0], b[0] - a[0], x0, x1), (a[1], b[1] - a[1], y0, y1)):
            if delta == 0:
                if not start <= origin <= end:
                    high = -1.0
                    break
            else:
                u, v = (start - origin) / delta, (end - origin) / delta
                low, high = max(low, min(u, v)), min(high, max(u, v))
        if low <= high:
            return 0.0
        for p in (a, b):
            distance = min(distance, math.hypot(max(x0 - p[0], 0, p[0] - x1), max(y0 - p[1], 0, p[1] - y1)))
        distance = min(distance, *(_point_segment_distance(p, a, b) for p in corners))
    return distance


def _abrupt_frames(t: Track, th: Thresholds) -> set[int]:
    """A turn needs two observed consecutive displacements and meaningful speed."""
    centers = t.centers()
    vel = velocities(centers)
    return {
        f for f in centers
        if f - 1 in centers and f + 1 in centers
        and max(math.hypot(*vel[f]), math.hypot(*vel[f + 1])) >= th.abrupt_min_speed_px
        and is_abrupt_change(vel[f], vel[f + 1], th.abrupt_angle_deg, th.abrupt_speed_ratio)
    }


def impact_frames(bey_tracks: list[Track], landings: dict[int, int | None], c: Calibration, th: Thresholds) -> set[int]:
    """Observed landings, box contacts, and meaningful abrupt rim motion."""
    impacts = {f for f in landings.values() if f is not None}
    tracks = [t for t in bey_tracks if t.cls == 'bey']
    boxes = [{d.frame: d.box for d in t.dets} for t in tracks]
    for a, b in combinations(boxes, 2):
        impacts.update(f for f in a.keys() & b.keys() if boxes_touch(a[f], b[f]))
    contour = np.asarray(c.stadium, dtype=np.float32)
    for t in tracks:
        centers = t.centers()
        impacts.update(f for f in _abrupt_frames(t, th)
                       if abs(cv2.pointPolygonTest(contour, centers[f], True)) < th.rim_margin_px)
    return impacts


def detect_touch(disp: dict[int, float], launchers: list[Track], sides_of_launcher: dict[int, Side], impacts: set[int], window: tuple[int, int], c: Calibration, th: Thresholds) -> tuple[dict[Side, TouchState], list[str]]:
    """Attribute each above-threshold shake onset using observed launcher bodies.

    Positive evidence survives later uncertainty; uncertainty survives later no-touch
    evidence. Every frame in the half-open window [start, end) needs finite
    observed displacement to establish absence of touch.
    """
    touch: dict[Side, TouchState] = {'X': 'no', 'B': 'no'}
    notes = []

    def unknown():
        for side in _SIDES:
            if touch[side] != 'yes':
                touch[side] = 'unknown'

    observed = sorted(f for f in disp if window[0] <= f < window[1])
    if len(observed) < window[1] - window[0]:
        unknown()
        notes.append('stadium_motion_missing')
    for f in observed:
        if not math.isfinite(disp[f]):
            unknown()
            notes.append(f'stadium_motion_unknown@{f}')
            continue
        if disp[f] <= th.shake_px or (math.isfinite(disp.get(f - 1, math.nan)) and disp[f - 1] > th.shake_px):
            continue
        if any(abs(f - impact) <= th.impact_coincide_frames for impact in impacts):
            notes.append(f'shake_by_bey_impact@{f}')
            continue
        bodies = [t for t in launchers if t.cls == 'launcher' and any(d.frame == f and d.cls == 'launcher' for d in t.dets)]
        near = [t for t in bodies if _box_rim_distance(t.box_at(f), c) <= th.rim_margin_px]
        if not bodies:
            unknown()
            notes.append(f'shake_cause_unknown@{f}')
        elif not near:
            notes.append(f'shake_not_by_launcher_body@{f}')
        else:
            counts = Counter(t.track_id for t in bodies)
            if any(sides_of_launcher.get(t.track_id) not in _SIDES or counts[t.track_id] != 1 for t in near):
                unknown()
                notes.append(f'shake_cause_unknown@{f}')
            for t in near:
                side = sides_of_launcher.get(t.track_id)
                if side in _SIDES and counts[t.track_id] == 1:
                    touch[side] = 'yes'
    return touch, notes


def _air_coverage_notes(a: dict[int, Box], b: dict[int, Box], cutoff: float, th: Thresholds) -> list[str]:
    """Notes for each bey whose airborne observation coverage is too thin to prove "no".

    The airborne interval runs from the first observed frame of either bey to the
    earlier landing (or the last observed frame when no landing bounds it). Frames
    where a bey is not observed -- leading, trailing or internal -- are missing
    evidence; more than the configured fraction of the interval is incomplete.
    """
    seen = a.keys() | b.keys()
    if not seen:
        return []
    lo = min(seen)
    hi = cutoff if math.isfinite(cutoff) else max(seen) + 1
    span = hi - lo
    return [f'air_evidence_incomplete@{side}'
            for side, obs in zip(_SIDES, (a, b))
            if span - sum(1 for f in obs if lo <= f < hi) > th.air_evidence_max_missing_fraction * span]


def _air_collision_evidence(tx: Track, tb: Track, land_x: int | None, land_b: int | None, th: Thresholds) -> tuple[AirState, int | None, list[str]]:
    """Box contact plus both observed turns within the configured temporal radius.

    Returns ``(state, collision_frame, notes)``. An unobserved stretch of the airborne
    period is missing evidence: it can never produce a confident "no" (R31).
    """
    cutoff = min((f for f in (land_x, land_b) if f is not None), default=math.inf)
    a = {d.frame: d.box for d in tx.dets if d.frame < cutoff}
    b = {d.frame: d.box for d in tb.dets if d.frame < cutoff}
    incomplete = _air_coverage_notes(a, b, cutoff, th)
    common = a.keys() & b.keys()
    if not common:
        return 'maybe', None, incomplete
    # Restrict the turn's complete three-frame support to the airborne period.
    changes_a = {f for f in _abrupt_frames(tx, th) if f + 1 < cutoff}
    changes_b = {f for f in _abrupt_frames(tb, th) if f + 1 < cutoff}
    possible = False
    for f in sorted(common):
        overlap = boxes_touch(a[f], b[f])
        turns = (any(abs(f - g) <= th.air_collision_frames for g in changes_a)
                 and any(abs(f - g) <= th.air_collision_frames for g in changes_b))
        if overlap and turns:
            return 'yes', f, []
        possible |= overlap or turns
    missing = len(common) != max(common) - min(common) + 1
    return ('maybe' if possible or missing or incomplete else 'no'), None, incomplete

def detect_air_collision(tx: Track, tb: Track, land_x: int | None, land_b: int | None, th: Thresholds) -> AirState:
    return _air_collision_evidence(tx, tb, land_x, land_b, th)[0]


@dataclass
class LaunchAnalysis:
    check: LaunchCheck
    sides: dict[int, Side]
    window: tuple[int, int]
    impacts: set[int]
    air_collision_frame: int | None = None


def analyze_launch(bey_tracks: list[Track], launchers: list[Track], disp: dict[int, float], c: Calibration, th: Thresholds, fps: float) -> LaunchAnalysis:
    """Combine launch evidence without choosing between missing/duplicate side tracks.

    The missing-landing cap bounds the search for battle-area settling. Within that
    search, a battle-area episode takes precedence over an earlier lid/rim episode.
    Physical settle impacts and the first actual floor cutoff remain independent
    of that eligibility deadline. The touch window is half-open [start, end).
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    sides = assign_sides(bey_tracks, c)
    valid = [t for t in bey_tracks if t.track_id in sides]
    by_side = {side: [t for t in valid if sides[t.track_id] == side] for side in _SIDES}
    first = min((t.first_frame() for t in bey_tracks if t.cls == 'bey' and t.dets), default=0)
    start = max(0, first - round(th.launch_pre_s * fps))
    cap = first + round(th.launch_missing_landing_s * fps)
    all_episodes = {t.track_id: find_settles(t, c, th) for t in valid}
    eligible_episodes = {
        tid: [(f, z) for f, z in settles if f + th.settle_frames - 1 <= cap]
        for tid, settles in all_episodes.items()
    }
    physical_floor = {tid: next((f for f, z in settles if z == 'battle_area'), None)
                      for tid, settles in all_episodes.items()}
    landing_frame = dict.fromkeys(_SIDES)
    landing_zone = dict.fromkeys(_SIDES)
    landed = dict.fromkeys(_SIDES, False)
    notes = []
    for side, tracks in by_side.items():
        if len(tracks) != 1:
            notes.append(f'track_evidence_{"missing" if not tracks else "duplicate"}@{side}')
            continue
        settles = eligible_episodes[tracks[0].track_id]
        late = next((f for f, z in all_episodes[tracks[0].track_id]
                     if z == 'battle_area' and f + th.settle_frames - 1 > cap), None)
        if late is not None and not any(z == 'battle_area' for _, z in settles):
            notes.append(f'landing_timing_unknown@{side}:{late}')
        selected = next(((f, z) for f, z in settles if z == 'battle_area'), settles[0] if settles else (None, None))
        landing_frame[side], landing_zone[side] = selected
        landed[side] = selected[1] == 'battle_area'
    end = max(landing_frame.values()) + round(th.launch_post_s * fps) if all(landed.values()) else cap
    impacts = impact_frames(valid, physical_floor, c, th)
    # Every physical settle can cause a shake, including post-deadline episodes.
    impacts.update(f for settles in all_episodes.values() for f, _ in settles)
    launcher_counts = Counter(t.track_id for t in launchers)
    launcher_sides = {}
    for t in launchers:
        if t.cls == 'launcher' and t.dets and launcher_counts[t.track_id] == 1:
            side = _launch_side(t.center_at(t.first_frame()), c)
            if side is not None:
                launcher_sides[t.track_id] = side
    touch, touch_notes = detect_touch(disp, launchers, launcher_sides, impacts, (start, end), c, th)
    notes.extend(touch_notes)
    if all(len(tracks) == 1 for tracks in by_side.values()):
        air, air_frame, air_notes = _air_collision_evidence(
            by_side['X'][0], by_side['B'][0],
            physical_floor[by_side['X'][0].track_id],
            physical_floor[by_side['B'][0].track_id], th)
        notes.extend(air_notes)
    else:
        air, air_frame = 'maybe', None
        for side in _SIDES:
            if touch[side] != 'yes':
                touch[side] = 'unknown'
    return LaunchAnalysis(LaunchCheck(touch, landed, landing_frame, landing_zone, air, notes), sides, (start, end), impacts, air_frame)

def check_launch(bey_tracks: list[Track], launchers: list[Track], disp: dict[int, float], c: Calibration, th: Thresholds, fps: float) -> tuple[LaunchCheck, dict[int, Side]]:
    """Public launch verdict, sharing the exact analysis used by motion exports."""
    analysis = analyze_launch(bey_tracks, launchers, disp, c, th, fps)
    return analysis.check, analysis.sides
