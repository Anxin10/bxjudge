"""Observed spin proxies and conservative match-finish evidence.

Orbit direction is a configured proxy, not measured axial spin. Slow optical-flow
reversals retain the requested classification, but Task13 must require review
until known-direction real-video acceptance (R33). Crop keys are original frames;
gaps and nonfinite measurements never count as consecutive motion evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from bxjudge.config import Thresholds
from bxjudge.geometry import boxes_touch, point_in_polygon, zone_of
from bxjudge.types import Box, Calibration, Detection, Finish, Polygon, Side, SpinDir, Track

# Farneback solver parameters, not rule/decision thresholds. Inputs are 64x64
# gray crops per the interface; no rule compares motion to these solver values.
_FLOW_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                    poly_n=5, poly_sigma=1.2, flags=0)
_SIDES: tuple[Side, Side] = ('X', 'B')


@dataclass
class FinishResult:
    finish: Finish | None
    loser: Side | None
    frame: int | None
    confidence: float
    reasons: list[str]


def _gray(crop: np.ndarray) -> np.ndarray:
    if crop.shape != (64, 64) or crop.ndim != 2:
        raise ValueError('spin crops must be 64x64 grayscale images')
    return crop.astype(np.float32)


def spin_activity(crops: dict[int, np.ndarray]) -> dict[int, float]:
    """Mean absolute difference to the previous observed consecutive frame.

    The first crop, gaps, and nonfinite image pairs have unknown (NaN) activity.
    """
    gray = {f: _gray(crop) for f, crop in sorted(crops.items())}
    return {f: float(np.mean(cv2.absdiff(gray[f - 1], cur)))
            if f - 1 in gray and np.isfinite(cur).all() and np.isfinite(gray[f - 1]).all()
            else math.nan for f, cur in gray.items()}


def rotation_rate(crops: dict[int, np.ndarray]) -> dict[int, float]:
    """Signed apparent screen degrees/frame; clockwise is positive.

    Fit the tangential Farneback field v=omega*(-y,x) in the crop's inscribed
    disk. This is the radius-squared weighted mean angular speed, avoids dividing
    at the center, and excludes square corners. It does not disambiguate aliasing.
    """
    gray = {f: _gray(crop) for f, crop in sorted(crops.items())}
    out = dict.fromkeys(gray, math.nan)
    y, x = np.indices((64, 64), dtype=np.float32)
    x -= (64 - 1) / 2
    y -= (64 - 1) / 2
    radius2 = x * x + y * y
    disk = radius2 <= ((64 - 1) / 2) ** 2
    denominator = float(np.sum(radius2[disk]))
    for f, cur in gray.items():
        if f - 1 not in gray or not np.isfinite(cur).all() or not np.isfinite(gray[f - 1]).all():
            continue
        flow = cv2.calcOpticalFlowFarneback(gray[f - 1], cur, None, **_FLOW_PARAMS)
        if np.isfinite(flow).all():
            out[f] = math.degrees(float(np.sum((x * flow[..., 1] - y * flow[..., 0])[disk])) / denominator)
    return out


def _positive_fps(fps: float) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')


def orbit_direction(t: Track, c: Calibration, landing_frame: int | None,
                    th: Thresholds, fps: float) -> SpinDir:
    """Configured orbital proxy over [landing, landing + window), never axial spin.

    Missing, out-of-floor, center-degenerate, or inconsistent angular observations
    cannot establish a direction. Zero angular steps do not change the sign.
    """
    _positive_fps(fps)
    if landing_frame is None:
        return 'unknown'
    centers = t.centers()
    end = landing_frame + round(th.spin_dir_window_s * fps)
    frames = sorted(f for f in centers if landing_frame <= f < end)
    if len(frames) < 2 or frames[0] != landing_frame:
        return 'unknown'
    moments = cv2.moments(np.asarray(c.battle_area, dtype=np.float32))
    if moments['m00'] == 0:
        return 'unknown'
    cx, cy = moments['m10'] / moments['m00'], moments['m01'] / moments['m00']
    relative = []
    for f in frames:
        x, y = centers[f]
        if not all(map(math.isfinite, (x, y))) or zone_of((x, y), c) != 'battle_area' or (x, y) == (cx, cy):
            return 'unknown'
        relative.append((x - cx, y - cy))
    turns = []
    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1] + 1:
            return 'unknown'
        ax, ay = relative[i - 1]
        bx, by = relative[i]
        turns.append(math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by)))
    if any(v > 0 for v in turns) and any(v < 0 for v in turns):
        return 'unknown'
    total = sum(turns)
    if abs(total) < th.spin_dir_min_turn_deg or total == 0:
        return 'unknown'
    return th.orbit_to_spin['cw' if total > 0 else 'ccw']


def stop_frame(t: Track, activity: dict[int, float], rate: dict[int, float],
               spin_dir: SpinDir, th: Thresholds) -> tuple[int | None, str | None]:
    """First fully observed still/slow-reversal run, by run start (R33 applies)."""
    centers = t.centers()
    baseline = next((1 if orbit == 'cw' else -1
                     for orbit, spin in th.orbit_to_spin.items() if spin == spin_dir), None)
    still_run = reverse_run = learn_run = 0
    learn_sign = None
    candidates = []
    for f in sorted(centers):
        moving = math.inf
        if f - 1 in centers:
            moving = math.dist(centers[f - 1], centers[f])
        if not math.isfinite(moving):
            moving = math.inf
        a, r = activity.get(f, math.nan), rate.get(f, math.nan)
        still = moving < th.still_px and math.isfinite(a) and a < th.still_activity
        still_run = still_run + 1 if still else 0
        if still_run == th.still_frames:
            candidates.append((f - still_run + 1, 'still'))
        slow = (moving < th.reverse_max_move_px and math.isfinite(r)
                and 0 < abs(r) < th.reverse_max_rate_deg)
        sign = 1 if r > 0 else -1
        if baseline is None:
            learn_run = learn_run + 1 if slow and sign == learn_sign else int(slow)
            learn_sign = sign if slow else None
            if learn_run >= th.reverse_frames:
                baseline = sign
        reversed_now = slow and baseline is not None and sign != baseline
        reverse_run = reverse_run + 1 if reversed_now else 0
        if reverse_run == th.reverse_frames:
            candidates.append((f - reverse_run + 1, 'reversed'))
    return min(candidates, default=(None, None), key=lambda item: item[0])


def id_swap_risk(tx: Track, tb: Track, th: Thresholds) -> bool:
    """Only an observed consecutive run of box contacts establishes swap risk."""
    a, b = ({d.frame: d.box for d in t.dets} for t in (tx, tb))
    previous = None
    run = 0
    for f in sorted(a.keys() & b.keys()):
        if boxes_touch(a[f], b[f]):
            run = run + 1 if previous == f - 1 else 1
        else:
            run = 0
        if run >= th.id_swap_overlap_frames:
            return True
        previous = f
    return False


def _segment_in_box_interior(a, b, box: Box) -> bool:
    low, high = 0.0, 1.0
    for origin, delta, start, end in (
            (a[0], b[0] - a[0], box[0], box[2]),
            (a[1], b[1] - a[1], box[1], box[3])):
        if delta == 0:
            if not start < origin < end:
                return False
        else:
            u, v = (start - origin) / delta, (end - origin) / delta
            low, high = max(low, min(u, v)), min(high, max(u, v))
    return low < high


def _whole_box_inside(box: Box, polygon: Polygon) -> bool:
    x0, y0, x1, y1 = box
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    # Corners alone are insufficient for concave pockets: a notch can cross the box.
    return (all(point_in_polygon(p, polygon) for p in corners)
            and not any(_segment_in_box_interior(a, b, box)
                        for a, b in zip(polygon, polygon[1:] + polygon[:1])))


def _score(dets: list[Detection]) -> float:
    scores = [d.score for d in dets]
    return float(np.mean(scores)) if scores and all(math.isfinite(s) for s in scores) else 0.0


def detect_finish(bey: dict[Side, Track], parts: list[Track],
                  stops: dict[Side, int | None], c: Calibration,
                  th: Thresholds, fps: float) -> FinishResult:
    """Earliest evidenced finish; stop reasons are propagated by Task13 (R21).

    Whole-box pocket entry must remain the final observed position. A return
    cancels the earlier exit conservatively, since this API has no crop rates.
    Unexplained early track loss cannot establish a finish or be hidden by a
    later event. Stops require both sides observed in the battle area.
    """
    _positive_fps(fps)
    reasons = []
    observations = {}
    for side in _SIDES:
        t = bey.get(side)
        if t is None or not t.dets:
            reasons.append(f'track_missing@{side}')
            continue
        if (t.cls != 'bey' or len({d.frame for d in t.dets}) != len(t.dets)
                or any(d.cls != 'bey' or not all(map(math.isfinite, d.box)) for d in t.dets)):
            reasons.append(f'track_evidence_invalid@{side}')
            continue
        observations[side] = {d.frame: d for d in sorted(t.dets, key=lambda d: d.frame)}
    if len(observations) != len(_SIDES):
        return FinishResult(None, None, None, 0.0, reasons)

    centers = {side: t.centers() for side, t in bey.items()}
    pockets = [('xtreme', c.xtreme), ('over', c.over_left), ('over', c.over_right)]
    events: list[FinishResult] = []
    unresolved = []
    for side, dets in observations.items():
        frames = list(dets)
        if any(b != a + 1 for a, b in zip(frames, frames[1:])):
            reasons.append(f'track_evidence_gap@{side}')
        whole = {f: next((i for i, (_, poly) in enumerate(pockets)
                          if _whole_box_inside(d.box, poly)), None) for f, d in dets.items()}
        entered = [f for f in frames if whole[f] is not None]
        if entered and any(f > entered[0] and zone_of(centers[side][f], c) == 'battle_area' for f in frames):
            reasons.append(f'exit_returned@{side}')
        pocket = whole[frames[-1]]
        if pocket is not None:
            start = frames[-1]
            for f in reversed(frames[:-1]):
                if f != start - 1 or whole[f] != pocket:
                    break
                start = f
            events.append(FinishResult(pockets[pocket][0], side, start,
                                       _score([dets[f] for f in frames if f >= start]), []))
        # Chronological uncertainty is independent of the final whole-pocket
        # event. A later pocket observation does not prove when an earlier
        # partial/offscreen exit occurred; only an observed whole-body return
        # to the battle floor explicitly cancels that potential earlier exit.
        ambiguous = {}
        for f in frames:
            if _whole_box_inside(dets[f].box, c.battle_area):
                ambiguous.clear()
            elif whole[f] is None:
                zone = zone_of(centers[side][f], c)
                if zone in ('xtreme', 'over_left', 'over_right', 'outside'):
                    reason = 'position_unknown' if zone == 'outside' else 'pocket_entry_unconfirmed'
                    ambiguous.setdefault(f'{reason}@{side}', f)
        reasons.extend(ambiguous)
        unresolved.extend(ambiguous.values())

    for part in parts:
        if part.cls != 'bey_part' or not part.dets:
            continue
        f = part.first_frame()
        pd = next(d for d in part.dets if d.frame == f)
        if pd.cls != 'bey_part':
            continue
        near = []
        for side, dets in observations.items():
            last = max(dets)
            if (last < f <= last + round(th.burst_part_window_s * fps)
                    and math.dist(centers[side][last], part.center_at(f)) <= th.burst_part_radius_px):
                near.append(side)
        if len(near) == 1:
            side = near[0]
            events.append(FinishResult('burst', side, f,
                                       _score([observations[side][max(observations[side])], pd]), []))
        elif near:
            reasons.append('burst_side_unknown')
            unresolved.append(f)

    for side in _SIDES:
        f = stops.get(side)
        if f is None:
            continue
        if all(f in ds and zone_of(centers[s][f], c) == 'battle_area'
               for s, ds in observations.items()):
            events.append(FinishResult('spin', side, f,
                                       _score([ds[f] for ds in observations.values()]), []))
        else:
            reasons.append(f'stop_evidence_missing@{side}')
            unresolved.append(f)

    latest = max(max(ds) for ds in observations.values())
    for side, dets in observations.items():
        last = max(dets)
        if last < latest and not any(e.loser == side and e.frame <= last + round(th.burst_part_window_s * fps)
                                      for e in events):
            reasons.append(f'track_lost@{side}')
            unresolved.append(last)

    events.sort(key=lambda e: e.frame)
    if not events or (unresolved and min(unresolved) <= events[0].frame):
        result = FinishResult(None, None, None, 0.0, reasons + ['finish_unknown'])
    else:
        first = events[0]
        simultaneous = [e for e in events if e.loser != first.loser
                        and e.frame - first.frame <= th.simultaneous_stop_frames]
        if simultaneous:
            result = FinishResult(None, None, first.frame,
                                  float(np.mean([first.confidence, simultaneous[0].confidence])), reasons + ['draw'])
        else:
            result = FinishResult(first.finish, first.loser, first.frame, first.confidence, reasons)
    if any(r.startswith('track_evidence_gap@') for r in reasons):
        result.confidence *= th.track_gap_confidence_factor
    if id_swap_risk(bey['X'], bey['B'], th):
        result.confidence *= th.id_swap_confidence_factor
        result.reasons.append('id_swap_risk')
    return result
