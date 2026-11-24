"""Observed spin proxies and conservative match-finish evidence.

Orbit direction is a configured proxy, not measured axial spin. Slow optical-flow
reversals retain the requested classification, but Task13 must require review
until known-direction real-video acceptance (R33). Crop keys are original frames;
gaps and nonfinite measurements never count as consecutive motion evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from bxjudge.config import Thresholds
from bxjudge.geometry import boxes_touch, point_in_polygon, zone_of
from bxjudge.types import (
    Box,
    Calibration,
    Detection,
    Finish,
    Polygon,
    Side,
    SpinDir,
    Track,
)

DEFAULT_FINISH_FUSION_CONFIG = Path(__file__).resolve().parents[2] / "config/finish_fusion.yaml"


def load_finish_fusion_config(path: Path | str | None = DEFAULT_FINISH_FUSION_CONFIG) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path).expanduser()
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw if isinstance(raw, dict) else {}

# Farneback solver parameters, not rule/decision thresholds. Inputs are 64x64
# gray crops per the interface; no rule compares motion to these solver values.
_FLOW_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                    poly_n=5, poly_sigma=1.2, flags=0)
_SIDES: tuple[Side, Side] = ('X', 'B')
# R22/R55: reasons with this prefix are informational notes about evidence that
# arose only after the result was settled; the pipeline does not gate on them.
AFTER_FINISH_PREFIX = 'after_finish:'


@dataclass
class FinishResult:
    finish: Finish | None
    loser: Side | None
    frame: int | None
    confidence: float
    reasons: list[str]


@dataclass
class _Note:
    frame: int | None        # None: not tied to a time, always review-relevant
    reason: str
    side: Side | None
    blocks: bool             # a possible earlier finish at an unknown time
    owner: Side | None = None  # transit into this side's own pocket entry (R55a)
    deferrable: bool = True  # False: contradicts a pocket finish, never a note (R58)


def _settled_frame(result: FinishResult, th: Thresholds) -> int | None:
    """Last frame at which the non-losing side's evidence can change a result.

    A competing stop starting inside the draw window is only established
    max(still_frames, reverse_frames) frames later (R55b).
    """
    if result.finish is None or result.loser is None or result.frame is None:
        return None
    return (result.frame + th.simultaneous_stop_frames
            + max(th.still_frames, th.reverse_frames) - 1)


def is_after_finish(result: FinishResult, side: Side | None, frame: int,
                    th: Thresholds) -> bool:
    """Whether evidence of ``side`` at ``frame`` arose only after a known result.

    The loser's later observations cannot change who lost; other evidence
    matters until a competing draw-window stop would be established (R55b).
    """
    settled = _settled_frame(result, th)
    if settled is None:
        return False
    return frame > (result.frame if side == result.loser else settled)


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


def laplacian_variance(crops: dict[int, np.ndarray]) -> dict[int, float]:
    """Variance of the Laplacian for 64x64 grayscale crops; higher variance indicates sharper texture."""
    out: dict[int, float] = {}
    for f, crop in sorted(crops.items()):
        if crop.shape == (64, 64) and np.isfinite(crop).all():
            out[f] = float(cv2.Laplacian(crop.astype(np.float32), cv2.CV_32F).var())
        else:
            out[f] = math.nan
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


def stop_frame(
    t: Track,
    activity: dict[int, float],
    rate: dict[int, float],
    spin_dir: SpinDir,
    th: Thresholds,
    *,
    laplacian: dict[int, float] | None = None,
    fusion: dict[str, Any] | None = None,
    landing_frame: int | None = None,
) -> tuple[int | None, str | None]:
    """First fully observed still/slow-reversal run, by run start (R33 applies)."""
    centers = t.centers()
    baseline = next((1 if orbit == 'cw' else -1
                     for orbit, spin in th.orbit_to_spin.items() if spin == spin_dir), None)
    fusion_enabled = bool(fusion and fusion.get("enabled", False))
    min_frame = landing_frame if (fusion_enabled and landing_frame is not None) else 0

    still_run = reverse_run = learn_run = 0
    learn_sign = None
    candidates: list[tuple[int, str]] = []

    req_still_frames = fusion.get("still_frames", th.still_frames) if fusion_enabled else th.still_frames
    still_moving_px = fusion.get("still_moving_px", 2.5) if fusion_enabled else th.still_px
    still_max_rot = fusion.get("still_max_rot_rate_deg", 0.3) if fusion_enabled else 0.5
    still_min_lap = fusion.get("still_min_laplacian_var", 100.0) if fusion_enabled else 100.0
    rev_min_lap = fusion.get("reverse_min_laplacian_var", 180.0) if fusion_enabled else 180.0
    guard_disp = fusion.get("terminal_motion_guard_px", 20.0) if fusion_enabled else None

    sorted_frames = sorted(centers)
    frame_indices = {f: i for i, f in enumerate(sorted_frames)}

    for f in sorted_frames:
        moving = math.inf
        if f - 1 in centers:
            moving = math.dist(centers[f - 1], centers[f])
        if not math.isfinite(moving):
            moving = math.inf
        a, r = activity.get(f, math.nan), rate.get(f, math.nan)
        lap = laplacian.get(f, math.nan) if laplacian is not None else math.nan

        still = moving < th.still_px and math.isfinite(a) and a < th.still_activity
        if fusion_enabled:
            fusion_still = (
                moving < still_moving_px
                and math.isfinite(r) and abs(r) < still_max_rot
                and math.isfinite(lap) and lap >= still_min_lap
            )
            still = still or fusion_still

        still_run = still_run + 1 if still else 0
        if still_run == req_still_frames:
            cand_frame = f - still_run + 1
            if cand_frame >= min_frame:
                candidates.append((cand_frame, 'still'))

        slow = (moving < th.reverse_max_move_px and math.isfinite(r)
                and 0 < abs(r) < th.reverse_max_rate_deg)
        sign = 1 if r > 0 else -1
        if baseline is None:
            learn_run = learn_run + 1 if slow and sign == learn_sign else int(slow)
            learn_sign = sign if slow else None
            if learn_run >= th.reverse_frames:
                baseline = sign
        reversed_now = slow and baseline is not None and sign != baseline

        if fusion_enabled and reversed_now and not (math.isfinite(lap) and lap >= rev_min_lap):
            reversed_now = False

        reverse_run = reverse_run + 1 if reversed_now else 0
        if reverse_run == th.reverse_frames:
            cand_frame = f - reverse_run + 1
            if cand_frame >= min_frame:
                candidates.append((cand_frame, 'reversed'))

    if fusion_enabled and guard_disp is not None and candidates:
        valid_candidates = []
        for cand_frame, kind in candidates:
            idx = frame_indices[cand_frame]
            pt = centers[cand_frame]
            active = [centers[sorted_frames[k]] for k in range(idx, max(idx + 1, len(sorted_frames) - 30))]
            if not any(math.dist(pt, p2) > guard_disp for p2 in active):
                valid_candidates.append((cand_frame, kind))
        candidates = valid_candidates

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


def _single_corner_overhang(box, polygon, fraction: float) -> bool:
    """The whole box, except for one small corner rectangle, must be in the pocket."""
    x0, y0, x1, y1 = box
    if not (x0 < x1 and y0 < y1 and fraction > 0):
        return False
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    outside = [
        (x, y) for x in (x0, x1) for y in (y0, y1)
        if cv2.pointPolygonTest(contour, (float(x), float(y)), False) < 0
    ]
    if len(outside) != 1:
        return False
    [(x, y)] = outside
    dx, dy = (x1 - x0) * fraction, (y1 - y0) * fraction
    across_x = (x0 + dx, y0, x1, y1) if x == x0 else (x0, y0, x1 - dx, y1)
    across_y = (x0, y0 + dy, x1, y1) if y == y0 else (x0, y0, x1, y1 - dy)
    return _whole_box_inside(across_x, polygon) and _whole_box_inside(across_y, polygon)


def _join_corner_jitter(runs, dets, centers, pockets, calibration, th, confirm):
    """Join observed short holes while retaining the original whole-box count."""
    result = []
    jitter_entries = set()
    index = 0
    while index < len(runs):
        start, end, pocket = runs[index]
        zone, polygon = pockets[pocket]
        strict = end - start + 1
        last = index
        while last + 1 < len(runs):
            next_start, next_end, next_pocket = runs[last + 1]
            gap = next_start - end - 1
            if next_pocket != pocket or not 0 < gap <= th.pocket_corner_jitter_frames:
                break
            if not all(
                f in dets and zone_of(centers[f], calibration) == zone
                and _single_corner_overhang(dets[f].box, polygon, th.pocket_corner_jitter_frac)
                for f in range(end + 1, next_start)
            ):
                break
            strict += next_end - next_start + 1
            end = next_end
            last += 1
        if last > index and strict >= confirm and strict * 2 > end - start + 1:
            result.append([start, end, pocket])
            jitter_entries.add(start)
        else:
            result.extend(runs[index:last + 1])
        index = last + 1
    return result, jitter_entries


def detect_finish(bey: dict[Side, Track], parts: list[Track],
                  stops: dict[Side, int | None], c: Calibration,
                  th: Thresholds, fps: float) -> FinishResult:
    """Earliest evidenced finish; stop reasons are propagated by Task13 (R21).

    Evidence is judged up to the decisive event (R55). A whole-box pocket run
    is an entry when it is the final observation, or supplies the configured
    whole-box sample count without a return to the battle floor. Short observed
    corner-overhang holes can join runs but do not count as whole samples. This API has no crop
    rates, so any floor return counts as a possibly rotating return and
    cancels the run (R25/R58), unless the top's stop was observed inside the
    pocket (or immediately on the return); ``exit_returned`` always remains a
    review trigger. Persistence alone only decides disappearance. Partial or
    offscreen frames leading into that entry belong to it: they bound its
    earliest time against the other side instead of blocking it. Unexplained
    earlier track loss or position uncertainty cannot be hidden by a later
    event. Stops require both sides observed in the battle area. Uncertainty
    arising only after the result is settled is reported with
    ``AFTER_FINISH_PREFIX`` as an informational note.
    """
    _positive_fps(fps)
    notes: list[_Note] = []
    observations = {}
    for side in _SIDES:
        t = bey.get(side)
        if t is None or not t.dets:
            notes.append(_Note(None, f'track_missing@{side}', side, False))
            continue
        if (t.cls != 'bey' or len({d.frame for d in t.dets}) != len(t.dets)
                or any(d.cls != 'bey' or not all(map(math.isfinite, d.box)) for d in t.dets)):
            notes.append(_Note(None, f'track_evidence_invalid@{side}', side, False))
            continue
        observations[side] = {d.frame: d for d in sorted(t.dets, key=lambda d: d.frame)}
    if len(observations) != len(_SIDES):
        return FinishResult(None, None, None, 0.0, [n.reason for n in notes])

    centers = {side: t.centers() for side, t in bey.items()}
    pockets = [('xtreme', c.xtreme), ('over_left', c.over_left), ('over_right', c.over_right)]
    confirm = round(th.pocket_confirm_s * fps)
    window = round(th.burst_part_window_s * fps)
    events: list[tuple[FinishResult, int]] = []  # (event, earliest possible frame)
    for side, dets in observations.items():
        frames = list(dets)
        gaps = [a + 1 for a, b in zip(frames, frames[1:]) if b != a + 1]
        if gaps:
            notes.append(_Note(gaps[0], f'track_evidence_gap@{side}', side, False))
        whole = {f: next((i for i, (_, poly) in enumerate(pockets)
                          if _whole_box_inside(d.box, poly)), None) for f, d in dets.items()}
        runs: list[list[int]] = []  # [start, end, pocket] of consecutive whole-pocket frames
        for f in frames:
            if whole[f] is None:
                continue
            if runs and runs[-1][1] == f - 1 and runs[-1][2] == whole[f]:
                runs[-1][1] = f
            else:
                runs.append([f, f, whole[f]])
        runs, jitter_entries = _join_corner_jitter(runs, dets, centers[side], pockets, c, th, confirm)
        stop = stops.get(side)

        def is_entry(i: int, frames=frames, runs=runs, stop=stop, side=side) -> bool:
            start, end, _ = runs[i]
            if end == frames[-1]:
                return True
            if end - start + 1 < confirm:
                return False
            # R58: the first battle-floor centre before the next pocket run is
            # a possibly rotating return unless the top was seen stopped first.
            following = runs[i + 1][0] if i + 1 < len(runs) else math.inf
            back = next((f for f in frames if end < f < following
                         and zone_of(centers[side][f], c) == 'battle_area'), None)
            return back is None or (stop is not None and start <= stop <= back + 1)

        entry = next((r for i, r in enumerate(runs) if is_entry(i)), None)
        # Chronological uncertainty since the last whole-body battle-floor frame.
        # A later pocket observation does not prove when an earlier partial,
        # offscreen or transient exit occurred, unless that run of frames leads
        # straight into this side's own confirmed entry (R55a).
        pending: dict[str, int] = {}
        entered = False
        earliest = None
        for f in frames:
            if entry is not None and f == entry[0]:
                notes.extend(_Note(g, r, side, True, owner=side) for r, g in pending.items())
                earliest = min(pending.values(), default=f)
                pending.clear()
            if whole[f] is not None:
                entered = True
            elif entered and zone_of(centers[side][f], c) == 'battle_area':
                notes.append(_Note(f, f'exit_returned@{side}', side, False, deferrable=False))
                entered = False
            if entry is not None and entry[0] <= f <= entry[1]:
                continue
            if _whole_box_inside(dets[f].box, c.battle_area):
                pending.clear()
            elif whole[f] is not None:
                pending.setdefault(f'pocket_entry_unconfirmed@{side}', f)
            else:
                zone = zone_of(centers[side][f], c)
                if zone in ('xtreme', 'over_left', 'over_right', 'outside'):
                    reason = 'position_unknown' if zone == 'outside' else 'pocket_entry_unconfirmed'
                    pending.setdefault(f'{reason}@{side}', f)
        notes.extend(_Note(g, r, side, True) for r, g in pending.items())
        if entry is not None:
            start, end, pocket = entry
            finish = "xtreme" if pockets[pocket][0] == "xtreme" else "over"
            review = [f"pocket_corner_jitter@{side}"] if start in jitter_entries else []
            events.append((FinishResult(finish, side, start,
                                        _score([dets[f] for f in frames if start <= f <= end]), review),
                           earliest))

    overlap = round(th.burst_part_overlap_s * fps)
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
            # A split may co-exist with the top's final boxes (spec 6.5, R55d).
            if (last - overlap <= f <= last + window
                    and math.dist(centers[side][last], part.center_at(f)) <= th.burst_part_radius_px):
                near.append(side)
        if len(near) == 1:
            side = near[0]
            events.append((FinishResult('burst', side, f,
                                        _score([observations[side][max(observations[side])], pd]), []),
                           f))
        elif near:
            notes.append(_Note(f, 'burst_side_unknown', None, True))

    for side in _SIDES:
        f = stops.get(side)
        if f is None:
            continue
        matched_f = next(
            (
                cand_f
                for cand_f in range(f, f + 4)
                if all(
                    cand_f in ds and zone_of(centers[s][cand_f], c) == "battle_area"
                    for s, ds in observations.items()
                )
            ),
            None,
        )
        if matched_f is not None:
            events.append(
                (
                    FinishResult(
                        "spin",
                        side,
                        matched_f,
                        _score([ds[matched_f] for ds in observations.values()]),
                        [],
                    ),
                    matched_f,
                )
            )
        else:
            notes.append(_Note(f, f"stop_evidence_missing@{side}", side, True))

    latest = max(max(ds) for ds in observations.values())
    for side, dets in observations.items():
        last = max(dets)
        if last < latest and not any(e.loser == side and e.frame <= last + window for e, _ in events):
            notes.append(_Note(last, f'track_lost@{side}', side, True))

    events.sort(key=lambda item: item[0].frame)
    first = events[0][0] if events else None
    if first is None or any(n.blocks and n.frame is not None and n.frame <= first.frame
                            and n.owner != first.loser for n in notes):
        result = FinishResult(None, None, None, 0.0, [n.reason for n in notes] + ['finish_unknown'])
    else:
        simultaneous = [e for e, earliest in events if e.loser != first.loser
                        and earliest - first.frame <= th.simultaneous_stop_frames]
        if simultaneous:
            result = FinishResult(None, None, first.frame,
                                  float(np.mean([first.confidence, simultaneous[0].confidence])),
                                  [n.reason for n in notes] + ['draw'])
        else:
            result = FinishResult(first.finish, first.loser, first.frame, first.confidence, [])
            result.reasons = [
                *first.reasons,
                *(
                    AFTER_FINISH_PREFIX + n.reason
                    if n.deferrable and n.frame is not None
                    and is_after_finish(result, n.side, n.frame, th) else n.reason
                    for n in notes if n.owner != first.loser
                ),
            ]
    result.reasons = list(dict.fromkeys(result.reasons))
    if any(r.startswith('track_evidence_gap@') for r in result.reasons):
        result.confidence *= th.track_gap_confidence_factor
    settled = _settled_frame(result, th)
    early = {side: replace(bey[side], dets=[d for d in bey[side].dets
                                           if settled is None or d.frame <= settled])
             for side in _SIDES}
    if id_swap_risk(early['X'], early['B'], th):
        result.confidence *= th.id_swap_confidence_factor
        result.reasons.append('id_swap_risk')
    elif id_swap_risk(bey['X'], bey['B'], th):
        result.reasons.append(AFTER_FINISH_PREFIX + 'id_swap_risk')
    return result
