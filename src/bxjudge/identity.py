"""Observed launcher-region identity anchors, independent of physical release.

Inputs must share the calibration's reference coordinates. No screen-half guess,
missing-frame interpolation, or human label enters association.
"""

import math
from dataclasses import dataclass

from .geometry import boxes_touch, point_in_polygon
from .launch_check import _launch_side
from .types import Calibration, Side, Track


@dataclass
class IdentityAnchors:
    sides: dict[int, Side]
    evidence: dict[int, list[dict]]

    def record(self):
        return {
            "method": "observed_prefix_launcher_region",
            "sides": {str(k): v for k, v in self.sides.items()},
            "evidence": {str(k): v for k, v in self.evidence.items()},
        }


def anchor_sides(
    tracks: list[Track], c: Calibration, *, required_frames: int, body_margin_px: float,
    source_frames: dict[int, list[int]],
) -> IdentityAnchors:
    if type(required_frames) is not int or required_frames < 2:
        raise ValueError("identity anchor needs at least two observed frames")
    if not math.isfinite(body_margin_px) or body_margin_px < 0:
        raise ValueError("identity body margin must be nonnegative finite")
    bodies = {}
    for track in tracks:
        if track.cls == "launcher":
            for d in track.dets:
                center = ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2)
                bodies.setdefault(d.frame, []).append(
                    (track.track_id, d.box, _launch_side(center, c))
                )
    sides, evidence = {}, {}
    for track in tracks:
        if track.cls != "bey" or not track.dets:
            continue
        first_unmapped = min(
            set(source_frames[track.track_id]) - {d.frame for d in track.dets},
            default=math.inf,
        )
        run = []
        prefix_side = None
        for d in sorted(track.dets, key=lambda d: d.frame):
            if d.frame >= first_unmapped:
                break  # Raw observation existed, but its geometry is unknown.
            point = track.center_at(d.frame)
            in_launch = point_in_polygon(point, c.launch_left) or point_in_polygon(
                point, c.launch_right
            )
            nearby = [
                (tid, side)
                for tid, box, side in bodies.get(d.frame, [])
                if boxes_touch(d.box, box, margin=body_margin_px)
            ]
            context = {side for _, side in nearby}
            # Do not search through unsupported observations for a later pickup.
            if not in_launch or len(context) != 1 or None in context:
                break
            side = next(iter(context))
            top_region = _launch_side(point, c)
            if (top_region is not None and top_region != side) or (
                prefix_side is not None and side != prefix_side
            ):
                break
            prefix_side = side
            if run and d.frame != run[-1]["frame"] + 1:
                run = []
            run.append({"frame": d.frame, "launcher_ids": sorted({tid for tid, _ in nearby})})
            if len(run) == required_frames:
                sides[track.track_id] = side
                evidence[track.track_id] = run
                break
    return IdentityAnchors(sides, evidence)


def inherited_sides(hints: dict[int, Side], members: dict[int, list[int]]) -> dict[int, Side]:
    """Only verified component membership carries an anchor into a canonical ID."""
    result = {}
    for root, ids in members.items():
        known = {hints[tid] for tid in ids if tid in hints}
        if len(known) == 1:
            side = next(iter(known))
            if side in ("X", "B"):
                result[root] = side
    return result
