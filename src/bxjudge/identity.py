"""Observed launcher ownership, independent of physical release or missing geometry.

Ownership uses source boxes. Valid registrations and separated observed launcher
pairs supply side evidence; missing registration never supplies a transform.
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
            "method": "observed_prefix_launcher_context",
            "coordinate_space": "source_frame_pixels",
            "physical_release_inferred": False,
            "sides": {str(k): v for k, v in self.sides.items()},
            "evidence": {str(k): v for k, v in self.evidence.items()},
        }


def anchor_sides(
    tracks: list[Track],
    reference_tracks: list[Track],
    c: Calibration,
    *,
    required_frames: int,
    body_margin_px: float,
) -> IdentityAnchors:
    if type(required_frames) is not int or required_frames < 2:
        raise ValueError("identity anchor needs at least two observed frames")
    if not math.isfinite(body_margin_px) or body_margin_px < 0:
        raise ValueError("identity body margin must be nonnegative finite")
    reference = {
        (t.track_id, d.frame): (d.box, t.center_at(d.frame))
        for t in reference_tracks
        for d in t.dets
    }
    bodies = {}
    for track in tracks:
        if track.cls == "launcher":
            for d in track.dets:
                bodies.setdefault(d.frame, []).append((track.track_id, d.box))
    sides, evidence = {}, {}
    for track in tracks:
        if track.cls != "bey" or not track.dets:
            continue
        run = []
        prefix_side = None
        for d in sorted(track.dets, key=lambda d: d.frame):
            observed = bodies.get(d.frame, [])
            measured = {
                tid: _launch_side(reference[tid, d.frame][1], c)
                for tid, _ in observed
                if (tid, d.frame) in reference
            }
            paired = {}
            if len(observed) == 2:
                left, right = sorted(
                    observed, key=lambda item: (item[1][0] + item[1][2]) / 2
                )
                # Overlapping duplicate body boxes do not establish two players.
                if left[1][2] <= right[1][0]:
                    paired = {left[0]: "X", right[0]: "B"}
            conflict = any(
                side is not None and tid in paired and paired[tid] != side
                for tid, side in measured.items()
            )
            roles, cues = {}, {}
            for tid, _ in observed:
                role_evidence = {
                    kind: side
                    for kind, side in (
                        ("registered_launcher_region", measured.get(tid)),
                        ("separated_launcher_pair", paired.get(tid)),
                    )
                    if side is not None
                }
                known = set(role_evidence.values())
                roles[tid] = (
                    next(iter(known)) if len(known) == 1 and not conflict else None
                )
                cues[tid] = role_evidence
            nearby = [
                (tid, roles[tid])
                for tid, box in observed
                if boxes_touch(d.box, box, margin=body_margin_px)
            ]
            if not nearby:
                # An observed free prefix cannot become a later pickup anchor.
                break
            registered = reference.get((track.track_id, d.frame))
            point = registered[1] if registered is not None else None
            if point is not None and not (
                point_in_polygon(point, c.launch_left)
                or point_in_polygon(point, c.launch_right)
            ):
                break
            context = {side for _, side in nearby}
            if None in context:
                run = []
                continue
            if len(context) != 1:
                break
            side = next(iter(context))
            top_region = _launch_side(point, c) if point is not None else None
            if (top_region is not None and top_region != side) or (
                prefix_side is not None and side != prefix_side
            ):
                break
            prefix_side = side
            if run and d.frame != run[-1]["frame"] + 1:
                run = []
            run.append(
                {
                    "frame": d.frame,
                    "launcher_ids": sorted({tid for tid, _ in nearby}),
                    "source_bey_box": list(d.box),
                    "source_launcher_boxes": {
                        str(tid): list(box) for tid, box in observed
                    },
                    "side_evidence": {str(tid): cues[tid] for tid, _ in nearby},
                }
            )
            if len(run) == required_frames:
                sides[track.track_id] = side
                evidence[track.track_id] = run
                break
    return IdentityAnchors(sides, evidence)


def inherited_sides(
    hints: dict[int, Side], members: dict[int, list[int]]
) -> dict[int, Side]:
    """Only verified component membership carries an anchor into a canonical ID."""
    result = {}
    for root, ids in members.items():
        known = {hints[tid] for tid in ids if tid in hints}
        if len(known) == 1:
            side = next(iter(known))
            if side in ("X", "B"):
                result[root] = side
    return result
