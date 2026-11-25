"""Observed launcher ownership, independent of physical release or missing geometry.

Ownership uses source boxes. Valid registrations and separated observed launcher
pairs supply side evidence; missing registration never supplies a transform.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import config_loader
from .geometry import boxes_touch, point_in_polygon, polygon_mask
from .launch_check import _launch_side
from .types import Calibration, Side, Track

DEFAULT_IDENTITY_CONFIG = Path(__file__).resolve().parents[2] / "config/identity.yaml"


@config_loader
def load_identity_policy(path=DEFAULT_IDENTITY_CONFIG):
    value = yaml.safe_load(Path(path).read_text())
    switches = (
        "prelaunch_setup_anchor_enabled",
        "release_clock_trim_enabled",
        "phase_release_evidence_enabled",
        "geometry_window_from_landing_enabled",
    )
    required = {
        "context_wait_s",
        "body_duplicate_overlap_min",
        "prelaunch_setup_wait_s",
        *switches,
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or any(type(value[k]) is not bool for k in switches)
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for k, v in value.items()
            if k not in switches
        )
        or value["body_duplicate_overlap_min"] > 1
    ):
        raise ValueError("invalid identity context policy")
    return value


def _body_pair(observed, overlap_min):
    groups = [
        {"ids": [tid], "box": list(box), "originals": [box]} for tid, box in observed
    ]
    changed = True
    while changed:
        changed = False
        for i, first in enumerate(groups):
            for j in range(i + 1, len(groups)):
                second = groups[j]
                a, b = first["box"], second["box"]
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                small = min(
                    (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
                )
                if small > 0 and intersection / small >= overlap_min:
                    # Duplicate components must agree pairwise. A broad box
                    # spanning disjoint players is ambiguous, never a bridge.
                    for x in first["originals"]:
                        for y in second["originals"]:
                            area = max(0, min(x[2], y[2]) - max(x[0], y[0])) * max(
                                0, min(x[3], y[3]) - max(x[1], y[1])
                            )
                            minimum = min(
                                (x[2] - x[0]) * (x[3] - x[1]),
                                (y[2] - y[0]) * (y[3] - y[1]),
                            )
                            if minimum <= 0 or area / minimum < overlap_min:
                                return {}
                    first["originals"] += second["originals"]
                    first["ids"] += second["ids"]
                    first["box"] = [
                        min(a[0], b[0]),
                        min(a[1], b[1]),
                        max(a[2], b[2]),
                        max(a[3], b[3]),
                    ]
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    if len(groups) != 2:
        return {}
    left, right = sorted(groups, key=lambda g: g["box"][0] + g["box"][2])
    if left["box"][2] > right["box"][0]:
        return {}
    return {
        tid: side for group, side in ((left, "X"), (right, "B")) for tid in group["ids"]
    }


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
    fps: float,
    config: Path = DEFAULT_IDENTITY_CONFIG,
    policy: dict | None = None,
) -> IdentityAnchors:
    if policy is None:
        policy = load_identity_policy(config)
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("identity needs positive source FPS")
    if type(required_frames) is not int or required_frames < 2:
        raise ValueError("identity anchor needs at least two observed frames")
    if not math.isfinite(body_margin_px) or body_margin_px < 0:
        raise ValueError("identity body margin must be nonnegative finite")
    launch_mask = polygon_mask(c.launch_left, c.frame_size) | polygon_mask(
        c.launch_right, c.frame_size
    )
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
    prelaunch_setup = policy.get("prelaunch_setup_anchor_enabled", False)
    for track in tracks:
        if track.cls != "bey" or not track.dets:
            continue
        run = []
        prefix_side = None
        waiting_for_body = False
        track_top_entry = False
        for d in sorted(track.dets, key=lambda d: d.frame):
            max_wait_s = (
                policy.get("prelaunch_setup_wait_s", 1.5)
                if (prelaunch_setup and track_top_entry)
                else policy["context_wait_s"]
            )
            if (
                waiting_for_body
                and (d.frame - track.first_frame()) / fps > max_wait_s
            ):
                break
            observed = bodies.get(d.frame, [])
            measured = {
                tid: _launch_side(reference[tid, d.frame][1], c)
                for tid, _ in observed
                if (tid, d.frame) in reference
            }
            paired = _body_pair(observed, policy["body_duplicate_overlap_min"])
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
            registered = reference.get((track.track_id, d.frame))
            point = registered[1] if registered is not None else None
            in_launch = registered is not None and (
                point_in_polygon(point, c.launch_left)
                or point_in_polygon(point, c.launch_right)
            )
            if prelaunch_setup and registered is not None:
                ba_min_y = min(p[1] for p in c.battle_area)
                l_min_x = min(p[0] for p in c.launch_left + c.launch_right)
                l_max_x = max(p[0] for p in c.launch_left + c.launch_right)
                is_top_zone = (
                    d.box[1] <= 0
                    or (
                        point[1] < ba_min_y
                        and l_min_x - body_margin_px <= point[0] <= l_max_x + body_margin_px
                    )
                ) and not point_in_polygon(point, c.battle_area)
                if is_top_zone:
                    track_top_entry = True
            if not nearby:
                if prelaunch_setup and registered is not None and not in_launch:
                    observed_free = not track_top_entry
                else:
                    observed_free = registered is not None and not in_launch
                if prelaunch_setup and track_top_entry:
                    if (
                        (d.frame - track.first_frame()) / fps > max_wait_s
                        or observed_free
                    ):
                        break
                    run = []
                    waiting_for_body = True
                    continue
                below_bodies = (
                    bool(observed)
                    and d.box[1] > max(box[3] for _, box in observed) + body_margin_px
                )
                if (
                    below_bodies
                    or observed_free
                    or paired
                    or prefix_side is not None
                    or any(side is not None for side in measured.values())
                    or (d.frame - track.first_frame()) / fps > max_wait_s
                ):
                    # Missing one launcher is not evidence of free motion; a
                    # measured free prefix or expired unknown window still is
                    # forbidden from acquiring an anchor at later pickup.
                    break
                run = []
                waiting_for_body = True
                continue
            registered = reference.get((track.track_id, d.frame))
            point = registered[1] if registered is not None else None
            if point is not None and not (
                point_in_polygon(point, c.launch_left)
                or point_in_polygon(point, c.launch_right)
            ):
                if prelaunch_setup and track_top_entry:
                    pass
                else:
                    # A clipped box's centre cannot establish a free-flight prefix.
                    # Wait only while a real nearby launcher is observed but its
                    # role is unknown; later ownership still needs the full run.
                    x0, y0, x1, y1 = d.box
                    clipped = (
                        x0 <= 0 or y0 <= 0 or x1 >= c.frame_size[0] or y1 >= c.frame_size[1]
                    )
                    bx0, by0, bx1, by1 = registered[0]
                    left, top = max(0, math.floor(bx0)), max(0, math.floor(by0))
                    right = min(c.frame_size[0], math.ceil(bx1))
                    bottom = min(c.frame_size[1], math.ceil(by1))
                    overlaps_launch = (
                        left < right
                        and top < bottom
                        and bool(launch_mask[top:bottom, left:right].any())
                    )
                    if (
                        clipped
                        and overlaps_launch
                        and not conflict
                        and prefix_side is None
                        and all(side is None for _, side in nearby)
                        and (d.frame - track.first_frame()) / fps
                        <= max_wait_s
                    ):
                        run = []
                        waiting_for_body = True
                        continue
                    break
            context = {side for _, side in nearby}
            if None in context:
                run = []
                continue
            if len(context) != 1:
                break
            waiting_for_body = False
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
