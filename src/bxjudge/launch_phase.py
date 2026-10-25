"""Observed mounting/release and mutual-unique short fragment continuation.

This never fabricates points or uses floor/finish evidence to define launch time.
Unknown release/identity remains review-required instead of being silently resolved.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import config_loader
from .types import Track

DEFAULT_LAUNCH_POLICY = Path(__file__).resolve().parents[2] / "config/launch_phase.yaml"


@config_loader
def load_launch_policy(path=DEFAULT_LAUNCH_POLICY):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = {
        "held_s",
        "separation_s",
        "body_margin_px",
        "link_gap_s",
        "link_error_px",
        "link_scale_ratio",
        "link_min_direction_cos",
        "link_velocity_error_px_s",
        "link_min_speed_px_s",
    }
    if not isinstance(raw, dict) or set(raw) != names:
        raise ValueError("incomplete launch_phase policy")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0
        for v in raw.values()
    ):
        raise ValueError("launch_phase values must be positive finite")
    if raw["link_min_direction_cos"] > 1:
        raise ValueError("link_min_direction_cos must be <=1")
    if raw["link_scale_ratio"] < 1:
        raise ValueError("link_scale_ratio must be >=1")
    return dict(raw)


@dataclass
class PreparedLaunch:
    tracks: list[Track]
    members: dict[int, list[int]]
    releases: dict[int, int]
    excluded_held: list[int]
    notes: list[str]
    diagnostics: dict[int, dict]

    def record(self):
        return {
            "members": {str(k): v for k, v in self.members.items()},
            "releases": {str(k): v for k, v in self.releases.items()},
            "excluded_held": self.excluded_held,
            "notes": self.notes,
            "diagnostics": {str(k): v for k, v in self.diagnostics.items()},
        }


def _distance(a, b):
    return math.hypot(
        max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0)
    )


def _size(box):
    return math.hypot(box[2] - box[0], box[3] - box[1])


def _box_clear_runs(detections, bodies, owner, margin, required):
    """Observed rectangle clearance only; never a physical-release estimate."""
    if owner is None:
        return []
    runs = []
    current = []

    def finish():
        if len(current) >= required:
            runs.append(
                {
                    "body_track_id": owner,
                    "first_frame": current[0],
                    "confirmed_at_frame": current[required - 1],
                    "last_frame": current[-1],
                }
            )

    for detection in detections:
        observed = [box for tid, box in bodies.get(detection.frame, []) if tid == owner]
        clear = len(observed) == 1 and _distance(detection.box, observed[0]) > margin
        if not clear or (current and detection.frame != current[-1] + 1):
            finish()
            current = []
        if clear:
            current.append(detection.frame)
    finish()
    return runs


def prepare_launch_tracks(tracks, fps, policy):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive finite")
    held_n = max(2, math.ceil(policy["held_s"] * fps))
    free_n = max(2, math.ceil(policy["separation_s"] * fps))
    bodies = {}
    for t in tracks:
        if t.cls == "launcher":
            for d in t.dets:
                bodies.setdefault(d.frame, []).append((t.track_id, d.box))
    prepared = []
    releases = {}
    excluded = []
    notes = []
    diagnostics = {}
    for t in tracks:
        if t.cls != "bey" or not t.dets:
            prepared.append(Track(t.track_id, t.cls, list(t.dets)))
            continue
        ordered = sorted(t.dets, key=lambda d: d.frame)
        held = []
        free = []
        owner = None
        release = None
        mounted = False
        interruption = None
        for index, d in enumerate(ordered):
            if mounted and d.frame != ordered[index - 1].frame + 1:
                interruption = {
                    "reason": "bey_observation_gap",
                    "frame": ordered[index - 1].frame + 1,
                    "next_bey_frame": d.frame,
                }
                break
            candidates = [
                tid
                for tid, b in bodies.get(d.frame, [])
                if _distance(d.box, b) <= policy["body_margin_px"]
            ]
            if not mounted:
                if len(candidates) == 1:
                    if held and (d.frame != held[-1] + 1 or owner != candidates[0]):
                        held = []
                    owner = candidates[0]
                    held.append(d.frame)
                    if len(held) >= held_n:
                        mounted = True
                else:
                    held = []
                # Only an observed prefix mounting can redefine this track's first appearance.
                if not held:
                    interruption = {
                        "reason": "prefix_launcher_ambiguous"
                        if candidates
                        else "prefix_launcher_missing",
                        "frame": d.frame,
                    }
                    break
            else:
                observed = [b for tid, b in bodies.get(d.frame, []) if tid == owner]
                if len(observed) != 1:
                    interruption = {
                        "reason": "launcher_observation_missing"
                        if not observed
                        else "launcher_observation_ambiguous",
                        "frame": d.frame,
                    }
                    break
                if (
                    len(observed) == 1
                    and _distance(d.box, observed[0]) > policy["body_margin_px"]
                ):
                    if free and d.frame != free[-1] + 1:
                        free = []
                    free.append(d.frame)
                    if len(free) >= free_n:
                        release = free[0]
                        break
                else:
                    free = []
        # This evidence can recover after an interruption, but cannot change the
        # strict release result, track lifetime, or eligibility deadline.
        diagnostics[t.track_id] = {
            "original_first_frame": ordered[0].frame,
            "body_track_id": owner if mounted else None,
            "interruption": interruption,
            "box_clear_runs": _box_clear_runs(
                ordered,
                bodies,
                owner if mounted else None,
                policy["body_margin_px"],
                free_n,
            ),
        }
        if release is not None:
            releases[t.track_id] = release
            ordered = [d for d in ordered if d.frame >= release]
        elif mounted:
            # Exclude a wholly held observed track; an observation without a body is
            # not evidence of separation and remains in the review population.
            all_held = all(
                any(
                    _distance(d.box, b) <= policy["body_margin_px"]
                    for tid, b in bodies.get(d.frame, [])
                    if tid == owner
                )
                for d in ordered
            )
            if all_held:
                excluded.append(t.track_id)
                continue
            notes.append(f"launch_release_unobserved@{t.track_id}")
        prepared.append(Track(t.track_id, t.cls, ordered))
    members = {t.track_id: [t.track_id] for t in prepared}
    beys = [t for t in prepared if t.cls == "bey" and t.dets]
    candidates = []
    for a in beys:
        if len(a.dets) < 2:
            continue
        da = sorted(a.dets, key=lambda d: d.frame)
        p0 = a.center_at(da[-2].frame)
        p1 = a.center_at(da[-1].frame)
        dt = da[-1].frame - da[-2].frame
        if dt != 1:
            continue
        velocity = ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)
        for b in beys:
            if a is b:
                continue
            gap = b.first_frame() - a.last_frame()
            if gap <= 0 or gap / fps > policy["link_gap_s"]:
                continue
            # Never join a newly mounted/released top onto another top's identity.
            if b.track_id in releases or any(
                n.endswith(f"@{b.track_id}") for n in notes
            ):
                continue
            db = sorted(b.dets, key=lambda d: d.frame)
            if len(db) < 2 or db[1].frame - db[0].frame != 1:
                continue
            pb = b.center_at(b.first_frame())
            next_b = b.center_at(db[1].frame)
            velocity_b = (next_b[0] - pb[0], next_b[1] - pb[1])
            va = math.hypot(*velocity) * fps
            vb = math.hypot(*velocity_b) * fps
            if (
                math.hypot(velocity[0] - velocity_b[0], velocity[1] - velocity_b[1])
                * fps
                > policy["link_velocity_error_px_s"]
            ):
                continue
            if max(va, vb) >= policy["link_min_speed_px_s"]:
                if min(va, vb) < policy["link_min_speed_px_s"]:
                    continue
                cosine = (
                    (velocity[0] * velocity_b[0] + velocity[1] * velocity_b[1])
                    * fps
                    * fps
                    / (va * vb)
                )
                if cosine < policy["link_min_direction_cos"]:
                    continue
            pred = (p1[0] + velocity[0] * gap, p1[1] + velocity[1] * gap)
            sa = _size(da[-1].box)
            sb = _size(b.box_at(b.first_frame()))
            if (
                min(sa, sb) <= 0
                or max(sa, sb) / min(sa, sb) > policy["link_scale_ratio"]
            ):
                continue
            if math.dist(pred, pb) <= policy["link_error_px"]:
                candidates.append((a.track_id, b.track_id))
    outgoing = {}
    incoming = {}
    for a, b in candidates:
        outgoing.setdefault(a, []).append(b)
        incoming.setdefault(b, []).append(a)
    links = {
        a: bs[0]
        for a, bs in outgoing.items()
        if len(bs) == 1 and len(incoming[bs[0]]) == 1
    }
    by_id = {t.track_id: t for t in prepared}
    consumed = set()
    result = []
    for t in sorted(prepared, key=lambda t: t.first_frame() if t.dets else -1):
        if t.track_id in consumed:
            continue
        ids = [t.track_id]
        dets = list(t.dets)
        tail = t.track_id
        while tail in links:
            tail = links[tail]
            ids.append(tail)
            dets.extend(by_id[tail].dets)
            consumed.add(tail)
        members[t.track_id] = ids
        result.append(Track(t.track_id, t.cls, sorted(dets, key=lambda d: d.frame)))
    return PreparedLaunch(
        result,
        {t.track_id: members[t.track_id] for t in result},
        releases,
        excluded,
        notes,
        diagnostics,
    )
