"""Per-round physical tracks using observed points only; estimates stay in the viewer."""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise

import cv2

from .geometry import box_center, boxes_touch, zone_of
from .identity import IdentityAnchors
from .launch_phase import LaunchSegment
from .two_bey_tracking import TwoBeyTracker
from .types import Calibration, Detection, Side, Track


def appearance_descriptor(image, box, bins):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = (
        max(0, math.floor(x1)),
        max(0, math.floor(y1)),
        min(width, math.ceil(x2)),
        min(height, math.ceil(y2)),
    )
    if min(x2 - x1, y2 - y1) < 2:
        return None
    crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([crop], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]).ravel()
    return (hist / hist.sum()).tolist()


@dataclass
class PhysicalRound:
    tracks: list[Track]
    members: dict[int, list[int]]
    record: dict



def infer_opponent_side(
    anchors: IdentityAnchors,
    physical: PhysicalRound,
    registered: list[Track],
    calibration: Calibration,
    *,
    required_frames: int,
) -> dict | None:
    """Infer the other player in a verified two-top prefix of this round.

    The caller supplies observed, phase-eligible boxes registered to the
    calibration. This evidence only assigns a player, never a release or landing.
    """
    record = physical.record
    if len(registered) != 2 or record["pair_confirmed_at_frame"] is None:
        return None
    claims = {
        track.track_id: {
            anchors.sides[native_id]
            for native_id in physical.members[track.track_id]
            if native_id in anchors.sides
        }
        for track in registered
    }
    known, unknown = sorted(registered, key=lambda track: -len(claims[track.track_id]))
    if len(claims[known.track_id]) != 1 or claims[unknown.track_id]:
        return None
    (side,) = claims[known.track_id]
    blocked_frames = {
        d["frame"] for d in record["unassigned"]
        if d["adjudication_eligible"] and d["reason"] != "duplicate_box"
    }
    # Once a preceding identity was lost, two new boxes alone cannot recover it.
    ambiguous_from = min(
        [*record["identity_unknown_since"].values(),
         *(start for spans in record["identity_unknown_intervals"].values() for start, _ in spans)],
        default=math.inf,
    )
    known_boxes, unknown_boxes = ({d.frame: d.box for d in t.dets} for t in (known, unknown))
    run = []
    for frame in sorted(known_boxes.keys() & unknown_boxes.keys()):
        if frame >= ambiguous_from:
            return None
        a, b = known_boxes[frame], unknown_boxes[frame]
        if (frame in blocked_frames or boxes_touch(a, b)
                or any(zone_of(box_center(box), calibration) != "battle_area" for box in (a, b))):
            run.clear()
            continue
        if run and frame != run[-1] + 1:
            run.clear()
        run.append(frame)
        if len(run) == required_frames:
            break
    if len(run) != required_frames:
        return None

    # A direct launcher anchor must predate the first clean pair and belong
    # to a native track actually observed in this actor's prefix.
    observed = {
        d["native_id"] for d in record["observations"]
        if d["identity_valid"] and d["frame"] <= run[0]
    }
    evidence = [
        (anchors.evidence[native_id][-1]["frame"], native_id)
        for native_id in physical.members[known.track_id]
        if native_id in observed and anchors.evidence.get(native_id)
        and anchors.evidence[native_id][-1]["frame"] <= run[0]
    ]
    if not evidence:
        return None
    last, native_id = min(evidence)
    return {
        "track_id": unknown.track_id,
        "side": "B" if side == "X" else "X",
        "opponent_track_id": known.track_id,
        "launcher_anchor_native_id": native_id,
        "launcher_anchor_last_frame": last,
        "frames": run,
    }


def round_release_evidence(
    physical: PhysicalRound,
    registered: list[Track],
    sides: dict[int, Side],
    segments: dict[int, list[LaunchSegment]],
    calibration: Calibration,
    *,
    required_frames: int,
) -> dict[int, dict]:
    """Use only release evidence assigned to this physical actor in this round."""
    record = physical.record
    if len(registered) != 2 or record["pair_confirmed_at_frame"] is None:
        return {}
    counts = Counter(sides[t.track_id] for t in registered if t.track_id in sides)
    observed = {
        (o["entity_id"], o["frame"]): o["native_id"]
        for o in record["observations"] if o["identity_valid"]
    }
    candidates = defaultdict(list)
    actor_ids = {t.track_id: actor for actor, t in enumerate(physical.tracks, 1)}
    for track in registered:
        actor = actor_ids[track.track_id]
        if counts[sides.get(track.track_id)] != 1:
            continue
        for native_id in physical.members[track.track_id]:
            states = segments.get(native_id, [])
            for mount, release in pairwise(states):
                if release.state != "separated":
                    continue
                first = release.first_frame
                confirmed = release.confirmed_at_frame
                if not record["first_frame"] <= first <= confirmed <= record["last_frame"]:
                    continue
                if any(observed.get((actor, f)) != native_id for f in range(first, confirmed + 1)):
                    continue
                candidates[track.track_id].append({
                    "native_id": native_id, "launcher_id": release.launcher_id,
                    "mounted": [mount.first_frame, mount.last_frame],
                    "frame": first, "confirmed_at_frame": confirmed,
                })
    selected = {track_id: choices[0] for track_id, choices in candidates.items() if len(choices) == 1}
    if not selected:
        return {}

    # A later pickup and relaunch cannot supply the clock of an already
    # observed exchange. Projection is a veto only, never release evidence.
    a, b = ({d.frame: d.box for d in t.dets} for t in registered)
    before = max(proof["mounted"][0] for proof in selected.values())
    previous = None
    length = 0
    for f in sorted(a.keys() & b.keys()):
        if f >= before:
            break
        if boxes_touch(a[f], b[f]) or any(
            zone_of(box_center(box), calibration) != "battle_area" for box in (a[f], b[f])
        ):
            previous = None
            length = 0
            continue
        length = length + 1 if previous == f - 1 else 1
        previous = f
        if length >= required_frames:
            return {
                track_id: proof for track_id, proof in selected.items()
                if proof["mounted"][0] <= f
            }
    return selected

def reconstruct_round_tracks(
    tracks, first, last, fps, frame_size, features, policy, id_offset
):
    if (
        type(first) is not int
        or type(last) is not int
        or first < 0
        or last < first
        or type(id_offset) is not int
        or id_offset < 0
    ):
        raise ValueError("invalid round source span or identity namespace")
    by_frame = defaultdict(list)
    for t in tracks:
        if t.cls == "bey":
            for d in t.dets:
                if first <= d.frame <= last:
                    by_frame[d.frame].append(
                        {
                            "native_id": t.track_id,
                            "box": d.box,
                            "score": d.score,
                            "appearance": features.get((d.frame, tuple(d.box))),
                        }
                    )
    tracker = TwoBeyTracker(frame_size, fps, policy)
    observations = []
    unassigned = []
    for f in range(first, last + 1):
        result = tracker.update(f, by_frame.get(f, []))
        observations.extend(
            {k: v for k, v in d.items() if k != "appearance"}
            for d in result["observed"]
        )
        unassigned.extend(
            {k: v for k, v in d.items() if k != "appearance"}
            for d in result["unassigned"]
        )
    entities = sorted({d["entity_id"] for d in observations})
    per_frame = Counter(d["frame"] for d in observations)
    streak = 0
    confirmed_at = None
    for f in range(first, last + 1):
        streak = streak + 1 if per_frame[f] == 2 else 0
        if streak >= policy["pair_confirmation_frames"]:
            confirmed_at = f
            break
    hypotheses = []
    tentative_entities = len(entities)
    if confirmed_at is None:
        hypotheses = observations
        unassigned.extend(
            {
                **{k: v for k, v in d.items() if k != "entity_id"},
                "tentative_entity_id": d["entity_id"],
                "reason": "pair_not_confirmed",
            }
            for d in hypotheses
        )
        observations = []
        entities = []
    physical = [
        Track(
            id_offset + eid,
            "bey",
            [
                Detection(d["frame"], "bey", tuple(d["box"]), d["score"])
                for d in observations
                if d["entity_id"] == eid
            ],
        )
        for eid in entities
    ]
    members = {
        id_offset + eid: sorted(
            tid for tid, owner in tracker.native_anchor_owner.items() if owner == eid
        )
        for eid in entities
    }
    return PhysicalRound(
        physical,
        members,
        {
            "method": "two_physical_beys_per_round",
            "first_frame": first,
            "last_frame": last,
            "configured_slots": 2,
            "observed_entities": len(entities),
            "segment_status": "paired_observations"
            if confirmed_at is not None
            else "pair_not_confirmed",
            "pair_confirmed_at_frame": confirmed_at,
            "tentative_entities": tentative_entities,
            "hypothesis_observations": hypotheses,
            "max_physical_tops_per_frame": max(
                Counter(d["frame"] for d in observations).values(), default=0
            ),
            "raw_input_bey_observations": sum(map(len, by_frame.values())),
            "members": {str(k): v for k, v in members.items()},
            "observations": observations,
            "identity_unknown_since": {str(id_offset + eid): frame
                for eid, frame in tracker.identity_unknown_since.items() if eid in entities},
            "identity_unknown_intervals": {str(id_offset + eid): spans
                for eid, spans in tracker.identity_unknown_intervals.items() if eid in entities},
            "identity_merge_episodes": tracker.merge_episodes
                + ([{"merge_frame": tracker.pending_merge["start"], "split_frame": None,
                     "decision": "unknown", "reason": "no_split_observed"}]
                   if tracker.pending_merge is not None else []),
            "unassigned": unassigned,
            "reacquisitions": tracker.reacquisitions,
            "interpolation_used_for_judging": False,
        },
    )
