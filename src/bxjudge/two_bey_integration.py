"""Per-round physical tracks using observed points only; estimates stay in the viewer."""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2

from .two_bey_tracking import TwoBeyTracker
from .types import Detection, Track


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
            tid for tid, owner in tracker.native_owners.items() if owner == eid
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
            "max_physical_tops_per_frame": max(
                Counter(d["frame"] for d in observations).values(), default=0
            ),
            "raw_input_bey_observations": sum(map(len, by_frame.values())),
            "members": {str(k): v for k, v in members.items()},
            "observations": observations,
            "unassigned": unassigned,
            "interpolation_used_for_judging": False,
        },
    )
