"""Per-class ByteTrack using Roboflow's supported trackers/supervision API."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from .detect import Detector
from .detector_config import CLASSES, load_policy
from .types import Detection, Track


def track_frames(
    frames: Iterable[tuple[int, np.ndarray]],
    detector: Detector,
    fps: float,
    *,
    config: Path | str = "config/detection.yaml",
) -> list[Track]:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    cfg = load_policy(config)
    trackers = {
        cls: ByteTrackTracker(frame_rate=fps, **cfg["tracking"]) for cls in CLASSES
    }
    tracks: dict[tuple[str, int], Track] = {}
    pending: dict[object, list[Detection]] = {}
    previous = -1
    for frame_idx, frame in frames:
        if type(frame_idx) is not int or frame_idx < 0 or frame_idx <= previous:
            raise ValueError(
                "source frame indices must be nonnegative and strictly increasing"
            )
        previous = frame_idx
        detections = detector(frame, frame_idx)
        if any(d.frame != frame_idx or d.cls not in CLASSES for d in detections):
            raise ValueError("detector returned invalid source frame or class")
        for cls, tracker in trackers.items():
            current = [d for d in detections if d.cls == cls]
            sv_dets = sv.Detections(
                xyxy=np.array([d.box for d in current], dtype=float).reshape(-1, 4),
                confidence=np.array([d.score for d in current], dtype=float),
                class_id=np.full(len(current), CLASSES.index(cls), dtype=int),
            )
            # Absolute timestamps let the maintained tracker age/predict through gaps.
            before = set(tracker.tracks)
            tracked = tracker.update(sv_dets, timestamp=frame_idx / fps)
            # New tracklets always emit -1 until the next matching observation.
            # Their public object identity links the exact seed to its confirmed ID.
            unmatched = list(current)
            for tracklet in tracker.tracks:
                if tracklet not in before:
                    seed = next(
                        d
                        for d in unmatched
                        if np.array_equal(d.box, tracklet.get_state_bbox())
                    )
                    unmatched.remove(seed)
                    pending[tracklet] = [seed]
                if tracklet.tracker_id >= 0 and tracklet in pending:
                    key = (cls, int(tracklet.tracker_id))
                    tracks[key] = Track(len(tracks) + 1, cls, pending.pop(tracklet))
            alive = {t for tracker_ in trackers.values() for t in tracker_.tracks}
            pending = {
                t: observations for t, observations in pending.items() if t in alive
            }
            for box, score, identity in zip(
                tracked.xyxy, tracked.confidence, tracked.tracker_id, strict=True
            ):
                if identity < 0:
                    continue
                key = (cls, int(identity))
                if key not in tracks:
                    tracks[key] = Track(len(tracks) + 1, cls, [])
                tracks[key].dets.append(
                    Detection(frame_idx, cls, tuple(map(float, box)), float(score))
                )
    return list(tracks.values())
