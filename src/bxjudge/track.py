"""Native ByteTrack association, with high-score-only adjudication evidence."""

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
    tracks: list[Track] = []
    # Native tracklet objects remain stable when their initially -1 ID is confirmed.
    pending: dict[object, Detection] = {}
    accepted: dict[object, Track] = {}
    last_strong: dict[object, int] = {}
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
        if any(not np.isfinite(d.score) or not 0 <= d.score <= 1 for d in detections):
            raise ValueError("detector returned invalid confidence")
        for cls, tracker in trackers.items():
            # A low-score chain may update native motion, but cannot keep a public
            # identity alive after the native lost-track TIME budget without evidence.
            tracker.tracks = [
                t
                for t in tracker.tracks
                if (frame_idx - last_strong[t]) / fps
                <= tracker.maximum_time_without_update
            ]
            current = [
                d
                for d in detections
                if d.cls == cls and d.score >= cfg["candidate_confidence"]
            ]
            sv_dets = sv.Detections(
                xyxy=np.array([d.box for d in current], dtype=float).reshape(-1, 4),
                confidence=np.array([d.score for d in current], dtype=float),
                class_id=np.full(len(current), CLASSES.index(cls), dtype=int),
            )
            before = set(tracker.tracks)
            tracked = tracker.update(sv_dets, timestamp=frame_idx / fps)
            unmatched = list(current)
            for tracklet in tracker.tracks:
                if tracklet not in before:
                    seed = next(
                        d
                        for d in unmatched
                        if d.score >= cfg["confidence"]
                        and np.array_equal(d.box, tracklet.get_state_bbox())
                    )
                    unmatched.remove(seed)
                    pending[tracklet] = seed
                    last_strong[tracklet] = frame_idx
            by_id = {t.tracker_id: t for t in tracker.tracks if t.tracker_id >= 0}
            for box, score, identity in zip(
                tracked.xyxy, tracked.confidence, tracked.tracker_id, strict=True
            ):
                # Weak boxes participate only in association. A native ID assigned
                # through weak matches does not confirm the seed as strong evidence.
                if identity < 0 or score < cfg["confidence"]:
                    continue
                tracklet = by_id[int(identity)]
                detection = Detection(
                    frame_idx, cls, tuple(map(float, box)), float(score)
                )
                last_strong[tracklet] = frame_idx
                if tracklet not in accepted:
                    # Seed + independently matched HIGH observation: exactly the
                    # native seed/confirmation contract, with weak hits excluded.
                    result = Track(len(tracks) + 1, cls, [pending.pop(tracklet)])
                    tracks.append(result)
                    accepted[tracklet] = result
                accepted[tracklet].dets.append(detection)
            alive = {t for tr in trackers.values() for t in tr.tracks}
            pending = {t: seed for t, seed in pending.items() if t in alive}
            accepted = {t: result for t, result in accepted.items() if t in alive}
            last_strong = {t: f for t, f in last_strong.items() if t in alive}
    return tracks
