"""Report timestamp and whole-image anomalies for manual review."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bxjudge.timebase import Timeline
from bxjudge.video import iter_frames


@dataclass(frozen=True)
class AnomalyPolicy:
    timestamp_gap_ratio: float
    noise_percentile: float
    freeze_noise_ratio: float
    min_freeze_transitions: int
    content_jump_ratio: float
    analysis_long_edge: int

    def __post_init__(self) -> None:
        if (
            self.timestamp_gap_ratio <= 1
            or not 0 < self.noise_percentile < 50
            or not 0 < self.freeze_noise_ratio < 1
            or type(self.min_freeze_transitions) is not int
            or self.min_freeze_transitions < 2
            or self.content_jump_ratio <= 1
            or type(self.analysis_long_edge) is not int
            or self.analysis_long_edge < 16
            or not np.isfinite(self.timestamp_gap_ratio + self.content_jump_ratio)
        ):
            raise ValueError("invalid anomaly thresholds")


@dataclass(frozen=True)
class Anomaly:
    kind: str
    first_frame: int
    last_frame: int


def detect_anomalies(pts_s, mean_change, global_change, policy: AnomalyPolicy) -> list[Anomaly]:
    gaps = np.diff(np.asarray(pts_s, dtype=float))
    mean = np.asarray(mean_change, dtype=float)
    global_diff = np.asarray(global_change, dtype=float)
    if (
        gaps.shape != mean.shape or gaps.shape != global_diff.shape
        or not np.isfinite(gaps).all() or np.any(gaps <= 0)
        or not np.isfinite(mean).all() or np.any(mean < 0)
        or not np.isfinite(global_diff).all() or np.any(global_diff < 0)
    ):
        raise ValueError("invalid timestamp or image differences")
    if not len(gaps):
        return []
    jump = gaps > np.median(gaps) * policy.timestamp_gap_ratio
    normal = ~jump
    positive = mean[normal & (mean > 0)]
    floor = np.percentile(positive, policy.noise_percentile) if len(positive) else 0
    still = normal & (mean <= floor * policy.freeze_noise_ratio)
    typical = global_diff[normal & ~still & (global_diff > 0)]
    content_limit = np.median(typical) * policy.content_jump_ratio if len(typical) else np.inf
    content = normal & (global_diff > content_limit)
    events = [Anomaly("timestamp_gap", int(i), int(i + 1)) for i in np.flatnonzero(jump)]
    events += [Anomaly("content_jump", int(i), int(i + 1)) for i in np.flatnonzero(content)]
    ends = np.diff(np.r_[0, still, 0].astype(np.int8))
    for a, b in zip(np.flatnonzero(ends == 1), np.flatnonzero(ends == -1), strict=True):
        if b - a >= policy.min_freeze_transitions:
            events.append(Anomaly("freeze", int(a), int(b)))
    return sorted(events, key=lambda e: (e.first_frame, e.kind))


class ImageChanges:
    """Collect compact statistics from every decoded image in source order."""

    def __init__(self, policy: AnomalyPolicy) -> None:
        self.policy = policy
        self.previous = None
        self.mean_change = []
        self.global_change = []
        self.count = 0

    def observe(self, frame: int, image: np.ndarray) -> None:
        if frame != self.count:
            raise ValueError("decoder lost source frame ordering")
        height, width = image.shape[:2]
        scale = min(1, self.policy.analysis_long_edge / max(height, width))
        gray = cv2.cvtColor(
            cv2.resize(image, (max(4, round(width * scale)), max(4, round(height * scale))),
                       interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        if self.previous is not None:
            diff = cv2.absdiff(gray, self.previous)
            self.mean_change.append(float(diff.mean()))
            # A broad change must reach most of the 16 tiles.
            tiles = diff[:len(diff) // 4 * 4, :diff.shape[1] // 4 * 4]
            tiles = tiles.reshape(4, tiles.shape[0] // 4, 4, tiles.shape[1] // 4)
            self.global_change.append(float(np.percentile(tiles.mean(axis=(1, 3)), 25)))
        self.previous = gray
        self.count += 1

    def events(self, clock: Timeline) -> list[Anomaly]:
        if self.count != len(clock.pts_s):
            raise ValueError(f"decoded {self.count} frames but read {len(clock.pts_s)} timestamps")
        return detect_anomalies(clock.pts_s, self.mean_change, self.global_change, self.policy)


def scan_video(path: Path, clock: Timeline, policy: AnomalyPolicy) -> list[Anomaly]:
    changes = ImageChanges(policy)
    for i, image in iter_frames(path, long_edge=policy.analysis_long_edge, hwaccel="cpu"):
        changes.observe(i, image)
    return changes.events(clock)


def annotate_round(row, clock: Timeline, events, first: int, last: int, stop_windows=()) -> None:
    """Record video diagnostics after all decision fields have been computed."""
    records = []
    for event in events:
        if event.first_frame > last or event.last_frame < first:
            continue
        overlaps = [
            {"side": side, "kind": kind, "first_frame": a, "last_frame": b}
            for side, kind, a, b in stop_windows
            if event.kind == "freeze" and a <= event.last_frame and b >= event.first_frame
        ]
        records.append({
            "type": event.kind,
            "first_frame": event.first_frame,
            "last_frame": event.last_frame,
            "playback_ms": (clock.pts_s[event.last_frame] - clock.pts_s[event.first_frame]) * 1000,
            "physical_ms": clock.elapsed_ms(event.first_frame, event.last_frame),
            "overlapping_stop_evidence": overlaps,
        })
        row.review_reasons.append(
            f"video_anomaly:{event.kind}@{event.first_frame}-{event.last_frame}"
        )
    if records:
        if row.motion_evidence is None:
            row.motion_evidence = {}
        row.motion_evidence["video_anomalies"] = records
