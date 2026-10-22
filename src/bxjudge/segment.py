"""Round boundaries from active bey track spans."""
import math

from bxjudge.config import Thresholds
from bxjudge.types import Track


def segment_rounds(
    bey_tracks: list[Track], fps: float, th: Thresholds
) -> list[tuple[int, int]]:
    """Return inclusive round spans separated by at least ``round_gap_s`` idle.

    A track is active from its first through its last detection, inclusive;
    internal missing detections do not end its lifetime. Between two spans,
    only frames strictly after the old end and before the new start are idle.
    Empty tracks and non-bey classes carry no active bey evidence. The gap
    test is ``idle_frames / fps >= round_gap_s`` (R41), not the float
    product ``round_gap_s * fps``, which can miss an exact boundary.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    spans = sorted(
        (track.first_frame(), track.last_frame())
        for track in bey_tracks
        if track.cls == "bey" and track.dets
    )
    if not spans:
        return []

    rounds = []
    start, end = spans[0]
    for next_start, next_end in spans[1:]:
        idle_frames = next_start - end - 1
        if idle_frames / fps >= th.round_gap_s:
            rounds.append((start, end))
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    rounds.append((start, end))
    return rounds
