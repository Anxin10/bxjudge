"""Find physical settle episodes and battle-area landing frames (spec 6.4).

Airborne tops can project inside the battle-area polygon. Settling therefore needs
``settle_frames`` of slow vertical motion in one zone, after the first fast fall
if the track contains one. A track with no fast fall has no fall precondition.

Each episode is a maximal run of consecutive observed frames with slow vertical
motion in the same zone. A zone change, fast vertical motion, or any missing frame
starts a new run. Runs qualify only after ``settle_frames`` consecutive observed
frames; missing frames never count as evidence of quiet motion or zone membership.
The first sample has no velocity and is not checked, matching ``geometry.velocities``.

``find_settle`` reports the first episode in any zone. ``find_landing_frame``
reports the first battle-area episode, including after an earlier lid/rim pause.
Launch-window selection belongs to the caller, which can use ``find_settles``.
"""
from __future__ import annotations

from bxjudge.config import Thresholds
from bxjudge.geometry import velocities, zone_of
from bxjudge.types import Calibration, Track, Zone


def find_settles(t: Track, c: Calibration, th: Thresholds) -> list[tuple[int, Zone]]:
    """All distinct settle episodes as ``(first_frame, zone)``, in frame order."""
    centers = t.centers()
    frames = sorted(centers)
    vel = velocities(centers)
    first_fast = min((f for f in frames if f in vel and vel[f][1] > th.fall_speed_px), default=None)
    settles: list[tuple[int, Zone]] = []
    start: int | None = None
    run_zone: Zone | None = None
    emitted = False

    for i, f in enumerate(frames):
        if (first_fast is not None and f <= first_fast) or (
            f in vel and abs(vel[f][1]) > th.fall_speed_px
        ):
            start = None
            continue
        zone = zone_of(centers[f], c)
        if start is None or zone != run_zone or (i > 0 and f - frames[i - 1] != 1):
            start = f
            run_zone = zone
            emitted = False
        if not emitted and f - start + 1 >= th.settle_frames:
            settles.append((start, zone))
            emitted = True
    return settles


def find_settle(t: Track, c: Calibration, th: Thresholds) -> tuple[int | None, Zone | None]:
    """First settled frame and its zone, or ``(None, None)`` if the track never settles."""
    settles = find_settles(t, c, th)
    return settles[0] if settles else (None, None)


def find_landing_frame(t: Track, c: Calibration, th: Thresholds) -> int | None:
    """First physical battle-area settle, even after a lid/rim pause, or None."""
    return next((f for f, zone in find_settles(t, c, th) if zone == "battle_area"), None)
