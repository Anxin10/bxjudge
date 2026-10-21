"""Settle / landing-frame detection (spec 6.4: landing must not be judged from 2D position alone).

The camera looks at the stadium obliquely, so a top that is still falling can project into the
battle_area polygon. We therefore look for the first frame at which the top has *stopped
falling and stays put in one zone*:

1. a fall phase precedes it: some earlier velocity has ``vy > fall_speed_px`` (image y grows
   downward). A track that never falls this fast has no such precondition;
2. over a window of ``settle_frames`` frames starting at the candidate, ``|vy| <= fall_speed_px``;
3. all samples in the window are in the same zone.

Gap rule (detections can be missing): "consecutive ``settle_frames`` frames" is measured in
frame time, not in sample count. For a candidate sample at frame f the window is every
observed sample from f up to and including the first observed frame ``e >= f + settle_frames - 1``
(so it spans at least ``settle_frames`` frames, and a missing detection near the end of the
window only extends it by a frame instead of blocking settling forever). A window is
rejected if two neighbouring samples in it are further apart than ``max(1, settle_frames // 2)``
frames: with a long blackout the averaged velocity says nothing about what happened inside, so
the top then settles at the first sample after the blackout instead. Velocities come from
``geometry.velocities`` (already divided by the frame gap). The first sample of a track has no
velocity; it is not checked.
"""
from __future__ import annotations

from bxjudge.config import Thresholds
from bxjudge.geometry import velocities, zone_of
from bxjudge.types import Calibration, Track, Zone


def _window(frames: list[int], i: int, settle_frames: int) -> list[int] | None:
    """Observed frames from ``frames[i]`` up to the first one spanning ``settle_frames`` frames,
    or None if the track ends first or a gap inside is too long."""
    max_gap = max(1, settle_frames // 2)
    end = frames[i] + settle_frames - 1
    window = [frames[i]]
    for g in frames[i + 1 :]:
        if window[-1] >= end:
            break
        if g - window[-1] > max_gap:
            return None
        window.append(g)
    return window if window[-1] >= end else None


def find_settle(t: Track, c: Calibration, th: Thresholds) -> tuple[int | None, Zone | None]:
    """First settled frame and its zone, or ``(None, None)`` if the track never settles."""
    centers = t.centers()
    frames = sorted(centers)
    vel = velocities(centers)
    first_fast = min((f for f in frames if f in vel and vel[f][1] > th.fall_speed_px), default=None)

    for i, f in enumerate(frames):
        if first_fast is not None and f <= first_fast:
            continue
        window = _window(frames, i, th.settle_frames)
        if window is None:
            continue
        if any(g in vel and abs(vel[g][1]) > th.fall_speed_px for g in window):
            continue
        zones = {zone_of(centers[g], c) for g in window}
        if len(zones) == 1:
            return f, zones.pop()
    return None, None


def find_landing_frame(t: Track, c: Calibration, th: Thresholds) -> int | None:
    """Settled frame if the top settled on the battle_area floor, else None (lid, rim, over
    zones, xtreme, outside, or never settled all count as not landed)."""
    frame, zone = find_settle(t, c, th)
    return frame if zone == "battle_area" else None
