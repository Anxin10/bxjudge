"""E4b: a stop must be preceded by continuous observation (hand-occlusion guard)."""

import pytest

from bxjudge.config import load_thresholds
from bxjudge.finish import continuous_approach_frames, load_finish_fusion_config, stop_frame
from tests.helpers import make_track

BASE = {"enabled": True, "still_frames": 12, "still_moving_px": 3.8,
        "still_max_rot_rate_deg": 0.3, "still_min_laplacian_var": 80.0}


@pytest.fixture
def th():
    return load_thresholds()


def _still_after(gap):
    """Moving fast, then still from frame 101 (100 still jumps); `gap` frames are unobserved."""
    centers = {f: (100.0 + f * 10.0, 200.0) for f in range(100)}
    centers.update({f: (1100.0, 200.0) for f in range(100, 200)})
    for f in gap:
        centers.pop(f)
    frames = range(200)
    return (make_track(1, "bey", centers), {f: 8.0 for f in frames},
            {f: 0.0 for f in frames}, {f: 150.0 for f in frames})


def _run(th, track, activity, rate, lap, **extra):
    return stop_frame(track, activity, rate, "right", th, laplacian=lap,
                      fusion={**BASE, **extra})


def test_guard_default_off_in_shipped_configs():
    for name in ("config/finish_fusion.yaml", "config/finish_fusion_enabled.yaml"):
        policy = load_finish_fusion_config(name)
        assert policy["continuous_approach_guard_enabled"] is False
        assert continuous_approach_frames(policy) is None


def test_guard_off_keeps_stop_after_gap(th):
    assert _run(th, *_still_after(range(90, 95)))[0] == 101


def test_guard_rejects_stop_right_after_gap(th):
    args = _still_after(range(90, 95))
    assert _run(th, *args, continuous_approach_guard_enabled=True,
                continuous_approach_frames=30) == (None, None)


def test_guard_keeps_stop_with_continuous_approach(th):
    args = _still_after(range(40, 45))  # gap long before the look-back window
    assert _run(th, *args, continuous_approach_guard_enabled=True,
                continuous_approach_frames=30) == (101, "still")


def test_guard_rejects_stop_too_soon_after_track_start(th):
    args = _still_after(range(0, 80))
    assert _run(th, *args, continuous_approach_guard_enabled=True,
                continuous_approach_frames=30) == (None, None)


@pytest.mark.parametrize("extra", [
    {"continuous_approach_guard_enabled": 1, "continuous_approach_frames": 30},
    {"continuous_approach_guard_enabled": True, "continuous_approach_frames": 0},
    {"continuous_approach_guard_enabled": True, "continuous_approach_frames": 30.0},
    {"continuous_approach_guard_enabled": True},
])
def test_guard_rejects_invalid_config(th, extra):
    with pytest.raises(ValueError):
        _run(th, *_still_after(()), **extra)
