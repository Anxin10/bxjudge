"""Fixed or handheld camera from early whole-frame motion (plan 2026-11-12-dynamic-calibration,
Task 3)."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from bxjudge.camera_motion import (
    DEFAULT_CAMERA_MODE_CONFIG,
    CameraMotion,
    decide_camera_mode,
    load_camera_policy,
    measure_camera_motion,
    measure_video_motion,
    sample_for_motion,
    sampled_step_s,
)
from tests.helpers import make_video, shift


def textured_frame(seed: int = 3) -> np.ndarray:
    """A 1280x720 frame with plenty of corners (random blocks, slightly blurred)."""
    import cv2

    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)
    big = cv2.resize(small, (1280, 720), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(big, (3, 3), 0)


@pytest.fixture
def policy():
    return load_camera_policy()


def test_shipped_policy_loads_with_its_initial_values(policy):
    assert policy["detection"]["handheld_px_per_s"] == 20.0
    assert policy["handheld"] == {"refind_s": 0.5, "confirm_px": 12.0, "max_unchecked_s": 2.0}
    assert policy["overrides"] == {"batches": {}, "videos": {}}


def test_still_frames_measure_near_zero_and_moving_frames_measure_the_shift():
    base = textured_frame()
    still = measure_camera_motion([base] * 5, 1.0)
    moving = measure_camera_motion([shift(base, 40 * i, 0) for i in range(5)], 1.0)
    assert still.px_per_s < 1.0 and still.pairs == 4
    assert moving.px_per_s == pytest.approx(40, abs=2) and moving.pairs == 4
    # Per playback second: the same shift over half a second is twice as fast.
    half = measure_camera_motion([shift(base, 40 * i, 0) for i in range(5)], 0.5)
    assert half.px_per_s == pytest.approx(80, abs=4)


def test_motion_is_measured_at_the_inference_scale():
    """A 640 px wide video moving 20 px per second moves 40 px/s at the 1280 inference size,
    where the threshold lives."""
    import cv2

    base = cv2.resize(textured_frame(), (640, 360), interpolation=cv2.INTER_AREA)
    got = measure_camera_motion([shift(base, 20 * i, 0) for i in range(5)], 1.0)
    assert got.px_per_s == pytest.approx(40, abs=3)


def test_sampled_step_is_the_whole_frame_interval(policy):
    p = copy.deepcopy(policy)
    p["detection"].update(step_s=0.05)
    assert sampled_step_s(30, p) == pytest.approx(2 / 30)
    assert sampled_step_s(120, p) == pytest.approx(6 / 120)
    p["detection"].update(step_s=0.001)
    assert sampled_step_s(30, p) == pytest.approx(1 / 30)


def test_video_motion_divides_by_the_interval_actually_sampled(tmp_path, policy, monkeypatch):
    import bxjudge.camera_motion as cm

    seen = {}
    monkeypatch.setattr(cm, "sample_for_motion", lambda *a, **k: ["f0", "f1"])
    monkeypatch.setattr(cm, "measure_camera_motion",
                        lambda frames, step_s: seen.update(step_s=step_s) or CameraMotion(1.0, 1))
    p = copy.deepcopy(policy)
    p["detection"].update(step_s=0.05)
    assert measure_video_motion(tmp_path / "v.mp4", 30, p, long_edge=1280,
                                hwaccel="cpu") == CameraMotion(1.0, 1)
    assert seen["step_s"] == pytest.approx(2 / 30)


def test_featureless_frames_give_no_measurement():
    blank = np.full((720, 1280, 3), 90, np.uint8)
    got = measure_camera_motion([blank] * 4, 1.0)
    assert got == CameraMotion(px_per_s=None, pairs=0)
    assert measure_camera_motion([textured_frame()], 1.0) == CameraMotion(None, 0)


def test_mode_threshold_and_unmeasured_fall_back_to_handheld(policy):
    clip = Path("x/20261108/a.mp4")
    fixed = decide_camera_mode(clip, "20261108", CameraMotion(5.7, 9), policy)
    assert (fixed.mode, fixed.source) == ("fixed", "measured")
    assert decide_camera_mode(clip, "20261108", CameraMotion(56.0, 9), policy).mode == "handheld"
    for motion in (None, CameraMotion(None, 0), CameraMotion(3.0, 2)):  # min_pairs is 3
        got = decide_camera_mode(clip, "20261108", motion, policy)
        assert (got.mode, got.source, got.motion) == ("handheld", "unmeasured", motion)


def test_override_beats_measurement(policy):
    p = {**policy, "overrides": {"batches": {"20261101": "fixed"},
                                 "videos": {"a.mp4": "handheld", "c.mp4": "auto"}}}
    b = decide_camera_mode(Path("x/20261101/b.mp4"), "20261101", CameraMotion(70.0, 9), p)
    assert (b.mode, b.source) == ("fixed", "override") and b.motion == CameraMotion(70.0, 9)
    a = decide_camera_mode(Path("x/20261101/a.mp4"), "20261101", CameraMotion(1.0, 9), p)
    assert (a.mode, a.source) == ("handheld", "override")
    # "auto" for a video means no override of its own; the batch override then applies.
    c = decide_camera_mode(Path("x/20261101/c.mp4"), "20261101", CameraMotion(70.0, 9), p)
    assert (c.mode, c.source) == ("fixed", "override")
    p["overrides"]["batches"]["20261101"] = "auto"
    c = decide_camera_mode(Path("x/20261101/c.mp4"), "20261101", CameraMotion(70.0, 9), p)
    assert (c.mode, c.source) == ("handheld", "measured")


def _write(tmp_path, doc) -> Path:
    path = tmp_path / "camera_mode.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


@pytest.mark.parametrize("edit", [
    lambda d: d.pop("handheld"),
    lambda d: d["detection"].pop("min_pairs"),
    lambda d: d["detection"].update(extra=1),
    lambda d: d["detection"].update(min_pairs=2.5),
    lambda d: d["detection"].update(handheld_px_per_s=float("nan")),
    lambda d: d["detection"].update(step_s=0),
    lambda d: d["detection"].update(step_s=20.0),  # longer than the window
    lambda d: d["detection"].update(window_s=2.0),  # 2 frames at 1 s: never 3 pairs
    lambda d: d["handheld"].update(confirm_px=-1),
    lambda d: d["handheld"].update(max_unchecked_s=0.5),  # shorter than refind_s
    lambda d: d["overrides"]["batches"].update({"20261101": "tripod"}),
    lambda d: d["overrides"]["videos"].update({"a.mp4": True}),
    lambda d: d["overrides"].update(videos=["a.mp4"]),
])
def test_policy_rejects_missing_keys_and_unknown_override_values(tmp_path, edit):
    doc = yaml.safe_load(DEFAULT_CAMERA_MODE_CONFIG.read_text(encoding="utf-8"))
    load_camera_policy(_write(tmp_path, copy.deepcopy(doc)))  # the untouched copy loads
    edit(doc)
    with pytest.raises(ValueError):
        load_camera_policy(_write(tmp_path, doc))


@pytest.mark.parametrize("fps", [30, 120])
def test_sampling_steps_in_playback_seconds(tmp_path, policy, fps):
    clip = make_video(tmp_path / f"clip{fps}.mp4", seconds=2.5, fps=fps, size=(320, 180))
    p = copy.deepcopy(policy)
    p["detection"].update(window_s=2.0, step_s=0.5)
    frames = sample_for_motion(clip, fps, p, long_edge=320, hwaccel="cpu")
    # Frames 0, 0.5, 1.0 and 1.5 s: four at either rate, not a fixed frame step.
    assert len(frames) == 4 and frames[0].shape[:2] == (180, 320)
