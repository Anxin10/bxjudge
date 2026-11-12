"""Is the camera fixed or handheld? Whole-frame motion over the first seconds of a video.

A tripod camera barely moves the picture (~5.7 px per playback second at the 1280 inference
size, 20261108); a handheld one moves it 56-74 px/s (20261101, ledger 2026-11-12). Frames
``step_s`` playback seconds apart are matched (ORB, cross-checked, RANSAC homography) and the
mean displacement of the four frame corners is taken; the median over the window, per second,
decides. A video that cannot be measured is treated as handheld, the cautious mode. Overrides
per video file name, then per batch folder, come first (spec: automatic, overridable).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml

from bxjudge import PROJECT_ROOT
from bxjudge.video import iter_frames

DEFAULT_CAMERA_MODE_CONFIG = PROJECT_ROOT / "config" / "camera_mode.yaml"

_INFER_LONG_EDGE = 1280  # thresholds are in inference px (thresholds.yaml infer_long_edge)
_ORB_FEATURES = 3000
_MIN_MATCHES = 30
_RANSAC_PX = 3.0
_SECTIONS = {
    "detection": {"window_s", "step_s", "min_pairs", "handheld_px_per_s"},
    "handheld": {"refind_s", "confirm_px", "max_unchecked_s"},
}
_OVERRIDE_VALUES = ("fixed", "handheld", "auto")


def _positive(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v) and v > 0


def load_camera_policy(path: Path = DEFAULT_CAMERA_MODE_CONFIG) -> dict:
    """The camera-mode policy; ``ValueError`` for a missing or extra key, a value that is not
    positive and finite (``min_pairs``: a positive integer), a step longer than the window,
    ``max_unchecked_s`` not longer than ``refind_s``, or an override other than
    ``fixed``/``handheld``/``auto``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {*_SECTIONS, "overrides"}:
        raise ValueError(f"{path}: camera_mode needs exactly detection, handheld, overrides")
    for section, names in _SECTIONS.items():
        block = raw[section]
        if not isinstance(block, dict) or set(block) != names:
            raise ValueError(f"{path}: camera_mode.{section} needs exactly {sorted(names)}")
        for k, v in block.items():
            ok = (type(v) is int and v > 0) if k == "min_pairs" else _positive(v)
            if not ok:
                raise ValueError(f"{path}: camera_mode.{section}.{k} must be positive")
    det, hand = raw["detection"], raw["handheld"]
    if det["step_s"] >= det["window_s"]:
        raise ValueError(f"{path}: camera_mode.detection.step_s must be shorter than window_s")
    if math.ceil(det["window_s"] / det["step_s"]) - 1 < det["min_pairs"]:
        raise ValueError(f"{path}: camera_mode.detection window_s/step_s gives fewer than "
                         "min_pairs frame pairs")
    if hand["max_unchecked_s"] <= hand["refind_s"]:
        raise ValueError(f"{path}: camera_mode.handheld.max_unchecked_s must exceed refind_s")
    ov = raw["overrides"]
    if not isinstance(ov, dict) or set(ov) != {"batches", "videos"}:
        raise ValueError(f"{path}: camera_mode.overrides needs exactly batches, videos")
    for kind in ("batches", "videos"):
        table = ov[kind]
        if not isinstance(table, dict) or not all(
                isinstance(k, str) and v in _OVERRIDE_VALUES for k, v in table.items()):
            raise ValueError(f"{path}: camera_mode.overrides.{kind} maps names to "
                             f"{'/'.join(_OVERRIDE_VALUES)}")
    return {"detection": dict(det), "handheld": dict(hand),
            "overrides": {"batches": dict(ov["batches"]), "videos": dict(ov["videos"])}}


@dataclass(frozen=True)
class CameraMotion:
    px_per_s: float | None  # median corner displacement per playback second; None: unmeasured
    pairs: int  # frame pairs that gave a displacement


@dataclass(frozen=True)
class CameraMode:
    mode: Literal["fixed", "handheld"]
    source: Literal["measured", "override", "unmeasured"]
    motion: CameraMotion | None


def _pair_displacement(orb, matcher, a: np.ndarray, b: np.ndarray) -> float | None:
    def gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    ka, da = orb.detectAndCompute(gray(a), None)
    kb, db = orb.detectAndCompute(gray(b), None)
    if da is None or db is None:
        return None
    matches = matcher.match(da, db)
    if len(matches) < _MIN_MATCHES:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    h, _ = cv2.findHomography(pa, pb, cv2.RANSAC, _RANSAC_PX)
    if h is None or not np.isfinite(h).all():
        return None
    fh, fw = a.shape[:2]
    corners = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]).reshape(-1, 1, 2)
    d = float(np.linalg.norm(cv2.perspectiveTransform(corners, h) - corners, axis=2).mean())
    # At the inference scale (long edge 1280), where the threshold lives.
    d *= _INFER_LONG_EDGE / max(fh, fw)
    return d if math.isfinite(d) else None


def measure_camera_motion(frames: list[np.ndarray], step_s: float) -> CameraMotion:
    """Median displacement between consecutive ``frames`` (``step_s`` playback seconds apart)
    per playback second, in inference px (long edge 1280) whatever the frame size. A pair with fewer than 30 cross-checked matches or no homography is
    skipped; with no pair left the motion is unmeasured (``px_per_s=None``)."""
    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    shifts = [d for a, b in zip(frames, frames[1:])
              if (d := _pair_displacement(orb, matcher, a, b)) is not None]
    if not shifts:
        return CameraMotion(px_per_s=None, pairs=0)
    return CameraMotion(px_per_s=float(np.median(shifts)) / step_s, pairs=len(shifts))


def decide_camera_mode(video: Path, batch: str, motion: CameraMotion | None,
                       policy: dict) -> CameraMode:
    """``overrides.videos[file name]``, then ``overrides.batches[batch]`` (``auto`` = no
    override); else handheld when the measured motion exceeds ``handheld_px_per_s``; fewer
    than ``min_pairs`` measured pairs: handheld, ``source="unmeasured"``."""
    ov = policy["overrides"]
    for choice in (ov["videos"].get(Path(video).name), ov["batches"].get(batch)):
        if choice in ("fixed", "handheld"):
            return CameraMode(choice, "override", motion)
    det = policy["detection"]
    if motion is None or motion.px_per_s is None or motion.pairs < det["min_pairs"]:
        return CameraMode("handheld", "unmeasured", motion)
    mode = "handheld" if motion.px_per_s > det["handheld_px_per_s"] else "fixed"
    return CameraMode(mode, "measured", motion)


def _step_frames(fps: float, policy: dict) -> int:
    return max(1, round(policy["detection"]["step_s"] * fps))


def sampled_step_s(fps: float, policy: dict) -> float:
    """The playback interval actually sampled: ``step_s`` rounded to whole frames."""
    return _step_frames(fps, policy) / fps


def sample_for_motion(video: Path, fps: float, policy: dict, *, long_edge: int,
                      hwaccel: str) -> list[np.ndarray]:
    """One frame every ``step_s`` playback seconds (``max(1, round(step_s * fps))`` frames,
    ruling T4) within the first ``window_s``."""
    det = policy["detection"]
    step = _step_frames(fps, policy)
    return [frame for idx, frame in iter_frames(Path(video), 0.0, det["window_s"],
                                                long_edge=long_edge, hwaccel=hwaccel)
            if idx % step == 0]


def measure_video_motion(video: Path, fps: float, policy: dict, *, long_edge: int,
                         hwaccel: str) -> CameraMotion:
    """``sample_for_motion`` then ``measure_camera_motion`` over the interval actually
    sampled (``sampled_step_s``), not the configured one."""
    frames = sample_for_motion(video, fps, policy, long_edge=long_edge, hwaccel=hwaccel)
    return measure_camera_motion(frames, sampled_step_s(fps, policy))
