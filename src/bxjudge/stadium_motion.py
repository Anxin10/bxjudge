"""Per-frame stadium displacement: how far the rim / lid moved relative to the reference frame.

A launcher touching the stadium makes the (fixed-camera) stadium shake, so this series is the
only evidence for "the launcher touched the stadium". It measures only the rim and lid
(``stadium`` minus ``battle_area``, the same region ``calibration_similarity`` uses) so the
spinning tops inside the battle area cannot disturb it.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

import cv2
import numpy as np

from bxjudge.geometry import RIM_BLUR_SIGMA, RIM_MASK_MARGIN, rim_mask
from bxjudge.types import Calibration

# Implementation constants (alignment-quality tuning, not rule thresholds).
WORK_WIDTH = 640  # ECC runs on frames downscaled to at most this width; shift is scaled back
_ECC_ITERATIONS = 50
_ECC_EPS = 1e-5
_CROP_PAD = 16  # px at work size kept around the mask bounding box


class _Template:
    """Reference image, mask and crop box prepared once per frame size."""

    def __init__(self, c: Calibration, ref: np.ndarray, frame_wh: tuple[int, int]) -> None:
        fw, fh = frame_wh
        ww = min(fw, WORK_WIDTH)
        wh = max(1, round(fh * ww / fw))
        self.work_size = (ww, wh)
        self.scale = fw / ww  # work px -> frame px
        self.sigma = RIM_BLUR_SIGMA * ww / 320.0
        margin = max(1, round(RIM_MASK_MARGIN * ww / 320.0))
        mask = rim_mask(c, self.work_size, margin)
        ys, xs = np.nonzero(mask)
        self.empty = ys.size == 0
        if self.empty:
            return
        x0, x1 = max(0, xs.min() - _CROP_PAD), min(ww, xs.max() + 1 + _CROP_PAD)
        y0, y1 = max(0, ys.min() - _CROP_PAD), min(wh, ys.max() + 1 + _CROP_PAD)
        self.crop = (slice(y0, y1), slice(x0, x1))
        self.mask = np.ascontiguousarray(mask[self.crop].astype(np.uint8) * 255)
        self.ref = self.prepare(ref)

    def prepare(self, img: np.ndarray) -> np.ndarray:
        """Gray, work-size, lightly blurred float32, cropped to the mask's bounding box."""
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (g.shape[1], g.shape[0]) != self.work_size:
            g = cv2.resize(g, self.work_size, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (0, 0), self.sigma, borderType=cv2.BORDER_REPLICATE)
        return np.ascontiguousarray(g[self.crop].astype(np.float32))


def iter_stadium_displacement(
    frames: Iterable[tuple[int, np.ndarray]],
    c: Calibration,
    ref: np.ndarray,
) -> Iterator[tuple[int, np.ndarray, float]]:
    """Translation of the stadium rim relative to ``ref``, in pixels, per frame index.

    Region: ``stadium`` minus ``battle_area`` (``geometry.rim_mask``; the rim and lid edge
    features stay, the tops inside the battle area are excluded). Each frame and ``ref`` are
    converted to grayscale, lightly Gaussian-blurred and aligned with
    ``cv2.findTransformECC`` (``MOTION_TRANSLATION``, mask as ``inputMask``). The value is the
    length of the translation vector. Frames may be any size (they are scaled to the frame
    size consistently, ``ref`` and the calibration polygons likewise); the value is in pixels
    of the frame as given, not of ``c.frame_size``. ECC runs on a copy downscaled to at most
    ``WORK_WIDTH`` px wide for speed. When ECC does not converge the frame gets ``nan``.
    """
    templates: dict[tuple[int, int], _Template] = {}
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, _ECC_ITERATIONS, _ECC_EPS)
    for idx, frame in frames:
        key = (frame.shape[1], frame.shape[0])
        tpl = templates.get(key)
        if tpl is None:
            tpl = templates[key] = _Template(c, ref, key)
        if tpl.empty:
            yield idx, frame, math.nan
            continue
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                tpl.ref, tpl.prepare(frame), warp, cv2.MOTION_TRANSLATION, criteria, tpl.mask, 1
            )
            d = math.hypot(float(warp[0, 2]), float(warp[1, 2])) * tpl.scale
        except cv2.error:
            d = math.nan
        yield idx, frame, d if math.isfinite(d) else math.nan


def stadium_displacement(frames: Iterable[tuple[int, np.ndarray]], c: Calibration, ref: np.ndarray) -> dict[int, float]:
    """Collect displacement measurements; the pipeline consumes the streaming API."""
    return {idx: displacement for idx, _, displacement in iter_stadium_displacement(frames, c, ref)}
