"""Per-frame stadium displacement: how far the rim / lid moved relative to the reference frame.

A launcher touching the stadium makes the (fixed-camera) stadium shake, so this series is the
only evidence for "the launcher touched the stadium". It measures only the rim and lid
(``stadium`` minus ``battle_area``, the same region ``calibration_similarity`` uses) so the
spinning tops inside the battle area cannot disturb it.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import CancelledError, ThreadPoolExecutor

import cv2
import numpy as np

from bxjudge.geometry import RIM_BLUR_SIGMA, RIM_MASK_MARGIN, rim_mask
from bxjudge.types import Calibration

# Implementation constants (alignment-quality tuning, not rule thresholds).
WORK_WIDTH = 640  # ECC runs on frames downscaled to at most this width; shift is scaled back
_ECC_ITERATIONS = 50
_ECC_EPS = 1e-5
_CROP_PAD = 16  # px at work size kept around the mask bounding box
# ECC releases the GIL: two workers took 214505 from 0.99x to 0.76x playback with
# identical values (more contend with geometry for the shared OpenCV pool).
RAW_MOTION_WORKERS = 2


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


def _measure(tpl: _Template, frame: np.ndarray, criteria: tuple) -> float:
    if tpl.empty:
        return math.nan
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            tpl.ref, tpl.prepare(frame), warp, cv2.MOTION_TRANSLATION, criteria, tpl.mask, 1
        )
        d = math.hypot(float(warp[0, 2]), float(warp[1, 2])) * tpl.scale
    except cv2.error:
        d = math.nan
    return d if math.isfinite(d) else math.nan


def iter_stadium_displacement(
    frames: Iterable[tuple[int, np.ndarray]],
    c: Calibration,
    ref: np.ndarray,
    *,
    workers: int = 1,
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

    With ``workers`` > 1 frames are measured concurrently, at most two per worker ahead
    of the consumer; order, frame objects and values are those of the serial path. A
    source failure is raised after the frames already read, or on close if it was read
    but not yet delivered (cancellation excepted); closing joins the workers.
    """
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    templates: dict[tuple[int, int], _Template] = {}
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, _ECC_ITERATIONS, _ECC_EPS)

    def template(frame: np.ndarray) -> _Template:
        key = (frame.shape[1], frame.shape[0])
        tpl = templates.get(key)
        if tpl is None:
            tpl = templates[key] = _Template(c, ref, key)
        return tpl

    if workers == 1:
        for idx, frame in frames:
            yield idx, frame, _measure(template(frame), frame, criteria)
        return
    pending = deque()
    failure = None
    pool = ThreadPoolExecutor(workers, thread_name_prefix="bxjudge-raw-motion")
    try:
        source = iter(frames)
        while True:
            try:
                idx, frame = next(source)
            except StopIteration:
                break
            except Exception as exc:  # noqa: BLE001 - delivered after the frames already read
                failure = exc
                break
            pending.append((idx, frame, pool.submit(_measure, template(frame), frame, criteria)))
            if len(pending) >= 2 * workers:
                idx, frame, measured = pending.popleft()
                yield idx, frame, measured.result()
        while pending:
            idx, frame, measured = pending.popleft()
            yield idx, frame, measured.result()
    except GeneratorExit:
        # A later stage may stop on the shared event this source's own failure set,
        # closing this stage before it delivered that failure (as video.prefetched).
        if failure is not None and not isinstance(failure, CancelledError):
            raise failure from None
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if failure is not None:
        raise failure


def stadium_displacement(frames: Iterable[tuple[int, np.ndarray]], c: Calibration, ref: np.ndarray) -> dict[int, float]:
    """Collect displacement measurements; the pipeline consumes the streaming API."""
    return {idx: displacement for idx, _, displacement in iter_stadium_displacement(frames, c, ref)}
