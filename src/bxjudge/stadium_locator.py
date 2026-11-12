"""Where a calibration's stadium is in any frame, from the stadium keypoint model.

The keypoint fit (``stadium_finder.fit_from_keypoints``) places the stadium model in a frame
with all its checks (rail support, direction, each zone near the rail). A ``KeypointLocator``
knows where the model sits in the calibration's own frame (``model_to_calibration``), so one
fit in a new frame gives the homography from calibration px to that frame's px. Fixed cameras
use it to relocate after the rim match is lost; handheld cameras re-find the stadium with it
every ``refind_s`` (spec 2026-11-12-dynamic-calibration-design.md).

``keypoint_snapshot`` turns ``find_stadium``'s keypoint result into a ``keypoints``
calibration snapshot. Rail results are not stored: on the labelled frames the rail path
found 8 of 24 at 21 px against 24 of 24 at 6.6 px for keypoints (ledger 2026-11-12).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bxjudge.autocalib import Alignment, _decompose, _stadium_centroid, warp_calibration
from bxjudge.calibration_store import Snapshot, save_snapshot
from bxjudge.stadium_finder import (
    _AGREEMENT_ZONES,
    FoundStadium,
    StadiumFinderConfig,
    StadiumFit,
    StadiumModel,
    _alignment,
    _project,
    find_stadium,
    fit_from_keypoints,
    load_stadium_finder_config,
    load_stadium_model,
    rail_mask,
    zone_visibility,
)
from bxjudge.types import Calibration

_log = logging.getLogger(__name__)
_INFERENCE_ERRORS = (RuntimeError, ValueError, MemoryError, cv2.error)


@dataclass(frozen=True)
class StadiumTools:
    """The stadium model, its finder config and the keypoint ensemble, loaded once per run."""
    model: StadiumModel
    cfg: StadiumFinderConfig
    finder: object  # stadium_keypoints.KeypointFinder, or anything with predict/reference


def load_stadium_tools(device: str | int | None = None) -> StadiumTools:
    """Shipped model, config and keypoint ensemble (``device`` as for Ultralytics)."""
    from bxjudge.stadium_keypoints import KeypointFinder

    cfg = load_stadium_finder_config()
    return StadiumTools(model=load_stadium_model(), cfg=cfg,
                        finder=KeypointFinder.from_config(cfg, device=device))


@dataclass(frozen=True)
class LocatedStadium:
    homography: np.ndarray  # calibration px -> frame px, h22 = 1
    alignment: Alignment  # the same mapping, described at the stadium centroid (no bounds)
    fit: StadiumFit


def _finite_h(h) -> np.ndarray | None:
    h = np.asarray(h, np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or h[2, 2] == 0:
        return None
    h = h / h[2, 2]
    return h if np.isfinite(h).all() else None


def alignment_from_homography(c: Calibration, h, score: float) -> Alignment:
    """``h`` (calibration px -> frame px) as an ``Alignment``: shift of the stadium centroid
    over the frame width, rotation, scale and perspective at the centroid, measured as
    ``DynamicCalibration._compose`` does, without any bound."""
    h = np.asarray(h, np.float64)
    h = h / h[2, 2]
    centre = np.array([*_stadium_centroid(c), 1.0])
    moved = h @ centre
    shift = float(np.linalg.norm(moved[:2] / moved[2] - centre[:2])) / c.frame_size[0]
    rotation, scale, _, _, perspective = _decompose(h, centre[:2])
    return Alignment(tuple(tuple(float(v) for v in r) for r in h), float(score), shift,
                     rotation, scale, perspective)


def zone_disagreement_px(c: Calibration, h1, h2) -> float:
    """Largest per-zone mean vertex distance (frame px) between the judged zones
    (``_AGREEMENT_ZONES``) of ``c`` moved through ``h1`` and through ``h2``: each zone on its
    own, so one pocket off is not averaged away by the zones that agree. ``inf`` when a
    vertex is thrown off to infinity."""
    worst = 0.0
    for z in _AGREEMENT_ZONES:
        poly = np.array(getattr(c, z), np.float64).reshape(-1, 2)
        if len(poly) == 0:
            continue
        d = float(np.linalg.norm(_project(h1, poly) - _project(h2, poly), axis=1).mean())
        if not np.isfinite(d):
            return float("inf")
        worst = max(worst, d)
    return worst


class KeypointLocator:
    """Calibration ``c`` placed in any frame of its size by one keypoint fit.

    ``model_to_calibration`` maps stadium model px to calibration px. ``locate`` never raises
    for an inference failure (CUDA out of memory, a driver error): it returns ``None`` and
    says why in ``last_reason`` (``no_keypoints``, ``fit_rejected`` or
    ``keypoint_error: <type>: <message>``)."""

    def __init__(self, tools: StadiumTools, calibration: Calibration, model_to_calibration):
        m = _finite_h(model_to_calibration)
        if m is None:
            raise ValueError("model_to_calibration is not a finite homography")
        self.tools = tools
        self.c = calibration
        self.model_to_calibration = m
        self._calibration_to_model = np.linalg.inv(m)
        self.last_reason: str | None = None

    @classmethod
    def for_snapshot(cls, tools: StadiumTools, snap: Snapshot, calibration: Calibration,
                     reference_image: np.ndarray) -> "KeypointLocator | None":
        """Locator for ``calibration`` (``snap``'s calibration at the decoded frame size).
        A ``keypoints`` snapshot already knows where the model sits; any other is fitted once
        on ``reference_image`` (a frame of ``calibration``'s size whose view the calibration
        describes). ``None``, with a warning saying why, when that fit fails."""
        size = tuple(int(v) for v in calibration.frame_size)
        if snap.kind == "keypoints":
            sw, sh = snap.calibration.frame_size
            scale = np.diag([size[0] / sw, size[1] / sh, 1.0])
            return cls(tools, calibration, scale @ np.asarray(snap.homography, np.float64))
        if (reference_image.shape[1], reference_image.shape[0]) != size:
            raise ValueError("reference image size differs from the calibration")
        probe = cls(tools, calibration, np.eye(3))
        fit = probe._fit(reference_image)
        if fit is None:
            _log.warning("calibration %s: no keypoint locator (%s)", snap.id, probe.last_reason)
            return None
        return cls(tools, calibration, fit.homography)

    def _fit(self, image: np.ndarray) -> StadiumFit | None:
        try:
            pts = self.tools.finder.predict(image)
            if pts is None:
                self.last_reason = "no_keypoints"
                return None
            fit = fit_from_keypoints(pts, self.tools.finder.reference,
                                     rail_mask(image, self.tools.cfg), self.tools.model,
                                     self.tools.cfg)
        except _INFERENCE_ERRORS as e:
            self.last_reason = f"keypoint_error: {type(e).__name__}: {e}"
            return None
        if fit is None:
            self.last_reason = "fit_rejected"
            return None
        self.last_reason = None
        return fit

    def locate(self, image: np.ndarray) -> LocatedStadium | None:
        if (image.shape[1], image.shape[0]) != tuple(self.c.frame_size):
            raise ValueError("frame size differs from the calibration")
        fit = self._fit(image)
        if fit is None:
            return None
        h = _finite_h(np.asarray(fit.homography, np.float64) @ self._calibration_to_model)
        if h is None:
            self.last_reason = "fit_rejected"
            return None
        return LocatedStadium(h, alignment_from_homography(self.c, h,
                                                           min(fit.coverage, fit.precision)),
                              fit)


def _single_fit_stadium(video: Path, model: StadiumModel, idx: int, frame: np.ndarray,
                        fit: StadiumFit) -> FoundStadium:
    fh, fw = frame.shape[:2]
    cal = warp_calibration(model.calibration, _alignment(fit, model, (fw, fh)), (fw, fh),
                           f"found-{Path(video).stem}")
    return FoundStadium(calibration=cal, fit=fit, frame_index=int(idx), frame=frame,
                        zone_visibility=zone_visibility(cal), agreeing_frames=1,
                        method="keypoints_single")


def keypoint_snapshot(video: Path, out_dir: Path, tools: StadiumTools, autocalib, vetoed, *,
                      hwaccel: str = "cpu", allow_single: bool = False,
                      keypoint_errors: list[str] | None = None,
                      ) -> tuple[Snapshot | None, FoundStadium | None]:
    """``find_stadium`` with the keypoint ensemble, stored as a ``keypoints`` snapshot
    (reference image = the found frame, score = fit coverage, homography = model px ->
    that frame's px). Only a keypoint result is stored: a rail result comes back as
    ``(None, found)``. A vetoed id also gives ``(None, found)``.

    ``allow_single`` (handheld cameras only): when no two early keypoint fits agree, the one
    with the lowest keypoint error is used (``method="keypoints_single"``,
    ``agreeing_frames=1``). The handheld tracker must confirm it before emitting any zone.
    Keypoint inference errors are appended to ``keypoint_errors`` when given."""
    fits: list = []
    found = find_stadium(Path(video), tools.model, tools.cfg, hwaccel=hwaccel,
                         keypoints=tools.finder, keypoint_fits=fits,
                         keypoint_errors=keypoint_errors)
    if found is None or found.method != "keypoints":
        if not (allow_single and fits):
            return None, found
        idx, frame, fit = min(fits, key=lambda f: (f[2].keypoint_error_px, f[0]))
        found = _single_fit_stadium(video, tools.model, idx, frame, fit)
    snap = save_snapshot(out_dir, "keypoints", found.calibration, found.frame,
                         score=found.fit.coverage, homography=found.fit.homography,
                         source_frame=int(found.frame_index),
                         agreeing_frames=int(found.agreeing_frames), autocalib=autocalib)
    if snap.id in vetoed:
        return None, found
    return snap, found
