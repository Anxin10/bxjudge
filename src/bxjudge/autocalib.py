"""Automatic calibration core: align a manual calibration template to a moved camera frame.

``estimate_alignment`` finds the homography (template px -> frame px) that maps a manual
calibration's reference image onto a frame of a slightly moved camera, using only
polarity- and colour-free edge maps (blurred Sobel gradient magnitude, as
``calibration.calibration_similarity`` does) inside the template's rim region, so a white and
a black battle floor align the same way (spec §6.2). Estimates outside the configured bounds
or with a low edge-map score are rejected (``None``) rather than accepted misaligned.
``warp_calibration`` moves all 8 zones through the homography; ``mean_boundary_error``
measures how far two calibrations' zone boundaries are apart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np
import yaml

from bxjudge.calibration import ZONES, _edge_map, rim_mask, scale_calibration
from bxjudge.config import config_loader
from bxjudge.geometry import RIM_MASK_MARGIN
from bxjudge.types import Calibration, Polygon

_POLY_FIELDS = [z[0] for z in ZONES]

# Implementation constants (not policy). Pyramid: coarse similarity search at work/4, ECC
# refinement at work/2 then at work width.
_COARSE_DIV = 4
_REFINE_DIVS = (2, 1)
_COARSE_STEP_PX = 2.0  # rotation/scale grid step: <= this rim displacement at the coarse level
_COARSE_BLUR = 1.0  # extra blur of coarse edge maps (px at coarse level) to widen the basin
_NORM_CLIP = 4.0  # normalised edge values are clipped to +-this many std (contrast balance)
_ECC_GAUSS = 5  # findTransformECC gaussFiltSize
_MIN_LOCAL_VAR = 0.05  # coarse NCC: ignore windows whose masked frame variance is below this


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class AutoCalibConfig:
    work_width: int
    min_score: float
    max_shift_frac: float
    max_rotation_deg: float
    min_scale: float
    max_scale: float
    max_perspective: float
    ecc_iterations: int
    ecc_epsilon: float
    max_boundary_error_px: float
    auto_review_score: float
    snapshot_max_frame_side: int  # snapshot file bounds (C-X2(f), R40)
    snapshot_max_coord: float


_INT_FIELDS = ("work_width", "ecc_iterations", "snapshot_max_frame_side")


@config_loader
def load_autocalib_config(path: Path = Path("config/autocalib.yaml")) -> AutoCalibConfig:
    """Load the auto-calibration policy. Missing/unknown keys, wrong types, non-finite numbers
    or out-of-range values raise ``ValueError``."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a mapping of setting names to values")
    names = [f.name for f in fields(AutoCalibConfig)]
    unknown = sorted(set(raw) - set(names))
    missing = sorted(set(names) - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    values: dict = {}
    for name in names:
        v = raw[name]
        if name in _INT_FIELDS:
            if type(v) is not int or v <= 0:
                raise ValueError(f"{path}: {name} must be a positive int, got {v!r}")
        else:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{path}: {name} must be a finite number, got {v!r}")
            v = float(v)
        values[name] = v
    c = AutoCalibConfig(**values)
    if c.work_width < 160:
        raise ValueError(f"{path}: work_width must be >= 160, got {c.work_width}")
    if not 0.0 <= c.min_score <= 1.0:
        raise ValueError(f"{path}: min_score must be in [0, 1], got {c.min_score}")
    if not 0.0 < c.max_shift_frac <= 1.0:
        raise ValueError(f"{path}: max_shift_frac must be in (0, 1], got {c.max_shift_frac}")
    if not 0.0 <= c.max_rotation_deg < 90.0:
        raise ValueError(f"{path}: max_rotation_deg must be in [0, 90), got {c.max_rotation_deg}")
    if not 0.0 < c.min_scale < c.max_scale:
        raise ValueError(f"{path}: need 0 < min_scale < max_scale, got {c.min_scale}, {c.max_scale}")
    if c.max_perspective < 0.0:
        raise ValueError(f"{path}: max_perspective must be >= 0, got {c.max_perspective}")
    if c.ecc_epsilon <= 0.0:
        raise ValueError(f"{path}: ecc_epsilon must be > 0, got {c.ecc_epsilon}")
    if c.max_boundary_error_px <= 0.0:
        raise ValueError(f"{path}: max_boundary_error_px must be > 0, got {c.max_boundary_error_px}")
    if not c.min_score <= c.auto_review_score <= 1.0:
        raise ValueError(f"{path}: auto_review_score must be in [min_score, 1] = "
                         f"[{c.min_score}, 1], got {c.auto_review_score}")
    if c.snapshot_max_frame_side < c.work_width:
        raise ValueError(f"{path}: snapshot_max_frame_side must be >= work_width "
                         f"({c.work_width}), got {c.snapshot_max_frame_side}")
    if c.snapshot_max_coord < c.snapshot_max_frame_side:
        raise ValueError(f"{path}: snapshot_max_coord must be >= snapshot_max_frame_side "
                         f"({c.snapshot_max_frame_side}), got {c.snapshot_max_coord}")
    return c


# --------------------------------------------------------------------------- alignment
@dataclass(frozen=True)
class Alignment:
    homography: tuple[tuple[float, float, float], ...]  # 3x3 template-px -> frame-px, h22 = 1
    score: float  # 0-1 edge-map NCC over the warped rim mask
    shift_frac: float  # stadium-centroid movement / frame width
    rotation_deg: float  # counter-clockwise on screen is positive
    scale: float  # sqrt(det) of the local linear part at the stadium centroid
    perspective: float  # max(|h20|, |h21|) in frame px (template rescaled to the frame size)


def _gray(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def _normalise(edge: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Zero mean / unit std inside ``mask``, clipped so no single strong edge dominates."""
    v = edge[mask]
    if v.size < 16:
        return None
    sd = float(v.std())
    if not sd > 1e-6:
        return None
    return np.clip((edge - float(v.mean())) / sd, -_NORM_CLIP, _NORM_CLIP).astype(np.float32)


def _region_mask(template: Calibration, size: tuple[int, int]) -> np.ndarray:
    """Template rim region slightly dilated so the stadium and floor boundaries are included."""
    k = max(1, round(RIM_MASK_MARGIN * size[0] / 320))
    m = rim_mask(template, size, 0).astype(np.uint8)
    return cv2.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0


def _scale_h(h: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Express homography ``h`` in coordinates scaled by (fx, fy) on both sides."""
    d = np.diag([fx, fy, 1.0])
    return d @ h @ np.linalg.inv(d)


def _stadium_centroid(template: Calibration) -> tuple[float, float]:
    pts = np.array(template.stadium, np.float32).reshape(-1, 1, 2)
    m = cv2.moments(pts)
    if abs(m["m00"]) < 1e-9:
        p = pts.reshape(-1, 2).mean(axis=0)
        return float(p[0]), float(p[1])
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def _masked_ncc_map(region: np.ndarray, patch: np.ndarray, pmask: np.ndarray) -> np.ndarray:
    """Zero-mean NCC of ``patch`` (inside ``pmask``) at every position in ``region``."""
    m = pmask.astype(np.float32)
    n = float(m.sum())
    t = np.where(pmask, patch - float(patch[pmask].mean()), 0.0).astype(np.float32)
    tt = float((t * t).sum())
    num = cv2.matchTemplate(region, t, cv2.TM_CCORR)
    s1 = cv2.matchTemplate(region, m, cv2.TM_CCORR)
    s2 = cv2.matchTemplate(region * region, m, cv2.TM_CCORR)
    var = np.maximum(s2 - s1 * s1 / n, 0.0)
    ok = var > _MIN_LOCAL_VAR * n
    out = np.zeros_like(num)
    out[ok] = num[ok] / np.sqrt(tt * var[ok])
    return out


def _coarse_search(tmap: np.ndarray, tmask: np.ndarray, fmap: np.ndarray,
                   centre: tuple[float, float], cfg: AutoCalibConfig) -> np.ndarray | None:
    """Best similarity (template -> frame, coarse px) over a rotation x scale x shift grid."""
    h, w = fmap.shape
    r = math.ceil(cfg.max_shift_frac * w)
    ys, xs = np.nonzero(tmask)
    if xs.size == 0:
        return None
    radius = float(np.sqrt((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2).max())
    if radius < 1.0:
        return None
    step = _COARSE_STEP_PX / radius
    n_rot = math.ceil(2 * math.radians(cfg.max_rotation_deg) / step) + 1
    n_scl = math.ceil(math.log(cfg.max_scale / cfg.min_scale) / step) + 1
    rots = np.linspace(-cfg.max_rotation_deg, cfg.max_rotation_deg, n_rot)
    scls = np.exp(np.linspace(math.log(cfg.min_scale), math.log(cfg.max_scale), n_scl))
    pad = 2 * r
    fpad = cv2.copyMakeBorder(fmap, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0.0)
    canvas = (w + 2 * r, h + 2 * r)  # template warped into frame coords + r on each side
    tm = tmask.astype(np.float32)
    best_score, best = -np.inf, None
    for rot in rots:
        for s in scls:
            a = cv2.getRotationMatrix2D(centre, float(rot), float(s))
            a[:, 2] += r
            wt = cv2.warpAffine(tmap * tm, a, canvas, flags=cv2.INTER_LINEAR)
            wm = cv2.warpAffine(tm, a, canvas, flags=cv2.INTER_LINEAR) > 0.5
            yy, xx = np.nonzero(wm)
            if xx.size < 16:
                continue
            x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
            patch, pmask = wt[y0:y1, x0:x1], wm[y0:y1, x0:x1]
            # canvas (cx, cy) is frame (cx - r, cy - r) = fpad (cx + r, cy + r); search +-r.
            region = fpad[y0:y1 + 2 * r, x0:x1 + 2 * r]
            ncc = _masked_ncc_map(region, patch, pmask)
            _, score, _, loc = cv2.minMaxLoc(ncc)
            if score > best_score:
                dx, dy = loc[0] - r, loc[1] - r
                m = np.vstack([cv2.getRotationMatrix2D(centre, float(rot), float(s)), [0, 0, 1]])
                m[0, 2] += dx
                m[1, 2] += dy
                best_score, best = score, m
    return best


def _decompose(hn: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float, float]:
    """(rotation_deg, scale, smin, smax, perspective) of frame-px homography ``hn`` at ``p``."""
    a, hv = hn[:2, :2], hn[2, :2]
    wv = float(hv @ p + hn[2, 2])
    q = (a @ p + hn[:2, 2]) / wv
    jac = (a - np.outer(q, hv)) / wv
    u, sv, vt = np.linalg.svd(jac)
    rot = u @ vt
    det = float(np.linalg.det(jac))
    rotation = math.degrees(math.atan2(-rot[1, 0], rot[0, 0]))
    scale = math.sqrt(det) if det > 0 else float("nan")
    persp = float(max(abs(hn[2, 0]), abs(hn[2, 1])))
    return rotation, scale, float(sv.min()), float(sv.max()), persp if det > 0 else float("inf")


def _edge_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 16:
        return 0.0
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    den = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if den < 1e-9:
        return 0.0
    return float(np.clip((x * y).sum() / den, 0.0, 1.0))


def estimate_alignment(frame: np.ndarray, template: Calibration, template_image: np.ndarray,
                       cfg: AutoCalibConfig) -> Alignment | None:
    """Homography mapping ``template`` (manual calibration, reference image ``template_image``)
    onto ``frame``, or ``None`` when no in-bounds alignment with score >= ``min_score`` exists.

    Both images are resized to ``work_width`` (and /2, /4) and turned into edge maps
    (``calibration._edge_map``: blurred gradient magnitude, blind to brightness polarity and
    colour), each normalised to zero mean / unit std inside the template's rim mask dilated
    slightly (so the stadium and floor boundaries count). A coarse grid search over rotation,
    scale and shift within the configured bounds (masked zero-mean NCC at work/4) gives a
    similarity transform; ``cv2.findTransformECC`` (homography) refines it at work/2 and at
    work width. Any OpenCV failure, out-of-bounds estimate or low score returns ``None``."""
    fh, fw = frame.shape[:2]
    tw, th = template.frame_size
    if fw < 1 or fh < 1 or tw < 1 or th < 1:
        return None
    full = (cfg.work_width, max(1, round(cfg.work_width * fh / fw)))

    def level(div: int):
        size = (max(1, round(full[0] / div)), max(1, round(full[1] / div)))
        mask = _region_mask(template, size)
        te = _edge_map(_gray(template_image, size))
        fe = _edge_map(_gray(frame, size))
        if div == _COARSE_DIV:
            te = cv2.GaussianBlur(te, (0, 0), _COARSE_BLUR)
            fe = cv2.GaussianBlur(fe, (0, 0), _COARSE_BLUR)
        return size, mask, _normalise(te, mask), _normalise(fe, mask)

    try:
        size, mask, tmap, fmap = level(_COARSE_DIV)
        if tmap is None or fmap is None:
            return None
        cx, cy = _stadium_centroid(template)
        centre = (cx * size[0] / tw, cy * size[1] / th)
        h = _coarse_search(tmap, mask, fmap, centre, cfg)
        if h is None:
            return None
        prev = size
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, cfg.ecc_iterations, cfg.ecc_epsilon)
        for div in _REFINE_DIVS:
            size, mask, tmap, fmap = level(div)
            if tmap is None or fmap is None:
                return None
            h = _scale_h(h, size[0] / prev[0], size[1] / prev[1])
            prev = size
            # Roles swapped so the template's rim mask can be passed as ECC's inputMask
            # (input-image domain): ECC estimates the frame -> template warp.
            warp = np.linalg.inv(h).astype(np.float32)
            _, warp = cv2.findTransformECC(fmap, tmap, warp, cv2.MOTION_HOMOGRAPHY, crit,
                                           mask.astype(np.uint8) * 255, _ECC_GAUSS)
            h = np.linalg.inv(warp.astype(np.float64))
    except (cv2.error, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-12:
        return None
    h_work = h / h[2, 2]

    # Score: edge NCC of the warped template vs the frame over the warped rim mask.
    rim = rim_mask(template, full, max(1, round(RIM_MASK_MARGIN * full[0] / 320)))
    te = _edge_map(_gray(template_image, full))
    fe = _edge_map(_gray(frame, full))
    wte = cv2.warpPerspective(te, h_work, full, flags=cv2.INTER_LINEAR)
    wrim = cv2.warpPerspective(rim.astype(np.uint8), h_work, full, flags=cv2.INTER_NEAREST) > 0
    score = _edge_ncc(wte, fe, wrim)

    # Back to full resolution: template px -> work px -> frame px.
    s_t = np.diag([full[0] / tw, full[1] / th, 1.0])
    s_f = np.diag([full[0] / fw, full[1] / fh, 1.0])
    hfull = np.linalg.inv(s_f) @ h_work @ s_t
    hfull /= hfull[2, 2]
    # Motion in frame px: the template rescaled to the frame size, then moved by hn.
    s_tf = np.diag([fw / tw, fh / th, 1.0])
    hn = hfull @ np.linalg.inv(s_tf)
    hn /= hn[2, 2]
    c0 = s_tf[:2, :2] @ np.array([cx, cy])
    moved = hn @ np.array([c0[0], c0[1], 1.0])
    if abs(moved[2]) < 1e-12:
        return None
    shift_frac = float(np.hypot(*(moved[:2] / moved[2] - c0))) / fw
    rotation, scale, smin, smax, persp = _decompose(hn, c0)
    vals = (score, shift_frac, rotation, scale, smin, smax, persp)
    if not all(math.isfinite(v) for v in vals):
        return None
    if (shift_frac > cfg.max_shift_frac or abs(rotation) > cfg.max_rotation_deg
            or smin < cfg.min_scale or smax > cfg.max_scale or persp > cfg.max_perspective
            or score < cfg.min_score):
        return None
    return Alignment(
        homography=tuple(tuple(float(v) for v in row) for row in hfull),
        score=score, shift_frac=shift_frac, rotation_deg=rotation, scale=scale,
        perspective=persp,
    )


def warp_calibration(template: Calibration, alignment: Alignment, frame_size: tuple[int, int],
                     name: str) -> Calibration:
    """``template`` with all 8 zone polygons moved through ``alignment.homography``.
    ``frame_size`` is the (width, height) of the frame the alignment was estimated on;
    ``ref_frame`` is left empty (the caller stores the frame's own reference image)."""
    h = np.array(alignment.homography, np.float64)

    def tf(poly: Polygon) -> Polygon:
        if not poly:
            return []
        pts = cv2.perspectiveTransform(np.array(poly, np.float64).reshape(-1, 1, 2), h)
        return [(float(x), float(y)) for x, y in pts.reshape(-1, 2)]

    return Calibration(name=name, frame_size=(int(frame_size[0]), int(frame_size[1])),
                       ref_frame="", **{f: tf(getattr(template, f)) for f in _POLY_FIELDS})


# --------------------------------------------------------------------------- boundary error
def _resample(poly: Polygon, n: int) -> np.ndarray:
    """``n`` points equally spaced by arc length along the closed polygon (as complex)."""
    p = np.array(poly, np.float64).reshape(-1, 2)
    seg = np.roll(p, -1, axis=0) - p
    lens = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    total = cum[-1]
    if total < 1e-9:
        q = np.repeat(p[:1], n, axis=0)
    else:
        t = np.arange(n) * total / n
        i = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(p) - 1)
        frac = np.where(lens[i] > 0, (t - cum[i]) / np.where(lens[i] > 0, lens[i], 1.0), 0.0)
        q = p[i] + frac[:, None] * seg[i]
    return q[:, 0] + 1j * q[:, 1]


def _perimeter(poly: Polygon) -> float:
    p = np.array(poly, np.float64).reshape(-1, 2)
    seg = np.roll(p, -1, axis=0) - p
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def _boundary_distance(a: Polygon, b: Polygon) -> float:
    """Mean distance between corresponding points of two closed boundaries.

    Both are resampled to the same number of points (about 1 px apart) by arc length;
    correspondence is the cyclic offset and orientation minimising the summed squared
    distance (FFT circular correlation), refined on the mean distance itself. Vertex count,
    starting vertex and click direction therefore do not matter; the result is symmetric."""
    if not a or not b:
        return 0.0 if not a and not b else float("inf")
    n = int(min(4096, max(64, math.ceil(max(_perimeter(a), _perimeter(b))))))
    za = _resample(a, n)
    best = float("inf")
    for zb in (_resample(b, n), _resample(b, n)[::-1]):
        corr = np.fft.ifft(np.conj(np.fft.fft(za)) * np.fft.fft(zb)).real
        s0 = int(np.argmax(corr))
        for s in range(s0 - 2, s0 + 3):
            best = min(best, float(np.abs(za - np.roll(zb, -s)).mean()))
    return best


def mean_boundary_error(a: Calibration, b: Calibration, long_edge: int) -> float:
    """Mean over the 8 zones of the symmetric corresponding-point boundary distance
    (``_boundary_distance``), with each calibration first scaled so its long edge is
    ``long_edge`` px (inference px when ``long_edge`` = ``infer_long_edge``)."""

    def to_long(c: Calibration) -> Calibration:
        w, h = c.frame_size
        k = long_edge / max(w, h)
        return scale_calibration(c, (round(w * k), round(h * k)))

    sa, sb = to_long(a), to_long(b)
    return float(np.mean([_boundary_distance(getattr(sa, f), getattr(sb, f)) for f in _POLY_FIELDS]))
