"""Find the stadium in a frame from its green rail, without a per-scene template.

Every video shows the bright green rail around the battle floor. Its shape is fixed: an
oval whose near end has a notch (the "cat ears"), which also fixes the stadium's direction.
A canonical model (rail points plus the 8 zones of one manual calibration) is fitted to the
rail found in a frame through a homography; the zones then follow the same homography.
Unreliable fits return ``None`` so the caller marks the video ``recalibrate``.

X/B (user ruling 2026-11-11): with the cat ears facing the viewer, the stadium's left is X
and its right is B. The model template is upright, so its left zones stay X after warping.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np
import yaml

from bxjudge import PROJECT_ROOT
from bxjudge.calibration import ZONES
from bxjudge.config import config_loader
from bxjudge.geometry import polygon_mask
from bxjudge.types import Calibration, Polygon

DEFAULT_STADIUM_FINDER_CONFIG = PROJECT_ROOT / "config" / "stadium_finder.yaml"
DEFAULT_STADIUM_MODEL = PROJECT_ROOT / "config" / "stadium_model.json"

_POLY_FIELDS = [z[0] for z in ZONES]
_MODEL_SCHEMA = "bxjudge-stadium-model-v1"


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class StadiumFinderConfig:
    model_template: str
    model_points: int
    roi_margin_frac: float
    hue_search: tuple[int, int]
    hue_halfwidth: int
    min_saturation: int
    min_value: int
    max_thickness_frac: float
    min_elongation: float
    min_component_frac: float
    sample_window_s: float
    sample_step_s: float
    candidate_frames: int
    work_width: int
    coarse_rotation_step_deg: float
    coarse_scale_min: float
    coarse_scale_max: float
    coarse_scale_steps: int
    coarse_top_k: int
    refine_iterations: int
    inlier_px: float
    min_coverage: float
    min_precision: float
    min_rail_in_frame: float
    notch_halfwidth_frac: float
    min_notch_coverage: float
    min_quadrant_coverage: float
    orientation_min_gain: float
    max_perspective: float
    perspective_min_gain_px: float
    agreement_px: float
    zone_visible_min: float


_INT_FIELDS = ("model_points", "hue_halfwidth", "min_saturation", "min_value",
               "candidate_frames", "work_width", "coarse_scale_steps", "coarse_top_k",
               "refine_iterations")
_FRACTIONS = ("roi_margin_frac", "max_thickness_frac", "min_component_frac", "min_coverage",
              "min_precision", "min_rail_in_frame", "notch_halfwidth_frac", "min_notch_coverage",
              "min_quadrant_coverage", "orientation_min_gain", "zone_visible_min")


@config_loader
def load_stadium_finder_config(path: Path = DEFAULT_STADIUM_FINDER_CONFIG) -> StadiumFinderConfig:
    """Load the stadium-finder policy. Missing/unknown keys, wrong types, non-finite numbers
    or out-of-range values raise ``ValueError``."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a mapping of setting names to values")
    names = [f.name for f in fields(StadiumFinderConfig)]
    unknown = sorted(set(raw) - set(names))
    missing = sorted(set(names) - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    values: dict = {}
    for name in names:
        v = raw[name]
        if name == "model_template":
            if not isinstance(v, str) or not v or "/" in v or "\\" in v:
                raise ValueError(f"{path}: model_template must be a calibration name, got {v!r}")
        elif name == "hue_search":
            if (not isinstance(v, list) or len(v) != 2 or any(type(h) is not int for h in v)
                    or not 0 <= v[0] < v[1] <= 179):
                raise ValueError(f"{path}: hue_search must be [low, high] with 0 <= low < high "
                                 f"<= 179, got {v!r}")
            v = (v[0], v[1])
        elif name in _INT_FIELDS:
            if type(v) is not int or v <= 0:
                raise ValueError(f"{path}: {name} must be a positive int, got {v!r}")
        else:
            if (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    or v <= 0):
                raise ValueError(f"{path}: {name} must be a positive finite number, got {v!r}")
            v = float(v)
        values[name] = v
    c = StadiumFinderConfig(**values)
    for name in _FRACTIONS:
        if not 0.0 < getattr(c, name) <= 1.0:
            raise ValueError(f"{path}: {name} must be in (0, 1], got {getattr(c, name)}")
    if c.min_saturation > 255 or c.min_value > 255 or c.hue_halfwidth > 90:
        raise ValueError(f"{path}: min_saturation/min_value must be <= 255, hue_halfwidth <= 90")
    if not c.coarse_scale_min < c.coarse_scale_max:
        raise ValueError(f"{path}: need coarse_scale_min < coarse_scale_max")
    if c.coarse_rotation_step_deg > 90.0:
        raise ValueError(f"{path}: coarse_rotation_step_deg must be <= 90")
    if c.work_width < 160:
        raise ValueError(f"{path}: work_width must be >= 160, got {c.work_width}")
    return c


# --------------------------------------------------------------------------- rail mask
def rail_mask(frame: np.ndarray, cfg: StadiumFinderConfig,
              roi: np.ndarray | None = None) -> np.ndarray:
    """uint8 0/255 mask (frame-sized) of thin green rail pixels.

    The accepted hue band is centred on this frame's own green peak inside ``hue_search``, so
    the rail is found under different lighting and cameras. A component is kept only when it
    is a band: mean thickness (2 * area / perimeter) at most ``max_thickness_frac`` of the long
    edge and length / thickness (perimeter^2 / 4 area) at least ``min_elongation``. Solid
    blobs such as gloves and specks are dropped."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = cfg.hue_search
    cand = (hue >= lo) & (hue <= hi) & (sat >= cfg.min_saturation) & (val >= cfg.min_value)
    if roi is not None:
        cand &= roi > 0
    out = np.zeros((h, w), np.uint8)
    if not cand.any():
        return out
    hist = np.bincount(hue[cand].ravel(), minlength=180).astype(np.float64)
    hist = np.convolve(hist, np.ones(5) / 5.0, mode="same")
    peak = int(np.argmax(hist))
    band = cand & (np.abs(hue.astype(np.int16) - peak) <= cfg.hue_halfwidth)
    mask = cv2.morphologyEx(band.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    long_edge = max(h, w)
    min_area = cfg.min_component_frac * h * w
    max_thickness = cfg.max_thickness_frac * long_edge
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        if (perimeter <= 0 or 2.0 * area / perimeter > max_thickness
                or perimeter * perimeter / (4.0 * area) < cfg.min_elongation):
            continue
        out[labels == i] = 255
    return out


# --------------------------------------------------------------------------- model
@dataclass(frozen=True)
class StadiumModel:
    rail: np.ndarray  # (N, 2) float64, template px
    calibration: Calibration  # the template's 8 zones, template px
    source_sha256: str
    image_size: tuple[int, int]  # (width, height) of the template frame

    def __eq__(self, other):  # ndarray field: compare by value
        return (isinstance(other, StadiumModel) and np.array_equal(self.rail, other.rail)
                and self.calibration == other.calibration
                and self.source_sha256 == other.source_sha256
                and self.image_size == other.image_size)


def _calibration_doc(c: Calibration) -> dict:
    d = asdict(c)
    d["frame_size"] = list(c.frame_size)
    for f in _POLY_FIELDS:
        d[f] = [[float(x), float(y)] for x, y in getattr(c, f)]
    d["ref_frame"] = ""
    return d


def _calibration_from_doc(d: dict) -> Calibration:
    polys: dict[str, Polygon] = {f: [(float(x), float(y)) for x, y in d[f]] for f in _POLY_FIELDS}
    w, h = d["frame_size"]
    return Calibration(name=d["name"], frame_size=(int(w), int(h)), ref_frame="", **polys)


def build_stadium_model(template: Calibration, image: np.ndarray,
                        cfg: StadiumFinderConfig) -> StadiumModel:
    """Canonical model from one manual calibration and its reference image.

    The rail is extracted inside the template's battle area (grown by ``roi_margin_frac``) and
    ``cfg.model_points`` rail pixels are kept, evenly spread by angle around the rail's centre.
    Raises ``ValueError`` when the template shows too little rail."""
    h, w = image.shape[:2]
    if (w, h) != tuple(template.frame_size):
        raise ValueError(f"template image {w}x{h} does not match frame_size {template.frame_size}")
    roi = polygon_mask(template.battle_area, (w, h))
    grow = max(1, round(cfg.roi_margin_frac * w))
    roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    mask = rail_mask(image, cfg, roi)
    ys, xs = np.nonzero(_ridge(mask))  # centre line: fits are scored against frame centre lines
    if len(xs) < cfg.model_points:
        raise ValueError(f"template shows {len(xs)} rail pixels, need >= {cfg.model_points}")
    pts = np.stack([xs, ys], 1).astype(np.float64)
    centre = pts.mean(0)
    order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]), kind="stable")
    pick = order[np.linspace(0, len(order) - 1, cfg.model_points).round().astype(int)]
    rail = pts[pick]
    digest = hashlib.sha256()
    digest.update(json.dumps(_calibration_doc(template), sort_keys=True).encode())
    digest.update(np.ascontiguousarray(image).tobytes())
    cal = _calibration_from_doc(_calibration_doc(template))
    return StadiumModel(rail=rail, calibration=cal, source_sha256=digest.hexdigest(),
                        image_size=(int(w), int(h)))


# --------------------------------------------------------------------------- fitting
# Implementation constants (not policy).
_REF_LONG_EDGE = 1280  # inference px: refinement and scoring run at this long edge
_COARSE_BLUR = 1.5  # px at work width: widens the coarse basin
_ECC_BLUR = 3.0  # px at reference width: both rails are blurred into comparable bands
_DOT_RADIUS = 3  # px at reference width: model points drawn as discs before blurring
_ECC_EPS = 1e-6


@dataclass(frozen=True)
class StadiumFit:
    homography: tuple[tuple[float, float, float], ...]  # model (template) px -> frame px
    coverage: float  # in-frame model rail points within inlier_px of rail evidence
    precision: float  # rail evidence within reach of the fitted model rail
    mean_error_px: float  # mean distance (inference px) of in-frame model points to evidence
    rail_in_frame: float  # share of model rail points that land inside the frame
    # Matched share of the model's cat-ear (notch) points and of its weakest quadrant
    # (left/right x near/far), points outside the frame counting as unmatched: the direction
    # and every side must be seen, or the zones are not determined.
    notch_coverage: float = 0.0
    quadrant_coverage: float = 0.0


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), h).reshape(-1, 2)


def _blurred(mask: np.ndarray, size: tuple[int, int], sigma: float) -> np.ndarray:
    small = cv2.resize(mask, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    small = (small > 0).astype(np.float32)
    return cv2.GaussianBlur(small, (0, 0), sigma)


def _coarse_candidates(mask: np.ndarray, model: StadiumModel,
                       cfg: StadiumFinderConfig) -> list[tuple[float, np.ndarray]]:
    """Top similarity transforms (score, 3x3 model px -> frame px) by rail overlap."""
    fh, fw = mask.shape[:2]
    sf = cfg.work_width / fw
    ww, wh = cfg.work_width, max(1, round(fh * sf))
    ev = _blurred(mask, (ww, wh), _COARSE_BLUR)
    ev /= max(float(ev.max()), 1e-6)
    pad = ww // 2
    ev_pad = cv2.copyMakeBorder(ev, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    centre = model.rail.mean(0)
    base = cfg.work_width / model.image_size[0]
    rel = model.rail - centre
    found: list[tuple[float, float, float, np.ndarray]] = []
    thetas = np.arange(0.0, 360.0, cfg.coarse_rotation_step_deg)
    scales = np.geomspace(cfg.coarse_scale_min, cfg.coarse_scale_max, cfg.coarse_scale_steps)
    for deg in thetas:
        r = _rot(math.radians(deg))
        for sc in scales:
            pts = (rel @ r.T) * (sc * base)
            lo = np.floor(pts.min(0)).astype(int)
            hi = np.ceil(pts.max(0)).astype(int)
            tw, th = hi - lo + 1
            if tw >= ev_pad.shape[1] or th >= ev_pad.shape[0]:
                continue
            tpl = np.zeros((th, tw), np.float32)
            ij = np.round(pts - lo).astype(int)
            np.add.at(tpl, (ij[:, 1], ij[:, 0]), 1.0)
            res = cv2.matchTemplate(ev_pad, tpl, cv2.TM_CCORR)
            _, best, _, loc = cv2.minMaxLoc(res)
            score = best / len(pts)
            # model centre lands at (loc - lo) in padded work px
            cx = (loc[0] - lo[0] - pad) / sf
            cy = (loc[1] - lo[1] - pad) / sf
            a = r * (sc * base / sf)
            h = np.eye(3)
            h[:2, :2] = a
            h[:2, 2] = np.array([cx, cy]) - a @ centre
            found.append((score, deg, sc, h))
    found.sort(key=lambda t: -t[0])
    picked: list[tuple[float, np.ndarray]] = []
    for score, deg, sc, h in found:  # keep distinct directions/scales only
        if all(abs((deg - d + 180) % 360 - 180) > 2 * cfg.coarse_rotation_step_deg
               or abs(math.log(sc / s)) > 0.15 for _, d, s, _ in picked):
            picked.append((score, deg, sc, h))
        if len(picked) >= cfg.coarse_top_k:
            break
    return [(score, h) for score, _, _, h in picked]


def _ridge(mask_ref: np.ndarray) -> np.ndarray:
    """Centre line of the rail bands (bool): local maxima of the distance to the band edge, so
    a thick close-up rail and a thin distant one weigh the same."""
    dist = cv2.distanceTransform((mask_ref > 0).astype(np.uint8), cv2.DIST_L2, 5)
    peak = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    return (dist > 0) & (dist >= peak)


def _refine(mask_ref: np.ndarray, ref_scale: float, model: StadiumModel, h0: np.ndarray,
            cfg: StadiumFinderConfig, motion: int = cv2.MOTION_AFFINE) -> np.ndarray | None:
    """ECC-refined homography (model px -> frame px) or None when ECC fails.

    Two passes: the first aligns the whole model rail to all rail evidence; the second keeps
    only model points that found evidence and evidence within reach of the fitted rail, so an
    occluded stretch or a green finger next to the rail cannot pull the fit."""
    h = _ecc(mask_ref, ref_scale, model, h0, model.rail, cfg, motion)
    if h is None:
        return None
    pts = _project(h, model.rail) * ref_scale
    rh, rw = mask_ref.shape[:2]
    evidence = mask_ref > 0
    dt = cv2.distanceTransform((~evidence).astype(np.uint8), cv2.DIST_L2, 5)
    ij = np.round(pts).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < rw) & (ij[:, 1] >= 0) & (ij[:, 1] < rh)
    near = np.zeros(len(pts), bool)
    near[ok] = dt[ij[ok, 1], ij[ok, 0]] <= 3 * cfg.inlier_px
    if near.sum() < 0.25 * len(pts):
        return h
    reach = np.zeros((rh, rw), np.uint8)
    for x, y in ij[near]:
        cv2.circle(reach, (int(x), int(y)), round(3 * cfg.inlier_px), 255, -1)
    trimmed = np.where(evidence & (reach > 0), 255, 0).astype(np.uint8)
    h2 = _ecc(trimmed, ref_scale, model, h, model.rail[near], cfg, motion)
    return h if h2 is None else h2


def _ecc(mask_ref: np.ndarray, ref_scale: float, model: StadiumModel, h0: np.ndarray,
         rail: np.ndarray, cfg: StadiumFinderConfig, motion: int) -> np.ndarray | None:
    m_scale = _REF_LONG_EDGE / max(model.image_size)
    mw, mh = round(model.image_size[0] * m_scale), round(model.image_size[1] * m_scale)
    tpl = np.zeros((mh, mw), np.float32)
    for x, y in rail * m_scale:
        cv2.circle(tpl, (round(x), round(y)), _DOT_RADIUS, 1.0, -1)
    tpl = cv2.GaussianBlur(tpl, (0, 0), _ECC_BLUR)
    # Align centre line to centre line, so a thick close-up rail and a thin one weigh the same.
    centre = cv2.dilate(_ridge(mask_ref).astype(np.uint8), np.ones((3, 3), np.uint8))
    inp = cv2.GaussianBlur(centre.astype(np.float32), (0, 0), _ECC_BLUR)
    s_in = np.diag([ref_scale, ref_scale, 1.0])
    s_tpl_inv = np.diag([1 / m_scale, 1 / m_scale, 1.0])
    w3 = s_in @ h0 @ s_tpl_inv
    w3 /= w3[2, 2]
    affine = motion == cv2.MOTION_AFFINE
    w = (w3[:2] if affine else w3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, cfg.refine_iterations, _ECC_EPS)
    try:
        _, w = cv2.findTransformECC(tpl, inp, w, motion, crit, None, 5)
    except cv2.error:
        return None
    w3 = np.vstack([w.astype(np.float64), [0.0, 0.0, 1.0]]) if affine else w.astype(np.float64)
    h = np.linalg.inv(s_in) @ w3 @ np.linalg.inv(s_tpl_inv)
    return h / h[2, 2]


def _score(mask_ref: np.ndarray, ref_scale: float, model: StadiumModel, h: np.ndarray,
           cfg: StadiumFinderConfig) -> StadiumFit | None:
    rh, rw = mask_ref.shape[:2]
    centre = model.rail.mean(0)
    # local Jacobian at the rail centre: a mirror (or degenerate) mapping is never accepted
    eps = 1.0
    p = _project(h, np.array([centre, centre + [eps, 0], centre + [0, eps]]))
    jac = np.stack([p[1] - p[0], p[2] - p[0]], 1) / eps
    if not np.isfinite(jac).all() or np.linalg.det(jac) <= 0:
        return None
    pts = _project(h, model.rail) * ref_scale
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < rw) & (pts[:, 1] >= 0) & (pts[:, 1] < rh)
    in_frame = float(inside.mean())
    if inside.sum() == 0:
        return None
    evidence = mask_ref > 0
    dt = cv2.distanceTransform((~evidence).astype(np.uint8), cv2.DIST_L2, 5)
    ij = np.round(pts[inside]).astype(int)
    ij[:, 0] = np.clip(ij[:, 0], 0, rw - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, rh - 1)
    d = dt[ij[:, 1], ij[:, 0]]
    coverage = float((d <= cfg.inlier_px).mean())
    matched = np.zeros(len(pts), bool)
    matched[inside] = d <= cfg.inlier_px
    notch, quadrants = _model_regions(model, cfg)
    notch_cov = float(matched[notch].mean()) if notch.any() else 0.0
    quad_cov = min((float(matched[q].mean()) if q.any() else 0.0) for q in quadrants)
    reach = np.zeros((rh, rw), np.uint8)
    for x, y in ij:
        cv2.circle(reach, (int(x), int(y)), round(2 * cfg.inlier_px), 255, -1)
    ridge = _ridge(mask_ref)
    n_ev = int(ridge.sum())
    precision = float((reach[ridge] > 0).sum() / n_ev) if n_ev else 0.0
    # Continuous error against the rail centre line: coverage and precision saturate once
    # every point is inside the band, the centre-line distance still tells fits apart.
    dt_ridge = cv2.distanceTransform((~ridge).astype(np.uint8), cv2.DIST_L2, 5)
    dr = dt_ridge[ij[:, 1], ij[:, 0]]
    return StadiumFit(homography=tuple(tuple(float(v) for v in row) for row in h),
                      coverage=coverage, precision=precision,
                      mean_error_px=float(np.minimum(dr, 2 * cfg.inlier_px).mean()),
                      rail_in_frame=in_frame, notch_coverage=notch_cov, quadrant_coverage=quad_cov)


def _model_regions(model: StadiumModel,
                   cfg: StadiumFinderConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    """Boolean masks over ``model.rail``: the cat-ear notch (near half, within
    ``notch_halfwidth_frac`` of the rail width from the centre) and the four quadrants
    left/right x near/far. The model template is upright: its near end is at the bottom."""
    x, y = model.rail[:, 0], model.rail[:, 1]
    cx, cy = model.rail.mean(0)
    near = y > cy
    left = x < cx
    notch = near & (np.abs(x - cx) <= cfg.notch_halfwidth_frac * (x.max() - x.min()))
    return notch, [left & near, ~left & near, left & ~near, ~left & ~near]


_QUALITY_TIE = 0.02  # coverage x precision differences below this are a tie
_COARSE_KEEP = 0.85  # refine only candidates whose coarse overlap is this close to the best
_HOMOGRAPHY_SKIP = 0.1  # affine coverage this far below min_coverage: skip the homography


def _quality(fit: StadiumFit) -> float:
    return fit.coverage * fit.precision


def _rank(fit: StadiumFit) -> tuple[float, float]:
    """Higher is better: quality (rounded to ties), then lower centre-line error."""
    return (round(_quality(fit) / _QUALITY_TIE), -fit.mean_error_px)


def _perspective(h: np.ndarray, ref_scale: float, model_scale: float) -> float:
    """max(|h20|, |h21|) of ``h`` (model px -> frame px) with both ends in inference px:
    ``ref_scale`` scales the frame and ``model_scale`` the model template to long edge 1280."""
    hr = np.diag([ref_scale, ref_scale, 1.0]) @ h @ np.diag([1 / model_scale, 1 / model_scale, 1.0])
    return float(max(abs(hr[2, 0]), abs(hr[2, 1])) / abs(hr[2, 2]))


def fit_stadium(mask: np.ndarray, model: StadiumModel,
                cfg: StadiumFinderConfig) -> StadiumFit | None:
    """Best homography mapping the model rail onto the rail ``mask`` (frame px), or ``None``
    when no candidate reaches ``min_coverage``, ``min_precision`` and ``min_rail_in_frame``."""
    if not (mask > 0).any():
        return None
    fh, fw = mask.shape[:2]
    ref_scale = _REF_LONG_EDGE / max(fw, fh)
    rw, rh = round(fw * ref_scale), round(fh * ref_scale)
    mask_ref = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_AREA)
    mask_ref = np.where(mask_ref > 0, 255, 0).astype(np.uint8)
    best: StadiumFit | None = None
    candidates = _coarse_candidates(mask, model, cfg)
    top = candidates[0][0] if candidates else 0.0
    for coarse, h0 in candidates:
        if coarse < _COARSE_KEEP * top:  # far behind the best overlap: not worth refining
            continue
        h = _refine(mask_ref, ref_scale, model, h0, cfg)
        if h is None:
            continue
        fit = _score(mask_ref, ref_scale, model, h, cfg)
        if fit is None:
            continue
        if fit.coverage >= cfg.min_coverage - _HOMOGRAPHY_SKIP:
            # Perspective only where the rail shape supports it: an oval alone cannot pin the
            # two perspective terms, so the homography must clearly beat the affine fit and
            # stay mild. Far below min_coverage, perspective cannot rescue the candidate.
            hp = _refine(mask_ref, ref_scale, model, h, cfg, cv2.MOTION_HOMOGRAPHY)
            if hp is not None:
                fp = _score(mask_ref, ref_scale, model, hp, cfg)
                model_scale = _REF_LONG_EDGE / max(model.image_size)
                if (fp is not None and _perspective(hp, ref_scale, model_scale) <= cfg.max_perspective
                        and _quality(fp) >= _quality(fit) - _QUALITY_TIE
                        and fp.mean_error_px <= fit.mean_error_px - cfg.perspective_min_gain_px):
                    fit = fp
        # Only acceptable candidates compete: a higher-ranked candidate that fails the checks
        # must not crowd out one that passes them.
        if _acceptable(fit, cfg) and (best is None or _rank(fit) > _rank(best)):
            best = fit
    if best is None:
        return None
    if _orientation_ambiguous(mask_ref, ref_scale, model, best, cfg):
        return None
    return best


def _acceptable(fit: StadiumFit, cfg: StadiumFinderConfig) -> bool:
    return (fit.coverage >= cfg.min_coverage and fit.precision >= cfg.min_precision
            and fit.rail_in_frame >= cfg.min_rail_in_frame
            and fit.notch_coverage >= cfg.min_notch_coverage
            and fit.quadrant_coverage >= cfg.min_quadrant_coverage)


def _orientation_ambiguous(mask_ref: np.ndarray, ref_scale: float, model: StadiumModel,
                           best: StadiumFit, cfg: StadiumFinderConfig) -> bool:
    """True when the fit turned 180 degrees about the rail centre is itself an acceptable fit
    explaining the rail almost as well: then the direction (X/B sides, which end holds the
    Xtreme pocket) is not determined.

    The turned fit is scored as it is, after affine refinement and after a homography
    refinement; a failed refinement never discards evidence the unrefined turn already has.
    An alternative failing the acceptance checks (coverage, notch, quadrants, ...) cannot
    veto a direction the rail shows clearly."""
    cx, cy = model.rail.mean(0)
    turn = np.array([[-1.0, 0.0, 2 * cx], [0.0, -1.0, 2 * cy], [0.0, 0.0, 1.0]])
    raw = np.asarray(best.homography) @ turn
    hypotheses = [raw]
    affine = _refine(mask_ref, ref_scale, model, raw, cfg)
    if affine is not None:
        hypotheses.append(affine)
    hom = _refine(mask_ref, ref_scale, model, raw if affine is None else affine, cfg,
                  cv2.MOTION_HOMOGRAPHY)
    if hom is not None:
        hypotheses.append(hom)
    for h in hypotheses:
        alt = _score(mask_ref, ref_scale, model, h, cfg)
        if (alt is not None and _acceptable(alt, cfg)
                and _quality(alt) > _quality(best) - cfg.orientation_min_gain):
            return True
    return False


# --------------------------------------------------------------------------- whole video
VISIBILITY_ZONES = ("stadium", "battle_area", "over_left", "over_right", "xtreme")
_AGREEMENT_ZONES = ("battle_area", "over_left", "over_right", "xtreme")


def zone_visibility(c: Calibration) -> dict[str, float]:
    """Share of each judged zone's area that lies inside the frame (0.0 for a zone with no
    area). A pocket outside the frame cannot show a finish, so callers flag it for review."""
    w, h = c.frame_size
    out: dict[str, float] = {}
    for name in VISIBILITY_ZONES:
        poly = np.array(getattr(c, name), np.float64).reshape(-1, 2)
        if len(poly) < 3:
            out[name] = 0.0
            continue
        total = abs(cv2.contourArea(poly.astype(np.float32)))
        if total <= 0.0:
            out[name] = 0.0
            continue
        frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        inter, _ = cv2.intersectConvexConvex(_convex(poly), frame) if _is_convex(poly) else (
            _clipped_area(poly, w, h), None)
        out[name] = float(min(1.0, max(0.0, inter / total)))
    return out


def _is_convex(poly: np.ndarray) -> bool:
    return bool(cv2.isContourConvex(poly.astype(np.float32).reshape(-1, 1, 2)))


def _convex(poly: np.ndarray) -> np.ndarray:
    return poly.astype(np.float32).reshape(-1, 1, 2)


def _clipped_area(poly: np.ndarray, w: int, h: int) -> float:
    """Area of ``poly`` inside the frame for a non-convex polygon: rasterised on a canvas
    padded by the polygon's own extent, counted inside the frame."""
    lo = np.floor(np.minimum(poly.min(0), 0)).astype(int)
    hi = np.ceil(np.maximum(poly.max(0), [w, h])).astype(int)
    canvas = np.zeros((hi[1] - lo[1] + 1, hi[0] - lo[0] + 1), np.uint8)
    cv2.fillPoly(canvas, [np.round(poly - lo).astype(np.int32)], 1)
    inside = canvas[-lo[1]:-lo[1] + h, -lo[0]:-lo[0] + w]
    full = float(canvas.sum())
    total = abs(cv2.contourArea(poly.astype(np.float32)))
    return float(inside.sum()) / full * total if full else 0.0


@dataclass(frozen=True)
class FoundStadium:
    calibration: Calibration  # zones in the decoded (inference-size) frame
    fit: StadiumFit
    frame_index: int  # source frame index of the chosen frame
    frame: np.ndarray  # that frame (BGR, inference size)
    zone_visibility: dict[str, float]
    agreeing_frames: int  # candidate frames whose zones agree with the chosen one


def _zones(model: StadiumModel, fit: StadiumFit) -> np.ndarray:
    h = np.asarray(fit.homography, np.float64)
    return np.concatenate([_project(h, np.array(getattr(model.calibration, z), np.float64))
                           for z in _AGREEMENT_ZONES])


def _alignment(fit: StadiumFit, model: StadiumModel, frame_size: tuple[int, int]):
    """The ``autocalib.Alignment`` ``warp_calibration`` expects; only ``homography`` moves the
    zones, the other fields describe the mapping at the rail centre."""
    from bxjudge.autocalib import Alignment

    h = np.asarray(fit.homography, np.float64)
    centre = model.rail.mean(0)
    p = _project(h, np.array([centre, centre + [1.0, 0.0], centre + [0.0, 1.0]]))
    jac = np.stack([p[1] - p[0], p[2] - p[0]], 1)
    return Alignment(
        homography=fit.homography,
        score=min(fit.coverage, fit.precision),
        shift_frac=float(np.linalg.norm(p[0] - centre) / frame_size[0]),
        rotation_deg=float(math.degrees(math.atan2(jac[1, 0], jac[0, 0]))),
        scale=float(math.sqrt(abs(np.linalg.det(jac)))),
        perspective=float(max(abs(h[2, 0]), abs(h[2, 1]))),
    )


def find_stadium(video: Path, model: StadiumModel, cfg: StadiumFinderConfig, *,
                 hwaccel: str = "cpu") -> FoundStadium | None:
    """Stadium zones for ``video`` from its clearest early frames, or ``None``.

    Frames are sampled every ``sample_step_s`` within the first ``sample_window_s`` (playback
    seconds until the physical time base lands, ledger ruling T4), ranked by rail pixels, and
    the best ``candidate_frames`` are fitted in that order. A result needs two fits whose zones
    agree within ``agreement_px`` (inference px); the better-ranked of them is returned. A
    video without such a pair (no stadium, a moving camera, hands everywhere) gives ``None``."""
    from bxjudge.autocalib import warp_calibration
    from bxjudge.manifest import probe
    from bxjudge.video import VideoDecodeError, iter_frames

    try:
        fps = probe(Path(video)).fps
        step = max(1, round(cfg.sample_step_s * fps))
        samples = []
        for idx, frame in iter_frames(Path(video), 0.0, cfg.sample_window_s,
                                      long_edge=_REF_LONG_EDGE, hwaccel=hwaccel):
            if idx % step == 0:
                mask = rail_mask(frame, cfg)
                samples.append((int((mask > 0).sum()), idx, frame, mask))
    except (VideoDecodeError, ValueError, OSError):
        return None
    samples.sort(key=lambda s: (-s[0], s[1]))
    fits: list[tuple[int, np.ndarray, StadiumFit, np.ndarray]] = []
    for pixels, idx, frame, mask in samples[:cfg.candidate_frames]:
        if pixels == 0:
            break
        fit = fit_stadium(mask, model, cfg)
        if fit is None:
            continue
        zones = _zones(model, fit)
        fits.append((idx, frame, fit, zones))
        agree = [f for f in fits
                 if float(np.linalg.norm(f[3] - zones, axis=1).mean()) <= cfg.agreement_px]
        if len(agree) >= 2:
            idx, frame, fit, _ = max(agree, key=lambda f: _rank(f[2]))
            fh, fw = frame.shape[:2]
            cal = warp_calibration(model.calibration, _alignment(fit, model, (fw, fh)), (fw, fh),
                                   f"found-{Path(video).stem}")
            return FoundStadium(calibration=cal, fit=fit, frame_index=idx, frame=frame,
                                zone_visibility=zone_visibility(cal), agreeing_frames=len(agree))
    return None


# --------------------------------------------------------------------------- truth
# User ruling 2026-11-11: manual truth covers the stadium, the battle floor and the three
# pockets only; launch zones and the lid are generated but not validated.
TRUTH_ZONES: list[tuple[str, str, str]] = [z for z in ZONES if z[0] in VISIBILITY_ZONES]


def zone_boundary_errors(found: Calibration, truth: dict[str, Polygon],
                         truth_size: tuple[int, int], long_edge: int) -> dict[str, float]:
    """Per-zone mean boundary distance between ``found`` and manual ``truth`` polygons (given
    in a ``truth_size`` frame), both scaled so the long edge is ``long_edge`` px. Uses the
    same correspondence as ``autocalib.mean_boundary_error``."""
    from bxjudge.autocalib import _boundary_distance

    kf = long_edge / max(found.frame_size)
    kt = long_edge / max(truth_size)
    out = {}
    for name, poly in truth.items():
        a = [(x * kf, y * kf) for x, y in getattr(found, name)]
        b = [(x * kt, y * kt) for x, y in poly]
        out[name] = _boundary_distance(a, b)
    return out


def save_stadium_model(model: StadiumModel, path: Path) -> None:
    doc = {
        "schema": _MODEL_SCHEMA,
        "source_sha256": model.source_sha256,
        "image_size": list(model.image_size),
        "rail": [[float(x), float(y)] for x, y in model.rail],
        "calibration": _calibration_doc(model.calibration),
    }
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")


def load_stadium_model(path: Path = DEFAULT_STADIUM_MODEL) -> StadiumModel:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != _MODEL_SCHEMA:
        raise ValueError(f"{path}: not a {_MODEL_SCHEMA} file")
    rail = np.array(doc["rail"], np.float64).reshape(-1, 2)
    w, h = doc["image_size"]
    return StadiumModel(rail=rail, calibration=_calibration_from_doc(doc["calibration"]),
                        source_sha256=str(doc["source_sha256"]), image_size=(int(w), int(h)))
