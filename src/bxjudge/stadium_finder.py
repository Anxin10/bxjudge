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
import logging
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from bxjudge import PROJECT_ROOT
from bxjudge.calibration import ZONES
from bxjudge.config import config_loader
from bxjudge.geometry import polygon_mask
from bxjudge.types import Calibration, Polygon

_log = logging.getLogger(__name__)

DEFAULT_STADIUM_FINDER_CONFIG = PROJECT_ROOT / "config" / "stadium_finder.yaml"
DEFAULT_STADIUM_MODEL = PROJECT_ROOT / "config" / "stadium_model.json"

_POLY_FIELDS = [z[0] for z in ZONES]
_MODEL_SCHEMA = "bxjudge-stadium-model-v1"
_MODEL_IMAGE_WIDTH = 960  # stored reference image width (edge refinement runs at 640)


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
    edge_max_shift_px: float
    edge_refinement: bool
    agreement_px: float
    zone_visible_min: float
    # keypoint path (plan 2026-11-11-stadium-keypoints K4/K5)
    keypoint_weights: tuple[str, ...]
    keypoint_imgsz: int
    keypoint_min_conf: float
    keypoint_ransac_px: float
    keypoint_min_inliers: int
    keypoint_max_error_px: float
    keypoint_max_perspective: float
    keypoint_max_spread_px: float
    keypoint_max_rail_shift_px: float
    keypoint_min_rail_coverage: float
    keypoint_min_rail_precision: float
    keypoint_agreement_px: float
    # E9 (default off): when the first window finds no stadium, try the following windows of
    # the same length that start no later than later_windows_max_start_s. Optional keys so
    # older config copies keep loading; callers enable it for fixed cameras only.
    later_windows_enabled: bool = False
    later_windows_max_start_s: float = 30.0


_OPTIONAL = ("later_windows_enabled", "later_windows_max_start_s")
_INT_FIELDS = ("model_points", "hue_halfwidth", "min_saturation", "min_value",
               "candidate_frames", "work_width", "coarse_scale_steps", "coarse_top_k",
               "refine_iterations", "keypoint_imgsz", "keypoint_min_inliers")
_FRACTIONS = ("roi_margin_frac", "max_thickness_frac", "min_component_frac", "min_coverage",
              "min_precision", "min_rail_in_frame", "notch_halfwidth_frac", "min_notch_coverage",
              "min_quadrant_coverage", "orientation_min_gain", "zone_visible_min",
              "keypoint_min_conf", "keypoint_min_rail_coverage", "keypoint_min_rail_precision")


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
    missing = sorted(set(names) - set(raw) - set(_OPTIONAL))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    values: dict = {}
    for name in names:
        if name not in raw:
            continue
        v = raw[name]
        if name in ("edge_refinement", "later_windows_enabled"):
            if not isinstance(v, bool):
                raise ValueError(f"{path}: {name} must be true or false, got {v!r}")
        elif name == "model_template":
            if not isinstance(v, str) or not v or "/" in v or "\\" in v:
                raise ValueError(f"{path}: model_template must be a calibration name, got {v!r}")
        elif name == "keypoint_weights":
            # project-relative weight files; never absolute or leaving the project
            if (not isinstance(v, list)
                    or any(not isinstance(w, str) or not w or w.startswith(("/", "\\"))
                           or ":" in w or ".." in Path(w.replace("\\", "/")).parts for w in v)):
                raise ValueError(f"{path}: keypoint_weights must be a list of project-relative "
                                 f"paths, got {v!r}")
            v = tuple(v)
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
    if not 4 <= c.keypoint_min_inliers <= 10:
        raise ValueError(f"{path}: keypoint_min_inliers must be 4..10 (a homography needs 4 "
                         f"points, there are 10), got {c.keypoint_min_inliers}")
    if c.keypoint_imgsz < 320 or c.keypoint_imgsz % 32:
        raise ValueError(f"{path}: keypoint_imgsz must be a multiple of 32 and >= 320, "
                         f"got {c.keypoint_imgsz}")
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
    # The template's reference image, downscaled (model px * image_scale = image px); used to
    # refine the rail fit with the stadium's own edges. None for a model without one.
    image: np.ndarray | None = None
    image_scale: float = 1.0

    def __eq__(self, other):  # ndarray fields: compare by value
        return (isinstance(other, StadiumModel) and np.array_equal(self.rail, other.rail)
                and self.calibration == other.calibration
                and self.source_sha256 == other.source_sha256
                and self.image_size == other.image_size
                and self.image_scale == other.image_scale
                and ((self.image is None and other.image is None)
                     or (self.image is not None and other.image is not None
                         and np.array_equal(self.image, other.image))))


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
    _check_simple_zones(template, f"template {template.name}")
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
    scale = min(1.0, _MODEL_IMAGE_WIDTH / w)
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return StadiumModel(rail=rail, calibration=cal, source_sha256=digest.hexdigest(),
                        image_size=(int(w), int(h)), image=small,
                        image_scale=small.shape[1] / w)


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
    # Keypoint fits only: RANSAC inliers among the 10 points and their mean reprojection
    # error (inference px). 0 for rail fits.
    keypoint_inliers: int = 0
    keypoint_error_px: float = 0.0
    keypoint_rail_shift_px: float = 0.0  # worst zone's move under rail refinement (inference px)


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Points through homography ``h`` by plain homogeneous division. OpenCV's
    perspectiveTransform returns (0, 0) for a near-zero denominator, which would hide a point
    thrown towards infinity inside the frame; here it comes out huge or non-finite."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    h = np.asarray(h, np.float64)
    num = p @ h[:2, :2].T + h[:2, 2]
    w = p @ h[2, :2] + h[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / w[:, None]


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
    if not _zones_valid(model, h, (round(rw / ref_scale), round(rh / ref_scale))):
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


_MAX_ZONE_EXTENT = 10.0  # projected zone vertices stay within this many frame sizes


def _zones_valid(model: StadiumModel, h: np.ndarray, frame_size: tuple[int, int]) -> bool:
    """Every zone vertex projects finitely through ``h`` (model px -> frame px): its projective
    denominator has the rail centre's sign (the horizon does not cross the zones) and it lands
    within ``_MAX_ZONE_EXTENT`` frame sizes. A vertex near the horizon is thrown far away and
    fails the extent check; a steep but valid view with all zones on screen passes."""
    h = np.asarray(h, np.float64)
    verts = np.concatenate([np.array(getattr(model.calibration, z), np.float64).reshape(-1, 2)
                            for z in _POLY_FIELDS if getattr(model.calibration, z)])
    cx, cy = model.rail.mean(0)
    w_c = h[2, 0] * cx + h[2, 1] * cy + h[2, 2]
    w = verts @ h[2, :2] + h[2, 2]
    if not np.isfinite(w).all() or not np.isfinite(w_c) or w_c == 0:
        return False
    if (w / w_c).min() <= 0.0:
        return False
    pts = _project(h, verts)
    fw, fh = frame_size
    lim = _MAX_ZONE_EXTENT * max(fw, fh)
    return bool(np.isfinite(pts).all() and np.abs(pts).max() <= lim)


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


def fit_stadium(mask: np.ndarray, model: StadiumModel, cfg: StadiumFinderConfig, *,
                frame: np.ndarray | None = None, acfg=None) -> StadiumFit | None:
    """Best homography mapping the model rail onto the rail ``mask`` (frame px), or ``None``
    when no candidate passes the acceptance checks.

    With ``frame`` (the BGR image ``mask`` came from), ``acfg`` (``AutoCalibConfig``) and a
    model reference image, every candidate is refined with the stadium's own edges (see
    ``_edge_refine``): the rail alone cannot pin perspective, the lid, pockets and outline can.
    Without them only the rail decides, and a candidate whose affine and perspective fits
    place the zones apart is dropped."""
    if not (mask > 0).any():
        return None
    use_edges = frame is not None and acfg is not None and model.image is not None
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
        model_scale = _REF_LONG_EDGE / max(model.image_size)
        fp = None  # an acceptable perspective fit within the limit, if any
        if fit.coverage >= cfg.min_coverage - _HOMOGRAPHY_SKIP:
            # Far below min_coverage, perspective cannot rescue the candidate.
            hp = _refine(mask_ref, ref_scale, model, h, cfg, cv2.MOTION_HOMOGRAPHY)
            if hp is not None and _perspective(hp, ref_scale, model_scale) <= cfg.max_perspective:
                fp = _score(mask_ref, ref_scale, model, hp, cfg)
                if fp is not None and not _acceptable(fp, cfg):
                    fp = None
        if use_edges:
            # Refine from the affine and the perspective fit with the stadium's edges. The
            # refined fits must still match the rail, and when both seeds refine they must
            # agree on the zones; otherwise the edges did not settle the view.
            refined = []
            for seed in [fit] + ([fp] if fp is not None else []):
                he = _edge_refine(frame, model, np.asarray(seed.homography, np.float64), cfg, acfg)
                if he is None or _perspective(he, ref_scale, model_scale) > cfg.max_perspective:
                    continue
                fe = _score(mask_ref, ref_scale, model, he, cfg)
                if fe is not None and _acceptable(fe, cfg):
                    refined.append(fe)
            if not refined:
                continue
            if (len(refined) == 2 and float(np.linalg.norm(
                    _zones(model, refined[0]) - _zones(model, refined[1]), axis=1).mean())
                    * ref_scale > cfg.agreement_px):
                continue
            fit = min(refined, key=lambda f: f.mean_error_px)
        elif fp is not None:
            # Rail only. An acceptable perspective fit is compared by the centre-line error,
            # the continuous measure of how well the rail lines up (coverage x precision
            # saturates and drifts with stray green). Clearly better: it wins. About as good
            # but placing the zones elsewhere: the rail cannot tell which view is right, so the
            # candidate is dropped rather than published.
            if fp.mean_error_px <= fit.mean_error_px - cfg.perspective_min_gain_px:
                fit = fp
            elif (float(np.linalg.norm(_zones(model, fp) - _zones(model, fit), axis=1).mean())
                  * ref_scale > cfg.agreement_px):
                continue
        # Only acceptable candidates compete: a higher-ranked candidate that fails the checks
        # must not crowd out one that passes them.
        if _acceptable(fit, cfg) and (best is None or _rank(fit) > _rank(best)):
            best = fit
    if best is None:
        return None
    confirm = None
    if use_edges:
        def confirm(h: np.ndarray) -> StadiumFit | None:
            """The turned alternative as the edges see it: refined and still on the rail."""
            he = _edge_refine(frame, model, np.asarray(h, np.float64), cfg, acfg)
            if he is None or _perspective(he, ref_scale, _REF_LONG_EDGE / max(model.image_size)) \
                    > cfg.max_perspective:
                return None
            return _score(mask_ref, ref_scale, model, he, cfg)
    if _orientation_ambiguous(mask_ref, ref_scale, model, best, cfg, confirm):
        return None
    return best


def fit_from_keypoints(points: np.ndarray, reference, mask: np.ndarray, model: StadiumModel,
                       cfg: StadiumFinderConfig) -> StadiumFit | None:
    """Homography from the template's 10 reference points to the predicted ``points``
    ((10, 3) [x, y, confidence] in frame px, ``stadium_keypoints.KEYPOINTS`` order), checked
    against the rail ``mask`` (frame px), or ``None``.

    Points below ``keypoint_min_conf`` are not used. RANSAC (``keypoint_ransac_px``) drops a
    point placed on the wrong feature; the inliers (at least ``keypoint_min_inliers``) are then
    fitted by least squares and must include points left and right of the rail centre and in
    front of and behind it, or one side of the zones is not constrained. The fit is refused
    when the inliers' mean reprojection error exceeds ``keypoint_max_error_px``, its
    perspective ``keypoint_max_perspective``, when it mirrors the stadium (``_score``: X/B
    would swap) or throws a zone towards the horizon.

    The rail is independent evidence for the view the points describe: the placed model rail
    must match at least ``keypoint_min_rail_coverage`` of its in-frame points and explain at
    least ``keypoint_min_rail_precision`` of the frame's rail, and the fit turned 180 degrees,
    refined on the rail, must explain it clearly worse (``_orientation_ambiguous``, as the rail
    path): points that agree with each other but describe the stadium turned round (Xtreme at
    the near end, the Over pockets swapped) are refused (Codex review 2026-11-12 High)."""
    check_keypoint_reference(reference, model)
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    ok = np.isfinite(pts).all(1) & (pts[:, 2] >= cfg.keypoint_min_conf)
    if ok.sum() < cfg.keypoint_min_inliers or not (mask > 0).any():
        return None
    fh, fw = mask.shape[:2]
    ref_scale = _REF_LONG_EDGE / max(fw, fh)
    src, dst = reference.array()[ok], pts[ok, :2]
    h, inl = cv2.findHomography(src, dst, cv2.RANSAC, cfg.keypoint_ransac_px / ref_scale)
    if h is None or inl is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < cfg.keypoint_min_inliers:
        return None
    h, _ = cv2.findHomography(src[inl], dst[inl], 0)  # least squares on the inliers
    if h is None or not np.isfinite(h).all() or h[2, 2] == 0:
        return None
    h = h / h[2, 2]
    cx, cy = model.rail.mean(0)
    s = src[inl]
    if not ((s[:, 0] < cx).any() and (s[:, 0] > cx).any()
            and (s[:, 1] < cy).any() and (s[:, 1] > cy).any()):
        return None
    err = float(np.linalg.norm(_project(h, s) - dst[inl], axis=1).mean() * ref_scale)
    if not np.isfinite(err) or err > cfg.keypoint_max_error_px:
        return None
    if _perspective(h, ref_scale, _REF_LONG_EDGE / max(model.image_size)) > \
            cfg.keypoint_max_perspective:
        return None
    rw, rh = round(fw * ref_scale), round(fh * ref_scale)
    mask_ref = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_AREA)
    mask_ref = np.where(mask_ref > 0, 255, 0).astype(np.uint8)
    def rail_supports(f: StadiumFit | None) -> bool:
        return (f is not None and f.coverage >= cfg.keypoint_min_rail_coverage
                and f.precision >= cfg.keypoint_min_rail_precision)

    fit = _score(mask_ref, ref_scale, model, h, cfg)
    if not rail_supports(fit):
        return None
    # The rail pulls the fit to the view it shows, and each judged zone must stay near where
    # the rail puts it: a fit right on average can still have one pocket off (Codex re-review
    # 2026-11-12 High). The rail is a coarse check, not the truth: on the labelled frames the
    # refined fit is further from the labels than the keypoint fit (worst zone median 19 vs
    # 13 px), so the keypoint fit is kept and only gross errors are caught (worst zone moved
    # by refinement on correct fits: median 20, p95 41 px). A failed refinement gives no
    # support.
    hr = _refine(mask_ref, ref_scale, model, h, cfg, cv2.MOTION_HOMOGRAPHY)
    if hr is None:
        return None
    shift = _worst_zone_shift(model, h, hr) * ref_scale
    if not np.isfinite(shift) or shift > cfg.keypoint_max_rail_shift_px:
        return None
    # The turned view competes on the same standard as this one: when the rail supports both
    # directions about as well, the points alone decide the direction, and that is refused.
    if _orientation_ambiguous(mask_ref, ref_scale, model, fit, cfg, None, accept=rail_supports,
                              max_perspective=cfg.keypoint_max_perspective):
        return None
    return replace(fit, keypoint_inliers=int(inl.sum()), keypoint_error_px=err,
                   keypoint_rail_shift_px=shift)


def check_keypoint_reference(reference, model: StadiumModel) -> None:
    """The reference points must have been labelled on this model's template frame: same
    template name, frame size and model identity (``StadiumModel.source_sha256``, which
    covers the template calibration and its reference image). Raises ``ValueError``."""
    if (reference.template != model.calibration.name
            or tuple(reference.image_size) != tuple(model.image_size)
            or reference.model_sha256 != model.source_sha256):
        raise ValueError(f"keypoint reference ({reference.template} {reference.image_size} "
                         f"model {reference.model_sha256[:12]}) does not belong to the stadium "
                         f"model ({model.calibration.name} {model.image_size} "
                         f"{model.source_sha256[:12]})")


def _edge_refine(frame: np.ndarray, model: StadiumModel, h0: np.ndarray,
                 cfg: StadiumFinderConfig, acfg) -> np.ndarray | None:
    """Refine ``h0`` (model px -> frame px) with the stadium's own edges, or ``None``.

    The model's reference image is warped into the frame through ``h0``; the remaining
    motion is estimated as in ``autocalib.estimate_alignment`` but without its coarse search
    (``h0`` is already close): polarity- and colour-free edge maps normalised inside the warped
    template's rim region (lid, pockets, outline), homography ECC at work/2 then work width.
    Rejected when ECC fails, the edge score is below ``acfg.min_score``, or the zones would
    move more than ``edge_max_shift_px`` (inference px): the edges then do not confirm the
    rail fit's neighbourhood."""
    from bxjudge.autocalib import _ECC_GAUSS, _edge_ncc, _gray, _normalise, _region_mask, _scale_h
    from bxjudge.calibration import _edge_map
    from bxjudge.geometry import RIM_MASK_MARGIN, rim_mask

    fh, fw = frame.shape[:2]
    s = model.image_scale
    h_img = h0 @ np.diag([1.0 / s, 1.0 / s, 1.0])
    warped = cv2.warpPerspective(model.image, h_img, (fw, fh), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    polys = {f: [tuple(p) for p in _project(h0, np.array(getattr(model.calibration, f), np.float64))]
             for f in _POLY_FIELDS}
    wcal = Calibration(name="warped", frame_size=(fw, fh), ref_frame="", **polys)
    full = (acfg.work_width, max(1, round(acfg.work_width * fh / fw)))
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, acfg.ecc_iterations, acfg.ecc_epsilon)
    h = np.eye(3)
    prev = None
    try:
        for div in (2, 1):
            size = (max(1, round(full[0] / div)), max(1, round(full[1] / div)))
            mask = _region_mask(wcal, size)
            tmap = _normalise(_edge_map(_gray(warped, size)), mask)
            fmap = _normalise(_edge_map(_gray(frame, size)), mask)
            if tmap is None or fmap is None:
                return None
            if prev is not None:
                h = _scale_h(h, size[0] / prev[0], size[1] / prev[1])
            prev = size
            warp = np.linalg.inv(h).astype(np.float32)
            _, warp = cv2.findTransformECC(fmap, tmap, warp, cv2.MOTION_HOMOGRAPHY, crit,
                                           mask.astype(np.uint8) * 255, _ECC_GAUSS)
            h = np.linalg.inv(warp.astype(np.float64))
    except (cv2.error, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-12:
        return None
    h /= h[2, 2]
    rim = rim_mask(wcal, full, max(1, round(RIM_MASK_MARGIN * full[0] / 320)))
    te = _edge_map(_gray(warped, full))
    fe = _edge_map(_gray(frame, full))
    wte = cv2.warpPerspective(te, h, full, flags=cv2.INTER_LINEAR)
    wrim = cv2.warpPerspective(rim.astype(np.uint8), h, full, flags=cv2.INTER_NEAREST) > 0
    if _edge_ncc(wte, fe, wrim) < acfg.min_score:
        return None
    s_f = np.diag([full[0] / fw, full[1] / fh, 1.0])
    out = np.linalg.inv(s_f) @ h @ s_f @ h0
    out /= out[2, 2]
    moved = np.concatenate([_project(out, np.array(getattr(model.calibration, z), np.float64))
                            - _project(h0, np.array(getattr(model.calibration, z), np.float64))
                            for z in _AGREEMENT_ZONES])
    shift = float(np.linalg.norm(moved, axis=1).mean()) * (_REF_LONG_EDGE / max(fw, fh))
    if not np.isfinite(shift) or shift > cfg.edge_max_shift_px:
        return None
    return out


def _acceptable(fit: StadiumFit, cfg: StadiumFinderConfig) -> bool:
    return (fit.coverage >= cfg.min_coverage and fit.precision >= cfg.min_precision
            and fit.rail_in_frame >= cfg.min_rail_in_frame
            and fit.notch_coverage >= cfg.min_notch_coverage
            and fit.quadrant_coverage >= cfg.min_quadrant_coverage)


def _orientation_ambiguous(mask_ref: np.ndarray, ref_scale: float, model: StadiumModel,
                           best: StadiumFit, cfg: StadiumFinderConfig,
                           confirm=None, *, accept=None,
                           max_perspective: float | None = None) -> bool:
    """True when the fit turned 180 degrees about the rail centre is itself an acceptable fit
    explaining the rail almost as well: then the direction (X/B sides, which end holds the
    Xtreme pocket) is not determined.

    The turned fit is scored as it is, after affine refinement and after a homography
    refinement; a failed refinement never discards evidence the unrefined turn already has.
    An alternative failing the acceptance checks (coverage, notch, quadrants, ...) cannot
    veto a direction the rail shows clearly. With ``confirm`` (edge refinement of a
    hypothesis), an alternative counts only when the stadium's edges confirm it too.

    ``accept`` (default ``_acceptable``) and ``max_perspective`` (default
    ``cfg.max_perspective``) are the standard the alternative must meet: the same one the
    candidate itself met, so both directions compete on equal terms (Codex re-review
    2026-11-12 High: keypoint fits passed looser checks than their turned competitor had to)."""
    if accept is None:
        def accept(f):
            return _acceptable(f, cfg)
    if max_perspective is None:
        max_perspective = cfg.max_perspective
    cx, cy = model.rail.mean(0)
    turn = np.array([[-1.0, 0.0, 2 * cx], [0.0, -1.0, 2 * cy], [0.0, 0.0, 1.0]])
    raw = np.asarray(best.homography) @ turn
    hypotheses = [raw]
    affine = _refine(mask_ref, ref_scale, model, raw, cfg)
    if affine is not None:
        hypotheses.append(affine)
    hom = _refine(mask_ref, ref_scale, model, raw if affine is None else affine, cfg,
                  cv2.MOTION_HOMOGRAPHY)
    # The alternative obeys the same perspective limit as any accepted fit.
    if hom is not None and _perspective(hom, ref_scale, _REF_LONG_EDGE / max(model.image_size)) \
            <= max_perspective:
        hypotheses.append(hom)
    for h in hypotheses:
        alt = _score(mask_ref, ref_scale, model, h, cfg)
        if (alt is None or not accept(alt)
                or _quality(alt) <= _quality(best) - cfg.orientation_min_gain):
            continue
        if confirm is None:
            return True
        # With the frame at hand the stadium's edges also judge the turned view: a turn they
        # cannot confirm (refinement fails or leaves the rail) is not a competitor.
        seen = confirm(h)
        if seen is not None and accept(seen):
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
        total = abs(_area(poly))
        if not np.isfinite(total) or total <= 0.0 or not _is_simple(poly):
            out[name] = 0.0  # a self-intersecting zone's signed area cancels: never "visible"
            continue
        inside = abs(_area(_clip_to_rect(poly, w, h)))
        out[name] = float(min(1.0, max(0.0, inside / total)))
    return out


def _is_simple(poly: np.ndarray) -> bool:
    """True when no two non-adjacent edges of the closed polygon cross or touch."""
    p = np.asarray(poly, np.float64).reshape(-1, 2)
    n = len(p)

    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if v == 0 else (1 if v > 0 else -1)

    def on_seg(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue  # adjacent edges share a vertex
            c, d = p[j], p[(j + 1) % n]
            o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
            if o1 != o2 and o3 != o4:
                return False
            if (o1 == 0 and on_seg(a, b, c)) or (o2 == 0 and on_seg(a, b, d)) \
                    or (o3 == 0 and on_seg(c, d, a)) or (o4 == 0 and on_seg(c, d, b)):
                return False
    return True


def _check_simple_zones(c: Calibration, where: str) -> None:
    """The judged zones (whose visibility decides reviews) must not cross themselves. The lid
    and launch zones are not checked: the manual 20261014_202713 lid crosses itself (ledger)."""
    for name in VISIBILITY_ZONES:
        poly = getattr(c, name)
        if poly and not _is_simple(np.array(poly, np.float64)):
            raise ValueError(f"{where}: zone {name} crosses itself")


def _area(poly: np.ndarray) -> float:
    """Shoelace area (float64, any size) of a closed polygon."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_to_rect(poly: np.ndarray, w: float, h: float) -> np.ndarray:
    """Sutherland-Hodgman clip of ``poly`` to the frame rectangle. The clip region is convex,
    so the clipped area is exact for any simple polygon; no raster, no size limit."""
    pts = [tuple(p) for p in poly]
    for axis, bound, keep_low in ((0, 0.0, False), (0, float(w), True),
                                  (1, 0.0, False), (1, float(h), True)):
        inside = (lambda p: p[axis] <= bound) if keep_low else (lambda p: p[axis] >= bound)
        out: list[tuple[float, float]] = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(_cross(prev, cur, axis, bound))
                out.append(cur)
            elif inside(prev):
                out.append(_cross(prev, cur, axis, bound))
        pts = out
        if not pts:
            break
    return np.array(pts, np.float64).reshape(-1, 2)


def _cross(a: tuple[float, float], b: tuple[float, float], axis: int, bound: float) -> tuple[float, float]:
    t = (bound - a[axis]) / (b[axis] - a[axis])
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


@dataclass(frozen=True)
class FoundStadium:
    calibration: Calibration  # zones in the decoded (inference-size) frame
    fit: StadiumFit
    frame_index: int  # source frame index of the chosen frame
    frame: np.ndarray  # that frame (BGR, inference size)
    zone_visibility: dict[str, float]
    agreeing_frames: int  # candidate frames whose zones agree with the chosen one
    method: str = "rail"  # "keypoints" or "rail"


def _zones(model: StadiumModel, fit: StadiumFit) -> np.ndarray:
    return _zones_h(model, np.asarray(fit.homography, np.float64))


def _zones_h(model: StadiumModel, h: np.ndarray) -> np.ndarray:
    """Vertices of the judged zones (``_AGREEMENT_ZONES``) through homography ``h``."""
    return np.concatenate([_project(h, np.array(getattr(model.calibration, z), np.float64))
                           for z in _AGREEMENT_ZONES])


def _worst_zone_shift(model: StadiumModel, h1: np.ndarray, h2: np.ndarray) -> float:
    """Largest per-zone mean vertex distance between the judged zones through ``h1`` and
    ``h2`` (frame px): each zone on its own, so one pocket off is not averaged away by the
    zones that are right."""
    worst = 0.0
    for z in _AGREEMENT_ZONES:
        poly = np.array(getattr(model.calibration, z), np.float64)
        d = float(np.linalg.norm(_project(h1, poly) - _project(h2, poly), axis=1).mean())
        if not np.isfinite(d):
            return float("inf")
        worst = max(worst, d)
    return worst


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
                 hwaccel: str = "cpu", keypoints=None,
                 keypoint_errors: list[str] | None = None,
                 keypoint_fits: list | None = None,
                 later_windows: bool = False) -> FoundStadium | None:
    """Stadium zones for ``video`` from its clearest early frames, or ``None``.

    ``later_windows`` (callers pass it for fixed cameras only) lets
    ``cfg.later_windows_enabled`` retry later windows of the same length when the first
    finds nothing; each window still needs its own two agreeing fits.

    Frames are sampled every ``sample_step_s`` within the first ``sample_window_s`` (playback
    seconds until the physical time base lands, ledger ruling T4), ranked by rail pixels, and
    the best ``candidate_frames`` are fitted in that order. A result needs two fits whose zones
    agree within ``agreement_px`` (inference px); the better-ranked of them is returned. A
    video without such a pair (no stadium, a moving camera, hands everywhere) gives ``None``.

    With ``keypoints`` (a ``stadium_keypoints.KeypointFinder``, or anything with its
    ``predict(frame)`` and ``reference``) the same candidate frames are first fitted from the
    predicted keypoints (``fit_from_keypoints``); two keypoint fits agreeing within
    ``keypoint_agreement_px`` give the result (the one with the lower keypoint error). Only
    when they do not is the rail fitted as above. On the 24 validation images the keypoint
    path placed the zone features at a median 6.6 px from the user's labels, the rail path
    found 8 of the 24 at 21 px (inference px, ledger 2026-11-12).

    A frame whose keypoint inference or fit raises is skipped (the rail path still runs); the
    error is appended to ``keypoint_errors`` when given, so a caller can tell a resource
    failure from a video without a stadium. Every accepted keypoint fit is appended to
    ``keypoint_fits`` as ``(frame_index, frame, fit)`` when given: a handheld camera rarely
    holds still for two agreeing frames, and its tracker may start from one fit and confirm
    it later (plan 2026-11-12-dynamic-calibration)."""
    from bxjudge.autocalib import load_autocalib_config, warp_calibration
    from bxjudge.manifest import probe
    from bxjudge.video import VideoDecodeError, iter_frames

    acfg = load_autocalib_config()
    try:
        meta = probe(Path(video))
    except (VideoDecodeError, ValueError, OSError):
        return None
    starts = [0.0]
    if later_windows and cfg.later_windows_enabled:
        # E9: a stadium hidden at the start (scoreboard overlay, setup) may be clear later.
        k = 1
        while (k * cfg.sample_window_s <= cfg.later_windows_max_start_s
               and k * cfg.sample_window_s < meta.duration):
            starts.append(k * cfg.sample_window_s)
            k += 1
    for start in starts:
        result = _find_in_window(video, model, cfg, acfg, meta.fps, start, hwaccel=hwaccel,
                                 keypoints=keypoints, keypoint_errors=keypoint_errors,
                                 keypoint_fits=keypoint_fits)
        if result is not None:
            return result
    return None


def _find_in_window(video, model, cfg, acfg, fps, start, *, hwaccel, keypoints,
                    keypoint_errors, keypoint_fits) -> FoundStadium | None:
    from bxjudge.autocalib import warp_calibration
    from bxjudge.video import VideoDecodeError, iter_frames

    try:
        step = max(1, round(cfg.sample_step_s * fps))
        samples = []
        for idx, frame in iter_frames(Path(video), start, start + cfg.sample_window_s,
                                      long_edge=_REF_LONG_EDGE, hwaccel=hwaccel):
            if idx % step == 0:
                mask = rail_mask(frame, cfg)
                samples.append((int((mask > 0).sum()), idx, frame, mask))
    except (VideoDecodeError, ValueError, OSError):
        return None
    samples.sort(key=lambda s: (-s[0], s[1]))

    def found(idx: int, frame: np.ndarray, fit: StadiumFit, agreeing: int,
              method: str) -> FoundStadium:
        fh, fw = frame.shape[:2]
        cal = warp_calibration(model.calibration, _alignment(fit, model, (fw, fh)), (fw, fh),
                               f"found-{Path(video).stem}")
        return FoundStadium(calibration=cal, fit=fit, frame_index=idx, frame=frame,
                            zone_visibility=zone_visibility(cal), agreeing_frames=agreeing,
                            method=method)

    if keypoints is not None:
        # A reference for another model is a setup error: raise, do not fall back silently.
        check_keypoint_reference(keypoints.reference, model)
        kfits: list[tuple[int, np.ndarray, StadiumFit, np.ndarray]] = []
        for pixels, idx, frame, mask in samples[:cfg.candidate_frames]:
            if pixels == 0:
                break
            try:
                pts = keypoints.predict(frame)
                fit = (None if pts is None
                       else fit_from_keypoints(pts, keypoints.reference, mask, model, cfg))
            except (RuntimeError, ValueError, MemoryError, cv2.error) as e:
                # One frame's inference or fit failed: skip it, the rail path still runs.
                _log.warning("%s frame %d: keypoint fit failed: %s: %s", Path(video).name, idx,
                             type(e).__name__, e)
                if keypoint_errors is not None:
                    keypoint_errors.append(f"frame {idx}: {type(e).__name__}: {e}")
                continue
            if fit is None:
                continue
            if keypoint_fits is not None:
                keypoint_fits.append((idx, frame, fit))
            zones = _zones(model, fit)
            kfits.append((idx, frame, fit, zones))
            agree = [f for f in kfits if float(np.linalg.norm(f[3] - zones, axis=1).mean())
                     * (_REF_LONG_EDGE / max(frame.shape[:2])) <= cfg.keypoint_agreement_px]
            if len(agree) >= 2:
                idx, frame, fit, _ = min(agree, key=lambda f: (f[2].keypoint_error_px, f[0]))
                return found(idx, frame, fit, len(agree), "keypoints")
    fits: list[tuple[int, np.ndarray, StadiumFit, np.ndarray]] = []
    for pixels, idx, frame, mask in samples[:cfg.candidate_frames]:
        if pixels == 0:
            break
        fit = (fit_stadium(mask, model, cfg, frame=frame, acfg=acfg) if cfg.edge_refinement
               else fit_stadium(mask, model, cfg))
        if fit is None:
            continue
        zones = _zones(model, fit)
        fits.append((idx, frame, fit, zones))
        agree = [f for f in fits
                 if float(np.linalg.norm(f[3] - zones, axis=1).mean()) <= cfg.agreement_px]
        if len(agree) >= 2:
            idx, frame, fit, _ = max(agree, key=lambda f: _rank(f[2]))
            return found(idx, frame, fit, len(agree), "rail")
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
        "image_scale": model.image_scale,
        "image": Path(path).with_suffix(".png").name if model.image is not None else None,
    }
    if model.image is not None:
        if not cv2.imwrite(str(Path(path).with_suffix(".png")), model.image):
            raise OSError(f"cannot write {Path(path).with_suffix('.png')}")
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")


def load_stadium_model(path: Path = DEFAULT_STADIUM_MODEL) -> StadiumModel:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != _MODEL_SCHEMA:
        raise ValueError(f"{path}: not a {_MODEL_SCHEMA} file")
    rail = np.array(doc["rail"], np.float64).reshape(-1, 2)
    w, h = doc["image_size"]
    cal = _calibration_from_doc(doc["calibration"])
    _check_simple_zones(cal, str(path))
    image = None
    if doc.get("image"):
        image = cv2.imread(str(Path(path).parent / doc["image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"{path}: cannot read its reference image {doc['image']}")
    return StadiumModel(rail=rail, calibration=cal,
                        source_sha256=str(doc["source_sha256"]), image_size=(int(w), int(h)),
                        image=image, image_scale=float(doc.get("image_scale", 1.0)))
