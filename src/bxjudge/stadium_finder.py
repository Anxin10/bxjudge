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
    agreement_px: float
    zone_visible_min: float


_INT_FIELDS = ("model_points", "hue_halfwidth", "min_saturation", "min_value",
               "candidate_frames", "work_width", "coarse_scale_steps", "coarse_top_k",
               "refine_iterations")
_FRACTIONS = ("roi_margin_frac", "max_thickness_frac", "min_component_frac", "min_coverage",
              "min_precision", "zone_visible_min")


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
    the rail is found under different lighting and cameras. Solid blobs (mean thickness above
    ``max_thickness_frac`` of the long edge, e.g. gloves) and specks are dropped."""
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
        if perimeter <= 0 or 2.0 * area / perimeter > max_thickness:
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
    ys, xs = np.nonzero(mask)
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
