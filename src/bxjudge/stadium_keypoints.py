"""The 10 stadium keypoints: names, the template reference points and the trained ensemble.

Plan ``docs/superpowers/plans/2026-11-11-stadium-keypoints.md`` (K2, K4). The points sit on
features the zones are drawn from (cat-ear notch, Xtreme opening, Over pockets, inner corners),
so a homography from the template's points to a frame's points carries the template zones
into that frame (``stadium_finder.fit_from_keypoints``).

Left/right follow the stadium (user ruling 2026-11-11): with the cat ears facing the viewer,
the stadium's left is X. ``FLIP_IDX`` pairs every left point with its right partner, so a
mirrored image's prediction maps back to the right names.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bxjudge import PROJECT_ROOT

KEYPOINTS: tuple[str, ...] = (
    "notch_left", "notch_right", "xtreme_left", "xtreme_right", "over_left_inner",
    "over_right_inner", "field_left_upper", "field_left_lower", "field_right_upper",
    "field_right_lower")
FLIP_IDX: tuple[int, ...] = (1, 0, 3, 2, 5, 4, 8, 9, 6, 7)

DEFAULT_KEYPOINT_REFERENCE = PROJECT_ROOT / "config" / "stadium_keypoints_reference.json"
_REFERENCE_SCHEMA = "bxjudge-stadium-keypoints-reference-v2"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_INFER_LONG_EDGE = 1280  # inference px: spread limits are given at this long edge


@dataclass(frozen=True)
class KeypointReference:
    """The 10 points on the stadium model's template frame (template px).

    ``model_sha256`` is the ``StadiumModel.source_sha256`` of the model whose template frame
    the points were labelled on: the points only mean something with that model's zones.
    ``image_sha256`` records the labelled image itself (a JPEG copy of the template frame)."""
    points: dict[str, tuple[float, float]]
    template: str  # calibration name of the template (the stadium model's template)
    image_size: tuple[int, int]  # (width, height) of the template frame
    model_sha256: str
    image_sha256: str
    source: str = ""  # where the points came from (labelling project, task, annotation)

    def array(self) -> np.ndarray:
        """(10, 2) float64 in ``KEYPOINTS`` order."""
        return np.array([self.points[k] for k in KEYPOINTS], np.float64)


def save_keypoint_reference(ref: KeypointReference, path: Path) -> None:
    doc = {"schema": _REFERENCE_SCHEMA, "template": ref.template,
           "image_size": list(ref.image_size), "model_sha256": ref.model_sha256,
           "image_sha256": ref.image_sha256, "source": ref.source,
           "points": {k: [float(ref.points[k][0]), float(ref.points[k][1])] for k in KEYPOINTS}}
    Path(path).write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")


def load_keypoint_reference(path: Path = DEFAULT_KEYPOINT_REFERENCE) -> KeypointReference:
    """Read and validate a reference: exactly the 10 points, each two finite numbers inside
    the frame, a positive integer frame size and SHA-256 identities. Raises ``ValueError``."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != _REFERENCE_SCHEMA:
        raise ValueError(f"{path}: not a {_REFERENCE_SCHEMA} file")
    size = doc.get("image_size")
    if (not isinstance(size, list) or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)):
        raise ValueError(f"{path}: image_size must be two positive integers, got {size!r}")
    w, h = size
    for key in ("model_sha256", "image_sha256"):
        if not isinstance(doc.get(key), str) or not _SHA256.fullmatch(doc[key]):
            raise ValueError(f"{path}: {key} must be a lowercase SHA-256 hex digest")
    if not isinstance(doc.get("template"), str) or not doc["template"]:
        raise ValueError(f"{path}: template must name the model's calibration")
    pts = doc.get("points")
    if not isinstance(pts, dict):
        raise ValueError(f"{path}: points must be a mapping")
    missing = [k for k in KEYPOINTS if k not in pts]
    unknown = sorted(set(pts) - set(KEYPOINTS))
    if missing or unknown:
        raise ValueError(f"{path}: missing points {missing}, unknown points {unknown}")
    points: dict[str, tuple[float, float]] = {}
    for k in KEYPOINTS:
        p = pts[k]
        if (not isinstance(p, list) or len(p) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in p)
                or not all(math.isfinite(v) for v in p)
                or not (0 <= p[0] < w and 0 <= p[1] < h)):
            raise ValueError(f"{path}: point {k} must be [x, y] inside the {w}x{h} frame, "
                             f"got {p!r}")
        points[k] = (float(p[0]), float(p[1]))
    return KeypointReference(points=points, template=doc["template"], image_size=(w, h),
                             model_sha256=doc["model_sha256"], image_sha256=doc["image_sha256"],
                             source=str(doc.get("source", "")))


def merge_flipped(direct: list[np.ndarray], flipped: list[np.ndarray], width: int, *,
                  min_conf: float = 0.0, max_spread: float = math.inf) -> np.ndarray | None:
    """Combine (10, 3) predictions [x, y, confidence] of the frame and of its mirror image.

    A prediction on the mirrored frame names the mirrored stadium's points: its left notch is
    the real right notch. It is mapped back by swapping the ``FLIP_IDX`` pairs and mirroring x
    (``width - x``, continuous coordinates as Ultralytics uses them).

    Per point, the position is the mean of the predictions at least ``min_conf`` confident
    (a confident point is never pulled away by unsure ones) and the confidence is the mean
    over all predictions (the ensemble's agreement that the point is visible). A point with no
    confident prediction, or whose confident predictions lie more than ``max_spread`` (frame
    px) from their mean, gets confidence 0: the models do not agree where it is. ``None`` when
    there is no prediction at all."""
    acc = [np.asarray(p, np.float64).reshape(10, 3) for p in direct]
    for p in flipped:
        q = np.asarray(p, np.float64).reshape(10, 3)[list(FLIP_IDX)].copy()
        q[:, 0] = width - q[:, 0]
        acc.append(q)
    if not acc:
        return None
    stack = np.stack(acc)  # (members, 10, 3)
    out = np.zeros((10, 3), np.float64)
    for j in range(10):
        pts = stack[:, j]
        good = np.isfinite(pts).all(1) & (pts[:, 2] >= min_conf)
        if not good.any():
            out[j] = (np.nan, np.nan, 0.0)
            continue
        xy = pts[good, :2].mean(0)
        spread = float(np.linalg.norm(pts[good, :2] - xy, axis=1).max())
        conf = float(np.nan_to_num(pts[:, 2]).mean())
        out[j] = (xy[0], xy[1], conf if spread <= max_spread else 0.0)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_weights(paths) -> list[Path]:
    """Project-relative weight paths resolved (symlinks followed); each must be a file that
    stays inside the project. Raises ``ValueError`` / ``FileNotFoundError``."""
    if not paths:
        raise ValueError("no keypoint weights configured")
    root = PROJECT_ROOT.resolve()
    out = []
    for p in paths:
        real = (PROJECT_ROOT / p).resolve()
        if real != root and root not in real.parents:
            raise ValueError(f"keypoint weight {p} resolves outside the project: {real}")
        if not real.is_file():
            raise FileNotFoundError(f"keypoint weight not found: {real}")
        out.append(real)
    return out


class KeypointFinder:
    """The trained keypoint ensemble plus the template reference.

    Every model predicts the frame and its mirror image; the most confident stadium of each
    prediction is kept and the predictions are combined by ``merge_flipped``. On the 24
    validation images this halves the per-model scatter (median point error 21 px for one
    model, 12 px for four models with mirroring, 1920 px frames; ledger 2026-11-12)."""

    def __init__(self, weights: list[Path], imgsz: int, reference: KeypointReference, *,
                 min_conf: float, max_spread_px: float, device: str | int | None = None):
        from ultralytics import YOLO

        if not weights:
            raise ValueError("no keypoint weights")
        missing = [str(w) for w in weights if not Path(w).is_file()]
        if missing:
            raise FileNotFoundError(f"keypoint weights not found: {missing}")
        self.weights = [Path(w) for w in weights]
        self.weights_sha256 = [_sha256(w) for w in self.weights]
        self.models = [YOLO(str(w)) for w in self.weights]
        self.imgsz = imgsz
        self.reference = reference
        self.min_conf = min_conf
        self.max_spread_px = max_spread_px  # inference px
        self.device = device

    @classmethod
    def from_config(cls, cfg, reference: KeypointReference | None = None,
                    device: str | int | None = None) -> "KeypointFinder":
        """Weights, input size and merge limits from ``StadiumFinderConfig``, reference from
        ``config/stadium_keypoints_reference.json``."""
        return cls(resolve_weights(cfg.keypoint_weights), cfg.keypoint_imgsz,
                   reference or load_keypoint_reference(), min_conf=cfg.keypoint_min_conf,
                   max_spread_px=cfg.keypoint_max_spread_px, device=device)

    def _one(self, model, image: np.ndarray) -> np.ndarray | None:
        r = model.predict(image, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return None
        i = int(r.boxes.conf.argmax())
        xy = r.keypoints.xy[i].cpu().numpy().astype(np.float64)
        conf = r.keypoints.conf[i].cpu().numpy().astype(np.float64)
        return np.c_[xy, conf]

    def predict(self, frame: np.ndarray) -> np.ndarray | None:
        """(10, 3) [x, y, confidence] in ``frame`` px (BGR image), ``KEYPOINTS`` order, or
        ``None`` when no model sees a stadium."""
        frame = np.ascontiguousarray(frame)
        mirror = np.ascontiguousarray(frame[:, ::-1])
        direct, flipped = [], []
        for m in self.models:
            p = self._one(m, frame)
            if p is not None:
                direct.append(p)
            q = self._one(m, mirror)
            if q is not None:
                flipped.append(q)
        spread = self.max_spread_px * max(frame.shape[:2]) / _INFER_LONG_EDGE
        return merge_flipped(direct, flipped, frame.shape[1], min_conf=self.min_conf,
                             max_spread=spread)
