"""Camera calibration: stadium zones as polygons for one fixed camera setup.

A calibration is a JSON file plus a reference-frame PNG. ``calibration_similarity``
compares a video frame with the reference frame (inside the stadium polygon only) so a
moved camera is detected; per-video selection lives in ``calibration_store.select_snapshot``.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from bxjudge.geometry import RIM_BLUR_SIGMA, RIM_MASK_MARGIN, rim_mask
from bxjudge.types import Calibration, Polygon

SIM_WIDTH = 320  # similarity is computed on frames downscaled to this width

# Click order in the GUI: (Calibration field, English label, Chinese label).
# OpenCV putText cannot draw CJK, so only the English label is drawn on screen.
ZONES: list[tuple[str, str, str]] = [
    ("stadium", "stadium (whole stadium incl. lid)", "場地（含蓋子）"),
    ("lid", "lid (upper cover)", "上盤／蓋子"),
    ("battle_area", "battle_area (lower floor)", "下盤戰鬥區"),
    ("over_left", "over_left", "擊飛區域（左）"),
    ("over_right", "over_right", "擊飛區域（右）"),
    ("xtreme", "xtreme", "極限區域"),
    ("launch_left", "launch_left", "左發射區"),
    ("launch_right", "launch_right", "右發射區"),
]
_POLY_FIELDS = [z[0] for z in ZONES]


# --------------------------------------------------------------------------- persistence
def save_calibration(c: Calibration, path: Path) -> None:
    """Write ``c`` as JSON. An absolute ``ref_frame`` is stored relative to the JSON's folder
    (a relative one is assumed to already be relative to it and is stored as-is)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(c)
    d["frame_size"] = list(c.frame_size)
    for f in _POLY_FIELDS:
        d[f] = [list(p) for p in getattr(c, f)]
    ref = c.ref_frame
    if ref and os.path.isabs(ref):
        ref = os.path.relpath(ref, os.path.abspath(path.parent))
    d["ref_frame"] = Path(ref).as_posix() if ref else ""
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def load_calibration(path: Path) -> Calibration:
    """Read a calibration JSON; ``ref_frame`` is resolved against the JSON's folder."""
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    ref = d.get("ref_frame", "")
    if ref and not os.path.isabs(ref):
        ref = os.path.normpath(os.path.join(os.path.abspath(path.parent), ref))
    polys: dict[str, Polygon] = {f: [(float(x), float(y)) for x, y in d[f]] for f in _POLY_FIELDS}
    w, h = d["frame_size"]
    return Calibration(name=d["name"], frame_size=(int(w), int(h)), ref_frame=ref, **polys)


def scale_calibration(c: Calibration, size: tuple[int, int]) -> Calibration:
    """``c`` with ``frame_size`` and all zone polygons scaled to decoded ``(width, height)``."""
    sx, sy = size[0] / c.frame_size[0], size[1] / c.frame_size[1]
    return replace(
        c,
        frame_size=size,
        **{name: [(x * sx, y * sy) for x, y in getattr(c, name)] for name in _POLY_FIELDS},
    )


# --------------------------------------------------------------------------- similarity
def _gray_small(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Lightly blurred gradient magnitude (insensitive to brightness gain/offset).

    The magnitude is IEEE float32 arithmetic, not ``cv2.magnitude``: OpenCV's IPP path gave
    the same input a different last bit depending on earlier calls, which moved an auto
    snapshot (and its id) with whatever the batch had run before."""
    blurred = cv2.GaussianBlur(gray, (0, 0), RIM_BLUR_SIGMA)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def calibration_similarity(frame: np.ndarray, c: Calibration) -> float:
    """0-1 similarity between ``frame`` and the calibration's reference frame.

    Compared region: the ``stadium`` polygon minus the ``battle_area`` polygon (rim and lid
    area). The beys, launchers and hands move inside battle_area, so it is excluded; the
    static rim is what shows whether the camera moved. Both images are converted to
    grayscale, resized to 320 px wide, lightly Gaussian-blurred and turned into
    Sobel gradient-magnitude maps; the score is the zero-mean normalised cross-correlation
    of those maps over the region, clipped to 0-1. Gradient maps make the score insensitive
    to brightness gain/offset and sensitive to a few-pixel camera shift, where raw
    intensity NCC on smooth footage stays high. Blur/Sobel parameters are implementation
    constants; the decision threshold is ``calibration_min_similarity`` in config.
    Tuning on real footage is deferred to the user.
    """
    ref = cv2.imread(c.ref_frame, cv2.IMREAD_COLOR) if c.ref_frame else None
    if ref is None:
        raise FileNotFoundError(f"calibration {c.name!r}: cannot read reference frame {c.ref_frame!r}")
    w, h = c.frame_size
    size = (SIM_WIDTH, max(1, round(SIM_WIDTH * h / w)))
    mask = rim_mask(c, size, RIM_MASK_MARGIN)
    if not mask.any():
        return 0.0
    a = _edge_map(_gray_small(frame, size))[mask]
    b = _edge_map(_gray_small(ref, size))[mask]
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-6:
        return 0.0
    return float(np.clip((a * b).sum() / denom, 0.0, 1.0))


# --------------------------------------------------------------------------- frame grab
class FrameGrabError(ValueError):
    """ffmpeg could not decode a reference frame from the given video."""


def grab_frame(video: Path, at: float) -> np.ndarray:
    """Decode the single frame at ``at`` seconds with ffmpeg (minimal; the real decoder is Task 5).

    A missing file is ``FileNotFoundError``; a file ffmpeg cannot read a frame from is
    ``FrameGrabError`` (a ``ValueError``: the user passed an unusable video)."""
    if not Path(video).is_file():
        raise FileNotFoundError(f"{video}: video file not found")
    cmd =["ffmpeg", "-v", "error", "-ss", f"{at}", "-i", str(video),
           "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"]
    r = subprocess.run(cmd, capture_output=True)
    img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR) if r.stdout else None
    if r.returncode != 0 or img is None:
        raise FrameGrabError(f"ffmpeg could not read a frame at {at}s from {video}:{r.stderr.decode(errors='replace')}")
    return img


# --------------------------------------------------------------------------- interactive GUI
class PolygonCollector:
    """GUI-independent state machine for clicking the polygons in ``ZONES`` order."""

    def __init__(self, zones: list[tuple[str, str, str]] = ZONES):
        self.zones = zones
        self.done: dict[str, Polygon] = {}
        self.points: list[tuple[float, float]] = []

    @property
    def finished(self) -> bool:
        return len(self.done) == len(self.zones)

    @property
    def current(self) -> str | None:
        return None if self.finished else self.zones[len(self.done)][1]

    def click(self, x: float, y: float) -> None:
        if not self.finished:
            self.points.append((float(x), float(y)))

    def undo(self) -> None:
        if self.points:
            self.points.pop()

    def commit(self) -> bool:
        """Finish the current polygon (needs >= 3 points). Returns True if accepted."""
        if self.finished or len(self.points) < 3:
            return False
        self.done[self.zones[len(self.done)][0]] = self.points
        self.points = []
        return True


def interactive_calibrate(frame: np.ndarray, name: str, max_display_width: int = 1280) -> Calibration:
    """OpenCV window: click the polygons in ``ZONES`` order. Enter = finish polygon,
    ``u`` = undo last point, Esc = abort (raises ``KeyboardInterrupt``).
    The returned calibration has an empty ``ref_frame``; the caller saves the frame."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_display_width / w)
    base = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1.0 else frame.copy()
    col = PolygonCollector()
    win = f"calibrate {name}"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            col.click(x / scale, y / scale)

    def draw() -> np.ndarray:
        img = base.copy()
        shapes = [(p, True, (0, 200, 0)) for p in col.done.values()] + [(col.points, False, (0, 200, 255))]
        for pts, closed, color in shapes:
            if not pts:
                continue
            arr = np.round(np.array(pts) * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [arr], closed, color, 2)
            for q in arr.reshape(-1, 2):
                cv2.circle(img, (int(q[0]), int(q[1])), 3, color, -1)
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        msg = (
            "all done"
            if col.finished
            else f"[{len(col.done) + 1}/{len(col.zones)}] click: {col.current}   (Enter=finish polygon, u=undo, Esc=abort)"
        )
        cv2.putText(img, msg, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while not col.finished:
            cv2.imshow(win, draw())
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):
                col.commit()
            elif key == ord("u"):
                col.undo()
            elif key == 27:
                raise KeyboardInterrupt("calibration aborted")
    finally:
        cv2.destroyWindow(win)
    return Calibration(name=name, frame_size=(w, h), ref_frame="", **col.done)
