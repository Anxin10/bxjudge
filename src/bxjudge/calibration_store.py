"""Immutable, content-addressed calibration snapshots and per-video calibration selection.

A snapshot is ``<results>/calibrations/<id>.json`` plus ``<id>.png`` (its reference image).
``id`` is the first 16 hex of SHA-256 over the snapshot's canonical JSON and the PNG bytes, so
identical content always gets the same id and a later edit of ``calib/`` never changes what a
result used (spec §6.4). The JSON is a normal calibration file (``calibration.load_calibration``
reads it; ``ref_frame`` is stored relative to it so a moved results folder still loads) plus a
``snapshot`` block with the kind, score, template id, manual source hash and homography.

``select_snapshot`` picks the calibration for a video's first-second frame: a manual calibration
with enough similarity first (used unrefined, spec §6.5c), else an existing auto snapshot at the
same camera position (the grouping, spec §6.3), else a new auto snapshot made by aligning the
best manual template to the frame (spec §6.1/§6.2b: its reference image is the frame itself),
else ``None`` (the video is ``recalibrate``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bxjudge.autocalib import AutoCalibConfig, estimate_alignment, warp_calibration
from bxjudge.calibration import ZONES, calibration_similarity, load_calibration
from bxjudge.config import Thresholds
from bxjudge.types import Calibration

log = logging.getLogger(__name__)

_POLY_FIELDS = [z[0] for z in ZONES]
_ID_LEN = 16
_TEMPLATE_SCORE = 1.0  # a template snapshot is compared with its own reference image
_KINDS = ("manual", "auto")
_LOAD_ERRORS = (OSError, ValueError, KeyError, TypeError, cv2.error)
_PALETTE = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255),
            (0, 165, 255), (255, 128, 0), (0, 0, 255), (255, 0, 0)]  # one BGR colour per zone


@dataclass(frozen=True)
class Snapshot:
    id: str
    kind: str  # "manual" | "auto"
    calibration: Calibration  # ref_frame = absolute path of <id>.png
    score: float  # manual: similarity to the frame; auto: alignment score
    template_id: str | None  # auto: snapshot id of the manual template it was derived from
    source_path: str | None  # manual: the calib/ JSON it was copied from
    source_sha256: str | None  # manual: SHA-256 of that JSON's bytes
    homography: tuple | None  # auto: 3x3 template px -> frame px


def _cal_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "calibrations"


def _canonical(kind: str, c: Calibration, score: float, template_id: str | None,
               source_sha256: str | None, homography: tuple | None) -> str:
    d = {
        "kind": kind,
        "name": c.name,
        "frame_size": [int(c.frame_size[0]), int(c.frame_size[1])],
        "zones": {f: [[round(float(x), 2), round(float(y), 2)] for x, y in getattr(c, f)]
                  for f in _POLY_FIELDS},
        "score": round(float(score), 4),
        "template_id": template_id,
        "source_sha256": source_sha256,
        "homography": None if homography is None
        else [[round(float(v), 9) for v in row] for row in homography],
    }
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def save_snapshot(out_dir: Path, kind: str, calibration: Calibration, ref_image: np.ndarray, *,
                  score: float, template_id: str | None = None, source_path: Path | None = None,
                  homography: tuple | None = None) -> Snapshot:
    """Write the snapshot of ``calibration`` with reference image ``ref_image`` and return it.
    Idempotent: identical content gives the same id and an existing snapshot is never rewritten."""
    if kind not in _KINDS:
        raise ValueError(f"snapshot kind must be one of {_KINDS}, got {kind!r}")
    ok, buf = cv2.imencode(".png", ref_image)
    if not ok:
        raise ValueError("cannot encode the reference image as PNG")
    png = buf.tobytes()
    sha = hashlib.sha256(Path(source_path).read_bytes()).hexdigest() if source_path is not None else None
    hom = None if homography is None else tuple(tuple(float(v) for v in row) for row in homography)
    canon = _canonical(kind, calibration, score, template_id, sha, hom)
    sid = hashlib.sha256(canon.encode("utf-8") + png).hexdigest()[:_ID_LEN]
    d = _cal_dir(out_dir)
    jpath, ppath = d / f"{sid}.json", d / f"{sid}.png"
    if not jpath.is_file():
        d.mkdir(parents=True, exist_ok=True)
        ppath.write_bytes(png)
        doc = {
            "name": calibration.name,
            "frame_size": [int(calibration.frame_size[0]), int(calibration.frame_size[1])],
            **{f: [[float(x), float(y)] for x, y in getattr(calibration, f)] for f in _POLY_FIELDS},
            "ref_frame": ppath.name,
            "snapshot": {
                "id": sid, "kind": kind, "score": float(score), "template_id": template_id,
                "source_path": None if source_path is None else str(source_path),
                "source_sha256": sha,
                "homography": None if hom is None else [list(r) for r in hom],
            },
        }
        tmp = d / f"{sid}.json.tmp"  # the JSON appears last, so a snapshot is never half-written
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, jpath)
    return load_snapshot(out_dir, sid)


def load_snapshot(out_dir: Path, snapshot_id: str) -> Snapshot:
    """Read snapshot ``snapshot_id`` from ``<out_dir>/calibrations``; ``ref_frame`` is absolute.
    Raises ``FileNotFoundError`` for an unknown id."""
    if not snapshot_id or Path(snapshot_id).name != snapshot_id:
        raise FileNotFoundError(f"unknown calibration snapshot {snapshot_id!r}")
    jpath = _cal_dir(out_dir) / f"{snapshot_id}.json"
    if not jpath.is_file():
        raise FileNotFoundError(f"unknown calibration snapshot {snapshot_id!r}: {jpath}")
    meta = json.loads(jpath.read_text(encoding="utf-8"))["snapshot"]
    hom = meta.get("homography")
    return Snapshot(
        id=meta["id"], kind=meta["kind"], calibration=load_calibration(jpath),
        score=float(meta["score"]), template_id=meta.get("template_id"),
        source_path=meta.get("source_path"), source_sha256=meta.get("source_sha256"),
        homography=None if hom is None else tuple(tuple(float(v) for v in r) for r in hom),
    )


def _manual_templates(calib_dir: Path) -> list[tuple[Path, Calibration, np.ndarray]]:
    """Readable manual calibrations in ``calib_dir`` with their reference images; broken ones
    are skipped with a warning."""
    out = []
    for p in sorted(Path(calib_dir).glob("*.json")):
        try:
            c = load_calibration(p)
            img = cv2.imread(c.ref_frame, cv2.IMREAD_COLOR) if c.ref_frame else None
            if img is None:
                raise FileNotFoundError(f"cannot read reference frame {c.ref_frame!r}")
        except _LOAD_ERRORS as e:
            log.warning("skipping calibration %s: %s", p, e)
            continue
        out.append((p, c, img))
    return out


def _existing_auto_snapshots(out_dir: Path) -> list[Snapshot]:
    snaps = []
    for p in sorted(_cal_dir(out_dir).glob("*.json")):
        try:
            s = load_snapshot(out_dir, p.stem)
        except _LOAD_ERRORS as e:
            log.warning("skipping calibration snapshot %s: %s", p, e)
            continue
        if s.kind == "auto":
            snaps.append(s)
    return snaps


def select_snapshot(frame: np.ndarray, calib_dir: Path, out_dir: Path, th: Thresholds,
                    cfg: AutoCalibConfig) -> Snapshot | None:
    """Calibration snapshot for ``frame`` (a video's first-second frame) or ``None``.

    Order: (1) the best manual calibration in ``calib_dir`` with similarity >=
    ``th.calibration_min_similarity``; (2) the best existing auto snapshot in ``out_dir`` with
    similarity >= that threshold (same camera position = same snapshot); (3) a new auto snapshot
    from the template with the best alignment score >= ``cfg.min_score``, whose reference image
    is ``frame`` itself; (4) ``None``. Broken templates are skipped with a warning."""
    templates = _manual_templates(calib_dir)

    best_manual: tuple[Path, Calibration, np.ndarray] | None = None
    best_sim = -1.0
    for t in templates:
        try:
            sim = calibration_similarity(frame, t[1])
        except _LOAD_ERRORS as e:
            log.warning("skipping calibration %s: %s", t[0], e)
            continue
        if sim >= th.calibration_min_similarity and sim > best_sim:
            best_manual, best_sim = t, sim
    if best_manual is not None:
        path, cal, img = best_manual
        return save_snapshot(out_dir, "manual", cal, img, score=best_sim, source_path=path)

    best_auto: Snapshot | None = None
    best_sim = -1.0
    for s in _existing_auto_snapshots(out_dir):
        try:
            sim = calibration_similarity(frame, s.calibration)
        except _LOAD_ERRORS as e:
            log.warning("skipping calibration snapshot %s: %s", s.id, e)
            continue
        if sim >= th.calibration_min_similarity and sim > best_sim:
            best_auto, best_sim = s, sim
    if best_auto is not None:
        return best_auto

    best_t = None
    best_al = None
    for t in templates:
        al = estimate_alignment(frame, t[1], t[2], cfg)
        if al is not None and al.score >= cfg.min_score and (best_al is None or al.score > best_al.score):
            best_t, best_al = t, al
    if best_t is None or best_al is None:
        return None
    path, cal, img = best_t
    template = save_snapshot(out_dir, "manual", cal, img, score=_TEMPLATE_SCORE, source_path=path)
    h, w = frame.shape[:2]
    warped = warp_calibration(cal, best_al, (w, h), f"auto-{cal.name}")
    return save_snapshot(out_dir, "auto", warped, frame, score=best_al.score,
                         template_id=template.id, homography=best_al.homography)


def write_preview(out_dir: Path, snap: Snapshot) -> Path:
    """``<out_dir>/calibrations/<id>.preview.jpg``: the reference image with all 8 zones drawn
    and a caption ``"{kind} {name} template={template_id} score={score:.3f}"``."""
    img = cv2.imread(snap.calibration.ref_frame, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read reference frame {snap.calibration.ref_frame!r}")
    for (field, _en, _zh), color in zip(ZONES, _PALETTE):
        poly = getattr(snap.calibration, field)
        if len(poly) >= 2:
            pts = np.round(np.array(poly)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    text = f"{snap.kind} {snap.calibration.name} template={snap.template_id} score={snap.score:.3f}"
    scale = max(0.5, img.shape[1] / 1280.0 * 0.8)
    thick = max(1, round(scale * 2))
    (tw, text_h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (0, 0), (min(img.shape[1], tw + 16), text_h + base + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (8, text_h + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick,
                cv2.LINE_AA)
    path = _cal_dir(out_dir) / f"{snap.id}.preview.jpg"
    if not cv2.imwrite(str(path), img):
        raise OSError(f"cannot write preview {path}")
    return path
