"""Immutable, content-addressed calibration snapshots and per-video calibration selection.

A snapshot is ``<results>/calibrations/<id>.json`` plus ``<id>.png`` (its reference image).
``id`` is the first 16 hex of SHA-256 over the snapshot's canonical JSON and the PNG bytes, so
identical content always gets the same id and a later edit of ``calib/`` never changes what a
result used (spec §6.4). A manual snapshot's identity is its polygons, name, size, source-file
hash and image: the per-video similarity is *not* part of it (it is returned on the Snapshot and
recorded per video), so one manual calibration has exactly one id. An auto snapshot's alignment
score is fixed at creation, so it is part of its identity, as is its
``template_set`` (ruling F-2): the sorted manual snapshot ids of every readable template in
``calib/`` when it was created. The JSON is a normal calibration file
(``calibration.load_calibration`` reads it; ``ref_frame`` is stored relative to it so a moved
results folder still loads) plus a ``snapshot`` block with the kind, score, template id,
template set, manual source hash and homography.

``select_snapshot`` picks the calibration for a video's first-second frame: a manual calibration
with enough similarity first (used unrefined, spec §6.5c), else the auto snapshot made by
aligning every manual template to the frame and keeping the best (spec §6.1/§6.2b: its
reference image is the frame itself; spec §6.3: every video is aligned on its own frame, so its
calibration never depends on the other videos in the batch), else ``None`` (``recalibrate``).

A reviewer can veto an auto snapshot (``veto_calibration``; spec section 6.5, ruling T4-1): the
veto lives in ``<results>/calibration_vetoes.json``, keyed by snapshot id, and is honoured by
selection and by every ``matches.csv`` rebuild; it never rewrites a judged row.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from bxjudge.autocalib import (
    AutoCalibConfig,
    estimate_alignment,
    warp_calibration,
)
from bxjudge.calibration import ZONES, calibration_similarity, load_calibration
from bxjudge.config import Thresholds
from bxjudge.types import Calibration

log = logging.getLogger(__name__)

_POLY_FIELDS = [z[0] for z in ZONES]
_ID_LEN = 16
_MANUAL_STORED_SCORE = 1.0  # stored in a manual snapshot's file: its own image vs itself
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
    # auto: sorted manual snapshot ids of all readable calib/ templates at creation (F-2)
    template_set: tuple[str, ...] | None = None


def _cal_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "calibrations"


def _canonical(kind: str, c: Calibration, score: float, template_id: str | None,
               source_sha256: str | None, homography: tuple | None,
               template_set: tuple[str, ...] | None) -> str:
    d = {
        "kind": kind,
        "name": c.name,
        "frame_size": [int(c.frame_size[0]), int(c.frame_size[1])],
        "zones": {f: [[round(float(x), 2), round(float(y), 2)] for x, y in getattr(c, f)]
                  for f in _POLY_FIELDS},
        "score": round(float(score), 4) if kind == "auto" else None,
        "template_id": template_id,
        "template_set": None if template_set is None else list(template_set),
        "source_sha256": source_sha256,
        "homography": None if homography is None
        else [[round(float(v), 9) for v in row] for row in homography],
    }
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("cannot encode the reference image as PNG")
    return buf.tobytes()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _snapshot_id(kind: str, c: Calibration, score: float, template_id: str | None,
                 source_sha256: str | None, homography: tuple | None, png: bytes,
                 template_set: tuple[str, ...] | None = None) -> str:
    canon = _canonical(kind, c, score, template_id, source_sha256, homography, template_set)
    return hashlib.sha256(canon.encode("utf-8") + png).hexdigest()[:_ID_LEN]


def save_snapshot(out_dir: Path, kind: str, calibration: Calibration, ref_image: np.ndarray, *,
                  score: float, template_id: str | None = None, source_path: Path | None = None,
                  homography: tuple | None = None,
                  template_set: Iterable[str] | None = None,
                  autocalib: AutoCalibConfig) -> Snapshot:
    """Write the snapshot of ``calibration`` with reference image ``ref_image`` and return it.
    Idempotent: identical content gives the same id and an existing snapshot is never rewritten.
    For a manual snapshot ``score`` (the per-video similarity) is not part of the identity: it is
    returned on the Snapshot but the stored file keeps a fixed value. ``template_set`` (auto
    only, stored sorted) is part of the identity (ruling F-2). ``autocalib`` supplies the shape
    bounds the stored file is re-read with (``load_snapshot``)."""
    if kind not in _KINDS:
        raise ValueError(f"snapshot kind must be one of {_KINDS}, got {kind!r}")
    if kind == "manual" and template_set is not None:
        raise ValueError("a manual snapshot has no template_set")
    tset = None if template_set is None else tuple(sorted(set(template_set)))
    png = _encode_png(ref_image)
    sha = _sha256_file(source_path) if source_path is not None else None
    hom = None if homography is None else tuple(tuple(float(v) for v in row) for row in homography)
    sid = _snapshot_id(kind, calibration, score, template_id, sha, hom, png, tset)
    stored_score = float(score) if kind == "auto" else _MANUAL_STORED_SCORE
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
                "id": sid, "kind": kind, "score": stored_score, "template_id": template_id,
                "template_set": None if tset is None else list(tset),
                "source_path": None if source_path is None else str(source_path),
                "source_sha256": sha,
                "homography": None if hom is None else [list(r) for r in hom],
            },
        }
        tmp = d / f"{sid}.json.tmp"  # the JSON appears last, so a snapshot is never half-written
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, jpath)
    snap = load_snapshot(out_dir, sid, autocalib=autocalib)
    return replace(snap, score=float(score)) if kind == "manual" else snap


def load_snapshot(out_dir: Path, snapshot_id: str, *, autocalib: AutoCalibConfig) -> Snapshot:
    """Read snapshot ``snapshot_id`` from ``<out_dir>/calibrations``; ``ref_frame`` is absolute.
    Raises ``FileNotFoundError`` for an unknown id and ``ValueError`` for a malformed file.
    The shape bounds (``snapshot_max_frame_side``, ``snapshot_max_coord``) come from
    ``autocalib`` (R40), which the caller loads from an explicit path: it is never resolved
    from the current directory here (ruling C-X3(b)), so a missing config is the caller's
    global error, never an unknown snapshot."""
    if not snapshot_id or Path(snapshot_id).name != snapshot_id:
        raise FileNotFoundError(f"unknown calibration snapshot {snapshot_id!r}")
    jpath = _cal_dir(out_dir) / f"{snapshot_id}.json"
    if not jpath.is_file():
        raise FileNotFoundError(f"unknown calibration snapshot {snapshot_id!r}: {jpath}")
    try:
        doc = json.loads(jpath.read_text(encoding="utf-8"))
        meta = _checked_meta(doc, autocalib)
        calibration = load_calibration(jpath)
        if not calibration.ref_frame or not Path(calibration.ref_frame).is_file():
            raise ValueError(f"missing reference image {calibration.ref_frame!r}")
        hom = meta.get("homography")
        tset = meta.get("template_set")
        return Snapshot(
            id=meta["id"], kind=meta["kind"], calibration=calibration,
            score=float(meta["score"]), template_id=meta.get("template_id"),
            source_path=meta.get("source_path"), source_sha256=meta.get("source_sha256"),
            homography=None if hom is None else tuple(tuple(float(v) for v in r) for r in hom),
            template_set=None if tset is None else tuple(str(t) for t in tset),
        )
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as e:
        raise ValueError(f"{jpath}: malformed calibration snapshot ({e!r})") from None


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_frame_side(v, limit: int) -> bool:
    """A finite whole number in ``1..limit`` (safe for ``int()``)."""
    return _is_number(v) and math.isfinite(v) and 1 <= v <= limit and v == int(v)


def _is_coord(v, limit: float) -> bool:
    return _is_number(v) and math.isfinite(v) and abs(v) <= limit


def _checked_meta(doc, cfg: AutoCalibConfig) -> dict:
    """The ``snapshot`` block of a parsed snapshot file after a shape check; raises ValueError."""
    if not isinstance(doc, dict):
        raise TypeError("top level is not an object")
    meta = doc.get("snapshot")
    if not isinstance(meta, dict):
        raise TypeError("'snapshot' is not an object")
    if not isinstance(meta.get("id"), str):
        raise TypeError("snapshot id is not a string")
    if meta.get("kind") not in _KINDS:
        raise ValueError(f"unknown kind {meta.get('kind')!r}")
    if not _is_number(meta.get("score")):
        raise ValueError("score is not a number")
    for key in ("template_id", "source_path", "source_sha256"):
        if meta.get(key) is not None and not isinstance(meta[key], str):
            raise ValueError(f"{key} is not a string")
    hom = meta.get("homography")
    if hom is not None and not (
        isinstance(hom, list) and len(hom) == 3
        and all(isinstance(r, list) and len(r) == 3 and all(map(_is_number, r)) for r in hom)
    ):
        raise ValueError("homography is not a 3x3 matrix of numbers")
    tset = meta.get("template_set")
    if tset is not None and not (isinstance(tset, list) and all(isinstance(t, str) for t in tset)):
        raise ValueError("template_set is not a list of strings")
    size = doc.get("frame_size")
    if not (isinstance(size, list) and len(size) == 2 and all(_is_frame_side(v, cfg.snapshot_max_frame_side) for v in size)):
        raise ValueError("frame_size is not [width, height] of positive whole pixel counts")
    if not isinstance(doc.get("name"), str):
        raise TypeError("name is not a string")
    for f in _POLY_FIELDS:
        poly = doc.get(f)
        if not (isinstance(poly, list) and all(
                isinstance(pt, list) and len(pt) == 2
                and all(_is_coord(v, cfg.snapshot_max_coord) for v in pt)
                for pt in poly)):
            raise ValueError(f"{f} is not a list of [x, y] points")
    return meta


def validate_output_paths(out_dir: Path, calib_dir: Path) -> None:
    """Raise ``ValueError`` when ``out_dir`` and ``calib_dir`` overlap (resolved equal, or one
    inside the other, including through a ``calibrations`` symlink) or ``out_dir`` is a file.
    Called before anything is written, so ``calib/`` is never modified by a run."""
    out = Path(out_dir).expanduser()
    if out.is_file():
        raise ValueError(f"{out}: --out is a file, expected a results folder")
    source = Path(calib_dir).expanduser().resolve()
    output = out.resolve()
    snapshots = (output / "calibrations").resolve()
    for target in (output, snapshots):
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError(f"{out}: --out must be outside --calib {calib_dir} "
                             "(equal or nested folders would write results into calibrations)")


class _Template(NamedTuple):
    path: Path
    cal: Calibration
    img: np.ndarray
    id: str  # id of this template's manual snapshot (as ``select_snapshot`` returns it)


def _manual_templates(calib_dir: Path) -> list[_Template]:
    """Readable manual calibrations in ``calib_dir`` with their reference images and manual
    snapshot ids; broken ones are skipped with a warning."""
    out = []
    for p in sorted(Path(calib_dir).glob("*.json")):
        try:
            c = load_calibration(p)
            img = cv2.imread(c.ref_frame, cv2.IMREAD_COLOR) if c.ref_frame else None
            if img is None:
                raise FileNotFoundError(f"cannot read reference frame {c.ref_frame!r}")
            sid = _snapshot_id("manual", c, _MANUAL_STORED_SCORE, None, _sha256_file(p), None,
                               _encode_png(img))
        except _LOAD_ERRORS as e:
            log.warning("skipping calibration %s: %s", p, e)
            continue
        out.append(_Template(p, c, img, sid))
    return out


def _barred_templates(out_dir: Path, video: str | None,
                      vetoed: AbstractSet[str], template_ids: set[str],
                      cfg: AutoCalibConfig) -> set[str]:
    """All active template vetoes explicitly bound to this video, across source changes."""
    if not vetoed:
        return set()
    if not isinstance(video, str) or not video:
        raise ValueError("video identity is required when selecting with calibration vetoes")
    entries = _read_vetoes(out_dir)
    barred = set()
    for sid in sorted(vetoed):
        if sid not in entries:
            raise ValueError(f"veto {sid} has no stored video scope")
        if video not in entries[sid]["videos"]:
            continue
        snap = load_snapshot(out_dir, sid, autocalib=cfg)
        if snap.kind != "auto":
            raise ValueError(f"vetoed calibration {sid} is not auto")
        if snap.template_id in template_ids:
            if cv2.imread(snap.calibration.ref_frame, cv2.IMREAD_COLOR) is None:
                raise ValueError(f"unreadable veto reference image {snap.calibration.ref_frame!r}")
            barred.add(snap.template_id)
    return barred


def select_snapshot(frame: np.ndarray, calib_dir: Path, out_dir: Path, th: Thresholds,
                    cfg: AutoCalibConfig,
                    vetoed: AbstractSet[str] = frozenset(), *,
                    video: str | None = None) -> Snapshot | None:
    """Calibration snapshot for ``frame`` (a video's first-second frame) or ``None``.

    Order: (1) the best manual calibration in ``calib_dir`` with similarity >=
    ``th.calibration_min_similarity``; (2) the auto snapshot of this frame: every template is
    aligned to ``frame`` and the one with the best alignment score >= ``cfg.min_score`` wins;
    its reference image is ``frame`` itself and it records the current template set (ruling
    F-2: adding, removing or editing any template gives a new id); (3) ``None``. Broken
    templates are skipped with a warning. Auto snapshots other videos created in ``out_dir``
    are never reused, so a video's calibration depends only on its own frame and ``calib_dir``,
    never on which videos the batch (or an earlier run into the same folder) judged first.

    ``vetoed`` (ruling T4-1; vetoes apply to auto snapshots only, manual selection is unchanged):
    an alignment whose id is vetoed gives ``None``. Every active veto explicitly bound to
    ``video`` bars its template, including vetoes older than the latest selection or source
    replacement. Reference pixels and camera similarity never determine veto scope.
    Barring follows the template itself (its id
    still present in ``calib_dir``), not the template set: a template added elsewhere does not
    re-create a vetoed video's snapshot from the same template under a new id. Historical
    template barring is video-specific; exact snapshot-id vetoes still affect every video
    referencing that identical id (spec 6.3/6.5)."""
    validate_output_paths(out_dir, calib_dir)
    templates = _manual_templates(calib_dir)
    template_set = tuple(sorted({t.id for t in templates}))

    best_manual: _Template | None = None
    best_sim = -1.0
    for t in templates:
        try:
            sim = calibration_similarity(frame, t.cal)
        except _LOAD_ERRORS as e:
            log.warning("skipping calibration %s: %s", t.path, e)
            continue
        if sim >= th.calibration_min_similarity and sim > best_sim:
            best_manual, best_sim = t, sim
    if best_manual is not None:
        return save_snapshot(out_dir, "manual", best_manual.cal, best_manual.img, score=best_sim,
                             source_path=best_manual.path, autocalib=cfg)

    barred = _barred_templates(out_dir, video, vetoed, {t.id for t in templates}, cfg)

    best_t = None
    best_al = None
    for t in templates:
        if t.id in barred:
            continue
        al = estimate_alignment(frame, t.cal, t.img, cfg)
        if al is not None and al.score >= cfg.min_score and (best_al is None or al.score > best_al.score):
            best_t, best_al = t, al
    if best_t is None or best_al is None:
        return None
    cal = best_t.cal
    template = save_snapshot(out_dir, "manual", cal, best_t.img, score=_MANUAL_STORED_SCORE,
                             source_path=best_t.path, autocalib=cfg)
    h, w = frame.shape[:2]
    warped = warp_calibration(cal, best_al, (w, h), f"auto-{cal.name}")
    snap = save_snapshot(out_dir, "auto", warped, frame, score=best_al.score,
                         template_id=template.id, homography=best_al.homography,
                         template_set=template_set, autocalib=cfg)
    # A vetoed id is a known snapshot, so save_snapshot wrote nothing new for it.
    return None if snap.id in vetoed else snap


# --------------------------------------------------------------------------- vetoes (spec 6.5)

VETOES_FILE = "calibration_vetoes.json"


def _vetoes_path(out_dir: Path) -> Path:
    return Path(out_dir) / VETOES_FILE


def _read_vetoes(out_dir: Path) -> dict[str, dict]:
    path = _vetoes_path(out_dir)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, dict)
        and set(v) == {"at", "note", "videos"}
        and isinstance(v["at"], str) and isinstance(v["note"], str)
        and isinstance(v["videos"], list) and v["videos"]
        and all(isinstance(name, str) and name for name in v["videos"])
        and len(v["videos"]) == len(set(v["videos"]))
        for k, v in data.items()
    ):
        raise ValueError(f"{path}: expected {{snapshot_id: {{at, note, videos}}}} with explicit "
                         "video scope; unscoped veto records are not supported")
    return data


def _write_vetoes_and_rebuild(out_dir: Path, vetoes: dict[str, dict], rules) -> None:
    out_dir = Path(out_dir)
    path = _vetoes_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(vetoes, indent=2, sort_keys=True, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)
    if (out_dir / "rounds.json").is_file():
        # The single export path: matches.csv drops / restores the vetoed unreviewed rounds;
        # the rows themselves are rewritten unchanged (vetoes never rewrite judgments).
        from bxjudge.output import read_rounds, save_rounds
        from bxjudge.rules import load_rules

        save_rounds(read_rounds(out_dir), out_dir, rules if rules is not None else load_rules())


def vetoed_ids(out_dir: Path) -> set[str]:
    """Snapshot ids currently vetoed in ``<out_dir>/calibration_vetoes.json`` (empty if absent)."""
    return set(_read_vetoes(out_dir))


def veto_calibration(out_dir: Path, snapshot_id: str, *, autocalib: AutoCalibConfig,
                     note: str = "", rules=None, video: str | None = None) -> None:
    """「不採用此組校準」: record a veto of AUTO snapshot ``snapshot_id`` (ruling T4-1: a wrong
    manual calibration is fixed in ``calib/``) as ``{id: {at, note, videos}}`` in
    ``<out_dir>/calibration_vetoes.json`` and rebuild the CSVs through ``save_rounds``.
    The group's unreviewed rounds stop scoring in ``matches.csv``; no row is changed.
    Video scope comes from the snapshot's current rows/metadata, plus an explicit ``video``
    for a not-yet-judged selection. Reapplying a veto preserves every prior video binding.
    Raises ``ValueError`` for an unknown/manual snapshot or absent video scope."""
    try:
        snap = load_snapshot(out_dir, snapshot_id, autocalib=autocalib)
    except FileNotFoundError as e:
        raise ValueError(str(e)) from None
    if snap.kind != "auto":
        raise ValueError(f"calibration snapshot {snapshot_id} is {snap.kind}: only auto "
                         "calibrations can be vetoed (edit the manual calibration in calib/)")
    vetoes = _read_vetoes(out_dir)
    names = set(vetoes[snapshot_id]["videos"]) if snapshot_id in vetoes else set()
    if video is not None:
        if not isinstance(video, str) or not video:
            raise ValueError("video scope must be a nonempty video name")
        names.add(video)
    for row in json.loads((Path(out_dir) / "rounds.json").read_text()) \
            if (Path(out_dir) / "rounds.json").is_file() else []:
        if row.get("calibration_id") == snapshot_id:
            names.add(row["video"])
    path = Path(out_dir) / "videos.json"
    records = json.loads(path.read_text()) if path.is_file() else {}
    for name, entry in records.items():
        calibration = entry.get("calibration")
        if isinstance(calibration, dict) and calibration.get("id") == snapshot_id:
            names.add(name)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError("cannot veto a calibration without explicit video scope")
    vetoes[snapshot_id] = {"at": datetime.now().astimezone().isoformat(timespec="seconds"),
                           "note": str(note), "videos": sorted(names)}
    _write_vetoes_and_rebuild(out_dir, vetoes, rules)


def unveto_calibration(out_dir: Path, snapshot_id: str, *, autocalib: AutoCalibConfig,
                       rules=None) -> None:
    """Cancel the veto of ``snapshot_id`` (no-op when it is not vetoed) and rebuild the CSVs
    through ``save_rounds``. Raises ``ValueError`` for an id that is neither vetoed nor a
    known snapshot."""
    vetoes = _read_vetoes(out_dir)
    if snapshot_id not in vetoes:
        try:
            load_snapshot(out_dir, snapshot_id, autocalib=autocalib)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from None
        return
    del vetoes[snapshot_id]
    _write_vetoes_and_rebuild(out_dir, vetoes, rules)


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
