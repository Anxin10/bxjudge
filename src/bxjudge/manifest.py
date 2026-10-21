"""Video manifest: ffprobe metadata and the "already processed" record.

The processed-videos record is a small JSON file keyed by the video's absolute path, each
entry holding the ``size`` and ``mtime`` seen when it was processed. ``scan`` returns only
videos that are new or whose size/mtime changed; it never marks anything processed
(``mark_processed`` does, once a video is really done).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_RECORDED_AT = re.compile(r"(\d{8})_(\d{6})")


@dataclass
class VideoInfo:
    path: Path
    size: int
    mtime: float
    duration: float
    fps: float
    width: int  # as displayed (display rotation already applied), matching decoded frames
    height: int
    codec: str
    has_audio: bool
    recorded_at: datetime | None


def parse_recorded_at(name: str) -> datetime | None:
    """``YYYYMMDD_HHMMSS`` anywhere in the file name -> datetime, else None."""
    m = _RECORDED_AT.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _fraction(text: str | None) -> float:
    try:
        f = Fraction(text or "0")
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(f)


def probe(path: Path) -> VideoInfo:
    """ffprobe a video. Raises ``ValueError`` if it is not a readable video."""
    path = Path(path)
    cmd = ["ffprobe", "-v", "error", "-of", "json", "-show_format", "-show_streams", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffprobe not found; install ffmpeg") from e
    if r.returncode != 0:
        raise ValueError(f"ffprobe failed on {path}: {r.stderr.decode(errors='replace').strip()}")
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(f"ffprobe gave unreadable output for {path}") from e

    streams = data.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"
                and not s.get("disposition", {}).get("attached_pic")]
    if not vstreams:
        raise ValueError(f"{path}: no video stream")
    v = vstreams[0]

    width, height = int(v.get("width", 0)), int(v.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: video stream has no dimensions")
    for sd in v.get("side_data_list", []):
        rot = sd.get("rotation")
        if rot is not None and int(round(float(rot))) % 180 != 0:
            width, height = height, width  # ffmpeg auto-rotates when decoding
            break

    fps = _fraction(v.get("avg_frame_rate")) or _fraction(v.get("r_frame_rate"))
    if fps <= 0:
        raise ValueError(f"{path}: cannot determine frame rate")
    duration = v.get("duration") or data.get("format", {}).get("duration")
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        raise ValueError(f"{path}: cannot determine duration") from None

    st = path.stat()
    return VideoInfo(
        path=path,
        size=st.st_size,
        mtime=st.st_mtime,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        codec=str(v.get("codec_name", "")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        recorded_at=parse_recorded_at(path.name),
    )


# --------------------------------------------------------------------------- processed record
def _key(path: Path) -> str:
    return str(Path(path).resolve())


def _load_db(db_path: Path) -> dict:
    db_path = Path(db_path)
    if not db_path.exists():
        return {}
    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{db_path}: unreadable processed-videos record ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"{db_path}: expected a JSON object keyed by video path")
    return data


def mark_processed(info: VideoInfo, db_path: Path) -> None:
    """Record ``info`` (size + mtime) so ``scan`` stops returning it until it changes."""
    db_path = Path(db_path)
    data = _load_db(db_path)
    data[_key(info.path)] = {"size": info.size, "mtime": info.mtime}
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, db_path)


def scan(
    folder: Path,
    db_path: Path,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> list[VideoInfo]:
    """Probe every ``*.mp4`` under ``folder`` (recursive) that is new or changed since it was
    marked processed. Unreadable files are reported via ``on_error`` (default: a warning)
    and skipped. Read-only: neither the videos nor the record are modified."""
    folder = Path(folder)
    done = _load_db(db_path)
    out: list[VideoInfo] = []
    for p in sorted(q for q in folder.rglob("*") if q.is_file() and q.suffix.lower() == ".mp4"):
        st = p.stat()
        rec = done.get(_key(p))
        if rec and rec.get("size") == st.st_size and rec.get("mtime") == st.st_mtime:
            continue
        try:
            out.append(probe(p))
        except ValueError as e:
            if on_error:
                on_error(p, e)
            else:
                log.warning("skipping unreadable video %s: %s", p, e)
    return out
