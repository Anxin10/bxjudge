"""Seekable progressive trajectories from saved CSV; no models or source-video analysis."""

import csv
import hashlib
import io
import json
import math
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import trajectory_plot as plot
from .calibration import scale_calibration
from .config import config_loader
from .detector_config import CLASSES
from .output import trajectory_png
from .types import Calibration

DEFAULT_VIDEO_CONFIG = (
    Path(__file__).resolve().parents[2] / "config/trajectory_video.yaml"
)
# Frozen at import, so a running module never labels its output with newer source bytes.
_RENDERER_ID = hashlib.sha256(
    Path(__file__).read_bytes() + Path(plot.__file__).read_bytes()
).hexdigest()


@config_loader
def load_video_policy(path: Path | str = DEFAULT_VIDEO_CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != {
        "fps",
        "max_long_edge",
        "crf",
        "preset",
        "encoder_threads",
    }:
        raise ValueError(f"{path}: incomplete trajectory video policy")
    for key in ("fps", "max_long_edge", "encoder_threads"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{path}: {key} must be a positive integer")
    if type(cfg["crf"]) is not int or not 0 <= cfg["crf"] <= 51:
        raise ValueError(f"{path}: crf must be an integer in [0,51]")
    if cfg["preset"] not in (
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ):
        raise ValueError(f"{path}: unsupported libx264 preset")
    return cfg


@dataclass(frozen=True)
class Observation:
    side: str
    track_id: int
    cls: str
    frame: int
    t: float
    x: float
    y: float

    @property
    def key(self):
        return self.side, self.track_id, self.cls


def _parse_observations(raw: bytes) -> list[Observation]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if not {"frame", "t", "side", "track_id", "cls", "x", "y"} <= set(
        reader.fieldnames or []
    ):
        raise ValueError("trajectory CSV is missing observation fields")
    unique = {}
    for row in reader:
        if any(
            row.get(key) is None
            for key in ("frame", "t", "side", "track_id", "cls", "x", "y")
        ):
            raise ValueError("incomplete trajectory CSV row")
        if row["cls"] not in CLASSES:
            raise ValueError("invalid trajectory class")
        if row["cls"] not in ("bey", "bey_part"):
            continue
        if row["side"] not in ("", "X", "B"):
            raise ValueError("invalid trajectory side")
        if row["x"] == row["y"] == "":
            continue  # explicit missing observation, never a synthetic point
        try:
            o = Observation(
                row["side"],
                int(row["track_id"]),
                row["cls"],
                int(row["frame"]),
                float(row["t"]),
                float(row["x"]),
                float(row["y"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("invalid trajectory observation") from error
        if (
            o.frame < 0
            or o.track_id < 0
            or o.t < 0
            or not all(math.isfinite(v) for v in (o.t, o.x, o.y))
            or not all(-(2**31) <= v <= 2**31 - 1 for v in (o.x, o.y))
        ):
            raise ValueError("nonfinite or negative trajectory observation")
        key = o.key, o.frame
        if key in unique and unique[key] != o:
            raise ValueError("conflicting trajectory observations at one source frame")
        unique[key] = o
    points = sorted(
        unique.values(), key=lambda o: (o.t, o.frame, o.track_id, o.side, o.cls)
    )
    if not any(o.cls == "bey" for o in points):
        raise ValueError("沒有陀螺軌跡觀測，無法產生動畫")
    previous = {}
    for o in points:
        if o.key in previous and (
            o.frame <= previous[o.key].frame or o.t <= previous[o.key].t
        ):
            raise ValueError("trajectory frame/time order is inconsistent")
        previous[o.key] = o
    return points


def read_observations(path: Path) -> list[Observation]:
    return _parse_observations(Path(path).read_bytes())


def read_coordinate_size(csv_path: Path, raw: bytes) -> tuple[int, int]:
    data = json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"coordinate_size", "csv_sha256"}:
        raise ValueError(
            "missing/invalid trajectory coordinate metadata; regenerate trajectory outputs"
        )
    size = data["coordinate_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(n) is not int or n <= 0 for n in size)
    ):
        raise ValueError("invalid trajectory coordinate_size")
    if data["csv_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("trajectory coordinate metadata does not match CSV SHA256")
    return tuple(size)


def _bounds(row, points):
    start, end = row["t_launch"], row["t_end"]
    if (
        type(start) not in (int, float)
        or type(end) not in (int, float)
        or not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end < start
    ):
        raise ValueError("invalid round trajectory time range")
    return min(start, points[0].t), max(end, points[-1].t)


def animation_frames(
    row: dict,
    c: Calibration,
    points: list[Observation],
    cfg: dict,
    *,
    coordinate_size,
    reference=None,
):
    start, end = _bounds(row, points)
    scale = min(1.0, cfg["max_long_edge"] / max(coordinate_size))
    size = tuple(max(1, round(v * scale)) for v in coordinate_size)
    scaled = scale_calibration(c, size)
    sx, sy = size[0] / coordinate_size[0], size[1] / coordinate_size[1]
    canvas = plot.base_image(scaled, reference)
    last = {o.key: o.frame for o in points}
    previous = {}
    cursor = 0
    ticks = (
        int(
            ((Decimal(str(end)) - Decimal(str(start))) * cfg["fps"]).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        + 1
    )
    for index in range(ticks):
        current = end if index == ticks - 1 else min(end, start + index / cfg["fps"])
        while cursor < len(points) and points[cursor].t <= current:
            o = points[cursor]
            cursor += 1
            point = (round(o.x * sx), round(o.y * sy))
            color = plot.SIDE_COLORS.get(o.side, plot.UNKNOWN_COLOR)
            if o.cls == "bey":
                # Every actual point must appear, even when isolated between gaps.
                cv2.circle(canvas, point, 2, color, -1)
                old = previous.get(o.key)
                if old is not None and o.frame == old.frame + 1:
                    cv2.line(
                        canvas, (round(old.x * sx), round(old.y * sy)), point, color, 2
                    )
                if old is None:
                    plot.endpoint(canvas, point, o.side, o.track_id, o.frame)
                if o.frame == last[o.key]:
                    plot.endpoint(canvas, point, o.side, o.track_id, o.frame, end=True)
                events = []
                if (
                    o.side
                    and row["landing_frame"]
                    and row["landing_frame"][o.side] == o.frame
                ):
                    events.append(f"landing_{o.side}@{o.frame}")
                if o.side and row["finish"] is not None and o.t == row["t_end"]:
                    events.append(f"finish@{o.frame}")
                for event in events:
                    cv2.drawMarker(canvas, point, color, cv2.MARKER_CROSS, 12, 2)
                    plot.label(canvas, event, point, 0.4, color)
            elif row["finish"] == "burst" and o.t == row["t_end"]:
                cv2.drawMarker(canvas, point, (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
                plot.label(
                    canvas, f"finish@{o.frame} (part)", point, 0.4, (0, 255, 255)
                )
            previous[o.key] = o
        image = plot.with_legend(canvas)
        plot.label(
            image,
            f"Video t={current:.3f}s",
            (430, canvas.shape[0] + 31),
            0.5,
            (240, 240, 240),
        )
        plot.label(
            image,
            f"Elapsed {current - start:.3f}s / {end - start:.3f}s",
            (430, canvas.shape[0] + 56),
            0.45,
            (240, 240, 240),
        )
        yield image


def _encode(frames, path: Path, cfg):
    first = next(frames)
    height, width = first.shape[:2]
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(cfg["fps"]),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        cfg["preset"],
        "-crf",
        str(cfg["crf"]),
        "-threads",
        str(cfg["encoder_threads"]),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    with tempfile.TemporaryFile() as errors:
        try:
            proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
            )
        except OSError as error:
            raise RuntimeError(
                "ffmpeg unavailable; install ffmpeg with libx264"
            ) from error
        try:
            proc.stdin.write(first.tobytes())
            for frame in frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("ffmpeg encoding failed")
        except (BrokenPipeError, OSError, RuntimeError) as error:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            errors.seek(0)
            raise RuntimeError(
                f"ffmpeg encoding failed: {errors.read().decode(errors='replace')[-1500:]}"
            ) from error
        finally:
            # Kill before flushing/closing stdin, including on KeyboardInterrupt.
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            with suppress(OSError):
                proc.stdin.close()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not produce a complete MP4")


def generate_trajectory_video(
    out: Path, row: dict, c: Calibration, *, config=DEFAULT_VIDEO_CONFIG
) -> Path:
    cfg = load_video_policy(config)
    csv_path = trajectory_png(out, row["video"], row["round"]).with_suffix(".csv")
    raw = csv_path.read_bytes()
    points = _parse_observations(raw)
    coordinate_size = read_coordinate_size(csv_path, raw)
    _bounds(row, points)
    reference_raw = Path(c.ref_frame).read_bytes()
    try:
        reference = cv2.imdecode(
            np.frombuffer(reference_raw, np.uint8), cv2.IMREAD_COLOR
        )
    except cv2.error as error:
        raise ValueError("cannot decode recorded calibration reference") from error
    if reference is None:
        raise ValueError("cannot decode recorded calibration reference")
    timing = {key: row[key] for key in ("t_launch", "t_end", "finish", "landing_frame")}
    metadata = json.dumps(
        {
            "calibration": asdict(c),
            "coordinate_size": coordinate_size,
            "timing": timing,
            "policy": cfg,
            "renderer": _RENDERER_ID,
        },
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(raw + reference_raw + metadata).hexdigest()
    cache = csv_path.parent / "playback"
    cache.mkdir(exist_ok=True)
    output = cache / f"{csv_path.stem}-{digest}.mp4"
    if output.is_file():
        if output.stat().st_size == 0:
            raise RuntimeError(f"empty trajectory video cache: {output}")
        return output
    with tempfile.TemporaryDirectory(prefix=".render-", dir=cache) as directory:
        pending = Path(directory) / "movie.mp4"
        _encode(
            animation_frames(
                timing,
                c,
                points,
                cfg,
                coordinate_size=coordinate_size,
                reference=reference,
            ),
            pending,
            cfg,
        )
        pending.replace(output)
    return output
