"""Seekable progressive trajectories from saved CSV; no models or source-video analysis."""

import csv
import hashlib
import io
import json
import math
import subprocess
import tempfile
from contextlib import closing, suppress
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import autocalib, calibration, dynamic_calibration, geometry, launch_phase
from . import stadium_follow as follow
from . import trajectory_plot as plot
from .calibration import scale_calibration
from .config import config_loader
from .detector_config import CLASSES
from .manifest import probe
from .output import trajectory_png
from .types import Calibration
from .video import VideoDecodeError, iter_frames

DEFAULT_VIDEO_CONFIG = (
    Path(__file__).resolve().parents[2] / "config/trajectory_video.yaml"
)
# Frozen at import, so a running module never labels its output with newer source bytes.
_RENDERER_ID = hashlib.sha256(
    Path(__file__).read_bytes()
    + b"".join(
        Path(m.__file__).read_bytes()
        for m in (
            plot,
            follow,
            autocalib,
            calibration,
            geometry,
            dynamic_calibration,
            launch_phase,
        )
    )
    + cv2.__version__.encode()
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
        "stadium_follow",
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
    cfg["stadium_follow"] = follow.validate_follow_policy(cfg["stadium_follow"])
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
    background_frames=None,
    dynamic_policy=None,
    recorded_geometry=None,
    source_fps=None,
):
    start, end = _bounds(row, points)
    scale = min(1.0, cfg["max_long_edge"] / max(coordinate_size))
    size = tuple(max(1, round(v * scale)) for v in coordinate_size)
    scaled = scale_calibration(c, size)
    sx, sy = size[0] / coordinate_size[0], size[1] / coordinate_size[1]
    base = plot.base_image(scaled, reference)
    follower = (
        dynamic_calibration.DynamicCalibration(
            scaled,
            cv2.imread(c.ref_frame) if reference is None else reference,
            cfg["fps"],
            cfg["stadium_follow"],
            dynamic_calibration.load_dynamic_policy()
            if dynamic_policy is None
            else dynamic_policy,
        )
        if background_frames is not None
        else None
    )
    canvas = np.zeros_like(base)
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
        alignment_status = None
        if background_frames is None:
            image = base.copy()
        else:
            try:
                background = next(background_frames)
            except StopIteration as error:
                raise RuntimeError(
                    "source footage ended before the trajectory timeline"
                ) from error
            background = cv2.resize(background, size, interpolation=cv2.INTER_AREA)
            if recorded_geometry is None:
                sample = follower.update(index, background)
                alignment = sample.alignment
                status = sample.status
            else:
                frame_number = _frame_number(current, source_fps)
                sample = recorded_geometry["samples"].get(frame_number)
                alignment = (
                    None
                    if sample is None or sample["alignment"] is None
                    else autocalib.Alignment(
                        **{
                            **sample["alignment"],
                            "homography": tuple(
                                tuple(float(v) for v in r)
                                for r in _resize_recorded_h(
                                    sample["alignment"]["homography"],
                                    recorded_geometry["reference_size"],
                                    size,
                                )
                            ),
                        }
                    )
                )
                status = "missing" if sample is None else sample["status"]
            if alignment is None:
                image = background.copy()
                alignment_status = ("Alignment lost: zones hidden", (80, 80, 255))
            else:
                moved = autocalib.warp_calibration(scaled, alignment, size, scaled.name)
                image = plot.base_image(moved, background)
                alignment_status = (
                    f"{'Recorded geometry' if recorded_geometry is not None else 'Follow preview'}: {alignment.score:.2f} {status}",
                    (80, 200, 80),
                )
        mask = np.any(canvas != 0, axis=2)
        image[mask] = canvas[mask]
        image = plot.with_legend(image)
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
        if alignment_status is not None:
            text, color = alignment_status
            plot.label(image, text, (430, canvas.shape[0] + 76), 0.35, color)
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
                f"ffmpeg encoding failed: {error}; {errors.read().decode(errors='replace')[-1500:]}"
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


def _sample_time(start, end, fps):
    count = (
        int(
            ((Decimal(str(end)) - Decimal(str(start))) * fps).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        + 1
    )
    for index in range(count):
        yield end if index == count - 1 else min(end, start + index / fps)


def _frame_number(t, fps):
    value = t * fps
    nearest = round(value)
    return (
        nearest
        if math.isclose(value, nearest, rel_tol=0, abs_tol=1e-7)
        else math.floor(value)
    )


def footage_frames(
    source: Path, start: float, end: float, fps: int, long_edge: int, info
):
    """Sample the original displayed source frame at or before each animation time."""
    # The saved tracks count frames from source zero. Timestamp seeking can land at
    # another frame on phone/VFR clips, so discard the prefix by its actual frame index.
    with closing(
        iter_frames(source, start=0, end=None, long_edge=long_edge, hwaccel="cpu")
    ) as decoded:
        upcoming = next(decoded, None)
        last = None
        for current in _sample_time(start, end, fps):
            target = _frame_number(current, info.fps)
            while upcoming is not None and upcoming[0] <= target:
                last = upcoming
                upcoming = next(decoded, None)
            if last is None or last[0] != target:
                raise RuntimeError(
                    f"source footage missing frame {target}; no image substitution"
                )
            yield last[1]


def _source_hash(source):
    with source.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def generate_trajectory_video(
    out: Path, row: dict, c: Calibration, *, source: Path, config=DEFAULT_VIDEO_CONFIG
) -> Path:
    cfg = load_video_policy(config)
    dynamic_policy = dynamic_calibration.load_dynamic_policy()
    csv_path = trajectory_png(out, row["video"], row["round"]).with_suffix(".csv")
    raw = csv_path.read_bytes()
    points = _parse_observations(raw)
    coordinate_size = read_coordinate_size(csv_path, raw)
    start, end = _bounds(row, points)
    source = Path(source).resolve()
    source_before = _source_hash(source)
    info = probe(source)
    if end >= info.duration:
        raise ValueError("trajectory timeline extends beyond the source video")
    reference_raw = Path(c.ref_frame).read_bytes()
    try:
        reference = cv2.imdecode(
            np.frombuffer(reference_raw, np.uint8), cv2.IMREAD_COLOR
        )
    except cv2.error as error:
        raise ValueError("cannot decode recorded calibration reference") from error
    if reference is None:
        raise ValueError("cannot decode recorded calibration reference")
    recorded_geometry = None
    evidence = row.get("geometry_evidence")
    if evidence is not None:
        recorded_geometry = dynamic_calibration.read_geometry(out, row, source_before)
    timing = {key: row[key] for key in ("t_launch", "t_end", "finish", "landing_frame")}
    metadata = json.dumps(
        {
            "calibration": asdict(c),
            "coordinate_size": coordinate_size,
            "timing": timing,
            "policy": cfg,
            "dynamic_policy": dynamic_policy,
            "recorded_geometry": evidence,
            "renderer": _RENDERER_ID,
            "source": {
                "path": str(source),
                "sha256": source_before,
                "fps": info.fps,
                "size": [info.width, info.height],
            },
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
        try:
            with closing(
                footage_frames(
                    source, start, end, cfg["fps"], cfg["max_long_edge"], info
                )
            ) as backgrounds:
                _encode(
                    animation_frames(
                        timing,
                        c,
                        points,
                        cfg,
                        coordinate_size=coordinate_size,
                        reference=reference,
                        background_frames=backgrounds,
                        dynamic_policy=dynamic_policy,
                        recorded_geometry=recorded_geometry,
                        source_fps=info.fps,
                    ),
                    pending,
                    cfg,
                )
        except VideoDecodeError as error:
            raise RuntimeError(f"cannot decode source footage: {error}") from error
        if _source_hash(source) != source_before:
            raise RuntimeError(
                "source footage changed during rendering; result not published"
            )
        pending.replace(output)
    return output


def _resize_recorded_h(matrix, source_size, display_size):
    scale = np.diag(
        [display_size[0] / source_size[0], display_size[1] / source_size[1], 1.0]
    )
    return scale @ np.array(matrix) @ np.linalg.inv(scale)
