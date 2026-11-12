"""Playback timestamps indexed by original decoded presentation order."""

import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import yaml

from bxjudge.config import config_loader

CAPTURE_TAG = "com.android.capture.fps"
DEFAULT_VIDEO_SOURCES = Path(__file__).resolve().parents[2] / "config/video_sources.yaml"


def span_at_least(timestamps_s, start: int, end: int, min_ms: float) -> bool:
    return (timestamps_s[end] - timestamps_s[start]) * 1000 >= min_ms - 1e-8


@dataclass(frozen=True)
class Timeline:
    pts_s: tuple[float, ...]
    capture_fps: float
    playback_fps: float

    def __post_init__(self) -> None:
        if not self.pts_s or any(not math.isfinite(t) for t in self.pts_s):
            raise ValueError("expected finite video presentation timestamps")
        if any(b <= a for a, b in zip(self.pts_s, self.pts_s[1:])):
            raise ValueError("video presentation timestamps must be unique and increasing")
        if not all(math.isfinite(fps) and fps > 0 for fps in (self.capture_fps, self.playback_fps)):
            raise ValueError("expected positive capture and nominal playback rates")

    @property
    def playback_to_physical(self) -> float:
        return self.playback_fps / self.capture_fps

    def playback_s(self, frame: int) -> float:
        return self.pts_s[frame] - self.pts_s[0]

    def physical_s(self, frame: int) -> float:
        return self.playback_s(frame) * self.playback_to_physical

    def elapsed_ms(self, first: int, last: int) -> float:
        return (self.pts_s[last] - self.pts_s[first]) * self.playback_to_physical * 1000



@config_loader
def load_video_sources(path: Path = DEFAULT_VIDEO_SOURCES) -> dict:
    policy = yaml.safe_load(path.read_text())
    if set(policy) != {"batches", "anomalies"} or not policy["batches"]:
        raise ValueError(f"{path}: expected batches and anomalies")
    for batch, rates in policy["batches"].items():
        if (
            not isinstance(batch, str)
            or not batch.isdigit()
            or len(batch) != 8
            or set(rates) != {"capture_fps", "playback_fps"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in rates.values())
        ):
            raise ValueError(f"{path}: invalid video batch {batch!r}")
    return policy


def source_rates(path: Path, metadata: dict, expected: dict) -> tuple[float, float]:
    batch = path.resolve().parent.name
    if batch not in expected:
        raise ValueError(f"{path}: unknown video source batch {batch!r}")
    capture, playback = expected[batch]
    stream = metadata["streams"][0]
    tags = [
        obj.get("tags", {})[CAPTURE_TAG]
        for obj in (stream, metadata.get("format", {}))
        if CAPTURE_TAG in obj.get("tags", {})
    ]
    try:
        capture_ok = bool(tags) and all(float(tag) == capture for tag in tags)
    except (TypeError, ValueError):
        capture_ok = False
    if not capture_ok:
        raise ValueError(f"{path}: {CAPTURE_TAG} missing or conflicts with {capture}: {tags}")
    if Fraction(stream["r_frame_rate"]) != playback:
        raise ValueError(f"{path}: nominal playback rate disagrees with batch {batch}")
    return capture, playback


def probe_timeline(path: Path, policy: dict) -> Timeline:
    """Read one PTS per video packet, with cardinality checked against the stream.

    The approved sources are H.264/HEVC MP4. A caller consuming decoded frames
    also checks decoder cardinality before using the timestamps to judge.
    """
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        f"stream=codec_name,r_frame_rate,time_base,nb_frames:stream_tags={CAPTURE_TAG}:format_tags={CAPTURE_TAG}:packet=pts",
        "-of", "json", str(path),
    ]
    response = subprocess.run(command, capture_output=True, text=True, check=False)
    if response.returncode:
        raise ValueError(f"{path}: ffprobe timestamps failed: {response.stderr.strip()}")
    data = json.loads(response.stdout)
    rates = {k: (v["capture_fps"], v["playback_fps"]) for k, v in policy["batches"].items()}
    capture, playback = source_rates(path, data, rates)
    stream = data["streams"][0]
    packets = data.get("packets", [])
    if stream["codec_name"] not in ("h264", "hevc") or len(packets) != int(stream["nb_frames"]):
        raise ValueError(f"{path}: packet count does not establish a frame timestamp mapping")
    tick = Fraction(stream["time_base"])
    stamps = tuple(float(pts * tick) for pts in sorted(int(p["pts"]) for p in packets))
    return Timeline(stamps, capture, playback)


# Keep the original evidence-count floors where the consumers require them.
_FRAME_MINIMA = {
    "air_collision_frames": 1,
    "id_swap_overlap_frames": 1,
    "impact_coincide_frames": 1,
    "link_velocity_frames": 2,
    "pair_confirmation_frames": 2,
    "reacquire_confirm_frames": 1,
    "resume_frames": 2,
    "reverse_frames": 1,
    "settle_frames": 1,
    "simultaneous_stop_frames": 1,
    "still_frames": 1,
}
_PLAYBACK_RATES = {"link_min_speed_px_s", "link_velocity_error_px_s"}
_SECONDS_KEYS = {
    "bridge_s", "burst_part_overlap_s", "burst_part_window_s", "context_wait_s",
    "held_s", "launch_missing_landing_s", "launch_post_s", "launch_pre_s",
    "link_gap_s", "max_chain_s", "merge_observation_max_gap_s", "pocket_confirm_s",
    "reacquire_min_gap_s", "reanchor_motion_max_gap_s", "reanchor_s", "reject_s",
    "retry_s", "round_gap_s", "separation_s", "spin_dir_window_s",
}
_RATE_KEYS = _PLAYBACK_RATES | {
    "abrupt_min_speed_px", "fall_speed_px", "reverse_max_move_px",
    "reverse_max_rate_deg", "still_px",
}


@dataclass(frozen=True)
class JudgmentTime:
    timeline: Timeline
    durations_ms: dict[str, float]
    rates_per_s: dict[str, float]

    @property
    def values(self) -> dict[str, float | int]:
        """Convert once to the existing engine's playback seconds and frame units."""
        capture = self.timeline.capture_fps
        physical_ms_per_playback_second = 1000 * self.timeline.playback_to_physical
        values = {}
        for key, duration in self.durations_ms.items():
            if key in _FRAME_MINIMA:
                count = round(duration * capture / 1000)
                values[key] = max(_FRAME_MINIMA[key], count)
            else:
                # Existing call sites multiply by avg_fps for sample counts,
                # or divide a frame delta by avg_fps for duration comparisons.
                values[key] = duration / physical_ms_per_playback_second
        for key, rate in self.rates_per_s.items():
            values[key] = (
                rate * self.timeline.playback_to_physical
                if key in _PLAYBACK_RATES else rate / capture
            )
        return values


DEFAULT_PHYSICAL_TIME = DEFAULT_VIDEO_SOURCES.with_name("physical_time.yaml")


@config_loader
def load_physical_time(path: Path = DEFAULT_PHYSICAL_TIME) -> dict:
    policy = yaml.safe_load(path.read_text())
    if (
        set(policy) != {"durations_ms", "rates_per_s"}
        or set(policy["durations_ms"]) != _SECONDS_KEYS | _FRAME_MINIMA.keys()
        or set(policy["rates_per_s"]) != _RATE_KEYS
    ):
        raise ValueError(f"{path}: incomplete physical time policy")
    if any(
        not isinstance(values, dict)
        or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values.values())
        for values in policy.values()
    ):
        raise ValueError(f"{path}: invalid physical time policy")
    return policy
