"""Thresholds loaded from ``config/thresholds.yaml`` (no numbers hard-coded in code)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Thresholds:
    fps_default: int
    infer_long_edge: int
    decode_hwaccel: str  # "cuda" | "cpu"
    launch_pre_s: float
    launch_post_s: float
    launch_missing_landing_s: float
    shake_px: float
    impact_coincide_frames: int
    air_collision_frames: int
    abrupt_angle_deg: float
    abrupt_speed_ratio: float
    abrupt_min_speed_px: float
    settle_frames: int
    fall_speed_px: float
    rim_margin_px: float
    spin_dir_window_s: float
    spin_dir_min_turn_deg: float
    orbit_to_spin: dict[str, str]
    reverse_frames: int
    reverse_max_rate_deg: float
    reverse_max_move_px: float
    burst_part_radius_px: float
    id_swap_confidence_factor: float
    track_gap_confidence_factor: float
    still_px: float
    still_activity: float
    still_frames: int
    simultaneous_stop_frames: int
    burst_part_window_s: float
    id_swap_overlap_frames: int
    round_gap_s: float
    calibration_min_similarity: float
    review_confidence: float


_HWACCEL = ("cuda", "cpu")


def load_thresholds(path: str | Path = "config/thresholds.yaml") -> Thresholds:
    """Load thresholds; unknown keys, missing keys or wrong types raise ``ValueError``."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of threshold names to values")

    flds = {f.name: f for f in fields(Thresholds)}
    unknown = sorted(set(raw) - set(flds))
    missing = sorted(set(flds) - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")

    values: dict = {}
    for name, f in flds.items():
        v = raw[name]
        # `from __future__ import annotations` makes f.type a string
        if name == "orbit_to_spin":
            if (not isinstance(v, dict) or set(v) != {"cw", "ccw"}
                    or any(not isinstance(value, str) for value in v.values())
                    or set(v.values()) != {"left", "right"}):
                raise ValueError(f"{path}: orbit_to_spin must map cw/ccw bijectively to left/right")
            v = dict(v)
        elif f.type in ("int", int):
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{path}: {name} must be an int, got {v!r}")
        elif f.type in ("float", float):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{path}: {name} must be a number, got {v!r}")
            v = float(v)
        else:  # str
            if not isinstance(v, str):
                raise ValueError(f"{path}: {name} must be a string, got {v!r}")
        values[name] = v
    if values["decode_hwaccel"] not in _HWACCEL:
        raise ValueError(f"{path}: decode_hwaccel must be one of {_HWACCEL}")
    return Thresholds(**values)
