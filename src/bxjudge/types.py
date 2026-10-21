"""Shared plain dataclasses for bxjudge.

Everything here is a plain ``@dataclass`` (no cached/private fields) so that
``dataclasses.asdict`` works on any of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["X", "B"]
Finish = Literal["spin", "over", "burst", "xtreme"]
TouchState = Literal["yes", "no", "unknown"]
AirState = Literal["yes", "no", "maybe"]
SpinDir = Literal["right", "left", "unknown"]
Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x1, y1, x2, y2
Polygon = list[Point]
Zone = Literal[
    "battle_area", "lid", "over_left", "over_right", "xtreme", "stadium_other", "outside"
]


@dataclass
class Detection:
    frame: int
    cls: str  # "bey" | "bey_part" | "launcher"
    box: Box
    score: float


@dataclass
class Track:
    track_id: int
    cls: str
    dets: list[Detection]

    def first_frame(self) -> int:
        return min(d.frame for d in self.dets)

    def last_frame(self) -> int:
        return max(d.frame for d in self.dets)

    def box_at(self, frame: int) -> Box | None:
        for d in self.dets:
            if d.frame == frame:
                return d.box
        return None

    def center_at(self, frame: int) -> Point | None:
        b = self.box_at(frame)
        if b is None:
            return None
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def centers(self) -> dict[int, Point]:
        return {
            d.frame: ((d.box[0] + d.box[2]) / 2.0, (d.box[1] + d.box[3]) / 2.0)
            for d in self.dets
        }


@dataclass
class Calibration:
    name: str
    frame_size: tuple[int, int]  # (width, height)
    stadium: Polygon  # outer frame, includes the lid
    lid: Polygon  # 上盤: transparent cover
    battle_area: Polygon  # 下盤戰鬥區: floor where the beys run
    over_left: Polygon
    over_right: Polygon
    xtreme: Polygon
    launch_left: Polygon
    launch_right: Polygon
    ref_frame: str


@dataclass
class LaunchCheck:
    touch: dict[Side, TouchState]
    landed: dict[Side, bool]
    landing_frame: dict[Side, int | None]
    landing_zone: dict[Side, Zone | None]
    air_collision: AirState
    notes: list[str]


@dataclass
class RoundResult:
    video: str
    round: int
    t_launch: float
    t_end: float
    launch: LaunchCheck | None
    finish: Finish | None
    winner: Side | None
    points: int
    confidence: float
    needs_review: bool
    review_reasons: list[str]
    vlm_note: str | None = None
    reviewed: bool = False
    reviewer_fix: dict | None = None
    spin_dir: dict[Side, SpinDir] | None = None
