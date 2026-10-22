"""YAML rules engine: turns per-round judgments into scores and first-to-N matches.

All scoring numbers live in ``config/rules.yaml``; nothing numeric is hard-coded here
(except the fixed meaning of the ``opponent_1pt_on_2nd_miss`` penalty mode, which is
spelled out in its name).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import yaml

from bxjudge.types import Finish, RoundResult, Side

SIDES: tuple[Side, ...] = get_args(Side)
FINISHES: tuple[str, ...] = get_args(Finish)

_LAUNCH_ACTIONS = ("redo", "ignore")
_PENALTY_MODES = ("none", "opponent_1pt_on_2nd_miss")
_LAUNCH_KEYS = ("stadium_touch", "bey_not_in_battle_area", "mid_air_collision")
_DRAW_ACTIONS = ("redo",)
_PENALTY_MISSES = 2  # "2nd miss" in opponent_1pt_on_2nd_miss
_PENALTY_POINTS = 1  # "1pt" in opponent_1pt_on_2nd_miss
# Rounds with these review reasons are pipeline artefacts, not real rounds (Ruling R3).
_NOT_A_ROUND = frozenset({"recalibrate", "no_round", "decode_error", "analysis_error"})


def is_not_a_round(review_reasons: list[str]) -> bool:
    """True when ``review_reasons`` mark a diagnostic placeholder, not a real round."""
    return bool(_NOT_A_ROUND.intersection(review_reasons))


@dataclass
class Rules:
    rules_version: str
    win_points: int
    finish_points: dict[str, int]
    launch_check: dict[str, str]  # mutable on purpose: tests / tuning tweak it in place
    draw: dict[str, str]


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_keys(path, where: str, raw, expected) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: {where} must be a mapping, got {raw!r}")
    unknown = sorted(set(raw) - set(expected), key=str)
    missing = sorted(set(expected) - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: {where}: unknown keys {unknown}, missing keys {missing}")


def load_rules(path: str | Path = "config/rules.yaml") -> Rules:
    """Load rules; unknown/missing keys, wrong types or unsupported modes raise ``ValueError``."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _check_keys(path, "top level", raw, ("rules_version", "win_points", "finish_points",
                                          "launch_check", "draw"))

    if not isinstance(raw["rules_version"], str):
        raise ValueError(f"{path}: rules_version must be a string, got {raw['rules_version']!r}")

    win = raw["win_points"]
    if not _is_int(win) or win < 1:
        raise ValueError(f"{path}: win_points must be a positive int, got {win!r}")

    _check_keys(path, "finish_points", raw["finish_points"], FINISHES)
    for name, v in raw["finish_points"].items():
        if not _is_int(v) or v < 1:
            raise ValueError(f"{path}: finish_points.{name} must be a positive int, got {v!r}")

    lc = raw["launch_check"]
    _check_keys(path, "launch_check", lc, (*_LAUNCH_KEYS, "repeat_miss_penalty"))
    for k in _LAUNCH_KEYS:
        if lc[k] not in _LAUNCH_ACTIONS:
            raise ValueError(f"{path}: launch_check.{k} must be one of {_LAUNCH_ACTIONS}, got {lc[k]!r}")
    if lc["repeat_miss_penalty"] not in _PENALTY_MODES:
        raise ValueError(
            f"{path}: launch_check.repeat_miss_penalty must be one of {_PENALTY_MODES}, "
            f"got {lc['repeat_miss_penalty']!r}"
        )

    _check_keys(path, "draw", raw["draw"], ("action",))
    if raw["draw"]["action"] not in _DRAW_ACTIONS:
        raise ValueError(
            f"{path}: draw.action must be one of {_DRAW_ACTIONS}, got {raw['draw']['action']!r}"
        )

    return Rules(
        rules_version=raw["rules_version"],
        win_points=win,
        finish_points=dict(raw["finish_points"]),
        launch_check=dict(lc),
        draw=dict(raw["draw"]),
    )


def finish_points(rules: Rules, finish: Finish) -> int:
    try:
        return rules.finish_points[finish]
    except KeyError:
        raise ValueError(f"unknown finish {finish!r}; expected one of {FINISHES}") from None


@dataclass
class RoundOutcome:
    action: Literal["score", "redo"]
    scorer: Side | None
    points: int
    reasons: list[str] = field(default_factory=list)


def _other(side: Side) -> Side:
    return "B" if side == "X" else "X"


def decide_round(r: RoundResult, rules: Rules, misses: dict[Side, int]) -> RoundOutcome:
    """Decide one round. Mutates ``misses`` (per-match launch-miss counter per side).

    Order: launch failures -> redo (+ optional repeat-miss penalty); no finish -> draw
    action; otherwise score ``finish_points`` for ``r.winner``. ``r.launch is None``
    means "no launch failure". ``touch == "unknown"`` / ``air_collision == "maybe"`` never
    trigger a redo.

    Miss tally (official regulation v12, R23): a decided round (finish points or a
    repeat-miss penalty) resets BOTH sides to 0; air-collision, draw and other replays
    keep the tally. In ``opponent_1pt_on_2nd_miss`` mode a side's second miss gives the
    opponent 1 point; if both sides make their second miss in the same round those
    second misses are void and both stay at 1.
    """
    reasons: list[str] = []
    missers: list[Side] = []  # sides at fault (an air collision blames nobody)
    lc, cfg = r.launch, rules.launch_check
    if lc is not None:
        for side in SIDES:
            faults = []
            if lc.touch.get(side) == "yes" and cfg["stadium_touch"] == "redo":
                faults.append(f"touch_{side}")
            if lc.landed.get(side, True) is False and cfg["bey_not_in_battle_area"] == "redo":
                faults.append(f"not_landed_{side}")
            if faults:
                reasons += faults
                missers.append(side)
        if lc.air_collision == "yes" and cfg["mid_air_collision"] == "redo":
            reasons.append("air_collision")

    if reasons:
        for side in missers:
            misses[side] = misses.get(side, 0) + 1
        if cfg["repeat_miss_penalty"] == "opponent_1pt_on_2nd_miss":
            # Official v12 tally (R23): a second miss gives the opponent the point and
            # clears every miss so far; second misses by both sides in the same round
            # are void, leaving both at their first miss.
            penalised = [s for s in missers if misses[s] >= _PENALTY_MISSES]
            if len(penalised) == len(SIDES):
                for side in penalised:
                    misses[side] = _PENALTY_MISSES - 1
            elif penalised:
                _reset(misses)
                return RoundOutcome(
                    "score", _other(penalised[0]), _PENALTY_POINTS,
                    [*reasons, f"repeat_miss_penalty_{penalised[0]}"],
                )
        return RoundOutcome("redo", None, 0, reasons)

    if r.finish is None:
        return RoundOutcome(rules.draw["action"], None, 0, ["draw"])
    if r.winner is None:  # inconsistent upstream result; don't invent a scorer
        return RoundOutcome("redo", None, 0, ["finish_without_winner"])
    _reset(misses)  # a decided round ends the miss tally of that battle
    return RoundOutcome("score", r.winner, finish_points(rules, r.finish), [])


def _reset(misses: dict[Side, int]) -> None:
    for side in SIDES:
        misses[side] = 0


@dataclass
class Match:
    rounds: list[tuple[RoundResult, RoundOutcome]]
    score: dict[Side, int]
    winner: Side | None


def group_matches(results: list[RoundResult], rules: Rules) -> list[Match]:
    """Accumulate rounds into first-to-``win_points`` matches.

    Rounds are ordered by (video, t_launch, round): videos are named by recording
    timestamp, so name order is chronological and ``t_launch`` orders rounds inside a
    video (the sort is stable, so equal keys keep input order). Rounds whose
    ``review_reasons`` mark them as non-rounds (recalibrate / no_round / decode_error)
    are skipped. A match ends when a side reaches ``win_points``; the next round opens
    a new match with fresh miss counters. A reviewer-set ``match_start`` (R52) closes
    an unfinished match early so that round opens a new one; it cannot merge rounds
    across a match that already ended by reaching ``win_points``.
    """
    real = [r for r in results if not is_not_a_round(r.review_reasons)]
    real.sort(key=lambda r: (r.video, r.t_launch, r.round))

    matches: list[Match] = []
    cur: Match | None = None
    misses: dict[Side, int] = {}
    for r in real:
        if r.match_start:
            cur = None
        if cur is None:
            cur = Match(rounds=[], score={"X": 0, "B": 0}, winner=None)
            matches.append(cur)
            misses = {"X": 0, "B": 0}
        o = decide_round(r, rules, misses)
        cur.rounds.append((r, o))
        if o.action == "score" and o.scorer is not None:
            cur.score[o.scorer] += o.points
            if cur.score[o.scorer] >= rules.win_points:
                cur.winner = o.scorer
                cur = None
    return matches
