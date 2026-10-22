"""Evaluation against human-reviewed rounds and advisory threshold tuning (spec 10-11).

Truth is the ``effective()`` value of ``reviewed`` real rounds; predictions are the
ORIGINAL flat fields (``finish``, ``winner``, ``x_touch``, ``needs_review`` ...), which
corrections never overwrite. Diagnostic placeholders (``rules.is_not_a_round``) are never
rounds. A metric whose denominator is empty is ``nan`` ("unverified") and never passes.
Targets, tolerances and percentiles come from ``config/evaluation.yaml`` (R40). Nothing
here writes results, corrections or configuration.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .output import effective, matches_for, row_to_result
from .rules import SIDES, Rules, group_matches, is_not_a_round
from .vlm import UNAVAILABLE_PREFIX

NAN = math.nan
METRIC_NAMES = (
    "finish_accuracy_unflagged",
    "touch_recall",
    "air_recall",
    "review_ratio",
    "speed_ratio",
)
# Reviewer-correctable judgment fields whose effective value is the truth.
TRUTH_FIELDS = (
    "x_touch",
    "b_touch",
    "x_landed",
    "b_landed",
    "x_land_zone",
    "b_land_zone",
    "x_spin_dir",
    "b_spin_dir",
    "air_collision",
    "finish",
    "winner",
)
_TRUTH_ISSUES = (
    "duplicate_truth_key",
    "duplicate_prediction_key",
    "missing_prediction",
    "prediction_not_a_round",
    "segmentation_mismatch",
    "time_mismatch",
)
_ERROR_GROUPS = (
    "finish_wrong_unflagged",
    "touch_missed",
    "air_collision_missed",
    "corrected_after_flag",
    "touch_false_alarm",
    "air_collision_false_alarm",
)


# --------------------------------------------------------------------------- config


@dataclass
class EvalConfig:
    targets: dict[str, tuple[str, float]]
    speed_max_video_s: float
    latency_percentile: float
    truth_t_launch_tolerance_s: float
    shake_overlap_percentile: float
    spin_min_samples: int
    audit_confidence_level: float


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _keys(path, where, raw, expected) -> None:
    if not isinstance(raw, dict) or set(raw) != set(expected):
        got = sorted(raw) if isinstance(raw, dict) else raw
        raise ValueError(f"{path}: {where} must have exactly {sorted(expected)}, got {got!r}")


def load_eval_config(path: str | Path = "config/evaluation.yaml") -> EvalConfig:
    """Load evaluation policy; unknown/missing keys or invalid values raise ``ValueError``."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _keys(path, "top level", raw, (
        "targets", "speed_max_video_s", "latency_percentile", "truth_t_launch_tolerance_s",
        "shake_overlap_percentile", "spin_min_samples", "audit_confidence_level",
    ))
    level = raw["audit_confidence_level"]
    if not _number(level) or not 0 < level < 1:
        raise ValueError(f"{path}: audit_confidence_level must be in (0, 1)")
    _keys(path, "targets", raw["targets"], METRIC_NAMES)
    targets = {}
    for name in METRIC_NAMES:
        spec = raw["targets"][name]
        if not isinstance(spec, dict) or len(spec) != 1 or next(iter(spec)) not in ("min", "max"):
            raise ValueError(f"{path}: targets.{name} must be {{min: x}} or {{max: x}}")
        (op, value), = spec.items()
        if not _number(value) or value < 0:
            raise ValueError(f"{path}: targets.{name}.{op} must be a nonnegative number")
        targets[name] = (op, float(value))
    for name in ("latency_percentile", "shake_overlap_percentile"):
        if not _number(raw[name]) or not 0 <= raw[name] <= 100:
            raise ValueError(f"{path}: {name} must be a percentile in [0, 100]")
    for name in ("speed_max_video_s", "truth_t_launch_tolerance_s"):
        if not _number(raw[name]) or raw[name] < 0:
            raise ValueError(f"{path}: {name} must be a nonnegative number")
    if type(raw["spin_min_samples"]) is not int or raw["spin_min_samples"] < 1:
        raise ValueError(f"{path}: spin_min_samples must be a positive int")
    return EvalConfig(
        targets=targets,
        speed_max_video_s=float(raw["speed_max_video_s"]),
        latency_percentile=float(raw["latency_percentile"]),
        truth_t_launch_tolerance_s=float(raw["truth_t_launch_tolerance_s"]),
        shake_overlap_percentile=float(raw["shake_overlap_percentile"]),
        spin_min_samples=raw["spin_min_samples"],
        audit_confidence_level=float(level),
    )


# --------------------------------------------------------------------------- metrics


@dataclass
class Metrics:
    # Unflagged finish+winner accuracy (R62): exact when every unflagged real round is
    # reviewed, else the random audit point estimate (provisional if incomplete).
    finish_accuracy_unflagged: float
    touch_recall: float
    air_recall: float
    review_ratio: float
    # Worst observed per-video / duration when complete, otherwise available evidence.
    # Batch throughput is diagnostic; stage sums cannot establish observed per-video wall-latency compliance (C5).
    speed_ratio: float
    n_reviewed: int
    n_rounds: int = 0
    n_placeholders: int = 0
    n_unflagged_reviewed: int = 0
    n_touch_truth: int = 0
    n_air_truth: int = 0
    n_speed_videos: int = 0
    n_vlm_videos: int = 0
    n_observed_videos: int = 0
    n_required_speed_videos: int = 0
    n_deadline_videos: int = 0
    speed_evidence_complete: bool = False
    speed_violation: str | None = None
    speed_ratio_core: float = NAN  # diagnostic: median core-only ratio
    speed_ratio_sum: float = NAN  # median per-video (core + VLM) / duration
    speed_ratio_wall: float = NAN  # batch wall_s / video duration (R61)
    speed_basis: str = "unverified"  # sum-based per-video estimate | observed per-video wall
    finish_accuracy_unflagged_method: str = "unverified"
    finish_accuracy_unflagged_low: float = NAN
    finish_accuracy_unflagged_high: float = NAN


def _ratio(num: int, den: int) -> float:
    return num / den if den else NAN


def real_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not is_not_a_round(r.get("review_reasons") or [])]


def truth_rows(rows: list[dict]) -> list[dict]:
    """Reviewed real rounds: the only rows whose effective values are truth."""
    return [r for r in real_rows(rows) if r.get("reviewed") is True]


def _finish_right(row: dict) -> bool:
    eff = effective(row)
    return row["finish"] == eff["finish"] and row["winner"] == eff["winner"]


def _caught(row: dict, kind: str, side: str | None = None) -> tuple[bool, bool]:
    """(truth positive, predicted or flagged) for touch (per side) or air collision."""
    key = f"{side.lower()}_touch" if kind == "touch" else "air_collision"
    return effective(row)[key] == "yes", row[key] == "yes" or row["needs_review"] is True


def _speed_videos(rows: list[dict], timings: dict | None, max_s: float):
    """Eligible per-video timing rows plus excluded video names by reason (R59/R60).

    Eligible: a positive core measurement (``processing_s``) and playback duration, at
    least one real round (an early recalibrate/decode gate is not judgment throughput),
    and playback duration within the spec limit. ``vlm_s`` (absent = no VLM phase) is
    added for the stage-sum diagnostic. Durations are playback seconds as recorded.
    """
    excluded: dict[str, list[str]] = {"no_real_round": [], "unmeasured": [], "over_max_duration": []}
    if timings is None:
        return [], excluded
    judged = {r["video"] for r in real_rows(rows)}
    videos = []
    for video, entry in sorted(timings.items()):
        entry = entry if isinstance(entry, dict) else {}
        duration, core = entry.get("duration_s"), entry.get("processing_s")
        vlm = entry.get("vlm_s", 0.0)
        if video not in judged:
            excluded["no_real_round"].append(video)
        # A measured core run is never 0 s; a 0 is a placeholder, never a perfect speed.
        elif not (_number(duration) and duration > 0 and _number(core) and core > 0
                  and _number(vlm) and vlm >= 0):
            excluded["unmeasured"].append(video)
        elif duration > max_s:
            excluded["over_max_duration"].append(video)
        else:
            total = core + vlm
            videos.append({
                "video": video, "duration_s": duration, "core_s": core, "vlm_s": vlm,
                "total_s": total, "core_ratio": core / duration,
                "total_ratio": total / duration, "has_vlm": "vlm_s" in entry,
            })
    return videos, excluded


def _observed_speed(rows: list[dict], timings: dict | None, max_s: float) -> dict:
    """Require a valid observation for every judged video within the duration limit.

    Missing/invalid durations cannot silently remove a judged video from coverage.
    Stage measurements and batch execution history are not substitutes for observations.
    """
    observed, missing = [], []
    for video in sorted({r["video"] for r in real_rows(rows)}):
        entry = (timings or {}).get(video)
        entry = entry if isinstance(entry, dict) else {}
        duration = entry.get("duration_s")
        if _number(duration) and duration > max_s:
            continue
        wall = entry.get("pipeline_wall")
        seconds = wall.get("seconds") if isinstance(wall, dict) else None
        if (_number(duration) and duration > 0 and isinstance(wall, dict)
                and wall.get("scope") == "core_to_final_output"
                and _number(seconds) and seconds > 0):
            observed.append({"video": video, "seconds": seconds, "ratio": seconds / duration})
        else:
            missing.append(video)
    return {"videos": observed, "missing": missing,
            "complete": bool(observed) and not missing}


def _deadline_speed(rows: list[dict], timings: dict | None, max_s: float) -> dict:
    """Current complete-invocation certificates and observed output-ready lower bounds."""
    def valid(wall, scope, duration):
        return (_number(duration) and duration > 0 and isinstance(wall, dict)
                and wall.get("scope") == scope and _number(wall.get("seconds"))
                and wall["seconds"] > 0)

    certificates, lower, missing = [], [], []
    for video in sorted({r["video"] for r in real_rows(rows)}):
        entry = (timings or {}).get(video)
        entry = entry if isinstance(entry, dict) else {}
        duration = entry.get("duration_s")
        if _number(duration) and duration > max_s:
            continue
        wall = entry.get("invocation_wall")
        if (valid(wall, "complete_invocation", duration)
                and wall.get("kind") in ("exact", "completion_upper_bound")):
            certificates.append({"video": video, **wall, "ratio": wall["seconds"] / duration})
        else:
            missing.append(video)
        ready = entry.get("output_ready_wall")
        if valid(ready, "invocation_to_output_ready", duration):
            lower.append({"video": video, **ready, "ratio": ready["seconds"] / duration})
    return {"videos": certificates, "lower_bounds": lower, "missing": missing,
            "complete": bool(certificates) and not missing}


def _wall(rows: list[dict], timings: dict | None, runs: list[dict] | None, max_s: float) -> dict:
    """Batch wall-clock basis from ``runs.json`` records (R61).

    ``run`` records whose videos include at least one real round count (an early
    recalibrate gate is not judgment throughput); their wall_s already contains their
    own VLM phase, so ``vlm_wall_s`` is only a breakdown and never added again.
    Standalone ``vlm`` records add waiting time but no video duration. ``sum_s`` is the
    per-video core + VLM sum over the same videos, reported next to the wall-clock: it can
    exceed wall-clock (e.g. parallel videos) and is never hidden.
    """
    judged = {r["video"] for r in real_rows(rows)}
    def eligible(run):
        return {v: d for v, d in run["videos"].items()
                if v in judged and _number(d) and 0 < d <= max_s}

    used = [r for r in runs or [] if r["kind"] == "run" and eligible(r)]
    if used:
        used += [r for r in runs if r["kind"] == "vlm"]
    wall = sum(r["wall_s"] for r in used)
    vlm_wall = sum(r["vlm_wall_s"] or 0.0 for r in used)
    executions = [{"kind": r["kind"], "started_at": r["started_at"], "wall_s": r["wall_s"],
                   "duration_s": sum(eligible(r).values()),
                   "ratio": r["wall_s"] / sum(eligible(r).values()) if eligible(r) else NAN}
                  for r in used]
    durations = {v: d for r in used if r["kind"] == "run" for v, d in eligible(r).items()}
    duration = sum(e["duration_s"] for e in executions)
    n_core_runs = sum(r["kind"] == "run" for r in used)
    measured = [
        (e["processing_s"] + e.get("vlm_s", 0.0), durations[v])
        for v, e in (timings or {}).items()
        if v in durations and isinstance(e, dict) and _number(e.get("processing_s"))
        and e["processing_s"] > 0 and _number(e.get("vlm_s", 0.0))
    ]
    sum_s = sum(s for s, _ in measured) if measured else NAN
    sum_duration = sum(d for _, d in measured)
    ratio = wall / duration if used and duration > 0 else NAN
    return {
        "basis": ("cumulative wall-clock" if n_core_runs > 1 else "wall-clock") if used else "unverified",
        "executions": executions,
        "n_runs": len(used),
        "wall_s": wall if used else NAN,
        "vlm_wall_s": vlm_wall if used else NAN,
        "duration_s": duration if used else NAN,
        "unique_duration_s": sum(durations.values()) if used else NAN,
        "ratio": ratio,
        "sum_s": sum_s,
        "sum_ratio": sum_s / sum_duration if measured and sum_duration > 0 else NAN,
        "sum_exceeds_wall": bool(n_core_runs == 1 and measured and sum_s > wall),
    }


def _wilson(correct: int, n: int, level: float) -> tuple[float, float]:
    z = statistics.NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def unflagged_accuracy(rows: list[dict], level: float | None = None) -> dict:
    """Unflagged finish+winner accuracy without selection bias (R62).

    ``exact`` when every unflagged real round is reviewed; else ``audit_sample`` only
    when ALL selected members are reviewed, with a Wilson interval at ``level``.
    Partial review gives a provisional point estimate, never acceptance evidence.
    Reviewed unflagged rounds outside the
    sample were chosen by people, so they are reported separately as ``biased``.
    """
    unflagged = [r for r in real_rows(rows) if r["needs_review"] is False]
    reviewed = [r for r in unflagged if r.get("reviewed") is True]
    sampled = [r for r in unflagged if r.get("audit_sample") is True]
    sampled_reviewed = [r for r in sampled if r.get("reviewed") is True]
    biased = [r for r in reviewed if r.get("audit_sample") is not True]
    out = {"n_unflagged": len(unflagged), "n_reviewed": len(reviewed),
           "n_sampled": len(sampled), "biased": _accuracy(biased), "reason": None,
           "level": None}
    if unflagged and len(reviewed) == len(unflagged):
        k = sum(map(_finish_right, reviewed))
        p = k / len(reviewed)
        return {**out, "method": "exact", "n_used": len(reviewed), "correct": k,
                "estimate": p, "low": p, "high": p}
    if sampled_reviewed:
        k = sum(map(_finish_right, sampled_reviewed))
        if len(sampled_reviewed) != len(sampled):
            return {**out, "method": "incomplete_audit", "n_used": len(sampled_reviewed),
                    "correct": k, "estimate": k / len(sampled_reviewed),
                    "low": NAN, "high": NAN, "reason": "selected audit sample incompletely reviewed"}
        level = load_eval_config().audit_confidence_level if level is None else level
        lo, hi = _wilson(k, len(sampled_reviewed), level)
        return {**out, "method": "audit_sample", "n_used": len(sampled_reviewed),
                "correct": k, "estimate": k / len(sampled_reviewed), "low": lo, "high": hi,
                "level": level}
    reason = ("no unflagged rounds" if not unflagged
              else "audit sample not reviewed yet" if sampled
              else "selection bias / no random sample")
    return {**out, "method": "unverified", "n_used": 0, "correct": 0, "estimate": NAN,
            "low": NAN, "high": NAN, "reason": reason}


def evaluate(rows: list[dict], timings: dict[str, dict] | None = None, *,
             config: EvalConfig | None = None, runs: list[dict] | None = None) -> Metrics:
    """Spec section 11 metrics from flat ``rounds.json`` rows (+ ``videos.json`` timings
    and ``runs.json`` wall-clock records).

    Without complete observed per-video timing speed cannot PASS. ``config`` defaults to
    ``config/evaluation.yaml``.
    """
    real, truth = real_rows(rows), truth_rows(rows)
    cfg = config if config is not None else load_eval_config()
    touch = [_caught(r, "touch", s) for r in truth for s in SIDES]
    touch = [c for positive, c in touch if positive]
    air = [c for positive, c in (_caught(r, "air") for r in truth) if positive]
    speed = NAN, NAN
    worst_video_ratio = NAN
    n_speed = n_vlm = 0
    videos = []
    if timings is not None:
        videos, _ = _speed_videos(rows, timings, cfg.speed_max_video_s)
        if videos:
            worst_video_ratio = max(v["total_ratio"] for v in videos)
            speed = (statistics.median(v["total_ratio"] for v in videos),
                     statistics.median(v["core_ratio"] for v in videos))
        n_speed, n_vlm = len(videos), sum(v["has_vlm"] for v in videos)
    wall_metrics = _wall(rows, timings, runs, cfg.speed_max_video_s)
    wall = wall_metrics["ratio"]
    basis = "sum-based per-video estimate" if math.isfinite(worst_video_ratio) else "unverified"
    observation = _observed_speed(rows, timings, cfg.speed_max_video_s)
    deadline = _deadline_speed(rows, timings, cfg.speed_max_video_s)
    lower = [*observation["videos"], *deadline["lower_bounds"]]
    if lower:
        worst_video_ratio = max(v["ratio"] for v in lower)
        basis = "observed output-availability lower bound"
    if deadline["videos"]:
        worst_video_ratio = max(v["ratio"] for v in deadline["videos"])
        basis = ("complete" if deadline["complete"] else "partial") + " invocation deadline evidence"
    violation = None
    if any(not _meets(v["ratio"], cfg.targets["speed_ratio"])
           for v in deadline["videos"] if v["kind"] == "exact"):
        violation = "exact"
    elif any(not _meets(v["ratio"], cfg.targets["speed_ratio"]) for v in lower):
        violation = "observed lower-bound"
    elif any(not _meets(v["total_ratio"], cfg.targets["speed_ratio"])
             for v in videos if v["video"] in deadline["missing"]):
        violation = "sum-based"
    u = unflagged_accuracy(rows, None if config is None else config.audit_confidence_level)
    return Metrics(
        finish_accuracy_unflagged=u["estimate"],
        touch_recall=_ratio(sum(touch), len(touch)),
        air_recall=_ratio(sum(air), len(air)),
        review_ratio=_ratio(sum(r["needs_review"] is True for r in real), len(real)),
        speed_ratio=worst_video_ratio,
        n_reviewed=len(truth),
        n_rounds=len(real),
        n_placeholders=len(rows) - len(real),
        n_unflagged_reviewed=u["n_used"],
        n_touch_truth=len(touch),
        n_air_truth=len(air),
        n_speed_videos=n_speed,
        n_vlm_videos=n_vlm,
        n_observed_videos=len(observation["videos"]),
        n_required_speed_videos=len(deadline["videos"]) + len(deadline["missing"]),
        n_deadline_videos=len(deadline["videos"]),
        speed_evidence_complete=deadline["complete"],
        speed_violation=violation,
        speed_ratio_core=speed[1],
        speed_ratio_sum=speed[0],
        speed_ratio_wall=wall,
        speed_basis=basis,
        finish_accuracy_unflagged_method=u["method"],
        finish_accuracy_unflagged_low=u["low"],
        finish_accuracy_unflagged_high=u["high"],
    )


def _meets(value: float, target: tuple[str, float]) -> bool:
    op, limit = target
    if not math.isfinite(value):
        return False  # unverified never passes
    return value >= limit if op == "min" else value <= limit


def passes(m: Metrics, config: EvalConfig | None = None) -> dict[str, bool]:
    """Formal acceptance: a census proportion or a completed sample's confidence bound.

    A point estimate alone, unknown evidence, or an incomplete sample never passes.
    """
    cfg = config if config is not None else load_eval_config()
    result = {name: _meets(getattr(m, name), cfg.targets[name]) for name in METRIC_NAMES}
    name = "finish_accuracy_unflagged"
    method = m.finish_accuracy_unflagged_method
    evidence = NAN
    if method == "exact":
        evidence = m.finish_accuracy_unflagged
    elif method == "audit_sample":
        evidence = (m.finish_accuracy_unflagged_low if cfg.targets[name][0] == "min"
                    else m.finish_accuracy_unflagged_high)
    result[name] = _meets(evidence, cfg.targets[name])
    result["speed_ratio"] = (m.speed_evidence_complete and m.speed_violation is None
                             and _meets(m.speed_ratio, cfg.targets["speed_ratio"]))
    return result


def speed_verdict(m: Metrics, config: EvalConfig) -> str:
    """An over-budget upper bound alone is uncertainty, never an individual failure."""
    if m.speed_violation == "exact":
        return "FAIL"
    if m.speed_violation is not None:
        return f"FAIL ({m.speed_violation} deadline violation)"
    if passes(m, config)["speed_ratio"]:
        return "PASS"
    return "UNVERIFIED (FAIL)"


# --------------------------------------------------------------------------- R60 metrics


@dataclass
class ReportMetrics:
    latency_percentile: float
    videos: list[dict]
    excluded: dict[str, list[str]]
    core_s: dict
    core_ratio: dict
    total_s: dict
    total_ratio: dict
    speed_pass_rate: float
    core_pass_rate: float
    vlm_request_s: dict
    vlm_unrecorded_videos: list[str]
    vlm_trigger_rate: float
    n_vlm_rounds: int
    flagged_vlm: dict
    referee: dict
    match_agreement: dict | None
    wall: dict  # R61 batch wall-clock basis
    unflagged: dict  # R62 unbiased unflagged accuracy
    tradeoff: dict  # R63 joint block


def _stat(values: list[float], percentile: float) -> dict:
    if not values:
        return {"median": NAN, "p": NAN}
    return {"median": float(statistics.median(values)),
            "p": float(np.percentile(values, percentile))}


def _accuracy(rows: list[dict]) -> dict:
    correct = sum(map(_finish_right, rows))
    return {"n": len(rows), "correct": correct, "accuracy": _ratio(correct, len(rows))}


def _vlm_requests(rows, timings, percentile):
    judged = {r["video"] for r in real_rows(rows)}
    entries = {v: e for v, e in (timings or {}).items() if v in judged and isinstance(e, dict)}
    unrecorded = sorted(v for v, e in entries.items() if "vlm_s" in e and "vlm_requests" not in e)
    requests = [(v, q) for v, e in sorted(entries.items()) for q in e.get("vlm_requests", [])]
    if timings is None or unrecorded:
        empty = {"n": 0, "median": NAN}
        return ({"n": 0, "n_failed": 0, "median": NAN, "p": NAN, "max": NAN,
                 "cold": empty, "warm": dict(empty)}, unrecorded, None)
    seconds = [q["seconds"] for _, q in requests]

    def part(cold: bool) -> dict:
        s = [q["seconds"] for _, q in requests if q["ok"] and q["cold"] is cold]
        return {"n": len(s), "median": float(statistics.median(s)) if s else NAN}

    stats = {
        "n": len(seconds),
        "n_failed": sum(not q["ok"] for _, q in requests),
        **_stat(seconds, percentile),
        "max": float(max(seconds)) if seconds else NAN,
        "cold": part(True),
        "warm": part(False),
    }
    return stats, unrecorded, {(v, q["round"]) for v, q in requests}


def report_metrics(rows: list[dict], timings: dict[str, dict] | None, config: EvalConfig,
                   *, rules: Rules | None = None,
                   runs: list[dict] | None = None) -> ReportMetrics:
    """Latency, VLM and referee-accuracy metrics for the eval report (R60).

    Core = ``processing_s`` (excludes one-time model init and scan/probe), end-to-end =
    core + ``vlm_s``; VLM latency is per request actually sent (``vlm_requests``).
    """
    pct = config.latency_percentile
    videos, excluded = _speed_videos(rows, timings, config.speed_max_video_s)
    limit = config.targets["speed_ratio"][1]
    real, truth = real_rows(rows), truth_rows(rows)
    requests, unrecorded, sent = _vlm_requests(rows, timings, pct)
    keys = {(r["video"], r["round"]) for r in real}
    trigger = NAN if sent is None else _ratio(len(sent & keys), len(real))
    flagged = [r for r in real if r["needs_review"] is True]
    notes = [r.get("vlm_note") for r in flagged]
    answered = sum(isinstance(n, str) and not n.startswith(UNAVAILABLE_PREFIX) for n in notes)
    unavailable = sum(isinstance(n, str) and n.startswith(UNAVAILABLE_PREFIX) for n in notes)
    agreement = None
    if rules is not None:
        original = group_matches([row_to_result(r) for r in rows], rules)
        final = matches_for(rows, rules)
        reviewed = {(r["video"], r["round"]) for r in truth}
        full = [(i, m) for i, m in enumerate(final)
                if m.rounds and all((r.video, r.round) in reviewed for r, _ in m.rounds)]
        agree = sum(
            i < len(original) and (original[i].score, original[i].winner) == (m.score, m.winner)
            for i, m in full
        )
        agreement = {"n_matches": len(full), "agree": agree, "ratio": _ratio(agree, len(full)),
                     "n_partial": len(final) - len(full)}
    wall = _wall(rows, timings, runs, config.speed_max_video_s)
    u = unflagged_accuracy(rows, config.audit_confidence_level)
    pass_rate = _ratio(sum(v["total_ratio"] <= limit for v in videos), len(videos))
    speed = {"per_video_pass_rate": pass_rate, "per_video_basis": "sum-based stage time",
             "n_eligible": len(videos),
             "observed": _observed_speed(rows, timings, config.speed_max_video_s),
             "deadline": _deadline_speed(rows, timings, config.speed_max_video_s),
             "throughput_ratio": wall["ratio"],
             "throughput_basis": wall["basis"],
             "throughput_pass": wall["ratio"] <= limit if math.isfinite(wall["ratio"]) else None}
    return ReportMetrics(
        latency_percentile=pct,
        videos=videos,
        excluded=excluded,
        core_s=_stat([v["core_s"] for v in videos], pct),
        core_ratio=_stat([v["core_ratio"] for v in videos], pct),
        total_s=_stat([v["total_s"] for v in videos], pct),
        total_ratio=_stat([v["total_ratio"] for v in videos], pct),
        speed_pass_rate=pass_rate,
        core_pass_rate=_ratio(sum(v["core_ratio"] <= limit for v in videos), len(videos)),
        vlm_request_s=requests,
        vlm_unrecorded_videos=unrecorded,
        vlm_trigger_rate=trigger,
        n_vlm_rounds=0 if sent is None else len(sent & keys),
        flagged_vlm={
            "n_flagged": len(flagged), "answered": answered, "unavailable": unavailable,
            "not_asked": len(flagged) - answered - unavailable,
            "answered_share": _ratio(answered, len(flagged)),
            "unavailable_share": _ratio(unavailable, len(flagged)),
        },
        referee={
            "all": _accuracy(truth),
            "unflagged": _accuracy([r for r in truth if r["needs_review"] is False]),
            "flagged": _accuracy([r for r in truth if r["needs_review"] is True]),
        },
        match_agreement=agreement,
        wall=wall,
        unflagged=u,
        tradeoff=_tradeoff(real, u, speed),
    )


def _tradeoff(real: list[dict], u: dict, speed: dict) -> dict:
    """Original predictions, review coverage and speed, without unsupported extrapolation."""
    n = len(real)
    unflagged = [r for r in real if r["needs_review"] is False]
    flagged = [r for r in real if r["needs_review"] is True]
    flagged_reviewed = [r for r in flagged if r.get("reviewed") is True]
    reviewed = [r for r in real if r.get("reviewed") is True]
    correct = sum(map(_finish_right, reviewed))
    return {
        "automation_rate": _ratio(len(unflagged), n),
        "review_ratio": _ratio(len(flagged), n),
        "overall_original_accuracy": _ratio(correct, n) if len(reviewed) == n else NAN,
        "known_correct_original_lower_bound": _ratio(correct, n) if reviewed else NAN,
        "reviewed_output_agreement": {"n": len(reviewed), "accuracy": 1.0 if reviewed else NAN},
        "n_truth": len(reviewed), "n_unknown_truth": n - len(reviewed),
        "flagged_reviewed": (len(flagged_reviewed), len(flagged)),
        "unflagged_method": u["method"],
        "speed": speed,
    }


# --------------------------------------------------------------------------- errors


def _entry(row: dict, **extra) -> dict:
    return {"video": row["video"], "round": row["round"], **extra,
            "needs_review": row["needs_review"], "review_reasons": row.get("review_reasons") or []}


def error_groups(rows: list[dict]) -> dict[str, list[dict]]:
    """Reviewed real rounds where the original prediction disagrees with the truth."""
    groups: dict[str, list[dict]] = {name: [] for name in _ERROR_GROUPS}
    for row in truth_rows(rows):
        eff = effective(row)
        flagged = row["needs_review"] is True
        if not flagged and not _finish_right(row):
            groups["finish_wrong_unflagged"].append(_entry(
                row, original=f"{row['finish']}/{row['winner']}",
                truth=f"{eff['finish']}/{eff['winner']}"))
        for side in SIDES:
            key = f"{side.lower()}_touch"
            if eff[key] == "yes" and row[key] != "yes" and not flagged:
                groups["touch_missed"].append(_entry(row, side=side, original=row[key], truth="yes"))
            if row[key] == "yes" and eff[key] == "no":
                groups["touch_false_alarm"].append(_entry(row, side=side, original="yes", truth="no"))
        if eff["air_collision"] == "yes" and row["air_collision"] != "yes" and not flagged:
            groups["air_collision_missed"].append(
                _entry(row, original=row["air_collision"], truth="yes"))
        if row["air_collision"] == "yes" and eff["air_collision"] == "no":
            groups["air_collision_false_alarm"].append(_entry(row, original="yes", truth="no"))
        changed = [f for f in TRUTH_FIELDS if row.get(f) != eff.get(f)]
        if flagged and changed:
            groups["corrected_after_flag"].append(_entry(
                row, original={f: row.get(f) for f in changed},
                truth={f: eff.get(f) for f in changed}))
    return groups


# --------------------------------------------------------------------------- R37 truth


@dataclass
class TruthMatch:
    rows: list[dict]  # prediction rows; matched ones carry the reference truth as a fix
    n_truth: int  # distinct reviewed real truth keys
    matched: int
    issues: dict[str, list[tuple[str, int]]]
    coverage: float


def _key(row: dict) -> tuple[str, int]:
    return row["video"], row["round"]


def match_truth(pred_rows: list[dict], truth_rows_: list[dict], tolerance_s: float) -> TruthMatch:
    """Apply an explicit reviewed reference result set to fresh predictions (R37).

    Read-only: returns copies. A truth round is used only for a unique key present in
    both sets whose video has the same real-round count in both (segmentation) and whose
    ``t_launch`` differs by at most ``tolerance_s``. Everything else is reported and left
    without truth (never a nearest-round guess). Reviews stored in the prediction set are
    ignored here: the reference is the only truth.
    """
    issues: dict[str, list[tuple[str, int]]] = {name: [] for name in _TRUTH_ISSUES}
    pred_count = Counter(_key(r) for r in pred_rows)
    pred = {_key(r): r for r in pred_rows}
    truth_count = Counter(_key(r) for r in truth_rows(truth_rows_))
    truth = {_key(r): r for r in truth_rows(truth_rows_)}

    def rounds_per_video(rows):
        return Counter(v for v, _ in {_key(r) for r in real_rows(rows)})

    pred_rounds, ref_rounds = rounds_per_video(pred_rows), rounds_per_video(truth_rows_)
    matched: dict[tuple[str, int], dict] = {}
    for key in sorted(truth):
        if truth_count[key] > 1:
            issues["duplicate_truth_key"].append(key)
        elif pred_count[key] > 1:
            issues["duplicate_prediction_key"].append(key)
        elif key not in pred:
            issues["missing_prediction"].append(key)
        elif is_not_a_round(pred[key].get("review_reasons") or []):
            issues["prediction_not_a_round"].append(key)
        elif pred_rounds[key[0]] != ref_rounds[key[0]]:
            issues["segmentation_mismatch"].append(key)
        elif abs(pred[key]["t_launch"] - truth[key]["t_launch"]) > tolerance_s:
            issues["time_mismatch"].append(key)
        else:
            matched[key] = truth[key]
    rows = []
    for row in pred_rows:
        copy = {**row, "reviewed": False, "reviewer_fix": None}
        ref = matched.get(_key(row)) if pred_count[_key(row)] == 1 else None
        if ref is not None:
            eff = effective(ref)
            fix = {f: eff[f] for f in (*TRUTH_FIELDS, "points") if f in eff}
            if eff.get("match_start") is True:
                fix["match_start"] = True
            copy.update(reviewed=True, reviewer_fix=fix)
        rows.append(copy)
    return TruthMatch(rows=rows, n_truth=len(truth), matched=len(matched), issues=issues,
                      coverage=_ratio(len(matched), len(truth)))


# --------------------------------------------------------------------------- tuning


@dataclass
class ShakeSamples:
    samples: dict[str, list[float]] = field(
        default_factory=lambda: {"normal": [], "touch": [], "bey_impact": []})
    rows_normal: int = 0
    rows_touch: int = 0
    rows_touch_without_sample: int = 0
    rows_unknown_truth: int = 0
    rows_without_evidence: int = 0
    frames_unmeasured: int = 0
    frames_impact_excluded: int = 0


def shake_samples(rows: list[dict], impact_coincide_frames: int) -> ShakeSamples:
    """Stadium-displacement samples from reviewed rounds' R35 ``motion_evidence``.

    Only the recorded half-open launch window is used. Frames within
    ``impact_coincide_frames`` of a recorded physical impact go to ``bey_impact`` (the
    judge never counts them as touches); unmeasured (null) frames are skipped. Rounds
    whose effective truth is no touch on both sides contribute every remaining frame to
    ``normal``; a round with a true touch contributes ONE ``touch`` sample, its largest
    remaining displacement (its quiet frames are not positives). Rounds with unknown
    truth or without evidence are counted and skipped; nothing is fabricated.
    """
    out = ShakeSamples()
    for row in truth_rows(rows):
        ev = row.get("motion_evidence")
        if not isinstance(ev, dict):
            out.rows_without_evidence += 1
            continue
        truth = {effective(row)[f"{s.lower()}_touch"] for s in SIDES}
        if "yes" in truth:
            kind = "touch"
        elif truth == {"no"}:
            kind = "normal"
        else:
            out.rows_unknown_truth += 1
            continue
        lo, hi = ev["window"]
        impacts = ev["impact_frames"]
        clean = []
        for frame in range(lo, hi):
            value = ev["displacements"].get(str(frame))
            if value is None or not math.isfinite(value):
                out.frames_unmeasured += 1
            elif any(abs(frame - i) <= impact_coincide_frames for i in impacts):
                out.frames_impact_excluded += 1
                out.samples["bey_impact"].append(float(value))
            else:
                clean.append(float(value))
        if kind == "normal":
            out.rows_normal += 1
            out.samples["normal"].extend(clean)
        else:
            out.rows_touch += 1
            if clean:
                out.samples["touch"].append(max(clean))
            else:
                out.rows_touch_without_sample += 1
    return out


@dataclass
class ShakeDerivation:
    value: float
    method: str  # midpoint | normal_percentile | insufficient
    overlap: bool
    normal_max: float
    touch_min: float


def derive_shake(samples: dict[str, list[float]], percentile: float | None = None) -> ShakeDerivation:
    """Midpoint between the smallest touch and the largest normal displacement; when they
    overlap, the ``percentile`` of normal (config default) with ``overlap`` set as a
    warning. Without both classes the candidate is ``nan`` (unverified)."""
    normal, touch = samples.get("normal") or [], samples.get("touch") or []
    if not normal or not touch:
        return ShakeDerivation(NAN, "insufficient", False,
                               max(normal) if normal else NAN, min(touch) if touch else NAN)
    hi, lo = max(normal), min(touch)
    if lo > hi:
        return ShakeDerivation((lo + hi) / 2, "midpoint", False, hi, lo)
    if percentile is None:
        percentile = load_eval_config().shake_overlap_percentile
    return ShakeDerivation(float(np.percentile(normal, percentile)), "normal_percentile",
                           True, hi, lo)


def derive_shake_px(samples: dict[str, list[float]], *, percentile: float | None = None) -> float:
    return derive_shake(samples, percentile).value


def thresholds_diff(path: str | Path, name: str, value: float) -> str:
    """Advisory unified diff setting ``name`` to ``value``; the file is never written."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    pattern = re.compile(rf"^({re.escape(name)}:\s*)([^#\s]+)(.*)$", re.DOTALL)
    new = list(lines)
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            if float(m.group(2)) == value:
                return ""
            new[i] = f"{m.group(1)}{round(value, 3)!r}{m.group(3)}"
            break
    else:
        raise ValueError(f"{path}: no {name} entry")
    return "".join(difflib.unified_diff(lines, new, f"a/{path.name}", f"b/{path.name}"))


def spin_dir_check(rows: list[dict], min_samples: int) -> dict:
    """Known-direction accuracy of the spin-direction flag on reviewed rounds (Step 5b).

    ``known``: sides whose truth is left/right; ``original_unknown``: of those, sides the
    judge left unknown (a flag, not a mapping error); accuracy is over the rest. A swap of
    ``orbit_to_spin`` is suggested only below 0.5 with at least ``min_samples`` sides.
    """
    known = judged = correct = unknown = 0
    for row in truth_rows(rows):
        if not row.get("spin_dir_present"):
            continue
        eff = effective(row)
        for side in SIDES:
            key = f"{side.lower()}_spin_dir"
            if eff[key] not in ("left", "right"):
                continue
            known += 1
            if row[key] == "unknown":
                unknown += 1
            else:
                judged += 1
                correct += row[key] == eff[key]
    accuracy = _ratio(correct, judged)
    swap = None if judged < min_samples or not math.isfinite(accuracy) else accuracy < 0.5
    return {"known": known, "original_unknown": unknown, "judged": judged,
            "correct": correct, "accuracy": accuracy, "swap_suggested": swap}


def reversed_finish_check(rows: list[dict]) -> dict:
    """Reviewed spin finishes the judge derived from the LOSER's slow reversal (R33) and
    how many reviewers overturned (finish or winner changed)."""
    n = overturned = 0
    for row in truth_rows(rows):
        if row["finish"] != "spin" or row["winner"] not in SIDES:
            continue
        loser = "B" if row["winner"] == "X" else "X"
        if f"{loser}_stop_reversed" not in (row.get("review_reasons") or []):
            continue
        n += 1
        overturned += not _finish_right(row)
    return {"n": n, "overturned": overturned, "ratio": _ratio(overturned, n)}


# --------------------------------------------------------------------------- reports


def read_results(results_dir: str | Path) -> tuple[list[dict], dict | None, list | None]:
    """Rows, per-video timings and batch run records of a results directory (read-only;
    absent ``videos.json`` / ``runs.json`` give None)."""
    results_dir = Path(results_dir)
    rounds = results_dir / "rounds.json"
    if not rounds.is_file():
        raise ValueError(f"{results_dir}: no rounds.json (run `bxjudge run` first)")
    rows = json.loads(rounds.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{rounds}: expected a list of round mappings")
    from .corrections import AUDIT_META, validate_audit_sample

    audit_path = results_dir / AUDIT_META
    if audit_path.is_file():
        validate_audit_sample(rows, json.loads(audit_path.read_text(encoding="utf-8")))
    elif any(r.get("audit_sample") is True for r in rows):
        raise ValueError(f"{results_dir}: audit marks require frozen audit_sample.json metadata")
    videos = results_dir / "videos.json"
    timings = json.loads(videos.read_text(encoding="utf-8")) if videos.is_file() else None
    runs_path = results_dir / "runs.json"
    runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.is_file() else None
    return rows, timings, runs


def is_synthetic(rows: list[dict]) -> bool:
    """Any row judged by a synthetic-trained detector (R44/R48): smoke wiring only."""
    for row in rows:
        prov = ((row.get("detector_provenance") or {}).get("dataset") or {}).get("provenance") or {}
        if "synthetic_detector" in (row.get("review_reasons") or []) or prov.get("synthetic") is True:
            return True
    return False


def _f(value: float, digits: int = 3) -> str:
    return "unverified" if not math.isfinite(value) else f"{value:.{digits}f}"


def _frac(value: float, num: int, den: int) -> str:
    return f"{_f(value)} ({num}/{den})"


def _target(t: tuple[str, float]) -> str:
    return f"{'≥' if t[0] == 'min' else '≤'} {t[1]:g}"


def _verdict(value: float, ok: bool) -> str:
    if ok:
        return "PASS"
    return "UNVERIFIED (FAIL)" if not math.isfinite(value) else "FAIL"


_SMOKE = (
    "> **SMOKE ONLY — NOT ACCEPTANCE.** These rounds were judged with a synthetic-trained "
    "detector (synthetic_detector, R44/R48). The numbers prove the evaluation wiring only "
    "and say nothing about real-video accuracy or spec section 11 compliance."
)
_TIMING_NOTE = (
    "Timing definitions: **core** = `processing_s` per video (judge, trajectories and result "
    "export; excludes the one-time detector/model init and the scan/probe of the batch; "
    "includes first-inference warm-up). **VLM** = `vlm_s`, the separate VLM phase (frame "
    "decoding + requests; absent = no VLM phase for that video). **Stage sum** = core + VLM "
    "per video (sum-based diagnostic). **Output availability diagnostic** = videos.json "
    "`pipeline_wall` with scope `core_to_final_output`: elapsed perf_counter time from "
    "the beginning of that video's core processing until its final core exports or "
    "applicable VLM notes and stage timing finish in the SAME invocation. Includes "
    "intervening batch work, queue/wait, detector release and preflight; excludes shared "
    "initialization/probe before core, the observation's own JSON write and shared VLM "
    "session cleanup. Null seconds means pending/invalidated. Standalone VLM invalidates "
    "affected observations and cannot extend an earlier invocation. **Batch wall-clock** = "
    "runs.json `wall_s` of each `bxjudge run` "
    "from start to end (config + model init, probe, every video, its VLM phase once) plus "
    "standalone `bxjudge vlm` phases, over the playback duration of the videos judged (R61). "
    "**Complete invocation deadline evidence** = videos.json 'invocation_wall' with "
    "scope 'complete_invocation': the measured interval from before configuration, model "
    "load and scan/probe through all output work and VLM session cleanup. Only its own "
    "timing metadata writes are excluded. Kind 'exact' means one actual processed input; "
    "'completion_upper_bound' means multiple inputs, including placeholders and "
    "over-policy videos. This shared bound is not an individual elapsed point. A bound "
    "within each video's own budget proves compliance; exceeding only a bound leaves "
    "that video UNVERIFIED. 'output_ready_wall' scope 'invocation_to_output_ready' "
    "measures from invocation start until that video's output is ready, before shared "
    "cleanup; it is an observed lower bound, never complete latency. Exact or observed "
    "lower-bound violations FAIL even with incomplete coverage; stage-sum violations "
    "without complete evidence are labelled sum-based. PASS requires complete current "
    "evidence for every eligible video. New attempts and standalone VLM invalidate "
    "affected evidence; historical records cannot be combined or reconstructed. "
    "Use a fresh output directory and run --limit 1 for exact cold benchmarks. "
    "Batch throughput and core-only timing are "
    "diagnostics. Per-video Speed Pass Rate is the fraction of "
    "eligible videos whose own core+VLM stage sum meets its deadline, not a measurement "
    "of per-video wall latency. Repeated independent executions are listed separately; "
    "their wall costs and executed playback durations accumulate (repeated videos count "
    "again), whereas videos.json contains the latest core snapshot plus its VLM phases. "
    "This is execution cost, not a future-parallel throughput guarantee. VLM request latency "
    "is per request actually sent (`vlm_requests`); *cold* = first answer by that model in "
    "the phase (models are unloaded at phase end). Durations are the playback seconds "
    "recorded in videos.json; slow-motion capture is NOT rescaled."
)


def _lat(s: dict, pct: float, unit: str = " s") -> str:
    return f"median {_f(s['median'], 2)}{unit} / p{pct:g} {_f(s['p'], 2)}{unit}"


def _count(value: float, den: int) -> int:
    return round(value * den) if math.isfinite(value) else 0


def render_eval_report(results_dir, rows, timings, metrics: Metrics, ok: dict[str, bool],
                       extra: ReportMetrics, config: EvalConfig, *,
                       truth: TruthMatch | None = None, truth_dir=None, today: str) -> str:
    """Markdown eval report: R60 summary table first, then timing, R37 coverage, errors."""
    m, x, pct = metrics, extra, extra.latency_percentile
    q, fv, ref = extra.vlm_request_s, extra.flagged_vlm, extra.referee
    limit = config.targets["speed_ratio"][1]
    source = f"Results: `{results_dir}`"
    if truth_dir is not None:
        source += f" — truth from reference `{truth_dir}` (R37, read-only)"
    out = [f"# bxjudge 評估報告 eval-{today}", "", source, ""]
    if is_synthetic(rows):
        out += [_SMOKE, ""]
    u, w, t = x.unflagged, x.wall, x.tradeoff
    if u["method"] in ("audit_sample", "incomplete_audit"):
        how = (f"audit_sample: {u['correct']}/{u['n_used']} reviewed of {u['n_sampled']} "
               f"selected, {u['n_unflagged']} unflagged; ")
        how += (f"Wilson {u['level']:.0%} {_f(u['low'])}–{_f(u['high'])}; "
                f"acceptance uses confidence {'lower' if config.targets['finish_accuracy_unflagged'][0] == 'min' else 'upper'} bound"
                if u["method"] == "audit_sample" else
                "INCOMPLETE / UNVERIFIED; point estimate is provisional; cannot pass")
    elif u["method"] == "exact":
        how = (f"exact: all {u['n_unflagged']} unflagged rounds reviewed; "
               "finite batch only, no future-generalization claim")
    else:
        how = f"unverified: {u['reason']}"
        if u["n_sampled"]:
            how += f"; {u['n_used']}/{u['n_sampled']} selected audit rows reviewed"
    biased = u["biased"]
    spec_rows = [
        ("未標覆核回合判定正確率 finish_accuracy_unflagged", "finish_accuracy_unflagged",
         (f"{_f(m.finish_accuracy_unflagged)} [{how}]; reviewed outside the random sample "
          f"(selection-biased diagnostic; included only in a full census) "
          f"{_frac(biased['accuracy'], biased['correct'], biased['n'])}")),
        ("觸碰召回率 touch_recall", "touch_recall",
         _frac(m.touch_recall, _count(m.touch_recall, m.n_touch_truth), m.n_touch_truth)),
        ("空碰召回率 air_recall", "air_recall",
         _frac(m.air_recall, _count(m.air_recall, m.n_air_truth), m.n_air_truth)),
        ("需覆核比例 review_ratio", "review_ratio",
         _frac(m.review_ratio, _count(m.review_ratio, m.n_rounds), m.n_rounds)),
        ("處理速度比 speed_ratio", "speed_ratio",
         (f"{_f(m.speed_ratio)} [{m.speed_basis}; worst eligible video]; "
          f"observed coverage {m.n_observed_videos}/{m.n_required_speed_videos}; "
          f"batch throughput diagnostic {_f(m.speed_ratio_wall)}, "
          f"sum-based median (core+VLM) {_f(m.speed_ratio_sum)} over {m.n_speed_videos} "
          f"video(s), core-only {_f(m.speed_ratio_core)}")),
    ]
    out += ["| 指標 Metric | 數值 Value | 目標 Target | 判定 Result |", "|---|---|---|---|"]
    for label, name, value in spec_rows:
        verdict_value = getattr(m, name)
        if name == "finish_accuracy_unflagged" and u["method"] not in ("exact", "audit_sample"):
            verdict_value = NAN
        verdict = (speed_verdict(m, config) if name == "speed_ratio"
                   else _verdict(verdict_value, ok[name]))
        out.append(f"| {label} | {value} | {_target(config.targets[name])} | {verdict} |")
    if x.vlm_unrecorded_videos:
        vlm_lat = (f"unverified: vlm_s without per-request records for "
                   f"{len(x.vlm_unrecorded_videos)} video(s)")
    else:
        vlm_lat = (f"{_lat(q, pct)} / max {_f(q['max'], 2)} s; n={q['n']} "
                   f"(failed {q['n_failed']}); cold n={q['cold']['n']} median "
                   f"{_f(q['cold']['median'], 2)} s, warm n={q['warm']['n']} median "
                   f"{_f(q['warm']['median'], 2)} s")
    match = ""
    if x.match_agreement is not None:
        a = x.match_agreement
        match = (f"; match score agreement {_frac(a['ratio'], a['agree'], a['n_matches'])} "
                 f"over fully reviewed matches ({a['n_partial']} match(es) not fully "
                 "reviewed, excluded); agreement, not independent final accuracy")
    e2e_ok = sum(v["total_ratio"] <= limit for v in x.videos)
    core_ok = sum(v["core_ratio"] <= limit for v in x.videos)
    wall_note = ""
    if w["sum_exceeds_wall"]:
        wall_note = (f"; NOTE per-video sum {_f(w['sum_s'], 2)} s exceeds wall-clock "
                     f"{_f(w['wall_s'], 2)} s; stage times may overlap and are never added "
                     "to the outer wall-clock")

    acc = {k: _frac(v["accuracy"], v["correct"], v["n"]) for k, v in ref.items()}
    out += [
        (f"| 核心判定延遲 Core Latency | {_lat(x.core_s, pct)}; ratio "
         f"{_lat(x.core_ratio, pct, '')} ({len(x.videos)} video(s); excludes model init + "
         "probe) | — | diagnostic |"),
        f"| VLM 單次請求延遲 VLM Latency | {vlm_lat} | — | — |",
        (f"| VLM 觸發率 VLM Trigger Rate | "
         f"{_frac(x.vlm_trigger_rate, x.n_vlm_rounds, m.n_rounds)} real rounds sent; flagged "
         f"rows answered {_frac(fv['answered_share'], fv['answered'], fv['n_flagged'])}, "
         f"unavailable {_frac(fv['unavailable_share'], fv['unavailable'], fv['n_flagged'])} "
         "| — | — |"),
        (f"| 階段加總延遲 Stage Sum Latency | sum-based per video (core+VLM) "
         f"{_lat(x.total_s, pct)}; ratio {_lat(x.total_ratio, pct, '')}; {w['basis']} "
         f"{_f(w['wall_s'], 2)} s over {_f(w['duration_s'], 2)} s of video = ratio "
         f"{_f(w['ratio'])} ({w['n_runs']} run record(s), VLM phase {_f(w['vlm_wall_s'], 2)} s "
         f"inside, counted once){wall_note} | — | — |"),
        (f"| 速度達標率 Speed Pass Rate | sum-based per-video stage time "
         f"{_frac(x.speed_pass_rate, e2e_ok, len(x.videos))}; core-only "
         f"{_frac(x.core_pass_rate, core_ok, len(x.videos))} (diagnostic) "
         f"| ≤ {limit:g} × duration | — |"),
        (f"| 裁判正確率 Referee Accuracy | all reviewed {acc['all']}; unflagged "
         f"{acc['unflagged']}; flagged {acc['flagged']}{match} | — | — |"),
        "",
        (f"Truth = {m.n_reviewed} reviewed of {m.n_rounds} real rounds; {m.n_placeholders} "
         "diagnostic placeholder row(s) excluded. Unverified = empty denominator; it never "
         "passes. Referee Accuracy is a reviewed-subset diagnostic with selection bias; "
         "it is not overall original accuracy. The unflagged spec metric requires a "
         "completed random audit or a full finite-population census (R62)."),
        "",
        "**取捨 Trade-off (R63)** — side by side, no weighted score:",
        "",
        "| 項目 Item | 數值 Value (interval) | 依據 Coverage |",
        "|---|---|---|",
        (f"| 原始未標覆核正確率 Original Unflagged Accuracy | {_f(u['estimate'])} "
         f"({_f(u['low'])}–{_f(u['high'])}) | {how} |"),
        (f"| 原始需覆核比例 Original Review Ratio | {_f(t['review_ratio'])} | original "
         f"needs_review over {m.n_rounds} real rounds |"),
        (f"| 自動化率 Automation Rate | {_f(t['automation_rate'])} | 1 − review_ratio over "
         f"{m.n_rounds} real rounds |"),
        (f"| 原始整體正確率 Overall Original Accuracy | {_f(t['overall_original_accuracy'])} "
         f"| truth {t['n_truth']}/{m.n_rounds}; requires truth coverage for both groups |"),
        (f"| 已知正確比例保守下界 Known-correct Original Lower Bound | "
         f"{_f(t['known_correct_original_lower_bound'])} | unknown truth counted as errors; "
         "conservative finite-batch bound, not an unbiased accuracy estimate or CI |"),
        (f"| 覆核輸出與覆核參考一致率 Reviewed-output Agreement | "
         f"{_f(t['reviewed_output_agreement']['accuracy'])} | n={t['n_truth']}; fixes compared "
         "to themselves, not independent final accuracy |"),
        (f"| 已覆核子集 Reviewed-subset Diagnostic | {acc['all']} | selection bias; "
         f"flagged truth {t['flagged_reviewed'][0]}/{t['flagged_reviewed'][1]} |"),
        (f"| 每影片速度達標率 Per-video Speed Pass Rate | {_f(x.speed_pass_rate)} | "
         f"sum-based stage time, {e2e_ok}/{len(x.videos)} eligible videos |"),
        (f"| 批次吞吐是否達標 Batch Throughput Pass | {t['speed']['throughput_pass']} "
         f"(ratio {_f(t['speed']['throughput_ratio'])}) | "
         f"{t['speed']['throughput_basis']}; boolean, separate from per-video fraction |"),
        "",
        "## Timing",
        "",
        _TIMING_NOTE,
        "",
    ]
    if w["executions"]:
        out += [
            (f"Recorded execution cost: {_f(w['wall_s'], 2)} s total wall-clock / "
             f"{_f(w['duration_s'], 2)} s executed eligible playback "
             f"({_f(w['unique_duration_s'], 2)} s unique playback). Each record is an "
             "independent execution/experiment, not concurrent work inferred from timestamps; "
             "per-video stage diagnostics are the latest snapshot, not cumulative cost."),
            "",
            "| Execution | Kind | Started | Outer wall s | Eligible playback s | Ratio |",
            "|---|---|---|---|---|---|",
        ]
        out += [f"| {i} | {e['kind']} | {e['started_at']} | {_f(e['wall_s'], 2)} | "
                f"{_f(e['duration_s'], 2)} | {_f(e['ratio'])} |"
                for i, e in enumerate(w["executions"], 1)]
        out.append("")
    if timings is None:
        out += [("No videos.json: per-video stage/latency metrics are unverified. "
                "Recorded runs.json batch wall throughput remains a separate diagnostic; "
                "formal per-video wall-deadline compliance is unverified."), ""]
    deadline = t["speed"]["deadline"]
    out += [(f"Complete invocation coverage: {m.n_deadline_videos}/{m.n_required_speed_videos}; "
             f"missing/invalid: {', '.join(deadline['missing']) or 'none'}."), ""]
    if deadline["videos"]:
        out += ["| Video | Scope | Evidence kind | Seconds | Evidence / own duration |",
                "|---|---|---|---|---|"]
        out += [f"| {v['video']} | complete_invocation | {v['kind']} | {_f(v['seconds'])} | "
                f"{_f(v['ratio'])} |" for v in deadline["videos"]]
        out.append("")
    if deadline["lower_bounds"]:
        out += ["| Video | Invocation-to-output-ready lower bound s | Lower bound / duration |",
                "|---|---|---|"]
        out += [f"| {v['video']} | {_f(v['seconds'])} | {_f(v['ratio'])} |"
                for v in deadline["lower_bounds"]]
        out.append("")
    observation = t["speed"]["observed"]
    out += [(f"Core-to-output availability coverage: {m.n_observed_videos}/{m.n_required_speed_videos}; "
            f"missing/invalid: {', '.join(observation['missing']) or 'none'}."), ""]
    if observation["videos"]:
        out += ["| Observed video | core_to_final_output s | Observed / duration |",
                "|---|---|---|"]
        out += [f"| {v['video']} | {_f(v['seconds'])} | {_f(v['ratio'])} |"
                for v in observation["videos"]]
        out.append("")
    if x.videos:
        out += [("| video | duration s | core s | VLM s | stage sum s | core ratio | "
                 "stage sum ratio |"), "|---|---|---|---|---|---|---|"]
        out += [f"| {v['video']} | {v['duration_s']:.2f} | {v['core_s']:.2f} | "
                f"{v['vlm_s']:.2f}{'' if v['has_vlm'] else ' (no VLM phase)'} | "
                f"{v['total_s']:.2f} | {v['core_ratio']:.3f} | {v['total_ratio']:.3f} |"
                for v in x.videos]
        out.append("")
    out += [f"- excluded from speed ({reason}): {', '.join(names)}"
            for reason, names in x.excluded.items() if names]
    out.append("")
    if truth is not None:
        out += ["## Reference truth matching (R37)", "",
                (f"Coverage: {_frac(truth.coverage, truth.matched, truth.n_truth)} reviewed "
                 f"truth keys matched (t_launch tolerance "
                 f"{config.truth_t_launch_tolerance_s:g} s). Unmatched truths stay "
                 "unverified; neither result set was modified."), ""]
        for issue, keys in truth.issues.items():
            listed = ", ".join(f"{v}#{r}" for v, r in keys) or "none"
            out.append(f"- {issue} ({len(keys)}): {listed}")
        out.append("")
    out += ["## 錯誤列（依原因分組）Errors by cause", ""]
    for name, entries in error_groups(rows).items():
        out.append(f"### {name} ({len(entries)})")
        for e in entries:
            side = f" {e['side']}" if "side" in e else ""
            out.append(f"- {e['video']} round {e['round']}{side}: original {e['original']} "
                       f"→ truth {e['truth']}; needs_review={e['needs_review']}; "
                       f"reasons {e['review_reasons']}")
        out.append("")
    placeholders = [r for r in rows if is_not_a_round(r.get("review_reasons") or [])]
    out += [f"## 未判定影片 Diagnostic placeholders ({len(placeholders)})", ""]
    out += [f"- {r['video']}: {r.get('review_reasons')}" for r in placeholders] or ["- none"]
    out.append("")
    return "\n".join(out)


_PENDING = [
    ("`pocket_confirm_s` (0.5 playback s, about 62 ms real time at the 240 to 30 fps slow "
     "motion) only governs non-rotating removal after a pocket entry; calibrate on reviewed "
     "real finishes."),
    ("`air_evidence_max_missing_fraction` (0.1) is relative to the airborne span and tight "
     "on short spans; consider an absolute-frame floor once reviewed real air collisions "
     "exist."),
    ("Detector training batch sizes (config/detection.yaml) were measured on replicated "
     "synthetic data; recheck VRAM (<21 GB total card use) on the first real training."),
    "`orbit_to_spin`: confirm with known right/left-spin rounds (Step 5b, section above).",
    ("`calibration_min_similarity` and every per-frame/per-second launch, landing and stop "
     "threshold: real footage is 240 fps capture played at about 30 fps; thresholds are in "
     "playback units and were never rescaled. Tune only from reviewed real rounds."),
    "`shake_px`: a candidate is meaningful only when derived from reviewed REAL rounds.",
]


def render_tune_report(results_dir, rows, samples: ShakeSamples, derivation: ShakeDerivation,
                       diff: str, current: float, spin: dict, reversed_: dict, *,
                       impact_coincide_frames: int, thresholds_path,
                       truth: TruthMatch | None, today: str) -> str:
    """Markdown tune report: shake_px candidate + advisory diff, spin/reversal checks."""
    s, d = samples, derivation
    out = [f"# bxjudge 門檻校正建議 tune-{today}", "",
           (f"Results: `{results_dir}`. **Advisory only**: `{thresholds_path}` is never "
            "overwritten; apply a diff by hand after review."), ""]
    if is_synthetic(rows):
        out += [_SMOKE, ""]
    if truth is not None:
        coverage = _frac(truth.coverage, truth.matched, truth.n_truth)
        out += [f"Reference truth coverage (R37): {coverage}.", ""]
    out += [
        "## shake_px (stadium shake threshold)", "",
        ("Samples come from each reviewed round's recorded R35 `motion_evidence` (actual "
         "half-open launch window, raw displacements, physical impact frames); notes are "
         f"never parsed as numbers. Frames within ±{impact_coincide_frames} of an impact go "
         "to bey_impact (never touches); null displacements are skipped. A no-touch round "
         "gives every remaining frame as normal; a true-touch round gives ONE sample (its "
         "largest remaining displacement)."), "",
        (f"- rounds: normal {s.rows_normal}, touch {s.rows_touch} (without a clean frame "
         f"{s.rows_touch_without_sample}); skipped: unknown truth {s.rows_unknown_truth}, "
         f"no motion evidence {s.rows_without_evidence}"),
        (f"- frames: normal {len(s.samples['normal'])}, touch samples "
         f"{len(s.samples['touch'])}, bey_impact {len(s.samples['bey_impact'])} "
         f"(impact-excluded {s.frames_impact_excluded}), unmeasured {s.frames_unmeasured}"),
        (f"- max normal {_f(d.normal_max)} px, min touch {_f(d.touch_min)} px, current "
         f"shake_px {current:g}"),
    ]
    if d.method == "insufficient":
        out.append("- candidate: **unverified**: needs reviewed no-touch AND true-touch "
                   "rounds with motion evidence; no change proposed.")
    else:
        out.append(f"- candidate ({d.method}): {d.value:.3f} px")
    if d.overlap:
        out.append("- **WARNING: touch and normal samples overlap**; the candidate is the "
                   "normal percentile and will miss some touches. Review those rounds first.")
    swap = {None: "not enough reviewed known-direction sides",
            True: "**suggested** (accuracy < 0.5)", False: "no"}[spin["swap_suggested"]]
    out += [
        "", "```diff", diff.rstrip() or "(no change proposed)", "```", "",
        "## 旋轉方向 spin direction flag (orbit_to_spin, Step 5b)", "",
        (f"- known-direction sides {spin['known']}; judge left unknown "
         f"{spin['original_unknown']}; accuracy "
         f"{_frac(spin['accuracy'], spin['correct'], spin['judged'])}"),
        f"- swap orbit_to_spin: {swap}",
        "",
        "## reversed 轉停被推翻比例 reversal-derived spin finishes (R33)", "",
        (f"- reviewed spin finishes from the loser's slow reversal: {reversed_['n']}; "
         f"overturned {_frac(reversed_['ratio'], reversed_['overturned'], reversed_['n'])}. "
         "They stay review-required until real known-spin evidence justifies less "
         "conservatism."),
        "",
        "## 待人工覆核資料校正 Pending human-reviewed calibration", "",
    ]
    out += [f"- {item}" for item in _PENDING]
    out.append("")
    return "\n".join(out)
