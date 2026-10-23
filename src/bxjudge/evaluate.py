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

from .audit import audit_population, validate_audit
from .config import config_loader
from .output import ANALYSIS_STATUSES, effective, matches_for, row_to_result
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
    audit_min_n: int  # R64: smallest fully reviewed audit sample that can PASS


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _keys(path, where, raw, expected) -> None:
    if not isinstance(raw, dict) or set(raw) != set(expected):
        got = sorted(raw) if isinstance(raw, dict) else raw
        raise ValueError(f"{path}: {where} must have exactly {sorted(expected)}, got {got!r}")


@config_loader
def load_eval_config(path: str | Path = "config/evaluation.yaml") -> EvalConfig:
    """Load evaluation policy; unknown/missing keys or invalid values raise ``ValueError``."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _keys(path, "top level", raw, (
        "targets", "speed_max_video_s", "latency_percentile", "truth_t_launch_tolerance_s",
        "shake_overlap_percentile", "spin_min_samples", "audit_confidence_level",
        "audit_min_n",
    ))
    level = raw["audit_confidence_level"]
    if not _number(level) or not 0 < level < 1:
        raise ValueError(f"{path}: audit_confidence_level must be in (0, 1)")
    if type(raw["audit_min_n"]) is not int or raw["audit_min_n"] < 1:
        raise ValueError(f"{path}: audit_min_n must be a positive int, got {raw['audit_min_n']!r}")
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
        audit_min_n=raw["audit_min_n"],
    )


# --------------------------------------------------------------------------- metrics


@dataclass
class Metrics:
    # Unflagged finish+winner accuracy (R62): exact when every unflagged real round is
    # reviewed, else the reviewed random audit sample's estimate, else unverified (nan).
    finish_accuracy_unflagged: float
    touch_recall: float
    air_recall: float
    review_ratio: float
    # R66a: max(worst observed per-video ratio, pooled whole-batch ratio).
    # Missing, failed or pending evidence is unverified (nan), never PASS.
    speed_ratio: float
    n_reviewed: int
    n_rounds: int = 0
    n_placeholders: int = 0
    n_unflagged_reviewed: int = 0
    n_touch_truth: int = 0
    n_air_truth: int = 0
    n_speed_videos: int = 0
    n_vlm_videos: int = 0
    speed_ratio_core: float = NAN  # diagnostic: median core-only ratio
    speed_ratio_sum: float = NAN  # median per-video (core + VLM) / duration
    speed_ratio_wall: float = NAN  # required whole-batch wall / current playback (R66a)
    speed_ratio_median: float = NAN  # diagnostic: median observed per-video wall ratio (R66)
    speed_basis: str = "unverified"  # worst-video+batch | unverified
    speed_slowest_video: str | None = None  # highest observed ratio (even when unverified)
    speed_ratio_worst_observed: float = NAN  # its ratio
    n_wall_videos: int = 0  # required real videos with an observed wall_s
    n_wall_missing: int = 0  # required real videos without one (-> unverified)
    finish_accuracy_unflagged_method: str = "unverified"  # exact | audit_sample | unverified
    # R64 support for the verdict: Wilson bounds of the audit estimate (= the value when
    # exact) and the reviewed share of the audit sample.
    finish_accuracy_unflagged_low: float = NAN
    finish_accuracy_unflagged_high: float = NAN
    finish_accuracy_unflagged_coverage: float = NAN
    # Calibration (spec 6.6b): vetoed / failed calibrations never leave the population.
    n_vetoed_rounds: int = 0  # real rounds judged with a vetoed snapshot (reviewed or not)
    # Per-video failure counts from videos.json ``analysis_status`` (R68); None = unverified
    # (no videos.json, or an entry without analysis_status: never inferred).
    n_recalibrate_videos: int | None = None  # latest attempt: calibration selection failed
    n_vetoed_groups: int = 0  # vetoed calibration snapshots
    n_vetoed_videos: int = 0  # videos recorded (videos.json) with a vetoed snapshot
    # videos whose latest attempt succeeded with a non-vetoed snapshot / all videos in
    # videos.json; nan = unverified (as above)
    coverage: float = NAN
    # recalibrate + decode/analysis failures + vetoed videos (await recalibration or a rerun)
    n_pending_videos: int | None = None
    n_decode_error_videos: int | None = None  # latest attempt: decode / probe failure
    n_analysis_error_videos: int | None = None  # latest attempt: unexpected analysis error


_FAILURE_STATUSES = ("recalibrate", "decode_error", "analysis_error", "pending")
STATUS_UNVERIFIED = "unverified (no analysis_status; rerun bxjudge run)"


def _vetoed_pending(row: dict, vetoed: set[str]) -> bool:
    """An unreviewed round judged with a vetoed calibration: it needs human work (6.6b)."""
    return row.get("reviewed") is not True and row.get("calibration_id") in vetoed


def calibration_coverage(rows: list[dict], timings: dict[str, dict] | None,
                         vetoed: set[str] | None = None) -> dict:
    """Spec 6.6b calibration summary: vetoed rounds/videos/groups, per-video failures of the
    latest analysis attempt (``recalibrate`` / ``decode_error`` / ``analysis_error``: video
    level, R53), effective coverage = videos of ``videos.json`` whose latest attempt succeeded
    (``ok``) with a recorded, non-vetoed snapshot / all of them, and pending = failed + vetoed
    videos. Vetoed videos come from ``videos.json`` (else, without it, from the vetoed rows).

    Ruling R68: failures and coverage come only from ``videos.json[video]["analysis_status"]``
    (a failed rerun may keep a video's rows and calibration, and a reviewed round 1 may hide
    the placeholder, T4-M4). When ``videos.json`` is missing or any entry lacks that status,
    ``status_known`` is False: the failure lists, pending and coverage are unverified
    (``None`` / nan), never inferred from ``calibration: null`` or placeholder rows."""
    vetoed = set(vetoed or ())
    real = real_rows(rows)
    vetoed_rows = [r for r in real if r.get("calibration_id") in vetoed]

    def recorded(entry) -> str | None:
        cal = entry.get("calibration") if isinstance(entry, dict) else None
        return cal.get("id") if isinstance(cal, dict) else None

    def status(entry) -> str | None:
        return entry.get("analysis_status") if isinstance(entry, dict) else None

    status_known = timings is not None and all(status(e) in ANALYSIS_STATUSES for e in timings.values())
    failures: dict[str, list[str] | None] = {s: None for s in _FAILURE_STATUSES}
    covered: list[str] | None = None
    total = 0
    if timings is None:
        vetoed_videos = {r["video"] for r in vetoed_rows}
    else:
        vetoed_videos = {v for v, e in timings.items() if recorded(e) in vetoed}
        total = len(timings)
    if status_known:
        failures = {s: sorted(v for v, e in timings.items() if status(e) == s)
                    for s in _FAILURE_STATUSES}
        covered = sorted(v for v, e in timings.items() if status(e) == "ok"
                         and recorded(e) is not None and recorded(e) not in vetoed)
    pending = (None if not status_known
               else sorted(vetoed_videos.union(*failures.values())))
    groups = {}
    for sid in sorted(vetoed):
        group_rows = [r for r in vetoed_rows if r["calibration_id"] == sid]
        # Same source as vetoed_videos: videos.json when present, else the rows.
        videos = ({r["video"] for r in group_rows} if timings is None
                  else {v for v, e in timings.items() if recorded(e) == sid})
        groups[sid] = {"videos": sorted(videos), "rounds": len(group_rows),
                       "unreviewed": sum(_vetoed_pending(r, vetoed) for r in group_rows)}
    return {
        "n_vetoed_rounds": len(vetoed_rows),
        "n_vetoed_unreviewed": sum(_vetoed_pending(r, vetoed) for r in vetoed_rows),
        "status_known": status_known,
        "recalibrate_videos": failures["recalibrate"],
        "decode_error_videos": failures["decode_error"],
        "analysis_error_videos": failures["analysis_error"],
        "vetoed_videos": sorted(vetoed_videos),
        "groups": groups,
        "n_covered": None if covered is None else len(covered),
        "n_videos": total,
        "coverage": NAN if covered is None else _ratio(len(covered), total),
        "pending_videos": pending,
    }


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
    added for the end-to-end total. Durations are playback seconds as recorded.
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
    """R66a: max(retained judgment interval, latest attempt interval) / current probe.

    Failed or pending attempts remain unverified. Their valid observations are retained
    as diagnostics, so a cheap failure cannot hide the slow rows it preserved.
    """
    videos, missing, failed = [], [], []
    for video in sorted({r["video"] for r in real_rows(rows)}):
        entry = (timings or {}).get(video)
        entry = entry if isinstance(entry, dict) else {}
        duration = entry.get("duration_s")
        judged, attempt = entry.get("judged_wall_s"), entry.get("attempt_wall_s")
        status = entry.get("analysis_status")
        core, vlm = entry.get("processing_s"), entry.get("vlm_s", 0.0)
        if _number(duration) and duration > max_s:
            continue
        if not (_number(duration) and duration > 0 and _number(judged) and judged > 0
                and _number(attempt) and attempt > 0 and status in ANALYSIS_STATUSES
                and _number(core) and core > 0 and _number(vlm) and vlm >= 0):
            missing.append(video)
            continue
        if status != "ok":
            failed.append(video)
        wall = max(judged, attempt)
        videos.append({"video": video, "duration_s": duration,
                       "judged_wall_s": judged, "attempt_wall_s": attempt,
                       "observed_s": wall, "ratio": wall / duration, "status": status})
    slowest = max(videos, key=lambda v: v["ratio"]) if videos else None
    return {
        "videos": videos, "missing": missing, "failed": failed,
        "complete": bool(videos) and not missing and not failed,
        "slowest": slowest["video"] if slowest else None,
        "worst": slowest["ratio"] if slowest else NAN,
        "median": statistics.median(v["ratio"] for v in videos) if videos else NAN,
    }


def _wall(rows: list[dict], timings: dict | None, runs: list[dict] | None,
          max_s: float) -> dict:
    """Whole invocation wall over current eligible playback, each video in its last run.

    No work or overhead is subtracted. Every required measured video must have a valid
    run record; standalone VLM invocations add wall without adding playback duration.
    """
    eligible = {v["video"]: v for v in _speed_videos(rows, timings, max_s)[0]}
    valid = isinstance(runs, list) and bool(runs) and all(
        isinstance(r, dict) and r.get("kind") in {"run", "vlm"}
        and _number(r.get("wall_s")) and r["wall_s"] > 0
        and (r.get("vlm_wall_s") is None or (
            _number(r["vlm_wall_s"]) and 0 <= r["vlm_wall_s"] <= r["wall_s"]))
        and isinstance(r.get("videos"), dict)
        and all(isinstance(v, str) and _number(d) and d >= 0 for v, d in r["videos"].items())
        for r in runs or []
    )
    run_records = [r for r in runs or [] if isinstance(r, dict) and r.get("kind") == "run"] if valid else []
    last = {v: i for i, r in enumerate(run_records) for v in r["videos"]}
    counted = sorted(v for v in eligible if v in last)
    used_ids = {last[v] for v in counted}
    used = [run_records[i] for i in sorted(used_ids)]
    if used:
        used += [r for r in runs if r["kind"] == "vlm"]
    excluded = sorted(v for v, i in last.items() if i in used_ids and v not in eligible)
    if not valid:
        reason = "missing or invalid run records"
    elif not eligible:
        reason = "no eligible video in run records"
    elif set(counted) != set(eligible):
        reason = "missing batch evidence for measured videos"
    elif any(run_records[last[v]]["videos"][v] <= 0 for v in counted):
        reason = "invalid measured playback in run records"
    else:
        reason = None
    wall = sum(r["wall_s"] for r in used)
    duration = sum(eligible[v]["duration_s"] for v in counted)
    sum_s = sum(eligible[v]["total_s"] for v in counted) if counted else NAN
    return {
        "n_runs": len(used), "n_videos": len(counted), "excluded": excluded,
        "reason": reason, "wall_s": wall if used else NAN,
        "raw_wall_s": wall if used else NAN,
        "vlm_wall_s": sum(r["vlm_wall_s"] or 0.0 for r in used) if used else NAN,
        "duration_s": duration if used else NAN,
        "ratio": wall / duration if reason is None and duration > 0 else NAN,
        "sum_s": sum_s,
        "sum_ratio": sum_s / duration if counted and duration > 0 else NAN,
        "sum_exceeds_wall": bool(used and counted and sum_s > wall),
    }


def _speed_gate(observed: dict, wall: dict) -> tuple[str, float]:
    complete = observed["complete"] and math.isfinite(wall["ratio"])
    return ("worst-video+batch", max(observed["worst"], wall["ratio"])) if complete else ("unverified", NAN)


def _wilson(correct: int, n: int, level: float) -> tuple[float, float]:
    z = statistics.NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def audit_invalidation(rows: list[dict], audit: dict | None) -> str | None:
    """R69: why the frozen audit sample no longer stands, else None.

    Every selected (video, round) must still be a row of ``rounds.json`` that is marked
    ``audit_sample``, and no other row may carry the mark; a successful re-analysis replaces
    unreviewed rows (dropping their marks), which is what this detects.
    """
    if audit is None:
        if any(r.get("audit_sample") is True for r in rows):
            raise ValueError("audit metadata required for audit_sample marks; pass read_audit(results_dir)")
        return None
    validate_audit(audit)
    if audit_population(rows) != sorted(audit["population"]):
        return "unflagged population changed since the frozen audit draw"
    selection = {tuple(k) for k in audit["selection"]}
    marked = {(r["video"], r["round"]) for r in rows if r.get("audit_sample") is True}
    missing = selection - {(r["video"], r["round"]) for r in rows}
    unmarked = (selection - missing) - marked
    if not (missing or unmarked or marked - selection):
        return None
    return (f"{len(missing)} selected round(s) missing and {len(unmarked)} no longer marked "
            f"(selection of {len(selection)}, {len(marked - selection)} stray mark(s))")


def unflagged_accuracy(rows: list[dict], level: float | None = None,
                       audit: dict | None = None) -> dict:
    """Unflagged finish+winner accuracy without selection bias (R62).

    ``exact`` when every unflagged real round is reviewed (C5-1: the exact value needs no sample,
    so a changed audit sample cannot invalidate it); else ``audit_sample``: the
    reviewed members of the seeded random audit sample, with a Wilson interval at
    ``level`` (config default); else unverified. Reviewed unflagged rounds outside the
    sample were chosen by people, so they are reported separately as ``biased``.
    """
    unflagged = [r for r in real_rows(rows) if r["needs_review"] is False]
    reviewed = [r for r in unflagged if r.get("reviewed") is True]
    sampled = [r for r in unflagged if r.get("audit_sample") is True]
    invalid = audit_invalidation(rows, audit)
    sampled_reviewed = [r for r in sampled if r.get("reviewed") is True]
    biased = [r for r in reviewed if r.get("audit_sample") is not True]
    out = {"n_unflagged": len(unflagged), "n_reviewed": len(reviewed),
           "n_sampled": len(sampled), "biased": _accuracy(biased), "reason": None,
           "level": None,
           # share of the audit sample already reviewed (< 1 = warn in the report)
           "coverage": _ratio(len(sampled_reviewed), len(sampled))}
    if unflagged and len(reviewed) == len(unflagged):
        k = sum(map(_finish_right, reviewed))
        p = k / len(reviewed)
        return {**out, "method": "exact", "n_used": len(reviewed), "correct": k,
                "estimate": p, "low": p, "high": p}
    if invalid is not None:  # R69: a sample estimate is never trusted, never a PASS
        return {**out, "method": "invalidated", "n_used": 0, "correct": 0, "estimate": NAN,
                "low": NAN, "high": NAN, "reason": f"audit sample invalidated: {invalid}",
                "audit_n": audit.get("n"), "audit_seed": audit.get("seed")}
    if sampled_reviewed:
        level = load_eval_config().audit_confidence_level if level is None else level
        k = sum(map(_finish_right, sampled_reviewed))
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
             config: EvalConfig | None = None, runs: list[dict] | None = None,
             vetoed: set[str] | None = None, audit: dict | None = None) -> Metrics:
    """Spec section 11 metrics from flat ``rounds.json`` rows (+ ``videos.json`` timings
    and ``runs.json`` wall-clock records).

    Speed (R66a) gates both the worst continuous per-video interval and whole-batch ratio.
    Missing/failed current attempt evidence or missing batch records is unverified. ``config`` defaults to
    ``config/evaluation.yaml``. ``vetoed`` calibration snapshot ids (spec 6.6b): their rounds
    stay in every population with their original predictions; the unreviewed ones also count
    as needing human work in ``review_ratio``; the calibration fields summarise coverage.
    """
    vetoed = set(vetoed or ())
    real, truth = real_rows(rows), truth_rows(rows)
    touch = [_caught(r, "touch", s) for r in truth for s in SIDES]
    touch = [c for positive, c in touch if positive]
    air = [c for positive, c in (_caught(r, "air") for r in truth) if positive]
    speed = NAN, NAN
    n_speed = n_vlm = 0
    if timings is not None:
        cfg = config if config is not None else load_eval_config()
        videos, _ = _speed_videos(rows, timings, cfg.speed_max_video_s)
        if videos:
            speed = (statistics.median(v["total_ratio"] for v in videos),
                     statistics.median(v["core_ratio"] for v in videos))
        n_speed, n_vlm = len(videos), sum(v["has_vlm"] for v in videos)
        wall = _wall(rows, timings, runs, cfg.speed_max_video_s)
        observed = _observed_speed(rows, timings, cfg.speed_max_video_s)
    else:
        wall = {"ratio": NAN}  # no videos.json: no valid batch denominator
        observed = _observed_speed(rows, None, math.inf)  # every real video is missing
    basis, speed_gate = _speed_gate(observed, wall)
    u = unflagged_accuracy(rows, None if config is None else config.audit_confidence_level,
                           audit)
    cal = calibration_coverage(rows, timings, vetoed)
    return Metrics(
        finish_accuracy_unflagged=u["estimate"],
        touch_recall=_ratio(sum(touch), len(touch)),
        air_recall=_ratio(sum(air), len(air)),
        review_ratio=_ratio(
            sum(r["needs_review"] is True or _vetoed_pending(r, vetoed) for r in real), len(real)
        ),
        speed_ratio=speed_gate,
        n_reviewed=len(truth),
        n_rounds=len(real),
        n_placeholders=len(rows) - len(real),
        n_unflagged_reviewed=u["n_used"],
        n_touch_truth=len(touch),
        n_air_truth=len(air),
        n_speed_videos=n_speed,
        n_vlm_videos=n_vlm,
        speed_ratio_core=speed[1],
        speed_ratio_sum=speed[0],
        speed_ratio_wall=wall["ratio"],
        speed_ratio_median=observed["median"],
        speed_basis=basis,
        speed_slowest_video=observed["slowest"],
        speed_ratio_worst_observed=observed["worst"],
        n_wall_videos=len(observed["videos"]),
        n_wall_missing=len(observed["missing"]),
        finish_accuracy_unflagged_method=u["method"],
        finish_accuracy_unflagged_low=u["low"],
        finish_accuracy_unflagged_high=u["high"],
        finish_accuracy_unflagged_coverage=u["coverage"],
        n_vetoed_rounds=cal["n_vetoed_rounds"],
        n_recalibrate_videos=_n_videos(cal["recalibrate_videos"]),
        n_vetoed_groups=len(vetoed),
        n_vetoed_videos=len(cal["vetoed_videos"]),
        coverage=cal["coverage"],
        n_pending_videos=_n_videos(cal["pending_videos"]),
        n_decode_error_videos=_n_videos(cal["decode_error_videos"]),
        n_analysis_error_videos=_n_videos(cal["analysis_error_videos"]),
    )


def _n_videos(videos: list[str] | None) -> int | None:
    return None if videos is None else len(videos)


def _meets(value: float, target: tuple[str, float]) -> bool:
    op, limit = target
    if not math.isfinite(value):
        return False  # unverified never passes
    return value >= limit if op == "min" else value <= limit


PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
INVALIDATED = "INVALIDATED"  # R69: the frozen audit sample no longer stands (never a PASS)


def _accuracy_verdict(m: Metrics, cfg: EvalConfig) -> str:
    """R64: the point estimate alone never passes. PASS needs the exact value, or a FULLY
    reviewed audit sample of at least ``audit_min_n`` whose Wilson bound also meets the
    target; a point estimate that meets the target without that support is INCONCLUSIVE."""
    target = cfg.targets["finish_accuracy_unflagged"]
    if m.finish_accuracy_unflagged_method == "invalidated":
        return INVALIDATED
    if not _meets(m.finish_accuracy_unflagged, target):
        return FAIL
    if m.finish_accuracy_unflagged_method == "exact":
        return PASS
    bound = m.finish_accuracy_unflagged_low if target[0] == "min" else m.finish_accuracy_unflagged_high
    if (m.finish_accuracy_unflagged_method == "audit_sample"
            and m.finish_accuracy_unflagged_coverage == 1
            and m.n_unflagged_reviewed >= cfg.audit_min_n and _meets(bound, target)):
        return PASS
    return INCONCLUSIVE


def passes(m: Metrics, config: EvalConfig | None = None) -> dict[str, str]:
    """Spec section 11 verdict per metric: ``PASS`` / ``FAIL`` / ``INCONCLUSIVE``.

    Targets come from ``config/evaluation.yaml``; an unverified (nan) value is ``FAIL``.
    Only ``finish_accuracy_unflagged`` can be ``INCONCLUSIVE`` (R64, 樣本不足).
    """
    cfg = config if config is not None else load_eval_config()
    out = {name: PASS if _meets(getattr(m, name), cfg.targets[name]) else FAIL
           for name in METRIC_NAMES}
    out["finish_accuracy_unflagged"] = _accuracy_verdict(m, cfg)
    return out


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
    wall: dict  # R61 batch wall-clock (diagnostic since R66)
    observed: dict  # R66 observed per-video wall (_observed_speed)
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
                   runs: list[dict] | None = None,
                   vetoed: set[str] | None = None,
                   audit: dict | None = None) -> ReportMetrics:
    """Latency, VLM and referee-accuracy metrics for the eval report (R60).

    Core = ``processing_s`` (excludes one-time model init and scan/probe), end-to-end =
    core + ``vlm_s``; VLM latency is per request actually sent (``vlm_requests``).
    ``vetoed`` snapshot ids: their unreviewed rounds need human work in the R63 block, as in
    ``review_ratio`` (spec 6.6b).
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
        # Unreviewed rounds agree with themselves trivially: only matches whose rounds
        # are ALL reviewed are truth; none = unverified (fix round 1, I2).
        reviewed = {(r["video"], r["round"]) for r in truth}
        full = [(i, m) for i, m in enumerate(final)
                if all((rr.video, rr.round) in reviewed for rr, _ in m.rounds)]
        agree = sum(
            i < len(original) and (original[i].score, original[i].winner) == (m.score, m.winner)
            for i, m in full
        )
        agreement = {"n_matches": len(full), "agree": agree, "ratio": _ratio(agree, len(full)),
                     "n_partial": len(final) - len(full)}
    wall = _wall(rows, timings, runs, config.speed_max_video_s)
    u = unflagged_accuracy(rows, config.audit_confidence_level, audit)
    pass_rate = _ratio(sum(v["total_ratio"] <= limit for v in videos), len(videos))
    observed = _observed_speed(rows, timings, config.speed_max_video_s)
    # R66: the pass basis is the worst video; the pass rate is the share of required real
    # videos whose observed wall meets the limit (a missing one counts as not verified).
    seen = observed["videos"]
    basis, gate_ratio = _speed_gate(observed, wall)
    speed = {"basis": basis,
             "ratio": gate_ratio,
             "pass_rate": _ratio(sum(v["ratio"] <= limit for v in seen),
                                 len(seen) + len(observed["missing"])),
             "slowest": observed["slowest"]}
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
        observed=observed,
        unflagged=u,
        tradeoff=_tradeoff(real, u, speed, set(vetoed or ())),
    )


def _tradeoff(real: list[dict], u: dict, speed: dict,
              vetoed: set[str] = frozenset()) -> dict:
    """R63 joint block, side by side (no weighted score).

    auto accuracy = what the judge gets right on its own over ALL real rounds: unflagged
    rounds at the unflagged estimate, flagged rounds by their reviewed original accuracy
    with unreviewed flagged rounds counted as errors. final accuracy = after the human
    review done so far: reviewed rounds are right, unreviewed unflagged rounds at the
    estimate, unreviewed flagged rounds still pending (counted as errors). An unreviewed
    round of a vetoed calibration needs human work (spec 6.6b) and is treated as flagged,
    so automation rate = 1 - review_ratio.
    """
    n = len(real)

    def needs_human(r: dict) -> bool:
        return r["needs_review"] is True or _vetoed_pending(r, vetoed)

    unflagged = [r for r in real if r["needs_review"] is False and not needs_human(r)]
    flagged = [r for r in real if needs_human(r)]
    flagged_reviewed = [r for r in flagged if r.get("reviewed") is True]
    n_u, n_u_rev = len(unflagged), sum(r.get("reviewed") is True for r in unflagged)
    correct_flagged = sum(map(_finish_right, flagged_reviewed))

    def mix(weight: int, fixed: int) -> dict:
        # No human truth at all = unverified, never a "measured" 0.0 from counting
        # every pending flagged round as an error.
        no_truth = not any(r.get("reviewed") is True for r in real)
        if not n or no_truth or (weight and not math.isfinite(u["estimate"])):
            return {"estimate": NAN, "low": NAN, "high": NAN}
        p = (u["estimate"], u["low"], u["high"]) if weight else (0.0, 0.0, 0.0)
        est, lo, hi = ((weight * v + fixed) / n for v in p)
        return {"estimate": est, "low": lo, "high": hi}

    return {
        "automation_rate": _ratio(n_u, n),
        "auto_accuracy": mix(n_u, correct_flagged),
        "final_accuracy": mix(n_u - n_u_rev, n_u_rev + len(flagged_reviewed)),
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


def audit_remedy(results_dir, n, seed) -> str:
    return f"bxjudge audit-sample {results_dir} --n {n} --seed {seed} --replace"


def read_audit(results_dir: str | Path) -> dict | None:
    """The frozen audit sample record (R69), None when no sample was drawn.

    Raises ValueError when rows carry ``audit_sample`` marks but ``audit_sample.json`` is
    missing or malformed: the draw can no longer be verified.
    """
    path = Path(results_dir) / "audit_sample.json"
    rounds = Path(results_dir) / "rounds.json"
    if path.is_file():
        meta = json.loads(path.read_text(encoding="utf-8"))
        try:
            return validate_audit(meta)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from None
    rows = json.loads(rounds.read_text(encoding="utf-8")) if rounds.is_file() else []
    if any(r.get("audit_sample") is True for r in rows):
        raise ValueError(f"{path}: rows are marked audit_sample but audit_sample.json is "
                         "missing; re-draw with `bxjudge audit-sample <dir> --n N --seed S "
                         "--replace`")
    return None


def read_results(results_dir: str | Path) -> tuple[list[dict], dict | None, list | None]:
    """Rows, per-video timings and batch run records of a results directory (read-only;
    absent ``videos.json`` / ``runs.json`` give None)."""
    results_dir = Path(results_dir)
    rounds = results_dir / "rounds.json"
    if not rounds.is_file():
        raise ValueError(f"{results_dir}: no rounds.json (run `bxjudge run` first)")
    rows = json.loads(rounds.read_text(encoding="utf-8"))
    read_audit(results_dir)  # R69: marks without audit_sample.json are an error
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


def _verdict(value: float, verdict: str) -> str:
    if verdict == PASS:
        return PASS
    if verdict == INVALIDATED:
        return "INVALIDATED (resample)"
    if verdict == INCONCLUSIVE:
        return "INCONCLUSIVE (樣本不足)"  # R64: meets the target without statistical support
    return "UNVERIFIED (FAIL)" if not math.isfinite(value) else FAIL


_SMOKE = (
    "> **SMOKE ONLY — NOT ACCEPTANCE.** These rounds were judged with a synthetic-trained "
    "detector (synthetic_detector, R44/R48). The numbers prove the evaluation wiring only "
    "and say nothing about real-video accuracy or spec section 11 compliance."
)
_TIMING_NOTE = (
    "Timing definitions: **core** = processing_s (judge and exports, without batch config/model "
    "init/probe); **VLM** = vlm_s (frame decoding and requests), reported separately as diagnostics. "
    "**Observed per-video wall** uses max(judged_wall_s, attempt_wall_s): one continuous timer "
    "from first decode/selection through final result writing, including waiting and after_run "
    "VLM, without adding disconnected spans. judged_wall_s preserves evidence behind retained "
    "rows, including slower R36 reviewed rows; attempt_wall_s belongs to the latest attempt. "
    "duration_s is the latest probe of the CURRENT source; judged_duration_s describes the "
    "retained analysis. Failed/pending attempts, missing or invalid timing/probe evidence and "
    "missing batch records are unverified and never PASS. **Batch wall-clock** is the complete "
    "runs.json wall_s for each used invocation, including config/model/probe/release overhead "
    "and its VLM phase exactly once, plus standalone VLM invocations. No work is subtracted; "
    "each eligible video's current playback duration is counted once in its last run. "
    "Speed (R66a) requires BOTH the worst per-video ratio and pooled batch ratio to meet the "
    "YAML target; speed_ratio is their maximum. Median/core/sum-based ratios remain diagnostic. "
    "Use bxjudge run <folder> --out <dir> --calib <calib> --rejudge to obtain new timing "
    "evidence while keeping reviewed rows. VLM request latency is per request actually sent "
    "(vlm_requests); cold = first answer by that model in the phase. Playback seconds are "
    "never rescaled for slow-motion capture."
)


def _lat(s: dict, pct: float, unit: str = " s") -> str:
    return f"median {_f(s['median'], 2)}{unit} / p{pct:g} {_f(s['p'], 2)}{unit}"


def _count(value: float, den: int) -> int:
    return round(value * den) if math.isfinite(value) else 0


def render_eval_report(results_dir, rows, timings, metrics: Metrics, ok: dict[str, str],
                       extra: ReportMetrics, config: EvalConfig, *,
                       truth: TruthMatch | None = None, truth_dir=None, today: str,
                       vetoed: set[str] | None = None) -> str:
    """Markdown eval report: R60 summary table first, then calibration (spec 6.6b), timing,
    R37 coverage, errors."""
    m, x, pct = metrics, extra, extra.latency_percentile
    q, fv, ref = extra.vlm_request_s, extra.flagged_vlm, extra.referee
    limit = config.targets["speed_ratio"][1]
    source = f"Results: `{results_dir}`"
    if truth_dir is not None:
        source += f" — truth from reference `{truth_dir}` (R37, read-only)"
    out = [f"# bxjudge 評估報告 eval-{today}", "", source, ""]
    if is_synthetic(rows):
        out += [_SMOKE, ""]
    u, w, t, o = x.unflagged, x.wall, x.tradeoff, x.observed
    slowest = (f"slowest video {o['slowest']} at {_f(o['worst'])}" if o["slowest"]
               else "no observed video")
    missing = (f"; UNVERIFIED: missing current timing for {', '.join(o['missing'])} (run bxjudge run --rejudge)"
               if o["missing"] else "")
    wall_ok = sum(v["ratio"] <= limit for v in o["videos"])
    if u["method"] == "audit_sample":
        how = (f"audit_sample: {u['correct']}/{u['n_used']} reviewed of {u['n_sampled']} "
               f"sampled, {u['n_unflagged']} unflagged; Wilson {u['level']:.0%} "
               f"{_f(u['low'])}–{_f(u['high'])}")
        if u["coverage"] < 1:
            how += (f"; WARNING: audit sample only partly reviewed ({u['n_used']}/"
                    f"{u['n_sampled']} sampled rounds, coverage {_f(u['coverage'])}): "
                    "review the rest with the 抽樣稽核 filter")
        how += (f"; R64: PASS needs a fully reviewed sample of ≥ {config.audit_min_n} "
                "with the Wilson bound meeting the target")
    elif u["method"] == "invalidated":
        remedy = audit_remedy(results_dir, u["audit_n"], u["audit_seed"])
        how = f"{u['reason']}; rerun `{remedy}`"
    elif u["method"] == "exact":
        how = f"exact: all {u['n_unflagged']} unflagged rounds reviewed"
    else:
        how = f"unverified: {u['reason']}"
    biased = u["biased"]
    spec_rows = [
        ("未標覆核回合判定正確率 finish_accuracy_unflagged", "finish_accuracy_unflagged",
         (f"{_f(m.finish_accuracy_unflagged)} [{how}]; reviewed outside the random sample "
          f"(biased, not used) {_frac(biased['accuracy'], biased['correct'], biased['n'])}")),
        ("觸碰召回率 touch_recall", "touch_recall",
         _frac(m.touch_recall, _count(m.touch_recall, m.n_touch_truth), m.n_touch_truth)),
        ("空碰召回率 air_recall", "air_recall",
         _frac(m.air_recall, _count(m.air_recall, m.n_air_truth), m.n_air_truth)),
        ("需覆核比例 review_ratio", "review_ratio",
         _frac(m.review_ratio, _count(m.review_ratio, m.n_rounds), m.n_rounds)),
        ("處理速度比 speed_ratio", "speed_ratio",
         (f"{_f(m.speed_ratio)} [{m.speed_basis}: max observed wall / duration; {slowest}; "
          f"{m.n_wall_videos}/{m.n_wall_videos + m.n_wall_missing} real video(s) with "
          f"current timing{missing}]; pooled batch gate {_f(m.speed_ratio_wall)}; "
          f"failed/pending attempts: {', '.join(o['failed']) or 'none'}; "
          f"batch evidence: {w['reason'] or 'complete'}; diagnostics: "
          f"median per-video wall {_f(m.speed_ratio_median)}, "
          f"sum-based median (core+VLM) {_f(m.speed_ratio_sum)} over {m.n_speed_videos} "
          f"video(s), core-only {_f(m.speed_ratio_core)}")),
    ]
    out += ["| 指標 Metric | 數值 Value | 目標 Target | 判定 Result |", "|---|---|---|---|"]
    for label, name, value in spec_rows:
        out.append(f"| {label} | {value} | {_target(config.targets[name])} | "
                   f"{_verdict(getattr(m, name), ok[name])} |")
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
                 "reviewed, excluded)")
    e2e_ok = sum(v["total_ratio"] <= limit for v in x.videos)
    core_ok = sum(v["core_ratio"] <= limit for v in x.videos)
    wall_note = ""
    if w["sum_exceeds_wall"]:
        wall_note = (f"; NOTE per-video sum {_f(w['sum_s'], 2)} s exceeds wall-clock "
                     f"{_f(w['wall_s'], 2)} s (overlapping work)")

    def interval(d: dict) -> str:
        return f"{_f(d['estimate'])} ({_f(d['low'])}–{_f(d['high'])})"
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
        (f"| 端到端延遲 End-to-End Latency | sum-based per video (core+VLM) "
         f"{_lat(x.total_s, pct)}; ratio {_lat(x.total_ratio, pct, '')}; batch wall-clock "
         f"{_f(w['wall_s'], 2)} s over {_f(w['duration_s'], 2)} s of video = ratio "
         f"{_f(w['ratio'])} ({w['n_runs']} run record(s), VLM phase {_f(w['vlm_wall_s'], 2)} s "
         f"inside, counted once; {w['n_videos']} eligible video(s), excluded from both "
         f"terms: {', '.join(w['excluded']) or 'none'}"
         f"{'; ' + w['reason'] if w['reason'] else ''}){wall_note} | — | — |"),
        (f"| 速度達標率 Speed Pass Rate | [observed per-video wall] "
         f"{_frac(t['speed']['pass_rate'], wall_ok, len(o['videos']) + len(o['missing']))}; "
         f"sum-based end-to-end {_frac(x.speed_pass_rate, e2e_ok, len(x.videos))}; core-only "
         f"{_frac(x.core_pass_rate, core_ok, len(x.videos))} (diagnostic) "
         f"| ≤ {limit:g} × duration | — |"),
        (f"| 裁判正確率 Referee Accuracy | all reviewed {acc['all']}; unflagged "
         f"{acc['unflagged']}; flagged {acc['flagged']}{match} | — | — |"),
        "",
        (f"Truth = {m.n_reviewed} reviewed of {m.n_rounds} real rounds; {m.n_placeholders} "
         "diagnostic placeholder row(s) excluded. Unverified = empty denominator; it never "
         "passes. Referee Accuracy is descriptive over the reviewed rows; the unflagged "
         "spec metric uses only an unbiased basis (R62)."),
        "",
        "**取捨 Trade-off (R63)** — side by side, no weighted score:",
        "",
        "| 項目 Item | 數值 Value (interval) | 依據 Coverage |",
        "|---|---|---|",
        (f"| 自動化率 Automation Rate | {_f(t['automation_rate'])} | 1 − review_ratio over "
         f"{m.n_rounds} real rounds |"),
        (f"| 自動判定估計正確率 Auto Accuracy (all real rounds) | {interval(t['auto_accuracy'])} "
         f"| unflagged at [{t['unflagged_method']}]; flagged (incl. vetoed unreviewed) by "
         f"reviewed original accuracy, unreviewed flagged counted as errors "
         f"({t['flagged_reviewed'][0]}/"
         f"{t['flagged_reviewed'][1]} flagged reviewed) |"),
        (f"| 覆核後估計正確率 Final Accuracy after review | {interval(t['final_accuracy'])} | "
         "reviewed rounds right; unreviewed unflagged at the estimate; unreviewed flagged "
         "still pending (errors) |"),
        (f"| 速度達標 Speed Pass | {_f(t['speed']['pass_rate'])} (worst ratio "
         f"{_f(t['speed']['ratio'])}) | {t['speed']['basis']}; {slowest} |"),
        "",
        *_calibration_section(rows, timings, m, vetoed),
        "## Timing",
        "",
        _TIMING_NOTE,
        "",
    ]
    if timings is None:
        out += ["No videos.json: speed and latency are unverified (R11).", ""]
    if o["videos"] or o["missing"]:
        out += [(f"Observed per-video wall (R66): {len(o['videos'])} measured, "
                 f"missing timing: {', '.join(o['missing']) or 'none'}; slowest first."), ""]
    if o["videos"]:
        out += ["| video | status | current duration s | judged wall s | attempt wall s | observed / duration |",
                "|---|---|---|---|---|---|"]
        out += [f"| {v['video']} | {v['status']} | {v['duration_s']:.2f} | {v['judged_wall_s']:.2f} | "
                f"{v['attempt_wall_s']:.2f} | {v['ratio']:.3f} |"
                for v in sorted(o["videos"], key=lambda v: -v["ratio"])]
        out.append("")
    if x.videos:
        out += [("| video | duration s | core s | VLM s | end-to-end s | core ratio | "
                 "end-to-end ratio |"), "|---|---|---|---|---|---|---|"]
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


def _calibration_section(rows, timings, m: Metrics, vetoed: set[str] | None) -> list[str]:
    """「校準 Calibration」 (spec 6.6b): failures and vetoes stay visible, never hidden."""
    c = calibration_coverage(rows, timings, vetoed)

    def names(videos: list[str]) -> str:
        return ", ".join(videos) or "none"

    unverified = ("unverified (no videos.json)" if timings is None
                  else STATUS_UNVERIFIED if not c["status_known"] else None)
    coverage = unverified or _frac(c["coverage"], c["n_covered"], c["n_videos"])

    def failed(key: str) -> str:
        videos = c[key]
        return unverified or f"{len(videos)}: {names(videos)}"

    out = [
        "## 校準 Calibration",
        "",
        ("Vetoed rounds stay in every accuracy population with their original predictions; "
         "unreviewed ones count as needing human work in review_ratio and are left out of "
         "matches.csv; the random audit sample (R62) is unchanged. Failures are counted per "
         "video from the latest attempt's `analysis_status` (spec 6.6b, R68); a decode or "
         "analysis failure of a previously judged video keeps its earlier rows."),
        "",
        "| 項目 Item | 數值 Value |",
        "|---|---|",
        (f"| 有效處理覆蓋率 Coverage | {coverage} videos judged with an accepted (non-vetoed) "
         "calibration |"),
        f"| 校準失敗影片 Recalibrate videos | {failed('recalibrate_videos')} |",
        f"| 解碼失敗影片 Decode-error videos | {failed('decode_error_videos')} |",
        f"| 分析失敗影片 Analysis-error videos | {failed('analysis_error_videos')} |",
        (f"| 被否決的校準組 Vetoed groups | {m.n_vetoed_groups} group(s); "
         f"{m.n_vetoed_videos} video(s); {m.n_vetoed_rounds} round(s), "
         f"{c['n_vetoed_unreviewed']} unreviewed |"),
        (f"| 待處理影片 Pending videos | {failed('pending_videos')} "
         "(failed + vetoed) |"),
        "",
    ]
    out += [f"- vetoed `{sid}`: videos {names(g['videos'])}; {g['rounds']} round(s), "
            f"{g['unreviewed']} unreviewed" for sid, g in c["groups"].items()]
    return out + [""] if c["groups"] else out


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
