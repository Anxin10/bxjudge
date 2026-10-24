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
    # Ruling R66 (replaces R61's pass rule): the WORST observed per-video wall / playback
    # duration over every measured real video; nan (unverified) when any of them lacks an
    # observed wall AND the pooled batch ratio (R66a); the median ratio is a diagnostic.
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
    speed_ratio_wall: float = NAN  # batch wall_s / measured duration (R61; gated by R66a)
    speed_ratio_median: float = NAN  # diagnostic: median observed per-video wall ratio (R66)
    speed_basis: str = "unverified"  # worst-video+batch | unverified (R66a)
    speed_slowest_video: str | None = None  # highest observed ratio (even when unverified)
    speed_ratio_worst_observed: float = NAN  # its ratio
    n_wall_videos: int = 0  # required real videos with observed wall times
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
    # R67 (same values as ReportMetrics.tradeoff): reviewed-and-right / ALL real rounds, and
    # the exact value only with full truth coverage (None = unverified).
    accuracy_lower_bound: float = NAN
    overall_accuracy: float | None = None


_FAILURE_STATUSES = ("recalibrate", "decode_error", "analysis_error")
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

    status_known = timings is not None and all(status(e) is not None for e in timings.values())
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


def _judgment_sources(rows: list[dict], entry: dict, video: str) -> list[dict] | None:
    """Require explicit producing measurements covering every current raw prediction once."""
    sources = entry.get("judgment_sources")
    if not isinstance(sources, list) or not sources:
        return None
    expected = {r["round"] for r in rows if r["video"] == video}
    covered = set()
    for source in sources:
        if not isinstance(source, dict):
            return None
        rounds = source.get("rounds")
        rid, wall, core = source.get("run_id"), source.get("wall_s"), source.get("processing_s")
        if not (isinstance(rounds, list) and rounds and all(type(n) is int for n in rounds)
                and len(set(rounds)) == len(rounds) and not covered.intersection(rounds)
                and isinstance(rid, str) and rid and _number(wall) and wall > 0
                and _number(core) and core > 0):
            return None
        covered.update(rounds)
    return sources if covered == expected else None


def _observed_speed(rows: list[dict], timings: dict | None, max_s: float) -> dict:
    """Rulings R66/R66a: the observed per-video wall of every measured real video.

    Required: each video with at least one real round whose playback duration is not over
    ``max_s`` (an unknown duration stays required). Its observed time is
    ``max(max(source.wall_s) + vlm_s, attempt_wall_s)``: every producing run behind its
    current rows is retained in ``judgment_sources``, plus their accumulated VLM time (R59),
    or its latest attempt (a failed one keeps the rows but not their wall, R68), whichever is slower, over ``duration_s``, the latest
    probe of the current source file. A required video without positive wall times, a valid
    duration, complete producing-source coverage or an ``analysis_status`` is ``missing``
    -> unverified (a legacy ``wall_s`` field is not read: no compatibility path)."""
    videos, missing = [], []
    for video in sorted({r["video"] for r in real_rows(rows)}):
        entry = (timings or {}).get(video)
        entry = entry if isinstance(entry, dict) else {}
        duration = entry.get("duration_s")
        sources = _judgment_sources(rows, entry, video)
        judged = max(s["wall_s"] for s in sources) if sources else None
        attempt = entry.get("attempt_wall_s")
        status, vlm = entry.get("analysis_status"), entry.get("vlm_s", 0.0)
        if _number(duration) and duration > max_s:
            continue
        if not (_number(duration) and duration > 0 and _number(judged) and judged > 0
                and _number(attempt) and attempt > 0
                and status in ANALYSIS_STATUSES and _number(vlm) and vlm >= 0):
            missing.append(video)
            continue
        observed = max(judged + vlm, attempt)
        videos.append({"video": video, "duration_s": duration, "judged_wall_s": judged,
                       "attempt_wall_s": attempt, "vlm_s": vlm, "observed_s": observed,
                       "ratio": observed / duration, "status": status,
                       # C-X4(b): the runs that measured this timing
                       "judgment_sources": sources,
                       "attempt_run_id": entry.get("attempt_run_id"),
                       "has_vlm": "vlm_s" in entry,
                       "vlm_run_ids": entry.get("vlm_run_ids")})
    slowest = max(videos, key=lambda v: v["ratio"]) if videos else None
    return {
        "videos": videos,
        "missing": missing,
        "complete": bool(videos) and not missing,
        "slowest": slowest["video"] if slowest else None,
        "worst": slowest["ratio"] if slowest else NAN,
        "median": statistics.median(v["ratio"] for v in videos) if videos else NAN,
    }


def _linked_to_complete_runs(video: dict, runs: list[dict] | None) -> bool:
    """Ruling C-X4(b): the CURRENT timing of ``video`` was measured by runs that completed.
    Every ``judgment_sources`` run and ``attempt_run_id`` must name a ``run`` record listing the
    video, and every id in ``vlm_run_ids`` (when it has VLM time) must name a record. A
    missing id (an interrupted run, or a pre-C-X4 entry) is never linked."""
    listing: dict[str, set] = {}
    for r in runs or []:
        rid = r.get("run_id")
        if isinstance(rid, str) and rid:
            listing.setdefault(rid, set()).update(r["videos"] if r["kind"] == "run" else ())

    def lists(rid) -> bool:
        return isinstance(rid, str) and video["video"] in listing.get(rid, ())

    ok = (all(lists(s["run_id"]) for s in video["judgment_sources"])
          and lists(video["attempt_run_id"]))
    if video["has_vlm"]:
        ids = video["vlm_run_ids"]
        ok = ok and isinstance(ids, list) and bool(ids) and all(
            isinstance(i, str) and i in listing for i in ids)
    return ok


def speed_gate(observed: dict, batch: dict, runs: list[dict] | None = None) -> tuple[str, float]:
    """Ruling R66a: speed is verified only when every required video has its wall times AND
    a pooled batch ratio exists; the gated ratio is then the larger of the worst per-video
    ratio and the batch ratio (both must be within the target). Ruling C-X3(c): the batch
    ratio counts only when the batch records (``_wall``) cover EVERY measured real video; a
    video judged by a run that never wrote its record (interrupted) leaves speed unverified.
    Ruling C-X4(b): coverage is by run id, not file name: each video's current timing must
    have been measured by runs whose batch records are complete (``_linked_to_complete_runs``),
    so a stale record of an earlier run never verifies a newer, unrecorded timing.
    Ruling C-X5(b): the gated ratio also takes the batch ratio of EACH run that measured a
    video's current timing (``judgment_sources`` and ``attempt_run_id``), so a fast failed retry
    never hides the batch overhead of the run that produced the kept rows; any of those
    ratios missing or not finite leaves speed unverified."""
    covered = set(batch.get("videos", ()))
    run_ratios = batch.get("run_ratios", {})
    linked = [run_ratios.get(rid) if isinstance(rid, str) else None
              for v in observed["videos"]
              for rid in [*(s["run_id"] for s in v["judgment_sources"]), v["attempt_run_id"]]]
    if (observed["complete"] and math.isfinite(batch["ratio"])
            and all(v["video"] in covered and _linked_to_complete_runs(v, runs)
                    for v in observed["videos"])
            and all(r is not None and math.isfinite(r) for r in linked)):
        return "worst-video+batch", max(observed["worst"], batch["ratio"], *linked)
    return "unverified", NAN


def _wall(rows: list[dict], timings: dict | None, runs: list[dict] | None,
          max_s: float) -> dict:
    """Batch wall-clock basis from ``runs.json`` records (R61).

    Ruling C-X8: every number here comes ONLY from what each run record stored when it was
    written: its ``wall_s``, the ``videos`` durations of the sources THAT run probed, the
    ``work`` it measured per video (C-X7(a)) and ``real`` (whether the video had real rows
    after that run's attempt). The current ``videos.json`` never changes a run's ratio; it only
    decides which runs matter (the videos required now, each in the last run listing it).

    A listed video is in a run's scope when that run recorded it with real rows and a stored
    duration in ``(0, max_s]``; a record without its ``real`` map is unusable (C-X9): its own
    ratio, and the pooled ratio when it is a used run, are NaN, so speed is unverified. A
    run's own ratio (C-X5(b)) is ``(wall_s - stored work of its out-of-scope videos) / stored
    duration of its in-scope videos``. The pooled ratio counts each currently eligible video
    ONCE, in the last ``run`` record that lists it (a gated video retried later is not
    counted twice), with that record's stored duration when it is in that record's scope.
    Its ``wall_s`` already contains its own VLM phase, so ``vlm_wall_s`` is only a breakdown
    and never added again; standalone ``vlm`` records add waiting time but no video
    duration. The stored work of every listed video left out of the denominator of a used run
    whose last listing it is, is subtracted; any other overhead (init, probe, earlier gate
    attempts) stays in, which is conservative. ``sum_s`` is the stored per-video work of the
    counted videos, reported next to the wall-clock (it can exceed it, e.g. parallel videos).
    """
    eligible = {v["video"] for v in _speed_videos(rows, timings, max_s)[0]}
    run_records = [r for r in runs or [] if r["kind"] == "run"]

    def stored_work(record: dict, video: str) -> float:
        w = record.get("work")
        v = w.get(video) if isinstance(w, dict) else None
        return v if _number(v) and v >= 0 else 0.0  # absent: subtract nothing (conservative)

    def stored_duration(record: dict, video: str) -> float:
        v = record["videos"].get(video)
        return v if _number(v) else 0.0

    def usable(record: dict) -> bool:
        # C-X9: a record without its ``real`` map cannot say which listed videos it judged
        # with real rows, so it is never used for speed (no all-real fallback).
        return isinstance(record.get("real"), dict)

    def in_scope(record: dict, video: str) -> bool:
        return (record["real"].get(video) is True
                and 0 < stored_duration(record, video) <= max_s)

    run_ratios: dict[str, float] = {}
    for r in run_records:
        rid = r.get("run_id")
        if not (isinstance(rid, str) and rid):
            continue
        if not usable(r):
            run_ratios[rid] = NAN
            continue
        listed = list(r["videos"])
        dur = sum(stored_duration(r, v) for v in listed if in_scope(r, v))
        own = max(0.0, r["wall_s"] - sum(stored_work(r, v) for v in listed
                                         if not in_scope(r, v)))
        run_ratios[rid] = own / dur if dur > 0 else NAN

    last: dict[str, int] = {}
    for i, r in enumerate(run_records):
        for video in r["videos"]:
            last[video] = i
    counted = sorted(v for v in eligible if v in last)
    unusable = any(not usable(run_records[last[v]]) for v in counted)
    in_denominator = [v for v in counted if usable(run_records[last[v]])
                      and in_scope(run_records[last[v]], v)]
    used_ids = {last[v] for v in counted}
    excluded = sorted(v for v, i in last.items()
                      if i in used_ids and v not in in_denominator)
    if not runs:
        reason = "no run records"
    elif unusable:
        reason = "a run record without its real map (written before C-X8)"
    elif not in_denominator:
        reason = "no eligible video in run records"
    else:
        reason = None
    used = [run_records[i] for i in sorted(used_ids)]
    if used:
        used += [r for r in runs if r["kind"] == "vlm"]
    raw_wall = sum(r["wall_s"] for r in used)
    wall = max(0.0, raw_wall - sum(stored_work(run_records[last[v]], v) for v in excluded))
    vlm_wall = sum(r["vlm_wall_s"] or 0.0 for r in used)
    duration = sum(stored_duration(run_records[last[v]], v) for v in in_denominator)
    sum_s = (sum(stored_work(run_records[last[v]], v) for v in in_denominator)
             if in_denominator else NAN)
    return {
        "n_runs": len(used),
        "n_videos": len(in_denominator),
        "videos": counted,
        "run_ratios": run_ratios,
        "excluded": excluded,
        "reason": reason,
        "wall_s": wall if used else NAN,
        "raw_wall_s": raw_wall if used else NAN,
        "vlm_wall_s": vlm_wall if used else NAN,
        "duration_s": duration if used else NAN,
        "ratio": wall / duration if used and duration > 0 and not unusable else NAN,
        "sum_s": sum_s,
        "sum_ratio": sum_s / duration if in_denominator and duration > 0 else NAN,
        "sum_exceeds_wall": bool(used and in_denominator and sum_s > wall),
    }


def _wilson(correct: int, n: int, level: float) -> tuple[float, float]:
    z = statistics.NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _is_key(k) -> bool:
    return (isinstance(k, list) and len(k) == 2 and isinstance(k[0], str)
            and type(k[1]) is int)


def check_audit(audit) -> dict:
    """The frozen audit sample record (R69) after a schema check: ``seed`` and ``n`` ints,
    ``population`` and ``selection`` lists of ``[video, round]``, with ``n`` distinct selected
    rounds, all in the population: ``len(selection) == n <= len(population)`` and selection
    within population (C-X3(d)). Anything else (including the legacy ``keys`` record: no
    compatibility path) raises ``ValueError``."""
    ok = (isinstance(audit, dict)
          and type(audit.get("seed")) is int and type(audit.get("n")) is int
          and isinstance(audit.get("population"), list)
          and isinstance(audit.get("selection"), list)
          and all(map(_is_key, audit["population"]))
          and all(map(_is_key, audit["selection"])))
    if ok:
        selection = {tuple(k) for k in audit["selection"]}
        population = {tuple(k) for k in audit["population"]}
        ok = (len(selection) == len(audit["selection"]) == audit["n"]
              <= len(population) and selection <= population)
    if not ok:
        raise ValueError(f"malformed audit sample record (audit_sample.json): {audit!r}; "
                         "re-draw with `bxjudge audit-sample <dir> --n N --seed S --replace`")
    return audit


def _require_audit(rows: list[dict], audit: dict | None) -> None:
    """C-X2(d): rows with audit marks are never evaluated without their frozen record."""
    if audit is None and any(r.get("audit_sample") is True for r in rows):
        raise ValueError("rows are marked audit_sample but no audit sample record was given; "
                         "pass read_audit(<results dir>) so the R69 freeze checks run")
    if audit is not None:
        check_audit(audit)


def unflagged_population(rows: list[dict]) -> list[tuple[str, int]]:
    """The audit population (R62): real rounds whose ORIGINAL ``needs_review`` is False."""
    return sorted((r["video"], r["round"]) for r in real_rows(rows)
                  if r["needs_review"] is False)


def audit_invalidation(rows: list[dict], audit: dict | None) -> str | None:
    """R69: why the frozen audit sample no longer stands, else None.

    The current unflagged population must equal the frozen ``population`` (new or removed
    unflagged real rounds mean the old draw no longer represents what is evaluated), every
    selected (video, round) must still be a row of ``rounds.json`` that is marked
    ``audit_sample``, and no other row may carry the mark; a successful re-analysis replaces
    unreviewed rows (dropping their marks), which is what this detects.
    """
    _require_audit(rows, audit)
    if audit is None:
        return None
    selection = {tuple(k) for k in audit["selection"]}
    frozen = {tuple(k) for k in audit["population"]}
    current = set(unflagged_population(rows))
    added, removed = current - frozen, frozen - current
    marked = {(r["video"], r["round"]) for r in rows if r.get("audit_sample") is True}
    missing = selection - {(r["video"], r["round"]) for r in rows}
    unmarked = (selection - missing) - marked
    if not (missing or unmarked or marked - selection or added or removed):
        return None
    return (f"{len(missing)} selected round(s) missing and {len(unmarked)} no longer marked "
            f"(selection of {len(selection)}, {len(marked - selection)} stray mark(s)); "
            f"population changed: {len(added)} unflagged round(s) added, {len(removed)} "
            "removed")


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

    Speed (R66a) is the larger of the worst observed per-video wall ratio and the pooled batch
    ratio, unverified without ``timings``, without a batch record, when the batch records do
    not cover every measured real video (C-X3(c)), or when a measured real video lacks its
    wall times; the median ratio is a diagnostic. ``audit`` (the frozen
    ``read_audit`` record) is required when rows carry audit marks. ``config`` defaults to
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
        batch = _wall(rows, timings, runs, cfg.speed_max_video_s)
        observed = _observed_speed(rows, timings, cfg.speed_max_video_s)
    else:
        # no videos.json: no eligible videos, so no wall-clock denominator
        batch = {"ratio": NAN, "videos": []}
        observed = _observed_speed(rows, None, math.inf)  # every real video is missing
    wall = batch["ratio"]
    basis, gated = speed_gate(observed, batch, runs)
    u = unflagged_accuracy(rows, None if config is None else config.audit_confidence_level,
                           audit)
    cal = calibration_coverage(rows, timings, vetoed)
    lower_bound, overall = _accuracy_bounds(real)
    return Metrics(
        finish_accuracy_unflagged=u["estimate"],
        touch_recall=_ratio(sum(touch), len(touch)),
        air_recall=_ratio(sum(air), len(air)),
        review_ratio=_ratio(
            sum(r["needs_review"] is True or _vetoed_pending(r, vetoed) for r in real), len(real)
        ),
        speed_ratio=gated,
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
        speed_ratio_wall=wall,
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
        accuracy_lower_bound=lower_bound,
        overall_accuracy=overall,
    )


def _accuracy_bounds(real: list[dict]) -> tuple[float, float | None]:
    """R67: (lower bound, overall accuracy) over the real rounds; one definition shared by
    ``Metrics`` and the report's trade-off block."""
    n = len(real)
    reviewed = [r for r in real if r.get("reviewed") is True]
    correct = sum(map(_finish_right, reviewed))
    return (correct / n if n else NAN,
            correct / n if n and len(reviewed) == n else None)


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
    # R66a: the pass basis is the worst video and the pooled batch ratio; the pass rate is
    # the share of required real videos whose observed wall meets the limit (a missing one
    # counts as not verified).
    seen = observed["videos"]
    basis, gated = speed_gate(observed, wall, runs)
    speed = {"basis": basis, "ratio": gated,
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
    """R63/R67 joint block, side by side (no weighted score, no extrapolation).

    ``accuracy_lower_bound`` = reviewed real rounds whose ORIGINAL finish and winner are
    right / ALL real rounds: every unreviewed round counts as not (yet) right, so it never
    overstates. ``overall_accuracy`` is the exact value, defined only when every real round
    is reviewed (None = unverified, with ``truth_coverage`` = (reviewed, real)). An
    unreviewed round of a vetoed calibration stays in the population and needs human work
    (spec 6.6b), so automation rate = 1 - review_ratio.
    """
    n = len(real)
    reviewed = [r for r in real if r.get("reviewed") is True]
    lower_bound, overall = _accuracy_bounds(real)
    flagged = [r for r in real if r["needs_review"] is True or _vetoed_pending(r, vetoed)]
    flagged_reviewed = [r for r in flagged if r.get("reviewed") is True]
    return {
        "automation_rate": _ratio(n - len(flagged), n),
        "accuracy_lower_bound": lower_bound,
        "overall_accuracy": overall,
        "truth_coverage": (len(reviewed), n),
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


REJUDGE_REMEDY = "bxjudge run --rejudge"  # R66b: recovers missing wall times


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
            return check_audit(meta)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
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
    "Timing definitions: **core** = `processing_s`, the slowest surviving producing "
    "measurement per video (judge, trajectories and result "
    "export; excludes the one-time detector/model init and the scan/probe of the batch; "
    "includes first-inference warm-up). **VLM** = `vlm_s`, the separate VLM phase (frame "
    "decoding + requests; absent = no VLM phase for that video). **End-to-end** = core + VLM "
    "per video (sum-based). **Batch wall-clock** = runs.json `wall_s` of each `bxjudge run` "
    "from start to end (config + model init, probe, every video, its VLM phase once) plus "
    "standalone `bxjudge vlm` phases, minus the measured work of ineligible videos, over "
    "the playback duration of the speed-eligible videos (real round, measured, "
    "≤ speed_max_video_s; each once, in its last run) (R61). **Observed per-video wall** "
    "= max(max(`judgment_sources[].wall_s`) + `vlm_s`, `attempt_wall_s`) from videos.json: "
    "producing runs cover every retained/new raw round; one timer from the "
    "video's first decode (incl. its calibration selection pass) to its result write, for "
    "each run that produced its current rows and for its latest attempt (a failed attempt "
    "keeps the rows, R68, but never hides them), over `duration_s`, the latest probe of the "
    "current source file. Speed pass/fail (R66a) needs BOTH the WORST observed per-video "
    "ratio over every real video ≤ speed_max_video_s AND the pooled batch wall-clock ratio "
    "to be ≤ the target; a real video without wall times, or a batch record that does not "
    "cover every measured real video, or whose run id differs from the run id stamped on a "
    "video's current timing (e.g. an interrupted run or --rejudge), makes speed "
    "unverified. Re-judge with `bxjudge run --rejudge`; retained reviewed predictions "
    "without source evidence remain unverified: use a fresh --out and evaluate it with "
    "--truth pointing to the old results. The median per-video ratio, the "
    "sum-based median and core-only are diagnostics (R59/R61). VLM request latency "
    "is per request actually sent (`vlm_requests`); *cold* = first answer by that model in "
    "the phase (models are unloaded at phase end). Durations are the playback seconds "
    "recorded in videos.json; slow-motion capture is NOT rescaled."
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
    missing = (f"; UNVERIFIED: no wall times for {', '.join(o['missing'])} (re-judge with "
               f"`{REJUDGE_REMEDY}`)"
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
         (f"{_f(m.speed_ratio)} [{m.speed_basis}: max(worst observed wall / duration, "
          f"batch wall-clock {_f(m.speed_ratio_wall)}); {slowest}; "
          f"{m.n_wall_videos}/{m.n_wall_videos + m.n_wall_missing} real video(s) with "
          f"wall times{missing}]; diagnostics: "
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

    cov = t["truth_coverage"]
    overall = (_f(t["overall_accuracy"]) if t["overall_accuracy"] is not None
               else f"unverified (truth coverage {cov[0]}/{cov[1]})")
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
        (f"| 正確率下界 Accuracy Lower Bound | {_f(t['accuracy_lower_bound'])} | reviewed "
         f"real rounds whose original finish and winner are right "
         f"({t['truth_coverage'][0]} reviewed of {t['truth_coverage'][1]} real rounds); "
         "unreviewed rounds (incl. vetoed) count as not right, no extrapolation (R67) |"),
        (f"| 整體正確率 Overall Accuracy | {overall} | exact, only when every real round is "
         "reviewed |"),
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
                 f"missing wall times: {', '.join(o['missing']) or 'none'}; slowest first."),
                ""]
    if o["videos"]:
        out += [("| video | status | duration s | judged wall s | VLM s | attempt wall s | "
                 "producing runs (rounds; wall s) | attempt run | observed / duration |"),
                "|---|---|---|---|---|---|---|---|---|"]
        out += [f"| {v['video']} | {v['status']} | {v['duration_s']:.2f} | "
                f"{v['judged_wall_s']:.2f} | {v['vlm_s']:.2f} | {v['attempt_wall_s']:.2f} | "
                + "; ".join(f"{s['run_id']} ({','.join(map(str, s['rounds']))}; {s['wall_s']:.2f})"
                            for s in v["judgment_sources"])
                + f" | {v['attempt_run_id']} | {v['ratio']:.3f} |"
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
