"""Merge-safe authoritative results and local evidence exports."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Set as AbstractSet
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .calibration import ZONES
from .geometry import zone_of
from .rules import Match, Rules, group_matches
from .types import Calibration, LaunchCheck, RoundResult, Side, Track

if TYPE_CHECKING:
    from .calibration_store import Snapshot

ROUND_FIELDS = [
    "video",
    "round",
    "t_launch",
    "t_end",
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
    "points",
    "confidence",
    "needs_review",
    "review_reasons",
    "vlm_note",
    "reviewed",
    "reviewer_fix",
]


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_rounds(out_dir: Path) -> list[dict]:
    """Authoritative flat rows of ``out_dir/rounds.json`` (empty when absent)."""
    path = Path(out_dir) / "rounds.json"
    rows = _read_json(path, [])
    if not isinstance(rows, list):
        raise ValueError(f"{path}: rounds.json must be a list of rows")  # noqa: TRY004 (callers expect ValueError)
    return rows


def effective(row: dict) -> dict:
    """Return corrected flat fields without mutating the authoritative prediction."""
    return {**row, **(row.get("reviewer_fix") or {})}


def _result_row(r: RoundResult) -> dict:
    row = asdict(r)
    launch = row.pop("launch")
    spin = row.pop("spin_dir")
    row["launch_present"] = launch is not None
    row["spin_dir_present"] = spin is not None
    row["landing_frame"] = launch["landing_frame"] if launch else None
    row["launch_notes"] = launch["notes"] if launch else []
    for side in ("X", "B"):
        prefix = side.lower()
        row[f"{prefix}_touch"] = launch["touch"][side] if launch else "unknown"
        row[f"{prefix}_landed"] = launch["landed"][side] if launch else False
        row[f"{prefix}_land_zone"] = launch["landing_zone"][side] if launch else None
        row[f"{prefix}_spin_dir"] = spin[side] if spin else "unknown"
    row["air_collision"] = launch["air_collision"] if launch else "maybe"
    if not row["match_start"]:
        del row["match_start"]  # sparse: absent means automatic grouping (R52)
    if not row["audit_sample"]:
        del row["audit_sample"]  # sparse: absent means not in the audit sample (R62)
    # ``calibration_id`` stays in every JSON row (None = judged without a snapshot); it is
    # metadata, not a ROUND_FIELDS CSV column (ruling P1).
    return row


def row_to_result(row: dict) -> RoundResult:
    """Rebuild a judgment from authoritative or already-effective flat fields."""
    launch = None
    if row["launch_present"]:
        launch = LaunchCheck(
            touch={s: row[f"{s.lower()}_touch"] for s in ("X", "B")},
            landed={s: row[f"{s.lower()}_landed"] for s in ("X", "B")},
            landing_frame=row["landing_frame"],
            landing_zone={s: row[f"{s.lower()}_land_zone"] for s in ("X", "B")},
            air_collision=row["air_collision"],
            notes=row["launch_notes"],
        )
    values = {
        key: row[key]
        for key in (
            "video",
            "round",
            "t_launch",
            "t_end",
            "finish",
            "winner",
            "points",
            "confidence",
            "needs_review",
            "review_reasons",
            "vlm_note",
            "reviewed",
            "reviewer_fix",
            "motion_evidence",
            "detector_provenance",
        )
    }
    match_start = row.get("match_start", False)
    if type(match_start) is not bool:
        raise ValueError(f"match_start must be a bool, got {match_start!r}")
    audit_sample = row.get("audit_sample", False)
    if type(audit_sample) is not bool:
        raise ValueError(f"audit_sample must be a bool, got {audit_sample!r}")
    calibration_id = row.get("calibration_id")  # absent in rows written before snapshots
    if calibration_id is not None and type(calibration_id) is not str:
        raise ValueError(f"calibration_id must be a string or null, got {calibration_id!r}")
    return RoundResult(
        **values,
        match_start=match_start,
        audit_sample=audit_sample,
        calibration_id=calibration_id,
        launch=launch,
        spin_dir={s: row[f"{s.lower()}_spin_dir"] for s in ("X", "B")}
        if row["spin_dir_present"]
        else None,
    )


def _csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def write_results(new: list[RoundResult], out_dir: Path, rules: Rules) -> None:
    """Replace complete affected-video analyses, preserving all reviewed rows."""
    incoming = {}
    for r in new:
        key = (r.video, r.round)
        if key in incoming:
            raise ValueError(f"duplicate result key: {key}")
        incoming[key] = _result_row(r)
    out_dir = Path(out_dir)
    affected = {r.video for r in new}
    rows = {}
    seen = set()
    for row in read_rounds(out_dir):
        key = (row["video"], row["round"])
        if key in seen:
            raise ValueError(f"duplicate persisted result key: {key}")
        seen.add(key)
        if row["video"] not in affected or row["reviewed"]:
            rows[key] = row
    for key, row in incoming.items():
        rows.setdefault(key, row)
    save_rounds(list(rows.values()), out_dir, rules)


def scoring_rows(rows: list[dict], vetoed: AbstractSet[str] = frozenset()) -> list[dict]:
    """``rows`` without the UNREVIEWED rows judged with a vetoed calibration snapshot: their
    zones are untrusted, so they are no valid judgment to score (spec section 6.5). Reviewed
    rows are human truth and always score."""
    return [row for row in rows
            if row.get("reviewed") is True or row.get("calibration_id") not in vetoed]


def matches_for(rows: list[dict], rules: Rules,
                vetoed: AbstractSet[str] = frozenset()) -> list[Match]:
    """Matches from the effective (reviewer-corrected) flat rows; unreviewed rows of a
    ``vetoed`` calibration snapshot are left out (``scoring_rows``)."""
    return group_matches(
        [row_to_result(effective(row)) for row in scoring_rows(rows, vetoed)], rules
    )


def save_rounds(rows: list[dict], out_dir: Path, rules: Rules) -> None:
    """Persist the COMPLETE flat row set and rebuild ``rounds.csv`` and ``matches.csv``.

    The single export path shared by the judge writer, human corrections and calibration
    vetoes: ``matches.csv`` leaves out the unreviewed rows whose calibration snapshot is
    vetoed in ``<out_dir>/calibration_vetoes.json``; ``rounds.json`` / ``rounds.csv`` keep
    every row unchanged.
    """
    # Deferred: calibration_store pulls in autocalib (and imports this module lazily).
    from .calibration_store import vetoed_ids

    out_dir = Path(out_dir)
    keys = [(row["video"], row["round"]) for row in rows]
    if len(set(keys)) != len(keys):
        dup = next(k for k in keys if keys.count(k) > 1)
        raise ValueError(f"duplicate result key: {dup}")
    merged = sorted(rows, key=lambda r: (r["video"], r["round"]))
    match_rows = [
        {
            "match": i,
            "x_score": m.score["X"],
            "b_score": m.score["B"],
            "winner": m.winner,
            "rounds": [
                {
                    "video": r.video,
                    "round": r.round,
                    "action": outcome.action,
                    "scorer": outcome.scorer,
                    "points": outcome.points,
                }
                for r, outcome in m.rounds
            ],
        }
        for i, m in enumerate(matches_for(merged, rules, vetoed_ids(out_dir)), 1)
    ]
    _write_json(out_dir / "rounds.json", merged)
    _csv(out_dir / "rounds.csv", ROUND_FIELDS, merged)
    _csv(
        out_dir / "matches.csv",
        ["match", "x_score", "b_score", "winner", "rounds"],
        match_rows,
    )


def record_video_timing(
    out_dir: Path,
    video: str,
    duration_s: float,
    processing_s: float,
    *,
    source_video: Path | None = None,
) -> None:
    path = Path(out_dir) / "videos.json"
    rows = _read_json(path, {})
    rows[video] = {"duration_s": duration_s, "processing_s": processing_s}
    if source_video is not None:
        rows[video]["source_video"] = str(source_video)
    _write_json(path, rows)


# Ruling R68: the outcome of a video's latest analysis attempt, recorded explicitly so the
# evaluation never infers it (e.g. from ``calibration: null``).
ANALYSIS_STATUSES = ("ok", "recalibrate", "decode_error", "analysis_error")
# Transient failures: a previously judged video keeps its rows on these (R68).
TRANSIENT_FAILURES = frozenset({"decode_error", "analysis_error"})


def record_video_status(out_dir: Path, video: str, status: str) -> None:
    """Record ``videos.json[video]["analysis_status"]`` (one of ``ANALYSIS_STATUSES``) without
    touching anything else in the entry, so a failed rerun that keeps a video's rows can say
    so while the timing and calibration of those rows stay as they were (ruling R68). Call
    after ``record_video_timing``, which replaces the entry."""
    if status not in ANALYSIS_STATUSES:
        raise ValueError(f"unknown analysis_status {status!r}; expected one of {ANALYSIS_STATUSES}")
    path = Path(out_dir) / "videos.json"
    rows = _read_json(path, {})
    rows.setdefault(video, {})["analysis_status"] = status
    _write_json(path, rows)


def record_video_calibration(out_dir: Path, video: str, snap: Snapshot | None) -> None:
    """Record in ``videos.json[video]["calibration"]`` the snapshot the video was judged with
    (``{"id", "kind", "name", "score", "template_id"}``; ``score`` is this video's similarity
    for a manual snapshot, the alignment score for an auto one) or ``None`` (spec section 6.4).
    Call after ``record_video_timing``, which replaces the entry."""
    path = Path(out_dir) / "videos.json"
    rows = _read_json(path, {})
    rows.setdefault(video, {})["calibration"] = None if snap is None else {
        "id": snap.id,
        "kind": snap.kind,
        "name": snap.calibration.name,
        "score": float(snap.score),
        "template_id": snap.template_id,
    }
    _write_json(path, rows)


VLM_REQUEST_KEYS = ("round", "seconds", "model", "ok", "cold")


def _check_vlm_request(q) -> None:
    ok = (
        isinstance(q, dict)
        and tuple(sorted(q)) == tuple(sorted(VLM_REQUEST_KEYS))
        and type(q["round"]) is int
        and isinstance(q["seconds"], (int, float))
        and not isinstance(q["seconds"], bool)
        and math.isfinite(q["seconds"])
        and q["seconds"] >= 0
        and (q["model"] is None or isinstance(q["model"], str))
        and type(q["ok"]) is bool
        and (q["cold"] is None or type(q["cold"]) is bool)
    )
    if not ok:
        raise ValueError(f"invalid VLM request record: {q!r}")


def record_vlm_timing(out_dir: Path, video: str, vlm_s: float, requests: list[dict]) -> None:
    """Add VLM phase time to ``vlm_s``, kept apart from core ``processing_s``.

    ``requests`` holds one record per request actually sent in this phase
    (``VLM_REQUEST_KEYS``: round, seconds, answering model or None, ok, cold = first
    answer by that model in the phase or None when it failed); they accumulate in
    ``vlm_requests``. A later core analysis (``record_video_timing``) replaces the entry
    and its rounds, so accumulated VLM time always belongs to the current round set.
    """
    for q in requests:
        _check_vlm_request(q)
    path = Path(out_dir) / "videos.json"
    rows = _read_json(path, {})
    entry = rows.setdefault(video, {})
    entry["vlm_s"] = entry.get("vlm_s", 0.0) + vlm_s
    entry["vlm_requests"] = [*entry.get("vlm_requests", []), *(dict(q) for q in requests)]
    _write_json(path, rows)


RUN_KINDS = ("run", "vlm")


def record_run_timing(
    out_dir: Path,
    *,
    kind: str,
    started_at: str,
    wall_s: float,
    vlm_wall_s: float | None,
    videos: dict[str, float],
) -> None:
    """Append one batch wall-clock record to ``runs.json`` (R61).

    ``kind="run"``: one ``bxjudge run`` from start to end, including config/model init,
    scan/probe and its VLM phase (``vlm_wall_s`` is a breakdown INSIDE ``wall_s``, never
    extra time); ``videos`` maps each video judged in that run to its playback duration.
    ``kind="vlm"``: a standalone ``bxjudge vlm`` phase (``videos`` empty: it adds waiting
    time but no new video).
    """

    def seconds(v) -> bool:
        return (isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and v >= 0)

    ok = (
        kind in RUN_KINDS
        and isinstance(started_at, str)
        and seconds(wall_s)
        and (vlm_wall_s is None or (seconds(vlm_wall_s) and vlm_wall_s <= wall_s))
        and isinstance(videos, dict)
        and all(isinstance(k, str) and seconds(v) for k, v in videos.items())
    )
    if not ok:
        raise ValueError(f"invalid run timing record: {kind!r} {wall_s!r} {vlm_wall_s!r} {videos!r}")
    path = Path(out_dir) / "runs.json"
    runs = _read_json(path, [])
    runs.append({"kind": kind, "started_at": started_at, "wall_s": wall_s,
                 "vlm_wall_s": vlm_wall_s, "videos": dict(videos)})
    _write_json(path, runs)


def trajectory_png(out_dir: Path, video: str, round: int) -> Path:
    """Where ``write_trajectory`` puts the round's trajectory image."""
    stem = f"{Path(video).with_suffix('').as_posix().replace('/', '__')}_r{round}"
    return Path(out_dir) / "trajectories" / f"{stem}.png"


def write_trajectory(
    out_dir: Path,
    r: RoundResult,
    bey: dict[Side, Track],
    parts: list[Track],
    c: Calibration,
    fps: float,
    events: dict[str, int | None],
    *,
    unassigned: list[Track] | None = None,
) -> tuple[Path, Path]:
    png_p = trajectory_png(out_dir, r.video, r.round)
    csv_p = png_p.with_suffix(".csv")
    png_p.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(c.ref_frame)
    if image is None:
        raise FileNotFoundError(c.ref_frame)
    image = cv2.resize(image, c.frame_size, interpolation=cv2.INTER_AREA)

    def label(text, point, size, color):
        (width, height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, size, 1
        )
        x = max(0, min(int(point[0]), image.shape[1] - width))
        y = max(height, min(int(point[1]), image.shape[0] - baseline))
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)

    for field, _, _ in ZONES:
        pts = np.rint(getattr(c, field)).astype(np.int32)
        cv2.polylines(image, [pts], True, (80, 200, 80), 1)
        anchor = tuple(pts[0])
        label(field, anchor, 0.35, (80, 200, 80))
    rows = []
    for side, track in [
        *bey.items(),
        *(("", t) for t in [*parts, *(unassigned or [])]),
    ]:
        centers = track.centers()
        observations = {d.frame: d for d in track.dets}
        for f in range(track.first_frame(), track.last_frame() + 1):
            d = observations.get(f)
            x, y = centers[f] if d else (None, None)
            rows.append(
                {
                    "frame": f,
                    "t": f / fps,
                    "side": side,
                    "track_id": track.track_id,
                    "cls": track.cls,
                    "x": x,
                    "y": y,
                    "zone": zone_of((x, y), c) if d else "unknown",
                    "score": d.score if d else None,
                }
            )
        if side or track.cls == "bey":
            color = {"X": (255, 0, 0), "B": (0, 0, 255)}.get(side, (150, 150, 150))
            # Missing observations remain gaps, never invented connected evidence.
            for f, p in sorted(centers.items()):
                if f - 1 in centers:
                    cv2.line(
                        image,
                        tuple(np.rint(centers[f - 1]).astype(int)),
                        tuple(np.rint(p).astype(int)),
                        color,
                        2,
                    )
            for event, frame in events.items() if side else []:
                if event.startswith("landing_") and event != f"landing_{side}":
                    continue
                if frame in centers:
                    point = tuple(np.rint(centers[frame]).astype(int))
                    cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 12, 2)
                    label(f"{event}@{frame}", point, 0.4, color)
        if (
            track.cls == "bey_part"
            and r.finish == "burst"
            and events.get("finish") in centers
        ):
            frame = events["finish"]
            point = tuple(np.rint(centers[frame]).astype(int))
            cv2.drawMarker(image, point, (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
            label(f"finish@{frame} (part)", point, 0.4, (0, 255, 255))
    _csv(
        csv_p,
        ["frame", "t", "side", "track_id", "cls", "x", "y", "zone", "score"],
        rows,
    )
    if not cv2.imwrite(str(png_p), image):
        raise OSError(f"cannot write {png_p}")
    return csv_p, png_p
