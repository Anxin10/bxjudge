"""Merge-safe authoritative results and local evidence exports."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .calibration import ZONES
from .geometry import zone_of
from .rules import Rules, group_matches
from .types import Calibration, LaunchCheck, RoundResult, Side, Track

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
    return RoundResult(
        **values,
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
    for row in _read_json(out_dir / "rounds.json", []):
        key = (row["video"], row["round"])
        if key in seen:
            raise ValueError(f"duplicate persisted result key: {key}")
        seen.add(key)
        if row["video"] not in affected or row["reviewed"]:
            rows[key] = row
    for key, row in incoming.items():
        rows.setdefault(key, row)
    merged = sorted(rows.values(), key=lambda r: (r["video"], r["round"]))
    _write_json(out_dir / "rounds.json", merged)
    _csv(out_dir / "rounds.csv", ROUND_FIELDS, merged)
    matches = group_matches([row_to_result(effective(row)) for row in merged], rules)
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
        for i, m in enumerate(matches, 1)
    ]
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


def record_vlm_timing(out_dir: Path, video: str, vlm_s: float) -> None:
    """Add VLM phase time to ``vlm_s``, kept apart from core ``processing_s``.

    A later core analysis (``record_video_timing``) replaces the entry and its rounds,
    so accumulated VLM time always belongs to the current round set.
    """
    path = Path(out_dir) / "videos.json"
    rows = _read_json(path, {})
    entry = rows.setdefault(video, {})
    entry["vlm_s"] = entry.get("vlm_s", 0.0) + vlm_s
    _write_json(path, rows)


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
    folder = Path(out_dir) / "trajectories"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(r.video).with_suffix('').as_posix().replace('/', '__')}_r{r.round}"
    csv_p, png_p = folder / f"{stem}.csv", folder / f"{stem}.png"
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
