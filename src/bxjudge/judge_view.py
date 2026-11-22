"""Read-only per-round view data for the local workbench `/judge/` page.

Only reads an existing run directory (rounds, trajectories, geometry,
calibrations); never re-runs inference and never writes into the run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from .autocalib import load_autocalib_config
from .calibration import scale_calibration
from .calibration_store import load_snapshot
from .dynamic_calibration import read_geometry
from .output import effective, read_rounds
from .two_bey_tracking import identity_unknown_at

STAGES = (
    ("detect", "偵測"),
    ("track", "兩顆追蹤"),
    ("geometry", "場地幾何"),
    ("identity", "身分"),
    ("finish", "終結判定"),
    ("other", "其他"),
)
ZONES = ("stadium", "battle_area", "xtreme", "over_left", "over_right")
FINISH_NAMES = {"spin": "耐久", "burst": "爆裂", "over": "超出", "xtreme": "極限"}
_EXACT = {
    "synthetic_detector": "detect",
    "launch_release_unobserved": "detect",
    "no_round": "detect",
    "id_swap_risk": "identity",
    "position_unknown": "track",
    "recalibrate": "geometry",
    "stadium_motion_unknown": "geometry",
    "stop_evidence_missing": "finish",
    "pocket_entry_unconfirmed": "finish",
    "exit_returned": "finish",
    "landing_timing_unknown": "finish",
}
_PREFIX = (
    ("after_finish:", "finish"),
    ("two_bey_", "track"),
    ("track_", "track"),
    ("calibration_", "geometry"),
    ("geometry_", "geometry"),
    ("shake_", "geometry"),
    ("identity_", "identity"),
    ("finish_", "finish"),
    ("air_", "finish"),
)


def stage_of(reason: str) -> str:
    """Pipeline stage a review reason points at; unknown codes stay visible as 'other'."""
    code = str(reason).split("@", 1)[0]
    if code in _EXACT:
        return _EXACT[code]
    for prefix, stage in _PREFIX:
        if code.startswith(prefix):
            return stage
    if "_stop_" in code:
        return "finish"
    return "other"


def stage_states(reasons: list[str]) -> list[dict]:
    flagged: dict[str, list[str]] = {}
    for reason in reasons:
        flagged.setdefault(stage_of(reason), []).append(reason)
    return [
        {
            "id": key,
            "label": label,
            "status": "flag" if key in flagged else "ok",
            "reasons": flagged.get(key, []),
        }
        for key, label in STAGES
        if key != "other" or key in flagged
    ]


def round_summary(row: dict, review_confidence: float) -> dict:
    shown = effective(row)
    reasons = list(row.get("review_reasons") or [])
    confidence = float(shown.get("confidence") or 0.0)
    finish = shown.get("finish")
    review = bool(row.get("needs_review")) and not row.get("reviewed")
    return {
        "video": row["video"],
        "round": row["round"],
        "state": "review" if review else "judged",
        "finish": finish,
        "finish_label": FINISH_NAMES.get(finish, finish) if finish else "未判定",
        "winner": shown.get("winner"),
        "points": shown.get("points") or 0,
        "confidence": confidence,
        "threshold": review_confidence,
        "above_threshold": confidence >= review_confidence,
        "reasons": reasons,
        "reviewed": bool(row.get("reviewed")),
        "t_launch": row.get("t_launch"),
        "t_end": row.get("t_end"),
    }


def pack_trajectory(path: Path, max_points: int = 6000) -> tuple[dict, dict]:
    """Retain entity/run tags; max_points applies to each continuous entity run."""
    if type(max_points) is not int or max_points < 1:
        raise ValueError("trajectory display limit must be a positive integer")
    sides: dict[str, list] = {}
    times: dict[int, float] = {}
    if not Path(path).is_file():
        return sides, times
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                frame, t = int(row["frame"]), float(row["t"])
                x, y = float(row["x"]), float(row["y"])
                score = float(row["score"] or "nan")
            except ValueError:
                continue  # Missing position rows (gaps) carry blank coordinates.
            if not all(math.isfinite(v) for v in (t, x, y)):
                continue
            times[frame] = t
            sides.setdefault(row["side"], []).append(
                [
                    round(t, 4),
                    round(x, 1),
                    round(y, 1),
                    round(score, 3) if math.isfinite(score) else None,
                    int(row["track_id"]),
                    row["zone"],
                    frame,
                ]
            )
    for side, points in sides.items():
        points.sort(key=lambda p: (p[6], p[4]))
        entities = defaultdict(list)
        for point in points:
            entities[point[4]].append(point)
        for entity_points in entities.values():
            previous, segment = None, 0
            for point in sorted(entity_points, key=lambda p: p[6]):
                if previous is None or point[6] != previous + 1:
                    segment += 1
                point.append(segment)
                previous = point[6]
        segments = defaultdict(list)
        for point in points:
            segments[point[4], point[7]].append(point)
        shown = []
        for segment_points in segments.values():
            if len(segment_points) <= max_points:
                shown.extend(segment_points)
            elif max_points == 1:
                shown.append(segment_points[0])
            else:
                last = len(segment_points) - 1
                shown.extend(
                    segment_points[i * last // (max_points - 1)]
                    for i in range(max_points)
                )
        sides[side] = sorted(shown, key=lambda p: (p[6], p[4]))
    return sides, times


def _warp(points, h):
    out = []
    for x, y in points:
        w = h[2][0] * x + h[2][1] * y + h[2][2]
        if not w:
            return None
        out.append(
            [
                round((h[0][0] * x + h[0][1] * y + h[0][2]) / w, 1),
                round((h[1][0] * x + h[1][1] * y + h[1][2]) / w, 1),
            ]
        )
    return out


def stadium_zones(
    calibration: dict | None, geometry: dict | None, frame: int | None
) -> dict:
    """A verified launch-frame reference only, never live or borrowed geometry."""
    if not calibration:
        return {"status": "missing", "zones": {}}
    zones = {k: calibration[k] for k in ZONES if calibration.get(k)}
    samples = geometry["samples"] if geometry is not None else {}
    best = samples.get(frame)
    if (
        best is None
        or best["alignment"] is None
        or best["status"] not in {"tracking", "relocated", "temporal_verified"}
    ):
        return {"status": "unavailable", "zones": {}}
    warped = {k: _warp(v, best["alignment"]["homography"]) for k, v in zones.items()}
    if any(v is None for v in warped.values()):
        return {"status": "unavailable", "zones": {}}
    return {
        "status": "launch_reference",
        "zones": warped,
        "sample_frame": best["frame"],
        "sample_status": best["status"],
    }


def round_boxes(
    two_bey: dict | None, number: int, times: dict, fps: float, initial_sides: dict
) -> list:
    """Assigned observations (primary) and unassigned candidates, as time-sorted frames."""
    rnd = ((two_bey or {}).get("rounds") or {}).get(str(number))
    if not rnd:
        return []
    local_ids = sorted({obs["entity_id"] for obs in rnd["observations"]})
    physical_ids = sorted(int(k) for k in rnd["members"])
    entity_map = dict(zip(local_ids, physical_ids, strict=True))
    if len({physical - local for local, physical in entity_map.items()}) > 1:
        raise ValueError("inconsistent physical entity namespace")
    frames: dict[int, list] = {}
    for obs in rnd.get("observations") or []:
        physical_id = entity_map[obs["entity_id"]]
        side = initial_sides[str(physical_id)]
        if not obs["identity_valid"] or identity_unknown_at(
            rnd, physical_id, obs["frame"]
        ):
            side = None
        frames.setdefault(obs["frame"], []).append(
            [
                *(round(v, 1) for v in obs["box"]),
                round(obs["score"], 3),
                "primary",
                side or f"#{physical_id} · 身分未知",
            ]
        )
    for obs in rnd.get("unassigned") or []:
        frames.setdefault(obs["frame"], []).append(
            [
                *(round(v, 1) for v in obs["box"]),
                round(obs["score"], 3),
                "candidate",
                obs.get("reason") or "unassigned",
            ]
        )
    return [
        [round(times.get(f, f / fps), 4), f, boxes]
        for f, boxes in sorted(frames.items())
    ]


def _geometry_for(run: Path, row: dict, source: str | None) -> dict | None:
    if row["geometry_evidence"] is None:
        return None
    path = (run / row["geometry_evidence"]["path"]).resolve()
    if not path.is_relative_to(run.resolve() / "geometry"):
        raise ValueError("geometry evidence path outside result geometry folder")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != row["geometry_evidence"]["sha256"]:
        raise ValueError("geometry evidence SHA256 mismatch")
    raw = json.loads(content)
    geometry = read_geometry(run, row, raw["source_sha256"])
    if source is None or Path(geometry["source"]).resolve() != Path(source).resolve():
        raise ValueError("geometry evidence source differs from video metadata")
    return geometry


def media_offset(source: str | None) -> float:
    """Video stream start time: browsers present frame n at start + n/fps, the pipeline at n/fps."""
    if not source or not Path(source).is_file():
        return 0.0
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=start_time:format=start_time",
                "-of",
                "json",
                source,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        data = json.loads(out)
        stream = float(data["streams"][0]["start_time"])
        container = float((data.get("format") or {}).get("start_time") or 0.0)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        return 0.0
    offset = stream - container
    return round(offset, 6) if math.isfinite(offset) else 0.0


def _fps(times: dict, default: float = 30.0) -> float:
    rates = sorted(f / t for f, t in times.items() if t > 0 and f > 0)
    return rates[len(rates) // 2] if rates else default


def store_counts(stores: dict[str, Path]) -> dict:
    return {
        name: len(list(Path(p).glob("*.json"))) if Path(p).is_dir() else 0
        for name, p in stores.items()
    }


def build_view(
    run: Path, review_confidence: float, stores: dict[str, Path] | None = None
) -> dict:
    """Index plus per-round payloads for one run directory (read-only)."""
    run = Path(run).resolve()
    rows = read_rounds(run)
    videos = (
        json.loads((run / "videos.json").read_text())
        if (run / "videos.json").is_file()
        else {}
    )
    geometry_cache = {}
    calibration_cache = {}
    offsets: dict[str, float] = {}
    rounds, payloads = [], {}
    for row in sorted(rows, key=lambda r: (r["video"], r["round"])):
        video, number = row["video"], row["round"]
        info = videos.get(video) or {}
        summary = round_summary(row, review_confidence)
        sides, times = pack_trajectory(
            run / "trajectories" / f"{Path(video).stem}_r{number}.csv"
        )
        fps = _fps(times)
        evidence = row["geometry_evidence"]
        key_evidence = (
            (
                evidence["path"],
                evidence["sha256"],
                row["calibration_id"],
                info.get("source_video"),
            )
            if evidence
            else None
        )
        if key_evidence not in geometry_cache:
            geometry_cache[key_evidence] = _geometry_for(
                run, row, info.get("source_video")
            )
        geometry = geometry_cache[key_evidence]
        calib_id = row["calibration_id"]
        if calib_id is not None and calib_id not in calibration_cache:
            calibration_cache[calib_id] = load_snapshot(
                run, calib_id, autocalib=load_autocalib_config()
            )
        snap = calibration_cache.get(calib_id)
        calibration = (
            asdict(
                scale_calibration(snap.calibration, tuple(geometry["reference_size"]))
            )
            if snap is not None and geometry is not None
            else None
        )
        launch = row.get("t_launch")
        launch_frame = round(launch * fps) if isinstance(launch, (int, float)) else None
        key = f"{Path(video).stem}_r{number}"
        provenance = row.get("detector_provenance") or {}
        two_bey = geometry.get("two_bey_tracking") if geometry else None
        initial_sides = (
            geometry["identity"]["physical_initial_sides"]
            if two_bey is not None
            else {}
        )
        payloads[key] = {
            **summary,
            "key": key,
            "src": "/video/" + video if info.get("source_video") else None,
            "fps": fps,
            "media_offset": offsets.setdefault(
                video, media_offset(info.get("source_video"))
            ),
            "coordinate_size": (calibration or {}).get("frame_size") or [1280, 720],
            "stages": stage_states(summary["reasons"]),
            "geometry": stadium_zones(calibration, geometry, launch_frame),
            "calibration": {
                "id": calib_id,
                "kind": snap.kind if snap else None,
                "score": snap.score if snap else None,
            },
            "tracks": sides,
            "boxes": round_boxes(two_bey, number, times, fps, initial_sides),
            "detector": {
                "kind": provenance.get("kind"),
                "weights": Path(provenance["weights"]).name
                if provenance.get("weights")
                else None,
            },
            "vlm_note": row.get("vlm_note"),
        }
        rounds.append(
            {
                k: summary[k]
                for k in (
                    "video",
                    "round",
                    "state",
                    "finish_label",
                    "winner",
                    "confidence",
                    "reviewed",
                )
            }
            | {
                "key": key,
                "reason_count": len(summary["reasons"]),
                "stages": [
                    s["id"] for s in payloads[key]["stages"] if s["status"] == "flag"
                ],
            }
        )
    reasons = Counter(
        r.split("@", 1)[0] for row in rows for r in row.get("review_reasons") or []
    )
    index = {
        "run": str(run),
        "round_count": len(rows),
        "review_count": sum(r["state"] == "review" for r in rounds),
        "threshold": review_confidence,
        "top_reasons": [
            [code, n, stage_of(code)] for code, n in reasons.most_common(8)
        ],
        "stores": store_counts(stores or {}),
        "rounds": sorted(
            rounds, key=lambda r: (r["state"] != "review", -r["reason_count"], r["key"])
        ),
    }
    return {
        "index": index,
        "rounds": payloads,
        "videos": {
            v: i["source_video"] for v, i in videos.items() if i.get("source_video")
        },
    }
