"""Full-rate, single streaming decode and batch orchestration."""

from __future__ import annotations

import gc
import json
import logging
import math
import sys
import time
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
from itertools import chain
from pathlib import Path

import cv2
import numpy as np

from .calibration import find_calibration, scale_calibration
from .config import Thresholds, load_thresholds
from .detect import Detector, load_detector
from .detector_config import load_policy
from .finish import (
    AFTER_FINISH_PREFIX,
    detect_finish,
    is_after_finish,
    orbit_direction,
    rotation_rate,
    spin_activity,
    stop_frame,
)
from .launch_check import analyze_launch
from .manifest import VideoInfo, mark_processed, scan
from .output import (
    read_rounds,
    record_pipeline_timing,
    record_run_timing,
    record_video_timing,
    row_to_result,
    write_results,
    write_trajectory,
)
from .rules import Rules, finish_points, is_not_a_round, load_rules
from .segment import segment_rounds
from .stadium_motion import iter_stadium_displacement
from .track import track_frames
from .types import RoundResult, Track
from .video import VideoDecodeError, iter_frames
from .vlm import load_vlm_config, needs_vlm, review_results_dir

log = logging.getLogger(__name__)


def _review(video: str, duration: float, reason: str) -> RoundResult:
    return RoundResult(
        video, 1, 0.0, duration, None, None, None, 0, 0.0, True, [reason]
    )


def _provenance(
    results: list[RoundResult], provenance: dict | None
) -> list[RoundResult]:
    for r in results:
        r.detector_provenance = provenance
        if provenance and provenance["dataset"]["provenance"].get("synthetic") is True:
            if "synthetic_detector" not in r.review_reasons:
                r.review_reasons.append("synthetic_detector")
            r.needs_review = True
    return results


def _crop(frame: np.ndarray, box) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
    x1, y1 = min(w, math.ceil(x1)), min(h, math.ceil(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.resize(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (64, 64))


def judge_video(
    info: VideoInfo,
    calib_dir: Path,
    detector: Detector,
    rules: Rules,
    th: Thresholds,
    *,
    out_dir: Path | None = None,
    provenance: dict | None = None,
    detection_config: Path | str = "config/detection.yaml",
) -> list[RoundResult]:
    """Analyze one video using playback fps and untouched source frame indices.

    Only the first second is buffered to select a calibration. Subsequent frames
    flow through motion measurement, detection and tracking once; only crops and
    compact observations survive. Short clips use their final available frame.
    """
    try:
        with closing(
            iter_frames(
                info.path, long_edge=th.infer_long_edge, hwaccel=th.decode_hwaccel
            )
        ) as frames:
            prefix = []
            for idx, frame in frames:
                prefix.append((idx, frame))
                if idx >= round(info.fps):
                    break
            if not prefix:
                raise VideoDecodeError(f"{info.path}: no frames decoded")
            c = find_calibration(
                prefix[-1][1], calib_dir, th.calibration_min_similarity
            )
            if c is None:
                return _provenance(
                    [_review(info.path.name, info.duration, "recalibrate")], provenance
                )
            c = scale_calibration(c, (prefix[-1][1].shape[1], prefix[-1][1].shape[0]))
            ref = cv2.imread(c.ref_frame)
            if ref is None:
                raise FileNotFoundError(c.ref_frame)
            disp = {}
            crops = {}

            def measured():
                for f, image, displacement in iter_stadium_displacement(
                    chain(prefix, frames), c, ref
                ):
                    disp[f] = displacement
                    yield f, image
                prefix.clear()

            def detecting(image, f):
                detections = detector(image, f)
                for d in detections:
                    if d.cls == "bey":
                        crop = _crop(image, d.box)
                        if crop is not None:
                            crops[(d.frame, tuple(d.box))] = crop
                return detections

            tracks = track_frames(
                measured(), detecting, info.fps, config=detection_config
            )
        bey_tracks = [t for t in tracks if t.cls == "bey"]
        rounds = segment_rounds(bey_tracks, info.fps, th)
        if not rounds:
            return _provenance(
                [_review(info.path.name, info.duration, "no_round")], provenance
            )
        results = []
        last_decoded_frame = max(disp)
        for number, (start, end) in enumerate(rounds, 1):
            next_start = (
                rounds[number][0] if number < len(rounds) else last_decoded_frame + 1
            )
            part_end = min(
                end + round(th.burst_part_window_s * info.fps),
                next_start - 1,
                last_decoded_frame,
            )
            selected = [
                Track(
                    t.track_id,
                    t.cls,
                    [
                        d
                        for d in t.dets
                        if start
                        <= d.frame
                        <= (part_end if t.cls == "bey_part" else end)
                    ],
                )
                for t in tracks
            ]
            current = [t for t in selected if t.cls == "bey" and t.dets]
            launchers = [t for t in tracks if t.cls == "launcher"]
            analysis = analyze_launch(current, launchers, disp, c, th, info.fps)
            launch, sides = analysis.check, analysis.sides
            counts = Counter(sides.values())
            bey = {
                sides[t.track_id]: t
                for t in current
                if t.track_id in sides and counts[sides[t.track_id]] == 1
            }
            parts = [t for t in selected if t.cls == "bey_part" and t.dets]
            spin, stops, stop_kinds = {}, {}, {}
            for side in ("X", "B"):
                t = bey.get(side)
                spin[side] = (
                    orbit_direction(t, c, launch.landing_frame[side], th, info.fps)
                    if t
                    else "unknown"
                )
                if t:
                    observed_crops = {
                        d.frame: crops[(d.frame, tuple(d.box))]
                        for d in t.dets
                        if (d.frame, tuple(d.box)) in crops
                    }
                    stopped, reason = stop_frame(
                        t,
                        spin_activity(observed_crops),
                        rotation_rate(observed_crops),
                        spin[side],
                        th,
                    )
                    stops[side] = stopped
                    if reason:
                        stop_kinds[side] = reason
                else:
                    stops[side] = None
            finish = detect_finish(bey, parts, stops, c, th, info.fps)
            # A stop arising only after the result is settled is a note (R55).
            stop_reasons = [
                (
                    AFTER_FINISH_PREFIX
                    if is_after_finish(finish, side, stops[side], th)
                    else ""
                )
                + f"{side}_stop_{reason}"
                for side, reason in stop_kinds.items()
            ]
            reasons = list(
                dict.fromkeys([*launch.notes, *finish.reasons, *stop_reasons])
            )
            # These explain observed evidence, while reversals remain review-required.
            uncertainty = [
                note
                for note in reasons
                if not (
                    note.startswith(
                        (
                            "shake_by_bey_impact@",
                            "shake_not_by_launcher_body@",
                            AFTER_FINISH_PREFIX,
                        )
                    )
                    or note in ("X_stop_still", "B_stop_still")
                )
            ]
            unknown = (
                any(v == "unknown" for v in launch.touch.values())
                or launch.air_collision == "maybe"
                or "unknown" in spin.values()
            )
            needs_review = (
                finish.confidence < th.review_confidence
                or unknown
                or finish.finish is None
                or bool(uncertainty)
            )
            winner = ("B" if finish.loser == "X" else "X") if finish.loser else None
            r = RoundResult(
                info.path.name,
                number,
                start / info.fps,
                (finish.frame if finish.frame is not None else end) / info.fps,
                launch,
                finish.finish,
                winner,
                finish_points(rules, finish.finish) if finish.finish else 0,
                finish.confidence,
                needs_review,
                reasons,
                spin_dir=spin,
            )
            lo, hi = analysis.window
            r.motion_evidence = {
                "video": r.video,
                "round": number,
                "window": [lo, hi],
                "window_semantics": "[start,end)",
                "fps": info.fps,
                "displacements": {
                    str(f): disp[f] if math.isfinite(disp.get(f, math.nan)) else None
                    for f in range(lo, hi)
                },
                "impact_frames": sorted(analysis.impacts),
            }
            results.append(r)
            if out_dir is not None:
                write_trajectory(
                    out_dir,
                    r,
                    bey,
                    parts,
                    c,
                    info.fps,
                    {
                        "landing_X": launch.landing_frame["X"],
                        "landing_B": launch.landing_frame["B"],
                        "air_collision": analysis.air_collision_frame,
                        "finish": finish.frame,
                    },
                    unassigned=[
                        t
                        for t in current
                        if t.track_id not in {v.track_id for v in bey.values()}
                    ],
                )
        return _provenance(results, provenance)
    except VideoDecodeError:
        log.exception("cannot decode %s", info.path)
        return _provenance(
            [_review(info.path.name, info.duration, "decode_error")], provenance
        )


def _load_selection(path: Path, th: Thresholds) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("kind") not in ("rfdetr", "yolo26"):
        raise ValueError(f"{path}: expected selected detector kind")
    try:
        dataset_provenance = data["dataset"]["provenance"]
        if not isinstance(dataset_provenance, dict) or not dataset_provenance:
            raise ValueError(f"{path}: expected nonempty dataset provenance")
        synthetic = dataset_provenance.get("synthetic", False)
        edge = data["inference"]["long_edge"]
        weights = Path(data["weights"])
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path}: incomplete detector selection/provenance") from e
    if type(synthetic) is not bool or edge != th.infer_long_edge:
        raise ValueError(f"{path}: invalid provenance or inference policy mismatch")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return {**data, "selection_path": str(path.resolve())}


def _validate_thresholds(th: Thresholds) -> None:
    for name, value in asdict(th).items():
        if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
            raise ValueError(f"threshold {name} must be finite and nonnegative")
    for name in (
        "infer_long_edge",
        "fps_default",
        "settle_frames",
        "still_frames",
        "reverse_frames",
        "id_swap_overlap_frames",
    ):
        if getattr(th, name) <= 0:
            raise ValueError(f"threshold {name} must be positive")
    for name in (
        "review_confidence",
        "calibration_min_similarity",
        "id_swap_confidence_factor",
        "track_gap_confidence_factor",
    ):
        if getattr(th, name) > 1:
            raise ValueError(f"threshold {name} must be in [0, 1]")


def run_batch(
    folder: Path,
    calib_dir: Path,
    out_dir: Path,
    *,
    selection: Path = Path("weights/selected.json"),
    thresholds: Path = Path("config/thresholds.yaml"),
    rules_path: Path = Path("config/rules.yaml"),
    detection_config: Path = Path("config/detection.yaml"),
    limit: int | None = None,
    detector: Detector | None = None,
    rules: Rules | None = None,
    th: Thresholds | None = None,
    vlm_config: Path | None = None,
) -> None:
    # R61: batch wall-clock from here (before config/model init) to the end of the VLM phase.
    run_began = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    folder, out_dir = Path(folder).expanduser(), Path(out_dir).expanduser()
    if not folder.is_dir():
        raise ValueError(f"{folder}: expected video folder")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"]
    duplicates = [
        name
        for name, count in Counter(p.name.casefold() for p in paths).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"duplicate video basename/artifact(s): {duplicates}")
    # A wrong calibration directory is global, not a per-video "recalibrate" (R56).
    calib_dir = Path(calib_dir).expanduser()
    if not calib_dir.is_dir():
        raise ValueError(f"{calib_dir}: expected calibration directory (--calib)")
    # Configuration and model initialization fail globally, before any result/cache writes.
    injected_thresholds = th is not None
    th = load_thresholds(thresholds) if th is None else th
    rules = load_rules(rules_path) if rules is None else rules
    _validate_thresholds(th)
    policy = load_policy(detection_config)
    # Spec section 8: the optional VLM phase is configured (and its loopback endpoint
    # validated) up front; None or after_run: false keeps the run Ollama-free.
    vlm = load_vlm_config(vlm_config) if vlm_config is not None else None
    if vlm is not None and not vlm.after_run:
        vlm = None
    provenance = None
    if detector is None:
        provenance = _load_selection(Path(selection).expanduser(), th)
        provenance["runtime"] = {
            "detection_config": str(Path(detection_config).resolve()),
            "thresholds_path": None
            if injected_thresholds
            else str(Path(thresholds).resolve()),
            "rules_path": str(Path(rules_path).resolve()),
            "confidence": policy["confidence"],
            "tracking": policy["tracking"],
            "device": policy["device"],
            "thresholds": asdict(th),
            "rules": asdict(rules),
        }
        detector = load_detector(
            provenance["kind"],
            Path(provenance["weights"]),
            config=detection_config,
            thresholds=th,
        )
    db = out_dir / "processed_videos.json"
    failures = []
    infos = scan(
        folder, db, on_error=lambda path, error: failures.append((path, error))
    )
    entries = sorted([(i.path, i) for i in infos] + [(p, None) for p, _ in failures])
    if limit is not None:
        entries = entries[:limit]
    # Invalidate the entire scheduled set before any core/output work can fail.
    for path, _ in entries:
        record_pipeline_timing(out_dir, path.name, None)
    starts: dict[str, float] = {}

    def complete(video: str) -> None:
        record_pipeline_timing(out_dir, video, time.perf_counter() - starts[video])

    for path, info in entries:
        began = time.perf_counter()
        starts[path.name] = began
        if info is None:
            results = _provenance([_review(path.name, 0.0, "decode_error")], provenance)
        else:
            try:
                results = judge_video(
                    info,
                    calib_dir,
                    detector,
                    rules,
                    th,
                    out_dir=out_dir,
                    provenance=provenance,
                    detection_config=detection_config,
                )
            except Exception:
                log.exception("analysis failed for %s; continuing batch", path)
                results = _provenance(
                    [_review(path.name, info.duration, "analysis_error")], provenance
                )
        write_results(results, out_dir, rules)
        processing = time.perf_counter() - began
        record_video_timing(
            out_dir,
            path.name,
            info.duration if info else 0.0,
            processing,
            source_video=path.resolve(),
        )
        # Placeholder-only videos await corrected input and are retried (R56).
        if info is not None and not all(is_not_a_round(r.review_reasons) for r in results):
            mark_processed(info, db)
        log.info(
            "%s: processing %.3fs / playback %.3fs",
            path.name,
            processing,
            info.duration if info else 0.0,
        )
        # The writer may preserve reviewed/audited rows instead of fresh results.
        # Match the advisory phase's persisted snapshot before deciding to wait.
        pending_vlm = vlm is not None and any(
            needs_vlm(row_to_result(row))
            for row in read_rounds(out_dir) if row["video"] == path.name
        )
        if not pending_vlm:
            complete(path.name)
    vlm_wall_s = None
    if vlm is not None and entries:
        # The detector must be gone before the ~18-20 GB VLM loads (spec section 5).
        vlm_began = time.perf_counter()
        del detector
        _release_gpu_memory()
        review_results_dir(
            out_dir,
            calib_dir=calib_dir,
            th=th,
            rules=rules,
            config=vlm,
            videos={path.name for path, _ in entries},
            on_video_complete=complete,
        )
        vlm_wall_s = time.perf_counter() - vlm_began
    if entries:  # nothing judged = no judgment wall-clock to report
        record_run_timing(
            out_dir,
            kind="run",
            started_at=started_at,
            wall_s=time.perf_counter() - run_began,
            vlm_wall_s=vlm_wall_s,
            videos={path.name: info.duration if info else 0.0 for path, info in entries},
        )


def _release_gpu_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
