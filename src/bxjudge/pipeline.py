"""Full-rate, single streaming decode and batch orchestration."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime
from itertools import chain
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from . import DEFAULT_AUTOCALIB_CONFIG
from .autocalib import AutoCalibConfig, load_autocalib_config
from .calibration import scale_calibration
from .calibration_store import (
    Snapshot,
    load_snapshot,
    select_snapshot,
    validate_output_paths,
    vetoed_ids,
    write_preview,
)
from .camera_motion import (
    DEFAULT_CAMERA_MODE_CONFIG,
    decide_camera_mode,
    load_camera_policy,
    measure_video_motion,
)
from .config import Thresholds, load_thresholds
from .detect import Detector, load_detector
from .detector_config import load_policy
from .dynamic_calibration import (
    DynamicCalibration,
    load_dynamic_policy,
    normalize_tracks,
    save_geometry,
)
from .finish import (
    AFTER_FINISH_PREFIX,
    detect_finish,
    is_after_finish,
    orbit_direction,
    rotation_rate,
    spin_activity,
    stop_frame,
)
from .handheld_calibration import HandheldCalibration
from .identity import anchor_sides, inherited_sides, load_identity_policy
from .launch_check import analyze_launch
from .launch_mask_review import (
    SamLaunchMaskReviewer,
    build_release_probes,
    review_release_probes,
)
from .launch_phase import (
    launch_sample_count,
    load_launch_policy,
    observe_launch_segments,
    prepare_launch_tracks,
)
from .manifest import (
    VideoInfo,
    forget_processed,
    mark_processed,
    processed_calibration,
    processed_videos,
    scan,
)
from .output import (
    TRANSIENT_FAILURES,
    read_rounds,
    record_run_timing,
    record_video_analysis_policy,
    record_video_attempt,
    record_video_calibration,
    record_video_status,
    record_video_timing,
    write_results,
    write_trajectory,
)
from .rules import Rules, finish_points, is_not_a_round, load_rules
from .segment import segment_rounds
from .stadium_finder import zone_visibility
from .stadium_locator import KeypointLocator, keypoint_snapshot, load_stadium_tools
from .stadium_motion import RAW_MOTION_WORKERS, iter_stadium_displacement
from .temporal_geometry import load_temporal_policy
from .timebase import (
    JudgmentTime,
    load_physical_time,
    load_video_sources,
    probe_timeline,
)
from .track import track_frames
from .trajectory_video import load_video_policy
from .two_bey_integration import (
    appearance_descriptor,
    infer_opponent_side,
    reconstruct_round_tracks,
    round_release_evidence,
)
from .two_bey_tracking import identity_unknown_at
from .two_bey_tracking import load_policy as load_two_bey_policy
from .types import RoundResult, Track
from .video import VideoDecodeError, iter_frames, prefetched
from .video_anomalies import AnomalyPolicy, ImageChanges, annotate_round
from .vlm import load_vlm_config, review_results_dir

log = logging.getLogger(__name__)

# Ruling F-1: a round judged with an auto snapshot scoring below auto_review_score.
CALIBRATION_LOW_SCORE = "calibration_low_score"


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


def _calibrated(results: list[RoundResult], snap: Snapshot) -> list[RoundResult]:
    for r in results:
        r.calibration_id = snap.id
    return results


def _crop(frame: np.ndarray, box) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
    x1, y1 = min(w, math.ceil(x1)), min(h, math.ceil(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.resize(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (64, 64))


def _first_second(
    frames: Iterable[tuple[int, np.ndarray]], fps: float
) -> Iterator[tuple[int, np.ndarray]]:
    """Frames of ``frames`` up to and including source index ``round(fps)`` (fewer for a short
    clip); its last frame is the one a calibration is selected on. Pulls no frame beyond it."""
    for idx, frame in frames:
        yield idx, frame
        if idx >= round(fps):
            return


def first_second_frame(info: VideoInfo, th: Thresholds) -> np.ndarray:
    """The calibration-selection frame of ``info``, decoded exactly as ``judge_video`` decodes
    it (inference size, same frame index). Raises ``VideoDecodeError``."""
    last = None
    with closing(
        iter_frames(info.path, long_edge=th.infer_long_edge, hwaccel=th.decode_hwaccel)
    ) as frames:
        for _, last in _first_second(frames, info.fps):
            pass
    if last is None:
        raise VideoDecodeError(f"{info.path}: no frames decoded")
    return last


_SELECT = (
    object()
)  # judge_video default: select the snapshot from the decoded first second


def judge_video(
    info: VideoInfo,
    calib_dir: Path,
    detector: Detector,
    rules: Rules,
    th: Thresholds,
    *,
    out_dir: Path,
    provenance: dict | None = None,
    detection_config: Path | str = "config/detection.yaml",
    autocalib: AutoCalibConfig,
    snapshot: Snapshot | None | object = _SELECT,
    dynamic_config: Path | None = None,
    two_bey_config: Path | None = None,
    on_snapshot: Callable[[Snapshot | None], None] | None = None,
    vetoed: set[str] | frozenset[str] = frozenset(),
    timing: JudgmentTime | None = None,
    anomaly_policy: AnomalyPolicy | None = None,
    stadium_tools=None,
    camera_policy: dict | None = None,
    launch_mask_reviewer: SamLaunchMaskReviewer | None = None,
) -> list[RoundResult]:
    """Analyze one video using playback fps and untouched source frame indices.

    Only the first second is buffered to select a calibration snapshot
    (``select_snapshot``: manual, else this frame's own auto, else ``None`` = ``recalibrate``),
    unless the caller already selected one on that same frame (``snapshot``, may be ``None``);
    ``vetoed`` snapshot ids are honoured by that selection (ruling T4-1).
    The chosen snapshot (or ``None``) is reported to ``on_snapshot``. Its polygons are scaled to
    the decoded frame size; the stadium-displacement reference is that same first-second frame
    of this video (ruling T3-1: the snapshot's image serves only selection and previews);
    every result carries the snapshot id and its preview is written. Subsequent frames
    flow through motion measurement, detection and tracking once; only crops and
    compact observations survive. Short clips use their final available frame.

    ``stadium_tools`` (``stadium_locator.StadiumTools``; plan 2026-11-12-dynamic-calibration):
    when no snapshot is selected, the stadium keypoints give a third choice
    (``keypoint_snapshot``), and the dynamic trackers may relocate from keypoints.
    ``camera_policy`` (``camera_motion.load_camera_policy``): the camera is judged fixed or
    handheld from early motion. A handheld camera needs ``dynamic_config`` and a keypoint
    locator, else the video is ``recalibrate``; its geometry is ``HandheldCalibration``,
    its stadium displacement is not measured (camera motion would read as shake) and launch
    touch is unknown (``camera_handheld``). Without either argument nothing changes.
    """
    if two_bey_config is not None and dynamic_config is None:
        raise ValueError('two-bey tracking requires dynamic calibration')
    if launch_mask_reviewer is not None and two_bey_config is None:
        raise ValueError("launch mask review requires two-bey tracking")
    values = timing.values if timing is not None else {}

    def converted(policy):
        return {key: values.get(key, value) for key, value in policy.items()}

    if values:
        th = replace(th, **{key: values[key] for key in asdict(th).keys() & values.keys()})
    two_policy = (
        converted(load_two_bey_policy(two_bey_config)) if two_bey_config is not None else None
    )
    evidence_confidence = load_policy(detection_config)["confidence"]
    source_before = None
    if dynamic_config is not None:
        with info.path.open("rb") as f:
            source_before = hashlib.file_digest(f, "sha256").hexdigest()
    cancelled = Event()
    changes = ImageChanges(anomaly_policy) if timing and anomaly_policy else None
    try:
        with closing(
            iter_frames(
                info.path, long_edge=th.infer_long_edge, hwaccel=th.decode_hwaccel,
                cancelled=cancelled
            )
        ) as frames:
            prefix = list(_first_second(frames, info.fps))
            if not prefix:
                raise VideoDecodeError(f"{info.path}: no frames decoded")
            first = prefix[-1][1]
            camera = motion = motion_error = None
            if camera_policy is not None:
                try:
                    motion = measure_video_motion(
                        info.path, info.fps, camera_policy, long_edge=th.infer_long_edge,
                        hwaccel=th.decode_hwaccel,
                    )
                except (VideoDecodeError, ValueError, OSError, RuntimeError, MemoryError,
                        cv2.error) as e:
                    # Unmeasured: decide_camera_mode treats it as handheld (cautious).
                    motion, motion_error = None, f"{type(e).__name__}: {e}"
                camera = decide_camera_mode(info.path, info.path.parent.name, motion,
                                            camera_policy)
            handheld = camera is not None and camera.mode == "handheld"
            camera_record = None if camera is None else {
                "mode": camera.mode,
                "source": camera.source,
                "px_per_s": None if motion is None else motion.px_per_s,
                "pairs": 0 if motion is None else motion.pairs,
                **({"error": motion_error} if motion_error is not None else {}),
            }
            snap = (
                select_snapshot(first, calib_dir, out_dir, th, autocalib, vetoed,
                                video=info.path.name)
                if snapshot is _SELECT
                else snapshot
            )
            found = None
            keypoint_errors: list[str] = []
            if snap is None and stadium_tools is not None:
                # Third choice. A handheld camera may start from one keypoint fit, which
                # its tracker confirms before emitting any zone.
                snap, found = keypoint_snapshot(info.path, out_dir, stadium_tools, autocalib,
                                                vetoed, hwaccel=th.decode_hwaccel,
                                                allow_single=handheld,
                                                keypoint_errors=keypoint_errors)
                if snap is None:
                    found = None
            if on_snapshot is not None:
                on_snapshot(snap)
            if snap is None:
                placeholder = _review(info.path.name, info.duration, "recalibrate")
                if stadium_tools is not None:
                    # Say which routes failed: the keypoints too, by inference error or not.
                    placeholder.review_reasons.append("stadium_unlocated")
                    if keypoint_errors:
                        placeholder.review_reasons.append("keypoint_inference_error")
                    if handheld:
                        placeholder.review_reasons.append("camera_handheld")
                if camera_record is not None:
                    placeholder.motion_evidence = {"camera": camera_record}
                return _provenance([placeholder], provenance)
            if handheld and dynamic_config is None:
                placeholder = _review(info.path.name, info.duration, "recalibrate")
                placeholder.review_reasons.append("handheld_requires_dynamic_calibration")
                return _provenance([placeholder], provenance)
            keypoint_record = None
            if snap.kind == "keypoints":
                # Its stored origin decides, whether found now or handed over by the batch.
                fit = None if found is None else found.fit
                keypoint_record = {
                    "snapshot_id": snap.id,
                    "method": "keypoints" if snap.agreeing_frames >= 2 else "keypoints_single",
                    "frame_index": snap.source_frame,
                    "agreeing_frames": snap.agreeing_frames,
                    "coverage": float(snap.score if fit is None else fit.coverage),
                    "precision": None if fit is None else float(fit.precision),
                    "keypoint_error_px": None if fit is None else float(fit.keypoint_error_px),
                    "keypoint_rail_shift_px": (
                        None if fit is None else float(fit.keypoint_rail_shift_px)
                    ),
                }
            write_preview(out_dir, snap)
            # Ruling F-1: an accepted but loosely aligned auto snapshot (min_score <= score <
            # auto_review_score) is never scored silently. Manual snapshots are unaffected.
            calibration_notes = (
                [CALIBRATION_LOW_SCORE]
                if snap.kind == "auto" and snap.score < autocalib.auto_review_score
                else []
            )
            if snap.kind == "keypoints" and stadium_tools is not None:
                # A pocket mostly outside the frame cannot show a finish (auto-stadium spec).
                visibility = (found.zone_visibility if found is not None
                              else zone_visibility(snap.calibration))
                calibration_notes += [
                    f"zone_out_of_frame@{zone}"
                    for zone in ("over_left", "over_right", "xtreme")
                    if visibility.get(zone, 0.0) < stadium_tools.cfg.zone_visible_min
                ]
            if handheld:
                calibration_notes.append("camera_handheld")
            c = scale_calibration(snap.calibration, (first.shape[1], first.shape[0]))
            # Ruling T3-1 (spec 6.2b): shake is measured against this video's own camera view,
            # never the snapshot's reference image (another video's frame, or the manual
            # template), whose small offset would read as constant stadium displacement.
            ref = first
            geometry = None
            if dynamic_config is not None:
                reference = cv2.imread(snap.calibration.ref_frame)
                locator = None
                if stadium_tools is not None:
                    locator = KeypointLocator.for_snapshot(
                        stadium_tools, snap, c,
                        cv2.resize(reference, c.frame_size, interpolation=cv2.INTER_AREA),
                    )
                if handheld and locator is None:
                    placeholder = _review(info.path.name, info.duration, "recalibrate")
                    placeholder.review_reasons.append("handheld_reference_unlocated")
                    return _provenance([placeholder], provenance)
                if handheld:
                    temporal = (converted(load_temporal_policy()) if timing
                                else load_temporal_policy())
                    single = snap.kind == "keypoints" and snap.agreeing_frames < 2
                    start_frame = (
                        snap.source_frame if snap.kind == "keypoints"
                        else prefix[-1][0] if snap.kind == "auto" else None
                    )
                    geometry = HandheldCalibration(
                        c,
                        reference,
                        info.fps,
                        load_video_policy()["stadium_follow"],
                        converted(camera_policy["handheld"]),
                        locator,
                        minimum_visible=temporal["min_visible_fraction"],
                        foreground_margin_px=temporal["foreground_margin_px"],
                        start_confirmed=not single,
                        start_frame=start_frame,
                    )
                else:
                    geometry = DynamicCalibration(
                        c,
                        reference,
                        info.fps,
                        load_video_policy()["stadium_follow"],
                        converted(load_dynamic_policy(dynamic_config)),
                        **({"temporal_policy": converted(load_temporal_policy())}
                           if timing else {}),
                        keypoint_locator=locator,
                        fixed_camera=(
                            camera_record is not None
                            and camera_record.get("mode") == "fixed"
                        ),
                    )
            disp = {}
            crops = {}
            appearances={}

            def measured():
                if handheld:
                    # Camera motion would read as stadium shake: displacement is unknown.
                    for f, image in chain(prefix, frames):
                        disp[f] = math.nan
                        if changes is not None:
                            changes.observe(f, image)
                        yield f, image
                    prefix.clear()
                    return
                displacements = iter_stadium_displacement(
                    chain(prefix, frames), c, ref, workers=RAW_MOTION_WORKERS
                )
                # Close explicitly: an error the measurement raises on close then
                # reaches prefetched instead of being dropped at finalization.
                try:
                    for f, image, displacement in displacements:
                        disp[f] = displacement
                        if changes is not None:
                            changes.observe(f, image)
                        yield f, image
                finally:
                    displacements.close()
                prefix.clear()

            frame_detections = {}

            def detected():
                # Raw motion always uses untouched source pixels. Its producer can
                # prepare the next frame while this worker detects the current one.
                with prefetched(measured(), cancelled=cancelled) as raw_frames:
                    for f, image in raw_frames:
                        frame_detections[f] = detecting(image, f)
                        yield f, image

            def aligned():
                # Detection runs one stage ahead: the foreground mask still uses
                # the current frame's boxes while the next frame is detected.
                with prefetched(detected(), cancelled=cancelled) as detected_frames:
                    for f, image in detected_frames:
                        geometry.set_foreground(frame_detections[f])
                        geometry.update(f, image)
                        yield f, image

            def detecting(image, f):
                detections = detector(image, f)
                for d in detections:
                    if d.cls == "bey" and d.score >= evidence_confidence:
                        if two_policy is not None:
                            appearances[(d.frame,tuple(d.box))]=appearance_descriptor(image,d.box,two_policy['appearance_bins'])
                        crop = _crop(image, d.box)
                        if crop is not None:
                            crops[(d.frame, tuple(d.box))] = crop
                return detections

            prepared = aligned() if geometry is not None else measured()
            with prefetched(prepared, cancelled=cancelled) as prepared_frames:
                tracks = track_frames(
                    prepared_frames,
                    (lambda image, f: frame_detections.pop(f)) if geometry is not None else detecting,
                    info.fps, config=detection_config
                )
        if geometry is not None:
            geometry.finish()  # the handheld tracker re-finds its tail; fixed: nothing
        unobserved = None
        if geometry is not None:
            # Observations there were dropped: they cannot prove a launch miss.
            unobserved = {f for f, sample in geometry.samples.items()
                          if sample.alignment is None or sample.status == "bridged"}
        if timing and len(disp) != len(timing.timeline.pts_s):
            raise ValueError(f"{info.path}: decoded frame count does not match PTS count")
        anomalies = changes.events(timing.timeline) if changes is not None else []
        phase = None
        opponent_anchors = {}
        native_tracks=list(tracks)
        side_hints = {}
        geometry_evidence = None
        if geometry is not None:
            phase_policy = converted(load_launch_policy())
            required_anchor_frames = launch_sample_count(phase_policy["held_s"], info.fps)
            anchors = anchor_sides(
                tracks, normalize_tracks(tracks, geometry.samples), c,
                required_frames=required_anchor_frames,
                body_margin_px=phase_policy["body_margin_px"],
                fps=info.fps,
                **({"policy": converted(load_identity_policy())} if timing else {}),
            )
            phase = prepare_launch_tracks(
                tracks, info.fps, phase_policy, side_hints=anchors.sides
            )
            side_hints = inherited_sides(anchors.sides, phase.members)
            tracks = phase.tracks
            with info.path.open("rb") as source:
                source_sha = hashlib.file_digest(source, "sha256").hexdigest()
            if source_sha != source_before:
                raise ValueError("source changed during dynamic judging")
            record = {
                **geometry.record(),
                "calibration_id": snap.id,
                "source": str(info.path.resolve()),
                "source_sha256": source_sha,
                **({"camera": camera_record, "fps": info.fps}
                   if camera_record is not None else {}),
                **({"stadium_keypoints": keypoint_record}
                   if keypoint_record is not None else {}),
                "follow_policy": load_video_policy()["stadium_follow"],
                "phase_policy": phase_policy,
                "phase": phase.record(),
                "identity": {
                    "anchors": anchors.record(),
                    "sides": {str(k): v for k, v in side_hints.items()},
                },
            }
            if two_policy is None:
                geometry_evidence = save_geometry(out_dir, info.path.name, record)
        raw_tracks = tracks
        raw_by_id = {t.track_id: t for t in raw_tracks}
        bey_tracks = [t for t in raw_tracks if t.cls == "bey"]
        rounds = segment_rounds(bey_tracks, info.fps, th)
        physical_rounds={}
        adjudication_rounds={}
        releases_by_round = {}
        launch_mask_reviews = {}
        launch_mask_probes = []
        if two_policy is not None:
            native_segments = observe_launch_segments(native_tracks, info.fps, phase_policy, c.frame_size)
            required_separation_frames = launch_sample_count(phase_policy["separation_s"], info.fps)
            offset=max((t.track_id for t in native_tracks),default=0)
            phase_keys={(d.frame,tuple(d.box),d.score) for t in raw_tracks if t.cls=='bey' for d in t.dets}
            for number,(start,end) in enumerate(rounds,1):
                physical=reconstruct_round_tracks(native_tracks,start,end,info.fps,c.frame_size,appearances,two_policy,offset+number*2)
                for observation in physical.record['unassigned']:
                    observation['adjudication_eligible'] = (
                        observation['frame'], tuple(observation['box']), observation['score']
                    ) in phase_keys
                physical_rounds[number]=physical
                adjudication_rounds[number]=[Track(t.track_id,t.cls,[d for d in t.dets
                    if (d.frame,tuple(d.box),d.score) in phase_keys
                    and not identity_unknown_at(physical.record, t.track_id, d.frame)])
                    for t in physical.tracks]
                adjudication_rounds[number] = normalize_tracks(
                    adjudication_rounds[number], geometry.samples
                )
                for track in physical.tracks:
                    raw_by_id[track.track_id] = track
                side_hints.update(inherited_sides(anchors.sides, physical.members))
                proof = infer_opponent_side(
                    anchors, physical, adjudication_rounds[number], c,
                    required_frames=required_anchor_frames,
                )
                if proof is not None:
                    side_hints[proof["track_id"]] = proof["side"]
                    opponent_anchors[number] = proof
                releases_by_round[number] = round_release_evidence(
                    physical, adjudication_rounds[number], side_hints, native_segments, c,
                    required_frames=required_separation_frames,
                )
                if launch_mask_reviewer is not None:
                    launch_mask_probes.extend(build_release_probes(
                        number, physical, side_hints, native_segments, releases_by_round[number],
                    ))
            if launch_mask_probes:
                launch_mask_reviews = review_release_probes(
                    launch_mask_probes, native_tracks,
                    lambda: iter_frames(info.path, long_edge=th.infer_long_edge,
                                        hwaccel=th.decode_hwaccel),
                    launch_mask_reviewer,
                )
                with info.path.open("rb") as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != source_before:
                        raise ValueError("source changed during launch mask review")
            record["launch_timing"] = {
                "native_segments": {str(k): [asdict(segment) for segment in segments]
                                    for k, segments in native_segments.items()},
                "round_releases": releases_by_round,
            }
            if launch_mask_reviewer is not None:
                record["launch_timing"]["mask_review"] = launch_mask_reviews
            record["identity"]["opponent_anchors"] = opponent_anchors
            record['two_bey_tracking']={'policy':two_policy,'rounds':{str(n):p.record for n,p in physical_rounds.items()},
                'raw_source_tracks':[asdict(t) for t in native_tracks], 'association_source':'native_tracks',
                'adjudication_filter':'observed_phase_prepared_points_only'}
            record['identity']['physical_initial_sides']={str(t.track_id):side_hints.get(t.track_id) for p in physical_rounds.values() for t in p.tracks}
            record['identity']['physical_identity_unknown_since']={k:v for p in physical_rounds.values() for k,v in p.record['identity_unknown_since'].items()}
            record['identity']['physical_identity_unknown_intervals']={k:v for p in physical_rounds.values() for k,v in p.record['identity_unknown_intervals'].items()}
            geometry_evidence=save_geometry(out_dir,info.path.name,record)
        if geometry is not None:
            tracks = normalize_tracks(raw_tracks, geometry.samples)
        if not rounds:
            empty_results=[_review(info.path.name,info.duration,'no_round')]
            if two_policy is not None:
                empty_results[0].geometry_evidence=geometry_evidence
            return _calibrated(
                _provenance(
                    empty_results, provenance
                ),
                snap,
            )
        results = []
        last_decoded_frame = max(disp)
        for number, (start, end) in enumerate(rounds, 1):
            round_raw_tracks=raw_tracks
            round_tracks=tracks
            if number in physical_rounds:
                round_raw_tracks=[t for t in raw_tracks if t.cls!='bey']+physical_rounds[number].tracks
                round_tracks = [t for t in tracks if t.cls != "bey"] + adjudication_rounds[number]
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
                for t in round_tracks
            ]
            current = [t for t in selected if t.cls == "bey" and t.dets]
            if number in physical_rounds and physical_rounds[number].record['segment_status']=='pair_not_confirmed':
                # This candidate stays in the evaluation/review population, but
                # cannot acquire a new physical ID or fabricated launch analysis.
                r=RoundResult(info.path.name,number,start/info.fps,end/info.fps,None,None,None,0,0.,True,
                    ['two_bey_tracking_requires_review','two_bey_singleton_segment','two_bey_unassigned_identity_unknown',*calibration_notes])
                r.geometry_evidence=geometry_evidence
                if timing:
                    annotate_round(r, timing.timeline, anomalies, start, end)
                results.append(r)
                write_trajectory(out_dir,r,{},[],c,info.fps,
                    {'landing_X':None,'landing_B':None,'air_collision':None,'finish':None},
                    judgment_tracks=[],unassigned=[])
                continue
            launchers = [t for t in tracks if t.cls == "launcher"]
            release_options = (
                {"release_frames": {track_id: proof["frame"] for track_id, proof in
                                    releases_by_round[number].items()}}
                if number in releases_by_round else {}
            )
            analysis = analyze_launch(
                current,
                launchers,
                disp,
                c,
                th,
                info.fps,
                **({"sides_override": side_hints} if geometry is not None else {}),
                **release_options,
                **({"motion_unknown_reason": "camera_handheld"} if handheld else {}),
                **({"unobserved_frames": unobserved, "round_start": start}
                   if unobserved is not None else {}),
            )
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
                    observed_crops = {}
                    for d in t.dets:
                        raw_box = raw_by_id[t.track_id].box_at(d.frame)
                        key = (d.frame, tuple(raw_box))
                        if key in crops:
                            observed_crops[d.frame] = crops[key]
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
            # Voids judge evidence only up to the settled candidate result (R55);
            # an earlier void must not change what a later check counts.
            candidate = finish

            def relevant(frame, candidate=candidate):
                return not is_after_finish(candidate, None, frame, th)

            dynamic_notes = [
                f"launch_mask_{review['status']}@{side_hints[track_id]}:{review['frame']}"
                for track_id, review in launch_mask_reviews.get(number, {}).items()
                if review["status"] in {"disagrees", "candidate_without_box_release"}
            ]
            if geometry is not None:
                missing = [
                    f for f in range(start, end + 1)
                    if geometry.samples[f].alignment is None or geometry.samples[f].status == "bridged"
                ]
                if missing and not any(map(relevant, missing)):
                    dynamic_notes.append(AFTER_FINISH_PREFIX + "calibration_geometry_missing")
                elif missing:
                    dynamic_notes.append("calibration_geometry_missing")
                    # An unknown image cannot be treated as the top leaving the arena.
                    finish = type(finish)(
                        None,
                        None,
                        None,
                        0.0,
                        [*finish.reasons, "calibration_geometry_missing"],
                    )
                if any(
                    geometry.samples[f].segment > 0 for f in range(start, end + 1)
                ):
                    dynamic_notes.append("calibration_relocated")
                # Geometry may hide a raw unidentified opponent. Its absence in
                # normalized tracks cannot turn it into evidence for a winner.
                active_ids = {
                    t.track_id for t in round_raw_tracks
                    if t.cls == "bey" and any(start <= d.frame <= end for d in t.dets)
                }
                unanchored = sorted(active_ids - side_hints.keys())
                dynamic_notes.extend(f"identity_unanchored@{tid}" for tid in unanchored)
                if unanchored:
                    # Unidentified observations may belong to a missing opponent.
                    finish = type(finish)(
                        None, None, None, 0.0,
                        [*finish.reasons, "identity_unanchored"],
                    )
                dynamic_notes.extend(
                    n for n in phase.notes if int(n.rsplit("@", 1)[1]) in active_ids or
                    (number in physical_rounds and any(int(n.rsplit('@',1)[1]) in ids for ids in physical_rounds[number].members.values()))
                )
                if number in physical_rounds:
                    dynamic_notes.append('two_bey_tracking_requires_review')
                    releases = releases_by_round[number]
                    if (len(releases) == 2 and candidate.frame is not None
                            and candidate.frame < max(p["confirmed_at_frame"] for p in releases.values())):
                        finish = type(finish)(
                            None, None, None, 0.0,
                            [*finish.reasons, "finish_before_confirmed_launch"],
                        )
                    proof = opponent_anchors.get(number)
                    if proof is not None:
                        dynamic_notes.append("identity_inferred_from_observed_pair")
                        if candidate.frame is not None and candidate.frame < proof["frames"][-1]:
                            finish = type(finish)(
                                None, None, None, 0.0,
                                [*finish.reasons, "identity_pair_evidence_after_candidate"],
                            )
                    unknown_since = physical_rounds[number].record['identity_unknown_since']
                    if any(e['decision'] == 'recovered' for e in physical_rounds[number].record['identity_merge_episodes']):
                        dynamic_notes.append('identity_recovered_after_merge')
                    if unknown_since:
                        first_unknown = min(unknown_since.values())
                        if not relevant(first_unknown):
                            dynamic_notes.append('after_finish:identity_after_merge_unknown')
                        else:
                            dynamic_notes.append('identity_after_merge_unknown')
                            finish = type(finish)(None, None, None, 0.,
                                [*finish.reasons, 'identity_after_merge_unknown'])
                    if physical_rounds[number].record['observed_entities']<2:
                        dynamic_notes.append('two_bey_singleton_segment')
                    if physical_rounds[number].record['unassigned']:
                        dynamic_notes.append('two_bey_unassigned_observations')
                        eligible = [d['frame'] for d in physical_rounds[number].record['unassigned']
                                    if d['adjudication_eligible'] and d['reason'] != 'duplicate_box']
                        if eligible and not any(map(relevant, eligible)):
                            dynamic_notes.append(AFTER_FINISH_PREFIX + 'two_bey_unassigned_identity_unknown')
                        elif eligible:
                            dynamic_notes.append('two_bey_unassigned_identity_unknown')
                            finish=type(finish)(None,None,None,0.,[*finish.reasons,'two_bey_unassigned_identity_unknown'])
                if any(len(phase.members.get(tid, [])) > 1 for tid in active_ids):
                    dynamic_notes.append("identity_join_requires_review")
                if candidate.finish is not None and finish.finish is None:
                    # Audit trail: what a void withheld, never a published result.
                    dynamic_notes.append(
                        f"finish_candidate_voided:{candidate.finish}:{candidate.loser}:{candidate.frame}"
                    )
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
                dict.fromkeys(
                    [
                        *launch.notes,
                        *finish.reasons,
                        *stop_reasons,
                        *calibration_notes,
                        *dynamic_notes,
                    ]
                )
            )
            # These explain observed evidence, while reversals remain review-required;
            # calibration_low_score is not among them: it always requires review (F-1).
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
            r.geometry_evidence = geometry_evidence
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
            if camera_record is not None:
                r.motion_evidence["camera"] = camera_record
            if keypoint_record is not None:
                r.motion_evidence["stadium_keypoints"] = keypoint_record
            if number in releases_by_round:
                r.motion_evidence["landing_deadlines"] = analysis.deadlines
            if timing:
                stop_windows = [
                    (
                        side,
                        kind,
                        stops[side],
                        stops[side] + (th.still_frames if kind == "still" else th.reverse_frames) - 1,
                    )
                    for side, kind in stop_kinds.items()
                ]
                annotate_round(r, timing.timeline, anomalies, start, end, stop_windows)
            results.append(r)
            write_trajectory(
                out_dir,
                r,
                {
                    side: Track(
                        t.track_id,
                        t.cls,
                        [
                            d
                            for d in raw_by_id[t.track_id].dets
                            if start <= d.frame <= end
                        ],
                    )
                    for side, t in bey.items()
                },
                [
                    Track(
                        t.track_id,
                        t.cls,
                        [
                            d
                            for d in raw_by_id[t.track_id].dets
                            if start <= d.frame <= part_end
                        ],
                    )
                    for t in parts
                ],
                c,
                info.fps,
                {
                    "landing_X": launch.landing_frame["X"],
                    "landing_B": launch.landing_frame["B"],
                    "air_collision": analysis.air_collision_frame,
                    "finish": finish.frame,
                },
                judgment_tracks=selected if geometry is not None else None,
                unknown_side_since={int(k):v for k,v in physical_rounds[number].record['identity_unknown_since'].items()}
                    if number in physical_rounds else None,
                unknown_side_intervals={int(k):v for k,v in physical_rounds[number].record['identity_unknown_intervals'].items()}
                    if number in physical_rounds else None,
                unassigned=[
                    Track(
                        t.track_id,
                        t.cls,
                        [d for d in t.dets if start <= d.frame <= end],
                    )
                    for t in round_raw_tracks
                    if t.cls == "bey"
                    and any(start <= d.frame <= end for d in t.dets)
                    and t.track_id not in {v.track_id for v in bey.values()}
                ],
            )
        return _calibrated(_provenance(results, provenance), snap)
    except VideoDecodeError:
        log.exception("cannot decode %s", info.path)
        return _provenance(
            [_review(info.path.name, info.duration, "decode_error")], provenance
        )


def _load_selection(path: Path, th: Thresholds) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: no selected detector; train both detectors (`bxjudge train`) and "
            "write the selection with `bxjudge compare` first (README 完整流程)"
        )
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
    if not 0 <= th.pocket_corner_jitter_frac < 0.5:
        raise ValueError("pocket_corner_jitter_frac must be in [0, 0.5)")
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


def analysis_status(results: list[RoundResult]) -> str:
    """``videos.json`` ``analysis_status`` of one analysis attempt (ruling R68): the failure
    placeholder's reason (``recalibrate`` / ``decode_error`` / ``analysis_error``), else ``ok``
    (real rounds, or a completed analysis that found none: ``no_round``)."""
    for r in results:
        for reason in ("recalibrate", "decode_error", "analysis_error"):
            if reason in r.review_reasons:
                return reason
    return "ok"


# Bump when judge_video changes what dynamic judging hands the launch analysis, so cached
# dynamic analyses are redone (pipeline.py itself is not hashed: every edit would rejudge all).
# 1: unknown-geometry frames reach analyze_launch without --stadium-keypoints too.
_DYNAMIC_JUDGING_REVISION = 1


def _keypoint_signature(tools, camera_policy: dict) -> dict:
    """JSON-normal identity of the keypoint route for the analysis signature."""
    from dataclasses import is_dataclass

    cfg = getattr(tools, "cfg", None)
    finder = getattr(tools, "finder", None)
    reference = getattr(finder, "reference", None)
    signature = {
        "camera": camera_policy,
        "finder_config": asdict(cfg) if is_dataclass(cfg) else None,
        "weights_sha256": getattr(finder, "weights_sha256", None),
        "model_sha256": getattr(getattr(tools, "model", None), "source_sha256", None),
        "reference": asdict(reference) if is_dataclass(reference) else None,
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("camera_motion.py", "handheld_calibration.py", "stadium_locator.py",
                         "stadium_finder.py", "stadium_keypoints.py")
        },
    }
    return json.loads(json.dumps(signature))  # tuples as lists, as read back from disk


def _keypoint_preselection(out_dir: Path, recorded: str | None, vetoed: set[str],
                           info: VideoInfo, tools, autocalib: AutoCalibConfig,
                           th: Thresholds, *, reuse: bool) -> Snapshot | None:
    """The batch's current keypoint choice for a video no template matches (Codex Task 6 M2).
    A recorded, non-vetoed keypoints snapshot stays current: it is a function of the
    unchanged video and of the keypoint route, whose identity is in the analysis signature,
    so the video is not re-judged on every run; ``reuse`` says that signature is unchanged
    (otherwise the stored snapshot may no longer be what the route finds, and judge_video
    selects afresh). A vetoed recorded snapshot gets a fresh find (a veto may now have an
    alternative); anything else has no keypoint choice here."""
    if recorded is None:
        return None
    if recorded in vetoed:
        snap, _ = keypoint_snapshot(info.path, out_dir, tools, autocalib, vetoed,
                                    hwaccel=th.decode_hwaccel)
        return snap
    if not reuse:
        return None
    try:
        snap = load_snapshot(out_dir, recorded, autocalib=autocalib)
    except (FileNotFoundError, ValueError):
        return None
    return snap if snap.kind == "keypoints" else None


def needs_reanalysis(
    recorded: str | None, current: str | None, vetoed: set[str]
) -> bool:
    """Spec section 6.3b items 3-4 for an unchanged, already processed video: reanalyse when the
    calibration snapshot selected now (``current``, ``None`` = selection fails) differs from the
    one it was judged with (``recorded``) — a new manual calibration, an edited or new template,
    or a failure after a previous success.

    Ruling T4-3: a video judged with a now-vetoed snapshot is reanalysed only when selection
    yields a usable, non-vetoed snapshot (manual added, template changed, veto lifted); a veto
    alone keeps its rows and original predictions (spec 6.6b), never a recalibrate placeholder."""
    if recorded in vetoed:
        return current is not None and current not in vetoed
    return current != recorded


def _read_statuses(out_dir: Path) -> dict[str, str | None]:
    """Latest ``analysis_status`` per video from ``videos.json`` (R68); absent = None."""
    path = Path(out_dir) / "videos.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(rows, dict):
        raise ValueError(f"{path}: expected a JSON object keyed by video")  # noqa: TRY004
    return {
        v: e.get("analysis_status") if isinstance(e, dict) else None
        for v, e in rows.items()
    }


def _has_real_rows(out_dir: Path, video: str) -> bool:
    """Whether ``rounds.json`` holds at least one real (non-placeholder) row of ``video``."""
    return any(
        r["video"] == video and not is_not_a_round(r["review_reasons"])
        for r in read_rounds(out_dir)
    )


def _read_vlm_seconds(out_dir: Path) -> dict[str, float]:
    """``vlm_s`` per video from ``videos.json`` (0 when absent)."""
    path = Path(out_dir) / "videos.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    out = {}
    for video, entry in rows.items() if isinstance(rows, dict) else ():
        v = entry.get("vlm_s") if isinstance(entry, dict) else None
        out[video] = (
            float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0
        )
    return out


def _read_calibration_ids(out_dir: Path) -> dict[str, str | None]:
    """The snapshot id each video's current rows were judged with, from ``videos.json``."""
    path = Path(out_dir) / "videos.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    out: dict[str, str | None] = {}
    for video, entry in rows.items() if isinstance(rows, dict) else ():
        cal = entry.get("calibration") if isinstance(entry, dict) else None
        sid = cal.get("id") if isinstance(cal, dict) else None
        out[video] = sid if isinstance(sid, str) else None
    return out


def run_batch(
    folder: Path,
    calib_dir: Path,
    out_dir: Path,
    *,
    selection: Path = Path("weights/selected.json"),
    trt_manifest: Path | None = None,
    thresholds: Path = Path("config/thresholds.yaml"),
    rules_path: Path = Path("config/rules.yaml"),
    detection_config: Path = Path("config/detection.yaml"),
    limit: int | None = None,
    detector: Detector | None = None,
    rules: Rules | None = None,
    th: Thresholds | None = None,
    vlm_config: Path | None = None,
    autocalib_config: Path = DEFAULT_AUTOCALIB_CONFIG,
    autocalib: AutoCalibConfig | None = None,
    rejudge: bool = False,
    dynamic_config: Path | None = None,
    two_bey_config: Path | None = None,
    stadium_keypoints: bool = False,
    camera_config: Path = DEFAULT_CAMERA_MODE_CONFIG,
    launch_mask_weights: Path | None = None,
) -> None:
    # R61: batch wall-clock from here (before config/model init) to the end of the VLM phase.
    run_began = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    # C-X4(b): one id per run, on every video timing it writes and on its batch record; a
    # timing whose run never wrote its record (interrupted) cannot verify speed.
    run_id = uuid.uuid4().hex
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
    validate_output_paths(out_dir, calib_dir)
    if dynamic_config is not None:
        load_dynamic_policy(dynamic_config)
        load_launch_policy()
        load_video_policy()
    if two_bey_config is not None:
        if dynamic_config is None:
            raise ValueError('two-bey tracking requires dynamic calibration')
        load_two_bey_policy(two_bey_config)
    launch_mask_reviewer = None
    launch_mask_signature = None
    if launch_mask_weights is not None:
        if two_bey_config is None:
            raise ValueError("launch mask review requires --two-bey-tracking")
        weights = Path(launch_mask_weights).expanduser().resolve()
        with weights.open("rb") as source:
            weights_sha = hashlib.file_digest(source, "sha256").hexdigest()
        launch_mask_signature = {
            "weights_sha256": weights_sha,
            "implementation_sha256": hashlib.sha256(
                Path(__file__).with_name("launch_mask_review.py").read_bytes()
            ).hexdigest(),
        }
        launch_mask_reviewer = SamLaunchMaskReviewer(weights)
    # Plan 2026-11-12-dynamic-calibration: keypoint stadiums and camera modes are opt-in.
    camera_policy = (
        load_camera_policy(Path(camera_config).expanduser()) if stadium_keypoints else None
    )
    # Configuration and model initialization fail globally, before any result/cache writes.
    injected_thresholds = th is not None
    th = load_thresholds(thresholds) if th is None else th
    rules = load_rules(rules_path) if rules is None else rules
    _validate_thresholds(th)
    if autocalib is None:
        autocalib = load_autocalib_config(Path(autocalib_config).expanduser())
    policy = load_policy(detection_config)
    # Spec section 8: the optional VLM phase is configured (and its loopback endpoint
    # validated) up front; None or after_run: false keeps the run Ollama-free.
    vlm = load_vlm_config(vlm_config) if vlm_config is not None else None
    if vlm is not None and not vlm.after_run:
        vlm = None
    provenance = None
    source_policy = physical_policy = anomaly_policy = None
    if trt_manifest is not None and detector is not None:
        raise ValueError("trt_manifest cannot be combined with an injected detector")
    if detector is None:
        provenance = _load_selection(Path(selection).expanduser(), th)
        if not provenance["dataset"]["provenance"].get("synthetic", False):
            source_policy = load_video_sources()
            physical_policy = load_physical_time()
            anomaly_policy = AnomalyPolicy(**source_policy["anomalies"])
        provenance["runtime"] = {
            "detection_config": str(Path(detection_config).resolve()),
            "thresholds_path": None
            if injected_thresholds
            else str(Path(thresholds).resolve()),
            "rules_path": str(Path(rules_path).resolve()),
            "confidence": policy["confidence"],
            "candidate_confidence": policy["candidate_confidence"],
            "tracking": policy["tracking"],
            "device": policy["device"],
            "thresholds": asdict(th),
            "rules": asdict(rules),
            "physical_time": physical_policy,
            "video_sources": source_policy,
        }
        backend_options = {} if trt_manifest is None else {"trt_manifest": trt_manifest}
        detector = load_detector(
            provenance["kind"],
            Path(provenance["weights"]),
            conf=policy["candidate_confidence"],
            config=detection_config,
            thresholds=th,
            **backend_options,
        )
        if trt_manifest is not None:
            provenance["runtime"]["detector"] = detector.runtime
        else:
            provenance["runtime"]["detector"] = {"backend": "pytorch"}
    stadium_tools = None
    if stadium_keypoints:
        # Loaded once for the whole batch, next to the detector on the GPU (check nvidia-smi
        # first: the GPU may be shared).
        log.info("loading the stadium keypoint ensemble on device %s", policy["device"])
        stadium_tools = load_stadium_tools(device=policy["device"])
        if provenance is not None:
            provenance["runtime"]["stadium_keypoints"] = {
                "weights_sha256": getattr(stadium_tools.finder, "weights_sha256", None),
                "camera_policy": camera_policy,
            }
    # Framework imports may reset OpenCV's global thread count. Apply the requested
    # budget once after initialization and before any decoding/prefetch worker starts.
    cv2.setNumThreads(policy["opencv_threads"])
    if provenance is not None:
        provenance["runtime"]["opencv_threads"] = {
            "requested": policy["opencv_threads"],
            "actual": cv2.getNumThreads(),
        }
    dynamic_signature = None
    if dynamic_config is not None:
        dynamic_signature = {
            "recovery": load_dynamic_policy(dynamic_config),
            "follow": load_video_policy()["stadium_follow"],
            "phase": load_launch_policy(),
            "identity": load_identity_policy(),
            "temporal": load_temporal_policy(),
            "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("identity.py", "dynamic_calibration.py", "temporal_geometry.py", "stadium_follow.py",
                             "launch_check.py")},
            "judging_revision": _DYNAMIC_JUDGING_REVISION,
        }
        if two_bey_config is not None:
            dynamic_signature['two_bey_tracking']={
                'policy':load_two_bey_policy(two_bey_config),
                'implementation_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                         for name in ('two_bey_tracking.py','two_bey_integration.py')},
            }
        if launch_mask_signature is not None:
            dynamic_signature["launch_mask_review"] = launch_mask_signature
        if provenance is not None:
            provenance["dynamic_calibration"] = dynamic_signature
    if source_policy is not None:
        dynamic_signature = {
            **(dynamic_signature or {}),
            "physical_time": physical_policy,
            "video_sources": source_policy,
        }
    video_records_path = out_dir / "videos.json"
    video_records = (
        json.loads(video_records_path.read_text())
        if video_records_path.exists()
        else {}
    )
    if stadium_keypoints:
        # The keypoint route changes judgments with or without dynamic geometry: its
        # policy, model, weights, reference and code are part of the analysis signature.
        dynamic_signature = {
            **(dynamic_signature or {}),
            "stadium_keypoints": _keypoint_signature(stadium_tools, camera_policy),
        }
    old_analysis_policies = {
        name: entry.get("analysis_policy") for name, entry in video_records.items()
    }
    db = out_dir / "processed_videos.json"
    failures = []

    def on_error(path, error):
        failures.append((path, error))

    infos = scan(folder, db, on_error=on_error)
    # Spec section 6.3b: an unchanged processed video is reanalysed when its calibration
    # changed. Selection needs only the first second, so it runs for every processed video
    # (right before that video would be judged); a reanalysed video is judged with this same
    # selection (not selected twice). Rulings T4-1/T4-3: a vetoed auto snapshot is never
    # selected again; a video recorded with one keeps its rows until selection yields a
    # non-vetoed snapshot (then it is reanalysed). Ruling R66b: ``rejudge`` ignores the
    # processed cache and re-analyses every video (reviewed rows are kept, R36), except that
    # ruling C-X3(a) keeps the T4-3 rule: a video recorded with a vetoed snapshot and no
    # usable non-vetoed alternative keeps its rows, exactly as on a plain rerun.
    vetoed = vetoed_ids(out_dir)
    processed = processed_videos(folder, db, on_error=on_error)
    statuses = _read_statuses(out_dir)
    recorded_calibrations = _read_calibration_ids(out_dir)
    # Ruling R68: "previously judged" = at least one real (non-placeholder) row on disk.
    judged_before = {
        r["video"]
        for r in read_rounds(out_dir)
        if not is_not_a_round(r["review_reasons"])
    }
    candidates = sorted(
        [(i.path, i, False) for i in infos]
        + [(p, None, False) for p, _ in failures]
        + [(i.path, i, True) for i in processed]
    )
    entries: list[tuple[Path, VideoInfo | None]] = []
    # C-X7(a): the work this run measured per video, stored in its batch record so a later
    # retry never rewrites this run's overhead.
    work: dict[str, float] = {}
    # C-X8: per listed video, whether it had real rows after this run's attempt (the run's
    # own speed scope), and the videos only kept (never judged) by the T4-3 veto rule.
    real: dict[str, bool] = {}
    kept_only: set[str] = set()
    kept_after_judging: set[str] = set()  # judged, rows kept (veto rule): no VLM pass
    for path, info, cached in candidates:
        # C-X9: veto-kept videos (no judging) never use up --limit; once it is reached, only
        # veto candidates still get their attempt recorded (nothing else is judged).
        limit_reached = (
            limit is not None
            and len([p for p, _ in entries if p.name not in kept_only]) >= limit
        )
        # Ruling R66a: ONE timer per video, from its first decode (the selection pass of a
        # processed video, else judge_video's own) to its result write; selection and
        # judging run back to back, so no other video's work falls inside it.
        began = time.perf_counter()
        preselected: dict = {}
        # C-X5(a): the snapshot the video's current rows were judged with: its processed
        # record, else (a changed file whose record was forgotten) videos.json.
        recorded = None
        if info is not None:
            recorded = processed_calibration(
                db, info.path
            ) or recorded_calibrations.get(path.name)
        # A previously judged video recorded with a vetoed snapshot goes through the same
        # selection even when scan reports it as new/changed (e.g. a touched mtime), so the
        # T4-3 keep rule below applies on every path.
        vetoed_before = (
            info is not None and path.name in judged_before and recorded in vetoed
        )
        if limit_reached and not vetoed_before:
            continue
        if cached or vetoed_before:
            try:
                frame = first_second_frame(info, th)
                snap = select_snapshot(
                    frame, calib_dir, out_dir, th, autocalib, vetoed, video=info.path.name
                )
                if snap is None and stadium_tools is not None:
                    snap = _keypoint_preselection(
                        out_dir, recorded, vetoed, info, stadium_tools, autocalib, th,
                        reuse=old_analysis_policies.get(path.name) == dynamic_signature,
                    )
            except Exception as error:
                if recorded in vetoed and path.name in judged_before:
                    # C-X7(b): a vetoed, previously judged video whose selection failed is
                    # never reanalysed (with no usable alternative that could only give a
                    # recalibrate placeholder, T4-3): its rows stay and the attempt is a
                    # transient failure (R68), retried on the next run.
                    status = (
                        "decode_error"
                        if isinstance(error, VideoDecodeError)
                        else "analysis_error"
                    )
                    log.exception(
                        "%s: calibration selection failed (%s); keeping its "
                        "vetoed rows",
                        path.name,
                        status,
                    )
                    entries.append((path, info))
                    kept_only.add(path.name)  # C-X9: kept, not judged: no --limit use
                    wall = time.perf_counter() - began
                    record_video_status(
                        out_dir,
                        path.name,
                        status,
                        wall_s=wall,
                        duration_s=info.duration,
                        run_id=run_id,
                    )
                    work[path.name], real[path.name] = wall, True
                    continue
                # e.g. an unreadable first second: the normal per-video flow handles it.
                log.exception(
                    "cannot re-select the calibration of %s; reanalysing it", info.path
                )
            else:
                current = snap.id if snap is not None else None
                unchanged = not needs_reanalysis(recorded, current, vetoed)
                # C-X4(a)/C-X5(a): the T4-3 veto rule comes first. A video recorded with a
                # vetoed snapshot and no usable non-vetoed alternative keeps its rows on any
                # rerun (plain or --rejudge, C-X3(a)), whatever its latest analysis_status and
                # whether it came from the cache or from scan: judging it again could only
                # give a recalibrate placeholder.
                if recorded in vetoed and unchanged:
                    # C-X8: still an attempt: its own wall under this run's id and the
                    # CURRENT probed duration are recorded (status and judged wall kept), and
                    # the run lists it with what it measured.
                    entries.append((path, info))
                    kept_only.add(path.name)
                    wall = time.perf_counter() - began
                    record_video_attempt(
                        out_dir,
                        path.name,
                        wall_s=wall,
                        duration_s=info.duration,
                        run_id=run_id,
                    )
                    work[path.name], real[path.name] = wall, True
                    continue
                if limit_reached:  # C-X9: --limit is used up; nothing more is judged
                    continue
                # C-X2(a): otherwise only a latest status of ``ok`` lets an unchanged cached
                # video keep its rows unjudged (never under --rejudge, R66b); any other status
                # (a transient failure, R68, or none recorded) is re-judged.
                if (
                    cached
                    and not rejudge
                    and old_analysis_policies.get(path.name) == dynamic_signature
                    and statuses.get(path.name) == "ok"
                    and unchanged
                ):
                    continue
                preselected = {"snapshot": snap}
        entries.append((path, info))
        chosen: list[Snapshot | None] = []
        if info is None:
            results = _provenance([_review(path.name, 0.0, "decode_error")], provenance)
        else:
            timing_options = {}
            if source_policy is not None:
                # Reject missing or conflicting capture metadata before judging.
                timeline = probe_timeline(info.path, source_policy)
                timing_options = {
                    "timing": JudgmentTime(timeline, **physical_policy),
                    "anomaly_policy": anomaly_policy,
                }
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
                    autocalib=autocalib,
                    on_snapshot=chosen.append,
                    vetoed=vetoed,
                    **(
                        {"dynamic_config": dynamic_config}
                        if dynamic_config is not None
                        else {}
                    ),
                    **preselected,
                    **timing_options,
                    **(
                        {"stadium_tools": stadium_tools, "camera_policy": camera_policy}
                        if stadium_keypoints
                        else {}
                    ),
                    **({'two_bey_config':two_bey_config} if two_bey_config is not None else {}),
                    **({"launch_mask_reviewer": launch_mask_reviewer}
                       if launch_mask_reviewer is not None else {}),
                )
            except Exception:
                log.exception("analysis failed for %s; continuing batch", path)
                results = _provenance(
                    [_review(path.name, info.duration, "analysis_error")], provenance
                )
        snap = chosen[-1] if chosen else None
        status = analysis_status(results)
        if vetoed_before and status == "recalibrate":
            # T4-3: a veto alone never turns judged rows into a recalibrate placeholder, also
            # when the alternative found for it (a keypoint stadium) still gives none (e.g. a
            # handheld camera without dynamic calibration). Rows, calibration and judged wall
            # stay; this judged attempt counts toward --limit (C-X9) and its recalibrate is
            # the latest analysis status (R68), but the VLM has nothing new to look at.
            log.warning("%s: the alternative to its vetoed calibration gives no round "
                        "(%s); keeping its vetoed rows", path.name,
                        "、".join(results[0].review_reasons) if results else "")
            kept_after_judging.add(path.name)
            wall = time.perf_counter() - began
            record_video_status(out_dir, path.name, status, wall_s=wall,
                                duration_s=info.duration, run_id=run_id)
            work[path.name], real[path.name] = wall, True
            continue
        if status in TRANSIENT_FAILURES and path.name in judged_before:
            # Ruling R68: a transient failure (decode, probe, analysis) of a previously judged
            # video is no replacement analysis. Its rows (predictions, audit flags, calibration
            # ids; vetoed ones too, T4-3), its judged_wall_s, processing_s, VLM time and
            # calibration, and its processed record stay as they are. With the record left as
            # it was (or still mismatching the changed file), the next run retries the video.
            # R66a: the attempt's own wall (attempt_wall_s) and the duration of the CURRENT
            # source file (its latest probe; none when the probe failed) are recorded with
            # the status, so neither the kept rows nor an old file's length can hide it.
            log.warning(
                "%s: %s on reanalysis; keeping its existing rows", path.name, status
            )
            wall = time.perf_counter() - began
            record_video_status(
                out_dir,
                path.name,
                status,
                wall_s=wall,
                duration_s=info.duration if info is not None else None,
                run_id=run_id,
            )
            work[path.name], real[path.name] = wall, True  # its rows are kept
            continue
        written = write_results(results, out_dir, rules)
        processing = time.perf_counter() - began
        record_video_timing(
            out_dir,
            path.name,
            info.duration if info else 0.0,
            processing,
            source_video=path.resolve(),
            wall_s=processing,
            run_id=run_id,
            produced_rounds={n for v, n in written if v == path.name},
        )
        work[path.name] = processing
        real[path.name] = _has_real_rows(out_dir, path.name)
        record_video_calibration(out_dir, path.name, snap)
        record_video_status(out_dir, path.name, status)
        if status == "ok":
            record_video_analysis_policy(out_dir, path.name, dynamic_signature)
        # Placeholder-only videos await corrected input and are retried (R56): their record
        # is removed, never left stale with an earlier success's id (ruling T4-5).
        if all(is_not_a_round(r.review_reasons) for r in results):
            forget_processed(db, path)
        elif info is not None:
            mark_processed(
                info, db, calibration_id=snap.id if snap is not None else None
            )
        log.info(
            "%s: processing %.3fs / playback %.3fs",
            path.name,
            processing,
            info.duration if info else 0.0,
        )
    vlm_wall_s = None
    if vlm is not None and {path.name for path, _ in entries} - kept_only - kept_after_judging:
        # The detector must be gone before the ~18-20 GB VLM loads (spec section 5).
        vlm_began = time.perf_counter()
        del detector
        stadium_tools = None  # the keypoint ensemble leaves the GPU with the detector
        _release_gpu_memory()
        vlm_before = _read_vlm_seconds(out_dir)
        summary = review_results_dir(
            out_dir,
            th=th,
            rules=rules,
            config=vlm,
            # veto-kept videos were not judged in this run: their rows are left alone
            videos={path.name for path, _ in entries} - kept_only - kept_after_judging,
            autocalib=autocalib,  # C-X3(b): run's --autocalib, never the cwd file
            run_id=run_id,
        )
        # R65: an unreachable server skips the phase (bounded preflight, still inside the
        # honest batch wall_s) and records no VLM phase.
        if not (summary or {}).get("skipped"):
            vlm_wall_s = time.perf_counter() - vlm_began
        # C-X7(a): this phase's VLM time per video belongs to this run's measured work.
        for video, seconds in _read_vlm_seconds(out_dir).items():
            added = seconds - vlm_before.get(video, 0.0)
            if video in work and added > 0:
                work[video] += added
    if entries:  # nothing judged = no judgment wall-clock to report
        record_run_timing(
            out_dir,
            kind="run",
            run_id=run_id,
            started_at=started_at,
            wall_s=time.perf_counter() - run_began,
            vlm_wall_s=vlm_wall_s,
            videos={
                path.name: info.duration if info else 0.0 for path, info in entries
            },
            work={path.name: work.get(path.name, 0.0) for path, _ in entries},
            real={path.name: real.get(path.name, False) for path, _ in entries},
        )


def _release_gpu_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
