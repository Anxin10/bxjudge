"""Confirmed reference-anchored geometry; failures are unknown, never held transforms."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .autocalib import (
    Alignment,
    _decompose,
    _stadium_centroid,
    mean_boundary_error,
    warp_calibration,
)
from .config import config_loader
from .geometry import zone_of
from .stadium_follow import StadiumFollower
from .temporal_geometry import TemporalRimTracker, _points, load_temporal_policy
from .types import Calibration, Detection, Track

DEFAULT_DYNAMIC_CONFIG = (
    Path(__file__).resolve().parents[2] / "config/dynamic_calibration.yaml"
)


@config_loader
def load_dynamic_policy(path=DEFAULT_DYNAMIC_CONFIG):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "reject_s",
        "retry_s",
        "confirmations",
        "resume_frames",
        "consistent_error_px",
        "max_shift_frac",
        "max_rotation_deg",
        "min_scale",
        "max_scale",
        "min_score",
    }
    defaults = {
        "conditional_gap_enabled": False,
        "conditional_gap_max_s": 0.80,
        "conditional_gap_max_error_px": 4.0,
        "conditional_gap_min_pocket_margin_px": 12.0,
    }
    if not isinstance(raw, dict) or not required.issubset(raw) or not set(raw).issubset(required | defaults.keys()):
        raise ValueError("incomplete dynamic_calibration policy")
    merged = {**defaults, **raw}
    if type(merged["conditional_gap_enabled"]) is not bool:
        raise ValueError("dynamic_calibration.conditional_gap_enabled must be bool")
    for k, v in merged.items():
        if k == "conditional_gap_enabled":
            continue
        if k in {"confirmations", "resume_frames"}:
            if type(v) is not int or v < 2:
                raise ValueError(f"dynamic_calibration.{k} must be integer >=2")
        elif type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"dynamic_calibration.{k} must be positive finite")
    if (
        merged["min_score"] > 1
        or merged["max_shift_frac"] > 1
        or merged["max_rotation_deg"] >= 90
        or not merged["min_scale"] <= 1 <= merged["max_scale"]
        or merged["conditional_gap_max_s"] > 2.0
        or merged["conditional_gap_max_error_px"] > 20.0
    ):
        raise ValueError("dynamic_calibration policy outside bounds")
    return merged


@dataclass(frozen=True)
class GeometrySample:
    frame: int
    alignment: Alignment | None
    segment: int
    status: str


class DynamicCalibration:
    """Small-motion tracking and stable re-localization against the original snapshot.

    A new anchor needs several spaced, consistent independently validated registrations.
    Relocation never backfills its rejected confirmation prefix. Source-observed
    flow chains gain geometry only after template closure; bridges stay ineligible.

    With a ``keypoint_locator`` (``stadium_locator.KeypointLocator``), a relocation attempt
    whose rim match against the reference fails tries one keypoint fit; the candidate must
    lie within the same broad bounds relative to the original calibration and goes through
    the same ``confirmations``. Without one, behaviour is unchanged.
    """

    def __init__(self, c: Calibration, reference, fps, follow_policy, policy, *,
                 temporal_policy=None, keypoint_locator=None, fixed_camera=False):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive finite")
        self.c = c
        self.reference = reference
        self.fps = fps
        self.policy = policy
        self.follow_policy = follow_policy
        self.fixed_camera = bool(fixed_camera)
        self.check_stride = max(1, round(fps / 15)) if self.fixed_camera else 1
        self._last_good_alignment = None
        self._last_good_frame = -math.inf
        self._blur_grace_s = 0.5
        self.anchor = StadiumFollower(c, reference, follow_policy)
        broad = {
            **follow_policy,
            **{
                k: policy[k]
                for k in (
                    "max_shift_frac",
                    "max_rotation_deg",
                    "min_scale",
                    "max_scale",
                )
            },
        }
        broad["min_edge_score"] = max(
            follow_policy["min_edge_score"], policy["min_score"]
        )
        broad["min_region_score"] = max(
            follow_policy["min_region_score"], policy["min_score"]
        )
        self.relocator = StadiumFollower(c, reference, broad)
        self.anchor_h = np.eye(3)
        self.previous = -1
        self.rejected_at = None
        self.last_retry = -math.inf
        self.last_keypoint_retry = -math.inf
        self.pending = []
        self.good_run = 0
        self.lost = False
        self.segments = [
            {
                "id": 0,
                "start_frame": 0,
                "kind": "recorded_snapshot",
                "homography": np.eye(3).tolist(),
                "confirmed_frames": [],
            }
        ]
        self.samples = {}
        self.diagnostics = {}
        self.temporal = TemporalRimTracker(
            c, reference, fps, follow_policy,
            load_temporal_policy() if temporal_policy is None else temporal_policy,
        )
        self.foreground_boxes = []
        self._current_bey_boxes = []
        self._bey_boxes_by_frame = {}
        self.last_verified = None
        self.keypoint_locator = keypoint_locator
        self._keypoint_frames = set()  # pending candidates that came from keypoints

    def _compose(self, a):
        h = np.array(a.homography) @ self.anchor_h
        h /= h[2, 2]
        centre = np.array([*_stadium_centroid(self.c), 1.0])
        moved = h @ centre
        shift = (
            float(np.linalg.norm(moved[:2] / moved[2] - centre[:2]))
            / self.c.frame_size[0]
        )
        rotation, scale, small, large, perspective = _decompose(h, centre[:2])
        if (
            not np.isfinite(h).all()
            or shift > self.policy["max_shift_frac"]
            or abs(rotation) > self.policy["max_rotation_deg"]
            or small < self.policy["min_scale"]
            or large > self.policy["max_scale"]
        ):
            return None
        return Alignment(
            tuple(tuple(float(v) for v in r) for r in h),
            a.score,
            shift,
            rotation,
            scale,
            perspective,
        )

    def set_foreground(self, detections):
        self.foreground_boxes = [d.box for d in detections]
        self._current_bey_boxes = [
            d.box for d in detections if getattr(d, "cls", None) == "bey"
        ]

    def finish(self):
        """End of the video: nothing to check in fixed mode unless conditional_gap_enabled."""
        if not self.fixed_camera or not self.policy.get("conditional_gap_enabled", False):
            return
        frames = sorted(self.samples)
        if not frames:
            return
        pockets = [
            np.asarray(poly, np.float32).reshape(-1, 1, 2)
            for poly in (self.c.xtreme, self.c.over_left, self.c.over_right)
            if poly
        ]
        max_s = self.policy.get("conditional_gap_max_s", 0.80)
        max_err = self.policy.get("conditional_gap_max_error_px", 4.0)
        min_pocket = self.policy.get("conditional_gap_min_pocket_margin_px", 12.0)
        spans = []
        cur = None
        for f in frames:
            s = self.samples[f]
            if s.alignment is None or s.status == "bridged":
                if cur is None or f != cur[1] + 1:
                    if cur is not None:
                        spans.append(cur)
                    cur = [f, f]
                else:
                    cur[1] = f
            elif cur is not None:
                spans.append(cur)
                cur = None
        if cur is not None:
            spans.append(cur)
        for s0, s1 in spans:
            if (s1 - s0 + 1) / self.fps > max_s:
                continue
            before = self.samples.get(s0 - 1)
            after = self.samples.get(s1 + 1)
            if (
                before is None
                or after is None
                or before.alignment is None
                or after.alignment is None
                or before.status in ("bridged", "conditional_gap")
                or after.status in ("bridged", "conditional_gap")
            ):
                continue
            if any(
                self.diagnostics.get(f, {}).get("reason") == "temporal_template_disagreement"
                for f in range(s0, s1 + 1)
            ):
                continue
            cb = warp_calibration(self.c, before.alignment, self.c.frame_size, self.c.name)
            ca = warp_calibration(self.c, after.alignment, self.c.frame_size, self.c.name)
            mbe = mean_boundary_error(cb, ca, max(self.c.frame_size))
            if not math.isfinite(mbe) or mbe > max_err:
                continue
            hb = np.array(before.alignment.homography, dtype=np.float64)
            ha = np.array(after.alignment.homography, dtype=np.float64)
            try:
                hb_inv = np.linalg.inv(hb)
                ha_inv = np.linalg.inv(ha)
            except np.linalg.LinAlgError:
                continue
            req_margin = max(min_pocket, 3.0 * mbe)
            total_beys = 0
            interior_safe = True
            for f in range(s0, s1 + 1):
                for x0, y0, x1, y1 in self._bey_boxes_by_frame.get(f, []):
                    total_beys += 1
                    pt_src = np.array([[[(x0 + x1) / 2.0, (y0 + y1) / 2.0]]], dtype=np.float64)
                    for h_inv in (hb_inv, ha_inv):
                        pt = cv2.perspectiveTransform(pt_src, h_inv)[0, 0]
                        px, py = float(pt[0]), float(pt[1])
                        if zone_of((px, py), self.c) not in ("battle_area", "lid"):
                            interior_safe = False
                            break
                        if any(-cv2.pointPolygonTest(pk, (px, py), True) < req_margin for pk in pockets):
                            interior_safe = False
                            break
                    if not interior_safe:
                        break
                if not interior_safe:
                    break
            if not interior_safe or total_beys == 0:
                continue
            min_score = min(before.alignment.score, after.alignment.score)
            span_denom = float(after.frame - before.frame)
            candidates = {}
            for f in range(s0, s1 + 1):
                t = (f - before.frame) / span_denom
                interp = (1.0 - t) * hb + t * ha
                raw_al = self.temporal.alignment(interp, min_score)
                orig_anchor = self.anchor_h
                self.anchor_h = np.eye(3)
                composed = self._compose(raw_al)
                self.anchor_h = orig_anchor
                if composed is None:
                    candidates.clear()
                    break
                candidates[f] = composed
            for f, composed in candidates.items():
                self.samples[f] = GeometrySample(f, composed, after.segment, "conditional_gap")
                self.diagnostics[f]["conditional_gap"] = {
                    "endpoints": [before.frame, after.frame],
                    "mean_boundary_error_px": round(float(mbe), 4),
                }
                self.diagnostics[f]["reason"] = None

    def _keypoint_candidate(self, image, diagnostic):
        """One keypoint fit as a relocation candidate within the broad global bounds."""
        located = self.keypoint_locator.locate(image)
        if located is None:
            diagnostic["relocation"]["keypoints"] = self.keypoint_locator.last_reason
            return None
        original = self.anchor_h
        self.anchor_h = np.eye(3)
        candidate = self._compose(located.alignment)
        self.anchor_h = original
        diagnostic["relocation"]["keypoints"] = (
            "located" if candidate is not None else "keypoints_out_of_bounds"
        )
        return candidate

    def _temporal_sample(self, frame, a, segment):
        temporal = self.temporal
        if a is None:
            return a, None
        matrix = np.array(a.homography)
        confirmed = temporal.confirm(frame, matrix, segment)
        if temporal.last_diagnostic["reason"] == "temporal_template_disagreement":
            return None, "temporal_template_disagreement"
        for f in confirmed:
            if (
                f == frame
                or f not in self.samples
                or self.samples[f].alignment is not None
            ):
                continue
            candidate = temporal.alignment(temporal.pending[f], a.score)
            # Same global bounds as direct registration, never unlimited chain drift.
            original = self.anchor_h
            self.anchor_h = np.eye(3)
            candidate = self._compose(candidate)
            self.anchor_h = original
            if candidate is not None:
                self.samples[f] = GeometrySample(
                    f, candidate, segment, "temporal_verified"
                )
                self.diagnostics[f]["reason"] = None
                self.diagnostics[f]["closure_frame"] = frame
                self.diagnostics[f]["closure_error_px"] = temporal.last_diagnostic[
                    "closure_error_px"
                ]
        if self.last_verified is not None:
            before = self.samples[self.last_verified]
            gap = frame - before.frame - 1
            if (
                gap > 0
                and (gap + 1) / self.fps <= temporal.cfg["bridge_s"]
                and before.segment == segment
            ):
                h0 = np.array(before.alignment.homography)
                delta = np.linalg.norm(
                    _points(matrix, self.c.stadium) - _points(h0, self.c.stadium),
                    axis=1,
                ).max()
                if (
                    delta / (gap + 1) / self.c.frame_size[0]
                    <= temporal.cfg["step_shift_frac"]
                ):
                    for f in range(before.frame + 1, frame):
                        # An explicit chain/template contradiction stays a plain rejection.
                        if (
                            f in self.samples
                            and self.samples[f].alignment is None
                            and self.diagnostics[f].get("reason")
                            != "temporal_template_disagreement"
                        ):
                            t = (f - before.frame) / (frame - before.frame)
                            interpolated = (1 - t) * h0 + t * matrix
                            self.samples[f] = GeometrySample(
                                f,
                                temporal.alignment(
                                    interpolated, min(before.alignment.score, a.score)
                                ),
                                segment,
                                "bridged",
                            )
                            self.diagnostics[f]["bridge_reason"] = (
                                "bounded_bridge_not_judging_evidence"
                            )
                            self.diagnostics[f]["bridge_endpoints"] = [
                                before.frame,
                                frame,
                            ]
        self.last_verified = frame
        if (
            temporal.h is None
            or temporal.segment != segment
            or (frame - temporal.anchor_frame) / self.fps >= temporal.cfg["reanchor_s"]
        ):
            temporal.anchor(frame, matrix, segment)
        return a, None

    def update(self, frame, image):
        if type(frame) is not int or frame < 0 or frame <= self.previous:
            raise ValueError("source frames must increase")
        if (image.shape[1], image.shape[0]) != self.c.frame_size:
            raise ValueError("geometry frame size mismatch")
        self.previous = frame
        self._bey_boxes_by_frame[frame] = list(self._current_bey_boxes)
        if (
            self.fixed_camera
            and not self.lost
            and self._last_good_alignment is not None
            and frame % self.check_stride != 0
        ):
            sample = GeometrySample(
                frame, self._last_good_alignment, len(self.segments) - 1, "tracking"
            )
            self.samples[frame] = sample
            self.diagnostics[frame] = {
                "tracking": {"reason": None, "strided_reuse": True},
                "relocation": None,
                "reason": None,
            }
            return sample
        proposal = self.temporal.observe(frame, image, self.foreground_boxes)
        for follower in (self.anchor, self.relocator):
            follower.set_foreground_mask(
                self.temporal.foreground_mask, self.temporal.cfg["min_visible_fraction"]
            )
        local = self.anchor.align(image)
        a = None if local is None else self._compose(local)
        diagnostic = {
            "tracking": dict(self.anchor.last_diagnostic),
            "relocation": None,
            "reason": self.anchor.last_diagnostic["reason"],
        }
        if local is not None and a is None:
            diagnostic["reason"] = "global_transform_out_of_bounds"
        status = "tracking"
        _SOFT_BLUR_REASONS = (
            "distinct_matches_insufficient",
            "final_inliers_insufficient",
            "edge_score_below_minimum",
            "ecc_nonconvergent",
        )
        if (
            a is None
            and self.fixed_camera
            and not self.lost
            and self._last_good_alignment is not None
            and diagnostic["reason"] in _SOFT_BLUR_REASONS
            and (frame - self._last_good_frame) / self.fps <= self._blur_grace_s
        ):
            sample = GeometrySample(
                frame, self._last_good_alignment, len(self.segments) - 1, "tracking"
            )
            self.samples[frame] = sample
            diagnostic["tracking"]["soft_blur_reuse"] = diagnostic["reason"]
            diagnostic["reason"] = None
            self.diagnostics[frame] = diagnostic
            return sample
        if a is not None:
            self.pending.clear()
            self.good_run += 1
            if self.lost and self.good_run < self.policy["resume_frames"]:
                a = None
                status = "confirming"
                diagnostic["reason"] = "recovery_confirmation_pending"
            else:
                self.lost = False
                self.rejected_at = None
                self._last_good_alignment = a
                self._last_good_frame = frame
        else:
            self._last_good_alignment = None
            self.good_run = 0
            self.lost = True
            if self.rejected_at is None:
                self.rejected_at = frame
            status = "lost"
            if (frame - self.rejected_at) / self.fps >= self.policy["reject_s"] and (
                frame - self.last_retry
            ) / self.fps >= self.policy["retry_s"]:
                self.last_retry = frame
                candidate = self.relocator.align(image)
                diagnostic["relocation"] = dict(self.relocator.last_diagnostic)
                keypoint_skipped_cooldown = False
                if candidate is None and self.keypoint_locator is not None:
                    if (
                        frame - self.last_keypoint_retry
                    ) / self.fps >= max(self.policy["retry_s"], 0.5):
                        self.last_keypoint_retry = frame
                        candidate = self._keypoint_candidate(image, diagnostic)
                        if candidate is not None:
                            self._keypoint_frames.add(frame)
                    else:
                        keypoint_skipped_cooldown = True
                if candidate is None:
                    if not keypoint_skipped_cooldown:
                        self.pending.clear()
                else:
                    if self.pending:
                        old = warp_calibration(
                            self.c, self.pending[-1][1], self.c.frame_size, self.c.name
                        )
                        new = warp_calibration(
                            self.c, candidate, self.c.frame_size, self.c.name
                        )
                        if (
                            mean_boundary_error(old, new, max(self.c.frame_size))
                            > self.policy["consistent_error_px"]
                        ):
                            self.pending.clear()
                    self.pending.append((frame, candidate))
                    status = "candidate"
                    diagnostic["reason"] = "relocation_confirmation_pending"
                    if len(self.pending) >= self.policy["confirmations"]:
                        # This frame alone starts the segment. Earlier failed frames stay unknown.
                        moved = warp_calibration(
                            self.c, candidate, self.c.frame_size, self.c.name
                        )
                        self.anchor = StadiumFollower(
                            moved, image.copy(), self.follow_policy
                        )
                        self.anchor_h = np.array(candidate.homography)
                        self.segments.append(
                            {
                                "id": len(self.segments),
                                "start_frame": frame,
                                "kind": (
                                    "relocated_keypoints"
                                    if any(f in self._keypoint_frames for f, _ in self.pending)
                                    else "relocated"
                                ),
                                "homography": self.anchor_h.tolist(),
                                "confirmed_frames": [f for f, _ in self.pending],
                                "score": candidate.score,
                            }
                        )
                        self.pending.clear()
                        self._keypoint_frames.clear()
                        self.rejected_at = None
                        self.lost = False
                        self.good_run = 0
                        a = candidate
                        status = "relocated"
                        diagnostic["reason"] = None
        a, temporal_reason = self._temporal_sample(frame, a, len(self.segments) - 1)
        diagnostic["temporal"] = dict(self.temporal.last_diagnostic)
        if temporal_reason is not None:
            diagnostic["reason"] = temporal_reason
            status = "lost"
        if a is None and proposal is not None and temporal_reason is None:
            status = "temporal_pending"
        sample = GeometrySample(frame, a, len(self.segments) - 1, status)
        self.samples[frame] = sample
        self.diagnostics[frame] = diagnostic
        return sample

    def record(self):
        return {
            "schema": 1,
            "reference_size": list(self.c.frame_size),
            "coordinate_mapping": "reference_to_source",
            "policy": self.policy,
            "temporal_policy": self.temporal.cfg,
            "segments": self.segments,
            "samples": [
                asdict(s)
                | {
                    "rejection_reason": self.diagnostics[s.frame]["reason"],
                    "diagnostics": self.diagnostics[s.frame],
                }
                for s in self.samples.values()
            ],
        }


def normalize_tracks(tracks, samples):
    """Inverse-transform each actual box; remove unknown geometry, never interpolate.

    Preserve the transformed actual centre. Axis-aligned envelopes around that centre
    bound the rotated body conservatively; raw image boxes are retained separately.
    """
    out = []
    for t in tracks:
        dets = []
        for d in t.dets:
            sample = samples.get(d.frame)
            if sample is None or sample.alignment is None or sample.status == "bridged":
                continue
            inverse = np.linalg.inv(np.array(sample.alignment.homography))
            x0, y0, x1, y1 = d.box
            pts = np.array(
                [
                    (x0, y0),
                    (x1, y0),
                    (x1, y1),
                    (x0, y1),
                    ((x0 + x1) / 2, (y0 + y1) / 2),
                ],
                np.float64,
            )
            mapped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), inverse).reshape(
                -1, 2
            )
            centre = mapped[-1]
            extent = np.max(np.abs(mapped[:-1] - centre), axis=0)
            box = (*map(float, centre - extent), *map(float, centre + extent))
            dets.append(Detection(d.frame, d.cls, box, d.score))
        out.append(Track(t.track_id, t.cls, dets))
    return out


def save_geometry(out_dir, video, record):
    """Content-addressed immutable evidence: old reviewed rows keep their exact file."""
    import hashlib
    import json

    raw = json.dumps(record, sort_keys=True, allow_nan=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    relative = Path("geometry") / f"{Path(video).stem}-{digest}.json"
    target = Path(out_dir) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        pending = target.with_suffix(".json.tmp")
        pending.write_bytes(raw)
        pending.replace(target)
    return {
        "path": relative.as_posix(),
        "sha256": digest,
        "coordinates": "reference_pixels",
    }


def read_geometry(out_dir, row, source_sha):
    """Validate immutable row evidence; never load current calibration or recompute it."""
    import hashlib
    import json

    evidence = row["geometry_evidence"]
    path = (Path(out_dir) / evidence["path"]).resolve()
    if not path.is_relative_to(Path(out_dir).resolve() / "geometry"):
        raise ValueError("geometry evidence path outside result geometry folder")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != evidence["sha256"]:
        raise ValueError("geometry evidence SHA256 mismatch")
    data = json.loads(raw)
    if data["schema"] != 1 or data["coordinate_mapping"] != "reference_to_source":
        raise ValueError("invalid geometry evidence schema")
    if (
        data["source_sha256"] != source_sha
        or data["calibration_id"] != row["calibration_id"]
    ):
        raise ValueError("geometry evidence source or calibration mismatch")
    size = data["reference_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(v) is not int or v <= 0 for v in size)
    ):
        raise ValueError("invalid geometry reference size")
    samples = {}
    previous = -1
    for sample in data["samples"]:
        frame = sample["frame"]
        if type(frame) is not int or frame <= previous:
            raise ValueError("invalid geometry source frame order")
        previous = frame
        if sample["alignment"] is not None:
            matrix = np.array(sample["alignment"]["homography"], dtype=float)
            if (
                matrix.shape != (3, 3)
                or not np.isfinite(matrix).all()
                or abs(np.linalg.det(matrix)) < 1e-12
            ):
                raise ValueError("invalid geometry homography")
        samples[frame] = sample
    return {**data, "samples": samples}


def recorded_calibration(c, record, frame, size):
    sample = record["samples"].get(frame)
    if sample is None or sample["alignment"] is None or sample["status"] == "bridged":
        return None
    original = record["reference_size"]
    sf = np.diag([size[0] / original[0], size[1] / original[1], 1.0])
    from .calibration import scale_calibration

    scaled = scale_calibration(c, size)
    matrix = sf @ np.array(sample["alignment"]["homography"]) @ np.linalg.inv(sf)
    a = Alignment(
        **{
            **sample["alignment"],
            "homography": tuple(tuple(float(v) for v in row) for row in matrix),
        }
    )
    return warp_calibration(scaled, a, size, c.name)
