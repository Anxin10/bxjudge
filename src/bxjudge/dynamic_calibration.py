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
from .stadium_follow import StadiumFollower
from .types import Calibration, Detection, Track

DEFAULT_DYNAMIC_CONFIG = (
    Path(__file__).resolve().parents[2] / "config/dynamic_calibration.yaml"
)


@config_loader
def load_dynamic_policy(path=DEFAULT_DYNAMIC_CONFIG):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = {
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
    if not isinstance(raw, dict) or set(raw) != names:
        raise ValueError("incomplete dynamic_calibration policy")
    for k, v in raw.items():
        if k in {"confirmations", "resume_frames"}:
            if type(v) is not int or v < 2:
                raise ValueError(f"dynamic_calibration.{k} must be integer >=2")
        elif type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"dynamic_calibration.{k} must be positive finite")
    if (
        raw["min_score"] > 1
        or raw["max_shift_frac"] > 1
        or raw["max_rotation_deg"] >= 90
        or not raw["min_scale"] <= 1 <= raw["max_scale"]
    ):
        raise ValueError("dynamic_calibration policy outside bounds")
    return dict(raw)


@dataclass(frozen=True)
class GeometrySample:
    frame: int
    alignment: Alignment | None
    segment: int
    status: str


class DynamicCalibration:
    """Small-motion tracking and stable re-localization against the original snapshot.

    A new anchor needs several spaced, consistent independently validated registrations.
    No rejected frame or confirmation prefix gains geometry retroactively.
    """

    def __init__(self, c: Calibration, reference, fps, follow_policy, policy):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive finite")
        self.c = c
        self.reference = reference
        self.fps = fps
        self.policy = policy
        self.follow_policy = follow_policy
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

    def update(self, frame, image):
        if type(frame) is not int or frame < 0 or frame <= self.previous:
            raise ValueError("source frames must increase")
        if (image.shape[1], image.shape[0]) != self.c.frame_size:
            raise ValueError("geometry frame size mismatch")
        self.previous = frame
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
        else:
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
                if candidate is None:
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
                                "kind": "relocated",
                                "homography": self.anchor_h.tolist(),
                                "confirmed_frames": [f for f, _ in self.pending],
                                "score": candidate.score,
                            }
                        )
                        self.pending.clear()
                        self.rejected_at = None
                        self.lost = False
                        self.good_run = 0
                        a = candidate
                        status = "relocated"
                        diagnostic["reason"] = None
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
            if sample is None or sample.alignment is None:
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
    if sample is None or sample["alignment"] is None:
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
