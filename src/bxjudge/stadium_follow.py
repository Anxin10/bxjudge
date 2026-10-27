"""Small-motion stadium registration for review playback, never judging evidence.

Each frame matches the immutable calibration reference. No previous transform is carried
forward. Only spatially distributed rim matches may move the eight calibration polygons.
"""

import math

import cv2
import numpy as np

from .autocalib import Alignment, _edge_ncc, _stadium_centroid
from .calibration import _edge_map
from .geometry import rim_mask
from .types import Calibration

_FIELDS = {
    "work_width",
    "features",
    "ratio_test",
    "ransac_reprojection_px",
    "min_inliers",
    "min_inlier_fraction",
    "min_quadrants",
    "max_shift_frac",
    "max_rotation_deg",
    "min_scale",
    "max_scale",
    "min_edge_score",
    "min_region_score",
    "ecc_iterations",
    "ecc_epsilon",
}


def validate_follow_policy(raw: dict) -> dict:
    if not isinstance(raw, dict) or set(raw) != _FIELDS:
        raise ValueError("incomplete stadium_follow policy")
    ints = {"work_width", "features", "min_inliers", "min_quadrants", "ecc_iterations"}
    for key, value in raw.items():
        if key in ints:
            if type(value) is not int or value <= 0:
                raise ValueError(f"stadium_follow.{key} must be a positive integer")
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"stadium_follow.{key} must be finite")
    if (
        raw["work_width"] < 160
        or raw["min_inliers"] < 3
        or raw["features"] < raw["min_inliers"]
        or raw["min_quadrants"] > 4
        or not 0 < raw["ratio_test"] < 1
        or raw["ransac_reprojection_px"] <= 0
        or not 0 < raw["min_inlier_fraction"] <= 1
        or not 0 < raw["max_shift_frac"] <= 1
        or not 0 <= raw["max_rotation_deg"] < 90
        or not 0 < raw["min_scale"] <= 1 <= raw["max_scale"]
        or not 0 < raw["min_edge_score"] <= 1
        or not 0 < raw["min_region_score"] <= 1
        or raw["ecc_epsilon"] <= 0
    ):
        raise ValueError("stadium_follow policy is outside valid bounds")
    return dict(raw)


class StadiumFollower:
    """Reference-rim ORB + RANSAC seed, then bounded ECC affine registration per frame.

    Rim edge correlation must pass globally and in multiple image quadrants, independently
    of the matched features. This rejects coincidental matches and occlusions.
    Returned homography maps calibration pixels to current-frame pixels; failure is None, never a stale/identity substitute.
    """

    def __init__(self, c: Calibration, reference: np.ndarray, policy: dict):
        self.cfg = validate_follow_policy(policy)
        if reference is None:
            raise ValueError("cannot read stadium-follow reference")
        self.size = c.frame_size
        width = min(c.frame_size[0], self.cfg["work_width"])
        self.work_size = (
            width,
            max(1, round(c.frame_size[1] * width / c.frame_size[0])),
        )
        cx, cy = _stadium_centroid(c)
        self.centre = np.array(
            [
                cx * self.work_size[0] / self.size[0],
                cy * self.work_size[1] / self.size[1],
                1.0,
            ]
        )
        self.mask = rim_mask(c, self.work_size, 0).astype(np.uint8) * 255
        # Features may move out of the original rim; search only its bounded neighbourhood.
        radius = math.ceil(
            width * self.cfg["max_shift_frac"]
            + math.hypot(*self.work_size)
            * (
                math.radians(self.cfg["max_rotation_deg"])
                + max(1 - self.cfg["min_scale"], self.cfg["max_scale"] - 1)
            )
        )
        self.search_mask = cv2.dilate(
            self.mask, np.ones((2 * radius + 1,) * 2, np.uint8)
        )
        self.orb = cv2.ORB_create(nfeatures=self.cfg["features"])
        gray = self._gray(reference)
        self.edge = _edge_map(gray.astype(np.float32))
        self.keypoints, self.descriptors = self.orb.detectAndCompute(gray, self.mask)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last_diagnostic = {"reason": "not_attempted"}

    def _reject(self, reason):
        self.last_diagnostic["reason"] = reason

    def _gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.resize(gray, self.work_size, interpolation=cv2.INTER_AREA)

    def align(self, frame: np.ndarray) -> Alignment | None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError("stadium-follow frame size differs from calibration")
        self.last_diagnostic = {"reason": None}
        if self.descriptors is None:
            return self._reject("reference_features_missing")
        gray = self._gray(frame)
        keypoints, descriptors = self.orb.detectAndCompute(gray, self.search_mask)
        self.last_diagnostic["frame_features"] = len(keypoints)
        if descriptors is None or len(descriptors) < 2:
            return self._reject("frame_features_missing")
        candidates = [
            pair[0]
            for pair in self.matcher.knnMatch(self.descriptors, descriptors, k=2)
            if len(pair) == 2
            and pair[0].distance < self.cfg["ratio_test"] * pair[1].distance
        ]
        # Keep a one-to-one correspondence, not several reference points on one feature.
        unique = {}
        for match in sorted(candidates, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        self.last_diagnostic["distinct_matches"] = len(matches)
        if len(matches) < self.cfg["min_inliers"]:
            return self._reject("distinct_matches_insufficient")
        src = np.float32([self.keypoints[m.queryIdx].pt for m in matches])
        dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
        matrix, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.cfg["ransac_reprojection_px"],
        )
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            return self._reject("affine_seed_invalid")
        current_edge = _edge_map(gray.astype(np.float32))
        # Same fixed reference, input-domain rim mask. Refine mild camera tilt as well
        # as translation/rotation; no previous-frame transform or unchecked fallback.
        try:
            _, inverse = cv2.findTransformECC(
                current_edge,
                self.edge,
                cv2.invertAffineTransform(matrix).astype(np.float32),
                cv2.MOTION_AFFINE,
                (
                    cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                    self.cfg["ecc_iterations"],
                    self.cfg["ecc_epsilon"],
                ),
                self.mask,
                5,
            )
            matrix = cv2.invertAffineTransform(inverse).astype(np.float64)
        except cv2.error:
            return self._reject("ecc_nonconvergent")
        if not np.isfinite(matrix).all() or np.linalg.det(matrix[:, :2]) <= 0:
            return self._reject("affine_refinement_invalid")
        # Validate the final warp, including both directional scales (not just area).
        u, singular, vt = np.linalg.svd(matrix[:, :2])
        rotation_matrix = u @ vt
        rotation = math.degrees(
            math.atan2(-rotation_matrix[1, 0], rotation_matrix[0, 0])
        )
        scale = math.sqrt(float(np.linalg.det(matrix[:, :2])))
        w, h = self.work_size
        shift = float(np.linalg.norm(matrix @ self.centre - self.centre[:2])) / w
        self.last_diagnostic.update(
            {
                "shift_frac": shift,
                "rotation_deg": rotation,
                "min_scale": float(singular.min()),
                "max_scale": float(singular.max()),
            }
        )
        if (
            shift > self.cfg["max_shift_frac"]
            or abs(rotation) > self.cfg["max_rotation_deg"]
            or singular.min() < self.cfg["min_scale"]
            or singular.max() > self.cfg["max_scale"]
        ):
            self.last_diagnostic["limits"] = {
                k: self.cfg[k]
                for k in (
                    "max_shift_frac",
                    "max_rotation_deg",
                    "min_scale",
                    "max_scale",
                )
            }
            return self._reject("transform_out_of_bounds")
        predicted = src @ matrix[:, :2].T + matrix[:, 2]
        keep = (
            np.linalg.norm(predicted - dst, axis=1)
            <= self.cfg["ransac_reprojection_px"]
        )
        self.last_diagnostic.update(
            {
                "final_inliers": int(keep.sum()),
                "final_inlier_fraction": float(keep.mean()),
            }
        )
        if (
            keep.sum() < self.cfg["min_inliers"]
            or keep.mean() < self.cfg["min_inlier_fraction"]
        ):
            return self._reject("final_inliers_insufficient")
        aligned = cv2.warpAffine(
            current_edge,
            matrix,
            self.work_size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        score = _edge_ncc(self.edge, aligned, self.mask > 0)
        self.last_diagnostic["edge_score"] = score
        if score < self.cfg["min_edge_score"]:
            return self._reject("edge_score_below_minimum")
        supported = 0
        for y in (slice(0, h // 2), slice(h // 2, h)):
            for x in (slice(0, w // 2), slice(w // 2, w)):
                if (
                    _edge_ncc(self.edge[y, x], aligned[y, x], self.mask[y, x] > 0)
                    >= self.cfg["min_region_score"]
                ):
                    supported += 1
        self.last_diagnostic["supported_quadrants"] = supported
        if supported < self.cfg["min_quadrants"]:
            return self._reject("regional_support_insufficient")
        # Both image axes have their own exact resize factor (rounding may differ).
        resize = np.diag([w / self.size[0], h / self.size[1], 1.0])
        homography = np.linalg.inv(resize) @ np.vstack([matrix, [0, 0, 1]]) @ resize
        return Alignment(
            homography=tuple(tuple(float(v) for v in row) for row in homography),
            score=score,
            shift_frac=shift,
            rotation_deg=rotation,
            scale=scale,
            perspective=0.0,
        )
