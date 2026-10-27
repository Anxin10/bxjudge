"""Observed rim flow with template closure and an independent robust ellipse cue.

A forward chain is pending evidence until a template registration closes it. Gaps
are never flow observations. All distances are source pixels unless documented.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from .autocalib import Alignment, _decompose, _stadium_centroid
from .config import config_loader
from .geometry import polygon_mask, rim_mask

DEFAULT_TEMPORAL_CONFIG = (
    Path(__file__).resolve().parents[2] / "config/temporal_geometry.yaml"
)
_KEYS = {
    "reanchor_s",
    "max_chain_s",
    "bridge_s",
    "consistency_px",
    "foreground_margin_px",
    "motion_threshold",
    "motion_dilate_px",
    "min_visible_fraction",
    "lk_window",
    "lk_levels",
    "lk_iterations",
    "lk_epsilon",
    "forward_backward_px",
    "feature_quality",
    "feature_spacing_px",
    "step_shift_frac",
    "ellipse_band_px",
    "ellipse_error_px",
    "ellipse_inlier_px",
    "ellipse_support",
    "ellipse_iterations",
    "ellipse_points",
    "ellipse_min_axis_fraction",
    "ellipse_max_axis_fraction",
    "ellipse_track_max_axis_fraction",
    "ellipse_min_battle_area_enclosed",
    "canny_low",
    "canny_high",
}
_INTS = {
    "motion_threshold",
    "motion_dilate_px",
    "lk_window",
    "lk_levels",
    "lk_iterations",
    "ellipse_iterations",
    "ellipse_points",
    "canny_low",
    "canny_high",
}


@config_loader
def load_temporal_policy(path=DEFAULT_TEMPORAL_CONFIG):
    p = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(p, dict) or set(p) != _KEYS:
        raise ValueError("incomplete temporal geometry policy")
    for k, v in p.items():
        if (
            type(v) not in (int, float)
            or not math.isfinite(v)
            or v <= 0
            or (k in _INTS and type(v) is not int)
        ):
            raise ValueError(f"invalid temporal geometry {k}")
    if (
        any(
            p[k] > 1
            for k in (
                "min_visible_fraction",
                "feature_quality",
                "step_shift_frac",
                "ellipse_support",
                "ellipse_min_battle_area_enclosed",
            )
        )
        or p["reanchor_s"] > p["max_chain_s"]
        or p["bridge_s"] > p["max_chain_s"]
        or p["lk_window"] % 2 != 1
        or p["canny_low"] >= p["canny_high"]
        or p["canny_high"] > 255
        or p["ellipse_points"] < 5
        or p["motion_threshold"] > 255
        or p["lk_window"] < 3
        or not p["ellipse_min_axis_fraction"] <= p["ellipse_max_axis_fraction"]
    ):
        raise ValueError("temporal geometry policy outside bounds")
    return p


def _points(h, points):
    return cv2.perspectiveTransform(np.float64(points).reshape(-1, 1, 2), h).reshape(
        -1, 2
    )


def _ellipse_curve(e):
    centre, axes, angle = e
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    p = np.column_stack([axes[0] / 2 * np.cos(t), axes[1] / 2 * np.sin(t)])
    a = math.radians(angle)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return p @ rotation.T + centre


def _ellipse_residual(e, p):
    centre, axes, angle = e
    if min(axes) <= 0:
        return np.full(len(p), np.inf)
    a = math.radians(angle)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    q = (p - centre) @ rotation
    r = np.linalg.norm(q / np.array(axes) * 2, axis=1)
    return abs(r - 1) * min(axes) / 2


def _enclosed_fraction(e, p, band):
    """Share of ``p`` inside ellipse ``e`` or at most ``band`` px outside it.

    Outside distance uses the semi-major axis, an upper bound on the true
    distance, so an elongated ellipse cannot pass points far beyond its ends."""
    centre, axes, angle = e
    a = math.radians(angle)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    r = np.linalg.norm((p - centre) @ rotation / np.array(axes) * 2, axis=1)
    return float(np.mean((r - 1) * max(axes) / 2 <= band))


class TemporalRimTracker:
    def __init__(self, c, reference, fps, follow_policy, policy):
        self.c, self.fps, self.follow, self.cfg = c, fps, follow_policy, policy
        w = min(c.frame_size[0], follow_policy["work_width"])
        self.size = (w, round(c.frame_size[1] * w / c.frame_size[0]))
        self.resize = np.diag(
            [self.size[0] / c.frame_size[0], self.size[1] / c.frame_size[1], 1.0]
        )
        self.rim = rim_mask(c, self.size, 0).astype(np.uint8) * 255
        self.reference_gray = self.gray(reference)
        self.reference_ellipse = self.ellipse(
            self.reference_gray, np.eye(3), np.zeros_like(self.rim)
        )
        self.previous = None
        self.previous_mask = None
        self.frame = None
        self.h = None
        self.anchor_frame = None
        self.segment = None
        self.pending = {}
        self.last_diagnostic = {"reason": "not_attempted"}
        self.foreground_mask = np.zeros((c.frame_size[1], c.frame_size[0]), np.uint8)

    def gray(self, image):
        return cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            self.size,
            interpolation=cv2.INTER_AREA,
        )

    def ellipse(self, gray, h, excluded):
        # Polygon only defines a search band. The second cue is fitted to actual
        # unmasked image edges, not to the calibration coordinates themselves.
        initial = not hasattr(self, "reference_ellipse")
        if initial:
            band = cv2.resize(
                polygon_mask(self.c.stadium, self.c.frame_size),
                self.size,
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            if self.reference_ellipse is None:
                return None
            expected = _points(h, self.reference_ellipse) @ self.resize[:2, :2].T
            e = cv2.fitEllipse(expected.astype(np.float32))
            band = np.zeros_like(gray)
            cv2.ellipse(
                band,
                e,
                255,
                max(1, round(2 * self.cfg["ellipse_band_px"] * self.resize[0, 0])),
            )
        edges = cv2.Canny(gray, self.cfg["canny_low"], self.cfg["canny_high"])
        ys, xs = np.where((edges > 0) & (band > 0) & (excluded == 0))
        points = np.column_stack([xs, ys]).astype(np.float32)
        if len(points) < self.follow["min_inliers"]:
            return None
        rng = np.random.default_rng(0)
        if len(points) > self.cfg["ellipse_points"]:
            points = points[
                rng.choice(len(points), self.cfg["ellipse_points"], replace=False)
            ]
        if initial:
            contours, _ = cv2.findContours(
                ((edges > 0) & (band > 0) & (excluded == 0)).astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE,
            )
            choices = []
            for contour in contours:
                if len(contour) < self.follow["min_inliers"]:
                    continue
                try:
                    fit = cv2.fitEllipse(contour)
                except cv2.error:
                    continue
                if (
                    min(fit[1]) < min(self.size) * self.cfg["ellipse_min_axis_fraction"]
                    or max(fit[1])
                    > max(self.size) * self.cfg["ellipse_max_axis_fraction"]
                ):
                    continue
                curve = _ellipse_curve(fit)
                if (
                    np.any(curve < 0)
                    or np.any(curve[:, 0] >= self.size[0])
                    or np.any(curve[:, 1] >= self.size[1])
                ):
                    continue
                near = (
                    _ellipse_residual(fit, points)
                    <= self.cfg["ellipse_band_px"] * self.resize[0, 0]
                )
                if near.sum() < self.follow["min_inliers"]:
                    continue
                # A stadium rim surrounds the battle floor. An arc inside it (for
                # example a lid reflection) is not the rim, however well it fits.
                floor = np.float64(self.c.battle_area) @ self.resize[:2, :2].T
                if (
                    _enclosed_fraction(
                        fit, floor, self.cfg["ellipse_band_px"] * self.resize[0, 0]
                    )
                    < self.cfg["ellipse_min_battle_area_enclosed"]
                ):
                    continue
                inliers = (
                    _ellipse_residual(fit, points[near])
                    <= self.cfg["ellipse_inlier_px"] * self.resize[0, 0]
                )
                offsets = points[near][inliers] - fit[0]
                if (
                    len({(int(x >= 0), int(y >= 0)) for x, y in offsets})
                    < self.follow["min_quadrants"]
                    or inliers.mean() < self.cfg["ellipse_support"]
                ):
                    continue
                choices.append(
                    (int(inliers.sum()) * math.sqrt(fit[1][0] * fit[1][1]), curve)
                )
            return (
                max(choices, key=lambda item: item[0])[1]
                / np.array([self.resize[0, 0], self.resize[1, 1]])
                if choices
                else None
            )
        best = None
        count = 0
        best_error = math.inf
        tol = self.cfg["ellipse_inlier_px"] * self.resize[0, 0]
        contours, _ = cv2.findContours(
            ((edges > 0) & (band > 0) & (excluded == 0)).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )
        candidates = [
            p.reshape(-1, 2) for p in contours if len(p) >= self.follow["min_inliers"]
        ]
        candidates += [
            points[rng.choice(len(points), 5, replace=False)]
            for _ in range(self.cfg["ellipse_iterations"])
        ]
        for candidate in candidates:
            try:
                fit = cv2.fitEllipse(candidate)
            except cv2.error:
                continue
            limit = self.cfg["ellipse_track_max_axis_fraction"] * max(self.size)
            if min(fit[1]) <= 0 or max(fit[1]) > limit:
                continue
            keep = _ellipse_residual(fit, points) <= tol
            error = float(np.mean(_ellipse_residual(fit, expected)))
            if (
                keep.sum() >= self.follow["min_inliers"]
                and keep.mean() >= self.cfg["ellipse_support"]
                and error < best_error
            ):
                best, count, best_error = keep, int(keep.sum()), error
        if (
            best is None
            or count < self.follow["min_inliers"]
            or count / len(points) < self.cfg["ellipse_support"]
        ):
            return None
        try:
            fit = cv2.fitEllipse(points[best])
        except cv2.error:
            return None
        # Partial arcs alone cannot establish the stadium's pose.
        offsets = points[best] - fit[0]
        quadrants = {(int(x >= 0), int(y >= 0)) for x, y in offsets}
        if len(quadrants) < self.follow["min_quadrants"]:
            return None
        return _ellipse_curve(fit) / np.array([self.resize[0, 0], self.resize[1, 1]])

    def reject(self, reason):
        self.last_diagnostic["reason"] = reason
        self.pending.clear()
        self.h = None

    def observe(self, frame, image, boxes):
        gray = self.gray(image)
        excluded = np.zeros_like(gray)
        margin = self.cfg["foreground_margin_px"]
        for box in boxes:
            x0, y0, x1, y1 = box
            cv2.rectangle(
                excluded,
                (
                    math.floor((x0 - margin) * self.resize[0, 0]),
                    math.floor((y0 - margin) * self.resize[1, 1]),
                ),
                (
                    math.ceil((x1 + margin) * self.resize[0, 0]),
                    math.ceil((y1 + margin) * self.resize[1, 1]),
                ),
                255,
                -1,
            )
        self.foreground_mask = cv2.resize(
            excluded, self.c.frame_size, interpolation=cv2.INTER_NEAREST
        )
        previous, prev_frame = self.previous, self.frame
        previous_mask = self.previous_mask
        self.previous_mask = excluded.copy()
        self.previous, self.frame = gray, frame
        self.last_diagnostic = {"reason": None, "foreground_boxes": len(boxes)}
        if previous is None or self.h is None:
            return self.reject("temporal_anchor_missing")
        if frame != prev_frame + 1:
            return self.reject("temporal_source_gap")
        if (frame - self.anchor_frame) / self.fps > self.cfg["max_chain_s"]:
            return self.reject("template_reanchor_expired")
        previous_rim = cv2.warpPerspective(
            self.rim, self.resize @ self.h @ np.linalg.inv(self.resize), self.size
        )
        mask = previous_rim.copy()
        mask[previous_mask > 0] = 0
        if (
            np.count_nonzero(mask) / max(1, np.count_nonzero(previous_rim))
            < self.cfg["min_visible_fraction"]
        ):
            return self.reject("temporal_foreground_occlusion")
        pts = cv2.goodFeaturesToTrack(
            previous,
            self.follow["features"],
            self.cfg["feature_quality"],
            self.cfg["feature_spacing_px"] * self.resize[0, 0],
            mask=mask,
        )
        if pts is None or len(pts) < self.follow["min_inliers"]:
            return self.reject("temporal_features_insufficient")
        lk = {
            "winSize": (self.cfg["lk_window"],) * 2,
            "maxLevel": self.cfg["lk_levels"],
            "criteria": (
                cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                self.cfg["lk_iterations"],
                self.cfg["lk_epsilon"],
            ),
        }
        dst, valid, _ = cv2.calcOpticalFlowPyrLK(previous, gray, pts, None, **lk)
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, previous, dst, None, **lk)
        keep = (
            (valid.ravel() > 0)
            & (reverse.ravel() > 0)
            & (
                np.linalg.norm(back - pts, axis=2).ravel()
                <= self.cfg["forward_backward_px"] * self.resize[0, 0]
            )
        )
        src, dst = pts.reshape(-1, 2)[keep], dst.reshape(-1, 2)[keep]
        if len(src) < self.follow["min_inliers"]:
            return self.reject("temporal_flow_inconsistent")
        m, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.follow["ransac_reprojection_px"],
        )
        if m is None or not np.isfinite(m).all():
            return self.reject("temporal_transform_invalid")
        # Camera-compensated residual motion is foreground, not a moving rim.
        predicted = cv2.warpAffine(previous, m, self.size)
        motion = (cv2.absdiff(predicted, gray) > self.cfg["motion_threshold"]).astype(
            np.uint8
        ) * 255
        k = max(1, round(self.cfg["motion_dilate_px"] * self.resize[0, 0]))
        motion = cv2.dilate(motion, np.ones((2 * k + 1,) * 2, np.uint8))
        excluded |= motion
        self.previous_mask = excluded.copy()
        # Export the motion mask before any rejection: template alignment needs it
        # most exactly when the occlusion is heavy enough to break the chain.
        self.foreground_mask = cv2.resize(
            excluded, self.c.frame_size, interpolation=cv2.INTER_NEAREST
        )
        current_rim = cv2.warpAffine(previous_rim, m, self.size)
        if (
            np.count_nonzero((current_rim > 0) & (excluded == 0))
            / max(1, np.count_nonzero(current_rim))
            < self.cfg["min_visible_fraction"]
        ):
            return self.reject("temporal_motion_occlusion")
        xy = np.round(dst).astype(int)
        inside = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < self.size[0])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < self.size[1])
        )
        clean = np.zeros(len(dst), bool)
        clean[inside] = excluded[xy[inside, 1], xy[inside, 0]] == 0
        clean &= inliers.ravel() > 0
        if (
            clean.sum() < self.follow["min_inliers"]
            or clean.mean() < self.follow["min_inlier_fraction"]
        ):
            return self.reject("temporal_unmasked_support_insufficient")
        q = dst[clean]
        quadrants = {
            (int(x >= self.size[0] / 2), int(y >= self.size[1] / 2)) for x, y in q
        }
        if len(quadrants) < self.follow["min_quadrants"]:
            return self.reject("temporal_regional_support_insufficient")
        m, _ = cv2.estimateAffinePartial2D(
            src[clean],
            dst[clean],
            method=cv2.RANSAC,
            ransacReprojThreshold=self.follow["ransac_reprojection_px"],
        )
        if m is None:
            return self.reject("temporal_transform_invalid")
        step = np.linalg.inv(self.resize) @ np.vstack([m, [0, 0, 1]]) @ self.resize
        if (
            np.max(
                np.linalg.norm(_points(step, self.c.stadium) - self.c.stadium, axis=1)
            )
            / self.c.frame_size[0]
            > self.cfg["step_shift_frac"]
        ):
            return self.reject("temporal_step_out_of_bounds")
        h = step @ self.h
        observed = self.ellipse(gray, h, excluded)
        if observed is None or self.reference_ellipse is None:
            return self.reject("temporal_ellipse_missing")
        predicted = _points(h, self.reference_ellipse)
        dist = np.linalg.norm(predicted[:, None, :] - observed[None, :, :], axis=2)
        error = max(float(dist.min(axis=0).mean()), float(dist.min(axis=1).mean()))
        self.last_diagnostic["ellipse_error_px"] = error
        if error > self.cfg["ellipse_error_px"]:
            return self.reject("temporal_ellipse_disagreement")
        self.h = h
        self.pending[frame] = h.copy()
        return h.copy()

    def anchor(self, frame, h, segment):
        self.h = np.array(h, float)
        self.anchor_frame = frame
        self.segment = segment
        self.pending.clear()

    def confirm(self, frame, h, segment):
        if self.segment != segment or frame not in self.pending:
            return []
        expected = _points(np.array(h, float), self.c.stadium)
        actual = _points(self.pending[frame], self.c.stadium)
        error = float(np.linalg.norm(actual - expected, axis=1).max())
        self.last_diagnostic["closure_error_px"] = error
        if error > self.cfg["consistency_px"]:
            self.reject("temporal_template_disagreement")
            return []
        return list(self.pending)

    def alignment(self, h, score):
        centre = np.array(_stadium_centroid(self.c))
        moved = _points(h, [centre])[0]
        rotation, scale, _, _, perspective = _decompose(h, centre)
        return Alignment(
            tuple(tuple(float(v) for v in row) for row in h),
            score,
            float(np.linalg.norm(moved - centre)) / self.c.frame_size[0],
            rotation,
            scale,
            perspective,
        )
