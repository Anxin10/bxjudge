"""Offline appearance measurements, never identity probabilities or bindings."""

import math
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_policy(path: Path) -> dict:
    policy = yaml.safe_load(Path(path).read_text())
    required = {
        "purpose",
        "window_s",
        "max_samples",
        "min_crop_side_px",
        "max_outside_fraction",
        "color_bins",
        "gradient_bins",
    }
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("unexpected or missing appearance policy fields")
    if policy["purpose"] != "offline_diagnostic":
        raise ValueError("appearance policy is diagnostic only")
    for key, low, high in [("window_s", 0, 1), ("max_outside_fraction", 0, 1)]:
        value = policy[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"invalid {key}")
    if policy["window_s"] <= 0 or policy["max_outside_fraction"] >= 1:
        raise ValueError("invalid appearance sampling/crop window")
    for key, low, high in [
        ("max_samples", 3, 32),
        ("min_crop_side_px", 2, 4096),
        ("gradient_bins", 2, 64),
    ]:
        if type(policy[key]) is not int or not low <= policy[key] <= high:
            raise ValueError(f"invalid {key}")
    bins = policy["color_bins"]
    if (
        not isinstance(bins, list)
        or len(bins) != 3
        or any(type(v) is not int or not 2 <= v <= 32 for v in bins)
    ):
        raise ValueError("invalid color histogram bins")
    return policy


def sample_observations(track, endpoint, direction, fps, policy):
    if (
        direction not in ("before", "after")
        or type(endpoint) is not int
        or endpoint < 0
    ):
        raise ValueError("invalid endpoint")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid source FPS")
    observations = track["dets"]
    frames = [d["frame"] for d in observations]
    if (
        track["cls"] != "bey"
        or any(type(f) is not int or f < 0 for f in frames)
        or len(set(frames)) != len(frames)
    ):
        raise ValueError("invalid or duplicate native observations")
    if frames.count(endpoint) != 1:
        raise ValueError("endpoint is not an actual native observation")
    for d in observations:
        if (
            d["cls"] != "bey"
            or type(d["score"]) not in (int, float)
            or not math.isfinite(d["score"])
            or not 0 <= d["score"] <= 1
        ):
            raise ValueError("invalid native detection class or score")
    span = policy["window_s"] * fps
    selected = sorted(
        (
            d
            for d in observations
            if (
                0 <= endpoint - d["frame"] <= span
                if direction == "before"
                else 0 <= d["frame"] - endpoint <= span
            )
        ),
        key=lambda d: d["frame"],
    )
    indices = np.linspace(
        0, len(selected) - 1, min(len(selected), policy["max_samples"]), dtype=int
    )
    return [selected[i] for i in indices]


def describe_crop(image, box, policy):
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("expected raw uint8 BGR image")
    result = {
        "descriptor": None,
        "quality": {"visibility": "unverified"},
        "reason_codes": [],
    }
    if len(box) != 4 or any(
        type(x) not in (float, int) or not math.isfinite(x) for x in box
    ):
        result["reason_codes"] = ["invalid_box"]
        return result
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        result["reason_codes"] = ["empty_crop"]
        return result
    height, width = image.shape[:2]
    area = (x2 - x1) * (y2 - y1)
    inside = max(0, min(width, x2) - max(0, x1)) * max(0, min(height, y2) - max(0, y1))
    outside = 1 - inside / area
    bounds = [
        max(0, min(width, math.floor(x1))),
        max(0, min(height, math.floor(y1))),
        max(0, min(width, math.ceil(x2))),
        max(0, min(height, math.ceil(y2))),
    ]
    result["crop_bounds"] = bounds
    result["quality"]["outside_fraction"] = outside
    if outside > policy["max_outside_fraction"]:
        result["reason_codes"] = ["truncated_crop"]
        return result
    left, top, right, bottom = bounds
    if min(right - left, bottom - top) < policy["min_crop_side_px"]:
        result["reason_codes"] = ["small_crop"]
        return result
    crop = image[top:bottom, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color = cv2.calcHist(
        [hsv], [0, 1, 2], None, policy["color_bins"], [0, 180, 0, 256, 0, 256]
    ).ravel()
    color /= color.sum()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(dx, dy)
    gradient = np.histogram(
        angle, bins=policy["gradient_bins"], range=(0, 2 * np.pi), weights=magnitude
    )[0]
    if gradient.sum() > 0:
        gradient = (gradient / gradient.sum()).tolist()
    else:
        gradient = None
        result["reason_codes"].append("texture_absent")
    result["quality"].update(
        laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        mean_gradient=float(magnitude.mean()),
    )
    result["descriptor"] = {"color": color.tolist(), "gradient": gradient}
    return result


def distances(left, right):
    if left["descriptor"] is None or right["descriptor"] is None:
        raise ValueError("cannot compare unusable crops")
    result = {}
    for field, name in [
        ("color", "color_hellinger"),
        ("gradient", "gradient_hellinger"),
    ]:
        a, b = left["descriptor"][field], right["descriptor"][field]
        result[name] = (
            None
            if a is None or b is None
            else float(
                cv2.compareHist(
                    np.asarray(a, dtype=np.float32),
                    np.asarray(b, dtype=np.float32),
                    cv2.HISTCMP_BHATTACHARYYA,
                )
            )
        )
    return result
