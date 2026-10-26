"""Shared static/progressive trajectory drawing; coordinates are calibration pixels."""

import cv2
import numpy as np

from .calibration import ZONES
from .types import Calibration

SIDE_COLORS = {"X": (0, 159, 230), "B": (178, 114, 0)}
UNKNOWN_COLOR = (150, 150, 150)


def label(image, text, point, size, color):
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
    x = max(0, min(int(point[0]), image.shape[1] - width))
    y = max(height, min(int(point[1]), image.shape[0] - baseline))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)


def base_image(c: Calibration, reference: np.ndarray | None = None) -> np.ndarray:
    image = cv2.imread(c.ref_frame) if reference is None else reference
    if image is None:
        raise FileNotFoundError(c.ref_frame)
    image = cv2.resize(image, c.frame_size, interpolation=cv2.INTER_AREA)
    for field, _, _ in ZONES:
        pts = np.rint(getattr(c, field)).astype(np.int32)
        cv2.polylines(image, [pts], True, (80, 200, 80), 1)
        label(image, field, tuple(pts[0]), 0.35, (80, 200, 80))
    return image


def endpoint(image, point, side, track_id, frame, *, end=False,color=None):
    color = SIDE_COLORS.get(side, UNKNOWN_COLOR) if color is None else color
    identity = f"{side or '?'} #{track_id}"
    if end:
        cv2.drawMarker(image, point, (15, 15, 15), cv2.MARKER_TILTED_CROSS, 16, 5)
        cv2.drawMarker(image, point, color, cv2.MARKER_TILTED_CROSS, 16, 2)
        label(
            image,
            f"{identity} end f{frame}",
            (point[0] + 11, point[1] + 18),
            0.35,
            color,
        )
    else:
        cv2.circle(image, point, 9, (15, 15, 15), 4)
        cv2.circle(image, point, 7, color, 2)
        label(
            image,
            f"{identity} start f{frame}",
            (point[0] + 11, point[1] - 11),
            0.35,
            color,
        )


def with_legend(image,*,entries=None):
    footer_y = image.shape[0]
    image = cv2.copyMakeBorder(
        image, 0, 86, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)
    )
    labels=[(side+' side',SIDE_COLORS[side]) for side in ('X','B')] if entries is None else entries
    for (name,color),x in zip(labels,(18,190) if entries is None else (18,240)):
        cv2.line(image, (x, footer_y + 26), (x + 25, footer_y + 26), color, 3)
        label(image, name, (x + 33, footer_y + 31), 0.5, color)
    cv2.circle(image, (26, footer_y + 51), 7, (240, 240, 240), 2)
    label(image, "Start", (42, footer_y + 56), 0.45, (240, 240, 240))
    cv2.drawMarker(
        image, (198, footer_y + 51), (240, 240, 240), cv2.MARKER_TILTED_CROSS, 14, 2
    )
    label(image, "End", (216, footer_y + 56), 0.45, (240, 240, 240))
    label(image, "Unknown side / fragments" if entries is None else 'Dashed: interpolated, not judgment', (18, footer_y + 76), 0.4, UNKNOWN_COLOR)
    return image
