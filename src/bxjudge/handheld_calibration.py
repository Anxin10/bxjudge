"""Handheld camera geometry: periodic keypoint re-finds, the follower in between.

A handheld camera moves the stadium far beyond the fixed-camera follower's bounds within a
round, so the zones are re-found from keypoints every ``refind_s`` playback seconds and the
``StadiumFollower`` carries them between re-finds, always against the frame of the latest
re-find (spec 2026-11-12-dynamic-calibration-design.md, handheld mode).

Two states. **Confirmed**: each frame's follower alignment is emitted (``tracking``; a
re-find that agrees is ``refound``). **Unconfirmed**: nothing is emitted. A re-find agrees
when its zones and the follower's lie within ``confirm_px`` (inference px, long edge 1280)
of each other, judged zone by zone. A confirmed span that a re-find contradicts is
invalidated back to the last confirmation: a wrong zone is worse than an unknown one. An
unconfirmed span ends only when a re-find agrees with the follower anchored on an earlier
re-find of another frame (two re-finds agreeing through the follower; a re-find of the
anchor's own frame is no new evidence); it is never backfilled. A confirmed span without an
agreeing re-find for ``max_unchecked_s`` stops being emitted, and what it emitted stays open
to contradiction: any later re-find that contradicts the same anchor invalidates it. Output is the
``GeometrySample`` stream and ``record()`` of ``DynamicCalibration``, plus the mode and every
re-find, so downstream code does not care which mode produced it.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import cv2
import numpy as np

from bxjudge.autocalib import warp_calibration
from bxjudge.dynamic_calibration import GeometrySample
from bxjudge.stadium_finder import _REF_LONG_EDGE
from bxjudge.stadium_follow import StadiumFollower
from bxjudge.stadium_locator import alignment_from_homography, zone_disagreement_px
from bxjudge.types import Calibration


def _normalised(h) -> np.ndarray | None:
    h = np.asarray(h, np.float64)
    if not np.isfinite(h).all() or h[2, 2] == 0:
        return None
    h = h / h[2, 2]
    return h if np.isfinite(h).all() else None


class HandheldCalibration:
    def __init__(self, c: Calibration, reference: np.ndarray, fps: float, follow_policy: dict,
                 handheld_policy: dict, locator, *, minimum_visible: float,
                 foreground_margin_px: float = 0.0, start_confirmed: bool = True,
                 start_frame: int | None = None):
        """``c`` describes ``reference`` (its zones in that image's px), the video's frame
        ``start_frame`` when known. ``start_confirmed`` False: the start is one unconfirmed
        keypoint fit (``keypoint_snapshot(allow_single=True)``), so nothing is emitted until
        a re-find of another frame agrees with it."""
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive finite")
        if not start_confirmed and not (type(start_frame) is int and start_frame >= 0):
            # Without it a re-find of the start's own frame could confirm the start by itself.
            raise ValueError("an unconfirmed start needs its source start_frame")
        self.c = c
        self.fps = fps
        self.follow_policy = follow_policy
        self.policy = dict(handheld_policy)
        self.locator = locator
        self.minimum_visible = minimum_visible
        self.margin = foreground_margin_px
        # Playback seconds -> frames (ruling T4); lengths from inference px to frame px.
        self.refind_n = max(1, round(self.policy["refind_s"] * fps))
        self.max_unchecked_n = max(1, round(self.policy["max_unchecked_s"] * fps))
        self.confirm_px = self.policy["confirm_px"] * max(c.frame_size) / _REF_LONG_EDGE
        self.follower = StadiumFollower(c, reference, follow_policy)
        self.anchor_h = np.eye(3)
        self.anchor_frame = start_frame  # source frame of the follower's reference
        self.anchor_emitted = False  # frames emitted through the current anchor
        self.confirmed = start_confirmed
        self.unconfirmed_reason = None if start_confirmed else "provisional_start"
        self.last_confirm = 0  # for max_unchecked_s
        self.invalidate_after = -1  # a contradiction invalidates emitted frames after this
        self.refind_base = 0  # re-finds are scheduled every refind_n frames from here
        self.refind_at = None  # the frame of the latest re-find actually run
        self.previous = -1
        self.foreground_boxes = []
        self._last = None
        self.segments = [{
            "id": 0,
            "start_frame": 0,
            "kind": "recorded_snapshot" if start_confirmed else "unconfirmed_snapshot",
            "homography": np.eye(3).tolist(),
            "confirmed_frames": [],
        }]
        self.samples: dict[int, GeometrySample] = {}
        self.diagnostics: dict[int, dict] = {}
        self.refinds: list[dict] = []

    def set_foreground(self, detections):
        self.foreground_boxes = [d.box for d in detections]

    def _foreground_mask(self, image) -> np.ndarray:
        mask = np.zeros(image.shape[:2], np.uint8)
        m = self.margin
        for x0, y0, x1, y1 in self.foreground_boxes:
            cv2.rectangle(mask, (math.floor(x0 - m), math.floor(y0 - m)),
                          (math.ceil(x1 + m), math.ceil(y1 + m)), 255, -1)
        return mask

    def _anchor(self, frame: int, image: np.ndarray, h: np.ndarray) -> None:
        """Follow from ``image`` (frame ``frame``), whose zones are ``c`` through ``h``."""
        moved = warp_calibration(self.c, alignment_from_homography(self.c, h, 1.0),
                                 self.c.frame_size, self.c.name)
        self.follower = StadiumFollower(moved, image.copy(), self.follow_policy)
        self.anchor_h = h
        self.anchor_frame = frame

    def _set(self, frame: int, alignment, status: str, reason) -> None:
        self.samples[frame] = GeometrySample(frame, alignment, len(self.segments) - 1, status)
        self.diagnostics[frame]["reason"] = reason

    def update(self, frame: int, image: np.ndarray) -> GeometrySample:
        if type(frame) is not int or frame < 0 or frame <= self.previous:
            raise ValueError("source frames must increase")
        if (image.shape[1], image.shape[0]) != tuple(self.c.frame_size):
            raise ValueError("geometry frame size mismatch")
        self.previous = frame
        self.follower.set_foreground_mask(self._foreground_mask(image), self.minimum_visible)
        p = self.follower.align(image)
        h_p = None if p is None else _normalised(np.asarray(p.homography) @ self.anchor_h)
        self.diagnostics[frame] = {"tracking": dict(self.follower.last_diagnostic),
                                   "refind": None, "reason": None}
        if self.confirmed and h_p is None:
            self.confirmed = False
            self.unconfirmed_reason = "follower_lost"
            self._set(frame, None, "lost", self.follower.last_diagnostic["reason"])
        elif self.confirmed and frame - self.last_confirm > self.max_unchecked_n:
            self.confirmed = False
            self.unconfirmed_reason = "unchecked_too_long"
            self._set(frame, None, "unconfirmed", self.unconfirmed_reason)
        elif self.confirmed:
            self._set(frame, alignment_from_homography(self.c, h_p, p.score), "tracking", None)
            self.anchor_emitted = True
        else:
            self._set(frame, None, "unconfirmed", self.unconfirmed_reason)
        self._last = (frame, image.copy(), h_p)  # finish() re-finds on exactly this frame
        if frame - self.refind_base >= self.refind_n:
            self._refind(frame, image, h_p)
        return self.samples[frame]

    def _refind(self, frame: int, image: np.ndarray, h_p) -> None:
        self.refind_base = self.refind_at = frame
        k = self.locator.locate(image)
        entry = {"frame": frame, "result": None, "disagreement_px": None, "reason": None}
        self.refinds.append(entry)
        self.diagnostics[frame]["refind"] = entry
        if k is None:
            entry.update(result="no_fit", reason=self.locator.last_reason)
            return
        d = None if h_p is None else zone_disagreement_px(self.c, h_p, k.homography)
        entry["disagreement_px"] = d if d is not None and math.isfinite(d) else None
        agrees = d is not None and d <= self.confirm_px
        found = alignment_from_homography(self.c, k.homography, k.alignment.score)
        if agrees and frame == self.anchor_frame:
            entry["result"] = "same_frame"  # the anchor's own frame: no second piece of evidence
            return
        if agrees:
            entry["result"] = "agree"
            if not self.confirmed:
                self.segments.append({
                    "id": len(self.segments),
                    "start_frame": frame,
                    "kind": "refound",
                    "homography": k.homography.tolist(),
                    "confirmed_frames": [self.anchor_frame, frame],
                })
                self.confirmed = True
                self.unconfirmed_reason = None
            self.last_confirm = frame
            self.invalidate_after = frame
            self._anchor(frame, image, k.homography)
            self.anchor_emitted = False
            self._set(frame, found, "refound", None)
            return
        # A follower result through an anchor that emitted frames, contradicted: those frames
        # are unknown, whether emission is still on or stopped since (unchecked, a loss).
        if self.confirmed or (h_p is not None and self.anchor_emitted):
            entry["result"] = "disagree"
            for f, s in list(self.samples.items()):
                if f > self.invalidate_after and s.alignment is not None:
                    self.samples[f] = GeometrySample(f, None, s.segment, "invalidated")
                    self.diagnostics[f]["reason"] = "refind_disagreement"
            self.samples[frame] = GeometrySample(frame, None, self.samples[frame].segment,
                                                 "invalidated")
            self.diagnostics[frame]["reason"] = "refind_disagreement"
            self.invalidate_after = frame
            self.confirmed = False
            self.unconfirmed_reason = "refind_disagreement"
        else:
            entry["result"] = "provisional"
        # Either way this re-find becomes the provisional reference; it is confirmed only by
        # a later re-find that agrees with the follower anchored here.
        self._anchor(frame, image, k.homography)
        self.anchor_emitted = False

    def finish(self) -> None:
        """Re-find on the last frame when it was not a re-find frame, so the tail since the
        last confirmation is checked like any other span."""
        if self._last is not None and self._last[0] != self.refind_at:
            self._refind(*self._last)

    def record(self) -> dict:
        return {
            "schema": 1,
            "reference_size": list(self.c.frame_size),
            "coordinate_mapping": "reference_to_source",
            "policy": {"handheld": self.policy, "stadium_follow": self.follow_policy},
            "temporal_policy": None,
            "mode": "handheld",
            "segments": self.segments,
            "refinds": self.refinds,
            "samples": [
                asdict(s) | {"rejection_reason": self.diagnostics[s.frame]["reason"],
                             "diagnostics": self.diagnostics[s.frame]}
                for s in self.samples.values()
            ],
        }
