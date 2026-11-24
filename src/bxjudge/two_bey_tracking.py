"""Two physical top identities per round; raw observations and display estimates differ."""

import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from .config import config_loader


@config_loader
def load_policy(path):
    p = yaml.safe_load(Path(path).read_text())
    keys = {
        "max_targets",
        "pair_confirmation_frames",
        "duplicate_iou",
        "position_weight",
        "appearance_weight",
        "native_id_bonus",
        "unmatched_cost",
        "initial_separation_ratio",
        "kalman_process_noise",
        "kalman_measurement_noise",
        "position_gate_ratio",
        "appearance_gate",
        "appearance_bins",
        "appearance_update_alpha",
        "missing_velocity_decay",
        "min_frame_border_px",
        "merge_observation_max_gap_s",
        "merge_center_margin_ratio",
        "reanchor_motion_max_gap_s",
        "reanchor_motion_margin",
        "reanchor_appearance_min_separation",
        "reanchor_appearance_margin",
        "duplicate_containment",
        "reacquire_min_gap_s",
        "reacquire_confirm_frames",
        "reacquire_min_score",
        "circular_prediction_enabled",
        "circular_history_size",
        "circular_min_points",
        "circular_max_residual_ratio",
        "max_prediction_drift_ratio",
    }
    if (
        not isinstance(p, dict)
        or set(p) != keys
        or type(p["max_targets"]) is not int
        or p["max_targets"] != 2
    ):
        raise ValueError("two-bey policy must explicitly configure exactly two targets")
    if type(p['pair_confirmation_frames']) is not int or p['pair_confirmation_frames']<2:
        raise ValueError('two-bey pair requires at least two actual source frames')
    if type(p["reacquire_confirm_frames"]) is not int:
        raise ValueError("re-acquisition needs a whole number of source frames")
    if type(p["circular_prediction_enabled"]) is not bool:
        raise ValueError("circular_prediction_enabled must be a boolean")
    if (
        type(p["circular_history_size"]) is not int
        or type(p["circular_min_points"]) is not int
        or p["circular_min_points"] < 3
        or p["circular_history_size"] < p["circular_min_points"]
    ):
        raise ValueError("invalid circular history policy")
    for k in keys - {
        "max_targets",
        "appearance_bins",
        "circular_prediction_enabled",
        "circular_history_size",
        "circular_min_points",
    }:
        if type(p[k]) not in (int, float) or not math.isfinite(p[k]) or p[k] <= 0:
            raise ValueError("invalid two-bey policy value")
    if any(
        p[k] > 1
        for k in (
            "duplicate_iou",
            "position_gate_ratio",
            "appearance_gate",
            "appearance_update_alpha",
            "missing_velocity_decay",
            "merge_center_margin_ratio",
            "reanchor_appearance_min_separation",
            "reanchor_appearance_margin",
            "duplicate_containment",
            "reacquire_min_score",
            "circular_max_residual_ratio",
            "max_prediction_drift_ratio",
        )
    ):
        raise ValueError("invalid bounded matching gate")
    if (
        not isinstance(p["appearance_bins"], list)
        or len(p["appearance_bins"]) != 3
        or any(type(v) is not int or not 2 <= v <= 32 for v in p["appearance_bins"])
    ):
        raise ValueError("invalid appearance histogram bins")
    return p


def center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=float)


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union


def containment(a, b):
    """Fraction of box ``a``'s own area inside box ``b`` (asymmetric)."""
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]))


def _appearance(value):
    if value is None:
        return None
    a = np.asarray(value, dtype=np.float32)
    if (
        a.ndim != 1
        or not len(a)
        or not np.isfinite(a).all()
        or np.any(a < 0)
        or a.sum() <= 0
    ):
        raise ValueError("invalid appearance descriptor")
    return a / a.sum()


def identity_unknown_at(record, track_id, frame):
    """True when a physical round record leaves ``track_id``'s player unknown at ``frame``."""
    key = str(track_id)
    since = record["identity_unknown_since"].get(key)
    if since is not None and frame >= since:
        return True
    return any(
        start <= frame <= end
        for start, end in record["identity_unknown_intervals"].get(key, [])
    )


def _vote(margin, required):
    # Positive margin favours keeping slot order; None means no opinion.
    if margin is None or abs(margin) <= required:
        return None
    return "identity" if margin > 0 else "exchange"


def _fit_circle_lsq(points: np.ndarray) -> tuple[float, float, float] | None:
    """Fit a 2D circle (cx, cy, radius) via algebraic least squares; None if degenerate."""
    if len(points) < 3:
        return None
    x = points[:, 0]
    y = points[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(points), dtype=np.float64)])
    b = x * x + y * y
    sol, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank < 3:
        return None
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    r_sq = c + cx * cx + cy * cy
    if not math.isfinite(r_sq) or r_sq <= 0:
        return None
    return cx, cy, math.sqrt(r_sq)


def _predict_circular_arc(
    history: list[tuple[int, np.ndarray]],
    target_frame: int,
    min_points: int,
    max_residual_ratio: float,
    diagonal: float,
) -> np.ndarray | None:
    """Extrapolate circular arc position for association; returns None on degenerate arcs."""
    if len(history) < min_points:
        return None
    frames = np.array([f for f, _ in history], dtype=np.float64)
    if frames[-1] <= frames[0] or target_frame <= frames[-1]:
        return None
    pts = np.array([p for _, p in history], dtype=np.float64)
    fit = _fit_circle_lsq(pts)
    if fit is None:
        return None
    cx, cy, radius = fit
    if not (0.03 * diagonal <= radius <= 1.2 * diagonal):
        return None
    radial = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    rms_residual = float(np.sqrt(np.mean((radial - radius) ** 2)))
    if rms_residual > max_residual_ratio * radius:
        return None
    angles = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    if abs(float(angles[-1] - angles[0])) < 0.05:
        return None
    omega = float(angles[-1] - angles[0]) / float(frames[-1] - frames[0])
    theta = float(angles[-1]) + omega * float(target_frame - frames[-1])
    return np.array([cx + radius * math.cos(theta), cy + radius * math.sin(theta)], dtype=np.float64)


def _clamp_to_rim(
    pred: np.ndarray,
    last_pos: np.ndarray,
    arena_circle: tuple[float, float, float] | None,
    max_drift: float,
) -> np.ndarray:
    """Bound prediction drift and clamp outward motion onto the stadium rim circle."""
    out = pred.astype(np.float64, copy=True)
    delta = out - last_pos
    drift = float(np.hypot(delta[0], delta[1]))
    if drift > max_drift > 0:
        out = last_pos + delta * (max_drift / drift)
    if arena_circle is not None:
        acx, acy, ar = arena_circle
        rel_x = float(out[0]) - acx
        rel_y = float(out[1]) - acy
        dist = math.hypot(rel_x, rel_y)
        if dist > ar > 0:
            scale = ar / dist
            out = np.array([acx + rel_x * scale, acy + rel_y * scale], dtype=np.float64)
    return out


class TwoBeyTracker:
    """Bounded one-to-one assignment, distinct from detector and referee evidence."""

    def __init__(self, frame_size, fps, policy, *, arena_circle=None):
        if (
            len(frame_size) != 2
            or any(type(v) is not int or v <= 0 for v in frame_size)
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("invalid source coordinates or FPS")
        if arena_circle is not None:
            if (
                len(arena_circle) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in arena_circle)
                or arena_circle[2] <= 0
            ):
                raise ValueError("invalid arena_circle")
            arena_circle = tuple(float(v) for v in arena_circle)
        self.frame_size = tuple(frame_size)
        self.fps = fps
        self.policy = dict(policy)
        self.arena_circle = arena_circle
        self.slots = []
        self.previous = -1
        self.diagonal = math.hypot(*frame_size)
        self.native_owners = {}
        # First owner of each native ID: player anchors are keyed by native IDs
        # seen before any merge, so a later re-anchor must not move them.
        self.native_anchor_owner = {}
        # Open (unresolved) unknown spans; a recovered span moves to the closed
        # intervals and the entity becomes valid again from the split frame.
        self.identity_unknown_since = {}
        self.identity_unknown_intervals = {}
        self.merge_episodes = []
        self.pending_merge = None
        # Elimination re-acquisition: (native_id, slot, last frame, consecutive frames).
        self.reacquire_streak = None
        self.reacquisitions = []

    @property
    def entity_count(self):
        return len(self.slots)

    def _create(self, d, appearance, isolated):
        k = cv2.KalmanFilter(4, 2, 0, cv2.CV_64F)
        k.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        k.processNoiseCov = np.eye(4) * self.policy["kalman_process_noise"]
        k.measurementNoiseCov = np.eye(2) * self.policy["kalman_measurement_noise"]
        k.errorCovPost = np.eye(4) * 1000
        k.statePost = np.array([*center(d["box"]), 0, 0], dtype=np.float64).reshape(
            4, 1
        )
        slot = {
            "entity_id": len(self.slots) + 1,
            "kalman": k,
            "appearance": appearance,
            # Re-anchoring evidence: only crops that overlap no other box.
            "clean_appearance": appearance.copy()
            if isolated and appearance is not None
            else None,
            "native_id": d["native_id"],
            "last_box": list(d["box"]),
            "last_observed_frame": self.previous,
            "history": deque(
                [(self.previous, center(d["box"]))],
                maxlen=self.policy.get("circular_history_size", 10),
            ),
        }
        self.slots.append(slot)
        self._own(d["native_id"], slot["entity_id"])
        return slot

    def _eliminate(self, frame, kept, assigned, predicted):
        """Assign a persistent sole leftover to the only stale slot (two tops per round).

        Mutates ``assigned``; returns the slot index whose appearance restarts, or None.
        """
        p = self.policy
        candidate = None
        if len(self.slots) == 2 and self.pending_merge is None and len(assigned) == 1:
            [(seen, j_seen)] = assigned.items()
            k = 1 - seen
            leftovers = [j for j in range(len(kept)) if j != j_seen]
            if (
                frame - self.slots[k]["last_observed_frame"] >= round(p["reacquire_min_gap_s"] * self.fps)
                and len(leftovers) == 1
            ):
                j = leftovers[0]
                d = kept[j][0]
                owner = self.native_owners.get(d["native_id"])
                if (
                    d["score"] >= p["reacquire_min_score"]
                    and iou(d["box"], kept[j_seen][0]["box"]) == 0
                    and owner in (None, self.slots[k]["entity_id"])
                    and float(np.linalg.norm(predicted[k] - center(d["box"]))) / self.diagonal
                    <= p["position_gate_ratio"]
                ):
                    candidate = (k, j, d["native_id"])
        if candidate is None:
            self.reacquire_streak = None
            return None
        k, j, native = candidate
        streak = self.reacquire_streak
        count = streak[3] + 1 if (
            streak is not None and streak[:2] == (native, k) and frame == streak[2] + 1
        ) else 1
        self.reacquire_streak = (native, k, frame, count)
        if count < p["reacquire_confirm_frames"]:
            return None
        self.reacquire_streak = None
        assigned[k] = j
        return k

    def _own(self, native_id, entity_id):
        self.native_owners[native_id] = entity_id
        # Anchor membership gives the whole physical track its player; a native
        # joining while the player is unknown is spatial evidence only.
        if entity_id not in self.identity_unknown_since:
            self.native_anchor_owner.setdefault(native_id, entity_id)

    def _reanchor(self, frame, kept, predicted):
        """Decide slot -> observation from pre-merge motion and appearance only."""
        m = self.pending_merge
        boxes = [d["box"] for d, _ in kept]
        centres = [center(b) for b in boxes]
        motion = None
        gap = max(frame - s["last_observed_frame"] for s in m["slots"]) / self.fps
        if gap <= self.policy["reanchor_motion_max_gap_s"]:
            p = [
                s["position"] + s["velocity"] * (frame - s["last_observed_frame"])
                for s in m["slots"]
            ]

            def cost(a, b):
                return np.linalg.norm(p[0] - centres[a]) + np.linalg.norm(
                    p[1] - centres[b]
                )

            scale = np.mean([math.hypot(b[2] - b[0], b[3] - b[1]) for b in boxes])
            motion = float((cost(1, 0) - cost(0, 1)) / scale)
        appearance = None
        a = [s["appearance"] for s in m["slots"]]
        o = [x for _, x in kept]

        def bh(x, y):
            return float(cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA))

        # Indistinguishable tops (same model) cannot vote either way.
        if (
            all(v is not None for v in (*a, *o))
            and bh(a[0], a[1]) >= self.policy["reanchor_appearance_min_separation"]
        ):
            appearance = (bh(a[0], o[1]) + bh(a[1], o[0])) - (
                bh(a[0], o[0]) + bh(a[1], o[1])
            )
        votes = {
            "motion": _vote(motion, self.policy["reanchor_motion_margin"]),
            "appearance": _vote(appearance, self.policy["reanchor_appearance_margin"]),
        }
        cast = {v for v in votes.values() if v is not None}
        # Votes are relative to these orders: "identity" maps slot i to
        # observation i, "exchange" maps slot i to the other observation.
        episode = {
            "merge_frame": m["start"],
            "split_frame": frame,
            "slot_entities": [s["entity_id"] for s in m["slots"]],
            "slot_native_ids": [s["native_id"] for s in m["slots"]],
            "observation_native_ids": [d["native_id"] for d, _ in kept],
            "motion_margin": motion,
            "appearance_margin": appearance,
            "votes": votes,
        }
        if len(cast) != 1:
            episode.update(
                decision="unknown",
                reason="conflicting_cues" if cast else "no_cue_past_margin",
            )
            return episode, None
        mapping = [0, 1] if cast == {"identity"} else [1, 0]
        # A relative margin cannot vouch for observations that normal tracking
        # would reject for either top.
        if any(
            np.linalg.norm(predicted[i] - centres[j]) / self.diagonal
            > self.policy["position_gate_ratio"]
            or (
                a[i] is not None
                and o[j] is not None
                and bh(a[i], o[j]) > self.policy["appearance_gate"]
            )
            for i, j in enumerate(mapping)
        ):
            episode.update(decision="unknown", reason="split_outside_gates")
            return episode, None
        episode.update(
            decision="recovered",
            identity_recovered_by=sorted(k for k, v in votes.items() if v),
            native_ids_swapped=any(
                kept[j][0]["native_id"] != s["native_id"]
                for s, j in zip(m["slots"], mapping, strict=True)
            ),
        )
        return episode, mapping

    def update(self, frame, detections):
        if type(frame) is not int or frame < 0 or frame <= self.previous:
            raise ValueError("source frames must increase strictly")
        ids = set()
        prepared = []
        for d in detections:
            b = d["box"]
            if (
                type(d["native_id"]) is not int
                or d["native_id"] < 0
                or d["native_id"] in ids
                or len(b) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in b)
                or b[0] >= b[2]
                or b[1] >= b[3]
                or not math.isfinite(d["score"])
                or not 0 <= d["score"] <= 1
            ):
                raise ValueError("invalid or duplicate native detection")
            ids.add(d["native_id"])
            prepared.append((d, _appearance(d["appearance"])))
        shapes = {a.shape for _, a in prepared if a is not None}
        shapes.update(
            s["appearance"].shape for s in self.slots if s["appearance"] is not None
        )
        if len(shapes) > 1:
            raise ValueError("appearance dimensions changed")
        prepared.sort(key=lambda item: (-item[0]["score"], item[0]["native_id"]))
        kept = []
        rejected = []
        for d, a in prepared:
            border = self.policy["min_frame_border_px"]
            if (
                d["box"][0] < border
                or d["box"][1] < border
                or d["box"][2] > self.frame_size[0] - border
                or d["box"][3] > self.frame_size[1] - border
            ):
                rejected.append(
                    {
                        **d,
                        "frame": frame,
                        "reason": "border_truncated_object_unverified",
                    }
                )
                continue
            duplicate = next(
                (
                    other
                    for other, _ in kept
                    if iou(d["box"], other["box"]) >= self.policy["duplicate_iou"]
                    and not (
                        d["native_id"] in self.native_owners
                        and other["native_id"] in self.native_owners
                        and self.native_owners[d["native_id"]]
                        != self.native_owners[other["native_id"]]
                    )
                ),
                None,
            )
            if duplicate is not None:
                rejected.append(
                    {
                        **d,
                        "frame": frame,
                        "reason": "duplicate_box",
                        "duplicate_of": duplicate["native_id"],
                    }
                )
            else:
                kept.append((d, a))
        dt = 1 if self.previous < 0 else frame - self.previous
        # Pre-merge motion: predict() overwrites statePost, so read it first.
        before = [
            {
                "entity_id": s["entity_id"],
                "native_id": s["native_id"],
                "position": center(s["last_box"]),
                "velocity": s["kalman"].statePost[2:, 0].copy(),
                "last_observed_frame": s["last_observed_frame"],
                "appearance": None
                if s["clean_appearance"] is None
                else s["clean_appearance"].copy(),
            }
            for s in self.slots
        ]
        predicted = []
        for s in self.slots:
            s["kalman"].transitionMatrix = np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float64,
            )
            pred = s["kalman"].predict()[:2, 0]
            if self.policy.get("circular_prediction_enabled"):
                circ = _predict_circular_arc(
                    list(s["history"]),
                    frame,
                    self.policy["circular_min_points"],
                    self.policy["circular_max_residual_ratio"],
                    self.diagonal,
                )
                if circ is not None:
                    pred = circ
            max_drift = self.policy.get("max_prediction_drift_ratio", 0.35) * self.diagonal
            pred = _clamp_to_rim(pred, center(s["last_box"]), self.arena_circle, max_drift)
            predicted.append(pred)
        # A single observed box covering both recently observed predictions is
        # ambiguous. Native ByteTrack IDs cannot carry player ownership through
        # this merge: retaining their spatial slots does not re-anchor X/B.
        if len(self.slots) == 2 and len(kept) == 1:
            box = kept[0][0]["box"]
            margin = self.policy["merge_center_margin_ratio"] * math.hypot(
                box[2] - box[0], box[3] - box[1])
            if all(
                (frame - slot["last_observed_frame"]) / self.fps <= self.policy["merge_observation_max_gap_s"]
                and box[0] - margin <= point[0] <= box[2] + margin
                and box[1] - margin <= point[1] <= box[3] + margin
                for slot, point in zip(self.slots, predicted, strict=True)
            ):
                # Only two known identities can be re-anchored after the split;
                # an identity already lost stays lost.
                if self.pending_merge is None and not self.identity_unknown_since:
                    self.pending_merge = {"start": frame, "slots": before}
                for slot in self.slots:
                    self.identity_unknown_since.setdefault(slot["entity_id"], frame)
        self.previous = frame
        assigned = {}
        forced = None
        # Decide at the first frame with two disjoint boxes: overlapping crops
        # mix both tops and cannot carry either one's appearance.
        if (
            self.pending_merge is not None
            and len(self.slots) == 2
            and len(kept) == 2
            and iou(kept[0][0]["box"], kept[1][0]["box"]) == 0
        ):
            episode, forced = self._reanchor(frame, kept, predicted)
            self.merge_episodes.append(episode)
            if forced is not None:
                start = self.pending_merge["start"]
                for s, snapshot in zip(self.slots, self.pending_merge["slots"], strict=True):
                    # The merged box may have fed one slot's appearance.
                    if snapshot["appearance"] is not None:
                        s["appearance"] = snapshot["appearance"].copy()
                    del self.identity_unknown_since[s["entity_id"]]
                    self.identity_unknown_intervals.setdefault(
                        s["entity_id"], []
                    ).append([start, frame - 1])
            self.pending_merge = None
        if forced is not None:
            assigned = dict(enumerate(forced))
        elif self.slots:
            costs = np.full(
                (len(self.slots), len(kept) + len(self.slots)),
                self.policy["unmatched_cost"],
                dtype=float,
            )
            for i, s in enumerate(self.slots):
                for j, (d, a) in enumerate(kept):
                    distance = (
                        float(np.linalg.norm(predicted[i] - center(d["box"])))
                        / self.diagonal
                    )
                    appearance = 0.0
                    if a is not None and s["appearance"] is not None:
                        if a.shape != s["appearance"].shape:
                            raise ValueError("appearance dimensions changed")
                        appearance = float(
                            cv2.compareHist(
                                a, s["appearance"], cv2.HISTCMP_BHATTACHARYYA
                            )
                        )
                    if (
                        distance > self.policy["position_gate_ratio"]
                        or appearance > self.policy["appearance_gate"]
                        or (
                            d["native_id"] in self.native_owners
                            and self.native_owners[d["native_id"]] != s["entity_id"]
                        )
                    ):
                        costs[i, j] = self.policy["unmatched_cost"] + 1
                    else:
                        costs[i, j] = (
                            distance * self.policy["position_weight"]
                            + appearance * self.policy["appearance_weight"]
                            - (
                                self.policy["native_id_bonus"]
                                if d["native_id"] == s["native_id"]
                                else 0
                            )
                        )
            ii, jj = linear_sum_assignment(costs)
            assigned = {
                int(i): int(j)
                for i, j in zip(ii, jj, strict=True)
                if j < len(kept) and costs[i, j] < self.policy["unmatched_cost"]
            }
        reseed = self._eliminate(frame, kept, assigned, predicted)
        used = set(assigned.values())
        for i, s in enumerate(self.slots):
            if i not in assigned:
                state = s["kalman"].statePost.copy()
                state[2:] *= self.policy["missing_velocity_decay"]
                s["kalman"].statePost = state
        observed = []
        for i, j in assigned.items():
            d, appearance = kept[j]
            s = self.slots[i]
            gap = frame - s["last_observed_frame"]
            if forced is None and gap >= round(self.policy["reacquire_min_gap_s"] * self.fps):
                # Ordinary gates and two-top elimination establish space only.
                # A stale template or reused native ID cannot restore the player;
                # only the explicit, validated post-merge re-anchor can do that.
                self.identity_unknown_since.setdefault(s["entity_id"], frame)
                self.reacquisitions.append({
                    "frame": frame,
                    "entity_id": s["entity_id"],
                    "native_id": d["native_id"],
                    "unobserved_frames": gap,
                })
            s["kalman"].correct(center(d["box"]).reshape(2, 1))
            s["native_id"] = d["native_id"]
            s["last_box"] = list(d["box"])
            s["last_observed_frame"] = frame
            s["history"].append((frame, center(d["box"])))
            self._own(d["native_id"], s["entity_id"])
            if i == reseed and appearance is not None:
                # The stale tracking template locked this top out; restart only it.
                # clean_appearance keeps the contradicting identity evidence.
                s["appearance"] = appearance.copy()
            if appearance is not None and not any(
                iou(d["box"], other["box"]) >= self.policy["duplicate_iou"]
                and self.native_owners.get(other["native_id"])
                not in (None, s["entity_id"])
                for other, _ in kept
                if other is not d
            ):
                if s["appearance"] is None:
                    s["appearance"] = appearance.copy()
                else:
                    alpha = self.policy["appearance_update_alpha"]
                    s["appearance"] = (1 - alpha) * s["appearance"] + alpha * appearance
            if (
                appearance is not None
                and self.pending_merge is None
                and s["entity_id"] not in self.identity_unknown_since
                and all(iou(d["box"], other["box"]) == 0 for other, _ in kept if other is not d)
            ):
                clean = s["clean_appearance"]
                alpha = self.policy["appearance_update_alpha"]
                s["clean_appearance"] = (
                    appearance.copy()
                    if clean is None
                    else (1 - alpha) * clean + alpha * appearance
                )
            observed.append(
                {
                    **d,
                    "box": list(d["box"]),
                    "entity_id": s["entity_id"],
                    "frame": frame,
                    "identity_valid": s["entity_id"] not in self.identity_unknown_since,
                }
            )
        for j, (d, a) in enumerate(kept):
            if j in used:
                continue
            if (
                d["native_id"] not in self.native_owners
                and len(self.slots) < 2
                and all(
                    np.linalg.norm(center(d["box"]) - center(s["last_box"]))
                    > self.policy["initial_separation_ratio"]
                    * max(
                        math.hypot(
                            d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]
                        ),
                        math.hypot(
                            s["last_box"][2] - s["last_box"][0],
                            s["last_box"][3] - s["last_box"][1],
                        ),
                    )
                    for s in self.slots
                )
            ):
                s = self._create(
                    d,
                    a,
                    all(iou(d["box"], other["box"]) == 0 for other, _ in kept if other is not d),
                )
                observed.append(
                    {
                        **d,
                        "box": list(d["box"]),
                        "entity_id": s["entity_id"],
                        "frame": frame,
                        "identity_valid": s["entity_id"] not in self.identity_unknown_since,
                    }
                )
            else:
                # Mostly inside exactly one observed top and touching no other:
                # a box reaching into both tops stays an ambiguous observation, and
                # so does a native the other top owns.
                inside = next(
                    (
                        o for o in observed
                        if len(observed) == 2
                        and self.native_owners.get(d["native_id"]) in (None, o["entity_id"])
                        and containment(d["box"], o["box"]) >= self.policy["duplicate_containment"]
                        and all(
                            containment(d["box"], other["box"]) == 0
                            for other in observed if other is not o
                        )
                    ),
                    None,
                )
                rejected.append(
                    {
                        **d,
                        "frame": frame,
                        "reason": "unassigned_two_target_capacity_or_gate",
                    }
                    if inside is None
                    else {
                        **d,
                        "frame": frame,
                        "reason": "duplicate_box",
                        "duplicate_of": inside["native_id"],
                        "duplicate_kind": "contained",
                    }
                )
        for d in rejected:
            if (
                d["reason"] == "duplicate_box"
                and d.get("duplicate_kind") != "contained"
                and d["native_id"] not in self.native_owners
                and any(r["native_id"] == d["duplicate_of"] for r in observed)
            ):
                self._own(d["native_id"], self.native_owners[d["duplicate_of"]])
        return {
            "observed": sorted(observed, key=lambda r: r["entity_id"]),
            "unassigned": rejected,
        }


def continuous_points(observed):
    """Linear display bridges between actual endpoints; never extra evidence."""
    by_entity = {}
    for r in observed:
        by_entity.setdefault(r["entity_id"], []).append(r)
    result = []
    for rows in by_entity.values():
        rows = sorted(rows, key=lambda r: r["frame"])
        for i, r in enumerate(rows):
            if i and r["frame"] <= rows[i - 1]["frame"]:
                raise ValueError("duplicate entity source frame")
            if i:
                a = rows[i - 1]
                for f in range(a["frame"] + 1, r["frame"]):
                    ratio = (f - a["frame"]) / (r["frame"] - a["frame"])
                    result.append(
                        {
                            "entity_id": r["entity_id"],
                            "frame": f,
                            "box": [
                                x + (y - x) * ratio
                                for x, y in zip(a["box"], r["box"], strict=True)
                            ],
                            "kind": "interpolated",
                            "adjudication_evidence": False,
                            "endpoint_frames": [a["frame"], r["frame"]],
                        }
                    )
            result.append({**r, "kind": "observed", "adjudication_evidence": True})
    return sorted(result, key=lambda r: (r["frame"], r["entity_id"]))
