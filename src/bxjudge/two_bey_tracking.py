"""Two physical top identities per round; raw observations and display estimates differ."""

import math
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
    for k in keys - {"max_targets", "appearance_bins"}:
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


class TwoBeyTracker:
    """Bounded one-to-one assignment, distinct from detector and referee evidence."""

    def __init__(self, frame_size, fps, policy):
        if (
            len(frame_size) != 2
            or any(type(v) is not int or v <= 0 for v in frame_size)
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("invalid source coordinates or FPS")
        self.frame_size = tuple(frame_size)
        self.fps = fps
        self.policy = dict(policy)
        self.slots = []
        self.previous = -1
        self.diagonal = math.hypot(*frame_size)
        self.native_owners = {}

    @property
    def entity_count(self):
        return len(self.slots)

    def _create(self, d, appearance):
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
            "native_id": d["native_id"],
            "last_box": list(d["box"]),
        }
        self.slots.append(slot)
        self.native_owners[d["native_id"]] = slot["entity_id"]
        return slot

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
        predicted = []
        for s in self.slots:
            s["kalman"].transitionMatrix = np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float64,
            )
            predicted.append(s["kalman"].predict()[:2, 0])
        self.previous = frame
        assigned = {}
        if self.slots:
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
            s["kalman"].correct(center(d["box"]).reshape(2, 1))
            s["native_id"] = d["native_id"]
            s["last_box"] = list(d["box"])
            self.native_owners[d["native_id"]] = s["entity_id"]
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
            observed.append(
                {
                    **d,
                    "box": list(d["box"]),
                    "entity_id": s["entity_id"],
                    "frame": frame,
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
                s = self._create(d, a)
                observed.append(
                    {
                        **d,
                        "box": list(d["box"]),
                        "entity_id": s["entity_id"],
                        "frame": frame,
                    }
                )
            else:
                rejected.append(
                    {
                        **d,
                        "frame": frame,
                        "reason": "unassigned_two_target_capacity_or_gate",
                    }
                )
        for d in rejected:
            if (
                d["reason"] == "duplicate_box"
                and d["native_id"] not in self.native_owners
                and any(r["native_id"] == d["duplicate_of"] for r in observed)
            ):
                self.native_owners[d["native_id"]] = self.native_owners[
                    d["duplicate_of"]
                ]
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
