"""Targeted mask review for a known pair around a possible launch."""

from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass

import cv2
import numpy as np

BEFORE_FRAMES = 8
AFTER_FRAMES = 5
SUPPORT = 0.70
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


@dataclass(frozen=True)
class ReleaseProbe:
    round: int
    track_id: int
    native_id: int
    launcher_id: int
    anchor: int
    box_release: int | None
    frames: tuple[int, ...]


def build_release_probes(number, physical, sides, segments, proofs):
    """Use a core release or one unambiguous initial mount for a known side."""
    record = physical.record
    pair_frame = record["pair_confirmed_at_frame"]
    if pair_frame is None or len(physical.tracks) != 2:
        return []
    if {sides.get(t.track_id) for t in physical.tracks} != {"X", "B"}:
        return []

    observed = defaultdict(dict)
    for row in record["observations"]:
        if row["identity_valid"]:
            observed[row["entity_id"]][row["frame"]] = row["native_id"]

    probes = []
    for entity, track in enumerate(physical.tracks, 1):
        proof = proofs.get(track.track_id)
        if proof is not None:
            native_id, launcher_id = proof["native_id"], proof["launcher_id"]
            anchor = box_release = proof["frame"]
        else:
            mounts = [
                (native, s)
                for native in physical.members[track.track_id]
                for s in segments.get(native, ())
                if s.state == "mounted"
                and s.launcher_id is not None
                and pair_frame - BEFORE_FRAMES
                <= s.last_frame
                <= pair_frame + AFTER_FRAMES
                and observed[entity].get(s.last_frame) == native
            ]
            if len(mounts) != 1:
                continue
            native_id, mount = mounts[0]
            launcher_id, anchor, box_release = (
                mount.launcher_id,
                mount.last_frame + 1,
                None,
            )
        frames = tuple(
            f
            for f in range(anchor - BEFORE_FRAMES, anchor + AFTER_FRAMES + 1)
            if observed[entity].get(f) == native_id
        )
        if frames:
            probes.append(
                ReleaseProbe(
                    number,
                    track.track_id,
                    native_id,
                    launcher_id,
                    anchor,
                    box_release,
                    frames,
                )
            )
    return probes


def release_candidate(ratios):
    # Four exact source frames, including two observations on each side of 50%.
    for first in sorted(ratios):
        span = [ratios.get(f) for f in range(first, first + 4)]
        if all(value is not None for value in span) and (
            span[0] < 0.5 and span[1] < 0.5 <= span[2] and span[3] >= 0.5
        ):
            return first + 2, first + 3
    return None


def review_release_probes(probes, tracks, frames, measure):
    wanted = defaultdict(list)
    ids = {tid for p in probes for tid in (p.native_id, p.launcher_id)}
    boxes = {
        t.track_id: {d.frame: d.box for d in t.dets}
        for t in tracks
        if t.track_id in ids
    }
    ratios = {p: {} for p in probes}
    for p in probes:
        bey, launcher = boxes.get(p.native_id, {}), boxes.get(p.launcher_id, {})
        for frame in p.frames:
            if frame in bey and frame in launcher:
                wanted[frame].append((p, bey[frame], launcher[frame]))
    if wanted:
        last = max(wanted)
        with closing(iter(frames())) as source:
            for frame, image in source:
                for probe, bey, launcher in wanted.get(frame, ()):
                    ratios[probe][frame] = measure(image, bey, launcher)
                if frame >= last:
                    break
    reviews = defaultdict(dict)
    for p, measurements in ratios.items():
        candidate = release_candidate(measurements)
        status = "insufficient"
        if candidate:
            status = (
                "candidate_without_box_release"
                if p.box_release is None
                else "agrees"
                if abs(candidate[0] - p.box_release) <= 1
                else "disagrees"
            )
        reviews[p.round][p.track_id] = {
            "native_id": p.native_id,
            "launcher_id": p.launcher_id,
            "anchor_frame": p.anchor,
            "box_release_frame": p.box_release,
            "frame": candidate[0] if candidate else None,
            "confirmed_at_frame": candidate[1] if candidate else None,
            "status": status,
            "observations": {f: measurements.get(f) for f in p.frames},
        }
    return dict(reviews)


class SamLaunchMaskReviewer:
    def __init__(self, weights):
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ValueError(
                "launch mask review requires the SAM 2.1 runtime on PYTHONPATH"
            ) from exc
        if not torch.cuda.is_available():
            raise ValueError("launch mask review requires CUDA")
        self.torch = torch
        self.predictor = SAM2ImagePredictor(
            build_sam2(
                MODEL_CONFIG, str(weights), device="cuda", apply_postprocessing=False
            )
        )

    def _mask(self, box):
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(box, dtype=np.float32), multimask_output=True
        )
        x0, y0, x1, y1 = np.rint(box).astype(int)
        for index in np.argsort(-scores):
            mask = masks[index].astype(bool, copy=False)
            area = int(mask.sum())
            if area and mask[max(0, y0) : y1, max(0, x0) : x1].sum() >= SUPPORT * area:
                return mask
        return None

    def __call__(self, image, bey_box, launcher_box):
        with (
            self.torch.inference_mode(),
            self.torch.autocast("cuda", dtype=self.torch.bfloat16),
        ):
            self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            bey = self._mask(bey_box)
            launcher = self._mask(launcher_box)
        if bey is None or launcher is None:
            return None
        return 1 - int((bey & launcher).sum()) / int(bey.sum())
