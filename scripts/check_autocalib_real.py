"""Opt-in real-clip check of the auto-alignment, outside pytest.

Run: uv run python scripts/check_autocalib_real.py <video> --calib calib [--out <scratch dir>]

Decodes the same calibration-selection frame the pipeline uses, aligns every manual
calibration in ``--calib`` to it and prints each score, the estimated transform of the best
one and the preview path. Previews and snapshots go ONLY under ``--out`` (default: a fresh
temporary directory), never into ``--calib``; the calibration folder is only read.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from bxjudge import DEFAULT_AUTOCALIB_CONFIG, manifest
from bxjudge.autocalib import estimate_alignment, load_autocalib_config
from bxjudge.calibration import load_calibration
from bxjudge.calibration_store import (
    select_snapshot,
    validate_output_paths,
    write_preview,
)
from bxjudge.config import load_thresholds
from bxjudge.pipeline import first_second_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--calib", type=Path, required=True, help="manual calibration folder (read only)")
    parser.add_argument("--out", type=Path, default=None,
                        help="scratch folder for snapshots/previews (default: new temp dir)")
    parser.add_argument("--thresholds", type=Path, default=Path("config/thresholds.yaml"))
    parser.add_argument("--autocalib", type=Path, default=DEFAULT_AUTOCALIB_CONFIG)
    args = parser.parse_args(argv)

    calib_dir = args.calib.expanduser()
    if not calib_dir.is_dir():
        print(f"error: {calib_dir}: --calib is not a folder", file=sys.stderr)
        return 2
    out_dir = (args.out.expanduser() if args.out
               else Path(tempfile.mkdtemp(prefix="check_autocalib_")))
    validate_output_paths(out_dir, calib_dir)  # refuses an --out inside/over --calib
    th = load_thresholds(args.thresholds.expanduser())
    cfg = load_autocalib_config(args.autocalib.expanduser())

    info = manifest.probe(args.video.expanduser())
    frame = first_second_frame(info, th)
    print(f"video {info.path.name}: frame {frame.shape[1]}x{frame.shape[0]}, "
          f"min_score {cfg.min_score}, auto_review_score {cfg.auto_review_score}")

    best = None
    for json_path in sorted(calib_dir.glob("*.json")):
        cal = load_calibration(json_path)
        img = cv2.imread(cal.ref_frame, cv2.IMREAD_COLOR) if cal.ref_frame else None
        if img is None:
            print(f"template {json_path.stem}: no readable reference frame, skipped")
            continue
        al = estimate_alignment(frame, cal, img, cfg)
        if al is None:
            print(f"template {json_path.stem}: rejected (no in-bounds alignment >= min_score)")
            continue
        print(f"template {json_path.stem}: score {al.score:.3f}")
        if best is None or al.score > best[1].score:
            best = (json_path.stem, al)
    if best is None:
        print("result: REJECTED (recalibrate)")
        return 0
    name, al = best
    print(f"best template {name}: score {al.score:.3f}  shift {al.shift_frac:.4f} of width  "
          f"rotation {al.rotation_deg:+.2f} deg  scale {al.scale:.4f}  "
          f"perspective {al.perspective:.2e}")
    print("homography (template px -> frame px):")
    print(np.array2string(np.array(al.homography), precision=5, suppress_small=True))
    snap = select_snapshot(frame, calib_dir, out_dir, th, cfg)
    if snap is None:
        print("select_snapshot: no accepted calibration")
        return 0
    band = "needs review" if snap.score < cfg.auto_review_score else "auto"
    print(f"selected {snap.kind} snapshot {snap.id} (template {snap.template_id}, "
          f"score {snap.score:.3f}, {band})")
    print(f"preview: {write_preview(out_dir, snap)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
