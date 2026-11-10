"""Click manual truth for the 5 judged stadium zones on one video frame (user ruling 2026-11-11:
stadium, battle area, both Over pockets, Xtreme; launch zones and the lid are not clicked).

Read only on the video and calib/. Usage:
    uv run python scripts/stadium_truth.py <video> --at <seconds> [--out data/stadium-truth-20261111]

Click order: stadium (whole stadium incl. lid) -> battle_area (lower floor) -> over_left ->
over_right -> xtreme. Enter finishes a polygon, u undoes the last point, Esc aborts.
Writes <out>/<video stem>.json (zones in full-resolution frame px) and <video stem>.png.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from bxjudge import PROJECT_ROOT
from bxjudge.calibration import collect_polygons, grab_frame
from bxjudge.stadium_finder import TRUTH_ZONES

DEFAULT_OUT = PROJECT_ROOT / "data" / "stadium-truth-20261111"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video", type=Path)
    p.add_argument("--at", type=float, required=True, help="seconds into the video")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)
    out = a.out.expanduser().resolve()
    calib = (PROJECT_ROOT / "calib").resolve()
    if out == calib or calib in out.parents:
        raise SystemExit(f"{out}: truth must stay outside calib/")
    video = a.video.expanduser()
    frame = grab_frame(video, a.at)
    zones = collect_polygons(frame, f"truth {video.stem} @ {a.at:.2f}s", TRUTH_ZONES)
    h, w = frame.shape[:2]
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "video": video.name,
        "video_sha256": _sha256(video),
        "at_s": a.at,
        "frame_size": [w, h],
        "zones": {k: [[float(x), float(y)] for x, y in v] for k, v in zones.items()},
    }
    (out / f"{video.stem}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    cv2.imwrite(str(out / f"{video.stem}.png"), frame)
    print(f"saved {out / (video.stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
