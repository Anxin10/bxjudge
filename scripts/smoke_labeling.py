"""Opt-in real SAM GPU smoke, outside pytest; no external footage or accuracy claim.
Run: uv run python scripts/smoke_labeling.py --out <scratch directory>
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
video = out / "synthetic.mp4"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (256, 192))
for index in range(4):
    frame = np.zeros((192, 256, 3), np.uint8)
    cv2.circle(frame, (80 + index * 8, 100), 20, (0, 160, 255), -1)
    writer.write(frame)
writer.release()
cfg = yaml.safe_load(Path("config/labeling.yaml").read_text())
cfg.update(
    input_folder=str(out), output_dir=str(out / "labels"), window_frames=2, per_video=4
)
config = out / "labeling.yaml"
config.write_text(yaml.safe_dump(cfg))
for action in ["sample", "prelabel"]:
    subprocess.run(
        ["uv", "run", "--locked", "bxjudge", "label", action, "--config", str(config)],
        check=True,
    )
tasks = json.loads((out / "labels/tasks.json").read_text())
assert [t["data"]["frame"] for t in tasks] == [0, 1, 2, 3]
assert all(Path(t["data"]["image_path"]).is_file() for t in tasks)
print(
    json.dumps(
        {
            "selected_images": len(tasks),
            "predicted_boxes": sum(len(t["predictions"][0]["result"]) for t in tasks),
        }
    )
)
