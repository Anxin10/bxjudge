"""Manual Task 13 GPU/CLI smoke. All generated calibration is synthetic-only.

uv run python tests/smoke_task13.py runs/task13-smoke-20261022
Uses existing synthetic-trained weights; never establishes real accuracy.
"""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import cv2
import yaml
from helpers import cal_with, make_calibration

from bxjudge.calibration import save_calibration
from bxjudge.manifest import probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    selection = Path("runs/task12-smoke-20261022-final/fix1/selected.json").resolve()
    source = Path("runs/task12-smoke-20261022-final/dataset/images/val/1.png").resolve()
    image = cv2.imread(str(source))
    h, w = image.shape[:2]
    calib_dir = root / "synthetic-calib"
    calib_dir.mkdir()
    c = cal_with(image, calib_dir, "synthetic-only")
    template = make_calibration()
    for zone in (
        "stadium",
        "lid",
        "battle_area",
        "over_left",
        "over_right",
        "xtreme",
        "launch_left",
        "launch_right",
    ):
        setattr(
            c, zone, [(x * w / 640, y * h / 360) for x, y in getattr(template, zone)]
        )
    save_calibration(c, calib_dir / "synthetic-only.json")
    folder = root / "synthetic-videos"
    folder.mkdir()
    video = folder / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-loop",
            "1",
            "-framerate",
            "30",
            "-i",
            str(source),
            "-t",
            "8",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    cfg = yaml.safe_load(Path("config/detection.yaml").read_text())
    # Explicit smoke-only operating point, no production-policy modification.
    cfg["confidence"] = cfg["candidate_confidence"] = cfg["evaluation_confidence"]
    for key in ("track_activation_threshold", "high_conf_det_threshold"):
        cfg["tracking"][key] = cfg["evaluation_confidence"]
    config = root / "synthetic-detection.yaml"
    config.write_text(yaml.safe_dump(cfg))
    samples, stop = [], threading.Event()

    def monitor():
        while not stop.is_set():
            samples.append(
                {
                    "time": time.time(),
                    "sample": subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    ).strip(),
                }
            )
            stop.wait(0.25)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    env = {
        **os.environ,
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "YOLO_OFFLINE": "true",
    }
    commands = []
    try:
        batches = [("synthetic", folder, calib_dir, config)]
        if not args.synthetic_only:
            batches.append(
                (
                    "real",
                    Path("/home/<user>/bx/videos/20261001"),
                    root / "no-real-calibration",
                    Path("config/detection.yaml"),
                )
            )
        for name, inputs, calibration, config_path in batches:
            cmd = [
                "uv",
                "run",
                "bxjudge",
                "run",
                str(inputs),
                "--calib",
                str(calibration),
                "--out",
                str(root / f"{name}-results"),
                "--selection",
                str(selection),
                "--thresholds",
                str(Path("config/thresholds.yaml").resolve()),
                "--config",
                str(config_path),
                "--limit",
                "3",
            ]
            commands.append(cmd)
            with (root / f"{name}.log").open("w") as log:
                subprocess.run(
                    cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env
                )
    finally:
        stop.set()
        thread.join()
        (root / "commands.json").write_text(json.dumps(commands, indent=2))
        (root / "gpu-memory.json").write_text(json.dumps(samples, indent=2))
        (root / "gpu-summary.json").write_text(
            json.dumps(
                {
                    "samples": len(samples),
                    "max_total_device_used_mib": max(
                        int(s["sample"].split(",")[0]) for s in samples
                    ),
                },
                indent=2,
            )
        )
    synthetic = json.loads((root / "synthetic-results/rounds.json").read_text())
    real = []
    if not args.synthetic_only:
        real = json.loads((root / "real-results/rounds.json").read_text())
    assert synthetic and all(
        "synthetic_detector" in r["review_reasons"] for r in synthetic
    )
    assert list((root / "synthetic-results/trajectories").glob("*.png"))
    assert (
        args.synthetic_only
        or len(real) == 3
        and all("recalibrate" in r["review_reasons"] for r in real)
    )
    assert not Path("weights/selected.json").exists()
    metadata = {
        r["video"]: vars(probe(Path("/home/<user>/bx/videos/20261001") / r["video"]))
        for r in real
    }
    (root / "real-metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    print(
        json.dumps(
            {
                "synthetic_rounds": len(synthetic),
                "real_review_artifacts": len(real),
                "root": str(root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
