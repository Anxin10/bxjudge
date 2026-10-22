"""Manual GPU wiring evidence; run via uv run python tests/smoke_task12.py ROOT.

ROOT must be a fresh WSL ext4 directory under runs/. Produces only synthetic evidence.
Training explicitly selects the smoke policy (1 epoch, 384, no augmentation);
comparison runs at the production inference size from config/thresholds.yaml.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

SMOKE_POLICY = str(Path("config/detection_smoke.yaml").resolve())


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    if root.exists():
        raise ValueError("smoke output must not already exist")
    root.mkdir(parents=True)
    dataset = root / "dataset"
    (dataset / "annotations").mkdir(parents=True)
    rng = np.random.default_rng(42)
    for split, count in (("train", 8), ("val", 4)):
        folder = dataset / "images" / split
        folder.mkdir(parents=True)
        data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i + 1, "name": name}
                for i, name in enumerate(("bey", "bey_part", "launcher"))
            ],
        }
        for i in range(count):
            pixels = rng.integers(0, 35, (192, 320, 3), dtype=np.uint8)
            image_id = i + 1
            data["images"].append(
                {
                    "id": image_id,
                    "width": 320,
                    "height": 192,
                    "file_name": f"images/{split}/{image_id}.png",
                }
            )
            for cls, color in enumerate(
                ((20, 20, 230), (20, 230, 20), (230, 20, 20)), 1
            ):
                x = 15 + (cls - 1) * 95 + int(rng.integers(0, 10))
                y = 50 + int(rng.integers(0, 20))
                w, h = (46, 46) if cls < 3 else (65, 28)
                cv2.rectangle(pixels, (x, y), (x + w - 1, y + h - 1), color, -1)
                data["annotations"].append(
                    {
                        "id": len(data["annotations"]) + 1,
                        "image_id": image_id,
                        "category_id": cls,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
            cv2.imwrite(str(folder / f"{image_id}.png"), pixels)
        (dataset / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(data)
        )
    (dataset / "provenance.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "seed": 42,
                "purpose": "Task12 GPU wiring only; no accuracy claim",
                "generator": str(Path(__file__).resolve()),
                "train_images": 8,
                "val_images": 4,
                "classes": ["bey", "bey_part", "launcher"],
            },
            indent=2,
        )
    )

    stop = threading.Event()
    samples = []

    def monitor():
        while not stop.is_set():
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            samples.append({"time": time.time(), "sample": raw})
            with (root / "gpu-memory.json").open("w") as file:
                json.dump(samples, file, indent=2)
            stop.wait(0.25)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    env = dict(
        os.environ,
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
        YOLO_OFFLINE="true",
        YOLO_CONFIG_DIR=str(root / "ultralytics-settings"),
    )
    commands = [
        [
            "bxjudge",
            "train",
            "--kind",
            "rfdetr",
            "--dataset",
            str(dataset),
            "--out",
            str(root / "rfdetr"),
            "--training-policy",
            SMOKE_POLICY,
        ],
        [
            "bxjudge",
            "train",
            "--kind",
            "yolo26",
            "--dataset",
            str(dataset),
            "--out",
            str(root / "yolo26"),
            "--training-policy",
            SMOKE_POLICY,
        ],
        [
            "bxjudge",
            "compare",
            "--dataset",
            str(dataset),
            "--rfdetr",
            str(root / "rfdetr/fit/checkpoint_best_total.pth"),
            "--yolo26",
            str(root / "yolo26/fit/weights/best.pt"),
            "--output",
            str(root / "compare.json"),
            "--selection-out",
            str(root / "selected.json"),
        ],
    ]
    (root / "commands.json").write_text(json.dumps(commands, indent=2))
    try:
        for command, name in zip(
            commands, ("rfdetr", "yolo26", "compare"), strict=True
        ):
            print("Running", " ".join(command), flush=True)
            with (root / f"{name}.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True
                )
            print(name, "completed", flush=True)
    finally:
        stop.set()
        thread.join()
        maximum = max(float(item["sample"].split(",")[1]) for item in samples)
        summary = {
            "nvidia_smi_peak_total_used_mib": maximum,
            "baseline_mib": float(samples[0]["sample"].split(",")[1]),
            "sample_count": len(samples),
            "limit_mib": 21 * 1024,
            "below_limit": maximum < 21 * 1024,
        }
        (root / "gpu-summary.json").write_text(json.dumps(summary, indent=2))
        print(summary, flush=True)


if __name__ == "__main__":
    main()
