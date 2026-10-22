"""Detector/training policy, separate from adjudication thresholds."""

import math
import os
from pathlib import Path

import yaml

CLASSES = ("bey", "bey_part", "launcher")
KINDS = ("rfdetr", "yolo26")


def load_policy(path: Path | str = "config/detection.yaml") -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    required = {
        "confidence",
        "evaluation_confidence",
        "device",
        "selection_tie_ap",
        "latency_warmup_frames",
        "tracking",
        "training",
    }
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError("detector config has missing or unknown keys")
    for key in ("confidence", "evaluation_confidence", "selection_tie_ap"):
        value = cfg[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"invalid {key}")
    if (
        type(cfg["latency_warmup_frames"]) is not int
        or cfg["latency_warmup_frames"] < 0
    ):
        raise ValueError("latency_warmup_frames must be a nonnegative integer")
    if type(cfg["training"]["epochs"]) is not int or cfg["training"]["epochs"] <= 0:
        raise ValueError("epochs must be a positive integer")
    tracking = cfg["tracking"]
    tracking_keys = {
        "lost_track_buffer",
        "track_activation_threshold",
        "minimum_consecutive_frames",
        "minimum_iou_threshold",
        "high_conf_det_threshold",
    }
    if not isinstance(tracking, dict) or set(tracking) != tracking_keys:
        raise ValueError(
            f"tracking requires exactly {sorted(tracking_keys)}; got {tracking!r}"
        )
    if (
        type(tracking["lost_track_buffer"]) is not int
        or tracking["lost_track_buffer"] < 0
    ):
        raise ValueError("tracking lost_track_buffer must be a nonnegative integer")
    if type(tracking["minimum_consecutive_frames"]) is not int or tracking[
        "minimum_consecutive_frames"
    ] not in (1, 2):
        raise ValueError(
            "tracking minimum_consecutive_frames must be 1 or 2 for observation retention"
        )
    for key in (
        "track_activation_threshold",
        "minimum_iou_threshold",
        "high_conf_det_threshold",
    ):
        value = tracking[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"tracking {key} must be a finite number in [0, 1]")
    return cfg


def disable_uploads() -> None:
    """Disable framework analytics and external experiment/media integrations."""
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["YOLO_OFFLINE"] = "true"
    # Read before Ultralytics import; no changes to the user's shared settings.
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(Path("runs/ultralytics-settings").resolve())
    )
    from ultralytics import settings

    settings.update(
        {
            key: False
            for key in (
                "sync",
                "clearml",
                "comet",
                "dvc",
                "mlflow",
                "raytune",
                "tensorboard",
                "wandb",
                "vscode_msg",
            )
        }
    )
