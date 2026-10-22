"""Detector/training policy, separate from adjudication thresholds."""

import math
import os
from pathlib import Path

import yaml

from .config import config_loader

CLASSES = ("bey", "bey_part", "launcher")
KINDS = ("rfdetr", "yolo26")

# Framework size identities (not policy): RF-DETR Nano needs patch_size 16 *
# num_windows 2; Ultralytics check_imgsz rounds to the YOLO26 max stride.
RFDETR_BLOCK = 32
YOLO_STRIDE = 32
TRAINING_POLICIES = ("production", "smoke")
# Production derives the training size from thresholds.yaml infer_long_edge (R15/R57).
DERIVED_IMAGE_SIZE = "infer_long_edge"
_TRAINING_KEYS = {"policy", "image_size", "epochs", "rfdetr", "yolo26"}
# Supplied by train() itself (size, data, output, schedule); never in the policy.
_RESERVED_FIT = {
    "resolution",
    "imgsz",
    "epochs",
    "device",
    "data",
    "project",
    "name",
    "dataset_dir",
    "dataset_file",
    "output_dir",
    "class_names",
}


def _positive_int(value) -> bool:
    return type(value) is int and value > 0


def validate_training(training) -> dict:
    """Validate one training policy block (production or smoke)."""
    if not isinstance(training, dict) or set(training) != _TRAINING_KEYS:
        raise ValueError(
            f"training requires exactly {sorted(_TRAINING_KEYS)}; got {training!r}"
        )
    policy, size = training["policy"], training["image_size"]
    if policy not in TRAINING_POLICIES:
        raise ValueError(f"training policy must be one of {TRAINING_POLICIES}")
    if policy == "production" and size != DERIVED_IMAGE_SIZE:
        raise ValueError(
            f"production training image_size must be {DERIVED_IMAGE_SIZE!r} "
            "(derived from thresholds.yaml)"
        )
    if policy == "smoke" and not _positive_int(size):
        raise ValueError("smoke training image_size must be a positive integer")
    if not _positive_int(training["epochs"]):
        raise ValueError("training epochs must be a positive integer")
    for kind in KINDS:
        block = training[kind]
        # RF-DETR model: constructor kwargs mapping; YOLO26 model: weights path.
        model_type = dict if kind == "rfdetr" else str
        if (
            not isinstance(block, dict)
            or set(block) != {"model", "fit"}
            or not isinstance(block["fit"], dict)
            or not isinstance(block["model"], model_type)
            or (model_type is str and not block["model"])
        ):
            raise ValueError(
                f"training {kind} requires exactly a fit mapping and a "
                f"{'kwargs mapping' if model_type is dict else 'weights path'} model"
            )
        reserved = _RESERVED_FIT & set(block["fit"])
        if kind == "rfdetr":
            reserved |= _RESERVED_FIT & set(block["model"])
            batches = ("batch_size", "grad_accum_steps")
        else:
            batches = ("batch",)
        if reserved:
            raise ValueError(
                f"training {kind} must not set {sorted(reserved)} "
                "(image size comes only from training.image_size)"
            )
        for key in batches:
            if not _positive_int(block["fit"].get(key)):
                raise ValueError(
                    f"training {kind} fit.{key} must be an explicit positive integer"
                )
    return training


@config_loader
def load_training_policy(path: Path | str) -> dict:
    """Explicitly selected training policy file holding only a training block."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {"training"}:
        raise ValueError(f"training policy file must contain only 'training': {path}")
    return validate_training(data["training"])


def _nearest_multiple(edge: int, divisor: int) -> int:
    return max(divisor, int(edge / divisor + 0.5) * divisor)


def resolve_training(training: dict, infer_long_edge: int) -> dict:
    """Framework keyword arguments with the image size each framework accepts."""
    training = validate_training(training)
    if not _positive_int(infer_long_edge):
        raise ValueError("infer_long_edge must be a positive integer")
    edge = (
        infer_long_edge
        if training["policy"] == "production"
        else training["image_size"]
    )
    sizes = {
        "rfdetr": _nearest_multiple(edge, RFDETR_BLOCK),
        "yolo26": _nearest_multiple(edge, YOLO_STRIDE),
    }
    return {
        "policy": training["policy"],
        "epochs": training["epochs"],
        "image_size": sizes,
        "rfdetr": {
            "model": {**training["rfdetr"]["model"], "resolution": sizes["rfdetr"]},
            "fit": dict(training["rfdetr"]["fit"]),
        },
        "yolo26": {
            "model": training["yolo26"]["model"],
            "fit": {**training["yolo26"]["fit"], "imgsz": sizes["yolo26"]},
        },
    }


@config_loader
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
    validate_training(cfg["training"])
    if cfg["training"]["policy"] != "production":
        # Smoke settings live only in their own explicitly selected file.
        raise ValueError("detector config training policy must be production")
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
