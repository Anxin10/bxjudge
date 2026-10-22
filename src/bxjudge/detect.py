"""RF-DETR Nano and YOLO26 adapters for OpenCV BGR frames."""

from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np

from .config import Thresholds, load_thresholds
from .detector_config import CLASSES, disable_uploads, load_policy
from .types import Detection


class Detector(Protocol):
    def __call__(self, frame: np.ndarray, frame_idx: int) -> list[Detection]: ...


def _check_frame(frame: np.ndarray, frame_idx: int) -> None:
    if type(frame_idx) is not int or frame_idx < 0:
        raise ValueError("frame_idx must be a nonnegative source frame index")
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected BGR uint8 HWC frame")


def _confidence(value: float | None, cfg: dict) -> float:
    conf = cfg["confidence"] if value is None else value
    if not np.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError("confidence must be in [0, 1]")
    return conf


class RFDETRDetector:
    def __init__(
        self,
        weights: Path,
        conf: float | None = None,
        *,
        config: Path | str = "config/detection.yaml",
        thresholds: Thresholds | Path | str = "config/thresholds.yaml",
    ):
        cfg = load_policy(config)
        self.conf = _confidence(conf, cfg)
        th = (
            thresholds
            if isinstance(thresholds, Thresholds)
            else load_thresholds(thresholds)
        )
        self.inference_long_edge = th.infer_long_edge
        if not Path(weights).is_file():
            raise FileNotFoundError(weights)
        disable_uploads()
        from rfdetr import RFDETRNano

        # The positional grid is built at the inference size (RF-DETR interpolates a
        # checkpoint's grid once at load), independent of any training policy block.
        self.model = RFDETRNano(
            pretrain_weights=str(Path(weights).resolve()),
            device=cfg["device"],
            resolution=self.inference_long_edge,
        )
        if tuple(self.model.class_names) != CLASSES:
            raise ValueError(f"RF-DETR checkpoint class names must be {CLASSES}")

    def __call__(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        _check_frame(frame, frame_idx)
        # RF-DETR predict documents RGB; decoder frames are BGR.
        result = self.model.predict(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            threshold=self.conf,
            include_source_image=False,
            # RF-DETR's native preprocessing resizes to an explicit H/W square,
            # independently of the checkpoint's training resolution.
            shape=(self.inference_long_edge, self.inference_long_edge),
        )
        detections = []
        for box, score, cid in zip(
            result.xyxy, result.confidence, result.class_id, strict=True
        ):
            # Installed RF-DETR marks logit slot num_classes as __background__.
            # It can exceed a low evaluation threshold; it is not a fourth class.
            if int(cid) == len(CLASSES):
                continue
            if not 0 <= cid < len(CLASSES):
                raise ValueError(f"unexpected RF-DETR class slot: {cid}")
            detections.append(
                Detection(
                    frame_idx, CLASSES[int(cid)], tuple(map(float, box)), float(score)
                )
            )
        return detections


class YOLODetector:
    def __init__(
        self,
        weights: Path,
        conf: float | None = None,
        *,
        config: Path | str = "config/detection.yaml",
        thresholds: Thresholds | Path | str = "config/thresholds.yaml",
    ):
        self.cfg = load_policy(config)
        self.conf = _confidence(conf, self.cfg)
        th = (
            thresholds
            if isinstance(thresholds, Thresholds)
            else load_thresholds(thresholds)
        )
        self.inference_long_edge = th.infer_long_edge
        if not Path(weights).is_file():
            raise FileNotFoundError(weights)
        disable_uploads()
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        if self.model.names != dict(enumerate(CLASSES)):
            raise ValueError(f"YOLO checkpoint class names must be {CLASSES}")

    def __call__(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        _check_frame(frame, frame_idx)
        # Ultralytics numpy source documents BGR (its preprocessor converts RGB).
        result = self.model.predict(
            frame,
            conf=self.conf,
            device=self.cfg["device"],
            imgsz=self.inference_long_edge,
            verbose=False,
        )[0].boxes
        return [
            Detection(
                frame_idx, CLASSES[int(cid)], tuple(map(float, box)), float(score)
            )
            for box, score, cid in zip(
                result.xyxy.cpu().numpy(),
                result.conf.cpu().numpy(),
                result.cls.cpu().numpy(),
                strict=True,
            )
        ]


def load_detector(
    kind: Literal["rfdetr", "yolo26"],
    weights: Path,
    *,
    conf: float | None = None,
    config: Path | str = "config/detection.yaml",
    thresholds: Thresholds | Path | str = "config/thresholds.yaml",
) -> Detector:
    if kind == "rfdetr":
        return RFDETRDetector(weights, conf, config=config, thresholds=thresholds)
    if kind == "yolo26":
        return YOLODetector(weights, conf, config=config, thresholds=thresholds)
    raise ValueError(f"unknown detector kind: {kind}")
