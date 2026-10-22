import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import supervision as sv
import yaml


@pytest.mark.parametrize("threshold_source", ["path", "object"])
@pytest.mark.parametrize("kind", ["rfdetr", "yolo26"])
@pytest.mark.parametrize("inference_edge,training_edge", [(1280, 384), (960, 640)])
def test_adapter_color_axes_classes_source_frame(
    monkeypatch, tmp_path, kind, inference_edge, training_edge, threshold_source
):
    from bxjudge.detect import load_detector

    image = np.zeros((31, 71, 3), np.uint8)
    image[7, 60] = [11, 22, 233]
    xyxy = np.array([[51, 4, 68, 20]], dtype=float)
    seen = {}

    class RF:
        class_names: ClassVar[list[str]] = ["bey", "bey_part", "launcher"]

        def __init__(self, **kwargs):
            pass

        def predict(self, pixels, **kwargs):
            seen["pixels"] = pixels
            seen["kwargs"] = kwargs
            return sv.Detections(
                xyxy=xyxy, confidence=np.array([0.8]), class_id=np.array([2])
            )

    class YO:
        names: ClassVar[dict[int, str]] = {0: "bey", 1: "bey_part", 2: "launcher"}

        def __init__(self, *args):
            pass

        def predict(self, pixels, **kwargs):
            import torch

            seen["pixels"] = pixels
            seen["kwargs"] = kwargs
            boxes = SimpleNamespace(
                xyxy=torch.tensor(xyxy), cls=torch.tensor([2]), conf=torch.tensor([0.8])
            )
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRNano=RF))
    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=YO, settings={})
    )
    weights = tmp_path / "weights.pt"
    weights.touch()
    cfg = yaml.safe_load(Path("config/detection.yaml").read_text())
    cfg["training"]["rfdetr"]["model"]["resolution"] = training_edge
    cfg["training"]["yolo26"]["fit"]["imgsz"] = training_edge
    config = tmp_path / "detection.yaml"
    config.write_text(yaml.safe_dump(cfg))
    thresholds = yaml.safe_load(Path("config/thresholds.yaml").read_text())
    thresholds["infer_long_edge"] = inference_edge
    thresholds_path = tmp_path / "thresholds.yaml"
    thresholds_path.write_text(yaml.safe_dump(thresholds))
    from bxjudge.config import load_thresholds

    thresholds_arg = (
        load_thresholds(thresholds_path)
        if threshold_source == "object"
        else thresholds_path
    )
    detector = load_detector(kind, weights, config=config, thresholds=thresholds_arg)
    result = detector(image, 137)
    if kind == "rfdetr":
        assert seen["kwargs"]["shape"] == (inference_edge, inference_edge)
    else:
        assert seen["kwargs"]["imgsz"] == inference_edge
    assert len(result) == 1
    assert result[0].frame == 137 and result[0].cls == "launcher"
    assert result[0].box == (51.0, 4.0, 68.0, 20.0)
    assert result[0].score == pytest.approx(0.8)
    assert seen["pixels"].shape == (31, 71, 3)
    expected = [233, 22, 11] if kind == "rfdetr" else [11, 22, 233]
    assert seen["pixels"][7, 60].tolist() == expected
    assert image[7, 60].tolist() == [11, 22, 233]


@pytest.mark.parametrize("kind", ["rfdetr", "yolo26"])
def test_adapter_rejects_wrong_checkpoint_class_contract(monkeypatch, tmp_path, kind):
    from bxjudge.detect import load_detector

    class Wrong:
        class_names: ClassVar[list[str]] = ["bey", "launcher", "bey_part"]
        names: ClassVar[dict[int, str]] = {0: "bey", 1: "launcher", 2: "bey_part"}

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRNano=Wrong))
    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=Wrong, settings={})
    )
    weights = tmp_path / "wrong.pt"
    weights.touch()
    with pytest.raises(ValueError, match="class"):
        load_detector(kind, weights)


@pytest.mark.parametrize("cid", [3, 4])
def test_rfdetr_background_is_removed_unknown_head_slot_is_rejected(
    monkeypatch, tmp_path, cid
):
    from bxjudge.detect import RFDETRDetector

    class RF:
        class_names = ("bey", "bey_part", "launcher")

        def __init__(self, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return sv.Detections(
                xyxy=np.array([[1.0, 2.0, 4.0, 6.0]]),
                confidence=np.array([0.002]),
                class_id=np.array([cid]),
            )

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRNano=RF))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(settings={}))
    weights = tmp_path / "w.pth"
    weights.touch()
    detector = RFDETRDetector(weights, conf=0.001)
    if cid == 3:
        assert detector(np.zeros((12, 20, 3), np.uint8), 11) == []
    else:
        with pytest.raises(ValueError, match="class"):
            detector(np.zeros((12, 20, 3), np.uint8), 11)
