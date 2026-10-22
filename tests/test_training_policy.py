"""R57: production training policy is the default; smoke only when explicit."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "annotations").mkdir(parents=True)
    categories = [
        {"id": i + 1, "name": c} for i, c in enumerate(("bey", "bey_part", "launcher"))
    ]
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        cv2.imwrite(str(root / "images" / split / "1.png"), np.zeros((31, 71, 3)))
        data = {
            "images": [
                {"id": 1, "file_name": f"images/{split}/1.png", "width": 71, "height": 31}
            ],
            "categories": categories,
            "annotations": [
                {
                    "id": i + 1,
                    "image_id": 1,
                    "category_id": i + 1,
                    "bbox": [10 + i * 15, 5, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                }
                for i in range(3)
            ],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(data))
    (root / "provenance.json").write_text(json.dumps({"synthetic": True, "seed": 42}))
    return root


def _thresholds(tmp_path, edge):
    th = yaml.safe_load(Path("config/thresholds.yaml").read_text())
    th["infer_long_edge"] = edge
    path = tmp_path / f"thresholds-{edge}.yaml"
    path.write_text(yaml.safe_dump(th))
    return path


def test_default_training_policy_is_production_at_inference_size():
    from bxjudge.config import load_thresholds
    from bxjudge.detector_config import load_policy, resolve_training

    training = load_policy()["training"]
    assert training["policy"] == "production"
    assert training["image_size"] == "infer_long_edge"
    assert training["epochs"] > 1
    edge = load_thresholds("config/thresholds.yaml").infer_long_edge
    resolved = resolve_training(training, edge)
    assert resolved["image_size"] == {"rfdetr": edge, "yolo26": edge} == {
        "rfdetr": 1280,
        "yolo26": 1280,
    }
    assert resolved["rfdetr"]["model"]["resolution"] == 1280
    assert resolved["yolo26"]["fit"]["imgsz"] == 1280
    # Augmentations are on (framework defaults), not the smoke all-zero policy.
    rf, yo = resolved["rfdetr"]["fit"], resolved["yolo26"]["fit"]
    assert rf["multi_scale"] is True and rf["expanded_scales"] is True
    assert rf["scale_jitter"] is True
    assert yo["mosaic"] > 0 and yo["fliplr"] > 0 and yo["scale"] > 0
    assert yo["translate"] > 0 and min(yo["hsv_h"], yo["hsv_s"], yo["hsv_v"]) > 0
    # Effective batches: RF-DETR accumulation, Ultralytics nominal batch.
    assert rf["batch_size"] * rf["grad_accum_steps"] >= 16
    assert yo["nbs"] >= yo["batch"]


@pytest.mark.parametrize(
    "edge,expected", [(1280, 1280), (960, 960), (1000, 992), (1010, 1024)]
)
def test_production_training_size_follows_infer_long_edge(tmp_path, edge, expected):
    from bxjudge.config import load_thresholds
    from bxjudge.detector_config import load_policy, resolve_training

    infer = load_thresholds(_thresholds(tmp_path, edge)).infer_long_edge
    resolved = resolve_training(load_policy()["training"], infer)
    assert resolved["rfdetr"]["model"]["resolution"] == expected
    assert resolved["yolo26"]["fit"]["imgsz"] == expected


def test_rfdetr_training_divisor_matches_installed_framework():
    from rfdetr.config import RFDETRNanoConfig

    from bxjudge.detector_config import RFDETR_BLOCK

    fields = RFDETRNanoConfig.model_fields
    assert RFDETR_BLOCK == fields["patch_size"].default * fields["num_windows"].default


def test_smoke_policy_only_from_its_explicit_file(tmp_path):
    from bxjudge.detector_config import (
        load_policy,
        load_training_policy,
        resolve_training,
    )

    smoke = load_training_policy("config/detection_smoke.yaml")
    assert smoke["policy"] == "smoke" and smoke["epochs"] == 1
    resolved = resolve_training(smoke, 1280)
    assert resolved["image_size"] == {"rfdetr": 384, "yolo26": 384}
    assert resolved["yolo26"]["fit"]["mosaic"] == 0
    # The smoke policy can never become the default detector config.
    policy = yaml.safe_load(Path("config/detection.yaml").read_text())
    policy["training"] = smoke
    config = tmp_path / "detection.yaml"
    config.write_text(yaml.safe_dump(policy))
    with pytest.raises(ValueError, match="production"):
        load_policy(config)
    # A training policy file holds only the training block.
    with pytest.raises(ValueError, match="training"):
        load_training_policy("config/detection.yaml")


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t.update(policy="fast"),
        lambda t: t.update(image_size=1280),
        lambda t: t.update(image_size="1280"),
        lambda t: t["rfdetr"]["model"].update(resolution=1280),
        lambda t: t["rfdetr"]["fit"].update(resolution=1280),
        lambda t: t["yolo26"]["fit"].update(imgsz=1280),
        lambda t: t["yolo26"]["fit"].update(epochs=3),
        lambda t: t["rfdetr"]["fit"].update(device="cpu"),
        lambda t: t["rfdetr"]["fit"].update(batch_size="auto"),
        lambda t: t["rfdetr"]["fit"].update(grad_accum_steps=0),
        lambda t: t["yolo26"]["fit"].update(batch=-1),
        lambda t: t["yolo26"].update(model=None),
        lambda t: t.update(unknown=1),
        lambda t: t.pop("image_size"),
        lambda t: t.update(epochs=0),
    ],
)
def test_invalid_production_training_combinations_rejected(tmp_path, change):
    from bxjudge.detector_config import load_policy

    policy = yaml.safe_load(Path("config/detection.yaml").read_text())
    change(policy["training"])
    config = tmp_path / "detection.yaml"
    config.write_text(yaml.safe_dump(policy))
    with pytest.raises(ValueError, match="training"):
        load_policy(config)


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t.update(image_size="infer_long_edge"),
        lambda t: t.update(image_size=0),
        lambda t: t.update(image_size=True),
        lambda t: t.update(policy="production"),
    ],
)
def test_invalid_smoke_training_combinations_rejected(tmp_path, change):
    from bxjudge.detector_config import load_training_policy

    smoke = yaml.safe_load(Path("config/detection_smoke.yaml").read_text())
    change(smoke["training"])
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(smoke))
    with pytest.raises(ValueError, match="training"):
        load_training_policy(path)


@pytest.fixture
def fake_training(monkeypatch):
    """Fake CUDA + both frameworks; records exactly what train() hands them."""
    import torch

    from bxjudge import train as module

    seen = {}

    class RF:
        def __init__(self, **kwargs):
            seen["rfdetr_init"] = kwargs

        def train(self, **kwargs):
            seen["rfdetr_train"] = kwargs
            fit = Path(kwargs["output_dir"])
            fit.mkdir(parents=True)
            (fit / "checkpoint_best_total.pth").touch()

    class YO:
        def __init__(self, model):
            seen["yolo26_init"] = model

        def train(self, **kwargs):
            seen["yolo26_train"] = kwargs
            best = Path(kwargs["project"]) / kwargs["name"] / "weights/best.pt"
            best.parent.mkdir(parents=True)
            best.touch()

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRNano=RF))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=YO))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("set_device", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: 0)
    monkeypatch.setattr(module, "disable_uploads", lambda: None)

    def checked(kind, weights, **kwargs):
        seen["reload"] = (kind, Path(weights), kwargs)
        return object()

    monkeypatch.setattr(module, "load_detector", checked)
    return seen


@pytest.mark.parametrize("kind", ["rfdetr", "yolo26"])
def test_train_defaults_to_production_policy(fake_training, dataset, kind):
    from bxjudge.train import train

    out = dataset.parent / f"{kind}-run"
    best = train(kind, dataset, out, epochs=1)
    assert best.is_file()
    if kind == "rfdetr":
        assert fake_training["rfdetr_init"]["resolution"] == 1280
        fit = fake_training["rfdetr_train"]
        assert fit["multi_scale"] is True and fit["epochs"] == 1
        assert "resolution" not in fit
    else:
        fit = fake_training["yolo26_train"]
        assert fit["imgsz"] == 1280 and fit["mosaic"] > 0 and fit["epochs"] == 1
    record = json.loads((out / "training.json").read_text())
    assert record["training_policy"]["policy"] == "production"
    assert record["training_policy"]["image_size"] == 1280
    assert record["training_policy"]["infer_long_edge"] == 1280
    assert record["training_policy"]["source"].endswith("config/detection.yaml")


@pytest.mark.parametrize("kind", ["rfdetr", "yolo26"])
def test_train_uses_smoke_policy_only_when_explicitly_selected(
    fake_training, dataset, kind
):
    from bxjudge.train import train

    out = dataset.parent / f"{kind}-smoke"
    train(kind, dataset, out, training_policy="config/detection_smoke.yaml")
    if kind == "rfdetr":
        assert fake_training["rfdetr_init"]["resolution"] == 384
        assert fake_training["rfdetr_train"]["epochs"] == 1
    else:
        fit = fake_training["yolo26_train"]
        assert fit["imgsz"] == 384 and fit["mosaic"] == 0 and fit["epochs"] == 1
    record = json.loads((out / "training.json").read_text())
    assert record["training_policy"]["policy"] == "smoke"
    assert record["training_policy"]["source"].endswith("config/detection_smoke.yaml")


def test_smoke_policy_rejects_reviewed_dataset(fake_training, dataset):
    from bxjudge.train import train

    (dataset / "provenance.json").write_text(json.dumps({"reviewed": True}))
    with pytest.raises(ValueError, match="smoke"):
        train(
            "yolo26",
            dataset,
            dataset.parent / "run",
            training_policy="config/detection_smoke.yaml",
        )
    assert "yolo26_train" not in fake_training


def test_cli_train_exposes_explicit_training_policy():
    from bxjudge.cli import build_parser

    parser = build_parser()
    base = ["train", "--kind", "yolo26", "--dataset", "d", "--out", "o"]
    assert parser.parse_args(base).training_policy is None
    assert parser.parse_args(base).thresholds == "config/thresholds.yaml"
    smoke = parser.parse_args(
        [*base, "--training-policy", "config/detection_smoke.yaml"]
    )
    assert smoke.training_policy == "config/detection_smoke.yaml"


def test_smoke_harness_selects_smoke_policy_explicitly():
    source = Path("tests/smoke_task12.py").read_text()
    assert source.count('"--training-policy"') == 2
    assert "config/detection_smoke.yaml" in source
