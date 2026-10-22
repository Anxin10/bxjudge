import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from bxjudge.types import Detection


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "annotations").mkdir(parents=True)
    categories = [
        {"id": i + 1, "name": c} for i, c in enumerate(("bey", "bey_part", "launcher"))
    ]
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        image = np.zeros((31, 71, 3), np.uint8)
        image[7, 60] = [11, 22, 233]
        cv2.imwrite(str(root / "images" / split / "1.png"), image)
        data = {
            "images": [
                {
                    "id": 1,
                    "file_name": f"images/{split}/1.png",
                    "width": 71,
                    "height": 31,
                }
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


def test_framework_layouts_preserve_images_and_map_all_classes(dataset, tmp_path):
    from bxjudge.train import prepare_dataset

    rf = prepare_dataset("rfdetr", dataset, tmp_path / "rf")
    data = json.loads((rf / "valid/_annotations.coco.json").read_text())
    assert [a["category_id"] for a in data["annotations"]] == [1, 2, 3]
    assert cv2.imread(str(rf / "valid" / data["images"][0]["file_name"]))[
        7, 60
    ].tolist() == [11, 22, 233]
    yo = prepare_dataset("yolo26", dataset, tmp_path / "yo")
    spec = yaml.safe_load(yo.read_text())
    assert spec["names"] == ["bey", "bey_part", "launcher"]
    labels = (yo.parent / "labels/val/1.txt").read_text().splitlines()
    assert [int(line.split()[0]) for line in labels] == [0, 1, 2]
    assert list(map(float, labels[2].split()[1:])) == pytest.approx(
        [45 / 71, 10 / 31, 10 / 71, 10 / 31]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "empty_val",
        "missing_class",
        "wrong_ids",
        "bad_image",
        "outside_root",
        "bad_box",
        "missing_provenance",
    ],
)
def test_reject_unusable_dataset(dataset, tmp_path, fault):
    from bxjudge.train import prepare_dataset

    path = dataset / "annotations/instances_val.json"
    data = json.loads(path.read_text())
    if fault == "empty_val":
        data["images"] = []
    elif fault == "missing_class":
        data["annotations"].pop()
    elif fault == "wrong_ids":
        data["categories"][0]["id"] = 0
    elif fault == "bad_image":
        (dataset / data["images"][0]["file_name"]).write_text("not an image")
    elif fault == "outside_root":
        data["images"][0]["file_name"] = "../outside.png"
    elif fault == "bad_box":
        data["annotations"][0]["bbox"] = [0, 0, -1, 1]
    else:
        (dataset / "provenance.json").unlink()
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_dataset("rfdetr", dataset, tmp_path / "out")


def test_comparison_uses_coco_ap_and_explicit_smoke_selection(
    monkeypatch, dataset, tmp_path
):
    from bxjudge import train as module

    def fake_loader(kind, weights, **kwargs):
        def predict(frame, idx):
            assert frame.shape == (31, 71, 3)
            assert frame[7, 60].tolist() == [11, 22, 233]
            return [
                Detection(idx, cls, (10 + i * 15, 5, 20 + i * 15, 15), 0.9)
                for i, cls in enumerate(("bey", "bey_part", "launcher"))
            ]

        return predict

    monkeypatch.setattr(module, "load_detector", fake_loader)
    weights = {}
    for kind in ("rfdetr", "yolo26"):
        weights[kind] = tmp_path / f"{kind}.pt"
        weights[kind].touch()
    output = tmp_path / "compare.json"
    selection = tmp_path / "selected.json"
    result = module.compare(weights, dataset, output=output, selection_out=selection)
    for scores in result.values():
        assert scores["bey_map50_95"] == pytest.approx(1)
        assert scores["ap_per_class"] == pytest.approx(
            dict.fromkeys(("bey", "bey_part", "launcher"), 1)
        )
        assert scores["latency_ms"] > 0
    saved = json.loads(selection.read_text())
    assert saved["dataset"]["path"] == str(dataset.resolve())
    assert saved["dataset"]["provenance"]["synthetic"] is True
    assert saved["dataset"]["sha256"]["instances_val.json"]
    assert json.loads(output.read_text())["models"] == result


def test_no_predictions_are_zero_ap_not_nan(monkeypatch, dataset, tmp_path):
    from bxjudge import train as module

    monkeypatch.setattr(module, "load_detector", lambda *a, **k: lambda f, i: [])
    result = module.compare(
        dict.fromkeys(("rfdetr", "yolo26"), tmp_path / "w"),
        dataset,
        output=tmp_path / "compare.json",
        selection_out=tmp_path / "selected.json",
    )
    assert all(scores["bey_map50_95"] == 0 for scores in result.values())


def test_synthetic_output_cannot_select_canonical_weights(dataset, tmp_path):
    from bxjudge.train import compare

    with pytest.raises(ValueError, match="synthetic"):
        compare(
            dict.fromkeys(("rfdetr", "yolo26"), Path("unused")),
            dataset,
            output=tmp_path / "compare.json",
            selection_out=Path("weights/selected.json"),
        )


def test_selection_one_point_rule_and_invalid_metrics():
    from bxjudge.train import select_model

    a = {"bey_map50_95": 0.50, "latency_ms": 10.0}
    b = {"bey_map50_95": 0.505, "latency_ms": 20.0}
    assert select_model({"rfdetr": a, "yolo26": b}, 0.01) == "rfdetr"
    b["bey_map50_95"] = 0.52
    assert select_model({"rfdetr": a, "yolo26": b}, 0.01) == "yolo26"
    a["bey_map50_95"], b["bey_map50_95"] = 0.11, 0.10
    a["latency_ms"], b["latency_ms"] = 20.0, 10.0
    assert select_model({"rfdetr": a, "yolo26": b}, 0.01) == "rfdetr"
    b["bey_map50_95"] = float("nan")
    with pytest.raises(ValueError):
        select_model({"rfdetr": a, "yolo26": b}, 0.01)


def test_cli_train_compare_contract():
    from bxjudge.cli import build_parser

    parser = build_parser()
    train = parser.parse_args(
        [
            "train",
            "--kind",
            "rfdetr",
            "--dataset",
            "data/reviewed",
            "--out",
            "runs/rf",
            "--epochs",
            "1",
        ]
    )
    assert train.epochs == 1
    compare = parser.parse_args(
        [
            "compare",
            "--dataset",
            "data/reviewed",
            "--rfdetr",
            "rf.pth",
            "--yolo26",
            "yo.pt",
            "--selection-out",
            "runs/smoke/selected.json",
        ]
    )
    assert compare.selection_out == "runs/smoke/selected.json"


def test_synthetic_dataset_beside_canonical_weights_still_cannot_select(
    monkeypatch, dataset, tmp_path
):
    from bxjudge.train import compare

    config = Path("config/detection.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="synthetic"):
        compare(
            dict.fromkeys(("rfdetr", "yolo26"), Path("unused")),
            dataset,
            output=tmp_path / "compare.json",
            selection_out=tmp_path / "weights/selected.json",
            config=config,
        )
