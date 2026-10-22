"""Local training and common COCO AP evaluation for both detector families."""

import gc
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .config import load_thresholds
from .detect import load_detector
from .detector_config import (
    CLASSES,
    KINDS,
    disable_uploads,
    load_policy,
    load_training_policy,
    resolve_training,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_dataset(dataset: Path) -> tuple[dict, dict]:
    dataset = dataset.resolve()
    provenance = json.loads((dataset / "provenance.json").read_text())
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("dataset provenance must be a nonempty mapping")
    splits = {}
    hashes = {
        "provenance.json": hashlib.sha256(
            (dataset / "provenance.json").read_bytes()
        ).hexdigest()
    }
    for split in ("train", "val"):
        path = dataset / "annotations" / f"instances_{split}.json"
        data = json.loads(path.read_text())
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if {c["id"]: c["name"] for c in data["categories"]} != dict(
            enumerate(CLASSES, 1)
        ):
            raise ValueError("COCO classes must be 1=bey, 2=bey_part, 3=launcher")
        if not data["images"] or not data["annotations"]:
            raise ValueError(f"{split} split must have images and annotations")
        images = {image["id"]: image for image in data["images"]}
        if len(images) != len(data["images"]):
            raise ValueError("duplicate image IDs")
        for image in images.values():
            relative = Path(image["file_name"])
            source = (dataset / relative).resolve()
            if relative.is_absolute() or not source.is_relative_to(dataset):
                raise ValueError("image file_name must be dataset-root relative")
            pixels = cv2.imread(str(source))
            if pixels is None or pixels.shape[:2] != (image["height"], image["width"]):
                raise ValueError(f"unreadable image or dimensions differ: {source}")
        annotated = set()
        ids = set()
        for ann in data["annotations"]:
            if ann["id"] in ids or ann["image_id"] not in images:
                raise ValueError("invalid annotation ID or image reference")
            ids.add(ann["id"])
            if ann["category_id"] not in (1, 2, 3) or ann.get("iscrowd", 0):
                raise ValueError("expected non-crowd three-class annotations")
            image = images[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            if (
                not all(math.isfinite(v) for v in (x, y, w, h))
                or min(x, y) < 0
                or min(w, h) <= 0
                or x + w > image["width"]
                or y + h > image["height"]
            ):
                raise ValueError("invalid COCO box")
            annotated.add(ann["category_id"])
        if annotated != {1, 2, 3}:
            raise ValueError(
                f"{split} split requires annotations for all three classes"
            )
        splits[split] = data
    return splits, {"path": str(dataset), "provenance": provenance, "sha256": hashes}


def _check_synthetic_output(metadata: dict, output: Path) -> None:
    # Synthetic data is useful for wiring tests, never an implicit real judge.
    if metadata["provenance"].get("synthetic") is True:
        smoke_root = Path(metadata["path"]).parent
        if not output.resolve().is_relative_to(
            smoke_root
        ) or output.resolve().is_relative_to(Path("weights").resolve()):
            raise ValueError(
                "synthetic artifacts must remain in their isolated smoke output"
            )


def prepare_dataset(kind: str, dataset: Path, out: Path) -> Path:
    """Build a framework layout from the canonical Task 11 COCO, without copying pixels."""
    if kind not in KINDS:
        raise ValueError(f"unknown detector kind: {kind}")
    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    splits, _ = _read_dataset(dataset)
    if out.exists() and any(out.iterdir()):
        raise ValueError("prepared dataset output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    for split, data in splits.items():
        if kind == "rfdetr":
            destination = out / ("valid" if split == "val" else "train")
            destination.mkdir()
            # Original file_names are images/{train,val}/..., rooted at the dataset.
            (destination / "images").symlink_to(
                dataset / "images", target_is_directory=True
            )
            _write_json(destination / "_annotations.coco.json", data)
        else:
            image_dir, labels = out / "images" / split, out / "labels" / split
            image_dir.mkdir(parents=True)
            labels.mkdir(parents=True)
            for image in data["images"]:
                source = dataset / image["file_name"]
                name = str(image["id"])
                (image_dir / (name + source.suffix)).symlink_to(source)
                rows = []
                for ann in data["annotations"]:
                    if ann["image_id"] == image["id"]:
                        x, y, w, h = ann["bbox"]
                        rows.append(
                            f"{ann['category_id'] - 1} "
                            f"{(x + w / 2) / image['width']} {(y + h / 2) / image['height']} "
                            f"{w / image['width']} {h / image['height']}"
                        )
                (labels / f"{name}.txt").write_text("\n".join(rows) + "\n")
    if kind == "rfdetr":
        return out
    spec = out / "data.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "path": str(out),
                "train": "images/train",
                "val": "images/val",
                "names": list(CLASSES),
            }
        )
    )
    return spec


def train(
    kind: str,
    dataset: Path,
    out: Path,
    epochs: int | None = None,
    *,
    config: Path | str = "config/detection.yaml",
    training_policy: Path | str | None = None,
    thresholds: Path | str = "config/thresholds.yaml",
) -> Path:
    """Train one detector; the production policy of ``config`` unless a training
    policy file (e.g. config/detection_smoke.yaml) is explicitly selected."""
    cfg = load_policy(config)
    if kind not in KINDS:
        raise ValueError(f"unknown detector kind: {kind}")
    source = Path(config if training_policy is None else training_policy)
    training = (
        cfg["training"]
        if training_policy is None
        else load_training_policy(training_policy)
    )
    infer_long_edge = load_thresholds(thresholds).infer_long_edge
    resolved = resolve_training(training, infer_long_edge)
    epochs = resolved["epochs"] if epochs is None else epochs
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    _, metadata = _read_dataset(Path(dataset))
    if (
        resolved["policy"] == "smoke"
        and metadata["provenance"].get("synthetic") is not True
    ):
        raise ValueError("the smoke training policy is for synthetic datasets only")
    out = Path(out).resolve()
    _check_synthetic_output(metadata, out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("training output must be empty (use a new run directory)")
    if not torch.cuda.is_available() or not str(cfg["device"]).startswith("cuda"):
        raise RuntimeError("training requires the configured CUDA GPU")
    disable_uploads()
    prepared = prepare_dataset(kind, dataset, out / "dataset")
    torch.cuda.set_device(cfg["device"])
    torch.cuda.reset_peak_memory_stats(cfg["device"])
    started = time.perf_counter()
    if kind == "rfdetr":
        from rfdetr import RFDETRNano

        model = RFDETRNano(device=cfg["device"], **resolved[kind]["model"])
        model.train(
            dataset_dir=str(prepared),
            dataset_file="roboflow",
            output_dir=str(out / "fit"),
            epochs=epochs,
            class_names=list(CLASSES),
            device=cfg["device"],
            **resolved[kind]["fit"],
        )
        best = out / "fit/checkpoint_best_total.pth"
    else:
        from ultralytics import YOLO

        model = YOLO(resolved[kind]["model"])
        model.train(
            data=str(prepared),
            project=str(out),
            name="fit",
            epochs=epochs,
            device=cfg["device"],
            **resolved[kind]["fit"],
        )
        best = out / "fit/weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"training produced no best checkpoint: {best}")
    _write_json(
        out / "training.json",
        {
            "kind": kind,
            "weights": str(best),
            "epochs": epochs,
            "dataset": metadata,
            "policy": {**cfg, "training": training},
            "training_policy": {
                "source": str(source.resolve()),
                "policy": resolved["policy"],
                "image_size": resolved["image_size"][kind],
                "infer_long_edge": infer_long_edge,
                "thresholds": str(Path(thresholds).resolve()),
                "framework_kwargs": resolved[kind],
            },
            "elapsed_s": time.perf_counter() - started,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                cfg["device"]
            ),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(cfg["device"]),
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # Reloading verifies trained checkpoint names/head slots before publishing its path.
    checked = load_detector(kind, best, config=config, thresholds=thresholds)
    del checked
    gc.collect()
    torch.cuda.empty_cache()
    return best


def select_model(scores: dict[str, dict], tie_ap: float) -> str:
    if set(scores) != set(KINDS):
        raise ValueError("comparison requires both rfdetr and yolo26")
    for metrics in scores.values():
        ap, latency = metrics["bey_map50_95"], metrics["latency_ms"]
        if (
            not math.isfinite(ap)
            or not 0 <= ap <= 1
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError("unusable validation metrics")
    ranked = sorted(scores, key=lambda k: scores[k]["bey_map50_95"], reverse=True)
    gap = abs(scores[ranked[0]]["bey_map50_95"] - scores[ranked[1]]["bey_map50_95"])
    # Floating point roundoff must not turn exactly one point into a speed tie.
    if gap < tie_ap and not math.isclose(gap, tie_ap):
        return min(ranked, key=lambda k: scores[k]["latency_ms"])
    return ranked[0]


def compare(
    weights: dict[str, Path],
    dataset: Path,
    *,
    output: Path = Path("runs/compare.json"),
    selection_out: Path = Path("weights/selected.json"),
    config: Path | str = "config/detection.yaml",
    thresholds: Path | str = "config/thresholds.yaml",
) -> dict[str, dict]:
    """Common pycocotools bbox AP50:95 and synchronized adapter latency (no image decoding)."""
    if set(weights) != set(KINDS):
        raise ValueError("comparison requires both rfdetr and yolo26")
    cfg = load_policy(config)
    dataset = Path(dataset).resolve()
    splits, metadata = _read_dataset(dataset)
    output, selection_out = Path(output), Path(selection_out)
    _check_synthetic_output(metadata, output)
    _check_synthetic_output(metadata, selection_out)
    edge = load_thresholds(thresholds).infer_long_edge
    inference = {
        "long_edge": edge,
        "thresholds": str(Path(thresholds).resolve()),
        "rfdetr_shape": [edge, edge],
        "yolo26_imgsz": edge,
    }
    scores = {}
    coco = COCO()
    coco.dataset = splits["val"]
    coco.createIndex()
    images = splits["val"]["images"]
    for kind in KINDS:
        detector = load_detector(
            kind,
            weights[kind],
            conf=cfg["evaluation_confidence"],
            config=config,
            thresholds=thresholds,
        )
        first = cv2.imread(str(dataset / images[0]["file_name"]))
        for i in range(cfg["latency_warmup_frames"]):
            detector(first, i)
        predictions = []
        elapsed = 0.0
        for index, image in enumerate(images):
            frame = cv2.imread(str(dataset / image["file_name"]))
            if torch.cuda.is_available():
                torch.cuda.synchronize(cfg["device"])
            started = time.perf_counter()
            detected = detector(frame, index)
            if torch.cuda.is_available():
                torch.cuda.synchronize(cfg["device"])
            elapsed += time.perf_counter() - started
            for detection in detected:
                x1, y1, x2, y2 = detection.box
                predictions.append(
                    {
                        "image_id": image["id"],
                        "category_id": CLASSES.index(detection.cls) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": detection.score,
                    }
                )
        if predictions:
            result = coco.loadRes(predictions)
        else:
            # COCO.loadRes([]) indexes its first element; an empty COCO result
            # retains the targets and gives true zero AP instead of a sentinel.
            result = COCO()
            result.dataset = {
                "images": images,
                "categories": splits["val"]["categories"],
                "annotations": [],
            }
            result.createIndex()
        evaluator = COCOeval(coco, result, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        per_class = {}
        for class_index, cls in enumerate(CLASSES):
            precision = evaluator.eval["precision"][:, :, class_index, 0, -1]
            valid = precision[precision >= 0]
            if not valid.size or not np.isfinite(valid).all():
                raise ValueError(f"unusable validation for {cls}")
            per_class[cls] = float(valid.mean())
        scores[kind] = {
            "bey_map50_95": per_class["bey"],
            "ap_per_class": per_class,
            "map50_95": float(evaluator.stats[0]),
            "latency_ms": elapsed / len(images) * 1000,
            "validation_images": len(images),
        }
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selected = select_model(scores, cfg["selection_tie_ap"])
    selection = {
        "kind": selected,
        "weights": str(Path(weights[selected]).resolve()),
        "dataset": metadata,
        "metrics": scores[selected],
        "inference": inference,
        "comparison": str(output.resolve()),
    }
    _write_json(
        output,
        {
            "models": scores,
            "dataset": metadata,
            "selection": selection,
            "policy": cfg,
            "inference": inference,
            "metric": "COCO bbox AP50:95 (IoU .50:.05:.95), AP@100",
            "latency": "mean warmed synchronized adapter ms/image; excludes image decode",
        },
    )
    _write_json(selection_out, selection)
    return scores
