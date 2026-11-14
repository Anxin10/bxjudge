"""Human-reviewed Label Studio rectangles to a video-disjoint COCO dataset."""

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .prelabel import CLASSES


def _cover_every_class(videos, val_n, reviewed):
    """Hash-order val, then swap whole videos so val holds every class that at
    least two videos contain (train keeps one). Rare classes otherwise land
    entirely in train and cannot be validated."""
    counts = {v: [0] * len(CLASSES) for v in videos}
    for data, _, _, _, boxes in reviewed:
        for category, _ in boxes:
            counts[data["video_sha256"]][category - 1] += 1

    def covered(vs):
        return {c for v in vs for c in range(len(CLASSES)) if counts[v][c]}

    val, swaps = list(videos[:val_n]), []
    if not val:
        return val, swaps
    for c in range(len(CLASSES)):
        holders = [v for v in videos if counts[v][c]]
        if len(holders) < 2 or any(v in val for v in holders):
            continue
        incoming = min(holders, key=lambda v: (counts[v][c], v))
        before = covered(val)
        outgoing = next(
            (
                v
                for v in reversed(val)
                if covered([u for u in val if u != v] + [incoming]) >= before
            ),
            None,
        )
        if outgoing is not None:
            val.remove(outgoing)
        val.append(incoming)
        swaps.append({"category": CLASSES[c], "in": incoming, "out": outgoing})
    return val, swaps


def label_studio_to_coco(
    export_json: Path, out_dir: Path, val_ratio: float | None = None
) -> None:
    """Export reviewed labels using the same image bytes that were decoded and checked."""
    with tempfile.TemporaryDirectory(prefix="bxjudge-reviewed-images-") as folder:
        _write_coco_export(export_json, out_dir, val_ratio, Path(folder))


def _write_coco_export(export_json: Path, out_dir: Path, val_ratio: float | None,
                       snapshot_dir: Path) -> None:
    ratio = load_config()["val_ratio"] if val_ratio is None else val_ratio
    if not math.isfinite(ratio) or not 0 <= ratio < 1:
        raise ValueError("val_ratio must be in [0,1)")
    tasks = json.loads(Path(export_json).read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("expected a nonempty Label Studio task export")
    reviewed = []
    snapshot_hashes = {}
    seen = set()
    for task in tasks:
        data = task["data"]
        if (
            not data.get("video_id")
            or not data.get("video_sha256")
            or type(data.get("frame")) is not int
        ):
            raise ValueError("missing video provenance or source frame")
        key = (data["video_sha256"], data["frame"])
        if key in seen:
            raise ValueError("duplicate source frame")
        seen.add(key)
        annotations = [
            a
            for a in task.get("annotations", [])
            if not a.get("was_cancelled", False) and not a.get("skipped", False)
        ]
        if len(annotations) != 1:
            raise ValueError(
                "each task requires exactly one completed human annotation; adjudicate multiple annotations first"
            )
        image = Path(data["image_path"])
        try:
            image_bytes = image.read_bytes()
            pixels = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        except (OSError, cv2.error) as error:
            raise ValueError(f"missing image: {image}") from error
        if pixels is None:
            raise ValueError(f"missing image: {image}")
        h, w = pixels.shape[:2]
        boxes = []
        for result in annotations[0]["result"]:
            if result["type"] != "rectanglelabels":
                raise ValueError("only rectanglelabels are supported")
            v = result["value"]
            labels = v["rectanglelabels"]
            if (
                len(labels) != 1
                or labels[0] not in CLASSES
                or v.get("rotation", 0) != 0
                or result.get("image_rotation", 0) != 0
            ):
                raise ValueError("invalid class or rotated rectangle")
            if result["original_width"] != w or result["original_height"] != h:
                raise ValueError("annotation and image dimensions differ")
            x, y, bw, bh = [float(v[k]) for k in ["x", "y", "width", "height"]]
            if not all(math.isfinite(z) for z in [x, y, bw, bh]) or not (
                -1e-6 <= x < 100
                and -1e-6 <= y < 100
                and bw > 0
                and bh > 0
                and x + bw <= 100 + 1e-6
                and y + bh <= 100 + 1e-6
            ):
                raise ValueError("rectangle outside image")
            # Label Studio stores edge-snapped boxes with float noise past 0/100.
            x, y = max(x, 0.0), max(y, 0.0)
            bw, bh = min(bw, 100 - x), min(bh, 100 - y)
            boxes.append(
                (
                    CLASSES.index(labels[0]) + 1,
                    [x * w / 100, y * h / 100, bw * w / 100, bh * h / 100],
                )
            )
        frozen = snapshot_dir / f"{len(reviewed):09d}.image"
        frozen.write_bytes(image_bytes)
        snapshot_hashes[frozen] = hashlib.sha256(image_bytes).hexdigest()
        reviewed.append((data, frozen, w, h, boxes))
    # Stable content-hash order, whole videos only. A one-video dataset stays train-only.
    videos = sorted({d["video_sha256"] for d, _, _, _, _ in reviewed})
    val_n = (
        min(len(videos) - 1, max(1, round(len(videos) * ratio)))
        if ratio > 0 and len(videos) > 1
        else 0
    )
    val, swaps = _cover_every_class(videos, val_n, reviewed)
    val_videos = set(val)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("output dataset directory must be empty")
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]
    coco = {
        s: {"images": [], "annotations": [], "categories": categories}
        for s in ["train", "val"]
    }
    provenance = {
        "source_export": str(Path(export_json).resolve()),
        "split": "whole video, sorted video_sha256, class-covering swaps",
        "val_ratio": ratio,
        "val_class_swaps": swaps,
        "images": [],
    }
    for image_id, (data, image, w, h, boxes) in enumerate(reviewed, 1):
        split = "val" if data["video_sha256"] in val_videos else "train"
        # Image IDs are generated, never use untrusted source IDs as filesystem paths.
        target = Path("images") / split / f"{image_id:09d}.jpg"
        (out_dir / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, out_dir / target)
        if _file_digest(out_dir / target) != snapshot_hashes[image]:
            raise ValueError("reviewed image snapshot changed during export")
        coco[split]["images"].append(
            {
                "id": image_id,
                "file_name": target.as_posix(),
                "width": w,
                "height": h,
                "video_id": data["video_id"],
                "frame": data["frame"],
            }
        )
        for category, box in boxes:
            coco[split]["annotations"].append(
                {
                    "id": len(coco[split]["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
            )
        provenance["images"].append(
            {
                "image_id": image_id,
                "split": split,
                "file_name": target.as_posix(),
                **data,
                "image_sha256": snapshot_hashes[image],
            }
        )
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    for split, dataset in coco.items():
        (out_dir / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(dataset, indent=2), encoding="utf-8"
        )
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _extension_baseline(folder: Path):
    raw = {name: (folder / name).read_bytes() for name in [
        "annotations/instances_train.json", "annotations/instances_val.json", "provenance.json"]}
    # Bind parsed labels to these exact bytes, not a later revision read for hashing.
    metadata_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in raw.items()}
    splits = {s: json.loads(raw[f"annotations/instances_{s}.json"]) for s in ["train", "val"]}
    provenance = json.loads(raw["provenance.json"])
    metadata = {p["image_id"]: p for p in provenance["images"]}
    images = [i for data in splits.values() for i in data["images"]]
    if len(metadata) != len(provenance["images"]) or len({i["id"] for i in images}) != len(images):
        raise ValueError("inconsistent baseline provenance or image IDs")
    if set(metadata) != {i["id"] for i in images}:
        raise ValueError("baseline provenance does not cover every image")
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]
    files = {"provenance.json", "annotations/instances_train.json", "annotations/instances_val.json"}
    videos = {"train": set(), "val": set()}
    ids_by_hash, hashes_by_id = {}, {}
    frames = set()
    for split, data in splits.items():
        if data["categories"] != categories:
            raise ValueError("baseline categories must match the labeling classes")
        for image in data["images"]:
            p = metadata[image["id"]]
            if (type(image["id"]) is not int or image["id"] <= 0
                    or p["split"] != split or p["file_name"] != image["file_name"]
                    or p["frame"] != image["frame"] or p["video_id"] != image["video_id"]):
                raise ValueError("inconsistent baseline provenance")
            key = (p["video_sha256"], p["frame"])
            if key in frames:
                raise ValueError("duplicate source frame in baseline provenance")
            frames.add(key)
            digest, video_id = p["video_sha256"], p["video_id"]
            if (digest in ids_by_hash and ids_by_hash[digest] != video_id
                    or video_id in hashes_by_id and hashes_by_id[video_id] != digest):
                raise ValueError("inconsistent video identity in baseline provenance")
            ids_by_hash[digest], hashes_by_id[video_id] = video_id, digest
            videos[split].add(p["video_sha256"])
            path = folder / image["file_name"]
            if not path.resolve().is_relative_to(folder) or not path.is_file():
                raise ValueError("baseline image must exist inside the dataset")
            files.add(image["file_name"])
    if videos["train"] & videos["val"]:
        raise ValueError("baseline provenance leaks a video across train/val")
    return splits, provenance, metadata, frames, files, metadata_hashes


def extend_training_dataset(baseline_dir: Path, export_json: Path, out_dir: Path) -> None:
    """Append reviewed frames from existing TRAIN videos into a new dataset.

    Validation JSON/images and existing training labels remain byte-for-byte unchanged.
    No split is recomputed, no unassigned video is admitted, and neither input nor an
    existing output directory is overwritten. The baseline must be a project COCO export.
    """
    baseline = Path(baseline_dir).resolve()
    source = Path(export_json).resolve()
    out = Path(out_dir).resolve()
    if out.exists() or out.is_relative_to(baseline):
        raise ValueError("output must be a new directory outside the baseline dataset")
    splits, provenance, metadata, existing_frames, files, metadata_hashes = _extension_baseline(baseline)
    before = {**metadata_hashes,
              **{name: _file_digest(baseline / name) for name in files - metadata_hashes.keys()}}
    source_hash = _file_digest(source)
    train_videos = {p["video_sha256"]: p["video_id"]
                    for p in metadata.values() if p["split"] == "train"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bxjudge-train-extension-", dir=out.parent) as temp:
        temp = Path(temp)
        additions = temp / "reviewed"
        # Reuse the canonical human-annotation, class, box and source-key validation.
        label_studio_to_coco(source, additions, val_ratio=0)
        added = json.loads((additions / "annotations/instances_train.json").read_text())
        new_meta = json.loads((additions / "provenance.json").read_text())["images"]
        for image, p in zip(added["images"], new_meta, strict=True):
            if (p["image_id"] != image["id"] or p["file_name"] != image["file_name"]
                    or p["split"] != "train" or p["frame"] != image["frame"]
                    or p["video_id"] != image["video_id"]):
                raise ValueError("inconsistent reviewed additions provenance")
            if p["video_sha256"] not in train_videos:
                raise ValueError("new frames must belong to an existing training video")
            if p["video_id"] != train_videos[p["video_sha256"]]:
                raise ValueError("new frame video identity differs from the baseline")
            if (p["video_sha256"], p["frame"]) in existing_frames:
                raise ValueError("source frame already exists in baseline")
        staged = temp / "dataset"
        for name, digest in before.items():
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(baseline / name, target)
            if _file_digest(target) != digest:
                raise ValueError("baseline changed while copying dataset")
        next_image = max(metadata, default=0) + 1
        next_annotation = max((a["id"] for a in splits["train"]["annotations"]), default=0) + 1
        id_map = {}
        for image, p in zip(added["images"], new_meta, strict=True):
            new_id = next_image + len(id_map)
            id_map[image["id"]] = new_id
            target = Path("images/train") / f"{new_id:09d}.jpg"
            (staged / target).parent.mkdir(parents=True, exist_ok=True)
            if (staged / target).exists():
                raise ValueError("new image path already exists in the baseline")
            shutil.copyfile(additions / image["file_name"], staged / target)
            splits["train"]["images"].append({**image, "id": new_id, "file_name": target.as_posix()})
            provenance["images"].append({**p, "image_id": new_id, "file_name": target.as_posix(),
                                          "annotation_source": str(source)})
        for a in added["annotations"]:
            splits["train"]["annotations"].append(
                {**a, "id": next_annotation, "image_id": id_map[a["image_id"]]})
            next_annotation += 1
        provenance["training_extension"] = {
            "baseline": str(baseline), "baseline_sha256": before,
            "human_export": str(source), "human_export_sha256": source_hash,
            "images_added": len(added["images"]), "validation_frozen": True,
        }
        (staged / "annotations/instances_train.json").write_text(
            json.dumps(splits["train"], indent=2), encoding="utf-8")
        (staged / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        frozen = {"annotations/instances_val.json"} | {
            image["file_name"] for image in splits["val"]["images"]}
        if any(_file_digest(staged / p) != before[p] for p in frozen):
            raise ValueError("validation corpus changed during training extension")
        if (_file_digest(source) != source_hash
                or any(_file_digest(baseline / p) != h for p, h in before.items())):
            raise ValueError("input changed during training extension")
        if out.exists():
            raise ValueError("output appeared during training extension")
        staged.rename(out)
