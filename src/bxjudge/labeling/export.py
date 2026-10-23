"""Human-reviewed Label Studio rectangles to a video-disjoint COCO dataset."""

import json
import math
import shutil
from pathlib import Path

import cv2

from .config import load_config
from .sam3_prelabel import CLASSES


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
    ratio = load_config()["val_ratio"] if val_ratio is None else val_ratio
    if not math.isfinite(ratio) or not 0 <= ratio < 1:
        raise ValueError("val_ratio must be in [0,1)")
    tasks = json.loads(Path(export_json).read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("expected a nonempty Label Studio task export")
    reviewed = []
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
        pixels = cv2.imread(str(image))
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
        reviewed.append((data, image, w, h, boxes))
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
