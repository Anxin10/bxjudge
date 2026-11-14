"""Sampled frame prelabels and Label Studio task generation."""

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from bxjudge.manifest import probe
from bxjudge.types import Box
from bxjudge.video import iter_frames

from .config import load_config
from .locateanything import MODEL_ID, MODEL_REVISION, MODEL_VERSION

CLASSES = ("bey", "bey_part", "launcher")


def video_digest(video: Path) -> str:
    with video.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def to_label_studio(
    image: str, w: int, h: int, boxes: list[tuple[str, Box, float | None]]
) -> dict:
    if w <= 0 or h <= 0:
        raise ValueError("image dimensions must be positive")
    result = []
    for cls, (x1, y1, x2, y2), score in boxes:
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("score must be a finite probability in [0,1]")
        if cls not in CLASSES or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("invalid class or box coordinates")
        result.append(
            {
                "id": str(len(result)),
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": w,
                "original_height": h,
                "image_rotation": 0,
                **({"score": float(score)} if score is not None else {}),
                "value": {
                    "x": 100 * x1 / w,
                    "y": 100 * y1 / h,
                    "width": 100 * (x2 - x1) / w,
                    "height": 100 * (y2 - y1) / h,
                    "rotation": 0,
                    "rectanglelabels": [cls],
                },
            }
        )
    return {
        "data": {"image": image},
        "predictions": [{"model_version": MODEL_VERSION, "result": result}],
    }


def filter_launcher_boxes(
    boxes: list[tuple[str, Box, float | None]], max_aspect: float | None = None
) -> list[tuple[str, Box, float | None]]:
    if max_aspect is None:
        max_aspect = load_config()["launcher_max_aspect"]
    if max_aspect < 1:
        raise ValueError("max_aspect must be >= 1")
    return [
        b
        for b in boxes
        if b[0] != "launcher"
        or (
            min(b[1][2] - b[1][0], b[1][3] - b[1][1]) > 0
            and max(b[1][2] - b[1][0], b[1][3] - b[1][1])
            / min(b[1][2] - b[1][0], b[1][3] - b[1][1])
            <= max_aspect
        )
    ]


def write_json(path: Path, data) -> None:
    """Publish complete JSON without replacing a previous result on a failed write."""
    with tempfile.TemporaryDirectory(prefix=".prelabel-", dir=path.parent) as temp:
        pending = Path(temp) / path.name
        pending.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        pending.replace(path)


def prelabel_video(
    video: Path,
    frames: list[int],
    out_json: Path,
    config: dict,
    predict: Callable[[np.ndarray], list[tuple[str, Box, float | None]]],
    *,
    expected_sha256: str | None = None,
) -> int:
    """Decode sequentially, infer only selected source frames, and export tasks."""
    if not frames or any(type(i) is not int or i < 0 for i in frames):
        raise ValueError("frames must contain nonnegative indices")
    video = Path(video).resolve()
    digest = video_digest(video)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"source video changed since sampling: {video}")
    info = probe(video)
    selected = set(frames)
    last = max(selected)
    out_json = Path(out_json)
    image_dir = out_json.parent / "images" / digest
    image_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    launcher_filtered = 0
    inference_s = 0.0
    with tempfile.TemporaryDirectory(prefix=".prelabel-", dir=image_dir) as temp:
        staging = Path(temp)
        decoder = iter_frames(video, long_edge=max(info.width, info.height), hwaccel="cpu")
        with closing(decoder):
            for index, frame in decoder:
                if index not in selected:
                    continue
                started = time.perf_counter()
                boxes = predict(frame)
                inference_s += time.perf_counter() - started
                kept = filter_launcher_boxes(boxes, config["launcher_max_aspect"])
                launcher_filtered += len(boxes) - len(kept)
                height, width = frame.shape[:2]
                image = image_dir / f"{index:09d}.jpg"
                task = to_label_studio(
                    "/data/local-files/?d=" + quote(image.relative_to(out_json.parent).as_posix()),
                    width, height, kept,
                )
                if not cv2.imwrite(str(staging / image.name), frame):
                    raise OSError(f"cannot save selected image: {image}")
                task["data"].update(
                    video_id=digest, video_sha256=digest, source_video=str(video),
                    frame=index, fps=info.fps, image_path=str(image.resolve()),
                )
                tasks.append(task)
                if index == last:
                    break
        if {t["data"]["frame"] for t in tasks} != selected:
            raise ValueError("selected frame outside decoded video")
        if video_digest(video) != digest:
            raise ValueError(f"source video changed during prelabeling: {video}")
        for asset in staging.iterdir():
            target = image_dir / asset.name
            # Preserve existing Label Studio assets at their content-addressed paths.
            try:
                os.link(asset, target)
            except FileExistsError:
                existing = cv2.imread(str(target)) if target.is_file() and not target.is_symlink() else None
                if existing is None or existing.shape[:2] != (height, width):
                    raise ValueError(f"Existing prelabel image is invalid: {target}") from None
    count = sum(len(t["predictions"][0]["result"]) for t in tasks)
    report = {
        "video": str(video), "video_sha256": digest, "model": MODEL_ID,
        "model_revision": MODEL_REVISION, "frames": sorted(selected), "boxes": count,
        "launcher_filtered": launcher_filtered, "inference_s": inference_s,
        "prompts": config["prompts"], "max_new_tokens": config["max_new_tokens"],
    }
    write_json(out_json.with_suffix(".report.json"), report)
    write_json(out_json, tasks)
    return count


def run_prelabel_jobs(jobs: list[dict], config: dict) -> list[int]:
    """Run one isolated, offline model process for the whole batch."""
    if not jobs:
        return []
    root = Path(__file__).resolve().parents[3]
    python = root / "scripts/locateanything-runtime/.venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"LocateAnything runtime missing: {python}. Run bash scripts/setup_labeling.sh")
    config = {**config, "model_dir": str(Path(config["model_dir"]).expanduser().resolve())}
    env = dict(os.environ, PYTHONPATH=str(root / "src"), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    with tempfile.TemporaryDirectory(prefix="bxjudge-label-job-") as temp:
        request = Path(temp) / "request.json"
        result = Path(temp) / "result.json"
        request.write_text(json.dumps({"jobs": jobs, "config": config}), encoding="utf-8")
        subprocess.run(
            [str(python), "-m", "bxjudge.labeling.worker", str(request), str(result)],
            cwd=root, env=env, check=True,
        )
        return json.loads(result.read_text(encoding="utf-8"))["counts"]
