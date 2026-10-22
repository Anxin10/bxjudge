"""SAM 3.1 multiplex video labeling through one pinned direct-model contract.

Each temporary resource contains <= window_frames contiguous original frames.
No full-resolution masks are retained by application code. IDs reset per window.
"""

import gc
import hashlib
import json
import math
import resource
import tempfile
from pathlib import Path
from urllib.parse import quote

import cv2

from bxjudge.manifest import probe
from bxjudge.types import Box
from bxjudge.video import iter_frames

from .config import load_config

CLASSES = ("bey", "bey_part", "launcher")
SAM_REVISION = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"


def video_digest(video: Path) -> str:
    with video.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def to_label_studio(
    image: str, w: int, h: int, boxes: list[tuple[str, Box, float]]
) -> dict:
    if w <= 0 or h <= 0:
        raise ValueError("image dimensions must be positive")
    result = []
    for cls, (x1, y1, x2, y2), score in boxes:
        if not math.isfinite(score) or not 0 <= score <= 1:
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
                "score": float(score),
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
        "predictions": [{"model_version": f"sam3.1-{SAM_REVISION}", "result": result}],
    }


def filter_launcher_boxes(
    boxes: list[tuple[str, Box, float]], max_aspect: float | None = None
) -> list[tuple[str, Box, float]]:
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


def _build_predictor(config):
    from sam3.model_builder import build_sam3_predictor

    checkpoint = Path(config["checkpoint"]).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM 3.1 checkpoint missing: {checkpoint}")
    predictor = build_sam3_predictor(
        version="sam3.1",
        checkpoint_path=str(checkpoint),
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        async_loading_frames=False,
    )
    # The pinned builder hardcodes batched_grounding_batch_size=16 (no builder kwarg);
    # the model's public constructor attribute is read per propagation (R47).
    model = predictor.model
    if type(getattr(model, "batched_grounding_batch_size", None)) is not int:
        raise RuntimeError(
            "SAM SDK no longer exposes model.batched_grounding_batch_size; "
            "re-verify the pinned SDK batch policy"
        )
    model.batched_grounding_batch_size = config["batched_grounding_batch_size"]
    proof = verify_checkpoint_keys(predictor.model, checkpoint)
    print("SAM final checkpoint contract: " + json.dumps(proof), flush=True)
    return predictor


def _boxes(outputs, width, height):
    result = []
    for xywh, prob in zip(outputs["out_boxes_xywh"], outputs["out_probs"], strict=True):
        x, y, w, h = map(float, xywh)
        prob = float(prob)
        if not all(math.isfinite(v) for v in (x, y, w, h, prob)):
            raise ValueError("SAM output coordinates and probability must be finite")
        if not 0 <= prob <= 1:
            raise ValueError("SAM output probability must be in [0,1]")
        x1, y1 = max(0.0, x * width), max(0.0, y * height)
        x2, y2 = (
            min(float(width), (x + w) * width),
            min(float(height), (y + h) * height),
        )
        if x2 > x1 and y2 > y1:
            result.append(((x1, y1, x2, y2), float(prob)))
    return result


def _infer_window(model, folder, mapping, selected, prompts, width, height):
    state = None
    records = {i: [] for i in selected}
    try:
        state = model.init_state(
            resource_path=str(folder),
            offload_video_to_cpu=True,
            async_loading_frames=False,
        )
        for cls, prompt in prompts.items():
            model.reset_state(state)
            first_index, first_outputs = model.add_prompt(
                inference_state=state, frame_idx=0, text_str=prompt
            )
            original = mapping[first_index]
            if original in records:
                records[original] = [b for b in records[original] if b[0] != cls] + [
                    (cls, b, s) for b, s in _boxes(first_outputs, width, height)
                ]
            del first_outputs
            iterator = model.propagate_in_video(
                inference_state=state,
                start_frame_idx=0,
                max_frame_num_to_track=None,
                reverse=False,
                is_last_batch=True,
            )
            try:
                for local, outputs in iterator:
                    original = mapping[local]
                    if original in records:
                        records[original] = [
                            b for b in records[original] if b[0] != cls
                        ] + [(cls, b, s) for b, s in _boxes(outputs, width, height)]
                    del outputs
            finally:
                iterator.close()
    finally:
        if state is not None:
            try:
                model.reset_state(state)
            finally:
                state.clear()
        gc.collect()
    return records


def _prelabel_direct(
    video: Path,
    frames: list[int],
    prompts: dict[str, str],
    out_json: Path,
    *,
    window_frames: int | None = None,
    config: dict | None = None,
    predictor=None,
) -> int:
    import torch

    cfg = load_config() if config is None else config
    limit = cfg["window_frames"] if window_frames is None else window_frames
    if type(limit) is not int or limit <= 0:
        raise ValueError("window_frames must be positive")
    if any(type(i) is not int or i < 0 for i in frames) or not frames:
        raise ValueError("frames must contain nonnegative indices")
    if (
        not prompts
        or set(prompts) - set(CLASSES)
        or any(not p.strip() for p in prompts.values())
    ):
        raise ValueError("invalid class prompts")
    video = Path(video).resolve()
    info = probe(video)
    selected = set(frames)
    digest = video_digest(video)
    video_id = digest
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    image_dir = out_json.parent / "images" / video_id
    image_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    report = {
        "video": str(video),
        "video_sha256": digest,
        "sdk_revision": SAM_REVISION,
        "window_limit": limit,
        "windows": [],
        "launcher_filtered": 0,
        "identities": "reset per window",
    }
    owns_predictor = predictor is None
    decoder = iter_frames(video, long_edge=max(info.width, info.height), hwaccel="cpu")
    needed_windows = {i // limit for i in selected}
    # Disk windows and the SDK tensors are bounded independently of total clip length.
    try:
        if predictor is None:
            predictor = _build_predictor(cfg)
        with (
            torch.inference_mode(),
            tempfile.TemporaryDirectory(prefix="bxjudge-sam-") as temp,
        ):
            folder = Path(temp)
            mapping = []
            image_width, image_height = info.width, info.height

            def process():
                if not mapping:
                    return
                selected_here = selected.intersection(mapping)
                boxes = _infer_window(
                    predictor.model,
                    folder,
                    mapping,
                    selected_here,
                    prompts,
                    image_width,
                    image_height,
                )
                for index in sorted(selected_here):
                    kept = filter_launcher_boxes(
                        boxes[index], cfg["launcher_max_aspect"]
                    )
                    report["launcher_filtered"] += len(boxes[index]) - len(kept)
                    image = image_dir / f"{index:09d}.jpg"
                    task = to_label_studio(
                        "/data/local-files/?d="
                        + quote(image.relative_to(out_json.parent).as_posix()),
                        image_width,
                        image_height,
                        kept,
                    )
                    task["data"].update(
                        video_id=video_id,
                        video_sha256=digest,
                        source_video=str(video),
                        frame=index,
                        fps=info.fps,
                        image_path=str(image.resolve()),
                    )
                    tasks.append(task)
                torch.cuda.empty_cache()
                report["windows"].append(
                    {
                        "first_frame": mapping[0],
                        "last_frame": mapping[-1],
                        "frames": len(mapping),
                        "selected": sorted(selected_here),
                        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                        "peak_rss_kib": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss,
                    }
                )
                print(
                    f"SAM window {mapping[0]}..{mapping[-1]} ({len(mapping)} frames), selected={len(selected_here)}",
                    flush=True,
                )
                for file in folder.iterdir():
                    file.unlink()
                mapping.clear()

            for index, frame in decoder:
                image_height, image_width = frame.shape[:2]
                if index // limit not in needed_windows:
                    continue
                if mapping and index // limit != mapping[0] // limit:
                    process()
                if not cv2.imwrite(str(folder / f"{len(mapping):06d}.jpg"), frame):
                    raise OSError("cannot save SAM resource frame")
                mapping.append(index)
                if index in selected and not cv2.imwrite(
                    str(image_dir / f"{index:09d}.jpg"), frame
                ):
                    raise OSError("cannot save selected image")
                if len(mapping) == limit:
                    process()
                if index >= max(selected) and (index + 1) % limit == 0:
                    break
            process()
        if {t["data"]["frame"] for t in tasks} != selected:
            raise ValueError("selected frame outside decoded video")
        report["boxes"] = sum(len(t["predictions"][0]["result"]) for t in tasks)
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        out_json.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        out_json.with_suffix(".report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            f"{report['boxes']} boxes; filtered {report['launcher_filtered']} thin launcher boxes",
            flush=True,
        )
        return report["boxes"]
    finally:
        decoder.close()
        if predictor is not None and owns_predictor:
            predictor.shutdown()
        # Wrapper shutdown clears only registered sessions. The CLI process boundary
        # releases model allocations and the SDK's constructor-entered autocast context.


def run_prelabel_jobs(jobs: list[dict], config: dict) -> list[int]:
    """One explicit supported SDK runtime, one model load for the full CLI batch."""
    import os
    import shutil
    import subprocess

    if not jobs:
        return []
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for the isolated SAM runtime")
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env.pop("VIRTUAL_ENV", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(root / "scripts/sam-runtime/.venv")
    with tempfile.TemporaryDirectory(prefix="bxjudge-label-job-") as temp:
        request = Path(temp) / "request.json"
        result = Path(temp) / "result.json"
        request.write_text(
            json.dumps({"jobs": jobs, "config": config}), encoding="utf-8"
        )
        subprocess.run(
            [
                uv,
                "run",
                "--locked",
                "--project",
                str(root / "scripts/sam-runtime"),
                "python",
                "-m",
                "bxjudge.labeling.worker",
                str(request),
                str(result),
            ],
            cwd=root,
            env=env,
            check=True,
        )
        return json.loads(result.read_text(encoding="utf-8"))["counts"]


def prelabel(
    video: Path, frames: list[int], prompts: dict[str, str], out_json: Path
) -> int:
    cfg = load_config()
    cfg["prompts"] = prompts
    return run_prelabel_jobs(
        [
            {
                "video": str(Path(video).resolve()),
                "frames": frames,
                "out_json": str(Path(out_json).resolve()),
            }
        ],
        cfg,
    )[0]


def verify_checkpoint_keys(model, checkpoint: Path) -> dict:
    """Fail closed if official factory's non-strict final load left random tensors.

    The supported HF SAM3.1 checkpoint is a flat, already-remapped state dict.
    mmap avoids a second full resident checkpoint allocation for this key audit.
    """
    import torch

    weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    expected = set(model.state_dict())
    actual = set(weights)
    proof = {
        "checkpoint_keys": len(actual),
        "missing_keys": sorted(expected - actual),
        "unexpected_keys": sorted(actual - expected),
    }
    if proof["missing_keys"] or proof["unexpected_keys"]:
        raise ValueError(f"SAM checkpoint/model contract mismatch: {proof}")
    return proof
