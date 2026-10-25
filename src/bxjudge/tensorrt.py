"""Validated RF-DETR TensorRT artifacts; upstream owns tensor execution and decoding."""

import hashlib
import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from .detector_config import CLASSES

_MANIFEST_KEYS = {
    "engine",
    "engine_sha256",
    "weights_sha256",
    "long_edge",
    "batch_size",
    "fp16",
    "opset_version",
    "tensorrt_version",
    "rfdetr_version",
    "gpu_name",
    "compute_capability",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_trt_manifest(path: Path, weights: Path, long_edge: int) -> dict:
    """Verify the checkpoint and engine bytes before loading either as a TensorRT model."""
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != _MANIFEST_KEYS:
        raise ValueError(f"{path}: incomplete or unknown TensorRT manifest fields")
    for key in ("engine", "tensorrt_version", "rfdetr_version", "gpu_name"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{path}: invalid {key}")
    for key in ("engine_sha256", "weights_sha256"):
        if not isinstance(data[key], str) or not re.fullmatch(
            "[0-9a-f]{64}", data[key]
        ):
            raise ValueError(f"{path}: invalid SHA256 {key}")
    for key in ("long_edge", "batch_size", "opset_version"):
        if type(data[key]) is not int or data[key] <= 0:
            raise ValueError(f"{path}: invalid {key}")
    capability = data["compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(n) is not int or n < 0 for n in capability)
    ):
        raise ValueError(f"{path}: invalid compute_capability")
    if (
        data["long_edge"] != long_edge
        or data["batch_size"] != 1
        or data["fp16"] is not True
    ):
        raise ValueError(
            f"{path}: expected matching resolution, batch 1, FP16-enabled export"
        )
    engine = Path(data["engine"]).expanduser()
    if not engine.is_absolute():
        engine = path.parent / engine
    engine = engine.resolve()
    if sha256(engine) != data["engine_sha256"]:
        raise ValueError(f"{path}: engine SHA256 mismatch")
    if sha256(Path(weights)) != data["weights_sha256"]:
        raise ValueError(f"{path}: checkpoint SHA256 mismatch")
    return {**data, "engine": str(engine), "manifest_path": str(path)}


def attach_tensorrt(model, artifact: dict, device: str) -> None:
    """Attach only the network call; RF-DETR retains all pre/postprocessing."""
    try:
        for package, key in (
            ("tensorrt-cu13", "tensorrt_version"),
            ("rfdetr", "rfdetr_version"),
        ):
            if version(package) != artifact[key]:
                raise RuntimeError(
                    f"TensorRT artifact {key} differs from installed {package}; re-export"
                )
        from rfdetr.export._tensorrt.inference import TRTInference
    except (ImportError, PackageNotFoundError) as error:
        raise RuntimeError(
            "TensorRT runtime missing; install with `uv sync --extra tensorrt`"
        ) from error
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TensorRT requires an available CUDA device")
    if (
        torch.cuda.get_device_name(device) != artifact["gpu_name"]
        or list(torch.cuda.get_device_capability(device))
        != artifact["compute_capability"]
    ):
        raise RuntimeError(
            "TensorRT artifact GPU differs from this device; re-export on this GPU"
        )
    runner = TRTInference(artifact["engine"], device=device, sync_mode=True)
    edge = artifact["long_edge"]
    if runner.input_names != ["input"] or set(runner.output_names) != {
        "dets",
        "labels",
    }:
        raise ValueError("unexpected RF-DETR TensorRT input/output names")
    binding = runner.bindings["input"]
    boxes, logits = runner.bindings["dets"].shape, runner.bindings["labels"].shape
    if (
        tuple(binding.shape) != (1, 3, edge, edge)
        or binding.dtype != np.float32
        or runner.bindings["dets"].dtype != np.float32
        or runner.bindings["labels"].dtype != np.float32
        or len(boxes) != 3
        or len(logits) != 3
        or boxes[0] != 1
        or logits[0] != 1
        or boxes[1] != logits[1]
        or boxes[2] != 4
        or logits[2] != len(CLASSES) + 1
    ):
        raise ValueError("unexpected RF-DETR TensorRT shapes/dtype")

    class EngineCall(torch.nn.Module):
        def forward(self, batch):
            # execute_v2 does not share PyTorch's current preprocessing stream.
            batch = batch.contiguous()
            torch.cuda.current_stream(device).synchronize()
            values = runner({"input": batch})
            return values["dets"], values["labels"]

    model.inference(compile=False, dtype=torch.float32)
    model.model.inference_model = EngineCall()


def export_rfdetr_trt(
    weights: Path,
    output: Path,
    *,
    policy: Path | str = "config/tensorrt.yaml",
    thresholds: Path | str = "config/thresholds.yaml",
    detection_config: Path | str = "config/detection.yaml",
) -> Path:
    """Build a local FP16-enabled engine; publish its manifest only after a complete export."""
    import yaml

    from .config import config_loader, load_thresholds
    from .detect import load_detector
    from .detector_config import load_policy

    @config_loader
    def read_settings(path):
        settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if (
            not isinstance(settings, dict)
            or set(settings) != {"fp16", "batch_size", "opset_version"}
            or settings["fp16"] is not True
            or type(settings["batch_size"]) is not int
            or settings["batch_size"] != 1
            or type(settings["opset_version"]) is not int
            or settings["opset_version"] <= 0
        ):
            raise ValueError(
                f"{path}: expected FP16, batch 1, positive integer opset_version"
            )
        return settings

    settings = read_settings(policy)
    edge = load_thresholds(thresholds).infer_long_edge
    cfg = load_policy(detection_config)
    weights, output = (
        Path(weights).expanduser().resolve(),
        Path(output).expanduser().resolve(),
    )
    checkpoint_hash = sha256(weights)
    output.mkdir(parents=True, exist_ok=False)
    detector = load_detector(
        "rfdetr", weights, config=detection_config, thresholds=thresholds
    )
    with torch.cuda.device(cfg["device"]):
        engine = Path(
            detector.model.export(
                output_dir=str(output),
                format="tensorrt",
                shape=(edge, edge),
                verbose=False,
                **settings,
            )
        ).resolve()
    if not engine.is_relative_to(output):
        raise ValueError("exported engine must be in the new output directory")
    if checkpoint_hash != sha256(weights):
        raise ValueError("checkpoint SHA256 changed during export")
    record = {
        **settings,
        "engine": str(engine.relative_to(output)),
        "engine_sha256": sha256(engine),
        "weights_sha256": checkpoint_hash,
        "long_edge": edge,
        "tensorrt_version": version("tensorrt-cu13"),
        "rfdetr_version": version("rfdetr"),
        "gpu_name": torch.cuda.get_device_name(cfg["device"]),
        "compute_capability": list(torch.cuda.get_device_capability(cfg["device"])),
    }
    pending = output / "manifest.json.tmp"
    pending.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    load_trt_manifest(pending, weights, edge)
    manifest = output / "manifest.json"
    pending.replace(manifest)
    return manifest
