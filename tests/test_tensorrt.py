"""TensorRT artifact provenance and adapter boundary, without loading a GPU model."""

import contextlib
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def no_real_cuda_context(monkeypatch):
    # These tests exercise our boundary; TensorRT/GPU execution is a separate real probe.
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def artifact(tmp_path):
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"trained checkpoint")
    engine = tmp_path / "model.trt"
    engine.write_bytes(b"exported engine")
    data = {
        "engine": engine.name,
        "engine_sha256": digest(engine),
        "weights_sha256": digest(weights),
        "long_edge": 1280,
        "batch_size": 1,
        "fp16": True,
        "opset_version": 17,
        "tensorrt_version": "10.16.1.11",
        "rfdetr_version": "1.11.2",
        "gpu_name": "Test GPU",
        "compute_capability": [8, 6],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path, weights, engine, data


def test_manifest_resolves_relative_engine_independently_of_cwd(artifact, monkeypatch):
    from bxjudge.tensorrt import load_trt_manifest

    path, weights, engine, data = artifact
    monkeypatch.chdir(path.parent.parent)
    record = load_trt_manifest(path, weights, 1280)
    assert record["engine"] == str(engine.resolve())
    assert record["manifest_path"] == str(path.resolve())
    assert record["weights_sha256"] == digest(weights)
    assert record["engine_sha256"] == data["engine_sha256"]


@pytest.mark.parametrize("target", ["engine", "weights"])
def test_modified_artifact_is_rejected(artifact, target):
    from bxjudge.tensorrt import load_trt_manifest

    path, weights, engine, _ = artifact
    (engine if target == "engine" else weights).write_bytes(b"different")
    with pytest.raises(ValueError, match="SHA256"):
        load_trt_manifest(path, weights, 1280)


@pytest.mark.parametrize(
    "change",
    [
        {"long_edge": 960},
        {"batch_size": True},
        {"batch_size": 2},
        {"fp16": False},
        {"fp16": 1},
        {"opset_version": "17"},
        {"gpu_name": ""},
        {"compute_capability": [8]},
        {"engine_sha256": "0"},
        {"unknown": 1},
    ],
)
def test_invalid_manifest_is_rejected(artifact, change):
    from bxjudge.tensorrt import load_trt_manifest

    path, weights, _, data = artifact
    data.update(change)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_trt_manifest(path, weights, 1280)


def test_incomplete_manifest_is_not_assumed_native(artifact):
    from bxjudge.tensorrt import load_trt_manifest

    path, weights, _, data = artifact
    del data["weights_sha256"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest"):
        load_trt_manifest(path, weights, 1280)


@pytest.fixture
def runtime(monkeypatch, artifact):
    from bxjudge import tensorrt as module

    path, weights, _, data = artifact
    record = module.load_trt_manifest(path, weights, 1280)
    events = []
    monkeypatch.setattr(
        module,
        "version",
        lambda name: {
            "tensorrt-cu13": data["tensorrt_version"],
            "rfdetr": data["rfdetr_version"],
        }[name],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: data["gpu_name"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(
            synchronize=lambda: events.append("synchronize")
        ),
    )
    boxes, logits = torch.ones(1, 5, 4), torch.zeros(1, 5, 4)

    class Runner:
        input_names: ClassVar[list[str]] = ["input"]
        output_names: ClassVar[list[str]] = ["labels", "dets"]
        bindings: ClassVar[dict] = {
            "input": SimpleNamespace(shape=(1, 3, 1280, 1280), dtype=np.float32),
            "dets": SimpleNamespace(shape=tuple(boxes.shape), dtype=np.float32),
            "labels": SimpleNamespace(shape=tuple(logits.shape), dtype=np.float32),
        }

        def __init__(self, engine_path, **kwargs):
            events.append((engine_path, kwargs))

        def __call__(self, batch):
            events.append("execute")
            assert batch["input"].is_contiguous()
            return {"labels": logits, "dets": boxes}

    monkeypatch.setitem(
        sys.modules,
        "rfdetr.export._tensorrt.inference",
        SimpleNamespace(TRTInference=Runner),
    )

    class Model:
        model = SimpleNamespace(inference_model=None)

        def inference(self, **kwargs):
            events.append(kwargs)

    return module, record, Model(), Runner, events, boxes, logits


def test_runtime_preserves_upstream_prepostprocessing_and_synchronizes(runtime):
    module, record, model, _, events, boxes, logits = runtime
    module.attach_tensorrt(model, record, "cuda:0")
    result = model.model.inference_model(torch.zeros(1, 3, 2, 4).transpose(2, 3))
    assert result[0] is boxes and result[1] is logits
    assert events[-2:] == ["synchronize", "execute"]
    assert events[0][1] == {"device": "cuda:0", "sync_mode": True}
    assert events[1] == {"compile": False, "dtype": torch.float32}


@pytest.mark.parametrize("mismatch", ["version", "gpu", "shape", "dtype", "outputs"])
def test_runtime_refuses_incompatible_environment_or_bindings(
    runtime, monkeypatch, mismatch
):
    module, record, model, runner, _, _, _ = runtime
    if mismatch == "version":
        monkeypatch.setattr(module, "version", lambda name: "wrong")
    elif mismatch == "gpu":
        monkeypatch.setattr(
            torch.cuda, "get_device_name", lambda device: "Different GPU"
        )
    elif mismatch == "shape":
        runner.bindings["input"].shape = (1, 3, 640, 640)
    elif mismatch == "dtype":
        runner.bindings["input"].dtype = np.float16
    else:
        runner.output_names = ["dets"]
    with pytest.raises((ValueError, RuntimeError)):
        module.attach_tensorrt(model, record, "cuda:0")
    assert model.model.inference_model is None


def test_trt_cli_flag_reaches_pipeline(tmp_path, monkeypatch):
    from bxjudge import pipeline
    from bxjudge.cli import main

    captured = {}
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **kw: captured.update(kw))
    manifest = tmp_path / "manifest.json"
    assert (
        main(["run", str(tmp_path), "--trt-manifest", str(manifest), "--no-vlm"]) == 0
    )
    assert captured["trt_manifest"] == manifest


def test_requested_trt_failure_leaves_batch_output_absent(artifact, tmp_path):
    from bxjudge.pipeline import run_batch

    path, weights, engine, _ = artifact
    engine.write_bytes(b"corrupt")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "kind": "rfdetr",
                "weights": str(weights),
                "dataset": {"provenance": {"source_export": "human"}},
                "inference": {"long_edge": 1280},
            }
        )
    )
    calib = tmp_path / "calib"
    calib.mkdir()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA256"):
        run_batch(tmp_path, calib, output, selection=selection, trt_manifest=path)
    assert not output.exists()


def test_pipeline_records_exact_backend_evidence(artifact, tmp_path, monkeypatch):
    from bxjudge import pipeline
    from bxjudge.types import RoundResult
    from tests.helpers import make_video

    path, weights, _, _ = artifact
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "kind": "rfdetr",
                "weights": str(weights),
                "dataset": {"provenance": {"source_export": "human"}},
                "inference": {"long_edge": 1280},
            }
        )
    )
    videos = tmp_path / "videos"
    videos.mkdir()
    make_video(videos / "a.mp4", seconds=0.1)
    calib = tmp_path / "calib"
    calib.mkdir()
    evidence = {"backend": "tensorrt", "engine_sha256": "abc123"}

    class FakeDetector:
        runtime = evidence

    def load(kind, weights, **kwargs):
        assert kwargs["trt_manifest"] == path
        return FakeDetector()

    monkeypatch.setattr(pipeline, "load_detector", load)

    def judge(*args, **kwargs):
        return pipeline._provenance(
            [RoundResult("a.mp4", 1, 0.0, 0.1, None, None, 0.0, True, ["no_round"])],
            kwargs["provenance"],
        )

    monkeypatch.setattr(pipeline, "judge_video", judge)
    from bxjudge.timebase import Timeline

    monkeypatch.setattr(
        pipeline,
        "probe_timeline",
        lambda *_: Timeline(tuple(i / 60 for i in range(6)), 120, 60),
    )
    output = tmp_path / "output"
    pipeline.run_batch(videos, calib, output, selection=selection, trt_manifest=path)
    rounds = json.loads((output / "rounds.json").read_text())
    assert rounds[0]["detector_provenance"]["runtime"]["detector"] == evidence


@pytest.mark.parametrize("existing", [True, False])
def test_export_uses_yaml_and_publishes_manifest_last(tmp_path, monkeypatch, existing):
    import yaml

    from bxjudge import detect
    from bxjudge import tensorrt as module
    from bxjudge.tensorrt import export_rfdetr_trt

    weights = tmp_path / "model.pth"
    weights.write_bytes(b"weights")
    output = tmp_path / "export"
    config = tmp_path / "export.yaml"
    config.write_text(
        yaml.safe_dump({"fp16": True, "batch_size": 1, "opset_version": 17})
    )
    monkeypatch.setattr(
        module,
        "version",
        lambda package: {"tensorrt-cu13": "10.16.1.11", "rfdetr": "1.11.2"}[package],
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    seen = []

    class Model:
        def export(self, **kwargs):
            assert not (output / "manifest.json").exists()
            seen.append(kwargs)
            engine = output / "model.trt"
            engine.write_bytes(b"engine")
            return str(engine)

    monkeypatch.setattr(
        detect, "load_detector", lambda *a, **kw: SimpleNamespace(model=Model())
    )
    if existing:
        output.mkdir()
        (output / "original").write_bytes(b"keep")
        with pytest.raises(FileExistsError):
            export_rfdetr_trt(weights, output, policy=config)
        assert not seen and (output / "original").read_bytes() == b"keep"
    else:
        path = export_rfdetr_trt(weights, output, policy=config)
        assert seen == [
            {
                "output_dir": str(output),
                "format": "tensorrt",
                "fp16": True,
                "shape": (1280, 1280),
                "batch_size": 1,
                "opset_version": 17,
                "verbose": False,
            }
        ]
        record = module.load_trt_manifest(path, weights, 1280)
        assert record["engine_sha256"] == digest(output / "model.trt")
        assert record["weights_sha256"] == digest(weights)


def test_failed_export_does_not_publish_manifest(tmp_path, monkeypatch):
    from bxjudge import detect
    from bxjudge.tensorrt import export_rfdetr_trt

    weights = tmp_path / "model.pth"
    weights.write_bytes(b"weights")
    output = tmp_path / "export"

    def fail(**kwargs):
        raise RuntimeError("build failed")

    monkeypatch.setattr(
        detect,
        "load_detector",
        lambda *a, **kw: SimpleNamespace(model=SimpleNamespace(export=fail)),
    )
    with pytest.raises(RuntimeError, match="build failed"):
        export_rfdetr_trt(weights, output)
    assert not (output / "manifest.json").exists()


def test_rfdetr_constructor_activates_engine_and_records_hashes(
    runtime, artifact, monkeypatch
):
    import supervision as sv

    from bxjudge.detect import load_detector

    module, _, _, _, _, _, _ = runtime
    path, weights, _, data = artifact
    captured = []

    class RF:
        class_names = ("bey", "bey_part", "launcher")

        def __init__(self, **kwargs):
            pass

        def predict(self, frame, **kwargs):
            assert frame[0, 0].tolist() == [33, 22, 11]
            return sv.Detections(
                xyxy=np.array([[4.0, 2.0, 18.0, 10.0]]),
                confidence=np.array([0.8]),
                class_id=np.array([0]),
            )

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRNano=RF))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(settings={}))
    monkeypatch.setattr(module, "attach_tensorrt", lambda *args: captured.append(args))
    detector = load_detector("rfdetr", weights, trt_manifest=path)
    image = np.full((20, 30, 3), [11, 22, 33], dtype=np.uint8)
    result = detector(image, 147)
    assert len(captured) == 1 and captured[0][0] is detector.model
    assert detector.runtime["backend"] == "tensorrt"
    assert detector.runtime["engine_sha256"] == data["engine_sha256"]
    assert result[0].frame == 147 and result[0].box == (4.0, 2.0, 18.0, 10.0)
    assert result[0].cls == "bey" and result[0].score == pytest.approx(0.8)


def test_trt_manifest_cannot_silently_select_yolo(artifact):
    from bxjudge.detect import load_detector

    path, weights, _, _ = artifact
    with pytest.raises(ValueError, match="rfdetr"):
        load_detector("yolo26", weights, trt_manifest=path)


def test_contiguous_copy_finishes_before_trt_execution(runtime):
    module, record, model, _, events, _, _ = runtime
    module.attach_tensorrt(model, record, "cuda:0")

    class PendingCopy:
        def contiguous(self):
            events.append("contiguous")
            return torch.zeros(1, 3, 2, 2)

    model.model.inference_model(PendingCopy())
    assert events[-3:] == ["contiguous", "synchronize", "execute"]


def test_integer_output_bindings_fail_before_attachment(runtime):
    module, record, model, runner, _, _, _ = runtime
    runner.bindings["labels"].dtype = np.int32
    with pytest.raises(ValueError, match="dtype"):
        module.attach_tensorrt(model, record, "cuda:0")
    assert model.model.inference_model is None


def test_export_selects_configured_cuda_device(tmp_path, monkeypatch):
    import contextlib

    import yaml

    from bxjudge import detect
    from bxjudge import tensorrt as module

    cfg = yaml.safe_load(Path("config/detection.yaml").read_text())
    cfg["device"] = "cuda:1"
    config = tmp_path / "detection.yaml"
    config.write_text(yaml.safe_dump(cfg))
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    output = tmp_path / "export"
    active = []

    @contextlib.contextmanager
    def device_context(device):
        active.append(device)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "GPU 1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    monkeypatch.setattr(module, "version", lambda name: "test")

    def export(**kwargs):
        assert active == ["cuda:1"]
        engine = output / "model.trt"
        engine.write_bytes(b"engine")
        return str(engine)

    monkeypatch.setattr(
        detect,
        "load_detector",
        lambda *a, **kw: SimpleNamespace(model=SimpleNamespace(export=export)),
    )
    module.export_rfdetr_trt(weights, output, detection_config=config)
