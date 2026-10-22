"""Labeling contracts: real sampling/export, exact pinned SAM direct API fake."""

import json
import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


def video_file(tmp_path, count=13):
    p = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(count):
        writer.write(np.full((48, 64, 3), 255 if i == 6 else 0, np.uint8))
    writer.release()
    return p


def test_select_frames_hits_peaks():
    from bxjudge.labeling.sample_frames import select_frames

    e = {f: (100.0 if f in (60, 300, 600) else 1.0) for f in range(900)}
    s = select_frames(e, 60, n=10)
    assert {60, 300, 600} <= set(s) and len(s) == 10
    assert min(b - a for a, b in pairwise(s)) >= 6


def test_selection_handles_short_sparse_and_impossible_inputs():
    from bxjudge.labeling.sample_frames import select_frames

    assert select_frames({0: 1, 9: 2, 20: 3}, 10, 10, min_gap_s=1) == [0, 20]
    assert select_frames({}, 60, 8) == []
    with pytest.raises(ValueError):
        select_frames({0: 1}, 0, 1)


def test_change_energy_reads_adjacent_frames(tmp_path):
    from bxjudge.labeling.sample_frames import change_energy

    e = change_energy(video_file(tmp_path))
    assert len(e) == 13 and e[0] == 0
    assert e[6] > 200 and e[7] > 200


def test_to_label_studio_percent_units():
    from bxjudge.labeling.sam3_prelabel import to_label_studio

    t = to_label_studio("a.jpg", 1000, 500, [("bey", (100, 50, 200, 150), 0.9)])
    v = t["predictions"][0]["result"][0]["value"]
    assert (v["x"], v["y"], v["width"], v["height"], v["rectanglelabels"]) == (
        10.0,
        10.0,
        10.0,
        20.0,
        ["bey"],
    )


def test_prelabel_drops_elongated_launcher_boxes():
    from bxjudge.labeling.sam3_prelabel import filter_launcher_boxes

    boxes = [
        ("launcher", (100, 100, 140, 130), 0.9),
        ("launcher", (100, 100, 400, 110), 0.8),
        ("bey", (0, 0, 100, 1), 0.5),
    ]
    assert filter_launcher_boxes(boxes) == [boxes[0], boxes[2]]


class ExactModel:
    """Pinned Sam3MultiplexTrackingWithInteractivity signatures, never **kwargs."""

    def __init__(self, fail=False):
        self.states = []
        self.resources = []
        self.closed = 0
        self.fail = fail

    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
        input_is_mp4=False,
    ):
        files = sorted(Path(resource_path).glob("*.jpg"))
        assert offload_video_to_cpu and not async_loading_frames
        assert [p.name for p in files] == [f"{i:06d}.jpg" for i in range(len(files))]
        self.resources.append(len(files))
        state = {"n": len(files)}
        self.states.append(state)
        return state

    def reset_state(self, inference_state):
        pass

    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        clear_old_points=True,
        points=None,
        point_labels=None,
        boxes_xywh=None,
        box_labels=None,
        clear_old_boxes=True,
        output_prob_thresh=0.5,
        obj_id=None,
        rel_coordinates=True,
    ):
        assert frame_idx == 0
        return frame_idx, self.output()

    def output(self):
        return {
            "out_obj_ids": np.array([1]),
            "out_probs": np.array([0.9]),
            "out_boxes_xywh": np.array([[0.1, 0.2, 0.3, 0.4]]),
            "out_binary_masks": np.ones((1, 48, 64), bool),
        }

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        output_prob_thresh=0.5,
        compute_stability_score=False,
        is_instance_processing=False,
        is_last_batch=False,
    ):
        assert start_frame_idx == 0 and max_frame_num_to_track is None
        assert is_last_batch and not reverse
        try:
            for i in range(inference_state["n"]):
                if self.fail:
                    raise RuntimeError("injected propagation failure")
                yield i, self.output()
        finally:
            self.closed += 1


def test_bounded_direct_video_api_original_frames_and_cleanup(tmp_path, monkeypatch):
    from bxjudge.labeling import sam3_prelabel as sam

    model = ExactModel()
    monkeypatch.setattr(
        sam,
        "_build_predictor",
        lambda config: type(
            "Predictor", (), {"model": model, "shutdown": lambda self: None}
        )(),
    )
    out = tmp_path / "tasks.json"
    assert (
        sam._prelabel_direct(
            video_file(tmp_path),
            [0, 3, 7, 12],
            {"bey": "spinning top"},
            out,
            window_frames=5,
        )
        == 4
    )
    assert model.resources == [5, 5, 3] and all(not state for state in model.states)
    assert model.closed == 3
    tasks = json.loads(out.read_text())
    assert [t["data"]["frame"] for t in tasks] == [0, 3, 7, 12]
    assert len({t["data"]["video_id"] for t in tasks}) == 1
    assert all(Path(t["data"]["image_path"]).is_file() for t in tasks)
    assert tasks[0]["predictions"][0]["result"][0]["value"]["x"] == pytest.approx(10)


def test_failed_propagation_closes_iterator_clears_state(tmp_path, monkeypatch):
    from bxjudge.labeling import sam3_prelabel as sam

    model = ExactModel(fail=True)
    monkeypatch.setattr(
        sam,
        "_build_predictor",
        lambda config: type(
            "Predictor", (), {"model": model, "shutdown": lambda self: None}
        )(),
    )
    with pytest.raises(RuntimeError, match="injected"):
        sam._prelabel_direct(
            video_file(tmp_path),
            [0],
            {"bey": "top"},
            tmp_path / "tasks.json",
            window_frames=5,
        )
    assert model.closed == 1 and model.states == [{}]
    assert not (tmp_path / "tasks.json").exists()


def test_export_by_video_and_human_annotations_only(tmp_path):
    from bxjudge.labeling.export import label_studio_to_coco
    from bxjudge.labeling.sam3_prelabel import to_label_studio

    tasks = []
    for video in ["a", "a", "b", "b", "c"]:
        image = tmp_path / f"{len(tasks)}.jpg"
        cv2.imwrite(str(image), np.zeros((50, 100, 3), np.uint8))
        t = to_label_studio("ignored", 100, 50, [("launcher", (1, 2, 11, 12), 0.9)])
        t["data"].update(
            video_id=video,
            video_sha256=video * 64,
            frame=len(tasks),
            image_path=str(image),
        )
        t["annotations"] = [
            {"result": [] if len(tasks) == 0 else t["predictions"][0]["result"]}
        ]
        tasks.append(t)
    source = tmp_path / "human.json"
    source.write_text(json.dumps(tasks))
    out = tmp_path / "dataset"
    label_studio_to_coco(source, out)
    train = json.loads((out / "annotations/instances_train.json").read_text())
    val = json.loads((out / "annotations/instances_val.json").read_text())
    assert not (
        {i["video_id"] for i in train["images"]}
        & {i["video_id"] for i in val["images"]}
    )
    assert len(train["images"]) + len(val["images"]) == 5
    assert len(train["annotations"]) + len(val["annotations"]) == 4
    assert all(
        (out / i["file_name"]).is_file() for i in train["images"] + val["images"]
    )
    assert [c["name"] for c in train["categories"]] == ["bey", "bey_part", "launcher"]
    assert (out / "provenance.json").is_file()
    tasks[0].pop("annotations")
    source.write_text(json.dumps(tasks))
    with pytest.raises(ValueError, match="human"):
        label_studio_to_coco(source, tmp_path / "bad")


def test_label_cli_sample_writes_manifest(tmp_path):
    from bxjudge.cli import main

    video_file(tmp_path)
    out = tmp_path / "labels"
    assert (
        main(["label", "sample", str(tmp_path), "--per-video", "3", "--out", str(out)])
        == 0
    )
    sample = json.loads((out / "samples.json").read_text())
    assert len(sample["videos"]) == 1 and len(sample["videos"][0]["frames"]) == 3


def test_export_rejects_cancelled_and_rotated_annotations(tmp_path):
    from bxjudge.labeling.export import label_studio_to_coco
    from bxjudge.labeling.sam3_prelabel import to_label_studio

    image = tmp_path / "image.jpg"
    cv2.imwrite(str(image), np.zeros((48, 64, 3), np.uint8))
    t = to_label_studio("image", 64, 48, [("bey", (6.4, 9.6, 25.6, 28.8), 0.9)])
    t["data"].update(
        video_id="a", video_sha256="a" * 64, frame=0, image_path=str(image)
    )
    t["annotations"] = [
        {"was_cancelled": True, "result": t["predictions"][0]["result"]}
    ]
    source = tmp_path / "human.json"
    source.write_text(json.dumps([t]))
    with pytest.raises(ValueError, match="human"):
        label_studio_to_coco(source, tmp_path / "cancelled")
    t["annotations"][0]["was_cancelled"] = False
    t["annotations"][0]["result"][0]["value"]["rotation"] = 10
    source.write_text(json.dumps([t]))
    with pytest.raises(ValueError, match="rotated"):
        label_studio_to_coco(source, tmp_path / "rotated")


def test_non_square_normalized_box_roundtrip_and_local_asset(tmp_path, monkeypatch):
    from urllib.parse import parse_qs, urlparse

    from bxjudge.labeling import sam3_prelabel as sam
    from bxjudge.labeling.export import label_studio_to_coco

    model = ExactModel()
    monkeypatch.setattr(
        sam,
        "_build_predictor",
        lambda config: type("P", (), {"model": model, "shutdown": lambda self: None})(),
    )
    out = tmp_path / "tasks.json"
    sam._prelabel_direct(
        video_file(tmp_path), [3], {"bey": "top"}, out, window_frames=5
    )
    tasks = json.loads(out.read_text())
    t = tasks[0]
    asset = tmp_path / parse_qs(urlparse(t["data"]["image"]).query)["d"][0]
    assert asset.is_file() and asset.resolve() == Path(t["data"]["image_path"])
    r = t["predictions"][0]["result"][0]
    assert (r["original_width"], r["original_height"]) == (64, 48)
    assert [r["value"][k] for k in ["x", "y", "width", "height"]] == pytest.approx(
        [10, 20, 30, 40]
    )
    t["annotations"] = [{"result": [r]}]
    out.write_text(json.dumps(tasks))
    label_studio_to_coco(out, tmp_path / "dataset")
    coco = json.loads(
        (tmp_path / "dataset/annotations/instances_train.json").read_text()
    )
    assert coco["annotations"][0]["bbox"] == pytest.approx([6.4, 9.6, 19.2, 19.2])


def test_empty_job_does_not_load_gpu(monkeypatch):
    from bxjudge.labeling.sam3_prelabel import run_prelabel_jobs

    assert run_prelabel_jobs([], {}) == []


def test_config_rejects_unbounded_or_bad_policy(tmp_path):
    import yaml

    from bxjudge.labeling.config import load_config

    cfg = load_config()
    assert (
        cfg["uniform_ratio"],
        cfg["min_gap_s"],
        cfg["launcher_max_aspect"],
        cfg["val_ratio"],
        cfg["window_frames"],
    ) == (0.3, 0.1, 4.0, 0.2, 120)
    cfg["window_frames"] = 0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="window_frames"):
        load_config(path)


@pytest.mark.parametrize("value", [0, -4, 2.0, "4", True, None])
def test_config_rejects_bad_grounding_batch_size(tmp_path, value):
    import yaml

    from bxjudge.labeling.config import load_config

    cfg = load_config()
    # R47 measured value: 120-frame 1080p window peaked 12381 MiB at 4 vs ~24 GiB at 16.
    assert cfg["batched_grounding_batch_size"] == 4 and cfg["window_frames"] == 120
    cfg["batched_grounding_batch_size"] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="batched_grounding_batch_size"):
        load_config(path)


def _fake_sam_builder(monkeypatch, tmp_path, model):
    from types import ModuleType

    from bxjudge.labeling import sam3_prelabel as sam

    calls = []

    class Predictor:
        def __init__(self):
            self.model = model

    def build_sam3_predictor(**kwargs):
        calls.append(kwargs)
        return Predictor()

    builder = ModuleType("sam3.model_builder")
    builder.build_sam3_predictor = build_sam3_predictor
    monkeypatch.setitem(sys.modules, "sam3", ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder)
    monkeypatch.setattr(sam, "verify_checkpoint_keys", lambda m, c: {"ok": True})
    checkpoint = tmp_path / "sam.pt"
    checkpoint.touch()
    return sam, calls, checkpoint


def test_predictor_factory_applies_configured_grounding_batch(tmp_path, monkeypatch):
    from bxjudge.labeling.config import load_config

    # Pinned SDK builder hardcodes both batches to 16 on the public model attributes.
    model = SimpleNamespace(batched_grounding_batch_size=16, postprocess_batch_size=16)
    sam, calls, checkpoint = _fake_sam_builder(monkeypatch, tmp_path, model)
    cfg = {**load_config(), "checkpoint": str(checkpoint)}
    predictor = sam._build_predictor(cfg)
    assert predictor.model is model
    assert model.batched_grounding_batch_size == cfg["batched_grounding_batch_size"] == 4
    assert model.postprocess_batch_size == 16  # R47: postprocess batch unchanged
    assert len(calls) == 1 and "batched_grounding_batch_size" not in calls[0]
    cfg["batched_grounding_batch_size"] = 2
    sam._build_predictor(cfg)
    assert model.batched_grounding_batch_size == 2


def test_predictor_factory_fails_closed_if_sdk_attribute_changes(
    tmp_path, monkeypatch
):
    from bxjudge.labeling.config import load_config

    model = SimpleNamespace(postprocess_batch_size=16)
    sam, _, checkpoint = _fake_sam_builder(monkeypatch, tmp_path, model)
    with pytest.raises(RuntimeError, match="batched_grounding_batch_size"):
        sam._build_predictor({**load_config(), "checkpoint": str(checkpoint)})


def test_checkpoint_contract_rejects_missing_or_extra_model_tensors(tmp_path):
    import torch

    from bxjudge.labeling.sam3_prelabel import verify_checkpoint_keys

    model = torch.nn.Linear(2, 1)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    assert verify_checkpoint_keys(model, path) == {
        "checkpoint_keys": 2,
        "missing_keys": [],
        "unexpected_keys": [],
    }
    torch.save({"weight": model.weight}, path)
    with pytest.raises(ValueError, match="checkpoint"):
        verify_checkpoint_keys(model, path)


def test_export_split_uses_video_content_even_if_display_ids_differ(tmp_path):
    from bxjudge.labeling.export import label_studio_to_coco
    from bxjudge.labeling.sam3_prelabel import to_label_studio

    image = tmp_path / "image.jpg"
    cv2.imwrite(str(image), np.zeros((48, 64, 3), np.uint8))
    tasks = []
    for frame, display_id in enumerate(["copy-a", "copy-b"]):
        t = to_label_studio("image", 64, 48, [])
        t["data"].update(
            video_id=display_id,
            video_sha256="a" * 64,
            frame=frame,
            image_path=str(image),
        )
        t["annotations"] = [{"result": []}]
        tasks.append(t)
    export = tmp_path / "human.json"
    export.write_text(json.dumps(tasks))
    label_studio_to_coco(export, tmp_path / "dataset", val_ratio=0.5)
    train = json.loads(
        (tmp_path / "dataset/annotations/instances_train.json").read_text()
    )
    val = json.loads((tmp_path / "dataset/annotations/instances_val.json").read_text())
    assert len(train["images"]) == 2 and not val["images"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", range(5))
def test_sam_output_rejects_nonfinite_before_clipping(value, field):
    from bxjudge.labeling.sam3_prelabel import _boxes

    row = [0.1, 0.2, 0.3, 0.4, 0.9]
    row[field] = value
    with pytest.raises(ValueError, match="finite"):
        _boxes({"out_boxes_xywh": [row[:4]], "out_probs": [row[4]]}, 1000, 500)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_label_studio_rejects_invalid_probability(score):
    from bxjudge.labeling.sam3_prelabel import to_label_studio

    with pytest.raises(ValueError, match="score"):
        to_label_studio("image.jpg", 100, 50, [("bey", (1, 2, 11, 12), score)])


def test_sample_cli_rejects_identical_copies_before_video_sampling(
    tmp_path, monkeypatch
):
    import shutil

    from bxjudge.cli import main
    from bxjudge.labeling import cli as label_cli

    original = video_file(tmp_path)
    duplicate = tmp_path / "copy.mp4"
    shutil.copyfile(original, duplicate)
    out = tmp_path / "labels"

    def unexpected_probe(path):
        raise AssertionError("duplicate videos must fail before sampling starts")

    monkeypatch.setattr(label_cli, "probe", unexpected_probe)
    with pytest.raises(ValueError, match="duplicate video content") as error:
        main(["--debug", "label", "sample", str(tmp_path), "--out", str(out)])
    assert str(original) in str(error.value) and str(duplicate) in str(error.value)
    assert not out.exists()


def test_prelabel_cli_rejects_duplicate_manifest_before_gpu(tmp_path, monkeypatch):
    import hashlib
    import shutil

    from bxjudge.cli import main
    from bxjudge.labeling import sam3_prelabel as sam

    original = video_file(tmp_path)
    duplicate = tmp_path / "copy.mp4"
    shutil.copyfile(original, duplicate)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    manifest = tmp_path / "samples.json"
    manifest.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video": str(path),
                        "video_sha256": digest,
                        "fps": 10,
                        "frames": [0],
                    }
                    for path in [original, duplicate]
                ]
            }
        )
    )
    out = tmp_path / "labels"

    def unexpected_worker(jobs, config):
        raise AssertionError("duplicate manifest must fail before SAM worker starts")

    monkeypatch.setattr(sam, "run_prelabel_jobs", unexpected_worker)
    with pytest.raises(ValueError, match="duplicate video content") as error:
        main(["--debug", "label", "prelabel", "--samples", str(manifest), "--out", str(out)])
    assert str(original) in str(error.value) and str(duplicate) in str(error.value)
    assert not out.exists()


@pytest.mark.parametrize("prompts", [None, [], ["bey", "bey_part", "launcher"], "bey", 1])
def test_config_rejects_non_mapping_prompts(tmp_path, prompts):
    import yaml

    from bxjudge.labeling.config import load_config

    cfg = load_config()
    cfg["prompts"] = prompts
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="prompts"):
        load_config(path)
