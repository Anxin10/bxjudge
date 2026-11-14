"""Labeling contracts: frame sampling, isolated prelabels and reviewed export."""

import json
from itertools import pairwise
from pathlib import Path

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
    from bxjudge.labeling.prelabel import to_label_studio

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
    from bxjudge.labeling.prelabel import filter_launcher_boxes

    boxes = [
        ("launcher", (100, 100, 140, 130), 0.9),
        ("launcher", (100, 100, 400, 110), 0.8),
        ("bey", (0, 0, 100, 1), 0.5),
    ]
    assert filter_launcher_boxes(boxes) == [boxes[0], boxes[2]]


def test_prelabel_infers_only_selected_frames_and_records_provenance(tmp_path):
    from bxjudge.labeling import prelabel as labeling
    from bxjudge.labeling.config import load_config

    observed = []

    def predict(frame):
        observed.append(int(frame[0, 0, 0]))
        return [("bey", (6.4, 9.6, 25.6, 28.8), None)]

    out = tmp_path / "tasks.json"
    video = video_file(tmp_path)
    assert labeling.prelabel_video(video, [12, 0, 6], out, load_config(), predict) == 3
    assert observed[0] == observed[2] == 0 and observed[1] > 240
    tasks = json.loads(out.read_text())
    assert [t["data"]["frame"] for t in tasks] == [0, 6, 12]
    assert all(t["data"]["video_sha256"] == labeling.video_digest(video) for t in tasks)
    assert all(Path(t["data"]["image_path"]).is_file() for t in tasks)
    prediction = tasks[0]["predictions"][0]
    assert prediction["model_version"].startswith("locateanything-3b-")
    assert "score" not in prediction["result"][0]
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["frames"] == [0, 6, 12] and report["boxes"] == 3
    assert report["model_revision"] in prediction["model_version"]


def test_failed_inference_leaves_no_task_json(tmp_path):
    from bxjudge.labeling.config import load_config
    from bxjudge.labeling.prelabel import prelabel_video

    def predict(frame):
        raise RuntimeError("injected inference failure")

    out = tmp_path / "tasks.json"
    with pytest.raises(RuntimeError, match="injected"):
        prelabel_video(video_file(tmp_path), [0], out, load_config(), predict)
    assert not out.exists()
    assert not out.with_suffix(".report.json").exists()


def test_export_by_video_and_human_annotations_only(tmp_path):
    from bxjudge.labeling.export import label_studio_to_coco
    from bxjudge.labeling.prelabel import to_label_studio

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
    from bxjudge.labeling.prelabel import to_label_studio

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


def test_export_split_swaps_in_a_video_so_val_has_every_class(tmp_path):
    from bxjudge.labeling.export import label_studio_to_coco
    from bxjudge.labeling.prelabel import to_label_studio

    # Hash order puts only "a" in val (5 videos x 0.2), but bey_part lives in d and e.
    parts = {"a": 0, "b": 0, "c": 0, "d": 2, "e": 1}
    tasks = []
    for video, n_parts in parts.items():
        image = tmp_path / f"{video}.jpg"
        cv2.imwrite(str(image), np.zeros((50, 100, 3), np.uint8))
        boxes = [("bey", (1, 2, 11, 12), 0.9), ("launcher", (40, 2, 50, 12), 0.9)]
        boxes += [("bey_part", (60 + 5 * i, 30, 64 + 5 * i, 34), 0.9) for i in range(n_parts)]
        t = to_label_studio("ignored", 100, 50, boxes)
        t["data"].update(
            video_id=video, video_sha256=video * 64, frame=0, image_path=str(image)
        )
        t["annotations"] = [{"result": t["predictions"][0]["result"]}]
        tasks.append(t)
    source = tmp_path / "human.json"
    source.write_text(json.dumps(tasks))
    out = tmp_path / "dataset"
    label_studio_to_coco(source, out, val_ratio=0.2)
    split = {
        s: json.loads((out / f"annotations/instances_{s}.json").read_text())
        for s in ("train", "val")
    }
    # The part holder with the fewest parts (e) replaces a; val size stays 1.
    assert {i["video_id"] for i in split["val"]["images"]} == {"e"}
    assert {i["video_id"] for i in split["train"]["images"]} == {"a", "b", "c", "d"}
    for s in ("train", "val"):
        assert {a["category_id"] for a in split[s]["annotations"]} == {1, 2, 3}
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["val_class_swaps"] == [
        {"category": "bey_part", "in": "e" * 64, "out": "a" * 64}
    ]


def test_export_accepts_edge_snapped_box_and_rejects_real_overflow(tmp_path):
    from bxjudge.labeling.export import label_studio_to_coco
    from bxjudge.labeling.prelabel import to_label_studio

    image = tmp_path / "image.jpg"
    cv2.imwrite(str(image), np.zeros((48, 64, 3), np.uint8))
    t = to_label_studio("image", 64, 48, [("launcher", (6.4, 0.0, 25.6, 9.6), 0.9)])
    t["data"].update(
        video_id="a", video_sha256="a" * 64, frame=0, image_path=str(image)
    )
    t["annotations"] = [{"result": t["predictions"][0]["result"]}]
    value = t["annotations"][0]["result"][0]["value"]
    # Label Studio stores a box snapped to the top edge as a tiny negative float.
    value["y"] = -1e-13
    source = tmp_path / "human.json"
    source.write_text(json.dumps([t]))
    out = tmp_path / "snapped"
    label_studio_to_coco(source, out)
    [ann] = json.loads((out / "annotations/instances_train.json").read_text())[
        "annotations"
    ]
    assert ann["bbox"][1] == 0.0
    assert ann["bbox"][3] == pytest.approx(9.6)
    value["y"] = -1.0
    source.write_text(json.dumps([t]))
    with pytest.raises(ValueError, match="outside image"):
        label_studio_to_coco(source, tmp_path / "overflow")


def test_non_square_normalized_box_roundtrip_and_local_asset(tmp_path):
    from urllib.parse import parse_qs, urlparse

    from bxjudge.labeling.config import load_config
    from bxjudge.labeling.export import label_studio_to_coco
    from bxjudge.labeling.locateanything import parse_boxes
    from bxjudge.labeling.prelabel import prelabel_video

    cfg = load_config()
    out = tmp_path / "tasks.json"

    def predict(frame):
        h, w = frame.shape[:2]
        return parse_boxes("<ref>spinning top</ref><box><100><200><400><600></box>",
                           cfg["prompts"], w, h)

    prelabel_video(video_file(tmp_path), [3], out, cfg, predict)
    tasks = json.loads(out.read_text())
    t = tasks[0]
    asset = tmp_path / parse_qs(urlparse(t["data"]["image"]).query)["d"][0]
    assert asset.is_file() and asset.resolve() == Path(t["data"]["image_path"])
    r = t["predictions"][0]["result"][0]
    assert (r["original_width"], r["original_height"]) == (64, 48)
    assert [r["value"][k] for k in ["x", "y", "width", "height"]] == pytest.approx([10, 20, 30, 40])
    t["annotations"] = [{"result": [r]}]
    out.write_text(json.dumps(tasks))
    label_studio_to_coco(out, tmp_path / "dataset")
    coco = json.loads((tmp_path / "dataset/annotations/instances_train.json").read_text())
    assert coco["annotations"][0]["bbox"] == pytest.approx([6.4, 9.6, 19.2, 19.2])


def test_empty_job_does_not_load_gpu(monkeypatch):
    from bxjudge.labeling.prelabel import run_prelabel_jobs

    assert run_prelabel_jobs([], {}) == []


@pytest.mark.parametrize("key,value", [
    ("max_new_tokens", 0), ("max_new_tokens", True), ("model_dir", ""),
    ("prompts", {"bey": "top", "bey_part": "Top", "launcher": "toy launcher"}),
    ("prompts", {"bey": "top</c>person", "bey_part": "part", "launcher": "toy launcher"}),
    ("window_frames", 120),
])
def test_config_rejects_invalid_locateanything_policy(tmp_path, key, value):
    import yaml

    from bxjudge.labeling.config import load_config

    cfg = load_config()
    assert cfg["max_new_tokens"] == 8192
    cfg[key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_config(path)


def test_locateanything_parser_uses_exact_labels_and_keeps_only_valid_boxes():
    from bxjudge.labeling.locateanything import parse_boxes

    answer = (
        "<box><1><2><3><4></box>"
        "<ref> Spinning TOP </ref><box><100><200><300><400></box>"
        "<box><500><600><1000><1000></box><box><100><100><100><200></box>"
        "<box><1><1><1001><4></box><box><-1><1><4><4></box><box><1><1></box>"
        "<ref>unknown</ref><box><1><2><3><4></box>"
        "<ref>toy launcher</ref><box>None</box>"
        "<ref>small plastic part</ref><box><0><0><100><100></box><|im_end|>"
    )
    prompts = {"bey": "spinning top", "bey_part": "small plastic part", "launcher": "toy launcher"}
    assert parse_boxes(answer, prompts, 1000, 500) == [
        ("bey", (100.0, 100.0, 300.0, 200.0), None),
        ("bey", (500.0, 300.0, 1000.0, 500.0), None),
        ("bey_part", (0.0, 0.0, 100.0, 50.0), None),
    ]


def test_default_cli_prelabel_dispatches_locateanything_job(tmp_path, monkeypatch):
    from bxjudge.cli import main
    from bxjudge.labeling import prelabel as labeling

    video = video_file(tmp_path)
    digest = labeling.video_digest(video)
    manifest = tmp_path / "samples.json"
    manifest.write_text(json.dumps({"videos": [
        {"video": str(video), "video_sha256": digest, "fps": 10, "frames": [6]}
    ]}))
    out = tmp_path / "labels"

    def worker(jobs, config):
        assert len(jobs) == 1 and jobs[0]["frames"] == [6]
        assert config["model_dir"].startswith("weights/locateanything/")
        assert config["prompts"]["bey"] == "spinning top"
        return [labeling.prelabel_video(video, [6], Path(jobs[0]["out_json"]), config,
                                        lambda frame: [("bey", (1, 2, 11, 12), None)])]

    monkeypatch.setattr(labeling, "run_prelabel_jobs", worker)
    assert main(["--debug", "label", "prelabel", "--samples", str(manifest), "--out", str(out)]) == 0
    [task] = json.loads((out / "tasks.json").read_text())
    assert task["data"]["frame"] == 6
    assert task["predictions"][0]["model_version"].startswith("locateanything-3b-")


def test_export_split_uses_video_content_even_if_display_ids_differ(tmp_path):
    from bxjudge.labeling.export import label_studio_to_coco
    from bxjudge.labeling.prelabel import to_label_studio

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


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_label_studio_rejects_invalid_probability(score):
    from bxjudge.labeling.prelabel import to_label_studio

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
    from bxjudge.labeling import prelabel as labeling

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
        raise AssertionError("duplicate manifest must fail before the worker starts")

    monkeypatch.setattr(labeling, "run_prelabel_jobs", unexpected_worker)
    with pytest.raises(ValueError, match="duplicate video content") as error:
        main(["--debug", "label", "prelabel", "--samples", str(manifest), "--out", str(out)])
    assert str(original) in str(error.value) and str(duplicate) in str(error.value)
    assert not out.exists()



def test_changed_source_does_not_overwrite_existing_prelabel_asset(tmp_path):
    from bxjudge.labeling.config import load_config
    from bxjudge.labeling.prelabel import prelabel_video, video_digest

    video = video_file(tmp_path)
    out = tmp_path / "tasks.json"
    asset = tmp_path / "images" / video_digest(video) / "000000000.jpg"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"previous image")
    out.write_text("previous tasks")

    def predict(frame):
        with video.open("ab") as stream:
            stream.write(b"changed")
        return []

    with pytest.raises(ValueError, match="changed during prelabeling"):
        prelabel_video(video, [0], out, load_config(), predict)
    assert asset.read_bytes() == b"previous image"
    assert out.read_text() == "previous tasks"


def test_verify_model_rejects_tampered_code_before_import(tmp_path, monkeypatch):
    import hashlib

    from bxjudge.labeling import locateanything

    model = tmp_path / "model"
    model.mkdir()
    code = model / "modeling_locateanything.py"
    code.write_text("official")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "repo": locateanything.MODEL_ID,
        "revision": locateanything.MODEL_REVISION,
        "files": {code.name: hashlib.sha256(code.read_bytes()).hexdigest()},
    }))
    monkeypatch.setattr(locateanything, "MODEL_MANIFEST", manifest)
    locateanything.verify_model(model)
    extra = model / "tokenizer.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="Unexpected"):
        locateanything.verify_model(model)
    extra.unlink()
    code.write_text("modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        locateanything.LocateAnything({"model_dir": str(model)})



def test_prelabel_rejects_a_corrupt_existing_image(tmp_path):
    from bxjudge.labeling.config import load_config
    from bxjudge.labeling.prelabel import prelabel_video, video_digest

    video = video_file(tmp_path)
    out = tmp_path / "tasks.json"
    asset = tmp_path / "images" / video_digest(video) / "000000000.jpg"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"")
    with pytest.raises(ValueError, match="Existing prelabel image"):
        prelabel_video(video, [0], out, load_config(), lambda frame: [])
    assert not out.exists()
    assert asset.read_bytes() == b""


def test_model_response_requires_end_marker():
    from bxjudge.labeling.locateanything import parse_response

    answer = "<ref>spinning top</ref><box>None</box>"
    prompts = {"bey": "spinning top"}
    with pytest.raises(ValueError, match="did not finish"):
        parse_response(answer, prompts, 640, 360)
    assert parse_response(answer + "<|im_end|>", prompts, 640, 360) == []
