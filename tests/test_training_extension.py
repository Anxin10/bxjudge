"""Human additions must never change the frozen validation corpus."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from bxjudge.labeling.export import label_studio_to_coco
from bxjudge.labeling.sam3_prelabel import to_label_studio


def task(folder, video, frame, *, reviewed=True, background=False):
    image = folder / f"{video}-{frame}.jpg"
    assert cv2.imwrite(str(image), np.full((50, 100, 3), frame + 20, np.uint8))
    result = to_label_studio("unused", 100, 50, [("bey", (1, 2, 11, 12), 0.9)])
    result["data"].update(video_id=video, video_sha256=video * 64, frame=frame,
                          image_path=str(image), training_use="train_candidate")
    if reviewed:
        result["annotations"] = [{"result": [] if background else result["predictions"][0]["result"]}]
    return result


def hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}


def read(folder, split):
    return json.loads((folder / "annotations" / f"instances_{split}.json").read_text())


@pytest.fixture
def baseline(tmp_path):
    source = tmp_path / "initial-human.json"
    source.write_text(json.dumps([task(tmp_path, v, 0) for v in "abc"]))
    folder = tmp_path / "base"
    label_studio_to_coco(source, folder, val_ratio=0.34)
    assert [x["video_id"] for x in read(folder, "val")["images"]] == ["a"]
    return folder


def extend(base, source, out):
    from bxjudge.labeling.export import extend_training_dataset

    return extend_training_dataset(base, source, out)


def test_training_extension_preserves_validation_bytes_and_existing_training_labels(tmp_path, baseline):
    before = hashes(baseline)
    source = tmp_path / "new-human.json"
    source.write_text(json.dumps([task(tmp_path, "b", 1), task(tmp_path, "c", 2)]))
    out = tmp_path / "extended"
    extend(baseline, source, out)
    assert hashes(baseline) == before
    assert (out / "annotations/instances_val.json").read_bytes() == (
        baseline / "annotations/instances_val.json").read_bytes()
    old_train, new_train = read(baseline, "train"), read(out, "train")
    assert new_train["images"][:2] == old_train["images"]
    assert new_train["annotations"][:2] == old_train["annotations"]
    assert len(new_train["images"]) == len(new_train["annotations"]) == 4
    ids = [i["id"] for split in ["train", "val"] for i in read(out, split)["images"]]
    assert len(ids) == len(set(ids)) == 5
    for image in read(out, "val")["images"]:
        assert (out / image["file_name"]).read_bytes() == (baseline / image["file_name"]).read_bytes()
    provenance = json.loads((out / "provenance.json").read_text())
    assert len(provenance["images"]) == 5
    assert provenance["training_extension"]["images_added"] == 2
    assert provenance["training_extension"]["human_export_sha256"] == hashlib.sha256(
        source.read_bytes()).hexdigest()


@pytest.mark.parametrize("video", ["a", "d"])
def test_validation_and_unassigned_videos_cannot_enter_training(tmp_path, baseline, video):
    before = hashes(baseline)
    source = tmp_path / "wrong-video.json"
    source.write_text(json.dumps([task(tmp_path, video, 2)]))
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="existing training video"):
        extend(baseline, source, out)
    assert not out.exists() and hashes(baseline) == before


def test_existing_source_frame_is_never_replaced(tmp_path, baseline):
    source = tmp_path / "duplicate.json"
    source.write_text(json.dumps([task(tmp_path, "b", 0)]))
    with pytest.raises(ValueError, match="already exists"):
        extend(baseline, source, tmp_path / "rejected")


def test_predictions_without_human_submit_cannot_extend_training(tmp_path, baseline):
    source = tmp_path / "prelabels.json"
    source.write_text(json.dumps([task(tmp_path, "b", 1, reviewed=False)]))
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="completed human annotation"):
        extend(baseline, source, out)
    assert not out.exists()


@pytest.mark.parametrize("destination", ["inside", "existing"])
def test_input_or_existing_output_is_never_overwritten(tmp_path, baseline, destination):
    source = tmp_path / "human.json"
    source.write_text(json.dumps([task(tmp_path, "b", 1)]))
    out = baseline / "nested" if destination == "inside" else tmp_path / "existing"
    if destination == "existing":
        out.mkdir()
    before = hashes(baseline)
    with pytest.raises(ValueError, match="output"):
        extend(baseline, source, out)
    assert hashes(baseline) == before


def test_a_human_reviewed_background_frame_is_valid_training_data(tmp_path, baseline):
    source = tmp_path / "background.json"
    source.write_text(json.dumps([task(tmp_path, "b", 1, background=True)]))
    out = tmp_path / "extended"
    extend(baseline, source, out)
    assert len(read(out, "train")["images"]) == 3
    assert len(read(out, "train")["annotations"]) == 2


def test_inconsistent_baseline_provenance_is_rejected(tmp_path, baseline):
    path = baseline / "provenance.json"
    data = json.loads(path.read_text())
    data["images"][0]["split"] = "train"
    path.write_text(json.dumps(data))
    source = tmp_path / "human.json"
    source.write_text(json.dumps([task(tmp_path, "b", 1)]))
    with pytest.raises(ValueError, match="baseline provenance"):
        extend(baseline, source, tmp_path / "rejected")


def test_cli_exports_training_extension_without_loading_a_model(tmp_path, baseline):
    from bxjudge.cli import main

    source = tmp_path / "human.json"
    source.write_text(json.dumps([task(tmp_path, "b", 1)]))
    out = tmp_path / "extended"
    assert main(["label", "extend-training", str(baseline), str(source), "--out", str(out)]) == 0
    assert len(read(out, "train")["images"]) == 3


def test_training_video_hash_cannot_carry_a_validation_video_id(tmp_path, baseline):
    item = task(tmp_path, "b", 1)
    item["data"]["video_id"] = "a"
    source = tmp_path / "mixed-identity.json"
    source.write_text(json.dumps([item]))
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="video identity"):
        extend(baseline, source, out)
    assert not out.exists()


def test_new_image_filename_cannot_replace_a_validation_image(tmp_path, baseline):
    val_path = baseline / "annotations/instances_val.json"
    val = json.loads(val_path.read_text())
    old_name = val["images"][0]["file_name"]
    new_name = "images/train/000000004.jpg"
    (baseline / old_name).rename(baseline / new_name)
    val["images"][0]["file_name"] = new_name
    val_path.write_text(json.dumps(val))
    meta_path = baseline / "provenance.json"
    metadata = json.loads(meta_path.read_text())
    metadata["images"][0]["file_name"] = new_name
    meta_path.write_text(json.dumps(metadata))
    before = hashes(baseline)
    source = tmp_path / "new-human.json"
    source.write_text(json.dumps([task(tmp_path, "b", 1)]))
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="image path"):
        extend(baseline, source, out)
    assert not out.exists() and hashes(baseline) == before


def test_baseline_change_between_parsing_and_copy_is_rejected(tmp_path, baseline, monkeypatch):
    from bxjudge.labeling import export

    original = export._extension_baseline

    def changed_after_parse(folder):
        snapshot = original(folder)
        path = folder / "annotations/instances_train.json"
        changed = json.loads(path.read_text())
        changed["annotations"][0]["bbox"] = [9, 2, 3, 4]
        path.write_text(json.dumps(changed))
        return snapshot

    monkeypatch.setattr(export, "_extension_baseline", changed_after_parse)
    source = tmp_path / "human.json"
    source.write_text(json.dumps([task(tmp_path, "b", 1)]))
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="baseline changed|input changed"):
        extend(baseline, source, out)
    assert not out.exists()
    assert read(baseline, "train")["annotations"][0]["bbox"] == [9, 2, 3, 4]


def test_annotation_image_bytes_cannot_change_between_decode_and_copy(tmp_path, baseline, monkeypatch):
    from bxjudge.labeling import export

    item = task(tmp_path, "b", 1)
    image = tmp_path / "b-1.jpg"
    annotated_bytes = image.read_bytes()
    original_read, original_decode = export.cv2.imread, export.cv2.imdecode
    changed = []

    def replace_image():
        if not changed:
            assert cv2.imwrite(str(image), np.full((50, 100, 3), 99, np.uint8))
            changed.append(True)

    def after_read(path, *args):
        result = original_read(path, *args)
        if str(path) == str(image):
            replace_image()
        return result

    def after_decode(*args):
        result = original_decode(*args)
        replace_image()
        return result

    monkeypatch.setattr(export.cv2, "imread", after_read)
    monkeypatch.setattr(export.cv2, "imdecode", after_decode)
    source = tmp_path / "human.json"
    source.write_text(json.dumps([item]))
    out = tmp_path / "extended"
    extend(baseline, source, out)
    assert changed and image.read_bytes() != annotated_bytes
    new_image = read(out, "train")["images"][-1]
    assert (out / new_image["file_name"]).read_bytes() == annotated_bytes
    metadata = json.loads((out / "provenance.json").read_text())["images"][-1]
    assert metadata["image_sha256"] == hashlib.sha256(annotated_bytes).hexdigest()
    assert metadata["image_path"] == str(image)
