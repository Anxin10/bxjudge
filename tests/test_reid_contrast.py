import copy
import json
from pathlib import Path

import pytest

from scripts.analyze_reid_contrast import build, select_pairs
from tests.test_reid_appearance import fixture, sha


def row(frame=1):
    return {
        "video": "synthetic.mp4",
        "frame": frame,
        "confirmed": True,
        "missed_or_bad_detection": False,
        "objects": [
            {
                "index": i,
                "detector_class": "bey",
                "side": side,
                "state": "separated",
                "box": box,
                "score": 0.8,
            }
            for i, side, box in [(0, "X", [4, 8, 20, 24]), (1, "B", [36, 8, 52, 24])]
        ],
    }


def test_negative_relations_use_only_unique_confirmed_simultaneous_separated_sides():
    rows = [row(i) for i in range(6)]
    rows[1]["objects"][0]["state"] = "mounted"
    rows[2]["confirmed"] = False
    rows[3]["missed_or_bad_detection"] = True
    rows[4]["objects"].append(copy.deepcopy(rows[4]["objects"][0]))
    rows[5]["objects"][1]["side"] = "unknown"
    chosen, population = select_pairs(rows)
    assert [r["frame"] for r in chosen] == [0]
    assert population == {
        "reference_frames": 6,
        "eligible_frames": 1,
        "unconfirmed": 1,
        "bad_detection": 1,
        "non_unique_sides": 2,
        "not_separated": 1,
    }


def test_duplicate_source_frame_and_identical_boxes_cannot_be_negative_gt():
    with pytest.raises(ValueError, match="duplicate"):
        select_pairs([row(), row()])
    r = row()
    r["objects"][1]["box"] = r["objects"][0]["box"]
    with pytest.raises(ValueError, match="distinct"):
        select_pairs([r])


def inputs(tmp_path):
    from scripts.analyze_reid_appearance import build as appearance_build

    m, labels, annotations, config = fixture(tmp_path)
    appearance = tmp_path / "appearance"
    appearance_build(m, labels, annotations, config, appearance)
    source = tmp_path / "synthetic.mp4"
    r = row(2)
    r.update(
        source_sha256=sha(source),
        fps=10,
        coordinate_size=[64, 48],
        image_sha256="a" * 64,
    )
    reference = tmp_path / "confirmed.json"
    reference.write_text(json.dumps([r]))
    pointer = tmp_path / "pointer.json"
    pointer.write_text(
        json.dumps(
            {
                "reference": str(reference),
                "sha256": sha(reference),
                "revision": "b" * 64,
            }
        )
    )
    fm = {
        "frame": 2,
        "image_sha256": "a" * 64,
        "objects": [
            {"index": o["index"], "box": o["box"], "score": o["score"], "class": "bey"}
            for o in r["objects"]
        ],
    }
    manifest = tmp_path / "identity-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "video": source.name,
                        "source": str(source),
                        "source_sha256": sha(source),
                        "coordinate_size": [64, 48],
                        "fps": 10,
                        "frames": [fm],
                    }
                ]
            }
        )
    )
    return pointer, manifest, appearance, config


def test_build_freezes_exact_human_boxes_no_track_or_side_propagation(tmp_path):
    args = inputs(tmp_path)
    out = tmp_path / "contrast"
    before = {str(p): sha(p) for p in (args[0], args[1], args[3])}
    report = build(*args, out)
    assert report["population"]["eligible_frames"] == 1
    assert report["negative_frame_pairs"] == 1
    pair = report["cases"][0]
    assert pair["relation"] == "different" and pair["applied"] is False
    assert pair["frame"] == 2 and set(pair["samples"]) == {"X", "B"}
    assert pair["distances"]["color_hellinger"] == pytest.approx(0)
    assert pair["samples"]["X"]["source_box"] == [4, 8, 20, 24]
    assert report["decision"] == "no_threshold_selected"
    assert report["positive_cases"] == 1
    assert all(sha(p) == digest for p, digest in before.items())
    assert (out / "index.html").exists()
    assert "不同顆" in (out / "index.html").read_text()
    with pytest.raises(ValueError, match="new directory"):
        build(*args, out)


def test_reference_tamper_or_unverified_box_never_publishes(tmp_path):
    args = inputs(tmp_path)
    pointer = json.loads(args[0].read_bytes())
    reference = Path(pointer["reference"])
    reference.write_text("[]")
    with pytest.raises(ValueError, match="SHA"):
        build(*args, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_no_negatives_is_unverified_not_perfect_accuracy(tmp_path):
    args = inputs(tmp_path)
    p = json.loads(args[0].read_bytes())
    ref = Path(p["reference"])
    data = json.loads(ref.read_bytes())
    data[0]["objects"][0]["state"] = "mounted"
    ref.write_text(json.dumps(data))
    p["sha256"] = sha(ref)
    args[0].write_text(json.dumps(p))
    report = build(*args, tmp_path / "out")
    assert report["negative_frame_pairs"] == 0
    assert report["comparison"]["color_hellinger"]["negative"] is None
    assert report["decision"] == "no_threshold_selected"


def test_changed_human_box_cannot_be_attached_to_the_original_frame(tmp_path):
    args = inputs(tmp_path)
    p = json.loads(args[0].read_bytes())
    ref = Path(p["reference"])
    rows = json.loads(ref.read_bytes())
    rows[0]["objects"][0]["box"][0] += 1
    ref.write_text(json.dumps(rows))
    p["sha256"] = sha(ref)
    args[0].write_text(json.dumps(p))
    with pytest.raises(ValueError, match="original observation"):
        build(*args, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_decode_failure_does_not_publish_and_closes_stream(tmp_path, monkeypatch):
    import numpy as np

    from scripts import analyze_reid_contrast as module

    args = inputs(tmp_path)
    closed = []

    def broken(*args, **kwargs):
        try:
            yield 0, np.zeros((48, 64, 3), np.uint8)
            raise RuntimeError("synthetic failure")
        finally:
            closed.append(True)

    monkeypatch.setattr(module, "iter_frames", broken)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        build(*args, tmp_path / "out")
    assert closed == [True]
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".reid-contrast-*"))


def test_quality_rejection_keeps_the_human_negative_population(tmp_path):
    import yaml

    from scripts.analyze_reid_appearance import build as appearance_build

    m, labels, annotations, cfg = fixture(tmp_path)
    policy = yaml.safe_load(cfg.read_text())
    policy["min_crop_side_px"] = 32
    cfg.write_text(yaml.safe_dump(policy))
    positive = tmp_path / "appearance"
    appearance_build(m, labels, annotations, cfg, positive)
    # Create the human reference inputs in a separate synthetic fixture root.
    other = tmp_path / "other"
    other.mkdir()
    pointer, manifest, _, _ = inputs(other)
    r = build(pointer, manifest, positive, cfg, tmp_path / "out")
    assert r["negative_frame_pairs"] == 1 and r["usable_negative_pairs"] == 0
    assert r["cases"][0]["distances"] is None
    assert r["comparison"]["color_hellinger"]["negative"] is None
    assert r["comparison"]["color_hellinger"]["range_overlap"] is None
    assert r["cases"][0]["samples"]["X"]["reason_codes"] == ["small_crop"]


@pytest.mark.parametrize(
    "tamper", ["crop", "frozen", "positive_version", "missing_frozen_hash"]
)
def test_publication_validates_crops_frozen_reference_and_positive_evidence(
    tmp_path, tamper
):
    from scripts.build_workbench import validate_contrast_set

    args = inputs(tmp_path)
    out = tmp_path / "out"
    r = build(*args, out)
    validate_contrast_set(out, args[2])
    if tamper == "crop":
        (out / r["cases"][0]["samples"]["X"]["crop_file"]).write_bytes(b"bad")
    elif tamper == "frozen":
        (out / "confirmed-reference.json").write_text("[]")
    elif tamper == "positive_version":
        (args[2] / "report.json").write_text("{}")
    else:
        r["frozen_files_sha256"] = {}
        (out / "report.json").write_text(json.dumps(r))
    with pytest.raises(ValueError, match="SHA|version"):
        validate_contrast_set(out, args[2])
