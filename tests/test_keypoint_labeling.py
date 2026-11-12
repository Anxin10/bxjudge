"""scripts/keypoint_labeling.py: frame sampling, prelabels and the Label Studio config."""
from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from bxjudge import PROJECT_ROOT


def _load():
    spec = importlib.util.spec_from_file_location(
        "keypoint_labeling", PROJECT_ROOT / "scripts" / "keypoint_labeling.py")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod  # dataclasses look their module up while being defined
    spec.loader.exec_module(mod)
    return mod


kl = _load()


def _videos(tmp_path, spec):
    out = []
    for batch, n in spec.items():
        for i in range(n):
            out.append(kl.VideoRef(path=tmp_path / batch / f"v{i}.mp4", batch=batch, duration_s=10.0))
    return out


def test_ten_keypoints_and_left_right_pairs():
    assert len(kl.KEYPOINTS) == 10 and len(set(kl.KEYPOINTS)) == 10
    assert kl.FLIP_IDX == [1, 0, 3, 2, 5, 4, 8, 9, 6, 7]
    for i, j in enumerate(kl.FLIP_IDX):
        a, b = kl.KEYPOINTS[i], kl.KEYPOINTS[j]
        assert a.replace("left", "X").replace("right", "X") == b.replace("left", "X").replace("right", "X")
        assert ("left" in a) != ("left" in b)


def test_sampling_is_stratified_capped_and_repeatable(tmp_path):
    videos = _videos(tmp_path, {"a": 3, "b": 40, "c": 10})
    quota = {"a": 5, "b": 20, "c": 10}
    frames = kl.sample_keypoint_frames(videos, quota, seed=7)
    assert len(frames) == 35
    for batch, n in quota.items():
        picked = [f for f in frames if f.video.batch == batch]
        assert len(picked) == min(n, 2 * sum(v.batch == batch for v in videos))
        per_video = {}
        for f in picked:
            per_video[f.video.path] = per_video.get(f.video.path, 0) + 1
        assert max(per_video.values()) <= 2
    assert all(0.5 <= f.time_s <= 9.5 for f in frames)
    assert kl.sample_keypoint_frames(videos, quota, seed=7) == frames


def test_label_config_has_only_the_ten_points():
    # 2026-11-11: a stadium box covering the image swallowed every click meant for a point,
    # so the box is not labelled; the export derives it from the points (plan K3).
    root = ET.fromstring(kl.label_config())
    kps = root.find("KeyPointLabels")
    assert [l.get("value") for l in kps.findall("Label")] == [kl.LABEL_TEXT[k] for k in kl.KEYPOINTS]
    assert root.find("RectangleLabels") is None


def test_prediction_uses_percent_coordinates_and_skips_points_outside():
    pts = {"notch_left": (960.0, 540.0), "notch_right": (2500.0, 100.0)}  # second is outside 1920x1080
    pred = kl.prediction(pts, size=(1920, 1080))
    assert all(r["type"] == "keypointlabels" for r in pred["result"])
    [kp] = pred["result"]
    assert kp["value"]["x"] == pytest.approx(50.0) and kp["value"]["y"] == pytest.approx(50.0)
    assert kp["value"]["keypointlabels"] == [kl.LABEL_TEXT["notch_left"]]


def test_no_prelabel_when_the_rail_fit_fails(monkeypatch):
    monkeypatch.setattr(kl, "_rail_fit", lambda frame: None)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert kl.prelabel_points(frame) == {}
