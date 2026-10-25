import json
from pathlib import Path

import pytest
import yaml

from bxjudge.detector_config import load_policy


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None, "4", float("inf")])
def test_opencv_thread_policy_requires_positive_integer(tmp_path, value):
    config = yaml.safe_load(Path("config/detection.yaml").read_text())
    config["opencv_threads"] = value
    path = tmp_path / "detection.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="opencv_threads"):
        load_policy(path)


@pytest.mark.parametrize("requested,actual", [(4, 4), (8, 2)])
def test_thread_budget_applied_after_model_load_before_analysis(
    tmp_path, monkeypatch, requested, actual
):
    from bxjudge import pipeline
    from tests.helpers import make_video, rr

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    make_video(inputs / "clip.mp4", seconds=1)
    calib = tmp_path / "calib"
    calib.mkdir()
    out = tmp_path / "out"
    config = yaml.safe_load(Path("config/detection.yaml").read_text())
    config["opencv_threads"] = requested
    config_path = tmp_path / "detection.yaml"
    config_path.write_text(yaml.safe_dump(config))
    events = []
    threads = {"value": 1}

    def set_threads(value):
        threads["value"] = actual
        events.append(("threads", value))

    def load(*args, **kwargs):
        events.append(("model", None))
        threads["value"] = 1  # a framework import may reset process-wide OpenCV state
        return lambda *args: []

    def judge(info, *args, **kwargs):
        events.append(("analysis", threads["value"]))
        result = rr("spin", "X", video=info.path.name)
        result.detector_provenance = kwargs["provenance"]
        return [result]

    monkeypatch.setattr(
        pipeline,
        "_load_selection",
        lambda *args: {
            "kind": "rfdetr",
            "weights": "unused.pth",
            "dataset": {"provenance": {"synthetic": False}},
        },
    )
    monkeypatch.setattr(pipeline, "load_detector", load)
    monkeypatch.setattr(pipeline.cv2, "setNumThreads", set_threads)
    monkeypatch.setattr(pipeline.cv2, "getNumThreads", lambda: threads["value"])
    monkeypatch.setattr(pipeline, "judge_video", judge)
    pipeline.run_batch(inputs, calib, out, detection_config=config_path)
    assert events == [("model", None), ("threads", requested), ("analysis", actual)]
    row = json.loads((out / "rounds.json").read_text())[0]
    assert row["detector_provenance"]["runtime"]["opencv_threads"] == {
        "requested": requested,
        "actual": actual,
    }
