import io
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bxjudge.types import Calibration
from scripts import judge_actor_events as runner


def test_window_selection_preserves_early_non_scoring_and_late_scoring_locators():
    physical = {"first_frame": 1, "last_frame": 1000, "pair_confirmed_at_frame": 25}
    locators = [
        {"frame_interval": [f, f + 1], "to_zone": zone}
        for f, zone in [(50, "lid"), (200, "over_left"), (900, "xtreme")]
    ]
    wanted, selected = runner.wanted_frames(
        physical,
        locators,
        30,
        SimpleNamespace(event_window_s=1),
        {"overview_frames": 5, "window_frames": 5, "max_windows": 2},
    )
    assert [e["to_zone"] for e in selected] == ["lid", "xtreme"]
    assert {1, 25, 50, 51, 900, 901, 1000} <= wanted
    assert min(wanted) >= 1 and max(wanted) <= 1000


@pytest.mark.parametrize("truncated", [False, True])
def test_event_run_keeps_original_rows_and_invalid_response_without_fallback(
    tmp_path, monkeypatch, truncated
):
    core = tmp_path / "core"
    core.mkdir()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"synthetic source; no real model/video used")
    physical = {
        "segment_status": "paired_observations",
        "first_frame": 1,
        "last_frame": 8,
        "pair_confirmed_at_frame": 1,
        "members": {"110": [1], "111": [2]},
        "observations": [
            {"entity_id": i, "frame": f, "box": box, "score": 0.9}
            for i, f, box in [
                (1, 1, [10, 10, 20, 20]),
                (1, 2, [110, 10, 120, 20]),
                (2, 1, [50, 50, 60, 60]),
                (2, 2, [50, 50, 60, 60]),
            ]
        ],
    }
    rows = [
        {
            "video": "clip.mp4",
            "round": 1,
            "reviewed": True,
            "winner": None,
            "calibration_id": "frozen",
            "geometry_evidence": {"path": "geometry/g.json"},
        }
    ]
    (core / "rounds.json").write_text(json.dumps(rows))
    (core / "videos.json").write_text(
        json.dumps({"clip.mp4": {"source_video": str(source)}})
    )
    before = {p: p.read_bytes() for p in core.iterdir()}
    square = lambda a, b, c, d: [(a, b), (c, b), (c, d), (a, d)]
    calibration = Calibration(
        "test",
        (200, 200),
        square(0, 0, 180, 180),
        square(0, 100, 180, 180),
        square(0, 0, 100, 100),
        square(100, 0, 140, 100),
        square(140, 0, 180, 100),
        square(100, 100, 180, 140),
        [],
        [],
        "frozen.png",
    )
    context = {
        "source": source,
        "source_sha256": runner.sha(source),
        "fps": 30,
        "calibration": calibration,
        "record": {"two_bey_tracking": {"rounds": {"1": physical}}},
        "samples": {
            f: SimpleNamespace(alignment=SimpleNamespace(homography=np.eye(3)))
            for f in range(1, 9)
        },
        "eligible": {
            (d["frame"], tuple(d["box"]), d["score"]) for d in physical["observations"]
        },
        "anchors": SimpleNamespace(sides={1: "B"}, record=lambda: {"sides": {1: "B"}}),
    }
    monkeypatch.setattr(runner, "reconstruct", lambda *a: context)
    monkeypatch.setattr(runner, "preflight", lambda c: None)
    monkeypatch.setattr(
        runner,
        "_opener",
        lambda: SimpleNamespace(
            open=lambda *a, **k: io.BytesIO(
                json.dumps({"models": [{"name": "qwen3.8:27b"}]}).encode()
            )
        ),
    )

    def extract(source, wanted, folder, edge, video):
        folder.mkdir()
        for f in wanted:
            assert cv2.imwrite(
                str(folder / f"f{f}.png"), np.zeros((100, 100, 3), np.uint8)
            )

    monkeypatch.setattr(runner, "extract", extract)
    monkeypatch.setattr(
        runner,
        "annotate",
        lambda folder, frames, *a: [
            cv2.imread(str(folder / f"f{f}.png")) for f in sorted(frames)
        ],
    )
    requests = []

    def request(images, frames, actors, config):
        requests.append(config.model)
        assert len(images) == len(frames) and actors == {110, 111}
        return {
            "done": True,
            "done_reason": "length" if truncated else "stop",
            "message": {
                "content": json.dumps(
                    {
                        "two_free_complete_beys_visible": True,
                        "battle_frames": [1],
                        "unknowns": [],
                        "events": [
                            {
                                "actor_id": 110,
                                "kind": "entered_over_left",
                                "frame_interval": [1, 2],
                                "observation": "合成進袋。",
                            }
                        ],
                    }
                )
            },
        }

    monkeypatch.setattr(runner, "request_events", request)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["judge_actor_events", "--results", str(core), "--out", str(out)]
    )
    runner.main()
    assert all(p.read_bytes() == data for p, data in before.items())
    assert requests == ["qwen3.8:27b"]
    assert (out / "clip-r1/response.json").is_file()
    result = json.loads((out / "clip-r1/result.json").read_bytes())
    assert result["needs_review"] and not result["user_confirmed"]
    if truncated:
        assert result["corroboration"] is None and "VlmReplyError" in result["error"]
    else:
        assert result["corroboration"]["events"][0]["conditional_points"] == 2
        assert result["corroboration"]["automatic_accepted"] == 0
    completion = json.loads((out / "completion.json").read_bytes())
    assert completion["source_runtime_and_original_results_unchanged"]
