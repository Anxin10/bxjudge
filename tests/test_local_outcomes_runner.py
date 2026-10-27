import hashlib
import io
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import judge_local_outcomes as runner


@pytest.mark.parametrize("truncated", [False, True])
def test_local_outcome_run_preserves_core_and_saves_invalid_model_replies(
    tmp_path, monkeypatch, truncated
):
    core = tmp_path / "core"
    core.mkdir()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"synthetic source, never decoded by a real model")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    physical = {
        "segment_status": "paired_observations",
        "first_frame": 1,
        "last_frame": 8,
        "pair_confirmed_at_frame": 2,
        "members": {"10": [1], "11": [2]},
    }
    geometry = {
        "source": str(source),
        "source_sha256": source_sha,
        "samples": [{"frame": f} for f in range(11)],
        "phase": {"releases": {"1": 3, "2": 3}},
        "two_bey_tracking": {"rounds": {"1": physical}},
    }
    (core / "geometry").mkdir()
    raw = json.dumps(geometry).encode()
    (core / "geometry/g.json").write_bytes(raw)
    rows = [
        {
            "video": "clip.mp4",
            "round": 1,
            "winner": None,
            "reviewed": True,
            "geometry_evidence": {
                "path": "geometry/g.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        }
    ]
    (core / "rounds.json").write_text(json.dumps(rows))
    (core / "videos.json").write_text(
        json.dumps({"clip.mp4": {"source_video": str(source)}})
    )
    before = {str(p): p.read_bytes() for p in core.rglob("*") if p.is_file()}
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["judge_local_outcomes", "--results", str(core), "--out", str(out)]
    )
    monkeypatch.setattr(runner, "preflight", lambda config: None)
    monkeypatch.setattr(
        runner,
        "_opener",
        lambda: SimpleNamespace(
            open=lambda *a, **k: io.BytesIO(
                json.dumps({"models": [{"name": "qwen3.8:27b"}]}).encode()
            )
        ),
    )
    monkeypatch.setattr(runner, "probe", lambda _: SimpleNamespace(fps=30))
    monkeypatch.setattr(
        runner,
        "iter_frames",
        lambda *a, **k: iter((f, np.full((30, 40, 3), f, np.uint8)) for f in range(11)),
    )
    requests = []

    def request(images, frames, source_last, config):
        requests.append(frames)
        assert source_last == 10 and len(images) == len(frames)
        return {
            "done": True,
            "done_reason": "length" if truncated else "stop",
            "message": {
                "content": json.dumps(
                    {
                        "decision": "score",
                        "winner": "B",
                        "finish": "burst",
                        "event_interval": [1, 8],
                        "launch_interval": [1, 3],
                        "identity_evidence_frames": [1],
                        "launch_validity": "unknown",
                        "reason": "合成回答，只驗證儲存與證據界線。",
                        "uncertainties": ["第一落點未知"],
                    }
                )
            },
        }

    monkeypatch.setattr(runner, "request_judgment", request)
    runner.main()
    assert all(p.read_bytes() == before[str(p)] for p in core.rglob("*") if p.is_file())
    result = json.loads((out / "clip-r1/result.json").read_bytes())
    assert result["needs_review"] and not result["user_confirmed"]
    assert not result["core_or_human_results_modified"]
    assert (out / "clip-r1/response-1.json").is_file()
    assert result["proposed_points"] == (0 if truncated else 2)
    assert len(requests) == (1 if truncated else 2)
    if truncated:
        assert result["judgment"] is None and "VlmReplyError" in result["error"]
    else:
        # Refinement must preserve all initial context, even when the model's
        # inspection window points at the wrong portion of the bout.
        assert requests[0] <= requests[1]
    pixels = cv2.imread(str(out / "clip-r1/frames/f8.png"))
    assert np.all(pixels[32:] == 8)
    completion = json.loads((out / "completion.json").read_bytes())
    assert (
        completion["input_results_unchanged"] and completion["automatic_accepted"] == 0
    )
