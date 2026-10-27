import json

import numpy as np
import pytest

from bxjudge import vlm_judgment
from bxjudge.vlm import VlmReplyError, load_vlm_config


def answer(**changes):
    data = {
        "decision": "score",
        "winner": "B",
        "finish": "burst",
        "event_interval": [20, 30],
        "launch_interval": [10, 20],
        "identity_evidence_frames": [10],
        "launch_validity": "unknown",
        "reason": "看見解體，初始歸屬有畫面支持。",
        "uncertainties": ["第一落點未看清"],
    }
    return data | changes


def response(data):
    return {
        "done": True,
        "done_reason": "stop",
        "message": {"content": json.dumps(data)},
    }


def test_candidate_keeps_conditional_launch_and_never_becomes_human_truth():
    data = answer()
    parsed = vlm_judgment.parse_judgment(response(data), {10, 20, 30}, 40)
    assert parsed == data
    assert "points" not in parsed and "reviewed" not in parsed


@pytest.mark.parametrize(
    "changes",
    [
        {"event_interval": [20, 25]},  # Not a supplied source image.
        {"event_interval": [30, 20]},
        {"event_interval": [20, 20]},
        {"identity_evidence_frames": []},
        {"identity_evidence_frames": [11]},
        {"winner": None},
        {"launch_validity": "invalid"},
        {"decision": "unknown"},  # Cannot attach a certain winner to unknown.
        {
            "decision": "incomplete",
            "winner": None,
            "finish": None,
        },  # Source end unseen.
        {"reason": ""},
        {"points": 99},
    ],
)
def test_invalid_or_unsupported_judgments_are_not_accepted(changes):
    with pytest.raises(VlmReplyError):
        vlm_judgment.parse_judgment(response(answer(**changes)), {10, 20, 30}, 40)


def test_incomplete_requires_observing_actual_source_end():
    data = answer(
        decision="incomplete",
        winner=None,
        finish=None,
        event_interval=[],
        identity_evidence_frames=[],
    )
    assert vlm_judgment.parse_judgment(response(data), {10, 20, 40}, 40) == data


def test_redo_is_an_observed_event_not_unknown_mapped_to_redo():
    data = answer(decision="redo", winner=None, finish=None, launch_validity="invalid")
    assert vlm_judgment.parse_judgment(response(data), {10, 20, 30}, 40) == data
    with pytest.raises(VlmReplyError):
        vlm_judgment.parse_judgment(
            response(data | {"event_interval": []}), {10, 20, 30}, 40
        )


def test_request_uses_single_local_model_and_preserves_raw_response(monkeypatch):
    seen = []
    raw = response(answer())

    def post(host, payload, **kw):
        seen.append((host, payload, kw))
        return raw

    monkeypatch.setattr(vlm_judgment, "_post", post)
    actual = vlm_judgment.request_judgment(
        [np.zeros((100, 100, 3), np.uint8)], {10, 20, 30}, 40, load_vlm_config()
    )
    parsed = vlm_judgment.parse_judgment(actual, {10, 20, 30}, 40)
    assert actual is raw and parsed == answer() and len(seen) == 1
    host, payload, _ = seen[0]
    assert host == "http://127.0.0.1:11435"
    assert payload["model"] == "qwen3.8:27b"
    assert "40" in payload["messages"][0]["content"]
    assert all(
        "event_interval" in branch["properties"]
        for branch in payload["format"]["oneOf"]
    )


@pytest.mark.parametrize("change", [{"done": False}, {"done_reason": "length"}])
def test_truncated_replies_never_become_judgments(change):
    with pytest.raises(VlmReplyError):
        vlm_judgment.parse_judgment(response(answer()) | change, {10, 20, 30}, 40)
