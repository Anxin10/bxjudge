import json

import pytest

from bxjudge import vlm_events
from bxjudge.types import Calibration, Detection, Track
from bxjudge.vlm import VlmReplyError


def polygon(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


@pytest.fixture
def calibration():
    return Calibration(
        "test",
        (200, 200),
        polygon(0, 0, 180, 180),
        polygon(0, 110, 100, 180),
        polygon(0, 0, 100, 100),
        polygon(100, 0, 140, 100),
        polygon(140, 0, 180, 100),
        polygon(100, 100, 180, 140),
        [],
        [],
        "reference.png",
    )


def track(actor, frames):
    return Track(actor, "bey", [Detection(f, "bey", box, 0.9) for f, box in frames])


def facts(**changes):
    return {
        "two_free_complete_beys_visible": True,
        "battle_frames": [1],
        "events": [
            {
                "actor_id": 110,
                "kind": "entered_over_left",
                "frame_interval": [1, 2],
                "observation": "同一顆進入口袋。",
            }
        ],
        "unknowns": [],
    } | changes


def response(data):
    return {
        "done": True,
        "done_reason": "stop",
        "message": {"content": json.dumps(data)},
    }


def test_hallucinated_entry_with_absent_actor_never_creates_outcome(calibration):
    # Real negative-control pattern: actor absent at first endpoint, on floor later.
    tracks = [track(110, [(2, (10, 10, 20, 20))]), track(111, [(2, (50, 50, 60, 60))])]
    result = vlm_events.corroborate_events(
        facts(), tracks, tracks, calibration, {110: "B", 111: "X"}
    )
    assert result["events"][0]["status"] == "rejected"
    assert "actor_not_observed" in result["events"][0]["reasons"]
    assert result["events"][0]["conditional_outcome"] is None


def test_actual_center_entry_remains_conditional_and_uses_local_side(calibration):
    tracks = [
        track(110, [(1, (10, 10, 20, 20)), (2, (110, 10, 120, 20))]),
        track(111, [(1, (50, 50, 60, 60)), (2, (50, 50, 60, 60))]),
    ]
    result = vlm_events.corroborate_events(
        facts(), tracks, tracks, calibration, {110: "B", 111: None}
    )
    event = result["events"][0]
    assert event["status"] == "center_transition_supported"
    assert event["conditional_outcome"] == {
        "decision": "score",
        "winner": "X",
        "finish": "over",
    }
    assert result["side_mapping"] == {110: "B", 111: "X"}
    assert result["complement_frames"] == [1]
    assert event["accepted_for_final_score"] is False
    assert "entry_persistence_and_no_return" in event["conditions"]


def test_missing_geometry_does_not_turn_missing_reference_box_into_exit(calibration):
    raw = [track(110, [(1, (10, 10, 20, 20)), (2, (110, 10, 120, 20))])]
    normalized = [track(110, [(1, (10, 10, 20, 20))])]
    result = vlm_events.corroborate_events(
        facts(battle_frames=[]), raw, normalized, calibration, {110: "B"}
    )
    assert result["events"][0]["status"] == "rejected"
    assert "geometry_unavailable" in result["events"][0]["reasons"]
    assert result["events"][0]["conditional_outcome"] is None


@pytest.mark.parametrize("box", [(10, 10, 20, 20), (141, 10, 150, 20)])
def test_floor_or_other_pocket_endpoint_cannot_confirm_named_entry(calibration, box):
    tracks = [track(110, [(1, (10, 10, 20, 20)), (2, box)])]
    result = vlm_events.corroborate_events(
        facts(), tracks, tracks, calibration, {110: "B"}
    )
    assert result["events"][0]["status"] == "rejected"


@pytest.mark.parametrize("second", [None, (10, 10, 20, 20), (110, 10, 120, 20)])
def test_two_bey_complement_needs_same_frame_distinct_floor_observations(
    calibration, second
):
    tracks = [
        track(110, [(1, (10, 10, 20, 20)), (2, (110, 10, 120, 20))]),
        track(111, [(1, second)] if second is not None else [(2, (50, 50, 60, 60))]),
    ]
    result = vlm_events.corroborate_events(
        facts(), tracks, tracks, calibration, {110: "B", 111: None}
    )
    assert result["side_mapping"][111] is None
    assert result["complement_frames"] == []


def test_projected_outside_is_not_confirmed_physical_exit(calibration):
    tracks = [track(110, [(1, (10, 10, 20, 20)), (2, (185, 185, 195, 195))])]
    data = facts(events=[facts()["events"][0] | {"kind": "exited_arena_non_scoring"}])
    event = vlm_events.corroborate_events(
        data, tracks, tracks, calibration, {110: "B"}
    )["events"][0]
    assert event["status"] == "projected_exit_requires_review"
    assert event["conditional_outcome"] == {
        "decision": "redo",
        "winner": None,
        "finish": None,
    }
    assert "physical_exit_not_just_airborne_projection" in event["conditions"]
    assert event["accepted_for_final_score"] is False
    assert "non_scoring_exit@B" not in json.dumps(event)


@pytest.mark.parametrize(
    "change",
    [
        {
            "events": [
                {
                    "actor_id": 999,
                    "kind": "entered_over_left",
                    "frame_interval": [1, 2],
                    "observation": "x",
                }
            ]
        },
        {"battle_frames": [3]},
        {
            "events": [
                {
                    "actor_id": 110,
                    "kind": "entered_over_left",
                    "frame_interval": [2, 2],
                    "observation": "x",
                }
            ]
        },
        {"winner": "X"},
        {"two_free_complete_beys_visible": 1},
        {"unknowns": [None]},
    ],
)
def test_bad_actor_frame_or_verdict_fields_are_rejected(change):
    with pytest.raises(VlmReplyError):
        vlm_events.parse_events(response(facts(**change)), {1, 2}, {110, 111})


def test_source_references_are_validated_but_battle_frame_order_is_not_semantic():
    data = facts(battle_frames=[2, 1])
    assert vlm_events.parse_events(response(data), {1, 2}, {110, 111}) == data


def test_truncated_thinking_never_becomes_a_visual_event():
    with pytest.raises(VlmReplyError):
        vlm_events.parse_events(
            response(facts()) | {"done_reason": "length"}, {1, 2}, {110, 111}
        )


def test_conditional_sequence_respects_bout_wide_replay_and_later_finish(calibration):
    tracks = [
        track(110, [(1, (10, 10, 20, 20)), (2, (185, 185, 195, 195))]),
        track(111, [(3, (50, 50, 60, 60)), (4, (110, 10, 120, 20))]),
    ]
    exit_event = {
        "actor_id": 110,
        "kind": "exited_arena_non_scoring",
        "frame_interval": [1, 2],
        "observation": "非得分區離盤候選。",
    }
    entry_event = {
        "actor_id": 111,
        "kind": "entered_over_left",
        "frame_interval": [3, 4],
        "observation": "後來進袋。",
    }
    result = vlm_events.corroborate_events(
        facts(events=[entry_event, exit_event]),
        tracks,
        tracks,
        calibration,
        {110: "B", 111: "X"},
    )
    assert (
        result["conditional_sequence"]["if_events_are_confirmed"]["decision"] == "redo"
    )
    assert result["conditional_sequence"]["accepted_for_final_score"] is False
    # Reverse the physical timeline: a later exit cannot undo an earlier score.
    tracks = [
        track(110, [(3, (10, 10, 20, 20)), (4, (185, 185, 195, 195))]),
        track(111, [(1, (50, 50, 60, 60)), (2, (110, 10, 120, 20))]),
    ]
    result = vlm_events.corroborate_events(
        facts(
            events=[
                exit_event | {"frame_interval": [3, 4]},
                entry_event | {"frame_interval": [1, 2]},
            ]
        ),
        tracks,
        tracks,
        calibration,
        {110: "B", 111: "X"},
    )
    assert (
        result["conditional_sequence"]["if_events_are_confirmed"]["decision"] == "score"
    )


def test_overlapping_event_intervals_have_no_conditional_round_winner():
    def event(a, b):
        return {
            "status": "center_transition_supported",
            "model_event": {"frame_interval": [a, b]},
            "conditional_outcome": {
                "decision": "score",
                "winner": "X",
                "finish": "over",
            },
            "conditions": [],
        }

    assert vlm_events.conditional_sequence([event(1, 3), event(2, 4)]) is None
