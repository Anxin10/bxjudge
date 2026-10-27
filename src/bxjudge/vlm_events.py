"""Local visual event proposals checked against immutable tracker/geometry evidence.

The model does not assign X/B or apply rules. Projected entries/exits remain
conditional evidence, never confirmed physical events or final referee results.
"""

import json

from .geometry import boxes_touch, zone_of
from .vlm import VlmReplyError, _encode, _post, validate_endpoint, validate_model

POCKETS = {
    "entered_over_left": "over_left",
    "entered_over_right": "over_right",
    "entered_xtreme": "xtreme",
}
KINDS = (
    *POCKETS,
    "exited_arena_non_scoring",
    "components_separated",
    "axial_spin_stopped",
    "picked_up",
)

PROMPT = """只記錄圖片中看得到的物件與事件，不判勝負、不算分、不指定X/B。
框上ID是本地追蹤器的候選物件身分；不能按現在所在左右或顏色改ID。框可能錯，發射頭也可能誤框為陀螺。ID無法對上實體就不要記事件，列unknowns。
two_free_complete_beys_visible只在可見兩顆完整、已離開發射器的陀螺時true，battle_frames列支持這件事的實際f；不需要拍到碰撞。綠線只在當格校準有效時畫出，沒有綠線就是幾何未知。
battle_area是戰鬥區；over_left/over_right與xtreme是得分口。只有可見同一ID從盤內進口，才記事件。exited_arena_non_scoring指真正從非得分處離開整個戰鬥盤，不是上蓋面短暫跳起、被手取走、離開候選框或掉偵；不能確認物理離場就列unknowns。發射頭不是解體零件，位移小不等於自轉停止。
事件frame_interval必須是兩個已提供且不同的f，按先後排序，僅界定可見變化區間。不要造中間影格、事件、勝方或分數。缺證據就空events。理由用繁體中文，只回JSON。
"""


def event_schema(frames, actors):
    frame = {"type": "integer", "enum": sorted(frames)}
    event = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "actor_id": {"type": "integer", "enum": sorted(actors)},
            "kind": {"type": "string", "enum": list(KINDS)},
            "frame_interval": {
                "type": "array",
                "items": frame,
                "minItems": 2,
                "maxItems": 2,
            },
            "observation": {"type": "string"},
        },
        "required": ["actor_id", "kind", "frame_interval", "observation"],
    }
    properties = {
        "two_free_complete_beys_visible": {"type": ["boolean", "null"]},
        "battle_frames": {"type": "array", "items": frame},
        "events": {"type": "array", "items": event},
        "unknowns": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def parse_events(response, frames, actors):
    if (
        not isinstance(response, dict)
        or response.get("done") is not True
        or response.get("done_reason") != "stop"
    ):
        raise VlmReplyError("visual event request did not finish normally")
    try:
        data = json.loads(response["message"]["content"])
    except (KeyError, TypeError, ValueError) as error:
        raise VlmReplyError("visual events are not JSON content") from error
    if not isinstance(data, dict) or data.keys() != {
        "two_free_complete_beys_visible",
        "battle_frames",
        "events",
        "unknowns",
    }:
        raise VlmReplyError("visual events have missing or unexpected fields")
    if (
        data["two_free_complete_beys_visible"] is not None
        and type(data["two_free_complete_beys_visible"]) is not bool
    ):
        raise VlmReplyError("invalid two-bey visual observation")

    def source_frames(value):
        return (
            isinstance(value, list)
            and all(type(f) is int and f in frames for f in value)
            and len(value) == len(set(value))
        )

    if not source_frames(data["battle_frames"]):
        raise VlmReplyError(
            "battle observation cites unavailable or duplicate source frames"
        )
    if not isinstance(data["unknowns"], list) or any(
        not isinstance(s, str) or not s.strip() for s in data["unknowns"]
    ):
        raise VlmReplyError("invalid visual event uncertainties")
    if not isinstance(data["events"], list):
        raise VlmReplyError("invalid visual event list")
    for event in data["events"]:
        if not isinstance(event, dict) or event.keys() != {
            "actor_id",
            "kind",
            "frame_interval",
            "observation",
        }:
            raise VlmReplyError("visual event has missing or unexpected fields")
        if (
            type(event["actor_id"]) is not int
            or event["actor_id"] not in actors
            or event["kind"] not in KINDS
        ):
            raise VlmReplyError("visual event cites an unknown actor or event kind")
        interval = event["frame_interval"]
        if (
            not source_frames(interval)
            or len(interval) != 2
            or interval[0] >= interval[1]
        ):
            raise VlmReplyError("visual event needs two ordered observed endpoints")
        if (
            not isinstance(event["observation"], str)
            or not event["observation"].strip()
        ):
            raise VlmReplyError("visual event lacks an observation")
    return data


def request_events(images, frames, actors, config):
    host, model = validate_endpoint(config.host), validate_model(config.model)
    if (
        not images
        or len(images) != len(frames)
        or not actors
        or any(type(f) is not int or f < 0 for f in frames)
        or any(type(actor) is not int for actor in actors)
    ):
        raise ValueError("visual events require matching actual frames and actor IDs")
    return _post(
        host,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT
                    + f"\n實際f：{sorted(frames)}；允許ID：{sorted(actors)}。",
                    "images": [_encode(image) for image in images],
                }
            ],
            "stream": False,
            "think": config.think,
            "format": event_schema(frames, actors),
            "keep_alive": config.keep_alive,
            "options": {"num_ctx": config.num_ctx, "num_predict": config.num_predict}
            | (
                {}
                if config.temperature is None
                else {"temperature": config.temperature}
            ),
        },
        timeout=config.timeout_s,
    )


def corroborate_events(facts, source_tracks, reference_tracks, calibration, sides):
    """Check actual endpoints, not model assertions, without filling missing geometry.

    Source tracks must contain only phase-eligible physical observations; reference
    tracks are the same measurements mapped using valid recorded geometry. The
    two-bey complement needs an observed same-frame disjoint pair on the floor.
    It cannot derive a second player from a duplicated box or a launcher head.
    """
    raw = {t.track_id: {d.frame: d for d in t.dets} for t in source_tracks}
    reference = {t.track_id: {d.frame: d for d in t.dets} for t in reference_tracks}
    if (
        len(raw) != len(source_tracks)
        or set(raw) != set(reference)
        or set(sides) != set(raw)
        or any(s not in ("X", "B", None) for s in sides.values())
    ):
        raise ValueError("inconsistent actor evidence or side mapping")
    mapping = dict(sides)
    complement_frames = []
    known = [side for side in mapping.values() if side is not None]
    if (
        facts["two_free_complete_beys_visible"] is True
        and len(raw) == 2
        and len(known) == 1
    ):
        a, b = sorted(raw)
        for f in facts["battle_frames"]:
            da, db = reference[a].get(f), reference[b].get(f)
            if (
                da is not None
                and db is not None
                and not boxes_touch(da.box, db.box)
                and zone_of(
                    ((da.box[0] + da.box[2]) / 2, (da.box[1] + da.box[3]) / 2),
                    calibration,
                )
                == "battle_area"
                and zone_of(
                    ((db.box[0] + db.box[2]) / 2, (db.box[1] + db.box[3]) / 2),
                    calibration,
                )
                == "battle_area"
            ):
                complement_frames.append(f)
        if complement_frames:
            mapping = {
                actor: side or ("X" if known[0] == "B" else "B")
                for actor, side in mapping.items()
            }
    conflicting_sides = len([s for s in mapping.values() if s is not None]) != len(
        {s for s in mapping.values() if s is not None}
    )
    results = []
    for event in facts["events"]:
        actor = event["actor_id"]
        states, reasons = [], []
        for f in event["frame_interval"]:
            detection = reference[actor].get(f)
            observed = f in raw[actor]
            zone = (
                None
                if detection is None
                else zone_of(
                    (
                        (detection.box[0] + detection.box[2]) / 2,
                        (detection.box[1] + detection.box[3]) / 2,
                    ),
                    calibration,
                )
            )
            states.append(
                {
                    "frame": f,
                    "actor_observed": observed,
                    "geometry_valid": detection is not None if observed else None,
                    "center_zone": zone,
                }
            )
            if not observed:
                reasons.append("actor_not_observed")
            elif detection is None:
                reasons.append("geometry_unavailable")
        status, conditional = "visual_review_required", None
        conditions = [
            "earlier_events_and_launch_validity",
            "identity_and_unassigned_observations",
        ]
        kind = event["kind"]
        if kind in POCKETS:
            if not reasons and (
                states[0]["center_zone"] != "battle_area"
                or states[1]["center_zone"] != POCKETS[kind]
            ):
                reasons.append("named_zone_transition_not_observed")
            if not reasons:
                status = "center_transition_supported"
                conditions += ["whole_body_entry", "entry_persistence_and_no_return"]
                side = mapping[actor]
                if side is not None and not conflicting_sides:
                    conditional = {
                        "decision": "score",
                        "winner": "X" if side == "B" else "B",
                        "finish": "xtreme" if kind == "entered_xtreme" else "over",
                    }
        elif kind == "exited_arena_non_scoring":
            if not reasons and (
                states[0]["center_zone"] not in ("battle_area", "lid", "stadium_other")
                or states[1]["center_zone"] != "outside"
            ):
                reasons.append("projected_exit_not_observed")
            if not reasons:
                status = "projected_exit_requires_review"
                conditions += [
                    "physical_exit_not_just_airborne_projection",
                    "non_scoring_path_and_event_order",
                ]
                conditional = {"decision": "redo", "winner": None, "finish": None}
        else:
            conditions.append("event_specific_visual_evidence")
        if reasons:
            status = "rejected"
        results.append(
            {
                "model_event": event,
                "actual_states": states,
                "status": status,
                "reasons": list(dict.fromkeys(reasons)),
                "conditions": conditions,
                "conditional_outcome": conditional,
                "accepted_for_final_score": False,
            }
        )
    return {
        "side_mapping": mapping,
        "complement_frames": sorted(complement_frames),
        "side_conflict": conflicting_sides,
        "events": results,
        "conditional_sequence": conditional_sequence(results),
        "needs_review": True,
        "user_confirmed": False,
        "automatic_accepted": 0,
    }


def conditional_sequence(events):
    """Order supported hypotheses; overlapping or earlier unverified events block.

    This is explicitly conditional on each event's outstanding physical checks.
    It never emits a confirmed non_scoring_exit reason for the rules engine.
    """
    candidates = sorted(
        (event for event in events if event["status"] != "rejected"),
        key=lambda event: event["model_event"]["frame_interval"][0],
    )
    if not candidates:
        return None
    first = candidates[0]
    interval = first["model_event"]["frame_interval"]
    if first["conditional_outcome"] is None or any(
        event["model_event"]["frame_interval"][0] <= interval[1]
        for event in candidates[1:]
    ):
        return None
    return {
        "if_events_are_confirmed": first["conditional_outcome"],
        "first_event": first["model_event"],
        "conditions": first["conditions"],
        "accepted_for_final_score": False,
    }
