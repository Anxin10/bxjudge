"""Evidence-linked local model outcome candidates, independent of core predictions.

Replies remain unverified model proposals. This module neither assigns human truth
nor edits core results, and never substitutes another model after a failed request.
"""

import json

from .vlm import (
    VlmConfig,
    VlmReplyError,
    _encode,
    _post,
    validate_endpoint,
    validate_model,
)

PROMPT = """你是本機戰鬥陀螺影片分析模型。畫面由原片抽取，f是原始來源影格編號，圖片按f排序；抽樣間隔不保證連續，沒有標出陀螺框或提供人工答案。
判斷此段是否真正對戰；若有可見決定性事件，提出勝方、終結與證據範圍。
X是本回合一開始由左方發射器釋放的那顆，B是右方。只依本回合的可見發射上下文追蹤；不能按最後所在左右、跨影片顏色或猜測指定X/B。側別不明時decision=unknown，在reason描述外觀勝方，winner與finish必須為null。
兩顆已釋放在盤內活動就可能是對戰，不需要剛好拍到碰撞。後來有人取走或收拾不會取消先前已發生的對戰。只有整段都無可見對戰、僅準備或收拾才是not_bout；未看見發射不能據此宣稱發射違規。
spin需自轉停止的時序證據，不能只用位移小或單張靜止圖；burst需可見解體，發射器與發射頭不是陀螺零件；over需進側邊得分口，xtreme需進極限口。拆件例外或口袋位置看不清時列不確定。按專案政策，先從非得分區離場、後來才入得分口應redo，不能只看最後位置。
launch_interval列出從掛載到兩顆已釋放的兩個實際f端點：前一格仍掛載、後一格已分離，不可用兩格都掛載的區間。這是最需要放大檢查的區間，不代表已確認物理分離；未涵蓋就空陣列。只能以圖片上實際提供的兩個不同f編號填event_interval，表示最早可見決定性變化區間，不假稱精確發生時刻。identity_evidence_frames只能列實際支持本回合X/B原始歸屬的圖片f編號。沒有事件就空陣列，不得造出中間未見影格。
score只是待覆核終結候選，launch_validity另列confirmed/invalid/unknown；第一落點沒看清就unknown。有明確更早發射違規則redo。只有原片最後一格也在提供圖片內且仍在打，才可incomplete；若只是候選片段或取樣結束，不能假稱原片結束。
redo、incomplete、not_bout、unknown的winner及finish都必須null，redo仍須提供事件區間。結論必須互相一致；抽樣不足就明列不確定，不硬猜。只回JSON，理由用繁體中文。
"""


def reply_schema(frames):
    frame = {"type": "integer", "enum": sorted(frames)}
    properties = {
        "event_interval": {"type": "array", "items": frame, "maxItems": 2},
        "launch_interval": {"type": "array", "items": frame, "maxItems": 2},
        "identity_evidence_frames": {"type": "array", "items": frame},
        "launch_validity": {
            "type": "string",
            "enum": ["confirmed", "invalid", "unknown"],
        },
        "reason": {"type": "string"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    }
    branches = []
    for score in (True, False):
        fields = properties | {
            "decision": {
                "type": "string",
                "enum": ["score"]
                if score
                else [
                    "redo",
                    "incomplete",
                    "not_bout",
                    "unknown",
                ],
            },
            "winner": {"type": "string", "enum": ["X", "B"]}
            if score
            else {"type": "null"},
            "finish": {"type": "string", "enum": ["spin", "burst", "over", "xtreme"]}
            if score
            else {"type": "null"},
        }
        if score:
            fields["event_interval"] = properties["event_interval"] | {"minItems": 2}
            fields["identity_evidence_frames"] = properties[
                "identity_evidence_frames"
            ] | {"minItems": 1}
            fields["launch_validity"] = {
                "type": "string",
                "enum": ["confirmed", "unknown"],
            }
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": fields,
                "required": list(fields),
            }
        )
    return {"oneOf": branches}


def parse_judgment(response, frames: set[int], source_last_frame: int) -> dict:
    if (
        not isinstance(response, dict)
        or response.get("done") is not True
        or response.get("done_reason") != "stop"
    ):
        raise VlmReplyError("local judgment did not finish normally")
    try:
        data = json.loads(response["message"]["content"])
    except (KeyError, TypeError, ValueError) as error:
        raise VlmReplyError("local judgment is not JSON content") from error
    fields = {
        "decision",
        "winner",
        "finish",
        "event_interval",
        "launch_interval",
        "identity_evidence_frames",
        "launch_validity",
        "reason",
        "uncertainties",
    }
    if not isinstance(data, dict) or data.keys() != fields:
        raise VlmReplyError("local judgment has missing or unexpected fields")
    if data["decision"] not in ("score", "redo", "incomplete", "not_bout", "unknown"):
        raise VlmReplyError("invalid judgment decision")
    if data["launch_validity"] not in ("confirmed", "invalid", "unknown"):
        raise VlmReplyError("invalid launch validity")
    for key in ("event_interval", "launch_interval", "identity_evidence_frames"):
        value = data[key]
        if not isinstance(value, list) or any(
            type(f) is not int or f not in frames for f in value
        ):
            raise VlmReplyError("judgment cites unavailable source frames")
        if value != sorted(set(value)):
            raise VlmReplyError("judgment source frames are duplicated or out of order")
    interval = data["event_interval"]
    if len(interval) not in (0, 2) or len(data["launch_interval"]) not in (0, 2):
        raise VlmReplyError("judgment event needs two distinct observed endpoints")
    if data["decision"] == "score":
        if (
            data["winner"] not in ("X", "B")
            or data["finish"] not in ("spin", "burst", "over", "xtreme")
            or not interval
            or not data["identity_evidence_frames"]
            or data["launch_validity"] == "invalid"
        ):
            raise VlmReplyError(
                "score candidate lacks supported identity, event or valid launch status"
            )
    elif data["winner"] is not None or data["finish"] is not None:
        raise VlmReplyError("non-score judgment cannot assert a winner or finish")
    if data["decision"] == "redo" and not interval:
        raise VlmReplyError("redo requires observed event evidence")
    if data["decision"] == "incomplete" and source_last_frame not in frames:
        raise VlmReplyError("incomplete requires observing actual source end")
    if not isinstance(data["reason"], str) or not data["reason"].strip():
        raise VlmReplyError("judgment has no reason")
    if not isinstance(data["uncertainties"], list) or any(
        not isinstance(s, str) or not s.strip() for s in data["uncertainties"]
    ):
        raise VlmReplyError("invalid judgment uncertainties")
    return data


def request_judgment(
    images, frames: set[int], source_last_frame: int, config: VlmConfig
):
    """Return verbatim response so callers can save it before validating a candidate."""
    host = validate_endpoint(config.host)
    model = validate_model(config.model)
    if not images or not frames or any(type(f) is not int or f < 0 for f in frames):
        raise ValueError("local judgment requires actual images and source indices")
    if type(source_last_frame) is not int or source_last_frame < max(frames):
        raise ValueError("source end must contain all supplied images")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT
                + f"\n實際提供f編號：{sorted(frames)}。原片最後f：{source_last_frame}。",
                "images": [_encode(image) for image in images],
            }
        ],
        "stream": False,
        "think": config.think,
        "format": reply_schema(frames),
        "keep_alive": config.keep_alive,
        "options": {"num_ctx": config.num_ctx, "num_predict": config.num_predict}
        | ({} if config.temperature is None else {"temperature": config.temperature}),
    }
    return _post(host, payload, timeout=config.timeout_s)
