"""Validation for the existing identity/launch review page (active schema)."""

PRIORITY = {("20261001_201622.mp4", f) for f in (1565, 1566, 1567)} | {
    ("20261001_201712.mp4", f) for f in (1645, 1658, 1704, 1713)
}
SIDES = {"unlabeled", "X", "B", "not_bey", "unknown"}
STATES = {"unlabeled", "mounted", "separated", "unknown"}


def validate(payload, manifest, manifest_sha):
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "manifest_sha256",
        "base_commit",
        "exported_at",
        "scope",
        "cases",
    }:
        raise ValueError("核對資料欄位不完整")
    if (
        payload["schema"] != "bxjudge.identity-launch-labels.v1"
        or payload["manifest_sha256"] != manifest_sha
        or payload["base_commit"] != manifest["base_commit"]
    ):
        raise ValueError("核對資料與目前影格清單不符，草稿未刪除")
    if not isinstance(payload["exported_at"], str) or not isinstance(
        payload["scope"], str
    ):
        raise ValueError("核對資料格式錯誤")  # noqa: TRY004 -- preserve the active validator contract
    if not isinstance(payload["cases"], list) or len(payload["cases"]) != len(
        manifest["cases"]
    ):
        raise ValueError("片段數量不符")
    content = False
    priority = confirmed = marked = 0
    for case, expected in zip(payload["cases"], manifest["cases"], strict=True):
        if not isinstance(case, dict) or set(case) != {
            "video",
            "source_sha256",
            "fps",
            "coordinate_size",
            "frames",
        }:
            raise ValueError("片段欄位不符")
        if any(
            case[k] != expected[k]
            for k in ("video", "source_sha256", "fps", "coordinate_size")
        ):
            raise ValueError("來源影片或座標資訊不符")
        if not isinstance(case["frames"], list) or len(case["frames"]) != len(
            expected["frames"]
        ):
            raise ValueError("影格數量不符")
        for frame, original in zip(case["frames"], expected["frames"], strict=True):
            if not isinstance(frame, dict) or set(frame) != {
                "frame",
                "image_sha256",
                "confirmed",
                "missed_or_bad_detection",
                "note",
                "objects",
            }:
                raise ValueError("影格欄位不符")
            if (
                type(frame["frame"]) is not int
                or frame["frame"] != original["frame"]
                or frame["image_sha256"] != original["image_sha256"]
            ):
                raise ValueError("來源影格識別不符")
            if (
                type(frame["confirmed"]) is not bool
                or type(frame["missed_or_bad_detection"]) is not bool
                or not isinstance(frame["note"], str)
            ):
                raise ValueError("人工確認欄位格式錯誤")
            confirmed += frame["confirmed"]
            priority += (
                frame["confirmed"] and (case["video"], frame["frame"]) in PRIORITY
            )
            content |= (
                frame["confirmed"]
                or frame["missed_or_bad_detection"]
                or bool(frame["note"])
            )
            if not isinstance(frame["objects"], list) or len(frame["objects"]) != len(
                original["objects"]
            ):
                raise ValueError("物件數量不符")
            for obj, raw in zip(frame["objects"], original["objects"], strict=True):
                if not isinstance(obj, dict) or set(obj) != {
                    "index",
                    "detector_class",
                    "box",
                    "score",
                    "narrow_track_ids",
                    "side",
                    "state",
                }:
                    raise ValueError("物件欄位不符")
                if (
                    type(obj["index"]) is not int
                    or type(obj["score"]) not in (int, float)
                    or not isinstance(obj["box"], list)
                    or len(obj["box"]) != 4
                    or any(type(x) not in (int, float) for x in obj["box"])
                    or not isinstance(obj["narrow_track_ids"], list)
                    or any(type(x) is not int for x in obj["narrow_track_ids"])
                ):
                    raise ValueError("原始偵測數值型別不符")
                if any(
                    obj[k] != raw[v]
                    for k, v in [
                        ("index", "index"),
                        ("detector_class", "class"),
                        ("box", "box"),
                        ("score", "score"),
                        ("narrow_track_ids", "track_ids"),
                    ]
                ):
                    raise ValueError("原始偵測證據被改動")
                if (
                    not isinstance(obj["side"], str)
                    or obj["side"] not in SIDES
                    or not isinstance(obj["state"], str)
                    or obj["state"] not in STATES
                ):
                    raise ValueError("標記選項無效")
                labelled = obj["side"] != "unlabeled" or obj["state"] != "unlabeled"
                marked += labelled
                content |= labelled
    if not content:
        raise ValueError("此瀏覽器尚無任何標記；請在完成標記的 Edge 頁面儲存")
    return {
        "confirmed_priority": priority,
        "confirmed_frames": confirmed,
        "marked_objects": marked,
    }
