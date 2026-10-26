"""Three-state local annotations; never an automatic identity/round decision."""

import re
from datetime import datetime

LABELS_SCHEMA = "bxjudge.reid-review-labels.v1"
MANIFEST_SCHEMA = "bxjudge.reid-review-manifest.v1"
FIELDS = {
    "relation": {"unreviewed", "same", "different", "uncertain"},
    "side_before": {"unreviewed", "X", "B", "unknown"},
    "side_after": {"unreviewed", "X", "B", "unknown"},
    "visibility_before": {
        "unreviewed",
        "visible",
        "partially_occluded",
        "unusable",
        "unknown",
    },
    "visibility_after": {
        "unreviewed",
        "visible",
        "partially_occluded",
        "unusable",
        "unknown",
    },
    "collision": {"unreviewed", "yes", "no", "unknown"},
    "scope": {"unreviewed", "same_round", "different_round", "unknown"},
}


def validate_submission(payload, manifest, manifest_sha):
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "manifest_sha256",
        "submitted_at",
        "supersedes",
        "cases",
    }:
        raise ValueError("核對資料欄位不完整")
    if payload["schema"] != LABELS_SCHEMA or payload["manifest_sha256"] != manifest_sha:
        raise ValueError("核對頁版本不符；草稿保留，請先匯出再檢查頁面版本")
    stamp = payload["submitted_at"]
    if not isinstance(stamp, str) or not stamp:
        raise ValueError("時間格式錯誤")
    try:
        parsed = datetime.fromisoformat(stamp)
        if parsed.tzinfo is None:
            raise ValueError("missing timezone")
    except ValueError as exc:
        raise ValueError("時間必須包含時區") from exc
    parent = payload["supersedes"]
    if parent is not None and (
        not isinstance(parent, str) or re.fullmatch(r"[0-9a-f]{64}", parent) is None
    ):
        raise ValueError("修訂來源格式錯誤")
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != len(manifest["cases"]):
        raise ValueError("片段清單不符")
    stats = {"reviewed_cases": 0, "same": 0, "different": 0, "uncertain": 0}
    for case, expected in zip(cases, manifest["cases"], strict=True):
        if not isinstance(case, dict) or set(case) != {
            "case_id",
            "reviewed",
            "note",
            *FIELDS,
        }:
            raise ValueError("片段欄位不符")
        if case["case_id"] != expected["case_id"] or type(case["reviewed"]) is not bool:
            raise ValueError("片段識別或確認欄位不符")
        if not isinstance(case["note"], str) or len(case["note"]) > 4000:
            raise ValueError("註記格式錯誤或過長")
        for field, allowed in FIELDS.items():
            if not isinstance(case[field], str) or case[field] not in allowed:
                raise ValueError("核對選項無效")
        if case["reviewed"]:
            if case["relation"] == "unreviewed" or case["scope"] == "unreviewed":
                raise ValueError(
                    "確認本組前，請選同物件關係與同回合狀態；看不清可選未知"
                )
            stats["reviewed_cases"] += 1
            stats[case["relation"]] += 1
    if not stats["reviewed_cases"]:
        raise ValueError("尚未確認任何一組；草稿不會被刪除")
    return stats
