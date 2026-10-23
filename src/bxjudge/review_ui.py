"""Read-only helpers for the review app: labels, filters, media and evidence frames.

Kept free of Streamlit so it can be unit-tested; all writes go through
:mod:`bxjudge.corrections`.
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from .calibration import scale_calibration
from .calibration_store import Snapshot, load_snapshot, vetoed_ids
from .config import Thresholds
from .corrections import Correction, current_value
from .manifest import probe
from .output import VIDEO_FAILURE_STATUSES
from .rules import is_not_a_round
from .types import Calibration
from .video import VideoDecodeError, iter_frames
from .vlm import overlay_zones, recorded_source

FILTERS = ("全部", "需覆核", "已覆核", "抽樣稽核")  # 抽樣稽核 = R62 random audit sample

_TOUCH = {"yes": "有碰觸", "no": "沒有碰觸", "unknown": "不確定"}
_LANDED = {True: "是", False: "否"}
_ZONE = {
    None: "（未知）",
    "battle_area": "下盤戰鬥區",
    "lid": "上盤／蓋子",
    "over_left": "擊飛區（左）",
    "over_right": "擊飛區（右）",
    "xtreme": "極限區",
    "stadium_other": "盤上其他",
    "outside": "盤外",
}
_SPIN = {"right": "右旋", "left": "左旋", "unknown": "不確定"}
_AIR = {"yes": "有空碰", "no": "沒有空碰", "maybe": "不確定"}
_FINISH = {
    None: "（無終結：重來）",
    "spin": "Spin 停轉（1 分）",
    "over": "Over 擊飛（2 分）",
    "burst": "Burst 爆裂（2 分）",
    "xtreme": "Xtreme 極限（3 分）",
}
_WINNER = {None: "（無）", "X": "X（左方）", "B": "B（右方）"}

# field -> (Traditional Chinese label, value -> label); order = form order
FIELDS: dict[str, tuple[str, dict]] = {
    "x_touch": ("X 方發射觸碰盤面", _TOUCH),
    "x_landed": ("X 方落入戰鬥區", _LANDED),
    "x_land_zone": ("X 方落地區域", _ZONE),
    "x_spin_dir": ("X 方旋轉方向", _SPIN),
    "b_touch": ("B 方發射觸碰盤面", _TOUCH),
    "b_landed": ("B 方落入戰鬥區", _LANDED),
    "b_land_zone": ("B 方落地區域", _ZONE),
    "b_spin_dir": ("B 方旋轉方向", _SPIN),
    "air_collision": ("空中碰撞（空判）", _AIR),
    "finish": ("結束方式", _FINISH),
    "winner": ("獲勝方", _WINNER),
    "match_start": ("此回合另起新場", _LANDED),
}


class ReviewMediaError(RuntimeError):
    """Evidence media cannot be shown; the message says why."""


def is_diagnostic(row: dict) -> bool:
    """recalibrate / no_round / decode_error / analysis_error placeholders (R53)."""
    return is_not_a_round(row.get("review_reasons") or [])


def status(row: dict) -> str:
    if is_diagnostic(row):
        return "分析診斷"
    if row.get("reviewed"):
        return "已覆核"
    return "待覆核" if row.get("needs_review") else "未標記"


def filter_rows(rows: list[dict], choice: str) -> list[dict]:
    if choice == "需覆核":
        return [r for r in rows if status(r) in ("待覆核", "分析診斷")]
    if choice == "已覆核":
        return [r for r in rows if status(r) == "已覆核"]
    if choice == "抽樣稽核":
        return [r for r in rows if r.get("audit_sample") is True]
    return list(rows)


def counts(rows: list[dict]) -> dict[str, int]:
    states = [status(r) for r in rows]
    return {
        "回合": len(rows) - states.count("分析診斷"),
        "待覆核": states.count("待覆核"),
        "已覆核": states.count("已覆核"),
        "分析診斷": states.count("分析診斷"),
    }


def is_synthetic(row: dict) -> bool:
    """Smoke-wiring output from a synthetic-data detector, not a trusted judgment."""
    if "synthetic_detector" in (row.get("review_reasons") or []):
        return True
    provenance = row.get("detector_provenance") or {}
    dataset = provenance.get("dataset") if isinstance(provenance, dict) else None
    details = dataset.get("provenance") if isinstance(dataset, dict) else None
    return isinstance(details, dict) and details.get("synthetic") is True


def source_video(videos: dict, video: str) -> Path | str:
    """Absolute ``videos.json`` ``source_video``, or why it is unavailable (no guessing)."""
    return recorded_source(videos.get(video))


def form_changes(row: dict, values: dict, at: str) -> list[Correction]:
    """Corrections for the form values that differ from the current effective values."""
    out = []
    for field, new in values.items():
        old = current_value(row, field)
        if not (type(old) is type(new) and old == new):
            out.append(Correction(row["video"], row["round"], field, old, new, at))
    return out


def repair_steps(row: dict, source: Path | str, results_dir: Path) -> list[str]:
    """Concrete next steps for a diagnostic placeholder row."""
    video = str(source) if isinstance(source, Path) else "<原始影片路徑>"
    folder = str(source.parent) if isinstance(source, Path) else "<影片資料夾>"
    rerun = (
        f"修好後重新分析（此影片不會被標為已處理，會自動重跑）："
        f"`bxjudge run {folder} --out {results_dir}`"
    )
    reasons = row.get("review_reasons") or []
    steps = []
    if "recalibrate" in reasons:
        steps += [
            "影片第一秒的畫面與 calib/ 內所有校準都不相符（鏡頭或場地位置改變）。",
            f"為這個鏡頭重新校準：`bxjudge calibrate {video} --name <新校準名稱> --out calib`",
        ]
    if "no_round" in reasons:
        steps += [
            "整支影片沒有切出任何回合（沒有偵測到兩顆陀螺發射）。",
            f"確認影片內容確實有對戰；若有，檢查偵測器與 config/thresholds.yaml 的分段門檻：`ffprobe {video}`",
        ]
    if "decode_error" in reasons:
        steps += [
            "影片無法解碼（檔案損毀、未同步完成或格式不支援）。",
            f"檢查檔案：`ffprobe {video}`；必要時以 rclone copy --checksum 重新同步。",
        ]
    if "analysis_error" in reasons:
        steps += [
            "分析過程發生例外錯誤。",
            "查看 `bxjudge run` 的紀錄中 `analysis failed for` 那一段的錯誤訊息並修正原因。",
        ]
    return [*steps, rerun]


def row_calibration(out_dir: Path, row: dict) -> Snapshot | None:
    """The immutable snapshot used for this row, including retained historical reviews.

    A missing or unreadable record is unavailable; never choose a replacement.
    """
    snapshot_id = row.get("calibration_id")
    if not snapshot_id:
        return None
    try:
        return load_snapshot(out_dir, snapshot_id)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def calibration_status(out_dir: Path, row: dict) -> str:
    """Calibration availability and whether a human review used an obsolete snapshot."""
    snapshot = row_calibration(out_dir, row)
    if snapshot is None:
        return "no_record"
    path = Path(out_dir) / "videos.json"
    videos = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    entry = videos.get(row["video"], {})
    if entry.get("analysis_status") in VIDEO_FAILURE_STATUSES:
        return "pending"
    current = (entry.get("calibration") or {}).get("id")
    vetoed = snapshot.id in vetoed_ids(out_dir)
    if row.get("reviewed") and (vetoed or snapshot.id != current):
        return "changed_since_review"
    return "vetoed" if vetoed else "ok"


def evidence_frame(
    source: Path, t: float, calibration: Calibration | None, th: Thresholds
) -> tuple[np.ndarray, float, int]:
    """RGB frame at playback second ``t`` with all 8 zones drawn (when calibrated).

    Returns ``(image, actual_t, source_frame_index)``; ``t`` past the end is clamped to
    the last frame.
    """
    try:
        info = probe(source)
        last = max(0, int(np.ceil(info.duration * info.fps - 1e-6)) - 1)
        index = min(max(0, round(t * info.fps)), last)
        start = index / info.fps
        frame = None
        with closing(
            iter_frames(
                source,
                start=start,
                end=start + 1 / info.fps,
                long_edge=th.infer_long_edge,
                hwaccel=th.decode_hwaccel,
            )
        ) as frames:
            for idx, image in frames:
                index, frame = idx, image
                break
    except (VideoDecodeError, ValueError, OSError, RuntimeError) as e:
        raise ReviewMediaError(str(e)) from e
    if frame is None:
        raise ReviewMediaError(f"{source}: no frame at {t:.3f} s")
    actual = index / info.fps
    if calibration is not None:
        scaled = scale_calibration(calibration, (frame.shape[1], frame.shape[0]))
        frame = overlay_zones(frame, scaled, actual)
    return cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB), actual, index
