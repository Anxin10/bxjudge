"""Local review UI (Streamlit). Started by ``bxjudge review RESULTS`` on 127.0.0.1.

Shows each round's evidence (video from ``t_launch - 1`` s, launch/end frames with the
calibration zones, trajectory image, review reasons, VLM note) and saves human
corrections through :mod:`bxjudge.corrections`. Diagnostic placeholder rows only get
repair steps (R53).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import streamlit as st

from bxjudge import review_ui as ui
from bxjudge.calibration_store import (
    Snapshot,
    unveto_calibration,
    veto_calibration,
    vetoed_ids,
)
from bxjudge.config import load_thresholds
from bxjudge.corrections import (
    CorrectionError,
    apply_corrections,
    confirm_reviewed,
    load_corrections,
    now,
)
from bxjudge.output import effective, matches_for, read_rounds, trajectory_png
from bxjudge.rules import finish_points, load_rules
from bxjudge.vlm import overlay_zones


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="review_app")
    p.add_argument("results")
    p.add_argument("--rules", default="config/rules.yaml")
    p.add_argument("--thresholds", default="config/thresholds.yaml")
    return p.parse_args(sys.argv[1:])


@st.cache_data(show_spinner="擷取畫面…", max_entries=256)
def _frame(source: str, mtime: float, t: float, snapshot_id: str | None, thresholds: str, _calibration):
    # The immutable snapshot id is part of the cache key, including after reanalysis.
    return ui.evidence_frame(Path(source), t, _calibration, load_thresholds(thresholds))


_NONE = "__none__"


def _row_key(row: dict) -> str:
    return f"{row['video']}::{row['round']}"


def _flash(message: str) -> None:
    st.session_state["flash"] = message
    st.rerun()


def _evidence(row: dict, source, out: Path, a: argparse.Namespace, snapshot: Snapshot | None) -> None:
    st.subheader("影片")
    if isinstance(source, str):
        st.info(f"無法播放影片：{source}")
    else:
        start = max(0.0, float(row["t_launch"]) - 1.0)
        st.video(str(source), start_time=start)
        st.caption(
            f"{source} · 從 {start:.2f} 秒開始（發射 {row['t_launch']:.2f} 秒前 1 秒）；"
            f"結束 {row['t_end']:.2f} 秒"
        )
    st.subheader("發射與結束瞬間")
    calibration = snapshot.calibration if snapshot else None
    if not isinstance(source, str):
        mtime = source.stat().st_mtime
        cols = st.columns(2)
        for col, name, t in zip(
            cols, ("發射瞬間", "結束瞬間"), (row["t_launch"], row["t_end"]), strict=True
        ):
            with col:
                try:
                    image, actual, index = _frame(
                        str(source), mtime, float(t), snapshot.id if snapshot else None, a.thresholds, calibration
                    )
                except ui.ReviewMediaError as e:
                    st.warning(f"{name}畫面無法擷取：{e}")
                    continue
                zones = f"校準 {calibration.name} 八區" if calibration else "未疊校準區域"
                st.image(
                    image, caption=f"{name} {actual:.3f} 秒（第 {index} 格）· {zones}",
                    width="stretch",
                )
    else:
        st.info("沒有原始影片，無法擷取截圖。")
    st.subheader("該回合軌跡圖")
    png = trajectory_png(out, row["video"], row["round"])
    if png.is_file():
        st.image(
            str(png), caption=f"{png.name}（X 藍、B 紅；缺失觀測不連線）", width="stretch"
        )
    else:
        st.info(f"此回合沒有軌跡圖（{png}）。")


def _calibration(row: dict, out: Path, videos: dict, rules) -> Snapshot | None:
    state = ui.calibration_status(out, row)
    snapshot = ui.row_calibration(out, row)
    if snapshot is None:
        st.warning("此結果沒有校準紀錄，請用新版重新執行 bxjudge run")
        return None
    if state == "changed_since_review":
        st.warning("覆核當時使用的校準已被否決／更換，請確認是否需要重新覆核")
    elif state == "vetoed":
        st.warning("校準被否決，需重新校準")
    current = videos.get(row["video"], {}).get("calibration") or {}
    score = current.get("score", snapshot.score) if current.get("id") == snapshot.id else snapshot.score
    st.caption(f"校準：{snapshot.kind}（{snapshot.calibration.name}）分數 {score:.3f}")
    st.caption(f"校準組：{snapshot.id}")
    with st.expander("校準組預覽"):
        image = cv2.imread(snapshot.calibration.ref_frame)
        if image is None:
            st.info("此組校準的參考圖無法讀取。")
        else:
            image = overlay_zones(image, snapshot.calibration, 0)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
    if snapshot.kind == "auto":
        vetoed = snapshot.id in vetoed_ids(out)
        label = "恢復採用此組校準" if vetoed else "不採用此組校準"
        key = f"unveto-{snapshot.id}" if vetoed else f"veto-{snapshot.id}"
        if st.button(label, key=key):
            action = unveto_calibration if vetoed else veto_calibration
            action(out, snapshot.id, rules=rules)
            _flash("已恢復採用此組校準。" if vetoed else "已否決此組校準。")
    return snapshot


def _judgment(row: dict, rows: list[dict], rules, vetoed: set[str]) -> None:
    eff = effective(row)
    flagged = "需覆核" if row.get("needs_review") else "不需覆核"
    human = "已完成" if row.get("reviewed") else "尚未"
    confidence = row.get("confidence")
    shown = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "—"
    st.markdown(f"**模型原始標記：{flagged}** · 模型信心（原始）：{shown}")
    st.markdown(f"**人工覆核：{human}**")
    st.markdown("**覆核原因（review_reasons）**")
    reasons = row.get("review_reasons") or []
    st.markdown("\n".join(f"- `{r}`" for r in reasons) if reasons else "（無）")
    st.markdown("**VLM 第二意見（僅供參考，不改判）**")
    st.info(row.get("vlm_note") or "（無）")
    points = finish_points(rules, eff["finish"]) if eff["finish"] else 0
    st.markdown(
        f"**目前有效判定：** {ui.FIELDS['finish'][1].get(eff['finish'], eff['finish'])}，"
        f"獲勝方 {ui.FIELDS['winner'][1].get(eff['winner'], eff['winner'])}，{points} 分"
    )
    for number, match in enumerate(matches_for(rows, rules, vetoed=vetoed), 1):
        for r, outcome in match.rounds:
            if (r.video, r.round) == (row["video"], row["round"]):
                result = (
                    f"{outcome.scorer} 得 {outcome.points} 分"
                    if outcome.scorer
                    else "重來"
                )
                why = f"；{'、'.join(outcome.reasons)}" if outcome.reasons else ""
                st.markdown(
                    f"**第 {number} 場** 目前比分 X {match.score['X']} : "
                    f"{match.score['B']} B（本回合：{result}{why}）"
                )
    table = [
        {
            "欄位": label,
            "模型原判定": options.get(ui.current_value({**row, "reviewer_fix": None}, field)),
            "目前有效值": options.get(ui.current_value(row, field)),
        }
        for field, (label, options) in ui.FIELDS.items()
    ]
    st.dataframe(table, hide_index=True, width="stretch")


def _form(row: dict, out: Path, rules) -> None:
    eff = effective(row)
    version = json.dumps([row.get("reviewer_fix"), row.get("reviewed")], sort_keys=True)
    prefix = f"{_row_key(row)}::{version}"

    def select(field):
        # A None option is a real choice ("unknown zone", "no finish"); a sentinel keeps
        # it distinct from Streamlit's "nothing selected".
        label, options = ui.FIELDS[field]
        choices = [_NONE if v is None else v for v in options]
        value = _NONE if eff.get(field) is None else eff.get(field)
        chosen = st.selectbox(
            label,
            choices,
            index=choices.index(value) if value in choices else None,
            format_func=lambda v, o=options: o[None if v == _NONE else v],
            key=f"{prefix}::{field}",
        )
        return None if chosen == _NONE else chosen

    with st.form(f"form::{prefix}"):
        st.markdown("**修正判定**（只會記錄有變更的欄位；原判定永遠保留）")
        values = {}
        left, right = st.columns(2)
        for col, side in ((left, "x"), (right, "b")):
            with col:
                for suffix in ("touch", "landed", "land_zone", "spin_dir"):
                    values[f"{side}_{suffix}"] = select(f"{side}_{suffix}")
        for field in ("air_collision", "finish", "winner"):
            values[field] = select(field)
        values["match_start"] = st.checkbox(
            ui.FIELDS["match_start"][0],
            value=bool(eff.get("match_start", False)),
            help="勾選：上一場尚未打完也在此回合結束，從本回合開新的一場。取消勾選恢復自動分組。",
            key=f"{prefix}::match_start",
        )
        submitted = st.form_submit_button("儲存修正", type="primary")
    if submitted:
        changes = ui.form_changes(row, values, now())
        if not changes:
            st.info("沒有任何變更；若原判定正確，請按「確認無誤」。")
        else:
            try:
                applied = apply_corrections(out, changes, rules=rules)
            except CorrectionError as e:
                st.error(f"未儲存：{e}")
            else:
                _flash(f"已儲存 {len(applied)} 項修正，比分已重新計算。")
    if st.button(
        "確認無誤",
        key=f"confirm::{prefix}",
        disabled=bool(row.get("reviewed")),
        help="不改任何欄位，只標記為已人工覆核。" if not row.get("reviewed") else "已覆核。",
    ):
        try:
            confirm_reviewed(out, row["video"], row["round"], rules=rules)
        except CorrectionError as e:
            st.error(f"未儲存：{e}")
        else:
            _flash("已確認無誤（未變更任何欄位）。")


def main() -> None:
    a = _args()
    st.set_page_config(page_title="BXJUDGE 覆核", layout="wide")
    out = Path(a.results).expanduser()
    try:
        rules = load_rules(a.rules)
        load_thresholds(a.thresholds)
        rows = read_rounds(out)
        videos_path = out / "videos.json"
        videos = (
            json.loads(videos_path.read_text(encoding="utf-8"))
            if videos_path.exists()
            else {}
        )
    except (OSError, ValueError) as e:
        st.error(f"無法載入 {out}：{e}")
        return
    st.title("BXJUDGE 覆核")
    st.caption(f"結果資料夾：{out}")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    for col, (label, value) in zip(st.columns(4), ui.counts(rows).items(), strict=True):
        col.metric(label, value)

    choice = st.sidebar.radio("篩選", ui.FILTERS, key="filter")
    shown = ui.filter_rows(rows, choice)
    if not shown:
        st.info(f"沒有「{choice}」的回合。")
        return
    labels = {
        _row_key(r): f"{r['video']} · 第 {r['round']} 回合 · {ui.status(r)}"
        + (" · 抽樣稽核" if r.get("audit_sample") is True else "")
        for r in shown
    }
    # The labels carry each row's status; a fresh widget per label set keeps them
    # current after a save, while the selection itself lives in session state.
    keys = list(labels)
    selected = st.session_state.get("selected")
    signature = hashlib.sha1("|".join(labels.values()).encode()).hexdigest()[:12]
    key = st.sidebar.selectbox(
        "回合",
        keys,
        index=keys.index(selected) if selected in labels else 0,
        format_func=labels.get,
        key=f"row::{signature}",
    )
    st.session_state["selected"] = key
    row = next(r for r in shown if _row_key(r) == key)

    vetoed = vetoed_ids(out)
    with st.sidebar.expander("場次比分（依目前有效判定）"):
        st.dataframe(
            [
                {
                    "場": i,
                    "X": m.score["X"],
                    "B": m.score["B"],
                    "勝方": m.winner or "未完成",
                    "回合數": len(m.rounds),
                }
                for i, m in enumerate(matches_for(rows, rules, vetoed=vetoed), 1)
            ],
            hide_index=True,
        )

    st.header(f"{row['video']} · 第 {row['round']} 回合 · {ui.status(row)}")
    if ui.is_synthetic(row):
        st.warning(
            "synthetic_detector：此判定來自合成資料訓練的偵測器，只是流程（smoke）測試，"
            "不是可信的正式判定。"
        )
    source = ui.source_video(videos, row["video"])
    if ui.is_diagnostic(row):
        st.error(
            "這一列不是可判定的回合（分析診斷："
            + "、".join(row.get("review_reasons") or [])
            + "），不能修改判定或確認為已覆核，也不列入比分。"
        )
        st.markdown("**修復步驟**")
        st.markdown(
            "\n".join(f"{i}. {s}" for i, s in enumerate(ui.repair_steps(row, source, out), 1))
        )
        return

    snapshot = _calibration(row, out, videos, rules)
    left, right = st.columns([3, 2], gap="large")
    with left:
        _evidence(row, source, out, a, snapshot)
    with right:
        _judgment(row, rows, rules, vetoed)
        _form(row, out, rules)
        history = [
            {
                "時間": c.at,
                "欄位": ui.FIELDS.get(c.field, (c.field,))[0],
                "原值": json.dumps(c.old, ensure_ascii=False),
                "新值": json.dumps(c.new, ensure_ascii=False),
            }
            for c in load_corrections(out)
            if (c.video, c.round) == (row["video"], row["round"])
        ]
        st.markdown("**本回合修正紀錄**")
        if history:
            st.dataframe(history, hide_index=True, width="stretch")
        else:
            st.caption("（尚無）")
    with st.expander("原始資料列（rounds.json）"):
        st.json(row, expanded=False)


if __name__ == "__main__":
    main()
