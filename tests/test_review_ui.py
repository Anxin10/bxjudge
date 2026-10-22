"""Review UI: read-only helpers and the Streamlit app driven through AppTest."""

import sys
from pathlib import Path

import numpy as np
import pytest

from bxjudge import review_ui, vlm
from bxjudge.corrections import Correction, load_corrections
from bxjudge.output import effective, read_rounds, record_video_timing, write_results
from tests.helpers import make_calibration, make_video, rr

APP = Path(__file__).resolve().parents[1] / "src" / "bxjudge" / "review_app.py"
AT = "2026-10-23T00:00:00"


def row(**updates):
    return {
        "video": "nested/a.mp4",
        "round": 1,
        "review_reasons": [],
        "needs_review": True,
        "reviewed": False,
        "finish": "spin",
        "winner": "X",
        "reviewer_fix": None,
        **updates,
    }


# ----------------------------------------------------------------------------- helpers


def test_every_editable_field_has_a_chinese_label():
    from bxjudge.corrections import EDITABLE_FIELDS

    assert set(review_ui.FIELDS) == set(EDITABLE_FIELDS)
    for field, (label, options) in review_ui.FIELDS.items():
        allowed = EDITABLE_FIELDS[field]
        assert label and set(options) == ({True, False} if allowed is bool else set(allowed))


def test_status_distinguishes_pending_model_flag_from_human_review():
    assert review_ui.status(row()) == "待覆核"
    assert review_ui.status(row(needs_review=False)) == "未標記"
    # human review completes the original flag, which stays visible as a diagnostic
    assert review_ui.status(row(reviewed=True)) == "已覆核"
    assert review_ui.status(row(review_reasons=["decode_error"], reviewed=True)) == "分析診斷"


def test_filters_and_counts():
    rows = [
        row(),
        row(round=2, reviewed=True),
        row(round=3, needs_review=False),
        row(round=4, review_reasons=["recalibrate"]),
    ]
    assert review_ui.FILTERS == ("全部", "需覆核", "已覆核")
    assert review_ui.filter_rows(rows, "全部") == rows
    assert review_ui.filter_rows(rows, "需覆核") == [rows[0], rows[3]]
    assert review_ui.filter_rows(rows, "已覆核") == [rows[1]]
    assert review_ui.counts(rows) == {"回合": 3, "待覆核": 1, "已覆核": 1, "分析診斷": 1}


def test_synthetic_detector_rows_are_flagged():
    assert review_ui.is_synthetic(row(review_reasons=["synthetic_detector"]))
    prov = {"dataset": {"provenance": {"synthetic": True}}}
    assert review_ui.is_synthetic(row(detector_provenance=prov))
    assert not review_ui.is_synthetic(row(detector_provenance={"dataset": {"provenance": {}}}))


def test_source_video_only_from_explicit_absolute_metadata(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    assert review_ui.source_video({"a.mp4": {"source_video": str(video)}}, "a.mp4") == video
    for videos in ({}, {"a.mp4": {}}, {"a.mp4": {"source_video": "a.mp4"}},
                   {"a.mp4": {"source_video": str(tmp_path / "gone.mp4")}}):
        assert isinstance(review_ui.source_video(videos, "a.mp4"), str)


def test_form_changes_use_current_effective_values_and_skip_unchanged():
    item = row(x_touch="no", reviewer_fix={"winner": "B"})
    changes = review_ui.form_changes(
        item, {"winner": "B", "x_touch": "yes", "match_start": False}, AT
    )
    assert changes == [Correction("nested/a.mp4", 1, "x_touch", "no", "yes", AT)]
    changes = review_ui.form_changes(item, {"winner": "X", "match_start": True}, AT)
    assert changes == [
        Correction("nested/a.mp4", 1, "winner", "B", "X", AT),
        Correction("nested/a.mp4", 1, "match_start", False, True, AT),
    ]


@pytest.mark.parametrize("reason", ["recalibrate", "no_round", "decode_error", "analysis_error"])
def test_repair_steps_are_concrete(tmp_path, reason):
    steps = review_ui.repair_steps(row(review_reasons=[reason]), Path("/v/a.mp4"), tmp_path)
    assert steps and any("bxjudge" in s for s in steps)
    assert any(str(tmp_path) in s for s in steps)


def test_evidence_frame_draws_scaled_zones_at_playback_time(tmp_path, monkeypatch, th):
    video = make_video(tmp_path / "v.mp4", seconds=3, fps=30, size=(320, 180))
    monkeypatch.setattr(vlm, "find_calibration", lambda *a: make_calibration())
    calibration = review_ui.video_calibration(video, tmp_path, th)
    image, t, index = review_ui.evidence_frame(video, 1.5, calibration, th)
    assert index == 45 and t == pytest.approx(1.5)
    assert image.shape == (180, 320, 3)
    # battle_area top edge (640x360 calibration -> y=30 at 320x180) drawn in green (RGB)
    edge = image[29:32, 100:200].reshape(-1, 3).astype(int)
    assert (edge[:, 1] - edge[:, 0] > 60).any()
    # the end moment may sit on the last frame: clamp instead of failing
    _, _, last = review_ui.evidence_frame(video, 3.0, calibration, th)
    assert last == 89


def test_missing_calibration_is_reported_not_invented(tmp_path, monkeypatch, th):
    video = make_video(tmp_path / "v.mp4", seconds=1, fps=30, size=(320, 180))
    monkeypatch.setattr(vlm, "find_calibration", lambda *a: None)
    with pytest.raises(review_ui.ReviewMediaError, match="calibration"):
        review_ui.video_calibration(video, tmp_path, th)


# ----------------------------------------------------------------------------- app


@pytest.fixture
def batch(tmp_path, rules, monkeypatch):
    out = tmp_path / "results"
    source = make_video(tmp_path / "a.mp4", seconds=3, fps=30, size=(320, 180))
    real = rr("spin", "X", video="a.mp4", round=1, t_launch=1.0, review_reasons=["low_confidence"])
    real.needs_review = True
    real.confidence = 0.4
    real.vlm_note = "[m] 問：x 答：否"
    other = rr("spin", "X", video="b.mp4", round=1)
    broken = rr(None, None, video="c.mp4", round=1, review_reasons=["recalibrate"], launch=False)
    broken.needs_review = True
    write_results([real, other, broken], out, rules)
    record_video_timing(out, "a.mp4", 3.0, 1.0, source_video=source)
    record_video_timing(out, "c.mp4", 3.0, 1.0, source_video=tmp_path / "c.mp4")
    (out / "trajectories").mkdir()
    import cv2

    cv2.imwrite(str(out / "trajectories" / "a_r1.png"), np.zeros((18, 32, 3), np.uint8))
    monkeypatch.setattr(vlm, "find_calibration", lambda *a: make_calibration())
    monkeypatch.setattr(sys, "argv", ["review_app.py", str(out), "--calib", str(tmp_path)])
    return out


def app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=60)


def pick(at, key):
    widget(at.sidebar.selectbox, "回合").set_value(key).run()
    assert not at.exception, at.exception
    return at


def widget(items, label):
    found = [w for w in items if w.label == label]
    assert len(found) == 1, [w.label for w in items]
    return found[0]


def test_app_lists_rounds_and_shows_evidence(batch):
    at = app().run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["回合"] == "2" and metrics["分析診斷"] == "1"
    pick(at, "a.mp4::1")
    text = " ".join(e.value for e in [*at.markdown, *at.caption, *at.info, *at.warning])
    assert "low_confidence" in text and "問：x 答：否" in text
    assert "40.0%" in text  # original model confidence kept as a diagnostic
    assert len(at.get("video")) == 1
    assert len(at.get("image")) >= 3  # launch, end, trajectory
    widget(at.selectbox, "獲勝方")
    widget(at.checkbox, "此回合另起新場")


def test_app_saves_a_correction_that_persists(batch):
    at = pick(app().run(), "a.mp4::1")
    widget(at.selectbox, "獲勝方").set_value("B")
    widget(at.selectbox, "X 方落地區域").set_value("battle_area")
    widget(at.checkbox, "此回合另起新場").check()
    widget(at.button, "儲存修正").click().run()
    assert not at.exception, at.exception
    rows = {(r["video"], r["round"]): r for r in read_rounds(batch)}
    fixed = rows[("a.mp4", 1)]
    assert fixed["reviewed"] and effective(fixed)["winner"] == "B"
    assert fixed["reviewer_fix"] == {
        "winner": "B", "x_land_zone": "battle_area", "match_start": True,
    }
    assert {c.field for c in load_corrections(batch)} == {"winner", "x_land_zone", "match_start"}
    assert any("已儲存 3 項修正" in s.value for s in at.success)
    # widgets now show the saved effective values (fresh session = page reload)
    at = pick(app().run(), "a.mp4::1")
    assert widget(at.selectbox, "獲勝方").value == "B"
    assert widget(at.checkbox, "此回合另起新場").value is True


def test_app_rejects_invalid_combination_without_writing(batch):
    before = (batch / "rounds.json").read_bytes()
    at = pick(app().run(), "a.mp4::1")
    widget(at.selectbox, "結束方式").select_index(0)  # no finish, winner kept
    widget(at.button, "儲存修正").click().run()
    assert not at.exception, at.exception
    assert any("winner" in e.value for e in at.error)
    assert (batch / "rounds.json").read_bytes() == before


def test_app_confirm_without_change(batch):
    at = pick(app().run(), "a.mp4::1")
    widget(at.button, "確認無誤").click().run()
    assert not at.exception, at.exception
    # the list label follows the new status and the same round stays selected
    box = widget(at.sidebar.selectbox, "回合")
    assert box.value == "a.mp4::1" and "a.mp4 · 第 1 回合 · 已覆核" in box.options
    fixed = next(r for r in read_rounds(batch) if r["video"] == "a.mp4")
    assert fixed["reviewed"] is True and fixed["reviewer_fix"] is None
    assert load_corrections(batch) == []
    at.sidebar.radio(key="filter").set_value("已覆核").run()
    assert widget(at.sidebar.selectbox, "回合").options == ["a.mp4 · 第 1 回合 · 已覆核"]


def test_app_diagnostic_row_offers_repair_not_judgment(batch):
    at = pick(app().run(), "c.mp4::1")
    assert any("bxjudge calibrate" in e.value for e in [*at.markdown, *at.code])
    labels = {b.label for b in at.button}
    assert "儲存修正" not in labels and "確認無誤" not in labels
    assert not [s for s in at.selectbox if s.label == "獲勝方"]
    assert not (batch / "corrections.jsonl").exists()


def test_app_missing_media_and_trajectory_do_not_crash(batch):
    at = pick(app().run(), "b.mp4::1")
    text = " ".join(e.value for e in [*at.info, *at.warning])
    assert "source_video" in text and "軌跡圖" in text
    widget(at.button, "確認無誤")  # still reviewable without media


# ----------------------------------------------------------------------------- cli


def test_review_cli_launches_loopback_streamlit_without_telemetry(tmp_path, monkeypatch):
    import subprocess

    from bxjudge import cli

    calls = []
    monkeypatch.setattr(subprocess, "call", lambda argv, **kw: calls.append((argv, kw)) or 0)
    (tmp_path / "rounds.json").write_text("[]", encoding="utf-8")
    assert cli.main(["review", str(tmp_path), "--port", "8502"]) == 0
    (argv, kw), = calls
    assert argv[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert argv[4].endswith("review_app.py")
    flags = dict(zip(argv[5:argv.index("--")][::2], argv[5:argv.index("--")][1::2], strict=True))
    assert flags["--server.address"] == "127.0.0.1"
    assert flags["--server.port"] == "8502"
    assert flags["--browser.gatherUsageStats"] == "false"
    assert flags["--server.headless"] == "true"
    assert argv[argv.index("--") + 1] == str(tmp_path.resolve())
    assert kw["env"]["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"


def test_review_cli_rejects_folder_without_results(tmp_path, capsys):
    from bxjudge import cli

    assert cli.main(["review", str(tmp_path / "missing")]) == 2
    assert "error:" in capsys.readouterr().err
