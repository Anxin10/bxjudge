import json
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import yaml

from bxjudge.cli import main
from bxjudge.evaluate import (
    METRIC_NAMES,
    Metrics,
    derive_shake,
    derive_shake_px,
    error_groups,
    evaluate,
    load_eval_config,
    match_truth,
    passes,
    report_metrics,
    reversed_finish_check,
    shake_samples,
    spin_dir_check,
    thresholds_diff,
)
from tests.helpers import rev

ROOT = Path(__file__).resolve().parents[1]
EVAL_CONFIG = ROOT / "config" / "evaluation.yaml"
THRESHOLDS = ROOT / "config" / "thresholds.yaml"


# ---- brief tests (verbatim) ------------------------------------------------


def test_finish_accuracy_unflagged():
    rows = [rev(finish="spin", fix={}), rev(finish="spin", fix={"finish":"over"}), rev(finish="spin", needs_review=True, fix={"finish":"burst"})]
    assert evaluate(rows).finish_accuracy_unflagged == 0.5


def test_touch_recall_counts_flagged_as_caught():
    rows = [rev(x_touch="no", needs_review=True, fix={"x_touch":"yes"}), rev(x_touch="no", fix={"x_touch":"yes"})]
    assert evaluate(rows).touch_recall == 0.5


def test_passes_thresholds():
    exact = {"finish_accuracy_unflagged_method": "exact"}  # R64: a value needs its basis
    assert all(v == "PASS" for v in passes(Metrics(0.96, 0.92, 0.91, 0.25, 0.8, 50, **exact)).values())
    assert passes(Metrics(0.90, 0.92, 0.91, 0.25, 0.8, 50))["finish_accuracy_unflagged"] == "FAIL"
    assert passes(Metrics(0.96, 0.92, 0.91, 0.35, 0.8, 50))["review_ratio"] == "FAIL"


def test_derive_shake_px_midpoint():
    assert derive_shake_px({"normal":[0.5,1.0], "touch":[3.0,4.0], "bey_impact":[2.0]}) == 2.0


# ---- evaluate: truth, denominators, flags -----------------------------------


def _rows_of_videos(*videos):
    return [rev(video=v, round=1) for v in videos]


def test_no_rows_is_unverified_never_perfect():
    m = evaluate([])
    assert all(math.isnan(getattr(m, name)) for name in METRIC_NAMES)
    assert m.n_reviewed == 0
    assert set(passes(m)) == set(METRIC_NAMES)
    assert "PASS" not in passes(m).values()


def test_only_flagged_truth_leaves_finish_accuracy_unverified():
    m = evaluate([rev(needs_review=True), rev(round=2, needs_review=True)])
    assert math.isnan(m.finish_accuracy_unflagged)
    assert m.review_ratio == 1.0 and m.n_reviewed == 2
    assert passes(m)["finish_accuracy_unflagged"] == "FAIL"


def test_no_true_touch_or_air_leaves_recalls_unverified():
    m = evaluate([rev(fix={})])
    assert math.isnan(m.touch_recall) and math.isnan(m.air_recall)
    assert m.n_touch_truth == 0 and m.n_air_truth == 0


def test_unreviewed_rows_are_never_truth():
    rows = [rev(reviewed=False, fix={"finish": "over", "x_touch": "yes"})]
    m = evaluate(rows)
    assert m.n_reviewed == 0
    assert math.isnan(m.finish_accuracy_unflagged) and math.isnan(m.touch_recall)
    assert m.review_ratio == 0.0  # predictions still count for the review ratio


def test_winner_correction_counts_as_wrong_finish():
    m = evaluate([rev(fix={"winner": "B"}), rev(round=2, fix={})])
    assert m.finish_accuracy_unflagged == 0.5


def test_confirmed_without_fix_counts_as_correct():
    m = evaluate([rev(fix=None)])
    assert m.finish_accuracy_unflagged == 1.0


def test_touch_recall_counts_each_true_side():
    rows = [
        rev(x_touch="yes", b_touch="no", fix={"b_touch": "yes"}),  # X caught, B missed
        rev(round=2, x_touch="yes"),  # X caught (detected)
    ]
    m = evaluate(rows)
    assert m.n_touch_truth == 3
    assert m.touch_recall == pytest.approx(2 / 3)


def test_air_recall_detected_or_flagged():
    rows = [
        rev(air_collision="yes"),
        rev(round=2, needs_review=True, fix={"air_collision": "yes"}),
        rev(round=3, fix={"air_collision": "yes"}),
        rev(round=4, air_collision="yes", fix={"air_collision": "no"}),  # false alarm, no truth
    ]
    m = evaluate(rows)
    assert m.n_air_truth == 3
    assert m.air_recall == pytest.approx(2 / 3)


def test_review_ratio_uses_original_flags_and_excludes_placeholders():
    rows = [
        rev(needs_review=True, reviewed=True, fix={}),
        rev(round=2),
        rev(round=3, reviewed=False),
        rev(round=4, reviewed=False, needs_review=True),
        rev(video="p.mp4", needs_review=True, reviewed=False, finish=None, winner=None,
            review_reasons=["recalibrate"]),
    ]
    m = evaluate(rows)
    assert m.review_ratio == 0.5
    assert m.n_rounds == 4 and m.n_placeholders == 1


def test_placeholder_rows_are_never_truth():
    rows = [rev(finish=None, winner=None, review_reasons=["analysis_error"], fix={"finish": "spin"})]
    m = evaluate(rows)
    assert m.n_reviewed == 0 and m.n_rounds == 0 and math.isnan(m.review_ratio)


# ---- speed (R11, R59) -------------------------------------------------------


def test_speed_without_timings_is_unverified_and_fails():
    m = evaluate(_rows_of_videos("a.mp4"))
    assert math.isnan(m.speed_ratio) and math.isnan(m.speed_ratio_core)
    assert passes(m)["speed_ratio"] == "FAIL"


def test_speed_total_includes_vlm_phase_and_core_is_diagnostic():
    rows = _rows_of_videos("a.mp4")
    timings = {"a.mp4": {"duration_s": 100.0, "processing_s": 50.0, "vlm_s": 70.0}}
    m = evaluate(rows, timings)
    assert m.speed_ratio == pytest.approx(1.2)
    assert m.speed_ratio_core == pytest.approx(0.5)
    assert m.n_speed_videos == 1 and m.n_vlm_videos == 1
    assert passes(m)["speed_ratio"] == "FAIL"


def test_speed_without_vlm_phase_is_core_time():
    rows = _rows_of_videos("a.mp4", "b.mp4", "c.mp4")
    timings = {
        "a.mp4": {"duration_s": 100.0, "processing_s": 50.0},
        "b.mp4": {"duration_s": 100.0, "processing_s": 80.0},
        "c.mp4": {"duration_s": 100.0, "processing_s": 200.0},
    }
    m = evaluate(rows, timings)
    assert m.speed_ratio == pytest.approx(0.8) == m.speed_ratio_core
    assert m.n_vlm_videos == 0
    assert passes(m)["speed_ratio"] == "PASS"


def test_speed_excludes_placeholder_only_unmeasured_and_overlong_videos():
    rows = [
        *_rows_of_videos("a.mp4", "long.mp4", "vlm_only.mp4", "zero_core.mp4"),
        rev(video="gate.mp4", finish=None, winner=None, review_reasons=["recalibrate"],
            reviewed=False, needs_review=True),
    ]
    timings = {
        "a.mp4": {"duration_s": 100.0, "processing_s": 90.0},
        "gate.mp4": {"duration_s": 100.0, "processing_s": 1.0},  # early gate, not judgment
        "long.mp4": {"duration_s": 301.0, "processing_s": 1.0},  # spec: videos up to 5 min
        "vlm_only.mp4": {"vlm_s": 5.0},  # no core measurement
        "zero.mp4": {"duration_s": 0.0, "processing_s": 1.0},
        # a measured core run is never 0 s: a 0 is a placeholder, not a perfect speed
        "zero_core.mp4": {"duration_s": 100.0, "processing_s": 0.0},
    }
    m = evaluate(rows, timings)
    assert m.speed_ratio == pytest.approx(0.9)
    assert m.n_speed_videos == 1
    excluded = report_metrics(rows, timings, load_eval_config(EVAL_CONFIG)).excluded
    assert excluded == {"no_real_round": ["gate.mp4", "zero.mp4"],
                        "unmeasured": ["vlm_only.mp4", "zero_core.mp4"],
                        "over_max_duration": ["long.mp4"]}


# ---- passes / config (R40) ------------------------------------------------


def test_passes_boundaries_are_inclusive_and_nan_fails():
    exact = {"finish_accuracy_unflagged_method": "exact"}
    assert all(v == "PASS" for v in passes(Metrics(0.95, 0.90, 0.90, 0.30, 1.0, 1, **exact)).values())
    assert passes(Metrics(0.96, 0.92, 0.91, 0.25, math.nan, 1))["speed_ratio"] == "FAIL"


def test_targets_come_from_yaml(tmp_path):
    raw = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    raw["targets"]["review_ratio"]["max"] = 0.2
    p = tmp_path / "evaluation.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_eval_config(p)
    assert passes(Metrics(0.96, 0.92, 0.91, 0.25, 0.8, 50), cfg)["review_ratio"] == "FAIL"


def test_default_targets_are_spec_section_11():
    cfg = load_eval_config(EVAL_CONFIG)
    assert cfg.targets == {
        "finish_accuracy_unflagged": ("min", 0.95),
        "touch_recall": ("min", 0.90),
        "air_recall": ("min", 0.90),
        "review_ratio": ("max", 0.30),
        "speed_ratio": ("max", 1.0),
    }
    assert cfg.speed_max_video_s == 300


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(extra=1),
    lambda d: d.pop("truth_t_launch_tolerance_s"),
    lambda d: d["targets"].pop("air_recall"),
    lambda d: d["targets"].update(touch_recall={"above": 0.9}),
    lambda d: d["targets"].update(touch_recall={"min": "high"}),
    lambda d: d.update(shake_overlap_percentile=101),
    lambda d: d.update(truth_t_launch_tolerance_s=-1),
    lambda d: d.update(audit_confidence_level=1.0),
    lambda d: d.pop("audit_confidence_level"),
])
def test_eval_config_rejects_invalid(tmp_path, mutate):
    raw = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    mutate(raw)
    p = tmp_path / "evaluation.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_config(p)


# ---- error grouping -----------------------------------------------------------


def test_error_groups_by_cause():
    rows = [
        rev(fix={"finish": "over"}),  # unflagged wrong finish
        rev(round=2, fix={"x_touch": "yes"}),  # missed touch
        rev(round=3, fix={"air_collision": "yes"}),  # missed air collision
        rev(round=4, needs_review=True, fix={"winner": "B"}),  # flagged then corrected
        rev(round=5, fix={}),  # correct
    ]
    groups = error_groups(rows)
    assert [(e["video"], e["round"]) for e in groups["finish_wrong_unflagged"]] == [("v1.mp4", 1)]
    assert [e["round"] for e in groups["touch_missed"]] == [2]
    assert groups["touch_missed"][0]["side"] == "X"
    assert [e["round"] for e in groups["air_collision_missed"]] == [3]
    assert [e["round"] for e in groups["corrected_after_flag"]] == [4]


# ---- derive_shake_px / samples (R35) ----------------------------------------


def test_derive_shake_overlap_uses_normal_percentile_and_warns():
    normal = [float(v) for v in range(1, 101)]
    d = derive_shake({"normal": normal, "touch": [50.0], "bey_impact": []}, percentile=99)
    assert d.overlap and d.method == "normal_percentile"
    assert d.value == pytest.approx(float(np.percentile(normal, 99)))
    assert derive_shake_px({"normal": normal, "touch": [50.0], "bey_impact": []}) == pytest.approx(d.value)


def test_derive_shake_equal_extremes_count_as_overlap():
    d = derive_shake({"normal": [1.0, 2.0], "touch": [2.0], "bey_impact": []}, percentile=99)
    assert d.overlap


def test_derive_shake_without_both_classes_is_unverified():
    assert math.isnan(derive_shake_px({"normal": [1.0], "touch": [], "bey_impact": [2.0]}))
    assert math.isnan(derive_shake_px({"normal": [], "touch": [3.0], "bey_impact": []}))


def _evidence(video, disp, impacts, window=None):
    frames = sorted(int(f) for f in disp)
    lo, hi = window or (frames[0], frames[-1] + 1)
    return {"video": video, "round": 1, "window": [lo, hi], "window_semantics": "[start,end)",
            "fps": 30.0, "displacements": {str(f): v for f, v in disp.items()},
            "impact_frames": impacts}


def test_shake_samples_exclude_impact_windows_and_unknown_frames():
    quiet = _evidence("q.mp4", {0: 0.5, 1: None, 2: 0.7, 3: 9.0, 4: 8.0, 5: 0.6, 6: 0.4}, [3])
    touch = _evidence("t.mp4", {0: 0.2, 1: 3.5, 2: 0.3, 3: 7.0, 4: None}, [3])
    rows = [
        rev(video="q.mp4", motion_evidence=quiet, fix={}),
        # original said no, reviewer says X touched: positive row from the effective truth
        rev(video="t.mp4", motion_evidence=touch, fix={"x_touch": "yes"}),
        rev(video="u.mp4", motion_evidence=quiet, fix={"b_touch": "unknown"}),  # unknown truth
        rev(video="n.mp4", motion_evidence=quiet, reviewed=False),  # not truth
        rev(video="none.mp4", fix={}),  # no motion evidence: never fabricated
    ]
    s = shake_samples(rows, impact_coincide_frames=1)
    # q: frames 2,3,4 are within 1 of impact 3; frame 1 is unmeasured
    assert sorted(s.samples["normal"]) == [0.4, 0.5, 0.6]
    assert sorted(s.samples["bey_impact"]) == [0.3, 0.7, 7.0, 8.0, 9.0]
    # t: one summary sample per positive row = max non-impact finite displacement
    assert s.samples["touch"] == [3.5]
    assert s.rows_normal == 1 and s.rows_touch == 1
    assert s.rows_unknown_truth == 1 and s.rows_without_evidence == 1
    # unmeasured: q frame 1, t frame 4; impact-excluded measured frames: q 2,3,4 + t 2,3
    assert s.frames_unmeasured == 2 and s.frames_impact_excluded == 5


def test_shake_samples_positive_row_without_clean_frames_gives_no_sample():
    ev = _evidence("t.mp4", {0: 5.0, 1: None}, [0])
    s = shake_samples([rev(video="t.mp4", x_touch="yes", motion_evidence=ev)], impact_coincide_frames=0)
    assert s.samples["touch"] == [] and s.rows_touch_without_sample == 1


def test_shake_samples_only_inside_half_open_window():
    ev = _evidence("q.mp4", {0: 0.5, 1: 0.6, 2: 0.7}, [], window=(0, 2))
    s = shake_samples([rev(video="q.mp4", motion_evidence=ev)], impact_coincide_frames=3)
    assert sorted(s.samples["normal"]) == [0.5, 0.6]


def test_thresholds_diff_is_advisory_unified_diff(tmp_path):
    p = tmp_path / "thresholds.yaml"
    original = THRESHOLDS.read_text(encoding="utf-8")
    p.write_text(original, encoding="utf-8")
    diff = thresholds_diff(p, "shake_px", 2.75)
    assert "-shake_px: 2.0" in diff and "+shake_px: 2.75" in diff
    assert sum(1 for line in diff.splitlines() if line.startswith(("+s", "-s"))) == 2
    assert p.read_text(encoding="utf-8") == original
    assert thresholds_diff(p, "shake_px", 2.0) == ""


# ---- spin direction / reversed finishes (Step 5b tooling) ---------------------


def test_spin_dir_check_accuracy_and_swap_advice():
    rows = [
        rev(round=i, x_spin_dir="right", b_spin_dir="left",
            fix={"x_spin_dir": "left", "b_spin_dir": "right"})
        for i in range(1, 6)
    ] + [rev(round=9, x_spin_dir="unknown", b_spin_dir="right", fix={"x_spin_dir": "left"})]
    c = spin_dir_check(rows, min_samples=10)
    assert c["known"] == 12 and c["original_unknown"] == 1
    assert c["judged"] == 11 and c["correct"] == 1
    assert c["accuracy"] == pytest.approx(1 / 11)
    assert c["swap_suggested"] is True
    assert spin_dir_check(rows, min_samples=50)["swap_suggested"] is None  # too few


def test_spin_dir_check_without_truth_is_unverified():
    c = spin_dir_check([rev(reviewed=False)], min_samples=1)
    assert math.isnan(c["accuracy"]) and c["swap_suggested"] is None


def test_reversed_finish_overturn_ratio():
    rows = [
        rev(winner="X", review_reasons=["B_stop_reversed"], needs_review=True, fix={"finish": None, "winner": None}),
        rev(round=2, winner="X", review_reasons=["B_stop_reversed"], needs_review=True, fix={}),
        rev(round=3, winner="X", review_reasons=["after_finish:B_stop_reversed"], fix={"winner": "B"}),
        rev(round=4, winner="X", review_reasons=["X_stop_reversed"], fix={"winner": "B"}),  # winner's own
        rev(round=5, winner="X", review_reasons=["B_stop_reversed"], reviewed=False),
    ]
    c = reversed_finish_check(rows)
    assert (c["n"], c["overturned"]) == (2, 1) and c["ratio"] == 0.5
    assert math.isnan(reversed_finish_check([])["ratio"])


# ---- R37 explicit read-only reference truth ---------------------------------


def _pred(video, round, t_launch=0.0, **kw):
    return rev(video=video, round=round, t_launch=t_launch, reviewed=False, **kw)


def test_match_truth_applies_reference_truth_to_fresh_predictions():
    pred = [_pred("a.mp4", 1, finish="spin"), _pred("a.mp4", 2, t_launch=20.0)]
    truth = [rev(video="a.mp4", round=1, finish="over", fix={"finish": "spin"}),
             rev(video="a.mp4", round=2, t_launch=20.2, fix={"x_touch": "yes"})]
    m = match_truth(pred, truth, tolerance_s=0.5)
    assert m.matched == 2 and m.n_truth == 2 and m.coverage == 1.0
    assert not any(m.issues.values())
    rows = {(r["video"], r["round"]): r for r in m.rows}
    # prediction originals kept, reference effective values become the truth
    assert rows[("a.mp4", 1)]["finish"] == "spin" and rows[("a.mp4", 1)]["reviewed"] is True
    assert rows[("a.mp4", 1)]["reviewer_fix"]["finish"] == "spin"
    assert rows[("a.mp4", 2)]["reviewer_fix"]["x_touch"] == "yes"
    met = evaluate(m.rows)
    assert met.finish_accuracy_unflagged == 1.0 and met.touch_recall == 0.0


def test_match_truth_reports_and_excludes_ambiguous_matches():
    pred = [
        _pred("dup.mp4", 1), _pred("dup.mp4", 2),
        _pred("seg.mp4", 1),  # truth has 2 rounds: segmentation changed
        _pred("late.mp4", 1, t_launch=3.0),
        _pred("ok.mp4", 1),
        _pred("own.mp4", 1, fix={"finish": "over"}),  # a review in the prediction dir is not truth
        _pred("ph.mp4", 1, finish=None, winner=None, review_reasons=["recalibrate"]),
    ]
    pred[-2]["reviewed"] = True
    truth = [
        rev(video="dup.mp4", round=1), rev(video="dup.mp4", round=1),
        rev(video="dup.mp4", round=2),
        rev(video="seg.mp4", round=1), rev(video="seg.mp4", round=2, reviewed=False),
        rev(video="late.mp4", round=1, t_launch=2.0),
        rev(video="gone.mp4", round=1),
        rev(video="ok.mp4", round=1),
        rev(video="ph.mp4", round=1),
        rev(video="unrev.mp4", round=1, reviewed=False),
    ]
    m = match_truth(pred, truth, tolerance_s=0.5)
    assert m.issues["duplicate_truth_key"] == [("dup.mp4", 1)]
    assert m.issues["segmentation_mismatch"] == [("seg.mp4", 1)]
    assert m.issues["time_mismatch"] == [("late.mp4", 1)]
    assert m.issues["missing_prediction"] == [("gone.mp4", 1)]
    assert m.issues["prediction_not_a_round"] == [("ph.mp4", 1)]
    assert m.matched == 2  # dup.mp4 round 2 and ok.mp4
    # distinct reviewed real truth keys: dup 1, dup 2, seg 1, late, gone, ok, ph
    assert m.n_truth == 7 and m.coverage == pytest.approx(2 / 7)
    reviewed = {(r["video"], r["round"]) for r in m.rows if r["reviewed"]}
    assert reviewed == {("dup.mp4", 2), ("ok.mp4", 1)}
    own = next(r for r in m.rows if r["video"] == "own.mp4")
    assert own["reviewer_fix"] is None


def test_match_truth_duplicate_prediction_keys_are_excluded():
    pred = [_pred("a.mp4", 1), _pred("a.mp4", 1)]
    m = match_truth(pred, [rev(video="a.mp4", round=1)], tolerance_s=0.5)
    assert m.issues["duplicate_prediction_key"] == [("a.mp4", 1)] and m.matched == 0
    assert math.isnan(evaluate(m.rows).finish_accuracy_unflagged)


def test_match_truth_without_reviewed_truth_has_unverified_coverage():
    m = match_truth([_pred("a.mp4", 1)], [], tolerance_s=0.5)
    assert m.n_truth == 0 and math.isnan(m.coverage)


# ---- CLI (eval / tune) ------------------------------------------------------


def _results(dirpath, rows, timings):
    dirpath.mkdir(parents=True)
    (dirpath / "rounds.json").write_text(json.dumps(rows), encoding="utf-8")
    (dirpath / "videos.json").write_text(json.dumps(timings), encoding="utf-8")
    return dirpath


def _snapshot(d):
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_cli_eval_writes_report_and_never_modifies_results(tmp_path, capsys):
    rows = [rev(video="a.mp4", fix={"finish": "over"}), rev(video="b.mp4", fix={}),
            rev(video="c.mp4", needs_review=True, reviewed=False,
                review_reasons=["synthetic_detector"])]
    res = _results(tmp_path / "res", rows,
                   {"a.mp4": {"duration_s": 10.0, "processing_s": 5.0, "vlm_s": 2.0}})
    before = _snapshot(res)
    report = tmp_path / "out" / "eval.md"
    assert main(["eval", str(res), "--config", str(EVAL_CONFIG), "--report", str(report)]) == 0
    assert _snapshot(res) == before
    text = report.read_text(encoding="utf-8")
    for name in METRIC_NAMES:
        assert name in text
    assert "finish_wrong_unflagged" in text and "a.mp4" in text
    assert "SMOKE" in text  # synthetic detector rows are never acceptance (R44/R48)
    assert "core" in text and "vlm" in text.lower()
    out = capsys.readouterr().out
    assert str(report) in out


def test_cli_eval_default_report_path(tmp_path, monkeypatch):
    res = _results(tmp_path / "res", [rev()], {})
    monkeypatch.chdir(tmp_path)
    assert main(["eval", str(res), "--config", str(EVAL_CONFIG),
                 "--rules", str(ROOT / "config" / "rules.yaml")]) == 0
    assert (tmp_path / "reports" / f"eval-{date.today():%Y%m%d}.md").is_file()


def test_cli_eval_with_reference_truth_is_read_only(tmp_path):
    pred = _results(tmp_path / "pred", [_pred("a.mp4", 1), _pred("b.mp4", 1)],
                    {"a.mp4": {"duration_s": 10.0, "processing_s": 5.0}})
    ref = _results(tmp_path / "ref", [rev(video="a.mp4", fix={"finish": "burst"}),
                                      rev(video="z.mp4")], {})
    (ref / "corrections.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    before = (_snapshot(pred), _snapshot(ref))
    report = tmp_path / "eval.md"
    assert main(["eval", str(pred), "--truth", str(ref), "--config", str(EVAL_CONFIG),
                 "--report", str(report)]) == 0
    assert (_snapshot(pred), _snapshot(ref)) == before
    text = report.read_text(encoding="utf-8")
    assert "coverage" in text.lower() and "missing_prediction" in text and "z.mp4" in text


def test_cli_eval_missing_rounds_fails(tmp_path, capsys):
    assert main(["eval", str(tmp_path), "--config", str(EVAL_CONFIG),
                 "--report", str(tmp_path / "r.md")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_tune_prints_advisory_diff_and_never_overwrites(tmp_path, capsys):
    quiet = _evidence("q.mp4", {0: 0.5, 1: 1.0}, [])
    touch = _evidence("t.mp4", {0: 4.0, 1: 0.2}, [])  # midpoint (1.0 + 4.0) / 2
    rows = [rev(video="q.mp4", motion_evidence=quiet, fix={}),
            rev(video="t.mp4", motion_evidence=touch, fix={"b_touch": "yes"})]
    res = _results(tmp_path / "res", rows, {})
    th = tmp_path / "thresholds.yaml"
    th.write_text(THRESHOLDS.read_text(encoding="utf-8"), encoding="utf-8")
    before = (_snapshot(res), th.read_bytes())
    report = tmp_path / "tune.md"
    assert main(["tune", str(res), "--thresholds", str(th), "--config", str(EVAL_CONFIG),
                 "--report", str(report)]) == 0
    assert (_snapshot(res), th.read_bytes()) == before
    out = capsys.readouterr().out
    assert re.search(r"^-shake_px: 2\.0$", out, re.MULTILINE)
    assert re.search(r"^\+shake_px: 2\.5$", out, re.MULTILINE)
    text = report.read_text(encoding="utf-8")
    assert "advisory" in text.lower() and "pocket_confirm_s" in text
    assert "air_evidence_max_missing_fraction" in text and "orbit_to_spin" in text


def test_cli_tune_without_samples_proposes_nothing(tmp_path, capsys):
    res = _results(tmp_path / "res", [rev()], {})
    report = tmp_path / "tune.md"
    assert main(["tune", str(res), "--thresholds", str(THRESHOLDS), "--config",
                 str(EVAL_CONFIG), "--report", str(report)]) == 0
    assert "+shake_px" not in capsys.readouterr().out
    assert "unverified" in report.read_text(encoding="utf-8").lower()


# ---- R60 report metrics -------------------------------------------------------


def _req(round, seconds, *, ok=True, cold=False, model="m"):
    return {"round": round, "seconds": seconds, "model": model if ok else None,
            "ok": ok, "cold": cold if ok else None}


def _cfg():
    return load_eval_config(EVAL_CONFIG)


def test_core_and_end_to_end_latency_median_p95():
    rows = _rows_of_videos("a.mp4", "b.mp4", "c.mp4")
    timings = {
        "a.mp4": {"duration_s": 100.0, "processing_s": 10.0},
        "b.mp4": {"duration_s": 100.0, "processing_s": 20.0, "vlm_s": 100.0,
                  "vlm_requests": [_req(1, 90.0, cold=True)]},
        "c.mp4": {"duration_s": 100.0, "processing_s": 30.0},
    }
    r = report_metrics(rows, timings, _cfg())
    assert r.latency_percentile == 95
    assert r.core_s == {"median": 20.0, "p": pytest.approx(np.percentile([10, 20, 30], 95))}
    assert r.core_ratio["median"] == pytest.approx(0.2)
    assert r.total_s["median"] == 30.0
    assert r.total_s["p"] == pytest.approx(np.percentile([10, 120, 30], 95))
    assert r.total_ratio["median"] == pytest.approx(0.3)
    assert [v["video"] for v in r.videos] == ["a.mp4", "b.mp4", "c.mp4"]
    assert r.videos[1]["total_s"] == 120.0 and r.videos[1]["total_ratio"] == pytest.approx(1.2)


def test_speed_pass_rate_uses_end_to_end_with_core_diagnostic():
    rows = _rows_of_videos("a.mp4", "b.mp4")
    timings = {
        "a.mp4": {"duration_s": 100.0, "processing_s": 100.0},  # exactly the limit passes
        "b.mp4": {"duration_s": 100.0, "processing_s": 50.0, "vlm_s": 60.0,
                  "vlm_requests": []},
    }
    r = report_metrics(rows, timings, _cfg())
    assert r.speed_pass_rate == 0.5 and r.core_pass_rate == 1.0


def test_latency_and_rates_without_timings_are_unverified():
    r = report_metrics(_rows_of_videos("a.mp4"), None, _cfg())
    assert math.isnan(r.core_s["median"]) and math.isnan(r.total_ratio["p"])
    assert math.isnan(r.speed_pass_rate) and math.isnan(r.core_pass_rate)
    assert math.isnan(r.vlm_request_s["median"]) and math.isnan(r.vlm_trigger_rate)


def test_vlm_request_latency_with_cold_and_warm():
    rows = [rev(video="a.mp4", round=i) for i in (1, 2, 3)] + [rev(video="b.mp4")]
    timings = {
        "a.mp4": {"duration_s": 60.0, "processing_s": 10.0, "vlm_s": 200.0,
                  "vlm_requests": [_req(1, 120.0, cold=True), _req(2, 20.0), _req(3, 30.0, ok=False)]},
        "b.mp4": {"duration_s": 60.0, "processing_s": 10.0, "vlm_s": 12.0,
                  "vlm_requests": [_req(1, 10.0)]},
    }
    r = report_metrics(rows, timings, _cfg())
    q = r.vlm_request_s
    assert q["n"] == 4 and q["n_failed"] == 1
    assert q["median"] == 25.0 and q["max"] == 120.0
    assert q["p"] == pytest.approx(np.percentile([120, 20, 30, 10], 95))
    assert q["cold"] == {"n": 1, "median": 120.0} and q["warm"] == {"n": 2, "median": 15.0}


def test_vlm_latency_unmeasured_when_phase_time_lacks_request_records():
    timings = {"a.mp4": {"duration_s": 60.0, "processing_s": 10.0, "vlm_s": 50.0}}
    r = report_metrics(_rows_of_videos("a.mp4"), timings, _cfg())
    assert math.isnan(r.vlm_request_s["median"]) and math.isnan(r.vlm_trigger_rate)
    assert r.vlm_unrecorded_videos == ["a.mp4"]


def test_vlm_trigger_rate_counts_real_rounds_actually_sent():
    rows = [
        rev(video="a.mp4", round=1, needs_review=True),
        rev(video="a.mp4", round=2, needs_review=True),
        rev(video="a.mp4", round=3),
        rev(video="a.mp4", round=4),
        rev(video="p.mp4", round=1, finish=None, winner=None, review_reasons=["recalibrate"]),
    ]
    timings = {"a.mp4": {"duration_s": 60.0, "processing_s": 10.0, "vlm_s": 30.0,
                         "vlm_requests": [_req(1, 10.0), _req(1, 5.0, ok=False), _req(2, 9.0),
                                          _req(9, 1.0)]}}  # round 9 is not a current round
    r = report_metrics(rows, timings, _cfg())
    assert r.n_vlm_rounds == 2 and r.vlm_trigger_rate == 0.5


def test_vlm_trigger_rate_is_zero_when_timed_but_never_sent():
    timings = {"a.mp4": {"duration_s": 60.0, "processing_s": 10.0}}
    assert report_metrics(_rows_of_videos("a.mp4"), timings, _cfg()).vlm_trigger_rate == 0.0


def test_flagged_rows_vlm_answer_and_unavailable_shares():
    rows = [
        rev(round=1, needs_review=True),
        rev(round=2, needs_review=True),
        rev(round=3, needs_review=True),
        rev(round=4, needs_review=True),
        rev(round=5),
    ]
    rows[0]["vlm_note"] = "[qwen] 問：q 答：a（理由：r）"
    rows[1]["vlm_note"] = "vlm_unavailable: frames: none"
    rows[2]["vlm_note"] = "vlm_unavailable: HTTP 500"
    rows[4]["vlm_note"] = "[qwen] stray note on an unflagged row"
    f = report_metrics(rows, None, _cfg()).flagged_vlm
    assert (f["n_flagged"], f["answered"], f["unavailable"], f["not_asked"]) == (4, 1, 2, 1)
    assert f["answered_share"] == 0.25 and f["unavailable_share"] == 0.5
    assert math.isnan(report_metrics([rev()], None, _cfg()).flagged_vlm["answered_share"])


def test_referee_accuracy_overall_with_flagged_breakdown():
    rows = [
        rev(round=1, fix={}),  # unflagged correct
        rev(round=2, fix={"finish": "over"}),  # unflagged wrong
        rev(round=3, needs_review=True, fix={}),  # flagged correct
        rev(round=4, needs_review=True, fix={"winner": "B"}),  # flagged wrong
        rev(round=5, needs_review=True, fix={"finish": "burst"}),  # flagged wrong
        rev(round=6, reviewed=False),
    ]
    a = report_metrics(rows, None, _cfg()).referee
    assert a["all"] == {"n": 5, "correct": 2, "accuracy": pytest.approx(0.4)}
    assert a["unflagged"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert a["flagged"] == {"n": 3, "correct": 1, "accuracy": pytest.approx(1 / 3)}
    # Referee "unflagged" is descriptive over the reviewed rows (here 1/2); the spec
    # metric needs a random sample or full review: round 6 is unflagged and unreviewed
    # and there is no audit sample, so it is unverified (R62).
    assert math.isnan(evaluate(rows).finish_accuracy_unflagged)
    assert evaluate(rows[:5]).finish_accuracy_unflagged == a["unflagged"]["accuracy"]
    assert math.isnan(report_metrics([], None, _cfg()).referee["all"]["accuracy"])


def test_match_agreement_compares_original_and_effective_matches(rules):
    rows = [rev(round=i, t_launch=float(i), finish="xtreme" if i < 2 else "spin", winner="X")
            for i in (1, 2)]
    rows[1]["reviewer_fix"] = {"finish": "over", "points": 2}  # X 3+2 instead of 3+1
    m = report_metrics(rows, None, _cfg(), rules=rules).match_agreement
    assert (m["n_matches"], m["agree"]) == (1, 0)
    rows[1]["reviewer_fix"] = {}
    m = report_metrics(rows, None, _cfg(), rules=rules).match_agreement
    assert (m["n_matches"], m["agree"], m["ratio"]) == (1, 1, 1.0)
    assert report_metrics(rows, None, _cfg()).match_agreement is None  # needs rules


def test_cli_eval_report_starts_with_r60_summary_table(tmp_path):
    rows = [rev(video="a.mp4", needs_review=True, fix={})]
    rows[0]["vlm_note"] = "[m] ok"
    res = _results(tmp_path / "res", rows, {"a.mp4": {
        "duration_s": 10.0, "processing_s": 4.0, "vlm_s": 8.0,
        "vlm_requests": [_req(1, 7.5, cold=True)]}})
    report = tmp_path / "eval.md"
    assert main(["eval", str(res), "--config", str(EVAL_CONFIG), "--report", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    summary = text.split("\n## ", 1)[0]  # the table precedes every other section
    for label in ("核心判定延遲 Core Latency", "VLM 單次請求延遲 VLM Latency",
                  "VLM 觸發率 VLM Trigger Rate", "端到端延遲 End-to-End Latency",
                  "速度達標率 Speed Pass Rate", "裁判正確率 Referee Accuracy",
                  "finish_accuracy_unflagged", "speed_ratio"):
        assert label in summary
    assert "FAIL" in summary  # speed: (4 + 8) / 10 > 1.0
    assert "model init" in text and "probe" in text


# ---- R61 batch wall-clock ------------------------------------------------------


def _run(wall, videos, *, kind="run", vlm=None):
    return {"kind": kind, "started_at": "2026-10-23T10:00:00+08:00", "wall_s": wall,
            "vlm_wall_s": vlm, "videos": videos}


def test_speed_uses_wall_clock_when_recorded_and_never_double_counts_vlm():
    rows = _rows_of_videos("a.mp4")
    timings = {"a.mp4": {"duration_s": 100.0, "processing_s": 50.0, "vlm_s": 40.0,
                         "vlm_requests": []}}
    # the run took 100 s wall-clock INCLUDING its 40 s VLM phase
    runs = [_run(100.0, {"a.mp4": 100.0}, vlm=40.0)]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "wall-clock"
    assert m.speed_ratio == pytest.approx(1.0)  # 100 / 100, not (100 + 40) / 100
    assert m.speed_ratio_wall == pytest.approx(1.0)
    assert m.speed_ratio_sum == pytest.approx(0.9)  # (50 + 40) / 100, VLM once per video
    assert passes(m)["speed_ratio"] == "PASS"
    w = report_metrics(rows, timings, _cfg(), runs=runs).wall
    assert (w["wall_s"], w["vlm_wall_s"], w["duration_s"]) == (100.0, 40.0, 100.0)


def test_separate_vlm_command_wall_time_adds_without_adding_video_duration():
    rows = _rows_of_videos("a.mp4")
    runs = [_run(60.0, {"a.mp4": 100.0}), _run(30.0, {}, kind="vlm", vlm=30.0)]
    m = evaluate(rows, {"a.mp4": {"duration_s": 100.0, "processing_s": 50.0}}, runs=runs)
    assert m.speed_ratio_wall == pytest.approx(0.9) and m.speed_basis == "wall-clock"


def test_sum_exceeding_wall_clock_is_reported_not_hidden():
    rows = _rows_of_videos("a.mp4", "b.mp4")
    timings = {v: {"duration_s": 100.0, "processing_s": 90.0, "vlm_s": 0.0, "vlm_requests": []}
               for v in ("a.mp4", "b.mp4")}
    runs = [_run(120.0, {"a.mp4": 100.0, "b.mp4": 100.0})]  # e.g. future parallel videos
    w = report_metrics(rows, timings, _cfg(), runs=runs).wall
    assert w["sum_s"] == 180.0 and w["wall_s"] == 120.0 and w["sum_exceeds_wall"] is True
    assert w["ratio"] == pytest.approx(0.6) and w["sum_ratio"] == pytest.approx(0.9)


def test_speed_without_run_records_is_labelled_sum_based():
    m = evaluate(_rows_of_videos("a.mp4"), {"a.mp4": {"duration_s": 100.0, "processing_s": 50.0}})
    assert m.speed_basis == "sum-based" and m.speed_ratio == m.speed_ratio_sum == 0.5
    assert math.isnan(m.speed_ratio_wall)
    assert evaluate(_rows_of_videos("a.mp4")).speed_basis == "unverified"


def test_wall_clock_ignores_runs_without_real_rounds():
    rows = [rev(video="g.mp4", finish=None, winner=None, review_reasons=["recalibrate"])]
    m = evaluate(rows, {}, runs=[_run(1.0, {"g.mp4": 100.0})])
    assert math.isnan(m.speed_ratio_wall) and m.speed_basis == "unverified"


# ---- R62 audit-sampled unflagged accuracy ---------------------------------------


def _unflagged(n, *, sampled=(), reviewed=(), wrong=()):
    rows = []
    for i in range(1, n + 1):
        r = rev(round=i, reviewed=i in reviewed, fix={"finish": "over"} if i in wrong else {})
        if i in sampled:
            r["audit_sample"] = True
        rows.append(r)
    return rows


def _wilson(k, n, level):
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def test_unflagged_accuracy_from_reviewed_audit_sample_with_wilson_interval():
    rows = _unflagged(20, sampled={1, 2, 3, 4, 5, 6}, reviewed={1, 2, 3, 4, 5, 15, 16, 17},
                      wrong={2})
    u = report_metrics(rows, None, _cfg()).unflagged
    assert u["method"] == "audit_sample" and u["n_sampled"] == 6 and u["n_used"] == 5
    assert u["estimate"] == pytest.approx(0.8)
    lo, hi = _wilson(4, 5, _cfg().audit_confidence_level)
    assert (u["low"], u["high"]) == (pytest.approx(lo), pytest.approx(hi))
    # reviewed but not randomly sampled: reported separately as biased
    assert u["biased"] == {"n": 3, "correct": 3, "accuracy": 1.0}
    m = evaluate(rows)
    assert m.finish_accuracy_unflagged == pytest.approx(0.8)
    assert m.finish_accuracy_unflagged_method == "audit_sample"


def test_unflagged_accuracy_is_exact_when_every_unflagged_round_is_reviewed():
    rows = _unflagged(4, reviewed={1, 2, 3, 4}, wrong={1})
    u = report_metrics(rows, None, _cfg()).unflagged
    assert u["method"] == "exact" and u["estimate"] == u["low"] == u["high"] == 0.75


def test_unflagged_accuracy_without_random_sample_is_unverified_selection_bias():
    rows = _unflagged(5, reviewed={1, 2})
    u = report_metrics(rows, None, _cfg()).unflagged
    assert u["method"] == "unverified" and u["reason"] == "selection bias / no random sample"
    assert math.isnan(u["estimate"]) and u["biased"]["n"] == 2
    assert math.isnan(evaluate(rows).finish_accuracy_unflagged)
    assert passes(evaluate(rows))["finish_accuracy_unflagged"] == "FAIL"


def test_no_human_truth_leaves_every_accuracy_unverified():
    rows = _unflagged(3) + [rev(round=9, needs_review=True, reviewed=False)]
    r = report_metrics(rows, None, _cfg())
    assert math.isnan(r.unflagged["estimate"])
    assert math.isnan(r.referee["all"]["accuracy"])
    assert math.isnan(r.tradeoff["auto_accuracy"]["estimate"])
    assert math.isnan(r.tradeoff["final_accuracy"]["estimate"])
    m = evaluate(rows)
    assert all(math.isnan(v) for v in (m.finish_accuracy_unflagged, m.touch_recall, m.air_recall))


# ---- R63 joint trade-off --------------------------------------------------------


def test_tradeoff_block_side_by_side_without_a_weighted_score():
    rows = _unflagged(10, reviewed=set(range(1, 11)), wrong={1, 2})
    rows += [rev(round=11, needs_review=True, fix={}),  # flagged, original right
             rev(round=12, needs_review=True, fix={"winner": "B"}),  # flagged, original wrong
             *(rev(round=i, needs_review=True, reviewed=False) for i in (13, 14, 15))]
    t = report_metrics(rows, None, _cfg()).tradeoff
    assert t["automation_rate"] == pytest.approx(10 / 15)
    assert t["auto_accuracy"]["estimate"] == pytest.approx((10 * 0.8 + 1) / 15)
    assert t["final_accuracy"]["estimate"] == pytest.approx((10 + 2) / 15)
    assert t["flagged_reviewed"] == (2, 5) and t["unflagged_method"] == "exact"
    assert set(t) == {"automation_rate", "auto_accuracy", "final_accuracy", "flagged_reviewed",
                      "unflagged_method", "speed"}


def test_tradeoff_intervals_propagate_the_audit_interval():
    rows = _unflagged(20, sampled={1, 2, 3, 4, 5}, reviewed={1, 2, 3, 4, 5}, wrong={2})
    r = report_metrics(rows, None, _cfg())
    a = r.tradeoff["auto_accuracy"]
    assert a["estimate"] == pytest.approx(0.8)
    assert (a["low"], a["high"]) == (pytest.approx(r.unflagged["low"]),
                                     pytest.approx(r.unflagged["high"]))
    f = r.tradeoff["final_accuracy"]  # 5 reviewed now right, 15 unreviewed at the estimate
    assert f["estimate"] == pytest.approx((15 * 0.8 + 5) / 20)


def test_tradeoff_speed_uses_wall_clock_basis_when_present():
    rows = _rows_of_videos("a.mp4")
    timings = {"a.mp4": {"duration_s": 100.0, "processing_s": 150.0}}
    sum_based = report_metrics(rows, timings, _cfg()).tradeoff["speed"]
    assert sum_based == {"basis": "sum-based", "ratio": 1.5, "pass_rate": 0.0}
    wall = report_metrics(rows, timings, _cfg(), runs=[_run(80.0, {"a.mp4": 100.0})])
    assert wall.tradeoff["speed"] == {"basis": "wall-clock", "ratio": 0.8, "pass_rate": 1.0}


def test_cli_eval_report_shows_wall_audit_and_tradeoff_blocks(tmp_path):
    rows = _unflagged(3, sampled={1}, reviewed={1})
    res = _results(tmp_path / "res", rows, {"v1.mp4": {"duration_s": 10.0, "processing_s": 4.0}})
    (res / "runs.json").write_text(json.dumps([_run(6.0, {"v1.mp4": 10.0})]), encoding="utf-8")
    report = tmp_path / "eval.md"
    assert main(["eval", str(res), "--config", str(EVAL_CONFIG), "--report", str(report)]) == 0
    summary = report.read_text(encoding="utf-8").split("\n## ", 1)[0]
    for text in ("wall-clock", "sum-based", "Wilson", "audit_sample", "自動化率 Automation Rate",
                 "自動判定估計正確率", "覆核後估計正確率", "no weighted score"):
        assert text in summary, text


def test_tradeoff_without_any_human_truth_is_unverified_even_if_all_flagged():
    rows = [rev(round=i, needs_review=True, reviewed=False) for i in (1, 2, 3)]
    t = report_metrics(rows, None, _cfg()).tradeoff
    assert t["automation_rate"] == 0.0
    assert math.isnan(t["auto_accuracy"]["estimate"])  # not a measured 0.0
    assert math.isnan(t["final_accuracy"]["estimate"])


# ---- fix round 1: I1 wall-clock eligible denominator ----------------------------


def test_wall_clock_excludes_gate_video_duration_probe():
    # review probe2: 100 s video with 150 s core + a 200 s recalibrate gate that took 1 s
    rows = _rows_of_videos("a.mp4") + [
        rev(video="g.mp4", finish=None, winner=None, review_reasons=["recalibrate"])]
    timings = {"a.mp4": {"duration_s": 100.0, "processing_s": 150.0},
               "g.mp4": {"duration_s": 200.0, "processing_s": 1.0}}
    runs = [_run(151.0, {"a.mp4": 100.0, "g.mp4": 200.0})]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "wall-clock"
    assert m.speed_ratio == pytest.approx(1.5)  # gate work and duration both left out
    assert passes(m)["speed_ratio"] == "FAIL"  # was 0.503 PASS before the fix
    w = report_metrics(rows, timings, _cfg(), runs=runs).wall
    assert w["duration_s"] == 100.0 and w["excluded"] == ["g.mp4"]


def test_wall_clock_excludes_over_limit_video_work_and_duration():
    rows = _rows_of_videos("a.mp4", "long.mp4")
    timings = {"a.mp4": {"duration_s": 100.0, "processing_s": 150.0},
               "long.mp4": {"duration_s": 900.0, "processing_s": 400.0, "vlm_s": 10.0}}
    runs = [_run(560.0, {"a.mp4": 100.0, "long.mp4": 900.0})]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_ratio == pytest.approx(1.5)  # (560 - 410) / 100, not 560 / 1000
    assert m.speed_basis == "wall-clock"


def test_wall_clock_counts_a_retried_gate_video_once():
    rows = _rows_of_videos("a.mp4", "g.mp4")  # g was gated in run 1, judged in run 2
    timings = {"a.mp4": {"duration_s": 100.0, "processing_s": 50.0},
               "g.mp4": {"duration_s": 200.0, "processing_s": 90.0}}
    runs = [_run(52.0, {"a.mp4": 100.0, "g.mp4": 200.0}), _run(95.0, {"g.mp4": 200.0})]
    w = report_metrics(rows, timings, _cfg(), runs=runs).wall
    assert w["duration_s"] == 300.0  # not 500
    assert w["ratio"] == pytest.approx(147.0 / 300.0)


def test_wall_clock_without_eligible_video_falls_back_to_sum_based_with_reason():
    rows = _rows_of_videos("long.mp4")
    timings = {"long.mp4": {"duration_s": 900.0, "processing_s": 400.0}}
    runs = [_run(400.0, {"long.mp4": 900.0})]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "unverified" and math.isnan(m.speed_ratio_wall)
    w = report_metrics(rows, timings, _cfg(), runs=runs).wall
    assert w["reason"] == "no eligible video in run records"


# ---- fix round 1: I2 match agreement only over fully reviewed matches -------------


def _match_rows(reviewed):
    # xtreme (3) + spin (1): X wins one 4-0 match over two rounds
    return [rev(round=i, t_launch=float(i), finish="xtreme" if i == 1 else "spin", winner="X",
                reviewed=i in reviewed) for i in (1, 2)]


def test_match_agreement_without_reviewed_rounds_is_unverified(rules):
    m = report_metrics(_match_rows(set()), None, _cfg(), rules=rules).match_agreement
    assert (m["n_matches"], m["agree"], m["n_partial"]) == (0, 0, 1)
    assert math.isnan(m["ratio"])


def test_match_agreement_skips_partly_reviewed_matches(rules):
    m = report_metrics(_match_rows({1}), None, _cfg(), rules=rules).match_agreement
    assert m["n_matches"] == 0 and m["n_partial"] == 1 and math.isnan(m["ratio"])
    m = report_metrics(_match_rows({1, 2}), None, _cfg(), rules=rules).match_agreement
    assert (m["n_matches"], m["agree"], m["ratio"], m["n_partial"]) == (1, 1, 1.0, 0)


def test_cli_report_shows_match_agreement_unverified_without_truth(tmp_path):
    rows = _match_rows(set())
    res = _results(tmp_path / "res", rows, None)
    report = tmp_path / "eval.md"
    assert main(["eval", str(res), "--config", str(EVAL_CONFIG), "--report", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert "match score agreement unverified (0/0" in text


# ---- fix round 1: m1 YAML confidence level, m4 partial audit coverage --------------


def test_audit_confidence_level_comes_from_yaml(tmp_path):
    rows = _unflagged(20, sampled={1, 2, 3, 4, 5}, reviewed={1, 2, 3, 4, 5}, wrong={2})
    raw = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    raw["audit_confidence_level"] = 0.80
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_eval_config(path)
    u = report_metrics(rows, None, cfg).unflagged
    lo, hi = _wilson(4, 5, 0.80)
    assert u["level"] == 0.80
    assert (u["low"], u["high"]) == (pytest.approx(lo), pytest.approx(hi))
    lo95, hi95 = _wilson(4, 5, 0.95)
    assert u["low"] > lo95 + 0.01 and u["high"] < hi95  # narrower than the 0.95 default
    m = evaluate(rows, config=cfg)
    assert m.finish_accuracy_unflagged == pytest.approx(0.8)


def test_partly_reviewed_audit_sample_reports_coverage_and_warns(tmp_path):
    rows = _unflagged(20, sampled={1, 2, 3}, reviewed={1})
    u = report_metrics(rows, None, _cfg()).unflagged
    assert u["method"] == "audit_sample" and u["coverage"] == pytest.approx(1 / 3)
    res = _results(tmp_path / "res", rows, None)
    report = tmp_path / "eval.md"
    assert main(["eval", str(res), "--config", str(EVAL_CONFIG), "--report", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert "WARNING: audit sample only partly reviewed (1/3" in text
    full = _unflagged(20, sampled={1, 2}, reviewed={1, 2})
    assert report_metrics(full, None, _cfg()).unflagged["coverage"] == 1.0


# ---- R64: the unflagged-accuracy target needs statistical support ---------------


def _eval_cfg(tmp_path, **change):
    raw = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))
    for key, value in change.items():
        if key == "accuracy_min":
            raw["targets"]["finish_accuracy_unflagged"] = {"min": value}
        else:
            raw[key] = value
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_eval_config(path)


def test_r64_one_of_thirty_reviewed_audit_rounds_is_inconclusive_not_pass():
    # final review probe: 100 unflagged, sample of 30, one reviewed (correct)
    rows = _unflagged(100, sampled=set(range(1, 31)), reviewed={1})
    m = evaluate(rows, config=_cfg())
    assert m.finish_accuracy_unflagged == 1.0 and m.finish_accuracy_unflagged_method == "audit_sample"
    assert m.finish_accuracy_unflagged_coverage == pytest.approx(1 / 30)
    assert passes(m, _cfg())["finish_accuracy_unflagged"] == "INCONCLUSIVE"


def test_r64_fully_reviewed_five_of_five_is_inconclusive_under_lower_bound_rule():
    rows = _unflagged(100, sampled=set(range(1, 6)), reviewed=set(range(1, 6)))
    m = evaluate(rows, config=_cfg())
    assert m.finish_accuracy_unflagged == 1.0
    assert m.finish_accuracy_unflagged_low == pytest.approx(_wilson(5, 5, 0.95)[0])  # ~0.566
    assert m.finish_accuracy_unflagged_coverage == 1.0
    assert passes(m, _cfg())["finish_accuracy_unflagged"] == "INCONCLUSIVE"


def test_r64_large_clean_fully_reviewed_sample_clears_lower_bound_and_passes():
    rows = _unflagged(200, sampled=set(range(1, 81)), reviewed=set(range(1, 81)))
    m = evaluate(rows, config=_cfg())
    assert m.finish_accuracy_unflagged_low == pytest.approx(_wilson(80, 80, 0.95)[0])
    assert m.finish_accuracy_unflagged_low >= 0.95 and m.n_unflagged_reviewed == 80
    assert passes(m, _cfg())["finish_accuracy_unflagged"] == "PASS"


def test_r64_audit_min_n_comes_from_yaml(tmp_path):
    rows = _unflagged(100, sampled=set(range(1, 6)), reviewed=set(range(1, 6)))
    # target 0.5: Wilson low ~0.566 clears it, so only audit_min_n decides
    low_target = _eval_cfg(tmp_path, accuracy_min=0.5, audit_min_n=5)
    assert passes(evaluate(rows, config=low_target), low_target)["finish_accuracy_unflagged"] == "PASS"
    strict = _eval_cfg(tmp_path, accuracy_min=0.5, audit_min_n=6)
    assert passes(evaluate(rows, config=strict), strict)["finish_accuracy_unflagged"] == "INCONCLUSIVE"


def test_r64_point_estimate_below_target_fails():
    rows = _unflagged(200, sampled=set(range(1, 81)), reviewed=set(range(1, 81)),
                      wrong=set(range(1, 11)))
    m = evaluate(rows, config=_cfg())
    assert passes(m, _cfg())["finish_accuracy_unflagged"] == "FAIL"


def test_r64_exact_basis_passes_on_the_exact_value():
    rows = _unflagged(4, reviewed={1, 2, 3, 4})
    m = evaluate(rows, config=_cfg())
    assert m.finish_accuracy_unflagged_method == "exact"
    assert passes(m, _cfg())["finish_accuracy_unflagged"] == "PASS"


def test_r64_value_without_a_basis_never_passes():
    m = Metrics(0.99, 0.92, 0.91, 0.25, 0.8, 50)  # method defaults to unverified
    assert passes(m)["finish_accuracy_unflagged"] == "INCONCLUSIVE"


@pytest.mark.parametrize("bad", [0, -3, 1.5, True, "30"])
def test_r64_audit_min_n_is_validated(tmp_path, bad):
    with pytest.raises(ValueError, match="audit_min_n"):
        _eval_cfg(tmp_path, audit_min_n=bad)


def test_r64_inconclusive_shown_in_cli_summary_and_report_table(tmp_path, capsys):
    rows = _unflagged(100, sampled=set(range(1, 31)), reviewed={1})
    res = _results(tmp_path / "res", rows, None)
    report = tmp_path / "eval.md"
    assert main(["eval", str(res), "--config", str(EVAL_CONFIG), "--report", str(report)]) == 0
    assert "finish_accuracy_unflagged=1.000 INCONCLUSIVE" in capsys.readouterr().out
    row = next(line for line in report.read_text(encoding="utf-8").splitlines()
               if line.startswith("| 未標覆核回合判定正確率"))
    assert row.endswith("| INCONCLUSIVE (樣本不足) |")
    assert "WARNING: audit sample only partly reviewed" in row
