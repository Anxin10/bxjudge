"""Human corrections: reviewer_fix overrides, audit log, shared CSV/match rebuild."""

import csv
import json

import pytest

from bxjudge.corrections import (
    EDITABLE_FIELDS,
    Correction,
    CorrectionError,
    apply_correction,
    apply_corrections,
    confirm_reviewed,
    load_corrections,
)
from bxjudge.output import effective, read_rounds, write_results
from tests.helpers import rr

AT = "2026-10-21T00:00:00"


def res(video, round, *, finish=None, winner=None, reasons=(), **kw):
    return rr(finish, winner, video=video, round=round, review_reasons=list(reasons), **kw)


@pytest.fixture
def rules():
    from bxjudge.rules import load_rules

    return load_rules()


def snapshot(out):
    log = out / "corrections.jsonl"
    return (
        (out / "rounds.json").read_bytes(),
        (out / "rounds.csv").read_bytes(),
        (out / "matches.csv").read_bytes(),
        log.read_bytes() if log.exists() else None,
    )


def matches(out):
    return list(csv.DictReader((out / "matches.csv").open(encoding="utf-8")))


def test_apply_correction_marks_reviewed(tmp_path, rules):
    write_results([res("a.mp4", 1, finish="spin", winner="X")], tmp_path, rules)
    apply_correction(tmp_path, Correction("a.mp4", 1, "winner", "X", "B", AT))
    row = read_rounds(tmp_path)[0]
    assert row["reviewed"] and effective(row)["winner"] == "B"
    assert len(load_corrections(tmp_path)) == 1


def test_original_prediction_and_diagnostics_are_preserved(tmp_path, rules):
    r = res("a.mp4", 1, finish="spin", winner="X", reasons=["low_confidence"])
    r.confidence, r.needs_review, r.points = 0.42, True, 1
    r.vlm_note = "note"
    r.motion_evidence = {"fps": 60.0, "window": [0, 3]}
    r.detector_provenance = {"kind": "yolo26", "dataset": {"provenance": {}}}
    write_results([r], tmp_path, rules)
    before = read_rounds(tmp_path)[0]
    apply_correction(tmp_path, Correction("a.mp4", 1, "winner", "X", "B", AT))
    row = read_rounds(tmp_path)[0]
    assert row["reviewer_fix"]["winner"] == "B"
    untouched = {k: v for k, v in row.items() if k not in ("reviewed", "reviewer_fix")}
    assert untouched == {k: v for k, v in before.items() if k not in ("reviewed", "reviewer_fix")}
    assert row["winner"] == "X" and row["needs_review"] is True and row["confidence"] == 0.42


def test_old_value_must_be_the_current_effective_value(tmp_path, rules):
    write_results([res("a.mp4", 1, finish="spin", winner="X")], tmp_path, rules)
    before = snapshot(tmp_path)
    with pytest.raises(CorrectionError, match="current"):
        apply_correction(tmp_path, Correction("a.mp4", 1, "winner", "B", "X", AT))
    assert snapshot(tmp_path) == before


def test_repeated_edits_audit_each_change_against_effective_value(tmp_path, rules):
    write_results([res("a.mp4", 1, finish="spin", winner="X")], tmp_path, rules)
    apply_correction(tmp_path, Correction("a.mp4", 1, "winner", "X", "B", AT))
    # the row is already reviewed: a later human correction still works
    apply_correction(tmp_path, Correction("a.mp4", 1, "winner", "B", "X", "2026-10-21T00:01:00"))
    with pytest.raises(CorrectionError, match="current"):  # stale: effective is X again
        apply_correction(tmp_path, Correction("a.mp4", 1, "winner", "B", "X", AT))
    log = load_corrections(tmp_path)
    assert [(c.old, c.new, c.at) for c in log] == [
        ("X", "B", AT),
        ("B", "X", "2026-10-21T00:01:00"),
    ]
    row = read_rounds(tmp_path)[0]
    assert row["reviewed"] and effective(row)["winner"] == "X" and row["winner"] == "X"


def test_no_op_writes_nothing_and_does_not_mark_reviewed(tmp_path, rules):
    write_results([res("a.mp4", 1, finish="spin", winner="X")], tmp_path, rules)
    before = snapshot(tmp_path)
    assert apply_corrections(tmp_path, [Correction("a.mp4", 1, "winner", "X", "X", AT)]) == []
    assert snapshot(tmp_path) == before
    assert not read_rounds(tmp_path)[0]["reviewed"]
    assert load_corrections(tmp_path) == []


def test_batch_audits_only_actual_changes(tmp_path, rules):
    r = res("a.mp4", 1, finish="spin", winner="X")
    r.spin_dir = {"X": "unknown", "B": "right"}
    write_results([r], tmp_path, rules)
    applied = apply_corrections(
        tmp_path,
        [
            Correction("a.mp4", 1, "winner", "X", "X", AT),  # unchanged
            Correction("a.mp4", 1, "x_touch", "no", "yes", AT),
            Correction("a.mp4", 1, "x_spin_dir", "unknown", "left", AT),
        ],
        rules=rules,
    )
    assert [c.field for c in applied] == ["x_touch", "x_spin_dir"]
    assert [c.field for c in load_corrections(tmp_path)] == ["x_touch", "x_spin_dir"]
    row = read_rounds(tmp_path)[0]
    assert row["reviewer_fix"] == {"x_touch": "yes", "x_spin_dir": "left"}


def test_confirm_without_change_only_sets_reviewed(tmp_path, rules):
    r = res("a.mp4", 1, finish="spin", winner="X", reasons=["low_confidence"])
    r.needs_review, r.confidence = True, 0.3
    write_results([r], tmp_path, rules)
    assert confirm_reviewed(tmp_path, "a.mp4", 1, rules=rules) is True
    row = read_rounds(tmp_path)[0]
    assert row["reviewed"] is True and row["reviewer_fix"] is None
    assert row["needs_review"] is True and row["confidence"] == 0.3
    assert load_corrections(tmp_path) == []
    before = snapshot(tmp_path)
    assert confirm_reviewed(tmp_path, "a.mp4", 1, rules=rules) is False  # idempotent
    assert snapshot(tmp_path) == before
    # confirmed rows survive a rerun of the video (R36 keeps reviewed rows)
    write_results([res("a.mp4", 1, finish="over", winner="B")], tmp_path, rules)
    assert effective(read_rounds(tmp_path)[0])["finish"] == "spin"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("points", 3),
        ("confidence", 1.0),
        ("needs_review", False),
        ("reviewed", True),
        ("reviewer_fix", {}),
        ("video", "b.mp4"),
        ("t_launch", 1.0),
        ("nonsense", 1),
        ("x_touch", True),
        ("x_touch", "maybe"),
        ("b_landed", "yes"),
        ("b_landed", 1),
        ("x_spin_dir", "cw"),
        ("air_collision", "unknown"),
        ("finish", "ko"),
        ("finish", ""),
        ("winner", "C"),
        ("winner", "x"),
        ("x_land_zone", "moon"),
        ("b_land_zone", 3),
        ("match_start", 1),
        ("match_start", None),
    ],
)
def test_invalid_field_or_value_rejected_without_writes(tmp_path, rules, field, value):
    write_results([res("a.mp4", 1, finish="spin", winner="X")], tmp_path, rules)
    before = snapshot(tmp_path)
    current = effective(read_rounds(tmp_path)[0]).get(field)
    with pytest.raises(CorrectionError):
        apply_correction(tmp_path, Correction("a.mp4", 1, field, current, value, AT), rules=rules)
    assert snapshot(tmp_path) == before


def test_unknown_round_rejected(tmp_path, rules):
    write_results([res("a.mp4", 1, finish="spin", winner="X")], tmp_path, rules)
    with pytest.raises(CorrectionError, match="no round"):
        apply_correction(tmp_path, Correction("a.mp4", 2, "winner", "X", "B", AT), rules=rules)
    with pytest.raises(CorrectionError, match="no round"):
        confirm_reviewed(tmp_path, "b.mp4", 1, rules=rules)


def test_editable_fields_are_the_brief_fields_plus_r52():
    assert set(EDITABLE_FIELDS) == {
        "x_touch", "b_touch", "x_landed", "b_landed", "x_spin_dir", "b_spin_dir",
        "air_collision", "finish", "winner", "x_land_zone", "b_land_zone", "match_start",
    }


def test_finish_and_winner_must_agree(tmp_path, rules):
    write_results([res("a.mp4", 1, finish="spin", winner="X")], tmp_path, rules)
    before = snapshot(tmp_path)
    with pytest.raises(CorrectionError, match="winner"):
        apply_correction(tmp_path, Correction("a.mp4", 1, "finish", "spin", None, AT), rules=rules)
    with pytest.raises(CorrectionError, match="winner"):
        apply_correction(tmp_path, Correction("a.mp4", 1, "winner", "X", None, AT), rules=rules)
    assert snapshot(tmp_path) == before
    # together in one batch they describe a consistent no-finish (redo) round
    apply_corrections(
        tmp_path,
        [
            Correction("a.mp4", 1, "finish", "spin", None, AT),
            Correction("a.mp4", 1, "winner", "X", None, AT),
        ],
        rules=rules,
    )
    fix = read_rounds(tmp_path)[0]["reviewer_fix"]
    assert fix == {"finish": None, "winner": None, "points": 0}
    assert matches(tmp_path)[0]["x_score"] == "0"


def test_landed_and_land_zone_must_agree(tmp_path, rules):
    write_results([res("a.mp4", 1, finish="spin", winner="X")], tmp_path, rules)
    with pytest.raises(CorrectionError, match="land"):
        # landed=True (helper default) while the first stable zone is the lid
        apply_correction(tmp_path, Correction("a.mp4", 1, "x_land_zone", None, "lid", AT), rules=rules)
    apply_corrections(
        tmp_path,
        [
            Correction("a.mp4", 1, "x_land_zone", None, "lid", AT),
            Correction("a.mp4", 1, "x_landed", True, False, AT),
            Correction("a.mp4", 1, "b_land_zone", None, "battle_area", AT),
        ],
        rules=rules,
    )
    row = read_rounds(tmp_path)[0]
    assert effective(row)["x_land_zone"] == "lid" and effective(row)["b_land_zone"] == "battle_area"
    assert row["x_land_zone"] is None  # original preserved
    # X not landed -> redo, so the corrected land evidence changes the match result
    assert matches(tmp_path)[0]["x_score"] == "0"


def test_points_recomputed_from_rules_after_effective_finish(tmp_path, rules):
    write_results(
        [res("a.mp4", 1, finish="spin", winner="X"), res("b.mp4", 1, finish="spin", winner="X")],
        tmp_path,
        rules,
    )
    assert matches(tmp_path)[0]["x_score"] == "2"
    rules.finish_points["xtreme"] = 5  # points come from the rules passed in, not a constant
    apply_correction(tmp_path, Correction("a.mp4", 1, "finish", "spin", "xtreme", AT), rules=rules)
    row = read_rounds(tmp_path)[0]
    assert row["reviewer_fix"] == {"finish": "xtreme", "points": 5}
    assert effective(row)["points"] == 5 and row["points"] == 0  # original untouched
    assert matches(tmp_path)[0]["x_score"] == "5" and matches(tmp_path)[0]["winner"] == "X"
    exported = next(csv.DictReader((tmp_path / "rounds.csv").open(encoding="utf-8")))
    assert exported["reviewed"] == "True"
    assert json.loads(exported["reviewer_fix"]) == {"finish": "xtreme", "points": 5}


def test_correcting_one_round_keeps_untouched_rounds_of_the_same_video(tmp_path, rules):
    write_results(
        [res("a.mp4", n, finish="spin", winner="B", t_launch=float(n)) for n in (1, 2, 3)],
        tmp_path,
        rules,
    )
    before = read_rounds(tmp_path)
    apply_correction(tmp_path, Correction("a.mp4", 2, "winner", "B", "X", AT), rules=rules)
    after = read_rounds(tmp_path)
    assert [r["round"] for r in after] == [1, 2, 3]
    assert after[0] == before[0] and after[2] == before[2]
    assert not after[0]["reviewed"] and after[1]["reviewed"]


@pytest.mark.parametrize("reason", ["recalibrate", "no_round", "decode_error", "analysis_error"])
def test_diagnostic_rows_reject_corrections_and_confirmation(tmp_path, rules, reason):
    write_results(
        [res("a.mp4", 1, reasons=[reason], launch=False), res("b.mp4", 1, finish="spin", winner="X")],
        tmp_path,
        rules,
    )
    before = snapshot(tmp_path)
    with pytest.raises(CorrectionError, match="not a round"):
        apply_correction(tmp_path, Correction("a.mp4", 1, "winner", None, "X", AT), rules=rules)
    with pytest.raises(CorrectionError, match="not a round"):
        apply_correction(tmp_path, Correction("a.mp4", 1, "match_start", False, True, AT), rules=rules)
    with pytest.raises(CorrectionError, match="not a round"):
        confirm_reviewed(tmp_path, "a.mp4", 1, rules=rules)
    assert snapshot(tmp_path) == before
    assert not read_rounds(tmp_path)[0]["reviewed"]
    # the real round next to it remains fully editable
    apply_correction(tmp_path, Correction("b.mp4", 1, "winner", "X", "B", AT), rules=rules)
    assert effective(read_rounds(tmp_path)[1])["winner"] == "B"


def test_launch_fields_need_a_recorded_launch_check(tmp_path, rules):
    write_results([res("a.mp4", 1, finish="spin", winner="X", launch=False)], tmp_path, rules)
    with pytest.raises(CorrectionError, match="launch"):
        apply_correction(tmp_path, Correction("a.mp4", 1, "x_touch", "unknown", "yes", AT), rules=rules)
    with pytest.raises(CorrectionError, match="spin"):
        apply_correction(tmp_path, Correction("a.mp4", 1, "x_spin_dir", "unknown", "left", AT), rules=rules)


# ----------------------------------------------------------------------------- R52


def four_rounds(tmp_path, rules):
    write_results(
        [
            res("a.mp4", 1, finish="spin", winner="X"),
            res("b.mp4", 1, finish="spin", winner="X"),
            res("c.mp4", 1, finish="spin", winner="B"),
            res("d.mp4", 1, finish="spin", winner="B"),
        ],
        tmp_path,
        rules,
    )


def test_manual_match_start_splits_and_removing_it_restores_grouping(tmp_path, rules):
    four_rounds(tmp_path, rules)
    assert [(m["x_score"], m["b_score"]) for m in matches(tmp_path)] == [("2", "2")]
    apply_correction(tmp_path, Correction("c.mp4", 1, "match_start", False, True, AT), rules=rules)
    row = read_rounds(tmp_path)[2]
    assert row["reviewer_fix"] == {"match_start": True} and "match_start" not in {
        k for k in row if k != "reviewer_fix"
    }
    got = matches(tmp_path)
    assert [(m["x_score"], m["b_score"], m["winner"]) for m in got] == [
        ("2", "0", ""),
        ("0", "2", ""),
    ]
    assert [r["video"] for r in json.loads(got[1]["rounds"])] == ["c.mp4", "d.mp4"]
    # CSV schema unchanged
    header = next(csv.reader((tmp_path / "rounds.csv").open(encoding="utf-8")))
    assert "match_start" not in header
    apply_correction(tmp_path, Correction("c.mp4", 1, "match_start", True, False, AT), rules=rules)
    assert [(m["x_score"], m["b_score"]) for m in matches(tmp_path)] == [("2", "2")]
    assert [c.field for c in load_corrections(tmp_path)] == ["match_start", "match_start"]


def test_land_zone_and_boundary_corrections_persist_across_reruns(tmp_path, rules):
    four_rounds(tmp_path, rules)
    apply_corrections(
        tmp_path,
        [
            Correction("c.mp4", 1, "match_start", False, True, AT),
            Correction("c.mp4", 1, "b_land_zone", None, "battle_area", AT),
        ],
        rules=rules,
    )
    # a later pipeline rerun of the same video must not drop the reviewed row
    write_results([res("c.mp4", 1, finish="over", winner="X")], tmp_path, rules)
    row = read_rounds(tmp_path)[2]
    assert effective(row)["b_land_zone"] == "battle_area" and effective(row)["match_start"] is True
    assert len(matches(tmp_path)) == 2


def test_load_corrections_round_trip_and_empty(tmp_path, rules):
    assert load_corrections(tmp_path) == []
    write_results([res("a.mp4", 1, finish="spin", winner="X")], tmp_path, rules)
    c = Correction("a.mp4", 1, "x_land_zone", None, "battle_area", AT)
    apply_corrections(tmp_path, [c, Correction("a.mp4", 1, "x_landed", True, True, AT)], rules=rules)
    assert load_corrections(tmp_path) == [c]
    line = json.loads((tmp_path / "corrections.jsonl").read_text(encoding="utf-8"))
    assert line == {
        "video": "a.mp4", "round": 1, "field": "x_land_zone",
        "old": None, "new": "battle_area", "at": AT,
    }
