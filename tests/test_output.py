"""Authoritative merge, review overrides, exports and trajectory evidence."""

import csv
import json

import cv2
import pytest

from bxjudge import output
from bxjudge.calibration_store import unveto_calibration, veto_calibration
from bxjudge.corrections import confirm_reviewed
from tests.helpers import (
    auto_snapshot_ids,
    cal_with,
    make_ref_img,
    make_track,
    read_matches,
    results_with_two_positions,
    rr,
    shipped_autocalib,
)

FIELDS = [
    "video",
    "round",
    "t_launch",
    "t_end",
    "x_touch",
    "b_touch",
    "x_landed",
    "b_landed",
    "x_land_zone",
    "b_land_zone",
    "x_spin_dir",
    "b_spin_dir",
    "air_collision",
    "finish",
    "winner",
    "points",
    "confidence",
    "needs_review",
    "review_reasons",
    "vlm_note",
    "reviewed",
    "reviewer_fix",
]


def test_merge_removes_obsolete_unreviewed_preserves_every_reviewed(tmp_path, rules):
    output.write_results([rr("spin", "X", round=n) for n in (1, 2, 3)], tmp_path, rules)
    path = tmp_path / "rounds.json"
    rows = json.loads(path.read_text())
    rows[0]["reviewed"] = True
    rows[0]["reviewer_fix"] = {"finish": "xtreme", "winner": "B", "points": 3}
    rows[2]["reviewed"] = True
    path.write_text(json.dumps(rows))
    output.write_results(
        [rr("over", "X", round=1), rr("spin", "B", round=4)], tmp_path, rules
    )
    merged = json.loads(path.read_text())
    assert [r["round"] for r in merged] == [1, 3, 4]
    assert merged[0] == rows[0] and merged[1] == rows[2]
    assert output.effective(merged[0])["finish"] == "xtreme"
    assert merged[0]["finish"] == "spin"
    matches = list(csv.DictReader((tmp_path / "matches.csv").open()))
    assert matches[0]["b_score"] == "4"
    assert list(csv.DictReader((tmp_path / "rounds.csv").open()).fieldnames) == FIELDS


def test_roundtrip_and_effective_flat_launch_fields(tmp_path, rules):
    r = rr("spin", "X")
    r.spin_dir = {"X": "right", "B": "left"}
    output.write_results([r], tmp_path, rules)
    row = json.loads((tmp_path / "rounds.json").read_text())[0]
    assert output.row_to_result(row) == r
    row["reviewer_fix"] = {"x_touch": "yes", "b_landed": False, "x_spin_dir": "left"}
    fixed = output.row_to_result(output.effective(row))
    assert fixed.launch.touch["X"] == "yes" and not fixed.launch.landed["B"]
    assert fixed.spin_dir["X"] == "left"


def test_duplicate_new_keys_rejected_before_writes(tmp_path, rules):
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        output.write_results([rr("spin", "X"), rr("over", "B")], tmp_path, rules)
    assert not (tmp_path / "rounds.json").exists()


def test_video_timing_updates_one_key(tmp_path):
    output.record_video_timing(tmp_path, "a.mp4", 10, 2, produced_rounds={1})
    output.record_video_timing(tmp_path, "b.mp4", 20, 3, produced_rounds={1})
    output.record_video_timing(tmp_path, "a.mp4", 10, 1, produced_rounds={1})
    data = json.loads((tmp_path / "videos.json").read_text())
    assert len(data) == 2 and data["a.mp4"] == {
        "duration_s": 10, "processing_s": 1,
        "judgment_sources": [{"rounds": [1], "run_id": None, "wall_s": None, "processing_s": 1}],
    }


def test_trajectory_all_zones_source_indices_and_scaled_reference(
    tmp_path, monkeypatch
):
    c = cal_with(make_ref_img(), tmp_path)
    # Calibration is at 640x360; tracks in its pixels.
    bey = {
        "X": make_track(1, "bey", {f: (300, 100) for f in (30, 31)}),
        "B": make_track(2, "bey", {f: (320, 320) for f in (30, 31)}),
    }
    parts = [make_track(3, "bey_part", {31: (300, 100)})]
    seen = []
    original = cv2.polylines

    def polygons(img, points, *args, **kwargs):
        seen.append(points)
        return original(img, points, *args, **kwargs)

    monkeypatch.setattr(cv2, "polylines", polygons)
    csv_p, png_p = output.write_trajectory(
        tmp_path,
        rr("spin", "X", video="a.mp4"),
        bey,
        parts,
        c,
        30,
        {"landing_X": 30, "landing_B": 31, "air_collision": None, "finish": 31},
    )
    rows = list(csv.DictReader(csv_p.open()))
    assert len(rows) == 5 and rows[0]["frame"] == "30" and rows[0]["t"] == "1.0"
    assert {r["zone"] for r in rows if r["side"] == "X"} == {"battle_area"}
    assert {r["zone"] for r in rows if r["side"] == "B"} == {"xtreme"}
    assert rows[-1]["side"] == ""
    assert len(seen) >= 8 and cv2.imread(str(png_p)).shape == (446, 640, 3)


def test_trajectory_missing_observation_is_explicit(tmp_path):
    c = cal_with(make_ref_img(), tmp_path)
    bey = {"X": make_track(1, "bey", {30: (300, 100), 32: (310, 100)})}
    path, _ = output.write_trajectory(tmp_path, rr("spin", "X"), bey, [], c, 30, {})
    rows = list(csv.DictReader(path.open()))
    assert [r["frame"] for r in rows] == ["30", "31", "32"]
    assert rows[1]["x"] == rows[1]["y"] == rows[1]["score"] == ""
    assert rows[1]["zone"] == "unknown"


def test_unassigned_bey_trajectory_preserves_uncertain_tracks(tmp_path):
    c = cal_with(make_ref_img(), tmp_path)
    unknown = make_track(9, "bey", {0: (300, 100), 2: (310, 100)})
    path, _ = output.write_trajectory(
        tmp_path, rr("spin", "X"), {}, [], c, 30, {}, unassigned=[unknown]
    )
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 3 and {r["side"] for r in rows} == {""}
    assert {r["cls"] for r in rows} == {"bey"}
    assert {r["track_id"] for r in rows} == {"9"}


def test_trajectory_labels_stay_inside_image(tmp_path, monkeypatch):
    c = cal_with(make_ref_img(), tmp_path)
    positions = []
    original = cv2.putText

    def text(image, label, position, font, size, color, thickness, *args):
        (width, height), baseline = cv2.getTextSize(label, font, size, thickness)
        positions.append((position, width, height, baseline))
        return original(image, label, position, font, size, color, thickness, *args)

    monkeypatch.setattr(cv2, "putText", text)
    _, png = output.write_trajectory(
        tmp_path,
        rr("spin", "X"),
        {"X": make_track(1, "bey", {0: (638, 358)})},
        [],
        c,
        30,
        {"finish": 0},
    )
    assert all(
        x >= 0 and y >= height and x + width <= 640 and y + baseline <= cv2.imread(str(png)).shape[0]
        for (x, y), width, height, baseline in positions
    )


def test_burst_finish_marker_uses_trailing_part_observation(tmp_path, monkeypatch):
    c = cal_with(make_ref_img(), tmp_path)
    markers = []
    original = cv2.drawMarker

    def marker(image, point, *args, **kwargs):
        markers.append(point)
        return original(image, point, *args, **kwargs)

    monkeypatch.setattr(cv2, "drawMarker", marker)
    bey = {
        "X": make_track(1, "bey", {f: (200, 180) for f in range(11)}),
        "B": make_track(2, "bey", {f: (450, 180) for f in range(11)}),
    }
    parts = [make_track(3, "bey_part", {11: (205, 180)})]
    output.write_trajectory(
        tmp_path, rr("burst", "B"), bey, parts, c, 30, {"finish": 11}
    )
    assert (205, 180) in markers


def test_vlm_timing_is_separate_and_accumulates(tmp_path):
    output.record_video_timing(tmp_path, "a.mp4", 10, 2, source_video=tmp_path / "a.mp4", produced_rounds={1})
    first = {"round": 1, "seconds": 30.0, "model": "m", "ok": True, "cold": True}
    second = {"round": 2, "seconds": 1.0, "model": None, "ok": False, "cold": None}
    output.record_vlm_timing(tmp_path, "a.mp4", 30.5, [first])
    output.record_vlm_timing(tmp_path, "a.mp4", 1.5, [second])
    data = json.loads((tmp_path / "videos.json").read_text())
    assert data["a.mp4"]["processing_s"] == 2 and data["a.mp4"]["vlm_s"] == 32.0
    # per-request seconds accumulate next to the phase total (R60)
    assert data["a.mp4"]["vlm_requests"] == [first, second]
    assert data["a.mp4"]["source_video"] == str(tmp_path / "a.mp4")
    # A fresh core analysis replaces the rounds, so earlier VLM time no longer applies.
    output.record_video_timing(tmp_path, "a.mp4", 10, 3, produced_rounds={1})
    entry = json.loads((tmp_path / "videos.json").read_text())["a.mp4"]
    assert "vlm_s" not in entry and "vlm_requests" not in entry


@pytest.mark.parametrize("bad", [
    {"round": 1, "seconds": -1.0, "model": "m", "ok": True, "cold": True},
    {"round": 1, "seconds": float("nan"), "model": "m", "ok": True, "cold": True},
    {"round": 1, "seconds": 1.0, "model": "m", "ok": True},
    {"round": True, "seconds": 1.0, "model": "m", "ok": True, "cold": True},
])
def test_vlm_request_records_are_validated(tmp_path, bad):
    with pytest.raises(ValueError):
        output.record_vlm_timing(tmp_path, "a.mp4", 1.0, [bad])
    assert not (tmp_path / "videos.json").exists()


def test_match_start_is_sparse_typed_metadata(tmp_path, rules):
    import pytest

    output.write_results([rr("spin", "X")], tmp_path, rules)
    row = json.loads((tmp_path / "rounds.json").read_text())[0]
    assert "match_start" not in row  # absent = automatic grouping
    assert output.row_to_result(row).match_start is False
    row["reviewer_fix"] = {"match_start": True}
    assert output.row_to_result(output.effective(row)).match_start is True
    row["reviewer_fix"] = {"match_start": "yes"}
    with pytest.raises(ValueError, match="match_start"):
        output.row_to_result(output.effective(row))


def test_read_rounds_and_save_rounds_rebuild_exports(tmp_path, rules):
    import pytest

    assert output.read_rounds(tmp_path) == []
    output.write_results([rr("spin", "X"), rr("spin", "B", video="v2.mp4")], tmp_path, rules)
    rows = output.read_rounds(tmp_path)
    rows[1]["reviewer_fix"] = {"winner": "X", "points": 1}
    rows[1]["reviewed"] = True
    output.save_rounds(rows, tmp_path, rules)
    matches = list(csv.DictReader((tmp_path / "matches.csv").open()))
    assert matches[0]["x_score"] == "2"
    assert output.read_rounds(tmp_path) == rows
    with pytest.raises(ValueError, match="duplicate"):
        output.save_rounds([rows[0], rows[0]], tmp_path, rules)


def test_trajectory_png_path_matches_writer(tmp_path):
    r = rr("spin", "X", video="nested/a.mp4", round=2)
    assert output.trajectory_png(tmp_path, r.video, r.round) == (
        tmp_path / "trajectories" / "nested__a_r2.png"
    )


# ---- calibration vetoes (auto-calibration Task 4) -----------------------------------------

S1, S2 = auto_snapshot_ids()


def test_veto_excludes_unreviewed_rounds_from_matches_only(tmp_path):
    out = results_with_two_positions(tmp_path)  # videos a.mp4 (snapshot S1) and b.mp4 (S2), 2 rounds each
    veto_calibration(out, S1, autocalib=shipped_autocalib())
    matches = read_matches(out)
    assert all(r["video"] == "b.mp4" for m in matches for r in m["rounds"])
    assert len(json.loads((out / "rounds.json").read_text())) == 4
    unveto_calibration(out, S1, autocalib=shipped_autocalib())
    assert {r["video"] for m in read_matches(out) for r in m["rounds"]} == {"a.mp4", "b.mp4"}


def test_reviewed_rows_under_veto_still_score(tmp_path):
    out = results_with_two_positions(tmp_path); confirm_reviewed(out, "a.mp4", 1); veto_calibration(out, S1, autocalib=shipped_autocalib())
    assert ("a.mp4", 1) in {(r["video"], r["round"]) for m in read_matches(out) for r in m["rounds"]}


def test_veto_keeps_rows_byte_identical_and_every_rebuild_honours_it(tmp_path, rules):
    out = results_with_two_positions(tmp_path)
    before_rows = (out / "rounds.json").read_bytes()
    before_csv = (out / "rounds.csv").read_bytes()
    veto_calibration(out, S1, note="zones off by a pocket", autocalib=shipped_autocalib())
    assert (out / "rounds.json").read_bytes() == before_rows  # vetoes never rewrite judgments
    assert (out / "rounds.csv").read_bytes() == before_csv
    # A later export through the shared path (judge writer / corrections) still honours it.
    output.save_rounds(output.read_rounds(out), out, rules)
    assert {r["video"] for m in read_matches(out) for r in m["rounds"]} == {"b.mp4"}
    confirm_reviewed(out, "b.mp4", 2)
    assert {r["video"] for m in read_matches(out) for r in m["rounds"]} == {"b.mp4"}


def test_matches_for_excludes_only_unreviewed_vetoed_rows(rules):
    rows = [output._result_row(rr("spin", "X", video="a.mp4", round=n)) for n in (1, 2, 3)]
    for r in rows:
        r["calibration_id"] = "S1"
    rows[1]["reviewed"] = True
    scored = [(r.video, r.round) for m in output.matches_for(rows, rules, vetoed={"S1"})
              for r, _ in m.rounds]
    assert scored == [("a.mp4", 2)]
    assert len([r for m in output.matches_for(rows, rules) for r in m.rounds]) == 3


@pytest.mark.parametrize("payload", ["{}", "null", "\"x\"", "3"])
def test_read_rounds_rejects_a_non_list_rounds_json(tmp_path, payload):
    (tmp_path / "rounds.json").write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="rounds.json must be a list of rows"):
        output.read_rounds(tmp_path)


def test_vlm_timing_run_ids_accumulate_and_a_new_analysis_resets_them(tmp_path, rules):
    """C-X4(b): each VLM phase's run id is kept beside the time it added."""
    output.write_results([rr("spin", "X", video="a.mp4")], tmp_path, rules)
    output.record_video_timing(tmp_path, "a.mp4", 10.0, 2.0, wall_s=2.0, run_id="r1",
                               produced_rounds={1})
    output.record_vlm_timing(tmp_path, "a.mp4", 1.0, [], run_id="r1")
    output.record_vlm_timing(tmp_path, "a.mp4", 1.5, [], run_id="v2")
    e = json.loads((tmp_path / "videos.json").read_text())["a.mp4"]
    assert e["vlm_run_ids"] == ["r1", "v2"] and e["judgment_sources"][0]["run_id"] == e["attempt_run_id"] == "r1"
    output.record_video_status(tmp_path, "a.mp4", "analysis_error", wall_s=0.5, run_id="r3")
    e = json.loads((tmp_path / "videos.json").read_text())["a.mp4"]
    assert (e["judgment_sources"][0]["run_id"], e["attempt_run_id"]) == ("r1", "r3")
    output.record_video_timing(tmp_path, "a.mp4", 10.0, 2.0, wall_s=2.0, run_id="r4",
                               produced_rounds={1})
    e = json.loads((tmp_path / "videos.json").read_text())["a.mp4"]
    assert "vlm_run_ids" not in e and e["judgment_sources"][0]["run_id"] == "r4"
    with pytest.raises(ValueError, match="run_id"):
        output.record_video_timing(tmp_path, "a.mp4", 10.0, 2.0, wall_s=2.0, run_id="", produced_rounds={1})


def test_trajectory_side_colors_and_observed_endpoints(tmp_path):
    c = cal_with(make_ref_img(), tmp_path)
    bey = {
        "X": make_track(1, "bey", {30: (100, 150), 31: (120, 150), 32: (140, 150)}),
        "B": make_track(2, "bey", {30: (100, 230), 31: (120, 230), 32: (140, 230)}),
    }
    unknown = make_track(9, "bey", {30: (100, 280), 31: (120, 280), 32: (140, 280)})
    csv_p, png_p = output.write_trajectory(
        tmp_path, rr("spin", "X"), bey, [], c, 30, {}, unassigned=[unknown]
    )
    image = cv2.imread(str(png_p))
    orange, blue, gray = (0, 159, 230), (178, 114, 0), (150, 150, 150)
    for y, color in [(150, orange), (230, blue), (280, gray)]:
        assert tuple(image[y, 120]) == color  # actual contiguous trajectory
        assert tuple(image[y + 7, 100]) == color  # start circle
        assert tuple(image[y + 3, 143]) == color  # end tilted cross
    rows = list(csv.DictReader(csv_p.open()))
    assert {row["side"] for row in rows if row["track_id"] == "9"} == {""}
    assert {row["side"] for row in rows if row["track_id"] == "1"} == {"X"}


def test_trajectory_endpoint_labels_use_frame_order_and_show_unknown_identity(tmp_path, monkeypatch):
    c = cal_with(make_ref_img(), tmp_path)
    labels = []
    original = cv2.putText
    def text(image, label, *args, **kwargs):
        labels.append(label)
        return original(image, label, *args, **kwargs)
    monkeypatch.setattr(cv2, "putText", text)
    unordered = make_track(9, "bey", {35: (300, 180), 30: (200, 180)})
    output.write_trajectory(tmp_path, rr(None, None), {}, [], c, 30, {}, unassigned=[unordered])
    assert "? #9 start f30" in labels
    assert "? #9 end f35" in labels
    assert "X side" in labels and "B side" in labels
    assert "Unknown side / fragments" in labels


def test_trajectory_legend_does_not_cover_observed_evidence(tmp_path):
    c = cal_with(make_ref_img(), tmp_path)
    bey = {"X": make_track(1, "bey", {0: (30, 20), 1: (160, 20)})}
    part = make_track(2, "bey_part", {1: (250, 40)})
    _, png = output.write_trajectory(tmp_path, rr("burst", "X"), bey, [part], c, 30, {"finish": 1})
    image = cv2.imread(str(png))
    assert tuple(image[20, 100]) == (0, 159, 230)
    assert tuple(image[40, 250]) == (0, 255, 255)
    assert image.shape == (446, 640, 3)  # footer outside the original 360px evidence frame
