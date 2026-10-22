"""Authoritative merge, review overrides, exports and trajectory evidence."""

import csv
import json

import cv2

from bxjudge import output
from tests.helpers import cal_with, make_ref_img, make_track, rr

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
    output.record_video_timing(tmp_path, "a.mp4", 10, 2)
    output.record_video_timing(tmp_path, "b.mp4", 20, 3)
    output.record_video_timing(tmp_path, "a.mp4", 10, 1)
    data = json.loads((tmp_path / "videos.json").read_text())
    assert len(data) == 2 and data["a.mp4"] == {"duration_s": 10, "processing_s": 1}


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
    assert len(seen) >= 8 and cv2.imread(str(png_p)).shape == (360, 640, 3)


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
    output.write_trajectory(
        tmp_path,
        rr("spin", "X"),
        {"X": make_track(1, "bey", {0: (638, 358)})},
        [],
        c,
        30,
        {"finish": 0},
    )
    assert all(
        x >= 0 and y >= height and x + width <= 640 and y + baseline <= 360
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
