import json

import pytest

from bxjudge import judge_view as jv


def test_stage_of_maps_known_reasons_and_keeps_unknown_visible():
    assert jv.stage_of("two_bey_singleton_segment") == "track"
    assert jv.stage_of("track_missing@X") == "track"
    assert jv.stage_of("calibration_geometry_missing") == "geometry"
    assert jv.stage_of("identity_unanchored") == "identity"
    assert jv.stage_of("X_stop_reversed") == "finish"
    assert jv.stage_of("after_finish:id_swap_risk") == "finish"
    assert jv.stage_of("finish_candidate_voided:spin:B:2017") == "finish"
    assert jv.stage_of("synthetic_detector") == "detect"
    assert jv.stage_of("brand_new_code") == "other"


def test_stage_states_flags_only_stages_with_reasons():
    states = {
        s["id"]: s
        for s in jv.stage_states(["track_missing@X", "calibration_low_score"])
    }
    assert states["track"]["status"] == "flag"
    assert states["geometry"]["reasons"] == ["calibration_low_score"]
    assert states["finish"]["status"] == "ok"
    assert "other" not in states
    assert jv.stage_states(["weird"])[-1] == {
        "id": "other",
        "label": "其他",
        "status": "flag",
        "reasons": ["weird"],
    }


def row(**kw):
    base = {
        "video": "v.mp4",
        "round": 1,
        "t_launch": 1.0,
        "t_end": 3.0,
        "finish": "spin",
        "winner": "X",
        "points": 1,
        "confidence": 0.9,
        "needs_review": False,
        "review_reasons": [],
        "reviewed": False,
        "reviewer_fix": None,
        "geometry_evidence": None,
        "calibration_id": None,
    }
    return base | kw


def test_round_summary_judged_vs_review():
    judged = jv.round_summary(row(), 0.7)
    assert judged["state"] == "judged" and judged["above_threshold"]
    assert judged["finish_label"] == "耐久"
    review = jv.round_summary(
        row(
            needs_review=True,
            confidence=0.2,
            finish=None,
            review_reasons=["identity_unanchored"],
        ),
        0.7,
    )
    assert review["state"] == "review" and not review["above_threshold"]
    assert review["finish_label"] == "未判定" and review["reasons"] == [
        "identity_unanchored"
    ]


def write_csv(path, rows):
    path.write_text(
        "frame,t,side,track_id,cls,x,y,zone,score\n"
        + "".join(",".join(map(str, r)) + "\n" for r in rows)
    )


def test_pack_trajectory_groups_by_side_and_handles_header_only(tmp_path):
    empty = tmp_path / "e.csv"
    write_csv(empty, [])
    assert jv.pack_trajectory(empty) == ({}, {})
    assert jv.pack_trajectory(tmp_path / "missing.csv") == ({}, {})
    path = tmp_path / "t.csv"
    write_csv(
        path,
        [
            (30, 1.0, "X", 5, "bey", 10, 20, "battle", 0.9),
            (30, 1.0, "B", 6, "bey", 50, 60, "battle", 0.8),
            (31, 1.0333, "X", 5, "bey", 11, 21, "battle", 0.9),
            (32, 1.0667, "X", 5, "bey", "", "", "unknown", ""),
        ],
    )
    sides, times = jv.pack_trajectory(path)
    assert [p[1] for p in sides["X"]] == [10.0, 11.0] and len(sides["B"]) == 1
    assert times == {30: 1.0, 31: 1.0333}
    sides, _ = jv.pack_trajectory(path, max_points=1)
    assert len(sides["X"]) == 1


def test_stadium_zones_missing_unaligned_and_warped():
    assert jv.stadium_zones(None, None, 0)["status"] == "missing"
    calib = {"stadium": [[0, 0], [10, 0], [10, 10]]}
    assert jv.stadium_zones(calib, {"samples": {}}, 0)["status"] == "unavailable"
    shift = [[1, 0, 5], [0, 1, -2], [0, 0, 1]]
    geometry = {
        "samples": {
            0: {"frame": 0, "status": "tracking", "alignment": None},
            100: {
                "frame": 100,
                "status": "tracking",
                "alignment": {"homography": shift},
            },
        }
    }
    out = jv.stadium_zones(calib, geometry, 100)
    assert out["status"] == "launch_reference" and out["sample_frame"] == 100
    assert out["zones"]["stadium"][1] == [15.0, -2.0]


def test_round_boxes_marks_primary_side_and_candidates():
    two_bey = {
        "rounds": {
            "1": {
                "members": {"5": [1]},
                "identity_unknown_since": {},
                "identity_unknown_intervals": {},
                "observations": [
                    {
                        "frame": 30,
                        "box": [0, 0, 20, 30],
                        "score": 0.9,
                        "entity_id": 1,
                        "identity_valid": True,
                    }
                ],
                "unassigned": [
                    {
                        "frame": 30,
                        "box": [40, 40, 60, 60],
                        "score": 0.4,
                        "reason": "duplicate_box",
                    }
                ],
            }
        }
    }
    boxes = jv.round_boxes(two_bey, 1, {30: 1.0}, 30.0, {"5": "X"})
    assert boxes[0][:2] == [1.0, 30]
    kinds = {b[5]: b[6] for b in boxes[0][2]}
    assert kinds == {"primary": "X", "candidate": "duplicate_box"}
    assert jv.round_boxes(two_bey, 2, {}, 30.0, {}) == []


@pytest.fixture
def run(tmp_path):
    run = tmp_path / "run"
    (run / "trajectories").mkdir(parents=True)
    rows = [
        row(),
        row(
            round=2,
            needs_review=True,
            confidence=0.1,
            review_reasons=["calibration_geometry_missing", "track_missing@X"],
        ),
    ]
    (run / "rounds.json").write_text(json.dumps(rows))
    (run / "videos.json").write_text(
        json.dumps(
            {
                "v.mp4": {
                    "source_video": "/videos/v.mp4",
                    "calibration": {"id": "c1", "kind": "auto", "score": 0.8},
                }
            }
        )
    )
    write_csv(
        run / "trajectories" / "v_r1.csv",
        [(30, 1.0, "X", 5, "bey", 10, 20, "battle", 0.9)],
    )
    return run


def test_build_view_reads_run_without_writing(run, tmp_path):
    before = sorted((p.name, p.stat().st_mtime_ns) for p in run.rglob("*"))
    store = tmp_path / "store"
    store.mkdir()
    (store / "reid-a.json").write_text("{}")
    view = jv.build_view(run, 0.7, {"reid-review": store, "absent": tmp_path / "nope"})
    assert sorted((p.name, p.stat().st_mtime_ns) for p in run.rglob("*")) == before
    index = view["index"]
    assert index["round_count"] == 2 and index["review_count"] == 1
    assert index["rounds"][0]["key"] == "v_r2"  # review rounds first
    assert index["stores"] == {"reid-review": 1, "absent": 0}
    assert view["videos"] == {"v.mp4": "/videos/v.mp4"}
    r2 = view["rounds"]["v_r2"]
    assert r2["src"] == "/video/v.mp4" and r2["geometry"]["status"] == "missing"
    assert {s["id"] for s in r2["stages"] if s["status"] == "flag"} == {
        "geometry",
        "track",
    }
    assert view["rounds"]["v_r1"]["tracks"]["X"][0][1] == 10.0


def test_media_offset_reads_stream_start_and_degrades_to_zero(tmp_path, monkeypatch):
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")

    class Done:
        stdout = json.dumps(
            {
                "streams": [{"start_time": "0.939000"}],
                "format": {"start_time": "0.000000"},
            }
        )

    monkeypatch.setattr(jv.subprocess, "run", lambda *a, **k: Done)
    assert jv.media_offset(str(clip)) == 0.939
    assert (
        jv.media_offset(None) == 0.0
        and jv.media_offset(str(tmp_path / "gone.mp4")) == 0.0
    )

    def fail(*a, **k):
        raise jv.subprocess.CalledProcessError(1, "ffprobe")

    monkeypatch.setattr(jv.subprocess, "run", fail)
    assert jv.media_offset(str(clip)) == 0.0
