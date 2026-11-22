"""HUD evidence must retain entity identity, unknowns and source geometry."""

import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bxjudge import judge_view as jv
from bxjudge.dynamic_calibration import save_geometry
from tests.helpers import make_calibration


def observations(*, valid=True, intervals=None):
    return {
        "rounds": {
            "1": {
                "members": {"101": [1], "102": [2]},
                "identity_unknown_since": {},
                "identity_unknown_intervals": intervals or {},
                "observations": [
                    {
                        "frame": 10,
                        "entity_id": 1,
                        "box": [0, 0, 20, 20],
                        "score": 0.9,
                        "identity_valid": True,
                    },
                    {
                        "frame": 10,
                        "entity_id": 2,
                        "box": [5, 0, 25, 20],
                        "score": 0.9,
                        "identity_valid": valid,
                    },
                ],
                "unassigned": [],
            }
        }
    }


def test_overlapping_boxes_use_saved_entity_ownership_not_containment():
    rows = jv.round_boxes(observations(), 1, {10: 1.0}, 10, {"101": "X", "102": "B"})
    assert [b[6] for b in rows[0][2]] == ["X", "B"]


@pytest.mark.parametrize(
    "data",
    [
        observations(valid=False),
        observations(intervals={"102": [[10, 12]]}),
    ],
)
def test_unknown_observations_never_borrow_the_other_players_label(data):
    rows = jv.round_boxes(data, 1, {10: 1.0}, 10, {"101": "X", "102": "B"})
    assert rows[0][2][1][6] == "#102 · 身分未知"


@pytest.mark.parametrize("status", ["bridged", "lost", "temporal_pending"])
def test_unverified_launch_geometry_is_hidden(status):
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    geometry = {
        "samples": {10: {"frame": 10, "status": status, "alignment": {"homography": h}}}
    }
    got = jv.stadium_zones({"stadium": [[0, 0], [10, 0], [10, 10]]}, geometry, 10)
    assert got["zones"] == {}
    assert got["status"] == "unavailable"


def test_missing_launch_geometry_does_not_borrow_a_far_future_sample():
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    geometry = {
        "samples": {
            1000: {"frame": 1000, "status": "tracking", "alignment": {"homography": h}}
        }
    }
    got = jv.stadium_zones({"stadium": [[0, 0], [10, 0], [10, 10]]}, geometry, 10)
    assert got["zones"] == {}
    assert got["status"] == "unavailable"


def test_unknown_traces_keep_entity_and_gap_segments_before_decimation(tmp_path):
    path = tmp_path / "trace.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["frame", "t", "side", "track_id", "cls", "x", "y", "zone", "score"]
        )
        writer.writerows(
            [
                [0, 0, "", 101, "bey", 10, 10, "battle_area", 0.9],
                [0, 0, "", 102, "bey", 100, 10, "battle_area", 0.9],
                [1, 0.1, "", 101, "bey", 11, 10, "battle_area", 0.9],
                [1, 0.1, "", 102, "bey", 101, 10, "battle_area", 0.9],
                [4, 0.4, "", 101, "bey", 14, 10, "battle_area", 0.9],
            ]
        )
    traces, _ = jv.pack_trajectory(path)
    by_entity = {tid: [p for p in traces[""] if p[4] == tid] for tid in [101, 102]}
    assert by_entity[101][0][7] == by_entity[101][1][7]
    assert by_entity[101][-1][7] != by_entity[101][0][7]
    assert [p[6] for p in by_entity[101]] == [0, 1, 4]
    thinned, _ = jv.pack_trajectory(path, max_points=2)
    assert all(len(p) == 8 for p in thinned[""])


def test_decimation_does_not_drop_one_of_two_interleaved_unknown_entities(tmp_path):
    path = tmp_path / "long.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["frame", "t", "side", "track_id", "cls", "x", "y", "zone", "score"]
        )
        for frame in range(4000):
            for entity in [101, 102]:
                writer.writerow(
                    [
                        frame,
                        frame / 30,
                        "",
                        entity,
                        "bey",
                        entity + frame % 10,
                        10,
                        "battle_area",
                        0.9,
                    ]
                )
    traces, _ = jv.pack_trajectory(path)
    assert {
        entity: sum(p[4] == entity for p in traces[""]) for entity in [101, 102]
    } == {101: 4000, 102: 4000}


def test_decimation_keeps_each_entity_segment_and_its_endpoints(tmp_path):
    path = tmp_path / "segments.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["frame", "t", "side", "track_id", "cls", "x", "y", "zone", "score"]
        )
        for frame in [*range(10), *range(20, 30)]:
            writer.writerow(
                [frame, frame / 30, "X", 101, "bey", frame, 10, "battle_area", 0.9]
            )
    traces, _ = jv.pack_trajectory(path, max_points=2)
    assert [p[6] for p in traces["X"]] == [0, 9, 20, 29]
    assert len({p[7] for p in traces["X"]}) == 2


def geometry_record(calibration_id, x):
    return {
        "schema": 1,
        "coordinate_mapping": "reference_to_source",
        "reference_size": [640, 360],
        "source": "/videos/v.mp4",
        "source_sha256": "original-source",
        "calibration_id": calibration_id,
        "samples": [
            {
                "frame": 10,
                "status": "tracking",
                "segment": 0,
                "alignment": {"homography": [[1, 0, x], [0, 1, 0], [0, 0, 1]]},
            }
        ],
    }


def test_geometry_uses_exact_row_reference_even_with_same_source_versions(tmp_path):
    old = save_geometry(tmp_path, "v.mp4", geometry_record("a" * 16, 10))
    new = save_geometry(tmp_path, "v.mp4", geometry_record("b" * 16, 20))
    row = {"video": "v.mp4", "calibration_id": "b" * 16, "geometry_evidence": new}
    got = jv._geometry_for(tmp_path, row, "/videos/v.mp4")
    assert got["calibration_id"] == "b" * 16
    assert got["samples"][10]["alignment"]["homography"][0][2] == 20
    assert (tmp_path / old["path"]).is_file()
    (tmp_path / new["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="SHA256"):
        jv._geometry_for(tmp_path, row, "/videos/v.mp4")


def test_build_view_keeps_each_rounds_own_geometry_and_calibration(
    tmp_path, monkeypatch
):
    evidence = [
        save_geometry(tmp_path, "v.mp4", geometry_record(c * 16, shift))
        for c, shift in [("a", 10), ("b", 20)]
    ]
    rows = [
        {
            "video": "v.mp4",
            "round": i + 1,
            "calibration_id": c * 16,
            "geometry_evidence": ref,
            "t_launch": 1 / 3,
            "confidence": 0,
            "finish": None,
            "winner": None,
            "needs_review": True,
            "review_reasons": [],
            "reviewed": False,
            "reviewer_fix": None,
        }
        for i, (c, ref) in enumerate(zip(["a", "b"], evidence, strict=True))
    ]
    (tmp_path / "rounds.json").write_text(json.dumps(rows))
    (tmp_path / "videos.json").write_text(
        json.dumps(
            {
                "v.mp4": {
                    "source_video": "/videos/v.mp4",
                    "calibration": {"id": "b" * 16},
                }
            }
        )
    )
    calls = []

    def snapshot(run, identifier, **kwargs):
        calls.append(identifier)
        return SimpleNamespace(calibration=make_calibration(), kind="manual", score=1.0)

    monkeypatch.setattr(jv, "load_snapshot", snapshot)
    view = jv.build_view(tmp_path, 0.7)
    assert calls == ["a" * 16, "b" * 16]
    assert view["rounds"]["v_r1"]["calibration"]["id"] == "a" * 16
    assert view["rounds"]["v_r2"]["calibration"]["id"] == "b" * 16
    assert (
        view["rounds"]["v_r2"]["geometry"]["zones"]["stadium"][0][0]
        - view["rounds"]["v_r1"]["geometry"]["zones"]["stadium"][0][0]
        == 10
    )

    rows[1]["video"] = "wrong.mp4"
    rows[1]["geometry_evidence"] = evidence[0]
    rows[1]["calibration_id"] = "a" * 16
    (tmp_path / "rounds.json").write_text(json.dumps(rows))
    (tmp_path / "videos.json").write_text(
        json.dumps(
            {
                "v.mp4": {"source_video": "/videos/v.mp4"},
                "wrong.mp4": {"source_video": "/videos/wrong.mp4"},
            }
        )
    )
    with pytest.raises(ValueError, match="source"):
        jv.build_view(tmp_path, 0.7)


def test_trail_renderer_does_not_connect_different_entities_or_gap_segments():
    script = Path(jv.__file__).with_name("web") / "judge_view.js"
    code = r"""
      const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
      const text=fs.readFileSync(process.argv[1],'utf8');
      const found=text.match(/function trailSegments\(tracks,t\)\{[\s\S]*?\n\}/);
      assert.ok(found,'A pure grouping helper must drive the actual renderer');
      const ctx={};vm.runInNewContext(found[0],ctx);
      const tracks={'':[[0,10,10,.9,101,'z',0,1],[0,100,10,.9,102,'z',0,1],
                        [.1,11,10,.9,101,'z',1,1],[.4,14,10,.9,101,'z',4,2]]};
      const segments=ctx.trailSegments(tracks,.5);
      assert.equal(segments.length,3);
      assert.deepEqual(Array.from(segments.map(s=>s.points.length)).sort(),[1,1,2]);
      assert.ok(segments.every(s=>s.points.every(p=>p[4]===s.entity)));
      assert.ok(/trailSegments\(d\.tracks,t\)/.test(text),'Draw must use the grouping helper');
    """
    subprocess.run(
        ["node", "-e", code, str(script)], check=True, capture_output=True, text=True
    )


def test_box_overlay_never_borrows_identity_from_a_neighbouring_source_frame():
    script = Path(jv.__file__).with_name("web") / "judge_view.js"
    code = r"""
      const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
      const text=fs.readFileSync(process.argv[1],'utf8');
      const found=text.match(/function sourceBoxes\(d,t\)\{[\s\S]*?\n\}/);
      assert.ok(found,'The actual renderer must use exact source-frame boxes');
      const ctx={};vm.runInNewContext(found[0],ctx);
      const d={fps:30,boxes:[[10/30,10,[[0,0,10,10,.9,'primary','X']]],
                            [12/30,12,[[0,0,10,10,.9,'primary','B']]]]};
      assert.equal(ctx.sourceBoxes(d,11/30),null);
      assert.equal(ctx.sourceBoxes(d,12/30)[1],12);
      assert.equal(ctx.sourceBoxes(d,-1),null);
      assert.ok(/sourceBoxes\(d,t\)/.test(text),'Draw must use exact source frame helper');
      assert.ok(!/function nearestBoxes/.test(text),'Remove the old nearest-frame path');
    """
    subprocess.run(
        ["node", "-e", code, str(script)], check=True, capture_output=True, text=True
    )
