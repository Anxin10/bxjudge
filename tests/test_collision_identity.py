import json
from pathlib import Path

from bxjudge.two_bey_tracking import TwoBeyTracker, load_policy


def test_real_210958_split_cannot_keep_the_swapped_player_identity():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/210958-merge-2796-2840.json").read_bytes()
    )
    tracker = TwoBeyTracker(
        tuple(fixture["frame_size"]),
        fixture["fps"],
        load_policy(Path("config/two_bey_tracking.yaml")),
    )
    after = []
    for frame in fixture["observations"]:
        result = tracker.update(frame["frame"], frame["detections"])
        if frame["frame"] >= 2828:
            after.extend(result["observed"])
    assert after
    assert all(row["identity_valid"] is False for row in after)
    assert len(tracker.identity_unknown_since) == 2
    assert max(tracker.identity_unknown_since.values()) < 2828


def test_single_far_object_loss_does_not_invent_a_merge():
    policy = load_policy(Path("config/two_bey_tracking.yaml"))
    tracker = TwoBeyTracker((300, 200), 30, policy)

    def detection(tid, x):
        return {
            "native_id": tid,
            "box": [x, 50, x + 20, 70],
            "score": 0.9,
            "appearance": None,
        }

    tracker.update(0, [detection(1, 30), detection(2, 220)])
    result = tracker.update(1, [detection(1, 32)])
    assert tracker.identity_unknown_since == {}
    assert result["observed"][0]["identity_valid"] is True


def test_real_merge_is_preserved_in_physical_round_record():
    from bxjudge.two_bey_integration import reconstruct_round_tracks
    from bxjudge.types import Detection, Track

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/210958-merge-2796-2840.json").read_bytes()
    )
    ids = sorted(
        {d["native_id"] for r in fixture["observations"] for d in r["detections"]}
    )
    tracks = [
        Track(
            tid,
            "bey",
            [
                Detection(r["frame"], "bey", tuple(d["box"]), d["score"])
                for r in fixture["observations"]
                for d in r["detections"]
                if d["native_id"] == tid
            ],
        )
        for tid in ids
    ]
    result = reconstruct_round_tracks(
        tracks,
        2796,
        2840,
        fixture["fps"],
        tuple(fixture["frame_size"]),
        {},
        load_policy(Path("config/two_bey_tracking.yaml")),
        100,
    )
    assert len(result.record["identity_unknown_since"]) == 2
    assert sum(len(t.dets) for t in tracks) == (
        len(result.record["observations"]) + len(result.record["unassigned"])
    )
    assert all(
        d["identity_valid"] is False
        for d in result.record["observations"]
        if d["frame"] >= 2828
    )


def test_trajectory_never_exports_old_player_after_merge(tmp_path):
    import csv

    from bxjudge.output import write_trajectory
    from bxjudge.types import RoundResult
    from tests.helpers import make_track, stadium_scene

    image, calibration = stadium_scene(size=(640, 360))
    import cv2

    path = tmp_path / "reference.png"
    assert cv2.imwrite(str(path), image)
    calibration.ref_frame = str(path)
    track = make_track(1, "bey", {f: (300 + f, 200) for f in range(5)})
    result = RoundResult("clip.mp4", 1, 0, 1, None, None, None, 0, 0.0, True, [])
    csv_path, _ = write_trajectory(
        tmp_path,
        result,
        {"X": track},
        [],
        calibration,
        30,
        {},
        unknown_side_since={1: 2},
    )
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["side"] for r in rows] == ["X", "X", "", "", ""]


def test_pipeline_blocks_finish_and_exports_unknown_after_actual_merge(
    tmp_path, monkeypatch, rules, th
):
    import csv

    from bxjudge.types import Detection, Track
    from tests.test_two_bey_integration import invoke_two_pipeline, pipeline_setup

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/210958-merge-2796-2840.json").read_bytes()
    )
    tracks = [
        Track(
            tid,
            "bey",
            [
                Detection(r["frame"] - 2796, "bey", tuple(d["box"]), d["score"])
                for r in fixture["observations"]
                for d in r["detections"]
                if d["native_id"] == native
            ],
        )
        for tid, native in ((1, 6), (2, 10))
    ]
    source = pipeline_setup(monkeypatch, tmp_path, tracks)
    [result] = invoke_two_pipeline(source, tmp_path, rules, th)
    assert result.winner is None
    assert "identity_after_merge_unknown" in result.review_reasons
    paths = list((tmp_path / "out/trajectories").glob("*.csv"))
    with paths[0].open() as stream:
        rows = list(csv.DictReader(stream))
    assert any(r["side"] in ("X", "B") for r in rows if int(r["frame"]) < 10)
    assert all(r["side"] == "" for r in rows if int(r["frame"]) >= 32)


def test_side_specific_core_never_receives_post_merge_observations(
    tmp_path, monkeypatch, rules, th
):
    from bxjudge import pipeline
    from bxjudge.types import Detection, Track
    from tests.test_two_bey_integration import invoke_two_pipeline, pipeline_setup

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/210958-merge-2796-2840.json").read_bytes()
    )
    tracks = [
        Track(
            tid,
            "bey",
            [
                Detection(r["frame"] - 2796, "bey", tuple(d["box"]), d["score"])
                for r in fixture["observations"]
                for d in r["detections"]
                if d["native_id"] == native
            ],
        )
        for tid, native in ((1, 6), (2, 10))
    ]
    source = pipeline_setup(monkeypatch, tmp_path, tracks)
    original = pipeline.analyze_launch
    seen = []

    def check(current, *args, **kwargs):
        seen.extend(d.frame for t in current for d in t.dets)
        assert all(d.frame < 12 for t in current for d in t.dets)
        return original(current, *args, **kwargs)

    monkeypatch.setattr(pipeline, "analyze_launch", check)
    invoke_two_pipeline(source, tmp_path, rules, th)
    assert seen
