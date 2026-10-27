import pytest

from bxjudge.identity import anchor_sides
from tests.helpers import make_calibration, make_track


def test_short_missing_left_launcher_prefix_is_unknown_not_observed_free_motion():
    tracks = [
        make_track(1, "bey", {f: (200, 80) for f in range(8)}),
        make_track(2, "launcher", {f: (440, 40) for f in range(8)}, size=60),
        make_track(3, "launcher", {f: (200, 40) for f in range(2, 8)}, size=60),
    ]
    result = anchor_sides(
        tracks, [], make_calibration(), required_frames=4, fps=30, body_margin_px=12
    )
    assert result.sides == {1: "X"}
    assert result.evidence[1][0]["frame"] == 2


def test_duplicate_right_launcher_components_do_not_hide_separated_left_and_right_bodies():
    tracks = [
        make_track(1, "bey", {f: (440, 80) for f in range(6)}),
        make_track(2, "launcher", {f: (200, 40) for f in range(6)}, size=60),
        make_track(3, "launcher", {f: (440, 40) for f in range(6)}, size=60),
        make_track(4, "launcher", {f: (440, 50) for f in range(6)}, size=30),
    ]
    result = anchor_sides(
        tracks, [], make_calibration(), required_frames=4, fps=30, body_margin_px=12
    )
    assert result.sides == {1: "B"}


def test_long_unknown_prefix_cannot_turn_later_pickup_into_launch_origin():
    tracks = [
        make_track(
            1,
            "bey",
            {
                **{f: (500, 220) for f in range(40)},
                **{f: (200, 80) for f in range(40, 48)},
            },
        ),
        make_track(2, "launcher", {f: (440, 40) for f in range(48)}, size=60),
        make_track(3, "launcher", {f: (200, 40) for f in range(40, 48)}, size=60),
    ]
    assert (
        anchor_sides(
            tracks, [], make_calibration(), required_frames=4, fps=30, body_margin_px=12
        ).sides
        == {}
    )


def test_real_210958_launch_bodies_anchor_x_and_b_before_the_confirmed_swap():
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from bxjudge.dynamic_calibration import normalize_tracks
    from bxjudge.types import Calibration, Detection, Track

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/210958-launch-context.json").read_bytes()
    )
    calibration = Calibration(**fixture["calibration"])
    tracks = [
        Track(
            t["track_id"],
            t["cls"],
            [
                Detection(d["frame"], d["cls"], tuple(d["box"]), d["score"])
                for d in t["dets"]
            ],
        )
        for t in fixture["raw_source_tracks"]
    ]
    samples = {
        s["frame"]: SimpleNamespace(
            status=s["status"],
            alignment=SimpleNamespace(**s["alignment"]) if s["alignment"] else None,
        )
        for s in fixture["samples"]
    }
    result = anchor_sides(
        tracks,
        normalize_tracks(tracks, samples),
        calibration,
        required_frames=7,
        body_margin_px=12,
        fps=30,
    )
    assert result.sides == {5: "B", 6: "X"}
    assert {d["frame"] for d in result.evidence[6]} == set(range(2228, 2235))


@pytest.mark.parametrize("fps,expected", [(30, {}), (120, {1: "X"})])
def test_missing_launcher_wait_is_source_time_bounded(fps, expected):
    tracks = [
        make_track(1, "bey", {f: (200, 80) for f in range(48)}),
        make_track(2, "launcher", {f: (440, 40) for f in range(48)}, size=60),
        make_track(3, "launcher", {f: (200, 40) for f in range(40, 48)}, size=60),
    ]
    assert (
        anchor_sides(
            tracks,
            [],
            make_calibration(),
            required_frames=4,
            fps=fps,
            body_margin_px=12,
        ).sides
        == expected
    )


def test_identity_policy_missing_fields_are_not_filled(tmp_path):
    from bxjudge.identity import load_identity_policy

    path = tmp_path / "policy.yaml"
    path.write_text("context_wait_s: .75\n")
    with pytest.raises(ValueError):
        load_identity_policy(path)


def test_sparse_unknown_prefix_cannot_skip_the_source_time_deadline():
    tracks = [
        make_track(1, "bey", {0: (200, 80), **{f: (200, 80) for f in range(40, 48)}}),
        make_track(2, "launcher", {f: (440, 40) for f in range(48)}, size=60),
        make_track(3, "launcher", {f: (200, 40) for f in range(40, 48)}, size=60),
    ]
    assert (
        anchor_sides(
            tracks, [], make_calibration(), required_frames=4, fps=30, body_margin_px=12
        ).sides
        == {}
    )


def test_wide_body_spanning_two_players_cannot_merge_them_into_one_side():
    from bxjudge.identity import _body_pair

    boxes = [
        (1, (0, 0, 100, 100)),
        (2, (150, 0, 250, 100)),
        (3, (0, 0, 250, 100)),
        (4, (300, 0, 400, 100)),
    ]
    assert _body_pair(boxes, 0.8) == {}
