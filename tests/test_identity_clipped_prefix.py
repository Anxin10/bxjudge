"""A clipped prefix with unknown launcher context is not observed free flight."""

import json
from pathlib import Path
from types import SimpleNamespace

from bxjudge.dynamic_calibration import normalize_tracks
from bxjudge.identity import anchor_sides
from bxjudge.types import Calibration, Detection, Track
from tests.helpers import make_calibration, make_track


def raised_launch_zones():
    c = make_calibration()
    c.launch_left = [(40, 30), (320, 30), (320, 120), (40, 120)]
    c.launch_right = [(320, 30), (600, 30), (600, 120), (320, 120)]
    return c


def launch_prefix(frames):
    top = {f: ((450, 20) if f < 2 else (450, 80)) for f in frames}
    body = {f: ((450, 20) if f < 2 else (450, 65)) for f in frames}
    return [
        make_track(1, "bey", top, size=40),
        make_track(2, "launcher", body, size=40),
    ]


def test_clipped_near_launcher_unknown_prefix_waits_for_registered_side():
    tracks = launch_prefix(range(9))
    result = anchor_sides(
        tracks,
        tracks,
        raised_launch_zones(),
        required_frames=7,
        body_margin_px=12,
        fps=30,
    )
    assert result.sides == {1: "B"}
    assert [e["frame"] for e in result.evidence[1]] == list(range(2, 9))


def test_unclipped_free_prefix_cannot_anchor_at_later_pickup():
    tracks = [
        make_track(
            1, "bey", {0: (450, 220), **{f: (450, 80) for f in range(1, 10)}}, size=20
        ),
        make_track(
            2,
            "launcher",
            {0: (450, 200), **{f: (450, 65) for f in range(1, 10)}},
            size=40,
        ),
    ]
    assert (
        anchor_sides(
            tracks,
            tracks,
            raised_launch_zones(),
            required_frames=7,
            body_margin_px=12,
            fps=30,
        ).sides
        == {}
    )


def test_clipped_prefix_still_obeys_the_source_time_deadline():
    tracks = launch_prefix([0, *range(40, 48)])
    assert (
        anchor_sides(
            tracks,
            tracks,
            raised_launch_zones(),
            required_frames=7,
            body_margin_px=12,
            fps=30,
        ).sides
        == {}
    )


def test_bottom_clipped_free_prefix_cannot_anchor_at_later_pickup():
    tracks = [
        make_track(
            1, "bey", {0: (450, 350), **{f: (450, 80) for f in range(1, 9)}}, size=20
        ),
        make_track(
            2,
            "launcher",
            {0: (450, 325), **{f: (450, 65) for f in range(1, 9)}},
            size=60,
        ),
    ]
    assert (
        anchor_sides(
            tracks,
            tracks,
            raised_launch_zones(),
            required_frames=7,
            body_margin_px=12,
            fps=30,
        ).sides
        == {}
    )


def test_intermediate_unknown_context_does_not_clear_the_prefix_deadline():
    tracks = [
        make_track(
            1,
            "bey",
            {0: (450, 20), 1: (450, 40), **{f: (450, 80) for f in range(40, 47)}},
            size=40,
        ),
        make_track(
            2,
            "launcher",
            {0: (450, 20), 1: (450, 20), **{f: (450, 65) for f in range(40, 47)}},
            size=40,
        ),
    ]
    assert (
        anchor_sides(
            tracks,
            tracks,
            raised_launch_zones(),
            required_frames=7,
            body_margin_px=12,
            fps=30,
        ).sides
        == {}
    )


def test_clipped_prefix_cannot_hide_conflicting_registered_launcher_roles():
    frames = range(8)
    tracks = [
        make_track(
            1, "bey", {f: ((450, 20) if f == 0 else (450, 80)) for f in frames}, size=40
        ),
        make_track(
            2,
            "launcher",
            {f: ((450, 20) if f == 0 else (450, 65)) for f in frames},
            size=40,
        ),
        make_track(3, "launcher", {f: (200, 65) for f in frames}, size=40),
    ]
    reference = [
        tracks[0],
        make_track(
            2,
            "launcher",
            {f: ((200, 60) if f == 0 else (450, 65)) for f in frames},
            size=40,
        ),
        make_track(
            3,
            "launcher",
            {f: ((450, 65) if f == 0 else (200, 65)) for f in frames},
            size=40,
        ),
    ]
    assert (
        anchor_sides(
            tracks,
            reference,
            raised_launch_zones(),
            required_frames=7,
            body_margin_px=12,
            fps=30,
        ).sides
        == {}
    )


def test_real_212413_prefix_uses_only_seven_later_verified_observations():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/212413-launch-prefix.json").read_bytes()
    )
    c = Calibration(**fixture["calibration"])
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
            alignment=(SimpleNamespace(**s["alignment"]) if s["alignment"] else None),
        )
        for s in fixture["samples"]
    }
    result = anchor_sides(
        tracks,
        normalize_tracks(tracks, samples),
        c,
        required_frames=7,
        body_margin_px=12,
        fps=fixture["fps"],
    )
    assert result.sides == {7: "B"}
    assert [e["frame"] for e in result.evidence[7]] == list(range(929, 936))
