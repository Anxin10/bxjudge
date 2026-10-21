from dataclasses import replace

import pytest

from bxjudge.segment import segment_rounds
from tests.helpers import make_track


def trk(first, last, track_id=1, cls="bey"):
    return make_track(track_id, cls, {f: (200, 180) for f in range(first, last + 1)})


def test_single_round(th):
    assert segment_rounds([trk(0, 300), trk(2, 310)], 60, th) == [(0, 310)]


def test_two_rounds(th):
    tracks = [trk(0, 300), trk(0, 290), trk(600, 900), trk(602, 880)]
    assert segment_rounds(tracks, 60, th) == [(0, 300), (600, 900)]


def test_no_tracks(th):
    assert segment_rounds([], 60, th) == []


def test_unsorted_nested_and_bridging_tracks_use_latest_active_end(th):
    tracks = [trk(602, 880), trk(200, 700), trk(0, 300), trk(20, 30)]
    original_order = tracks[:]
    assert segment_rounds(tracks, 60, th) == [(0, 880)]
    assert tracks == original_order


def test_short_track_gap_stays_in_same_round(th):
    assert segment_rounds([trk(0, 30), trk(100, 200)], 60, th) == [(0, 200)]


@pytest.mark.parametrize(
    "fps,gap_s,next_start,want",
    [
        (60, 2.0, 420, [(0, 500)]),
        (60, 2.0, 421, [(0, 300), (421, 500)]),
        (60, 2.0, 422, [(0, 300), (422, 500)]),
        (30, 2.0, 361, [(0, 300), (361, 500)]),
        (60, 3.0, 421, [(0, 500)]),
        (29.97, 0.5, 315, [(0, 500)]),
        (29.97, 0.5, 316, [(0, 300), (316, 500)]),
    ],
)
def test_gap_uses_inactive_frames_actual_fps_and_configured_threshold(
    th, fps, gap_s, next_start, want
):
    assert segment_rounds(
        [trk(0, 300), trk(next_start, 500)],
        fps,
        replace(th, round_gap_s=gap_s),
    ) == want


def test_empty_and_non_bey_tracks_do_not_extend_or_bridge_rounds(th):
    tracks = [
        make_track(0, "bey", {}),
        trk(0, 300),
        trk(290, 650, cls="bey_part"),
        trk(200, 800, cls="launcher"),
        trk(600, 900),
    ]
    assert segment_rounds(tracks, 60, th) == [(0, 300), (600, 900)]
    assert segment_rounds([make_track(0, "bey", {})], 60, th) == []

