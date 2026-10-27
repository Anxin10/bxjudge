from dataclasses import replace

from bxjudge.launch_phase import load_launch_policy
from tests.helpers import make_calibration, make_track, rect


def anchors(tracks, *, c=None, frames=4):
    from bxjudge.identity import anchor_sides

    return anchor_sides(
        tracks,
        tracks,
        c or make_calibration(),
        required_frames=frames,
        body_margin_px=load_launch_policy()["body_margin_px"],
    )


def test_identity_keeps_launch_origin_when_the_same_top_crosses_screen():
    bey = make_track(1, "bey", {**{f: (200, 80) for f in range(4)}, 4: (500, 220)})
    launcher = make_track(2, "launcher", {f: (200, 40) for f in range(4)}, size=60)
    result = anchors([bey, launcher])
    assert result.sides == {1: "X"}
    assert [x["frame"] for x in result.evidence[1]] == [0, 1, 2, 3]
    assert result.evidence[1][0]["launcher_ids"] == [2]
    assert bey.center_at(4) == (500, 220)


def test_midbattle_first_appearance_has_no_side_even_in_right_half():
    bey = make_track(1, "bey", {f: (500, 220) for f in range(8)})
    assert anchors([bey]).sides == {}


def test_both_launch_regions_can_be_disambiguated_by_one_launcher_side():
    c = replace(
        make_calibration(),
        launch_left=rect(40, 0, 400, 120),
        launch_right=rect(300, 0, 600, 120),
    )
    bey = make_track(1, "bey", {f: (350, 80) for f in range(4)})
    body = make_track(2, "launcher", {f: (450, 40) for f in range(4)}, size=220)
    assert anchors([bey, body], c=c).sides == {1: "B"}


def test_duplicate_launcher_ids_from_one_side_do_not_create_identity_conflict():
    bey = make_track(1, "bey", {f: (200, 80) for f in range(4)})
    bodies = [
        make_track(i, "launcher", {f: (200, 40) for f in range(4)}, size=60)
        for i in (2, 3)
    ]
    result = anchors([bey, *bodies])
    assert result.sides == {1: "X"}
    assert result.evidence[1][0]["launcher_ids"] == [2, 3]


def test_ambiguous_launcher_region_and_conflicting_top_region_stay_unknown():
    bey = make_track(1, "bey", {f: (300, 80) for f in range(4)})
    body = make_track(2, "launcher", {f: (400, 40) for f in range(4)}, size=220)
    assert anchors([bey, body]).sides == {}
    ambiguous = make_track(3, "launcher", {f: (320, 40) for f in range(4)}, size=220)
    assert anchors([bey, ambiguous]).sides == {}


def test_missing_frames_restart_anchor_support_without_interpolating():
    frames = [0, 1, 2, 4, 5, 6]
    bey = make_track(1, "bey", {f: (200, 80) for f in frames})
    body = make_track(2, "launcher", {f: (200, 40) for f in frames}, size=60)
    assert anchors([bey, body]).sides == {}
    extended = [*frames, 7]
    bey = make_track(1, "bey", {f: (200, 80) for f in extended})
    body = make_track(2, "launcher", {f: (200, 40) for f in extended}, size=60)
    result = anchors([bey, body])
    assert [x["frame"] for x in result.evidence[1]] == [4, 5, 6, 7]


def test_later_pickup_cannot_replace_an_unsupported_initial_identity():
    bey = make_track(1, "bey", {0: (500, 220), **{f: (200, 80) for f in range(1, 8)}})
    body = make_track(2, "launcher", {f: (200, 40) for f in range(1, 8)}, size=60)
    assert anchors([bey, body]).sides == {}


def test_side_change_before_anchor_confirmation_is_rejected():
    bey = make_track(1, "bey", {0: (200, 80), **{f: (440, 80) for f in range(1, 8)}})
    body = make_track(
        2, "launcher", {0: (200, 40), **{f: (440, 40) for f in range(1, 8)}}, size=60
    )
    assert anchors([bey, body]).sides == {}


def test_only_verified_members_inherit_side_and_conflicts_stay_unknown():
    from bxjudge.identity import inherited_sides

    hints = {2: "X", 3: "B", 4: "X"}
    assert inherited_sides(hints, {1: [1, 2], 3: [3], 8: [8]}) == {1: "X", 3: "B"}
    assert inherited_sides(hints, {1: [1, 2, 3], 4: [4]}) == {4: "X"}
    assert hints == {2: "X", 3: "B", 4: "X"}


def test_anchor_validation_rejects_invalid_evidence_budgets():
    import pytest

    from bxjudge.identity import anchor_sides

    for count in (0, 1, True, 2.5):
        with pytest.raises(ValueError):
            anchor_sides(
                [], [], make_calibration(), required_frames=count, body_margin_px=12
            )
    for margin in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            anchor_sides(
                [], [], make_calibration(), required_frames=2, body_margin_px=margin
            )


def test_missing_geometry_prefix_cannot_be_replaced_by_later_pickup_anchor():
    from bxjudge.identity import anchor_sides

    bey = make_track(1, "bey", {f: (200, 80) for f in range(10, 14)})
    body = make_track(2, "launcher", {f: (200, 40) for f in range(10, 14)}, size=60)
    raw = [
        make_track(
            1,
            "bey",
            {
                **{f: (500, 220) for f in range(10)},
                **{f: (200, 80) for f in range(10, 14)},
            },
        ),
        make_track(2, "launcher", {f: (200, 40) for f in range(14)}, size=60),
    ]
    result = anchor_sides(
        raw, [bey, body], make_calibration(), required_frames=4, body_margin_px=12
    )
    assert result.sides == {}


def test_internal_missing_geometry_is_not_relabelled_as_a_detector_gap():
    from bxjudge.identity import anchor_sides

    normalized = [
        make_track(1, "bey", {f: (200, 80) for f in (0, 2, 3, 4, 5)}),
        make_track(2, "launcher", {f: (200, 40) for f in range(6)}, size=60),
    ]
    # The raw top was observed at f1: it was geometry, not detection, that failed.
    raw = [
        make_track(
            1, "bey", {f: ((500, 220) if f == 1 else (200, 80)) for f in range(6)}
        ),
        normalized[1],
    ]
    result = anchor_sides(
        raw, normalized, make_calibration(), required_frames=4, body_margin_px=12
    )
    assert result.sides == {}


def test_overlapping_body_duplicates_without_geometry_do_not_imply_two_players():
    from bxjudge.identity import anchor_sides

    tracks = [
        make_track(1, "bey", {f: (200, 80) for f in range(4)}),
        *[
            make_track(i, "launcher", {f: (200, 40) for f in range(4)}, size=60)
            for i in (2, 3)
        ],
    ]
    assert (
        anchor_sides(
            tracks, [], make_calibration(), required_frames=4, body_margin_px=12
        ).sides
        == {}
    )


def test_one_body_without_registered_position_has_no_guessed_screen_side():
    from bxjudge.identity import anchor_sides

    tracks = [
        make_track(1, "bey", {f: (440, 80) for f in range(4)}),
        make_track(2, "launcher", {f: (440, 40) for f in range(4)}, size=60),
    ]
    assert (
        anchor_sides(
            tracks, [], make_calibration(), required_frames=4, body_margin_px=12
        ).sides
        == {}
    )


def test_conflicting_registered_regions_and_source_pair_remain_unknown():
    from bxjudge.identity import anchor_sides

    tracks = [
        make_track(1, "bey", {f: (200, 80) for f in range(4)}),
        make_track(2, "bey", {f: (440, 80) for f in range(4)}),
        make_track(3, "launcher", {f: (200, 40) for f in range(4)}, size=60),
        make_track(4, "launcher", {f: (440, 40) for f in range(4)}, size=60),
    ]
    c = replace(
        make_calibration(),
        launch_left=rect(40, 0, 500, 120),
        launch_right=rect(500, 0, 600, 120),
    )
    assert (
        anchor_sides(tracks, tracks, c, required_frames=4, body_margin_px=12).sides
        == {}
    )
