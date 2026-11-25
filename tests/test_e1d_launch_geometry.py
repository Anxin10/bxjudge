"""E1d: launch-phase release as evidence; geometry voids counted from landing."""

from pathlib import Path

import pytest

from bxjudge.identity import load_identity_policy
from bxjudge.launch_phase import LaunchSegment, missing_geometry_start
from bxjudge.two_bey_integration import reconstruct_round_tracks, round_release_evidence
from bxjudge.two_bey_tracking import load_policy
from tests.helpers import make_calibration, make_track


def test_e1d_switches_default_off():
    policy = load_identity_policy()
    assert policy["phase_release_evidence_enabled"] is False
    assert policy["geometry_window_from_landing_enabled"] is False


@pytest.mark.parametrize("key", ["phase_release_evidence_enabled",
                                 "geometry_window_from_landing_enabled"])
def test_e1d_switches_must_be_boolean(tmp_path, key):
    base = {**load_identity_policy(), key: 1}
    path = tmp_path / "identity.yaml"
    path.write_text("\n".join(
        f"{k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in base.items()))
    with pytest.raises(ValueError):
        load_identity_policy(path)


def _physical():
    c = make_calibration()
    raw = [make_track(i, "bey", {f: (x, 200) for f in range(10, 18)})
           for i, x in ((1, 200), (2, 440))]
    physical = reconstruct_round_tracks(
        raw, 10, 17, 30, c.frame_size, {},
        load_policy(Path("config/two_bey_tracking.yaml")), 10)
    return physical, c


def test_phase_release_counts_only_when_enabled():
    physical, c = _physical()
    # The per-frame classifier never saw a separation (e.g. duplicate launcher boxes).
    segments = {1: [LaunchSegment("unknown", 10, 17)]}
    args = physical, physical.tracks, {11: "X", 12: "B"}, segments, c
    assert round_release_evidence(*args, required_frames=3) == {}
    proof = round_release_evidence(*args, required_frames=3, phase_releases={1: (10, 3)})
    assert proof[11]["frame"] == 10
    assert proof[11]["confirmed_at_frame"] == 12
    assert proof[11]["launcher_id"] == 3
    assert proof[11]["source"] == "launch_phase"


def test_phase_release_never_adds_to_a_segment_proof():
    physical, c = _physical()
    segments = {1: [LaunchSegment("mounted", 4, 9, 3, 9),
                    LaunchSegment("separated", 10, 17, 3, 12)]}
    args = physical, physical.tracks, {11: "X", 12: "B"}, segments, c
    proof = round_release_evidence(*args, required_frames=3, phase_releases={1: (14, 3)})
    assert proof[11]["frame"] == 10
    assert "source" not in proof[11]


def test_phase_release_outside_round_or_unobserved_is_rejected():
    physical, c = _physical()
    args = physical, physical.tracks, {11: "X", 12: "B"}, {}, c
    # Separation run would end after the round.
    assert round_release_evidence(*args, required_frames=3, phase_releases={1: (16, 3)}) == {}
    # Before the round.
    assert round_release_evidence(*args, required_frames=3, phase_releases={1: (5, 3)}) == {}
    # The actor is not observed as this native top during the separation run.
    row = next(o for o in physical.record["observations"]
               if o["entity_id"] == 1 and o["frame"] == 11)
    row["identity_valid"] = False
    assert round_release_evidence(*args, required_frames=3, phase_releases={1: (10, 3)}) == {}


def test_phase_release_needs_a_unique_side():
    physical, c = _physical()
    args = physical, physical.tracks, {11: "X", 12: "X"}, {}, c
    assert round_release_evidence(*args, required_frames=3,
                                  phase_releases={1: (10, 3), 2: (10, 4)}) == {}


def test_geometry_window_starts_at_first_observed_landing():
    assert missing_geometry_start(100, 3000, {"X": 900, "B": 880}) == 880


@pytest.mark.parametrize("landing", [
    None, {"X": None, "B": 880}, {"X": 900, "B": None}, {"X": 50, "B": 880},
    {"X": 900, "B": 3100}, {"X": 900.0, "B": 880},
])
def test_geometry_window_keeps_whole_round_without_both_landings(landing):
    assert missing_geometry_start(100, 3000, landing) == 100


def test_geometry_window_rejects_invalid_span():
    with pytest.raises(ValueError):
        missing_geometry_start(100, 50, {"X": 60, "B": 70})
