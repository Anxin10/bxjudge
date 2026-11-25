"""E1c: an observed launcher release clocks the round start (setup is not battle)."""

import pytest

from bxjudge.identity import load_identity_policy
from bxjudge.launch_phase import trim_rounds_to_release


def test_e1c_disabled_by_default():
    assert load_identity_policy()["release_clock_trim_enabled"] is False


def test_e1c_flag_must_be_boolean(tmp_path):
    base = load_identity_policy()
    path = tmp_path / "identity.yaml"
    path.write_text(
        "\n".join(f"{k}: {str(v).lower() if isinstance(v, bool) else v}"
                  for k, v in {**base, "release_clock_trim_enabled": 1}.items())
    )
    with pytest.raises(ValueError):
        load_identity_policy(path)


def test_round_starts_at_earliest_release_minus_pre_frames():
    # Setup beys seen from frame 100; releases observed at 900 and 905.
    assert trim_rounds_to_release([(100, 3000)], {7: 905, 9: 900}, 60) == [(840, 3000)]


def test_round_without_release_is_unchanged():
    assert trim_rounds_to_release([(100, 3000)], {}, 60) == [(100, 3000)]


def test_release_outside_round_is_ignored():
    rounds = [(100, 1000), (2000, 3000)]
    assert trim_rounds_to_release(rounds, {4: 1500}, 60) == [(100, 1000), (2000, 3000)]


def test_trim_never_moves_start_backwards():
    # A release right after the start (within pre_frames) keeps the original start.
    assert trim_rounds_to_release([(100, 3000)], {1: 130}, 60) == [(100, 3000)]


def test_each_round_uses_only_its_own_releases():
    rounds = [(0, 1000), (1500, 4000)]
    releases = {1: 500, 2: 3000, 3: 3100}
    assert trim_rounds_to_release(rounds, releases, 60) == [(440, 1000), (2940, 4000)]


def test_trim_keeps_round_non_empty():
    with pytest.raises(ValueError):
        trim_rounds_to_release([(100, 3000)], {1: 900}, -1)
