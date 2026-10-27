"""User ruling: a confirmed earlier non-scoring exit replays the whole bout."""

from pathlib import Path

import pytest
import yaml

from bxjudge.rules import decide_round, group_matches, load_rules
from tests.helpers import rr


@pytest.mark.parametrize("side", ["X", "B", "unknown"])
def test_confirmed_non_scoring_exit_precedes_later_finish(rules, side):
    result = rr("xtreme", "X", review_reasons=[f"non_scoring_exit@{side}"])
    misses = {"X": 1, "B": 0}
    outcome = decide_round(result, rules, misses)
    assert (outcome.action, outcome.scorer, outcome.points) == ("redo", None, 0)
    assert outcome.reasons == [f"non_scoring_exit@{side}"]
    assert misses == {"X": 1, "B": 0}  # A mid-bout exit is not a launch miss.


def test_confirmed_exit_without_any_scoring_finish_is_not_called_a_draw(rules):
    result = rr(None, None, review_reasons=["non_scoring_exit@B"])
    outcome = decide_round(result, rules, {"X": 0, "B": 0})
    assert outcome.reasons == ["non_scoring_exit@B"]


def test_replayed_round_stays_in_population_and_does_not_score_later_xtreme(rules):
    replay = rr("xtreme", "X", round=1, review_reasons=["non_scoring_exit@B"])
    following = rr("xtreme", "X", round=2)
    [match] = group_matches([replay, following], rules)
    assert len(match.rounds) == 2
    assert [outcome.action for _, outcome in match.rounds] == ["redo", "score"]
    assert match.score == {"X": 3, "B": 0} and match.winner is None
    assert replay.finish == "xtreme" and replay.winner == "X"  # Original event retained.


@pytest.mark.parametrize(
    "reason",
    [
        "after_finish:non_scoring_exit@B",
        "non_scoring_exit_unconfirmed@B",
        "non_scoring_exit_timing_unknown@B",
    ],
)
def test_late_or_unconfirmed_exit_does_not_automatically_undo_a_finish(rules, reason):
    result = rr("xtreme", "X", review_reasons=[reason])
    result.needs_review = True
    outcome = decide_round(result, rules, {"X": 0, "B": 0})
    assert (outcome.action, outcome.scorer, outcome.points) == ("score", "X", 3)
    assert result.needs_review  # Still a proposal; uncertainty was not cleared.


@pytest.mark.parametrize(
    "interval,finish,expected",
    [
        ((100, 102), 200, "non_scoring_exit@B"),
        ((220, 222), 200, "after_finish:non_scoring_exit@B"),
        ((199, 201), 200, "non_scoring_exit_timing_unknown@B"),
        ((200, 200), 200, "non_scoring_exit_timing_unknown@B"),
        ((100, 102), None, "non_scoring_exit@B"),
    ],
)
def test_exit_interval_preserves_earlier_later_and_unknown_order(
    interval, finish, expected
):
    from bxjudge.rules import non_scoring_exit_reason

    assert non_scoring_exit_reason("B", interval, finish) == expected


def test_global_exit_rule_must_be_explicit_in_yaml(tmp_path):
    data = yaml.safe_load(Path("config/rules.yaml").read_text(encoding="utf-8"))
    assert data["battle_check"] == {"non_scoring_exit": "redo"}
    del data["battle_check"]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="battle_check"):
        load_rules(path)


def test_global_exit_rule_rejects_an_unsupported_scoring_action(tmp_path):
    data = yaml.safe_load(Path("config/rules.yaml").read_text(encoding="utf-8"))
    data["battle_check"]["non_scoring_exit"] = "score"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="non_scoring_exit"):
        load_rules(path)


@pytest.mark.parametrize(
    "interval,finish",
    [
        ((101, 100), 200),
        ((-1, 100), 200),
        ((True, 102), 200),
        ((100, 102), -1),
        ((100, 102), 200.5),
    ],
)
def test_invalid_exit_timing_does_not_become_a_replay(interval, finish):
    from bxjudge.rules import non_scoring_exit_reason

    with pytest.raises(ValueError):
        non_scoring_exit_reason("B", interval, finish)
