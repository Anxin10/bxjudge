import pytest
import yaml

from bxjudge.rules import decide_round, finish_points, group_matches, load_rules
from tests.helpers import rr


def test_finish_points(rules):
    assert [finish_points(rules, f) for f in ("spin", "over", "burst", "xtreme")] == [1, 2, 2, 3]


def test_xtreme_scores_three(rules):
    o = decide_round(rr("xtreme", "X"), rules, {"X": 0, "B": 0})
    assert (o.action, o.scorer, o.points) == ("score", "X", 3)


def test_touch_yes_redo_and_counts_miss(rules):
    m = {"X": 0, "B": 0}
    o = decide_round(rr("spin", "B", touch={"X": "yes", "B": "no"}), rules, m)
    assert o.action == "redo" and m == {"X": 1, "B": 0}


def test_not_landed_redo(rules):
    o = decide_round(rr("spin", "X", landed={"X": True, "B": False}), rules, {"X": 0, "B": 0})
    assert o.action == "redo"


def test_air_collision_yes_redo_no_miss(rules):
    m = {"X": 0, "B": 0}
    assert decide_round(rr("spin", "X", air="yes"), rules, m).action == "redo"
    assert m == {"X": 0, "B": 0}


def test_unknown_touch_and_maybe_air_still_score(rules):
    o = decide_round(
        rr("over", "B", touch={"X": "unknown", "B": "no"}, air="maybe"), rules, {"X": 0, "B": 0}
    )
    assert (o.action, o.scorer, o.points) == ("score", "B", 2)


def test_draw_redo(rules):
    assert decide_round(rr(None, None), rules, {"X": 0, "B": 0}).action == "redo"


def test_second_miss_penalty(rules):
    rules.launch_check["repeat_miss_penalty"] = "opponent_1pt_on_2nd_miss"
    m = {"X": 1, "B": 0}
    o = decide_round(rr("spin", "X", touch={"X": "yes", "B": "no"}), rules, m)
    assert (o.action, o.scorer, o.points) == ("score", "B", 1)


def test_group_matches_first_to_four(rules):
    rs = [rr("xtreme", "X"), rr("spin", "B"), rr("spin", "X"), rr("over", "B")]
    ms = group_matches(rs, rules)
    assert ms[0].winner == "X" and ms[0].score == {"X": 4, "B": 1}
    assert len(ms) == 2 and ms[1].winner is None and ms[1].score == {"X": 0, "B": 2}


# ---- additional behaviour -------------------------------------------------


def test_launch_none_proceeds_to_finish(rules):
    o = decide_round(rr("burst", "B", launch=False), rules, {"X": 0, "B": 0})
    assert (o.action, o.scorer, o.points) == ("score", "B", 2)


def test_first_miss_with_penalty_mode_is_only_redo(rules):
    rules.launch_check["repeat_miss_penalty"] = "opponent_1pt_on_2nd_miss"
    m = {"X": 0, "B": 0}
    o = decide_round(rr("spin", "X", touch={"X": "yes", "B": "no"}), rules, m)
    assert o.action == "redo" and m == {"X": 1, "B": 0}


def test_second_miss_without_penalty_mode_is_redo(rules):
    m = {"X": 1, "B": 0}
    o = decide_round(rr("spin", "B", touch={"X": "yes", "B": "no"}), rules, m)
    assert o.action == "redo" and m == {"X": 2, "B": 0}


def test_touch_and_not_landed_same_side_counts_one_miss(rules):
    m = {"X": 0, "B": 0}
    decide_round(rr("spin", "B", touch={"X": "yes"}, landed={"X": False}), rules, m)
    assert m == {"X": 1, "B": 0}


def test_both_sides_second_miss_does_not_penalise_either(rules):
    rules.launch_check["repeat_miss_penalty"] = "opponent_1pt_on_2nd_miss"
    m = {"X": 1, "B": 1}
    o = decide_round(rr("spin", "X", touch={"X": "yes", "B": "yes"}), rules, m)
    assert o.action == "redo" and m == {"X": 2, "B": 2}


def test_redo_reasons_reported(rules):
    o = decide_round(
        rr("spin", "X", touch={"X": "yes"}, landed={"B": False}, air="yes"),
        rules,
        {"X": 0, "B": 0},
    )
    assert o.action == "redo"
    assert set(o.reasons) == {"touch_X", "not_landed_B", "air_collision"}


def test_ignored_launch_failure_scores(rules):
    rules.launch_check["stadium_touch"] = "ignore"
    m = {"X": 0, "B": 0}
    o = decide_round(rr("spin", "B", touch={"X": "yes"}), rules, m)
    assert (o.action, o.scorer, o.points) == ("score", "B", 1) and m == {"X": 0, "B": 0}


def test_finish_without_winner_is_redo_not_crash(rules):
    o = decide_round(rr("spin", None), rules, {"X": 0, "B": 0})
    assert o.action == "redo" and o.reasons == ["finish_without_winner"]


def test_group_matches_resets_misses_per_match(rules):
    rules.launch_check["repeat_miss_penalty"] = "opponent_1pt_on_2nd_miss"
    miss_x = dict(touch={"X": "yes"})
    rs = [
        rr("spin", "B", **miss_x, round=1),
        rr("xtreme", "X", round=2),
        rr("spin", "X", round=3),  # X now 4 -> match 1 over
        rr("spin", "B", **miss_x, round=4),  # first miss of match 2 -> must be redo
    ]
    ms = group_matches(rs, rules)
    assert len(ms) == 2
    assert ms[0].score == {"X": 4, "B": 0}
    assert ms[1].score == {"X": 0, "B": 0}
    assert ms[1].rounds[0][1].action == "redo"


def test_group_matches_includes_redo_rounds(rules):
    rs = [rr("spin", "X", air="yes", round=1), rr("spin", "X", round=2)]
    (m,) = group_matches(rs, rules)
    assert [o.action for _, o in m.rounds] == ["redo", "score"]
    assert m.score == {"X": 1, "B": 0}


def test_group_matches_skips_non_rounds(rules):
    rs = [
        rr("xtreme", "X", round=1),
        rr("xtreme", "B", round=2, review_reasons=["recalibrate"]),
        rr("xtreme", "B", round=3, review_reasons=["no_round"]),
        rr("xtreme", "B", round=4, review_reasons=["decode_error", "other"]),
        rr("spin", "X", round=5, review_reasons=["low_confidence"]),  # real round, kept
    ]
    (m,) = group_matches(rs, rules)
    assert m.score == {"X": 4, "B": 0}
    assert len(m.rounds) == 2


def test_group_matches_sorts_by_video_then_t_launch(rules):
    rs = [
        rr("spin", "B", video="b.mp4", t_launch=1.0),
        rr("over", "X", video="a.mp4", t_launch=50.0),
        rr("spin", "X", video="a.mp4", t_launch=5.0),
    ]
    (m,) = group_matches(rs, rules)
    assert [(r.video, r.t_launch) for r, _ in m.rounds] == [
        ("a.mp4", 5.0),
        ("a.mp4", 50.0),
        ("b.mp4", 1.0),
    ]


def test_group_matches_empty(rules):
    assert group_matches([], rules) == []


# ---- rules.yaml loading ---------------------------------------------------


def test_default_rules_values(rules):
    assert rules.rules_version == "takara-tomy-bx-2025-reg8"
    assert rules.win_points == 4
    assert rules.finish_points == {"spin": 1, "burst": 2, "over": 2, "xtreme": 3}
    assert rules.launch_check == {
        "stadium_touch": "redo",
        "bey_not_in_battle_area": "redo",
        "mid_air_collision": "redo",
        "repeat_miss_penalty": "none",
    }
    assert rules.draw == {"action": "redo"}


def _write(tmp_path, mutate):
    data = yaml.safe_load(open("config/rules.yaml", encoding="utf-8"))
    mutate(data)
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.update(bogus=1), "bogus"),
        (lambda d: d.pop("win_points"), "win_points"),
        (lambda d: d.update(win_points=0), "win_points"),
        (lambda d: d.update(win_points="4"), "win_points"),
        (lambda d: d["finish_points"].pop("xtreme"), "xtreme"),
        (lambda d: d["finish_points"].update(bogus=1), "bogus"),
        (lambda d: d["finish_points"].update(spin=-1), "spin"),
        (lambda d: d["launch_check"].update(repeat_miss_penalty="banana"), "repeat_miss_penalty"),
        (lambda d: d["launch_check"].update(stadium_touch="explode"), "stadium_touch"),
        (lambda d: d["launch_check"].pop("mid_air_collision"), "mid_air_collision"),
        (lambda d: d["draw"].update(action="score"), "draw"),
        (lambda d: d.update(draw={}), "draw"),
        (lambda d: d.update(rules_version=3), "rules_version"),
    ],
)
def test_invalid_rules_fail_loudly(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        load_rules(_write(tmp_path, mutate))


def test_non_mapping_rules_fail(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_rules(p)


# ---- R52: manual match boundary ---------------------------------------------


def _start(r):
    r.match_start = True
    return r


def test_manual_boundary_closes_unfinished_match(rules):
    rs = [rr("spin", "X", round=1), rr("spin", "X", round=2),
          _start(rr("spin", "B", round=3)), rr("over", "B", round=4)]
    ms = group_matches(rs, rules)
    assert [m.score for m in ms] == [{"X": 2, "B": 0}, {"X": 0, "B": 3}]
    assert ms[0].winner is None and [r.round for r, _ in ms[1].rounds] == [3, 4]


def test_removing_manual_boundary_restores_automatic_grouping(rules):
    rs = [rr("spin", "X", round=1), rr("spin", "X", round=2),
          rr("spin", "B", round=3), rr("over", "B", round=4)]
    (m,) = group_matches(rs, rules)
    assert m.score == {"X": 2, "B": 3}


def test_win_points_still_ends_match_before_a_manual_boundary(rules):
    # X reaches 4 in round 2; a boundary on round 3 changes nothing, and a boundary
    # can never merge round 3 back into the completed match.
    rs = [rr("xtreme", "X", round=1), rr("spin", "X", round=2),
          _start(rr("spin", "B", round=3)), rr("spin", "B", round=4)]
    ms = group_matches(rs, rules)
    assert [m.score for m in ms] == [{"X": 4, "B": 0}, {"X": 0, "B": 2}]
    assert ms[0].winner == "X"
    rs[2].match_start = False
    assert [m.score for m in group_matches(rs, rules)] == [{"X": 4, "B": 0}, {"X": 0, "B": 2}]


def test_manual_boundary_resets_both_miss_counters(rules):
    rules.launch_check["repeat_miss_penalty"] = "opponent_1pt_on_2nd_miss"
    miss_x = {"touch": {"X": "yes"}}
    miss_b = {"touch": {"B": "yes"}}
    rs = [rr("spin", "B", **miss_x, round=1), rr("spin", "X", **miss_b, round=2),
          _start(rr("spin", "B", **miss_x, round=3)), rr("spin", "X", **miss_b, round=4)]
    ms = group_matches(rs, rules)
    assert len(ms) == 2
    # without the reset both second misses would award a penalty point
    assert [o.action for _, o in ms[1].rounds] == ["redo", "redo"]
    assert ms[1].score == {"X": 0, "B": 0}


def test_manual_boundary_on_first_round_is_harmless(rules):
    (m,) = group_matches([_start(rr("spin", "X", round=1)), rr("spin", "X", round=2)], rules)
    assert m.score == {"X": 2, "B": 0}
