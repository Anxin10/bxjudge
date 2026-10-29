"""Association repairs found on the task-6 rerun (task 7).

A slot seeded at launch can learn an appearance the spinning top never matches
again (214505: Bhattacharyya 0.95 for 800 frames). With exactly two tops per
round, a persistent sole leftover while the other slot is observed is that slot's
top in space only: its player stays unknown. A box mostly inside one observed top's
box (and no other), while both tops are observed, is a second box of that top.
"""
from pathlib import Path

from bxjudge.two_bey_tracking import TwoBeyTracker, load_policy

A, B, C = (1.0, 0.0), (0.0, 1.0), (0.05, 0.95)
POLICY = load_policy(Path("config/two_bey_tracking.yaml"))


def det(tid, x, color, score=0.9, y=10, w=10):
    return {"native_id": tid, "box": [x, y, x + w, y + w], "score": score,
            "appearance": list(color)}


def stale(gap=40):
    """Entity 1 (A at x=10) is seen once, then only entity 2 (B) for ``gap`` frames."""
    t = TwoBeyTracker((200, 100), 30, POLICY)
    t.update(0, [det(1, 10, A), det(2, 100, B)])
    for f in range(1, gap + 1):
        t.update(f, [det(2, 100, B)])
    return t


def entity_of(result, native):
    return {r["native_id"]: r["entity_id"] for r in result["observed"]}.get(native)


def test_policy_names_the_new_values():
    assert POLICY["duplicate_containment"] == 0.8
    assert POLICY["reacquire_min_gap_s"] == 1.0
    assert POLICY["reacquire_confirm_frames"] == 5
    assert POLICY["reacquire_min_score"] == 0.5


def test_a_persistent_sole_leftover_is_reacquired_by_the_stale_slot():
    t = stale()
    rows = [t.update(f, [det(2, 100, B), det(7, 40, C)]) for f in range(41, 47)]
    # Four frames stay audited as unassigned; the fifth confirms elimination.
    assert [entity_of(r, 7) for r in rows] == [None, None, None, None, 1, 1]
    assert all(r["unassigned"][0]["native_id"] == 7 for r in rows[:4])
    assert t.reacquisitions == [
        {"frame": 45, "entity_id": 1, "native_id": 7, "unobserved_frames": 45}
    ]
    # Re-seeded appearance keeps tracking it without elimination.
    later = t.update(47, [det(7, 42, C)])
    assert entity_of(later, 7) == 1


def test_a_recently_seen_slot_is_not_reacquired_by_elimination():
    t = stale(gap=10)
    rows = [t.update(f, [det(2, 100, B), det(7, 40, C)]) for f in range(11, 20)]
    assert all(entity_of(r, 7) is None for r in rows)
    assert t.reacquisitions == []


def test_elimination_needs_exactly_one_disjoint_confident_leftover():
    t = stale()
    two = [t.update(f, [det(2, 100, B), det(7, 40, C), det(8, 150, C)]) for f in range(41, 50)]
    assert all(entity_of(r, 7) is None and entity_of(r, 8) is None for r in two)
    t = stale()
    weak = [t.update(f, [det(2, 100, B), det(7, 40, C, score=0.3)]) for f in range(41, 50)]
    assert all(entity_of(r, 7) is None for r in weak)
    t = stale()
    touching = [t.update(f, [det(2, 100, B), det(7, 105, C)]) for f in range(41, 50)]
    assert all(entity_of(r, 7) is None for r in touching)
    assert t.reacquisitions == []


def test_an_interrupted_leftover_restarts_its_confirmation():
    t = stale()
    for f in range(41, 45):
        t.update(f, [det(2, 100, B), det(7, 40, C)])
    t.update(45, [det(2, 100, B)])
    rows = [t.update(f, [det(2, 100, B), det(7, 40, C)]) for f in range(46, 51)]
    assert [entity_of(r, 7) for r in rows] == [None, None, None, None, 1]


def test_a_box_inside_an_observed_top_is_a_duplicate_when_both_tops_are_seen():
    t = TwoBeyTracker((200, 100), 30, POLICY)
    t.update(0, [det(1, 10, A), det(2, 100, B)])
    inner = {"native_id": 3, "box": [101, 11, 108, 19], "score": 0.8, "appearance": list(B)}
    result = t.update(1, [det(1, 10, A), det(2, 100, B), inner])
    [rejected] = result["unassigned"]
    assert rejected["reason"] == "duplicate_box"
    assert rejected["duplicate_kind"] == "contained"
    assert rejected["duplicate_of"] == 2
    # Containment says nothing about which player owns the extra native ID.
    assert 3 not in t.native_owners and 3 not in t.native_anchor_owner


def test_a_contained_box_stays_unassigned_while_the_other_top_is_unseen():
    t = TwoBeyTracker((200, 100), 30, POLICY)
    t.update(0, [det(1, 10, A), det(2, 100, B)])
    inner = {"native_id": 3, "box": [101, 11, 108, 19], "score": 0.8, "appearance": list(B)}
    result = t.update(1, [det(2, 100, B), inner])
    [rejected] = result["unassigned"]
    assert rejected["reason"] == "unassigned_two_target_capacity_or_gate"


def test_round_record_lists_reacquisitions():
    from bxjudge.two_bey_integration import reconstruct_round_tracks
    from bxjudge.types import Detection, Track

    def track(tid, frames, x, color):
        return Track(tid, "bey", [Detection(f, "bey", (x, 10, x + 10, 20), 0.9) for f in frames])

    tracks = [track(1, [0, 1], 10, A), track(2, range(60), 100, B), track(7, range(41, 60), 40, C)]
    colors = {1: A, 2: B, 7: C}
    features = {(d.frame, d.box): list(colors[tr.track_id]) for tr in tracks for d in tr.dets}
    result = reconstruct_round_tracks(tracks, 0, 59, 30, (200, 100), features, POLICY, 100)
    assert result.record["reacquisitions"] == [
        {"frame": 45, "entity_id": 1, "native_id": 7, "unobserved_frames": 44}
    ]
    # The eliminated native is tracked in space but never anchors the slot's player.
    assert result.members[101] == [1]


# Codex task-7 check (2026-10-28): elimination is spatial evidence only.
def test_reacquisition_never_restores_a_valid_player_identity():
    t = stale()
    rows = [t.update(f, [det(2, 100, B), det(7, 40, B)]) for f in range(41, 47)]
    got = [r for r in rows[4]["observed"] if r["native_id"] == 7]
    assert got and got[0]["entity_id"] == 1
    # Appearance B contradicts entity 1's A: the player stays unknown from here on.
    assert got[0]["identity_valid"] is False
    assert t.identity_unknown_since == {1: 45}
    # The contradicting clean template is kept, never overwritten by elimination.
    assert list(t.slots[0]["clean_appearance"]) == list(A)


def test_reacquisition_counts_only_consecutive_source_frames():
    t = stale()
    rows = [t.update(f, [det(2, 100, B), det(7, 40, C)]) for f in (41, 51, 61, 71, 81)]
    assert all(entity_of(r, 7) is None for r in rows)
    assert t.reacquisitions == []


def test_a_box_spanning_both_observed_tops_is_not_a_duplicate():
    t = TwoBeyTracker((200, 100), 30, POLICY)
    t.update(0, [det(1, 10, A), det(2, 100, B)])
    wide = {"native_id": 3, "box": [5, 5, 150, 50], "score": 0.8, "appearance": list(B)}
    result = t.update(1, [det(1, 10, A), det(2, 100, B), wide])
    [rejected] = result["unassigned"]
    assert rejected["reason"] == "unassigned_two_target_capacity_or_gate"


def test_a_small_box_containing_part_of_an_observed_top_is_not_a_duplicate():
    # Only the candidate's own area decides containment, not the observed box's.
    t = TwoBeyTracker((200, 100), 30, POLICY)
    t.update(0, [det(1, 10, A), det(2, 100, B)])
    around = {"native_id": 3, "box": [95, 5, 140, 40], "score": 0.8, "appearance": list(B)}
    result = t.update(1, [det(1, 10, A), det(2, 100, B), around])
    [rejected] = result["unassigned"]
    assert rejected["reason"] == "unassigned_two_target_capacity_or_gate"


# Codex task-7 re-review (2026-10-29): spatial ownership must not import player anchors.
def test_a_reacquired_native_never_joins_player_anchor_membership():
    t = stale()
    for f in range(41, 51):
        t.update(f, [det(2, 100, B), det(7, 40, C)])
    assert t.native_owners[7] == 1
    # Membership feeds the pipeline's X/B anchor for the whole physical track.
    assert 7 not in t.native_anchor_owner


def test_a_native_joining_an_identity_unknown_entity_never_anchors_it():
    t = stale()
    for f in range(41, 46):
        t.update(f, [det(2, 100, B), det(7, 40, C)])
    # The reseeded template now follows native 9 with no identity evidence either.
    rows = [t.update(f, [det(2, 100, B), det(9, 42, C)]) for f in range(46, 50)]
    assert entity_of(rows[-1], 9) == 1
    assert 9 not in t.native_anchor_owner


def test_a_box_owned_by_the_other_top_is_never_hidden_as_a_duplicate():
    t = TwoBeyTracker((200, 100), 30, POLICY)
    t.update(0, [det(1, 10, A), det(2, 100, B)])
    t.update(1, [det(3, 10, A), det(2, 100, B)])
    assert t.native_owners[1] == 1
    # Native 1 still belongs to entity 1 but sits inside entity 2's box.
    inner = {"native_id": 1, "box": [101, 11, 108, 19], "score": 0.8, "appearance": list(A)}
    result = t.update(2, [det(3, 10, A), det(2, 100, B), inner])
    [rejected] = result["unassigned"]
    assert rejected["reason"] == "unassigned_two_target_capacity_or_gate"
