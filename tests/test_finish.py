from dataclasses import asdict, replace

import numpy as np
import pytest
import yaml

from bxjudge.config import load_thresholds
from bxjudge.finish import (
    AFTER_FINISH_PREFIX, detect_finish, id_swap_risk, orbit_direction, rotation_rate,
    spin_activity, stop_frame,
)
from tests.helpers import (
    circle_path, make_calibration, make_track, rotating_crops, rr, scene, slowing_bey,
)


def test_spin_activity_uses_signed_safe_pixel_differences_and_observed_pairs():
    crops = {0: np.full((64, 64), 250, np.uint8),
             1: np.full((64, 64), 5, np.uint8),
             3: np.zeros((64, 64), np.uint8)}
    activity = spin_activity(crops)
    assert np.isnan(activity[0])
    assert activity[1] == 245.0
    assert np.isnan(activity[3])


@pytest.mark.parametrize('speed', [5, -5])
def test_rotation_rate_sign_and_scale(speed):
    rate = rotation_rate(rotating_crops(speed))
    assert np.isnan(rate[0])
    assert all(0.7 < v / speed < 1.3 for f, v in rate.items() if f > 0)


def test_rotation_rate_missing_frames_are_unknown():
    crops = rotating_crops()
    del crops[4]
    assert np.isnan(rotation_rate(crops)[5])


@pytest.mark.parametrize('direction', [4, -4])
def test_orbit_direction_configured_proxy(direction, th):
    track = make_track(1, 'bey', circle_path(deg_per_frame=direction))
    assert orbit_direction(track, make_calibration(), 0, th, 60) == (
        th.orbit_to_spin['cw' if direction > 0 else 'ccw'])


def test_orbit_insufficient_inconsistent_missing_and_no_landing_are_unknown(th):
    short = make_track(1, 'bey', circle_path(deg_per_frame=1, n=60))
    assert orbit_direction(short, make_calibration(), 0, th, 60) == 'unknown'
    centers = circle_path(deg_per_frame=4)
    centers.update({f: centers[119 - f] for f in range(60, 120)})
    mixed = make_track(1, 'bey', centers)
    assert orbit_direction(mixed, make_calibration(), 0, th, 60) == 'unknown'
    track = make_track(1, 'bey', circle_path())
    track.dets = [d for d in track.dets if d.frame != 50]
    assert orbit_direction(track, make_calibration(), 0, th, 60) == 'unknown'
    assert orbit_direction(short, make_calibration(), None, th, 60) == 'unknown'


def test_orbit_uses_actual_centroid_and_only_post_landing_window(th):
    c = make_calibration()
    c.battle_area = [(100, 100), (500, 100), (500, 300), (100, 300)]
    centers = circle_path(center=(300, 200), deg_per_frame=4)
    centers.update({f: (300, 200) for f in range(120, 150)})
    track = make_track(1, 'bey', centers)
    assert orbit_direction(track, c, 0, th, 60) == th.orbit_to_spin['cw']
    assert orbit_direction(track, c, 120, th, 60) == 'unknown'
    assert orbit_direction(track, c, 0, replace(th, spin_dir_window_s=0.5), 60) == 'unknown'


def test_stop_by_reversal_when_slow(th):
    assert stop_frame(*slowing_bey(reverse_at=200), 'right', th) == (200, 'reversed')


def test_fast_sign_flips_ignored(th):
    assert stop_frame(*slowing_bey(aliasing_flips_at=[50, 90]), 'right', th) == (None, None)


def test_still_detected_without_requiring_rotation_rate(th):
    track, activity, _ = slowing_bey(still_at=260)
    assert stop_frame(track, activity, {}, 'right', th) == (260, 'still')


def test_unknown_baseline_learns_only_stable_slow_direction(th):
    assert stop_frame(*slowing_bey(reverse_at=200), 'unknown', th) == (200, 'reversed')


def test_stop_mapping_and_move_limits_are_configured(th):
    track, activity, rate = slowing_bey(reverse_at=200)
    swapped = replace(th, orbit_to_spin={'cw': 'right', 'ccw': 'left'})
    assert stop_frame(track, activity, rate, 'right', swapped) == (180, 'reversed')
    moving = make_track(1, 'bey', {f: (100 + 4 * f, 180) for f in range(320)})
    assert stop_frame(moving, activity, rate, 'right', th) == (None, None)
    assert stop_frame(track, activity, rate, 'right',
                      replace(th, reverse_max_rate_deg=5)) == (None, None)


@pytest.mark.parametrize('reason', ['still', 'reversed'])
def test_stop_confirmation_cannot_bridge_observation_gaps(reason, th):
    n = th.still_frames if reason == 'still' else th.reverse_frames
    frames = [0] + list(range(1, n)) + list(range(n + 1, 2 * n + 2))
    track = make_track(1, 'bey', {f: (250, 180) for f in frames})
    activity = {f: 0.0 if reason == 'still' else 10.0 for f in frames}
    rate = {f: -5.0 if reason == 'still' else 5.0 for f in frames}
    assert stop_frame(track, activity, rate, 'right', th) == (n + 2, reason)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_stop_evidence_breaks_confirmation(value, th):
    track = make_track(1, 'bey', {f: (250, 180) for f in range(40)})
    activity = {f: 0.0 for f in range(40)}
    activity[15] = value
    assert stop_frame(track, activity, {}, 'unknown', th) == (None, None)
    rate = {f: 5.0 for f in range(10)}
    rate[4] = value
    assert stop_frame(track, {f: 10.0 for f in range(40)}, rate, 'right', th) == (None, None)


def test_stop_returns_earliest_start_even_if_it_confirms_later(th):
    track, activity, rate = slowing_bey(reverse_at=200, still_at=190)
    rate.update({f: 5.0 for f in range(200, 320)})
    assert stop_frame(track, activity, rate, 'right', th) == (190, 'still')


@pytest.mark.parametrize(('kwargs', 'finish', 'loser', 'frame'), [
    ({'b_path_into': 'xtreme'}, 'xtreme', 'B', 100),
    ({'x_path_into': 'over_left'}, 'over', 'X', 100),
    ({'b_path_into': 'over_right'}, 'over', 'B', 100),
    ({'b_burst_at': 80}, 'burst', 'B', 80),
    ({'stops': {'X': 300, 'B': 250}}, 'spin', 'B', 250),
])
def test_finish_events(kwargs, finish, loser, frame, th):
    result = detect_finish(*scene(**kwargs), th, 60)
    assert (result.finish, result.loser, result.frame) == (finish, loser, frame)
    assert result.confidence == pytest.approx(0.9)


def test_earliest_event_wins(th):
    result = detect_finish(*scene(b_path_into='xtreme', b_enter_at=100,
                                  stops={'X': 200, 'B': None}), th, 60)
    assert (result.finish, result.loser, result.frame) == ('xtreme', 'B', 100)


@pytest.mark.parametrize('kwargs', [
    {'stops': {'X': 250, 'B': 252}},
    {'x_path_into': 'over_left', 'b_path_into': 'xtreme', 'b_enter_at': 102},
])
def test_simultaneous_events_are_draw(kwargs, th):
    result = detect_finish(*scene(**kwargs), th, 60)
    assert result.finish is None and result.loser is None
    assert 'draw' in result.reasons


def test_simultaneous_window_is_configured(th):
    result = detect_finish(*scene(stops={'X': 250, 'B': 252}),
                           replace(th, simultaneous_stop_frames=1), 60)
    assert (result.finish, result.loser) == ('spin', 'X')


def test_id_swap_risk_lowers_confidence_and_requires_consecutive_frames(th):
    args = scene(overlap_frames=5, stops={'X': 300, 'B': 250})
    assert id_swap_risk(args[0]['X'], args[0]['B'], th)
    result = detect_finish(*args, th, 60)
    assert 'id_swap_risk' in result.reasons and result.confidence < th.review_confidence
    assert result.confidence == pytest.approx(0.45)
    x = make_track(1, 'bey', {f: (300, 180) for f in [0, 2, 4, 6]})
    b = make_track(2, 'bey', {f: (300, 180) for f in [0, 2, 4, 6]})
    assert not id_swap_risk(x, b, th)


def test_pocket_requires_whole_top_not_center_and_respects_concave_boundary(th):
    bey, parts, stops, c = scene()
    bey['B'] = make_track(2, 'bey', {f: (285, 310) for f in range(10)})
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish is None and 'pocket_entry_unconfirmed@B' in result.reasons
    c.xtreme = [(280, 290), (360, 290), (360, 340), (325, 340),
                (325, 310), (315, 310), (315, 340), (280, 340)]
    bey['B'] = make_track(2, 'bey', {f: (320, 310) for f in range(10)}, size=40)
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish is None


def test_rotating_return_cancels_exit(th):
    bey, parts, stops, c = scene(b_path_into='xtreme')
    bey['B'].dets = [d for d in bey['B'].dets if d.frame < 110]
    bey['B'].dets.extend(make_track(2, 'bey', {f: (450 - f, 180)
                                            for f in range(110, 150)}).dets)
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish is None and 'exit_returned@B' in result.reasons


def test_return_then_later_final_exit_uses_new_entry(th):
    bey, parts, stops, c = scene(b_path_into='xtreme', b_enter_at=200)
    for d in bey['B'].dets:
        if 100 <= d.frame < 110:
            d.box = (310, 310, 330, 330)
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert (result.finish, result.frame) == ('xtreme', 200)


def test_burst_requires_new_nearby_part_after_track_disappearance(th):
    bey, parts, stops, c = scene(b_burst_at=80)
    parts = [make_track(3, 'bey_part', {80: (100, 100)})]
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish is None and 'track_lost@B' in result.reasons
    parts = [make_track(3, 'bey_part', {110: (450, 180)})]
    assert detect_finish(bey, parts, stops, c, th, 60).finish is None
    bey, _, stops, c = scene()
    parts = [make_track(3, 'bey_part', {80: (450, 180)})]
    assert detect_finish(bey, parts, stops, c, th, 60).finish is None


def test_prior_unexplained_disappearance_prevents_later_spin_claim(th):
    bey, parts, stops, c = scene(stops={'X': 200, 'B': None})
    bey['B'].dets = [d for d in bey['B'].dets if d.frame < 80]
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish is None and 'track_lost@B' in result.reasons
    assert result.confidence < th.review_confidence


def test_missing_side_and_no_event_have_explicit_uncertainty(th):
    bey, parts, stops, c = scene()
    del bey['B']
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish is None and 'track_missing@B' in result.reasons
    result = detect_finish(*scene(), th, 60)
    assert result.finish is None and result.confidence == 0
    assert 'finish_unknown' in result.reasons


def test_track_gaps_lower_finish_confidence(th):
    bey, parts, stops, c = scene(stops={'X': 300, 'B': 250})
    bey['B'].dets = [d for d in bey['B'].dets if d.frame != 20]
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish == 'spin'
    assert result.confidence < th.review_confidence
    assert 'track_evidence_gap@B' in result.reasons


def test_orbit_mapping_is_validated_and_spin_flag_serializes(tmp_path):
    with open('config/thresholds.yaml', encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    data['orbit_to_spin'] = {'cw': 'right', 'ccw': 'right'}
    path = tmp_path / 'thresholds.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    with pytest.raises(ValueError, match='orbit_to_spin'):
        load_thresholds(path)
    result = rr('spin', 'X')
    result.spin_dir = {'X': 'right', 'B': 'unknown'}
    assert asdict(result)['spin_dir'] == {'X': 'right', 'B': 'unknown'}


def test_earlier_ambiguous_pocket_entry_blocks_later_known_finish(th):
    bey, parts, stops, c = scene(x_path_into='over_left', x_enter_at=100)
    for d in bey['B'].dets:
        if d.frame >= 80:
            d.box = (275, 300, 295, 320)  # center in xtreme; left edge still outside
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish is None and 'pocket_entry_unconfirmed@B' in result.reasons
    assert result.confidence < th.review_confidence


def test_offscreen_final_positions_are_unknown_with_reasons(th):
    bey, parts, stops, c = scene()
    for track in bey.values():
        for d in track.dets:
            if d.frame >= 80:
                d.box = (-100, -100, -80, -80)
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish is None and result.confidence == 0
    assert 'position_unknown@X' in result.reasons and 'position_unknown@B' in result.reasons


@pytest.mark.parametrize('mapping', [
    {'cw': ['left'], 'ccw': 'right'}, {'cw': 'left'}, ['left', 'right'],
])
def test_malformed_orbit_mapping_has_clear_config_error(mapping, tmp_path):
    with open('config/thresholds.yaml', encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    data['orbit_to_spin'] = mapping
    path = tmp_path / 'thresholds.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    with pytest.raises(ValueError, match='orbit_to_spin'):
        load_thresholds(path)


def test_burst_timing_radius_and_evidence_score(th):
    bey, _, stops, c = scene(b_burst_at=80)
    # Last top frame79 -> part97 is exactly the configured 18-frame window.
    parts = [make_track(3, 'bey_part', {97: (490, 180)}, score=0.5)]
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert (result.finish, result.loser, result.frame) == ('burst', 'B', 97)
    assert result.confidence == pytest.approx(0.7)
    assert detect_finish(bey, parts, stops, c, replace(th, burst_part_radius_px=39), 60).finish is None
    assert detect_finish(bey, parts, stops, c, replace(th, burst_part_window_s=0.25), 60).finish is None


def test_ambiguous_burst_side_is_unknown(th):
    bey = {'X': make_track(1, 'bey', {f: (280, 180) for f in range(80)}),
           'B': make_track(2, 'bey', {f: (320, 180) for f in range(80)})}
    parts = [make_track(3, 'bey_part', {80: (300, 180)})]
    result = detect_finish(bey, parts, {}, make_calibration(), th, 60)
    assert result.finish is None and 'burst_side_unknown' in result.reasons


def test_id_swap_exact_run_length_and_confidence_factor_are_configured(th):
    bey, parts, stops, c = scene(overlap_frames=3, stops={'X': 300, 'B': 250})
    assert id_swap_risk(bey['X'], bey['B'], th)
    assert not id_swap_risk(bey['X'], bey['B'], replace(th, id_swap_overlap_frames=4))
    result = detect_finish(bey, parts, stops, c, replace(th, id_swap_confidence_factor=0.2), 60)
    assert result.confidence == pytest.approx(0.18)


@pytest.mark.parametrize(('box', 'reason'), [
    ((-100, -100, -80, -80), 'position_unknown@B'),
    ((275, 300, 295, 320), 'pocket_entry_unconfirmed@B'),
])
def test_ambiguous_interval_before_final_whole_pocket_blocks_intervening_finish(box, reason, th):
    bey, parts, stops, c = scene(x_path_into='over_left', x_enter_at=100,
                                b_path_into='xtreme', b_enter_at=150)
    for d in bey['B'].dets:
        if 80 <= d.frame < 150:
            d.box = box
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish is None and result.loser is None and result.frame is None
    assert reason in result.reasons and 'finish_unknown' in result.reasons
    assert result.confidence == 0


@pytest.mark.parametrize('box', [
    (-100, -100, -80, -80), (275, 300, 295, 320),
])
def test_observed_whole_battle_return_cancels_earlier_ambiguous_interval(box, th):
    bey, parts, stops, c = scene(x_path_into='over_left', x_enter_at=100,
                                b_path_into='xtreme', b_enter_at=150)
    for d in bey['B'].dets:
        if 80 <= d.frame < 90:
            d.box = box
    # Frames90..149 visibly place B's whole box back on the battle floor.
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert (result.finish, result.loser, result.frame) == ('over', 'X', 100)
    assert result.confidence == pytest.approx(0.9)
    assert 'position_unknown@B' not in result.reasons
    assert 'pocket_entry_unconfirmed@B' not in result.reasons


def test_initial_outside_launch_is_resolved_by_observed_battle_entry(th):
    bey, parts, stops, c = scene(x_path_into='over_left', x_enter_at=100,
                                b_path_into='xtreme', b_enter_at=150)
    for d in bey['B'].dets:
        if d.frame < 5:
            d.box = (-100, -100, -80, -80)
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert (result.finish, result.loser, result.frame) == ('over', 'X', 100)
    assert result.confidence == pytest.approx(0.9)
    assert 'position_unknown@B' not in result.reasons


def test_center_only_battle_return_does_not_resolve_exit_uncertainty(th):
    bey, parts, stops, c = scene(x_path_into='over_left', x_enter_at=100,
                                b_path_into='xtreme', b_enter_at=150)
    for d in bey['B'].dets:
        if 80 <= d.frame < 90:
            d.box = (-100, -100, -80, -80)
        elif 90 <= d.frame < 150:
            d.box = (115, 170, 135, 190)  # center inside floor, whole body unproven
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish is None and 'position_unknown@B' in result.reasons


# R55: finish evidence is evaluated up to the decisive event.
assert AFTER_FINISH_PREFIX == 'after_finish:'


def _triggers(result):
    return [r for r in result.reasons if not r.startswith(AFTER_FINISH_PREFIX)]


def _gradual_b_into_xtreme(n=340):
    b = {f: (320, 180) for f in range(100)}
    b.update({f: (320, min(320, 270 + 5 * (f - 100))) for f in range(100, n)})
    return b


def _gradual_x_into_over_left(n=340):
    x = {f: (180, 180) for f in range(100)}
    x.update({f: (max(75, 180 - 4 * (f - 100)), min(250, 180 + 3 * (f - 100)))
              for f in range(100, n)})
    return x


@pytest.mark.parametrize(('side', 'finish', 'frame'), [('B', 'xtreme', 106), ('X', 'over', 120)])
def test_gradual_pocket_slide_transit_belongs_to_own_entry(side, finish, frame, th):
    bey, parts, stops, c = scene()
    if side == 'B':
        bey['B'] = make_track(2, 'bey', _gradual_b_into_xtreme())
    else:
        bey['X'] = make_track(1, 'bey', _gradual_x_into_over_left())
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert (result.finish, result.loser, result.frame) == (finish, side, frame)
    assert result.confidence == pytest.approx(0.9)
    assert result.reasons == []


def test_gradual_entry_transit_still_blocks_other_side_event_inside_it(th):
    bey, parts, stops, c = scene(x_path_into='over_left', x_enter_at=105)
    bey['B'] = make_track(2, 'bey', _gradual_b_into_xtreme())
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert result.finish is None and result.loser is None
    assert 'pocket_entry_unconfirmed@B' in result.reasons
    assert 'finish_unknown' in result.reasons


def test_loser_still_in_pocket_after_finish_is_informational(th):
    result = detect_finish(*scene(b_path_into='xtreme', stops={'X': None, 'B': 110}), th, 60)
    assert (result.finish, result.loser, result.frame) == ('xtreme', 'B', 100)
    assert _triggers(result) == []
    assert AFTER_FINISH_PREFIX + 'stop_evidence_missing@B' in result.reasons
    assert result.confidence == pytest.approx(0.9)


def test_winner_track_ending_long_after_finish_is_informational(th):
    bey, parts, stops, c = scene(b_path_into='xtreme')
    bey['X'].dets = [d for d in bey['X'].dets if d.frame < 200]
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert (result.finish, result.loser, result.frame) == ('xtreme', 'B', 100)
    assert result.reasons == [AFTER_FINISH_PREFIX + 'track_lost@X']


def test_beys_lifted_after_spin_finish_are_informational(th):
    bey, parts, stops, c = scene(stops={'X': 330, 'B': 250})
    for track in bey.values():
        for d in track.dets:
            if d.frame >= 320:
                d.box = (d.box[0], -60, d.box[2], -40)
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert (result.finish, result.loser, result.frame) == ('spin', 'B', 250)
    assert _triggers(result) == []
    assert {AFTER_FINISH_PREFIX + r for r in
            ('position_unknown@X', 'position_unknown@B', 'stop_evidence_missing@X')} <= set(result.reasons)
    assert result.confidence == pytest.approx(0.9)


def test_winner_evidence_before_competing_stop_is_established_still_triggers(th):
    # A winner stop starting inside the draw window is only established
    # max(still_frames, reverse_frames) frames later; losing the winner before
    # then must stay a review trigger.
    cutoff = 250 + th.simultaneous_stop_frames + max(th.still_frames, th.reverse_frames) - 1
    for last, informational in ((cutoff, False), (cutoff + 1, True)):
        bey, parts, stops, c = scene(stops={'X': None, 'B': 250})
        bey['X'].dets = [d for d in bey['X'].dets if d.frame <= last]
        result = detect_finish(bey, parts, stops, c, th, 60)
        assert (result.finish, result.loser) == ('spin', 'B')
        expected = (AFTER_FINISH_PREFIX if informational else '') + 'track_lost@X'
        assert result.reasons == [expected]


def test_post_finish_gap_and_contact_do_not_lower_confidence(th):
    bey, parts, stops, c = scene(stops={'X': None, 'B': 250})
    bey['X'].dets = [d for d in bey['X'].dets if d.frame != 295]
    for d in bey['X'].dets:
        if 300 <= d.frame < 310:
            d.box = bey['B'].dets[d.frame].box  # post-finish box contact
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert (result.finish, result.loser, result.frame) == ('spin', 'B', 250)
    assert result.confidence == pytest.approx(0.9)
    assert set(result.reasons) == {AFTER_FINISH_PREFIX + 'track_evidence_gap@X',
                                   AFTER_FINISH_PREFIX + 'id_swap_risk'}


@pytest.mark.parametrize('lifted_box', [(310, 200, 330, 220), (310, -60, 330, -40)])
def test_non_rotating_removal_after_confirmed_pocket_finish_keeps_it(lifted_box, th):
    bey, parts, stops, c = scene(b_path_into='xtreme')
    for d in bey['B'].dets:
        if d.frame >= 330:
            d.box = lifted_box
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert (result.finish, result.loser, result.frame) == ('xtreme', 'B', 100)
    assert _triggers(result) == []
    assert result.confidence == pytest.approx(0.9)


def test_return_before_pocket_confirmation_still_cancels(th):
    confirm = round(th.pocket_confirm_s * 60)
    for stay, expected in ((confirm - 1, None), (confirm, 'xtreme')):
        bey, parts, stops, c = scene(b_path_into='xtreme')
        bey['B'].dets = [d for d in bey['B'].dets if d.frame < 100 + stay]
        bey['B'].dets.extend(make_track(2, 'bey', {f: (330, 200)
                                                for f in range(100 + stay, 340)}).dets)
        result = detect_finish(bey, parts, stops, c, th, 60)
        assert result.finish == expected
        if expected is None:
            assert 'exit_returned@B' in result.reasons
        else:
            assert result.loser == 'B' and _triggers(result) == []


@pytest.mark.parametrize('first', [79, 78])
def test_burst_part_overlapping_last_bey_frame_is_burst(first, th):
    bey, _, stops, c = scene(b_burst_at=80)  # B last frame 79
    parts = [make_track(3, 'bey_part', {first: (455, 185)})]
    result = detect_finish(bey, parts, stops, c, th, 60)
    assert (result.finish, result.loser, result.frame) == ('burst', 'B', first)
    assert result.reasons == []


def test_burst_overlap_tolerance_is_configured(th):
    bey, _, stops, c = scene(b_burst_at=80)
    tolerance = round(th.burst_part_overlap_s * 60)
    early = [make_track(3, 'bey_part', {79 - tolerance - 1: (455, 185)})]
    assert detect_finish(bey, early, stops, c, th, 60).finish is None
    edge = [make_track(3, 'bey_part', {79 - tolerance: (455, 185)})]
    assert detect_finish(bey, edge, stops, c, th, 60).finish == 'burst'
    same = [make_track(3, 'bey_part', {79: (455, 185)})]
    assert detect_finish(bey, same, stops, c, replace(th, burst_part_overlap_s=0), 60).finish == 'burst'
    assert detect_finish(bey, edge, stops, c, replace(th, burst_part_overlap_s=0), 60).finish is None
