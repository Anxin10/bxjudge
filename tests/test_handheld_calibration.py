"""Handheld camera: periodic keypoint re-finds confirm the follower (plan
2026-11-12-dynamic-calibration, Task 5)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from bxjudge.dynamic_calibration import read_geometry, recorded_calibration, save_geometry
from bxjudge.handheld_calibration import HandheldCalibration
from bxjudge.stadium_locator import LocatedStadium, alignment_from_homography
from bxjudge.trajectory_video import load_video_policy
from tests.helpers import stadium_scene

SIZE = (640, 360)
POLICY = {"refind_s": 1.0, "confirm_px": 12.0, "max_unchecked_s": 3.0}


@lru_cache(maxsize=None)
def _scene(dx: float):
    img, _ = stadium_scene(size=SIZE, shift=(dx, 0.0))
    img.setflags(write=False)
    return img


def _calib():
    return stadium_scene(size=SIZE)[1]


def _t(dx):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class FakeLocator:
    """Keypoint locator stand-in driven by the frame the test is at: the true camera shift
    (``truth(frame)``), a wrong one for frames in ``wrong_at``, or nothing for frames in
    ``missing`` / always when ``truth`` is None."""

    def __init__(self, c, truth=None, wrong_at=None, missing=(), reason="no_keypoints"):
        self.c, self.truth, self.wrong_at = c, truth, wrong_at or {}
        self.missing, self.reason = set(missing), reason
        self.frame = None
        self.calls: list[int] = []
        self.last_reason = None

    def locate(self, image):
        self.calls.append(self.frame)
        if self.truth is None or self.frame in self.missing:
            self.last_reason = self.reason
            return None
        dx = self.wrong_at.get(self.frame, 0.0) + self.truth(self.frame)
        self.last_reason = None
        h = _t(dx)
        return LocatedStadium(h, alignment_from_homography(self.c, h, 0.9), None)


def run(shifts, locator_kw=None, *, fps=30, policy=POLICY, start_confirmed=True, finish=True,
        start_frame=None):
    """Feed frames whose stadium is shifted by ``shifts[f]`` px; the fake locator knows them."""
    c = _calib()
    kw = {"truth": lambda f: shifts[f]} if locator_kw is None else locator_kw
    loc = FakeLocator(c, **kw)
    g = HandheldCalibration(c, _scene(0.0), fps, load_video_policy()["stadium_follow"], policy,
                            loc, minimum_visible=0.6, start_confirmed=start_confirmed,
                            start_frame=start_frame)
    g.set_foreground([])
    for f, dx in enumerate(shifts):
        loc.frame = f
        g.update(f, _scene(dx))
    if finish:
        g.finish()
    return g, loc


def pan(n, speed=0.5):
    return [round(speed * f, 1) for f in range(n)]


def refind_frames(g):
    return [r["frame"] for r in g.refinds]


def test_follows_a_slow_pan_and_refinds_reset_the_anchor():
    shifts = pan(100)
    g, _ = run(shifts)
    assert all(g.samples[f].alignment is not None for f in range(100))
    assert [s["kind"] for s in g.segments] == ["recorded_snapshot"]
    assert [r["result"] for r in g.refinds] == ["agree"] * 4  # 30, 60, 90 and the tail 99
    for f in (10, 50, 95):
        h = np.array(g.samples[f].alignment.homography)
        assert abs(h[0, 2] - shifts[f]) < 1.0
    assert [g.samples[f].status for f in (30, 31)] == ["refound", "tracking"]


def test_disagreeing_refind_invalidates_back_to_the_last_confirmation():
    shifts = pan(130)
    c = _calib()
    g, _ = run(shifts, {"truth": lambda f: shifts[f], "wrong_at": {60: 50.0}})
    assert [g.samples[f].status for f in (30, 31, 59, 60)] == \
        ["refound", "invalidated", "invalidated", "invalidated"]
    assert g.samples[30].alignment is not None
    assert all(g.samples[f].alignment is None for f in range(31, 120))
    assert [r["result"] for r in g.refinds[:4]] == ["agree", "disagree", "provisional", "agree"]
    assert g.refinds[1]["disagreement_px"] > 12.0
    # Two refinds that agree through the follower open a new segment at the second.
    assert g.segments[-1]["kind"] == "refound" and g.segments[-1]["start_frame"] == 120
    assert g.segments[-1]["confirmed_frames"] == [90, 120]
    assert g.samples[120].status == "refound" and g.samples[121].alignment is not None
    assert c.frame_size == SIZE


def test_whip_pan_stays_unknown_until_a_refind_confirms():
    shifts = [0.0] * 45 + [60.0] * 105  # 60 px at 640 wide: beyond the follower's 5 % bound
    g, _ = run(shifts)
    assert g.samples[45].status == "lost"
    assert all(g.samples[f].alignment is None for f in range(45, 90))
    assert g.refinds[1]["frame"] == 60 and g.refinds[1]["result"] == "provisional"
    assert g.samples[90].status == "refound" and g.segments[-1]["start_frame"] == 90
    assert abs(np.array(g.samples[100].alignment.homography)[0, 2] - 60.0) < 1.0


def test_refind_interval_uses_playback_fps():
    g, _ = run(pan(130, 0.1), fps=30, finish=False)
    assert refind_frames(g) == [30, 60, 90, 120]
    g, _ = run(pan(250, 0.05), fps=120, finish=False)
    assert refind_frames(g) == [120, 240]


def test_refind_error_is_recorded_and_tracking_continues():
    shifts = pan(70, 0.1)
    g, _ = run(shifts, {"truth": lambda f: shifts[f], "missing": {30},
                        "reason": "keypoint_error: RuntimeError: CUDA out of memory"})
    assert g.refinds[0] == {"frame": 30, "result": "no_fit", "disagreement_px": None,
                            "reason": "keypoint_error: RuntimeError: CUDA out of memory"}
    assert all(g.samples[f].alignment is not None for f in range(70))
    assert g.refinds[1]["result"] == "agree"


def test_long_unchecked_tracking_stops_being_emitted():
    g, _ = run(pan(130, 0.1), {"truth": None})
    assert all(g.samples[f].alignment is not None for f in range(91))
    assert all(g.samples[f].alignment is None for f in range(91, 130))
    assert g.record()["samples"][100]["rejection_reason"] == "unchecked_too_long"
    assert {r["result"] for r in g.refinds} == {"no_fit"}


def test_finish_checks_the_tail():
    shifts = pan(100)
    g, _ = run(shifts, {"truth": lambda f: shifts[f], "wrong_at": {99: 40.0}})
    assert refind_frames(g) == [30, 60, 90, 99]
    assert g.samples[90].status == "refound"
    assert all(g.samples[f].status == "invalidated" for f in range(91, 100))
    g, _ = run(pan(90))  # the last frame (89) is not a refind frame either; it agrees
    assert g.refinds[-1]["frame"] == 89 and g.samples[89].status == "refound"


def test_single_fit_start_emits_nothing_until_a_refind_agrees():
    shifts = pan(70, 0.1)
    g, _ = run(shifts, start_confirmed=False, start_frame=12)
    assert g.segments[0]["kind"] == "unconfirmed_snapshot"
    assert all(g.samples[f].alignment is None for f in range(30))
    assert g.samples[30].status == "refound" and g.segments[-1]["start_frame"] == 30
    assert all(g.samples[f].alignment is not None for f in range(30, 70))


def test_confirm_threshold_is_in_inference_px():
    """12 px at the 1280 inference size is 6 px at 640: a 9 px refind error disagrees."""
    shifts = pan(40, 0.1)
    g, _ = run(shifts, {"truth": lambda f: shifts[f], "wrong_at": {30: 9.0}}, finish=False)
    assert g.refinds[0]["result"] == "disagree"


def test_record_round_trips_through_saved_geometry(tmp_path):
    shifts = pan(130)
    g, _ = run(shifts, {"truth": lambda f: shifts[f], "wrong_at": {60: 50.0}})
    data = {**g.record(), "calibration_id": "abc", "source_sha256": "src"}
    assert data["mode"] == "handheld" and data["temporal_policy"] is None
    assert [r["frame"] for r in data["refinds"]] == [30, 60, 90, 120, 129]
    evidence = save_geometry(tmp_path, "v.mp4", data)
    rec = read_geometry(tmp_path, {"geometry_evidence": evidence, "calibration_id": "abc"}, "src")
    assert rec["mode"] == "handheld"
    moved = recorded_calibration(_calib(), rec, 20, SIZE)
    assert moved is not None
    assert abs(np.array(moved.battle_area).mean(0)[0]
               - np.array(_calib().battle_area).mean(0)[0] - shifts[20]) < 1.0
    assert recorded_calibration(_calib(), rec, 45, SIZE) is None  # invalidated


def test_frames_must_increase_and_match_the_size():
    c = _calib()
    g = HandheldCalibration(c, _scene(0.0), 30, load_video_policy()["stadium_follow"], POLICY,
                            FakeLocator(c), minimum_visible=0.6)
    g.update(3, _scene(0.0))
    with pytest.raises(ValueError):
        g.update(3, _scene(0.0))
    with pytest.raises(ValueError):
        g.update(4, _scene(0.0)[:100])


# ---- Codex review R1 (Task 5) ---------------------------------------------------------


def test_a_contradiction_after_the_unchecked_timeout_still_invalidates():
    """Emitted frames stay open to contradiction after emission stopped for lack of checks."""
    shifts = pan(92, 0.1)
    g, _ = run(shifts, {"truth": lambda f: shifts[f], "missing": {30, 60, 90},
                        "wrong_at": {91: 50.0}})
    assert g.refinds[-1]["frame"] == 91 and g.refinds[-1]["result"] == "disagree"
    assert all(g.samples[f].alignment is None for f in range(92))
    assert g.samples[45].status == "invalidated"


def test_a_contradiction_after_recovery_from_a_loss_still_invalidates():
    """Lost, then the follower finds the same anchor again: a refind that contradicts it
    contradicts the frames it emitted before the loss as well."""
    shifts = [0.0] * 10 + [60.0] * 5 + [0.0] * 30
    g, _ = run(shifts, {"truth": lambda f: shifts[f], "wrong_at": {30: 50.0}}, finish=False)
    assert g.samples[10].status == "lost"
    assert g.refinds[0]["result"] == "disagree"
    assert all(g.samples[f].alignment is None for f in range(31))


def test_single_fit_start_cannot_confirm_itself_on_its_own_frame():
    shifts = pan(70, 0.1)
    c = _calib()
    loc = FakeLocator(c, truth=lambda f: shifts[f])
    g = HandheldCalibration(c, _scene(0.0), 30, load_video_policy()["stadium_follow"], POLICY,
                            loc, minimum_visible=0.6, start_confirmed=False, start_frame=30)
    g.set_foreground([])
    for f, dx in enumerate(shifts):
        loc.frame = f
        g.update(f, _scene(dx))
    assert g.refinds[0]["frame"] == 30 and g.refinds[0]["result"] == "same_frame"
    assert all(g.samples[f].alignment is None for f in range(60))
    assert g.samples[60].status == "refound"
    assert g.segments[-1]["confirmed_frames"] == [30, 60]


class _ImageLocator(FakeLocator):
    """Keeps a copy of every image it is given."""

    def __init__(self, c, **kw):
        super().__init__(c, **kw)
        self.images = []

    def locate(self, image):
        self.images.append(image.copy())
        return super().locate(image)


def test_finish_checks_the_frame_it_saw_not_a_reused_buffer():
    c = _calib()
    loc = _ImageLocator(c, truth=lambda f: 0.0)
    g = HandheldCalibration(c, _scene(0.0), 30, load_video_policy()["stadium_follow"], POLICY,
                            loc, minimum_visible=0.6)
    g.set_foreground([])
    buffer = _scene(0.0).copy()
    for f in range(5):
        loc.frame = f
        g.update(f, buffer)
    buffer[:] = 0  # the caller reuses its buffer
    g.finish()
    assert len(loc.images) == 1 and np.array_equal(loc.images[0], _scene(0.0))


def test_a_one_frame_video_is_checked_at_finish():
    g, loc = run([0.0], {"truth": lambda f: 0.0, "wrong_at": {0: 50.0}})
    assert loc.calls == [0]
    assert g.samples[0].alignment is None and g.samples[0].status == "invalidated"


def test_first_contradiction_reaches_frame_zero_and_finish_runs_once():
    shifts = pan(31, 0.1)
    g, loc = run(shifts, {"truth": lambda f: shifts[f], "wrong_at": {30: 50.0}})
    assert all(g.samples[f].alignment is None for f in range(31))
    assert loc.calls == [30]  # the last frame was a refind frame: finish adds nothing
    g.finish()
    assert loc.calls == [30]


def test_an_unconfirmed_start_must_name_its_source_frame():
    c = _calib()
    for bad in (None, -1, 2.0, True):
        with pytest.raises(ValueError, match="start_frame"):
            HandheldCalibration(c, _scene(0.0), 30, load_video_policy()["stadium_follow"],
                                POLICY, FakeLocator(c), minimum_visible=0.6,
                                start_confirmed=False, start_frame=bad)
