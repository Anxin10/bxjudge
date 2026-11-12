"""Locating the calibrated stadium in any frame from keypoints (plan 2026-11-12-dynamic-calibration,
Task 2)."""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pytest

from bxjudge.autocalib import Alignment, warp_calibration
from bxjudge.calibration_store import save_snapshot
from bxjudge.stadium_finder import _AGREEMENT_ZONES, FoundStadium, zone_visibility
from bxjudge.stadium_locator import (
    KeypointLocator,
    StadiumTools,
    alignment_from_homography,
    keypoint_snapshot,
    zone_disagreement_px,
)
from bxjudge.types import Calibration
from tests.helpers import shipped_autocalib
from tests.test_stadium_finder import _frame_for, similarity, synth_model  # noqa: F401
from tests.test_stadium_keypoints import _points, _ref

H_BASE = similarity(rot_deg=8, scale=0.85, shift=(25, -10), persp=(0.00012, -0.00008))
MOVE = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class FakeFinder:
    """``KeypointFinder`` stand-in: the reference points seen through ``view`` (model px ->
    frame px), whatever the frame; ``None`` when ``view`` is None."""

    def __init__(self, reference, view=None):
        self.reference = reference
        self.view = view
        self.raise_with = None
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        if self.raise_with is not None:
            raise self.raise_with
        return None if self.view is None else _points(self.view)


def _plain_alignment(h) -> Alignment:
    return Alignment(tuple(tuple(float(v) for v in r) for r in np.asarray(h)), 1.0, 0.0, 0.0,
                     1.0, 0.0)


@pytest.fixture
def tools_fake(synth_model):  # noqa: F811
    cfg, model = synth_model
    return StadiumTools(model=model, cfg=cfg, finder=FakeFinder(_ref(model)))


@pytest.fixture
def calib(tools_fake) -> Calibration:
    """The stadium zones in the reference view ``H_BASE`` (calibration px = frame px)."""
    return warp_calibration(tools_fake.model.calibration, _plain_alignment(H_BASE), (1280, 720),
                            "base")


def test_locate_maps_calibration_to_a_moved_view(tools_fake, calib):
    loc = KeypointLocator(tools_fake, calib, H_BASE)
    tools_fake.finder.view = MOVE @ H_BASE  # the camera moved 40 px right of the calibration
    got = loc.locate(_frame_for(MOVE @ H_BASE))
    assert got is not None and loc.last_reason is None
    assert zone_disagreement_px(calib, got.homography, MOVE) < 2.0
    assert got.homography[2, 2] == pytest.approx(1.0)
    assert got.alignment.shift_frac == pytest.approx(40 / 1280, abs=0.003)
    assert got.fit.keypoint_inliers == 10


def test_locate_without_points_or_fit_gives_a_reason(tools_fake, calib):
    loc = KeypointLocator(tools_fake, calib, H_BASE)
    assert loc.locate(_frame_for(H_BASE)) is None and loc.last_reason == "no_keypoints"
    tools_fake.finder.view = MOVE @ H_BASE  # points of one view, rail of another
    assert loc.locate(_frame_for(similarity(rot_deg=40, scale=0.7))) is None
    assert loc.last_reason == "fit_rejected"


@pytest.mark.parametrize("error", [RuntimeError("CUDA out of memory"),
                                   MemoryError("Unable to allocate array")])
def test_locate_turns_inference_errors_into_a_reason(tools_fake, calib, error):
    tools_fake.finder.raise_with = error
    loc = KeypointLocator(tools_fake, calib, H_BASE)
    assert loc.locate(_frame_for(H_BASE)) is None
    assert loc.last_reason == f"keypoint_error: {type(error).__name__}: {error}"


def test_for_snapshot_fits_the_reference_of_a_manual_snapshot_and_gives_none_when_it_cannot(
        tools_fake, calib, tmp_path, caplog):
    ref = _frame_for(H_BASE)
    snap = save_snapshot(tmp_path, "manual", calib, ref, score=1.0, autocalib=shipped_autocalib())
    tools_fake.finder.view = H_BASE
    loc = KeypointLocator.for_snapshot(tools_fake, snap, calib, ref)
    assert loc is not None
    same = loc.locate(ref)
    assert zone_disagreement_px(calib, same.homography, np.eye(3)) < 1.0
    tools_fake.finder.view = None
    with caplog.at_level(logging.WARNING):
        assert KeypointLocator.for_snapshot(tools_fake, snap, calib, ref) is None
    assert "no_keypoints" in caplog.text


def test_for_snapshot_uses_a_keypoints_snapshot_homography_without_inference(
        tools_fake, calib, tmp_path):
    ref = _frame_for(H_BASE)
    snap = save_snapshot(tmp_path, "keypoints", calib, ref, score=0.9, source_frame=0,
                         agreeing_frames=2,
                         homography=tuple(map(tuple, H_BASE)), autocalib=shipped_autocalib())
    loc = KeypointLocator.for_snapshot(tools_fake, snap, calib, ref)
    assert loc is not None and tools_fake.finder.calls == 0
    np.testing.assert_allclose(loc.model_to_calibration, H_BASE / H_BASE[2, 2], atol=1e-9)


def test_for_snapshot_scales_a_keypoints_snapshot_to_the_decoded_size(tools_fake, calib,
                                                                       tmp_path):
    """A 1280x720 snapshot used on 640x360 frames: model px -> calibration px halves."""
    import cv2

    from bxjudge.calibration import scale_calibration

    ref = _frame_for(H_BASE)
    snap = save_snapshot(tmp_path, "keypoints", calib, ref, score=0.9, source_frame=0,
                         agreeing_frames=2,
                         homography=tuple(map(tuple, H_BASE)), autocalib=shipped_autocalib())
    small = scale_calibration(calib, (640, 360))
    loc = KeypointLocator.for_snapshot(tools_fake, snap, small, cv2.resize(ref, (640, 360)))
    half = np.diag([0.5, 0.5, 1.0])
    np.testing.assert_allclose(loc.model_to_calibration, half @ H_BASE / H_BASE[2, 2],
                               atol=1e-9)
    # A frame of the moved view at 640 wide: the calibration lands 20 px (40 at 1280) right.
    tools_fake.finder.view = half @ MOVE @ H_BASE
    got = loc.locate(cv2.resize(_frame_for(MOVE @ H_BASE), (640, 360)))
    assert got is not None
    assert zone_disagreement_px(small, got.homography, half @ MOVE @ np.linalg.inv(half)) < 2.0


def _found(model, h, method, *, frame_index=12, agreeing=2):
    from bxjudge.stadium_finder import fit_from_keypoints, load_stadium_finder_config, rail_mask

    cfg = load_stadium_finder_config()
    frame = _frame_for(h)
    fit = fit_from_keypoints(_points(h), _ref(model), rail_mask(frame, cfg), model, cfg)
    cal = warp_calibration(model.calibration, _plain_alignment(fit.homography), (1280, 720),
                           "found-clip")
    return FoundStadium(calibration=cal, fit=fit, frame_index=frame_index, frame=frame,
                        zone_visibility=zone_visibility(cal), agreeing_frames=agreeing,
                        method=method)


def test_keypoint_snapshot_rejects_rail_results_and_vetoed_ids(tmp_path, monkeypatch, tools_fake):
    import bxjudge.stadium_locator as sl

    found = _found(tools_fake.model, H_BASE, "keypoints")
    monkeypatch.setattr(sl, "find_stadium", lambda *a, **k: found)
    snap, got = keypoint_snapshot(tmp_path / "clip.mp4", tmp_path / "a", tools_fake,
                                  shipped_autocalib(), set())
    assert got is found and snap.kind == "keypoints"
    assert snap.homography == tuple(tuple(float(v) for v in r) for r in found.fit.homography)
    assert snap.score == pytest.approx(found.fit.coverage)
    assert (snap.source_frame, snap.agreeing_frames) == (12, 2)
    # The same snapshot, vetoed, is refused.
    again, got = keypoint_snapshot(tmp_path / "clip.mp4", tmp_path / "a", tools_fake,
                                   shipped_autocalib(), {snap.id})
    assert again is None and got is found
    # A rail result is never stored: the rail path is less accurate (ledger 2026-11-12).
    rail = replace(found, method="rail")
    monkeypatch.setattr(sl, "find_stadium", lambda *a, **k: rail)
    none, got = keypoint_snapshot(tmp_path / "clip.mp4", tmp_path / "b", tools_fake,
                                  shipped_autocalib(), set())
    assert none is None and got is rail and not (tmp_path / "b").exists()


def test_single_keypoint_fit_is_offered_only_when_asked(tmp_path, monkeypatch, tools_fake):
    """A handheld camera rarely gives two early frames that agree; its start may be one keypoint
    fit, which the handheld tracker confirms before anything is emitted (ledger ruling)."""
    import bxjudge.stadium_locator as sl

    worse = _found(tools_fake.model, MOVE @ H_BASE, "keypoints", frame_index=40, agreeing=1)
    better = _found(tools_fake.model, H_BASE, "keypoints", frame_index=80, agreeing=1)
    worse = replace(worse, fit=replace(worse.fit, keypoint_error_px=3.0))
    better = replace(better, fit=replace(better.fit, keypoint_error_px=1.0))

    def fake_find(*a, keypoint_fits=None, **k):
        for f in (worse, better):
            keypoint_fits.append((f.frame_index, f.frame, f.fit))
        return None

    monkeypatch.setattr(sl, "find_stadium", fake_find)
    snap, got = keypoint_snapshot(tmp_path / "clip.mp4", tmp_path / "a", tools_fake,
                                  shipped_autocalib(), set())
    assert snap is None and got is None
    snap, got = keypoint_snapshot(tmp_path / "clip.mp4", tmp_path / "a", tools_fake,
                                  shipped_autocalib(), set(), allow_single=True)
    assert snap.kind == "keypoints" and got.method == "keypoints_single"
    assert got.frame_index == 80 and got.agreeing_frames == 1
    assert (snap.source_frame, snap.agreeing_frames) == (80, 1)


def test_zone_disagreement_is_the_worst_zone_not_the_mean(calib):
    # Scale about the battle area's centre: it barely moves, the outer zones move most.
    centre = np.array(calib.battle_area, np.float64).mean(0)
    t = np.array([[1.0, 0.0, centre[0]], [0.0, 1.0, centre[1]], [0.0, 0.0, 1.0]])
    grow = t @ np.diag([1.05, 1.05, 1.0]) @ np.linalg.inv(t)
    per_zone = []
    for z in _AGREEMENT_ZONES:
        poly = np.array(getattr(calib, z), np.float64)
        moved = np.c_[poly, np.ones(len(poly))] @ grow.T
        per_zone.append(float(np.linalg.norm(moved[:, :2] - poly, axis=1).mean()))
    got = zone_disagreement_px(calib, np.eye(3), grow)
    assert got == pytest.approx(max(per_zone)) and got > np.mean(per_zone)
    assert zone_disagreement_px(calib, grow, grow) == 0.0


def test_alignment_from_homography_describes_the_move_without_bounds(calib):
    far = np.array([[1.0, 0.0, 600.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # beyond any follow bound
    a = alignment_from_homography(calib, far, 0.8)
    assert a.shift_frac == pytest.approx(600 / 1280) and a.score == 0.8
    assert a.rotation_deg == pytest.approx(0.0) and a.scale == pytest.approx(1.0)
    np.testing.assert_allclose(np.array(a.homography), far)
