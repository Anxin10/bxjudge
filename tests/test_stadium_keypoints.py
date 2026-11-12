"""Stadium zones from the 10 stadium keypoints (plan 2026-11-11-stadium-keypoints, K4/K5)."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from bxjudge import PROJECT_ROOT
from bxjudge.stadium_finder import find_stadium, fit_from_keypoints, rail_mask
from bxjudge.stadium_keypoints import (
    FLIP_IDX,
    KEYPOINTS,
    KeypointFinder,
    KeypointReference,
    load_keypoint_reference,
    merge_flipped,
    resolve_weights,
    save_keypoint_reference,
)
from tests.test_stadium_finder import (
    _frame_for,
    _write_cfg,
    apply_h,
    similarity,
    synth_model,  # noqa: F401  (fixture)
    write_video,
    zone_error,
)

# Synthetic reference: features on the synthetic stadium (template px), as the 10 real points
# sit on the notch, the Xtreme opening, the Over pockets and the inner corners.
SYNTH_REF = {
    "notch_left": (600.0, 520.0), "notch_right": (680.0, 520.0),
    "xtreme_left": (560.0, 125.0), "xtreme_right": (720.0, 125.0),
    "over_left_inner": (215.0, 300.0), "over_right_inner": (1065.0, 300.0),
    "field_left_upper": (260.0, 470.0), "field_left_lower": (380.0, 585.0),
    "field_right_upper": (1020.0, 470.0), "field_right_lower": (900.0, 585.0),
}
NEAR = ("notch_left", "notch_right", "field_left_upper", "field_left_lower",
        "field_right_upper", "field_right_lower")
FAR = ("xtreme_left", "xtreme_right", "over_left_inner", "over_right_inner")


def _ref(model) -> KeypointReference:
    return KeypointReference(points=dict(SYNTH_REF), template=model.calibration.name,
                             image_size=tuple(model.image_size), model_sha256=model.source_sha256,
                             image_sha256="0" * 64)


def _points(h, conf=0.9, drop=(), move=None) -> np.ndarray:
    """(10, 3) x, y, confidence of the reference points seen through homography ``h``."""
    xy = apply_h(h, [SYNTH_REF[k] for k in KEYPOINTS])
    out = np.c_[xy, np.full(10, conf)]
    for k in drop:
        out[KEYPOINTS.index(k), 2] = 0.0
    for k, d in (move or {}).items():
        out[KEYPOINTS.index(k), :2] += d
    return out


def _turn(model) -> np.ndarray:
    """180 degrees about the model rail centre (template px)."""
    cx, cy = model.rail.mean(0)
    return np.array([[-1.0, 0.0, 2 * cx], [0.0, -1.0, 2 * cy], [0.0, 0.0, 1.0]])


H_VIEW = similarity(rot_deg=8, scale=0.85, shift=(25, -10), persp=(0.00012, -0.00008))
H_RAIL = similarity(rot_deg=6, scale=0.85, shift=(20, 10))  # a view the rail path finds

_MASKS: dict[bytes, np.ndarray] = {}


def _mask(h):
    """Rail mask of the synthetic frame seen through ``h`` (cached: rail_mask is slow)."""
    from bxjudge.stadium_finder import load_stadium_finder_config

    key = np.asarray(h, np.float64).tobytes()
    if key not in _MASKS:
        _MASKS[key] = rail_mask(_frame_for(h), load_stadium_finder_config())
    return _MASKS[key]


# --------------------------------------------------------------------------- reference
def test_reference_roundtrip(tmp_path, synth_model):
    _, model = synth_model
    p = tmp_path / "ref.json"
    save_keypoint_reference(_ref(model), p)
    assert load_keypoint_reference(p) == _ref(model)


@pytest.mark.parametrize("edit", [
    lambda d: d["points"].pop("field_right_lower"),
    lambda d: d["points"].update(extra=[1.0, 2.0]),
    lambda d: d["points"].update(notch_left=[1.0, 2.0, 3.0]),
    lambda d: d["points"].update(notch_left=[5000.0, 10.0]),  # outside the frame
    lambda d: d["points"].update(notch_left=[float("nan"), 10.0]),
    lambda d: d.update(image_size=[1280.5, 720]),
    lambda d: d.update(image_size=[0, 720]),
    lambda d: d.update(model_sha256="abc"),
    lambda d: d.update(schema="bxjudge-stadium-keypoints-reference-v1"),
])
def test_invalid_reference_is_refused(tmp_path, synth_model, edit):
    _, model = synth_model
    p = tmp_path / "ref.json"
    save_keypoint_reference(_ref(model), p)
    doc = json.loads(p.read_text())
    edit(doc)
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_keypoint_reference(p)


def test_reference_for_another_model_is_refused(synth_model):
    cfg, model = synth_model
    other = KeypointReference(points=dict(SYNTH_REF), template=model.calibration.name,
                              image_size=tuple(model.image_size), model_sha256="f" * 64,
                              image_sha256="0" * 64)
    with pytest.raises(ValueError, match="does not belong"):
        fit_from_keypoints(_points(H_VIEW), other, _mask(H_VIEW), model, cfg)


def test_shipped_reference_belongs_to_the_shipped_model():
    from bxjudge.stadium_finder import check_keypoint_reference, load_stadium_model

    ref = load_keypoint_reference()
    check_keypoint_reference(ref, load_stadium_model())
    # cat ears facing the viewer: every left point is left of its right partner
    for i, j in enumerate(FLIP_IDX):
        if i < j:
            assert ref.points[KEYPOINTS[i]][0] < ref.points[KEYPOINTS[j]][0]


# --------------------------------------------------------------------------- merging
def test_flip_idx_swaps_left_right_pairs():
    for i, j in enumerate(FLIP_IDX):
        assert FLIP_IDX[j] == i and i != j
        a, b = KEYPOINTS[i], KEYPOINTS[j]
        assert a.replace("left", "right") == b or a.replace("right", "left") == b


def test_mirrored_prediction_is_mapped_back_by_hand_example():
    # Frame 100 px wide; the real left notch is at x=30, the right one at x=45. In the mirror
    # image the stadium's left notch is the real right one, at 100 - 45 = 55, and its right
    # notch the real left one at 100 - 30 = 70. Worked out by hand, not with the code's formula.
    mirrored = np.zeros((10, 3))
    mirrored[0] = (55.0, 20.0, 0.9)  # "notch_left" in the mirror image
    mirrored[1] = (70.0, 21.0, 0.8)  # "notch_right" in the mirror image
    merged = merge_flipped([], [mirrored], 100, min_conf=0.5)
    np.testing.assert_allclose(merged[0], (30.0, 21.0, 0.8))  # real notch_left
    np.testing.assert_allclose(merged[1], (45.0, 20.0, 0.9))  # real notch_right


def test_unsure_predictions_do_not_move_confident_points():
    # Codex review 2026-11-12 Medium: a 0.02 prediction 120 px off pulled a 0.99 one by 60 px.
    good = np.c_[np.full((10, 2), 200.0), np.full(10, 0.99)]
    bad = good.copy()
    bad[:, 0] += 120.0
    bad[:, 2] = 0.02
    merged = merge_flipped([good, bad], [], 1280, min_conf=0.5)
    np.testing.assert_allclose(merged[:, 0], 200.0)
    np.testing.assert_allclose(merged[:, 2], (0.99 + 0.02) / 2)


def test_confident_predictions_far_apart_make_the_point_unusable():
    a = np.c_[np.full((10, 2), 200.0), np.full(10, 0.9)]
    b = a.copy()
    b[3, 0] += 200.0  # two confident models put xtreme_right 200 px apart
    merged = merge_flipped([a, b], [], 1280, min_conf=0.5, max_spread=60.0)
    assert merged[3, 2] == 0.0
    assert (merged[np.arange(10) != 3, 2] == 0.9).all()


def test_point_without_confident_prediction_is_unusable():
    a = np.c_[np.full((10, 2), 200.0), np.full(10, 0.2)]
    merged = merge_flipped([a], [], 1280, min_conf=0.5)
    assert (merged[:, 2] == 0.0).all()


def test_merge_without_any_prediction_is_none():
    assert merge_flipped([], [], 1280) is None


# --------------------------------------------------------------------------- points -> zones
def test_exact_points_recover_zones(synth_model):
    cfg, model = synth_model
    fit = fit_from_keypoints(_points(H_VIEW), _ref(model), _mask(H_VIEW), model, cfg)
    assert fit is not None
    assert zone_error(fit.homography, H_VIEW) < 1.0
    assert fit.keypoint_inliers == 10 and fit.keypoint_error_px < 0.5


def test_fewer_than_four_points_return_none(synth_model):
    cfg, model = synth_model
    pts = _points(H_VIEW, drop=KEYPOINTS[3:])
    assert fit_from_keypoints(pts, _ref(model), _mask(H_VIEW), model, cfg) is None


def test_low_confidence_points_are_not_used(synth_model):
    cfg, model = synth_model
    pts = _points(H_VIEW, conf=cfg.keypoint_min_conf / 2)
    assert fit_from_keypoints(pts, _ref(model), _mask(H_VIEW), model, cfg) is None


def test_points_on_one_side_return_none(synth_model):
    cfg, model = synth_model
    for side in ("right", "left"):
        pts = _points(H_VIEW, drop=[k for k in KEYPOINTS if side in k])
        assert fit_from_keypoints(pts, _ref(model), _mask(H_VIEW), model, cfg) is None


@pytest.mark.parametrize("drop", [FAR, NEAR], ids=["near-only", "far-only"])
def test_points_only_near_or_only_far_return_none(synth_model, drop):
    cfg, model = synth_model
    cfg_low = dataclasses.replace(cfg, keypoint_min_inliers=4)  # far-only leaves 4 points
    pts = _points(H_VIEW, drop=drop)
    assert fit_from_keypoints(pts, _ref(model), _mask(H_VIEW), model, cfg_low) is None


def test_one_wild_point_is_rejected_by_ransac(synth_model):
    cfg, model = synth_model
    pts = _points(H_VIEW, move={"over_right_inner": (90.0, -60.0)})
    fit = fit_from_keypoints(pts, _ref(model), _mask(H_VIEW), model, cfg)
    assert fit is not None
    assert fit.keypoint_inliers == 9
    assert zone_error(fit.homography, H_VIEW) < 1.0


def test_mirrored_points_are_refused(synth_model):
    # left and right labels swapped (as a model confusing the sides would): X/B must not swap
    cfg, model = synth_model
    pts = _points(H_VIEW)[list(FLIP_IDX)]
    assert fit_from_keypoints(pts, _ref(model), _mask(H_VIEW), model, cfg) is None


def test_points_describing_the_stadium_turned_round_are_refused(synth_model):
    # Codex review 2026-11-12 High: self-consistent points of the view turned 180 degrees
    # (Xtreme at the near end, the Over pockets swapped) passed on 30% rail coverage.
    cfg, model = synth_model
    pts = _points(H_VIEW @ _turn(model))
    assert fit_from_keypoints(pts, _ref(model), _mask(H_VIEW), model, cfg) is None


@pytest.mark.parametrize("dx", [60.0, 50.0, 45.0])
def test_points_shifted_off_the_rail_are_refused(synth_model, dx):
    # Codex reviews 2026-11-12 High: every point 50 px (then 45 px) to the side passed on the
    # coverage checks. The rail pulls such a fit back further than keypoint_max_rail_shift_px.
    cfg, model = synth_model
    assert dx > cfg.keypoint_max_rail_shift_px
    pts = _points(H_VIEW, move={k: (dx, 0.0) for k in KEYPOINTS})
    assert fit_from_keypoints(pts, _ref(model), _mask(H_VIEW), model, cfg) is None


@pytest.mark.parametrize("tilt", [0.0002, 0.0003])
def test_one_pocket_off_while_the_mean_is_right_is_refused(synth_model, tilt):
    # Codex re-review 2026-11-12 (round 2) High: points of the view tilted about the rail
    # centre moved the zones 21 px on average, the left pocket 42 px, and passed a bound on
    # the mean over all zones. Each zone is now checked on its own.
    cfg, model = synth_model
    cx, cy = model.rail.mean(0)
    t = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    tilted = H_VIEW @ t @ np.array([[1.0, 0, 0], [0, 1.0, 0], [tilt, 0, 1.0]]) @ np.linalg.inv(t)
    assert fit_from_keypoints(_points(tilted), _ref(model), _mask(H_VIEW), model, cfg) is None


def test_small_shift_within_the_rail_tolerance_is_kept(synth_model):
    # the bound is the rail tolerance: a 10 px shift stays (and stays within 10 px)
    cfg, model = synth_model
    pts = _points(H_VIEW, move={k: (10.0, 0.0) for k in KEYPOINTS})
    fit = fit_from_keypoints(pts, _ref(model), _mask(H_VIEW), model, cfg)
    assert fit is not None and zone_error(fit.homography, H_VIEW) < 11.0
    assert fit.keypoint_rail_shift_px <= cfg.keypoint_max_rail_shift_px


def test_direction_the_partial_rail_cannot_tell_is_refused(synth_model):
    # Codex re-review 2026-11-12 High: with only part of the rail visible and that part
    # consistent with both directions, the points alone would decide the direction.
    import cv2

    cfg, model = synth_model
    both = np.maximum(_mask(H_VIEW), _mask(H_VIEW @ _turn(model)))
    both[:, 650:] = 0
    for h in (H_VIEW, H_VIEW @ _turn(model)):
        assert fit_from_keypoints(_points(h), _ref(model), both, model, cfg) is None
    assert cv2.countNonZero(both) > 0


def test_points_the_rail_does_not_confirm_are_refused(synth_model):
    cfg, model = synth_model
    other = similarity(rot_deg=-10, scale=0.7, shift=(-120, 40))
    assert fit_from_keypoints(_points(H_VIEW), _ref(model), _mask(other), model, cfg) is None


def test_noisy_points_stay_within_the_noise(synth_model):
    cfg, model = synth_model
    rng = np.random.default_rng(3)
    pts = _points(H_VIEW)
    pts[:, :2] += rng.normal(0, 3.0, (10, 2))
    fit = fit_from_keypoints(pts, _ref(model), _mask(H_VIEW), model, cfg)
    assert fit is not None and zone_error(fit.homography, H_VIEW) < 6.0


# --------------------------------------------------------------------------- config
@pytest.mark.parametrize("change", [
    {"keypoint_min_inliers": 3}, {"keypoint_min_inliers": 11}, {"keypoint_min_conf": 1.5},
    {"keypoint_ransac_px": 0}, {"keypoint_weights": "one.pt"},
    {"keypoint_weights": ["../outside.pt"]}, {"keypoint_weights": ["/abs/model.pt"]},
    {"keypoint_weights": ["C:/w/model.pt"]}, {"keypoint_weights": ["weights\\..\\..\\x.pt"]},
    {"keypoint_imgsz": 100}, {"keypoint_imgsz": 1000}, {"keypoint_agreement_px": ...},
    {"keypoint_max_spread_px": -1}, {"keypoint_min_rail_precision": 2.0},
])
def test_keypoint_config_is_validated(tmp_path, change):
    from bxjudge.stadium_finder import load_stadium_finder_config

    with pytest.raises((ValueError, TypeError)):
        load_stadium_finder_config(_write_cfg(tmp_path, **change))


def test_weights_resolving_outside_the_project_are_refused(tmp_path, monkeypatch):
    import bxjudge.stadium_keypoints as sk

    project = tmp_path / "project"
    (project / "weights").mkdir(parents=True)
    outside = tmp_path / "elsewhere.pt"
    outside.write_bytes(b"x")
    (project / "weights" / "link.pt").symlink_to(outside)
    (project / "weights" / "ok.pt").write_bytes(b"y")
    monkeypatch.setattr(sk, "PROJECT_ROOT", project)
    with pytest.raises(ValueError, match="outside the project"):
        resolve_weights(["weights/link.pt"])
    with pytest.raises(FileNotFoundError):
        resolve_weights(["weights/missing.pt"])
    with pytest.raises(ValueError):
        resolve_weights([])
    assert resolve_weights(["weights/ok.pt"]) == [(project / "weights" / "ok.pt").resolve()]


# --------------------------------------------------------------------------- find_stadium
class _FakeFinder:
    """Stands in for the YOLO ensemble: returns fixed points (or nothing) for every frame."""

    def __init__(self, points, reference):
        self.points, self.reference, self.calls = points, reference, 0

    def predict(self, frame):
        self.calls += 1
        return None if self.points is None else self.points.copy()


def test_find_stadium_prefers_keypoints(tmp_path, synth_model):
    cfg, model = synth_model
    clip = write_video(tmp_path / "steady.mp4", [_frame_for(H_VIEW)] * 60)
    finder = _FakeFinder(_points(H_VIEW), _ref(model))
    found = find_stadium(clip, model, cfg, keypoints=finder)
    assert found is not None and found.method == "keypoints"
    assert found.agreeing_frames >= 2 and finder.calls >= 2
    assert zone_error(found.fit.homography, H_VIEW) < 1.0


def _rail_only(tmp_path, cfg, model):
    clip = write_video(tmp_path / "rail.mp4", [_frame_for(H_RAIL)] * 60)
    return clip, find_stadium(clip, model, cfg)


def test_find_stadium_falls_back_to_the_rail_when_no_points(tmp_path, synth_model):
    cfg, model = synth_model
    clip, rail = _rail_only(tmp_path, cfg, model)
    found = find_stadium(clip, model, cfg, keypoints=_FakeFinder(None, _ref(model)))
    assert rail is not None and found is not None and found.method == "rail"
    # exactly the rail path's result
    assert found.frame_index == rail.frame_index and found.fit == rail.fit
    assert found.calibration == rail.calibration


def test_find_stadium_falls_back_to_the_rail_when_points_disagree(tmp_path, synth_model):
    cfg, model = synth_model

    class OneFrame(_FakeFinder):
        def predict(self, frame):  # points on the first candidate frame only: no second fit
            self.calls += 1
            return _points(H_RAIL) if self.calls == 1 else None

    clip, rail = _rail_only(tmp_path, cfg, model)
    finder = OneFrame(None, _ref(model))
    # that frame's own fit is accepted, so the fallback really comes from the agreement rule
    assert fit_from_keypoints(_points(H_RAIL), _ref(model), _mask(H_RAIL), model, cfg) is not None
    found = find_stadium(clip, model, cfg, keypoints=finder)
    assert finder.calls >= 2
    assert found is not None and found.method == "rail"
    assert found.frame_index == rail.frame_index and found.fit == rail.fit
    assert found.calibration == rail.calibration


def test_find_stadium_falls_back_to_the_rail_when_inference_fails(tmp_path, synth_model):
    # Codex review 2026-11-12 Medium: an exception in the keypoint path skipped the rail.
    cfg, model = synth_model

    class Broken(_FakeFinder):
        def predict(self, frame):
            self.calls += 1
            raise RuntimeError("CUDA error: out of memory")

    clip, rail = _rail_only(tmp_path, cfg, model)
    finder = Broken(None, _ref(model))
    errors: list[str] = []
    found = find_stadium(clip, model, cfg, keypoints=finder, keypoint_errors=errors)
    assert finder.calls >= 1 and errors and "out of memory" in errors[0]
    assert found is not None and found.method == "rail"
    assert found.frame_index == rail.frame_index and found.fit == rail.fit
    assert found.calibration == rail.calibration


def test_find_stadium_falls_back_to_the_rail_when_the_fit_fails(tmp_path, synth_model,
                                                                  monkeypatch):
    import cv2

    import bxjudge.stadium_finder as sf

    cfg, model = synth_model

    def broken_fit(*a, **k):
        raise cv2.error("findHomography failed")

    clip, rail = _rail_only(tmp_path, cfg, model)
    monkeypatch.setattr(sf, "fit_from_keypoints", broken_fit)
    errors: list[str] = []
    found = sf.find_stadium(clip, model, cfg, keypoints=_FakeFinder(_points(H_RAIL), _ref(model)),
                            keypoint_errors=errors)
    assert errors and found is not None and found.method == "rail"
    assert found.frame_index == rail.frame_index and found.fit == rail.fit
    assert found.calibration == rail.calibration


def test_find_stadium_refuses_a_reference_for_another_model(tmp_path, synth_model):
    cfg, model = synth_model
    clip = write_video(tmp_path / "steady.mp4", [_frame_for(H_VIEW)] * 30)
    wrong = KeypointReference(points=dict(SYNTH_REF), template="other", image_size=(1280, 720),
                              model_sha256="f" * 64, image_sha256="0" * 64)
    with pytest.raises(ValueError):
        find_stadium(clip, model, cfg, keypoints=_FakeFinder(_points(H_VIEW), wrong))


def test_find_stadium_never_reports_consistent_turned_points(tmp_path, synth_model):
    # Codex review 2026-11-12 High, end to end: two frames agreeing on the turned view.
    cfg, model = synth_model
    clip = write_video(tmp_path / "steady.mp4", [_frame_for(H_VIEW)] * 60)
    found = find_stadium(clip, model, cfg,
                         keypoints=_FakeFinder(_points(H_VIEW @ _turn(model)), _ref(model)))
    assert found is None or found.method == "rail"


def test_find_stadium_without_keypoints_is_the_rail_method(tmp_path, synth_model):
    cfg, model = synth_model
    found = find_stadium(write_video(tmp_path / "s.mp4", [_frame_for(H_RAIL)] * 60), model, cfg)
    assert found is not None and found.method == "rail"


# --------------------------------------------------------------------------- real weights
def _weights_ok() -> bool:
    from bxjudge.stadium_finder import load_stadium_finder_config

    try:
        resolve_weights(load_stadium_finder_config().keypoint_weights)
        return True
    except (ValueError, OSError):
        return False


_TEMPLATE = PROJECT_ROOT / "calib" / "20261014_202713.png"


@pytest.mark.gpu
@pytest.mark.skipif(not (_weights_ok() and _TEMPLATE.exists()), reason="keypoint weights not installed")
def test_real_ensemble_finds_the_template_points():
    import cv2

    from bxjudge.stadium_finder import load_stadium_finder_config

    cfg = load_stadium_finder_config()
    finder = KeypointFinder.from_config(cfg)
    frame = cv2.imread(str(_TEMPLATE))
    pts = finder.predict(frame)
    assert pts is not None
    ref = finder.reference.array()
    seen = pts[:, 2] >= cfg.keypoint_min_conf
    assert seen.sum() >= 8
    assert np.median(np.linalg.norm(pts[seen, :2] - ref[seen], axis=1)) < 15.0
