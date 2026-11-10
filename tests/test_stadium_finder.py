"""Automatic stadium finding from the green rail (plan 2026-11-11-auto-stadium)."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from bxjudge.stadium_finder import (
    DEFAULT_STADIUM_FINDER_CONFIG,
    build_stadium_model,
    find_stadium,
    fit_stadium,
    zone_visibility,
    load_stadium_finder_config,
    load_stadium_model,
    rail_mask,
    save_stadium_model,
)
from bxjudge.types import Calibration

W, H = 1280, 720


# --------------------------------------------------------------------------- synthetic scene
def rail_polyline(n: int = 720) -> np.ndarray:
    """Model-plane rail: a rounded rectangle (superellipse, like the real rail's straight sides)
    whose near (bottom) end has a notch, so it has an up/down."""
    cx, cy, a, b, p = 640.0, 360.0, 420.0, 230.0, 3.0
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = cx + a * np.sign(c) * np.abs(c) ** (2 / p)
    y = cy + b * np.sign(s) * np.abs(s) ** (2 / p)
    d = np.abs(t - math.pi / 2)
    notch = d < 0.35
    y[notch] -= 70.0 * (1.0 - d[notch] / 0.35)
    return np.stack([x, y], 1)


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def synthetic_calibration() -> Calibration:
    return Calibration(
        name="synthetic",
        frame_size=(W, H),
        stadium=rect(150, 60, 1130, 690),
        lid=rect(250, 70, 1030, 300),
        battle_area=[(float(x), float(y)) for x, y in _ellipse(640, 360, 450, 260)],
        over_left=rect(160, 260, 215, 420),
        over_right=rect(1065, 260, 1120, 420),
        xtreme=rect(560, 70, 720, 125),
        launch_left=rect(0, 0, 640, 200),
        launch_right=rect(640, 0, 1280, 200),
        ref_frame="",
    )


def _ellipse(cx, cy, a, b, n=48):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], 1)


def draw_rail(points: np.ndarray, *, hue: int = 60, size=(W, H), thickness: int = 14,
              background: int = 128) -> np.ndarray:
    """BGR image with the rail drawn in an HSV colour of the given OpenCV hue."""
    img = np.full((size[1], size[0], 3), background, np.uint8)
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    cv2.polylines(img, [np.round(points).astype(np.int32)], True, bgr, thickness, cv2.LINE_AA)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def rail_pixels(points: np.ndarray, size=(W, H), thickness: int = 14) -> np.ndarray:
    m = np.zeros((size[1], size[0]), np.uint8)
    cv2.polylines(m, [np.round(points).astype(np.int32)], True, 255, thickness)
    return m > 0


# --------------------------------------------------------------------------- Task 1: config
def _write_cfg(tmp_path: Path, **changes) -> Path:
    raw = yaml.safe_load(DEFAULT_STADIUM_FINDER_CONFIG.read_text(encoding="utf-8"))
    for k, v in changes.items():
        if v is ...:
            raw.pop(k)
        else:
            raw[k] = v
    p = tmp_path / "stadium_finder.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_default_config_loads():
    cfg = load_stadium_finder_config()
    assert cfg.model_points > 0 and 0 < cfg.min_coverage <= 1


@pytest.mark.parametrize("change", [
    {"model_points": ...},
    {"model_points": -5},
    {"min_coverage": float("nan")},
    {"inlier_px": 0},
    {"unknown_key": 1},
    {"hue_search": [90, 30]},
])
def test_config_rejects_missing_or_nonpositive_keys(tmp_path, change):
    with pytest.raises(ValueError):
        load_stadium_finder_config(_write_cfg(tmp_path, **change))


# --------------------------------------------------------------------------- Task 2: rail mask
@pytest.mark.parametrize("hue", [45, 70])
def test_rail_mask_adapts_to_hue_shift(hue):
    cfg = load_stadium_finder_config()
    pts = rail_polyline()
    mask = rail_mask(draw_rail(pts, hue=hue), cfg)
    truth = rail_pixels(pts, thickness=10)  # core of the drawn band
    assert mask.shape == (H, W) and mask.dtype == np.uint8
    assert (mask[truth] > 0).mean() >= 0.9


def test_rail_mask_drops_compact_blobs():
    cfg = load_stadium_finder_config()
    pts = rail_polyline()
    img = draw_rail(pts)
    bgr = cv2.cvtColor(np.uint8([[[60, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    cv2.ellipse(img, (640, 360), (70, 50), 0, 0, 360, bgr, -1)  # a solid green glove
    mask = rail_mask(img, cfg)
    glove = np.zeros((H, W), np.uint8)
    cv2.ellipse(glove, (640, 360), (60, 40), 0, 0, 360, 255, -1)
    assert (mask[glove > 0] > 0).mean() < 0.05
    assert (mask[rail_pixels(pts, thickness=10)] > 0).mean() >= 0.9


def test_rail_mask_keeps_thick_close_up_rail():
    cfg = load_stadium_finder_config()
    pts = rail_polyline()
    mask = rail_mask(draw_rail(pts, thickness=40), cfg)
    assert (mask[rail_pixels(pts, thickness=30)] > 0).mean() >= 0.9


def test_rail_mask_empty_on_gray():
    cfg = load_stadium_finder_config()
    img = np.full((H, W, 3), 128, np.uint8)
    assert not rail_mask(img, cfg).any()


# --------------------------------------------------------------------------- Task 1: model
def test_build_model_from_synthetic_rail():
    cfg = load_stadium_finder_config()
    pts = rail_polyline()
    img = draw_rail(pts)
    model = build_stadium_model(synthetic_calibration(), img, cfg)
    assert model.rail.shape == (cfg.model_points, 2)
    green = rail_pixels(pts, thickness=18)
    dist = cv2.distanceTransform((~green).astype(np.uint8), cv2.DIST_L2, 3)
    xs = np.clip(np.round(model.rail[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(model.rail[:, 1]).astype(int), 0, H - 1)
    assert dist[ys, xs].max() <= 2.0
    assert model.image_size == (W, H)


# --------------------------------------------------------------------------- Task 3: fitting
FIT_ZONES = ("battle_area", "over_left", "over_right", "xtreme")


def similarity(rot_deg=0.0, scale=1.0, shift=(0.0, 0.0), centre=(640.0, 360.0), persp=(0.0, 0.0)):
    """Homography: rotate/scale about ``centre``, then shift, then a slight perspective."""
    a = math.radians(rot_deg)
    c, s = math.cos(a) * scale, math.sin(a) * scale
    cx, cy = centre
    m = np.array([[c, -s, cx - c * cx + s * cy + shift[0]],
                  [s, c, cy - s * cx - c * cy + shift[1]],
                  [0.0, 0.0, 1.0]])
    p = np.array([[1.0, 0, 0], [0, 1.0, 0], [persp[0], persp[1], 1.0]])
    # perspective about the centre so the stadium stays in frame
    t = np.array([[1.0, 0, cx], [0, 1.0, cy], [0, 0, 1.0]])
    return t @ p @ np.linalg.inv(t) @ m


def apply_h(h, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(h, np.float64)).reshape(-1, 2)


def zone_error(h_found, h_true) -> float:
    """Mean vertex distance (px) of the zones near the rail warped by two homographies."""
    cal = synthetic_calibration()
    errs = [np.linalg.norm(apply_h(h_found, getattr(cal, z)) - apply_h(h_true, getattr(cal, z)), axis=1)
            for z in FIT_ZONES]
    return float(np.concatenate(errs).mean())


@pytest.fixture(scope="module")
def synth_model():
    cfg = load_stadium_finder_config()
    return cfg, build_stadium_model(synthetic_calibration(), draw_rail(rail_polyline()), cfg)


def _frame_for(h, *, cover=None, extra=None, thickness=14):
    img = draw_rail(apply_h(h, rail_polyline()), thickness=thickness)
    if cover is not None:
        x0, y0, x1, y1 = cover
        img[y0:y1, x0:x1] = 128
    if extra is not None:
        extra(img)
    return img


def test_fit_recovers_known_affine(synth_model):
    cfg, model = synth_model
    shear = np.array([[1.0, 0.08, -30.0], [0.0, 0.82, 60.0], [0.0, 0.0, 1.0]])  # tilted view
    h = similarity(rot_deg=12, scale=0.85, shift=(30, -20)) @ shear
    fit = fit_stadium(rail_mask(_frame_for(h), cfg), model, cfg)
    assert fit is not None
    assert zone_error(fit.homography, h) < 2.0
    assert fit.coverage >= cfg.min_coverage and fit.precision >= cfg.min_precision


def test_slight_perspective_is_approximated(synth_model):
    # The rail cannot pin perspective (an oval stays an oval); a mild one must stay small.
    cfg, model = synth_model
    h = similarity(rot_deg=-6, scale=0.85, persp=(2e-5, -3e-5))
    fit = fit_stadium(rail_mask(_frame_for(h), cfg), model, cfg)
    assert fit is not None and zone_error(fit.homography, h) < 6.0


def test_strong_perspective_uses_homography(synth_model):
    # A steeper camera than the template: the near end is magnified differently from the far
    # end. The rail's straight sides pin this, so the fit must not stay affine.
    cfg, model = synth_model
    h = similarity(rot_deg=4, scale=0.8, persp=(0.0, 4e-4))
    fit = fit_stadium(rail_mask(_frame_for(h), cfg), model, cfg)
    assert fit is not None and zone_error(fit.homography, h) < 3.0


def test_thick_close_up_rail_still_fits(synth_model):
    # A close camera shows a much thicker rail than the template did.
    cfg, model = synth_model
    h = similarity(rot_deg=-3, scale=1.0)
    fit = fit_stadium(rail_mask(_frame_for(h, thickness=30), cfg), model, cfg)
    assert fit is not None and zone_error(fit.homography, h) < 3.0


@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_rotated_and_mirrored_orientation(synth_model, rot):
    cfg, model = synth_model
    h = similarity(rot_deg=rot, scale=0.6)
    fit = fit_stadium(rail_mask(_frame_for(h), cfg), model, cfg)
    assert fit is not None
    assert zone_error(fit.homography, h) < 3.0  # the notch fixes the direction: no 180° flip


def test_mirrored_rail_keeps_left_zones_left(synth_model):
    cfg, model = synth_model
    img = cv2.flip(_frame_for(similarity(scale=0.9)), 1)
    fit = fit_stadium(rail_mask(img, cfg), model, cfg)
    if fit is not None:  # the rail is left/right symmetric; a mirror must never be fitted
        assert np.linalg.det(np.asarray(fit.homography)[:2, :2]) > 0
        cal = synthetic_calibration()
        left = apply_h(fit.homography, cal.over_left).mean(0)[0]
        right = apply_h(fit.homography, cal.over_right).mean(0)[0]
        assert left < right


def test_partial_rail_still_fits(synth_model):
    # A hand over part of the far rail: the rest still pins every zone.
    cfg, model = synth_model
    h = similarity(rot_deg=-8, scale=0.9, shift=(-20, 10))
    fit = fit_stadium(rail_mask(_frame_for(h, cover=(500, 0, 760, 330)), cfg), model, cfg)
    assert fit is not None and zone_error(fit.homography, h) < 4.0


def test_large_occlusion_returns_none(synth_model):
    # With a whole side hidden the hidden zones are not determined: refuse rather than guess.
    cfg, model = synth_model
    h = similarity(rot_deg=-8, scale=0.9, shift=(-20, 10))
    assert fit_stadium(rail_mask(_frame_for(h, cover=(820, 0, 1280, 720)), cfg), model, cfg) is None


def test_green_glove_does_not_shift_fit(synth_model):
    cfg, model = synth_model
    h = similarity(rot_deg=5, scale=0.9)
    bgr = cv2.cvtColor(np.uint8([[[60, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()

    def finger(img):  # thin green strokes just inside the rail, like a gloved finger
        cv2.line(img, (420, 250), (560, 330), bgr, 10)
        cv2.line(img, (430, 270), (540, 360), bgr, 10)

    fit = fit_stadium(rail_mask(_frame_for(h, extra=finger), cfg), model, cfg)
    assert fit is None or zone_error(fit.homography, h) < 4.0


def test_low_coverage_returns_none(synth_model):
    cfg, model = synth_model
    h = similarity(scale=0.9)
    assert fit_stadium(rail_mask(_frame_for(h, cover=(0, 0, 1020, 720)), cfg), model, cfg) is None


# --------------------------------------------------------------------------- review fixes
def rail_without_notch(n: int = 720) -> np.ndarray:
    cx, cy, a, b, p = 640.0, 360.0, 420.0, 230.0, 3.0
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.stack([cx + a * np.sign(c) * np.abs(c) ** (2 / p),
                     cy + b * np.sign(s) * np.abs(s) ** (2 / p)], 1)


def test_ring_without_notch_returns_none(synth_model):
    # Codex review High 1: without the cat ears the direction (and so X/B, Xtreme) is unknown.
    cfg, model = synth_model
    h = similarity(rot_deg=3, scale=0.9)
    img = draw_rail(apply_h(h, rail_without_notch()))
    assert fit_stadium(rail_mask(img, cfg), model, cfg) is None


def test_hidden_notch_returns_none(synth_model):
    # Hands over the near end: the rest of the ring cannot tell 0 from 180 degrees.
    cfg, model = synth_model
    h = similarity(scale=0.9)
    assert fit_stadium(rail_mask(_frame_for(h, cover=(470, 460, 810, 720)), cfg), model, cfg) is None


def test_partly_cut_notch_still_fits(synth_model):
    # 11/08 close-ups: the near end runs off the bottom edge, part of the notch stays visible.
    cfg, model = synth_model
    h = similarity(scale=0.95, shift=(0, 190))  # notch coverage about 0.6
    fit = fit_stadium(rail_mask(_frame_for(h), cfg), model, cfg)
    assert fit is not None and zone_error(fit.homography, h) < 4.0
    assert fit.notch_coverage < 0.8  # the case is really cut


def test_cropped_rail_returns_none(synth_model):
    # Codex review High 2: a third of the rail outside the frame leaves zones unconstrained.
    cfg, model = synth_model
    h = similarity(rot_deg=180, scale=1.2, shift=(0, -250))
    fit = fit_stadium(rail_mask(_frame_for(h), cfg), model, cfg)
    assert fit is None or zone_error(fit.homography, h) < 8.0


def _turn(model, h):
    cx, cy = model.rail.mean(0)
    return np.asarray(h) @ np.array([[-1.0, 0, 2 * cx], [0, -1.0, 2 * cy], [0, 0, 1.0]])


def test_orientation_check_keeps_an_unrefined_competitor(synth_model, monkeypatch):
    # Codex re-review High: a rail drawn both ways round; the 180-degree alternative is as
    # good even when its refinement fails, so the direction must count as ambiguous.
    import bxjudge.stadium_finder as sf
    from bxjudge.stadium_finder import _orientation_ambiguous, _score

    monkeypatch.setattr(sf, "_refine", lambda *a, **k: None)  # ECC fails on the alternative

    cfg, _ = synth_model
    model = load_stadium_model()  # the shipped model (1920 px template), as Codex reproduced
    k = W / model.image_size[0]
    h = np.diag([k, k, 1.0]) @ similarity(scale=0.85, centre=tuple(model.rail.mean(0)),
                                          persp=(2e-5, 1e-5))
    mask = np.zeros((H, W), np.uint8)
    for hh in (h, _turn(model, h)):
        for x, y in apply_h(hh, model.rail):
            cv2.circle(mask, (round(x), round(y)), 5, 255, -1)
    best = _score(mask, 1.0, model, h, cfg)
    assert best is not None and best.coverage > 0.9
    assert _orientation_ambiguous(mask, 1.0, model, best, cfg)


def test_unqualified_alternative_does_not_veto(synth_model):
    # Codex re-review Medium: extra green dilutes precision; a 180-degree alternative that
    # fails the acceptance checks itself must not veto a clear direction.
    cfg, model = synth_model
    h = similarity(scale=0.6)
    bgr = cv2.cvtColor(np.uint8([[[60, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()

    def lines(img):
        cv2.line(img, (80, 20), (700, 20), bgr, 8)
        cv2.line(img, (80, 690), (700, 690), bgr, 8)

    fit = fit_stadium(rail_mask(_frame_for(h, extra=lines), cfg), model, cfg)
    assert fit is not None and zone_error(fit.homography, h) < 2.0


def test_unacceptable_top_candidate_does_not_crowd_out_an_acceptable_one(monkeypatch):
    # Codex round 4: rank only candidates that pass the acceptance checks.
    import bxjudge.stadium_finder as sf
    from bxjudge.stadium_finder import StadiumFit

    cfg = load_stadium_finder_config()
    ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    good = dict(rail_in_frame=1.0, notch_coverage=1.0, quadrant_coverage=1.0, mean_error_px=1.0)
    # 1 ranks higher (quality 1.0 x 0.29) but fails min_precision; 2 (0.8 x 0.31) passes.
    fits = {1: StadiumFit(ident, coverage=1.0, precision=cfg.min_precision - 0.01, **good),
            2: StadiumFit(ident, coverage=cfg.min_coverage, precision=cfg.min_precision + 0.01, **good)}
    assert fits[1].coverage * fits[1].precision > fits[2].coverage * fits[2].precision
    hs = {1: np.eye(3) * 1.0, 2: np.eye(3) * 2.0}
    monkeypatch.setattr(sf, "_coarse_candidates", lambda *a: [(0.9, hs[1]), (0.89, hs[2])])
    monkeypatch.setattr(sf, "_refine", lambda m, r, mo, h, c, motion=None: None if motion else h)
    monkeypatch.setattr(sf, "_score", lambda m, r, mo, h, c: fits[1 if h[0, 0] == 1.0 else 2])
    monkeypatch.setattr(sf, "_orientation_ambiguous", lambda *a: False)
    mask = np.zeros((H, W), np.uint8)
    mask[10, 10] = 255
    got = sf.fit_stadium(mask, None, cfg)
    assert got is fits[2]


def _fit(cov, prec, err=1.0):
    from bxjudge.stadium_finder import StadiumFit

    ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return StadiumFit(ident, coverage=cov, precision=prec, mean_error_px=err, rail_in_frame=1.0,
                      notch_coverage=1.0, quadrant_coverage=1.0)


def test_unacceptable_homography_does_not_replace_an_acceptable_affine_fit(monkeypatch):
    # Codex round 5: the perspective fit (coverage just below the floor) beat the affine fit
    # on error and replaced it, then failed acceptance: the good affine fit was lost.
    import bxjudge.stadium_finder as sf

    cfg = load_stadium_finder_config()
    affine = _fit(cfg.min_coverage + 0.01, 0.60, err=1.0)
    persp = _fit(cfg.min_coverage - 0.001, 0.61, err=0.1)
    h_aff, h_hom = np.eye(3), np.eye(3) * 2.0
    monkeypatch.setattr(sf, "_coarse_candidates", lambda *a: [(0.9, h_aff)])
    monkeypatch.setattr(sf, "_refine", lambda m, r, mo, h, c, motion=None: h_hom if motion else h_aff)
    monkeypatch.setattr(sf, "_score", lambda m, r, mo, h, c: persp if h[0, 0] == 2.0 else affine)
    monkeypatch.setattr(sf, "_perspective", lambda *a: 0.0)
    monkeypatch.setattr(sf, "_orientation_ambiguous", lambda *a: False)

    class M:
        image_size = (1920, 1080)

    mask = np.zeros((H, W), np.uint8)
    mask[10, 10] = 255
    assert sf.fit_stadium(mask, M(), cfg) is affine


def test_turned_alternative_beyond_the_perspective_limit_cannot_veto(monkeypatch, synth_model):
    # Codex round 5: the 180-degree homography alternative must obey max_perspective too.
    import bxjudge.stadium_finder as sf

    cfg, model = synth_model
    best = _fit(1.0, 0.54)
    wild = np.array([[1.0, 0, 0], [0, 1.0, 0], [cfg.max_perspective * 2, 0, 1.0]])
    monkeypatch.setattr(sf, "_refine", lambda m, r, mo, h, c, motion=None: wild if motion else None)
    monkeypatch.setattr(sf, "_score", lambda m, r, mo, h, c: _fit(1.0, 0.46) if h is wild else None)
    mask = np.zeros((H, W), np.uint8)
    assert not sf._orientation_ambiguous(mask, 1.0, model, best, cfg)


def test_zones_must_project_finitely(synth_model):
    # Codex round 5: a projective denominator crossing zero inside the zones throws them to
    # infinity; such a mapping is never a stadium.
    from bxjudge.stadium_finder import _zones_valid

    cfg, model = synth_model
    xmax = max(x for x, _ in model.calibration.stadium)
    near_pole = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0 / (xmax - 0.5), 0, 1.0]])
    assert not _zones_valid(model, near_pole, (W, H))
    assert _zones_valid(model, np.eye(3), (W, H))


def test_zone_visibility_of_a_huge_polygon_is_cheap():
    # Codex round 5: rasterising a polygon reaching 1e9 px asked for petabytes.
    from dataclasses import replace

    cal = synthetic_calibration()
    huge = [(-1e9, -1e9), (0.0, -5e8), (1e9, -1e9), (1e9, 1e9), (-1e9, 1e9)]  # non-convex
    area = 4e18 - 1e9 * 5e8  # square minus the notch triangle
    vis = zone_visibility(replace(cal, stadium=huge))
    assert vis["stadium"] == pytest.approx(W * H / area, rel=1e-6)


def test_perspective_is_measured_in_inference_px():
    # Codex review Medium 3: both ends of the homography must be in inference px.
    from bxjudge.stadium_finder import _perspective

    h = np.array([[1.0, 0, 0], [0, 1.0, 0], [0.0008, 0, 1.0]])  # model 1920 px -> frame 1280 px
    assert _perspective(h, ref_scale=1.0, model_scale=1280 / 1920) == pytest.approx(0.0012)


# --------------------------------------------------------------------------- Task 4: video
def write_video(path: Path, frames, fps: int = 30) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        out.write(f)
    out.release()
    return path


def test_finds_stadium_in_a_steady_video(tmp_path, synth_model):
    cfg, model = synth_model
    h = similarity(rot_deg=6, scale=0.85, shift=(20, 10))
    clip = write_video(tmp_path / "steady.mp4", [_frame_for(h)] * 60)
    found = find_stadium(clip, model, cfg)
    assert found is not None and found.agreeing_frames >= 2
    assert found.calibration.frame_size == (W, H)
    assert zone_error(found.fit.homography, h) < 4.0
    assert found.zone_visibility["battle_area"] == pytest.approx(1.0)


def test_skips_occluded_frames(tmp_path, synth_model):
    cfg, model = synth_model
    h = similarity(rot_deg=-4, scale=0.9)
    hidden = _frame_for(h, cover=(0, 0, 1000, 720))  # hands over most of the rail at launch
    clip = write_video(tmp_path / "occluded.mp4", [hidden] * 45 + [_frame_for(h)] * 45)
    found = find_stadium(clip, model, cfg)
    assert found is not None and found.frame_index >= 45
    assert zone_error(found.fit.homography, h) < 4.0


@pytest.mark.parametrize("value", [0, 128])
def test_no_stadium_returns_none(tmp_path, synth_model, value):
    cfg, model = synth_model
    blank = np.full((H, W, 3), value, np.uint8)
    assert find_stadium(write_video(tmp_path / f"blank{value}.mp4", [blank] * 30), model, cfg) is None


def test_disagreeing_frames_return_none(tmp_path, synth_model):
    # A camera panning through the whole window: no two sampled frames show the same stadium.
    cfg, model = synth_model
    frames = [_frame_for(similarity(scale=0.8, shift=(-150 + 5 * i, 0))) for i in range(60)]
    assert find_stadium(write_video(tmp_path / "pan.mp4", frames), model, cfg) is None


# --------------------------------------------------------------------------- Task 5: visibility
def test_zone_visibility_out_of_frame():
    cal = synthetic_calibration()
    x0, y0 = cal.xtreme[0]
    x1, y1 = cal.xtreme[2]
    half = (x1 - x0) / 2
    moved = [(x + (W - x0 - half), y) for x, y in cal.xtreme]  # half of xtreme past the right edge
    from dataclasses import replace
    vis = zone_visibility(replace(cal, xtreme=moved))
    assert vis["xtreme"] == pytest.approx(0.5, abs=0.02)
    assert vis["battle_area"] == pytest.approx(1.0) and vis["over_left"] == pytest.approx(1.0)
    assert set(vis) == {"stadium", "battle_area", "over_left", "over_right", "xtreme"}


def test_zone_visibility_degenerate():
    from dataclasses import replace
    cal = replace(synthetic_calibration(), over_left=[(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)])
    assert zone_visibility(cal)["over_left"] == 0.0


# --------------------------------------------------------------------------- Task 6: truth
def test_truth_zones_are_the_five_judged_zones():
    from bxjudge.calibration import PolygonCollector
    from bxjudge.stadium_finder import TRUTH_ZONES

    assert [z[0] for z in TRUTH_ZONES] == ["stadium", "battle_area", "over_left", "over_right", "xtreme"]
    col = PolygonCollector(zones=TRUTH_ZONES)
    for _ in TRUTH_ZONES:
        for p in [(0, 0), (10, 0), (10, 10)]:
            col.click(*p)
        assert col.commit()
    assert col.finished and set(col.done) == {z[0] for z in TRUTH_ZONES}


def test_zone_boundary_errors_identical_is_zero():
    from bxjudge.stadium_finder import zone_boundary_errors

    cal = synthetic_calibration()
    truth = {z: getattr(cal, z) for z in ("stadium", "battle_area", "over_left", "over_right", "xtreme")}
    errs = zone_boundary_errors(cal, truth, cal.frame_size, 1280)
    assert set(errs) == set(truth) and max(errs.values()) == pytest.approx(0.0, abs=1e-6)


def test_zone_boundary_errors_shifted_and_rescaled():
    from bxjudge.stadium_finder import zone_boundary_errors

    cal = synthetic_calibration()  # 1280 x 720 found zones
    # Truth clicked on the full 1920 x 1080 frame, 10 px (full-res) to the right.
    truth = {z: [(x * 1.5 + 10, y * 1.5) for x, y in getattr(cal, z)] for z in ("battle_area", "xtreme")}
    errs = zone_boundary_errors(cal, truth, (1920, 1080), 1280)
    assert set(errs) == {"battle_area", "xtreme"}
    assert errs["battle_area"] == pytest.approx(10 * 1280 / 1920, abs=0.3)


def test_model_roundtrip(tmp_path):
    cfg = load_stadium_finder_config()
    model = build_stadium_model(synthetic_calibration(), draw_rail(rail_polyline()), cfg)
    path = tmp_path / "model.json"
    save_stadium_model(model, path)
    back = load_stadium_model(path)
    assert np.array_equal(back.rail, model.rail)
    assert back.calibration == model.calibration
    assert back.source_sha256 == model.source_sha256 and len(back.source_sha256) == 64
