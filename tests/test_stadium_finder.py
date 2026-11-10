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
    fit_stadium,
    load_stadium_finder_config,
    load_stadium_model,
    rail_mask,
    save_stadium_model,
)
from bxjudge.types import Calibration

W, H = 1280, 720


# --------------------------------------------------------------------------- synthetic scene
def rail_polyline(n: int = 720) -> np.ndarray:
    """Model-plane rail: an ellipse whose near (bottom) end has a notch, so it has an up/down."""
    cx, cy, a, b = 640.0, 360.0, 420.0, 230.0
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    x = cx + a * np.cos(t)
    y = cy + b * np.sin(t)
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


def _frame_for(h, *, cover=None, extra=None):
    img = draw_rail(apply_h(h, rail_polyline()))
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
    cfg, model = synth_model
    h = similarity(rot_deg=-8, scale=0.9, shift=(-20, 10))
    fit = fit_stadium(rail_mask(_frame_for(h, cover=(820, 0, 1280, 720)), cfg), model, cfg)
    assert fit is not None and zone_error(fit.homography, h) < 4.0


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


def test_model_roundtrip(tmp_path):
    cfg = load_stadium_finder_config()
    model = build_stadium_model(synthetic_calibration(), draw_rail(rail_polyline()), cfg)
    path = tmp_path / "model.json"
    save_stadium_model(model, path)
    back = load_stadium_model(path)
    assert np.array_equal(back.rail, model.rail)
    assert back.calibration == model.calibration
    assert back.source_sha256 == model.source_sha256 and len(back.source_sha256) == 64
