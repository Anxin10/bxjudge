import numpy as np
import pytest

from bxjudge.autocalib import (
    estimate_alignment,
    load_autocalib_config,
    mean_boundary_error,
    warp_calibration,
)
from tests.helpers import stadium_scene


def test_recovers_shift_rotation_scale_within_tolerance():
    tpl_img, tpl = stadium_scene()
    frame, truth = stadium_scene(shift=(40, -25), rotation_deg=3.0, scale=1.06)
    cfg = load_autocalib_config()
    a = estimate_alignment(frame, tpl, tpl_img, cfg)
    assert a is not None and a.score >= cfg.min_score
    got = warp_calibration(tpl, a, (1280, 720), "auto")
    assert mean_boundary_error(got, truth, 1280) < 2.0


def test_black_floor_aligns_to_white_template():
    tpl_img, tpl = stadium_scene(floor="white")
    frame, truth = stadium_scene(floor="black", shift=(20, 10))
    a = estimate_alignment(frame, tpl, tpl_img, load_autocalib_config())
    assert a is not None
    assert mean_boundary_error(warp_calibration(tpl, a, (1280, 720), "auto"), truth, 1280) < 3.0


@pytest.mark.parametrize("frame_kind", ["rotated_30deg", "unrelated_noise", "blank"])
def test_out_of_range_or_unrelated_frames_are_rejected(frame_kind):
    tpl_img, tpl = stadium_scene()
    frame = {"rotated_30deg": stadium_scene(rotation_deg=30.0)[0],
             "unrelated_noise": np.random.default_rng(0).integers(0, 255, (720, 1280, 3), np.uint8),
             "blank": np.zeros((720, 1280, 3), np.uint8)}[frame_kind]
    assert estimate_alignment(frame, tpl, tpl_img, load_autocalib_config()) is None


def test_hand_over_rim_is_accurate_or_rejected():
    tpl_img, tpl = stadium_scene()
    frame, truth = stadium_scene(shift=(30, 0))
    frame[0:260, 300:700] = (60, 90, 140)  # skin-coloured block over the far rim
    a = estimate_alignment(frame, tpl, tpl_img, load_autocalib_config())
    if a is not None:
        assert mean_boundary_error(warp_calibration(tpl, a, (1280, 720), "auto"), truth, 1280) < 8.0


def test_mean_boundary_error_identity_and_offset():
    _, c = stadium_scene()
    assert mean_boundary_error(c, c, 1280) == pytest.approx(0.0, abs=1e-6)
    _, moved = stadium_scene(shift=(10, 0))
    assert 8.0 < mean_boundary_error(c, moved, 1280) < 12.0


def test_config_rejects_bad_values(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("work_width: 640\nmin_score: 1.5\n")
    with pytest.raises(ValueError):
        load_autocalib_config(p)


def test_config_loads_shipped_values():
    c = load_autocalib_config()
    assert (c.work_width, c.min_score, c.max_shift_frac, c.max_rotation_deg) == (640, 0.6, 0.15, 8.0)
    assert (c.min_scale, c.max_scale, c.max_perspective) == (0.85, 1.15, 0.0005)
    assert (c.ecc_iterations, c.ecc_epsilon, c.max_boundary_error_px) == (200, 1.0e-6, 8.0)


@pytest.mark.parametrize("key, value", [
    ("min_score", "1.5"), ("min_score", "-0.1"), ("min_scale", "1.2"), ("max_scale", "0.85"),
    ("max_perspective", ".nan"), ("ecc_epsilon", ".inf"), ("work_width", "640.0"),
    ("ecc_iterations", "true"), ("extra_key", "1"),
])
def test_config_rejects_each_bad_value(tmp_path, key, value):
    from pathlib import Path

    lines = [ln for ln in Path("config/autocalib.yaml").read_text().splitlines()
             if not ln.startswith(f"{key}:")]
    p = tmp_path / "a.yaml"
    p.write_text("\n".join([*lines, f"{key}: {value}"]) + "\n")
    with pytest.raises(ValueError):
        load_autocalib_config(p)


def test_warp_calibration_moves_all_zones_and_clears_ref_frame():
    tpl_img, tpl = stadium_scene()
    frame, truth = stadium_scene(shift=(-35, 20), rotation_deg=-2.0, scale=0.95)
    a = estimate_alignment(frame, tpl, tpl_img, load_autocalib_config())
    assert a is not None
    got = warp_calibration(tpl, a, (1280, 720), "auto")
    assert (got.name, got.frame_size, got.ref_frame) == ("auto", (1280, 720), "")
    for zone in ("stadium", "lid", "battle_area", "over_left", "over_right", "xtreme",
                 "launch_left", "launch_right"):
        assert len(getattr(got, zone)) == len(getattr(tpl, zone))
        assert np.abs(np.array(getattr(got, zone)) - np.array(getattr(truth, zone))).max() < 3.0
    assert a.rotation_deg == pytest.approx(-2.0, abs=0.3)
    assert a.scale == pytest.approx(0.95, abs=0.01)
