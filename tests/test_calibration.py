import numpy as np
import pytest

from bxjudge import calibration
from bxjudge.calibration import (
    PolygonCollector,
    calibration_similarity,
    load_calibration,
    save_calibration,
)
from bxjudge.cli import main
from bxjudge.geometry import polygon_mask
import cv2

from tests.helpers import cal_with, make_calibration, make_ref_img, make_smooth_img


def test_roundtrip(tmp_path):
    c = make_calibration()
    save_calibration(c, tmp_path / "c.json")
    assert load_calibration(tmp_path / "c.json") == c


def test_roundtrip_ref_frame_is_relative_in_json(tmp_path, ref_img):
    c = cal_with(ref_img, tmp_path)
    save_calibration(c, tmp_path / "c.json")
    assert '"ref_frame": "test.png"' in (tmp_path / "c.json").read_text()
    assert load_calibration(tmp_path / "c.json") == c


def test_similarity_identical_high(tmp_path, ref_img):
    assert calibration_similarity(ref_img, cal_with(ref_img, tmp_path)) > 0.95


def test_similarity_shifted_low(tmp_path, ref_img):
    shifted = np.roll(ref_img, 40, axis=1)
    assert calibration_similarity(shifted, cal_with(ref_img, tmp_path)) < 0.8


def test_polygon_collector_order_undo_and_min_points():
    col = PolygonCollector()
    assert col.current.startswith("stadium")
    col.click(0, 0)
    col.click(10, 0)
    assert not col.commit()  # fewer than 3 points
    col.click(10, 10)
    col.click(99, 99)
    col.undo()
    assert col.commit()
    assert col.done["stadium"] == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    assert col.current.startswith("lid")


def test_cli_calibrate_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        main(["calibrate", "--help"])
    assert e.value.code == 0


def test_cli_calibrate_wires_to_interactive_calibrate(tmp_path, ref_img, monkeypatch):
    monkeypatch.setattr(calibration, "grab_frame", lambda video, at: ref_img)
    monkeypatch.setattr(calibration, "interactive_calibrate", lambda frame, name: make_calibration())
    rc = main(["calibrate", "v.mp4", "--at", "1.0", "--name", "n1", "--out", str(tmp_path / "calib")])
    assert rc == 0
    assert (tmp_path / "calib" / "n1.png").exists()
    c = load_calibration(tmp_path / "calib" / "n1.json")
    assert c.ref_frame == str(tmp_path / "calib" / "n1.png")
    assert calibration_similarity(ref_img, c) > 0.95


@pytest.mark.parametrize("shift", [10, 15, 20])
def test_smooth_content_small_shift_detected(tmp_path, shift):
    img = make_smooth_img()
    c = cal_with(img, tmp_path)
    assert calibration_similarity(img, c) > 0.95
    assert calibration_similarity(np.roll(img, shift, axis=1), c) < 0.8
    assert calibration_similarity(np.roll(img, shift, axis=0), c) < 0.8


def test_change_outside_stadium_ignored(tmp_path):
    img = make_smooth_img()
    c = cal_with(img, tmp_path)
    changed = img.copy()
    changed[:15] = make_ref_img(seed=5)[:15]  # y < 20: above the stadium polygon (y 20..340)
    changed[:, :30] = 0  # x < 40: left of the stadium
    assert calibration_similarity(changed, c) > 0.95


def test_change_inside_masked_region_drops_similarity(tmp_path):
    img = make_smooth_img()
    c = cal_with(img, tmp_path)
    changed = img.copy()
    ring = polygon_mask(c.stadium, c.frame_size) > 0
    ring &= polygon_mask(c.battle_area, c.frame_size) == 0  # rim + lid region only
    changed[ring] = make_ref_img(seed=6)[ring]
    assert calibration_similarity(changed, c) < 0.8


def test_change_inside_battle_area_ignored(tmp_path):
    """Beys, launchers and hands in the reference/test frame must not break a genuine match."""
    img = make_smooth_img()
    c = cal_with(img, tmp_path)
    changed = img.copy()
    cv2.circle(changed, (320, 200), 40, (255, 255, 255), -1)
    changed[150:250, 200:260] = make_ref_img(seed=7)[150:250, 200:260]
    assert calibration_similarity(changed, c) > 0.95


@pytest.mark.parametrize("gain,offset", [(1.0, 30), (0.8, 0), (1.2, -20), (0.7, 25)])
def test_brightness_gain_offset_keeps_similarity_high(tmp_path, gain, offset):
    img = make_smooth_img()
    c = cal_with(img, tmp_path)
    bright = np.clip(img.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
    assert calibration_similarity(bright, c) > 0.9


def test_edge_map_is_reproducible_whatever_ran_before():
    # OpenCV's IPP magnitude gave a different last bit for the same input depending on the
    # magnitude calls before it, so one first-second frame got a different auto snapshot
    # (homography, id) and stadium edge map depending on what the batch had run earlier.
    rng = np.random.default_rng(3)
    gray = (rng.random((360, 640)) * 255).astype(np.float32)
    before = calibration._edge_map(gray)
    for shape in [(90, 160), (45, 80), (720, 1280), (7, 13), (180, 320), (360, 640)]:
        other = rng.random(shape).astype(np.float32)
        cv2.magnitude(other, other)
        np.testing.assert_array_equal(calibration._edge_map(gray), before)
