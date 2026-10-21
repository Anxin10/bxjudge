import cv2
import numpy as np

from bxjudge.geometry import polygon_mask
from bxjudge.stadium_motion import stadium_displacement
from tests.helpers import draw_blob, make_calibration, shift


def test_detects_shift(ref_img):
    frames = [(0, ref_img), (1, shift(ref_img, 3, 0)), (2, shift(ref_img, 0, -5))]
    d = stadium_displacement(frames, make_calibration(), ref_img)
    assert abs(d[0]) < 0.5 and abs(d[1] - 3) < 0.5 and abs(d[2] - 5) < 0.5


def test_ignores_motion_inside_battle_area(ref_img):
    moved = draw_blob(ref_img.copy(), (300, 170), r=15)  # a "bey" inside the battle area
    assert stadium_displacement([(0, moved)], make_calibration(), ref_img)[0] < 0.5


def test_diagonal_shift_is_vector_length(ref_img):
    d = stadium_displacement([(0, shift(ref_img, 3, 4))], make_calibration(), ref_img)
    assert abs(d[0] - 5) < 0.5


def test_frames_smaller_than_calibration(ref_img):
    """Calibration and ref are 640x360; decoded frames are 320x180. The result is in frame px."""
    small = cv2.resize(ref_img, (320, 180), interpolation=cv2.INTER_AREA)
    frames = [(10, small), (11, shift(small, 3, 0)), (12, shift(small, 0, -5))]
    d = stadium_displacement(frames, make_calibration(), ref_img)
    assert set(d) == {10, 11, 12}
    assert abs(d[10]) < 0.5 and abs(d[11] - 3) < 0.5 and abs(d[12] - 5) < 0.5


def test_blob_inside_battle_area_at_reduced_size(ref_img):
    small = cv2.resize(ref_img, (320, 180), interpolation=cv2.INTER_AREA)
    moved = draw_blob(small.copy(), (150, 85), r=8)
    assert stadium_displacement([(0, moved)], make_calibration(), ref_img)[0] < 0.5


def test_non_convergence_gives_nan(ref_img):
    flat = np.full_like(ref_img, 128)  # no texture to align on
    d = stadium_displacement([(0, flat), (1, ref_img)], make_calibration(), ref_img)
    assert np.isnan(d[0])
    assert abs(d[1]) < 0.5  # a bad frame does not poison the next one


def test_empty_input(ref_img):
    assert stadium_displacement([], make_calibration(), ref_img) == {}


def test_accepts_generator_and_gray_frames(ref_img):
    gray = cv2.cvtColor(shift(ref_img, 3, 0), cv2.COLOR_BGR2GRAY)
    d = stadium_displacement(((i, f) for i, f in [(5, gray)]), make_calibration(), ref_img)
    assert abs(d[5] - 3) < 0.5


def _blob_moves_between_ref_and_frame(ref_img):
    """A "bey" at A in the reference and at B in the frame, both inside battle_area."""
    ref = draw_blob(ref_img.copy(), (200, 170), r=50)
    frame = draw_blob(ref_img.copy(), (440, 170), r=50)
    return ref, frame


def test_blob_moving_inside_battle_area_is_ignored(ref_img):
    ref, frame = _blob_moves_between_ref_and_frame(ref_img)
    d = stadium_displacement([(0, frame)], make_calibration(), ref)[0]
    assert d < 0.5


def test_blob_moving_inside_battle_area_with_real_shift(ref_img):
    ref, frame = _blob_moves_between_ref_and_frame(ref_img)
    d = stadium_displacement([(0, shift(frame, 3, 0))], make_calibration(), ref)[0]
    assert abs(d - 3) < 0.5


def test_negative_control_full_stadium_mask_is_disturbed_by_the_blob(ref_img, monkeypatch):
    """Proves the test above discriminates: with battle_area INCLUDED in the mask, the moving
    blob wrecks the alignment (nan or a large value), so the real mask is what ignores it."""
    import bxjudge.stadium_motion as sm

    def full_stadium(c, size, margin):
        k = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
        poly = cv2.resize(polygon_mask(c.stadium, c.frame_size), size, interpolation=cv2.INTER_NEAREST)
        return cv2.erode(poly, k) > 0

    monkeypatch.setattr(sm, "rim_mask", full_stadium)
    ref, frame = _blob_moves_between_ref_and_frame(ref_img)
    d = sm.stadium_displacement([(0, frame)], make_calibration(), ref)[0]
    assert np.isnan(d) or d >= 0.5
