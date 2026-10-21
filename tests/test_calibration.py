import numpy as np
import pytest

from bxjudge import calibration
from bxjudge.calibration import (
    PolygonCollector,
    calibration_similarity,
    find_calibration,
    load_calibration,
    save_calibration,
)
from bxjudge.cli import build_parser, main
from tests.helpers import cal_with, make_calibration


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


def test_find_calibration_none_when_no_match(tmp_path, ref_img):
    save_calibration(cal_with(ref_img, tmp_path), tmp_path / "a.json")
    assert find_calibration(np.roll(ref_img, 40, axis=1), tmp_path, 0.8) is None


def test_find_calibration_picks_best_match(tmp_path, ref_img):
    from tests.helpers import make_ref_img

    save_calibration(cal_with(make_ref_img(seed=1), tmp_path, name="other"), tmp_path / "other.json")
    save_calibration(cal_with(ref_img, tmp_path, name="mine"), tmp_path / "mine.json")
    found = find_calibration(ref_img, tmp_path, 0.8)
    assert found is not None and found.name == "mine"


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


def test_cli_has_calibrate_subcommand():
    assert "calibrate" in build_parser()._subparsers._group_actions[0].choices
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


def test_interactive_calibrate_importable():
    assert callable(calibration.interactive_calibrate)
