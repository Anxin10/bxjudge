import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from bxjudge.autocalib import load_autocalib_config, mean_boundary_error
from bxjudge.calibration import ZONES, save_calibration
from bxjudge.calibration_store import (
    load_snapshot,
    save_snapshot,
    select_snapshot,
    write_preview,
)
from bxjudge.config import load_thresholds
from tests.helpers import stadium_scene


def th():
    return load_thresholds("config/thresholds.yaml")


def make_calib_dir(tmp_path, scene, name="scene"):
    """A calib folder holding the scene as a manual calibration; returns (calib_dir, out_dir)."""
    img, calib = scene
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(calib_dir / f"{name}.png"), img)
    calib.name = name
    calib.ref_frame = str(calib_dir / f"{name}.png")
    save_calibration(calib, calib_dir / f"{name}.json")
    return calib_dir, tmp_path / "out"


def test_manual_wins_when_similar(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame, _ = stadium_scene()
    s = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert s.kind == "manual" and s.source_sha256 and s.template_id is None
    assert s.source_path == str(calib / "scene.json")
    assert (out / "calibrations" / f"{s.id}.json").is_file()


def test_shifted_frame_gets_auto_snapshot_with_its_own_reference(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame, truth = stadium_scene(shift=(45, 20))
    s = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert s.kind == "auto" and s.template_id
    assert s.homography is not None and s.score >= load_autocalib_config().min_score
    assert np.array_equal(cv2.imread(s.calibration.ref_frame), frame)
    assert mean_boundary_error(s.calibration, truth, 1280) < 3.0
    assert not list(calib.glob("auto*")) and len(list(calib.glob("*.json"))) == 1
    tpl = load_snapshot(out, s.template_id)
    assert tpl.kind == "manual" and tpl.calibration.name == "scene"


def test_second_video_at_same_position_reuses_the_auto_snapshot(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    b = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    assert a.kind == "auto" and a.id == b.id


def test_different_position_gets_a_distinct_auto_snapshot(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    b = select_snapshot(stadium_scene(shift=(-50, 25))[0], calib, out, th(), load_autocalib_config())
    assert a.kind == b.kind == "auto" and a.id != b.id


def test_snapshot_survives_manual_file_edit_and_folder_move(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    s = select_snapshot(stadium_scene()[0], calib, out, th(), load_autocalib_config())
    (calib / "scene.json").write_text("{}")
    moved = tmp_path / "moved"
    shutil.copytree(out, moved)
    again = load_snapshot(moved, s.id)
    assert again.calibration.stadium == s.calibration.stadium and Path(again.calibration.ref_frame).is_file()
    assert Path(again.calibration.ref_frame).parent == (moved / "calibrations").resolve()


def test_unreadable_template_is_skipped(tmp_path, caplog):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    (calib / "broken.json").write_text(json.dumps({**json.loads((calib / "scene.json").read_text()), "name": "broken", "ref_frame": "missing.png"}))
    assert select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config()).kind == "auto"
    assert "broken" in caplog.text


def test_no_match_returns_none(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    assert select_snapshot(np.zeros((720, 1280, 3), np.uint8), calib, out, th(), load_autocalib_config()) is None


def test_missing_or_empty_calib_dir_returns_none(tmp_path):
    out = tmp_path / "out"
    frame = stadium_scene()[0]
    assert select_snapshot(frame, tmp_path / "nope", out, th(), load_autocalib_config()) is None


def test_snapshot_id_is_content_addressed(tmp_path):
    img, c = stadium_scene()
    a = save_snapshot(tmp_path, "manual", c, img, score=1.0)
    b = save_snapshot(tmp_path, "manual", c, img, score=1.0)
    assert a.id == b.id and len(a.id) == 16
    other = save_snapshot(tmp_path, "manual", c, img, score=0.9)
    assert other.id != a.id
    assert len(list((tmp_path / "calibrations").glob("*.json"))) == 2


def test_load_unknown_snapshot_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path, "0123456789abcdef")


def test_preview_is_written_with_all_zones(tmp_path):
    img, c = stadium_scene()
    s = save_snapshot(tmp_path, "manual", c, img, score=0.95)
    p = write_preview(tmp_path, s)
    assert p == tmp_path / "calibrations" / f"{s.id}.preview.jpg"
    out = cv2.imread(str(p))
    assert out is not None and out.shape[:2] == img.shape[:2]
    assert len(ZONES) == 8
