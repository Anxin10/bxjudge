"""``bxjudge autocalib``: acceptance/preview tool for auto-calibration (spec 6.6)."""

import re

import numpy as np

from bxjudge.calibration import save_calibration
from bxjudge.cli import main
from tests.helpers import add_manual_calibration, stadium_scene, write_scene_video

SHIFT = (45, 20)


def make_calib(tmp_path):
    calib = tmp_path / "calib"
    add_manual_calibration(calib, stadium_scene())
    return calib


def render_video(tmp_path, scene, name="clip.mp4"):
    return write_scene_video(tmp_path / name, scene[0], seconds=2.0)


def save_truth(tmp_path, calibration, name="truth.json"):
    path = tmp_path / name
    save_calibration(calibration, path)
    return path


def lines(capsys):
    return [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]


def test_autocalib_cli_reports_auto_and_passes_truth(tmp_path, capsys):
    calib = make_calib(tmp_path)
    scene = stadium_scene(shift=SHIFT)
    video = render_video(tmp_path, scene)
    truth = save_truth(tmp_path, scene[1])
    out_dir = tmp_path / "o"
    assert main(["autocalib", str(video), "--calib", str(calib), "--out", str(out_dir),
                 "--truth", str(truth)]) == 0
    (line,) = lines(capsys)
    assert re.fullmatch(r"clip\.mp4: auto name=auto-scene score=\d\.\d{3} "
                        r"boundary_error_px=\d+\.\d{2} PASS", line), line
    assert sorted(p.name for p in calib.iterdir()) == ["scene.json", "scene.png"]  # never written
    assert len(list((out_dir / "calibrations").glob("*.preview.jpg"))) == 1


def test_autocalib_cli_fails_on_bad_truth(tmp_path, capsys):
    calib = make_calib(tmp_path)
    video = render_video(tmp_path, stadium_scene(shift=SHIFT))
    truth = save_truth(tmp_path, stadium_scene(shift=(SHIFT[0] + 60, SHIFT[1]))[1])  # 60 px off
    assert main(["autocalib", str(video), "--calib", str(calib), "--out", str(tmp_path / "o"),
                 "--truth", str(truth)]) == 1
    (line,) = lines(capsys)
    assert line.startswith("clip.mp4: auto ") and line.endswith("FAIL"), line
    assert sorted(p.name for p in calib.iterdir()) == ["scene.json", "scene.png"]


def test_autocalib_cli_manual_match_and_several_videos(tmp_path, capsys):
    calib = make_calib(tmp_path)
    same = render_video(tmp_path, stadium_scene(), "same.mp4")
    moved = render_video(tmp_path, stadium_scene(shift=SHIFT), "moved.mp4")
    assert main(["autocalib", str(same), str(moved), "--calib", str(calib),
                 "--out", str(tmp_path / "o")]) == 0
    first, second = lines(capsys)
    assert re.fullmatch(r"same\.mp4: manual name=scene score=\d\.\d{3}", first), first
    assert re.fullmatch(r"moved\.mp4: auto name=auto-scene score=\d\.\d{3}", second), second


def test_autocalib_cli_recalibrate_is_not_a_failure(tmp_path, capsys):
    calib = make_calib(tmp_path)
    blank = (np.full((720, 1280, 3), 90, np.uint8), None)
    video = render_video(tmp_path, blank, "blank.mp4")
    truth = save_truth(tmp_path, stadium_scene()[1])
    out_dir = tmp_path / "o"
    assert main(["autocalib", str(video), "--calib", str(calib), "--out", str(out_dir),
                 "--truth", str(truth)]) == 0
    (line,) = lines(capsys)
    assert line == "blank.mp4: recalibrate name=- score=- boundary_error_px=- n/a", line
    assert not list((out_dir / "calibrations").glob("*.preview.jpg"))


def test_autocalib_cli_unreadable_video_is_an_error_line(tmp_path, capsys):
    calib = make_calib(tmp_path)
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    good = render_video(tmp_path, stadium_scene(), "good.mp4")
    code = main(["autocalib", str(broken), str(good), "--calib", str(calib),
                 "--out", str(tmp_path / "o")])
    captured = capsys.readouterr()
    assert code == 2
    assert "broken.mp4" in captured.err and "Traceback" not in captured.err
    assert "good.mp4: manual" in captured.out  # the next video is still judged
