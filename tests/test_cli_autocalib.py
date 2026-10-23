"""``bxjudge autocalib``: acceptance/preview tool for auto-calibration (spec 6.6)."""

import re

import numpy as np

from bxjudge.calibration import save_calibration
from bxjudge.cli import main
from tests.helpers import add_manual_calibration, stadium_scene, write_scene_video

SHIFT = (45, 20)
ID_PREVIEW = r"id=([0-9a-f]{16}) preview=(\S+/calibrations/\1\.preview\.jpg)"


def make_calib(tmp_path):
    calib = tmp_path / "calib"
    add_manual_calibration(calib, stadium_scene())
    return calib


def render_video(tmp_path, scene, name="clip.mp4"):
    return write_scene_video(tmp_path / name, scene[0], seconds=2.0)


def blank_video(tmp_path, name="blank.mp4"):
    return render_video(tmp_path, (np.full((720, 1280, 3), 90, np.uint8), None), name)


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
    line, summary = lines(capsys)
    m = re.fullmatch(rf"clip\.mp4: auto name=auto-scene score=\d\.\d{{3}} {ID_PREVIEW} "
                     r"boundary_error_px=\d+\.\d{2} PASS", line)
    assert m, line
    assert summary == "PASS 1 / FAIL 0 / REJECTED 0"
    assert sorted(p.name for p in calib.iterdir()) == ["scene.json", "scene.png"]  # never written
    assert [p.name for p in (out_dir / "calibrations").glob("*.preview.jpg")] == \
        [f"{m.group(1)}.preview.jpg"]


def test_autocalib_cli_fails_on_bad_truth(tmp_path, capsys):
    calib = make_calib(tmp_path)
    video = render_video(tmp_path, stadium_scene(shift=SHIFT))
    truth = save_truth(tmp_path, stadium_scene(shift=(SHIFT[0] + 60, SHIFT[1]))[1])  # 60 px off
    assert main(["autocalib", str(video), "--calib", str(calib), "--out", str(tmp_path / "o"),
                 "--truth", str(truth)]) == 1
    line, summary = lines(capsys)
    assert line.startswith("clip.mp4: auto ") and line.endswith("FAIL"), line
    assert summary == "PASS 0 / FAIL 1 / REJECTED 0"
    assert sorted(p.name for p in calib.iterdir()) == ["scene.json", "scene.png"]


def test_autocalib_cli_manual_match_and_several_videos(tmp_path, capsys):
    calib = make_calib(tmp_path)
    same = render_video(tmp_path, stadium_scene(), "same.mp4")
    moved = render_video(tmp_path, stadium_scene(shift=SHIFT), "moved.mp4")
    assert main(["autocalib", str(same), str(moved), "--calib", str(calib),
                 "--out", str(tmp_path / "o")]) == 0
    first, second = lines(capsys)  # no --truth: no summary line
    assert re.fullmatch(rf"same\.mp4: manual name=scene score=\d\.\d{{3}} {ID_PREVIEW}", first), first
    assert re.fullmatch(
        rf"moved\.mp4: auto name=auto-scene score=\d\.\d{{3}} {ID_PREVIEW}", second), second


def test_autocalib_cli_lines_let_the_reader_find_each_preview(tmp_path, capsys):
    calib = make_calib(tmp_path)
    a = render_video(tmp_path, stadium_scene(shift=SHIFT), "a.mp4")
    b = render_video(tmp_path, stadium_scene(shift=SHIFT), "b.mp4")  # same position: same group
    assert main(["autocalib", str(a), str(b), "--calib", str(calib),
                 "--out", str(tmp_path / "o")]) == 0
    first, second = lines(capsys)
    ids = [re.search(r" id=(\w+) preview=(\S+)", ln).groups() for ln in (first, second)]
    assert ids[0] == ids[1]
    assert (tmp_path / "o" / "calibrations" / f"{ids[0][0]}.preview.jpg").is_file()


def test_autocalib_cli_manual_result_with_truth_passes(tmp_path, capsys):
    calib = make_calib(tmp_path)
    video = render_video(tmp_path, stadium_scene())
    truth = save_truth(tmp_path, stadium_scene()[1])
    assert main(["autocalib", str(video), "--calib", str(calib), "--out", str(tmp_path / "o"),
                 "--truth", str(truth)]) == 0
    line, summary = lines(capsys)
    assert " manual " in line and line.endswith("PASS") and summary == "PASS 1 / FAIL 0 / REJECTED 0"


def test_autocalib_cli_recalibrate_with_truth_is_rejected_and_fails(tmp_path, capsys):
    calib = make_calib(tmp_path)
    video = blank_video(tmp_path)
    truth = save_truth(tmp_path, stadium_scene()[1])
    out_dir = tmp_path / "o"
    assert main(["autocalib", str(video), "--calib", str(calib), "--out", str(out_dir),
                 "--truth", str(truth)]) == 1
    line, summary = lines(capsys)
    assert line == "blank.mp4: recalibrate name=- score=- boundary_error_px=- REJECTED", line
    assert summary == "PASS 0 / FAIL 0 / REJECTED 1"
    assert not list((out_dir / "calibrations").glob("*.preview.jpg"))


def test_autocalib_cli_allow_recalibrate_keeps_rejected_visible_but_exits_0(tmp_path, capsys):
    calib = make_calib(tmp_path)
    video = blank_video(tmp_path)
    truth = save_truth(tmp_path, stadium_scene()[1])
    assert main(["autocalib", str(video), "--calib", str(calib), "--out", str(tmp_path / "o"),
                 "--truth", str(truth), "--allow-recalibrate"]) == 0
    line, summary = lines(capsys)
    assert line.endswith("boundary_error_px=- REJECTED") and summary == "PASS 0 / FAIL 0 / REJECTED 1"


def test_autocalib_cli_allow_recalibrate_still_fails_on_a_misaligned_result(tmp_path, capsys):
    calib = make_calib(tmp_path)
    bad = render_video(tmp_path, stadium_scene(shift=SHIFT), "bad.mp4")
    blank = blank_video(tmp_path)
    truth = save_truth(tmp_path, stadium_scene(shift=(SHIFT[0] + 60, SHIFT[1]))[1])
    assert main(["autocalib", str(bad), str(blank), "--calib", str(calib),
                 "--out", str(tmp_path / "o"), "--truth", str(truth),
                 "--allow-recalibrate"]) == 1
    assert lines(capsys)[-1] == "PASS 0 / FAIL 1 / REJECTED 1"


def test_autocalib_cli_recalibrate_without_truth_exits_0(tmp_path, capsys):
    calib = make_calib(tmp_path)
    video = blank_video(tmp_path)
    assert main(["autocalib", str(video), "--calib", str(calib), "--out", str(tmp_path / "o")]) == 0
    assert lines(capsys) == ["blank.mp4: recalibrate name=- score=-"]


def test_autocalib_cli_missing_calib_folder_is_an_error_line(tmp_path, capsys):
    video = render_video(tmp_path, stadium_scene())
    assert main(["autocalib", str(video), "--calib", str(tmp_path / "typo"),
                 "--out", str(tmp_path / "o")]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ") and "typo" in captured.err
    assert "Traceback" not in captured.err and captured.out == ""
    assert not (tmp_path / "o").exists()


def test_autocalib_cli_calib_that_is_a_file_is_an_error_line(tmp_path, capsys):
    video = render_video(tmp_path, stadium_scene())
    not_a_dir = tmp_path / "calib.json"
    not_a_dir.write_text("{}")
    assert main(["autocalib", str(video), "--calib", str(not_a_dir),
                 "--out", str(tmp_path / "o")]) == 2
    assert capsys.readouterr().err.startswith("error: ")


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
