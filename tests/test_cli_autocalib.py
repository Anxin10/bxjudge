"""Acceptance CLI exercises real local encoding, decoding, selection and truth metrics."""
import json

import cv2
import numpy as np
import pytest

from bxjudge.calibration import save_calibration
from bxjudge.calibration_store import load_snapshot, veto_calibration
from bxjudge.cli import main
from tests.helpers import add_manual_calibration, stadium_scene, write_scene_video


@pytest.fixture
def inputs(tmp_path):
    calib = tmp_path / "calib"
    add_manual_calibration(calib, stadium_scene())
    moved = stadium_scene(shift=(45, 20))
    video = write_scene_video(tmp_path / "moved.mp4", moved[0], seconds=1.5, fps=4)
    truth = tmp_path / "truth.json"
    save_calibration(moved[1], truth)
    return calib, video, truth, tmp_path / "out"


def argv(inputs, *, truth=True):
    calib, video, reference, out = inputs
    return ["autocalib", str(video), "--calib", str(calib), "--out", str(out),
            *(["--truth", str(reference)] if truth else [])]


def test_autocalib_cli_reports_auto_and_passes_truth(inputs, capsys):
    calib, _, _, out = inputs
    before = {p.name: p.read_bytes() for p in calib.iterdir()}
    assert main(argv(inputs)) == 0
    printed = capsys.readouterr().out
    assert ": auto name=auto-scene score=" in printed
    assert "boundary_error_px=" in printed and "PASS" in printed
    snapshots = [load_snapshot(out, p.stem) for p in (out / "calibrations").glob("*.json")]
    auto = next(s for s in snapshots if s.kind == "auto")
    preview = out / "calibrations" / f"{auto.id}.preview.jpg"
    assert cv2.imread(str(preview)).shape == (720, 1280, 3)
    assert {p.name: p.read_bytes() for p in calib.iterdir()} == before


def test_autocalib_cli_fails_on_bad_truth(inputs, capsys):
    _, _, truth, _ = inputs
    save_calibration(stadium_scene(shift=(105, 20))[1], truth)  # 60 px from the video
    assert main(argv(inputs)) == 1
    printed = capsys.readouterr().out
    assert "boundary_error_px=" in printed and "FAIL" in printed and "PASS" not in printed


def test_rejected_video_with_truth_cannot_pass(inputs, tmp_path, capsys):
    calib, _, truth, out = inputs
    blank = write_scene_video(tmp_path / "blank.mp4", np.zeros((720, 1280, 3), np.uint8),
                              seconds=1.5, fps=4)
    assert main(argv((calib, blank, truth, out))) == 1
    printed = capsys.readouterr().out
    assert ": recalibrate name=none score=0.000" in printed
    assert "boundary_error_px=unverified FAIL" in printed and "PASS" not in printed
    assert not list(out.rglob("*.preview.jpg"))


def test_no_truth_is_explicitly_unverified(inputs, capsys):
    assert main(argv(inputs, truth=False)) == 0
    printed = capsys.readouterr().out
    assert ": auto " in printed and "unverified" in printed
    assert "PASS" not in printed and "FAIL" not in printed


def test_multiple_inputs_report_all_and_keep_any_failure(inputs, tmp_path, capsys):
    calib, video, truth, out = inputs
    blank = write_scene_video(tmp_path / "blank.mp4", np.zeros((720, 1280, 3), np.uint8),
                              seconds=1.5, fps=4)
    args = ["autocalib", str(blank), str(video), "--calib", str(calib),
            "--out", str(out), "--truth", str(truth)]
    assert main(args) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith(f"{blank}: recalibrate") and "FAIL" in lines[0]
    assert lines[1].startswith(f"{video}: auto") and "PASS" in lines[1]


def test_manual_selection_and_first_second_frame(inputs, tmp_path, capsys, monkeypatch):
    from bxjudge import calibration

    calib, _, truth, out = inputs
    video = write_scene_video(tmp_path / "manual.mp4", stadium_scene()[0], seconds=1.5, fps=4)
    seen = []
    original = calibration.grab_frame

    def grab(path, at):
        seen.append((path, at))
        return original(path, at)

    monkeypatch.setattr(calibration, "grab_frame", grab)
    assert main(argv((calib, video, truth, out), truth=False)) == 0
    assert ": manual name=scene score=" in capsys.readouterr().out
    assert seen == [(video, 1.0)]


def test_existing_veto_is_honoured(inputs, capsys):
    assert main(argv(inputs, truth=False)) == 0
    out = inputs[3]
    auto = next(load_snapshot(out, p.stem) for p in (out / "calibrations").glob("*.json")
                if load_snapshot(out, p.stem).kind == "auto")
    veto_calibration(out, auto.id)
    capsys.readouterr()
    assert main(argv(inputs)) == 1
    assert ": recalibrate " in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["video", "config", "calib", "truth", "out"])
def test_input_errors_follow_cli_conventions(inputs, tmp_path, capsys, bad):
    calib, video, truth, out = inputs
    args = argv(inputs)
    target = {"video": video, "calib": calib, "truth": truth, "out": out}
    if bad == "config":
        config = tmp_path / "bad.yaml"
        config.write_text("min_score: nope\n")
        args += ["--autocalib", str(config)]
    else:
        args[args.index(str(target[bad]))] = str(tmp_path / "missing")
        if bad == "out":
            (tmp_path / "missing").write_text("a file, not a directory")
    assert main(args) == 2
    assert capsys.readouterr().err.startswith("error: ")


@pytest.mark.parametrize("bad", [{}, [], {"frame_size": [0, 720]}, {"stadium": []},
                                {"stadium": [[float("nan"), 0]] * 3}])
def test_invalid_truth_is_an_input_error(inputs, capsys, bad):
    truth = inputs[2]
    data = json.loads(truth.read_text())
    truth.write_text(json.dumps({**data, **bad} if isinstance(bad, dict) and bad else bad))
    assert main(argv(inputs)) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_out_cannot_write_inside_source_template_directory(inputs, capsys):
    calib, video, truth, _ = inputs
    before = {p.name: p.read_bytes() for p in calib.iterdir()}
    assert main(argv((calib, video, truth, calib))) == 2
    assert capsys.readouterr().err.startswith("error: ")
    assert {p.name: p.read_bytes() for p in calib.iterdir()} == before


def test_custom_acceptance_policy_is_used(inputs, tmp_path, capsys):
    from pathlib import Path

    _, _, truth, _ = inputs
    save_calibration(stadium_scene(shift=(105, 20))[1], truth)
    config = tmp_path / "wide.yaml"
    config.write_text(Path("config/autocalib.yaml").read_text().replace(
        "max_boundary_error_px: 8.0", "max_boundary_error_px: 80.0"))
    assert main([*argv(inputs), "--autocalib", str(config)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_output_calibrations_symlink_cannot_write_source(inputs, capsys):
    calib, video, truth, out = inputs
    out.mkdir()
    (out / "calibrations").symlink_to(calib, target_is_directory=True)
    before = {p.name: p.read_bytes() for p in calib.iterdir()}
    assert main(argv((calib, video, truth, out))) == 2
    assert capsys.readouterr().err.startswith("error: ")
    assert {p.name: p.read_bytes() for p in calib.iterdir()} == before


def test_truth_metric_uses_inference_pixels_across_resolutions(inputs, capsys):
    from bxjudge.calibration import load_calibration, scale_calibration

    truth = inputs[2]
    save_calibration(scale_calibration(load_calibration(truth), (1920, 1080)), truth)
    assert main(argv(inputs)) == 0
    assert "PASS" in capsys.readouterr().out


@pytest.mark.parametrize("error", [8.0, float("nan"), float("inf")])
def test_boundary_error_at_limit_or_nonfinite_fails(inputs, capsys, monkeypatch, error):
    from bxjudge import autocalib

    monkeypatch.setattr(autocalib, "mean_boundary_error", lambda *args: error)
    assert main(argv(inputs)) == 1
    assert f"boundary_error_px={error:.2f} FAIL" in capsys.readouterr().out


def test_output_symlink_to_source_descendant_cannot_change_source_subtree(inputs, capsys):
    calib, video, truth, out = inputs
    generated = calib / "generated"
    generated.mkdir()
    (generated / "keep.txt").write_bytes(b"preserve existing source descendant bytes")
    out.mkdir()
    (out / "calibrations").symlink_to(generated, target_is_directory=True)
    before = {p.relative_to(calib): p.read_bytes() for p in calib.rglob("*") if p.is_file()}

    code = main(argv((calib, video, truth, out)))

    after = {p.relative_to(calib): p.read_bytes() for p in calib.rglob("*") if p.is_file()}
    assert after.keys() == before.keys()
    assert after == before
    assert code == 2
    assert capsys.readouterr().err.startswith("error: ")
