"""cli.main: expected user errors are one `error:` line + exit 2; bugs keep their traceback."""

from pathlib import Path

import pytest
import yaml

from bxjudge import pipeline
from bxjudge.cli import main
from tests.helpers import make_video

ROOT = Path(__file__).resolve().parents[1]
BAD_YAML = "targets: [1\n"  # unterminated flow sequence


def _user_error(capsys, argv: list[str], *needles: str) -> str:
    assert main(argv) == 2
    captured = capsys.readouterr()
    err = captured.err
    assert err.startswith("error: "), err
    assert "Traceback" not in err and "Traceback" not in captured.out
    for needle in needles:
        assert needle in err
    return err


def _bad_yaml(tmp_path, name="bad.yaml"):
    path = tmp_path / name
    path.write_text(BAD_YAML, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- calibrate


def test_calibrate_missing_video_is_error_line(tmp_path, capsys):
    _user_error(capsys, ["calibrate", str(tmp_path / "missing.mp4"), "--name", "x",
                         "--out", str(tmp_path / "c")], "missing.mp4")
    assert not (tmp_path / "c").exists()


def test_calibrate_unreadable_video_is_error_line(tmp_path, capsys):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    _user_error(capsys, ["calibrate", str(broken), "--name", "x", "--out", str(tmp_path / "c")],
                "broken.mp4")


# --------------------------------------------------------------------------- train / compare


def test_train_missing_dataset_is_error_line(tmp_path, capsys):
    _user_error(capsys, ["train", "--kind", "yolo26", "--dataset", str(tmp_path / "missingds"),
                         "--out", str(tmp_path / "tr"), "--epochs", "1"], "missingds")


def test_train_malformed_yaml_config_is_error_line(tmp_path, capsys):
    _user_error(capsys, ["train", "--kind", "yolo26", "--dataset", str(tmp_path / "ds"),
                         "--out", str(tmp_path / "tr"), "--config", str(_bad_yaml(tmp_path))],
                "bad.yaml")


def test_compare_missing_dataset_is_error_line(tmp_path, capsys):
    _user_error(capsys, ["compare", "--dataset", str(tmp_path / "missingds"), "--rfdetr", "a.pt",
                         "--yolo26", "b.pt", "--output", str(tmp_path / "cmp.json"),
                         "--selection-out", str(tmp_path / "sel.json")], "missingds")


def test_compare_malformed_yaml_config_is_error_line(tmp_path, capsys):
    _user_error(capsys, ["compare", "--dataset", str(tmp_path / "ds"), "--rfdetr", "a.pt",
                         "--yolo26", "b.pt", "--config", str(_bad_yaml(tmp_path))], "bad.yaml")


# --------------------------------------------------------------------------- label


def test_label_sample_empty_folder_is_error_line(tmp_path, capsys):
    (tmp_path / "vids").mkdir()
    _user_error(capsys, ["label", "sample", str(tmp_path / "vids"), "--out",
                         str(tmp_path / "lab")], "no MP4 videos")


def test_label_sample_malformed_yaml_is_error_line(tmp_path, capsys):
    _user_error(capsys, ["label", "sample", str(tmp_path), "--config", str(_bad_yaml(tmp_path))],
                "bad.yaml")


def test_label_config_type_error_from_validation_is_error_line(tmp_path, capsys):
    # A wrongly shaped value makes validation itself raise TypeError (set(5)):
    # it is still a config mistake, reported with the file name.
    raw = yaml.safe_load((ROOT / "config" / "labeling.yaml").read_text(encoding="utf-8"))
    raw["prompts"] = 5
    path = tmp_path / "labeling.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    _user_error(capsys, ["label", "sample", str(tmp_path), "--config", str(path)],
                "labeling.yaml", "TypeError")


# --------------------------------------------------------------------------- eval / tune / run


def _results(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    (res / "rounds.json").write_text("[]", encoding="utf-8")
    return res


def test_eval_malformed_yaml_is_error_line(tmp_path, capsys):
    _user_error(capsys, ["eval", str(_results(tmp_path)), "--config", str(_bad_yaml(tmp_path)),
                         "--report", str(tmp_path / "r.md")], "bad.yaml")
    assert not (tmp_path / "r.md").exists()


def test_eval_config_wrong_value_type_is_error_line(tmp_path, capsys):
    raw = yaml.safe_load((ROOT / "config" / "evaluation.yaml").read_text(encoding="utf-8"))
    raw["targets"]["review_ratio"] = {"max": [0.3]}  # wrong type: a list, not a number
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    _user_error(capsys, ["eval", str(_results(tmp_path)), "--config", str(path),
                         "--report", str(tmp_path / "r.md")], "evaluation.yaml")


def test_tune_malformed_thresholds_yaml_is_error_line(tmp_path, capsys):
    _user_error(capsys, ["tune", str(_results(tmp_path)),
                         "--thresholds", str(_bad_yaml(tmp_path, "thresholds.yaml")),
                         "--report", str(tmp_path / "t.md")], "thresholds.yaml")


def test_run_malformed_rules_yaml_is_error_line(tmp_path, capsys):
    make_video(tmp_path / "a.mp4", seconds=0.1)
    (tmp_path / "calib").mkdir()
    _user_error(capsys, ["run", str(tmp_path), "--calib", str(tmp_path / "calib"),
                         "--out", str(tmp_path / "out"),
                         "--rules", str(_bad_yaml(tmp_path, "rules.yaml"))], "rules.yaml")


def test_run_missing_selection_points_at_train_and_compare(tmp_path, capsys):
    make_video(tmp_path / "a.mp4", seconds=0.1)
    (tmp_path / "calib").mkdir()
    _user_error(capsys, ["run", str(tmp_path), "--calib", str(tmp_path / "calib"),
                         "--out", str(tmp_path / "out"),
                         "--selection", str(tmp_path / "selected.json")],
                "selected.json", "bxjudge compare")


# --------------------------------------------------------------------------- bugs stay loud


@pytest.mark.parametrize("bug", [TypeError("unsupported operand"), KeyError("finish"),
                                 AttributeError("x")])
def test_genuine_bugs_keep_their_traceback(tmp_path, monkeypatch, bug):
    def broken(*a, **k):
        raise bug

    monkeypatch.setattr(pipeline, "run_batch", broken)
    with pytest.raises(type(bug)):
        main(["run", str(tmp_path), "--no-vlm"])


def test_runtime_error_outside_run_keeps_traceback(tmp_path, monkeypatch):
    from bxjudge import manifest

    monkeypatch.setattr(manifest, "scan", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("internal")))
    with pytest.raises(RuntimeError, match="internal"):
        main(["scan", str(tmp_path), "--db", str(tmp_path / "db.json")])


def test_debug_flag_reraises_user_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["--debug", "calibrate", str(tmp_path / "missing.mp4"), "--name", "x",
              "--out", str(tmp_path / "c")])


def test_audit_sample_json_error_is_error_line(tmp_path, capsys):
    res = tmp_path / "res"
    res.mkdir()
    (res / "rounds.json").write_text("{not json", encoding="utf-8")
    _user_error(capsys, ["audit-sample", str(res), "--n", "1", "--seed", "1"])
