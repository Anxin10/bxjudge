"""CLI operability: --debug anywhere, Ctrl+C."""
import pytest

from bxjudge import cli


def test_debug_after_subcommand_reraises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.main(["calibrate", str(tmp_path / "missing.mp4"), "--name", "x",
                  "--out", str(tmp_path / "c"), "--debug"])


def test_debug_after_run_subcommand_reraises(tmp_path):
    with pytest.raises((ValueError, OSError)):
        cli.main(["run", str(tmp_path / "missing"), "--debug"])


def test_debug_after_nested_subcommand_reraises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.main(["label", "export", str(tmp_path / "missing"), "--debug"])


def test_debug_still_before_subcommand_and_off_by_default(tmp_path, capsys):
    with pytest.raises(FileNotFoundError):
        cli.main(["--debug", "calibrate", str(tmp_path / "m.mp4"), "--name", "x",
                  "--out", str(tmp_path / "c")])
    assert cli.main(["calibrate", str(tmp_path / "m.mp4"), "--name", "x",
                     "--out", str(tmp_path / "c")]) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_ctrl_c_is_cancelled_exit_130(monkeypatch, capsys):
    def interrupt(args):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "_cmd_calibrate", interrupt)
    assert cli.main(["calibrate", "missing", "--name", "x"]) == 130
    assert capsys.readouterr().err.strip() == "cancelled"


def test_runtime_error_is_still_not_caught(monkeypatch):
    def boom(args):
        raise RuntimeError("cuda")
    monkeypatch.setattr(cli, "_cmd_calibrate", boom)
    with pytest.raises(RuntimeError, match="cuda"):
        cli.main(["calibrate", "missing", "--name", "x"])
