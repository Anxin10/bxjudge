"""Setup/sync shell scripts: user tool PATH, safe batch names, LF + executable."""

import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "scripts" / "sync_videos.sh"
SETUP = ROOT / "scripts" / "setup_wsl.sh"
FOLDER_ID = "1ExampleDriveFolderId0123456789ab"


def _fake_home(tmp_path):
    """HOME whose ~/.local/bin holds a fake rclone that only records its arguments."""
    home = tmp_path / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    fake = bin_dir / "rclone"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$HOME/rclone-args"\n')
    fake.chmod(0o755)
    return home


def _sync(home, *args):
    # A normal login PATH without ~/.local/bin, where the standalone rclone lives (R28).
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    return subprocess.run(["bash", str(SYNC), *args], env=env, capture_output=True, text=True,
                          check=False)


def test_sync_finds_user_installed_rclone_and_stays_under_videos(tmp_path):
    home = _fake_home(tmp_path)
    done = _sync(home, FOLDER_ID, "20261001")
    assert done.returncode == 0, done.stderr
    args = (home / "rclone-args").read_text().splitlines()
    assert args[:3] == ["copy", "gdrive:", f"{home}/bx/videos/20261001"]
    assert args[3:5] == ["--drive-root-folder-id", FOLDER_ID]
    # copy only: never a destructive or writing remote operation
    assert not {"sync", "move", "delete", "purge", "--delete-after"} & set(args)


@pytest.mark.parametrize("batch", ["../evil", "..", ".", "/tmp/x", "a/b", "", "-rf", ".hidden",
                                   "a b", "20261001/../../x"])
def test_sync_rejects_unsafe_batch_names(tmp_path, batch):
    home = _fake_home(tmp_path)
    done = _sync(home, FOLDER_ID, batch)
    assert done.returncode == 2
    assert "batch" in done.stderr
    assert not (home / "rclone-args").exists()


@pytest.mark.parametrize("folder_id", ["", "--delete", "a b", "../x"])
def test_sync_rejects_malformed_folder_ids(tmp_path, folder_id):
    home = _fake_home(tmp_path)
    assert _sync(home, folder_id, "20261001").returncode == 2
    assert not (home / "rclone-args").exists()


def test_setup_puts_user_tools_on_path_unconditionally():
    text = SETUP.read_text()
    export = 'export PATH="$HOME/.local/bin:$PATH"'
    assert export in text
    before = text.split(export, 1)[0]
    # not nested inside the "install uv only if missing" branch
    assert before.count("\nif ") == before.count("\nfi")
    assert text.index(export) < text.index("command -v uv")


@pytest.mark.parametrize("script", [SYNC, SETUP])
def test_scripts_are_lf_and_executable(script):
    assert b"\r" not in script.read_bytes()
    mode = stat.S_IMODE(os.stat(script).st_mode)
    assert mode == 0o755
