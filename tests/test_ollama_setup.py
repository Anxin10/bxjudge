"""setup_ollama.py verifies inputs before touching its destination; serve_ollama.sh env.

Uses a local fake archive (plain tar behind the zstd seam); no network, no zstandard needed.
"""
import hashlib
import importlib.util
import io
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("setup_ollama", ROOT / "scripts/setup_ollama.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def _tar(tmp_path, entries, name="release.tar"):
    path = tmp_path / name
    with tarfile.open(path, "w") as tar:
        for entry, data in entries.items():
            info = tarfile.TarInfo(entry)
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
    return path


def _open_plain(archive: Path):
    return tarfile.open(archive, "r|")


@pytest.fixture
def pinned(monkeypatch):
    """Make a given local archive the pinned release (size + digest) and read it as plain tar."""
    def pin(archive: Path):
        data = archive.read_bytes()
        monkeypatch.setattr(setup, "SIZE", len(data))
        monkeypatch.setattr(setup, "SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(setup, "_open_release", _open_plain)
    return pin


def test_valid_archive_installs(tmp_path, pinned):
    archive = _tar(tmp_path, {"bin/ollama": b"#!/bin/sh\n"})
    pinned(archive)
    runtime = tmp_path / "runtime"
    setup.install(archive, runtime)
    assert (runtime / "bin/ollama").read_bytes() == b"#!/bin/sh\n"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".ollama-stage")] == []


def test_checksum_mismatch_extracts_nothing(tmp_path, pinned, monkeypatch):
    archive = _tar(tmp_path, {"bin/ollama": b"x"})
    pinned(archive)
    monkeypatch.setattr(setup, "SHA256", "0" * 64)
    opened = []
    monkeypatch.setattr(setup, "_open_release", lambda a: opened.append(a))
    runtime = tmp_path / "runtime"
    with pytest.raises(ValueError, match="SHA256"):
        setup.install(archive, runtime)
    assert not runtime.exists() and opened == []


def test_invalid_size_does_not_create_runtime(tmp_path):
    archive = tmp_path / "archive"
    archive.write_bytes(b"bad")
    runtime = tmp_path / "runtime"
    with pytest.raises(ValueError, match="bytes"):
        setup.install(archive, runtime)
    assert not runtime.exists()


@pytest.mark.parametrize("name", ["../escape", "/abs/escape", "bin/../../escape"])
def test_traversal_entry_is_refused(tmp_path, pinned, name):
    archive = _tar(tmp_path, {"bin/ollama": b"x", name: b"pwn"})
    pinned(archive)
    runtime = tmp_path / "sub" / "runtime"
    with pytest.raises(ValueError, match="unsafe path"):
        setup.install(archive, runtime)
    assert not runtime.exists()
    assert not (tmp_path / "escape").exists() and not (tmp_path / "sub/escape").exists()


def test_existing_runtime_is_refused_without_force(tmp_path, pinned):
    archive = _tar(tmp_path, {"bin/ollama": b"new"})
    pinned(archive)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    sentinel = runtime / "keep"
    sentinel.write_text("existing service")
    with pytest.raises(ValueError, match="already exists"):
        setup.install(archive, runtime)
    assert sentinel.read_text() == "existing service"


def test_force_replaces_existing_runtime(tmp_path, pinned):
    archive = _tar(tmp_path, {"bin/ollama": b"new"})
    pinned(archive)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "old").write_text("old")
    setup.install(archive, runtime, force=True)
    assert (runtime / "bin/ollama").read_bytes() == b"new" and not (runtime / "old").exists()


def test_failed_forced_install_keeps_existing_runtime(tmp_path, pinned):
    archive = _tar(tmp_path, {"../escape": b"pwn"})
    pinned(archive)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "old").write_text("old")
    with pytest.raises(ValueError):
        setup.install(archive, runtime, force=True)
    assert (runtime / "old").read_text() == "old"


def test_archive_without_binary_is_refused(tmp_path, pinned):
    archive = _tar(tmp_path, {"lib/other": b"x"})
    pinned(archive)
    runtime = tmp_path / "runtime"
    with pytest.raises(ValueError, match="bin/ollama"):
        setup.install(archive, runtime)
    assert not runtime.exists()


def test_cli_refuses_existing_runtime_without_force(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    done = subprocess.run([sys.executable, str(ROOT / "scripts/setup_ollama.py"),
                           "--runtime", str(runtime), "--archive", str(tmp_path / "none")],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 2 and "--force" in done.stderr


# --------------------------------------------------------------------------- serve script


def test_serve_script_is_lf_and_executable():
    path = ROOT / "scripts/serve_ollama.sh"
    assert b"\r" not in path.read_bytes()
    assert path.stat().st_mode & stat.S_IXUSR


def test_serve_script_sets_isolated_environment(tmp_path):
    runtime = tmp_path / "rt"
    (runtime / "bin").mkdir(parents=True)
    fake = runtime / "bin/ollama"
    fake.write_text('#!/usr/bin/env bash\necho "$OLLAMA_HOST|$OLLAMA_MODELS|$OLLAMA_NO_CLOUD|$1"\n')
    fake.chmod(0o755)
    models = tmp_path / "models"
    done = subprocess.run(["bash", str(ROOT / "scripts/serve_ollama.sh"), str(runtime), str(models)],
                          capture_output=True, text=True, check=False,
                          env={"PATH": os.environ["PATH"], "OLLAMA_HOST": "0.0.0.0:1"})
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"127.0.0.1:11435|{models}|1|serve"


def test_serve_script_without_runtime_fails_cleanly(tmp_path):
    done = subprocess.run(["bash", str(ROOT / "scripts/serve_ollama.sh"), str(tmp_path / "none")],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 2 and "setup_ollama.py" in done.stderr
