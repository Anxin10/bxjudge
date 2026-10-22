"""Installer verifies inputs before changing its destination."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("setup_ollama", Path("scripts/setup_ollama.py"))
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_existing_runtime_is_never_overwritten(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    sentinel = runtime / "keep"
    sentinel.write_text("existing service")
    with pytest.raises(ValueError, match="already exists"):
        setup.install(tmp_path / "missing-archive", runtime)
    assert sentinel.read_text() == "existing service"


def test_invalid_size_does_not_create_runtime(tmp_path):
    archive = tmp_path / "archive"
    archive.write_bytes(b"bad")
    runtime = tmp_path / "runtime"
    with pytest.raises(ValueError, match="bytes"):
        setup.install(archive, runtime)
    assert not runtime.exists()


def test_invalid_checksum_does_not_create_runtime(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    archive.write_bytes(b"bad")
    monkeypatch.setattr(setup, "SIZE", 3)
    runtime = tmp_path / "runtime"
    with pytest.raises(ValueError, match="SHA256"):
        setup.install(archive, runtime)
    assert not runtime.exists()
