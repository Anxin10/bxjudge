#!/usr/bin/env python3
"""Install the verified official Ollama 0.40.7 release into the project-owned runtime directory.

Downloads the pinned archive to a temp path, verifies size and SHA256 BEFORE extracting, then
extracts with path-traversal protection into ``weights/ollama-runtime``. It never pulls models
and never touches a system Ollama. Needs ``zstandard`` (not a project dependency), so run it
through an isolated tool environment, e.g.::

    uv run --no-project --with zstandard python scripts/setup_ollama.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

VERSION = "0.40.7"
URL = f"https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-linux-amd64.tar.zst"
SHA256 = "7f082e625b4c334df84ed1558fe282d6d8091fe3545790aef30b3a9c50d44d41"
SIZE = 1505607169
ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_MARGIN = 1 << 20  # bytes tolerated beyond SIZE before the download is aborted
CHUNK = 1 << 20


def _download(source, target: Path) -> None:
    """Copy ``source`` to ``target``, never writing past ``SIZE + DOWNLOAD_MARGIN`` bytes.

    Past the cap the partial file is removed and ``ValueError`` raised before more is written.
    """
    cap = SIZE + DOWNLOAD_MARGIN
    total = 0
    try:
        with target.open("wb") as out:
            while chunk := source.read(CHUNK):
                total += len(chunk)
                if total > cap:
                    raise ValueError(f"download is larger than expected ({SIZE} bytes); aborted")
                out.write(chunk)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def check_runtime_path(runtime: Path, *, force: bool, root: Path | None = None) -> None:
    """Refuse a runtime that is, or sits below, a symlink inside the project; with ``force``
    also refuse any runtime outside ``<project>/weights`` (it would be deleted).

    ``runtime`` must be absolute and NOT symlink-resolved (the caller uses ``abspath``).
    """
    root = ROOT if root is None else root
    weights = root / "weights"
    for part in (runtime, *runtime.parents):
        if part.is_symlink():
            raise ValueError(f"runtime path goes through a symlink: {part}; refusing")
        if part == root:
            break
    if force and not runtime.is_relative_to(weights):
        raise ValueError(f"--force only replaces a runtime inside {weights}; got {runtime}")
    if force and runtime == weights:
        raise ValueError(f"--force will not replace the weights folder itself: {runtime}")


def _open_release(archive: Path):
    """Streaming tar reader over the zstd archive (the only zstandard use; tests replace it)."""
    import zstandard

    source = archive.open("rb")
    decoded = zstandard.ZstdDecompressor().stream_reader(source)
    return tarfile.open(fileobj=decoded, mode="r|")


def _check_member(member: tarfile.TarInfo) -> None:
    """Refuse absolute, parent-escaping and device/fifo entries before anything is written."""
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe path in archive: {member.name}")
    if member.isdev():
        raise ValueError(f"unsafe special file in archive: {member.name}")


def install(archive: Path, runtime: Path, *, force: bool = False) -> None:
    """Verify, extract to a staging directory, then swap into place.

    An existing runtime is refused unless ``force``; with ``force`` it is removed only after the
    new one extracted successfully, so a failed install never destroys a working runtime.
    """
    if (runtime.exists() or runtime.is_symlink()) and not force:
        raise ValueError(f"runtime already exists: {runtime}; use --force to replace it")
    if archive.stat().st_size != SIZE:
        raise ValueError(f"archive must be {SIZE} bytes")
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError("archive SHA256 mismatch")

    runtime.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ollama-stage-", dir=runtime.parent) as stage_name:
        stage = Path(stage_name) / "runtime"
        stage.mkdir()
        with _open_release(archive) as release:
            for member in release:
                _check_member(member)
                release.extract(member, stage, filter="data")
        if not (stage / "bin/ollama").is_file():
            raise ValueError("official archive has no bin/ollama")
        if runtime.is_symlink() or runtime.is_file():
            runtime.unlink()
        elif runtime.exists():
            shutil.rmtree(runtime)
        stage.rename(runtime)
    print(f"Ollama {VERSION} installed: {runtime}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, help="reuse a downloaded official archive")
    parser.add_argument("--runtime", type=Path, default=ROOT / "weights/ollama-runtime")
    parser.add_argument("--force", action="store_true", help="replace an existing runtime")
    args = parser.parse_args()
    try:
        runtime = Path(os.path.abspath(args.runtime.expanduser()))  # not resolved: see below
        check_runtime_path(runtime, force=args.force)
        if (runtime.exists() or runtime.is_symlink()) and not args.force:
            raise ValueError(f"runtime already exists: {runtime}; use --force to replace it")
        if args.archive:
            install(args.archive.expanduser(), runtime, force=args.force)
        else:
            runtime.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="ollama-download-", dir=runtime.parent) as scratch:
                archive = Path(scratch) / "ollama-linux-amd64.tar.zst"
                print(f"Downloading official Ollama {VERSION} ({SIZE} bytes)", flush=True)
                with urllib.request.urlopen(URL, timeout=60) as source:
                    _download(source, archive)
                install(archive, runtime, force=args.force)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
