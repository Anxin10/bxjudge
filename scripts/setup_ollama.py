#!/usr/bin/env python3
"""Install the verified official Ollama release into a new project-owned directory.

Run via the pinned isolated tool environment documented in README (no project sync).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = "0.40.7"
URL = f"https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-linux-amd64.tar.zst"
SHA256 = "7f082e625b4c334df84ed1558fe282d6d8091fe3545790aef30b3a9c50d44d41"
SIZE = 1505607169
ROOT = Path(__file__).resolve().parents[1]


def install(archive: Path, runtime: Path) -> None:
    """Verify before extracting; never merge with or overwrite an existing runtime."""
    if runtime.exists() or runtime.is_symlink():
        raise ValueError(f"runtime already exists: {runtime}; choose a new directory")
    if archive.stat().st_size != SIZE:
        raise ValueError(f"archive must be {SIZE} bytes")
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError("archive SHA256 mismatch")
    import zstandard

    runtime.mkdir(parents=True, exist_ok=False)
    installed = False
    try:
        with (
            archive.open("rb") as source,
            zstandard.ZstdDecompressor().stream_reader(source) as decoded,
            tarfile.open(fileobj=decoded, mode="r|") as release,
        ):
            release.extractall(runtime, filter="data")
        if not (runtime / "bin/ollama").is_file():
            raise ValueError("official archive has no bin/ollama")
        installed = True
    finally:
        if not installed:
            shutil.rmtree(runtime)
    print(f"Ollama {VERSION} installed: {runtime}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="reuse a downloaded official archive")
    parser.add_argument("--runtime", type=Path, default=ROOT / "weights/ollama-runtime")
    args = parser.parse_args()
    try:
        runtime = args.runtime.expanduser().resolve()
        if runtime.exists():
            raise ValueError(f"runtime already exists: {runtime}; choose a new directory")
        if args.archive:
            install(args.archive.expanduser(), runtime)
        else:
            runtime.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="ollama-download-", dir=runtime.parent) as scratch:
                archive = Path(scratch) / "ollama-linux-amd64.tar.zst"
                print(f"Downloading official Ollama {VERSION} ({SIZE} bytes)", flush=True)
                with urllib.request.urlopen(URL, timeout=60) as source, archive.open("wb") as target:
                    shutil.copyfileobj(source, target)
                install(archive, runtime)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
