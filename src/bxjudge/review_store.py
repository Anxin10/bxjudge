"""Content-addressed review snapshots. Revisions never overwrite earlier labels."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def save_immutable(folder: Path, payload: dict, *, prefix: str):
    if prefix not in {"labels", "reid"}:
        raise ValueError("invalid review type")
    data = (
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode()
    digest = hashlib.sha256(data).hexdigest()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{prefix}-{digest}.json"
    with tempfile.NamedTemporaryFile(dir=folder, prefix=".pending-", delete=False) as f:
        temp = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(temp, target)
        except FileExistsError:
            if target.read_bytes() != data:
                raise ValueError("review content identity collision")
    finally:
        temp.unlink()
    return target, digest
