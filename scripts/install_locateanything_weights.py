"""Install the verified model snapshot used by the local labeling worker."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from bxjudge.labeling.config import load_config
from bxjudge.labeling.locateanything import (
    MODEL_ID,
    MODEL_MANIFEST,
    MODEL_REVISION,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-snapshot", type=Path, help="reuse a local copy of the pinned snapshot")
    args = parser.parse_args()
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    directory = Path(load_config()["model_dir"]).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        verify_model(directory)
    else:
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".locateanything-", dir=directory.parent) as temp:
            staging = Path(temp) / directory.name
            staging.mkdir()
            if args.from_snapshot is None:
                snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=staging,
                                  allow_patterns=list(manifest["files"]))
            else:
                source = args.from_snapshot.expanduser().resolve()
                for name in manifest["files"]:
                    shutil.copyfile(source / name, staging / name)
            verify_model(staging)
            staging.replace(directory)
    print(f"Verified LocateAnything {MODEL_REVISION}: {directory}")


if __name__ == "__main__":
    main()
