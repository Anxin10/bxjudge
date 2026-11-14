"""Entry point for the isolated LocateAnything runtime."""

import json
import sys
from pathlib import Path

from .locateanything import LocateAnything
from .prelabel import prelabel_video


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cfg = request["config"]
    predictor = LocateAnything(cfg)
    counts = [
        prelabel_video(
            Path(job["video"]), job["frames"], Path(job["out_json"]), cfg, predictor,
            expected_sha256=job["video_sha256"],
        )
        for job in request["jobs"]
    ]
    Path(sys.argv[2]).write_text(json.dumps({"counts": counts}), encoding="utf-8")


if __name__ == "__main__":
    main()
