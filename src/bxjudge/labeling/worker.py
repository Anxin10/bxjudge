"""Private process entry point for the supported, pinned SAM 3.1 environment."""

import json
import sys
from pathlib import Path

from .sam3_prelabel import _build_predictor, _prelabel_direct


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cfg = request["config"]
    predictor = _build_predictor(cfg)
    try:
        counts = [
            _prelabel_direct(
                Path(job["video"]),
                job["frames"],
                cfg["prompts"],
                Path(job["out_json"]),
                config=cfg,
                predictor=predictor,
            )
            for job in request["jobs"]
        ]
        Path(sys.argv[2]).write_text(json.dumps({"counts": counts}), encoding="utf-8")
    finally:
        predictor.shutdown()
    # Constructor autocast and model memory are scoped to this terminating process.


if __name__ == "__main__":
    main()
