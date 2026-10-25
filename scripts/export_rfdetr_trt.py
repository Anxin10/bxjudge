#!/usr/bin/env python3
"""Export a local RF-DETR checkpoint using the verified project TensorRT runtime."""

import argparse
from pathlib import Path

from bxjudge.tensorrt import export_rfdetr_trt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument(
        "--out", required=True, type=Path, help="new directory; never overwritten"
    )
    parser.add_argument("--policy", default="config/tensorrt.yaml", type=Path)
    parser.add_argument("--thresholds", default="config/thresholds.yaml", type=Path)
    parser.add_argument("--config", default="config/detection.yaml", type=Path)
    args = parser.parse_args()
    try:
        path = export_rfdetr_trt(
            args.weights,
            args.out,
            policy=args.policy,
            thresholds=args.thresholds,
            detection_config=args.config,
        )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")
    print(path)


if __name__ == "__main__":
    main()
