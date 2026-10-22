"""bxjudge command line.

Adding a subcommand = one ``_add_<name>(sub)`` function that registers the parser and
sets ``func`` (taking the parsed ``argparse.Namespace``, returning an int exit code),
plus one entry in ``_REGISTRARS``.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import cv2

    from bxjudge import calibration  # looked up as attributes so tests can patch them

    out = Path(args.out)
    frame = calibration.grab_frame(Path(args.video), args.at)
    c = calibration.interactive_calibrate(frame, args.name)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{args.name}.png"
    if not cv2.imwrite(str(png), frame):
        raise RuntimeError(f"cannot write {png}")
    c.ref_frame = f"{args.name}.png"  # relative to the JSON's folder
    json_path = out / f"{args.name}.json"
    calibration.save_calibration(c, json_path)
    print(f"saved {json_path} and {png}")
    return 0


def _add_calibrate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("calibrate", help="click the stadium zones on one video frame")
    p.add_argument("video", help="video file to take the reference frame from")
    p.add_argument("--at", type=float, default=1.0, help="seconds into the video (default 1.0)")
    p.add_argument("--name", required=True, help="calibration name, e.g. 20261001")
    p.add_argument("--out", default="calib", help="output directory (default calib/)")
    p.set_defaults(func=_cmd_calibrate)


def _cmd_scan(args: argparse.Namespace) -> int:
    import sys

    from bxjudge import manifest

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"error: {folder} is not a folder", file=sys.stderr)
        return 2
    failed: list[Path] = []

    def on_error(path: Path, err: Exception) -> None:
        failed.append(path)
        print(f"warning: skipped {path}: {err}", file=sys.stderr)

    found = manifest.scan(folder, Path(args.db).expanduser(), on_error=on_error)
    for i in found:
        print(f"{i.path.name}  {i.duration:.1f}s  {i.fps:.2f}fps  {i.width}x{i.height}  "
              f"{i.codec}  audio={'yes' if i.has_audio else 'no'}")
    print(f"{len(found)} new/changed video(s), {len(failed)} unreadable", file=sys.stderr)
    return 1 if failed else 0


def _add_scan(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("scan", help="list new/changed videos in a folder (does not mark them processed)")
    p.add_argument("folder", help="folder of synced videos, e.g. ~/bx/videos/20261001")
    p.add_argument("--db", default="data/processed_videos.json",
                   help="processed-videos record (default data/processed_videos.json)")
    p.set_defaults(func=_cmd_scan)


from bxjudge.labeling.cli import add_label


def _cmd_train(args: argparse.Namespace) -> int:
    from bxjudge.train import train

    print(train(args.kind, Path(args.dataset), Path(args.out), args.epochs,
                config=args.config, training_policy=args.training_policy,
                thresholds=args.thresholds))
    return 0


def _add_train(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="train one detector on the reviewed COCO dataset")
    p.add_argument("--kind", required=True, choices=("rfdetr", "yolo26"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True, help="new isolated run directory")
    p.add_argument("--epochs", type=int, help="defaults to config training.epochs")
    p.add_argument("--config", default="config/detection.yaml",
                   help="detector config; its training block is the production policy")
    p.add_argument("--training-policy",
                   help="explicit training policy file instead of the production policy, "
                        "e.g. config/detection_smoke.yaml (synthetic datasets only)")
    p.add_argument("--thresholds", default="config/thresholds.yaml",
                   help="infer_long_edge source for the production training size")
    p.set_defaults(func=_cmd_train)


def _cmd_compare(args: argparse.Namespace) -> int:
    import json

    from bxjudge.train import compare

    print(json.dumps(compare(
        {"rfdetr": Path(args.rfdetr), "yolo26": Path(args.yolo26)},
        Path(args.dataset), output=Path(args.output),
        selection_out=Path(args.selection_out), config=args.config,
        thresholds=args.thresholds), indent=2))
    return 0


def _add_compare(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("compare", help="compare both detectors using common COCO validation")
    p.add_argument("--dataset", required=True)
    p.add_argument("--rfdetr", required=True, help="trained RF-DETR Nano checkpoint")
    p.add_argument("--yolo26", required=True, help="trained YOLO26 Nano checkpoint")
    p.add_argument("--output", default="runs/compare.json")
    p.add_argument("--selection-out", default="weights/selected.json",
                   help="selection manifest (synthetic datasets require an isolated path)")
    p.add_argument("--config", default="config/detection.yaml")
    p.add_argument("--thresholds", default="config/thresholds.yaml",
                   help="canonical inference long-edge policy")
    p.set_defaults(func=_cmd_compare)


def _cmd_run(args: argparse.Namespace) -> int:
    import logging
    import sys

    from bxjudge.pipeline import run_batch

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    folder = Path(args.folder).expanduser()
    try:
        run_batch(folder, Path(args.calib).expanduser(),
                  Path(args.out).expanduser() if args.out else Path("results") / folder.name,
                  selection=Path(args.selection).expanduser(),
                  thresholds=Path(args.thresholds).expanduser(),
                  rules_path=Path(args.rules).expanduser(),
                  detection_config=Path(args.config).expanduser(), limit=args.limit)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _add_run(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="judge new/changed videos into a result-local batch")
    p.add_argument("folder")
    p.add_argument("--calib", default="calib")
    p.add_argument("--out", help="defaults to results/<folder name>")
    p.add_argument("--selection", default="weights/selected.json",
                   help="explicit selected detector manifest; synthetic output requires review")
    p.add_argument("--thresholds", default="config/thresholds.yaml")
    p.add_argument("--rules", default="config/rules.yaml")
    p.add_argument("--config", default="config/detection.yaml")
    p.add_argument("--limit", type=int, help="maximum new/changed videos")
    p.set_defaults(func=_cmd_run)


_REGISTRARS = [_add_calibrate, _add_scan, add_label, _add_train, _add_compare, _add_run]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bxjudge", description="Local AI referee for Beyblade X videos")
    sub = parser.add_subparsers(dest="command", required=True)
    for register in _REGISTRARS:
        register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
