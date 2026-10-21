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


_REGISTRARS = [_add_calibrate]


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
