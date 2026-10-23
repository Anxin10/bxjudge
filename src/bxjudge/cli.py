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
                  detection_config=Path(args.config).expanduser(), limit=args.limit,
                  vlm_config=None if args.no_vlm else Path(args.vlm_config).expanduser(),
                  autocalib_config=Path(args.autocalib).expanduser())
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
    p.add_argument("--vlm-config", default="config/vlm.yaml",
                   help="local VLM second opinion after judging (its after_run key enables it)")
    p.add_argument("--no-vlm", action="store_true", help="skip the VLM phase")
    p.add_argument("--autocalib", default="config/autocalib.yaml",
                   help="auto-calibration policy for videos no manual calibration matches")
    p.set_defaults(func=_cmd_run)


def _cmd_vlm(args: argparse.Namespace) -> int:
    import json
    import logging
    import sys
    import time
    from datetime import datetime

    from bxjudge import vlm
    from bxjudge.config import load_thresholds
    from bxjudge.output import record_run_timing
    from bxjudge.rules import load_rules

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = Path(args.results).expanduser()
    began = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        if not results.is_dir():
            raise ValueError(f"{results}: expected a results directory")
        config = vlm.load_vlm_config(Path(args.config).expanduser())
        summary = vlm.review_results_dir(
            results,
            calib_dir=Path(args.calib).expanduser(),
            th=load_thresholds(Path(args.thresholds).expanduser()),
            rules=load_rules(Path(args.rules).expanduser()),
            config=config,
        )
        if summary.get("skipped"):  # R65: nothing asked, so no VLM run is recorded
            print(f"warning: VLM phase skipped, {config.host} unreachable; start the local "
                  "Ollama (README 完整流程) and rerun `bxjudge vlm`", file=sys.stderr)
        if summary.get("asked"):
            wall = time.perf_counter() - began  # R61: standalone VLM phase wall-clock
            record_run_timing(results, kind="vlm", started_at=started_at, wall_s=wall,
                              vlm_wall_s=wall, videos={})
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _add_vlm(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("vlm", help="ask the local VLM about unreviewed review-required rounds")
    p.add_argument("results", help="results directory written by `bxjudge run`")
    p.add_argument("--calib", default="calib", help="calibration directory used by the run")
    p.add_argument("--config", default="config/vlm.yaml")
    p.add_argument("--thresholds", default="config/thresholds.yaml")
    p.add_argument("--rules", default="config/rules.yaml")
    p.set_defaults(func=_cmd_vlm)


def _cmd_review(args: argparse.Namespace) -> int:
    import os
    import subprocess
    import sys

    results = Path(args.results).expanduser().resolve()
    if not (results / "rounds.json").is_file():
        print(f"error: {results} has no rounds.json (run `bxjudge run` first)", file=sys.stderr)
        return 2
    app = Path(__file__).with_name("review_app.py")
    argv = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.address", "127.0.0.1",  # loopback only: never reachable from the LAN
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--",
        str(results),
        "--calib", str(Path(args.calib).expanduser().resolve()),
        "--rules", str(Path(args.rules).expanduser().resolve()),
        "--thresholds", str(Path(args.thresholds).expanduser().resolve()),
    ]
    env = {**os.environ, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
    print(f"review UI: http://127.0.0.1:{args.port}  (Ctrl+C to stop)", file=sys.stderr)
    return subprocess.call(argv, env=env)


def _add_review(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("review", help="local web UI to review and correct judged rounds")
    p.add_argument("results", help="results directory written by `bxjudge run`")
    p.add_argument("--port", type=int, default=8501, help="loopback port (default 8501)")
    p.add_argument("--calib", default="calib",
                   help="calibration directory used by the run (for the zone overlay)")
    p.add_argument("--rules", default="config/rules.yaml")
    p.add_argument("--thresholds", default="config/thresholds.yaml")
    p.set_defaults(func=_cmd_review)


def _eval_inputs(args: argparse.Namespace):
    """Prediction rows/timings, optionally with R37 reference truth applied (read-only)."""
    from bxjudge import evaluate as ev

    cfg = ev.load_eval_config(Path(args.config).expanduser())
    results = Path(args.results).expanduser()
    rows, timings, runs = ev.read_results(results)
    truth = None
    if args.truth:
        truth_rows, _, _ = ev.read_results(Path(args.truth).expanduser())
        truth = ev.match_truth(rows, truth_rows, cfg.truth_t_launch_tolerance_s)
        rows = truth.rows
    return cfg, results, rows, timings, runs, truth


def _report_path(args: argparse.Namespace, kind: str) -> tuple[Path, str]:
    from datetime import date

    today = f"{date.today():%Y%m%d}"
    if args.report:
        return Path(args.report).expanduser(), today
    return Path("reports") / f"{kind}-{today}.md", today


def _write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _cmd_eval(args: argparse.Namespace) -> int:
    import math
    import sys

    from bxjudge import evaluate as ev
    from bxjudge.calibration_store import vetoed_ids
    from bxjudge.rules import load_rules

    try:
        cfg, results, rows, timings, runs, truth = _eval_inputs(args)
        rules = load_rules(Path(args.rules).expanduser())
        vetoed = vetoed_ids(results)
        metrics = ev.evaluate(rows, timings, config=cfg, runs=runs, vetoed=vetoed)
        ok = ev.passes(metrics, cfg)
        extra = ev.report_metrics(rows, timings, cfg, rules=rules, runs=runs)
        path, today = _report_path(args, "eval")
        _write_report(path, ev.render_eval_report(
            results, rows, timings, metrics, ok, extra, cfg, truth=truth,
            truth_dir=Path(args.truth).expanduser() if args.truth else None, today=today,
            vetoed=vetoed))
    except (ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    def shown(name: str) -> str:
        value = getattr(metrics, name)
        return f"{value:.3f}" if math.isfinite(value) else "unverified"

    summary = ", ".join(f"{name}={shown(name)} {ok[name]}"
                        for name in ev.METRIC_NAMES)
    print(f"{summary}; reviewed={metrics.n_reviewed}/{metrics.n_rounds}")
    if ev.is_synthetic(rows):
        print("SMOKE ONLY: synthetic-detector rounds are not acceptance evidence")
    print(f"report: {path}")
    return 0


def _add_eval(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("eval", help="spec section 11 metrics from reviewed rounds (read-only)")
    p.add_argument("results", help="results directory written by `bxjudge run`")
    p.add_argument("--truth", help="explicit reviewed reference results directory used as "
                                   "truth for these (fresh) predictions; read-only (R37)")
    p.add_argument("--config", default="config/evaluation.yaml")
    p.add_argument("--rules", default="config/rules.yaml", help="for match score agreement")
    p.add_argument("--report", help="report path (default reports/eval-YYYYMMDD.md)")
    p.set_defaults(func=_cmd_eval)


def _cmd_tune(args: argparse.Namespace) -> int:
    import sys

    from bxjudge import evaluate as ev
    from bxjudge.config import load_thresholds

    try:
        cfg, results, rows, _, _, truth = _eval_inputs(args)
        th_path = Path(args.thresholds).expanduser()
        th = load_thresholds(th_path)
        samples = ev.shake_samples(rows, th.impact_coincide_frames)
        derivation = ev.derive_shake(samples.samples, cfg.shake_overlap_percentile)
        diff = ("" if derivation.method == "insufficient"
                else ev.thresholds_diff(th_path, "shake_px", derivation.value))
        path, today = _report_path(args, "tune")
        _write_report(path, ev.render_tune_report(
            results, rows, samples, derivation, diff, th.shake_px,
            ev.spin_dir_check(rows, cfg.spin_min_samples), ev.reversed_finish_check(rows),
            impact_coincide_frames=th.impact_coincide_frames, thresholds_path=th_path,
            truth=truth, today=today))
    except (ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(diff.rstrip() if diff else f"shake_px: no change proposed ({derivation.method})")
    print(f"advisory only, {th_path} not modified; report: {path}")
    return 0


def _add_tune(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("tune", help="advisory shake_px / spin-direction checks from reviewed rounds")
    p.add_argument("results", help="results directory written by `bxjudge run`")
    p.add_argument("--truth", help="explicit reviewed reference results directory (R37)")
    p.add_argument("--thresholds", default="config/thresholds.yaml",
                   help="current thresholds (read only; the diff is printed, never applied)")
    p.add_argument("--config", default="config/evaluation.yaml")
    p.add_argument("--report", help="report path (default reports/tune-YYYYMMDD.md)")
    p.set_defaults(func=_cmd_tune)


def _cmd_audit_sample(args: argparse.Namespace) -> int:
    import sys

    from bxjudge.corrections import CorrectionError, mark_audit_sample
    from bxjudge.rules import load_rules

    try:
        picked = mark_audit_sample(Path(args.results).expanduser(), args.n, seed=args.seed,
                                   rules=load_rules(Path(args.rules).expanduser()))
    except (CorrectionError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for video, number in picked:
        print(f"{video}\t{number}")
    print(f"{len(picked)} unflagged round(s) in the audit sample (seed {args.seed}); review "
          "them all with the 抽樣稽核 filter in `bxjudge review`")
    return 0


def _add_audit_sample(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("audit-sample",
                       help="mark a seeded random sample of unflagged rounds for audit review")
    p.add_argument("results", help="results directory written by `bxjudge run`")
    p.add_argument("--n", type=int, required=True, help="sample size")
    p.add_argument("--seed", type=int, required=True, help="random seed (reproducible)")
    p.add_argument("--rules", default="config/rules.yaml")
    p.set_defaults(func=_cmd_audit_sample)


_REGISTRARS = [_add_calibrate, _add_scan, add_label, _add_train, _add_compare, _add_run,
               _add_vlm, _add_review, _add_eval, _add_tune, _add_audit_sample]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bxjudge", description="Local AI referee for Beyblade X videos")
    parser.add_argument("--debug", action="store_true",
                        help="show the full traceback instead of a one-line error")
    sub = parser.add_subparsers(dest="command", required=True)
    for register in _REGISTRARS:
        register(sub)
    return parser


def _user_errors() -> tuple[type[BaseException], ...]:
    """Exceptions that mean the user's input is wrong, not that the program is.

    - ``OSError``: missing / unreadable / unwritable paths (FileNotFoundError, ...).
    - ``yaml.YAMLError``: malformed YAML outside a config loader.
    - ``ValueError``: invalid values; every config loader reports wrong values AND wrong
      shapes (the TypeError/KeyError its validation hits) as ``config.ConfigError``,
      a ValueError naming the file. JSON decode errors and CorrectionError are ValueErrors.

    Deliberately NOT caught: TypeError, KeyError, AttributeError, RuntimeError (CUDA, ...)
    raised outside config loading: those are bugs or environment faults whose traceback
    is the evidence needed to fix them.
    """
    import yaml

    return (OSError, yaml.YAMLError, ValueError)


def main(argv: list[str] | None = None) -> int:
    import sys

    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except _user_errors() as error:
        if args.debug:
            raise
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
