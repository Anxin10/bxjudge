"""Evaluate automatic stadium finding over whole video batches (read only on videos and calib/).

For every video: run ``find_stadium``, save the found calibration and a preview, and record
the fit, the chosen frame and pocket visibility. With ``--reference`` the current template
path (``select_snapshot`` on the first-second frame, as ``bxjudge autocalib``) is also run and
the 5 judged zones are compared. With ``--truth`` manual truth from ``stadium_truth.py`` is
compared and graded PASS/FAIL at ``autocalib.max_boundary_error_px``.

Usage:
    uv run python scripts/eval_stadium_finder.py --out <dir> [--reference] [--truth <dir>]
        [--workers 4] [--only-truth] <batch folder> [<batch folder> ...]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from bxjudge import PROJECT_ROOT
from bxjudge.calibration import load_calibration, save_calibration
from bxjudge.stadium_finder import (
    TRUTH_ZONES,
    find_stadium,
    load_stadium_finder_config,
    load_stadium_model,
    zone_boundary_errors,
)

_ZONE_COLORS = {"stadium": (255, 255, 255), "battle_area": (0, 0, 255), "over_left": (255, 0, 0),
                "over_right": (0, 165, 255), "xtreme": (0, 255, 255)}
_TRUTH_NAMES = [z[0] for z in TRUTH_ZONES]


def _preview(frame: np.ndarray, cal, path: Path, label: str) -> None:
    img = frame.copy()
    for name, color in _ZONE_COLORS.items():
        poly = np.round(np.array(getattr(cal, name))).astype(np.int32)
        cv2.polylines(img, [poly], True, color, 3)
    k = 640 / img.shape[1]
    img = cv2.resize(img, (640, round(img.shape[0] * k)))
    cv2.putText(img, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 80])


def _one(job: tuple[str, str, str, bool]) -> dict:
    video, batch, out, reference = job
    cv2.setNumThreads(1)
    video_p, out_p = Path(video), Path(out)
    cfg = load_stadium_finder_config()
    model = load_stadium_model()
    row: dict = {"batch": batch, "video": video_p.name}
    t = time.time()
    try:
        found = find_stadium(video_p, model, cfg)
    except Exception as e:  # recorded per video; a crash is a finding, not a skipped video
        row.update(found=False, error=f"{type(e).__name__}: {e}")
        found = None
    row["seconds"] = round(time.time() - t, 1)
    if found is not None:
        cal_path = out_p / "found" / f"{video_p.stem}.json"
        save_calibration(found.calibration, cal_path)
        f = found.fit
        row.update(found=True, frame_index=found.frame_index, coverage=round(f.coverage, 3),
                   precision=round(f.precision, 3), mean_error_px=round(f.mean_error_px, 2),
                   rail_in_frame=round(f.rail_in_frame, 3), agreeing_frames=found.agreeing_frames,
                   perspective=float(max(abs(f.homography[2][0]), abs(f.homography[2][1]))),
                   zone_visibility={k: round(v, 3) for k, v in found.zone_visibility.items()},
                   pocket_out_of_frame=[k for k in ("over_left", "over_right", "xtreme")
                                        if found.zone_visibility[k] < cfg.zone_visible_min])
        _preview(found.frame, found.calibration, out_p / "preview" / f"{video_p.stem}.jpg",
                 f"{batch}/{video_p.stem} f{found.frame_index} cov {f.coverage:.2f} prec {f.precision:.2f}")
    else:
        row.setdefault("found", False)
    if reference:
        row["reference"] = _reference(video_p, out_p, found)
    return row


def _reference(video: Path, out: Path, found) -> dict:
    """The current template path, compared on the 5 judged zones. It runs on the very frame
    the finder chose (the camera may move between the first second and that frame); without a
    finder result it runs on the first-second frame, as ``bxjudge autocalib`` does."""
    from bxjudge import manifest
    from bxjudge.autocalib import load_autocalib_config
    from bxjudge.calibration_store import select_snapshot
    from bxjudge.config import load_thresholds
    from bxjudge.pipeline import first_second_frame

    th = load_thresholds()
    acfg = load_autocalib_config()
    try:
        frame = found.frame if found is not None else first_second_frame(manifest.probe(video), th)
        snap = select_snapshot(frame, PROJECT_ROOT / "calib", out / "reference", th, acfg)
    except Exception as e:
        return {"kind": "error", "error": f"{type(e).__name__}: {e}"}
    if snap is None:
        return {"kind": "recalibrate"}
    ref = {"kind": snap.kind, "score": round(snap.score, 3)}
    if found is not None:
        truth = {z: getattr(snap.calibration, z) for z in _TRUTH_NAMES}
        errs = zone_boundary_errors(found.calibration, truth, snap.calibration.frame_size,
                                    th.infer_long_edge)
        ref["boundary_error_px"] = {k: round(v, 2) for k, v in errs.items()}
        ref["mean_boundary_error_px"] = round(float(np.mean(list(errs.values()))), 2)
    return ref


def _truth_rows(truth_dir: Path, out: Path, limit_px: float, long_edge: int) -> list[dict]:
    rows = []
    for path in sorted(truth_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        found_path = out / "found" / f"{Path(doc['video']).stem}.json"
        row = {"video": doc["video"], "at_s": doc["at_s"]}
        if not found_path.exists():
            row.update(status="REJECTED")
        else:
            found = load_calibration(found_path)
            truth = {k: [tuple(p) for p in v] for k, v in doc["zones"].items()}
            errs = zone_boundary_errors(found, truth, tuple(doc["frame_size"]), long_edge)
            worst = max(errs.values())
            row.update(boundary_error_px={k: round(v, 2) for k, v in errs.items()},
                       worst_px=round(worst, 2), status="PASS" if worst < limit_px else "FAIL")
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folders", nargs="*", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", action="store_true")
    p.add_argument("--truth", type=Path, default=None)
    p.add_argument("--only-truth", action="store_true", help="grade existing results only")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    out = a.out.expanduser()
    calib = (PROJECT_ROOT / "calib").resolve()
    if out.resolve() == calib or calib in out.resolve().parents:
        raise SystemExit(f"{out}: output must stay outside calib/")
    rows_path = out / "rows.jsonl"
    if not a.only_truth:
        for sub in ("found", "preview", "reference"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        done = set()
        if rows_path.exists():  # resumable: videos already evaluated are skipped
            done = {(r["batch"], r["video"]) for r in map(json.loads, rows_path.read_text().splitlines())}
        jobs = [(str(v), f.name, str(out), a.reference)
                for f in map(lambda x: x.expanduser(), a.folders) for v in sorted(f.glob("*.mp4"))
                if (f.name, v.name) not in done]
        with ProcessPoolExecutor(max_workers=a.workers) as pool, open(rows_path, "a") as fh:
            for row in pool.map(_one, jobs):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"{row['batch']}/{row['video']}: {'found' if row['found'] else 'none'} "
                      f"{row['seconds']}s", flush=True)
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
    summary: dict = {}
    for batch in sorted({r["batch"] for r in rows}):
        rs = [r for r in rows if r["batch"] == batch]
        ok = [r for r in rs if r["found"]]
        refs = [r["reference"] for r in rs if "reference" in r]
        summary[batch] = {
            "videos": len(rs), "found": len(ok),
            "errors": sum(1 for r in rs if "error" in r),
            "pocket_out_of_frame": sum(1 for r in ok if r["pocket_out_of_frame"]),
            "median_seconds": float(np.median([r["seconds"] for r in rs])) if rs else None,
        }
        if refs:
            both = [r["reference"]["mean_boundary_error_px"] for r in rs
                    if "mean_boundary_error_px" in r.get("reference", {})]
            summary[batch]["reference_kinds"] = {k: sum(1 for x in refs if x["kind"] == k)
                                                 for k in sorted({x["kind"] for x in refs})}
            summary[batch]["vs_reference_px"] = (
                {"n": len(both), "median": float(np.median(both)), "max": float(np.max(both)),
                 "over_8px": sum(1 for x in both if x >= 8.0)} if both else None)
    if a.truth:
        from bxjudge.autocalib import load_autocalib_config
        from bxjudge.config import load_thresholds

        truth_rows = _truth_rows(a.truth.expanduser(), out, load_autocalib_config().max_boundary_error_px,
                                 load_thresholds().infer_long_edge)
        summary["truth"] = {"rows": truth_rows,
                            "counts": {s: sum(1 for r in truth_rows if r["status"] == s)
                                       for s in ("PASS", "FAIL", "REJECTED")}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "truth"}, ensure_ascii=False, indent=1))
    if "truth" in summary:
        print("truth:", summary["truth"]["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
