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

def protected_dirs(folders: list[Path]) -> list[Path]:
    """calib/ and every input video folder, resolved: never written to."""
    return [(PROJECT_ROOT / "calib").resolve(), *(f.expanduser().resolve() for f in folders)]


def ensure_safe(path: Path, protected: list[Path]) -> None:
    """Refuse a write target that resolves (symlinks and junctions included, for the file and
    every existing parent) into a protected folder."""
    real = path.expanduser().resolve()
    for p in protected:
        if real == p or p in real.parents:
            raise SystemExit(f"{path}: refusing to write inside {p}")


def check_output_dir(out: Path, folders: list[Path]) -> None:
    """Refuse an output folder inside calib/ or inside any input video folder: sources stay
    read only. Each later write is checked again with ``ensure_safe``."""
    ensure_safe(out, protected_dirs(folders))


def batch_names(folders: list[Path]) -> list[str]:
    """Batch id per input folder (its name); two folders with one name would share result
    files and resume records, so they are refused."""
    names = [f.expanduser().name for f in folders]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"input folders share the batch name(s) {dup}; evaluate them separately")
    return names


def result_path(out: Path, kind: str, batch: str, video: str, suffix: str) -> Path:
    """Per-batch result file, so equal video names in two batches never share a file."""
    return out / kind / batch / f"{Path(video).stem}{suffix}"


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


def _one(job: tuple[str, str, str, bool, list[str]]) -> dict:
    video, batch, out, reference, protected_s = job
    cv2.setNumThreads(1)
    video_p, out_p = Path(video), Path(out)
    protected = [Path(p) for p in protected_s]
    cfg = load_stadium_finder_config()
    model = load_stadium_model()
    row: dict = {"batch": batch, "video": video_p.name}
    try:
        from bxjudge.manifest import probe

        row["fps"] = probe(video_p).fps
    except Exception as e:
        row["fps_error"] = f"{type(e).__name__}: {e}"
    t = time.time()
    try:
        found = find_stadium(video_p, model, cfg)
    except Exception as e:  # recorded per video; a crash is a finding, not a skipped video
        row.update(found=False, error=f"{type(e).__name__}: {e}")
        found = None
    row["seconds"] = round(time.time() - t, 1)
    if found is not None:
        cal_path = result_path(out_p, "found", batch, video_p.name, ".json")
        ensure_safe(cal_path, protected)
        save_calibration(found.calibration, cal_path)
        f = found.fit
        row.update(found=True, frame_index=found.frame_index, coverage=round(f.coverage, 3),
                   precision=round(f.precision, 3), mean_error_px=round(f.mean_error_px, 2),
                   rail_in_frame=round(f.rail_in_frame, 3), agreeing_frames=found.agreeing_frames,
                   notch_coverage=round(f.notch_coverage, 3),
                   quadrant_coverage=round(f.quadrant_coverage, 3),
                   perspective=float(max(abs(f.homography[2][0]), abs(f.homography[2][1]))),
                   zone_visibility={k: round(v, 3) for k, v in found.zone_visibility.items()},
                   pocket_out_of_frame=[k for k in ("over_left", "over_right", "xtreme")
                                        if found.zone_visibility[k] < cfg.zone_visible_min])
        prev = result_path(out_p, "preview", batch, video_p.name, ".jpg")
        ensure_safe(prev, protected)
        prev.parent.mkdir(parents=True, exist_ok=True)
        _preview(found.frame, found.calibration, prev,
                 f"{batch}/{video_p.stem} f{found.frame_index} cov {f.coverage:.2f} prec {f.precision:.2f}")
    else:
        row.setdefault("found", False)
    if reference:
        ref_dir = out_p / "reference" / batch
        ensure_safe(ref_dir, protected)
        row["reference"] = _reference(video_p, ref_dir, found)
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
        snap = select_snapshot(frame, PROJECT_ROOT / "calib", out, th, acfg)
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


def truth_rows(truth_dir: Path, out: Path, rows: list[dict], limit_px: float,
               long_edge: int) -> list[dict]:
    """Grade manual truth against the found zones of the same video and the same frame.

    A video name found in two batches is AMBIGUOUS; truth clicked on another frame than the
    finder chose (more than one source frame apart) is FRAME_MISMATCH: the camera may have
    moved, so neither a pass nor a fail would mean anything."""
    graded = []
    for path in sorted(truth_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        row = {"video": doc["video"], "at_s": doc["at_s"]}
        matches = [r for r in rows if r["video"] == doc["video"]]
        if len(matches) > 1:
            row.update(status="AMBIGUOUS")
            graded.append(row)
            continue
        r = matches[0] if matches else None
        found_path = result_path(out, "found", r["batch"], r["video"], ".json") if r else None
        if r is None or not r.get("found"):
            row.update(status="REJECTED")
        elif "fps" not in r or abs(round(doc["at_s"] * r["fps"]) - r["frame_index"]) > 1:
            row.update(status="FRAME_MISMATCH", frame_index=r.get("frame_index"))
        elif not found_path.exists():
            row.update(status="REJECTED")
        else:
            found = load_calibration(found_path)
            truth = {k: [tuple(p) for p in v] for k, v in doc["zones"].items()}
            errs = zone_boundary_errors(found, truth, tuple(doc["frame_size"]), long_edge)
            worst = max(errs.values())
            row.update(boundary_error_px={k: round(v, 2) for k, v in errs.items()},
                       worst_px=round(worst, 2), status="PASS" if worst < limit_px else "FAIL")
        graded.append(row)
    return graded


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
    check_output_dir(out, a.folders)
    rows_path = out / "rows.jsonl"
    if not a.only_truth:
        for sub in ("found", "preview", "reference"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        done = set()
        if rows_path.exists():  # resumable: videos already evaluated are skipped
            done = {(r["batch"], r["video"]) for r in map(json.loads, rows_path.read_text().splitlines())}
        names = batch_names(a.folders)
        protected = [str(p) for p in protected_dirs(a.folders)]
        for sub in ("found", "preview", "reference"):
            ensure_safe(out / sub, [Path(p) for p in protected])
        ensure_safe(rows_path, [Path(p) for p in protected])
        jobs = [(str(v), name, str(out), a.reference, protected)
                for f, name in zip(map(lambda x: x.expanduser(), a.folders), names)
                for v in sorted(f.glob("*.mp4")) if (name, v.name) not in done]
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

        graded = truth_rows(a.truth.expanduser(), out, rows,
                            load_autocalib_config().max_boundary_error_px,
                            load_thresholds().infer_long_edge)
        summary["truth"] = {"rows": graded,
                            "counts": {s: sum(1 for r in graded if r["status"] == s)
                                       for s in ("PASS", "FAIL", "REJECTED", "FRAME_MISMATCH",
                                                 "AMBIGUOUS")}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "truth"}, ensure_ascii=False, indent=1))
    if "truth" in summary:
        print("truth:", summary["truth"]["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
