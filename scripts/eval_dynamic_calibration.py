"""Measure dynamic calibration (plan 2026-11-12-dynamic-calibration, Task 8). Read only on
videos, calib/ and the keypoint dataset; every result goes under --out.

Subcommands:
  modes <folder>...          camera mode per video (measured motion, overrides), per batch
  refind-accuracy            one keypoint fit per labelled image vs the user's labels, per
                             batch (image SHA -> stadium-keypoints manifest), train/val apart
  geometry <folder>...       decode only (no detector): keypoint snapshot, then the fixed or
                             handheld tracker over every frame; coverage, status counts,
                             re-find drift (disagreement_px), invalidated frames, time

Usage:
    uv run python scripts/eval_dynamic_calibration.py modes --out <dir> <batch folder>...
    uv run python scripts/eval_dynamic_calibration.py refind-accuracy --out <dir> \
        [--dataset runs/stadium-kp-20261112] [--manifest <stadium-keypoints manifest.json>]
    uv run python scripts/eval_dynamic_calibration.py geometry --out <dir> [--limit N] \
        <batch folder>...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from bxjudge import PROJECT_ROOT
from bxjudge.autocalib import load_autocalib_config
from bxjudge.calibration import scale_calibration
from bxjudge.camera_motion import decide_camera_mode, load_camera_policy, measure_video_motion
from bxjudge.dynamic_calibration import DynamicCalibration, load_dynamic_policy
from bxjudge.handheld_calibration import HandheldCalibration
from bxjudge.manifest import probe
from bxjudge.stadium_finder import (
    _REF_LONG_EDGE,
    _worst_zone_shift,
    fit_from_keypoints,
    rail_mask,
)
from bxjudge.stadium_locator import KeypointLocator, keypoint_snapshot, load_stadium_tools
from bxjudge.temporal_geometry import load_temporal_policy
from bxjudge.trajectory_video import load_video_policy
from bxjudge.video import iter_frames

DEFAULT_DATASET = PROJECT_ROOT / "runs" / "stadium-kp-20261112"
DEFAULT_MANIFEST = (PROJECT_ROOT / "data" / "detector-hard-cases-20261030"
                    / "stadium-keypoints-20261111" / "manifest.json")
# Points on the stadium plane (the user's 10-point labels: 7a/8a/8b sit 24-38 px off it).
COPLANAR = (0, 1, 2, 3, 4, 5, 7)


# --------------------------------------------------------------------------- output safety
def protected_dirs(folders: list[Path], extra: list[Path] = ()) -> list[Path]:
    """calib/, the keypoint dataset (default and given), the manifest's folder and every input
    video folder, resolved: never written."""
    return [(PROJECT_ROOT / "calib").resolve(), DEFAULT_DATASET.resolve(),
            *(Path(f).expanduser().resolve() for f in [*folders, *extra])]


def ensure_safe(path: Path, protected: list[Path]) -> None:
    real = Path(path).expanduser().resolve()
    for p in protected:
        if real == p or p in real.parents:
            raise SystemExit(f"{path}: refusing to write inside {p}")


def ensure_inside(path: Path, root: Path) -> None:
    """A write target must resolve (links followed) inside --out."""
    real, top = Path(path).expanduser().resolve(), Path(root).expanduser().resolve()
    if real != top and top not in real.parents:
        raise SystemExit(f"{path}: resolves to {real}, outside {top}")


def _target(out: Path, name: str, protected: list[Path]) -> Path:
    path = out / name
    ensure_safe(path, protected)
    ensure_inside(path, out)
    return path


def _key(r: dict) -> tuple[str, str, str]:
    """A row's identity: (batch, "", video) for a video; ("", split, file) for a labelled
    image, whose batch (looked up by its SHA) is an attribute that may change."""
    split = r.get("split") or ""
    return ("" if split else r["batch"]), split, r["video"]


def ensure_no_links(root: Path) -> None:
    """Refuse a write folder holding any symlink (folder or file): a snapshot PNG/JSON written
    through one could land outside --out."""
    root = Path(root)
    if root.is_symlink():
        raise SystemExit(f"{root}: is a link; refusing to write through it")
    if root.is_dir():
        for p in root.rglob("*"):
            if p.is_symlink():
                raise SystemExit(f"{p}: is a link; refusing to write through it")


def _done(rows_path: Path) -> dict[tuple[str, str, str], dict]:
    """Rows measured for good, by identity; an execution error (``error: ...``) is retried."""
    return {_key(r): r for r in _rows(rows_path)  # each sample's latest row only
            if not str(r.get("result", "")).startswith("error")}


def _is_done(done: dict, key: tuple[str, str, str], fingerprint: dict) -> bool:
    """Measured, from the same input (video path, size and mtime; image and label SHA)."""
    row = done.get(key)
    return row is not None and all(row.get(k) == v for k, v in fingerprint.items())


def _video_fingerprint(video: Path) -> dict:
    st = Path(video).stat()
    return {"source": str(Path(video).resolve()), "size": st.st_size,
            "mtime_ns": st.st_mtime_ns}


def _rows(rows_path: Path) -> list[dict]:
    if not rows_path.exists():
        return []
    latest: dict = {}
    for r in map(json.loads, rows_path.read_text().splitlines()):
        latest[_key(r)] = r
    return list(latest.values())


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_manifest(cmd: str, camera_policy: dict, device: str | None,
                 hwaccel: str | None = None) -> dict:
    """What decides the results: subcommand, camera policy, tracker policies (dynamic,
    stadium follow, temporal), auto-calibration, finder config, keypoint weights, stadium model
    and reference, decoder, and the code. A resumed run must match it exactly."""
    from dataclasses import asdict, is_dataclass

    from bxjudge.stadium_finder import DEFAULT_STADIUM_MODEL, load_stadium_finder_config
    from bxjudge.stadium_keypoints import DEFAULT_KEYPOINT_REFERENCE, resolve_weights

    cfg = load_stadium_finder_config()
    code = [Path(__file__), *(PROJECT_ROOT / "src" / "bxjudge" / n for n in (
        "camera_motion.py", "handheld_calibration.py", "dynamic_calibration.py",
        "stadium_follow.py", "stadium_locator.py", "stadium_finder.py", "stadium_keypoints.py",
        "temporal_geometry.py"))]
    return json.loads(json.dumps({
        "cmd": cmd, "camera_policy": camera_policy, "device": device, "hwaccel": hwaccel,
        "dynamic_policy": load_dynamic_policy(),
        "follow_policy": load_video_policy()["stadium_follow"],
        "temporal_policy": load_temporal_policy(),
        "autocalib": (asdict(autocalib) if is_dataclass(autocalib := load_autocalib_config())
                      else None),
        "finder_config": asdict(cfg),
        "weights_sha256": [_sha(w) for w in resolve_weights(cfg.keypoint_weights)],
        "model_sha256": _sha(DEFAULT_STADIUM_MODEL),
        "reference_sha256": _sha(DEFAULT_KEYPOINT_REFERENCE),
        "code_sha256": {c.name: _sha(c) for c in code},
    }))


def check_run_manifest(out: Path, manifest: dict, has_rows: bool, protected: list[Path],
                       sources: dict[str, str] | None = None) -> None:
    """Write the manifest of a fresh output folder; refuse to resume one measured otherwise
    (another subcommand, policy, model, weights or code): use a new --out. ``sources`` binds
    each batch name to its folder: another batch is appended, a batch name already bound to
    another folder is refused."""
    path = _target(out, "manifest.json", protected)
    sources = dict(sources or {})
    if path.exists():
        old = json.loads(path.read_text())
        bound = old.pop("sources", {})
        if old != manifest:
            raise SystemExit(f"{path}: run manifest differs (another subcommand, policy, "
                             "model, weights or code); use a new --out")
        for batch, folder in sources.items():
            if bound.get(batch, folder) != folder:
                raise SystemExit(f"{path}: batch {batch} was measured from {bound[batch]}, "
                                 f"not {folder}; use a new --out")
        sources = {**bound, **sources}
    elif has_rows:
        raise SystemExit(f"{path}: rows without a run manifest; use a new --out")
    path.write_text(json.dumps({**manifest, "sources": sources}, ensure_ascii=False, indent=1))


def batch_sources(folders: list[Path]) -> dict[str, str]:
    """Batch name (folder name) -> resolved folder; one name for two folders is refused."""
    out: dict[str, str] = {}
    for f in folders:
        name, real = Path(f).expanduser().name, str(Path(f).expanduser().resolve())
        if out.get(name, real) != real:
            raise SystemExit(f"batch {name}: two folders ({out[name]}, {real})")
        out[name] = real
    return out


def _videos(folders: list[Path], limit: int | None) -> list[tuple[str, Path]]:
    out = []
    for f in folders:
        vids = sorted(Path(f).expanduser().glob("*.mp4"))
        out += [(Path(f).expanduser().name, v) for v in vids[:limit]]
    return out


# --------------------------------------------------------------------------- modes
def mode_row(batch: str, video: Path, policy: dict) -> dict:
    began = time.perf_counter()
    fps = probe(video).fps
    try:
        motion = measure_video_motion(video, fps, policy, long_edge=_REF_LONG_EDGE,
                                      hwaccel="cpu")
        error = None
    except (ValueError, OSError) as e:
        motion, error = None, f"{type(e).__name__}: {e}"
    mode = decide_camera_mode(video, batch, motion, policy)
    return {"batch": batch, "video": video.name, **_video_fingerprint(video),
            "mode": mode.mode, "mode_source": mode.source,
            "px_per_s": None if motion is None else motion.px_per_s,
            "pairs": 0 if motion is None else motion.pairs, "error": error,
            "seconds": round(time.perf_counter() - began, 3)}


def summarise_modes(rows: list[dict]) -> dict:
    out = {}
    for batch in sorted({r["batch"] for r in rows}):
        rs = [r for r in rows if r["batch"] == batch]
        speeds = [r["px_per_s"] for r in rs if r["px_per_s"] is not None]
        handheld = sum(r["mode"] == "handheld" for r in rs)
        out[batch] = {
            "videos": len(rs), "handheld": handheld, "fixed": len(rs) - handheld,
            "handheld_fraction": round(handheld / len(rs), 3),
            "unmeasured": sum(r["mode_source"] == "unmeasured" for r in rs),
            "overridden": sum(r["mode_source"] == "override" for r in rs),
            "px_per_s": None if not speeds else {
                "median": float(np.median(speeds)), "p10": float(np.percentile(speeds, 10)),
                "p90": float(np.percentile(speeds, 90)), "max": float(np.max(speeds))},
        }
    return out


# --------------------------------------------------------------------------- refind accuracy
def _label_points(label: Path, size: tuple[int, int]) -> np.ndarray:
    """(10, 3) x, y, visibility in image px from a YOLO-pose label line."""
    values = [float(v) for v in label.read_text().split()]
    pts = np.array(values[5:5 + 30], np.float64).reshape(10, 3)
    pts[:, 0] *= size[0]
    pts[:, 1] *= size[1]
    return pts


def refind_row(image_path: Path, label_path: Path, split: str, batch: str, tools) -> dict:
    """One keypoint fit on the labelled image (decoded at the 1280 inference size, as the
    pipeline does) against the homography of the labelled coplanar points: worst zone mean
    vertex distance, inference px."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    row = {"image": image_path.name, "split": split, "batch": batch}
    if img is None:
        return {**row, "result": "unreadable"}
    scale = _REF_LONG_EDGE / max(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)),
                     interpolation=cv2.INTER_AREA)
    pts = _label_points(label_path, (img.shape[1], img.shape[0]))
    ref = tools.finder.reference.array()
    use = [i for i in COPLANAR if pts[i, 2] > 0 and np.isfinite(pts[i, :2]).all()]
    if len(use) < 4:
        return {**row, "result": "too_few_labels"}
    truth, _ = cv2.findHomography(ref[use], pts[use, :2], 0)
    if (truth is None or not np.isfinite(truth).all() or abs(truth[2, 2]) < 1e-12
            or abs(np.linalg.det(truth)) < 1e-12):
        return {**row, "result": "invalid_truth"}  # degenerate labels, not a fit error
    began = time.perf_counter()
    try:
        predicted = tools.finder.predict(img)
        fit = None if predicted is None else fit_from_keypoints(
            predicted, tools.finder.reference, rail_mask(img, tools.cfg), tools.model, tools.cfg)
    except (RuntimeError, ValueError, MemoryError, cv2.error) as e:
        return {**row, "result": f"error: {type(e).__name__}: {e}"}
    seconds = round(time.perf_counter() - began, 3)
    if fit is None:
        return {**row, "result": "no_fit", "seconds": seconds}
    worst = _worst_zone_shift(tools.model, np.asarray(fit.homography), truth)
    return {**row, "result": "fit", "worst_zone_px": float(worst), "seconds": seconds}


def manifest_batches(manifest: Path) -> dict[str, str]:
    frames = json.loads(Path(manifest).read_text())["frames"]
    return {f["image_sha256"]: f["batch"] for f in frames}


def summarise_refinds(rows: list[dict]) -> dict:
    out = {}
    for key in sorted({(r["batch"], r["split"]) for r in rows}):
        rs = [r for r in rows if (r["batch"], r["split"]) == key]
        err = [r["worst_zone_px"] for r in rs if r["result"] == "fit"]
        out[f"{key[0]}/{key[1]}"] = {
            "images": len(rs), "fit": len(err),
            "results": dict(Counter(r["result"].split(":")[0] for r in rs)),
            "worst_zone_px": None if not err else {
                "median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
                "max": float(np.max(err))},
        }
    return out


# --------------------------------------------------------------------------- geometry
def geometry_row(batch: str, video: Path, tools, camera_policy: dict, out: Path,
                 device_hwaccel: str = "cuda") -> dict:
    """Keypoint snapshot plus the tracker its camera mode selects, over the whole video; no
    detector (no foreground boxes)."""
    began = time.perf_counter()
    info = probe(video)
    row = {"batch": batch, "video": video.name, **_video_fingerprint(video)}
    try:
        motion = measure_video_motion(video, info.fps, camera_policy, long_edge=_REF_LONG_EDGE,
                                      hwaccel=device_hwaccel)
    except (ValueError, OSError):
        motion = None
    mode = decide_camera_mode(video, batch, motion, camera_policy)
    handheld = mode.mode == "handheld"
    row.update(mode=mode.mode, mode_source=mode.source,
               px_per_s=None if motion is None else motion.px_per_s)
    snap, found = keypoint_snapshot(video, out / "calibrations-run", tools,
                                    load_autocalib_config(), set(), hwaccel=device_hwaccel,
                                    allow_single=handheld)
    row["snapshot_seconds"] = round(time.perf_counter() - began, 3)
    if snap is None:
        return {**row, "result": "no_snapshot", "seconds": row["snapshot_seconds"]}
    row.update(snapshot_method=found.method, snapshot_frame=found.frame_index)
    frames = iter_frames(video, long_edge=_REF_LONG_EDGE, hwaccel=device_hwaccel)
    first_idx, first = next(frames)
    c = scale_calibration(snap.calibration, (first.shape[1], first.shape[0]))
    reference = cv2.imread(snap.calibration.ref_frame)
    locator = KeypointLocator.for_snapshot(tools, snap, c, cv2.resize(reference, c.frame_size))
    follow = load_video_policy()["stadium_follow"]
    temporal = load_temporal_policy()
    if handheld:
        if locator is None:
            return {**row, "result": "reference_unlocated"}
        single = found.agreeing_frames < 2
        geometry = HandheldCalibration(
            c, reference, info.fps, follow, camera_policy["handheld"], locator,
            minimum_visible=temporal["min_visible_fraction"],
            foreground_margin_px=temporal["foreground_margin_px"],
            start_confirmed=not single, start_frame=found.frame_index)
    else:
        geometry = DynamicCalibration(c, reference, info.fps, follow,
                                      load_dynamic_policy(), keypoint_locator=locator)
    geometry.set_foreground([])
    track_began = time.perf_counter()
    geometry.update(first_idx, first)
    for idx, image in frames:
        geometry.update(idx, image)
    geometry.finish()
    record = geometry.record()
    samples = record["samples"]
    statuses = Counter(s["status"] for s in samples)
    emitted = sum(s["alignment"] is not None and s["status"] != "bridged" for s in samples)
    drift = [r["disagreement_px"] for r in record.get("refinds", [])
             if r["disagreement_px"] is not None]
    return {**row, "result": "tracked", "frames": len(samples),
            "coverage": round(emitted / max(1, len(samples)), 4),
            "statuses": dict(statuses), "invalidated": statuses.get("invalidated", 0),
            "segments": [s["kind"] for s in record["segments"]],
            "refinds": dict(Counter(r["result"] for r in record.get("refinds", []))),
            "refind_disagreement_px": drift,
            "track_seconds": round(time.perf_counter() - track_began, 3),
            "seconds": round(time.perf_counter() - began, 3)}


def summarise_geometry(rows: list[dict]) -> dict:
    out = {}
    for key in sorted({(r["batch"], r.get("mode")) for r in rows}):
        rs = [r for r in rows if (r["batch"], r.get("mode")) == key]
        tracked = [r for r in rs if r["result"] == "tracked"]
        drift = [d for r in tracked for d in r["refind_disagreement_px"]]
        frames = sum(r["frames"] for r in tracked)
        out[f"{key[0]}/{key[1]}"] = {
            "videos": len(rs), "results": dict(Counter(r["result"] for r in rs)),
            "coverage_of_tracked_frames": None if not frames else round(
                sum(r["coverage"] * r["frames"] for r in tracked) / frames, 4),
            "coverage_per_video_median": None if not tracked else float(
                np.median([r["coverage"] for r in tracked])),
            "invalidated_frames": sum(r["invalidated"] for r in tracked),
            "refind_disagreement_px": None if not drift else {
                "n": len(drift), "median": float(np.median(drift)),
                "p90": float(np.percentile(drift, 90)), "max": float(np.max(drift))},
            "seconds_median": None if not rs else float(np.median([r["seconds"] for r in rs])),
        }
    return out


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("modes")
    r = sub.add_parser("refind-accuracy")
    g = sub.add_parser("geometry")
    for q in (m, r, g):
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--camera-mode", type=Path, default=PROJECT_ROOT / "config" /
                       "camera_mode.yaml")
    for q in (m, g):
        q.add_argument("folders", nargs="+", type=Path)
        q.add_argument("--limit", type=int, default=None, help="first N videos per folder")
    for q in (r, g):
        q.add_argument("--device", default="0")
    r.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    r.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    g.add_argument("--hwaccel", default="cuda")
    a = p.parse_args(argv)
    out = a.out.expanduser()
    folders = getattr(a, "folders", [])
    extra = ([a.dataset.expanduser(), a.manifest.expanduser().parent]
             if a.cmd == "refind-accuracy" else [])
    protected = protected_dirs(folders, extra)
    ensure_safe(out, protected)
    ensure_inside(out, out)
    policy = load_camera_policy(a.camera_mode.expanduser())
    if a.cmd == "geometry":
        # Snapshots go to calibrations-run/calibrations: checked before the first write.
        snaps = _target(out, "calibrations-run", protected)
        _target(snaps, "calibrations", protected)
        ensure_inside(snaps / "calibrations", out)
        ensure_no_links(snaps)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = _target(out, "rows.jsonl", protected)
    summary_path = _target(out, "summary.json", protected)
    check_run_manifest(out, run_manifest(a.cmd, policy, getattr(a, "device", None),
                                         getattr(a, "hwaccel", None)),
                       rows_path.exists() and rows_path.stat().st_size > 0, protected,
                       batch_sources(folders))
    done = _done(rows_path)
    tools = None
    if a.cmd in ("refind-accuracy", "geometry"):
        device = a.device if a.device == "cpu" else int(a.device)
        tools = load_stadium_tools(device=device)
    with open(rows_path, "a") as fh:
        def emit(row):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(json.dumps({k: v for k, v in row.items()
                              if k not in ("refind_disagreement_px", "statuses")},
                             ensure_ascii=False), flush=True)

        if a.cmd == "modes":
            for batch, video in _videos(folders, a.limit):
                if not _is_done(done, (batch, "", video.name), _video_fingerprint(video)):
                    emit(mode_row(batch, video, policy))
        elif a.cmd == "geometry":
            _target(out, "calibrations-run", protected).mkdir(exist_ok=True)
            for batch, video in _videos(folders, a.limit):
                if not _is_done(done, (batch, "", video.name), _video_fingerprint(video)):
                    ensure_no_links(out / "calibrations-run")  # before each snapshot write
                    emit(geometry_row(batch, video, tools, policy, out, a.hwaccel))
        else:
            batches = manifest_batches(a.manifest.expanduser())
            for split in ("train", "val"):
                for image in sorted((a.dataset / "images" / split).glob("*.jpg")):
                    sha = hashlib.sha256(image.read_bytes()).hexdigest()
                    batch = batches.get(sha, "unmatched")
                    label = a.dataset / "labels" / split / f"{image.stem}.txt"
                    prints = {"image_sha256": sha, "batch": batch,
                              "label_sha256": _sha(label) if label.exists() else None}
                    if _is_done(done, ("", split, image.name), prints):
                        continue
                    emit({**refind_row(image, label, split, batch, tools),
                          "video": image.name, **prints})
    rows = _rows(rows_path)
    summary = {"modes": summarise_modes, "geometry": summarise_geometry,
               "refind-accuracy": summarise_refinds}[a.cmd](rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
