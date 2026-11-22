"""Build an immutable, local review set from saved refusal cases and raw tracks.

No model inference, identity assignment or changes to existing referee results.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from bxjudge.manifest import probe
from bxjudge.reid_review import MANIFEST_SCHEMA
from bxjudge.workbench_theme import render_reid_page


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clip_bounds(before: int, after: int, fps: float, context_s: float):
    if (
        type(before) is not int
        or type(after) is not int
        or min(before, after) < 0
        or not math.isfinite(fps)
        or fps <= 0
        or not math.isfinite(context_s)
        or context_s <= 0
    ):
        raise ValueError("invalid source frame/context budget")
    padding = math.ceil(context_s * fps)
    return max(0, min(before, after) - padding), max(before, after) + padding + 1


def build(
    diagnostics: Path,
    tracks_dir: Path,
    output: Path,
    coordinate_size: tuple[int, int],
    context_s: float,
):
    diagnostics, tracks_dir, output = (
        Path(diagnostics).resolve(),
        Path(tracks_dir).resolve(),
        Path(output).resolve(),
    )
    if output.exists():
        raise ValueError(
            "output must be a new directory; preserve existing review versions"
        )
    if len(coordinate_size) != 2 or any(
        type(x) is not int or x <= 0 for x in coordinate_size
    ):
        raise ValueError("invalid inference coordinate size")
    data = json.loads(diagnostics.read_text())
    pairs = data["remaining_pairs"]
    if not pairs:
        raise ValueError("no review cases")
    cases, source_hashes, native_hashes, loaded = [], {}, {}, {}
    web = Path(__file__).resolve().parents[1] / "src/bxjudge/web"
    code_hashes = {str(Path(__file__).resolve()): sha(__file__)}
    code_hashes.update(
        {str(web / n): sha(web / n) for n in ("reid_review.js", "reid_review.css")}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reid-build-", dir=output.parent) as temp:
        stage = Path(temp) / "site"
        stage.mkdir()
        for pair in pairs:
            video = pair["video"]
            if Path(video).name != video or not video.lower().endswith(".mp4"):
                raise ValueError("invalid video name")
            native_path = tracks_dir / (Path(video).stem + "-tracks.json")
            if video not in loaded:
                native = json.loads(native_path.read_text())
                source = Path(native["source"]).resolve()
                info = probe(source)
                digest = sha(source)
                if digest != native["source_sha256"] or source.name != video:
                    raise ValueError(
                        "original video no longer matches native observations"
                    )
                if not math.isclose(info.fps, native["fps"], rel_tol=1e-7):
                    raise ValueError("source FPS does not match native observations")
                if not math.isclose(
                    info.width / info.height,
                    coordinate_size[0] / coordinate_size[1],
                    rel_tol=1e-5,
                ):
                    raise ValueError("source/observation aspect ratio mismatch")
                loaded[video] = native, source, info
                source_hashes[str(source)] = digest
                native_hashes[str(native_path)] = sha(native_path)
            native, source, info = loaded[video]
            ids = [pair["from_track"], pair["to_track"]]
            frames = pair["source_frames"]
            if len(frames) != 2 or len(ids) != 2:
                raise ValueError("two endpoints required")
            start, end = clip_bounds(*frames, info.fps, context_s)
            tracks = {t["track_id"]: t for t in native["tracks"]}
            objects = []
            for tid, frame in zip(ids, frames, strict=True):
                track = tracks[tid]
                candidates = [
                    d
                    for d in track["dets"]
                    if d["frame"] == frame and d["cls"] == "bey"
                ]
                if track["cls"] != "bey" or len(candidates) != 1:
                    raise ValueError(
                        "endpoint must be one actual native bey observation"
                    )
                observation = candidates[0]
                box = observation["box"]
                if (
                    len(box) != 4
                    or not all(math.isfinite(v) for v in box)
                    or not box[0] < box[2]
                    or not box[1] < box[3]
                ):
                    raise ValueError("invalid original box")
                objects.append(observation)
            identity = {
                "source_sha256": source_hashes[str(source)],
                "native_ids": ids,
                "source_frames": frames,
            }
            case_id = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()[:16]
            folder = stage / case_id
            folder.mkdir()  # Repeated candidate identities are invalid, never collapsed.
            clip = folder / "context.mp4"
            frame_text = rf"drawtext=text='source frame %{{eif\:n+{start}\:d}}':x=20:y=20:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.7"
            filters = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,{frame_text}"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-filter_threads",
                    "1",
                    "-i",
                    str(source),
                    "-vf",
                    filters,
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-threads",
                    "4",
                    "-movflags",
                    "+faststart",
                    "-fps_mode",
                    "vfr",
                    str(clip),
                ],
                check=True,
            )
            meta = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=nb_frames,width,height",
                        "-of",
                        "json",
                        str(clip),
                    ],
                    text=True,
                )
            )["streams"][0]
            if int(meta["nb_frames"]) != end - start:
                raise ValueError("source context was truncated or frame count changed")
            unique = sorted(set(frames))
            expr = "+".join(rf"eq(n\,{f})" for f in unique)
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-filter_threads",
                    "1",
                    "-i",
                    str(source),
                    "-vf",
                    "select=" + expr,
                    "-fps_mode",
                    "vfr",
                    "-frames:v",
                    str(len(unique)),
                    str(folder / "endpoint-%02d.png"),
                ],
                check=True,
            )
            endpoints = []
            for frame, obs in zip(frames, objects, strict=True):
                image = folder / f"endpoint-{unique.index(frame) + 1:02d}.png"
                sx, sy = (
                    info.width / coordinate_size[0],
                    info.height / coordinate_size[1],
                )
                endpoints.append(
                    {
                        "frame": frame,
                        "native_box": obs["box"],
                        "score": obs["score"],
                        "box": [
                            v * (sx if i % 2 == 0 else sy)
                            for i, v in enumerate(obs["box"])
                        ],
                        "image": f"{case_id}/{image.name}",
                        "image_sha256": sha(image),
                    }
                )
            cases.append(
                {
                    "case_id": case_id,
                    "video": video,
                    "source_sha256": source_hashes[str(source)],
                    "fps": info.fps,
                    "coordinate_size": list(coordinate_size),
                    "source_size": [info.width, info.height],
                    "native_ids": ids,
                    "before": endpoints[0],
                    "after": endpoints[1],
                    "clip": {
                        "path": f"{case_id}/context.mp4",
                        "sha256": sha(clip),
                        "first_source_frame": start,
                        "last_source_frame": end - 1,
                        "frame_count": end - start,
                        "source_frame_counter_baked_in": True,
                    },
                    "scope_status": "unverified",
                    "reasons": pair["gate_rejections"],
                }
            )
        for path, digest in {**source_hashes, **native_hashes, **code_hashes}.items():
            if sha(path) != digest:
                raise ValueError("source changed during evidence export")
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "purpose": "local manual review only; not reidentification predictions or ground truth",
            "diagnostics_sha256": sha(diagnostics),
            "source_files_sha256": source_hashes,
            "generator_files_sha256": code_hashes,
            "native_files_sha256": native_hashes,
            "context_s": context_s,
            "cases": cases,
        }
        raw = (
            json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        ).encode()
        (stage / "manifest.json").write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        page = render_reid_page(manifest, digest)
        (stage / "index.html").write_text(page)
        web = Path(__file__).resolve().parents[1] / "src/bxjudge/web"
        for name in ("reid_review.js", "reid_review.css"):
            shutil.copyfile(web / name, stage / name)
        stage.rename(output)
    return {"manifest_sha256": digest, "case_count": len(cases), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coordinate-size", type=int, nargs=2, required=True)
    parser.add_argument("--context-s", type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.diagnostics,
                args.tracks_dir,
                args.out,
                tuple(args.coordinate_size),
                args.context_s,
            )
        )
    )


if __name__ == "__main__":
    main()
