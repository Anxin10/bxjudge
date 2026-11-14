"""Labeling CLI orchestration; GPU runtime executes in one isolated process."""

import json
from pathlib import Path

from bxjudge.manifest import probe

from .config import load_config


def _reject_duplicate_content(entries: list[dict]) -> None:
    seen = {}
    for entry in entries:
        digest = entry["video_sha256"]
        if digest in seen:
            raise ValueError(
                f"duplicate video content: {entry['video']} matches {seen[digest]} "
                f"(SHA256 {digest}); remove the duplicate input"
            )
        seen[digest] = entry["video"]


def command(args):
    cfg = load_config(args.config)
    out = Path(args.out or cfg["output_dir"]).expanduser().resolve()
    if args.label_command == "sample":
        from .prelabel import video_digest
        from .sample_frames import change_energy, select_frames

        folder = Path(args.folder or cfg["input_folder"]).expanduser()
        videos = sorted(
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"
        )
        if not videos:
            raise ValueError(f"no MP4 videos in {folder}")
        entries = [
            {"video": str(video.resolve()), "video_sha256": video_digest(video)}
            for video in videos
        ]
        _reject_duplicate_content(entries)
        for entry in entries:
            video = Path(entry["video"])
            info = probe(video)
            energy = change_energy(video, cfg["energy_long_edge"])
            frames = select_frames(
                energy,
                info.fps,
                args.per_video if args.per_video is not None else cfg["per_video"],
                cfg["uniform_ratio"],
                cfg["min_gap_s"],
            )
            entry.update(fps=info.fps, frames=frames)
        out.mkdir(parents=True, exist_ok=True)
        (out / "samples.json").write_text(
            json.dumps({"videos": entries}, indent=2), encoding="utf-8"
        )
        print(
            f"{len(entries)} videos, {sum(len(v['frames']) for v in entries)} selected frames; {out}/samples.json"
        )
    elif args.label_command == "prelabel":
        from .prelabel import run_prelabel_jobs, video_digest, write_json

        manifest = Path(args.samples) if args.samples else out / "samples.json"
        entries = json.loads(manifest.read_text(encoding="utf-8"))["videos"]
        _reject_duplicate_content(entries)
        jobs = []
        for entry in entries:
            if video_digest(Path(entry["video"])) != entry["video_sha256"]:
                raise ValueError("source video changed since sampling")
            jobs.append(
                {
                    "video": str(Path(entry["video"]).resolve()),
                    "video_sha256": entry["video_sha256"],
                    "frames": entry["frames"],
                    "out_json": str(out / (entry["video_sha256"] + ".json")),
                }
            )
        run_prelabel_jobs(jobs, cfg)
        tasks = []
        for job in jobs:
            tasks.extend(json.loads(Path(job["out_json"]).read_text(encoding="utf-8")))
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "tasks.json", tasks)
        print(f"Import {out}/tasks.json into Label Studio")
    elif args.label_command == "export":
        from .export import label_studio_to_coco

        label_studio_to_coco(
            Path(args.export_json),
            out,
            args.val_ratio if args.val_ratio is not None else cfg["val_ratio"],
        )
        print(f"COCO dataset saved: {out}")
    else:
        from .export import extend_training_dataset

        extend_training_dataset(Path(args.baseline), Path(args.export_json), out)
        print(f"Training dataset extended; validation unchanged: {out}")
    return 0


def add_label(sub):
    p = sub.add_parser(
        "label", help="sample videos, LocateAnything prelabel, or export reviewed COCO"
    )
    actions = p.add_subparsers(dest="label_command", required=True)
    for name in ["sample", "prelabel", "export", "extend-training"]:
        action = actions.add_parser(name)
        action.add_argument("--config", default="config/labeling.yaml")
        action.add_argument(
            "--out", required=name == "extend-training",
            help="new dataset directory (required for extend-training; otherwise from config)"
        )
        action.set_defaults(func=command)
        if name == "sample":
            action.add_argument("folder", nargs="?")
            action.add_argument("--per-video", type=int)
        elif name == "prelabel":
            action.add_argument(
                "--samples", help="sample manifest (default output_dir/samples.json)"
            )
        elif name == "export":
            action.add_argument(
                "export_json", help="completed human Label Studio JSON export"
            )
            action.add_argument("--val-ratio", type=float)
        else:
            action.add_argument("baseline", help="existing project COCO dataset; validation stays frozen")
            action.add_argument("export_json", help="new completed human Label Studio JSON export")
