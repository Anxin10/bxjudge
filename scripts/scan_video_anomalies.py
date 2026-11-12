"""Scan the inventory locally and save candidate anomalies per original video."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from bxjudge.timebase import DEFAULT_VIDEO_SOURCES, load_video_sources, probe_timeline
from bxjudge.video_anomalies import AnomalyPolicy, scan_video


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_VIDEO_SOURCES)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    policy = load_video_sources(args.policy)
    policy_hash = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    settings = AnomalyPolicy(**policy["anomalies"])
    if args.output.resolve().is_relative_to(args.video_root.resolve()):
        raise ValueError("scan output must be outside the source video directory")
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    errors = []
    for n, item in enumerate(inventory, 1):
        path = args.video_root / item["batch"] / item["file"]
        target = args.output / item["batch"] / (path.name + ".json")
        target.parent.mkdir(exist_ok=True)
        try:
            before = path.stat()
            signature = {"size": before.st_size, "mtime_ns": before.st_mtime_ns, "policy_sha256": policy_hash}
            if target.exists():
                existing = json.loads(target.read_text())
                if existing["signature"] == signature:
                    counts[str(path)] = len(existing["events"])
                    continue
            clock = probe_timeline(path, policy)
            events = scan_video(path, clock, settings)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("source changed during scan")
            result = [
                {
                    **asdict(event),
                    "first_playback_s": clock.playback_s(event.first_frame),
                    "last_playback_s": clock.playback_s(event.last_frame),
                    "physical_duration_ms": clock.elapsed_ms(event.first_frame, event.last_frame),
                }
                for event in events
            ]
            write_json(target, {"video": str(path), "signature": signature, "frames": len(clock.pts_s), "events": result})
            counts[str(path)] = len(events)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            errors.append({"video": str(path), "error": str(error)})
        print(f"{n}/{len(inventory)} {item['file']}", flush=True)
    write_json(args.output / "summary.json", {
        "expected": len(inventory), "scanned": len(counts), "videos_with_events": sum(v > 0 for v in counts.values()),
        "events": sum(counts.values()), "errors": errors, "policy_sha256": policy_hash,
    })
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
