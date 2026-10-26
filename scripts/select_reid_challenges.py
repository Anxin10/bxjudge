"""Pick a bounded human review population; heuristics are never ground truth."""

import argparse
import hashlib
import itertools
import json
import math
import tempfile
import time
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import yaml

from bxjudge.manifest import probe
from bxjudge.reid_appearance import describe_crop, distances, load_policy
from bxjudge.video import iter_frames
from scripts.analyze_reid_appearance import overlap


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def endpoint_pair(dets, center, fps, offset_s, search_s):
    before = [
        d["frame"] for d in dets if offset_s <= (center - d["frame"]) / fps <= search_s
    ]
    after = [
        d["frame"] for d in dets if offset_s <= (d["frame"] - center) / fps <= search_s
    ]
    return [max(before), min(after)] if before and after else None


def choose_candidates(candidates, excluded, *, per_category, context_s):
    if (
        type(per_category) is not int
        or not 1 <= per_category <= 4
        or not math.isfinite(context_s)
        or context_s <= 0
    ):
        raise ValueError("invalid bounded selection policy")
    for c in candidates:
        if (
            c["category"] not in ("overlap", "similar", "blur")
            or type(c["center_frame"]) is not int
            or c["center_frame"] < 0
            or not math.isfinite(c["fps"])
            or c["fps"] <= 0
            or not math.isfinite(c["rank"])
        ):
            raise ValueError("invalid candidate metric or source time")
    selected, used = [], defaultdict(list)
    for category in ("overlap", "similar", "blur"):
        pool = sorted(
            (c for c in candidates if c["category"] == category),
            key=lambda c: (
                c["rank"],
                c["video"],
                c["center_frame"],
                c["from_track"],
                c["to_track"],
            ),
        )
        category_count = 0
        for c in pool:
            first = min(c["source_frames"]) - math.ceil(context_s * c["fps"])
            last = max(c["source_frames"]) + math.ceil(context_s * c["fps"])
            if any(
                first <= end and last >= start
                for start, end in excluded.get(c["video"], []) + used[c["video"]]
            ):
                continue
            selected.append(c)
            used[c["video"]].append((first, last))
            category_count += 1
            if category_count == per_category:
                break
    return selected


def load_selection_policy(path):
    p = yaml.safe_load(Path(path).read_text())
    if (
        not isinstance(p, dict)
        or set(p)
        != {
            "mode",
            "frame_stride",
            "per_category",
            "context_s",
            "endpoint_offset_s",
            "endpoint_search_s",
            "coordinate_size",
        }
        or p["mode"] != "review_candidates"
    ):
        raise ValueError("invalid selection policy fields")
    for field, low, high in [("frame_stride", 1, 120), ("per_category", 1, 4)]:
        if type(p[field]) is not int or not low <= p[field] <= high:
            raise ValueError("invalid selection sampling bound")
    for field in ("context_s", "endpoint_offset_s", "endpoint_search_s"):
        if (
            type(p[field]) not in (int, float)
            or not math.isfinite(p[field])
            or not 0 < p[field] <= 5
        ):
            raise ValueError("invalid candidate time window")
    if p["endpoint_offset_s"] > p["endpoint_search_s"]:
        raise ValueError("endpoint search must include the requested offset")
    if (
        not isinstance(p["coordinate_size"], list)
        or len(p["coordinate_size"]) != 2
        or any(type(v) is not int or v < 2 for v in p["coordinate_size"])
    ):
        raise ValueError("invalid inference coordinate size")
    return p


def build(
    tracks_dir,
    prior_manifest_path,
    reference_pointer_path,
    appearance_policy_path,
    selection_policy_path,
    output,
):
    started = time.perf_counter()
    (
        tracks_dir,
        prior_manifest_path,
        reference_pointer_path,
        appearance_policy_path,
        selection_policy_path,
        output,
    ) = (
        Path(p).resolve()
        for p in (
            tracks_dir,
            prior_manifest_path,
            reference_pointer_path,
            appearance_policy_path,
            selection_policy_path,
            output,
        )
    )
    if output.exists():
        raise ValueError("output must be a new directory")
    policy = load_selection_policy(selection_policy_path)
    appearance_policy = load_policy(appearance_policy_path)
    module_root = Path(__file__).resolve().parents[1] / "src/bxjudge"
    inputs = {
        str(p): sha(p)
        for p in (
            prior_manifest_path,
            reference_pointer_path,
            appearance_policy_path,
            selection_policy_path,
            Path(__file__).resolve(),
            Path(__file__).with_name("analyze_reid_appearance.py"),
            *(
                module_root / name
                for name in ("reid_appearance.py", "video.py", "manifest.py")
            ),
        )
    }
    prior = read(prior_manifest_path)
    pointer = read(reference_pointer_path)
    reference_path = Path(pointer["reference"]).resolve()
    if sha(reference_path) != pointer["sha256"]:
        raise ValueError("confirmed reference SHA mismatch")
    inputs[str(reference_path)] = pointer["sha256"]
    excluded = defaultdict(list)
    for case in prior["cases"]:
        excluded[case["video"]].append(
            (case["clip"]["first_source_frame"], case["clip"]["last_source_frame"])
        )
    for row in read(reference_path):
        pad = math.ceil(policy["context_s"] * row["fps"])
        excluded[row["video"]].append((row["frame"] - pad, row["frame"] + pad))
    candidates, population = (
        [],
        {"sampled_frames": 0, "measured_boxes": 0, "unusable_boxes": 0},
    )
    for native_path in sorted(tracks_dir.glob("*-tracks.json")):
        native = read(native_path)
        inputs[str(native_path)] = sha(native_path)
        source = Path(native["source"]).resolve()
        if (
            source.suffix.lower() != ".mp4"
            or native_path.stem != source.stem + "-tracks"
        ):
            raise ValueError("native source identity mismatch")
        inputs[str(source)] = sha(source)
        if (
            inputs[str(source)] != native["source_sha256"]
            or prior["source_files_sha256"].get(str(source)) != native["source_sha256"]
        ):
            raise ValueError("source SHA mismatch")
        info = probe(source)
        width, height = policy["coordinate_size"]
        if not math.isclose(info.fps, native["fps"], rel_tol=1e-7) or not math.isclose(
            info.width / info.height, width / height, rel_tol=1e-5
        ):
            raise ValueError("source FPS or coordinates mismatch")
        tracks = {t["track_id"]: t for t in native["tracks"]}
        if len(tracks) != len(native["tracks"]):
            raise ValueError("duplicate native track")
        by_frame = defaultdict(list)
        for track in tracks.values():
            if track["cls"] != "bey":
                continue
            seen = set()
            for d in track["dets"]:
                if (
                    type(d["frame"]) is not int
                    or d["frame"] < 0
                    or d["frame"] in seen
                    or d["cls"] != "bey"
                ):
                    raise ValueError("invalid native observation")
                seen.add(d["frame"])
                if d["frame"] % policy["frame_stride"] == 0:
                    by_frame[d["frame"]].append((track["track_id"], d))
        if not by_frame:
            continue

        def candidate(
            category, center, ids, frames, rank, reason, video=source.name, fps=info.fps
        ):
            return {
                "category": category,
                "center_frame": center,
                "rank": float(rank),
                "video": video,
                "fps": fps,
                "from_track": ids[0],
                "to_track": ids[1],
                "source_frames": frames,
                "gate_rejections": [
                    reason,
                    "挑片線索未經人工確認；不是碰撞、模糊或同物件真值",
                ],
                "applied": False,
            }

        remaining = set(by_frame)
        with closing(
            iter_frames(
                source, start=0, long_edge=max(info.width, info.height), hwaccel="cpu"
            )
        ) as decoded:
            for frame, image in decoded:
                if frame not in by_frame:
                    continue
                population["sampled_frames"] += 1
                measured = []
                for tid, d in by_frame[frame]:
                    box = [
                        v * (info.width / width if i % 2 == 0 else info.height / height)
                        for i, v in enumerate(d["box"])
                    ]
                    desc = describe_crop(image, box, appearance_policy)
                    if desc["descriptor"] is None:
                        population["unusable_boxes"] += 1
                        continue
                    population["measured_boxes"] += 1
                    measured.append((tid, d, desc))
                    frames = endpoint_pair(
                        tracks[tid]["dets"],
                        frame,
                        info.fps,
                        policy["endpoint_offset_s"],
                        policy["endpoint_search_s"],
                    )
                    if frames:
                        laplacian = desc["quality"]["laplacian_variance"]
                        candidates.append(
                            candidate(
                                "blur",
                                frame,
                                [tid, tid],
                                frames,
                                laplacian,
                                f"低清晰度測量候選：中心f{frame} Laplacian variance={laplacian:.2f}；前後native ID相同也不保證同物件",
                            )
                        )
                for a, b in itertools.combinations(measured, 2):
                    distance = distances(a[2], b[2])["color_hellinger"]
                    candidates.append(
                        candidate(
                            "similar",
                            frame,
                            [a[0], b[0]],
                            [frame, frame],
                            distance,
                            f"同格外觀相近候選：色彩距離={distance:.3f}；可能不同顆、同顆重複框或誤偵",
                        )
                    )
                    iou = overlap(a[1]["box"], b[1]["box"])
                    frames = endpoint_pair(
                        tracks[a[0]]["dets"],
                        frame,
                        info.fps,
                        policy["endpoint_offset_s"],
                        policy["endpoint_search_s"],
                    )
                    if iou > 0 and frames:
                        candidates.append(
                            candidate(
                                "overlap",
                                frame,
                                [a[0], a[0]],
                                frames,
                                -iou,
                                f"重疊前後候選：中心f{frame} native {a[0]}/{b[0]} IoU={iou:.3f}；可能遮擋、重複框或誤偵，不等於碰撞",
                            )
                        )
                remaining.remove(frame)
                if not remaining:
                    break
        if remaining:
            raise ValueError("requested native frames missing from source")
    chosen = choose_candidates(
        candidates,
        excluded,
        per_category=policy["per_category"],
        context_s=policy["context_s"],
    )
    if not chosen:
        raise ValueError(
            "no new non-overlapping challenge candidates; no review set published"
        )
    for p, digest in inputs.items():
        if sha(p) != digest:
            raise ValueError("input SHA changed during selection")
    report = {
        "schema": "bxjudge.reid-challenge-selection.v1",
        "remaining_pairs": chosen,
        "input_sha256": inputs,
        "population": population,
        "scored_candidate_count": len(candidates),
        "selected_by_category": {
            k: sum(c["category"] == k for c in chosen)
            for k in ("overlap", "similar", "blur")
        },
        "selection_policy": policy,
        "excluded_intervals": dict(excluded),
        "applied": False,
        "limitations": [
            "selection_bias",
            "two_reused_development_videos",
            "no_ground_truth_or_accuracy",
            "scope_unverified",
        ],
        "extraction_wall_s": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".challenge-select-", dir=output.parent
    ) as temp:
        stage = Path(temp) / "selection"
        stage.mkdir()
        (stage / "selection.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        stage.rename(output)
    return report


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "tracks-dir",
        "prior-manifest",
        "reference-pointer",
        "appearance-policy",
        "selection-policy",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    r = build(
        a.tracks_dir,
        a.prior_manifest,
        a.reference_pointer,
        a.appearance_policy,
        a.selection_policy,
        a.out,
    )
    print(
        json.dumps(
            {
                "selected_by_category": r["selected_by_category"],
                "population": r["population"],
                "offline_total_wall_s": time.perf_counter() - started,
            }
        )
    )


if __name__ == "__main__":
    main()
