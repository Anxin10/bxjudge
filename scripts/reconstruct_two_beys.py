"""Replay saved detections into two physical trajectories per recorded round."""

import argparse
import hashlib
import json
import math
import tempfile
import time
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import cv2

from bxjudge.manifest import probe
from bxjudge.reid_review import validate_submission
from bxjudge.two_bey_tracking import TwoBeyTracker, continuous_points, load_policy
from bxjudge.video import iter_frames


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def histogram(frame, box, bins):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = (
        max(0, math.floor(x1)),
        max(0, math.floor(y1)),
        min(w, math.ceil(x2)),
        min(h, math.ceil(y2)),
    )
    if min(x2 - x1, y2 - y1) < 2:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]).ravel()
    return (hist / hist.sum()).tolist()


def build(
    tracks_dir,
    rounds_path,
    reference_pointer,
    config_path,
    review_manifest,
    labels_path,
    annotations_path,
    output,
):
    started = time.perf_counter()
    (
        tracks_dir,
        rounds_path,
        reference_pointer,
        config_path,
        review_manifest,
        labels_path,
        annotations_path,
        output,
    ) = (
        Path(p).resolve()
        for p in (
            tracks_dir,
            rounds_path,
            reference_pointer,
            config_path,
            review_manifest,
            labels_path,
            annotations_path,
            output,
        )
    )
    if output.exists():
        raise ValueError("output must be a new directory")
    policy = load_policy(config_path)
    root = Path(__file__).resolve().parents[1]
    inputs = {
        str(p): sha(p)
        for p in (
            rounds_path,
            reference_pointer,
            config_path,
            Path(__file__).resolve(),
            root / "src/bxjudge/two_bey_tracking.py",
            root / "src/bxjudge/video.py",
            root / "src/bxjudge/reid_review.py",
            review_manifest,
            labels_path,
            annotations_path,
        )
    }
    rounds = read(rounds_path)
    pointer = read(reference_pointer)
    refpath = Path(pointer["reference"])
    if sha(refpath) != pointer["sha256"]:
        raise ValueError("human reference SHA mismatch")
    inputs[str(refpath)] = pointer["sha256"]
    references = read(refpath)
    manifest = read(review_manifest)
    validate_submission(read(labels_path), manifest, sha(review_manifest))
    annotations = read(annotations_path)
    if annotations["labels_sha256"] != sha(labels_path):
        raise ValueError("endpoint annotation revision mismatch")
    excluded = set()
    cases = {c["case_id"]: c for c in manifest["cases"]}
    coordinate_by_video = {
        c["video"]: tuple(c["coordinate_size"]) for c in manifest["cases"]
    }
    for a in annotations["endpoints"]:
        if (
            a["object_class"] != "launcher_tip"
            or a["source"] != "user_explicit_correction"
            or a["endpoint"] not in ("before", "after")
        ):
            raise ValueError("invalid explicit non-bey annotation")
        c = cases[a["case_id"]]
        tid = c["native_ids"][0 if a["endpoint"] == "before" else 1]
        excluded.add((c["video"], tid, c[a["endpoint"]]["frame"]))
    records = []
    for cache in sorted(tracks_dir.glob("*-tracks.json")):
        native = read(cache)
        source = Path(native["source"]).resolve()
        inputs[str(cache)] = sha(cache)
        inputs[str(source)] = sha(source)
        if inputs[str(source)] != native["source_sha256"]:
            raise ValueError("source video SHA mismatch")
        info = probe(source)
        if not math.isclose(info.fps, native["fps"], rel_tol=1e-7):
            raise ValueError("source FPS mismatch")
        coordinate_size = coordinate_by_video[source.name]
        if manifest["source_files_sha256"].get(str(source)) != native["source_sha256"]:
            raise ValueError(
                "source does not match the observation coordinate manifest"
            )
        if not math.isclose(
            coordinate_size[0] / coordinate_size[1],
            info.width / info.height,
            rel_tol=1e-5,
        ):
            raise ValueError("source coordinate aspect mismatch")
        selected = []
        for row in rounds:
            if row["video"] != source.name:
                continue
            first = math.ceil(row["t_launch"] * info.fps - 1e-6)
            last = math.floor(row["t_end"] * info.fps + 1e-6)
            selected.append(
                {
                    "round": row["round"],
                    "first_frame": first,
                    "last_frame": last,
                    "tracker": TwoBeyTracker(coordinate_size, info.fps, policy),
                    "observed": [],
                    "unassigned": [],
                }
            )
        if not selected:
            continue
        by_frame = defaultdict(list)
        manual_exclusions = []
        for track in native["tracks"]:
            if track["cls"] == "bey":
                for d in track["dets"]:
                    if (source.name, track["track_id"], d["frame"]) in excluded:
                        manual_exclusions.append(
                            {
                                **d,
                                "native_id": track["track_id"],
                                "reason": "human_confirmed_launcher_tip",
                            }
                        )
                        continue
                    by_frame[d["frame"]].append(
                        {
                            "native_id": track["track_id"],
                            "box": d["box"],
                            "score": d["score"],
                        }
                    )
        with closing(
            iter_frames(source, start=0, long_edge=max(coordinate_size), hwaccel="cpu")
        ) as frames:
            for f, image in frames:
                if f > max(r["last_frame"] for r in selected):
                    break
                if image.shape[1::-1] != coordinate_size:
                    raise ValueError("decoded coordinate size mismatch")
                rows = [
                    dict(
                        d,
                        appearance=histogram(
                            image, d["box"], policy["appearance_bins"]
                        ),
                    )
                    for d in by_frame.get(f, [])
                ]
                for r in selected:
                    if not r["first_frame"] <= f <= r["last_frame"]:
                        continue
                    result = r["tracker"].update(f, rows)
                    for key in ("observed", "unassigned"):
                        r[key].extend(
                            {k: v for k, v in d.items() if k != "appearance"}
                            for d in result[key]
                        )
        for r in selected:
            tracker = r.pop("tracker")
            r.update(
                video=source.name,
                source=str(source),
                source_sha256=native["source_sha256"],
                fps=info.fps,
                coordinate_size=list(coordinate_size),
                configured_entity_slots=2,
                observed_entity_count=tracker.entity_count,
                mode="offline_two_target_reconstruction",
                points=continuous_points(r["observed"]),
                native_owners={str(k): v for k, v in tracker.native_owners.items()},
                manual_exclusions=[
                    d
                    for d in manual_exclusions
                    if r["first_frame"] <= d["frame"] <= r["last_frame"]
                ],
            )
            relevant = [
                q
                for q in references
                if q["video"] == source.name
                and q["confirmed"]
                and not q["missed_or_bad_detection"]
                and r["first_frame"] <= q["frame"] <= r["last_frame"]
            ]
            matches = []
            for ref in relevant:
                for obj in ref["objects"]:
                    if obj["detector_class"] != "bey" or obj["side"] not in ("X", "B"):
                        continue
                    actual = [
                        d
                        for d in r["observed"]
                        if d["frame"] == ref["frame"]
                        and d["box"] == obj["box"]
                        and d["score"] == obj["score"]
                    ]
                    matches.append(
                        {
                            "frame": ref["frame"],
                            "expected_side": obj["side"],
                            "entity_id": actual[0]["entity_id"]
                            if len(actual) == 1
                            else None,
                            "reference_object": obj["index"],
                        }
                    )
            r["human_observation_matches"] = matches
            r["interpolated_frames"] = sum(
                p["kind"] == "interpolated" for p in r["points"]
            )
            records.append(r)
    for p, digest in inputs.items():
        if sha(p) != digest:
            raise ValueError("input changed during reconstruction")
    report = {
        "schema": "bxjudge.two-bey-reconstruction.v1",
        'origin':'offline_native_replay',
        "rounds": records,
        "input_sha256": inputs,
        "policy": policy,
        "applied_to_referee": False,
        "source_reference_revision": pointer["revision"],
        "limitations": [
            "two_slots_not_a_proof_of_correct_identity",
            "interpolation_not_measurement",
            "recorded_round_boundaries_not_revalidated",
            "launcher_and_parts_source_evidence_preserved",
        ],
        "reconstruction_wall_s": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".two-bey-", dir=output.parent) as temp:
        stage = Path(temp) / "reconstruction"
        stage.mkdir()
        (stage / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        stage.rename(output)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "tracks-dir",
        "rounds",
        "reference-pointer",
        "config",
        "review-manifest",
        "labels",
        "endpoint-annotations",
        "out",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    r = build(
        a.tracks_dir,
        a.rounds,
        a.reference_pointer,
        a.config,
        a.review_manifest,
        a.labels,
        a.endpoint_annotations,
        a.out,
    )
    print(
        json.dumps(
            [
                {
                    "video": q["video"],
                    "round": q["round"],
                    "entities": q["observed_entity_count"],
                    "observed": len(q["observed"]),
                    "unassigned": len(q["unassigned"]),
                    "interpolated": q["interpolated_frames"],
                    "human_matched": sum(
                        m["entity_id"] is not None
                        for m in q["human_observation_matches"]
                    ),
                    "human_total": len(q["human_observation_matches"]),
                    "side_entity_ids": {
                        side: sorted(
                            {
                                m["entity_id"]
                                for m in q["human_observation_matches"]
                                if m["expected_side"] == side
                                and m["entity_id"] is not None
                            }
                        )
                        for side in ("X", "B")
                    },
                }
                for q in r["rounds"]
            ],
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
