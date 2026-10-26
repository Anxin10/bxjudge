"""Export the formal two-top observations for playback, without rerunning tracking."""

import argparse
import json
import tempfile
from pathlib import Path

from bxjudge.dynamic_calibration import read_geometry
from bxjudge.manifest import probe
from bxjudge.two_bey_tracking import continuous_points
from scripts.reconstruct_two_beys import read, sha


def build(results, reference_pointer, output):
    results, reference_pointer, output = (
        Path(p).resolve() for p in (results, reference_pointer, output)
    )
    if output.exists():
        raise ValueError("output must be a new directory")
    rows_path = results / "rounds.json"
    pointer = read(reference_pointer)
    reference = Path(pointer["reference"]).resolve()
    if sha(reference) != pointer["sha256"]:
        raise ValueError("manual reference SHA mismatch")
    inputs = {
        str(p): sha(p)
        for p in (
            rows_path,
            reference_pointer,
            reference,
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1] / "src/bxjudge/two_bey_tracking.py",
            Path(__file__).resolve().parents[1] / "src/bxjudge/dynamic_calibration.py",
        )
    }
    human = read(reference)
    records = []
    policy = None
    for row in read(rows_path):
        if row.get("geometry_evidence") is None:
            raise ValueError("formal two-bey row is missing saved geometry")
        path = (results / row["geometry_evidence"]["path"]).resolve()
        if (
            not path.is_relative_to(results / "geometry")
            or sha(path) != row["geometry_evidence"]["sha256"]
        ):
            raise ValueError("formal geometry path or SHA mismatch")
        raw = read(path)
        source = Path(raw["source"]).resolve()
        digest = sha(source)
        geometry = read_geometry(results, row, digest)
        if "two_bey_tracking" not in geometry:
            raise ValueError("formal analysis did not use two-bey tracking")
        inputs[str(path.resolve())] = sha(path)
        inputs[str(source)] = digest
        phase = geometry["two_bey_tracking"]
        if policy is not None and policy != phase["policy"]:
            raise ValueError("mixed two-bey policies in one result set")
        policy = phase["policy"]
        stored = phase["rounds"].get(str(row["round"]))
        if stored is None:
            if "no_round" in row.get("review_reasons", []):
                continue
            raise ValueError("round not in saved physical tracking evidence")
        observations = stored["observations"]
        info = probe(source)
        entities = sorted({d["entity_id"] for d in observations})
        if len(entities) > 2 or stored["configured_slots"] != 2:
            raise ValueError("formal geometry contains more than two physical tops")
        if (
            stored["segment_status"] == "paired_observations" and len(entities) != 2
        ) or (stored["segment_status"] == "pair_not_confirmed" and entities):
            raise ValueError(
                "formal paired-segment status contradicts physical trajectories"
            )
        matches = []
        for r in human:
            if (
                r["video"] != row["video"]
                or not r["confirmed"]
                or r["missed_or_bad_detection"]
                or not stored["first_frame"] <= r["frame"] <= stored["last_frame"]
            ):
                continue
            if (
                r["source_sha256"] != digest
                or r["coordinate_size"] != geometry["reference_size"]
            ):
                raise ValueError("human observation coordinate/source mismatch")
            for obj in r["objects"]:
                if obj["detector_class"] != "bey" or obj["side"] not in ("X", "B"):
                    continue
                actual = [
                    d
                    for d in observations
                    if d["frame"] == r["frame"]
                    and d["box"] == obj["box"]
                    and d["score"] == obj["score"]
                ]
                matches.append(
                    {
                        "frame": r["frame"],
                        "expected_side": obj["side"],
                        "entity_id": actual[0]["entity_id"]
                        if len(actual) == 1
                        else None,
                        "reference_object": obj["index"],
                    }
                )
        owners = {}
        for ids in stored["members"].values():
            assigned = {d["entity_id"] for d in observations if d["native_id"] in ids}
            if len(assigned) == 1:
                for tid in ids:
                    owners[str(tid)] = next(iter(assigned))
        points = continuous_points(observations)
        records.append(
            {
                "video": row["video"],
                "round": row["round"],
                "first_frame": stored["first_frame"],
                "last_frame": stored["last_frame"],
                "source": str(source),
                "source_sha256": digest,
                "fps": info.fps,
                "coordinate_size": geometry["reference_size"],
                "configured_entity_slots": 2,
                "observed_entity_count": len(entities),
                "segment_status": stored["segment_status"],
                "tentative_entity_count": stored["tentative_entities"],
                "hypothesis_observations": stored["hypothesis_observations"],
                "observed": observations,
                "unassigned": stored["unassigned"],
                "points": points,
                "native_owners": owners,
                "human_observation_matches": matches,
                "interpolated_frames": sum(p["kind"] == "interpolated" for p in points),
                "manual_exclusions": [],
                "formal_geometry": row["geometry_evidence"],
                "native_launcher_tracks": [
                    t for t in phase["raw_source_tracks"] if t["cls"] == "launcher"
                ],
            }
        )
    for p, digest in inputs.items():
        if sha(p) != digest:
            raise ValueError("source SHA changed during export")
    report = {
        "schema": "bxjudge.two-bey-reconstruction.v1",
        "origin": "formal_saved_two_bey_geometry",
        "rounds": records,
        "input_sha256": inputs,
        "policy": policy,
        "applied_to_referee": False,
        "formal_core_applied_two_bey_tracking": True,
        "source_reference_revision": pointer["revision"],
        "limitations": [
            "human_labels_used_only_for_regression_and_legend",
            "interpolation_not_adjudication",
            "not_independent_accuracy_acceptance",
            "singleton_segment_not_a_confirmed_two_top_battle",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".two-export-", dir=output.parent) as temp:
        stage = Path(temp) / "export"
        stage.mkdir()
        (stage / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        stage.rename(output)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "reference-pointer", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    report = build(a.results, a.reference_pointer, a.out)
    print(json.dumps({"rounds": len(report["rounds"]), "output": str(a.out)}))


if __name__ == "__main__":
    main()
