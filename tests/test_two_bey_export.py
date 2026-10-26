import json
from pathlib import Path

import pytest

from scripts.export_two_bey_results import build
from scripts.reconstruct_two_beys import sha
from tests.test_reid_appearance import fixture


def test_export_reads_formal_saved_geometry_without_rerunning_association(tmp_path):
    fixture(tmp_path)
    source = tmp_path / "synthetic.mp4"
    results = tmp_path / "results"
    (results / "geometry").mkdir(parents=True)
    observations = [
        {
            "frame": f,
            "entity_id": 1,
            "native_id": 7,
            "box": [4 + f, 4, 14 + f, 14],
            "score": 0.8,
        }
        for f in (0, 1, 3)
    ]
    record = {
        "schema": 1,
        "coordinate_mapping": "reference_to_source",
        "source_sha256": sha(source),
        "source": str(source),
        "calibration_id": "test",
        "reference_size": [64, 48],
        "samples": [
            {"frame": f, "alignment": None, "segment": 0, "status": "lost"}
            for f in range(4)
        ],
        "two_bey_tracking": {
            "policy": {},
            "raw_source_tracks": [],
            "rounds": {
                "1": {
                    "first_frame": 0,
                    "last_frame": 3,
                    "configured_slots": 2,
                    "observed_entities": 1,
                    "observations": observations,
                    "unassigned": [],
                    "members": {"10": [7]},
                    "interpolation_used_for_judging": False,
                }
            },
        },
    }
    p = results / "geometry/test.json"
    p.write_text(json.dumps(record))
    row = {
        "video": source.name,
        "round": 1,
        "calibration_id": "test",
        "geometry_evidence": {
            "path": "geometry/test.json",
            "sha256": sha(p),
            "coordinates": "reference_pixels",
        },
    }
    (results / "rounds.json").write_text(json.dumps([row]))
    ref = tmp_path / "ref.json"
    ref.write_text("[]")
    pointer = tmp_path / "pointer.json"
    pointer.write_text(
        json.dumps({"reference": str(ref), "sha256": sha(ref), "revision": "a" * 64})
    )
    report = build(results, pointer, tmp_path / "export")
    r = report["rounds"][0]
    assert r["observed"] == observations
    assert r["configured_entity_slots"] == 2 and r["observed_entity_count"] == 1
    assert [p["kind"] for p in r["points"]] == [
        "observed",
        "observed",
        "interpolated",
        "observed",
    ]
    assert report["origin"] == "formal_saved_two_bey_geometry"
    assert r["native_owners"] == {"7": 1}
    from scripts.render_two_beys import build as render

    preview = render(
        tmp_path / "export/report.json",
        Path("config/trajectory_video.yaml"),
        tmp_path / "preview",
    )
    assert preview["origin"] == "formal_saved_two_bey_geometry"
    assert preview["formal_core_applied_two_bey_tracking"] is True
    page = (tmp_path / "preview/index.html").read_text()
    assert "正式流程保存" in page and "<details>" in page
    assert "尚未取代正式判決流程" not in page
    assert sha(results / "geometry/test.json") == row["geometry_evidence"]["sha256"]
    with pytest.raises(ValueError, match="new directory"):
        build(results, pointer, tmp_path / "export")
    p.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        build(results, pointer, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()
