import pytest

from scripts.select_reid_challenges import choose_candidates, endpoint_pair


def candidate(category, frame, score, video="a.mp4"):
    return {
        "category": category,
        "center_frame": frame,
        "rank": score,
        "video": video,
        "fps": 10,
        "source_frames": [frame, frame + 1],
        "from_track": 1,
        "to_track": 1,
    }


def test_selection_does_not_reuse_reviewed_context_or_same_local_event():
    cs = [
        candidate("blur", 10, 0),
        candidate("blur", 50, 1),
        candidate("blur", 52, 2),
        candidate("blur", 80, 3),
        candidate("overlap", 110, 0),
        candidate("similar", 10, 0, "b.mp4"),
    ]
    chosen = choose_candidates(cs, {"a.mp4": [(0, 20)]}, per_category=2, context_s=1)
    assert [(c["category"], c["center_frame"]) for c in chosen] == [
        ("overlap", 110),
        ("similar", 10),
        ("blur", 50),
        ("blur", 80),
    ]
    assert all("relation" not in c and "side" not in c for c in chosen)


def test_missing_category_does_not_get_filled_from_another_category():
    cs = [candidate("blur", f, f) for f in (10, 40, 70)]
    assert len(choose_candidates(cs, {}, per_category=2, context_s=1)) == 2


def test_endpoint_pairs_require_real_observations_on_both_sides():
    ds = [{"frame": f, "cls": "bey"} for f in (0, 1, 3, 7, 9)]
    assert endpoint_pair(ds, 3, 10, 0.15, 0.5) == [1, 7]
    assert endpoint_pair(ds, 0, 10, 0.15, 0.5) is None
    assert endpoint_pair(ds, 9, 10, 0.15, 0.5) is None


def test_bad_candidate_time_is_rejected_not_sorted_into_a_review():
    c = candidate("blur", 10, 0)
    c["fps"] = float("nan")
    with pytest.raises(ValueError):
        choose_candidates([c], {}, per_category=2, context_s=1)


def test_real_synthetic_scan_is_bounded_and_never_prefills_labels(tmp_path):
    import json
    import shutil

    import yaml

    from scripts.select_reid_challenges import build
    from tests.test_reid_appearance import fixture, sha

    _, _, _, appearance = fixture(tmp_path)
    native = tmp_path / "synthetic-tracks.json"
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    shutil.move(native, tracks / native.name)
    source = tmp_path / "synthetic.mp4"
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps({"cases": [], "source_files_sha256": {str(source): sha(source)}})
    )
    ref = tmp_path / "reference.json"
    ref.write_text("[]")
    pointer = tmp_path / "pointer.json"
    pointer.write_text(json.dumps({"reference": str(ref), "sha256": sha(ref)}))
    policy = tmp_path / "selection.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "mode": "review_candidates",
                "frame_stride": 1,
                "per_category": 1,
                "context_s": 0.1,
                "endpoint_offset_s": 0.1,
                "endpoint_search_s": 0.3,
                "coordinate_size": [64, 48],
            }
        )
    )
    out = tmp_path / "out"
    report = build(tracks, prior, pointer, appearance, policy, out)
    assert report["selected_by_category"] == {"overlap": 0, "similar": 0, "blur": 1}
    assert report["applied"] is False
    assert report["population"]["measured_boxes"] == 8
    assert all("relation" not in c for c in report["remaining_pairs"])
    for c in report["remaining_pairs"]:
        assert c["from_track"] == c["to_track"]
        assert c["source_frames"][0] < c["center_frame"] < c["source_frames"][1]
    with pytest.raises(ValueError, match="new directory"):
        build(tracks, prior, pointer, appearance, policy, out)
    source.write_bytes(b"bad source")
    with pytest.raises(ValueError, match="SHA"):
        build(tracks, prior, pointer, appearance, policy, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_review_package_validation_rejects_modified_media(tmp_path):
    import json

    from scripts.build_reid_review import build
    from scripts.build_workbench import validate_review_set
    from tests.test_reid_appearance import fixture

    fixture(tmp_path)
    diagnostics = tmp_path / "pairs.json"
    diagnostics.write_text(
        json.dumps(
            {
                "remaining_pairs": [
                    {
                        "video": "synthetic.mp4",
                        "from_track": 1,
                        "to_track": 2,
                        "source_frames": [3, 5],
                        "gate_rejections": ["synthetic only"],
                    }
                ]
            }
        )
    )
    out = tmp_path / "review"
    build(diagnostics, tmp_path, out, (64, 48), 0.1)
    m, _ = validate_review_set(out)
    assert m["cases"][0]["scope_status"] == "unverified"
    (out / m["cases"][0]["before"]["image"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA"):
        validate_review_set(out)


def test_challenge_publication_rejects_changed_selection_and_its_sources(tmp_path):
    import hashlib
    import json

    from scripts.build_workbench import validate_challenge_selection

    source = tmp_path / "source.json"
    source.write_text("original")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": "bxjudge.reid-challenge-selection.v1",
                "applied": False,
                "input_sha256": {str(source): digest(source)},
            }
        )
    )
    manifest = {"diagnostics_sha256": digest(selection)}
    validate_challenge_selection(tmp_path, manifest)
    source.write_text("changed")
    with pytest.raises(ValueError, match="SHA"):
        validate_challenge_selection(tmp_path, manifest)
    selection.write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        validate_challenge_selection(tmp_path, manifest)
