import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from bxjudge.reid_appearance import (
    describe_crop,
    distances,
    load_policy,
    sample_observations,
)


def policy(tmp_path):
    data = {
        "purpose": "offline_diagnostic",
        "window_s": 0.3,
        "max_samples": 4,
        "min_crop_side_px": 8,
        "max_outside_fraction": 0.1,
        "color_bins": [8, 4, 4],
        "gradient_bins": 8,
    }
    p = tmp_path / "policy.yaml"
    p.write_text(yaml.safe_dump(data))
    return p, load_policy(p)


def test_color_distance_is_separate_from_texture_and_is_not_a_probability(tmp_path):
    _, p = policy(tmp_path)
    a = np.full((32, 32, 3), (0, 0, 255), np.uint8)
    b = np.full_like(a, (255, 0, 0))
    x = describe_crop(a, [0, 0, 32, 32], p)
    y = describe_crop(b, [0, 0, 32, 32], p)
    assert distances(x, x)["color_hellinger"] == pytest.approx(0)
    assert distances(x, y)["color_hellinger"] > 0.9
    assert distances(x, y)["gradient_hellinger"] is None
    assert x["quality"]["visibility"] == "unverified"
    assert "match" not in x and "confidence" not in x


@pytest.mark.parametrize(
    "box", ([2, 2, 2, 10], [0, 0, 4, 4], [float("nan"), 0, 10, 10], [-20, 0, 20, 20])
)
def test_bad_or_severely_truncated_crops_never_produce_descriptors(tmp_path, box):
    _, p = policy(tmp_path)
    assert describe_crop(np.zeros((32, 32, 3), np.uint8), box, p)["descriptor"] is None


def test_sampling_is_directional_bounded_and_does_not_fill_missing_frames(tmp_path):
    _, p = policy(tmp_path)
    track = {
        "cls": "bey",
        "dets": [
            {"frame": f, "cls": "bey", "box": [0, 0, 10, 10], "score": 0.5}
            for f in (0, 1, 3, 6, 9)
        ],
    }
    assert [d["frame"] for d in sample_observations(track, 3, "before", 10, p)] == [
        0,
        1,
        3,
    ]
    assert [d["frame"] for d in sample_observations(track, 3, "after", 10, p)] == [3, 6]
    assert track["dets"][0]["frame"] == 0
    with pytest.raises(ValueError):
        sample_observations(track, 2, "before", 10, p)


def test_policy_is_strict_and_cannot_enable_binding(tmp_path):
    path, _ = policy(tmp_path)
    data = yaml.safe_load(path.read_text())
    for key, value in [
        ("match_threshold", 0.9),
        ("max_samples", True),
        ("window_s", float("nan")),
        ("color_bins", [8, 4, 0]),
    ]:
        bad = dict(data, **{key: value})
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):
            load_policy(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path):
    from bxjudge.reid_review import LABELS_SCHEMA, MANIFEST_SCHEMA
    from bxjudge.review_store import save_immutable

    source = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=64x48:rate=10:duration=1",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
    )
    det = lambda f: {"frame": f, "cls": "bey", "box": [8, 8, 24, 24], "score": 0.8}
    native = {
        "source": str(source),
        "source_sha256": sha(source),
        "fps": 10,
        "tracks": [
            {"track_id": 1, "cls": "bey", "dets": [det(f) for f in range(4)]},
            {"track_id": 2, "cls": "bey", "dets": [det(f) for f in range(5, 9)]},
        ],
    }
    cache = tmp_path / "synthetic-tracks.json"
    cache.write_text(json.dumps(native))
    case = {
        "case_id": "case",
        "video": source.name,
        "source_sha256": sha(source),
        "fps": 10,
        "coordinate_size": [64, 48],
        "source_size": [64, 48],
        "native_ids": [1, 2],
        "before": {"frame": 3, "native_box": det(3)["box"]},
        "after": {"frame": 5, "native_box": det(5)["box"]},
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "cases": [case],
        "source_files_sha256": {str(source): sha(source)},
        "native_files_sha256": {str(cache): sha(cache)},
    }
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(manifest))
    label = {
        "case_id": "case",
        "reviewed": True,
        "note": "",
        "relation": "same",
        "scope": "same_round",
        **{
            k: "unreviewed"
            for k in (
                "side_before",
                "side_after",
                "visibility_before",
                "visibility_after",
                "collision",
            )
        },
    }
    payload = {
        "schema": LABELS_SCHEMA,
        "manifest_sha256": sha(m),
        "submitted_at": "2026-10-26T00:00:00Z",
        "supersedes": None,
        "cases": [label],
    }
    labels, _ = save_immutable(tmp_path / "labels", payload, prefix="reid")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"labels_sha256": sha(labels), "endpoints": []}))
    cfg, _ = policy(tmp_path)
    return m, labels, annotations, cfg


def test_real_decode_build_freezes_evidence_without_claiming_matching(tmp_path):
    from scripts.analyze_reid_appearance import build

    args = fixture(tmp_path)
    before = {str(p): sha(p) for p in args}
    out = tmp_path / "evidence"
    report = build(*args, out)
    assert report["population"] == {
        "total": 1,
        "same": 1,
        "different": 0,
        "uncertain": 0,
        "unreviewed": 0,
    }
    pair = report["cases"][0]
    assert pair["applied"] is False and pair["side_assignment"] == "unknown"
    assert pair["comparison"]["pair_count"] == 16
    assert [s["frame"] for s in pair["samples"]["before"]] == [0, 1, 2, 3]
    assert pair["comparison"]["color_hellinger"]["median"] == pytest.approx(0)
    assert "negative_pairs_missing" in report["limitations"]
    assert pair["decision"] == "diagnostic_only"
    assert all(s["identity_verified"] is False for s in pair["samples"]["before"])
    assert (out / "index.html").is_file()
    assert all(sha(p) == digest for p, digest in before.items())
    with pytest.raises(ValueError, match="new directory"):
        build(*args, out)


def test_endpoint_correction_preserves_unknown_and_excludes_only_that_pair(tmp_path):
    from bxjudge.review_store import save_immutable
    from scripts.analyze_reid_appearance import build

    m, labels, annotations, cfg = fixture(tmp_path)
    updated = json.loads(labels.read_bytes())
    updated["cases"][0]["relation"] = "uncertain"
    labels, _ = save_immutable(tmp_path / "labels", updated, prefix="reid")
    annotations.write_text(
        json.dumps(
            {
                "labels_sha256": sha(labels),
                "endpoints": [
                    {
                        "case_id": "case",
                        "endpoint": "before",
                        "object_class": "launcher_tip",
                        "source": "user_explicit_correction",
                    }
                ],
            }
        )
    )
    r = build(m, labels, annotations, cfg, tmp_path / "evidence")
    assert r["population"]["uncertain"] == 1 and r["population"]["total"] == 1
    pair = r["cases"][0]
    assert pair["human_review"]["relation"] == "uncertain"
    assert (
        pair["comparison"] is None and "human_non_bey_endpoint" in pair["reason_codes"]
    )
    assert pair["samples"]["before"] == []
    assert len(pair["samples"]["after"]) == 4


def test_changed_source_or_correction_version_never_publishes_output(tmp_path):
    from scripts.analyze_reid_appearance import build

    args = fixture(tmp_path)
    annotations = json.loads(args[2].read_bytes())
    annotations["labels_sha256"] = "f" * 64
    args[2].write_text(json.dumps(annotations))
    with pytest.raises(ValueError, match="annotation"):
        build(*args, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    annotations["labels_sha256"] = sha(args[1])
    args[2].write_text(json.dumps(annotations))
    source = Path(next(iter(json.loads(args[0].read_bytes())["source_files_sha256"])))
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        build(*args, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_decode_failure_closes_stream_and_does_not_publish(tmp_path, monkeypatch):
    from scripts import analyze_reid_appearance as module

    args = fixture(tmp_path)
    closed = []

    def broken(*args, **kwargs):
        try:
            yield 0, np.zeros((48, 64, 3), np.uint8)
            raise RuntimeError("synthetic decoder failure")
        finally:
            closed.append(True)

    monkeypatch.setattr(module, "iter_frames", broken)
    with pytest.raises(RuntimeError, match="decoder failure"):
        module.build(*args, tmp_path / "failed")
    assert closed == [True]
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".reid-appearance-*"))


def test_quality_rejection_never_becomes_a_human_non_bey_claim(tmp_path):
    from scripts.analyze_reid_appearance import build

    args = fixture(tmp_path)
    policy = yaml.safe_load(args[3].read_text())
    policy["min_crop_side_px"] = 32
    args[3].write_text(yaml.safe_dump(policy))
    out = tmp_path / "evidence"
    report = build(*args, out)
    assert report["population"]["same"] == 1
    assert "human_non_bey_endpoint" not in report["cases"][0]["reason_codes"]
    text = (out / "index.html").read_text()
    assert "有人工作證為非陀螺" not in text
    assert "small_crop" in text


def test_annotation_of_one_endpoint_does_not_label_the_other_bad_crop(tmp_path):
    from scripts.analyze_reid_appearance import build

    m, labels, annotations, cfg = fixture(tmp_path)
    annotations.write_text(
        json.dumps(
            {
                "labels_sha256": sha(labels),
                "endpoints": [
                    {
                        "case_id": "case",
                        "endpoint": "before",
                        "object_class": "launcher_tip",
                        "source": "user_explicit_correction",
                    }
                ],
            }
        )
    )
    data = yaml.safe_load(cfg.read_text())
    data["min_crop_side_px"] = 32
    cfg.write_text(yaml.safe_dump(data))
    out = tmp_path / "evidence"
    build(m, labels, annotations, cfg, out)
    text = (out / "index.html").read_text()
    assert text.count("有人工作證為非陀螺") == 1
    assert "small_crop" in text


@pytest.mark.parametrize(
    "copy_name", ["human-review.json", "endpoint-annotations.json"]
)
def test_publishing_rejects_changed_frozen_human_copies(tmp_path, copy_name):
    from scripts.analyze_reid_appearance import build
    from scripts.build_workbench import validate_appearance_set

    args = fixture(tmp_path)
    out = tmp_path / "evidence"
    build(*args, out)
    validate_appearance_set(out, args[0].parent, sha(args[0]))
    (out / copy_name).write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="frozen"):
        validate_appearance_set(out, args[0].parent, sha(args[0]))
