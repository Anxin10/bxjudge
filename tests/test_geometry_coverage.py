import hashlib
import json
from types import SimpleNamespace

import pytest

from bxjudge import geometry_coverage


def sample(frame, valid, status):
    return {
        "frame": frame,
        "alignment": {"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
        if valid
        else None,
        "status": status,
    }


def test_missing_frames_and_confirmations_never_count_as_valid_geometry():
    report = geometry_coverage.measure_window(
        [
            sample(1, True, "tracking"),
            sample(2, False, "lost"),
            sample(4, False, "confirming"),
            sample(5, True, "relocated"),
        ],
        1,
        5,
    )
    assert report["total_frames"] == 5 and report["valid_frames"] == 2
    assert report["coverage"] == 0.4
    assert report["missing_reasons"] == {
        "alignment_rejected_cause_unrecorded": 1,
        "sample_absent": 1,
        "confirming": 1,
    }
    assert report["missing_intervals"] == [
        {
            "first_frame": 2,
            "last_frame": 4,
            "frames": 3,
            "reason_counts": {
                "alignment_rejected_cause_unrecorded": 1,
                "sample_absent": 1,
                "confirming": 1,
            },
        }
    ]


def test_frame_zero_and_inclusive_single_frame_window_are_measured():
    report = geometry_coverage.measure_window([sample(0, True, "tracking")], 0, 0)
    assert report["coverage"] == 1 and report["total_frames"] == 1
    assert report["missing_intervals"] == []


def test_candidate_union_does_not_double_count_overlapping_round_windows():
    report = geometry_coverage.measure_windows(
        [sample(f, f == 2, "tracking" if f == 2 else "lost") for f in range(5)],
        [(0, 2), (2, 4)],
    )
    assert report["total_frames"] == 5 and report["valid_frames"] == 1
    assert report["coverage"] == 0.2


@pytest.mark.parametrize(
    "samples,lo,hi",
    [
        ([sample(1, True, "tracking"), sample(1, False, "lost")], 0, 3),
        ([sample(2, True, "tracking"), sample(1, False, "lost")], 0, 3),
        ([sample(True, True, "tracking")], 0, 3),
        ([sample(1, True, "tracking")], 3, 2),
    ],
)
def test_corrupt_sample_order_or_frame_range_is_rejected(samples, lo, hi):
    with pytest.raises(ValueError):
        geometry_coverage.measure_window(samples, lo, hi)


def test_empty_population_has_unknown_coverage_not_perfect_coverage():
    assert geometry_coverage.measure_windows([], [])["coverage"] is None


def test_saved_detailed_reason_is_used_without_guessing_occlusion():
    values = [sample(1, False, "lost") | {"rejection_reason": "bounds"}]
    report = geometry_coverage.measure_window(values, 1, 1)
    assert report["missing_reasons"] == {"bounds": 1}


@pytest.mark.parametrize("frames", [[1], []])
def test_report_preserves_missing_leading_and_trailing_geometry(
    tmp_path, monkeypatch, frames
):
    from scripts import report_geometry_coverage as runner

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"synthetic source; never decoded")
    (tmp_path / "geometry").mkdir()
    (tmp_path / "calibrations").mkdir()
    for suffix in (".json", ".png"):
        (tmp_path / "calibrations" / ("manual" + suffix)).write_bytes(b"test snapshot")
    raw = {
        "schema": 1,
        "coordinate_mapping": "reference_to_source",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "calibration_id": "manual",
        "reference_size": [100, 100],
        "samples": [sample(f, True, "tracking") for f in frames],
        "two_bey_tracking": {
            "rounds": {
                "1": {
                    "first_frame": 0,
                    "last_frame": 2,
                    "segment_status": "paired_observations",
                }
            }
        },
    }
    content = json.dumps(raw).encode()
    (tmp_path / "geometry/g.json").write_bytes(content)
    rows = [
        {
            "video": "clip.mp4",
            "round": 1,
            "calibration_id": "manual",
            "geometry_evidence": {
                "path": "geometry/g.json",
                "sha256": hashlib.sha256(content).hexdigest(),
            },
        }
    ]
    (tmp_path / "rounds.json").write_text(json.dumps(rows))
    (tmp_path / "videos.json").write_text(
        json.dumps({"clip.mp4": {"source_video": str(source)}})
    )
    monkeypatch.setattr(
        runner,
        "load_snapshot",
        lambda *a, **kw: SimpleNamespace(
            id="manual", kind="manual", calibration=SimpleNamespace(name="scene")
        ),
    )
    report = runner.coverage_report(
        tmp_path,
        {"target_coverage": 0.9, "worst_clips": 10, "failure_probes_per_clip": 3},
    )
    assert report["rounds"][0]["total_frames"] == 3
    assert report["rounds"][0]["valid_frames"] == len(frames)
    assert report["rounds"][0]["missing_reasons"] == {"sample_absent": 3 - len(frames)}
