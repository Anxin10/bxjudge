import numpy as np
import pytest

from bxjudge.autocalib import mean_boundary_error, warp_calibration
from bxjudge.dynamic_calibration import (
    DynamicCalibration,
    load_dynamic_policy,
    normalize_tracks,
)
from bxjudge.trajectory_video import load_video_policy
from tests.helpers import make_track, stadium_scene


def tracker(c, ref):
    policy = load_dynamic_policy()
    return DynamicCalibration(c, ref, 30, load_video_policy()["stadium_follow"], policy)


def test_persistent_move_recovers_after_confirmed_relocation():
    ref, c = stadium_scene(size=(640, 360))
    moved, truth = stadium_scene(size=(640, 360), shift=(45, -4))
    obj = tracker(c, ref)
    first = obj.update(0, ref)
    assert first.alignment is not None
    samples = [obj.update(f, moved) for f in range(1, 65)]
    assert samples[0].alignment is None
    assert any(s.status == "relocated" for s in samples)
    good = [s for s in samples if s.alignment is not None][-1]
    assert (
        mean_boundary_error(
            warp_calibration(c, good.alignment, c.frame_size, c.name), truth, 640
        )
        < 2
    )
    assert len(obj.segments) == 2
    assert obj.record()["reference_size"] == [640, 360]


def test_single_occlusion_does_not_reuse_matrix_or_create_segment():
    ref, c = stadium_scene(size=(640, 360))
    obj = tracker(c, ref)
    assert obj.update(0, ref).alignment is not None
    assert obj.update(1, np.full_like(ref, 128)).alignment is None
    assert len(obj.segments) == 1
    assert obj.update(2, ref).alignment is None
    assert obj.update(3, ref).alignment is None
    assert obj.update(4, ref).alignment is not None


def test_frame_order_and_dimensions_are_explicit():
    ref, c = stadium_scene(size=(640, 360))
    obj = tracker(c, ref)
    obj.update(1, ref)
    with pytest.raises(ValueError):
        obj.update(1, ref)
    with pytest.raises(ValueError):
        obj.update(2, ref[:100])


def test_inverse_geometry_normalizes_observations_but_does_not_invent_missing():
    ref, c = stadium_scene(size=(640, 360))
    obj = tracker(c, ref)
    moved, _ = stadium_scene(size=(640, 360), shift=(12, -8))
    sample = obj.update(0, moved)
    bad = obj.update(1, np.full_like(ref, 128))
    track = make_track(7, "bey", {0: (312, 172), 1: (313, 172)})
    result = normalize_tracks([track], {0: sample, 1: bad})
    assert len(result) == 1 and result[0].track_id == 7
    assert list(result[0].centers()) == [0]
    assert np.linalg.norm(np.array(result[0].center_at(0)) - [300, 180]) < 1
    assert track.center_at(0) == (312, 172)
    assert result[0].dets[0].score == track.dets[0].score


def test_relocation_never_accepts_flat_or_out_of_global_bounds():
    ref, c = stadium_scene(size=(640, 360))
    obj = tracker(c, ref)
    obj.update(0, ref)
    flat = np.full_like(ref, 128)
    assert all(obj.update(f, flat).alignment is None for f in range(1, 61))
    huge, _ = stadium_scene(size=(640, 360), shift=(180, 0))
    assert all(obj.update(f, huge).alignment is None for f in range(61, 121))
    assert len(obj.segments) == 1


def test_normalized_landing_and_finish_match_still_reference(th):
    from bxjudge.autocalib import Alignment
    from bxjudge.dynamic_calibration import GeometrySample
    from bxjudge.finish import detect_finish
    from bxjudge.landing import find_landing_frame
    from tests.helpers import scene

    bey, parts, stops, c = scene(b_path_into="over_left", b_enter_at=100)
    samples = {}
    raw = []
    for t in bey.values():
        centres = {}
        for f, p in t.centers().items():
            dx = 20 * f / 300
            dy = 4 * f / 300
            centres[f] = (p[0] + dx, p[1] + dy)
            a = Alignment(
                ((1.0, 0.0, dx), (0.0, 1.0, dy), (0.0, 0.0, 1.0)),
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
            )
            samples[f] = GeometrySample(f, a, 0, "tracking")
        raw.append(make_track(t.track_id, t.cls, centres))
    normal = normalize_tracks(raw, samples)
    mapped = dict(zip(bey, normal))
    baseline = detect_finish(bey, parts, stops, c, th, 60)
    moved = detect_finish(mapped, parts, stops, c, th, 60)
    assert (moved.finish, moved.loser, moved.frame) == (
        baseline.finish,
        baseline.loser,
        baseline.frame,
    )
    assert find_landing_frame(normal[0], c, th) == find_landing_frame(bey["X"], c, th)


def test_recorded_evidence_hash_and_source_are_required(tmp_path):
    from bxjudge.dynamic_calibration import read_geometry, save_geometry

    ref, c = stadium_scene(size=(640, 360))
    obj = tracker(c, ref)
    obj.update(0, ref)
    data = {**obj.record(), "calibration_id": "abc", "source_sha256": "source"}
    evidence = save_geometry(tmp_path, "v.mp4", data)
    row = {"geometry_evidence": evidence, "calibration_id": "abc"}
    record = read_geometry(tmp_path, row, "source")
    assert 0 in record["samples"]
    with pytest.raises(ValueError, match="source"):
        read_geometry(tmp_path, row, "wrong")
    path = tmp_path / evidence["path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        read_geometry(tmp_path, row, "source")
