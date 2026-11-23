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


# ---- keypoint relocation (plan 2026-11-12-dynamic-calibration, Task 4) -----------------


class FakeLocator:
    """``KeypointLocator`` stand-in: every call gives ``h`` (calibration px -> frame px) or,
    with ``h=None``, nothing and ``reason``."""

    def __init__(self, c, h=None, reason="no_keypoints"):
        self.c, self.h, self.reason = c, h, reason
        self.calls = 0
        self.last_reason = None

    def locate(self, image):
        from bxjudge.stadium_locator import LocatedStadium, alignment_from_homography

        self.calls += 1
        if self.h is None:
            self.last_reason = self.reason
            return None
        self.last_reason = None
        return LocatedStadium(self.h, alignment_from_homography(self.c, self.h, 0.9), None)


def _translation(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _keypoint_tracker(c, ref, locator):
    return DynamicCalibration(c, ref, 30, load_video_policy()["stadium_follow"],
                              load_dynamic_policy(), keypoint_locator=locator)


def test_keypoints_relocate_when_the_reference_relocator_cannot():
    ref, c = stadium_scene(size=(640, 360))
    moved, truth = stadium_scene(size=(640, 360), shift=(45, -4))
    fake = FakeLocator(c, _translation(45, -4))
    obj = _keypoint_tracker(c, ref, fake)
    obj.relocator.align = lambda image: None  # the rim match against the reference fails
    assert obj.update(0, ref).alignment is not None
    samples = [obj.update(f, moved) for f in range(1, 65)]
    assert fake.calls >= 2
    kinds = [s["kind"] for s in obj.segments]
    assert kinds == ["recorded_snapshot", "relocated_keypoints"]
    start = obj.segments[-1]["start_frame"]
    # Nothing before the confirming frame is backfilled.
    assert all(obj.samples[f].alignment is None for f in range(1, start))
    assert obj.samples[start].status == "relocated"
    good = [s for s in samples if s.alignment is not None][-1]
    assert mean_boundary_error(warp_calibration(c, good.alignment, c.frame_size, c.name),
                               truth, 640) < 2
    tried = [d["relocation"] for d in obj.diagnostics.values() if d["relocation"]]
    assert tried and all(r["keypoints"] == "located" for r in tried)


def test_keypoint_relocation_obeys_the_global_bounds():
    ref, c = stadium_scene(size=(640, 360))
    moved, _ = stadium_scene(size=(640, 360), shift=(45, -4))
    far = FakeLocator(c, _translation(300, 0))  # shift 0.47 of the width > max_shift_frac
    obj = _keypoint_tracker(c, ref, far)
    obj.relocator.align = lambda image: None
    obj.update(0, ref)
    assert all(obj.update(f, moved).alignment is None for f in range(1, 65))
    assert len(obj.segments) == 1 and far.calls >= 2
    tried = [d["relocation"]["keypoints"] for d in obj.diagnostics.values() if d["relocation"]]
    assert set(tried) == {"keypoints_out_of_bounds"}


def test_keypoint_failure_is_recorded_and_tracking_goes_on():
    ref, c = stadium_scene(size=(640, 360))
    flat = np.full_like(ref, 128)
    broken = FakeLocator(c, None, "keypoint_error: RuntimeError: CUDA out of memory")
    obj = _keypoint_tracker(c, ref, broken)
    obj.update(0, ref)
    assert all(obj.update(f, flat).alignment is None for f in range(1, 40))
    assert obj.update(40, ref).alignment is None  # recovery still needs resume_frames
    assert [obj.update(f, ref).alignment is not None for f in (41, 42)][-1]
    tried = [d["relocation"]["keypoints"] for d in obj.diagnostics.values() if d["relocation"]]
    assert tried and set(tried) == {"keypoint_error: RuntimeError: CUDA out of memory"}


def test_without_a_locator_behaviour_is_unchanged():
    ref, c = stadium_scene(size=(640, 360))
    moved, _ = stadium_scene(size=(640, 360), shift=(45, -4))
    plain, none = tracker(c, ref), _keypoint_tracker(c, ref, None)
    for f, img in enumerate([ref] + [moved] * 64):
        plain.update(f, img)
        none.update(f, img)
    assert plain.record() == none.record()
    assert [s["kind"] for s in plain.segments] == ["recorded_snapshot", "relocated"]
    none.finish()  # a no-op in fixed mode
    assert plain.record() == none.record()


def test_keypoint_cooldown_preserves_pending_at_120fps():
    ref, c = stadium_scene(size=(640, 360))
    moved, _ = stadium_scene(size=(640, 360), shift=(45, -4))
    fake = FakeLocator(c, _translation(45, -4))
    policy = load_dynamic_policy()
    policy["reject_s"] = 0.0625
    policy["retry_s"] = 0.0625
    obj = DynamicCalibration(
        c, ref, 120.0, load_video_policy()["stadium_follow"], policy, keypoint_locator=fake
    )
    obj.relocator.align = lambda image: None
    assert obj.update(0, ref).alignment is not None
    samples = [obj.update(f, moved) for f in range(1, 100)]
    assert fake.calls == 2
    assert [s["kind"] for s in obj.segments] == ["recorded_snapshot", "relocated_keypoints"]
    assert any(s.status == "relocated" for s in samples)


def test_fixed_camera_reuses_alignment_between_checks_and_survives_brief_blur():
    ref, c = stadium_scene(size=(640, 360))
    obj = DynamicCalibration(
        c, ref, 120.0, load_video_policy()["stadium_follow"], load_dynamic_policy(),
        fixed_camera=True,
    )
    assert obj.check_stride == 8
    assert obj.update(0, ref).status == "tracking"
    # Intermediate frames reuse the verified alignment without calling align().
    calls = 0
    orig_align = obj.anchor.align
    def counting_align(img):
        nonlocal calls
        calls += 1
        return orig_align(img)
    obj.anchor.align = counting_align
    for f in range(1, 8):
        s = obj.update(f, ref)
        assert s.status == "tracking" and s.alignment is not None
    assert calls == 0
    # A transient soft-blur failure on a check frame stays tracking within grace window.
    obj.anchor.align = lambda img: (
        obj.anchor.last_diagnostic.update({"reason": "distinct_matches_insufficient"}) or None
    )
    s8 = obj.update(8, ref)
    assert s8.status == "tracking" and s8.alignment is not None
