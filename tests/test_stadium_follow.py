"""Bounded reference-anchored stadium following; bad frames never reuse old geometry."""

import cv2
import numpy as np
import pytest
import yaml

from bxjudge.autocalib import mean_boundary_error, warp_calibration
from bxjudge.stadium_follow import StadiumFollower, validate_follow_policy
from tests.helpers import stadium_scene


def policy():
    from bxjudge.trajectory_video import DEFAULT_VIDEO_CONFIG

    return yaml.safe_load(DEFAULT_VIDEO_CONFIG.read_text())["stadium_follow"]


@pytest.mark.parametrize(
    "shift,rotation,scale",
    [((0, 0), 0, 1), ((9, -5), 0, 1), ((4, 2), 1.5, 1), ((0, 0), 0, 1.02)],
)
def test_follows_small_rigid_motion_from_recorded_reference(shift, rotation, scale):
    ref, c = stadium_scene(size=(640, 360))
    frame, truth = stadium_scene(
        size=(640, 360), shift=shift, rotation_deg=rotation, scale=scale
    )
    follower = StadiumFollower(c, ref, policy())
    result = follower.align(frame)
    assert result is not None
    actual = warp_calibration(c, result, c.frame_size, c.name)
    assert mean_boundary_error(actual, truth, 640) < 1.5


def test_floor_motion_does_not_drag_the_stadium():
    ref, c = stadium_scene(size=(640, 360))
    frame = ref.copy()
    cv2.circle(frame, (290, 240), 42, (0, 0, 255), -1)
    result = StadiumFollower(c, ref, policy()).align(frame)
    assert result is not None
    assert result.shift_frac < 0.002


def test_bad_frame_hides_alignment_then_recovers_from_original_reference():
    ref, c = stadium_scene(size=(640, 360))
    follower = StadiumFollower(c, ref, policy())
    moved, _ = stadium_scene(size=(640, 360), shift=(8, -3))
    assert follower.align(moved) is not None
    assert follower.align(np.full_like(ref, 128)) is None
    result = follower.align(ref)
    assert result is not None
    assert result.shift_frac < 0.002


@pytest.mark.parametrize(
    "shift,rotation,scale", [((90, 0), 0, 1), ((0, 0), 10, 1), ((0, 0), 0, 1.12)]
)
def test_large_motion_is_rejected(shift, rotation, scale):
    ref, c = stadium_scene(size=(640, 360))
    moved, _ = stadium_scene(
        size=(640, 360), shift=shift, rotation_deg=rotation, scale=scale
    )
    assert StadiumFollower(c, ref, policy()).align(moved) is None


def test_heavy_rim_occlusion_is_rejected():
    ref, c = stadium_scene(size=(640, 360))
    frame = ref.copy()
    frame[:250] = 90
    assert StadiumFollower(c, ref, policy()).align(frame) is None


def test_output_matrix_is_in_full_calibration_pixels():
    ref, c = stadium_scene(size=(1280, 720))
    moved, truth = stadium_scene(size=(1280, 720), shift=(18, -10), rotation_deg=1)
    result = StadiumFollower(c, ref, policy()).align(moved)
    assert result is not None
    assert (
        mean_boundary_error(
            warp_calibration(c, result, c.frame_size, c.name), truth, 1280
        )
        < 2
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("min_inliers", True),
        ("max_rotation_deg", float("nan")),
        ("ratio_test", 1.0),
        ("min_quadrants", 5),
        ("min_scale", 1.1),
    ],
)
def test_invalid_policy_is_rejected(key, value):
    cfg = policy()
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_follow_policy(cfg)


def test_missing_policy_keys_are_not_filled_in():
    cfg = policy()
    del cfg["min_inliers"]
    with pytest.raises(ValueError):
        validate_follow_policy(cfg)


def test_small_camera_tilt_can_follow_without_uniform_scale_assumption():
    ref, c = stadium_scene(size=(640, 360))
    matrix = np.array([[1.01, -0.018, -3.0], [0.004, 1.005, -1.0]], np.float32)
    frame = cv2.warpAffine(ref, matrix, (640, 360))
    result = StadiumFollower(c, ref, policy()).align(frame)
    assert result is not None
    predicted = np.array(result.homography)[:2]
    probes = np.array([[150, 70, 1], [500, 90, 1], [180, 310, 1], [480, 320, 1]])
    assert (
        np.max(np.linalg.norm(probes @ predicted.T - probes @ matrix.T, axis=1)) < 1.5
    )


def test_one_surviving_rim_patch_is_not_enough_for_global_alignment():
    ref, c = stadium_scene(size=(640, 360))
    frame = np.full_like(ref, 90)
    frame[:180, :320] = ref[:180, :320]
    assert StadiumFollower(c, ref, policy()).align(frame) is None


def test_anisotropic_deformation_is_rejected_even_with_near_unit_area():
    ref, c = stadium_scene(size=(640, 360))
    matrix = np.array([[1.10, 0, -32], [0, 1 / 1.10, 16]], np.float32)
    frame = cv2.warpAffine(ref, matrix, (640, 360))
    assert StadiumFollower(c, ref, policy()).align(frame) is None


def test_textureless_reference_does_not_claim_identity_alignment():
    ref, c = stadium_scene(size=(640, 360))
    flat = np.full_like(ref, 128)
    assert StadiumFollower(c, flat, policy()).align(flat) is None


def test_motion_limit_uses_stadium_centre_in_an_off_centre_view():
    from bxjudge.autocalib import _stadium_centroid

    ref, c = stadium_scene(size=(640, 360), shift=(40, 0))
    matrix = cv2.getRotationMatrix2D((320, 180), -2.0, 1.0)
    matrix[1, 2] += 11
    frame = cv2.warpAffine(ref, matrix, (640, 360))
    centre = np.array([*_stadium_centroid(c), 1.0])
    actual_shift = np.linalg.norm(matrix @ centre - centre[:2]) / 640
    image_shift = 11 / 640
    assert actual_shift > image_shift
    cfg = policy()
    cfg["max_shift_frac"] = float((actual_shift + image_shift) / 2)
    assert StadiumFollower(c, ref, cfg).align(frame) is None
