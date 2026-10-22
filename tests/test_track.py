import numpy as np
import pytest

from bxjudge.types import Detection


def blank():
    return np.zeros((60, 160, 3), dtype=np.uint8)


def test_track_frames_keeps_identity():
    from bxjudge.track import track_frames

    def detector(frame, idx):
        return [Detection(idx, "bey", (10 + 5 * idx, 10, 30 + 5 * idx, 30), 0.9)]

    tracks = track_frames(((i, blank()) for i in range(20)), detector, 60)
    assert len(tracks) == 1
    assert [d.frame for d in tracks[0].dets] == list(range(20))


def test_tracks_split_by_class_and_ids_are_unique():
    from bxjudge.track import track_frames

    def detector(frame, idx):
        return [
            Detection(idx, cls, (10, 10, 40, 30), 0.9)
            for cls in ("bey", "bey_part", "launcher")
        ]

    tracks = track_frames(((i, blank()) for i in range(5)), detector, 60)
    assert sorted(t.cls for t in tracks) == ["bey", "bey_part", "launcher"]
    assert len({t.track_id for t in tracks}) == 3


def test_source_frame_gaps_age_tracks_and_keep_indices():
    from bxjudge.track import track_frames

    def detector(frame, idx):
        return [Detection(idx, "bey", (10, 10, 40, 30), 0.9)]

    tracks = track_frames(((i, blank()) for i in (100, 101, 400, 401)), detector, 60)
    assert len(tracks) == 2
    assert [[d.frame for d in t.dets] for t in tracks] == [[100, 101], [400, 401]]


@pytest.mark.parametrize("indices", [[2, 2], [2, 1], [-1]])
def test_reject_invalid_source_frames(indices):
    from bxjudge.track import track_frames

    with pytest.raises(ValueError):
        track_frames(((i, blank()) for i in indices), lambda f, i: [], 60)


def test_reject_mismatched_detection_frame():
    from bxjudge.track import track_frames

    with pytest.raises(ValueError, match="frame"):
        track_frames(
            [(10, blank())], lambda f, i: [Detection(0, "bey", (0, 0, 5, 5), 0.9)], 60
        )


def test_never_confirmed_short_track_is_discarded():
    from bxjudge.track import track_frames

    tracks = track_frames(
        [(50, blank())], lambda f, i: [Detection(i, "bey", (0, 0, 20, 20), 0.9)], 60
    )
    assert tracks == []


def test_short_gap_preserves_identity_without_inventing_detections():
    from bxjudge.track import track_frames

    tracks = track_frames(
        ((i, blank()) for i in (10, 11, 14, 15)),
        lambda f, i: [Detection(i, "bey", (0, 0, 20, 20), 0.9)],
        60,
    )
    assert len(tracks) == 1
    assert [d.frame for d in tracks[0].dets] == [10, 11, 14, 15]


@pytest.mark.parametrize(
    "change",
    [
        {"unknown_tracker_option": 1},
        {"lost_track_buffer": -1},
        {"lost_track_buffer": 1.5},
        {"lost_track_buffer": True},
        {"minimum_consecutive_frames": True},
        {"minimum_consecutive_frames": 3},
        {"track_activation_threshold": float("nan")},
        {"track_activation_threshold": -0.1},
        {"track_activation_threshold": 1.1},
        {"track_activation_threshold": True},
        {"minimum_iou_threshold": float("inf")},
        {"minimum_iou_threshold": -1},
        {"minimum_iou_threshold": "0.1"},
        {"high_conf_det_threshold": 2},
    ],
)
def test_tracking_policy_rejects_invalid_supported_values(tmp_path, change):
    from pathlib import Path

    import yaml

    from bxjudge.detector_config import load_policy

    policy = yaml.safe_load(Path("config/detection.yaml").read_text())
    policy["tracking"].update(change)
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump(policy))
    with pytest.raises(ValueError, match="tracking"):
        load_policy(config)


@pytest.mark.parametrize("tracking", [None, [], {}])
def test_tracking_policy_requires_complete_mapping(tmp_path, tracking):
    from pathlib import Path

    import yaml

    from bxjudge.detector_config import load_policy

    policy = yaml.safe_load(Path("config/detection.yaml").read_text())
    policy["tracking"] = tracking
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump(policy))
    with pytest.raises(ValueError, match="tracking"):
        load_policy(config)


def test_track_frames_shares_startup_tracking_validation(tmp_path):
    from pathlib import Path

    import yaml

    from bxjudge.track import track_frames

    policy = yaml.safe_load(Path("config/detection.yaml").read_text())
    policy["tracking"]["unknown_tracker_option"] = 1
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump(policy))
    with pytest.raises(ValueError, match="tracking"):
        track_frames([], lambda *args: [], 30, config=config)


def test_tracking_policy_valid_boundaries_use_native_constructor(tmp_path):
    from pathlib import Path

    import yaml

    from bxjudge.detector_config import load_policy
    from bxjudge.track import track_frames

    policy = yaml.safe_load(Path("config/detection.yaml").read_text())
    policy["tracking"] = {
        "lost_track_buffer": 0,
        "minimum_consecutive_frames": 2,
        "track_activation_threshold": 0,
        "minimum_iou_threshold": 1,
        "high_conf_det_threshold": 1,
    }
    config = tmp_path / "valid.yaml"
    config.write_text(yaml.safe_dump(policy))
    assert load_policy(config)["tracking"] == policy["tracking"]
    assert track_frames([], lambda *args: [], 30, config=config) == []
