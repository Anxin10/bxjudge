import numpy as np
import pytest

from bxjudge.types import Detection


@pytest.fixture
def association_policy(tmp_path):
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(Path("config/detection.yaml").read_text())
    cfg["candidate_confidence"] = 0.1
    path = tmp_path / "association.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


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
    policy["confidence"] = 0
    policy["candidate_confidence"] = 0
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


@pytest.mark.parametrize("cls", ["bey", "bey_part", "launcher"])
def test_weak_associations_restore_id_but_never_become_judgment_evidence(
    cls, association_policy
):
    from bxjudge.track import track_frames

    scores = [0.9, 0.9, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9]
    result = track_frames(
        ((f, blank()) for f in range(len(scores))),
        lambda image, f: [Detection(f, cls, (10 + f, 10, 30 + f, 30), scores[f])],
        30,
        config=association_policy,
    )
    assert len(result) == 1
    assert [d.frame for d in result[0].dets] == [0, 1, 5, 6, 7]
    assert all(d.score >= 0.3 for d in result[0].dets)


@pytest.mark.parametrize("scores", [[0.9, 0.2, 0.2, 0.2], [0.2] * 8])
def test_weak_matches_do_not_confirm_single_strong_seed_or_spawn_new_tracks(
    scores, association_policy
):
    from bxjudge.track import track_frames

    result = track_frames(
        ((f, blank()) for f in range(len(scores))),
        lambda image, f: [Detection(f, "bey", (10, 10, 30, 30), scores[f])],
        30,
        config=association_policy,
    )
    assert result == []


@pytest.mark.parametrize("fps", [30, 120])
def test_long_weak_chain_expires_strong_identity_and_does_not_merge_rounds(
    fps, association_policy
):
    from bxjudge.config import load_thresholds
    from bxjudge.segment import segment_rounds
    from bxjudge.track import track_frames

    length = int(3 * fps)
    scores = [0.9, 0.9] + [0.2] * length + [0.9, 0.9]
    result = track_frames(
        ((f, blank()) for f in range(len(scores))),
        lambda image, f: [Detection(f, "bey", (10, 10, 30, 30), scores[f])],
        fps,
        config=association_policy,
    )
    assert [[d.frame for d in t.dets] for t in result] == [
        [0, 1],
        [length + 2, length + 3],
    ]
    assert segment_rounds(result, fps, load_thresholds()) == [
        (0, 1),
        (length + 2, length + 3),
    ]


@pytest.mark.parametrize("fps", [30, 120])
def test_only_strong_evidence_refreshes_native_time_budget(fps, association_policy):
    from bxjudge.track import track_frames

    def run(gap):
        scores = [0.9, 0.9] + [0.2] * (gap - 1) + [0.9, 0.9]
        return track_frames(
            ((f, blank()) for f in range(len(scores))),
            lambda image, f: [Detection(f, "bey", (10, 10, 30, 30), scores[f])],
            fps,
            config=association_policy,
        )

    assert len(run(fps)) == 1  # high frame exactly1s after previous high
    assert len(run(fps + 1)) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"candidate_confidence": None},
        {"candidate_confidence": True},
        {"candidate_confidence": -0.1},
        {"candidate_confidence": float("nan")},
        {"candidate_confidence": 0.4},
    ],
)
def test_candidate_policy_is_explicit_valid_and_not_above_evidence(tmp_path, change):
    from pathlib import Path

    import yaml

    from bxjudge.detector_config import load_policy

    cfg = yaml.safe_load(Path("config/detection.yaml").read_text())
    cfg.update(change)
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_policy(path)


def test_candidate_policy_cannot_be_missing(tmp_path):
    from pathlib import Path

    import yaml

    from bxjudge.detector_config import load_policy

    cfg = yaml.safe_load(Path("config/detection.yaml").read_text())
    cfg.pop("candidate_confidence", None)
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_policy(path)


@pytest.mark.parametrize(
    "field", ["track_activation_threshold", "high_conf_det_threshold"]
)
def test_weak_candidates_cannot_bypass_evidence_floor_by_tracking_settings(
    tmp_path, field
):
    from pathlib import Path

    import yaml

    from bxjudge.detector_config import load_policy

    cfg = yaml.safe_load(Path("config/detection.yaml").read_text())
    cfg["tracking"][field] = 0.2
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="evidence"):
        load_policy(path)


@pytest.mark.parametrize("fps", [30, 120])
def test_confirmed_high_track_with_weak_tail_does_not_extend_output_lifetime(
    fps, association_policy
):
    from bxjudge.track import track_frames

    scores = [0.9, 0.9] + [0.2] * int(2 * fps)
    tracks = track_frames(
        ((f, blank()) for f in range(len(scores))),
        lambda image, f: [Detection(f, "bey", (10, 10, 30, 30), scores[f])],
        fps,
        config=association_policy,
    )
    assert len(tracks) == 1 and [d.frame for d in tracks[0].dets] == [0, 1]


def test_unconfirmed_strong_seed_recovers_only_on_a_second_strong_match(
    association_policy,
):
    from bxjudge.track import track_frames

    scores = [0.9, 0.2, 0.2, 0.9]
    tracks = track_frames(
        ((f, blank()) for f in range(len(scores))),
        lambda image, f: [Detection(f, "bey", (10, 10, 30, 30), scores[f])],
        30,
        config=association_policy,
    )
    assert len(tracks) == 1 and [d.frame for d in tracks[0].dets] == [0, 3]


def test_opt_in_recovery_helps_fast_small_object_without_enabling_unvalidated_default(
    association_policy,
):
    from bxjudge.detector_config import load_policy
    from bxjudge.track import track_frames

    scores = [0.9, 0.9, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9]

    def detector(image, f):
        return [Detection(f, "bey", (100 + 5 * f, 100, 112 + 5 * f, 112), scores[f])]

    native = track_frames(((f, blank()) for f in range(len(scores))), detector, 30)
    trial = track_frames(
        ((f, blank()) for f in range(len(scores))),
        detector,
        30,
        config=association_policy,
    )
    assert load_policy()["candidate_confidence"] == load_policy()["confidence"] == 0.3
    assert [[d.frame for d in t.dets] for t in native] == [[0, 1], [5, 6, 7]]
    assert len(trial) == 1 and [d.frame for d in trial[0].dets] == [0, 1, 5, 6, 7]
