from types import SimpleNamespace

import numpy as np

from bxjudge.launch_mask_review import (
    ReleaseProbe,
    build_release_probes,
    release_candidate,
    review_release_probes,
)
from bxjudge.launch_phase import LaunchSegment
from bxjudge.types import Detection, Track


def test_review_anchors_candidate_without_inventing_evidence_across_gap():
    assert release_candidate({20: 0.1, 21: 0.2, 22: 0.5, 23: 0.7}) == (22, 23)
    assert release_candidate({20: 0.1, 21: 0.2, 23: 0.5, 24: 0.7}) is None
    assert release_candidate({20: 0.1, 21: 0.2, 22: 0.6, 23: 0.1}) is None
    assert release_candidate({20: 0.6, 21: 0.7}) is None


def test_review_decodes_only_when_needed_and_uses_same_native_boxes():
    box_bey = (2, 3, 4, 5)
    box_launcher = (0, 0, 8, 8)
    bey = Track(7, "bey", [Detection(f, "bey", box_bey, 1) for f in range(10, 16)])
    launcher = Track(
        8,
        "launcher",
        [Detection(f, "launcher", box_launcher, 1) for f in range(10, 16)],
    )
    probe = ReleaseProbe(1, 42, 7, 8, 14, 14, tuple(range(10, 16)))
    consumed = []

    def frames():
        for frame in range(50):
            consumed.append(frame)
            yield frame, np.full((1, 1, 3), frame, dtype=np.uint8)

    def mask(image, bey_box, launcher_box):
        assert (bey_box, launcher_box) == (box_bey, box_launcher)
        return 0.0 if image[0, 0, 0] < 12 else 0.75

    assert review_release_probes([], [bey, launcher], frames, mask) == {}
    assert not consumed
    result = review_release_probes([probe], [bey, launcher], frames, mask)[1][42]
    assert result["status"] == "disagrees"
    assert (result["frame"], result["confirmed_at_frame"]) == (12, 13)
    assert consumed[-1] == 15

    launcher.dets = [det for det in launcher.dets if det.frame != 12]
    result = review_release_probes([probe], [bey, launcher], frames, mask)[1][42]
    assert result["observations"][12] is None
    assert result["status"] == "insufficient"


def test_probe_without_box_release_requires_unique_mount_and_valid_identity():
    physical = SimpleNamespace(
        tracks=[Track(40, "bey", []), Track(41, "bey", [])],
        members={40: [7], 41: [9]},
        record={
            "pair_confirmed_at_frame": 20,
            "observations": [
                {"entity_id": 1, "frame": f, "native_id": 7, "identity_valid": f != 23}
                for f in range(17, 26)
            ],
        },
    )
    mounts = {7: [LaunchSegment("mounted", 17, 20, 8, 18)]}
    assert not build_release_probes(3, physical, {40: "X"}, mounts, {})
    [probe] = build_release_probes(3, physical, {40: "X", 41: "B"}, mounts, {})
    assert (probe.round, probe.track_id, probe.launcher_id, probe.anchor) == (
        3,
        40,
        8,
        21,
    )
    assert probe.box_release is None
    assert 23 not in probe.frames
