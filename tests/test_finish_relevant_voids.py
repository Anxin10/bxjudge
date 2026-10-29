"""Round voids judge only evidence up to the settled result (R55).

Missing geometry and unassigned observations arising only after the result is
settled cannot change who won; they stay visible as after_finish notes.
"""
from bxjudge.autocalib import Alignment
from bxjudge.dynamic_calibration import DynamicCalibration, GeometrySample
from bxjudge.finish import FinishResult
from bxjudge.identity import IdentityAnchors
from tests.helpers import make_track
from tests.test_two_bey_integration import invoke_two_pipeline, pipeline_setup

EYE = Alignment(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), 1.0, 0.0, 0.0, 1.0, 0.0)
FINISH = 50  # settled at FINISH + 3 + max(30, 6) - 1 = 82 with the shipped thresholds


def judged(tmp_path, monkeypatch, th, rules, *, third=(), missing=()):
    from bxjudge import pipeline

    tracks = [
        make_track(tid, "bey", {f: (x, 360) for f in range(35, 120)})
        for tid, x in [(1, 350), (2, 900)]
    ]
    if third:
        tracks.append(make_track(3, "bey", {f: (640, 600) for f in third}))
    source = pipeline_setup(monkeypatch, tmp_path, tracks)
    monkeypatch.setattr(
        pipeline, "anchor_sides",
        lambda *a, **k: IdentityAnchors({1: "X", 2: "B", 3: "X"}, {}),
    )

    def update(self, frame, image):
        gone = frame in missing
        self.samples[frame] = GeometrySample(frame, None if gone else EYE, 0,
                                             "lost" if gone else "tracking")
        reason = "synthetic_geometry_missing" if gone else None
        self.diagnostics[frame] = {"tracking": {"reason": reason}, "relocation": None,
                                   "reason": reason}
        return self.samples[frame]

    monkeypatch.setattr(DynamicCalibration, "update", update)
    monkeypatch.setattr(
        pipeline, "detect_finish", lambda *a: FinishResult("over", "B", FINISH, 0.95, [])
    )
    [r] = invoke_two_pipeline(source, tmp_path, rules, th)
    return r


def test_settled_frame_matches_the_shipped_thresholds(th):
    from bxjudge.finish import _settled_frame

    assert _settled_frame(FinishResult("over", "B", FINISH, 0.95, []), th) == 82


def test_clean_round_keeps_the_candidate_winner(tmp_path, monkeypatch, th, rules):
    r = judged(tmp_path, monkeypatch, th, rules)
    assert (r.finish, r.winner) == ("over", "X")


def test_geometry_missing_only_after_the_settled_frame_is_a_note(
    tmp_path, monkeypatch, th, rules
):
    r = judged(tmp_path, monkeypatch, th, rules, missing=range(83, 120))
    assert (r.finish, r.winner) == ("over", "X")
    assert "calibration_geometry_missing" not in r.review_reasons
    assert "after_finish:calibration_geometry_missing" in r.review_reasons


def test_geometry_missing_at_the_settled_frame_still_voids(
    tmp_path, monkeypatch, th, rules
):
    r = judged(tmp_path, monkeypatch, th, rules, missing=[82])
    assert r.finish is None and r.winner is None
    assert "calibration_geometry_missing" in r.review_reasons
    assert "finish_candidate_voided:over:B:50" in r.review_reasons


def test_geometry_missing_before_the_finish_still_voids(
    tmp_path, monkeypatch, th, rules
):
    r = judged(tmp_path, monkeypatch, th, rules, missing=[40])
    assert r.finish is None and r.winner is None
    assert "calibration_geometry_missing" in r.review_reasons


def test_unassigned_observation_only_after_the_settled_frame_is_a_note(
    tmp_path, monkeypatch, th, rules
):
    r = judged(tmp_path, monkeypatch, th, rules, third=range(90, 100))
    assert (r.finish, r.winner) == ("over", "X")
    assert "two_bey_unassigned_identity_unknown" not in r.review_reasons
    assert "after_finish:two_bey_unassigned_identity_unknown" in r.review_reasons


def test_unassigned_observation_up_to_the_settled_frame_still_voids(
    tmp_path, monkeypatch, th, rules
):
    r = judged(tmp_path, monkeypatch, th, rules, third=range(80, 83))
    assert r.finish is None and r.winner is None
    assert "two_bey_unassigned_identity_unknown" in r.review_reasons
    assert "finish_candidate_voided:over:B:50" in r.review_reasons


def test_an_earlier_void_does_not_hide_a_later_relevance_check(
    tmp_path, monkeypatch, th, rules
):
    # Geometry voids first; the unassigned check must still judge against the
    # candidate, not against the already-voided result.
    r = judged(tmp_path, monkeypatch, th, rules, missing=[40], third=range(90, 100))
    assert r.finish is None
    assert "after_finish:two_bey_unassigned_identity_unknown" in r.review_reasons
    assert "two_bey_unassigned_identity_unknown" not in r.review_reasons
