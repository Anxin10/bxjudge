"""Single-pass pipeline, uncertainty gates and batch isolation."""

import json
from pathlib import Path

import numpy as np
import pytest

from bxjudge import pipeline
from bxjudge.finish import FinishResult
from bxjudge.manifest import VideoInfo
from bxjudge.rules import is_not_a_round
from tests.helpers import (
    cal_with,
    make_calibration,
    make_ref_img,
    make_track,
    make_video,
    rr,
)


def info(path=Path("a.mp4")):
    return VideoInfo(path, 1, 1, 4, 30, 1280, 720, "h264", False, None)


def _empty_calib(tmp_path):
    """An empty calibration folder disjoint from the results folder tmp_path/out."""
    calib = tmp_path / "calib"
    calib.mkdir(exist_ok=True)
    return calib


def _kw(tmp_path):
    """judge_video's required keywords for unit tests."""
    return {"out_dir": tmp_path / "out", "autocalib": _ac()}


def _snapshot(c, sid="0123456789abcdef", kind="manual", score=0.95):
    from bxjudge.calibration_store import Snapshot

    if kind == "auto":
        return Snapshot(sid, "auto", c, score, "f" * 16, None, None, None)
    return Snapshot(sid, "manual", c, score, None, "calib/test.json", "0" * 64, None)


def calibrated(monkeypatch, tmp_path, *, tracks=None, snap_kind="manual", snap_score=0.95):
    c = cal_with(make_ref_img(), tmp_path)
    frame = np.zeros((720, 1280, 3), np.uint8)
    calls = []

    def frames(*args, **kwargs):
        calls.append(kwargs)
        for f in range(120):
            yield f, frame

    monkeypatch.setattr(pipeline, "iter_frames", frames)
    monkeypatch.setattr(pipeline, "select_snapshot",
                        lambda frame, *a: _snapshot(c, kind=snap_kind, score=snap_score))
    monkeypatch.setattr(pipeline, "write_preview", lambda out_dir, snap: None)

    def measured(frames, scaled, ref):
        assert scaled.frame_size == (1280, 720)
        assert scaled.battle_area == [(240, 120), (1040, 120), (1040, 560), (240, 560)]
        assert ref is frame  # the video's own first-second frame, not the 640x360 snapshot PNG
        for f, image in frames:
            yield f, image, 0.0 if f != 2 else float("nan")

    monkeypatch.setattr(pipeline, "iter_stadium_displacement", measured)
    if tracks is not None:

        def tracker(frames, detector, fps, **kwargs):
            for f, image in frames:
                detector(image, f)
            return tracks

        monkeypatch.setattr(pipeline, "track_frames", tracker)
    return calls


def test_recalibrate_closes_single_decode(monkeypatch, tmp_path, th, rules):
    closed = []

    def frames(*a, **k):
        try:
            for f in range(120):
                yield f, np.zeros((16, 32, 3), np.uint8)
        finally:
            closed.append(True)

    monkeypatch.setattr(pipeline, "iter_frames", frames)
    monkeypatch.setattr(pipeline, "select_snapshot", lambda *a: None)
    [r] = pipeline.judge_video(info(), Path("calib"), lambda *a: [], rules, th, **_kw(tmp_path))
    assert r.review_reasons == ["recalibrate"] and r.needs_review
    assert closed == [True]


def test_no_round_one_decode_and_resolution_alignment(monkeypatch, tmp_path, th, rules):
    calls = calibrated(monkeypatch, tmp_path)
    [r] = pipeline.judge_video(info(), tmp_path, lambda *a: [], rules, th, **_kw(tmp_path))
    assert r.review_reasons == ["no_round"] and r.calibration_id == "0123456789abcdef"
    assert len(calls) == 1 and calls[0]["long_edge"] == th.infer_long_edge


@pytest.mark.parametrize("reason,review", [("still", False), ("reversed", True)])
def test_stop_reasons_gating_motion_provenance(
    monkeypatch, tmp_path, th, rules, reason, review
):
    tracks = [
        make_track(1, "bey", {f: (360, 360) for f in range(120)}),
        make_track(2, "bey", {f: (900, 360) for f in range(120)}),
    ]
    calibrated(monkeypatch, tmp_path, tracks=tracks)
    from bxjudge.launch_check import LaunchAnalysis
    from bxjudge.types import LaunchCheck

    launch = LaunchCheck(
        {"X": "no", "B": "no"},
        {"X": True, "B": True},
        {"X": 5, "B": 6},
        {"X": "battle_area", "B": "battle_area"},
        "no",
        ["shake_by_bey_impact@5"],
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_launch",
        lambda *a: LaunchAnalysis(launch, {1: "X", 2: "B"}, (0, 21), {5, 6}),
    )
    monkeypatch.setattr(pipeline, "orbit_direction", lambda *a: "right")
    monkeypatch.setattr(pipeline, "stop_frame", lambda *a: (70, reason))
    monkeypatch.setattr(
        pipeline, "detect_finish", lambda *a: FinishResult("spin", "B", 70, 0.95, [])
    )
    [r] = pipeline.judge_video(
        info(), tmp_path, lambda *a: [], rules, th, out_dir=tmp_path / "out", autocalib=_ac()
    )
    assert r.needs_review == review and f"X_stop_{reason}" in r.review_reasons
    assert (
        r.t_end == 70 / 30
        and r.points == rules.finish_points["spin"]
        and r.winner == "X"
    )
    assert r.motion_evidence["window"] == [0, 21]
    assert r.calibration_id == "0123456789abcdef"
    assert r.motion_evidence["displacements"]["2"] is None
    assert r.motion_evidence["impact_frames"] == [5, 6]
    assert (tmp_path / "out/trajectories/a_r1.png").exists()


@pytest.mark.parametrize(
    "finish,stop_at,review,stop_reasons",
    [
        # Pocket finish at 30: loser B and winner X reversals long after it are notes.
        (
            FinishResult("xtreme", "B", 30, 0.95, ["after_finish:track_lost@X"]),
            70,
            False,
            {"after_finish:X_stop_reversed", "after_finish:B_stop_reversed"},
        ),
        # The loser's later stop is a note; the winner's stop inside the window
        # where a competing stop is not yet established still triggers review.
        (
            FinishResult("xtreme", "B", 30, 0.95, []),
            40,
            True,
            {"X_stop_reversed", "after_finish:B_stop_reversed"},
        ),
        # Unprefixed finish uncertainty still flags a confident result.
        (
            FinishResult("xtreme", "B", 30, 0.95, ["track_lost@X"]),
            70,
            True,
            {"after_finish:X_stop_reversed", "after_finish:B_stop_reversed"},
        ),
    ],
)
def test_post_finish_notes_do_not_trigger_review(
    monkeypatch, tmp_path, th, rules, finish, stop_at, review, stop_reasons
):
    tracks = [
        make_track(1, "bey", {f: (360, 360) for f in range(120)}),
        make_track(2, "bey", {f: (900, 360) for f in range(120)}),
    ]
    calibrated(monkeypatch, tmp_path, tracks=tracks)
    from bxjudge.launch_check import LaunchAnalysis
    from bxjudge.types import LaunchCheck

    launch = LaunchCheck(
        {"X": "no", "B": "no"},
        {"X": True, "B": True},
        {"X": 5, "B": 6},
        {"X": "battle_area", "B": "battle_area"},
        "no",
        [],
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_launch",
        lambda *a: LaunchAnalysis(launch, {1: "X", 2: "B"}, (0, 21), {5, 6}),
    )
    monkeypatch.setattr(pipeline, "orbit_direction", lambda *a: "right")
    monkeypatch.setattr(pipeline, "stop_frame", lambda *a: (stop_at, "reversed"))
    monkeypatch.setattr(pipeline, "detect_finish", lambda *a: finish)
    [r] = pipeline.judge_video(info(), tmp_path, lambda *a: [], rules, th, **_kw(tmp_path))
    assert r.needs_review == review
    assert (r.finish, r.winner) == ("xtreme", "X")
    assert stop_reasons <= set(r.review_reasons)


def _gate_round(monkeypatch, tmp_path, th, rules, **change):
    """Judge one fully confident, known round; ``change`` perturbs one input."""
    from bxjudge.launch_check import LaunchAnalysis
    from bxjudge.types import LaunchCheck

    tracks = [
        make_track(1, "bey", {f: (360, 360) for f in range(120)}),
        make_track(2, "bey", {f: (900, 360) for f in range(120)}),
    ]
    calibrated(monkeypatch, tmp_path, tracks=tracks,
               **{k: change[k] for k in ("snap_kind", "snap_score") if k in change})
    launch = LaunchCheck(
        {"X": "no", "B": "no", **change.get("touch", {})},
        {"X": True, "B": True},
        {"X": 5, "B": 6},
        {"X": "battle_area", "B": "battle_area"},
        change.get("air", "no"),
        # Informational evidence notes only (R22).
        ["shake_by_bey_impact@5", "shake_not_by_launcher_body@6", *change.get("notes", [])],
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_launch",
        lambda *a: LaunchAnalysis(launch, {1: "X", 2: "B"}, (0, 21), {5, 6}),
    )
    monkeypatch.setattr(
        pipeline, "orbit_direction", lambda *a: change.get("spin_dir", "right")
    )
    monkeypatch.setattr(
        pipeline, "stop_frame", lambda *a: (70, change.get("stop_reason", "still"))
    )
    finish = change.get(
        "finish", FinishResult("spin", "B", 70, change.get("confidence", 0.95), [])
    )
    monkeypatch.setattr(pipeline, "detect_finish", lambda *a: finish)
    [r] = pipeline.judge_video(info(), tmp_path, lambda *a: [], rules, th, **_kw(tmp_path))
    return r


def test_confident_known_round_with_informational_notes_is_not_review(
    monkeypatch, tmp_path, th, rules
):
    r = _gate_round(monkeypatch, tmp_path, th, rules)
    assert r.finish == "spin" and r.confidence >= th.review_confidence
    assert "shake_by_bey_impact@5" in r.review_reasons
    assert "X_stop_still" in r.review_reasons
    assert r.needs_review is False


@pytest.mark.parametrize(
    "kind, offset, flagged",
    [
        ("auto", -0.01, True),  # below the review band: review-triggering, not informational
        ("auto", 0.0, False),  # at the band
        ("auto", 0.1, False),
        ("manual", -0.5, False),  # manual snapshots are never flagged by the band
    ],
)
def test_auto_calibration_below_review_band_requires_review(
    monkeypatch, tmp_path, th, rules, kind, offset, flagged
):
    """Ruling F-1: a round judged with an auto snapshot scoring below auto_review_score
    gets needs_review with reason calibration_low_score; an otherwise clean round."""
    score = _ac().auto_review_score + offset
    r = _gate_round(monkeypatch, tmp_path, th, rules, snap_kind=kind, snap_score=score)
    assert ("calibration_low_score" in r.review_reasons) is flagged
    assert r.needs_review is flagged


def test_confidence_at_threshold_is_not_review(monkeypatch, tmp_path, th, rules):
    r = _gate_round(
        monkeypatch, tmp_path, th, rules, confidence=th.review_confidence
    )
    assert r.needs_review is False


def test_after_finish_notes_alone_do_not_trigger_review(
    monkeypatch, tmp_path, th, rules
):
    r = _gate_round(
        monkeypatch,
        tmp_path,
        th,
        rules,
        stop_reason="reversed",
        finish=FinishResult(
            "xtreme", "B", 30, 0.95, ["after_finish:track_lost@X"]
        ),
    )
    assert {"after_finish:X_stop_reversed", "after_finish:track_lost@X"} <= set(
        r.review_reasons
    )
    assert r.needs_review is False


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"confidence": "below"}, id="low-confidence"),
        pytest.param({"touch": {"X": "unknown"}}, id="touch-unknown"),
        pytest.param({"air": "maybe"}, id="air-maybe"),
        pytest.param({"spin_dir": "unknown"}, id="spin-dir-unknown"),
        pytest.param(
            {"finish": FinishResult(None, None, None, 0.95, [])}, id="finish-none"
        ),
    ],
)
def test_binding_review_gate_each_unknown_forces_review(
    monkeypatch, tmp_path, th, rules, change
):
    # Each case alters exactly one input; no uncertainty reason is present, so
    # only the confidence / unknown / maybe / finish-None gate can flag it.
    if change.get("confidence") == "below":
        change = {"confidence": th.review_confidence - 0.01}
    r = _gate_round(monkeypatch, tmp_path, th, rules, **change)
    assert not any(
        note.startswith(("track_evidence", "landing_timing", "id_swap"))
        for note in r.review_reasons
    )
    assert r.needs_review is True


def test_uncertainty_note_forces_review_even_when_confident(
    monkeypatch, tmp_path, th, rules
):
    r = _gate_round(
        monkeypatch, tmp_path, th, rules, notes=["landing_timing_unknown@X:5"]
    )
    assert r.needs_review is True


def test_missing_side_and_synthetic_always_review(monkeypatch, tmp_path, th, rules):
    calibrated(
        monkeypatch,
        tmp_path,
        tracks=[make_track(1, "bey", {f: (360, 360) for f in range(120)})],
    )
    provenance = {"dataset": {"provenance": {"synthetic": True}}}
    [r] = pipeline.judge_video(
        info(), tmp_path, lambda *a: [], rules, th, provenance=provenance, **_kw(tmp_path)
    )
    assert r.needs_review and "synthetic_detector" in r.review_reasons
    assert any("track_evidence_missing" in note for note in r.review_reasons)
    assert r.detector_provenance == provenance


def test_launch_metadata_uses_same_analysis(th):
    from bxjudge.launch_check import analyze_launch, check_launch

    c = make_calibration()
    tracks = [
        make_track(1, "bey", {f: (180, 180) for f in range(120)}),
        make_track(2, "bey", {f: (450, 180) for f in range(120)}),
    ]
    disp = dict.fromkeys(range(120), 0.0)
    analysis = analyze_launch(tracks, [], disp, c, th, 30)
    assert check_launch(tracks, [], disp, c, th, 30) == (analysis.check, analysis.sides)
    assert analysis.window[0] == 0 and analysis.window[1] > 0


def _synthetic_selection(tmp_path):
    selection = tmp_path / "selection.json"
    weights = tmp_path / "model.pt"
    weights.touch()
    selection.write_text(
        json.dumps(
            {
                "kind": "yolo26",
                "weights": str(weights),
                "dataset": {"provenance": {"synthetic": True}},
                "inference": {"long_edge": 1280},
            }
        )
    )
    return selection


def test_batch_placeholder_only_videos_continue_and_are_never_cached(
    tmp_path, monkeypatch
):
    folder = tmp_path / "batch"
    folder.mkdir()
    make_video(folder / "ok.mp4", seconds=0.1)
    (folder / "bad.mp4").write_bytes(b"bad")
    calib = tmp_path / "calib"
    calib.mkdir()  # exists but holds no matching calibration -> recalibrate (R56)
    monkeypatch.setattr(pipeline, "load_detector", lambda *a, **k: lambda *a: [])
    selection = _synthetic_selection(tmp_path)
    for name in ("out1", "out2"):
        out = tmp_path / name
        pipeline.run_batch(folder, calib, out, selection=selection)
        rows = json.loads((out / "rounds.json").read_text())
        assert {r["video"] for r in rows} == {"bad.mp4", "ok.mp4"}
        assert (
            rows[0]["detector_provenance"]["runtime"]["thresholds"]["infer_long_edge"]
            == 1280
        )
        assert rows[0]["detector_provenance"]["runtime"]["confidence"] == 0.3
        reasons = {r["video"]: r["review_reasons"] for r in rows}
        assert "decode_error" in reasons["bad.mp4"]
        assert "recalibrate" in reasons["ok.mp4"]
        assert "source_video" in json.loads((out / "videos.json").read_text())["ok.mp4"]
        db = out / "processed_videos.json"
        assert not db.exists() or json.loads(db.read_text()) == {}


@pytest.mark.parametrize("reason", ["recalibrate", "no_round", "analysis_error"])
def test_placeholder_only_video_is_retried_until_real_rounds(
    tmp_path, monkeypatch, rules, th, reason
):
    make_video(tmp_path / "a.mp4", seconds=0.1)
    out = tmp_path / "out"
    calls = []
    outcomes = [[pipeline._review("a.mp4", 0.1, reason)], [rr("spin", "X", video="a.mp4")]]

    def judge(*args, **kwargs):
        calls.append(1)
        result = outcomes[min(len(calls), 2) - 1]
        if reason == "analysis_error" and len(calls) == 1:
            raise RuntimeError("transient CUDA OOM")
        return result

    monkeypatch.setattr(pipeline, "judge_video", judge)

    def run():
        pipeline.run_batch(
            tmp_path, _empty_calib(tmp_path), out, detector=lambda *a: [], rules=rules, th=th
        )

    run()  # placeholder only: nothing recorded
    assert json.loads((out / "rounds.json").read_text())[0]["review_reasons"] == [reason]
    run()  # corrected rerun re-analyses and now yields a real round
    assert len(calls) == 2
    assert json.loads((out / "rounds.json").read_text())[0]["finish"] == "spin"
    run()  # real rounds are recorded: the video is skipped from now on
    assert len(calls) == 2


def test_mixed_real_and_diagnostic_rows_video_is_cached(
    tmp_path, monkeypatch, rules, th
):
    # A mixed result (real round plus a diagnostic row) is not placeholder-only.
    make_video(tmp_path / "a.mp4", seconds=0.1)
    diagnostic = pipeline._review("a.mp4", 0.1, "no_round")
    diagnostic.round = 2
    monkeypatch.setattr(
        pipeline,
        "judge_video",
        lambda *a, **k: [rr("spin", "X", video="a.mp4"), diagnostic],
    )
    out = tmp_path / "out"
    pipeline.run_batch(
        tmp_path, _empty_calib(tmp_path), out, detector=lambda *a: [], rules=rules, th=th
    )
    assert "a.mp4" in (out / "processed_videos.json").read_text()


def test_missing_calibration_directory_fails_before_any_output(tmp_path, monkeypatch):
    make_video(tmp_path / "a.mp4", seconds=0.1)

    def forbidden(*a, **k):
        pytest.fail("global calibration error reached model/analysis")

    monkeypatch.setattr(pipeline, "load_detector", forbidden)
    monkeypatch.setattr(pipeline, "judge_video", forbidden)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="calib"):
        pipeline.run_batch(tmp_path, tmp_path / "calbi-typo", out)
    assert not out.exists()
    calib_file = tmp_path / "calib.json"
    calib_file.write_text("{}")
    with pytest.raises(ValueError, match="calib"):
        pipeline.run_batch(tmp_path, calib_file, out)
    assert not out.exists()


def test_run_cli_missing_calibration_is_error_line(tmp_path, capsys):
    from bxjudge.cli import main

    make_video(tmp_path / "a.mp4", seconds=0.1)
    out = tmp_path / "out"
    code = main(
        ["run", str(tmp_path), "--calib", str(tmp_path / "nope"), "--out", str(out)]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "nope" in err
    assert not out.exists()


def test_invalid_global_selection_fails_without_output(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.run_batch(
            tmp_path, _empty_calib(tmp_path), tmp_path / "out", selection=tmp_path / "absent.json"
        )
    assert not (tmp_path / "out").exists()


def test_duplicate_basenames_rejected_before_model_or_outputs(tmp_path):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "same.mp4").touch()
    with pytest.raises(ValueError, match="basename"):
        pipeline.run_batch(tmp_path, tmp_path / "calib", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_cli_paths_limit_and_startup_error(tmp_path, monkeypatch, capsys):
    from bxjudge.cli import main

    calls = []
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **kw: calls.append((a, kw)))
    assert (
        main(
            [
                "run",
                str(tmp_path),
                "--calib",
                "calib",
                "--out",
                "result",
                "--selection",
                "smoke/selected.json",
                "--thresholds",
                "config/thresholds.yaml",
                "--limit",
                "3",
            ]
        )
        == 0
    )
    assert calls[0][0] == (tmp_path, Path("calib"), Path("result"))
    assert calls[0][1]["limit"] == 3 and calls[0][1]["selection"] == Path(
        "smoke/selected.json"
    )

    def fail(*a, **kw):
        raise ValueError("invalid selection")

    monkeypatch.setattr(pipeline, "run_batch", fail)
    assert main(["run", str(tmp_path)]) == 2
    assert "invalid selection" in capsys.readouterr().err


def test_air_collision_marker_comes_from_canonical_launch_analysis(th):
    from bxjudge.launch_check import analyze_launch

    c = make_calibration()
    # Contact plus simultaneous sharp turns; exact event frame must survive.
    x = make_track(1, "bey", {0: (100, 20), 1: (200, 30), 2: (300, 40), 3: (200, 30)})
    b = make_track(2, "bey", {0: (500, 20), 1: (400, 30), 2: (300, 40), 3: (400, 30)})
    analysis = analyze_launch([x, b], [], dict.fromkeys(range(100), 0.0), c, th, 30)
    assert analysis.check.air_collision == "yes"
    assert analysis.air_collision_frame == 2


@pytest.mark.parametrize(
    "change",
    [{"infer_long_edge": 0}, {"shake_px": float("nan")}, {"review_confidence": 2}],
)
def test_invalid_numeric_config_fails_before_model_and_outputs(
    tmp_path, monkeypatch, change
):
    from dataclasses import asdict

    import yaml

    from bxjudge.config import load_thresholds

    raw = {**asdict(load_thresholds()), **change}
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text(yaml.safe_dump(raw))

    def forbidden(*a, **k):
        pytest.fail("model must not load for invalid global configuration")

    monkeypatch.setattr(pipeline, "load_detector", forbidden)
    with pytest.raises(ValueError, match="threshold"):
        pipeline.run_batch(tmp_path, _empty_calib(tmp_path), tmp_path / "out", thresholds=thresholds)
    assert not (tmp_path / "out").exists()


def test_run_batch_injected_dependencies_require_no_selection(
    tmp_path, monkeypatch, rules, th
):
    make_video(tmp_path / "a.mp4", seconds=0.1)
    monkeypatch.setattr(
        pipeline, "judge_video", lambda *a, **k: [rr("spin", "X", video="a.mp4")]
    )
    out = tmp_path / "out"
    pipeline.run_batch(
        tmp_path, _empty_calib(tmp_path), out, detector=lambda *a: [], rules=rules, th=th
    )
    assert json.loads((out / "rounds.json").read_text())[0]["finish"] == "spin"


def test_streamed_crop_seed_and_all_source_frames(monkeypatch, tmp_path, th, rules):
    from bxjudge.types import Detection

    calibrated(monkeypatch, tmp_path)
    observed = []

    def detector(image, f):
        observed.append(f)
        return [
            Detection(f, "bey", (350.0, 350.0, 370.0, 370.0), 0.99),
            Detection(f, "bey", (890.0, 350.0, 910.0, 370.0), 0.99),
        ]

    crop_frames = []

    def activity(crops):
        crop_frames.append(set(crops))
        assert all(c.shape == (64, 64) and c.ndim == 2 for c in crops.values())
        return {}

    monkeypatch.setattr(pipeline, "spin_activity", activity)
    monkeypatch.setattr(pipeline, "rotation_rate", lambda crops: {})
    pipeline.judge_video(info(), tmp_path, detector, rules, th, **_kw(tmp_path))
    assert observed == list(range(120))
    assert crop_frames == [set(range(120)), set(range(120))]


def test_decode_failure_returns_review_artifact(monkeypatch, tmp_path, th, rules):
    from bxjudge.video import VideoDecodeError

    def broken(*a, **kw):
        raise VideoDecodeError("corrupt")
        yield

    monkeypatch.setattr(pipeline, "iter_frames", broken)
    [r] = pipeline.judge_video(info(), Path("calib"), lambda *a: [], rules, th, **_kw(tmp_path))
    assert r.review_reasons == ["decode_error"] and r.needs_review


@pytest.mark.parametrize("marker", ["absent", False, True, "yes", None])
def test_selection_accepts_current_export_provenance(tmp_path, th, marker):
    provenance = {
        "source_export": "/data/reviewed.json",
        "split": "whole video, sorted video_sha256",
        "val_ratio": 0.2,
        "images": [],
    }
    if marker != "absent":
        provenance["synthetic"] = marker
    weights = tmp_path / "model.pt"
    weights.touch()
    path = tmp_path / "selection.json"
    data = {
        "kind": "yolo26",
        "weights": str(weights),
        "dataset": {"provenance": provenance},
        "inference": {"long_edge": th.infer_long_edge},
    }
    path.write_text(json.dumps(data))
    if marker in ("yes", None):
        with pytest.raises(ValueError, match="provenance"):
            pipeline._load_selection(path, th)
    else:
        selected = pipeline._load_selection(path, th)
        [result] = pipeline._provenance([rr("spin", "X")], selected)
        assert ("synthetic_detector" in result.review_reasons) == (marker is True)
        assert result.needs_review == (marker is True)


def test_trailing_burst_part_keeps_event_time_and_trajectory(
    monkeypatch, tmp_path, th, rules
):
    import csv

    tracks = [
        make_track(1, "bey", {f: (360, 360) for f in range(11)}),
        make_track(2, "bey", {f: (900, 360) for f in range(11)}),
        make_track(3, "bey_part", {11: (365, 360)}),
    ]
    calibrated(monkeypatch, tmp_path, tracks=tracks)
    [r] = pipeline.judge_video(
        info(), tmp_path, lambda *a: [], rules, th, out_dir=tmp_path / "out", autocalib=_ac()
    )
    assert r.finish == "burst" and r.winner == "B" and r.t_end == 11 / 30
    path = tmp_path / "out/trajectories/a_r1.csv"
    rows = list(csv.DictReader(path.open()))
    assert any(row["cls"] == "bey_part" and row["frame"] == "11" for row in rows)


def test_trailing_parts_do_not_leak_from_next_round(monkeypatch, tmp_path, th, rules):
    from dataclasses import replace

    th = replace(th, round_gap_s=0)
    tracks = [
        make_track(1, "bey", {f: (360, 360) for f in range(11)}),
        make_track(2, "bey", {f: (900, 360) for f in range(11)}),
        make_track(3, "bey", {f: (360, 360) for f in range(12, 20)}),
        make_track(4, "bey", {f: (900, 360) for f in range(12, 20)}),
        make_track(5, "bey_part", {11: (365, 360), 12: (365, 360), 20: (365, 360)}),
    ]
    calibrated(monkeypatch, tmp_path, tracks=tracks)
    seen = []
    actual = pipeline.detect_finish

    def finish(bey, parts, *args):
        seen.append([d.frame for t in parts for d in t.dets])
        return actual(bey, parts, *args)

    monkeypatch.setattr(pipeline, "detect_finish", finish)
    results = pipeline.judge_video(info(), tmp_path, lambda *a: [], rules, th, **_kw(tmp_path))
    assert len(results) == 2
    assert seen == [[11], [12, 20]]
    assert results[0].finish == "burst" and results[0].t_end == 11 / 30


@pytest.mark.parametrize("window,frame", [(0.1, 14), (10, 120)])
def test_trailing_part_window_and_video_end_are_bounded(
    monkeypatch, tmp_path, th, rules, window, frame
):
    from dataclasses import replace

    th = replace(th, burst_part_window_s=window)
    tracks = [
        make_track(1, "bey", {f: (360, 360) for f in range(11)}),
        make_track(2, "bey", {f: (900, 360) for f in range(11)}),
        make_track(3, "bey_part", {frame: (365, 360)}),
    ]
    calibrated(monkeypatch, tmp_path, tracks=tracks)
    captured = []
    actual = pipeline.detect_finish

    def finish(bey, parts, *args):
        captured.extend(parts)
        return actual(bey, parts, *args)

    monkeypatch.setattr(pipeline, "detect_finish", finish)
    [r] = pipeline.judge_video(info(), tmp_path, lambda *a: [], rules, th, **_kw(tmp_path))
    assert captured == [] and r.finish is None


def test_injected_thresholds_are_the_detector_policy(tmp_path, monkeypatch, th, rules):
    from dataclasses import replace

    th = replace(th, infer_long_edge=640)
    weights = tmp_path / "model.pt"
    weights.touch()
    selection = tmp_path / "selected.json"
    selection.write_text(
        json.dumps(
            {
                "kind": "yolo26",
                "weights": str(weights),
                "dataset": {"provenance": {"synthetic": False}},
                "inference": {"long_edge": 640},
            }
        )
    )
    captured = []

    def loader(*args, **kwargs):
        captured.append(kwargs["thresholds"])
        return lambda *args: []

    monkeypatch.setattr(pipeline, "load_detector", loader)
    pipeline.run_batch(
        tmp_path, _empty_calib(tmp_path), tmp_path / "out", selection=selection, th=th, rules=rules
    )
    assert captured == [th]


@pytest.mark.parametrize(
    "tracking_change",
    [
        {"unknown_tracker_option": 1},
        {"lost_track_buffer": -1},
        {"minimum_iou_threshold": float("nan")},
    ],
)
def test_invalid_tracking_is_startup_failure(tmp_path, monkeypatch, tracking_change):
    import yaml

    policy = yaml.safe_load(Path("config/detection.yaml").read_text())
    policy["tracking"].update(tracking_change)
    config = tmp_path / "detection.yaml"
    config.write_text(yaml.safe_dump(policy))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid global tracker config reached model construction")

    monkeypatch.setattr(pipeline, "load_detector", forbidden)
    with pytest.raises(ValueError, match="tracking"):
        pipeline.run_batch(
            tmp_path, _empty_calib(tmp_path), tmp_path / "out", detection_config=config
        )
    assert not (tmp_path / "out").exists()


def test_case_variant_video_artifact_collision_fails_preflight(tmp_path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "a.MP4").touch()
    with pytest.raises(ValueError, match="basename|artifact"):
        pipeline.run_batch(tmp_path, _empty_calib(tmp_path), tmp_path / "out")
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- VLM phase (Task 14)


def _vlm_config(tmp_path, **change):
    import yaml

    raw = yaml.safe_load(Path("config/vlm.yaml").read_text(encoding="utf-8"))
    raw.update(change)
    path = tmp_path / "vlm.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_run_records_batch_wall_clock_with_vlm_phase_inside(tmp_path, monkeypatch):
    import time

    folder = tmp_path / "batch"
    folder.mkdir()
    make_video(folder / "a.mp4", seconds=0.1)

    def load(*a, **k):
        time.sleep(0.05)  # model init counts toward the batch wall-clock
        return lambda image, frame: []

    def review(out_dir, **kw):
        time.sleep(0.05)
        return {}

    monkeypatch.setattr(pipeline, "load_detector", load)
    monkeypatch.setattr(pipeline, "judge_video", lambda *a, **k: [rr("spin", "X", video="a.mp4")])
    monkeypatch.setattr(pipeline, "review_results_dir", review)
    out = tmp_path / "out"
    pipeline.run_batch(folder, _empty_calib(tmp_path), out, selection=_synthetic_selection(tmp_path),
                       vlm_config=_vlm_config(tmp_path))
    [run] = json.loads((out / "runs.json").read_text())
    videos = json.loads((out / "videos.json").read_text())
    assert run["kind"] == "run" and set(run["videos"]) == {"a.mp4"}
    assert run["videos"]["a.mp4"] == pytest.approx(videos["a.mp4"]["duration_s"])
    assert run["vlm_wall_s"] >= 0.05
    # wall-clock covers init + core + VLM once: at least core + VLM phase, never their sum twice
    assert run["wall_s"] >= 0.1 + videos["a.mp4"]["processing_s"]
    assert run["wall_s"] >= run["vlm_wall_s"] + videos["a.mp4"]["processing_s"]
    # nothing new to judge: no record (init alone is not judgment wall-clock)
    pipeline.run_batch(folder, _empty_calib(tmp_path), out, selection=_synthetic_selection(tmp_path),
                       vlm_config=_vlm_config(tmp_path))
    assert len(json.loads((out / "runs.json").read_text())) == 1


def test_vlm_phase_runs_after_detector_is_released(tmp_path, monkeypatch):
    import weakref

    folder = tmp_path / "batch"
    folder.mkdir()
    make_video(folder / "a.mp4", seconds=0.1)
    refs = []

    class Detector:
        def __call__(self, image, frame):
            return []

    def load(*a, **k):
        d = Detector()
        refs.append(weakref.ref(d))
        return d

    monkeypatch.setattr(pipeline, "load_detector", load)
    monkeypatch.setattr(
        pipeline, "judge_video", lambda *a, **k: [rr("spin", "X", video="a.mp4")]
    )
    phases = []

    def review(out_dir, **kw):
        assert refs and refs[0]() is None, "detector still alive when the VLM loads"
        assert (out_dir / "rounds.json").exists()
        phases.append((out_dir, kw))
        return {"asked": 0, "answered": 0, "unavailable": 0, "models": {}}

    monkeypatch.setattr(pipeline, "review_results_dir", review)
    out = tmp_path / "out"
    pipeline.run_batch(
        folder,
        _empty_calib(tmp_path),
        out,
        selection=_synthetic_selection(tmp_path),
        vlm_config=_vlm_config(tmp_path),
    )
    [(out_dir, kw)] = phases
    assert out_dir == out and kw["videos"] == {"a.mp4"}
    assert "calib_dir" not in kw and kw["config"].model == "qwen3.8:27b"


def test_detector_freed_then_gpu_cache_released_before_first_vlm_work(
    tmp_path, monkeypatch
):
    import weakref

    folder = tmp_path / "batch"
    folder.mkdir()
    make_video(folder / "a.mp4", seconds=0.1)
    events = []

    class Detector:
        def __call__(self, image, frame):
            return []

    def load(*a, **k):
        d = Detector()
        weakref.finalize(d, events.append, "detector_freed")
        return d

    def judge(info, calib_dir, detector, *a, **k):
        detector(np.zeros((4, 4, 3), np.uint8), 0)  # used, then dropped by the callee
        return [rr("spin", "X", video="a.mp4")]

    real_release = pipeline._release_gpu_memory
    monkeypatch.setattr(pipeline, "load_detector", load)
    monkeypatch.setattr(pipeline, "judge_video", judge)
    monkeypatch.setattr(
        pipeline,
        "_release_gpu_memory",
        lambda: (events.append("gpu_release"), real_release()),
    )
    monkeypatch.setattr(
        pipeline,
        "review_results_dir",
        lambda out_dir, **kw: events.append("vlm_phase") or {},
    )
    pipeline.run_batch(
        folder,
        _empty_calib(tmp_path),
        tmp_path / "out",
        selection=_synthetic_selection(tmp_path),
        vlm_config=_vlm_config(tmp_path),
    )
    assert events == ["detector_freed", "gpu_release", "vlm_phase"]


def test_release_gpu_memory_collects_then_empties_torch_cache(monkeypatch):
    import sys
    import types

    events = []
    cuda = types.SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: events.append("empty_cache")
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(pipeline.gc, "collect", lambda: events.append("gc") or 0)
    pipeline._release_gpu_memory()
    assert events == ["gc", "empty_cache"]
    # never imports torch just to release (CPU-only / injected runs)
    events.clear()
    monkeypatch.delitem(sys.modules, "torch")
    pipeline._release_gpu_memory()
    assert events == ["gc"] and "torch" not in sys.modules


def test_vlm_phase_is_opt_in_and_config_guarded(tmp_path, monkeypatch, rules, th):
    make_video(tmp_path / "a.mp4", seconds=0.1)
    monkeypatch.setattr(
        pipeline, "judge_video", lambda *a, **k: [rr("spin", "X", video="a.mp4")]
    )
    monkeypatch.setattr(
        pipeline, "review_results_dir", lambda *a, **k: pytest.fail("VLM ran")
    )
    for name, vlm_config in (
        ("default", None),
        ("disabled", _vlm_config(tmp_path, after_run=False)),
    ):
        pipeline.run_batch(
            tmp_path,
            _empty_calib(tmp_path),
            tmp_path / name,
            detector=lambda *a: [],
            rules=rules,
            th=th,
            vlm_config=vlm_config,
        )


def test_invalid_vlm_config_fails_before_model_and_outputs(tmp_path, monkeypatch):
    make_video(tmp_path / "a.mp4", seconds=0.1)

    def forbidden(*a, **k):
        pytest.fail("reached model/judging with invalid VLM config")

    monkeypatch.setattr(pipeline, "load_detector", forbidden)
    monkeypatch.setattr(pipeline, "judge_video", forbidden)
    with pytest.raises(ValueError, match="loopback"):
        pipeline.run_batch(
            tmp_path,
            _empty_calib(tmp_path),
            tmp_path / "out",
            vlm_config=_vlm_config(tmp_path, host="http://192.0.2.10:11435"),
        )
    assert not (tmp_path / "out").exists()


def test_run_cli_passes_vlm_config_unless_disabled(tmp_path, monkeypatch):
    from bxjudge.cli import main

    calls = []
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **kw: calls.append(kw))
    assert main(["run", str(tmp_path)]) == 0
    assert calls[-1]["vlm_config"] == Path("config/vlm.yaml")
    assert main(["run", str(tmp_path), "--vlm-config", "x/vlm.yaml"]) == 0
    assert calls[-1]["vlm_config"] == Path("x/vlm.yaml")
    assert main(["run", str(tmp_path), "--no-vlm"]) == 0
    assert calls[-1]["vlm_config"] is None


def test_unreachable_vlm_phase_records_no_vlm_wall_but_keeps_batch_wall(tmp_path, monkeypatch):
    folder = tmp_path / "batch"
    folder.mkdir()
    make_video(folder / "a.mp4", seconds=0.1)
    monkeypatch.setattr(pipeline, "load_detector", lambda *a, **k: (lambda image, frame: []))
    monkeypatch.setattr(pipeline, "judge_video", lambda *a, **k: [rr("spin", "X", video="a.mp4")])
    monkeypatch.setattr(pipeline, "review_results_dir", lambda out_dir, **kw: {
        "asked": 0, "answered": 0, "unavailable": 1, "skipped": "vlm_unreachable", "models": {}})
    out = tmp_path / "out"
    pipeline.run_batch(folder, _empty_calib(tmp_path), out, selection=_synthetic_selection(tmp_path),
                       vlm_config=_vlm_config(tmp_path))
    [run] = json.loads((out / "runs.json").read_text())
    assert run["vlm_wall_s"] is None and run["wall_s"] > 0  # R65


# --------------------------------------------------------------------------- calibration snapshots (auto-calibration Task 3)


def _ac():
    from bxjudge.autocalib import load_autocalib_config

    return load_autocalib_config()


def _cpu_th():
    from dataclasses import replace

    from bxjudge.config import load_thresholds

    return replace(load_thresholds(), decode_hwaccel="cpu")


def _clip_setup(tmp_path, *, shift=(45, 20), fps=30, seconds=3.0):
    """videos/clip.mp4 rendered from stadium_scene(shift); calib/ holds only the unshifted manual."""
    from tests.helpers import add_manual_calibration, stadium_scene, write_scene_video

    videos, calib, out = tmp_path / "videos", tmp_path / "calib", tmp_path / "out"
    videos.mkdir()
    write_scene_video(videos / "clip.mp4", stadium_scene(shift=shift)[0], seconds=seconds, fps=fps)
    add_manual_calibration(calib, stadium_scene())
    return videos, calib, out


class _JudgeSpy:
    """Records which videos ``judge_video`` analysed (and still runs the real one)."""

    def __init__(self, monkeypatch):
        self.videos: set[str] = set()
        real = pipeline.judge_video

        def spy(info, *a, **k):
            self.videos.add(info.path.name)
            return real(info, *a, **k)

        monkeypatch.setattr(pipeline, "judge_video", spy)


def _run(videos, calib, out, rules, detector=None, **kw):
    from tests.helpers import FakeDetector

    pipeline.run_batch(videos, calib, out, detector=detector or FakeDetector(),
                       th=kw.pop("th", _cpu_th()), rules=rules,
                       autocalib=kw.pop("autocalib", _ac()), **kw)


def _freeze_audit(out):
    """R69: audit marks written by hand need their audit_sample.json record."""
    from bxjudge.output import _write_json, read_rounds
    from tests.helpers import audit_of

    _write_json(out / "audit_sample.json", audit_of(read_rounds(out)))


def _confirm_reviewed(out, video, round_no, rules):
    from bxjudge.output import read_rounds, save_rounds

    rows = read_rounds(out)
    for r in rows:
        if r["video"] == video and r["round"] == round_no:
            r["reviewed"] = True
    save_rounds(rows, out, rules)


def test_moved_camera_is_judged_with_auto_snapshot_and_recorded(tmp_path, rules):
    videos, calib, out = _clip_setup(tmp_path)
    _run(videos, calib, out, rules)
    v = json.loads((out / "videos.json").read_text())["clip.mp4"]
    assert v["calibration"]["kind"] == "auto"
    assert v["calibration"]["name"] == "auto-scene" and v["calibration"]["template_id"]
    assert v["calibration"]["score"] >= _ac().min_score
    rows = json.loads((out / "rounds.json").read_text())
    assert rows and not any("recalibrate" in r["review_reasons"] for r in rows)
    assert {r["calibration_id"] for r in rows} == {v["calibration"]["id"]}
    assert (out / "calibrations" / f"{v['calibration']['id']}.preview.jpg").is_file()
    db = json.loads((out / "processed_videos.json").read_text())
    assert [rec["calibration_id"] for rec in db.values()] == [v["calibration"]["id"]]
    assert not list(calib.glob("auto*"))  # auto results never go into calib/


@pytest.mark.parametrize("band, flagged", [(1.0, True), (None, False)])
def test_run_flags_rounds_of_a_low_scoring_auto_snapshot(tmp_path, rules, band, flagged):
    """Ruling F-1 end to end: the synthetic clip aligns at ~0.99 (< 1); a band of 1.0 flags
    every real round, the shipped band (0.75) does not; manual rounds are never flagged."""
    from dataclasses import replace

    from tests.helpers import stadium_scene, write_scene_video

    videos, calib, out = _clip_setup(tmp_path)
    write_scene_video(videos / "still.mp4", stadium_scene()[0], seconds=3.0, fps=30)
    ac = _ac() if band is None else replace(_ac(), auto_review_score=band)
    _run(videos, calib, out, rules, autocalib=ac)
    v = json.loads((out / "videos.json").read_text())
    assert v["clip.mp4"]["calibration"]["kind"] == "auto"
    assert _ac().auto_review_score < v["clip.mp4"]["calibration"]["score"] < 1.0
    assert v["still.mp4"]["calibration"]["kind"] == "manual"
    rows = json.loads((out / "rounds.json").read_text())
    auto_rows = [r for r in rows if r["video"] == "clip.mp4"]
    manual_rows = [r for r in rows if r["video"] == "still.mp4"]
    assert auto_rows and manual_rows
    assert not any(is_not_a_round(r["review_reasons"]) for r in rows)
    for r in auto_rows:
        assert ("calibration_low_score" in r["review_reasons"]) is flagged
        if flagged:
            assert r["needs_review"] is True
    assert not any("calibration_low_score" in r["review_reasons"] for r in manual_rows)


def test_adding_a_manual_calibration_reanalyses_an_auto_video(tmp_path, rules, monkeypatch):
    from tests.helpers import add_manual_calibration, stadium_scene

    videos, calib, out = _clip_setup(tmp_path)
    _run(videos, calib, out, rules)
    add_manual_calibration(calib, stadium_scene(shift=(45, 20)), name="moved")
    spy = _JudgeSpy(monkeypatch)
    selections = []
    real_select = pipeline.select_snapshot
    monkeypatch.setattr(pipeline, "select_snapshot",
                        lambda *a, **k: selections.append(1) or real_select(*a, **k))
    _run(videos, calib, out, rules)
    assert spy.videos == {"clip.mp4"}
    assert len(selections) == 1  # the pre-run selection is reused by judging, not repeated
    v = json.loads((out / "videos.json").read_text())["clip.mp4"]
    assert v["calibration"]["kind"] == "manual" and v["calibration"]["name"] == "moved"
    assert v["calibration"]["template_id"] is None and v["calibration"]["score"] >= 0.8
    rows = json.loads((out / "rounds.json").read_text())
    assert {r["calibration_id"] for r in rows} == {v["calibration"]["id"]}


def test_added_better_template_rejudges_an_auto_video_once(tmp_path, rules, monkeypatch):
    """Final review I2 probe (ruling F-2): run 1 has only a noisy template A (auto 0.973);
    adding template B (manual similarity 0.639, alignment 0.996) gives a new auto snapshot
    from B and the video is re-judged on the next run; a further run reuses it."""
    import cv2

    from bxjudge.calibration_store import select_snapshot
    from tests.helpers import add_manual_calibration, stadium_scene, write_scene_video

    videos, calib, out = tmp_path / "videos", tmp_path / "calib", tmp_path / "out"
    videos.mkdir()
    write_scene_video(videos / "clip.mp4", stadium_scene(shift=(45, 20))[0])
    img_a, cal_a = stadium_scene()
    noisy = np.clip(img_a + np.random.default_rng(1).normal(0, 45, img_a.shape), 0, 255)
    add_manual_calibration(calib, (noisy.astype(np.uint8), cal_a), name="A_old")
    _run(videos, calib, out, rules)
    rec1 = json.loads((out / "videos.json").read_text())["clip.mp4"]["calibration"]
    assert rec1["kind"] == "auto" and rec1["name"] == "auto-A_old"

    add_manual_calibration(calib, stadium_scene(shift=(37, 15)), name="B_new")
    b_id = select_snapshot(cv2.imread(str(calib / "B_new.png")), calib, tmp_path / "ids",
                           _cpu_th(), _ac()).id
    spy = _JudgeSpy(monkeypatch)
    _run(videos, calib, out, rules)
    assert spy.videos == {"clip.mp4"}
    rec2 = json.loads((out / "videos.json").read_text())["clip.mp4"]["calibration"]
    assert rec2["kind"] == "auto" and rec2["name"] == "auto-B_new"
    assert rec2["id"] != rec1["id"] and rec2["template_id"] == b_id
    assert rec2["score"] > rec1["score"]
    rows = json.loads((out / "rounds.json").read_text())
    assert rows and {r["calibration_id"] for r in rows} == {rec2["id"]}

    spy = _JudgeSpy(monkeypatch)
    _run(videos, calib, out, rules)  # template set unchanged: reused, nothing re-judged
    assert spy.videos == set()
    assert json.loads((out / "videos.json").read_text())["clip.mp4"]["calibration"] == rec2


def test_unchanged_calibration_is_not_reanalysed(tmp_path, rules, monkeypatch):
    videos, calib, out = _clip_setup(tmp_path)
    _run(videos, calib, out, rules)
    spy = _JudgeSpy(monkeypatch)
    _run(videos, calib, out, rules)
    assert spy.videos == set()


def test_reanalysis_keeps_reviewed_rows(tmp_path, rules):
    from tests.helpers import add_manual_calibration, stadium_scene

    videos, calib, out = _clip_setup(tmp_path)
    _run(videos, calib, out, rules)
    first_id = json.loads((out / "videos.json").read_text())["clip.mp4"]["calibration"]["id"]
    _confirm_reviewed(out, "clip.mp4", 1, rules)
    add_manual_calibration(calib, stadium_scene(shift=(45, 20)), name="moved")
    _run(videos, calib, out, rules)
    v = json.loads((out / "videos.json").read_text())["clip.mp4"]
    assert v["calibration"]["id"] != first_id
    row = next(r for r in json.loads((out / "rounds.json").read_text()) if r["round"] == 1)
    assert row["reviewed"] is True and row["calibration_id"] == first_id


def test_template_at_1080p_scales_to_720p_decode_and_keeps_120fps_timing(
    tmp_path, rules, monkeypatch
):
    from bxjudge.calibration import scale_calibration
    from bxjudge.calibration_store import load_snapshot
    from tests.helpers import (
        FakeDetector,
        add_manual_calibration,
        stadium_scene,
        write_scene_video,
    )

    videos, calib, out = tmp_path / "videos", tmp_path / "calib", tmp_path / "out"
    videos.mkdir()
    write_scene_video(videos / "clip.mp4", stadium_scene()[0], seconds=2.5, fps=120)
    add_manual_calibration(calib, stadium_scene(size=(1920, 1080)))
    used = []
    real = pipeline.analyze_launch
    monkeypatch.setattr(pipeline, "analyze_launch",
                        lambda tracks, launchers, disp, c, *a: used.append(c)
                        or real(tracks, launchers, disp, c, *a))
    detector = FakeDetector(launch_frame=150)
    _run(videos, calib, out, rules, detector=detector)
    vid = json.loads((out / "videos.json").read_text())["clip.mp4"]
    snap = load_snapshot(out, vid["calibration"]["id"])
    assert snap.kind == "manual" and snap.calibration.frame_size == (1920, 1080)
    expected = scale_calibration(snap.calibration, (1280, 720))
    assert expected.frame_size == (1280, 720)
    assert detector.shapes == {(1280, 720)}
    assert used and all(c.frame_size == (1280, 720) for c in used)
    assert used[0].battle_area == pytest.approx(expected.battle_area)
    rows = json.loads((out / "rounds.json").read_text())
    assert rows[0]["t_launch"] == pytest.approx(150 / 120.0, abs=1e-6)
    assert rows[0]["calibration_id"] == snap.id


def test_selection_failing_after_a_previous_success_reanalyses(tmp_path, rules, monkeypatch):
    videos, calib, out = _clip_setup(tmp_path)
    _run(videos, calib, out, rules)
    (calib / "scene.json").unlink()  # no template left -> selection returns None
    spy = _JudgeSpy(monkeypatch)
    _run(videos, calib, out, rules)
    assert spy.videos == {"clip.mp4"}
    assert json.loads((out / "videos.json").read_text())["clip.mp4"]["calibration"] is None
    rows = json.loads((out / "rounds.json").read_text())
    assert [r["review_reasons"] for r in rows] == [["recalibrate"]]
    assert rows[0]["calibration_id"] is None


def test_unreadable_first_second_in_preselection_falls_back_to_normal_flow(
    tmp_path, rules, monkeypatch
):
    from bxjudge.video import VideoDecodeError

    videos, calib, out = _clip_setup(tmp_path)
    _run(videos, calib, out, rules)

    def broken(info, th):
        raise VideoDecodeError(f"{info.path}: corrupt")

    monkeypatch.setattr(pipeline, "first_second_frame", broken)
    spy = _JudgeSpy(monkeypatch)
    _run(videos, calib, out, rules)  # must not crash the batch
    assert spy.videos == {"clip.mp4"}


@pytest.mark.parametrize(
    "recorded,current,vetoed,expected",
    [
        ("a", "a", set(), False),
        ("a", "b", set(), True),  # new manual / edited template / better template
        ("a", None, set(), True),  # selection now fails after a previous success
        (None, None, set(), False),  # never succeeded, still failing
        (None, "a", set(), True),
        # Ruling T4-3: a vetoed record is reanalysed only for a usable non-vetoed snapshot.
        ("a", None, {"a"}, False),  # veto alone: rows and original predictions stay
        ("a", "a", {"a"}, False),  # (selection never returns a vetoed id)
        ("a", "b", {"a"}, True),  # manual added / template changed
        ("a", "a", {"b"}, False),  # another group's veto
    ],
)
def test_needs_reanalysis_rules(recorded, current, vetoed, expected):
    assert pipeline.needs_reanalysis(recorded, current, vetoed) is expected


def test_row_calibration_id_round_trips(rules, tmp_path):
    from bxjudge.output import read_rounds, row_to_result, write_results

    r = rr("spin", "X", video="a.mp4")
    r.calibration_id = "0123456789abcdef"
    write_results([r], tmp_path, rules)
    [row] = read_rounds(tmp_path)
    assert row["calibration_id"] == "0123456789abcdef"
    assert row_to_result(row).calibration_id == "0123456789abcdef"
    assert "calibration_id" not in (tmp_path / "rounds.csv").read_text().splitlines()[0]
    with pytest.raises(ValueError, match="calibration_id"):
        row_to_result({**row, "calibration_id": 7})


def test_run_cli_passes_autocalib_config(tmp_path, monkeypatch):
    from bxjudge.cli import main

    calls = []
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **kw: calls.append(kw))
    assert main(["run", str(tmp_path)]) == 0
    assert calls[-1]["autocalib_config"] == Path("config/autocalib.yaml")
    assert main(["run", str(tmp_path), "--autocalib", "x/ac.yaml"]) == 0
    assert calls[-1]["autocalib_config"] == Path("x/ac.yaml")


# --------------------------------------------------------------------------- displacement reference (ruling T3-1)

_SHAKE = {f: (4, 0) for f in range(42, 46)}  # a 4 px stadium shake right after launch (frame 40)


def _displacement_spy(monkeypatch):
    """Records each judged video's displacement series ({frame: px}) in call order."""
    series = []
    real = pipeline.iter_stadium_displacement

    def spy(frames, c, ref):
        s = {}
        series.append(s)
        for f, image, d in real(frames, c, ref):
            s[f] = d
            yield f, image, d

    monkeypatch.setattr(pipeline, "iter_stadium_displacement", spy)
    return series


def _assert_own_reference(series, th, rows=None):
    """Static frames stay below shake_px; the synthetic shake gives an onset in the window."""
    static = [d for f, d in series.items() if f not in _SHAKE and np.isfinite(d)]
    assert static and max(static) < th.shake_px
    assert all(series[f] > th.shake_px for f in _SHAKE)
    assert series[min(_SHAKE) - 1] <= th.shake_px  # an onset: detect_touch can see it
    if rows is not None:
        window = rows[0]["motion_evidence"]["displacements"]
        assert all(window[str(f)] > th.shake_px for f in _SHAKE)


def test_reused_auto_snapshot_measures_shake_against_the_videos_own_frame(
    tmp_path, rules, monkeypatch
):
    from tests.helpers import add_manual_calibration, stadium_scene, write_scene_video

    videos, calib, out = tmp_path / "videos", tmp_path / "calib", tmp_path / "out"
    videos.mkdir()
    write_scene_video(videos / "a.mp4", stadium_scene(shift=(45, 20))[0])
    # b: the camera is 4 px away from a's frame, which becomes the group reference
    write_scene_video(videos / "b.mp4", stadium_scene(shift=(49, 20))[0], shake=_SHAKE)
    add_manual_calibration(calib, stadium_scene())
    series = _displacement_spy(monkeypatch)
    _run(videos, calib, out, rules)
    recorded = json.loads((out / "videos.json").read_text())
    assert recorded["a.mp4"]["calibration"]["kind"] == "auto"
    assert recorded["b.mp4"]["calibration"]["id"] == recorded["a.mp4"]["calibration"]["id"]
    rows = [r for r in json.loads((out / "rounds.json").read_text()) if r["video"] == "b.mp4"]
    _assert_own_reference(series[1], _cpu_th(), rows)

    # Reanalysis path (judge_video(snapshot=...) from the pre-run selection): same reference.
    db = out / "processed_videos.json"
    records = json.loads(db.read_text())
    key = str((videos / "b.mp4").resolve())
    records[key]["calibration_id"] = "0000000000000000"  # recorded id differs -> reanalysis
    db.write_text(json.dumps(records))
    series.clear()
    spy = _JudgeSpy(monkeypatch)
    _run(videos, calib, out, rules)
    assert spy.videos == {"b.mp4"} and len(series) == 1
    _assert_own_reference(series[0], _cpu_th())


def test_manual_match_with_moved_camera_measures_shake_against_the_videos_own_frame(
    tmp_path, rules, monkeypatch
):
    from tests.helpers import add_manual_calibration, stadium_scene, write_scene_video

    videos, calib, out = tmp_path / "videos", tmp_path / "calib", tmp_path / "out"
    videos.mkdir()
    write_scene_video(videos / "clip.mp4", stadium_scene(shift=(3, 0))[0], shake=_SHAKE)
    add_manual_calibration(calib, stadium_scene())
    series = _displacement_spy(monkeypatch)
    _run(videos, calib, out, rules)
    v = json.loads((out / "videos.json").read_text())["clip.mp4"]
    assert v["calibration"]["kind"] == "manual"
    rows = json.loads((out / "rounds.json").read_text())
    _assert_own_reference(series[0], _cpu_th(), rows)


# --------------------------------------------------------------------------- vetoes (auto-calibration Task 4, ruling T4-1)


def _veto_setup(tmp_path, rules):
    """clip.mp4 and clip2.mp4 at the same moved camera position, judged (one auto group, one
    real round each); clip.mp4 round 1 confirmed and clip2.mp4 round 1 in the audit sample;
    then that group vetoed. Returns (videos, calib, out, snapshot id)."""
    from bxjudge.calibration_store import veto_calibration
    from bxjudge.output import read_rounds, save_rounds
    from tests.helpers import stadium_scene, write_scene_video

    videos, calib, out = _clip_setup(tmp_path)
    write_scene_video(videos / "clip2.mp4", stadium_scene(shift=(46, 21))[0], seconds=3.0, fps=30)
    _run(videos, calib, out, rules)
    v = json.loads((out / "videos.json").read_text())
    sid = v["clip.mp4"]["calibration"]["id"]
    assert v["clip2.mp4"]["calibration"]["id"] == sid  # one group
    _confirm_reviewed(out, "clip.mp4", 1, rules)
    rows = read_rounds(out)
    for r in rows:
        if r["video"] == "clip2.mp4":
            r["audit_sample"] = True
    save_rounds(rows, out, rules)
    _freeze_audit(out)
    veto_calibration(out, sid, rules=rules)
    return videos, calib, out, sid


def _eval(out):
    from bxjudge.calibration_store import vetoed_ids
    from bxjudge.evaluate import evaluate, read_audit, read_results

    rows, timings, runs = read_results(out)
    return evaluate(rows, timings, runs=runs, vetoed=vetoed_ids(out), audit=read_audit(out))


def test_rerun_after_a_veto_keeps_the_rows_and_the_evaluation_population(tmp_path, rules):
    from tests.helpers import (
        FakeDetector,
        read_matches,
        stadium_scene,
        write_scene_video,
    )

    videos, calib, out, sid = _veto_setup(tmp_path, rules)
    assert [(r["video"], r["round"]) for m in read_matches(out) for r in m["rounds"]] == [
        ("clip.mp4", 1)]  # reviewed rows still score; the unreviewed vetoed one does not
    before_rows = (out / "rounds.json").read_bytes()
    before = _eval(out)
    assert before.n_rounds == 2 and before.n_vetoed_rounds == 2
    # A new video at the vetoed position joins the group as recalibrate (no detector).
    write_scene_video(videos / "clip3.mp4", stadium_scene(shift=(45, 21))[0], seconds=3.0, fps=30)
    for _ in range(2):  # a second plain run changes nothing either
        detector = FakeDetector()
        _run(videos, calib, out, rules, detector=detector)
        assert detector.frames == []
        rows = json.loads((out / "rounds.json").read_text())
        assert [r for r in rows if r["video"] != "clip3.mp4"] == json.loads(before_rows)
        assert [r["review_reasons"] for r in rows if r["video"] == "clip3.mp4"] == [["recalibrate"]]
        after = _eval(out)
        assert (after.n_rounds, after.review_ratio, after.n_vetoed_rounds) == (
            before.n_rounds, before.review_ratio, before.n_vetoed_rounds)
        assert after.n_recalibrate_videos == 1 and after.n_vetoed_videos == 2
        assert after.n_pending_videos == 3
        v = json.loads((out / "videos.json").read_text())
        assert v["clip.mp4"]["calibration"]["id"] == v["clip2.mp4"]["calibration"]["id"] == sid
        assert next(r for r in rows if r["video"] == "clip2.mp4")["audit_sample"] is True


def test_manual_calibration_after_a_veto_reanalyses_the_group(tmp_path, rules):
    from tests.helpers import add_manual_calibration, stadium_scene

    videos, calib, out, sid = _veto_setup(tmp_path, rules)
    add_manual_calibration(calib, stadium_scene(shift=(45, 20)), name="moved")
    _run(videos, calib, out, rules)
    v = json.loads((out / "videos.json").read_text())
    new = v["clip2.mp4"]["calibration"]
    assert new["kind"] == "manual" and new["id"] != sid
    rows = {(r["video"], r["round"]): r for r in json.loads((out / "rounds.json").read_text())}
    assert rows[("clip2.mp4", 1)]["calibration_id"] == new["id"]  # unreviewed row replaced
    kept = rows[("clip.mp4", 1)]
    assert kept["reviewed"] is True and kept["calibration_id"] == sid  # reviewed row kept
    assert not any(is_not_a_round(r["review_reasons"]) for r in rows.values())


@pytest.mark.parametrize("rerun_while_vetoed", [False, True])
def test_unveto_then_rerun_keeps_the_rounds(tmp_path, rules, rerun_while_vetoed):
    from bxjudge.calibration_store import unveto_calibration
    from tests.helpers import FakeDetector, read_matches

    videos, calib, out, sid = _veto_setup(tmp_path, rules)
    before_rows = json.loads((out / "rounds.json").read_text())
    if rerun_while_vetoed:
        _run(videos, calib, out, rules)
    unveto_calibration(out, sid, rules=rules)
    detector = FakeDetector()
    _run(videos, calib, out, rules, detector=detector)
    assert detector.frames == []  # same snapshot as recorded: nothing to redo
    rows = json.loads((out / "rounds.json").read_text())
    assert rows == before_rows
    assert {(r["video"], r["round"]) for m in read_matches(out) for r in m["rounds"]} == {
        ("clip.mp4", 1), ("clip2.mp4", 1)}


def test_reviewed_video_whose_calibration_later_fails_stays_in_the_failure_counts(
    tmp_path, rules
):
    """Final review T4-M4 probe: a reviewed round 1, then the template disappears; the
    recalibrate placeholder collides with the kept reviewed row, but the video must still be
    counted as recalibrate / pending by the evaluation (video level, videos.json)."""
    videos, calib, out = _clip_setup(tmp_path)
    _run(videos, calib, out, rules)
    _confirm_reviewed(out, "clip.mp4", 1, rules)
    before = _eval(out)
    assert (before.n_recalibrate_videos, before.n_pending_videos, before.coverage) == (0, 0, 1.0)
    (calib / "scene.json").unlink()
    _run(videos, calib, out, rules)
    rows = json.loads((out / "rounds.json").read_text())
    assert [(r["round"], r["reviewed"]) for r in rows] == [(1, True)]  # the reviewed row is kept
    assert json.loads((out / "videos.json").read_text())["clip.mp4"]["calibration"] is None
    after = _eval(out)
    assert after.n_rounds == before.n_rounds
    assert (after.n_recalibrate_videos, after.n_pending_videos, after.coverage) == (1, 1, 0.0)


def test_placeholder_only_result_removes_the_processed_record_so_recovery_rejudges(
    tmp_path, rules
):
    """Ruling T4-5: a selection failure after a success leaves no stale record behind."""
    videos, calib, out = _clip_setup(tmp_path)
    _run(videos, calib, out, rules)
    template = calib / "scene.json"
    saved = template.read_bytes()
    template.unlink()  # selection fails once
    _run(videos, calib, out, rules)
    assert [r["review_reasons"] for r in json.loads((out / "rounds.json").read_text())] == [
        ["recalibrate"]]
    assert json.loads((out / "processed_videos.json").read_text()) == {}
    template.write_bytes(saved)  # selection recovers
    _run(videos, calib, out, rules)
    rows = json.loads((out / "rounds.json").read_text())
    assert rows and not any(is_not_a_round(r["review_reasons"]) for r in rows)
    sid = json.loads((out / "videos.json").read_text())["clip.mp4"]["calibration"]["id"]
    assert {r["calibration_id"] for r in rows} == {sid}
    db = json.loads((out / "processed_videos.json").read_text())
    assert [rec["calibration_id"] for rec in db.values()] == [sid]


def test_run_batch_rejects_out_overlapping_calib_before_writing(tmp_path, rules):
    videos, calib, _out = _clip_setup(tmp_path)
    before = sorted(p.name for p in calib.rglob("*"))
    for bad in (calib, calib / "results"):
        with pytest.raises(ValueError):
            _run(videos, calib, bad, rules)
    assert sorted(p.name for p in calib.rglob("*")) == before


def test_run_batch_rejects_out_that_is_a_file(tmp_path, rules):
    videos, calib, out = _clip_setup(tmp_path)
    out.write_text("x")
    with pytest.raises(ValueError):
        _run(videos, calib, out, rules)


# --------------------------------------------------------------------------- transient re-analysis failures (ruling R68)


def _status(out, video):
    return json.loads((out / "videos.json").read_text())[video].get("analysis_status")


def _break_judging(monkeypatch, mode):
    """Make every judging attempt fail transiently: ``decode`` (a decode error mid-stream),
    ``analysis`` (an unexpected exception: ``analysis_error``) or ``probe`` (ffprobe fails, so
    the batch never gets as far as judging: ``decode_error``)."""
    from bxjudge import manifest
    from bxjudge.video import VideoDecodeError

    def track_fails(*a, **k):
        raise VideoDecodeError("corrupt mid-stream") if mode == "decode" else RuntimeError("OOM")

    def probe_fails(path):
        raise ValueError(f"ffprobe failed on {path}")

    if mode == "probe":
        monkeypatch.setattr(manifest, "probe", probe_fails)
    else:
        monkeypatch.setattr(pipeline, "track_frames", track_fails)


@pytest.mark.parametrize("mode, status", [("decode", "decode_error"), ("probe", "decode_error"),
                                          ("analysis", "analysis_error")])
def test_transient_failure_of_a_judged_video_keeps_its_rows_and_is_retried(
    tmp_path, rules, monkeypatch, mode, status
):
    from bxjudge.output import read_rounds, save_rounds
    from tests.helpers import add_manual_calibration, stadium_scene

    videos, calib, out = _clip_setup(tmp_path)
    _run(videos, calib, out, rules)
    assert _status(out, "clip.mp4") == "ok"
    rows = read_rounds(out)
    rows[0]["audit_sample"] = True
    save_rounds(rows, out, rules)
    _freeze_audit(out)
    before_rows = (out / "rounds.json").read_bytes()
    before_db = (out / "processed_videos.json").read_bytes()
    before_entry = json.loads((out / "videos.json").read_text())["clip.mp4"]
    old_id = before_entry["calibration"]["id"]
    # A new manual calibration makes the video due for reanalysis (spec 6.3b item 4).
    add_manual_calibration(calib, stadium_scene(shift=(45, 20)), name="moved")
    with monkeypatch.context() as m:
        _break_judging(m, mode)
        _run(videos, calib, out, rules)
    assert (out / "rounds.json").read_bytes() == before_rows  # incl. audit_sample, calibration_id
    kept = json.loads(before_rows)
    assert kept[0]["audit_sample"] is True and kept[0]["calibration_id"] == old_id
    assert (out / "processed_videos.json").read_bytes() == before_db  # left for retry
    entry = json.loads((out / "videos.json").read_text())["clip.mp4"]
    # R66a: only the status, the attempt's own wall and the current file's probed duration
    # (none when the probe failed) change; the rows' judged_wall_s and timing stay.
    expected = {**before_entry, "analysis_status": status,
                "attempt_wall_s": entry["attempt_wall_s"]}
    if mode == "probe":
        del expected["duration_s"]
    assert entry == expected
    assert entry["attempt_wall_s"] > 0 and entry["processing_s"] == before_entry["processing_s"]
    assert entry["judged_wall_s"] == before_entry["judged_wall_s"]
    spy = _JudgeSpy(monkeypatch)
    _run(videos, calib, out, rules)  # the failure has cleared: retried
    assert spy.videos == {"clip.mp4"}
    assert _status(out, "clip.mp4") == "ok"
    rows = json.loads((out / "rounds.json").read_text())
    new_id = json.loads((out / "videos.json").read_text())["clip.mp4"]["calibration"]["id"]
    assert new_id != old_id and rows and {r["calibration_id"] for r in rows} == {new_id}


def test_transient_failure_of_a_vetoed_group_keeps_its_original_predictions(
    tmp_path, rules, monkeypatch
):
    """R68 + T4-3: a usable non-vetoed alternative triggers reanalysis, the reanalysis fails:
    the vetoed rows (unreviewed predictions, audit flag, calibration id) stay untouched."""
    from tests.helpers import add_manual_calibration, stadium_scene

    videos, calib, out, sid = _veto_setup(tmp_path, rules)
    before_rows = (out / "rounds.json").read_bytes()
    before_db = (out / "processed_videos.json").read_bytes()
    before = _eval(out)
    add_manual_calibration(calib, stadium_scene(shift=(45, 20)), name="moved")
    with monkeypatch.context() as m:
        _break_judging(m, "decode")
        _run(videos, calib, out, rules)
    assert (out / "rounds.json").read_bytes() == before_rows
    clip2 = next(r for r in json.loads(before_rows) if r["video"] == "clip2.mp4")
    assert clip2["audit_sample"] is True and clip2["calibration_id"] == sid
    assert clip2["reviewed"] is False
    assert (out / "processed_videos.json").read_bytes() == before_db
    v = json.loads((out / "videos.json").read_text())
    assert v["clip2.mp4"]["analysis_status"] == "decode_error"
    assert v["clip2.mp4"]["calibration"]["id"] == sid
    after = _eval(out)
    assert (after.n_rounds, after.n_vetoed_rounds, after.review_ratio) == (
        before.n_rounds, before.n_vetoed_rounds, before.review_ratio)
    assert after.n_decode_error_videos == 2 and after.n_recalibrate_videos == 0


def test_first_ever_decode_failure_still_writes_the_placeholder(tmp_path, rules, monkeypatch):
    videos, calib, out = _clip_setup(tmp_path)
    _break_judging(monkeypatch, "decode")
    _run(videos, calib, out, rules)
    assert [r["review_reasons"] for r in json.loads((out / "rounds.json").read_text())] == [
        ["decode_error"]]
    assert _status(out, "clip.mp4") == "decode_error"
    db = out / "processed_videos.json"
    assert not db.exists() or json.loads(db.read_text()) == {}  # T4-5: retried as new


def test_deliberate_recalibrate_of_a_judged_video_is_still_a_placeholder(tmp_path, rules):
    """R68 covers transient failures only: selection returning None follows spec 6.3b."""
    videos, calib, out = _clip_setup(tmp_path)
    _run(videos, calib, out, rules)
    (calib / "scene.json").unlink()
    _run(videos, calib, out, rules)
    assert [r["review_reasons"] for r in json.loads((out / "rounds.json").read_text())] == [
        ["recalibrate"]]
    assert _status(out, "clip.mp4") == "recalibrate"


def test_record_video_status_accepts_only_the_four_statuses(tmp_path):
    from bxjudge.output import record_video_status

    for status in ("ok", "recalibrate", "decode_error", "analysis_error"):
        record_video_status(tmp_path, "a.mp4", status)
        assert _status(tmp_path, "a.mp4") == status
    for bad in ("success", "no_round", None):
        with pytest.raises(ValueError, match="analysis_status"):
            record_video_status(tmp_path, "a.mp4", bad)


def test_unchanged_video_after_a_transient_failure_is_rejudged_not_marked_ok(
    tmp_path, rules, monkeypatch
):
    """C-X2(a) / Codex T4 reproduction: a probe failure of a judged video (status
    decode_error, record kept); next run the first second selects the recorded calibration
    again. The video must be re-judged (R68 retry), never flipped to ok with zero judge calls."""
    videos, calib, out = _clip_setup(tmp_path)
    _run(videos, calib, out, rules)
    with monkeypatch.context() as m:
        _break_judging(m, "probe")
        _run(videos, calib, out, rules)
    assert _status(out, "clip.mp4") == "decode_error"
    spy = _JudgeSpy(monkeypatch)
    _run(videos, calib, out, rules)
    assert spy.videos == {"clip.mp4"}
    assert _status(out, "clip.mp4") == "ok"
