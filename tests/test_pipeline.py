"""Single-pass pipeline, uncertainty gates and batch isolation."""

import json
from pathlib import Path

import numpy as np
import pytest

from bxjudge import pipeline
from bxjudge.finish import FinishResult
from bxjudge.manifest import VideoInfo
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


def calibrated(monkeypatch, tmp_path, *, tracks=None):
    c = cal_with(make_ref_img(), tmp_path)
    frame = np.zeros((720, 1280, 3), np.uint8)
    calls = []

    def frames(*args, **kwargs):
        calls.append(kwargs)
        for f in range(120):
            yield f, frame

    monkeypatch.setattr(pipeline, "iter_frames", frames)
    monkeypatch.setattr(pipeline, "find_calibration", lambda frame, *a: c)

    def measured(frames, scaled, ref):
        assert scaled.frame_size == (1280, 720)
        assert scaled.battle_area == [(240, 120), (1040, 120), (1040, 560), (240, 560)]
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


def test_recalibrate_closes_single_decode(monkeypatch, th, rules):
    closed = []

    def frames(*a, **k):
        try:
            for f in range(120):
                yield f, np.zeros((16, 32, 3), np.uint8)
        finally:
            closed.append(True)

    monkeypatch.setattr(pipeline, "iter_frames", frames)
    monkeypatch.setattr(pipeline, "find_calibration", lambda *a: None)
    [r] = pipeline.judge_video(info(), Path("calib"), lambda *a: [], rules, th)
    assert r.review_reasons == ["recalibrate"] and r.needs_review
    assert closed == [True]


def test_no_round_one_decode_and_resolution_alignment(monkeypatch, tmp_path, th, rules):
    calls = calibrated(monkeypatch, tmp_path)
    [r] = pipeline.judge_video(info(), tmp_path, lambda *a: [], rules, th)
    assert r.review_reasons == ["no_round"]
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
        info(), tmp_path, lambda *a: [], rules, th, out_dir=tmp_path / "out"
    )
    assert r.needs_review == review and f"X_stop_{reason}" in r.review_reasons
    assert (
        r.t_end == 70 / 30
        and r.points == rules.finish_points["spin"]
        and r.winner == "X"
    )
    assert r.motion_evidence["window"] == [0, 21]
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
    [r] = pipeline.judge_video(info(), tmp_path, lambda *a: [], rules, th)
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
    calibrated(monkeypatch, tmp_path, tracks=tracks)
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
    [r] = pipeline.judge_video(info(), tmp_path, lambda *a: [], rules, th)
    return r


def test_confident_known_round_with_informational_notes_is_not_review(
    monkeypatch, tmp_path, th, rules
):
    r = _gate_round(monkeypatch, tmp_path, th, rules)
    assert r.finish == "spin" and r.confidence >= th.review_confidence
    assert "shake_by_bey_impact@5" in r.review_reasons
    assert "X_stop_still" in r.review_reasons
    assert r.needs_review is False


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
        info(), tmp_path, lambda *a: [], rules, th, provenance=provenance
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
            tmp_path, tmp_path, out, detector=lambda *a: [], rules=rules, th=th
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
        tmp_path, tmp_path, out, detector=lambda *a: [], rules=rules, th=th
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
            tmp_path, tmp_path, tmp_path / "out", selection=tmp_path / "absent.json"
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
        pipeline.run_batch(tmp_path, tmp_path, tmp_path / "out", thresholds=thresholds)
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
        tmp_path, tmp_path, out, detector=lambda *a: [], rules=rules, th=th
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
    pipeline.judge_video(info(), tmp_path, detector, rules, th)
    assert observed == list(range(120))
    assert crop_frames == [set(range(120)), set(range(120))]


def test_decode_failure_returns_review_artifact(monkeypatch, th, rules):
    from bxjudge.video import VideoDecodeError

    def broken(*a, **kw):
        raise VideoDecodeError("corrupt")
        yield

    monkeypatch.setattr(pipeline, "iter_frames", broken)
    [r] = pipeline.judge_video(info(), Path("calib"), lambda *a: [], rules, th)
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
        info(), tmp_path, lambda *a: [], rules, th, out_dir=tmp_path / "out"
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
    results = pipeline.judge_video(info(), tmp_path, lambda *a: [], rules, th)
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
    [r] = pipeline.judge_video(info(), tmp_path, lambda *a: [], rules, th)
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
        tmp_path, tmp_path, tmp_path / "out", selection=selection, th=th, rules=rules
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
            tmp_path, tmp_path, tmp_path / "out", detection_config=config
        )
    assert not (tmp_path / "out").exists()


def test_case_variant_video_artifact_collision_fails_preflight(tmp_path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "a.MP4").touch()
    with pytest.raises(ValueError, match="basename|artifact"):
        pipeline.run_batch(tmp_path, tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- VLM phase (Task 14)


def _vlm_config(tmp_path, **change):
    import yaml

    raw = yaml.safe_load(Path("config/vlm.yaml").read_text(encoding="utf-8"))
    raw.update(change)
    path = tmp_path / "vlm.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


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
        tmp_path,
        out,
        selection=_synthetic_selection(tmp_path),
        vlm_config=_vlm_config(tmp_path),
    )
    [(out_dir, kw)] = phases
    assert out_dir == out and kw["videos"] == {"a.mp4"}
    assert kw["calib_dir"] == tmp_path and kw["config"].model == "qwen3.8:27b"


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
        tmp_path,
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
            tmp_path,
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
            tmp_path,
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
