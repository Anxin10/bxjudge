"""Observed same-invocation latency through public pipeline and saved evaluation."""
import json
from dataclasses import replace

import pytest

from bxjudge import pipeline, vlm
from bxjudge.cli import main
from bxjudge.evaluate import (
    evaluate,
    load_eval_config,
    passes,
    read_results,
    speed_verdict,
)
from bxjudge.manifest import VideoInfo
from bxjudge.output import record_video_timing, save_rounds, write_results
from tests.helpers import rev, rr
from tests.test_pipeline import _vlm_config


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def batch(tmp_path, monkeypatch, th, rules, *, enabled=True, candidates=True, unavailable=False):
    clock = Clock()
    monkeypatch.setattr(pipeline.time, "perf_counter", clock)
    paths = [tmp_path / name for name in ("a.mp4", "b.mp4")]
    for path in paths:
        path.touch()
    infos = [VideoInfo(p, 1, 1, 100, 30, 8, 8, "h264", False, None) for p in paths]
    def scan(*a, **k):
        clock.advance(7)  # shared probe is outer-wall only
        return infos
    monkeypatch.setattr(pipeline, "scan", scan)
    def judge(info, *a, **k):
        clock.advance(2 if info.path.name == "a.mp4" else 3)
        return [replace(rr("spin", "X", video=info.path.name), needs_review=candidates)]
    monkeypatch.setattr(pipeline, "judge_video", judge)
    def release():
        clock.advance(4)
    monkeypatch.setattr(pipeline, "_release_gpu_memory", release)
    def preflight(config):
        assert clock.now >= (16 if enabled else 12)  # release before embedded VLM
        clock.advance(5)
        if unavailable:
            raise vlm.VlmError("offline")
    monkeypatch.setattr(vlm, "preflight", preflight)
    def note(r, get_frames, session):
        if not session.available():
            return vlm.UNAVAILABLE_PREFIX + session.unavailable
        clock.advance(6 if r.video == "a.mp4" else 8)
        return "test advisory"
    monkeypatch.setattr(vlm, "_note", note)
    config = _vlm_config(tmp_path) if enabled else None
    out = tmp_path / "out"
    def run():
        pipeline.run_batch(tmp_path, tmp_path, out, detector=lambda *a: [], th=th,
                           rules=rules, vlm_config=config)
    return clock, out, run


def observed(timings, video):
    return timings[video]["pipeline_wall"]


def test_two_phase_span_includes_other_core_release_preflight_and_wait(tmp_path, monkeypatch, th, rules):
    clock, out, run = batch(tmp_path, monkeypatch, th, rules)
    run()
    rows, timings, runs = read_results(out)
    assert observed(timings, "a.mp4") == {"scope": "core_to_final_output", "seconds": 20.0}
    assert observed(timings, "b.mp4")["seconds"] == 26.0
    assert timings["a.mp4"]["processing_s"] + timings["a.mp4"]["vlm_s"] == 13
    assert runs[0]["wall_s"] == clock.now == 35
    assert runs[0]["vlm_wall_s"] == 23
    metrics = evaluate(rows, timings, runs=runs)
    assert metrics.speed_basis == "observed per-video wall"
    assert metrics.speed_ratio == 0.26 and passes(metrics)["speed_ratio"]
    report = tmp_path / "eval.md"
    main(["eval", str(out), "--report", str(report)])
    assert "core_to_final_output" in report.read_text()
    assert "observed per-video wall" in report.read_text()


@pytest.mark.parametrize("enabled,candidates", [(False, True), (True, False)])
def test_no_applicable_vlm_finishes_at_core_output(tmp_path, monkeypatch, th, rules, enabled, candidates):
    _, out, run = batch(tmp_path, monkeypatch, th, rules, enabled=enabled, candidates=candidates)
    run()
    _, timing, _ = read_results(out)
    assert observed(timing, "a.mp4")["seconds"] == 2
    assert observed(timing, "b.mp4")["seconds"] == 3


def test_unavailable_preflight_finishes_all_pending_notes(tmp_path, monkeypatch, th, rules):
    real_note = vlm._note
    _, out, run = batch(tmp_path, monkeypatch, th, rules, unavailable=True)
    monkeypatch.setattr(vlm, "_note", real_note)
    monkeypatch.setattr(vlm, "event_frames", lambda *a, **k: pytest.fail("offline decoded frames"))
    run()
    rows, timing, _ = read_results(out)
    assert all(r["vlm_note"].startswith(vlm.UNAVAILABLE_PREFIX) for r in rows)
    assert observed(timing, "a.mp4")["seconds"] == 14
    assert observed(timing, "b.mp4")["seconds"] == 12


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failed_phase_never_retains_prior_observation(tmp_path, monkeypatch, th, rules, error):
    _, out, run = batch(tmp_path, monkeypatch, th, rules)
    run()
    def broken(r, *a):
        if r.video == "b.mp4":
            raise error("stop")
        return "fresh advisory"
    monkeypatch.setattr(vlm, "_note", broken)
    with pytest.raises(error):
        run()
    rows, timing, _ = read_results(out)
    assert observed(timing, "a.mp4")["seconds"] is not None
    assert observed(timing, "b.mp4")["seconds"] is None
    assert not passes(evaluate(rows, timing))["speed_ratio"]


def test_interruption_before_new_core_output_invalidates_old_measurement(tmp_path, monkeypatch, th, rules):
    _, out, run = batch(tmp_path, monkeypatch, th, rules)
    run()
    monkeypatch.setattr(pipeline, "judge_video", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        run()
    _, timing, _ = read_results(out)
    assert observed(timing, "a.mp4")["seconds"] is None


def test_standalone_vlm_cannot_extend_or_fabricate_pipeline_span(tmp_path, monkeypatch, th, rules):
    _, out, run = batch(tmp_path, monkeypatch, th, rules, enabled=False)
    run()
    vlm.review_results_dir(out, calib_dir=tmp_path, th=th, rules=rules)
    rows, timing, _ = read_results(out)
    assert all(observed(timing, v)["seconds"] is None for v in timing)
    assert not passes(evaluate(rows, timing))["speed_ratio"]


@pytest.mark.parametrize("other", [None, {"duration_s": 100, "processing_s": 1}])
def test_observed_violation_is_not_hidden_by_missing_or_stage_only_video(other):
    timings = {"short.mp4": {"duration_s": 10, "processing_s": 1,
               "pipeline_wall": {"scope": "core_to_final_output", "seconds": 20}}}
    if other is not None:
        timings["long.mp4"] = other
    rows = [rev(video=v) for v in ("short.mp4", "long.mp4")]
    m = evaluate(rows, timings)
    assert not passes(m)["speed_ratio"]
    assert speed_verdict(m, load_eval_config()).startswith("FAIL")


@pytest.mark.parametrize("observation", [None, {"scope": "standalone_vlm", "seconds": 1},
                                         {"scope": "core_to_final_output", "seconds": -1},
                                         {"scope": "core_to_final_output", "seconds": True}])
def test_missing_or_invalid_observation_cannot_pass(tmp_path, rules, observation):
    out = tmp_path / "saved"
    write_results([rr("spin", "X", video="a.mp4")], out, rules)
    record_video_timing(out, "a.mp4", 100, 1)
    if observation is not None:
        path = out / "videos.json"
        timings = json.loads(path.read_text())
        timings["a.mp4"]["pipeline_wall"] = observation
        path.write_text(json.dumps(timings))
    rows, timings, _ = read_results(out)
    assert not passes(evaluate(rows, timings))["speed_ratio"]


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_standalone_failed_phase_invalidates_all_scheduled_observations(tmp_path, monkeypatch, th, rules, error):
    _, out, run = batch(tmp_path, monkeypatch, th, rules, enabled=False)
    run()
    def broken(*a):
        raise error("interrupted standalone phase")
    monkeypatch.setattr(vlm, "_note", broken)
    with pytest.raises(error):
        vlm.review_results_dir(out, calib_dir=tmp_path, th=th, rules=rules)
    rows, timings, _ = read_results(out)
    assert all(observed(timings, v)["seconds"] is None for v in timings)
    assert not passes(evaluate(rows, timings))["speed_ratio"]


def test_complete_observations_compare_each_video_not_pooled_or_median():
    rows = [rev(video=v) for v in ("short.mp4", "long.mp4")]
    timings = {v: {"duration_s": d, "processing_s": 1,
                  "pipeline_wall": {"scope": "core_to_final_output", "seconds": s}}
               for v, d, s in (("short.mp4", 10, 20), ("long.mp4", 100, 1))}
    metrics = evaluate(rows, timings)
    assert metrics.speed_ratio == 2
    assert speed_verdict(metrics, load_eval_config()) == "FAIL"
    assert not passes(metrics)["speed_ratio"]
    del timings["short.mp4"]["pipeline_wall"]
    metrics = evaluate(rows, timings)
    assert metrics.speed_basis == "partial observed per-video wall"
    assert speed_verdict(metrics, load_eval_config()) == "UNVERIFIED (FAIL)"


def test_real_elapsed_lightweight_pipeline_smoke(tmp_path, monkeypatch, th, rules):
    # Real monotonic clock, output I/O and public orchestration; no decoder/model/service.
    path = tmp_path / "a.mp4"
    path.touch()
    monkeypatch.setattr(pipeline, "scan", lambda *a, **k: [
        VideoInfo(path, 1, 1, 100, 30, 8, 8, "h264", False, None)])
    monkeypatch.setattr(pipeline, "judge_video", lambda *a, **k: [rr("spin", "X", video=path.name)])
    out = tmp_path / "out"
    pipeline.run_batch(tmp_path, tmp_path, out, detector=lambda *a: [], th=th, rules=rules)
    rows, timings, runs = read_results(out)
    span = observed(timings, path.name)["seconds"]
    assert runs[0]["wall_s"] >= span >= timings[path.name]["processing_s"] > 0
    assert passes(evaluate(rows, timings))["speed_ratio"]


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf"), True, "1"])
def test_observation_writer_rejects_invalid_seconds(tmp_path, seconds):
    from bxjudge.output import record_pipeline_timing
    with pytest.raises(ValueError):
        record_pipeline_timing(tmp_path, "a.mp4", seconds)
    assert not (tmp_path / "videos.json").exists()


def test_standalone_skipped_videos_keep_their_completed_pipeline_observation(tmp_path, monkeypatch, th, rules):
    _, out, run = batch(tmp_path, monkeypatch, th, rules, candidates=False)
    run()
    _, before, _ = read_results(out)
    summary = vlm.review_results_dir(out, calib_dir=tmp_path, th=th, rules=rules)
    _, after, _ = read_results(out)
    assert summary["asked"] == 0
    assert before == after


def test_observed_coverage_excludes_over_limit_and_placeholder_only_videos():
    rows = [rev(video="a.mp4"), rev(video="long.mp4"),
            rev(video="bad.mp4", review_reasons=["decode_error"])]
    timings = {"a.mp4": {"duration_s": 10, "processing_s": 1,
               "pipeline_wall": {"scope": "core_to_final_output", "seconds": 2}},
               "long.mp4": {"duration_s": load_eval_config().speed_max_video_s + 1}}
    metrics = evaluate(rows, timings)
    assert metrics.n_observed_videos == metrics.n_required_speed_videos == 1
    assert passes(metrics)["speed_ratio"]


@pytest.mark.parametrize("reviewed_videos", [("a.mp4", "b.mp4"), ("a.mp4",)])
@pytest.mark.parametrize("preserved_note", [None, "preserved successful advisory"])
def test_rerun_preserved_reviewed_rows_complete_at_actual_core_output(
    tmp_path, monkeypatch, th, rules, reviewed_videos, preserved_note,
):
    _, out, run = batch(tmp_path, monkeypatch, th, rules)
    run()
    rows, _, _ = read_results(out)
    for row in rows:
        if row["video"] in reviewed_videos:
            row["reviewed"] = True
            row["reviewer_fix"] = {"winner": "B"}
            row["vlm_note"] = preserved_note
    save_rounds(rows, out, rules)
    preserved = [row for row in rows if row["video"] in reviewed_videos]
    run()  # fresh judge still requests VLM; writer preserves saved reviewed rows
    after, timings, runs = read_results(out)
    assert [row for row in after if row["video"] in reviewed_videos] == preserved
    assert observed(timings, "a.mp4")["seconds"] == 2
    assert observed(timings, "b.mp4")["seconds"] == (3 if "b.mp4" in reviewed_videos else 20)
    assert ("vlm_s" not in timings["a.mp4"])
    assert runs[-1]["wall_s"] == (16 if "b.mp4" in reviewed_videos else 29)
    metrics = evaluate(after, timings)
    assert metrics.n_observed_videos == metrics.n_required_speed_videos == 2
    assert passes(metrics)["speed_ratio"]
