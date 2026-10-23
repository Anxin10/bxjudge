"""Ruling R66: speed passes only when every measured real video finishes within its own
playback time (observed per-video ``wall_s``); batch and median ratios are diagnostics."""
import json
import math
from pathlib import Path

import pytest

from bxjudge import pipeline
from bxjudge.cli import main
from bxjudge.evaluate import (
    evaluate,
    load_eval_config,
    passes,
    read_results,
    report_metrics,
)
from bxjudge.manifest import VideoInfo
from bxjudge.output import record_video_timing
from tests.helpers import rev, rr

EVAL_CONFIG = Path(__file__).resolve().parents[1] / "config" / "evaluation.yaml"


def _cfg():
    return load_eval_config(EVAL_CONFIG)


def _entry(duration, wall, *, processing=None, status="ok", vlm=None):
    e = {"duration_s": duration, "processing_s": processing if processing is not None else wall,
         "analysis_status": status}
    if wall is not None:
        e["judged_wall_s"] = wall + (vlm or 0.0)
        e["attempt_wall_s"] = wall + (vlm or 0.0)
    if vlm is not None:
        e["vlm_s"] = vlm
    return e


def _runs(timings):
    return [{"kind": "run", "started_at": "t",
             "wall_s": sum(e.get("attempt_wall_s", 1.0) for e in timings.values()),
             "vlm_wall_s": None, "videos": {v: e["duration_s"] for v, e in timings.items()}}]


def _rows(*videos):
    return [rev(video=v, round=1) for v in videos]


def _ten_with_one_slow():
    videos = [f"v{i}.mp4" for i in range(10)]
    timings = {v: _entry(100.0, 40.0 + i) for i, v in enumerate(videos[:9])}
    timings["v9.mp4"] = _entry(100.0, 130.0)  # ratio 1.3
    return _rows(*videos), timings


def test_one_slow_video_among_fast_ones_fails_and_is_named():
    rows, timings = _ten_with_one_slow()
    # the pooled batch is well within playback time: it must not hide the outlier
    runs = [{"kind": "run", "started_at": "t", "wall_s": 600.0, "vlm_wall_s": None,
             "videos": {v: 100.0 for v in timings}}]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "worst-video+batch"
    assert m.speed_ratio == pytest.approx(1.3)
    assert m.speed_slowest_video == "v9.mp4"
    assert m.n_wall_videos == 10 and m.n_wall_missing == 0
    assert passes(m)["speed_ratio"] == "FAIL"
    # diagnostics stay: pooled batch ratio and the median per-video ratio
    assert m.speed_ratio_wall == pytest.approx(0.6)
    assert m.speed_ratio_median == pytest.approx(0.445)


def test_every_video_within_playback_passes_boundary_inclusive():
    rows = _rows("a.mp4", "b.mp4")
    timings = {"a.mp4": _entry(100.0, 100.0), "b.mp4": _entry(50.0, 10.0)}
    m = evaluate(rows, timings, runs=_runs(timings))
    assert m.speed_basis == "worst-video+batch" and m.speed_ratio == pytest.approx(1.0)
    assert m.speed_slowest_video == "a.mp4"
    assert passes(m)["speed_ratio"] == "PASS"


def test_a_real_video_without_wall_s_makes_speed_unverified():
    rows = _rows("a.mp4", "b.mp4")
    timings = {"a.mp4": _entry(100.0, 10.0), "b.mp4": _entry(100.0, None, processing=10.0)}
    m = evaluate(rows, timings, runs=_runs(timings))
    assert m.speed_basis == "unverified"
    assert m.n_wall_videos == 1 and m.n_wall_missing == 1
    assert passes(m)["speed_ratio"] == "FAIL"  # unverified never passes
    r = report_metrics(rows, timings, _cfg(), runs=_runs(timings))
    assert r.observed["missing"] == ["b.mp4"]


def test_a_real_video_without_analysis_status_is_unverified():
    timings = {"a.mp4": _entry(100.0, 10.0)}
    del timings["a.mp4"]["analysis_status"]
    m = evaluate(_rows("a.mp4"), timings, runs=_runs(timings))
    assert m.speed_basis == "unverified" and passes(m)["speed_ratio"] == "FAIL"


def test_a_real_video_without_any_timing_entry_is_unverified():
    m = evaluate(_rows("a.mp4", "b.mp4"), {"a.mp4": _entry(100.0, 10.0)})
    assert m.speed_basis == "unverified" and m.n_wall_missing == 1


def test_failed_latest_attempt_counts_its_own_wall_not_the_old_timing():
    # R68 kept the old rows and processing_s (20 s); the failed attempt itself took 150 s.
    rows = _rows("a.mp4", "b.mp4")
    timings = {
        "a.mp4": _entry(100.0, 150.0, processing=20.0, status="decode_error", vlm=5.0),
        "b.mp4": _entry(100.0, 30.0),
    }
    m = evaluate(rows, timings, runs=_runs(timings))
    assert m.speed_basis == "unverified"  # failed current attempt never certifies speed
    assert m.speed_ratio_worst_observed == pytest.approx(1.55)  # the attempt's wall, without the old VLM time
    assert m.speed_slowest_video == "a.mp4"
    assert passes(m)["speed_ratio"] == "FAIL"


def test_ok_video_observed_time_includes_its_vlm_phase():
    timings = {"a.mp4": _entry(100.0, 60.0, vlm=50.0)}
    m = evaluate(_rows("a.mp4"), timings, runs=_runs(timings))
    assert m.speed_ratio == pytest.approx(1.1) and passes(m)["speed_ratio"] == "FAIL"


def test_placeholder_only_and_over_limit_videos_are_not_required():
    rows = _rows("a.mp4", "long.mp4") + [
        rev(video="gate.mp4", finish=None, winner=None, review_reasons=["recalibrate"])]
    timings = {"a.mp4": _entry(100.0, 50.0), "long.mp4": _entry(900.0, None, processing=5.0),
               "gate.mp4": _entry(100.0, None, processing=1.0, status="recalibrate")}
    m = evaluate(rows, timings, runs=_runs(timings))
    assert m.speed_basis == "worst-video+batch" and m.speed_ratio == pytest.approx(0.52)
    assert m.n_wall_videos == 1 and m.n_wall_missing == 0


def test_report_and_cli_name_the_slowest_video(tmp_path, capsys):
    rows, timings = _ten_with_one_slow()
    res = tmp_path / "res"
    res.mkdir()
    (res / "rounds.json").write_text(json.dumps(rows), encoding="utf-8")
    (res / "videos.json").write_text(json.dumps(timings), encoding="utf-8")
    (res / "runs.json").write_text(json.dumps(_runs(timings)), encoding="utf-8")
    report = tmp_path / "eval.md"
    assert main(["eval", str(res), "--config", str(EVAL_CONFIG), "--report", str(report)]) == 0
    summary = report.read_text(encoding="utf-8").split("\n## ", 1)[0]
    assert "worst-video+batch" in summary and "v9.mp4" in summary
    assert "slowest" in capsys.readouterr().out


def test_record_video_timing_stores_wall_s(tmp_path):
    record_video_timing(tmp_path, "a.mp4", 10.0, 2.0, judged_wall_s=3.0)
    entry = json.loads((tmp_path / "videos.json").read_text())["a.mp4"]
    assert entry["judged_wall_s"] == entry["attempt_wall_s"] == 3.0 and entry["processing_s"] == 2.0


# --------------------------------------------------------------------------- run_batch


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_run_batch_records_wall_s_for_each_video(tmp_path, monkeypatch, th, rules):
    clock = Clock()
    monkeypatch.setattr(pipeline.time, "perf_counter", clock)
    folder = tmp_path / "videos"
    folder.mkdir()
    calib = tmp_path / "calib"
    calib.mkdir()
    out = tmp_path / "out"
    paths = [folder / n for n in ("a.mp4", "b.mp4")]
    for p in paths:
        p.touch()
    infos = {p.name: VideoInfo(p, 1, 1, 100.0, 30, 8, 8, "h264", False, None) for p in paths}
    work = {"a.mp4": 2.0, "b.mp4": 3.0}
    fail = set()

    def judge(info, *a, **k):
        clock.now += work[info.path.name]
        if info.path.name in fail:
            raise RuntimeError("OOM")
        return [rr("spin", "X", video=info.path.name)]

    def run():
        pipeline.run_batch(folder, calib, out, detector=lambda *a: [], th=th, rules=rules)

    monkeypatch.setattr(pipeline, "judge_video", judge)
    monkeypatch.setattr(pipeline, "scan", lambda *a, **k: list(infos.values()))
    monkeypatch.setattr(pipeline, "processed_videos", lambda *a, **k: [])
    run()
    _, timings, _ = read_results(out)
    assert timings["a.mp4"]["judged_wall_s"] == 2.0 and timings["b.mp4"]["judged_wall_s"] == 3.0

    # Rerun: a.mp4 is due for reanalysis; its selection pass (1 s) counts in its wall.
    def first_second(info, th):
        clock.now += 1.0

    monkeypatch.setattr(pipeline, "scan", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "processed_videos", lambda *a, **k: [infos["a.mp4"]])
    monkeypatch.setattr(pipeline, "first_second_frame", first_second)
    monkeypatch.setattr(pipeline, "select_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "needs_reanalysis", lambda *a: True)
    work["a.mp4"] = 4.0
    run()
    _, timings, _ = read_results(out)
    assert timings["a.mp4"]["judged_wall_s"] == 5.0
    assert timings["b.mp4"]["judged_wall_s"] == 3.0  # not attempted: untouched

    # A transient failure keeps the old rows and processing_s but records the attempt's wall.
    fail.add("a.mp4")
    work["a.mp4"] = 6.0
    old_processing = timings["a.mp4"]["processing_s"]
    run()
    _, timings, _ = read_results(out)
    entry = timings["a.mp4"]
    assert entry["analysis_status"] == "analysis_error"
    assert entry["processing_s"] == old_processing
    assert entry["attempt_wall_s"] == 7.0 and entry["judged_wall_s"] == 5.0
    assert math.isfinite(entry["judged_wall_s"])
