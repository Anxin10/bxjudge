"""Rulings R66/R66a/R66b: speed passes only when every measured real video finishes within
its own playback time (``max(judged_wall_s + vlm_s, attempt_wall_s)`` over the duration of the
current source file) AND the pooled batch ratio (batch wall / measured duration) is within it.
"""
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
    read_audit,
    read_results,
    report_metrics,
)
from bxjudge.manifest import VideoInfo
from bxjudge.output import record_video_status, record_video_timing, record_vlm_timing
from tests.helpers import rev, rr

ROOT = Path(__file__).resolve().parents[1]
EVAL_CONFIG = ROOT / "config" / "evaluation.yaml"


def _cfg():
    return load_eval_config(EVAL_CONFIG)


def _entry(duration, judged, *, attempt=None, processing=None, status="ok", vlm=None,
           run_id="r1"):
    e = {"duration_s": duration,
         "processing_s": processing if processing is not None else (judged or 1.0),
         "analysis_status": status}
    if judged is not None:
        e["judged_wall_s"] = judged
        e["judgment_sources"] = [{"rounds": [1], "run_id": run_id, "wall_s": judged,
                                  "processing_s": e["processing_s"]}]
    attempt = judged if attempt is None else attempt
    if attempt is not None:
        e["attempt_wall_s"], e["attempt_run_id"] = attempt, run_id
    if vlm is not None:
        e["vlm_s"], e["vlm_run_ids"] = vlm, [run_id]
    return e


def _rows(*videos):
    return [rev(video=v, round=1) for v in videos]


def _run(wall, videos, run_id="r1", work=None, real=None):
    return {"kind": "run", "run_id": run_id, "started_at": "t", "wall_s": wall,
            "vlm_wall_s": None, "videos": dict(videos),
            "work": dict(work) if work is not None else {v: 0.0 for v in videos},
            "real": dict(real) if real is not None else {v: True for v in videos}}


def _fast_batch(timings):
    """A run record whose pooled ratio is well inside playback time (0.5)."""
    durations = {v: e["duration_s"] for v, e in timings.items()}
    return [_run(0.5 * sum(durations.values()), durations)]


def _ten_with_one_slow():
    videos = [f"v{i}.mp4" for i in range(10)]
    timings = {v: _entry(100.0, 40.0 + i) for i, v in enumerate(videos[:9])}
    timings["v9.mp4"] = _entry(100.0, 130.0)  # ratio 1.3
    return _rows(*videos), timings


# --------------------------------------------------------------------------- evaluation


def test_one_slow_video_among_fast_ones_fails_and_is_named():
    rows, timings = _ten_with_one_slow()
    runs = [_run(600.0, {v: 100.0 for v in timings})]  # the pooled batch must not hide it
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "worst-video+batch"
    assert m.speed_ratio == pytest.approx(1.3)
    assert m.speed_slowest_video == "v9.mp4"
    assert m.n_wall_videos == 10 and m.n_wall_missing == 0
    assert passes(m)["speed_ratio"] == "FAIL"
    assert m.speed_ratio_wall == pytest.approx(0.6)
    assert m.speed_ratio_median == pytest.approx(0.445)


def test_every_video_and_the_batch_within_playback_passes_boundary_inclusive():
    rows = _rows("a.mp4", "b.mp4")
    timings = {"a.mp4": _entry(100.0, 100.0), "b.mp4": _entry(50.0, 10.0)}
    m = evaluate(rows, timings, runs=[_run(150.0, {"a.mp4": 100.0, "b.mp4": 50.0})])
    assert m.speed_basis == "worst-video+batch" and m.speed_ratio == pytest.approx(1.0)
    assert m.speed_slowest_video == "a.mp4"
    assert passes(m)["speed_ratio"] == "PASS"


def test_one_video_batch_whose_overhead_exceeds_playback_fails():
    # R66a: per-batch overhead (model init, probe, release) counts: 80 s of judging fits the
    # 100 s video, but the batch took 120 s end to end.
    rows = _rows("a.mp4")
    timings = {"a.mp4": _entry(100.0, 80.0)}
    m = evaluate(rows, timings, runs=[_run(120.0, {"a.mp4": 100.0})])
    assert m.speed_ratio_worst_observed == pytest.approx(0.8)
    assert m.speed_ratio_wall == pytest.approx(1.2)
    assert m.speed_ratio == pytest.approx(1.2)
    assert passes(m)["speed_ratio"] == "FAIL"


def test_without_a_batch_record_speed_is_unverified():
    rows = _rows("a.mp4")
    m = evaluate(rows, {"a.mp4": _entry(100.0, 50.0)})
    assert m.speed_basis == "unverified" and passes(m)["speed_ratio"] == "FAIL"


def test_a_real_video_without_wall_times_makes_speed_unverified():
    rows = _rows("a.mp4", "b.mp4")
    timings = {"a.mp4": _entry(100.0, 10.0), "b.mp4": _entry(100.0, None, processing=10.0)}
    m = evaluate(rows, timings, runs=_fast_batch(timings))
    assert m.speed_basis == "unverified"
    assert m.n_wall_videos == 1 and m.n_wall_missing == 1
    assert passes(m)["speed_ratio"] == "FAIL"
    r = report_metrics(rows, timings, _cfg(), runs=_fast_batch(timings))
    assert r.observed["missing"] == ["b.mp4"]


@pytest.mark.parametrize("drop", ["judgment_sources", "attempt_wall_s", "analysis_status"])
def test_each_missing_field_makes_speed_unverified(drop):
    timings = {"a.mp4": _entry(100.0, 10.0)}
    del timings["a.mp4"][drop]
    m = evaluate(_rows("a.mp4"), timings, runs=_fast_batch(timings))
    assert m.speed_basis == "unverified" and passes(m)["speed_ratio"] == "FAIL"


def test_a_legacy_wall_s_field_is_not_a_wall_time():
    # no compatibility path: a pre-R66a entry is re-judged with --rejudge, never trusted
    timings = {"a.mp4": {"duration_s": 100.0, "processing_s": 10.0, "wall_s": 10.0,
                         "analysis_status": "ok"}}
    m = evaluate(_rows("a.mp4"), timings, runs=_fast_batch(timings))
    assert m.speed_basis == "unverified"


def test_a_real_video_without_any_timing_entry_is_unverified():
    timings = {"a.mp4": _entry(100.0, 10.0)}
    m = evaluate(_rows("a.mp4", "b.mp4"), timings, runs=_fast_batch(timings))
    assert m.speed_basis == "unverified" and m.n_wall_missing == 1


def test_slow_failed_attempt_counts_over_the_faster_kept_rows():
    rows = _rows("a.mp4", "b.mp4")
    timings = {
        "a.mp4": _entry(100.0, 20.0, attempt=150.0, status="decode_error", vlm=5.0),
        "b.mp4": _entry(100.0, 30.0),
    }
    m = evaluate(rows, timings, runs=_fast_batch(timings))
    assert m.speed_ratio == pytest.approx(1.5) and m.speed_slowest_video == "a.mp4"
    assert passes(m)["speed_ratio"] == "FAIL"


def test_fast_failed_attempt_never_hides_the_slow_rows_it_kept():
    timings = {"a.mp4": _entry(100.0, 150.0, attempt=2.0, status="analysis_error", vlm=30.0)}
    m = evaluate(_rows("a.mp4"), timings, runs=_fast_batch(timings))
    assert m.speed_ratio == pytest.approx(1.8) and passes(m)["speed_ratio"] == "FAIL"


def test_ok_video_observed_time_includes_its_vlm_phase():
    timings = {"a.mp4": _entry(100.0, 60.0, vlm=50.0)}
    m = evaluate(_rows("a.mp4"), timings, runs=_fast_batch(timings))
    assert m.speed_ratio == pytest.approx(1.1) and passes(m)["speed_ratio"] == "FAIL"


def test_placeholder_only_and_over_limit_videos_are_not_required():
    rows = _rows("a.mp4", "long.mp4") + [
        rev(video="gate.mp4", finish=None, winner=None, review_reasons=["recalibrate"])]
    timings = {"a.mp4": _entry(100.0, 50.0), "long.mp4": _entry(900.0, None, processing=5.0),
               "gate.mp4": _entry(100.0, None, processing=1.0, status="recalibrate")}
    m = evaluate(rows, timings, runs=[_run(50.0, {"a.mp4": 100.0})])
    assert m.speed_basis == "worst-video+batch" and m.speed_ratio == pytest.approx(0.5)
    assert m.n_wall_videos == 1 and m.n_wall_missing == 0


def test_report_and_cli_name_the_slowest_video(tmp_path, capsys):
    rows, timings = _ten_with_one_slow()
    res = tmp_path / "res"
    res.mkdir()
    (res / "rounds.json").write_text(json.dumps(rows), encoding="utf-8")
    (res / "videos.json").write_text(json.dumps(timings), encoding="utf-8")
    (res / "runs.json").write_text(json.dumps(_fast_batch(timings)), encoding="utf-8")
    report = tmp_path / "eval.md"
    assert main(["eval", str(res), "--config", str(EVAL_CONFIG), "--report", str(report)]) == 0
    summary = report.read_text(encoding="utf-8").split("\n## ", 1)[0]
    assert "worst-video" in summary and "v9.mp4" in summary
    assert "slowest" in capsys.readouterr().out


def test_unverified_speed_remedy_points_to_rejudge(tmp_path, capsys):
    # R66b: a plain rerun never re-judges unchanged videos; --rejudge does.
    rows = _rows("a.mp4")
    timings = {"a.mp4": _entry(100.0, None, processing=10.0)}
    res = tmp_path / "res"
    res.mkdir()
    (res / "rounds.json").write_text(json.dumps(rows), encoding="utf-8")
    (res / "videos.json").write_text(json.dumps(timings), encoding="utf-8")
    report = tmp_path / "eval.md"
    assert main(["eval", str(res), "--config", str(EVAL_CONFIG), "--report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "bxjudge run --rejudge" in out and "rerun bxjudge run)" not in out
    text = report.read_text(encoding="utf-8")
    assert "bxjudge run --rejudge" in text and "(rerun bxjudge run)" not in text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--rejudge" in readme and "重跑 `bxjudge run` 即可" not in readme


def test_record_video_timing_stores_both_wall_times(tmp_path):
    record_video_timing(tmp_path, "a.mp4", 10.0, 2.0, wall_s=3.0, produced_rounds={1})
    entry = json.loads((tmp_path / "videos.json").read_text())["a.mp4"]
    assert entry["judged_wall_s"] == entry["attempt_wall_s"] == 3.0
    assert entry["processing_s"] == 2.0 and "wall_s" not in entry
    record_video_status(tmp_path, "a.mp4", "decode_error", wall_s=0.5, duration_s=4.0)
    entry = json.loads((tmp_path / "videos.json").read_text())["a.mp4"]
    assert (entry["judged_wall_s"], entry["attempt_wall_s"]) == (3.0, 0.5)
    assert entry["duration_s"] == 4.0 and entry["processing_s"] == 2.0


# --------------------------------------------------------------------------- run_batch


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Batch:
    """``run_batch`` with a fake clock and fake decode/judge steps; ``events`` records
    (kind, video, time) for every decode, judge and result write."""

    def __init__(self, tmp_path, monkeypatch, th, rules, names=("a.mp4",), duration=100.0):
        self.clock = clock = Clock()
        monkeypatch.setattr(pipeline.time, "perf_counter", clock)
        self.folder, self.calib, self.out = tmp_path / "videos", tmp_path / "calib", tmp_path / "out"
        self.folder.mkdir()
        self.calib.mkdir()
        self.th, self.rules = th, rules
        self.infos = {}
        for n in names:
            (self.folder / n).touch()
            self.set_duration(n, duration)
        self.work = {n: 1.0 for n in names}
        self.select_s = {n: 0.0 for n in names}
        self.fail: set[str] = set()
        self.flagged = False
        self.round_numbers = [1]
        self.calibration_id = None
        self.new, self.processed = set(names), set()
        self.due = False
        self.scan_s = 0.0
        self.events: list[tuple[str, str, float]] = []
        self.judged: list[str] = []

        def judge(info, *a, **k):
            name = info.path.name
            self.events.append(("judge", name, clock.now))
            self.judged.append(name)
            clock.now += self.work[name]
            if name in self.fail:
                raise RuntimeError("OOM")
            results = [rr("spin", "X", video=name, round=n) for n in self.round_numbers]
            for r in results:
                r.needs_review = self.flagged
                r.calibration_id = self.calibration_id
            return results

        def first_second(info, th):
            self.events.append(("decode", info.path.name, clock.now))
            clock.now += self.select_s[info.path.name]

        real_write = pipeline.write_results

        def write(results, out_dir, rules):
            written = real_write(results, out_dir, rules)
            for v in {r.video for r in results}:
                self.events.append(("write", v, clock.now))
            return written

        def scan(*a, **k):
            clock.now += self.scan_s
            return [self.infos[n] for n in sorted(self.new)]

        monkeypatch.setattr(pipeline, "judge_video", judge)
        monkeypatch.setattr(pipeline, "first_second_frame", first_second)
        monkeypatch.setattr(pipeline, "write_results", write)
        monkeypatch.setattr(pipeline, "scan", scan)
        monkeypatch.setattr(pipeline, "processed_videos",
                            lambda *a, **k: [self.infos[n] for n in sorted(self.processed)])
        monkeypatch.setattr(pipeline, "select_snapshot", lambda *a, **k: None)
        monkeypatch.setattr(pipeline, "needs_reanalysis", lambda *a: self.due)

    def set_duration(self, name, duration):
        p = self.folder / name
        self.infos[name] = VideoInfo(p, 1, 1, duration, 30, 8, 8, "h264", False, None)

    def run(self, **kw):
        self.events.clear()
        self.judged.clear()
        pipeline.run_batch(self.folder, self.calib, self.out, detector=lambda *a: [],
                           th=self.th, rules=self.rules, **kw)
        self.processed |= self.new  # the fake scan reports them as processed from now on
        self.new = set()

    def timings(self):
        return json.loads((self.out / "videos.json").read_text())

    def evaluate(self):
        rows, timings, runs = read_results(self.out)
        return evaluate(rows, timings, runs=runs, audit=read_audit(self.out))


def test_run_batch_records_both_wall_times_for_each_video(tmp_path, monkeypatch, th, rules):
    b = Batch(tmp_path, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
    b.work.update({"a.mp4": 2.0, "b.mp4": 3.0})
    b.run()
    t = b.timings()
    assert (t["a.mp4"]["judged_wall_s"], t["a.mp4"]["attempt_wall_s"]) == (2.0, 2.0)
    assert (t["b.mp4"]["judged_wall_s"], t["b.mp4"]["attempt_wall_s"]) == (3.0, 3.0)
    assert "wall_s" not in t["a.mp4"]


def test_one_timer_from_first_decode_to_result_write_without_gaps(
    tmp_path, monkeypatch, th, rules
):
    # Codex T6: two 60 s selection passes, then 20 s judging; summing the separate spans gave
    # 80 s for a video whose first-decode-to-write wall time was 140 s.
    b = Batch(tmp_path, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
    b.run()
    b.due = True
    b.select_s.update({"a.mp4": 60.0, "b.mp4": 60.0})
    b.work.update({"a.mp4": 20.0, "b.mp4": 20.0})
    b.run()
    t = b.timings()
    for video in ("a.mp4", "b.mp4"):
        first = min(ts for kind, v, ts in b.events if v == video and kind in ("decode", "judge"))
        written = max(ts for kind, v, ts in b.events if v == video and kind == "write")
        assert t[video]["attempt_wall_s"] == pytest.approx(written - first), video
        assert t[video]["judged_wall_s"] == t[video]["attempt_wall_s"]
        assert t[video]["attempt_wall_s"] >= 80.0  # its selection pass + judging, never less


def test_fast_transient_failure_after_a_slow_analysis_still_fails(
    tmp_path, monkeypatch, th, rules
):
    # Task 6 review I-1: run 1 took 150 s + 30 s VLM on a 100 s video (1.8x); a rerun failing
    # after 2 s used to overwrite wall_s and PASS behind the kept slow rows.
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.work["a.mp4"] = 150.0
    b.run()
    record_vlm_timing(b.out, "a.mp4", 30.0, [])
    b.due, b.work["a.mp4"] = True, 2.0
    b.fail.add("a.mp4")
    b.run()
    t = b.timings()["a.mp4"]
    assert t["analysis_status"] == "analysis_error"
    assert (t["judged_wall_s"], t["attempt_wall_s"]) == (150.0, 2.0)
    m = b.evaluate()
    assert m.speed_ratio_worst_observed == pytest.approx(1.8)
    assert passes(m)["speed_ratio"] == "FAIL"


def test_replaced_shorter_video_failing_uses_the_current_files_duration(
    tmp_path, monkeypatch, th, rules
):
    # Codex T6: a 300 s video is replaced by a 10 s one whose analysis fails after 20 s; the
    # kept 300 s duration made the ratio 0.067 (PASS) instead of a FAIL.
    b = Batch(tmp_path, monkeypatch, th, rules, duration=300.0)
    b.work["a.mp4"] = 100.0
    b.run()
    b.set_duration("a.mp4", 10.0)
    b.processed.discard("a.mp4")
    b.new.add("a.mp4")  # changed file: scan returns it with its new probe
    b.work["a.mp4"] = 20.0
    b.fail.add("a.mp4")
    b.run()
    t = b.timings()["a.mp4"]
    assert t["analysis_status"] == "analysis_error" and t["duration_s"] == 10.0
    m = b.evaluate()
    assert m.speed_ratio_worst_observed >= 2.0
    assert passes(m)["speed_ratio"] == "FAIL"


def test_one_video_batch_overhead_above_playback_fails_end_to_end(
    tmp_path, monkeypatch, th, rules
):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.scan_s, b.work["a.mp4"] = 50.0, 80.0  # probe overhead + judging: 130 s for a 100 s video
    b.run()
    m = b.evaluate()
    assert m.speed_ratio_worst_observed == pytest.approx(0.8)
    assert m.speed_ratio_wall == pytest.approx(1.3)
    assert passes(m)["speed_ratio"] == "FAIL"


def test_rejudge_gives_unchanged_videos_their_wall_times_back(tmp_path, monkeypatch, th, rules):
    # R66b: an old results dir (no wall times) stays unverified on a plain rerun; --rejudge
    # re-analyses every video and keeps reviewed rows (R36).
    from bxjudge.corrections import confirm_reviewed

    b = Batch(tmp_path, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
    b.run()
    confirm_reviewed(b.out, "b.mp4", 1, rules=rules)
    reviewed = next(r for r in json.loads((b.out / "rounds.json").read_text())
                    if r["video"] == "b.mp4")
    old = b.timings()
    for entry in old.values():  # as written before R66a
        del entry["judgment_sources"], entry["judged_wall_s"], entry["attempt_wall_s"]
    (b.out / "videos.json").write_text(json.dumps(old), encoding="utf-8")
    b.run()
    assert b.judged == []
    assert b.evaluate().speed_basis == "unverified"
    b.work.update({"a.mp4": 5.0, "b.mp4": 6.0})
    b.run(rejudge=True)
    assert sorted(b.judged) == ["a.mp4", "b.mp4"]
    t = b.timings()
    assert t["a.mp4"]["judged_wall_s"] == 5.0
    assert t["b.mp4"]["attempt_wall_s"] == 6.0
    kept = next(r for r in json.loads((b.out / "rounds.json").read_text())
                if r["video"] == "b.mp4")
    assert kept == reviewed
    # An old reviewed prediction has no measured producer to recover by rejudging.
    assert b.evaluate().speed_basis == "unverified"


def test_run_cli_passes_rejudge(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **kw: calls.append(kw))
    assert main(["run", str(tmp_path)]) == 0
    assert calls[-1]["rejudge"] is False
    assert main(["run", str(tmp_path), "--rejudge"]) == 0
    assert calls[-1]["rejudge"] is True


def test_unchanged_video_without_an_ok_status_is_rejudged(tmp_path, monkeypatch, th, rules):
    # C-X2(a): only an `ok` latest status lets an unchanged video skip re-judging.
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.run()
    old = b.timings()
    del old["a.mp4"]["analysis_status"]  # an entry written before R68
    (b.out / "videos.json").write_text(json.dumps(old), encoding="utf-8")
    b.run()
    assert b.judged == ["a.mp4"]
    assert b.timings()["a.mp4"]["analysis_status"] == "ok"
    b.run()
    assert b.judged == []  # ok and unchanged: skipped


def test_vlm_phases_skip_a_video_whose_latest_attempt_failed(tmp_path, monkeypatch, th, rules):
    # C-X2(b) / Codex T4: with Ollama unreachable the after_run phase wrote vlm_unavailable notes
    # onto the preserved rows of a video whose reanalysis had just failed.
    import yaml

    from bxjudge import vlm

    monkeypatch.setattr(vlm, "preflight", lambda config: "server unreachable at test")
    raw = yaml.safe_load((ROOT / "config" / "vlm.yaml").read_text(encoding="utf-8"))
    raw["after_run"] = True
    cfg = tmp_path / "vlm.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.flagged = True  # review-required rows: VLM targets
    b.run()  # no VLM phase on the first run
    before = (b.out / "rounds.json").read_bytes()
    assert json.loads(before)[0]["vlm_note"] is None
    b.due = True
    b.fail.add("a.mp4")
    b.run(vlm_config=cfg)
    assert b.timings()["a.mp4"]["analysis_status"] == "analysis_error"
    assert (b.out / "rounds.json").read_bytes() == before
    assert main(["vlm", str(b.out), "--config", str(cfg)]) == 0  # the separate command too
    assert (b.out / "rounds.json").read_bytes() == before
    assert math.isfinite(b.timings()["a.mp4"]["attempt_wall_s"])


# ------------------------------------------------- C-X3(c): the batch record covers every video


def test_batch_record_covering_only_some_measured_videos_is_unverified():
    # Codex fix-2 review: a finite batch ratio from a record listing only a.mp4 must not
    # verify b.mp4, whose run never wrote a batch record (interrupted).
    timings = {"a.mp4": _entry(100.0, 40.0), "b.mp4": _entry(100.0, 40.0)}
    rows = _rows("a.mp4", "b.mp4")
    runs = [_run(50.0, {"a.mp4": 100.0})]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "unverified" and math.isnan(m.speed_ratio)
    assert passes(m)["speed_ratio"] != "PASS"
    assert report_metrics(rows, timings, _cfg(), runs=runs).tradeoff["speed"]["basis"] == "unverified"
    full = [_run(100.0, {"a.mp4": 100.0, "b.mp4": 100.0})]
    assert evaluate(rows, timings, runs=full).speed_basis == "worst-video+batch"


def test_interrupted_run_without_its_batch_record_does_not_pass(tmp_path, monkeypatch, th, rules):
    b = Batch(tmp_path, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
    b.new = {"a.mp4"}
    b.run()  # a.mp4 judged, batch record written
    assert passes(b.evaluate())["speed_ratio"] == "PASS"
    b.new = {"b.mp4"}

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "record_run_timing", interrupted)
    with pytest.raises(KeyboardInterrupt):
        b.run()  # b.mp4 judged and timed, then the run dies before its batch record
    t = b.timings()
    assert t["b.mp4"]["judged_wall_s"] > 0 and t["b.mp4"]["analysis_status"] == "ok"
    assert [set(r["videos"]) for r in json.loads((b.out / "runs.json").read_text())] == [
        {"a.mp4"}]
    m = b.evaluate()
    assert m.speed_basis == "unverified"
    assert passes(m)["speed_ratio"] != "PASS"


# ------------------------------------------------- C-X4(b): run ids link timings to batch records


def test_stale_batch_record_never_verifies_a_newer_unrecorded_timing():
    # Codex fix-3 review: a complete batch recorded a.mp4 at 50 s (run r1); a rejudge (r2)
    # updated its wall to 80 of 100 s and stopped before its batch record.
    rows = _rows("a.mp4")
    runs = [_run(50.0, {"a.mp4": 100.0}, run_id="r1")]
    timings = {"a.mp4": _entry(100.0, 80.0, run_id="r2")}
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "unverified" and math.isnan(m.speed_ratio)
    assert passes(m)["speed_ratio"] != "PASS"
    assert report_metrics(rows, timings, _cfg(), runs=runs).tradeoff["speed"]["basis"] == (
        "unverified")
    # its own complete record verifies it
    done = [*runs, _run(85.0, {"a.mp4": 100.0}, run_id="r2")]
    assert evaluate(rows, timings, runs=done).speed_basis == "worst-video+batch"


@pytest.mark.parametrize("field", ["run_id", "attempt_run_id", "vlm_run_ids"])
def test_timing_without_a_matching_run_id_is_unverified(field):
    rows = _rows("a.mp4")
    timings = {"a.mp4": _entry(100.0, 40.0, vlm=1.0)}
    runs = [_run(50.0, {"a.mp4": 100.0})]
    assert evaluate(rows, timings, runs=runs).speed_basis == "worst-video+batch"
    target = timings["a.mp4"]["judgment_sources"][0] if field == "run_id" else timings["a.mp4"]
    target[field] = ["r9"] if field == "vlm_run_ids" else "r9"
    assert evaluate(rows, timings, runs=runs).speed_basis == "unverified"
    del target[field]  # no id at all (pre-C-X4 entry): no compatibility path
    assert evaluate(rows, timings, runs=runs).speed_basis == "unverified"


def test_interrupted_rejudge_after_a_complete_batch_does_not_pass(
    tmp_path, monkeypatch, th, rules
):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.work["a.mp4"] = 50.0
    b.run()
    m = b.evaluate()
    assert m.speed_basis == "worst-video+batch" and passes(m)["speed_ratio"] == "PASS"
    b.work["a.mp4"] = 80.0

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "record_run_timing", interrupted)
    with pytest.raises(KeyboardInterrupt):
        b.run(rejudge=True)
    t = b.timings()["a.mp4"]
    assert t["judged_wall_s"] == 80.0
    [run] = json.loads((b.out / "runs.json").read_text())
    assert t["judgment_sources"][0]["run_id"] == t["attempt_run_id"] != run["run_id"]
    m = b.evaluate()
    assert m.speed_basis == "unverified"
    assert passes(m)["speed_ratio"] != "PASS"


def test_run_batch_stamps_one_run_id_on_timings_and_the_batch_record(
    tmp_path, monkeypatch, th, rules
):
    b = Batch(tmp_path, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
    b.run()
    b.fail.add("b.mp4")
    b.due = True
    b.run()
    first, second = json.loads((b.out / "runs.json").read_text())
    assert first["run_id"] != second["run_id"]
    t = b.timings()
    assert t["a.mp4"]["judgment_sources"][0]["run_id"] == t["a.mp4"]["attempt_run_id"] == second["run_id"]
    # b.mp4 failed in run 2: its kept rows still belong to run 1, its attempt to run 2
    assert t["b.mp4"]["judgment_sources"][0]["run_id"] == first["run_id"]
    assert t["b.mp4"]["attempt_run_id"] == second["run_id"]


# ------------------------------------- C-X5(b): every run behind the current timing is gated


def test_fast_failed_retry_never_hides_the_judged_runs_batch_overhead():
    # Codex fix-4 review: a 100 s video judged with a 30 s wall inside a 160 s batch (r1),
    # then a failed 1 s retry (r2): the gate returned 0.3 PASS, hiding r1's 1.6 batch ratio.
    rows = _rows("a.mp4")
    timings = {"a.mp4": _entry(100.0, 30.0, attempt=1.0, status="analysis_error")}
    timings["a.mp4"]["attempt_run_id"] = "r2"
    runs = [_run(160.0, {"a.mp4": 100.0}, run_id="r1"), _run(1.0, {"a.mp4": 100.0}, run_id="r2")]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "worst-video+batch"
    assert m.speed_ratio == pytest.approx(1.6)
    assert passes(m)["speed_ratio"] == "FAIL"
    s = report_metrics(rows, timings, _cfg(), runs=runs).tradeoff["speed"]
    assert s["basis"] == "worst-video+batch" and s["ratio"] == pytest.approx(1.6)


def test_judged_run_without_its_batch_record_is_unverified_even_if_the_attempt_has_one():
    rows = _rows("a.mp4")
    timings = {"a.mp4": _entry(100.0, 30.0, attempt=1.0, status="analysis_error")}
    timings["a.mp4"]["attempt_run_id"] = "r2"
    runs = [_run(1.0, {"a.mp4": 100.0}, run_id="r2")]  # r1 never wrote its record
    assert evaluate(rows, timings, runs=runs).speed_basis == "unverified"


def test_run_cli_default_autocalib_is_the_repositorys_not_the_cwds(tmp_path, monkeypatch):
    """C-X5(c): without --autocalib the run uses <repo>/config/autocalib.yaml from any cwd;
    an explicit path is used as given."""
    calls = []
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **kw: calls.append(kw))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main(["run", str(tmp_path)]) == 0
    assert Path(calls[-1]["autocalib_config"]) == ROOT / "config" / "autocalib.yaml"
    assert main(["run", str(tmp_path), "--autocalib", "mine.yaml"]) == 0
    assert Path(calls[-1]["autocalib_config"]) == Path("mine.yaml")


# ------------------------------------- C-X7(a): a run's ratio uses the work it measured itself


def test_a_later_retry_never_rewrites_an_earlier_runs_overhead():
    # Codex fix-5 review: r1 took 200 s (A eligible: 100 s playback, 30 s work; B excluded:
    # 10 s work). A's failed 1 s retry (r2), then B's separate 250 s retry (r3): with B's
    # CURRENT 250 s subtracted, r1's ratio fell from 1.9 to 0 and speed flipped to PASS 0.3.
    rows = _rows("a.mp4")  # b.mp4 has no real round: excluded from the speed population
    timings = {"a.mp4": _entry(100.0, 30.0, attempt=1.0, status="analysis_error"),
               "b.mp4": {"duration_s": 50.0, "processing_s": 250.0}}
    timings["a.mp4"]["attempt_run_id"] = "r2"
    runs = [_run(200.0, {"a.mp4": 100.0, "b.mp4": 50.0}, run_id="r1",
                 work={"a.mp4": 30.0, "b.mp4": 10.0},
                 real={"a.mp4": True, "b.mp4": False}),  # B: no real rows in r1 (C-X8)
            _run(1.0, {"a.mp4": 100.0}, run_id="r2", work={"a.mp4": 1.0}),
            _run(250.0, {"b.mp4": 50.0}, run_id="r3", work={"b.mp4": 250.0})]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "worst-video+batch"
    assert m.speed_ratio == pytest.approx(1.9)
    assert passes(m)["speed_ratio"] == "FAIL"


def test_run_record_stores_the_work_it_measured(tmp_path, monkeypatch, th, rules):
    b = Batch(tmp_path, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
    b.work.update({"a.mp4": 3.0, "b.mp4": 4.0})
    b.run()
    [run] = json.loads((b.out / "runs.json").read_text())
    assert run["work"] == {"a.mp4": 3.0, "b.mp4": 4.0}


# ------------------------------- C-X8: historical ratios use only what each run record stored


def test_replaced_source_never_rewrites_an_earlier_runs_ratio():
    # Codex fix-6 review: r1 judged a 100 s video in 30 s inside a 160 s batch (1.6, FAIL);
    # the source was replaced by a 200 s video and a 1 s retry (r2) failed. With the CURRENT
    # 200 s as r1's denominator, r1 became 0.8 and speed passed.
    rows = _rows("a.mp4")
    timings = {"a.mp4": _entry(200.0, 30.0, attempt=1.0, status="analysis_error")}
    timings["a.mp4"]["attempt_run_id"] = "r2"
    runs = [_run(160.0, {"a.mp4": 100.0}, run_id="r1", work={"a.mp4": 30.0}),
            _run(1.0, {"a.mp4": 200.0}, run_id="r2", work={"a.mp4": 1.0})]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "worst-video+batch"
    assert m.speed_ratio == pytest.approx(1.6)
    assert passes(m)["speed_ratio"] == "FAIL"


def test_run_ratio_scope_is_what_the_run_recorded():
    # a gate video (no real rows in that run) is left out of that run's denominator and its
    # stored work subtracted, whatever videos.json says about it now
    rows = _rows("a.mp4", "g.mp4")  # g.mp4 has real rows NOW (judged later)
    timings = {"a.mp4": _entry(100.0, 50.0), "g.mp4": _entry(100.0, 50.0, run_id="r2")}
    runs = [_run(160.0, {"a.mp4": 100.0, "g.mp4": 100.0}, work={"a.mp4": 50.0, "g.mp4": 10.0},
                 real={"a.mp4": True, "g.mp4": False}),
            _run(60.0, {"g.mp4": 100.0}, run_id="r2", work={"g.mp4": 50.0})]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_ratio == pytest.approx(1.5)  # r1: (160 - 10) / 100


def test_veto_kept_attempt_on_a_shorter_source_fails_against_its_current_duration():
    # Codex fix-6 review: judged 100 s video, 30 s processing in a 40 s batch (r1); vetoed;
    # source replaced by a 10 s video; --rejudge keeps the rows (r2, 0.5 s attempt). With the
    # current 10 s duration recorded by r2 (C-X8) the gate sees 3.0, not the stale 0.4.
    rows = _rows("a.mp4")
    timings = {"a.mp4": _entry(10.0, 30.0, attempt=0.5)}
    timings["a.mp4"]["attempt_run_id"] = "r2"
    runs = [_run(40.0, {"a.mp4": 100.0}, run_id="r1", work={"a.mp4": 30.0}),
            _run(2.0, {"a.mp4": 10.0}, run_id="r2", work={"a.mp4": 0.5})]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_ratio == pytest.approx(3.0)
    assert passes(m)["speed_ratio"] == "FAIL"


# --------------------------- C-X9: a run record without its `real` map is unusable for speed


def test_run_record_without_real_map_cannot_verify_speed():
    # Codex fix-7 review: r1 160 s for A (100 s, real, 30 s) and B (100 s placeholder, 10 s)
    # with no `real` map; A then fails a 1 s retry. Treating B as real gave 0.8 PASS.
    rows = _rows("a.mp4")
    timings = {"a.mp4": _entry(100.0, 30.0, attempt=1.0, status="analysis_error"),
               "b.mp4": {"duration_s": 100.0, "processing_s": 10.0}}
    timings["a.mp4"]["attempt_run_id"] = "r2"
    r1 = _run(160.0, {"a.mp4": 100.0, "b.mp4": 100.0}, run_id="r1",
              work={"a.mp4": 30.0, "b.mp4": 10.0})
    del r1["real"]
    runs = [r1, _run(1.0, {"a.mp4": 100.0}, run_id="r2", work={"a.mp4": 1.0})]
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "unverified" and math.isnan(m.speed_ratio)
    assert passes(m)["speed_ratio"] != "PASS"
    r1["real"] = {"a.mp4": True, "b.mp4": False}  # with its map: (160 - 10) / 100
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_ratio == pytest.approx(1.5) and passes(m)["speed_ratio"] == "FAIL"
