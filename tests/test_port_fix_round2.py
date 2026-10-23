"""Round-two lifecycle, frozen evidence and timing regressions (synthetic runtimes only)."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bxjudge import manifest, pipeline, vlm
from bxjudge.autocalib import load_autocalib_config
from bxjudge.calibration_store import load_snapshot, save_snapshot
from bxjudge.cli import main
from bxjudge.corrections import confirm_reviewed, mark_audit_sample
from bxjudge.evaluate import (
    evaluate,
    load_eval_config,
    passes,
    read_audit,
    read_results,
    report_metrics,
)
from bxjudge.output import read_rounds, save_rounds, write_results
from tests.helpers import make_calibration, make_ref_img, rev, rr


class Runtime:
    def __init__(self, tmp_path, monkeypatch, th, rules, names=("a.mp4",)):
        self.now = 0.0
        self.duration = 100.0
        self.work = 1.0
        self.selection = 0.0
        self.fail = False
        self.probe_fail = False
        self.calls = []
        self.folder = tmp_path / "videos"
        self.calib = tmp_path / "calib"
        self.out = tmp_path / "out"
        self.folder.mkdir()
        self.calib.mkdir()
        for name in names:
            (self.folder / name).touch()
        self.th, self.rules = th, rules
        monkeypatch.setattr(pipeline.time, "perf_counter", lambda: self.now)
        monkeypatch.setattr(manifest, "probe", self.probe)
        monkeypatch.setattr(pipeline, "judge_video", self.judge)
        monkeypatch.setattr(pipeline, "first_second_frame", self.frame)
        monkeypatch.setattr(pipeline, "select_snapshot", lambda *a, **k: None)

    def probe(self, path):
        if self.probe_fail:
            self.now += self.work
            raise ValueError("unreadable current source")
        st = path.stat()
        return manifest.VideoInfo(path, st.st_size, st.st_mtime, self.duration,
                                  30, 8, 8, "h264", False, None)

    def frame(self, *a):
        self.now += self.selection

    def judge(self, info, *a, **k):
        self.calls.append(info.path.name)
        self.now += self.work
        if self.fail:
            raise RuntimeError("transient")
        result = rr("spin", "X", video=info.path.name)
        result.needs_review = True
        return [result]

    def run(self, **kwargs):
        pipeline.run_batch(self.folder, self.calib, self.out, detector=lambda *a: [],
                           th=self.th, rules=self.rules, **kwargs)

    def entry(self, name="a.mp4"):
        return json.loads((self.out / "videos.json").read_text())[name]


def _runs(duration=100.0, wall=1.0, names=("a.mp4",)):
    return [{"kind": "run", "started_at": "t", "wall_s": wall, "vlm_wall_s": None,
             "videos": {n: duration for n in names}}]


def _timing(judged, attempt, *, duration=100.0, status="ok"):
    return {"duration_s": duration, "processing_s": judged,
            "judged_wall_s": judged, "attempt_wall_s": attempt, "analysis_status": status}


def test_failed_cached_video_is_actually_rejudged_then_normal_cache_skips(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.run()
    rt.fail = True
    # Same processed identity + same calibration, but the latest attempt explicitly failed.
    entry = rt.entry()
    entry["analysis_status"] = "analysis_error"
    (rt.out / "videos.json").write_text(json.dumps({"a.mp4": entry}))
    before = (rt.out / "rounds.json").read_bytes()
    rt.run()
    assert rt.calls == ["a.mp4", "a.mp4"]
    assert rt.entry()["analysis_status"] == "analysis_error"
    assert (rt.out / "rounds.json").read_bytes() == before
    rt.fail = False
    rt.run()
    assert len(rt.calls) == 3 and rt.entry()["analysis_status"] == "ok"
    rt.run()
    assert len(rt.calls) == 3


@pytest.mark.parametrize("after_run", [False, True])
@pytest.mark.parametrize("reachable", [False, True])
def test_vlm_skips_latest_failed_videos_without_notes_frames_requests_or_timing(
        tmp_path, monkeypatch, th, rules, after_run, reachable):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.run()
    row = read_rounds(rt.out)[0]
    row["vlm_note"] = "vlm_unavailable: preserved"
    save_rounds([row], rt.out, rules)
    before = (rt.out / "rounds.json").read_bytes()
    e = rt.entry()
    e["analysis_status"] = "analysis_error"
    (rt.out / "videos.json").write_text(json.dumps({"a.mp4": e}))
    cfg = replace(vlm.load_vlm_config(), after_run=True)
    monkeypatch.setattr(pipeline, "load_vlm_config", lambda *a: cfg)
    monkeypatch.setattr(pipeline, "_release_gpu_memory", lambda: None)
    actions = []
    monkeypatch.setattr(vlm, "preflight", lambda *a: actions.append("preflight") or
                        (None if reachable else "offline"))
    monkeypatch.setattr(vlm, "event_frames", lambda *a: actions.append("frames"))
    monkeypatch.setattr(vlm.VlmSession, "ask", lambda *a, **k: actions.append("request"))
    if after_run:
        rt.fail = True
        monkeypatch.setattr(pipeline, "needs_reanalysis", lambda *a: True)
        rt.run(vlm_config=tmp_path / "fake.yaml")
    else:
        summary = vlm.review_results_dir(rt.out, th=th, rules=rules, config=cfg)
        assert summary["asked"] == 0
    assert (rt.out / "rounds.json").read_bytes() == before
    assert actions == []
    assert "vlm_s" not in rt.entry() and "vlm_requests" not in rt.entry()
    if after_run:
        assert json.loads((rt.out / "runs.json").read_text())[-1]["vlm_wall_s"] is None


@pytest.mark.parametrize("change", ["add", "remove"])
def test_frozen_population_addition_or_unsampled_removal_invalidates(tmp_path, rules, change):
    out = tmp_path / "out"
    write_results([rr("spin", "X", video=f"v{i}.mp4") for i in range(5)], out, rules)
    picked = mark_audit_sample(out, 2, seed=7, rules=rules)
    for video, number in picked:
        confirm_reviewed(out, video, number, rules=rules)
    if change == "add":
        write_results([rr("spin", "X", video="new.mp4")], out, rules)
    else:
        save_rounds([r for r in read_rounds(out) if r["video"] !=
                     next(f"v{i}.mp4" for i in range(5) if (f"v{i}.mp4", 1) not in picked)],
                    out, rules)
    rows, timings, runs = read_results(out)
    m = evaluate(rows, timings, runs=runs, audit=read_audit(out))
    assert passes(m)["finish_accuracy_unflagged"] == "INVALIDATED"
    for r in read_rounds(out):
        confirm_reviewed(out, r["video"], r["round"], rules=rules)
    rows, timings, runs = read_results(out)
    assert evaluate(rows, timings, runs=runs, audit=read_audit(out)).finish_accuracy_unflagged_method == "exact"


def test_public_read_results_evaluate_cannot_skip_frozen_sample_validation(tmp_path, rules):
    out = tmp_path / "out"
    write_results([rr("spin", "X", video=f"v{i}.mp4") for i in range(5)], out, rules)
    picked = mark_audit_sample(out, 2, seed=7, rules=rules)
    save_rounds([r for r in read_rounds(out) if (r["video"], r["round"]) != picked[0]], out, rules)
    rows, timings, runs = read_results(out)
    with pytest.raises(ValueError, match="audit"):
        evaluate(rows, timings, runs=runs)
    assert evaluate(rows, timings, runs=runs, audit=read_audit(out)).finish_accuracy_unflagged_method == "invalidated"


@pytest.mark.parametrize("bad", [
    {"keys": [["a.mp4", 1]], "population": 1},
    {"selection": [1]},
    {"seed": 1, "n": 1, "population": [["a.mp4", 1]], "selection": [1], "at": "x"},
    {"seed": True, "n": 1, "population": [["a.mp4", 1]], "selection": [["a.mp4", 1]], "at": "x"},
    {"seed": 1, "n": float("inf"), "population": [], "selection": [], "at": "x"},
    {"seed": 1, "n": 1, "population": [["a.mp4", True]], "selection": [["a.mp4", True]], "at": "x"},
    {"seed": 1, "n": 1, "population": [["a.mp4", 1]], "selection": [["b.mp4", 1]], "at": "x"},
])
def test_only_current_well_formed_audit_schema_is_accepted(tmp_path, bad):
    (tmp_path / "rounds.json").write_text("[]")
    (tmp_path / "audit_sample.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="audit"):
        read_audit(tmp_path)


def test_snapshot_bounds_are_validated_yaml_policy_and_apply_when_loading(tmp_path):
    import yaml

    raw = yaml.safe_load(Path("config/autocalib.yaml").read_text(encoding="utf-8"))
    assert raw["snapshot_max_frame_side"] == 100_000
    assert raw["snapshot_max_coordinate"] == 1e7
    snap = save_snapshot(tmp_path, "manual", make_calibration(), make_ref_img(), score=1.0)
    cfg = load_autocalib_config()
    assert load_snapshot(tmp_path, snap.id, config=cfg).id == snap.id
    with pytest.raises(ValueError, match="frame_size"):
        load_snapshot(tmp_path, snap.id, config=replace(cfg, snapshot_max_frame_side=100))
    with pytest.raises(ValueError, match="points"):
        load_snapshot(tmp_path, snap.id, config=replace(cfg, snapshot_max_coordinate=1))


@pytest.mark.parametrize("key,value", [
    ("snapshot_max_frame_side", True), ("snapshot_max_frame_side", 0),
    ("snapshot_max_frame_side", 2.5), ("snapshot_max_coordinate", float("inf")),
    ("snapshot_max_coordinate", -1), ("snapshot_max_coordinate", "large"),
])
def test_snapshot_yaml_policy_rejects_invalid_bounds(tmp_path, key, value):
    import yaml

    raw = yaml.safe_load(Path("config/autocalib.yaml").read_text(encoding="utf-8"))
    raw[key] = value
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_autocalib_config(path)


def test_fast_transient_failure_cannot_erase_slow_retained_judgment(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.work = 150.0
    rt.run()
    rt.fail, rt.work = True, 2.0
    monkeypatch.setattr(pipeline, "needs_reanalysis", lambda *a: True)
    before = (rt.out / "rounds.json").read_bytes()
    rt.run()
    e = rt.entry()
    assert e["judged_wall_s"] == 150.0 and e["attempt_wall_s"] == 2.0
    assert e["analysis_status"] == "analysis_error"
    assert (rt.out / "rounds.json").read_bytes() == before
    rows, timings, runs = read_results(rt.out)
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_ratio_worst_observed == 1.5
    assert passes(m)["speed_ratio"] == "FAIL"


def test_two_preselection_passes_count_the_gap_until_final_write(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
    rt.run()
    rt.selection, rt.work = 60.0, 20.0
    monkeypatch.setattr(pipeline, "needs_reanalysis", lambda *a: True)
    rt.run()
    assert rt.entry("a.mp4")["judged_wall_s"] == 140.0
    rows, timings, runs = read_results(rt.out)
    assert passes(evaluate(rows, timings, runs=runs))["speed_ratio"] == "FAIL"


@pytest.mark.parametrize("probe_fails", [False, True])
def test_failed_replaced_source_uses_current_duration_or_is_unverified(
        tmp_path, monkeypatch, th, rules, probe_fails):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.duration = 300.0
    rt.run()
    rt.duration, rt.work, rt.fail, rt.probe_fail = 10.0, 20.0, True, probe_fails
    (rt.folder / "a.mp4").write_bytes(b"changed source")
    rt.run()
    e = rt.entry()
    assert e["duration_s"] == (None if probe_fails else 10.0)
    assert e["judged_duration_s"] == 300.0
    rows, timings, runs = read_results(rt.out)
    m = evaluate(rows, timings, runs=runs)
    assert passes(m)["speed_ratio"] == "FAIL"
    assert m.speed_basis == "unverified"  # current failure never certifies speed
    if not probe_fails:
        assert m.speed_ratio_worst_observed == 2.0


def test_one_video_batch_overhead_fails_helper_report_cli_and_tradeoff(tmp_path, capsys):
    rows = [rev(video="a.mp4")]
    timings = {"a.mp4": _timing(10.0, 10.0)}
    runs = _runs(wall=120.0)
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_ratio == 1.2 and passes(m)["speed_ratio"] == "FAIL"
    extra = report_metrics(rows, timings, load_eval_config(), runs=runs)
    assert extra.tradeoff["speed"]["ratio"] == 1.2
    for name, obj in (("rounds", rows), ("videos", timings), ("runs", runs)):
        (tmp_path / f"{name}.json").write_text(json.dumps(obj))
    report = tmp_path / "eval.md"
    assert main(["eval", str(tmp_path), "--report", str(report)]) == 0
    assert "speed_ratio=1.200 FAIL" in capsys.readouterr().out
    assert "batch" in report.read_text() and "1.200" in report.read_text()


def test_missing_batch_evidence_cannot_pass():
    m = evaluate([rev(video="a.mp4")], {"a.mp4": _timing(10.0, 10.0)})
    assert m.speed_basis == "unverified" and passes(m)["speed_ratio"] == "FAIL"


def test_rejudge_regains_missing_timing_and_preserves_reviewed_rows(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.run()
    confirm_reviewed(rt.out, "a.mp4", 1, rules=rules)
    before = (rt.out / "rounds.json").read_bytes()
    e = rt.entry()
    e.pop("judged_wall_s", None)
    e.pop("attempt_wall_s", None)
    e.pop("wall_s", None)
    (rt.out / "videos.json").write_text(json.dumps({"a.mp4": e}))
    rt.run(rejudge=True)
    assert rt.calls == ["a.mp4", "a.mp4"]
    assert rt.entry()["judged_wall_s"] > 0 and rt.entry()["attempt_wall_s"] > 0
    assert (rt.out / "rounds.json").read_bytes() == before


def test_rejudge_cli_passes_actual_flag(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **k: captured.append(k))
    assert main(["run", str(tmp_path), "--rejudge", "--no-vlm"]) == 0
    assert captured[0]["rejudge"] is True


def test_interrupted_attempt_remains_pending_and_next_run_retries(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.run()
    monkeypatch.setattr(pipeline, "needs_reanalysis", lambda *a: True)
    real_judge = rt.judge

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "judge_video", interrupted)
    before = (rt.out / "rounds.json").read_bytes()
    with pytest.raises(KeyboardInterrupt):
        rt.run()
    assert rt.entry()["analysis_status"] == "pending"
    assert "attempt_wall_s" not in rt.entry()
    assert (rt.out / "rounds.json").read_bytes() == before
    monkeypatch.setattr(pipeline, "judge_video", real_judge)
    monkeypatch.setattr(pipeline, "needs_reanalysis", lambda *a: False)
    rt.run()
    assert rt.calls == ["a.mp4", "a.mp4"] and rt.entry()["analysis_status"] == "ok"


def test_reviewed_rows_cannot_lose_their_slower_judged_timing(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.work = 150.0
    rt.run()
    confirm_reviewed(rt.out, "a.mp4", 1, rules=rules)
    rt.work = 2.0
    rt.run(rejudge=True)
    e = rt.entry()
    assert e["judged_wall_s"] == 150.0 and e["attempt_wall_s"] == 2.0
    rows, timings, runs = read_results(rt.out)
    assert passes(evaluate(rows, timings, runs=runs))["speed_ratio"] == "FAIL"


def test_after_run_vlm_wait_is_in_single_continuous_interval(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
    cfg = replace(vlm.load_vlm_config(), after_run=True)
    monkeypatch.setattr(pipeline, "load_vlm_config", lambda *a: cfg)

    def release():
        rt.now += 20.0

    def review(*a, **k):
        rt.now += 50.0
        return {"asked": 2}

    monkeypatch.setattr(pipeline, "_release_gpu_memory", release)
    monkeypatch.setattr(pipeline, "review_results_dir", review)
    rt.run(vlm_config=tmp_path / "fake.yaml")
    assert rt.entry("a.mp4")["judged_wall_s"] == 72.0
    assert rt.entry("b.mp4")["judged_wall_s"] == 71.0


def test_failed_vetoed_group_without_replacement_keeps_raw_rows_and_audit_marks(
        tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.run()
    row = read_rounds(rt.out)[0]
    row["calibration_id"], row["audit_sample"] = "vetoed", True
    save_rounds([row], rt.out, rules)
    e = rt.entry()
    e["analysis_status"] = "analysis_error"
    e["calibration"] = {"id": "vetoed", "kind": "auto"}
    (rt.out / "videos.json").write_text(json.dumps({"a.mp4": e}))
    db = rt.out / "processed_videos.json"
    records = json.loads(db.read_text())
    records[str((rt.folder / "a.mp4").resolve())]["calibration_id"] = "vetoed"
    db.write_text(json.dumps(records))
    before = (rt.out / "rounds.json").read_bytes()
    monkeypatch.setattr(pipeline, "vetoed_ids", lambda *a: {"vetoed"})

    def cannot_recalibrate(info, *a, **k):
        return [pipeline._review(info.path.name, info.duration, "recalibrate")]

    monkeypatch.setattr(pipeline, "judge_video", cannot_recalibrate)
    rt.run()
    assert (rt.out / "rounds.json").read_bytes() == before
    assert rt.entry()["analysis_status"] != "ok"


def test_actual_batch_config_overhead_is_gated(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    load = pipeline.load_policy

    def slow_config(*a, **k):
        rt.now += 120.0
        return load(*a, **k)

    monkeypatch.setattr(pipeline, "load_policy", slow_config)
    rt.run()
    rows, timings, runs = read_results(rt.out)
    assert timings["a.mp4"]["judged_wall_s"] == 1.0
    assert runs[0]["wall_s"] == 121.0
    assert passes(evaluate(rows, timings, runs=runs))["speed_ratio"] == "FAIL"


def test_interrupted_after_run_vlm_stays_pending_and_is_retried(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    cfg = replace(vlm.load_vlm_config(), after_run=True)
    monkeypatch.setattr(pipeline, "load_vlm_config", lambda *a: cfg)
    monkeypatch.setattr(pipeline, "_release_gpu_memory", lambda: None)

    def interrupt(*a, **k):
        rt.now += 5.0
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "review_results_dir", interrupt)
    with pytest.raises(KeyboardInterrupt):
        rt.run(vlm_config=tmp_path / "fake.yaml")
    assert rt.entry()["analysis_status"] == "pending"
    rows, timings, runs = read_results(rt.out)
    assert passes(evaluate(rows, timings, runs=runs))["speed_ratio"] == "FAIL"
    rt.run()
    assert rt.calls == ["a.mp4", "a.mp4"] and rt.entry()["analysis_status"] == "ok"


@pytest.mark.parametrize("bad", [-1.0, float("nan"), "unknown"])
def test_invalid_vlm_timing_evidence_cannot_pass(bad):
    entry = {**_timing(10.0, 10.0), "vlm_s": bad}
    assert passes(evaluate([rev(video="a.mp4"), rev(video="b.mp4")],
                           {"a.mp4": entry, "b.mp4": _timing(10.0, 10.0)},
                           runs=_runs(names=("a.mp4", "b.mp4"))))["speed_ratio"] == "FAIL"


def test_legacy_video_wall_is_unverified_without_current_observations():
    entry = {"duration_s": 100.0, "wall_s": 10.0, "analysis_status": "ok", "processing_s": 10.0}
    assert passes(evaluate([rev(video="a.mp4")], {"a.mp4": entry}, runs=_runs()))["speed_ratio"] == "FAIL"


def test_actual_batch_release_overhead_is_gated(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules)

    def slow_release():
        rt.now += 120.0

    monkeypatch.setattr(pipeline, "_release_gpu_memory", slow_release)
    rt.run()
    rows, timings, runs = read_results(rt.out)
    assert runs[0]["wall_s"] == 121.0
    assert passes(evaluate(rows, timings, runs=runs))["speed_ratio"] == "FAIL"


@pytest.mark.parametrize("probe_fails", [False, True])
def test_standalone_vlm_records_its_latest_current_source_probe(
        tmp_path, monkeypatch, th, rules, probe_fails):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.work = 15.0
    rt.run()
    snap = save_snapshot(rt.out, "manual", make_calibration(), make_ref_img(), score=1.0)
    rows = read_rounds(rt.out)
    rows[0]["calibration_id"] = snap.id
    save_rounds(rows, rt.out, rules)
    rt.duration, rt.probe_fail = 10.0, probe_fails
    monkeypatch.setattr(vlm, "probe", rt.probe)
    image = make_ref_img()
    monkeypatch.setattr(vlm, "iter_frames", lambda *a, **k: iter((i, image) for i in range(100)))
    monkeypatch.setattr(vlm, "preflight", lambda *a: None)
    monkeypatch.setattr(vlm.VlmSession, "ask", lambda *a, **k: vlm.VlmAnswer("over", "evidence", "fake"))
    vlm.review_results_dir(rt.out, th=th, rules=rules, config=vlm.load_vlm_config())
    assert rt.entry()["duration_s"] == (None if probe_fails else 10.0)
    rows, timings, runs = read_results(rt.out)
    assert passes(evaluate(rows, timings, runs=runs))["speed_ratio"] == "FAIL"


def test_standalone_note_change_cannot_hide_slow_video_in_fast_pool(
        tmp_path, monkeypatch, th, rules, capsys):
    rt = Runtime(tmp_path, monkeypatch, th, rules, names=("a.mp4", "b.mp4"))
    rt.run()
    rows = read_rounds(rt.out)
    rows[1]["needs_review"] = False
    save_rounds(rows, rt.out, rules)
    original = rt.entry()
    review = vlm.review_low_confidence

    def slow_review(*a, **k):
        rt.now += 150.0
        return review(*a, **k)

    monkeypatch.setattr(vlm, "preflight", lambda *a: None)
    monkeypatch.setattr(vlm, "review_low_confidence", slow_review)
    assert main(["vlm", str(rt.out)]) == 0
    rows, timings, runs = read_results(rt.out)
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_ratio_wall == 0.76  # pooled batch alone would hide 150s on a 100s video
    assert m.speed_basis == "unverified" and passes(m)["speed_ratio"] == "FAIL"
    assert timings["a.mp4"]["attempt_wall_s"] is None
    assert timings["a.mp4"]["judged_wall_s"] == original["judged_wall_s"]
    assert timings["a.mp4"]["judged_duration_s"] == original["judged_duration_s"]
    assert timings["a.mp4"]["vlm_s"] == 150.0
    assert timings["b.mp4"]["attempt_wall_s"] == 1.0
    report = tmp_path / "eval.md"
    assert main(["eval", str(rt.out), "--report", str(report)]) == 0
    assert "speed_ratio=unverified FAIL" in capsys.readouterr().out
    assert "--rejudge" in report.read_text()


def test_standalone_identical_unavailable_note_work_still_invalidates(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.run()
    rows = read_rounds(rt.out)
    rows[0]["vlm_note"] = vlm.UNAVAILABLE_PREFIX + vlm.NO_CALIBRATION_RECORD
    save_rounds(rows, rt.out, rules)
    before = (rt.out / "rounds.json").read_bytes()
    review = vlm.review_low_confidence

    def same_note_work(*a, **k):
        rt.now += 150.0
        return review(*a, **k)

    monkeypatch.setattr(vlm, "preflight", lambda *a: None)
    monkeypatch.setattr(vlm, "review_low_confidence", same_note_work)
    vlm.review_results_dir(rt.out, th=th, rules=rules, config=vlm.load_vlm_config())
    assert (rt.out / "rounds.json").read_bytes() == before
    assert rt.entry()["attempt_wall_s"] is None and rt.entry()["vlm_s"] == 150.0


def test_standalone_no_target_keeps_continuous_timing_untouched(tmp_path, monkeypatch, th, rules):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.run()
    confirm_reviewed(rt.out, "a.mp4", 1, rules=rules)
    before = (rt.out / "videos.json").read_bytes()
    monkeypatch.setattr(vlm, "preflight", lambda *a: pytest.fail("no targets need no preflight"))
    summary = vlm.review_results_dir(rt.out, th=th, rules=rules, config=vlm.load_vlm_config())
    assert summary["asked"] == 0
    assert (rt.out / "videos.json").read_bytes() == before


@pytest.mark.parametrize("boundary", ["detector_release", "vlm_close", "batch_record"])
def test_interrupted_finalization_never_borrows_previous_batch_and_retries(
        tmp_path, monkeypatch, th, rules, boundary):
    rt = Runtime(tmp_path, monkeypatch, th, rules)
    rt.run()
    confirm_reviewed(rt.out, "a.mp4", 1, rules=rules)
    before_rows = (rt.out / "rounds.json").read_bytes()
    before_runs = (rt.out / "runs.json").read_bytes()
    cfg = replace(vlm.load_vlm_config(), after_run=True)
    kwargs = {"rejudge": True}

    def interrupted(*a, **k):
        rt.now += 150.0
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        if boundary == "detector_release":
            m.setattr(pipeline, "_release_gpu_memory", interrupted)
        elif boundary == "vlm_close":
            m.setattr(pipeline, "load_vlm_config", lambda *a: cfg)
            m.setattr(vlm, "preflight", lambda *a: None)
            m.setattr(vlm.VlmSession, "close", interrupted)
            kwargs["vlm_config"] = tmp_path / "fake.yaml"
        else:
            m.setattr(pipeline, "record_run_timing", interrupted)
        with pytest.raises(KeyboardInterrupt):
            rt.run(**kwargs)
    assert (rt.out / "rounds.json").read_bytes() == before_rows
    assert (rt.out / "runs.json").read_bytes() == before_runs
    assert rt.entry()["analysis_status"] == "pending"
    rows, timings, runs = read_results(rt.out)
    m = evaluate(rows, timings, runs=runs)
    assert m.speed_basis == "unverified" and passes(m)["speed_ratio"] == "FAIL"
    rt.run()
    assert rt.calls == ["a.mp4", "a.mp4", "a.mp4"]
    assert rt.entry()["analysis_status"] == "ok"
