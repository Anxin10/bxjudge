"""Requirement-complete deadline evidence through public orchestration."""
import math
from dataclasses import replace

import pytest

from bxjudge import pipeline, vlm
from bxjudge.cli import main
from bxjudge.evaluate import (
    evaluate,
    load_eval_config,
    passes,
    read_results,
    report_metrics,
    speed_verdict,
)
from bxjudge.manifest import VideoInfo
from bxjudge.output import save_rounds
from tests.helpers import rev, rr
from tests.test_observed_timing import Clock, batch
from tests.test_pipeline import _vlm_config


@pytest.mark.parametrize("setup,cleanup,expected", [(20, 0, "FAIL"), (1, 0, "PASS"), (1, 20, "FAIL")])
def test_cold_single_full_wall_saved_cli_report(tmp_path, monkeypatch, th, rules, capsys, setup, cleanup, expected):
    clock = Clock()
    monkeypatch.setattr(pipeline.time, "perf_counter", clock)
    path = tmp_path / "single.mp4"
    path.touch()
    def scan(*a, **k):
        clock.advance(setup)
        return [VideoInfo(path, 1, 1, 10, 30, 8, 8, "h264", False, None)]
    monkeypatch.setattr(pipeline, "scan", scan)
    def judge(*a, **k):
        clock.advance(2)
        return [replace(rr("spin", "X", video=path.name), needs_review=True)]
    monkeypatch.setattr(pipeline, "judge_video", judge)
    monkeypatch.setattr(pipeline, "_release_gpu_memory", lambda: None)
    monkeypatch.setattr(vlm, "_note", lambda *a: "advisory")
    monkeypatch.setattr(vlm.VlmSession, "close", lambda self: clock.advance(cleanup))
    out = tmp_path / "out"
    pipeline.run_batch(tmp_path, tmp_path, out, detector=lambda *a: [], th=th, rules=rules,
                       vlm_config=_vlm_config(tmp_path))
    rows, timings, runs = read_results(out)
    evidence = timings[path.name]["invocation_wall"]
    assert evidence == {"scope": "complete_invocation", "kind": "exact", "seconds": setup + 2 + cleanup}
    assert timings[path.name]["pipeline_wall"]["seconds"] == 2
    assert runs[-1]["wall_s"] == setup + 2 + cleanup
    metrics = evaluate(rows, timings, runs=runs)
    assert passes(metrics)["speed_ratio"] == (expected == "PASS")
    assert speed_verdict(metrics, load_eval_config()) == expected
    report = tmp_path / "eval.md"
    assert main(["eval", str(out), "--report", str(report)]) == 0
    stdout, text = capsys.readouterr().out, report.read_text()
    for output in (stdout, text):
        assert "complete_invocation" in output
        assert "exact" in output
        assert expected in output
    assert f"speed_ratio={((setup + 2 + cleanup) / 10):.3f} {expected}" in stdout


@pytest.mark.parametrize("duration,expected", [(100, "PASS"), (40, "UNVERIFIED (FAIL)"), (10, "FAIL (observed lower-bound deadline violation)")])
def test_shared_bound_is_not_individual_elapsed(tmp_path, monkeypatch, th, rules, duration, expected):
    _, out, run = batch(tmp_path, monkeypatch, th, rules)
    original_scan = pipeline.scan
    monkeypatch.setattr(pipeline, "scan", lambda *a, **k: [replace(i, duration=duration) for i in original_scan(*a, **k)])
    monkeypatch.setattr(vlm.VlmSession, "close", lambda self: pipeline.time.perf_counter.advance(10))
    run()
    rows, timings, _ = read_results(out)
    assert all(e["invocation_wall"] == {"scope": "complete_invocation", "kind": "completion_upper_bound", "seconds": 45} for e in timings.values())
    metrics = evaluate(rows, timings)
    assert speed_verdict(metrics, load_eval_config()) == expected
    assert passes(metrics)["speed_ratio"] == (expected == "PASS")


@pytest.mark.parametrize("second", ["placeholder", "over_policy"])
def test_actual_input_count_not_eligible_count_controls_exactness(tmp_path, monkeypatch, th, rules, second):
    _, out, run = batch(tmp_path, monkeypatch, th, rules, enabled=False)
    original_scan, original_judge = pipeline.scan, pipeline.judge_video
    def scan(*a, **k):
        infos = original_scan(*a, **k)
        if second == "over_policy":
            infos[1] = replace(infos[1], duration=1000)
        return infos
    def judge(info, *a, **k):
        if info.path.name == "b.mp4" and second == "placeholder":
            return [rr(None, None, video=info.path.name, review_reasons=["no_round"])]
        return original_judge(info, *a, **k)
    monkeypatch.setattr(pipeline, "scan", scan)
    monkeypatch.setattr(pipeline, "judge_video", judge)
    run()
    rows, timings, _ = read_results(out)
    assert timings["a.mp4"]["invocation_wall"]["kind"] == "completion_upper_bound"
    assert evaluate(rows, timings).n_required_speed_videos == 1


@pytest.mark.parametrize("stage", ["scan", "load_policy"])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_early_failed_attempt_invalidates_old_complete_evidence(tmp_path, monkeypatch, th, rules, stage, error):
    _, out, run = batch(tmp_path, monkeypatch, th, rules, enabled=False)
    run()
    def fail(*a, **k):
        raise error("early failure")
    monkeypatch.setattr(pipeline, stage, fail)
    with pytest.raises(error):
        run()
    rows, timings, _ = read_results(out)
    assert all(e.get("invocation_wall") is None for e in timings.values())
    assert not passes(evaluate(rows, timings))["speed_ratio"]


@pytest.mark.parametrize("reviewed", [set(), {1}, {1, 2}])
def test_match_agreement_requires_full_human_truth(tmp_path, rules, reviewed):
    rows = [rev(round=i, t_launch=float(i), finish="xtreme" if i == 1 else "spin",
                winner="X", reviewed=i in reviewed) for i in (1, 2)]
    agreement = report_metrics(rows, None, load_eval_config(), rules=rules).match_agreement
    full = int(len(reviewed) == 2)
    assert (agreement["n_matches"], agreement["agree"], agreement["n_partial"]) == (full, full, 1 - full)
    assert agreement["ratio"] == 1 if full else math.isnan(agreement["ratio"])
    out = tmp_path / "out"
    save_rounds(rows, out, rules)
    report = tmp_path / "eval.md"
    assert main(["eval", str(out), "--report", str(report)]) == 0
    text = report.read_text()
    assert "fully reviewed matches" in text
    if not full:
        assert "match score agreement unverified (0/0)" in text
