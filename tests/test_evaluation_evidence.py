"""Acceptance regressions: missing hard cases and throughput cannot inflate metrics."""

import math
from dataclasses import replace

import pytest

from bxjudge.cli import main
from bxjudge.corrections import confirm_reviewed, mark_audit_sample
from bxjudge.evaluate import Metrics, evaluate, load_eval_config, passes, report_metrics
from bxjudge.output import write_results
from tests.helpers import rev, rr


def sample_rows(n_selected, n_reviewed):
    return [dict(rev(round=i, reviewed=i <= n_reviewed), audit_sample=i <= n_selected)
            for i in range(1, 101)]


def test_incomplete_frozen_sample_cannot_support_acceptance():
    rows = sample_rows(10, 1)
    metrics = evaluate(rows)
    evidence = report_metrics(rows, None, load_eval_config()).unflagged
    assert evidence["method"] == "incomplete_audit"
    assert (evidence["n_used"], evidence["n_sampled"]) == (1, 10)
    assert evidence["estimate"] == 1.0  # clearly provisional, never formal evidence
    assert math.isnan(evidence["low"])
    assert not passes(metrics)["finish_accuracy_unflagged"]


def test_complete_small_random_sample_has_perfect_point_but_insufficient_evidence():
    rows = sample_rows(1, 1)
    metrics = evaluate(rows)
    evidence = report_metrics(rows, None, load_eval_config()).unflagged
    assert evidence["method"] == "audit_sample"
    assert metrics.finish_accuracy_unflagged == 1.0
    assert evidence["low"] == pytest.approx(0.20654931437723745)
    assert not passes(metrics)["finish_accuracy_unflagged"]


def test_large_complete_sample_can_support_target_but_unknown_method_never_does():
    metrics = evaluate(sample_rows(90, 90))
    assert passes(metrics)["finish_accuracy_unflagged"]
    unknown = Metrics(1.0, 1.0, 1.0, 0.0, 0.5, 90)
    assert not passes(unknown)["finish_accuracy_unflagged"]
    assert not passes(unknown)["speed_ratio"]


def test_configured_confidence_level_changes_acceptance_without_changing_point():
    rows = sample_rows(70, 70)
    config = load_eval_config()
    strict = evaluate(rows, config=config)
    lower_confidence = replace(config, audit_confidence_level=0.90)
    relaxed = evaluate(rows, config=lower_confidence)
    assert strict.finish_accuracy_unflagged == relaxed.finish_accuracy_unflagged == 1.0
    assert not passes(strict, config)["finish_accuracy_unflagged"]
    assert passes(relaxed, lower_confidence)["finish_accuracy_unflagged"]


def test_complete_population_is_exact_for_this_batch():
    rows = [rev(round=i) for i in range(100)]
    rows[-1]["reviewer_fix"] = {"winner": "B"}
    evidence = report_metrics(rows, None, load_eval_config()).unflagged
    assert evidence["method"] == "exact"
    assert evidence["estimate"] == evidence["low"] == evidence["high"] == 0.99
    assert passes(evaluate(rows))["finish_accuracy_unflagged"]


def test_video_deadline_rate_and_batch_throughput_are_separate():
    rows = [rev(video="short.mp4"), rev(video="long.mp4")]
    timings = {"short.mp4": {"duration_s": 10.0, "processing_s": 20.0},
               "long.mp4": {"duration_s": 100.0, "processing_s": 1.0}}
    runs = [{"kind": "run", "started_at": "fixture", "wall_s": 21.0,
             "vlm_wall_s": None, "videos": {"short.mp4": 10.0, "long.mp4": 100.0}}]
    report = report_metrics(rows, timings, load_eval_config(), runs=runs)
    speed = report.tradeoff["speed"]
    assert report.speed_pass_rate == speed["per_video_pass_rate"] == 0.5
    assert speed["throughput_ratio"] == pytest.approx(21 / 110)
    assert speed["throughput_pass"] is True
    assert "pass_rate" not in speed
    metrics = evaluate(rows, timings, runs=runs)
    assert metrics.speed_ratio == 2.0
    assert metrics.speed_basis == "sum-based per-video estimate"
    assert not passes(metrics)["speed_ratio"]


def test_nonviolating_stage_sums_and_fast_batch_wall_do_not_prove_video_wall_deadlines():
    rows = [rev(video="a.mp4"), rev(video="b.mp4")]
    timings = {v: {"duration_s": 100.0, "processing_s": 20.0} for v in ("a.mp4", "b.mp4")}
    runs = [{"kind": "run", "started_at": "fixture", "wall_s": 50.0,
             "vlm_wall_s": None, "videos": {"a.mp4": 100.0, "b.mp4": 100.0}}]
    metrics = evaluate(rows, timings, runs=runs)
    assert metrics.speed_ratio == 0.2
    assert metrics.speed_basis == "sum-based per-video estimate"
    assert metrics.speed_ratio_wall == 0.25
    assert not passes(metrics)["speed_ratio"]


def test_favorable_pooled_history_cannot_hide_a_retained_video_deadline_violation():
    rows = [rev(video="slow.mp4"), rev(video="fast.mp4")]
    timings = {"slow.mp4": {"duration_s": 10.0, "processing_s": 20.0},
               "fast.mp4": {"duration_s": 100.0, "processing_s": 1.0}}
    runs = [{"kind": "run", "started_at": f"execution-{i}", "wall_s": seconds,
             "vlm_wall_s": None, "videos": {video: duration}}
            for i, (seconds, video, duration) in enumerate(
                ((20.0, "slow.mp4", 10.0), (1.0, "fast.mp4", 100.0)))]
    metrics = evaluate(rows, timings, runs=runs)
    assert metrics.speed_ratio_wall < 1.0
    assert metrics.speed_ratio == 2.0 and not passes(metrics)["speed_ratio"]


def test_partial_flagged_truth_is_not_overall_accuracy_or_final_accuracy():
    rows = [rev(round=1), rev(round=2, needs_review=True),
            rev(round=3, needs_review=True, reviewed=False)]
    joint = report_metrics(rows, None, load_eval_config()).tradeoff
    assert math.isnan(joint["overall_original_accuracy"])
    assert joint["known_correct_original_lower_bound"] == pytest.approx(2 / 3)
    assert joint["reviewed_output_agreement"]["accuracy"] == 1.0
    assert "auto_accuracy" not in joint and "final_accuracy" not in joint


def test_independent_executions_are_labelled_and_latest_stages_are_not_cumulative_cost():
    rows = [rev(video="a.mp4")]
    timings = {"a.mp4": {"duration_s": 100.0, "processing_s": 50.0}}
    runs = [{"kind": "run", "started_at": f"experiment-{i}", "wall_s": seconds,
             "vlm_wall_s": None, "videos": {"a.mp4": 100.0}}
            for i, seconds in enumerate((150.0, 60.0))]
    wall = report_metrics(rows, timings, load_eval_config(), runs=runs).wall
    assert wall["basis"] == "cumulative wall-clock"
    assert wall["wall_s"] == 210.0 and wall["duration_s"] == 200.0
    assert wall["unique_duration_s"] == 100.0
    assert [e["ratio"] for e in wall["executions"]] == [1.5, 0.6]
    assert wall["sum_s"] == 50.0  # latest snapshot only; not an independent wall measurement


def test_placeholder_and_overlong_duration_cannot_dilute_wall_throughput_ratio():
    rows = [rev(video="a.mp4"), rev(video="too-long.mp4"),
            rev(video="failed.mp4", finish=None, winner=None, review_reasons=["decode_error"])]
    runs = [{"kind": "run", "started_at": "fixture", "wall_s": 120.0,
             "vlm_wall_s": None,
             "videos": {"a.mp4": 100.0, "too-long.mp4": 400.0, "failed.mp4": 500.0}}]
    metrics = evaluate(rows, runs=runs)
    assert metrics.speed_ratio_wall == 1.2
    assert not passes(metrics)["speed_ratio"]


def test_cli_incomplete_audit_report_is_explicit_and_original_evidence_survives(tmp_path, rules):
    out = tmp_path / "results"
    write_results([rr("spin", "X", video=f"v{i}.mp4") for i in range(100)], out, rules)
    selected = mark_audit_sample(out, 10, seed=2, rules=rules)
    confirm_reviewed(out, *selected[0], rules=rules)
    report = tmp_path / "report.md"
    assert main(["eval", str(out), "--report", str(report)]) == 0
    content = report.read_text(encoding="utf-8")
    metric_line = next(line for line in content.splitlines()
                       if "finish_accuracy_unflagged" in line)
    assert "1/1 reviewed of 10 selected" in metric_line
    assert "point estimate is provisional" in metric_line
    assert "UNVERIFIED (FAIL)" in metric_line
    assert "Overall Original Accuracy | unverified" in content
    assert "selection bias" in content and "not independent final accuracy" in content


@pytest.mark.parametrize("with_stages", [False, True])
def test_cli_separates_unverified_video_deadlines_from_measured_throughput(
    tmp_path, rules, capsys, with_stages
):
    import json

    out = tmp_path / "results"
    write_results([rr("spin", "X", video="v.mp4")], out, rules)
    (out / "runs.json").write_text(json.dumps([{
        "kind": "run", "started_at": "fixture", "wall_s": 50.0,
        "vlm_wall_s": None, "videos": {"v.mp4": 100.0}}]))
    if with_stages:
        (out / "videos.json").write_text(json.dumps({"v.mp4": {
            "duration_s": 100.0, "processing_s": 20.0}}))
    report = tmp_path / "report.md"
    assert main(["eval", str(out), "--report", str(report)]) == 0
    assert "UNVERIFIED (FAIL)" in capsys.readouterr().out
    content = report.read_text(encoding="utf-8")
    metric_line = next(line for line in content.splitlines() if "speed_ratio" in line)
    assert "UNVERIFIED (FAIL)" in metric_line
    assert "Batch Throughput Pass | True (ratio 0.500)" in content
    if not with_stages:
        assert "No videos.json: per-video stage/latency metrics are unverified" in content
        assert "speed and latency are unverified" not in content


def test_cli_displays_video_violation_as_fail_even_when_throughput_passes(tmp_path, rules, capsys):
    import json

    out = tmp_path / "results"
    write_results([rr("spin", "X", video=v) for v in ("short.mp4", "long.mp4")], out, rules)
    (out / "videos.json").write_text(json.dumps({
        "short.mp4": {"duration_s": 10.0, "processing_s": 20.0},
        "long.mp4": {"duration_s": 100.0, "processing_s": 1.0}}))
    (out / "runs.json").write_text(json.dumps([{
        "kind": "run", "started_at": "fixture", "wall_s": 21.0,
        "vlm_wall_s": None, "videos": {"short.mp4": 10.0, "long.mp4": 100.0}}]))
    report = tmp_path / "report.md"
    assert main(["eval", str(out), "--report", str(report)]) == 0
    assert "speed_ratio=2.000 FAIL (sum-based deadline violation)" in capsys.readouterr().out
    content = report.read_text(encoding="utf-8")
    metric_line = next(line for line in content.splitlines() if "speed_ratio" in line)
    assert "FAIL (sum-based deadline violation)" in metric_line
    assert "Batch Throughput Pass | True (ratio 0.191)" in content
