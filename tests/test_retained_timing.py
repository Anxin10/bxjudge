"""Actual merge + pipeline + eval regressions for retained producing measurements."""
import json

import pytest

from bxjudge.corrections import (
    Correction,
    apply_correction,
    confirm_reviewed,
    mark_audit_sample,
)
from bxjudge.evaluate import passes, read_audit, read_results, report_metrics
from bxjudge.output import record_vlm_timing
from tests.test_observed_timing import Batch, _cfg


def _review(b, n=1):
    confirm_reviewed(b.out, "a.mp4", n, rules=b.rules)


def _runs(b):
    return json.loads((b.out / "runs.json").read_text())


def _sources(b):
    return b.timings()["a.mp4"]["judgment_sources"]


def test_reviewed_150_second_prediction_cannot_be_retimed_as_two_seconds(
    tmp_path, monkeypatch, th, rules
):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.work["a.mp4"] = 150.0
    b.calibration_id = "original-geometry"
    b.run()
    mark_audit_sample(b.out, 1, seed=7, rules=rules)
    apply_correction(b.out, Correction("a.mp4", 1, "winner", "X", "B", "review-time"), rules=rules)
    audit = (b.out / "audit_sample.json").read_bytes()
    corrections = (b.out / "corrections.jsonl").read_bytes()
    saved = (b.out / "rounds.json").read_bytes()
    first = _runs(b)[0]
    assert passes(b.evaluate())["speed_ratio"] == "FAIL"
    b.work["a.mp4"] = 2.0
    b.calibration_id = "new-geometry"
    b.run(rejudge=True)
    assert (b.out / "rounds.json").read_bytes() == saved
    assert (b.out / "audit_sample.json").read_bytes() == audit
    assert (b.out / "corrections.jsonl").read_bytes() == corrections
    [row] = json.loads(saved)
    assert row["calibration_id"] == "original-geometry"
    assert row["reviewer_fix"]["winner"] == "B" and row["audit_sample"] is True
    m = b.evaluate()
    assert passes(m)["speed_ratio"] == "FAIL"
    assert m.speed_ratio == pytest.approx(1.5)
    t = b.timings()["a.mp4"]
    assert (t["judged_wall_s"], t["attempt_wall_s"]) == (150.0, 2.0)
    assert _sources(b) == [{"rounds": [1], "run_id": first["run_id"],
                            "wall_s": 150.0, "processing_s": 150.0}]
    rows, timings, runs = read_results(b.out)
    observed = report_metrics(rows, timings, _cfg(), runs=runs, audit=read_audit(b.out)).observed["videos"][0]
    assert observed["judgment_sources"] == _sources(b)
    assert observed["attempt_run_id"] != first["run_id"]
    assert _runs(b)[0] == first


def test_mixed_generations_and_failed_retry_keep_all_live_sources(
    tmp_path, monkeypatch, th, rules
):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.round_numbers = [1, 2]
    b.work["a.mp4"] = 150.0
    b.run()
    _review(b, 1)
    b.round_numbers = [1, 2, 3]
    b.work["a.mp4"] = 20.0
    b.run(rejudge=True)
    _review(b, 2)
    b.fail.add("a.mp4")
    b.work["a.mp4"] = 1.0
    b.run(rejudge=True)
    assert b.timings()["a.mp4"]["analysis_status"] == "analysis_error"
    b.fail.clear()
    b.round_numbers = [1, 2, 3, 4]
    b.work["a.mp4"] = 2.0
    b.run(rejudge=True)
    first, second, failed, fourth = _runs(b)
    assert [(s["rounds"], s["run_id"], s["wall_s"]) for s in _sources(b)] == [
        ([1], first["run_id"], 150.0), ([2], second["run_id"], 20.0),
        ([3, 4], fourth["run_id"], 2.0)]
    assert failed["run_id"] not in [s["run_id"] for s in _sources(b)]
    assert passes(b.evaluate())["speed_ratio"] == "FAIL"
    # A fast repeat cannot shed either older producing batch.
    b.run(rejudge=True)
    assert [s["rounds"] for s in _sources(b)] == [[1], [2], [3, 4]]
    assert _sources(b)[0]["run_id"] == first["run_id"]
    assert _sources(b)[1]["run_id"] == second["run_id"]
    assert passes(b.evaluate())["speed_ratio"] == "FAIL"


def test_retained_fast_core_cannot_shed_slow_producing_batch(tmp_path, monkeypatch, th, rules):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.scan_s, b.work["a.mp4"] = 130.0, 30.0
    b.run()
    _review(b)
    first = _runs(b)[0]
    b.scan_s, b.work["a.mp4"] = 0.0, 2.0
    b.run(rejudge=True)
    b.run(rejudge=True)
    assert b.evaluate().speed_ratio == pytest.approx(1.6)
    assert passes(b.evaluate())["speed_ratio"] == "FAIL"
    assert _runs(b)[0] == first


def test_full_unreviewed_replacement_can_shed_superseded_measurements(
    tmp_path, monkeypatch, th, rules
):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.work["a.mp4"] = 150.0
    b.run()
    first = _runs(b)[0]
    b.work["a.mp4"] = 2.0
    b.run(rejudge=True)
    assert passes(b.evaluate())["speed_ratio"] == "PASS"
    assert b.evaluate().speed_ratio == pytest.approx(0.02)
    assert _sources(b)[0]["run_id"] != first["run_id"]
    assert _runs(b)[0] == first


@pytest.mark.parametrize("damage", ["absent", "missing_round", "missing_wall", "missing_core", "missing_run"])
def test_missing_source_evidence_cannot_be_verified_or_recreated_by_rejudge(
    tmp_path, monkeypatch, th, rules, damage
):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.run()
    _review(b)
    t = b.timings()
    source = t["a.mp4"].get("judgment_sources")
    if damage == "absent":
        t["a.mp4"].pop("judgment_sources", None)
    elif source is not None:
        if damage == "missing_round":
            source[0]["rounds"] = []
        else:
            source[0].pop({"missing_wall": "wall_s", "missing_core": "processing_s",
                           "missing_run": "run_id"}[damage])
    (b.out / "videos.json").write_text(json.dumps(t))
    assert b.evaluate().speed_basis == "unverified"
    b.work["a.mp4"] = 2.0
    b.run(rejudge=True)
    assert b.evaluate().speed_basis == "unverified"
    assert passes(b.evaluate())["speed_ratio"] == "FAIL"


def test_retained_source_with_lost_batch_record_is_unverified(tmp_path, monkeypatch, th, rules):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.run()
    _review(b)
    b.run(rejudge=True)
    (b.out / "runs.json").write_text(json.dumps(_runs(b)[1:]))
    assert b.evaluate().speed_basis == "unverified"
    assert passes(b.evaluate())["speed_ratio"] == "FAIL"


def test_retained_measurement_uses_current_duration_but_keeps_historical_record(
    tmp_path, monkeypatch, th, rules
):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.work["a.mp4"] = 30.0
    b.run()
    _review(b)
    first = _runs(b)[0]
    b.set_duration("a.mp4", 10.0)
    b.work["a.mp4"] = 2.0
    b.run(rejudge=True)
    assert b.evaluate().speed_ratio == pytest.approx(3.0)
    assert _runs(b)[0] == first


def test_each_retained_generations_batch_is_required_even_when_not_the_slowest_core(
    tmp_path, monkeypatch, th, rules
):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.round_numbers = [1, 2]
    b.work["a.mp4"] = 30.0
    b.run()
    _review(b, 1)
    b.scan_s, b.work["a.mp4"] = 130.0, 20.0
    b.run(rejudge=True)
    _review(b, 2)
    second = _runs(b)[1]
    b.scan_s, b.work["a.mp4"] = 0.0, 2.0
    b.round_numbers = [1, 2, 3]
    b.run(rejudge=True)
    # The first source has the slowest core (30), but the second has a slow batch (150).
    assert b.evaluate().speed_ratio == pytest.approx(1.5)
    assert passes(b.evaluate())["speed_ratio"] == "FAIL"
    (b.out / "runs.json").write_text(json.dumps([r for r in _runs(b)
                                                if r["run_id"] != second["run_id"]]))
    assert b.evaluate().speed_basis == "unverified"


def test_retained_predictions_keep_accumulated_vlm_measurements(
    tmp_path, monkeypatch, th, rules
):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.work["a.mp4"] = 80.0
    b.run()
    record_vlm_timing(b.out, "a.mp4", 30.0, [], run_id=_runs(b)[0]["run_id"])
    _review(b)
    b.work["a.mp4"] = 2.0
    b.run(rejudge=True)
    assert b.timings()["a.mp4"]["vlm_s"] == 30.0
    assert b.evaluate().speed_ratio == pytest.approx(1.1)
    assert passes(b.evaluate())["speed_ratio"] == "FAIL"
    # Retaining any old row also retains later accumulated VLM evidence.
    b.round_numbers = [1, 2]
    b.run(rejudge=True)
    record_vlm_timing(b.out, "a.mp4", 5.0, [], run_id=_runs(b)[-1]["run_id"])
    b.run(rejudge=True)
    assert b.timings()["a.mp4"]["vlm_s"] == 35.0


def test_fully_replaced_predictions_drop_old_vlm_evidence(tmp_path, monkeypatch, th, rules):
    b = Batch(tmp_path, monkeypatch, th, rules)
    b.run()
    record_vlm_timing(b.out, "a.mp4", 150.0, [], run_id=_runs(b)[0]["run_id"])
    b.run(rejudge=True)
    assert "vlm_s" not in b.timings()["a.mp4"]
    assert passes(b.evaluate())["speed_ratio"] == "PASS"
