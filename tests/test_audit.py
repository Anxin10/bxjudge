"""R62 random audit sample of unflagged real rounds and R61 run-timing records."""

import json

import pytest

from bxjudge import output
from bxjudge.cli import main
from bxjudge.corrections import CorrectionError, confirm_reviewed, mark_audit_sample
from bxjudge.evaluate import evaluate, passes, read_audit, read_results
from bxjudge.output import read_rounds, row_to_result, write_results
from tests.helpers import rr


def _results(tmp_path, rules, n_unflagged=6):
    out = tmp_path / "res"
    rows = [rr("spin", "X", video=f"v{i}.mp4") for i in range(n_unflagged)]
    flagged = rr("over", "B", video="f.mp4")
    flagged.needs_review = True
    broken = rr(None, None, video="p.mp4", review_reasons=["recalibrate"], launch=False)
    write_results([*rows, flagged, broken], out, rules)
    return out


def _judgments(rows):
    return [{k: v for k, v in r.items() if k != "audit_sample"} for r in rows]


def test_marks_seeded_sample_of_unflagged_real_rounds_only(tmp_path, rules):
    out = _results(tmp_path, rules)
    confirm_reviewed(out, "v0.mp4", 1, rules=rules)  # reviewed rows stay eligible and kept
    before = read_rounds(out)
    picked = mark_audit_sample(out, 3, seed=7, rules=rules)
    after = read_rounds(out)
    assert len(picked) == 3
    marked = {(r["video"], r["round"]) for r in after if r.get("audit_sample")}
    assert marked == set(picked)
    assert all(v.startswith("v") for v, _ in marked)  # never flagged or placeholder rows
    assert _judgments(after) == before  # judgments, reviews and fixes untouched
    meta = json.loads((out / "audit_sample.json").read_text())
    assert (meta["seed"], meta["n"]) == (7, 3)
    assert meta["population"] == sorted([f"v{i}.mp4", 1] for i in range(6))
    assert meta["selection"] == [list(k) for k in picked]


def test_same_seed_is_idempotent_and_reproducible(tmp_path, rules):
    out = _results(tmp_path, rules)
    first = mark_audit_sample(out, 2, seed=11, rules=rules)
    snapshot = (out / "rounds.json").read_bytes()
    assert mark_audit_sample(out, 2, seed=11, rules=rules) == first
    assert (out / "rounds.json").read_bytes() == snapshot
    other = _results(tmp_path / "again", rules)
    assert mark_audit_sample(other, 2, seed=11, rules=rules) == first


def test_refuses_more_than_available_and_a_different_existing_sample(tmp_path, rules):
    out = _results(tmp_path, rules, n_unflagged=3)
    snapshot = (out / "rounds.json").read_bytes()
    with pytest.raises(CorrectionError, match="only 3"):
        mark_audit_sample(out, 4, seed=1, rules=rules)
    with pytest.raises(CorrectionError):
        mark_audit_sample(out, 0, seed=1, rules=rules)
    assert (out / "rounds.json").read_bytes() == snapshot
    mark_audit_sample(out, 1, seed=1, rules=rules)
    with pytest.raises(CorrectionError, match="already"):
        mark_audit_sample(out, 2, seed=99, rules=rules)


def test_audit_mark_survives_snapshot_rewrites_and_is_sparse(tmp_path, rules):
    out = _results(tmp_path, rules)
    picked = mark_audit_sample(out, 6, seed=3, rules=rules)
    # the VLM phase rewrites a video's complete snapshot through row_to_result
    snapshot = [row_to_result(r) for r in read_rounds(out) if r["video"] == "v1.mp4"]
    write_results(snapshot, out, rules)
    rows = {(r["video"], r["round"]): r for r in read_rounds(out)}
    assert all(rows[k]["audit_sample"] is True for k in picked)
    assert "audit_sample" not in rows[("f.mp4", 1)]
    bad = dict(rows[("v1.mp4", 1)], audit_sample="yes")
    with pytest.raises(ValueError):
        row_to_result(bad)


def test_audit_sample_cli(tmp_path, rules, capsys):
    out = _results(tmp_path, rules)
    assert main(["audit-sample", str(out), "--n", "2", "--seed", "5"]) == 0
    assert "2" in capsys.readouterr().out
    assert sum(bool(r.get("audit_sample")) for r in read_rounds(out)) == 2
    assert main(["audit-sample", str(out), "--n", "50", "--seed", "5"]) == 2
    assert "error:" in capsys.readouterr().err


# ---- R61 run timing writer ------------------------------------------------------


def test_run_timing_records_append_and_validate(tmp_path):
    output.record_run_timing(tmp_path, kind="run", started_at="2026-10-23T10:00:00+08:00",
                             wall_s=12.5, vlm_wall_s=4.0, videos={"a.mp4": 10.0})
    output.record_run_timing(tmp_path, kind="vlm", started_at="2026-10-23T11:00:00+08:00",
                             wall_s=3.0, vlm_wall_s=3.0, videos={})
    runs = json.loads((tmp_path / "runs.json").read_text())
    assert [r["kind"] for r in runs] == ["run", "vlm"]
    assert runs[0] == {"kind": "run", "started_at": "2026-10-23T10:00:00+08:00",
                       "wall_s": 12.5, "vlm_wall_s": 4.0, "videos": {"a.mp4": 10.0}}
    for bad in ({"kind": "other"}, {"wall_s": -1.0}, {"vlm_wall_s": 20.0},
                {"videos": {"a.mp4": "long"}}):
        args = {"kind": "run", "started_at": "x", "wall_s": 12.5, "vlm_wall_s": None,
                "videos": {}, **bad}
        with pytest.raises(ValueError):
            output.record_run_timing(tmp_path, **args)
    assert len(json.loads((tmp_path / "runs.json").read_text())) == 2


# ---- R69 audit freeze ------------------------------------------------------------


def _review_all(out, rules):
    for r in read_rounds(out):
        if r.get("audit_sample"):
            confirm_reviewed(out, r["video"], r["round"], rules=rules)


def _eval(out):
    rows, timings, runs = read_results(out)
    return evaluate(rows, timings, runs=runs, audit=read_audit(out))


def test_rerun_replacing_a_sampled_row_invalidates_the_estimate(tmp_path, rules):
    out = _results(tmp_path, rules)
    picked = mark_audit_sample(out, 3, seed=7, rules=rules)
    assert _eval(out).finish_accuracy_unflagged_method != "invalidated"
    video = picked[0][0]  # unreviewed: a successful re-analysis replaces it
    write_results([rr("spin", "X", video=video)], out, rules)  # successful re-analysis
    assert not [r for r in read_rounds(out) if r["video"] == video and r.get("audit_sample")]
    m = _eval(out)
    assert m.finish_accuracy_unflagged_method == "invalidated"
    assert passes(m)["finish_accuracy_unflagged"] == "INVALIDATED"


def test_vanished_selected_row_invalidates_too(tmp_path, rules):
    out = _results(tmp_path, rules)
    picked = mark_audit_sample(out, 2, seed=7, rules=rules)
    rows, _, _ = read_results(out)
    rows = [r for r in rows if (r["video"], r["round"]) != picked[0]]
    assert evaluate(rows, audit=read_audit(out)).finish_accuracy_unflagged_method == "invalidated"


def test_remark_after_population_change_needs_replace(tmp_path, rules):
    out = _results(tmp_path, rules)
    mark_audit_sample(out, 2, seed=7, rules=rules)
    write_results([rr("spin", "X", video="new.mp4")], out, rules)  # population grows
    with pytest.raises(ValueError, match="--replace"):
        mark_audit_sample(out, 2, seed=7, rules=rules)
    assert main(["audit-sample", str(out), "--n", "2", "--seed", "7"]) == 2
    picked = mark_audit_sample(out, 2, seed=8, rules=rules, replace=True)
    marked = {(r["video"], r["round"]) for r in read_rounds(out) if r.get("audit_sample")}
    assert marked == set(picked)
    meta = json.loads((out / "audit_sample.json").read_text())
    assert ["new.mp4", 1] in meta["population"] and meta["seed"] == 8
    assert _eval(out).finish_accuracy_unflagged_method != "invalidated"
    assert main(["audit-sample", str(out), "--n", "2", "--seed", "9", "--replace"]) == 0


def test_marks_without_the_meta_file_are_an_error(tmp_path, rules):
    out = _results(tmp_path, rules)
    mark_audit_sample(out, 2, seed=7, rules=rules)
    (out / "audit_sample.json").unlink()
    with pytest.raises(ValueError, match="audit_sample.json"):
        read_results(out)


def test_unchanged_sample_behaves_as_before(tmp_path, rules):
    out = _results(tmp_path, rules)
    mark_audit_sample(out, 3, seed=7, rules=rules)
    _review_all(out, rules)
    rows, timings, runs = read_results(out)
    with_meta = evaluate(rows, timings, runs=runs, audit=read_audit(out))
    assert with_meta.finish_accuracy_unflagged_method == "audit_sample"
    assert passes(with_meta)["finish_accuracy_unflagged"] == "INCONCLUSIVE"  # R64: n < min


def test_eval_cli_reports_invalidated_with_remedy(tmp_path, rules, capsys):
    out = _results(tmp_path, rules)
    picked = mark_audit_sample(out, 3, seed=7, rules=rules)
    write_results([rr("spin", "X", video=picked[0][0])], out, rules)
    report = tmp_path / "r.md"
    assert main(["eval", str(out), "--report", str(report)]) == 0
    text = capsys.readouterr().out
    assert "INVALIDATED" in text and "audit-sample" in text and "--n 3 --seed 7 --replace" in text
    assert "INVALIDATED (resample)" in report.read_text(encoding="utf-8")
    assert "--n 3 --seed 7 --replace" in report.read_text(encoding="utf-8")


def test_exact_basis_beats_invalidation_when_every_unflagged_round_is_reviewed(tmp_path, rules):
    # C5-1: re-analysing a sampled row invalidates the *sample*, but once every unflagged round
    # is reviewed the exact value needs no sample.
    out = _results(tmp_path, rules)
    picked = mark_audit_sample(out, 3, seed=7, rules=rules)
    write_results([rr("spin", "X", video=picked[0][0])], out, rules)
    assert _eval(out).finish_accuracy_unflagged_method == "invalidated"  # sample-based: still so
    for r in read_rounds(out):
        if r["video"].startswith("v"):  # the six unflagged real rounds
            confirm_reviewed(out, r["video"], r["round"], rules=rules)
    m = _eval(out)
    assert m.finish_accuracy_unflagged_method == "exact"
    assert passes(m)["finish_accuracy_unflagged"] != "INVALIDATED"


# ---- fix round 2: C-X2 (c)-(e) ----------------------------------------------------


def test_new_unflagged_rounds_change_the_frozen_population_and_invalidate(tmp_path, rules):
    # Codex T5: new unflagged real rounds join after the draw; the old sample no longer
    # represents the evaluated population (R69).
    out = _results(tmp_path, rules)
    mark_audit_sample(out, 3, seed=7, rules=rules)
    _review_all(out, rules)
    assert _eval(out).finish_accuracy_unflagged_method == "audit_sample"
    write_results([rr("spin", "X", video="new.mp4")], out, rules)  # unflagged, unreviewed
    m = _eval(out)
    assert m.finish_accuracy_unflagged_method == "invalidated"
    assert passes(m)["finish_accuracy_unflagged"] == "INVALIDATED"


def test_removed_unflagged_rounds_change_the_frozen_population_and_invalidate(tmp_path, rules):
    out = _results(tmp_path, rules)
    picked = mark_audit_sample(out, 2, seed=7, rules=rules)
    _review_all(out, rules)
    rows, timings, runs = read_results(out)
    gone = next(r for r in rows if r["video"].startswith("v")
                and (r["video"], r["round"]) not in picked)
    rows = [r for r in rows if r is not gone]  # an unsampled population member vanished
    m = evaluate(rows, timings, runs=runs, audit=read_audit(out))
    assert m.finish_accuracy_unflagged_method == "invalidated"


def test_evaluate_without_audit_metadata_on_marked_rows_is_an_error(tmp_path, rules):
    # Codex T5: read_results -> evaluate without the frozen record silently skipped the
    # freeze checks and could PASS on a shrunk sample.
    from bxjudge.evaluate import load_eval_config, report_metrics, unflagged_accuracy

    out = _results(tmp_path, rules)
    picked = mark_audit_sample(out, 2, seed=7, rules=rules)
    rows, timings, runs = read_results(out)
    rows = [r for r in rows if (r["video"], r["round"]) != picked[0]]  # shrunk sample
    with pytest.raises(ValueError, match="audit"):
        evaluate(rows, timings, runs=runs)
    with pytest.raises(ValueError, match="audit"):
        report_metrics(rows, timings, load_eval_config(), runs=runs)
    with pytest.raises(ValueError, match="audit"):
        unflagged_accuracy(rows, 0.95)


def _write_meta(out, meta):
    (out / "audit_sample.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.mark.parametrize("meta", [
    {"seed": 7, "n": 1, "keys": [["v0.mp4", 1]]},  # legacy record: no compatibility path
    {"selection": [1]},
    {"seed": 7, "n": 1, "population": [["v0.mp4", 1]], "selection": [1]},
    {"seed": 7, "n": 1, "population": [["v0.mp4", 1]], "selection": [["v0.mp4", "1"]]},
    {"seed": 7, "n": 1, "population": 3, "selection": [["v0.mp4", 1]]},
    {"seed": "7", "n": 1, "population": [["v0.mp4", 1]], "selection": [["v0.mp4", 1]]},
    {"seed": 7, "n": True, "population": [["v0.mp4", 1]], "selection": [["v0.mp4", 1]]},
])
def test_malformed_or_legacy_audit_records_are_rejected_with_valueerror(tmp_path, rules, meta):
    out = _results(tmp_path, rules)
    mark_audit_sample(out, 1, seed=7, rules=rules)
    rows, timings, runs = read_results(out)
    with pytest.raises(ValueError, match="audit"):
        evaluate(rows, timings, runs=runs, audit=meta)
    _write_meta(out, meta)
    with pytest.raises(ValueError, match="audit"):
        read_audit(out)
