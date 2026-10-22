"""R62 random audit sample of unflagged real rounds and R61 run-timing records."""

import json

import pytest

from bxjudge import output
from bxjudge.cli import main
from bxjudge.corrections import CorrectionError, confirm_reviewed, mark_audit_sample
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
    assert (meta["seed"], meta["n"], meta["population"]) == (7, 3, 6)


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
