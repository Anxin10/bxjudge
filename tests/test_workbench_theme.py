import hashlib
import json
import re

import pytest

from bxjudge import workbench_restyle as wr
from bxjudge import workbench_theme as wt


def test_reid_page_keeps_script_contract_and_escapes_embedded_json():
    manifest = {"schema": "s", "cases": [{"case_id": "a", "video": "</script><b>x"}]}
    page = wt.render_reid_page(manifest, "f" * 64)
    for element in wt.REID_IDS:
        assert f'id="{element}"' in page
    assert '<script type="module" src="reid_review.js"></script>' in page
    assert 'href="reid_review.css"' in page and "/_theme/bx_theme.css" in page
    data = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', page, re.DOTALL
    )[1]
    assert "</script>" not in data
    assert json.loads(data) == {"manifest": manifest, "sha": "f" * 64}


def test_inject_shell_is_idempotent_and_preserves_page_scripts():
    original = '<html><style>b{}</style><script>localStorage.getItem("k")</script>'
    once = wt.inject_shell(original, "逐格", "identity-audit")
    twice = wt.inject_shell(once, "逐格", "identity-audit")
    assert once == twice and once.count("<!--bx-shell-->") == 1
    assert '<script>localStorage.getItem("k")</script>' in once
    assert "aria-current=page>逐格核對" in once
    with_body = wt.inject_shell("<html><body class='x'><p>hi</p>", "t", "")
    assert with_body.index("<!--bx-shell-->") > with_body.index("<body")
    with pytest.raises(ValueError):
        wt.inject_shell("<p>nothing</p>", "t", "")


def test_hub_shows_judge_counts_and_escapes_values():
    snapshot = {
        "speed_gate": "PASS",
        "speed_ratio": 0.9,
        "human_reviewed_rows_in_previous_results": 3,
        "side_assignment": {"correct": 1, "wrong": 0, "unknown": 2},
        "core_commit": "abc",
        "rounds": [
            {
                "video": "<v>.mp4",
                "round": 1,
                "ai_outcome": "x",
                "ai_needs_review": True,
                "human_reviewed_in_previous_results": False,
                "human_outcome_recorded": "y",
            }
        ],
    }
    judge = {
        "run": "/r",
        "round_count": 4,
        "review_count": 3,
        "top_reasons": [["identity_unanchored", 3, "identity"]],
        "stores": {"reid-review": 2},
    }
    page = wt.render_hub(snapshot, judge, previous_hub="index.before-v2.html")
    assert "3／4 待覆核" in page and "&lt;v&gt;.mp4" in page and "<v>.mp4" not in page
    assert 'href="/index.before-v2.html"' in page and 'data-bx-workbench="1"' in page
    assert "尚未產生回合裁判資料" in wt.render_hub(snapshot, None)


def make_site(tmp_path):
    site = tmp_path / "site"
    for name in wr.REID_PAGES:
        folder = site / name
        folder.mkdir(parents=True)
        raw = json.dumps({"schema": "s", "cases": []}).encode()
        (folder / "manifest.json").write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        (folder / "index.html").write_text(
            f'<html><script id="data">{{"manifest": {{}}, "sha": "{digest}"}}</script>'
        )
    (site / "_workbench").mkdir()
    (site / "_workbench/snapshot.json").write_text(
        json.dumps({"speed_gate": "PASS", "speed_ratio": 1.0})
    )
    (site / "index.html").write_text("<html>old hub")
    (site / "identity-audit").mkdir()
    (site / "identity-audit/index.html").write_text(
        "<html><style></style><script>x()</script>"
    )
    run = tmp_path / "run"
    (run / "trajectories").mkdir(parents=True)
    (run / "rounds.json").write_text(
        json.dumps(
            [
                {
                    "video": "v.mp4",
                    "round": 1,
                    "needs_review": True,
                    "review_reasons": ["identity_unanchored"],
                    "confidence": 0.0,
                    "geometry_evidence": None,
                    "calibration_id": None,
                }
            ]
        )
    )
    return site, run


def test_restyle_installs_pages_with_one_time_backups(tmp_path):
    site, run = make_site(tmp_path)
    out = wr.restyle(site, run, {}, 0.7)
    assert out["reid_pages"] == list(wr.REID_PAGES) and out["review"] == 1
    assert (site / "index.before-v2.html").read_text() == "<html>old hub"
    assert "工作台總覽" in (site / "index.html").read_text()
    assert (site / "judge/index.json").is_file() and (
        site / "judge/data/v_r1.json"
    ).is_file()
    assert (site / "_theme/bx_theme.css").is_file()
    assert "<script>x()</script>" in (site / "identity-audit/index.html").read_text()
    wr.restyle(site, run, {}, 0.7)
    assert (site / "index.before-v2.html").read_text() == "<html>old hub"
    assert (
        (site / "reid-review/index.before-v2.html")
        .read_text()
        .startswith("<html><script")
    )


def test_restyle_refuses_mismatched_manifest_and_foreign_judge_dir(tmp_path):
    site, run = make_site(tmp_path)
    (site / "reid-review/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        wr.restyle(site, run, {}, 0.7)
    site2, run2 = make_site(tmp_path / "b")
    (site2 / "judge").mkdir()
    with pytest.raises(ValueError, match="not created"):
        wr.restyle(site2, run2, {}, 0.7)
    with pytest.raises(ValueError, match="separate"):
        wr.restyle(run2 / "x", run2, {}, 0.7)
