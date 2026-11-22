"""Install the HUD workbench pages into an existing local site.

Reads the run directory and the site's validated snapshot only; writes inside
the site (theme assets, /judge/, re-rendered review pages, injected shells),
keeping a one-time `.before-v2` backup of every page it replaces. Review
manifests, submissions, original videos and the run itself are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from .judge_view import build_view
from .workbench_theme import (
    inject_shell,
    render_hub,
    render_judge_page,
    render_reid_page,
)

WEB = Path(__file__).resolve().parent / "web"
REID_PAGES = ("reid-review", "reid-challenge")
LEGACY = (
    ("identity-audit/index.html", "逐格核對", "identity-audit"),
    ("two-bey/index.html", "兩顆連續軌跡", "two-bey"),
    ("batch-outcomes/index.html", "批次結果", ""),
    ("fragment-refinement/index.html", "碎片接回對照", ""),
    ("reid-appearance/index.html", "外觀診斷", ""),
    ("reid-contrast/index.html", "不同顆對照", ""),
    ("launch-motion/index.html", "發射動作", ""),
    ("launch-state/index.html", "發射狀態", ""),
    ("launch-diagnostics/index.html", "發射診斷", ""),
    ("optimization/index.html", "速度與驗收記錄", ""),
    ("index-before-unified.html", "統一前舊首頁", ""),
)
JUDGE_MARKER = ".bx-judge"


def backup(path: Path) -> None:
    target = path.with_name(path.stem + ".before-v2" + path.suffix)
    if path.is_file() and not target.exists():
        shutil.copyfile(path, target)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def install_theme(site: Path) -> None:
    folder = site / "_theme"
    folder.mkdir(exist_ok=True)
    for name in ("bx_theme.css", "bx_fx.js"):
        shutil.copyfile(WEB / name, folder / name)


def install_judge(site: Path, view: dict) -> None:
    final = site / "judge"
    if final.exists() and not (final / JUDGE_MARKER).is_file():
        raise ValueError("site/judge exists but was not created by this tool; refusing to replace")
    with tempfile.TemporaryDirectory(prefix=".judge-", dir=site) as temp:
        stage = Path(temp) / "judge"
        (stage / "data").mkdir(parents=True)
        (stage / JUDGE_MARKER).write_text(view["index"]["run"] + "\n")
        _write(stage / "index.html", render_judge_page())
        shutil.copyfile(WEB / "judge_view.js", stage / "judge_view.js")
        _write(stage / "index.json", json.dumps(view["index"], ensure_ascii=False, allow_nan=False))
        for key, payload in view["rounds"].items():
            _write(stage / "data" / f"{key}.json", json.dumps(payload, ensure_ascii=False, allow_nan=False))
        if final.exists():
            shutil.rmtree(final)
        stage.rename(final)


def check_reid(site: Path, name: str) -> tuple[dict, str] | None:
    """Manifest and digest of an installed review page, refusing if the page has drifted."""
    folder = site / name
    page = folder / "index.html"
    if not (folder / "manifest.json").is_file() or not page.is_file():
        return None
    raw = (folder / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    embedded = re.search(r'"sha":\s*"([0-9a-f]{64})"', page.read_text(encoding="utf-8"))
    if embedded is None or embedded[1] != digest:
        raise ValueError(f"{name}: installed page does not match its manifest; drafts would orphan")
    return json.loads(raw), digest


def restyle_reid(site: Path, name: str, manifest: dict, digest: str) -> None:
    folder = site / name
    backup(folder / "index.html")
    _write(folder / "index.html", render_reid_page(manifest, digest))
    for asset in ("reid_review.js", "reid_review.css"):
        shutil.copyfile(WEB / asset, folder / asset)


def restyle_legacy(site: Path) -> list[str]:
    done = []
    for relative, title, active in LEGACY:
        page = site / relative
        if not page.is_file():
            continue
        original = page.read_text(encoding="utf-8")
        updated = inject_shell(original, title, active)
        if updated != original:
            backup(page)
            _write(page, updated)
        done.append(relative)
    return done


def restyle(site: Path, judge_run: Path, stores: dict[str, Path], review_confidence: float) -> dict:
    site, judge_run = Path(site).resolve(), Path(judge_run).resolve()
    if site == judge_run or site.is_relative_to(judge_run) or judge_run.is_relative_to(site):
        raise ValueError("site and judge run must be separate directories")
    reid_pages = {name: found for name in REID_PAGES if (found := check_reid(site, name))}
    judge_dir = site / "judge"
    if judge_dir.exists() and not (judge_dir / JUDGE_MARKER).is_file():
        raise ValueError("site/judge exists but was not created by this tool; refusing to replace")
    view = build_view(judge_run, review_confidence, stores)
    install_theme(site)
    install_judge(site, view)
    for name, (manifest, digest) in reid_pages.items():
        restyle_reid(site, name, manifest, digest)
    reid = list(reid_pages)
    snapshot_path = site / "_workbench" / "snapshot.json"
    if snapshot_path.is_file():
        hub = site / "index.html"
        backup(hub)
        _write(hub, render_hub(json.loads(snapshot_path.read_text()), view["index"],
                               previous_hub="index.before-v2.html"))
    legacy = restyle_legacy(site)
    return {"site": str(site), "judge_run": str(judge_run), "rounds": view["index"]["round_count"],
            "review": view["index"]["review_count"], "reid_pages": reid, "legacy_pages": legacy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--judge-run", type=Path, required=True)
    parser.add_argument("--store", nargs=2, action="append", default=[], metavar=("NAME", "DIR"))
    parser.add_argument("--review-confidence", type=float)
    args = parser.parse_args()
    threshold = args.review_confidence
    if threshold is None:
        from .config import load_thresholds

        threshold = load_thresholds().review_confidence
    stores = {name: Path(folder) for name, folder in args.store}
    print(json.dumps(restyle(args.site, args.judge_run, stores, threshold), ensure_ascii=False))


if __name__ == "__main__":
    main()
