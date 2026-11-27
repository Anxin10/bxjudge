"""Render the workbench page shells with the real Python renderers, using synthetic data.

Writes storybook/generated/*.html (git-ignored). Loads bxjudge/workbench_theme.py by file path
so only the standard library is needed (no project venv, no run data, no video frames).
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "generated"
FIXTURES = HERE / "fixtures"

spec = importlib.util.spec_from_file_location(
    "workbench_theme", ROOT / "src" / "bxjudge" / "workbench_theme.py"
)
theme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(theme)

BODY = re.compile(r"<body[^>]*>(.*)</body>", re.DOTALL)
# Page scripts are loaded by each story instead, so a story controls when they run.
PAGE_SCRIPT = re.compile(r'<script type="module" src="[^"]+"></script>')


def body_of(document: str) -> str:
    match = BODY.search(document)
    if match is None:
        raise ValueError("renderer output has no <body>")
    return PAGE_SCRIPT.sub("", match.group(1)).strip() + "\n"


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def synthetic_frame(cx: int, cy: int) -> str:
    """A drawn 1280x720 stand-in for a source frame (never a real video frame)."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">'
        '<rect width="1280" height="720" fill="#20262e"/>'
        '<ellipse cx="640" cy="380" rx="430" ry="300" fill="#d9dde2"/>'
        '<ellipse cx="640" cy="380" rx="300" ry="205" fill="none" stroke="#9aa3ad" stroke-width="6"/>'
        f'<circle cx="{cx}" cy="{cy}" r="26" fill="#3a5f8f"/>'
        f'<circle cx="{cx}" cy="{cy}" r="9" fill="#e8b04a"/></svg>'
    )
    return "data:image/svg+xml;utf8," + quote(svg)


def with_frames(manifest: dict) -> dict:
    for case in manifest["cases"]:
        for end in ("before", "after"):
            image = case[end]["image"]
            if image.startswith("@synthetic-frame:"):
                cx, cy = (int(v) for v in image.split(":", 1)[1].split(","))
                case[end]["image"] = synthetic_frame(cx, cy)
    return manifest


def main() -> None:
    OUT.mkdir(exist_ok=True)
    snapshot = load("hub_snapshot.json")
    judge = load("hub_judge.json")
    reid = load("reid_manifest.json")
    reid["manifest"] = with_frames(reid["manifest"])
    pages = {
        "judge.html": theme.render_judge_page(),
        "reid.html": theme.render_reid_page(reid["manifest"], reid["sha"]),
        "hub.html": theme.render_hub(snapshot, judge, previous_hub="index-20261120.html"),
        "hub-no-run.html": theme.render_hub(snapshot, None),
    }
    for name, document in pages.items():
        (OUT / name).write_text(body_of(document), encoding="utf-8")
    print(f"wrote {len(pages)} page shells to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
