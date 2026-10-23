"""README end-to-end workflow (final review I2): complete, in order, and copy-pasteable."""

import re
import shlex
from pathlib import Path

import yaml

from bxjudge.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _workflow() -> str:
    start = README.index("## 完整流程")
    return README[start:README.index("\n## ", start + 1)]


def _commands(text: str) -> list[str]:
    """Shell lines of the fenced bash blocks, backslash continuations joined."""
    lines = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        joined = re.sub(r"\\\n\s*", " ", block)
        lines += [line.strip() for line in joined.splitlines() if line.strip()]
    return lines


def test_workflow_covers_every_step_in_order():
    text = "\n".join(_commands(_workflow()))  # the commands themselves, not the prose
    steps = ["scripts/sync_videos.sh", "bxjudge calibrate", "bxjudge label sample",
             "bxjudge label prelabel", "bxjudge label export", "bxjudge train --kind rfdetr",
             "bxjudge train --kind yolo26", "bxjudge compare", "weights/selected.json",
             "scripts/serve_ollama.sh", "bxjudge run", "bxjudge vlm", "bxjudge review",
             "bxjudge audit-sample", "bxjudge eval", "bxjudge tune"]
    positions = [text.index(step) for step in steps]
    assert positions == sorted(positions)


def test_local_vlm_start_command_matches_the_isolated_runtime_config():
    vlm = yaml.safe_load((ROOT / "config" / "vlm.yaml").read_text(encoding="utf-8"))
    host = vlm["host"].removeprefix("http://")
    commands = _commands(_workflow())
    assert any("scripts/setup_ollama.py" in c for c in commands)
    assert "scripts/serve_ollama.sh" in commands
    script = (ROOT / "scripts" / "serve_ollama.sh").read_text(encoding="utf-8")
    assert f"OLLAMA_HOST={host}" in script and "OLLAMA_NO_CLOUD=1" in script
    assert 'OLLAMA_MODELS="$models"' in script and "weights/ollama" in script
    pulls = [c for c in commands if " pull " in c]
    assert [c.split()[-1] for c in pulls] == [vlm["model"], vlm["fallback"]]
    assert all(c.startswith(f"OLLAMA_HOST={host} ") for c in pulls)
    assert "-cloud" not in _workflow()


def test_readme_documents_separate_calib_and_out_folders():
    workflow = _workflow()
    assert "--calib" in workflow and "--out" in workflow and "互不包含" in workflow


def test_every_bxjudge_command_in_the_workflow_parses():
    parser = build_parser()
    commands = [c for c in _commands(_workflow()) if c.startswith("uv run bxjudge ")]
    assert len(commands) >= 14
    for command in commands:
        argv = shlex.split(command.split("#")[0].replace("<任一支影片>", "a"))[3:]
        parser.parse_args(argv)  # SystemExit (argparse error) fails the test
