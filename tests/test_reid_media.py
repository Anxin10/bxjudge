import importlib.util
from pathlib import Path

import pytest


def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_reid_review.py"
    spec = importlib.util.spec_from_file_location("review_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preview_context_uses_actual_frame_indices_and_inclusive_endpoints():
    module = builder()
    assert module.clip_bounds(100, 120, 30, 1) == (70, 151)
    assert module.clip_bounds(20, 20, 30, 1) == (0, 51)


@pytest.mark.parametrize(
    "before,after,fps,context",
    [
        (True, 20, 30, 1),
        (-1, 20, 30, 1),
        (1, 2, 0, 1),
        (1, 2, float("nan"), 1),
        (1, 2, 30, float("inf")),
        (1, 2, 30, 0),
    ],
)
def test_preview_rejects_invalid_frame_or_time_values(before, after, fps, context):
    with pytest.raises(ValueError):
        builder().clip_bounds(before, after, fps, context)


def test_builder_never_overwrites_a_previously_published_review_set(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "manifest.json"
    marker.write_text("original")
    with pytest.raises(ValueError, match="new directory"):
        builder().build(tmp_path / "missing", tmp_path, out, (1280, 720), 1)
    assert marker.read_text() == "original"
