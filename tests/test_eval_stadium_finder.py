"""Safety of scripts/eval_stadium_finder.py (Codex review Medium 4-6, 2026-11-11)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bxjudge import PROJECT_ROOT


def _load():
    spec = importlib.util.spec_from_file_location(
        "eval_stadium_finder", PROJECT_ROOT / "scripts" / "eval_stadium_finder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load()


def test_output_inside_a_video_folder_is_refused(tmp_path):
    videos = tmp_path / "videos" / "20261108"
    videos.mkdir(parents=True)
    with pytest.raises(SystemExit):
        ev.check_output_dir(videos / "eval", [videos])
    with pytest.raises(SystemExit):
        ev.check_output_dir(videos, [videos])
    ev.check_output_dir(tmp_path / "eval", [videos])  # outside: fine


def test_output_reaching_a_video_folder_through_a_symlink_is_refused(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    link = tmp_path / "link"
    link.symlink_to(videos, target_is_directory=True)
    with pytest.raises(SystemExit):
        ev.check_output_dir(link / "eval", [videos])


def test_two_input_folders_with_one_name_are_refused(tmp_path):
    # Codex re-review Medium: archiveA/batch and archiveB/batch would share result files.
    a, b = tmp_path / "archiveA" / "batch", tmp_path / "archiveB" / "batch"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    with pytest.raises(SystemExit):
        ev.batch_names([a, b])
    assert ev.batch_names([a, tmp_path / "archiveA"]) == ["batch", "archiveA"]


def test_batch_names_differing_only_in_case_are_refused(tmp_path):
    # Codex round 3: Windows paths ignore case, so batch and BATCH share result files.
    a, b = tmp_path / "archiveA" / "batch", tmp_path / "archiveB" / "BATCH"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    with pytest.raises(SystemExit):
        ev.batch_names([a, b])


def test_resume_from_another_source_folder_is_refused(tmp_path):
    # Codex round 3: an earlier run of archiveA/batch must not make archiveB/batch look done.
    out = tmp_path / "eval"
    out.mkdir()
    old = {"batch": "batch", "video": "clip.mp4", "source": str((tmp_path / "archiveA" / "batch").resolve())}
    (out / "rows.jsonl").write_text(json.dumps(old) + "\n")
    with pytest.raises(SystemExit):
        ev.done_rows(out / "rows.jsonl", {"batch": (tmp_path / "archiveB" / "batch").resolve()})
    same = ev.done_rows(out / "rows.jsonl", {"batch": (tmp_path / "archiveA" / "batch").resolve()})
    assert same == {("batch", "clip.mp4")}


def test_summary_through_a_symlink_is_refused(tmp_path):
    # Codex round 3: out/summary.json linking into a video folder would overwrite a source.
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "original.json").write_text("{}")
    out = tmp_path / "eval"
    out.mkdir()
    (out / "summary.json").symlink_to(videos / "original.json")
    with pytest.raises(SystemExit):
        ev.main(["--out", str(out), "--only-truth", str(videos)])
    assert (videos / "original.json").read_text() == "{}"


def test_writes_through_a_symlinked_result_folder_are_refused(tmp_path):
    # Codex re-review Medium: out/ is safe but out/found links into a video folder.
    videos = tmp_path / "videos"
    videos.mkdir()
    out = tmp_path / "eval"
    out.mkdir()
    (out / "found").symlink_to(videos, target_is_directory=True)
    protected = ev.protected_dirs([videos])
    with pytest.raises(SystemExit):
        ev.ensure_safe(ev.result_path(out, "found", "videos", "clip.mp4", ".json"), protected)
    ev.ensure_safe(ev.result_path(out, "preview", "videos", "clip.mp4", ".jpg"), protected)


def test_results_are_kept_per_batch(tmp_path):
    a = ev.result_path(tmp_path, "found", "20261024", "clip.mp4", ".json")
    b = ev.result_path(tmp_path, "found", "20261108", "clip.mp4", ".json")
    assert a != b and a.parent.name == "20261024"


def _truth(tmp_path, at_s, video="clip.mp4"):
    d = tmp_path / "truth"
    d.mkdir(exist_ok=True)
    zones = {"battle_area": [[0, 0], [100, 0], [100, 100]]}
    (d / "clip.json").write_text(json.dumps(
        {"video": video, "video_sha256": "x", "at_s": at_s, "frame_size": [1920, 1080], "zones": zones}))
    return d


def test_truth_on_a_different_frame_is_not_graded(tmp_path):
    rows = [{"batch": "20261108", "video": "clip.mp4", "found": True, "frame_index": 390, "fps": 120.0}]
    out = tmp_path / "eval"
    out.mkdir()
    [row] = ev.truth_rows(_truth(tmp_path, at_s=1.0), out, rows, 8.0, 1280)
    assert row["status"] == "FRAME_MISMATCH"


def test_truth_for_a_name_in_two_batches_is_not_graded(tmp_path):
    rows = [{"batch": b, "video": "clip.mp4", "found": True, "frame_index": 390, "fps": 120.0}
            for b in ("20261024", "20261108")]
    out = tmp_path / "eval"
    out.mkdir()
    [row] = ev.truth_rows(_truth(tmp_path, at_s=3.25), out, rows, 8.0, 1280)
    assert row["status"] == "AMBIGUOUS"
