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
    (tmp_path / "archiveA" / "batch").mkdir(parents=True)
    (tmp_path / "archiveA" / "batch" / "clip.mp4").write_bytes(b"clip")  # still there
    old ={"batch": "batch", "video": "clip.mp4", "source": str((tmp_path / "archiveA" / "batch").resolve())}
    (out / "rows.jsonl").write_text(json.dumps(old) + "\n")
    with pytest.raises(SystemExit):
        ev.done_rows(out / "rows.jsonl", {"batch": (tmp_path / "archiveB" / "batch").resolve()})
    same = ev.done_rows(out / "rows.jsonl", {"batch": (tmp_path / "archiveA" / "batch").resolve()})
    assert same == {("batch", "clip.mp4")}


def test_only_truth_protects_the_recorded_source_folders(tmp_path):
    # Codex round 4: --only-truth without folders must still protect where the videos were.
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "original.json").write_text("{}")
    out = tmp_path / "eval"
    out.mkdir()
    row = {"batch": "videos", "video": "clip.mp4", "source": str(videos.resolve()), "found": False,
           "seconds": 1.0}
    (out / "rows.jsonl").write_text(json.dumps(row) + "\n")
    (out / "summary.json").symlink_to(videos / "original.json")
    with pytest.raises(SystemExit):
        ev.main(["--out", str(out), "--only-truth"])
    assert (videos / "original.json").read_text() == "{}"


def test_only_truth_without_recorded_sources_needs_the_folders(tmp_path):
    out = tmp_path / "eval"
    out.mkdir()
    (out / "rows.jsonl").write_text(json.dumps({"batch": "b", "video": "c.mp4", "found": False,
                                                "seconds": 1.0}) + "\n")
    with pytest.raises(SystemExit):
        ev.main(["--out", str(out), "--only-truth"])


def test_symlinks_inside_the_reference_tree_are_refused(tmp_path):
    # Codex round 4: rglob does not descend into a linked folder; refuse links in the tree.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    ref = tmp_path / "eval" / "reference" / "batch"
    ref.mkdir(parents=True)
    (ref / "calibrations").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(SystemExit):
        ev.ensure_tree_has_no_links(ref)
    ev.ensure_tree_has_no_links(tmp_path / "eval" / "reference" / "missing")  # absent: fine


def test_a_junction_at_the_reference_root_is_refused(tmp_path, monkeypatch):
    # Codex round 5: the root was only checked with is_symlink; Windows junctions are not.
    root = tmp_path / "eval" / "reference" / "batch" / "clip"
    root.mkdir(parents=True)
    real = ev.os.path.isjunction
    monkeypatch.setattr(ev.os, "name", "nt")
    # compare as text: with os.name patched, Path() would build a WindowsPath
    monkeypatch.setattr(ev.os.path, "isjunction", lambda p: str(p) == str(root) or real(p))
    with pytest.raises(SystemExit):
        ev.ensure_tree_has_no_links(root)


def test_resume_compares_batch_names_without_case(tmp_path):
    # Codex round 4: archiveA/batch earlier, archiveB/BATCH now -> same files on Windows.
    out = tmp_path / "eval"
    out.mkdir()
    old = {"batch": "batch", "video": "clip.mp4", "source": str((tmp_path / "archiveA" / "batch").resolve())}
    (out / "rows.jsonl").write_text(json.dumps(old) + "\n")
    with pytest.raises(SystemExit):
        ev.done_rows(out / "rows.jsonl", {"BATCH": (tmp_path / "archiveB" / "BATCH").resolve()})


def test_reference_folders_are_per_video(tmp_path):
    # Codex round 4: parallel workers sharing one snapshot folder race on its temp file.
    a = ev.reference_dir(tmp_path, "batch", "a.mp4")
    b = ev.reference_dir(tmp_path, "batch", "b.mp4")
    assert a != b and a.parent == b.parent


def test_truth_for_another_video_with_the_same_name_is_not_graded(tmp_path):
    # Codex round 4: frame and name match, but the truth was clicked on a different file.
    rows = [{"batch": "20261108", "video": "clip.mp4", "found": True, "frame_index": 390,
             "fps": 120.0, "video_sha256": "a" * 64}]
    out = tmp_path / "eval"
    out.mkdir()
    [row] = ev.truth_rows(_truth(tmp_path, at_s=3.25), out, rows, 8.0, 1280)  # truth sha "x"
    assert row["status"] == "VIDEO_MISMATCH"


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


def test_resuming_with_another_setup_is_refused(tmp_path):
    # Codex review 2026-11-12 Medium: rail results then --keypoints, or new weights, in one --out
    out = tmp_path / "eval"
    rail = {"method": "rail", "stadium_finder_config_sha256": "a", "stadium_model_sha256": "b"}
    ev.check_run_manifest(out, rail, False, [])
    ev.check_run_manifest(out, rail, True, [])  # same setup: resumes
    kp = {**rail, "method": "keypoints", "keypoint_weights_sha256": ["w1"]}
    with pytest.raises(SystemExit):
        ev.check_run_manifest(out, kp, True, [])
    out2 = tmp_path / "eval2"
    ev.check_run_manifest(out2, kp, False, [])
    with pytest.raises(SystemExit):
        ev.check_run_manifest(out2, {**kp, "keypoint_weights_sha256": ["w2"]}, True, [])


def test_rows_without_a_manifest_are_not_resumed(tmp_path):
    # Codex re-review 2026-11-12: results without run.json cannot be shown to share the setup
    out = tmp_path / "eval"
    out.mkdir()
    rail = {"method": "rail", "stadium_finder_config_sha256": "a", "stadium_model_sha256": "b"}
    for manifest in (rail, {**rail, "method": "keypoints"}):
        with pytest.raises(SystemExit):
            ev.check_run_manifest(out, manifest, True, [])
    assert not (out / "run.json").exists()


def test_manifest_records_device_and_reference_comparison():
    a = ev.run_manifest(False, reference=True)
    b = ev.run_manifest(False, reference=False)
    assert a != b and a["reference"] is True


def test_result_files_must_stay_inside_out(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out = tmp_path / "eval"
    out.mkdir()
    (out / "preview").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(SystemExit):
        ev.ensure_inside(ev.result_path(out, "preview", "b", "clip.mp4", ".jpg"), out)
    ev.ensure_inside(ev.result_path(out, "found", "b", "clip.mp4", ".json"), out)


def test_resume_reruns_model_load_errors_and_refuses_changed_videos(tmp_path):
    src = tmp_path / "batch"
    src.mkdir()
    (src / "a.mp4").write_bytes(b"one")
    (src / "b.mp4").write_bytes(b"two")
    out = tmp_path / "eval"
    out.mkdir()
    rows = [{"batch": "batch", "video": "a.mp4", "source": str(src.resolve()),
             "video_sha256": ev._sha256(src / "a.mp4"), "found": True},
            {"batch": "batch", "video": "b.mp4", "source": str(src.resolve()),
             "video_sha256": ev._sha256(src / "b.mp4"), "found": False,
             "error": f"{ev.RETRY_ERROR_PREFIX} OutOfMemoryError: CUDA"}]
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert ev.done_rows(out / "rows.jsonl", {"batch": src.resolve()}) == {("batch", "a.mp4")}
    (src / "a.mp4").write_bytes(b"another video, same name")
    with pytest.raises(SystemExit):
        ev.done_rows(out / "rows.jsonl", {"batch": src.resolve()})


def test_resume_reruns_videos_not_found_after_inference_errors(tmp_path):
    # Codex re-review 2026-11-12 (round 2) Medium: a video not found while its frames' keypoint
    # inference ran out of memory is not a result; one found despite an error is.
    src = tmp_path / "batch"
    src.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (src / name).write_bytes(name.encode())
    out = tmp_path / "eval"
    out.mkdir()
    base = {"batch": "batch", "source": str(src.resolve())}
    rows = [{**base, "video": "a.mp4", "found": False, "keypoint_errors": ["frame 0: OOM"]},
            {**base, "video": "b.mp4", "found": True, "keypoint_errors": ["frame 0: OOM"]},
            {**base, "video": "c.mp4", "found": False, "keypoint_errors": []}]
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert ev.done_rows(out / "rows.jsonl", {"batch": src.resolve()}) == {
        ("batch", "b.mp4"), ("batch", "c.mp4")}


def test_resume_refuses_a_recorded_video_that_is_gone(tmp_path):
    src = tmp_path / "batch"
    src.mkdir()
    out = tmp_path / "eval"
    out.mkdir()
    row = {"batch": "batch", "video": "gone.mp4", "source": str(src.resolve()), "found": True,
           "video_sha256": "0" * 64}
    (out / "rows.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(SystemExit, match="gone"):
        ev.done_rows(out / "rows.jsonl", {"batch": src.resolve()})


@pytest.mark.parametrize("retry", [
    {"error": f"{ev.RETRY_ERROR_PREFIX} OutOfMemoryError: CUDA"},
    {"keypoint_errors": ["frame 0: OOM"]}])
@pytest.mark.parametrize("change", ["gone", "changed"])
def test_resume_checks_the_video_of_a_row_left_to_retry(tmp_path, retry, change):
    # Codex re-review 2026-11-12 (round 3) Low: a row left to retry skipped the file and
    # SHA-256 checks, so a gone video stayed in the summary and a changed one re-ran unseen.
    src = tmp_path / "batch"
    src.mkdir()
    (src / "a.mp4").write_bytes(b"one")
    out = tmp_path / "eval"
    out.mkdir()
    row = {"batch": "batch", "video": "a.mp4", "source": str(src.resolve()), "found": False,
           "video_sha256": ev._sha256(src / "a.mp4"), **retry}
    (out / "rows.jsonl").write_text(json.dumps(row) + "\n")
    assert ev.done_rows(out / "rows.jsonl", {"batch": src.resolve()}) == set()
    if change == "gone":
        (src / "a.mp4").unlink()
    else:
        (src / "a.mp4").write_bytes(b"another video, same name")
    with pytest.raises(SystemExit, match=change):
        ev.done_rows(out / "rows.jsonl", {"batch": src.resolve()})


@pytest.mark.parametrize("name", ["rows.jsonl", "summary.json"])
def test_rows_and_summary_links_out_of_out_are_refused(tmp_path, name):
    # Codex re-review 2026-11-12 (round 2) Low: a link under --out must not carry the rows or
    # the summary into another (unprotected) folder.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out = tmp_path / "eval"
    out.mkdir()
    (out / name).symlink_to(elsewhere / name)
    with pytest.raises(SystemExit, match="outside"):
        ev.main(["--out", str(out), "--only-truth"])
    assert not (elsewhere / name).exists()


def test_summary_counts_a_rerun_video_once():
    rows = [{"batch": "b", "video": "a.mp4", "found": False, "error": "keypoint model: x"},
            {"batch": "b", "video": "a.mp4", "found": True}]
    assert ev.latest_rows(rows) == [rows[1]]


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
