"""scripts/eval_dynamic_calibration.py (plan 2026-11-12-dynamic-calibration, Task 8)."""
from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from bxjudge import PROJECT_ROOT
from bxjudge.camera_motion import CameraMotion
from tests.helpers import make_video, shipped_autocalib, stadium_scene
from tests.test_stadium_finder import synth_model  # noqa: F401  (fixture)


def _load():
    spec = importlib.util.spec_from_file_location(
        "eval_dynamic_calibration", PROJECT_ROOT / "scripts" / "eval_dynamic_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load()


def _batch(root, name, videos):
    folder = root / name
    folder.mkdir(parents=True)
    for v in videos:
        make_video(folder / v, seconds=1, fps=30, size=(160, 90))
    return folder


def test_modes_summarises_per_batch(tmp_path, monkeypatch):
    a = _batch(tmp_path / "videos", "20261101", ["h1.mp4", "h2.mp4", "f1.mp4"])
    b = _batch(tmp_path / "videos", "20261108", ["f2.mp4"])
    speeds = {"h1.mp4": 60.0, "h2.mp4": 70.0, "f1.mp4": 5.0, "f2.mp4": 6.0}
    monkeypatch.setattr(ev, "measure_video_motion",
                        lambda video, *a, **k: CameraMotion(speeds[video.name], 9))
    out = tmp_path / "eval"
    assert ev.main(["modes", "--out", str(out), str(a), str(b)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["20261101"]["handheld"] == 2 and summary["20261101"]["fixed"] == 1
    assert summary["20261101"]["handheld_fraction"] == pytest.approx(0.667)
    assert summary["20261108"] == {**summary["20261108"], "videos": 1, "handheld": 0}
    assert summary["20261101"]["px_per_s"]["median"] == 60.0
    # Resumable: a second run measures nothing again.
    monkeypatch.setattr(ev, "measure_video_motion", lambda *a, **k: pytest.fail("re-measured"))
    assert ev.main(["modes", "--out", str(out), str(a), str(b)]) == 0
    assert len((out / "rows.jsonl").read_text().splitlines()) == 4


class _IdentityLocator:
    last_reason = None

    def __init__(self, c):
        self.c = c

    def locate(self, image):
        from bxjudge.stadium_locator import LocatedStadium, alignment_from_homography

        return LocatedStadium(np.eye(3), alignment_from_homography(self.c, np.eye(3), 0.9), None)


def test_geometry_reports_coverage_and_refind_drift(tmp_path, monkeypatch):
    from bxjudge.calibration_store import save_snapshot
    from tests.test_stadium_finder import write_video

    ref, c = stadium_scene()
    folder = tmp_path / "videos" / "20261101"
    folder.mkdir(parents=True)
    write_video(folder / "clip.mp4", [ref] * 75)
    snap = save_snapshot(tmp_path / "snaps", "keypoints", c, ref, score=0.9, source_frame=0,
                         agreeing_frames=2,
                         homography=np.eye(3).tolist(), autocalib=shipped_autocalib())
    found = SimpleNamespace(method="keypoints", frame_index=0, agreeing_frames=2)
    monkeypatch.setattr(ev, "measure_video_motion", lambda *a, **k: CameraMotion(60.0, 9))
    monkeypatch.setattr(ev, "keypoint_snapshot", lambda *a, **k: (snap, found))
    monkeypatch.setattr(ev.KeypointLocator, "for_snapshot",
                        classmethod(lambda cls, tools, s, cal, ref_image: _IdentityLocator(cal)))
    row = ev.geometry_row("20261101", folder / "clip.mp4", None, ev.load_camera_policy(),
                          tmp_path / "eval", "cpu")
    assert row["result"] == "tracked" and row["mode"] == "handheld"
    assert row["frames"] == 75 and row["coverage"] == 1.0 and row["invalidated"] == 0
    assert row["refinds"] == {"agree": 3}  # frames 30, 60 and the tail 74
    assert len(row["refind_disagreement_px"]) == 3
    assert max(row["refind_disagreement_px"]) < 2.0
    summary = ev.summarise_geometry([row])
    assert summary["20261101/handheld"]["coverage_of_tracked_frames"] == 1.0
    assert summary["20261101/handheld"]["refind_disagreement_px"]["n"] == 3


def test_outputs_stay_inside_out(tmp_path):
    videos = _batch(tmp_path / "videos", "20261101", ["a.mp4"])
    with pytest.raises(SystemExit):
        ev.main(["modes", "--out", str(videos / "eval"), str(videos)])
    outside = tmp_path / "elsewhere.jsonl"
    out = tmp_path / "eval"
    out.mkdir()
    (out / "rows.jsonl").symlink_to(outside)
    with pytest.raises(SystemExit):
        ev.main(["modes", "--out", str(out), str(videos)])
    assert not outside.exists()


# ---- Codex review R1 (Task 8) ---------------------------------------------------------


def _labelled_dataset(root, h_truth, names=("task1",), split="val"):
    """A tiny keypoint dataset: synthetic stadium frames seen through ``h_truth`` with YOLO-pose
    labels of the reference points (visibility 2)."""
    import cv2

    from tests.test_stadium_finder import _frame_for
    from tests.test_stadium_keypoints import SYNTH_REF
    from bxjudge.stadium_keypoints import KEYPOINTS
    from tests.test_stadium_finder import apply_h

    images, labels = root / "images" / split, root / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    xy = apply_h(h_truth, [SYNTH_REF[k] for k in KEYPOINTS])
    line = "0 0.5 0.5 1 1 " + " ".join(f"{x / 1280:.6f} {y / 720:.6f} 2" for x, y in xy)
    for n in names:
        cv2.imwrite(str(images / f"{n}.jpg"), _frame_for(h_truth))
        (labels / f"{n}.txt").write_text(line)
    return root


class _ShiftedFinder:
    """Predicts the reference points through ``view`` (model px -> frame px)."""

    def __init__(self, reference, view):
        self.reference, self.view = reference, view

    def predict(self, frame):
        from tests.test_stadium_keypoints import _points

        return _points(self.view)


def test_refind_accuracy_measures_a_known_offset(tmp_path, synth_model):  # noqa: F811
    import hashlib

    from tests.test_stadium_keypoints import H_VIEW, _ref

    cfg, model = synth_model
    data = _labelled_dataset(tmp_path / "kp", H_VIEW)
    shift = np.array([[1.0, 0, 6.0], [0, 1.0, 0], [0, 0, 1.0]])
    tools = SimpleNamespace(model=model, cfg=cfg, finder=_ShiftedFinder(_ref(model), shift @ H_VIEW))
    image = data / "images" / "val" / "task1.jpg"
    row = ev.refind_row(image, data / "labels" / "val" / "task1.txt", "val", "20261101", tools)
    assert row["result"] == "fit"
    assert row["worst_zone_px"] == pytest.approx(6.0, abs=1.5)
    exact = ev.refind_row(image, data / "labels" / "val" / "task1.txt", "val", "20261101",
                          SimpleNamespace(model=model, cfg=cfg,
                                          finder=_ShiftedFinder(_ref(model), H_VIEW)))
    assert exact["worst_zone_px"] < 1.5
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"frames": [
        {"image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "batch": "20261101"}]}))
    assert ev.manifest_batches(manifest) == {
        hashlib.sha256(image.read_bytes()).hexdigest(): "20261101"}


def test_degenerate_labels_are_an_invalid_truth_not_a_crash(tmp_path, synth_model):  # noqa: F811
    from tests.test_stadium_keypoints import H_VIEW, _ref

    cfg, model = synth_model
    data = _labelled_dataset(tmp_path / "kp", H_VIEW)
    label = data / "labels" / "val" / "task1.txt"
    values = label.read_text().split()
    for i in range(10):  # every point on one spot
        values[5 + 3 * i], values[6 + 3 * i] = "0.5", "0.5"
    label.write_text(" ".join(values))
    tools = SimpleNamespace(model=model, cfg=cfg, finder=_ShiftedFinder(_ref(model), H_VIEW))
    row = ev.refind_row(data / "images" / "val" / "task1.jpg", label, "val", "x", tools)
    assert row["result"] == "invalid_truth"


def test_rows_are_keyed_by_split_and_errors_are_retried(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps(r) for r in [
        {"batch": "b", "split": "train", "video": "a.jpg", "result": "fit"},
        {"batch": "b", "split": "val", "video": "a.jpg", "result": "no_fit"},
        {"batch": "b", "split": "val", "video": "c.jpg", "result": "error: RuntimeError: OOM"},
    ]) + "\n")
    assert set(ev._done(rows)) == {("", "train", "a.jpg"), ("", "val", "a.jpg")}
    assert len(ev._rows(rows)) == 3
    # An image whose batch changed (new SHA, new manifest) is the same image: latest wins.
    with rows.open("a") as fh:
        fh.write(json.dumps({"batch": "unmatched", "split": "val", "video": "a.jpg",
                             "result": "fit"}) + "\n")
    latest = [r for r in ev._rows(rows) if (r["split"], r["video"]) == ("val", "a.jpg")]
    assert len(latest) == 1 and latest[0]["batch"] == "unmatched"


def test_resume_refuses_another_configuration(tmp_path, monkeypatch):
    a = _batch(tmp_path / "videos", "20261101", ["h1.mp4"])
    monkeypatch.setattr(ev, "measure_video_motion", lambda video, *x, **k: CameraMotion(60.0, 9))
    out = tmp_path / "eval"
    assert ev.main(["modes", "--out", str(out), str(a)]) == 0
    custom = tmp_path / "camera.yaml"
    doc = __import__("yaml").safe_load((PROJECT_ROOT / "config" / "camera_mode.yaml").read_text())
    doc["detection"]["handheld_px_per_s"] = 30.0
    custom.write_text(__import__("yaml").safe_dump(doc))
    with pytest.raises(SystemExit, match="manifest"):
        ev.main(["modes", "--out", str(out), "--camera-mode", str(custom), str(a)])
    with pytest.raises(SystemExit, match="manifest"):  # another subcommand, same folder
        ev.main(["geometry", "--out", str(out), str(a)])


def test_a_custom_dataset_is_protected(tmp_path):
    data = tmp_path / "kp"
    (data / "images").mkdir(parents=True)
    with pytest.raises(SystemExit):
        ev.main(["refind-accuracy", "--dataset", str(data), "--out", str(data / "eval"),
                 "--manifest", str(tmp_path / "m.json"), "--device", "cpu"])
    assert not (data / "eval").exists()


# ---- Codex re-review R2 (Task 8) ------------------------------------------------------


def test_resume_refuses_another_tracker_configuration(tmp_path, monkeypatch):
    a = _batch(tmp_path / "videos", "20261101", ["h1.mp4"])
    monkeypatch.setattr(ev, "measure_video_motion", lambda video, *x, **k: CameraMotion(60.0, 9))
    out = tmp_path / "eval"
    assert ev.main(["modes", "--out", str(out), str(a)]) == 0
    real = ev.load_dynamic_policy
    monkeypatch.setattr(ev, "load_dynamic_policy", lambda *x: {**real(), "confirmations": 5})
    with pytest.raises(SystemExit, match="manifest"):
        ev.main(["modes", "--out", str(out), str(a)])


def test_a_batch_name_stays_bound_to_its_source_folder(tmp_path, monkeypatch):
    a = _batch(tmp_path / "A", "20261101", ["clip.mp4"])
    b = _batch(tmp_path / "B", "20261101", ["clip.mp4"])
    c = _batch(tmp_path / "C", "20261108", ["clip.mp4"])
    monkeypatch.setattr(ev, "measure_video_motion", lambda video, *x, **k: CameraMotion(60.0, 9))
    out = tmp_path / "eval"
    assert ev.main(["modes", "--out", str(out), str(a)]) == 0
    with pytest.raises(SystemExit, match="20261101"):
        ev.main(["modes", "--out", str(out), str(b)])
    with pytest.raises(SystemExit, match="20261101"):
        ev.main(["modes", "--out", str(tmp_path / "eval2"), str(a), str(b)])
    assert ev.main(["modes", "--out", str(out), str(a), str(c)]) == 0  # another batch: appended
    assert len((out / "rows.jsonl").read_text().splitlines()) == 2


def test_a_changed_video_or_label_is_measured_again(tmp_path, monkeypatch):
    import os

    a = _batch(tmp_path / "videos", "20261101", ["h1.mp4"])
    calls = []
    monkeypatch.setattr(ev, "measure_video_motion",
                        lambda video, *x, **k: calls.append(video.name) or CameraMotion(60.0, 9))
    out = tmp_path / "eval"
    ev.main(["modes", "--out", str(out), str(a)])
    ev.main(["modes", "--out", str(out), str(a)])
    assert calls == ["h1.mp4"]
    st = (a / "h1.mp4").stat()
    os.utime(a / "h1.mp4", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    ev.main(["modes", "--out", str(out), str(a)])
    assert calls == ["h1.mp4", "h1.mp4"]
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"batch": "b", "split": "val", "video": "x.jpg", "result": "fit",
                                "image_sha256": "1", "label_sha256": "2"}) + "\n")
    done = ev._done(rows)
    assert ev._is_done(done, ("", "val", "x.jpg"), {"image_sha256": "1", "label_sha256": "2"})
    assert not ev._is_done(done, ("", "val", "x.jpg"), {"image_sha256": "1",
                                                         "label_sha256": "3"})


def test_snapshot_writes_cannot_escape_through_a_nested_link(tmp_path):
    videos = _batch(tmp_path / "videos", "20261101", ["a.mp4"])
    out = tmp_path / "eval"
    (out / "calibrations-run").mkdir(parents=True)
    (out / "calibrations-run" / "calibrations").symlink_to(videos, target_is_directory=True)
    with pytest.raises(SystemExit):
        ev.main(["geometry", "--out", str(out), str(videos), "--device", "cpu"])
    assert sorted(p.name for p in videos.iterdir()) == ["a.mp4"]


def test_a_file_link_inside_the_snapshot_folder_is_refused_before_any_write(tmp_path):
    videos = _batch(tmp_path / "videos", "20261101", ["a.mp4"])
    out = tmp_path / "eval"
    snaps = out / "calibrations-run" / "calibrations"
    snaps.mkdir(parents=True)
    (snaps / "0123456789abcdef.png").symlink_to(videos / "a.mp4")
    with pytest.raises(SystemExit, match="link"):
        ev.main(["geometry", "--out", str(out), str(videos), "--device", "cpu"])
    assert not (out / "manifest.json").exists()


def test_a_changed_batch_mapping_re_measures_and_replaces_the_image(tmp_path, monkeypatch):
    import hashlib

    data = tmp_path / "kp"
    (data / "images" / "val").mkdir(parents=True)
    (data / "labels" / "val").mkdir(parents=True)
    (data / "images" / "val" / "x.jpg").write_bytes(b"image bytes")
    (data / "labels" / "val" / "x.txt").write_text("0 0.5 0.5 1 1")
    sha = hashlib.sha256(b"image bytes").hexdigest()
    manifest = tmp_path / "m" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"frames": []}))
    monkeypatch.setattr(ev, "load_stadium_tools", lambda device=None: object())
    calls = []
    monkeypatch.setattr(ev, "refind_row", lambda image, label, split, batch, tools: calls.append(
        batch) or {"image": image.name, "split": split, "batch": batch, "result": "fit",
                   "worst_zone_px": 1.0})
    out = tmp_path / "eval"
    args = ["refind-accuracy", "--dataset", str(data), "--manifest", str(manifest),
            "--out", str(out), "--device", "cpu"]
    assert ev.main(args) == 0
    assert ev.main(args) == 0
    assert calls == ["unmatched"]
    manifest.write_text(json.dumps({"frames": [{"image_sha256": sha, "batch": "20261101"}]}))
    assert ev.main(args) == 0
    assert calls == ["unmatched", "20261101"]
    summary = json.loads((out / "summary.json").read_text())
    assert list(summary) == ["20261101/val"]


def test_completion_follows_the_latest_row_not_an_older_success(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps(r) for r in [
        {"batch": "A", "split": "val", "video": "x.jpg", "result": "fit", "image_sha256": "1"},
        {"batch": "B", "split": "val", "video": "x.jpg", "result": "error: RuntimeError: OOM",
         "image_sha256": "1"},
    ]) + "\n")
    done = ev._done(rows)
    # Mapped back to A: the latest row is B's error, so it is measured again.
    assert not ev._is_done(done, ("", "val", "x.jpg"), {"image_sha256": "1", "batch": "A"})
    assert [r["batch"] for r in ev._rows(rows)] == ["B"]
