import json
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from bxjudge.autocalib import load_autocalib_config, mean_boundary_error
from bxjudge.calibration import ZONES, save_calibration
from bxjudge.calibration_store import (
    load_snapshot,
    save_snapshot,
    select_snapshot,
    unveto_calibration,
    veto_calibration,
    vetoed_ids,
    write_preview,
)
from bxjudge.config import load_thresholds
from tests.helpers import results_with_two_positions, stadium_scene


def th():
    return load_thresholds("config/thresholds.yaml")


def make_calib_dir(tmp_path, scene, name="scene"):
    """A calib folder holding the scene as a manual calibration; returns (calib_dir, out_dir)."""
    img, calib = scene
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(calib_dir / f"{name}.png"), img)
    calib.name = name
    calib.ref_frame = str(calib_dir / f"{name}.png")
    save_calibration(calib, calib_dir / f"{name}.json")
    return calib_dir, tmp_path / "out"


def test_manual_wins_when_similar(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame, _ = stadium_scene()
    s = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert s.kind == "manual" and s.source_sha256 and s.template_id is None
    assert s.source_path == str(calib / "scene.json")
    assert (out / "calibrations" / f"{s.id}.json").is_file()


def test_shifted_frame_gets_auto_snapshot_with_its_own_reference(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame, truth = stadium_scene(shift=(45, 20))
    s = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert s.kind == "auto" and s.template_id
    assert s.homography is not None and s.score >= load_autocalib_config().min_score
    assert np.array_equal(cv2.imread(s.calibration.ref_frame), frame)
    assert mean_boundary_error(s.calibration, truth, 1280) < 3.0
    assert not list(calib.glob("auto*")) and len(list(calib.glob("*.json"))) == 1
    tpl = load_snapshot(out, s.template_id)
    assert tpl.kind == "manual" and tpl.calibration.name == "scene"


def test_second_video_at_same_position_reuses_the_auto_snapshot(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    b = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    assert a.kind == "auto" and a.id == b.id


def test_different_position_gets_a_distinct_auto_snapshot(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    b = select_snapshot(stadium_scene(shift=(-50, 25))[0], calib, out, th(), load_autocalib_config())
    assert a.kind == b.kind == "auto" and a.id != b.id


def test_snapshot_survives_manual_file_edit_and_folder_move(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    s = select_snapshot(stadium_scene()[0], calib, out, th(), load_autocalib_config())
    (calib / "scene.json").write_text("{}")
    moved = tmp_path / "moved"
    shutil.copytree(out, moved)
    again = load_snapshot(moved, s.id)
    assert again.calibration.stadium == s.calibration.stadium and Path(again.calibration.ref_frame).is_file()
    assert Path(again.calibration.ref_frame).parent == (moved / "calibrations").resolve()


def test_unreadable_template_is_skipped(tmp_path, caplog):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    (calib / "broken.json").write_text(json.dumps({**json.loads((calib / "scene.json").read_text()), "name": "broken", "ref_frame": "missing.png"}))
    assert select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config()).kind == "auto"
    assert "broken" in caplog.text


def test_no_match_returns_none(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    assert select_snapshot(np.zeros((720, 1280, 3), np.uint8), calib, out, th(), load_autocalib_config()) is None


def test_missing_or_empty_calib_dir_returns_none(tmp_path):
    out = tmp_path / "out"
    frame = stadium_scene()[0]
    assert select_snapshot(frame, tmp_path / "nope", out, th(), load_autocalib_config()) is None


def test_snapshot_id_is_content_addressed(tmp_path):
    img, c = stadium_scene()
    a = save_snapshot(tmp_path, "manual", c, img, score=1.0)
    b = save_snapshot(tmp_path, "manual", c, img, score=1.0)
    assert a.id == b.id and len(a.id) == 16
    other = save_snapshot(tmp_path, "manual", c, img, score=0.9)  # per-video score: not identity
    assert other.id == a.id and other.score == 0.9 and a.score == 1.0
    assert len(list((tmp_path / "calibrations").glob("*.json"))) == 1
    moved = replace(c, name="other")
    assert save_snapshot(tmp_path, "manual", moved, img, score=1.0).id != a.id
    auto_a = save_snapshot(tmp_path, "auto", c, img, score=0.9, template_id=a.id)
    auto_b = save_snapshot(tmp_path, "auto", c, img, score=0.8, template_id=a.id)
    assert auto_a.id != auto_b.id  # an auto snapshot's alignment score is part of its identity


def test_load_unknown_snapshot_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path, "0123456789abcdef")


def test_preview_is_written_with_all_zones(tmp_path):
    img, c = stadium_scene()
    s = save_snapshot(tmp_path, "manual", c, img, score=0.95)
    p = write_preview(tmp_path, s)
    assert p == tmp_path / "calibrations" / f"{s.id}.preview.jpg"
    out = cv2.imread(str(p))
    assert out is not None and out.shape[:2] == img.shape[:2]
    assert len(ZONES) == 8


def test_one_manual_calibration_has_one_snapshot_across_frames(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    snaps = []
    for seed in range(4):
        frame = stadium_scene()[0].copy()
        rng = np.random.default_rng(seed)
        frame = np.clip(frame + rng.normal(0, 3 + seed, frame.shape), 0, 255).astype(np.uint8)
        snaps.append(select_snapshot(frame, calib, out, th(), load_autocalib_config()))
    assert all(s.kind == "manual" for s in snaps)
    assert len({s.id for s in snaps}) == 1
    assert len({s.score for s in snaps}) > 1  # the per-video similarity is still reported
    assert len(list((out / "calibrations").glob("*.json"))) == 1
    assert len(list((out / "calibrations").glob("*.png"))) == 1


def test_auto_template_id_is_the_manual_id_used_for_judging(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    auto = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    manual = select_snapshot(stadium_scene()[0], calib, out, th(), load_autocalib_config())
    assert auto.kind == "auto" and manual.kind == "manual"
    assert auto.template_id == manual.id
    assert len(list((out / "calibrations").glob("*.json"))) == 2  # one manual + one auto


def test_manual_wins_over_an_existing_auto_and_is_unrefined(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    img, tpl = stadium_scene()
    select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    s = select_snapshot(img, calib, out, th(), load_autocalib_config())
    assert s.kind == "manual" and s.calibration.stadium == tpl.stadium


def _edit_template(calib):
    d = json.loads((calib / "scene.json").read_text())
    d["launch_left"] = [[x + 7, y + 3] for x, y in d["launch_left"]]  # outside the rim region
    (calib / "scene.json").write_text(json.dumps(d))


def test_edited_template_makes_its_auto_snapshot_ineligible(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame = stadium_scene(shift=(45, 20))[0]
    a = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert select_snapshot(frame, calib, out, th(), load_autocalib_config()).id == a.id
    _edit_template(calib)
    b = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert b.kind == "auto" and b.id != a.id and b.template_id != a.template_id
    assert select_snapshot(frame, calib, out, th(), load_autocalib_config()).id == b.id
    assert load_snapshot(out, a.id).id == a.id  # the old snapshot stays on disk, untouched


def test_removed_template_makes_its_auto_snapshot_ineligible(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame = stadium_scene(shift=(45, 20))[0]
    assert select_snapshot(frame, calib, out, th(), load_autocalib_config()).kind == "auto"
    (calib / "scene.json").unlink()
    assert select_snapshot(frame, calib, out, th(), load_autocalib_config()) is None


# ---- vetoes (Task 4, ruling T4-1) ------------------------------------------------------


def test_veto_unknown_id_rejected(tmp_path):
    with pytest.raises(ValueError):
        veto_calibration(results_with_two_positions(tmp_path), "0" * 16)


def test_veto_file_records_time_and_note_and_unveto_restores(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    assert vetoed_ids(out) == set()
    veto_calibration(out, a.id, note="rim off by 10 px")
    assert vetoed_ids(out) == {a.id}
    entry = json.loads((out / "calibration_vetoes.json").read_text())[a.id]
    assert entry["note"] == "rim off by 10 px" and entry["at"]
    unveto_calibration(out, a.id)
    assert vetoed_ids(out) == set()
    unveto_calibration(out, a.id)  # idempotent
    with pytest.raises(ValueError):
        unveto_calibration(out, "0" * 16)


def test_veto_applies_to_auto_snapshots_only(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    manual = select_snapshot(stadium_scene()[0], calib, out, th(), load_autocalib_config())
    assert manual.kind == "manual"
    with pytest.raises(ValueError, match="manual"):
        veto_calibration(out, manual.id)
    assert not (out / "calibration_vetoes.json").exists()


def test_vetoed_auto_snapshot_is_never_reused(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    other = stadium_scene(shift=(46, 21))[0]  # same camera position, a different video frame
    assert select_snapshot(other, calib, out, th(), load_autocalib_config()).id == a.id
    # Its own fresh alignment would be a different snapshot (its own reference frame) ...
    fresh = select_snapshot(other, calib, tmp_path / "fresh", th(), load_autocalib_config())
    assert fresh.kind == "auto" and fresh.id != a.id
    # ... but a vetoed group is not re-aligned automatically from another of its videos.
    assert select_snapshot(other, calib, out, th(), load_autocalib_config(), {a.id}) is None
    assert len(list((out / "calibrations").glob("*.json"))) == 2  # template + a, nothing new


def test_fresh_alignment_equal_to_a_vetoed_id_returns_none(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame = stadium_scene(shift=(45, 20))[0]
    a = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert select_snapshot(frame, calib, tmp_path / "again", th(), load_autocalib_config()).id == a.id
    assert select_snapshot(frame, calib, out, th(), load_autocalib_config(), {a.id}) is None


def test_veto_leaves_other_positions_and_manual_selection_alone(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    far = stadium_scene(shift=(-50, 25))[0]
    b = select_snapshot(far, calib, out, th(), load_autocalib_config())
    assert b.kind == "auto" and b.id != a.id
    assert select_snapshot(far, calib, out, th(), load_autocalib_config(), {a.id}).id == b.id
    manual = select_snapshot(stadium_scene()[0], calib, out, th(), load_autocalib_config(), {a.id})
    assert manual.kind == "manual"


@pytest.mark.parametrize("document", [[], None, {"snapshot": []}, {"snapshot": None}, {}])
def test_load_snapshot_rejects_malformed_document_or_metadata_shape(tmp_path, document):
    directory = tmp_path / "calibrations"
    directory.mkdir()
    (directory / "broken.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="snapshot"):
        load_snapshot(tmp_path, "broken")


@pytest.mark.parametrize("policy", [
    {"min_score": 1.0}, {"max_shift_frac": 0.01}, {"max_rotation_deg": 0.1},
    {"max_scale": 1.01}, {"min_scale": 1.1}, {"max_perspective": 0.0},
    {"min_consistent_fraction": 1.0},
])
def test_cached_auto_obeys_current_policy_like_fresh_selection(tmp_path, policy):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame = stadium_scene(shift=(45, 20), rotation_deg=3, scale=1.06)[0]
    cfg = load_autocalib_config()
    original = select_snapshot(frame, calib, out, th(), cfg)
    assert original.kind == "auto"
    cfg = replace(cfg, **policy)
    assert select_snapshot(frame, calib, tmp_path / "fresh", th(), cfg) is None
    assert select_snapshot(frame, calib, out, th(), cfg) is None
    assert load_snapshot(out, original.id) == original


def test_eligible_cache_reuse_does_not_repeat_estimation(tmp_path, monkeypatch):
    import bxjudge.calibration_store as store

    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame = stadium_scene(shift=(45, 20))[0]
    original = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    def unexpected(*args):
        raise AssertionError("eligible grouping must not repeat full estimation")
    monkeypatch.setattr(store, "estimate_alignment", unexpected)
    assert select_snapshot(frame, calib, out, th(), load_autocalib_config()).id == original.id


@pytest.mark.parametrize("axis, policy", [
    (0, {"max_scale": 1.03}), (1, {"max_scale": 1.03}),
    (0, {"min_scale": 0.97}), (1, {"min_scale": 0.97}),
])
def test_cached_policy_checks_principal_stretches_not_average_scale(tmp_path, axis, policy):
    image, _ = stadium_scene()
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    scales = [1.06, 0.94] if axis == 0 else [0.94, 1.06]
    matrix = np.array([[scales[0], 0, 45 + 640 * (1 - scales[0])],
                       [0, scales[1], 20 + 360 * (1 - scales[1])], [0, 0, 1]])
    frame = cv2.warpPerspective(image, matrix, (1280, 720))
    cfg = load_autocalib_config()
    from bxjudge.autocalib import validate_alignment, warp_calibration
    from bxjudge.calibration import load_calibration

    template = load_calibration(calib / "scene.json")
    # Isolate cache geometry policy from the coarse estimator's convergence basin: this
    # exact known transform is a legitimate accepted snapshot, not an approximate ECC fit.
    alignment = validate_alignment(frame, template, image, matrix, cfg)
    assert alignment is not None
    manual = select_snapshot(image, calib, out, th(), cfg)
    original = save_snapshot(out, "auto", warp_calibration(template, alignment, (1280, 720), "auto-scene"),
                             frame, score=alignment.score, template_id=manual.id,
                             homography=alignment.homography)
    assert select_snapshot(frame, calib, out, th(), cfg).id == original.id
    cfg = replace(cfg, **policy)
    assert cfg.min_scale < alignment.scale < cfg.max_scale  # average scale alone would pass
    assert validate_alignment(frame, template, image, matrix, cfg) is None
    fresh = select_snapshot(frame, calib, tmp_path / "fresh", th(), cfg)
    cached = select_snapshot(frame, calib, out, th(), cfg)
    assert cached is None or cached.id != original.id
    assert (cached.id if cached else None) == (fresh.id if fresh else None)



@pytest.mark.parametrize("x", [24, 900])
def test_old_locally_inconsistent_cache_cannot_bypass_new_policy(tmp_path, x):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame, _ = stadium_scene(shift=(49.019032876749094, -34.65765420413904),
                             rotation_deg=-0.9879950469803092, scale=1.018190471590685)
    displaced = cv2.warpAffine(frame, np.float32([[1, 0, 25], [0, 1, 0]]), (1280, 720))
    frame[8:366, x:x + 334] = displaced[8:366, x:x + 334]
    cfg = load_autocalib_config()
    prior = select_snapshot(frame, calib, out, th(), replace(cfg, min_consistent_fraction=0.125))
    assert prior is not None and prior.kind == "auto"
    assert select_snapshot(frame, calib, tmp_path / "fresh", th(), cfg) is None
    assert select_snapshot(frame, calib, out, th(), cfg) is None
    assert load_snapshot(out, prior.id) == prior
