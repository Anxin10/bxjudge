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
from tests.helpers import results_with_two_positions, shipped_autocalib, stadium_scene


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


@pytest.mark.parametrize('target', ['same', 'nested', 'ancestor', 'symlink'])
def test_select_snapshot_rejects_overlapping_output_before_writing(tmp_path, target):
    import hashlib

    frame, scene = stadium_scene()
    calib, _ = make_calib_dir(tmp_path, (frame, scene))
    out = {'same': calib, 'nested': calib / 'generated', 'ancestor': tmp_path,
           'symlink': tmp_path / 'alias'}[target]
    if target == 'symlink':
        out.symlink_to(calib, target_is_directory=True)
    before = {p.relative_to(calib): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in calib.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='outside --calib'):
        select_snapshot(frame, calib, out, th(), load_autocalib_config())
    after = {p.relative_to(calib): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in calib.rglob('*') if p.is_file()}
    assert after == before and not (out / 'calibrations').exists()


def test_shifted_frame_gets_auto_snapshot_with_its_own_reference(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame, truth = stadium_scene(shift=(45, 20))
    s = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert s.kind == "auto" and s.template_id
    assert s.homography is not None and s.score >= load_autocalib_config().min_score
    assert np.array_equal(cv2.imread(s.calibration.ref_frame), frame)
    assert mean_boundary_error(s.calibration, truth, 1280) < 3.0
    assert not list(calib.glob("auto*")) and len(list(calib.glob("*.json"))) == 1
    tpl = load_snapshot(out, s.template_id, autocalib=shipped_autocalib())
    assert tpl.kind == "manual" and tpl.calibration.name == "scene"


def test_same_frame_gets_the_same_auto_snapshot(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    b = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    assert a.kind == "auto" and a.id == b.id


def test_auto_snapshot_does_not_depend_on_videos_judged_before(tmp_path):
    # Every video is aligned on its own first-second frame: a video at the camera position of
    # an earlier video gets the snapshot it would get alone, whatever else is in the batch.
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    first = stadium_scene(shift=(45, 20))[0]
    other = stadium_scene(shift=(46, 21))[0]  # same camera position, a different frame
    alone = select_snapshot(other, calib, tmp_path / "alone", th(), load_autocalib_config())
    a = select_snapshot(first, calib, out, th(), load_autocalib_config())
    b = select_snapshot(other, calib, out, th(), load_autocalib_config())
    assert a.kind == b.kind == alone.kind == "auto"
    assert b.id == alone.id != a.id
    assert np.array_equal(cv2.imread(b.calibration.ref_frame), other)


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
    again = load_snapshot(moved, s.id, autocalib=shipped_autocalib())
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
    a = save_snapshot(tmp_path, "manual", c, img, score=1.0, autocalib=shipped_autocalib())
    b = save_snapshot(tmp_path, "manual", c, img, score=1.0, autocalib=shipped_autocalib())
    assert a.id == b.id and len(a.id) == 16
    other = save_snapshot(tmp_path, "manual", c, img, score=0.9, autocalib=shipped_autocalib())  # per-video score: not identity
    assert other.id == a.id and other.score == 0.9 and a.score == 1.0
    assert len(list((tmp_path / "calibrations").glob("*.json"))) == 1
    moved = replace(c, name="other")
    assert save_snapshot(tmp_path, "manual", moved, img, score=1.0, autocalib=shipped_autocalib()).id != a.id
    auto_a = save_snapshot(tmp_path, "auto", c, img, score=0.9, template_id=a.id, autocalib=shipped_autocalib())
    auto_b = save_snapshot(tmp_path, "auto", c, img, score=0.8, template_id=a.id, autocalib=shipped_autocalib())
    assert auto_a.id != auto_b.id  # an auto snapshot's alignment score is part of its identity


def test_keypoint_snapshot_round_trips_and_score_is_part_of_identity(tmp_path):
    img, c = stadium_scene()
    h = ((1.0, 0.0, 5.0), (0.0, 1.0, 7.0), (0.0, 0.0, 1.0))
    a = save_snapshot(tmp_path, "keypoints", c, img, score=0.91, homography=h,
                      autocalib=shipped_autocalib())
    b = save_snapshot(tmp_path, "keypoints", c, img, score=0.92, homography=h,
                      autocalib=shipped_autocalib())
    assert a.kind == "keypoints" and a.homography == h and a.score == pytest.approx(0.91)
    assert a.template_id is None and a.template_set is None and a.source_path is None
    assert a.id != b.id
    assert load_snapshot(tmp_path, a.id, autocalib=shipped_autocalib()) == a


def test_keypoint_snapshot_keeps_the_exact_score_and_ids_follow_the_canonical_rounding(tmp_path):
    img, c = stadium_scene()
    h = ((1.0, 0.0, 5.0), (0.0, 1.0, 7.0), (0.0, 0.0, 1.0))
    a = save_snapshot(tmp_path, "keypoints", c, img, score=0.91234567, homography=h,
                      autocalib=shipped_autocalib())
    stored = json.loads((tmp_path / "calibrations" / f"{a.id}.json").read_text())
    assert stored["snapshot"]["score"] == 0.91234567 and a.score == 0.91234567
    b = save_snapshot(tmp_path / "b", "keypoints", c, img, score=0.91234999, homography=h,
                      autocalib=shipped_autocalib())
    assert b.id == a.id  # the id rounds the score to 4 decimals, as for auto


@pytest.mark.parametrize("kind", ["auto", "keypoints"])
@pytest.mark.parametrize("bad", ["score", "homography"])
def test_non_finite_score_or_homography_is_refused(tmp_path, kind, bad):
    img, c = stadium_scene()
    h = [[1.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]]
    score = 0.9
    if bad == "score":
        score = float("nan")
    else:
        h[0][2] = float("inf")
    with pytest.raises(ValueError):
        save_snapshot(tmp_path, kind, c, img, score=score, homography=h,
                      autocalib=shipped_autocalib())
    # A stored file with the value broken by hand is refused on load as well.
    good = save_snapshot(tmp_path, kind, c, img, score=0.9, homography=[[1.0, 0, 5], [0, 1, 7],
                                                                        [0, 0, 1]],
                         autocalib=shipped_autocalib())
    path = tmp_path / "calibrations" / f"{good.id}.json"
    doc = json.loads(path.read_text())
    if bad == "score":
        doc["snapshot"]["score"] = float("nan")
    else:
        doc["snapshot"]["homography"][0][2] = float("inf")
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="malformed"):
        load_snapshot(tmp_path, good.id, autocalib=shipped_autocalib())


def test_keypoint_snapshot_requires_a_homography(tmp_path):
    img, c = stadium_scene()
    with pytest.raises(ValueError, match="homography"):
        save_snapshot(tmp_path, "keypoints", c, img, score=0.9, autocalib=shipped_autocalib())
    assert not (tmp_path / "calibrations").exists()


def test_keypoint_snapshot_can_be_vetoed_without_barring_templates(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    img, c = stadium_scene(shift=(45, 20))
    h = ((1.0, 0.0, 5.0), (0.0, 1.0, 7.0), (0.0, 0.0, 1.0))
    kp = save_snapshot(out, "keypoints", c, img, score=0.9, homography=h,
                       autocalib=shipped_autocalib())
    veto_calibration(out, kp.id, autocalib=shipped_autocalib(), video="clip.mp4")
    assert vetoed_ids(out) == {kp.id}
    # The template route is unaffected by a vetoed keypoint snapshot of the same video.
    assert select_snapshot(img, calib, out, th(), load_autocalib_config(), vetoed_ids(out),
                           video="clip.mp4").kind == "auto"


def test_load_unknown_snapshot_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path, "0123456789abcdef", autocalib=shipped_autocalib())


def test_preview_is_written_with_all_zones(tmp_path):
    img, c = stadium_scene()
    s = save_snapshot(tmp_path, "manual", c, img, score=0.95, autocalib=shipped_autocalib())
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
    assert load_snapshot(out, a.id, autocalib=shipped_autocalib()).id == a.id  # the old snapshot stays on disk, untouched


def test_removed_template_makes_its_auto_snapshot_ineligible(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame = stadium_scene(shift=(45, 20))[0]
    assert select_snapshot(frame, calib, out, th(), load_autocalib_config()).kind == "auto"
    (calib / "scene.json").unlink()
    assert select_snapshot(frame, calib, out, th(), load_autocalib_config()) is None


# ---- template set (ruling F-2) ---------------------------------------------------------


def _noisy_template_and_better_one(tmp_path):
    """calib/ with a noisy (worse-aligning) template A_old; returns (calib, out, frame, add_b)
    where add_b() adds template B_new, a better match for the frame (manual similarity < 0.8)."""
    from tests.helpers import add_manual_calibration

    calib, out = tmp_path / "calib", tmp_path / "out"
    img_a, cal_a = stadium_scene()
    noisy = np.clip(img_a + np.random.default_rng(1).normal(0, 45, img_a.shape), 0, 255)
    add_manual_calibration(calib, (noisy.astype(np.uint8), cal_a), name="A_old")
    frame = stadium_scene(shift=(45, 20))[0]
    return calib, out, frame, lambda: add_manual_calibration(
        calib, stadium_scene(shift=(37, 15)), name="B_new")


def test_auto_snapshot_records_the_sorted_template_set(tmp_path):
    calib, out, frame, add_b = _noisy_template_and_better_one(tmp_path)
    add_b()
    ids = sorted(select_snapshot(img, calib, tmp_path / "ids", th(), load_autocalib_config()).id
                 for img in (cv2.imread(str(calib / "A_old.png")),
                             cv2.imread(str(calib / "B_new.png"))))
    s = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert s.kind == "auto" and s.template_set == tuple(ids)
    doc = json.loads((out / "calibrations" / f"{s.id}.json").read_text())
    assert doc["snapshot"]["template_set"] == ids
    assert load_snapshot(out, s.id, autocalib=shipped_autocalib()).template_set == tuple(ids)


def test_template_set_is_part_of_an_auto_snapshot_identity(tmp_path):
    img, c = stadium_scene()
    a = save_snapshot(tmp_path, "auto", c, img, score=0.9, template_id="a" * 16,
                      template_set=["a" * 16], autocalib=shipped_autocalib())
    b = save_snapshot(tmp_path, "auto", c, img, score=0.9, template_id="a" * 16,
                      template_set=["b" * 16, "a" * 16], autocalib=shipped_autocalib())
    assert a.id != b.id and b.template_set == ("a" * 16, "b" * 16)


def test_added_better_template_replaces_an_existing_auto_snapshot(tmp_path):
    """Final review I2 probe: spec 6.1 (best over all templates) and 6.3b item 4."""
    calib, out, frame, add_b = _noisy_template_and_better_one(tmp_path)
    cfg = load_autocalib_config()
    first = select_snapshot(frame, calib, out, th(), cfg)
    assert first.kind == "auto" and first.calibration.name == "auto-A_old"
    add_b()
    b_id = select_snapshot(cv2.imread(str(calib / "B_new.png")), calib, tmp_path / "ids", th(),
                           cfg).id
    second = select_snapshot(frame, calib, out, th(), cfg)
    assert second.kind == "auto" and second.id != first.id
    assert second.template_id == b_id and second.calibration.name == "auto-B_new"
    assert second.score > first.score
    # unchanged template set: the same frame gets the same snapshot again
    assert select_snapshot(frame, calib, out, th(), cfg).id == second.id
    # a neighbouring video is aligned on its own frame, with the same better template
    neighbour = stadium_scene(shift=(46, 21))[0]
    own = select_snapshot(neighbour, calib, out, th(), cfg)
    assert own.id != second.id and own.template_id == b_id


def test_removing_an_unrelated_template_forces_a_fresh_alignment(tmp_path):
    from tests.helpers import add_manual_calibration

    calib, out = make_calib_dir(tmp_path, stadium_scene())
    add_manual_calibration(calib, stadium_scene(rotation_deg=30.0), name="other")
    frame = stadium_scene(shift=(45, 20))[0]
    a = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert a.kind == "auto" and a.calibration.name == "auto-scene"
    (calib / "other.json").unlink()
    b = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    assert b.kind == "auto" and b.id != a.id and b.template_id == a.template_id
    assert select_snapshot(frame, calib, out, th(), load_autocalib_config()).id == b.id


def test_adding_a_template_keeps_a_vetoed_group_barred_for_its_template(tmp_path):
    """Ruling T4-4 survives F-2: a template change elsewhere does not re-create a vetoed
    group from the same (unchanged) template under a new id."""
    from tests.helpers import add_manual_calibration

    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame = stadium_scene(shift=(45, 20))[0]
    a = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    veto_calibration(out, a.id, autocalib=shipped_autocalib(), video="clip.mp4")
    add_manual_calibration(calib, stadium_scene(rotation_deg=30.0), name="other")  # cannot align
    assert select_snapshot(frame, calib, out, th(), load_autocalib_config(),
                           vetoed_ids(out), video="clip.mp4") is None
    # Barring follows the explicit video scope, never a neighbour's (spec 6.3).
    other = stadium_scene(shift=(46, 21))[0]
    assert select_snapshot(other, calib, out, th(), load_autocalib_config(),
                           vetoed_ids(out), video="neighbour.mp4").kind == "auto"


# ---- vetoes (Task 4, ruling T4-1) ------------------------------------------------------


def test_veto_unknown_id_rejected(tmp_path):
    with pytest.raises(ValueError):
        veto_calibration(results_with_two_positions(tmp_path), "0" * 16, autocalib=shipped_autocalib())


def test_veto_file_records_time_and_note_and_unveto_restores(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    assert vetoed_ids(out) == set()
    veto_calibration(out, a.id, note="rim off by 10 px", autocalib=shipped_autocalib(),
                     video="clip.mp4")
    assert vetoed_ids(out) == {a.id}
    entry = json.loads((out / "calibration_vetoes.json").read_text())[a.id]
    assert entry["note"] == "rim off by 10 px" and entry["at"]
    assert entry["videos"] == ["clip.mp4"]
    unveto_calibration(out, a.id, autocalib=shipped_autocalib())
    assert vetoed_ids(out) == set()
    unveto_calibration(out, a.id, autocalib=shipped_autocalib())  # idempotent
    with pytest.raises(ValueError):
        unveto_calibration(out, "0" * 16, autocalib=shipped_autocalib())


def test_veto_applies_to_auto_snapshots_only(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    manual = select_snapshot(stadium_scene()[0], calib, out, th(), load_autocalib_config())
    assert manual.kind == "manual"
    with pytest.raises(ValueError, match="manual"):
        veto_calibration(out, manual.id, autocalib=shipped_autocalib())
    assert not (out / "calibration_vetoes.json").exists()


def test_veto_of_one_video_leaves_a_neighbouring_video_alone(tmp_path):
    # A veto is about one video's snapshot: another video at the same camera position keeps
    # the snapshot of its own frame (its predictions stay in the evaluation population).
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    veto_calibration(out, a.id, autocalib=shipped_autocalib(), video="clip.mp4")
    other = stadium_scene(shift=(46, 21))[0]  # same camera position, a different video frame
    alone = select_snapshot(other, calib, tmp_path / "alone", th(), load_autocalib_config())
    vetoed = select_snapshot(other, calib, out, th(), load_autocalib_config(), {a.id},
                            video="neighbour.mp4")
    assert alone.kind == "auto" and alone.id != a.id and vetoed.id == alone.id


def test_fresh_alignment_equal_to_a_vetoed_id_returns_none(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame = stadium_scene(shift=(45, 20))[0]
    a = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    veto_calibration(out, a.id, autocalib=shipped_autocalib(), video="clip.mp4")
    assert select_snapshot(frame, calib, tmp_path / "again", th(), load_autocalib_config()).id == a.id
    assert select_snapshot(frame, calib, out, th(), load_autocalib_config(), {a.id},
                           video="clip.mp4") is None


def test_veto_leaves_other_positions_and_manual_selection_alone(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    a = select_snapshot(stadium_scene(shift=(45, 20))[0], calib, out, th(), load_autocalib_config())
    veto_calibration(out, a.id, autocalib=shipped_autocalib(), video="clip.mp4")
    far = stadium_scene(shift=(-50, 25))[0]
    b = select_snapshot(far, calib, out, th(), load_autocalib_config())
    assert b.kind == "auto" and b.id != a.id
    assert select_snapshot(far, calib, out, th(), load_autocalib_config(), {a.id},
                           video="far.mp4").id == b.id
    manual = select_snapshot(stadium_scene()[0], calib, out, th(), load_autocalib_config(), {a.id},
                             video="clip.mp4")
    assert manual.kind == "manual"


# --------------------------------------------------------------------------- shape validation (codex port Task 1)


def _stored_manual_snapshot(tmp_path):
    calib, out = make_calib_dir(tmp_path, stadium_scene())
    frame, _ = stadium_scene()
    snap = select_snapshot(frame, calib, out, th(), load_autocalib_config())
    return out, snap, out / "calibrations" / f"{snap.id}.json"


def _edit_snapshot_json(path, edit):
    doc = json.loads(path.read_text(encoding="utf-8"))
    edit(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _break_polygon(doc):
    doc[ZONES[0][0]] = [[1]]


def _break_kind(doc):
    doc["snapshot"]["kind"] = "bogus"


def _drop_score(doc):
    del doc["snapshot"]["score"]


def _snapshot_is_list(doc):
    doc["snapshot"] = []


def _break_homography(doc):
    doc["snapshot"]["homography"] = [[1, 2, 3]]


def _break_template_set(doc):
    doc["snapshot"]["template_set"] = "abc"


@pytest.mark.parametrize("edit", [_break_polygon, _break_kind, _drop_score, _snapshot_is_list,
                                  _break_homography, _break_template_set])
def test_load_snapshot_rejects_a_malformed_file_with_valueerror(tmp_path, edit):
    out, snap, jpath = _stored_manual_snapshot(tmp_path)
    _edit_snapshot_json(jpath, edit)
    with pytest.raises(ValueError, match="malformed"):
        load_snapshot(out, snap.id, autocalib=shipped_autocalib())


def test_load_snapshot_rejects_a_top_level_list(tmp_path):
    out, snap, jpath = _stored_manual_snapshot(tmp_path)
    jpath.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_snapshot(out, snap.id, autocalib=shipped_autocalib())


def test_load_snapshot_rejects_a_missing_reference_image(tmp_path):
    out, snap, _ = _stored_manual_snapshot(tmp_path)
    (out / "calibrations" / f"{snap.id}.png").unlink()
    with pytest.raises(ValueError, match="malformed"):
        load_snapshot(out, snap.id, autocalib=shipped_autocalib())


def test_load_snapshot_still_reads_a_good_file(tmp_path):
    out, snap, _ = _stored_manual_snapshot(tmp_path)
    assert load_snapshot(out, snap.id, autocalib=shipped_autocalib()).id == snap.id


def test_validate_output_paths_rejects_overlap_and_files(tmp_path):
    from bxjudge.calibration_store import validate_output_paths

    calib = tmp_path / "calib"
    calib.mkdir()
    validate_output_paths(tmp_path / "out", calib)  # disjoint: fine
    for bad in (calib, calib / "sub", tmp_path, tmp_path / "calib" / ".." / "calib"):
        with pytest.raises(ValueError):
            validate_output_paths(bad, calib)
    afile = tmp_path / "file.txt"
    afile.write_text("x")
    with pytest.raises(ValueError, match="file"):
        validate_output_paths(afile, calib)


@pytest.mark.parametrize("bad", [
    [1e309, 720], [1280, float("nan")], [1e30, 720], [1280.5, 720], [0, 720], [-1280, 720],
    [1280, 1e12],
])
def test_load_snapshot_rejects_a_non_finite_huge_or_fractional_frame_size(tmp_path, bad):
    out, snap, jpath = _stored_manual_snapshot(tmp_path)
    _edit_snapshot_json(jpath, lambda doc: doc.__setitem__("frame_size", bad))
    with pytest.raises(ValueError, match="malformed"):
        load_snapshot(out, snap.id, autocalib=shipped_autocalib())


@pytest.mark.parametrize("bad", [1e309, float("nan"), -1e309, 1e30])
def test_load_snapshot_rejects_a_non_finite_or_huge_polygon_coordinate(tmp_path, bad):
    out, snap, jpath = _stored_manual_snapshot(tmp_path)
    _edit_snapshot_json(jpath, lambda doc: doc[ZONES[0][0]].append([bad, 1.0]))
    with pytest.raises(ValueError, match="malformed"):
        load_snapshot(out, snap.id, autocalib=shipped_autocalib())


# --------------------------------------------------------------------------- C-X2(f): bounds are policy (R40)


def test_snapshot_bounds_come_from_the_autocalib_config(tmp_path):
    import bxjudge.calibration_store as store

    assert not hasattr(store, "_MAX_FRAME_SIDE") and not hasattr(store, "_MAX_COORD")
    out, snap, _ = _stored_manual_snapshot(tmp_path)
    shipped = load_autocalib_config()
    assert load_snapshot(out, snap.id, autocalib=shipped).id == snap.id
    small_frame = replace(shipped, snapshot_max_frame_side=1000)  # 1280 px frame is over it
    with pytest.raises(ValueError, match="malformed"):
        load_snapshot(out, snap.id, autocalib=small_frame)
    small_coord = replace(shipped, snapshot_max_coord=10.0)  # zone points lie beyond 10 px
    with pytest.raises(ValueError, match="malformed"):
        load_snapshot(out, snap.id, autocalib=small_coord)
